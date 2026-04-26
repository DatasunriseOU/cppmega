import sys
import types
import warnings

import pytest
import torch

from cppmega.megatron import fp8_activations as fp8


requires_te_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not fp8._TE_AVAILABLE,
    reason="TE FP8 extraction requires CUDA and Transformer Engine",
)


def _install_fake_te(monkeypatch, version: str):
    fake_te = types.ModuleType("transformer_engine")
    fake_te.__version__ = version
    fake_te.__path__ = []

    fake_pytorch = types.ModuleType("transformer_engine.pytorch")
    fake_pytorch.__path__ = []

    fake_tensor = types.ModuleType("transformer_engine.pytorch.tensor")

    class FakeQuantizedTensor:
        pass

    fake_tensor.QuantizedTensor = FakeQuantizedTensor
    fake_pytorch.tensor = fake_tensor
    fake_te.pytorch = fake_pytorch

    monkeypatch.setitem(sys.modules, "transformer_engine", fake_te)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", fake_pytorch)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor", fake_tensor)
    return FakeQuantizedTensor


def _fake_float8_tensor(fake_cls):
    packed = fake_cls()
    packed._data = torch.empty(4, dtype=torch.float8_e4m3fn)
    packed._scale_inv = torch.tensor(1.0)
    return packed


def test_extract_te_float8_private_attr_warns_once_for_untested_version(monkeypatch):
    from cppmega.megatron.sparse_mla_ops import sparse_mla

    fake_cls = _install_fake_te(monkeypatch, "9.9.9")
    packed = _fake_float8_tensor(fake_cls)
    monkeypatch.setattr(sparse_mla, "_te_private_attr_warning_emitted", False)

    with pytest.warns(
        RuntimeWarning,
        match="untested Transformer Engine version '9\\.9\\.9'",
    ):
        fp8_data, scale = sparse_mla._extract_fp8_data(packed)

    assert fp8_data is packed._data
    assert scale is packed._scale_inv

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fp8_data, scale = sparse_mla._extract_fp8_data(packed)

    assert fp8_data is packed._data
    assert scale is packed._scale_inv
    assert captured == []


def test_extract_te_float8_private_attr_allowlisted_version_is_quiet(monkeypatch):
    from cppmega.megatron.sparse_mla_ops import sparse_mla

    fake_cls = _install_fake_te(monkeypatch, "2.14.0+local")
    packed = _fake_float8_tensor(fake_cls)
    monkeypatch.setattr(sparse_mla, "_te_private_attr_warning_emitted", False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fp8_data, scale = sparse_mla._extract_fp8_data(packed)

    assert fp8_data is packed._data
    assert scale is packed._scale_inv
    assert captured == []


@requires_te_cuda
def test_extract_te_float8_tensor_uint8_storage_zero_copy():
    import transformer_engine  # noqa: F401
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor import Float8CurrentScalingQuantizer

    from cppmega.megatron.sparse_mla_ops.sparse_mla import _extract_fp8_data

    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    quantizer = Float8CurrentScalingQuantizer(
        tex.DType.kFloat8E4M3,
        device=torch.device("cuda"),
        rowwise=True,
        columnwise=False,
    )
    packed = tex.quantize(x, quantizer)

    fp8_data, scale = _extract_fp8_data(packed)
    assert fp8_data is not None
    assert scale is not None
    assert fp8_data.dtype == torch.float8_e4m3fn
    assert fp8_data.data_ptr() == packed._data.data_ptr()

    ref = tex.dequantize(packed, tex.DType.kBFloat16)
    out = (fp8_data.to(torch.float32) * scale).to(torch.bfloat16)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_sparse_mla_te_zero_copy_is_forced(monkeypatch):
    from cppmega.megatron.sparse_mla_ops.sparse_mla import (
        _sparse_mla_fp8_quant_backend,
        _use_te_sparse_mla_fp8_zero_copy,
    )

    monkeypatch.delenv("CPPMEGA_SPARSE_MLA_FP8_QUANT", raising=False)
    assert _sparse_mla_fp8_quant_backend() == "te_tensorwise"
    assert _use_te_sparse_mla_fp8_zero_copy()

    monkeypatch.setenv("CPPMEGA_SPARSE_MLA_FP8_QUANT", "local_per_token")
    with pytest.raises(RuntimeError, match="hard-wired to TE"):
        _use_te_sparse_mla_fp8_zero_copy()


@requires_te_cuda
def test_sparse_mla_te_tensorwise_quant_backend(monkeypatch):
    from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_fwd_fp8 import (
        per_token_cast_to_fp8,
    )

    monkeypatch.delenv("CPPMEGA_SPARSE_MLA_FP8_QUANT", raising=False)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    fp8_data, scale = per_token_cast_to_fp8(x)

    assert fp8_data.dtype == torch.float8_e4m3fn
    assert fp8_data.shape == x.shape
    assert scale.shape == x.shape[:-1]
    assert torch.all(scale == scale.reshape(-1)[0])
    out = fp8_data.to(torch.float32) * scale.unsqueeze(-1)
    assert torch.isfinite(out).all()


@requires_te_cuda
def test_sparse_mla_local_per_token_quant_backend_is_rejected(monkeypatch):
    from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_fwd_fp8 import (
        per_token_cast_to_fp8,
    )

    monkeypatch.setenv("CPPMEGA_SPARSE_MLA_FP8_QUANT", "local_per_token")
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="hard-wired to TE"):
        per_token_cast_to_fp8(x)
