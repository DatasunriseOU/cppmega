import pytest
import torch

from cppmega.megatron import fp8_activations as fp8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fp8._TE_AVAILABLE,
    reason="TE FP8 extraction requires CUDA and Transformer Engine",
)


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


def test_sparse_mla_te_zero_copy_requires_aggressive_backend(monkeypatch):
    from cppmega.megatron.sparse_mla_ops.sparse_mla import (
        _use_te_sparse_mla_fp8_zero_copy,
    )

    monkeypatch.delenv("CPPMEGA_SPARSE_MLA_FP8_QUANT", raising=False)
    assert not _use_te_sparse_mla_fp8_zero_copy()

    monkeypatch.setenv("CPPMEGA_SPARSE_MLA_FP8_QUANT", "te_tensorwise")
    assert _use_te_sparse_mla_fp8_zero_copy()


def test_sparse_mla_te_tensorwise_quant_backend(monkeypatch):
    from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_fwd_fp8 import (
        per_token_cast_to_fp8,
    )

    monkeypatch.setenv("CPPMEGA_SPARSE_MLA_FP8_QUANT", "te_tensorwise")
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    fp8_data, scale = per_token_cast_to_fp8(x)

    assert fp8_data.dtype == torch.float8_e4m3fn
    assert fp8_data.shape == x.shape
    assert scale.shape == x.shape[:-1]
    assert torch.all(scale == scale.reshape(-1)[0])
    out = fp8_data.to(torch.float32) * scale.unsqueeze(-1)
    assert torch.isfinite(out).all()


def test_sparse_mla_local_per_token_quant_backend_is_default(monkeypatch):
    from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_fwd_fp8 import (
        per_token_cast_to_fp8,
    )

    monkeypatch.delenv("CPPMEGA_SPARSE_MLA_FP8_QUANT", raising=False)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    fp8_data, scale = per_token_cast_to_fp8(x)

    assert fp8_data.dtype == torch.float8_e4m3fn
    assert scale.shape == x.shape[:-1]
    assert not torch.all(scale == scale.reshape(-1)[0])
