from __future__ import annotations

import atexit
import importlib
import sys
from types import ModuleType

import pytest
import torch


class _FakeMXFP8Tensor:
    def __init__(self, name: str):
        self.name = name
        self._rowwise_data = None
        self._rowwise_scale_inv = None
        self._columnwise_data = None
        self._columnwise_scale_inv = None
        self._with_gemm_swizzled_scales = False


class _FakeMXFP8Quantizer:
    def quantize(self, tensor, *args, **kwargs):
        return _FakeMXFP8Tensor("quantized")

    def update_quantized(self, src, dst, *args, **kwargs):
        return dst


def _fresh_emit_shim(monkeypatch, quantizer_cls=_FakeMXFP8Quantizer):
    for key in (
        "CPPMEGA_TE_MXFP8_DGRAD_BF16",
        "CPPMEGA_TE_MXFP8_WGRAD_BF16",
        "CPPMEGA_TE_MXFP8_BWD_BACKEND",
        "CPPMEGA_TE_MXFP8_COMPACT_COLUMNWISE_BACKWARD",
        "CPPMEGA_TE_MXFP8_DENSE_SAVED_OPERANDS",
        "CPPMEGA_TE_MXFP8_LINEAR_KERNEL_CONTRACT",
        "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_SWIZZLED",
        "CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND",
        "NVTE_BACKWARD_OVERRIDE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER", "1")
    monkeypatch.setenv("CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND", "te")
    monkeypatch.setenv("CPPMEGA_TE_VERSION_STRICT", "0")
    monkeypatch.setattr(atexit, "register", lambda func, *args, **kwargs: func)
    monkeypatch.delitem(sys.modules, "scripts.cppmega_fp8_shim", raising=False)

    class _FakeTransformerEngineBaseModule:
        @staticmethod
        def grad_output_preprocess(ctx, grad_output, row_parallel_mode, *args, **kwargs):
            return grad_output

    class _FakeFP8GlobalStateManager:
        @staticmethod
        def get_fp8_recipe():
            return None

    fake_te = ModuleType("transformer_engine")
    fake_te.__version__ = "2.16.0.dev0"
    fake_te.__path__ = []
    fake_common = ModuleType("transformer_engine.common")
    fake_recipe = ModuleType("transformer_engine.common.recipe")
    fake_recipe.MXFP8BlockScaling = type("MXFP8BlockScaling", (), {})
    fake_recipe.NVFP4BlockScaling = type("NVFP4BlockScaling", (), {})
    fake_pytorch = ModuleType("transformer_engine.pytorch")
    fake_tensor = ModuleType("transformer_engine.pytorch.tensor")
    fake_tensor.MXFP8Quantizer = quantizer_cls
    fake_mxfp8_tensor = ModuleType("transformer_engine.pytorch.tensor.mxfp8_tensor")
    fake_mxfp8_tensor.MXFP8Tensor = _FakeMXFP8Tensor
    fake_module = ModuleType("transformer_engine.pytorch.module")
    fake_linear = ModuleType("transformer_engine.pytorch.module.linear")
    fake_base = ModuleType("transformer_engine.pytorch.module.base")
    fake_base.TransformerEngineBaseModule = _FakeTransformerEngineBaseModule
    fake_quantization = ModuleType("transformer_engine.pytorch.quantization")
    fake_quantization.FP8GlobalStateManager = _FakeFP8GlobalStateManager
    fake_tex = ModuleType("transformer_engine_torch")
    fake_tex.split_quantize = lambda *args, **kwargs: None
    fake_dsa_parent = ModuleType(
        "megatron.core.transformer.experimental_attention_variant"
    )
    fake_dsa = ModuleType("megatron.core.transformer.experimental_attention_variant.dsa")
    fake_dsa.unfused_dsa_fn = lambda *args, **kwargs: None
    fake_sparse_mla = ModuleType("cppmega.megatron.sparse_mla_ops.sparse_mla")
    fake_sparse_mla.SparseMLA = type("SparseMLA", (), {})
    fake_sparse_mla.SparseMLA_FP8 = type("SparseMLA_FP8", (), {})
    fake_sparse_mla.sparse_mla_as_unfused_dsa = lambda *args, **kwargs: None
    fake_sparse_mla.sparse_mla_fp8_as_unfused_dsa = lambda *args, **kwargs: None

    fake_common.recipe = fake_recipe
    fake_pytorch.tensor = fake_tensor
    fake_pytorch.module = fake_module
    fake_te.common = fake_common
    fake_te.pytorch = fake_pytorch
    fake_module.linear = fake_linear
    fake_module.base = fake_base
    fake_dsa_parent.dsa = fake_dsa

    for name, module in {
        "transformer_engine": fake_te,
        "transformer_engine.common": fake_common,
        "transformer_engine.common.recipe": fake_recipe,
        "transformer_engine.pytorch": fake_pytorch,
        "transformer_engine.pytorch.tensor": fake_tensor,
        "transformer_engine.pytorch.tensor.mxfp8_tensor": fake_mxfp8_tensor,
        "transformer_engine.pytorch.module": fake_module,
        "transformer_engine.pytorch.module.linear": fake_linear,
        "transformer_engine.pytorch.module.base": fake_base,
        "transformer_engine.pytorch.quantization": fake_quantization,
        "transformer_engine_torch": fake_tex,
        "megatron.core.transformer.experimental_attention_variant": fake_dsa_parent,
        "megatron.core.transformer.experimental_attention_variant.dsa": fake_dsa,
        "cppmega.megatron.sparse_mla_ops.sparse_mla": fake_sparse_mla,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    try:
        shim = importlib.import_module("scripts.cppmega_fp8_shim")
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - host dependency guard
        pytest.skip(
            "cppmega_fp8_shim MXFP8 path unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
    for attr in (
        "_cppmega_attach_mxfp8_rowwise_transpose",
        "_cppmega_mxfp8_rowwise_2d_shape",
        "_cppmega_flattened_lastdim_shape",
    ):
        if not hasattr(shim, attr):
            pytest.skip(f"cppmega_fp8_shim MXFP8 helper {attr} was not installed")
    return shim


def test_transpose_emit_flattens_3d_source_before_quantize(monkeypatch):
    emit_calls = []

    class RecordingQuantizer(_FakeMXFP8Quantizer):
        def quantize_rowwise_transpose(self, source, columnwise_scale, **kwargs):
            emit_calls.append((source, columnwise_scale, kwargs))
            sidecar = _FakeMXFP8Tensor("sidecar")
            sidecar._with_gemm_swizzled_scales = True
            return sidecar

    shim = _fresh_emit_shim(monkeypatch, RecordingQuantizer)
    out = _FakeMXFP8Tensor("out")
    out._columnwise_scale_inv = torch.empty((4, 16), dtype=torch.uint8)
    source = torch.empty((8, 2, 16), dtype=torch.bfloat16)

    result = shim._cppmega_attach_mxfp8_rowwise_transpose(
        out, RecordingQuantizer(), source
    )

    assert result is out
    assert len(emit_calls) == 1
    emit_source, _scale, _kwargs = emit_calls[0]
    # Regression: TE transpose-emit must receive the flattened (S*B, H) 2D
    # source, not the original [sequence, batch, hidden] 3D activation.
    assert emit_source.dim() == 2
    assert tuple(emit_source.shape) == (16, 16)
    stats = shim.cppmega_te_mxfp8_bwd_stats
    assert stats["mxfp8_tn_adapter_te_emit_failed"] == 0
    sidecar_attr = shim._cppmega_mxfp8_tn_sidecar_attr
    assert getattr(out, sidecar_attr, None) is not None


def test_mxfp8_rowwise_2d_shape_flattens_3d_payload(monkeypatch):
    shim = _fresh_emit_shim(monkeypatch)

    payload = _FakeMXFP8Tensor("payload")
    payload._rowwise_data = torch.empty((8, 2, 16))
    assert shim._cppmega_mxfp8_rowwise_2d_shape(payload) == (16, 16)

    payload._rowwise_data = torch.empty((4, 16))
    assert shim._cppmega_mxfp8_rowwise_2d_shape(payload) == (4, 16)


def test_flattened_lastdim_shape_crash_shapes(monkeypatch):
    shim = _fresh_emit_shim(monkeypatch)

    # Shapes from the dense MoE crash: TE hands [sequence, batch, hidden]
    # while the flattened grad output is [sequence*batch, hidden].
    crash_source = torch.empty((4096, 4, 3584), device="meta")
    assert shim._cppmega_flattened_lastdim_shape(crash_source) == (16384, 3584)

    flattened = torch.empty((16384, 3584), device="meta")
    assert shim._cppmega_flattened_lastdim_shape(flattened) == (16384, 3584)

    assert shim._cppmega_flattened_lastdim_shape(None) is None
