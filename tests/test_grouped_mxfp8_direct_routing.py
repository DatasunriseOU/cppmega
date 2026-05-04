from __future__ import annotations

import atexit
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


class _FakeMXFP8Tensor:
    def __init__(self, name: str):
        self.name = name
        self._rowwise_data = object()
        self._rowwise_scale_inv = object()
        self._columnwise_data = object()
        self._columnwise_scale_inv = object()
        self._with_gemm_swizzled_scales = False


class _FakeMXFP8Quantizer:
    optimize_for_gemm = True

    def quantize(self, _tensor, *args, **kwargs):
        assert args == ()
        assert kwargs == {}
        return _FakeMXFP8Tensor("quantized")

    def update_quantized(self, _src, dst, *args, **kwargs):
        assert args == ()
        assert kwargs == {}
        return dst

    def quantize_rowwise_transpose(self, _tensor, _scale, *args, **kwargs):
        assert args == ()
        out = _FakeMXFP8Tensor("rowwise-transpose")
        out._with_gemm_swizzled_scales = bool(
            kwargs.get("with_gemm_swizzled_scales", False)
        )
        return out


def _install_fake_te_stack(monkeypatch):
    te_root = ModuleType("transformer_engine")
    te_root.__version__ = "2.16.0.dev0"
    te_common = ModuleType("transformer_engine.common")
    te_recipe = ModuleType("transformer_engine.common.recipe")
    te_recipe.MXFP8BlockScaling = type("MXFP8BlockScaling", (), {})
    te_recipe.NVFP4BlockScaling = type("NVFP4BlockScaling", (), {})
    te_pytorch = ModuleType("transformer_engine.pytorch")
    te_tensor = ModuleType("transformer_engine.pytorch.tensor")
    te_tensor.MXFP8Quantizer = _FakeMXFP8Quantizer
    te_mxfp8_tensor = ModuleType("transformer_engine.pytorch.tensor.mxfp8_tensor")
    te_mxfp8_tensor.MXFP8Tensor = _FakeMXFP8Tensor
    te_module = ModuleType("transformer_engine.pytorch.module")
    te_base = ModuleType("transformer_engine.pytorch.module.base")
    te_linear = ModuleType("transformer_engine.pytorch.module.linear")
    te_quantization = ModuleType("transformer_engine.pytorch.quantization")
    te_torch = ModuleType("transformer_engine_torch")

    class _FakeBaseModule:
        @staticmethod
        def grad_output_preprocess(*_args, **_kwargs):
            return None, None

    class _FakeFP8GlobalStateManager:
        @staticmethod
        def get_fp8_recipe():
            return te_recipe.MXFP8BlockScaling()

    te_base.TransformerEngineBaseModule = _FakeBaseModule
    te_quantization.FP8GlobalStateManager = _FakeFP8GlobalStateManager
    te_quantization.check_mxfp8_support = lambda *_args, **_kwargs: None

    def split_quantize(_tensor, split_sections, _quantizers, *args, **kwargs):
        assert args == ()
        assert kwargs == {}
        return [_FakeMXFP8Tensor(f"single-{idx}") for idx, _ in enumerate(split_sections)]

    te_torch.split_quantize = split_quantize
    monkeypatch.setitem(sys.modules, "transformer_engine", te_root)
    monkeypatch.setitem(sys.modules, "transformer_engine.common", te_common)
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", te_recipe)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te_pytorch)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor", te_tensor)
    monkeypatch.setitem(
        sys.modules,
        "transformer_engine.pytorch.tensor.mxfp8_tensor",
        te_mxfp8_tensor,
    )
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.module", te_module)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.module.base", te_base)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.module.linear", te_linear)
    monkeypatch.setitem(
        sys.modules,
        "transformer_engine.pytorch.quantization",
        te_quantization,
    )
    monkeypatch.setitem(sys.modules, "transformer_engine_torch", te_torch)

    dsa_mod = ModuleType("megatron.core.transformer.experimental_attention_variant.dsa")
    dsa_mod.unfused_dsa_fn = lambda *args, **kwargs: None
    sparse_mla_mod = ModuleType("cppmega.megatron.sparse_mla_ops.sparse_mla")
    sparse_mla_mod.SparseMLA = type("SparseMLA", (), {})
    sparse_mla_mod.SparseMLA_FP8 = type("SparseMLA_FP8", (), {})
    sparse_mla_mod.sparse_mla_as_unfused_dsa = lambda *args, **kwargs: None
    sparse_mla_mod.sparse_mla_fp8_as_unfused_dsa = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.transformer.experimental_attention_variant.dsa",
        dsa_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "cppmega.megatron.sparse_mla_ops.sparse_mla",
        sparse_mla_mod,
    )
    return torch, te_torch


def _fresh_grouped_shim(
    monkeypatch,
    *,
    grouped_direct: bool = False,
    grouped_quantize_producer: str = "single_output",
):
    _install_fake_te_stack(monkeypatch)
    for key in (
        "CPPMEGA_TE_MXFP8_DGRAD_BF16",
        "CPPMEGA_TE_MXFP8_WGRAD_BF16",
        "NVTE_BACKWARD_OVERRIDE",
        "CPPMEGA_TE_MXFP8_GROUPED_DIRECT_BACKWARD",
        "CPPMEGA_TE_MXFP8_GROUPED_GEMM_READY_BACKWARD",
        "CPPMEGA_TE_MXFP8_GROUPED_QUANTIZE_PRODUCER",
        "CPPMEGA_DSA_SPARSE_MODE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER", "1")
    monkeypatch.setenv(
        "CPPMEGA_TE_MXFP8_GROUPED_DIRECT_BACKWARD",
        "1" if grouped_direct else "0",
    )
    monkeypatch.setenv(
        "CPPMEGA_TE_MXFP8_GROUPED_QUANTIZE_PRODUCER",
        grouped_quantize_producer,
    )
    monkeypatch.setenv(
        "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND",
        "te" if grouped_quantize_producer == "multi_output" else "off",
    )
    monkeypatch.setenv("CPPMEGA_TE_VERSION_STRICT", "0")
    monkeypatch.setenv("CPPMEGA_DSA_SPARSE_MODE", "gather_scatter")
    monkeypatch.setenv(
        "CPPMEGA_I_UNDERSTAND_DSA_GATHER_SCATTER_IS_DEPRECATED_AND_SLOW",
        "1",
    )
    monkeypatch.setattr(atexit, "register", lambda func, *args, **kwargs: func)
    monkeypatch.delitem(sys.modules, "scripts.cppmega_fp8_shim", raising=False)

    try:
        shim = importlib.import_module("scripts.cppmega_fp8_shim")
    except Exception as exc:  # pragma: no cover - host dependency guard
        pytest.skip(
            "cppmega_fp8_shim MXFP8 path unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
    if not hasattr(shim, "_cppmega_wrap_general_grouped_gemm"):
        pytest.skip("cppmega_fp8_shim MXFP8 grouped wrapper was not installed")
    return shim


def _reset_bwd_stats(shim) -> None:
    for key, value in shim.cppmega_te_mxfp8_bwd_stats.items():
        if isinstance(value, dict):
            value.clear()
        else:
            shim.cppmega_te_mxfp8_bwd_stats[key] = 0


def _wrap_fake_grouped_module(shim):
    orig_calls = []

    def general_grouped_gemm(A, B, out, *args, **kwargs):
        orig_calls.append((A, B, out, args, kwargs))
        return "fallback-result"

    module = SimpleNamespace(general_grouped_gemm=general_grouped_gemm)
    assert shim._cppmega_wrap_general_grouped_gemm(module)
    return module, orig_calls


def _mark_rowwise_transpose(tensor):
    tensor._te_rowwise_transpose_for_backward_operand = True
    return tensor


@pytest.mark.parametrize(
    ("layout", "op_kind", "mark_a", "mark_b"),
    [
        ("NN", "dgrad", True, False),
        ("NT", "wgrad", True, True),
    ],
)
def test_grouped_mxfp8_gemm_ready_route_uses_existing_operands_without_copy(
    monkeypatch,
    layout,
    op_kind,
    mark_a,
    mark_b,
):
    shim = _fresh_grouped_shim(monkeypatch, grouped_direct=False)
    _reset_bwd_stats(shim)

    def fail_copy_bridge(*_args, **_kwargs):
        raise AssertionError("GEMM-ready grouped path used the transpose-copy bridge")

    monkeypatch.setattr(
        shim,
        "_cppmega_mxfp8_colwise_as_rowwise_transpose",
        fail_copy_bridge,
    )

    module, orig_calls = _wrap_fake_grouped_module(shim)
    A = [_FakeMXFP8Tensor(f"{layout}-A0"), _FakeMXFP8Tensor(f"{layout}-A1")]
    B = [_FakeMXFP8Tensor(f"{layout}-B0"), _FakeMXFP8Tensor(f"{layout}-B1")]
    if mark_a:
        A = [_mark_rowwise_transpose(item) for item in A]
    if mark_b:
        B = [_mark_rowwise_transpose(item) for item in B]

    result = module.general_grouped_gemm(
        A,
        B,
        object(),
        "splits",
        layout=layout,
        grad=True,
    )

    assert result == "fallback-result"
    assert len(orig_calls) == 1
    call_A, call_B, _out, call_args, call_kwargs = orig_calls[0]
    assert call_A == A
    assert call_B == B
    assert call_args == ("splits",)
    assert call_kwargs["layout"] == "TN"
    assert call_kwargs["use_split_accumulator"] is False

    stats = shim.cppmega_te_mxfp8_bwd_stats
    assert stats[f"mxfp8_grouped_gemm_ready_{op_kind}"] == 1
    assert stats[f"mxfp8_grouped_gemm_ready_miss_{op_kind}"] == 0
    assert stats[f"mxfp8_grouped_transpose_copy_fallback_{op_kind}"] == 0
    assert stats["mxfp8_tn_adapter_copy_transpose"] == 0


def test_grouped_mxfp8_direct_hits_bypass_transpose_and_sidecars(monkeypatch):
    shim = _fresh_grouped_shim(monkeypatch, grouped_direct=True)

    def fail_sidecar_path(*_args, **_kwargs):
        raise AssertionError("direct grouped MXFP8 path touched transpose sidecars")

    monkeypatch.setattr(
        shim,
        "_cppmega_mxfp8_colwise_as_rowwise_transpose",
        fail_sidecar_path,
    )
    monkeypatch.setattr(shim, "_cppmega_get_mxfp8_sidecar", fail_sidecar_path)
    monkeypatch.setattr(shim, "_cppmega_get_mxfp8_sidecar_entry", fail_sidecar_path)

    for layout, op_kind in (("NN", "dgrad"), ("NT", "wgrad")):
        _reset_bwd_stats(shim)
        expected = object()
        backend_calls = []

        def try_grouped_direct(A, B, out, *args, **kwargs):
            backend_calls.append((A, B, out, args, kwargs))
            for item in (*A, *B):
                assert not getattr(
                    item,
                    "_cppmega_mxfp8_rowwise_transpose_operand",
                    False,
                )
                assert not getattr(
                    item,
                    "_te_rowwise_transpose_for_backward_operand",
                    False,
                )
            return True, expected

        fake_backend = ModuleType("cppmega.megatron.grouped_mxfp8_gemm")
        fake_backend.try_grouped_direct = try_grouped_direct
        monkeypatch.setattr(shim, "_cppmega_grouped_mxfp8_module", [fake_backend])

        module, orig_calls = _wrap_fake_grouped_module(shim)
        A = [_FakeMXFP8Tensor(f"{layout}-A0"), _FakeMXFP8Tensor(f"{layout}-A1")]
        B = [_FakeMXFP8Tensor(f"{layout}-B0"), _FakeMXFP8Tensor(f"{layout}-B1")]
        out = object()

        result = module.general_grouped_gemm(
            A,
            B,
            out,
            "splits",
            layout=layout,
            grad=True,
        )

        assert result is expected
        assert orig_calls == []
        assert len(backend_calls) == 1
        call_A, call_B, call_out, call_args, call_kwargs = backend_calls[0]
        assert call_A is A
        assert call_B is B
        assert call_out is out
        assert call_args == ("splits",)
        assert call_kwargs["layout"] == layout
        assert call_kwargs["grad"] is True

        stats = shim.cppmega_te_mxfp8_bwd_stats
        assert stats[f"mxfp8_grouped_direct_{op_kind}"] == 1
        assert stats[f"mxfp8_grouped_direct_miss_{op_kind}"] == 0
        assert stats[f"mxfp8_grouped_transpose_copy_fallback_{op_kind}"] == 0
        assert stats[f"mxfp8_tn_adapter_{op_kind}"] == 0
        assert stats["mxfp8_tn_adapter_copy_transpose"] == 0
        assert stats["mxfp8_tn_sidecar_consumed"] == 0


def test_grouped_mxfp8_direct_missing_backend_api_counts_explicit_fallback(monkeypatch):
    shim = _fresh_grouped_shim(monkeypatch, grouped_direct=True)
    _reset_bwd_stats(shim)

    fake_backend = ModuleType("cppmega.megatron.grouped_mxfp8_gemm")
    monkeypatch.setattr(shim, "_cppmega_grouped_mxfp8_module", [fake_backend])

    transpose_calls = []

    def fake_transpose(tensor):
        transpose_calls.append(tensor)
        return ("transpose", tensor)

    monkeypatch.setattr(
        shim,
        "_cppmega_mxfp8_colwise_as_rowwise_transpose",
        fake_transpose,
    )

    module, orig_calls = _wrap_fake_grouped_module(shim)
    A = [_FakeMXFP8Tensor("A0"), _FakeMXFP8Tensor("A1")]
    B = [_FakeMXFP8Tensor("B0"), _FakeMXFP8Tensor("B1")]

    result = module.general_grouped_gemm(A, B, object(), layout="NN", grad=True)

    assert result == "fallback-result"
    assert transpose_calls == A
    assert len(orig_calls) == 1
    converted_A, fallback_B, _out, _args, fallback_kwargs = orig_calls[0]
    assert converted_A == [("transpose", A[0]), ("transpose", A[1])]
    assert fallback_B is B
    assert fallback_kwargs["layout"] == "TN"
    assert fallback_kwargs["use_split_accumulator"] is False

    stats = shim.cppmega_te_mxfp8_bwd_stats
    assert stats["mxfp8_grouped_direct_dgrad"] == 0
    assert stats["mxfp8_grouped_direct_miss_dgrad"] == 1
    assert stats["mxfp8_grouped_transpose_copy_fallback_dgrad"] == 1
    assert stats["mxfp8_tn_adapter_dgrad"] == 1
    assert any(
        "grouped MXFP8 backend exposes neither try_grouped_direct" in reason
        for reason in stats["fallback_reasons"]
    )


def test_grouped_mxfp8_multi_output_producer_consumes_gemm_ready_operands(monkeypatch):
    shim = _fresh_grouped_shim(
        monkeypatch,
        grouped_quantize_producer="multi_output",
    )
    _reset_bwd_stats(shim)
    tex = sys.modules["transformer_engine_torch"]
    tensor = torch.empty((5, 4), dtype=torch.bfloat16)
    quantizers = [_FakeMXFP8Quantizer(), _FakeMXFP8Quantizer()]
    split_sections = [2, 3]
    calls = []

    def multi_output(
        call_tensor,
        call_split_sections,
        call_quantizers,
        disable_bulk_allocation,
        transpose_scales_with_gemm_swizzled,
    ):
        calls.append(
            (
                call_tensor,
                call_split_sections,
                call_quantizers,
                disable_bulk_allocation,
                transpose_scales_with_gemm_swizzled,
            )
        )
        outputs = [_FakeMXFP8Tensor("out0"), _FakeMXFP8Tensor("out1")]
        for idx, out in enumerate(outputs):
            out._te_gemm_ready_rowwise_transpose_for_backward = _FakeMXFP8Tensor(
                f"rowwise-t-{idx}"
            )
        return outputs

    tex.mxfp8_split_quantize_with_rowwise_transpose = multi_output
    result = tex.split_quantize(
        tensor,
        split_sections,
        quantizers,
        disable_bulk_allocation=True,
        transpose_scales_with_gemm_swizzled=False,
    )

    assert len(result) == 2
    assert calls == [(tensor, split_sections, quantizers, True, False)]
    for out in result:
        operand = out._te_gemm_ready_rowwise_transpose_for_backward
        assert getattr(operand, "_te_rowwise_transpose_for_backward_operand") is True
        assert getattr(out, "_te_rowwise_transpose_for_backward") is operand

    stats = shim.cppmega_te_mxfp8_bwd_stats
    assert stats["mxfp8_grouped_quantize_producer_multi_output"] == 1
    assert stats["mxfp8_grouped_quantize_producer_multi_output_consumed"] == 2
    assert stats["mxfp8_grouped_quantize_producer_single_output"] == 0
    assert stats["mxfp8_grouped_quantize_producer_multi_output_missing_api"] == 0


def test_grouped_mxfp8_multi_output_missing_api_fails_fast(monkeypatch):
    shim = _fresh_grouped_shim(
        monkeypatch,
        grouped_quantize_producer="multi_output",
    )
    _reset_bwd_stats(shim)
    tex = sys.modules["transformer_engine_torch"]

    with pytest.raises(RuntimeError, match="mxfp8_split_quantize_with_rowwise_transpose"):
        tex.split_quantize(
            torch.empty((5, 4), dtype=torch.bfloat16),
            [2, 3],
            [_FakeMXFP8Quantizer(), _FakeMXFP8Quantizer()],
        )

    stats = shim.cppmega_te_mxfp8_bwd_stats
    assert stats["mxfp8_grouped_quantize_producer_multi_output_missing_api"] == 1
    assert stats["mxfp8_grouped_quantize_producer_multi_output"] == 0
    assert stats["mxfp8_grouped_quantize_producer_single_output"] == 0


def test_grouped_mxfp8_multi_output_rejects_non_grouped_call_shape(monkeypatch):
    shim = _fresh_grouped_shim(
        monkeypatch,
        grouped_quantize_producer="multi_output",
    )
    _reset_bwd_stats(shim)
    tex = sys.modules["transformer_engine_torch"]
    tex.mxfp8_split_quantize_with_rowwise_transpose = lambda *_args, **_kwargs: []

    with pytest.raises(RuntimeError, match="refusing to fall back to single_output"):
        tex.split_quantize(
            torch.empty((5, 4), dtype=torch.bfloat16),
            5,
            _FakeMXFP8Quantizer(),
        )

    stats = shim.cppmega_te_mxfp8_bwd_stats
    assert stats["mxfp8_grouped_quantize_producer_multi_output"] == 0
    assert stats["mxfp8_grouped_quantize_producer_single_output"] == 0
