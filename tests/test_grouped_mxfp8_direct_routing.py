from __future__ import annotations

import atexit
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


class _FakeMXFP8Tensor:
    def __init__(self, name: str):
        self.name = name
        self._rowwise_data = object()
        self._rowwise_scale_inv = object()
        self._columnwise_data = object()
        self._columnwise_scale_inv = object()
        self._with_gemm_swizzled_scales = False


def _fresh_grouped_shim(
    monkeypatch,
    *,
    grouped_direct: bool = False,
    grouped_quantize_producer: str = "single_output",
    tex_split_quantize=None,
    tex_multi_output_quantize=None,
):
    for key in (
        "CPPMEGA_TE_MXFP8_DGRAD_BF16",
        "CPPMEGA_TE_MXFP8_WGRAD_BF16",
        "NVTE_BACKWARD_OVERRIDE",
        "CPPMEGA_TE_MXFP8_GROUPED_DIRECT_BACKWARD",
        "CPPMEGA_TE_MXFP8_GROUPED_GEMM_READY_BACKWARD",
        "CPPMEGA_TE_MXFP8_GROUPED_QUANTIZE_PRODUCER",
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
    monkeypatch.setattr(atexit, "register", lambda func, *args, **kwargs: func)
    monkeypatch.delitem(sys.modules, "scripts.cppmega_fp8_shim", raising=False)

    class FakeMXFP8Quantizer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def quantize(self, tensor, *args, **kwargs):
            return _FakeMXFP8Tensor("quantized")

        def update_quantized(self, src, dst, *args, **kwargs):
            return dst

    fake_te = ModuleType("transformer_engine")
    fake_te.__version__ = "2.16.0.dev0"
    fake_te.__path__ = []
    fake_common = ModuleType("transformer_engine.common")
    fake_recipe = ModuleType("transformer_engine.common.recipe")
    fake_recipe.MXFP8BlockScaling = type("MXFP8BlockScaling", (), {})
    fake_recipe.NVFP4BlockScaling = type("NVFP4BlockScaling", (), {})
    fake_pytorch = ModuleType("transformer_engine.pytorch")
    fake_tensor = ModuleType("transformer_engine.pytorch.tensor")
    fake_tensor.MXFP8Quantizer = FakeMXFP8Quantizer
    fake_mxfp8_tensor = ModuleType("transformer_engine.pytorch.tensor.mxfp8_tensor")
    fake_mxfp8_tensor.MXFP8Tensor = _FakeMXFP8Tensor
    fake_module = ModuleType("transformer_engine.pytorch.module")
    fake_linear = ModuleType("transformer_engine.pytorch.module.linear")
    fake_quantization = ModuleType("transformer_engine.pytorch.quantization")
    fake_quantization.FP8GlobalStateManager = type("FP8GlobalStateManager", (), {})
    fake_tex = ModuleType("transformer_engine_torch")
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

    if tex_split_quantize is None:

        def tex_split_quantize(tensor, split_sections, quantizers, *args, **kwargs):
            return [_FakeMXFP8Tensor(f"split{i}") for i, _ in enumerate(split_sections)]

    fake_tex.split_quantize = tex_split_quantize
    if tex_multi_output_quantize is not None:
        fake_tex.mxfp8_split_quantize_with_rowwise_transpose = tex_multi_output_quantize

    fake_common.recipe = fake_recipe
    fake_pytorch.tensor = fake_tensor
    fake_pytorch.module = fake_module
    fake_te.common = fake_common
    fake_te.pytorch = fake_pytorch
    fake_module.linear = fake_linear
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
        "transformer_engine.pytorch.quantization": fake_quantization,
        "transformer_engine_torch": fake_tex,
        "megatron.core.transformer.experimental_attention_variant": fake_dsa_parent,
        "megatron.core.transformer.experimental_attention_variant.dsa": fake_dsa,
        "cppmega.megatron.sparse_mla_ops.sparse_mla": fake_sparse_mla,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

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


def test_grouped_mxfp8_multi_output_producer_sets_consumed_operands(monkeypatch):
    class FakeQuantizer:
        pass

    quantizers = [FakeQuantizer(), FakeQuantizer()]
    outputs = [_FakeMXFP8Tensor("out0"), _FakeMXFP8Tensor("out1")]
    operands = [_FakeMXFP8Tensor("out0.T"), _FakeMXFP8Tensor("out1.T")]
    for output, operand in zip(outputs, operands):
        output._te_gemm_ready_rowwise_transpose_for_backward = operand

    def fused_multi_output(*args):
        assert args == (tensor, [2, 2], quantizers, False, True)
        return outputs

    def fail_original_split(*_args, **_kwargs):
        raise AssertionError("multi-output path called original split_quantize")

    shim = _fresh_grouped_shim(
        monkeypatch,
        grouped_direct=False,
        grouped_quantize_producer="multi_output",
        tex_split_quantize=fail_original_split,
        tex_multi_output_quantize=fused_multi_output,
    )
    _reset_bwd_stats(shim)
    tensor = shim._torch.empty((4, 2), device="cpu")

    result = shim._tex.split_quantize(tensor, [2, 2], quantizers)

    assert result == outputs
    for operand in operands:
        assert getattr(operand, "_te_rowwise_transpose_for_backward_operand", False)

    stats = shim.cppmega_te_mxfp8_bwd_stats
    assert stats["mxfp8_grouped_quantize_producer_multi_output"] == 1
    assert stats["mxfp8_grouped_quantize_producer_multi_output_consumed"] == 2
    assert stats["mxfp8_grouped_quantize_producer_single_output"] == 0
    assert stats["mxfp8_grouped_quantize_producer_multi_output_missing_api"] == 0
    assert stats["mxfp8_tn_adapter_saved_transpose_operand"] == 0
    assert stats["mxfp8_tn_adapter_copy_transpose"] == 0
    snapshot = shim.cppmega_te_mxfp8_bwd_stats_snapshot()
    assert snapshot["mxfp8_tn_sidecar_registry_persistent"] == 0


def test_grouped_mxfp8_multi_output_producer_missing_api_fails_fast(
    monkeypatch,
):
    class FakeQuantizer:
        pass

    def fail_original_split(*_args, **_kwargs):
        raise AssertionError("missing multi-output API must not call original split_quantize")

    shim = _fresh_grouped_shim(
        monkeypatch,
        grouped_direct=False,
        grouped_quantize_producer="multi_output",
        tex_split_quantize=fail_original_split,
    )
    _reset_bwd_stats(shim)

    with pytest.raises(RuntimeError, match="requires transformer_engine_torch"):
        shim._tex.split_quantize(
            shim._torch.empty((4, 2), device="cpu"),
            [2, 2],
            [FakeQuantizer(), FakeQuantizer()],
        )

    stats = shim.cppmega_te_mxfp8_bwd_stats
    assert stats["mxfp8_grouped_quantize_producer_multi_output_missing_api"] == 1
    assert stats["mxfp8_grouped_quantize_producer_multi_output"] == 0
