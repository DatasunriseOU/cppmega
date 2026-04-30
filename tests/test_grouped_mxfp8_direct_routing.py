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


def _fresh_grouped_shim(monkeypatch, *, grouped_direct: bool = False):
    for key in (
        "CPPMEGA_TE_MXFP8_DGRAD_BF16",
        "CPPMEGA_TE_MXFP8_WGRAD_BF16",
        "NVTE_BACKWARD_OVERRIDE",
        "CPPMEGA_TE_MXFP8_GROUPED_DIRECT_BACKWARD",
        "CPPMEGA_TE_MXFP8_GROUPED_GEMM_READY_BACKWARD",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER", "1")
    monkeypatch.setenv(
        "CPPMEGA_TE_MXFP8_GROUPED_DIRECT_BACKWARD",
        "1" if grouped_direct else "0",
    )
    monkeypatch.setenv("CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND", "off")
    monkeypatch.setenv("CPPMEGA_TE_VERSION_STRICT", "0")
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
