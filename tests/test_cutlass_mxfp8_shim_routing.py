from __future__ import annotations

import atexit
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch


class _FakeRowwiseMXFP8:
    def __init__(self, *, swizzled: bool):
        self._rowwise_data = torch.empty((128, 128), dtype=torch.uint8)
        self._rowwise_scale_inv = torch.empty((128, 4), dtype=torch.uint8)
        self._with_gemm_swizzled_scales = swizzled


class _FakeDenseMXFP8:
    def __init__(self, *, swizzled: bool = False):
        self._rowwise_data = torch.empty((128, 128), dtype=torch.uint8)
        self._rowwise_scale_inv = torch.empty((128, 4), dtype=torch.uint8)
        self._columnwise_data = torch.empty((128, 128), dtype=torch.uint8)
        self._columnwise_scale_inv = torch.empty((4, 128), dtype=torch.uint8)
        self._with_gemm_swizzled_scales = swizzled


def _fresh_shim(monkeypatch, *, scale_backend: str, backend: str = "cutlass_native"):
    for key in (
        "CPPMEGA_TE_MXFP8_DGRAD_BF16",
        "CPPMEGA_TE_MXFP8_WGRAD_BF16",
        "NVTE_BACKWARD_OVERRIDE",
        "CPPMEGA_TE_MXFP8_COMPACT_COLUMNWISE_BACKWARD",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER", "1")
    monkeypatch.setenv("CPPMEGA_TE_MXFP8_BWD_BACKEND", backend)
    monkeypatch.setenv("CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND", scale_backend)
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
    if not hasattr(shim, "_cppmega_cutlass_tn_gemm"):
        pytest.skip("cppmega_fp8_shim CUTLASS route was not installed")
    return shim


def test_swizzled_cutlass_route_uses_stock_gemm_and_packs_only_compact_scale(
    monkeypatch,
):
    shim = _fresh_shim(monkeypatch, scale_backend="swizzled")
    compact = _FakeRowwiseMXFP8(swizzled=False)
    ready = _FakeRowwiseMXFP8(swizzled=True)
    calls = []
    swizzled_outputs = []

    def swizzle_rowwise_scale(scale, rows, cols):
        calls.append(("swizzle", scale, rows, cols))
        out = torch.empty((rows * (cols // 32),), dtype=torch.uint8)
        swizzled_outputs.append(out)
        return out

    def tn_gemm_swizzled_scale(a_data, a_scale, b_data, b_scale, **kwargs):
        calls.append(("stock", a_data, a_scale, b_data, b_scale, kwargs))
        return "stock-result"

    def tn_gemm_direct_rowwise(*_args, **_kwargs):
        raise AssertionError("swizzled scale backend should not use compact direct GEMM")

    monkeypatch.setattr(
        shim,
        "_cppmega_cutlass_mxfp8_module",
        [
            SimpleNamespace(
                swizzle_rowwise_scale=swizzle_rowwise_scale,
                tn_gemm_swizzled_scale=tn_gemm_swizzled_scale,
                tn_gemm_direct_rowwise=tn_gemm_direct_rowwise,
            )
        ],
    )

    result, *_ = shim._cppmega_cutlass_tn_gemm(
        compact,
        ready,
        {"out_dtype": torch.bfloat16},
    )

    assert result == "stock-result"
    assert calls[0] == ("swizzle", compact._rowwise_scale_inv, 128, 128)
    assert calls[1][0] == "stock"
    assert calls[1][2] is swizzled_outputs[0]
    assert calls[1][4] is ready._rowwise_scale_inv
    assert shim.cppmega_te_mxfp8_bwd_stats["mxfp8_cutlass_native_stock_swizzled"] == 1
    assert shim.cppmega_te_mxfp8_bwd_stats["mxfp8_cutlass_native_stock_scale_swizzle"] == 1


def test_compact_cutlass_route_keeps_direct_rowwise_gemm(monkeypatch):
    shim = _fresh_shim(monkeypatch, scale_backend="compact")
    a = _FakeRowwiseMXFP8(swizzled=False)
    b = _FakeRowwiseMXFP8(swizzled=False)
    calls = []

    def tn_gemm_direct_rowwise(a_data, a_scale, b_data, b_scale, **kwargs):
        calls.append((a_data, a_scale, b_data, b_scale, kwargs))
        return "direct-result"

    monkeypatch.setattr(
        shim,
        "_cppmega_cutlass_mxfp8_module",
        [
            SimpleNamespace(
                tn_gemm_direct_rowwise=tn_gemm_direct_rowwise,
            )
        ],
    )

    result, *_ = shim._cppmega_cutlass_tn_gemm(a, b, {"out_dtype": torch.bfloat16})

    assert result == "direct-result"
    assert len(calls) == 1
    assert calls[0][0] is a._rowwise_data
    assert calls[0][1] is a._rowwise_scale_inv
    assert calls[0][2] is b._rowwise_data
    assert calls[0][3] is b._rowwise_scale_inv


def test_wave16c_dgrad_route_uses_colwise_weight_without_transpose(monkeypatch):
    shim = _fresh_shim(
        monkeypatch,
        scale_backend="compact",
        backend="wave16c_streaming_stock",
    )
    weight = _FakeDenseMXFP8()
    dy = _FakeDenseMXFP8()
    calls = []

    def dgrad_nn_gemm_streaming_swizzled_stock(
        dy_data,
        dy_scale,
        weight_data,
        weight_scale,
        **kwargs,
    ):
        calls.append((dy_data, dy_scale, weight_data, weight_scale, kwargs))
        return "wave16c-dgrad"

    monkeypatch.setattr(
        shim,
        "_cppmega_mxfp8_colwise_as_rowwise_transpose",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("wave16c route must not request a rowwise transpose")
        ),
    )
    monkeypatch.setattr(
        shim,
        "_cppmega_cutlass_mxfp8_module",
        [SimpleNamespace(dgrad_nn_gemm_streaming_swizzled_stock=dgrad_nn_gemm_streaming_swizzled_stock)],
    )

    ok, result = shim._cppmega_try_mxfp8_tn_adapter(
        None,
        [weight, dy],
        {"layout": "NN", "grad": True, "out_dtype": torch.bfloat16},
    )

    assert ok is True
    assert result[0] == "wave16c-dgrad"
    assert len(calls) == 1
    assert calls[0][0] is dy._rowwise_data
    assert calls[0][1] is dy._rowwise_scale_inv
    assert calls[0][2] is weight._columnwise_data
    assert calls[0][3] is weight._columnwise_scale_inv


def test_wave16c_wgrad_route_uses_colwise_operands_without_transpose(monkeypatch):
    shim = _fresh_shim(
        monkeypatch,
        scale_backend="compact",
        backend="wave16c_streaming_stock",
    )
    x = _FakeDenseMXFP8()
    dy = _FakeDenseMXFP8()
    calls = []

    def wgrad_nt_gemm_streaming_swizzled_stock_colwise(
        dy_data,
        dy_scale,
        x_data,
        x_scale,
        **kwargs,
    ):
        calls.append((dy_data, dy_scale, x_data, x_scale, kwargs))
        return "wave16c-wgrad"

    monkeypatch.setattr(
        shim,
        "_cppmega_mxfp8_colwise_as_rowwise_transpose",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("wave16c route must not request a rowwise transpose")
        ),
    )
    monkeypatch.setattr(
        shim,
        "_cppmega_cutlass_mxfp8_module",
        [
            SimpleNamespace(
                wgrad_nt_gemm_streaming_swizzled_stock_colwise=(
                    wgrad_nt_gemm_streaming_swizzled_stock_colwise
                )
            )
        ],
    )

    ok, result = shim._cppmega_try_mxfp8_tn_adapter(
        None,
        [x, dy],
        {"layout": "NT", "grad": True, "out_dtype": torch.bfloat16},
    )

    assert ok is True
    assert result[0] == "wave16c-wgrad"
    assert len(calls) == 1
    assert calls[0][0] is dy._columnwise_data
    assert calls[0][1] is dy._columnwise_scale_inv
    assert calls[0][2] is x._columnwise_data
    assert calls[0][3] is x._columnwise_scale_inv
