"""Current owner-output contract for FP8 vecmat Path C."""

from __future__ import annotations

import inspect

import pytest


_PKG = "cppmega_mlx.nn._tilelang.fp8_vecmat_path_c"
mod = pytest.importorskip(_PKG)
mx = pytest.importorskip("mlx.core")


def _inputs():
    x = mx.zeros((4,), dtype=mx.uint8)
    weight = mx.zeros((2, 4), dtype=mx.uint8)
    scale_x = mx.ones((1,), dtype=mx.float32)
    scale_w = mx.ones((2,), dtype=mx.float32)
    return x, weight, scale_x, scale_w


def test_no_out_path_fails_closed() -> None:
    x, weight, scale_x, scale_w = _inputs()

    with pytest.raises(mod.FP8VecmatPathCLegacyError, match="owner-output"):
        mod.fp8_scaled_vecmat_path_c(
            x,
            weight,
            scale_x=scale_x,
            scale_w=scale_w,
        )


def test_owner_output_is_the_only_runtime_cache_surface() -> None:
    assert hasattr(mod, "_FP8_VECMAT_TVM_FFI_KERNEL_CACHE")
    assert hasattr(mod, "_FP8_VECMAT_TVM_FFI_KERNEL_CACHE_LOCK")
    assert not hasattr(mod, "_FP8_VECMAT_KERNEL_CACHE")
    assert not hasattr(mod, "_FP8_VECMAT_KERNEL_CACHE_LOCK")

    signature = inspect.signature(mod.fp8_scaled_vecmat_path_c)
    assert "out" in signature.parameters
    assert signature.parameters["out"].default is None


def test_module_documents_tvm_ffi_owner_output_replacement() -> None:
    doc = mod.__doc__ or ""
    assert "tvm-ffi owner-output" in doc
    assert "caller-owned" in doc
    assert "historical no-``out`` API is retired" in doc
