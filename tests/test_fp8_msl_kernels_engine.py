"""Regression coverage for the retired FP8 direct-MSL compatibility surface."""

from __future__ import annotations

import importlib
import inspect
import os
from contextlib import contextmanager
from typing import Iterator

import pytest


_PKG = "cppmega_mlx.nn._tilelang.fp8_msl_kernels"
pytest.importorskip(_PKG)


@contextmanager
def _engine_mode(mode: str | None) -> Iterator[None]:
    previous = os.environ.get("CPPMEGA_MLX_TILELANG_ENGINE")
    if mode is None:
        os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
    else:
        os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
        else:
            os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = previous


@pytest.mark.parametrize("mode", [None, "auto", "shim", "engine"])
def test_retired_status_is_engine_flag_invariant(mode: str | None) -> None:
    with _engine_mode(mode):
        mod = importlib.reload(importlib.import_module(_PKG))
        status = mod.fp8_msl_status()

    assert status.available is False
    assert status.dispatch_surface == "retired_direct_msl_pure_mlx_reference"
    assert "retired" in status.reason.lower()
    assert "fp8_matmul_path_c.py" in status.reason
    assert "fp8_vecmat_path_c.py" in status.reason


def test_retired_module_does_not_construct_direct_msl_runtime() -> None:
    mod = importlib.import_module(_PKG)
    source = inspect.getsource(mod)

    assert "mx.fast.metal_kernel(" not in source
    assert "_msl_transform.dispatch(" not in source
    assert "make_metal_kernel(" not in source


def test_reference_oracle_api_remains_explicit() -> None:
    mod = importlib.import_module(_PKG)
    expected = {
        "fp8_to_half",
        "half_to_fp8",
        "fp8_scaled_matmul",
        "fp8_scaled_matmul_raw",
        "fp8_scaled_vecmat",
    }

    assert expected.issubset(set(mod.__all__))
    assert all(callable(getattr(mod, name)) for name in expected)
    assert "pure-MLX reference/oracle" in (mod.__doc__ or "")
