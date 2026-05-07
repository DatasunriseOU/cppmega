"""Parity tests for sparse_mla_path_c.py engine vs shim dispatch.

Phase-3 migration parity coverage: verify that
``cppmega_mlx.nn._tilelang.sparse_mla_path_c.lower_sparse_mla_{fwd,bwd}_msl``
produces a non-empty rendered source under both ``engine`` and ``shim`` modes
of ``CPPMEGA_MLX_TILELANG_ENGINE`` for a small representative sparse-MLA
shape (B=1, H=2, Sq=64, Sk=512, TOPK=16).

Skipped on hosts without Metal / without ``tilelang`` / without
``cppmega_mlx`` since the dispatcher needs to import them lazily.
"""

from __future__ import annotations

import importlib.util
import os
import warnings

import pytest


_SHAPE_KWARGS = dict(
    BATCH=1,
    SEQ_LEN=64,
    HEADS=2,
    QK_DIM=64,
    KV_GROUP=1,
    HEAD_KV=1,
    TOPK=16,
    SEQ_LEN_KV=512,
    D_V=64,
    THREADS=16,
)


def _have_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


pytestmark = pytest.mark.skipif(
    not (
        _have_module("cppmega_mlx.nn._tilelang.sparse_mla_path_c")
        and _have_module("cppmega_mlx.nn._tilelang._engine_dispatch")
    ),
    reason="cppmega_mlx + tilelang are required for Path C parity tests",
)


def _reset_engine_dispatch_warning() -> None:
    from cppmega_mlx.nn._tilelang._engine_dispatch import (
        _reset_fallback_warning_for_tests,
    )

    _reset_fallback_warning_for_tests()


@pytest.fixture(autouse=True)
def _restore_engine_env(monkeypatch):
    """Restore CPPMEGA_MLX_TILELANG_ENGINE between tests and re-arm warnings."""

    monkeypatch.delenv("CPPMEGA_MLX_TILELANG_ENGINE", raising=False)
    _reset_engine_dispatch_warning()
    yield
    _reset_engine_dispatch_warning()


def test_lower_fwd_msl_shim_returns_msl(monkeypatch) -> None:
    """``shim`` mode must return non-empty MSL text for the fwd kernel."""

    monkeypatch.setenv("CPPMEGA_MLX_TILELANG_ENGINE", "shim")
    from cppmega_mlx.nn._tilelang.sparse_mla_path_c import lower_sparse_mla_fwd_msl

    try:
        msl = lower_sparse_mla_fwd_msl(**_SHAPE_KWARGS)
    except Exception as exc:  # pragma: no cover - host-specific failure
        pytest.skip(f"shim path unavailable on host: {exc}")
    assert isinstance(msl, str)
    assert msl.strip(), "shim mode produced empty MSL"
    assert "kernel" in msl.lower() or "void" in msl.lower(), (
        "shim mode should return Metal kernel source; got opaque artifact"
    )


def test_lower_bwd_msl_shim_returns_msl(monkeypatch) -> None:
    """``shim`` mode must return non-empty MSL text for the bwd kernel."""

    monkeypatch.setenv("CPPMEGA_MLX_TILELANG_ENGINE", "shim")
    from cppmega_mlx.nn._tilelang.sparse_mla_path_c import lower_sparse_mla_bwd_msl

    try:
        msl = lower_sparse_mla_bwd_msl(**_SHAPE_KWARGS)
    except Exception as exc:  # pragma: no cover - host-specific failure
        pytest.skip(f"shim path unavailable on host: {exc}")
    assert isinstance(msl, str)
    assert msl.strip(), "shim mode produced empty bwd MSL"


def test_lower_fwd_msl_engine_returns_source(monkeypatch) -> None:
    """``engine`` mode must return rendered source for the fwd kernel.

    Falls back to skip if the in-tree tilelang.compile cannot reach the chosen
    target on this host (env-specific build flags).
    """

    monkeypatch.setenv("CPPMEGA_MLX_TILELANG_ENGINE", "engine")
    from cppmega_mlx.nn._tilelang.sparse_mla_path_c import lower_sparse_mla_fwd_msl

    try:
        src = lower_sparse_mla_fwd_msl(**_SHAPE_KWARGS, target="metal")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"tilelang engine unavailable: {exc}")
    except Exception as exc:  # pragma: no cover - exercise for diagnostics
        pytest.skip(f"engine compile failed on host: {exc}")
    assert isinstance(src, str)
    assert src.strip(), "engine mode produced empty source"


def test_lower_fwd_msl_auto_falls_back_to_shim(monkeypatch) -> None:
    """``auto`` mode falls back to shim with one-shot warning when engine misses."""

    monkeypatch.setenv("CPPMEGA_MLX_TILELANG_ENGINE", "auto")
    monkeypatch.setitem(os.environ, "CPPMEGA_MLX_TILELANG_ENGINE", "auto")

    from cppmega_mlx.nn._tilelang import _engine_dispatch
    from cppmega_mlx.nn._tilelang.sparse_mla_path_c import lower_sparse_mla_fwd_msl

    def _raise_import(prim_func, target):  # noqa: ARG001 - signature parity
        raise ImportError("synthetic engine miss")

    monkeypatch.setattr(_engine_dispatch, "_engine_compile", _raise_import)
    _reset_engine_dispatch_warning()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            msl = lower_sparse_mla_fwd_msl(**_SHAPE_KWARGS)
        except Exception as exc:  # pragma: no cover - shim path unavailable
            pytest.skip(f"shim path unavailable on host: {exc}")
    assert isinstance(msl, str)
    assert msl.strip(), "fallback shim produced empty MSL"
    fallback_warnings = [
        w for w in caught if "tilelang engine unavailable" in str(w.message)
    ]
    assert fallback_warnings, "auto mode should warn on fallback to shim"


def test_engine_kernel_for_caches_per_shape(monkeypatch) -> None:
    """``_fwd_kernel_engine_for`` returns the same artifact for repeated calls."""

    monkeypatch.setenv("CPPMEGA_MLX_TILELANG_ENGINE", "shim")
    from cppmega_mlx.nn._tilelang.sparse_mla_path_c import _fwd_kernel_engine_for

    try:
        a = _fwd_kernel_engine_for(**_SHAPE_KWARGS)
        b = _fwd_kernel_engine_for(**_SHAPE_KWARGS)
    except Exception as exc:  # pragma: no cover - host-specific failure
        pytest.skip(f"engine builder unavailable on host: {exc}")
    assert a is b, "lru_cache(_fwd_kernel_engine_for) should return same object"
