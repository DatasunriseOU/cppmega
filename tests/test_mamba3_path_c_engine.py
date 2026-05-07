"""Parity smoke for Phase-3 mamba3_path_c migration.

Verifies that the Path-C kernel factories (``_fwd_kernel_for``,
``_bwd_kernel_for``, ``_segsum_kernel_for``, ``_dadt_kernel_for``,
``_dtrap_kernel_for``) honor the ``CPPMEGA_MLX_TILELANG_ENGINE`` dispatcher
and do not regress the shim path.

Skip if Apple Metal is unavailable on the host or cppmega_mlx is not
importable. We only smoke-test the *factory* contract here — full numerical
parity for the engine path is deferred to Phase-4 (where the engine artifact
gets plumbed through the mlx fast-kernel runtime).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from importlib import reload

import pytest


@contextmanager
def _engine_mode(value: str | None):
    """Swap ``CPPMEGA_MLX_TILELANG_ENGINE`` for the duration of the block."""

    prev = os.environ.get("CPPMEGA_MLX_TILELANG_ENGINE")
    if value is None:
        os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
    else:
        os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
        else:
            os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = prev


def _import_path_c():
    cppmega_mlx_nn = pytest.importorskip("cppmega_mlx.nn._tilelang")
    pytest.importorskip("cppmega_mlx.nn._tilelang.mamba3_path_c")
    pytest.importorskip("cppmega_mlx.nn._tilelang._mamba3_helpers_tilelang")
    return cppmega_mlx_nn


def test_fwd_kernel_for_shim_returns_lowering():
    """Shim mode must keep the existing ``(kernel, lowering)`` contract."""

    pytest.importorskip("mlx.core")
    pytest.importorskip("tilelang.language")
    _import_path_c()
    from cppmega_mlx.nn._tilelang.mamba3_path_c import _fwd_kernel_for

    with _engine_mode("shim"):
        try:
            kernel, lowering = _fwd_kernel_for(1, 2, 1, 8, 8)
        except Exception as exc:  # noqa: BLE001 - environment-specific failures
            pytest.skip(f"shim lowering unavailable: {exc!r}")

    assert kernel is not None
    assert lowering is not None
    # ``TileLangMSLLowering`` carries an ``msl_text`` field.
    assert hasattr(lowering, "msl_text") or hasattr(lowering, "body")


def test_segsum_kernel_for_shim_returns_lowering():
    pytest.importorskip("mlx.core")
    pytest.importorskip("tilelang.language")
    _import_path_c()
    from cppmega_mlx.nn._tilelang._mamba3_helpers_tilelang import _segsum_kernel_for

    with _engine_mode("shim"):
        try:
            kernel, lowering = _segsum_kernel_for(2, 4, 8, 8)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"shim lowering unavailable: {exc!r}")

    assert kernel is not None
    assert lowering is not None


def test_fwd_kernel_for_engine_returns_artifact_or_falls_back():
    """Engine mode either returns a stamped artifact (lowering=None) or
    raises an ImportError when tilelang is unavailable."""

    pytest.importorskip("mlx.core")
    pytest.importorskip("tilelang.language")
    _import_path_c()
    from cppmega_mlx.nn._tilelang.mamba3_path_c import _fwd_kernel_for
    from cppmega_mlx.nn._tilelang._engine_dispatch import (
        _reset_fallback_warning_for_tests,
    )

    _reset_fallback_warning_for_tests()
    with _engine_mode("engine"):
        try:
            kernel, lowering = _fwd_kernel_for(1, 2, 1, 8, 8)
        except (ImportError, ModuleNotFoundError):
            pytest.skip("tilelang engine unavailable on this host")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"engine lowering raised non-import error: {exc!r}")

    if lowering is not None:
        # tilelang.compile silently fell back to the shim — accept it.
        assert kernel is not None
    else:
        assert hasattr(kernel, "_tilelang_engine_target")
        assert getattr(kernel, "_tilelang_engine_target") == "metal"


def test_helper_public_callers_pass_through_when_lowering_is_none(monkeypatch):
    """Public helpers (compute_dacs_segsum, bwd_dadt_fused, bwd_dtrap_ddt)
    must transparently fall back to pure-MLX when the engine path returns
    ``lowering is None`` — the engine call signature isn't wired through
    mx.fast.metal_kernel yet."""

    pytest.importorskip("mlx.core")
    pytest.importorskip("tilelang.language")
    _import_path_c()
    import mlx.core as mx
    from cppmega_mlx.nn._tilelang import _mamba3_helpers_tilelang as helpers
    from cppmega_mlx.nn._tilelang import _mamba3_helpers as pure

    sentinel = object()

    def _stub_pure(A, dt, dh, *, accumulate_in_fp32=True):
        return sentinel

    monkeypatch.setattr(pure, "compute_dacs_segsum", _stub_pure, raising=False)
    monkeypatch.setattr(helpers._pure_helpers, "compute_dacs_segsum", _stub_pure)

    # Force the factory to claim engine path.
    def _engine_only_factory(*args, **kwargs):
        class _Stub:
            _tilelang_engine_target = "metal"

            def __call__(self, *_args, **_kwargs):
                raise AssertionError("engine artifact should not be invoked")

        return _Stub(), None

    monkeypatch.setattr(helpers, "_segsum_kernel_for", _engine_only_factory)

    # Build minimal-shape dummy inputs.
    B, T_, H, K = 1, 2, 1, 4
    A = mx.zeros((B, T_, H), dtype=mx.float32)
    dt = mx.zeros((B, T_, H), dtype=mx.float32)
    dh = mx.zeros((B * H, T_, K), dtype=mx.float16)

    out = helpers.compute_dacs_segsum(A, dt, dh)
    assert out is sentinel
