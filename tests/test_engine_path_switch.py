"""Phase-1 migration smoke tests for the unified TileLang engine dispatcher.

The dispatcher (``cppmega_mlx.nn._tilelang._engine_dispatch.dispatch_lower``)
chooses between the unified ``tilelang.compile`` engine path and the legacy
``_msl_transform.lower_tilelang_to_msl_inline`` shim based on the
``CPPMEGA_MLX_TILELANG_ENGINE`` env var. These tests verify the three modes
(``engine`` / ``shim`` / ``auto``) without exercising the Metal runtime.
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Iterator

import pytest


_dispatch_mod_name = "cppmega_mlx.nn._tilelang._engine_dispatch"
pytest.importorskip(_dispatch_mod_name)


@contextmanager
def _engine_mode(mode: str | None) -> Iterator[None]:
    prev = os.environ.get("CPPMEGA_MLX_TILELANG_ENGINE")
    if mode is None:
        os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
    else:
        os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = mode
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
        else:
            os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = prev


def _make_test_prim_func():
    """Build a tiny TileLang PrimFunc usable by both engine and shim paths."""

    tilelang_lang = pytest.importorskip("tilelang.language")
    T = tilelang_lang  # noqa: N806 - matches TileLang DSL idiom

    @T.prim_func
    def kernel(  # type: ignore[no-untyped-def]
        X: T.Tensor((128,), "float32"),  # type: ignore[name-defined]
        Y: T.Tensor((128,), "float32"),  # type: ignore[name-defined]
    ):
        with T.Kernel(1, threads=32) as bx:  # type: ignore[name-defined]
            for i in T.Parallel(128):  # type: ignore[name-defined]
                Y[i] = X[i]

    return kernel


def test_engine_mode_returns_compiled_artifact_with_target_tag():
    """Force ``engine`` mode -> result carries ``_tilelang_engine_target``."""

    pytest.importorskip("tilelang")
    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower

    prim = _make_test_prim_func()
    with _engine_mode("engine"):
        try:
            artifact = dispatch_lower(prim, "metal -thread_warp_size=32")
        except Exception as exc:  # pragma: no cover - tilelang/Metal env-dep
            pytest.skip(f"engine mode unavailable on this host: {exc}")
        assert hasattr(artifact, "_tilelang_engine_target"), (
            "engine path must stamp _tilelang_engine_target on the compiled artifact"
        )
        assert artifact._tilelang_engine_target.startswith("metal")


def test_shim_mode_returns_msl_string_lowering():
    """Force ``shim`` mode -> result is a TileLangMSLLowering with ``msl_text``."""

    pytest.importorskip("tilelang")
    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower
    from cppmega_mlx.nn._tilelang._msl_transform import (
        TileLangMSLLowering,
    )

    prim = _make_test_prim_func()
    with _engine_mode("shim"):
        try:
            lowering = dispatch_lower(prim, "metal")
        except Exception as exc:  # pragma: no cover - shim depends on tilelang lower path
            pytest.skip(f"MSL shim unavailable on this host: {exc}")
        assert isinstance(lowering, TileLangMSLLowering), (
            f"shim mode must return TileLangMSLLowering, got {type(lowering).__name__}"
        )
        assert isinstance(lowering.msl_text, str) and lowering.msl_text, (
            "shim mode must produce a non-empty msl_text"
        )


def test_auto_mode_falls_back_to_shim_with_warning_when_engine_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``auto`` mode + tilelang.compile ImportError => one-shot UserWarning + shim."""

    from cppmega_mlx.nn._tilelang import _engine_dispatch

    _engine_dispatch._reset_fallback_warning_for_tests()

    pytest.importorskip("tilelang")
    prim = _make_test_prim_func()

    def _raise_import(*_args, **_kwargs):
        raise ImportError("simulated tilelang outage for fallback test")

    monkeypatch.setattr(_engine_dispatch, "_engine_compile", _raise_import)

    with _engine_mode("auto"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            lowering = _engine_dispatch.dispatch_lower(prim, "metal")
        except Exception as exc:  # pragma: no cover - shim env-dep
            pytest.skip(f"shim path unavailable on this host: {exc}")
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("falling back to MSL shim" in m for m in msgs), (
            f"expected one-shot fallback UserWarning; got {msgs!r}"
        )
        assert hasattr(lowering, "msl_text"), (
            "fallback must yield a TileLangMSLLowering (msl_text attribute)"
        )


def test_invalid_mode_warns_and_defaults_to_auto():
    """Garbage env value -> UserWarning, mode treated as auto."""

    from cppmega_mlx.nn._tilelang._engine_dispatch import tilelang_engine_mode

    with _engine_mode("garbage"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mode = tilelang_engine_mode()
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert mode == "auto"
        assert any("CPPMEGA_MLX_TILELANG_ENGINE" in m for m in msgs)
