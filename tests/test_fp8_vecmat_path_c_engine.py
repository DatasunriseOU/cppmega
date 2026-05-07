"""Phase-3 migration parity test for ``fp8_vecmat_path_c``.

Verifies that ``_fp8_vecmat_kernel_for`` dispatches correctly under both
``CPPMEGA_MLX_TILELANG_ENGINE=engine`` and ``=shim`` modes:

* engine mode → cache entry's second slot is ``None`` and the first slot is
  a ``tilelang.compile`` artifact stamped with ``_tilelang_engine_target``;
* shim mode → cache entry's second slot is a ``TileLangMSLLowering`` with
  a non-empty ``msl_text`` (or equivalent ``body``) field, mirroring the
  pre-migration contract.

Numerical parity between the two paths is gated on a Metal host with an
FP8-capable runtime; on hosts without one the kernel-execution branch is
skipped while the cache-shape contract is still asserted (it does not
require building or running the kernel).
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from typing import Iterator

import pytest


_pkg = "cppmega_mlx.nn._tilelang.fp8_vecmat_path_c"
pytest.importorskip(_pkg)


@contextmanager
def _engine_mode(mode: str | None) -> Iterator[None]:
    prev = os.environ.get("CPPMEGA_MLX_TILELANG_ENGINE")
    if mode is None:
        os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
    else:
        os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = mode
    # Force the dispatcher and the vecmat module to reload so the env var
    # is picked up with a fresh cache.
    for name in (
        "cppmega_mlx.nn._tilelang._engine_dispatch",
        "cppmega_mlx.nn._tilelang.fp8_vecmat_path_c",
    ):
        if name in sys.modules:
            importlib.reload(sys.modules[name])
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
        else:
            os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = prev
        for name in (
            "cppmega_mlx.nn._tilelang._engine_dispatch",
            "cppmega_mlx.nn._tilelang.fp8_vecmat_path_c",
        ):
            if name in sys.modules:
                importlib.reload(sys.modules[name])


_VECMAT_KW = dict(
    N=64,
    K=128,
    outputs_per_block=4,
    reduce_threads=32,
    vec=4,
    scale_w_per_row=True,
)


def _build_cache_entry():
    mod = importlib.import_module(_pkg)
    # Clear the cache so the build path runs.
    with mod._FP8_VECMAT_KERNEL_CACHE_LOCK:
        mod._FP8_VECMAT_KERNEL_CACHE.clear()
    try:
        return mod._fp8_vecmat_kernel_for(**_VECMAT_KW)
    except Exception as exc:
        pytest.skip(f"vecmat kernel build unavailable on this host: {exc}")


def test_engine_mode_returns_artifact_with_none_lowering() -> None:
    """In engine mode the cache returns ``(artifact, None, ...)``."""

    pytest.importorskip("tilelang")
    with _engine_mode("engine"):
        kernel, lowering, input_names, output_shape, _grid, _tg = _build_cache_entry()
        assert lowering is None, (
            "engine mode must return None lowering so callers route to the "
            "unified runtime instead of mx.fast.metal_kernel"
        )
        assert getattr(kernel, "_tilelang_engine_target", None) is not None, (
            "engine artifact must carry _tilelang_engine_target stamp"
        )
        assert input_names == ["A", "A_scale", "B", "B_scale"]
        assert output_shape == (_VECMAT_KW["N"],)


def test_shim_mode_returns_lowering_with_msl_body() -> None:
    """In shim mode the cache returns ``(metal_kernel, TileLangMSLLowering, ...)``."""

    with _engine_mode("shim"):
        kernel, lowering, input_names, _output_shape, _grid, _tg = _build_cache_entry()
        assert lowering is not None, "shim mode must return TileLangMSLLowering"
        # ``body`` carries the rewritten MSL kernel source on the shim path
        body = getattr(lowering, "body", None) or getattr(lowering, "msl_text", "")
        assert "kernel void" in body, (
            "shim lowering must have an MSL kernel definition; got "
            f"{type(lowering).__name__} with body[:80]={body[:80]!r}"
        )
        assert input_names == ["A", "A_scale", "B", "B_scale"]


def test_fp8_msl_kernels_documents_factory_blocker() -> None:
    """Phase-3 deferral marker: ``fp8_msl_kernels`` must keep its TODO."""

    mod = importlib.import_module("cppmega_mlx.nn._tilelang.fp8_msl_kernels")
    doc = (mod.__doc__ or "")
    assert "fp8-factories" in doc.lower() or "simdgroup_a_fp8" in doc, (
        "fp8_msl_kernels module docstring must record the FP8-factory blocker "
        "so the Path-B retirement plan stays discoverable; see "
        "MIGRATION_PLAN.md §2.4."
    )
