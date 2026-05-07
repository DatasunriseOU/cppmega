"""Wave-6 migration parity test: sparse_mla_fp8 (Path B, raw MSL).

The Path-B FP8 kernel in :mod:`cppmega_mlx.nn._tilelang.sparse_mla_fp8`
is built from raw-MSL strings via ``_msl_transform.make_metal_kernel``.
There is no Path-C TileLang sister: the FP8 e4m3 inline byte-level
dequant has no DSL equivalent until ``simdgroup_a_fp8`` /
``simdgroup_b_fp8`` ``Fragment`` factories land in
``tilelang/language/extern.py`` AND Apple ships a documented MSL
``float8 simdgroup_matrix`` MMA path. See module docstring for the
wave-6 deferral note.

This regression test locks in two contracts:

* the public 4-call-site API surface
  (``sparse_mla_fp8_metal_status``, ``sparse_mla_fp8_fwd_metal``,
  ``sparse_mla_fp8_bwd_metal``, ``sparse_mla_fp8_apply``,
  ``sparse_mla_fp8_reference``, ``sparse_mla_fp8_metal_apply``,
  ``_fp8_roundtrip_ste``) keeps importing without raising regardless
  of the ``CPPMEGA_MLX_TILELANG_ENGINE`` env value — Path-B bypasses
  the engine entirely;

* the wave-6 deferral marker stays in the module docstring so future
  migrators know FP8 factory work is the gate, not an oversight.

Skips when MLX or cppmega_mlx are unavailable on the host.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from typing import Iterator

import pytest


_pb_mod_name = "cppmega_mlx.nn._tilelang.sparse_mla_fp8"

pytest.importorskip(_pb_mod_name)


_PUBLIC_API = (
    "SparseMLAFp8MetalStatus",
    "sparse_mla_fp8_metal_status",
    "sparse_mla_fp8_fwd_metal",
    "sparse_mla_fp8_bwd_metal",
    "sparse_mla_fp8_reference",
    "sparse_mla_fp8_metal_apply",
    "sparse_mla_fp8_apply",
)


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


@pytest.mark.parametrize("mode", ["shim", "engine", "auto", None])
def test_path_b_module_is_engine_flag_invariant(mode: str | None) -> None:
    """Path-B FP8 kernel is raw MSL; CPPMEGA_MLX_TILELANG_ENGINE must not break import."""

    with _engine_mode(mode):
        mod = importlib.reload(importlib.import_module(_pb_mod_name))
        for name in _PUBLIC_API:
            assert hasattr(mod, name), f"missing {name} after reload with mode={mode!r}"


def test_wave6_deferral_marker_present_in_docstring() -> None:
    """Module docstring documents the FP8 factory blocker."""

    pb = importlib.import_module(_pb_mod_name)
    doc = pb.__doc__ or ""
    assert "Wave-6" in doc, "missing wave-6 deferral note"
    assert "simdgroup_a_fp8" in doc or "FP8" in doc, "must mention FP8 factory blocker"
    assert "Apple" in doc or "MSL" in doc, "must mention MSL float8 hardware blocker"


def test_path_c_fp8_sister_carries_todo_marker() -> None:
    """`sparse_mla_fp8_path_c` module exists but is marked deferred until factories land."""

    pc = importlib.import_module("cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c")
    doc = (pc.__doc__ or "")
    # The Phase-3 sparse_mla agent (commit a3dd633) added an FP8 factory TODO
    # to this sister module — keep that marker locked in until the factories
    # land in tilelang/language/extern.py.
    assert any(
        token in doc for token in ("TODO", "tirx.metal.fp8_e4m3_dot4", "simdgroup_a_fp8", "Wave-6", "wave-6", "deferred")
    ), "fp8 path_c sister must keep its FP8 factory deferral marker"
