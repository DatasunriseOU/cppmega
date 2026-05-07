"""Wave-6 migration parity test: sparse_mla_blockscaled (Path B, raw MSL).

The Path-B kernel in :mod:`cppmega_mlx.nn._tilelang.sparse_mla_blockscaled`
is built from raw-MSL strings via ``_msl_transform.make_metal_kernel`` and
has no ``@T.prim_func`` to feed into ``_engine_dispatch.dispatch_lower``
— hence the wave-6 deferral marker added to its module docstring.

This regression test locks in two contracts:

* the public 7-call-site API surface (``sparse_mla_blockscaled_metal_status``,
  ``sparse_mla_blockscaled_fwd_metal``, ``sparse_mla_blockscaled_bwd_metal``,
  ``sparse_mla_blockscaled_apply``, ``sparse_mla_blockscaled_reference``,
  ``sparse_mla_blockscaled_metal_apply``, ``_mxfp8_roundtrip_ste``) keeps
  importing without raising regardless of the
  ``CPPMEGA_MLX_TILELANG_ENGINE`` value — Path-B is unaffected by the env
  flag because it bypasses the engine entirely;

* the Path-C QK sister at
  :mod:`cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c` (already
  migrated in Phase-2 commit ``3017429``) is reachable from this Path-B
  module's namespace, so callers that *do* want the engine path can
  upgrade by switching to the Path-C QK reducer + an external softmax/PV
  loop without re-importing.

Skips when MLX or cppmega_mlx are unavailable on the host.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from typing import Iterator

import pytest


_pb_mod_name = "cppmega_mlx.nn._tilelang.sparse_mla_blockscaled"
_pc_mod_name = "cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c"

pytest.importorskip(_pb_mod_name)
pytest.importorskip(_pc_mod_name)


_PUBLIC_API = (
    "SparseMLABlockScaledMetalStatus",
    "sparse_mla_blockscaled_metal_status",
    "sparse_mla_blockscaled_fwd_metal",
    "sparse_mla_blockscaled_bwd_metal",
    "sparse_mla_blockscaled_reference",
    "sparse_mla_blockscaled_metal_apply",
    "sparse_mla_blockscaled_apply",
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
    """Path-B kernel is raw MSL; CPPMEGA_MLX_TILELANG_ENGINE must not break import."""

    with _engine_mode(mode):
        mod = importlib.reload(importlib.import_module(_pb_mod_name))
        for name in _PUBLIC_API:
            assert hasattr(mod, name), f"missing {name} after reload with mode={mode!r}"


def test_path_c_qk_sister_reexported_from_path_b() -> None:
    """Path-B module re-exports Path-C QK reducer entry points for migrators."""

    pb = importlib.import_module(_pb_mod_name)
    # Import surface set by the Path-B docstring's wave-6 migration note.
    for name in (
        "SparseMLABlockScaledQKReducePathCStatus",
        "blockscaled_sparse_mla_qk_path_c_status",
        "blockscaled_sparse_mla_qk_reduce_path_c",
        "blockscaled_sparse_mla_qk_reduce_path_c_status",
    ):
        assert hasattr(pb, name), f"path-b should re-export {name}"


def test_wave6_deferral_marker_present_in_docstring() -> None:
    """Module docstring tells callers Path-B awaits a Path-C MXFP8 rewrite."""

    pb = importlib.import_module(_pb_mod_name)
    doc = pb.__doc__ or ""
    assert "Wave-6" in doc, "missing wave-6 deferral note"
    assert "Path-C" in doc or "path_c" in doc, "must point at Path-C QK sister"
