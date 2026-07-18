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


def test_docstring_records_direct_msl_retirement() -> None:
    """Path B must remain an explicit retired compatibility surface."""

    pb = importlib.import_module(_pb_mod_name)
    doc = pb.__doc__ or ""
    assert "retired" in doc.lower()
    assert "direct-msl" in doc.lower()
    assert "sparse_mla_fp8_path_c" in doc


def test_path_c_fp8_sister_documents_prepared_owner_output_contract() -> None:
    """The replacement route must consume prepared buffers through tvm-ffi."""

    pc = importlib.import_module("cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c")
    doc = pc.__doc__ or ""
    assert "prepared" in doc.lower()
    assert "tvm-ffi" in doc.lower()
    assert "direct-MSL Path B" in doc
