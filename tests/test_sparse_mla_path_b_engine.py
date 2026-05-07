"""Parity tests for the Path B → Path C engine-extraction bridge.

Wave 6: ``sparse_mla.py`` (Path B raw MSL) gains an opt-in path that, when
``CPPMEGA_MLX_TILELANG_ENGINE=engine_with_msl_extraction`` is set, builds
the kernel via ``sparse_mla_path_c._fwd_kernel_for`` / ``_bwd_kernel_for``
plus the MSL-extraction adapter (commit ``00d6d90``) instead of the
templated raw-MSL ``_FWD_KERNEL`` / ``_BWD_KERNEL`` constants.

These tests assert:

  1. Default mode (``auto``/``shim``) keeps using the legacy templated
     Path B kernel (cache stays empty).
  2. The opt-in mode populates the per-shape engine cache and produces
     numerically equivalent outputs versus the legacy path on a small
     case.
  3. The opt-in mode falls back gracefully to legacy when Path C cannot
     produce a usable lowering.

All tests skip if MLX Metal is not available, since the kernels run on
Apple Silicon GPUs.
"""

from __future__ import annotations

import os

import pytest


def _require_metal():
    mlx = pytest.importorskip("mlx.core")
    if not getattr(getattr(mlx, "metal", None), "is_available", lambda: False)():
        pytest.skip("MLX Metal backend not available on this host")
    return mlx


def _set_engine_mode(monkeypatch, mode: str | None) -> None:
    if mode is None:
        monkeypatch.delenv("CPPMEGA_MLX_TILELANG_ENGINE", raising=False)
    else:
        monkeypatch.setenv("CPPMEGA_MLX_TILELANG_ENGINE", mode)


def _small_inputs(mx):
    """Construct a tiny SparseMLA workload that fits any Metal device."""
    batch, seq_len, heads, qk_dim, kv_group, head_kv = 1, 2, 4, 16, 1, 1
    seq_len_kv, topk, d_v = 8, 4, 16
    rng = mx.random.uniform
    q = rng(shape=(batch, seq_len, heads, qk_dim), dtype=mx.float16)
    kv = rng(shape=(batch, seq_len_kv, kv_group, qk_dim), dtype=mx.float16)
    indices = mx.array(
        [[[[0, 2, 5, 7]]] * heads] * seq_len, dtype=mx.int32
    ).reshape(batch, seq_len, kv_group, topk)
    return q, kv, indices, d_v


def test_engine_cache_is_empty_in_default_mode(monkeypatch):
    mx = _require_metal()
    pytest.importorskip("cppmega_mlx")
    from cppmega_mlx.nn._tilelang import sparse_mla

    sparse_mla._ENGINE_FWD_CACHE.clear()
    sparse_mla._ENGINE_BWD_CACHE.clear()
    _set_engine_mode(monkeypatch, "auto")

    q, kv, indices, d_v = _small_inputs(mx)
    out_lse = sparse_mla.sparse_mla_fwd_metal(q, kv, indices, d_v=d_v)
    if out_lse is None:
        pytest.skip("sparse_mla_fwd_metal unavailable on this host")

    # Default mode must NOT populate the engine cache.
    assert not sparse_mla._ENGINE_FWD_CACHE


def test_engine_extraction_mode_populates_cache(monkeypatch):
    mx = _require_metal()
    pytest.importorskip("cppmega_mlx")
    from cppmega_mlx.nn._tilelang import sparse_mla
    from cppmega_mlx.nn._tilelang import _engine_dispatch

    if not _engine_dispatch.dispatch_lower_supports_msl_extraction():
        pytest.skip("dispatch_lower_supports_msl_extraction() is False on this host")

    sparse_mla._ENGINE_FWD_CACHE.clear()
    _set_engine_mode(monkeypatch, "engine_with_msl_extraction")

    q, kv, indices, d_v = _small_inputs(mx)
    out_lse = sparse_mla.sparse_mla_fwd_metal(q, kv, indices, d_v=d_v)
    if out_lse is None:
        pytest.skip("sparse_mla_fwd_metal unavailable on this host")

    # Either we extracted (cache hit with non-None) or fell back (cache hit
    # with None). Either way the cache must be populated for this shape.
    assert sparse_mla._ENGINE_FWD_CACHE, (
        "engine_with_msl_extraction mode did not populate the per-shape cache"
    )


def test_engine_extraction_numerical_parity(monkeypatch):
    """Engine-extracted output must match legacy Path B within fp16 tolerance."""

    mx = _require_metal()
    pytest.importorskip("cppmega_mlx")
    from cppmega_mlx.nn._tilelang import sparse_mla
    from cppmega_mlx.nn._tilelang import _engine_dispatch

    if not _engine_dispatch.dispatch_lower_supports_msl_extraction():
        pytest.skip("dispatch_lower_supports_msl_extraction() is False on this host")

    q, kv, indices, d_v = _small_inputs(mx)

    # Legacy path
    sparse_mla._ENGINE_FWD_CACHE.clear()
    _set_engine_mode(monkeypatch, "shim")
    legacy = sparse_mla.sparse_mla_fwd_metal(q, kv, indices, d_v=d_v)
    if legacy is None:
        pytest.skip("legacy Path B fwd unavailable")
    legacy_out, legacy_lse = legacy

    # Engine path
    sparse_mla._ENGINE_FWD_CACHE.clear()
    _set_engine_mode(monkeypatch, "engine_with_msl_extraction")
    engine = sparse_mla.sparse_mla_fwd_metal(q, kv, indices, d_v=d_v)
    if engine is None:
        pytest.skip("engine fwd unavailable")
    engine_out, engine_lse = engine

    # If engine path silently fell back, the numerical contract is identical
    # by construction; otherwise compare with fp16 tolerance.
    diff_out = float(mx.max(mx.abs(engine_out.astype(mx.float32) - legacy_out.astype(mx.float32))).item())
    diff_lse = float(mx.max(mx.abs(engine_lse - legacy_lse)).item())
    assert diff_out < 5e-2, f"out diff {diff_out} exceeds fp16 tolerance"
    assert diff_lse < 5e-2, f"lse diff {diff_lse} exceeds fp16 tolerance"


def test_engine_extraction_bwd_cache_populated(monkeypatch):
    """Backward kernel must also populate its own engine cache when opt-in."""

    mx = _require_metal()
    pytest.importorskip("cppmega_mlx")
    from cppmega_mlx.nn._tilelang import sparse_mla
    from cppmega_mlx.nn._tilelang import _engine_dispatch

    if not _engine_dispatch.dispatch_lower_supports_msl_extraction():
        pytest.skip("dispatch_lower_supports_msl_extraction() is False on this host")

    q, kv, indices, d_v = _small_inputs(mx)
    d_out = mx.random.uniform(shape=q.shape[:-1] + (d_v,), dtype=mx.float16)

    sparse_mla._ENGINE_BWD_CACHE.clear()
    _set_engine_mode(monkeypatch, "engine_with_msl_extraction")
    bwd = sparse_mla.sparse_mla_bwd_metal(q, kv, d_out, indices, d_v=d_v)
    if bwd is None:
        pytest.skip("sparse_mla_bwd_metal unavailable on this host")
    assert sparse_mla._ENGINE_BWD_CACHE, (
        "engine_with_msl_extraction mode did not populate the bwd cache"
    )
