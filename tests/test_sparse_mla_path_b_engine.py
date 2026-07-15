"""Regression coverage for sparse-MLA after direct-MSL Path B retirement."""

from __future__ import annotations

import inspect

import pytest


sparse_mla = pytest.importorskip("cppmega_mlx.nn._tilelang.sparse_mla")
mx = pytest.importorskip("mlx.core")
reference = pytest.importorskip("cppmega_mlx.nn.sparse_mla")


def _small_inputs():
    q = mx.random.uniform(shape=(1, 2, 4, 16), dtype=mx.float16)
    kv = mx.random.uniform(shape=(1, 8, 1, 16), dtype=mx.float16)
    indices = mx.array([[[[0, 2, 5, 7]], [[1, 3, 4, 6]]]], dtype=mx.int32)
    return q, kv, indices


def test_direct_msl_status_is_explicitly_unavailable() -> None:
    status = sparse_mla.sparse_mla_metal_status()
    assert status.available is False
    assert status.reason


def test_force_metal_fails_closed_instead_of_proxying() -> None:
    q, kv, indices = _small_inputs()

    with pytest.raises(RuntimeError, match="Metal path unavailable"):
        sparse_mla.sparse_mla_apply(q, kv, indices, force_metal=True)


def test_default_compatibility_surface_matches_reference() -> None:
    q, kv, indices = _small_inputs()
    actual = sparse_mla.sparse_mla_apply(q, kv, indices)
    expected = reference.sparse_mla_attention_reference(q, kv, indices)
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=1e-4, atol=1e-4).item()


def test_retired_module_has_no_engine_caches_or_direct_dispatch() -> None:
    source = inspect.getsource(sparse_mla)
    assert not hasattr(sparse_mla, "_ENGINE_FWD_CACHE")
    assert not hasattr(sparse_mla, "_ENGINE_BWD_CACHE")
    assert "_msl_transform.dispatch(" not in source
    assert "mx.fast.metal_kernel(" not in source
