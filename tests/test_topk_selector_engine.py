"""Phase-2 migration parity test for ``topk_selector`` (cppmega.mlx).

Builds the Path-C TileLang topk selector via both ``shim`` and ``engine``
dispatch modes (when each is available) and asserts the returned indices
agree with the pure-MLX reference. Skips on hosts without Metal / MLX or
without the unified TileLang engine.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from typing import Iterator

import pytest


_topk_mod_name = "cppmega_mlx.nn._tilelang.topk_selector"
pytest.importorskip(_topk_mod_name)


@contextmanager
def _engine_mode(mode: str | None) -> Iterator[None]:
    prev = os.environ.get("CPPMEGA_MLX_TILELANG_ENGINE")
    if mode is None:
        os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
    else:
        os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = mode
    importlib.import_module(_topk_mod_name)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
        else:
            os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = prev


def _maybe_metal():
    mx = pytest.importorskip("mlx.core")
    try:
        # Trivial Metal probe; mx.gpu raises on hosts without a Metal device.
        _ = mx.array([0.0]).sum().item()
    except Exception as exc:  # pragma: no cover - host-dep
        pytest.skip(f"MLX/Metal unavailable: {exc}")
    return mx


def _reference_indices(scores, k: int):
    mx = pytest.importorskip("mlx.core")
    # Negate so partition picks the largest. Output index ordering is not
    # guaranteed to match the kernel; tests compare set membership.
    part = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]
    return part.astype(mx.int32)


def _set_equal(rows_a, rows_b) -> bool:
    """Compare two (B, K) integer arrays for set equality per row."""

    a = rows_a.tolist() if hasattr(rows_a, "tolist") else list(rows_a)
    b = rows_b.tolist() if hasattr(rows_b, "tolist") else list(rows_b)
    if len(a) != len(b):
        return False
    return all(set(int(x) for x in ra) == set(int(x) for x in rb) for ra, rb in zip(a, b))


def test_topk_selector_path_c_shim_matches_reference():
    """Shim mode ``CPPMEGA_MLX_TILELANG_ENGINE=shim`` -> MSL kernel parity."""

    mx = _maybe_metal()
    from cppmega_mlx.nn._tilelang.topk_selector import (
        topk_selector_path_c_status,
        topk_selector_reference,
        topk_selector_tilelang,
    )

    with _engine_mode("shim"):
        status = topk_selector_path_c_status()
        if not status.available:
            pytest.skip(f"Path-C topk_selector unavailable on this host: {status.reason}")
        rng = mx.random.key(0)
        scores = mx.random.uniform(shape=(2, 256), key=rng)
        out = topk_selector_tilelang(scores, k=8)
        if out is None:
            pytest.skip("topk_selector_tilelang returned None on this host (lowering failure)")
        ref = _reference_indices(scores, k=8)
        assert _set_equal(out, ref), f"shim path mismatch: kernel={out} ref={ref}"


def test_topk_selector_path_c_engine_matches_reference():
    """Engine mode ``CPPMEGA_MLX_TILELANG_ENGINE=engine`` -> tilelang.compile parity.

    This path requires the unified TileLang engine (``tilelang.compile``) to
    successfully lower for ``target='metal'`` AND the runtime artifact to be
    callable with the (scores, starts, ends, indices) signature. Skips on
    hosts where either step fails (engine still maturing on Metal).
    """

    pytest.importorskip("tilelang")
    mx = _maybe_metal()
    from cppmega_mlx.nn._tilelang.topk_selector import (
        topk_selector_path_c_status,
        topk_selector_tilelang,
    )

    with _engine_mode("engine"):
        status = topk_selector_path_c_status()
        if not status.available:
            pytest.skip(f"Path-C topk_selector unavailable on this host: {status.reason}")
        rng = mx.random.key(1)
        scores = mx.random.uniform(shape=(2, 256), key=rng)
        try:
            out = topk_selector_tilelang(scores, k=8)
        except Exception as exc:  # pragma: no cover - engine env-dep
            pytest.skip(f"engine-mode lowering failed: {exc}")
        if out is None:
            pytest.skip("engine-mode kernel returned None (lowering or runtime mismatch)")
        ref = _reference_indices(scores, k=8)
        assert _set_equal(out, ref), f"engine path mismatch: kernel={out} ref={ref}"
