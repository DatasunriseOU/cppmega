"""Parity scaffold for the mamba3.py Path-B engine flip (wave-6 follow-up).

This test exercises the existing raw-MSL Path-B path and is the regression
gate for the future wave-6 flip to ``dispatch_lower(prim, "metal",
return_msl=True)``. When the flip lands, the same numerical comparison must
hold; the test will then activate a second pass through the engine path and
assert byte-for-byte (or fp32 within rtol=1e-4) parity between the two.

Until wave-6 lands the ``engine`` pass is skipped via
``pytest.importorskip`` on a sentinel that the wave-6 commit will introduce
(``mamba3_mimo_fwd_prim`` factory). The shim path runs whenever Metal is
available so we lock in the current numerical contract before the rewrite.
"""

from __future__ import annotations

import pytest


def _metal_available() -> bool:
    try:
        import mlx.core as mx
    except Exception:
        return False
    metal = getattr(mx, "metal", None)
    is_avail = getattr(metal, "is_available", None)
    return bool(is_avail()) if callable(is_avail) else False


@pytest.fixture(scope="module")
def _mamba3_module():
    pytest.importorskip("cppmega_mlx.nn._tilelang.mamba3")
    if not _metal_available():
        pytest.skip("Apple Metal not available on this host")
    from cppmega_mlx.nn._tilelang import mamba3 as _mamba3

    return _mamba3


def _make_inputs(B: int = 1, T: int = 8, H: int = 2, P: int = 4, N: int = 4):
    import mlx.core as mx

    rng = mx.random.key(0xC0FFEE)
    sub = lambda i: mx.random.split(rng, i + 2)[i]  # noqa: E731
    x = mx.random.normal(shape=(B, T, H, P), key=sub(0)).astype(mx.float16)
    Bp = mx.random.normal(shape=(B, T, H, N), key=sub(1)).astype(mx.float16)
    Cp = mx.random.normal(shape=(B, T, H, N), key=sub(2)).astype(mx.float16)
    z = mx.random.normal(shape=(B, T, H, P), key=sub(3)).astype(mx.float16)
    A = mx.random.normal(shape=(B, T, H), key=sub(4)).astype(mx.float32) * -0.1
    dt = (
        mx.random.uniform(shape=(B, T, H), key=sub(5)).astype(mx.float32) * 0.1
        + 1e-3
    )
    D = mx.random.normal(shape=(H,), key=sub(6)).astype(mx.float32)
    h0 = mx.zeros((B, H, P, N), dtype=mx.float32)
    return x, Bp, Cp, z, A, dt, D, h0


def test_mamba3_fwd_metal_matches_reference(_mamba3_module):
    """Path-B forward must match the pure-MLX reference within fp16 tolerance.

    Locks in the numerical contract that wave-6's prim_func port has to
    preserve byte-for-byte (or within fp32 rtol=1e-4 once both paths run in
    fp32 internal accumulators).
    """

    import mlx.core as mx

    x, Bp, Cp, z, A, dt, D, h0 = _make_inputs()
    status = _mamba3_module.mamba3_mimo_metal_status()
    if not status.available:
        pytest.skip(f"mamba3 metal kernel unavailable: {status.reason}")

    y_metal, h_last_metal = _mamba3_module.mamba3_mimo_fwd_metal(
        x, Bp, Cp, z, A, dt, D, h0
    )
    y_ref, h_last_ref = _mamba3_module.mamba3_mimo_reference(
        x, Bp, Cp, z, A, dt, D, h0
    )

    mx.eval(y_metal, h_last_metal, y_ref, h_last_ref)

    diff_y = mx.abs(y_metal.astype(mx.float32) - y_ref.astype(mx.float32))
    diff_h = mx.abs(h_last_metal.astype(mx.float32) - h_last_ref.astype(mx.float32))
    assert float(mx.max(diff_y)) < 5e-2, (
        f"y diff too large: {float(mx.max(diff_y))}"
    )
    assert float(mx.max(diff_h)) < 5e-2, (
        f"h_last diff too large: {float(mx.max(diff_h))}"
    )


def test_mamba3_fwd_engine_path_pending_wave6(_mamba3_module):
    """Wave-6 sentinel: skips until ``mamba3_mimo_fwd_prim`` is exported.

    Once wave-6 lands the ``@T.prim_func`` factory + ``dispatch_lower`` flip,
    this test will compare the engine path's numerical output against the
    Path-B kernel exercised in ``test_mamba3_fwd_metal_matches_reference``.
    """

    fwd_prim = getattr(_mamba3_module, "mamba3_mimo_fwd_prim", None)
    if fwd_prim is None:
        pytest.skip(
            "wave-6: mamba3_mimo_fwd_prim not yet exported "
            "(see _tilelang/mamba3.py module docstring)"
        )

    import mlx.core as mx
    from cppmega_mlx.nn._tilelang import _engine_dispatch as ed

    if not getattr(ed, "dispatch_lower_supports_msl_extraction", lambda: False)():
        pytest.skip("MSL-extraction adapter not active in this build")

    x, Bp, Cp, z, A, dt, D, h0 = _make_inputs()
    y_pathb, h_last_pathb = _mamba3_module.mamba3_mimo_fwd_metal(
        x, Bp, Cp, z, A, dt, D, h0
    )
    # Engine path under the wave-6 hookup. The exact entry-point name will be
    # frozen when wave-6 lands; for now this is a best-effort guard.
    y_engine, h_last_engine = _mamba3_module.mamba3_mimo_fwd_metal_via_engine(
        x, Bp, Cp, z, A, dt, D, h0
    )

    mx.eval(y_pathb, h_last_pathb, y_engine, h_last_engine)
    assert float(mx.max(mx.abs(y_pathb.astype(mx.float32) - y_engine.astype(mx.float32)))) < 1e-4
    assert float(mx.max(mx.abs(h_last_pathb.astype(mx.float32) - h_last_engine.astype(mx.float32)))) < 1e-4
