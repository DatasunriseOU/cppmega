"""Wave-6 deferral contract for the m2rnn.py Path-B engine flip.

The wave-6 caller-flip pass (cppmega.mlx commit ``af33938``) marked both
``_FWD_KERNEL`` and ``_BWD_KERNEL`` of ``cppmega_mlx.nn._tilelang.m2rnn``
as wave-7 TODOs: the kernels are hand-written MSL constructed via
``_msl_transform.make_metal_kernel`` (no ``@T.prim_func`` to feed
``dispatch_lower``), so the MSL-extraction adapter (commit ``00d6d90``)
goes the wrong direction for them. A full TileLang DSL rewrite of both
the forward scan and the backward two-pass walk is the wave-7 work.

This file is the regression gate that locks in the wave-6 deferral
contract. It mirrors the shape of ``test_mamba3_path_b_engine.py`` and
asserts:

* ``cppmega_mlx.nn._tilelang.m2rnn`` imports cleanly when cppmega.mlx is
  installed (otherwise ``importorskip`` keeps CI green).
* The wave-7 deferral marker survives in the source — both
  ``_FWD_KERNEL`` and ``_BWD_KERNEL`` blocks must continue to carry a
  ``TODO(wave-7)`` (or ``TODO(wave-6)``) marker so a future flip cannot
  silently land without updating the contract here.
* The public entry-points ``m2rnn_metal_status``, ``m2rnn_fwd_metal``,
  ``m2rnn_bwd_metal``, ``m2rnn_apply``, ``m2rnn_apply_with_state``,
  ``m2rnn_reference`` are callables under each of the four
  ``CPPMEGA_MLX_TILELANG_ENGINE`` env modes (``auto`` / ``engine`` /
  ``shim`` / ``engine_with_msl_extraction``).
* The Metal forward kernel actually launches when Metal is available —
  the wave-6 deferral must NOT regress the existing Path-B contract.
"""

from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path

import pytest


_ENGINE_MODES = ("auto", "engine", "shim", "engine_with_msl_extraction")
_PUBLIC_ENTRY_POINTS = (
    "m2rnn_metal_status",
    "m2rnn_fwd_metal",
    "m2rnn_bwd_metal",
    "m2rnn_apply",
    "m2rnn_apply_with_state",
    "m2rnn_reference",
)


def _metal_available() -> bool:
    try:
        import mlx.core as mx
    except Exception:
        return False
    metal = getattr(mx, "metal", None)
    is_avail = getattr(metal, "is_available", None)
    return bool(is_avail()) if callable(is_avail) else False


@pytest.fixture(scope="module")
def _m2rnn_module():
    return pytest.importorskip("cppmega_mlx.nn._tilelang.m2rnn")


@pytest.fixture(scope="module")
def _m2rnn_source(_m2rnn_module) -> str:
    src_path = inspect.getsourcefile(_m2rnn_module)
    assert src_path is not None, "m2rnn module has no source file"
    return Path(src_path).read_text(encoding="utf-8")


def test_m2rnn_module_imports_cleanly(_m2rnn_module):
    """The Path-B module must remain importable on every host."""

    assert _m2rnn_module is not None


@pytest.mark.parametrize("entry", _PUBLIC_ENTRY_POINTS)
def test_m2rnn_public_entry_points_are_callable(_m2rnn_module, entry):
    """Public surface (per af33938 module docstring) must stay callable."""

    fn = getattr(_m2rnn_module, entry, None)
    assert fn is not None, f"missing public entry point: {entry}"
    assert callable(fn), f"{entry} is not callable: {type(fn).__name__}"


@pytest.mark.parametrize("kernel_block", ("_FWD_KERNEL", "_BWD_KERNEL"))
def test_wave6_deferral_marker_survives(_m2rnn_source, kernel_block):
    """Both kernels must still carry an explicit wave-7 / wave-6 TODO.

    A future flip MUST update this assertion (and its docstring) at the
    same time as it lands the prim_func factories, so the contract here
    cannot silently rot out from under us.
    """

    assert kernel_block in _m2rnn_source, (
        f"expected {kernel_block} block in m2rnn.py source"
    )
    # Locate the block and check the surrounding region for the marker.
    block_index = _m2rnn_source.index(kernel_block)
    window = _m2rnn_source[max(0, block_index - 800):block_index + 800]
    has_marker = ("TODO(wave-7)" in window) or ("TODO(wave-6)" in window)
    assert has_marker, (
        f"wave-6 deferral marker missing near {kernel_block} "
        "(expected TODO(wave-7) or TODO(wave-6) in the surrounding "
        "comment block per cppmega.mlx commit af33938)"
    )


@pytest.mark.parametrize("mode", _ENGINE_MODES)
def test_engine_mode_does_not_break_module_import(_m2rnn_module, mode):
    """Setting CPPMEGA_MLX_TILELANG_ENGINE must not break m2rnn import.

    The wave-6 deferral path stays on raw MSL irrespective of mode; we
    just need the module to keep loading and exposing its surface.
    """

    previous = os.environ.get("CPPMEGA_MLX_TILELANG_ENGINE")
    try:
        os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = mode
        mod = importlib.reload(_m2rnn_module)
        for entry in _PUBLIC_ENTRY_POINTS:
            assert callable(getattr(mod, entry, None)), (
                f"entry point {entry} not callable under "
                f"CPPMEGA_MLX_TILELANG_ENGINE={mode}"
            )
    finally:
        if previous is None:
            os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
        else:
            os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = previous


def test_m2rnn_fwd_metal_launches_when_metal_available(_m2rnn_module):
    """Sanity: the existing Path-B forward kernel still launches.

    We do not re-validate numerical parity here (that is the parent test
    in ``test_mamba3_path_b_engine.py`` for the sister kernel and a
    deferred wave-7 follow-up for m2rnn). This test only locks in that
    the wave-6 deferral marker has not regressed the launch contract.
    """

    if not _metal_available():
        pytest.skip("Apple Metal not available on this host")

    import mlx.core as mx

    status = _m2rnn_module.m2rnn_metal_status()
    if not status.available:
        pytest.skip(f"m2rnn Metal kernel unavailable: {status.reason}")

    B, S, H, K, V = 1, 4, 1, 4, 4
    rng = mx.random.key(0xBEEF)
    keys = mx.random.split(rng, 5)
    q = mx.random.normal(shape=(B, S, H, K), key=keys[0]).astype(mx.float16)
    k = mx.random.normal(shape=(B, S, H, K), key=keys[1]).astype(mx.float16)
    v = mx.random.normal(shape=(B, S, H, V), key=keys[2]).astype(mx.float16)
    W = mx.random.normal(shape=(H, V, V), key=keys[3]).astype(mx.float16) * 0.1
    xf = mx.sigmoid(
        mx.random.normal(shape=(B, S, H), key=keys[4]).astype(mx.float16)
    )
    h0 = mx.zeros((B, H, K, V), dtype=mx.float16)

    result = _m2rnn_module.m2rnn_fwd_metal(q, k, v, W, xf, h0)

    mx.eval(*result if isinstance(result, tuple) else (result,))
