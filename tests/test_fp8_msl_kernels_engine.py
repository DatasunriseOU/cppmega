"""Wave-6 audit test for ``fp8_msl_kernels``.

This module is currently NOT engine-flipped (see the module docstring).
The wave-6 audit reclassified the four kernels here as LUT-based fp32-fma
shaders that don't actually need FP8 SIMDgroup factories. Until wave-7
lands ``tl.extern_intrinsic`` on the Metal target plus a constant-table
extern in ``codegen_metal.cc``, callers stay on ``mx.fast.metal_kernel``.

This test asserts:

* the per-kernel classification comments survive in source (regression
  detector for accidental wave-7 partial flips);
* ``_wave7_engine_flip_blocked_reason()`` returns a non-empty string with
  the LUT keyword (so any wave-7 commit that lifts the blocker has to
  update or remove this helper);
* ``fp8_msl_status()`` still works under both engine modes (env-flag
  dispatch is read but not honoured by this module yet -- regression
  test that toggling the env doesn't crash imports).

Numerical parity between engine and shim is **not** asserted: this
module has no engine path yet.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from contextlib import contextmanager
from typing import Iterator

import pytest


_pkg = "cppmega_mlx.nn._tilelang.fp8_msl_kernels"
pytest.importorskip(_pkg)


@contextmanager
def _engine_mode(mode: str | None) -> Iterator[None]:
    prev = os.environ.get("CPPMEGA_MLX_TILELANG_ENGINE")
    if mode is None:
        os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
    else:
        os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = mode
    for name in (
        "cppmega_mlx.nn._tilelang._engine_dispatch",
        "cppmega_mlx.nn._tilelang.fp8_msl_kernels",
    ):
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("CPPMEGA_MLX_TILELANG_ENGINE", None)
        else:
            os.environ["CPPMEGA_MLX_TILELANG_ENGINE"] = prev


def test_wave7_blocked_reason_mentions_lut_and_extern_intrinsic() -> None:
    """The blocked-reason helper must call out the real wave-7 prerequisites.

    If wave-7 lands and this assertion fails, the helper has been removed or
    the wording changed -- either is fine, but the failure forces the wave-7
    commit author to also update the module docstring TODO and any callers
    that branch on the helper.
    """

    mod = importlib.import_module(_pkg)
    reason = mod._wave7_engine_flip_blocked_reason()
    assert isinstance(reason, str) and reason
    # The two real blockers must be named explicitly.
    assert "extern_intrinsic" in reason.lower()
    assert "lut" in reason.lower() or "constant-table" in reason.lower()
    # The FP8-factory red herring must be debunked, not propagated.
    assert "do not need" in reason.lower() or "do NOT need" in reason


def test_per_kernel_classification_comments_present() -> None:
    """Source-level guard: the wave-6 classification block must survive."""

    import inspect

    mod = importlib.import_module(_pkg)
    src = inspect.getsource(mod)
    # Each of the four kernels must be classified.
    classifications = re.findall(
        r"_FP8_(?:TO_HALF|MATMUL|VECMAT)_KERNEL\s*:\s*fp8-bordered", src
    )
    assert len(classifications) >= 3, classifications
    assert "_HALF_TO_FP8_KERNEL      : fp8-pure" in src


def test_fp8_msl_status_survives_env_toggling() -> None:
    """``fp8_msl_status()`` must not crash under any ``CPPMEGA_MLX_TILELANG_ENGINE`` value.

    The env flag is read by ``_engine_dispatch`` and by sibling Path-C
    modules, but ``fp8_msl_kernels`` itself does not branch on it yet.
    Verify that still holds (no accidental wave-7 partial flip).
    """

    for mode in (None, "auto", "shim", "engine", "engine_with_msl_extraction"):
        with _engine_mode(mode):
            mod = importlib.import_module(_pkg)
            status = mod.fp8_msl_status()
            assert hasattr(status, "available")
            assert hasattr(status, "reason")
            # When MLX Metal isn't on the host, status should be unavailable
            # with a descriptive reason. When Metal IS available, status may
            # be either available=True or available=False depending on whether
            # mx.fast.metal_kernel compiled successfully.
            assert isinstance(status.available, bool)
            assert isinstance(status.reason, str) and status.reason


def test_module_docstring_acknowledges_wave6_audit() -> None:
    """The module docstring TODO must reflect the wave-6 audit, not the
    pre-audit FP8-factory blocker that turned out to be a red herring."""

    mod = importlib.import_module(_pkg)
    assert mod.__doc__ is not None
    doc = mod.__doc__.lower()
    # New (post-audit) language:
    assert "wave-6" in doc or "wave-7" in doc
    assert "lut" in doc
    # The originally-suspected blocker must be explicitly debunked.
    assert "not what gates" in doc or "not actually a blocker" in doc
