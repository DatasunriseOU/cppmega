"""Cheap tests for CUTLASS MXFP8 Python-side argument handling."""

from __future__ import annotations

import pytest

from cppmega.megatron.cutlass_mxfp8_gemm import _resolve_beta


def test_resolve_beta_defaults_for_accumulate_and_overwrite() -> None:
    assert _resolve_beta(None, accumulate=True) == 1.0
    assert _resolve_beta(None, accumulate=False) == 0.0


def test_resolve_beta_accepts_explicit_accumulate_values() -> None:
    assert _resolve_beta(0.0, accumulate=True) == 0.0
    assert _resolve_beta(0.25, accumulate=True) == 0.25


def test_resolve_beta_rejects_nonzero_overwrite_beta() -> None:
    assert _resolve_beta(0.0, accumulate=False) == 0.0
    with pytest.raises(ValueError, match="accumulate=True"):
        _resolve_beta(1.0, accumulate=False)
