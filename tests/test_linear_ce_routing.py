"""Tests for cppmega LinearCE routing helpers."""

from __future__ import annotations

import pytest

from cppmega.megatron.apply_linear_ce_patch import _cce_filter_eps_from_env


@pytest.mark.parametrize("value", ["", "none", "off", "false", "0"])
def test_cce_filter_eps_default_exact(monkeypatch, value):
    monkeypatch.setenv("CPPMEGA_CCE_FILTER_EPS", value)

    assert _cce_filter_eps_from_env() is None


@pytest.mark.parametrize("value", ["auto", "high"])
def test_cce_filter_eps_named_modes(monkeypatch, value):
    monkeypatch.setenv("CPPMEGA_CCE_FILTER_EPS", value)

    assert _cce_filter_eps_from_env() == value


def test_cce_filter_eps_float(monkeypatch):
    monkeypatch.setenv("CPPMEGA_CCE_FILTER_EPS", "0.0001")

    assert _cce_filter_eps_from_env() == pytest.approx(0.0001)


def test_cce_filter_eps_rejects_invalid(monkeypatch):
    monkeypatch.setenv("CPPMEGA_CCE_FILTER_EPS", "fast")

    with pytest.raises(ValueError, match="CPPMEGA_CCE_FILTER_EPS"):
        _cce_filter_eps_from_env()
