"""Tests for FastMTP layer implementation."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch

# Stub out megatron imports for local testing
if "megatron" not in sys.modules:
    from importlib.machinery import ModuleSpec

    _megatron = MagicMock()
    # A real ``ModuleSpec`` is required so that downstream tests that call
    # ``importlib.util.find_spec("megatron")`` during their collection phase
    # (Python 3.12+ enforces ``module.__spec__`` be a valid ModuleSpec) do
    # not raise ``ValueError: megatron.__spec__ is not set``.
    for _name in (
        "megatron",
        "megatron.core",
        "megatron.core.tensor_parallel",
        "megatron.core.transformer",
        "megatron.core.transformer.module",
        "megatron.core.transformer.transformer_config",
    ):
        _mod = _megatron
        for _part in _name.split(".")[1:]:
            _mod = getattr(_mod, _part)
        _mod.__spec__ = ModuleSpec(_name, loader=None)
        sys.modules[_name] = _mod
    # Provide a base class for MegatronModule
    _megatron.core.transformer.module.MegatronModule = torch.nn.Module

from cppmega.megatron.fastmtp_layer import (
    _compute_step_weights,
    _fused_linear_cross_entropy,
    _roll_and_mask_ids,
    _roll_and_mask_targets,
    fastmtp_enabled,
    get_fastmtp_decay,
    get_fastmtp_depth,
    get_fastmtp_lambda,
)


class TestRollAndMask:
    def test_roll_and_mask_targets_basic(self):
        x = torch.tensor([[10, 20, 30, 40]])
        result = _roll_and_mask_targets(x)
        assert result.shape == (1, 4)
        assert result[0, 0].item() == 20
        assert result[0, 1].item() == 30
        assert result[0, 2].item() == 40
        assert result[0, 3].item() == -1

    def test_roll_and_mask_ids_basic(self):
        x = torch.tensor([[10, 20, 30, 40]])
        result = _roll_and_mask_ids(x)
        assert result.shape == (1, 4)
        assert result[0, 0].item() == 20
        assert result[0, 1].item() == 30
        assert result[0, 2].item() == 40
        assert result[0, 3].item() == 0

    def test_cumulative_rolls_mask_tail(self):
        """After K cumulative rolls, last K positions should be masked."""
        x = torch.tensor([[1, 2, 3, 4, 5, 6]])
        r1 = _roll_and_mask_targets(x)
        assert r1[0, -1].item() == -1
        r2 = _roll_and_mask_targets(r1)
        assert r2[0, -1].item() == -1
        assert r2[0, -2].item() == -1
        r3 = _roll_and_mask_targets(r2)
        assert r3[0, -1].item() == -1
        assert r3[0, -2].item() == -1
        assert r3[0, -3].item() == -1


class TestStepWeights:
    def test_depth_1(self):
        w = _compute_step_weights(1, 0.6)
        assert len(w) == 1
        assert abs(w[0].item() - 1.0) < 1e-6

    def test_depth_3_sums_to_1(self):
        w = _compute_step_weights(3, 0.6)
        assert len(w) == 3
        assert abs(w.sum().item() - 1.0) < 1e-6

    def test_decay_1_uniform(self):
        w = _compute_step_weights(4, 1.0)
        for i in range(4):
            assert abs(w[i].item() - 0.25) < 1e-6

    def test_decay_ordering(self):
        w = _compute_step_weights(3, 0.5)
        assert w[0] > w[1] > w[2]


class TestFusedLinearCE:
    def test_fallback_path_runs(self):
        """Test the non-Liger fallback path produces valid loss."""
        B, T, D, V = 2, 8, 16, 32
        hidden = torch.randn(B * T, D, dtype=torch.bfloat16)
        weight = torch.randn(V, D, dtype=torch.bfloat16)
        targets = torch.randint(0, V, (B * T,))
        targets[-1] = -1  # one ignored position

        loss = _fused_linear_cross_entropy(hidden, weight, targets, ignore_index=-1)
        assert loss.shape == (B * T,)
        assert not torch.isnan(loss).any()
        # Ignored position should have 0 loss
        assert loss[-1].item() == 0.0


class TestEnvConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("CPPMEGA_FASTMTP", raising=False)
        monkeypatch.delenv("CPPMEGA_FASTMTP_DEPTH", raising=False)
        monkeypatch.delenv("CPPMEGA_FASTMTP_DECAY", raising=False)
        monkeypatch.delenv("CPPMEGA_FASTMTP_LAMBDA", raising=False)
        assert not fastmtp_enabled()
        assert get_fastmtp_depth() == 1
        assert abs(get_fastmtp_decay() - 0.6) < 1e-6
        assert abs(get_fastmtp_lambda() - 0.3) < 1e-6

    def test_enabled(self, monkeypatch):
        monkeypatch.setenv("CPPMEGA_FASTMTP", "1")
        assert fastmtp_enabled()

    def test_custom_values(self, monkeypatch):
        monkeypatch.setenv("CPPMEGA_FASTMTP_DEPTH", "3")
        monkeypatch.setenv("CPPMEGA_FASTMTP_DECAY", "0.8")
        monkeypatch.setenv("CPPMEGA_FASTMTP_LAMBDA", "0.5")
        assert get_fastmtp_depth() == 3
        assert abs(get_fastmtp_decay() - 0.8) < 1e-6
        assert abs(get_fastmtp_lambda() - 0.5) < 1e-6
