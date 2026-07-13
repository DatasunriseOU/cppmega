"""Tests for FastMTP layer implementation."""

from __future__ import annotations

import sys

import pytest
import torch

# Stub out megatron imports for local testing.  Uses the shared helper so
# the ``__spec__`` handling stays consistent across the test suite.
from tests._megatron_stub import install_megatron_stub

install_megatron_stub()

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
        reference = torch.nn.functional.cross_entropy(
            torch.nn.functional.linear(hidden, weight).float(),
            targets,
            ignore_index=-1,
            reduction="mean",
        )
        assert loss.shape == ()
        assert torch.allclose(loss, reference)

    def test_liger_path_never_uses_unsafe_none_reduction(self, monkeypatch):
        from cppmega.megatron import fastmtp_layer

        seen = {}

        class FakeLiger:
            @staticmethod
            def apply(*args):
                seen["reduction"] = args[8]
                return args[0].sum() * 0.0

        monkeypatch.setenv("CPPMEGA_FASTMTP_USE_LIGER", "1")
        monkeypatch.setattr(fastmtp_layer, "_get_liger_fused_ce", lambda: FakeLiger)
        loss = fastmtp_layer._fused_linear_cross_entropy(
            torch.randn(4, 3),
            torch.randn(8, 3),
            torch.tensor([1, 2, -1, 3]),
        )

        assert loss.shape == ()
        assert seen["reduction"] == "mean"


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
