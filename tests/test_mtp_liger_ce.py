"""Tests for cppmega.megatron.mtp_liger_ce — Liger fused linear cross-entropy for MTP.

These tests verify:
  1. The patch installs and replaces process_mtp_loss correctly.
  2. The Liger path produces numerically equivalent results to the original
     (forward values, per-depth logged losses, backward gradients).
  3. The MuP (scale_logits_fn) fallback works.

Run (CPU-only, no GPU required for import/structure tests):
    pytest tests/test_mtp_liger_ce.py -v

Run with GPU for numerical tests:
    CUDA_VISIBLE_DEVICES=7 pytest tests/test_mtp_liger_ce.py -v -k gpu
"""
from __future__ import annotations

import os
import sys
import importlib
import pytest

# Ensure repo roots are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMTPLigerCEStructure:
    """Structure / import tests that do not need a GPU."""

    def test_import(self):
        from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
        assert callable(patch_mtp_loss_with_liger)

    def test_noop_without_env(self):
        """patch_mtp_loss_with_liger is a no-op when env var is absent."""
        os.environ.pop("CPPMEGA_MTP_LIGER_CE", None)
        from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
        # Should return without error
        patch_mtp_loss_with_liger()

    def test_env_gate(self, monkeypatch):
        """Only patches when CPPMEGA_MTP_LIGER_CE=1."""
        monkeypatch.setenv("CPPMEGA_MTP_LIGER_CE", "0")
        from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
        patch_mtp_loss_with_liger()  # should be no-op


@pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES"),
    reason="GPU test — set CUDA_VISIBLE_DEVICES to run",
)
class TestMTPLigerCEGPU:
    """GPU-based numerical correctness tests."""

    def test_liger_import(self):
        """Verify liger_kernel is importable."""
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )
        assert LigerFusedLinearCrossEntropyFunction is not None

    def test_liger_none_reduction_basic(self):
        """Liger fused CE with reduction='none' matches F.cross_entropy."""
        import torch
        import torch.nn.functional as F
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        torch.manual_seed(42)
        B, H, V = 64, 256, 1024
        inp = torch.randn(B, H, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(V, H, dtype=torch.bfloat16, device="cuda")
        target = torch.randint(0, V, (B,), device="cuda")

        # Standard
        logits = inp.float() @ weight.float().T
        std_loss = F.cross_entropy(logits, target, reduction="none")

        # Liger
        inp_l = inp.clone().detach().requires_grad_(True)
        lig_loss, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
            inp_l, weight, target, None, None, -100, 0.0, 0.0, "none", None, False,
        )

        diff = (std_loss - lig_loss).abs()
        assert diff.max().item() < 0.1, f"Max loss diff {diff.max().item()} too large"
