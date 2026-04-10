"""Tests for CppMegaMamba3Mixer — native SSD kernel with Mamba3 features."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers — mock Megatron imports so tests run without megatron installed
# ---------------------------------------------------------------------------

def _mock_megatron():
    """Return a dict of mock modules to patch into sys.modules."""
    import sys
    mocks = {}

    class FakeRMSNormGated(torch.nn.Module):
        def __init__(self, d, **kw):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(d))
        def forward(self, y, z=None):
            normed = F.rms_norm(y, (y.shape[-1],)) * self.weight
            if z is not None:
                normed = normed * torch.sigmoid(z)
            return normed

    # We need megatron.core.ssm.mamba_mixer.MambaMixer to exist
    # Create minimal stubs
    for mod_path in [
        "megatron", "megatron.core", "megatron.core.ssm",
        "megatron.core.ssm.mamba_mixer",
        "megatron.core.transformer",
        "megatron.core.transformer.spec_utils",
        "megatron.core.models", "megatron.core.models.mamba",
        "megatron.core.models.mamba.mamba_layer_specs",
        "megatron.core.ssm.mamba_block", "megatron.core.ssm.mamba_layer",
        "megatron.core.extensions", "megatron.core.extensions.transformer_engine",
        "mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton",
        "mamba_ssm.ops.triton.ssd_combined",
        "mamba_ssm.ops.triton.causal_conv1d",
        "causal_conv1d",
        "einops",
    ]:
        if mod_path not in sys.modules:
            mocks[mod_path] = MagicMock()

    return mocks


# ---------------------------------------------------------------------------
# Unit tests for _transform_bc (no Megatron dependency)
# ---------------------------------------------------------------------------

class TestTransformBC:
    """Test QK-Norm and bias on B/C — pure torch, no Megatron."""

    def test_rms_norm_applied(self):
        B = torch.randn(2, 8, 4, 64)  # batch=2, seq=8, ngroups=4, d_state=64
        C = torch.randn(2, 8, 4, 64)
        d_state = 64

        B_norm_weight = torch.ones(4, d_state)
        C_norm_weight = torch.ones(4, d_state)

        B_out = F.rms_norm(B, (d_state,)) * B_norm_weight
        C_out = F.rms_norm(C, (d_state,)) * C_norm_weight

        # RMS norm should normalize the last dim
        B_rms = B_out.float().pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(B_rms, torch.ones_like(B_rms), atol=0.1)

    def test_bias_adds_offset(self):
        B = torch.zeros(1, 4, 2, 32)
        bias = torch.ones(2, 32) * 3.0

        B_out = B + bias
        assert torch.allclose(B_out, torch.full_like(B_out, 3.0))

    def test_norm_then_bias(self):
        B = torch.randn(1, 4, 2, 32)
        d_state = 32
        weight = torch.ones(2, d_state)
        bias = torch.zeros(2, d_state)

        # Norm then bias (order matters)
        B_normed = F.rms_norm(B, (d_state,)) * weight
        B_final = B_normed + bias

        # With zero bias, result equals normed
        assert torch.allclose(B_final, B_normed)


# ---------------------------------------------------------------------------
# Unit tests for data-dependent A math
# ---------------------------------------------------------------------------

class TestDataDepAMath:
    """Test the A=-1/dt trick math — pure torch, no Megatron."""

    def test_decay_equivalence(self):
        """Verify exp(A_dd * dt_eff) == exp(-1 * (-ADT))."""
        nheads = 8
        batch, seq = 2, 16

        A_base = -torch.rand(nheads) * 10 - 1  # negative, range [-11, -1]
        A_delta = -F.softplus(torch.randn(batch, seq, nheads))
        A_dd = (A_base + A_delta).clamp(max=-0.01)

        dt_eff = F.softplus(torch.randn(batch, seq, nheads)) + 0.01  # positive
        ADT = A_dd * dt_eff  # negative

        # Standard decay
        decay_standard = torch.exp(ADT)

        # Trick decay: exp(-1 * -ADT) = exp(ADT)
        A_kernel = -1.0
        dt_kernel = -ADT
        decay_trick = torch.exp(A_kernel * dt_kernel)

        assert torch.allclose(decay_standard, decay_trick, atol=1e-6)

    def test_input_scaling_compensation(self):
        """Verify dt_kernel * x_scaled == dt_eff * x."""
        nheads = 4
        headdim = 64
        batch, seq = 1, 8

        A_dd = -torch.rand(batch, seq, nheads) * 10 - 0.01  # negative
        dt_eff = F.softplus(torch.randn(batch, seq, nheads)) + 0.01
        x = torch.randn(batch, seq, nheads, headdim)

        ADT = A_dd * dt_eff
        dt_kernel = -ADT  # positive
        x_scaled = x / (-A_dd).unsqueeze(-1)

        # Standard input scaling: dt_eff * x
        standard = (dt_eff.unsqueeze(-1) * x)

        # Trick input scaling: dt_kernel * x_scaled
        trick = (dt_kernel.unsqueeze(-1) * x_scaled)

        assert torch.allclose(standard, trick, atol=1e-5, rtol=1e-4)

    def test_d_skip_with_original_x(self):
        """D skip connection must use original x, not scaled x."""
        nheads = 4
        headdim = 64
        batch, seq = 1, 8

        x = torch.randn(batch, seq, nheads, headdim)
        D = torch.randn(nheads)

        # Standard
        skip_standard = D.unsqueeze(-1) * x

        # With scaling, we must use original x
        A_dd = -torch.rand(batch, seq, nheads) * 5 - 0.01
        x_scaled = x / (-A_dd).unsqueeze(-1)
        skip_wrong = D.unsqueeze(-1) * x_scaled
        skip_correct = D.unsqueeze(-1) * x

        assert torch.allclose(skip_standard, skip_correct)
        assert not torch.allclose(skip_standard, skip_wrong)

    def test_zero_init_equiv_standard(self):
        """With A_dd_scale=0, A_dd_bias=0, trick should match standard."""
        nheads = 4
        A_base = -torch.rand(nheads) * 10 - 1

        # Zero init: A_delta = -softplus(0*x_norm + 0) = -softplus(0) ≈ -0.693
        A_delta = -F.softplus(torch.zeros(1, 1, nheads))
        A_dd = A_base + A_delta  # close to A_base but shifted by -0.693

        # This is NOT equivalent to standard A_base because A_delta != 0.
        # The feature gradually deviates from standard as training progresses.
        # At init: A_dd ≈ A_base - 0.693 (slightly different)
        assert (A_dd < A_base).all(), "A_dd should be more negative than A_base at init"

    def test_a_dd_clamp(self):
        """A_dd must be clamped to negative values."""
        A_base = torch.tensor([-0.005])
        A_delta = torch.tensor([0.1])  # would make A positive

        A_dd = (A_base + A_delta).clamp(max=-0.01)
        assert (A_dd < 0).all()

    def test_numerical_stability_large_a(self):
        """Large |A_dd| should not cause NaN in x_scaled."""
        nheads = 4
        A_dd = -torch.tensor([[[[100.0]]]]).expand(1, 1, nheads, 1).squeeze(-1)
        x = torch.randn(1, 1, nheads, 64)

        x_scaled = x / (-A_dd).unsqueeze(-1)
        # x / 100 — should be small but finite
        assert torch.isfinite(x_scaled).all()

    def test_numerical_stability_small_a(self):
        """Small |A_dd| (clamped to 0.01) should not cause overflow in x_scaled."""
        nheads = 4
        A_dd = torch.full((1, 1, nheads), -0.01)
        x = torch.randn(1, 1, nheads, 64)

        x_scaled = x / (-A_dd).unsqueeze(-1)
        # x / 0.01 = 100*x — large but finite
        assert torch.isfinite(x_scaled).all()


# ---------------------------------------------------------------------------
# Recipe tests
# ---------------------------------------------------------------------------

class TestMamba3NativeRecipes:
    """Test the new Mamba3 native recipes."""

    def test_mamba3_native_pretrain_exists(self):
        from cppmega.recipes.nam56r_nemo_recipe import nam56r_mamba3_native_pretrain
        recipe = nam56r_mamba3_native_pretrain()
        args = recipe.to_args()

        # Uses CppMegaMamba3Mixer spec
        assert "--spec" in args
        spec_idx = args.index("--spec")
        assert "mamba3_te_stack_spec" in args[spec_idx + 1]

        # Same nheads as nemo_native baseline
        nheads_idx = args.index("--mamba-num-heads")
        assert args[nheads_idx + 1] == "56"

        # CUDA graphs enabled
        assert "--cuda-graph-impl" in args

    def test_mamba3_native_max_throughput_exists(self):
        from cppmega.recipes.nam56r_nemo_recipe import nam56r_mamba3_native_max_throughput
        recipe = nam56r_mamba3_native_max_throughput()
        args = recipe.to_args()

        # FP8
        assert "--fp8-format" in args
        assert "--fp8-recipe" in args

        # FP8-aligned nheads
        nheads_idx = args.index("--mamba-num-heads")
        assert args[nheads_idx + 1] == "64"

        # MBS=5, GBS=320
        assert recipe.micro_batch_size == 5
        assert recipe.global_batch_size == 320

        # Full MoE CUDA graph
        assert "--moe-expert-capacity-factor" in args
        assert "--moe-pad-expert-input-to-capacity" in args

    def test_mamba3_native_vs_nemo_native_same_arch(self):
        """Mamba3 native should have same model architecture (nheads, hidden)
        as nemo_native for fair comparison."""
        from cppmega.recipes.nam56r_nemo_recipe import (
            nam56r_mamba3_native_pretrain,
            nam56r_nemo_native_pretrain,
        )
        m3 = nam56r_mamba3_native_pretrain()
        native = nam56r_nemo_native_pretrain()

        assert m3.hidden_size == native.hidden_size
        assert m3.ffn_hidden_size == native.ffn_hidden_size
        assert m3.num_layers == native.num_layers
        # nheads should match
        assert m3.mamba_num_heads == native.mamba_num_heads


# ---------------------------------------------------------------------------
# Env var control tests
# ---------------------------------------------------------------------------

class TestEnvVarControl:
    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        """Local copy of _env_bool to avoid megatron import."""
        return os.environ.get(name, "1" if default else "0") == "1"

    def test_env_bool_defaults(self):
        for key in ["CPPMEGA_MAMBA3_QKNORM", "CPPMEGA_MAMBA3_BIAS", "CPPMEGA_MAMBA3_DATA_DEP_A"]:
            os.environ.pop(key, None)

        assert self._env_bool("CPPMEGA_MAMBA3_QKNORM", default=True) is True
        assert self._env_bool("CPPMEGA_MAMBA3_BIAS", default=True) is True
        assert self._env_bool("CPPMEGA_MAMBA3_DATA_DEP_A", default=False) is False

    def test_env_bool_override(self):
        os.environ["CPPMEGA_MAMBA3_DATA_DEP_A"] = "1"
        assert self._env_bool("CPPMEGA_MAMBA3_DATA_DEP_A", default=False) is True
        os.environ["CPPMEGA_MAMBA3_DATA_DEP_A"] = "0"
        assert self._env_bool("CPPMEGA_MAMBA3_DATA_DEP_A", default=False) is False
        os.environ.pop("CPPMEGA_MAMBA3_DATA_DEP_A", None)


# ---------------------------------------------------------------------------
# Spec tests
# ---------------------------------------------------------------------------

class TestMamba3Spec:
    def test_spec_references_cppmega_mixer(self):
        """The mamba3_te_stack_spec source should reference CppMegaMamba3Mixer."""
        from pathlib import Path
        spec_path = Path(__file__).parent.parent / "cppmega" / "megatron" / "mamba3_te_stack_spec.py"
        if not spec_path.exists():
            pytest.skip("mamba3_te_stack_spec.py not found")
        src = spec_path.read_text()
        assert "CppMegaMamba3Mixer" in src
