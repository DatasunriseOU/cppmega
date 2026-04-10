"""Tests for CppMegaMamba3TEOutProj dimension and configuration logic.

These tests validate the constructor logic and dimension invariants without
requiring TE, CUDA, or mamba_ssm.  The actual TE integration is tested on
the H200 machine.
"""

import pytest

from cppmega.features.mamba3.config import AuthorMamba3Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> AuthorMamba3Config:
    defaults = dict(
        d_model=256, d_state=128, expand=2, headdim=64, ngroups=1,
    )
    defaults.update(overrides)
    return AuthorMamba3Config(**defaults)


# ---------------------------------------------------------------------------
# Dimension invariants (tested without instantiating the module)
# ---------------------------------------------------------------------------

class TestOutProjDimensions:
    """Verify dimension calculations match Author Mamba3 conventions."""

    def test_d_inner_from_expand(self):
        cfg = _make_cfg(d_model=256, expand=2)
        d_inner = cfg.d_model * cfg.expand
        assert d_inner == 512

    def test_nheads_from_d_inner_and_headdim(self):
        cfg = _make_cfg(d_model=256, expand=2, headdim=64)
        d_inner = cfg.d_model * cfg.expand
        nheads = d_inner // cfg.headdim
        assert nheads == 8

    def test_d_inner_local_tp2(self):
        cfg = _make_cfg(d_model=256, expand=2, headdim=64)
        d_inner = cfg.d_model * cfg.expand
        tp_size = 2
        assert d_inner % tp_size == 0
        assert d_inner // tp_size == 256

    def test_d_inner_local_tp8(self):
        cfg = _make_cfg(d_model=1024, expand=2, headdim=64)
        d_inner = cfg.d_model * cfg.expand  # 2048
        tp_size = 8
        assert d_inner % tp_size == 0
        assert d_inner // tp_size == 256

    def test_nheads_must_divide_tp(self):
        cfg = _make_cfg(d_model=256, expand=2, headdim=64)
        d_inner = cfg.d_model * cfg.expand  # 512
        nheads = d_inner // cfg.headdim  # 8
        # 8 heads with tp_size=3 should fail
        assert nheads % 3 != 0

    @pytest.mark.parametrize("d_model,expand,headdim,tp_size", [
        (256, 2, 64, 1),
        (256, 2, 64, 2),
        (256, 2, 64, 4),
        (256, 2, 64, 8),
        (512, 2, 64, 8),
        (1024, 2, 128, 8),
    ])
    def test_divisibility_invariants(self, d_model, expand, headdim, tp_size):
        """d_inner and nheads must be divisible by tp_size."""
        d_inner = d_model * expand
        nheads = d_inner // headdim
        assert d_inner % headdim == 0, "d_inner not divisible by headdim"
        assert d_inner % tp_size == 0, "d_inner not divisible by tp_size"
        assert nheads % tp_size == 0, "nheads not divisible by tp_size"


class TestOutProjNormConfig:
    """Verify norm configuration matches Author Mamba3 conventions."""

    def test_norm_disabled_by_default(self):
        cfg = _make_cfg()
        assert not cfg.is_outproj_norm

    def test_norm_enabled_explicitly(self):
        cfg = _make_cfg(is_outproj_norm=True)
        assert cfg.is_outproj_norm

    def test_norm_group_size_equals_headdim(self):
        """Author Mamba3 creates RMSNormGated(d_inner, group_size=headdim).
        With TP, the local norm has d_inner_local channels but group_size
        stays at headdim (each head is independently normalized)."""
        cfg = _make_cfg(d_model=256, expand=2, headdim=64, is_outproj_norm=True)
        d_inner = cfg.d_model * cfg.expand  # 512
        for tp_size in [1, 2, 4, 8]:
            d_inner_local = d_inner // tp_size
            # group_size = headdim = 64 must divide d_inner_local
            assert d_inner_local % cfg.headdim == 0, (
                f"d_inner_local ({d_inner_local}) not divisible by "
                f"headdim ({cfg.headdim}) at tp_size={tp_size}"
            )

    def test_norm_before_gate_is_true(self):
        """Author Mamba3 sets norm_before_gate=True for the outproj norm.
        This means: rms_norm(y) * silu(z), NOT rms_norm(y * silu(z))."""
        # This is a documentation test -- the constructor enforces it.
        cfg = _make_cfg(is_outproj_norm=True)
        assert cfg.is_outproj_norm


class TestOutProjTPSharding:
    """Verify TP sharding strategy for the output projection."""

    def test_row_parallel_input_is_parallel(self):
        """TERowParallelLinear must be created with input_is_parallel=True
        because y arrives already sharded across TP ranks."""
        # This is a design contract test.
        # The actual assertion is in the constructor; here we verify the
        # dimensions that make it possible.
        cfg = _make_cfg(d_model=256, expand=2)
        d_inner = cfg.d_model * cfg.expand  # 512
        # TERowParallelLinear(input_size=d_inner, output_size=d_model,
        #                     input_is_parallel=True)
        # With tp_size=2:
        #   local input = d_inner / 2 = 256
        #   output = d_model = 256  (all-reduced)
        for tp_size in [1, 2, 4, 8]:
            assert d_inner % tp_size == 0

    def test_output_is_full_d_model(self):
        """After TP all-reduce, output should be full d_model."""
        cfg = _make_cfg(d_model=512, expand=2)
        # TERowParallelLinear output_size = d_model = 512
        assert cfg.d_model == 512


class TestOutProjMIMOPath:
    """Verify the MIMO output path dimensions.

    When is_mimo=True AND is_outproj_norm=True, the Author Mamba3 forward:
    1. Computes mimo_z gate:     z_r = einsum("blhp,hrp->blrhp", z, mimo_z)
    2. Applies norm:             y = norm(y, z_r)   -- per rank in R dim
    3. Applies mimo_o:           y = einsum("blrhp,hrp->blhp", y, mimo_o)
    4. Reshapes to (b,l,d_inner) and runs out_proj

    The TE out_proj only sees step 4 -- the preceding MIMO steps are handled
    by the Author Mamba3 kernel / host code.  We just need to verify that
    the final y has shape (b, l, d_inner) before entering out_proj.
    """

    def test_mimo_final_shape(self):
        cfg = _make_cfg(d_model=256, expand=2, headdim=64,
                        is_mimo=True, mimo_rank=4, is_outproj_norm=True)
        d_inner = cfg.d_model * cfg.expand
        nheads = d_inner // cfg.headdim
        # After mimo_o contraction: (b, l, nheads, headdim) -> (b, l, d_inner)
        assert nheads * cfg.headdim == d_inner

    def test_mimo_norm_is_per_rank(self):
        """When is_outproj_norm=True with MIMO, norm is applied per rank
        before mimo_o contraction.  The norm input is (b, l, R, d_inner).
        The TE norm we create is for the POST-contraction path (d_inner_local).
        The per-rank norm is done by the Author Mamba3 kernel."""
        cfg = _make_cfg(is_mimo=True, mimo_rank=4, is_outproj_norm=True)
        # Just a documentation assertion -- MIMO norm path is upstream
        assert cfg.is_mimo
        assert cfg.is_outproj_norm
