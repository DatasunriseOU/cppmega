"""Tests for CppMegaMamba3TEInProj dimension calculations.

These tests validate the dimension math without requiring TE or CUDA.
The actual TE integration is tested on the H200 machine.
"""

import pytest

from cppmega.features.mamba3.config import AuthorMamba3Config
from cppmega.megatron.mamba3_te_in_proj import (
    Mamba3InProjDims,
    _compute_num_rope_angles,
    compute_mamba3_in_proj_dims,
    compute_mamba3_te_output_size,
    compute_mamba3_tp_partition_sizes,
)


# ---------------------------------------------------------------------------
# _compute_num_rope_angles
# ---------------------------------------------------------------------------

def test_rope_angles_half_fraction_even_state():
    # d_state=128, rope_fraction=0.5 -> split=64 -> angles=32
    assert _compute_num_rope_angles(128, 0.5) == 32


def test_rope_angles_full_fraction():
    # d_state=128, rope_fraction=1.0 -> split=128 -> angles=64
    assert _compute_num_rope_angles(128, 1.0) == 64


def test_rope_angles_half_fraction_odd_state():
    # d_state=65, rope_fraction=0.5 -> split=32 (int(32.5)=32, even) -> angles=16
    assert _compute_num_rope_angles(65, 0.5) == 16


def test_rope_angles_odd_split_adjusted():
    # d_state=3, rope_fraction=1.0 -> split=3, odd -> split=2 -> angles=1
    assert _compute_num_rope_angles(3, 1.0) == 1


# ---------------------------------------------------------------------------
# compute_mamba3_in_proj_dims - matches Author Mamba3.__init__ formula
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> AuthorMamba3Config:
    defaults = dict(
        d_model=256, d_state=128, expand=2, headdim=64, ngroups=1,
    )
    defaults.update(overrides)
    return AuthorMamba3Config(**defaults)


def test_dims_default_siso():
    """Validate against Author Mamba3 formula with default SISO config."""
    cfg = _make_cfg()
    dims = compute_mamba3_in_proj_dims(cfg)

    # d_inner = 256*2 = 512
    assert dims.d_inner == 512
    # nheads = 512/64 = 8
    assert dims.nheads == 8
    # SISO: mimo_rank=1, d_bc = 128*1*1 = 128
    assert dims.d_bc == 128
    # rope: split=64, angles=32
    assert dims.num_rope_angles == 32

    # total = 2*512 + 2*128 + 3*8 + 32 = 1024 + 256 + 24 + 32 = 1336
    assert dims.total == 1336


def test_dims_mimo():
    cfg = _make_cfg(is_mimo=True, mimo_rank=4, ngroups=8)
    dims = compute_mamba3_in_proj_dims(cfg)

    assert dims.d_inner == 512
    assert dims.nheads == 8
    # mimo: d_bc = 128*8*4 = 4096
    assert dims.d_bc == 4096
    assert dims.total == 2 * 512 + 2 * 4096 + 3 * 8 + 32


def test_dims_full_rope():
    cfg = _make_cfg(rope_fraction=1.0)
    dims = compute_mamba3_in_proj_dims(cfg)
    assert dims.num_rope_angles == 64
    assert dims.total == 2 * 512 + 2 * 128 + 3 * 8 + 64


def test_split_sizes_match_total():
    cfg = _make_cfg(ngroups=8, is_mimo=True, mimo_rank=4)
    dims = compute_mamba3_in_proj_dims(cfg)
    assert sum(dims.split_sizes) == dims.total


def test_split_sizes_order():
    """Verify the split order matches Author Mamba3 forward: z,x,B,C,dt,A,trap,angles."""
    cfg = _make_cfg()
    dims = compute_mamba3_in_proj_dims(cfg)
    z, x, b, c, dt, a, trap, angles = dims.split_sizes
    assert z == dims.d_inner
    assert x == dims.d_inner
    assert b == dims.d_bc
    assert c == dims.d_bc
    assert dt == dims.nheads
    assert a == dims.nheads
    assert trap == dims.nheads
    assert angles == dims.num_rope_angles


# ---------------------------------------------------------------------------
# TP partition sizes
# ---------------------------------------------------------------------------

def test_tp1_partition_equals_split_sizes():
    cfg = _make_cfg(ngroups=8)
    dims = compute_mamba3_in_proj_dims(cfg)
    parts = compute_mamba3_tp_partition_sizes(dims, tp_size=1)
    assert parts == dims.split_sizes


def test_tp2_partition_sizes():
    cfg = _make_cfg(ngroups=8)
    dims = compute_mamba3_in_proj_dims(cfg)
    parts = compute_mamba3_tp_partition_sizes(dims, tp_size=2)

    assert parts[0] == dims.d_inner // 2   # z
    assert parts[1] == dims.d_inner // 2   # x
    assert parts[2] == dims.d_bc // 2      # B
    assert parts[3] == dims.d_bc // 2      # C
    assert parts[4] == dims.nheads // 2    # dd_dt
    assert parts[5] == dims.nheads // 2    # dd_A
    assert parts[6] == dims.nheads // 2    # trap
    assert parts[7] == dims.num_rope_angles  # angles (replicated)


def test_tp8_partition_sizes():
    cfg = _make_cfg(ngroups=8)
    dims = compute_mamba3_in_proj_dims(cfg)
    parts = compute_mamba3_tp_partition_sizes(dims, tp_size=8)

    assert parts[0] == 512 // 8  # z: 64
    assert parts[1] == 512 // 8  # x: 64
    assert parts[4] == 8 // 8    # dd_dt: 1
    assert parts[7] == 32        # angles: replicated


def test_tp_partition_fails_indivisible():
    cfg = _make_cfg(ngroups=1)
    dims = compute_mamba3_in_proj_dims(cfg)
    # nheads=8, tp_size=3 -> d_inner and nheads not divisible
    with pytest.raises(AssertionError, match="divisible by tp_size"):
        compute_mamba3_tp_partition_sizes(dims, tp_size=3)


# ---------------------------------------------------------------------------
# TE output_size
# ---------------------------------------------------------------------------

def test_te_output_size_tp1():
    cfg = _make_cfg(ngroups=8)
    dims = compute_mamba3_in_proj_dims(cfg)
    te_out = compute_mamba3_te_output_size(dims, tp_size=1)
    # TP=1: te_output_size = sum(local) * 1 = total
    assert te_out == dims.total


def test_te_output_size_tp2():
    """TE divides output_size by tp_size to get local. We verify the local
    size equals sum(partition_sizes)."""
    cfg = _make_cfg(ngroups=8)
    dims = compute_mamba3_in_proj_dims(cfg)
    te_out = compute_mamba3_te_output_size(dims, tp_size=2)
    local_sizes = compute_mamba3_tp_partition_sizes(dims, tp_size=2)
    assert te_out // 2 == sum(local_sizes)


def test_te_output_size_tp8():
    cfg = _make_cfg(ngroups=8)
    dims = compute_mamba3_in_proj_dims(cfg)
    te_out = compute_mamba3_te_output_size(dims, tp_size=8)
    local_sizes = compute_mamba3_tp_partition_sizes(dims, tp_size=8)
    assert te_out // 8 == sum(local_sizes)


# ---------------------------------------------------------------------------
# Cross-check with Author Mamba3.__init__ formula
# ---------------------------------------------------------------------------

def test_total_matches_author_formula():
    """Directly verify our total matches the Author Mamba3.__init__ formula:
        d_in_proj = 2*d_inner + 2*d_state*ngroups*mimo_rank + 3*nheads + num_rope_angles
    """
    for d_model, d_state, expand, headdim, ngroups, is_mimo, mimo_rank, rope_fraction in [
        (256, 128, 2, 64, 1, False, 4, 0.5),
        (256, 128, 2, 64, 8, True, 4, 0.5),
        (512, 64, 2, 64, 1, False, 1, 1.0),
        (1024, 128, 2, 128, 8, True, 2, 0.5),
        (768, 128, 2, 64, 4, False, 4, 0.5),
    ]:
        cfg = AuthorMamba3Config(
            d_model=d_model, d_state=d_state, expand=expand,
            headdim=headdim, ngroups=ngroups, is_mimo=is_mimo,
            mimo_rank=mimo_rank, rope_fraction=rope_fraction,
        )
        dims = compute_mamba3_in_proj_dims(cfg)

        # Reproduce the Author formula
        effective_mimo_rank = mimo_rank if is_mimo else 1
        d_inner = d_model * expand
        nheads = d_inner // headdim
        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        expected_angles = split_tensor_size // 2
        expected_total = (
            2 * d_inner
            + 2 * d_state * ngroups * effective_mimo_rank
            + 3 * nheads
            + expected_angles
        )

        assert dims.total == expected_total, (
            f"Mismatch for d_model={d_model}, d_state={d_state}, expand={expand}, "
            f"headdim={headdim}, ngroups={ngroups}, is_mimo={is_mimo}, "
            f"mimo_rank={mimo_rank}, rope_fraction={rope_fraction}: "
            f"got {dims.total}, expected {expected_total}"
        )
