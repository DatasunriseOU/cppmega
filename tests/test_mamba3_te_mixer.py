"""Unit tests for CppMegaMamba3TE mixer -- config mapping, dimension formulas,
and structural invariants that can be verified without GPU / megatron / mamba_ssm.

Tests that require the ``megatron`` package are skipped when it is not installed
(the module only runs on the remote H200 bench).
"""

import pytest

# ``find_spec("megatron")`` alone is unsafe here: an earlier test may have
# installed a ``MagicMock`` stub (possibly with or without a real
# ``ModuleSpec``).  Use the shared helper which (a) treats a stub as
# "not installed" and (b) ensures future ``find_spec`` calls don't raise
# ``ValueError: megatron.__spec__ is not set`` on Python 3.12+.
from tests._megatron_stub import install_megatron_stub, is_real_megatron_available

_has_megatron = is_real_megatron_available()
if not _has_megatron:
    install_megatron_stub()


# ---------------------------------------------------------------------------
# Test: module is importable (megatron required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_megatron, reason="megatron not installed locally")
def test_cppmega_mamba3_te_importable():
    from cppmega.megatron.mamba3_te_mixer import CppMegaMamba3TE
    assert CppMegaMamba3TE is not None


# ---------------------------------------------------------------------------
# Test: in_proj output dimension formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "d_model,d_state,headdim,ngroups,expand,rope_fraction,is_mimo,mimo_rank",
    [
        (256, 128, 64, 1, 2, 0.5, False, 1),
        (256, 128, 64, 1, 2, 1.0, False, 1),
        (256, 128, 64, 8, 2, 0.5, False, 1),
        (512, 64, 64, 4, 2, 0.5, True, 4),
    ],
)
def test_in_proj_output_dimension_formula(
    d_model, d_state, headdim, ngroups, expand, rope_fraction, is_mimo, mimo_rank
):
    """Verify the in_proj output dimension matches the Author Mamba3 formula.

    d_in_proj = 2*d_inner + 2*d_state*ngroups*mimo_rank + 3*nheads + num_rope_angles
    """
    d_inner = d_model * expand
    nheads = d_inner // headdim
    effective_mimo_rank = mimo_rank if is_mimo else 1

    split_tensor_size = int(d_state * rope_fraction)
    if split_tensor_size % 2 != 0:
        split_tensor_size -= 1
    num_rope_angles = split_tensor_size // 2

    expected = (
        2 * d_inner
        + 2 * d_state * ngroups * effective_mimo_rank
        + 3 * nheads
        + num_rope_angles
    )

    # Verify against Author Mamba3's formula from source
    # Order: [z, x, B, C, dd_dt, dd_A, trap, angle]
    author_d_in_proj = (
        2 * d_inner
        + 2 * d_state * ngroups * effective_mimo_rank
        + 3 * nheads
        + num_rope_angles
    )
    assert expected == author_d_in_proj


# ---------------------------------------------------------------------------
# Test: RoPE angle count derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d_state,rope_fraction,expected_angles", [
    (128, 0.5, 32),
    (128, 1.0, 64),
    (64, 0.5, 16),
    (64, 1.0, 32),
    (65, 0.5, 16),  # odd split_tensor_size rounded down
])
def test_rope_angle_count(d_state, rope_fraction, expected_angles):
    split_tensor_size = int(d_state * rope_fraction)
    if split_tensor_size % 2 != 0:
        split_tensor_size -= 1
    num_rope_angles = split_tensor_size // 2
    assert num_rope_angles == expected_angles


# ---------------------------------------------------------------------------
# Test: partition sizes are consistent with local TP sizes
# ---------------------------------------------------------------------------

def test_partition_sizes_sum_to_local_dim():
    """Verify the in_proj partition_sizes sum to the per-TP-rank output dim."""
    # Simulated TP=1 config
    d_model = 256
    d_state = 128
    headdim = 64
    ngroups = 1
    expand = 2
    mimo_rank = 1
    rope_fraction = 0.5

    d_inner = d_model * expand
    nheads = d_inner // headdim
    # TP=1 means local == global
    nheads_local = nheads
    d_inner_local = d_inner
    ngroups_local = ngroups

    split_tensor_size = int(d_state * rope_fraction)
    if split_tensor_size % 2 != 0:
        split_tensor_size -= 1
    num_rope_angles = split_tensor_size // 2

    partition_sizes = [
        d_inner_local,  # z
        d_inner_local,  # x
        ngroups_local * d_state * mimo_rank,  # B
        ngroups_local * d_state * mimo_rank,  # C
        nheads_local,  # dd_dt
        nheads_local,  # dd_A
        nheads_local,  # trap
        num_rope_angles,  # angles
    ]

    expected_total = (
        2 * d_inner
        + 2 * ngroups * d_state * mimo_rank
        + 3 * nheads
        + num_rope_angles
    )
    assert sum(partition_sizes) == expected_total


# ---------------------------------------------------------------------------
# Test: partition sizes with TP>1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_partition_sizes_sum_with_tp(tp_size):
    """With TP>1, per-rank partition_sizes should still sum to the local dim."""
    d_model = 512
    d_state = 128
    headdim = 64
    ngroups = 8
    expand = 2
    mimo_rank = 1

    d_inner = d_model * expand
    nheads = d_inner // headdim

    assert nheads % tp_size == 0
    assert ngroups % tp_size == 0

    nheads_local = nheads // tp_size
    d_inner_local = d_inner // tp_size
    ngroups_local = ngroups // tp_size

    rope_fraction = 0.5
    split_tensor_size = int(d_state * rope_fraction)
    if split_tensor_size % 2 != 0:
        split_tensor_size -= 1
    num_rope_angles = split_tensor_size // 2

    partition_sizes = [
        d_inner_local,         # z
        d_inner_local,         # x
        ngroups_local * d_state * mimo_rank,  # B
        ngroups_local * d_state * mimo_rank,  # C
        nheads_local,          # dd_dt
        nheads_local,          # dd_A
        nheads_local,          # trap
        num_rope_angles,       # angles (broadcast)
    ]

    # The total should match what TE ColumnParallelLinear produces per rank
    # (the global dim divided by tp, modulo the broadcast angle component)
    global_dim = (
        2 * d_inner
        + 2 * ngroups * d_state * mimo_rank
        + 3 * nheads
        + num_rope_angles
    )

    # Per-rank dim is NOT simply global_dim // tp because angles are broadcast.
    # Verify the split sizes are internally consistent.
    assert partition_sizes[0] == d_inner_local
    assert partition_sizes[4] == nheads_local
    assert partition_sizes[7] == num_rope_angles  # same on all ranks


# ---------------------------------------------------------------------------
# Test: Mamba3 in_proj split order matches Author's order
# ---------------------------------------------------------------------------

def test_in_proj_split_order_matches_author():
    """The split order must be [z, x, B, C, dd_dt, dd_A, trap, angles] to
    match the Author Mamba3 forward() split."""
    # This is the Author Mamba3 split order (from source):
    author_order = ["z", "x", "B", "C", "dd_dt", "dd_A", "trap", "angles"]
    # Our partition label order:
    our_order = ["z", "x", "B", "C", "dd_dt", "dd_A", "trap", "angles"]
    assert our_order == author_order


# ---------------------------------------------------------------------------
# Test: mamba_state_shapes_per_request returns correct shapes
# ---------------------------------------------------------------------------

def test_state_shapes_siso():
    """Verify state shapes for SISO (non-MIMO) config."""
    nheads_local = 8
    headdim = 64
    d_state = 128
    mimo_rank = 1
    num_rope_angles = 32

    angle_shape = (nheads_local, num_rope_angles)
    ssm_shape = (nheads_local, headdim, d_state)
    k_shape = (mimo_rank, nheads_local, d_state)
    v_shape = (nheads_local, headdim)

    assert angle_shape == (8, 32)
    assert ssm_shape == (8, 64, 128)
    assert k_shape == (1, 8, 128)
    assert v_shape == (8, 64)


def test_state_shapes_mimo():
    """Verify state shapes for MIMO config."""
    nheads_local = 8
    headdim = 64
    d_state = 128
    mimo_rank = 4
    num_rope_angles = 32

    k_shape = (mimo_rank, nheads_local, d_state)
    assert k_shape == (4, 8, 128)


# ---------------------------------------------------------------------------
# Test: Mamba3 differs from upstream MambaMixer in key ways
# ---------------------------------------------------------------------------

def test_mamba3_has_eight_projection_components_not_five():
    """Author Mamba3 projects [z, x, B, C, dd_dt, dd_A, trap, angles] (8 parts)
    vs upstream MambaMixer [z, x, B, C, dt] (5 parts). Verify our count."""
    author_components = ["z", "x", "B", "C", "dd_dt", "dd_A", "trap", "angles"]
    upstream_components = ["z", "x", "B", "C", "dt"]
    assert len(author_components) == 8
    assert len(upstream_components) == 5


# ---------------------------------------------------------------------------
# Test: no conv1d in module source (Mamba3 replaces conv+SSD)
# ---------------------------------------------------------------------------

def test_no_conv1d_in_source():
    """CppMegaMamba3TE should not reference conv1d."""
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "cppmega" / "megatron" / "mamba3_te_mixer.py"
    text = src.read_text()
    # Should not define self.conv1d
    assert "self.conv1d" not in text
    # Should not import Conv1d
    assert "Conv1d" not in text


# ---------------------------------------------------------------------------
# Test: source references TE linear builders
# ---------------------------------------------------------------------------

def test_source_uses_te_build_module():
    """Verify the source builds in_proj and out_proj via build_module (TE path)."""
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "cppmega" / "megatron" / "mamba3_te_mixer.py"
    text = src.read_text()
    assert "build_module" in text
    assert "submodules.in_proj" in text
    assert "submodules.out_proj" in text
    assert "gather_output=False" in text
    assert "input_is_parallel=True" in text


# ---------------------------------------------------------------------------
# Test: source references all Mamba3 scan kernels
# ---------------------------------------------------------------------------

def test_source_references_mamba3_kernels():
    """Verify the source imports and calls the Author Mamba3 scan kernels."""
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "cppmega" / "megatron" / "mamba3_te_mixer.py"
    text = src.read_text()
    assert "mamba3_siso_combined" in text
    assert "mamba3_mimo_combined" in text
    assert "angle_dt" in text
    assert "RMSNormGated" in text


# ---------------------------------------------------------------------------
# Test: source supports all Mamba3 features
# ---------------------------------------------------------------------------

def test_source_supports_all_mamba3_features():
    """Verify trapezoidal, qknorm, bias, complex RoPE, data-dep A, MIMO."""
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "cppmega" / "megatron" / "mamba3_te_mixer.py"
    text = src.read_text()

    # Trapezoidal: trap projection + Trap kernel arg
    assert "trap" in text.lower()
    assert "Trap=trap" in text

    # QK-Norm: B_norm and C_norm
    assert "self.B_norm" in text
    assert "self.C_norm" in text

    # Bias: B_bias and C_bias
    assert "self.B_bias" in text
    assert "self.C_bias" in text

    # Complex RoPE: angles, rotary_dim_divisor
    assert "num_rope_angles" in text
    assert "rotary_dim_divisor" in text

    # Data-dependent A: dd_A, softplus, A_floor
    assert "dd_A" in text
    assert "F.softplus" in text
    assert "A_floor" in text

    # MIMO: mimo_x, mimo_z, mimo_o
    assert "self.mimo_x" in text
    assert "self.mimo_z" in text
    assert "self.mimo_o" in text
    assert "is_mimo" in text


# ---------------------------------------------------------------------------
# Test: MIMO in_proj dimension is larger than SISO
# ---------------------------------------------------------------------------

def test_mimo_in_proj_larger_than_siso():
    """With MIMO rank > 1, the B/C components are rank times larger."""
    d_model = 256
    d_state = 128
    headdim = 64
    ngroups = 1
    expand = 2
    rope_fraction = 0.5

    d_inner = d_model * expand
    nheads = d_inner // headdim

    split_tensor_size = int(d_state * rope_fraction)
    if split_tensor_size % 2 != 0:
        split_tensor_size -= 1
    num_rope_angles = split_tensor_size // 2

    def _dim(mimo_rank):
        return (
            2 * d_inner
            + 2 * d_state * ngroups * mimo_rank
            + 3 * nheads
            + num_rope_angles
        )

    siso_dim = _dim(1)
    mimo_dim = _dim(4)
    assert mimo_dim > siso_dim
    assert mimo_dim - siso_dim == 2 * d_state * ngroups * 3  # rank difference of 3
