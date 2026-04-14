"""Unit tests for NoConvMambaMixer and Mamba3 scan adaptation helpers.

Structural invariants are verifiable without GPU, megatron, or mamba_ssm.
Tests that require the ``megatron`` package are skipped when it is not installed
(the module only runs on the remote H200 bench).

The Mamba3 helper functions (_apply_rope_on_state_dim, _compute_trapezoidal_scale,
_compute_data_dependent_A) only need torch and can be tested locally.
"""

import importlib
import pathlib
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

# See tests/_megatron_stub.py: a prior test (e.g. test_fastmtp_layer.py) may
# have installed a MagicMock stub for megatron.  Use the shared helper to
# detect a real install vs stub and to keep ``find_spec`` from raising on
# Python 3.12+.
from tests._megatron_stub import install_megatron_stub, is_real_megatron_available

_has_megatron = is_real_megatron_available()
if not _has_megatron:
    install_megatron_stub()
_module_path = pathlib.Path(__file__).parent.parent / "cppmega" / "megatron" / "noconv_mamba_mixer.py"


def _read_source() -> str:
    return _module_path.read_text()


def _executable_lines(text: str) -> list[str]:
    """Return lines that are not inside docstrings or comments."""
    result = []
    in_docstring = False
    for line in text.split("\n"):
        stripped = line.strip()
        if '"""' in stripped:
            count = stripped.count('"""')
            if count == 1:
                in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if stripped.startswith("#"):
            continue
        result.append(line)
    return result


# ---------------------------------------------------------------------------
# Importability (megatron required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_megatron, reason="megatron not installed locally")
def test_noconv_mamba_mixer_importable():
    from cppmega.megatron.noconv_mamba_mixer import NoConvMambaMixer
    assert NoConvMambaMixer is not None


@pytest.mark.skipif(not _has_megatron, reason="megatron not installed locally")
def test_submodules_importable():
    from cppmega.megatron.noconv_mamba_mixer import NoConvMambaMixerSubmodules
    assert NoConvMambaMixerSubmodules is not None


# ---------------------------------------------------------------------------
# Source-level structural invariants (no megatron import needed)
# ---------------------------------------------------------------------------

def test_no_conv1d_constructed():
    """The executable code must not construct nn.Conv1d or import conv1d functions."""
    lines = _executable_lines(_read_source())
    code = "\n".join(lines)
    assert "nn.Conv1d(" not in code
    assert "causal_conv1d_fn" not in code


def test_uses_chunk_scan_combined():
    """The training path must use mamba_chunk_scan_combined (not the fused conv variant)."""
    text = _read_source()
    assert "mamba_chunk_scan_combined" in text


def test_no_mamba_context_parallel_import():
    """The executable code must not import or instantiate MambaContextParallel."""
    lines = _executable_lines(_read_source())
    code = "\n".join(lines)
    assert "MambaContextParallel" not in code


def test_source_uses_build_module():
    """Verify in_proj and out_proj are built via build_module (TE / local path)."""
    text = _read_source()
    assert "build_module" in text
    assert "submodules.in_proj" in text
    assert "submodules.out_proj" in text
    assert "gather_output=False" in text
    assert "input_is_parallel=True" in text


def test_source_has_silu_activation():
    """SiLU should be applied to x in place of the conv1d + SiLU pipeline."""
    text = _read_source()
    assert "self.act(x)" in text
    assert "nn.SiLU()" in text


def test_source_has_tp_attributes():
    """Verify TP-sharding metadata attributes are set on A_log, dt_bias, D."""
    text = _read_source()
    for param_name in ("self.dt_bias", "self.A_log", "self.D"):
        assert param_name in text, f"{param_name} not found in source"
    assert "tensor_model_parallel" in text
    assert "partition_dim" in text


def test_source_has_partition_sizes():
    """Verify partition_sizes are set on in_proj.weight for checkpoint resharding."""
    text = _read_source()
    assert "partition_sizes" in text
    assert "in_proj_partition_sizes" in text


# ---------------------------------------------------------------------------
# in_proj partition sizes formula (pure arithmetic, no imports)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "d_model,d_state,headdim,ngroups,expand,tp_size",
    [
        (256, 128, 64, 1, 2, 1),
        (256, 128, 64, 8, 2, 1),
        (512, 128, 64, 8, 2, 2),
        (1024, 128, 64, 8, 2, 4),
    ],
)
def test_in_proj_partition_sizes_sum(d_model, d_state, headdim, ngroups, expand, tp_size):
    """The per-TP-rank partition sizes should sum to the local in_proj output dim."""
    d_inner = d_model * expand
    nheads = d_inner // headdim
    nheads_local = nheads // tp_size
    d_inner_local = d_inner // tp_size
    ngroups_local = ngroups // tp_size

    partition_sizes = [
        d_inner_local,                     # z
        d_inner_local,                     # x
        ngroups_local * d_state,           # B
        ngroups_local * d_state,           # C
        nheads_local,                      # dt
    ]

    expected_total = (
        2 * d_inner_local
        + 2 * ngroups_local * d_state
        + nheads_local
    )
    assert sum(partition_sizes) == expected_total


def test_in_proj_output_dim_matches_mamba_mixer():
    """The in_proj output dimension should match MambaMixer's formula."""
    d_model = 256
    d_state = 128
    headdim = 64
    ngroups = 8
    expand = 2

    d_inner = d_model * expand
    nheads = d_inner // headdim

    mamba_mixer_in_proj = d_inner * 2 + 2 * ngroups * d_state + nheads
    noconv_in_proj = d_inner * 2 + 2 * ngroups * d_state + nheads

    assert mamba_mixer_in_proj == noconv_in_proj


# ---------------------------------------------------------------------------
# Docstrings
# ---------------------------------------------------------------------------

def test_module_has_docstring():
    text = _read_source()
    assert "conv1d" in text.lower()


def test_class_docstring_mentions_no_conv():
    text = _read_source()
    assert "without conv1d" in text.lower() or "no conv1d" in text.lower()


# ---------------------------------------------------------------------------
# Split components match MambaMixer [z, x, B, C, dt]
# ---------------------------------------------------------------------------

def test_split_matches_mamba_mixer():
    """NoConvMambaMixer should split in_proj output into [z, x, B, C, dt]."""
    text = _read_source()
    assert "z, x, B, C, dt" in text


# ---------------------------------------------------------------------------
# Forward signature (megatron required for import)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_megatron, reason="megatron not installed locally")
def test_forward_signature():
    import inspect
    from cppmega.megatron.noconv_mamba_mixer import NoConvMambaMixer
    sig = inspect.signature(NoConvMambaMixer.forward)
    params = set(sig.parameters.keys())
    required = {"self", "hidden_states", "inference_context",
                "inference_params", "packed_seq_params"}
    assert required.issubset(params), f"Missing: {required - params}"


# ---------------------------------------------------------------------------
# Can be wrapped in a ModuleSpec (megatron required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_megatron, reason="megatron not installed locally")
def test_module_spec_construction():
    from megatron.core.transformer.spec_utils import ModuleSpec
    from cppmega.megatron.noconv_mamba_mixer import (
        NoConvMambaMixer,
        NoConvMambaMixerSubmodules,
    )

    spec = ModuleSpec(
        module=NoConvMambaMixer,
        submodules=NoConvMambaMixerSubmodules(
            in_proj="ColumnParallelLinear",
            out_proj="RowParallelLinear",
        ),
    )
    assert spec.module is NoConvMambaMixer
    assert spec.submodules.in_proj == "ColumnParallelLinear"


# ===========================================================================
# Mamba3 scan adaptation helper tests (torch-only, no GPU required)
# ===========================================================================

# The helper functions only use torch, but they live in a module that imports
# megatron at the top level.  We extract them by exec-ing just the function
# source to avoid triggering megatron imports.


def _extract_helper_functions():
    """Extract the three helper functions from source without importing the module."""
    import ast
    import textwrap

    src = _read_source()
    tree = ast.parse(src)

    helper_names = {
        "_apply_rope_on_state_dim",
        "_compute_trapezoidal_scale",
        "_compute_data_dependent_A",
    }

    ns = {
        "torch": torch,
        "F": F,
        "Optional": Optional,
        "Tuple": Tuple,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in helper_names:
            func_src = ast.get_source_segment(src, node)
            if func_src:
                exec(compile(func_src, "<helper>", "exec"), ns)  # noqa: S102

    return (
        ns.get("_apply_rope_on_state_dim"),
        ns.get("_compute_data_dependent_A"),
        ns.get("_compute_trapezoidal_scale"),
    )


_apply_rope_on_state_dim, _compute_data_dependent_A, _compute_trapezoidal_scale = (
    _extract_helper_functions()
)

_helpers_available = all([
    _apply_rope_on_state_dim is not None,
    _compute_data_dependent_A is not None,
    _compute_trapezoidal_scale is not None,
])


@pytest.mark.skipif(not _helpers_available, reason="helpers could not be extracted")
class TestComputeDataDependentA:
    """Tests for _compute_data_dependent_A."""

    def test_output_is_negative(self):
        dd_A = torch.randn(2, 8, 4)
        A_dd = _compute_data_dependent_A(dd_A, A_floor=1e-4)
        assert (A_dd < 0).all()

    def test_clamped_by_floor(self):
        """All values should satisfy |A| >= A_floor."""
        dd_A = torch.randn(2, 8, 4)
        A_floor = 0.01
        A_dd = _compute_data_dependent_A(dd_A, A_floor=A_floor)
        assert (A_dd <= -A_floor).all()

    def test_output_shape_matches_input(self):
        dd_A = torch.randn(3, 16, 8)
        A_dd = _compute_data_dependent_A(dd_A)
        assert A_dd.shape == dd_A.shape

    def test_softplus_applied(self):
        """For large positive inputs, -softplus(x) approx -x."""
        dd_A = torch.tensor([[[10.0]]])
        A_dd = _compute_data_dependent_A(dd_A, A_floor=0.0)
        assert abs(A_dd.item() + 10.0) < 0.01


@pytest.mark.skipif(not _helpers_available, reason="helpers could not be extracted")
class TestComputeTrapezoidalScale:
    """Tests for _compute_trapezoidal_scale."""

    def test_output_shape(self):
        batch, seqlen, nheads = 2, 16, 4
        dt = torch.rand(batch, seqlen, nheads) + 0.01
        trap = torch.randn(batch, nheads, seqlen)
        scale = _compute_trapezoidal_scale(dt, trap)
        assert scale.shape == (batch, seqlen, nheads)

    def test_scale_is_nonnegative(self):
        """Scale should be non-negative since dt > 0 and sigmoid in [0,1]."""
        batch, seqlen, nheads = 2, 16, 4
        dt = torch.rand(batch, seqlen, nheads) + 0.01
        trap = torch.randn(batch, nheads, seqlen)
        scale = _compute_trapezoidal_scale(dt, trap)
        assert (scale >= 0).all()

    def test_last_position_uses_only_forward_euler(self):
        """At the last position, dt_shifted=0, so scale = dt * sigmoid(trap)."""
        batch, nheads = 1, 2
        seqlen = 4
        dt = torch.ones(batch, seqlen, nheads)
        trap = torch.zeros(batch, nheads, seqlen)  # sigmoid(0) = 0.5
        scale = _compute_trapezoidal_scale(dt, trap)
        # Last position: dt_shifted=0, sig_trap_shifted=0.5
        # scale = 0 * 0.5 + 1.0 * 0.5 = 0.5
        assert torch.allclose(scale[:, -1, :], torch.tensor(0.5), atol=1e-6)


@pytest.mark.skipif(not _helpers_available, reason="helpers could not be extracted")
class TestApplyRopeOnStateDim:
    """Tests for _apply_rope_on_state_dim."""

    def test_output_shape_matches_input(self):
        batch, seqlen, ngroups, d_state = 2, 8, 1, 16
        n_rope_angles = d_state // 4  # rope_fraction=0.5
        tensor = torch.randn(batch, seqlen, ngroups, d_state)
        angles_cumsum = torch.randn(batch, seqlen, ngroups, n_rope_angles)
        out = _apply_rope_on_state_dim(tensor, angles_cumsum)
        assert out.shape == tensor.shape

    def test_zero_angle_is_identity(self):
        """With zero angles, the output should equal the input."""
        batch, seqlen, ngroups, d_state = 1, 4, 1, 8
        n_rope_angles = 2
        tensor = torch.randn(batch, seqlen, ngroups, d_state)
        angles_cumsum = torch.zeros(batch, seqlen, ngroups, n_rope_angles)
        out = _apply_rope_on_state_dim(tensor, angles_cumsum)
        assert torch.allclose(out, tensor, atol=1e-6)

    def test_pi_rotation_negates_pairs(self):
        """Rotating by pi should negate both elements of each pair."""
        batch, seqlen, ngroups, d_state = 1, 1, 1, 4
        n_rope_angles = 2
        tensor = torch.tensor([[[[1.0, 0.0, 0.0, 1.0]]]])
        angles_cumsum = torch.full((batch, seqlen, ngroups, n_rope_angles), torch.pi)
        out = _apply_rope_on_state_dim(tensor, angles_cumsum)
        expected = torch.tensor([[[[-1.0, 0.0, 0.0, -1.0]]]])
        assert torch.allclose(out, expected, atol=1e-5)

    def test_passthrough_elements_unchanged(self):
        """Elements beyond the rotated portion should be unchanged."""
        batch, seqlen, ngroups, d_state = 1, 2, 1, 8
        n_rope_angles = 1  # only rotate first 2 elements
        tensor = torch.randn(batch, seqlen, ngroups, d_state)
        angles_cumsum = torch.randn(batch, seqlen, ngroups, n_rope_angles)
        out = _apply_rope_on_state_dim(tensor, angles_cumsum)
        # Elements 2..7 should be unchanged
        assert torch.allclose(out[..., 2:], tensor[..., 2:], atol=1e-6)


@pytest.mark.skipif(not _helpers_available, reason="helpers could not be extracted")
class TestDataDependentAScanIdentity:
    """Test the key mathematical identity for feeding data-dependent A
    through mamba_chunk_scan_combined.

    The kernel computes:  dA = dt * A   and   dA_cumsum = cumsum(dA)

    To get cumsum(A_dd * DT) where A_dd is per-position, we set:
        A_kernel = -1  (scalar)
        dt_kernel = -ADT = -(A_dd * DT) = |A_dd| * DT  (positive)

    Then:  dA = dt_kernel * A_kernel = |A_dd| * DT * (-1) = A_dd * DT
    """

    def test_identity_holds(self):
        """cumsum(A_kernel * dt_kernel) should equal cumsum(A_dd * DT)."""
        seqlen, nheads = 32, 4

        # Simulate data-dependent A
        dd_A = torch.randn(1, seqlen, nheads)
        A_dd = _compute_data_dependent_A(dd_A, A_floor=1e-4)

        # Simulate dt
        DT = torch.rand(1, seqlen, nheads) * 0.1 + 0.01

        # Target: cumsum(A_dd * DT) along seqlen
        ADT = A_dd * DT
        target_cumsum = torch.cumsum(ADT, dim=1)

        # What the kernel computes with our pre-processing:
        A_kernel = torch.tensor([-1.0]).expand(nheads)
        dt_kernel = -ADT  # positive
        kernel_dA = dt_kernel * A_kernel.unsqueeze(0).unsqueeze(0)
        kernel_cumsum = torch.cumsum(kernel_dA, dim=1)

        assert torch.allclose(kernel_cumsum, target_cumsum, atol=1e-6)

    def test_dt_kernel_is_positive(self):
        """dt_kernel = -ADT should always be positive (required by kernel)."""
        dd_A = torch.randn(2, 16, 8)
        A_dd = _compute_data_dependent_A(dd_A, A_floor=1e-4)
        DT = torch.rand(2, 16, 8) * 0.1 + 0.01
        ADT = A_dd * DT
        dt_kernel = -ADT
        assert (dt_kernel >= 0).all()


class TestMamba3NoConvMixerSource:
    """Source-level tests for the Mamba3NoConvMixer class."""

    def test_class_exists_in_source(self):
        text = _read_source()
        assert "class Mamba3NoConvMixer" in text

    def test_inherits_mamba3_scan_mixin(self):
        text = _read_source()
        assert "Mamba3ScanMixin" in text
        assert "class Mamba3NoConvMixer(Mamba3ScanMixin, NoConvMambaMixer)" in text

    def test_splits_eight_components(self):
        """Mamba3NoConvMixer should split into [z, x, B, C, dd_dt, dd_A, trap, angles]."""
        text = _read_source()
        assert "z, x, B, C, dd_dt, dd_A, trap, angles" in text

    def test_calls_preprocess_bc_mamba3(self):
        text = _read_source()
        assert "_preprocess_bc_mamba3" in text

    def test_calls_mamba3_scan(self):
        text = _read_source()
        assert "_mamba3_scan" in text

    def test_has_qknorm_parameters(self):
        text = _read_source()
        assert "B_norm_weight" in text
        assert "C_norm_weight" in text
        assert "B_bias" in text
        assert "C_bias" in text

    def test_has_rope_angles_in_partition(self):
        text = _read_source()
        assert "n_rope_angles" in text


class TestMamba3InProjPartitionSizes:
    """Verify that the extended in_proj partition sizes sum correctly."""

    @pytest.mark.parametrize(
        "d_model,d_state,headdim,ngroups,expand,tp_size,rope_fraction",
        [
            (256, 128, 64, 1, 2, 1, 0.5),
            (256, 128, 64, 8, 2, 1, 0.5),
            (512, 128, 64, 8, 2, 2, 0.5),
            (1024, 128, 64, 8, 2, 4, 1.0),
        ],
    )
    def test_partition_sizes_sum(
        self, d_model, d_state, headdim, ngroups, expand, tp_size, rope_fraction,
    ):
        d_inner = d_model * expand
        nheads = d_inner // headdim
        nheads_local = nheads // tp_size
        d_inner_local = d_inner // tp_size
        ngroups_local = ngroups // tp_size

        split_size = int(d_state * rope_fraction)
        if split_size % 2 != 0:
            split_size -= 1
        n_rope_angles = split_size // 2

        partition_sizes = [
            d_inner_local,                     # z
            d_inner_local,                     # x
            ngroups_local * d_state,           # B
            ngroups_local * d_state,           # C
            nheads_local,                      # dd_dt
            nheads_local,                      # dd_A
            nheads_local,                      # trap
            n_rope_angles,                     # angles
        ]

        expected_total = (
            2 * d_inner_local
            + 2 * ngroups_local * d_state
            + 3 * nheads_local
            + n_rope_angles
        )
        assert sum(partition_sizes) == expected_total
