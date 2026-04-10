"""Megatron-native Mamba mixer that skips the conv1d pre-processing stage.

Mamba3 removed conv1d entirely, replacing it with RoPE.  When we want to use
Megatron's TP-aware ``ColumnParallelLinear`` / ``RowParallelLinear`` data flow
(which ``AuthorMamba3Mixer`` does not) while dropping the conv1d that
Megatron's stock ``MambaMixer`` hard-wires, this mixer is the answer.

The forward path is:
    in_proj(hidden_states) -> [z, x, B, C, dt]  (ColumnParallelLinear, TP-sharded)
    SiLU(x)                                       (replaces conv1d + SiLU)
    mamba_chunk_scan_combined(x, dt, A, B, C, ...)  (the conv-free SSM kernel)
    out_proj(y)                                   (RowParallelLinear, TP-sharded)

Compared to ``MambaMixer``:
  - No ``nn.Conv1d``, no ``MambaContextParallel`` wrapper.
  - ``in_proj`` still packs [z, x, B, C, dt] with the same partition layout.
  - Uses ``mamba_chunk_scan_combined`` (pre-split inputs) instead of the fused
    ``mamba_split_conv1d_scan_combined`` (which bakes in conv).
  - TP attributes (``tensor_model_parallel``, ``partition_dim``) are set on
    ``A_log``, ``dt_bias``, ``D`` identically to ``MambaMixer``.

Why this exists alongside ``CppMegaMamba3TE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``CppMegaMamba3TE`` uses the **author Mamba3 scan kernels** (``mamba3_siso_combined``
/ ``mamba3_mimo_combined``) which natively support data-dependent A, trapezoidal
discretization, complex RoPE, and QK normalization.

``NoConvMambaMixer`` instead uses the **Mamba2 SSD kernel**
(``mamba_chunk_scan_combined``) with the conv1d stage removed.  This is useful for:

  - Ablation studies isolating the effect of conv1d removal
  - Hybrid stacks mixing conv and no-conv Mamba2 layers
  - Environments where only the Mamba2 kernel is available
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None
    RMSNormGated = None


# ---------------------------------------------------------------------------
# Mamba3 feature helpers
# ---------------------------------------------------------------------------


def _apply_rope_on_state_dim(
    tensor: torch.Tensor,
    angles_cumsum: torch.Tensor,
    rope_fraction: float = 0.5,
) -> torch.Tensor:
    """Apply complex RoPE to B or C along the state dimension.

    Args:
        tensor: (batch, seqlen, ngroups, d_state) -- the B or C projection.
        angles_cumsum: (batch, seqlen, nheads, n_rope_angles) -- accumulated
            angles, where n_rope_angles = d_state * rope_fraction // 2.

    The rotation is applied to the first ``2 * n_rope_angles`` elements of the
    state dimension, treating consecutive pairs as (real, imag).  Remaining
    elements are left untouched.

    For grouped-query style layouts where ngroups < nheads, we take angles
    from the first group's worth of heads (angles[:, :, :ngroups, :]).
    """
    b, l, g, n = tensor.shape
    n_rope = angles_cumsum.shape[-1]  # number of angle pairs
    rot_dim = 2 * n_rope
    if rot_dim > n:
        rot_dim = n
        n_rope = rot_dim // 2

    # Select angles for the group dimension
    # angles_cumsum: (batch, seqlen, nheads, n_rope_angles)
    # We need (batch, seqlen, ngroups, n_rope_angles) -- take first ngroups heads
    ang = angles_cumsum[:, :, :g, :n_rope]  # (b, l, g, n_rope)

    cos_a = torch.cos(ang)  # (b, l, g, n_rope)
    sin_a = torch.sin(ang)

    # Split tensor into rotated and pass-through parts
    t_rot = tensor[..., :rot_dim]   # (b, l, g, rot_dim)
    t_pass = tensor[..., rot_dim:]  # (b, l, g, n - rot_dim)

    # Reshape to pairs: (b, l, g, n_rope, 2)
    t_pairs = t_rot.reshape(b, l, g, n_rope, 2)
    t0 = t_pairs[..., 0]  # (b, l, g, n_rope)
    t1 = t_pairs[..., 1]

    # Complex rotation: (t0 + i*t1) * (cos + i*sin)
    # Real part: t0*cos - t1*sin
    # Imag part: t0*sin + t1*cos
    r0 = t0 * cos_a - t1 * sin_a
    r1 = t0 * sin_a + t1 * cos_a

    # Reassemble
    rotated = torch.stack([r0, r1], dim=-1).reshape(b, l, g, rot_dim)
    if t_pass.shape[-1] > 0:
        return torch.cat([rotated, t_pass], dim=-1)
    return rotated


def _compute_trapezoidal_scale(
    dt: torch.Tensor,
    trap: torch.Tensor,
) -> torch.Tensor:
    """Compute the trapezoidal discretisation scale factor.

    In Author Mamba3 the scale applied to K (our B) at position t is:

        scale[t] = dt[t+1] * (1 - sigmoid(trap[t+1])) + dt[t] * sigmoid(trap[t])

    This mixes the forward-Euler contribution from position t (weighted by
    sigmoid(trap[t])) with the backward-Euler contribution from position t+1
    (weighted by 1-sigmoid(trap[t+1])).

    Args:
        dt: (batch, seqlen, nheads) -- post-softplus dt values.
        trap: (batch, nheads, seqlen) -- raw trapezoidal gate logits from
              the projection.  We apply sigmoid internally.

    Returns:
        scale: (batch, seqlen, nheads) -- per-position scale factor.
    """
    # trap layout: (batch, nheads, seqlen) -> (batch, seqlen, nheads)
    trap_bln = trap.transpose(1, 2)
    sig_trap = torch.sigmoid(trap_bln)  # (b, l, h)

    # Shifted versions (position t+1): pad with the last position value
    # dt[t+1] for the last position is 0 (no future contribution)
    dt_shifted = F.pad(dt[:, 1:, :], (0, 0, 0, 1), value=0.0)
    sig_trap_shifted = F.pad(sig_trap[:, 1:, :], (0, 0, 0, 1), value=0.5)

    scale = dt_shifted * (1.0 - sig_trap_shifted) + dt * sig_trap
    return scale  # (batch, seqlen, nheads)


def _compute_data_dependent_A(
    dd_A: torch.Tensor,
    A_floor: float = 1e-4,
) -> torch.Tensor:
    """Compute data-dependent A from projected logits.

    Args:
        dd_A: (batch, seqlen, nheads) -- raw projected values.
        A_floor: Minimum magnitude for A (prevents vanishing decay).

    Returns:
        A_dd: (batch, seqlen, nheads) -- negative, clamped A values.
    """
    A_dd = -F.softplus(dd_A.float())
    A_dd = torch.clamp(A_dd, max=-A_floor)
    return A_dd


class Mamba3ScanMixin:
    """Mixin providing the modified scan call for Mamba3 features.

    This mixin provides ``_mamba3_scan`` which wraps
    ``mamba_chunk_scan_combined`` with pre-processing that emulates the four
    Mamba3-specific features using the unmodified SSD kernel.

    The key insight is that ``_chunk_cumsum_fwd`` inside the kernel computes::

        dt_out = softplus(dt + dt_bias)    # when dt_softplus=True
        dA_cumsum = cumsum(A * dt_out)     # A is (nheads,) scalar

    To inject data-dependent A, we pre-compute ``ADT = A_dd * DT`` outside
    and pass ``A = -torch.ones(nheads)`` (constant) with ``dt = -ADT``
    (positive) and ``dt_softplus=False, dt_bias=None``.  The kernel then
    computes ``cumsum(-1 * (-ADT)) = cumsum(ADT) = cumsum(A_dd * DT)``.

    For trapezoidal discretisation, we scale B by the trapezoidal factor
    before the kernel call.  For RoPE, we apply rotary embeddings to B and C
    in Python.  For QK-norm, we apply RMSNorm + bias to B and C.
    """

    def _mamba3_scan(
        self,
        x: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        z: torch.Tensor,
        dd_dt: torch.Tensor,
        dd_A: Optional[torch.Tensor],
        trap: Optional[torch.Tensor],
        angles: Optional[torch.Tensor],
        dt_bias: torch.Tensor,
        D: torch.Tensor,
        chunk_size: int,
        rmsnorm: bool,
        A_floor: float = 1e-4,
        rope_fraction: float = 0.5,
        return_final_states: bool = False,
    ) -> torch.Tensor:
        """Run ``mamba_chunk_scan_combined`` with Mamba3 pre-processing.

        Args:
            x: (batch, seqlen, nheads, headdim) -- input values.
            B: (batch, seqlen, ngroups, d_state) -- SSM input projection.
            C: (batch, seqlen, ngroups, d_state) -- SSM output projection.
            z: (batch, seqlen, nheads, headdim) -- gate tensor.
            dd_dt: (batch, seqlen, nheads) -- raw dt projection (pre-softplus).
            dd_A: (batch, seqlen, nheads) -- raw data-dep A, or None for fixed.
            trap: (batch, nheads, seqlen) -- trapezoidal gate logits, or None.
            angles: (batch, seqlen, nheads, n_angles) -- RoPE angles, or None.
            dt_bias: (nheads,) -- learnable dt bias.
            D: (nheads,) or (nheads, headdim) -- skip connection.
            chunk_size: Kernel chunk size.
            rmsnorm: Controls z gating in kernel.
            A_floor: Minimum |A| magnitude.
            rope_fraction: Fraction of state dim to rotate.
            return_final_states: Whether to return final SSM states.

        Returns:
            y or (y, final_states) from the kernel.
        """
        batch, seqlen, nheads, headdim = x.shape

        # --- Compute dt and handle data-dependent A ---
        DT = F.softplus(dd_dt + dt_bias)  # (batch, seqlen, nheads)

        if dd_A is not None:
            # Data-dependent A: per-position decay
            A_dd = _compute_data_dependent_A(dd_A, A_floor=A_floor)
            # ADT = A_dd * DT, where A_dd is negative, DT is positive
            ADT = A_dd * DT  # (batch, seqlen, nheads) -- negative values

            # To make the kernel produce cumsum(ADT):
            # Kernel computes: dA = dt_kernel * A_kernel, then cumsum(dA).
            # Set A_kernel = -1, dt_kernel = -ADT (positive).
            # Then dA = (-ADT) * (-1) = ADT. cumsum(dA) = cumsum(ADT).
            A_kernel = torch.full(
                (nheads,), -1.0, device=x.device, dtype=torch.float32,
            )
            dt_kernel = -ADT  # positive values

            y = mamba_chunk_scan_combined(
                x, dt_kernel, A_kernel, B, C, chunk_size,
                D=D,
                z=z if not rmsnorm else None,
                dt_bias=None,
                dt_softplus=False,
                return_final_states=return_final_states,
            )
        else:
            # Fixed A from A_log -- standard Mamba2 path
            A = -torch.exp(self.A_log.float())  # (nheads,)
            y = mamba_chunk_scan_combined(
                x, dd_dt, A, B, C, chunk_size,
                D=D,
                z=z if not rmsnorm else None,
                dt_bias=dt_bias,
                dt_softplus=True,
                return_final_states=return_final_states,
            )

        return y

    def _preprocess_bc_mamba3(
        self,
        B: torch.Tensor,
        C: torch.Tensor,
        dt: torch.Tensor,
        dt_bias: torch.Tensor,
        dd_dt: torch.Tensor,
        trap: Optional[torch.Tensor],
        angles: Optional[torch.Tensor],
        rope_fraction: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply QK-norm, trapezoidal scaling, and RoPE to B and C.

        Call this BEFORE ``_mamba3_scan``.

        Args:
            B, C: (batch, seqlen, ngroups, d_state).
            dt: (batch, seqlen, nheads) -- post-softplus dt.
            dt_bias: (nheads,) -- learnable bias.
            dd_dt: (batch, seqlen, nheads) -- raw pre-softplus dt.
            trap: (batch, nheads, seqlen) -- trapezoidal gate logits, or None.
            angles: (batch, seqlen, nheads, n_angles) -- angle rates, or None.
            rope_fraction: Fraction of d_state to rotate.

        Returns:
            (B_transformed, C_transformed) with same shapes as input.
        """
        d_state = B.shape[-1]

        # --- QK-norm on B and C ---
        if hasattr(self, "m3_qknorm") and self.m3_qknorm:
            B = F.rms_norm(B, (d_state,)) * self.B_norm_weight
            C = F.rms_norm(C, (d_state,)) * self.C_norm_weight
        if hasattr(self, "m3_bias") and self.m3_bias:
            B = B + self.B_bias
            C = C + self.C_bias

        # --- Trapezoidal discretisation: scale B ---
        if trap is not None:
            dt_for_trap = F.softplus(dd_dt + dt_bias)  # (b, l, h)
            scale = _compute_trapezoidal_scale(dt_for_trap, trap)  # (b, l, h)
            # Expand from nheads to ngroups
            ngroups = B.shape[2]
            nheads = scale.shape[2]
            heads_per_group = nheads // ngroups
            scale_g = scale.reshape(
                scale.shape[0], scale.shape[1], ngroups, heads_per_group,
            ).mean(dim=-1)  # (b, l, ngroups)
            B = B * scale_g.unsqueeze(-1)

        # --- RoPE on B and C along state dimension ---
        if angles is not None:
            dt_for_angles = F.softplus(dd_dt + dt_bias)  # (b, l, h)
            angle_dt = angles * dt_for_angles.unsqueeze(-1)
            angles_cumsum = torch.cumsum(angle_dt, dim=1)
            B = _apply_rope_on_state_dim(B, angles_cumsum, rope_fraction)
            C = _apply_rope_on_state_dim(C, angles_cumsum, rope_fraction)

        return B, C


class NoConvMambaMixerSubmodules:
    """Module specs for the input and output linear layers."""

    def __init__(self, in_proj=None, out_proj=None):
        self.in_proj = in_proj
        self.out_proj = out_proj


class NoConvMambaMixer(MegatronModule):
    """Megatron-native Mamba mixer without conv1d.

    This is the Megatron-parallel equivalent of running a Mamba SSM block with
    no causal convolution preprocessing, matching the architectural choice made
    by Mamba3 (which uses RoPE instead of conv1d for positional information).

    The mixer preserves full compatibility with Megatron's:
      - Tensor-model parallelism (TP) via ``ColumnParallelLinear`` / ``RowParallelLinear``
      - Sequence parallelism (the input arrives seq-partitioned)
      - ``TransformerConfig`` knobs (``mamba_state_dim``, ``mamba_head_dim``, etc.)

    It does NOT support context parallelism (CP > 1) because the upstream CP
    implementation is tightly coupled to the conv1d weight layout.
    Use ``cp_size == 1``.

    Args:
        config: Megatron ``TransformerConfig``.
        submodules: Module specs for ``in_proj`` and ``out_proj``.
        d_model: Hidden size of the model.
        expand: Expansion factor for the inner dimension.
        D_has_hdim: Whether the D skip parameter is per-hidden-dim (True)
            or per-head (False).
        rmsnorm: Whether to apply RMSNormGated after the SSM.
        norm_before_gate: Whether to normalise before gating.
        dt_min / dt_max / dt_init_floor: dt bias initialisation range.
        A_init_range: Uniform init range for A.
        bias: Whether the linear layers have bias.
        chunk_size: Chunk size for the SSM kernel.
        layer_number: 1-based layer index in the stack.
        pg_collection: Required process-group collection (must have TP).
        pp_layer_offset: Pipeline-parallel layer offset.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: NoConvMambaMixerSubmodules,
        d_model: int,
        expand: int = 2,
        A_init_range: Tuple[float, float] = (1, 16),
        D_has_hdim: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        chunk_size: int = 128,
        layer_number: int | None = None,
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: int = 0,
    ):
        if mamba_chunk_scan_combined is None:
            raise ImportError(
                "mamba-ssm is required for NoConvMambaMixer. "
                "Install with: pip install mamba-ssm"
            )
        if rearrange is None:
            raise ImportError("einops is required for NoConvMambaMixer")

        super().__init__(config)
        self.config = config
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.chunk_size = chunk_size
        self.layer_number = layer_number
        self.pp_layer_offset = pp_layer_offset

        assert pg_collection is not None, "pg_collection must be provided"
        self.pg_collection = pg_collection

        self.d_state = self.config.mamba_state_dim
        self.headdim = self.config.mamba_head_dim
        self.ngroups = self.config.mamba_num_groups

        assert self.d_state is not None and self.d_state > 0
        assert self.headdim is not None and self.headdim > 0
        assert self.ngroups is not None and self.ngroups > 0

        if self.config.mamba_num_heads is not None:
            self.nheads = self.config.mamba_num_heads
            self.d_inner = self.nheads * self.headdim
        else:
            assert self.d_inner % self.headdim == 0
            self.nheads = self.d_inner // self.headdim

        tp_size = self.pg_collection.tp.size()
        assert self.nheads % tp_size == 0, "nheads must be divisible by tp_size"
        self.nheads_local = self.nheads // tp_size
        self.d_inner_local = self.d_inner // tp_size
        assert self.ngroups % tp_size == 0, "ngroups must be divisible by tp_size"
        self.ngroups_local = self.ngroups // tp_size
        assert self.nheads % self.ngroups == 0

        assert not bias
        assert not self.norm_before_gate

        # -- in_proj: [z, x, B, C, dt] packed into one ColumnParallelLinear --
        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            self.d_inner * 2 + 2 * self.ngroups * self.d_state + self.nheads,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )
        in_proj_partition_sizes = [
            self.d_inner_local,   # z
            self.d_inner_local,   # x
            self.ngroups_local * self.d_state,  # B
            self.ngroups_local * self.d_state,  # C
            self.nheads_local,    # dt
        ]
        setattr(self.in_proj.weight, "partition_sizes", in_proj_partition_sizes)

        # -- NO conv1d -- this is the whole point.
        self.act = nn.SiLU()

        # -- dt_bias, A_log, D: same as MambaMixer but without conv1d --
        with get_cuda_rng_tracker().fork():
            dt = torch.exp(
                torch.rand(
                    self.nheads_local,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            setattr(self.dt_bias, "tensor_model_parallel", True)
            setattr(self.dt_bias, "partition_dim", 0)

            assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
            A = torch.empty(
                self.nheads_local, dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            if self.config.perform_initialization:
                A = A.uniform_(*A_init_range)
            self.A_log = nn.Parameter(torch.log(A))
            setattr(self.A_log, "tensor_model_parallel", True)
            setattr(self.A_log, "partition_dim", 0)

        self.D = nn.Parameter(
            torch.ones(
                self.d_inner_local if self.D_has_hdim else self.nheads_local,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.D, "tensor_model_parallel", True)
        setattr(self.D, "partition_dim", 0)

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_inner_local,
                eps=1e-5,
                group_size=self.d_inner_local // self.ngroups_local,
                norm_before_gate=self.norm_before_gate,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
            setattr(self.norm.weight, "tensor_model_parallel", True)
            setattr(self.norm.weight, "partition_dim", 0)

        # -- out_proj --
        self.out_proj = build_module(
            submodules.out_proj,
            self.d_inner,
            self.d_model,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_context=None,
        *,
        inference_params=None,
        packed_seq_params=None,
    ):
        """Forward pass.

        Args:
            hidden_states: ``(seq, batch, hidden)`` tensor.

        Returns:
            ``(out, out_bias)`` tuple, same shape as ``hidden_states``.
        """
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError(
                "NoConvMambaMixer does not support Megatron inference paths yet"
            )
        if packed_seq_params is not None:
            raise NotImplementedError(
                "NoConvMambaMixer does not support packed sequences yet"
            )

        # in_proj: (seq, batch, hidden) -> (seq, batch, proj_dim)
        zxBCdt, _ = self.in_proj(hidden_states)

        # Training path: use mamba_chunk_scan_combined directly (no conv)
        y = self._ssm_noconv(zxBCdt)

        out, out_bias = self.out_proj(y)
        return out, out_bias

    def _ssm_noconv(self, zxBCdt: torch.Tensor) -> torch.Tensor:
        """SSM computation without conv1d preprocessing.

        Instead of the fused kernel which bakes in conv1d, we:
          1. Split ``[z, x, B, C, dt]`` manually.
          2. Apply SiLU activation to ``x`` (replaces conv1d + SiLU).
          3. Call ``mamba_chunk_scan_combined`` with pre-split tensors.
        """
        # transpose: (seq, batch, proj) -> (batch, seq, proj)
        zxBCdt = rearrange(zxBCdt, "l b d -> b l d").contiguous()

        A = -torch.exp(self.A_log.float())

        z, x, B, C, dt = torch.split(
            zxBCdt,
            [
                self.d_inner_local,
                self.d_inner_local,
                self.ngroups_local * self.d_state,
                self.ngroups_local * self.d_state,
                self.nheads_local,
            ],
            dim=-1,
        )

        # --- This is where conv1d + SiLU would be in MambaMixer ---
        # We apply SiLU directly to x (no conv1d).
        # B and C pass through without activation, matching Mamba3's approach
        # where in_proj outputs are used directly.
        x = self.act(x)

        # Reshape for SSM kernel
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim).contiguous()
        dt = dt.contiguous()
        B = rearrange(B, "b l (g n) -> b l g n", n=self.d_state).contiguous()
        C = rearrange(C, "b l (g n) -> b l g n", n=self.d_state).contiguous()
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim).contiguous()

        D = (
            rearrange(self.D.float(), "(h p) -> h p", p=self.headdim)
            if self.D_has_hdim
            else self.D
        )

        y = mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            self.chunk_size,
            D=D,
            z=z if not self.rmsnorm else None,
            dt_bias=self.dt_bias.float(),
            dt_softplus=True,
            return_final_states=False,
        )

        # Back to (seq, batch, inner)
        y = rearrange(y, "b l h p -> l b (h p)").contiguous()

        if self.rmsnorm:
            z = rearrange(z, "b l h p -> l b (h p)").contiguous()
            y = self.norm(y, z)

        return y

    def mamba_state_shapes_per_request(self):
        """Return per-request state shapes for inference pre-allocation."""
        raise NotImplementedError(
            "NoConvMambaMixer does not support Megatron inference cache shapes yet"
        )


class Mamba3NoConvMixer(Mamba3ScanMixin, NoConvMambaMixer):
    """NoConvMambaMixer with all four Mamba3 features.

    Adds data-dependent A, trapezoidal discretisation, complex RoPE on B/C,
    and QK-normalisation, all as pre-processing before calling the unchanged
    mamba_chunk_scan_combined kernel.

    The in_proj output is [z, x, B, C, dd_dt, dd_A, trap, angles] instead of
    the base [z, x, B, C, dt].
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: NoConvMambaMixerSubmodules,
        d_model: int,
        expand: int = 2,
        A_init_range: Tuple[float, float] = (1, 16),
        D_has_hdim: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        bias: bool = False,
        chunk_size: int = 64,
        rope_fraction: float = 0.5,
        layer_number: int | None = None,
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: int = 0,
    ):
        self._rope_fraction = rope_fraction
        self._A_floor = A_floor

        d_state = config.mamba_state_dim
        assert rope_fraction in (0.5, 1.0)
        split_size = int(d_state * rope_fraction)
        if split_size % 2 != 0:
            split_size -= 1
        self._n_rope_angles = split_size // 2
        assert self._n_rope_angles > 0

        super().__init__(
            config, submodules, d_model,
            expand=expand, A_init_range=A_init_range, D_has_hdim=D_has_hdim,
            rmsnorm=rmsnorm, norm_before_gate=norm_before_gate,
            dt_min=dt_min, dt_max=dt_max, dt_init_floor=dt_init_floor,
            bias=bias, chunk_size=chunk_size, layer_number=layer_number,
            pg_collection=pg_collection, pp_layer_offset=pp_layer_offset,
        )

        # Override in_proj to include dd_A, trap, angles
        new_out_dim = (
            self.d_inner * 2
            + 2 * self.ngroups * self.d_state
            + 3 * self.nheads
            + self._n_rope_angles
        )

        self.in_proj = build_module(
            submodules.in_proj, self.d_model, new_out_dim,
            config=self.config, init_method=self.config.init_method,
            gather_output=False, bias=bias, skip_bias_add=False,
            is_expert=False, tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )

        in_proj_partition_sizes = [
            self.d_inner_local,
            self.d_inner_local,
            self.ngroups_local * self.d_state,
            self.ngroups_local * self.d_state,
            self.nheads_local,
            self.nheads_local,
            self.nheads_local,
            self._n_rope_angles,
        ]
        setattr(self.in_proj.weight, "partition_sizes", in_proj_partition_sizes)

        # QK-norm and bias parameters
        self.m3_qknorm = True
        self.m3_bias = True

        with get_cuda_rng_tracker().fork():
            self.B_norm_weight = nn.Parameter(
                torch.ones(self.ngroups_local, self.d_state,
                           device=torch.cuda.current_device()))
            self.C_norm_weight = nn.Parameter(
                torch.ones(self.ngroups_local, self.d_state,
                           device=torch.cuda.current_device()))
            self.B_bias = nn.Parameter(
                torch.ones(self.ngroups_local, self.d_state,
                           device=torch.cuda.current_device()))
            self.C_bias = nn.Parameter(
                torch.ones(self.ngroups_local, self.d_state,
                           device=torch.cuda.current_device()))

    def _ssm_noconv(self, zxBCdt: torch.Tensor) -> torch.Tensor:
        """SSM computation with full Mamba3 feature pre-processing."""
        zxBCdt = rearrange(zxBCdt, "l b d -> b l d").contiguous()

        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdt,
            [
                self.d_inner_local, self.d_inner_local,
                self.ngroups_local * self.d_state,
                self.ngroups_local * self.d_state,
                self.nheads_local, self.nheads_local,
                self.nheads_local, self._n_rope_angles,
            ],
            dim=-1,
        )

        x = self.act(x)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim).contiguous()
        B = rearrange(B, "b l (g n) -> b l g n", n=self.d_state).contiguous()
        C = rearrange(C, "b l (g n) -> b l g n", n=self.d_state).contiguous()
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim).contiguous()

        angles = angles.unsqueeze(2).expand(-1, -1, self.nheads_local, -1)
        trap = rearrange(trap, "b l h -> b h l")

        D = (
            rearrange(self.D.float(), "(h p) -> h p", p=self.headdim)
            if self.D_has_hdim else self.D
        )

        B, C = self._preprocess_bc_mamba3(
            B, C,
            dt=F.softplus(dd_dt + self.dt_bias),
            dt_bias=self.dt_bias, dd_dt=dd_dt,
            trap=trap, angles=angles,
            rope_fraction=self._rope_fraction,
        )

        y = self._mamba3_scan(
            x, B, C, z,
            dd_dt=dd_dt, dd_A=dd_A,
            trap=None, angles=None,
            dt_bias=self.dt_bias.float(), D=D,
            chunk_size=self.chunk_size, rmsnorm=self.rmsnorm,
            A_floor=self._A_floor, rope_fraction=self._rope_fraction,
            return_final_states=False,
        )

        y = rearrange(y, "b l h p -> l b (h p)").contiguous()

        if self.rmsnorm:
            z = rearrange(z, "b l h p -> l b (h p)").contiguous()
            y = self.norm(y, z)

        return y
