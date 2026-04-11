"""CppMegaMamba3TE -- Author Mamba3 scan kernels wrapped in TE linear layers.

Drop-in replacement for MambaMixer in the mamba_stack_spec.  Keeps TE fusion
(TELayerNormColumnParallelLinear for in_proj, TERowParallelLinear for out_proj)
while replacing the upstream conv1d + SSD scan path with the Author Mamba3
scan kernels (mamba3_siso_combined / mamba3_mimo_combined) that support:

  - Trapezoidal discretization
  - QK-Norm (RMSNormGated on B/C)
  - Learnable B/C bias
  - Complex RoPE on B/C
  - Data-dependent A
  - MIMO (multi-input multi-output)

The TE linear layers handle tensor-parallel sharding and fused LayerNorm; the
Mamba3 scan kernels handle the SSM computation proper.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.inference.contexts.base_context import BaseInferenceContext
from megatron.core.inference.contexts.static_context import deprecate_inference_params

from cppmega.features.mamba3 import build_author_mamba3_config

# NO FALLBACKS: Author Mamba3 kernels are REQUIRED.
# If mamba_ssm is not installed or kernels are missing, crash immediately.
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as mamba3_mimo_combined
from mamba_ssm.ops.triton.angle_cumsum import angle_dt
from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import (
    apply_rotary_qk_inference_fwd,
)

try:
    from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
except ImportError:
    mamba3_step_fn = None


class CppMegaMamba3TE(MegatronModule):
    """Author Mamba3 scan kernels with TE-fused linear projections.

    Matches the ``MambaMixer`` constructor and forward signatures so it can be
    used as a drop-in replacement in any ``mamba_stack_spec``.

    Parameters come from two sources:

    * **TE linear layers** (in_proj / out_proj) are constructed from the
      ``submodules`` spec exactly like upstream ``MambaMixer``, giving us
      ``TELayerNormColumnParallelLinear`` and ``TERowParallelLinear``.

    * **Mamba3 SSM parameters** (dt_bias, B_bias, C_bias, B_norm, C_norm, D,
      RoPE angles, MIMO projections, etc.) are constructed from the
      ``AuthorMamba3Config`` derived from ``TransformerConfig``.

    The forward path is:

    1. ``in_proj`` (TE-fused norm + column-parallel linear) produces the
       packed ``[z, x, B, C, dd_dt, dd_A, trap, angles]`` projection.
    2. The Author Mamba3 scan kernel (``mamba3_siso_combined`` or
       ``mamba3_mimo_combined``) runs the SSM computation.
    3. ``out_proj`` (TE row-parallel linear) produces the output.

    There is **no conv1d** -- Author Mamba3 replaces the conv+SSD path with
    its own data-dependent-A, trapezoidal, complex-RoPE scan.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaMixerSubmodules,
        d_model: int,
        # Absorbed kwargs from MambaMixer interface (unused by Mamba3 path)
        d_conv: int = 4,
        conv_init=None,
        expand: int = 2,
        A_init_range: tuple = (1, 16),
        D_has_hdim: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        chunk_size: int = 128,
        layer_number: int | None = None,
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: int = 0,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.d_model = d_model
        self.layer_number = layer_number
        self.pp_layer_offset = pp_layer_offset

        assert pg_collection is not None, "pg_collection must be provided for CppMegaMamba3TE"
        self.pg_collection = pg_collection

        # -----------------------------------------------------------------
        # Derive Mamba3 config from Megatron's TransformerConfig
        # -----------------------------------------------------------------
        m3cfg = build_author_mamba3_config(config, d_model=d_model)

        self.d_state = m3cfg.d_state
        self.headdim = m3cfg.headdim
        self.chunk_size = m3cfg.chunk_size
        self.A_floor = m3cfg.A_floor
        self.is_outproj_norm = m3cfg.is_outproj_norm
        self.is_mimo = m3cfg.is_mimo
        self.mimo_rank = m3cfg.mimo_rank if m3cfg.is_mimo else 1

        self.d_inner = m3cfg.d_model * m3cfg.expand
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.ngroups = m3cfg.ngroups
        self.num_bc_heads = m3cfg.ngroups  # Author Mamba3 naming

        if self.is_mimo:
            assert mamba3_mimo_combined is not None, (
                "MIMO mode requires mamba3_mimo kernels from mamba_ssm. "
                "Ensure TileLang is installed."
            )

        # ----- Tensor parallel sizes -----
        tp_size = pg_collection.tp.size()
        assert self.nheads % tp_size == 0, "nheads must be divisible by tp_size"
        self.nheads_local_tp = self.nheads // tp_size
        self.d_inner_local_tp = self.d_inner // tp_size
        assert self.ngroups % tp_size == 0, "ngroups must be divisible by tp_size"
        self.ngroups_local_tp = self.ngroups // tp_size

        # ----- RoPE configuration -----
        assert m3cfg.rope_fraction in (0.5, 1.0)
        self.rope_fraction = m3cfg.rope_fraction
        self.rotary_dim_divisor = int(2 / m3cfg.rope_fraction)
        self.split_tensor_size = int(self.d_state * m3cfg.rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # -----------------------------------------------------------------
        # in_proj output dimension:
        #   [z, x, B, C, dd_dt, dd_A, trap, angles]
        #
        # For TP sharding we split per-head / per-group components across
        # ranks.  The TE ColumnParallelLinear handles the split automatically.
        # -----------------------------------------------------------------
        # Per-TP-rank sizes of each component:
        z_size = self.d_inner_local_tp
        x_size = self.d_inner_local_tp
        B_size = self.ngroups_local_tp * self.d_state * self.mimo_rank
        C_size = self.ngroups_local_tp * self.d_state * self.mimo_rank
        dd_dt_size = self.nheads_local_tp
        dd_A_size = self.nheads_local_tp
        trap_size = self.nheads_local_tp
        angle_size = self.num_rope_angles  # broadcast, not TP-sharded per head

        # Total per-TP-rank output size (before TP concat)
        d_in_proj_local = (
            z_size + x_size + B_size + C_size
            + dd_dt_size + dd_A_size + trap_size + angle_size
        )
        # Global output size for the full (unsharded) projection
        d_in_proj_global = (
            2 * self.d_inner
            + 2 * self.ngroups * self.d_state * self.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )

        # Build TE in_proj (TELayerNormColumnParallelLinear)
        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            d_in_proj_global,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )
        # Store partition sizes for checkpoint resharding
        in_proj_partition_sizes = [
            z_size, x_size, B_size, C_size,
            dd_dt_size, dd_A_size, trap_size, angle_size,
        ]
        setattr(self.in_proj.weight, "partition_sizes", in_proj_partition_sizes)

        # Build TE out_proj (TERowParallelLinear)
        self.out_proj = build_module(
            submodules.out_proj,
            self.d_inner,
            self.d_model,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
        )

        # -----------------------------------------------------------------
        # SSM parameters (not sharded by TE -- per-head, live on each rank)
        # -----------------------------------------------------------------
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        dtype = config.params_dtype

        # dt_bias
        _dt = torch.exp(
            torch.rand(self.nheads_local_tp, device=device, dtype=torch.float32)
            * (math.log(m3cfg.dt_max) - math.log(m3cfg.dt_min))
            + math.log(m3cfg.dt_min)
        )
        _dt = torch.clamp(_dt, min=m3cfg.dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)

        # B and C biases (learnable, per-head per-state)
        self.B_bias = nn.Parameter(
            torch.ones(
                self.nheads_local_tp, self.mimo_rank, self.d_state,
                dtype=torch.float32, device=device,
            ),
            requires_grad=True,
        )
        self.C_bias = nn.Parameter(
            torch.ones(
                self.nheads_local_tp, self.mimo_rank, self.d_state,
                dtype=torch.float32, device=device,
            ),
            requires_grad=True,
        )

        # QK-Norm on B and C (RMSNormGated)
        assert RMSNormGated is not None, (
            "RMSNormGated from mamba_ssm is required for CppMegaMamba3TE"
        )
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, device=device, dtype=dtype)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, device=device, dtype=dtype)

        # MIMO projections
        if self.is_mimo:
            self.mimo_x = nn.Parameter(
                torch.ones(
                    self.nheads_local_tp, self.mimo_rank, self.headdim, device=device
                ) / self.mimo_rank,
                requires_grad=True,
            )
            self.mimo_z = nn.Parameter(
                torch.ones(
                    self.nheads_local_tp, self.mimo_rank, self.headdim, device=device
                ),
                requires_grad=True,
            )
            self.mimo_o = nn.Parameter(
                torch.ones(
                    self.nheads_local_tp, self.mimo_rank, self.headdim, device=device
                ) / self.mimo_rank,
                requires_grad=True,
            )

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads_local_tp, device=device))
        self.D._no_weight_decay = True
        setattr(self.D, "tensor_model_parallel", True)
        setattr(self.D, "partition_dim", 0)

        # Output-projection norm (optional, used with MIMO outproj_norm)
        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner_local_tp,
                eps=1e-5,
                norm_before_gate=True,
                group_size=self.headdim,
                device=device,
                dtype=dtype,
            )

        self.tp_group = pg_collection.tp

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_context=None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Args:
            hidden_states: (seq, batch, d_model) -- Megatron layout
            inference_context: Megatron inference context (or None during training)
            packed_seq_params: packed-sequence metadata (or None)

        Returns:
            (output, output_bias) where output is (seq, batch, d_model)
        """
        inference_context = deprecate_inference_params(
            inference_context, inference_params
        )

        # --- TE in_proj: fused LayerNorm + ColumnParallelLinear ---
        # Input is (seq, batch, d_model); output is (seq, batch, d_in_proj_local)
        zxBCdt_packed, _ = self.in_proj(hidden_states)

        # Transpose to batch-first for scan kernels: (seq, batch, d) -> (batch, seq, d)
        zxBCdt_packed = rearrange(zxBCdt_packed, "l b d -> b l d").contiguous()

        batch, seqlen, _ = zxBCdt_packed.shape

        # --- Split the packed projection ---
        z_size = self.d_inner_local_tp
        x_size = self.d_inner_local_tp
        B_size = self.ngroups_local_tp * self.d_state * self.mimo_rank
        C_size = self.ngroups_local_tp * self.d_state * self.mimo_rank
        dd_dt_size = self.nheads_local_tp
        dd_A_size = self.nheads_local_tp
        trap_size = self.nheads_local_tp

        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdt_packed,
            [
                z_size, x_size, B_size, C_size,
                dd_dt_size, dd_A_size, trap_size,
                self.num_rope_angles,
            ],
            dim=-1,
        )

        # Reshape for scan kernels
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(
            B, "b l (r g n) -> b l r g n",
            r=self.mimo_rank, g=self.ngroups_local_tp,
        )
        C = rearrange(
            C, "b l (r g n) -> b l r g n",
            r=self.mimo_rank, g=self.ngroups_local_tp,
        )
        trap = rearrange(trap, "b l h -> b h l")

        # --- Data-dependent A and dt (fp32 for Author kernel contract) ---
        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus((dd_dt + self.dt_bias).to(torch.float32))
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # --- Complex RoPE angles ---
        angles = angles.unsqueeze(-2).expand(
            -1, -1, self.nheads_local_tp, -1
        )  # (B, L, nheads, num_rope_angles)

        # --- QK-Norm on B and C ---
        B = self.B_norm(B)
        C = self.C_norm(C)

        # --- Packed sequence support ---
        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q

        # --- SSM scan ---
        if self.is_mimo:
            angles = angle_dt(angles, DT.transpose(-1, -2))
            # MIMO kernel: chunk_size*R must be <= 64 for shared memory.
            # B_bias/C_bias must be fp32 per kernel dtype contract.
            mimo_chunk = min(self.chunk_size, max(1, 64 // self.mimo_rank))
            # Author mamba3_mimo_combined kernel requires fp32 for all
            # learnable bias/projection tensors (D, B_bias, C_bias, mimo_*)
            y = mamba3_mimo_combined(
                Q=C,
                K=B,
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias.float(),
                K_bias=self.B_bias.float(),
                MIMO_V=self.mimo_x.float(),
                MIMO_Z=self.mimo_z.float(),
                MIMO_Out=self.mimo_o.float() if not self.is_outproj_norm else None,
                Angles=angles,
                D=self.D.float(),
                Z=z if not self.is_outproj_norm else None,
                chunk_size=mimo_chunk,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype,
                return_state=False,
                cu_seqlens=cu_seqlens,
            )
            if self.is_outproj_norm:
                z_f = torch.einsum(
                    "blhp,hrp->blrhp", z.float(), self.mimo_z
                )
                z_f = rearrange(z_f, "b l r h p -> b l r (h p)")
                y = rearrange(y, "b l r h p -> b l r (h p)").float()
                y = self.norm(y, z_f)
                y = rearrange(
                    y, "b l r (h p) -> b l r h p", p=self.headdim
                )
                y = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o)
            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            y = mamba3_siso_combined(
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                Input_States=None,
                return_final_states=False,
                cu_seqlens=cu_seqlens,
            )
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.is_outproj_norm:
                z_flat = rearrange(z, "b l h p -> b l (h p)")
                y = self.norm(y, z_flat)

        # Transpose back to Megatron layout: (batch, seq, d) -> (seq, batch, d)
        y = rearrange(y, "b l d -> l b d").contiguous()

        # --- TE out_proj: RowParallelLinear ---
        out, out_bias = self.out_proj(y.to(hidden_states.dtype))

        return out, out_bias

    # ------------------------------------------------------------------
    # Inference state helpers (for Megatron inference framework)
    # ------------------------------------------------------------------

    def mamba_state_shapes_per_request(self) -> Tuple[Tuple[int, ...], ...]:
        """Return state shapes for the Megatron inference cache allocator.

        Mamba3 uses four states: (angle_dt, ssm, k, v) instead of
        upstream's (conv, ssm).  We return them as a tuple of shapes.
        """
        angle_shape = (self.nheads_local_tp, self.num_rope_angles)
        ssm_shape = (self.nheads_local_tp, self.headdim, self.d_state)
        k_shape = (self.mimo_rank, self.nheads_local_tp, self.d_state)
        v_shape = (self.nheads_local_tp, self.headdim)
        return (angle_shape, ssm_shape, k_shape, v_shape)
