"""TP-aware replacement for ``AuthorMamba3Mixer`` with full 7/7 Mamba3 MIMO.

This is the *Megatron-native* pattern (mirroring
``megatron/core/ssm/mamba_mixer.py``), adapted for the Mamba3 packed
projection plus the MIMO-rank multiplier on B/C and the per-head data-
dependent ``dd_A`` / ``trap``.

What is sharded across TP and what is not
-----------------------------------------

* ``in_proj``  (TELayerNormColumnParallelLinear) packs
  ``[z, x, B, C, dd_dt, dd_A, trap]`` and is column-parallel sharded
  along the head/group axes.  Each TP rank receives only its local heads
  and groups; partition_sizes is annotated per-component to keep the
  distributed-checkpoint planner from concatenating heterogeneous blocks
  contiguously.
* ``out_proj`` (TERowParallelLinear) takes the local-d_inner output
  from the SSM kernel and all-reduces back to the full d_model.
* ``angle_proj`` is a SEPARATE plain ``nn.Linear`` whose weight is
  **replicated** across TP ranks (NOT marked tensor_model_parallel).
  Mamba3 broadcasts the same per-token angles to every head via
  ``angles.unsqueeze(-2).expand(-1, -1, nheads, -1)``; if we sliced this
  tensor across TP ranks each rank would rotate its local heads with a
  different angle.  Replicating the projection guarantees bit-identical
  angles on every rank.  This is the only structural difference from
  the upstream ``MambaMixer`` TP pattern.
* Per-head parameters (``dt_bias``, ``A_log``, ``D``, ``B_bias``,
  ``C_bias``, ``mimo_x``, ``mimo_z``, ``mimo_o``) are stored at
  ``nheads_local_tp`` and tagged ``tensor_model_parallel=True,
  partition_dim=0``.

The TileLang ``cppmega_tilelang_mimo_combined`` kernel is shape-agnostic
and consumes the local shards directly.

Parity contract for the unit test
---------------------------------

To get bit-exact TP=1 vs TP=2 parity in the parity test we initialize
the per-head custom parameters from a *single* CPU generator seeded by
``layer_number`` and slice the full tensor; rank 0 gets ``[0:nh_loc]``,
rank 1 gets ``[nh_loc:2*nh_loc]``.  The TE linear layers handle their
own TP-aware RNG fork via ``get_cuda_rng_tracker``; provided the test
seeds the global tracker identically before init, their per-rank
weights line up after concatenation along the partition dim.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from cppmega.features.mamba3 import build_author_mamba3_config
from cppmega.megatron.mamba_local_spec import build_cppmega_local_stack_spec
from cppmega.megatron.tilelang_mimo_autograd import cppmega_tilelang_mimo_combined


# ---------------------------------------------------------------------------
# Small TP topology helpers (work with or without torch.distributed init)
# ---------------------------------------------------------------------------

def _group_world_size(group) -> int:
    if group is None:
        return 1
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


def _group_rank(group) -> int:
    if group is None:
        return 0
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank(group=group)


def _assert_divisible(name: str, value: int, tp_size: int) -> int:
    if value % tp_size != 0:
        raise ValueError(
            f"CppmegaMamba3TPMixer: {name}={value} is not evenly divisible by "
            f"tp_world_size={tp_size}; every TP rank must receive an integer "
            f"number of heads/groups"
        )
    return value // tp_size


# ---------------------------------------------------------------------------
# CppmegaMamba3TPMixer
# ---------------------------------------------------------------------------

class CppmegaMamba3TPMixer(MegatronModule):
    """TP-aware Author Mamba3 MIMO mixer (full 7/7 feature surface).

    Drop-in replacement for ``AuthorMamba3Mixer`` with the same
    ``(config, d_model, submodules, layer_number, pg_collection,
    pp_layer_offset)`` constructor signature, so it can be plugged into
    ``MambaLayer`` via a ``MambaMixerSubmodules`` spec containing
    ``in_proj=TELayerNormColumnParallelLinear`` and
    ``out_proj=TERowParallelLinear``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        d_model: int,
        submodules: MambaMixerSubmodules | None = None,
        layer_number: int | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: int = 0,
    ) -> None:
        del pp_layer_offset

        super().__init__(config)

        if pg_collection is None:
            raise ValueError("pg_collection must be provided for CppmegaMamba3TPMixer")
        if submodules is None or submodules.in_proj is None or submodules.out_proj is None:
            raise ValueError(
                "CppmegaMamba3TPMixer requires submodules with non-None in_proj/out_proj "
                "(use MambaMixerSubmodules(in_proj=TELayerNormColumnParallelLinear, "
                "out_proj=TERowParallelLinear))"
            )

        self.config = config
        self.d_model = d_model
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.tp_world_size = _group_world_size(self.tp_group)
        self.tp_rank = _group_rank(self.tp_group)

        cp_world_size = _group_world_size(getattr(pg_collection, "cp", None))
        if cp_world_size != 1:
            raise NotImplementedError(
                "CppmegaMamba3TPMixer currently supports context-parallel-size=1 only"
            )

        # ------------------------------------------------------------------
        # Mamba3 config -> structural dims
        # ------------------------------------------------------------------
        m3cfg = build_author_mamba3_config(config, d_model=d_model)
        self.d_state = m3cfg.d_state
        self.headdim = m3cfg.headdim
        self.chunk_size = m3cfg.chunk_size
        self.A_floor = m3cfg.A_floor
        self.is_outproj_norm = m3cfg.is_outproj_norm
        self.is_mimo = m3cfg.is_mimo
        self.mimo_rank = m3cfg.mimo_rank if m3cfg.is_mimo else 1

        if not self.is_mimo:
            raise NotImplementedError(
                "CppmegaMamba3TPMixer currently implements the MIMO path only "
                "(set cppmega_mamba3_is_mimo=True in the TransformerConfig)"
            )

        self.d_inner = m3cfg.d_model * m3cfg.expand
        if self.d_inner % self.headdim != 0:
            raise ValueError(
                f"d_inner={self.d_inner} must be divisible by headdim={self.headdim}"
            )
        self.nheads = self.d_inner // self.headdim
        self.ngroups = m3cfg.ngroups

        # ------------------------------------------------------------------
        # TP topology arithmetic (every count must divide cleanly)
        # ------------------------------------------------------------------
        self.nheads_local_tp = _assert_divisible("nheads", self.nheads, self.tp_world_size)
        self.ngroups_local_tp = _assert_divisible(
            "ngroups", self.ngroups, self.tp_world_size
        )
        self.d_inner_local_tp = self.nheads_local_tp * self.headdim  # == d_inner // tp
        assert self.d_inner_local_tp * self.tp_world_size == self.d_inner

        if self.nheads % self.ngroups != 0:
            raise ValueError(
                f"nheads={self.nheads} must be divisible by ngroups={self.ngroups}"
            )

        # ------------------------------------------------------------------
        # RoPE
        # ------------------------------------------------------------------
        if m3cfg.rope_fraction not in (0.5, 1.0):
            raise ValueError(f"unsupported rope_fraction={m3cfg.rope_fraction}")
        self.rope_fraction = m3cfg.rope_fraction
        self.rotary_dim_divisor = int(2 / m3cfg.rope_fraction)
        self.split_tensor_size = int(self.d_state * m3cfg.rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        if self.num_rope_angles <= 0:
            raise ValueError("num_rope_angles must be positive")

        # ------------------------------------------------------------------
        # Packed in_proj output dim (NO angles -- those go through angle_proj)
        # Order on the last axis: [z, x, B, C, dd_dt, dd_A, trap]
        # ------------------------------------------------------------------
        d_in_proj_full = (
            2 * self.d_inner
            + 2 * self.ngroups * self.d_state * self.mimo_rank
            + 3 * self.nheads
        )

        # ------------------------------------------------------------------
        # in_proj via build_module -- TE handles the per-rank split
        # ------------------------------------------------------------------
        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            d_in_proj_full,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )
        # Annotate per-component partition sizes so the dist-ckpt planner
        # interleaves correctly when resharding across different TP sizes.
        in_proj_partition_sizes = [
            self.d_inner_local_tp,                                       # z
            self.d_inner_local_tp,                                       # x
            self.ngroups_local_tp * self.d_state * self.mimo_rank,       # B
            self.ngroups_local_tp * self.d_state * self.mimo_rank,       # C
            self.nheads_local_tp,                                        # dd_dt
            self.nheads_local_tp,                                        # dd_A
            self.nheads_local_tp,                                        # trap
        ]
        if hasattr(self.in_proj, "weight") and self.in_proj.weight is not None:
            setattr(self.in_proj.weight, "partition_sizes", in_proj_partition_sizes)

        # ------------------------------------------------------------------
        # angle_proj -- REPLICATED across TP ranks (NOT tp-sharded).
        # Each rank computes the same angles from the same hidden_states
        # and applies them to its local heads.
        # ------------------------------------------------------------------
        device = None
        if torch.cuda.is_available() and not getattr(config, "use_cpu_initialization", False):
            device = torch.cuda.current_device()
        params_dtype = config.params_dtype

        self.angle_proj = nn.Linear(
            self.d_model,
            self.num_rope_angles,
            bias=False,
            device=device,
            dtype=params_dtype,
        )
        # Mark as NOT tensor-parallel: every rank keeps an identical full copy.
        setattr(self.angle_proj.weight, "tensor_model_parallel", False)
        # Initialize from the same deterministic generator as the per-head
        # params so TP=1 and TP=2 produce bit-identical angle_proj weights
        # when the parity test seeds them with the same layer_number.
        self._init_angle_proj_(layer_number)

        # ------------------------------------------------------------------
        # Per-head / per-group parameters at LOCAL shard size.
        #
        # We initialise the FULL tensor with a CPU generator seeded by
        # ``layer_number`` and slice the local rows.  This is the only way
        # to guarantee bit-exact rank-0||rank-1 == TP=1 parity for the
        # custom Mamba3 init formulas.
        # ------------------------------------------------------------------
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1337 + 100003 * (layer_number or 0))

        # dt_bias: per-head, init via inverse-softplus of uniform[dt_min,dt_max]
        full_dt = torch.exp(
            torch.rand(self.nheads, generator=gen, dtype=torch.float32)
            * (math.log(m3cfg.dt_max) - math.log(m3cfg.dt_min))
            + math.log(m3cfg.dt_min)
        ).clamp(min=m3cfg.dt_init_floor)
        full_dt_bias = full_dt + torch.log(-torch.expm1(-full_dt))
        local_dt_bias = full_dt_bias[
            self.tp_rank * self.nheads_local_tp : (self.tp_rank + 1) * self.nheads_local_tp
        ].clone()
        if device is not None:
            local_dt_bias = local_dt_bias.to(device=device)
        self.dt_bias = nn.Parameter(local_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)

        # B_bias / C_bias: ones-initialised, per-head, mimo_rank, d_state, fp32
        bias_shape_local = (self.nheads_local_tp, self.mimo_rank, self.d_state)
        self.B_bias = nn.Parameter(
            torch.ones(bias_shape_local, dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.C_bias = nn.Parameter(
            torch.ones(bias_shape_local, dtype=torch.float32, device=device),
            requires_grad=True,
        )
        for p in (self.B_bias, self.C_bias):
            setattr(p, "tensor_model_parallel", True)
            setattr(p, "partition_dim", 0)

        # MIMO projection params (deterministic ones init, no RNG)
        mimo_shape_local = (self.nheads_local_tp, self.mimo_rank, self.headdim)
        self.mimo_x = nn.Parameter(
            torch.ones(mimo_shape_local, dtype=torch.float32, device=device) / self.mimo_rank,
            requires_grad=True,
        )
        self.mimo_z = nn.Parameter(
            torch.ones(mimo_shape_local, dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.mimo_o = nn.Parameter(
            torch.ones(mimo_shape_local, dtype=torch.float32, device=device) / self.mimo_rank,
            requires_grad=True,
        )
        for p in (self.mimo_x, self.mimo_z, self.mimo_o):
            setattr(p, "tensor_model_parallel", True)
            setattr(p, "partition_dim", 0)

        # D "skip" parameter: per-head ones, fp32
        self.D = nn.Parameter(
            torch.ones(self.nheads_local_tp, dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.D._no_weight_decay = True
        setattr(self.D, "tensor_model_parallel", True)
        setattr(self.D, "partition_dim", 0)

        # ------------------------------------------------------------------
        # B/C RMSNormGated -- operates only over the d_state axis, no
        # cross-rank communication needed.
        # ------------------------------------------------------------------
        from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
        self.B_norm = RMSNormGated(
            self.d_state, eps=1e-5, device=device, dtype=params_dtype,
        )
        self.C_norm = RMSNormGated(
            self.d_state, eps=1e-5, device=device, dtype=params_dtype,
        )

        # Optional outproj normalisation (used only with is_outproj_norm)
        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner_local_tp,
                eps=1e-5,
                norm_before_gate=True,
                group_size=self.headdim,
                device=device,
                dtype=params_dtype,
            )

        # ------------------------------------------------------------------
        # out_proj via build_module -- TE row-parallel linear (allreduce)
        # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # angle_proj initialisation -- needs the same value on every rank
    # ------------------------------------------------------------------
    def _init_angle_proj_(self, layer_number: int | None) -> None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(7919 + 100003 * (layer_number or 0))
        # Initialise on CPU first so the tensor is identical regardless of
        # which device is current; cast/move afterwards.
        w = torch.empty(
            self.angle_proj.weight.shape,
            dtype=torch.float32,
            device="cpu",
        )
        # Same scale as torch.nn.Linear default reset_parameters: kaiming_uniform_
        bound = 1.0 / math.sqrt(self.d_model)
        w.uniform_(-bound, bound, generator=gen)
        with torch.no_grad():
            self.angle_proj.weight.copy_(
                w.to(
                    dtype=self.angle_proj.weight.dtype,
                    device=self.angle_proj.weight.device,
                )
            )

    # ==================================================================
    # Forward
    # ==================================================================
    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_context=None,
        *,
        inference_params=None,
        packed_seq_params=None,
    ):
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError(
                "CppmegaMamba3TPMixer does not support Megatron inference paths yet"
            )
        if packed_seq_params is not None:
            raise NotImplementedError(
                "CppmegaMamba3TPMixer does not support packed sequences yet"
            )
        if hidden_states.ndim != 3:
            raise ValueError(
                "CppmegaMamba3TPMixer expects hidden_states shaped [seq, batch, hidden]"
            )

        # ------------------------------------------------------------------
        # 1) angle_proj on the (possibly sequence-parallel) hidden_states.
        #    With SP, hidden_states is sharded as (L/tp, B, H) and
        #    angle_proj's local output is (L/tp, B, num_rope_angles).  We
        #    must all-gather along the sequence axis so the SSM kernel
        #    sees the full sequence.  Without SP, hidden_states is the
        #    full (L, B, H) and no gather is needed.
        # ------------------------------------------------------------------
        angles_raw = self.angle_proj(hidden_states)
        if (
            getattr(self.config, "sequence_parallel", False)
            and self.tp_world_size > 1
        ):
            # Gather along sequence axis (dim 0) so every rank sees full L.
            from megatron.core.tensor_parallel.mappings import (
                gather_from_sequence_parallel_region,
            )
            angles_raw = gather_from_sequence_parallel_region(
                angles_raw, group=self.tp_group,
            )

        # ------------------------------------------------------------------
        # 2) TE in_proj: fused LayerNorm + ColumnParallelLinear, sequence
        #    first; output is already sharded along the head/group axis.
        # ------------------------------------------------------------------
        zxBCdt_packed, _ = self.in_proj(hidden_states)
        # Scan kernels want batch-first: (L, B, D) -> (B, L, D)
        zxBCdt_packed = rearrange(zxBCdt_packed, "l b d -> b l d").contiguous()

        # ------------------------------------------------------------------
        # 3) Split the packed projection into local-shard components.
        # ------------------------------------------------------------------
        z_size = self.d_inner_local_tp
        x_size = self.d_inner_local_tp
        B_size = self.ngroups_local_tp * self.d_state * self.mimo_rank
        C_size = self.ngroups_local_tp * self.d_state * self.mimo_rank
        dd_dt_size = self.nheads_local_tp
        dd_A_size = self.nheads_local_tp
        trap_size = self.nheads_local_tp

        z, x, B, C, dd_dt, dd_A, trap = torch.split(
            zxBCdt_packed,
            [z_size, x_size, B_size, C_size, dd_dt_size, dd_A_size, trap_size],
            dim=-1,
        )

        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        # NOTE: linear axis layout is (g, r, n) so the per-TP slice along
        # the linear axis is a clean subset of g (the *group* dimension).
        # Upstream Mamba3 uses (r, g, n) which would mix slices of r across
        # ranks; we deliberately deviate to make TP sharding work cleanly.
        B = rearrange(
            B, "b l (g r n) -> b l r g n",
            g=self.ngroups_local_tp, r=self.mimo_rank,
        )
        C = rearrange(
            C, "b l (g r n) -> b l r g n",
            g=self.ngroups_local_tp, r=self.mimo_rank,
        )
        trap = rearrange(trap, "b l h -> b h l")

        # ------------------------------------------------------------------
        # 4) Data-dependent A and dt -- match upstream Mamba3 dtypes EXACTLY
        #    (dd_A -> fp32 then softplus then clamp; dd_dt + dt_bias stays
        #    in dd_dt's dtype, then softplus, then ADT in fp32 from the
        #    multiplication).
        # ------------------------------------------------------------------
        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # ------------------------------------------------------------------
        # 5) Replicated angles -> broadcast to local nheads, then angle_dt.
        #    angles_raw is currently (L, B, num_rope_angles); rearrange to
        #    batch-first then expand the head axis.
        # ------------------------------------------------------------------
        angles = rearrange(angles_raw, "l b s -> b l s").contiguous()
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads_local_tp, -1)
        from mamba_ssm.ops.triton.angle_cumsum import angle_dt
        angles = angle_dt(angles, DT.transpose(-1, -2))

        # ------------------------------------------------------------------
        # 6) QK-Norm on B and C (per-state-axis, fully local)
        # ------------------------------------------------------------------
        B = self.B_norm(B)
        C = self.C_norm(C)

        # ------------------------------------------------------------------
        # 7) MIMO scan via cppmega TileLang autograd kernel
        # ------------------------------------------------------------------
        y = cppmega_tilelang_mimo_combined(
            Q=C,
            K=B,
            V=x,
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=self.C_bias,
            K_bias=self.B_bias,
            MIMO_V=self.mimo_x,
            MIMO_Z=self.mimo_z,
            MIMO_Out=self.mimo_o if not self.is_outproj_norm else None,
            Angles=angles,
            D=self.D,
            Z=z if not self.is_outproj_norm else None,
            chunk_size=self.chunk_size,
            rotary_dim_divisor=self.rotary_dim_divisor,
            dtype=x.dtype,
            return_state=False,
            cu_seqlens=None,
        )

        if self.is_outproj_norm:
            z_f = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
            z_f = rearrange(z_f, "b l r h p -> b l r (h p)")
            y = rearrange(y, "b l r h p -> b l r (h p)").float()
            y = self.norm(y, z_f)
            y = rearrange(y, "b l r (h p) -> b l r h p", p=self.headdim)
            y = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o)

        # ------------------------------------------------------------------
        # 8) Repack to (L, B, d_inner_local) and run TE out_proj which
        #    all-reduces across TP back to full d_model.
        # ------------------------------------------------------------------
        y = rearrange(y, "b l h p -> b l (h p)")
        y = rearrange(y, "b l d -> l b d").contiguous()
        out, _out_bias = self.out_proj(y.to(hidden_states.dtype))
        return out, None

    # ------------------------------------------------------------------
    # Inference cache shapes -- not yet supported
    # ------------------------------------------------------------------
    def mamba_state_shapes_per_request(self):
        raise NotImplementedError(
            "CppmegaMamba3TPMixer does not support Megatron inference cache shapes yet"
        )


# ---------------------------------------------------------------------------
# Stack spec helpers
# ---------------------------------------------------------------------------

def build_cppmega_mamba3_tp_stack_spec() -> ModuleSpec:
    """Return a stack spec that uses ``CppmegaMamba3TPMixer`` as the mamba mixer.

    Imports TE submodule classes lazily so that the module remains importable
    in CPU-only environments (used by tests that introspect the source).
    """
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
    return build_cppmega_local_stack_spec(
        mamba_mixer_spec=ModuleSpec(
            module=CppmegaMamba3TPMixer,
            submodules=MambaMixerSubmodules(
                in_proj=TELayerNormColumnParallelLinear,
                out_proj=TERowParallelLinear,
            ),
        )
    )
