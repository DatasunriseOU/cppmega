"""TE-fused LayerNorm + in_proj for Author Mamba3.

Replaces Mamba3's ``nn.Linear`` in_proj with ``TELayerNormColumnParallelLinear``
so that the preceding RMSNorm and the projection happen in a single fused TE
kernel.  The module computes the correct total output dimension from the
Author Mamba3 config and annotates ``partition_sizes`` on the weight so that
Megatron checkpoint resharding across different TP sizes works correctly.

Output layout (concatenated along the last dim, matching Author Mamba3 split):
    [z | x | B | C | dd_dt | dd_A | trap | angles]

TP sharding strategy:
    - z, x          : shard by head   (nheads splits into nheads/tp)
    - B, C          : shard by group  (ngroups splits into ngroups/tp)
    - dd_dt, dd_A, trap : shard by head (nheads/tp each)
    - angles        : replicated      (shared across all heads)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cppmega.features.mamba3.config import AuthorMamba3Config


def _compute_num_rope_angles(d_state: int, rope_fraction: float) -> int:
    """Reproduce Mamba3.__init__ logic for num_rope_angles."""
    split_tensor_size = int(d_state * rope_fraction)
    if split_tensor_size % 2 != 0:
        split_tensor_size -= 1
    num_rope_angles = split_tensor_size // 2
    assert num_rope_angles > 0, (
        f"num_rope_angles must be positive (d_state={d_state}, rope_fraction={rope_fraction})"
    )
    return num_rope_angles


@dataclass(frozen=True)
class Mamba3InProjDims:
    """Dimensions of each slice in the Mamba3 in_proj output.

    These are the *global* (pre-TP-sharding) sizes.
    """

    d_inner: int         # z and x are each this size
    d_bc: int            # B and C are each this size (d_state * ngroups * mimo_rank)
    nheads: int          # dd_dt, dd_A, trap are each this size
    num_rope_angles: int # angles

    @property
    def total(self) -> int:
        return (
            2 * self.d_inner           # z + x
            + 2 * self.d_bc            # B + C
            + 3 * self.nheads          # dd_dt + dd_A + trap
            + self.num_rope_angles     # angles
        )

    @property
    def split_sizes(self) -> list[int]:
        """The sizes list for torch.split(..., dim=-1), matching Author Mamba3."""
        return [
            self.d_inner,           # z
            self.d_inner,           # x
            self.d_bc,              # B
            self.d_bc,              # C
            self.nheads,            # dd_dt
            self.nheads,            # dd_A
            self.nheads,            # trap
            self.num_rope_angles,   # angles
        ]


def compute_mamba3_in_proj_dims(cfg: AuthorMamba3Config) -> Mamba3InProjDims:
    """Compute projection dimensions from an AuthorMamba3Config."""
    d_inner = cfg.d_model * cfg.expand
    assert d_inner % cfg.headdim == 0
    nheads = d_inner // cfg.headdim
    mimo_rank = cfg.mimo_rank if cfg.is_mimo else 1
    d_bc = cfg.d_state * cfg.ngroups * mimo_rank
    num_rope_angles = _compute_num_rope_angles(cfg.d_state, cfg.rope_fraction)
    return Mamba3InProjDims(
        d_inner=d_inner,
        d_bc=d_bc,
        nheads=nheads,
        num_rope_angles=num_rope_angles,
    )


def compute_mamba3_tp_partition_sizes(
    dims: Mamba3InProjDims,
    tp_size: int,
) -> list[int]:
    """Per-TP-rank block sizes along the output dim for checkpoint resharding.

    Each component is sharded independently:
      - z, x: by nheads (d_inner_local = d_inner / tp)
      - B, C: by ngroups (d_bc_local = d_bc / tp)
      - dd_dt, dd_A, trap: by nheads (nheads_local = nheads / tp)
      - angles: replicated (not TP-sharded)

    ``partition_sizes`` is set on the weight tensor so Megatron's checkpoint
    utilities know how to repartition across TP sizes.  For TP=1 this is
    just the global split_sizes.
    """
    assert dims.d_inner % tp_size == 0, (
        f"d_inner ({dims.d_inner}) must be divisible by tp_size ({tp_size})"
    )
    assert dims.nheads % tp_size == 0, (
        f"nheads ({dims.nheads}) must be divisible by tp_size ({tp_size})"
    )
    # ngroups divisibility is implicitly checked via d_bc
    # d_bc = d_state * ngroups * mimo_rank -- ngroups must divide by tp
    # We check the actual d_bc divisibility directly.
    assert dims.d_bc % tp_size == 0, (
        f"d_bc ({dims.d_bc}) must be divisible by tp_size ({tp_size})"
    )

    d_inner_local = dims.d_inner // tp_size
    d_bc_local = dims.d_bc // tp_size
    nheads_local = dims.nheads // tp_size

    return [
        d_inner_local,      # z
        d_inner_local,      # x
        d_bc_local,         # B
        d_bc_local,         # C
        nheads_local,       # dd_dt
        nheads_local,       # dd_A
        nheads_local,       # trap
        dims.num_rope_angles,  # angles (replicated)
    ]


def compute_mamba3_te_output_size(dims: Mamba3InProjDims, tp_size: int) -> int:
    """Total output_size to pass to TELayerNormColumnParallelLinear.

    TELayerNormColumnParallelLinear takes the *global* output size and
    internally divides by tp_size.  But angles are replicated, so we must
    account for that: the global output we declare must equal
    ``sum(local_partition_sizes) * tp_size`` only if everything is sharded
    uniformly.  Since angles are NOT sharded, we need a different approach.

    The actual per-rank output is ``sum(partition_sizes_local)``.
    TE divides ``output_size`` by ``tp_size`` to get the local size.
    So we set ``output_size = sum(partition_sizes_local) * tp_size``.

    This means TE will create a weight of shape ``[local_out, d_model]``
    where ``local_out = sum(partition_sizes_local)`` -- which is exactly
    what we want.
    """
    local_sizes = compute_mamba3_tp_partition_sizes(dims, tp_size)
    return sum(local_sizes) * tp_size


class CppMegaMamba3TEInProj(nn.Module):
    """Fused LayerNorm + in_proj for Author Mamba3 using TE.

    Wraps ``TELayerNormColumnParallelLinear`` with the correct output
    dimension for Mamba3's combined ``[z, x, B, C, dd_dt, dd_A, trap, angles]``
    projection, and sets ``partition_sizes`` on the weight for proper TP
    resharding.

    Usage::

        from cppmega.features.mamba3.config import build_author_mamba3_config

        author_cfg = build_author_mamba3_config(megatron_config, d_model=d_model)
        te_in_proj = CppMegaMamba3TEInProj(
            author_cfg=author_cfg,
            megatron_config=megatron_config,
            tp_group=pg_collection.tp,
        )

        # In forward:
        # hidden_states: (seq, batch, d_model)
        fused_out, _ = te_in_proj(hidden_states)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            fused_out, te_in_proj.split_sizes_local, dim=-1
        )
    """

    def __init__(
        self,
        author_cfg: AuthorMamba3Config,
        megatron_config,
        tp_group=None,
    ) -> None:
        super().__init__()

        self.dims = compute_mamba3_in_proj_dims(author_cfg)

        # Determine TP size
        if tp_group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            self.tp_size = torch.distributed.get_world_size(group=tp_group)
        else:
            self.tp_size = 1

        partition_sizes = compute_mamba3_tp_partition_sizes(self.dims, self.tp_size)
        self.split_sizes_local = partition_sizes

        # Compute the output_size to declare to TE.
        te_output_size = compute_mamba3_te_output_size(self.dims, self.tp_size)

        try:
            from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
        except ImportError as exc:
            raise ImportError(
                "TELayerNormColumnParallelLinear is not available. "
                "Ensure Transformer Engine and Megatron-Core are installed."
            ) from exc

        self.proj = TELayerNormColumnParallelLinear(
            input_size=author_cfg.d_model,
            output_size=te_output_size,
            config=megatron_config,
            init_method=megatron_config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=tp_group,
        )

        # Annotate partition_sizes for checkpoint resharding
        setattr(self.proj.weight, "partition_sizes", partition_sizes)

    def forward(self, hidden_states: torch.Tensor):
        """Forward: fused LayerNorm + Linear.

        Args:
            hidden_states: (seq, batch, d_model) if sequence_parallel,
                           or (batch, seq, d_model).

        Returns:
            (output, bias) where output has shape (..., sum(split_sizes_local))
            and bias is None (no bias in this projection).
        """
        return self.proj(hidden_states)
