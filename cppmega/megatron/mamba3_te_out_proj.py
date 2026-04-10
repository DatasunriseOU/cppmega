"""TE-fused out_proj for Author Mamba3.

Replaces Mamba3's ``nn.Linear`` out_proj with ``TERowParallelLinear`` so the
output projection participates in Megatron's tensor-parallel all-reduce and
sequence-parallel scatter.

Handles two Mamba3 output norms:

1. ``is_outproj_norm=False`` (default): the upstream Mamba3 kernel fuses the
   SiLU gate (z) and mimo_o contraction internally, producing ``y`` of shape
   ``(batch, seq, d_inner)``.  We simply run ``out_proj(y)`` via TE.

2. ``is_outproj_norm=True``: the kernel returns pre-norm activations.  The
   author code applies ``RMSNormGated(d_inner, group_size=headdim,
   norm_before_gate=True)`` with the ``z`` gate *before* ``out_proj``.
   With TP>1 we must shard the norm weight across TP ranks (each rank holds
   ``d_inner_local = d_inner / tp_size`` channels).  We keep the author
   ``RMSNormGated`` but construct it with the local size and tag its weight
   for Megatron checkpoint resharding.

TP sharding strategy:
    - ``y`` arrives with its last dimension already local to the TP rank
      (``d_inner_local``).  ``TERowParallelLinear`` is constructed with
      ``input_is_parallel=True`` so it consumes the local shard directly.
    - The norm weight is sharded along dim 0 (one contiguous block per rank).
    - ``out_proj`` maps ``d_inner -> d_model`` with an all-reduce across TP
      ranks (or scatter for sequence-parallel).

Usage::

    te_out = CppMegaMamba3TEOutProj(
        author_cfg=author_cfg,
        megatron_config=megatron_config,
        tp_group=pg_collection.tp,
    )

    # In forward (after Mamba3 kernel):
    # y: (seq, batch, d_inner_local)  -- already TP-local
    # z: (seq, batch, d_inner_local)  -- only used when is_outproj_norm=True
    out, out_bias = te_out(y, z=z)
"""

from __future__ import annotations

import torch
from torch import nn

from cppmega.features.mamba3.config import AuthorMamba3Config


def _tp_world_size(tp_group) -> int:
    if tp_group is None:
        return 1
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=tp_group)


class CppMegaMamba3TEOutProj(nn.Module):
    """TE-fused output projection for Author Mamba3.

    Replaces:
      - ``self.norm``    (optional RMSNormGated, when is_outproj_norm=True)
      - ``self.out_proj`` (nn.Linear)

    with:
      - A TP-sharded RMSNormGated (when is_outproj_norm=True)
      - TERowParallelLinear (input_is_parallel=True)
    """

    def __init__(
        self,
        author_cfg: AuthorMamba3Config,
        megatron_config,
        tp_group=None,
    ) -> None:
        super().__init__()

        self.tp_size = _tp_world_size(tp_group)
        self.is_outproj_norm = author_cfg.is_outproj_norm

        d_inner = author_cfg.d_model * author_cfg.expand
        assert d_inner % author_cfg.headdim == 0
        nheads = d_inner // author_cfg.headdim
        self.d_inner = d_inner
        self.d_model = author_cfg.d_model
        self.headdim = author_cfg.headdim
        self.nheads = nheads

        # --- TP-local sizes ---
        assert nheads % self.tp_size == 0, (
            f"nheads ({nheads}) must be divisible by tp_size ({self.tp_size})"
        )
        self.d_inner_local = d_inner // self.tp_size
        self.nheads_local = nheads // self.tp_size

        # --- Optional gated RMSNorm (before out_proj) ---
        if self.is_outproj_norm:
            # The author Mamba3 creates:
            #   self.norm = RMSNormGated(
            #       d_inner, eps=1e-5, norm_before_gate=True,
            #       group_size=headdim, ...)
            #
            # With TP, we construct it for the *local* shard only.
            # The norm is element-wise within each head (group_size=headdim),
            # and heads are partitioned across TP ranks, so each rank can
            # independently normalize its local heads.
            try:
                from mamba_ssm.ops.triton.layernorm_gated import RMSNormGated
            except ImportError as exc:
                raise ImportError(
                    "RMSNormGated is required for is_outproj_norm=True. "
                    "Install mamba_ssm with Triton support."
                ) from exc

            device = None
            if torch.cuda.is_available() and not getattr(megatron_config, "use_cpu_initialization", False):
                device = torch.cuda.current_device()

            self.norm = RMSNormGated(
                self.d_inner_local,
                eps=1e-5,
                norm_before_gate=True,
                group_size=self.headdim,
                device=device,
                dtype=megatron_config.params_dtype,
            )

            # Tag norm weight for Megatron TP-aware checkpoint save/load.
            # The weight is sharded along dim 0 across TP ranks.
            setattr(self.norm.weight, "tensor_model_parallel", True)
            setattr(self.norm.weight, "partition_dim", 0)
        else:
            self.norm = None

        # --- TERowParallelLinear out_proj ---
        try:
            from megatron.core.extensions.transformer_engine import TERowParallelLinear
        except ImportError as exc:
            raise ImportError(
                "TERowParallelLinear is not available. "
                "Ensure Transformer Engine and Megatron-Core are installed."
            ) from exc

        self.out_proj = TERowParallelLinear(
            input_size=d_inner,
            output_size=author_cfg.d_model,
            config=megatron_config,
            init_method=megatron_config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=tp_group,
        )

    def forward(
        self,
        y: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply optional gated norm then TE out_proj.

        Args:
            y: SSM output, shape ``(seq, batch, d_inner_local)`` (seq-first,
               already TP-local along the last dim).
            z: Gate tensor, shape ``(seq, batch, d_inner_local)``.
               Required when ``is_outproj_norm=True``; ignored otherwise.

        Returns:
            ``(out, out_bias)`` where ``out`` has shape
            ``(seq, batch, d_model)`` after the TP all-reduce, and
            ``out_bias`` is None (no bias).
        """
        if self.is_outproj_norm:
            if z is None:
                raise ValueError(
                    "z (gate tensor) must be provided when is_outproj_norm=True"
                )
            # RMSNormGated: norm(y, z) = rms_norm(y) * silu(z)  (norm_before_gate=True)
            y = self.norm(y, z)

        # TERowParallelLinear: all-reduce across TP ranks
        out, out_bias = self.out_proj(y)
        return out, out_bias
