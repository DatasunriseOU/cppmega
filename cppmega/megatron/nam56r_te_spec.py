"""NAM56R stack spec using upstream TE submodules for maximum throughput.

This spec only overrides the Mamba mixer (to use Author Mamba3 or M²RNN)
while keeping ALL other submodules from the upstream mamba_stack_spec.
This ensures TE norms, TE attention, TE MoE all remain fused and fast.
"""

from __future__ import annotations

import os
from typing import Iterable

from torch import nn

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

from cppmega.megatron.author_mamba3_spec import AuthorMamba3Mixer
from cppmega.megatron.m2rnn_spec import CppMegaM2RNNMixer
from cppmega.megatron.nam56r_layout import (
    FULL_NAM56R_DEPTH,
    FULL_NAM56R_PATTERN,
    load_r_layer_indices,
)

# Try to import xma Triton M²RNN
try:
    from xma.layers.m2rnn import m2rnn as xma_m2rnn
    HAS_XMA_M2RNN = True
except ImportError:
    HAS_XMA_M2RNN = False


class CppMegaSelectiveMambaMixerTE(nn.Module):
    """Select Author Mamba3 or M²RNN per layer, preserving TE submodules."""

    def __init__(
        self,
        config,
        d_model: int,
        submodules=None,
        layer_number: int | None = None,
        pg_collection=None,
        pp_layer_offset: int = 0,
        r_layer_indices: Iterable[int] = (),
    ) -> None:
        super().__init__()
        indices = frozenset(int(i) for i in r_layer_indices)
        layer_idx = 1 if layer_number is None else layer_number
        mixer_cls = CppMegaM2RNNMixer if layer_idx in indices else AuthorMamba3Mixer
        self.impl = mixer_cls(
            config=config,
            d_model=d_model,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
        )

    def forward(self, *args, **kwargs):
        return self.impl(*args, **kwargs)

    def mamba_state_shapes_per_request(self):
        return self.impl.mamba_state_shapes_per_request()


def build_cppmega_nam56r_te_stack_spec(config):
    """Build NAM56R spec using upstream TE submodules + Author Mamba3 mixer.

    This is the performance-optimized version that keeps:
    - TE fused LayerNorm (via IdentityOp + TELayerNormColumnParallelLinear)
    - TE fused attention (TEDotProductAttention)
    - TE MoE (TEGroupedMLP)
    - TE norms for MoE (TENorm)

    Only the Mamba mixer is replaced with Author Mamba3 / M²RNN.
    """
    r_layer_indices = load_r_layer_indices()
    upstream = mamba_stack_spec.submodules

    # Get the upstream Mamba layer submodules (TE norms, TE projections)
    upstream_mamba_sub = upstream.mamba_layer.submodules

    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    # Keep upstream TE norm (IdentityOp - norm fused into in_proj)
                    norm=upstream_mamba_sub.norm,
                    # Replace ONLY the mixer with our selective mixer
                    mixer=ModuleSpec(
                        module=CppMegaSelectiveMambaMixerTE,
                        params={"r_layer_indices": r_layer_indices},
                    ),
                    # Keep upstream bias-dropout-add
                    mamba_bda=upstream_mamba_sub.mamba_bda,
                ),
            ),
            # Keep ALL upstream layers unchanged (TE-optimized)
            gdn_layer=upstream.gdn_layer,
            attention_layer=upstream.attention_layer,
            mlp_layer=upstream.mlp_layer,
            moe_layer=upstream.moe_layer,
            mtp_block_spec=upstream.mtp_block_spec,
        ),
    )
