"""Activation checkpointing for Mamba/M2RNN layers in MambaBlock.

Megatron's selective recompute only covers TransformerLayer sub-modules
(core_attn, moe_act, mlp, mla_up_proj, etc.). Mamba and M2RNN layers
inside MambaBlock.forward() are NEVER checkpointed, meaning their full
intermediate activations (in_proj output, SSM states, M2RNN scan tensors)
are held in memory for backward.

For NAM56R with 27 Mamba layers + 4 M2RNN layers, this consumes ~28+ GiB
per microbatch of uncheckpointed activations. With the interleaved 1F1B
pipeline schedule (PP=2 VPP=2), multiple microbatches' activations are
live simultaneously, pushing total activation memory above 120 GiB.

This monkey-patch wraps each non-TransformerLayer in MambaBlock.layers
with a thin wrapper that forwards to ``torch.utils.checkpoint.checkpoint()``,
so the existing MambaBlock.forward() loop calls them as normal but gets
automatic activation checkpointing.

Usage: set ``CPPMEGA_MAMBA_RECOMPUTE=1`` and call ``apply_mamba_recompute_patch()``
from the training shim (before training loop starts).
"""

from __future__ import annotations

import logging

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

logger = logging.getLogger(__name__)


class _CheckpointedMambaLayerWrapper(torch.nn.Module):
    """Wraps a Mamba/M2RNN layer to use activation checkpointing."""

    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.layer = layer
        # Proxy layer_number for MambaBlock's inner_quant_context
        self.layer_number = getattr(layer, "layer_number", 0)

    def forward(self, hidden_states, **kwargs):
        def _run(hs):
            return self.layer(hidden_states=hs, **kwargs)

        return torch_checkpoint(_run, hidden_states, use_reentrant=False)

    def __getattr__(self, name):
        if name in ("layer", "layer_number", "training"):
            return super().__getattr__(name)
        return getattr(self.layer, name)


def apply_mamba_recompute_patch() -> bool:
    """Wrap Mamba/M2RNN layers in all MambaBlock instances with checkpoint.

    This is called AFTER model construction but BEFORE training starts.
    It iterates over all MambaBlock.layers and wraps non-TransformerLayer
    entries with _CheckpointedMambaLayerWrapper.

    Returns True if any layers were wrapped.
    """
    try:
        from megatron.core.ssm.mamba_block import MambaBlock
        from megatron.core.transformer.transformer_layer import TransformerLayer
    except ImportError as e:
        logger.warning("mamba_recompute_patch: cannot import: %s", e)
        return False

    if getattr(MambaBlock, "_cppmega_mamba_recompute_patched", False):
        return True

    # Patch the __init__ to auto-wrap layers after construction
    _orig_init = MambaBlock.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)

        if not getattr(self.config, "recompute_granularity", None):
            return

        wrapped = 0
        new_layers = torch.nn.ModuleList()
        for layer in self.layers:
            if isinstance(layer, TransformerLayer):
                new_layers.append(layer)
            else:
                new_layers.append(_CheckpointedMambaLayerWrapper(layer))
                wrapped += 1
        self.layers = new_layers
        if wrapped > 0:
            print(
                f"[mamba_recompute_patch] Wrapped {wrapped} Mamba/M2RNN layers "
                f"with activation checkpointing"
            )

    MambaBlock.__init__ = _patched_init
    MambaBlock._cppmega_mamba_recompute_patched = True
    print("[mamba_recompute_patch] Patch installed (will wrap layers on construction)")
    return True
