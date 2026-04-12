"""Activation checkpointing for Mamba/M2RNN layers in MambaBlock.

Megatron's selective recompute only covers TransformerLayer sub-modules
(core_attn, moe_act, mlp, mla_up_proj, etc.). Mamba and M2RNN layers
inside MambaBlock.forward() are NEVER checkpointed, meaning their full
intermediate activations (in_proj output, SSM states, M2RNN scan tensors)
are held in memory for backward.

This monkey-patch wraps each non-TransformerLayer in MambaBlock.layers
with a thin wrapper that uses ``torch.utils.checkpoint.checkpoint()``
when NOT inside CUDA graph capture, and runs directly when capturing
(matching Megatron's ``CheckpointWithoutOutput`` / PR #3919 pattern).

Usage: set ``CPPMEGA_MAMBA_RECOMPUTE=1`` and call ``apply_mamba_recompute_patch()``
from the training shim (before training loop starts).
"""

from __future__ import annotations

import logging

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

logger = logging.getLogger(__name__)

# Import Megatron's CUDA graph state checkers (PR #3919 pattern).
# These are module-level booleans toggled by CudaGraphManager.
try:
    from megatron.core.transformer.cuda_graphs import (
        is_graph_capturing,
        is_graph_warmup,
    )
except ImportError:
    def is_graph_capturing():
        return False

    def is_graph_warmup():
        return False


class _CheckpointedMambaLayerWrapper(torch.nn.Module):
    """Wraps a Mamba/M2RNN layer to use activation checkpointing.

    During CUDA graph capture/warmup, runs the layer directly (no checkpoint)
    so the ops are recorded into the graph. During normal forward, uses
    torch.utils.checkpoint to discard activations and recompute in backward.
    """

    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.layer = layer
        self.layer_number = getattr(layer, "layer_number", 0)

    def forward(self, hidden_states, **kwargs):
        # During CUDA graph capture/warmup: run directly (PR #3919 pattern)
        if is_graph_capturing() or is_graph_warmup():
            return self.layer(hidden_states=hidden_states, **kwargs)

        # Normal forward: use activation checkpoint
        def _run(hs):
            return self.layer(hidden_states=hs, **kwargs)

        return torch_checkpoint(_run, hidden_states, use_reentrant=False)

    def __getattr__(self, name):
        if name in ("layer", "layer_number", "training"):
            return super().__getattr__(name)
        return getattr(self.layer, name)


def apply_mamba_recompute_patch() -> bool:
    """Wrap Mamba/M2RNN layers in all MambaBlock instances with checkpoint."""
    try:
        from megatron.core.transformer.transformer_layer import TransformerLayer
    except ImportError as e:
        logger.warning("mamba_recompute_patch: cannot import TransformerLayer: %s", e)
        return False

    try:
        from megatron.core.ssm.mamba_block import MambaBlock
    except ImportError:
        try:
            from megatron.core.ssm.mamba_block import MambaStack as MambaBlock
        except ImportError as e:
            logger.warning("mamba_recompute_patch: cannot import MambaBlock/MambaStack: %s", e)
            return False

    if getattr(MambaBlock, "_cppmega_mamba_recompute_patched", False):
        return True

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
                f"with activation checkpointing (CG-aware)"
            )

    MambaBlock.__init__ = _patched_init
    MambaBlock._cppmega_mamba_recompute_patched = True
    print("[mamba_recompute_patch] Patch installed (CG-aware: skips checkpoint during graph capture)")
    return True
