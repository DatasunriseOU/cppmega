"""Mamba-3 TE stack spec: upstream TE submodules + CppMegaMamba3Mixer.

Replaces ONLY the Mamba mixer module with CppMegaMamba3Mixer (Mamba-3 features:
QK-Norm, learnable B/C bias, trapezoidal discretization) while keeping ALL
other submodules from the upstream mamba_stack_spec:

  - TE fused LayerNorm via IdentityOp + TELayerNormColumnParallelLinear
  - TE fused attention (TEDotProductAttention)
  - TE MoE (TEGroupedMLP)
  - TE norms for MoE (TENorm)
  - TE GatedDeltaNet, MLP, MTP

The key invariant: MambaLayerSubmodules.norm stays as IdentityOp (the default)
because the actual norm is fused into TELayerNormColumnParallelLinear inside
the mixer's in_proj.  The mixer ModuleSpec specifies CppMegaMamba3Mixer with
MambaMixerSubmodules(in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear).
"""

from __future__ import annotations

from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

from cppmega.megatron.mamba3_mixer import CppMegaMamba3Mixer


def _build():
    upstream = mamba_stack_spec.submodules
    upstream_mamba_sub = upstream.mamba_layer.submodules
    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    # norm stays as IdentityOp (the default) -- the real norm
                    # is fused into TELayerNormColumnParallelLinear in in_proj.
                    norm=upstream_mamba_sub.norm,
                    # Replace ONLY the mixer with CppMegaMamba3Mixer, keeping
                    # the same TE projection submodules as upstream.
                    mixer=ModuleSpec(
                        module=CppMegaMamba3Mixer,
                        submodules=MambaMixerSubmodules(
                            in_proj=TELayerNormColumnParallelLinear,
                            out_proj=TERowParallelLinear,
                        ),
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


# Module-level constant (Megatron's import_module expects a ModuleSpec, not a function)
cppmega_mamba3_te_stack_spec = _build()


def build_cppmega_mamba3_te_stack_spec(config=None):
    """Callable entry point for --spec. Returns the pre-built ModuleSpec."""
    return cppmega_mamba3_te_stack_spec
