"""Author Mamba3 stack spec: CppMegaMamba3TE (Author kernels) + upstream TE submodules.

Uses CppMegaMamba3TE as the Mamba mixer — this is the REAL Mamba3 with:
  - Trapezoidal discretization (mamba3_siso_combined Triton kernel)
  - QK-Norm + learnable B/C bias
  - Complex RoPE on B/C
  - Data-dependent A
  - MIMO R=4 (mamba3_mimo_combined TileLang kernel, optional)

All other submodules (attention, MoE, norms) use upstream TE-optimized layers.

Requires --cuda-graph-impl local (not transformer_engine) for CUDA graph compat.
For MIMO, set TILELANG_EXECUTION_BACKEND=nvrtc and patch TransformerConfig with
cppmega_mamba3_is_mimo=True.
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

from cppmega.megatron.mamba3_te_mixer import CppMegaMamba3TE


def _build():
    upstream = mamba_stack_spec.submodules
    upstream_mamba_sub = upstream.mamba_layer.submodules
    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    norm=upstream_mamba_sub.norm,
                    mixer=ModuleSpec(
                        module=CppMegaMamba3TE,
                        submodules=MambaMixerSubmodules(
                            in_proj=TELayerNormColumnParallelLinear,
                            out_proj=TERowParallelLinear,
                        ),
                    ),
                    mamba_bda=upstream_mamba_sub.mamba_bda,
                ),
            ),
            gdn_layer=upstream.gdn_layer,
            attention_layer=upstream.attention_layer,
            mlp_layer=upstream.mlp_layer,
            moe_layer=upstream.moe_layer,
            mtp_block_spec=upstream.mtp_block_spec,
        ),
    )


cppmega_mamba3_author_stack_spec = _build()


def build_cppmega_mamba3_author_stack_spec(config=None):
    """Callable entry point for --spec."""
    return cppmega_mamba3_author_stack_spec
