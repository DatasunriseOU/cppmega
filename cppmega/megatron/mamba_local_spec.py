"""Non-TE Mamba stack glue for the first cppmega H200 smoke.

This is intentionally narrow: it replaces the Transformer-Engine-bound
`mamba_stack_spec` with local Megatron builders only where the upstream stack
currently hard-fails without TE. It is a smoke-path compatibility shim, not a
fork of Megatron's training stack.
"""

from __future__ import annotations

from contextlib import nullcontext

from torch import nn

from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


class CppMegaLocalMambaStack(MambaStack):
    """Top-level Mamba stack with a local final norm for no-TE environments."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaStackSubmodules,
        pre_process: bool = True,
        layer_type_list: list[str] | None = None,
        pp_layer_offset: int = 0,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ) -> None:
        del device, dtype
        GraphableMegatronModule.__init__(self, config=config)
        self.pre_process = pre_process
        self.post_layer_norm = post_layer_norm
        self.post_process = post_process
        self.is_mtp_layer = is_mtp_layer

        assert pg_collection is not None, "pg_collection must be provided for MambaStack"

        self.pp_group = pg_collection.pp
        self.tp_group = pg_collection.tp
        self.input_tensor = None
        self.pg_collection = pg_collection

        assert layer_type_list is not None, (
            "layer_type_list must be provided. It should be pre-computed from "
            "--hybrid-layer-pattern by MambaModel."
        )
        self.layer_type_list = layer_type_list

        self.layers = nn.ModuleList()
        for i, layer_type in enumerate(self.layer_type_list):
            layer_number = i + 1 + pp_layer_offset
            if self.config.fp8:
                quant_init_context = get_fp8_context(self.config, i + pp_layer_offset, is_init=True)
            elif self.config.fp4:
                quant_init_context = get_fp4_context(self.config, i + pp_layer_offset, is_init=True)
            else:
                quant_init_context = nullcontext()
            with quant_init_context:
                if layer_type == LayerSymbols.MAMBA:
                    layer = build_module(
                        submodules.mamba_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pp_layer_offset=pp_layer_offset,
                        pg_collection=pg_collection,
                    )
                elif layer_type == LayerSymbols.ATTENTION:
                    layer = build_module(
                        submodules.attention_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif layer_type == LayerSymbols.MLP:
                    layer = build_module(
                        submodules.mlp_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                    )
                elif layer_type == LayerSymbols.MOE:
                    layer = build_module(
                        submodules.moe_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                    )
                elif layer_type == LayerSymbols.GDN:
                    layer = build_module(
                        submodules.gdn_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                    )
                else:
                    raise AssertionError("unexpected layer_type")
            self.layers.append(layer)

        self.num_layers_per_pipeline_rank = len(self.layers)

        if self.post_process and self.post_layer_norm:
            # Upstream hard-codes TENorm here; this keeps the stack usable
            # when TE is absent but torch RMSNorm is available.
            self.final_norm = WrappedTorchNorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

    @staticmethod
    def get_default_submodules(*, mamba_mixer_spec: ModuleSpec | None = None) -> MambaStackSubmodules:
        return MambaStackSubmodules(
            mamba_layer=build_cppmega_local_mamba_layer_spec(mamba_mixer_spec=mamba_mixer_spec),
            gdn_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    self_attention=ModuleSpec(
                        module=GatedDeltaNet,
                        submodules=GatedDeltaNetSubmodules(
                            in_proj=ColumnParallelLinear,
                            out_norm=WrappedTorchNorm,
                            out_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                ),
            ),
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=WrappedTorchNorm,
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                ),
            ),
            mlp_layer=ModuleSpec(
                module=MLPLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=WrappedTorchNorm,
                    mlp=ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(
                            linear_fc1=ColumnParallelLinear,
                            linear_fc2=RowParallelLinear,
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
            moe_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=WrappedTorchNorm,
                    mlp=_local_moe,
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
            mtp_block_spec=_local_mtp_block_spec,
        )


_local_moe = get_moe_module_spec(
    use_te=False,
    num_experts=8,
    moe_grouped_gemm=False,
)

_local_mtp_block_spec = ModuleSpec(
    module=MultiTokenPredictionBlock,
    submodules=MultiTokenPredictionBlockSubmodules(
        layer_specs=[
            ModuleSpec(
                module=MultiTokenPredictionLayer,
                submodules=MultiTokenPredictionLayerSubmodules(
                    enorm=WrappedTorchNorm,
                    hnorm=WrappedTorchNorm,
                    eh_proj=ColumnParallelLinear,
                    mtp_model_layer=None,
                    layer_norm=WrappedTorchNorm,
                ),
            )
        ]
    ),
)


_default_mamba_mixer_spec = ModuleSpec(
    module=MambaMixer,
    submodules=MambaMixerSubmodules(
        in_proj=ColumnParallelLinear,
        out_proj=RowParallelLinear,
    ),
)


def build_cppmega_local_mamba_layer_spec(
    *,
    mamba_mixer_spec: ModuleSpec | None = None,
) -> ModuleSpec:
    return ModuleSpec(
        module=MambaLayer,
        submodules=MambaLayerSubmodules(
            norm=WrappedTorchNorm,
            mixer=_default_mamba_mixer_spec if mamba_mixer_spec is None else mamba_mixer_spec,
            mamba_bda=get_bias_dropout_add,
        ),
    )


def build_cppmega_local_stack_spec(
    *,
    mamba_mixer_spec: ModuleSpec | None = None,
) -> ModuleSpec:
    return ModuleSpec(
        module=CppMegaLocalMambaStack,
        submodules=CppMegaLocalMambaStack.get_default_submodules(mamba_mixer_spec=mamba_mixer_spec),
    )


cppmega_mamba_stack_spec = build_cppmega_local_stack_spec()
