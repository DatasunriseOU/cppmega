"""Shared MLA layer-spec helpers for cppmega NAM56R specs."""

from __future__ import annotations

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_inference_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention, MLASelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

# AbsorbedMLA from Megatron dev (PR #3193, merged Feb 2026).
# Absorbs K up-projection into Q, keeps KV compressed → MQA-style attention.
# Saves ~2-3 GiB per DSA layer by avoiding full K/V decompression.
try:
    from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
        AbsorbedMLASelfAttention,
        AbsorbedMLASelfAttentionSubmodules,
    )

    _HAS_ABSORBED_MLA = True
except ImportError:
    _HAS_ABSORBED_MLA = False


class CppMegaMLASelfAttentionAdapter(MLASelfAttention):
    def __init__(self, *args, pp_layer_offset=None, **kwargs):
        del pp_layer_offset
        super().__init__(*args, **kwargs)

    def forward(self, *args, rotary_pos_emb=None, **kwargs):
        del rotary_pos_emb
        return super().forward(*args, rotary_pos_emb=None, **kwargs)


class CppMegaFusedMLASelfAttentionAdapter(FusedMLASelfAttention):
    def __init__(self, *args, pp_layer_offset=None, **kwargs):
        del pp_layer_offset
        super().__init__(*args, **kwargs)

    def forward(self, *args, rotary_pos_emb=None, **kwargs):
        del rotary_pos_emb
        return super().forward(*args, rotary_pos_emb=None, **kwargs)


if _HAS_ABSORBED_MLA:

    class CppMegaAbsorbedMLASelfAttentionAdapter(AbsorbedMLASelfAttention):
        def __init__(self, *args, pp_layer_offset=None, **kwargs):
            del pp_layer_offset
            super().__init__(*args, **kwargs)

        def forward(self, *args, rotary_pos_emb=None, **kwargs):
            del rotary_pos_emb
            return super().forward(*args, rotary_pos_emb=None, **kwargs)


def adapt_mla_self_attention_spec(self_attention_spec):
    if isinstance(self_attention_spec, ModuleSpec):
        if self_attention_spec.module is MLASelfAttention:
            return ModuleSpec(
                module=CppMegaMLASelfAttentionAdapter,
                params=self_attention_spec.params,
                submodules=self_attention_spec.submodules,
            )
        if self_attention_spec.module is FusedMLASelfAttention:
            return ModuleSpec(
                module=CppMegaFusedMLASelfAttentionAdapter,
                params=self_attention_spec.params,
                submodules=self_attention_spec.submodules,
            )
        if _HAS_ABSORBED_MLA and self_attention_spec.module is AbsorbedMLASelfAttention:
            return ModuleSpec(
                module=CppMegaAbsorbedMLASelfAttentionAdapter,
                params=self_attention_spec.params,
                submodules=self_attention_spec.submodules,
            )
    return self_attention_spec


def _get_gpt_attention_submodules(config):
    if config.transformer_impl == "inference_optimized":
        return get_gpt_layer_with_inference_submodules(
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=True,
            qk_l2_norm=config.qk_l2_norm,
        )
    return get_gpt_layer_with_transformer_engine_submodules(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=True,
        qk_l2_norm=config.qk_l2_norm,
        use_te_op_fuser=False,
        use_kitchen=getattr(config, "use_kitchen", False),
        use_te_activation_func=getattr(config, "use_te_activation_func", False),
        use_kitchen_attention=getattr(config, "use_kitchen_attention", False),
        kitchen_attention_backend=getattr(config, "kitchen_attention_backend", "sdpa"),
        mla_down_proj_fusion=getattr(config, "mla_down_proj_fusion", False),
    )


def build_attention_layer_spec_from_self_attention_spec(config, self_attention_spec) -> ModuleSpec:
    gpt_submodules = _get_gpt_attention_submodules(config)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=gpt_submodules.input_layernorm,
            self_attention=adapt_mla_self_attention_spec(self_attention_spec),
            self_attn_bda=gpt_submodules.self_attn_bda,
            pre_mlp_layernorm=IdentityOp,
            mlp=IdentityOp,
            mlp_bda=IdentityFuncOp,
        ),
    )



def build_mla_attention_layer_spec(config) -> ModuleSpec:
    gpt_submodules = _get_gpt_attention_submodules(config)
    return build_attention_layer_spec_from_self_attention_spec(config, gpt_submodules.self_attention)
