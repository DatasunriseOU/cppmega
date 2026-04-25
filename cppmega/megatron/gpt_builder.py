"""Cppmega GPT builder with polymorphic embedding substitution."""

from __future__ import annotations

import megatron.legacy.model
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_decoder_layer_specs,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

from cppmega.megatron.custom_gpt_model import CppMegaGPTModel
from cppmega.megatron.deprecated_paths import require_deprecated_ack


def cppmega_gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0('building cppmega GPT model ...')
    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)
    if args.use_legacy_models:
        require_deprecated_ack(
            feature="--use-legacy-models in cppmega GPT builder",
            ack_env="CPPMEGA_I_UNDERSTAND_MEGATRON_LEGACY_GPT_MODEL_IS_DEPRECATED",
            replacement="Megatron Core GPTModel via --use-mcore-models",
            reason="The legacy Megatron model path bypasses cppmega/MCore feature hooks.",
        )
        return megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        use_te = args.transformer_impl == "transformer_engine"
        if args.experimental_attention_variant is not None:
            transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
                config=config, vp_stage=vp_stage
            )
        elif args.num_experts:
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config,
                use_transformer_engine=use_te,
                normalization=args.normalization,
                qk_l2_norm=args.qk_l2_norm,
                vp_stage=vp_stage,
            )
        elif args.heterogeneous_layers_config_path is not None:
            transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    config.num_moe_experts,
                    config.moe_grouped_gemm,
                    config.qk_layernorm,
                    config.multi_latent_attention,
                    config.experimental_attention_variant,
                    qk_l2_norm=config.qk_l2_norm,
                    use_kitchen=config.use_kitchen,
                    use_te_activation_func=config.use_te_activation_func,
                    use_kitchen_attention=config.use_kitchen_attention,
                    kitchen_attention_backend=config.kitchen_attention_backend,
                    mla_down_proj_fusion=getattr(config, "mla_down_proj_fusion", False),
                )
            elif config.transformer_impl == "inference_optimized":
                transformer_layer_spec = get_gpt_layer_with_inference_spec(
                    config.qk_layernorm,
                    config.multi_latent_attention,
                    qk_l2_norm=config.qk_l2_norm,
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    config.num_moe_experts,
                    config.moe_grouped_gemm,
                    config.qk_layernorm,
                    config.multi_latent_attention,
                    config.experimental_attention_variant,
                    normalization=config.normalization,
                    use_kitchen=config.use_kitchen,
                    use_kitchen_attention=config.use_kitchen_attention,
                    kitchen_attention_backend=config.kitchen_attention_backend,
                )

    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        use_te = args.transformer_impl == "transformer_engine"
        if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
            transformer_layer_spec_for_mtp = get_gpt_decoder_layer_specs(
                config,
                use_transformer_engine=use_te,
                normalization=args.normalization,
                qk_l2_norm=args.qk_l2_norm,
                vp_stage=vp_stage,
            )[-1]
        else:
            transformer_layer_spec_for_mtp = get_gpt_decoder_layer_specs(
                config,
                use_transformer_engine=use_te,
                normalization=args.normalization,
                qk_l2_norm=args.qk_l2_norm,
                vp_stage=vp_stage,
            )[-1]
        mtp_block_spec = get_gpt_mtp_block_spec(
            config,
            transformer_layer_spec_for_mtp,
            use_transformer_engine=use_te,
            vp_stage=vp_stage,
        )

    return CppMegaGPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )
