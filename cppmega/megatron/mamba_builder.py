"""Cppmega Mamba builder with polymorphic embedding substitution."""

from __future__ import annotations

from megatron.core.models.mamba import MambaModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import import_module
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from cppmega.megatron.custom_mamba_model import CppMegaMambaModel


def cppmega_mamba_builder(
    args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None
):
    print_rank_0("building cppmega MAMBA model ...")
    if config is None:
        config = core_transformer_config_from_args(args, TransformerConfig)
    assert args.use_legacy_models is False, "Mamba only supported in Mcore!"

    if args.spec is None:
        raise ValueError("cppmega_mamba_builder requires --spec")

    spec_or_factory = import_module(args.spec)
    mamba_stack_spec = spec_or_factory(config) if callable(spec_or_factory) else spec_or_factory

    model = CppMegaMambaModel(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        hybrid_layer_pattern=args.hybrid_layer_pattern,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )

    return model
