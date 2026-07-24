"""Cppmega Mamba builder with polymorphic embedding substitution."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

from megatron.core.models.mamba import MambaModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import import_module
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from cppmega.megatron.custom_mamba_model import CppMegaMambaModel
from cppmega.megatron.fa4_score_mod_adapter import (
    CppMegaFA4ScoreModAttention,
    fa4_score_mod_enabled,
)
from cppmega.megatron.graph_objective_loss import (
    graph_objective_requested,
    require_active_dsa_graph_objective,
)


def _is_dense_attention(module_cls) -> bool:
    """Return True if *module_cls* is a known dense (non-DSA) attention class.

    FA4 score_mod should ONLY replace standard dense dot-product attention
    (TEDotProductAttention, DotProductAttention).  DSA (Dynamic Sparse
    Attention) has its own core_attention with indexer logic; replacing it
    would silently break the sparse routing path.
    """
    if module_cls is CppMegaFA4ScoreModAttention:
        return False  # already swapped
    qualname = f"{getattr(module_cls, '__module__', '')}.{getattr(module_cls, '__qualname__', '')}".lower()
    # Reject DSA / DSAttention / any sparse-attention variant.
    if "dsa" in qualname or "dsattention" in qualname or "sparse" in qualname:
        return False
    # Accept known dense attention classes by name.
    cls_name = getattr(module_cls, "__name__", "") or ""
    if cls_name in ("TEDotProductAttention", "DotProductAttention"):
        return True
    # Fallback: accept anything in the transformer_engine or megatron
    # dot_product_attention namespace that isn't explicitly DSA.
    if "dotproductattention" in cls_name.lower():
        return True
    return False


def _swap_core_attention_for_fa4(spec):
    """Return ``spec`` with every ``core_attention`` module swapped to FA4.

    Recursively walks the ``ModuleSpec`` / submodule dataclass tree and rebuilds
    any node whose ``core_attention`` field points at a *dense* dot-product
    attention (TEDotProductAttention / DotProductAttention) so it instead uses
    ``CppMegaFA4ScoreModAttention``.  DSA-specific attention modules are left
    untouched because they carry their own indexer/sparse-routing logic.
    Nodes are rebuilt with ``dataclasses.replace`` (never mutated in place) so
    the shared upstream ``mamba_stack_spec`` submodules are left untouched.
    """

    if isinstance(spec, (list, tuple)):
        swapped = [_swap_core_attention_for_fa4(item) for item in spec]
        return type(spec)(swapped)
    if not is_dataclass(spec) or isinstance(spec, type):
        return spec

    changes: dict[str, object] = {}
    for field in fields(spec):
        value = getattr(spec, field.name)
        if field.name == "core_attention":
            if getattr(value, "module", None) is not None:
                # ModuleSpec-wrapped: only replace if it's a known dense attention.
                # DSA attention (DSAttention etc.) must NOT be replaced.
                if _is_dense_attention(value.module):
                    changes[field.name] = replace(value, module=CppMegaFA4ScoreModAttention)
            elif isinstance(value, type) and _is_dense_attention(value):
                # Direct class reference (e.g. core_attention=TEDotProductAttention)
                changes[field.name] = CppMegaFA4ScoreModAttention
        else:
            new_value = _swap_core_attention_for_fa4(value)
            if new_value is not value:
                changes[field.name] = new_value
    if not changes:
        return spec
    return replace(spec, **changes)


def cppmega_mamba_builder(
    args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None
):
    print_rank_0("building cppmega MAMBA model ...")
    if config is None:
        config = core_transformer_config_from_args(args, TransformerConfig)
    require_active_dsa_graph_objective(
        config, required=graph_objective_requested()
    )
    assert not getattr(args, "use_legacy_models", False), "Mamba only supported in Mcore!"

    if args.spec is None:
        raise ValueError("cppmega_mamba_builder requires --spec")

    spec_or_factory = import_module(args.spec)
    mamba_stack_spec = spec_or_factory(config) if callable(spec_or_factory) else spec_or_factory
    if fa4_score_mod_enabled():
        mamba_stack_spec = _swap_core_attention_for_fa4(mamba_stack_spec)
        print_rank_0(
            "cppmega: CPPMEGA_FA4_SCORE_MOD=1 -> core_attention swapped "
            "for CppMegaFA4ScoreModAttention (chunk-native)"
        )
    vocab_size = getattr(args, "padded_vocab_size", None) or getattr(args, "vocab_size")

    model = CppMegaMambaModel(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=vocab_size,
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
