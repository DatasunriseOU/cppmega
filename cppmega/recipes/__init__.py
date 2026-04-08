"""Recipe translation helpers for cppmega."""

from cppmega.recipes.nam56r_megatron import (
    MegatronHybridPlan,
    TranslationIssue,
    build_nam56r_reference_plan,
    count_layer_types,
    parse_nem_pattern,
    translate_nanochat_pattern_to_megatron,
)

__all__ = [
    "MegatronHybridPlan",
    "TranslationIssue",
    "build_nam56r_reference_plan",
    "count_layer_types",
    "parse_nem_pattern",
    "translate_nanochat_pattern_to_megatron",
]
