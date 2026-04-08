"""Megatron-side helpers for translating nanochat NAM-style recipes.

This module is intentionally lightweight and importable on macOS. It does not
import Megatron or torch at import time. The actual Megatron runtime exists
only on the remote H200 node.
"""

from __future__ import annotations

from dataclasses import dataclass

from cppmega.features.engram import EngramConfig, NgramHashConfig
from cppmega.features.mhc.config import MHCConfig
from cppmega.features.mod.config import MoDConfig, MoDAConfig
from cppmega.features.structure.config import StructureConfig


_SUPPORTED_NEMOTRON_SYMBOLS = frozenset({"A", "M", "D", "E", "G", "R", "|"})
_NEMOTRON_TO_MEGATRON = {
    "A": "*",
    "M": "M",
    "D": "G",
    "E": "E",
}


@dataclass(frozen=True)
class TranslationIssue:
    symbol: str
    message: str


@dataclass(frozen=True)
class MegatronHybridPlan:
    source_pattern: str
    translated_pattern: str
    requires_custom_mamba3: bool
    requires_custom_m2rnn: bool
    requires_mtp_suffix: bool
    issues: tuple[TranslationIssue, ...]
    engram: EngramConfig | None = None
    ngram_hash: NgramHashConfig | None = None
    mhc: MHCConfig | None = None
    mod: MoDConfig | None = None
    moda: MoDAConfig | None = None
    structure: StructureConfig | None = None

    @property
    def is_fully_native(self) -> bool:
        return (
            not self.requires_custom_mamba3
            and not self.requires_custom_m2rnn
            and not self.issues
        )


def parse_nem_pattern(pattern: str, depth: int) -> list[str]:
    """Match nanochat's non-pipe tiling semantics for Nemotron-style patterns."""

    if not pattern:
        raise ValueError("pattern must be non-empty")
    upper = pattern.upper()
    invalid = sorted({ch for ch in upper if ch not in _SUPPORTED_NEMOTRON_SYMBOLS})
    if invalid:
        raise ValueError(
            f"invalid pattern chars {invalid!r}; supported symbols are A, M, D, E, G, R and |"
        )

    if "|" in upper:
        segments = upper.split("|")
        flat = "".join(segments)
        if len(flat) != depth:
            raise ValueError(
                f"pipe-delimited pattern expands to {len(flat)} layers, expected depth={depth}"
            )
        return list(flat)

    return [upper[i % len(upper)] for i in range(depth)]


def count_layer_types(pattern: str, depth: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in parse_nem_pattern(pattern, depth):
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def translate_nanochat_pattern_to_megatron(
    *,
    pattern: str,
    depth: int,
    mtp_depths: int = 0,
    force_author_mamba3: bool = True,
    engram: EngramConfig | None = None,
    ngram_hash: NgramHashConfig | None = None,
    mhc: MHCConfig | None = None,
    mod: MoDConfig | None = None,
    moda: MoDAConfig | None = None,
    structure: StructureConfig | None = None,
) -> MegatronHybridPlan:
    """Translate a nanochat Nemotron pattern into Megatron hybrid syntax.

    Fail closed on ambiguous symbols:
    - `R` has no Megatron-native equivalent today, so it is reported as a custom
      seam and not silently remapped.
    - `M` is translated to Megatron `M`, but when `force_author_mamba3=True` the
      result is still flagged as requiring a custom Mamba3-backed Megatron layer.
    """

    layer_types = parse_nem_pattern(pattern, depth)
    translated: list[str] = []
    issues: list[TranslationIssue] = []
    requires_custom_m2rnn = False
    requires_custom_mamba3 = False

    for symbol in layer_types:
        if symbol == "R":
            requires_custom_m2rnn = True
            issues.append(
                TranslationIssue(
                    symbol="R",
                    message=(
                        "nanochat R=M2RNN has no Megatron-native equivalent; "
                        "custom port required before this pattern can run end-to-end"
                    ),
                )
            )
            translated.append("R")
            continue

        translated_symbol = _NEMOTRON_TO_MEGATRON.get(symbol)
        if translated_symbol is None:
            issues.append(
                TranslationIssue(
                    symbol=symbol,
                    message=f"no Megatron translation rule is defined for symbol {symbol!r}",
                )
            )
            translated.append(symbol)
            continue

        if symbol == "M" and force_author_mamba3:
            requires_custom_mamba3 = True

        translated.append(translated_symbol)

    translated_pattern = "".join(translated)
    requires_mtp_suffix = mtp_depths > 0
    if requires_mtp_suffix:
        translated_pattern = translated_pattern + "/" + "/".join("*-" for _ in range(mtp_depths))

    return MegatronHybridPlan(
        source_pattern=pattern,
        translated_pattern=translated_pattern,
        requires_custom_mamba3=requires_custom_mamba3,
        requires_custom_m2rnn=requires_custom_m2rnn,
        requires_mtp_suffix=requires_mtp_suffix,
        issues=tuple(issues),
        engram=engram,
        ngram_hash=ngram_hash,
        mhc=mhc,
        mod=mod,
        moda=moda,
        structure=structure,
    )


def build_nam56r_reference_plan() -> MegatronHybridPlan:
    """Return the current grounded NAM56R translation plan.

    The upstream nanochat launchers use `AEMEAEMEAEMR` at depth 52.
    This expands to 13 A, 13 M, 22 E, and 4 R layers.
    """

    return translate_nanochat_pattern_to_megatron(
        pattern="AEMEAEMEAEMR",
        depth=52,
        mtp_depths=1,
        force_author_mamba3=True,
    )


def build_nam56r_feature_plan(
    *,
    pattern: str,
    depth: int,
    mtp_depths: int = 0,
    force_author_mamba3: bool = True,
    engram_enabled: bool = False,
    engram_layers: str = "",
    engram_ngram_orders: str = "2,3,4",
    engram_bottleneck_dim: int = 0,
    engram_dropout: float = 0.0,
    engram_gated: bool = False,
    engram_gate_sqrt_compress: bool = False,
    engram_conv_kernel: int = 0,
    engram_conv_impl: str = "xla_safe",
    ngram_hash_enabled: bool = False,
    ngram_hash_orders: str = "2,3",
    ngram_hash_heads: int = 8,
    ngram_hash_table_size: int = 500_000,
    ngram_hash_embed_dim: int = 16,
    ngram_hash_offload: bool = False,
    mhc_enabled: bool = False,
    mhc_layers: str = "",
    mhc_n_streams: int = 4,
    mhc_sinkhorn_iters: int = 5,
    mhc_temperature: float = 1.0,
    mhc_epsilon: float = 1e-6,
    mhc_blend_alpha: float = 1.0,
    mhc_dynamic: bool = False,
    mhc_dynamic_mode: str = "maxtext",
    mhc_fused_ops: bool = False,
    mhc_recompute_group_size: int = 0,
    mod_enabled: bool = False,
    mod_layers: str = "",
    mod_capacity: float = 0.5,
    mod_aux_loss_weight: float = 0.01,
    mod_routing: str = "topk",
    mod_target: str = "",
    mod_scorer: str = "",
    mod_selector: str = "",
    mod_schedule: str = "",
    mod_executor: str = "auto",
    mod_ffn_only: bool = False,
    mod_skip_first_n: int = 4,
    mod_skip_mamba: bool = True,
    moda_enabled: bool = False,
    structure_enabled: bool = False,
    structure_components: str = "core",
    max_ast_depth: int = 20,
    max_sibling_index: int = 10,
    num_node_types: int = 64,
    structure_bottleneck_dim: int = 64,
    relation_bias_enabled: bool = False,
    tree_ffn_enabled: bool = False,
    tree_ffn_steps: int = 3,
    tree_ffn_dropout: float = 0.0,
    platform_embed_enabled: bool = False,
) -> MegatronHybridPlan:
    engram = EngramConfig.from_nanochat_args(
        enabled=engram_enabled,
        layers=engram_layers,
        ngram_orders=engram_ngram_orders,
        bottleneck_dim=engram_bottleneck_dim,
        dropout=engram_dropout,
        gated=engram_gated,
        gate_sqrt_compress=engram_gate_sqrt_compress,
        conv_kernel=engram_conv_kernel,
        conv_impl=engram_conv_impl,
    )
    ngram_hash = NgramHashConfig.from_nanochat_args(
        enabled=ngram_hash_enabled,
        orders=ngram_hash_orders,
        heads=ngram_hash_heads,
        table_size=ngram_hash_table_size,
        embed_dim=ngram_hash_embed_dim,
        offload=ngram_hash_offload,
    )
    mhc = MHCConfig.from_nanochat_args(
        enabled=mhc_enabled,
        layers=mhc_layers,
        n_streams=mhc_n_streams,
        sinkhorn_iters=mhc_sinkhorn_iters,
        temperature=mhc_temperature,
        epsilon=mhc_epsilon,
        blend_alpha=mhc_blend_alpha,
        dynamic=mhc_dynamic,
        dynamic_mode=mhc_dynamic_mode,
        fused_ops=mhc_fused_ops,
        recompute_group_size=mhc_recompute_group_size,
    )
    mod = MoDConfig.from_nanochat_args(
        enabled=mod_enabled,
        layers=mod_layers,
        capacity=mod_capacity,
        aux_loss_weight=mod_aux_loss_weight,
        routing=mod_routing,
        target=mod_target,
        scorer=mod_scorer,
        selector=mod_selector,
        schedule=mod_schedule,
        executor=mod_executor,
        ffn_only=mod_ffn_only,
        skip_first_n=mod_skip_first_n,
        skip_mamba=mod_skip_mamba,
    )
    moda = MoDAConfig.from_nanochat_args(enabled=moda_enabled)
    structure = StructureConfig.from_nanochat_args(
        enabled=structure_enabled,
        components=structure_components,
        max_ast_depth=max_ast_depth,
        max_sibling_index=max_sibling_index,
        num_node_types=num_node_types,
        bottleneck_dim=structure_bottleneck_dim,
        relation_bias_enabled=relation_bias_enabled,
        tree_ffn_enabled=tree_ffn_enabled,
        tree_ffn_steps=tree_ffn_steps,
        tree_ffn_dropout=tree_ffn_dropout,
        platform_embed_enabled=platform_embed_enabled,
    )
    if engram is not None:
        invalid = [index for index in engram.layer_indices if index >= depth]
        if invalid:
            raise ValueError(f"Engram layer indices {invalid} exceed depth={depth}")
    if mhc is not None:
        invalid = [index for index in mhc.layer_indices if index >= depth]
        if invalid:
            raise ValueError(f"mHC layer indices {invalid} exceed depth={depth}")
    if mod is not None:
        invalid = [index for index in mod.layer_indices if index >= depth]
        if invalid:
            raise ValueError(f"MoD layer indices {invalid} exceed depth={depth}")
    return translate_nanochat_pattern_to_megatron(
        pattern=pattern,
        depth=depth,
        mtp_depths=mtp_depths,
        force_author_mamba3=force_author_mamba3,
        engram=engram,
        ngram_hash=ngram_hash,
        mhc=mhc,
        mod=mod,
        moda=moda,
        structure=structure,
    )
