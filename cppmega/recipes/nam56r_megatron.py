"""Megatron-side helpers for translating nanochat NAM-style recipes.

This module is intentionally lightweight and importable on macOS. It does not
import Megatron or torch at import time. The actual Megatron runtime exists
only on the remote H200 node.
"""

from __future__ import annotations

from dataclasses import dataclass

from cppmega.features.engram import EngramConfig, NgramHashConfig


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
    if engram is not None:
        invalid = [index for index in engram.layer_indices if index >= depth]
        if invalid:
            raise ValueError(f"Engram layer indices {invalid} exceed depth={depth}")
    return translate_nanochat_pattern_to_megatron(
        pattern=pattern,
        depth=depth,
        mtp_depths=mtp_depths,
        force_author_mamba3=force_author_mamba3,
        engram=engram,
        ngram_hash=ngram_hash,
    )
