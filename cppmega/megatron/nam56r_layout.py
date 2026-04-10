"""Import-safe layout helpers for NAM56R pattern-derived routing."""

from __future__ import annotations

import os

from cppmega.recipes.nam56r_launch import get_custom_layer_indices
from cppmega.recipes.nam56r_megatron import parse_nem_pattern

FULL_NAM56R_PATTERN = "AEMEAEMEAEMR"
FULL_NAM56R_DEPTH = 52


def parse_indices(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return tuple(values)


def load_pattern() -> tuple[str, int]:
    pattern = os.environ.get("CPPMEGA_NEM_PATTERN", FULL_NAM56R_PATTERN)
    depth = int(os.environ.get("CPPMEGA_LAYER_DEPTH", str(FULL_NAM56R_DEPTH)))
    return pattern, depth


def load_r_layer_indices() -> tuple[int, ...]:
    raw = os.environ.get("CPPMEGA_R_LAYER_INDICES", "").strip()
    if raw:
        return parse_indices(raw)
    pattern, depth = load_pattern()
    return get_custom_layer_indices(pattern=pattern, depth=depth, custom_symbols=("R",))


def load_dsa_a_layer_ranks() -> tuple[int, ...]:
    raw = os.environ.get("CPPMEGA_DSA_A_LAYER_RANKS", "").strip()
    if not raw:
        return ()
    return parse_indices(raw)


def load_attention_layer_numbers() -> tuple[int, ...]:
    pattern, depth = load_pattern()
    return tuple(index + 1 for index, symbol in enumerate(parse_nem_pattern(pattern, depth)) if symbol == "A")
