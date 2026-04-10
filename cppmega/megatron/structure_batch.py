"""Helpers for threading structure metadata through cppmega Megatron models."""

from __future__ import annotations

from collections.abc import Mapping

_STRUCTURE_KEYS = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)


def extract_structure_inputs(batch: Mapping[str, object] | None) -> dict[str, object] | None:
    if batch is None:
        return None
    extracted = {key: batch[key] for key in _STRUCTURE_KEYS if key in batch and batch[key] is not None}
    return extracted or None


def maybe_set_structure_inputs(model, batch: Mapping[str, object] | None) -> dict[str, object] | None:
    structure_inputs = extract_structure_inputs(batch)
    if structure_inputs is None:
        return None
    setter = getattr(model, "set_cppmega_structure_inputs", None)
    if setter is not None:
        setter(structure_inputs)
    return structure_inputs
