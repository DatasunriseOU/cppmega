"""Helpers for threading cppmega sidecar metadata through Megatron models."""

from __future__ import annotations

from collections.abc import Mapping

TOKEN_SIDECAR_KEYS = (
    "domain_ids",
    "role_ids",
    "entity_ids",
    "scope_ids",
    "source_doc_ids",
    "confidence_ids",
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
    "platform_ids",
    "def_use",
    "change_mask_pre",
    "change_mask_post",
)

GRAPH_ROUTE_KEYS = (
    "graph_call_edges",
    "graph_call_edge_counts",
    "graph_type_edges",
    "graph_type_edge_counts",
    "graph_domain_edges",
    "graph_domain_edge_counts",
    "graph_build_edges",
    "graph_build_edge_counts",
    "graph_shell_edges",
    "graph_shell_edge_counts",
    "graph_diagnostic_edges",
    "graph_diagnostic_edge_counts",
    "graph_cross_domain_edges",
    "graph_cross_domain_edge_counts",
    "graph_chunk_starts",
    "graph_chunk_ends",
    "graph_chunk_kinds",
    "graph_chunk_dep_levels",
    "graph_chunk_counts",
)


def extract_structure_inputs(batch: Mapping[str, object] | None) -> dict[str, object] | None:
    if batch is None:
        return None
    keys = TOKEN_SIDECAR_KEYS + GRAPH_ROUTE_KEYS
    extracted = {key: batch[key] for key in keys if key in batch and batch[key] is not None}
    return extracted or None


def maybe_set_structure_inputs(model, batch: Mapping[str, object] | None) -> dict[str, object] | None:
    structure_inputs = extract_structure_inputs(batch)
    if structure_inputs is None:
        return None
    setter = getattr(model, "set_cppmega_structure_inputs", None)
    if setter is not None:
        setter(structure_inputs)
    return structure_inputs
