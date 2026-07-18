"""Canonical graph supervision recipe shared by data and H200 runtime."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from typing import Any, Mapping


STAGE1_GRAPH_RECIPE_SCHEMA = "cppmega_stage1_graph_recipe_v1"
STAGE1_GRAPH_LEGACY_RECIPE_SHA256 = (
    "0cfbc70d139215546b59acbaf07ea91dea272edfc1148ba2cd54f86add737a33"
)
STAGE1_GRAPH_RELATIONS = (
    "call",
    "type",
    "domain",
    "build",
    "shell",
    "diagnostic",
    "cross_domain",
)
STAGE1_GRAPH_TOPK = 256
STAGE1_GRAPH_BIAS_BETA = "1"
STAGE1_GRAPH_SCORE_FORMULA = "i_neural_plus_beta_s_graph_v1"
STAGE1_GRAPH_SCORE_STAGE = "before_topk"
STAGE1_GRAPH_EXACT_WEIGHTS = {
    "global_weight": "1",
    "indexer_weight": "1/1000",
    "layer_weight": "1",
    "bce_weight": "1/10",
    "coverage_weight": "1/20",
    "pos_weight": "1",
    "margin": "1",
}
STAGE1_GRAPH_PAIR_MASK = "causal_same_document_upstream_v1"
STAGE1_GRAPH_CHUNK_EDGE_EXPANSION = "cartesian_token_spans_v1"
STAGE1_GRAPH_RUNTIME = "megatron_dsa_indexer_v1"
STAGE1_GRAPH_LAYER_REDUCTION = "sum"


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def stage1_graph_recipe_payload() -> dict[str, object]:
    return {
        "schema": STAGE1_GRAPH_RECIPE_SCHEMA,
        "relations": list(STAGE1_GRAPH_RELATIONS),
        "topk": STAGE1_GRAPH_TOPK,
        "bias_beta": STAGE1_GRAPH_BIAS_BETA,
        "score_formula": STAGE1_GRAPH_SCORE_FORMULA,
        "score_stage": STAGE1_GRAPH_SCORE_STAGE,
        **STAGE1_GRAPH_EXACT_WEIGHTS,
        "layer_reduction": STAGE1_GRAPH_LAYER_REDUCTION,
        "runtime": STAGE1_GRAPH_RUNTIME,
        "pair_mask": STAGE1_GRAPH_PAIR_MASK,
        "chunk_edge_expansion": STAGE1_GRAPH_CHUNK_EDGE_EXPANSION,
    }


STAGE1_GRAPH_RECIPE_SHA256 = (
    "6a44969c8ae2f7d789a8305db88027e899f1f9b8c2ce52b70d47d21dece10cbf"
)
if _canonical_sha256(stage1_graph_recipe_payload()) != STAGE1_GRAPH_RECIPE_SHA256:
    raise RuntimeError(
        "Stage-1 graph recipe payload changed without a versioned SHA update"
    )


def stage1_graph_recipe_binding() -> dict[str, str]:
    return {
        "schema": STAGE1_GRAPH_RECIPE_SCHEMA,
        "sha256": STAGE1_GRAPH_RECIPE_SHA256,
    }


def stage1_graph_config_kwargs() -> dict[str, Any]:
    return {
        "relations": STAGE1_GRAPH_RELATIONS,
        "topk": STAGE1_GRAPH_TOPK,
        "bias_beta": float(Fraction(STAGE1_GRAPH_BIAS_BETA)),
        **{
            field: float(Fraction(value))
            for field, value in STAGE1_GRAPH_EXACT_WEIGHTS.items()
        },
    }


def validate_stage1_graph_contract(graph: Mapping[str, object]) -> None:
    binding = graph.get("recipe")
    if (
        isinstance(binding, Mapping)
        and binding.get("schema") == STAGE1_GRAPH_RECIPE_SCHEMA
        and binding.get("sha256") == STAGE1_GRAPH_LEGACY_RECIPE_SHA256
    ):
        raise ValueError(
            "legacy Stage-1 graph recipe binding detected; migration required: "
            "regenerate the objective contract and objective materialization artifact"
        )
    if binding != stage1_graph_recipe_binding():
        raise ValueError("graph_auxiliary.recipe binding is missing or stale")
    recipe = stage1_graph_recipe_payload()
    for field, expected in recipe.items():
        if field == "schema":
            continue
        actual = graph.get(field)
        if actual != expected:
            raise ValueError(
                f"graph_auxiliary.{field} differs from Stage-1 recipe: "
                f"{actual!r} != {expected!r}"
            )


__all__ = [
    "STAGE1_GRAPH_CHUNK_EDGE_EXPANSION",
    "STAGE1_GRAPH_BIAS_BETA",
    "STAGE1_GRAPH_EXACT_WEIGHTS",
    "STAGE1_GRAPH_LEGACY_RECIPE_SHA256",
    "STAGE1_GRAPH_LAYER_REDUCTION",
    "STAGE1_GRAPH_PAIR_MASK",
    "STAGE1_GRAPH_RECIPE_SCHEMA",
    "STAGE1_GRAPH_RECIPE_SHA256",
    "STAGE1_GRAPH_RELATIONS",
    "STAGE1_GRAPH_RUNTIME",
    "STAGE1_GRAPH_SCORE_FORMULA",
    "STAGE1_GRAPH_SCORE_STAGE",
    "STAGE1_GRAPH_TOPK",
    "stage1_graph_config_kwargs",
    "stage1_graph_recipe_binding",
    "stage1_graph_recipe_payload",
    "validate_stage1_graph_contract",
]
