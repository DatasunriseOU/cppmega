from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

from cppmega.megatron.graph_recipe import (
    STAGE1_GRAPH_RECIPE_SCHEMA,
    STAGE1_GRAPH_RECIPE_SHA256,
    stage1_graph_recipe_binding,
    stage1_graph_recipe_payload,
    validate_stage1_graph_contract,
)


_LEGACY_STAGE1_GRAPH_RECIPE_SHA256 = (
    "0cfbc70d139215546b59acbaf07ea91dea272edfc1148ba2cd54f86add737a33"
)
_EXPECTED_STAGE1_GRAPH_RECIPE_PAYLOAD = {
    "schema": "cppmega_stage1_graph_recipe_v1",
    "relations": [
        "call",
        "type",
        "domain",
        "build",
        "shell",
        "diagnostic",
        "cross_domain",
    ],
    "topk": 256,
    "bias_beta": "1",
    "score_formula": "i_neural_plus_beta_s_graph_v1",
    "score_stage": "before_topk",
    "global_weight": "1",
    "indexer_weight": "1/1000",
    "layer_weight": "1",
    "bce_weight": "1/10",
    "coverage_weight": "1/20",
    "pos_weight": "1",
    "margin": "1",
    "layer_reduction": "sum",
    "runtime": "megatron_dsa_indexer_v1",
    "pair_mask": "causal_same_document_upstream_v1",
    "chunk_edge_expansion": "cartesian_token_spans_v1",
}
_EXPECTED_STAGE1_GRAPH_RECIPE_BYTES = json.dumps(
    _EXPECTED_STAGE1_GRAPH_RECIPE_PAYLOAD,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
).encode("ascii")
_EXPECTED_STAGE1_GRAPH_RECIPE_SHA256 = (
    "6a44969c8ae2f7d789a8305db88027e899f1f9b8c2ce52b70d47d21dece10cbf"
)


def _peer_graph_recipe_module():
    configured = os.environ.get("CPPMEGA_RECIPE_PARITY_PEER_ROOT")
    expected_commit = os.environ.get("CPPMEGA_RECIPE_PARITY_PEER_COMMIT")
    if not configured or not expected_commit:
        pytest.fail(
            "recipe parity requires explicit CPPMEGA_RECIPE_PARITY_PEER_ROOT "
            "and CPPMEGA_RECIPE_PARITY_PEER_COMMIT"
        )
    peer_root = Path(configured).expanduser().resolve()
    if not peer_root.is_dir():
        pytest.fail(f"cross-repository recipe parity worktree is unavailable: {peer_root}")
    actual_commit = subprocess.run(
        ("git", "-C", str(peer_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit:
        pytest.fail(
            "recipe parity checkout commit mismatch: "
            f"expected={expected_commit} actual={actual_commit}"
        )
    module_path = peer_root / "cppmega_mlx/data/graph_recipe.py"
    spec = importlib.util.spec_from_file_location("_peer_mlx_graph_recipe", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load peer graph recipe module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graph_contract() -> dict[str, object]:
    return {
        **stage1_graph_recipe_payload(),
        "recipe": stage1_graph_recipe_binding(),
    }


def test_stage1_graph_recipe_literal_sha_matches_canonical_payload() -> None:
    payload = json.dumps(
        stage1_graph_recipe_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

    assert hashlib.sha256(payload).hexdigest() == STAGE1_GRAPH_RECIPE_SHA256


def test_stage1_graph_recipe_matches_frozen_bytes() -> None:
    payload = json.dumps(
        stage1_graph_recipe_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    assert stage1_graph_recipe_payload() == _EXPECTED_STAGE1_GRAPH_RECIPE_PAYLOAD
    assert payload == _EXPECTED_STAGE1_GRAPH_RECIPE_BYTES
    assert STAGE1_GRAPH_RECIPE_SHA256 == _EXPECTED_STAGE1_GRAPH_RECIPE_SHA256
    assert hashlib.sha256(payload).hexdigest() == _EXPECTED_STAGE1_GRAPH_RECIPE_SHA256


def test_stage1_graph_recipe_matches_mlx_peer() -> None:
    peer = _peer_graph_recipe_module()
    assert peer.stage1_graph_recipe_payload() == stage1_graph_recipe_payload()
    assert peer.STAGE1_GRAPH_RECIPE_SCHEMA == STAGE1_GRAPH_RECIPE_SCHEMA
    assert peer.STAGE1_GRAPH_RECIPE_SHA256 == STAGE1_GRAPH_RECIPE_SHA256
    assert peer.stage1_graph_recipe_binding() == stage1_graph_recipe_binding()


def test_stage1_graph_contract_rejects_legacy_recipe_with_migration_error() -> None:
    graph = _graph_contract()
    graph["recipe"] = {
        "schema": STAGE1_GRAPH_RECIPE_SCHEMA,
        "sha256": _LEGACY_STAGE1_GRAPH_RECIPE_SHA256,
    }

    with pytest.raises(ValueError, match="legacy.*migration required.*regenerate"):
        validate_stage1_graph_contract(graph)


def test_stage1_graph_contract_rejects_stale_recipe_sha() -> None:
    graph = _graph_contract()
    graph["recipe"] = {
        **stage1_graph_recipe_binding(),
        "sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="recipe binding is missing or stale"):
        validate_stage1_graph_contract(graph)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("relations", ["call", "type"]),
        ("topk", 8),
        ("bias_beta", "2"),
        ("score_formula", "legacy_score_v0"),
        ("score_stage", "after_topk"),
        ("indexer_weight", "1/500"),
    ),
)
def test_stage1_graph_contract_rejects_recipe_field_drift(
    field: str,
    bad_value: object,
) -> None:
    graph = copy.deepcopy(_graph_contract())
    graph[field] = bad_value

    with pytest.raises(ValueError, match=rf"graph_auxiliary\.{field}"):
        validate_stage1_graph_contract(graph)
