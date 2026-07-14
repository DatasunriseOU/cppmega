"""Fail-closed contract for pre-materialized production objectives.

Megatron consumes one causal token stream per indexed document. Non-causal
training tasks therefore have to be materialized upstream as shifted-LM
documents plus an aligned loss mask. This module validates the receipt for that
materialization and tracks every document written by the parquet converter.

The receipt deliberately names authoritative token columns. Rendered prompt or
comment text is not an accepted source for IFIM or commit objectives.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

OBJECTIVE_CONTRACT_SCHEMA = "cppmega_pre_materialized_objectives_v1"
OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA = (
    "cppmega_objective_materialization_artifact_v1"
)
OBJECTIVE_TOKEN_SIDE_CHANNELS: tuple[tuple[str, str], ...] = (
    ("loss_mask", "uint8"),
    ("doc_ids", "uint16"),
    ("token_domain_ids", "uint16"),
    ("token_role_ids", "uint16"),
    ("token_entity_ids", "uint32"),
    ("token_scope_ids", "uint32"),
    ("token_source_doc_ids", "uint32"),
    ("token_confidence_ids", "uint8"),
    ("token_structure_ids", "uint8"),
    ("token_dep_levels", "uint16"),
    ("token_ast_depth", "uint16"),
    ("token_sibling_index", "uint16"),
    ("token_ast_node_type", "uint16"),
    ("token_symbol_ids", "uint64"),
    ("token_call_targets", "uint64"),
    ("token_type_refs", "uint64"),
    ("token_def_use", "uint8"),
    ("token_change_mask_pre", "uint8"),
    ("token_change_mask_post", "uint8"),
)
OBJECTIVE_GRAPH_SIDECARS: tuple[tuple[str, str, str], ...] = (
    ("token_call_edges", "edge_pairs", "int32"),
    ("token_type_edges", "edge_pairs", "int32"),
    ("token_domain_edges", "edge_triples", "int32"),
    ("token_build_edges", "edge_triples", "int32"),
    ("token_shell_edges", "edge_triples", "int32"),
    ("token_diagnostic_edges", "edge_triples", "int32"),
    ("token_cross_domain_edges", "edge_triples", "int32"),
    ("token_chunk_starts", "ragged_1d", "uint32"),
    ("token_chunk_ends", "ragged_1d", "uint32"),
    ("token_chunk_kinds", "ragged_1d", "uint16"),
    ("token_chunk_dep_levels", "ragged_1d", "uint16"),
)
OBJECTIVE_IDS: dict[str, int] = {
    "causal_lm": 1,
    "fim": 2,
    "ast_fim": 3,
    "ifim": 4,
    "commit_diff": 5,
    "pre_to_post": 6,
    "symbol_recovery": 7,
    "type_recovery": 8,
    "callee_recovery": 9,
}
REQUIRED_PRODUCTION_OBJECTIVES = frozenset(
    {"causal_lm", "fim", "ast_fim", "ifim", "commit_diff", "pre_to_post"}
)

_EXPECTED_TYPED_SOURCES = {
    "ifim_instruction": "ifim_instruction_token_ids",
    "commit_message": "commit_msg_token_ids",
    "diff": "diff_token_ids",
    "pre": "pre_token_ids",
    "post": "post_token_ids",
    "missing_fields": "ineligible",
    "rendered_text_parsing": False,
}


def _mapping(value: object, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    return value


def _positive_int(value: object, *, where: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{where} must be >= {minimum}, got {value}")
    return value


def _fraction(value: object, *, where: str, positive: bool = False) -> Fraction:
    if not isinstance(value, str):
        raise ValueError(
            f"{where} must be an exact integer/fraction string, got {value!r}"
        )
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{where} is not an exact fraction: {value!r}") from exc
    if parsed < 0 or (positive and parsed <= 0):
        relation = "> 0" if positive else ">= 0"
        raise ValueError(f"{where} must be {relation}, got {value!r}")
    return parsed


def _hamilton_quotas(
    rates: Mapping[str, Fraction], task_order: tuple[str, ...], window: int
) -> dict[str, int]:
    raw = {task: rates[task] * window for task in task_order}
    quotas = {task: value.numerator // value.denominator for task, value in raw.items()}
    remaining = window - sum(quotas.values())
    order_index = {task: index for index, task in enumerate(task_order)}
    ranked = sorted(
        task_order,
        key=lambda task: (-(raw[task] - quotas[task]), order_index[task]),
    )
    for task in ranked[:remaining]:
        quotas[task] += 1
    return quotas


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ValidatedObjectiveContract:
    payload: dict[str, Any]
    sha256: str
    task_order: tuple[str, ...]
    planned_samples: dict[str, int]


@dataclass(frozen=True)
class ObjectiveMaterializationArtifact:
    path: Path
    input_dir: Path
    contract_path: Path
    parquet_paths: tuple[Path, ...]
    contract: ValidatedObjectiveContract
    payload: dict[str, Any]
    artifact_set_sha256: str
    file_sha256: str


def validate_objective_contract(
    raw_contract: Mapping[str, Any],
) -> ValidatedObjectiveContract:
    """Validate one upstream objective receipt without filling any defaults."""

    contract = copy.deepcopy(dict(raw_contract))
    if contract.get("schema") != OBJECTIVE_CONTRACT_SCHEMA:
        raise ValueError(
            f"schema must be {OBJECTIVE_CONTRACT_SCHEMA!r}, got "
            f"{contract.get('schema')!r}"
        )
    if contract.get("algorithm") != "hamilton_eligibility_bipartite_v1":
        raise ValueError("algorithm must be 'hamilton_eligibility_bipartite_v1'")
    _positive_int(contract.get("seed"), where="seed", allow_zero=True)
    window = _positive_int(
        contract.get("quota_window_samples"), where="quota_window_samples"
    )

    raw_order = contract.get("task_order")
    if not isinstance(raw_order, list) or not raw_order:
        raise ValueError("task_order must be a non-empty list")
    task_order = tuple(raw_order)
    if any(not isinstance(task, str) for task in task_order):
        raise ValueError("task_order entries must be strings")
    if len(set(task_order)) != len(task_order):
        raise ValueError("task_order contains duplicate objectives")
    unknown = sorted(set(task_order) - set(OBJECTIVE_IDS))
    if unknown:
        raise ValueError(f"task_order contains unknown objectives: {unknown}")
    missing_required = sorted(REQUIRED_PRODUCTION_OBJECTIVES - set(task_order))
    if missing_required:
        raise ValueError(
            f"task_order is missing production objectives: {missing_required}"
        )
    objective_ids = _mapping(contract.get("objective_ids"), where="objective_ids")
    expected_objective_ids = {task: OBJECTIVE_IDS[task] for task in task_order}
    if dict(objective_ids) != expected_objective_ids:
        raise ValueError(
            "objective_ids must use the canonical encounter-order-independent "
            f"mapping: {expected_objective_ids}"
        )

    configured = _mapping(contract.get("configured_rates"), where="configured_rates")
    if set(configured) != set(task_order):
        raise ValueError("configured_rates keys must exactly match task_order")
    rates = {
        task: _fraction(
            configured[task], where=f"configured_rates.{task}", positive=True
        )
        for task in task_order
    }
    if sum(rates.values(), Fraction()) != Fraction(1):
        raise ValueError("configured_rates must sum exactly to 1")

    totals = _mapping(contract.get("totals"), where="totals")
    total_samples = _positive_int(totals.get("samples"), where="totals.samples")
    total_input_tokens = _positive_int(
        totals.get("input_tokens"), where="totals.input_tokens"
    )
    total_loss_tokens = _positive_int(
        totals.get("loss_tokens"), where="totals.loss_tokens"
    )
    if total_samples % window:
        raise ValueError(
            "totals.samples must contain complete quota windows: "
            f"samples={total_samples}, window={window}"
        )
    window_quotas = _hamilton_quotas(rates, task_order, window)
    window_count = total_samples // window
    expected_planned = {
        task: quota * window_count for task, quota in window_quotas.items()
    }

    planned = _mapping(contract.get("planned_samples"), where="planned_samples")
    if set(planned) != set(task_order):
        raise ValueError("planned_samples keys must exactly match task_order")
    planned_samples = {
        task: _positive_int(
            planned[task], where=f"planned_samples.{task}", allow_zero=True
        )
        for task in task_order
    }
    if planned_samples != expected_planned:
        raise ValueError(
            "planned_samples do not match deterministic Hamilton quotas: "
            f"expected={expected_planned}, got={planned_samples}"
        )

    realized = _mapping(contract.get("realized"), where="realized")
    if set(realized) != set(task_order):
        raise ValueError("realized keys must exactly match task_order")
    realized_totals = {"samples": 0, "input_tokens": 0, "loss_tokens": 0}
    for task in task_order:
        row = _mapping(realized[task], where=f"realized.{task}")
        samples = _positive_int(
            row.get("samples"), where=f"realized.{task}.samples", allow_zero=True
        )
        if samples != planned_samples[task]:
            raise ValueError(
                f"realized objective {task} samples={samples} differs from "
                f"planned={planned_samples[task]}"
            )
        input_tokens = _positive_int(
            row.get("input_tokens"),
            where=f"realized.{task}.input_tokens",
            allow_zero=samples == 0,
        )
        loss_tokens = _positive_int(
            row.get("loss_tokens"),
            where=f"realized.{task}.loss_tokens",
            allow_zero=samples == 0,
        )
        if samples == 0 and (input_tokens or loss_tokens):
            raise ValueError(
                f"realized.{task} has zero samples but nonzero token accounting"
            )
        if loss_tokens > input_tokens:
            raise ValueError(
                f"realized.{task}.loss_tokens={loss_tokens} exceeds "
                f"input_tokens={input_tokens}"
            )
        realized_totals["samples"] += samples
        realized_totals["input_tokens"] += input_tokens
        realized_totals["loss_tokens"] += loss_tokens
    expected_totals = {
        "samples": total_samples,
        "input_tokens": total_input_tokens,
        "loss_tokens": total_loss_tokens,
    }
    if realized_totals != expected_totals:
        raise ValueError(
            f"totals do not equal realized accounting: expected={realized_totals}, "
            f"got={expected_totals}"
        )

    typed_sources = _mapping(contract.get("typed_sources"), where="typed_sources")
    for field, expected in _EXPECTED_TYPED_SOURCES.items():
        actual = typed_sources.get(field)
        if actual != expected:
            raise ValueError(
                f"typed_sources.{field} must be {expected!r}, got {actual!r}"
            )

    graph = _mapping(contract.get("graph_auxiliary"), where="graph_auxiliary")
    relations = graph.get("relations")
    if (
        not isinstance(relations, list)
        or not relations
        or any(not isinstance(item, str) or not item for item in relations)
    ):
        raise ValueError("graph_auxiliary.relations must be a non-empty string list")
    _positive_int(
        graph.get("eligible_samples"), where="graph_auxiliary.eligible_samples"
    )
    _positive_int(graph.get("positive_edges"), where="graph_auxiliary.positive_edges")
    for field in (
        "global_weight",
        "indexer_weight",
        "layer_weight",
        "bce_weight",
        "coverage_weight",
    ):
        _fraction(graph.get(field), where=f"graph_auxiliary.{field}", positive=True)
    if graph.get("layer_reduction") != "sum":
        raise ValueError("graph_auxiliary.layer_reduction must be 'sum'")
    for field in ("pos_weight", "margin"):
        _fraction(
            graph.get(field),
            where=f"graph_auxiliary.{field}",
            positive=field == "pos_weight",
        )
    _positive_int(graph.get("topk"), where="graph_auxiliary.topk")
    if graph.get("included_in_total_loss") is not True:
        raise ValueError("graph_auxiliary.included_in_total_loss must be true")
    if graph.get("runtime") != "megatron_dsa_indexer_v1":
        raise ValueError("graph_auxiliary.runtime must be 'megatron_dsa_indexer_v1'")
    if graph.get("pair_mask") != "causal_same_document_upstream_v1":
        raise ValueError(
            "graph_auxiliary.pair_mask must be "
            "'causal_same_document_upstream_v1'"
        )
    if graph.get("chunk_edge_expansion") != "cartesian_token_spans_v1":
        raise ValueError(
            "graph_auxiliary.chunk_edge_expansion must be "
            "'cartesian_token_spans_v1'"
        )

    materialization = _mapping(contract.get("materialization"), where="materialization")
    if materialization.get("format") != "shifted_lm_document_v1":
        raise ValueError("materialization.format must be 'shifted_lm_document_v1'")
    expected_materialization_columns = {
        "token_column": "input_ids",
        "loss_mask_column": "loss_mask",
        "length_column": "valid_token_count",
        "objective_column": "objective_kind",
        "document_id_column": "doc_ids",
        "source_document_id_column": "token_source_doc_ids",
    }
    for field, expected in expected_materialization_columns.items():
        if materialization.get(field) != expected:
            raise ValueError(f"materialization.{field} must be {expected!r}")

    return ValidatedObjectiveContract(
        payload=contract,
        sha256=_canonical_sha256(contract),
        task_order=task_order,
        planned_samples=planned_samples,
    )


def _artifact_file(root: Path, value: object, *, where: str) -> Path:
    if not isinstance(value, str) or not value or os.path.isabs(value):
        raise ValueError(f"{where} must be a non-empty relative path")
    path = (root / value).resolve()
    if path.parent != root:
        raise ValueError(f"{where} must name a file directly inside the artifact dir")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_objective_materialization_artifact(
    path: str | os.PathLike[str],
) -> ObjectiveMaterializationArtifact:
    """Open and byte-verify the canonical CASE1 objective handoff."""

    artifact_path = Path(path).resolve()
    with artifact_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    artifact = dict(_mapping(raw, where="objective materialization artifact"))
    expected_top = {
        "schema",
        "documents",
        "objective_contract",
        "parquet_shards",
        "converter",
        "artifact_set_sha256",
    }
    if set(artifact) != expected_top:
        raise ValueError(
            "objective materialization artifact keys must be exactly "
            f"{sorted(expected_top)}"
        )
    if artifact.get("schema") != OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA:
        raise ValueError(
            "objective materialization artifact schema must be "
            f"{OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA!r}"
        )
    artifact_set_payload = dict(artifact)
    artifact_set_sha256 = artifact_set_payload.pop("artifact_set_sha256", None)
    if artifact_set_sha256 != _canonical_sha256(artifact_set_payload):
        raise ValueError(
            "objective materialization artifact_set_sha256 does not match payload"
        )
    root = artifact_path.parent.resolve()

    contract_ref = _mapping(
        artifact.get("objective_contract"), where="objective_contract"
    )
    expected_contract_ref_keys = {
        "path",
        "sha256",
        "size_bytes",
        "file_sha256",
    }
    if set(contract_ref) != expected_contract_ref_keys:
        raise ValueError(
            "objective_contract must contain path, sha256, size_bytes, "
            "and file_sha256"
        )
    contract_path = _artifact_file(
        root, contract_ref.get("path"), where="objective_contract.path"
    )
    expected_contract_size = _positive_int(
        contract_ref.get("size_bytes"), where="objective_contract.size_bytes"
    )
    if contract_path.stat().st_size != expected_contract_size:
        raise ValueError("objective_contract.size_bytes does not match")
    if contract_ref.get("file_sha256") != _file_sha256(contract_path):
        raise ValueError("objective_contract.file_sha256 does not match")
    with contract_path.open(encoding="utf-8") as handle:
        contract_raw = json.load(handle)
    contract = validate_objective_contract(
        _mapping(contract_raw, where="objective contract")
    )
    if contract_ref.get("sha256") != contract.sha256:
        raise ValueError("objective_contract.sha256 does not match contract payload")
    documents = _positive_int(artifact.get("documents"), where="documents")
    if documents != contract.payload["totals"]["samples"]:
        raise ValueError("artifact documents does not match objective contract totals")

    converter = _mapping(artifact.get("converter"), where="converter")
    expected_converter_keys = {
        "split",
        "token_column",
        "length_column",
        "side_channels",
        "graph_sidecars",
        "source_platform_sidecar",
        "graph_relations",
        "graph_pair_mask",
        "chunk_edge_expansion",
    }
    if set(converter) != expected_converter_keys:
        raise ValueError(
            f"converter keys must be exactly {sorted(expected_converter_keys)}"
        )
    expected_side_channels = [
        {"column": column, "dtype": dtype}
        for column, dtype in OBJECTIVE_TOKEN_SIDE_CHANNELS
    ]
    expected_graph_sidecars = [
        {"column": column, "kind": kind, "dtype": dtype}
        for column, kind, dtype in OBJECTIVE_GRAPH_SIDECARS
    ]
    materialization = contract.payload["materialization"]
    graph = contract.payload["graph_auxiliary"]
    expected_converter = {
        "split": "all",
        "token_column": materialization["token_column"],
        "length_column": materialization["length_column"],
        "side_channels": expected_side_channels,
        "graph_sidecars": expected_graph_sidecars,
        "source_platform_sidecar": "require",
        "graph_relations": graph["relations"],
        "graph_pair_mask": graph["pair_mask"],
        "chunk_edge_expansion": graph["chunk_edge_expansion"],
    }
    if dict(converter) != expected_converter:
        raise ValueError("objective materialization converter contract drifted")

    shard_rows = artifact.get("parquet_shards")
    if not isinstance(shard_rows, list) or not shard_rows:
        raise ValueError("parquet_shards must be a non-empty list")
    parquet_paths: list[Path] = []
    names: list[str] = []
    for index, raw_row in enumerate(shard_rows):
        row = _mapping(raw_row, where=f"parquet_shards[{index}]")
        if set(row) != {"path", "size_bytes", "sha256"}:
            raise ValueError(
                f"parquet_shards[{index}] must contain path, size_bytes, sha256"
            )
        shard_path = _artifact_file(
            root, row.get("path"), where=f"parquet_shards[{index}].path"
        )
        if shard_path.suffix != ".parquet":
            raise ValueError(f"parquet_shards[{index}].path must end in .parquet")
        expected_size = _positive_int(
            row.get("size_bytes"), where=f"parquet_shards[{index}].size_bytes"
        )
        if shard_path.stat().st_size != expected_size:
            raise ValueError(f"parquet_shards[{index}].size_bytes does not match")
        if row.get("sha256") != _file_sha256(shard_path):
            raise ValueError(f"parquet_shards[{index}].sha256 does not match")
        parquet_paths.append(shard_path)
        names.append(shard_path.name)
    if names != sorted(names) or len(set(names)) != len(names):
        raise ValueError("parquet_shards must be unique and sorted by path")
    unlisted = sorted(
        candidate.name
        for candidate in root.glob("*.parquet")
        if candidate.resolve() not in set(parquet_paths)
    )
    if unlisted:
        raise ValueError(f"objective artifact directory contains unlisted parquet: {unlisted}")

    return ObjectiveMaterializationArtifact(
        path=artifact_path,
        input_dir=root,
        contract_path=contract_path,
        parquet_paths=tuple(parquet_paths),
        contract=contract,
        payload=copy.deepcopy(artifact),
        artifact_set_sha256=str(artifact_set_sha256),
        file_sha256=_file_sha256(artifact_path),
    )


def materialized_objective_artifact_manifest(
    artifact: ObjectiveMaterializationArtifact,
) -> dict[str, Any]:
    """Return the immutable artifact binding embedded in converted datasets."""

    manifest = copy.deepcopy(artifact.payload)
    manifest["artifact_file_sha256"] = artifact.file_sha256
    return manifest


@dataclass
class _Counts:
    samples: int = 0
    input_tokens: int = 0
    loss_tokens: int = 0


class ObjectiveMaterializationTracker:
    """Write document objective IDs and verify exact receipt accounting."""

    def __init__(
        self, contract: ValidatedObjectiveContract, output_prefix: str
    ) -> None:
        self.contract = contract
        self.output_prefix = output_prefix
        self._path = f"{output_prefix}_objective_ids.bin"
        self._writer = open(self._path, "wb")
        self._counts = {task: _Counts() for task in contract.task_order}
        self._graph_samples = 0
        self._graph_edges = 0
        self._closed = False

    def append(
        self,
        task: str,
        *,
        input_tokens: int,
        loss_tokens: int,
        graph_edges: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("objective materialization tracker is closed")
        if task not in self._counts:
            raise ValueError(f"objective_kind {task!r} is absent from the contract")
        input_count = _positive_int(
            input_tokens, where=f"materialized.{task}.input_tokens"
        )
        loss_count = _positive_int(
            loss_tokens, where=f"materialized.{task}.loss_tokens"
        )
        if loss_count > input_count:
            raise ValueError(
                f"materialized.{task}.loss_tokens={loss_count} exceeds "
                f"input_tokens={input_count}"
            )
        edges = _positive_int(
            graph_edges,
            where=f"materialized.{task}.graph_edges",
            allow_zero=True,
        )
        np.asarray([OBJECTIVE_IDS[task]], dtype=np.uint8).tofile(self._writer)
        row = self._counts[task]
        row.samples += 1
        row.input_tokens += input_count
        row.loss_tokens += loss_count
        if edges:
            self._graph_samples += 1
            self._graph_edges += edges

    def close(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("objective materialization tracker is closed")
        self._writer.close()
        self._closed = True

        realized = self.contract.payload["realized"]
        for task, counts in self._counts.items():
            expected = realized[task]
            for field in ("samples", "input_tokens", "loss_tokens"):
                actual_value = getattr(counts, field)
                expected_value = int(expected[field])
                if actual_value != expected_value:
                    raise ValueError(
                        f"materialized {field} for {task}={actual_value} differs "
                        f"from contract={expected_value}"
                    )
        graph = self.contract.payload["graph_auxiliary"]
        if self._graph_samples != int(graph["eligible_samples"]):
            raise ValueError(
                "materialized graph eligible_samples="
                f"{self._graph_samples} differs from contract="
                f"{graph['eligible_samples']}"
            )
        if self._graph_edges != int(graph["positive_edges"]):
            raise ValueError(
                f"materialized graph positive_edges={self._graph_edges} differs "
                f"from contract={graph['positive_edges']}"
            )
        return {
            "schema": OBJECTIVE_CONTRACT_SCHEMA,
            "sha256": self.contract.sha256,
            "payload": copy.deepcopy(self.contract.payload),
            "objective_id_sidecar": {
                "path": os.path.basename(self._path),
                "dtype": "uint8",
                "document_aligned": True,
            },
        }

    def abort_close(self) -> None:
        if not self._closed:
            self._writer.close()
            self._closed = True


def validate_materialized_objective_contract(
    value: object,
    *,
    base_dir: str | None = None,
    document_count: int | None = None,
) -> ValidatedObjectiveContract:
    """Validate an embedded converter receipt and its objective-ID sidecar."""

    wrapper = _mapping(value, where="objective_contract")
    if wrapper.get("schema") != OBJECTIVE_CONTRACT_SCHEMA:
        raise ValueError(
            f"objective_contract.schema must be {OBJECTIVE_CONTRACT_SCHEMA!r}"
        )
    payload = _mapping(wrapper.get("payload"), where="objective_contract.payload")
    validated = validate_objective_contract(payload)
    if wrapper.get("sha256") != validated.sha256:
        raise ValueError("objective_contract.sha256 does not match its payload")
    sidecar = _mapping(
        wrapper.get("objective_id_sidecar"),
        where="objective_contract.objective_id_sidecar",
    )
    if sidecar.get("dtype") != "uint8" or sidecar.get("document_aligned") is not True:
        raise ValueError(
            "objective_contract.objective_id_sidecar must be document-aligned uint8"
        )
    rel_path = sidecar.get("path")
    if not isinstance(rel_path, str) or not rel_path or os.path.isabs(rel_path):
        raise ValueError("objective_contract objective ID path must be relative")
    if base_dir is not None:
        path = os.path.normpath(os.path.join(base_dir, rel_path))
        base = os.path.normpath(base_dir)
        if path != base and not path.startswith(base + os.sep):
            raise ValueError("objective_contract objective ID path escapes dataset")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        expected_documents = int(validated.payload["totals"]["samples"])
        if document_count is not None and document_count != expected_documents:
            raise ValueError(
                "objective contract sample count must equal document_count: "
                f"samples={expected_documents}, documents={document_count}"
            )
        if os.path.getsize(path) != expected_documents:
            raise ValueError(
                "objective ID sidecar byte count must equal document_count: "
                f"bytes={os.path.getsize(path)}, documents={expected_documents}"
            )
        objective_ids = np.memmap(path, mode="r", dtype=np.uint8)
        allowed_ids = {OBJECTIVE_IDS[task] for task in validated.task_order}
        present_ids = {int(value) for value in np.unique(objective_ids)}
        unknown_ids = sorted(present_ids - allowed_ids)
        if unknown_ids:
            raise ValueError(
                f"objective ID sidecar contains unknown objective IDs: {unknown_ids}"
            )
        histogram = np.bincount(objective_ids, minlength=256)
        expected_histogram = {
            OBJECTIVE_IDS[task]: int(validated.planned_samples[task])
            for task in validated.task_order
        }
        actual_histogram = {
            objective_id: int(histogram[objective_id])
            for objective_id in sorted(allowed_ids)
        }
        if actual_histogram != expected_histogram:
            raise ValueError(
                "objective ID sidecar histogram differs from objective contract: "
                f"expected={expected_histogram}, got={actual_histogram}"
            )
    return validated


def validate_materialized_objective_artifact(
    value: object,
    *,
    objective_contract: ValidatedObjectiveContract,
    document_count: int,
) -> None:
    """Validate the artifact binding embedded by the canonical converter."""

    binding = _mapping(value, where="objective_materialization")
    expected_keys = {
        "schema",
        "artifact_file_sha256",
        "documents",
        "objective_contract",
        "parquet_shards",
        "converter",
        "artifact_set_sha256",
    }
    if set(binding) != expected_keys:
        raise ValueError(
            f"objective_materialization keys must be exactly {sorted(expected_keys)}"
        )
    if binding.get("schema") != OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA:
        raise ValueError("objective_materialization schema is invalid")
    for field in ("artifact_set_sha256", "artifact_file_sha256"):
        value = binding.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"objective_materialization.{field} must be sha256 hex")
    artifact_payload = copy.deepcopy(dict(binding))
    artifact_payload.pop("artifact_file_sha256")
    expected_set_sha256 = artifact_payload.pop("artifact_set_sha256")
    if expected_set_sha256 != _canonical_sha256(artifact_payload):
        raise ValueError(
            "objective_materialization artifact_set_sha256 does not match payload"
        )
    if binding.get("documents") != document_count:
        raise ValueError(
            "objective_materialization.documents does not match dataset document_count"
        )
    contract_ref = _mapping(
        binding.get("objective_contract"),
        where="objective_materialization.objective_contract",
    )
    if contract_ref.get("sha256") != objective_contract.sha256:
        raise ValueError(
            "objective_materialization objective contract hash does not match"
        )
    shards = binding.get("parquet_shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("objective_materialization.parquet_shards must be non-empty")
    names: list[str] = []
    for index, raw_shard in enumerate(shards):
        shard = _mapping(raw_shard, where=f"objective_materialization.parquet_shards[{index}]")
        if set(shard) != {"path", "size_bytes", "sha256"}:
            raise ValueError(
                "objective_materialization shard bindings require path/size_bytes/sha256"
            )
        name = shard.get("path")
        if not isinstance(name, str) or not name or os.path.basename(name) != name:
            raise ValueError("objective_materialization shard path must be a filename")
        _positive_int(shard.get("size_bytes"), where=f"shard[{index}].size_bytes")
        digest = shard.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"objective_materialization shard[{index}] sha256 is invalid")
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("objective_materialization shards must be sorted and unique")


__all__ = [
    "OBJECTIVE_CONTRACT_SCHEMA",
    "OBJECTIVE_GRAPH_SIDECARS",
    "OBJECTIVE_IDS",
    "OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA",
    "OBJECTIVE_TOKEN_SIDE_CHANNELS",
    "ObjectiveMaterializationArtifact",
    "ObjectiveMaterializationTracker",
    "ValidatedObjectiveContract",
    "load_objective_materialization_artifact",
    "materialized_objective_artifact_manifest",
    "validate_materialized_objective_artifact",
    "validate_materialized_objective_contract",
    "validate_objective_contract",
]
