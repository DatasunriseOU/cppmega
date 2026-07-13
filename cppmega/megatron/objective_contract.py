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
from typing import Any

import numpy as np

OBJECTIVE_CONTRACT_SCHEMA = "cppmega_pre_materialized_objectives_v1"
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
        task: _positive_int(planned[task], where=f"planned_samples.{task}")
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
            row.get("input_tokens"), where=f"realized.{task}.input_tokens"
        )
        loss_tokens = _positive_int(
            row.get("loss_tokens"), where=f"realized.{task}.loss_tokens"
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
    for field in ("global_weight", "bce_weight", "coverage_weight"):
        _fraction(graph.get(field), where=f"graph_auxiliary.{field}", positive=True)
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
        if document_count is not None and os.path.getsize(path) != document_count:
            raise ValueError(
                "objective ID sidecar byte count must equal document_count: "
                f"bytes={os.path.getsize(path)}, documents={document_count}"
            )
    return validated


__all__ = [
    "OBJECTIVE_CONTRACT_SCHEMA",
    "OBJECTIVE_IDS",
    "ObjectiveMaterializationTracker",
    "ValidatedObjectiveContract",
    "validate_materialized_objective_contract",
    "validate_objective_contract",
]
