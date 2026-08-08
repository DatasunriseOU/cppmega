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
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from cppmega.megatron.graph_recipe import (
    STAGE1_GRAPH_RELATIONS,
    stage1_graph_recipe_binding,
    validate_stage1_graph_total_loss_contract,
)

OBJECTIVE_CONTRACT_SCHEMA = "cppmega_pre_materialized_objectives_v1"
OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA = (
    "cppmega_objective_materialization_artifact_v2"
)
LEGACY_OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA = (
    "cppmega_objective_materialization_artifact_v1"
)
LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1 = (
    "source_token_predicts_next_v1"
)
OBJECTIVE_TOKEN_SIDE_CHANNELS: tuple[tuple[str, str], ...] = (
    ("loss_mask", "uint8"),
    ("doc_ids", "uint32"),
    ("token_domain_ids", "uint16"),
    ("token_role_ids", "uint16"),
    ("token_entity_ids", "uint32"),
    ("token_scope_ids", "uint32"),
    ("token_source_doc_ids", "uint32"),
    ("token_source_identity_ids", "uint64"),
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
    ("token_chunk_kinds", "ragged_1d", "uint8"),
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
GRAPH_RELATIONS = frozenset(STAGE1_GRAPH_RELATIONS)
OBJECTIVE_SOURCE_SELECTION_SCHEMA = "cppmega_objective_source_selection_v3"
OBJECTIVE_SOURCE_RESUME_SCHEMA = "cppmega_objective_source_resume_v1"
OBJECTIVE_SCHEDULE_WINDOW_SCHEMA = "cppmega_objective_schedule_window_v1"
OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA = "cppmega_objective_schedule_v1"
OBJECTIVE_SCHEDULE_ALGORITHM = (
    "bounded_eligibility_bipartite_graph_capability_v1"
)
GRAPH_ELIGIBILITY_RECEIPT_SCHEMA = "cppmega_objective_graph_eligibility_v1"
OBJECTIVE_REALIZATION_RECEIPT_SCHEMA = "cppmega_objective_realization_v1"
OBJECTIVE_SOURCE_SNAPSHOT_SCHEMA = "cppmega_objective_source_snapshot_v2"
OBJECTIVE_SOURCE_POOL_MANIFEST_SCHEMA = "cppmega_ci_objective_pool_manifest_v1"
OBJECTIVE_SOURCE_POOL_SCHEDULE = "alternate_primary_seed_v1"
OBJECTIVE_SOURCE_POOL_NAMES = ("primary_ci", "objective_seed")
OBJECTIVE_SOURCE_POOL_MANIFEST_PATH = "objective_source_pool_manifest.json"
OBJECTIVE_CI_EXPORT_RECEIPT_PATH = "ci_export_receipt.json"
OBJECTIVE_CI_EXPORT_SCHEMAS = {
    "cppmega_ci_content_store_case5_export_v2",
    "cppmega_ci_content_store_case5_export_v4",
}
OBJECTIVE_SOURCE_BUCKETS = (1024, 2048, 4096, 8192, 16384)
SUPPORTED_OBJECTIVE_SOURCE_BUCKETS = (*OBJECTIVE_SOURCE_BUCKETS, 32768, 65536)
OBJECTIVE_SOURCE_BUCKET_PROFILES = frozenset(
    {OBJECTIVE_SOURCE_BUCKETS, SUPPORTED_OBJECTIVE_SOURCE_BUCKETS}
)
_BOUNDED_SOURCE_SAMPLING_MODE = (
    "deterministic_shard_row_group_record_batch_shuffle_v2"
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


def _canonical_value_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_digest(value: object, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _source_file_records(
    raw_records: object,
    *,
    where: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"{where} must be a non-empty list")
    records: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, where=f"{where}[{index}]")
        if set(record) != {"path", "rows", "size_bytes", "sha256"}:
            raise ValueError(f"{where}[{index}] file binding keys are invalid")
        raw_path = record.get("path")
        path = PurePosixPath(raw_path) if isinstance(raw_path, str) else None
        if (
            path is None
            or not raw_path
            or path.is_absolute()
            or path.as_posix() != raw_path
            or ".." in path.parts
            or path.suffix != ".parquet"
        ):
            raise ValueError(f"{where}[{index}].path is invalid")
        paths.append(raw_path)
        records.append(
            {
                "path": raw_path,
                "rows": _positive_int(
                    record.get("rows"), where=f"{where}[{index}].rows"
                ),
                "size_bytes": _positive_int(
                    record.get("size_bytes"),
                    where=f"{where}[{index}].size_bytes",
                ),
                "sha256": _sha256_digest(
                    record.get("sha256"),
                    where=f"{where}[{index}].sha256",
                ),
            }
        )
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{where} paths must be unique and sorted")
    return records


def _source_artifact_set_sha256(records: list[dict[str, Any]]) -> str:
    return _canonical_value_sha256(
        [
            {
                "path": record["path"],
                "size": record["size_bytes"],
                "sha256": record["sha256"],
            }
            for record in records
        ]
    )


def _validate_bounded_source_pool(
    raw_pool: object,
    *,
    sequence_length: int,
    where: str,
) -> tuple[list[dict[str, Any]], int]:
    pool = _mapping(raw_pool, where=where)
    if set(pool) != {
        "schema",
        "sequence_length",
        "file_count",
        "row_count",
        "files",
        "sampling",
        "artifact_set_sha256",
    }:
        raise ValueError(f"{where} keys are invalid")
    if (
        pool.get("schema") != "cppmega_objective_source_snapshot_v1"
        or pool.get("sequence_length") != sequence_length
    ):
        raise ValueError(f"{where} schema or sequence length is invalid")
    records = _source_file_records(pool.get("files"), where=f"{where}.files")
    row_count = sum(record["rows"] for record in records)
    if (
        pool.get("file_count") != len(records)
        or pool.get("row_count") != row_count
        or pool.get("artifact_set_sha256")
        != _source_artifact_set_sha256(records)
    ):
        raise ValueError(f"{where} file accounting drifted")

    sampling = _mapping(pool.get("sampling"), where=f"{where}.sampling")
    if set(sampling) != {
        "mode",
        "seed",
        "requested_samples",
        "full_passes",
        "tail_rows",
        "min_row_reuse",
        "max_row_reuse",
        "record_batch_rows",
        "producer",
        "ordering",
        "cursor_semantics",
        "final_cursor",
    }:
        raise ValueError(f"{where}.sampling keys are invalid")
    if sampling.get("mode") != _BOUNDED_SOURCE_SAMPLING_MODE:
        raise ValueError(f"{where}.sampling.mode is invalid")
    _positive_int(
        sampling.get("seed"), where=f"{where}.sampling.seed", allow_zero=True
    )
    requested = _positive_int(
        sampling.get("requested_samples"),
        where=f"{where}.sampling.requested_samples",
    )
    full_passes, tail_rows = divmod(requested, row_count)
    if (
        sampling.get("full_passes") != full_passes
        or sampling.get("tail_rows") != tail_rows
        or sampling.get("min_row_reuse") != full_passes
        or sampling.get("max_row_reuse")
        != full_passes + int(tail_rows > 0)
    ):
        raise ValueError(f"{where}.sampling reuse accounting drifted")
    batch_rows = _positive_int(
        sampling.get("record_batch_rows"),
        where=f"{where}.sampling.record_batch_rows",
    )
    producer = _mapping(
        sampling.get("producer"), where=f"{where}.sampling.producer"
    )
    if set(producer) != {"name", "version", "row_group_rows"} or (
        producer.get("name") != "pyarrow.parquet.ParquetFile.iter_batches"
        or producer.get("version") != 1
    ):
        raise ValueError(f"{where}.sampling.producer is invalid")
    row_group_rows = producer.get("row_group_rows")
    if (
        not isinstance(row_group_rows, list)
        or len(row_group_rows) != len(records)
    ):
        raise ValueError(f"{where}.sampling.producer row groups are invalid")
    normalized_groups: list[list[int]] = []
    for index, raw_groups in enumerate(row_group_rows):
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError(f"{where}.sampling.producer row groups are invalid")
        groups = [
            _positive_int(
                rows,
                where=f"{where}.sampling.producer.row_group_rows[{index}]",
            )
            for rows in raw_groups
        ]
        if sum(groups) != records[index]["rows"]:
            raise ValueError(f"{where}.sampling.producer row counts drifted")
        normalized_groups.append(groups)
    if sampling.get("ordering") != {
        "permutation": "sha256_sort_key_v1",
        "epochs": "ascending",
        "shards": "seeded_permutation_per_epoch",
        "row_groups": "seeded_permutation_per_shard_epoch",
        "record_batches": "physical_order_within_row_group",
        "rows": "seeded_permutation_within_record_batch",
    } or sampling.get("cursor_semantics") != "last_yielded_row_v1":
        raise ValueError(f"{where}.sampling ordering or cursor semantics drifted")

    cursor = _mapping(
        sampling.get("final_cursor"), where=f"{where}.sampling.final_cursor"
    )
    cursor_keys = {
        "epoch",
        "shard_position",
        "shard_index",
        "row_group_position",
        "row_group_index",
        "record_batch_index",
        "row_shuffle_position",
        "row_index_in_record_batch",
        "source_index",
    }
    if set(cursor) != cursor_keys:
        raise ValueError(f"{where}.sampling.final_cursor keys are invalid")
    values = {
        key: _positive_int(
            cursor.get(key),
            where=f"{where}.sampling.final_cursor.{key}",
            allow_zero=True,
        )
        for key in cursor_keys
    }
    if (
        values["source_index"] != requested - 1
        or values["epoch"] != (requested - 1) // row_count
        or values["shard_position"] >= len(records)
        or values["shard_index"] >= len(records)
    ):
        raise ValueError(f"{where}.sampling.final_cursor is outside the source")
    groups = normalized_groups[values["shard_index"]]
    if (
        values["row_group_position"] >= len(groups)
        or values["row_group_index"] >= len(groups)
    ):
        raise ValueError(f"{where}.sampling.final_cursor row group is invalid")
    group_rows = groups[values["row_group_index"]]
    batch_start = values["record_batch_index"] * batch_rows
    current_batch_rows = min(batch_rows, group_rows - batch_start)
    if (
        current_batch_rows < 1
        or values["row_shuffle_position"] >= current_batch_rows
        or values["row_index_in_record_batch"] >= current_batch_rows
    ):
        raise ValueError(f"{where}.sampling.final_cursor record batch is invalid")
    return records, requested


def _validate_two_pool_source_snapshot(contract: Mapping[str, Any]) -> None:
    raw_snapshot = contract.get("source_snapshot")
    if raw_snapshot is None:
        return
    snapshot = _mapping(raw_snapshot, where="source_snapshot")
    schema = snapshot.get("schema")
    if schema == "cppmega_objective_source_snapshot_v1":
        return
    if schema != OBJECTIVE_SOURCE_SNAPSHOT_SCHEMA:
        raise ValueError("source_snapshot schema is unsupported")
    if set(snapshot) != {
        "schema",
        "sequence_length",
        "algorithm",
        "pool_order",
        "source_pool_manifest",
        "ci_export_receipt",
        "pools",
    }:
        raise ValueError("source_snapshot keys are invalid")
    sequence_length = _positive_int(
        snapshot.get("sequence_length"), where="source_snapshot.sequence_length"
    )
    if (
        snapshot.get("algorithm") != OBJECTIVE_SOURCE_POOL_SCHEDULE
        or snapshot.get("pool_order") != list(OBJECTIVE_SOURCE_POOL_NAMES)
    ):
        raise ValueError("source_snapshot pool schedule drifted")
    for field, expected_path in (
        ("source_pool_manifest", OBJECTIVE_SOURCE_POOL_MANIFEST_PATH),
        ("ci_export_receipt", OBJECTIVE_CI_EXPORT_RECEIPT_PATH),
    ):
        descriptor = _mapping(
            snapshot.get(field), where=f"source_snapshot.{field}"
        )
        if set(descriptor) != {"path", "size_bytes", "sha256"} or (
            descriptor.get("path") != expected_path
        ):
            raise ValueError(f"source_snapshot.{field} descriptor is invalid")
        _positive_int(
            descriptor.get("size_bytes"),
            where=f"source_snapshot.{field}.size_bytes",
        )
        _sha256_digest(
            descriptor.get("sha256"), where=f"source_snapshot.{field}.sha256"
        )

    pools = _mapping(snapshot.get("pools"), where="source_snapshot.pools")
    if set(pools) != set(OBJECTIVE_SOURCE_POOL_NAMES):
        raise ValueError("source_snapshot.pools are invalid")
    requested = {
        name: _validate_bounded_source_pool(
            pools[name],
            sequence_length=sequence_length,
            where=f"source_snapshot.pools.{name}",
        )[1]
        for name in OBJECTIVE_SOURCE_POOL_NAMES
    }
    selection = _mapping(
        contract.get("source_selection"), where="source_selection"
    )
    consumed = _positive_int(
        selection.get("source_rows_consumed"),
        where="source_selection.source_rows_consumed",
    )
    if requested != {
        "primary_ci": (consumed + 1) // 2,
        "objective_seed": consumed // 2,
    }:
        raise ValueError("source_snapshot pool consumption drifted")
    cursor = _mapping(
        _mapping(
            selection.get("resume"), where="source_selection.resume"
        ).get("last_yielded_cursor"),
        where="source_selection.resume.last_yielded_cursor",
    )
    last_pool = (consumed - 1) % 2
    if (
        cursor.get("pool_index") != last_pool
        or cursor.get("pool_source_index")
        != requested[OBJECTIVE_SOURCE_POOL_NAMES[last_pool]] - 1
        or cursor.get("primary_rows_yielded") != requested["primary_ci"]
        or cursor.get("objective_seed_rows_yielded")
        != requested["objective_seed"]
        or cursor.get("next_pool_index") != consumed % 2
    ):
        raise ValueError("source_selection two-pool replay cursor drifted")
    windows = selection.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("source_selection windows are missing")
    for index, raw_window in enumerate(windows):
        window = _mapping(raw_window, where=f"source_selection.windows[{index}]")
        selected = window.get("selected_source_indices")
        if (
            not isinstance(selected, list)
            or not any(value % 2 == 0 for value in selected)
            or not any(value % 2 == 1 for value in selected)
        ):
            raise ValueError(
                f"source_selection.windows[{index}] must use both source pools"
            )


def _schedule_receipt_locations(value: object, *, where: str) -> list[str]:
    """Find every schedule receipt so production has one unambiguous source."""

    locations: list[str] = []
    if isinstance(value, Mapping):
        if value.get("schema") == OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA:
            locations.append(where)
        for key, child in value.items():
            locations.extend(
                _schedule_receipt_locations(
                    child,
                    where=f"{where}.{key}",
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            locations.extend(
                _schedule_receipt_locations(
                    child,
                    where=f"{where}[{index}]",
                )
            )
    return locations


def _require_canonical_schedule_receipt(contract: Mapping[str, Any]) -> None:
    locations = _schedule_receipt_locations(contract, where="contract")
    expected = ["contract.source_selection.schedule"]
    if not locations:
        raise ValueError(
            "production objective contract requires one canonical schedule "
            "receipt at source_selection.schedule"
        )
    if locations != expected:
        raise ValueError(
            "production objective contract has ambiguous schedule receipts; "
            f"expected only source_selection.schedule, found {locations}"
        )


def _validate_graph_eligibility_receipt(
    raw_receipt: object,
    *,
    task: str,
    relations: tuple[str, ...],
    where: str,
) -> tuple[bool, int]:
    receipt = _mapping(raw_receipt, where=where)
    required_keys = {
        "schema",
        "objective",
        "eligible",
        "reason",
        "positive_edges",
        "relations",
        "route_mode",
        "route_receipt",
    }
    keys = set(receipt)
    if keys != required_keys and keys != required_keys | {"detail"}:
        raise ValueError(f"{where} keys are invalid")
    if receipt.get("schema") != GRAPH_ELIGIBILITY_RECEIPT_SCHEMA:
        raise ValueError(f"{where}.schema is invalid")
    if receipt.get("objective") != task:
        raise ValueError(f"{where}.objective differs from its assignment")
    if receipt.get("relations") != list(relations):
        raise ValueError(f"{where}.relations differ from graph_auxiliary.relations")
    positive_edges = _positive_int(
        receipt.get("positive_edges"),
        where=f"{where}.positive_edges",
        allow_zero=True,
    )
    eligible = receipt.get("eligible")
    if not isinstance(eligible, bool) or eligible != (positive_edges > 0):
        raise ValueError(f"{where}.eligible is inconsistent with positive_edges")
    reason = receipt.get("reason")
    if eligible:
        if reason is not None:
            raise ValueError(f"{where}.reason must be null when eligible")
    elif not isinstance(reason, str) or not reason:
        raise ValueError(f"{where}.reason must explicitly explain ineligibility")

    route_mode = receipt.get("route_mode")
    route_receipt = receipt.get("route_receipt")
    if route_mode == "unavailable":
        if (
            reason != "missing_exact_source_token_route_map"
            or route_receipt is not None
        ):
            raise ValueError(f"{where} unavailable route receipt is inconsistent")
    else:
        route = _mapping(route_receipt, where=f"{where}.route_receipt")
        if route.get("mode") != route_mode:
            raise ValueError(f"{where}.route_mode differs from route_receipt.mode")
    if task in {"commit_diff", "pre_to_post"} and (
        eligible
        or route_mode != "excluded"
        or reason != "exact_source_route_map_unavailable"
    ):
        raise ValueError(
            f"{where}: commit objectives without exact route maps must be "
            "explicitly graph-ineligible"
        )
    return eligible, positive_edges


def _validate_objective_realization_receipt(
    raw_receipt: object,
    *,
    task: str,
    where: str,
) -> None:
    receipt = _mapping(raw_receipt, where=where)
    expected_keys = {
        "schema",
        "task",
        "selected_packet_index",
        "example_sha256",
        "input_tokens",
        "loss_tokens",
    }
    if set(receipt) != expected_keys:
        raise ValueError(f"{where} keys are invalid")
    if receipt.get("schema") != OBJECTIVE_REALIZATION_RECEIPT_SCHEMA:
        raise ValueError(f"{where}.schema is invalid")
    if receipt.get("task") != task:
        raise ValueError(f"{where}.task differs from its assignment")
    _positive_int(
        receipt.get("selected_packet_index"),
        where=f"{where}.selected_packet_index",
        allow_zero=True,
    )
    digest = receipt.get("example_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{where}.example_sha256 is invalid")
    _positive_int(
        receipt.get("input_tokens"),
        where=f"{where}.input_tokens",
        allow_zero=True,
    )
    _positive_int(
        receipt.get("loss_tokens"),
        where=f"{where}.loss_tokens",
        allow_zero=True,
    )


def validate_objective_source_selection(
    raw_receipt: object,
    *,
    total_samples: int,
    quota_window_samples: int,
    window_quotas: Mapping[str, int],
    graph_relations: tuple[str, ...],
) -> None:
    """Validate the canonical bounded source-selection schedule receipt."""

    receipt = _mapping(raw_receipt, where="source_selection")
    expected_keys = {
        "schema",
        "algorithm",
        "output_samples",
        "source_rows_consumed",
        "unused_buffered_sources",
        "quota_window_samples",
        "quota_lookahead_samples",
        "max_source_pool_samples",
        "max_source_pool_observed",
        "required_graph_relations",
        "windows",
        "windows_sha256",
        "resume",
        "schedule",
    }
    if set(receipt) != expected_keys:
        raise ValueError(
            f"source_selection keys must be exactly {sorted(expected_keys)}"
        )
    if receipt.get("schema") != OBJECTIVE_SOURCE_SELECTION_SCHEMA:
        raise ValueError("source_selection.schema is invalid")
    if receipt.get("algorithm") != OBJECTIVE_SCHEDULE_ALGORITHM:
        raise ValueError("source_selection.algorithm is invalid")
    if receipt.get("output_samples") != total_samples:
        raise ValueError("source_selection.output_samples differs from totals.samples")
    if receipt.get("quota_window_samples") != quota_window_samples:
        raise ValueError("source_selection quota window differs from the contract")
    lookahead = _positive_int(
        receipt.get("quota_lookahead_samples"),
        where="source_selection.quota_lookahead_samples",
        allow_zero=True,
    )
    max_pool = _positive_int(
        receipt.get("max_source_pool_samples"),
        where="source_selection.max_source_pool_samples",
    )
    if max_pool != quota_window_samples + lookahead:
        raise ValueError("source_selection max pool does not bind its lookahead")
    max_observed = _positive_int(
        receipt.get("max_source_pool_observed"),
        where="source_selection.max_source_pool_observed",
    )
    if not quota_window_samples <= max_observed <= max_pool:
        raise ValueError("source_selection max observed pool is outside its bound")
    consumed = _positive_int(
        receipt.get("source_rows_consumed"),
        where="source_selection.source_rows_consumed",
    )
    unused = _positive_int(
        receipt.get("unused_buffered_sources"),
        where="source_selection.unused_buffered_sources",
        allow_zero=True,
    )
    if consumed != total_samples + unused:
        raise ValueError(
            "source_selection source_rows_consumed must equal output plus buffered"
        )
    if receipt.get("required_graph_relations") != list(graph_relations):
        raise ValueError("source_selection graph relations drifted")

    raw_windows = receipt.get("windows")
    if not isinstance(raw_windows, list):
        raise ValueError("source_selection.windows must be a list")
    if len(raw_windows) != total_samples // quota_window_samples:
        raise ValueError("source_selection window count is invalid")
    window_digest = _canonical_value_sha256(raw_windows)
    if receipt.get("windows_sha256") != window_digest:
        raise ValueError("source_selection.windows_sha256 is invalid")
    schedule = _mapping(receipt.get("schedule"), where="source_selection.schedule")
    if dict(schedule) != {
        "schema": OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA,
        "algorithm": OBJECTIVE_SCHEDULE_ALGORITHM,
        "windows_sha256": window_digest,
    }:
        raise ValueError("source_selection.schedule binding is invalid")

    expected_window_keys = {
        "schema",
        "algorithm",
        "start_step",
        "output_samples",
        "source_pool_samples",
        "source_pool_source_indices",
        "source_rows_consumed",
        "selected_source_indices",
        "task_counts",
        "assignments",
        "graph_positive_assignments",
        "graph_positive_edges",
    }
    expected_assignment_keys = {
        "source_index",
        "source_pool_index",
        "task",
        "realization",
        "graph_eligibility",
    }
    selected_sources: set[int] = set()
    previous_consumed = 0
    for window_index, raw_window in enumerate(raw_windows):
        where = f"source_selection.windows[{window_index}]"
        window = _mapping(raw_window, where=where)
        if set(window) != expected_window_keys:
            raise ValueError(f"{where} keys are invalid")
        if window.get("schema") != OBJECTIVE_SCHEDULE_WINDOW_SCHEMA:
            raise ValueError(f"{where}.schema is invalid")
        if window.get("algorithm") != OBJECTIVE_SCHEDULE_ALGORITHM:
            raise ValueError(f"{where}.algorithm is invalid")
        if window.get("start_step") != window_index * quota_window_samples:
            raise ValueError(f"{where}.start_step is not contiguous")
        if window.get("output_samples") != quota_window_samples:
            raise ValueError(f"{where}.output_samples differs from the quota window")
        pool_samples = _positive_int(
            window.get("source_pool_samples"),
            where=f"{where}.source_pool_samples",
        )
        if not quota_window_samples <= pool_samples <= max_pool:
            raise ValueError(f"{where}.source_pool_samples is outside its bound")
        raw_pool_source_indices = window.get("source_pool_source_indices")
        if (
            not isinstance(raw_pool_source_indices, list)
            or len(raw_pool_source_indices) != pool_samples
        ):
            raise ValueError(f"{where}.source_pool_source_indices is invalid")
        pool_source_indices = [
            _positive_int(
                source_index,
                where=f"{where}.source_pool_source_indices",
                allow_zero=True,
            )
            for source_index in raw_pool_source_indices
        ]
        if len(set(pool_source_indices)) != len(pool_source_indices):
            raise ValueError(f"{where}.source_pool_source_indices reuses a source row")
        window_consumed = _positive_int(
            window.get("source_rows_consumed"),
            where=f"{where}.source_rows_consumed",
        )
        if not previous_consumed <= window_consumed <= consumed:
            raise ValueError(f"{where}.source_rows_consumed is not monotonic")
        if any(source_index >= window_consumed for source_index in pool_source_indices):
            raise ValueError(
                f"{where}.source_pool_source_indices is ahead of the consumed cursor"
            )
        previous_consumed = window_consumed
        assignments = window.get("assignments")
        if not isinstance(assignments, list) or len(assignments) != quota_window_samples:
            raise ValueError(f"{where}.assignments must fill one quota window")
        rows = [
            _mapping(assignment, where=f"{where}.assignments[{index}]")
            for index, assignment in enumerate(assignments)
        ]
        if any(set(row) != expected_assignment_keys for row in rows):
            raise ValueError(f"{where} assignment keys are invalid")
        source_indices = [
            _positive_int(
                row.get("source_index"),
                where=f"{where}.assignments.source_index",
                allow_zero=True,
            )
            for row in rows
        ]
        if len(set(source_indices)) != quota_window_samples:
            raise ValueError(f"{where} reuses a source row")
        if selected_sources.intersection(source_indices):
            raise ValueError("source_selection reuses a source row across windows")
        selected_sources.update(source_indices)
        if window.get("selected_source_indices") != source_indices:
            raise ValueError(f"{where}.selected_source_indices drifted")
        task_counts = Counter(str(row.get("task")) for row in rows)
        if window.get("task_counts") != dict(sorted(task_counts.items())):
            raise ValueError(f"{where}.task_counts differ from assignments")
        if task_counts != Counter(window_quotas):
            raise ValueError(f"{where}.task_counts differ from Hamilton quotas")

        positive_assignments = 0
        positive_edges = 0
        for assignment_index, row in enumerate(rows):
            pool_index = _positive_int(
                row.get("source_pool_index"),
                where=f"{where}.assignments[{assignment_index}].source_pool_index",
                allow_zero=True,
            )
            if pool_index >= pool_samples:
                raise ValueError(f"{where} assignment pool index is invalid")
            if source_indices[assignment_index] != pool_source_indices[pool_index]:
                raise ValueError(f"{where} assignment source pool binding drifted")
            _validate_objective_realization_receipt(
                row.get("realization"),
                task=str(row.get("task")),
                where=f"{where}.assignments[{assignment_index}].realization",
            )
            eligible, edges = _validate_graph_eligibility_receipt(
                row.get("graph_eligibility"),
                task=str(row.get("task")),
                relations=graph_relations,
                where=f"{where}.assignments[{assignment_index}].graph_eligibility",
            )
            positive_assignments += int(eligible)
            positive_edges += edges
        if window.get("graph_positive_assignments") != positive_assignments:
            raise ValueError(f"{where}.graph_positive_assignments drifted")
        if window.get("graph_positive_edges") != positive_edges:
            raise ValueError(f"{where}.graph_positive_edges drifted")
        if positive_assignments < 1:
            raise ValueError(f"{where} has no graph-positive assignment")

    if previous_consumed != consumed:
        raise ValueError("source_selection final window consumption drifted")
    resume = _mapping(receipt.get("resume"), where="source_selection.resume")
    if resume.get("schema") != OBJECTIVE_SOURCE_RESUME_SCHEMA:
        raise ValueError("source_selection.resume.schema is invalid")
    if resume.get("cursor_semantics") != (
        "replay_buffered_rows_then_continue_after_last_yielded_v1"
    ):
        raise ValueError("source_selection.resume.cursor_semantics is invalid")
    last_cursor = _mapping(
        resume.get("last_yielded_cursor"),
        where="source_selection.resume.last_yielded_cursor",
    )
    if last_cursor.get("source_index") != consumed - 1:
        raise ValueError("source_selection resume cursor is not the final source row")
    buffered = resume.get("buffered_source_cursors")
    if not isinstance(buffered, list) or len(buffered) != unused:
        raise ValueError("source_selection buffered resume cursors are invalid")


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


ObjectiveShardStat = tuple[int, int, int, int, int]


def validate_objective_contract(
    raw_contract: Mapping[str, Any],
    *,
    require_schedule_receipt: bool = False,
) -> ValidatedObjectiveContract:
    """Validate one upstream objective receipt without filling any defaults.

    ``require_schedule_receipt`` is reserved for the production Megatron
    handoff.  Generic callers may inspect older receipts, but production
    ingress must prove exactly one canonical ``source_selection.schedule``.
    """

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
    zero_required_quotas = sorted(
        task for task in REQUIRED_PRODUCTION_OBJECTIVES if window_quotas[task] == 0
    )
    if zero_required_quotas:
        raise ValueError(
            "required objectives must have a nonzero planned quota in every "
            f"Hamilton window: {zero_required_quotas}"
        )
    window_count = total_samples // window

    planned = _mapping(contract.get("planned_samples"), where="planned_samples")
    if set(planned) != set(task_order):
        raise ValueError("planned_samples keys must exactly match task_order")
    planned_samples = {
        task: _positive_int(
            planned[task], where=f"planned_samples.{task}", allow_zero=True
        )
        for task in task_order
    }
    zero_required_planned = sorted(
        task for task in REQUIRED_PRODUCTION_OBJECTIVES if planned_samples[task] == 0
    )
    if zero_required_planned:
        raise ValueError(
            "required objectives must have nonzero planned_samples: "
            f"{zero_required_planned}"
        )
    non_window_aligned = {
        task: samples
        for task, samples in planned_samples.items()
        if samples % window_count
    }
    if non_window_aligned:
        raise ValueError(
            "planned_samples cannot be represented by identical Hamilton quota "
            f"windows: window_count={window_count}, values={non_window_aligned}"
        )
    planned_per_window = {
        task: samples // window_count for task, samples in planned_samples.items()
    }
    if planned_per_window != window_quotas:
        raise ValueError(
            "planned_samples do not match the deterministic Hamilton schedule "
            f"in every quota window: expected={window_quotas}, "
            f"got={planned_per_window}"
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
    if len(set(relations)) != len(relations):
        raise ValueError("graph_auxiliary.relations contains duplicates")
    unknown_relations = sorted(set(relations) - GRAPH_RELATIONS)
    if unknown_relations:
        raise ValueError(
            "graph_auxiliary.relations contains unknown relations: "
            f"{unknown_relations}"
        )
    validate_stage1_graph_total_loss_contract(graph)
    eligible_samples = _positive_int(
        graph.get("eligible_samples"), where="graph_auxiliary.eligible_samples"
    )
    if eligible_samples > total_samples:
        raise ValueError(
            "graph_auxiliary.eligible_samples cannot exceed totals.samples"
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
    if graph.get("runtime") != "megatron_dsa_indexer_v1":
        raise ValueError("graph_auxiliary.runtime must be 'megatron_dsa_indexer_v1'")
    if graph.get("pair_mask") != "causal_same_document_upstream_v1":
        raise ValueError(
            "graph_auxiliary.pair_mask must be 'causal_same_document_upstream_v1'"
        )
    if graph.get("chunk_edge_expansion") != "cartesian_token_spans_v1":
        raise ValueError(
            "graph_auxiliary.chunk_edge_expansion must be 'cartesian_token_spans_v1'"
        )
    if "source_selection" in contract:
        validate_objective_source_selection(
            contract["source_selection"],
            total_samples=total_samples,
            quota_window_samples=window,
            window_quotas=window_quotas,
            graph_relations=tuple(relations),
        )
    _validate_two_pool_source_snapshot(contract)
    if require_schedule_receipt:
        _require_canonical_schedule_receipt(contract)

    materialization = _mapping(contract.get("materialization"), where="materialization")
    if materialization.get("format") != "shifted_lm_document_v1":
        raise ValueError("materialization.format must be 'shifted_lm_document_v1'")
    expected_materialization_columns = {
        "token_column": "input_ids",
        "loss_mask_column": "loss_mask",
        "loss_mask_alignment": LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1,
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


def validate_production_objective_contract(
    raw_contract: Mapping[str, Any],
) -> ValidatedObjectiveContract:
    """Validate the fail-closed contract admitted to production training."""

    return validate_objective_contract(raw_contract, require_schedule_receipt=True)


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


def _verify_two_pool_source_bindings(
    contract: Mapping[str, Any],
    *,
    root: Path,
) -> None:
    raw_snapshot = contract.get("source_snapshot")
    if not isinstance(raw_snapshot, Mapping) or (
        raw_snapshot.get("schema") != OBJECTIVE_SOURCE_SNAPSHOT_SCHEMA
    ):
        return
    snapshot = raw_snapshot
    bound: dict[str, tuple[Mapping[str, Any], bytes]] = {}
    for field in ("source_pool_manifest", "ci_export_receipt"):
        descriptor = _mapping(
            snapshot.get(field), where=f"source_snapshot.{field}"
        )
        path = _artifact_file(
            root,
            descriptor.get("path"),
            where=f"source_snapshot.{field}.path",
        )
        raw = path.read_bytes()
        if (
            len(raw) != descriptor.get("size_bytes")
            or hashlib.sha256(raw).hexdigest() != descriptor.get("sha256")
        ):
            raise ValueError(f"source_snapshot.{field} byte binding drifted")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"source_snapshot.{field} is not valid JSON") from exc
        bound[field] = (
            _mapping(payload, where=f"source_snapshot.{field}"),
            raw,
        )

    manifest, _manifest_raw = bound["source_pool_manifest"]
    raw_sequence_lengths = manifest.get("sequence_lengths")
    sequence_lengths = (
        tuple(raw_sequence_lengths)
        if isinstance(raw_sequence_lengths, list)
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw_sequence_lengths
        )
        else ()
    )
    if set(manifest) != {
        "schema",
        "algorithm",
        "sequence_lengths",
        "ci_export",
        "primary_ci",
        "objective_seed",
        "producer",
    } or (
        manifest.get("schema") != OBJECTIVE_SOURCE_POOL_MANIFEST_SCHEMA
        or manifest.get("algorithm") != OBJECTIVE_SOURCE_POOL_SCHEDULE
        or sequence_lengths not in OBJECTIVE_SOURCE_BUCKET_PROFILES
    ):
        raise ValueError("source pool manifest contract is invalid")
    producer = _mapping(
        manifest.get("producer"), where="source pool manifest producer"
    )
    if set(producer) != {
        "repository",
        "git_commit",
        "script",
        "script_sha256",
    } or (
        producer.get("repository") != "cppmega"
        or producer.get("script")
        != "scripts/data/prepare_ci_objective_source_manifest.py"
        or not isinstance(producer.get("git_commit"), str)
        or len(producer["git_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in producer["git_commit"]
        )
    ):
        raise ValueError("source pool manifest producer binding is invalid")
    _sha256_digest(
        producer.get("script_sha256"),
        where="source pool manifest producer.script_sha256",
    )

    primary = _mapping(
        manifest.get("primary_ci"), where="source pool manifest primary_ci"
    )
    files_by_length = _mapping(
        primary.get("files_by_sequence_length"),
        where="source pool manifest primary_ci.files_by_sequence_length",
    )
    if (
        set(primary) != {"files_by_sequence_length"}
        or set(files_by_length)
        != {str(bucket) for bucket in sequence_lengths}
    ):
        raise ValueError("source pool manifest primary inventory is invalid")
    manifest_primary = {
        bucket: _source_file_records(
            files_by_length[str(bucket)],
            where=(
                "source pool manifest primary_ci.files_by_sequence_length."
                f"{bucket}"
            ),
        )
        for bucket in sequence_lengths
    }
    seed = _mapping(
        manifest.get("objective_seed"),
        where="source pool manifest objective_seed",
    )
    if set(seed) != {"files"}:
        raise ValueError("source pool manifest objective seed is invalid")
    manifest_seed = _source_file_records(
        seed.get("files"), where="source pool manifest objective_seed.files"
    )
    pools = _mapping(snapshot.get("pools"), where="source_snapshot.pools")
    sequence_length = int(snapshot["sequence_length"])
    if sequence_length not in sequence_lengths:
        raise ValueError(
            "source_snapshot.sequence_length is not in the source pool profile"
        )
    snapshot_primary = _source_file_records(
        _mapping(
            pools.get("primary_ci"), where="source_snapshot.pools.primary_ci"
        ).get("files"),
        where="source_snapshot.pools.primary_ci.files",
    )
    snapshot_seed = _source_file_records(
        _mapping(
            pools.get("objective_seed"),
            where="source_snapshot.pools.objective_seed",
        ).get("files"),
        where="source_snapshot.pools.objective_seed.files",
    )
    if (
        manifest_primary.get(sequence_length) != snapshot_primary
        or manifest_seed != snapshot_seed
    ):
        raise ValueError("source pool manifest differs from source_snapshot pools")

    ci_export = _mapping(
        manifest.get("ci_export"), where="source pool manifest ci_export"
    )
    if set(ci_export) != {
        "path",
        "sha256",
        "schema",
        "status",
        "source_completion",
    } or (
        ci_export.get("path") != "export_receipt.json"
        or ci_export.get("schema") not in OBJECTIVE_CI_EXPORT_SCHEMAS
        or ci_export.get("status") != "complete"
        or ci_export.get("sha256")
        != snapshot["ci_export_receipt"]["sha256"]
    ):
        raise ValueError("source pool manifest CI export binding is invalid")
    receipt, receipt_raw = bound["ci_export_receipt"]
    if (
        hashlib.sha256(receipt_raw).hexdigest() != ci_export.get("sha256")
        or receipt.get("schema") != ci_export.get("schema")
        or receipt.get("status") != "complete"
    ):
        raise ValueError("source pool manifest differs from its CI export receipt")
    completion = _mapping(
        ci_export.get("source_completion"),
        where="source pool manifest ci_export.source_completion",
    )
    if (
        completion.get("schema") != receipt.get("schema")
        or completion.get("status") != "complete"
    ):
        raise ValueError("source pool manifest CI completion binding is invalid")
    if receipt["schema"].endswith("_v4"):
        if (
            receipt.get("completion_mode") != "inventory-exhaustive"
            or receipt.get("production_complete") is not True
            or completion.get("completion_mode") != "inventory-exhaustive"
            or completion.get("production_complete") is not True
        ):
            raise ValueError("production CI source pool is not inventory-exhaustive")
    elif (
        receipt.get("production_complete") is True
        or receipt.get("completion_mode") == "inventory-exhaustive"
        or completion.get("production_complete") is True
        or completion.get("completion_mode") == "inventory-exhaustive"
    ):
        raise ValueError("threshold CI source pool claims exhaustive completion")


def _stat_signature(value: os.stat_result) -> ObjectiveShardStat:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def verify_objective_materialization_shard(
    artifact: ObjectiveMaterializationArtifact,
    shard_path: str | os.PathLike[str],
    *,
    previous_stat: ObjectiveShardStat | None = None,
) -> ObjectiveShardStat:
    """Re-verify one bound parquet and detect replacement during consumption."""

    path = Path(shard_path).resolve()
    try:
        index = artifact.parquet_paths.index(path)
    except ValueError as exc:
        raise ValueError(
            f"parquet shard is not bound by objective artifact: {path}"
        ) from exc
    raw_binding = artifact.payload["parquet_shards"][index]
    binding = _mapping(raw_binding, where=f"parquet_shards[{index}]")
    if binding.get("path") != path.name:
        raise ValueError(f"parquet_shards[{index}].path no longer matches artifact")

    before = path.stat()
    digest = _file_sha256(path)
    after = path.stat()
    before_signature = _stat_signature(before)
    after_signature = _stat_signature(after)
    if before_signature != after_signature:
        raise ValueError(
            f"parquet_shards[{index}] changed while its bytes were being verified"
        )
    if after.st_size != binding.get("size_bytes"):
        raise ValueError(f"parquet_shards[{index}].size_bytes does not match")
    if digest != binding.get("sha256"):
        raise ValueError(f"parquet_shards[{index}].sha256 does not match")
    if previous_stat is not None and after_signature != previous_stat:
        raise ValueError(
            f"parquet_shards[{index}] stat changed while the shard was consumed"
        )
    return after_signature


def load_objective_materialization_artifact(
    path: str | os.PathLike[str],
) -> ObjectiveMaterializationArtifact:
    """Open and byte-verify the canonical CASE1 objective handoff."""

    artifact_path = Path(path).resolve()
    with artifact_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    artifact = dict(_mapping(raw, where="objective materialization artifact"))
    if artifact.get("schema") == LEGACY_OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA:
        raise ValueError(
            "legacy objective materialization artifact schema detected; migration "
            "required: regenerate the objective contract and artifact"
        )
    expected_top = {
        "schema",
        "graph_recipe",
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
    if artifact.get("graph_recipe") != stage1_graph_recipe_binding():
        raise ValueError(
            "objective materialization graph recipe binding is missing or stale; "
            "regenerate the objective artifact"
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
            "objective_contract must contain path, sha256, size_bytes, and file_sha256"
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
    contract = validate_production_objective_contract(
        _mapping(contract_raw, where="objective contract")
    )
    if contract_ref.get("sha256") != contract.sha256:
        raise ValueError("objective_contract.sha256 does not match contract payload")
    _verify_two_pool_source_bindings(contract.payload, root=root)
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
        "loss_mask_alignment",
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
        "loss_mask_alignment": LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1,
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
        raise ValueError(
            f"objective artifact directory contains unlisted parquet: {unlisted}"
        )

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
    require_schedule_receipt: bool = False,
) -> ValidatedObjectiveContract:
    """Validate an embedded converter receipt and its objective-ID sidecar."""

    wrapper = _mapping(value, where="objective_contract")
    if wrapper.get("schema") != OBJECTIVE_CONTRACT_SCHEMA:
        raise ValueError(
            f"objective_contract.schema must be {OBJECTIVE_CONTRACT_SCHEMA!r}"
        )
    payload = _mapping(wrapper.get("payload"), where="objective_contract.payload")
    validated = (
        validate_production_objective_contract(payload)
        if require_schedule_receipt
        else validate_objective_contract(payload)
    )
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
    if binding.get("schema") == LEGACY_OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA:
        raise ValueError(
            "legacy objective materialization artifact schema detected; migration "
            "required: regenerate the objective contract and artifact"
        )
    expected_keys = {
        "schema",
        "graph_recipe",
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
    if binding.get("graph_recipe") != stage1_graph_recipe_binding():
        raise ValueError(
            "objective_materialization graph recipe binding is missing or stale; "
            "regenerate the objective artifact"
        )
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
        shard = _mapping(
            raw_shard, where=f"objective_materialization.parquet_shards[{index}]"
        )
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
            raise ValueError(
                f"objective_materialization shard[{index}] sha256 is invalid"
            )
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("objective_materialization shards must be sorted and unique")


__all__ = [
    "GRAPH_ELIGIBILITY_RECEIPT_SCHEMA",
    "LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1",
    "OBJECTIVE_CONTRACT_SCHEMA",
    "OBJECTIVE_GRAPH_SIDECARS",
    "OBJECTIVE_SCHEDULE_ALGORITHM",
    "OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA",
    "OBJECTIVE_SCHEDULE_WINDOW_SCHEMA",
    "OBJECTIVE_SOURCE_RESUME_SCHEMA",
    "OBJECTIVE_SOURCE_SELECTION_SCHEMA",
    "OBJECTIVE_IDS",
    "OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA",
    "OBJECTIVE_REALIZATION_RECEIPT_SCHEMA",
    "LEGACY_OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA",
    "OBJECTIVE_TOKEN_SIDE_CHANNELS",
    "ObjectiveMaterializationArtifact",
    "ObjectiveMaterializationTracker",
    "ValidatedObjectiveContract",
    "load_objective_materialization_artifact",
    "materialized_objective_artifact_manifest",
    "verify_objective_materialization_shard",
    "validate_materialized_objective_artifact",
    "validate_materialized_objective_contract",
    "validate_objective_contract",
    "validate_production_objective_contract",
    "validate_objective_source_selection",
]
