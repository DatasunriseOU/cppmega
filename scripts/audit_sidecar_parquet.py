#!/usr/bin/env python3
"""Audit cppmega packed parquet sidecar coverage and shape correctness.

The conveyor emits parquet first, then Megatron sidecar indexed data.  This
script is the fail-closed gate before uploading parquet shards to shared object
storage: it checks every selected parquet shard, validates token-aligned sidecar
lengths and graph/span endpoints, and writes JSON/Markdown receipts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def _load_schema_contracts():
    """Load dependency-free data contracts without importing the MLX data package."""

    import cppmega

    data_dir = REPO_ROOT / "cppmega" / "data"

    def load(name: str, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load schema contract: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    tokenizer = load(
        "_cppmega_audit_tokenizer_contract", data_dir / "tokenizer_contract.py"
    )
    source_identity = load(
        "_cppmega_audit_source_identity", data_dir / "source_identity.py"
    )
    data_name = "cppmega.data"
    tokenizer_name = f"{data_name}.tokenizer_contract"
    missing = object()
    previous_data = sys.modules.get(data_name, missing)
    previous_tokenizer = sys.modules.get(tokenizer_name, missing)
    previous_data_attr = cppmega.__dict__.get("data", missing)
    lightweight_data = ModuleType(data_name)
    lightweight_data.__path__ = [str(data_dir)]
    lightweight_data.__package__ = "cppmega"
    try:
        sys.modules[data_name] = lightweight_data
        sys.modules[tokenizer_name] = tokenizer
        cppmega.data = lightweight_data
        domain = load("_cppmega_audit_domain_schema", data_dir / "domain_schema.py")
    finally:
        for name, previous in (
            (tokenizer_name, previous_tokenizer),
            (data_name, previous_data),
        ):
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        if previous_data_attr is missing:
            cppmega.__dict__.pop("data", None)
        else:
            cppmega.data = previous_data_attr

    return (
        domain.DomainKind,
        domain.DomainRoleKind,
        domain.DomainEdgeKind,
        domain.ParseConfidence,
        domain.DOMAIN_EDGE_FAMILIES,
        domain.DOMAIN_DELIMITER_ROLES,
        domain.DOMAIN_SCHEMA_SHA256,
        domain.TOKEN_DOMAIN_EDGE_COLUMN_FAMILIES,
        domain.validate_case5_contract_metadata,
        domain.validate_domain_edge_kind,
        tokenizer.DOMAIN_DELIMITER_TOKEN_IDS,
        tokenizer.DOMAIN_DELIMITER_CONTRACT_METADATA_KEY,
        tokenizer.DOMAIN_DELIMITER_CONTRACT_SHA256,
        tokenizer.TOKENIZER_CONTRACT_SHA256,
        source_identity.validate_source_identity_registry,
    )


(
    DomainKind,
    DomainRoleKind,
    DomainEdgeKind,
    ParseConfidence,
    DOMAIN_EDGE_FAMILIES,
    DOMAIN_DELIMITER_ROLES,
    DOMAIN_SCHEMA_SHA256,
    TOKEN_DOMAIN_EDGE_COLUMN_FAMILIES,
    validate_case5_contract_metadata,
    validate_domain_edge_kind,
    DOMAIN_DELIMITER_TOKEN_IDS,
    DOMAIN_DELIMITER_CONTRACT_METADATA_KEY,
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256,
    validate_source_identity_registry,
) = _load_schema_contracts()

_EDGE_FAMILY_BY_FIELD = {
    "token_domain_edges": "domain",
    "token_build_edges": "build",
    "token_shell_edges": "shell",
    "token_diagnostic_edges": "diagnostic",
    "token_cross_domain_edges": "cross_domain",
}
_EDGE_KIND_IDS_BY_FAMILY = {
    family: frozenset(int(kind) for kind in kinds)
    for family, kinds in DOMAIN_EDGE_FAMILIES.items()
}
_DOMAIN_DELIMITER_BY_ID: dict[int, tuple[int, str, int]] = {}
for _domain, (_start_role, _end_role) in DOMAIN_DELIMITER_ROLES.items():
    _start_id = int(DOMAIN_DELIMITER_TOKEN_IDS[_start_role])
    _end_id = int(DOMAIN_DELIMITER_TOKEN_IDS[_end_role])
    _DOMAIN_DELIMITER_BY_ID[_start_id] = (int(_domain), "start", _end_id)
    _DOMAIN_DELIMITER_BY_ID[_end_id] = (int(_domain), "end", _end_id)


TOKEN_COLUMNS = (
    "input_ids",
    "target_ids",
    "loss_mask",
    "doc_ids",
)

SOURCE_COLUMNS = (
    "source_doc_ids",
    "source_doc_token_lengths",
    "source_platform_ids",
    "source_repo_stable_ids",
    "source_filepath_stable_ids",
    "source_file_local_commit_indices",
)

TOKEN_ALIGNED_FIELDS = (
    "token_domain_ids",
    "token_role_ids",
    "token_entity_ids",
    "token_scope_ids",
    "token_source_doc_ids",
    "token_source_identity_ids",
    "token_confidence_ids",
    "token_platform_ids",
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
    "token_symbol_ids",
    "token_call_targets",
    "token_type_refs",
    "token_def_use",
    "token_change_mask_pre",
    "token_change_mask_post",
    "hunk_id_per_token",
    "edit_op_per_token",
)

CHUNK_ALIGNED_FIELDS = (
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
)

LIST_FIELDS = (
    "platform_ids",
    "token_call_edges",
    "token_type_edges",
    "token_domain_edges",
    "token_build_edges",
    "token_shell_edges",
    "token_diagnostic_edges",
    "token_cross_domain_edges",
    "changed_chunk_ids",
    "changed_chunk_spans",
    "source_identity_registry",
)

ALL_FIELDS = (
    *TOKEN_COLUMNS,
    *SOURCE_COLUMNS,
    *TOKEN_ALIGNED_FIELDS,
    *CHUNK_ALIGNED_FIELDS,
    *LIST_FIELDS,
)

CASE5_REQUIRED_COLUMNS = frozenset(
    {
        *TOKEN_COLUMNS,
        "valid_token_count",
        "trained_token_count",
        "num_docs",
        "source_doc_token_lengths",
        "token_domain_ids",
        "token_role_ids",
        "token_entity_ids",
        "token_scope_ids",
        "token_source_doc_ids",
        "token_source_identity_ids",
        "token_confidence_ids",
        "token_symbol_ids",
        "token_call_targets",
        "token_type_refs",
        "token_call_edges",
        "token_type_edges",
        "token_domain_edges",
        "token_build_edges",
        "token_shell_edges",
        "token_diagnostic_edges",
        "token_cross_domain_edges",
        "token_chunk_starts",
        "token_chunk_ends",
        "token_chunk_kinds",
        "token_chunk_dep_levels",
        "source_identity_registry",
    }
)

_EDGE_PAIR_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("from", pa.uint16()),
            pa.field("to", pa.uint16()),
        ]
    )
)
_EDGE_TRIPLE_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("from", pa.uint32()),
            pa.field("to", pa.uint32()),
            pa.field("kind", pa.int32()),
        ]
    )
)
_SOURCE_IDENTITY_REGISTRY_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("source_identity_id", pa.uint64(), nullable=False),
            pa.field("canonical_sha256", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
        ]
    )
)
CASE5_ARROW_TYPES = {
    "doc_ids": pa.list_(pa.uint32()),
    "token_source_doc_ids": pa.list_(pa.uint32()),
    "token_source_identity_ids": pa.list_(pa.uint64()),
    "token_symbol_ids": pa.list_(pa.uint64()),
    "token_call_targets": pa.list_(pa.uint64()),
    "token_type_refs": pa.list_(pa.uint64()),
    "token_domain_ids": pa.list_(pa.uint16()),
    "token_role_ids": pa.list_(pa.uint16()),
    "token_entity_ids": pa.list_(pa.uint32()),
    "token_scope_ids": pa.list_(pa.uint32()),
    "token_confidence_ids": pa.list_(pa.uint8()),
    "token_call_edges": _EDGE_PAIR_TYPE,
    "token_type_edges": _EDGE_PAIR_TYPE,
    "token_domain_edges": _EDGE_TRIPLE_TYPE,
    "token_build_edges": _EDGE_TRIPLE_TYPE,
    "token_shell_edges": _EDGE_TRIPLE_TYPE,
    "token_diagnostic_edges": _EDGE_TRIPLE_TYPE,
    "token_cross_domain_edges": _EDGE_TRIPLE_TYPE,
    "token_chunk_starts": pa.list_(pa.uint32()),
    "token_chunk_ends": pa.list_(pa.uint32()),
    "token_chunk_kinds": pa.list_(pa.uint8()),
    "token_chunk_dep_levels": pa.list_(pa.uint16()),
    "source_identity_registry": _SOURCE_IDENTITY_REGISTRY_TYPE,
}

_DIAGNOSTIC_DOMAIN_IDS = {int(domain) for domain in DomainKind if int(domain) >= 40}

_DIAGNOSTIC_DELIMITER_IDS = np.asarray(
    [
        DOMAIN_DELIMITER_TOKEN_IDS[name]
        for domain, roles in DOMAIN_DELIMITER_ROLES.items()
        if int(domain) >= 40
        for name in roles
    ],
    dtype=np.int64,
)

_DELIMITER_BY_TOKEN_ID: dict[int, tuple[int, bool]] = {}
for _domain, (_start_role, _end_role) in DOMAIN_DELIMITER_ROLES.items():
    _DELIMITER_BY_TOKEN_ID[DOMAIN_DELIMITER_TOKEN_IDS[_start_role]] = (
        int(_domain),
        True,
    )
    _DELIMITER_BY_TOKEN_ID[DOMAIN_DELIMITER_TOKEN_IDS[_end_role]] = (
        int(_domain),
        False,
    )


@dataclass
class FieldStats:
    rows_present: int = 0
    rows_nonempty: int = 0
    rows_nonzero: int = 0
    slots_total: int = 0
    slots_nonzero: int = 0
    bad_length_rows: int = 0
    bad_value_rows: int = 0
    missing_files: int = 0

    def add(self, other: "FieldStats") -> None:
        self.rows_present += other.rows_present
        self.rows_nonempty += other.rows_nonempty
        self.rows_nonzero += other.rows_nonzero
        self.slots_total += other.slots_total
        self.slots_nonzero += other.slots_nonzero
        self.bad_length_rows += other.bad_length_rows
        self.bad_value_rows += other.bad_value_rows
        self.missing_files += other.missing_files

    def as_dict(self, rows_total: int, files_total: int) -> dict[str, Any]:
        def pct(num: int, den: int) -> float:
            return round(100.0 * num / den, 6) if den else 0.0

        return {
            "rows_present": self.rows_present,
            "rows_present_pct": pct(self.rows_present, rows_total),
            "rows_nonempty": self.rows_nonempty,
            "rows_nonempty_pct": pct(self.rows_nonempty, rows_total),
            "rows_nonzero": self.rows_nonzero,
            "rows_nonzero_pct": pct(self.rows_nonzero, rows_total),
            "slots_total": self.slots_total,
            "slots_nonzero": self.slots_nonzero,
            "slots_nonzero_pct": pct(self.slots_nonzero, self.slots_total),
            "bad_length_rows": self.bad_length_rows,
            "bad_value_rows": self.bad_value_rows,
            "missing_files": self.missing_files,
            "missing_files_pct": pct(self.missing_files, files_total),
        }


@dataclass
class AuditStats:
    files: int = 0
    rows: int = 0
    capacity_tokens: int = 0
    valid_tokens: int = 0
    trained_tokens: int = 0
    slack_tokens: int = 0
    bad_rows: int = 0
    bad_files: int = 0
    edge_count: dict[str, int] = field(
        default_factory=lambda: {
            "token_call_edges": 0,
            "token_type_edges": 0,
            "token_domain_edges": 0,
            "token_build_edges": 0,
            "token_shell_edges": 0,
            "token_diagnostic_edges": 0,
            "token_cross_domain_edges": 0,
        }
    )
    field_stats: dict[str, FieldStats] = field(
        default_factory=lambda: {name: FieldStats() for name in ALL_FIELDS}
    )
    errors: list[str] = field(default_factory=list)

    def add(self, other: "AuditStats") -> None:
        self.files += other.files
        self.rows += other.rows
        self.capacity_tokens += other.capacity_tokens
        self.valid_tokens += other.valid_tokens
        self.trained_tokens += other.trained_tokens
        self.slack_tokens += other.slack_tokens
        self.bad_rows += other.bad_rows
        self.bad_files += other.bad_files
        for key, value in other.edge_count.items():
            self.edge_count[key] = self.edge_count.get(key, 0) + value
        for name, stats in other.field_stats.items():
            self.field_stats[name].add(stats)
        self.errors.extend(other.errors)

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "rows": self.rows,
            "capacity_tokens": self.capacity_tokens,
            "valid_tokens": self.valid_tokens,
            "trained_tokens": self.trained_tokens,
            "slack_tokens": self.slack_tokens,
            "pad_tokens": self.capacity_tokens - self.valid_tokens,
            "pad_pct": round(
                100.0
                * (self.capacity_tokens - self.valid_tokens)
                / self.capacity_tokens,
                6,
            )
            if self.capacity_tokens
            else 0.0,
            "bad_rows": self.bad_rows,
            "bad_files": self.bad_files,
            "edge_count": self.edge_count,
            "fields": {
                name: stats.as_dict(self.rows, self.files)
                for name, stats in sorted(self.field_stats.items())
            },
            "errors": self.errors[:2000],
        }


def _flatten_edge(edge: Any) -> tuple[int | None, int | None]:
    if isinstance(edge, dict):
        return _as_int(edge.get("from")), _as_int(edge.get("to"))
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        return _as_int(edge[0]), _as_int(edge[1])
    return None, None


def _flatten_edge_triple(edge: Any) -> tuple[int | None, int | None, int | None]:
    if isinstance(edge, dict):
        return (
            _as_int(edge.get("from", edge.get("src"))),
            _as_int(edge.get("to", edge.get("dst"))),
            _as_int(edge.get("kind")),
        )
    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
        return _as_int(edge[0]), _as_int(edge[1]), _as_int(edge[2])
    return None, None, None


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_positive_valid_token_source_ids(
    *,
    stats: AuditStats,
    column: Any,
    valid: np.ndarray,
    row_bad: np.ndarray,
    path: str | Path,
    field: str = "token_source_doc_ids",
    max_value: int = (1 << 32) - 1,
) -> None:
    bad_rows = 0
    for row_idx, values in enumerate(column.to_pylist()):
        limit = int(valid[row_idx])
        valid_values = list(values or [])[:limit]
        if len(valid_values) != limit or any(
            (value := _as_int(raw)) is None or value <= 0 or value > max_value
            for raw in valid_values
        ):
            row_bad[row_idx] = True
            bad_rows += 1
    if bad_rows:
        stats.field_stats[field].bad_value_rows += bad_rows
        stats.errors.append(
            f"{path}: {bad_rows} rows have zero/invalid {field} on valid tokens"
        )


def _validate_source_identity_foreign_keys(
    *,
    stats: AuditStats,
    identity_column: Any,
    registry_column: Any,
    valid: np.ndarray,
    row_bad: np.ndarray,
    path: str,
) -> None:
    bad_rows = 0
    for row_idx, (identity_values, registry_entries) in enumerate(
        zip(
            identity_column.to_pylist(),
            registry_column.to_pylist(),
            strict=True,
        )
    ):
        try:
            validate_source_identity_registry(
                registry_entries or [],
                referenced_ids=(identity_values or [])[: int(valid[row_idx])],
            )
        except (TypeError, ValueError):
            row_bad[row_idx] = True
            bad_rows += 1
    if bad_rows:
        stats.field_stats["source_identity_registry"].bad_value_rows += bad_rows
        stats.errors.append(
            f"{path}: {bad_rows} rows have invalid source identity registry/FKs"
        )


def _nonzero_count(values: Any, *, hunk: bool = False) -> int:
    if values is None:
        return 0
    out = 0
    for value in values:
        iv = _as_int(value)
        if iv is None:
            continue
        if hunk:
            out += int(iv >= 0)
        else:
            out += int(iv != 0)
    return out


def _len_or_none(values: Any) -> int | None:
    if values is None:
        return None
    try:
        return len(values)
    except TypeError:
        return None


def _list_lengths(column: Any) -> np.ndarray:
    lengths = pc.fill_null(pc.list_value_length(column), 0).combine_chunks()
    return np.asarray(lengths.to_numpy(zero_copy_only=False), dtype=np.int64)


def _column_numpy(column: Any) -> np.ndarray:
    arr = column.combine_chunks()
    return np.asarray(arr.to_numpy(zero_copy_only=False))


def _flat_numpy(column: Any) -> np.ndarray:
    flat = pc.list_flatten(column).combine_chunks()
    if len(flat) == 0:
        return np.asarray([], dtype=np.int64)
    return np.asarray(flat.to_numpy(zero_copy_only=False))


def _rows_with_flat_mask(lengths: np.ndarray, flat_mask: np.ndarray) -> np.ndarray:
    rows = np.zeros(len(lengths), dtype=bool)
    if len(lengths) == 0 or len(flat_mask) == 0:
        return rows
    nonempty = lengths > 0
    if not np.any(nonempty):
        return rows
    starts = np.empty(len(lengths), dtype=np.int64)
    starts[0] = 0
    if len(lengths) > 1:
        np.cumsum(lengths[:-1], out=starts[1:])
    reduced = np.add.reduceat(flat_mask.astype(np.int8), starts[nonempty])
    rows[nonempty] = reduced > 0
    return rows


def _add_numeric_list_stats(
    *,
    stats: AuditStats,
    field: str,
    column: Any,
    expected_lengths: np.ndarray | int | None,
    hunk: bool = False,
    row_bad: np.ndarray | None = None,
) -> np.ndarray:
    fs = stats.field_stats[field]
    lengths = _list_lengths(column)
    fs.rows_present += len(lengths)
    nonempty = lengths > 0
    fs.rows_nonempty += int(np.count_nonzero(nonempty))
    fs.slots_total += int(lengths.sum())

    flat = _flat_numpy(column)
    if len(flat):
        if hunk:
            nz_mask = flat >= 0
        else:
            nz_mask = flat != 0
        fs.slots_nonzero += int(np.count_nonzero(nz_mask))
        fs.rows_nonzero += int(np.count_nonzero(_rows_with_flat_mask(lengths, nz_mask)))

    if expected_lengths is not None:
        if isinstance(expected_lengths, int):
            bad_len = lengths != expected_lengths
        else:
            bad_len = lengths != expected_lengths
        bad_len_count = int(np.count_nonzero(bad_len))
        fs.bad_length_rows += bad_len_count
        if row_bad is not None and bad_len_count:
            row_bad |= bad_len

    return lengths


def _add_generic_list_stats(stats: AuditStats, field: str, column: Any) -> np.ndarray:
    fs = stats.field_stats[field]
    lengths = _list_lengths(column)
    fs.rows_present += len(lengths)
    nonempty = lengths > 0
    count = int(np.count_nonzero(nonempty))
    fs.rows_nonempty += count
    fs.rows_nonzero += count
    slots = int(lengths.sum())
    fs.slots_total += slots
    fs.slots_nonzero += slots
    return lengths


def _count_bad_flat_values(
    *,
    stats: AuditStats,
    field: str,
    column: Any,
    lengths: np.ndarray,
    bad_mask: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    rows = _rows_with_flat_mask(lengths, bad_mask)
    count = int(np.count_nonzero(rows))
    if count:
        stats.field_stats[field].bad_value_rows += count
        row_bad |= rows


def _validate_packed_document_boundaries(
    *,
    stats: AuditStats,
    loss_mask_col: Any,
    doc_ids_col: Any,
    source_doc_token_lengths_col: Any,
    input_lengths: np.ndarray,
    valid: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    """Validate boundaries against independent logical-document lengths.

    ``doc_ids`` cannot be its own oracle: an old producer assigned the same ID
    to distinct functions sharing file provenance, and a mask derived from those
    IDs looked internally consistent while still training across documents.
    ``source_doc_token_lengths`` is the independent packed-document contract.
    """
    doc_rows = doc_ids_col.to_pylist()
    mask_rows = loss_mask_col.to_pylist()
    source_length_rows = source_doc_token_lengths_col.to_pylist()
    bad_doc_ids = 0
    bad_masks = 0

    for row_index, (capacity, valid_count) in enumerate(zip(input_lengths, valid)):
        capacity = int(capacity)
        valid_count = int(valid_count)
        raw_lengths = source_length_rows[row_index]
        if raw_lengths is None:
            row_bad[row_index] = True
            stats.errors.append(f"row {row_index}: missing source_doc_token_lengths")
            continue
        logical_lengths = [int(value) for value in raw_lengths]
        if (
            (valid_count > 0 and not logical_lengths)
            or any(length <= 0 for length in logical_lengths)
            or sum(logical_lengths) != valid_count
        ):
            row_bad[row_index] = True
            stats.field_stats["source_doc_token_lengths"].bad_value_rows += 1
            stats.errors.append(
                f"row {row_index}: source_doc_token_lengths={logical_lengths!r} "
                f"do not partition valid_token_count={valid_count}"
            )
            continue

        expected_doc_ids: list[int] = []
        expected_loss_mask: list[int] = []
        for local_doc_id, length in enumerate(logical_lengths, start=1):
            expected_doc_ids.extend([local_doc_id] * length)
            expected_loss_mask.extend([1] * (length - 1))
            expected_loss_mask.append(0)
        pad_doc_id = len(logical_lengths) if logical_lengths else 0
        expected_doc_ids.extend([pad_doc_id] * (capacity - valid_count))
        expected_loss_mask.extend([0] * (capacity - valid_count))

        actual_doc_ids = [int(value) for value in (doc_rows[row_index] or [])]
        actual_loss_mask = [int(value) for value in (mask_rows[row_index] or [])]
        if actual_doc_ids != expected_doc_ids:
            row_bad[row_index] = True
            bad_doc_ids += 1
        if actual_loss_mask != expected_loss_mask:
            row_bad[row_index] = True
            bad_masks += 1

    if bad_doc_ids:
        stats.field_stats["doc_ids"].bad_value_rows += bad_doc_ids
    if bad_masks:
        stats.field_stats["loss_mask"].bad_value_rows += bad_masks


def _validate_target_ids_shift(
    *,
    stats: AuditStats,
    input_ids_col: Any,
    target_ids_col: Any,
    input_lengths: np.ndarray,
    row_bad: np.ndarray,
    pad_token_id: int = 0,
) -> None:
    """Check `target_ids == input_ids[1:] + PAD` for each fixed-width row."""
    rows = len(input_lengths)
    if rows == 0:
        return
    lengths = input_lengths.astype(np.int64)
    n = int(lengths.sum())
    input_flat = _flat_numpy(input_ids_col)
    target_flat = _flat_numpy(target_ids_col)
    if len(input_flat) != n or len(target_flat) != n:
        return

    expected = np.full(n, pad_token_id, dtype=np.asarray(target_flat).dtype)
    nonempty = lengths > 0
    if np.any(nonempty):
        starts = np.zeros(rows, dtype=np.int64)
        if rows > 1:
            np.cumsum(lengths[:-1], out=starts[1:])
        ends = starts + lengths
        for start, end in zip(starts[nonempty], ends[nonempty]):
            if end - start > 1:
                expected[start : end - 1] = input_flat[start + 1 : end]
    mismatch = target_flat != expected
    bad_rows = _rows_with_flat_mask(lengths, mismatch)
    count = int(np.count_nonzero(bad_rows))
    if count:
        stats.field_stats["target_ids"].bad_value_rows += count
        row_bad |= bad_rows


def _validate_trained_count_against_loss_mask(
    *,
    stats: AuditStats,
    trained: np.ndarray,
    loss_mask_col: Any,
    input_lengths: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    """Check `trained_token_count` is exactly the positive loss-mask sum."""
    rows = len(input_lengths)
    if rows == 0:
        return
    lengths = input_lengths.astype(np.int64)
    lm_flat = _flat_numpy(loss_mask_col)
    if len(lm_flat) != int(lengths.sum()) or len(trained) != rows:
        return

    expected = np.zeros(rows, dtype=np.int64)
    nonempty = lengths > 0
    if np.any(nonempty):
        starts = np.zeros(rows, dtype=np.int64)
        if rows > 1:
            np.cumsum(lengths[:-1], out=starts[1:])
        reduced = np.add.reduceat((lm_flat > 0).astype(np.int64), starts[nonempty])
        expected[nonempty] = reduced

    mismatch = trained.astype(np.int64) != expected
    count = int(np.count_nonzero(mismatch))
    if count:
        row_bad |= mismatch
        stats.errors.append(f"{count} rows have trained_token_count != sum(loss_mask)")


def _iter_or_empty(values: Any) -> Any:
    if values is None:
        return ()
    return values


def _edge_stats_and_validate(
    *,
    stats: AuditStats,
    field: str,
    rows: list[Any],
    n_chunks: np.ndarray,
    chunk_starts: list[Any],
    chunk_ends: list[Any],
    logical_doc_rows: list[Any],
    valid_lengths: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    fs = stats.field_stats[field]
    fs.rows_present += len(rows)
    for idx, edges in enumerate(rows):
        edges = _iter_or_empty(edges)
        if len(edges):
            fs.rows_nonempty += 1
            fs.rows_nonzero += 1
        fs.slots_total += len(edges)
        fs.slots_nonzero += len(edges)
        stats.edge_count[field] = stats.edge_count.get(field, 0) + len(edges)
        limit = int(n_chunks[idx]) if idx < len(n_chunks) else 0
        valid_limit = int(valid_lengths[idx]) if idx < len(valid_lengths) else 0
        for edge in edges:
            src, dst = _flatten_edge(edge)
            if (
                src is None
                or dst is None
                or src < 0
                or dst < 0
                or src >= limit
                or dst >= limit
            ):
                row_bad[idx] = True
                fs.bad_value_rows += 1
                break
            starts = list(_iter_or_empty(chunk_starts[idx]))
            ends = list(_iter_or_empty(chunk_ends[idx]))
            logical_ids = list(_iter_or_empty(logical_doc_rows[idx]))
            endpoint_docs: list[int] = []
            for chunk_index in (src, dst):
                start = (
                    _as_int(starts[chunk_index]) if chunk_index < len(starts) else None
                )
                end = _as_int(ends[chunk_index]) if chunk_index < len(ends) else None
                if (
                    start is None
                    or end is None
                    or start < 0
                    or end <= start
                    or end > valid_limit
                    or end > len(logical_ids)
                ):
                    endpoint_docs = []
                    break
                docs = {
                    int(value)
                    for value in logical_ids[start:end]
                    if _as_int(value) is not None
                }
                if len(docs) != 1 or next(iter(docs)) <= 0:
                    endpoint_docs = []
                    break
                endpoint_docs.append(next(iter(docs)))
            if len(endpoint_docs) != 2 or endpoint_docs[0] != endpoint_docs[1]:
                row_bad[idx] = True
                fs.bad_value_rows += 1
                break


def _edge_triple_stats_and_validate(
    *,
    stats: AuditStats,
    field: str,
    rows: list[Any],
    valid_lengths: np.ndarray,
    logical_doc_rows: list[Any],
    row_bad: np.ndarray,
) -> None:
    fs = stats.field_stats[field]
    fs.rows_present += len(rows)
    for idx, edges in enumerate(rows):
        edges = _iter_or_empty(edges)
        if len(edges):
            fs.rows_nonempty += 1
            fs.rows_nonzero += 1
        fs.slots_total += len(edges)
        fs.slots_nonzero += len(edges)
        stats.edge_count[field] = stats.edge_count.get(field, 0) + len(edges)
        limit = int(valid_lengths[idx]) if idx < len(valid_lengths) else 0
        logical_ids = list(_iter_or_empty(logical_doc_rows[idx]))
        for edge in edges:
            src, dst, kind = _flatten_edge_triple(edge)
            valid_kind = True
            if kind is not None:
                try:
                    validate_domain_edge_kind(
                        kind,
                        family=_EDGE_FAMILY_BY_FIELD[field],
                    )
                except (TypeError, ValueError):
                    valid_kind = False
            if (
                src is None
                or dst is None
                or kind is None
                or src < 0
                or dst < 0
                or not valid_kind
                or src >= limit
                or dst >= limit
                or src >= len(logical_ids)
                or dst >= len(logical_ids)
                or _as_int(logical_ids[src]) is None
                or _as_int(logical_ids[dst]) is None
                or int(logical_ids[src]) <= 0
                or int(logical_ids[dst]) <= 0
                or int(logical_ids[src]) != int(logical_ids[dst])
            ):
                row_bad[idx] = True
                fs.bad_value_rows += 1
                break


def _validate_domain_delimiter_sidecars(
    *,
    stats: AuditStats,
    table: Any,
    names: set[str],
    input_ids_col: Any,
    valid_lengths: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    token_rows = input_ids_col.to_pylist()
    rows_with_delimiter = np.asarray(
        [
            any(
                int(token_id) in _DOMAIN_DELIMITER_BY_ID
                for token_id in list(_iter_or_empty(tokens))[
                    : int(valid_lengths[index])
                ]
            )
            for index, tokens in enumerate(token_rows)
        ],
        dtype=bool,
    )
    required = ("token_domain_ids", "token_role_ids", "token_confidence_ids")
    if np.any(rows_with_delimiter) and any(name not in names for name in required):
        row_bad |= rows_with_delimiter
        missing = [name for name in required if name not in names]
        stats.errors.append(
            f"{int(np.count_nonzero(rows_with_delimiter))} rows contain domain "
            f"delimiter tokens but missing sidecars: {missing}"
        )
        return
    if "token_role_ids" not in names:
        return
    if any(name not in names for name in required):
        role_rows = table.column("token_role_ids").to_pylist()
        for row_index, roles in enumerate(role_rows):
            valid = int(valid_lengths[row_index])
            if any(
                int(role) == int(DomainRoleKind.DELIMITER)
                for role in list(_iter_or_empty(roles))[:valid]
            ):
                row_bad[row_index] = True
                stats.field_stats["token_role_ids"].bad_value_rows += 1
        return
    domain_rows = table.column("token_domain_ids").to_pylist()
    role_rows = table.column("token_role_ids").to_pylist()
    confidence_rows = table.column("token_confidence_ids").to_pylist()
    malformed_rows = 0
    for row_index, (tokens, domains, roles, confidence) in enumerate(
        zip(token_rows, domain_rows, role_rows, confidence_rows, strict=True)
    ):
        valid = int(valid_lengths[row_index])
        tokens = list(_iter_or_empty(tokens))[:valid]
        domains = list(_iter_or_empty(domains))[:valid]
        roles = list(_iter_or_empty(roles))[:valid]
        confidence = list(_iter_or_empty(confidence))[:valid]
        if not (len(tokens) == len(domains) == len(roles) == len(confidence) == valid):
            continue
        stack: list[tuple[int, int]] = []
        bad_fields: set[str] = set()
        for token_id, domain_id, role_id, confidence_id in zip(
            tokens, domains, roles, confidence, strict=True
        ):
            marker = _DOMAIN_DELIMITER_BY_ID.get(int(token_id))
            if marker is None:
                expected_domain = stack[-1][0] if stack else int(DomainKind.UNKNOWN)
                if int(role_id) == int(DomainRoleKind.DELIMITER):
                    bad_fields.add("token_role_ids")
                if int(domain_id) != expected_domain:
                    bad_fields.add("token_domain_ids")
                continue
            expected_domain, direction, expected_close = marker
            if int(domain_id) != expected_domain:
                bad_fields.add("token_domain_ids")
            if int(role_id) != int(DomainRoleKind.DELIMITER):
                bad_fields.add("token_role_ids")
            if int(confidence_id) != int(ParseConfidence.EXACT):
                bad_fields.add("token_confidence_ids")
            if direction == "start":
                stack.append((expected_domain, expected_close))
            elif not stack or stack[-1] != (expected_domain, int(token_id)):
                bad_fields.add("token_role_ids")
            else:
                stack.pop()
        if stack:
            bad_fields.add("token_role_ids")
        if bad_fields:
            row_bad[row_index] = True
            malformed_rows += 1
            for field in bad_fields:
                stats.field_stats[field].bad_value_rows += 1
    if malformed_rows:
        stats.errors.append(
            f"{malformed_rows} rows have malformed/mismatched domain delimiter pairs"
        )


def _validate_token_source_doc_ids(
    *,
    stats: AuditStats,
    table: Any,
    names: set[str],
    input_lengths: np.ndarray,
    valid_lengths: np.ndarray,
    row_bad: np.ndarray,
) -> list[Any]:
    field = "token_source_doc_ids"
    nonempty_rows = valid_lengths > 0
    if field not in names:
        row_bad |= nonempty_rows
        stats.errors.append(
            f"{int(np.count_nonzero(nonempty_rows))} rows are missing required {field}"
        )
        return [[] for _ in range(len(valid_lengths))]

    column = table.column(field)
    value_type = getattr(column.type, "value_type", None)
    if value_type != pa.uint32():
        row_bad |= nonempty_rows
        stats.field_stats[field].bad_value_rows += int(np.count_nonzero(nonempty_rows))
        stats.errors.append(f"{field} must be list<uint32>, got {column.type}")

    rows = column.to_pylist()
    for row_index, values in enumerate(rows):
        capacity = int(input_lengths[row_index])
        valid = int(valid_lengths[row_index])
        values = list(_iter_or_empty(values))
        bad = len(values) != capacity or any(
            _as_int(value) is None or int(value) <= 0 or int(value) > (1 << 32) - 1
            for value in values[:valid]
        )
        if bad:
            row_bad[row_index] = True
            stats.field_stats[field].bad_value_rows += 1
    return rows


def _validate_diagnostic_rows_have_edges_or_raw(
    *,
    stats: AuditStats,
    table: Any,
    names: set[str],
    input_ids_col: Any,
    input_lengths: np.ndarray,
    valid_lengths: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    """Fail logical diagnostic documents without parsed edges unless marked RAW.

    Diagnostics are world observations, not comments. A structured diagnostic
    document should carry token_diagnostic_edges. If a parser could not recover
    edges, that logical document must explicitly mark its diagnostic tokens RAW;
    an edge or RAW token from another document in the packed row cannot certify it.
    """

    flat_input = _flat_numpy(input_ids_col)
    if len(flat_input) == 0:
        return

    diagnostic_flat = np.isin(flat_input.astype(np.int64), _DIAGNOSTIC_DELIMITER_IDS)
    if "token_domain_ids" in names:
        domain_flat = _flat_numpy(table.column("token_domain_ids"))
        if len(domain_flat) == len(flat_input):
            diagnostic_flat |= np.isin(
                domain_flat.astype(np.int64), list(_DIAGNOSTIC_DOMAIN_IDS)
            )

    if "doc_ids" not in names:
        return
    logical_doc_flat = _flat_numpy(table.column("doc_ids"))
    if len(logical_doc_flat) != len(flat_input):
        return
    if "token_confidence_ids" in names:
        conf_flat = _flat_numpy(table.column("token_confidence_ids"))
    else:
        conf_flat = np.asarray([], dtype=np.int64)
    edge_rows = (
        table.column("token_diagnostic_edges").to_pylist()
        if "token_diagnostic_edges" in names
        else [[] for _ in range(len(input_lengths))]
    )

    bad_rows = 0
    bad_docs = 0
    offset = 0
    for row_index, (capacity, valid, edges) in enumerate(
        zip(input_lengths, valid_lengths, edge_rows, strict=True)
    ):
        capacity = int(capacity)
        valid = int(valid)
        row_slice = slice(offset, offset + valid)
        row_diagnostic = diagnostic_flat[row_slice]
        row_doc_ids = logical_doc_flat[row_slice]
        diagnostic_docs = {
            int(row_doc_ids[position])
            for position in np.flatnonzero(row_diagnostic)
            if int(row_doc_ids[position]) > 0
        }
        if not diagnostic_docs:
            offset += capacity
            continue

        raw_docs: set[int] = set()
        if len(conf_flat) == len(flat_input):
            row_confidence = conf_flat[row_slice]
            raw_docs = {
                int(row_doc_ids[position])
                for position in np.flatnonzero(
                    row_diagnostic
                    & (row_confidence.astype(np.int64) == int(ParseConfidence.RAW))
                )
                if int(row_doc_ids[position]) > 0
            }

        edge_docs: set[int] = set()
        for edge in _iter_or_empty(edges):
            src, dst, _kind = _flatten_edge_triple(edge)
            if (
                src is not None
                and dst is not None
                and 0 <= src < valid
                and 0 <= dst < valid
                and int(row_doc_ids[src]) > 0
                and int(row_doc_ids[src]) == int(row_doc_ids[dst])
            ):
                edge_docs.add(int(row_doc_ids[src]))

        missing = diagnostic_docs - raw_docs - edge_docs
        if missing:
            row_bad[row_index] = True
            bad_rows += 1
            bad_docs += len(missing)
        offset += capacity

    if bad_rows:
        stats.field_stats["token_diagnostic_edges"].bad_value_rows += bad_rows
        stats.errors.append(
            f"{bad_docs} diagnostic/error logical docs across {bad_rows} rows have "
            "no token_diagnostic_edges and no explicit ParseConfidence.RAW"
        )


def _audit_table(
    *,
    path: Path,
    table: Any,
    names: set[str],
    expected_len: int | None,
    vocab_size: int | None,
) -> AuditStats:
    """Audit one bounded parquet row group and return additive statistics."""

    stats = AuditStats()
    stats.rows = table.num_rows
    row_bad = np.zeros(stats.rows, dtype=bool)

    try:
        if "input_ids" not in names:
            stats.bad_rows = stats.rows
            stats.errors.append(f"{path}: missing input_ids")
            return stats

        input_ids = table.column("input_ids")
        input_lengths = _add_numeric_list_stats(
            stats=stats,
            field="input_ids",
            column=input_ids,
            expected_lengths=expected_len,
            row_bad=row_bad,
        )
        stats.capacity_tokens += int(input_lengths.sum())

        valid = (
            _column_numpy(table.column("valid_token_count")).astype(np.int64)
            if "valid_token_count" in names
            else input_lengths
        )
        trained = (
            _column_numpy(table.column("trained_token_count")).astype(np.int64)
            if "trained_token_count" in names
            else valid
        )
        slack = (
            _column_numpy(table.column("slack_tokens")).astype(np.int64)
            if "slack_tokens" in names
            else np.maximum(0, input_lengths - valid)
        )
        stats.valid_tokens += int(valid.sum())
        stats.trained_tokens += int(trained.sum())
        stats.slack_tokens += int(slack.sum())
        bad_counts = (
            (valid < 0)
            | (valid > input_lengths)
            | (trained < 0)
            | (trained > input_lengths)
        )
        if np.any(bad_counts):
            row_bad |= bad_counts
            stats.errors.append(
                f"{path}: {int(np.count_nonzero(bad_counts))} rows have invalid valid/trained counts"
            )

        if vocab_size is not None:
            flat_input = _flat_numpy(input_ids)
            if len(flat_input):
                bad_vocab = (flat_input < 0) | (flat_input >= vocab_size)
                _count_bad_flat_values(
                    stats=stats,
                    field="input_ids",
                    column=input_ids,
                    lengths=input_lengths,
                    bad_mask=bad_vocab,
                    row_bad=row_bad,
                )

        _validate_domain_delimiter_sidecars(
            stats=stats,
            table=table,
            names=names,
            input_ids_col=input_ids,
            valid_lengths=valid,
            row_bad=row_bad,
        )
        _validate_diagnostic_rows_have_edges_or_raw(
            stats=stats,
            table=table,
            names=names,
            input_ids_col=input_ids,
            input_lengths=input_lengths,
            valid_lengths=valid,
            row_bad=row_bad,
        )

        for name in ("target_ids", "loss_mask", "doc_ids"):
            if name in names:
                _add_numeric_list_stats(
                    stats=stats,
                    field=name,
                    column=table.column(name),
                    expected_lengths=input_lengths,
                    row_bad=row_bad,
                )

        # Derive both doc_ids and loss_mask from logical source-document lengths.
        # Never trust doc_ids as the oracle for its own boundary correctness.
        if all(
            name in names
            for name in ("loss_mask", "doc_ids", "source_doc_token_lengths")
        ):
            _validate_packed_document_boundaries(
                stats=stats,
                loss_mask_col=table.column("loss_mask"),
                doc_ids_col=table.column("doc_ids"),
                source_doc_token_lengths_col=table.column("source_doc_token_lengths"),
                input_lengths=input_lengths,
                valid=valid,
                row_bad=row_bad,
            )

        if "target_ids" in names:
            _validate_target_ids_shift(
                stats=stats,
                input_ids_col=input_ids,
                target_ids_col=table.column("target_ids"),
                input_lengths=input_lengths,
                row_bad=row_bad,
            )

        if "trained_token_count" in names and "loss_mask" in names:
            _validate_trained_count_against_loss_mask(
                stats=stats,
                trained=trained,
                loss_mask_col=table.column("loss_mask"),
                input_lengths=input_lengths,
                row_bad=row_bad,
            )

        for name in TOKEN_ALIGNED_FIELDS:
            if name in names:
                _add_numeric_list_stats(
                    stats=stats,
                    field=name,
                    column=table.column(name),
                    expected_lengths=input_lengths,
                    hunk=(name == "hunk_id_per_token"),
                    row_bad=row_bad,
                )

        _validate_token_source_doc_ids(
            stats=stats,
            table=table,
            names=names,
            input_lengths=input_lengths,
            valid_lengths=valid,
            row_bad=row_bad,
        )
        logical_doc_rows = (
            table.column("doc_ids").to_pylist()
            if "doc_ids" in names
            else [[] for _ in range(stats.rows)]
        )
        if "token_source_identity_ids" in names:
            _validate_positive_valid_token_source_ids(
                stats=stats,
                column=table.column("token_source_identity_ids"),
                valid=valid,
                row_bad=row_bad,
                path=path,
                field="token_source_identity_ids",
                max_value=(1 << 64) - 1,
            )
        if "source_identity_registry" in names:
            _add_generic_list_stats(
                stats,
                "source_identity_registry",
                table.column("source_identity_registry"),
            )
        if {
            "token_source_identity_ids",
            "source_identity_registry",
        } <= names:
            _validate_source_identity_foreign_keys(
                stats=stats,
                identity_column=table.column("token_source_identity_ids"),
                registry_column=table.column("source_identity_registry"),
                valid=valid,
                row_bad=row_bad,
                path=str(path),
            )

        for name in SOURCE_COLUMNS:
            if name in names:
                _add_generic_list_stats(stats, name, table.column(name))

        n_chunks = (
            _add_numeric_list_stats(
                stats=stats,
                field="token_chunk_starts",
                column=table.column("token_chunk_starts"),
                expected_lengths=None,
            )
            if "token_chunk_starts" in names
            else np.zeros(stats.rows, dtype=np.int64)
        )
        for name in ("token_chunk_ends", "token_chunk_kinds", "token_chunk_dep_levels"):
            if name in names:
                _add_numeric_list_stats(
                    stats=stats,
                    field=name,
                    column=table.column(name),
                    expected_lengths=n_chunks,
                    row_bad=row_bad,
                )

        if "token_chunk_starts" in names:
            starts = _flat_numpy(table.column("token_chunk_starts"))
            if len(starts):
                bad_starts = (starts < 0) | (starts >= np.repeat(valid, n_chunks))
                _count_bad_flat_values(
                    stats=stats,
                    field="token_chunk_starts",
                    column=table.column("token_chunk_starts"),
                    lengths=n_chunks,
                    bad_mask=bad_starts,
                    row_bad=row_bad,
                )
        if "token_chunk_ends" in names:
            ends = _flat_numpy(table.column("token_chunk_ends"))
            if len(ends):
                bad_ends = (ends <= 0) | (ends > np.repeat(valid, n_chunks))
                _count_bad_flat_values(
                    stats=stats,
                    field="token_chunk_ends",
                    column=table.column("token_chunk_ends"),
                    lengths=n_chunks,
                    bad_mask=bad_ends,
                    row_bad=row_bad,
                )
        if "token_chunk_starts" in names and "token_chunk_ends" in names:
            start_rows = table.column("token_chunk_starts").to_pylist()
            end_rows = table.column("token_chunk_ends").to_pylist()
            for row_idx, (row_starts, row_ends) in enumerate(
                zip(start_rows, end_rows, strict=True)
            ):
                if len(row_starts or []) != len(row_ends or []):
                    continue
                if any(
                    int(start) >= int(end)
                    for start, end in zip(row_starts or [], row_ends or [], strict=True)
                ):
                    row_bad[row_idx] = True
                    stats.field_stats["token_chunk_ends"].bad_value_rows += 1

        if "platform_ids" in names:
            _add_numeric_list_stats(
                stats=stats,
                field="platform_ids",
                column=table.column("platform_ids"),
                expected_lengths=None,
            )

        if "changed_chunk_ids" in names:
            _add_numeric_list_stats(
                stats=stats,
                field="changed_chunk_ids",
                column=table.column("changed_chunk_ids"),
                expected_lengths=None,
            )
            for row_idx, values in enumerate(
                table.column("changed_chunk_ids").to_pylist()
            ):
                limit = int(n_chunks[row_idx]) if row_idx < len(n_chunks) else 0
                for value in _iter_or_empty(values):
                    idx = _as_int(value)
                    if idx is None or idx < 0 or idx >= limit:
                        row_bad[row_idx] = True
                        stats.field_stats["changed_chunk_ids"].bad_value_rows += 1
                        break

        if "changed_chunk_spans" in names:
            span_rows = table.column("changed_chunk_spans").to_pylist()
            _add_generic_list_stats(
                stats, "changed_chunk_spans", table.column("changed_chunk_spans")
            )
            for row_idx, spans in enumerate(span_rows):
                for span in _iter_or_empty(spans):
                    if isinstance(span, dict):
                        start, end = (
                            _as_int(span.get("start")),
                            _as_int(span.get("end")),
                        )
                    elif isinstance(span, (list, tuple)) and len(span) >= 2:
                        start, end = _as_int(span[0]), _as_int(span[1])
                    else:
                        start = end = None
                    if (
                        start is None
                        or end is None
                        or start < 0
                        or end <= start
                        or end > valid[row_idx]
                    ):
                        row_bad[row_idx] = True
                        stats.field_stats["changed_chunk_spans"].bad_value_rows += 1
                        break

        for name in ("token_call_edges", "token_type_edges"):
            if name in names:
                _edge_stats_and_validate(
                    stats=stats,
                    field=name,
                    rows=table.column(name).to_pylist(),
                    n_chunks=n_chunks,
                    chunk_starts=(
                        table.column("token_chunk_starts").to_pylist()
                        if "token_chunk_starts" in names
                        else [[] for _ in range(stats.rows)]
                    ),
                    chunk_ends=(
                        table.column("token_chunk_ends").to_pylist()
                        if "token_chunk_ends" in names
                        else [[] for _ in range(stats.rows)]
                    ),
                    logical_doc_rows=logical_doc_rows,
                    valid_lengths=valid,
                    row_bad=row_bad,
                )

        for name in (
            "token_domain_edges",
            "token_build_edges",
            "token_shell_edges",
            "token_diagnostic_edges",
            "token_cross_domain_edges",
        ):
            if name in names:
                _edge_triple_stats_and_validate(
                    stats=stats,
                    field=name,
                    rows=table.column(name).to_pylist(),
                    valid_lengths=valid,
                    logical_doc_rows=logical_doc_rows,
                    row_bad=row_bad,
                )

    except Exception as exc:
        # Fail loud: a crash mid-audit cannot certify a shard. Re-raise with
        # where+what instead of recording partial stats and returning "ok-ish".
        raise RuntimeError(
            f"{path}: audit failed: {type(exc).__name__}: {exc}"
        ) from exc

    stats.bad_rows = int(np.count_nonzero(row_bad))
    return stats


def _case5_schema_errors(path: Path, schema: pa.Schema) -> list[str]:
    errors: list[str] = []
    names = set(schema.names)
    missing = sorted(CASE5_REQUIRED_COLUMNS - names)
    if missing:
        errors.append(f"{path}: missing required CASE5 columns: {missing}")
    for name, expected_type in CASE5_ARROW_TYPES.items():
        if name in names and schema.field(name).type != expected_type:
            errors.append(
                f"{path}: {name} type {schema.field(name).type} != {expected_type}"
            )
    metadata = schema.metadata or {}
    try:
        validate_case5_contract_metadata(metadata, where=path)
    except ValueError as exc:
        errors.append(str(exc))
    if metadata.get(b"cppmega.case5_schema") != b"case5_domain_routes_v1":
        errors.append(f"{path}: missing cppmega.case5_schema receipt metadata")
    return errors


def _audit_file(
    path_str: str, kind: str, bucket: str, vocab_size: int | None
) -> dict[str, Any]:
    path = Path(path_str)
    stats = AuditStats(files=1)
    try:
        pf = pq.ParquetFile(path)
        top_level_names = set(pf.schema_arrow.names)
        cols = [name for name in ALL_FIELDS if name in top_level_names]
        cols += [
            name
            for name in (
                "valid_token_count",
                "trained_token_count",
                "slack_tokens",
                "num_docs",
            )
            if name in top_level_names
        ]
    except Exception as exc:
        raise RuntimeError(
            f"{path}: parquet open failed: {type(exc).__name__}: {exc}"
        ) from exc

    names = set(cols)
    schema_errors = _case5_schema_errors(path, pf.schema_arrow)
    if schema_errors:
        stats.bad_files = 1
        stats.errors.extend(schema_errors)
    for name in ALL_FIELDS:
        if name not in names:
            stats.field_stats[name].missing_files += 1

    if "input_ids" not in names:
        stats.rows = int(pf.metadata.num_rows)
        stats.bad_files = 1
        stats.bad_rows = stats.rows
        stats.errors.append(f"{path}: missing input_ids")
    else:
        expected_len = int(bucket) if bucket.isdigit() else None
        for row_group_idx in range(pf.metadata.num_row_groups):
            try:
                table = pf.read_row_group(row_group_idx, columns=cols)
            except Exception as exc:
                raise RuntimeError(
                    f"{path}#row_group{row_group_idx}: parquet read failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            stats.add(
                _audit_table(
                    path=path,
                    table=table,
                    names=names,
                    expected_len=expected_len,
                    vocab_size=vocab_size,
                )
            )

    if stats.bad_rows:
        stats.bad_files = 1

    return {
        "kind": kind,
        "bucket": bucket,
        "path": path_str,
        "stats": stats.as_dict(),
    }


def _discover(
    root: Path, kind: str, buckets: set[str] | None
) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for path in sorted(root.glob("*/*.parquet")):
        bucket = path.parent.name
        if buckets is not None and bucket not in buckets:
            continue
        out.append((str(path), kind, bucket))
    return out


def _rollup(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = AuditStats()
    by_kind: dict[str, AuditStats] = {}
    by_bucket: dict[str, AuditStats] = {}
    by_kind_bucket: dict[str, AuditStats] = {}
    bad_files: list[str] = []

    for result in results:
        kind = result["kind"]
        bucket = result["bucket"]
        stats = _stats_from_dict(result["stats"])
        total.add(stats)
        by_kind.setdefault(kind, AuditStats()).add(stats)
        by_bucket.setdefault(bucket, AuditStats()).add(stats)
        by_kind_bucket.setdefault(f"{kind}/{bucket}", AuditStats()).add(stats)
        if result["stats"]["bad_files"] or result["stats"]["bad_rows"]:
            bad_files.append(result["path"])

    return {
        "total": total.as_dict(),
        "by_kind": {k: v.as_dict() for k, v in sorted(by_kind.items())},
        "by_bucket": {
            k: v.as_dict()
            for k, v in sorted(
                by_bucket.items(),
                key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0],
            )
        },
        "by_kind_bucket": {k: v.as_dict() for k, v in sorted(by_kind_bucket.items())},
        "bad_files": bad_files[:10000],
    }


def _stats_from_dict(data: dict[str, Any]) -> AuditStats:
    stats = AuditStats()
    stats.files = data["files"]
    stats.rows = data["rows"]
    stats.capacity_tokens = data["capacity_tokens"]
    stats.valid_tokens = data["valid_tokens"]
    stats.trained_tokens = data["trained_tokens"]
    stats.slack_tokens = data["slack_tokens"]
    stats.bad_rows = data["bad_rows"]
    stats.bad_files = data["bad_files"]
    stats.edge_count = dict(data["edge_count"])
    stats.errors = list(data.get("errors") or [])
    for name, fdata in data["fields"].items():
        fs = FieldStats()
        for key in (
            "rows_present",
            "rows_nonempty",
            "rows_nonzero",
            "slots_total",
            "slots_nonzero",
            "bad_length_rows",
            "bad_value_rows",
            "missing_files",
        ):
            setattr(fs, key, int(fdata[key]))
        stats.field_stats[name] = fs
    return stats


def _write_md(report: dict[str, Any], path: Path) -> None:
    def fmt_int(value: int) -> str:
        return f"{value:,}"

    lines = ["# Sidecar Parquet Audit", ""]
    total = report["total"]
    lines.extend(
        [
            "## Totals",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| files | {fmt_int(total['files'])} |",
            f"| rows | {fmt_int(total['rows'])} |",
            f"| capacity tokens | {fmt_int(total['capacity_tokens'])} |",
            f"| valid tokens | {fmt_int(total['valid_tokens'])} |",
            f"| trained tokens | {fmt_int(total['trained_tokens'])} |",
            f"| pad pct | {total['pad_pct']} |",
            f"| bad files | {fmt_int(total['bad_files'])} |",
            f"| bad rows | {fmt_int(total['bad_rows'])} |",
            "",
            "## By Kind",
            "",
            "| kind | files | rows | valid tokens | capacity tokens | bad rows |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for kind, stats in report["by_kind"].items():
        lines.append(
            f"| {kind} | {fmt_int(stats['files'])} | {fmt_int(stats['rows'])} | "
            f"{fmt_int(stats['valid_tokens'])} | {fmt_int(stats['capacity_tokens'])} | {fmt_int(stats['bad_rows'])} |"
        )
    lines.extend(
        [
            "",
            "## By Kind/Bucket",
            "",
            "| split | files | rows | valid tokens | capacity tokens | pad pct | bad rows |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, stats in report["by_kind_bucket"].items():
        lines.append(
            f"| {key} | {fmt_int(stats['files'])} | {fmt_int(stats['rows'])} | "
            f"{fmt_int(stats['valid_tokens'])} | {fmt_int(stats['capacity_tokens'])} | {stats['pad_pct']} | {fmt_int(stats['bad_rows'])} |"
        )
    lines.extend(
        [
            "",
            "## Field Coverage And Correctness",
            "",
            "| field | rows present % | rows nonempty % | rows nonzero % | slots nonzero % | bad length rows | bad value rows | missing files |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for field_name, stats in total["fields"].items():
        lines.append(
            f"| `{field_name}` | {stats['rows_present_pct']} | {stats['rows_nonempty_pct']} | "
            f"{stats['rows_nonzero_pct']} | {stats['slots_nonzero_pct']} | "
            f"{fmt_int(stats['bad_length_rows'])} | {fmt_int(stats['bad_value_rows'])} | {fmt_int(stats['missing_files'])} |"
        )
    if report["bad_files"]:
        lines.extend(["", "## Bad Files", ""])
        lines.extend(f"- `{item}`" for item in report["bad_files"][:200])
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code-root", type=Path, required=True)
    ap.add_argument("--commit-root", type=Path, required=True)
    ap.add_argument("--pr-root", type=Path, required=True)
    ap.add_argument(
        "--buckets",
        default="",
        help="Comma-separated buckets to include, e.g. 1024,2048",
    )
    ap.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2))
    )
    ap.add_argument("--vocab-size", type=int, default=65536)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/sidecar_audit"))
    ap.add_argument(
        "--allow-bad",
        action="store_true",
        help=(
            "Escape hatch: do NOT block on bad files/rows. By default this gate "
            "is FAIL-CLOSED -- any bad file or bad row exits non-zero so the "
            "upload is blocked. Pass this flag to opt out explicitly."
        ),
    )
    ap.add_argument(
        "--fail-on-bad",
        action="store_true",
        help="Deprecated no-op: failing on bad files/rows is now the default.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    buckets = {item.strip() for item in args.buckets.split(",") if item.strip()} or None
    files: list[tuple[str, str, str]] = []
    files.extend(_discover(args.code_root, "code", buckets))
    files.extend(_discover(args.commit_root, "commits", buckets))
    files.extend(_discover(args.pr_root, "pr", buckets))
    if not files:
        raise SystemExit("no parquet files selected")

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_audit_file, path, kind, bucket, args.vocab_size)
            for path, kind, bucket in files
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if idx % 100 == 0 or idx == len(futures):
                print(f"audited {idx}/{len(futures)} parquet files", flush=True)

    report = _rollup(results)
    has_bad = bool(report["total"]["bad_files"] or report["total"]["bad_rows"])
    status = (
        "overridden"
        if has_bad and args.allow_bad
        else "failed"
        if has_bad
        else "passed"
    )
    report["status"] = status
    report["receipt"] = {
        "contract": "cppmega_case5_domain_routes_v1",
        "status": status,
        "successful": status == "passed",
        "tokenizer_domain_contract_sha256": DOMAIN_DELIMITER_CONTRACT_SHA256,
        "domain_schema_sha256": DOMAIN_SCHEMA_SHA256,
        "tokenizer_contract_sha256": TOKENIZER_CONTRACT_SHA256,
        "files": report["total"]["files"],
        "rows": report["total"]["rows"],
        "valid_tokens": report["total"]["valid_tokens"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sidecar_parquet_audit.json").write_text(
        json.dumps(report, indent=2)
    )
    _write_md(report, args.out_dir / "sidecar_parquet_audit.md")
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "files": report["total"]["files"],
                "valid_tokens": report["total"]["valid_tokens"],
                "capacity_tokens": report["total"]["capacity_tokens"],
                "bad_files": report["total"]["bad_files"],
                "bad_rows": report["total"]["bad_rows"],
                "status": status,
            },
            indent=2,
        )
    )
    if has_bad and not args.allow_bad:
        # FAIL-CLOSED default: block the upload by exiting non-zero.
        print(
            "AUDIT FAILED (fail-closed): "
            f"bad_files={report['total']['bad_files']} "
            f"bad_rows={report['total']['bad_rows']}. "
            "Upload must be blocked. Pass --allow-bad to override explicitly.",
            flush=True,
        )
        return 2
    if has_bad:
        print("AUDIT OVERRIDDEN: receipt is not successful", flush=True)
    else:
        print("AUDIT PASSED: successful CASE5 receipt written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
