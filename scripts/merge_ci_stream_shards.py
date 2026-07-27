#!/usr/bin/env python3
"""Auditably merge frozen CI fetch shards before CASE5 export.

The merge is deliberately conservative.  Every source artifact is named by a
canonical manifest that binds its original producer path to the staged path
being read.  Source SQLite files are immutable snapshots, source stores are
fully receipt-verified, and source CAS/fetch-state joins are replayed before
any output can be published.

The destination is built in a sibling partial directory.  Store batches use
``CIContentStore.add_chunks`` and are safe to replay if a process dies after
the store commit but before the external journal advances.  Fetch-state rows
and their external source-ID maps commit together through an attached SQLite
transaction.  The destination fetch-state itself remains the exact standard
v3 schema; merge maps and progress never leak into that schema.

One completed full inventory is the publication anchor.  Additional inputs
may either name the same byte-identical completed inventory or use the exact
``cppmega_ci_inventory_time_shard_v1`` projection produced for time-sharded
fetching.  A time shard is accepted only after every run row is proven
byte-for-byte identical to the anchor; it need not cover its whole declared
time window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import sqlite3
import stat
import struct
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping, Sequence, cast
import zlib


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci_content_store import (  # noqa: E402
    CIContentStore,
    ContentStoreError,
    ThresholdNotMetError,
    _hash_records,
    _sqlite_schema_sha256,
)
from scripts.ci_stream_fetch import (  # noqa: E402
    RECEIPT_SCHEMA as FETCH_RECEIPT_SCHEMA,
    SCHEMA_VERSION as FETCH_STATE_SCHEMA,
    _BINDING_KEYS,
    _STATE_SCHEMA as FETCH_STATE_SQL_SCHEMA,
    ExactTokenizer,
)
from scripts.ci_stream_inventory import (  # noqa: E402
    InventoryDB,
    InventoryError,
    RECEIPT_SCHEMA as INVENTORY_RECEIPT_SCHEMA,
    _SCHEMA_SQL as INVENTORY_SQL,
    format_utc_instant,
    parse_utc_instant,
)
from scripts.export_ci_content_store_case5 import (  # noqa: E402
    ExportError,
    FrozenFetchState,
    FrozenStore,
    SnapshotFile,
    _expected_fetch_state_schema_sha256,
    _fetch_state_logical_digest,
    _publish_directory_no_replace,
    _require_frozen_sqlite,
    _sha256_file,
)


MANIFEST_SCHEMA = "cppmega_ci_stream_shard_union_manifest_v1"
MERGE_RECEIPT_SCHEMA = "cppmega_ci_stream_shard_union_receipt_v2"
JOURNAL_SCHEMA = "cppmega_ci_stream_shard_union_journal_v2"
REQUEST_MAP_SCHEMA = "cppmega_ci_stream_request_id_map_v1"
BINDING_MAP_SCHEMA = "cppmega_ci_stream_binding_id_map_v2"
ATTEMPT_MAP_SCHEMA = "cppmega_ci_stream_attempt_resolution_v1"
MEMBER_MAP_SCHEMA = "cppmega_ci_stream_member_resolution_v1"
TIME_SHARD_INVENTORY_SCHEMA = "cppmega_ci_inventory_time_shard_v1"
INVENTORY_BINDING_SCHEMA = "cppmega_ci_stream_union_inventory_binding_v1"

_INVENTORY_NAME = "inventory.sqlite3"
_INVENTORY_RECEIPT_NAME = "inventory_receipt.json"
_STORE_DIRECTORY = "content_store"
_STORE_RECEIPT_NAME = "store_receipt.json"
_FETCH_STATE_NAME = "fetch_state.sqlite3"
_FETCH_RECEIPT_NAME = "fetch_receipt.json"
_JOURNAL_NAME = "merge_journal.sqlite3"
_MERGE_RECEIPT_NAME = "merge_receipt.json"
_LEDGER_DIRECTORY = "ledgers"
_LEDGER_FILENAMES = (
    "attempt_resolutions.jsonl",
    "member_resolutions.jsonl",
    "request_id_map.jsonl",
    "binding_id_map.jsonl",
)
_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_SHARD_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_HARD_MAX_SHARDS = 128
_HARD_MAX_JSON_BYTES = 64 * 1024 * 1024
_HARD_MAX_TOKENIZER_BYTES = 256 * 1024 * 1024
_SIGKILL_FAULT_POINTS = (
    "journal-file-created",
    "journal-schema-created",
    "journal-before-settings-commit",
    "state-file-created",
    "state-schema-created",
    "state-before-settings-commit",
)

_ATTEMPT_COLUMNS = (
    "repo",
    "run_id",
    "attempt",
    "created_at",
    "run_metadata_sha256",
    "run_metadata_raw_size",
    "run_metadata_zlib",
    "run_metadata_source",
    "run_metadata_source_attempt",
    "run_metadata_exact",
    "inventory_seed_attempt",
    "inventory_seed_metadata_sha256",
    "status",
    "tries",
    "archive_source",
    "archive_sha256",
    "archive_size",
    "jobs_sha256",
    "jobs_raw_size",
    "jobs_zlib",
    "member_count",
    "chunk_count",
    "occurrence_tokens",
    "terminal_http_status",
    "terminal_body_sha256",
    "error_class",
    "error_message",
    "discovered_at",
    "updated_at",
)
_ATTEMPT_KEY = ("repo", "run_id", "attempt")
_INVENTORY_RUN_COLUMNS = (
    "repo_key",
    "run_id",
    "run_attempt",
    "created_at",
    "updated_at",
    "run_started_at",
    "status",
    "conclusion",
    "workflow_id",
    "workflow_name",
    "event",
    "head_branch",
    "head_sha",
    "run_number",
    "html_url",
    "api_url",
    "metadata_blob",
    "metadata_sha256",
    "first_seen_at",
)
_TIME_SHARD_META_KEYS = {
    "schema",
    "source_inventory_path",
    "created_at",
    "created_at_gte",
    "created_at_lt",
    "run_count",
}
_ATTEMPT_IMMUTABLE_EVIDENCE = (
    "repo",
    "run_id",
    "attempt",
    "created_at",
    "run_metadata_sha256",
    "run_metadata_raw_size",
    "run_metadata_zlib",
    "run_metadata_source",
    "run_metadata_source_attempt",
    "run_metadata_exact",
    "inventory_seed_attempt",
    "inventory_seed_metadata_sha256",
)
_MEMBER_COLUMNS = (
    "repo",
    "run_id",
    "attempt",
    "archive_member",
    "job_key",
    "raw_sha256",
    "raw_size",
    "canonical_sha256",
    "dedup_sha256",
    "sidecar_sha256",
    "sidecar_raw_size",
    "sidecar_zlib",
    "chunk_count",
    "occurrence_tokens",
)
_MEMBER_KEY = ("repo", "run_id", "attempt", "archive_member")
_REQUEST_COLUMNS = (
    "id",
    "requested_at",
    "repo",
    "run_id",
    "attempt",
    "endpoint",
    "page_no",
    "request_attempt",
    "http_status",
    "outcome",
    "latency_ms",
    "error_class",
    "error_message",
)
_BINDING_COLUMNS = (
    "id",
    "binding_key",
    "from_sha256",
    "to_sha256",
    "reason",
    "upgraded_at",
)

_JOURNAL_SQL = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS store_progress (
    shard_id TEXT PRIMARY KEY,
    cursor_json TEXT,
    processed_rows INTEGER NOT NULL DEFAULT 0 CHECK(processed_rows >= 0),
    batches INTEGER NOT NULL DEFAULT 0 CHECK(batches >= 0),
    done INTEGER NOT NULL DEFAULT 0 CHECK(done IN (0,1))
);
CREATE TABLE IF NOT EXISTS state_progress (
    shard_id TEXT NOT NULL,
    table_name TEXT NOT NULL CHECK(
      table_name IN ('attempts','members','request_ledger','binding_upgrades')
    ),
    cursor_json TEXT,
    processed_rows INTEGER NOT NULL DEFAULT 0 CHECK(processed_rows >= 0),
    batches INTEGER NOT NULL DEFAULT 0 CHECK(batches >= 0),
    done INTEGER NOT NULL DEFAULT 0 CHECK(done IN (0,1)),
    PRIMARY KEY(shard_id,table_name)
);
CREATE TABLE IF NOT EXISTS attempt_map (
    shard_id TEXT NOT NULL,
    source_key_json TEXT NOT NULL,
    source_row_sha256 TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK(
      outcome IN (
        'inserted','exact_overlap','pending_shadowed_by_done',
        'done_replaced_zero_evidence_pending'
      )
    ),
    PRIMARY KEY(shard_id,source_key_json)
);
CREATE TABLE IF NOT EXISTS member_map (
    shard_id TEXT NOT NULL,
    source_key_json TEXT NOT NULL,
    source_row_sha256 TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK(outcome IN ('inserted','exact_overlap')),
    PRIMARY KEY(shard_id,source_key_json)
);
CREATE TABLE IF NOT EXISTS request_id_map (
    shard_id TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    destination_id INTEGER NOT NULL UNIQUE,
    source_row_sha256 TEXT NOT NULL,
    PRIMARY KEY(shard_id,source_id)
);
CREATE TABLE IF NOT EXISTS binding_id_map (
    shard_id TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    destination_id INTEGER NOT NULL,
    source_row_sha256 TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK(
      outcome IN ('inserted','exact_overlap','canonical_overlap')
    ),
    PRIMARY KEY(shard_id,source_id)
);
"""


class MergeError(RuntimeError):
    """A shard cannot be proven safe to merge or publish."""


class MergePaused(MergeError):
    """An operator-requested batch bound stopped a resumable merge."""


def _fault_inject_sigkill(
    configured: str | None,
    point: str,
) -> None:
    if configured == point:
        os.kill(os.getpid(), signal.SIGKILL)


@dataclass(frozen=True)
class ReceiptDescriptor:
    path: Path
    sha256: str


@dataclass(frozen=True)
class FileDescriptor:
    path: Path
    sha256: str
    receipt: ReceiptDescriptor


@dataclass(frozen=True)
class InventoryDescriptor:
    path: Path
    sha256: str
    receipt: ReceiptDescriptor | None


@dataclass(frozen=True)
class StoreDescriptor:
    path: Path
    artifact_set_sha256: str
    receipt: ReceiptDescriptor


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    original_inventory: str
    original_store: str
    original_state: str
    inventory: InventoryDescriptor
    store: StoreDescriptor
    state: FileDescriptor


@dataclass(frozen=True)
class Limits:
    max_shards: int
    occurrences_per_batch: int
    state_rows_per_batch: int
    uncompressed_bytes_per_batch: int
    max_content_bytes: int
    max_provenance_bytes: int
    max_state_blob_bytes: int


@dataclass(frozen=True)
class Manifest:
    path: Path
    sha256: str
    value: dict[str, Any]
    destination: Path
    target_unique_tokens: int
    tokenizer_path: Path
    tokenizer_sha256: str
    limits: Limits
    shards: tuple[ShardSpec, ...]


@dataclass(frozen=True)
class SourceAudit:
    spec: ShardSpec
    store_receipt: dict[str, Any]
    fetch_receipt: dict[str, Any]
    inventory_receipt: dict[str, Any] | None
    inventory_role: str
    inventory_proof: dict[str, Any]
    store_files: tuple[SnapshotFile, ...]
    state_file: SnapshotFile
    inventory_file: SnapshotFile
    receipt_files: tuple[SnapshotFile, ...]
    state_binding: dict[str, Any]
    store_counts: dict[str, int]
    state_counts: dict[str, int]


@dataclass(frozen=True)
class InventoryContract:
    anchor: SourceAudit
    binding: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise MergeError(f"value is not canonical JSON: {exc}") from exc


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MergeError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, where: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise MergeError(f"{where} is missing or unsafe: {path}")
    if path.stat().st_size > _HARD_MAX_JSON_BYTES:
        raise MergeError(f"{where} exceeds the hard JSON byte bound")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MergeError(f"{where} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"{where} must contain one JSON object")
    return value, raw


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    where: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise MergeError(f"{where} keys differ; missing={missing}, extra={extra}")


def _require_mapping(value: object, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MergeError(f"{where} must be an object")
    return value


def _require_string(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise MergeError(f"{where} must be a non-empty string")
    return value


def _require_hex64(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise MergeError(f"{where} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MergeError(f"{where} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MergeError(f"{where} must be a non-negative integer")
    return value


def _original_path(value: object, *, where: str) -> str:
    raw = _require_string(value, where=where)
    pure = PurePosixPath(raw)
    if not pure.is_absolute() or str(pure) != raw or ".." in pure.parts:
        raise MergeError(f"{where} must be a canonical absolute POSIX path")
    return raw


def _local_path(value: object, *, where: str) -> Path:
    raw = _require_string(value, where=where)
    path = Path(raw)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != raw
        or path.resolve(strict=False) != path
    ):
        raise MergeError(
            f"{where} must be a canonical resolved absolute local path"
        )
    return path


def _receipt_descriptor(
    value: object,
    *,
    where: str,
) -> ReceiptDescriptor:
    item = _require_mapping(value, where=where)
    _require_exact_keys(item, {"path", "sha256"}, where=where)
    return ReceiptDescriptor(
        path=_local_path(item["path"], where=f"{where}.path"),
        sha256=_require_hex64(item["sha256"], where=f"{where}.sha256"),
    )


def _file_descriptor(value: object, *, where: str) -> FileDescriptor:
    item = _require_mapping(value, where=where)
    _require_exact_keys(item, {"path", "sha256", "receipt"}, where=where)
    return FileDescriptor(
        path=_local_path(item["path"], where=f"{where}.path"),
        sha256=_require_hex64(item["sha256"], where=f"{where}.sha256"),
        receipt=_receipt_descriptor(item["receipt"], where=f"{where}.receipt"),
    )


def _inventory_descriptor(
    value: object,
    *,
    where: str,
) -> InventoryDescriptor:
    item = _require_mapping(value, where=where)
    if set(item) not in ({"path", "sha256"}, {"path", "sha256", "receipt"}):
        raise MergeError(
            f"{where} keys differ; expected path/sha256 with an optional receipt"
        )
    raw_receipt = item.get("receipt")
    return InventoryDescriptor(
        path=_local_path(item["path"], where=f"{where}.path"),
        sha256=_require_hex64(item["sha256"], where=f"{where}.sha256"),
        receipt=(
            None
            if raw_receipt is None
            else _receipt_descriptor(raw_receipt, where=f"{where}.receipt")
        ),
    )


def _store_descriptor(value: object, *, where: str) -> StoreDescriptor:
    item = _require_mapping(value, where=where)
    _require_exact_keys(
        item,
        {"path", "artifact_set_sha256", "receipt"},
        where=where,
    )
    return StoreDescriptor(
        path=_local_path(item["path"], where=f"{where}.path"),
        artifact_set_sha256=_require_hex64(
            item["artifact_set_sha256"],
            where=f"{where}.artifact_set_sha256",
        ),
        receipt=_receipt_descriptor(item["receipt"], where=f"{where}.receipt"),
    )


def load_manifest(path: str | os.PathLike[str]) -> Manifest:
    """Load and strictly validate one canonical shard-union manifest."""

    manifest_path = Path(path).expanduser().resolve()
    value, raw = _load_json(manifest_path, where="union manifest")
    if raw != _canonical_json_bytes(value) + b"\n":
        raise MergeError("union manifest is not canonical compact JSON plus newline")
    _require_exact_keys(
        value,
        {"schema", "destination", "tokenizer", "limits", "shards"},
        where="union manifest",
    )
    if value["schema"] != MANIFEST_SCHEMA:
        raise MergeError(f"union manifest schema must be {MANIFEST_SCHEMA!r}")

    destination = _require_mapping(value["destination"], where="destination")
    _require_exact_keys(
        destination,
        {"bundle_path", "target_exact_unique_payload_tokens"},
        where="destination",
    )
    bundle_path = _local_path(
        destination["bundle_path"],
        where="destination.bundle_path",
    )
    target = _require_nonnegative_int(
        destination["target_exact_unique_payload_tokens"],
        where="destination.target_exact_unique_payload_tokens",
    )

    tokenizer = _require_mapping(value["tokenizer"], where="tokenizer")
    _require_exact_keys(tokenizer, {"path", "sha256"}, where="tokenizer")
    tokenizer_path = _local_path(tokenizer["path"], where="tokenizer.path")
    tokenizer_sha256 = _require_hex64(
        tokenizer["sha256"],
        where="tokenizer.sha256",
    )

    raw_limits = _require_mapping(value["limits"], where="limits")
    _require_exact_keys(
        raw_limits,
        {
            "max_shards",
            "occurrences_per_batch",
            "state_rows_per_batch",
            "uncompressed_bytes_per_batch",
            "max_content_bytes",
            "max_provenance_bytes",
            "max_state_blob_bytes",
        },
        where="limits",
    )
    limits = Limits(
        max_shards=_require_positive_int(
            raw_limits["max_shards"], where="limits.max_shards"
        ),
        occurrences_per_batch=_require_positive_int(
            raw_limits["occurrences_per_batch"],
            where="limits.occurrences_per_batch",
        ),
        state_rows_per_batch=_require_positive_int(
            raw_limits["state_rows_per_batch"],
            where="limits.state_rows_per_batch",
        ),
        uncompressed_bytes_per_batch=_require_positive_int(
            raw_limits["uncompressed_bytes_per_batch"],
            where="limits.uncompressed_bytes_per_batch",
        ),
        max_content_bytes=_require_positive_int(
            raw_limits["max_content_bytes"],
            where="limits.max_content_bytes",
        ),
        max_provenance_bytes=_require_positive_int(
            raw_limits["max_provenance_bytes"],
            where="limits.max_provenance_bytes",
        ),
        max_state_blob_bytes=_require_positive_int(
            raw_limits["max_state_blob_bytes"],
            where="limits.max_state_blob_bytes",
        ),
    )
    if limits.max_shards > _HARD_MAX_SHARDS:
        raise MergeError(f"limits.max_shards cannot exceed {_HARD_MAX_SHARDS}")
    if (
        limits.max_content_bytes > limits.uncompressed_bytes_per_batch
        or limits.max_provenance_bytes > limits.uncompressed_bytes_per_batch
    ):
        raise MergeError("per-record limits cannot exceed the batch byte limit")

    raw_shards = value["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise MergeError("shards must be a non-empty list")
    if len(raw_shards) > limits.max_shards:
        raise MergeError("manifest shard count exceeds its explicit max_shards")
    shards: list[ShardSpec] = []
    for index, raw_shard in enumerate(raw_shards):
        where = f"shards[{index}]"
        item = _require_mapping(raw_shard, where=where)
        _require_exact_keys(
            item,
            {"id", "original_paths", "staged"},
            where=where,
        )
        shard_id = _require_string(item["id"], where=f"{where}.id")
        if _SHARD_ID_RE.fullmatch(shard_id) is None:
            raise MergeError(f"{where}.id is not a canonical shard identifier")
        originals = _require_mapping(
            item["original_paths"],
            where=f"{where}.original_paths",
        )
        _require_exact_keys(
            originals,
            {"inventory", "content_store", "fetch_state"},
            where=f"{where}.original_paths",
        )
        staged = _require_mapping(item["staged"], where=f"{where}.staged")
        _require_exact_keys(
            staged,
            {"inventory", "content_store", "fetch_state"},
            where=f"{where}.staged",
        )
        shards.append(
            ShardSpec(
                shard_id=shard_id,
                original_inventory=_original_path(
                    originals["inventory"],
                    where=f"{where}.original_paths.inventory",
                ),
                original_store=_original_path(
                    originals["content_store"],
                    where=f"{where}.original_paths.content_store",
                ),
                original_state=_original_path(
                    originals["fetch_state"],
                    where=f"{where}.original_paths.fetch_state",
                ),
                inventory=_inventory_descriptor(
                    staged["inventory"],
                    where=f"{where}.staged.inventory",
                ),
                store=_store_descriptor(
                    staged["content_store"],
                    where=f"{where}.staged.content_store",
                ),
                state=_file_descriptor(
                    staged["fetch_state"],
                    where=f"{where}.staged.fetch_state",
                ),
            )
        )
    shard_ids = [shard.shard_id for shard in shards]
    if shard_ids != sorted(shard_ids) or len(set(shard_ids)) != len(shard_ids):
        raise MergeError("shards must be uniquely sorted by canonical id")
    inventory_paths = [shard.inventory.path for shard in shards]
    inventory_receipt_paths = [
        shard.inventory.receipt.path
        for shard in shards
        if shard.inventory.receipt is not None
    ]
    exclusive_paths = [
        path
        for shard in shards
        for path in (
            shard.store.path,
            shard.store.receipt.path,
            shard.state.path,
            shard.state.receipt.path,
        )
    ]
    if len({str(item) for item in exclusive_paths}) != len(exclusive_paths):
        raise MergeError("non-inventory staged artifact paths must be distinct")
    inventory_artifact_paths = {
        str(path) for path in (*inventory_paths, *inventory_receipt_paths)
    }
    if len(inventory_artifact_paths) != (
        len({str(path) for path in inventory_paths})
        + len({str(path) for path in inventory_receipt_paths})
    ):
        raise MergeError(
            "an inventory database path cannot also be an inventory receipt path"
        )
    if inventory_artifact_paths & {str(path) for path in exclusive_paths}:
        raise MergeError("inventory and non-inventory staged paths overlap")
    return Manifest(
        path=manifest_path,
        sha256=_sha256_bytes(raw),
        value=value,
        destination=bundle_path,
        target_unique_tokens=target,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha256,
        limits=limits,
        shards=tuple(shards),
    )


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_document_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, _json_document_bytes(value))


def _snapshot_file(path: Path, *, label: str) -> SnapshotFile:
    if path.is_symlink() or not path.is_file():
        raise MergeError(f"{label} is missing or unsafe: {path}")
    stat_before = path.stat()
    digest = _sha256_file(path)
    stat_after = path.stat()
    identity_before = (
        stat_before.st_size,
        stat_before.st_mtime_ns,
        stat_before.st_ino,
    )
    identity_after = (
        stat_after.st_size,
        stat_after.st_mtime_ns,
        stat_after.st_ino,
    )
    if identity_before != identity_after:
        raise MergeError(f"{label} changed while it was hashed")
    return SnapshotFile(
        relative_path=path.name,
        size=stat_after.st_size,
        mtime_ns=stat_after.st_mtime_ns,
        inode=stat_after.st_ino,
        sha256=digest,
    )


def _store_artifact_set_digest(files: Iterable[SnapshotFile]) -> str:
    records = (
        {
            "path": item.relative_path,
            "byte_size": item.size,
            "sha256": item.sha256,
        }
        for item in files
    )
    return _hash_records("cppmega-ci-frozen-store-artifact-set-v1", records)


def frozen_store_artifact_set_sha256(
    store_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
) -> str:
    """Return the manifest digest for one fully verified frozen store."""

    with FrozenStore(Path(store_path), Path(receipt_path)) as store:
        result = _store_artifact_set_digest(store._initial_snapshot)
        store.require_unchanged()
        return result


def _load_bound_receipt(descriptor: ReceiptDescriptor, *, where: str) -> dict[str, Any]:
    value, raw = _load_json(descriptor.path, where=where)
    actual = _sha256_bytes(raw)
    if actual != descriptor.sha256:
        raise MergeError(f"{where} SHA-256 differs from the manifest")
    return value


def _state_blob_limits(connection: sqlite3.Connection, limits: Limits) -> None:
    text_fields = {
        "settings": ("key", "value"),
        "attempts": tuple(
            column
            for column in _ATTEMPT_COLUMNS
            if column not in {"run_metadata_zlib", "jobs_zlib"}
        ),
        "members": tuple(
            column for column in _MEMBER_COLUMNS if column != "sidecar_zlib"
        ),
        "request_ledger": _REQUEST_COLUMNS,
        "binding_upgrades": _BINDING_COLUMNS,
    }
    for table, columns in text_fields.items():
        predicates = " OR ".join(
            f"COALESCE(length(CAST({column} AS BLOB)),0)>?"
            for column in columns
        )
        oversized = connection.execute(
            f"SELECT rowid FROM {table} WHERE {predicates} LIMIT 1",
            (limits.max_state_blob_bytes,) * len(columns),
        ).fetchone()
        if oversized is not None:
            raise MergeError(
                f"fetch-state {table} field exceeds max_state_blob_bytes"
            )
    oversized_attempt = connection.execute(
        """
        SELECT repo,run_id,attempt
        FROM attempts
        WHERE run_metadata_raw_size > ?
           OR COALESCE(jobs_raw_size,0) > ?
           OR length(run_metadata_zlib) > ?
           OR COALESCE(length(jobs_zlib),0) > ?
        LIMIT 1
        """,
        (
            limits.max_state_blob_bytes,
            limits.max_state_blob_bytes,
            limits.max_state_blob_bytes,
            limits.max_state_blob_bytes,
        ),
    ).fetchone()
    if oversized_attempt is not None:
        raise MergeError(
            "fetch-state attempt blob exceeds max_state_blob_bytes: "
            f"{tuple(oversized_attempt)}"
        )
    oversized_member = connection.execute(
        """
        SELECT repo,run_id,attempt,archive_member
        FROM members
        WHERE sidecar_raw_size > ? OR length(sidecar_zlib) > ?
        LIMIT 1
        """,
        (limits.max_state_blob_bytes, limits.max_state_blob_bytes),
    ).fetchone()
    if oversized_member is not None:
        raise MergeError(
            "fetch-state member sidecar exceeds max_state_blob_bytes: "
            f"{tuple(oversized_member)}"
        )


def _reject_unsafe_attempt_states(connection: sqlite3.Connection) -> None:
    processing = connection.execute(
        """
        SELECT repo,run_id,attempt FROM attempts
        WHERE status='processing' LIMIT 1
        """
    ).fetchone()
    if processing is not None:
        raise MergeError(f"processing fetch attempt cannot be frozen: {tuple(processing)}")
    cas_non_done = connection.execute(
        """
        SELECT repo,run_id,attempt,status
        FROM attempts
        WHERE status!='done' AND (chunk_count>0 OR occurrence_tokens>0)
        LIMIT 1
        """
    ).fetchone()
    if cas_non_done is not None:
        raise MergeError(
            "CAS-bearing non-done fetch attempt is unsupported: "
            f"{tuple(cas_non_done)}"
        )
    positive_member = connection.execute(
        """
        SELECT members.repo,members.run_id,members.attempt,
               members.archive_member,attempts.status
        FROM members
        JOIN attempts USING(repo,run_id,attempt)
        WHERE attempts.status!='done'
          AND (members.chunk_count>0 OR members.occurrence_tokens>0)
        LIMIT 1
        """
    ).fetchone()
    if positive_member is not None:
        raise MergeError(
            "CAS-bearing member belongs to a non-done attempt: "
            f"{tuple(positive_member)}"
        )


def _validate_binding_history(
    connection: sqlite3.Connection,
    *,
    current_bindings: Mapping[str, str],
) -> None:
    currents = {
        binding_key: _require_hex64(
            current_bindings.get(binding_key, ""),
            where=f"fetch-state {binding_key}",
        )
        for binding_key in _BINDING_KEYS
    }
    histories: dict[str, list[sqlite3.Row]] = {
        binding_key: [] for binding_key in _BINDING_KEYS
    }
    for row in connection.execute(
        """
        SELECT binding_key,from_sha256,to_sha256,upgraded_at
        FROM binding_upgrades
        ORDER BY id
        """
    ):
        binding_key = str(row["binding_key"])
        if binding_key not in histories:
            raise MergeError("fetch-state binding history has an unsupported key")
        _canonical_binding_upgrade_time(row["upgraded_at"])
        histories[binding_key].append(row)
    for binding_key, rows in histories.items():
        previous_to: str | None = None
        for row in rows:
            source = _require_hex64(
                row["from_sha256"],
                where=f"{binding_key} upgrade from_sha256",
            )
            destination = _require_hex64(
                row["to_sha256"],
                where=f"{binding_key} upgrade to_sha256",
            )
            if source == destination:
                raise MergeError(
                    f"{binding_key} history contains a no-op upgrade"
                )
            if previous_to is not None and source != previous_to:
                raise MergeError(
                    f"{binding_key} history is not a linear chain"
                )
            previous_to = destination
        if previous_to is not None and previous_to != currents[binding_key]:
            raise MergeError(
                f"{binding_key} history does not terminate at its current binding"
            )


def _canonical_binding_upgrade_time(value: object) -> str:
    raw = str(value)
    try:
        canonical = format_utc_instant(parse_utc_instant(raw))
    except ValueError as exc:
        raise MergeError("binding upgrade timestamp is invalid") from exc
    if raw != canonical:
        raise MergeError("binding upgrade timestamp is not canonical UTC")
    return canonical


def _create_seen_chunks(path: Path) -> sqlite3.Connection:
    _remove_scratch_sqlite(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        CREATE TABLE seen_chunks (
          repo TEXT NOT NULL,
          run_id INTEGER NOT NULL,
          attempt INTEGER NOT NULL,
          archive_member TEXT NOT NULL,
          chunk_ordinal INTEGER NOT NULL,
          token_count INTEGER NOT NULL,
          PRIMARY KEY(repo,run_id,attempt,archive_member,chunk_ordinal)
        );
        """
    )
    return connection


def _remove_scratch_sqlite(path: Path) -> None:
    for candidate in (
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        Path(f"{path}-journal"),
    ):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise MergeError(f"coverage scratch is unsafe: {candidate}")
        candidate.unlink()


def _verify_cas_fetch_join(
    store: FrozenStore,
    state: FrozenFetchState,
    *,
    scratch_path: Path,
    limits: Limits,
) -> int:
    orphan = store.connection.execute(
        """
        SELECT contents.sha256
        FROM contents
        LEFT JOIN occurrences ON occurrences.content_sha256=contents.sha256
        WHERE occurrences.content_sha256 IS NULL
        LIMIT 1
        """
    ).fetchone()
    if orphan is not None:
        raise MergeError(f"content has no occurrence witness: {orphan[0]}")
    oversized_content = store.connection.execute(
        "SELECT sha256 FROM contents WHERE raw_size>? LIMIT 1",
        (limits.max_content_bytes,),
    ).fetchone()
    if oversized_content is not None:
        raise MergeError(
            f"content exceeds max_content_bytes: {oversized_content[0]}"
        )
    oversized_provenance = store.connection.execute(
        """
        SELECT repo,run_attempt,job,step,chunk_ordinal
        FROM occurrences WHERE provenance_raw_size>? LIMIT 1
        """,
        (limits.max_provenance_bytes,),
    ).fetchone()
    if oversized_provenance is not None:
        raise MergeError(
            "occurrence provenance exceeds max_provenance_bytes: "
            f"{tuple(oversized_provenance)}"
        )
    coverage = _create_seen_chunks(scratch_path)
    count = 0
    try:
        coverage.execute("BEGIN")
        cursor = store.connection.execute(
            """
            SELECT repo,run_attempt,job,step,chunk_ordinal,
                   content_sha256,provenance_sha256,
                   provenance_raw_size,provenance_zlib
            FROM occurrences
            ORDER BY repo,run_attempt,job,step,chunk_ordinal
            """
        )
        for row in cursor:
            occurrence = store._occurrence_record(row)
            member = state.validate_occurrence(occurrence)
            content = store.get_content_record(occurrence.content_sha256)
            try:
                coverage.execute(
                    """
                    INSERT INTO seen_chunks(
                      repo,run_id,attempt,archive_member,
                      chunk_ordinal,token_count
                    ) VALUES (?,?,?,?,?,?)
                    """,
                    (*member.key, occurrence.key[4], content.token_count),
                )
            except sqlite3.IntegrityError as exc:
                raise MergeError(
                    "CAS contains duplicate fetch-member chunk coverage"
                ) from exc
            count += 1
            if count % limits.state_rows_per_batch == 0:
                coverage.commit()
                coverage.execute("BEGIN")
        coverage.commit()
        state.verify_member_coverage(coverage)
    finally:
        coverage.close()
        _remove_scratch_sqlite(scratch_path)
        _fsync_directory(scratch_path.parent)
    return count


def _snapshot_receipts(spec: ShardSpec) -> tuple[SnapshotFile, ...]:
    snapshots: list[SnapshotFile] = []
    if spec.inventory.receipt is not None:
        snapshots.append(
            _snapshot_file(
                spec.inventory.receipt.path,
                label="inventory receipt",
            )
        )
    snapshots.extend(
        (
            _snapshot_file(spec.store.receipt.path, label="store receipt"),
            _snapshot_file(spec.state.receipt.path, label="fetch receipt"),
        )
    )
    return tuple(snapshots)


def _integer_counts(value: Mapping[str, Any], fields: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in fields:
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise MergeError(f"receipt counter {field!r} is invalid")
        result[field] = raw
    return result


_TIME_SHARD_SQL = """
CREATE TABLE runs (
    repo_key TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    run_attempt INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    run_started_at TEXT,
    status TEXT,
    conclusion TEXT,
    workflow_id INTEGER,
    workflow_name TEXT,
    event TEXT,
    head_branch TEXT,
    head_sha TEXT,
    run_number INTEGER,
    html_url TEXT,
    api_url TEXT,
    metadata_blob BLOB NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    PRIMARY KEY(repo_key, run_id, run_attempt)
);
CREATE INDEX idx_runs_created
    ON runs(repo_key, created_at, run_id, run_attempt);
CREATE TABLE shard_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _normalized_sql(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _time_shard_schema_contract(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    objects = [
        [str(row["type"]), str(row["name"]), str(row["tbl_name"])]
        for row in connection.execute(
            """
            SELECT type,name,tbl_name
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type,name
            """
        )
    ]
    tables: dict[str, list[list[object]]] = {}
    foreign_keys: dict[str, list[list[object]]] = {}
    for table in ("runs", "shard_meta"):
        tables[table] = [
            [
                int(row["cid"]),
                str(row["name"]),
                str(row["type"]),
                int(row["notnull"]),
                row["dflt_value"],
                int(row["pk"]),
            ]
            for row in connection.execute(f"PRAGMA table_info({table})")
        ]
        foreign_keys[table] = [
            list(row)
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        ]
    indexes = [
        [
            int(row["seqno"]),
            int(row["cid"]),
            None if row["name"] is None else str(row["name"]),
            int(row["desc"]),
            str(row["coll"]),
            int(row["key"]),
        ]
        for row in connection.execute("PRAGMA index_xinfo(idx_runs_created)")
    ]
    sql = {
        str(row["name"]): _normalized_sql(row["sql"])
        for row in connection.execute(
            """
            SELECT name,sql FROM sqlite_schema
            WHERE name IN ('runs','idx_runs_created','shard_meta')
            ORDER BY name
            """
        )
    }
    return {
        "objects": objects,
        "tables": tables,
        "foreign_keys": foreign_keys,
        "idx_runs_created": indexes,
        "sql": sql,
    }


def _expected_time_shard_schema_contract() -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(_TIME_SHARD_SQL)
        return _time_shard_schema_contract(connection)
    finally:
        connection.close()


def _canonical_utc(value: object, *, where: str) -> tuple[str, int]:
    raw = _require_string(value, where=where)
    try:
        epoch = parse_utc_instant(raw)
    except ValueError as exc:
        raise MergeError(f"{where} is not a valid UTC instant: {exc}") from exc
    if format_utc_instant(epoch) != raw:
        raise MergeError(f"{where} must use canonical second-precision UTC form")
    return raw, epoch


def _inventory_row_hash_record(row: sqlite3.Row) -> list[object]:
    values: list[object] = []
    for column in _INVENTORY_RUN_COLUMNS:
        value = row[column]
        if column == "metadata_blob":
            if not isinstance(value, bytes):
                raise MergeError("inventory metadata_blob is not a SQLite BLOB")
            values.append(
                {
                    "byte_size": len(value),
                    "sha256": _sha256_bytes(value),
                }
            )
        else:
            values.append(value)
    return values


def _record_digest(domain: str) -> Any:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    return digest


def _update_record_digest(digest: Any, record: object) -> None:
    encoded = _canonical_json_bytes(record)
    digest.update(struct.pack(">Q", len(encoded)))
    digest.update(encoded)


def _bounded_zlib_decode(
    blob: bytes,
    *,
    max_blob_bytes: int,
    where: str,
) -> bytes:
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(blob, max_blob_bytes + 1)
        if len(raw) > max_blob_bytes or decompressor.unconsumed_tail:
            raise MergeError(f"{where} exceeds its decoded bound")
        raw += decompressor.flush(max_blob_bytes + 1 - len(raw))
    except zlib.error as exc:
        raise MergeError(f"{where} is invalid zlib") from exc
    if (
        len(raw) > max_blob_bytes
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise MergeError(f"{where} is invalid or exceeds its decoded bound")
    return raw


def _validate_inventory_run_rows(
    connection: sqlite3.Connection,
    manifest: Manifest,
    *,
    label: str,
) -> tuple[int, str]:
    count = 0
    digest = _record_digest("cppmega-ci-inventory-run-rows-v1")
    columns = ",".join(_INVENTORY_RUN_COLUMNS)
    for row in connection.execute(
        f"""
        SELECT {columns}
        FROM runs
        ORDER BY repo_key,run_id,run_attempt
        """
    ):
        key = f"{row['repo_key']}/{row['run_id']}/{row['run_attempt']}"
        blob = row["metadata_blob"]
        if not isinstance(blob, bytes):
            raise MergeError(f"{label} inventory metadata is not a BLOB: {key}")
        if len(blob) > manifest.limits.max_state_blob_bytes:
            raise MergeError(f"{label} inventory metadata BLOB exceeds its bound: {key}")
        raw = _bounded_zlib_decode(
            blob,
            max_blob_bytes=manifest.limits.max_state_blob_bytes,
            where=f"{label} inventory metadata",
        )
        if _sha256_bytes(raw) != row["metadata_sha256"]:
            raise MergeError(f"{label} inventory metadata digest mismatch: {key}")
        _update_record_digest(digest, _inventory_row_hash_record(row))
        count += 1
    return count, digest.hexdigest()


def _validate_time_shard_inventory(
    connection: sqlite3.Connection,
    manifest: Manifest,
    *,
    label: str,
) -> dict[str, Any]:
    schema_contract = _time_shard_schema_contract(connection)
    if schema_contract != _expected_time_shard_schema_contract():
        raise MergeError(f"{label} time-shard inventory schema is not exact v1")
    meta_count = int(
        connection.execute("SELECT COUNT(*) FROM shard_meta").fetchone()[0]
    )
    if meta_count != len(_TIME_SHARD_META_KEYS):
        raise MergeError(
            f"{label} time-shard metadata has {meta_count} rows; "
            f"expected {len(_TIME_SHARD_META_KEYS)}"
        )
    oversized_meta = connection.execute(
        """
        SELECT key FROM shard_meta
        WHERE length(CAST(key AS BLOB)) > ?
           OR length(CAST(value AS BLOB)) > ?
        LIMIT 1
        """,
        (
            manifest.limits.max_state_blob_bytes,
            manifest.limits.max_state_blob_bytes,
        ),
    ).fetchone()
    if oversized_meta is not None:
        raise MergeError(f"{label} time-shard metadata exceeds its bound")
    meta: dict[str, str] = {}
    for row in connection.execute(
        "SELECT key,value FROM shard_meta ORDER BY key"
    ):
        key = str(row["key"])
        if key in meta:
            raise MergeError(f"{label} time-shard metadata has a duplicate key")
        meta[key] = str(row["value"])
    if set(meta) != _TIME_SHARD_META_KEYS:
        raise MergeError(
            f"{label} time-shard metadata keys differ; "
            f"missing={sorted(_TIME_SHARD_META_KEYS - set(meta))}, "
            f"extra={sorted(set(meta) - _TIME_SHARD_META_KEYS)}"
        )
    if meta["schema"] != TIME_SHARD_INVENTORY_SCHEMA:
        raise MergeError(f"{label} time-shard metadata schema is not v1")
    source_inventory_path = _original_path(
        meta["source_inventory_path"],
        where=f"{label} shard_meta.source_inventory_path",
    )
    created_at, _created_epoch = _canonical_utc(
        meta["created_at"],
        where=f"{label} shard_meta.created_at",
    )
    created_at_gte, lower = _canonical_utc(
        meta["created_at_gte"],
        where=f"{label} shard_meta.created_at_gte",
    )
    created_at_lt, upper = _canonical_utc(
        meta["created_at_lt"],
        where=f"{label} shard_meta.created_at_lt",
    )
    if lower >= upper:
        raise MergeError(f"{label} time-shard bounds are empty or reversed")
    raw_count = meta["run_count"]
    if (
        len(raw_count) > 19
        or re.fullmatch(r"0|[1-9][0-9]*", raw_count) is None
    ):
        raise MergeError(f"{label} time-shard run_count is not canonical")
    declared_count = int(raw_count)
    if declared_count > 0x7FFF_FFFF_FFFF_FFFF:
        raise MergeError(f"{label} time-shard run_count exceeds SQLite bounds")
    run_count, logical_sha256 = _validate_inventory_run_rows(
        connection,
        manifest,
        label=label,
    )
    if run_count != declared_count:
        raise MergeError(f"{label} time-shard run_count differs from its rows")
    for row in connection.execute(
        """
        SELECT repo_key,run_id,run_attempt,created_at
        FROM runs ORDER BY repo_key,run_id,run_attempt
        """
    ):
        _raw_created, created_epoch = _canonical_utc(
            row["created_at"],
            where=(
                f"{label} run {row['repo_key']}/{row['run_id']}/"
                f"{row['run_attempt']} created_at"
            ),
        )
        if not lower <= created_epoch < upper:
            raise MergeError(f"{label} time-shard run lies outside its bounds")
    return {
        "schema": TIME_SHARD_INVENTORY_SCHEMA,
        "sqlite_schema_sha256": _sqlite_schema_sha256(connection),
        "sqlite_schema_contract_sha256": _sha256_bytes(
            _canonical_json_bytes(schema_contract)
        ),
        "source_inventory_path": source_inventory_path,
        "created_at": created_at,
        "created_at_gte": created_at_gte,
        "created_at_lt": created_at_lt,
        "run_count": run_count,
        "run_rows_logical_sha256": logical_sha256,
    }


def _verify_state_inventory_join(
    inventory: sqlite3.Connection,
    state: sqlite3.Connection,
    *,
    label: str,
    max_blob_bytes: int,
) -> tuple[int, str]:
    joined = 0
    digest = _record_digest("cppmega-ci-fetch-state-inventory-join-v1")
    for row in state.execute(
        """
        SELECT repo,run_id,attempt,run_metadata_sha256,
               run_metadata_raw_size,run_metadata_zlib,
               run_metadata_source,run_metadata_source_attempt,
               inventory_seed_attempt,inventory_seed_metadata_sha256
        FROM attempts ORDER BY repo,run_id,attempt
        """
    ):
        seed = inventory.execute(
            """
            SELECT metadata_blob,metadata_sha256
            FROM runs
            WHERE repo_key=? AND run_id=? AND run_attempt=?
            """,
            (
                str(row["repo"]),
                int(row["run_id"]),
                int(row["inventory_seed_attempt"]),
            ),
        ).fetchone()
        key = f"{row['repo']}/{row['run_id']}/{row['attempt']}"
        if seed is None:
            raise MergeError(f"{label} fetch-state attempt lacks an inventory seed: {key}")
        if row["inventory_seed_metadata_sha256"] != seed["metadata_sha256"]:
            raise MergeError(f"{label} fetch-state inventory seed binding differs: {key}")
        if row["run_metadata_source"] == "inventory-run-list":
            inventory_raw = _bounded_zlib_decode(
                bytes(seed["metadata_blob"]),
                max_blob_bytes=max_blob_bytes,
                where=f"{label} inventory seed metadata",
            )
            state_raw = _bounded_zlib_decode(
                bytes(row["run_metadata_zlib"]),
                max_blob_bytes=max_blob_bytes,
                where=f"{label} fetch-state run metadata",
            )
            if (
                int(row["run_metadata_source_attempt"])
                != int(row["inventory_seed_attempt"])
                or row["run_metadata_sha256"] != seed["metadata_sha256"]
                or int(row["run_metadata_raw_size"]) != len(inventory_raw)
                or _sha256_bytes(state_raw) != row["run_metadata_sha256"]
                or state_raw != inventory_raw
            ):
                raise MergeError(
                    f"{label} fetch-state inventory metadata binding differs: {key}"
                )
        _update_record_digest(
            digest,
            [
                str(row["repo"]),
                int(row["run_id"]),
                int(row["attempt"]),
                int(row["inventory_seed_attempt"]),
                str(row["inventory_seed_metadata_sha256"]),
            ],
        )
        joined += 1
    return joined, digest.hexdigest()


def _preflight_source(
    manifest: Manifest,
    spec: ShardSpec,
    tokenizer: ExactTokenizer,
    scratch_directory: Path,
) -> SourceAudit:
    receipt_files = _snapshot_receipts(spec)
    receipt_descriptors: list[tuple[ReceiptDescriptor, str]] = []
    if spec.inventory.receipt is not None:
        receipt_descriptors.append(
            (
                spec.inventory.receipt,
                f"{spec.shard_id} inventory receipt",
            )
        )
    receipt_descriptors.extend(
        (
            (spec.store.receipt, f"{spec.shard_id} store receipt"),
            (spec.state.receipt, f"{spec.shard_id} fetch receipt"),
        )
    )
    for snapshot, (descriptor, label) in zip(
        receipt_files,
        receipt_descriptors,
        strict=True,
    ):
        if snapshot.sha256 != descriptor.sha256:
            raise MergeError(f"{label} SHA-256 differs from the manifest")
    _require_frozen_sqlite(spec.inventory.path, label=f"{spec.shard_id} inventory")
    _require_frozen_sqlite(spec.state.path, label=f"{spec.shard_id} fetch state")
    inventory_file = _snapshot_file(
        spec.inventory.path,
        label=f"{spec.shard_id} inventory",
    )
    state_file = _snapshot_file(
        spec.state.path,
        label=f"{spec.shard_id} fetch state",
    )
    if inventory_file.sha256 != spec.inventory.sha256:
        raise MergeError(f"{spec.shard_id} inventory hash differs from manifest")
    if state_file.sha256 != spec.state.sha256:
        raise MergeError(f"{spec.shard_id} fetch-state hash differs from manifest")
    inventory_connection = sqlite3.connect(
        f"{spec.inventory.path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    inventory_connection.row_factory = sqlite3.Row
    try:
        integrity = [
            str(row[0])
            for row in inventory_connection.execute(
                "PRAGMA integrity_check"
            ).fetchall()
        ]
        if integrity != ["ok"]:
            raise MergeError(
                f"{spec.shard_id} inventory integrity_check failed: {integrity}"
            )
        if inventory_connection.execute("PRAGMA foreign_key_check").fetchall():
            raise MergeError(f"{spec.shard_id} inventory foreign_key_check failed")
        schema_sha256 = _sqlite_schema_sha256(inventory_connection)
        if schema_sha256 == _expected_inventory_schema_sha256():
            if spec.inventory.receipt is None:
                raise MergeError(
                    f"{spec.shard_id} completed anchor candidate lacks its receipt"
                )
            run_count, rows_sha256 = _validate_inventory_run_rows(
                inventory_connection,
                manifest,
                label=spec.shard_id,
            )
            inventory_role = "anchor_candidate"
            inventory_proof = {
                "schema": INVENTORY_RECEIPT_SCHEMA,
                "sqlite_schema_sha256": schema_sha256,
                "run_count": run_count,
                "run_rows_logical_sha256": rows_sha256,
            }
        else:
            if spec.inventory.receipt is not None:
                raise MergeError(
                    f"{spec.shard_id} non-anchor inventory must not claim "
                    "an anchor completion receipt"
                )
            inventory_role = "exact_subset_candidate"
            inventory_proof = _validate_time_shard_inventory(
                inventory_connection,
                manifest,
                label=spec.shard_id,
            )
    finally:
        inventory_connection.close()
    inventory_receipt = (
        None
        if spec.inventory.receipt is None
        else _load_bound_receipt(
            spec.inventory.receipt,
            where=f"{spec.shard_id} inventory receipt",
        )
    )
    store_receipt_declared = _load_bound_receipt(
        spec.store.receipt,
        where=f"{spec.shard_id} store receipt",
    )
    fetch_receipt = _load_bound_receipt(
        spec.state.receipt,
        where=f"{spec.shard_id} fetch receipt",
    )
    if inventory_role == "anchor_candidate":
        assert inventory_receipt is not None
        if (
            inventory_receipt.get("schema") != INVENTORY_RECEIPT_SCHEMA
            or inventory_receipt.get("database") != spec.original_inventory
        ):
            raise MergeError(
                f"{spec.shard_id} inventory receipt does not bind its original path"
            )
        try:
            computed_inventory_receipt = InventoryDB(
                spec.inventory.path
            ).completion_receipt()
        except (InventoryError, OSError, ValueError, sqlite3.Error) as exc:
            raise MergeError(
                f"{spec.shard_id} anchor completion proof is invalid"
            ) from exc
        if _inventory_logical_projection(
            computed_inventory_receipt
        ) != _inventory_logical_projection(inventory_receipt):
            raise MergeError(
                f"{spec.shard_id} anchor completion receipt differs from SQLite"
            )
        inventory_proof["completion_receipt_sha256"] = (
            spec.inventory.receipt.sha256
        )
    if fetch_receipt.get("schema") != FETCH_RECEIPT_SCHEMA:
        raise MergeError(f"{spec.shard_id} fetch receipt schema is not v3")
    if fetch_receipt.get("inventory_path") != spec.original_inventory:
        raise MergeError(
            f"{spec.shard_id} fetch receipt inventory path is not original"
        )

    source_index = spec.store.path / "index.sqlite3"
    _require_frozen_sqlite(source_index, label=f"{spec.shard_id} store")
    source_index_connection = sqlite3.connect(
        f"{source_index.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    source_index_connection.row_factory = sqlite3.Row
    try:
        oversized_content = source_index_connection.execute(
            """
            SELECT sha256 FROM contents
            WHERE raw_size > ? OR compressed_size > ?
            LIMIT 1
            """,
            (
                manifest.limits.max_content_bytes,
                manifest.limits.uncompressed_bytes_per_batch,
            ),
        ).fetchone()
        if oversized_content is not None:
            raise MergeError(
                f"{spec.shard_id} store content exceeds its bound: "
                f"{oversized_content['sha256']}"
            )
        oversized_provenance = source_index_connection.execute(
            """
            SELECT repo,run_attempt,job,step,chunk_ordinal
            FROM occurrences
            WHERE provenance_raw_size > ?
               OR length(provenance_zlib) > ?
            LIMIT 1
            """,
            (
                manifest.limits.max_provenance_bytes,
                manifest.limits.max_provenance_bytes,
            ),
        ).fetchone()
        if oversized_provenance is not None:
            raise MergeError(
                f"{spec.shard_id} store provenance exceeds its bound: "
                f"{tuple(oversized_provenance)}"
            )
    finally:
        source_index_connection.close()

    with FrozenStore(spec.store.path, spec.store.receipt.path) as store:
        if store.receipt != store_receipt_declared:
            raise MergeError(f"{spec.shard_id} store receipt changed while loaded")
        artifact_digest = _store_artifact_set_digest(store._initial_snapshot)
        if artifact_digest != spec.store.artifact_set_sha256:
            raise MergeError(
                f"{spec.shard_id} store artifact-set hash differs from manifest"
            )
        if fetch_receipt.get("content_store_receipt") != store.receipt:
            raise MergeError(
                f"{spec.shard_id} fetch receipt does not embed its store receipt"
            )
        if (
            fetch_receipt.get("target_exact_unique_payload_tokens")
            != store.receipt.get("target_exact_unique_payload_tokens")
        ):
            raise MergeError(
                f"{spec.shard_id} fetch receipt target differs from its store"
            )
        mismatched_tokenizer = store.connection.execute(
            """
            SELECT sha256 FROM contents
            WHERE tokenizer_fingerprint IS NULL
               OR tokenizer_fingerprint != ?
            LIMIT 1
            """,
            (tokenizer.fingerprint,),
        ).fetchone()
        if mismatched_tokenizer is not None:
            raise MergeError(
                f"{spec.shard_id} store content tokenizer differs from fetch state"
            )
        if (
            fetch_receipt.get("tokenizer_contract") != tokenizer.contract
            or fetch_receipt.get("tokenizer_fingerprint") != tokenizer.fingerprint
        ):
            raise MergeError(f"{spec.shard_id} fetch receipt tokenizer differs")
        state_precheck = sqlite3.connect(
            f"{spec.state.path.as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        state_precheck.row_factory = sqlite3.Row
        try:
            _state_blob_limits(state_precheck, manifest.limits)
            _reject_unsafe_attempt_states(state_precheck)
            state_settings = {
                str(row["key"]): str(row["value"])
                for row in state_precheck.execute(
                    "SELECT key,value FROM settings"
                )
            }
            _validate_binding_history(
                state_precheck,
                current_bindings=state_settings,
            )
        finally:
            state_precheck.close()
        binding_store = SimpleNamespace(
            root=Path(spec.original_store).resolve(),
            receipt=store.receipt,
        )
        with FrozenFetchState(
            spec.state.path,
            tokenizer=tokenizer,
            store=cast(Any, binding_store),
        ) as state:
            if state.settings["inventory_path"] != spec.original_inventory:
                raise MergeError(
                    f"{spec.shard_id} state inventory binding differs from manifest"
                )
            if state.settings["content_store_path"] != spec.original_store:
                raise MergeError(
                    f"{spec.shard_id} state store binding differs from manifest"
                )
            _state_blob_limits(state.connection, manifest.limits)
            _reject_unsafe_attempt_states(state.connection)
            if fetch_receipt.get("fetch_state") != state.summary:
                raise MergeError(
                    f"{spec.shard_id} fetch receipt summary differs from state"
                )
            expected_binding = state.receipt_binding()
            declared_binding = _require_mapping(
                fetch_receipt.get("frozen_fetch_state"),
                where=f"{spec.shard_id} frozen_fetch_state",
            )
            declared_artifact = _require_mapping(
                declared_binding.get("artifact"),
                where=f"{spec.shard_id} frozen_fetch_state.artifact",
            )
            expected_artifact = cast(
                Mapping[str, Any],
                expected_binding["artifact"],
            )
            if (
                set(declared_binding) != set(expected_binding)
                or set(declared_artifact) != set(expected_artifact)
                or isinstance(declared_artifact.get("mtime_ns"), bool)
                or not isinstance(declared_artifact.get("mtime_ns"), int)
                or int(declared_artifact["mtime_ns"]) < 0
                or isinstance(declared_artifact.get("inode"), bool)
                or not isinstance(declared_artifact.get("inode"), int)
                or int(declared_artifact["inode"]) < 0
            ):
                raise MergeError(
                    f"{spec.shard_id} frozen fetch-state receipt shape differs"
                )
            declared_semantics = {
                key: value
                for key, value in declared_binding.items()
                if key != "artifact"
            }
            expected_semantics = {
                key: value
                for key, value in expected_binding.items()
                if key != "artifact"
            }
            if (
                declared_semantics != expected_semantics
                or declared_artifact.get("path") != spec.original_state
                or declared_artifact.get("byte_size")
                != expected_artifact["byte_size"]
                or declared_artifact.get("sha256")
                != expected_artifact["sha256"]
            ):
                raise MergeError(
                    f"{spec.shard_id} frozen fetch-state receipt binding differs"
                )
            join_count = _verify_cas_fetch_join(
                store,
                state,
                scratch_path=scratch_directory
                / f"{spec.shard_id}-source-coverage.sqlite3",
                limits=manifest.limits,
            )
            source_inventory = sqlite3.connect(
                f"{spec.inventory.path.as_uri()}?mode=ro&immutable=1",
                uri=True,
            )
            source_inventory.row_factory = sqlite3.Row
            try:
                (
                    inventory_join_count,
                    inventory_join_sha256,
                ) = _verify_state_inventory_join(
                    source_inventory,
                    state.connection,
                    label=spec.shard_id,
                    max_blob_bytes=manifest.limits.max_state_blob_bytes,
                )
            finally:
                source_inventory.close()
            state.require_unchanged()
            state_counts = {
                "attempts": int(
                    state.connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                ),
                "members": int(
                    state.connection.execute("SELECT COUNT(*) FROM members").fetchone()[0]
                ),
                "requests": int(
                    state.connection.execute(
                        "SELECT COUNT(*) FROM request_ledger"
                    ).fetchone()[0]
                ),
                "bindings": int(
                    state.connection.execute(
                        "SELECT COUNT(*) FROM binding_upgrades"
                    ).fetchone()[0]
                ),
                "chunks": int(
                    state.connection.execute(
                        "SELECT COALESCE(SUM(chunk_count),0) FROM members"
                    ).fetchone()[0]
                ),
                "occurrence_tokens": int(
                    state.connection.execute(
                        "SELECT COALESCE(SUM(occurrence_tokens),0) FROM members"
                    ).fetchone()[0]
                ),
                "joined_occurrences": join_count,
                "inventory_joined_attempts": inventory_join_count,
            }
            if inventory_join_count != state_counts["attempts"]:
                raise MergeError(
                    f"{spec.shard_id} inventory join did not cover every attempt"
                )
            inventory_proof["source_state_join_sha256"] = (
                inventory_join_sha256
            )
            state_binding = state.receipt_binding()
        store.require_unchanged()
        counters = cast(Mapping[str, Any], store.receipt["counters"])
        store_counts = _integer_counts(
            counters,
            (
                "raw_occurrence_bytes",
                "unique_bytes",
                "duplicate_bytes",
                "unique_content_count",
                "occurrence_count",
                "tokenized_unique_content_count",
                "unique_token_sequence_count",
                "exact_unique_payload_tokens",
            ),
        )
        if join_count != store_counts["occurrence_count"]:
            raise MergeError(
                f"{spec.shard_id} CAS/fetch join did not cover every occurrence"
            )
        store_files = store._initial_snapshot
    if _snapshot_receipts(spec) != receipt_files:
        raise MergeError(f"{spec.shard_id} source receipts changed during preflight")
    return SourceAudit(
        spec=spec,
        store_receipt=store_receipt_declared,
        fetch_receipt=fetch_receipt,
        inventory_receipt=inventory_receipt,
        inventory_role=inventory_role,
        inventory_proof=inventory_proof,
        store_files=store_files,
        state_file=state_file,
        inventory_file=inventory_file,
        receipt_files=receipt_files,
        state_binding=state_binding,
        store_counts=store_counts,
        state_counts=state_counts,
    )


def _inventory_logical_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in receipt.items()
        if key not in {"completed_at", "database"}
    }


def _expected_inventory_schema_sha256() -> str:
    connection = sqlite3.connect(":memory:")
    try:
        connection.row_factory = sqlite3.Row
        connection.executescript(INVENTORY_SQL)
        return _sqlite_schema_sha256(connection)
    finally:
        connection.close()


def _sqlite_values_byte_identical(left: object, right: object) -> bool:
    if isinstance(left, memoryview):
        left = left.tobytes()
    if isinstance(right, memoryview):
        right = right.tobytes()
    return type(left) is type(right) and left == right


def _prove_exact_subset(
    subset: SourceAudit,
    anchor: SourceAudit,
) -> dict[str, Any]:
    if (
        subset.inventory_proof["source_inventory_path"]
        != anchor.spec.original_inventory
    ):
        raise MergeError(
            f"{subset.spec.shard_id} time-shard source_inventory_path "
            "does not bind the selected anchor original path"
        )
    assert anchor.inventory_receipt is not None
    anchor_interval = _require_mapping(
        anchor.inventory_receipt.get("interval"),
        where="anchor completion receipt interval",
    )
    _require_exact_keys(
        anchor_interval,
        {"start", "end", "semantics"},
        where="anchor completion receipt interval",
    )
    if anchor_interval["semantics"] != "[start,end)":
        raise MergeError("anchor completion receipt interval semantics differ")
    _anchor_start, anchor_lower = _canonical_utc(
        anchor_interval["start"],
        where="anchor completion receipt interval.start",
    )
    _anchor_end, anchor_upper = _canonical_utc(
        anchor_interval["end"],
        where="anchor completion receipt interval.end",
    )
    _subset_start, subset_lower = _canonical_utc(
        subset.inventory_proof["created_at_gte"],
        where=f"{subset.spec.shard_id} time-shard lower bound",
    )
    _subset_end, subset_upper = _canonical_utc(
        subset.inventory_proof["created_at_lt"],
        where=f"{subset.spec.shard_id} time-shard upper bound",
    )
    if subset_lower < anchor_lower or subset_upper > anchor_upper:
        raise MergeError(
            f"{subset.spec.shard_id} time-shard bounds escape the anchor interval"
        )
    subset_connection = sqlite3.connect(
        f"{subset.spec.inventory.path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    anchor_connection = sqlite3.connect(
        f"{anchor.spec.inventory.path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    subset_connection.row_factory = sqlite3.Row
    anchor_connection.row_factory = sqlite3.Row
    matched_count = 0
    match_digest = _record_digest(
        "cppmega-ci-time-shard-anchor-match-v1"
    )
    columns = ",".join(_INVENTORY_RUN_COLUMNS)
    try:
        for subset_row in subset_connection.execute(
            f"""
            SELECT {columns}
            FROM runs ORDER BY repo_key,run_id,run_attempt
            """
        ):
            anchor_row = anchor_connection.execute(
                f"""
                SELECT {columns}
                FROM runs
                WHERE repo_key=? AND run_id=? AND run_attempt=?
                """,
                (
                    subset_row["repo_key"],
                    subset_row["run_id"],
                    subset_row["run_attempt"],
                ),
            ).fetchone()
            key = (
                f"{subset_row['repo_key']}/{subset_row['run_id']}/"
                f"{subset_row['run_attempt']}"
            )
            if anchor_row is None:
                raise MergeError(
                    f"{subset.spec.shard_id} time-shard run is absent "
                    f"from the anchor: {key}"
                )
            for column in _INVENTORY_RUN_COLUMNS:
                if not _sqlite_values_byte_identical(
                    subset_row[column],
                    anchor_row[column],
                ):
                    raise MergeError(
                        f"{subset.spec.shard_id} time-shard column {column} "
                        f"differs from the anchor: {key}"
                    )
            _update_record_digest(
                match_digest,
                _inventory_row_hash_record(subset_row),
            )
            matched_count += 1
    finally:
        subset_connection.close()
        anchor_connection.close()
    if matched_count != subset.inventory_proof["run_count"]:
        raise MergeError(
            f"{subset.spec.shard_id} time-shard match count changed during proof"
        )
    return {
        **subset.inventory_proof,
        "anchor_source_id": anchor.spec.shard_id,
        "anchor_original_path": anchor.spec.original_inventory,
        "matched_run_count": matched_count,
        "anchor_match_logical_sha256": match_digest.hexdigest(),
    }


def _validate_cross_source_contracts(
    audits: Sequence[SourceAudit],
) -> InventoryContract:
    anchor_candidates = [
        audit for audit in audits if audit.inventory_role == "anchor_candidate"
    ]
    if not anchor_candidates:
        raise MergeError("exactly one completed full inventory anchor is required")
    subsets = [
        audit
        for audit in audits
        if audit.inventory_role == "exact_subset_candidate"
    ]
    if subsets:
        subset_source_paths = {
            str(audit.inventory_proof["source_inventory_path"])
            for audit in subsets
        }
        if len(subset_source_paths) != 1:
            raise MergeError(
                "time-shard inventories disagree on their full anchor path"
            )
        anchor_path = next(iter(subset_source_paths))
        matching_candidates = [
            audit
            for audit in anchor_candidates
            if audit.spec.original_inventory == anchor_path
        ]
        if not matching_candidates:
            raise MergeError(
                "no completed anchor binds the time-shard source_inventory_path"
            )
        anchor = matching_candidates[0]
    else:
        anchor = anchor_candidates[0]
    assert anchor.inventory_receipt is not None
    inventory_sha = anchor.inventory_file.sha256
    inventory_projection = _inventory_logical_projection(anchor.inventory_receipt)
    for candidate in anchor_candidates:
        assert candidate.inventory_receipt is not None
        if (
            candidate.inventory_file.sha256 != inventory_sha
            or _inventory_logical_projection(candidate.inventory_receipt)
            != inventory_projection
        ):
            raise MergeError(
                "multiple distinct completed inventory anchors are ambiguous"
            )

    first = audits[0]
    semantic_settings = {
        key: value
        for key, value in cast(
            Mapping[str, Any],
            first.state_binding["settings"],
        ).items()
        if key
        not in {
            "inventory_path",
            "content_store_path",
            "content_store_script_sha256",
            "created_at",
        }
    }
    for audit in audits[1:]:
        candidate_settings = {
            key: value
            for key, value in cast(
                Mapping[str, Any],
                audit.state_binding["settings"],
            ).items()
            if key
            not in {
                "inventory_path",
                "content_store_path",
                "content_store_script_sha256",
                "created_at",
            }
        }
        if candidate_settings != semantic_settings:
            raise MergeError("source fetch-state semantic settings conflict")

    subset_proofs = {
        audit.spec.shard_id: _prove_exact_subset(audit, anchor)
        for audit in subsets
    }
    source_bindings: list[dict[str, Any]] = []
    for audit in audits:
        if audit.inventory_role == "exact_subset_candidate":
            role = "byte_identical_row_subset"
            proof = subset_proofs[audit.spec.shard_id]
        elif audit is anchor:
            role = "anchor"
            proof = audit.inventory_proof
        else:
            role = "byte_identical_anchor_alias"
            proof = {
                **audit.inventory_proof,
                "anchor_source_id": anchor.spec.shard_id,
                "anchor_original_path": anchor.spec.original_inventory,
            }
        source_bindings.append(
            {
                "source_id": audit.spec.shard_id,
                "role": role,
                "original_inventory_path": audit.spec.original_inventory,
                "staged_inventory_path": str(audit.spec.inventory.path),
                "manifest_inventory_sha256": audit.spec.inventory.sha256,
                "artifact_sha256": audit.inventory_file.sha256,
                "artifact_byte_size": audit.inventory_file.size,
                "receipt_sha256": (
                    None
                    if audit.spec.inventory.receipt is None
                    else audit.spec.inventory.receipt.sha256
                ),
                "proof": proof,
                "source_state_joined_attempts": audit.state_counts[
                    "inventory_joined_attempts"
                ],
            }
        )
    binding = {
        "schema": INVENTORY_BINDING_SCHEMA,
        "policy": "completed-anchor-with-time-bounded-row-subsets-v1",
        "coverage_semantics": "subset_only_no_range_completeness",
        "anchor": {
            "source_id": anchor.spec.shard_id,
            "original_inventory_path": anchor.spec.original_inventory,
            "staged_inventory_path": str(anchor.spec.inventory.path),
            "artifact_sha256": anchor.inventory_file.sha256,
            "artifact_byte_size": anchor.inventory_file.size,
            "completion_receipt_sha256": (
                anchor.spec.inventory.receipt.sha256
            ),
            "completion_receipt_logical_sha256": _sha256_bytes(
                _canonical_json_bytes(inventory_projection)
            ),
            "sqlite_schema_sha256": anchor.inventory_proof[
                "sqlite_schema_sha256"
            ],
            "interval": anchor.inventory_receipt["interval"],
            "db_logical_sha256": anchor.inventory_receipt[
                "db_logical_sha256"
            ],
            "run_count": anchor.inventory_proof["run_count"],
            "run_rows_logical_sha256": anchor.inventory_proof[
                "run_rows_logical_sha256"
            ],
        },
        "source_count": len(audits),
        "time_subset_count": len(subsets),
        "time_subset_run_count": sum(
            int(proof["matched_run_count"])
            for proof in subset_proofs.values()
        ),
        "sources": source_bindings,
    }
    binding["binding_sha256"] = _sha256_bytes(_canonical_json_bytes(binding))
    return InventoryContract(anchor=anchor, binding=binding)


def _validate_output_geometry(manifest: Manifest) -> Path:
    destination = manifest.destination
    partial = destination.with_name(f".{destination.name}.partial")
    lock_path = destination.with_name(f".{destination.name}.merge.lock")
    staged_paths = [manifest.path, manifest.tokenizer_path]
    for shard in manifest.shards:
        staged_paths.append(shard.inventory.path)
        if shard.inventory.receipt is not None:
            staged_paths.append(shard.inventory.receipt.path)
        staged_paths.extend(
            (
                shard.store.path,
                shard.store.receipt.path,
                shard.state.path,
                shard.state.receipt.path,
            )
        )
    for path in staged_paths:
        if (
            path == destination
            or destination in path.parents
            or path == partial
            or partial in path.parents
            or path == lock_path
        ):
            raise MergeError("source artifacts cannot be inside the output bundle")
    for shard in manifest.shards:
        if (
            shard.store.path == destination
            or shard.store.path in destination.parents
            or shard.store.path == partial
            or shard.store.path in partial.parents
            or shard.store.path in lock_path.parents
        ):
            raise MergeError("output bundle cannot be nested in a source store")
    return partial


def _ensure_safe_partial(manifest: Manifest) -> Path:
    destination = manifest.destination
    partial = _validate_output_geometry(manifest)
    if destination.exists() or destination.is_symlink():
        raise MergeError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        if partial.is_symlink() or not partial.is_dir():
            raise MergeError(f"partial bundle is unsafe: {partial}")
        for path in partial.rglob("*"):
            if path.is_symlink():
                raise MergeError(f"partial bundle contains a symlink: {path}")
        _cleanup_known_partial_temps(partial)
    else:
        partial.mkdir()
        _fsync_directory(partial.parent)
    return partial


def _acquire_merge_lock(destination: Path) -> int:
    path = destination.with_name(f".{destination.name}.merge.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise MergeError(f"cannot open exclusive merge lock: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise MergeError(f"exclusive merge lock is not a regular file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MergeError(
                f"another shard union owns the destination lock: {destination}"
            ) from exc
        _fsync_directory(path.parent)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _release_merge_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _cleanup_known_partial_temps(partial: Path) -> None:
    roots = {
        partial: {
            _INVENTORY_NAME,
            _INVENTORY_RECEIPT_NAME,
            _STORE_RECEIPT_NAME,
            _FETCH_STATE_NAME,
            _FETCH_RECEIPT_NAME,
            _JOURNAL_NAME,
            _MERGE_RECEIPT_NAME,
        },
        partial / _LEDGER_DIRECTORY: set(_LEDGER_FILENAMES),
    }
    for directory, names in roots.items():
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise MergeError(f"partial output directory is unsafe: {directory}")
        patterns = tuple(
            re.compile(
                rf"\.{re.escape(name)}\.tmp-[0-9]+"
                r"(?:-(?:journal|shm|wal))?"
            )
            for name in sorted(names)
        )
        changed = False
        for path in directory.iterdir():
            if not any(pattern.fullmatch(path.name) for pattern in patterns):
                continue
            if path.is_symlink() or not path.is_file():
                raise MergeError(f"partial temporary artifact is unsafe: {path}")
            path.unlink()
            changed = True
        if changed:
            _fsync_directory(directory)


def _install_file_no_replace(temporary: Path, destination: Path) -> bool:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        installed = False
    else:
        installed = True
    finally:
        if temporary.exists():
            temporary.unlink()
    _fsync_directory(destination.parent)
    return installed


def _initialize_journal_file(
    path: Path,
    manifest: Manifest,
    *,
    merge_script_sha256: str,
    fault_inject_sigkill_after: str | None,
) -> None:
    connection = sqlite3.connect(path, isolation_level=None, timeout=60.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        _fault_inject_sigkill(
            fault_inject_sigkill_after,
            "journal-file-created",
        )
        connection.executescript(_JOURNAL_SQL)
        _fault_inject_sigkill(
            fault_inject_sigkill_after,
            "journal-schema-created",
        )
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.executemany(
                "INSERT INTO settings(key,value) VALUES (?,?)",
                sorted(
                    {
                        "schema": JOURNAL_SCHEMA,
                        "manifest_sha256": manifest.sha256,
                        "merge_script_sha256": merge_script_sha256,
                        "destination": str(manifest.destination),
                        "phase": "store",
                        "created_at": _utc_now(),
                    }.items()
                ),
            )
            connection.executemany(
                "INSERT INTO store_progress(shard_id) VALUES (?)",
                ((shard.shard_id,) for shard in manifest.shards),
            )
            connection.executemany(
                """
                INSERT INTO state_progress(shard_id,table_name)
                VALUES (?,?)
                """,
                (
                    (shard.shard_id, table)
                    for shard in manifest.shards
                    for table in (
                        "attempts",
                        "members",
                        "request_ledger",
                        "binding_upgrades",
                    )
                ),
            )
            _fault_inject_sigkill(
                fault_inject_sigkill_after,
                "journal-before-settings-commit",
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        if _sqlite_schema_sha256(connection) != _expected_journal_schema_sha256():
            raise MergeError("new merge journal schema is not canonical v2")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise MergeError("new merge journal failed integrity_check")
    finally:
        connection.close()
    _require_frozen_sqlite(path, label="new merge journal")
    _fsync_file(path)


def _open_journal(
    partial: Path,
    manifest: Manifest,
    *,
    merge_script_sha256: str,
    fault_inject_sigkill_after: str | None,
) -> sqlite3.Connection:
    path = partial / _JOURNAL_NAME
    if path.is_symlink():
        raise MergeError("merge journal cannot be a symlink")
    if not path.exists():
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        if temporary.exists() or temporary.is_symlink():
            raise MergeError("merge journal temporary path is already occupied")
        try:
            _initialize_journal_file(
                temporary,
                manifest,
                merge_script_sha256=merge_script_sha256,
                fault_inject_sigkill_after=fault_inject_sigkill_after,
            )
            _install_file_no_replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    if not path.is_file():
        raise MergeError("merge journal is missing or unsafe")
    connection = sqlite3.connect(path, isolation_level=None, timeout=60.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    if _sqlite_schema_sha256(connection) != _expected_journal_schema_sha256():
        connection.close()
        raise MergeError("existing merge journal schema is not canonical v2")
    expected = {
        "schema": JOURNAL_SCHEMA,
        "manifest_sha256": manifest.sha256,
        "merge_script_sha256": merge_script_sha256,
        "destination": str(manifest.destination),
    }
    current = {
        str(row["key"]): str(row["value"])
        for row in connection.execute("SELECT key,value FROM settings")
    }
    if current:
        for key, value in expected.items():
            if current.get(key) != value:
                connection.close()
                raise MergeError(f"merge journal binding differs for {key}")
        if current.get("phase") not in {
            "store",
            "inventory",
            "state",
            "verify",
            "ready",
        }:
            connection.close()
            raise MergeError("merge journal phase is invalid")
        expected_setting_keys = {
            *expected,
            "phase",
            "created_at",
        }
        if current["phase"] == "ready":
            expected_setting_keys.add("completed_at")
        if set(current) != expected_setting_keys:
            connection.close()
            raise MergeError("merge journal settings have an unsupported shape")
    else:
        connection.close()
        raise MergeError("existing merge journal has no durable binding")
    expected_shards = {shard.shard_id for shard in manifest.shards}
    actual_store_shards = {
        str(row[0])
        for row in connection.execute("SELECT shard_id FROM store_progress")
    }
    expected_state_progress = {
        (shard_id, table)
        for shard_id in expected_shards
        for table in (
            "attempts",
            "members",
            "request_ledger",
            "binding_upgrades",
        )
    }
    actual_state_progress = {
        (str(row[0]), str(row[1]))
        for row in connection.execute(
            "SELECT shard_id,table_name FROM state_progress"
        )
    }
    if (
        actual_store_shards != expected_shards
        or actual_state_progress != expected_state_progress
    ):
        connection.close()
        raise MergeError("merge journal progress rows differ from the manifest")
    for table in (
        "attempt_map",
        "member_map",
        "request_id_map",
        "binding_id_map",
    ):
        unknown = connection.execute(
            f"""
            SELECT shard_id FROM {table}
            WHERE shard_id NOT IN (
              SELECT shard_id FROM store_progress
            )
            LIMIT 1
            """
        ).fetchone()
        if unknown is not None:
            connection.close()
            raise MergeError(f"merge journal {table} contains an unknown shard")
    return connection


def _expected_journal_schema_sha256() -> str:
    connection = sqlite3.connect(":memory:")
    try:
        connection.row_factory = sqlite3.Row
        connection.executescript(_JOURNAL_SQL)
        return _sqlite_schema_sha256(connection)
    finally:
        connection.close()


def _journal_phase(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT value FROM settings WHERE key='phase'"
    ).fetchone()
    if row is None:
        raise MergeError("merge journal phase is missing")
    return str(row[0])


def _set_journal_phase(connection: sqlite3.Connection, phase: str) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            "UPDATE settings SET value=? WHERE key='phase'",
            (phase,),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _decode_cursor(value: object, expected_items: int) -> tuple[Any, ...] | None:
    if value is None:
        return None
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise MergeError("merge journal cursor is invalid JSON") from exc
    if not isinstance(decoded, list) or len(decoded) != expected_items:
        raise MergeError("merge journal cursor shape is invalid")
    return tuple(decoded)


def _source_occurrence_batch(
    store: FrozenStore,
    cursor: tuple[Any, ...] | None,
    limits: Limits,
) -> list[sqlite3.Row]:
    where = ""
    parameters: list[Any] = []
    if cursor is not None:
        where = (
            "WHERE (o.repo,o.run_attempt,o.job,o.step,o.chunk_ordinal) "
            "> (?,?,?,?,?)"
        )
        parameters.extend(cursor)
    parameters.append(limits.occurrences_per_batch)
    rows = store.connection.execute(
        f"""
        SELECT o.repo,o.run_attempt,o.job,o.step,o.chunk_ordinal,
               o.content_sha256,o.provenance_sha256,
               o.provenance_raw_size,o.provenance_zlib,
               c.raw_size,c.token_count,c.tokenizer_fingerprint,
               c.token_sequence_sha256
        FROM occurrences AS o
        JOIN contents AS c ON c.sha256=o.content_sha256
        {where}
        ORDER BY o.repo,o.run_attempt,o.job,o.step,o.chunk_ordinal
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    selected: list[sqlite3.Row] = []
    total = 0
    for row in rows:
        content_bytes = int(row["raw_size"])
        provenance_bytes = int(row["provenance_raw_size"])
        if content_bytes > limits.max_content_bytes:
            raise MergeError(f"content exceeds max_content_bytes: {row['content_sha256']}")
        if provenance_bytes > limits.max_provenance_bytes:
            raise MergeError("occurrence provenance exceeds max_provenance_bytes")
        increment = content_bytes + provenance_bytes
        if increment > limits.uncompressed_bytes_per_batch:
            raise MergeError("one occurrence exceeds uncompressed batch byte limit")
        if selected and total + increment > limits.uncompressed_bytes_per_batch:
            break
        selected.append(row)
        total += increment
    return selected


def _merge_store(
    partial: Path,
    manifest: Manifest,
    journal: sqlite3.Connection,
    *,
    batch_budget: list[int | None],
) -> dict[str, Any] | None:
    destination_root = partial / _STORE_DIRECTORY
    with CIContentStore(destination_root) as destination:
        for spec in manifest.shards:
            progress = journal.execute(
                "SELECT * FROM store_progress WHERE shard_id=?",
                (spec.shard_id,),
            ).fetchone()
            if progress is None:
                raise MergeError(f"store progress is missing for {spec.shard_id}")
            if int(progress["done"]):
                continue
            cursor = _decode_cursor(progress["cursor_json"], 5)
            with FrozenStore(spec.store.path, spec.store.receipt.path) as source:
                while True:
                    rows = _source_occurrence_batch(
                        source,
                        cursor,
                        manifest.limits,
                    )
                    if not rows:
                        journal.execute("BEGIN IMMEDIATE")
                        try:
                            journal.execute(
                                """
                                UPDATE store_progress SET done=1
                                WHERE shard_id=?
                                """,
                                (spec.shard_id,),
                            )
                            journal.commit()
                        except BaseException:
                            journal.rollback()
                            raise
                        break
                    content_cache: dict[str, bytes] = {}
                    records: list[dict[str, object]] = []
                    for row in rows:
                        occurrence = source._occurrence_record(row)
                        content_sha256 = occurrence.content_sha256
                        content = content_cache.get(content_sha256)
                        if content is None:
                            content = source.read_content(
                                source.get_content_record(content_sha256)
                            )
                            content_cache[content_sha256] = content
                        records.append(
                            {
                                "content": content,
                                "provenance": occurrence.provenance,
                                "occurrence_key": occurrence.key,
                                "token_count": int(row["token_count"]),
                                "tokenizer_fingerprint": str(
                                    row["tokenizer_fingerprint"]
                                ),
                                "token_sequence_sha256": str(
                                    row["token_sequence_sha256"]
                                ),
                            }
                        )
                    try:
                        destination.add_chunks(records)
                    except ContentStoreError as exc:
                        raise MergeError(
                            f"CAS conflict while merging shard {spec.shard_id}: {exc}"
                        ) from exc
                    last = rows[-1]
                    cursor = (
                        str(last["repo"]),
                        str(last["run_attempt"]),
                        str(last["job"]),
                        str(last["step"]),
                        int(last["chunk_ordinal"]),
                    )
                    journal.execute("BEGIN IMMEDIATE")
                    try:
                        journal.execute(
                            """
                            UPDATE store_progress
                            SET cursor_json=?,
                                processed_rows=processed_rows+?,
                                batches=batches+1
                            WHERE shard_id=?
                            """,
                            (
                                _canonical_json(list(cursor)),
                                len(rows),
                                spec.shard_id,
                            ),
                        )
                        journal.commit()
                    except BaseException:
                        journal.rollback()
                        raise
                    if batch_budget[0] is not None:
                        batch_budget[0] -= 1
                        if batch_budget[0] == 0:
                            source.require_unchanged()
                            return None
                source.require_unchanged()
        try:
            receipt = destination.completion_receipt(
                target_unique_tokens=manifest.target_unique_tokens
            )
        except ThresholdNotMetError as exc:
            raise MergeError(
                "global exact-deduplicated union is below the requested token target"
            ) from exc
    _write_json(partial / _STORE_RECEIPT_NAME, receipt)
    _set_journal_phase(journal, "inventory")
    return receipt


def _load_destination_store_receipt(partial: Path) -> dict[str, Any]:
    value, _raw = _load_json(
        partial / _STORE_RECEIPT_NAME,
        where="destination store receipt",
    )
    return value


def _copy_and_validate_inventory(
    partial: Path,
    manifest: Manifest,
    inventory_contract: InventoryContract,
    journal: sqlite3.Connection,
) -> None:
    anchor = inventory_contract.anchor
    destination = partial / _INVENTORY_NAME
    if not destination.exists():
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            with anchor.spec.inventory.path.open("rb") as source:
                with temporary.open("xb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
            os.replace(temporary, destination)
            _fsync_directory(partial)
        finally:
            if temporary.exists():
                temporary.unlink()
    copied = _snapshot_file(destination, label="destination inventory")
    if copied.sha256 != anchor.inventory_file.sha256:
        raise MergeError("destination inventory copy differs from frozen source")
    try:
        inventory_receipt = InventoryDB(destination).completion_receipt()
    except (InventoryError, OSError, ValueError, sqlite3.Error) as exc:
        raise MergeError("copied union inventory failed full validation") from exc
    inventory_receipt["database"] = str(
        manifest.destination / _INVENTORY_NAME
    )
    assert anchor.inventory_receipt is not None
    expected = _inventory_logical_projection(anchor.inventory_receipt)
    if _inventory_logical_projection(inventory_receipt) != expected:
        raise MergeError("validated union inventory differs from source receipt")
    _require_frozen_sqlite(destination, label="destination inventory")
    _write_json(partial / _INVENTORY_RECEIPT_NAME, inventory_receipt)
    _set_journal_phase(journal, "state")


def _destination_state_settings(
    manifest: Manifest,
    audits: Sequence[SourceAudit],
    store_receipt: Mapping[str, Any],
) -> dict[str, str]:
    source_settings = [
        cast(Mapping[str, Any], audit.state_binding["settings"])
        for audit in audits
    ]
    created_at = min(str(settings["created_at"]) for settings in source_settings)
    first = source_settings[0]
    return {
        "schema": FETCH_STATE_SCHEMA,
        "inventory_path": str(manifest.destination / _INVENTORY_NAME),
        "content_store_path": str(manifest.destination / _STORE_DIRECTORY),
        "tokenizer_contract": str(first["tokenizer_contract"]),
        "tokenizer_fingerprint": str(first["tokenizer_fingerprint"]),
        "fetcher_script_sha256": str(first["fetcher_script_sha256"]),
        "parser_script_sha256": str(first["parser_script_sha256"]),
        "content_store_script_sha256": str(store_receipt["script_sha256"]),
        "chunk_semantics": str(first["chunk_semantics"]),
        "created_at": created_at,
    }


def _initialize_destination_state(
    path: Path,
    settings: Mapping[str, str],
    *,
    fault_inject_sigkill_after: str | None = None,
) -> None:
    if path.is_symlink():
        raise MergeError("partial destination fetch state cannot be a symlink")
    if path.exists():
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            if _sqlite_schema_sha256(connection) != _expected_fetch_state_schema_sha256():
                raise MergeError("partial destination fetch state has wrong schema")
            current = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key,value FROM settings")
            }
            if current != dict(settings):
                raise MergeError("partial destination fetch-state settings conflict")
        finally:
            connection.close()
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise MergeError("fetch-state temporary path is already occupied")
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        _fault_inject_sigkill(
            fault_inject_sigkill_after,
            "state-file-created",
        )
        connection.executescript(FETCH_STATE_SQL_SCHEMA)
        _fault_inject_sigkill(
            fault_inject_sigkill_after,
            "state-schema-created",
        )
        connection.executemany(
            "INSERT INTO settings(key,value) VALUES (?,?)",
            sorted(settings.items()),
        )
        _fault_inject_sigkill(
            fault_inject_sigkill_after,
            "state-before-settings-commit",
        )
        connection.commit()
    finally:
        connection.close()
    try:
        _require_frozen_sqlite(temporary, label="new destination fetch state")
        _fsync_file(temporary)
        _install_file_no_replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if path.is_symlink() or not path.is_file():
        raise MergeError("destination fetch state was not installed safely")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        if _sqlite_schema_sha256(connection) != _expected_fetch_state_schema_sha256():
            raise MergeError("installed destination fetch state has wrong schema")
        current = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM settings")
        }
        if current != dict(settings):
            raise MergeError("installed destination fetch-state settings conflict")
    finally:
        connection.close()


def _normalized_sql_value(value: object) -> object:
    if isinstance(value, memoryview):
        return bytes(value)
    return value


def _row_values(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[object, ...]:
    return tuple(_normalized_sql_value(row[column]) for column in columns)


def _row_sha256(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    encoded: list[object] = []
    for column in columns:
        value = _normalized_sql_value(row[column])
        if isinstance(value, bytes):
            encoded.append(
                {
                    "column": column,
                    "byte_size": len(value),
                    "sha256": _sha256_bytes(value),
                }
            )
        else:
            encoded.append({"column": column, "value": value})
    return _sha256_bytes(_canonical_json_bytes(encoded))


def _is_zero_evidence_pending(row: Mapping[str, Any]) -> bool:
    nullable_evidence = (
        "archive_source",
        "archive_sha256",
        "archive_size",
        "jobs_sha256",
        "jobs_raw_size",
        "jobs_zlib",
        "terminal_http_status",
        "terminal_body_sha256",
        "error_class",
        "error_message",
    )
    return (
        row["status"] == "pending"
        and int(row["tries"]) == 0
        and int(row["member_count"]) == 0
        and int(row["chunk_count"]) == 0
        and int(row["occurrence_tokens"]) == 0
        and all(row[field] is None for field in nullable_evidence)
    )


def _mapping_row(row: sqlite3.Row) -> dict[str, Any]:
    return {key: _normalized_sql_value(row[key]) for key in row.keys()}


def _insert_named_row(
    connection: sqlite3.Connection,
    *,
    schema: str,
    table: str,
    columns: Sequence[str],
    values: Sequence[object],
) -> None:
    placeholders = ",".join("?" for _ in columns)
    connection.execute(
        f"INSERT INTO {schema}.{table}({','.join(columns)}) "
        f"VALUES ({placeholders})",
        values,
    )


def _merge_attempt_row(
    connection: sqlite3.Connection,
    shard_id: str,
    row: sqlite3.Row,
) -> None:
    incoming = _mapping_row(row)
    key = tuple(incoming[column] for column in _ATTEMPT_KEY)
    existing_row = connection.execute(
        """
        SELECT * FROM destination.attempts
        WHERE repo=? AND run_id=? AND attempt=?
        """,
        key,
    ).fetchone()
    outcome = "inserted"
    if existing_row is None:
        _insert_named_row(
            connection,
            schema="destination",
            table="attempts",
            columns=_ATTEMPT_COLUMNS,
            values=_row_values(incoming, _ATTEMPT_COLUMNS),
        )
    else:
        existing = _mapping_row(existing_row)
        if _row_values(existing, _ATTEMPT_COLUMNS) == _row_values(
            incoming, _ATTEMPT_COLUMNS
        ):
            outcome = "exact_overlap"
        elif (
            existing["status"] == "done"
            and _is_zero_evidence_pending(incoming)
            and _row_values(existing, _ATTEMPT_IMMUTABLE_EVIDENCE)
            == _row_values(incoming, _ATTEMPT_IMMUTABLE_EVIDENCE)
        ):
            outcome = "pending_shadowed_by_done"
        elif (
            incoming["status"] == "done"
            and _is_zero_evidence_pending(existing)
            and _row_values(existing, _ATTEMPT_IMMUTABLE_EVIDENCE)
            == _row_values(incoming, _ATTEMPT_IMMUTABLE_EVIDENCE)
        ):
            assignments = ",".join(
                f"{column}=?" for column in _ATTEMPT_COLUMNS
            )
            connection.execute(
                f"""
                UPDATE destination.attempts SET {assignments}
                WHERE repo=? AND run_id=? AND attempt=?
                """,
                (*_row_values(incoming, _ATTEMPT_COLUMNS), *key),
            )
            outcome = "done_replaced_zero_evidence_pending"
        else:
            raise MergeError(f"conflicting fetch attempt overlap: {key}")
    key_json = _canonical_json(list(key))
    connection.execute(
        """
        INSERT INTO attempt_map(
          shard_id,source_key_json,source_row_sha256,outcome
        ) VALUES (?,?,?,?)
        """,
        (
            shard_id,
            key_json,
            _row_sha256(incoming, _ATTEMPT_COLUMNS),
            outcome,
        ),
    )


def _merge_member_row(
    connection: sqlite3.Connection,
    shard_id: str,
    row: sqlite3.Row,
) -> None:
    incoming = _mapping_row(row)
    key = tuple(incoming[column] for column in _MEMBER_KEY)
    existing_row = connection.execute(
        """
        SELECT * FROM destination.members
        WHERE repo=? AND run_id=? AND attempt=? AND archive_member=?
        """,
        key,
    ).fetchone()
    outcome = "inserted"
    if existing_row is None:
        _insert_named_row(
            connection,
            schema="destination",
            table="members",
            columns=_MEMBER_COLUMNS,
            values=_row_values(incoming, _MEMBER_COLUMNS),
        )
    else:
        existing = _mapping_row(existing_row)
        if _row_values(existing, _MEMBER_COLUMNS) != _row_values(
            incoming, _MEMBER_COLUMNS
        ):
            raise MergeError(f"conflicting fetch member overlap: {key}")
        outcome = "exact_overlap"
    connection.execute(
        """
        INSERT INTO member_map(
          shard_id,source_key_json,source_row_sha256,outcome
        ) VALUES (?,?,?,?)
        """,
        (
            shard_id,
            _canonical_json(list(key)),
            _row_sha256(incoming, _MEMBER_COLUMNS),
            outcome,
        ),
    )


def _merge_request_row(
    connection: sqlite3.Connection,
    shard_id: str,
    row: sqlite3.Row,
) -> None:
    incoming = _mapping_row(row)
    destination_id = int(
        connection.execute(
            "SELECT COALESCE(MAX(id),0)+1 FROM destination.request_ledger"
        ).fetchone()[0]
    )
    values = list(_row_values(incoming, _REQUEST_COLUMNS))
    source_id = int(values[0])
    values[0] = destination_id
    _insert_named_row(
        connection,
        schema="destination",
        table="request_ledger",
        columns=_REQUEST_COLUMNS,
        values=values,
    )
    connection.execute(
        """
        INSERT INTO request_id_map(
          shard_id,source_id,destination_id,source_row_sha256
        ) VALUES (?,?,?,?)
        """,
        (
            shard_id,
            source_id,
            destination_id,
            _row_sha256(incoming, _REQUEST_COLUMNS),
        ),
    )


def _merge_binding_row(
    connection: sqlite3.Connection,
    shard_id: str,
    row: sqlite3.Row,
) -> None:
    incoming = _mapping_row(row)
    source_id = int(incoming["id"])
    existing_row = connection.execute(
        """
        SELECT * FROM destination.binding_upgrades
        WHERE binding_key=? AND from_sha256=?
        """,
        (
            incoming["binding_key"],
            incoming["from_sha256"],
        ),
    ).fetchone()
    outcome = "inserted"
    if existing_row is None:
        destination_id = int(
            connection.execute(
                "SELECT COALESCE(MAX(id),0)+1 FROM destination.binding_upgrades"
            ).fetchone()[0]
        )
        values = list(_row_values(incoming, _BINDING_COLUMNS))
        values[0] = destination_id
        _insert_named_row(
            connection,
            schema="destination",
            table="binding_upgrades",
            columns=_BINDING_COLUMNS,
            values=values,
        )
    else:
        existing = _mapping_row(existing_row)
        destination_id = int(existing["id"])
        if existing["to_sha256"] != incoming["to_sha256"]:
            raise MergeError("conflicting binding-upgrade branch")
        if existing["reason"] != incoming["reason"]:
            raise MergeError("conflicting binding-upgrade overlap")
        existing_time = _canonical_binding_upgrade_time(
            existing["upgraded_at"]
        )
        incoming_time = _canonical_binding_upgrade_time(
            incoming["upgraded_at"]
        )
        if existing_time == incoming_time:
            outcome = "exact_overlap"
        else:
            canonical_time = min(existing_time, incoming_time)
            connection.execute(
                """
                UPDATE destination.binding_upgrades
                SET upgraded_at=?
                WHERE id=?
                """,
                (canonical_time, destination_id),
            )
            outcome = "canonical_overlap"
    connection.execute(
        """
        INSERT INTO binding_id_map(
          shard_id,source_id,destination_id,source_row_sha256,outcome
        ) VALUES (?,?,?,?,?)
        """,
        (
            shard_id,
            source_id,
            destination_id,
            _row_sha256(incoming, _BINDING_COLUMNS),
            outcome,
        ),
    )


def _state_batch(
    source: sqlite3.Connection,
    *,
    table: str,
    cursor: tuple[Any, ...] | None,
    limit: int,
) -> list[sqlite3.Row]:
    if table == "attempts":
        order = "repo,run_id,attempt"
        cursor_columns = "(repo,run_id,attempt)"
        placeholders = "(?,?,?)"
    elif table == "members":
        order = "repo,run_id,attempt,archive_member"
        cursor_columns = "(repo,run_id,attempt,archive_member)"
        placeholders = "(?,?,?,?)"
    elif table in {"request_ledger", "binding_upgrades"}:
        order = "id"
        cursor_columns = "id"
        placeholders = "?"
    else:
        raise AssertionError(f"unsupported state table {table}")
    where = ""
    parameters: list[Any] = []
    if cursor is not None:
        where = f"WHERE {cursor_columns} > {placeholders}"
        parameters.extend(cursor)
    parameters.append(limit)
    return source.execute(
        f"SELECT * FROM {table} {where} ORDER BY {order} LIMIT ?",
        parameters,
    ).fetchall()


def _row_cursor(table: str, row: sqlite3.Row) -> tuple[Any, ...]:
    if table == "attempts":
        return (str(row["repo"]), int(row["run_id"]), int(row["attempt"]))
    if table == "members":
        return (
            str(row["repo"]),
            int(row["run_id"]),
            int(row["attempt"]),
            str(row["archive_member"]),
        )
    return (int(row["id"]),)


def _merge_state_table(
    journal: sqlite3.Connection,
    source: sqlite3.Connection,
    *,
    shard_id: str,
    table: str,
    limit: int,
    batch_budget: list[int | None],
) -> bool:
    expected_cursor_items = 3 if table == "attempts" else 4 if table == "members" else 1
    progress = journal.execute(
        """
        SELECT * FROM state_progress
        WHERE shard_id=? AND table_name=?
        """,
        (shard_id, table),
    ).fetchone()
    if progress is None:
        raise MergeError(f"state progress is missing for {shard_id}/{table}")
    if int(progress["done"]):
        return True
    cursor = _decode_cursor(progress["cursor_json"], expected_cursor_items)
    while True:
        rows = _state_batch(
            source,
            table=table,
            cursor=cursor,
            limit=limit,
        )
        if not rows:
            journal.execute("BEGIN IMMEDIATE")
            try:
                journal.execute(
                    """
                    UPDATE state_progress SET done=1
                    WHERE shard_id=? AND table_name=?
                    """,
                    (shard_id, table),
                )
                journal.commit()
            except BaseException:
                journal.rollback()
                raise
            return True
        journal.execute("BEGIN IMMEDIATE")
        try:
            for row in rows:
                if table == "attempts":
                    _merge_attempt_row(journal, shard_id, row)
                elif table == "members":
                    _merge_member_row(journal, shard_id, row)
                elif table == "request_ledger":
                    _merge_request_row(journal, shard_id, row)
                else:
                    _merge_binding_row(journal, shard_id, row)
            cursor = _row_cursor(table, rows[-1])
            journal.execute(
                """
                UPDATE state_progress
                SET cursor_json=?,processed_rows=processed_rows+?,batches=batches+1
                WHERE shard_id=? AND table_name=?
                """,
                (
                    _canonical_json(list(cursor)),
                    len(rows),
                    shard_id,
                    table,
                ),
            )
            journal.commit()
        except BaseException:
            journal.rollback()
            raise
        if batch_budget[0] is not None:
            batch_budget[0] -= 1
            if batch_budget[0] == 0:
                return False


def _merge_fetch_state(
    partial: Path,
    manifest: Manifest,
    audits: Sequence[SourceAudit],
    journal: sqlite3.Connection,
    *,
    batch_budget: list[int | None],
    fault_inject_sigkill_after: str | None,
) -> bool:
    store_receipt = _load_destination_store_receipt(partial)
    state_path = partial / _FETCH_STATE_NAME
    settings = _destination_state_settings(manifest, audits, store_receipt)
    _initialize_destination_state(
        state_path,
        settings,
        fault_inject_sigkill_after=fault_inject_sigkill_after,
    )
    attached = False
    try:
        journal.execute("ATTACH DATABASE ? AS destination", (str(state_path),))
        attached = True
        journal.execute("PRAGMA foreign_keys=ON")
        for audit in audits:
            source_path = audit.spec.state.path
            _require_frozen_sqlite(
                source_path,
                label=f"{audit.spec.shard_id} fetch state",
            )
            source = sqlite3.connect(
                f"{source_path.as_uri()}?mode=ro&immutable=1",
                uri=True,
            )
            source.row_factory = sqlite3.Row
            try:
                for table in (
                    "attempts",
                    "members",
                    "request_ledger",
                    "binding_upgrades",
                ):
                    completed = _merge_state_table(
                        journal,
                        source,
                        shard_id=audit.spec.shard_id,
                        table=table,
                        limit=manifest.limits.state_rows_per_batch,
                        batch_budget=batch_budget,
                    )
                    if not completed:
                        return False
            finally:
                source.close()
        _canonicalize_destination_binding_history(journal)
    finally:
        if attached:
            journal.execute("DETACH DATABASE destination")
    _fsync_file(state_path)
    _require_frozen_sqlite(state_path, label="destination fetch state")
    _set_journal_phase(journal, "verify")
    return True


def _canonicalize_destination_binding_history(
    connection: sqlite3.Connection,
) -> None:
    ordered: list[dict[str, Any]] = []
    rows_by_key: dict[str, list[dict[str, Any]]] = {
        binding_key: [] for binding_key in _BINDING_KEYS
    }
    for row in connection.execute(
        "SELECT * FROM destination.binding_upgrades ORDER BY id"
    ):
        materialized = _mapping_row(row)
        binding_key = str(materialized["binding_key"])
        if binding_key not in rows_by_key:
            raise MergeError(
                "destination binding history has an unsupported key"
            )
        rows_by_key[binding_key].append(materialized)

    for binding_key in _BINDING_KEYS:
        current_row = connection.execute(
            """
            SELECT value FROM destination.settings
            WHERE key=?
            """,
            (binding_key,),
        ).fetchone()
        if current_row is None:
            raise MergeError(
                f"destination {binding_key} binding is missing"
            )
        current = _require_hex64(
            current_row[0],
            where=f"destination {binding_key}",
        )
        rows = rows_by_key[binding_key]
        by_from: dict[str, dict[str, Any]] = {}
        destinations: set[str] = set()
        for row in rows:
            source = _require_hex64(
                row["from_sha256"],
                where=f"destination {binding_key} from_sha256",
            )
            destination = _require_hex64(
                row["to_sha256"],
                where=f"destination {binding_key} to_sha256",
            )
            _canonical_binding_upgrade_time(row["upgraded_at"])
            if source == destination or source in by_from:
                raise MergeError(
                    f"destination {binding_key} history contains a branch"
                )
            by_from[source] = row
            destinations.add(destination)
        if not rows:
            continue
        starts = sorted(set(by_from) - destinations)
        if len(starts) != 1:
            raise MergeError(
                f"destination {binding_key} history is not one linear chain"
            )
        cursor = starts[0]
        visited: set[str] = set()
        key_ordered: list[dict[str, Any]] = []
        while cursor in by_from:
            if cursor in visited:
                raise MergeError(
                    f"destination {binding_key} history contains a cycle"
                )
            visited.add(cursor)
            row = by_from[cursor]
            key_ordered.append(row)
            cursor = str(row["to_sha256"])
        if len(key_ordered) != len(rows) or cursor != current:
            raise MergeError(
                f"destination {binding_key} history does not terminate at "
                "its current binding"
            )
        ordered.extend(key_ordered)

    if not ordered:
        return

    old_ids = [int(row["id"]) for row in ordered]
    if old_ids == list(range(1, len(ordered) + 1)):
        return
    offset = max(old_ids) + len(old_ids) + 1
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            "UPDATE destination.binding_upgrades SET id=id+?",
            (offset,),
        )
        connection.execute(
            "UPDATE binding_id_map SET destination_id=destination_id+?",
            (offset,),
        )
        for destination_id, row in enumerate(ordered, start=1):
            temporary_id = int(row["id"]) + offset
            connection.execute(
                """
                UPDATE destination.binding_upgrades SET id=?
                WHERE id=?
                """,
                (destination_id, temporary_id),
            )
            connection.execute(
                """
                UPDATE binding_id_map SET destination_id=?
                WHERE destination_id=?
                """,
                (destination_id, temporary_id),
            )
        connection.execute(
            """
            DELETE FROM destination.sqlite_sequence
            WHERE name='binding_upgrades'
            """
        )
        connection.execute(
            """
            INSERT INTO destination.sqlite_sequence(name,seq)
            VALUES ('binding_upgrades',?)
            """,
            (len(ordered),),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _write_canonical_ledger(
    path: Path,
    *,
    domain: str,
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    artifact_digest = hashlib.sha256()
    logical_digest = hashlib.sha256()
    logical_digest.update(domain.encode("ascii"))
    logical_digest.update(b"\0")
    count = 0
    try:
        with temporary.open("xb") as handle:
            for record in records:
                encoded = _canonical_json_bytes(record)
                line = encoded + b"\n"
                handle.write(line)
                artifact_digest.update(line)
                logical_digest.update(len(encoded).to_bytes(8, "big"))
                logical_digest.update(encoded)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "artifact": path.name,
        "rows": count,
        "artifact_sha256": artifact_digest.hexdigest(),
        "logical_sha256": logical_digest.hexdigest(),
    }


def _mapping_ledgers(
    partial: Path,
    audits: Sequence[SourceAudit],
    journal: sqlite3.Connection,
) -> dict[str, dict[str, Any]]:
    ledger_root = partial / _LEDGER_DIRECTORY
    ledger_root.mkdir(exist_ok=True)
    _fsync_directory(partial)
    state_hashes = {
        audit.spec.shard_id: audit.state_file.sha256 for audit in audits
    }

    attempt_records = (
        {
            "schema": ATTEMPT_MAP_SCHEMA,
            "shard_id": str(row["shard_id"]),
            "source_state_sha256": state_hashes[str(row["shard_id"])],
            "source_key": json.loads(str(row["source_key_json"])),
            "source_row_sha256": str(row["source_row_sha256"]),
            "outcome": str(row["outcome"]),
        }
        for row in journal.execute(
            """
            SELECT * FROM attempt_map
            ORDER BY shard_id,source_key_json
            """
        )
    )
    member_records = (
        {
            "schema": MEMBER_MAP_SCHEMA,
            "shard_id": str(row["shard_id"]),
            "source_state_sha256": state_hashes[str(row["shard_id"])],
            "source_key": json.loads(str(row["source_key_json"])),
            "source_row_sha256": str(row["source_row_sha256"]),
            "outcome": str(row["outcome"]),
        }
        for row in journal.execute(
            """
            SELECT * FROM member_map
            ORDER BY shard_id,source_key_json
            """
        )
    )
    request_records = (
        {
            "schema": REQUEST_MAP_SCHEMA,
            "shard_id": str(row["shard_id"]),
            "source_state_sha256": state_hashes[str(row["shard_id"])],
            "source_id": int(row["source_id"]),
            "destination_id": int(row["destination_id"]),
            "source_row_sha256": str(row["source_row_sha256"]),
        }
        for row in journal.execute(
            """
            SELECT * FROM request_id_map
            ORDER BY shard_id,source_id
            """
        )
    )
    binding_records = (
        {
            "schema": BINDING_MAP_SCHEMA,
            "shard_id": str(row["shard_id"]),
            "source_state_sha256": state_hashes[str(row["shard_id"])],
            "source_id": int(row["source_id"]),
            "destination_id": int(row["destination_id"]),
            "source_row_sha256": str(row["source_row_sha256"]),
            "outcome": str(row["outcome"]),
        }
        for row in journal.execute(
            """
            SELECT * FROM binding_id_map
            ORDER BY shard_id,source_id
            """
        )
    )
    return {
        "attempts": _write_canonical_ledger(
            ledger_root / "attempt_resolutions.jsonl",
            domain="cppmega-ci-stream-attempt-resolution-ledger-v1",
            records=attempt_records,
        ),
        "members": _write_canonical_ledger(
            ledger_root / "member_resolutions.jsonl",
            domain="cppmega-ci-stream-member-resolution-ledger-v1",
            records=member_records,
        ),
        "requests": _write_canonical_ledger(
            ledger_root / "request_id_map.jsonl",
            domain="cppmega-ci-stream-request-id-map-ledger-v1",
            records=request_records,
        ),
        "bindings": _write_canonical_ledger(
            ledger_root / "binding_id_map.jsonl",
            domain="cppmega-ci-stream-binding-id-map-ledger-v1",
            records=binding_records,
        ),
    }


def _journal_logical_sha256(connection: sqlite3.Connection) -> str:
    tables = (
        ("settings", "key"),
        ("store_progress", "shard_id"),
        ("state_progress", "shard_id,table_name"),
        ("attempt_map", "shard_id,source_key_json"),
        ("member_map", "shard_id,source_key_json"),
        ("request_id_map", "shard_id,source_id"),
        ("binding_id_map", "shard_id,source_id"),
    )

    def records() -> Iterator[list[object]]:
        for table, order in tables:
            for row in connection.execute(
                f"SELECT * FROM {table} ORDER BY {order}"
            ):
                yield [table, *row]

    return _hash_records("cppmega-ci-stream-union-journal-logical-v1", records())


def _source_still_unchanged(audit: SourceAudit) -> None:
    spec = audit.spec
    _require_frozen_sqlite(spec.inventory.path, label=f"{spec.shard_id} inventory")
    _require_frozen_sqlite(spec.state.path, label=f"{spec.shard_id} fetch state")
    if (
        _snapshot_file(spec.inventory.path, label=f"{spec.shard_id} inventory")
        != audit.inventory_file
        or _snapshot_file(spec.state.path, label=f"{spec.shard_id} fetch state")
        != audit.state_file
        or _snapshot_receipts(spec) != audit.receipt_files
    ):
        raise MergeError(f"source shard {spec.shard_id} changed during merge")
    with FrozenStore(spec.store.path, spec.store.receipt.path) as store:
        if store._initial_snapshot != audit.store_files:
            raise MergeError(f"source store {spec.shard_id} changed during merge")
        store.require_unchanged()


def _state_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "attempts": int(connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]),
        "members": int(connection.execute("SELECT COUNT(*) FROM members").fetchone()[0]),
        "requests": int(
            connection.execute("SELECT COUNT(*) FROM request_ledger").fetchone()[0]
        ),
        "bindings": int(
            connection.execute("SELECT COUNT(*) FROM binding_upgrades").fetchone()[0]
        ),
        "chunks": int(
            connection.execute(
                "SELECT COALESCE(SUM(chunk_count),0) FROM members"
            ).fetchone()[0]
        ),
        "occurrence_tokens": int(
            connection.execute(
                "SELECT COALESCE(SUM(occurrence_tokens),0) FROM members"
            ).fetchone()[0]
        ),
    }


def _outcome_counts(
    connection: sqlite3.Connection,
    table: str,
) -> dict[str, int]:
    return {
        str(row["outcome"]): int(row["n"])
        for row in connection.execute(
            f"SELECT outcome,COUNT(*) AS n FROM {table} GROUP BY outcome ORDER BY outcome"
        )
    }


def _validate_destination_inventory(
    partial: Path,
    manifest: Manifest,
    inventory_contract: InventoryContract,
) -> None:
    anchor = inventory_contract.anchor
    path = partial / _INVENTORY_NAME
    _require_frozen_sqlite(path, label="destination inventory")
    snapshot = _snapshot_file(path, label="destination inventory")
    if (
        snapshot.sha256 != anchor.inventory_file.sha256
        or snapshot.size != anchor.inventory_file.size
    ):
        raise MergeError("destination inventory differs from its frozen source")
    try:
        computed = InventoryDB(path).completion_receipt()
    except (InventoryError, OSError, ValueError, sqlite3.Error) as exc:
        raise MergeError("destination inventory failed final validation") from exc
    computed["database"] = str(manifest.destination / _INVENTORY_NAME)
    declared, _raw = _load_json(
        partial / _INVENTORY_RECEIPT_NAME,
        where="destination inventory receipt",
    )
    completed_at = declared.get("completed_at")
    if (
        set(declared) != set(computed)
        or not isinstance(completed_at, str)
        or not completed_at
        or _inventory_logical_projection(declared)
        != _inventory_logical_projection(computed)
    ):
        raise MergeError("destination inventory receipt is not exact")


def _expected_progress_cursor(
    connection: sqlite3.Connection,
    table: str,
) -> tuple[int, tuple[object, ...] | None]:
    if table == "occurrences":
        columns = ("repo", "run_attempt", "job", "step", "chunk_ordinal")
    elif table == "attempts":
        columns = _ATTEMPT_KEY
    elif table == "members":
        columns = _MEMBER_KEY
    elif table in {"request_ledger", "binding_upgrades"}:
        columns = ("id",)
    else:
        raise AssertionError(f"unsupported progress table {table}")
    order = ",".join(f"{column} DESC" for column in columns)
    row = connection.execute(
        f"SELECT {','.join(columns)} FROM {table} ORDER BY {order} LIMIT 1"
    ).fetchone()
    count = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    if row is None:
        return count, None
    return count, tuple(_normalized_sql_value(row[column]) for column in columns)


def _require_completed_progress_row(
    progress: sqlite3.Row | None,
    *,
    expected_rows: int,
    expected_cursor: tuple[object, ...] | None,
    minimum_batches: int,
    maximum_batches: int,
    where: str,
) -> None:
    if progress is None:
        raise MergeError(f"{where} progress row is missing")
    declared_cursor = (
        None
        if progress["cursor_json"] is None
        else _decode_cursor(progress["cursor_json"], len(expected_cursor or ()))
    )
    if (
        int(progress["done"]) != 1
        or int(progress["processed_rows"]) != expected_rows
        or declared_cursor != expected_cursor
        or not (
            minimum_batches
            <= int(progress["batches"])
            <= maximum_batches
        )
    ):
        raise MergeError(f"{where} progress is not an exact completed traversal")


def _resolution_map_digest(connection: sqlite3.Connection) -> str:
    tables = (
        ("attempt_map", "shard_id,source_key_json"),
        ("member_map", "shard_id,source_key_json"),
        ("request_id_map", "shard_id,source_id"),
        ("binding_id_map", "shard_id,source_id"),
    )

    def records() -> Iterator[list[object]]:
        for table, order in tables:
            for row in connection.execute(
                f"SELECT * FROM {table} ORDER BY {order}"
            ):
                yield [table, *row]

    return _hash_records(
        "cppmega-ci-stream-union-resolution-maps-v1",
        records(),
    )


def _replay_and_validate_state_union(
    audits: Sequence[SourceAudit],
    journal: sqlite3.Connection,
    destination: sqlite3.Connection,
    *,
    scratch_directory: Path,
    expected_settings: Mapping[str, str],
) -> None:
    replay_journal_path = scratch_directory / "journal-replay.sqlite3"
    replay_state_path = scratch_directory / "state-replay.sqlite3"
    _remove_scratch_sqlite(replay_journal_path)
    _remove_scratch_sqlite(replay_state_path)
    replay_journal = sqlite3.connect(
        replay_journal_path,
        isolation_level=None,
    )
    replay_journal.row_factory = sqlite3.Row
    attached = False
    try:
        replay_journal.execute("PRAGMA journal_mode=DELETE")
        replay_journal.execute("PRAGMA synchronous=FULL")
        replay_journal.executescript(_JOURNAL_SQL)
        replay_state = sqlite3.connect(replay_state_path)
        try:
            replay_state.execute("PRAGMA journal_mode=DELETE")
            replay_state.execute("PRAGMA synchronous=FULL")
            replay_state.executescript(FETCH_STATE_SQL_SCHEMA)
            replay_state.executemany(
                "INSERT INTO settings(key,value) VALUES (?,?)",
                sorted(expected_settings.items()),
            )
            replay_state.commit()
        finally:
            replay_state.close()
        replay_journal.execute(
            "ATTACH DATABASE ? AS destination",
            (str(replay_state_path),),
        )
        attached = True
        replay_journal.execute("PRAGMA foreign_keys=ON")
        replay_journal.execute("BEGIN IMMEDIATE")
        try:
            for audit in audits:
                source = sqlite3.connect(
                    f"{audit.spec.state.path.as_uri()}?mode=ro&immutable=1",
                    uri=True,
                )
                source.row_factory = sqlite3.Row
                try:
                    for row in source.execute(
                        "SELECT * FROM attempts ORDER BY repo,run_id,attempt"
                    ):
                        _merge_attempt_row(
                            replay_journal,
                            audit.spec.shard_id,
                            row,
                        )
                    for row in source.execute(
                        """
                        SELECT * FROM members
                        ORDER BY repo,run_id,attempt,archive_member
                        """
                    ):
                        _merge_member_row(
                            replay_journal,
                            audit.spec.shard_id,
                            row,
                        )
                    for row in source.execute(
                        "SELECT * FROM request_ledger ORDER BY id"
                    ):
                        _merge_request_row(
                            replay_journal,
                            audit.spec.shard_id,
                            row,
                        )
                    for row in source.execute(
                        "SELECT * FROM binding_upgrades ORDER BY id"
                    ):
                        _merge_binding_row(
                            replay_journal,
                            audit.spec.shard_id,
                            row,
                        )
                finally:
                    source.close()
            replay_journal.commit()
        except BaseException:
            replay_journal.rollback()
            raise
        _canonicalize_destination_binding_history(replay_journal)
        replay_journal.execute("DETACH DATABASE destination")
        attached = False

        expected_state = sqlite3.connect(
            f"{replay_state_path.as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        expected_state.row_factory = sqlite3.Row
        try:
            if _fetch_state_logical_digest(expected_state) != (
                _fetch_state_logical_digest(destination)
            ):
                raise MergeError(
                    "destination fetch state differs from deterministic replay"
                )
        finally:
            expected_state.close()
        if _resolution_map_digest(replay_journal) != _resolution_map_digest(
            journal
        ):
            raise MergeError(
                "merge resolution maps differ from deterministic replay"
            )
    finally:
        if attached:
            replay_journal.execute("DETACH DATABASE destination")
        replay_journal.close()
        _remove_scratch_sqlite(replay_journal_path)
        _remove_scratch_sqlite(replay_state_path)
        _fsync_directory(scratch_directory)


def _validate_completed_journal(
    manifest: Manifest,
    audits: Sequence[SourceAudit],
    journal: sqlite3.Connection,
    destination_state: Path,
) -> None:
    if _journal_phase(journal) not in {"verify", "ready"}:
        raise MergeError("merge journal is not ready for completion validation")
    destination = sqlite3.connect(
        f"{destination_state.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    destination.row_factory = sqlite3.Row
    try:
        store_receipt = _load_destination_store_receipt(
            destination_state.parent
        )
        expected_settings = _destination_state_settings(
            manifest,
            audits,
            store_receipt,
        )
        actual_settings = {
            str(row["key"]): str(row["value"])
            for row in destination.execute(
                "SELECT key,value FROM settings ORDER BY key"
            )
        }
        if actual_settings != expected_settings:
            raise MergeError(
                "destination fetch-state settings differ from frozen inputs"
            )
        for audit in audits:
            shard_id = audit.spec.shard_id
            store_rows = 0
            store_batches = 0
            store_cursor: tuple[object, ...] | None = None
            with FrozenStore(
                audit.spec.store.path,
                audit.spec.store.receipt.path,
            ) as source_store:
                while True:
                    batch = _source_occurrence_batch(
                        source_store,
                        store_cursor,
                        manifest.limits,
                    )
                    if not batch:
                        break
                    last = batch[-1]
                    store_cursor = (
                        str(last["repo"]),
                        str(last["run_attempt"]),
                        str(last["job"]),
                        str(last["step"]),
                        int(last["chunk_ordinal"]),
                    )
                    store_rows += len(batch)
                    store_batches += 1
                source_store.require_unchanged()
            _require_completed_progress_row(
                journal.execute(
                    "SELECT * FROM store_progress WHERE shard_id=?",
                    (shard_id,),
                ).fetchone(),
                expected_rows=store_rows,
                expected_cursor=store_cursor,
                minimum_batches=store_batches,
                maximum_batches=store_batches,
                where=f"{shard_id} store",
            )

            source_state = sqlite3.connect(
                f"{audit.spec.state.path.as_uri()}?mode=ro&immutable=1",
                uri=True,
            )
            source_state.row_factory = sqlite3.Row
            try:
                for table in (
                    "attempts",
                    "members",
                    "request_ledger",
                    "binding_upgrades",
                ):
                    row_count, cursor = _expected_progress_cursor(
                        source_state,
                        table,
                    )
                    batches = (
                        row_count + manifest.limits.state_rows_per_batch - 1
                    ) // manifest.limits.state_rows_per_batch
                    _require_completed_progress_row(
                        journal.execute(
                            """
                            SELECT * FROM state_progress
                            WHERE shard_id=? AND table_name=?
                            """,
                            (shard_id, table),
                        ).fetchone(),
                        expected_rows=row_count,
                        expected_cursor=cursor,
                        minimum_batches=batches,
                        maximum_batches=batches,
                        where=f"{shard_id} {table}",
                    )
            finally:
                source_state.close()
        _replay_and_validate_state_union(
            audits,
            journal,
            destination,
            scratch_directory=destination_state.parent
            / ".source-verification",
            expected_settings=expected_settings,
        )
    finally:
        destination.close()


def _finalize_receipts(
    partial: Path,
    manifest: Manifest,
    audits: Sequence[SourceAudit],
    inventory_contract: InventoryContract,
    tokenizer: ExactTokenizer,
    journal: sqlite3.Connection,
    *,
    merge_script_sha256: str,
) -> dict[str, Any]:
    _validate_destination_inventory(partial, manifest, inventory_contract)
    _validate_completed_journal(
        manifest,
        audits,
        journal,
        partial / _FETCH_STATE_NAME,
    )
    ledgers = _mapping_ledgers(partial, audits, journal)
    store_receipt = _load_destination_store_receipt(partial)
    state_path = partial / _FETCH_STATE_NAME
    binding_store = SimpleNamespace(
        root=(manifest.destination / _STORE_DIRECTORY).resolve(),
        receipt=store_receipt,
    )
    with FrozenStore(
        partial / _STORE_DIRECTORY,
        partial / _STORE_RECEIPT_NAME,
    ) as store:
        with FrozenFetchState(
            state_path,
            tokenizer=tokenizer,
            store=cast(Any, binding_store),
        ) as state:
            _state_blob_limits(state.connection, manifest.limits)
            _reject_unsafe_attempt_states(state.connection)
            _validate_binding_history(
                state.connection,
                current_bindings=state.settings,
            )
            joined = _verify_cas_fetch_join(
                store,
                state,
                scratch_path=partial / ".union-coverage.sqlite3",
                limits=manifest.limits,
            )
            destination_inventory = sqlite3.connect(
                (
                    f"{(partial / _INVENTORY_NAME).as_uri()}"
                    "?mode=ro&immutable=1"
                ),
                uri=True,
            )
            destination_inventory.row_factory = sqlite3.Row
            try:
                (
                    inventory_joined_attempts,
                    inventory_join_sha256,
                ) = _verify_state_inventory_join(
                    destination_inventory,
                    state.connection,
                    label="destination",
                    max_blob_bytes=manifest.limits.max_state_blob_bytes,
                )
            finally:
                destination_inventory.close()
            frozen_binding = state.receipt_binding()
            frozen_binding["artifact"]["path"] = str(
                manifest.destination / _FETCH_STATE_NAME
            )
            destination_state_counts = _state_counts(state.connection)
            if inventory_joined_attempts != destination_state_counts["attempts"]:
                raise MergeError(
                    "destination full-anchor join did not cover every attempt"
                )
            state.require_unchanged()
        store.require_unchanged()
    source_scratch = partial / ".source-verification"
    if source_scratch.exists():
        if source_scratch.is_symlink() or not source_scratch.is_dir():
            raise MergeError("source-verification scratch directory is unsafe")
        try:
            source_scratch.rmdir()
        except OSError as exc:
            raise MergeError(
                "source-verification scratch directory is not empty"
            ) from exc

    if _journal_phase(journal) == "verify":
        journal.execute("BEGIN IMMEDIATE")
        try:
            completed_at = _utc_now()
            journal.execute(
                "INSERT INTO settings(key,value) VALUES ('completed_at',?)",
                (completed_at,),
            )
            journal.execute(
                "UPDATE settings SET value='ready' WHERE key='phase'"
            )
            journal.commit()
        except BaseException:
            journal.rollback()
            raise
    completed_at_row = journal.execute(
        "SELECT value FROM settings WHERE key='completed_at'"
    ).fetchone()
    if completed_at_row is None:
        raise MergeError("ready journal lacks completed_at")
    completed_at = str(completed_at_row[0])

    fetch_receipt = {
        "schema": FETCH_RECEIPT_SCHEMA,
        "completed_at": completed_at,
        "target_exact_unique_payload_tokens": manifest.target_unique_tokens,
        "fetch_state": frozen_binding["summary"],
        "frozen_fetch_state": frozen_binding,
        "content_store_receipt": store_receipt,
        "inventory_path": str(manifest.destination / _INVENTORY_NAME),
        "inventory_binding": inventory_contract.binding,
        "tokenizer_contract": tokenizer.contract,
        "tokenizer_fingerprint": tokenizer.fingerprint,
    }
    _write_json(partial / _FETCH_RECEIPT_NAME, fetch_receipt)

    input_store_totals = {
        field: sum(audit.store_counts[field] for audit in audits)
        for field in audits[0].store_counts
    }
    output_store_counts = _integer_counts(
        cast(Mapping[str, Any], store_receipt["counters"]),
        tuple(input_store_totals),
    )
    store_overlap = {
        "unique_contents": (
            input_store_totals["unique_content_count"]
            - output_store_counts["unique_content_count"]
        ),
        "occurrences": (
            input_store_totals["occurrence_count"]
            - output_store_counts["occurrence_count"]
        ),
        "token_sequences": (
            input_store_totals["unique_token_sequence_count"]
            - output_store_counts["unique_token_sequence_count"]
        ),
        "exact_unique_payload_tokens": (
            input_store_totals["exact_unique_payload_tokens"]
            - output_store_counts["exact_unique_payload_tokens"]
        ),
        "raw_occurrence_bytes": (
            input_store_totals["raw_occurrence_bytes"]
            - output_store_counts["raw_occurrence_bytes"]
        ),
        "unique_bytes": (
            input_store_totals["unique_bytes"]
            - output_store_counts["unique_bytes"]
        ),
    }
    if any(value < 0 for value in store_overlap.values()):
        raise MergeError("store conservation produced a negative overlap")
    input_state_totals = {
        field: sum(audit.state_counts[field] for audit in audits)
        for field in (
            "attempts",
            "members",
            "requests",
            "bindings",
            "chunks",
            "occurrence_tokens",
        )
    }
    if destination_state_counts["requests"] != input_state_totals["requests"]:
        raise MergeError("request multiplicity was not conserved")
    request_map_count = int(
        journal.execute("SELECT COUNT(*) FROM request_id_map").fetchone()[0]
    )
    if request_map_count != input_state_totals["requests"]:
        raise MergeError("request ID map is not exhaustive")
    binding_map_count = int(
        journal.execute("SELECT COUNT(*) FROM binding_id_map").fetchone()[0]
    )
    if binding_map_count != input_state_totals["bindings"]:
        raise MergeError("binding ID map is not exhaustive")
    attempt_outcomes = _outcome_counts(journal, "attempt_map")
    member_outcomes = _outcome_counts(journal, "member_map")
    binding_outcomes = _outcome_counts(journal, "binding_id_map")
    attempt_map_count = int(
        journal.execute("SELECT COUNT(*) FROM attempt_map").fetchone()[0]
    )
    member_map_count = int(
        journal.execute("SELECT COUNT(*) FROM member_map").fetchone()[0]
    )
    if attempt_map_count != input_state_totals["attempts"]:
        raise MergeError("attempt resolution map is not exhaustive")
    if member_map_count != input_state_totals["members"]:
        raise MergeError("member resolution map is not exhaustive")
    state_overlap = {
        field: input_state_totals[field] - destination_state_counts[field]
        for field in (
            "attempts",
            "members",
            "bindings",
            "chunks",
            "occurrence_tokens",
        )
    }
    if any(value < 0 for value in state_overlap.values()):
        raise MergeError("fetch-state conservation produced a negative overlap")
    if joined != output_store_counts["occurrence_count"]:
        raise MergeError("full CAS/fetch join count differs from union occurrences")

    journal_path = partial / _JOURNAL_NAME
    integrity = [
        str(row[0])
        for row in journal.execute("PRAGMA integrity_check").fetchall()
    ]
    if integrity != ["ok"]:
        raise MergeError(f"merge journal integrity_check failed: {integrity}")
    if journal.execute("PRAGMA foreign_key_check").fetchall():
        raise MergeError("merge journal foreign_key_check failed")
    _require_frozen_sqlite(journal_path, label="merge journal")
    journal_snapshot = _snapshot_file(journal_path, label="merge journal")
    artifacts = []
    for relative in (
        _INVENTORY_NAME,
        _INVENTORY_RECEIPT_NAME,
        _FETCH_STATE_NAME,
        _STORE_RECEIPT_NAME,
        _FETCH_RECEIPT_NAME,
    ):
        path = partial / relative
        artifacts.append(
            {
                "path": relative,
                "byte_size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    for ledger in ledgers.values():
        relative = f"{_LEDGER_DIRECTORY}/{ledger['artifact']}"
        path = partial / relative
        artifacts.append(
            {
                "path": relative,
                "byte_size": path.stat().st_size,
                "sha256": ledger["artifact_sha256"],
            }
        )
    artifacts.append(
        {
            "path": _JOURNAL_NAME,
            "byte_size": journal_snapshot.size,
            "sha256": journal_snapshot.sha256,
        }
    )
    store_files = []
    for path in sorted((partial / _STORE_DIRECTORY).rglob("*")):
        if path.is_file():
            store_files.append(
                {
                    "path": path.relative_to(partial).as_posix(),
                    "byte_size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    artifacts.extend(store_files)

    inventory_sources = {
        str(item["source_id"]): item
        for item in cast(
            Sequence[Mapping[str, Any]],
            inventory_contract.binding["sources"],
        )
    }
    receipt = {
        "schema": MERGE_RECEIPT_SCHEMA,
        "status": "complete",
        "completed_at": completed_at,
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.sha256,
            "schema": MANIFEST_SCHEMA,
        },
        "merge_script_sha256": merge_script_sha256,
        "destination": str(manifest.destination),
        "target_exact_unique_payload_tokens": manifest.target_unique_tokens,
        "sources": [
            {
                "id": audit.spec.shard_id,
                "original_paths": {
                    "inventory": audit.spec.original_inventory,
                    "content_store": audit.spec.original_store,
                    "fetch_state": audit.spec.original_state,
                },
                "staged": {
                    "inventory": {
                        "path": str(audit.spec.inventory.path),
                        "sha256": audit.inventory_file.sha256,
                        "receipt_sha256": (
                            None
                            if audit.spec.inventory.receipt is None
                            else audit.spec.inventory.receipt.sha256
                        ),
                        "binding": inventory_sources[audit.spec.shard_id],
                    },
                    "content_store": {
                        "path": str(audit.spec.store.path),
                        "artifact_set_sha256": audit.spec.store.artifact_set_sha256,
                        "receipt_sha256": audit.spec.store.receipt.sha256,
                    },
                    "fetch_state": {
                        "path": str(audit.spec.state.path),
                        "sha256": audit.state_file.sha256,
                        "receipt_sha256": audit.spec.state.receipt.sha256,
                        "sqlite_logical_sha256": audit.state_binding[
                            "sqlite_logical_sha256"
                        ],
                    },
                },
                "verified_before_merge": True,
                "unchanged_after_merge": True,
            }
            for audit in audits
        ],
        "inventory": inventory_contract.binding,
        "store_conservation": {
            "input_multiplicity": input_store_totals,
            "output_union": output_store_counts,
            "overlap": store_overlap,
            "equations": {
                "unique_contents": (
                    input_store_totals["unique_content_count"]
                    == output_store_counts["unique_content_count"]
                    + store_overlap["unique_contents"]
                ),
                "occurrences": (
                    input_store_totals["occurrence_count"]
                    == output_store_counts["occurrence_count"]
                    + store_overlap["occurrences"]
                ),
                "token_sequences": (
                    input_store_totals["unique_token_sequence_count"]
                    == output_store_counts["unique_token_sequence_count"]
                    + store_overlap["token_sequences"]
                ),
                "tokens": (
                    input_store_totals["exact_unique_payload_tokens"]
                    == output_store_counts["exact_unique_payload_tokens"]
                    + store_overlap["exact_unique_payload_tokens"]
                ),
                "raw_occurrence_bytes": (
                    input_store_totals["raw_occurrence_bytes"]
                    == output_store_counts["raw_occurrence_bytes"]
                    + store_overlap["raw_occurrence_bytes"]
                ),
                "unique_bytes": (
                    input_store_totals["unique_bytes"]
                    == output_store_counts["unique_bytes"]
                    + store_overlap["unique_bytes"]
                ),
            },
        },
        "fetch_state_conservation": {
            "input_multiplicity": input_state_totals,
            "output_union": destination_state_counts,
            "overlap": state_overlap,
            "equations": {
                "attempts": (
                    input_state_totals["attempts"]
                    == destination_state_counts["attempts"]
                    + state_overlap["attempts"]
                ),
                "members": (
                    input_state_totals["members"]
                    == destination_state_counts["members"]
                    + state_overlap["members"]
                ),
                "requests": (
                    input_state_totals["requests"]
                    == destination_state_counts["requests"]
                ),
                "bindings": (
                    input_state_totals["bindings"]
                    == destination_state_counts["bindings"]
                    + state_overlap["bindings"]
                ),
                "chunks": (
                    input_state_totals["chunks"]
                    == destination_state_counts["chunks"]
                    + state_overlap["chunks"]
                ),
                "occurrence_tokens": (
                    input_state_totals["occurrence_tokens"]
                    == destination_state_counts["occurrence_tokens"]
                    + state_overlap["occurrence_tokens"]
                ),
            },
            "request_multiplicity_preserved": True,
            "attempt_map_count": attempt_map_count,
            "member_map_count": member_map_count,
            "request_map_count": request_map_count,
            "binding_map_count": binding_map_count,
            "attempt_outcomes": attempt_outcomes,
            "member_outcomes": member_outcomes,
            "binding_outcomes": binding_outcomes,
        },
        "verification": {
            "full_cas_fetch_join": True,
            "joined_occurrences": joined,
            "full_anchor_fetch_state_join": True,
            "inventory_joined_attempts": inventory_joined_attempts,
            "inventory_join_sha256": inventory_join_sha256,
            "sources_frozen_before_after": True,
            "destination_frozen": True,
            "threshold_recomputed_after_global_dedup": True,
        },
        "frozen_fetch_state": frozen_binding,
        "ledgers": ledgers,
        "journal": {
            "schema": JOURNAL_SCHEMA,
            "sqlite_schema_sha256": _sqlite_schema_sha256(journal),
            "sqlite_logical_sha256": _journal_logical_sha256(journal),
            "artifact_sha256": journal_snapshot.sha256,
        },
        "artifacts": sorted(artifacts, key=lambda item: str(item["path"])),
    }
    _write_json(partial / _MERGE_RECEIPT_NAME, receipt)
    return receipt


def _fsync_tree(root: Path) -> None:
    directories = {root}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise MergeError(f"bundle contains a symlink: {path}")
        if path.is_file():
            _fsync_file(path)
        elif path.is_dir():
            directories.add(path)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _require_complete_bundle_tree(
    partial: Path,
    receipt: Mapping[str, Any],
) -> None:
    raw_artifacts = receipt.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise MergeError("merge receipt artifacts must be a list")
    expected_files = {_MERGE_RECEIPT_NAME}
    expected_artifacts: dict[str, tuple[int, str]] = {}
    for index, raw_artifact in enumerate(raw_artifacts):
        artifact = _require_mapping(
            raw_artifact,
            where=f"merge receipt artifact {index}",
        )
        _require_exact_keys(
            artifact,
            {"path", "byte_size", "sha256"},
            where=f"merge receipt artifact {index}",
        )
        relative = _require_string(
            artifact.get("path"),
            where=f"merge receipt artifact {index}.path",
        )
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or str(pure) != relative
            or relative in {"", "."}
            or ".." in pure.parts
        ):
            raise MergeError("merge receipt contains an unsafe artifact path")
        if relative in expected_files:
            raise MergeError("merge receipt contains a duplicate artifact path")
        expected_files.add(relative)
        expected_artifacts[relative] = (
            _require_nonnegative_int(
                artifact.get("byte_size"),
                where=f"merge receipt artifact {index}.byte_size",
            ),
            _require_hex64(
                artifact.get("sha256"),
                where=f"merge receipt artifact {index}.sha256",
            ),
        )

    expected_directories: set[str] = set()
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            expected_directories.add(str(parent))
            parent = parent.parent

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in partial.rglob("*"):
        if path.is_symlink():
            raise MergeError(f"bundle contains a symlink: {path}")
        relative = path.relative_to(partial).as_posix()
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise MergeError(f"bundle contains an unsupported artifact: {path}")
    if actual_files != expected_files:
        raise MergeError(
            "partial bundle files differ from the receipt; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    if actual_directories != expected_directories:
        raise MergeError(
            "partial bundle directories differ from the receipt; "
            f"missing={sorted(expected_directories - actual_directories)}, "
            f"unexpected={sorted(actual_directories - expected_directories)}"
        )
    for relative, (expected_size, expected_sha256) in sorted(
        expected_artifacts.items()
    ):
        path = partial / relative
        stat_before = path.stat()
        if stat_before.st_size != expected_size:
            raise MergeError(f"bundle artifact size changed: {relative}")
        actual_sha256 = _sha256_file(path)
        stat_after = path.stat()
        if (
            actual_sha256 != expected_sha256
            or (
                stat_before.st_size,
                stat_before.st_mtime_ns,
                stat_before.st_ino,
            )
            != (
                stat_after.st_size,
                stat_after.st_mtime_ns,
                stat_after.st_ino,
            )
        ):
            raise MergeError(f"bundle artifact changed after its receipt: {relative}")
    declared_receipt, raw_receipt = _load_json(
        partial / _MERGE_RECEIPT_NAME,
        where="on-disk merge receipt",
    )
    if declared_receipt != dict(receipt) or raw_receipt != _json_document_bytes(receipt):
        raise MergeError("on-disk merge receipt differs from the finalized receipt")


def _load_pinned_tokenizer(
    source: Path,
    expected: SnapshotFile,
    scratch: Path,
) -> ExactTokenizer:
    pinned = scratch / "pinned-tokenizer.json"
    if pinned.exists() or pinned.is_symlink():
        if pinned.is_symlink() or not pinned.is_file():
            raise MergeError("pinned tokenizer scratch path is unsafe")
        pinned.unlink()
        _fsync_directory(scratch)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_handle:
            with pinned.open("xb") as output_handle:
                while True:
                    block = input_handle.read(1024 * 1024)
                    if not block:
                        break
                    output_handle.write(block)
                    digest.update(block)
                output_handle.flush()
                os.fsync(output_handle.fileno())
        if digest.hexdigest() != expected.sha256:
            raise MergeError("tokenizer changed while its pinned copy was created")
        tokenizer = ExactTokenizer(pinned)
        if tokenizer.artifact_sha256 != expected.sha256:
            raise MergeError("pinned tokenizer hash differs from its snapshot")
        return tokenizer
    finally:
        if pinned.exists():
            pinned.unlink()
            _fsync_directory(scratch)


def _require_control_inputs_unchanged(
    manifest: Manifest,
    *,
    manifest_snapshot: SnapshotFile,
    tokenizer_snapshot: SnapshotFile,
    merge_script_snapshot: SnapshotFile,
) -> None:
    if (
        _snapshot_file(manifest.path, label="union manifest")
        != manifest_snapshot
        or _snapshot_file(
            manifest.tokenizer_path,
            label="tokenizer artifact",
        )
        != tokenizer_snapshot
        or _snapshot_file(
            Path(__file__).resolve(),
            label="merge script",
        )
        != merge_script_snapshot
    ):
        raise MergeError(
            "manifest, tokenizer, or merge script changed during merge"
        )


def merge_shards(
    manifest_path: str | os.PathLike[str],
    *,
    max_batches: int | None = None,
    fault_inject_sigkill_after: str | None = None,
) -> dict[str, Any]:
    """Merge and atomically publish the manifest's frozen shards.

    ``max_batches`` is an operational pause bound.  It is useful for scheduled
    runs and for exercising the same replay window as a crash: committed data
    may be ahead of the external cursor, and replay remains idempotent.
    """

    if max_batches is not None:
        max_batches = _require_positive_int(max_batches, where="max_batches")
    if (
        fault_inject_sigkill_after is not None
        and fault_inject_sigkill_after not in _SIGKILL_FAULT_POINTS
    ):
        raise MergeError("unsupported SIGKILL fault-injection point")
    manifest = load_manifest(manifest_path)
    manifest_snapshot = _snapshot_file(manifest.path, label="union manifest")
    tokenizer_snapshot = _snapshot_file(
        manifest.tokenizer_path,
        label="tokenizer artifact",
    )
    merge_script_snapshot = _snapshot_file(
        Path(__file__).resolve(),
        label="merge script",
    )
    if manifest_snapshot.sha256 != manifest.sha256:
        raise MergeError("manifest changed while it was loaded")
    if tokenizer_snapshot.size > _HARD_MAX_TOKENIZER_BYTES:
        raise MergeError("tokenizer artifact exceeds the hard byte bound")
    if tokenizer_snapshot.sha256 != manifest.tokenizer_sha256:
        raise MergeError("tokenizer artifact differs from the manifest")
    _validate_output_geometry(manifest)
    lock_descriptor = _acquire_merge_lock(manifest.destination)
    try:
        partial = _ensure_safe_partial(manifest)
        journal = _open_journal(
            partial,
            manifest,
            merge_script_sha256=merge_script_snapshot.sha256,
            fault_inject_sigkill_after=fault_inject_sigkill_after,
        )
    except BaseException:
        _release_merge_lock(lock_descriptor)
        raise
    batch_budget: list[int | None] = [max_batches]
    scratch = partial / ".source-verification"
    audits: list[SourceAudit] = []
    try:
        scratch.mkdir(exist_ok=True)
        tokenizer = _load_pinned_tokenizer(
            manifest.tokenizer_path,
            tokenizer_snapshot,
            scratch,
        )
        for spec in manifest.shards:
            audits.append(
                _preflight_source(
                    manifest,
                    spec,
                    tokenizer,
                    scratch,
                )
            )
        inventory_contract = _validate_cross_source_contracts(audits)
        phase = _journal_phase(journal)
        if phase == "store":
            receipt = _merge_store(
                partial,
                manifest,
                journal,
                batch_budget=batch_budget,
            )
            if receipt is None:
                raise MergePaused("merge paused at the requested batch bound")
            phase = _journal_phase(journal)
        if phase == "inventory":
            _copy_and_validate_inventory(
                partial,
                manifest,
                inventory_contract,
                journal,
            )
            phase = _journal_phase(journal)
        if phase == "state":
            completed = _merge_fetch_state(
                partial,
                manifest,
                audits,
                journal,
                batch_budget=batch_budget,
                fault_inject_sigkill_after=fault_inject_sigkill_after,
            )
            if not completed:
                raise MergePaused("merge paused at the requested batch bound")
            phase = _journal_phase(journal)
        if phase in {"verify", "ready"}:
            receipt = _finalize_receipts(
                partial,
                manifest,
                audits,
                inventory_contract,
                tokenizer,
                journal,
                merge_script_sha256=merge_script_snapshot.sha256,
            )
        else:
            raise MergeError(f"unsupported terminal merge phase {phase!r}")
        _require_control_inputs_unchanged(
            manifest,
            manifest_snapshot=manifest_snapshot,
            tokenizer_snapshot=tokenizer_snapshot,
            merge_script_snapshot=merge_script_snapshot,
        )
        _fsync_tree(partial)
        for audit in audits:
            _source_still_unchanged(audit)
        _require_control_inputs_unchanged(
            manifest,
            manifest_snapshot=manifest_snapshot,
            tokenizer_snapshot=tokenizer_snapshot,
            merge_script_snapshot=merge_script_snapshot,
        )
        if manifest.destination.exists() or manifest.destination.is_symlink():
            raise MergeError(f"destination appeared during merge: {manifest.destination}")
        _require_complete_bundle_tree(partial, receipt)
        for audit in audits:
            _source_still_unchanged(audit)
        _require_control_inputs_unchanged(
            manifest,
            manifest_snapshot=manifest_snapshot,
            tokenizer_snapshot=tokenizer_snapshot,
            merge_script_snapshot=merge_script_snapshot,
        )
        if manifest.destination.exists() or manifest.destination.is_symlink():
            raise MergeError("destination appeared during final verification")
        _publish_directory_no_replace(partial, manifest.destination)
        _fsync_directory(manifest.destination.parent)
        return receipt
    except MergePaused:
        for audit in audits:
            _source_still_unchanged(audit)
        _require_control_inputs_unchanged(
            manifest,
            manifest_snapshot=manifest_snapshot,
            tokenizer_snapshot=tokenizer_snapshot,
            merge_script_snapshot=merge_script_snapshot,
        )
        raise
    except (ExportError, sqlite3.Error) as exc:
        for audit in audits:
            _source_still_unchanged(audit)
        raise MergeError(str(exc)) from exc
    except BaseException:
        for audit in audits:
            _source_still_unchanged(audit)
        raise
    finally:
        try:
            journal.close()
        finally:
            _release_merge_lock(lock_descriptor)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="canonical union manifest JSON")
    parser.add_argument(
        "--max-batches",
        type=int,
        help="durably pause after this many store/state batches",
    )
    parser.add_argument(
        "--fault-inject-sigkill-after",
        choices=_SIGKILL_FAULT_POINTS,
        help="diagnostic crash injection for resume certification",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        receipt = merge_shards(
            args.manifest,
            max_batches=args.max_batches,
            fault_inject_sigkill_after=args.fault_inject_sigkill_after,
        )
    except MergePaused as exc:
        print(json.dumps({"status": "paused", "reason": str(exc)}, sort_keys=True))
        return 2
    except MergeError as exc:
        print(f"merge failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
