#!/usr/bin/env python3
"""Export a frozen CI content-store snapshot to audited CASE5 Parquet.

The exporter is deliberately fail closed:

* a fully verified content-store completion receipt is mandatory;
* SQLite is opened read-only with ``immutable=1`` and every logical/pack digest
  is compared with that receipt;
* a second immutable fetch-state snapshot must bind exact run metadata and the
  canonical parser sidecar for every occurrence;
* nested ZIP members with high invalid-UTF-8 ratios are conserved in an
  exclusion ledger and never enter representative selection;
* every occurrence must carry exact-attempt v3 provenance and exhaustive v2
  chunk training sidecars;
* every build-action source input is audited against its parser generation and
  emitted through a receipt-bound source-binding projection ledger;
* every unique content is re-tokenized with ``ExactTokenizer`` before token
  sequence representatives are selected;
* payload token arrays are split before BOS/domain framing, with no truncation;
* output is published by one directory rename only after the current CASE5
  auditor accepts every Parquet file and the input snapshot is unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pyarrow.parquet as pq  # noqa: E402

from cppmega.data.domain_schema import (  # noqa: E402
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
    delimiter_token_ids,
    domain_edge_family,
)
from cppmega.data.ci_training_scope import (  # noqa: E402
    AUX_JS_TS_ROUTE,
    AUX_PYTHON_ROUTE,
    CITrainingScopeError,
    PRIMARY_ROUTE,
    TRAINING_SCOPE_DECISION_SCHEMA,
    classify_ci_training_sidecars,
    training_scope_policy,
)
from cppmega.data.nanochat_pipeline.tokenized_enriched import (  # noqa: E402
    _char_position_to_token_index,
    _chars_to_tokens_structure_ids,
)
from cppmega.data.nanochat_pipeline.tokenized_enriched_schema import (  # noqa: E402
    SOURCE_IDENTITY_REGISTRY_COLUMN,
    TOKEN_BUILD_EDGES_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_CONFIDENCE_IDS_COLUMN,
    TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
    TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    TOKEN_DOMAIN_EDGES_COLUMN,
    TOKEN_DOMAIN_IDS_COLUMN,
    TOKEN_ENTITY_IDS_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_ROLE_IDS_COLUMN,
    TOKEN_SCOPE_IDS_COLUMN,
    TOKEN_SHELL_EDGES_COLUMN,
    TOKEN_SOURCE_DOC_IDS_COLUMN,
    TOKEN_SOURCE_IDENTITY_IDS_COLUMN,
)
from cppmega.data.source_identity import source_identity  # noqa: E402
from cppmega.data.tokenizer_contract import EXPECTED_VOCAB_SIZE  # noqa: E402
from cppmega.megatron.domain_route_contract import (  # noqa: E402
    CASE5_SCHEMA_VERSION,
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    DOMAIN_SCHEMA_SHA256,
    TOKENIZER_CONTRACT_SHA256,
    VALID_DOMAIN_CONFIDENCE_IDS,
    VALID_DOMAIN_EDGE_KINDS,
    VALID_DOMAIN_IDS,
    VALID_DOMAIN_ROLE_IDS,
)
from scripts.audit_sidecar_parquet import _audit_file  # noqa: E402
from scripts.canonical_parquet_ledger import (  # noqa: E402
    CanonicalParquetLedgerWriter,
    iter_canonical_parquet_ledger,
)
from scripts.ci_content_store import (  # noqa: E402
    RECEIPT_SCHEMA as STORE_RECEIPT_SCHEMA,
    STORE_SCHEMA,
    TOKEN_SEQUENCE_ENCODING,
    _FRAME_HEADER,
    _FRAME_MAGIC,
    _PACK_MAGIC,
    _hash_records,
    _sqlite_schema_sha256,
    hash_token_sequence,
)
from scripts.ci_source_binding_projection import (  # noqa: E402
    MAX_SOURCE_BINDING_PROJECTION_RECORD_BYTES,
    SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
    SOURCE_BINDING_PROJECTION_SCHEMA,
    SourceBindingProjectionError,
    SourceBindingProjector,
    SourceBindingProjectionRouter,
    projection_record_key,
    projection_script_sha256,
    target_parser_script_sha256,
)
from scripts.ci_log_sidecars import SIDECAR_SCHEMA as PARSER_SIDECAR_SCHEMA  # noqa: E402
from scripts.ci_stream_fetch import (  # noqa: E402
    COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
    COMPLETION_MODE_THRESHOLD,
    EXHAUSTIVE_RECEIPT_SCHEMA,
    SCHEMA_VERSION as FETCH_STATE_SCHEMA,
    _STATE_SCHEMA as FETCH_STATE_SQL_SCHEMA,
    _job_for_member,
    _validate_run_metadata_identity,
    ExactTokenizer,
    MalformedResponseError,
)
from scripts.ci_zlib_evidence import (  # noqa: E402
    MAX_CONTENT_FRAME_BYTES,
    MAX_CONTENT_FRAME_COMPRESSED_BYTES,
    MAX_JOBS_EVIDENCE_BYTES,
    MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
    MAX_RUN_METADATA_BYTES,
    MAX_RUN_METADATA_COMPRESSED_BYTES,
    MAX_STATE_JSON_EVIDENCE_BYTES,
    MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES,
    ZlibEvidenceError,
    constrain_sqlite_evidence_rows,
    content_store_evidence_bound_violation,
    fetch_state_evidence_bound_violation,
    strict_bounded_zlib_decode,
)
from scripts.ci_stream_inventory import (  # noqa: E402
    CompletionError as InventoryCompletionError,
    verify_inventory_completion_receipt,
)
from scripts.ci_stream_receipts import (  # noqa: E402
    ReceiptFinalizationError,
    convergent_transition_layout,
    exhaustive_coverage_proof,
)
from scripts.nanochat_data.pack_enriched_rows import (  # noqa: E402
    NormalizedDoc,
    normalize_document_record,
    pack_documents,
    rows_to_table,
)


EXPORT_SCHEMA = "cppmega_ci_content_store_case5_export_v2"
PRODUCTION_EXPORT_SCHEMA = "cppmega_ci_content_store_case5_export_v4"
PRODUCTION_MERGE_RECEIPT_SCHEMA = "cppmega_ci_stream_shard_union_receipt_v3"
REPRESENTATIVE_LEDGER_SCHEMA = "cppmega_ci_token_sequence_representative_ledger_v1"
REPRESENTATIVE_METADATA_SCHEMA = "cppmega_ci_case5_representative_metadata_v3"
DERIVED_CLASSIFICATION_SCHEMA = "cppmega_ci_case5_derived_classifications_v2"
OCCURRENCE_METADATA_SCHEMA = "cppmega_ci_case5_occurrence_metadata_v1"
OCCURRENCE_METADATA_LEDGER_DOMAIN = (
    "cppmega-ci-case5-occurrence-metadata-ledger-v1"
)
OPAQUE_ARTIFACT_LEDGER_SCHEMA = "cppmega_ci_case5_excluded_opaque_artifact_v1"
OPAQUE_ARTIFACT_POLICY_SCHEMA = "cppmega_ci_opaque_artifact_policy_v1"
TRAINING_SCOPE_EXCLUSION_LEDGER_SCHEMA = (
    "cppmega_ci_case5_excluded_training_scope_v1"
)
TRAINING_SCOPE_EXCLUSION_LEDGER_DOMAIN = (
    "cppmega-ci-case5-excluded-training-scope-ledger-v1"
)
OPAQUE_INVALID_RATIO_PPM_THRESHOLD = 100_000
OCCURRENCE_SCHEMA = "cppmega_ci_chunk_occurrence_v3"
TRAINING_SIDECAR_SCHEMA = "cppmega_ci_chunk_training_sidecars_v2"
BUCKETS = (1024, 2048, 4096, 8192, 16384)
PARQUET_SHARD_ROWS = 512
PARQUET_SHARD_TOKEN_BUDGET = 16_384
PARQUET_ZSTD_LEVEL = 9
SPLIT_CONTRACT = {
    "schema": "cppmega_ci_token_sequence_split_v1",
    "hash": "token_sequence_sha256",
    "hex_prefix_chars": 16,
    "projection": "int(first-16-lowercase-hex,16)-mod-10000",
    "modulus": 10_000,
    "ranges": {
        "train": [0, 9800],
        "validation": [9800, 9900],
        "test": [9900, 10_000],
    },
}
_SQLITE_NAME = "index.sqlite3"
_PACK_RE = re.compile(r"pack-[0-9]{8}\.cicp")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_RUN_METADATA_SOURCES = {
    "inventory-run-list",
    "github-workflow-run-attempt-api",
}
_SOURCE_LANGUAGE_BY_EXTENSION = {
    ".C": "C++",
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cppm": "C++ module",
    ".cu": "CUDA",
    ".cuh": "CUDA",
    ".cxx": "C++",
    ".h": "C/C++",
    ".hh": "C++",
    ".hpp": "C++",
    ".hxx": "C++",
    ".ixx": "C++ module",
    ".mpp": "C++ module",
}
_BUILD_SYSTEM_BY_TOOL = {
    "autoconf": "autotools",
    "autoheader": "autotools",
    "automake": "autotools",
    "autoreconf": "autotools",
    "bazel": "bazel",
    "bazelisk": "bazel",
    "cmake": "cmake",
    "configure": "autotools",
    "devenv": "msbuild",
    "gmake": "make",
    "gn": "gn",
    "make": "make",
    "meson": "meson",
    "mingw32-make": "make",
    "msbuild": "msbuild",
    "ninja": "ninja",
    "nmake": "make",
    "scons": "scons",
    "xmake": "xmake",
}
_SPAN_RECORD_GROUPS = (
    "entities",
    "commands",
    "build_actions",
    "tests",
    "diagnostics",
)
_EDGE_COLUMN_BY_FAMILY = {
    "domain": TOKEN_DOMAIN_EDGES_COLUMN,
    "build": TOKEN_BUILD_EDGES_COLUMN,
    "shell": TOKEN_SHELL_EDGES_COLUMN,
    "diagnostic": TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    "cross_domain": TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
}


class ExportError(RuntimeError):
    """The frozen input or generated CASE5 output violated its contract."""


def _constrain_evidence_connection(
    connection: sqlite3.Connection,
    *,
    where: str,
) -> None:
    try:
        constrain_sqlite_evidence_rows(connection)
    except ZlibEvidenceError as exc:
        raise ExportError(
            f"{where} SQLite evidence row limit could not be constrained"
        ) from exc


def _require_bounded_fetch_state_evidence(
    connection: sqlite3.Connection,
) -> None:
    violation = fetch_state_evidence_bound_violation(connection)
    if violation is None:
        return
    record_type, repo, run_id, attempt, field = violation
    raise ExportError(
        "fetch-state evidence is not exact and bounded by its versioned "
        "SQLite byte contract: "
        f"{record_type} {repo}#{run_id}/{attempt} {field}"
    )


def _require_bounded_content_store_evidence(
    connection: sqlite3.Connection,
) -> None:
    violation = content_store_evidence_bound_violation(connection)
    if violation is None:
        return
    repo, run_attempt, job, step, chunk_ordinal = violation
    raise ExportError(
        "content-store provenance is not exact and bounded by its versioned "
        "SQLite byte contract: "
        f"{repo}/{run_attempt}/{job}/{step}/{chunk_ordinal}"
    )


@dataclass(frozen=True)
class ContentRecord:
    sha256: str
    raw_size: int
    pack_id: int
    offset: int
    frame_size: int
    compressed_size: int
    token_count: int
    tokenizer_fingerprint: str
    token_sequence_sha256: str


@dataclass(frozen=True)
class OccurrenceRecord:
    key: tuple[str, str, str, str, int]
    content_sha256: str
    provenance_sha256: str
    provenance: dict[str, Any]

    @property
    def key_dict(self) -> dict[str, object]:
        return {
            "repo": self.key[0],
            "run_attempt": self.key[1],
            "job": self.key[2],
            "step": self.key[3],
            "chunk_ordinal": self.key[4],
        }


@dataclass(frozen=True)
class ProjectedContent:
    token_ids: list[int]
    token_spans: list[tuple[int, int]]
    token_domain_ids: list[int]
    token_role_ids: list[int]
    token_entity_ids: list[int]
    token_confidence_ids: list[int]
    edges: list[dict[str, Any]]
    cross_chunk_edges: list[dict[str, Any]]


@dataclass(frozen=True)
class SnapshotFile:
    relative_path: str
    size: int
    mtime_ns: int
    inode: int
    sha256: str


@dataclass(frozen=True)
class FetchMemberEvidence:
    key: tuple[str, int, int, str]
    job_key: str
    job: Mapping[str, Any] | None
    raw_sha256: str
    raw_size: int
    canonical_sha256: str
    dedup_sha256: str
    sidecar_sha256: str
    chunk_count: int
    occurrence_tokens: int
    sidecar: Mapping[str, Any]
    opaque: bool
    exclusion_reason: str | None
    decode_status: str
    invalid_sequence_count: int
    replacement_char_count: int
    invalid_ratio_ppm: int


def _source_binding_projection_writer(
    path: Path,
) -> CanonicalParquetLedgerWriter:
    return CanonicalParquetLedgerWriter(
        path,
        domain=SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
        max_record_bytes=MAX_SOURCE_BINDING_PROJECTION_RECORD_BYTES - 1,
    )


class CanonicalSequenceHasher:
    """Incrementally hash a canonical framed sequence without retaining it."""

    def __init__(self, *, domain: str | None = None):
        self._digest = hashlib.sha256()
        if domain is not None:
            self._digest.update(domain.encode("ascii"))
            self._digest.update(b"\0")
        self.count = 0

    def append(self, value: object) -> None:
        encoded = _canonical_json_bytes(value)
        self._digest.update(len(encoded).to_bytes(8, "big"))
        self._digest.update(encoded)
        self.count += 1

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


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
        raise ExportError(f"value is not canonical JSON: {exc}") from exc


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: object, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExportError(f"{where} must be an object")
    return value


def _require_list(value: object, *, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExportError(f"{where} must be a list")
    return value


def _require_int(
    value: object,
    *,
    where: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExportError(f"{where} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ExportError(f"{where} must be >= {minimum}")
    return result


def _require_nonempty_string(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExportError(f"{where} must be a non-empty string")
    return value


def _require_hex64(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise ExportError(f"{where} must be 64 lowercase hexadecimal characters")
    return value


def _sequence_digest(values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = _canonical_json_bytes(value)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _load_json_object(
    path: Path,
    *,
    where: str,
) -> tuple[dict[str, Any], str]:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise ExportError(f"{where} is missing or unsafe: {path}")
    raw = path.read_bytes()

    def reject_duplicates(
        pairs: Sequence[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ExportError(f"{where} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"{where} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportError(f"{where} must contain one JSON object")
    return value, _sha256_bytes(raw)


def _load_receipt(path: Path) -> tuple[dict[str, Any], str]:
    value, digest = _load_json_object(path, where="store receipt")
    if value.get("schema") != STORE_RECEIPT_SCHEMA:
        raise ExportError(f"store receipt schema must be {STORE_RECEIPT_SCHEMA!r}")
    if value.get("store_schema") != STORE_SCHEMA or value.get("status") != "complete":
        raise ExportError("store receipt is not a completed content-store receipt")
    verification = _require_mapping(
        value.get("verification"), where="store receipt verification"
    )
    if verification.get("mode") != "full" or verification.get("ok") is not True:
        raise ExportError("store receipt does not bind a successful full verification")
    return value, digest


def _verify_exhaustive_export_provenance(
    *,
    store_root: Path,
    store_receipt_path: Path,
    fetch_state_path: Path,
    inventory_path: Path,
    inventory_receipt_path: Path,
    fetch_receipt_path: Path,
    merge_receipt_path: Path,
) -> dict[str, object]:
    """Fail closed unless all production acquisition/merge proofs recompute."""

    try:
        inventory_receipt, inventory_receipt_sha256 = (
            verify_inventory_completion_receipt(
                inventory_path,
                inventory_receipt_path,
                require_production=True,
                expected_original_database_path=inventory_path,
            )
        )
    except InventoryCompletionError as exc:
        raise ExportError(
            f"production inventory provenance refused: {exc}"
        ) from exc
    fetch_receipt, fetch_receipt_sha256 = _load_json_object(
        fetch_receipt_path,
        where="fetch receipt",
    )
    merge_receipt, merge_receipt_sha256 = _load_json_object(
        merge_receipt_path,
        where="merge receipt",
    )
    store_receipt, store_receipt_sha256 = _load_receipt(
        store_receipt_path
    )
    if (
        fetch_receipt.get("schema") != EXHAUSTIVE_RECEIPT_SCHEMA
        or fetch_receipt.get("completion_mode")
        != COMPLETION_MODE_INVENTORY_EXHAUSTIVE
        or fetch_receipt.get("production_complete") is not True
        or fetch_receipt.get("coverage_semantics")
        != "exact-production-inventory-attempt-equality"
    ):
        raise ExportError(
            "production export requires an inventory-exhaustive fetch "
            "receipt v4; threshold-only v3 is non-production"
        )
    if fetch_receipt.get("content_store_receipt") != store_receipt:
        raise ExportError(
            "fetch receipt does not bind the supplied content-store receipt"
        )
    frozen_state = _require_mapping(
        fetch_receipt.get("frozen_fetch_state"),
        where="fetch receipt frozen_fetch_state",
    )
    state_artifact = _require_mapping(
        frozen_state.get("artifact"),
        where="fetch receipt frozen_fetch_state.artifact",
    )
    if (
        state_artifact.get("path") != str(fetch_state_path)
        or state_artifact.get("byte_size") != fetch_state_path.stat().st_size
        or state_artifact.get("sha256") != _sha256_file(fetch_state_path)
    ):
        raise ExportError(
            "frozen fetch-state bytes/path differ from the v4 receipt"
        )
    inventory_binding = _require_mapping(
        fetch_receipt.get("inventory_binding"),
        where="fetch receipt inventory_binding",
    )
    bound_database = _require_mapping(
        inventory_binding.get("database"),
        where="fetch receipt inventory_binding.database",
    )
    bound_inventory_receipt = _require_mapping(
        inventory_binding.get("completion_receipt"),
        where="fetch receipt inventory_binding.completion_receipt",
    )
    inventory_artifact = _require_mapping(
        inventory_receipt.get("database_artifact"),
        where="inventory receipt database_artifact",
    )
    if (
        bound_database.get("path") != str(inventory_path)
        or bound_database.get("sha256")
        != inventory_artifact.get("sha256")
        or bound_database.get("db_logical_sha256")
        != inventory_receipt.get("db_logical_sha256")
        or bound_inventory_receipt.get("path")
        != str(inventory_receipt_path)
        or bound_inventory_receipt.get("sha256")
        != inventory_receipt_sha256
    ):
        raise ExportError(
            "fetch receipt inventory binding differs from the supplied "
            "production inventory/receipt"
        )
    inventory_connection = sqlite3.connect(
        f"{inventory_path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    state_connection = sqlite3.connect(
        f"{fetch_state_path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    inventory_connection.row_factory = sqlite3.Row
    state_connection.row_factory = sqlite3.Row
    try:
        try:
            proof = exhaustive_coverage_proof(
                inventory_connection,
                state_connection,
                inventory_receipt=inventory_receipt,
                require_discovery_eof=False,
            )
        except ReceiptFinalizationError as exc:
            raise ExportError(
                f"fetch-state exhaustive equality proof failed: {exc}"
            ) from exc
    finally:
        state_connection.close()
        inventory_connection.close()
    if fetch_receipt.get("exhaustive_coverage") != proof:
        raise ExportError(
            "fetch receipt exhaustive proof differs from inventory/state"
        )

    if (
        merge_receipt.get("schema")
        != PRODUCTION_MERGE_RECEIPT_SCHEMA
        or merge_receipt.get("status") != "complete"
        or merge_receipt.get("completion_mode")
        != COMPLETION_MODE_INVENTORY_EXHAUSTIVE
        or merge_receipt.get("production_complete") is not True
    ):
        raise ExportError(
            "production export requires a production merge receipt v3"
        )
    verification = _require_mapping(
        merge_receipt.get("verification"),
        where="merge receipt verification",
    )
    if (
        verification.get("exact_production_inventory_attempt_equality")
        is not True
        or verification.get("full_cas_fetch_join") is not True
        or verification.get("destination_frozen") is not True
    ):
        raise ExportError(
            "merge receipt lacks exact inventory/CAS/frozen verification"
        )
    destination = Path(
        _require_nonempty_string(
            merge_receipt.get("destination"),
            where="merge receipt destination",
        )
    ).expanduser().resolve()
    if (
        inventory_path.parent != destination
        or fetch_state_path.parent != destination
        or store_root.parent != destination
        or store_receipt_path.parent != destination
        or fetch_receipt_path.parent != destination
        or merge_receipt_path.parent != destination
    ):
        raise ExportError(
            "production inputs are not the single receipt-bound merge bundle"
        )
    raw_artifacts = merge_receipt.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ExportError("merge receipt artifacts must be a list")
    artifacts: dict[str, Mapping[str, Any]] = {}
    for raw_item in raw_artifacts:
        item = _require_mapping(raw_item, where="merge receipt artifact")
        relative = item.get("path")
        if not isinstance(relative, str) or relative in artifacts:
            raise ExportError(
                "merge receipt has an invalid/duplicate artifact path"
            )
        artifacts[relative] = item
    required_artifacts = {
        "inventory.sqlite3": inventory_path,
        "inventory_receipt.json": inventory_receipt_path,
        "fetch_state.sqlite3": fetch_state_path,
        "store_receipt.json": store_receipt_path,
        "fetch_receipt.json": fetch_receipt_path,
    }
    for relative, actual_path in required_artifacts.items():
        artifact = artifacts.get(relative)
        if (
            artifact is None
            or artifact.get("byte_size") != actual_path.stat().st_size
            or artifact.get("sha256") != _sha256_file(actual_path)
        ):
            raise ExportError(
                f"merge receipt artifact binding differs for {relative}"
            )
    return {
        "completion_mode": COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        "production_complete": True,
        "inventory": {
            "path": str(inventory_path),
            "sha256": inventory_artifact["sha256"],
            "logical_sha256": inventory_receipt["db_logical_sha256"],
            "receipt_path": str(inventory_receipt_path),
            "receipt_sha256": inventory_receipt_sha256,
        },
        "fetch": {
            "state_path": str(fetch_state_path),
            "state_sha256": state_artifact["sha256"],
            "receipt_path": str(fetch_receipt_path),
            "receipt_sha256": fetch_receipt_sha256,
            "attempt_set_sha256": proof["attempt_set_sha256"],
            "terminal_proof_sha256": proof["terminal_proof_sha256"],
        },
        "store": {
            "path": str(store_root),
            "receipt_path": str(store_receipt_path),
            "receipt_sha256": store_receipt_sha256,
        },
        "merge": {
            "receipt_path": str(merge_receipt_path),
            "receipt_sha256": merge_receipt_sha256,
            "schema": PRODUCTION_MERGE_RECEIPT_SCHEMA,
        },
    }


def _strict_zlib_decompress(
    compressed: bytes,
    *,
    expected_size: int,
    expected_sha256: str,
    max_raw_size: int,
    max_compressed_size: int,
    where: str,
) -> bytes:
    try:
        return strict_bounded_zlib_decode(
            compressed,
            expected_raw_size=expected_size,
            expected_sha256=expected_sha256,
            max_raw_size=max_raw_size,
            max_compressed_size=max_compressed_size,
            where=where,
        )
    except ZlibEvidenceError as exc:
        raise ExportError(f"{where} zlib stream is not exact and bounded") from exc


def _decode_canonical_zlib_value(
    compressed: bytes,
    *,
    expected_size: int,
    expected_sha256: str,
    where: str,
    max_raw_size: int = MAX_STATE_JSON_EVIDENCE_BYTES,
    max_compressed_size: int = MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES,
) -> tuple[object, bytes]:
    raw = _strict_zlib_decompress(
        compressed,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        max_raw_size=max_raw_size,
        max_compressed_size=max_compressed_size,
        where=where,
    )
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"{where} is invalid JSON") from exc
    if _canonical_json_bytes(value) != raw:
        raise ExportError(f"{where} is not canonical JSON")
    return value, raw


def _decode_canonical_zlib_mapping(
    compressed: bytes,
    *,
    expected_size: int,
    expected_sha256: str,
    where: str,
    max_raw_size: int = MAX_STATE_JSON_EVIDENCE_BYTES,
    max_compressed_size: int = MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES,
) -> tuple[Mapping[str, Any], bytes]:
    value, raw = _decode_canonical_zlib_value(
        compressed,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        where=where,
        max_raw_size=max_raw_size,
        max_compressed_size=max_compressed_size,
    )
    if not isinstance(value, Mapping):
        raise ExportError(f"{where} is not one canonical JSON object")
    return value, raw


def _decode_provenance(row: sqlite3.Row) -> tuple[dict[str, Any], bytes]:
    value, raw = _decode_canonical_zlib_mapping(
        row["provenance_zlib"],
        expected_size=int(row["provenance_raw_size"]),
        expected_sha256=str(row["provenance_sha256"]),
        where="occurrence provenance",
    )
    return dict(value), raw


def _content_set_digest(connection: sqlite3.Connection) -> str:
    cursor = connection.execute(
        """
        SELECT sha256, raw_size, token_count, tokenizer_fingerprint,
               token_sequence_sha256
        FROM contents
        ORDER BY sha256
        """
    )
    return _hash_records(
        "cppmega-ci-content-set-v1",
        (
            {
                "sha256": str(row["sha256"]),
                "raw_size": int(row["raw_size"]),
                "token_count": (
                    None if row["token_count"] is None else int(row["token_count"])
                ),
                "tokenizer_fingerprint": row["tokenizer_fingerprint"],
                "token_sequence_sha256": row["token_sequence_sha256"],
            }
            for row in cursor
        ),
    )


def _token_sequence_set_digest(connection: sqlite3.Connection) -> str:
    cursor = connection.execute(
        """
        SELECT token_sequence_sha256, token_count, tokenizer_fingerprint
        FROM token_sequences
        ORDER BY token_sequence_sha256
        """
    )
    return _hash_records(
        "cppmega-ci-token-sequence-set-v1",
        (
            {
                "token_sequence_sha256": str(row["token_sequence_sha256"]),
                "token_count": int(row["token_count"]),
                "tokenizer_fingerprint": str(row["tokenizer_fingerprint"]),
                "encoding": TOKEN_SEQUENCE_ENCODING,
            }
            for row in cursor
        ),
    )


def _occurrence_set_digest(connection: sqlite3.Connection) -> str:
    cursor = connection.execute(
        """
        SELECT repo, run_attempt, job, step, chunk_ordinal,
               content_sha256, provenance_sha256,
               provenance_raw_size, provenance_zlib
        FROM occurrences
        ORDER BY repo, run_attempt, job, step, chunk_ordinal
        """
    )

    def records() -> Iterable[object]:
        for row in cursor:
            provenance, _ = _decode_provenance(row)
            yield {
                "repo": str(row["repo"]),
                "run_attempt": str(row["run_attempt"]),
                "job": str(row["job"]),
                "step": str(row["step"]),
                "chunk_ordinal": int(row["chunk_ordinal"]),
                "content_sha256": str(row["content_sha256"]),
                "provenance_sha256": str(row["provenance_sha256"]),
                "provenance": provenance,
            }

    return _hash_records("cppmega-ci-occurrence-set-v1", records())


def _sqlite_logical_digest(connection: sqlite3.Connection) -> str:
    def records() -> Iterable[object]:
        for row in connection.execute("SELECT key, value FROM settings ORDER BY key"):
            yield ["settings", str(row["key"]), str(row["value"])]
        for row in connection.execute(
            """
            SELECT pack_id, filename, committed_end, content_count
            FROM packs ORDER BY pack_id
            """
        ):
            yield [
                "packs",
                int(row["pack_id"]),
                str(row["filename"]),
                int(row["committed_end"]),
                int(row["content_count"]),
            ]
        for row in connection.execute(
            """
            SELECT token_sequence_sha256, token_count, tokenizer_fingerprint
            FROM token_sequences ORDER BY token_sequence_sha256
            """
        ):
            yield [
                "token_sequences",
                str(row["token_sequence_sha256"]),
                int(row["token_count"]),
                str(row["tokenizer_fingerprint"]),
            ]
        for row in connection.execute(
            """
            SELECT sha256, raw_size, pack_id, offset, frame_size,
                   compressed_size, token_count, tokenizer_fingerprint,
                   token_sequence_sha256
            FROM contents ORDER BY sha256
            """
        ):
            yield [
                "contents",
                str(row["sha256"]),
                int(row["raw_size"]),
                int(row["pack_id"]),
                int(row["offset"]),
                int(row["frame_size"]),
                int(row["compressed_size"]),
                None if row["token_count"] is None else int(row["token_count"]),
                row["tokenizer_fingerprint"],
                row["token_sequence_sha256"],
            ]
        for row in connection.execute(
            """
            SELECT repo, run_attempt, job, step, chunk_ordinal,
                   content_sha256, provenance_sha256,
                   provenance_raw_size, provenance_zlib
            FROM occurrences
            ORDER BY repo, run_attempt, job, step, chunk_ordinal
            """
        ):
            yield [
                "occurrences",
                str(row["repo"]),
                str(row["run_attempt"]),
                str(row["job"]),
                str(row["step"]),
                int(row["chunk_ordinal"]),
                str(row["content_sha256"]),
                str(row["provenance_sha256"]),
                int(row["provenance_raw_size"]),
                _sha256_bytes(bytes(row["provenance_zlib"])),
            ]
        row = connection.execute("SELECT * FROM stats WHERE singleton = 1").fetchone()
        if row is None:
            raise ExportError("content-store stats row is missing")
        yield [
            "stats",
            int(row["raw_occurrence_bytes"]),
            int(row["unique_bytes"]),
            int(row["duplicate_bytes"]),
            int(row["unique_content_count"]),
            int(row["occurrence_count"]),
            int(row["tokenized_unique_content_count"]),
            int(row["unique_token_sequence_count"]),
            int(row["exact_unique_payload_tokens"]),
            row["tokenizer_fingerprint"],
        ]

    return _hash_records("cppmega-ci-sqlite-logical-v1", records())


def _require_frozen_sqlite(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ExportError(f"{label} SQLite is missing or unsafe: {path}")
    for suffix in ("-wal", "-journal"):
        pending = Path(f"{path}{suffix}")
        if pending.is_symlink() or (
            pending.exists() and pending.stat().st_size != 0
        ):
            raise ExportError(
                f"{label} is not a frozen SQLite snapshot; found {pending.name}"
            )
    shm = Path(f"{path}-shm")
    if shm.is_symlink() or (shm.exists() and not shm.is_file()):
        raise ExportError(
            f"{label} has an unsafe SQLite sidecar: {shm.name}"
        )


def _open_immutable_sqlite(
    path: Path,
    *,
    label: str,
) -> tuple[sqlite3.Connection, tuple[int, int, int]]:
    stat_before = path.stat()
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    try:
        _constrain_evidence_connection(
            connection,
            where=label,
        )
    except BaseException:
        connection.close()
        raise
    stat_after = path.stat()
    before_identity = (
        stat_before.st_size,
        stat_before.st_mtime_ns,
        stat_before.st_ino,
    )
    after_identity = (
        stat_after.st_size,
        stat_after.st_mtime_ns,
        stat_after.st_ino,
    )
    if before_identity != after_identity:
        connection.close()
        raise ExportError(f"{label} changed while it was opened")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection, after_identity


class FrozenStore:
    """Receipt-bound, immutable/read-only view of a CI content store."""

    def __init__(self, root: Path, receipt_path: Path):
        self.root = root.resolve()
        if root.is_symlink() or not self.root.is_dir():
            raise ExportError(f"content store is missing or unsafe: {root}")
        self.receipt_path = receipt_path.expanduser()
        self.receipt, self.receipt_sha256 = _load_receipt(self.receipt_path)
        self.db_path = self.root / _SQLITE_NAME
        if self.db_path.is_symlink() or not self.db_path.is_file():
            raise ExportError(
                f"store SQLite index is missing or unsafe: {self.db_path}"
            )
        _require_frozen_sqlite(self.db_path, label="store")
        self.connection, self._opened_db_identity = _open_immutable_sqlite(
            self.db_path,
            label="store SQLite index",
        )
        self.pack_paths: dict[int, Path] = {}
        self._pack_fds: OrderedDict[int, int] = OrderedDict()
        self._verified_snapshots: dict[str, SnapshotFile] = {}
        self._initial_snapshot: tuple[SnapshotFile, ...] = ()

    def __enter__(self) -> "FrozenStore":
        try:
            before = self.snapshot_files(include_hashes=False)
            opened_index = next(
                (item for item in before if item.relative_path == _SQLITE_NAME),
                None,
            )
            if (
                opened_index is None
                or (
                    opened_index.size,
                    opened_index.mtime_ns,
                    opened_index.inode,
                )
                != self._opened_db_identity
            ):
                raise ExportError(
                    "store SQLite index differs from the immutable connection"
                )
            self.verify()
            after = self.snapshot_files(reuse_verified=True)
            before_metadata = [
                (item.relative_path, item.size, item.mtime_ns, item.inode)
                for item in before
            ]
            after_metadata = [
                (item.relative_path, item.size, item.mtime_ns, item.inode)
                for item in after
            ]
            if before_metadata != after_metadata:
                raise ExportError("content store changed during initial verification")
            self._initial_snapshot = after
        except BaseException:
            self._close_pack_fds()
            self.connection.close()
            raise
        return self

    def __exit__(self, *_args: object) -> None:
        self._close_pack_fds()
        self.connection.close()

    def _close_pack_fds(self) -> None:
        for descriptor in self._pack_fds.values():
            os.close(descriptor)
        self._pack_fds.clear()

    def _pack_fd(self, pack_id: int, path: Path) -> int:
        descriptor = self._pack_fds.pop(pack_id, None)
        if descriptor is not None:
            self._pack_fds[pack_id] = descriptor
            return descriptor
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        self._pack_fds[pack_id] = descriptor
        if len(self._pack_fds) > 16:
            _evicted_pack_id, evicted_descriptor = self._pack_fds.popitem(last=False)
            os.close(evicted_descriptor)
        return descriptor

    def _settings(self) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in self.connection.execute(
                "SELECT key, value FROM settings ORDER BY key"
            )
        }

    def _pack_receipt_by_name(self) -> dict[str, Mapping[str, Any]]:
        raw = _require_list(self.receipt.get("pack_hashes"), where="pack_hashes")
        result: dict[str, Mapping[str, Any]] = {}
        for index, item in enumerate(raw):
            record = _require_mapping(item, where=f"pack_hashes[{index}]")
            filename = _require_nonempty_string(
                record.get("filename"), where=f"pack_hashes[{index}].filename"
            )
            if filename in result:
                raise ExportError(f"duplicate pack receipt for {filename}")
            result[filename] = record
        return result

    def _verify_packs(self) -> None:
        receipt_by_name = self._pack_receipt_by_name()
        rows = self.connection.execute(
            """
            SELECT pack_id, filename, committed_end, content_count
            FROM packs ORDER BY pack_id
            """
        ).fetchall()
        actual_names = {str(row["filename"]) for row in rows}
        if actual_names != set(receipt_by_name):
            raise ExportError("receipt pack set differs from the SQLite pack set")
        disk_names = {
            path.name
            for path in self.root.iterdir()
            if path.is_file() and _PACK_RE.fullmatch(path.name)
        }
        if disk_names != actual_names:
            raise ExportError("disk pack set differs from the frozen receipt")

        for pack in rows:
            pack_id = int(pack["pack_id"])
            filename = str(pack["filename"])
            if filename != f"pack-{pack_id:08d}.cicp":
                raise ExportError(f"unsafe pack filename: {filename!r}")
            path = self.root / filename
            if path.is_symlink() or not path.is_file():
                raise ExportError(f"pack is missing or unsafe: {filename}")
            committed_end = int(pack["committed_end"])
            stat_before = path.stat()
            if stat_before.st_size != committed_end:
                raise ExportError(f"{filename} size differs from committed_end")
            expected = receipt_by_name[filename]
            pack_sha256 = _sha256_file(path)
            stat_after = path.stat()
            if (
                stat_before.st_size,
                stat_before.st_mtime_ns,
                stat_before.st_ino,
            ) != (
                stat_after.st_size,
                stat_after.st_mtime_ns,
                stat_after.st_ino,
            ):
                raise ExportError(f"{filename} changed while it was hashed")
            if (
                _require_int(
                    expected.get("committed_end"),
                    where=f"{filename}.committed_end",
                )
                != committed_end
                or _require_int(
                    expected.get("content_count"),
                    where=f"{filename}.content_count",
                )
                != int(pack["content_count"])
                or _require_hex64(expected.get("sha256"), where=f"{filename}.sha256")
                != pack_sha256
            ):
                raise ExportError(f"{filename} differs from the frozen receipt")
            with path.open("rb") as handle:
                if handle.read(len(_PACK_MAGIC)) != _PACK_MAGIC:
                    raise ExportError(f"{filename} has an invalid pack header")
            self.pack_paths[pack_id] = path
            self._verified_snapshots[filename] = SnapshotFile(
                relative_path=filename,
                size=stat_after.st_size,
                mtime_ns=stat_after.st_mtime_ns,
                inode=stat_after.st_ino,
                sha256=pack_sha256,
            )

            expected_offset = len(_PACK_MAGIC)
            count = 0
            content_rows = self.connection.execute(
                "SELECT * FROM contents WHERE pack_id = ? ORDER BY offset",
                (pack_id,),
            )
            for content_row in content_rows:
                raw_size = int(content_row["raw_size"])
                compressed_size = int(content_row["compressed_size"])
                frame_size = int(content_row["frame_size"])
                if (
                    raw_size < 0
                    or raw_size > MAX_CONTENT_FRAME_BYTES
                    or compressed_size < 0
                    or compressed_size
                    > MAX_CONTENT_FRAME_COMPRESSED_BYTES
                ):
                    raise ExportError(
                        f"{filename} frame exceeds its semantic byte bounds"
                    )
                if (
                    int(content_row["offset"]) != expected_offset
                    or frame_size != _FRAME_HEADER.size + compressed_size
                ):
                    raise ExportError(f"{filename} has a frame gap or overlap")
                expected_offset += frame_size
                count += 1
            if expected_offset != committed_end or count != int(pack["content_count"]):
                raise ExportError(f"{filename} frame accounting is inconsistent")

    def _verify_counters(self) -> None:
        aggregate = self.connection.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM contents) AS unique_content_count,
              (SELECT COALESCE(SUM(raw_size), 0) FROM contents) AS unique_bytes,
              (SELECT COUNT(*) FROM occurrences) AS occurrence_count,
              (
                SELECT COALESCE(SUM(contents.raw_size), 0)
                FROM occurrences
                JOIN contents ON contents.sha256 = occurrences.content_sha256
              ) AS raw_occurrence_bytes,
              (
                SELECT COUNT(*) FROM contents
                WHERE token_sequence_sha256 IS NOT NULL
              ) AS tokenized_unique_content_count,
              (
                SELECT COUNT(DISTINCT token_sequence_sha256)
                FROM contents
                WHERE token_sequence_sha256 IS NOT NULL
              ) AS referenced_unique_token_sequence_count,
              (SELECT COUNT(*) FROM token_sequences)
                AS unique_token_sequence_count,
              (SELECT COALESCE(SUM(token_count), 0) FROM token_sequences)
                AS exact_unique_payload_tokens
            """
        ).fetchone()
        stats = self.connection.execute(
            "SELECT * FROM stats WHERE singleton = 1"
        ).fetchone()
        if aggregate is None or stats is None:
            raise ExportError("content-store aggregate or stats row is missing")
        expected = {
            "raw_occurrence_bytes": int(aggregate["raw_occurrence_bytes"]),
            "unique_bytes": int(aggregate["unique_bytes"]),
            "duplicate_bytes": (
                int(aggregate["raw_occurrence_bytes"]) - int(aggregate["unique_bytes"])
            ),
            "unique_content_count": int(aggregate["unique_content_count"]),
            "occurrence_count": int(aggregate["occurrence_count"]),
            "tokenized_unique_content_count": int(
                aggregate["tokenized_unique_content_count"]
            ),
            "unique_token_sequence_count": int(
                aggregate["unique_token_sequence_count"]
            ),
            "exact_unique_payload_tokens": int(
                aggregate["exact_unique_payload_tokens"]
            ),
        }
        receipt_counters = _require_mapping(
            self.receipt.get("counters"), where="store receipt counters"
        )
        for field, value in expected.items():
            if int(stats[field]) != value or int(receipt_counters[field]) != value:
                raise ExportError(f"content-store counter mismatch for {field}")
        if (
            expected["tokenized_unique_content_count"]
            != expected["unique_content_count"]
        ):
            raise ExportError("not every unique content has exact token metadata")
        if (
            int(aggregate["referenced_unique_token_sequence_count"])
            != expected["unique_token_sequence_count"]
        ):
            raise ExportError("token_sequences contains an orphan record")
        if (
            receipt_counters.get("exact_unique_payload_tokens")
            != expected["exact_unique_payload_tokens"]
        ):
            raise ExportError("receipt has no exact unique payload-token total")
        inconsistent_token_binding = self.connection.execute(
            """
            SELECT contents.sha256
            FROM contents
            JOIN token_sequences
              ON token_sequences.token_sequence_sha256 =
                 contents.token_sequence_sha256
            WHERE contents.token_count != token_sequences.token_count
               OR contents.tokenizer_fingerprint !=
                  token_sequences.tokenizer_fingerprint
            LIMIT 1
            """
        ).fetchone()
        if inconsistent_token_binding is not None:
            raise ExportError("content token metadata disagrees with token_sequences")

    def verify(self) -> None:
        integrity = [
            str(row[0])
            for row in self.connection.execute("PRAGMA integrity_check").fetchall()
        ]
        if integrity != ["ok"]:
            raise ExportError(f"SQLite integrity_check failed: {integrity}")
        if self.connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ExportError("SQLite foreign_key_check failed")
        settings = self._settings()
        if settings.get("schema") != STORE_SCHEMA:
            raise ExportError("SQLite store schema is missing or stale")
        try:
            policy = json.loads(settings["policy"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ExportError("SQLite store policy is invalid") from exc
        if policy != self.receipt.get("policy"):
            raise ExportError("SQLite store policy differs from the receipt")
        policy_sha = _sha256_bytes(_canonical_json_bytes(policy))
        if policy_sha != self.receipt.get("policy_sha256"):
            raise ExportError("store policy SHA-256 differs from the receipt")
        if settings.get("creator_script_sha256") != self.receipt.get("script_sha256"):
            raise ExportError("store creator script differs from the receipt")
        schema_sha = _sqlite_schema_sha256(self.connection)
        if (
            settings.get("sqlite_schema_sha256") != schema_sha
            or self.receipt.get("sqlite_schema_sha256") != schema_sha
        ):
            raise ExportError("SQLite schema SHA-256 differs from the receipt")
        _require_bounded_content_store_evidence(self.connection)
        expected_digests = {
            "logical_content_set_sha256": _content_set_digest(self.connection),
            "logical_token_sequence_set_sha256": _token_sequence_set_digest(
                self.connection
            ),
            "occurrence_set_sha256": _occurrence_set_digest(self.connection),
            "sqlite_logical_sha256": _sqlite_logical_digest(self.connection),
        }
        for field, digest in expected_digests.items():
            if self.receipt.get(field) != digest:
                raise ExportError(f"store receipt digest mismatch for {field}")
        self._verify_counters()
        self._verify_packs()

    def snapshot_files(
        self,
        *,
        include_hashes: bool = True,
        reuse_verified: bool = False,
    ) -> tuple[SnapshotFile, ...]:
        paths: list[Path] = []
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ExportError(f"frozen store contains a symlink: {path}")
            if path.is_file():
                paths.append(path)
        records: list[SnapshotFile] = []
        for path in sorted(paths):
            stat = path.stat()
            relative_path = path.relative_to(self.root).as_posix()
            verified = self._verified_snapshots.get(relative_path)
            can_reuse = (
                reuse_verified
                and verified is not None
                and (
                    verified.size,
                    verified.mtime_ns,
                    verified.inode,
                )
                == (stat.st_size, stat.st_mtime_ns, stat.st_ino)
            )
            records.append(
                SnapshotFile(
                    relative_path=relative_path,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    inode=stat.st_ino,
                    sha256=(
                        ""
                        if not include_hashes
                        else verified.sha256
                        if can_reuse
                        else _sha256_file(path)
                    ),
                )
            )
        return tuple(records)

    def require_unchanged(self) -> None:
        if (
            self.receipt_path.is_symlink()
            or not self.receipt_path.is_file()
            or _sha256_file(self.receipt_path) != self.receipt_sha256
        ):
            raise ExportError("content-store receipt changed during export")
        _require_frozen_sqlite(self.db_path, label="content store")
        expected_packs = {path.name for path in self.pack_paths.values()}
        actual_packs = {
            path.name
            for path in self.root.iterdir()
            if path.is_file() and _PACK_RE.fullmatch(path.name)
        }
        if actual_packs != expected_packs:
            raise ExportError("content-store pack set changed during export")
        if self.snapshot_files() != self._initial_snapshot:
            raise ExportError("content store changed while export was running")

    @staticmethod
    def _content_record(row: sqlite3.Row) -> ContentRecord:
        values = (
            row["token_count"],
            row["tokenizer_fingerprint"],
            row["token_sequence_sha256"],
        )
        if any(value is None for value in values):
            raise ExportError(
                f"content {row['sha256']} is missing exact token metadata"
            )
        return ContentRecord(
            sha256=str(row["sha256"]),
            raw_size=int(row["raw_size"]),
            pack_id=int(row["pack_id"]),
            offset=int(row["offset"]),
            frame_size=int(row["frame_size"]),
            compressed_size=int(row["compressed_size"]),
            token_count=int(row["token_count"]),
            tokenizer_fingerprint=str(row["tokenizer_fingerprint"]),
            token_sequence_sha256=str(row["token_sequence_sha256"]),
        )

    def iter_contents(
        self, *, by_token_sequence: bool = False
    ) -> Iterable[ContentRecord]:
        order = "token_sequence_sha256, sha256" if by_token_sequence else "sha256"
        cursor = self.connection.execute(f"SELECT * FROM contents ORDER BY {order}")
        for row in cursor:
            yield self._content_record(row)

    def get_content_record(self, sha256: str) -> ContentRecord:
        row = self.connection.execute(
            "SELECT * FROM contents WHERE sha256 = ?", (sha256,)
        ).fetchone()
        if row is None:
            raise ExportError(f"content disappeared: {sha256}")
        return self._content_record(row)

    def read_content(self, record: ContentRecord) -> bytes:
        row = self.connection.execute(
            "SELECT * FROM contents WHERE sha256 = ?", (record.sha256,)
        ).fetchone()
        if row is None:
            raise ExportError(f"content disappeared: {record.sha256}")
        return self.read_content_row(row)

    def read_content_row(self, row: sqlite3.Row) -> bytes:
        pack_id = int(row["pack_id"])
        path = self.pack_paths.get(pack_id)
        if path is None:
            filename_row = self.connection.execute(
                "SELECT filename FROM packs WHERE pack_id = ?", (pack_id,)
            ).fetchone()
            if filename_row is None:
                raise ExportError(f"content references unknown pack {pack_id}")
            path = self.root / str(filename_row["filename"])
        offset = int(row["offset"])
        descriptor = self._pack_fd(pack_id, path)
        header = os.pread(descriptor, _FRAME_HEADER.size, offset)
        if len(header) != _FRAME_HEADER.size:
            raise ExportError("content frame header is truncated")
        magic, digest_bytes, raw_size, compressed_size = _FRAME_HEADER.unpack(header)
        if (
            raw_size > MAX_CONTENT_FRAME_BYTES
            or compressed_size > MAX_CONTENT_FRAME_COMPRESSED_BYTES
        ):
            raise ExportError(
                "content frame exceeds the versioned raw/compressed byte bound"
            )
        compressed = os.pread(
            descriptor,
            compressed_size,
            offset + _FRAME_HEADER.size,
        )
        if magic != _FRAME_MAGIC or len(compressed) != compressed_size:
            raise ExportError("content frame is malformed or truncated")
        if int(row["frame_size"]) != _FRAME_HEADER.size + compressed_size:
            raise ExportError("content frame size differs from SQLite")
        if (
            int(row["compressed_size"]) != compressed_size
            or int(row["raw_size"]) != raw_size
            or bytes.fromhex(str(row["sha256"])) != digest_bytes
        ):
            raise ExportError("content frame metadata differs from SQLite")
        content = _strict_zlib_decompress(
            compressed,
            expected_size=raw_size,
            expected_sha256=str(row["sha256"]),
            max_raw_size=MAX_CONTENT_FRAME_BYTES,
            max_compressed_size=MAX_CONTENT_FRAME_COMPRESSED_BYTES,
            where="content frame",
        )
        return content

    @staticmethod
    def _occurrence_record(row: sqlite3.Row) -> OccurrenceRecord:
        provenance, _ = _decode_provenance(row)
        return OccurrenceRecord(
            key=(
                str(row["repo"]),
                str(row["run_attempt"]),
                str(row["job"]),
                str(row["step"]),
                int(row["chunk_ordinal"]),
            ),
            content_sha256=str(row["content_sha256"]),
            provenance_sha256=str(row["provenance_sha256"]),
            provenance=provenance,
        )

    def iter_occurrences(self) -> Iterable[OccurrenceRecord]:
        cursor = self.connection.execute(
            """
            SELECT repo, run_attempt, job, step, chunk_ordinal,
                   content_sha256, provenance_sha256,
                   provenance_raw_size, provenance_zlib
            FROM occurrences
            ORDER BY repo, run_attempt, job, step, chunk_ordinal
            """
        )
        for row in cursor:
            yield self._occurrence_record(row)

    def iter_occurrences_for_content(
        self, content_sha256: str
    ) -> Iterable[OccurrenceRecord]:
        cursor = self.connection.execute(
            """
            SELECT repo, run_attempt, job, step, chunk_ordinal,
                   content_sha256, provenance_sha256,
                   provenance_raw_size, provenance_zlib
            FROM occurrences
            WHERE content_sha256 = ?
            ORDER BY repo, run_attempt, job, step, chunk_ordinal
            """,
            (content_sha256,),
        )
        for row in cursor:
            yield self._occurrence_record(row)

    def get_occurrence(
        self,
        key: tuple[str, str, str, str, int],
    ) -> OccurrenceRecord:
        row = self.connection.execute(
            """
            SELECT repo, run_attempt, job, step, chunk_ordinal,
                   content_sha256, provenance_sha256,
                   provenance_raw_size, provenance_zlib
            FROM occurrences
            WHERE repo = ? AND run_attempt = ? AND job = ?
              AND step = ? AND chunk_ordinal = ?
            """,
            key,
        ).fetchone()
        if row is None:
            raise ExportError(f"occurrence disappeared: {key}")
        return self._occurrence_record(row)


def _expected_fetch_state_schema_sha256() -> str:
    connection = sqlite3.connect(":memory:")
    try:
        connection.row_factory = sqlite3.Row
        connection.executescript(FETCH_STATE_SQL_SCHEMA)
        return _sqlite_schema_sha256(connection)
    finally:
        connection.close()


def _fetch_state_logical_digest(connection: sqlite3.Connection) -> str:
    table_order = (
        ("settings", "key"),
        ("attempts", "repo,run_id,attempt"),
        ("members", "repo,run_id,attempt,archive_member"),
        ("request_ledger", "id"),
        ("binding_upgrades", "id"),
    )

    def records() -> Iterable[list[object]]:
        for table, order_by in table_order:
            for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order_by}"):
                values: list[object] = [table]
                for value in row:
                    if isinstance(value, bytes):
                        values.append(
                            {
                                "byte_size": len(value),
                                "sha256": _sha256_bytes(value),
                            }
                        )
                    else:
                        values.append(value)
                yield values

    return _hash_records("cppmega-ci-fetch-state-logical-v1", records())


def _decode_fetch_run_metadata(
    row: sqlite3.Row,
    *,
    key: tuple[str, int, int],
) -> Mapping[str, Any]:
    compressed = row["run_metadata_zlib"]
    if not isinstance(compressed, (bytes, bytearray, memoryview)):
        raise ExportError(f"fetch-state attempt {key} run_metadata_zlib is not a BLOB")
    metadata, _ = _decode_canonical_zlib_mapping(
        compressed,
        expected_size=_require_int(
            row["run_metadata_raw_size"],
            where=f"fetch-state attempt {key} run_metadata_raw_size",
            minimum=0,
        ),
        expected_sha256=_require_hex64(
            row["run_metadata_sha256"],
            where=f"fetch-state attempt {key} run_metadata_sha256",
        ),
        where=f"fetch-state attempt {key} metadata",
        max_raw_size=MAX_RUN_METADATA_BYTES,
        max_compressed_size=MAX_RUN_METADATA_COMPRESSED_BYTES,
    )
    if int(row["run_metadata_exact"]) != 1:
        raise ExportError(f"fetch-state attempt {key} run metadata is not marked exact")
    source = str(row["run_metadata_source"])
    source_attempt = _require_int(
        row["run_metadata_source_attempt"],
        where=f"fetch-state attempt {key} run_metadata_source_attempt",
        minimum=1,
    )
    seed_attempt = _require_int(
        row["inventory_seed_attempt"],
        where=f"fetch-state attempt {key} inventory_seed_attempt",
        minimum=1,
    )
    metadata_sha256 = _require_hex64(
        row["run_metadata_sha256"],
        where=f"fetch-state attempt {key} run_metadata_sha256",
    )
    seed_sha256 = _require_hex64(
        row["inventory_seed_metadata_sha256"],
        where=f"fetch-state attempt {key} inventory_seed_metadata_sha256",
    )
    if source not in _RUN_METADATA_SOURCES or source_attempt != key[2]:
        raise ExportError(
            f"fetch-state attempt {key} exact metadata source is inconsistent"
        )
    if source == "inventory-run-list" and (
        seed_attempt != key[2] or seed_sha256 != metadata_sha256
    ):
        raise ExportError(
            f"fetch-state attempt {key} inventory metadata binding is inconsistent"
        )
    if source == "github-workflow-run-attempt-api" and seed_attempt <= key[2]:
        raise ExportError(
            f"fetch-state attempt {key} API metadata lacks a newer inventory seed"
        )
    try:
        _validate_run_metadata_identity(
            metadata,
            run_id=key[1],
            attempt=key[2],
        )
    except MalformedResponseError as exc:
        raise ExportError(
            f"fetch-state attempt {key} run metadata identity is inconsistent"
        ) from exc
    return metadata


def _decode_fetch_jobs(
    row: sqlite3.Row,
    *,
    key: tuple[str, int, int],
) -> tuple[Mapping[str, Any], ...]:
    jobs_fields = (
        row["jobs_sha256"],
        row["jobs_raw_size"],
        row["jobs_zlib"],
    )
    fields_present = tuple(value is not None for value in jobs_fields)
    jobs_required = (
        str(row["status"]) in {"done", "empty"} or int(row["member_count"]) > 0
    )
    if not any(fields_present):
        if jobs_required:
            raise ExportError(
                f"fetch-state attempt {key} exact jobs evidence is missing"
            )
        return ()
    if not all(fields_present):
        raise ExportError(
            f"fetch-state attempt {key} jobs evidence is only partially present"
        )
    compressed = row["jobs_zlib"]
    if not isinstance(compressed, (bytes, bytearray, memoryview)):
        raise ExportError(f"fetch-state attempt {key} jobs_zlib is not a BLOB")
    value, _ = _decode_canonical_zlib_value(
        compressed,
        expected_size=_require_int(
            row["jobs_raw_size"],
            where=f"fetch-state attempt {key} jobs_raw_size",
            minimum=0,
        ),
        expected_sha256=_require_hex64(
            row["jobs_sha256"],
            where=f"fetch-state attempt {key} jobs_sha256",
        ),
        where=f"fetch-state attempt {key} jobs",
        max_raw_size=MAX_JOBS_EVIDENCE_BYTES,
        max_compressed_size=MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
    )
    if not isinstance(value, list):
        raise ExportError(f"fetch-state attempt {key} jobs must be a JSON list")
    jobs: list[Mapping[str, Any]] = []
    seen_ids: set[int] = set()
    for index, raw_job in enumerate(value):
        if not isinstance(raw_job, Mapping):
            raise ExportError(
                f"fetch-state attempt {key} jobs[{index}] must be an object"
            )
        job_id = _require_int(
            raw_job.get("id"),
            where=f"fetch-state attempt {key} jobs[{index}].id",
            minimum=1,
        )
        if job_id in seen_ids:
            raise ExportError(
                f"fetch-state attempt {key} jobs contain duplicate id {job_id}"
            )
        seen_ids.add(job_id)
        jobs.append(dict(raw_job))
    return tuple(jobs)


class FrozenFetchState:
    """Immutable parser-sidecar evidence joined to every CAS occurrence."""

    def __init__(
        self,
        path: Path,
        *,
        tokenizer: ExactTokenizer,
        store: FrozenStore,
        bound_store_path: Path | None = None,
    ):
        candidate = path.expanduser()
        self.path = candidate.resolve()
        if candidate.is_symlink() or not self.path.is_file():
            raise ExportError(f"fetch-state snapshot is missing or unsafe: {path}")
        _require_frozen_sqlite(self.path, label="fetch state")
        self.connection, self._opened_identity = _open_immutable_sqlite(
            self.path,
            label="fetch-state SQLite",
        )
        self.tokenizer = tokenizer
        self.store = store
        self.bound_store_path = (
            store.root
            if bound_store_path is None
            else bound_store_path.expanduser().resolve()
        )
        self.settings: dict[str, str] = {}
        self.summary: dict[str, object] = {}
        self.sqlite_schema_sha256 = ""
        self.sqlite_logical_sha256 = ""
        self.sidecar_set_sha256 = ""
        self._snapshot: SnapshotFile | None = None
        self._member_cache: OrderedDict[
            tuple[str, int, int, str], FetchMemberEvidence
        ] = OrderedDict()
        self._attempt_cache: OrderedDict[
            tuple[str, int, int],
            tuple[
                sqlite3.Row,
                Mapping[str, Any],
                tuple[Mapping[str, Any], ...],
            ],
        ] = OrderedDict()

    def __enter__(self) -> "FrozenFetchState":
        try:
            stat_before = self.path.stat()
            current_identity = (
                stat_before.st_size,
                stat_before.st_mtime_ns,
                stat_before.st_ino,
            )
            if current_identity != self._opened_identity:
                raise ExportError(
                    "fetch-state SQLite differs from the immutable connection"
                )
            sha256 = _sha256_file(self.path)
            stat_after = self.path.stat()
            final_identity = (
                stat_after.st_size,
                stat_after.st_mtime_ns,
                stat_after.st_ino,
            )
            if final_identity != current_identity:
                raise ExportError("fetch-state SQLite changed while it was hashed")
            self._snapshot = SnapshotFile(
                relative_path=self.path.name,
                size=stat_after.st_size,
                mtime_ns=stat_after.st_mtime_ns,
                inode=stat_after.st_ino,
                sha256=sha256,
            )
            self.verify()
        except BaseException:
            self.connection.close()
            raise
        return self

    def __exit__(self, *_args: object) -> None:
        self.connection.close()

    def verify(self) -> None:
        integrity = [
            str(row[0])
            for row in self.connection.execute("PRAGMA integrity_check").fetchall()
        ]
        if integrity != ["ok"]:
            raise ExportError(f"fetch-state integrity_check failed: {integrity}")
        if self.connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ExportError("fetch-state foreign_key_check failed")
        self.sqlite_schema_sha256 = _sqlite_schema_sha256(self.connection)
        if self.sqlite_schema_sha256 != _expected_fetch_state_schema_sha256():
            raise ExportError("fetch-state SQLite schema is not the frozen v4 schema")
        _require_bounded_fetch_state_evidence(self.connection)
        self.settings = {
            str(row["key"]): str(row["value"])
            for row in self.connection.execute(
                "SELECT key,value FROM settings ORDER BY key"
            )
        }
        expected_setting_keys = {
            "schema",
            "inventory_path",
            "content_store_path",
            "tokenizer_contract",
            "tokenizer_fingerprint",
            "fetcher_script_sha256",
            "parser_script_sha256",
            "content_store_script_sha256",
            "chunk_semantics",
            "created_at",
        }
        if set(self.settings) != expected_setting_keys:
            raise ExportError(
                "fetch-state settings do not match the frozen v4 contract"
            )
        if self.settings["schema"] != FETCH_STATE_SCHEMA:
            raise ExportError("fetch-state schema setting is unsupported")
        try:
            tokenizer_contract = json.loads(self.settings["tokenizer_contract"])
        except json.JSONDecodeError as exc:
            raise ExportError("fetch-state tokenizer contract is invalid JSON") from exc
        if (
            tokenizer_contract != self.tokenizer.contract
            or self.settings["tokenizer_fingerprint"] != self.tokenizer.fingerprint
        ):
            raise ExportError("fetch-state tokenizer binding differs from the export")
        if (
            Path(self.settings["content_store_path"]).resolve()
            != self.bound_store_path
        ):
            raise ExportError("fetch-state content-store path binding is inconsistent")
        if self.settings["content_store_script_sha256"] != self.store.receipt.get(
            "script_sha256"
        ):
            raise ExportError(
                "fetch-state content-store script binding is inconsistent"
            )
        for field in (
            "fetcher_script_sha256",
            "parser_script_sha256",
            "content_store_script_sha256",
        ):
            _require_hex64(self.settings[field], where=f"fetch-state setting {field}")
        if self.settings["chunk_semantics"] != (
            "parser-dedup-text-cppmega-training-tokenizer-payload-only-no-framing-v2"
        ):
            raise ExportError("fetch-state chunk semantics are unsupported")

        status_counts = {
            str(row["status"]): int(row["n"])
            for row in self.connection.execute(
                "SELECT status,COUNT(*) AS n FROM attempts GROUP BY status"
            )
        }
        for row in self.connection.execute(
            """
            SELECT
              attempts.repo,
              attempts.run_id,
              attempts.attempt,
              attempts.member_count,
              attempts.chunk_count,
              attempts.occurrence_tokens,
              COUNT(members.archive_member) AS actual_member_count,
              COALESCE(SUM(members.chunk_count), 0) AS actual_chunk_count,
              COALESCE(SUM(members.occurrence_tokens), 0)
                AS actual_occurrence_tokens
            FROM attempts
            LEFT JOIN members
              ON members.repo = attempts.repo
             AND members.run_id = attempts.run_id
             AND members.attempt = attempts.attempt
            GROUP BY
              attempts.repo,
              attempts.run_id,
              attempts.attempt,
              attempts.member_count,
              attempts.chunk_count,
              attempts.occurrence_tokens
            ORDER BY attempts.repo, attempts.run_id, attempts.attempt
            """
        ):
            if (
                int(row["member_count"]) != int(row["actual_member_count"])
                or int(row["chunk_count"]) != int(row["actual_chunk_count"])
                or int(row["occurrence_tokens"]) != int(row["actual_occurrence_tokens"])
            ):
                raise ExportError(
                    "fetch-state per-attempt member accounting is inconsistent: "
                    f"{row['repo']}/{row['run_id']}/{row['attempt']}"
                )
        for row in self.connection.execute(
            """
            SELECT * FROM attempts
            ORDER BY repo,run_id,attempt
            """
        ):
            key = (
                str(row["repo"]),
                int(row["run_id"]),
                int(row["attempt"]),
            )
            if str(row["status"]) in {"done", "empty"}:
                _decode_fetch_run_metadata(row, key=key)
            _decode_fetch_jobs(
                row,
                key=key,
            )
            if str(row["status"]) == "empty":
                member_count = int(row["member_count"])
                if (
                    int(row["chunk_count"]) != 0
                    or int(row["occurrence_tokens"]) != 0
                    or member_count < 0
                    or (member_count == 0) != (row["archive_zlib"] is not None)
                ):
                    raise ExportError(
                        "fetch-state empty proof modes are inconsistent: "
                        f"{key}"
                    )
                if member_count:
                    nonempty_member = self.connection.execute(
                        """
                        SELECT archive_member FROM members
                        WHERE repo=? AND run_id=? AND attempt=?
                          AND (chunk_count!=0 OR occurrence_tokens!=0)
                        LIMIT 1
                        """,
                        key,
                    ).fetchone()
                    if nonempty_member is not None:
                        raise ExportError(
                            "fetch-state parsed-empty member retains training "
                            f"content: {key}"
                        )
        totals = self.connection.execute(
            """
            SELECT COUNT(*) AS attempts,
                   COALESCE(SUM(member_count),0) AS members,
                   COALESCE(SUM(chunk_count),0) AS chunks,
                   COALESCE(SUM(occurrence_tokens),0) AS occurrence_tokens
            FROM attempts
            WHERE status IN ('done','empty','terminal_404','terminal_410')
            """
        ).fetchone()
        if totals is None:
            raise ExportError("fetch-state summary aggregate is missing")
        member_totals = self.connection.execute(
            """
            SELECT COUNT(*) AS members,
                   COALESCE(SUM(chunk_count),0) AS chunks,
                   COALESCE(SUM(occurrence_tokens),0) AS occurrence_tokens
            FROM members
            """
        ).fetchone()
        if member_totals is None or (
            int(member_totals["members"]) != int(totals["members"])
            or int(member_totals["chunks"]) != int(totals["chunks"])
            or int(member_totals["occurrence_tokens"])
            != int(totals["occurrence_tokens"])
        ):
            raise ExportError("fetch-state attempt/member accounting is inconsistent")
        sidecar_digest = hashlib.sha256()
        for row in self.connection.execute(
            """
            SELECT repo,run_id,attempt,archive_member,sidecar_sha256
            FROM members
            ORDER BY repo,run_id,attempt,archive_member
            """
        ):
            sidecar_digest.update(
                (
                    f"{row['repo']}\t{row['run_id']}\t{row['attempt']}\t"
                    f"{row['archive_member']}\t{row['sidecar_sha256']}\n"
                ).encode("utf-8")
            )
        self.sidecar_set_sha256 = sidecar_digest.hexdigest()
        self.sqlite_logical_sha256 = _fetch_state_logical_digest(self.connection)
        self.summary = {
            "attempt_statuses": dict(sorted(status_counts.items())),
            "attempts_terminal": int(totals["attempts"]),
            "members": int(totals["members"]),
            "chunks": int(totals["chunks"]),
            "occurrence_tokens": int(totals["occurrence_tokens"]),
            "requests": int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM request_ledger"
                ).fetchone()[0]
            ),
            "sidecar_set_sha256": self.sidecar_set_sha256,
        }

    def _load_attempt(
        self,
        key: tuple[str, int, int],
    ) -> tuple[
        sqlite3.Row,
        Mapping[str, Any],
        tuple[Mapping[str, Any], ...],
    ]:
        cached = self._attempt_cache.pop(key, None)
        if cached is not None:
            self._attempt_cache[key] = cached
            return cached
        row = self.connection.execute(
            """
            SELECT * FROM attempts
            WHERE repo=? AND run_id=? AND attempt=?
            """,
            key,
        ).fetchone()
        if row is None:
            raise ExportError(f"fetch-state attempt is missing: {key}")
        if str(row["status"]) not in {"done", "empty"}:
            raise ExportError(
                "fetch-state run metadata is not exact canonical evidence"
            )
        result = (
            row,
            _decode_fetch_run_metadata(row, key=key),
            _decode_fetch_jobs(row, key=key),
        )
        self._attempt_cache[key] = result
        if len(self._attempt_cache) > 16:
            self._attempt_cache.popitem(last=False)
        return result

    def _load_member(
        self,
        key: tuple[str, int, int, str],
    ) -> FetchMemberEvidence:
        cached = self._member_cache.pop(key, None)
        if cached is not None:
            self._member_cache[key] = cached
            return cached
        row = self.connection.execute(
            """
            SELECT * FROM members
            WHERE repo=? AND run_id=? AND attempt=? AND archive_member=?
            """,
            key,
        ).fetchone()
        if row is None:
            raise ExportError(f"fetch-state member is missing: {key}")
        _attempt_row, _run_metadata, jobs = self._load_attempt(key[:3])
        selected_job = _job_for_member(key[3], jobs)
        selected_job_id = None if selected_job is None else selected_job.get("id")
        expected_job_key = (
            f"{selected_job_id if isinstance(selected_job_id, int) else 'unresolved'}:"
            f"{key[3]}"
        )
        if str(row["job_key"]) != expected_job_key:
            raise ExportError(
                "fetch-state member job key differs from exact jobs evidence"
            )
        sidecar, _ = _decode_canonical_zlib_mapping(
            row["sidecar_zlib"],
            expected_size=int(row["sidecar_raw_size"]),
            expected_sha256=str(row["sidecar_sha256"]),
            where=f"fetch-state member {key} sidecar",
        )
        if sidecar.get("schema") != PARSER_SIDECAR_SCHEMA:
            raise ExportError("fetch-state sidecar is not canonical parser evidence")
        declared_internal_sha256 = _require_hex64(
            sidecar.get("sidecar_sha256"),
            where="fetch-state sidecar.sidecar_sha256",
        )
        sidecar_without_internal_hash = dict(sidecar)
        del sidecar_without_internal_hash["sidecar_sha256"]
        if (
            _sha256_bytes(_canonical_json_bytes(sidecar_without_internal_hash))
            != declared_internal_sha256
        ):
            raise ExportError("fetch-state sidecar internal SHA-256 is inconsistent")

        raw_evidence = _require_mapping(
            sidecar.get("raw"), where="fetch-state sidecar.raw"
        )
        if (
            raw_evidence.get("input_type") != "bytes"
            or raw_evidence.get("encoding") != "utf-8"
        ):
            raise ExportError(
                "fetch-state sidecar raw decoding contract is unsupported"
            )
        decode_status = _require_nonempty_string(
            raw_evidence.get("status"),
            where="fetch-state sidecar.raw.status",
        )
        if decode_status not in {"valid", "invalid_replaced"}:
            raise ExportError(
                "fetch-state archive member has unsupported decode status"
            )
        invalid_sequence_count = _require_int(
            raw_evidence.get("invalid_sequence_count"),
            where="fetch-state sidecar.raw.invalid_sequence_count",
            minimum=0,
        )
        replacement_char_count = _require_int(
            raw_evidence.get("replacement_char_count"),
            where="fetch-state sidecar.raw.replacement_char_count",
            minimum=0,
        )
        raw_size = _require_int(
            raw_evidence.get("raw_byte_count"),
            where="fetch-state sidecar.raw.raw_byte_count",
            minimum=0,
        )
        raw_sha256 = _require_hex64(
            raw_evidence.get("raw_sha256"),
            where="fetch-state sidecar.raw.raw_sha256",
        )
        if (
            raw_size != int(row["raw_size"])
            or raw_sha256 != str(row["raw_sha256"])
            or invalid_sequence_count > raw_size
            or (
                decode_status == "valid"
                and (invalid_sequence_count or replacement_char_count)
            )
            or (
                decode_status == "invalid_replaced"
                and (
                    invalid_sequence_count < 1
                    or replacement_char_count != invalid_sequence_count
                )
            )
        ):
            raise ExportError(
                "fetch-state sidecar raw decoding evidence is inconsistent"
            )
        invalid_ratio_ppm = (
            0 if raw_size == 0 else (invalid_sequence_count * 1_000_000) // raw_size
        )
        opaque = (
            key[3].casefold().endswith(".zip")
            and decode_status == "invalid_replaced"
            and invalid_sequence_count * 1_000_000
            >= OPAQUE_INVALID_RATIO_PPM_THRESHOLD * raw_size
        )
        exclusion_reason = (
            "nested-zip-invalid-utf8-ratio-ge-100000ppm" if opaque else None
        )

        canonicalization = _require_mapping(
            sidecar.get("canonicalization"),
            where="fetch-state sidecar.canonicalization",
        )
        deduplication = _require_mapping(
            sidecar.get("deduplication"),
            where="fetch-state sidecar.deduplication",
        )
        conservation = _require_mapping(
            sidecar.get("conservation"),
            where="fetch-state sidecar.conservation",
        )
        canonical_sha256 = _require_hex64(
            canonicalization.get("canonical_sha256"),
            where="fetch-state sidecar canonical SHA-256",
        )
        dedup_sha256 = _require_hex64(
            deduplication.get("sha256"),
            where="fetch-state sidecar dedup SHA-256",
        )
        chunk_index = _require_list(
            sidecar.get("chunk_index"),
            where="fetch-state sidecar.chunk_index",
        )
        if (
            canonical_sha256 != str(row["canonical_sha256"])
            or dedup_sha256 != str(row["dedup_sha256"])
            or conservation.get("canonical_sha256") != canonical_sha256
            or conservation.get("dedup_sha256") != dedup_sha256
            or _require_int(
                conservation.get("chunk_count"),
                where="fetch-state sidecar.conservation.chunk_count",
                minimum=0,
            )
            != int(row["chunk_count"])
            or len(chunk_index) != int(row["chunk_count"])
        ):
            raise ExportError(
                "fetch-state sidecar member/chunk hashes are inconsistent"
            )

        evidence = FetchMemberEvidence(
            key=key,
            job_key=str(row["job_key"]),
            job=selected_job,
            raw_sha256=str(row["raw_sha256"]),
            raw_size=int(row["raw_size"]),
            canonical_sha256=str(row["canonical_sha256"]),
            dedup_sha256=str(row["dedup_sha256"]),
            sidecar_sha256=str(row["sidecar_sha256"]),
            chunk_count=int(row["chunk_count"]),
            occurrence_tokens=int(row["occurrence_tokens"]),
            sidecar=sidecar,
            opaque=opaque,
            exclusion_reason=exclusion_reason,
            decode_status=decode_status,
            invalid_sequence_count=invalid_sequence_count,
            replacement_char_count=replacement_char_count,
            invalid_ratio_ppm=invalid_ratio_ppm,
        )
        self._member_cache[key] = evidence
        if len(self._member_cache) > 4:
            self._member_cache.popitem(last=False)
        return evidence

    def validate_occurrence(
        self,
        occurrence: OccurrenceRecord,
    ) -> FetchMemberEvidence:
        provenance = occurrence.provenance
        run_id = _require_int(
            provenance.get("run_id"), where="provenance.run_id", minimum=1
        )
        run_attempt = _require_int(
            provenance.get("run_attempt"),
            where="provenance.run_attempt",
            minimum=1,
        )
        repo = _require_nonempty_string(
            provenance.get("repository_scope_key"),
            where="provenance.repository_scope_key",
        )
        archive = _require_mapping(
            provenance.get("archive"), where="provenance.archive"
        )
        archive_member = _require_nonempty_string(
            archive.get("member"), where="provenance.archive.member"
        )
        key = (repo, run_id, run_attempt, archive_member)
        attempt_row, run_metadata, _jobs = self._load_attempt(key[:3])
        if str(attempt_row["status"]) != "done":
            raise ExportError("CAS occurrence belongs to a non-done fetch attempt")
        evidence = _require_mapping(
            provenance.get("run_metadata_evidence"),
            where="provenance.run_metadata_evidence",
        )
        if (
            str(attempt_row["run_metadata_sha256"]) != evidence.get("sha256")
            or str(attempt_row["run_metadata_source"]) != evidence.get("source")
            or int(attempt_row["run_metadata_source_attempt"])
            != evidence.get("source_attempt")
            or int(attempt_row["inventory_seed_attempt"])
            != evidence.get("inventory_seed_attempt")
            or str(attempt_row["inventory_seed_metadata_sha256"])
            != evidence.get("inventory_seed_metadata_sha256")
        ):
            raise ExportError(
                "occurrence run metadata evidence differs from fetch state"
            )
        expected_workflow = {
            "id": run_metadata.get("workflow_id"),
            "name": run_metadata.get("name"),
            "path": run_metadata.get("path"),
            "event": run_metadata.get("event"),
            "run_number": run_metadata.get("run_number"),
            "status": run_metadata.get("status"),
            "conclusion": run_metadata.get("conclusion"),
            "created_at": run_metadata.get("created_at"),
            "updated_at": run_metadata.get("updated_at"),
            "started_at": run_metadata.get("run_started_at"),
            "display_title": run_metadata.get("display_title"),
            "head_branch": run_metadata.get("head_branch"),
            "head_sha": run_metadata.get("head_sha"),
            "head_commit": run_metadata.get("head_commit"),
            "actor": run_metadata.get("actor"),
            "triggering_actor": run_metadata.get("triggering_actor"),
        }
        if provenance.get("workflow") != expected_workflow:
            raise ExportError("occurrence workflow differs from exact fetch metadata")

        member = self._load_member(key)
        if (
            provenance.get("job") != member.job
            or member.job_key != occurrence.key[2]
            or member.raw_sha256 != archive.get("member_raw_sha256")
            or provenance.get("parser_sidecar_sha256")
            != member.sidecar.get("sidecar_sha256")
        ):
            raise ExportError(
                "occurrence job/member/sidecar binding differs from fetch state"
            )
        chunk = _require_mapping(provenance.get("chunk"), where="provenance.chunk")
        ordinal = _require_int(
            chunk.get("ordinal"), where="provenance.chunk.ordinal", minimum=0
        )
        chunk_index = _require_list(
            member.sidecar.get("chunk_index"),
            where="fetch-state sidecar.chunk_index",
        )
        if ordinal >= len(chunk_index):
            raise ExportError("occurrence chunk ordinal is absent from fetch sidecar")
        sidecar_chunk = _require_mapping(
            chunk_index[ordinal],
            where=f"fetch-state sidecar.chunk_index[{ordinal}]",
        )
        occurrence_chunk_projection = {
            key: value
            for key, value in chunk.items()
            if key not in {"role_spans", "domain_spans", "training_sidecars"}
        }
        if occurrence_chunk_projection != sidecar_chunk:
            raise ExportError("occurrence chunk differs from fetch sidecar chunk index")
        return member

    def verify_member_coverage(
        self,
        eligibility_connection: sqlite3.Connection,
    ) -> None:
        expected_nonempty_members = 0
        for row in self.connection.execute(
            """
            SELECT repo,run_id,attempt,archive_member,chunk_count,occurrence_tokens
            FROM members
            ORDER BY repo,run_id,attempt,archive_member
            """
        ):
            key = (
                str(row["repo"]),
                int(row["run_id"]),
                int(row["attempt"]),
                str(row["archive_member"]),
            )
            chunk_count = int(row["chunk_count"])
            observed = eligibility_connection.execute(
                """
                SELECT COUNT(*) AS chunks,
                       COALESCE(SUM(token_count),0) AS occurrence_tokens
                FROM seen_chunks
                WHERE repo=? AND run_id=? AND attempt=? AND archive_member=?
                """,
                key,
            ).fetchone()
            if observed is None:
                raise ExportError("fetch-state coverage aggregate is missing")
            if int(observed["chunks"]) != chunk_count or int(
                observed["occurrence_tokens"]
            ) != int(row["occurrence_tokens"]):
                raise ExportError(f"CAS/fetch-state member coverage differs for {key}")
            if chunk_count:
                expected_nonempty_members += 1
            else:
                self._load_member(key)
        observed_nonempty_members = int(
            eligibility_connection.execute(
                """
                SELECT COUNT(*) FROM (
                  SELECT repo,run_id,attempt,archive_member
                  FROM seen_chunks
                  GROUP BY repo,run_id,attempt,archive_member
                )
                """
            ).fetchone()[0]
        )
        if observed_nonempty_members != expected_nonempty_members:
            raise ExportError("CAS/fetch-state member set coverage is inconsistent")

    def require_unchanged(self) -> None:
        if self._snapshot is None:
            raise ExportError("fetch-state snapshot was not initialized")
        _require_frozen_sqlite(self.path, label="fetch state")
        stat = self.path.stat()
        current = SnapshotFile(
            relative_path=self.path.name,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            inode=stat.st_ino,
            sha256=_sha256_file(self.path),
        )
        if current != self._snapshot:
            raise ExportError("fetch-state snapshot changed while export was running")

    def parser_lineage(self) -> tuple[str, ...]:
        """Return every unique generation in convergent transition evidence."""

        rows = self.connection.execute(
            """
            SELECT from_sha256,to_sha256
            FROM binding_upgrades
            WHERE binding_key='parser_script_sha256'
            ORDER BY id
            """
        ).fetchall()
        current = _require_hex64(
            self.settings["parser_script_sha256"],
            where="current parser binding",
        )
        if not rows:
            return (current,)
        materialized: set[tuple[str, str]] = set()
        first_seen: dict[str, int] = {}
        for row in rows:
            source = _require_hex64(
                row["from_sha256"],
                where="parser binding upgrade from_sha256",
            )
            target = _require_hex64(
                row["to_sha256"],
                where="parser binding upgrade to_sha256",
            )
            edge = (source, target)
            if source == target or edge in materialized:
                raise ExportError(
                    "fetch-state parser binding history contains an invalid edge"
                )
            materialized.add(edge)
            first_seen.setdefault(source, len(first_seen))
            first_seen.setdefault(target, len(first_seen))
        try:
            component_by_node, component_distance = (
                convergent_transition_layout(
                    materialized,
                    current=current,
                )
            )
        except ValueError as exc:
            raise ExportError(
                "fetch-state parser binding history diverges or cannot return "
                "to the current parser"
            ) from exc
        lineage = sorted(
            (node for node in component_by_node if node != current),
            key=lambda node: (
                -component_distance[component_by_node[node]],
                first_seen.get(node, len(first_seen)),
                node,
            ),
        )
        lineage.append(current)
        return tuple(lineage)

    def receipt_binding(self) -> dict[str, object]:
        if self._snapshot is None:
            raise ExportError("fetch-state snapshot was not initialized")
        return {
            "schema": FETCH_STATE_SCHEMA,
            "artifact": {
                "path": str(self.path),
                "byte_size": self._snapshot.size,
                "mtime_ns": self._snapshot.mtime_ns,
                "inode": self._snapshot.inode,
                "sha256": self._snapshot.sha256,
            },
            "sqlite_schema_sha256": self.sqlite_schema_sha256,
            "sqlite_logical_sha256": self.sqlite_logical_sha256,
            "settings": dict(sorted(self.settings.items())),
            "summary": self.summary,
            "sidecar_set_sha256": self.sidecar_set_sha256,
        }


def _confidence_id(score: object, *, where: str) -> int:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ExportError(f"{where} must be a finite confidence score")
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ExportError(f"{where} must be within [0, 1]")
    if value == 0.0:
        return int(ParseConfidence.ABSENT)
    if value >= 1.0:
        return int(ParseConfidence.EXACT)
    if value >= 0.8:
        return int(ParseConfidence.PARTIAL)
    if value > 0.0:
        return int(ParseConfidence.HEURISTIC)
    raise AssertionError("positive confidence mapping is not exhaustive")


def _validate_rle(
    raw: object,
    *,
    char_count: int,
    id_field: str,
    valid_ids: frozenset[int],
    where: str,
    materialize: bool = True,
) -> tuple[list[int], list[int]]:
    spans = _require_list(raw, where=where)
    if char_count == 0:
        if spans:
            raise ExportError(f"{where} must be empty for empty content")
        return [], []
    if not spans:
        raise ExportError(f"{where} does not cover non-empty content")
    char_ids: list[int] = []
    char_confidence: list[int] = []
    cursor = 0
    for index, raw_span in enumerate(spans):
        span = _require_mapping(raw_span, where=f"{where}[{index}]")
        start = _require_int(
            span.get("start_char"), where=f"{where}[{index}].start_char"
        )
        end = _require_int(span.get("end_char"), where=f"{where}[{index}].end_char")
        value_id = _require_int(
            span.get(id_field), where=f"{where}[{index}].{id_field}"
        )
        if start != cursor or end <= start or end > char_count:
            raise ExportError(f"{where} is not one exact contiguous RLE partition")
        if value_id not in valid_ids:
            raise ExportError(f"{where}[{index}] has unknown {id_field}={value_id}")
        confidence = _confidence_id(
            span.get("confidence"), where=f"{where}[{index}].confidence"
        )
        if materialize:
            char_ids.extend([value_id] * (end - start))
            char_confidence.extend([confidence] * (end - start))
        cursor = end
    if cursor != char_count:
        raise ExportError(f"{where} does not cover every character exactly once")
    return char_ids, char_confidence


def _validate_span_record(
    raw: object,
    *,
    char_count: int,
    where: str,
) -> Mapping[str, Any]:
    record = _require_mapping(raw, where=where)
    start = _require_int(record.get("start_char"), where=f"{where}.start_char")
    end = _require_int(record.get("end_char"), where=f"{where}.end_char")
    if not 0 <= start < end <= char_count:
        raise ExportError(f"{where} span is outside chunk-local characters")
    return record


def _validate_occurrence_v3(
    occurrence: OccurrenceRecord,
    *,
    content_text: str,
) -> Mapping[str, Any]:
    provenance = occurrence.provenance
    if provenance.get("schema") != OCCURRENCE_SCHEMA:
        raise ExportError(f"occurrence {occurrence.key} must use {OCCURRENCE_SCHEMA}")
    run_id = _require_int(
        provenance.get("run_id"), where="provenance.run_id", minimum=1
    )
    run_attempt = _require_int(
        provenance.get("run_attempt"),
        where="provenance.run_attempt",
        minimum=1,
    )
    if occurrence.key[1] != f"{run_id}:{run_attempt}":
        raise ExportError("occurrence key disagrees with v3 run/attempt provenance")
    repository_scope_key = _require_nonempty_string(
        provenance.get("repository_scope_key"),
        where="provenance.repository_scope_key",
    )
    if occurrence.key[0] != repository_scope_key:
        raise ExportError("occurrence repo key disagrees with repository scope")
    for field in (
        "repository",
        "repository_requested",
        "source_repository",
    ):
        _require_nonempty_string(
            provenance.get(field),
            where=f"provenance.{field}",
        )
    if provenance.get("repository_requested") != repository_scope_key:
        raise ExportError("requested repository disagrees with repository scope")
    for field in ("repository_id", "source_repository_id"):
        _require_int(
            provenance.get(field),
            where=f"provenance.{field}",
            minimum=1,
        )
    if provenance.get("repository") == provenance.get(
        "source_repository"
    ) and provenance.get("repository_id") != provenance.get("source_repository_id"):
        raise ExportError("same repository name carries contradictory IDs")
    evidence = _require_mapping(
        provenance.get("run_metadata_evidence"),
        where="provenance.run_metadata_evidence",
    )
    if evidence.get("exact_attempt_match") is not True:
        raise ExportError("v3 run metadata is not an exact attempt match")
    evidence_source = evidence.get("source")
    if evidence_source not in _RUN_METADATA_SOURCES:
        raise ExportError("v3 run metadata source is not recognized")
    source_attempt = _require_int(
        evidence.get("source_attempt"),
        where="run_metadata_evidence.source_attempt",
        minimum=1,
    )
    seed_attempt = _require_int(
        evidence.get("inventory_seed_attempt"),
        where="run_metadata_evidence.inventory_seed_attempt",
        minimum=1,
    )
    if source_attempt != run_attempt or seed_attempt < run_attempt:
        raise ExportError("v3 run metadata attempt evidence is inconsistent")
    evidence_sha256 = _require_hex64(
        evidence.get("sha256"), where="run_metadata_evidence.sha256"
    )
    inventory_seed_sha256 = _require_hex64(
        evidence.get("inventory_seed_metadata_sha256"),
        where="run_metadata_evidence.inventory_seed_metadata_sha256",
    )
    if evidence_source == "inventory-run-list" and (
        seed_attempt != run_attempt or evidence_sha256 != inventory_seed_sha256
    ):
        raise ExportError(
            "inventory run metadata evidence disagrees with its seed binding"
        )
    if (
        evidence_source == "github-workflow-run-attempt-api"
        and seed_attempt <= run_attempt
    ):
        raise ExportError(
            "attempt API metadata evidence requires a newer inventory seed"
        )
    _require_hex64(
        provenance.get("parser_sidecar_sha256"),
        where="provenance.parser_sidecar_sha256",
    )
    workflow = _require_mapping(
        provenance.get("workflow"),
        where="provenance.workflow",
    )
    _require_int(
        workflow.get("id"),
        where="provenance.workflow.id",
        minimum=1,
    )
    _require_int(
        workflow.get("run_number"),
        where="provenance.workflow.run_number",
        minimum=1,
    )
    for field in (
        "name",
        "path",
        "event",
        "status",
        "conclusion",
        "created_at",
        "updated_at",
        "started_at",
        "display_title",
        "head_branch",
    ):
        _require_nonempty_string(
            workflow.get(field),
            where=f"provenance.workflow.{field}",
        )
    head_sha = workflow.get("head_sha")
    if not isinstance(head_sha, str) or _GIT_OID_RE.fullmatch(head_sha) is None:
        raise ExportError("provenance.workflow.head_sha is not a Git object ID")
    head_commit = workflow.get("head_commit")
    head_commit_record = _require_mapping(
        head_commit,
        where="provenance.workflow.head_commit",
    )
    if head_commit_record.get("id") != head_sha:
        raise ExportError("provenance workflow head_commit.id disagrees with head_sha")
    actor = _require_mapping(
        workflow.get("actor"),
        where="provenance.workflow.actor",
    )
    _require_nonempty_string(
        actor.get("login"),
        where="provenance.workflow.actor.login",
    )
    archive = _require_mapping(provenance.get("archive"), where="provenance.archive")
    archive_member = _require_nonempty_string(
        archive.get("member"), where="provenance.archive.member"
    )
    _require_hex64(
        archive.get("member_raw_sha256"),
        where="provenance.archive.member_raw_sha256",
    )
    raw_job = provenance.get("job")
    job_id = (
        raw_job.get("id")
        if isinstance(raw_job, Mapping)
        and isinstance(raw_job.get("id"), int)
        and not isinstance(raw_job.get("id"), bool)
        else "unresolved"
    )
    if isinstance(job_id, int) and job_id < 1:
        raise ExportError("provenance.job.id must be positive")
    if occurrence.key[2] != f"{job_id}:{archive_member}":
        raise ExportError("occurrence job key disagrees with job/archive evidence")

    chunk = _require_mapping(provenance.get("chunk"), where="provenance.chunk")
    char_count = len(content_text)
    content_sha256 = _sha256_bytes(content_text.encode("utf-8"))
    if (
        occurrence.content_sha256 != content_sha256
        or _require_hex64(chunk.get("sha256"), where="provenance.chunk.sha256")
        != content_sha256
    ):
        raise ExportError("occurrence/chunk content SHA-256 binding is inconsistent")
    ordinal = _require_int(
        chunk.get("ordinal"), where="provenance.chunk.ordinal", minimum=0
    )
    if ordinal != occurrence.key[4]:
        raise ExportError("occurrence key disagrees with chunk ordinal")
    if (
        _require_nonempty_string(
            chunk.get("chunk_id"), where="provenance.chunk.chunk_id"
        )
        != f"chunk:{ordinal:06d}"
    ):
        raise ExportError("chunk ID disagrees with its ordinal")
    _require_hex64(
        chunk.get("canonical_sha256"),
        where="provenance.chunk.canonical_sha256",
    )
    section_ordinal = _require_int(
        chunk.get("section_ordinal"),
        where="provenance.chunk.section_ordinal",
        minimum=0,
    )
    section_id = _require_nonempty_string(
        chunk.get("section_id"), where="provenance.chunk.section_id"
    )
    raw_step_ordinal = chunk.get("step_ordinal")
    if raw_step_ordinal is not None:
        _require_int(
            raw_step_ordinal,
            where="provenance.chunk.step_ordinal",
            minimum=0,
        )
    expected_step = (
        f"{section_id}:{raw_step_ordinal if raw_step_ordinal is not None else 'none'}"
    )
    if occurrence.key[3] != expected_step:
        raise ExportError("occurrence step key disagrees with chunk section/step")
    chunk_start = _require_int(
        chunk.get("char_start"),
        where="provenance.chunk.char_start",
        minimum=0,
    )
    chunk_end = _require_int(
        chunk.get("char_end"),
        where="provenance.chunk.char_end",
        minimum=0,
    )
    if chunk_end - chunk_start != char_count:
        raise ExportError("occurrence chunk char span differs from content length")
    if (
        _require_int(
            chunk.get("dedup_char_start"),
            where="provenance.chunk.dedup_char_start",
        )
        != chunk_start
        or _require_int(
            chunk.get("dedup_char_end"),
            where="provenance.chunk.dedup_char_end",
        )
        != chunk_end
    ):
        raise ExportError("canonical/dedup chunk coordinates differ")
    if chunk.get("semantic_span_offset_basis") != "chunk_local_canonical_chars":
        raise ExportError("occurrence semantic span coordinate space is stale")

    section = _require_mapping(provenance.get("section"), where="provenance.section")
    if (
        section.get("section_id") != section_id
        or _require_int(
            section.get("ordinal"),
            where="provenance.section.ordinal",
            minimum=0,
        )
        != section_ordinal
        or section.get("step_ordinal") != raw_step_ordinal
    ):
        raise ExportError("occurrence section disagrees with its chunk")
    section_start = _require_int(
        section.get("char_start"),
        where="provenance.section.char_start",
        minimum=0,
    )
    section_end = _require_int(
        section.get("char_end"),
        where="provenance.section.char_end",
        minimum=0,
    )
    if not section_start <= chunk_start < chunk_end <= section_end:
        raise ExportError("chunk character span is outside its section")
    _require_hex64(
        section.get("canonical_sha256"),
        where="provenance.section.canonical_sha256",
    )
    _require_hex64(
        section.get("dedup_sha256"),
        where="provenance.section.dedup_sha256",
    )
    _validate_rle(
        chunk.get("role_spans"),
        char_count=char_count,
        id_field="role_id",
        valid_ids=VALID_DOMAIN_ROLE_IDS,
        where="provenance.chunk.role_spans",
        materialize=False,
    )
    _validate_rle(
        chunk.get("domain_spans"),
        char_count=char_count,
        id_field="domain_id",
        valid_ids=VALID_DOMAIN_IDS,
        where="provenance.chunk.domain_spans",
        materialize=False,
    )

    training = _require_mapping(
        chunk.get("training_sidecars"),
        where="provenance.chunk.training_sidecars",
    )
    if training.get("schema") != TRAINING_SIDECAR_SCHEMA:
        raise ExportError(f"chunk training sidecars must use {TRAINING_SIDECAR_SCHEMA}")
    if (
        training.get("coordinate_space") != "chunk_local_dedup_chars_v1"
        or training.get("dedup_offsets_equal_canonical_offsets") is not True
        or _require_int(
            training.get("chunk_char_count"),
            where="training_sidecars.chunk_char_count",
        )
        != char_count
    ):
        raise ExportError("chunk training-sidecar coordinate contract is invalid")

    entity_ids: set[str] = set()
    entity_starts: dict[str, int] = {}
    entities = _require_list(
        training.get("entities"), where="training_sidecars.entities"
    )
    for index, raw_entity in enumerate(entities):
        entity = _validate_span_record(
            raw_entity,
            char_count=char_count,
            where=f"training_sidecars.entities[{index}]",
        )
        entity_id = _require_nonempty_string(
            entity.get("entity_id"),
            where=f"training_sidecars.entities[{index}].entity_id",
        )
        if entity_id in entity_ids:
            raise ExportError(f"duplicate local training entity ID {entity_id!r}")
        entity_ids.add(entity_id)
        entity_starts[entity_id] = int(entity["start_char"])
        if (
            _require_int(
                entity.get("domain_id"),
                where=f"training_sidecars.entities[{index}].domain_id",
            )
            not in VALID_DOMAIN_IDS
        ):
            raise ExportError("training entity has an unknown domain ID")
        if (
            _require_int(
                entity.get("role_id"),
                where=f"training_sidecars.entities[{index}].role_id",
            )
            not in VALID_DOMAIN_ROLE_IDS
        ):
            raise ExportError("training entity has an unknown role ID")

    for group in _SPAN_RECORD_GROUPS[1:]:
        records = _require_list(training.get(group), where=f"training_sidecars.{group}")
        for index, record in enumerate(records):
            _validate_span_record(
                record,
                char_count=char_count,
                where=f"training_sidecars.{group}[{index}]",
            )

    edges = _require_list(training.get("edges"), where="training_sidecars.edges")
    for index, raw_edge in enumerate(edges):
        edge = _require_mapping(raw_edge, where=f"training_sidecars.edges[{index}]")
        _require_nonempty_string(
            edge.get("edge_id"),
            where=f"training_sidecars.edges[{index}].edge_id",
        )
        source = _require_nonempty_string(
            edge.get("source"), where=f"training_sidecars.edges[{index}].source"
        )
        target = _require_nonempty_string(
            edge.get("target"), where=f"training_sidecars.edges[{index}].target"
        )
        if source not in entity_ids or target not in entity_ids:
            raise ExportError("in-chunk graph edge has a non-local entity endpoint")
        for field in ("from_char", "to_char"):
            point = _require_int(
                edge.get(field),
                where=f"training_sidecars.edges[{index}].{field}",
            )
            if not 0 <= point < char_count:
                raise ExportError("in-chunk graph edge point is outside content")
        if (
            int(edge["from_char"]) != entity_starts[source]
            or int(edge["to_char"]) != entity_starts[target]
        ):
            raise ExportError("in-chunk graph edge does not anchor entity starts")
        kind = _require_int(
            edge.get("kind_id"),
            where=f"training_sidecars.edges[{index}].kind_id",
        )
        if kind not in VALID_DOMAIN_EDGE_KINDS:
            raise ExportError(f"training edge has unknown kind ID {kind}")
        if edge.get("kind") != DomainEdgeKind(kind).name:
            raise ExportError("training edge kind label disagrees with its kind ID")
        try:
            family = domain_edge_family(kind)
        except ValueError as exc:
            raise ExportError(f"training edge kind {kind} has no family") from exc
        if edge.get("family") not in (None, family):
            raise ExportError("training edge family disagrees with its kind")

    cross_chunk_edges = _require_list(
        training.get("cross_chunk_edges"),
        where="training_sidecars.cross_chunk_edges",
    )
    outbound_accounting_records: list[dict[str, object]] = []
    for index, raw_edge in enumerate(cross_chunk_edges):
        edge = _require_mapping(
            raw_edge, where=f"training_sidecars.cross_chunk_edges[{index}]"
        )
        edge_id = _require_nonempty_string(
            edge.get("edge_id"),
            where=f"training_sidecars.cross_chunk_edges[{index}].edge_id",
        )
        if edge.get("target_coordinate_space") != "canonical_member_chars_v1":
            raise ExportError("cross-chunk edge target coordinate space is stale")
        source = _require_nonempty_string(
            edge.get("source"),
            where=f"training_sidecars.cross_chunk_edges[{index}].source",
        )
        _require_nonempty_string(
            edge.get("target"),
            where=f"training_sidecars.cross_chunk_edges[{index}].target",
        )
        if source not in entity_ids:
            raise ExportError("cross-chunk edge source is not a local entity")
        from_char = _require_int(
            edge.get("from_char"),
            where=f"training_sidecars.cross_chunk_edges[{index}].from_char",
        )
        if not 0 <= from_char < char_count:
            raise ExportError("cross-chunk edge source point is outside content")
        if from_char != entity_starts[source]:
            raise ExportError("cross-chunk edge does not anchor its local entity")
        to_member_char = _require_int(
            edge.get("to_member_char"),
            where=f"training_sidecars.cross_chunk_edges[{index}].to_member_char",
            minimum=0,
        )
        kind = _require_int(
            edge.get("kind_id"),
            where=f"training_sidecars.cross_chunk_edges[{index}].kind_id",
        )
        if kind not in VALID_DOMAIN_EDGE_KINDS:
            raise ExportError("cross-chunk edge has an unknown kind ID")
        if edge.get("kind") != DomainEdgeKind(kind).name:
            raise ExportError("cross-chunk edge kind label disagrees with its kind ID")
        try:
            family = domain_edge_family(kind)
        except ValueError as exc:
            raise ExportError(f"cross-chunk edge kind {kind} has no family") from exc
        if edge.get("family") not in (None, family):
            raise ExportError("cross-chunk edge family disagrees with its kind")
        outbound_accounting_records.append(
            {
                "edge_id": edge_id,
                "kind_id": kind,
                "from_char": chunk_start + from_char,
                "to_char": to_member_char,
            }
        )

    accounting = _require_mapping(
        training.get("cross_chunk_edge_accounting"),
        where="training_sidecars.cross_chunk_edge_accounting",
    )
    count = _require_int(
        accounting.get("count"),
        where="cross_chunk_edge_accounting.count",
        minimum=0,
    )
    outbound = _require_int(
        accounting.get("outbound_count"),
        where="cross_chunk_edge_accounting.outbound_count",
        minimum=0,
    )
    accounting_sha256 = _require_hex64(
        accounting.get("sha256"),
        where="cross_chunk_edge_accounting.sha256",
    )
    if outbound != len(cross_chunk_edges):
        raise ExportError("cross-chunk edge accounting is inconsistent")
    if count != outbound:
        raise ExportError(
            "training sidecar v2 omits non-outbound cross-chunk edge identities"
        )
    if accounting_sha256 != _sequence_digest(outbound_accounting_records):
        raise ExportError("cross-chunk edge accounting digest is inconsistent")
    return chunk


def _project_content(
    *,
    tokenizer: ExactTokenizer,
    text: str,
    chunk: Mapping[str, Any],
) -> ProjectedContent:
    direct_ids = tokenizer.encode_batch([text])[0]
    try:
        offset_ids, token_spans = tokenizer._tokenizer.encode_with_offsets(text)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"tokenizer offset projection failed: {exc}") from exc
    token_ids = [int(value) for value in direct_ids]
    token_spans = [(int(start), int(end)) for start, end in token_spans]
    if token_ids != [int(value) for value in offset_ids]:
        raise ExportError("offset-aware tokenizer IDs differ from ExactTokenizer")
    if len(token_ids) != len(token_spans):
        raise ExportError("tokenizer ID/offset cardinality differs")

    char_count = len(text)
    cursor = 0
    for index, (start, end) in enumerate(token_spans):
        if start != cursor or end <= start or end > char_count:
            raise ExportError(
                "tokenizer offsets are not one exact ordered character "
                f"partition at token {index}"
            )
        cursor = end
    if cursor != char_count:
        raise ExportError("tokenizer offsets do not cover the source text exactly")

    role_chars, role_confidence = _validate_rle(
        chunk.get("role_spans"),
        char_count=char_count,
        id_field="role_id",
        valid_ids=VALID_DOMAIN_ROLE_IDS,
        where="provenance.chunk.role_spans",
    )
    domain_chars, domain_confidence = _validate_rle(
        chunk.get("domain_spans"),
        char_count=char_count,
        id_field="domain_id",
        valid_ids=VALID_DOMAIN_IDS,
        where="provenance.chunk.domain_spans",
    )
    token_roles = _chars_to_tokens_structure_ids(role_chars, text, token_spans)
    token_domains = _chars_to_tokens_structure_ids(domain_chars, text, token_spans)
    role_conf = _chars_to_tokens_structure_ids(role_confidence, text, token_spans)
    domain_conf = _chars_to_tokens_structure_ids(domain_confidence, text, token_spans)
    token_confidence = [
        max(int(role), int(domain))
        for role, domain in zip(role_conf, domain_conf, strict=True)
    ]
    if any(value not in VALID_DOMAIN_CONFIDENCE_IDS for value in token_confidence):
        raise ExportError("projected token confidence ID is outside CASE5")
    if any(int(value) == int(DomainRoleKind.DELIMITER) for value in token_roles):
        raise ExportError("raw payload sidecar may not assert CASE5 delimiter roles")

    training = _require_mapping(
        chunk.get("training_sidecars"), where="chunk.training_sidecars"
    )
    entities = _require_list(
        training.get("entities"), where="training_sidecars.entities"
    )
    entity_chars = [0] * char_count
    ordered_entities = sorted(
        (_require_mapping(item, where="training entity") for item in entities),
        key=lambda item: (
            int(item["start_char"]),
            int(item["end_char"]),
            str(item["entity_id"]),
        ),
    )
    for entity_index, entity in enumerate(ordered_entities, start=1):
        start = int(entity["start_char"])
        end = int(entity["end_char"])
        for char_index in range(start, end):
            if entity_chars[char_index] == 0:
                entity_chars[char_index] = entity_index
    token_entities = _chars_to_tokens_structure_ids(entity_chars, text, token_spans)

    projected_edges: list[dict[str, Any]] = []
    for raw_edge in _require_list(
        training.get("edges"), where="training_sidecars.edges"
    ):
        edge = _require_mapping(raw_edge, where="training edge")
        from_char = int(edge["from_char"])
        to_char = int(edge["to_char"])
        from_token = _char_position_to_token_index(
            token_spans, from_char, source_length=char_count
        )
        to_token = _char_position_to_token_index(
            token_spans, to_char, source_length=char_count
        )
        if from_token is None or to_token is None:
            raise ExportError(
                f"training edge {edge.get('edge_id')!r} cannot map to token offsets"
            )
        kind = int(edge["kind_id"])
        projected_edges.append(
            {
                "edge_id": str(edge.get("edge_id", "")),
                "from": from_token,
                "to": to_token,
                "kind": kind,
                "family": domain_edge_family(kind),
            }
        )
    return ProjectedContent(
        token_ids=token_ids,
        token_spans=token_spans,
        token_domain_ids=[int(value) for value in token_domains],
        token_role_ids=[int(value) for value in token_roles],
        token_entity_ids=[int(value) for value in token_entities],
        token_confidence_ids=[int(value) for value in token_confidence],
        edges=projected_edges,
        cross_chunk_edges=[
            dict(_require_mapping(item, where="cross-chunk edge"))
            for item in _require_list(
                training.get("cross_chunk_edges"),
                where="training_sidecars.cross_chunk_edges",
            )
        ],
    )


def _split_for_sequence(token_sequence_sha256: str) -> str:
    digest = _require_hex64(token_sequence_sha256, where="token_sequence_sha256")
    prefix_chars = int(SPLIT_CONTRACT["hex_prefix_chars"])
    value = int(digest[:prefix_chars], 16) % int(SPLIT_CONTRACT["modulus"])
    ranges = _require_mapping(SPLIT_CONTRACT["ranges"], where="split ranges")
    for name in ("train", "validation", "test"):
        start, end = ranges[name]
        if int(start) <= value < int(end):
            return name
    raise AssertionError("split contract does not cover its modulus")


def _framed_length(domains: Sequence[int]) -> int:
    runs = 0
    previous = int(DomainKind.UNKNOWN)
    for raw_domain in domains:
        domain = int(raw_domain)
        if domain != int(DomainKind.UNKNOWN) and domain != previous:
            runs += 1
        previous = domain
    return 1 + len(domains) + 2 * runs


def _fragment_ranges(domains: Sequence[int]) -> list[tuple[int, int]]:
    if not domains:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(domains):
        end = start
        framed_length = 1  # BOS
        previous = int(DomainKind.UNKNOWN)
        while end < len(domains):
            domain = int(domains[end])
            added = 1
            if domain != int(DomainKind.UNKNOWN) and domain != previous:
                added += 2
            if framed_length + added > BUCKETS[-1]:
                break
            framed_length += added
            previous = domain
            end += 1
        if end == start:
            raise ExportError("one payload token cannot fit the largest CASE5 bucket")
        ranges.append((start, end))
        start = end
    return ranges


def _smallest_bucket(token_count: int) -> int:
    for bucket in BUCKETS:
        if token_count <= bucket:
            return bucket
    raise ExportError(f"framed fragment has {token_count} tokens (> {BUCKETS[-1]})")


def _frame_fragment(
    *,
    projected: ProjectedContent,
    start: int,
    end: int,
    source_identity_id: int,
) -> tuple[dict[str, Any], dict[int, int]]:
    payload_ids = projected.token_ids[start:end]
    domains = projected.token_domain_ids[start:end]
    roles = projected.token_role_ids[start:end]
    entities = projected.token_entity_ids[start:end]
    confidence = projected.token_confidence_ids[start:end]
    bos = int(tokenizer_bos_id())

    token_ids = [bos]
    token_domains = [int(DomainKind.UNKNOWN)]
    token_roles = [int(DomainRoleKind.NONE)]
    token_entities = [0]
    token_confidence = [int(ParseConfidence.ABSENT)]
    source_doc_ids = [1]
    source_identity_ids = [source_identity_id]
    index_map: dict[int, int] = {}

    local_index = 0
    while local_index < len(payload_ids):
        domain = int(domains[local_index])
        run_end = local_index + 1
        while run_end < len(payload_ids) and int(domains[run_end]) == domain:
            run_end += 1
        if domain != int(DomainKind.UNKNOWN):
            try:
                domain_kind = DomainKind(domain)
                start_id, end_id = delimiter_token_ids(domain_kind)
            except (KeyError, ValueError) as exc:
                raise ExportError(
                    f"payload uses domain {domain} without delimiter contract"
                ) from exc
            token_ids.append(int(start_id))
            token_domains.append(domain)
            token_roles.append(int(DomainRoleKind.DELIMITER))
            token_entities.append(0)
            token_confidence.append(int(ParseConfidence.EXACT))
            source_doc_ids.append(1)
            source_identity_ids.append(source_identity_id)
        for payload_index in range(local_index, run_end):
            index_map[start + payload_index] = len(token_ids)
            token_ids.append(int(payload_ids[payload_index]))
            token_domains.append(domain)
            token_roles.append(int(roles[payload_index]))
            token_entities.append(int(entities[payload_index]))
            token_confidence.append(int(confidence[payload_index]))
            source_doc_ids.append(1)
            source_identity_ids.append(source_identity_id)
        if domain != int(DomainKind.UNKNOWN):
            token_ids.append(int(end_id))
            token_domains.append(domain)
            token_roles.append(int(DomainRoleKind.DELIMITER))
            token_entities.append(0)
            token_confidence.append(int(ParseConfidence.EXACT))
            source_doc_ids.append(1)
            source_identity_ids.append(source_identity_id)
        local_index = run_end

    record: dict[str, Any] = {
        TOKEN_IDS_COLUMN: token_ids,
        TOKEN_DOMAIN_IDS_COLUMN: token_domains,
        TOKEN_ROLE_IDS_COLUMN: token_roles,
        TOKEN_ENTITY_IDS_COLUMN: token_entities,
        TOKEN_SCOPE_IDS_COLUMN: [0] * len(token_ids),
        TOKEN_SOURCE_DOC_IDS_COLUMN: source_doc_ids,
        TOKEN_SOURCE_IDENTITY_IDS_COLUMN: source_identity_ids,
        TOKEN_CONFIDENCE_IDS_COLUMN: token_confidence,
        TOKEN_CHUNK_STARTS_COLUMN: [0],
        TOKEN_CHUNK_ENDS_COLUMN: [len(token_ids)],
        TOKEN_CHUNK_KINDS_COLUMN: [0],
        TOKEN_CHUNK_DEP_LEVELS_COLUMN: [0],
        SOURCE_IDENTITY_REGISTRY_COLUMN: [],
    }
    for column in _EDGE_COLUMN_BY_FAMILY.values():
        record[column] = []
    return record, index_map


def _verify_framed_payload(
    *,
    record: Mapping[str, Any],
    index_map: Mapping[int, int],
    projected: ProjectedContent,
    start: int,
    end: int,
) -> None:
    expected_columns = (
        (TOKEN_IDS_COLUMN, projected.token_ids),
        (TOKEN_DOMAIN_IDS_COLUMN, projected.token_domain_ids),
        (TOKEN_ROLE_IDS_COLUMN, projected.token_role_ids),
        (TOKEN_ENTITY_IDS_COLUMN, projected.token_entity_ids),
        (TOKEN_CONFIDENCE_IDS_COLUMN, projected.token_confidence_ids),
    )
    if set(index_map) != set(range(start, end)):
        raise ExportError("framed fragment payload index map is not exhaustive")
    previous_framed_index = -1
    for payload_index in range(start, end):
        framed_index = index_map[payload_index]
        if framed_index <= previous_framed_index:
            raise ExportError("framed fragment reordered payload tokens")
        previous_framed_index = framed_index
        for column, expected in expected_columns:
            raw_values = record.get(column)
            if (
                not isinstance(raw_values, list)
                or not 0 <= framed_index < len(raw_values)
                or int(raw_values[framed_index]) != int(expected[payload_index])
            ):
                raise ExportError(
                    f"framed fragment changed payload identity in {column}"
                )


def _verify_packed_single_document(
    *,
    packed_row: Mapping[str, Any],
    doc: NormalizedDoc,
) -> None:
    valid = _require_int(
        packed_row.get("valid_token_count"),
        where="packed_row.valid_token_count",
        minimum=1,
    )
    if (
        valid != doc.token_count
        or packed_row.get("num_docs") != 1
        or packed_row.get("input_ids", [])[:valid] != doc.token_ids
    ):
        raise ExportError("CASE5 packer changed normalized token identity/order")
    for column in (
        TOKEN_DOMAIN_IDS_COLUMN,
        TOKEN_ROLE_IDS_COLUMN,
        TOKEN_ENTITY_IDS_COLUMN,
        TOKEN_SCOPE_IDS_COLUMN,
        TOKEN_SOURCE_DOC_IDS_COLUMN,
        TOKEN_SOURCE_IDENTITY_IDS_COLUMN,
        TOKEN_CONFIDENCE_IDS_COLUMN,
    ):
        if packed_row.get(column, [])[:valid] != doc.token_meta[column]:
            raise ExportError(f"CASE5 packer changed normalized {column}")
    expected_edges = (
        (TOKEN_DOMAIN_EDGES_COLUMN, doc.domain_edges),
        (TOKEN_BUILD_EDGES_COLUMN, doc.build_edges),
        (TOKEN_SHELL_EDGES_COLUMN, doc.shell_edges),
        (TOKEN_DIAGNOSTIC_EDGES_COLUMN, doc.diagnostic_edges),
        (TOKEN_CROSS_DOMAIN_EDGES_COLUMN, doc.cross_domain_edges),
    )
    for column, edges in expected_edges:
        if packed_row.get(column) != edges:
            raise ExportError(f"CASE5 packer changed normalized {column}")
    if (
        packed_row.get(TOKEN_CHUNK_STARTS_COLUMN) != doc.chunk_starts
        or packed_row.get(TOKEN_CHUNK_ENDS_COLUMN) != doc.chunk_ends
        or packed_row.get(TOKEN_CHUNK_KINDS_COLUMN) != doc.chunk_kinds
        or packed_row.get(TOKEN_CHUNK_DEP_LEVELS_COLUMN) != doc.chunk_dep_levels
        or packed_row.get(SOURCE_IDENTITY_REGISTRY_COLUMN)
        != [dict(item) for item in doc.source_identity_registry]
    ):
        raise ExportError("CASE5 packer changed normalized structural provenance")


def tokenizer_bos_id() -> int:
    """Return the frozen CASE5 BOS token without duplicating its numeric ID."""

    from cppmega.tokenizer.cpp_tokenizer import EXPECTED_SPECIAL_TOKENS

    return int(EXPECTED_SPECIAL_TOKENS["<BOS>"])


def _stable_doc_id(token_sequence_sha256: str, fragment_index: int) -> int:
    raw = hashlib.sha256(
        f"{token_sequence_sha256}:{fragment_index}".encode("ascii")
    ).digest()
    return int.from_bytes(raw[:4], "big") or 1


def _fsync_tree(root: Path) -> None:
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=False,
        followlinks=False,
    ):
        current = Path(directory)
        for name in file_names:
            path = current / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ExportError(f"export tree contains a symlink: {path}")
            if not stat.S_ISREG(metadata.st_mode):
                raise ExportError(
                    f"export tree contains an unsupported artifact: {path}"
                )
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        for name in directory_names:
            path = current / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ExportError(f"export tree contains a symlink: {path}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise ExportError(
                    f"export tree contains an unsupported artifact: {path}"
                )
        descriptor = os.open(current, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a same-filesystem directory without replacement."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            -2,
            source_bytes,
            -2,
            destination_bytes,
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            0x00000001,
        )
    else:
        raise ExportError(
            f"atomic no-replace directory publication is unsupported on {sys.platform}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ExportError(f"output appeared during export: {destination}")
    raise ExportError(
        f"atomic no-replace directory publication failed: {os.strerror(error_number)}"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _script_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _metadata_scalar(value: object, *, where: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > 16_384:
            raise ExportError(f"{where} exceeds the representative metadata limit")
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ExportError(f"{where} is not a bounded JSON scalar")


def _project_scalar_fields(
    record: Mapping[str, Any],
    fields: Sequence[str],
    *,
    where: str,
) -> dict[str, object]:
    return {
        field: _metadata_scalar(record[field], where=f"{where}.{field}")
        for field in fields
        if field in record
    }


def _project_string_list(value: object, *, where: str) -> list[str]:
    records = _require_list(value, where=where)
    if len(records) > 4_096:
        raise ExportError(f"{where} exceeds the representative metadata limit")
    output: list[str] = []
    for index, item in enumerate(records):
        if not isinstance(item, str):
            raise ExportError(f"{where}[{index}] must be a string")
        output.append(str(_metadata_scalar(item, where=f"{where}[{index}]")))
    return output


def _project_confidence(value: object, *, where: str) -> dict[str, object] | None:
    if value is None:
        return None
    record = _require_mapping(value, where=where)
    return _project_scalar_fields(
        record,
        ("score", "level", "source"),
        where=where,
    )


def _sanitize_actor(value: object, *, where: str) -> dict[str, object] | None:
    if value is None:
        return None
    actor = _require_mapping(value, where=where)
    return _project_scalar_fields(
        actor,
        ("login", "id", "node_id", "type", "site_admin", "name"),
        where=where,
    )


def _sanitize_head_commit(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    commit = _require_mapping(value, where="workflow.head_commit")
    output = _project_scalar_fields(
        commit,
        ("id", "tree_id", "timestamp"),
        where="workflow.head_commit",
    )
    message = commit.get("message")
    if message is not None:
        message_text = _metadata_scalar(message, where="workflow.head_commit.message")
        if not isinstance(message_text, str):
            raise ExportError("workflow.head_commit.message must be a string")
        output["message_char_count"] = len(message_text)
        output["message_sha256"] = _sha256_bytes(message_text.encode("utf-8"))
    for field in ("author", "committer"):
        identity = commit.get(field)
        if identity is None:
            output[field] = None
            continue
        identity_record = _require_mapping(
            identity, where=f"workflow.head_commit.{field}"
        )
        output[field] = _project_scalar_fields(
            identity_record,
            ("name", "username"),
            where=f"workflow.head_commit.{field}",
        )
    return output


def _sanitize_workflow(value: object) -> dict[str, object]:
    workflow = _require_mapping(value, where="representative workflow")
    output = _project_scalar_fields(
        workflow,
        (
            "id",
            "name",
            "path",
            "event",
            "run_number",
            "status",
            "conclusion",
            "created_at",
            "updated_at",
            "started_at",
            "display_title",
            "head_branch",
            "head_sha",
        ),
        where="workflow",
    )
    output["actor"] = _sanitize_actor(workflow.get("actor"), where="workflow.actor")
    output["triggering_actor"] = _sanitize_actor(
        workflow.get("triggering_actor"),
        where="workflow.triggering_actor",
    )
    output["head_commit"] = _sanitize_head_commit(workflow.get("head_commit"))
    return output


def _sanitize_job_step(value: object, *, index: int) -> dict[str, object]:
    step = _require_mapping(value, where=f"job.steps[{index}]")
    return _project_scalar_fields(
        step,
        ("number", "name", "status", "conclusion", "started_at", "completed_at"),
        where=f"job.steps[{index}]",
    )


def _sanitize_job(value: object) -> dict[str, object]:
    if value is None:
        return {}
    job = _require_mapping(value, where="representative job")
    output = _project_scalar_fields(
        job,
        (
            "id",
            "name",
            "status",
            "conclusion",
            "created_at",
            "started_at",
            "completed_at",
            "runner_id",
            "runner_name",
            "runner_group_id",
            "runner_group_name",
        ),
        where="job",
    )
    output["labels"] = _project_string_list(
        job.get("labels", []),
        where="job.labels",
    )
    raw_steps = _require_list(job.get("steps", []), where="job.steps")
    if len(raw_steps) > 4_096:
        raise ExportError("job.steps exceeds the representative metadata limit")
    output["steps"] = [
        _sanitize_job_step(step, index=index) for index, step in enumerate(raw_steps)
    ]
    return output


def _runner_evidence(job: Mapping[str, object]) -> dict[str, object]:
    labels = [str(value) for value in job.get("labels", [])]
    os_labels = [
        label
        for label in labels
        if label.casefold() in {"linux", "windows", "macos", "darwin"}
        or label.casefold().startswith(("ubuntu-", "windows-", "macos-"))
    ]
    architecture_labels = [
        label
        for label in labels
        if label.casefold() in {"x64", "x86", "amd64", "arm", "arm64", "aarch64"}
    ]
    return {
        "source": "github-actions-job-api-fields-and-labels",
        "runner_id": job.get("runner_id"),
        "runner_name": job.get("runner_name"),
        "runner_group_id": job.get("runner_group_id"),
        "runner_group_name": job.get("runner_group_name"),
        "labels": labels,
        "os_label_evidence": os_labels,
        "architecture_label_evidence": architecture_labels,
    }


def _sanitize_repository_binding(
    value: object,
    *,
    action_index: int,
    binding_index: int,
) -> dict[str, object]:
    where = (
        f"training_sidecars.build_actions[{action_index}]"
        f".repository_source_bindings[{binding_index}]"
    )
    binding = _require_mapping(value, where=where)
    output = _project_scalar_fields(
        binding,
        ("repository", "head_sha", "source_path"),
        where=where,
    )
    if "confidence" in binding:
        output["confidence"] = _project_confidence(
            binding["confidence"],
            where=f"{where}.confidence",
        )
    return output


def _sanitize_build_action(value: object, *, index: int) -> dict[str, object]:
    where = f"training_sidecars.build_actions[{index}]"
    action = _require_mapping(value, where=where)
    output = _project_scalar_fields(
        action,
        (
            "normalization_schema",
            "tool",
            "kind",
            "cwd",
            "target",
            "command_char_count",
            "command_sha256",
            "action_shape_sha256",
            "all_flags_sha256",
            "source_input_count",
            "all_source_inputs_sha256",
            "output_count",
            "all_outputs_sha256",
            "repository_source_binding_count",
            "action_entity_id",
            "start_char",
            "end_char",
            "line_index",
            "section_ordinal",
            "step_ordinal",
            "source_span_clipped",
            "occurrence_count",
        ),
        where=where,
    )
    for field in ("flags", "source_inputs", "outputs"):
        output[field] = _project_string_list(
            action.get(field, []),
            where=f"{where}.{field}",
        )
    raw_bindings = _require_list(
        action.get("repository_source_bindings", []),
        where=f"{where}.repository_source_bindings",
    )
    if len(raw_bindings) > 4_096:
        raise ExportError(f"{where}.repository_source_bindings is too large")
    output["repository_source_bindings"] = [
        _sanitize_repository_binding(
            binding,
            action_index=index,
            binding_index=binding_index,
        )
        for binding_index, binding in enumerate(raw_bindings)
    ]
    if "confidence" in action:
        output["confidence"] = _project_confidence(
            action["confidence"],
            where=f"{where}.confidence",
        )
    return output


def _sanitize_training_record(
    value: object,
    *,
    group: str,
    index: int,
    fields: Sequence[str],
) -> dict[str, object]:
    where = f"training_sidecars.{group}[{index}]"
    record = _require_mapping(value, where=where)
    output = _project_scalar_fields(record, fields, where=where)
    if "confidence" in record:
        output["confidence"] = _project_confidence(
            record["confidence"],
            where=f"{where}.confidence",
        )
    return output


def _language_evidence(training: Mapping[str, Any]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for index, raw_entity in enumerate(
        _require_list(training.get("entities"), where="training_sidecars.entities")
    ):
        entity = _require_mapping(
            raw_entity, where=f"training_sidecars.entities[{index}]"
        )
        attributes = entity.get("attributes")
        if attributes is None:
            continue
        attribute_record = _require_mapping(
            attributes,
            where=f"training_sidecars.entities[{index}].attributes",
        )
        for field in ("language", "likely_language"):
            value = attribute_record.get(field)
            if value is None:
                continue
            language = _metadata_scalar(
                value,
                where=f"training_sidecars.entities[{index}].attributes.{field}",
            )
            if not isinstance(language, str):
                raise ExportError("training language evidence must be a string")
            output.append(
                {
                    "entity_id": _metadata_scalar(
                        entity.get("entity_id"),
                        where=f"training_sidecars.entities[{index}].entity_id",
                    ),
                    "kind": _metadata_scalar(
                        entity.get("kind"),
                        where=f"training_sidecars.entities[{index}].kind",
                    ),
                    "start_char": _metadata_scalar(
                        entity.get("start_char"),
                        where=f"training_sidecars.entities[{index}].start_char",
                    ),
                    "end_char": _metadata_scalar(
                        entity.get("end_char"),
                        where=f"training_sidecars.entities[{index}].end_char",
                    ),
                    "language": language,
                    "source_field": (f"training_sidecars.entities.attributes.{field}"),
                }
            )
    return output


def _optional_classification_string(
    record: Mapping[str, object],
    field: str,
    *,
    where: str,
) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ExportError(f"{where}.{field} must be a non-empty string or null")
    return value


def _typed_string_set(
    values: Iterable[str],
    *,
    evidence_source: str,
) -> dict[str, object]:
    ordered = sorted(set(values), key=lambda value: (value.casefold(), value))
    return {
        "value_type": "set[string]",
        "status": "resolved" if ordered else "unresolved",
        "values": ordered if ordered else None,
        "evidence_source": evidence_source,
    }


def _tool_basename(value: str) -> str:
    basename = value.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return basename.removesuffix(".exe")


def _source_extension(value: str) -> str | None:
    basename = value.replace("\\", "/").rsplit("/", 1)[-1]
    match = re.search(r"(\.[A-Za-z0-9_+-]+)$", basename)
    if match is None:
        return None
    extension = match.group(1)
    return ".C" if extension == ".C" else extension.casefold()


def _derived_classifications(
    *,
    runner: Mapping[str, object],
    parser_sidecar: Mapping[str, Any],
    language_evidence: Sequence[Mapping[str, object]],
    build_actions: Sequence[Mapping[str, object]],
    tests: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    parser_classifications = _require_mapping(
        parser_sidecar.get("classifications"),
        where="parser_sidecar.classifications",
    )

    def parser_named_values(
        group: str,
        *,
        optional: bool = False,
    ) -> list[str]:
        values: list[str] = []
        raw_group = parser_classifications.get(group)
        if raw_group is None and optional:
            raw_group = []
        for index, raw_record in enumerate(
            _require_list(
                raw_group,
                where=f"parser_sidecar.classifications.{group}",
            )
        ):
            record = _require_mapping(
                raw_record,
                where=f"parser_sidecar.classifications.{group}[{index}]",
            )
            name = _optional_classification_string(
                record,
                "name",
                where=f"parser_sidecar.classifications.{group}[{index}]",
            )
            if name is not None:
                values.append(name)
        return values

    parser_languages = parser_named_values("languages")
    parser_shell_dialects = parser_named_values("shell_dialects")
    parser_sql_dialects = parser_named_values("sql_dialects", optional=True)
    parser_build_systems = parser_named_values("build_systems")
    parser_toolchains = parser_named_values("toolchains")
    source_paths: list[str] = []
    tools: list[str] = list(parser_toolchains)
    action_kinds: list[str] = []
    for index, action in enumerate(build_actions):
        tool = _optional_classification_string(
            action,
            "tool",
            where=f"derived_classifications.build_actions[{index}]",
        )
        if tool is not None:
            tools.append(tool)
        kind = _optional_classification_string(
            action,
            "kind",
            where=f"derived_classifications.build_actions[{index}]",
        )
        if kind is not None:
            action_kinds.append(kind)
        source_paths.extend(
            str(value)
            for value in _project_string_list(
                action.get("source_inputs", []),
                where=f"derived_classifications.build_actions[{index}].source_inputs",
            )
        )
        for binding_index, raw_binding in enumerate(
            _require_list(
                action.get("repository_source_bindings", []),
                where=(
                    "derived_classifications.build_actions"
                    f"[{index}].repository_source_bindings"
                ),
            )
        ):
            binding = _require_mapping(
                raw_binding,
                where=(
                    "derived_classifications.build_actions"
                    f"[{index}].repository_source_bindings[{binding_index}]"
                ),
            )
            source_path = _optional_classification_string(
                binding,
                "source_path",
                where=(
                    "derived_classifications.build_actions"
                    f"[{index}].repository_source_bindings[{binding_index}]"
                ),
            )
            if source_path is not None:
                source_paths.append(source_path)

    for index, diagnostic in enumerate(diagnostics):
        tool = _optional_classification_string(
            diagnostic,
            "tool",
            where=f"derived_classifications.diagnostics[{index}]",
        )
        if tool is not None:
            tools.append(tool)
        source_path = _optional_classification_string(
            diagnostic,
            "file",
            where=f"derived_classifications.diagnostics[{index}]",
        )
        if source_path is not None:
            source_paths.append(source_path)

    source_extensions = [
        extension
        for path in source_paths
        if (extension := _source_extension(path)) is not None
    ]
    extension_language_evidence = [
        {
            "extension": extension,
            "language": _SOURCE_LANGUAGE_BY_EXTENSION[extension],
            "source": "retained-source-path-extension-v1",
        }
        for extension in sorted(set(source_extensions))
        if extension in _SOURCE_LANGUAGE_BY_EXTENSION
    ]
    explicit_languages: list[str] = []
    for index, evidence in enumerate(language_evidence):
        language = _optional_classification_string(
            evidence,
            "language",
            where=f"derived_classifications.language_evidence[{index}]",
        )
        if language is not None:
            explicit_languages.append(language)
    languages = (
        parser_languages
        + explicit_languages
        + [str(item["language"]) for item in extension_language_evidence]
    )

    os_labels = [
        str(value)
        for value in _project_string_list(
            runner.get("os_label_evidence", []),
            where="derived_classifications.runner.os_label_evidence",
        )
    ]
    architecture_labels = [
        str(value)
        for value in _project_string_list(
            runner.get("architecture_label_evidence", []),
            where="derived_classifications.runner.architecture_label_evidence",
        )
    ]
    systems: list[str] = []
    for label in os_labels:
        folded = label.casefold()
        if folded == "linux" or folded.startswith("ubuntu-"):
            systems.append("linux")
        elif folded == "windows" or folded.startswith("windows-"):
            systems.append("windows")
        elif folded in {"macos", "darwin"} or folded.startswith("macos-"):
            systems.append("macos")

    parser_platform = _require_mapping(
        parser_classifications.get("platform"),
        where="parser_sidecar.classifications.platform",
    )
    projected_parser_platform: dict[str, object] = {}
    for field in ("os", "os_version", "runner_image", "architecture"):
        raw_field = parser_platform.get(field)
        if raw_field is None:
            continue
        field_record = _require_mapping(
            raw_field,
            where=f"parser_sidecar.classifications.platform.{field}",
        )
        value = _metadata_scalar(
            field_record.get("value"),
            where=f"parser_sidecar.classifications.platform.{field}.value",
        )
        if value is not None:
            projected_parser_platform[field] = value
    parser_os = projected_parser_platform.get("os")
    if isinstance(parser_os, str):
        folded = parser_os.casefold()
        if folded in {"linux", "windows"}:
            systems.append(folded)
        elif folded in {"macos", "darwin", "macosx", "osx"}:
            systems.append("macos")

    runner_value = {
        field: runner.get(field)
        for field in (
            "runner_id",
            "runner_name",
            "runner_group_id",
            "runner_group_name",
            "labels",
        )
    }
    runner_resolved = any(
        value not in (None, [], "") for value in runner_value.values()
    )
    platform_resolved = bool(
        os_labels or architecture_labels or projected_parser_platform
    )

    build_system_evidence: list[dict[str, str]] = []
    for tool in tools:
        basename = _tool_basename(tool)
        build_system = _BUILD_SYSTEM_BY_TOOL.get(basename)
        if build_system is not None:
            build_system_evidence.append(
                {
                    "tool": tool,
                    "normalized_tool": basename,
                    "build_system": build_system,
                }
            )

    test_frameworks: list[str] = []
    test_results: list[str] = []
    for index, test in enumerate(tests):
        framework = _optional_classification_string(
            test,
            "framework",
            where=f"derived_classifications.tests[{index}]",
        )
        if framework is not None:
            test_frameworks.append(framework)
        result = _optional_classification_string(
            test,
            "result",
            where=f"derived_classifications.tests[{index}]",
        )
        if result is not None:
            test_results.append(result)
    parser_tests = _require_list(
        parser_classifications.get("tests"),
        where="parser_sidecar.classifications.tests",
    )
    for index, raw_test in enumerate(parser_tests):
        parser_test = _require_mapping(
            raw_test,
            where=f"parser_sidecar.classifications.tests[{index}]",
        )
        framework = _optional_classification_string(
            parser_test,
            "framework",
            where=f"parser_sidecar.classifications.tests[{index}]",
        )
        if framework is not None:
            test_frameworks.append(framework)
        result = _optional_classification_string(
            parser_test,
            "result",
            where=f"parser_sidecar.classifications.tests[{index}]",
        )
        if result is not None:
            test_results.append(result)
    detected_test_count = max(len(tests), len(parser_tests))

    return {
        "schema": DERIVED_CLASSIFICATION_SCHEMA,
        "scope_contract": {
            "workflow_job_runner": "exact-occurrence-api-metadata",
            "parser_classifications": "archive-member",
            "training_sidecars": "chunk-local",
            "derived_values": (
                "typed union; evidence_source identifies member versus chunk scope"
            ),
        },
        "language": {
            **_typed_string_set(
                languages,
                evidence_source=(
                    "archive-member-parser-plus-chunk-training-entity-and-"
                    "retained-source-extension-v1"
                ),
            ),
            "source_extension_evidence": extension_language_evidence,
        },
        "shell_dialect": _typed_string_set(
            parser_shell_dialects,
            evidence_source="archive-member-parser-shell-dialect-v1",
        ),
        "sql_dialect": _typed_string_set(
            parser_sql_dialects,
            evidence_source="archive-member-exact-sql-client-command-v1",
        ),
        "source_extension": _typed_string_set(
            source_extensions,
            evidence_source="retained-build-input-binding-and-diagnostic-path-v1",
        ),
        "system": _typed_string_set(
            systems,
            evidence_source=(
                "exact-job-label-plus-archive-member-parser-platform-v1"
            ),
        ),
        "platform": {
            "value_type": "github_actions_runner_label_platform",
            "status": "resolved" if platform_resolved else "unresolved",
            "value": (
                {
                    "os_labels": os_labels,
                    "architecture_labels": architecture_labels,
                    "parser_classification": (projected_parser_platform or None),
                }
                if platform_resolved
                else None
            ),
            "evidence_source": "exact-github-actions-job-labels-v1",
            "completeness": (
                "complete"
                if (
                    (os_labels or projected_parser_platform.get("os"))
                    and (
                        architecture_labels
                        or projected_parser_platform.get("architecture")
                    )
                )
                else "partial"
                if platform_resolved
                else "unresolved"
            ),
        },
        "runner": {
            "value_type": "github_actions_runner",
            "status": "resolved" if runner_resolved else "unresolved",
            "value": runner_value if runner_resolved else None,
            "evidence_source": "exact-github-actions-job-api-fields-v1",
        },
        "build_system": {
            **_typed_string_set(
                [
                    *parser_build_systems,
                    *(item["build_system"] for item in build_system_evidence),
                ],
                evidence_source=(
                    "full-parser-build-system-and-retained-action-tool-map-v1"
                ),
            ),
            "tool_evidence": build_system_evidence,
        },
        "test": {
            "value_type": "detected_test_records",
            "status": "resolved" if detected_test_count else "unresolved",
            "value": (
                {
                    "record_count": detected_test_count,
                    "framework": _typed_string_set(
                        test_frameworks,
                        evidence_source="retained-training-test-framework-v1",
                    ),
                    "result": _typed_string_set(
                        test_results,
                        evidence_source="retained-training-test-result-v1",
                    ),
                }
                if detected_test_count
                else None
            ),
            "evidence_source": (
                "archive-member-parser-plus-chunk-training-test-records-v1"
            ),
        },
        "tool": _typed_string_set(
            tools,
            evidence_source="retained-build-action-and-diagnostic-tool-v1",
        ),
        "action_kind": _typed_string_set(
            action_kinds,
            evidence_source="retained-build-action-kind-v1",
        ),
    }


def _project_semantic_rle(
    value: object,
    *,
    group: str,
    id_field: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for index, raw_span in enumerate(_require_list(value, where=f"chunk.{group}")):
        span = _require_mapping(raw_span, where=f"chunk.{group}[{index}]")
        output.append(
            _project_scalar_fields(
                span,
                ("start_char", "end_char", id_field, "confidence"),
                where=f"chunk.{group}[{index}]",
            )
        )
    return output


def _representative_metadata_record(
    *,
    content: ContentRecord,
    occurrence: OccurrenceRecord,
    parser_sidecar: Mapping[str, Any],
    source_binding_projector: SourceBindingProjector
    | SourceBindingProjectionRouter,
) -> dict[str, object]:
    provenance = occurrence.provenance
    workflow = _sanitize_workflow(provenance.get("workflow"))
    job = _sanitize_job(provenance.get("job"))
    chunk = _require_mapping(provenance.get("chunk"), where="representative chunk")
    section = _require_mapping(
        provenance.get("section"), where="representative section"
    )
    training = _require_mapping(
        chunk.get("training_sidecars"),
        where="representative training sidecars",
    )
    build_actions: list[dict[str, object]] = []
    representative_projection_records: list[Mapping[str, object]] = []
    for index, raw_action in enumerate(
        _require_list(
            training.get("build_actions"),
            where="training_sidecars.build_actions",
        )
    ):
        action = _require_mapping(
            raw_action,
            where=f"training_sidecars.build_actions[{index}]",
        )
        projection = source_binding_projector.project_action(
            occurrence_key=occurrence.key_dict,
            provenance_sha256=occurrence.provenance_sha256,
            provenance=provenance,
            action=action,
            action_index=index,
        )
        projected_action = dict(action)
        projected_action["repository_source_bindings"] = list(
            projection.projected_bindings
        )
        projected_action["repository_source_binding_count"] = len(
            projection.projected_bindings
        )
        sanitized_action = _sanitize_build_action(
            projected_action,
            index=index,
        )
        sanitized_action["source_binding_projection"] = {
            "schema": SOURCE_BINDING_PROJECTION_SCHEMA,
            "mode": projection.selected_mode,
            "input_parser_script_sha256": (
                projection.selected_input_parser_sha256
            ),
            "target_parser_script_sha256": (
                source_binding_projector.target_parser_sha256
            ),
            "record_count": len(projection.records),
            "records_sha256": _hash_records(
                "cppmega-ci-source-binding-projection-action-v1",
                projection.records,
            ),
            "upstream_repository_source_binding_count": _require_int(
                action.get("repository_source_binding_count"),
                where=(
                    f"training_sidecars.build_actions[{index}]"
                    ".repository_source_binding_count"
                ),
                minimum=0,
            ),
            "projected_repository_source_binding_count": len(
                projection.projected_bindings
            ),
        }
        build_actions.append(sanitized_action)
        representative_projection_records.extend(projection.records)
    tests = [
        _sanitize_training_record(
            item,
            group="tests",
            index=index,
            fields=(
                "framework",
                "suite",
                "case",
                "result",
                "count",
                "duration_ms",
                "start_char",
                "end_char",
                "line_index",
                "section_ordinal",
                "step_ordinal",
                "source_span_clipped",
                "occurrence_count",
            ),
        )
        for index, item in enumerate(
            _require_list(training.get("tests"), where="training_sidecars.tests")
        )
    ]
    diagnostics = [
        _sanitize_training_record(
            item,
            group="diagnostics",
            index=index,
            fields=(
                "category",
                "tool",
                "severity",
                "code",
                "file",
                "source_line",
                "source_column",
                "symbol",
                "start_char",
                "end_char",
                "line_index",
                "section_ordinal",
                "step_ordinal",
                "source_span_clipped",
                "occurrence_count",
            ),
        )
        for index, item in enumerate(
            _require_list(
                training.get("diagnostics"),
                where="training_sidecars.diagnostics",
            )
        )
    ]
    evidence = _require_mapping(
        provenance.get("run_metadata_evidence"),
        where="representative run metadata evidence",
    )
    archive = _require_mapping(
        provenance.get("archive"), where="representative archive"
    )
    language_evidence = _language_evidence(training)
    runner_evidence = _runner_evidence(job)
    derived_classifications = _derived_classifications(
        runner=runner_evidence,
        parser_sidecar=parser_sidecar,
        language_evidence=language_evidence,
        build_actions=build_actions,
        tests=tests,
        diagnostics=diagnostics,
    )
    return {
        "schema": REPRESENTATIVE_METADATA_SCHEMA,
        "token_sequence_sha256": content.token_sequence_sha256,
        "content_sha256": content.sha256,
        "provenance_sha256": occurrence.provenance_sha256,
        "occurrence_key": occurrence.key_dict,
        "repository": _project_scalar_fields(
            provenance,
            (
                "repository",
                "repository_requested",
                "repository_id",
                "source_repository",
                "source_repository_id",
                "repository_scope_key",
            ),
            where="provenance",
        ),
        "run": {
            "run_id": _metadata_scalar(
                provenance.get("run_id"), where="provenance.run_id"
            ),
            "run_attempt": _metadata_scalar(
                provenance.get("run_attempt"), where="provenance.run_attempt"
            ),
            "metadata_evidence": _project_scalar_fields(
                evidence,
                (
                    "exact_attempt_match",
                    "source",
                    "source_attempt",
                    "sha256",
                    "inventory_seed_attempt",
                    "inventory_seed_metadata_sha256",
                ),
                where="run_metadata_evidence",
            ),
        },
        "workflow": workflow,
        "job": job,
        "step": {
            "occurrence_key": occurrence.key[3],
            **_project_scalar_fields(
                chunk,
                ("section_id", "section_ordinal", "step_ordinal"),
                where="chunk",
            ),
            **_project_scalar_fields(
                section,
                ("kind", "title"),
                where="section",
            ),
        },
        "runner_evidence": runner_evidence,
        "archive": _project_scalar_fields(
            archive,
            ("member", "member_raw_sha256"),
            where="archive",
        ),
        "parser_sidecar_sha256": _metadata_scalar(
            provenance.get("parser_sidecar_sha256"),
            where="provenance.parser_sidecar_sha256",
        ),
        "source_binding_projection": {
            "schema": SOURCE_BINDING_PROJECTION_SCHEMA,
            "mode": source_binding_projector.mode,
            "input_parser_script_sha256": (
                source_binding_projector.input_parser_sha256
            ),
            "target_parser_script_sha256": (
                source_binding_projector.target_parser_sha256
            ),
            **(
                {
                    "parser_lineage": list(
                        source_binding_projector.parser_lineage
                    ),
                    "selection_policy": (
                        source_binding_projector.SELECTION_POLICY
                    ),
                }
                if isinstance(
                    source_binding_projector,
                    SourceBindingProjectionRouter,
                )
                and source_binding_projector.mode
                == SourceBindingProjectionRouter.MIXED_MODE
                else {}
            ),
            "record_count": len(representative_projection_records),
            "records_sha256": _hash_records(
                "cppmega-ci-source-binding-projection-representative-v1",
                representative_projection_records,
            ),
        },
        "derived_classifications": derived_classifications,
        "training_sidecars": {
            "schema": training.get("schema"),
            "coordinate_space": training.get("coordinate_space"),
            "domain_spans": _project_semantic_rle(
                chunk.get("domain_spans"),
                group="domain_spans",
                id_field="domain_id",
            ),
            "role_spans": _project_semantic_rle(
                chunk.get("role_spans"),
                group="role_spans",
                id_field="role_id",
            ),
            "language_evidence": language_evidence,
            "build_actions": build_actions,
            "tests": tests,
            "diagnostics": diagnostics,
            "counts": {
                "commands": len(
                    _require_list(
                        training.get("commands"),
                        where="training_sidecars.commands",
                    )
                ),
                "build_actions": len(build_actions),
                "tests": len(tests),
                "diagnostics": len(diagnostics),
            },
        },
    }


def _occurrence_metadata_record(
    *,
    content: ContentRecord,
    occurrence: OccurrenceRecord,
    member: FetchMemberEvidence,
    scope_decision: Mapping[str, object],
    source_binding_projector: SourceBindingProjector
    | SourceBindingProjectionRouter,
) -> dict[str, object]:
    record = _representative_metadata_record(
        content=content,
        occurrence=occurrence,
        parser_sidecar=member.sidecar,
        source_binding_projector=source_binding_projector,
    )
    record["schema"] = OCCURRENCE_METADATA_SCHEMA
    record["scope"] = "one-record-per-frozen-cas-occurrence"
    effective_routes = _require_list(
        scope_decision.get("effective_routes"),
        where="training scope effective_routes",
    )
    if member.opaque:
        status = "excluded_opaque"
        reason = member.exclusion_reason
    elif PRIMARY_ROUTE in effective_routes:
        status = "eligible_primary"
        reason = None
    elif AUX_PYTHON_ROUTE in effective_routes and AUX_JS_TS_ROUTE in effective_routes:
        status = "aux_python_js_ts"
        reason = "exact_step_auxiliary_only"
    elif AUX_PYTHON_ROUTE in effective_routes:
        status = "aux_python"
        reason = "exact_step_auxiliary_only"
    elif AUX_JS_TS_ROUTE in effective_routes:
        status = "aux_js_ts"
        reason = "exact_step_auxiliary_only"
    else:
        status = "excluded_irrelevant"
        reason = "no_primary_or_auxiliary_scope_evidence"
    record["case5_eligibility"] = {
        "status": status,
        "reason": reason,
        "primary_eligible": status == "eligible_primary",
        "training_scope": dict(scope_decision),
    }
    return record


def export_store(
    *,
    store_root: str | os.PathLike[str],
    store_receipt: str | os.PathLike[str],
    fetch_state: str | os.PathLike[str],
    tokenizer_json: str | os.PathLike[str],
    output: str | os.PathLike[str],
    source_binding_projection_from_parser_sha256: str | None = None,
    require_current_parser_only: bool = False,
    required_eligible_exact_unique_payload_tokens: int | None = None,
    completion_mode: str = COMPLETION_MODE_THRESHOLD,
    inventory: str | os.PathLike[str] | None = None,
    inventory_receipt: str | os.PathLike[str] | None = None,
    fetch_receipt: str | os.PathLike[str] | None = None,
    merge_receipt: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Export one receipt-frozen store and atomically publish a CASE5 directory."""

    raw_store_root = Path(store_root).expanduser()
    raw_store_receipt = Path(store_receipt).expanduser()
    raw_fetch_state = Path(fetch_state).expanduser()
    raw_tokenizer = Path(tokenizer_json).expanduser()
    raw_output = Path(output).expanduser()
    for path, label, require_directory in (
        (raw_store_root, "content store", True),
        (raw_store_receipt, "store receipt", False),
        (raw_fetch_state, "fetch state", False),
        (raw_tokenizer, "tokenizer", False),
    ):
        if path.is_symlink() or not (
            path.is_dir() if require_directory else path.is_file()
        ):
            raise ExportError(f"{label} is missing or unsafe: {path}")
    if raw_output.is_symlink():
        raise ExportError(f"output cannot be a symlink: {raw_output}")
    resolved_store_root = raw_store_root.resolve()
    resolved_store_receipt = raw_store_receipt.resolve()
    resolved_fetch_state = raw_fetch_state.resolve()
    resolved_tokenizer = raw_tokenizer.resolve()
    exporter_script_sha256 = _script_sha256()
    source_binding_projection_script_sha256 = projection_script_sha256()
    tokenizer_snapshot = {
        "byte_size": resolved_tokenizer.stat().st_size,
        "sha256": _sha256_file(resolved_tokenizer),
    }
    output_path = raw_output.resolve()
    if completion_mode not in {
        COMPLETION_MODE_THRESHOLD,
        COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
    }:
        raise ExportError(
            f"unsupported completion mode: {completion_mode!r}"
        )
    if completion_mode == COMPLETION_MODE_INVENTORY_EXHAUSTIVE:
        # A production receipt must prove current-parser singleton input even
        # when a caller omitted the redundant CLI hardening flag.
        require_current_parser_only = True
    production_provenance: dict[str, object] | None = None
    production_paths: tuple[Path, Path, Path, Path] | None = None
    if completion_mode == COMPLETION_MODE_INVENTORY_EXHAUSTIVE:
        if (
            inventory is None
            or inventory_receipt is None
            or fetch_receipt is None
            or merge_receipt is None
        ):
            raise ExportError(
                "inventory-exhaustive export requires inventory, inventory "
                "receipt, fetch receipt, and merge receipt"
            )
        raw_production_paths = (
            Path(inventory).expanduser(),
            Path(inventory_receipt).expanduser(),
            Path(fetch_receipt).expanduser(),
            Path(merge_receipt).expanduser(),
        )
        for path, label in zip(
            raw_production_paths,
            (
                "inventory",
                "inventory receipt",
                "fetch receipt",
                "merge receipt",
            ),
            strict=True,
        ):
            if path.is_symlink() or not path.is_file():
                raise ExportError(f"{label} is missing or unsafe: {path}")
        production_paths = tuple(
            path.resolve() for path in raw_production_paths
        )
        production_provenance = _verify_exhaustive_export_provenance(
            store_root=resolved_store_root,
            store_receipt_path=resolved_store_receipt,
            fetch_state_path=resolved_fetch_state,
            inventory_path=production_paths[0],
            inventory_receipt_path=production_paths[1],
            fetch_receipt_path=production_paths[2],
            merge_receipt_path=production_paths[3],
        )
    if output_path.exists():
        raise ExportError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = ExactTokenizer(resolved_tokenizer)
    if tokenizer._tokenizer.vocab_size != EXPECTED_VOCAB_SIZE:
        raise ExportError("ExactTokenizer vocabulary differs from frozen CASE5")

    temp_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.partial-",
            dir=output_path.parent,
        )
    )
    published = False
    fragment_writer: CanonicalParquetLedgerWriter | None = None
    dropped_edge_writer: CanonicalParquetLedgerWriter | None = None
    representative_metadata_writer: CanonicalParquetLedgerWriter | None = None
    representative_ledger_writer: CanonicalParquetLedgerWriter | None = None
    excluded_opaque_writer: CanonicalParquetLedgerWriter | None = None
    excluded_training_scope_writer: CanonicalParquetLedgerWriter | None = None
    source_binding_projection_writer: CanonicalParquetLedgerWriter | None = None
    occurrence_metadata_writer: CanonicalParquetLedgerWriter | None = None
    eligibility_connection: sqlite3.Connection | None = None
    parquet_writers: dict[tuple[str, int], pq.ParquetWriter] = {}
    try:
        with (
            FrozenStore(
                resolved_store_root,
                resolved_store_receipt,
            ) as store,
            FrozenFetchState(
                resolved_fetch_state,
                tokenizer=tokenizer,
                store=store,
            ) as frozen_fetch_state,
        ):
            store_counters = _require_mapping(
                store.receipt.get("counters"),
                where="store receipt counters",
            )
            expected_content_count = _require_int(
                store_counters.get("unique_content_count"),
                where="store counters.unique_content_count",
                minimum=1,
            )
            expected_occurrence_count = _require_int(
                store_counters.get("occurrence_count"),
                where="store counters.occurrence_count",
                minimum=1,
            )
            expected_token_sequence_count = _require_int(
                store_counters.get("unique_token_sequence_count"),
                where="store counters.unique_token_sequence_count",
                minimum=1,
            )
            expected_unique_payload_tokens = _require_int(
                store_counters.get("exact_unique_payload_tokens"),
                where="store counters.exact_unique_payload_tokens",
                minimum=0,
            )
            cas_acquisition_target_tokens = _require_int(
                store.receipt.get("target_exact_unique_payload_tokens"),
                where="store receipt target_exact_unique_payload_tokens",
                minimum=0,
            )
            if required_eligible_exact_unique_payload_tokens is None:
                eligible_target_tokens = cas_acquisition_target_tokens
                eligible_target_source = "store_receipt"
            else:
                eligible_target_tokens = _require_int(
                    required_eligible_exact_unique_payload_tokens,
                    where="required eligible exact unique payload tokens",
                    minimum=0,
                )
                if eligible_target_tokens > cas_acquisition_target_tokens:
                    raise ExportError(
                        "required eligible exact unique payload tokens "
                        f"{eligible_target_tokens} exceed the receipt-bound CAS "
                        f"acquisition target {cas_acquisition_target_tokens}"
                    )
                eligible_target_source = "explicit_export_requirement"
            if expected_content_count < 1 or expected_occurrence_count < 1:
                raise ExportError("content store has no exportable occurrences")

            parser_lineage = frozen_fetch_state.parser_lineage()
            current_parser_sha256 = target_parser_script_sha256()
            if (
                require_current_parser_only
                and parser_lineage != (current_parser_sha256,)
            ):
                raise ExportError(
                    "current-parser-only export requires exactly one parser "
                    f"generation {current_parser_sha256}; observed lineage "
                    f"{list(parser_lineage)}"
                )
            source_binding_projector = SourceBindingProjectionRouter(
                parser_lineage,
                authorized_legacy_sha256=(
                    source_binding_projection_from_parser_sha256
                ),
            )
            source_binding_projection_writer = _source_binding_projection_writer(
                temp_path / "source_binding_projection.parquet"
            )
            occurrence_metadata_writer = CanonicalParquetLedgerWriter(
                temp_path / "occurrence_metadata.parquet",
                domain=OCCURRENCE_METADATA_LEDGER_DOMAIN,
                max_record_bytes=1024 * 1024,
            )
            source_binding_projection_counts: Counter[str] = Counter()
            previous_source_binding_projection_key: (
                tuple[str, str, str, str, int, int, int] | None
            ) = None

            eligibility_path = temp_path / ".eligibility.sqlite3"
            eligibility_connection = sqlite3.connect(eligibility_path)
            eligibility_connection.row_factory = sqlite3.Row
            eligibility_connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE TABLE seen_chunks (
                  repo TEXT NOT NULL,
                  run_id INTEGER NOT NULL,
                  attempt INTEGER NOT NULL,
                  archive_member TEXT NOT NULL,
                  chunk_ordinal INTEGER NOT NULL,
                  token_count INTEGER NOT NULL,
                  PRIMARY KEY (
                    repo,run_id,attempt,archive_member,chunk_ordinal
                  )
                );
                CREATE TABLE excluded_occurrences (
                  repo TEXT NOT NULL,
                  run_attempt TEXT NOT NULL,
                  job TEXT NOT NULL,
                  step TEXT NOT NULL,
                  chunk_ordinal INTEGER NOT NULL,
                  run_id INTEGER NOT NULL,
                  attempt INTEGER NOT NULL,
                  archive_member TEXT NOT NULL,
                  PRIMARY KEY (
                    repo,run_attempt,job,step,chunk_ordinal
                  )
                );
                CREATE TABLE occurrence_scope (
                  repo TEXT NOT NULL,
                  run_attempt TEXT NOT NULL,
                  job TEXT NOT NULL,
                  step TEXT NOT NULL,
                  chunk_ordinal INTEGER NOT NULL,
                  run_id INTEGER NOT NULL,
                  attempt INTEGER NOT NULL,
                  archive_member TEXT NOT NULL,
                  opaque INTEGER NOT NULL CHECK (opaque IN (0,1)),
                  local_primary INTEGER NOT NULL
                    CHECK (local_primary IN (0,1)),
                  local_aux_python INTEGER NOT NULL
                    CHECK (local_aux_python IN (0,1)),
                  local_aux_js_ts INTEGER NOT NULL
                    CHECK (local_aux_js_ts IN (0,1)),
                  effective_primary INTEGER NOT NULL DEFAULT 0
                    CHECK (effective_primary IN (0,1)),
                  effective_aux_python INTEGER NOT NULL DEFAULT 0
                    CHECK (effective_aux_python IN (0,1)),
                  effective_aux_js_ts INTEGER NOT NULL DEFAULT 0
                    CHECK (effective_aux_js_ts IN (0,1)),
                  decision_json TEXT NOT NULL,
                  PRIMARY KEY (
                    repo,run_attempt,job,step,chunk_ordinal
                  )
                );
                CREATE INDEX occurrence_scope_exact_step
                  ON occurrence_scope(repo,run_attempt,job,step);
                CREATE TABLE exact_step_scope (
                  repo TEXT NOT NULL,
                  run_attempt TEXT NOT NULL,
                  job TEXT NOT NULL,
                  step TEXT NOT NULL,
                  primary_route INTEGER NOT NULL
                    CHECK (primary_route IN (0,1)),
                  aux_python_route INTEGER NOT NULL
                    CHECK (aux_python_route IN (0,1)),
                  aux_js_ts_route INTEGER NOT NULL
                    CHECK (aux_js_ts_route IN (0,1)),
                  PRIMARY KEY (repo,run_attempt,job,step)
                ) WITHOUT ROWID;
                """
            )
            excluded_opaque_writer = CanonicalParquetLedgerWriter(
                temp_path / "excluded_opaque_artifacts.parquet",
                domain="cppmega-ci-case5-excluded-opaque-artifact-ledger-v1",
                max_record_bytes=1024 * 1024,
            )
            excluded_training_scope_writer = CanonicalParquetLedgerWriter(
                temp_path / "excluded_training_scope.parquet",
                domain=TRAINING_SCOPE_EXCLUSION_LEDGER_DOMAIN,
                max_record_bytes=1024 * 1024,
            )
            scope_policy = training_scope_policy()
            prevalidated_occurrence_count = 0
            excluded_opaque_occurrence_payload_tokens = 0
            eligibility_connection.execute("BEGIN")
            for occurrence in store.iter_occurrences():
                member = frozen_fetch_state.validate_occurrence(occurrence)
                raw_training = _require_mapping(
                    _require_mapping(
                        occurrence.provenance.get("chunk"),
                        where="source-binding projection chunk",
                    ).get("training_sidecars"),
                    where="source-binding projection training sidecars",
                )
                raw_actions = _require_list(
                    raw_training.get("build_actions"),
                    where="source-binding projection build_actions",
                )
                local_scope = classify_ci_training_sidecars(raw_training)
                local_scope_record = local_scope.as_dict()
                source_binding_projection_counts["occurrences"] += 1
                for action_index, raw_action in enumerate(raw_actions):
                    action = _require_mapping(
                        raw_action,
                        where=(
                            "source-binding projection "
                            f"build_actions[{action_index}]"
                        ),
                    )
                    projection = source_binding_projector.project_action(
                        occurrence_key=occurrence.key_dict,
                        provenance_sha256=occurrence.provenance_sha256,
                        provenance=occurrence.provenance,
                        action=action,
                        action_index=action_index,
                    )
                    source_binding_projection_counts["actions"] += 1
                    for record in projection.records:
                        projection_key = projection_record_key(record)
                        if (
                            previous_source_binding_projection_key is not None
                            and projection_key
                            <= previous_source_binding_projection_key
                        ):
                            raise ExportError(
                                "source-binding projection records are not in "
                                "strict canonical order"
                            )
                        previous_source_binding_projection_key = projection_key
                        source_binding_projection_writer.append(record)
                        source_binding_projection_counts["source_inputs"] += 1
                        if record["old_binding"] is not None:
                            source_binding_projection_counts[
                                "old_bindings"
                            ] += 1
                        if record["projected_binding"] is not None:
                            source_binding_projection_counts[
                                "projected_bindings"
                            ] += 1
                        source_binding_projection_counts[
                            f"change_kind:{record['change_kind']}"
                        ] += 1
                        source_binding_projection_counts[
                            f"selection_mode:{projection.selected_mode}"
                        ] += 1
                content = store.get_content_record(occurrence.content_sha256)
                try:
                    eligibility_connection.execute(
                        """
                        INSERT INTO seen_chunks(
                          repo,run_id,attempt,archive_member,
                          chunk_ordinal,token_count
                        ) VALUES (?,?,?,?,?,?)
                        """,
                        (
                            member.key[0],
                            member.key[1],
                            member.key[2],
                            member.key[3],
                            occurrence.key[4],
                            content.token_count,
                        ),
                    )
                    eligibility_connection.execute(
                        """
                        INSERT INTO occurrence_scope(
                          repo,run_attempt,job,step,chunk_ordinal,
                          run_id,attempt,archive_member,opaque,
                          local_primary,local_aux_python,local_aux_js_ts,
                          decision_json
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            *occurrence.key,
                            member.key[1],
                            member.key[2],
                            member.key[3],
                            int(member.opaque),
                            int(local_scope.primary),
                            int(local_scope.aux_python),
                            int(local_scope.aux_js_ts),
                            json.dumps(
                                local_scope_record,
                                allow_nan=False,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ExportError(
                        "CAS contains duplicate fetch-member chunk coverage"
                    ) from exc
                prevalidated_occurrence_count += 1
                if not member.opaque:
                    continue
                if member.exclusion_reason is None:
                    raise ExportError("opaque member has no exclusion reason")
                excluded_opaque_occurrence_payload_tokens += content.token_count
                excluded_opaque_writer.append(
                    {
                        "schema": OPAQUE_ARTIFACT_LEDGER_SCHEMA,
                        "occurrence_key": occurrence.key_dict,
                        "content_sha256": content.sha256,
                        "provenance_sha256": occurrence.provenance_sha256,
                        "token_sequence_sha256": content.token_sequence_sha256,
                        "exact_token_count": content.token_count,
                        "archive_member": member.key[3],
                        "archive_member_raw_sha256": member.raw_sha256,
                        "fetch_state_sidecar_sha256": member.sidecar_sha256,
                        "parser_sidecar_sha256": member.sidecar["sidecar_sha256"],
                        "reason": member.exclusion_reason,
                        "policy_schema": OPAQUE_ARTIFACT_POLICY_SCHEMA,
                        "decode_evidence": {
                            "status": member.decode_status,
                            "invalid_sequence_count": (member.invalid_sequence_count),
                            "replacement_char_count": (member.replacement_char_count),
                            "raw_byte_count": member.raw_size,
                            "invalid_ratio_ppm_floor": member.invalid_ratio_ppm,
                        },
                    }
                )
            eligibility_connection.commit()
            if prevalidated_occurrence_count != expected_occurrence_count:
                raise ExportError(
                    "fetch-state prevalidation occurrence count differs from CAS"
                )
            frozen_fetch_state.verify_member_coverage(eligibility_connection)

            # A parser signal may occur only in the command chunk while later
            # chunks carry its compile/test output.  Propagation is restricted
            # to the exact receipt-bound job section/step key.  It never crosses
            # a job, step, attempt, repository, or opaque member.
            eligibility_connection.executescript(
                """
                BEGIN;
                INSERT INTO exact_step_scope(
                  repo,run_attempt,job,step,
                  primary_route,aux_python_route,aux_js_ts_route
                )
                SELECT
                  repo,run_attempt,job,step,
                  MAX(local_primary),
                  MAX(local_aux_python),
                  MAX(local_aux_js_ts)
                FROM occurrence_scope
                WHERE opaque = 0
                GROUP BY repo,run_attempt,job,step;
                UPDATE occurrence_scope AS target
                SET effective_primary = CASE
                  WHEN target.opaque = 0 THEN COALESCE((
                      SELECT route.primary_route
                      FROM exact_step_scope AS route
                      WHERE route.repo = target.repo
                        AND route.run_attempt = target.run_attempt
                        AND route.job = target.job
                        AND route.step = target.step
                    ), 0) ELSE 0 END,
                    effective_aux_python = CASE
                  WHEN target.opaque = 0 AND COALESCE((
                    SELECT route.primary_route
                    FROM exact_step_scope AS route
                    WHERE route.repo = target.repo
                      AND route.run_attempt = target.run_attempt
                      AND route.job = target.job
                      AND route.step = target.step
                  ), 0) = 0 THEN COALESCE((
                    SELECT route.aux_python_route
                    FROM exact_step_scope AS route
                    WHERE route.repo = target.repo
                      AND route.run_attempt = target.run_attempt
                      AND route.job = target.job
                      AND route.step = target.step
                  ), 0) ELSE 0 END,
                    effective_aux_js_ts = CASE
                  WHEN target.opaque = 0 AND COALESCE((
                    SELECT route.primary_route
                    FROM exact_step_scope AS route
                    WHERE route.repo = target.repo
                      AND route.run_attempt = target.run_attempt
                      AND route.job = target.job
                      AND route.step = target.step
                  ), 0) = 0 THEN COALESCE((
                    SELECT route.aux_js_ts_route
                    FROM exact_step_scope AS route
                    WHERE route.repo = target.repo
                      AND route.run_attempt = target.run_attempt
                      AND route.job = target.job
                      AND route.step = target.step
                  ), 0) ELSE 0 END;
                INSERT INTO excluded_occurrences(
                  repo,run_attempt,job,step,chunk_ordinal,
                  run_id,attempt,archive_member
                )
                SELECT
                  repo,run_attempt,job,step,chunk_ordinal,
                  run_id,attempt,archive_member
                FROM occurrence_scope
                WHERE effective_primary = 0;
                COMMIT;
                """
            )

            training_scope_occurrence_counts: Counter[str] = Counter()
            training_scope_token_counts: Counter[str] = Counter()
            excluded_training_scope_occurrence_payload_tokens = 0
            scope_rows = iter(
                eligibility_connection.execute(
                    """
                    SELECT *
                    FROM occurrence_scope
                    ORDER BY repo,run_attempt,job,step,chunk_ordinal
                    """
                )
            )
            for occurrence in store.iter_occurrences():
                member = frozen_fetch_state.validate_occurrence(occurrence)
                content = store.get_content_record(occurrence.content_sha256)
                try:
                    scope_row = next(scope_rows)
                except StopIteration as exc:
                    raise ExportError(
                        "training scope decision disappeared after propagation"
                    ) from exc
                scope_key = (
                    str(scope_row["repo"]),
                    str(scope_row["run_attempt"]),
                    str(scope_row["job"]),
                    str(scope_row["step"]),
                    int(scope_row["chunk_ordinal"]),
                )
                if scope_key != occurrence.key:
                    raise ExportError(
                        "training scope decisions are not in occurrence order"
                    )
                local_decision = json.loads(str(scope_row["decision_json"]))
                if (
                    not isinstance(local_decision, dict)
                    or local_decision.get("schema")
                    != TRAINING_SCOPE_DECISION_SCHEMA
                ):
                    raise ExportError("stored training scope decision is malformed")
                effective_primary = bool(scope_row["effective_primary"])
                effective_aux_python = bool(
                    scope_row["effective_aux_python"]
                )
                effective_aux_js_ts = bool(scope_row["effective_aux_js_ts"])
                effective_routes: list[str] = []
                if effective_primary:
                    effective_routes.append(PRIMARY_ROUTE)
                else:
                    if effective_aux_python:
                        effective_routes.append(AUX_PYTHON_ROUTE)
                    if effective_aux_js_ts:
                        effective_routes.append(AUX_JS_TS_ROUTE)
                reasons = list(
                    _require_list(
                        local_decision.get("reasons"),
                        where="local training scope reasons",
                    )
                )
                if (
                    effective_primary
                    and not bool(local_decision.get("local_primary"))
                ):
                    reasons.append("propagated:exact_step_primary_evidence")
                if (
                    effective_aux_python
                    and not bool(local_decision.get("local_aux_python"))
                ):
                    reasons.append("propagated:exact_step_aux_python_evidence")
                if (
                    effective_aux_js_ts
                    and not bool(local_decision.get("local_aux_js_ts"))
                ):
                    reasons.append("propagated:exact_step_aux_js_ts_evidence")
                scope_decision: dict[str, object] = {
                    **local_decision,
                    "policy_schema": scope_policy["schema"],
                    "policy_sha256": scope_policy["sha256"],
                    "effective_primary": effective_primary,
                    "effective_aux_python": effective_aux_python,
                    "effective_aux_js_ts": effective_aux_js_ts,
                    "effective_routes": effective_routes,
                    "reasons": sorted(set(str(value) for value in reasons)),
                    "propagation": {
                        "schema": "cppmega_ci_exact_step_scope_propagation_v1",
                        "key": [
                            occurrence.key_dict["repo"],
                            occurrence.key_dict["run_attempt"],
                            occurrence.key_dict["job"],
                            occurrence.key_dict["step"],
                        ],
                        "opaque_members_never_inherit": True,
                    },
                }
                occurrence_metadata_writer.append(
                    _occurrence_metadata_record(
                        content=content,
                        occurrence=occurrence,
                        member=member,
                        scope_decision=scope_decision,
                        source_binding_projector=source_binding_projector,
                    )
                )
                if member.opaque:
                    route_status = "excluded_opaque"
                elif effective_primary:
                    route_status = PRIMARY_ROUTE
                elif effective_aux_python and effective_aux_js_ts:
                    route_status = "aux_python_js_ts"
                elif effective_aux_python:
                    route_status = AUX_PYTHON_ROUTE
                elif effective_aux_js_ts:
                    route_status = AUX_JS_TS_ROUTE
                else:
                    route_status = "excluded_irrelevant"
                training_scope_occurrence_counts[route_status] += 1
                training_scope_token_counts[route_status] += content.token_count
                if not effective_primary and not member.opaque:
                    excluded_training_scope_occurrence_payload_tokens += (
                        content.token_count
                    )
                    excluded_training_scope_writer.append(
                        {
                            "schema": TRAINING_SCOPE_EXCLUSION_LEDGER_SCHEMA,
                            "occurrence_key": occurrence.key_dict,
                            "content_sha256": content.sha256,
                            "provenance_sha256": (
                                occurrence.provenance_sha256
                            ),
                            "token_sequence_sha256": (
                                content.token_sequence_sha256
                            ),
                            "exact_token_count": content.token_count,
                            "effective_routes": effective_routes,
                            "scope_decision": scope_decision,
                        }
                    )
            try:
                next(scope_rows)
            except StopIteration:
                pass
            else:
                raise ExportError(
                    "training scope has decisions without CAS occurrences"
                )
            excluded_opaque_writer.close()
            excluded_training_scope_writer.close()
            source_binding_projection_writer.close()
            occurrence_metadata_writer.close()
            if (
                source_binding_projection_counts["occurrences"]
                != expected_occurrence_count
                or source_binding_projection_counts["source_inputs"]
                != source_binding_projection_writer.count
                or occurrence_metadata_writer.count
                != expected_occurrence_count
            ):
                raise ExportError(
                    "occurrence sidecar coverage differs from the CAS"
                )
            excluded_member_count = int(
                eligibility_connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT repo,run_id,attempt,archive_member
                      FROM excluded_occurrences
                      GROUP BY repo,run_id,attempt,archive_member
                    )
                    """
                ).fetchone()[0]
            )
            excluded_occurrence_count = int(
                eligibility_connection.execute(
                    "SELECT COUNT(*) FROM excluded_occurrences"
                ).fetchone()[0]
            )
            excluded_opaque_member_count = int(
                eligibility_connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT repo,run_id,attempt,archive_member
                      FROM occurrence_scope
                      WHERE opaque=1
                      GROUP BY repo,run_id,attempt,archive_member
                    )
                    """
                ).fetchone()[0]
            )
            excluded_training_scope_member_count = int(
                eligibility_connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT repo,run_id,attempt,archive_member
                      FROM occurrence_scope
                      WHERE opaque=0 AND effective_primary=0
                      GROUP BY repo,run_id,attempt,archive_member
                    )
                    """
                ).fetchone()[0]
            )

            ledger_path = temp_path / "representative_ledger.parquet"
            representative_ledger_writer = CanonicalParquetLedgerWriter(
                ledger_path,
                domain="cppmega-ci-case5-representative-ledger-v1",
                max_record_bytes=1024 * 1024,
            )
            parser_sidecar_sequence = CanonicalSequenceHasher(
                domain="cppmega-ci-parser-sidecar-occurrence-sequence-v1"
            )
            content_count = 0
            validated_occurrence_count = 0
            expected_payload = 0
            cas_unique_sequence_count = 0
            cas_unique_payload_tokens = 0
            excluded_unique_sequence_count = 0
            excluded_unique_payload_tokens = 0
            current_sequence_sha256: str | None = None
            current_sequence_token_count: int | None = None
            candidate_content_count = 0
            candidate_occurrence_count = 0
            candidate_content_hasher: CanonicalSequenceHasher | None = None
            representative_content: ContentRecord | None = None
            representative_occurrence: OccurrenceRecord | None = None

            def finish_representative_group() -> None:
                nonlocal cas_unique_payload_tokens
                nonlocal cas_unique_sequence_count
                nonlocal excluded_unique_payload_tokens
                nonlocal excluded_unique_sequence_count
                nonlocal expected_payload
                if current_sequence_sha256 is None:
                    return
                if (
                    candidate_content_hasher is None
                    or current_sequence_token_count is None
                ):
                    raise ExportError("representative selection state is incomplete")
                cas_unique_sequence_count += 1
                cas_unique_payload_tokens += current_sequence_token_count
                if representative_content is None or representative_occurrence is None:
                    if candidate_content_count or candidate_occurrence_count:
                        raise ExportError(
                            "excluded representative group has eligible candidates"
                        )
                    excluded_unique_sequence_count += 1
                    excluded_unique_payload_tokens += current_sequence_token_count
                    return
                representative_ledger_writer.append(
                    {
                        "schema": REPRESENTATIVE_LEDGER_SCHEMA,
                        "token_sequence_sha256": current_sequence_sha256,
                        "token_count": current_sequence_token_count,
                        "candidate_content_count": candidate_content_count,
                        "candidate_occurrence_count": candidate_occurrence_count,
                        "candidate_content_sha256_sequence_sha256": (
                            candidate_content_hasher.sha256
                        ),
                        "representative_content_sha256": (
                            representative_content.sha256
                        ),
                        "representative_occurrence_key": (
                            representative_occurrence.key_dict
                        ),
                        "representative_provenance_sha256": (
                            representative_occurrence.provenance_sha256
                        ),
                    }
                )
                expected_payload += current_sequence_token_count

            for content in store.iter_contents(by_token_sequence=True):
                content_count += 1
                if content.tokenizer_fingerprint != tokenizer.fingerprint:
                    raise ExportError(
                        f"content {content.sha256} tokenizer fingerprint mismatch"
                    )
                raw = store.read_content(content)
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ExportError(
                        f"content {content.sha256} is not strict UTF-8"
                    ) from exc
                actual_ids = tokenizer.encode_batch([text])[0]
                if (
                    len(actual_ids) != content.token_count
                    or hash_token_sequence(actual_ids) != content.token_sequence_sha256
                ):
                    raise ExportError(
                        f"content {content.sha256} exact token metadata mismatch"
                    )
                if current_sequence_sha256 != content.token_sequence_sha256:
                    finish_representative_group()
                    current_sequence_sha256 = content.token_sequence_sha256
                    current_sequence_token_count = content.token_count
                    candidate_content_count = 0
                    candidate_occurrence_count = 0
                    candidate_content_hasher = CanonicalSequenceHasher(
                        domain="cppmega-ci-candidate-content-sha256-sequence-v1"
                    )
                    representative_content = None
                    representative_occurrence = None
                elif current_sequence_token_count != content.token_count:
                    raise ExportError(
                        "one token-sequence digest has contradictory token counts"
                    )
                if candidate_content_hasher is None:
                    raise ExportError("candidate content hasher was not initialized")

                first_eligible_occurrence: OccurrenceRecord | None = None
                total_content_occurrence_count = 0
                eligible_content_occurrence_count = 0
                for occurrence in store.iter_occurrences_for_content(content.sha256):
                    total_content_occurrence_count += 1
                    validated_occurrence_count += 1
                    _validate_occurrence_v3(occurrence, content_text=text)
                    parser_sidecar_sequence.append(
                        occurrence.provenance["parser_sidecar_sha256"]
                    )
                    excluded = eligibility_connection.execute(
                        """
                        SELECT 1 FROM excluded_occurrences
                        WHERE repo=? AND run_attempt=? AND job=?
                          AND step=? AND chunk_ordinal=?
                        """,
                        occurrence.key,
                    ).fetchone()
                    if excluded is None:
                        if first_eligible_occurrence is None:
                            first_eligible_occurrence = occurrence
                        eligible_content_occurrence_count += 1
                if total_content_occurrence_count == 0:
                    raise ExportError(
                        f"content {content.sha256} has no occurrence provenance"
                    )
                if first_eligible_occurrence is not None:
                    if representative_content is None:
                        representative_content = content
                        representative_occurrence = first_eligible_occurrence
                    candidate_content_hasher.append(content.sha256)
                    candidate_content_count += 1
                    candidate_occurrence_count += eligible_content_occurrence_count
            finish_representative_group()
            representative_ledger_writer.close()
            if (
                content_count != expected_content_count
                or validated_occurrence_count != expected_occurrence_count
                or parser_sidecar_sequence.count != expected_occurrence_count
                or cas_unique_sequence_count != expected_token_sequence_count
                or cas_unique_payload_tokens != expected_unique_payload_tokens
                or (
                    expected_payload + excluded_unique_payload_tokens
                    != cas_unique_payload_tokens
                )
                or (
                    representative_ledger_writer.count + excluded_unique_sequence_count
                    != cas_unique_sequence_count
                )
            ):
                raise ExportError(
                    "streamed content/occurrence/token-sequence totals differ "
                    "from the receipt"
                )
            if expected_payload < eligible_target_tokens:
                raise ExportError(
                    "eligible exact unique payload tokens "
                    f"{expected_payload} are below target {eligible_target_tokens}"
                )
            representative_count = representative_ledger_writer.count
            if representative_count < 1:
                raise ExportError(
                    "primary training-scope policy excluded every token sequence"
                )
            ledger_digest = representative_ledger_writer.logical_sha256
            eligibility_connection.close()
            eligibility_connection = None
            eligibility_path.unlink()

            fragment_writer = CanonicalParquetLedgerWriter(
                temp_path / "fragment_ledger.parquet",
                max_record_bytes=1024 * 1024,
            )
            dropped_edge_writer = CanonicalParquetLedgerWriter(
                temp_path / "dropped_graph_edges.parquet",
                max_record_bytes=1024 * 1024,
            )
            representative_metadata_writer = CanonicalParquetLedgerWriter(
                temp_path / "representative_metadata.parquet",
                max_record_bytes=1024 * 1024,
            )
            row_buffers: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
            row_counts: Counter[tuple[str, int]] = Counter()
            shard_counts: Counter[tuple[str, int]] = Counter()
            shard_row_counts: Counter[tuple[str, int]] = Counter()
            shard_paths: dict[tuple[str, int], Path] = {}
            parquet_paths: list[Path] = []

            def close_shard(key: tuple[str, int]) -> None:
                writer = parquet_writers.pop(key, None)
                if writer is None:
                    return
                writer.close()
                parquet_paths.append(shard_paths.pop(key))
                shard_counts[key] += 1
                shard_row_counts[key] = 0

            def flush_rows(key: tuple[str, int]) -> None:
                buffered = row_buffers[key]
                if not buffered:
                    return
                split_name, bucket_size = key
                table = rows_to_table(buffered)
                if key not in parquet_writers:
                    # Keep the immutable export directly consumable by the
                    # production bundle builder.  The split remains encoded in
                    # the collision-free filename, while the bucket-first
                    # directory layout preserves the canonical
                    # ``<kind>/<bucket>/*.parquet`` snapshot geometry.
                    directory = temp_path / str(bucket_size)
                    directory.mkdir(parents=True, exist_ok=True)
                    shard_index = shard_counts[key]
                    path = directory / (
                        f"ci-case5-{split_name}-{bucket_size}-{shard_index:06d}.parquet"
                    )
                    parquet_writers[key] = pq.ParquetWriter(
                        path,
                        table.schema,
                        compression="zstd",
                        compression_level=PARQUET_ZSTD_LEVEL,
                    )
                    shard_paths[key] = path
                parquet_writers[key].write_table(
                    table,
                    row_group_size=len(buffered),
                )
                shard_row_counts[key] += len(buffered)
                buffered.clear()
                if shard_row_counts[key] >= PARQUET_SHARD_ROWS:
                    if shard_row_counts[key] != PARQUET_SHARD_ROWS:
                        raise ExportError("Parquet shard row bound was exceeded")
                    close_shard(key)

            count_totals: Counter[str] = Counter()
            graph_totals: Counter[str] = Counter()
            graph_by_family: Counter[str] = Counter()
            source_doc_index = 0

            for ledger_record, _encoded in iter_canonical_parquet_ledger(
                ledger_path,
                expected_domain=(
                    "cppmega-ci-case5-representative-ledger-v1"
                ),
                expected_record_schema=REPRESENTATIVE_LEDGER_SCHEMA,
                max_record_bytes=1024 * 1024,
            ):
                representative_content = store.get_content_record(
                    _require_hex64(
                        ledger_record.get("representative_content_sha256"),
                        where="representative_content_sha256",
                    )
                )
                raw_occurrence_key = _require_mapping(
                    ledger_record.get("representative_occurrence_key"),
                    where="representative_occurrence_key",
                )
                representative_occurrence = store.get_occurrence(
                    (
                        _require_nonempty_string(
                            raw_occurrence_key.get("repo"),
                            where="representative_occurrence_key.repo",
                        ),
                        _require_nonempty_string(
                            raw_occurrence_key.get("run_attempt"),
                            where="representative_occurrence_key.run_attempt",
                        ),
                        _require_nonempty_string(
                            raw_occurrence_key.get("job"),
                            where="representative_occurrence_key.job",
                        ),
                        _require_nonempty_string(
                            raw_occurrence_key.get("step"),
                            where="representative_occurrence_key.step",
                        ),
                        _require_int(
                            raw_occurrence_key.get("chunk_ordinal"),
                            where="representative_occurrence_key.chunk_ordinal",
                            minimum=0,
                        ),
                    )
                )
                if (
                    representative_occurrence.content_sha256
                    != representative_content.sha256
                    or representative_occurrence.provenance_sha256
                    != ledger_record.get("representative_provenance_sha256")
                    or representative_content.token_sequence_sha256
                    != ledger_record.get("token_sequence_sha256")
                ):
                    raise ExportError(
                        "representative ledger no longer binds its store records"
                    )
                try:
                    text = store.read_content(representative_content).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ExportError(
                        "representative content ceased to be strict UTF-8"
                    ) from exc
                chunk = _require_mapping(
                    representative_occurrence.provenance["chunk"],
                    where="representative provenance chunk",
                )
                representative_fetch_member = frozen_fetch_state.validate_occurrence(
                    representative_occurrence
                )
                if representative_fetch_member.opaque:
                    raise ExportError(
                        "excluded opaque occurrence reached CASE5 projection"
                    )
                representative_metadata_writer.append(
                    _representative_metadata_record(
                        content=representative_content,
                        occurrence=representative_occurrence,
                        parser_sidecar=representative_fetch_member.sidecar,
                        source_binding_projector=source_binding_projector,
                    )
                )
                projected = _project_content(
                    tokenizer=tokenizer,
                    text=text,
                    chunk=chunk,
                )
                if (
                    len(projected.token_ids) != representative_content.token_count
                    or hash_token_sequence(projected.token_ids)
                    != representative_content.token_sequence_sha256
                ):
                    raise ExportError("offset projection changed exact payload tokens")
                ranges = _fragment_ranges(projected.token_domain_ids)
                if not ranges:
                    raise ExportError("empty token sequence cannot produce CASE5 data")
                split = _split_for_sequence(
                    representative_content.token_sequence_sha256
                )
                fragment_for_token: dict[int, int] = {}
                for fragment_index, (start, end) in enumerate(ranges):
                    for token_index in range(start, end):
                        fragment_for_token[token_index] = fragment_index

                source = source_identity(
                    {
                        "source_doc_id": (
                            "ci-token-sequence:"
                            f"{representative_content.token_sequence_sha256}"
                        )
                    }
                )
                for edge in projected.edges:
                    family = str(edge["family"])
                    graph_totals["input_in_chunk_edges"] += 1
                    graph_by_family[f"input_{family}"] += 1
                emitted_for_sequence = 0
                for fragment_index, (start, end) in enumerate(ranges):
                    record, index_map = _frame_fragment(
                        projected=projected,
                        start=start,
                        end=end,
                        source_identity_id=source.source_identity_id,
                    )
                    _verify_framed_payload(
                        record=record,
                        index_map=index_map,
                        projected=projected,
                        start=start,
                        end=end,
                    )
                    record[SOURCE_IDENTITY_REGISTRY_COLUMN] = [source.as_dict()]
                    emitted_edges = 0
                    for edge in projected.edges:
                        edge_from = int(edge["from"])
                        edge_to = int(edge["to"])
                        from_fragment = fragment_for_token[edge_from]
                        to_fragment = fragment_for_token[edge_to]
                        family = str(edge["family"])
                        if from_fragment != to_fragment:
                            if fragment_index == min(from_fragment, to_fragment):
                                graph_totals["cross_fragment_edges_dropped"] += 1
                                graph_by_family[f"cross_fragment_dropped_{family}"] += 1
                                dropped_edge_writer.append(
                                    {
                                        "reason": "cross-fragment",
                                        "token_sequence_sha256": (
                                            representative_content.token_sequence_sha256
                                        ),
                                        "edge_id": edge["edge_id"],
                                        "family": family,
                                        "kind": int(edge["kind"]),
                                        "from_fragment": from_fragment,
                                        "to_fragment": to_fragment,
                                    }
                                )
                            continue
                        if from_fragment != fragment_index:
                            continue
                        column = _EDGE_COLUMN_BY_FAMILY[family]
                        record[column].append(
                            {
                                "from": index_map[edge_from],
                                "to": index_map[edge_to],
                                "kind": int(edge["kind"]),
                            }
                        )
                        emitted_edges += 1
                        graph_totals["emitted_edges"] += 1
                        graph_by_family[f"emitted_{family}"] += 1

                    if not record[TOKEN_DIAGNOSTIC_EDGES_COLUMN] and any(
                        int(domain) >= int(DomainKind.COMPILER_DIAGNOSTIC)
                        for domain in record[TOKEN_DOMAIN_IDS_COLUMN]
                    ):
                        for index, domain in enumerate(record[TOKEN_DOMAIN_IDS_COLUMN]):
                            if int(domain) >= int(
                                DomainKind.COMPILER_DIAGNOSTIC
                            ) and int(record[TOKEN_ROLE_IDS_COLUMN][index]) != int(
                                DomainRoleKind.DELIMITER
                            ):
                                record[TOKEN_CONFIDENCE_IDS_COLUMN][index] = int(
                                    ParseConfidence.RAW
                                )

                    record["source_doc_id"] = (
                        f"ci:{representative_content.token_sequence_sha256}:"
                        f"{fragment_index}"
                    )
                    doc = normalize_document_record(
                        record,
                        source_doc_index=source_doc_index,
                        stable_doc_id=_stable_doc_id(
                            representative_content.token_sequence_sha256,
                            fragment_index,
                        ),
                    )
                    framed_tokens = len(doc.token_ids)
                    bucket = _smallest_bucket(framed_tokens)
                    packed, overflow = pack_documents(
                        [doc],
                        target_length=bucket,
                        pad_token_id=0,
                        strategy="sequential",
                    )
                    if overflow or len(packed) != 1:
                        raise ExportError("pre-fragmented CASE5 row overflowed")
                    packed_row = packed[0]
                    if (
                        len(packed_row["input_ids"]) != bucket
                        or int(packed_row["valid_token_count"]) != framed_tokens
                    ):
                        raise ExportError("CASE5 packer changed fixed-width framing")
                    _verify_packed_single_document(
                        packed_row=packed_row,
                        doc=doc,
                    )
                    row_key = (split, bucket)
                    row_index_within_split_bucket = row_counts[row_key]
                    packed_row["pack_id"] = row_index_within_split_bucket
                    row_buffers[row_key].append(packed_row)
                    row_counts[row_key] += 1
                    buffered_row_limit = min(
                        PARQUET_SHARD_ROWS,
                        max(1, PARQUET_SHARD_TOKEN_BUDGET // bucket),
                    )
                    if len(row_buffers[row_key]) >= buffered_row_limit:
                        flush_rows(row_key)
                    payload_count = end - start
                    framing_count = framed_tokens - payload_count
                    padding_count = bucket - framed_tokens
                    trained_count = int(packed_row["trained_token_count"])
                    count_totals["fragments"] += 1
                    count_totals["payload_tokens"] += payload_count
                    count_totals["framing_tokens"] += framing_count
                    count_totals["valid_tokens"] += framed_tokens
                    count_totals["trained_tokens"] += trained_count
                    count_totals["padding_tokens"] += padding_count
                    count_totals["capacity_tokens"] += bucket
                    emitted_for_sequence += payload_count
                    fragment_writer.append(
                        {
                            "token_sequence_sha256": (
                                representative_content.token_sequence_sha256
                            ),
                            "fragment_index": fragment_index,
                            "fragment_count": len(ranges),
                            "split": split,
                            "bucket": bucket,
                            "row_index_within_split_bucket": (
                                row_index_within_split_bucket
                            ),
                            "payload_start": start,
                            "payload_end": end,
                            "payload_tokens": payload_count,
                            "framing_tokens": framing_count,
                            "valid_tokens": framed_tokens,
                            "trained_tokens": trained_count,
                            "padding_tokens": padding_count,
                            "emitted_graph_edges": emitted_edges,
                        }
                    )
                    source_doc_index += 1
                if emitted_for_sequence != representative_content.token_count:
                    raise ExportError("fragmentation did not conserve payload tokens")
                count_totals["representatives"] += 1

                training = _require_mapping(
                    chunk["training_sidecars"], where="training sidecars"
                )
                cross_accounting = _require_mapping(
                    training["cross_chunk_edge_accounting"],
                    where="cross-chunk accounting",
                )
                cross_reference_count = int(cross_accounting["count"])
                outbound_count = int(cross_accounting["outbound_count"])
                graph_totals["cross_chunk_reference_count_source_reported"] += (
                    cross_reference_count
                )
                graph_totals["cross_chunk_outbound_reference_count_validated"] += (
                    outbound_count
                )
                graph_totals["cross_chunk_non_outbound_reference_count_unresolved"] += (
                    cross_reference_count - outbound_count
                )
                graph_totals["cross_chunk_outbound_edges_dropped"] += len(
                    projected.cross_chunk_edges
                )
                dropped_edge_writer.append(
                    {
                        "reason": "cross-chunk-source-accounting",
                        "token_sequence_sha256": (
                            representative_content.token_sequence_sha256
                        ),
                        "source_reported_reference_count": cross_reference_count,
                        "validated_outbound_count": outbound_count,
                        "unresolved_non_outbound_count": (
                            cross_reference_count - outbound_count
                        ),
                        "source_accounting_sha256": str(cross_accounting["sha256"]),
                    }
                )
                for edge in projected.cross_chunk_edges:
                    family = domain_edge_family(int(edge["kind_id"]))
                    graph_by_family[f"cross_chunk_dropped_{family}"] += 1
                    dropped_edge_writer.append(
                        {
                            "reason": "cross-chunk",
                            "token_sequence_sha256": (
                                representative_content.token_sequence_sha256
                            ),
                            "edge_id": str(edge.get("edge_id", "")),
                            "family": family,
                            "kind": int(edge["kind_id"]),
                        }
                    )

            for row_key in sorted(row_buffers):
                flush_rows(row_key)
            for row_key in sorted(tuple(parquet_writers)):
                close_shard(row_key)
            fragment_writer.close()
            dropped_edge_writer.close()
            representative_metadata_writer.close()
            if (
                int(count_totals["representatives"]) != representative_count
                or representative_metadata_writer.count != representative_count
            ):
                raise ExportError("representative artifact counts disagree")

            if count_totals["payload_tokens"] != expected_payload:
                raise ExportError("global representative payload conservation failed")
            if count_totals["capacity_tokens"] != (
                count_totals["valid_tokens"] + count_totals["padding_tokens"]
            ):
                raise ExportError("global CASE5 capacity accounting failed")
            if graph_totals["input_in_chunk_edges"] != (
                graph_totals["emitted_edges"]
                + graph_totals["cross_fragment_edges_dropped"]
            ):
                raise ExportError("in-chunk graph accounting is not exhaustive")
            for key in (
                "input_in_chunk_edges",
                "emitted_edges",
                "cross_fragment_edges_dropped",
                "cross_chunk_reference_count_source_reported",
                "cross_chunk_outbound_reference_count_validated",
                "cross_chunk_non_outbound_reference_count_unresolved",
                "cross_chunk_outbound_edges_dropped",
            ):
                graph_totals.setdefault(key, 0)
            for family in _EDGE_COLUMN_BY_FAMILY:
                for prefix in (
                    "input",
                    "emitted",
                    "cross_fragment_dropped",
                    "cross_chunk_dropped",
                ):
                    graph_by_family.setdefault(f"{prefix}_{family}", 0)

            artifact_records: list[dict[str, Any]] = []
            audits: list[dict[str, Any]] = []
            for path in sorted(parquet_paths):
                bucket = int(path.parent.name)
                match = re.fullmatch(
                    rf"ci-case5-(train|validation|test)-{bucket}-[0-9]{{6}}\.parquet",
                    path.name,
                )
                if match is None:
                    raise ExportError(f"generated CASE5 path is not canonical: {path}")
                split = match.group(1)
                parquet_metadata = pq.ParquetFile(path).metadata
                parquet_rows = int(parquet_metadata.num_rows)
                parquet_codecs = {
                    str(
                        parquet_metadata.row_group(row_group)
                        .column(column)
                        .compression
                    )
                    for row_group in range(parquet_metadata.num_row_groups)
                    for column in range(parquet_metadata.num_columns)
                }
                if parquet_codecs != {"ZSTD"}:
                    raise ExportError(
                        f"generated CASE5 Parquet is not ZSTD-compressed: {path}"
                    )
                audit_result = _audit_file(
                    str(path), "ci", str(bucket), EXPECTED_VOCAB_SIZE
                )
                audit = _require_mapping(
                    audit_result.get("stats"), where=f"{path} CASE5 audit"
                )
                if int(audit["bad_files"]) or int(audit["bad_rows"]):
                    raise ExportError(
                        f"generated CASE5 audit failed for {path}: "
                        f"{list(audit['errors'])[:8]}"
                    )
                if int(audit["capacity_tokens"]) != bucket * parquet_rows:
                    raise ExportError(f"{path} does not have fixed bucket width")
                artifact_records.append(
                    {
                        "path": path.relative_to(temp_path).as_posix(),
                        "kind": "case5_parquet",
                        "split": split,
                        "bucket": bucket,
                        "rows": parquet_rows,
                        "byte_size": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
                audits.append(
                    {
                        "path": path.relative_to(temp_path).as_posix(),
                        "rows": int(audit["rows"]),
                        "capacity_tokens": int(audit["capacity_tokens"]),
                        "valid_tokens": int(audit["valid_tokens"]),
                        "trained_tokens": int(audit["trained_tokens"]),
                        "bad_files": int(audit["bad_files"]),
                        "bad_rows": int(audit["bad_rows"]),
                    }
                )

            for path, kind, ledger_rows in (
                (ledger_path, "representative_ledger", representative_count),
                (
                    fragment_writer.path,
                    "fragment_ledger",
                    fragment_writer.count,
                ),
                (
                    dropped_edge_writer.path,
                    "dropped_graph_edges",
                    dropped_edge_writer.count,
                ),
                (
                    representative_metadata_writer.path,
                    "representative_metadata",
                    representative_metadata_writer.count,
                ),
                (
                    excluded_opaque_writer.path,
                    "excluded_opaque_artifacts",
                    excluded_opaque_writer.count,
                ),
                (
                    excluded_training_scope_writer.path,
                    "excluded_training_scope",
                    excluded_training_scope_writer.count,
                ),
                (
                    source_binding_projection_writer.path,
                    "source_binding_projection",
                    source_binding_projection_writer.count,
                ),
                (
                    occurrence_metadata_writer.path,
                    "occurrence_metadata",
                    occurrence_metadata_writer.count,
                ),
            ):
                artifact_records.append(
                    {
                        "path": path.relative_to(temp_path).as_posix(),
                        "kind": kind,
                        "rows": ledger_rows,
                        "byte_size": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
            input_snapshot = [
                {
                    "path": item.relative_path,
                    "byte_size": item.size,
                    "mtime_ns": item.mtime_ns,
                    "inode": item.inode,
                    "sha256": item.sha256,
                }
                for item in store._initial_snapshot
            ]
            receipt = {
                "schema": (
                    PRODUCTION_EXPORT_SCHEMA
                    if completion_mode
                    == COMPLETION_MODE_INVENTORY_EXHAUSTIVE
                    else EXPORT_SCHEMA
                ),
                "status": "complete",
                "exporter_script_sha256": exporter_script_sha256,
                "input_store": {
                    "schema": STORE_SCHEMA,
                    "receipt_schema": STORE_RECEIPT_SCHEMA,
                    "receipt_sha256": store.receipt_sha256,
                    "policy_sha256": store.receipt["policy_sha256"],
                    "sqlite_schema_sha256": store.receipt["sqlite_schema_sha256"],
                    "logical_content_set_sha256": store.receipt[
                        "logical_content_set_sha256"
                    ],
                    "logical_token_sequence_set_sha256": store.receipt[
                        "logical_token_sequence_set_sha256"
                    ],
                    "occurrence_set_sha256": store.receipt["occurrence_set_sha256"],
                    "sqlite_logical_sha256": store.receipt["sqlite_logical_sha256"],
                    "pack_hashes": store.receipt["pack_hashes"],
                    "frozen_files": input_snapshot,
                    "verified_before_export": True,
                    "unchanged_after_export": True,
                },
                "input_fetch_state": frozen_fetch_state.receipt_binding(),
                "parser_generation_policy": {
                    "mode": (
                        "current-singleton-required"
                        if require_current_parser_only
                        else "audited-lineage"
                    ),
                    "expected_current_parser_script_sha256": (
                        current_parser_sha256
                    ),
                    "observed_parser_lineage": list(parser_lineage),
                    "current_singleton": parser_lineage
                    == (current_parser_sha256,),
                },
                "source_binding_projection": {
                    "schema": SOURCE_BINDING_PROJECTION_SCHEMA,
                    "mode": source_binding_projector.mode,
                    "projection_script_sha256": (
                        source_binding_projection_script_sha256
                    ),
                    "input_parser_script_sha256": (
                        source_binding_projector.input_parser_sha256
                    ),
                    "target_parser_script_sha256": (
                        source_binding_projector.target_parser_sha256
                    ),
                    **(
                        {
                            "parser_lineage": list(
                                source_binding_projector.parser_lineage
                            ),
                            "selection_policy": (
                                source_binding_projector.SELECTION_POLICY
                            ),
                            "selection_counts": {
                                key.removeprefix("selection_mode:"): value
                                for key, value in sorted(
                                    source_binding_projection_counts.items()
                                )
                                if key.startswith("selection_mode:")
                            },
                        }
                        if source_binding_projector.mode
                        == SourceBindingProjectionRouter.MIXED_MODE
                        else {}
                    ),
                    "input_occurrence_set_sha256": store.receipt[
                        "occurrence_set_sha256"
                    ],
                    "input_fetch_state_sqlite_logical_sha256": (
                        frozen_fetch_state.sqlite_logical_sha256
                    ),
                    "input_fetch_state_sidecar_set_sha256": (
                        frozen_fetch_state.sidecar_set_sha256
                    ),
                    "coverage": {
                        "order": (
                            "occurrence-key-then-action-index-then-source-input-index"
                        ),
                        "occurrence_count": source_binding_projection_counts[
                            "occurrences"
                        ],
                        "action_count": source_binding_projection_counts["actions"],
                        "source_input_count": source_binding_projection_counts[
                            "source_inputs"
                        ],
                        "old_binding_count": source_binding_projection_counts[
                            "old_bindings"
                        ],
                        "projected_binding_count": (
                            source_binding_projection_counts[
                                "projected_bindings"
                            ]
                        ),
                    },
                    "change_counts": {
                        key.removeprefix("change_kind:"): value
                        for key, value in sorted(
                            source_binding_projection_counts.items()
                        )
                        if key.startswith("change_kind:")
                    },
                    "ledger_artifact": (
                        source_binding_projection_writer.path.relative_to(
                            temp_path
                        ).as_posix()
                    ),
                    "ledger_record_count": source_binding_projection_writer.count,
                    "ledger_sha256": (
                        source_binding_projection_writer.logical_sha256
                    ),
                    "ledger_artifact_sha256": _sha256_file(
                        source_binding_projection_writer.path
                    ),
                    "claim_boundary": (
                        "derived source-binding semantics only; upstream parser "
                        "sidecars, parser hashes, occurrence provenance, payload "
                        "bytes, token IDs, token counts and CAS receipts are unchanged"
                    ),
                },
                "occurrence_metadata": {
                    "schema": OCCURRENCE_METADATA_SCHEMA,
                    "scope": "one-record-per-frozen-cas-occurrence",
                    "count": occurrence_metadata_writer.count,
                    "input_occurrence_set_sha256": store.receipt[
                        "occurrence_set_sha256"
                    ],
                    "logical_domain": OCCURRENCE_METADATA_LEDGER_DOMAIN,
                    "logical_sha256": occurrence_metadata_writer.logical_sha256,
                    "artifact": occurrence_metadata_writer.path.relative_to(
                        temp_path
                    ).as_posix(),
                    "artifact_sha256": _sha256_file(
                        occurrence_metadata_writer.path
                    ),
                    "physical_format": {
                        "container": "parquet",
                        "compression": "zstd",
                        "record_encoding": "canonical-json",
                    },
                    "claim_boundary": (
                        "sanitized occurrence API metadata and chunk-local "
                        "training sidecars; raw command and diagnostic message "
                        "text are omitted"
                    ),
                },
                "tokenizer": {
                    "exact_tokenizer_schema": tokenizer.contract["schema"],
                    "fingerprint": tokenizer.fingerprint,
                    "artifact_sha256": tokenizer.artifact_sha256,
                    "contract": tokenizer.contract,
                    "payload_token_sequence_encoding": TOKEN_SEQUENCE_ENCODING,
                },
                "case5_contract": {
                    "schema": CASE5_SCHEMA_VERSION,
                    "parquet_layout": "bucket-first-split-in-filename-v1",
                    "domain_delimiter_contract_sha256": (
                        DOMAIN_DELIMITER_CONTRACT_SHA256
                    ),
                    "domain_schema_sha256": DOMAIN_SCHEMA_SHA256,
                    "tokenizer_contract_sha256": TOKENIZER_CONTRACT_SHA256,
                    "buckets": list(BUCKETS),
                    "parquet_shard_max_rows": PARQUET_SHARD_ROWS,
                    "parquet_compression": {
                        "codec": "zstd",
                        "level": PARQUET_ZSTD_LEVEL,
                    },
                    "parquet_buffer_token_budget_per_split_bucket": (
                        PARQUET_SHARD_TOKEN_BUDGET
                    ),
                    "pad_token_id": 0,
                    "bos_token_id": tokenizer_bos_id(),
                    "overflow_rows": 0,
                    "split": SPLIT_CONTRACT,
                    "framing": ("bos-then-balanced-contiguous-domain-runs-v1"),
                    "confidence_projection": {
                        "schema": "cppmega_ci_confidence_projection_v1",
                        "mapping": {
                            "score_eq_0": int(ParseConfidence.ABSENT),
                            "score_gt_0_lt_0_8": int(ParseConfidence.HEURISTIC),
                            "score_gte_0_8_lt_1": int(ParseConfidence.PARTIAL),
                            "score_gte_1": int(ParseConfidence.EXACT),
                            "edge_free_diagnostic_payload": int(ParseConfidence.RAW),
                        },
                    },
                },
                "provenance_evidence": {
                    "occurrence_schema": OCCURRENCE_SCHEMA,
                    "training_sidecar_schema": TRAINING_SIDECAR_SCHEMA,
                    "validated_occurrence_count": validated_occurrence_count,
                    "parser_sidecar_occurrence_count": (parser_sidecar_sequence.count),
                    "parser_sidecar_occurrence_sequence_order": (
                        "token-sequence-sha256-content-sha256-occurrence-key"
                    ),
                    "parser_sidecar_occurrence_sequence_sha256": (
                        parser_sidecar_sequence.sha256
                    ),
                    "run_metadata_raw_sha256_recomputed_from_fetch_state": True,
                    "run_metadata_validation_scope": (
                        "every-done-or-empty-attempt-and-every-CAS-occurrence"
                    ),
                    "jobs_payload_sha256_recomputed_from_fetch_state": True,
                    "run_metadata_projection_cross_checks": [
                        "occurrence-key-run-id-attempt",
                        "exact-attempt-source-and-attempt",
                        "repository-scope",
                        "workflow-head-sha-and-head-commit-id",
                        "exact-job-selection-and-fields",
                    ],
                    "full_parser_sidecars_resolved": True,
                    "full_parser_language_classifications_resolved": True,
                    "full_parser_platform_classification_resolved": True,
                    "boundary": (
                        "the immutable fetch-state snapshot is strict-decompressed "
                        "and joined to every CAS occurrence; exact run metadata, "
                        "jobs payloads and member-to-job selection, member hashes, "
                        "chunk indexes, parser sidecars, and full classifications "
                        "are receipt-bound before eligibility"
                    ),
                },
                "eligibility": {
                    "policy": {
                        "schema": (
                            "cppmega_ci_primary_training_eligibility_policy_v1"
                        ),
                        "primary_route": PRIMARY_ROUTE,
                        "training_scope": scope_policy,
                        "exact_step_propagation": {
                            "schema": (
                                "cppmega_ci_exact_step_scope_propagation_v1"
                            ),
                            "key": [
                                "repo",
                                "run_attempt",
                                "job",
                                "step",
                            ],
                            "primary_priority": True,
                            "opaque_members_never_inherit": True,
                            "cross_step_propagation": False,
                            "cross_job_propagation": False,
                            "cross_attempt_propagation": False,
                        },
                        "opaque_artifact": {
                            "schema": OPAQUE_ARTIFACT_POLICY_SCHEMA,
                            "rule": (
                                "archive-member-casefold-suffix-.zip AND "
                                "decode-status-invalid_replaced AND "
                                "invalid-sequence-ratio-ge-threshold"
                            ),
                            "invalid_ratio_numerator": (
                                "invalid_sequence_count * 1000000"
                            ),
                            "invalid_ratio_denominator": "raw_byte_count",
                            "invalid_ratio_ppm_threshold": (
                                OPAQUE_INVALID_RATIO_PPM_THRESHOLD
                            ),
                            "raw_magic_retained": False,
                        },
                    },
                    "target_exact_unique_payload_tokens": eligible_target_tokens,
                    "target_source": eligible_target_source,
                    "cas_acquisition_target_exact_unique_payload_tokens": (
                        cas_acquisition_target_tokens
                    ),
                    "cas_reserve_exact_unique_payload_tokens": (
                        cas_acquisition_target_tokens - eligible_target_tokens
                    ),
                    "target_met": expected_payload >= eligible_target_tokens,
                    "cas": {
                        "unique_token_sequences": cas_unique_sequence_count,
                        "exact_unique_payload_tokens": cas_unique_payload_tokens,
                    },
                    "eligible": {
                        "unique_token_sequences": representative_count,
                        "exact_unique_payload_tokens": expected_payload,
                    },
                    "excluded_only": {
                        "unique_token_sequences": excluded_unique_sequence_count,
                        "exact_unique_payload_tokens": (excluded_unique_payload_tokens),
                    },
                    "excluded_occurrences": {
                        "members": excluded_member_count,
                        "occurrences": excluded_occurrence_count,
                        "summed_exact_tokens_with_occurrence_multiplicity": (
                            excluded_opaque_occurrence_payload_tokens
                            + excluded_training_scope_occurrence_payload_tokens
                        ),
                    },
                    "routing_accounting": {
                        "scope": "all-frozen-cas-occurrences",
                        "occurrence_counts": dict(
                            sorted(training_scope_occurrence_counts.items())
                        ),
                        "summed_exact_tokens_with_occurrence_multiplicity": (
                            dict(sorted(training_scope_token_counts.items()))
                        ),
                    },
                    "excluded_opaque_occurrences": {
                        "members": excluded_opaque_member_count,
                        "occurrences": excluded_opaque_writer.count,
                        "summed_exact_tokens_with_occurrence_multiplicity": (
                            excluded_opaque_occurrence_payload_tokens
                        ),
                        "ledger_schema": OPAQUE_ARTIFACT_LEDGER_SCHEMA,
                        "ledger": excluded_opaque_writer.path.relative_to(
                            temp_path
                        ).as_posix(),
                        "ledger_sha256": excluded_opaque_writer.logical_sha256,
                        "ledger_artifact_sha256": _sha256_file(
                            excluded_opaque_writer.path
                        ),
                    },
                    "excluded_training_scope_occurrences": {
                        "members": excluded_training_scope_member_count,
                        "occurrences": excluded_training_scope_writer.count,
                        "summed_exact_tokens_with_occurrence_multiplicity": (
                            excluded_training_scope_occurrence_payload_tokens
                        ),
                        "ledger_schema": (
                            TRAINING_SCOPE_EXCLUSION_LEDGER_SCHEMA
                        ),
                        "ledger": (
                            excluded_training_scope_writer.path.relative_to(
                                temp_path
                            ).as_posix()
                        ),
                        "ledger_sha256": (
                            excluded_training_scope_writer.logical_sha256
                        ),
                        "ledger_artifact_sha256": _sha256_file(
                            excluded_training_scope_writer.path
                        ),
                    },
                    "conservation": {
                        "exact_unique_payload_tokens": (
                            expected_payload + excluded_unique_payload_tokens
                            == cas_unique_payload_tokens
                        ),
                        "unique_token_sequences": (
                            representative_count + excluded_unique_sequence_count
                            == cas_unique_sequence_count
                        ),
                    },
                },
                "representatives": {
                    "schema": REPRESENTATIVE_LEDGER_SCHEMA,
                    "selection": (
                        "one-per-primary-eligible-token-sequence; "
                        "content-sha256-then-eligible-occurrence-key"
                    ),
                    "count": representative_count,
                    "ledger_artifact": ledger_path.relative_to(temp_path).as_posix(),
                    "ledger_sha256": ledger_digest,
                    "ledger_artifact_sha256": _sha256_file(ledger_path),
                },
                "representative_metadata": {
                    "schema": REPRESENTATIVE_METADATA_SCHEMA,
                    "count": representative_metadata_writer.count,
                    "sha256": representative_metadata_writer.logical_sha256,
                    "artifact": representative_metadata_writer.path.relative_to(
                        temp_path
                    ).as_posix(),
                    "artifact_sha256": _sha256_file(
                        representative_metadata_writer.path
                    ),
                    "projection": {
                        "identity": (
                            "actor-login-id-node-id-type-site-admin-no-email-or-url"
                        ),
                        "head_commit_message": "sha256-and-char-count-only",
                        "job_api": "allowlisted-fields-no-url-objects",
                        "runner_platform": (
                            "exact-job-fields-and-label-evidence-no-inference"
                        ),
                        "training_sidecars": (
                            "allowlisted-build-test-diagnostic-and-semantic-fields"
                        ),
                        "source_bindings": (
                            "receipt-bound-derived-projection-with-upstream-parser-"
                            "sidecars-and-hashes-unchanged"
                        ),
                        "derived_classifications": (
                            "explicit-typed-full-parser-plus-retained-chunk-evidence"
                        ),
                        "raw_command_or_diagnostic_message": "omitted",
                    },
                },
                "counts": dict(sorted(count_totals.items())),
                "graph_accounting": {
                    **dict(sorted(graph_totals.items())),
                    "cross_chunk_source_accounting_semantics": {
                        "count": (
                            "source-reported and provenance-bound; only outbound "
                            "records are present in training sidecar v2"
                        ),
                        "outbound_count": (
                            "independently matched to retained outbound records"
                        ),
                        "sha256": "independently recomputed for every accepted record",
                        "fail_closed_boundary": (
                            "count must equal outbound_count because v2 omits "
                            "non-outbound edge identities"
                        ),
                    },
                    "by_family": dict(sorted(graph_by_family.items())),
                    "dropped_edge_records_sha256": (dropped_edge_writer.logical_sha256),
                    "dropped_edge_record_count": dropped_edge_writer.count,
                    "dropped_edge_artifact": (
                        dropped_edge_writer.path.relative_to(temp_path).as_posix()
                    ),
                    "dropped_edge_artifact_sha256": _sha256_file(
                        dropped_edge_writer.path
                    ),
                },
                "fragment_ledger": {
                    "count": fragment_writer.count,
                    "sha256": fragment_writer.logical_sha256,
                    "artifact": fragment_writer.path.relative_to(temp_path).as_posix(),
                    "artifact_sha256": _sha256_file(fragment_writer.path),
                },
                "artifacts": artifact_records,
                "validation": {
                    "case5_audit": audits,
                    "all_passed": True,
                    "fixed_widths": True,
                    "zero_overflow": True,
                    "payload_conserved": True,
                    "payload_identity_and_order_verified": True,
                    "all_case5_parquet_zstd": True,
                    "post_normalize_pack_sidecars_and_edges_verified": True,
                },
            }
            if production_provenance is not None:
                receipt["completion_mode"] = completion_mode
                receipt["production_complete"] = True
                receipt["acquisition_provenance"] = production_provenance
            receipt_path = temp_path / "export_receipt.json"
            _write_json(receipt_path, receipt)
            _fsync_tree(temp_path)
            store.require_unchanged()
            frozen_fetch_state.require_unchanged()
            if _script_sha256() != exporter_script_sha256:
                raise ExportError("exporter script changed while export was running")
            if (
                projection_script_sha256()
                != source_binding_projection_script_sha256
            ):
                raise ExportError(
                    "source-binding projection script changed while export was running"
                )
            if (
                target_parser_script_sha256()
                != source_binding_projector.target_parser_sha256
            ):
                raise ExportError(
                    "target parser script changed while export was running"
                )
            if {
                "byte_size": resolved_tokenizer.stat().st_size,
                "sha256": _sha256_file(resolved_tokenizer),
            } != tokenizer_snapshot:
                raise ExportError(
                    "tokenizer changed while export was running"
                )
            if production_provenance is not None:
                assert production_paths is not None
                final_provenance = _verify_exhaustive_export_provenance(
                    store_root=resolved_store_root,
                    store_receipt_path=resolved_store_receipt,
                    fetch_state_path=resolved_fetch_state,
                    inventory_path=production_paths[0],
                    inventory_receipt_path=production_paths[1],
                    fetch_receipt_path=production_paths[2],
                    merge_receipt_path=production_paths[3],
                )
                if final_provenance != production_provenance:
                    raise ExportError(
                        "production control artifacts changed during export"
                    )
            _publish_directory_no_replace(temp_path, output_path)
            parent_descriptor = os.open(output_path.parent, os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
            published = True
            return receipt
    except SourceBindingProjectionError as exc:
        raise ExportError(f"source-binding projection refused: {exc}") from exc
    except CITrainingScopeError as exc:
        raise ExportError(f"CI training scope refused: {exc}") from exc
    finally:
        if fragment_writer is not None:
            fragment_writer.close()
        if dropped_edge_writer is not None:
            dropped_edge_writer.close()
        if representative_metadata_writer is not None:
            representative_metadata_writer.close()
        if representative_ledger_writer is not None:
            representative_ledger_writer.close()
        if excluded_opaque_writer is not None:
            excluded_opaque_writer.close()
        if excluded_training_scope_writer is not None:
            excluded_training_scope_writer.close()
        if source_binding_projection_writer is not None:
            source_binding_projection_writer.close()
        if occurrence_metadata_writer is not None:
            occurrence_metadata_writer.close()
        if eligibility_connection is not None:
            eligibility_connection.close()
        for writer in parquet_writers.values():
            writer.close()
        parquet_writers.clear()
        if not published and temp_path.exists():
            shutil.rmtree(temp_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--store-receipt", required=True, type=Path)
    parser.add_argument("--fetch-state", required=True, type=Path)
    parser.add_argument(
        "--completion-mode",
        choices=(
            COMPLETION_MODE_THRESHOLD,
            COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        ),
        default=COMPLETION_MODE_THRESHOLD,
    )
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-receipt", type=Path)
    parser.add_argument("--fetch-receipt", type=Path)
    parser.add_argument("--merge-receipt", type=Path)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-binding-projection-from-parser-sha256",
        help=(
            "explicitly authorize the exact legacy parser SHA-256 whose "
            "source bindings will be projected; required for legacy stores"
        ),
    )
    parser.add_argument(
        "--require-current-parser-only",
        action="store_true",
        help=(
            "refuse stores with any parser upgrade lineage; require every "
            "occurrence to come from the exact parser implementation used by "
            "this exporter"
        ),
    )
    parser.add_argument(
        "--required-eligible-exact-unique-payload-tokens",
        type=int,
        help=(
            "explicit training-eligible token minimum after opaque-artifact "
            "exclusion; must not exceed the receipt-bound CAS acquisition target"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = export_store(
            store_root=args.store,
            store_receipt=args.store_receipt,
            fetch_state=args.fetch_state,
            tokenizer_json=args.tokenizer_json,
            output=args.output,
            source_binding_projection_from_parser_sha256=(
                args.source_binding_projection_from_parser_sha256
            ),
            require_current_parser_only=args.require_current_parser_only,
            required_eligible_exact_unique_payload_tokens=(
                args.required_eligible_exact_unique_payload_tokens
            ),
            completion_mode=args.completion_mode,
            inventory=args.inventory,
            inventory_receipt=args.inventory_receipt,
            fetch_receipt=args.fetch_receipt,
            merge_receipt=args.merge_receipt,
        )
    except (
        ExportError,
        SourceBindingProjectionError,
        OSError,
        sqlite3.Error,
        ValueError,
    ) as exc:
        raise SystemExit(f"CI CASE5 export refused: {exc}") from exc
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": receipt["status"],
                "representatives": receipt["representatives"]["count"],
                "fragments": receipt["counts"]["fragments"],
                "valid_tokens": receipt["counts"]["valid_tokens"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
