#!/usr/bin/env python3
"""Stream GitHub Actions logs into the exact-deduplicated CI content store.

The inventory stage and this fetch stage deliberately use separate SQLite
databases.  The inventory may continue adding immutable run identities while
this process consumes the oldest visible runs.  A content-store commit happens
before an attempt is marked complete, so replay after a crash is idempotent.

Only canonical, secret-redacted payload chunks enter the content store.  Raw
ZIP archives are bounded temporary inputs.  A separately created rescue spool
can be imported through the same validation/parser/tokenizer path.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import inspect
import io
import json
import math
import multiprocessing
import os
import re
import sqlite3
import stat
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega.data.tokenizer_contract import (  # noqa: E402,RUF100
    TOKENIZER_CONTRACT_SHA256,
)
from cppmega.tokenizer.cpp_tokenizer import (  # noqa: E402,RUF100
    CppMegaTokenizer,
    TokenizerContractError,
    load_cppmega_tokenizer,
)
from scripts.ci_content_store import (  # noqa: E402,RUF100
    PRODUCTION_TARGET_UNIQUE_TOKENS,
    CIContentStore,
    hash_token_sequence,
)
from scripts.ci_log_sidecars import canonicalize_ci_log  # noqa: E402,RUF100
from scripts.ci_stream_inventory import (  # noqa: E402,RUF100
    GITHUB_API_VERSION,
    HTTPResponse,
    TokenPool,
    load_token_pool,
    verify_inventory_completion_receipt,
)
from scripts.ci_zlib_evidence import (  # noqa: E402,RUF100
    MAX_JOBS_EVIDENCE_BYTES,
    MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
    MAX_RUN_METADATA_BYTES,
    MAX_RUN_METADATA_COMPRESSED_BYTES,
    MAX_STATE_JSON_EVIDENCE_BYTES,
    ZlibEvidenceError,
    constrain_sqlite_evidence_rows,
    fetch_state_attempt_evidence_bound_violation,
    fetch_state_evidence_bound_violation,
    strict_bounded_zlib_decode,
)

SCHEMA_VERSION = "cppmega_ci_stream_fetch_v4"
PROGRESS_SCHEMA = "cppmega_ci_stream_fetch_progress_v4"
RECEIPT_SCHEMA = "cppmega_ci_stream_fetch_receipt_v3"
EXHAUSTIVE_RECEIPT_SCHEMA = "cppmega_ci_stream_fetch_receipt_v4"
EXHAUSTIVE_DISCOVERY_SCHEMA = "cppmega_ci_stream_fetch_discovery_v1"
COMPLETION_MODE_THRESHOLD = "threshold"
COMPLETION_MODE_INVENTORY_EXHAUSTIVE = "inventory-exhaustive"
DEFAULT_TOKENIZER = str(
    _REPO_ROOT / "data" / "tokenizer_v2" / "tokenizer.json"
)
DEFAULT_TARGET = PRODUCTION_TARGET_UNIQUE_TOKENS
DEFAULT_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES = 1024 * 1024
RESCUE_ARCHIVE_PROVENANCE_SCHEMA = (
    "cppmega_ci_rescue_archive_provenance_v1"
)
JOB_RESCUE_RECEIPT_SCHEMA = "cppmega_ci_job_log_rescue_receipt_v1"
JOB_RESCUE_RESOLVED_JOBS_SCHEMA = (
    "cppmega_ci_job_log_rescue_resolved_jobs_v1"
)
PRODUCER_LINEAGE_SCHEMA = "cppmega_ci_operator_producer_lineage_v1"
JOB_RESCUE_LEDGER_EVIDENCE_LEGACY_SCHEMA = (
    "cppmega_ci_job_rescue_ledger_evidence_v2"
)
JOB_RESCUE_LEDGER_EVIDENCE_SCHEMA = (
    "cppmega_ci_job_rescue_ledger_evidence_v3"
)
JOB_RESCUE_LEGACY_SEMANTIC_CONTRACT = (
    "cppmega-ci-job-rescue-v1:"
    "legacy-receipt-without-producer-binding;"
    "receipt-and-source-row-digests-in-operator-ledger"
)
JOB_RESCUE_LEGACY_SEMANTIC_CONTRACT_SHA256 = hashlib.sha256(
    JOB_RESCUE_LEGACY_SEMANTIC_CONTRACT.encode("utf-8")
).hexdigest()
JOB_RESCUE_SEMANTIC_CONTRACT = (
    "cppmega-ci-job-rescue-v3:"
    "exact-source-row-and-jobs-ledger;"
    "every-job-is-log-or-terminal-404-or-terminal-410;"
    "synthetic-zip-members-equal-full-logs;"
    "canonical-receipt-and-operator-ledger;"
    "append-only-producer-lineage-with-explicit-upgrades;"
    "empty-zip-only-when-all-jobs-are-terminal"
)
JOB_RESCUE_SEMANTIC_CONTRACT_SHA256 = hashlib.sha256(
    JOB_RESCUE_SEMANTIC_CONTRACT.encode("utf-8")
).hexdigest()
PRESERVED_RECOVERY_RECEIPT_SCHEMA = (
    "cppmega_ci_preserved_archive_recovery_v1"
)
PRESERVED_RECOVERY_LEDGER_LEGACY_SCHEMA = (
    "cppmega_ci_preserved_archive_recovery_ledger_v1"
)
PRESERVED_RECOVERY_LEDGER_SCHEMA = (
    "cppmega_ci_preserved_archive_recovery_ledger_v2"
)
PRESERVED_ARCHIVE_PROVENANCE_SCHEMA = (
    "cppmega_ci_preserved_archive_provenance_v1"
)
PRESERVED_RECOVERY_LEGACY_SEMANTIC_CONTRACT = (
    "cppmega-ci-preserved-archive-recovery-legacy-v1:"
    "receipt-proof-binds-original-recovery-script;"
    "operator-ledger-binds-recovery-receipt-and-source-row"
)
PRESERVED_RECOVERY_LEGACY_SEMANTIC_CONTRACT_SHA256 = hashlib.sha256(
    PRESERVED_RECOVERY_LEGACY_SEMANTIC_CONTRACT.encode("utf-8")
).hexdigest()
PRESERVED_RECOVERY_SEMANTIC_CONTRACT = (
    "cppmega-ci-preserved-archive-recovery-v2:"
    "exact-source-row-and-durable-member-witness-set;"
    "complete-bounded-zip-crc-replay;"
    "all-existing-members-match-name-size-sha256;"
    "append-only-producer-lineage-with-explicit-upgrades;"
    "canonical-receipt-and-operator-ledger"
)
PRESERVED_RECOVERY_SEMANTIC_CONTRACT_SHA256 = hashlib.sha256(
    PRESERVED_RECOVERY_SEMANTIC_CONTRACT.encode("utf-8")
).hexdigest()
_UNSPECIFIED_PRODUCER_BINDING = object()
DEFAULT_MAX_MEMBER_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_MEMBERS = 20_000
DEFAULT_MAX_CHUNK_CHARS = 128_000
DEFAULT_DISCOVERY_ROWS = 20_000
DEFAULT_API_ATTEMPTS = 12
DEFAULT_ARCHIVE_TRANSFER_ATTEMPTS = 16
DEFAULT_TIMEOUT = 90.0

_RUN_ATTEMPT_STATES = {
    "pending",
    "processing",
    "retry",
    "done",
    "empty",
    "terminal_404",
    "terminal_410",
    "failed",
}
_TERMINAL_STATES = {
    "done",
    "empty",
    "terminal_404",
    "terminal_410",
    "failed",
}
_RUN_METADATA_SOURCES = {
    "inventory-run-list",
    "github-workflow-run-attempt-api",
}
_MAIN_MEMBER_RE = re.compile(r"^(?P<ordinal>\d+)_(?P<name>.+)\.txt$")
_SECRET_QUERY_KEYS = {
    "sig",
    "signature",
    "token",
    "se",
    "sp",
    "sv",
    "srt",
    "spr",
}


class FetchError(RuntimeError):
    """Base fail-closed fetch error."""


class BindingError(FetchError):
    """Durable state does not match the current producer contract."""


class APIError(FetchError):
    """A GitHub API operation exhausted its safe retry policy."""


class MalformedResponseError(APIError):
    """A response cannot prove the expected endpoint contract."""


class ArchiveError(FetchError):
    """A workflow log archive violates a safety or conservation rule."""


class TerminalHTTP(FetchError):
    """An immutable endpoint result proves that no archive can be fetched."""

    def __init__(
        self,
        status: int,
        body: bytes,
        endpoint: str,
        *,
        jobs: Sequence[Mapping[str, object]] | None = None,
    ):
        super().__init__(f"GitHub HTTP {status} for {endpoint}")
        self.status = status
        self.body = body
        self.endpoint = endpoint
        self.jobs = None if jobs is None else [dict(job) for job in jobs]


@dataclass(frozen=True)
class Attempt:
    repo: str
    run_id: int
    attempt: int
    created_at: str
    run_metadata: dict[str, Any]
    run_metadata_sha256: str
    run_metadata_source: str
    run_metadata_source_attempt: int
    run_metadata_exact: bool
    inventory_seed_attempt: int
    inventory_seed_metadata_sha256: str

    @property
    def run_attempt_key(self) -> str:
        return f"{self.run_id}:{self.attempt}"


@dataclass(frozen=True)
class ExhaustiveInventoryBinding:
    receipt_path: Path
    receipt_sha256: str
    database_sha256: str
    db_logical_sha256: str
    expected_run_count: int
    expected_attempt_count: int
    expected_attempt_set_sha256: str


@dataclass(frozen=True)
class ArchiveSource:
    path: Path
    source: str
    raw_sha256: str
    raw_size: int
    recoverable: bool
    provenance: Mapping[str, object] | None = None


@dataclass(frozen=True)
class PreparedArchive:
    repository: str
    run_id: int
    attempt: int
    source: str
    inline_body: bytes | None
    signed_url: str | None


@dataclass(frozen=True)
class RequestResult:
    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class RepositoryIdentity:
    requested: str
    canonical: str
    repository_id: int | None
    source: str
    source_repository_id: int | None


_REPOSITORY_FULL_NAME_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/"
    r"[A-Za-z0-9_.-]+"
)


def _repository_object_identity(
    value: object,
    *,
    field: str,
) -> tuple[str, int | None] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise MalformedResponseError(
            f"run metadata {field} is not an object"
        )
    full_name = value.get("full_name")
    if (
        not isinstance(full_name, str)
        or _REPOSITORY_FULL_NAME_RE.fullmatch(full_name) is None
    ):
        raise MalformedResponseError(
            f"run metadata {field}.full_name is invalid"
        )
    raw_id = value.get("id")
    if raw_id is None:
        repository_id = None
    elif (
        isinstance(raw_id, bool)
        or not isinstance(raw_id, int)
        or raw_id <= 0
    ):
        raise MalformedResponseError(
            f"run metadata {field}.id is invalid"
        )
    else:
        repository_id = raw_id
    return full_name, repository_id


def _repository_identity(attempt: Attempt) -> RepositoryIdentity:
    canonical = _repository_object_identity(
        attempt.run_metadata.get("repository"),
        field="repository",
    )
    source = _repository_object_identity(
        attempt.run_metadata.get("head_repository"),
        field="head_repository",
    )
    canonical_name, repository_id = (
        (attempt.repo, None) if canonical is None else canonical
    )
    source_name, source_repository_id = (
        (canonical_name, repository_id) if source is None else source
    )
    return RepositoryIdentity(
        requested=attempt.repo,
        canonical=canonical_name,
        repository_id=repository_id,
        source=source_name,
        source_repository_id=source_repository_id,
    )


def _validate_rescue_archive_provenance(
    value: object,
    *,
    repo: str,
    canonical_repo: str,
    run_id: int,
    attempt: int,
    created_at: str,
    run_metadata_sha256: str,
    run_metadata_raw_size: int,
    archive_sha256: str,
    archive_size: int,
    jobs_sha256: str,
    jobs_raw_size: int,
    job_count: int,
    member_count: int,
    member_uncompressed_bytes: int,
    jobs: Sequence[Mapping[str, object]] | None = None,
    durable_members: Mapping[str, tuple[str, int]] | None = None,
) -> tuple[str, str]:
    """Validate persisted job-rescue provenance against one completed row."""

    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema",
            "manifest",
            "archive",
            "job_rescue_receipt",
            "resolved_jobs",
        }
        or value.get("schema") != RESCUE_ARCHIVE_PROVENANCE_SCHEMA
    ):
        raise BindingError("rescue archive provenance schema is invalid")
    manifest = value.get("manifest")
    archive = value.get("archive")
    receipt_evidence = value.get("job_rescue_receipt")
    resolved_evidence = value.get("resolved_jobs")
    if not all(
        isinstance(item, Mapping)
        for item in (
            manifest,
            archive,
            receipt_evidence,
            resolved_evidence,
        )
    ):
        raise BindingError("rescue archive provenance shape is invalid")
    assert isinstance(manifest, Mapping)
    assert isinstance(archive, Mapping)
    assert isinstance(receipt_evidence, Mapping)
    assert isinstance(resolved_evidence, Mapping)
    if (
        set(manifest)
        != {
            "manifest_file_sha256",
            "record",
            "record_sha256",
        }
        or set(archive) != {"source", "name", "bytes", "sha256"}
        or set(receipt_evidence)
        != {"name", "bytes", "sha256", "receipt"}
        or set(resolved_evidence)
        != {"name", "bytes", "sha256", "records"}
    ):
        raise BindingError("rescue archive provenance has extra/missing fields")
    manifest_record = manifest.get("record")
    receipt = receipt_evidence.get("receipt")
    if not isinstance(manifest_record, Mapping) or not isinstance(
        receipt,
        Mapping,
    ):
        raise BindingError("rescue manifest/receipt evidence is missing")
    if set(manifest_record) != {
        "repo",
        "run_id",
        "attempt",
        "created_at",
        "status",
        "bytes",
        "sha256",
        "finished_at",
    }:
        raise BindingError("rescue manifest record shape is invalid")
    manifest_record_bytes = _canonical_json_bytes(manifest_record)
    finished_at = manifest_record.get("finished_at")
    if (
        manifest.get("record_sha256")
        != _sha256_bytes(manifest_record_bytes)
        or not isinstance(manifest.get("manifest_file_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(manifest.get("manifest_file_sha256")),
        )
        is None
        or dict(manifest_record)
        != {
            "repo": repo,
            "run_id": str(run_id),
            "attempt": str(attempt),
            "created_at": created_at,
            "status": "zip",
            "bytes": str(archive_size),
            "sha256": archive_sha256,
            "finished_at": finished_at,
        }
        or not _is_canonical_utc_timestamp(created_at)
        or not _is_canonical_utc_timestamp(finished_at)
    ):
        raise BindingError("rescue manifest record binding is invalid")
    if dict(archive) != {
        "source": "rescue-spool",
        "name": f"{repo.replace('/', '__')}--{run_id}--attempt-{attempt}.zip",
        "bytes": archive_size,
        "sha256": archive_sha256,
    }:
        raise BindingError("rescue archive identity binding is invalid")

    receipt_fields = set(receipt)
    current_receipt_fields = {
        "schema",
        "completed_at",
        "producer_binding",
        "source_state",
        "coverage",
        "artifacts",
    }
    legacy_receipt_fields = current_receipt_fields - {"producer_binding"}
    if receipt_fields not in (
        current_receipt_fields,
        legacy_receipt_fields,
    ):
        raise BindingError("job-rescue receipt shape is invalid")
    _job_rescue_receipt_producer_binding(receipt)
    receipt_raw = _canonical_json_bytes(receipt) + b"\n"
    receipt_sha256 = _sha256_bytes(receipt_raw)
    if (
        receipt_evidence.get("name")
        != f"{repo.replace('/', '__')}--{run_id}--attempt-{attempt}.receipt.json"
        or receipt_evidence.get("bytes") != len(receipt_raw)
        or receipt_evidence.get("sha256") != receipt_sha256
        or receipt.get("schema") != JOB_RESCUE_RECEIPT_SCHEMA
        or receipt.get("completed_at") != finished_at
        or not _is_canonical_utc_timestamp(receipt.get("completed_at"))
    ):
        raise BindingError("job-rescue receipt artifact binding is invalid")
    source_state = receipt.get("source_state")
    coverage = receipt.get("coverage")
    artifacts = receipt.get("artifacts")
    if not all(
        isinstance(item, Mapping)
        for item in (source_state, coverage, artifacts)
    ):
        raise BindingError("job-rescue receipt shape is invalid")
    assert isinstance(source_state, Mapping)
    assert isinstance(coverage, Mapping)
    assert isinstance(artifacts, Mapping)
    if (
        set(source_state)
        != {
            "path",
            "repo",
            "canonical_repo",
            "run_id",
            "attempt",
            "created_at",
            "status",
            "tries",
            "error_class",
            "failed_raw_archive",
            "attempt_row_sha256",
            "run_metadata_sha256",
            "run_metadata_raw_size",
            "jobs_sha256",
            "jobs_raw_size",
            "jobs_ledger_sha256",
            "jobs_ledger_ids",
        }
        or set(coverage)
        != {
            "expected_jobs",
            "resolved_jobs",
            "unresolved_jobs",
            "full_logs",
            "terminal_404",
            "terminal_410",
            "zip_members",
            "uncompressed_log_bytes",
        }
        or set(artifacts) != {"resolved_jobs", "synthetic_zip"}
    ):
        raise BindingError("job-rescue receipt has extra/missing fields")
    failed_raw_archive = source_state.get("failed_raw_archive")
    ledger_ids = source_state.get("jobs_ledger_ids")
    error_class = source_state.get("error_class")
    source_row_sha256 = source_state.get("attempt_row_sha256")
    if (
        not isinstance(source_state.get("path"), str)
        or not str(source_state.get("path"))
        or "\x00" in str(source_state.get("path"))
        or source_state.get("repo") != repo
        or source_state.get("canonical_repo") != canonical_repo
        or source_state.get("run_id") != run_id
        or source_state.get("attempt") != attempt
        or source_state.get("created_at") != created_at
        or source_state.get("status") != "failed"
        or isinstance(source_state.get("tries"), bool)
        or not isinstance(source_state.get("tries"), int)
        or int(source_state.get("tries")) < 0
        or (
            error_class is not None
            and (
                not isinstance(error_class, str)
                or not error_class
                or any(char in error_class for char in ("\x00", "\r", "\n"))
            )
        )
        or source_state.get("run_metadata_sha256")
        != run_metadata_sha256
        or source_state.get("run_metadata_raw_size")
        != run_metadata_raw_size
        or source_state.get("jobs_sha256") != jobs_sha256
        or source_state.get("jobs_raw_size") != jobs_raw_size
        or not isinstance(source_row_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_row_sha256) is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(source_state.get("jobs_ledger_sha256")),
        )
        is None
        or not isinstance(ledger_ids, list)
        or len(ledger_ids) != max(1, math.ceil(job_count / 100))
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in ledger_ids
        )
        or ledger_ids != sorted(set(ledger_ids))
    ):
        raise BindingError("job-rescue source-state binding is invalid")
    if (
        not isinstance(failed_raw_archive, Mapping)
        or set(failed_raw_archive)
        != {"source", "sha256", "bytes", "preservation"}
        or failed_raw_archive.get("preservation")
        != "source fetcher artifact is not modified"
    ):
        raise BindingError("job-rescue failed-archive binding is invalid")
    failed_values = (
        failed_raw_archive.get("source"),
        failed_raw_archive.get("sha256"),
        failed_raw_archive.get("bytes"),
    )
    if not (
        all(item is None for item in failed_values)
        or (
            isinstance(failed_values[0], str)
            and bool(failed_values[0])
            and re.fullmatch(r"[0-9a-f]{64}", str(failed_values[1]))
            is not None
            and not isinstance(failed_values[2], bool)
            and isinstance(failed_values[2], int)
            and failed_values[2] >= 0
        )
    ):
        raise BindingError("job-rescue failed-archive evidence is inconsistent")
    full_logs = coverage.get("full_logs")
    terminal_404 = coverage.get("terminal_404")
    terminal_410 = coverage.get("terminal_410")
    count_values = (
        coverage.get("expected_jobs"),
        coverage.get("resolved_jobs"),
        coverage.get("unresolved_jobs"),
        full_logs,
        terminal_404,
        terminal_410,
        coverage.get("zip_members"),
        coverage.get("uncompressed_log_bytes"),
    )
    if (
        any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in count_values
        )
        or coverage.get("expected_jobs") != job_count
        or coverage.get("resolved_jobs") != job_count
        or coverage.get("unresolved_jobs") != 0
        or full_logs + terminal_404 + terminal_410 != job_count
        or full_logs != member_count
        or coverage.get("zip_members") != full_logs
        or coverage.get("uncompressed_log_bytes")
        != member_uncompressed_bytes
    ):
        raise BindingError("job-rescue coverage binding is invalid")
    synthetic = artifacts.get("synthetic_zip")
    resolved = artifacts.get("resolved_jobs")
    resolved_records = resolved_evidence.get("records")
    if (
        not isinstance(synthetic, Mapping)
        or set(synthetic) != {"name", "bytes", "sha256"}
        or dict(synthetic)
        != {
            "name": "synthetic.zip",
            "bytes": archive_size,
            "sha256": archive_sha256,
        }
        or not isinstance(resolved, Mapping)
        or set(resolved) != {"name", "bytes", "sha256"}
        or dict(resolved)
        != {
            "name": "resolved_jobs.jsonl",
            "bytes": resolved.get("bytes"),
            "sha256": resolved.get("sha256"),
        }
        or resolved_evidence.get("name")
        != f"{repo.replace('/', '__')}--{run_id}--attempt-{attempt}.resolved_jobs.jsonl"
        or resolved_evidence.get("bytes") != resolved.get("bytes")
        or resolved_evidence.get("sha256") != resolved.get("sha256")
        or not isinstance(resolved.get("bytes"), int)
        or isinstance(resolved.get("bytes"), bool)
        or int(resolved.get("bytes")) < 0
        or not isinstance(resolved.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(resolved.get("sha256")))
        is None
        or not isinstance(resolved_records, list)
        or len(resolved_records) != job_count
    ):
        raise BindingError("job-rescue artifact coverage binding is invalid")
    try:
        resolved_raw = b"".join(
            _canonical_json_bytes(record) + b"\n"
            for record in resolved_records
        )
    except (TypeError, ValueError) as exc:
        raise BindingError(
            "job-rescue resolved-jobs replay is not canonical JSONL"
        ) from exc
    if (
        resolved_evidence.get("bytes") != len(resolved_raw)
        or resolved_evidence.get("sha256")
        != _sha256_bytes(resolved_raw)
    ):
        raise BindingError(
            "job-rescue resolved-jobs artifact differs from embedded records"
        )
    if durable_members is not None and len(durable_members) != member_count:
        raise BindingError(
            "job-rescue durable-member cardinality differs from receipt"
        )
    replay_counts = {
        "log": 0,
        "terminal_404": 0,
        "terminal_410": 0,
    }
    replay_log_bytes = 0
    seen_job_ids: set[int] = set()
    seen_durable_members: set[str] = set()
    for ordinal, record in enumerate(resolved_records):
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "schema",
                "source_row_sha256",
                "jobs_sha256",
                "ordinal",
                "job_id",
                "job_name",
                "endpoint",
                "member_name",
                "outcome",
                "api_http_status",
                "signed_http_status",
                "log",
                "terminal",
            }
            or record.get("schema")
            != JOB_RESCUE_RESOLVED_JOBS_SCHEMA
            or record.get("source_row_sha256") != source_row_sha256
            or record.get("jobs_sha256") != jobs_sha256
            or record.get("ordinal") != ordinal
            or isinstance(record.get("job_id"), bool)
            or not isinstance(record.get("job_id"), int)
            or int(record["job_id"]) <= 0
            or int(record["job_id"]) in seen_job_ids
            or not isinstance(record.get("job_name"), str)
            or not record.get("job_name")
        ):
            raise BindingError(
                "job-rescue resolved-job identity binding is invalid"
            )
        job_id = int(record["job_id"])
        seen_job_ids.add(job_id)
        expected_endpoint = (
            f"/repos/{canonical_repo}/actions/jobs/{job_id}/logs"
        )
        if record.get("endpoint") != expected_endpoint:
            raise BindingError(
                "job-rescue resolved-job endpoint binding is invalid"
            )
        if jobs is not None:
            current_job = jobs[ordinal]
            if (
                current_job.get("id") != job_id
                or current_job.get("name") != record.get("job_name")
                or current_job.get("status") != "completed"
                or current_job.get("run_id", run_id) != run_id
                or current_job.get("run_attempt", attempt) != attempt
            ):
                raise BindingError(
                    "job-rescue resolved job differs from current jobs "
                    "evidence"
                )
        outcome = record.get("outcome")
        if outcome not in replay_counts:
            raise BindingError("job-rescue resolved-job outcome is invalid")
        replay_counts[str(outcome)] += 1
        api_status = record.get("api_http_status")
        signed_status = record.get("signed_http_status")
        log = record.get("log")
        terminal = record.get("terminal")
        if outcome == "log":
            expected_member = f"{ordinal}_{job_id}.txt"
            if (
                record.get("member_name") != expected_member
                or (
                    (api_status, signed_status)
                    not in {(200, None), (302, 200)}
                )
                or not isinstance(log, Mapping)
                or set(log) != {"path", "bytes", "sha256"}
                or log.get("path")
                != f"logs/{ordinal:06d}--{job_id}.log"
                or isinstance(log.get("bytes"), bool)
                or not isinstance(log.get("bytes"), int)
                or int(log["bytes"]) < 0
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(log.get("sha256")),
                )
                is None
                or terminal is not None
            ):
                raise BindingError(
                    "job-rescue resolved log evidence is invalid"
                )
            if durable_members is not None:
                durable = durable_members.get(expected_member)
                if durable != (
                    str(log["sha256"]),
                    int(log["bytes"]),
                ):
                    raise BindingError(
                        "job-rescue resolved log differs from current "
                        "durable member"
                    )
                seen_durable_members.add(expected_member)
            replay_log_bytes += int(log["bytes"])
        else:
            terminal_status = 404 if outcome == "terminal_404" else 410
            if (
                record.get("member_name") is not None
                or log is not None
                or (
                    (api_status, signed_status)
                    not in {
                        (terminal_status, None),
                        (302, terminal_status),
                    }
                )
                or not isinstance(terminal, Mapping)
                or set(terminal)
                != {
                    "http_status",
                    "body_prefix_bytes",
                    "body_prefix_sha256",
                    "body_truncated",
                }
                or terminal.get("http_status") != terminal_status
                or isinstance(terminal.get("body_prefix_bytes"), bool)
                or not isinstance(
                    terminal.get("body_prefix_bytes"),
                    int,
                )
                or not 0 <= int(terminal["body_prefix_bytes"]) <= 64 * 1024
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(terminal.get("body_prefix_sha256")),
                )
                is None
                or not isinstance(terminal.get("body_truncated"), bool)
            ):
                raise BindingError(
                    "job-rescue terminal-job evidence is invalid"
                )
    if (
        replay_counts["log"] != coverage.get("full_logs")
        or replay_counts["terminal_404"]
        != coverage.get("terminal_404")
        or replay_counts["terminal_410"]
        != coverage.get("terminal_410")
        or replay_log_bytes != coverage.get("uncompressed_log_bytes")
        or (
            durable_members is not None
            and seen_durable_members != set(durable_members)
        )
    ):
        raise BindingError(
            "job-rescue resolved-job conservation differs from receipt"
        )
    return receipt_sha256, source_row_sha256


def _is_canonical_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        value,
    ) is None:
        return False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=UTC
        )
    except ValueError:
        return False
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value


def _validated_producer_binding(
    value: object,
    *,
    source: str,
) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {"script_sha256", "semantic_contract_sha256"}
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(value.get("script_sha256")),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(value.get("semantic_contract_sha256")),
        )
        is None
    ):
        raise BindingError(f"{source} producer binding is invalid")
    return {
        "script_sha256": str(value["script_sha256"]),
        "semantic_contract_sha256": str(
            value["semantic_contract_sha256"]
        ),
    }


def _job_rescue_receipt_producer_binding(
    value: Mapping[str, object],
) -> dict[str, str] | None:
    producer_binding = value.get("producer_binding")
    if producer_binding is None:
        return None
    return _validated_producer_binding(
        producer_binding,
        source="job-rescue receipt",
    )


def _preserved_recovery_receipt_producer_binding(
    value: Mapping[str, object],
) -> dict[str, str]:
    proof = value.get("proof")
    proof_script_sha256 = (
        proof.get("recovery_script_sha256")
        if isinstance(proof, Mapping)
        else None
    )
    if re.fullmatch(r"[0-9a-f]{64}", str(proof_script_sha256)) is None:
        raise BindingError("preserved-recovery proof producer is invalid")
    producer_binding = value.get("producer_binding")
    if producer_binding is None:
        return {
            "script_sha256": str(proof_script_sha256),
            "semantic_contract_sha256": (
                PRESERVED_RECOVERY_LEGACY_SEMANTIC_CONTRACT_SHA256
            ),
        }
    normalized = _validated_producer_binding(
        producer_binding,
        source="preserved-recovery receipt",
    )
    if normalized["script_sha256"] != proof_script_sha256:
        raise BindingError(
            "preserved-recovery proof/receipt producer differs"
        )
    return normalized


def _parsed_producer_lineage(
    value: object,
) -> tuple[
    dict[str, str],
    dict[str, str],
    list[dict[str, object]],
]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "origin", "current", "upgrades"}
        or value.get("schema") != PRODUCER_LINEAGE_SCHEMA
        or not isinstance(value.get("upgrades"), list)
    ):
        raise BindingError("producer lineage shape/schema is invalid")
    origin = _validated_producer_binding(
        value.get("origin"),
        source="producer lineage origin",
    )
    current = _validated_producer_binding(
        value.get("current"),
        source="producer lineage current",
    )
    cursor = origin
    normalized: list[dict[str, object]] = []
    for raw_upgrade in value["upgrades"]:
        if (
            not isinstance(raw_upgrade, Mapping)
            or set(raw_upgrade)
            != {"from", "to", "reason", "authorized_at"}
        ):
            raise BindingError("producer lineage upgrade shape is invalid")
        from_binding = _validated_producer_binding(
            raw_upgrade.get("from"),
            source="producer lineage upgrade source",
        )
        to_binding = _validated_producer_binding(
            raw_upgrade.get("to"),
            source="producer lineage upgrade destination",
        )
        reason = raw_upgrade.get("reason")
        authorized_at = raw_upgrade.get("authorized_at")
        if (
            from_binding != cursor
            or to_binding == from_binding
            or not isinstance(reason, str)
            or not 1 <= len(reason) <= 200
            or any(
                ord(char) < 32 or ord(char) == 127
                for char in reason
            )
            or not _is_canonical_utc_timestamp(authorized_at)
        ):
            raise BindingError(
                "producer lineage upgrade order/authorization is invalid"
            )
        normalized.append(
            {
                "from": from_binding,
                "to": to_binding,
                "reason": reason,
                "authorized_at": str(authorized_at),
            }
        )
        cursor = to_binding
    if cursor != current:
        raise BindingError("producer lineage does not reach current binding")
    return origin, current, normalized


def _producer_lineage(
    origin_binding: Mapping[str, object],
) -> dict[str, object]:
    origin = _validated_producer_binding(
        origin_binding,
        source="producer lineage origin",
    )
    return {
        "schema": PRODUCER_LINEAGE_SCHEMA,
        "origin": origin,
        "current": dict(origin),
        "upgrades": [],
    }


def _authorize_producer_lineage_upgrade(
    lineage: object,
    *,
    current_binding: Mapping[str, object],
    allow_from_sha256: str | None,
    reason: str | None,
    authorized_at: str,
) -> dict[str, object]:
    origin, previous, upgrades = _parsed_producer_lineage(lineage)
    destination = _validated_producer_binding(
        current_binding,
        source="producer lineage upgrade destination",
    )
    if previous == destination:
        return {
            "schema": PRODUCER_LINEAGE_SCHEMA,
            "origin": origin,
            "current": previous,
            "upgrades": upgrades,
        }
    if (
        not isinstance(allow_from_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", allow_from_sha256) is None
        or allow_from_sha256 != previous["script_sha256"]
    ):
        raise ValueError(
            "explicit producer upgrade authorization must name the exact "
            "current lineage script SHA-256"
        )
    if (
        not isinstance(reason, str)
        or not 1 <= len(reason) <= 200
        or any(ord(char) < 32 or ord(char) == 127 for char in reason)
    ):
        raise ValueError(
            "producer upgrade reason must be 1-200 printable characters"
        )
    if not _is_canonical_utc_timestamp(authorized_at):
        raise ValueError("producer upgrade authorization time is invalid")
    upgraded = [
        *upgrades,
        {
            "from": previous,
            "to": destination,
            "reason": reason,
            "authorized_at": authorized_at,
        },
    ]
    return {
        "schema": PRODUCER_LINEAGE_SCHEMA,
        "origin": origin,
        "current": destination,
        "upgrades": upgraded,
    }


def _validate_producer_lineage(
    value: object,
    *,
    artifact_binding: Mapping[str, object],
    current_binding: Mapping[str, object],
) -> dict[str, str]:
    origin, current, _upgrades = _parsed_producer_lineage(value)
    expected_artifact = _validated_producer_binding(
        artifact_binding,
        source="producer lineage artifact",
    )
    expected_current = _validated_producer_binding(
        current_binding,
        source="producer lineage expected current",
    )
    if origin != expected_artifact or current != expected_current:
        raise BindingError(
            "producer lineage origin/current binding is invalid"
        )
    return origin


def _current_job_rescue_producer_binding() -> dict[str, str]:
    return {
        "script_sha256": _sha256_file(
            _REPO_ROOT / "scripts" / "ci_job_log_rescue.py"
        ),
        "semantic_contract_sha256": (
            JOB_RESCUE_SEMANTIC_CONTRACT_SHA256
        ),
    }


def _current_preserved_recovery_producer_binding() -> dict[str, str]:
    return {
        "script_sha256": _sha256_file(
            _REPO_ROOT / "scripts" / "recover_ci_preserved_archives.py"
        ),
        "semantic_contract_sha256": (
            PRESERVED_RECOVERY_SEMANTIC_CONTRACT_SHA256
        ),
    }


def _validate_job_rescue_operator_audit(
    value: object,
    *,
    receipt_sha256: str,
    source_row_sha256: str,
    source_state: Mapping[str, object],
    archive_sha256: str,
    archive_size: int,
    receipt_producer_binding: object = _UNSPECIFIED_PRODUCER_BINDING,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema",
            "producer_lineage",
            "receipt_sha256",
            "source_row_sha256",
            "source_state_sha256",
            "synthetic_zip",
            "jobs_ledger_sha256",
        }
        or value.get("schema") != JOB_RESCUE_LEDGER_EVIDENCE_SCHEMA
        or value.get("receipt_sha256") != receipt_sha256
        or value.get("source_row_sha256") != source_row_sha256
        or value.get("source_state_sha256")
        != _sha256_bytes(_canonical_json_bytes(source_state))
        or value.get("jobs_ledger_sha256")
        != source_state.get("jobs_ledger_sha256")
    ):
        raise BindingError("job-rescue operator ledger binding is invalid")
    lineage_origin, _lineage_current, _lineage_upgrades = (
        _parsed_producer_lineage(value.get("producer_lineage"))
    )
    if receipt_producer_binding is _UNSPECIFIED_PRODUCER_BINDING:
        artifact_binding = lineage_origin
    elif receipt_producer_binding is None:
        if (
            lineage_origin["semantic_contract_sha256"]
            != JOB_RESCUE_LEGACY_SEMANTIC_CONTRACT_SHA256
        ):
            raise BindingError(
                "legacy job-rescue receipt lacks its legacy producer lineage"
            )
        artifact_binding = lineage_origin
    else:
        artifact_binding = _validated_producer_binding(
            receipt_producer_binding,
            source="job-rescue receipt",
        )
    _validate_producer_lineage(
        value.get("producer_lineage"),
        artifact_binding=artifact_binding,
        current_binding=_current_job_rescue_producer_binding(),
    )
    synthetic_zip = value.get("synthetic_zip")
    if (
        not isinstance(synthetic_zip, Mapping)
        or dict(synthetic_zip)
        != {"sha256": archive_sha256, "bytes": archive_size}
    ):
        raise BindingError("job-rescue operator ZIP binding is invalid")


def _validate_preserved_recovery_receipt(
    value: object,
    *,
    repo: str,
    run_id: int,
    attempt: int,
    created_at: str,
    archive_sha256: str,
    archive_size: int,
    verified_at: str,
) -> tuple[
    str,
    str,
    str,
    list[dict[str, object]],
    dict[str, str],
]:
    if not isinstance(value, Mapping):
        raise BindingError(
            "preserved-recovery receipt shape/binding is invalid"
        )
    current_receipt_fields = {
        "schema",
        "status",
        "verified_at",
        "recovery_id",
        "producer_binding",
        "proof",
    }
    legacy_receipt_fields = current_receipt_fields - {"producer_binding"}
    if (
        set(value)
        not in (current_receipt_fields, legacy_receipt_fields)
        or value.get("schema") != PRESERVED_RECOVERY_RECEIPT_SCHEMA
        or value.get("status") != "verified"
        or value.get("verified_at") != verified_at
        or not _is_canonical_utc_timestamp(verified_at)
    ):
        raise BindingError("preserved-recovery receipt shape/binding is invalid")
    proof = value.get("proof")
    if (
        not isinstance(proof, Mapping)
        or set(proof)
        != {
            "state",
            "attempt",
            "durable_member_witness",
            "source_archive",
            "rescue_archive",
            "verification",
            "rejected_candidates",
            "recovery_script_sha256",
        }
        or value.get("recovery_id")
        != _sha256_bytes(_canonical_json_bytes(proof))
    ):
        raise BindingError("preserved-recovery proof binding is invalid")
    artifact_producer_binding = (
        _preserved_recovery_receipt_producer_binding(value)
    )
    state = proof.get("state")
    source_attempt = proof.get("attempt")
    witness = proof.get("durable_member_witness")
    source_archive = proof.get("source_archive")
    rescue_archive = proof.get("rescue_archive")
    verification = proof.get("verification")
    rejected = proof.get("rejected_candidates")
    if (
        not isinstance(state, Mapping)
        or set(state) != {"path", "attempt_row_sha256"}
        or not isinstance(state.get("path"), str)
        or not state.get("path")
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(state.get("attempt_row_sha256")),
        )
        is None
        or not isinstance(source_attempt, Mapping)
        or set(source_attempt)
        != {
            "repo",
            "run_id",
            "attempt",
            "created_at",
            "prior_status",
            "tries",
            "terminal_http_status",
            "terminal_body_sha256",
        }
        or source_attempt.get("repo") != repo
        or source_attempt.get("run_id") != run_id
        or source_attempt.get("attempt") != attempt
        or source_attempt.get("created_at") != created_at
        or source_attempt.get("prior_status")
        not in {"failed", "terminal_404", "terminal_410"}
        or isinstance(source_attempt.get("tries"), bool)
        or not isinstance(source_attempt.get("tries"), int)
        or int(source_attempt["tries"]) < 0
        or not isinstance(witness, Mapping)
        or set(witness)
        != {
            "count",
            "chunk_count",
            "occurrence_tokens",
            "set_sha256",
            "members",
        }
        or not isinstance(source_archive, Mapping)
        or set(source_archive)
        != {
            "path",
            "bytes",
            "sha256",
            "zip_members",
            "uncompressed_bytes",
        }
        or not isinstance(rescue_archive, Mapping)
        or set(rescue_archive) != {"path", "bytes", "sha256"}
        or not isinstance(verification, Mapping)
        or dict(verification)
        != {
            "complete_zip_crc_read": True,
            "all_durable_members_matched_name_size_sha256": True,
            "different_valid_archive_candidates_rejected": True,
        }
        or not isinstance(rejected, list)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"path", "reason"}
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("reason"), str)
            for item in rejected
        )
    ):
        raise BindingError("preserved-recovery nested proof is invalid")
    if (
        source_archive.get("bytes") != archive_size
        or source_archive.get("sha256") != archive_sha256
        or rescue_archive.get("bytes") != archive_size
        or rescue_archive.get("sha256") != archive_sha256
        or isinstance(source_archive.get("zip_members"), bool)
        or not isinstance(source_archive.get("zip_members"), int)
        or int(source_archive["zip_members"]) < 1
        or isinstance(source_archive.get("uncompressed_bytes"), bool)
        or not isinstance(source_archive.get("uncompressed_bytes"), int)
        or int(source_archive["uncompressed_bytes"]) < 0
    ):
        raise BindingError("preserved-recovery archive proof differs")
    members = witness.get("members")
    count_values = (
        witness.get("count"),
        witness.get("chunk_count"),
        witness.get("occurrence_tokens"),
    )
    if (
        not isinstance(members, list)
        or not members
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in count_values
        )
        or witness.get("count") != len(members)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(witness.get("set_sha256")),
        )
        is None
    ):
        raise BindingError("preserved-recovery witness aggregate is invalid")
    witness_tuples: list[tuple[object, ...]] = []
    seen_members: set[str] = set()
    normalized_members: list[dict[str, object]] = []
    for member in members:
        if (
            not isinstance(member, Mapping)
            or set(member)
            != {
                "archive_member",
                "job_key",
                "raw_sha256",
                "raw_size",
                "chunk_count",
                "occurrence_tokens",
            }
            or not isinstance(member.get("archive_member"), str)
            or not member.get("archive_member")
            or member.get("archive_member") in seen_members
            or not isinstance(member.get("job_key"), str)
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(member.get("raw_sha256")),
            )
            is None
            or any(
                isinstance(member.get(field), bool)
                or not isinstance(member.get(field), int)
                or int(member[field]) < 0
                for field in (
                    "raw_size",
                    "chunk_count",
                    "occurrence_tokens",
                )
            )
        ):
            raise BindingError(
                "preserved-recovery member witness is invalid"
            )
        seen_members.add(str(member["archive_member"]))
        normalized = dict(member)
        normalized_members.append(normalized)
        witness_tuples.append(
            (
                normalized["archive_member"],
                normalized["job_key"],
                normalized["raw_sha256"],
                normalized["raw_size"],
                normalized["chunk_count"],
                normalized["occurrence_tokens"],
            )
        )
    if (
        normalized_members
        != sorted(
            normalized_members,
            key=lambda item: str(item["archive_member"]),
        )
        or witness.get("set_sha256")
        != _sha256_bytes(_canonical_json_bytes(witness_tuples))
        or witness.get("chunk_count")
        != sum(int(item["chunk_count"]) for item in normalized_members)
        or witness.get("occurrence_tokens")
        != sum(
            int(item["occurrence_tokens"]) for item in normalized_members
        )
    ):
        raise BindingError("preserved-recovery witness conservation differs")
    return (
        str(value["recovery_id"]),
        str(state["attempt_row_sha256"]),
        str(witness["set_sha256"]),
        normalized_members,
        artifact_producer_binding,
    )


def _validate_preserved_recovery_operator_audit(
    value: object,
    *,
    recovery_id: str,
    receipt_name: str,
    receipt_bytes: int,
    receipt_sha256: str,
    source_row_sha256: str,
    witness_set_sha256: str,
    archive_sha256: str,
    archive_size: int,
    artifact_producer_binding: object = _UNSPECIFIED_PRODUCER_BINDING,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema",
            "producer_lineage",
            "recovery_id",
            "receipt",
            "source_row_sha256",
            "witness_set_sha256",
            "archive",
        }
        or value.get("schema") != PRESERVED_RECOVERY_LEDGER_SCHEMA
        or value.get("recovery_id") != recovery_id
        or value.get("source_row_sha256") != source_row_sha256
        or value.get("witness_set_sha256") != witness_set_sha256
        or value.get("receipt")
        != {
            "name": receipt_name,
            "bytes": receipt_bytes,
            "sha256": receipt_sha256,
        }
        or value.get("archive")
        != {"sha256": archive_sha256, "bytes": archive_size}
    ):
        raise BindingError("preserved-recovery operator ledger is invalid")
    lineage_origin, _lineage_current, _lineage_upgrades = (
        _parsed_producer_lineage(value.get("producer_lineage"))
    )
    artifact_binding = (
        lineage_origin
        if artifact_producer_binding is _UNSPECIFIED_PRODUCER_BINDING
        else artifact_producer_binding
    )
    _validate_producer_lineage(
        value.get("producer_lineage"),
        artifact_binding=artifact_binding,
        current_binding=_current_preserved_recovery_producer_binding(),
    )


def _validate_preserved_archive_provenance(
    value: object,
    *,
    repo: str,
    run_id: int,
    attempt: int,
    created_at: str,
    archive_sha256: str,
    archive_size: int,
) -> tuple[
    str,
    str,
    str,
    list[dict[str, object]],
    str,
    int,
    str,
]:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {"schema", "manifest", "archive", "recovery_receipt"}
        or value.get("schema") != PRESERVED_ARCHIVE_PROVENANCE_SCHEMA
    ):
        raise BindingError("preserved-archive provenance schema is invalid")
    manifest = value.get("manifest")
    archive = value.get("archive")
    receipt_evidence = value.get("recovery_receipt")
    if (
        not isinstance(manifest, Mapping)
        or set(manifest)
        != {
            "manifest_file_sha256",
            "record",
            "record_sha256",
        }
        or not isinstance(archive, Mapping)
        or dict(archive)
        != {
            "source": "preserved-local-archive",
            "name": (
                f"{repo.replace('/', '__')}--{run_id}"
                f"--attempt-{attempt}.zip"
            ),
            "bytes": archive_size,
            "sha256": archive_sha256,
        }
        or not isinstance(receipt_evidence, Mapping)
        or set(receipt_evidence)
        != {"name", "bytes", "sha256", "receipt"}
    ):
        raise BindingError("preserved-archive provenance shape is invalid")
    record = manifest.get("record")
    finished_at = (
        record.get("finished_at") if isinstance(record, Mapping) else None
    )
    if (
        not isinstance(record, Mapping)
        or set(record)
        != {
            "repo",
            "run_id",
            "attempt",
            "created_at",
            "status",
            "bytes",
            "sha256",
            "finished_at",
        }
        or dict(record)
        != {
            "repo": repo,
            "run_id": str(run_id),
            "attempt": str(attempt),
            "created_at": created_at,
            "status": "zip",
            "bytes": str(archive_size),
            "sha256": archive_sha256,
            "finished_at": finished_at,
        }
        or not _is_canonical_utc_timestamp(created_at)
        or not _is_canonical_utc_timestamp(finished_at)
        or manifest.get("record_sha256")
        != _sha256_bytes(_canonical_json_bytes(record))
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(manifest.get("manifest_file_sha256")),
        )
        is None
    ):
        raise BindingError("preserved-archive manifest binding is invalid")
    receipt = receipt_evidence.get("receipt")
    if not isinstance(receipt, Mapping):
        raise BindingError("preserved-recovery receipt is missing")
    receipt_raw = _canonical_json_bytes(receipt) + b"\n"
    receipt_name = receipt_evidence.get("name")
    if (
        not isinstance(receipt_name, str)
        or receipt_evidence.get("bytes") != len(receipt_raw)
        or receipt_evidence.get("sha256") != _sha256_bytes(receipt_raw)
    ):
        raise BindingError("preserved-recovery receipt artifact differs")
    (
        recovery_id,
        source_row,
        witness_sha,
        witnesses,
        _artifact_producer_binding,
    ) = (
        _validate_preserved_recovery_receipt(
            receipt,
            repo=repo,
            run_id=run_id,
            attempt=attempt,
            created_at=created_at,
            archive_sha256=archive_sha256,
            archive_size=archive_size,
            verified_at=str(finished_at),
        )
    )
    expected_receipt_name = (
        f"{repo.replace('/', '__')}--{run_id}--attempt-{attempt}"
        f".preserved-recovery-{recovery_id[:16]}.json"
    )
    if receipt_name != expected_receipt_name:
        raise BindingError("preserved-recovery receipt name is invalid")
    return (
        recovery_id,
        source_row,
        witness_sha,
        witnesses,
        receipt_name,
        len(receipt_raw),
        _sha256_bytes(receipt_raw),
    )


def _validate_run_metadata_identity(
    value: Mapping[str, object],
    *,
    run_id: int,
    attempt: int,
) -> None:
    metadata_run_id = value.get("id")
    metadata_attempt = value.get("run_attempt")
    if (
        isinstance(metadata_run_id, bool)
        or not isinstance(metadata_run_id, int)
        or metadata_run_id != run_id
    ):
        raise MalformedResponseError(
            f"run metadata id {metadata_run_id!r} does not match {run_id}"
        )
    if (
        isinstance(metadata_attempt, bool)
        or not isinstance(metadata_attempt, int)
        or metadata_attempt != attempt
    ):
        raise MalformedResponseError(
            "run metadata attempt "
            f"{metadata_attempt!r} does not match {attempt}"
        )
    created_at = value.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise MalformedResponseError("run metadata created_at is invalid")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _script_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _parser_sha256() -> str:
    import scripts.ci_log_sidecars as parser_module

    return _sha256_file(Path(parser_module.__file__).resolve())


def _content_store_sha256() -> str:
    import scripts.ci_content_store as store_module

    return _sha256_file(Path(store_module.__file__).resolve())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fetch_state_lease_path(
    state_path: str | os.PathLike[str],
) -> tuple[Path, Path]:
    state = Path(
        os.path.abspath(
            os.fspath(Path(state_path).expanduser())
        )
    )
    return state, state.with_name(f"{state.name}.lease")


_FETCH_STATE_INODE_LEASES: dict[
    int,
    tuple[int, int, Path, int, int],
] = {}
_FETCH_STATE_INODE_LEASES_LOCK = threading.Lock()


def _require_safe_fetch_state_path_if_present(
    state: Path,
) -> os.stat_result | None:
    try:
        state_stat = os.lstat(state)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise BindingError(f"fetch-state path is unsafe: {state}") from exc
    if (
        stat.S_ISLNK(state_stat.st_mode)
        or not stat.S_ISREG(state_stat.st_mode)
        or state_stat.st_nlink != 1
        or state_stat.st_uid != os.geteuid()
    ):
        raise BindingError(f"fetch-state path is unsafe: {state}")
    return state_stat


def _validate_fetch_state_lease_file(
    descriptor: int,
    *,
    state_path: str | os.PathLike[str],
) -> tuple[Path, os.stat_result | None]:
    state, lease_path = _fetch_state_lease_path(state_path)
    try:
        lease_stat = os.fstat(descriptor)
        path_stat = os.lstat(lease_path)
    except OSError as exc:
        raise BindingError(
            f"fetch-state lease path is unsafe: {lease_path}"
        ) from exc
    if (
        not stat.S_ISREG(lease_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or stat.S_ISLNK(path_stat.st_mode)
        or lease_stat.st_nlink != 1
        or path_stat.st_nlink != 1
        or lease_stat.st_uid != os.geteuid()
        or path_stat.st_uid != os.geteuid()
        or (lease_stat.st_dev, lease_stat.st_ino)
        != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise BindingError(
            f"fetch-state lease path is unsafe: {lease_path}"
        )
    return state, _require_safe_fetch_state_path_if_present(state)


def _acquire_fetch_state_inode_lease(
    descriptor: int,
    *,
    state_path: str | os.PathLike[str],
    create: bool,
) -> None:
    state, state_stat = _validate_fetch_state_lease_file(
        descriptor,
        state_path=state_path,
    )
    with _FETCH_STATE_INODE_LEASES_LOCK:
        held = _FETCH_STATE_INODE_LEASES.get(descriptor)
    if held is not None:
        (
            inode_lease_descriptor,
            state_descriptor,
            inode_lease_path,
            state_device,
            state_inode,
        ) = held
        if state_stat is None:
            raise BindingError(
                f"fetch-state path disappeared while leased: {state}"
            )
        try:
            guarded = os.fstat(state_descriptor)
            inode_lease_stat = os.fstat(inode_lease_descriptor)
            inode_lease_path_stat = os.lstat(inode_lease_path)
        except OSError as exc:
            raise BindingError(
                f"fetch-state inode lease is invalid: {state}"
            ) from exc
        if (
            not stat.S_ISREG(guarded.st_mode)
            or guarded.st_nlink != 1
            or guarded.st_uid != os.geteuid()
            or (guarded.st_dev, guarded.st_ino)
            != (state_stat.st_dev, state_stat.st_ino)
            or (guarded.st_dev, guarded.st_ino)
            != (state_device, state_inode)
            or not stat.S_ISREG(inode_lease_stat.st_mode)
            or inode_lease_stat.st_nlink != 1
            or inode_lease_stat.st_uid != os.geteuid()
            or stat.S_ISLNK(inode_lease_path_stat.st_mode)
            or (inode_lease_stat.st_dev, inode_lease_stat.st_ino)
            != (
                inode_lease_path_stat.st_dev,
                inode_lease_path_stat.st_ino,
            )
        ):
            raise BindingError(
                f"fetch-state inode changed while leased: {state}"
            )
        return
    if state_stat is None and not create:
        return
    flags = os.O_RDWR
    if create:
        flags |= os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        state_descriptor = os.open(state, flags, 0o600)
    except OSError as exc:
        raise BindingError(
            f"fetch-state inode cannot be leased safely: {state}"
        ) from exc
    inode_lease_descriptor = -1
    try:
        current = _require_safe_fetch_state_path_if_present(state)
        guarded = os.fstat(state_descriptor)
        if (
            current is None
            or not stat.S_ISREG(guarded.st_mode)
            or guarded.st_nlink != 1
            or guarded.st_uid != os.geteuid()
            or (guarded.st_dev, guarded.st_ino)
            != (current.st_dev, current.st_ino)
        ):
            raise BindingError(
                f"fetch-state inode cannot be leased safely: {state}"
            )
        inode_lease_directory = Path(tempfile.gettempdir()) / (
            f"cppmega-ci-fetch-state-inode-leases-{os.geteuid()}"
        )
        try:
            os.mkdir(inode_lease_directory, 0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise BindingError(
                "fetch-state inode lease directory cannot be created safely"
            ) from exc
        try:
            inode_directory_stat = os.lstat(inode_lease_directory)
        except OSError as exc:
            raise BindingError(
                "fetch-state inode lease directory cannot be inspected"
            ) from exc
        if (
            stat.S_ISLNK(inode_directory_stat.st_mode)
            or not stat.S_ISDIR(inode_directory_stat.st_mode)
            or inode_directory_stat.st_uid != os.geteuid()
            or stat.S_IMODE(inode_directory_stat.st_mode) & 0o077
        ):
            raise BindingError(
                "fetch-state inode lease directory is unsafe"
            )
        inode_lease_path = inode_lease_directory / (
            f"{guarded.st_dev:x}-{guarded.st_ino:x}.lease"
        )
        inode_lease_flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            inode_lease_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            inode_lease_flags |= os.O_NOFOLLOW
        try:
            inode_lease_descriptor = os.open(
                inode_lease_path,
                inode_lease_flags,
                0o600,
            )
        except OSError as exc:
            raise BindingError(
                f"fetch-state inode lease path is unsafe: {inode_lease_path}"
            ) from exc
        inode_lease_stat = os.fstat(inode_lease_descriptor)
        inode_lease_path_stat = os.lstat(inode_lease_path)
        if (
            not stat.S_ISREG(inode_lease_stat.st_mode)
            or not stat.S_ISREG(inode_lease_path_stat.st_mode)
            or stat.S_ISLNK(inode_lease_path_stat.st_mode)
            or inode_lease_stat.st_nlink != 1
            or inode_lease_path_stat.st_nlink != 1
            or inode_lease_stat.st_uid != os.geteuid()
            or inode_lease_path_stat.st_uid != os.geteuid()
            or (inode_lease_stat.st_dev, inode_lease_stat.st_ino)
            != (
                inode_lease_path_stat.st_dev,
                inode_lease_path_stat.st_ino,
            )
        ):
            raise BindingError(
                f"fetch-state inode lease path is unsafe: {inode_lease_path}"
            )
        try:
            fcntl.flock(
                inode_lease_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise BindingError(
                f"fetch state already has a live process lease: {state}"
            ) from exc
        current = _require_safe_fetch_state_path_if_present(state)
        guarded = os.fstat(state_descriptor)
        if (
            current is None
            or guarded.st_nlink != 1
            or (guarded.st_dev, guarded.st_ino)
            != (current.st_dev, current.st_ino)
        ):
            raise BindingError(
                f"fetch-state inode changed during lease acquisition: {state}"
            )
        inode_record = (
            f"device={guarded.st_dev}\ninode={guarded.st_ino}\n"
            f"state={state}\n"
        ).encode()
        os.ftruncate(inode_lease_descriptor, 0)
        view = memoryview(inode_record)
        while view:
            written = os.write(inode_lease_descriptor, view)
            if written <= 0:
                raise OSError(
                    "fetch-state inode lease write made no progress"
                )
            view = view[written:]
        os.fsync(inode_lease_descriptor)
        with _FETCH_STATE_INODE_LEASES_LOCK:
            if descriptor in _FETCH_STATE_INODE_LEASES:
                raise BindingError(
                    f"fetch-state inode lease was acquired twice: {state}"
                )
            _FETCH_STATE_INODE_LEASES[descriptor] = (
                inode_lease_descriptor,
                state_descriptor,
                inode_lease_path,
                guarded.st_dev,
                guarded.st_ino,
            )
        inode_lease_descriptor = -1
        state_descriptor = -1
    finally:
        if inode_lease_descriptor >= 0:
            try:
                fcntl.flock(
                    inode_lease_descriptor,
                    fcntl.LOCK_UN,
                )
            finally:
                os.close(inode_lease_descriptor)
        if state_descriptor >= 0:
            os.close(state_descriptor)


def _validate_fetch_state_process_lease(
    descriptor: int,
    *,
    state_path: str | os.PathLike[str],
) -> None:
    state, state_stat = _validate_fetch_state_lease_file(
        descriptor,
        state_path=state_path,
    )
    with _FETCH_STATE_INODE_LEASES_LOCK:
        held = _FETCH_STATE_INODE_LEASES.get(descriptor)
    if state_stat is None:
        if held is not None:
            raise BindingError(
                f"fetch-state path disappeared while leased: {state}"
            )
        return
    if held is None:
        raise BindingError(
            f"fetch-state inode has no live process lease: {state}"
        )
    (
        inode_lease_descriptor,
        state_descriptor,
        inode_lease_path,
        state_device,
        state_inode,
    ) = held
    try:
        guarded = os.fstat(state_descriptor)
        inode_lease_stat = os.fstat(inode_lease_descriptor)
        inode_lease_path_stat = os.lstat(inode_lease_path)
    except OSError as exc:
        raise BindingError(
            f"fetch-state inode lease is invalid: {state}"
        ) from exc
    if (
        not stat.S_ISREG(guarded.st_mode)
        or guarded.st_nlink != 1
        or guarded.st_uid != os.geteuid()
        or (guarded.st_dev, guarded.st_ino)
        != (state_stat.st_dev, state_stat.st_ino)
        or (guarded.st_dev, guarded.st_ino)
        != (state_device, state_inode)
        or not stat.S_ISREG(inode_lease_stat.st_mode)
        or inode_lease_stat.st_nlink != 1
        or inode_lease_stat.st_uid != os.geteuid()
        or stat.S_ISLNK(inode_lease_path_stat.st_mode)
        or (inode_lease_stat.st_dev, inode_lease_stat.st_ino)
        != (
            inode_lease_path_stat.st_dev,
            inode_lease_path_stat.st_ino,
        )
    ):
        raise BindingError(
            f"fetch-state inode changed while leased: {state}"
        )


def _acquire_fetch_state_process_lease(
    state_path: str | os.PathLike[str],
    *,
    owner: str,
) -> int:
    """Acquire the cross-process lease shared by fetch, rescue, and finalizers."""

    if (
        not isinstance(owner, str)
        or not owner
        or "\n" in owner
        or "\r" in owner
    ):
        raise ValueError("fetch-state lease owner must be one nonempty line")
    state, lease_path = _fetch_state_lease_path(state_path)
    state.parent.mkdir(parents=True, exist_ok=True)
    _require_safe_fetch_state_path_if_present(state)
    lease_flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        lease_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        lease_flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            lease_path,
            lease_flags,
            0o600,
        )
    except OSError as exc:
        raise BindingError(
            f"fetch-state lease path is unsafe: {lease_path}"
        ) from exc
    try:
        _validate_fetch_state_lease_file(
            descriptor,
            state_path=state,
        )
        try:
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise BindingError(
                f"fetch state already has a live process lease: {state}"
            ) from exc
        _acquire_fetch_state_inode_lease(
            descriptor,
            state_path=state,
            create=False,
        )
        lease_record = (
            f"pid={os.getpid()}\nstate={state}\nowner={owner}\n"
        ).encode()
        os.ftruncate(descriptor, 0)
        view = memoryview(lease_record)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("fetch-state lease write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        _validate_fetch_state_process_lease(
            descriptor,
            state_path=state,
        )
        return descriptor
    except BaseException:
        _release_fetch_state_process_lease(descriptor)
        raise


def _release_fetch_state_process_lease(descriptor: int) -> None:
    try:
        with _FETCH_STATE_INODE_LEASES_LOCK:
            held = _FETCH_STATE_INODE_LEASES.pop(
                descriptor,
                None,
            )
        if held is not None:
            (
                inode_lease_descriptor,
                state_descriptor,
                _inode_lease_path,
                _state_device,
                _state_inode,
            ) = held
            try:
                fcntl.flock(
                    inode_lease_descriptor,
                    fcntl.LOCK_UN,
                )
            finally:
                try:
                    os.close(inode_lease_descriptor)
                finally:
                    os.close(state_descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def atomic_write_json(path: str | os.PathLike[str], value: object) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n").encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def exhaustive_discovery_sidecar_path(
    state_path: str | os.PathLike[str],
) -> Path:
    state = Path(state_path).expanduser().resolve()
    return state.with_name(f"{state.name}.inventory-exhaustive.json")


def load_exhaustive_discovery_sidecar(
    path: str | os.PathLike[str],
) -> dict[str, object] | None:
    """Load and validate the durable exhaustive-inventory cursor sidecar."""

    source = Path(
        os.path.abspath(os.path.expanduser(os.fspath(path)))
    )
    if not source.exists():
        return None
    if source.is_symlink() or not source.is_file():
        raise BindingError(
            f"exhaustive discovery sidecar is missing or unsafe: {source}"
        )
    raw = source.read_bytes()
    if len(raw) > 1024 * 1024:
        raise BindingError("exhaustive discovery sidecar exceeds 1 MiB")

    def reject_duplicates(
        pairs: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise BindingError(
                    f"exhaustive discovery sidecar repeats key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BindingError(
            f"exhaustive discovery sidecar is invalid JSON: {exc}"
        ) from exc
    expected_keys = {
        "schema",
        "completion_mode",
        "inventory_receipt_sha256",
        "inventory_database_sha256",
        "inventory_db_logical_sha256",
        "expected_run_count",
        "expected_attempt_count",
        "expected_attempt_set_sha256",
        "cursor",
        "discovery_eof",
        "batches",
        "rows_seen",
        "started_at",
        "updated_at",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise BindingError(
            "exhaustive discovery sidecar has an unsupported shape"
        )
    if (
        value["schema"] != EXHAUSTIVE_DISCOVERY_SCHEMA
        or value["completion_mode"]
        != COMPLETION_MODE_INVENTORY_EXHAUSTIVE
    ):
        raise BindingError(
            "exhaustive discovery sidecar has an unsupported contract"
        )
    for key in (
        "inventory_receipt_sha256",
        "inventory_database_sha256",
        "inventory_db_logical_sha256",
        "expected_attempt_set_sha256",
    ):
        candidate = value[key]
        if (
            not isinstance(candidate, str)
            or re.fullmatch(r"[0-9a-f]{64}", candidate) is None
        ):
            raise BindingError(
                f"exhaustive discovery sidecar {key} is invalid"
            )
    for key in (
        "expected_run_count",
        "expected_attempt_count",
        "batches",
        "rows_seen",
    ):
        candidate = value[key]
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate < 0
        ):
            raise BindingError(
                f"exhaustive discovery sidecar {key} is invalid"
            )
    if not isinstance(value["discovery_eof"], bool):
        raise BindingError(
            "exhaustive discovery sidecar discovery_eof is invalid"
        )
    cursor = value["cursor"]
    if cursor is not None and (
        not isinstance(cursor, list)
        or len(cursor) != 4
        or not isinstance(cursor[0], str)
        or not cursor[0]
        or not isinstance(cursor[1], str)
        or not cursor[1]
        or isinstance(cursor[2], bool)
        or not isinstance(cursor[2], int)
        or cursor[2] <= 0
        or isinstance(cursor[3], bool)
        or not isinstance(cursor[3], int)
        or cursor[3] <= 0
    ):
        raise BindingError(
            "exhaustive discovery sidecar cursor is invalid"
        )
    for key in ("started_at", "updated_at"):
        if not isinstance(value[key], str) or not value[key]:
            raise BindingError(
                f"exhaustive discovery sidecar {key} is invalid"
            )
    rows_seen = int(value["rows_seen"])
    expected_runs = int(value["expected_run_count"])
    if rows_seen > expected_runs or (rows_seen and cursor is None):
        raise BindingError(
            "exhaustive discovery sidecar cursor/count invariant failed"
        )
    if bool(value["discovery_eof"]) and rows_seen != expected_runs:
        raise BindingError(
            "exhaustive discovery sidecar EOF count is incomplete"
        )
    return value


def _safe_error(value: object, secrets: Iterable[str] = ()) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text[:4000]


def _safe_url_for_ledger(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if not parsed.query:
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, "", "")
        )
    keys = []
    for key, _ in urllib.parse.parse_qsl(
        parsed.query, keep_blank_values=True
    ):
        keys.append("<redacted>" if key.casefold() in _SECRET_QUERY_KEYS else key)
    query = "&".join(f"{key}=<redacted>" for key in keys)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, query, "")
    )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _default_no_redirect_requester(
    method: str,
    url: str,
    headers: Mapping[str, str],
    timeout: float,
) -> HTTPResponse:
    request = urllib.request.Request(
        url, headers=dict(headers), method=method
    )
    try:
        with _NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
            return HTTPResponse(
                status=int(response.status),
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return HTTPResponse(
            status=int(exc.code),
            headers=dict(exc.headers.items()) if exc.headers is not None else {},
            body=exc.read(),
        )


def _default_archive_downloader(
    url: str,
    destination: Path,
    *,
    timeout: float,
    max_bytes: int,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    max_transfer_attempts: int = DEFAULT_ARCHIVE_TRANSFER_ATTEMPTS,
) -> tuple[int, str]:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ArchiveError("GitHub returned an unsafe signed archive URL")
    if (
        isinstance(max_transfer_attempts, bool)
        or not isinstance(max_transfer_attempts, int)
        or max_transfer_attempts < 1
    ):
        raise ValueError("max_transfer_attempts must be a positive integer")
    if destination.exists() or destination.is_symlink():
        raise ArchiveError("archive download destination already exists")

    open_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        destination_fd = os.open(destination, open_flags, 0o600)
    except FileExistsError as exc:
        raise ArchiveError(
            "archive download destination already exists"
        ) from exc
    except OSError as exc:
        raise ArchiveError(
            "archive download destination could not be created safely"
        ) from exc

    created_stat = os.fstat(destination_fd)
    created_identity = (created_stat.st_dev, created_stat.st_ino)

    def verify_destination_identity() -> os.stat_result:
        current = os.fstat(destination_fd)
        try:
            visible = destination.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArchiveError(
                "archive download destination identity changed"
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or not stat.S_ISREG(visible.st_mode)
            or (current.st_dev, current.st_ino) != created_identity
            or (visible.st_dev, visible.st_ino) != created_identity
        ):
            raise ArchiveError(
                "archive download destination identity changed"
            )
        return current

    def strong_etag(headers: Mapping[str, str]) -> str | None:
        value = headers.get("ETag")
        if value is None:
            return None
        value = value.strip()
        if (
            len(value) < 2
            or not value.startswith('"')
            or not value.endswith('"')
            or value[:2].casefold() == "w/"
        ):
            return None
        return value

    def write_all(payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:
                raise OSError("archive destination write made no progress")
            view = view[written:]

    def digest_created_file() -> str:
        verify_destination_identity()
        os.lseek(destination_fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while block := os.read(destination_fd, 1024 * 1024):
            digest.update(block)
        verify_destination_identity()
        return digest.hexdigest()

    expected_total: int | None = None
    validator: str | None = None
    last_transport_error: BaseException | None = None
    try:
        for transfer_attempt in range(1, max_transfer_attempts + 1):
            current = verify_destination_identity()
            offset = current.st_size
            resume_offset = offset if offset and validator is not None else 0

            # Deliberately no Authorization header: the signed URL is the
            # credential. A byte-range append is allowed only with a strong
            # ETag for this exact signed representation.
            headers = {"User-Agent": "cppmega-ci-stream-fetch/1"}
            if resume_offset:
                headers["Range"] = f"bytes={resume_offset}-"
                headers["If-Range"] = validator
            request = urllib.request.Request(
                url,
                headers=headers,
                method="GET",
            )
            try:
                with urlopen(request, timeout=timeout) as response:
                    status_code = int(response.status)
                    response_validator = strong_etag(response.headers)
                    response_end_exclusive: int | None = None
                    append = resume_offset > 0 and status_code == 206
                    if append:
                        content_range = response.headers.get("Content-Range")
                        match = (
                            None
                            if content_range is None
                            else re.fullmatch(
                                r"bytes ([0-9]+)-([0-9]+)/([0-9]+)",
                                content_range.strip(),
                            )
                        )
                        if match is None:
                            raise MalformedResponseError(
                                "signed archive resume lacks a valid "
                                "Content-Range"
                            )
                        start, end, total = (
                            int(value) for value in match.groups()
                        )
                        if (
                            start != resume_offset
                            or end < start
                            or end >= total
                            or total > max_bytes
                            or (
                                expected_total is not None
                                and total != expected_total
                            )
                            or response_validator != validator
                        ):
                            raise MalformedResponseError(
                                "signed archive resumed byte range is "
                                "inconsistent"
                            )
                        content_length = response.headers.get(
                            "Content-Length"
                        )
                        if content_length is not None:
                            try:
                                remaining = int(content_length)
                            except ValueError as exc:
                                raise MalformedResponseError(
                                    "signed archive Content-Length is not an "
                                    "integer"
                                ) from exc
                            if remaining != end - start + 1:
                                raise MalformedResponseError(
                                    "signed archive resumed Content-Length is "
                                    "inconsistent"
                                )
                        expected_total = total
                        response_end_exclusive = end + 1
                        os.lseek(destination_fd, resume_offset, os.SEEK_SET)
                    elif status_code == 206:
                        raise MalformedResponseError(
                            "signed archive returned a byte range without a "
                            "strong resume validator"
                        )
                    elif status_code == 200:
                        # If-Range permits a complete 200 when the
                        # representation changed. It is safe only as a full
                        # restart of the stable private file descriptor.
                        content_length = response.headers.get(
                            "Content-Length"
                        )
                        if content_length is None:
                            expected_total = None
                        else:
                            try:
                                expected_total = int(content_length)
                            except ValueError as exc:
                                raise MalformedResponseError(
                                    "signed archive Content-Length is not an "
                                    "integer"
                                ) from exc
                            if (
                                expected_total < 0
                                or expected_total > max_bytes
                            ):
                                raise ArchiveError(
                                    f"archive Content-Length {expected_total} "
                                    f"exceeds limit {max_bytes}"
                                )
                        validator = response_validator
                        response_end_exclusive = expected_total
                        os.ftruncate(destination_fd, 0)
                        os.lseek(destination_fd, 0, os.SEEK_SET)
                    else:
                        raise APIError(
                            f"signed archive URL returned HTTP {status_code}"
                        )

                    verify_destination_identity()
                    while True:
                        try:
                            block = response.read(1024 * 1024)
                        except (
                            urllib.error.URLError,
                            http.client.HTTPException,
                            TimeoutError,
                            ConnectionError,
                        ) as exc:
                            partial = getattr(exc, "partial", b"")
                            if isinstance(partial, bytes) and partial:
                                position = os.lseek(
                                    destination_fd, 0, os.SEEK_CUR
                                )
                                if (
                                    response_end_exclusive is not None
                                    and position + len(partial)
                                    > response_end_exclusive
                                ):
                                    raise MalformedResponseError(
                                        "signed archive response exceeded its "
                                        "declared byte range"
                                    ) from exc
                                if position + len(partial) > max_bytes:
                                    raise ArchiveError(
                                        f"archive exceeded byte limit "
                                        f"{max_bytes}"
                                    ) from exc
                                write_all(partial)
                            os.fsync(destination_fd)
                            raise
                        if not block:
                            break
                        position = os.lseek(
                            destination_fd, 0, os.SEEK_CUR
                        )
                        if (
                            response_end_exclusive is not None
                            and position + len(block)
                            > response_end_exclusive
                        ):
                            raise MalformedResponseError(
                                "signed archive response exceeded its "
                                "declared byte range"
                            )
                        if position + len(block) > max_bytes:
                            raise ArchiveError(
                                f"archive exceeded byte limit {max_bytes}"
                            )
                        write_all(block)
                    os.fsync(destination_fd)
                    position = os.lseek(
                        destination_fd, 0, os.SEEK_CUR
                    )
                    if (
                        response_end_exclusive is not None
                        and position != response_end_exclusive
                    ):
                        raise http.client.IncompleteRead(
                            b"",
                            max(0, response_end_exclusive - position),
                        )
                    completed = verify_destination_identity()
                    if (
                        expected_total is not None
                        and completed.st_size != expected_total
                    ):
                        raise http.client.IncompleteRead(
                            b"",
                            max(0, expected_total - completed.st_size),
                        )
                    if completed.st_size == 0:
                        raise ArchiveError(
                            "signed archive response was empty"
                        )
                    return completed.st_size, digest_created_file()
            except urllib.error.HTTPError as exc:
                raise APIError(
                    f"signed archive URL returned HTTP {exc.code}"
                ) from exc
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                ConnectionError,
            ) as exc:
                last_transport_error = exc
                current = verify_destination_identity()
                if current.st_size > max_bytes:
                    raise ArchiveError(
                        f"archive exceeded byte limit {max_bytes}"
                    ) from exc
                if transfer_attempt == max_transfer_attempts:
                    break
                continue

        assert last_transport_error is not None
        raise ArchiveError(
            "signed archive transport retries exhausted before EOF: "
            f"{type(last_transport_error).__name__}"
        ) from last_transport_error
    finally:
        os.close(destination_fd)


def _stream_signed_archive_response(
    response: Any,
    destination: Path,
    *,
    max_bytes: int,
) -> tuple[int, str]:
    """Durably stream one complete signed-URL response into a bounded file."""

    status_code = int(response.status)
    if status_code != 200:
        raise APIError(
            f"signed archive URL returned HTTP {status_code}"
        )
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise MalformedResponseError(
                "signed archive Content-Length is not an integer"
            ) from exc
        if declared < 0 or declared > max_bytes:
            raise ArchiveError(
                f"archive Content-Length {declared} exceeds limit "
                f"{max_bytes}"
            )

    digest = hashlib.sha256()
    total = 0
    with destination.open("xb") as output:
        while True:
            try:
                block = response.read(1024 * 1024)
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                ConnectionError,
            ) as exc:
                raise ArchiveError(
                    "signed archive transport failed before EOF: "
                    f"{type(exc).__name__}"
                ) from exc
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise ArchiveError(
                    f"archive exceeded byte limit {max_bytes}"
                )
            output.write(block)
            digest.update(block)
        output.flush()
        os.fsync(output.fileno())
    if total == 0:
        raise ArchiveError("signed archive response was empty")
    return total, digest.hexdigest()


def _semantic_callable_sha256(value: Callable[..., object]) -> str:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as exc:
        raise FetchError(
            "frozen tokenizer semantic source is unavailable"
        ) from exc
    normalized = (
        textwrap.dedent(source)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
        + "\n"
    )
    return _sha256_bytes(normalized.encode("utf-8"))


class ExactTokenizer:
    """Frozen training-tokenizer adapter with an auditable fingerprint."""

    def __init__(self, tokenizer_json: str | os.PathLike[str]):
        path = Path(tokenizer_json).expanduser().resolve()
        try:
            from tokenizers import __version__ as tokenizers_version
        except ImportError as exc:
            raise FetchError(
                "the existing project environment lacks the tokenizers package"
            ) from exc
        if not path.is_file() or path.is_symlink():
            raise FetchError(f"tokenizer.json is missing or unsafe: {path}")
        raw = path.read_bytes()
        self.path = path
        self.artifact_sha256 = _sha256_bytes(raw)
        try:
            self._tokenizer = load_cppmega_tokenizer(path)
        except TokenizerContractError as exc:
            raise FetchError(
                f"tokenizer.json does not satisfy the frozen cppmega "
                f"training contract: {path}: {exc}"
            ) from exc
        from cppmega.data.prompt_graph import (
            normalize_cpp_whitespace_with_offsets,
        )

        self.contract = {
            "schema": "cppmega_exact_ci_training_tokenizer_v3",
            "artifact_sha256": self.artifact_sha256,
            "tokenizer_contract_sha256": TOKENIZER_CONTRACT_SHA256,
            "library": "tokenizers",
            "library_version": str(tokenizers_version),
            "semantic_source_encoding": (
                "dedented-python-source-utf8-final-newline-v1"
            ),
            "semantic_function_sha256": {
                "tokenizer_init": _semantic_callable_sha256(
                    CppMegaTokenizer.__init__
                ),
                "tokenizer_encode": _semantic_callable_sha256(
                    CppMegaTokenizer.encode
                ),
                "tokenizer_encode_batch": _semantic_callable_sha256(
                    CppMegaTokenizer.encode_batch
                ),
                "tokenizer_loader": _semantic_callable_sha256(
                    load_cppmega_tokenizer
                ),
                "whitespace_normalizer": _semantic_callable_sha256(
                    normalize_cpp_whitespace_with_offsets
                ),
            },
            "training_adapter_semantics": (
                "frozen-training-tokenizer-encode-batch-v1"
            ),
            "whitespace_normalizer_semantics": (
                "cpp-whitespace-with-offsets-v1"
            ),
            "prepend_token": None,
            "append_token": None,
            "payload_only": True,
        }
        self.fingerprint = _sha256_bytes(
            _canonical_json_bytes(self.contract)
        )

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        if not texts:
            return []
        try:
            token_ids = self._tokenizer.encode_batch(list(texts))
        except (TypeError, ValueError) as exc:
            raise FetchError(
                f"cppmega training tokenizer rejected a CI payload: {exc}"
            ) from exc
        if len(token_ids) != len(texts):
            raise FetchError("tokenizer changed the batch cardinality")
        return token_ids


_PROCESS_TOKENIZERS: dict[str, ExactTokenizer] = {}


def _section_for_parsed_chunk(
    parsed: Mapping[str, object], chunk: Mapping[str, object]
) -> Mapping[str, object] | None:
    ordinal = chunk.get("section_ordinal")
    sections = parsed.get("sections")
    if (
        isinstance(ordinal, int)
        and not isinstance(ordinal, bool)
        and isinstance(sections, list)
        and 0 <= ordinal < len(sections)
        and isinstance(sections[ordinal], dict)
    ):
        return sections[ordinal]
    return None


def _materialize_parsed_member(
    raw: bytes,
    metadata: Mapping[str, object],
    *,
    max_chunk_chars: int,
    parser: Callable[..., Mapping[str, object]],
    tokenizer: ExactTokenizer,
) -> dict[str, object]:
    parsed = parser(raw, metadata, max_chunk_chars=max_chunk_chars)
    if not isinstance(parsed, Mapping):
        raise FetchError("CI parser returned a non-mapping result")
    canonical_text = parsed.get("canonical_text")
    dedup_text = parsed.get("dedup_text")
    chunks = parsed.get("chunks")
    sidecar = parsed.get("sidecar")
    if (
        not isinstance(canonical_text, str)
        or not isinstance(dedup_text, str)
        or not isinstance(chunks, list)
        or not isinstance(sidecar, dict)
        or any(not isinstance(item, dict) for item in chunks)
    ):
        raise FetchError("CI parser returned an invalid result contract")

    retained_chunks: list[dict[str, object]] = []
    chunk_texts: list[str] = []
    for raw_chunk in chunks:
        text = raw_chunk.get("text")
        if not isinstance(text, str):
            raise FetchError("parser chunk text is missing")
        if not text:
            continue
        retained_chunks.append(dict(raw_chunk))
        chunk_texts.append(text)
    token_batches = tokenizer.encode_batch(chunk_texts)
    materialized_chunks: list[dict[str, object]] = []
    for chunk, text, token_ids in zip(
        retained_chunks, chunk_texts, token_batches, strict=True
    ):
        ordinal = chunk.get("ordinal")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
        ):
            raise FetchError("parser chunk ordinal is invalid")
        section = _section_for_parsed_chunk(parsed, chunk)
        compact_chunk = {
            key: value
            for key, value in chunk.items()
            if key not in {"text", "canonical_text", "dedup_text"}
        }
        compact_section = None
        if section is not None:
            compact_section = {
                key: value
                for key, value in section.items()
                if key not in {"text", "dedup_text"}
            }
        materialized_chunks.append(
            {
                "ordinal": ordinal,
                "text": text,
                "token_count": len(token_ids),
                "token_sequence_sha256": hash_token_sequence(token_ids),
                "chunk": compact_chunk,
                "section": compact_section,
            }
        )
    return {
        "canonical_sha256": _sha256_bytes(canonical_text.encode("utf-8")),
        "dedup_sha256": _sha256_bytes(dedup_text.encode("utf-8")),
        "sidecar": sidecar,
        "chunks": materialized_chunks,
        "tokenizer_fingerprint": tokenizer.fingerprint,
    }


def _process_parse_member(
    raw: bytes,
    metadata: Mapping[str, object],
    max_chunk_chars: int,
    tokenizer_path: str,
) -> dict[str, object]:
    tokenizer = _PROCESS_TOKENIZERS.get(tokenizer_path)
    if tokenizer is None:
        tokenizer = ExactTokenizer(tokenizer_path)
        _PROCESS_TOKENIZERS[tokenizer_path] = tokenizer
    return _materialize_parsed_member(
        raw,
        metadata,
        max_chunk_chars=max_chunk_chars,
        parser=canonicalize_ci_log,
        tokenizer=tokenizer,
    )


_BINDING_KEYS = (
    "fetcher_script_sha256",
    "parser_script_sha256",
    "content_store_script_sha256",
)
_BINDING_UPGRADES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS binding_upgrades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    binding_key TEXT NOT NULL CHECK (
      binding_key IN (
        'fetcher_script_sha256',
        'parser_script_sha256',
        'content_store_script_sha256'
      )
    ),
    from_sha256 TEXT NOT NULL CHECK (length(from_sha256) = 64),
    to_sha256 TEXT NOT NULL CHECK (length(to_sha256) = 64),
    reason TEXT NOT NULL,
    upgraded_at TEXT NOT NULL,
    UNIQUE(binding_key,from_sha256,to_sha256)
)
"""

_STATE_SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    repo TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    run_metadata_sha256 TEXT NOT NULL,
    run_metadata_raw_size INTEGER NOT NULL,
    run_metadata_zlib BLOB NOT NULL,
    run_metadata_source TEXT NOT NULL CHECK (
      run_metadata_source IN (
        'inventory-run-list',
        'github-workflow-run-attempt-api'
      )
    ),
    run_metadata_source_attempt INTEGER NOT NULL CHECK (
      run_metadata_source_attempt >= 1
    ),
    run_metadata_exact INTEGER NOT NULL CHECK (
      run_metadata_exact IN (0,1)
    ),
    inventory_seed_attempt INTEGER NOT NULL CHECK (
      inventory_seed_attempt >= 1
    ),
    inventory_seed_metadata_sha256 TEXT NOT NULL CHECK (
      length(inventory_seed_metadata_sha256) = 64
    ),
    status TEXT NOT NULL CHECK (
      status IN (
        'pending','processing','retry','done','empty',
        'terminal_404','terminal_410','failed'
      )
    ),
    tries INTEGER NOT NULL DEFAULT 0,
    archive_source TEXT,
    archive_sha256 TEXT,
    archive_size INTEGER,
    archive_zlib BLOB,
    jobs_sha256 TEXT,
    jobs_raw_size INTEGER,
    jobs_zlib BLOB,
    member_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    occurrence_tokens INTEGER NOT NULL DEFAULT 0,
    terminal_http_status INTEGER,
    terminal_body_sha256 TEXT,
    error_class TEXT,
    error_message TEXT,
    discovered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(repo,run_id,attempt)
);
CREATE INDEX IF NOT EXISTS idx_attempts_work
ON attempts(status,created_at,repo,run_id,attempt);
CREATE TABLE IF NOT EXISTS members (
    repo TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    archive_member TEXT NOT NULL,
    job_key TEXT NOT NULL,
    raw_sha256 TEXT NOT NULL,
    raw_size INTEGER NOT NULL,
    canonical_sha256 TEXT NOT NULL,
    dedup_sha256 TEXT NOT NULL,
    sidecar_sha256 TEXT NOT NULL,
    sidecar_raw_size INTEGER NOT NULL,
    sidecar_zlib BLOB NOT NULL,
    chunk_count INTEGER NOT NULL,
    occurrence_tokens INTEGER NOT NULL,
    PRIMARY KEY(repo,run_id,attempt,archive_member),
    FOREIGN KEY(repo,run_id,attempt)
      REFERENCES attempts(repo,run_id,attempt)
);
CREATE TABLE IF NOT EXISTS request_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    requested_at TEXT NOT NULL,
    repo TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    endpoint TEXT NOT NULL,
    page_no INTEGER,
    request_attempt INTEGER NOT NULL,
    http_status INTEGER,
    outcome TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    error_class TEXT,
    error_message TEXT
);
""" + _BINDING_UPGRADES_TABLE_SQL + ";\n"


def _require_current_attempt_table_schema(
    connection: sqlite3.Connection,
) -> None:
    expected_connection = sqlite3.connect(":memory:")
    try:
        expected_connection.executescript(_STATE_SCHEMA)
        expected = [
            tuple(row)
            for row in expected_connection.execute(
                "PRAGMA table_info(attempts)"
            )
        ]
    finally:
        expected_connection.close()
    actual = [
        tuple(row)
        for row in connection.execute("PRAGMA table_info(attempts)")
    ]
    if actual != expected:
        raise BindingError(
            "fetch-state attempts schema is stale; legacy states without "
            "replayable empty-archive evidence cannot resume"
        )


def _validate_binding_upgrade_authorization(
    *,
    binding_key: str,
    source_sha256: str | None,
    reason: str | None,
    resume: bool,
) -> None:
    label = binding_key.removesuffix("_sha256").replace("_", " ")
    if source_sha256 is not None:
        if not resume:
            raise ValueError(f"{label} binding upgrade requires resume=True")
        if (
            not isinstance(source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
        ):
            raise ValueError(
                f"{label} binding upgrade source must be a lowercase SHA-256"
            )
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or reason != reason.strip()
            or len(reason) > 200
            or any(
                ord(character) < 0x20 or ord(character) == 0x7F
                for character in reason
            )
        ):
            raise ValueError(
                f"{label} binding upgrade reason must be 1-200 printable "
                "characters without surrounding whitespace"
            )
    elif reason is not None:
        raise ValueError(
            f"{label} binding upgrade reason requires an authorized source SHA-256"
        )


def _ensure_binding_upgrades_table_schema(
    connection: sqlite3.Connection,
) -> None:
    """Atomically widen either known legacy binding-upgrade ledger."""

    row = connection.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type='table' AND name='binding_upgrades'
        """
    ).fetchone()
    if row is None or not isinstance(row[0], str):
        raise BindingError("fetch-state binding-upgrade table is missing")
    table_sql = str(row[0])
    columns = tuple(
        str(item[1])
        for item in connection.execute(
            "PRAGMA table_info(binding_upgrades)"
        )
    )
    expected_columns = (
        "id",
        "binding_key",
        "from_sha256",
        "to_sha256",
        "reason",
        "upgraded_at",
    )
    if columns != expected_columns:
        raise BindingError("fetch-state binding-upgrade table is unsupported")
    if "'content_store_script_sha256'" in table_sql:
        return
    has_parser_binding = "'parser_script_sha256'" in table_sql
    required_legacy_fragments = (
        "length(from_sha256) = 64",
        "length(to_sha256) = 64",
        "UNIQUE(binding_key,from_sha256,to_sha256)",
    )
    compact_sql = " ".join(table_sql.split())
    if (
        any(fragment not in compact_sql for fragment in required_legacy_fragments)
        or "'fetcher_script_sha256'" not in compact_sql
        or (
            has_parser_binding
            and "binding_key IN" not in compact_sql
        )
        or (
            not has_parser_binding
            and "binding_key = 'fetcher_script_sha256'" not in compact_sql
        )
    ):
        raise BindingError(
            "fetch-state binding-upgrade table is not the known legacy schema"
        )
    allowed_legacy_keys = {"fetcher_script_sha256"}
    if has_parser_binding:
        allowed_legacy_keys.add("parser_script_sha256")
    stored_keys = {
        str(item[0])
        for item in connection.execute(
            "SELECT DISTINCT binding_key FROM binding_upgrades"
        )
    }
    if not stored_keys.issubset(allowed_legacy_keys):
        raise BindingError(
            "fetch-state binding-upgrade table contains unsupported keys"
        )
    unique_indexes = [
        str(index[1])
        for index in connection.execute(
            "PRAGMA index_list(binding_upgrades)"
        )
        if int(index[2]) == 1
    ]
    if len(unique_indexes) != 1:
        raise BindingError(
            "fetch-state binding-upgrade uniqueness contract is unsupported"
        )
    unique_columns = tuple(
        str(item[2])
        for item in connection.execute(
            f"PRAGMA index_info({unique_indexes[0]!r})"
        )
    )
    if unique_columns != (
        "binding_key",
        "from_sha256",
        "to_sha256",
    ):
        raise BindingError(
            "fetch-state binding-upgrade uniqueness contract is unsupported"
        )

    connection.execute(
        "ALTER TABLE binding_upgrades RENAME TO binding_upgrades_legacy"
    )
    connection.execute(_BINDING_UPGRADES_TABLE_SQL)
    connection.execute(
        """
        INSERT INTO binding_upgrades(
          id,binding_key,from_sha256,to_sha256,reason,upgraded_at
        )
        SELECT id,binding_key,from_sha256,to_sha256,reason,upgraded_at
        FROM binding_upgrades_legacy
        ORDER BY id
        """
    )
    connection.execute("DROP TABLE binding_upgrades_legacy")


def _require_bounded_fetch_state_evidence(
    connection: sqlite3.Connection,
) -> None:
    violation = fetch_state_evidence_bound_violation(connection)
    if violation is not None:
        record_type, repo, run_id, attempt, field = violation
        raise BindingError(
            "fetch-state evidence exceeds its versioned SQLite byte bounds: "
            f"{record_type} {repo}#{run_id}/{attempt} {field}"
        )


def _require_bounded_fetch_state_attempt_evidence(
    connection: sqlite3.Connection,
    key: tuple[str, int, int],
) -> None:
    repo, run_id, attempt = key
    field = fetch_state_attempt_evidence_bound_violation(
        connection,
        repo=repo,
        run_id=run_id,
        attempt=attempt,
    )
    if field is not None:
        raise BindingError(
            "fetch-state attempt evidence exceeds its versioned SQLite byte "
            f"bounds: {repo}#{run_id}/{attempt} {field}"
        )


class FetchState:
    """Durable attempt, request, and compact full-sidecar ledger."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        inventory_path: str | os.PathLike[str],
        content_store_path: str | os.PathLike[str],
        tokenizer: ExactTokenizer,
        resume: bool,
        content_store_creator_script_sha256: str | None = None,
        allow_fetcher_script_upgrade_from_sha256: str | None = None,
        fetcher_script_upgrade_reason: str | None = None,
        allow_parser_script_upgrade_from_sha256: str | None = None,
        parser_script_upgrade_reason: str | None = None,
        allow_content_store_script_upgrade_from_sha256: str | None = None,
        content_store_script_upgrade_reason: str | None = None,
        _adopted_lease_descriptor: int | None = None,
    ):
        self._lease_descriptor = -1
        if _adopted_lease_descriptor is not None:
            if (
                isinstance(_adopted_lease_descriptor, bool)
                or not isinstance(_adopted_lease_descriptor, int)
                or _adopted_lease_descriptor < 0
            ):
                raise BindingError(
                    "adopted fetch-state lease descriptor is invalid"
                )
            # Ownership transfers at the call boundary.  Record it before any
            # path or binding validation that can raise so cleanup cannot leak
            # the caller's live process lease.
            self._lease_descriptor = _adopted_lease_descriptor
        try:
            self.path, _lease_path = _fetch_state_lease_path(path)
            self._lease_path = _lease_path
            if _adopted_lease_descriptor is None:
                self._lease_descriptor = (
                    _acquire_fetch_state_process_lease(
                        self.path,
                        owner="fetch-state",
                    )
                )
            else:
                _validate_fetch_state_process_lease(
                    self._lease_descriptor,
                    state_path=self.path,
                )
            self.exhaustive_discovery_path = (
                exhaustive_discovery_sidecar_path(self.path)
            )
            self.inventory_path = (
                Path(inventory_path).expanduser().resolve()
            )
            self.content_store_path = (
                Path(content_store_path).expanduser().resolve()
            )
            content_store_binding = (
                _content_store_sha256()
                if content_store_creator_script_sha256 is None
                else content_store_creator_script_sha256
            )
            if (
                not isinstance(content_store_binding, str)
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    content_store_binding,
                )
                is None
            ):
                raise ValueError(
                    "content-store creator script binding must be a lowercase "
                    "SHA-256"
                )
            self._discovery_cursor: (
                tuple[str, str, int, int] | None
            ) = None
            self._lock = threading.RLock()
            _acquire_fetch_state_inode_lease(
                self._lease_descriptor,
                state_path=self.path,
                create=True,
            )
            _validate_fetch_state_process_lease(
                self._lease_descriptor,
                state_path=self.path,
            )
        except BaseException:
            descriptor = getattr(self, "_lease_descriptor", -1)
            if descriptor >= 0:
                self._lease_descriptor = -1
                _release_fetch_state_process_lease(descriptor)
            raise
        try:
            self._connection = sqlite3.connect(
                self.path,
                timeout=60.0,
                isolation_level=None,
                check_same_thread=False,
            )
            constrain_sqlite_evidence_rows(self._connection)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA busy_timeout=60000")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.executescript(_STATE_SCHEMA)
            _require_current_attempt_table_schema(self._connection)
            _require_bounded_fetch_state_evidence(self._connection)
        except BaseException:
            self._release_process_lease()
            raise
        expected = {
            "schema": SCHEMA_VERSION,
            "inventory_path": str(self.inventory_path),
            "content_store_path": str(self.content_store_path),
            "tokenizer_contract": _canonical_json(tokenizer.contract),
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "fetcher_script_sha256": _script_sha256(),
            "parser_script_sha256": _parser_sha256(),
            "content_store_script_sha256": content_store_binding,
            "chunk_semantics": (
                "parser-dedup-text-cppmega-training-tokenizer-"
                "payload-only-no-framing-v2"
            ),
        }
        upgrade_authorizations = {
            "fetcher_script_sha256": (
                allow_fetcher_script_upgrade_from_sha256,
                fetcher_script_upgrade_reason,
            ),
            "parser_script_sha256": (
                allow_parser_script_upgrade_from_sha256,
                parser_script_upgrade_reason,
            ),
            "content_store_script_sha256": (
                allow_content_store_script_upgrade_from_sha256,
                content_store_script_upgrade_reason,
            ),
        }
        try:
            for binding_key, (source_sha256, reason) in (
                upgrade_authorizations.items()
            ):
                _validate_binding_upgrade_authorization(
                    binding_key=binding_key,
                    source_sha256=source_sha256,
                    reason=reason,
                    resume=resume,
                )
        except BaseException:
            self._connection.close()
            self._release_process_lease()
            raise
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                current = dict(
                    self._connection.execute(
                        "SELECT key,value FROM settings"
                    ).fetchall()
                )
                if current:
                    if not resume:
                        raise BindingError(
                            f"fetch state exists at {self.path}; pass --resume"
                        )
                    script_upgrades: dict[
                        str,
                        tuple[str, str, str],
                    ] = {}
                    for binding_key in _BINDING_KEYS:
                        current_value = current.get(binding_key)
                        expected_value = expected[binding_key]
                        source_sha256, reason = upgrade_authorizations[
                            binding_key
                        ]
                        if (
                            current_value != expected_value
                            and current_value == source_sha256
                        ):
                            assert current_value is not None
                            assert reason is not None
                            script_upgrades[binding_key] = (
                                current_value,
                                expected_value,
                                reason,
                            )
                    mismatches = {
                        key: (current.get(key), value)
                        for key, value in expected.items()
                        if current.get(key) != value
                        and key not in script_upgrades
                    }
                    if mismatches:
                        rendered = ", ".join(
                            f"{key}={old!r}->{new!r}"
                            for key, (old, new) in sorted(mismatches.items())
                        )
                        raise BindingError(
                            f"fetch-state binding mismatch: {rendered}"
                        )
                    for binding_key in _BINDING_KEYS:
                        source_sha256, reason = upgrade_authorizations[
                            binding_key
                        ]
                        if (
                            source_sha256 is None
                            or binding_key in script_upgrades
                        ):
                            continue
                        replay = self._connection.execute(
                            """
                            SELECT from_sha256,to_sha256,reason
                            FROM binding_upgrades
                            WHERE binding_key=?
                            ORDER BY id DESC
                            LIMIT 1
                            """,
                            (binding_key,),
                        ).fetchone()
                        if replay is None or (
                            str(replay["from_sha256"]),
                            str(replay["to_sha256"]),
                            str(replay["reason"]),
                        ) != (
                            source_sha256,
                            expected[binding_key],
                            reason,
                        ):
                            raise BindingError(
                                f"{binding_key} upgrade authorization does not "
                                "replay the latest audited transition"
                            )
                    _ensure_binding_upgrades_table_schema(self._connection)
                    upgraded_at = _utc_now()
                    for binding_key in _BINDING_KEYS:
                        upgrade = script_upgrades.get(binding_key)
                        if upgrade is None:
                            continue
                        previous_script_sha256, next_script_sha256, reason = (
                            upgrade
                        )
                        self._connection.execute(
                            """
                            INSERT INTO binding_upgrades(
                              binding_key,from_sha256,to_sha256,
                              reason,upgraded_at
                            ) VALUES (?,?,?,?,?)
                            """,
                            (
                                binding_key,
                                previous_script_sha256,
                                next_script_sha256,
                                reason,
                                upgraded_at,
                            ),
                        )
                        self._connection.execute(
                            """
                            UPDATE settings SET value=?
                            WHERE key=?
                            """,
                            (next_script_sha256, binding_key),
                        )
                else:
                    self._connection.executemany(
                        "INSERT INTO settings(key,value) VALUES (?,?)",
                        sorted(expected.items()),
                    )
                    self._connection.execute(
                        "INSERT INTO settings(key,value) VALUES ('created_at',?)",
                        (_utc_now(),),
                    )
                if resume:
                    self._connection.execute(
                        """
                        UPDATE attempts SET status='retry',
                            error_class='InterruptedAttempt',
                            error_message='processing interrupted before closure',
                            updated_at=?
                        WHERE status='processing'
                        """,
                        (_utc_now(),),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                self._connection.close()
                self._release_process_lease()
                raise

    def close(self) -> None:
        try:
            self._connection.close()
        finally:
            self._release_process_lease()

    def _release_process_lease(self) -> None:
        descriptor = getattr(self, "_lease_descriptor", -1)
        if descriptor < 0:
            return
        self._lease_descriptor = -1
        _release_fetch_state_process_lease(descriptor)

    def _inventory_connection(self) -> sqlite3.Connection:
        if not self.inventory_path.is_file():
            raise FetchError(
                f"inventory SQLite does not exist: {self.inventory_path}"
            )
        connection = sqlite3.connect(
            f"file:{self.inventory_path}?mode=ro",
            uri=True,
            timeout=60.0,
        )
        constrain_sqlite_evidence_rows(connection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _fetch_inventory_page(
        self,
        inventory: sqlite3.Connection,
        *,
        row_limit: int,
        cursor: tuple[str, str, int, int] | None,
    ) -> list[sqlite3.Row]:
        """Read one metadata page from the snapshot that passed its BLOB check."""

        inventory.execute("BEGIN")
        try:
            if cursor is None:
                oversized = inventory.execute(
                    """
                    WITH page AS (
                      SELECT repo_key,run_id,run_attempt,
                             typeof(metadata_blob) AS metadata_type,
                             length(metadata_blob) AS compressed_bytes
                      FROM runs
                      ORDER BY created_at,repo_key,run_id,run_attempt
                      LIMIT ?
                    )
                    SELECT repo_key,run_id,run_attempt
                    FROM page
                    WHERE metadata_type!='blob' OR compressed_bytes>?
                    LIMIT 1
                    """,
                    (row_limit, MAX_RUN_METADATA_COMPRESSED_BYTES),
                ).fetchone()
                page_parameters: tuple[object, ...] = (row_limit,)
                page_query = """
                    SELECT repo_key,run_id,run_attempt,created_at,
                           metadata_blob,metadata_sha256
                    FROM runs
                    ORDER BY created_at,repo_key,run_id,run_attempt
                    LIMIT ?
                """
            else:
                created_at, repo_key, run_id, run_attempt = cursor
                oversized = inventory.execute(
                    """
                    WITH page AS (
                      SELECT repo_key,run_id,run_attempt,
                             typeof(metadata_blob) AS metadata_type,
                             length(metadata_blob) AS compressed_bytes
                      FROM runs
                      WHERE (created_at,repo_key,run_id,run_attempt)
                            > (?,?,?,?)
                      ORDER BY created_at,repo_key,run_id,run_attempt
                      LIMIT ?
                    )
                    SELECT repo_key,run_id,run_attempt
                    FROM page
                    WHERE metadata_type!='blob' OR compressed_bytes>?
                    LIMIT 1
                    """,
                    (
                        created_at,
                        repo_key,
                        run_id,
                        run_attempt,
                        row_limit,
                        MAX_RUN_METADATA_COMPRESSED_BYTES,
                    ),
                ).fetchone()
                page_parameters = (
                    created_at,
                    repo_key,
                    run_id,
                    run_attempt,
                    row_limit,
                )
                page_query = """
                    SELECT repo_key,run_id,run_attempt,created_at,
                           metadata_blob,metadata_sha256
                    FROM runs
                    WHERE (created_at,repo_key,run_id,run_attempt)
                          > (?,?,?,?)
                    ORDER BY created_at,repo_key,run_id,run_attempt
                    LIMIT ?
                """
            if oversized is not None:
                raise FetchError(
                    "inventory page metadata exceeds its versioned SQLite byte "
                    f"bound: {oversized['repo_key']}#{oversized['run_id']}/"
                    f"{oversized['run_attempt']}"
                )
            rows = inventory.execute(page_query, page_parameters).fetchall()
            inventory.execute("COMMIT")
            return rows
        except BaseException:
            if inventory.in_transaction:
                inventory.execute("ROLLBACK")
            raise

    def _insert_discovery_rows_locked(
        self,
        rows: Sequence[sqlite3.Row],
        *,
        now: str,
    ) -> int:
        inserted = 0
        for row in rows:
            blob = row["metadata_blob"]
            if not isinstance(blob, (bytes, bytearray, memoryview)):
                raise FetchError(
                    f"inventory metadata for "
                    f"{row['repo_key']}#{row['run_id']} is not a BLOB"
                )
            try:
                metadata_bytes = strict_bounded_zlib_decode(
                    blob,
                    expected_raw_size=None,
                    expected_sha256=str(row["metadata_sha256"]),
                    max_raw_size=MAX_RUN_METADATA_BYTES,
                    max_compressed_size=MAX_RUN_METADATA_COMPRESSED_BYTES,
                    where=(
                        f"inventory metadata for "
                        f"{row['repo_key']}#{row['run_id']}"
                    ),
                )
                metadata = json.loads(metadata_bytes)
            except (
                ZlibEvidenceError,
                UnicodeError,
                json.JSONDecodeError,
            ) as exc:
                raise FetchError(
                    f"corrupt inventory metadata for "
                    f"{row['repo_key']}#{row['run_id']}"
                ) from exc
            if not isinstance(metadata, dict):
                raise FetchError("inventory run metadata is not an object")
            metadata_sha = _sha256_bytes(metadata_bytes)
            if metadata_sha != str(row["metadata_sha256"]):
                raise FetchError("inventory run metadata digest mismatch")
            raw_attempt = int(row["run_attempt"])
            if raw_attempt < 1:
                raise FetchError("inventory run attempt must be positive")
            run_id = int(row["run_id"])
            _validate_run_metadata_identity(
                metadata,
                run_id=run_id,
                attempt=raw_attempt,
            )
            for attempt in range(1, raw_attempt + 1):
                exact = int(attempt == raw_attempt)
                cursor = self._connection.execute(
                    """
                            INSERT INTO attempts(
                              repo,run_id,attempt,created_at,
                              run_metadata_sha256,run_metadata_raw_size,
                              run_metadata_zlib,run_metadata_source,
                              run_metadata_source_attempt,run_metadata_exact,
                              inventory_seed_attempt,
                              inventory_seed_metadata_sha256,
                              status,discovered_at,updated_at
                            ) VALUES (
                              ?,?,?,?,?,?,?,
                              'inventory-run-list',?,?,?,?,
                              'pending',?,?
                            )
                            ON CONFLICT(repo,run_id,attempt) DO UPDATE SET
                              inventory_seed_attempt=
                                excluded.inventory_seed_attempt,
                              inventory_seed_metadata_sha256=
                                excluded.inventory_seed_metadata_sha256,
                              created_at=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.created_at
                                ELSE excluded.created_at
                              END,
                              run_metadata_sha256=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.run_metadata_sha256
                                ELSE excluded.run_metadata_sha256
                              END,
                              run_metadata_raw_size=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.run_metadata_raw_size
                                ELSE excluded.run_metadata_raw_size
                              END,
                              run_metadata_zlib=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.run_metadata_zlib
                                ELSE excluded.run_metadata_zlib
                              END,
                              run_metadata_source=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.run_metadata_source
                                ELSE excluded.run_metadata_source
                              END,
                              run_metadata_source_attempt=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN attempts.run_metadata_source_attempt
                                ELSE excluded.run_metadata_source_attempt
                              END,
                              run_metadata_exact=CASE
                                WHEN attempts.run_metadata_exact=1
                                  THEN 1
                                ELSE excluded.run_metadata_exact
                              END,
                              updated_at=excluded.updated_at
                            WHERE attempts.status IN ('pending','retry')
                    """,
                    (
                        str(row["repo_key"]),
                        run_id,
                        attempt,
                        str(row["created_at"]),
                        metadata_sha,
                        len(metadata_bytes),
                        sqlite3.Binary(zlib.compress(metadata_bytes, 6)),
                        raw_attempt,
                        exact,
                        raw_attempt,
                        metadata_sha,
                        now,
                        now,
                    ),
                )
                inserted += int(cursor.rowcount > 0)
        return inserted

    def discover(
        self,
        *,
        row_limit: int = DEFAULT_DISCOVERY_ROWS,
        exhaustive_inventory: ExhaustiveInventoryBinding | None = None,
    ) -> int:
        if row_limit <= 0:
            raise ValueError("row_limit must be positive")
        if exhaustive_inventory is not None:
            return self._discover_exhaustive(
                exhaustive_inventory,
                row_limit=row_limit,
            )
        inventory = self._inventory_connection()
        try:
            rows = self._fetch_inventory_page(
                inventory,
                row_limit=row_limit,
                cursor=self._discovery_cursor,
            )
        finally:
            inventory.close()
        if rows:
            final_row = rows[-1]
            self._discovery_cursor = (
                str(final_row["created_at"]),
                str(final_row["repo_key"]),
                int(final_row["run_id"]),
                int(final_row["run_attempt"]),
            )
        else:
            # Legacy threshold mode keeps sweeping a potentially live
            # inventory. Production exhaustive mode never resets its cursor.
            self._discovery_cursor = None
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                inserted = self._insert_discovery_rows_locked(
                    rows,
                    now=_utc_now(),
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return inserted

    def _discover_exhaustive(
        self,
        binding: ExhaustiveInventoryBinding,
        *,
        row_limit: int,
    ) -> int:
        with self._lock:
            now = _utc_now()
            sweep = load_exhaustive_discovery_sidecar(
                self.exhaustive_discovery_path
            )
            expected_binding = (
                binding.receipt_sha256,
                binding.database_sha256,
                binding.db_logical_sha256,
                binding.expected_run_count,
                binding.expected_attempt_count,
                binding.expected_attempt_set_sha256,
            )
            if sweep is None:
                sweep = {
                    "schema": EXHAUSTIVE_DISCOVERY_SCHEMA,
                    "completion_mode": (
                        COMPLETION_MODE_INVENTORY_EXHAUSTIVE
                    ),
                    "inventory_receipt_sha256": binding.receipt_sha256,
                    "inventory_database_sha256": binding.database_sha256,
                    "inventory_db_logical_sha256": (
                        binding.db_logical_sha256
                    ),
                    "expected_run_count": binding.expected_run_count,
                    "expected_attempt_count": (
                        binding.expected_attempt_count
                    ),
                    "expected_attempt_set_sha256": (
                        binding.expected_attempt_set_sha256
                    ),
                    "cursor": None,
                    "discovery_eof": False,
                    "batches": 0,
                    "rows_seen": 0,
                    "started_at": now,
                    "updated_at": now,
                }
            else:
                actual_binding = (
                    str(sweep["inventory_receipt_sha256"]),
                    str(sweep["inventory_database_sha256"]),
                    str(sweep["inventory_db_logical_sha256"]),
                    int(sweep["expected_run_count"]),
                    int(sweep["expected_attempt_count"]),
                    str(sweep["expected_attempt_set_sha256"]),
                )
                if actual_binding != expected_binding:
                    raise BindingError(
                        "persisted exhaustive discovery sweep is bound "
                        "to a different production inventory"
                    )
            if bool(sweep["discovery_eof"]):
                return 0

            cursor = sweep["cursor"]
            inventory = self._inventory_connection()
            try:
                if cursor is not None:
                    assert isinstance(cursor, list)
                    page_cursor = (
                        str(cursor[0]),
                        str(cursor[1]),
                        int(cursor[2]),
                        int(cursor[3]),
                    )
                else:
                    page_cursor = None
                rows = self._fetch_inventory_page(
                    inventory,
                    row_limit=row_limit,
                    cursor=page_cursor,
                )
            finally:
                inventory.close()

            self._connection.execute("BEGIN IMMEDIATE")
            try:
                inserted = self._insert_discovery_rows_locked(rows, now=now)
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

            next_sweep = dict(sweep)
            if rows:
                final_row = rows[-1]
                next_sweep["cursor"] = [
                    str(final_row["created_at"]),
                    str(final_row["repo_key"]),
                    int(final_row["run_id"]),
                    int(final_row["run_attempt"]),
                ]
            next_sweep["discovery_eof"] = len(rows) < row_limit
            next_sweep["batches"] = int(sweep["batches"]) + 1
            next_sweep["rows_seen"] = int(sweep["rows_seen"]) + len(rows)
            next_sweep["updated_at"] = now
            if int(next_sweep["rows_seen"]) > binding.expected_run_count:
                raise BindingError(
                    "exhaustive discovery exceeded the production inventory "
                    "run count"
                )
            if (
                bool(next_sweep["discovery_eof"])
                and int(next_sweep["rows_seen"])
                != binding.expected_run_count
            ):
                raise BindingError(
                    "exhaustive discovery EOF run count differs from the "
                    "production inventory receipt"
                )
            atomic_write_json(
                self.exhaustive_discovery_path,
                next_sweep,
            )
            return inserted

    def exhaustive_discovery_summary(self) -> dict[str, object] | None:
        with self._lock:
            sweep = load_exhaustive_discovery_sidecar(
                self.exhaustive_discovery_path
            )
        if sweep is None:
            return None
        return {
            "completion_mode": str(sweep["completion_mode"]),
            "inventory_receipt_sha256": str(
                sweep["inventory_receipt_sha256"]
            ),
            "inventory_database_sha256": str(
                sweep["inventory_database_sha256"]
            ),
            "inventory_db_logical_sha256": str(
                sweep["inventory_db_logical_sha256"]
            ),
            "expected_run_count": int(sweep["expected_run_count"]),
            "expected_attempt_count": int(sweep["expected_attempt_count"]),
            "expected_attempt_set_sha256": str(
                sweep["expected_attempt_set_sha256"]
            ),
            "discovery_eof": bool(sweep["discovery_eof"]),
            "batches": int(sweep["batches"]),
            "rows_seen": int(sweep["rows_seen"]),
        }

    def requeue_failed(self) -> int:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._connection.execute(
                    """
                    UPDATE attempts
                    SET status='retry',
                        error_class='ExplicitFailedRequeue',
                        error_message=(
                          'failed attempt explicitly reconsidered by operator'
                        ),
                        updated_at=?
                    WHERE status='failed'
                    """,
                    (_utc_now(),),
                )
                self._connection.execute("COMMIT")
                return int(cursor.rowcount)
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _decode_attempt(row: sqlite3.Row) -> Attempt:
        blob = row["run_metadata_zlib"]
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise FetchError("fetch-state run metadata is not a BLOB")
        try:
            raw = strict_bounded_zlib_decode(
                blob,
                expected_raw_size=int(row["run_metadata_raw_size"]),
                expected_sha256=str(row["run_metadata_sha256"]),
                max_raw_size=MAX_RUN_METADATA_BYTES,
                max_compressed_size=MAX_RUN_METADATA_COMPRESSED_BYTES,
                where="fetch-state run metadata",
            )
            value = json.loads(raw)
        except (
            ZlibEvidenceError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise FetchError("fetch-state run metadata is corrupt") from exc
        if not isinstance(value, dict):
            raise FetchError("fetch-state run metadata is not an object")
        digest = _sha256_bytes(raw)
        run_id = int(row["run_id"])
        attempt = int(row["attempt"])
        source = str(row["run_metadata_source"])
        source_attempt = int(row["run_metadata_source_attempt"])
        exact_raw = int(row["run_metadata_exact"])
        seed_attempt = int(row["inventory_seed_attempt"])
        seed_sha = str(row["inventory_seed_metadata_sha256"])
        if source not in _RUN_METADATA_SOURCES:
            raise FetchError("fetch-state run metadata source is invalid")
        if exact_raw not in {0, 1}:
            raise FetchError("fetch-state run metadata exactness is invalid")
        exact = bool(exact_raw)
        if seed_attempt < attempt:
            raise FetchError("inventory seed attempt precedes target attempt")
        if re.fullmatch(r"[0-9a-f]{64}", seed_sha) is None:
            raise FetchError("inventory seed metadata digest is invalid")
        if source == "inventory-run-list" and source_attempt != seed_attempt:
            raise FetchError("inventory metadata source attempt is inconsistent")
        if source == "github-workflow-run-attempt-api" and not exact:
            raise FetchError("attempt API metadata must be exact")
        if exact != (source_attempt == attempt):
            raise FetchError("fetch-state run metadata exactness is inconsistent")
        _validate_run_metadata_identity(
            value,
            run_id=run_id,
            attempt=source_attempt,
        )
        return Attempt(
            repo=str(row["repo"]),
            run_id=run_id,
            attempt=attempt,
            created_at=str(row["created_at"]),
            run_metadata=value,
            run_metadata_sha256=digest,
            run_metadata_source=source,
            run_metadata_source_attempt=source_attempt,
            run_metadata_exact=exact,
            inventory_seed_attempt=seed_attempt,
            inventory_seed_metadata_sha256=seed_sha,
        )

    def next_attempt(self, *, retry_only: bool = False) -> Attempt | None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                key_row = self._connection.execute(
                    """
                    SELECT repo,run_id,attempt FROM attempts
                    WHERE status='retry'
                    ORDER BY created_at,repo,run_id,attempt
                    LIMIT 1
                    """
                    if retry_only
                    else """
                    SELECT repo,run_id,attempt FROM attempts
                    WHERE status IN ('pending','retry')
                    ORDER BY created_at,repo,run_id,attempt
                    LIMIT 1
                    """
                ).fetchone()
                if key_row is None:
                    self._connection.execute("COMMIT")
                    return None
                key = (
                    str(key_row["repo"]),
                    int(key_row["run_id"]),
                    int(key_row["attempt"]),
                )
                _require_bounded_fetch_state_attempt_evidence(
                    self._connection,
                    key,
                )
                row = self._connection.execute(
                    """
                    SELECT repo,run_id,attempt,created_at,
                           run_metadata_sha256,run_metadata_raw_size,
                           run_metadata_zlib,run_metadata_source,
                           run_metadata_source_attempt,run_metadata_exact,
                           inventory_seed_attempt,
                           inventory_seed_metadata_sha256
                    FROM attempts
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    key,
                ).fetchone()
                if row is None:
                    raise FetchError("attempt disappeared during selection")
                self._connection.execute(
                    """
                    UPDATE attempts SET status='processing',tries=tries+1,
                      error_class=NULL,error_message=NULL,updated_at=?
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (
                        _utc_now(),
                        *key,
                    ),
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return self._decode_attempt(row)

    def bind_exact_run_metadata(
        self,
        attempt: Attempt,
        metadata: Mapping[str, object],
    ) -> Attempt:
        if attempt.run_metadata_exact:
            raise BindingError("run metadata is already exact")
        exact = dict(metadata)
        _validate_run_metadata_identity(
            exact,
            run_id=attempt.run_id,
            attempt=attempt.attempt,
        )
        seed_repository = _repository_identity(attempt)
        exact_repository = _repository_object_identity(
            exact.get("repository"),
            field="repository",
        )
        if exact_repository is None:
            raise MalformedResponseError(
                "attempt API metadata has no repository identity"
            )
        exact_name, exact_id = exact_repository
        if exact_name.casefold() != seed_repository.canonical.casefold():
            raise MalformedResponseError(
                "attempt API repository does not match inventory metadata"
            )
        if (
            exact_id is not None
            and seed_repository.repository_id is not None
            and exact_id != seed_repository.repository_id
        ):
            raise MalformedResponseError(
                "attempt API repository id does not match inventory metadata"
            )
        raw = _canonical_json_bytes(exact)
        if len(raw) > MAX_RUN_METADATA_BYTES:
            raise MalformedResponseError(
                "exact attempt metadata exceeds the versioned raw-byte limit"
            )
        digest = _sha256_bytes(raw)
        compressed = zlib.compress(raw, 6)
        if len(compressed) > MAX_RUN_METADATA_COMPRESSED_BYTES:
            raise MalformedResponseError(
                "exact attempt metadata exceeds the versioned "
                "compressed-byte limit"
            )
        now = _utc_now()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT status,run_metadata_sha256,run_metadata_exact,
                           (SELECT COUNT(*) FROM members
                            WHERE repo=attempts.repo
                              AND run_id=attempts.run_id
                              AND attempt=attempts.attempt) AS member_count
                    FROM attempts
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (attempt.repo, attempt.run_id, attempt.attempt),
                ).fetchone()
                if row is None:
                    raise BindingError("attempt disappeared before metadata bind")
                if str(row["status"]) != "processing":
                    raise BindingError(
                        "attempt metadata can bind only while processing"
                    )
                if int(row["run_metadata_exact"]) != 0:
                    raise BindingError("attempt metadata became exact concurrently")
                if str(row["run_metadata_sha256"]) != attempt.run_metadata_sha256:
                    raise BindingError("attempt seed metadata changed concurrently")
                if int(row["member_count"]) != 0:
                    raise BindingError(
                        "attempt metadata cannot change after member commits"
                    )
                self._connection.execute(
                    """
                    UPDATE attempts SET
                      created_at=?,
                      run_metadata_sha256=?,
                      run_metadata_raw_size=?,
                      run_metadata_zlib=?,
                      run_metadata_source=
                        'github-workflow-run-attempt-api',
                      run_metadata_source_attempt=?,
                      run_metadata_exact=1,
                      updated_at=?
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (
                        str(exact["created_at"]),
                        digest,
                        len(raw),
                        sqlite3.Binary(compressed),
                        attempt.attempt,
                        now,
                        attempt.repo,
                        attempt.run_id,
                        attempt.attempt,
                    ),
                )
                updated = self._connection.execute(
                    """
                    SELECT repo,run_id,attempt,created_at,
                           run_metadata_sha256,run_metadata_raw_size,
                           run_metadata_zlib,run_metadata_source,
                           run_metadata_source_attempt,run_metadata_exact,
                           inventory_seed_attempt,
                           inventory_seed_metadata_sha256
                    FROM attempts
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (attempt.repo, attempt.run_id, attempt.attempt),
                ).fetchone()
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        assert updated is not None
        return self._decode_attempt(updated)

    def record_request(
        self,
        attempt: Attempt,
        *,
        endpoint: str,
        page_no: int | None,
        request_attempt: int,
        http_status: int | None,
        outcome: str,
        latency_ms: int,
        error: BaseException | str | None = None,
        secrets: Iterable[str] = (),
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO request_ledger(
                  requested_at,repo,run_id,attempt,endpoint,page_no,
                  request_attempt,http_status,outcome,latency_ms,
                  error_class,error_message
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    _utc_now(),
                    attempt.repo,
                    attempt.run_id,
                    attempt.attempt,
                    endpoint,
                    page_no,
                    request_attempt,
                    http_status,
                    outcome,
                    latency_ms,
                    None if error is None else type(error).__name__,
                    None if error is None else _safe_error(error, secrets),
                ),
            )

    def job_rescue_audit(
        self,
        attempt: Attempt,
    ) -> Mapping[str, object] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT error_message FROM request_ledger
                WHERE repo=? AND run_id=? AND attempt=?
                  AND endpoint='operator/job_rescue'
                  AND outcome='operator/job_rescue'
                  AND error_class='JobRescueReceipt'
                ORDER BY id DESC LIMIT 1
                """,
                (attempt.repo, attempt.run_id, attempt.attempt),
            ).fetchone()
        if row is None or not isinstance(row["error_message"], str):
            return None
        raw = str(row["error_message"])
        try:
            evidence = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BindingError(
                "job-rescue ledger evidence is not JSON"
            ) from exc
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("schema")
            != JOB_RESCUE_LEDGER_EVIDENCE_SCHEMA
            or _canonical_json_bytes(evidence).decode("utf-8") != raw
        ):
            raise BindingError(
                "job-rescue ledger evidence is not canonical/current"
            )
        return dict(evidence)

    def preserved_recovery_audit(
        self,
        attempt: Attempt,
    ) -> Mapping[str, object] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT error_message FROM request_ledger
                WHERE repo=? AND run_id=? AND attempt=?
                  AND endpoint='operator/preserved_archive_recovery'
                  AND outcome='operator/preserved_archive_recovery'
                  AND error_class='PreservedArchiveRecoveryReceipt'
                ORDER BY id DESC LIMIT 1
                """,
                (attempt.repo, attempt.run_id, attempt.attempt),
            ).fetchone()
        if row is None or not isinstance(row["error_message"], str):
            return None
        raw = str(row["error_message"])
        try:
            evidence = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BindingError(
                "preserved-recovery ledger evidence is not JSON"
            ) from exc
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("schema") != PRESERVED_RECOVERY_LEDGER_SCHEMA
            or _canonical_json_bytes(evidence).decode("utf-8") != raw
        ):
            raise BindingError(
                "preserved-recovery ledger evidence is not canonical/current"
            )
        return dict(evidence)

    def archive_recovery_audit(
        self,
        attempt: Attempt,
    ) -> Mapping[str, object] | None:
        job_rescue = self.job_rescue_audit(attempt)
        preserved = self.preserved_recovery_audit(attempt)
        if job_rescue is not None and preserved is not None:
            raise BindingError(
                "attempt has conflicting archive-recovery producers"
            )
        return job_rescue if job_rescue is not None else preserved

    def store_member(
        self,
        attempt: Attempt,
        *,
        archive_member: str,
        job_key: str,
        raw_sha256: str,
        raw_size: int,
        canonical_sha256: str,
        dedup_sha256: str,
        sidecar: Mapping[str, object],
        chunk_count: int,
        occurrence_tokens: int,
    ) -> None:
        if not attempt.run_metadata_exact:
            raise BindingError(
                "cannot store a member without exact attempt metadata"
            )
        sidecar_bytes = _canonical_json_bytes(sidecar)
        sidecar_sha = _sha256_bytes(sidecar_bytes)
        compressed = zlib.compress(sidecar_bytes, 6)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                previous = self._connection.execute(
                    """
                    SELECT raw_sha256,canonical_sha256,dedup_sha256,
                           sidecar_sha256,chunk_count,occurrence_tokens
                    FROM members
                    WHERE repo=? AND run_id=? AND attempt=?
                      AND archive_member=?
                    """,
                    (
                        attempt.repo,
                        attempt.run_id,
                        attempt.attempt,
                        archive_member,
                    ),
                ).fetchone()
                identity = (
                    raw_sha256,
                    canonical_sha256,
                    dedup_sha256,
                    sidecar_sha,
                    chunk_count,
                    occurrence_tokens,
                )
                if previous is not None:
                    old = (
                        str(previous["raw_sha256"]),
                        str(previous["canonical_sha256"]),
                        str(previous["dedup_sha256"]),
                        str(previous["sidecar_sha256"]),
                        int(previous["chunk_count"]),
                        int(previous["occurrence_tokens"]),
                    )
                    if old != identity:
                        raise BindingError(
                            f"member replay changed: {archive_member}"
                        )
                else:
                    self._connection.execute(
                        """
                        INSERT INTO members(
                          repo,run_id,attempt,archive_member,job_key,
                          raw_sha256,raw_size,canonical_sha256,dedup_sha256,
                          sidecar_sha256,sidecar_raw_size,sidecar_zlib,
                          chunk_count,occurrence_tokens
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            attempt.repo,
                            attempt.run_id,
                            attempt.attempt,
                            archive_member,
                            job_key,
                            raw_sha256,
                            raw_size,
                            canonical_sha256,
                            dedup_sha256,
                            sidecar_sha,
                            len(sidecar_bytes),
                            sqlite3.Binary(compressed),
                            chunk_count,
                            occurrence_tokens,
                        ),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def replayed_member(
        self,
        attempt: Attempt,
        *,
        archive_member: str,
        job_key: str,
        raw_sha256: str,
        raw_size: int,
    ) -> tuple[int, int] | None:
        """Return durable member totals only after exact replay validation."""

        with self._lock:
            previous = self._connection.execute(
                """
                SELECT job_key,raw_sha256,raw_size,
                       chunk_count,occurrence_tokens
                FROM members
                WHERE repo=? AND run_id=? AND attempt=?
                  AND archive_member=?
                """,
                (
                    attempt.repo,
                    attempt.run_id,
                    attempt.attempt,
                    archive_member,
                ),
            ).fetchone()
        if previous is None:
            return None
        expected = (job_key, raw_sha256, raw_size)
        actual = (
            str(previous["job_key"]),
            str(previous["raw_sha256"]),
            int(previous["raw_size"]),
        )
        if actual != expected:
            raise BindingError(
                f"committed member replay changed: {archive_member}"
            )
        return (
            int(previous["chunk_count"]),
            int(previous["occurrence_tokens"]),
        )

    def fail_terminal_probe_with_durable_members(
        self,
        attempt: Attempt,
        *,
        error: TerminalHTTP,
        secrets: Iterable[str] = (),
    ) -> bool:
        """Refuse a terminal classification when this attempt already owns CAS.

        A retry can observe HTTP 404/410 after an earlier process parsed only
        part of a complete archive.  Marking that attempt terminal would hide
        its durable members from the normal terminal summary while leaving
        their CAS occurrences behind.  Keep the row explicitly failed so
        receipt finalization remains blocked until an audited archive recovery
        completes the attempt.
        """

        with self._lock, self._connection:
            durable_members = int(
                self._connection.execute(
                    """
                    SELECT COUNT(*) FROM members
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (attempt.repo, attempt.run_id, attempt.attempt),
                ).fetchone()[0]
            )
            if durable_members == 0:
                return False
            self._connection.execute(
                """
                UPDATE attempts SET
                  status='failed',
                  member_count=0,chunk_count=0,occurrence_tokens=0,
                  terminal_http_status=?,terminal_body_sha256=?,
                  error_class=?,error_message=?,updated_at=?
                WHERE repo=? AND run_id=? AND attempt=?
                """,
                (
                    error.status,
                    _sha256_bytes(error.body),
                    type(error).__name__,
                    _safe_error(error, secrets),
                    _utc_now(),
                    attempt.repo,
                    attempt.run_id,
                    attempt.attempt,
                ),
            )
        return True

    def finish_attempt(
        self,
        attempt: Attempt,
        *,
        status: str,
        archive_source: str | None = None,
        archive_sha256: str | None = None,
        archive_size: int | None = None,
        archive_bytes: bytes | None = None,
        archive_provenance: Mapping[str, object] | None = None,
        jobs: Sequence[Mapping[str, object]] | None = None,
        member_count: int = 0,
        member_uncompressed_bytes: int = 0,
        chunk_count: int = 0,
        occurrence_tokens: int = 0,
        terminal_http_status: int | None = None,
        terminal_body_sha256: str | None = None,
        error: BaseException | str | None = None,
        retry: bool = False,
        secrets: Iterable[str] = (),
    ) -> None:
        if status not in _RUN_ATTEMPT_STATES:
            raise ValueError(f"invalid attempt status {status!r}")
        if retry and status != "retry":
            raise ValueError("retry flag requires retry status")
        if status in {"done", "empty"} and not attempt.run_metadata_exact:
            raise BindingError(
                f"cannot mark {status} without exact attempt metadata"
            )
        archive_zlib: bytes | None = None
        if status == "empty":
            if (
                not isinstance(archive_source, str)
                or not archive_source
                or isinstance(archive_size, bool)
                or not isinstance(archive_size, int)
                or archive_size <= 0
                or not isinstance(archive_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
            ):
                raise BindingError(
                    "empty attempt requires exact archive identity"
                )
            if archive_bytes is not None:
                if (
                    not isinstance(archive_bytes, bytes)
                    or archive_size != len(archive_bytes)
                    or archive_sha256 != _sha256_bytes(archive_bytes)
                ):
                    raise BindingError(
                        "zero-member empty attempt requires exact replayable "
                        "archive bytes"
                    )
                _validate_empty_zip_bytes(archive_bytes)
                archive_zlib = zlib.compress(archive_bytes, 6)
            if jobs is None:
                raise BindingError(
                    "empty attempt requires durable jobs evidence"
                )
        elif archive_bytes is not None:
            raise ValueError(
                "archive_bytes is supported only for empty attempts"
            )
        jobs_bytes = (
            None if jobs is None else _canonical_json_bytes(list(jobs))
        )
        if (
            jobs_bytes is not None
            and len(jobs_bytes) > MAX_JOBS_EVIDENCE_BYTES
        ):
            raise BindingError(
                "jobs evidence exceeds the versioned raw-byte limit"
            )
        jobs_zlib = (
            None if jobs_bytes is None else zlib.compress(jobs_bytes, 6)
        )
        if (
            jobs_zlib is not None
            and len(jobs_zlib) > MAX_JOBS_EVIDENCE_COMPRESSED_BYTES
        ):
            raise BindingError(
                "jobs evidence exceeds the versioned compressed-byte limit"
            )
        jobs_sha256 = (
            None if jobs_bytes is None else _sha256_bytes(jobs_bytes)
        )
        rescue_provenance_message: str | None = None
        rescue_provenance_ledger: tuple[str, str, str] | None = None
        with self._lock, self._connection:
            if status in {"done", "empty"}:
                durable = self._connection.execute(
                    """
                    SELECT COUNT(*) AS member_count,
                           COALESCE(SUM(chunk_count),0) AS chunk_count,
                           COALESCE(SUM(occurrence_tokens),0)
                             AS occurrence_tokens
                    FROM members
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (attempt.repo, attempt.run_id, attempt.attempt),
                ).fetchone()
                if durable is None:
                    raise BindingError(
                        "completed attempt durable-member accounting is missing"
                    )
                durable_counts = (
                    int(durable["member_count"]),
                    int(durable["chunk_count"]),
                    int(durable["occurrence_tokens"]),
                )
                reported_counts = (
                    int(member_count),
                    int(chunk_count),
                    int(occurrence_tokens),
                )
                if any(
                    actual < reported
                    for actual, reported in zip(
                        durable_counts,
                        reported_counts,
                        strict=True,
                    )
                ):
                    raise BindingError(
                        "completed attempt counters exceed its durable members"
                    )
                member_count, chunk_count, occurrence_tokens = durable_counts
                if status == "empty":
                    if durable_counts[1:] != (0, 0):
                        raise BindingError(
                            "empty attempt retains training chunks or tokens"
                        )
                    if durable_counts[0] == 0:
                        if archive_zlib is None:
                            raise BindingError(
                                "zero-member empty attempt requires replayable "
                                "empty ZIP evidence"
                            )
                    else:
                        if archive_zlib is not None:
                            raise BindingError(
                                "parsed-empty attempt must not retain "
                                "zero-member ZIP evidence"
                            )
                        nonempty_member = self._connection.execute(
                            """
                            SELECT archive_member FROM members
                            WHERE repo=? AND run_id=? AND attempt=?
                              AND (chunk_count!=0 OR occurrence_tokens!=0)
                            LIMIT 1
                            """,
                            (
                                attempt.repo,
                                attempt.run_id,
                                attempt.attempt,
                            ),
                        ).fetchone()
                        if nonempty_member is not None:
                            raise BindingError(
                                "parsed-empty attempt has a nonempty member"
                            )
                if status == "done" and (
                    durable_counts[0] < 1
                    or durable_counts[1] < 1
                    or durable_counts[2] <= 0
                ):
                    raise BindingError(
                        "done attempt requires positive durable member, "
                        "chunk, and token evidence"
                    )
            if status == "done" and (
                not isinstance(archive_source, str)
                or not archive_source
                or not isinstance(archive_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
                or isinstance(archive_size, bool)
                or not isinstance(archive_size, int)
                or archive_size <= 0
                or jobs is None
            ):
                raise BindingError(
                    "done attempt requires exact archive identity and jobs "
                    "evidence"
                )
            rescue_authorized = False
            if (
                status in {"done", "empty"}
                and archive_source == "rescue-spool"
                and isinstance(archive_sha256, str)
                and isinstance(archive_size, int)
                and jobs_bytes is not None
                and jobs_sha256 is not None
            ):
                durable_members = {
                    str(row["archive_member"]): (
                        str(row["raw_sha256"]),
                        int(row["raw_size"]),
                    )
                    for row in self._connection.execute(
                        """
                        SELECT archive_member,raw_sha256,raw_size
                        FROM members
                        WHERE repo=? AND run_id=? AND attempt=?
                        ORDER BY archive_member
                        """,
                        (
                            attempt.repo,
                            attempt.run_id,
                            attempt.attempt,
                        ),
                    )
                }
                (
                    receipt_sha256,
                    source_row_sha256,
                ) = (
                    _validate_rescue_archive_provenance(
                        archive_provenance,
                        repo=attempt.repo,
                        canonical_repo=_repository_identity(
                            attempt
                        ).canonical,
                        run_id=attempt.run_id,
                        attempt=attempt.attempt,
                        created_at=attempt.created_at,
                        run_metadata_sha256=(
                            attempt.run_metadata_sha256
                        ),
                        run_metadata_raw_size=len(
                            _canonical_json_bytes(attempt.run_metadata)
                        ),
                        archive_sha256=archive_sha256,
                        archive_size=archive_size,
                        jobs_sha256=jobs_sha256,
                        jobs_raw_size=len(jobs_bytes),
                        job_count=len(jobs or ()),
                        member_count=member_count,
                        member_uncompressed_bytes=(
                            member_uncompressed_bytes
                        ),
                        jobs=jobs,
                        durable_members=durable_members,
                    )
                )
                audit = self.job_rescue_audit(attempt)
                receipt_evidence = (
                    archive_provenance.get("job_rescue_receipt")
                    if isinstance(archive_provenance, Mapping)
                    else None
                )
                receipt = (
                    receipt_evidence.get("receipt")
                    if isinstance(receipt_evidence, Mapping)
                    else None
                )
                source_state = (
                    receipt.get("source_state")
                    if isinstance(receipt, Mapping)
                    else None
                )
                try:
                    if not isinstance(source_state, Mapping):
                        raise BindingError(
                            "rescue receipt source state is missing"
                        )
                    assert isinstance(receipt, Mapping)
                    receipt_producer_binding = (
                        _job_rescue_receipt_producer_binding(receipt)
                    )
                    _validate_job_rescue_operator_audit(
                        audit,
                        receipt_sha256=receipt_sha256,
                        source_row_sha256=source_row_sha256,
                        source_state=source_state,
                        archive_sha256=archive_sha256,
                        archive_size=archive_size,
                        receipt_producer_binding=(
                            receipt_producer_binding
                        ),
                    )
                except BindingError as exc:
                    raise BindingError(
                        "rescue archive is not bound to operator ledger "
                        "evidence"
                    ) from exc
                rescue_provenance_bytes = _canonical_json_bytes(
                    archive_provenance
                )
                if (
                    len(rescue_provenance_bytes)
                    > MAX_STATE_JSON_EVIDENCE_BYTES
                ):
                    raise BindingError(
                        "rescue archive provenance exceeds its byte bound"
                    )
                rescue_provenance_message = (
                    rescue_provenance_bytes.decode("utf-8")
                )
                rescue_provenance_ledger = (
                    "operator/job_rescue",
                    "rescue_archive_consumed",
                    "RescueArchiveProvenance",
                )
                rescue_authorized = True
            elif (
                status in {"done", "empty"}
                and archive_source == "preserved-local-archive"
                and isinstance(archive_sha256, str)
                and isinstance(archive_size, int)
                and jobs_bytes is not None
            ):
                (
                    recovery_id,
                    source_row_sha256,
                    witness_set_sha256,
                    witnesses,
                    receipt_name,
                    receipt_bytes,
                    receipt_sha256,
                ) = _validate_preserved_archive_provenance(
                    archive_provenance,
                    repo=attempt.repo,
                    run_id=attempt.run_id,
                    attempt=attempt.attempt,
                    created_at=attempt.created_at,
                    archive_sha256=archive_sha256,
                    archive_size=archive_size,
                )
                audit = self.preserved_recovery_audit(attempt)
                recovery_receipt_evidence = (
                    archive_provenance.get("recovery_receipt")
                    if isinstance(archive_provenance, Mapping)
                    else None
                )
                recovery_receipt = (
                    recovery_receipt_evidence.get("receipt")
                    if isinstance(
                        recovery_receipt_evidence,
                        Mapping,
                    )
                    else None
                )
                if not isinstance(recovery_receipt, Mapping):
                    raise BindingError(
                        "preserved-recovery receipt is missing"
                    )
                artifact_producer_binding = (
                    _preserved_recovery_receipt_producer_binding(
                        recovery_receipt
                    )
                )
                _validate_preserved_recovery_operator_audit(
                    audit,
                    recovery_id=recovery_id,
                    receipt_name=receipt_name,
                    receipt_bytes=receipt_bytes,
                    receipt_sha256=receipt_sha256,
                    source_row_sha256=source_row_sha256,
                    witness_set_sha256=witness_set_sha256,
                    archive_sha256=archive_sha256,
                    archive_size=archive_size,
                    artifact_producer_binding=(
                        artifact_producer_binding
                    ),
                )
                for witness in witnesses:
                    row = self._connection.execute(
                        """
                        SELECT job_key,raw_sha256,raw_size,
                               chunk_count,occurrence_tokens
                        FROM members
                        WHERE repo=? AND run_id=? AND attempt=?
                          AND archive_member=?
                        """,
                        (
                            attempt.repo,
                            attempt.run_id,
                            attempt.attempt,
                            witness["archive_member"],
                        ),
                    ).fetchone()
                    if row is None or tuple(row) != (
                        witness["job_key"],
                        witness["raw_sha256"],
                        witness["raw_size"],
                        witness["chunk_count"],
                        witness["occurrence_tokens"],
                    ):
                        raise BindingError(
                            "preserved-recovery durable member witness "
                            "changed"
                        )
                rescue_provenance_bytes = _canonical_json_bytes(
                    archive_provenance
                )
                if (
                    len(rescue_provenance_bytes)
                    > MAX_STATE_JSON_EVIDENCE_BYTES
                ):
                    raise BindingError(
                        "preserved archive provenance exceeds its byte bound"
                    )
                rescue_provenance_message = (
                    rescue_provenance_bytes.decode("utf-8")
                )
                rescue_provenance_ledger = (
                    "operator/preserved_archive_recovery",
                    "preserved_archive_consumed",
                    "PreservedArchiveProvenance",
                )
                rescue_authorized = True
            elif archive_provenance is not None:
                raise BindingError(
                    "archive provenance is allowed only for an authorized "
                    "completed job-rescue attempt"
                )
            if status in {"done", "empty"}:
                repository = _repository_identity(attempt).canonical
                logs_endpoint = (
                    f"/repos/{repository}/actions/runs/{attempt.run_id}/"
                    f"attempts/{attempt.attempt}/logs"
                )
                jobs_endpoint = (
                    f"/repos/{repository}/actions/runs/{attempt.run_id}/"
                    f"attempts/{attempt.attempt}/jobs"
                )
                logs_success = self._connection.execute(
                    """
                    SELECT 1 FROM request_ledger
                    WHERE repo=? AND run_id=? AND attempt=?
                      AND endpoint=? AND http_status IN (200,302)
                      AND outcome='success'
                    LIMIT 1
                    """,
                    (
                        attempt.repo,
                        attempt.run_id,
                        attempt.attempt,
                        logs_endpoint,
                    ),
                ).fetchone()
                jobs_success = self._connection.execute(
                    """
                    SELECT 1 FROM request_ledger
                    WHERE repo=? AND run_id=? AND attempt=?
                      AND endpoint=? AND page_no=1
                      AND http_status=200 AND outcome='success'
                    LIMIT 1
                    """,
                    (
                        attempt.repo,
                        attempt.run_id,
                        attempt.attempt,
                        jobs_endpoint,
                    ),
                ).fetchone()
                if jobs_success is None or (
                    logs_success is None and not rescue_authorized
                ):
                    raise BindingError(
                        "completed attempt requires successful jobs plus "
                        "logs or exact archive-recovery evidence"
                    )
            if rescue_provenance_message is not None:
                assert rescue_provenance_ledger is not None
                self._connection.execute(
                    """
                    INSERT INTO request_ledger(
                      requested_at,repo,run_id,attempt,endpoint,page_no,
                      request_attempt,http_status,outcome,latency_ms,
                      error_class,error_message
                    ) VALUES (?,?,?,?,?,NULL,1,NULL,?,0,?,?)
                    """,
                    (
                        _utc_now(),
                        attempt.repo,
                        attempt.run_id,
                        attempt.attempt,
                        rescue_provenance_ledger[0],
                        rescue_provenance_ledger[1],
                        rescue_provenance_ledger[2],
                        rescue_provenance_message,
                    ),
                )
            self._connection.execute(
                """
                UPDATE attempts SET
                  status=?,archive_source=?,archive_sha256=?,archive_size=?,
                  archive_zlib=?,
                  jobs_sha256=?,jobs_raw_size=?,jobs_zlib=?,
                  member_count=?,chunk_count=?,occurrence_tokens=?,
                  terminal_http_status=?,terminal_body_sha256=?,
                  error_class=?,error_message=?,updated_at=?
                WHERE repo=? AND run_id=? AND attempt=?
                """,
                (
                    status,
                    archive_source,
                    archive_sha256,
                    archive_size,
                    (
                        None
                        if archive_zlib is None
                        else sqlite3.Binary(archive_zlib)
                    ),
                    jobs_sha256,
                    None if jobs_bytes is None else len(jobs_bytes),
                    None
                    if jobs_bytes is None
                    else sqlite3.Binary(jobs_zlib),
                    member_count,
                    chunk_count,
                    occurrence_tokens,
                    terminal_http_status,
                    terminal_body_sha256,
                    None if error is None else type(error).__name__,
                    None if error is None else _safe_error(error, secrets),
                    _utc_now(),
                    attempt.repo,
                    attempt.run_id,
                    attempt.attempt,
                ),
            )

    def summary(self) -> dict[str, object]:
        with self._lock:
            exhaustive_discovery = self.exhaustive_discovery_summary()
            status_counts = {
                str(row["status"]): int(row["n"])
                for row in self._connection.execute(
                    "SELECT status,COUNT(*) AS n FROM attempts GROUP BY status"
                )
            }
            totals = self._connection.execute(
                """
                SELECT COUNT(*) AS attempts,
                       COALESCE(SUM(member_count),0) AS members,
                       COALESCE(SUM(chunk_count),0) AS chunks,
                       COALESCE(SUM(occurrence_tokens),0) AS occurrence_tokens
                FROM attempts
                WHERE status IN (
                  'done','empty','terminal_404','terminal_410'
                )
                """
            ).fetchone()
            requests = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM request_ledger"
                ).fetchone()[0]
            )
            metadata_rows = self._connection.execute(
                """
                SELECT run_metadata_source,run_metadata_exact,status,
                       COUNT(*) AS n
                FROM attempts
                GROUP BY run_metadata_source,run_metadata_exact,status
                """
            ).fetchall()
            exact_metadata = sum(
                int(row["n"])
                for row in metadata_rows
                if int(row["run_metadata_exact"]) == 1
            )
            unresolved_by_status: dict[str, int] = {}
            exact_by_source: dict[str, int] = {}
            for row in metadata_rows:
                count = int(row["n"])
                if int(row["run_metadata_exact"]) == 1:
                    source = str(row["run_metadata_source"])
                    exact_by_source[source] = (
                        exact_by_source.get(source, 0) + count
                    )
                else:
                    status = str(row["status"])
                    unresolved_by_status[status] = (
                        unresolved_by_status.get(status, 0) + count
                    )
            content_without_exact_metadata = sum(
                count
                for status, count in unresolved_by_status.items()
                if status in {"done", "empty"}
            )
            if content_without_exact_metadata:
                raise BindingError(
                    "completed content attempt lacks exact run metadata"
                )
            sidecar_digest = hashlib.sha256()
            for row in self._connection.execute(
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
                    ).encode()
                )
            binding_upgrades = [
                {
                    "binding_key": str(row["binding_key"]),
                    "from_sha256": str(row["from_sha256"]),
                    "to_sha256": str(row["to_sha256"]),
                    "reason": str(row["reason"]),
                    "upgraded_at": str(row["upgraded_at"]),
                }
                for row in self._connection.execute(
                    """
                    SELECT binding_key,from_sha256,to_sha256,
                           reason,upgraded_at
                    FROM binding_upgrades
                    ORDER BY id
                    """
                )
            ]
            return {
                "attempt_statuses": status_counts,
                "attempts_terminal": int(totals["attempts"]),
                "members": int(totals["members"]),
                "chunks": int(totals["chunks"]),
                "occurrence_tokens": int(totals["occurrence_tokens"]),
                "requests": requests,
                "sidecar_set_sha256": sidecar_digest.hexdigest(),
                "run_metadata": {
                    "exact_attempts": exact_metadata,
                    "unresolved_attempts": sum(
                        unresolved_by_status.values()
                    ),
                    "exact_by_source": dict(sorted(exact_by_source.items())),
                    "unresolved_by_status": dict(
                        sorted(unresolved_by_status.items())
                    ),
                    "content_attempts_without_exact_metadata": 0,
                },
                "binding_upgrades": binding_upgrades,
                "exhaustive_discovery": exhaustive_discovery,
            }


class GitHubAttemptClient:
    """Jobs and attempt-log client that never forwards API auth to blob URLs."""

    def __init__(
        self,
        tokens: Sequence[str],
        state: FetchState,
        *,
        requester: Callable[
            [str, str, Mapping[str, str], float], HTTPResponse
        ] = _default_no_redirect_requester,
        archive_downloader: Callable[..., tuple[int, str]] = (
            _default_archive_downloader
        ),
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_API_ATTEMPTS,
        max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.pool = TokenPool(tokens, sleeper=sleeper)
        self.state = state
        self.requester = requester
        self.archive_downloader = archive_downloader
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.max_archive_bytes = max_archive_bytes
        self.sleeper = sleeper
        self.api_base = "https://api.github.com"

    @property
    def secrets(self) -> tuple[str, ...]:
        return self.pool.secrets

    @staticmethod
    def _body_message(body: bytes) -> str:
        try:
            value = json.loads(body)
        except (UnicodeError, json.JSONDecodeError):
            return body.decode("utf-8", errors="replace")[:1000]
        if isinstance(value, dict):
            return str(value.get("message") or value)[:1000]
        return str(value)[:1000]

    def _request(
        self,
        attempt: Attempt,
        endpoint: str,
        *,
        query: Mapping[str, object] | None = None,
        page_no: int | None = None,
        accepted: set[int],
    ) -> RequestResult:
        url = f"{self.api_base}{endpoint}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        not_found_tokens: dict[str, set[str]] = {}
        for request_attempt in range(1, self.max_attempts + 1):
            token_index, token = self.pool.acquire()
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "cppmega-ci-stream-fetch/1",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            }
            started = time.monotonic()
            try:
                response = self.requester(
                    "GET", url, headers, self.timeout
                )
            except Exception as exc:
                elapsed = int((time.monotonic() - started) * 1000)
                self.state.record_request(
                    attempt,
                    endpoint=endpoint,
                    page_no=page_no,
                    request_attempt=request_attempt,
                    http_status=None,
                    outcome="transport_retry",
                    latency_ms=elapsed,
                    error=exc,
                    secrets=self.secrets,
                )
                if request_attempt == self.max_attempts:
                    raise APIError(
                        f"transport retries exhausted for {endpoint}"
                    ) from exc
                self.sleeper(min(2 ** (request_attempt - 1), 30))
                continue
            elapsed = int((time.monotonic() - started) * 1000)
            self.pool.observe(token_index, response.headers)
            lowered = {
                str(key).casefold(): str(value)
                for key, value in response.headers.items()
            }
            message = self._body_message(response.body)
            rate_limited = response.status == 429 or (
                response.status == 403
                and (
                    lowered.get("x-ratelimit-remaining") == "0"
                    or "rate limit" in message.casefold()
                    or "abuse" in message.casefold()
                )
            )
            if rate_limited:
                self.pool.rate_limited(
                    token_index,
                    response.headers,
                    secondary="secondary" in message.casefold(),
                )
                self.state.record_request(
                    attempt,
                    endpoint=endpoint,
                    page_no=page_no,
                    request_attempt=request_attempt,
                    http_status=response.status,
                    outcome="rate_limit_retry",
                    latency_ms=elapsed,
                    error=message,
                    secrets=self.secrets,
                )
                if request_attempt == self.max_attempts:
                    raise APIError(
                        f"rate-limit retries exhausted for {endpoint}"
                    )
                continue
            if response.status >= 500:
                self.state.record_request(
                    attempt,
                    endpoint=endpoint,
                    page_no=page_no,
                    request_attempt=request_attempt,
                    http_status=response.status,
                    outcome="server_retry",
                    latency_ms=elapsed,
                    error=message,
                    secrets=self.secrets,
                )
                if request_attempt == self.max_attempts:
                    raise APIError(
                        f"server retries exhausted for {endpoint}"
                    )
                self.sleeper(min(2 ** (request_attempt - 1), 30))
                continue
            if response.status not in accepted:
                self.state.record_request(
                    attempt,
                    endpoint=endpoint,
                    page_no=page_no,
                    request_attempt=request_attempt,
                    http_status=response.status,
                    outcome="permanent_error",
                    latency_ms=elapsed,
                    error=message,
                    secrets=self.secrets,
                )
                raise APIError(
                    f"GitHub HTTP {response.status} for {endpoint}: "
                    f"{_safe_error(message, self.secrets)}"
                )
            if response.status in {404, 410}:
                body_sha256 = _sha256_bytes(response.body)
                token_sha256 = _sha256_bytes(token.encode("utf-8"))
                candidate_evidence = _canonical_json(
                    {
                        "schema": "cppmega_ci_terminal_http_candidate_v1",
                        "endpoint": endpoint,
                        "http_status": response.status,
                        "body_sha256": body_sha256,
                        "token_sha256": token_sha256,
                    }
                )
                self.state.record_request(
                    attempt,
                    endpoint=endpoint,
                    page_no=page_no,
                    request_attempt=request_attempt,
                    http_status=response.status,
                    outcome="terminal_candidate",
                    latency_ms=elapsed,
                    error=candidate_evidence,
                    secrets=self.secrets,
                )
                if response.status == 410:
                    return RequestResult(
                        status=response.status,
                        headers=response.headers,
                        body=response.body,
                    )
                corroborating_tokens = not_found_tokens.setdefault(
                    body_sha256,
                    set(),
                )
                corroborating_tokens.add(token_sha256)
                if len(corroborating_tokens) >= 2:
                    return RequestResult(
                        status=response.status,
                        headers=response.headers,
                        body=response.body,
                    )
                continue
            self.state.record_request(
                attempt,
                endpoint=endpoint,
                page_no=page_no,
                request_attempt=request_attempt,
                http_status=response.status,
                outcome="success",
                latency_ms=elapsed,
                secrets=self.secrets,
            )
            return RequestResult(
                status=response.status,
                headers=response.headers,
                body=response.body,
            )
        if not_found_tokens:
            raise APIError(
                f"uncorroborated GitHub HTTP 404 for {endpoint}; "
                "two distinct credentials are required"
            )
        raise AssertionError("unreachable request loop")

    def fetch_run_metadata(self, attempt: Attempt) -> dict[str, Any]:
        if attempt.run_metadata_exact:
            raise BindingError("exact run metadata does not need refetching")
        repository = _repository_identity(attempt).canonical
        endpoint = (
            f"/repos/{repository}/actions/runs/{attempt.run_id}/"
            f"attempts/{attempt.attempt}"
        )
        result = self._request(
            attempt,
            endpoint,
            accepted={200},
        )
        try:
            value = json.loads(result.body)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise MalformedResponseError(
                "workflow run attempt metadata is not JSON"
            ) from exc
        if not isinstance(value, dict):
            raise MalformedResponseError(
                "workflow run attempt metadata is not an object"
            )
        _validate_run_metadata_identity(
            value,
            run_id=attempt.run_id,
            attempt=attempt.attempt,
        )
        return dict(value)

    def fetch_jobs(self, attempt: Attempt) -> list[dict[str, Any]]:
        repository = _repository_identity(attempt).canonical
        endpoint = (
            f"/repos/{repository}/actions/runs/{attempt.run_id}/"
            f"attempts/{attempt.attempt}/jobs"
        )
        jobs: list[dict[str, Any]] = []
        total: int | None = None
        page = 1
        while True:
            result = self._request(
                attempt,
                endpoint,
                query={"filter": "all", "per_page": 100, "page": page},
                page_no=page,
                accepted={200},
            )
            try:
                payload = json.loads(result.body)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise MalformedResponseError(
                    f"jobs page {page} is not JSON"
                ) from exc
            if (
                not isinstance(payload, dict)
                or isinstance(payload.get("total_count"), bool)
                or not isinstance(payload.get("total_count"), int)
                or int(payload["total_count"]) < 0
                or not isinstance(payload.get("jobs"), list)
                or any(not isinstance(item, dict) for item in payload["jobs"])
            ):
                raise MalformedResponseError(
                    f"jobs page {page} has an invalid schema"
                )
            page_total = int(payload["total_count"])
            if total is None:
                total = page_total
            elif total != page_total:
                raise MalformedResponseError(
                    f"jobs total_count changed {total}->{page_total}"
                )
            page_jobs = [dict(item) for item in payload["jobs"]]
            expected_pages = max(1, math.ceil(page_total / 100))
            expected_items = (
                100
                if page < expected_pages
                else page_total - 100 * (expected_pages - 1)
            )
            if len(page_jobs) != expected_items:
                raise MalformedResponseError(
                    f"jobs page {page} has {len(page_jobs)} items, "
                    f"expected {expected_items}"
                )
            jobs.extend(page_jobs)
            if page >= expected_pages:
                break
            page += 1
        assert total is not None
        ids = []
        for job in jobs:
            job_id = job.get("id")
            if isinstance(job_id, bool) or not isinstance(job_id, int):
                raise MalformedResponseError("job id is not an integer")
            ids.append(job_id)
        if len(jobs) != total or len(set(ids)) != total:
            raise MalformedResponseError(
                "jobs enumeration is incomplete or contains duplicates"
            )
        return jobs

    def prepare_archive(self, attempt: Attempt) -> PreparedArchive:
        repository = _repository_identity(attempt).canonical
        endpoint = (
            f"/repos/{repository}/actions/runs/{attempt.run_id}/"
            f"attempts/{attempt.attempt}/logs"
        )
        result = self._request(
            attempt,
            endpoint,
            accepted={200, 302, 404, 410},
        )
        if result.status == 404:
            # GitHub can mask authorization failures as 404.  The log probe
            # is terminal only after two distinct credentials observed the
            # same body and the jobs endpoint proved attempt-level access.
            jobs = self.fetch_jobs(attempt)
            raise TerminalHTTP(
                result.status,
                result.body,
                endpoint,
                jobs=jobs,
            )
        if result.status == 410:
            raise TerminalHTTP(result.status, result.body, endpoint)
        if result.status == 200:
            if len(result.body) > self.max_archive_bytes:
                raise ArchiveError("inline archive exceeds byte limit")
            return PreparedArchive(
                repository=repository,
                run_id=attempt.run_id,
                attempt=attempt.attempt,
                source="github-inline",
                inline_body=result.body,
                signed_url=None,
            )
        location = None
        for key, value in result.headers.items():
            if str(key).casefold() == "location":
                location = str(value)
                break
        if not location:
            raise MalformedResponseError(
                "attempt-log redirect lacks Location"
            )
        return PreparedArchive(
            repository=repository,
            run_id=attempt.run_id,
            attempt=attempt.attempt,
            source="github-signed-url",
            inline_body=None,
            signed_url=location,
        )

    def fetch_archive(
        self,
        attempt: Attempt,
        destination: Path,
        *,
        prepared: PreparedArchive | None = None,
    ) -> ArchiveSource:
        archive = (
            self.prepare_archive(attempt)
            if prepared is None
            else prepared
        )
        if (
            archive.repository != _repository_identity(attempt).canonical
            or archive.run_id != attempt.run_id
            or archive.attempt != attempt.attempt
        ):
            raise BindingError(
                "prepared archive does not match the requested attempt"
            )
        if archive.source == "github-inline":
            if archive.inline_body is None or archive.signed_url is not None:
                raise BindingError("inline archive preparation is invalid")
            if len(archive.inline_body) > self.max_archive_bytes:
                raise ArchiveError("inline archive exceeds byte limit")
            with destination.open("xb") as output:
                output.write(archive.inline_body)
                output.flush()
                os.fsync(output.fileno())
            return ArchiveSource(
                path=destination,
                source="github-inline",
                raw_sha256=_sha256_bytes(archive.inline_body),
                raw_size=len(archive.inline_body),
                recoverable=False,
            )
        if (
            archive.source != "github-signed-url"
            or archive.inline_body is not None
            or archive.signed_url is None
        ):
            raise BindingError(
                "signed archive preparation is invalid"
            )
        size, digest = self.archive_downloader(
            archive.signed_url,
            destination,
            timeout=self.timeout,
            max_bytes=self.max_archive_bytes,
        )
        return ArchiveSource(
            path=destination,
            source="github-signed-url",
            raw_sha256=digest,
            raw_size=size,
            recoverable=False,
        )


def _normalized_job_name(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _member_job_hint(name: str) -> tuple[int | None, str]:
    posix = PurePosixPath(name)
    if len(posix.parts) >= 2 and posix.name.casefold() == "system.txt":
        return None, posix.parts[-2]
    match = _MAIN_MEMBER_RE.fullmatch(posix.name)
    if match:
        return int(match.group("ordinal")), match.group("name")
    return None, posix.stem


def _job_for_member(
    name: str, jobs: Sequence[Mapping[str, object]]
) -> dict[str, object] | None:
    name, duplicate_occurrence = _archive_member_name_and_occurrence(name)
    ordinal, hint = _member_job_hint(name)
    normalized_hint = _normalized_job_name(hint)
    exact = [
        dict(job)
        for job in jobs
        if _normalized_job_name(str(job.get("name") or "")) == normalized_hint
    ]
    if len(exact) == 1:
        return exact[0]
    if (
        duplicate_occurrence is not None
        and duplicate_occurrence < len(exact)
    ):
        return exact[duplicate_occurrence]
    if ordinal is not None and 0 <= ordinal < len(jobs):
        return dict(jobs[ordinal])
    return None


_DUPLICATE_MEMBER_PREFIX = "\\cppmega-duplicate-zip-member-v1:"


def _archive_member_name_and_occurrence(name: str) -> tuple[str, int | None]:
    if not name.startswith(_DUPLICATE_MEMBER_PREFIX):
        return name, None
    try:
        value = json.loads(name.removeprefix(_DUPLICATE_MEMBER_PREFIX))
    except json.JSONDecodeError as exc:
        raise ArchiveError("invalid duplicate ZIP member identity") from exc
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not isinstance(value[0], str)
        or isinstance(value[1], bool)
        or not isinstance(value[1], int)
        or value[1] < 0
    ):
        raise ArchiveError("invalid duplicate ZIP member identity")
    return value[0], value[1]


def _zip_member_identities(infos: Sequence[zipfile.ZipInfo]) -> list[str]:
    totals: dict[str, int] = {}
    for info in infos:
        totals[info.filename] = totals.get(info.filename, 0) + 1
    occurrences: dict[str, int] = {}
    identities: list[str] = []
    for info in infos:
        occurrence = occurrences.get(info.filename, 0)
        occurrences[info.filename] = occurrence + 1
        if totals[info.filename] == 1:
            identities.append(info.filename)
        else:
            identities.append(
                _DUPLICATE_MEMBER_PREFIX
                + json.dumps(
                    [info.filename, occurrence],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
    if len(set(identities)) != len(identities):
        raise ArchiveError("ZIP member identities are not unique")
    return identities


def _validated_zip_infos(
    handle: zipfile.ZipFile,
    *,
    max_members: int,
    max_member_bytes: int,
    max_uncompressed_bytes: int,
    allow_duplicate_names: bool = False,
) -> list[zipfile.ZipInfo]:
    infos = handle.infolist()
    if len(infos) > max_members:
        raise ArchiveError(
            f"ZIP member count {len(infos)} exceeds {max_members}"
        )
    names: set[str] = set()
    total = 0
    safe: list[zipfile.ZipInfo] = []
    for info in infos:
        name = info.filename
        if "\x00" in name or "\\" in name:
            raise ArchiveError(f"unsafe ZIP member name: {name!r}")
        pure = PurePosixPath(name)
        if pure.is_absolute() or any(part == ".." for part in pure.parts):
            raise ArchiveError(f"unsafe ZIP traversal member: {name!r}")
        if name in names and not allow_duplicate_names:
            raise ArchiveError(f"duplicate ZIP member: {name!r}")
        names.add(name)
        mode = (info.external_attr >> 16) & 0xFFFF
        if mode and stat.S_ISLNK(mode):
            raise ArchiveError(f"ZIP symlink member is forbidden: {name!r}")
        if info.flag_bits & 0x1:
            raise ArchiveError(f"encrypted ZIP member is forbidden: {name!r}")
        if info.file_size < 0 or info.compress_size < 0:
            raise ArchiveError("ZIP member has negative size")
        if info.file_size > max_member_bytes:
            raise ArchiveError(
                f"ZIP member {name!r} exceeds {max_member_bytes} bytes"
            )
        total += info.file_size
        if total > max_uncompressed_bytes:
            raise ArchiveError(
                "ZIP uncompressed total exceeds configured limit"
            )
        if not info.is_dir():
            safe.append(info)
    return safe


def _safe_zip_infos(
    archive: Path,
    *,
    max_members: int,
    max_member_bytes: int,
    max_uncompressed_bytes: int,
    allow_duplicate_names: bool = False,
) -> list[zipfile.ZipInfo]:
    if archive.is_symlink() or not archive.is_file():
        raise ArchiveError(f"archive path is unsafe: {archive}")
    try:
        with zipfile.ZipFile(archive) as handle:
            return _validated_zip_infos(
                handle,
                max_members=max_members,
                max_member_bytes=max_member_bytes,
                max_uncompressed_bytes=max_uncompressed_bytes,
                allow_duplicate_names=allow_duplicate_names,
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArchiveError(f"invalid ZIP archive: {exc}") from exc


def _validate_empty_zip_bytes(raw: bytes) -> None:
    if not raw or len(raw) > MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES:
        raise ArchiveError(
            "empty ZIP evidence exceeds its bounded raw-byte contract"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as handle:
            non_directories = _validated_zip_infos(
                handle,
                max_members=DEFAULT_MAX_MEMBERS,
                max_member_bytes=MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES,
                max_uncompressed_bytes=MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES,
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArchiveError(f"invalid empty ZIP evidence: {exc}") from exc
    if non_directories:
        raise ArchiveError(
            "empty ZIP evidence contains non-directory members"
        )


def _read_empty_archive_evidence(
    path: Path,
    *,
    expected_size: int,
) -> bytes:
    if expected_size > MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES:
        raise ArchiveError(
            "empty ZIP evidence exceeds its bounded raw-byte contract"
        )
    with path.open("rb") as handle:
        raw = handle.read(MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES + 1)
        trailing = handle.read(1)
    if (
        trailing
        or len(raw) > MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES
        or len(raw) != expected_size
    ):
        raise ArchiveError(
            "empty ZIP evidence size changed during bounded replay"
        )
    return raw


def _read_zip_member(
    archive: Path, info: zipfile.ZipInfo, *, max_member_bytes: int
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        with zipfile.ZipFile(archive) as handle, handle.open(info) as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > max_member_bytes or total > info.file_size:
                    raise ArchiveError(
                        f"ZIP member changed size while reading: {info.filename}"
                    )
                chunks.append(block)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ArchiveError(
            f"cannot read ZIP member {info.filename!r}: {exc}"
        ) from exc
    if total != info.file_size:
        raise ArchiveError(
            f"ZIP member {info.filename!r} is truncated "
            f"({total}!={info.file_size})"
        )
    return b"".join(chunks)


def _load_rescue_manifest(root: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    path = root / "manifest.tsv"
    if not path.exists():
        return {}
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_STATE_JSON_EVIDENCE_BYTES
    ):
        raise ArchiveError("rescue manifest path/size is unsafe")
    records: dict[tuple[str, int, int], dict[str, str]] = {}
    try:
        lines = path.read_bytes().decode("utf-8", errors="strict").splitlines()
    except UnicodeError as exc:
        raise ArchiveError("rescue manifest is not UTF-8") from exc
    if not lines:
        return {}
    fields = lines[0].split("\t")
    expected = [
        "repo",
        "run_id",
        "attempt",
        "created_at",
        "status",
        "bytes",
        "sha256",
        "finished_at",
    ]
    if fields != expected:
        raise ArchiveError("rescue manifest header is invalid")
    for line_no, line in enumerate(lines[1:], start=2):
        values = line.split("\t")
        if len(values) != len(fields):
            raise ArchiveError(
                f"rescue manifest line {line_no} has invalid field count"
            )
        record = dict(zip(fields, values))
        try:
            key = (
                record["repo"].casefold(),
                int(record["run_id"]),
                int(record["attempt"]),
            )
        except ValueError as exc:
            raise ArchiveError(
                f"rescue manifest line {line_no} has invalid identity"
            ) from exc
        previous = records.get(key)
        # The one-off rescue may have retried a failed record.  Prefer the
        # latest valid ZIP/terminal proof, otherwise retain the latest row.
        if previous is None or record["status"] in {"zip", "http410"}:
            records[key] = record
    return records


def _load_resolved_job_records(
    path: Path,
    *,
    archive_path: Path,
) -> tuple[list[dict[str, object]], list[zipfile.ZipInfo]]:
    raw = path.read_bytes()
    records: list[dict[str, object]] = []
    for line_no, line in enumerate(raw.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise ArchiveError(
                f"rescue resolved-jobs line {line_no} is not canonical"
            )
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ArchiveError(
                f"rescue resolved-jobs line {line_no} is not JSON"
            ) from exc
        if (
            not isinstance(value, dict)
            or line != _canonical_json_bytes(value) + b"\n"
        ):
            raise ArchiveError(
                f"rescue resolved-jobs line {line_no} is not canonical"
            )
        records.append(value)
    if raw and not raw.endswith(b"\n"):
        raise ArchiveError("rescue resolved-jobs evidence is truncated")
    infos = _safe_zip_infos(
        archive_path,
        max_members=DEFAULT_MAX_MEMBERS,
        max_member_bytes=DEFAULT_MAX_MEMBER_BYTES,
        max_uncompressed_bytes=DEFAULT_MAX_UNCOMPRESSED_BYTES,
    )
    infos_by_name = {info.filename: info for info in infos}
    log_records: dict[str, Mapping[str, object]] = {}
    for record in records:
        if record.get("outcome") != "log":
            continue
        member_name = record.get("member_name")
        if not isinstance(member_name, str) or member_name in log_records:
            raise ArchiveError(
                "rescue resolved-jobs has duplicate/invalid ZIP member"
            )
        log_records[member_name] = record
    if set(log_records) != set(infos_by_name):
        raise ArchiveError(
            "rescue resolved-job logs differ from synthetic ZIP members"
        )
    for member_name, record in log_records.items():
        info = infos_by_name[member_name]
        log = record.get("log")
        if (
            not isinstance(log, Mapping)
            or log.get("bytes") != info.file_size
        ):
            raise ArchiveError(
                "rescue resolved-job log size differs from synthetic ZIP"
            )
        member_raw = _read_zip_member(
            archive_path,
            info,
            max_member_bytes=DEFAULT_MAX_MEMBER_BYTES,
        )
        if log.get("sha256") != _sha256_bytes(member_raw):
            raise ArchiveError(
                "rescue resolved-job log digest differs from synthetic ZIP"
            )
    return records, infos


class RescueSpool:
    def __init__(self, root: str | os.PathLike[str] | None):
        self.root = (
            None if root is None else Path(root).expanduser().resolve()
        )
        self.manifest = (
            {} if self.root is None else _load_rescue_manifest(self.root)
        )
        manifest_path = (
            None if self.root is None else self.root / "manifest.tsv"
        )
        self.manifest_file_sha256 = (
            None
            if manifest_path is None or not manifest_path.is_file()
            else _sha256_file(manifest_path)
        )

    @staticmethod
    def _base_name(attempt: Attempt) -> str:
        return (
            f"{attempt.repo.replace('/', '__')}--{attempt.run_id}"
            f"--attempt-{attempt.attempt}"
        )

    def locate(
        self,
        attempt: Attempt,
        *,
        audit: Mapping[str, object] | None = None,
    ) -> ArchiveSource | TerminalHTTP | None:
        if self.root is None:
            return None
        key = (attempt.repo.casefold(), attempt.run_id, attempt.attempt)
        manifest = self.manifest.get(key)
        candidates: list[tuple[str, Path]] = []
        for directory in (self.root, self.root / "consumed"):
            base = directory / self._base_name(attempt)
            candidates.extend(
                [
                    ("zip", base.with_suffix(".zip")),
                    ("http410", base.with_suffix(".http410.json")),
                    ("invalid", base.with_suffix(".invalid")),
                ]
            )
        for kind, path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            size = path.stat().st_size
            digest = _sha256_file(path)
            if (
                manifest is not None
                and manifest.get("status") in {"zip", "http410"}
                and (
                    int(manifest["bytes"]) != size
                    or manifest["sha256"] != digest
                )
            ):
                raise ArchiveError(
                    f"rescue artifact digest mismatch: {path.name}"
                )
            if kind == "http410":
                return TerminalHTTP(410, path.read_bytes(), "rescue-spool")
            if kind == "invalid":
                raw = path.read_bytes()
                try:
                    payload = json.loads(raw)
                except (UnicodeError, json.JSONDecodeError):
                    payload = None
                if isinstance(payload, dict) and payload.get("status") in {
                    404,
                    410,
                }:
                    return TerminalHTTP(
                        int(payload["status"]), raw, "rescue-spool"
                    )
                if not zipfile.is_zipfile(path):
                    continue
            if kind == "zip":
                if (
                    manifest is None
                    or manifest.get("status") != "zip"
                    or manifest.get("repo", "").casefold()
                    != attempt.repo.casefold()
                    or manifest.get("run_id") != str(attempt.run_id)
                    or manifest.get("attempt") != str(attempt.attempt)
                    or manifest.get("created_at") != attempt.created_at
                    or self.manifest_file_sha256 is None
                ):
                    raise ArchiveError(
                        "rescue ZIP lacks an exact manifest binding"
                    )
                base_name = self._base_name(attempt)
                if (
                    isinstance(audit, Mapping)
                    and audit.get("schema")
                    == PRESERVED_RECOVERY_LEDGER_SCHEMA
                ):
                    audit_receipt = audit.get("receipt")
                    receipt_name = (
                        audit_receipt.get("name")
                        if isinstance(audit_receipt, Mapping)
                        else None
                    )
                    if (
                        not isinstance(receipt_name, str)
                        or Path(receipt_name).name != receipt_name
                    ):
                        raise ArchiveError(
                            "preserved-recovery receipt name is unsafe"
                        )
                    receipt_path = self.root / receipt_name
                    if (
                        receipt_path.is_symlink()
                        or not receipt_path.is_file()
                        or receipt_path.stat().st_size
                        > MAX_STATE_JSON_EVIDENCE_BYTES
                    ):
                        raise ArchiveError(
                            "preserved-recovery receipt is missing or unsafe"
                        )
                    receipt_raw = receipt_path.read_bytes()
                    try:
                        receipt = json.loads(receipt_raw)
                    except (
                        UnicodeError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise ArchiveError(
                            "preserved-recovery receipt is not JSON"
                        ) from exc
                    if (
                        not isinstance(receipt, Mapping)
                        or receipt_raw
                        != _canonical_json_bytes(receipt) + b"\n"
                    ):
                        raise ArchiveError(
                            "preserved-recovery receipt is not canonical"
                        )
                    provenance = {
                        "schema": PRESERVED_ARCHIVE_PROVENANCE_SCHEMA,
                        "manifest": {
                            "manifest_file_sha256": (
                                self.manifest_file_sha256
                            ),
                            "record": dict(manifest),
                            "record_sha256": _sha256_bytes(
                                _canonical_json_bytes(manifest)
                            ),
                        },
                        "archive": {
                            "source": "preserved-local-archive",
                            "name": path.name,
                            "bytes": size,
                            "sha256": digest,
                        },
                        "recovery_receipt": {
                            "name": receipt_path.name,
                            "bytes": len(receipt_raw),
                            "sha256": _sha256_bytes(receipt_raw),
                            "receipt": dict(receipt),
                        },
                    }
                    try:
                        (
                            recovery_id,
                            source_row_sha256,
                            witness_set_sha256,
                            witnesses,
                            validated_receipt_name,
                            validated_receipt_bytes,
                            validated_receipt_sha256,
                        ) = _validate_preserved_archive_provenance(
                            provenance,
                            repo=attempt.repo,
                            run_id=attempt.run_id,
                            attempt=attempt.attempt,
                            created_at=attempt.created_at,
                            archive_sha256=digest,
                            archive_size=size,
                        )
                        artifact_producer_binding = (
                            _preserved_recovery_receipt_producer_binding(
                                receipt
                            )
                        )
                        _validate_preserved_recovery_operator_audit(
                            audit,
                            recovery_id=recovery_id,
                            receipt_name=validated_receipt_name,
                            receipt_bytes=validated_receipt_bytes,
                            receipt_sha256=validated_receipt_sha256,
                            source_row_sha256=source_row_sha256,
                            witness_set_sha256=witness_set_sha256,
                            archive_sha256=digest,
                            archive_size=size,
                            artifact_producer_binding=(
                                artifact_producer_binding
                            ),
                        )
                        infos = _safe_zip_infos(
                            path,
                            max_members=DEFAULT_MAX_MEMBERS,
                            max_member_bytes=DEFAULT_MAX_MEMBER_BYTES,
                            max_uncompressed_bytes=(
                                DEFAULT_MAX_UNCOMPRESSED_BYTES
                            ),
                        )
                        proof = receipt["proof"]
                        assert isinstance(proof, Mapping)
                        source_archive = proof["source_archive"]
                        assert isinstance(source_archive, Mapping)
                        if (
                            source_archive.get("zip_members") != len(infos)
                            or source_archive.get("uncompressed_bytes")
                            != sum(info.file_size for info in infos)
                        ):
                            raise BindingError(
                                "preserved-recovery ZIP accounting differs"
                            )
                        infos_by_name = {
                            info.filename: info for info in infos
                        }
                        for witness in witnesses:
                            member_name = str(
                                witness["archive_member"]
                            )
                            info = infos_by_name.get(member_name)
                            if (
                                info is None
                                or info.file_size
                                != int(witness["raw_size"])
                            ):
                                raise BindingError(
                                    "preserved member witness is absent"
                                )
                            raw_member = _read_zip_member(
                                path,
                                info,
                                max_member_bytes=(
                                    DEFAULT_MAX_MEMBER_BYTES
                                ),
                            )
                            if _sha256_bytes(raw_member) != witness.get(
                                "raw_sha256"
                            ):
                                raise BindingError(
                                    "preserved member witness changed"
                                )
                    except (
                        BindingError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        raise ArchiveError(
                            "preserved archive lacks exact recovery "
                            "authorization"
                        ) from exc
                    return ArchiveSource(
                        path=path,
                        source="preserved-local-archive",
                        raw_sha256=digest,
                        raw_size=size,
                        recoverable=True,
                        provenance=provenance,
                    )
                receipt_path = self.root / f"{base_name}.receipt.json"
                resolved_path = (
                    self.root / f"{base_name}.resolved_jobs.jsonl"
                )
                for evidence_path in (receipt_path, resolved_path):
                    if (
                        evidence_path.is_symlink()
                        or not evidence_path.is_file()
                        or evidence_path.stat().st_size
                        > MAX_STATE_JSON_EVIDENCE_BYTES
                    ):
                        raise ArchiveError(
                            "rescue ZIP audit sidecar is missing or unsafe"
                        )
                receipt_raw = receipt_path.read_bytes()
                try:
                    receipt = json.loads(receipt_raw)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise ArchiveError(
                        "rescue ZIP receipt is not JSON"
                    ) from exc
                if (
                    not isinstance(receipt, Mapping)
                    or receipt_raw
                    != _canonical_json_bytes(receipt) + b"\n"
                ):
                    raise ArchiveError(
                        "rescue ZIP receipt is not canonical"
                    )
                resolved_size = resolved_path.stat().st_size
                resolved_sha256 = _sha256_file(resolved_path)
                artifacts = receipt.get("artifacts")
                resolved_receipt = (
                    artifacts.get("resolved_jobs")
                    if isinstance(artifacts, Mapping)
                    else None
                )
                if (
                    not isinstance(resolved_receipt, Mapping)
                    or resolved_receipt.get("bytes") != resolved_size
                    or resolved_receipt.get("sha256") != resolved_sha256
                ):
                    raise ArchiveError(
                        "rescue resolved-jobs artifact differs from receipt"
                    )
                resolved_records, infos = _load_resolved_job_records(
                    resolved_path,
                    archive_path=path,
                )
                provenance: Mapping[str, object] = {
                    "schema": RESCUE_ARCHIVE_PROVENANCE_SCHEMA,
                    "manifest": {
                        "manifest_file_sha256": (
                            self.manifest_file_sha256
                        ),
                        "record": dict(manifest),
                        "record_sha256": _sha256_bytes(
                            _canonical_json_bytes(manifest)
                        ),
                    },
                    "archive": {
                        "source": "rescue-spool",
                        "name": path.name,
                        "bytes": size,
                        "sha256": digest,
                    },
                    "job_rescue_receipt": {
                        "name": receipt_path.name,
                        "bytes": len(receipt_raw),
                        "sha256": _sha256_bytes(receipt_raw),
                        "receipt": dict(receipt),
                    },
                    "resolved_jobs": {
                        "name": resolved_path.name,
                        "bytes": resolved_size,
                        "sha256": resolved_sha256,
                        "records": resolved_records,
                    },
                }
                receipt_source = receipt.get("source_state")
                coverage = receipt.get("coverage")
                try:
                    if (
                        not isinstance(receipt_source, Mapping)
                        or not isinstance(coverage, Mapping)
                    ):
                        raise BindingError(
                            "rescue receipt source/coverage is missing"
                        )
                    (
                        receipt_sha256,
                        source_row_sha256,
                    ) = (
                        _validate_rescue_archive_provenance(
                            provenance,
                            repo=attempt.repo,
                            canonical_repo=_repository_identity(
                                attempt
                            ).canonical,
                            run_id=attempt.run_id,
                            attempt=attempt.attempt,
                            created_at=attempt.created_at,
                            run_metadata_sha256=(
                                attempt.run_metadata_sha256
                            ),
                            run_metadata_raw_size=len(
                                _canonical_json_bytes(
                                    attempt.run_metadata
                                )
                            ),
                            archive_sha256=digest,
                            archive_size=size,
                            jobs_sha256=str(
                                receipt_source.get("jobs_sha256")
                            ),
                            jobs_raw_size=int(
                                receipt_source.get("jobs_raw_size")
                            ),
                            job_count=int(
                                coverage.get("expected_jobs")
                            ),
                            member_count=len(infos),
                            member_uncompressed_bytes=sum(
                                info.file_size for info in infos
                            ),
                        )
                    )
                    receipt_producer_binding = (
                        _job_rescue_receipt_producer_binding(receipt)
                    )
                    _validate_job_rescue_operator_audit(
                        audit,
                        receipt_sha256=receipt_sha256,
                        source_row_sha256=source_row_sha256,
                        source_state=receipt_source,
                        archive_sha256=digest,
                        archive_size=size,
                        receipt_producer_binding=(
                            receipt_producer_binding
                        ),
                    )
                except (BindingError, TypeError, ValueError) as exc:
                    raise ArchiveError(
                        "rescue ZIP lacks exact operator-ledger authorization"
                    ) from exc
            else:
                provenance = None
            return ArchiveSource(
                path=path,
                source="rescue-spool",
                raw_sha256=digest,
                raw_size=size,
                recoverable=True,
                provenance=provenance,
            )
        return None

    def mark_consumed(self, source: ArchiveSource) -> None:
        if not source.recoverable or self.root is None:
            return
        consumed = self.root / "consumed"
        consumed.mkdir(exist_ok=True)
        if source.path.parent == consumed:
            return
        destination = consumed / source.path.name
        if destination.exists():
            if (
                destination.stat().st_size != source.raw_size
                or _sha256_file(destination) != source.raw_sha256
            ):
                raise ArchiveError(
                    f"conflicting consumed rescue archive: {destination.name}"
                )
            if source.path.exists():
                source.path.unlink()
            return
        os.replace(source.path, destination)
        _fsync_directory(consumed)
        _fsync_directory(self.root)


class CIStreamFetcher:
    def __init__(
        self,
        *,
        inventory_path: str | os.PathLike[str],
        inventory_receipt_path: str | os.PathLike[str] | None = None,
        state_path: str | os.PathLike[str],
        content_store_path: str | os.PathLike[str],
        tokenizer_path: str | os.PathLike[str],
        tokens: Sequence[str],
        progress_path: str | os.PathLike[str],
        receipt_path: str | os.PathLike[str],
        rescue_path: str | os.PathLike[str] | None = None,
        work_path: str | os.PathLike[str] | None = None,
        resume: bool = False,
        allow_fetcher_script_upgrade_from_sha256: str | None = None,
        fetcher_script_upgrade_reason: str | None = None,
        allow_parser_script_upgrade_from_sha256: str | None = None,
        parser_script_upgrade_reason: str | None = None,
        allow_content_store_script_upgrade_from_sha256: str | None = None,
        content_store_script_upgrade_reason: str | None = None,
        target_unique_tokens: int = DEFAULT_TARGET,
        completion_mode: str = COMPLETION_MODE_THRESHOLD,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
        max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
        max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
        max_members: int = DEFAULT_MAX_MEMBERS,
        parser_workers: int = 0,
        parser: Callable[..., Mapping[str, object]] = canonicalize_ci_log,
        requester: Callable[
            [str, str, Mapping[str, str], float], HTTPResponse
        ] = _default_no_redirect_requester,
        archive_downloader: Callable[..., tuple[int, str]] = (
            _default_archive_downloader
        ),
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if completion_mode not in {
            COMPLETION_MODE_THRESHOLD,
            COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        }:
            raise ValueError(f"unsupported completion_mode: {completion_mode!r}")
        if target_unique_tokens <= 0:
            raise ValueError("target_unique_tokens must be positive")
        if (
            isinstance(parser_workers, bool)
            or not isinstance(parser_workers, int)
            or parser_workers < 0
        ):
            raise ValueError("parser_workers must be a non-negative integer")
        if parser_workers and parser is not canonicalize_ci_log:
            raise ValueError(
                "parser_workers requires the canonical production parser"
            )
        self.inventory_path = Path(inventory_path).expanduser().resolve()
        self.completion_mode = completion_mode
        self.inventory_receipt_path = (
            None
            if inventory_receipt_path is None
            else Path(inventory_receipt_path).expanduser().resolve()
        )
        self.exhaustive_inventory: ExhaustiveInventoryBinding | None = None
        if completion_mode == COMPLETION_MODE_INVENTORY_EXHAUSTIVE:
            if self.inventory_receipt_path is None:
                raise ValueError(
                    "inventory-exhaustive completion requires "
                    "inventory_receipt_path"
                )
            receipt, receipt_sha256 = verify_inventory_completion_receipt(
                self.inventory_path,
                self.inventory_receipt_path,
                require_production=True,
            )
            artifact = receipt["database_artifact"]
            assert isinstance(artifact, Mapping)
            self.exhaustive_inventory = ExhaustiveInventoryBinding(
                receipt_path=self.inventory_receipt_path,
                receipt_sha256=receipt_sha256,
                database_sha256=str(artifact["sha256"]),
                db_logical_sha256=str(receipt["db_logical_sha256"]),
                expected_run_count=int(receipt["run_count"]),
                expected_attempt_count=int(
                    receipt["expected_attempt_count"]
                ),
                expected_attempt_set_sha256=str(
                    receipt["expected_attempt_set_sha256"]
                ),
            )
        self.progress_path = Path(progress_path).expanduser().resolve()
        self.receipt_path = Path(receipt_path).expanduser().resolve()
        self.state_path = _fetch_state_lease_path(state_path)[0]
        self.work_path = (
            Path(work_path).expanduser().resolve()
            if work_path is not None
            else self.state_path.with_suffix(".work")
        )
        lease_descriptor = _acquire_fetch_state_process_lease(
            self.state_path,
            owner="ci-stream-fetcher",
        )
        store: CIContentStore | None = None
        state: FetchState | None = None
        self._parser_executor: ProcessPoolExecutor | None = None
        try:
            _validate_fetch_state_process_lease(
                lease_descriptor,
                state_path=self.state_path,
            )
            self.work_path.mkdir(parents=True, exist_ok=True)
            (self.work_path / "tmp").mkdir(exist_ok=True)
            (self.work_path / "failed").mkdir(exist_ok=True)
            self.tokenizer = ExactTokenizer(tokenizer_path)
            store = CIContentStore(content_store_path)
            self.store = store
            adopted_lease_descriptor = lease_descriptor
            lease_descriptor = -1
            state = FetchState(
                self.state_path,
                inventory_path=self.inventory_path,
                content_store_path=content_store_path,
                tokenizer=self.tokenizer,
                resume=resume,
                # The store receipt reports the immutable producer binding,
                # not the hash of whichever read-only verifier opened it.
                content_store_creator_script_sha256=self.store.script_sha256,
                allow_fetcher_script_upgrade_from_sha256=(
                    allow_fetcher_script_upgrade_from_sha256
                ),
                fetcher_script_upgrade_reason=fetcher_script_upgrade_reason,
                allow_parser_script_upgrade_from_sha256=(
                    allow_parser_script_upgrade_from_sha256
                ),
                parser_script_upgrade_reason=parser_script_upgrade_reason,
                allow_content_store_script_upgrade_from_sha256=(
                    allow_content_store_script_upgrade_from_sha256
                ),
                content_store_script_upgrade_reason=(
                    content_store_script_upgrade_reason
                ),
                _adopted_lease_descriptor=adopted_lease_descriptor,
            )
            self.state = state
            self.client = GitHubAttemptClient(
                tokens,
                self.state,
                requester=requester,
                archive_downloader=archive_downloader,
                max_archive_bytes=max_archive_bytes,
                sleeper=sleeper,
            )
            self.rescue = RescueSpool(rescue_path)
            self.target_unique_tokens = target_unique_tokens
            self.max_chunk_chars = max_chunk_chars
            self.max_archive_bytes = max_archive_bytes
            self.max_member_bytes = max_member_bytes
            self.max_uncompressed_bytes = max_uncompressed_bytes
            self.max_members = max_members
            self.parser = parser
            self.parser_workers = parser_workers
            self._parser_executor = (
                ProcessPoolExecutor(
                    max_workers=parser_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
                if parser_workers
                else None
            )
            self.sleeper = sleeper
        except BaseException:
            try:
                if self._parser_executor is not None:
                    self._parser_executor.shutdown(
                        wait=True,
                        cancel_futures=False,
                    )
                    self._parser_executor = None
            finally:
                try:
                    if store is not None:
                        store.close()
                finally:
                    if state is not None:
                        state.close()
                    elif lease_descriptor >= 0:
                        _release_fetch_state_process_lease(
                            lease_descriptor
                        )
            raise

    def close(self) -> None:
        try:
            if self._parser_executor is not None:
                self._parser_executor.shutdown(
                    wait=True,
                    cancel_futures=False,
                )
                self._parser_executor = None
        finally:
            try:
                self.store.close()
            finally:
                self.state.close()

    def _temp_archive_path(self, attempt: Attempt) -> Path:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=(
                f"{attempt.repo.replace('/', '__')}--{attempt.run_id}"
                f"--{attempt.attempt}--"
            ),
            suffix=".zip.partial",
            dir=self.work_path / "tmp",
        )
        os.close(descriptor)
        path = Path(raw_path)
        # Downloaders require exclusive creation so remove only this empty,
        # freshly allocated path inside the validated temp directory.
        path.unlink()
        return path

    def _process_member(
        self,
        attempt: Attempt,
        *,
        archive: ArchiveSource,
        info: zipfile.ZipInfo,
        archive_member: str,
        jobs: Sequence[Mapping[str, object]],
    ) -> tuple[int, int]:
        if not attempt.run_metadata_exact:
            raise BindingError(
                "cannot parse a member without exact attempt metadata"
            )
        raw = _read_zip_member(
            archive.path, info, max_member_bytes=self.max_member_bytes
        )
        raw_sha = _sha256_bytes(raw)
        member_name, duplicate_occurrence = _archive_member_name_and_occurrence(
            archive_member
        )
        job = _job_for_member(archive_member, jobs)
        job_id = None if job is None else job.get("id")
        job_name = None if job is None else job.get("name")
        job_key = (
            f"{job_id if isinstance(job_id, int) else 'unresolved'}:"
            f"{archive_member}"
        )
        replayed = self.state.replayed_member(
            attempt,
            archive_member=archive_member,
            job_key=job_key,
            raw_sha256=raw_sha,
            raw_size=len(raw),
        )
        if replayed is not None:
            return replayed
        repository_identity = _repository_identity(attempt)
        metadata: dict[str, object] = dict(attempt.run_metadata)
        metadata.update(
            {
                "repository": repository_identity.canonical,
                "repository_requested": repository_identity.requested,
                "repository_id": repository_identity.repository_id,
                "source_repository": repository_identity.source,
                "source_repository_id": (
                    repository_identity.source_repository_id
                ),
                "run_id": attempt.run_id,
                "run_attempt": attempt.attempt,
                "job": job,
                "job_id": job_id,
                "job_name": job_name,
                "archive_member": member_name,
                "archive_member_identity": archive_member,
                "archive_member_duplicate_occurrence": duplicate_occurrence,
                "archive_member_raw_sha256": raw_sha,
            }
        )
        if self._parser_executor is None:
            materialized = _materialize_parsed_member(
                raw,
                metadata,
                max_chunk_chars=self.max_chunk_chars,
                parser=self.parser,
                tokenizer=self.tokenizer,
            )
        else:
            materialized = self._parser_executor.submit(
                _process_parse_member,
                raw,
                metadata,
                self.max_chunk_chars,
                str(self.tokenizer.path),
            ).result()
        sidecar = materialized.get("sidecar")
        chunks = materialized.get("chunks")
        if (
            materialized.get("tokenizer_fingerprint")
            != self.tokenizer.fingerprint
            or not isinstance(chunks, list)
            or not isinstance(sidecar, dict)
            or any(not isinstance(item, dict) for item in chunks)
        ):
            raise FetchError("materialized parser result is invalid")
        records: list[dict[str, object]] = []
        occurrence_tokens = 0
        for materialized_chunk in chunks:
            ordinal = materialized_chunk.get("ordinal")
            text = materialized_chunk.get("text")
            token_count = materialized_chunk.get("token_count")
            sequence_sha = materialized_chunk.get(
                "token_sequence_sha256"
            )
            chunk = materialized_chunk.get("chunk")
            compact_section = materialized_chunk.get("section")
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or not isinstance(text, str)
                or not text
                or isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count < 0
                or not isinstance(sequence_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", sequence_sha) is None
                or not isinstance(chunk, dict)
                or (
                    compact_section is not None
                    and not isinstance(compact_section, dict)
                )
            ):
                raise FetchError("materialized parser chunk is invalid")
            section_id = (
                str(chunk.get("section_id") or f"section:{ordinal}")
            )
            step_key = (
                f"{section_id}:"
                f"{chunk.get('step_ordinal') if chunk.get('step_ordinal') is not None else 'none'}"
            )
            provenance: dict[str, object] = {
                "schema": "cppmega_ci_chunk_occurrence_v3",
                "repository": repository_identity.canonical,
                "repository_requested": repository_identity.requested,
                "repository_id": repository_identity.repository_id,
                "source_repository": repository_identity.source,
                "source_repository_id": (
                    repository_identity.source_repository_id
                ),
                "repository_scope_key": attempt.repo,
                "run_id": attempt.run_id,
                "run_attempt": attempt.attempt,
                "run_metadata_evidence": {
                    "exact_attempt_match": attempt.run_metadata_exact,
                    "source": attempt.run_metadata_source,
                    "source_attempt": attempt.run_metadata_source_attempt,
                    "sha256": attempt.run_metadata_sha256,
                    "inventory_seed_attempt": (
                        attempt.inventory_seed_attempt
                    ),
                    "inventory_seed_metadata_sha256": (
                        attempt.inventory_seed_metadata_sha256
                    ),
                },
                "workflow": {
                    "id": attempt.run_metadata.get("workflow_id"),
                    "name": attempt.run_metadata.get("name"),
                    "path": attempt.run_metadata.get("path"),
                    "event": attempt.run_metadata.get("event"),
                    "run_number": attempt.run_metadata.get("run_number"),
                    "status": attempt.run_metadata.get("status"),
                    "conclusion": attempt.run_metadata.get("conclusion"),
                    "created_at": attempt.run_metadata.get("created_at"),
                    "updated_at": attempt.run_metadata.get("updated_at"),
                    "started_at": attempt.run_metadata.get(
                        "run_started_at"
                    ),
                    "display_title": attempt.run_metadata.get(
                        "display_title"
                    ),
                    "head_branch": attempt.run_metadata.get("head_branch"),
                    "head_sha": attempt.run_metadata.get("head_sha"),
                    "head_commit": attempt.run_metadata.get("head_commit"),
                    "actor": attempt.run_metadata.get("actor"),
                    "triggering_actor": attempt.run_metadata.get(
                        "triggering_actor"
                    ),
                },
                "job": job,
                "archive": {
                    "member": archive_member,
                    "original_member": member_name,
                    "duplicate_name_occurrence": duplicate_occurrence,
                    "member_raw_sha256": raw_sha,
                },
                "parser_sidecar_sha256": sidecar.get("sidecar_sha256"),
                "chunk": chunk,
                "section": compact_section,
            }
            records.append(
                {
                    "content": text,
                    "provenance": provenance,
                    "occurrence_key": {
                        "repo": attempt.repo,
                        "run_attempt": attempt.run_attempt_key,
                        "job": job_key,
                        "step": step_key,
                        "chunk_ordinal": ordinal,
                    },
                    "token_count": token_count,
                    "tokenizer_fingerprint": self.tokenizer.fingerprint,
                    "token_sequence_sha256": sequence_sha,
                }
            )
            occurrence_tokens += token_count
        if records:
            self.store.add_chunks(records)
        canonical_sha = materialized.get("canonical_sha256")
        dedup_sha = materialized.get("dedup_sha256")
        if (
            not isinstance(canonical_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", canonical_sha) is None
            or not isinstance(dedup_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", dedup_sha) is None
        ):
            raise FetchError("materialized member digests are invalid")
        self.state.store_member(
            attempt,
            archive_member=archive_member,
            job_key=job_key,
            raw_sha256=raw_sha,
            raw_size=len(raw),
            canonical_sha256=canonical_sha,
            dedup_sha256=dedup_sha,
            sidecar=sidecar,
            chunk_count=len(records),
            occurrence_tokens=occurrence_tokens,
        )
        return len(records), occurrence_tokens

    def process_attempt(self, attempt: Attempt) -> None:
        jobs: list[dict[str, Any]] | None = None
        archive: ArchiveSource | None = None
        temporary: Path | None = None
        try:
            rescued = self.rescue.locate(
                attempt,
                audit=self.state.archive_recovery_audit(attempt),
            )
            if isinstance(rescued, TerminalHTTP):
                raise APIError(
                    "rescue-spool terminal markers are diagnostic evidence, "
                    "not production GitHub endpoint proof"
                )
            if not attempt.run_metadata_exact:
                exact_metadata = self.client.fetch_run_metadata(attempt)
                attempt = self.state.bind_exact_run_metadata(
                    attempt,
                    exact_metadata,
                )
            if isinstance(rescued, ArchiveSource):
                archive = rescued
                jobs = self.client.fetch_jobs(attempt)
            else:
                temporary = self._temp_archive_path(attempt)
                prepared = self.client.prepare_archive(attempt)
                jobs = self.client.fetch_jobs(attempt)
                archive = self.client.fetch_archive(
                    attempt,
                    temporary,
                    prepared=prepared,
                )
            if archive.raw_size > self.max_archive_bytes:
                raise ArchiveError("archive exceeds configured byte limit")
            if archive.path.stat().st_size != archive.raw_size:
                raise ArchiveError("archive size changed before processing")
            if _sha256_file(archive.path) != archive.raw_sha256:
                raise ArchiveError("archive digest changed before processing")
            infos = _safe_zip_infos(
                archive.path,
                max_members=self.max_members,
                max_member_bytes=self.max_member_bytes,
                max_uncompressed_bytes=self.max_uncompressed_bytes,
                allow_duplicate_names=archive.source
                in {"github-inline", "github-signed-url"},
            )
            chunk_count = 0
            occurrence_tokens = 0
            for info, archive_member in zip(
                infos, _zip_member_identities(infos), strict=True
            ):
                member_chunks, member_tokens = self._process_member(
                    attempt,
                    archive=archive,
                    info=info,
                    archive_member=archive_member,
                    jobs=jobs,
                )
                chunk_count += member_chunks
                occurrence_tokens += member_tokens
            status = "done" if chunk_count else "empty"
            empty_archive_bytes = (
                _read_empty_archive_evidence(
                    archive.path,
                    expected_size=archive.raw_size,
                )
                if status == "empty" and not infos
                else None
            )
            self.state.finish_attempt(
                attempt,
                status=status,
                archive_source=archive.source,
                archive_sha256=archive.raw_sha256,
                archive_size=archive.raw_size,
                archive_bytes=empty_archive_bytes,
                archive_provenance=archive.provenance,
                jobs=jobs,
                member_count=len(infos),
                member_uncompressed_bytes=sum(
                    info.file_size for info in infos
                ),
                chunk_count=chunk_count,
                occurrence_tokens=occurrence_tokens,
                secrets=self.client.secrets,
            )
            self.rescue.mark_consumed(archive)
        except TerminalHTTP as exc:
            if exc.jobs is not None:
                jobs = exc.jobs
            if not self.state.fail_terminal_probe_with_durable_members(
                attempt,
                error=exc,
                secrets=self.client.secrets,
            ):
                status = (
                    "terminal_410" if exc.status == 410 else "terminal_404"
                )
                self.state.finish_attempt(
                    attempt,
                    status=status,
                    jobs=jobs,
                    terminal_http_status=exc.status,
                    terminal_body_sha256=_sha256_bytes(exc.body),
                    error=exc,
                    secrets=self.client.secrets,
                )
        except (APIError, ArchiveError, FetchError, OSError, zipfile.BadZipFile) as exc:
            with self.state._lock:
                tries_row = self.state._connection.execute(
                    """
                    SELECT tries FROM attempts
                    WHERE repo=? AND run_id=? AND attempt=?
                    """,
                    (attempt.repo, attempt.run_id, attempt.attempt),
                ).fetchone()
            tries = 1 if tries_row is None else int(tries_row[0])
            retry = tries < 4
            self.state.finish_attempt(
                attempt,
                status="retry" if retry else "failed",
                archive_source=None if archive is None else archive.source,
                archive_sha256=None if archive is None else archive.raw_sha256,
                archive_size=None if archive is None else archive.raw_size,
                jobs=jobs,
                error=exc,
                retry=retry,
                secrets=self.client.secrets,
            )
        finally:
            if temporary is not None and temporary.exists():
                if archive is not None:
                    failed = self.work_path / "failed" / temporary.name
                    if failed.exists():
                        failed = failed.with_name(
                            f"{failed.name}.{int(time.time())}"
                        )
                    # Preserve a failed raw archive for diagnosis.  Successful
                    # attempts were already durably committed and can discard
                    # their bounded network temporary.
                    with self.state._lock:
                        row = self.state._connection.execute(
                            """
                            SELECT status FROM attempts
                            WHERE repo=? AND run_id=? AND attempt=?
                            """,
                            (attempt.repo, attempt.run_id, attempt.attempt),
                        ).fetchone()
                    terminal = None if row is None else str(row[0])
                    if terminal in {"done", "empty"}:
                        temporary.unlink()
                    else:
                        os.replace(temporary, failed)
                        _fsync_directory(failed.parent)
                else:
                    temporary.unlink()

    def progress(self) -> dict[str, object]:
        store_status = self.store.status()
        inventory_progress = None
        candidate = self.inventory_path.with_suffix(".progress.json")
        if candidate.is_file():
            try:
                inventory_progress = json.loads(
                    candidate.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                inventory_progress = None
        return {
            "schema": PROGRESS_SCHEMA,
            "generated_at": _utc_now(),
            "inventory": (
                {"path": str(self.inventory_path)}
                if inventory_progress is None
                else inventory_progress
            ),
            "fetch": self.state.summary(),
            "content_store": store_status,
            "token_accounting": {
                "semantics": (
                    "exact unique token-id sequences over canonical "
                    "dedup payloads after cppmega training whitespace "
                    "normalization; excludes framing and padding"
                ),
                "tokenizer_contract": self.tokenizer.contract,
                "tokenizer_fingerprint": self.tokenizer.fingerprint,
            },
            "target_exact_unique_payload_tokens": self.target_unique_tokens,
            "completion_mode": self.completion_mode,
            "production_inventory_receipt": (
                None
                if self.exhaustive_inventory is None
                else {
                    "path": str(
                        self.exhaustive_inventory.receipt_path
                    ),
                    "sha256": (
                        self.exhaustive_inventory.receipt_sha256
                    ),
                }
            ),
        }

    def write_progress(self) -> dict[str, object]:
        value = self.progress()
        atomic_write_json(self.progress_path, value)
        return value

    def threshold_met(self) -> bool:
        counters = self.store.status()["counters"]
        assert isinstance(counters, dict)
        value = counters.get("exact_unique_payload_tokens")
        return value is not None and int(value) >= self.target_unique_tokens

    def exhaustive_completion_ready(self) -> bool:
        binding = self.exhaustive_inventory
        if binding is None:
            return False
        discovery = self.state.exhaustive_discovery_summary()
        if discovery is None or not bool(discovery["discovery_eof"]):
            return False
        statuses = self.state.summary()["attempt_statuses"]
        assert isinstance(statuses, Mapping)
        allowed = {"done", "empty", "terminal_404", "terminal_410"}
        return (
            set(statuses).issubset(allowed)
            and sum(int(count) for count in statuses.values())
            == binding.expected_attempt_count
        )

    def run(
        self,
        *,
        continuous: bool,
        max_runs: int | None = None,
        poll_seconds: float = 5.0,
        workers: int = 1,
    ) -> dict[str, object]:
        if workers <= 0:
            raise ValueError("workers must be positive")
        completion_mode = getattr(
            self,
            "completion_mode",
            COMPLETION_MODE_THRESHOLD,
        )
        exhaustive_inventory = getattr(
            self,
            "exhaustive_inventory",
            None,
        )
        processed = 0
        submitted = 0
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ci-stream-fetch"
        ) as executor:
            while True:
                if completion_mode == COMPLETION_MODE_INVENTORY_EXHAUSTIVE:
                    self.state.discover(
                        exhaustive_inventory=exhaustive_inventory
                    )
                else:
                    self.state.discover()
                futures: dict[Future[None], Attempt] = {}
                work_exhausted = False
                while True:
                    threshold_met = self.threshold_met()
                    while (
                        not work_exhausted
                        and len(futures) < workers
                        and (max_runs is None or submitted < max_runs)
                    ):
                        attempt = self.state.next_attempt(
                            retry_only=(
                                threshold_met
                                and completion_mode
                                == COMPLETION_MODE_THRESHOLD
                            )
                        )
                        if attempt is None:
                            work_exhausted = True
                            break
                        future = executor.submit(
                            self.process_attempt, attempt
                        )
                        futures[future] = attempt
                        submitted += 1
                    if not futures:
                        break
                    completed, _pending = wait(
                        futures,
                        timeout=max(0.1, poll_seconds),
                        return_when=FIRST_COMPLETED,
                    )
                    if not completed:
                        self.write_progress()
                        continue
                    for future in completed:
                        future.result()
                        futures.pop(future)
                        processed += 1
                        self.write_progress()
                if (
                    completion_mode == COMPLETION_MODE_THRESHOLD
                    and self.threshold_met()
                ):
                    return self.write_progress()
                if max_runs is not None and submitted >= max_runs:
                    return self.write_progress()
                if (
                    completion_mode
                    == COMPLETION_MODE_INVENTORY_EXHAUSTIVE
                ):
                    discovery = self.state.exhaustive_discovery_summary()
                    if discovery is None:
                        raise BindingError(
                            "inventory-exhaustive discovery state is missing"
                        )
                    if bool(discovery["discovery_eof"]):
                        return self.write_progress()
                    if not continuous:
                        return self.write_progress()
                    # The inventory is a verified frozen artifact. Continue
                    # immediately to the next finite discovery batch.
                    continue
                if not continuous:
                    return self.write_progress()
                self.write_progress()
                self.sleeper(max(0.1, poll_seconds))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream GitHub Actions attempt logs into the cppmega CI CAS"
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument(
        "--inventory-receipt",
        help=(
            "verified production inventory completion receipt; required by "
            "--completion-mode inventory-exhaustive"
        ),
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--content-store", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokens")
    parser.add_argument("--progress", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument(
        "--store-receipt",
        help=(
            "separate frozen content-store receipt; defaults beside --receipt"
        ),
    )
    parser.add_argument("--rescue-dir")
    parser.add_argument("--work-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-fetcher-script-upgrade-from-sha256",
        help=(
            "explicitly authorize one resume migration from this exact "
            "previous fetcher script SHA-256"
        ),
    )
    parser.add_argument(
        "--fetcher-script-upgrade-reason",
        help=(
            "required printable audit reason for an explicitly authorized "
            "fetcher script migration"
        ),
    )
    parser.add_argument(
        "--allow-parser-script-upgrade-from-sha256",
        help=(
            "explicitly authorize one resume migration from this exact "
            "previous CI sidecar parser SHA-256"
        ),
    )
    parser.add_argument(
        "--parser-script-upgrade-reason",
        help=(
            "required printable audit reason for an explicitly authorized "
            "CI sidecar parser migration"
        ),
    )
    parser.add_argument(
        "--allow-content-store-script-upgrade-from-sha256",
        help=(
            "explicitly authorize one resume migration from this exact "
            "previous content-store script SHA-256"
        ),
    )
    parser.add_argument(
        "--content-store-script-upgrade-reason",
        help=(
            "required printable audit reason for an explicitly authorized "
            "content-store script migration"
        ),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--completion-mode",
        choices=(
            COMPLETION_MODE_THRESHOLD,
            COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        ),
        default=COMPLETION_MODE_THRESHOLD,
        help=(
            "threshold is the legacy non-production token stop; "
            "inventory-exhaustive drains the exact production inventory "
            "attempt universe before a v4 receipt can be emitted"
        ),
    )
    parser.add_argument(
        "--requeue-failed",
        action="store_true",
        help=(
            "explicitly reconsider every failed attempt by moving it to retry"
        ),
    )
    parser.add_argument(
        "--continuation-seed-receipt",
        help=(
            "official clone seed receipt whose immutable base inclusion must "
            "be re-proven in the exhaustive v4 receipt"
        ),
    )
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--diagnostic-partial",
        action="store_true",
        help=(
            "allow a deliberately bounded --once/--max-runs exhaustive run "
            "to exit zero without publishing a completion receipt"
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--parser-workers",
        type=int,
        default=8,
        help=(
            "spawned CPU workers for canonicalization/tokenization; "
            "use 0 only for deterministic inline diagnostics"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--target-exact-unique-payload-tokens",
        type=int,
        default=DEFAULT_TARGET,
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
    )
    parser.add_argument(
        "--max-archive-bytes",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_BYTES,
    )
    parser.add_argument(
        "--max-member-bytes",
        type=int,
        default=DEFAULT_MAX_MEMBER_BYTES,
    )
    parser.add_argument(
        "--max-uncompressed-bytes",
        type=int,
        default=DEFAULT_MAX_UNCOMPRESSED_BYTES,
    )
    parser.add_argument("--max-members", type=int, default=DEFAULT_MAX_MEMBERS)
    return parser


def _incomplete_exhaustive_exit_code(
    *,
    diagnostic_partial: bool,
    once: bool,
    max_runs: int | None,
) -> int:
    explicitly_bounded = once or max_runs is not None
    return 0 if diagnostic_partial and explicitly_bounded else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if (
        args.completion_mode == COMPLETION_MODE_INVENTORY_EXHAUSTIVE
        and not args.inventory_receipt
    ):
        raise SystemExit(
            "--inventory-receipt is required for inventory-exhaustive mode"
        )
    if args.max_runs is not None and args.max_runs <= 0:
        raise SystemExit("--max-runs must be positive")
    if args.diagnostic_partial and not (args.once or args.max_runs is not None):
        raise SystemExit(
            "--diagnostic-partial requires an explicit --once or --max-runs "
            "bound"
        )
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.parser_workers < 0:
        raise SystemExit("--parser-workers must be non-negative")
    tokens = load_token_pool(args.tokens)
    fetcher: CIStreamFetcher | None = None
    finalize_mode: str | None = None
    exhaustive_incomplete = False
    try:
        fetcher = CIStreamFetcher(
            inventory_path=args.inventory,
            inventory_receipt_path=args.inventory_receipt,
            state_path=args.state,
            content_store_path=args.content_store,
            tokenizer_path=args.tokenizer,
            tokens=tokens,
            progress_path=args.progress,
            receipt_path=args.receipt,
            rescue_path=args.rescue_dir,
            work_path=args.work_dir,
            resume=args.resume,
            allow_fetcher_script_upgrade_from_sha256=(
                args.allow_fetcher_script_upgrade_from_sha256
            ),
            fetcher_script_upgrade_reason=(
                args.fetcher_script_upgrade_reason
            ),
            allow_parser_script_upgrade_from_sha256=(
                args.allow_parser_script_upgrade_from_sha256
            ),
            parser_script_upgrade_reason=(
                args.parser_script_upgrade_reason
            ),
            allow_content_store_script_upgrade_from_sha256=(
                args.allow_content_store_script_upgrade_from_sha256
            ),
            content_store_script_upgrade_reason=(
                args.content_store_script_upgrade_reason
            ),
            target_unique_tokens=args.target_exact_unique_payload_tokens,
            completion_mode=args.completion_mode,
            max_chunk_chars=args.max_chunk_chars,
            max_archive_bytes=args.max_archive_bytes,
            max_member_bytes=args.max_member_bytes,
            max_uncompressed_bytes=args.max_uncompressed_bytes,
            max_members=args.max_members,
            parser_workers=args.parser_workers,
        )
        if args.requeue_failed:
            fetcher.state.requeue_failed()
        result = fetcher.run(
            continuous=not args.once,
            max_runs=args.max_runs,
            poll_seconds=args.poll_seconds,
            workers=args.workers,
        )
        if args.completion_mode == COMPLETION_MODE_THRESHOLD:
            if fetcher.threshold_met():
                finalize_mode = COMPLETION_MODE_THRESHOLD
        else:
            if fetcher.exhaustive_completion_ready():
                finalize_mode = COMPLETION_MODE_INVENTORY_EXHAUSTIVE
            else:
                exhaustive_incomplete = True
    except (FetchError, sqlite3.Error, OSError, ValueError) as exc:
        print(f"[ci-stream-fetch] ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if fetcher is not None:
            fetcher.close()
    if finalize_mode is not None:
        try:
            from scripts.ci_stream_receipts import finalize_fetch_receipts

            result = finalize_fetch_receipts(
                state_path=args.state,
                content_store_path=args.content_store,
                tokenizer_path=args.tokenizer,
                target_unique_tokens=args.target_exact_unique_payload_tokens,
                fetch_receipt_path=args.receipt,
                store_receipt_path=args.store_receipt,
                original_state_path=args.state,
                original_content_store_path=args.content_store,
                original_inventory_path=args.inventory,
                completion_mode=finalize_mode,
                inventory_receipt_path=args.inventory_receipt,
                continuation_seed_receipt_path=(
                    args.continuation_seed_receipt
                ),
            )
        except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
            print(f"[ci-stream-fetch] ERROR: {exc}", file=sys.stderr)
            return 1
    elif exhaustive_incomplete:
        incomplete_exit = _incomplete_exhaustive_exit_code(
            diagnostic_partial=args.diagnostic_partial,
            once=args.once,
            max_runs=args.max_runs,
        )
        if incomplete_exit:
            print(
                "[ci-stream-fetch] ERROR: exhaustive inventory is incomplete; "
                "failed/retry/pending attempts require remediation",
                file=sys.stderr,
            )
            return incomplete_exit
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
