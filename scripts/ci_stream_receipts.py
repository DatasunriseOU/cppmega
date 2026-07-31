#!/usr/bin/env python3
"""Finalize immutable, merge-compatible CI stream receipts.

The fetch loop cannot honestly bind its SQLite artifact while its writer is
still open.  This module is deliberately pyarrow-free so the lightweight
fetch environment can close every writer first and only then create the
frozen store and fetch-state receipts consumed by the shard merger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci_content_store import (  # noqa: E402
    CIContentStore,
    _hash_records,
    _sqlite_schema_sha256,
    hash_token_sequence,
)
from scripts.ci_stream_fetch import (  # noqa: E402
    COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
    COMPLETION_MODE_THRESHOLD,
    EXHAUSTIVE_RECEIPT_SCHEMA,
    JOB_RESCUE_LEDGER_EVIDENCE_SCHEMA,
    MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES,
    PRESERVED_RECOVERY_LEDGER_SCHEMA,
    RECEIPT_SCHEMA,
    SCHEMA_VERSION,
    _STATE_SCHEMA,
    _acquire_fetch_state_process_lease,
    BindingError,
    ExactTokenizer,
    MalformedResponseError,
    _canonical_json_bytes,
    _repository_object_identity,
    _release_fetch_state_process_lease,
    _sha256_bytes,
    _sha256_file,
    _utc_now,
    _validate_empty_zip_bytes,
    _validate_job_rescue_operator_audit,
    _validate_preserved_archive_provenance,
    _validate_preserved_recovery_operator_audit,
    _validate_rescue_archive_provenance,
    atomic_write_json,
    exhaustive_discovery_sidecar_path,
    load_exhaustive_discovery_sidecar,
)
from scripts.ci_stream_inventory import (  # noqa: E402
    CompletionError as InventoryCompletionError,
    verify_inventory_completion_receipt,
)
from scripts.ci_zlib_evidence import (  # noqa: E402
    MAX_JOBS_EVIDENCE_BYTES,
    MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
    MAX_RUN_METADATA_BYTES,
    MAX_RUN_METADATA_COMPRESSED_BYTES,
    ZlibEvidenceError,
    constrain_sqlite_evidence_rows,
    content_store_evidence_bound_violation,
    fetch_state_evidence_bound_violation,
    strict_bounded_zlib_decode,
)


_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_TERMINAL_RECEIPT_STATES = (
    "done",
    "empty",
    "terminal_404",
    "terminal_410",
)


def convergent_transition_layout(
    edges: Iterable[tuple[str, str]],
    *,
    current: str,
) -> tuple[dict[str, int], dict[int, int]]:
    """Return SCC membership and distance for a graph converging on current.

    Binding rows are immutable transition evidence, not an acyclic version
    list.  A real producer can upgrade ``A -> B`` and later roll back
    ``B -> A``.  Strongly connected components preserve that evidence while
    the condensed graph must still have exactly one reachable sink: the
    component containing the current binding.
    """

    adjacency: dict[str, set[str]] = {current: set()}
    reverse_adjacency: dict[str, set[str]] = {current: set()}
    for source, destination in edges:
        adjacency.setdefault(source, set()).add(destination)
        adjacency.setdefault(destination, set())
        reverse_adjacency.setdefault(destination, set()).add(source)
        reverse_adjacency.setdefault(source, set())

    finish_order: list[str] = []
    visited: set[str] = set()
    for root in sorted(adjacency):
        if root in visited:
            continue
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish_order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            stack.extend(
                (destination, False)
                for destination in reversed(sorted(adjacency[node]))
                if destination not in visited
            )

    component_by_node: dict[str, int] = {}
    component_count = 0
    for root in reversed(finish_order):
        if root in component_by_node:
            continue
        pending = [root]
        component_by_node[root] = component_count
        while pending:
            node = pending.pop()
            for source in sorted(reverse_adjacency[node], reverse=True):
                if source in component_by_node:
                    continue
                component_by_node[source] = component_count
                pending.append(source)
        component_count += 1

    component_adjacency: dict[int, set[int]] = {
        component: set() for component in range(component_count)
    }
    reverse_component_adjacency: dict[int, set[int]] = {
        component: set() for component in range(component_count)
    }
    for source, destinations in adjacency.items():
        source_component = component_by_node[source]
        for destination in destinations:
            destination_component = component_by_node[destination]
            if source_component == destination_component:
                continue
            component_adjacency[source_component].add(destination_component)
            reverse_component_adjacency[destination_component].add(
                source_component
            )

    current_component = component_by_node[current]
    reaches_current = {current_component}
    pending_components = [current_component]
    while pending_components:
        component = pending_components.pop()
        for source_component in reverse_component_adjacency[component]:
            if source_component in reaches_current:
                continue
            reaches_current.add(source_component)
            pending_components.append(source_component)
    if len(reaches_current) != component_count:
        raise ValueError(
            "transition graph diverges or cannot reach the current binding"
        )

    distance_to_current = {current_component: 0}
    unresolved = set(component_adjacency) - {current_component}
    while unresolved:
        ready = {
            component
            for component in unresolved
            if component_adjacency[component]
            and component_adjacency[component] <= distance_to_current.keys()
        }
        if not ready:
            raise ValueError("transition component graph is not acyclic")
        for component in ready:
            distance_to_current[component] = 1 + max(
                distance_to_current[destination]
                for destination in component_adjacency[component]
            )
        unresolved -= ready

    return component_by_node, distance_to_current


_EXHAUSTIVE_TERMINAL_STATES = frozenset(_TERMINAL_RECEIPT_STATES)
_CAS_RETOKENIZE_BATCH_SIZE = 256
_CAS_RETOKENIZE_BATCH_BYTES = 4 * 1024 * 1024


class ReceiptFinalizationError(RuntimeError):
    """A mutable or inconsistent stream cannot publish a frozen receipt."""


def _require_bounded_fetch_state_evidence(
    connection: sqlite3.Connection,
) -> None:
    violation = fetch_state_evidence_bound_violation(connection)
    if violation is None:
        return
    record_type, repo, run_id, attempt, field = violation
    raise ReceiptFinalizationError(
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
    raise ReceiptFinalizationError(
        "content-store provenance is not exact and bounded by its versioned "
        "SQLite byte contract: "
        f"{repo}/{run_attempt}/{job}/{step}/{chunk_ordinal}"
    )


def _require_frozen_sqlite(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ReceiptFinalizationError(f"{label} is missing or unsafe: {path}")
    for suffix in ("-wal", "-journal"):
        pending = Path(f"{path}{suffix}")
        if pending.is_symlink() or (
            pending.exists() and pending.stat().st_size != 0
        ):
            raise ReceiptFinalizationError(
                f"{label} is not frozen; found {pending.name}"
            )
    shm = Path(f"{path}-shm")
    if shm.is_symlink() or (shm.exists() and not shm.is_file()):
        raise ReceiptFinalizationError(
            f"{label} has an unsafe SQLite sidecar: {shm.name}"
        )


def _stream_tree_artifact_set(root: Path) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise ReceiptFinalizationError(
            f"artifact tree is missing or unsafe: {root}"
        )
    digest = hashlib.sha256()
    digest.update(b"cppmega-ci-continuation-tree-v3\0")
    file_count = 0
    byte_size = 0

    def walk(directory: Path) -> Iterator[Path]:
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_symlink():
                raise ReceiptFinalizationError(
                    f"artifact tree contains a symlink: {path}"
                )
            if path.is_dir():
                yield from walk(path)
            elif path.is_file():
                yield path
            else:
                raise ReceiptFinalizationError(
                    f"artifact tree contains an unsafe file: {path}"
                )

    for path in walk(root):
        stat_before = path.stat()
        sha256 = _sha256_file(path)
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
            raise ReceiptFinalizationError(
                f"artifact changed while it was hashed: {path}"
            )
        record = _canonical_json_bytes(
            {
                "path": path.relative_to(root).as_posix(),
                "byte_size": stat_after.st_size,
                "sha256": sha256,
            }
        )
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
        file_count += 1
        byte_size += stat_after.st_size
    return {
        "file_count": file_count,
        "byte_size": byte_size,
        "artifact_set_sha256": digest.hexdigest(),
    }


def _freeze_fetch_state_sqlite(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ReceiptFinalizationError(
            f"fetch state is missing or unsafe: {path}"
        )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            path,
            isolation_level=None,
            timeout=0.25,
        )
        constrain_sqlite_evidence_rows(connection)
        connection.execute("PRAGMA busy_timeout=250")
        mode_row = connection.execute("PRAGMA journal_mode").fetchone()
        mode = "" if mode_row is None else str(mode_row[0]).lower()
        if mode == "wal":
            checkpoint = connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if (
                checkpoint is None
                or len(checkpoint) != 3
                or int(checkpoint[0]) != 0
            ):
                raise ReceiptFinalizationError(
                    f"fetch-state WAL checkpoint is busy: {checkpoint}"
                )
            mode_row = connection.execute(
                "PRAGMA journal_mode=DELETE"
            ).fetchone()
            mode = "" if mode_row is None else str(mode_row[0]).lower()
        if mode != "delete":
            raise ReceiptFinalizationError(
                f"fetch-state journal mode did not freeze: {mode!r}"
            )
    except sqlite3.Error as exc:
        raise ReceiptFinalizationError(
            f"fetch state could not be frozen: {exc}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    _require_frozen_sqlite(path, label="fetch state")


def _expected_fetch_state_schema_sha256() -> str:
    connection = sqlite3.connect(":memory:")
    try:
        connection.row_factory = sqlite3.Row
        connection.executescript(_STATE_SCHEMA)
        return _sqlite_schema_sha256(connection)
    finally:
        connection.close()


def _fetch_state_logical_digest(connection: sqlite3.Connection) -> str:
    constrain_sqlite_evidence_rows(connection)
    _require_bounded_fetch_state_evidence(connection)
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


def _canonical_summary(connection: sqlite3.Connection) -> tuple[dict[str, object], str]:
    _require_attempt_member_accounting(connection)
    statuses = {
        str(row["status"]): int(row["count"])
        for row in connection.execute(
            """
            SELECT status,COUNT(*) AS count
            FROM attempts GROUP BY status ORDER BY status
            """
        )
    }
    placeholders = ",".join("?" for _ in _TERMINAL_RECEIPT_STATES)
    totals = connection.execute(
        f"""
        SELECT COUNT(*) AS attempts,
               COALESCE(SUM(member_count),0) AS members,
               COALESCE(SUM(chunk_count),0) AS chunks,
               COALESCE(SUM(occurrence_tokens),0) AS occurrence_tokens
        FROM attempts WHERE status IN ({placeholders})
        """,
        _TERMINAL_RECEIPT_STATES,
    ).fetchone()
    member_totals = connection.execute(
        """
        SELECT COUNT(*) AS members,
               COALESCE(SUM(chunk_count),0) AS chunks,
               COALESCE(SUM(occurrence_tokens),0) AS occurrence_tokens
        FROM members
        """
    ).fetchone()
    if totals is None or member_totals is None:
        raise ReceiptFinalizationError("fetch-state accounting is missing")
    if (
        int(member_totals["members"]) != int(totals["members"])
        or int(member_totals["chunks"]) != int(totals["chunks"])
        or int(member_totals["occurrence_tokens"]) != int(totals["occurrence_tokens"])
    ):
        raise ReceiptFinalizationError(
            "fetch-state attempt/member accounting is inconsistent"
        )

    sidecar_digest = hashlib.sha256()
    for row in connection.execute(
        """
        SELECT repo,run_id,attempt,archive_member,sidecar_sha256
        FROM members ORDER BY repo,run_id,attempt,archive_member
        """
    ):
        sidecar_digest.update(
            (
                f"{row['repo']}\t{row['run_id']}\t{row['attempt']}\t"
                f"{row['archive_member']}\t{row['sidecar_sha256']}\n"
            ).encode("utf-8")
        )
    sidecar_set_sha256 = sidecar_digest.hexdigest()
    binding_upgrades = [
        {
            "binding_key": str(row["binding_key"]),
            "from_sha256": str(row["from_sha256"]),
            "to_sha256": str(row["to_sha256"]),
            "reason": str(row["reason"]),
            "upgraded_at": str(row["upgraded_at"]),
        }
        for row in connection.execute(
            """
            SELECT binding_key,from_sha256,to_sha256,reason,upgraded_at
            FROM binding_upgrades
            ORDER BY id
            """
        )
    ]
    return (
        {
            "attempt_statuses": statuses,
            "attempts_terminal": int(totals["attempts"]),
            "members": int(totals["members"]),
            "chunks": int(totals["chunks"]),
            "occurrence_tokens": int(totals["occurrence_tokens"]),
            "requests": int(
                connection.execute("SELECT COUNT(*) FROM request_ledger").fetchone()[0]
            ),
            "sidecar_set_sha256": sidecar_set_sha256,
            "binding_upgrades": binding_upgrades,
        },
        sidecar_set_sha256,
    )


def _key_bytes(repo: str, run_id: int, attempt: int) -> bytes:
    return f"{repo}\t{run_id}\t{attempt}\n".encode("utf-8")


def _require_hex64(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise ReceiptFinalizationError(f"{where} is not a lowercase SHA-256")
    return value


def exhaustive_coverage_proof(
    inventory: sqlite3.Connection,
    state: sqlite3.Connection,
    *,
    inventory_receipt: Mapping[str, object],
    require_discovery_eof: bool,
    discovery_sweep: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Prove exact equality between expanded inventory and fetch attempts."""

    constrain_sqlite_evidence_rows(inventory)
    constrain_sqlite_evidence_rows(state)
    _require_bounded_fetch_state_evidence(state)

    raw_ledger = inventory_receipt.get("per_repo_ledger")
    if not isinstance(raw_ledger, list):
        raise ReceiptFinalizationError(
            "production inventory receipt lacks a per-repository ledger"
        )
    repo_order: list[str] = []
    expected_repo_ledger: dict[str, Mapping[str, object]] = {}
    observed_repo: dict[str, dict[str, object]] = {}
    for index, raw_item in enumerate(raw_ledger):
        if not isinstance(raw_item, Mapping):
            raise ReceiptFinalizationError(
                f"inventory per-repository ledger item {index} is invalid"
            )
        repo = raw_item.get("repo")
        if not isinstance(repo, str) or not repo or repo in expected_repo_ledger:
            raise ReceiptFinalizationError(
                "inventory per-repository ledger has an invalid/duplicate repo"
            )
        expected_repo_ledger[repo] = raw_item
        repo_order.append(repo)
        observed_repo[repo] = {
            "attempt_count": 0,
            "attempt_digest": hashlib.sha256(),
            "terminal_digest": hashlib.sha256(),
            "statuses": {},
        }

    expected_digest = hashlib.sha256()
    observed_digest = hashlib.sha256()
    terminal_digest = hashlib.sha256()
    expected_count = 0
    observed_count = 0
    missing_count = 0
    extra_count = 0
    missing_sample: list[str] = []
    extra_sample: list[str] = []
    invalid_status_count = 0
    invalid_status_sample: list[str] = []
    invalid_evidence_errors: list[str] = []
    status_counts: dict[str, int] = {}

    state_rows: Iterator[sqlite3.Row] = iter(
        state.execute(
            """
            SELECT repo,run_id,attempt,status,created_at,
                   run_metadata_sha256,run_metadata_raw_size,
                   run_metadata_zlib,run_metadata_exact,
                   archive_source,archive_sha256,archive_size,archive_zlib,
                   jobs_sha256,jobs_raw_size,jobs_zlib,
                   member_count,chunk_count,occurrence_tokens,
                   terminal_http_status,terminal_body_sha256
            FROM attempts ORDER BY repo,run_id,attempt
            """
        )
    )
    current = next(state_rows, None)

    def state_key(row: sqlite3.Row) -> tuple[str, int, int]:
        return (
            str(row["repo"]),
            int(row["run_id"]),
            int(row["attempt"]),
        )

    def require_jobs_proof(
        row: sqlite3.Row,
        *,
        where: str,
    ) -> list[object]:
        jobs_sha256 = _require_hex64(
            row["jobs_sha256"],
            where=f"{where} jobs_sha256",
        )
        jobs_raw_size = row["jobs_raw_size"]
        jobs_zlib = row["jobs_zlib"]
        if (
            isinstance(jobs_raw_size, bool)
            or not isinstance(jobs_raw_size, int)
            or jobs_raw_size < 2
            or not isinstance(jobs_zlib, (bytes, memoryview))
        ):
            raise ReceiptFinalizationError(
                f"{where} lacks durable jobs access evidence"
            )
        try:
            raw = strict_bounded_zlib_decode(
                jobs_zlib,
                expected_raw_size=jobs_raw_size,
                expected_sha256=jobs_sha256,
                max_raw_size=MAX_JOBS_EVIDENCE_BYTES,
                max_compressed_size=MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
                where=f"{where} jobs evidence",
            )
        except ZlibEvidenceError as exc:
            raise ReceiptFinalizationError(
                f"{where} jobs evidence is not exact bounded zlib"
            ) from exc
        try:
            jobs = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReceiptFinalizationError(
                f"{where} jobs evidence is not JSON"
            ) from exc
        if (
            not isinstance(jobs, list)
            or any(not isinstance(job, dict) for job in jobs)
            or _canonical_json_bytes(jobs) != raw
        ):
            raise ReceiptFinalizationError(
                f"{where} jobs evidence is not a canonical job list"
            )
        return jobs

    def canonical_repository(row: sqlite3.Row) -> str:
        repo, run_id, attempt = state_key(row)
        raw_size = row["run_metadata_raw_size"]
        compressed = row["run_metadata_zlib"]
        if (
            row["run_metadata_exact"] != 1
            or isinstance(raw_size, bool)
            or not isinstance(raw_size, int)
            or raw_size < 2
            or not isinstance(compressed, (bytes, memoryview))
        ):
            raise ReceiptFinalizationError(
                f"{repo}#{run_id}/{attempt} lacks exact run metadata"
            )
        try:
            raw = strict_bounded_zlib_decode(
                compressed,
                expected_raw_size=raw_size,
                expected_sha256=str(row["run_metadata_sha256"]),
                max_raw_size=MAX_RUN_METADATA_BYTES,
                max_compressed_size=MAX_RUN_METADATA_COMPRESSED_BYTES,
                where=f"{repo}#{run_id}/{attempt} exact run metadata",
            )
        except ZlibEvidenceError as exc:
            raise ReceiptFinalizationError(
                f"{repo}#{run_id}/{attempt} exact run metadata differs"
            ) from exc
        try:
            metadata = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReceiptFinalizationError(
                f"{repo}#{run_id}/{attempt} exact run metadata is invalid"
            ) from exc
        if not isinstance(metadata, Mapping) or _canonical_json_bytes(
            metadata
        ) != raw:
            raise ReceiptFinalizationError(
                f"{repo}#{run_id}/{attempt} exact run metadata is not canonical"
            )
        try:
            repository = _repository_object_identity(
                metadata.get("repository"),
                field="repository",
            )
        except MalformedResponseError as exc:
            raise ReceiptFinalizationError(
                f"{repo}#{run_id}/{attempt} repository identity is invalid"
            ) from exc
        canonical = repo if repository is None else repository[0]
        return canonical

    def require_terminal_request_proof(row: sqlite3.Row) -> None:
        repo, run_id, attempt = state_key(row)
        endpoint_repo = canonical_repository(row)
        status = str(row["status"])
        http_status = int(row["terminal_http_status"])
        body_sha256 = _require_hex64(
            row["terminal_body_sha256"],
            where=f"{repo}#{run_id}/{attempt} terminal body",
        )
        log_endpoint = (
            f"/repos/{endpoint_repo}/actions/runs/{run_id}/attempts/"
            f"{attempt}/logs"
        )
        token_digests: set[str] = set()
        candidates = 0
        for request in state.execute(
            """
            SELECT endpoint,http_status,outcome,error_message
            FROM request_ledger
            WHERE repo=? AND run_id=? AND attempt=?
              AND endpoint=? AND http_status=?
              AND outcome='terminal_candidate'
            ORDER BY id
            """,
            (repo, run_id, attempt, log_endpoint, http_status),
        ):
            try:
                evidence = json.loads(str(request["error_message"]))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ReceiptFinalizationError(
                    f"{repo}#{run_id}/{attempt} terminal ledger evidence "
                    "is malformed"
                ) from exc
            if (
                not isinstance(evidence, Mapping)
                or evidence.get("schema")
                != "cppmega_ci_terminal_http_candidate_v1"
                or evidence.get("endpoint") != log_endpoint
                or evidence.get("http_status") != http_status
                or evidence.get("body_sha256") != body_sha256
            ):
                continue
            token_digest = _require_hex64(
                evidence.get("token_sha256"),
                where=(
                    f"{repo}#{run_id}/{attempt} terminal credential digest"
                ),
            )
            token_digests.add(token_digest)
            candidates += 1
        required_candidates = 2 if http_status == 404 else 1
        if candidates < required_candidates or (
            http_status == 404 and len(token_digests) < 2
        ):
            raise ReceiptFinalizationError(
                f"{repo}#{run_id}/{attempt} lacks corroborated endpoint/body "
                "terminal evidence"
            )
        if status == "terminal_404":
            require_jobs_proof(
                row,
                where=f"{repo}#{run_id}/{attempt} terminal 404",
            )
            jobs_endpoint = (
                f"/repos/{endpoint_repo}/actions/runs/{run_id}/attempts/"
                f"{attempt}/jobs"
            )
            access = state.execute(
                """
                SELECT 1 FROM request_ledger
                WHERE repo=? AND run_id=? AND attempt=?
                  AND endpoint=? AND http_status=200 AND outcome='success'
                LIMIT 1
                """,
                (repo, run_id, attempt, jobs_endpoint),
            ).fetchone()
            if access is None:
                raise ReceiptFinalizationError(
                    f"{repo}#{run_id}/{attempt} terminal 404 lacks distinct "
                    "jobs endpoint access proof"
                )

    def require_completed_request_proof(
        row: sqlite3.Row,
        *,
        where: str,
        jobs: Sequence[object],
        allow_rescue: bool,
    ) -> None:
        repo, run_id, attempt = state_key(row)
        endpoint_repo = canonical_repository(row)
        logs_endpoint = (
            f"/repos/{endpoint_repo}/actions/runs/{run_id}/attempts/"
            f"{attempt}/logs"
        )
        jobs_endpoint = (
            f"/repos/{endpoint_repo}/actions/runs/{run_id}/attempts/"
            f"{attempt}/jobs"
        )
        logs = state.execute(
            """
            SELECT 1 FROM request_ledger
            WHERE repo=? AND run_id=? AND attempt=?
              AND endpoint=? AND http_status IN (200,302)
              AND outcome='success'
            LIMIT 1
            """,
            (repo, run_id, attempt, logs_endpoint),
        ).fetchone()
        jobs_access = state.execute(
            """
            SELECT 1 FROM request_ledger
            WHERE repo=? AND run_id=? AND attempt=?
              AND endpoint=? AND page_no=1
              AND http_status=200 AND outcome='success'
            LIMIT 1
            """,
            (repo, run_id, attempt, jobs_endpoint),
        ).fetchone()
        if jobs_access is None:
            raise ReceiptFinalizationError(
                f"{where} lacks successful logs/jobs endpoint evidence"
            )
        archive_source = str(row["archive_source"])
        typed_archive_source = archive_source in {
            "rescue-spool",
            "preserved-local-archive",
        }
        if not typed_archive_source:
            if logs is not None:
                return
            raise ReceiptFinalizationError(
                f"{where} lacks successful logs or authorized rescue evidence"
            )
        if not allow_rescue:
            raise ReceiptFinalizationError(
                f"{where} typed archive recovery is not authorized"
            )
        if archive_source == "preserved-local-archive":
            audit_row = state.execute(
                """
                SELECT error_message FROM request_ledger
                WHERE repo=? AND run_id=? AND attempt=?
                  AND endpoint='operator/preserved_archive_recovery'
                  AND outcome='operator/preserved_archive_recovery'
                  AND error_class='PreservedArchiveRecoveryReceipt'
                ORDER BY id DESC LIMIT 1
                """,
                (repo, run_id, attempt),
            ).fetchone()
            if audit_row is None or not isinstance(
                audit_row["error_message"],
                str,
            ):
                raise ReceiptFinalizationError(
                    f"{where} lacks preserved-recovery ledger evidence"
                )
            try:
                audit = json.loads(str(audit_row["error_message"]))
            except json.JSONDecodeError as exc:
                raise ReceiptFinalizationError(
                    f"{where} preserved-recovery ledger is invalid"
                ) from exc
            if (
                not isinstance(audit, Mapping)
                or audit.get("schema") != PRESERVED_RECOVERY_LEDGER_SCHEMA
                or _canonical_json_bytes(audit).decode("utf-8")
                != str(audit_row["error_message"])
            ):
                raise ReceiptFinalizationError(
                    f"{where} preserved-recovery ledger is not canonical"
                )
            consumed_rows = state.execute(
                """
                SELECT error_message FROM request_ledger
                WHERE repo=? AND run_id=? AND attempt=?
                  AND endpoint='operator/preserved_archive_recovery'
                  AND outcome='preserved_archive_consumed'
                  AND error_class='PreservedArchiveProvenance'
                ORDER BY id
                """,
                (repo, run_id, attempt),
            ).fetchall()
            if not consumed_rows:
                raise ReceiptFinalizationError(
                    f"{where} lacks consumed preserved-archive provenance"
                )
            canonical_consumed: bytes | None = None
            for consumed_row in consumed_rows:
                raw_message = consumed_row["error_message"]
                if not isinstance(raw_message, str):
                    raise ReceiptFinalizationError(
                        f"{where} preserved provenance is malformed"
                    )
                try:
                    provenance = json.loads(raw_message)
                    (
                        recovery_id,
                        source_row_sha256,
                        witness_set_sha256,
                        witnesses,
                        receipt_name,
                        receipt_bytes,
                        receipt_sha256,
                    ) = _validate_preserved_archive_provenance(
                        provenance,
                        repo=repo,
                        run_id=run_id,
                        attempt=attempt,
                        created_at=str(row["created_at"]),
                        archive_sha256=str(row["archive_sha256"]),
                        archive_size=int(row["archive_size"]),
                    )
                    _validate_preserved_recovery_operator_audit(
                        audit,
                        recovery_id=recovery_id,
                        receipt_name=receipt_name,
                        receipt_bytes=receipt_bytes,
                        receipt_sha256=receipt_sha256,
                        source_row_sha256=source_row_sha256,
                        witness_set_sha256=witness_set_sha256,
                        archive_sha256=str(row["archive_sha256"]),
                        archive_size=int(row["archive_size"]),
                    )
                except (
                    json.JSONDecodeError,
                    BindingError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise ReceiptFinalizationError(
                        f"{where} preserved provenance differs"
                    ) from exc
                encoded = _canonical_json_bytes(provenance)
                if encoded.decode("utf-8") != raw_message:
                    raise ReceiptFinalizationError(
                        f"{where} preserved provenance is not canonical"
                    )
                if canonical_consumed is None:
                    canonical_consumed = encoded
                elif canonical_consumed != encoded:
                    raise ReceiptFinalizationError(
                        f"{where} has conflicting preserved provenance"
                    )
                for witness in witnesses:
                    member = state.execute(
                        """
                        SELECT job_key,raw_sha256,raw_size,
                               chunk_count,occurrence_tokens
                        FROM members
                        WHERE repo=? AND run_id=? AND attempt=?
                          AND archive_member=?
                        """,
                        (
                            repo,
                            run_id,
                            attempt,
                            witness["archive_member"],
                        ),
                    ).fetchone()
                    if member is None or tuple(member) != (
                        witness["job_key"],
                        witness["raw_sha256"],
                        witness["raw_size"],
                        witness["chunk_count"],
                        witness["occurrence_tokens"],
                    ):
                        raise ReceiptFinalizationError(
                            f"{where} preserved member witness changed"
                        )
            return
        if archive_source != "rescue-spool":
            raise ReceiptFinalizationError(
                f"{where} lacks successful logs or authorized rescue evidence"
            )

        audit_row = state.execute(
            """
            SELECT error_message FROM request_ledger
            WHERE repo=? AND run_id=? AND attempt=?
              AND endpoint='operator/job_rescue'
              AND outcome='operator/job_rescue'
              AND error_class='JobRescueReceipt'
            ORDER BY id DESC LIMIT 1
            """,
            (repo, run_id, attempt),
        ).fetchone()
        if audit_row is None or not isinstance(
            audit_row["error_message"],
            str,
        ):
            raise ReceiptFinalizationError(
                f"{where} lacks operator job-rescue ledger evidence"
            )
        try:
            audit = json.loads(str(audit_row["error_message"]))
        except json.JSONDecodeError as exc:
            raise ReceiptFinalizationError(
                f"{where} job-rescue ledger evidence is invalid"
            ) from exc
        if (
            not isinstance(audit, Mapping)
            or audit.get("schema")
            != JOB_RESCUE_LEDGER_EVIDENCE_SCHEMA
            or _canonical_json_bytes(audit).decode("utf-8")
            != str(audit_row["error_message"])
        ):
            raise ReceiptFinalizationError(
                f"{where} job-rescue ledger evidence is not canonical/current"
            )
        consumed_rows = state.execute(
            """
            SELECT error_message FROM request_ledger
            WHERE repo=? AND run_id=? AND attempt=?
              AND endpoint='operator/job_rescue'
              AND outcome='rescue_archive_consumed'
              AND error_class='RescueArchiveProvenance'
            ORDER BY id
            """,
            (repo, run_id, attempt),
        ).fetchall()
        if not consumed_rows:
            raise ReceiptFinalizationError(
                f"{where} lacks persisted rescue archive provenance"
            )
        member_bytes_row = state.execute(
            """
            SELECT COALESCE(SUM(raw_size),0) AS raw_bytes
            FROM members
            WHERE repo=? AND run_id=? AND attempt=?
            """,
            (repo, run_id, attempt),
        ).fetchone()
        if member_bytes_row is None:
            raise ReceiptFinalizationError(
                f"{where} lacks member byte accounting"
            )
        member_uncompressed_bytes = int(member_bytes_row["raw_bytes"])
        durable_members = {
            str(member_row["archive_member"]): (
                str(member_row["raw_sha256"]),
                int(member_row["raw_size"]),
            )
            for member_row in state.execute(
                """
                SELECT archive_member,raw_sha256,raw_size
                FROM members
                WHERE repo=? AND run_id=? AND attempt=?
                ORDER BY archive_member
                """,
                (repo, run_id, attempt),
            )
        }
        canonical_consumed: bytes | None = None
        for consumed_row in consumed_rows:
            raw_message = consumed_row["error_message"]
            if not isinstance(raw_message, str):
                raise ReceiptFinalizationError(
                    f"{where} rescue archive provenance is malformed"
                )
            try:
                provenance = json.loads(raw_message)
                receipt_sha256, source_row_sha256 = (
                    _validate_rescue_archive_provenance(
                        provenance,
                        repo=repo,
                        canonical_repo=canonical_repository(row),
                        run_id=run_id,
                        attempt=attempt,
                        created_at=str(row["created_at"]),
                        run_metadata_sha256=str(
                            row["run_metadata_sha256"]
                        ),
                        run_metadata_raw_size=int(
                            row["run_metadata_raw_size"]
                        ),
                        archive_sha256=str(row["archive_sha256"]),
                        archive_size=int(row["archive_size"]),
                        jobs_sha256=str(row["jobs_sha256"]),
                        jobs_raw_size=int(row["jobs_raw_size"]),
                        job_count=len(jobs),
                        member_count=int(row["member_count"]),
                        member_uncompressed_bytes=(
                            member_uncompressed_bytes
                        ),
                        jobs=jobs,
                        durable_members=durable_members,
                    )
                )
            except (
                json.JSONDecodeError,
                BindingError,
                TypeError,
                ValueError,
            ) as exc:
                raise ReceiptFinalizationError(
                    f"{where} rescue archive provenance differs"
                ) from exc
            encoded = _canonical_json_bytes(provenance)
            if encoded.decode("utf-8") != raw_message:
                raise ReceiptFinalizationError(
                    f"{where} rescue archive provenance is not canonical"
                )
            if canonical_consumed is None:
                canonical_consumed = encoded
            elif canonical_consumed != encoded:
                raise ReceiptFinalizationError(
                    f"{where} has conflicting rescue archive provenance"
                )
            receipt_evidence = (
                provenance.get("job_rescue_receipt")
                if isinstance(provenance, Mapping)
                else None
            )
            receipt_value = (
                receipt_evidence.get("receipt")
                if isinstance(receipt_evidence, Mapping)
                else None
            )
            source_state = (
                receipt_value.get("source_state")
                if isinstance(receipt_value, Mapping)
                else None
            )
            try:
                if not isinstance(source_state, Mapping):
                    raise BindingError(
                        "rescue receipt source state is missing"
                    )
                _validate_job_rescue_operator_audit(
                    audit,
                    receipt_sha256=receipt_sha256,
                    source_row_sha256=source_row_sha256,
                    source_state=source_state,
                    archive_sha256=str(row["archive_sha256"]),
                    archive_size=int(row["archive_size"]),
                )
            except BindingError as exc:
                raise ReceiptFinalizationError(
                    f"{where} rescue provenance lacks ledger binding"
                ) from exc

    def consume_observed(row: sqlite3.Row) -> None:
        nonlocal observed_count, invalid_status_count
        repo, run_id, attempt = state_key(row)
        key_line = _key_bytes(repo, run_id, attempt)
        observed_digest.update(key_line)
        observed_count += 1
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        repo_item = observed_repo.get(repo)
        if repo_item is not None:
            repo_item["attempt_count"] = int(repo_item["attempt_count"]) + 1
            cast_digest = repo_item["attempt_digest"]
            assert hasattr(cast_digest, "update")
            cast_digest.update(key_line)
            statuses = repo_item["statuses"]
            assert isinstance(statuses, dict)
            statuses[status] = int(statuses.get(status, 0)) + 1
        terminal_http = row["terminal_http_status"]
        terminal_body = row["terminal_body_sha256"]
        status_valid = status in _EXHAUSTIVE_TERMINAL_STATES
        if status == "terminal_404":
            status_valid = (
                terminal_http == 404
                and isinstance(terminal_body, str)
                and _HEX64_RE.fullmatch(terminal_body) is not None
                and int(row["member_count"]) == 0
                and int(row["chunk_count"]) == 0
                and int(row["occurrence_tokens"]) == 0
            )
            if status_valid:
                try:
                    require_terminal_request_proof(row)
                except ReceiptFinalizationError as exc:
                    status_valid = False
                    if len(invalid_evidence_errors) < 10:
                        invalid_evidence_errors.append(str(exc))
        elif status == "terminal_410":
            status_valid = (
                terminal_http == 410
                and isinstance(terminal_body, str)
                and _HEX64_RE.fullmatch(terminal_body) is not None
                and int(row["member_count"]) == 0
                and int(row["chunk_count"]) == 0
                and int(row["occurrence_tokens"]) == 0
            )
            if status_valid:
                try:
                    require_terminal_request_proof(row)
                except ReceiptFinalizationError as exc:
                    status_valid = False
                    if len(invalid_evidence_errors) < 10:
                        invalid_evidence_errors.append(str(exc))
        elif status in {"done", "empty"}:
            status_valid = terminal_http is None and terminal_body is None
            if status == "done" and status_valid:
                where = f"{repo}#{run_id}/{attempt} done attempt"
                try:
                    _require_hex64(
                        row["archive_sha256"],
                        where=f"{where} archive_sha256",
                    )
                    archive_size = row["archive_size"]
                    if (
                        not isinstance(row["archive_source"], str)
                        or not str(row["archive_source"])
                        or isinstance(archive_size, bool)
                        or not isinstance(archive_size, int)
                        or archive_size <= 0
                        or int(row["member_count"]) < 1
                        or int(row["chunk_count"]) < 1
                        or int(row["occurrence_tokens"]) <= 0
                    ):
                        raise ReceiptFinalizationError(
                            f"{where} lacks positive archive/member evidence"
                        )
                    jobs = require_jobs_proof(row, where=where)
                    require_completed_request_proof(
                        row,
                        where=where,
                        jobs=jobs,
                        allow_rescue=(
                            str(row["archive_source"])
                            in {
                                "rescue-spool",
                                "preserved-local-archive",
                            }
                        ),
                    )
                except ReceiptFinalizationError as exc:
                    status_valid = False
                    if len(invalid_evidence_errors) < 10:
                        invalid_evidence_errors.append(str(exc))
            elif status == "empty" and status_valid:
                where = f"{repo}#{run_id}/{attempt} empty attempt"
                try:
                    archive_sha256 = _require_hex64(
                        row["archive_sha256"],
                        where=f"{where} archive_sha256",
                    )
                    archive_size = row["archive_size"]
                    archive_zlib = row["archive_zlib"]
                    common_invalid = (
                        not isinstance(row["archive_source"], str)
                        or not str(row["archive_source"])
                        or isinstance(archive_size, bool)
                        or not isinstance(archive_size, int)
                        or int(archive_size) <= 0
                        or int(row["chunk_count"]) != 0
                        or int(row["occurrence_tokens"]) != 0
                    )
                    member_count = int(row["member_count"])
                    if common_invalid or member_count < 0:
                        raise ReceiptFinalizationError(
                            f"{where} lacks parsed-empty archive proof"
                        )
                    if member_count == 0:
                        if (
                            not 22 <= int(archive_size)
                            <= MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES
                            or not isinstance(
                                archive_zlib,
                                (bytes, memoryview),
                            )
                        ):
                            raise ReceiptFinalizationError(
                                f"{where} lacks zero-member ZIP proof"
                            )
                        try:
                            archive_raw = strict_bounded_zlib_decode(
                                archive_zlib,
                                expected_raw_size=int(archive_size),
                                expected_sha256=archive_sha256,
                                max_raw_size=(
                                    MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES
                                ),
                                max_compressed_size=(
                                    MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES
                                    + 64 * 1024
                                ),
                                where=(
                                    f"{where} replayable archive evidence"
                                ),
                            )
                        except ZlibEvidenceError as exc:
                            raise ReceiptFinalizationError(
                                f"{where} replayable archive evidence differs"
                            ) from exc
                        try:
                            _validate_empty_zip_bytes(archive_raw)
                        except RuntimeError as exc:
                            raise ReceiptFinalizationError(
                                f"{where} replayable archive is not a safe "
                                "empty ZIP"
                            ) from exc
                    elif archive_zlib is not None:
                        raise ReceiptFinalizationError(
                            f"{where} parsed-empty proof carries zero-member "
                            "ZIP evidence"
                        )
                    else:
                        nonempty_member = state.execute(
                            """
                            SELECT archive_member FROM members
                            WHERE repo=? AND run_id=? AND attempt=?
                              AND (chunk_count!=0 OR occurrence_tokens!=0)
                            LIMIT 1
                            """,
                            (repo, run_id, attempt),
                        ).fetchone()
                        if nonempty_member is not None:
                            raise ReceiptFinalizationError(
                                f"{where} parsed-empty member is nonempty"
                            )
                    jobs = require_jobs_proof(row, where=where)
                    require_completed_request_proof(
                        row,
                        where=where,
                        jobs=jobs,
                        allow_rescue=(
                            str(row["archive_source"])
                            in {
                                "rescue-spool",
                                "preserved-local-archive",
                            }
                        ),
                    )
                except ReceiptFinalizationError as exc:
                    status_valid = False
                    if len(invalid_evidence_errors) < 10:
                        invalid_evidence_errors.append(str(exc))
        if not status_valid:
            invalid_status_count += 1
            if len(invalid_status_sample) < 10:
                invalid_status_sample.append(
                    f"{repo}#{run_id}/{attempt}={status}"
                )
        proof_line = (
            f"{repo}\t{run_id}\t{attempt}\t{status}\t"
            f"{row['archive_source'] or ''}\t"
            f"{row['archive_sha256'] or ''}\t"
            f"{'' if row['archive_size'] is None else row['archive_size']}\t"
            f"{row['jobs_sha256'] or ''}\t"
            f"{'' if row['jobs_raw_size'] is None else row['jobs_raw_size']}\t"
            f"{row['member_count']}\t{row['chunk_count']}\t"
            f"{row['occurrence_tokens']}\t"
            f"{'' if terminal_http is None else terminal_http}\t"
            f"{terminal_body or ''}\n"
        ).encode("utf-8")
        terminal_digest.update(proof_line)
        if repo_item is not None:
            repo_terminal = repo_item["terminal_digest"]
            assert hasattr(repo_terminal, "update")
            repo_terminal.update(proof_line)

    run_count = 0
    previous_run: tuple[str, int] | None = None
    for run in inventory.execute(
        """
        SELECT repo_key,run_id,run_attempt
        FROM runs ORDER BY repo_key,run_id,run_attempt
        """
    ):
        repo = str(run["repo_key"])
        run_id = int(run["run_id"])
        run_attempt = int(run["run_attempt"])
        if run_attempt < 1:
            raise ReceiptFinalizationError(
                f"inventory run has non-positive attempt ceiling: "
                f"{repo}#{run_id}"
            )
        run_key = (repo, run_id)
        if previous_run == run_key:
            raise ReceiptFinalizationError(
                f"inventory has duplicate run attempt ceilings: {repo}#{run_id}"
            )
        previous_run = run_key
        run_count += 1
        for attempt in range(1, run_attempt + 1):
            expected_key = (repo, run_id, attempt)
            expected_digest.update(_key_bytes(*expected_key))
            expected_count += 1
            while current is not None and state_key(current) < expected_key:
                extra_key = state_key(current)
                if len(extra_sample) < 10:
                    extra_sample.append(
                        f"{extra_key[0]}#{extra_key[1]}/{extra_key[2]}"
                    )
                extra_count += 1
                consume_observed(current)
                current = next(state_rows, None)
            if current is None or state_key(current) != expected_key:
                missing_count += 1
                if len(missing_sample) < 10:
                    missing_sample.append(
                        f"{repo}#{run_id}/{attempt}"
                    )
                continue
            consume_observed(current)
            current = next(state_rows, None)
    while current is not None:
        extra_key = state_key(current)
        if len(extra_sample) < 10:
            extra_sample.append(
                f"{extra_key[0]}#{extra_key[1]}/{extra_key[2]}"
            )
        extra_count += 1
        consume_observed(current)
        current = next(state_rows, None)

    expected_sha256 = expected_digest.hexdigest()
    observed_sha256 = observed_digest.hexdigest()
    if (
        run_count != inventory_receipt.get("run_count")
        or expected_count
        != inventory_receipt.get("expected_attempt_count")
        or expected_sha256
        != inventory_receipt.get("expected_attempt_set_sha256")
    ):
        raise ReceiptFinalizationError(
            "expanded inventory attempt proof differs from its production "
            "completion receipt"
        )
    if missing_count or extra_count or expected_sha256 != observed_sha256:
        raise ReceiptFinalizationError(
            "inventory/fetch attempt sets are not exactly equal: "
            f"missing={missing_count} sample={missing_sample}; "
            f"extra={extra_count} sample={extra_sample}"
        )
    if invalid_status_count:
        if invalid_evidence_errors:
            raise ReceiptFinalizationError(invalid_evidence_errors[0])
        raise ReceiptFinalizationError(
            "inventory-exhaustive receipt requires only done, empty, "
            "terminal_404, or terminal_410 attempts: "
            f"invalid={invalid_status_count} sample={invalid_status_sample}"
        )

    if require_discovery_eof:
        if discovery_sweep is None:
            raise ReceiptFinalizationError(
                "inventory-exhaustive discovery sweep proof is missing"
            )
        eof = discovery_sweep.get(
            "eof",
            discovery_sweep.get("discovery_eof"),
        )
        if (
            discovery_sweep.get("completion_mode")
            != COMPLETION_MODE_INVENTORY_EXHAUSTIVE
            or eof is not True
            or discovery_sweep.get("source", "persisted-fetch-sweep")
            != "persisted-fetch-sweep"
            or discovery_sweep.get("rows_seen") != run_count
            or discovery_sweep.get("expected_run_count") != run_count
            or discovery_sweep.get("expected_attempt_count")
            != expected_count
            or discovery_sweep.get("expected_attempt_set_sha256")
            != expected_sha256
        ):
            raise ReceiptFinalizationError(
                "inventory-exhaustive discovery did not reach its exact EOF"
            )
        batches = discovery_sweep.get("batches")
        if (
            isinstance(batches, bool)
            or not isinstance(batches, int)
            or batches < 1
        ):
            raise ReceiptFinalizationError(
                "inventory-exhaustive discovery batch count is invalid"
            )
        for field in (
            "inventory_receipt_sha256",
            "inventory_database_sha256",
            "inventory_db_logical_sha256",
        ):
            _require_hex64(
                discovery_sweep.get(field),
                where=f"inventory-exhaustive discovery {field}",
            )
        discovery_proof: dict[str, object] = {
            "source": "persisted-fetch-sweep",
            "completion_mode": COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
            "eof": True,
            "batches": batches,
            "rows_seen": run_count,
            "expected_run_count": run_count,
            "expected_attempt_count": expected_count,
            "expected_attempt_set_sha256": expected_sha256,
            "inventory_receipt_sha256": str(
                discovery_sweep["inventory_receipt_sha256"]
            ),
            "inventory_database_sha256": str(
                discovery_sweep["inventory_database_sha256"]
            ),
            "inventory_db_logical_sha256": str(
                discovery_sweep["inventory_db_logical_sha256"]
            ),
        }
    else:
        discovery_proof = {
            "source": "merge-recomputed-exact-union",
            "eof": True,
            "batches": 0,
            "rows_seen": run_count,
        }

    per_repo: list[dict[str, object]] = []
    for repo in repo_order:
        expected = expected_repo_ledger[repo]
        observed = observed_repo[repo]
        attempt_digest = observed["attempt_digest"]
        repo_terminal = observed["terminal_digest"]
        # hashlib objects intentionally remain local and never enter receipts.
        observed_attempt_sha256 = attempt_digest.hexdigest()  # type: ignore[union-attr]
        terminal_sha256 = repo_terminal.hexdigest()  # type: ignore[union-attr]
        if (
            observed["attempt_count"]
            != expected.get("expected_attempt_count")
            or observed_attempt_sha256
            != expected.get("expected_attempt_set_sha256")
        ):
            raise ReceiptFinalizationError(
                f"per-repository attempt proof differs for {repo}"
            )
        per_repo.append(
            {
                "repo": repo,
                "canonical": expected.get("canonical"),
                "ordinal": expected.get("ordinal"),
                "expected_run_count": expected.get("run_count"),
                "expected_attempt_count": expected.get(
                    "expected_attempt_count"
                ),
                "observed_attempt_count": observed["attempt_count"],
                "attempt_set_sha256": observed_attempt_sha256,
                "terminal_statuses": dict(
                    sorted(
                        (
                            str(key),
                            int(value),
                        )
                        for key, value in (
                            observed["statuses"]  # type: ignore[union-attr]
                        ).items()
                    )
                ),
                "terminal_proof_sha256": terminal_sha256,
            }
        )
    per_repo_sha256 = _sha256_bytes(
        json.dumps(
            per_repo,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    return {
        "completion_mode": COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        "expected_run_count": run_count,
        "expected_attempt_count": expected_count,
        "observed_attempt_count": observed_count,
        "missing_attempt_count": 0,
        "extra_attempt_count": 0,
        "incomplete_attempt_count": 0,
        "attempt_set_sha256": expected_sha256,
        "terminal_statuses": dict(sorted(status_counts.items())),
        "terminal_proof_sha256": terminal_digest.hexdigest(),
        "per_repo_ledger": per_repo,
        "per_repo_ledger_sha256": per_repo_sha256,
        "discovery": discovery_proof,
    }


def _resolve_seed_relative_path(
    seed_parent: Path,
    raw_path: object,
    *,
    where: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ReceiptFinalizationError(f"{where} path is missing")
    pure = PurePosixPath(raw_path)
    if (
        pure.is_absolute()
        or pure.as_posix() != raw_path
        or raw_path in {".", ""}
        or ".." in pure.parts
    ):
        raise ReceiptFinalizationError(
            f"{where} path is not a canonical receipt-relative path"
        )
    candidate = seed_parent.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ReceiptFinalizationError(
            f"{where} artifact is missing or unsafe"
        ) from exc
    if resolved != candidate or seed_parent not in resolved.parents:
        raise ReceiptFinalizationError(
            f"{where} path escapes through a symlink"
        )
    return resolved


def _artifact_set_from_merge_receipt(
    merge_receipt: Mapping[str, Any],
    merge_receipt_raw: bytes,
) -> dict[str, object]:
    raw_artifacts = merge_receipt.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ReceiptFinalizationError(
            "staged base merge receipt lacks its artifact manifest"
        )
    artifacts: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        if (
            not isinstance(raw_artifact, Mapping)
            or set(raw_artifact) != {"path", "byte_size", "sha256"}
        ):
            raise ReceiptFinalizationError(
                "staged base merge receipt artifact "
                f"{index} is malformed"
            )
        relative = raw_artifact.get("path")
        size = raw_artifact.get("byte_size")
        sha256 = raw_artifact.get("sha256")
        if not isinstance(relative, str):
            raise ReceiptFinalizationError(
                "staged base merge receipt artifact path is invalid"
            )
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or pure.as_posix() != relative
            or relative in {"", ".", "merge_receipt.json"}
            or ".." in pure.parts
            or relative in seen
        ):
            raise ReceiptFinalizationError(
                "staged base merge receipt artifact path is unsafe or "
                "duplicated"
            )
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or _HEX64_RE.fullmatch(sha256) is None
        ):
            raise ReceiptFinalizationError(
                "staged base merge receipt artifact identity is invalid"
            )
        seen.add(relative)
        artifacts.append(
            {
                "path": relative,
                "byte_size": size,
                "sha256": sha256,
            }
        )
    artifacts.append(
        {
            "path": "merge_receipt.json",
            "byte_size": len(merge_receipt_raw),
            "sha256": _sha256_bytes(merge_receipt_raw),
        }
    )
    digest = hashlib.sha256()
    digest.update(b"cppmega-ci-continuation-tree-v3\0")
    byte_size = 0
    for artifact in sorted(artifacts, key=lambda item: str(item["path"])):
        record = _canonical_json_bytes(artifact)
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
        byte_size += int(artifact["byte_size"])
    return {
        "file_count": len(artifacts),
        "byte_size": byte_size,
        "artifact_set_sha256": digest.hexdigest(),
    }


def verify_continuation_seed_inclusion(
    seed_receipt_path: str | os.PathLike[str],
    *,
    final_state_path: Path,
    final_store_root: Path,
) -> dict[str, object]:
    """Recompute that an immutable base union is included in continuation."""

    seed_input = Path(seed_receipt_path).expanduser()
    if seed_input.is_symlink() or not seed_input.is_file():
        raise ReceiptFinalizationError(
            f"continuation seed receipt is missing or unsafe: {seed_input}"
        )
    seed_path = seed_input.resolve()
    raw = seed_path.read_bytes()
    seed_identity = (
        seed_path.stat().st_size,
        seed_path.stat().st_mtime_ns,
        seed_path.stat().st_ino,
        _sha256_bytes(raw),
    )

    def reject_duplicates(
        pairs: Sequence[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReceiptFinalizationError(
                    f"continuation seed contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        seed = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptFinalizationError(
            f"continuation seed receipt is invalid JSON: {exc}"
        ) from exc
    if (
        not isinstance(seed, Mapping)
        or seed.get("schema")
        != "cppmega_ci_stream_continuation_seed_receipt_v3"
        or seed.get("semantics")
        != "portable-content-bound-and-reverified-mutable-clone-v3"
        or seed.get("path_semantics")
        != "seed-receipt-parent-relative-posix-v1"
        or seed.get("status") != "complete"
    ):
        raise ReceiptFinalizationError(
            "continuation seed receipt is unsupported or incomplete"
        )
    seed_parent = seed_path.parent
    controls_binding = seed.get("controls")
    if not isinstance(controls_binding, Mapping):
        raise ReceiptFinalizationError(
            "continuation seed lacks its staged control binding"
        )
    controls_root = _resolve_seed_relative_path(
        seed_parent,
        controls_binding.get("path"),
        where="continuation staged controls",
    )
    if not controls_root.is_dir():
        raise ReceiptFinalizationError(
            "continuation staged controls are not a directory"
        )
    controls_snapshot = _stream_tree_artifact_set(controls_root)
    if (
        controls_binding.get("self_contained") is not True
        or controls_binding.get("artifact_set") != controls_snapshot
    ):
        raise ReceiptFinalizationError(
            "continuation staged controls changed after cloning"
        )

    base_union = seed.get("base_union")
    if not isinstance(base_union, Mapping):
        raise ReceiptFinalizationError(
            "continuation seed lacks its base union binding"
        )
    base_path = _resolve_seed_relative_path(
        seed_parent,
        base_union.get("path"),
        where="continuation staged base",
    )
    if (
        not base_path.is_dir()
        or controls_root not in base_path.parents
    ):
        raise ReceiptFinalizationError(
            "continuation staged base is outside its control directory"
        )
    merge_binding = base_union.get("merge_receipt")
    if not isinstance(merge_binding, Mapping):
        raise ReceiptFinalizationError(
            "continuation seed lacks its base merge receipt binding"
        )
    merge_path = _resolve_seed_relative_path(
        seed_parent,
        merge_binding.get("path"),
        where="continuation staged base merge receipt",
    )
    merge_raw = merge_path.read_bytes()
    if (
        merge_path != base_path / "merge_receipt.json"
        or not merge_path.is_file()
        or _sha256_bytes(merge_raw) != merge_binding.get("sha256")
    ):
        raise ReceiptFinalizationError(
            "continuation base merge receipt changed after cloning"
        )
    try:
        staged_merge_receipt = json.loads(
            merge_raw,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptFinalizationError(
            f"continuation staged merge receipt is invalid JSON: {exc}"
        ) from exc
    if (
        not isinstance(staged_merge_receipt, Mapping)
        or staged_merge_receipt.get("status") != "complete"
        or staged_merge_receipt.get("schema")
        != merge_binding.get("schema")
    ):
        raise ReceiptFinalizationError(
            "continuation staged merge receipt is unsupported or incomplete"
        )
    source_snapshot = _artifact_set_from_merge_receipt(
        staged_merge_receipt,
        merge_raw,
    )
    if source_snapshot != base_union.get("source_artifact_set"):
        raise ReceiptFinalizationError(
            "continuation original base artifact manifest differs from its "
            "clone-time content proof"
        )
    for field, expected_name in (
        ("fetch_receipt", "fetch_receipt.json"),
        ("inventory_receipt", "inventory_receipt.json"),
        ("content_store_receipt", "store_receipt.json"),
    ):
        binding = base_union.get(field)
        if not isinstance(binding, Mapping):
            raise ReceiptFinalizationError(
                f"continuation seed lacks its staged base {field} binding"
            )
        path = _resolve_seed_relative_path(
            seed_parent,
            binding.get("path"),
            where=f"continuation staged base {field}",
        )
        if (
            path != base_path / expected_name
            or not path.is_file()
            or _sha256_file(path) != binding.get("sha256")
        ):
            raise ReceiptFinalizationError(
                f"continuation staged base {field} changed after cloning"
            )

    def verify_staged_inventory(
        database_path: Path,
        receipt_path: Path,
        *,
        expected_receipt_sha256: object,
        where: str,
    ) -> dict[str, Any]:
        try:
            receipt_document = json.loads(
                receipt_path.read_bytes(),
                object_pairs_hook=reject_duplicates,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReceiptFinalizationError(
                f"{where} receipt is invalid JSON: {exc}"
            ) from exc
        if (
            not isinstance(receipt_document, Mapping)
            or not isinstance(receipt_document.get("database"), str)
        ):
            raise ReceiptFinalizationError(
                f"{where} receipt lacks its declared database identity"
            )
        try:
            verified, receipt_sha256 = (
                verify_inventory_completion_receipt(
                    database_path,
                    receipt_path,
                    require_production=True,
                    expected_original_database_path=receipt_document[
                        "database"
                    ],
                )
            )
        except (
            InventoryCompletionError,
            OSError,
            sqlite3.Error,
            ValueError,
        ) as exc:
            raise ReceiptFinalizationError(
                f"{where} controls failed semantic reverification: {exc}"
            ) from exc
        if receipt_sha256 != expected_receipt_sha256:
            raise ReceiptFinalizationError(
                f"{where} receipt hash differs from the seed binding"
            )
        return verified

    base_inventory_receipt_binding = base_union["inventory_receipt"]
    assert isinstance(base_inventory_receipt_binding, Mapping)
    verify_staged_inventory(
        base_path / "inventory.sqlite3",
        base_path / "inventory_receipt.json",
        expected_receipt_sha256=(
            base_inventory_receipt_binding.get("sha256")
        ),
        where="continuation staged base inventory",
    )

    tokenizer_binding = seed.get("tokenizer")
    if not isinstance(tokenizer_binding, Mapping):
        raise ReceiptFinalizationError(
            "continuation seed lacks its tokenizer binding"
        )
    tokenizer_path = _resolve_seed_relative_path(
        seed_parent,
        tokenizer_binding.get("path"),
        where="continuation staged tokenizer",
    )
    if not tokenizer_path.is_file():
        raise ReceiptFinalizationError(
            "continuation seed tokenizer is missing or unsafe"
        )
    tokenizer_identity = {
        "byte_size": tokenizer_path.stat().st_size,
        "sha256": _sha256_file(tokenizer_path),
    }
    if (
        tokenizer_binding.get("byte_size")
        != tokenizer_identity["byte_size"]
        or tokenizer_binding.get("sha256")
        != tokenizer_identity["sha256"]
    ):
        raise ReceiptFinalizationError(
            "continuation seed tokenizer changed after cloning"
        )

    inventory_binding = seed.get("continuation_inventory")
    if not isinstance(inventory_binding, Mapping):
        raise ReceiptFinalizationError(
            "continuation seed lacks its inventory binding"
        )
    source_inventory_path = _resolve_seed_relative_path(
        seed_parent,
        inventory_binding.get("source_path"),
        where="continuation staged source inventory",
    )
    source_receipt_path = _resolve_seed_relative_path(
        seed_parent,
        inventory_binding.get("source_receipt_path"),
        where="continuation staged source inventory receipt",
    )
    if (
        not source_inventory_path.is_file()
        or not source_receipt_path.is_file()
    ):
        raise ReceiptFinalizationError(
            "continuation staged inventory controls are unsafe"
        )
    source_inventory_identity = {
        "byte_size": source_inventory_path.stat().st_size,
        "sha256": _sha256_file(source_inventory_path),
    }
    source_receipt_identity = {
        "byte_size": source_receipt_path.stat().st_size,
        "sha256": _sha256_file(source_receipt_path),
    }
    if (
        source_inventory_identity["sha256"]
        != inventory_binding.get("database_sha256")
        or source_receipt_identity["sha256"]
        != inventory_binding.get("source_receipt_sha256")
    ):
        raise ReceiptFinalizationError(
            "continuation source inventory controls changed after cloning"
        )
    verified_source_inventory = verify_staged_inventory(
        source_inventory_path,
        source_receipt_path,
        expected_receipt_sha256=inventory_binding.get(
            "source_receipt_sha256"
        ),
        where="continuation staged source inventory",
    )
    if (
        verified_source_inventory.get("db_logical_sha256")
        != inventory_binding.get("db_logical_sha256")
        or verified_source_inventory.get("expected_attempt_set_sha256")
        != inventory_binding.get("expected_attempt_set_sha256")
    ):
        raise ReceiptFinalizationError(
            "continuation staged source inventory logical proof differs "
            "from the seed binding"
        )

    base_state_path = base_path / "fetch_state.sqlite3"
    base_store_index = base_path / "content_store" / "index.sqlite3"
    for path, label in (
        (base_state_path, "base fetch state"),
        (base_store_index, "base content-store index"),
        (final_state_path, "continuation fetch state"),
        (final_store_root / "index.sqlite3", "continuation content-store index"),
    ):
        _require_frozen_sqlite(path, label=label)
    base_state = sqlite3.connect(
        f"{base_state_path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    final_state = sqlite3.connect(
        f"{final_state_path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    base_store = sqlite3.connect(
        f"{base_store_index.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    final_store = sqlite3.connect(
        (
            f"{(final_store_root / 'index.sqlite3').as_uri()}"
            "?mode=ro&immutable=1"
        ),
        uri=True,
    )
    for connection in (base_state, final_state, base_store, final_store):
        constrain_sqlite_evidence_rows(connection)
        connection.row_factory = sqlite3.Row
    inclusion_digest = hashlib.sha256()
    counts: dict[str, int] = {}

    def row_values(row: sqlite3.Row) -> tuple[object, ...]:
        return tuple(
            bytes(value) if isinstance(value, memoryview) else value
            for value in row
        )

    def verify_table_inclusion(
        base_connection: sqlite3.Connection,
        final_connection: sqlite3.Connection,
        tables: Mapping[
            str,
            tuple[tuple[str, ...], str],
        ],
        *,
        digest_prefix: str,
        count_prefix: str,
    ) -> None:
        for table, (keys, order_by) in tables.items():
            count = 0
            for base_row in base_connection.execute(
                f"SELECT * FROM {table} ORDER BY {order_by}"
            ):
                key_values = tuple(base_row[key] for key in keys)
                predicate = " AND ".join(f"{key}=?" for key in keys)
                final_row = final_connection.execute(
                    f"SELECT * FROM {table} WHERE {predicate}",
                    key_values,
                ).fetchone()
                if (
                    final_row is None
                    or row_values(base_row) != row_values(final_row)
                ):
                    raise ReceiptFinalizationError(
                        "continuation changed/lost base "
                        f"{digest_prefix}{table} row {key_values}"
                    )
                inclusion_digest.update(
                    (
                        f"{digest_prefix}{table}\t"
                        + "\t".join(str(value) for value in key_values)
                        + "\n"
                    ).encode()
                )
                count += 1
            counts[f"{count_prefix}{table}"] = count

    try:
        _require_bounded_fetch_state_evidence(base_state)
        _require_bounded_fetch_state_evidence(final_state)
        _require_bounded_content_store_evidence(base_store)
        _require_bounded_content_store_evidence(final_store)
        attempt_columns = [
            str(row["name"])
            for row in base_state.execute("PRAGMA table_info(attempts)")
        ]
        attempt_column_indexes = {
            column: index for index, column in enumerate(attempt_columns)
        }
        immutable_attempt_columns = (
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
        attempt_count = 0
        for base_row in base_state.execute(
            "SELECT * FROM attempts ORDER BY repo,run_id,attempt"
        ):
            key = (
                str(base_row["repo"]),
                int(base_row["run_id"]),
                int(base_row["attempt"]),
            )
            final_row = final_state.execute(
                """
                SELECT * FROM attempts
                WHERE repo=? AND run_id=? AND attempt=?
                """,
                key,
            ).fetchone()
            if final_row is None:
                raise ReceiptFinalizationError(
                    f"continuation lost base attempt {key}"
                )
            base_values = row_values(base_row)
            final_values = row_values(final_row)
            for column in immutable_attempt_columns:
                index = attempt_column_indexes[column]
                if base_values[index] != final_values[index]:
                    raise ReceiptFinalizationError(
                        f"continuation changed immutable attempt evidence "
                        f"{key}: {column}"
                    )
            if str(base_row["status"]) in _EXHAUSTIVE_TERMINAL_STATES and (
                base_values != final_values
            ):
                raise ReceiptFinalizationError(
                    f"continuation changed terminal base attempt {key}"
                )
            inclusion_digest.update(
                f"A\t{key[0]}\t{key[1]}\t{key[2]}\n".encode()
            )
            attempt_count += 1
        counts["attempts"] = attempt_count

        state_tables = {
            "members": (
                ("repo", "run_id", "attempt", "archive_member"),
                "repo,run_id,attempt,archive_member",
            ),
            "request_ledger": (("id",), "id"),
            "binding_upgrades": (("id",), "id"),
        }
        verify_table_inclusion(
            base_state,
            final_state,
            state_tables,
            digest_prefix="",
            count_prefix="",
        )

        store_tables = {
            "token_sequences": (
                ("token_sequence_sha256",),
                "token_sequence_sha256",
            ),
            "contents": (("sha256",), "sha256"),
            "occurrences": (
                (
                    "repo",
                    "run_attempt",
                    "job",
                    "step",
                    "chunk_ordinal",
                ),
                "repo,run_attempt,job,step,chunk_ordinal",
            ),
        }
        verify_table_inclusion(
            base_store,
            final_store,
            store_tables,
            digest_prefix="CAS-",
            count_prefix="cas_",
        )
    finally:
        final_store.close()
        base_store.close()
        final_state.close()
        base_state.close()
    if _stream_tree_artifact_set(controls_root) != controls_snapshot:
        raise ReceiptFinalizationError(
            "continuation staged controls changed during inclusion "
            "verification"
        )
    if {
        "byte_size": tokenizer_path.stat().st_size,
        "sha256": _sha256_file(tokenizer_path),
    } != tokenizer_identity:
        raise ReceiptFinalizationError(
            "continuation tokenizer changed during inclusion verification"
        )
    if {
        "byte_size": source_inventory_path.stat().st_size,
        "sha256": _sha256_file(source_inventory_path),
    } != source_inventory_identity or {
        "byte_size": source_receipt_path.stat().st_size,
        "sha256": _sha256_file(source_receipt_path),
    } != source_receipt_identity:
        raise ReceiptFinalizationError(
            "continuation inventory controls changed during inclusion "
            "verification"
        )
    seed_stat = seed_path.stat()
    if (
        seed_stat.st_size,
        seed_stat.st_mtime_ns,
        seed_stat.st_ino,
        _sha256_file(seed_path),
    ) != seed_identity:
        raise ReceiptFinalizationError(
            "continuation seed receipt changed during verification"
        )
    return {
        "seed_receipt_path": str(seed_path),
        "seed_receipt_sha256": _sha256_bytes(raw),
        "schema": seed["schema"],
        "base_union_path": str(base_path),
        "base_merge_receipt_sha256": merge_binding["sha256"],
        "base_source_artifact_set_sha256": source_snapshot[
            "artifact_set_sha256"
        ],
        "staged_control_artifact_set_sha256": controls_snapshot[
            "artifact_set_sha256"
        ],
        "included_rows": dict(sorted(counts.items())),
        "base_inclusion_sha256": inclusion_digest.hexdigest(),
        "base_terminal_evidence_unchanged": True,
        "base_cas_rows_unchanged": True,
        "base_source_artifacts_unchanged": True,
    }


def _require_attempt_member_accounting(
    connection: sqlite3.Connection,
) -> None:
    mismatch = connection.execute(
        """
        SELECT
          attempts.repo,
          attempts.run_id,
          attempts.attempt,
          attempts.member_count,
          attempts.chunk_count,
          attempts.occurrence_tokens,
          COUNT(members.archive_member) AS actual_member_count,
          COALESCE(SUM(members.chunk_count),0) AS actual_chunk_count,
          COALESCE(SUM(members.occurrence_tokens),0)
            AS actual_occurrence_tokens
        FROM attempts
        LEFT JOIN members
          ON members.repo=attempts.repo
         AND members.run_id=attempts.run_id
         AND members.attempt=attempts.attempt
        GROUP BY attempts.repo,attempts.run_id,attempts.attempt
        HAVING attempts.member_count != actual_member_count
            OR attempts.chunk_count != actual_chunk_count
            OR attempts.occurrence_tokens != actual_occurrence_tokens
        ORDER BY attempts.repo,attempts.run_id,attempts.attempt
        LIMIT 1
        """
    ).fetchone()
    if mismatch is not None:
        raise ReceiptFinalizationError(
            "fetch-state per-attempt member accounting is inconsistent: "
            f"{mismatch['repo']}/{mismatch['run_id']}/{mismatch['attempt']} "
            f"declared=({mismatch['member_count']},"
            f"{mismatch['chunk_count']},{mismatch['occurrence_tokens']}) "
            f"actual=({mismatch['actual_member_count']},"
            f"{mismatch['actual_chunk_count']},"
            f"{mismatch['actual_occurrence_tokens']})"
        )


def _preflight_fetch_state_accounting(state_path: Path) -> None:
    """Reject broken derived counters before scanning the full content store."""

    connection = sqlite3.connect(
        f"{state_path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    constrain_sqlite_evidence_rows(connection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        _require_bounded_fetch_state_evidence(connection)
        _require_attempt_member_accounting(connection)
    finally:
        connection.close()


def _verify_cas_member_coverage(
    store: CIContentStore,
    state_path: Path,
    *,
    tokenizer: ExactTokenizer,
) -> dict[str, int]:
    """Retokenize CAS bytes and join occurrences using bounded scratch SQLite."""

    scratch_root = Path(
        tempfile.mkdtemp(
            prefix=".ci-stream-cas-coverage-",
            dir=state_path.parent,
        )
    )
    scratch_path = scratch_root / "coverage.sqlite3"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(scratch_path, uri=True)
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-8192;
            CREATE TABLE verified_contents (
              sha256 TEXT PRIMARY KEY,
              token_count INTEGER NOT NULL,
              token_sequence_sha256 TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE reconstructed_token_sequences (
              token_sequence_sha256 TEXT PRIMARY KEY,
              token_count INTEGER NOT NULL,
              tokenizer_fingerprint TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE expected_members (
              repo TEXT NOT NULL,
              run_id INTEGER NOT NULL,
              attempt INTEGER NOT NULL,
              archive_member TEXT NOT NULL,
              status TEXT NOT NULL,
              job_key TEXT NOT NULL,
              declared_chunks INTEGER NOT NULL,
              declared_tokens INTEGER NOT NULL,
              PRIMARY KEY(repo,run_id,attempt,archive_member)
            ) WITHOUT ROWID;
            CREATE TABLE seen_chunks (
              repo TEXT NOT NULL,
              run_id INTEGER NOT NULL,
              attempt INTEGER NOT NULL,
              archive_member TEXT NOT NULL,
              ordinal INTEGER NOT NULL,
              PRIMARY KEY(repo,run_id,attempt,archive_member,ordinal)
            ) WITHOUT ROWID;
            CREATE TABLE observed_members (
              repo TEXT NOT NULL,
              run_id INTEGER NOT NULL,
              attempt INTEGER NOT NULL,
              archive_member TEXT NOT NULL,
              chunks INTEGER NOT NULL,
              tokens INTEGER NOT NULL,
              PRIMARY KEY(repo,run_id,attempt,archive_member)
            ) WITHOUT ROWID;
            """
        )
        state_uri = f"{state_path.as_uri()}?mode=ro&immutable=1"
        store_index = store.root / "index.sqlite3"
        store_uri = f"{store_index.as_uri()}?mode=ro&immutable=1"
        connection.execute("ATTACH DATABASE ? AS state_db", (state_uri,))
        connection.execute("ATTACH DATABASE ? AS cas_db", (store_uri,))

        content_batch: list[dict[str, object]] = []
        content_batch_bytes = 0

        def verify_content_batch() -> None:
            nonlocal content_batch_bytes
            if not content_batch:
                return
            texts: list[str] = []
            for content in content_batch:
                raw = content.get("content")
                if not isinstance(raw, bytes):
                    raise ReceiptFinalizationError(
                        "CAS content iteration did not return immutable bytes"
                    )
                try:
                    texts.append(raw.decode("utf-8", errors="strict"))
                except UnicodeError as exc:
                    raise ReceiptFinalizationError(
                        "CAS content is not canonical UTF-8"
                    ) from exc
            token_batches = tokenizer.encode_batch(texts)
            if len(token_batches) != len(content_batch):
                raise ReceiptFinalizationError(
                    "exact tokenizer changed CAS batch cardinality"
                )
            for content, token_ids in zip(
                content_batch,
                token_batches,
                strict=True,
            ):
                content_sha256 = content.get("sha256")
                actual_token_count = len(token_ids)
                actual_sequence_sha256 = hash_token_sequence(token_ids)
                if (
                    not isinstance(content_sha256, str)
                    or content.get("token_count") != actual_token_count
                    or content.get("tokenizer_fingerprint")
                    != tokenizer.fingerprint
                    or content.get("token_sequence_sha256")
                    != actual_sequence_sha256
                ):
                    raise ReceiptFinalizationError(
                        "CAS content token metadata differs from exact "
                        f"retokenization: {content_sha256}"
                    )
                try:
                    connection.execute(
                        """
                        INSERT INTO verified_contents(
                          sha256,token_count,token_sequence_sha256
                        ) VALUES (?,?,?)
                        """,
                        (
                            content_sha256,
                            actual_token_count,
                            actual_sequence_sha256,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ReceiptFinalizationError(
                        "CAS content iteration repeated a content digest: "
                        f"{content_sha256}"
                    ) from exc
                connection.execute(
                    """
                    INSERT INTO reconstructed_token_sequences(
                      token_sequence_sha256,token_count,
                      tokenizer_fingerprint
                    ) VALUES (?,?,?)
                    ON CONFLICT(token_sequence_sha256) DO NOTHING
                    """,
                    (
                        actual_sequence_sha256,
                        actual_token_count,
                        tokenizer.fingerprint,
                    ),
                )
                reconstructed = connection.execute(
                    """
                    SELECT token_count,tokenizer_fingerprint
                    FROM reconstructed_token_sequences
                    WHERE token_sequence_sha256=?
                    """,
                    (actual_sequence_sha256,),
                ).fetchone()
                if (
                    reconstructed is None
                    or int(reconstructed["token_count"])
                    != actual_token_count
                    or reconstructed["tokenizer_fingerprint"]
                    != tokenizer.fingerprint
                ):
                    raise ReceiptFinalizationError(
                        "exact retokenization produced a conflicting "
                        "token-sequence binding"
                    )
            content_batch.clear()
            content_batch_bytes = 0

        for content in store.iter_contents(include_content=True):
            raw = content.get("content")
            raw_size = len(raw) if isinstance(raw, bytes) else 0
            if (
                content_batch
                and content_batch_bytes + raw_size
                > _CAS_RETOKENIZE_BATCH_BYTES
            ):
                verify_content_batch()
            content_batch.append(content)
            content_batch_bytes += raw_size
            if (
                len(content_batch) >= _CAS_RETOKENIZE_BATCH_SIZE
                or content_batch_bytes >= _CAS_RETOKENIZE_BATCH_BYTES
            ):
                verify_content_batch()
        verify_content_batch()

        reconstructed_missing = connection.execute(
            """
            SELECT token_sequence_sha256,token_count,
                   tokenizer_fingerprint
            FROM reconstructed_token_sequences
            EXCEPT
            SELECT token_sequence_sha256,token_count,
                   tokenizer_fingerprint
            FROM cas_db.token_sequences
            LIMIT 1
            """
        ).fetchone()
        indexed_extra = connection.execute(
            """
            SELECT token_sequence_sha256,token_count,
                   tokenizer_fingerprint
            FROM cas_db.token_sequences
            EXCEPT
            SELECT token_sequence_sha256,token_count,
                   tokenizer_fingerprint
            FROM reconstructed_token_sequences
            LIMIT 1
            """
        ).fetchone()
        if reconstructed_missing is not None or indexed_extra is not None:
            raise ReceiptFinalizationError(
                "CAS token_sequences differs from exact retokenization"
            )
        verified_content = connection.execute(
            """
            SELECT COUNT(*) AS content_count
            FROM verified_contents
            """
        ).fetchone()
        verified_sequences = connection.execute(
            """
            SELECT COUNT(*) AS sequence_count,
                   COALESCE(SUM(token_count),0) AS token_count
            FROM reconstructed_token_sequences
            """
        ).fetchone()
        assert verified_content is not None
        assert verified_sequences is not None
        verified_content_count = int(verified_content["content_count"])
        verified_sequence_count = int(verified_sequences["sequence_count"])
        verified_unique_tokens = int(verified_sequences["token_count"])

        connection.execute(
            """
            INSERT INTO expected_members
            SELECT members.repo,members.run_id,members.attempt,
                   members.archive_member,attempts.status,members.job_key,
                   members.chunk_count,members.occurrence_tokens
            FROM state_db.members AS members
            JOIN state_db.attempts AS attempts
              USING(repo,run_id,attempt)
            """
        )

        joined_rows = connection.execute(
            """
            SELECT occurrences.repo,occurrences.run_attempt,
                   occurrences.job,occurrences.step,
                   occurrences.chunk_ordinal,
                   occurrences.content_sha256,
                   verified_contents.token_count
            FROM cas_db.occurrences AS occurrences
            JOIN verified_contents
              ON verified_contents.sha256=occurrences.content_sha256
            ORDER BY occurrences.repo,occurrences.run_attempt,
                     occurrences.job,occurrences.step,
                     occurrences.chunk_ordinal
            """
        )
        occurrence_count = 0
        occurrence_tokens = 0
        for occurrence in store.iter_occurrences():
            joined = joined_rows.fetchone()
            if joined is None:
                raise ReceiptFinalizationError(
                    "CAS content/token join ended before occurrence iteration"
                )
            provenance = occurrence.get("provenance")
            occurrence_key = occurrence.get("occurrence_key")
            if not isinstance(provenance, Mapping) or not isinstance(
                occurrence_key, Mapping
            ):
                raise ReceiptFinalizationError(
                    "CAS occurrence lacks canonical provenance"
                )
            archive = provenance.get("archive")
            if not isinstance(archive, Mapping):
                raise ReceiptFinalizationError(
                    "CAS occurrence lacks archive provenance"
                )
            repo = provenance.get("repository_scope_key")
            run_id = provenance.get("run_id")
            attempt = provenance.get("run_attempt")
            archive_member = archive.get("member")
            chunk = provenance.get("chunk")
            if (
                not isinstance(repo, str)
                or not repo
                or isinstance(run_id, bool)
                or not isinstance(run_id, int)
                or run_id <= 0
                or isinstance(attempt, bool)
                or not isinstance(attempt, int)
                or attempt <= 0
                or not isinstance(archive_member, str)
                or not archive_member
                or not isinstance(chunk, Mapping)
            ):
                raise ReceiptFinalizationError(
                    "CAS occurrence fetch identity is invalid"
                )
            ordinal = chunk.get("ordinal")
            section_id = str(
                chunk.get("section_id") or f"section:{ordinal}"
            )
            step_ordinal = chunk.get("step_ordinal")
            expected_step = (
                f"{section_id}:"
                f"{step_ordinal if step_ordinal is not None else 'none'}"
            )
            joined_identity = (
                str(joined["repo"]),
                str(joined["run_attempt"]),
                str(joined["job"]),
                str(joined["step"]),
                int(joined["chunk_ordinal"]),
                str(joined["content_sha256"]),
            )
            occurrence_identity = (
                occurrence_key.get("repo"),
                occurrence_key.get("run_attempt"),
                occurrence_key.get("job"),
                occurrence_key.get("step"),
                occurrence_key.get("chunk_ordinal"),
                occurrence.get("content_sha256"),
            )
            if joined_identity != occurrence_identity:
                raise ReceiptFinalizationError(
                    "CAS content/token join order differs from occurrence "
                    "verification"
                )
            if (
                provenance.get("schema") != "cppmega_ci_chunk_occurrence_v3"
                or isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or occurrence_key.get("repo") != repo
                or occurrence_key.get("run_attempt")
                != f"{run_id}:{attempt}"
                or occurrence_key.get("chunk_ordinal") != ordinal
                or occurrence_key.get("step") != expected_step
            ):
                raise ReceiptFinalizationError(
                    "CAS occurrence key differs from its provenance"
                )
            member_key = (repo, run_id, attempt, archive_member)
            state_member = connection.execute(
                """
                SELECT status,job_key FROM expected_members
                WHERE repo=? AND run_id=? AND attempt=?
                  AND archive_member=?
                """,
                member_key,
            ).fetchone()
            if state_member is None:
                raise ReceiptFinalizationError(
                    f"CAS occurrence has no fetch-state member: {member_key}"
                )
            status = str(state_member["status"])
            if status != "done":
                raise ReceiptFinalizationError(
                    "CAS occurrence belongs to non-done attempt: "
                    f"{member_key + (status,)}"
                )
            if occurrence_key.get("job") != str(state_member["job_key"]):
                raise ReceiptFinalizationError(
                    "CAS occurrence job differs from fetch state: "
                    f"{member_key}"
                )
            token_count = joined["token_count"]
            if (
                isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count < 0
            ):
                raise ReceiptFinalizationError(
                    "CAS occurrence references content without a token count"
                )
            try:
                connection.execute(
                    """
                    INSERT INTO seen_chunks(
                      repo,run_id,attempt,archive_member,ordinal
                    ) VALUES (?,?,?,?,?)
                    """,
                    (*member_key, ordinal),
                )
            except sqlite3.IntegrityError as exc:
                raise ReceiptFinalizationError(
                    "CAS contains duplicate member chunk ordinal: "
                    f"{member_key + (ordinal,)}"
                ) from exc
            connection.execute(
                """
                INSERT INTO observed_members(
                  repo,run_id,attempt,archive_member,chunks,tokens
                ) VALUES (?,?,?,?,1,?)
                ON CONFLICT(repo,run_id,attempt,archive_member)
                DO UPDATE SET
                  chunks=chunks+1,
                  tokens=tokens+excluded.tokens
                """,
                (*member_key, token_count),
            )
            occurrence_count += 1
            occurrence_tokens += token_count
        if joined_rows.fetchone() is not None:
            raise ReceiptFinalizationError(
                "CAS content/token join contains an unverified occurrence"
            )

        declared_chunks = 0
        declared_tokens = 0
        member_count = 0
        for row in connection.execute(
            """
            SELECT expected.repo,expected.run_id,expected.attempt,
                   expected.archive_member,expected.status,
                   expected.declared_chunks,expected.declared_tokens,
                   COALESCE(observed.chunks,0) AS observed_chunks,
                   COALESCE(observed.tokens,0) AS observed_tokens
            FROM expected_members AS expected
            LEFT JOIN observed_members AS observed
              USING(repo,run_id,attempt,archive_member)
            ORDER BY expected.repo,expected.run_id,expected.attempt,
                     expected.archive_member
            """
        ):
            member_key = (
                str(row["repo"]),
                int(row["run_id"]),
                int(row["attempt"]),
                str(row["archive_member"]),
            )
            status = str(row["status"])
            expected_counts = (
                int(row["declared_chunks"]),
                int(row["declared_tokens"]),
            )
            actual_counts = (
                int(row["observed_chunks"]),
                int(row["observed_tokens"]),
            )
            if status == "done" and actual_counts != expected_counts:
                raise ReceiptFinalizationError(
                    "CAS/fetch-state member token conservation differs: "
                    f"{member_key + expected_counts + actual_counts}"
                )
            if status == "empty" and (
                expected_counts != (0, 0) or actual_counts != (0, 0)
            ):
                raise ReceiptFinalizationError(
                    "parsed-empty fetch-state member retains training "
                    f"content: {member_key + expected_counts + actual_counts}"
                )
            if status not in {"done", "empty"}:
                raise ReceiptFinalizationError(
                    "non-terminal-content fetch-state attempt retains a member: "
                    f"{member_key + (status,) + actual_counts}"
                )
            member_count += 1
            declared_chunks += expected_counts[0]
            declared_tokens += expected_counts[1]
        if (occurrence_count, occurrence_tokens) != (
            declared_chunks,
            declared_tokens,
        ):
            raise ReceiptFinalizationError(
                "CAS/fetch-state global token conservation differs"
            )
        return {
            "fetch_members": member_count,
            "fetch_chunks": declared_chunks,
            "fetch_occurrence_tokens": declared_tokens,
            "cas_occurrences": occurrence_count,
            "cas_occurrence_tokens": occurrence_tokens,
            "verified_unique_content_count": verified_content_count,
            "verified_tokenized_unique_content_count": verified_content_count,
            "verified_unique_token_sequence_count": verified_sequence_count,
            "verified_exact_unique_payload_tokens": verified_unique_tokens,
        }
    finally:
        if connection is not None:
            connection.close()
        primary_error_active = sys.exc_info()[0] is not None
        try:
            shutil.rmtree(scratch_root)
        except OSError:
            if not primary_error_active:
                raise


def _frozen_state_binding(
    state_path: Path,
    *,
    tokenizer: ExactTokenizer,
    store_receipt: Mapping[str, object],
    original_state_path: Path,
    original_store_path: Path,
    original_inventory_path: Path | None,
) -> tuple[dict[str, object], str]:
    _require_frozen_sqlite(state_path, label="fetch state")
    before = state_path.stat()
    connection = sqlite3.connect(
        f"{state_path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    constrain_sqlite_evidence_rows(connection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        integrity = [
            str(row[0])
            for row in connection.execute("PRAGMA integrity_check").fetchall()
        ]
        if integrity != ["ok"]:
            raise ReceiptFinalizationError(
                f"fetch-state integrity_check failed: {integrity}"
            )
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ReceiptFinalizationError("fetch-state foreign_key_check failed")
        schema_sha256 = _sqlite_schema_sha256(connection)
        if schema_sha256 != _expected_fetch_state_schema_sha256():
            raise ReceiptFinalizationError(
                "fetch-state SQLite schema is not the current frozen "
                f"{SCHEMA_VERSION} schema"
            )
        _require_bounded_fetch_state_evidence(connection)
        settings = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM settings ORDER BY key")
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
        if set(settings) != expected_setting_keys:
            raise ReceiptFinalizationError(
                "fetch-state settings do not match the current frozen "
                f"{SCHEMA_VERSION} contract"
            )
        if settings["schema"] != SCHEMA_VERSION:
            raise ReceiptFinalizationError("fetch-state schema is unsupported")
        try:
            tokenizer_contract = json.loads(settings["tokenizer_contract"])
        except json.JSONDecodeError as exc:
            raise ReceiptFinalizationError(
                "fetch-state tokenizer contract is invalid"
            ) from exc
        if (
            tokenizer_contract != tokenizer.contract
            or settings["tokenizer_fingerprint"] != tokenizer.fingerprint
        ):
            raise ReceiptFinalizationError("fetch-state tokenizer binding differs")
        if Path(settings["content_store_path"]).resolve() != original_store_path:
            raise ReceiptFinalizationError(
                "fetch-state content-store path differs from the original path"
            )
        if (
            original_inventory_path is not None
            and Path(settings["inventory_path"]).resolve() != original_inventory_path
        ):
            raise ReceiptFinalizationError(
                "fetch-state inventory path differs from the original path"
            )
        if settings["content_store_script_sha256"] != store_receipt.get(
            "script_sha256"
        ):
            raise ReceiptFinalizationError(
                "fetch-state content-store script binding differs"
            )
        for field in (
            "fetcher_script_sha256",
            "parser_script_sha256",
            "content_store_script_sha256",
        ):
            if _HEX64_RE.fullmatch(settings[field]) is None:
                raise ReceiptFinalizationError(
                    f"fetch-state setting {field} is not a SHA-256"
                )
        if settings["chunk_semantics"] != (
            "parser-dedup-text-cppmega-training-tokenizer-payload-only-no-framing-v2"
        ):
            raise ReceiptFinalizationError(
                "fetch-state chunk semantics are unsupported"
            )
        processing = connection.execute(
            "SELECT repo,run_id,attempt FROM attempts WHERE status='processing' LIMIT 1"
        ).fetchone()
        if processing is not None:
            raise ReceiptFinalizationError(
                f"processing attempt cannot be frozen: {tuple(processing)}"
            )
        cas_non_done = connection.execute(
            """
            SELECT repo,run_id,attempt,status FROM attempts
            WHERE status!='done' AND (chunk_count>0 OR occurrence_tokens>0)
            LIMIT 1
            """
        ).fetchone()
        if cas_non_done is not None:
            raise ReceiptFinalizationError(
                f"CAS-bearing non-done attempt cannot be frozen: {tuple(cas_non_done)}"
            )
        summary, sidecar_set_sha256 = _canonical_summary(connection)
        logical_sha256 = _fetch_state_logical_digest(connection)
    finally:
        connection.close()

    artifact_sha256 = _sha256_file(state_path)
    after = state_path.stat()
    before_identity = (before.st_size, before.st_mtime_ns, before.st_ino)
    after_identity = (after.st_size, after.st_mtime_ns, after.st_ino)
    if before_identity != after_identity:
        raise ReceiptFinalizationError(
            "fetch-state artifact changed during finalization"
        )
    return (
        {
            "schema": SCHEMA_VERSION,
            "artifact": {
                "path": str(original_state_path),
                "byte_size": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "inode": after.st_ino,
                "sha256": artifact_sha256,
            },
            "sqlite_schema_sha256": schema_sha256,
            "sqlite_logical_sha256": logical_sha256,
            "settings": dict(sorted(settings.items())),
            "summary": summary,
            "sidecar_set_sha256": sidecar_set_sha256,
        },
        settings["inventory_path"],
    )


def _default_store_receipt_path(fetch_receipt_path: Path) -> Path:
    return fetch_receipt_path.with_name(f"{fetch_receipt_path.stem}.store.json")


def _resolved_non_symlink_path(
    value: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ReceiptFinalizationError(f"{label} cannot be a symlink: {candidate}")
    return candidate.resolve()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _acquire_fetch_state_finalization_lease(state_path: Path) -> int:
    try:
        return _acquire_fetch_state_process_lease(
            state_path,
            owner="receipt-finalizer",
        )
    except (BindingError, ValueError) as exc:
        raise ReceiptFinalizationError(
            str(exc)
        ) from exc


def _release_fetch_state_finalization_lease(descriptor: int) -> None:
    _release_fetch_state_process_lease(descriptor)


def _finalize_fetch_receipts_under_lease(
    *,
    state_path: str | os.PathLike[str],
    content_store_path: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str],
    target_unique_tokens: int,
    fetch_receipt_path: str | os.PathLike[str],
    store_receipt_path: str | os.PathLike[str] | None = None,
    original_state_path: str | os.PathLike[str] | None = None,
    original_content_store_path: str | os.PathLike[str] | None = None,
    original_inventory_path: str | os.PathLike[str] | None = None,
    completion_mode: str = COMPLETION_MODE_THRESHOLD,
    inventory_receipt_path: str | os.PathLike[str] | None = None,
    continuation_seed_receipt_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Close-time finalization for one immutable fetch-state/CAS pair."""

    if completion_mode not in {
        COMPLETION_MODE_THRESHOLD,
        COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
    }:
        raise ValueError(f"unsupported completion_mode: {completion_mode!r}")
    if (
        isinstance(target_unique_tokens, bool)
        or not isinstance(target_unique_tokens, int)
        or target_unique_tokens <= 0
    ):
        raise ValueError("target_unique_tokens must be a positive integer")
    state = _resolved_non_symlink_path(state_path, label="fetch state")
    store_root = _resolved_non_symlink_path(
        content_store_path,
        label="content store",
    )
    fetch_receipt = _resolved_non_symlink_path(
        fetch_receipt_path,
        label="fetch receipt",
    )
    store_receipt = (
        _resolved_non_symlink_path(
            _default_store_receipt_path(fetch_receipt),
            label="content-store receipt",
        )
        if store_receipt_path is None
        else _resolved_non_symlink_path(
            store_receipt_path,
            label="content-store receipt",
        )
    )
    original_state = (
        state
        if original_state_path is None
        else Path(original_state_path).expanduser().resolve()
    )
    original_store = (
        store_root
        if original_content_store_path is None
        else Path(original_content_store_path).expanduser().resolve()
    )
    original_inventory = (
        None
        if original_inventory_path is None
        else Path(original_inventory_path).expanduser().resolve()
    )
    verified_inventory_receipt: dict[str, Any] | None = None
    inventory_receipt_sha256: str | None = None
    inventory_receipt_file: Path | None = None
    discovery_sidecar: Path | None = None
    discovery_sweep: dict[str, object] | None = None
    discovery_sidecar_identity: tuple[int, int, int, str] | None = None
    if completion_mode == COMPLETION_MODE_INVENTORY_EXHAUSTIVE:
        if original_inventory is None:
            raise ValueError(
                "inventory-exhaustive finalization requires "
                "original_inventory_path"
            )
        if inventory_receipt_path is None:
            raise ValueError(
                "inventory-exhaustive finalization requires "
                "inventory_receipt_path"
            )
        inventory_receipt_file = _resolved_non_symlink_path(
            inventory_receipt_path,
            label="inventory receipt",
        )
        try:
            (
                verified_inventory_receipt,
                inventory_receipt_sha256,
            ) = verify_inventory_completion_receipt(
                original_inventory,
                inventory_receipt_file,
                require_production=True,
                expected_original_database_path=original_inventory,
            )
        except InventoryCompletionError as exc:
            raise ReceiptFinalizationError(
                f"production inventory receipt refused: {exc}"
            ) from exc
        discovery_sidecar = exhaustive_discovery_sidecar_path(state)
        try:
            discovery_sweep = load_exhaustive_discovery_sidecar(
                discovery_sidecar
            )
        except BindingError as exc:
            raise ReceiptFinalizationError(
                f"exhaustive discovery sidecar refused: {exc}"
            ) from exc
        if discovery_sweep is not None:
            sidecar_stat = discovery_sidecar.stat()
            discovery_sidecar_identity = (
                sidecar_stat.st_size,
                sidecar_stat.st_mtime_ns,
                sidecar_stat.st_ino,
                _sha256_file(discovery_sidecar),
            )
    index_path = store_root / "index.sqlite3"
    tokenizer_file = _resolved_non_symlink_path(
        tokenizer_path,
        label="tokenizer",
    )
    if fetch_receipt == store_receipt:
        raise ValueError("fetch and content-store receipt paths must differ")
    protected_inputs = {state, index_path, tokenizer_file}
    if fetch_receipt in protected_inputs or store_receipt in protected_inputs:
        raise ValueError("a receipt path collides with an immutable input")
    if _is_within(fetch_receipt, store_root) or _is_within(
        store_receipt,
        store_root,
    ):
        raise ValueError("receipt paths must be outside the content store")
    _freeze_fetch_state_sqlite(state)
    _require_frozen_sqlite(index_path, label="content store")
    _preflight_fetch_state_accounting(state)
    initial_state = state.stat()
    initial_state_identity = (
        initial_state.st_size,
        initial_state.st_mtime_ns,
        initial_state.st_ino,
    )
    tokenizer = ExactTokenizer(tokenizer_file)

    store = CIContentStore(store_root)
    try:
        cas_conservation = _verify_cas_member_coverage(
            store,
            state,
            tokenizer=tokenizer,
        )
        verified_unique_tokens = cas_conservation[
            "verified_exact_unique_payload_tokens"
        ]
        if verified_unique_tokens < target_unique_tokens:
            raise ReceiptFinalizationError(
                "completion receipt refused: exact CAS retokenization "
                f"produced {verified_unique_tokens} unique payload tokens, "
                f"below target {target_unique_tokens}"
            )
        store_value = store.completion_receipt(
            target_unique_tokens=target_unique_tokens
        )
        store_counters = store_value.get("counters")
        if not isinstance(store_counters, Mapping):
            raise ReceiptFinalizationError(
                "content-store receipt counters are missing"
            )
        expected_verified_counters = {
            "unique_content_count": cas_conservation[
                "verified_unique_content_count"
            ],
            "tokenized_unique_content_count": cas_conservation[
                "verified_tokenized_unique_content_count"
            ],
            "unique_token_sequence_count": cas_conservation[
                "verified_unique_token_sequence_count"
            ],
            "exact_unique_payload_tokens": verified_unique_tokens,
            "occurrence_count": cas_conservation["cas_occurrences"],
        }
        for field, expected in expected_verified_counters.items():
            actual = store_counters.get(field)
            if (
                isinstance(actual, bool)
                or not isinstance(actual, int)
                or actual != expected
            ):
                raise ReceiptFinalizationError(
                    "content-store receipt counter differs from exact CAS "
                    f"reconstruction: {field}={actual!r}, expected={expected}"
                )
        if (
            store_counters.get("tokenizer_fingerprint")
            != tokenizer.fingerprint
            or store_value.get("exact_unique_payload_tokens")
            != verified_unique_tokens
            or store_value.get("target_exact_unique_payload_tokens")
            != target_unique_tokens
        ):
            raise ReceiptFinalizationError(
                "content-store receipt token binding differs from exact CAS "
                "reconstruction"
            )
    finally:
        store.close()
    _require_frozen_sqlite(index_path, label="content store")

    frozen_state, bound_inventory = _frozen_state_binding(
        state,
        tokenizer=tokenizer,
        store_receipt=store_value,
        original_state_path=original_state,
        original_store_path=original_store,
        original_inventory_path=original_inventory,
    )
    exhaustive_proof: dict[str, object] | None = None
    inventory_binding: dict[str, object] | None = None
    if completion_mode == COMPLETION_MODE_INVENTORY_EXHAUSTIVE:
        assert original_inventory is not None
        assert verified_inventory_receipt is not None
        assert inventory_receipt_sha256 is not None
        assert inventory_receipt_file is not None
        inventory_connection = sqlite3.connect(
            f"{original_inventory.as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        state_connection = sqlite3.connect(
            f"{state.as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        constrain_sqlite_evidence_rows(inventory_connection)
        constrain_sqlite_evidence_rows(state_connection)
        inventory_connection.row_factory = sqlite3.Row
        state_connection.row_factory = sqlite3.Row
        try:
            exhaustive_proof = exhaustive_coverage_proof(
                inventory_connection,
                state_connection,
                inventory_receipt=verified_inventory_receipt,
                require_discovery_eof=True,
                discovery_sweep=discovery_sweep,
            )
        finally:
            state_connection.close()
            inventory_connection.close()
        discovery = exhaustive_proof["discovery"]
        assert isinstance(discovery, Mapping)
        artifact = verified_inventory_receipt["database_artifact"]
        assert isinstance(artifact, Mapping)
        if (
            discovery.get("inventory_receipt_sha256")
            != inventory_receipt_sha256
            or discovery.get("inventory_database_sha256")
            != artifact.get("sha256")
            or discovery.get("inventory_db_logical_sha256")
            != verified_inventory_receipt.get("db_logical_sha256")
        ):
            raise ReceiptFinalizationError(
                "persisted discovery sweep does not bind the verified "
                "production inventory"
            )
        inventory_binding = {
            "database": {
                "path": str(original_inventory),
                "byte_size": artifact["byte_size"],
                "sha256": artifact["sha256"],
                "db_logical_sha256": verified_inventory_receipt[
                    "db_logical_sha256"
                ],
            },
            "completion_receipt": {
                "path": str(inventory_receipt_file),
                "sha256": inventory_receipt_sha256,
                "schema": verified_inventory_receipt["schema"],
            },
            "repo_count": verified_inventory_receipt["repo_list"][
                "repos"
            ],
            "expected_run_count": verified_inventory_receipt["run_count"],
            "expected_attempt_count": verified_inventory_receipt[
                "expected_attempt_count"
            ],
            "attempt_set_sha256": verified_inventory_receipt[
                "expected_attempt_set_sha256"
            ],
        }
        store_counters = store_value["counters"]
        assert isinstance(store_counters, Mapping)
        state_summary = frozen_state["summary"]
        assert isinstance(state_summary, Mapping)
        if store_counters.get("occurrence_count") != state_summary.get(
            "chunks"
        ):
            raise ReceiptFinalizationError(
                "CAS occurrence count differs from fetch-state chunk count"
            )
        if (
            int(store_counters["occurrence_count"])
            != cas_conservation["cas_occurrences"]
            or int(state_summary["members"])
            != cas_conservation["fetch_members"]
            or int(state_summary["chunks"])
            != cas_conservation["fetch_chunks"]
            or int(state_summary["occurrence_tokens"])
            != cas_conservation["fetch_occurrence_tokens"]
        ):
            raise ReceiptFinalizationError(
                "receipt summaries differ from the bounded CAS/state join"
            )
        conservation: dict[str, object] = {
            **cas_conservation,
            "exact_unique_payload_tokens": verified_unique_tokens,
            "secondary_minimum_exact_unique_payload_tokens": (
                target_unique_tokens
            ),
            "cas_member_chunk_join_complete": True,
            "occurrence_chunk_count_equal": True,
            "occurrence_token_count_equal": True,
            "secondary_token_minimum_met": (
                verified_unique_tokens >= target_unique_tokens
            ),
        }
    else:
        conservation = {}
    continuation_inclusion: dict[str, object] | None = None
    if continuation_seed_receipt_path is not None:
        if completion_mode != COMPLETION_MODE_INVENTORY_EXHAUSTIVE:
            raise ReceiptFinalizationError(
                "continuation seed inclusion is supported only for "
                "inventory-exhaustive receipts"
            )
        continuation_inclusion = verify_continuation_seed_inclusion(
            continuation_seed_receipt_path,
            final_state_path=state,
            final_store_root=store_root,
        )
    value = {
        "schema": (
            EXHAUSTIVE_RECEIPT_SCHEMA
            if completion_mode == COMPLETION_MODE_INVENTORY_EXHAUSTIVE
            else RECEIPT_SCHEMA
        ),
        "completed_at": _utc_now(),
        "target_exact_unique_payload_tokens": target_unique_tokens,
        "fetch_state": frozen_state["summary"],
        "frozen_fetch_state": frozen_state,
        "content_store_receipt": store_value,
        "inventory_path": bound_inventory,
        "tokenizer_contract": tokenizer.contract,
        "tokenizer_fingerprint": tokenizer.fingerprint,
    }
    if exhaustive_proof is not None:
        value["completion_mode"] = completion_mode
        value["production_complete"] = True
        value["coverage_semantics"] = (
            "exact-production-inventory-attempt-equality"
        )
        value["inventory_binding"] = inventory_binding
        value["exhaustive_coverage"] = exhaustive_proof
        value["conservation"] = conservation
        value["continuation_seed"] = continuation_inclusion
    final_state = state.stat()
    if (
        final_state.st_size,
        final_state.st_mtime_ns,
        final_state.st_ino,
    ) != initial_state_identity:
        raise ReceiptFinalizationError(
            "fetch-state artifact changed during receipt finalization"
        )
    if (
        discovery_sidecar is not None
        and discovery_sidecar_identity is not None
    ):
        final_sidecar = discovery_sidecar.stat()
        if (
            final_sidecar.st_size,
            final_sidecar.st_mtime_ns,
            final_sidecar.st_ino,
            _sha256_file(discovery_sidecar),
        ) != discovery_sidecar_identity:
            raise ReceiptFinalizationError(
                "exhaustive discovery sidecar changed during finalization"
            )
    atomic_write_json(store_receipt, store_value)
    atomic_write_json(fetch_receipt, value)
    _require_frozen_sqlite(state, label="fetch state")
    _require_frozen_sqlite(index_path, label="content store")
    return value


def finalize_fetch_receipts(
    *,
    state_path: str | os.PathLike[str],
    content_store_path: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str],
    target_unique_tokens: int,
    fetch_receipt_path: str | os.PathLike[str],
    store_receipt_path: str | os.PathLike[str] | None = None,
    original_state_path: str | os.PathLike[str] | None = None,
    original_content_store_path: str | os.PathLike[str] | None = None,
    original_inventory_path: str | os.PathLike[str] | None = None,
    completion_mode: str = COMPLETION_MODE_THRESHOLD,
    inventory_receipt_path: str | os.PathLike[str] | None = None,
    continuation_seed_receipt_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Finalize receipts while excluding every conforming stream writer."""

    state = _resolved_non_symlink_path(state_path, label="fetch state")
    descriptor = _acquire_fetch_state_finalization_lease(state)
    try:
        return _finalize_fetch_receipts_under_lease(
            state_path=state,
            content_store_path=content_store_path,
            tokenizer_path=tokenizer_path,
            target_unique_tokens=target_unique_tokens,
            fetch_receipt_path=fetch_receipt_path,
            store_receipt_path=store_receipt_path,
            original_state_path=original_state_path,
            original_content_store_path=original_content_store_path,
            original_inventory_path=original_inventory_path,
            completion_mode=completion_mode,
            inventory_receipt_path=inventory_receipt_path,
            continuation_seed_receipt_path=continuation_seed_receipt_path,
        )
    finally:
        _release_fetch_state_finalization_lease(descriptor)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize frozen store/fetch receipts after every stream writer exits"
        )
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--content-store", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-exact-unique-payload-tokens", required=True, type=int)
    parser.add_argument("--fetch-receipt", required=True)
    parser.add_argument("--store-receipt")
    parser.add_argument("--original-state-path")
    parser.add_argument("--original-content-store-path")
    parser.add_argument("--original-inventory-path")
    parser.add_argument(
        "--completion-mode",
        choices=(
            COMPLETION_MODE_THRESHOLD,
            COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        ),
        default=COMPLETION_MODE_THRESHOLD,
    )
    parser.add_argument("--inventory-receipt")
    parser.add_argument("--continuation-seed-receipt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        value = finalize_fetch_receipts(
            state_path=args.state,
            content_store_path=args.content_store,
            tokenizer_path=args.tokenizer,
            target_unique_tokens=args.target_exact_unique_payload_tokens,
            fetch_receipt_path=args.fetch_receipt,
            store_receipt_path=args.store_receipt,
            original_state_path=args.original_state_path,
            original_content_store_path=args.original_content_store_path,
            original_inventory_path=args.original_inventory_path,
            completion_mode=args.completion_mode,
            inventory_receipt_path=args.inventory_receipt,
            continuation_seed_receipt_path=(
                args.continuation_seed_receipt
            ),
        )
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
        print(f"[ci-stream-receipts] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
