#!/usr/bin/env python3
"""Resumable, receipt-bound GitHub Actions workflow-run inventory.

This stage inventories workflow-run metadata only.  It deliberately does not
download logs, artifacts, or jobs.  GitHub caps a filtered workflow-run listing
at 1,000 results, so every repository starts with one explicit UTC ``[start,
end)`` window and dense windows are bisected recursively before pagination.

Progress is committed page-by-page to SQLite.  A completion receipt is emitted
only after the database proves that every repository has a gap-free,
non-overlapping set of closed leaf windows, every page/count closes exactly,
and the repository scope still matches the canonical input.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request
import zlib

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci_zlib_evidence import (
    MAX_RUN_METADATA_BYTES,
    MAX_RUN_METADATA_COMPRESSED_BYTES,
    ZlibEvidenceError,
    strict_bounded_zlib_decode,
)


SCHEMA_VERSION = "cppmega_ci_stream_inventory_v4"
RECEIPT_SCHEMA = "cppmega_ci_stream_inventory_receipt_v6"
SOURCE_DRIFT_RECONCILIATION_SCHEMA = (
    "cppmega_ci_source_drift_reconciliation_v1"
)
PROGRESS_SCHEMA = "cppmega_ci_stream_inventory_progress_v3"
PREVIOUS_SCHEMA_VERSION = "cppmega_ci_stream_inventory_v3"
LEGACY_SCHEMA_VERSION = "cppmega_ci_stream_inventory_v2"
GITHUB_API_VERSION = "2022-11-28"
DEFAULT_PER_PAGE = 100
GITHUB_FILTER_LIMIT = 1000
METADATA_ENCODING = "zlib6-canonical-json-utf8-v1"
CONVERGENCE_MAX_PASSES = 64
MAX_UPGRADE_REASON_CHARS = 1000
MAX_INVENTORY_RECEIPT_BYTES = 128 * 1024 * 1024
MAX_INVENTORY_SQLITE_ROW_BYTES = (
    MAX_RUN_METADATA_COMPRESSED_BYTES
    + MAX_RUN_METADATA_BYTES
    + 256 * 1024
)
IMPORTED_UPGRADE_REASON = (
    "imported pre-v3 inventory producer upgrade audit record"
)

_OWNER_REPO_RE = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,99}))/"
    r"(?P<repo>[A-Za-z0-9_.-]{1,100})$"
)
class InventoryError(RuntimeError):
    """Base class for fail-closed inventory errors."""


class ScopeError(InventoryError):
    """The canonical repository list is malformed or unresolved."""


class BindingError(InventoryError):
    """An existing database does not match this invocation."""


class APIError(InventoryError):
    """GitHub returned a permanent or exhausted-retry error."""


class MalformedAPIError(APIError):
    """GitHub returned a response that cannot prove complete enumeration."""


class UnstableEnumerationError(APIError):
    """Repeated observations disagree, so no stable snapshot can be claimed."""


class PaginationDrift(UnstableEnumerationError):
    """A paginated leaf shifted and must be invalidated and subdivided."""

    def __init__(self, message: str, *, observed_total: int):
        super().__init__(message)
        self.observed_total = observed_total


class CompletionError(InventoryError):
    """The SQLite inventory cannot support a completion receipt."""


def _constrain_inventory_connection(connection: sqlite3.Connection) -> int:
    current = connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
    if current > MAX_INVENTORY_SQLITE_ROW_BYTES:
        connection.setlimit(
            sqlite3.SQLITE_LIMIT_LENGTH,
            MAX_INVENTORY_SQLITE_ROW_BYTES,
        )
    configured = connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
    if configured > MAX_INVENTORY_SQLITE_ROW_BYTES:
        raise InventoryError(
            "inventory SQLite row length limit could not be constrained"
        )
    return configured


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class Repo:
    key: str
    owner: str
    name: str
    canonical: str
    ordinal: int


@dataclass(frozen=True)
class RepoScope:
    path: str
    source_sha256: str
    scope_sha256: str
    repos: tuple[Repo, ...]
    original_repo_count: int
    unresolved_count: int
    smoke: bool
    max_repos: int | None


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class PageResponse:
    total_count: int
    workflow_runs: tuple[dict[str, Any], ...]
    payload_sha256: str


@dataclass
class _TokenState:
    token: str
    remaining: int | None = None
    reset_epoch: float = 0.0
    cooldown_until: float = 0.0


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _require_canonical_utc(value: object, *, where: str) -> str:
    if not isinstance(value, str):
        raise CompletionError(f"{where} must be a canonical UTC timestamp")
    try:
        epoch = parse_utc_instant(value)
    except ValueError as exc:
        raise CompletionError(
            f"{where} must be a canonical UTC timestamp"
        ) from exc
    if format_utc_instant(epoch) != value:
        raise CompletionError(f"{where} must be a canonical UTC timestamp")
    return value


def _require_canonical_decimal(
    value: object,
    *,
    where: str,
    minimum: int = 0,
) -> int:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise CompletionError(
            f"{where} must be a canonical decimal integer"
        )
    parsed = int(value)
    if parsed < minimum or str(parsed) != value:
        raise CompletionError(
            f"{where} must be a canonical decimal integer >= {minimum}"
        )
    return parsed


def _require_exact_json(
    actual: object,
    expected: object,
    *,
    where: str,
) -> None:
    """Require exact JSON shape, scalar type, and value equality."""

    if type(actual) is not type(expected):
        raise CompletionError(f"{where} has the wrong JSON type")
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        if set(actual) != set(expected):
            raise CompletionError(f"{where} has extra/missing fields")
        for key in expected:
            _require_exact_json(
                actual[key],
                expected[key],
                where=f"{where}.{key}",
            )
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        if len(actual) != len(expected):
            raise CompletionError(f"{where} has the wrong item count")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _require_exact_json(
                actual_item,
                expected_item,
                where=f"{where}[{index}]",
            )
        return
    if actual != expected:
        raise CompletionError(f"{where} differs from SQLite")


def _require_safe_checkpoint_sidecars(database: Path) -> None:
    for suffix in ("-wal", "-journal"):
        sidecar = Path(f"{database}{suffix}")
        if sidecar.is_symlink():
            raise CompletionError(
                "inventory database has a non-empty/unsafe checkpoint "
                f"sidecar: {sidecar.name}"
            )
        if sidecar.exists() and (
            not sidecar.is_file() or sidecar.stat().st_size != 0
        ):
            raise CompletionError(
                "inventory database has a non-empty/unsafe checkpoint "
                f"sidecar: {sidecar.name}"
            )
    shm = Path(f"{database}-shm")
    if shm.is_symlink() or (shm.exists() and not shm.is_file()):
        raise CompletionError(
            f"inventory database has an unsafe checkpoint sidecar: {shm.name}"
        )


def _read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
    where: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CompletionError(f"{where} is not a regular file")
        if before.st_size > max_bytes:
            raise CompletionError(
                f"{where} exceeds its {max_bytes}-byte limit"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise CompletionError(f"{where} changed while it was read")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise CompletionError(f"{where} changed while it was read")
        after = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CompletionError(f"{where} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _copy_database_snapshot_once(
    source: Path,
    destination: Path,
) -> tuple[int, str, tuple[int, int, int, int, int]]:
    """Copy and hash one stable database identity in a single bounded pass."""

    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    source_flags |= getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source, source_flags)
    destination_descriptor = -1
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CompletionError(
                "inventory database snapshot source is not a regular file"
            )
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise CompletionError(
                        "inventory database snapshot write made no progress"
                    )
                view = view[written:]
            copied += len(block)
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or copied != before.st_size
        ):
            raise CompletionError(
                "inventory database changed while its private snapshot "
                "was copied"
            )
        path_after = source.lstat()
        if (
            not stat.S_ISREG(path_after.st_mode)
            or identity
            != (
                path_after.st_dev,
                path_after.st_ino,
                path_after.st_size,
                path_after.st_mtime_ns,
                path_after.st_ctime_ns,
            )
        ):
            raise CompletionError(
                "inventory database path changed while its private snapshot "
                "was copied"
            )
        return copied, digest.hexdigest(), identity
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        os.close(source_descriptor)


def _hash_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_upgrade_reason(value: str | None) -> str:
    if value is None:
        raise BindingError(
            "an explicit inventory script upgrade requires a reason"
        )
    reason = value.strip()
    if (
        not reason
        or len(reason) > MAX_UPGRADE_REASON_CHARS
        or any(not character.isprintable() for character in reason)
    ):
        raise BindingError(
            "inventory script upgrade reason must be non-empty printable text "
            f"of at most {MAX_UPGRADE_REASON_CHARS} characters"
        )
    return reason


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_instant(value: str) -> int:
    """Parse a second-precision UTC timestamp and return its Unix epoch."""

    raw = value.strip()
    if not raw:
        raise ValueError("UTC timestamp must not be empty")
    normalized = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include an explicit UTC offset: {value!r}")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"timestamp must be UTC, got {value!r}")
    if parsed.microsecond:
        raise ValueError(
            "GitHub workflow-run timestamps are second precision; "
            f"fractional boundary is not allowed: {value!r}"
        )
    return int(parsed.timestamp())


def format_utc_instant(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_projection_digests(
    repo_key: str,
    members: Mapping[tuple[int, int], str],
) -> tuple[str, str]:
    ordered = sorted(members.items())
    membership_sha256 = _hash_lines(
        f"{repo_key}\t{run_id}\t{run_attempt}"
        for (run_id, run_attempt), _metadata_sha256 in ordered
    )
    metadata_sha256 = _hash_lines(
        f"{repo_key}\t{run_id}\t{run_attempt}\t{metadata_sha256}"
        for (run_id, run_attempt), metadata_sha256 in ordered
    )
    return membership_sha256, metadata_sha256


def _minimal_source_drift_roots(
    connection: sqlite3.Connection,
) -> list[dict[str, int | str]]:
    rows = connection.execute(
        """
        WITH child_totals AS (
          SELECT parent_id,SUM(expected_total) AS child_total,
                 COUNT(*) AS child_count
          FROM search_windows
          WHERE parent_id IS NOT NULL
          GROUP BY parent_id
        ),
        drift AS (
          SELECT parent.id,parent.repo_key,parent.start_epoch,
                 parent.end_epoch,parent.expected_total AS parent_total,
                 children.child_total
          FROM search_windows parent
          JOIN child_totals children ON children.parent_id=parent.id
          WHERE parent.status='split'
            AND children.child_count=2
            AND parent.expected_total!=children.child_total
        )
        SELECT drift.*
        FROM drift
        WHERE NOT EXISTS (
          SELECT 1
          FROM drift ancestor
          WHERE ancestor.repo_key=drift.repo_key
            AND ancestor.id!=drift.id
            AND ancestor.start_epoch<=drift.start_epoch
            AND ancestor.end_epoch>=drift.end_epoch
        )
        ORDER BY drift.repo_key,drift.start_epoch,drift.end_epoch,drift.id
        """
    ).fetchall()
    return [
        {
            "window_id": int(row["id"]),
            "repo": str(row["repo_key"]),
            "start_epoch": int(row["start_epoch"]),
            "end_epoch": int(row["end_epoch"]),
            "parent_total": int(row["parent_total"]),
            "child_total": int(row["child_total"]),
        }
        for row in rows
    ]


def _source_count_drift_summary(
    connection: sqlite3.Connection,
) -> dict[str, int | str]:
    rows = connection.execute(
        """
        WITH child_totals AS (
          SELECT parent_id,SUM(expected_total) AS child_total,
                 COUNT(*) AS child_count
          FROM search_windows
          WHERE parent_id IS NOT NULL
          GROUP BY parent_id
        )
        SELECT parent.repo_key,parent.start_epoch,parent.end_epoch,
               parent.expected_total AS parent_total,children.child_total
        FROM search_windows parent
        JOIN child_totals children ON children.parent_id=parent.id
        WHERE parent.status='split'
          AND children.child_count=2
          AND parent.expected_total!=children.child_total
        ORDER BY parent.repo_key,parent.start_epoch,parent.end_epoch,parent.id
        """
    ).fetchall()
    lines = sorted(
        f"S\t{row['repo_key']}\t{row['start_epoch']}\t{row['end_epoch']}\t"
        f"{row['parent_total']}\t{row['child_total']}\t"
        f"{int(row['parent_total']) - int(row['child_total'])}"
        for row in rows
    )
    parent_total = sum(int(row["parent_total"]) for row in rows)
    child_total = sum(int(row["child_total"]) for row in rows)
    return {
        "windows": len(rows),
        "parent_total": parent_total,
        "child_total": child_total,
        "net_parent_minus_children": parent_total - child_total,
        "absolute_delta": sum(
            abs(int(row["parent_total"]) - int(row["child_total"]))
            for row in rows
        ),
        "sha256": _hash_lines(lines),
        "semantics": (
            "GitHub total_count observations at each split parent "
            "versus its later child enumeration; nonzero means the "
            "source cardinality changed or pagination contradicted "
            "itself during inventory; zero means no such "
            "contradiction was observed, not proof of an atomic "
            "GitHub snapshot"
        ),
    }


def _stored_reconciliation_members(
    connection: sqlite3.Connection,
    root: Mapping[str, int | str],
) -> dict[tuple[int, int], str]:
    rows = connection.execute(
        """
        SELECT run_id,run_attempt,metadata_sha256
        FROM runs run
        WHERE repo_key=? AND created_at>=? AND created_at<?
          AND EXISTS (
              SELECT 1 FROM window_runs linked
              WHERE linked.repo_key=run.repo_key
                AND linked.run_id=run.run_id
                AND linked.run_attempt=run.run_attempt
          )
        ORDER BY run_id,run_attempt
        """,
        (
            root["repo"],
            format_utc_instant(int(root["start_epoch"])),
            format_utc_instant(int(root["end_epoch"])),
        ),
    ).fetchall()
    return {
        (int(row["run_id"]), int(row["run_attempt"])): str(
            row["metadata_sha256"]
        )
        for row in rows
    }


def _load_source_drift_proofs(
    connection: sqlite3.Connection,
) -> dict[int, dict[str, Any]]:
    proofs: dict[int, dict[str, Any]] = {}
    for row in connection.execute(
        """
        SELECT window_id,proof_blob,proof_raw_size,proof_sha256,
               producer_script_sha256,completed_at
        FROM source_drift_reconciliations
        ORDER BY window_id
        """
    ):
        window_id = int(row["window_id"])
        blob = row["proof_blob"]
        if not isinstance(blob, bytes):
            raise CompletionError(
                "source-drift reconciliation proof is not a BLOB"
            )
        try:
            raw = strict_bounded_zlib_decode(
                blob,
                expected_raw_size=int(row["proof_raw_size"]),
                expected_sha256=str(row["proof_sha256"]),
                max_raw_size=MAX_RUN_METADATA_BYTES,
                max_compressed_size=MAX_RUN_METADATA_COMPRESSED_BYTES,
                where="source-drift reconciliation proof",
            )
            proof = json.loads(raw)
        except (
            ZlibEvidenceError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise CompletionError(
                "source-drift reconciliation proof is corrupt"
            ) from exc
        if (
            not isinstance(proof, dict)
            or _canonical_json(proof).encode("utf-8") != raw
            or proof.get("window_id") != window_id
            or proof.get("producer_script_sha256")
            != row["producer_script_sha256"]
            or proof.get("completed_at") != row["completed_at"]
            or window_id in proofs
        ):
            raise CompletionError(
                "source-drift reconciliation proof is not canonically bound"
            )
        proofs[window_id] = proof
    return proofs


def _source_drift_reconciliation_payload(
    *,
    script_sha256: str,
    source_count_drift: Mapping[str, Any],
    roots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not roots or any(
        root.get("producer_script_sha256") != script_sha256
        for root in roots
    ):
        raise CompletionError(
            "source-drift roots do not share their bound producer"
        )
    return {
        "schema": SOURCE_DRIFT_RECONCILIATION_SCHEMA,
        "created_at": max(str(root["completed_at"]) for root in roots),
        "producer_script_sha256": script_sha256,
        "semantics": (
            "two exact repeated live enumerations were unioned with the "
            "frozen observed membership; upstream deletions remain retained "
            "and stable new runs are appended"
        ),
        "source_count_drift_sha256": source_count_drift["sha256"],
        "source_count_drift_windows": source_count_drift["windows"],
        "root_count": len(roots),
        "stored_run_count": sum(int(root["stored_count"]) for root in roots),
        "current_run_count": sum(int(root["current_count"]) for root in roots),
        "retained_upstream_deleted_count": sum(
            int(root["retained_upstream_deleted_count"]) for root in roots
        ),
        "new_current_run_count": sum(
            int(root["new_current_count"]) for root in roots
        ),
        "observed_union_run_count": sum(
            int(root["observed_union_count"]) for root in roots
        ),
        "two_pass_exact": True,
        "observed_union_complete": True,
        "roots": [dict(root) for root in roots],
    }


def _normalize_owner_repo(value: str) -> tuple[str, str]:
    candidate = value.strip()
    if candidate.startswith("git@github.com:"):
        candidate = candidate[len("git@github.com:") :]
    elif "://" in candidate:
        parsed = urllib.parse.urlparse(candidate)
        if parsed.hostname is None or parsed.hostname.casefold() != "github.com":
            raise ScopeError(f"not a GitHub repository: {value!r}")
        candidate = urllib.parse.unquote(parsed.path).strip("/")
    elif "/" in candidate:
        parsed = urllib.parse.urlparse(f"//{candidate}")
        if parsed.netloc.casefold() == "github.com":
            candidate = urllib.parse.unquote(parsed.path).strip("/")
    candidate = candidate.strip("/")
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    match = _OWNER_REPO_RE.fullmatch(candidate)
    if match is None:
        raise ScopeError(f"invalid GitHub owner/repository: {value!r}")
    owner = match.group("owner")
    name = match.group("repo")
    if name in {".", ".."}:
        raise ScopeError(f"invalid GitHub repository name: {value!r}")
    return owner, name


def load_repo_scope(
    path: str | os.PathLike[str],
    *,
    smoke: bool = False,
    max_repos: int | None = None,
) -> RepoScope:
    """Load and case-insensitively deduplicate the canonical GitHub repo scope."""

    source_path = Path(path).expanduser().resolve()
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise ScopeError(f"cannot read repository list {source_path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScopeError(f"repository list is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ScopeError("repository list root must be an object")

    unresolved = document.get("unresolved")
    if not isinstance(unresolved, list):
        raise ScopeError("repository list must contain an 'unresolved' array")
    if unresolved:
        raise ScopeError(
            f"repository list has {len(unresolved)} unresolved entries; "
            "production inventory requires zero"
        )
    names = document.get("repo_names")
    if not isinstance(names, list) or not names:
        raise ScopeError("repository list must contain a non-empty 'repo_names' array")

    deduplicated: dict[str, tuple[str, str]] = {}
    for index, item in enumerate(names):
        if not isinstance(item, str):
            raise ScopeError(f"repo_names[{index}] must be a string")
        owner, name = _normalize_owner_repo(item)
        key = f"{owner}/{name}".casefold()
        previous = deduplicated.get(key)
        if previous is not None:
            # GitHub names are case-insensitive.  Preserve the first spelling,
            # but reject a duplicate that is not actually the same identity.
            if (previous[0].casefold(), previous[1].casefold()) != (
                owner.casefold(),
                name.casefold(),
            ):
                raise ScopeError(f"ambiguous GitHub repository identity: {item!r}")
            continue
        deduplicated[key] = (owner, name)

    ordered_pairs = sorted(
        deduplicated.values(),
        key=lambda pair: (pair[0].casefold(), pair[1].casefold()),
    )
    original_repo_count = len(ordered_pairs)
    if max_repos is not None:
        if not smoke:
            raise ScopeError("--max-repos is allowed only with explicit --smoke")
        if max_repos <= 0:
            raise ScopeError("--max-repos must be positive")
        ordered_pairs = ordered_pairs[:max_repos]
    repos = tuple(
        Repo(
            key=f"{owner}/{name}".casefold(),
            owner=owner,
            name=name,
            canonical=f"{owner}/{name}",
            ordinal=ordinal,
        )
        for ordinal, (owner, name) in enumerate(ordered_pairs)
    )
    scope_hash = _hash_lines(repo.key for repo in repos)
    return RepoScope(
        path=str(source_path),
        source_sha256=_sha256_bytes(raw),
        scope_sha256=scope_hash,
        repos=repos,
        original_repo_count=original_repo_count,
        unresolved_count=0,
        smoke=smoke,
        max_repos=max_repos,
    )


def load_token_pool(
    token_file: str | os.PathLike[str] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Load newline-delimited tokens and append ``GH_TOKEN`` when present."""

    tokens: list[str] = []
    if token_file is not None:
        path = Path(token_file).expanduser()
        try:
            lines = path.read_text().splitlines()
        except OSError as exc:
            raise InventoryError(f"cannot read token pool {path}: {exc}") from exc
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                tokens.append(stripped)
    env = os.environ if environ is None else environ
    env_token = env.get("GH_TOKEN", "").strip()
    if env_token:
        tokens.append(env_token)
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    if not unique:
        raise InventoryError(
            "no GitHub tokens available; provide --tokens and/or GH_TOKEN"
        )
    return unique


class TokenPool:
    """Thread-safe token rotation driven by observed rate-limit state."""

    def __init__(
        self,
        tokens: Sequence[str],
        *,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        unique = list(dict.fromkeys(token.strip() for token in tokens if token.strip()))
        if not unique:
            raise ValueError("token pool must not be empty")
        self._states = [_TokenState(token=token) for token in unique]
        self._clock = clock
        self._sleeper = sleeper
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(state.token for state in self._states)

    def acquire(self) -> tuple[int, str]:
        with self._lock:
            now = self._clock()
            count = len(self._states)
            available = [
                (offset, self._states[(self._cursor + offset) % count])
                for offset in range(count)
                if self._states[(self._cursor + offset) % count].cooldown_until
                <= now
            ]
            if available:
                # The cursor provides fair rotation; cooldown state removes
                # exhausted tokens from the candidate set.
                offset, state = available[0]
                index = (self._cursor + offset) % count
                self._cursor = (index + 1) % count
                return index, state.token
            index = min(
                range(count), key=lambda item: self._states[item].cooldown_until
            )
            wait_seconds = max(0.0, self._states[index].cooldown_until - now)
        # Do not hold the pool lock while waiting.  A bounded sleep keeps
        # progress/status reporting responsive; the API response will enforce
        # the limit again if its reset time has not arrived.
        self._sleeper(min(wait_seconds, 60.0))
        with self._lock:
            self._cursor = (index + 1) % len(self._states)
            return index, self._states[index].token

    def observe(self, index: int, headers: Mapping[str, str]) -> None:
        lowered = {str(key).casefold(): str(value) for key, value in headers.items()}
        with self._lock:
            state = self._states[index]
            try:
                state.remaining = int(lowered["x-ratelimit-remaining"])
            except (KeyError, ValueError):
                pass
            try:
                state.reset_epoch = float(lowered["x-ratelimit-reset"])
            except (KeyError, ValueError):
                pass
            if state.remaining == 0:
                state.cooldown_until = max(
                    state.cooldown_until,
                    state.reset_epoch or self._clock() + 60.0,
                )

    def rate_limited(
        self,
        index: int,
        headers: Mapping[str, str],
        *,
        secondary: bool,
    ) -> None:
        lowered = {str(key).casefold(): str(value) for key, value in headers.items()}
        now = self._clock()
        retry_after = 0.0
        try:
            retry_after = max(0.0, float(lowered.get("retry-after", "0")))
        except ValueError:
            retry_after = 0.0
        try:
            reset = float(lowered.get("x-ratelimit-reset", "0"))
        except ValueError:
            reset = 0.0
        fallback = 60.0 if secondary else 5.0
        until = max(now + retry_after, reset, now + fallback)
        with self._lock:
            state = self._states[index]
            state.remaining = 0
            state.reset_epoch = reset
            state.cooldown_until = max(state.cooldown_until, until)


def _default_requester(
    method: str,
    url: str,
    headers: Mapping[str, str],
    timeout: float,
) -> HTTPResponse:
    request = urllib.request.Request(url, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
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


class GitHubClient:
    def __init__(
        self,
        token_pool: TokenPool,
        *,
        requester: Callable[
            [str, str, Mapping[str, str], float], HTTPResponse
        ] = _default_requester,
        sleeper: Callable[[float], None] = time.sleep,
        timeout: float = 60.0,
        max_attempts: int = 12,
        api_base: str = "https://api.github.com",
    ):
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.token_pool = token_pool
        self.requester = requester
        self.sleeper = sleeper
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.api_base = api_base.rstrip("/")

    def redact(self, message: object) -> str:
        text = str(message)
        for secret in self.token_pool.secrets:
            if secret:
                text = text.replace(secret, "<redacted>")
        return text[:4000]

    @staticmethod
    def _rate_headers(headers: Mapping[str, str]) -> dict[str, str | None]:
        lowered = {str(key).casefold(): str(value) for key, value in headers.items()}
        return {
            "rate_remaining": lowered.get("x-ratelimit-remaining"),
            "rate_reset": lowered.get("x-ratelimit-reset"),
            "retry_after": lowered.get("retry-after"),
        }

    @staticmethod
    def _body_message(body: bytes) -> str:
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body.decode("utf-8", errors="replace")[:1000]
        if isinstance(payload, dict):
            return str(payload.get("message") or payload)[:1000]
        return str(payload)[:1000]

    def get_workflow_runs(
        self,
        *,
        repo: Repo,
        start_epoch: int,
        end_epoch: int,
        page: int,
        per_page: int,
        ledger: Callable[..., None],
    ) -> PageResponse:
        if end_epoch <= start_epoch:
            raise ValueError("empty workflow-run search interval")
        created = (
            f"{format_utc_instant(start_epoch)}.."
            f"{format_utc_instant(end_epoch - 1)}"
        )
        endpoint = f"/repos/{repo.owner}/{repo.name}/actions/runs"
        query = urllib.parse.urlencode(
            {
                "created": created,
                "exclude_pull_requests": "false",
                "per_page": per_page,
                "page": page,
            }
        )
        url = f"{self.api_base}{endpoint}?{query}"

        for attempt in range(1, self.max_attempts + 1):
            token_index, token = self.token_pool.acquire()
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "cppmega-ci-stream-inventory/1",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            }
            started = time.monotonic()
            try:
                response = self.requester("GET", url, headers, self.timeout)
            except Exception as exc:
                elapsed = int((time.monotonic() - started) * 1000)
                message = self.redact(exc)
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=None,
                    outcome="transport_retry",
                    latency_ms=elapsed,
                    error_class=type(exc).__name__,
                    error_message=message,
                )
                if attempt == self.max_attempts:
                    raise APIError(
                        f"transport retries exhausted for {repo.canonical} "
                        f"window {created}: {message}"
                    ) from exc
                self.sleeper(min(2 ** (attempt - 1), 30))
                continue

            elapsed = int((time.monotonic() - started) * 1000)
            self.token_pool.observe(token_index, response.headers)
            rate = self._rate_headers(response.headers)
            body_message = self._body_message(response.body)
            message_lower = body_message.casefold()
            remaining_zero = rate["rate_remaining"] == "0"
            secondary = (
                "secondary rate limit" in message_lower
                or "abuse detection" in message_lower
            )
            rate_limited = response.status == 429 or (
                response.status == 403
                and (remaining_zero or secondary or "rate limit" in message_lower)
            )
            if rate_limited:
                self.token_pool.rate_limited(
                    token_index, response.headers, secondary=secondary
                )
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=response.status,
                    outcome="rate_limit_retry",
                    latency_ms=elapsed,
                    error_class="RateLimit",
                    error_message=self.redact(body_message),
                    **rate,
                )
                if attempt == self.max_attempts:
                    raise APIError(
                        f"rate-limit retries exhausted for {repo.canonical} "
                        f"window {created}"
                    )
                continue

            if response.status >= 500:
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=response.status,
                    outcome="server_retry",
                    latency_ms=elapsed,
                    error_class="GitHubServerError",
                    error_message=self.redact(body_message),
                    **rate,
                )
                if attempt == self.max_attempts:
                    raise APIError(
                        f"GitHub server retries exhausted for {repo.canonical}: "
                        f"HTTP {response.status}"
                    )
                self.sleeper(min(2 ** (attempt - 1), 30))
                continue

            if response.status != 200:
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=response.status,
                    outcome="permanent_error",
                    latency_ms=elapsed,
                    error_class="GitHubHTTPError",
                    error_message=self.redact(body_message),
                    **rate,
                )
                raise APIError(
                    f"GitHub HTTP {response.status} for {repo.canonical}: "
                    f"{self.redact(body_message)}"
                )

            try:
                payload = json.loads(response.body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=response.status,
                    outcome="malformed",
                    latency_ms=elapsed,
                    error_class=type(exc).__name__,
                    error_message=self.redact(exc),
                    **rate,
                )
                raise MalformedAPIError(
                    f"GitHub returned invalid JSON for {repo.canonical}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                problem = "response root is not an object"
            elif (
                isinstance(payload.get("total_count"), bool)
                or not isinstance(payload.get("total_count"), int)
                or int(payload["total_count"]) < 0
            ):
                problem = "total_count must be a non-negative integer"
            elif not isinstance(payload.get("workflow_runs"), list):
                problem = "workflow_runs must be an array"
            elif any(not isinstance(item, dict) for item in payload["workflow_runs"]):
                problem = "workflow_runs must contain objects"
            else:
                problem = ""
            if problem:
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=response.status,
                    outcome="malformed",
                    latency_ms=elapsed,
                    error_class="MalformedAPI",
                    error_message=problem,
                    **rate,
                )
                raise MalformedAPIError(
                    f"malformed GitHub response for {repo.canonical}: {problem}"
                )
            canonical_payload = {
                "total_count": int(payload["total_count"]),
                "workflow_runs": payload["workflow_runs"],
            }
            ledger(
                endpoint=endpoint,
                page=page,
                per_page=per_page,
                attempt=attempt,
                http_status=response.status,
                outcome="success",
                latency_ms=elapsed,
                error_class=None,
                error_message=None,
                **rate,
            )
            return PageResponse(
                total_count=int(payload["total_count"]),
                workflow_runs=tuple(dict(item) for item in payload["workflow_runs"]),
                payload_sha256=_sha256_json(canonical_payload),
            )
        raise AssertionError("unreachable retry loop")


_SCHEMA_SQL = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS inventory_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inventory_upgrades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_schema TEXT NOT NULL,
    to_schema TEXT NOT NULL,
    from_script_sha256 TEXT NOT NULL,
    to_script_sha256 TEXT NOT NULL,
    upgraded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inventory_binding_upgrades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_schema TEXT NOT NULL,
    to_schema TEXT NOT NULL,
    from_script_sha256 TEXT NOT NULL,
    to_script_sha256 TEXT NOT NULL,
    reason TEXT NOT NULL,
    upgraded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS repos (
    repo_key TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    canonical TEXT NOT NULL,
    ordinal INTEGER NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS search_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_key TEXT NOT NULL REFERENCES repos(repo_key),
    start_epoch INTEGER NOT NULL,
    end_epoch INTEGER NOT NULL,
    parent_id INTEGER REFERENCES search_windows(id),
    depth INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('open','fetching','split','done','failed')
    ),
    expected_total INTEGER,
    expected_pages INTEGER,
    pages_done INTEGER NOT NULL DEFAULT 0,
    raw_items INTEGER NOT NULL DEFAULT 0,
    distinct_items INTEGER NOT NULL DEFAULT 0,
    duplicate_items INTEGER NOT NULL DEFAULT 0,
    run_keys_sha256 TEXT,
    failure_class TEXT,
    failure_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(repo_key, start_epoch, end_epoch)
);
CREATE INDEX IF NOT EXISTS idx_windows_work
    ON search_windows(repo_key, status, start_epoch);
CREATE TABLE IF NOT EXISTS runs (
    repo_key TEXT NOT NULL REFERENCES repos(repo_key),
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
CREATE INDEX IF NOT EXISTS idx_runs_created
    ON runs(repo_key, created_at, run_id, run_attempt);
CREATE TABLE IF NOT EXISTS window_runs (
    window_id INTEGER NOT NULL REFERENCES search_windows(id),
    repo_key TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    run_attempt INTEGER NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    PRIMARY KEY(window_id, repo_key, run_id, run_attempt),
    FOREIGN KEY(repo_key, run_id, run_attempt)
        REFERENCES runs(repo_key, run_id, run_attempt)
);
CREATE INDEX IF NOT EXISTS idx_window_runs_identity
    ON window_runs(repo_key,run_id,run_attempt,window_id);
CREATE TABLE IF NOT EXISTS window_pages (
    window_id INTEGER NOT NULL REFERENCES search_windows(id),
    page_no INTEGER NOT NULL,
    total_count INTEGER NOT NULL,
    item_count INTEGER NOT NULL,
    distinct_item_count INTEGER NOT NULL,
    duplicate_item_count INTEGER NOT NULL,
    payload_sha256 TEXT NOT NULL,
    run_keys_sha256 TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY(window_id, page_no)
);
CREATE TABLE IF NOT EXISTS window_convergence (
    window_id INTEGER PRIMARY KEY REFERENCES search_windows(id),
    attempts INTEGER NOT NULL DEFAULT 0,
    candidate_total INTEGER,
    candidate_sha256 TEXT,
    stable_observations INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS convergence_passes (
    window_id INTEGER NOT NULL REFERENCES search_windows(id),
    pass_no INTEGER NOT NULL CHECK(pass_no >= 1),
    total_count INTEGER NOT NULL CHECK(total_count >= 0),
    page_count INTEGER NOT NULL CHECK(page_count >= 1),
    raw_item_count INTEGER NOT NULL CHECK(raw_item_count >= 0),
    distinct_item_count INTEGER NOT NULL CHECK(distinct_item_count >= 0),
    duplicate_item_count INTEGER NOT NULL CHECK(duplicate_item_count >= 0),
    page_payload_set_sha256 TEXT NOT NULL,
    run_keys_sha256 TEXT NOT NULL,
    accumulated_distinct_count INTEGER NOT NULL
        CHECK(accumulated_distinct_count >= 0),
    min_observation_count INTEGER NOT NULL CHECK(min_observation_count >= 0),
    observed_at TEXT NOT NULL,
    PRIMARY KEY(window_id, pass_no)
);
CREATE TABLE IF NOT EXISTS convergence_pass_pages (
    window_id INTEGER NOT NULL,
    pass_no INTEGER NOT NULL,
    page_no INTEGER NOT NULL CHECK(page_no >= 1),
    total_count INTEGER NOT NULL CHECK(total_count >= 0),
    item_count INTEGER NOT NULL CHECK(item_count >= 0),
    distinct_item_count INTEGER NOT NULL CHECK(distinct_item_count >= 0),
    duplicate_item_count INTEGER NOT NULL CHECK(duplicate_item_count >= 0),
    payload_sha256 TEXT NOT NULL,
    run_keys_sha256 TEXT NOT NULL,
    PRIMARY KEY(window_id, pass_no, page_no),
    FOREIGN KEY(window_id,pass_no)
        REFERENCES convergence_passes(window_id,pass_no)
);
CREATE TABLE IF NOT EXISTS convergence_runs (
    window_id INTEGER NOT NULL REFERENCES search_windows(id),
    repo_key TEXT NOT NULL REFERENCES repos(repo_key),
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
    first_pass INTEGER NOT NULL CHECK(first_pass >= 1),
    last_pass INTEGER NOT NULL CHECK(last_pass >= first_pass),
    observation_count INTEGER NOT NULL CHECK(observation_count >= 1),
    PRIMARY KEY(window_id, repo_key, run_id, run_attempt)
);
CREATE INDEX IF NOT EXISTS idx_convergence_runs_identity
    ON convergence_runs(repo_key,run_id,run_attempt,window_id);
CREATE TABLE IF NOT EXISTS convergence_pass_runs (
    window_id INTEGER NOT NULL,
    pass_no INTEGER NOT NULL,
    repo_key TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    run_attempt INTEGER NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    PRIMARY KEY(
        window_id,pass_no,repo_key,run_id,run_attempt
    ),
    FOREIGN KEY(window_id,pass_no)
        REFERENCES convergence_passes(window_id,pass_no),
    FOREIGN KEY(window_id,repo_key,run_id,run_attempt)
        REFERENCES convergence_runs(
            window_id,repo_key,run_id,run_attempt
        )
);
CREATE TABLE IF NOT EXISTS window_union_closures (
    window_id INTEGER PRIMARY KEY REFERENCES search_windows(id),
    total_count INTEGER NOT NULL CHECK(total_count >= 0),
    pass_count INTEGER NOT NULL CHECK(pass_count >= 2),
    first_pass_no INTEGER NOT NULL CHECK(first_pass_no >= 1),
    last_pass_no INTEGER NOT NULL CHECK(last_pass_no >= first_pass_no),
    observed_page_count INTEGER NOT NULL CHECK(observed_page_count >= 2),
    observed_item_count INTEGER NOT NULL CHECK(observed_item_count >= 0),
    distinct_run_count INTEGER NOT NULL CHECK(distinct_run_count >= 0),
    min_observation_count INTEGER NOT NULL CHECK(min_observation_count >= 2),
    pass_set_sha256 TEXT NOT NULL,
    run_keys_sha256 TEXT NOT NULL,
    closed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_drift_reconciliations (
    window_id INTEGER PRIMARY KEY REFERENCES search_windows(id),
    proof_blob BLOB NOT NULL,
    proof_raw_size INTEGER NOT NULL CHECK(proof_raw_size >= 0),
    proof_sha256 TEXT NOT NULL,
    producer_script_sha256 TEXT NOT NULL,
    completed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS request_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    requested_at TEXT NOT NULL,
    repo_key TEXT NOT NULL,
    window_id INTEGER NOT NULL,
    endpoint TEXT NOT NULL,
    page_no INTEGER NOT NULL,
    per_page INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    http_status INTEGER,
    outcome TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    rate_remaining TEXT,
    rate_reset TEXT,
    retry_after TEXT,
    error_class TEXT,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_request_ledger_window
    ON request_ledger(window_id, id);
"""

_SQLITE_SCHEMA_OBJECT_TYPES = frozenset(
    {"index", "table", "trigger", "view"}
)
_SQLITE_INTERNAL_SCHEMA_OBJECTS = frozenset(
    {
        (
            "index",
            "sqlite_autoindex_convergence_pass_pages_1",
            "convergence_pass_pages",
            None,
        ),
        (
            "index",
            "sqlite_autoindex_convergence_pass_runs_1",
            "convergence_pass_runs",
            None,
        ),
        (
            "index",
            "sqlite_autoindex_convergence_passes_1",
            "convergence_passes",
            None,
        ),
        (
            "index",
            "sqlite_autoindex_convergence_runs_1",
            "convergence_runs",
            None,
        ),
        (
            "index",
            "sqlite_autoindex_inventory_meta_1",
            "inventory_meta",
            None,
        ),
        (
            "index",
            "sqlite_autoindex_repos_1",
            "repos",
            None,
        ),
        (
            "index",
            "sqlite_autoindex_repos_2",
            "repos",
            None,
        ),
        (
            "index",
            "sqlite_autoindex_runs_1",
            "runs",
            None,
        ),
        (
            "index",
            "sqlite_autoindex_search_windows_1",
            "search_windows",
            None,
        ),
        (
            "index",
            "sqlite_autoindex_window_pages_1",
            "window_pages",
            None,
        ),
        (
            "index",
            "sqlite_autoindex_window_runs_1",
            "window_runs",
            None,
        ),
        (
            "table",
            "sqlite_sequence",
            "sqlite_sequence",
            "CREATE TABLE sqlite_sequence(name,seq)",
        ),
    }
)


def _sqlite_schema_rows(
    conn: sqlite3.Connection,
) -> tuple[tuple[str, str, str, str | None], ...]:
    rows: list[tuple[str, str, str, str | None]] = []
    for raw_row in conn.execute(
        """
        SELECT type,name,tbl_name,sql
        FROM sqlite_schema
        ORDER BY type,name,tbl_name,sql
        """
    ):
        object_type, name, table_name, sql = tuple(raw_row)
        if (
            not isinstance(object_type, str)
            or object_type not in _SQLITE_SCHEMA_OBJECT_TYPES
            or not isinstance(name, str)
            or not name
            or not isinstance(table_name, str)
            or not table_name
            or (sql is not None and not isinstance(sql, str))
        ):
            raise CompletionError(
                "inventory SQLite schema contains a malformed object"
            )
        row = (object_type, name, table_name, sql)
        if name.startswith("sqlite_") and (
            row not in _SQLITE_INTERNAL_SCHEMA_OBJECTS
        ):
            raise CompletionError(
                "inventory SQLite schema contains an unauthorized internal "
                f"object {object_type}:{name}"
            )
        rows.append(row)
    return tuple(rows)


def _build_expected_sqlite_schema_rows(
) -> tuple[tuple[str, str, str, str | None], ...]:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(_SCHEMA_SQL)
        rows = _sqlite_schema_rows(conn)
    finally:
        conn.close()
    observed_internal = frozenset(
        row for row in rows if row[1].startswith("sqlite_")
    )
    if observed_internal != _SQLITE_INTERNAL_SCHEMA_OBJECTS:
        raise RuntimeError(
            "versioned inventory schema internal-object policy is stale"
        )
    return rows


_EXPECTED_SQLITE_SCHEMA_ROWS = _build_expected_sqlite_schema_rows()
_EXPECTED_SQLITE_SCHEMA_SHA256 = _sha256_json(
    [
        {
            "type": object_type,
            "name": name,
            "tbl_name": table_name,
            "sql": sql,
        }
        for object_type, name, table_name, sql in _EXPECTED_SQLITE_SCHEMA_ROWS
    ]
)
_EXPECTED_INVENTORY_META_KEYS = frozenset(
    {
        "schema",
        "repo_list_path",
        "repo_list_sha256",
        "repo_scope_sha256",
        "repo_count",
        "original_repo_count",
        "unresolved_count",
        "start_epoch",
        "end_epoch",
        "start_utc",
        "end_utc",
        "script_sha256",
        "metadata_encoding",
        "smoke",
        "max_repos",
        "created_at",
    }
)
_LOGICAL_TABLE_ORDER = (
    ("inventory_meta", "key"),
    ("inventory_upgrades", "id"),
    ("inventory_binding_upgrades", "id"),
    ("repos", "ordinal,repo_key"),
    ("search_windows", "id"),
    ("runs", "repo_key,run_id,run_attempt"),
    ("window_runs", "window_id,repo_key,run_id,run_attempt"),
    ("window_pages", "window_id,page_no"),
    ("window_convergence", "window_id"),
    ("convergence_passes", "window_id,pass_no"),
    ("convergence_pass_pages", "window_id,pass_no,page_no"),
    (
        "convergence_runs",
        "window_id,repo_key,run_id,run_attempt",
    ),
    (
        "convergence_pass_runs",
        "window_id,pass_no,repo_key,run_id,run_attempt",
    ),
    ("window_union_closures", "window_id"),
    ("source_drift_reconciliations", "window_id"),
    ("request_ledger", "id"),
    ("sqlite_sequence", "name"),
)
_AUTOINCREMENT_TABLES = frozenset(
    {
        "inventory_upgrades",
        "inventory_binding_upgrades",
        "search_windows",
        "request_ledger",
    }
)
_REQUEST_OUTCOMES = frozenset(
    {
        "transport_retry",
        "rate_limit_retry",
        "server_retry",
        "permanent_error",
        "malformed",
        "success",
        "pagination_drift_split",
        "pagination_drift_converge",
        "window_error",
    }
)
_REQUEST_SYNTHETIC_OUTCOMES = frozenset(
    {
        "pagination_drift_split",
        "pagination_drift_converge",
        "window_error",
    }
)
_EXPECTED_SQLITE_TABLE_NAMES = frozenset(
    row[1]
    for row in _EXPECTED_SQLITE_SCHEMA_ROWS
    if row[0] == "table"
)
if (
    frozenset(table for table, _order_by in _LOGICAL_TABLE_ORDER)
    != _EXPECTED_SQLITE_TABLE_NAMES
):
    raise RuntimeError(
        "inventory logical-table coverage does not match the exact schema"
    )


class InventoryDB:
    """SQLite state and fail-closed completion validation."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        initialize_schema: bool = True,
    ):
        self.path = str(Path(path).expanduser().resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        if initialize_schema:
            database_path = Path(self.path)
            existing = (
                database_path.exists()
                and database_path.stat().st_size > 0
            )
            conn = (
                sqlite3.connect(self.path, timeout=60.0)
                if existing
                else self.connect()
            )
            try:
                _constrain_inventory_connection(conn)
                if existing:
                    conn.execute("PRAGMA busy_timeout=60000")
                    conn.execute("PRAGMA synchronous=FULL")
                    conn.execute("PRAGMA foreign_keys=ON")
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def connect(
        self,
        *,
        readonly: bool = False,
        immutable: bool = False,
    ) -> sqlite3.Connection:
        if immutable and not readonly:
            raise ValueError("immutable inventory connections are read-only")
        if readonly:
            uri = f"{Path(self.path).as_uri()}?mode=ro"
            if immutable:
                uri += "&immutable=1"
            conn = sqlite3.connect(uri, uri=True, timeout=60.0)
        else:
            conn = sqlite3.connect(self.path, timeout=60.0)
        try:
            _constrain_inventory_connection(conn)
        except BaseException:
            conn.close()
            raise
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=60000")
        if readonly:
            conn.execute("PRAGMA query_only=ON")
        else:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _freeze_for_receipt(self) -> None:
        path = Path(self.path)
        if path.is_symlink() or not path.is_file():
            raise CompletionError(f"inventory database is missing or unsafe: {path}")
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                path,
                isolation_level=None,
                timeout=5.0,
            )
            _constrain_inventory_connection(connection)
            connection.execute("PRAGMA busy_timeout=5000")
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
                    raise CompletionError(
                        f"inventory WAL checkpoint is busy: {checkpoint}"
                    )
                mode_row = connection.execute(
                    "PRAGMA journal_mode=DELETE"
                ).fetchone()
                mode = "" if mode_row is None else str(mode_row[0]).lower()
            if mode not in {"delete", "wal"}:
                raise CompletionError(
                    f"inventory journal mode is unsupported: {mode!r}"
                )
        except sqlite3.Error as exc:
            raise CompletionError(
                f"inventory database could not be frozen: {exc}"
            ) from exc
        finally:
            if connection is not None:
                connection.close()
        if mode != "delete":
            raise CompletionError(
                "inventory database did not enter DELETE journal mode"
            )
        # The WAL was truncated and the DELETE transition obtained SQLite's
        # exclusive journal-mode lock.  Only after that connection is closed is
        # it safe to remove stale zero-WAL/SHM artifacts copied with a private
        # merge destination.  Receipt hashing happens after this header change.
        removed_sidecar = False
        wal = Path(f"{path}-wal")
        if wal.is_symlink():
            raise CompletionError(
                "inventory database checkpoint left an unsafe "
                f"{wal.name}"
            )
        if wal.exists():
            wal_stat = wal.stat()
            if not wal.is_file() or wal_stat.st_size != 0:
                raise CompletionError(
                    "inventory database checkpoint left a non-empty/unsafe "
                    f"{wal.name}"
                )
            wal.unlink()
            removed_sidecar = True
        journal = Path(f"{path}-journal")
        if journal.exists() or journal.is_symlink():
            raise CompletionError(
                "inventory database checkpoint left an unsafe rollback "
                f"journal: {journal.name}"
            )
        shm = Path(f"{path}-shm")
        if shm.is_symlink() or (shm.exists() and not shm.is_file()):
            raise CompletionError(
                "inventory database checkpoint left an unsafe "
                f"{shm.name}"
            )
        if shm.exists():
            shm.unlink()
            removed_sidecar = True
        if removed_sidecar:
            _fsync_directory(path.parent)

    @staticmethod
    def _meta(conn: sqlite3.Connection) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in conn.execute("SELECT key,value FROM inventory_meta")
        }

    @staticmethod
    def _backfill_binding_upgrade_history_locked(
        conn: sqlite3.Connection,
        *,
        current_schema: str,
        current_script_sha256: str,
    ) -> None:
        legacy = list(
            conn.execute(
                """
                SELECT from_schema,to_schema,from_script_sha256,
                       to_script_sha256,upgraded_at
                FROM inventory_upgrades ORDER BY id
                """
            )
        )
        binding = list(
            conn.execute(
                """
                SELECT from_schema,to_schema,from_script_sha256,
                       to_script_sha256,reason,upgraded_at
                FROM inventory_binding_upgrades ORDER BY id
                """
            )
        )
        projected = [
            (
                str(row["from_schema"]),
                str(row["to_schema"]),
                str(row["from_script_sha256"]),
                str(row["to_script_sha256"]),
                str(row["upgraded_at"]),
            )
            for row in binding
        ]
        legacy_values = [
            (
                str(row["from_schema"]),
                str(row["to_schema"]),
                str(row["from_script_sha256"]),
                str(row["to_script_sha256"]),
                str(row["upgraded_at"]),
            )
            for row in legacy
        ]
        if binding:
            if projected != legacy_values:
                raise BindingError(
                    "inventory producer upgrade ledgers disagree before "
                    "migration"
                )
            for row in binding:
                _validate_upgrade_reason(str(row["reason"]))
        elif legacy_values:
            conn.executemany(
                """
                INSERT INTO inventory_binding_upgrades(
                    from_schema,to_schema,from_script_sha256,
                    to_script_sha256,reason,upgraded_at
                ) VALUES (?,?,?,?,?,?)
                """,
                [
                    (*row[:4], IMPORTED_UPGRADE_REASON, row[4])
                    for row in legacy_values
                ],
            )
        if legacy_values:
            for index, row in enumerate(legacy_values):
                if index and (
                    legacy_values[index - 1][1] != row[0]
                    or legacy_values[index - 1][3] != row[2]
                ):
                    raise BindingError(
                        "legacy inventory producer upgrade chain is broken"
                    )
            if (
                legacy_values[-1][1] != current_schema
                or legacy_values[-1][3] != current_script_sha256
            ):
                raise BindingError(
                    "legacy inventory producer upgrade chain does not bind "
                    "the current database producer"
                )

    def bind(
        self,
        *,
        scope: RepoScope,
        start_epoch: int,
        end_epoch: int,
        script_sha256: str,
        resume: bool,
        allow_script_upgrade_from_sha256: str | None = None,
        script_upgrade_reason: str | None = None,
    ) -> None:
        if start_epoch >= end_epoch:
            raise BindingError("inventory interval must satisfy start < end")
        expected = {
            "schema": SCHEMA_VERSION,
            "repo_list_path": scope.path,
            "repo_list_sha256": scope.source_sha256,
            "repo_scope_sha256": scope.scope_sha256,
            "repo_count": str(len(scope.repos)),
            "original_repo_count": str(scope.original_repo_count),
            "unresolved_count": str(scope.unresolved_count),
            "start_epoch": str(start_epoch),
            "end_epoch": str(end_epoch),
            "start_utc": format_utc_instant(start_epoch),
            "end_utc": format_utc_instant(end_epoch),
            "script_sha256": script_sha256,
            "metadata_encoding": METADATA_ENCODING,
            "smoke": "1" if scope.smoke else "0",
            "max_repos": "" if scope.max_repos is None else str(scope.max_repos),
        }
        conn = self.connect()
        try:
            with self._write_lock, conn:
                current = self._meta(conn)
                if current:
                    if not resume:
                        raise BindingError(
                            f"inventory database already exists at {self.path}; "
                            "pass --resume after verifying its binding"
                        )
                    current_schema = current.get("schema")
                    previous_script = current.get("script_sha256", "")
                    upgrade_v1 = (
                        current_schema == "cppmega_ci_stream_inventory_v1"
                    )
                    schema_upgrade = current_schema in {
                        LEGACY_SCHEMA_VERSION,
                        PREVIOUS_SCHEMA_VERSION,
                    }
                    same_schema_upgrade = (
                        current_schema == SCHEMA_VERSION
                        and previous_script != script_sha256
                        and (
                            allow_script_upgrade_from_sha256 is not None
                            or script_upgrade_reason is not None
                        )
                    )
                    upgrade_reason: str | None = None
                    if schema_upgrade:
                        if (
                            allow_script_upgrade_from_sha256
                            != previous_script
                        ):
                            raise BindingError(
                                "inventory schema migration requires "
                                "--allow-inventory-script-upgrade-from-sha256 "
                                "to match the exact bound producer"
                            )
                        upgrade_reason = _validate_upgrade_reason(
                            script_upgrade_reason
                        )
                    elif same_schema_upgrade:
                        if (
                            allow_script_upgrade_from_sha256
                            != previous_script
                        ):
                            raise BindingError(
                                "same-schema inventory producer migration "
                                "requires "
                                "--allow-inventory-script-upgrade-from-sha256 "
                                "to match the exact bound producer"
                            )
                        upgrade_reason = _validate_upgrade_reason(
                            script_upgrade_reason
                        )
                    elif (
                        allow_script_upgrade_from_sha256 is not None
                        or script_upgrade_reason is not None
                    ):
                        repeated_reason = _validate_upgrade_reason(
                            script_upgrade_reason
                        )
                        repeated_upgrade = conn.execute(
                            """
                            SELECT from_schema,to_schema,from_script_sha256,
                                   to_script_sha256,reason
                            FROM inventory_binding_upgrades
                            ORDER BY id DESC LIMIT 1
                            """
                        ).fetchone()
                        if (
                            current_schema != SCHEMA_VERSION
                            or previous_script != script_sha256
                            or repeated_upgrade is None
                            or str(repeated_upgrade["to_schema"])
                            != SCHEMA_VERSION
                            or str(
                                repeated_upgrade[
                                    "from_script_sha256"
                                ]
                            )
                            != allow_script_upgrade_from_sha256
                            or str(
                                repeated_upgrade["to_script_sha256"]
                            )
                            != script_sha256
                            or str(repeated_upgrade["reason"])
                            != repeated_reason
                        ):
                            raise BindingError(
                                "inventory script upgrade authorization does "
                                "not exactly replay the latest completed "
                                "producer migration"
                            )
                    ignored_upgrade_keys = (
                        {"schema", "script_sha256"}
                        if upgrade_v1 or schema_upgrade
                        else {"script_sha256"}
                        if same_schema_upgrade
                        else set()
                    )
                    mismatches = {
                        key: (current.get(key), value)
                        for key, value in expected.items()
                        if key not in ignored_upgrade_keys
                        and current.get(key) != value
                    }
                    if mismatches:
                        rendered = ", ".join(
                            f"{key}={old!r}->{new!r}"
                            for key, (old, new) in sorted(mismatches.items())
                        )
                        raise BindingError(
                            f"resume binding mismatch in {self.path}: {rendered}"
                        )
                    if upgrade_v1 or schema_upgrade or same_schema_upgrade:
                        self._backfill_binding_upgrade_history_locked(
                            conn,
                            current_schema=str(current_schema),
                            current_script_sha256=previous_script,
                        )
                        reason = (
                            "audited legacy inventory v1 recovery migration"
                            if upgrade_v1
                            else upgrade_reason
                        )
                        assert reason is not None
                        upgraded_at = _utc_now()
                        conn.execute(
                            """
                            INSERT INTO inventory_upgrades(
                                from_schema,to_schema,from_script_sha256,
                                to_script_sha256,upgraded_at
                            ) VALUES (?,?,?,?,?)
                            """,
                            (
                                current["schema"],
                                SCHEMA_VERSION,
                                previous_script,
                                script_sha256,
                                upgraded_at,
                            ),
                        )
                        conn.execute(
                            """
                            INSERT INTO inventory_binding_upgrades(
                                from_schema,to_schema,from_script_sha256,
                                to_script_sha256,reason,upgraded_at
                            ) VALUES (?,?,?,?,?,?)
                            """,
                            (
                                current["schema"],
                                SCHEMA_VERSION,
                                previous_script,
                                script_sha256,
                                reason,
                                upgraded_at,
                            ),
                        )
                        conn.execute(
                            """
                            UPDATE inventory_meta SET value=?
                            WHERE key='schema'
                            """,
                            (SCHEMA_VERSION,),
                        )
                        conn.execute(
                            """
                            UPDATE inventory_meta SET value=?
                            WHERE key='script_sha256'
                            """,
                            (script_sha256,),
                        )
                    elif current.get("script_sha256") != script_sha256:
                        raise BindingError(
                            "resume script hash mismatch; no authorized "
                            "inventory producer migration applies"
                        )
                else:
                    if (
                        allow_script_upgrade_from_sha256 is not None
                        or script_upgrade_reason is not None
                    ):
                        raise BindingError(
                            "inventory script upgrade authorization cannot be "
                            "used when creating a new database"
                        )
                    conn.executemany(
                        "INSERT INTO inventory_meta(key,value) VALUES (?,?)",
                        sorted(expected.items()),
                    )
                    conn.execute(
                        "INSERT INTO inventory_meta(key,value) VALUES ('created_at',?)",
                        (_utc_now(),),
                    )
                    conn.executemany(
                        """
                        INSERT INTO repos(repo_key,owner,name,canonical,ordinal)
                        VALUES (?,?,?,?,?)
                        """,
                        [
                            (
                                repo.key,
                                repo.owner,
                                repo.name,
                                repo.canonical,
                                repo.ordinal,
                            )
                            for repo in scope.repos
                        ],
                    )
                    now = _utc_now()
                    conn.executemany(
                        """
                        INSERT INTO search_windows(
                            repo_key,start_epoch,end_epoch,parent_id,depth,status,
                            created_at,updated_at
                        ) VALUES (?,?,?,NULL,0,'open',?,?)
                        """,
                        [
                            (repo.key, start_epoch, end_epoch, now, now)
                            for repo in scope.repos
                        ],
                    )
                database_repos = {
                    str(row["repo_key"]): (
                        str(row["owner"]),
                        str(row["name"]),
                        int(row["ordinal"]),
                    )
                    for row in conn.execute(
                        "SELECT repo_key,owner,name,ordinal FROM repos"
                    )
                }
                expected_repos = {
                    repo.key: (repo.owner, repo.name, repo.ordinal)
                    for repo in scope.repos
                }
                if database_repos != expected_repos:
                    raise BindingError(
                        "database repository scope is not exactly the canonical scope"
                    )
                if resume:
                    # A page transaction is atomic.  Retrying a failed window
                    # therefore resumes at its first absent page without loss.
                    conn.execute(
                        """
                        UPDATE search_windows
                        SET status=CASE
                              WHEN expected_total IS NULL THEN 'open'
                              ELSE 'fetching'
                            END,
                            failure_class=NULL,
                            failure_message=NULL,
                            updated_at=?
                        WHERE status='failed'
                        """,
                        (_utc_now(),),
                    )
                    conn.execute(
                        """
                        UPDATE search_windows
                        SET status='fetching',updated_at=?
                        WHERE id IN (SELECT window_id FROM window_convergence)
                        """,
                        (_utc_now(),),
                    )
        finally:
            conn.close()

    def record_request(
        self,
        conn: sqlite3.Connection,
        *,
        repo_key: str,
        window_id: int,
        endpoint: str,
        page: int,
        per_page: int,
        attempt: int,
        http_status: int | None,
        outcome: str,
        latency_ms: int,
        rate_remaining: str | None = None,
        rate_reset: str | None = None,
        retry_after: str | None = None,
        error_class: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._write_lock, conn:
            conn.execute(
                """
                INSERT INTO request_ledger(
                    requested_at,repo_key,window_id,endpoint,page_no,per_page,
                    attempt,http_status,outcome,latency_ms,rate_remaining,
                    rate_reset,retry_after,error_class,error_message
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    _utc_now(),
                    repo_key,
                    window_id,
                    endpoint,
                    page,
                    per_page,
                    attempt,
                    http_status,
                    outcome,
                    latency_ms,
                    rate_remaining,
                    rate_reset,
                    retry_after,
                    error_class,
                    error_message,
                ),
            )

    @staticmethod
    def _run_int(
        run: Mapping[str, Any],
        field: str,
        *,
        required: bool = False,
        minimum: int | None = None,
    ) -> int | None:
        value = run.get(field)
        if value is None and not required:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise MalformedAPIError(f"workflow run {field!r} must be an integer")
        if minimum is not None and value < minimum:
            raise MalformedAPIError(
                f"workflow run {field!r} must be >= {minimum}, got {value}"
            )
        return value

    @staticmethod
    def _run_text(run: Mapping[str, Any], field: str) -> str | None:
        value = run.get(field)
        if value is None:
            return None
        if not isinstance(value, str):
            raise MalformedAPIError(f"workflow run {field!r} must be text or null")
        return value

    def _normalize_run(
        self,
        repo_key: str,
        run: dict[str, Any],
        *,
        start_epoch: int,
        end_epoch: int,
    ) -> tuple[dict[str, Any], str, tuple[int, int]]:
        run_id = self._run_int(run, "id", required=True, minimum=1)
        assert run_id is not None
        attempt_value = run.get("run_attempt")
        if attempt_value is None:
            run_attempt = 0
        else:
            parsed_attempt = self._run_int(
                run, "run_attempt", required=True, minimum=1
            )
            assert parsed_attempt is not None
            run_attempt = parsed_attempt
        created_at = self._run_text(run, "created_at")
        if not created_at:
            raise MalformedAPIError("workflow run is missing created_at")
        try:
            created_epoch = parse_utc_instant(created_at)
        except ValueError as exc:
            raise MalformedAPIError(
                f"workflow run {run_id} has invalid created_at: {exc}"
            ) from exc
        if not start_epoch <= created_epoch < end_epoch:
            raise UnstableEnumerationError(
                f"workflow run {run_id} created_at={created_at} is outside "
                f"[{format_utc_instant(start_epoch)},"
                f"{format_utc_instant(end_epoch)})"
            )
        status = self._run_text(run, "status")
        conclusion = self._run_text(run, "conclusion")
        metadata_json = _canonical_json(run)
        metadata_bytes = metadata_json.encode("utf-8")
        if len(metadata_bytes) > MAX_RUN_METADATA_BYTES:
            raise MalformedAPIError(
                "workflow run metadata exceeds the versioned raw-byte limit"
            )
        metadata_sha = _sha256_bytes(metadata_bytes)
        metadata_blob = zlib.compress(metadata_bytes, level=6)
        if len(metadata_blob) > MAX_RUN_METADATA_COMPRESSED_BYTES:
            raise MalformedAPIError(
                "workflow run metadata exceeds the versioned compressed-byte "
                "limit"
            )
        normalized = {
            "repo_key": repo_key,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "created_at": created_at,
            "updated_at": self._run_text(run, "updated_at"),
            "run_started_at": self._run_text(run, "run_started_at"),
            "status": status,
            "conclusion": conclusion,
            "workflow_id": self._run_int(run, "workflow_id"),
            "workflow_name": self._run_text(run, "name"),
            "event": self._run_text(run, "event"),
            "head_branch": self._run_text(run, "head_branch"),
            "head_sha": self._run_text(run, "head_sha"),
            "run_number": self._run_int(run, "run_number"),
            "html_url": self._run_text(run, "html_url"),
            "api_url": self._run_text(run, "url"),
            "metadata_blob": sqlite3.Binary(metadata_blob),
            "metadata_sha256": metadata_sha,
        }
        return normalized, metadata_sha, (run_id, run_attempt)

    def split_window(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        observed_total: int,
    ) -> None:
        start = int(row["start_epoch"])
        end = int(row["end_epoch"])
        if end - start <= 1:
            raise UnstableEnumerationError(
                f"{row['repo_key']} has {observed_total} workflow runs in the "
                f"unsplittable one-second interval "
                f"[{format_utc_instant(start)},{format_utc_instant(end)})"
            )
        midpoint = start + (end - start) // 2
        now = _utc_now()
        with self._write_lock, conn:
            current = conn.execute(
                "SELECT status,expected_total FROM search_windows WHERE id=?",
                (int(row["id"]),),
            ).fetchone()
            if current is None:
                raise UnstableEnumerationError("search window disappeared")
            if current["status"] == "split":
                if int(current["expected_total"]) != observed_total:
                    raise UnstableEnumerationError(
                        "split-window total changed across resume"
                    )
                return
            if current["status"] not in {"open", "fetching"}:
                raise UnstableEnumerationError(
                    f"cannot split window in status {current['status']!r}"
                )
            conn.execute(
                """
                UPDATE search_windows
                SET status='split',expected_total=?,expected_pages=NULL,
                    updated_at=?
                WHERE id=?
                """,
                (observed_total, now, int(row["id"])),
            )
            conn.executemany(
                """
                INSERT OR IGNORE INTO search_windows(
                    repo_key,start_epoch,end_epoch,parent_id,depth,status,
                    created_at,updated_at
                ) VALUES (?,?,?,?,?,'open',?,?)
                """,
                [
                    (
                        str(row["repo_key"]),
                        start,
                        midpoint,
                        int(row["id"]),
                        int(row["depth"]) + 1,
                        now,
                        now,
                    ),
                    (
                        str(row["repo_key"]),
                        midpoint,
                        end,
                        int(row["id"]),
                        int(row["depth"]) + 1,
                        now,
                        now,
                    ),
                ],
            )

    def _clear_window_payload_locked(
        self, conn: sqlite3.Connection, *, window_id: int
    ) -> None:
        keys = [
            (str(row["repo_key"]), int(row["run_id"]), int(row["run_attempt"]))
            for row in conn.execute(
                """
                SELECT repo_key,run_id,run_attempt
                FROM window_runs WHERE window_id=?
                """,
                (window_id,),
            )
        ]
        conn.execute("DELETE FROM window_pages WHERE window_id=?", (window_id,))
        conn.execute("DELETE FROM window_runs WHERE window_id=?", (window_id,))
        for repo_key, run_id, run_attempt in keys:
            conn.execute(
                """
                DELETE FROM runs
                WHERE repo_key=? AND run_id=? AND run_attempt=?
                  AND NOT EXISTS (
                      SELECT 1 FROM window_runs wr
                      WHERE wr.repo_key=runs.repo_key
                        AND wr.run_id=runs.run_id
                        AND wr.run_attempt=runs.run_attempt
                  )
                """,
                (repo_key, run_id, run_attempt),
            )

    @staticmethod
    def _clear_convergence_proof_locked(
        conn: sqlite3.Connection, *, window_id: int
    ) -> None:
        conn.execute(
            "DELETE FROM window_union_closures WHERE window_id=?",
            (window_id,),
        )
        conn.execute(
            "DELETE FROM convergence_pass_runs WHERE window_id=?",
            (window_id,),
        )
        conn.execute(
            "DELETE FROM convergence_pass_pages WHERE window_id=?",
            (window_id,),
        )
        conn.execute(
            "DELETE FROM convergence_passes WHERE window_id=?",
            (window_id,),
        )
        conn.execute(
            "DELETE FROM convergence_runs WHERE window_id=?",
            (window_id,),
        )

    def recover_pagination_drift(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        observed_total: int,
        reason: str,
    ) -> str:
        """Invalidate an unstable leaf and atomically split or converge it."""

        window_id = int(row["id"])
        start = int(row["start_epoch"])
        end = int(row["end_epoch"])
        now = _utc_now()
        with self._write_lock, conn:
            current = conn.execute(
                "SELECT * FROM search_windows WHERE id=?", (window_id,)
            ).fetchone()
            if current is None:
                raise UnstableEnumerationError("search window disappeared")
            if current["status"] not in {"open", "fetching", "failed", "done"}:
                raise UnstableEnumerationError(
                    f"cannot recover window in status {current['status']!r}"
                )
            self._clear_window_payload_locked(conn, window_id=window_id)
            self._clear_convergence_proof_locked(
                conn, window_id=window_id
            )
            if end - start > 1:
                midpoint = start + (end - start) // 2
                conn.execute(
                    "DELETE FROM window_convergence WHERE window_id=?", (window_id,)
                )
                conn.execute(
                    """
                    UPDATE search_windows
                    SET status='split',expected_total=?,expected_pages=NULL,
                        pages_done=0,raw_items=0,distinct_items=0,
                        duplicate_items=0,run_keys_sha256=NULL,
                        failure_class=NULL,failure_message=NULL,updated_at=?
                    WHERE id=?
                    """,
                    (observed_total, now, window_id),
                )
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO search_windows(
                        repo_key,start_epoch,end_epoch,parent_id,depth,status,
                        created_at,updated_at
                    ) VALUES (?,?,?,?,?,'open',?,?)
                    """,
                    [
                        (
                            str(row["repo_key"]),
                            start,
                            midpoint,
                            window_id,
                            int(row["depth"]) + 1,
                            now,
                            now,
                        ),
                        (
                            str(row["repo_key"]),
                            midpoint,
                            end,
                            window_id,
                            int(row["depth"]) + 1,
                            now,
                            now,
                        ),
                    ],
                )
                return "split"

            if observed_total > GITHUB_FILTER_LIMIT:
                raise UnstableEnumerationError(
                    f"{row['repo_key']} has {observed_total} runs in one second; "
                    "the repository endpoint cannot prove a complete set above "
                    f"{GITHUB_FILTER_LIMIT}"
                )
            conn.execute(
                """
                UPDATE search_windows
                SET status='fetching',expected_total=?,expected_pages=?,
                    pages_done=0,raw_items=0,distinct_items=0,
                    duplicate_items=0,run_keys_sha256=NULL,
                    failure_class=NULL,failure_message=NULL,updated_at=?
                WHERE id=?
                """,
                (
                    observed_total,
                    max(1, math.ceil(observed_total / DEFAULT_PER_PAGE)),
                    now,
                    window_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO window_convergence(
                    window_id,attempts,candidate_total,candidate_sha256,
                    stable_observations,last_error,updated_at
                ) VALUES (?,0,NULL,NULL,0,?,?)
                ON CONFLICT(window_id) DO UPDATE SET
                    candidate_total=NULL,
                    candidate_sha256=NULL,
                    stable_observations=0,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at
                """,
                (window_id, reason[:4000], now),
            )
            return "converge"

    def convergence_state(
        self, conn: sqlite3.Connection, window_id: int
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM window_convergence WHERE window_id=?", (window_id,)
        ).fetchone()

    def prepare_convergence(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> None:
        window_id = int(row["id"])
        with self._write_lock, conn:
            state = conn.execute(
                "SELECT 1 FROM window_convergence WHERE window_id=?", (window_id,)
            ).fetchone()
            if state is None:
                raise UnstableEnumerationError(
                    f"window {window_id} lost convergence state"
                )
            self._clear_window_payload_locked(conn, window_id=window_id)
            conn.execute(
                """
                UPDATE search_windows
                SET status='fetching',pages_done=0,raw_items=0,
                    distinct_items=0,duplicate_items=0,run_keys_sha256=NULL,
                    failure_class=NULL,failure_message=NULL,updated_at=?
                WHERE id=?
                """,
                (_utc_now(), window_id),
            )

    @staticmethod
    def _convergence_pass_set_sha256(
        conn: sqlite3.Connection, *, window_id: int
    ) -> str:
        return _hash_lines(
            "\t".join(
                str(row[field])
                for field in (
                    "pass_no",
                    "total_count",
                    "page_count",
                    "raw_item_count",
                    "distinct_item_count",
                    "duplicate_item_count",
                    "page_payload_set_sha256",
                    "run_keys_sha256",
                    "accumulated_distinct_count",
                    "min_observation_count",
                )
            )
            for row in conn.execute(
                """
                SELECT pass_no,total_count,page_count,raw_item_count,
                       distinct_item_count,duplicate_item_count,
                       page_payload_set_sha256,run_keys_sha256,
                       accumulated_distinct_count,min_observation_count
                FROM convergence_passes
                WHERE window_id=?
                ORDER BY pass_no
                """,
                (window_id,),
            )
        )

    def accumulate_convergence_pass(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        pages: Sequence[PageResponse],
    ) -> tuple[bool, str | None]:
        """Accumulate one complete API pass into a cardinality-bound union.

        GitHub does not provide a stable tie-breaker when more than one page of
        workflow runs shares the same ``created_at`` second.  A single pass can
        therefore contain duplicates across pages.  The union is allowed to
        close only after it contains exactly ``total_count`` unique run keys
        and every key has appeared with identical metadata in at least two
        distinct passes.
        """

        if not pages:
            raise PaginationDrift(
                "convergence pass returned no pages", observed_total=0
            )
        total = pages[0].total_count
        if total > GITHUB_FILTER_LIMIT:
            raise UnstableEnumerationError(
                f"one-second convergence total {total} exceeds "
                f"{GITHUB_FILTER_LIMIT}"
            )
        expected_pages = max(1, math.ceil(total / DEFAULT_PER_PAGE))
        if len(pages) != expected_pages:
            raise PaginationDrift(
                f"convergence pass has {len(pages)} pages, expected "
                f"{expected_pages}",
                observed_total=total,
            )

        repo_key = str(row["repo_key"])
        normalized: dict[
            tuple[int, int], tuple[dict[str, Any], str]
        ] = {}
        page_lines: list[str] = []
        page_proofs: list[
            tuple[int, int, int, int, int, str, str]
        ] = []
        raw_item_count = 0
        for page_no, page in enumerate(pages, start=1):
            if page.total_count != total:
                raise PaginationDrift(
                    f"convergence total_count changed {total} -> "
                    f"{page.total_count}",
                    observed_total=page.total_count,
                )
            expected_items = (
                DEFAULT_PER_PAGE
                if page_no < expected_pages
                else total - DEFAULT_PER_PAGE * (expected_pages - 1)
            )
            if len(page.workflow_runs) != expected_items:
                raise PaginationDrift(
                    f"convergence page {page_no} has "
                    f"{len(page.workflow_runs)} items, expected {expected_items}",
                    observed_total=total,
                )
            page_keys: list[str] = []
            for run in page.workflow_runs:
                record, metadata_sha, key = self._normalize_run(
                    repo_key,
                    run,
                    start_epoch=int(row["start_epoch"]),
                    end_epoch=int(row["end_epoch"]),
                )
                previous = normalized.get(key)
                if previous is not None and previous[1] != metadata_sha:
                    raise PaginationDrift(
                        f"convergence run {key[0]} attempt {key[1]} "
                        "changed metadata within one pass",
                        observed_total=total,
                    )
                normalized[key] = (record, metadata_sha)
                page_keys.append(
                    f"{repo_key}\t{key[0]}\t{key[1]}\t{metadata_sha}"
                )
            raw_item_count += len(page.workflow_runs)
            page_key_digest = _hash_lines(sorted(page_keys))
            page_line = (
                f"{page_no}\t{page.total_count}\t"
                f"{len(page.workflow_runs)}\t{len(set(page_keys))}\t"
                f"{len(page_keys) - len(set(page_keys))}\t"
                f"{page.payload_sha256}\t{page_key_digest}"
            )
            page_lines.append(page_line)
            page_proofs.append(
                (
                    page_no,
                    page.total_count,
                    len(page.workflow_runs),
                    len(set(page_keys)),
                    len(page_keys) - len(set(page_keys)),
                    page.payload_sha256,
                    page_key_digest,
                )
            )
        if raw_item_count != total:
            raise PaginationDrift(
                f"convergence pass returned {raw_item_count} raw items "
                f"for total_count={total}",
                observed_total=total,
            )

        pass_run_sha256 = _hash_lines(
            f"{repo_key}\t{run_id}\t{run_attempt}\t{metadata_sha}"
            for (run_id, run_attempt), (_record, metadata_sha) in sorted(
                normalized.items()
            )
        )
        page_payload_set_sha256 = _hash_lines(page_lines)
        now = _utc_now()
        window_id = int(row["id"])
        with self._write_lock, conn:
            state = conn.execute(
                "SELECT * FROM window_convergence WHERE window_id=?",
                (window_id,),
            ).fetchone()
            if state is None:
                raise UnstableEnumerationError(
                    f"window {window_id} lost convergence state"
                )
            if (
                state["candidate_total"] is not None
                and int(state["candidate_total"]) != total
            ):
                raise PaginationDrift(
                    f"convergence total_count changed "
                    f"{state['candidate_total']} -> {total}",
                    observed_total=total,
                )
            pass_no = int(state["attempts"]) + 1
            for (run_id, run_attempt), (
                record,
                metadata_sha,
            ) in sorted(normalized.items()):
                existing = conn.execute(
                    """
                    SELECT metadata_sha256,last_pass,observation_count
                    FROM convergence_runs
                    WHERE window_id=? AND repo_key=? AND run_id=?
                      AND run_attempt=?
                    """,
                    (window_id, repo_key, run_id, run_attempt),
                ).fetchone()
                if existing is not None:
                    if str(existing["metadata_sha256"]) != metadata_sha:
                        raise PaginationDrift(
                            f"convergence run {run_id} attempt {run_attempt} "
                            "changed metadata across passes",
                            observed_total=total,
                        )
                    if int(existing["last_pass"]) != pass_no:
                        conn.execute(
                            """
                            UPDATE convergence_runs
                            SET last_pass=?,observation_count=observation_count+1
                            WHERE window_id=? AND repo_key=? AND run_id=?
                              AND run_attempt=?
                            """,
                            (
                                pass_no,
                                window_id,
                                repo_key,
                                run_id,
                                run_attempt,
                            ),
                        )
                    continue
                conn.execute(
                    """
                    INSERT INTO convergence_runs(
                        window_id,repo_key,run_id,run_attempt,created_at,
                        updated_at,run_started_at,status,conclusion,workflow_id,
                        workflow_name,event,head_branch,head_sha,run_number,
                        html_url,api_url,metadata_blob,metadata_sha256,
                        first_seen_at,first_pass,last_pass,observation_count
                    ) VALUES (
                        :window_id,:repo_key,:run_id,:run_attempt,:created_at,
                        :updated_at,:run_started_at,:status,:conclusion,
                        :workflow_id,:workflow_name,:event,:head_branch,
                        :head_sha,:run_number,:html_url,:api_url,:metadata_blob,
                        :metadata_sha256,:first_seen_at,:first_pass,:last_pass,1
                    )
                    """,
                    {
                        **record,
                        "window_id": window_id,
                        "first_seen_at": now,
                        "first_pass": pass_no,
                        "last_pass": pass_no,
                    },
                )

            aggregate = conn.execute(
                """
                SELECT COUNT(*) AS distinct_count,
                       COALESCE(MIN(observation_count),0) AS min_observations
                FROM convergence_runs WHERE window_id=?
                """,
                (window_id,),
            ).fetchone()
            distinct_count = int(aggregate["distinct_count"])
            min_observations = int(aggregate["min_observations"])
            if distinct_count > total:
                raise UnstableEnumerationError(
                    f"convergence union for window {window_id} contains "
                    f"{distinct_count} runs, above total_count={total}"
                )
            union_sha256 = _hash_lines(
                f"{item['repo_key']}\t{item['run_id']}\t"
                f"{item['run_attempt']}\t{item['metadata_sha256']}"
                for item in conn.execute(
                    """
                    SELECT repo_key,run_id,run_attempt,metadata_sha256
                    FROM convergence_runs WHERE window_id=?
                    ORDER BY repo_key,run_id,run_attempt
                    """,
                    (window_id,),
                )
            )
            conn.execute(
                """
                INSERT INTO convergence_passes(
                    window_id,pass_no,total_count,page_count,raw_item_count,
                    distinct_item_count,duplicate_item_count,
                    page_payload_set_sha256,run_keys_sha256,
                    accumulated_distinct_count,min_observation_count,observed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    window_id,
                    pass_no,
                    total,
                    expected_pages,
                    raw_item_count,
                    len(normalized),
                    raw_item_count - len(normalized),
                    page_payload_set_sha256,
                    pass_run_sha256,
                    distinct_count,
                    min_observations,
                    now,
                ),
            )
            conn.executemany(
                """
                INSERT INTO convergence_pass_pages(
                    window_id,pass_no,page_no,total_count,item_count,
                    distinct_item_count,duplicate_item_count,payload_sha256,
                    run_keys_sha256
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                [
                    (window_id, pass_no, *page_proof)
                    for page_proof in page_proofs
                ],
            )
            conn.executemany(
                """
                INSERT INTO convergence_pass_runs(
                    window_id,pass_no,repo_key,run_id,run_attempt,
                    metadata_sha256
                ) VALUES (?,?,?,?,?,?)
                """,
                [
                    (
                        window_id,
                        pass_no,
                        repo_key,
                        run_id,
                        run_attempt,
                        metadata_sha,
                    )
                    for (run_id, run_attempt), (
                        _record,
                        metadata_sha,
                    ) in sorted(normalized.items())
                ],
            )
            conn.execute(
                """
                UPDATE window_convergence
                SET attempts=?,candidate_total=?,candidate_sha256=?,
                    stable_observations=?,last_error=?,updated_at=?
                WHERE window_id=?
                """,
                (
                    pass_no,
                    total,
                    union_sha256,
                    min_observations if distinct_count == total else 0,
                    (
                        None
                        if distinct_count == total and min_observations >= 2
                        else (
                            f"cardinality union has {distinct_count}/{total} "
                            f"runs; minimum distinct-pass observations="
                            f"{min_observations}"
                        )
                    ),
                    now,
                    window_id,
                ),
            )
            if distinct_count != total or min_observations < 2:
                return False, None

            mismatch = conn.execute(
                """
                SELECT candidate.repo_key,candidate.run_id,
                       candidate.run_attempt
                FROM convergence_runs candidate
                JOIN runs existing
                  ON existing.repo_key=candidate.repo_key
                 AND existing.run_id=candidate.run_id
                 AND existing.run_attempt=candidate.run_attempt
                WHERE candidate.window_id=?
                  AND existing.metadata_sha256 != candidate.metadata_sha256
                LIMIT 1
                """,
                (window_id,),
            ).fetchone()
            if mismatch is not None:
                raise UnstableEnumerationError(
                    "convergence metadata differs from a previously recorded "
                    "inventory run for "
                    f"{mismatch['repo_key']}#{mismatch['run_id']} attempt "
                    f"{mismatch['run_attempt']}"
                )
            conn.execute(
                """
                INSERT OR IGNORE INTO runs(
                    repo_key,run_id,run_attempt,created_at,updated_at,
                    run_started_at,status,conclusion,workflow_id,workflow_name,
                    event,head_branch,head_sha,run_number,html_url,api_url,
                    metadata_blob,metadata_sha256,first_seen_at
                )
                SELECT repo_key,run_id,run_attempt,created_at,updated_at,
                       run_started_at,status,conclusion,workflow_id,
                       workflow_name,event,head_branch,head_sha,run_number,
                       html_url,api_url,metadata_blob,metadata_sha256,
                       first_seen_at
                FROM convergence_runs WHERE window_id=?
                """,
                (window_id,),
            )
            conn.execute(
                """
                INSERT INTO window_runs(
                    window_id,repo_key,run_id,run_attempt,metadata_sha256
                )
                SELECT window_id,repo_key,run_id,run_attempt,metadata_sha256
                FROM convergence_runs WHERE window_id=?
                ORDER BY repo_key,run_id,run_attempt
                """,
                (window_id,),
            )
            pass_stats = conn.execute(
                """
                SELECT COUNT(*) AS pass_count,MIN(pass_no) AS first_pass,
                       MAX(pass_no) AS last_pass,
                       SUM(page_count) AS observed_pages,
                       SUM(raw_item_count) AS observed_items
                FROM convergence_passes WHERE window_id=?
                """,
                (window_id,),
            ).fetchone()
            pass_set_sha256 = self._convergence_pass_set_sha256(
                conn, window_id=window_id
            )
            pass_count = int(pass_stats["pass_count"])
            observed_pages = int(pass_stats["observed_pages"])
            observed_items = int(pass_stats["observed_items"])
            conn.execute(
                """
                INSERT INTO window_union_closures(
                    window_id,total_count,pass_count,first_pass_no,last_pass_no,
                    observed_page_count,observed_item_count,distinct_run_count,
                    min_observation_count,pass_set_sha256,run_keys_sha256,
                    closed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    window_id,
                    total,
                    pass_count,
                    int(pass_stats["first_pass"]),
                    int(pass_stats["last_pass"]),
                    observed_pages,
                    observed_items,
                    distinct_count,
                    min_observations,
                    pass_set_sha256,
                    union_sha256,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE search_windows
                SET status='done',expected_total=?,expected_pages=?,
                    pages_done=?,raw_items=?,distinct_items=?,
                    duplicate_items=?,run_keys_sha256=?,failure_class=NULL,
                    failure_message=NULL,updated_at=?
                WHERE id=?
                """,
                (
                    total,
                    expected_pages,
                    observed_pages,
                    observed_items,
                    distinct_count,
                    observed_items - distinct_count,
                    union_sha256,
                    now,
                    window_id,
                ),
            )
            conn.execute(
                "DELETE FROM window_convergence WHERE window_id=?",
                (window_id,),
            )
            return True, union_sha256

    def store_page(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        page_no: int,
        page: PageResponse,
        per_page: int = DEFAULT_PER_PAGE,
    ) -> bool:
        """Atomically commit a page.  Return true when the leaf closes."""

        window_id = int(row["id"])
        repo_key = str(row["repo_key"])
        start = int(row["start_epoch"])
        end = int(row["end_epoch"])
        total = page.total_count
        if total > GITHUB_FILTER_LIMIT:
            raise ValueError("dense response must be split before storing a page")
        recorded_total = row["expected_total"]
        if recorded_total is not None and int(recorded_total) != total:
            raise PaginationDrift(
                f"total_count changed for window {window_id}: "
                f"{recorded_total} -> {total}",
                observed_total=total,
            )
        expected_pages = max(1, math.ceil(total / per_page))
        if page_no < 1 or page_no > expected_pages:
            raise MalformedAPIError(
                f"page {page_no} outside expected 1..{expected_pages}"
            )
        expected_items = (
            per_page
            if page_no < expected_pages
            else total - per_page * (expected_pages - 1)
        )
        if len(page.workflow_runs) != expected_items:
            raise PaginationDrift(
                f"incomplete page {page_no}: expected {expected_items} items "
                f"from total_count={total}, got {len(page.workflow_runs)}",
                observed_total=total,
            )

        normalized: list[tuple[dict[str, Any], str, tuple[int, int]]] = []
        for run in page.workflow_runs:
            normalized.append(
                self._normalize_run(
                    repo_key, run, start_epoch=start, end_epoch=end
                )
            )
        page_keys = [
            f"{repo_key}\t{run_id}\t{attempt}\t{metadata_sha}"
            for _, metadata_sha, (run_id, attempt) in normalized
        ]
        page_key_digest = _hash_lines(sorted(page_keys))
        duplicate_in_page = len(page_keys) - len(set(page_keys))

        now = _utc_now()
        with self._write_lock, conn:
            current = conn.execute(
                "SELECT * FROM search_windows WHERE id=?", (window_id,)
            ).fetchone()
            if current is None:
                raise UnstableEnumerationError("search window disappeared")
            if current["expected_total"] is not None and (
                int(current["expected_total"]) != total
            ):
                raise PaginationDrift(
                    f"total_count changed for window {window_id}: "
                    f"{current['expected_total']} -> {total}",
                    observed_total=total,
                )
            old_page = conn.execute(
                """
                SELECT payload_sha256,run_keys_sha256
                FROM window_pages WHERE window_id=? AND page_no=?
                """,
                (window_id, page_no),
            ).fetchone()
            if old_page is not None:
                if (
                    old_page["payload_sha256"] != page.payload_sha256
                    or old_page["run_keys_sha256"] != page_key_digest
                ):
                    raise PaginationDrift(
                        f"page {page_no} of window {window_id} changed on replay",
                        observed_total=total,
                    )
                return str(current["status"]) == "done"
            if current["status"] not in {"open", "fetching"}:
                raise UnstableEnumerationError(
                    f"cannot store page in window status {current['status']!r}"
                )

            for record, metadata_sha, (run_id, run_attempt) in normalized:
                existing = conn.execute(
                    """
                    SELECT metadata_sha256 FROM runs
                    WHERE repo_key=? AND run_id=? AND run_attempt=?
                    """,
                    (repo_key, run_id, run_attempt),
                ).fetchone()
                if existing is not None:
                    if str(existing["metadata_sha256"]) != metadata_sha:
                        raise PaginationDrift(
                            f"workflow run {repo_key}#{run_id} attempt "
                            f"{run_attempt} changed during enumeration",
                            observed_total=total,
                        )
                else:
                    conn.execute(
                        """
                        INSERT INTO runs(
                            repo_key,run_id,run_attempt,created_at,updated_at,
                            run_started_at,status,conclusion,workflow_id,
                            workflow_name,event,head_branch,head_sha,run_number,
                            html_url,api_url,metadata_blob,metadata_sha256,
                            first_seen_at
                        ) VALUES (
                            :repo_key,:run_id,:run_attempt,:created_at,:updated_at,
                            :run_started_at,:status,:conclusion,:workflow_id,
                            :workflow_name,:event,:head_branch,:head_sha,
                            :run_number,:html_url,:api_url,:metadata_blob,
                            :metadata_sha256,:first_seen_at
                        )
                        """,
                        {**record, "first_seen_at": now},
                    )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO window_runs(
                        window_id,repo_key,run_id,run_attempt,metadata_sha256
                    ) VALUES (?,?,?,?,?)
                    """,
                    (window_id, repo_key, run_id, run_attempt, metadata_sha),
                )

            conn.execute(
                """
                INSERT INTO window_pages(
                    window_id,page_no,total_count,item_count,
                    distinct_item_count,duplicate_item_count,payload_sha256,
                    run_keys_sha256,fetched_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    window_id,
                    page_no,
                    total,
                    len(page.workflow_runs),
                    len(set(page_keys)),
                    duplicate_in_page,
                    page.payload_sha256,
                    page_key_digest,
                    now,
                ),
            )
            aggregates = conn.execute(
                """
                SELECT COUNT(*) AS pages_done,
                       COALESCE(SUM(item_count),0) AS raw_items,
                       COALESCE(SUM(duplicate_item_count),0) AS duplicate_items
                FROM window_pages WHERE window_id=?
                """,
                (window_id,),
            ).fetchone()
            distinct_items = int(
                conn.execute(
                    "SELECT COUNT(*) FROM window_runs WHERE window_id=?",
                    (window_id,),
                ).fetchone()[0]
            )
            pages_done = int(aggregates["pages_done"])
            raw_items = int(aggregates["raw_items"])
            duplicate_items = raw_items - distinct_items
            status = "fetching"
            digest: str | None = None
            if pages_done == expected_pages:
                page_numbers = [
                    int(item[0])
                    for item in conn.execute(
                        """
                        SELECT page_no FROM window_pages
                        WHERE window_id=? ORDER BY page_no
                        """,
                        (window_id,),
                    )
                ]
                if page_numbers != list(range(1, expected_pages + 1)):
                    raise MalformedAPIError(
                        f"window {window_id} has non-contiguous page closure"
                    )
                if raw_items != total:
                    raise PaginationDrift(
                        f"window {window_id} raw item count {raw_items} "
                        f"does not equal total_count {total}",
                        observed_total=total,
                    )
                if distinct_items != total:
                    raise PaginationDrift(
                        f"window {window_id} returned {distinct_items} distinct "
                        f"runs for total_count={total}; duplicates make the "
                        "enumeration incomplete",
                        observed_total=total,
                    )
                digest = _hash_lines(
                    str(item["repo_key"])
                    + "\t"
                    + str(item["run_id"])
                    + "\t"
                    + str(item["run_attempt"])
                    + "\t"
                    + str(item["metadata_sha256"])
                    for item in conn.execute(
                        """
                        SELECT repo_key,run_id,run_attempt,metadata_sha256
                        FROM window_runs WHERE window_id=?
                        ORDER BY repo_key,run_id,run_attempt
                        """,
                        (window_id,),
                    )
                )
                status = "done"
            conn.execute(
                """
                UPDATE search_windows
                SET status=?,expected_total=?,expected_pages=?,pages_done=?,
                    raw_items=?,distinct_items=?,duplicate_items=?,
                    run_keys_sha256=?,failure_class=NULL,failure_message=NULL,
                    updated_at=?
                WHERE id=?
                """,
                (
                    status,
                    total,
                    expected_pages,
                    pages_done,
                    raw_items,
                    distinct_items,
                    duplicate_items,
                    digest,
                    now,
                    window_id,
                ),
            )
            return status == "done"

    def mark_failed(
        self,
        conn: sqlite3.Connection,
        window_id: int,
        exc: BaseException,
        *,
        redacted_message: str | None = None,
    ) -> None:
        with self._write_lock, conn:
            conn.execute(
                """
                UPDATE search_windows
                SET status='failed',failure_class=?,failure_message=?,updated_at=?
                WHERE id=? AND status NOT IN ('done','split')
                """,
                (
                    type(exc).__name__,
                    (redacted_message if redacted_message is not None else str(exc))[
                        :4000
                    ],
                    _utc_now(),
                    window_id,
                ),
            )

    def next_window(
        self, conn: sqlite3.Connection, repo_key: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM search_windows
            WHERE repo_key=? AND status IN ('open','fetching')
            ORDER BY depth,start_epoch,id LIMIT 1
            """,
            (repo_key,),
        ).fetchone()

    def progress(self) -> dict[str, Any]:
        conn = self.connect(readonly=True)
        try:
            conn.execute("BEGIN")
            meta = self._meta(conn)
            status_counts = {
                str(row["status"]): int(row["count"])
                for row in conn.execute(
                    """
                    SELECT status,COUNT(*) AS count
                    FROM search_windows GROUP BY status
                    """
                )
            }
            repo_done = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM repos r
                    WHERE NOT EXISTS (
                        SELECT 1 FROM search_windows w
                        WHERE w.repo_key=r.repo_key
                          AND w.status IN ('open','fetching','failed')
                    )
                    """
                ).fetchone()[0]
            )
            return {
                "schema": PROGRESS_SCHEMA,
                "generated_at": _utc_now(),
                "database": self.path,
                "repo_list_sha256": meta.get("repo_list_sha256"),
                "repo_scope_sha256": meta.get("repo_scope_sha256"),
                "interval": {
                    "start": meta.get("start_utc"),
                    "end": meta.get("end_utc"),
                    "semantics": "[start,end)",
                },
                "smoke": meta.get("smoke") == "1",
                "repos_total": int(meta.get("repo_count", "0")),
                "repos_closed": repo_done,
                "runs": int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]),
                "requests": int(
                    conn.execute("SELECT COUNT(*) FROM request_ledger").fetchone()[0]
                ),
                "windows": status_counts,
            }
        finally:
            conn.close()

    @staticmethod
    def _logical_table_ledgers(
        conn: sqlite3.Connection,
    ) -> list[dict[str, object]]:
        ledgers: list[dict[str, object]] = []
        for table, order_by in _LOGICAL_TABLE_ORDER:
            table_info = list(
                conn.execute(f'PRAGMA table_info("{table}")')
            )
            if not table_info:
                raise CompletionError(
                    f"inventory logical table {table!r} has no columns"
                )
            columns = [str(item["name"]) for item in table_info]
            declared_types = {
                str(item["name"]): str(item["type"]).upper()
                for item in table_info
            }
            digest = hashlib.sha256()
            digest.update(
                _canonical_json(
                    {
                        "domain": (
                            "cppmega-ci-inventory-complete-table-v1"
                        ),
                        "table": table,
                        "columns": [
                            {
                                "name": str(item["name"]),
                                "type": str(item["type"]).upper(),
                                "notnull": int(item["notnull"]),
                                "pk": int(item["pk"]),
                            }
                            for item in table_info
                        ],
                    }
                ).encode("utf-8")
            )
            digest.update(b"\n")
            selected_columns = ",".join(
                f'"{column}"' for column in columns
            )
            row_count = 0
            for row in conn.execute(
                f'SELECT {selected_columns} FROM "{table}" '
                f"ORDER BY {order_by}"
            ):
                digest.update(b"R")
                for column in columns:
                    value = row[column]
                    declared_type = declared_types[column]
                    if value is None:
                        digest.update(b"N")
                    elif type(value) is int:
                        if declared_type not in {"", "INTEGER"}:
                            raise CompletionError(
                                f"{table}.{column} has INTEGER storage "
                                f"under declared type {declared_type!r}"
                            )
                        raw = str(value).encode("ascii")
                        digest.update(b"I")
                        digest.update(len(raw).to_bytes(8, "big"))
                        digest.update(raw)
                    elif type(value) is str:
                        if declared_type not in {"", "TEXT"}:
                            raise CompletionError(
                                f"{table}.{column} has TEXT storage under "
                                f"declared type {declared_type!r}"
                            )
                        raw = value.encode("utf-8")
                        digest.update(b"T")
                        digest.update(len(raw).to_bytes(8, "big"))
                        digest.update(raw)
                    elif type(value) is bytes:
                        if declared_type != "BLOB":
                            raise CompletionError(
                                f"{table}.{column} has BLOB storage under "
                                f"declared type {declared_type!r}"
                            )
                        digest.update(b"B")
                        digest.update(len(value).to_bytes(8, "big"))
                        digest.update(hashlib.sha256(value).digest())
                    else:
                        raise CompletionError(
                            f"{table}.{column} has unsupported SQLite "
                            f"storage type {type(value).__name__}"
                        )
                    digest.update(b"\x1f")
                digest.update(b"\n")
                row_count += 1
            ledgers.append(
                {
                    "table": table,
                    "row_count": row_count,
                    "sha256": digest.hexdigest(),
                }
            )
        return ledgers

    @staticmethod
    def _validate_sqlite_sequence(conn: sqlite3.Connection) -> None:
        sequence_rows = list(
            conn.execute(
                "SELECT name,seq FROM sqlite_sequence ORDER BY name"
            )
        )
        observed: dict[str, int] = {}
        for row in sequence_rows:
            name = row["name"]
            sequence = row["seq"]
            if (
                type(name) is not str
                or name not in _AUTOINCREMENT_TABLES
                or type(sequence) is not int
                or sequence < 0
            ):
                raise CompletionError(
                    "inventory SQLite sequence ledger is malformed or "
                    "contains an unauthorized table"
                )
            if name in observed:
                raise CompletionError(
                    "inventory SQLite sequence ledger contains duplicate "
                    f"table {name!r}"
                )
            observed[name] = sequence
        expected: dict[str, int] = {}
        for table in sorted(_AUTOINCREMENT_TABLES):
            maximum = conn.execute(
                f'SELECT MAX(id) FROM "{table}"'
            ).fetchone()[0]
            if maximum is not None:
                if type(maximum) is not int or maximum < 1:
                    raise CompletionError(
                        f"inventory {table} primary-key maximum is invalid"
                    )
                expected[table] = maximum
        if observed != expected:
            raise CompletionError(
                "inventory SQLite sequence ledger differs from exact "
                f"table maxima: observed={observed}, expected={expected}"
            )

    def _validate_stored_run_projection(
        self,
        row: sqlite3.Row,
        metadata_bytes: bytes,
        *,
        start_epoch: int,
        end_epoch: int,
        where: str,
    ) -> None:
        try:
            metadata = json.loads(metadata_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompletionError(
                f"{where} metadata is not valid UTF-8 JSON: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise CompletionError(f"{where} metadata root is not an object")
        try:
            canonical_metadata = _canonical_json(metadata).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CompletionError(
                f"{where} metadata is not canonical JSON: {exc}"
            ) from exc
        if canonical_metadata != metadata_bytes:
            raise CompletionError(
                f"{where} metadata bytes are not the exact canonical JSON"
            )
        try:
            normalized, metadata_sha256, _identity = self._normalize_run(
                str(row["repo_key"]),
                metadata,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
            )
        except InventoryError as exc:
            raise CompletionError(
                f"{where} metadata cannot reconstruct its stored "
                f"projection: {exc}"
            ) from exc
        projection_columns = (
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
        )
        for column in projection_columns:
            if row[column] != normalized[column]:
                raise CompletionError(
                    f"{where}.{column} differs from its canonical decoded "
                    "metadata projection"
                )
        if str(row["metadata_sha256"]) != metadata_sha256:
            raise CompletionError(
                f"{where}.metadata_sha256 differs from canonical metadata"
            )
        for column in (
            "created_at",
            "updated_at",
            "run_started_at",
            "first_seen_at",
        ):
            value = row[column]
            if value is not None:
                _require_canonical_utc(
                    value,
                    where=f"{where}.{column}",
                )
        head_sha = row["head_sha"]
        if head_sha is not None and (
            not isinstance(head_sha, str)
            or re.fullmatch(r"[0-9a-f]{40}", head_sha) is None
        ):
            raise CompletionError(
                f"{where}.head_sha is not a lowercase 40-hex commit"
            )

    def store_source_drift_reconciliation(
        self,
        roots: Sequence[Mapping[str, Any]],
        new_runs: Sequence[Mapping[str, Any]],
    ) -> None:
        if not roots:
            raise CompletionError("source-drift reconciliation has no roots")
        encoded_roots: dict[int, tuple[bytes, bytes, str, str]] = {}
        for root in roots:
            window_id = root.get("window_id")
            completed_at = _require_canonical_utc(
                root.get("completed_at"),
                where="source-drift reconciliation completed_at",
            )
            producer = root.get("producer_script_sha256")
            if (
                isinstance(window_id, bool)
                or not isinstance(window_id, int)
                or window_id < 1
                or not isinstance(producer, str)
                or re.fullmatch(r"[0-9a-f]{64}", producer) is None
                or window_id in encoded_roots
            ):
                raise CompletionError(
                    "source-drift reconciliation root binding is invalid"
                )
            raw = _canonical_json(dict(root)).encode("utf-8")
            if len(raw) > MAX_RUN_METADATA_BYTES:
                raise CompletionError(
                    "source-drift reconciliation proof exceeds its "
                    "versioned raw-byte bound"
                )
            blob = zlib.compress(raw, 6)
            if len(blob) > MAX_RUN_METADATA_COMPRESSED_BYTES:
                raise CompletionError(
                    "source-drift reconciliation proof exceeds its "
                    "versioned compressed-byte bound"
                )
            encoded_roots[window_id] = (
                raw,
                blob,
                str(producer),
                completed_at,
            )

        records: dict[tuple[str, int, int], Mapping[str, Any]] = {}
        for record in new_runs:
            key = (
                str(record["repo_key"]),
                int(record["run_id"]),
                int(record["run_attempt"]),
            )
            if key in records:
                raise CompletionError(
                    "source-drift reconciliation repeats a new run"
                )
            records[key] = record

        conn = self.connect()
        try:
            with self._write_lock, conn:
                existing_proofs = {
                    int(row["window_id"]): row
                    for row in conn.execute(
                        """
                        SELECT window_id,proof_blob,proof_raw_size,
                               proof_sha256,producer_script_sha256,
                               completed_at
                        FROM source_drift_reconciliations
                        ORDER BY window_id
                        """
                    )
                }
                if existing_proofs and set(existing_proofs) != set(
                    encoded_roots
                ):
                    raise CompletionError(
                        "persisted source-drift reconciliation roots differ "
                        "from the stable live proof"
                    )

                now = _utc_now()
                for (repo_key, run_id, run_attempt), record in records.items():
                    same_run = list(
                        conn.execute(
                            """
                            SELECT run_attempt,metadata_sha256
                            FROM runs
                            WHERE repo_key=? AND run_id=?
                            ORDER BY run_attempt
                            """,
                            (repo_key, run_id),
                        )
                    )
                    if same_run:
                        if (
                            len(same_run) != 1
                            or int(same_run[0]["run_attempt"]) != run_attempt
                            or str(same_run[0]["metadata_sha256"])
                            != record["metadata_sha256"]
                        ):
                            raise CompletionError(
                                "source-drift reconciliation changed an "
                                f"existing attempt ceiling for {repo_key}#{run_id}"
                            )
                        linked = conn.execute(
                            """
                            SELECT 1 FROM window_runs
                            WHERE repo_key=? AND run_id=? AND run_attempt=?
                            LIMIT 1
                            """,
                            (repo_key, run_id, run_attempt),
                        ).fetchone()
                        if linked is not None:
                            raise CompletionError(
                                "source-drift reconciliation classified an "
                                "already linked run as new"
                            )
                        continue
                    conn.execute(
                        """
                        INSERT INTO runs(
                            repo_key,run_id,run_attempt,created_at,updated_at,
                            run_started_at,status,conclusion,workflow_id,
                            workflow_name,event,head_branch,head_sha,run_number,
                            html_url,api_url,metadata_blob,metadata_sha256,
                            first_seen_at
                        ) VALUES (
                            :repo_key,:run_id,:run_attempt,:created_at,:updated_at,
                            :run_started_at,:status,:conclusion,:workflow_id,
                            :workflow_name,:event,:head_branch,:head_sha,
                            :run_number,:html_url,:api_url,:metadata_blob,
                            :metadata_sha256,:first_seen_at
                        )
                        """,
                        {**record, "first_seen_at": now},
                    )

                for window_id, (
                    raw,
                    blob,
                    producer,
                    completed_at,
                ) in encoded_roots.items():
                    existing = existing_proofs.get(window_id)
                    proof_sha256 = _sha256_bytes(raw)
                    if existing is not None:
                        if (
                            existing["proof_blob"] != blob
                            or int(existing["proof_raw_size"]) != len(raw)
                            or str(existing["proof_sha256"]) != proof_sha256
                            or str(existing["producer_script_sha256"])
                            != producer
                            or str(existing["completed_at"]) != completed_at
                        ):
                            raise CompletionError(
                                "persisted source-drift reconciliation proof "
                                f"differs for window {window_id}"
                            )
                        continue
                    conn.execute(
                        """
                        INSERT INTO source_drift_reconciliations(
                            window_id,proof_blob,proof_raw_size,proof_sha256,
                            producer_script_sha256,completed_at
                        ) VALUES (?,?,?,?,?,?)
                        """,
                        (
                            window_id,
                            sqlite3.Binary(blob),
                            len(raw),
                            proof_sha256,
                            producer,
                            completed_at,
                        ),
                    )
        finally:
            conn.close()

    @staticmethod
    def _validate_request_ledger(
        conn: sqlite3.Connection,
    ) -> int:
        rows = conn.execute(
            """
            SELECT request.*,repo.canonical AS repo_canonical,
                   window.repo_key AS window_repo_key
            FROM request_ledger request
            LEFT JOIN repos repo
              ON repo.repo_key=request.repo_key
            LEFT JOIN search_windows window
              ON window.id=request.window_id
            ORDER BY request.id
            """
        )
        expected_id = 1
        for row in rows:
            request_id = int(row["id"])
            if request_id != expected_id:
                raise CompletionError(
                    "inventory request ledger IDs are not exact and "
                    "contiguous"
                )
            expected_id += 1
            where = f"request_ledger[{request_id}]"
            _require_canonical_utc(
                row["requested_at"],
                where=f"{where}.requested_at",
            )
            if (
                row["repo_canonical"] is None
                or row["window_repo_key"] is None
                or row["repo_key"] != row["window_repo_key"]
            ):
                raise CompletionError(
                    f"{where} is not bound to its repository window"
                )
            expected_endpoint = (
                f"/repos/{row['repo_canonical']}/actions/runs"
            )
            if row["endpoint"] != expected_endpoint:
                raise CompletionError(
                    f"{where}.endpoint differs from its exact repository "
                    "workflow-runs endpoint"
                )
            page_no = int(row["page_no"])
            per_page = int(row["per_page"])
            attempt = int(row["attempt"])
            latency_ms = int(row["latency_ms"])
            outcome = str(row["outcome"])
            http_status = row["http_status"]
            if (
                page_no < 1
                or per_page != DEFAULT_PER_PAGE
                or attempt < 0
                or latency_ms < 0
                or outcome not in _REQUEST_OUTCOMES
                or (
                    http_status is not None
                    and (
                        type(http_status) is not int
                        or not 100 <= http_status <= 599
                    )
                )
            ):
                raise CompletionError(
                    f"{where} has an invalid page/attempt/status/outcome/"
                    "latency domain"
                )
            error_class = row["error_class"]
            error_message = row["error_message"]
            if outcome == "success":
                if (
                    attempt < 1
                    or http_status != 200
                    or error_class is not None
                    or error_message is not None
                ):
                    raise CompletionError(
                        f"{where} success shape is invalid"
                    )
            else:
                if (
                    not isinstance(error_class, str)
                    or not error_class
                    or not isinstance(error_message, str)
                ):
                    raise CompletionError(
                        f"{where} error evidence shape is invalid"
                    )
                if outcome in _REQUEST_SYNTHETIC_OUTCOMES:
                    if (
                        attempt != 0
                        or http_status is not None
                        or latency_ms != 0
                    ):
                        raise CompletionError(
                            f"{where} synthetic recovery shape is invalid"
                        )
                elif attempt < 1:
                    raise CompletionError(
                        f"{where} API request attempt must be positive"
                    )
                if (
                    outcome == "transport_retry"
                    and http_status is not None
                ):
                    raise CompletionError(
                        f"{where} transport retry cannot have HTTP status"
                    )
                if outcome == "rate_limit_retry" and http_status not in {
                    403,
                    429,
                }:
                    raise CompletionError(
                        f"{where} rate-limit status is invalid"
                    )
                if outcome == "server_retry" and (
                    http_status is None or http_status < 500
                ):
                    raise CompletionError(
                        f"{where} server-retry status is invalid"
                    )
                if outcome == "malformed" and http_status != 200:
                    raise CompletionError(
                        f"{where} malformed-response status is invalid"
                    )
        return expected_id - 1

    def _validate_and_digests(
        self,
        *,
        immutable: bool = False,
    ) -> dict[str, Any]:
        conn = self.connect(readonly=True, immutable=immutable)
        try:
            # One SQLite read transaction makes every count and digest belong
            # to the same WAL snapshot, even if a separate process is still
            # writing request/page progress.
            conn.execute("BEGIN")
            _constrain_inventory_connection(conn)
            sqlite_schema_rows = _sqlite_schema_rows(conn)
            if sqlite_schema_rows != _EXPECTED_SQLITE_SCHEMA_ROWS:
                expected_by_identity = {
                    (row[0], row[1]): row
                    for row in _EXPECTED_SQLITE_SCHEMA_ROWS
                }
                actual_by_identity = {
                    (row[0], row[1]): row
                    for row in sqlite_schema_rows
                }
                missing = sorted(
                    set(expected_by_identity) - set(actual_by_identity)
                )
                extra = sorted(
                    set(actual_by_identity) - set(expected_by_identity)
                )
                altered = sorted(
                    identity
                    for identity in (
                        set(expected_by_identity) & set(actual_by_identity)
                    )
                    if (
                        expected_by_identity[identity]
                        != actual_by_identity[identity]
                    )
                )
                raise CompletionError(
                    "inventory SQLite schema differs from the exact "
                    "versioned contract "
                    f"(missing={missing}, extra={extra}, altered={altered})"
                )
            for table, identity_columns in (
                ("runs", "repo_key,run_id,run_attempt"),
                (
                    "convergence_runs",
                    "window_id,repo_key,run_id,run_attempt",
                ),
            ):
                invalid_blob = conn.execute(
                    f"""
                    SELECT {identity_columns},
                           typeof(metadata_blob) AS blob_type,
                           length(metadata_blob) AS compressed_bytes
                    FROM {table}
                    WHERE typeof(metadata_blob)!='blob'
                       OR length(metadata_blob)>?
                    LIMIT 1
                    """,
                    (MAX_RUN_METADATA_COMPRESSED_BYTES,),
                ).fetchone()
                if invalid_blob is not None:
                    raise CompletionError(
                        f"{table} metadata BLOB exceeds the versioned "
                        "compressed-byte bound or is not a BLOB"
                    )
            invalid_reconciliation = conn.execute(
                """
                SELECT window_id FROM source_drift_reconciliations
                WHERE typeof(proof_blob)!='blob'
                   OR length(proof_blob)>?
                   OR typeof(proof_raw_size)!='integer'
                   OR proof_raw_size<0
                   OR proof_raw_size>?
                   OR typeof(proof_sha256)!='text'
                   OR length(proof_sha256)!=64
                   OR proof_sha256 GLOB '*[^0-9a-f]*'
                   OR typeof(producer_script_sha256)!='text'
                   OR length(producer_script_sha256)!=64
                   OR producer_script_sha256 GLOB '*[^0-9a-f]*'
                   OR typeof(completed_at)!='text'
                LIMIT 1
                """,
                (
                    MAX_RUN_METADATA_COMPRESSED_BYTES,
                    MAX_RUN_METADATA_BYTES,
                ),
            ).fetchone()
            if invalid_reconciliation is not None:
                raise CompletionError(
                    "source-drift reconciliation proof exceeds its "
                    "versioned bounds or has invalid storage"
                )
            integrity_cursor = conn.execute("PRAGMA integrity_check")
            integrity_first = integrity_cursor.fetchone()
            integrity_second = integrity_cursor.fetchone()
            if (
                integrity_first is None
                or tuple(integrity_first) != ("ok",)
                or integrity_second is not None
            ):
                details = (
                    []
                    if integrity_first is None
                    else [str(integrity_first[0])]
                )
                if integrity_second is not None:
                    details.append(str(integrity_second[0]))
                raise CompletionError(
                    "inventory SQLite integrity check failed: "
                    + ("; ".join(details) if details else "no result")
                )
            foreign_key_violation = conn.execute(
                "PRAGMA foreign_key_check"
            ).fetchone()
            if foreign_key_violation is not None:
                raise CompletionError(
                    "inventory SQLite foreign-key check failed: "
                    f"{tuple(foreign_key_violation)!r}"
                )
            self._validate_sqlite_sequence(conn)
            logical_table_ledgers = self._logical_table_ledgers(conn)
            invalid_meta_storage = conn.execute(
                """
                SELECT key,typeof(key) AS key_type,
                       typeof(value) AS value_type
                FROM inventory_meta
                WHERE typeof(key)!='text' OR typeof(value)!='text'
                LIMIT 1
                """
            ).fetchone()
            if invalid_meta_storage is not None:
                raise CompletionError(
                    "inventory metadata keys and values must use exact TEXT "
                    "SQLite storage"
                )
            meta = self._meta(conn)
            if set(meta) != _EXPECTED_INVENTORY_META_KEYS:
                missing = sorted(
                    _EXPECTED_INVENTORY_META_KEYS - set(meta)
                )
                extra = sorted(set(meta) - _EXPECTED_INVENTORY_META_KEYS)
                raise CompletionError(
                    "database metadata keys differ from the exact versioned "
                    f"contract (missing={missing}, extra={extra})"
                )
            if meta["schema"] != SCHEMA_VERSION:
                raise CompletionError(f"unsupported database schema {meta['schema']!r}")
            if meta["metadata_encoding"] != METADATA_ENCODING:
                raise CompletionError(
                    "database workflow-run metadata encoding does not match "
                    f"{METADATA_ENCODING}"
                )
            for key in (
                "repo_list_sha256",
                "repo_scope_sha256",
                "script_sha256",
            ):
                if re.fullmatch(r"[0-9a-f]{64}", meta[key]) is None:
                    raise CompletionError(
                        f"database metadata {key} is not lowercase hex SHA-256"
                    )
            repo_count_binding = _require_canonical_decimal(
                meta["repo_count"],
                where="database metadata repo_count",
                minimum=1,
            )
            original_repo_count = _require_canonical_decimal(
                meta["original_repo_count"],
                where="database metadata original_repo_count",
                minimum=1,
            )
            unresolved_count = _require_canonical_decimal(
                meta["unresolved_count"],
                where="database metadata unresolved_count",
            )
            if unresolved_count != 0:
                raise CompletionError("unresolved repository count is not zero")
            if repo_count_binding > original_repo_count:
                raise CompletionError(
                    "database repository count exceeds its original scope"
                )
            if meta["smoke"] not in {"0", "1"}:
                raise CompletionError(
                    "database metadata smoke must be exactly '0' or '1'"
                )
            max_repos_text = meta["max_repos"]
            if not max_repos_text:
                if repo_count_binding != original_repo_count:
                    raise CompletionError(
                        "database metadata without max_repos must retain the "
                        "full original repository scope"
                    )
            else:
                max_repos = _require_canonical_decimal(
                    max_repos_text,
                    where="database metadata max_repos",
                    minimum=1,
                )
                if (
                    meta["smoke"] != "1"
                    or repo_count_binding
                    != min(max_repos, original_repo_count)
                ):
                    raise CompletionError(
                        "database metadata max_repos is inconsistent with "
                        "smoke mode and repository counts"
                    )
            if meta["smoke"] == "0" and max_repos_text:
                raise CompletionError(
                    "production inventory metadata cannot set max_repos"
                )
            _require_canonical_utc(
                meta["created_at"],
                where="database metadata created_at",
            )
            legacy_upgrades = [
                (
                    str(row["from_schema"]),
                    str(row["to_schema"]),
                    str(row["from_script_sha256"]),
                    str(row["to_script_sha256"]),
                    str(row["upgraded_at"]),
                )
                for row in conn.execute(
                    """
                    SELECT from_schema,to_schema,from_script_sha256,
                           to_script_sha256,upgraded_at
                    FROM inventory_upgrades ORDER BY id
                    """
                )
            ]
            binding_upgrades = [
                {
                    "from_schema": str(row["from_schema"]),
                    "to_schema": str(row["to_schema"]),
                    "from_script_sha256": str(
                        row["from_script_sha256"]
                    ),
                    "to_script_sha256": str(row["to_script_sha256"]),
                    "reason": str(row["reason"]),
                    "upgraded_at": str(row["upgraded_at"]),
                }
                for row in conn.execute(
                    """
                    SELECT from_schema,to_schema,from_script_sha256,
                           to_script_sha256,reason,upgraded_at
                    FROM inventory_binding_upgrades ORDER BY id
                    """
                )
            ]
            projected_upgrades = [
                (
                    row["from_schema"],
                    row["to_schema"],
                    row["from_script_sha256"],
                    row["to_script_sha256"],
                    row["upgraded_at"],
                )
                for row in binding_upgrades
            ]
            legacy_upgrade_ids = [
                int(row[0])
                for row in conn.execute(
                    "SELECT id FROM inventory_upgrades ORDER BY id"
                )
            ]
            binding_upgrade_ids = [
                int(row[0])
                for row in conn.execute(
                    """
                    SELECT id FROM inventory_binding_upgrades
                    ORDER BY id
                    """
                )
            ]
            expected_upgrade_ids = list(
                range(1, len(binding_upgrades) + 1)
            )
            if (
                legacy_upgrade_ids != expected_upgrade_ids
                or binding_upgrade_ids != expected_upgrade_ids
            ):
                raise CompletionError(
                    "inventory producer upgrade ledger IDs are not exact, "
                    "contiguous, and paired"
                )
            if legacy_upgrades != projected_upgrades:
                raise CompletionError(
                    "inventory producer upgrade ledgers disagree"
                )
            for index, upgrade in enumerate(binding_upgrades):
                if (
                    upgrade["from_schema"],
                    upgrade["to_schema"],
                ) not in {
                    (
                        "cppmega_ci_stream_inventory_v1",
                        LEGACY_SCHEMA_VERSION,
                    ),
                    (
                        "cppmega_ci_stream_inventory_v1",
                        PREVIOUS_SCHEMA_VERSION,
                    ),
                    (
                        "cppmega_ci_stream_inventory_v1",
                        SCHEMA_VERSION,
                    ),
                    (
                        LEGACY_SCHEMA_VERSION,
                        PREVIOUS_SCHEMA_VERSION,
                    ),
                    (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION),
                    (PREVIOUS_SCHEMA_VERSION, SCHEMA_VERSION),
                    (SCHEMA_VERSION, SCHEMA_VERSION),
                }:
                    raise CompletionError(
                        f"inventory producer upgrade {index} has an "
                        "unsupported schema transition"
                    )
                try:
                    _validate_upgrade_reason(upgrade["reason"])
                except BindingError as exc:
                    raise CompletionError(
                        f"inventory producer upgrade {index} reason is invalid"
                    ) from exc
                _require_canonical_utc(
                    upgrade["upgraded_at"],
                    where=(
                        f"inventory producer upgrade {index} upgraded_at"
                    ),
                )
                if index and (
                    binding_upgrades[index - 1]["to_schema"]
                    != upgrade["from_schema"]
                    or binding_upgrades[index - 1]["to_script_sha256"]
                    != upgrade["from_script_sha256"]
                ):
                    raise CompletionError(
                        f"inventory producer upgrade {index} breaks the "
                        "upgrade chain"
                    )
            if binding_upgrades and (
                binding_upgrades[-1]["to_schema"] != SCHEMA_VERSION
                or binding_upgrades[-1]["to_script_sha256"]
                != meta["script_sha256"]
            ):
                raise CompletionError(
                    "inventory producer upgrade chain does not bind the "
                    "completed producer"
                )
            start = _require_canonical_decimal(
                meta["start_epoch"],
                where="database metadata start_epoch",
            )
            end = _require_canonical_decimal(
                meta["end_epoch"],
                where="database metadata end_epoch",
            )
            if (
                meta["start_utc"] != format_utc_instant(start)
                or meta["end_utc"] != format_utc_instant(end)
            ):
                raise CompletionError(
                    "database UTC interval text differs from its epoch binding"
                )

            repos = list(
                conn.execute(
                    """
                    SELECT repo_key,owner,name,canonical,ordinal
                    FROM repos ORDER BY ordinal
                    """
                )
            )
            if len(repos) != repo_count_binding:
                raise CompletionError("database repository count differs from binding")
            for ordinal, repo in enumerate(repos):
                owner = str(repo["owner"])
                name = str(repo["name"])
                canonical = str(repo["canonical"])
                repo_key = str(repo["repo_key"])
                if (
                    int(repo["ordinal"]) != ordinal
                    or _OWNER_REPO_RE.fullmatch(f"{owner}/{name}") is None
                    or canonical != f"{owner}/{name}"
                    or repo_key != canonical.casefold()
                ):
                    raise CompletionError(
                        f"repository row {repo_key!r} violates its exact "
                        "owner/name/canonical/ordinal identity"
                    )
            scope_digest = _hash_lines(str(row["repo_key"]) for row in repos)
            if scope_digest != meta["repo_scope_sha256"]:
                raise CompletionError("database repository scope digest mismatch")
            request_count = self._validate_request_ledger(conn)

            unfinished = list(
                conn.execute(
                    """
                    SELECT id,repo_key,status,failure_class
                    FROM search_windows
                    WHERE status IN ('open','fetching','failed')
                    ORDER BY repo_key,start_epoch
                    LIMIT 10
                    """
                )
            )
            if unfinished:
                sample = ", ".join(
                    f"{row['repo_key']}:{row['id']}={row['status']}"
                    for row in unfinished
                )
                raise CompletionError(f"inventory has open/failed windows: {sample}")
            convergence_left = int(
                conn.execute("SELECT COUNT(*) FROM window_convergence").fetchone()[0]
            )
            if convergence_left:
                raise CompletionError(
                    f"inventory has {convergence_left} unresolved convergence proofs"
                )
            orphan_proof = conn.execute(
                """
                SELECT proof.window_id
                FROM (
                    SELECT window_id FROM convergence_passes
                    UNION
                    SELECT window_id FROM convergence_pass_pages
                    UNION
                    SELECT window_id FROM convergence_pass_runs
                    UNION
                    SELECT window_id FROM convergence_runs
                ) proof
                LEFT JOIN window_union_closures closure
                  ON closure.window_id=proof.window_id
                WHERE closure.window_id IS NULL
                LIMIT 1
                """
            ).fetchone()
            if orphan_proof is not None:
                raise CompletionError(
                    "inventory has convergence proof rows without a union "
                    f"closure for window {orphan_proof['window_id']}"
                )
            invalid_union_window = conn.execute(
                """
                SELECT closure.window_id
                FROM window_union_closures closure
                JOIN search_windows window ON window.id=closure.window_id
                WHERE window.status != 'done'
                   OR window.end_epoch - window.start_epoch != 1
                LIMIT 1
                """
            ).fetchone()
            if invalid_union_window is not None:
                raise CompletionError(
                    "inventory union closure is attached to an invalid window "
                    f"{invalid_union_window['window_id']}"
                )

            all_windows = list(
                conn.execute(
                    """
                    SELECT * FROM search_windows
                    ORDER BY repo_key,start_epoch,end_epoch,id
                    """
                )
            )
            by_repo: dict[str, list[sqlite3.Row]] = {}
            by_parent: dict[int, list[sqlite3.Row]] = {}
            windows_by_id = {
                int(row["id"]): row for row in all_windows
            }
            for row in all_windows:
                by_repo.setdefault(str(row["repo_key"]), []).append(row)
                _require_canonical_utc(
                    row["created_at"],
                    where=f"search_windows[{row['id']}].created_at",
                )
                _require_canonical_utc(
                    row["updated_at"],
                    where=f"search_windows[{row['id']}].updated_at",
                )
                if (
                    row["failure_class"] is not None
                    or row["failure_message"] is not None
                ):
                    raise CompletionError(
                        f"completed search window {row['id']} retains "
                        "failure evidence"
                    )
                if row["parent_id"] is not None:
                    parent_id = int(row["parent_id"])
                    parent = windows_by_id.get(parent_id)
                    if (
                        parent is None
                        or parent["repo_key"] != row["repo_key"]
                        or int(row["depth"]) != int(parent["depth"]) + 1
                    ):
                        raise CompletionError(
                            f"search window {row['id']} has a cross-repository "
                            "or depth-inconsistent parent"
                        )
                    by_parent.setdefault(parent_id, []).append(row)
                elif int(row["depth"]) != 0:
                    raise CompletionError(
                        f"root search window {row['id']} has nonzero depth"
                    )
            nonleaf_payload = conn.execute(
                """
                SELECT window.id
                FROM search_windows window
                WHERE window.status!='done'
                  AND (
                    EXISTS (
                      SELECT 1 FROM window_pages page
                      WHERE page.window_id=window.id
                    )
                    OR EXISTS (
                      SELECT 1 FROM window_runs member
                      WHERE member.window_id=window.id
                    )
                  )
                LIMIT 1
                """
            ).fetchone()
            if nonleaf_payload is not None:
                raise CompletionError(
                    "non-leaf search window retains page or run payload: "
                    f"{nonleaf_payload['id']}"
                )

            leaf_ids: list[int] = []
            union_closure_lines: list[str] = []
            split_count_drift_lines: list[str] = []
            split_count_drift_parent_total = 0
            split_count_drift_child_total = 0
            split_count_drift_absolute_delta = 0
            for repo in repos:
                repo_key = str(repo["repo_key"])
                windows = by_repo.get(repo_key, [])
                roots = [row for row in windows if row["parent_id"] is None]
                if len(roots) != 1:
                    raise CompletionError(
                        f"{repo_key} has {len(roots)} root search windows"
                    )
                root = roots[0]
                if (
                    int(root["start_epoch"]) != start
                    or int(root["end_epoch"]) != end
                ):
                    raise CompletionError(f"{repo_key} root interval binding mismatch")

                leaves = [row for row in windows if row["status"] == "done"]
                leaves.sort(key=lambda row: int(row["start_epoch"]))
                cursor = start
                for leaf in leaves:
                    leaf_start = int(leaf["start_epoch"])
                    leaf_end = int(leaf["end_epoch"])
                    if leaf_start != cursor:
                        relation = "overlap" if leaf_start < cursor else "gap"
                        raise CompletionError(
                            f"{repo_key} leaf-window {relation} at "
                            f"{format_utc_instant(cursor)}"
                        )
                    if leaf_end <= leaf_start:
                        raise CompletionError(f"{repo_key} has an empty leaf window")
                    cursor = leaf_end
                    leaf_ids.append(int(leaf["id"]))
                    total = int(leaf["expected_total"])
                    expected_pages = max(1, math.ceil(total / DEFAULT_PER_PAGE))
                    if total > GITHUB_FILTER_LIMIT:
                        raise CompletionError(
                            f"dense leaf window {leaf['id']} was not split"
                        )
                    window_id = int(leaf["id"])
                    pages = list(
                        conn.execute(
                            """
                            SELECT * FROM window_pages
                            WHERE window_id=? ORDER BY page_no
                            """,
                            (window_id,),
                        )
                    )
                    for page in pages:
                        _require_canonical_utc(
                            page["fetched_at"],
                            where=(
                                f"window_pages[{window_id},"
                                f"{page['page_no']}].fetched_at"
                            ),
                        )
                        for column in (
                            "payload_sha256",
                            "run_keys_sha256",
                        ):
                            if (
                                re.fullmatch(
                                    r"[0-9a-f]{64}",
                                    str(page[column]),
                                )
                                is None
                            ):
                                raise CompletionError(
                                    f"window page {window_id}:"
                                    f"{page['page_no']} has invalid {column}"
                                )
                    members = list(
                        conn.execute(
                            """
                            SELECT member.repo_key,member.run_id,
                                   member.run_attempt,
                                   member.metadata_sha256,
                                   run.metadata_sha256 AS run_metadata_sha256,
                                   run.created_at AS run_created_at
                            FROM window_runs member
                            JOIN runs run
                              ON run.repo_key=member.repo_key
                             AND run.run_id=member.run_id
                             AND run.run_attempt=member.run_attempt
                            WHERE member.window_id=?
                            ORDER BY member.repo_key,member.run_id,
                                     member.run_attempt
                            """,
                            (window_id,),
                        )
                    )
                    if len(members) != total or any(
                        member["repo_key"] != repo_key
                        or member["metadata_sha256"]
                        != member["run_metadata_sha256"]
                        or not (
                            leaf_start
                            <= parse_utc_instant(
                                _require_canonical_utc(
                                    member["run_created_at"],
                                    where=(
                                        f"window_runs[{window_id},"
                                        f"{member['repo_key']},"
                                        f"{member['run_id']},"
                                        f"{member['run_attempt']}]"
                                        ".run_created_at"
                                    ),
                                )
                            )
                            < leaf_end
                        )
                        for member in members
                    ):
                        raise CompletionError(
                            f"leaf window {window_id} run membership differs "
                            "from its repository, total, or canonical metadata"
                        )
                    union = conn.execute(
                        """
                        SELECT * FROM window_union_closures
                        WHERE window_id=?
                        """,
                        (window_id,),
                    ).fetchone()
                    if union is None:
                        if (
                            int(leaf["expected_pages"]) != expected_pages
                            or int(leaf["pages_done"]) != expected_pages
                            or int(leaf["raw_items"]) != total
                            or int(leaf["distinct_items"]) != total
                            or int(leaf["duplicate_items"]) != 0
                        ):
                            raise CompletionError(
                                f"leaf window {leaf['id']} has incomplete "
                                "page/count closure"
                            )
                        if [int(page["page_no"]) for page in pages] != list(
                            range(1, expected_pages + 1)
                        ):
                            raise CompletionError(
                                f"leaf window {leaf['id']} page sequence "
                                "is incomplete"
                            )
                        if any(
                            int(page["total_count"]) != total for page in pages
                        ):
                            raise CompletionError(
                                f"leaf window {leaf['id']} has unstable "
                                "total_count"
                            )
                        for page in pages:
                            page_no = int(page["page_no"])
                            expected_items = (
                                DEFAULT_PER_PAGE
                                if page_no < expected_pages
                                else total
                                - DEFAULT_PER_PAGE * (expected_pages - 1)
                            )
                            if (
                                int(page["item_count"]) != expected_items
                                or int(page["distinct_item_count"])
                                != expected_items
                                or int(page["duplicate_item_count"]) != 0
                            ):
                                raise CompletionError(
                                    f"ordinary leaf window {window_id} page "
                                    f"{page_no} item accounting is invalid"
                                )
                        stale_proof_rows = int(
                            conn.execute(
                                """
                                SELECT
                                  (SELECT COUNT(*)
                                   FROM convergence_passes
                                   WHERE window_id=?)
                                + (SELECT COUNT(*)
                                   FROM convergence_pass_pages
                                   WHERE window_id=?)
                                + (SELECT COUNT(*)
                                   FROM convergence_pass_runs
                                   WHERE window_id=?)
                                + (SELECT COUNT(*)
                                   FROM convergence_runs
                                   WHERE window_id=?)
                                """,
                                (
                                    window_id,
                                    window_id,
                                    window_id,
                                    window_id,
                                ),
                            ).fetchone()[0]
                        )
                        if stale_proof_rows:
                            raise CompletionError(
                                f"ordinary leaf window {leaf['id']} retains "
                                "convergence proof rows"
                            )
                    else:
                        if leaf_end - leaf_start != 1:
                            raise CompletionError(
                                f"union leaf window {leaf['id']} is not one second"
                            )
                        if pages:
                            raise CompletionError(
                                f"union leaf window {leaf['id']} also has "
                                "ordinary page rows"
                            )
                        passes = list(
                            conn.execute(
                                """
                                SELECT * FROM convergence_passes
                                WHERE window_id=? ORDER BY pass_no
                                """,
                                (window_id,),
                            )
                        )
                        pass_numbers = [
                            int(item["pass_no"]) for item in passes
                        ]
                        first_pass_no = int(union["first_pass_no"])
                        last_pass_no = int(union["last_pass_no"])
                        if (
                            len(passes) != int(union["pass_count"])
                            or not passes
                            or pass_numbers
                            != list(
                                range(first_pass_no, last_pass_no + 1)
                            )
                        ):
                            raise CompletionError(
                                f"union leaf window {leaf['id']} has an "
                                "invalid pass sequence"
                            )
                        observed_run_passes: dict[
                            tuple[str, int, int], list[int]
                        ] = {}
                        observed_run_metadata: dict[
                            tuple[str, int, int], str
                        ] = {}
                        for proof_pass in passes:
                            pass_no = int(proof_pass["pass_no"])
                            _require_canonical_utc(
                                proof_pass["observed_at"],
                                where=(
                                    f"convergence_passes[{window_id},"
                                    f"{pass_no}].observed_at"
                                ),
                            )
                            for column in (
                                "page_payload_set_sha256",
                                "run_keys_sha256",
                            ):
                                if (
                                    re.fullmatch(
                                        r"[0-9a-f]{64}",
                                        str(proof_pass[column]),
                                    )
                                    is None
                                ):
                                    raise CompletionError(
                                        f"convergence pass {window_id}:"
                                        f"{pass_no} has invalid {column}"
                                    )
                            pass_raw = int(proof_pass["raw_item_count"])
                            pass_distinct = int(
                                proof_pass["distinct_item_count"]
                            )
                            if (
                                int(proof_pass["total_count"]) != total
                                or int(proof_pass["page_count"])
                                != expected_pages
                                or pass_raw != total
                                or pass_distinct > total
                                or int(proof_pass["duplicate_item_count"])
                                != pass_raw - pass_distinct
                                or int(
                                    proof_pass["accumulated_distinct_count"]
                                )
                                > total
                            ):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} has "
                                    "invalid pass accounting"
                                )
                            proof_pages = list(
                                conn.execute(
                                    """
                                    SELECT * FROM convergence_pass_pages
                                    WHERE window_id=? AND pass_no=?
                                    ORDER BY page_no
                                    """,
                                    (window_id, pass_no),
                                )
                            )
                            if [
                                int(page["page_no"])
                                for page in proof_pages
                            ] != list(range(1, expected_pages + 1)):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} pass "
                                    f"{pass_no} page sequence is incomplete"
                                )
                            page_lines: list[str] = []
                            for page in proof_pages:
                                page_no = int(page["page_no"])
                                for column in (
                                    "payload_sha256",
                                    "run_keys_sha256",
                                ):
                                    if (
                                        re.fullmatch(
                                            r"[0-9a-f]{64}",
                                            str(page[column]),
                                        )
                                        is None
                                    ):
                                        raise CompletionError(
                                            f"convergence page {window_id}:"
                                            f"{pass_no}:{page_no} has "
                                            f"invalid {column}"
                                        )
                                item_count = int(page["item_count"])
                                page_distinct = int(
                                    page["distinct_item_count"]
                                )
                                expected_items = (
                                    DEFAULT_PER_PAGE
                                    if page_no < expected_pages
                                    else total
                                    - DEFAULT_PER_PAGE
                                    * (expected_pages - 1)
                                )
                                if (
                                    int(page["total_count"]) != total
                                    or item_count != expected_items
                                    or page_distinct > item_count
                                    or int(page["duplicate_item_count"])
                                    != item_count - page_distinct
                                ):
                                    raise CompletionError(
                                        f"union leaf window {leaf['id']} "
                                        f"pass {pass_no} page accounting "
                                        "is invalid"
                                    )
                                page_lines.append(
                                    f"{page_no}\t{page['total_count']}\t"
                                    f"{item_count}\t{page_distinct}\t"
                                    f"{page['duplicate_item_count']}\t"
                                    f"{page['payload_sha256']}\t"
                                    f"{page['run_keys_sha256']}"
                                )
                            if (
                                sum(
                                    int(page["item_count"])
                                    for page in proof_pages
                                )
                                != pass_raw
                                or _hash_lines(page_lines)
                                != str(
                                    proof_pass[
                                        "page_payload_set_sha256"
                                    ]
                                )
                            ):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} pass "
                                    f"{pass_no} page proof digest mismatch"
                                )
                            pass_members = list(
                                conn.execute(
                                    """
                                    SELECT repo_key,run_id,run_attempt,
                                           metadata_sha256
                                    FROM convergence_pass_runs
                                    WHERE window_id=? AND pass_no=?
                                    ORDER BY repo_key,run_id,run_attempt
                                    """,
                                    (window_id, pass_no),
                                )
                            )
                            pass_member_digest = _hash_lines(
                                f"{member['repo_key']}\t"
                                f"{member['run_id']}\t"
                                f"{member['run_attempt']}\t"
                                f"{member['metadata_sha256']}"
                                for member in pass_members
                            )
                            if (
                                len(pass_members) != pass_distinct
                                or pass_member_digest
                                != str(proof_pass["run_keys_sha256"])
                            ):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} pass "
                                    f"{pass_no} run-set proof mismatch"
                                )
                            for member in pass_members:
                                key = (
                                    str(member["repo_key"]),
                                    int(member["run_id"]),
                                    int(member["run_attempt"]),
                                )
                                if key[0] != repo_key:
                                    raise CompletionError(
                                        f"union leaf window {leaf['id']} pass "
                                        f"{pass_no} contains a cross-repository "
                                        "run"
                                    )
                                metadata_sha256 = str(
                                    member["metadata_sha256"]
                                )
                                previous_metadata = (
                                    observed_run_metadata.get(key)
                                )
                                if (
                                    previous_metadata is not None
                                    and previous_metadata
                                    != metadata_sha256
                                ):
                                    raise CompletionError(
                                        f"union leaf window {leaf['id']} "
                                        f"run {key} changed metadata "
                                        "across passes"
                                    )
                                observed_run_metadata[key] = (
                                    metadata_sha256
                                )
                                observed_run_passes.setdefault(
                                    key, []
                                ).append(pass_no)
                            reconstructed_minimum = min(
                                (
                                    len(observed)
                                    for observed in (
                                        observed_run_passes.values()
                                    )
                                ),
                                default=0,
                            )
                            if (
                                int(
                                    proof_pass[
                                        "accumulated_distinct_count"
                                    ]
                                )
                                != len(observed_run_passes)
                                or int(
                                    proof_pass[
                                        "min_observation_count"
                                    ]
                                )
                                != reconstructed_minimum
                            ):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} pass "
                                    f"{pass_no} cumulative proof mismatch"
                                )
                        observed_pages = sum(
                            int(item["page_count"]) for item in passes
                        )
                        observed_items = sum(
                            int(item["raw_item_count"]) for item in passes
                        )
                        pass_set_sha256 = (
                            self._convergence_pass_set_sha256(
                                conn, window_id=window_id
                            )
                        )
                        candidates = list(
                            conn.execute(
                                """
                                SELECT * FROM convergence_runs
                                WHERE window_id=?
                                ORDER BY repo_key,run_id,run_attempt
                                """,
                                (window_id,),
                            )
                        )
                        if len(candidates) != total:
                            raise CompletionError(
                                f"union leaf window {leaf['id']} candidate "
                                "count differs from total_count"
                            )
                        _require_canonical_utc(
                            union["closed_at"],
                            where=(
                                f"window_union_closures[{window_id}]"
                                ".closed_at"
                            ),
                        )
                        for column in (
                            "pass_set_sha256",
                            "run_keys_sha256",
                        ):
                            if (
                                re.fullmatch(
                                    r"[0-9a-f]{64}",
                                    str(union[column]),
                                )
                                is None
                            ):
                                raise CompletionError(
                                    f"union closure {window_id} has invalid "
                                    f"{column}"
                                )
                        candidate_digest = hashlib.sha256()
                        candidate_keys: set[
                            tuple[str, int, int]
                        ] = set()
                        for candidate in candidates:
                            candidate_key = (
                                str(candidate["repo_key"]),
                                int(candidate["run_id"]),
                                int(candidate["run_attempt"]),
                            )
                            candidate_keys.add(candidate_key)
                            if candidate_key[0] != repo_key:
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} has a "
                                    "cross-repository convergence candidate"
                                )
                            observed_passes = observed_run_passes.get(
                                candidate_key, []
                            )
                            observation_count = int(
                                candidate["observation_count"]
                            )
                            first_pass = int(candidate["first_pass"])
                            last_pass = int(candidate["last_pass"])
                            if (
                                observation_count < 2
                                or observation_count
                                != len(observed_passes)
                                or not observed_passes
                                or first_pass != observed_passes[0]
                                or last_pass != observed_passes[-1]
                                or str(candidate["metadata_sha256"])
                                != observed_run_metadata.get(candidate_key)
                            ):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} candidate "
                                    "observation proof is invalid"
                                )
                            metadata_blob = candidate["metadata_blob"]
                            if not isinstance(metadata_blob, bytes):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} metadata "
                                    "is not a BLOB"
                                )
                            try:
                                metadata_bytes = strict_bounded_zlib_decode(
                                    metadata_blob,
                                    expected_raw_size=None,
                                    expected_sha256=str(
                                        candidate["metadata_sha256"]
                                    ),
                                    max_raw_size=MAX_RUN_METADATA_BYTES,
                                    max_compressed_size=(
                                        MAX_RUN_METADATA_COMPRESSED_BYTES
                                    ),
                                    where=(
                                        f"union leaf window {leaf['id']} "
                                        "metadata"
                                    ),
                                )
                            except ZlibEvidenceError as exc:
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} metadata "
                                    f"is corrupt: {exc}"
                                ) from exc
                            self._validate_stored_run_projection(
                                candidate,
                                metadata_bytes,
                                start_epoch=start,
                                end_epoch=end,
                                where=(
                                    "convergence_runs["
                                    f"{window_id},{candidate['repo_key']},"
                                    f"{candidate['run_id']},"
                                    f"{candidate['run_attempt']}]"
                                ),
                            )
                            metadata_sha256 = _sha256_bytes(metadata_bytes)
                            candidate_digest.update(
                                (
                                    f"{candidate['repo_key']}\t"
                                    f"{candidate['run_id']}\t"
                                    f"{candidate['run_attempt']}\t"
                                    f"{metadata_sha256}\n"
                                ).encode()
                            )
                        if candidate_keys != set(observed_run_passes):
                            raise CompletionError(
                                f"union leaf window {leaf['id']} pass "
                                "membership and candidate sets disagree"
                            )
                        candidate_member_rows = {
                            (
                                str(candidate["repo_key"]),
                                int(candidate["run_id"]),
                                int(candidate["run_attempt"]),
                                str(candidate["metadata_sha256"]),
                            )
                            for candidate in candidates
                        }
                        final_member_rows = {
                            (
                                str(member["repo_key"]),
                                int(member["run_id"]),
                                int(member["run_attempt"]),
                                str(member["metadata_sha256"]),
                            )
                            for member in members
                        }
                        if candidate_member_rows != final_member_rows:
                            raise CompletionError(
                                f"union leaf window {leaf['id']} candidate "
                                "and final member sets disagree"
                            )
                        candidate_sha256 = candidate_digest.hexdigest()
                        minimum_observations = min(
                            (
                                len(observations)
                                for observations in (
                                    observed_run_passes.values()
                                )
                            ),
                            default=0,
                        )
                        if (
                            int(union["total_count"]) != total
                            or int(union["distinct_run_count"]) != total
                            or int(union["min_observation_count"])
                            != minimum_observations
                            or minimum_observations < 2
                            or int(union["observed_page_count"])
                            != observed_pages
                            or int(union["observed_item_count"])
                            != observed_items
                            or str(union["pass_set_sha256"])
                            != pass_set_sha256
                            or str(union["run_keys_sha256"])
                            != candidate_sha256
                            or int(leaf["expected_pages"]) != expected_pages
                            or int(leaf["pages_done"]) != observed_pages
                            or int(leaf["raw_items"]) != observed_items
                            or int(leaf["distinct_items"]) != total
                            or int(leaf["duplicate_items"])
                            != observed_items - total
                        ):
                            raise CompletionError(
                                f"union leaf window {leaf['id']} closure "
                                "accounting is invalid"
                            )
                        union_closure_lines.append(
                            f"U\t{repo_key}\t{leaf_start}\t{leaf_end}\t"
                            f"{union['pass_count']}\t"
                            f"{union['first_pass_no']}\t"
                            f"{union['last_pass_no']}\t{observed_pages}\t"
                            f"{observed_items}\t{total}\t"
                            f"{minimum_observations}\t{pass_set_sha256}\t"
                            f"{candidate_sha256}"
                        )
                    actual_leaf_digest = _hash_lines(
                        str(item["repo_key"])
                        + "\t"
                        + str(item["run_id"])
                        + "\t"
                        + str(item["run_attempt"])
                        + "\t"
                        + str(item["metadata_sha256"])
                        for item in members
                    )
                    if actual_leaf_digest != str(leaf["run_keys_sha256"]):
                        raise CompletionError(
                            f"leaf window {leaf['id']} run-set digest mismatch"
                        )
                if cursor != end:
                    raise CompletionError(
                        f"{repo_key} leaf windows stop at "
                        f"{format_utc_instant(cursor)}, expected "
                        f"{format_utc_instant(end)}"
                    )

                for window in windows:
                    window_id = int(window["id"])
                    children = sorted(
                        by_parent.get(window_id, []),
                        key=lambda row: int(row["start_epoch"]),
                    )
                    if window["status"] == "done":
                        if children:
                            raise CompletionError(
                                f"done window {window_id} unexpectedly has children"
                            )
                        continue
                    if window["status"] != "split":
                        raise CompletionError(
                            f"window {window_id} has nonterminal status "
                            f"{window['status']!r}"
                        )
                    if (
                        window["expected_pages"] is not None
                        or int(window["pages_done"]) != 0
                        or int(window["raw_items"]) != 0
                        or int(window["distinct_items"]) != 0
                        or int(window["duplicate_items"]) != 0
                        or window["run_keys_sha256"] is not None
                    ):
                        raise CompletionError(
                            f"split window {window_id} retains leaf page/run "
                            "accounting"
                        )
                    if len(children) != 2:
                        raise CompletionError(
                            f"split window {window_id} has {len(children)} children"
                        )
                    if (
                        int(children[0]["start_epoch"])
                        != int(window["start_epoch"])
                        or int(children[0]["end_epoch"])
                        != int(children[1]["start_epoch"])
                        or int(children[1]["end_epoch"]) != int(window["end_epoch"])
                    ):
                        raise CompletionError(
                            f"split window {window_id} children overlap or leave a gap"
                        )
                    child_total = sum(int(child["expected_total"]) for child in children)
                    parent_total = int(window["expected_total"])
                    if child_total != parent_total:
                        split_count_drift_parent_total += parent_total
                        split_count_drift_child_total += child_total
                        split_count_drift_absolute_delta += abs(
                            parent_total - child_total
                        )
                        split_count_drift_lines.append(
                            f"S\t{repo_key}\t{window['start_epoch']}\t"
                            f"{window['end_epoch']}\t{parent_total}\t"
                            f"{child_total}\t{parent_total - child_total}"
                        )

            if not leaf_ids and repos:
                raise CompletionError("inventory has no closed leaf windows")
            if leaf_ids:
                overlap = conn.execute(
                    """
                    SELECT wr.repo_key,wr.run_id,wr.run_attempt,
                           COUNT(*) AS appearances
                    FROM window_runs wr
                    JOIN search_windows w ON w.id=wr.window_id
                    WHERE w.status='done'
                    GROUP BY wr.repo_key,wr.run_id,wr.run_attempt
                    HAVING COUNT(*) > 1
                    LIMIT 1
                    """
                ).fetchone()
                if overlap is not None:
                    raise CompletionError(
                        "workflow run appears in overlapping leaf windows: "
                        f"{overlap['repo_key']}#{overlap['run_id']} attempt "
                        f"{overlap['run_attempt']}"
                    )

            unlinked_runs = [
                {
                    "repo": str(row["repo_key"]),
                    "run_id": int(row["run_id"]),
                    "run_attempt": int(row["run_attempt"]),
                    "metadata_sha256": str(row["metadata_sha256"]),
                }
                for row in conn.execute(
                    """
                    SELECT repo_key,run_id,run_attempt,metadata_sha256
                    FROM runs r
                    WHERE NOT EXISTS (
                        SELECT 1 FROM window_runs wr
                        WHERE wr.repo_key=r.repo_key
                          AND wr.run_id=r.run_id
                          AND wr.run_attempt=r.run_attempt
                    )
                    ORDER BY repo_key,run_id,run_attempt
                    """
                )
            ]
            unlinked_run_set_sha256 = _hash_lines(
                f"{row['repo']}\t{row['run_id']}\t{row['run_attempt']}\t"
                f"{row['metadata_sha256']}"
                for row in unlinked_runs
            )

            duplicate_run = conn.execute(
                """
                SELECT repo_key,run_id,COUNT(*) AS versions
                FROM runs
                GROUP BY repo_key,run_id
                HAVING COUNT(*) != 1
                LIMIT 1
                """
            ).fetchone()
            if duplicate_run is not None:
                raise CompletionError(
                    "inventory contains more than one attempt ceiling for a "
                    "workflow run: "
                    f"{duplicate_run['repo_key']}#{duplicate_run['run_id']}"
                )
            run_count = int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
            run_digest = hashlib.sha256()
            attempt_digest = hashlib.sha256()
            expected_attempt_count = 0
            repo_accumulators: dict[str, dict[str, Any]] = {
                str(repo["repo_key"]): {
                    "repo": str(repo["repo_key"]),
                    "canonical": str(repo["canonical"]),
                    "ordinal": int(repo["ordinal"]),
                    "run_count": 0,
                    "expected_attempt_count": 0,
                    "_run_digest": hashlib.sha256(),
                    "_attempt_digest": hashlib.sha256(),
                }
                for repo in repos
            }
            for row in conn.execute(
                """
                SELECT *
                FROM runs ORDER BY repo_key,run_id,run_attempt
                """
            ):
                run_attempt = int(row["run_attempt"])
                if run_attempt < 1:
                    raise CompletionError(
                        f"run {row['repo_key']}#{row['run_id']} has a "
                        "non-positive run_attempt"
                    )
                blob = row["metadata_blob"]
                if not isinstance(blob, bytes):
                    raise CompletionError(
                        f"run {row['repo_key']}#{row['run_id']} metadata is not a BLOB"
                    )
                try:
                    metadata_bytes = strict_bounded_zlib_decode(
                        blob,
                        expected_raw_size=None,
                        expected_sha256=str(row["metadata_sha256"]),
                        max_raw_size=MAX_RUN_METADATA_BYTES,
                        max_compressed_size=(
                            MAX_RUN_METADATA_COMPRESSED_BYTES
                        ),
                        where=(
                            f"run {row['repo_key']}#{row['run_id']} metadata"
                        ),
                    )
                except ZlibEvidenceError as exc:
                    raise CompletionError(
                        f"run {row['repo_key']}#{row['run_id']} metadata is corrupt: "
                        f"{exc}"
                    ) from exc
                self._validate_stored_run_projection(
                    row,
                    metadata_bytes,
                    start_epoch=start,
                    end_epoch=end,
                    where=(
                        f"runs[{row['repo_key']},{row['run_id']},"
                        f"{row['run_attempt']}]"
                    ),
                )
                actual_metadata_sha = _sha256_bytes(metadata_bytes)
                line = (
                    f"{row['repo_key']}\t{row['run_id']}\t{run_attempt}\t"
                    f"{actual_metadata_sha}\n"
                )
                run_digest.update(line.encode("utf-8"))
                repo_item = repo_accumulators[str(row["repo_key"])]
                repo_item["run_count"] += 1
                repo_item["expected_attempt_count"] += run_attempt
                repo_item["_run_digest"].update(line.encode("utf-8"))
                for attempt in range(1, run_attempt + 1):
                    attempt_line = (
                        f"{row['repo_key']}\t{row['run_id']}\t{attempt}\n"
                    ).encode("utf-8")
                    attempt_digest.update(attempt_line)
                    repo_item["_attempt_digest"].update(attempt_line)
                expected_attempt_count += run_attempt
            run_set_sha = run_digest.hexdigest()
            expected_attempt_set_sha = attempt_digest.hexdigest()
            per_repo_ledger = []
            for repo in repos:
                item = repo_accumulators[str(repo["repo_key"])]
                per_repo_ledger.append(
                    {
                        "repo": item["repo"],
                        "canonical": item["canonical"],
                        "ordinal": item["ordinal"],
                        "run_count": item["run_count"],
                        "expected_attempt_count": item[
                            "expected_attempt_count"
                        ],
                        "run_set_sha256": item["_run_digest"].hexdigest(),
                        "expected_attempt_set_sha256": item[
                            "_attempt_digest"
                        ].hexdigest(),
                    }
                )
            per_repo_ledger_sha256 = _sha256_json(per_repo_ledger)

            closure_lines = [
                f"W\t{row['repo_key']}\t{row['start_epoch']}\t"
                f"{row['end_epoch']}\t{row['status']}\t"
                f"{row['expected_total']}\t{row['run_keys_sha256'] or ''}"
                for row in all_windows
            ]
            closure_lines.extend(
                f"P\t{row['repo_key']}\t{row['start_epoch']}\t"
                f"{row['end_epoch']}\t{row['page_no']}\t"
                f"{row['total_count']}\t{row['item_count']}\t"
                f"{row['distinct_item_count']}\t{row['duplicate_item_count']}\t"
                f"{row['payload_sha256']}\t{row['run_keys_sha256']}"
                for row in conn.execute(
                    """
                    SELECT w.repo_key,w.start_epoch,w.end_epoch,p.page_no,
                           p.total_count,p.item_count,p.distinct_item_count,
                           p.duplicate_item_count,p.payload_sha256,
                           p.run_keys_sha256
                    FROM window_pages p
                    JOIN search_windows w ON w.id=p.window_id
                    ORDER BY w.repo_key,w.start_epoch,w.end_epoch,p.page_no
                    """
                )
            )
            closure_lines.extend(sorted(union_closure_lines))
            sorted_split_count_drift_lines = sorted(
                split_count_drift_lines
            )
            closure_lines.extend(sorted_split_count_drift_lines)
            closure_sha = _hash_lines(closure_lines)
            split_count_drift_net = (
                split_count_drift_parent_total
                - split_count_drift_child_total
            )
            source_count_drift = {
                "windows": len(sorted_split_count_drift_lines),
                "parent_total": split_count_drift_parent_total,
                "child_total": split_count_drift_child_total,
                "net_parent_minus_children": split_count_drift_net,
                "absolute_delta": split_count_drift_absolute_delta,
                "sha256": _hash_lines(sorted_split_count_drift_lines),
                "semantics": (
                    "GitHub total_count observations at each split parent "
                    "versus its later child enumeration; nonzero means the "
                    "source cardinality changed or pagination contradicted "
                    "itself during inventory; zero means no such "
                    "contradiction was observed, not proof of an atomic "
                    "GitHub snapshot"
                ),
            }
            logical_document = {
                "schema": SCHEMA_VERSION,
                "sqlite_schema_sha256": _EXPECTED_SQLITE_SCHEMA_SHA256,
                "inventory_meta_sha256": _sha256_json(meta),
                "logical_table_ledgers": logical_table_ledgers,
                "repo_list_sha256": meta["repo_list_sha256"],
                "repo_scope_sha256": meta["repo_scope_sha256"],
                "start_epoch": start,
                "end_epoch": end,
                "script_sha256": meta["script_sha256"],
                "repo_count": len(repos),
                "run_count": run_count,
                "run_set_sha256": run_set_sha,
                "expected_attempt_count": expected_attempt_count,
                "expected_attempt_set_sha256": expected_attempt_set_sha,
                "per_repo_ledger_sha256": per_repo_ledger_sha256,
                "window_closure_sha256": closure_sha,
                "binding_upgrades_sha256": _sha256_json(
                    binding_upgrades
                ),
                "source_count_drift": source_count_drift,
                "unlinked_run_count": len(unlinked_runs),
                "unlinked_run_set_sha256": unlinked_run_set_sha256,
            }
            return {
                "meta": meta,
                "repo_count": len(repos),
                "run_count": run_count,
                "run_set_sha256": run_set_sha,
                "expected_attempt_count": expected_attempt_count,
                "expected_attempt_set_sha256": expected_attempt_set_sha,
                "per_repo_ledger": per_repo_ledger,
                "per_repo_ledger_sha256": per_repo_ledger_sha256,
                "window_closure_sha256": closure_sha,
                "sqlite_schema_sha256": _EXPECTED_SQLITE_SCHEMA_SHA256,
                "logical_table_ledgers": logical_table_ledgers,
                "db_logical_sha256": _sha256_json(logical_document),
                "leaf_window_count": len(leaf_ids),
                "request_count": request_count,
                "binding_upgrades": binding_upgrades,
                "source_count_drift": source_count_drift,
                "unlinked_runs": unlinked_runs,
                "unlinked_run_set_sha256": unlinked_run_set_sha256,
            }
        except sqlite3.Error as exc:
            raise CompletionError(
                f"inventory SQLite validation failed: {exc}"
            ) from exc
        finally:
            conn.close()

    def _validate_source_drift_reconciliation(
        self,
        validated: Mapping[str, Any],
        value: object,
        *,
        immutable: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise CompletionError(
                "source-drift reconciliation must be an object"
            )
        expected_top_fields = {
            "schema",
            "created_at",
            "producer_script_sha256",
            "semantics",
            "source_count_drift_sha256",
            "source_count_drift_windows",
            "root_count",
            "stored_run_count",
            "current_run_count",
            "retained_upstream_deleted_count",
            "new_current_run_count",
            "observed_union_run_count",
            "two_pass_exact",
            "observed_union_complete",
            "roots",
        }
        if set(value) != expected_top_fields:
            raise CompletionError(
                "source-drift reconciliation has extra/missing fields"
            )
        if value.get("schema") != SOURCE_DRIFT_RECONCILIATION_SCHEMA:
            raise CompletionError(
                "source-drift reconciliation schema is unsupported"
            )
        created_at = _require_canonical_utc(
            value.get("created_at"),
            where="source-drift reconciliation created_at",
        )
        meta = validated["meta"]
        producer_script_sha256 = value.get("producer_script_sha256")
        allowed_producers = {meta["script_sha256"]}
        for upgrade in validated["binding_upgrades"]:
            allowed_producers.add(str(upgrade["from_script_sha256"]))
            allowed_producers.add(str(upgrade["to_script_sha256"]))
        if (
            not isinstance(producer_script_sha256, str)
            or producer_script_sha256 not in allowed_producers
        ):
            raise CompletionError(
                "source-drift reconciliation producer is outside the "
                "audited inventory producer chain"
            )
        semantics = (
            "two exact repeated live enumerations were unioned with the "
            "frozen observed membership; upstream deletions remain retained "
            "and stable new runs are appended"
        )
        if value.get("semantics") != semantics:
            raise CompletionError(
                "source-drift reconciliation semantics are invalid"
            )
        drift = validated["source_count_drift"]
        if (
            value.get("source_count_drift_sha256") != drift["sha256"]
            or value.get("source_count_drift_windows") != drift["windows"]
        ):
            raise CompletionError(
                "source-drift reconciliation differs from SQLite drift"
            )
        raw_roots = value.get("roots")
        if not isinstance(raw_roots, list):
            raise CompletionError(
                "source-drift reconciliation roots must be a list"
            )

        connection = self.connect(
            readonly=True,
            immutable=immutable,
        )
        try:
            expected_roots = _minimal_source_drift_roots(connection)
            if not expected_roots:
                raise CompletionError(
                    "source-drift reconciliation exists without SQLite drift"
                )
            if len(raw_roots) != len(expected_roots):
                raise CompletionError(
                    "source-drift reconciliation root count differs from SQLite"
                )
            proof_rows = _load_source_drift_proofs(connection)
            expected_window_ids = {
                int(root["window_id"]) for root in expected_roots
            }
            if set(proof_rows) != expected_window_ids:
                raise CompletionError(
                    "persisted source-drift proof roots differ from SQLite "
                    "drift roots"
                )
            validated_roots: list[dict[str, Any]] = []
            new_union_members: dict[tuple[str, int, int], str] = {}
            previous_interval: tuple[str, int] | None = None
            for index, (raw_root, expected_root) in enumerate(
                zip(raw_roots, expected_roots, strict=True)
            ):
                if not isinstance(raw_root, dict):
                    raise CompletionError(
                        f"source-drift reconciliation root {index} is not an object"
                    )
                repo_key = str(expected_root["repo"])
                start_epoch = int(expected_root["start_epoch"])
                end_epoch = int(expected_root["end_epoch"])
                if (
                    previous_interval is not None
                    and previous_interval[0] == repo_key
                    and previous_interval[1] > start_epoch
                ):
                    raise CompletionError(
                        "source-drift reconciliation roots overlap"
                    )
                previous_interval = (repo_key, end_epoch)
                persisted_root = proof_rows[
                    int(expected_root["window_id"])
                ]
                _require_exact_json(
                    raw_root,
                    persisted_root,
                    where=f"source-drift reconciliation roots[{index}]",
                )
                stored = _stored_reconciliation_members(
                    connection,
                    expected_root,
                )
                raw_members = raw_root.get("current_members")
                if not isinstance(raw_members, list):
                    raise CompletionError(
                        f"source-drift reconciliation root {index} members "
                        "must be a list"
                    )
                current: dict[tuple[int, int], str] = {}
                canonical_members: list[list[int | str]] = []
                for member_index, member in enumerate(raw_members):
                    if not isinstance(member, list) or len(member) != 3:
                        raise CompletionError(
                            "source-drift reconciliation member must be "
                            "[run_id,run_attempt,metadata_sha256]"
                        )
                    run_id, run_attempt, metadata_sha256 = member
                    if (
                        isinstance(run_id, bool)
                        or not isinstance(run_id, int)
                        or run_id < 1
                        or isinstance(run_attempt, bool)
                        or not isinstance(run_attempt, int)
                        or run_attempt < 1
                        or not isinstance(metadata_sha256, str)
                        or re.fullmatch(r"[0-9a-f]{64}", metadata_sha256)
                        is None
                    ):
                        raise CompletionError(
                            "source-drift reconciliation contains an invalid "
                            f"member at root {index}, index {member_index}"
                        )
                    key = (run_id, run_attempt)
                    if key in current:
                        raise CompletionError(
                            "source-drift reconciliation contains duplicate "
                            f"member {repo_key}#{run_id}/{run_attempt}"
                        )
                    current[key] = metadata_sha256
                    canonical_members.append(
                        [run_id, run_attempt, metadata_sha256]
                    )
                canonical_members.sort(key=lambda member: (member[0], member[1]))
                if raw_members != canonical_members:
                    raise CompletionError(
                        "source-drift reconciliation members are not canonical"
                    )
                new_keys = set(current).difference(stored)
                retained_keys = set(stored).difference(current)
                stored_membership_sha, stored_metadata_sha = (
                    _run_projection_digests(repo_key, stored)
                )
                current_membership_sha, current_metadata_sha = (
                    _run_projection_digests(repo_key, current)
                )
                observed_union = dict(stored)
                observed_union.update(
                    (key, current[key]) for key in new_keys
                )
                (
                    observed_union_membership_sha,
                    _observed_union_metadata_sha,
                ) = _run_projection_digests(repo_key, observed_union)
                metadata_changed = sum(
                    stored[key] != metadata_sha256
                    for key, metadata_sha256 in current.items()
                    if key in stored
                )
                raw_passes = raw_root.get("passes")
                if not isinstance(raw_passes, list) or len(raw_passes) != 2:
                    raise CompletionError(
                        "source-drift reconciliation requires two exact passes"
                    )
                validated_passes: list[dict[str, Any]] = []
                for pass_number, raw_pass in enumerate(raw_passes, start=1):
                    if not isinstance(raw_pass, dict):
                        raise CompletionError(
                            "source-drift reconciliation pass is not an object"
                        )
                    expected_pass_fields = {
                        "pass",
                        "page_observation_count",
                        "request_count",
                        "membership_sha256",
                        "metadata_sha256",
                        "page_ledger_sha256",
                    }
                    if set(raw_pass) != expected_pass_fields:
                        raise CompletionError(
                            "source-drift reconciliation pass has "
                            "extra/missing fields"
                        )
                    page_count = raw_pass.get("page_observation_count")
                    request_count = raw_pass.get("request_count")
                    page_ledger_sha = raw_pass.get("page_ledger_sha256")
                    if (
                        raw_pass.get("pass") != pass_number
                        or isinstance(page_count, bool)
                        or not isinstance(page_count, int)
                        or page_count < 1
                        or isinstance(request_count, bool)
                        or not isinstance(request_count, int)
                        or request_count < page_count
                        or raw_pass.get("membership_sha256")
                        != current_membership_sha
                        or raw_pass.get("metadata_sha256")
                        != current_metadata_sha
                        or not isinstance(page_ledger_sha, str)
                        or re.fullmatch(r"[0-9a-f]{64}", page_ledger_sha)
                        is None
                    ):
                        raise CompletionError(
                            "source-drift reconciliation pass proof is invalid"
                        )
                    validated_passes.append(dict(raw_pass))
                expected_interval = {
                    "start": format_utc_instant(start_epoch),
                    "end": format_utc_instant(end_epoch),
                    "semantics": "[start,end)",
                }
                root_completed_at = _require_canonical_utc(
                    raw_root.get("completed_at"),
                    where=(
                        f"source-drift reconciliation roots[{index}] "
                        "completed_at"
                    ),
                )
                expected_value = {
                    "window_id": int(expected_root["window_id"]),
                    "completed_at": root_completed_at,
                    "producer_script_sha256": producer_script_sha256,
                    "repo": repo_key,
                    "interval": expected_interval,
                    "parent_total": int(expected_root["parent_total"]),
                    "child_total": int(expected_root["child_total"]),
                    "stored_count": len(stored),
                    "stored_membership_sha256": stored_membership_sha,
                    "stored_metadata_sha256": stored_metadata_sha,
                    "current_count": len(current),
                    "current_membership_sha256": current_membership_sha,
                    "current_metadata_sha256": current_metadata_sha,
                    "retained_upstream_deleted_count": len(retained_keys),
                    "new_current_count": len(new_keys),
                    "observed_union_count": len(observed_union),
                    "observed_union_membership_sha256": (
                        observed_union_membership_sha
                    ),
                    "metadata_changed_count": metadata_changed,
                    "current_members": canonical_members,
                    "passes": validated_passes,
                }
                _require_exact_json(
                    raw_root,
                    expected_value,
                    where=f"source-drift reconciliation roots[{index}]",
                )
                validated_roots.append(expected_value)
                for run_id, run_attempt in new_keys:
                    union_key = (repo_key, run_id, run_attempt)
                    if union_key in new_union_members:
                        raise CompletionError(
                            "source-drift reconciliation repeats a new run "
                            "across roots"
                        )
                    new_union_members[union_key] = current[
                        (run_id, run_attempt)
                    ]
        finally:
            connection.close()

        unlinked_members = {
            (
                str(row["repo"]),
                int(row["run_id"]),
                int(row["run_attempt"]),
            ): str(row["metadata_sha256"])
            for row in validated["unlinked_runs"]
        }
        if unlinked_members != new_union_members:
            raise CompletionError(
                "unlinked inventory runs differ from the stable live "
                "source-drift delta"
            )
        expected_created_at = max(
            str(root["completed_at"]) for root in validated_roots
        )
        if created_at != expected_created_at:
            raise CompletionError(
                "source-drift reconciliation created_at differs from its "
                "last completed root"
            )
        expected_value = _source_drift_reconciliation_payload(
            script_sha256=producer_script_sha256,
            source_count_drift=drift,
            roots=validated_roots,
        )
        _require_exact_json(
            value,
            expected_value,
            where="source-drift reconciliation",
        )
        return expected_value

    def completion_receipt(
        self,
        *,
        allow_nonproduction: bool = False,
        source_drift_reconciliation: object = None,
    ) -> dict[str, Any]:
        self._freeze_for_receipt()
        validated = self._validate_and_digests()
        meta = validated["meta"]
        smoke = meta["smoke"] == "1"
        drift_windows = validated["source_count_drift"]["windows"]
        reconciliation: dict[str, Any] | None = None
        if drift_windows:
            if source_drift_reconciliation is not None:
                reconciliation = self._validate_source_drift_reconciliation(
                    validated,
                    source_drift_reconciliation,
                )
        elif source_drift_reconciliation is not None:
            raise CompletionError(
                "source-drift reconciliation was supplied without SQLite drift"
            )
        unlinked_run_count = len(validated["unlinked_runs"])
        source_snapshot_stable = (
            drift_windows == 0
            and unlinked_run_count == 0
        ) or reconciliation is not None
        source_snapshot_mode = (
            "reconciled-observed-union"
            if reconciliation is not None
            else "count-consistent-enumeration"
            if source_snapshot_stable
            else "unreconciled-source-drift"
        )
        production_complete = not smoke and source_snapshot_stable
        if not production_complete and not allow_nonproduction:
            reason = (
                "smoke inventory"
                if smoke
                else "unlinked runs lack a source-drift reconciliation"
                if unlinked_run_count
                else "source count drift prevents a stable production snapshot"
            )
            raise CompletionError(
                f"production inventory receipt refused: {reason}; "
                "request an explicit diagnostic non-production receipt if "
                "the incomplete proof must be retained"
            )
        database_path = Path(self.path)
        before = database_path.stat()
        artifact_sha256 = _sha256_file(database_path)
        after = database_path.stat()
        if (
            before.st_size,
            before.st_mtime_ns,
            before.st_ino,
        ) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise CompletionError(
                "inventory database changed while its receipt was built"
            )
        if self._validate_and_digests() != validated:
            raise CompletionError(
                "inventory logical contents changed while its receipt was built"
            )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "completed_at": _utc_now(),
            "enumeration_complete": True,
            "source_snapshot_stable": source_snapshot_stable,
            "source_snapshot_mode": source_snapshot_mode,
            "production_complete": production_complete,
            "mode": (
                "production"
                if production_complete
                else "smoke-diagnostic"
                if smoke
                else "unstable-diagnostic"
            ),
            "database": self.path,
            "database_artifact": {
                "path": self.path,
                "byte_size": after.st_size,
                "sha256": artifact_sha256,
            },
            "repo_list": {
                "path": meta["repo_list_path"],
                "sha256": meta["repo_list_sha256"],
                "scope_sha256": meta["repo_scope_sha256"],
                "repos": validated["repo_count"],
                "original_repos": int(meta["original_repo_count"]),
                "unresolved": int(meta["unresolved_count"]),
            },
            "interval": {
                "start": meta["start_utc"],
                "end": meta["end_utc"],
                "semantics": "[start,end)",
            },
            "script_sha256": meta["script_sha256"],
            "metadata_encoding": meta["metadata_encoding"],
            "run_count": validated["run_count"],
            "expected_attempt_count": validated[
                "expected_attempt_count"
            ],
            "expected_attempt_set_sha256": validated[
                "expected_attempt_set_sha256"
            ],
            "per_repo_ledger": validated["per_repo_ledger"],
            "per_repo_ledger_sha256": validated[
                "per_repo_ledger_sha256"
            ],
            "leaf_window_count": validated["leaf_window_count"],
            "request_count": validated["request_count"],
            "run_set_sha256": validated["run_set_sha256"],
            "window_closure_sha256": validated["window_closure_sha256"],
            "db_logical_sha256": validated["db_logical_sha256"],
            "binding_upgrades": validated["binding_upgrades"],
            "source_count_drift": validated["source_count_drift"],
            "source_drift_reconciliation": reconciliation,
        }
        if _sha256_file(database_path) != artifact_sha256:
            raise CompletionError(
                "inventory database changed after its receipt was built"
            )
        return receipt


def atomic_write_json(path: str | os.PathLike[str], document: Any) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def verify_inventory_completion_receipt(
    database_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    *,
    require_production: bool = True,
    expected_original_database_path: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Verify a frozen inventory and its byte-bound completion receipt."""

    raw_database = Path(database_path).expanduser()
    raw_receipt = Path(receipt_path).expanduser()
    for path, label in (
        (raw_database, "inventory database"),
        (raw_receipt, "inventory receipt"),
    ):
        if path.is_symlink() or not path.is_file():
            raise CompletionError(f"{label} is missing or unsafe: {path}")
    database = raw_database.resolve()
    receipt_file = raw_receipt.resolve()
    _require_safe_checkpoint_sidecars(database)
    raw = _read_bounded_regular_file(
        receipt_file,
        max_bytes=MAX_INVENTORY_RECEIPT_BYTES,
        where="inventory receipt",
    )

    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CompletionError(
                    f"inventory receipt contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionError(f"inventory receipt is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CompletionError("inventory receipt root must be an object")
    expected_top_fields = {
        "schema",
        "completed_at",
        "enumeration_complete",
        "source_snapshot_stable",
        "source_snapshot_mode",
        "production_complete",
        "mode",
        "database",
        "database_artifact",
        "repo_list",
        "interval",
        "script_sha256",
        "metadata_encoding",
        "run_count",
        "expected_attempt_count",
        "expected_attempt_set_sha256",
        "per_repo_ledger",
        "per_repo_ledger_sha256",
        "leaf_window_count",
        "request_count",
        "run_set_sha256",
        "window_closure_sha256",
        "db_logical_sha256",
        "binding_upgrades",
        "source_count_drift",
        "source_drift_reconciliation",
    }
    if set(value) != expected_top_fields:
        raise CompletionError(
            "inventory receipt root has extra/missing fields"
        )
    if value.get("schema") != RECEIPT_SCHEMA:
        raise CompletionError(
            f"inventory receipt schema must be {RECEIPT_SCHEMA!r}"
        )
    completed_at = _require_canonical_utc(
        value["completed_at"],
        where="inventory receipt completed_at",
    )
    original_input = (
        raw_database
        if expected_original_database_path is None
        else Path(expected_original_database_path).expanduser()
    )
    original = Path(os.path.abspath(original_input))
    if value.get("database") != str(original):
        raise CompletionError(
            "inventory receipt database path differs from the expected "
            "original path"
        )
    artifact = value.get("database_artifact")
    if not isinstance(artifact, dict):
        raise CompletionError("inventory receipt lacks its database artifact")
    if artifact.get("path") != str(original):
        raise CompletionError(
            "inventory receipt artifact path differs from its database path"
        )

    with tempfile.TemporaryDirectory(
        prefix="cppmega-inventory-verify-"
    ) as snapshot_directory:
        snapshot = Path(snapshot_directory) / "inventory.sqlite3"
        (
            database_size,
            database_sha256,
            database_identity,
        ) = _copy_database_snapshot_once(database, snapshot)
        if (
            artifact.get("byte_size") != database_size
            or artifact.get("sha256") != database_sha256
        ):
            raise CompletionError(
                "inventory database bytes differ from the completion receipt"
            )
        snapshot.chmod(0o400)
        snapshot_before = snapshot.stat()
        snapshot_identity = (
            snapshot_before.st_dev,
            snapshot_before.st_ino,
            snapshot_before.st_size,
            snapshot_before.st_mtime_ns,
            snapshot_before.st_ctime_ns,
        )
        snapshot_inventory = InventoryDB(
            snapshot,
            initialize_schema=False,
        )
        validated = snapshot_inventory._validate_and_digests(
            immutable=True
        )
        drift_windows = validated["source_count_drift"]["windows"]
        reconciliation: dict[str, Any] | None = None
        raw_reconciliation = value.get("source_drift_reconciliation")
        if drift_windows:
            if raw_reconciliation is not None:
                reconciliation = (
                    snapshot_inventory._validate_source_drift_reconciliation(
                        validated,
                        raw_reconciliation,
                        immutable=True,
                    )
                )
        elif raw_reconciliation is not None:
            raise CompletionError(
                "inventory receipt reconciles nonexistent source drift"
            )
        database_stable = (
            drift_windows == 0
            and not validated["unlinked_runs"]
        ) or reconciliation is not None
        source_snapshot_mode = (
            "reconciled-observed-union"
            if reconciliation is not None
            else "count-consistent-enumeration"
            if database_stable
            else "unreconciled-source-drift"
        )
        snapshot_after = snapshot.stat()
        if snapshot_identity != (
            snapshot_after.st_dev,
            snapshot_after.st_ino,
            snapshot_after.st_size,
            snapshot_after.st_mtime_ns,
            snapshot_after.st_ctime_ns,
        ):
            raise CompletionError(
                "private inventory database snapshot changed during "
                "logical validation"
            )

    final_database_stat = database.lstat()
    if (
        not stat.S_ISREG(final_database_stat.st_mode)
        or database_identity
        != (
            final_database_stat.st_dev,
            final_database_stat.st_ino,
            final_database_stat.st_size,
            final_database_stat.st_mtime_ns,
            final_database_stat.st_ctime_ns,
        )
    ):
        raise CompletionError(
            "inventory database changed while its receipt was verified"
        )
    _require_safe_checkpoint_sidecars(database)
    meta = validated["meta"]
    database_production = meta["smoke"] == "0" and database_stable
    expected_mode = (
        "production"
        if database_production
        else "smoke-diagnostic"
        if meta["smoke"] == "1"
        else "unstable-diagnostic"
    )
    expected_receipt = {
        "schema": RECEIPT_SCHEMA,
        "completed_at": completed_at,
        "enumeration_complete": True,
        "source_snapshot_stable": database_stable,
        "source_snapshot_mode": source_snapshot_mode,
        "production_complete": database_production,
        "mode": expected_mode,
        "database": str(original),
        "database_artifact": {
            "path": str(original),
            "byte_size": database_size,
            "sha256": database_sha256,
        },
        "repo_list": {
            "path": meta["repo_list_path"],
            "sha256": meta["repo_list_sha256"],
            "scope_sha256": meta["repo_scope_sha256"],
            "repos": validated["repo_count"],
            "original_repos": int(meta["original_repo_count"]),
            "unresolved": int(meta["unresolved_count"]),
        },
        "interval": {
            "start": meta["start_utc"],
            "end": meta["end_utc"],
            "semantics": "[start,end)",
        },
        "script_sha256": meta["script_sha256"],
        "metadata_encoding": meta["metadata_encoding"],
        "run_count": validated["run_count"],
        "expected_attempt_count": validated["expected_attempt_count"],
        "expected_attempt_set_sha256": validated[
            "expected_attempt_set_sha256"
        ],
        "per_repo_ledger": validated["per_repo_ledger"],
        "per_repo_ledger_sha256": validated[
            "per_repo_ledger_sha256"
        ],
        "leaf_window_count": validated["leaf_window_count"],
        "request_count": validated["request_count"],
        "run_set_sha256": validated["run_set_sha256"],
        "window_closure_sha256": validated[
            "window_closure_sha256"
        ],
        "db_logical_sha256": validated["db_logical_sha256"],
        "binding_upgrades": validated["binding_upgrades"],
        "source_count_drift": validated["source_count_drift"],
        "source_drift_reconciliation": reconciliation,
    }
    try:
        _require_exact_json(
            value,
            expected_receipt,
            where="inventory receipt",
        )
    except CompletionError as exc:
        if (
            value.get("source_snapshot_stable") is not database_stable
            or value.get("source_snapshot_mode") != source_snapshot_mode
            or value.get("production_complete") is not database_production
            or value.get("mode") != expected_mode
        ):
            raise CompletionError(
                "inventory receipt production classification differs from "
                "SQLite"
            ) from exc
        raise
    if require_production and not database_production:
        raise CompletionError(
            "inventory receipt is diagnostic/non-production or unstable"
        )
    return value, _sha256_bytes(raw)


class GitHubActionsInventory:
    """Orchestrate bounded window discovery, pagination, and receipts."""

    def __init__(
        self,
        *,
        db_path: str | os.PathLike[str],
        scope: RepoScope,
        start: str | int,
        end: str | int,
        tokens: Sequence[str],
        resume: bool = False,
        allow_script_upgrade_from_sha256: str | None = None,
        script_upgrade_reason: str | None = None,
        progress_path: str | os.PathLike[str] | None = None,
        requester: Callable[
            [str, str, Mapping[str, str], float], HTTPResponse
        ] = _default_requester,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        max_attempts: int = 12,
        progress_interval_seconds: float = 5.0,
        script_path: str | os.PathLike[str] = __file__,
    ):
        self.scope = scope
        self.start_epoch = parse_utc_instant(start) if isinstance(start, str) else start
        self.end_epoch = parse_utc_instant(end) if isinstance(end, str) else end
        if self.start_epoch >= self.end_epoch:
            raise BindingError("inventory interval must satisfy start < end")
        script_bytes = Path(script_path).resolve().read_bytes()
        self.script_sha256 = _sha256_bytes(script_bytes)
        self.db = InventoryDB(db_path)
        self.db.bind(
            scope=scope,
            start_epoch=self.start_epoch,
            end_epoch=self.end_epoch,
            script_sha256=self.script_sha256,
            resume=resume,
            allow_script_upgrade_from_sha256=(
                allow_script_upgrade_from_sha256
            ),
            script_upgrade_reason=script_upgrade_reason,
        )
        self.progress_path = (
            str(Path(progress_path).expanduser().resolve())
            if progress_path is not None
            else None
        )
        if progress_interval_seconds < 0:
            raise ValueError("progress_interval_seconds must be non-negative")
        self.progress_interval_seconds = progress_interval_seconds
        self._progress_clock = time.monotonic
        self._last_progress_monotonic: float | None = None
        pool = TokenPool(tokens, clock=clock, sleeper=sleeper)
        self.client = GitHubClient(
            pool,
            requester=requester,
            sleeper=sleeper,
            max_attempts=max_attempts,
        )
        self._progress_lock = threading.Lock()

    def _write_progress(self, *, force: bool = False) -> None:
        if self.progress_path is None:
            return
        with self._progress_lock:
            now = self._progress_clock()
            if (
                not force
                and self._last_progress_monotonic is not None
                and now - self._last_progress_monotonic
                < self.progress_interval_seconds
            ):
                return
            atomic_write_json(self.progress_path, self.db.progress())
            self._last_progress_monotonic = now

    def _converge_one_second(
        self,
        conn: sqlite3.Connection,
        repo: Repo,
        row: sqlite3.Row,
        ledger: Callable[..., None],
    ) -> None:
        window_id = int(row["id"])
        self.db.prepare_convergence(conn, row)
        expected_total = (
            None if row["expected_total"] is None else int(row["expected_total"])
        )
        for _ in range(CONVERGENCE_MAX_PASSES):
            pages: list[PageResponse] = []
            first = self.client.get_workflow_runs(
                repo=repo,
                start_epoch=int(row["start_epoch"]),
                end_epoch=int(row["end_epoch"]),
                page=1,
                per_page=DEFAULT_PER_PAGE,
                ledger=ledger,
            )
            pages.append(first)
            page_count = max(
                1, math.ceil(first.total_count / DEFAULT_PER_PAGE)
            )
            if first.total_count > GITHUB_FILTER_LIMIT:
                raise UnstableEnumerationError(
                    f"{repo.canonical} one-second convergence returned "
                    f"{first.total_count} runs, above the provable REST limit"
                )
            for page_no in range(2, page_count + 1):
                pages.append(
                    self.client.get_workflow_runs(
                        repo=repo,
                        start_epoch=int(row["start_epoch"]),
                        end_epoch=int(row["end_epoch"]),
                        page=page_no,
                        per_page=DEFAULT_PER_PAGE,
                        ledger=ledger,
                    )
                )
            try:
                total = pages[0].total_count
                if expected_total is not None and total != expected_total:
                    raise PaginationDrift(
                        f"one-second convergence total changed "
                        f"{expected_total} -> {total}",
                        observed_total=total,
                    )
                complete, _digest = self.db.accumulate_convergence_pass(
                    conn, row, pages
                )
            except PaginationDrift as exc:
                raise UnstableEnumerationError(
                    f"window {window_id} convergence pass is malformed: {exc}"
                ) from exc
            if not complete:
                self._write_progress()
                continue
            self._write_progress()
            return
        raise UnstableEnumerationError(
            f"window {window_id} did not accumulate total_count unique runs "
            "with two stable metadata observations each in "
            f"{CONVERGENCE_MAX_PASSES} passes"
        )

    def _process_repo(self, repo: Repo) -> None:
        conn = self.db.connect()
        try:
            while True:
                row = self.db.next_window(conn, repo.key)
                if row is None:
                    return
                window_id = int(row["id"])
                active_page = 1

                def ledger(**fields: Any) -> None:
                    self.db.record_request(
                        conn,
                        repo_key=repo.key,
                        window_id=window_id,
                        **fields,
                    )

                try:
                    if self.db.convergence_state(conn, window_id) is not None:
                        self._converge_one_second(conn, repo, row, ledger)
                        continue
                    expected_total = row["expected_total"]
                    if expected_total is None:
                        active_page = 1
                        first_page = self.client.get_workflow_runs(
                            repo=repo,
                            start_epoch=int(row["start_epoch"]),
                            end_epoch=int(row["end_epoch"]),
                            page=1,
                            per_page=DEFAULT_PER_PAGE,
                            ledger=ledger,
                        )
                        if first_page.total_count > GITHUB_FILTER_LIMIT:
                            self.db.split_window(
                                conn, row, observed_total=first_page.total_count
                            )
                            self._write_progress()
                            continue
                        self.db.store_page(
                            conn, row, page_no=1, page=first_page
                        )
                        self._write_progress()
                        row = conn.execute(
                            "SELECT * FROM search_windows WHERE id=?",
                            (window_id,),
                        ).fetchone()
                        assert row is not None
                    if row["status"] == "done":
                        continue
                    total = int(row["expected_total"])
                    pages = max(1, math.ceil(total / DEFAULT_PER_PAGE))
                    completed_pages = {
                        int(item[0])
                        for item in conn.execute(
                            "SELECT page_no FROM window_pages WHERE window_id=?",
                            (window_id,),
                        )
                    }
                    for page_no in range(1, pages + 1):
                        if page_no in completed_pages:
                            continue
                        active_page = page_no
                        response = self.client.get_workflow_runs(
                            repo=repo,
                            start_epoch=int(row["start_epoch"]),
                            end_epoch=int(row["end_epoch"]),
                            page=page_no,
                            per_page=DEFAULT_PER_PAGE,
                            ledger=ledger,
                        )
                        self.db.store_page(
                            conn, row, page_no=page_no, page=response
                        )
                        self._write_progress()
                except PaginationDrift as exc:
                    action = self.db.recover_pagination_drift(
                        conn,
                        row,
                        observed_total=exc.observed_total,
                        reason=str(exc),
                    )
                    self.db.record_request(
                        conn,
                        repo_key=repo.key,
                        window_id=window_id,
                        endpoint=f"/repos/{repo.owner}/{repo.name}/actions/runs",
                        page=active_page,
                        per_page=DEFAULT_PER_PAGE,
                        attempt=0,
                        http_status=None,
                        outcome=f"pagination_drift_{action}",
                        latency_ms=0,
                        error_class=type(exc).__name__,
                        error_message=self.client.redact(exc),
                    )
                    self._write_progress(force=True)
                    continue
                except BaseException as exc:
                    self.db.record_request(
                        conn,
                        repo_key=repo.key,
                        window_id=window_id,
                        endpoint=f"/repos/{repo.owner}/{repo.name}/actions/runs",
                        page=active_page,
                        per_page=DEFAULT_PER_PAGE,
                        attempt=0,
                        http_status=None,
                        outcome="window_error",
                        latency_ms=0,
                        error_class=type(exc).__name__,
                        error_message=self.client.redact(exc),
                    )
                    self.db.mark_failed(
                        conn,
                        window_id,
                        exc,
                        redacted_message=self.client.redact(exc),
                    )
                    self._write_progress(force=True)
                    raise
        finally:
            conn.close()

    def run(self, *, workers: int = 1) -> dict[str, Any]:
        if workers <= 0:
            raise ValueError("workers must be positive")
        self._write_progress(force=True)
        errors: list[tuple[str, BaseException]] = []
        if workers == 1:
            for repo in self.scope.repos:
                try:
                    self._process_repo(repo)
                except BaseException as exc:
                    errors.append((repo.canonical, exc))
                    break
        else:
            with ThreadPoolExecutor(
                max_workers=min(workers, len(self.scope.repos)),
                thread_name_prefix="ci-inventory",
            ) as executor:
                futures = {
                    executor.submit(self._process_repo, repo): repo
                    for repo in self.scope.repos
                }
                for future in as_completed(futures):
                    repo = futures[future]
                    try:
                        future.result()
                    except BaseException as exc:
                        errors.append((repo.canonical, exc))
        self._write_progress(force=True)
        if errors:
            details = "; ".join(
                f"{repo}: {type(exc).__name__}: {self.client.redact(exc)}"
                for repo, exc in errors[:10]
            )
            raise InventoryError(
                f"inventory failed for {len(errors)} repository/repositories: {details}"
            ) from errors[0][1]
        return self.db.progress()

    def _enumerate_reconciliation_window(
        self,
        repo: Repo,
        start_epoch: int,
        end_epoch: int,
        *,
        evidence_lines: list[str],
        request_counter: list[int],
        page_counter: list[int],
    ) -> dict[tuple[int, int], tuple[str, dict[str, Any]]]:
        def ledger(**fields: Any) -> None:
            request_counter[0] += 1
            evidence_lines.append(
                "R\t"
                + "\t".join(
                    (
                        str(fields.get("endpoint") or ""),
                        str(fields.get("page") or ""),
                        str(fields.get("attempt") or ""),
                        str(fields.get("http_status") or ""),
                        str(fields.get("outcome") or ""),
                    )
                )
            )

        def fetch(page_number: int) -> PageResponse:
            page = self.client.get_workflow_runs(
                repo=repo,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                page=page_number,
                per_page=DEFAULT_PER_PAGE,
                ledger=ledger,
            )
            page_counter[0] += 1
            evidence_lines.append(
                f"P\t{repo.key}\t{start_epoch}\t{end_epoch}\t"
                f"{page_number}\t{page.total_count}\t"
                f"{len(page.workflow_runs)}\t{page.payload_sha256}"
            )
            return page

        first = fetch(1)
        if first.total_count > GITHUB_FILTER_LIMIT:
            unstable = True
            pages = [first]
        else:
            unstable = False
            pages = [first]
            page_count = max(
                1,
                math.ceil(first.total_count / DEFAULT_PER_PAGE),
            )
            pages.extend(
                fetch(page_number)
                for page_number in range(2, page_count + 1)
            )
            unstable = any(
                page.total_count != first.total_count for page in pages
            )
        members: dict[
            tuple[int, int],
            tuple[str, dict[str, Any]],
        ] = {}
        if not unstable:
            for page in pages:
                for run in page.workflow_runs:
                    normalized, metadata_sha256, key = (
                        self.db._normalize_run(
                            repo.key,
                            run,
                            start_epoch=start_epoch,
                            end_epoch=end_epoch,
                        )
                    )
                    previous = members.setdefault(
                        key,
                        (metadata_sha256, normalized),
                    )
                    if previous[0] != metadata_sha256:
                        unstable = True
            if len(members) != first.total_count:
                unstable = True
        if not unstable:
            return members
        if end_epoch - start_epoch <= 1:
            raise CompletionError(
                f"source-drift reconciliation cannot close one-second "
                f"window {repo.key} "
                f"[{format_utc_instant(start_epoch)},"
                f"{format_utc_instant(end_epoch)})"
            )
        midpoint = start_epoch + (end_epoch - start_epoch) // 2
        left = self._enumerate_reconciliation_window(
            repo,
            start_epoch,
            midpoint,
            evidence_lines=evidence_lines,
            request_counter=request_counter,
            page_counter=page_counter,
        )
        right = self._enumerate_reconciliation_window(
            repo,
            midpoint,
            end_epoch,
            evidence_lines=evidence_lines,
            request_counter=request_counter,
            page_counter=page_counter,
        )
        overlap = set(left).intersection(right)
        if overlap:
            raise CompletionError(
                f"source-drift reconciliation found {len(overlap)} "
                f"cross-window run(s) in {repo.key}"
            )
        return left | right

    def _reconciliation_pass(
        self,
        repo: Repo,
        root: Mapping[str, int | str],
        pass_number: int,
    ) -> tuple[
        dict[tuple[int, int], tuple[str, dict[str, Any]]],
        dict[str, Any],
    ]:
        evidence_lines: list[str] = []
        request_counter = [0]
        page_counter = [0]
        members = self._enumerate_reconciliation_window(
            repo,
            int(root["start_epoch"]),
            int(root["end_epoch"]),
            evidence_lines=evidence_lines,
            request_counter=request_counter,
            page_counter=page_counter,
        )
        projected = {
            key: value[0] for key, value in members.items()
        }
        membership_sha256, metadata_sha256 = _run_projection_digests(
            repo.key,
            projected,
        )
        proof = {
            "pass": pass_number,
            "page_observation_count": page_counter[0],
            "request_count": request_counter[0],
            "membership_sha256": membership_sha256,
            "metadata_sha256": metadata_sha256,
            "page_ledger_sha256": _hash_lines(evidence_lines),
        }
        return members, proof

    def _reconcile_source_drift_root(
        self,
        root: Mapping[str, int | str],
        stored: Mapping[tuple[int, int], str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        repo = next(
            (
                candidate
                for candidate in self.scope.repos
                if candidate.key == root["repo"]
            ),
            None,
        )
        if repo is None:
            raise CompletionError(
                f"source-drift root {root['repo']} is outside repository scope"
            )
        first, first_proof = self._reconciliation_pass(repo, root, 1)
        second, second_proof = self._reconciliation_pass(repo, root, 2)
        first_projection = {
            key: value[0] for key, value in first.items()
        }
        current = {
            key: value[0] for key, value in second.items()
        }
        if first_projection != current:
            raise CompletionError(
                f"source-drift reconciliation changed between exact passes "
                f"for {repo.key}"
            )
        new_keys = set(current).difference(stored)
        retained_keys = set(stored).difference(current)
        stored_membership_sha, stored_metadata_sha = (
            _run_projection_digests(repo.key, stored)
        )
        current_membership_sha, current_metadata_sha = (
            _run_projection_digests(repo.key, current)
        )
        observed_union = dict(stored)
        observed_union.update(
            (key, current[key]) for key in new_keys
        )
        observed_union_membership_sha, _observed_union_metadata_sha = (
            _run_projection_digests(repo.key, observed_union)
        )
        completed_at = _utc_now()
        proof = {
            "window_id": int(root["window_id"]),
            "completed_at": completed_at,
            "producer_script_sha256": self.script_sha256,
            "repo": repo.key,
            "interval": {
                "start": format_utc_instant(int(root["start_epoch"])),
                "end": format_utc_instant(int(root["end_epoch"])),
                "semantics": "[start,end)",
            },
            "parent_total": int(root["parent_total"]),
            "child_total": int(root["child_total"]),
            "stored_count": len(stored),
            "stored_membership_sha256": stored_membership_sha,
            "stored_metadata_sha256": stored_metadata_sha,
            "current_count": len(current),
            "current_membership_sha256": current_membership_sha,
            "current_metadata_sha256": current_metadata_sha,
            "retained_upstream_deleted_count": len(retained_keys),
            "new_current_count": len(new_keys),
            "observed_union_count": len(observed_union),
            "observed_union_membership_sha256": (
                observed_union_membership_sha
            ),
            "metadata_changed_count": sum(
                stored[key] != metadata_sha256
                for key, metadata_sha256 in current.items()
                if key in stored
            ),
            "current_members": [
                [run_id, run_attempt, metadata_sha256]
                for (run_id, run_attempt), metadata_sha256 in sorted(
                    current.items()
                )
            ],
            "passes": [first_proof, second_proof],
        }
        return proof, [
            dict(second[key][1]) for key in sorted(new_keys)
        ]

    def reconcile_source_drift(
        self,
        *,
        workers: int = 4,
    ) -> dict[str, Any] | None:
        if workers <= 0:
            raise ValueError("source-drift reconciliation workers must be positive")
        connection = self.db.connect(readonly=True)
        try:
            drift = _source_count_drift_summary(connection)
            if drift["windows"] == 0:
                return None
            roots = _minimal_source_drift_roots(connection)
            persisted_proofs = _load_source_drift_proofs(connection)
            stored_by_window = {
                int(root["window_id"]): _stored_reconciliation_members(
                    connection,
                    root,
                )
                for root in roots
            }
            meta = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key,value FROM inventory_meta"
                )
            }
        finally:
            connection.close()

        if persisted_proofs:
            expected_ids = {
                int(root["window_id"]) for root in roots
            }
            if set(persisted_proofs) != expected_ids:
                raise CompletionError(
                    "persisted source-drift proof roots differ from current "
                    "inventory drift roots"
                )
            persisted_roots = [
                persisted_proofs[int(root["window_id"])]
                for root in roots
            ]
            payload = _source_drift_reconciliation_payload(
                script_sha256=str(
                    persisted_roots[0]["producer_script_sha256"]
                ),
                source_count_drift=drift,
                roots=persisted_roots,
            )
            return self.db._validate_source_drift_reconciliation(
                self.db._validate_and_digests(),
                payload,
            )

        reconciled_by_window: dict[
            int,
            tuple[dict[str, Any], list[dict[str, Any]]],
        ] = {}
        errors: list[tuple[str, BaseException]] = []
        with ThreadPoolExecutor(
            max_workers=min(workers, 4, len(roots)),
            thread_name_prefix="ci-source-drift",
        ) as executor:
            futures = {
                executor.submit(
                    self._reconcile_source_drift_root,
                    root,
                    stored_by_window[int(root["window_id"])],
                ): root
                for root in roots
            }
            for future in as_completed(futures):
                root = futures[future]
                try:
                    reconciled_by_window[int(root["window_id"])] = (
                        future.result()
                    )
                except BaseException as exc:
                    errors.append((str(root["repo"]), exc))
        if errors:
            details = "; ".join(
                f"{repo}: {type(exc).__name__}: {self.client.redact(exc)}"
                for repo, exc in errors[:10]
            )
            raise CompletionError(
                "source-drift reconciliation failed for "
                f"{len(errors)} root(s): {details}"
            ) from errors[0][1]
        reconciled_roots = [
            reconciled_by_window[int(root["window_id"])][0]
            for root in roots
        ]
        new_runs = [
            record
            for root in roots
            for record in reconciled_by_window[int(root["window_id"])][1]
        ]
        self.db.store_source_drift_reconciliation(
            reconciled_roots,
            new_runs,
        )
        payload = _source_drift_reconciliation_payload(
            script_sha256=meta["script_sha256"],
            source_count_drift=drift,
            roots=reconciled_roots,
        )
        return self.db._validate_source_drift_reconciliation(
            self.db._validate_and_digests(),
            payload,
        )

    def write_completion_receipt(
        self,
        path: str | os.PathLike[str],
        *,
        allow_nonproduction: bool = False,
        reconcile_source_drift: bool = False,
        reconciliation_workers: int = 4,
    ) -> dict[str, Any]:
        reconciliation = (
            self.reconcile_source_drift(
                workers=reconciliation_workers,
            )
            if reconcile_source_drift
            else None
        )
        receipt = self.db.completion_receipt(
            allow_nonproduction=allow_nonproduction,
            source_drift_reconciliation=reconciliation,
        )
        atomic_write_json(path, receipt)
        return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory GitHub Actions workflow-run metadata only"
    )
    parser.add_argument(
        "--mode",
        choices=("inventory-only",),
        default="inventory-only",
        help="this stage intentionally supports metadata inventory only",
    )
    parser.add_argument("--repo-list", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--start", help="inclusive UTC boundary")
    parser.add_argument("--end", help="exclusive UTC boundary")
    parser.add_argument("--tokens", help="newline-delimited GitHub token pool")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-inventory-script-upgrade-from-sha256",
        help=(
            "explicitly authorize one audited resume migration from this "
            "exact previously bound producer SHA-256"
        ),
    )
    parser.add_argument(
        "--inventory-script-upgrade-reason",
        help=(
            "required printable audit reason for an explicitly authorized "
            "inventory producer migration"
        ),
    )
    parser.add_argument(
        "--progress",
        help="atomic progress JSON (default: <db>.progress.json)",
    )
    parser.add_argument(
        "--receipt",
        help="atomic completion JSON (default: <db>.completion.json)",
    )
    parser.add_argument(
        "--progress-only",
        action="store_true",
        help="print current SQLite progress without making requests",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--diagnostic-nonproduction-receipt",
        action="store_true",
        help=(
            "explicitly allow a diagnostic receipt with "
            "production_complete=false; never accepted by exhaustive fetch, "
            "merge, or export"
        ),
    )
    parser.add_argument(
        "--reconcile-source-drift",
        action="store_true",
        help=(
            "after exhaustive enumeration, re-enumerate only minimal "
            "split-count drift roots twice, retain the frozen observed "
            "membership, and append any stable live delta under a durable "
            "reconciliation proof"
        ),
    )
    parser.add_argument("--max-repos", type=int)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="minimum seconds between nonterminal atomic progress snapshots",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.progress_only:
        progress = InventoryDB(args.db).progress()
        print(json.dumps(progress, indent=2, sort_keys=True))
        return 0
    if not args.start or not args.end:
        parser.error("--start and --end are required unless --progress-only is used")
    if args.max_repos is not None and not args.smoke:
        parser.error("--max-repos requires explicit --smoke")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.progress_interval < 0:
        parser.error("--progress-interval must be non-negative")

    try:
        scope = load_repo_scope(
            args.repo_list, smoke=args.smoke, max_repos=args.max_repos
        )
        tokens = load_token_pool(args.tokens)
        progress_path = args.progress or f"{args.db}.progress.json"
        receipt_path = args.receipt or f"{args.db}.completion.json"
        inventory = GitHubActionsInventory(
            db_path=args.db,
            scope=scope,
            start=args.start,
            end=args.end,
            tokens=tokens,
            resume=args.resume,
            allow_script_upgrade_from_sha256=(
                args.allow_inventory_script_upgrade_from_sha256
            ),
            script_upgrade_reason=args.inventory_script_upgrade_reason,
            progress_path=progress_path,
            max_attempts=args.max_attempts,
            progress_interval_seconds=args.progress_interval,
        )
        progress = inventory.run(workers=args.workers)
        receipt = inventory.write_completion_receipt(
            receipt_path,
            allow_nonproduction=(
                args.smoke or args.diagnostic_nonproduction_receipt
            ),
            reconcile_source_drift=args.reconcile_source_drift,
            reconciliation_workers=min(args.workers, 4),
        )
    except InventoryError as exc:
        print(f"[ci-stream-inventory] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"progress": progress, "receipt": receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
