#!/usr/bin/env python3
"""Fail-closed repository-source sidecars for final CI occurrences.

This module joins ``cppmega_ci_chunk_occurrence_v3`` build-action source
inputs to blobs in explicit local bare Git mirrors.  The resulting bytes are
always labelled ``repository_blob_content``: a Git blob at the recorded
source repository and commit, never a claim about an ephemeral runner
filesystem after generators or build steps have run.

The source CAS is binary-safe and crash-safe.  Pack bytes are fsynced before
their SQLite locator transaction commits.  Bytes after a durable
``committed_end`` (or an unindexed pack) are quarantined on the next open.
Logical receipts are insertion-order independent; the normal materialization
entry point also sorts bindings before writing packs for deterministic
physical receipts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import struct
import subprocess
import sys
import threading
from typing import Any
from urllib.parse import quote
import zlib


OCCURRENCE_SCHEMA = "cppmega_ci_chunk_occurrence_v3"
TRAINING_SIDECAR_SCHEMA = "cppmega_ci_chunk_training_sidecars_v2"
CONTENT_STORE_SCHEMA = "cppmega_ci_content_store_v1"
CONTENT_STORE_RECEIPT_SCHEMA = "cppmega_ci_content_store_receipt_v1"
FETCH_RECEIPT_SCHEMA = "cppmega_ci_stream_fetch_receipt_v3"
INVENTORY_SCHEMA = "cppmega_ci_source_binding_inventory_v1"
STORE_SCHEMA = "cppmega_ci_source_sidecar_store_v1"
PACK_SCHEMA = "cppmega_ci_source_blob_pack_v1"
RECEIPT_SCHEMA = "cppmega_ci_source_sidecar_receipt_v1"
SIDECAR_SCHEMA = "cppmega_ci_source_binding_sidecar_v1"
REFERENCE_LEDGER_SCHEMA = "cppmega_ci_source_reference_ledger_v1"
RESOLVER_SCHEMA = "cppmega_ci_local_git_blob_resolver_v1"
NORMALIZATION_SCHEMA = "cppmega_ci_source_path_normalization_v1"
RECOVERY_SCHEMA = "cppmega_ci_source_sidecar_recovery_v1"
CONTENT_SEMANTICS = "repository_blob_content"

RESOLVED = "resolved"
PATH_ABSENT = "path_absent"
COMMIT_ABSENT = "commit_absent"
REPO_UNAVAILABLE = "repo_unavailable"
PERMISSION_DENIED = "permission_denied"
DELETED_FORK = "deleted_fork"
UNSUPPORTED_OBJECT = "unsupported_object"
AMBIGUOUS_PATH = "ambiguous_path"
GENERATED_OR_MUTATED_UNRESOLVABLE = "generated_or_mutated_unresolvable"

GAP_STATUSES = frozenset(
    {
        PATH_ABSENT,
        COMMIT_ABSENT,
        REPO_UNAVAILABLE,
        PERMISSION_DENIED,
        DELETED_FORK,
        UNSUPPORTED_OBJECT,
        AMBIGUOUS_PATH,
        GENERATED_OR_MUTATED_UNRESOLVABLE,
    }
)
ALL_STATUSES = GAP_STATUSES | {RESOLVED}

DEFAULT_MAX_PACK_BYTES = 256 * 1024 * 1024
_PACK_MAGIC = b"CISSPK1\n"
_FRAME_MAGIC = b"CISSFRM1"
_FRAME_HEADER = struct.Struct(">8s32sQ")
_PACK_GLOB = "source-pack-*.cissp"
_SQLITE_NAME = "index.sqlite3"
_ORPHAN_DIRECTORY = "orphaned"
_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:/")
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"
_LFS_OID_RE = re.compile(rb"^oid sha256:([0-9a-f]{64})$", re.MULTILINE)
_LFS_SIZE_RE = re.compile(rb"^size ([0-9]+)$", re.MULTILINE)


class SourceSidecarError(RuntimeError):
    """Base exception for source-sidecar failures."""


class ExtractionError(SourceSidecarError):
    """The frozen CI store or its occurrence provenance is invalid."""


class ResolutionIntegrityError(SourceSidecarError):
    """Local Git bytes disagree with their object identity."""


class SourceStoreError(SourceSidecarError):
    """The binary source CAS failed validation or a durable operation."""


class BindingConflictError(SourceStoreError):
    """A replay disagrees with a previously committed binding."""


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
        raise ValueError(f"value is not canonical JSON: {exc}") from exc


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with path.open("rb") as handle:
        while remaining is None or remaining:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            block = handle.read(size)
            if not block:
                if remaining:
                    raise SourceStoreError(f"{path.name} ended before byte {limit}")
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def _hash_records(domain: str, records: Iterable[object]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    for record in records:
        encoded = _canonical_json_bytes(record)
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _require_hex64(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise ExtractionError(f"{where} must be a lowercase SHA-256")
    return value


def _require_git_oid(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _GIT_OID_RE.fullmatch(value) is None:
        raise ExtractionError(f"{where} must be a full lowercase Git object ID")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _script_sha256() -> str:
    return _sha256_file(Path(__file__))


def _sqlite_schema_sha256(connection: sqlite3.Connection) -> str:
    return _hash_records(
        "cppmega-ci-source-sqlite-schema-v1",
        (
            [
                str(row["type"]),
                str(row["name"]),
                str(row["tbl_name"]),
                None if row["sql"] is None else str(row["sql"]),
            ]
            for row in connection.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_schema
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            )
        ),
    )


def atomic_write_json(path: str | os.PathLike[str], value: object) -> None:
    """Durably replace *path* with deterministic pretty JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    for ordinal in range(1_000):
        candidate = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}-{ordinal}"
        )
        if not candidate.exists():
            temporary = candidate
            break
    if temporary is None:
        raise OSError("cannot allocate atomic JSON temporary file")
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


@dataclass(frozen=True)
class PathNormalization:
    """Result of resolving one action path against a checkout root."""

    status: str
    candidates: tuple[str, ...]
    source_input: str
    cwd: str | None
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": NORMALIZATION_SCHEMA,
            "status": self.status,
            "candidates": list(self.candidates),
            "source_input": self.source_input,
            "cwd": self.cwd,
            "reason": self.reason,
        }


def _slash_path(value: str) -> str:
    return value.replace("\\", "/")


def _is_absolute(value: str) -> bool:
    return value.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(value) is not None


def _safe_components(value: str) -> tuple[str, ...] | None:
    """Lexically normalize a root-relative path, rejecting root escape."""

    components: list[str] = []
    for component in value.split("/"):
        if component in {"", "."}:
            continue
        if component == "..":
            if not components:
                return None
            components.pop()
            continue
        if "\0" in component:
            return None
        components.append(component)
    return tuple(components)


def _workspace_suffixes(
    absolute_path: str,
    *,
    repository: str | None,
) -> tuple[str, ...]:
    """Return every defensible path suffix below a hosted-runner checkout."""

    value = _slash_path(absolute_path)
    lowered = value.lower()
    components = value.split("/")
    lower_components = lowered.split("/")
    roots: set[int] = set()

    # GitHub-hosted POSIX: /home/runner/work/name/name and /__w/name/name.
    # GitHub-hosted Windows: D:/a/name/name and self-hosted .../_work/name/name.
    for index in range(len(components) - 2):
        prefix = lower_components[index]
        if prefix in {"work", "_work", "__w", "a"}:
            left = components[index + 1]
            right = components[index + 2]
            if left and left.casefold() == right.casefold():
                roots.add(index + 3)

    basename = None
    if repository and "/" in repository:
        basename = repository.rsplit("/", 1)[1]
    if basename:
        for index in range(len(components) - 1):
            if (
                components[index].casefold() == basename.casefold()
                and components[index + 1].casefold() == basename.casefold()
            ):
                roots.add(index + 2)

    suffixes: set[str] = set()
    for root_end in roots:
        relative = "/".join(components[root_end:])
        normalized = _safe_components(relative)
        if normalized:
            suffixes.add("/".join(normalized))
        elif relative in {"", "."}:
            suffixes.add("")
    return tuple(sorted(suffixes))


def _relative_cwd_candidates(
    cwd: str | None,
    *,
    repository: str | None,
) -> tuple[str, ...] | None:
    if cwd is None or not cwd.strip() or cwd.strip() == ".":
        return ("",)
    value = _slash_path(cwd.strip())
    if _is_absolute(value):
        suffixes = _workspace_suffixes(value, repository=repository)
        return suffixes or None
    normalized = _safe_components(value)
    if normalized is None:
        return None
    return ("/".join(normalized),)


def normalize_source_candidates(
    source_input: str,
    cwd: str | None,
    *,
    repository: str | None = None,
) -> PathNormalization:
    """Normalize a raw build-action source path without permitting root escape.

    Hosted-runner absolute paths are accepted only when a checkout root can be
    identified.  Relative paths are joined to the action ``cwd`` lexically;
    no filesystem lookups or symlink dereferences occur.
    """

    if not isinstance(source_input, str) or not source_input.strip():
        return PathNormalization(
            GENERATED_OR_MUTATED_UNRESOLVABLE,
            (),
            source_input if isinstance(source_input, str) else repr(source_input),
            cwd,
            "empty_or_non_string_source_input",
        )
    raw = source_input.strip()
    if "\0" in raw or raw.startswith(("-", "@")):
        return PathNormalization(
            GENERATED_OR_MUTATED_UNRESOLVABLE,
            (),
            source_input,
            cwd,
            "not_a_literal_repository_path",
        )
    source = _slash_path(raw)
    candidates: set[str] = set()
    if _is_absolute(source):
        candidates.update(_workspace_suffixes(source, repository=repository))
        if not candidates:
            return PathNormalization(
                GENERATED_OR_MUTATED_UNRESOLVABLE,
                (),
                source_input,
                cwd,
                "absolute_path_outside_recognized_workspace",
            )
    else:
        cwd_candidates = _relative_cwd_candidates(cwd, repository=repository)
        if cwd_candidates is None:
            return PathNormalization(
                GENERATED_OR_MUTATED_UNRESOLVABLE,
                (),
                source_input,
                cwd,
                "cwd_outside_or_escaping_workspace",
            )
        for relative_cwd in cwd_candidates:
            combined = f"{relative_cwd}/{source}" if relative_cwd else source
            normalized = _safe_components(combined)
            if normalized:
                candidates.add("/".join(normalized))

    candidates.discard("")
    ordered = tuple(sorted(candidates))
    if not ordered:
        return PathNormalization(
            GENERATED_OR_MUTATED_UNRESOLVABLE,
            (),
            source_input,
            cwd,
            "path_escapes_workspace_or_names_checkout_root",
        )
    if len(ordered) > 1:
        return PathNormalization(
            AMBIGUOUS_PATH,
            ordered,
            source_input,
            cwd,
            "multiple_checkout_roots_produce_different_paths",
        )
    return PathNormalization(RESOLVED, ordered, source_input, cwd, None)


def normalize_source_path(
    source_input: str,
    cwd: str | None,
    *,
    repository: str | None = None,
) -> str:
    """Return one normalized path or raise when the join is not exact."""

    result = normalize_source_candidates(
        source_input,
        cwd,
        repository=repository,
    )
    if result.status != RESOLVED:
        raise ValueError(f"source path is not uniquely resolvable: {result.status}")
    return result.candidates[0]


def _decode_provenance(row: sqlite3.Row) -> dict[str, Any]:
    try:
        raw = zlib.decompress(bytes(row["provenance_zlib"]))
    except zlib.error as exc:
        raise ExtractionError("occurrence provenance zlib is invalid") from exc
    if len(raw) != int(row["provenance_raw_size"]):
        raise ExtractionError("occurrence provenance raw size mismatch")
    if _sha256_bytes(raw) != str(row["provenance_sha256"]):
        raise ExtractionError("occurrence provenance SHA-256 mismatch")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExtractionError("occurrence provenance JSON is invalid") from exc
    if not isinstance(value, dict) or _canonical_json_bytes(value) != raw:
        raise ExtractionError("occurrence provenance is not canonical JSON")
    return value


def _occurrence_set_digest(connection: sqlite3.Connection) -> str:
    def records() -> Iterator[object]:
        for row in connection.execute(
            """
            SELECT repo, run_attempt, job, step, chunk_ordinal,
                   content_sha256, provenance_sha256,
                   provenance_raw_size, provenance_zlib
            FROM occurrences
            ORDER BY repo, run_attempt, job, step, chunk_ordinal
            """
        ):
            yield {
                "repo": str(row["repo"]),
                "run_attempt": str(row["run_attempt"]),
                "job": str(row["job"]),
                "step": str(row["step"]),
                "chunk_ordinal": int(row["chunk_ordinal"]),
                "content_sha256": str(row["content_sha256"]),
                "provenance_sha256": str(row["provenance_sha256"]),
                "provenance": _decode_provenance(row),
            }

    return _hash_records("cppmega-ci-occurrence-set-v1", records())


def _file_snapshot(paths: Sequence[Path]) -> tuple[tuple[str, int, int, int], ...]:
    return tuple(
        (
            str(path),
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.stat().st_ino,
        )
        for path in paths
    )


def _read_json_object(path: Path, *, where: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ExtractionError(f"{where} must be a JSON object")
    return value, raw


def _verify_frozen_content_store(
    root: Path,
    receipt: Mapping[str, Any],
) -> tuple[sqlite3.Connection, str, tuple[tuple[str, int, int, int], ...]]:
    if receipt.get("schema") != CONTENT_STORE_RECEIPT_SCHEMA:
        raise ExtractionError("content-store receipt schema is missing or stale")
    if receipt.get("status") != "complete":
        raise ExtractionError("content-store receipt is not complete")
    verification = receipt.get("verification")
    if not isinstance(verification, Mapping) or verification.get("ok") is not True:
        raise ExtractionError("content-store receipt lacks full verification")
    expected_occurrence_set = _require_hex64(
        receipt.get("occurrence_set_sha256"),
        where="content-store receipt occurrence_set_sha256",
    )
    db_path = root / _SQLITE_NAME
    if not db_path.is_file() or db_path.is_symlink():
        raise ExtractionError("frozen content-store SQLite file is missing")
    for suffix in ("-wal", "-journal"):
        if Path(f"{db_path}{suffix}").exists():
            raise ExtractionError(
                f"frozen content store has a mutable SQLite {suffix} file"
            )

    pack_receipts = receipt.get("pack_hashes")
    if not isinstance(pack_receipts, list):
        raise ExtractionError("content-store receipt pack_hashes is invalid")
    pack_paths: list[Path] = []
    receipt_names: set[str] = set()
    for index, record in enumerate(pack_receipts):
        if not isinstance(record, Mapping):
            raise ExtractionError(f"pack_hashes[{index}] is not an object")
        filename = record.get("filename")
        committed_end = record.get("committed_end")
        digest = record.get("sha256")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or isinstance(committed_end, bool)
            or not isinstance(committed_end, int)
            or committed_end < 0
        ):
            raise ExtractionError(f"pack_hashes[{index}] has invalid metadata")
        _require_hex64(digest, where=f"pack_hashes[{index}].sha256")
        path = root / filename
        if (
            filename in receipt_names
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != committed_end
            or _sha256_file(path, limit=committed_end) != digest
        ):
            raise ExtractionError(f"content-store pack verification failed: {filename}")
        receipt_names.add(filename)
        pack_paths.append(path)
    actual_names = {path.name for path in root.glob("pack-*.cicp")}
    if actual_names != receipt_names:
        raise ExtractionError("content-store pack set differs from its receipt")

    snapshot_paths = [db_path, *sorted(pack_paths)]
    before = _file_snapshot(snapshot_paths)
    uri = f"file:{quote(str(db_path.resolve()), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        integrity = [
            str(row[0])
            for row in connection.execute("PRAGMA integrity_check").fetchall()
        ]
        if integrity != ["ok"]:
            raise ExtractionError(f"content-store integrity_check failed: {integrity}")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ExtractionError("content-store foreign_key_check failed")
        settings = dict(
            connection.execute("SELECT key, value FROM settings").fetchall()
        )
        if settings.get("schema") != CONTENT_STORE_SCHEMA:
            raise ExtractionError("content-store SQLite schema is missing or stale")
        actual_occurrence_set = _occurrence_set_digest(connection)
        if actual_occurrence_set != expected_occurrence_set:
            raise ExtractionError(
                "content-store occurrence set differs from verified receipt"
            )
    except BaseException:
        connection.close()
        raise
    return connection, actual_occurrence_set, before


def _unresolved_source_path(
    repository: str,
    head_sha: str,
    evidence: Mapping[str, Any],
) -> str:
    digest = _hash_records(
        "cppmega-ci-unresolved-source-path-v1",
        ([repository, head_sha, evidence],),
    )
    return f"!unresolved/{digest}"


def _inventory_binding_hash_records(
    bindings: Sequence[Mapping[str, Any]],
) -> Iterator[object]:
    for binding in bindings:
        yield {
            "repository": binding["repository"],
            "head_sha": binding["head_sha"],
            "source_path": binding["source_path"],
            "normalization_status": binding["normalization_status"],
            "normalized_candidates": binding["normalized_candidates"],
            "evidence_sha256": binding["evidence_sha256"],
            "evidence": binding["evidence"],
        }


def verify_binding_inventory(inventory: Mapping[str, Any]) -> None:
    """Validate the canonical unique binding inventory and its digest."""

    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ExtractionError("source binding inventory schema is missing or stale")
    occurrence_set = _require_hex64(
        inventory.get("occurrence_set_sha256"),
        where="inventory occurrence_set_sha256",
    )
    del occurrence_set
    _require_hex64(
        inventory.get("upstream_fetch_receipt_sha256"),
        where="inventory upstream_fetch_receipt_sha256",
    )
    bindings = inventory.get("bindings")
    if not isinstance(bindings, list):
        raise ExtractionError("inventory bindings must be a list")
    if inventory.get("binding_count") != len(bindings):
        raise ExtractionError("inventory binding_count is inconsistent")
    keys: list[tuple[str, str, str]] = []
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            raise ExtractionError(f"inventory bindings[{index}] is invalid")
        repository = binding.get("repository")
        source_path = binding.get("source_path")
        if (
            not isinstance(repository, str)
            or not repository
            or not isinstance(source_path, str)
            or not source_path
        ):
            raise ExtractionError(f"inventory bindings[{index}] has invalid key")
        head_sha = _require_git_oid(
            binding.get("head_sha"),
            where=f"inventory bindings[{index}].head_sha",
        )
        status = binding.get("normalization_status")
        candidates = binding.get("normalized_candidates")
        evidence = binding.get("evidence")
        if (
            status not in {RESOLVED, AMBIGUOUS_PATH, GENERATED_OR_MUTATED_UNRESOLVABLE}
            or not isinstance(candidates, list)
            or any(not isinstance(item, str) or not item for item in candidates)
            or not isinstance(evidence, list)
            or not evidence
        ):
            raise ExtractionError(
                f"inventory bindings[{index}] has invalid normalization evidence"
            )
        evidence_sha = _hash_records(
            "cppmega-ci-source-binding-evidence-v1",
            evidence,
        )
        if evidence_sha != binding.get("evidence_sha256"):
            raise ExtractionError(
                f"inventory bindings[{index}] evidence digest differs"
            )
        if status == RESOLVED and candidates != [source_path]:
            raise ExtractionError(
                f"inventory bindings[{index}] resolved path is inconsistent"
            )
        if status != RESOLVED and not source_path.startswith("!unresolved/"):
            raise ExtractionError(
                f"inventory bindings[{index}] gap lacks unresolved identity"
            )
        keys.append((repository, head_sha, source_path))
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ExtractionError("inventory bindings are not sorted and unique")
    actual_hash = _hash_records(
        "cppmega-ci-source-binding-inventory-v1",
        _inventory_binding_hash_records(bindings),
    )
    if actual_hash != inventory.get("binding_inventory_sha256"):
        raise ExtractionError("binding inventory SHA-256 differs")


def extract_binding_inventory(
    content_store_root: str | os.PathLike[str],
    upstream_fetch_receipt_path: str | os.PathLike[str],
    *,
    content_store_receipt_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Extract deterministic unique source bindings from a frozen CI store."""

    root = Path(content_store_root)
    fetch_path = Path(upstream_fetch_receipt_path)
    fetch_receipt, fetch_raw = _read_json_object(
        fetch_path,
        where="upstream fetch receipt",
    )
    if fetch_receipt.get("schema") != FETCH_RECEIPT_SCHEMA:
        raise ExtractionError("upstream fetch receipt schema is missing or stale")
    nested_receipt = fetch_receipt.get("content_store_receipt")
    if not isinstance(nested_receipt, Mapping):
        raise ExtractionError("fetch receipt lacks its content-store receipt")
    if content_store_receipt_path is not None:
        separate_receipt, _raw = _read_json_object(
            Path(content_store_receipt_path),
            where="content-store receipt",
        )
        if separate_receipt != nested_receipt:
            raise ExtractionError(
                "separate content-store receipt differs from fetch receipt"
            )
        content_receipt: Mapping[str, Any] = separate_receipt
    else:
        content_receipt = nested_receipt

    connection, occurrence_set_sha256, before = _verify_frozen_content_store(
        root,
        content_receipt,
    )
    upstream_fetch_receipt_sha256 = _sha256_bytes(fetch_raw)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        for row in connection.execute(
            """
            SELECT repo, run_attempt, job, step, chunk_ordinal,
                   content_sha256, provenance_sha256,
                   provenance_raw_size, provenance_zlib
            FROM occurrences
            ORDER BY repo, run_attempt, job, step, chunk_ordinal
            """
        ):
            provenance = _decode_provenance(row)
            if provenance.get("schema") != OCCURRENCE_SCHEMA:
                raise ExtractionError(
                    "every source-scanned occurrence must use final schema v3"
                )
            run_evidence = provenance.get("run_metadata_evidence")
            if (
                not isinstance(run_evidence, Mapping)
                or run_evidence.get("exact_attempt_match") is not True
            ):
                raise ExtractionError(
                    "v3 occurrence lacks exact-attempt run metadata evidence"
                )
            repository = provenance.get("source_repository")
            if not isinstance(repository, str) or "/" not in repository:
                raise ExtractionError(
                    "v3 occurrence lacks exact source_repository provenance"
                )
            workflow = provenance.get("workflow")
            if not isinstance(workflow, Mapping):
                raise ExtractionError("v3 occurrence workflow provenance is invalid")
            head_sha = _require_git_oid(
                workflow.get("head_sha"),
                where="v3 workflow.head_sha",
            )
            chunk = provenance.get("chunk")
            if not isinstance(chunk, Mapping):
                raise ExtractionError("v3 occurrence chunk is invalid")
            training = chunk.get("training_sidecars")
            if not isinstance(training, Mapping):
                raise ExtractionError("v3 occurrence lacks training sidecars")
            if training.get("schema") != TRAINING_SIDECAR_SCHEMA:
                raise ExtractionError("chunk training sidecar schema is stale")
            actions = training.get("build_actions")
            if not isinstance(actions, list):
                raise ExtractionError("training build_actions is not a list")

            occurrence_key = {
                "repo": str(row["repo"]),
                "run_attempt": str(row["run_attempt"]),
                "job": str(row["job"]),
                "step": str(row["step"]),
                "chunk_ordinal": int(row["chunk_ordinal"]),
            }
            for action_index, action in enumerate(actions):
                if not isinstance(action, Mapping):
                    raise ExtractionError("training build action is not an object")
                source_inputs = action.get("source_inputs")
                if not isinstance(source_inputs, list) or any(
                    not isinstance(item, str) for item in source_inputs
                ):
                    raise ExtractionError(
                        "training build action source_inputs is invalid"
                    )
                cwd_raw = action.get("cwd")
                if cwd_raw is not None and not isinstance(cwd_raw, str):
                    raise ExtractionError("training build action cwd is invalid")
                heuristic_bindings = action.get("repository_source_bindings", [])
                if not isinstance(heuristic_bindings, list):
                    raise ExtractionError(
                        "heuristic repository_source_bindings is invalid"
                    )
                for source_index, source_input in enumerate(source_inputs):
                    normalization = normalize_source_candidates(
                        source_input,
                        cwd_raw,
                        repository=repository,
                    )
                    evidence = {
                        "occurrence_key": occurrence_key,
                        "occurrence_content_sha256": str(row["content_sha256"]),
                        "occurrence_provenance_sha256": str(row["provenance_sha256"]),
                        "action_index": action_index,
                        "action_entity_id": action.get("action_entity_id"),
                        "action_shape_sha256": action.get("action_shape_sha256"),
                        "command_sha256": action.get("command_sha256"),
                        "source_input_index": source_index,
                        "source_input": source_input,
                        "cwd": cwd_raw,
                        "normalization": normalization.as_dict(),
                        "discarded_heuristic_bindings_sha256": _hash_records(
                            "cppmega-ci-discarded-heuristic-bindings-v1",
                            heuristic_bindings,
                        ),
                    }
                    if normalization.status == RESOLVED:
                        source_path = normalization.candidates[0]
                    else:
                        source_path = _unresolved_source_path(
                            repository,
                            head_sha,
                            evidence,
                        )
                    key = (repository, head_sha, source_path)
                    record = grouped.setdefault(
                        key,
                        {
                            "repository": repository,
                            "head_sha": head_sha,
                            "source_path": source_path,
                            "normalization_status": normalization.status,
                            "normalized_candidates": list(normalization.candidates),
                            "evidence": [],
                        },
                    )
                    if record["normalization_status"] != normalization.status or record[
                        "normalized_candidates"
                    ] != list(normalization.candidates):
                        raise ExtractionError(
                            "one source binding has inconsistent normalization"
                        )
                    record["evidence"].append(evidence)
    finally:
        connection.close()

    db_path = root / _SQLITE_NAME
    after_paths = [Path(item[0]) for item in before]
    if _file_snapshot(after_paths) != before:
        raise ExtractionError("content store changed during immutable extraction")
    if not db_path.is_file():
        raise ExtractionError("content store disappeared after extraction")

    bindings: list[dict[str, Any]] = []
    for key in sorted(grouped):
        record = grouped[key]
        unique_evidence = {_canonical_json(item): item for item in record["evidence"]}
        evidence = [unique_evidence[encoded] for encoded in sorted(unique_evidence)]
        record["evidence"] = evidence
        record["evidence_sha256"] = _hash_records(
            "cppmega-ci-source-binding-evidence-v1",
            evidence,
        )
        bindings.append(record)
    inventory_sha256 = _hash_records(
        "cppmega-ci-source-binding-inventory-v1",
        _inventory_binding_hash_records(bindings),
    )
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "occurrence_schema": OCCURRENCE_SCHEMA,
        "training_sidecar_schema": TRAINING_SIDECAR_SCHEMA,
        "normalization_schema": NORMALIZATION_SCHEMA,
        "content_semantics": CONTENT_SEMANTICS,
        "occurrence_set_sha256": occurrence_set_sha256,
        "upstream_fetch_receipt_sha256": upstream_fetch_receipt_sha256,
        "binding_count": len(bindings),
        "binding_inventory_sha256": inventory_sha256,
        "bindings": bindings,
    }
    verify_binding_inventory(inventory)
    return inventory


@dataclass(frozen=True)
class MirrorSpec:
    path: Path | None
    unavailable_status: str | None = None


def _normalize_mirror_mapping(
    mapping: Mapping[str, str | os.PathLike[str] | Mapping[str, object] | None],
) -> dict[str, MirrorSpec]:
    output: dict[str, MirrorSpec] = {}
    casefolded: set[str] = set()
    for repository, raw in mapping.items():
        if (
            not isinstance(repository, str)
            or "/" not in repository
            or repository.casefold() in casefolded
        ):
            raise ValueError("mirror repository keys must be unique owner/name strings")
        casefolded.add(repository.casefold())
        if raw is None:
            spec = MirrorSpec(None, REPO_UNAVAILABLE)
        elif isinstance(raw, (str, os.PathLike)):
            spec = MirrorSpec(Path(raw), None)
        elif isinstance(raw, Mapping):
            status = raw.get("status")
            path = raw.get("path")
            if status in {DELETED_FORK, REPO_UNAVAILABLE, PERMISSION_DENIED}:
                if path is not None:
                    raise ValueError("unavailable mirror mapping cannot also have path")
                spec = MirrorSpec(None, str(status))
            elif status is None and isinstance(path, str) and path:
                spec = MirrorSpec(Path(path), None)
            else:
                raise ValueError(f"invalid mirror mapping for {repository}")
        else:
            raise ValueError(f"invalid mirror mapping for {repository}")
        output[repository.casefold()] = spec
    return output


@dataclass(frozen=True)
class GitResolution:
    repository: str
    head_sha: str
    source_path: str
    status: str
    object_format: str | None
    commit_oid: str | None
    root_tree_oid: str | None
    parent_tree_oid: str | None
    object_oid: str | None
    blob_oid: str | None
    mode: str | None
    object_type: str | None
    content_kind: str | None
    content_sha256: str | None
    content_size: int | None
    lfs_oid_sha256: str | None
    lfs_size: int | None
    traversal: tuple[Mapping[str, object], ...]
    evidence: Mapping[str, object]
    content: bytes | None = None

    def durable_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "head_sha": self.head_sha,
            "source_path": self.source_path,
            "status": self.status,
            "content_semantics": CONTENT_SEMANTICS,
            "object_format": self.object_format,
            "commit_oid": self.commit_oid,
            "root_tree_oid": self.root_tree_oid,
            "parent_tree_oid": self.parent_tree_oid,
            "object_oid": self.object_oid,
            "blob_oid": self.blob_oid,
            "mode": self.mode,
            "object_type": self.object_type,
            "content_kind": self.content_kind,
            "content_sha256": self.content_sha256,
            "content_size": self.content_size,
            "lfs_oid_sha256": self.lfs_oid_sha256,
            "lfs_size": self.lfs_size,
            "traversal": list(self.traversal),
            "evidence": dict(self.evidence),
        }


class _GitObjectAbsent(Exception):
    pass


class _GitPermissionDenied(Exception):
    pass


class LocalGitResolver:
    """Resolve exact commit paths using only explicitly mapped local mirrors."""

    def __init__(
        self,
        mirror_mapping: Mapping[
            str, str | os.PathLike[str] | Mapping[str, object] | None
        ],
    ) -> None:
        self._mapping = _normalize_mirror_mapping(mirror_mapping)

    @staticmethod
    def _git_oid(object_format: str, object_type: str, payload: bytes) -> str:
        if object_format not in {"sha1", "sha256"}:
            raise ResolutionIntegrityError(
                f"unsupported Git object format {object_format!r}"
            )
        header = f"{object_type} {len(payload)}\0".encode("ascii")
        constructor = hashlib.sha1 if object_format == "sha1" else hashlib.sha256
        return constructor(header + payload).hexdigest()

    def _run_git(
        self,
        mirror: Path,
        args: Sequence[str],
        *,
        absent_ok: bool = False,
    ) -> bytes:
        environment = dict(os.environ)
        environment.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "protocol.allow",
                "GIT_CONFIG_VALUE_0": "never",
                "GIT_NO_LAZY_FETCH": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "LC_ALL": "C",
            }
        )
        try:
            result = subprocess.run(
                ["git", "--git-dir", str(mirror), *args],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                timeout=60,
            )
        except PermissionError as exc:
            raise _GitPermissionDenied from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ResolutionIntegrityError(
                f"local Git command failed: {type(exc).__name__}"
            ) from exc
        if result.returncode == 0:
            return result.stdout
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        if "permission denied" in stderr.casefold():
            raise _GitPermissionDenied
        absent_markers = (
            "not a valid object name",
            "bad object",
            "invalid object name",
            "could not get object info",
            "does not exist",
        )
        if absent_ok and any(marker in stderr.casefold() for marker in absent_markers):
            raise _GitObjectAbsent
        raise ResolutionIntegrityError(
            f"local Git command failed ({' '.join(args)}): {stderr}"
        )

    def _read_object(
        self,
        mirror: Path,
        oid: str,
        *,
        object_format: str,
        expected_type: str,
        absent_ok: bool = False,
    ) -> bytes:
        try:
            actual_type = (
                self._run_git(
                    mirror,
                    ["cat-file", "-t", oid],
                    absent_ok=absent_ok,
                )
                .decode("ascii", errors="strict")
                .strip()
            )
        except UnicodeError as exc:
            raise ResolutionIntegrityError("Git object type is not ASCII") from exc
        if actual_type != expected_type:
            if absent_ok:
                raise _GitObjectAbsent
            raise ResolutionIntegrityError(
                f"Git object {oid} is {actual_type}, expected {expected_type}"
            )
        payload = self._run_git(
            mirror,
            ["cat-file", expected_type, oid],
            absent_ok=absent_ok,
        )
        actual_oid = self._git_oid(object_format, expected_type, payload)
        if actual_oid != oid:
            raise ResolutionIntegrityError(
                f"Git {expected_type} bytes hash to {actual_oid}, expected {oid}"
            )
        return payload

    @staticmethod
    def _parse_tree(
        payload: bytes,
        *,
        oid_bytes: int,
    ) -> list[tuple[str, bytes, str]]:
        entries: list[tuple[str, bytes, str]] = []
        cursor = 0
        while cursor < len(payload):
            space = payload.find(b" ", cursor)
            nul = payload.find(b"\0", space + 1)
            if space <= cursor or nul < 0 or nul + 1 + oid_bytes > len(payload):
                raise ResolutionIntegrityError("Git tree object encoding is invalid")
            try:
                mode = payload[cursor:space].decode("ascii", errors="strict")
            except UnicodeError as exc:
                raise ResolutionIntegrityError("Git tree mode is not ASCII") from exc
            name = payload[space + 1 : nul]
            raw_oid = payload[nul + 1 : nul + 1 + oid_bytes]
            entries.append((mode, name, raw_oid.hex()))
            cursor = nul + 1 + oid_bytes
        return entries

    @staticmethod
    def _mode_type(mode: str) -> str:
        if mode in {"100644", "100755"}:
            return "blob"
        if mode == "120000":
            return "symlink"
        if mode == "160000":
            return "submodule"
        if mode in {"40000", "040000"}:
            return "tree"
        return "unsupported"

    @staticmethod
    def _classify_blob(
        content: bytes,
        *,
        mode_type: str,
    ) -> tuple[str, str | None, int | None]:
        if mode_type == "symlink":
            return "symlink", None, None
        if content.startswith(_LFS_HEADER):
            oid_match = _LFS_OID_RE.search(content)
            size_match = _LFS_SIZE_RE.search(content)
            if oid_match and size_match:
                return (
                    "lfs_pointer",
                    oid_match.group(1).decode("ascii"),
                    int(size_match.group(1)),
                )
        if b"\0" in content:
            return "binary", None, None
        try:
            content.decode("utf-8", errors="strict")
        except UnicodeError:
            return "binary", None, None
        return "text", None, None

    @staticmethod
    def _join_evidence(
        binding: Mapping[str, Any],
        resolver_evidence: Mapping[str, object],
    ) -> dict[str, object]:
        inventory_evidence = binding.get("evidence", [])
        if not isinstance(inventory_evidence, list):
            raise ValueError("binding inventory evidence must be a list")
        evidence_sha256 = _hash_records(
            "cppmega-ci-source-binding-evidence-v1",
            inventory_evidence,
        )
        expected_sha256 = binding.get("evidence_sha256")
        if expected_sha256 is not None and expected_sha256 != evidence_sha256:
            raise ValueError("binding inventory evidence SHA-256 differs")
        inventory_record = {
            "repository": binding.get("repository"),
            "head_sha": binding.get("head_sha"),
            "source_path": binding.get("source_path"),
            "normalization_status": binding.get(
                "normalization_status",
                RESOLVED,
            ),
            "normalized_candidates": binding.get(
                "normalized_candidates",
                [binding.get("source_path")],
            ),
            "evidence_sha256": evidence_sha256,
            "evidence": inventory_evidence,
        }
        return {
            **resolver_evidence,
            "binding_inventory_evidence_count": len(inventory_evidence),
            "binding_inventory_evidence_sha256": evidence_sha256,
            "binding_inventory_evidence": inventory_evidence,
            "binding_inventory_record": inventory_record,
            "runner_filesystem_equivalence_claimed": False,
        }

    @staticmethod
    def _gap(
        binding: Mapping[str, Any],
        status: str,
        *,
        evidence: Mapping[str, object],
        object_format: str | None = None,
        commit_oid: str | None = None,
        root_tree_oid: str | None = None,
        parent_tree_oid: str | None = None,
        object_oid: str | None = None,
        mode: str | None = None,
        object_type: str | None = None,
        traversal: Sequence[Mapping[str, object]] = (),
    ) -> GitResolution:
        return GitResolution(
            repository=str(binding["repository"]),
            head_sha=str(binding["head_sha"]),
            source_path=str(binding["source_path"]),
            status=status,
            object_format=object_format,
            commit_oid=commit_oid,
            root_tree_oid=root_tree_oid,
            parent_tree_oid=parent_tree_oid,
            object_oid=object_oid,
            blob_oid=None,
            mode=mode,
            object_type=object_type,
            content_kind=None,
            content_sha256=None,
            content_size=None,
            lfs_oid_sha256=None,
            lfs_size=None,
            traversal=tuple(traversal),
            evidence=LocalGitResolver._join_evidence(binding, evidence),
            content=None,
        )

    def resolve(self, binding: Mapping[str, Any]) -> GitResolution:
        repository = str(binding["repository"])
        head_sha = str(binding["head_sha"])
        source_path = str(binding["source_path"])
        normalization_status = binding.get("normalization_status", RESOLVED)
        if normalization_status != RESOLVED:
            status = (
                str(normalization_status)
                if normalization_status
                in {
                    AMBIGUOUS_PATH,
                    GENERATED_OR_MUTATED_UNRESOLVABLE,
                }
                else GENERATED_OR_MUTATED_UNRESOLVABLE
            )
            return self._gap(
                binding,
                status,
                evidence={
                    "resolver_schema": RESOLVER_SCHEMA,
                    "reason": "path_normalization_gap",
                    "normalized_candidates": binding.get("normalized_candidates", []),
                },
            )
        components = source_path.split("/")
        if (
            not source_path
            or source_path.startswith("/")
            or any(component in {"", ".", ".."} for component in components)
        ):
            return self._gap(
                binding,
                GENERATED_OR_MUTATED_UNRESOLVABLE,
                evidence={
                    "resolver_schema": RESOLVER_SCHEMA,
                    "reason": "unsafe_normalized_source_path",
                },
            )

        spec = self._mapping.get(repository.casefold())
        if spec is None:
            return self._gap(
                binding,
                REPO_UNAVAILABLE,
                evidence={
                    "resolver_schema": RESOLVER_SCHEMA,
                    "reason": "repository_not_explicitly_mapped",
                },
            )
        if spec.unavailable_status is not None:
            return self._gap(
                binding,
                spec.unavailable_status,
                evidence={
                    "resolver_schema": RESOLVER_SCHEMA,
                    "reason": "explicit_unavailable_repository_mapping",
                },
            )
        assert spec.path is not None
        mirror = spec.path
        try:
            try:
                mirror_stat = mirror.lstat()
            except FileNotFoundError:
                return self._gap(
                    binding,
                    REPO_UNAVAILABLE,
                    evidence={
                        "resolver_schema": RESOLVER_SCHEMA,
                        "reason": "mapped_mirror_missing_or_unsafe",
                    },
                )
            if stat.S_ISLNK(mirror_stat.st_mode) or not stat.S_ISDIR(
                mirror_stat.st_mode
            ):
                return self._gap(
                    binding,
                    REPO_UNAVAILABLE,
                    evidence={
                        "resolver_schema": RESOLVER_SCHEMA,
                        "reason": "mapped_mirror_missing_or_unsafe",
                    },
                )
            if not os.access(mirror, os.R_OK | os.X_OK):
                return self._gap(
                    binding,
                    PERMISSION_DENIED,
                    evidence={
                        "resolver_schema": RESOLVER_SCHEMA,
                        "reason": "mapped_mirror_is_not_readable",
                    },
                )
            is_bare = (
                self._run_git(
                    mirror,
                    ["rev-parse", "--is-bare-repository"],
                )
                .decode("ascii", errors="strict")
                .strip()
            )
            if is_bare != "true":
                return self._gap(
                    binding,
                    REPO_UNAVAILABLE,
                    evidence={
                        "resolver_schema": RESOLVER_SCHEMA,
                        "reason": "mapped_repository_is_not_bare",
                    },
                )
            object_format = (
                self._run_git(
                    mirror,
                    ["rev-parse", "--show-object-format"],
                )
                .decode("ascii", errors="strict")
                .strip()
            )
            expected_oid_length = 40 if object_format == "sha1" else 64
            if (
                object_format not in {"sha1", "sha256"}
                or len(head_sha) != expected_oid_length
            ):
                raise ResolutionIntegrityError(
                    "head SHA length disagrees with mirror object format"
                )
            try:
                commit = self._read_object(
                    mirror,
                    head_sha,
                    object_format=object_format,
                    expected_type="commit",
                    absent_ok=True,
                )
            except _GitObjectAbsent:
                return self._gap(
                    binding,
                    COMMIT_ABSENT,
                    object_format=object_format,
                    evidence={
                        "resolver_schema": RESOLVER_SCHEMA,
                        "reason": "exact_commit_not_in_local_mirror",
                    },
                )
            first_line = commit.partition(b"\n")[0]
            if not first_line.startswith(b"tree "):
                raise ResolutionIntegrityError("commit lacks a root tree header")
            try:
                root_tree_oid = first_line[5:].decode("ascii", errors="strict")
            except UnicodeError as exc:
                raise ResolutionIntegrityError("commit tree OID is not ASCII") from exc
            if (
                len(root_tree_oid) != expected_oid_length
                or re.fullmatch(r"[0-9a-f]+", root_tree_oid) is None
            ):
                raise ResolutionIntegrityError("commit root tree OID is invalid")

            current_tree_oid = root_tree_oid
            parent_tree_oid = root_tree_oid
            traversal: list[dict[str, object]] = []
            oid_bytes = expected_oid_length // 2
            selected_oid: str | None = None
            selected_mode: str | None = None
            selected_type: str | None = None
            for index, component in enumerate(components):
                tree_payload = self._read_object(
                    mirror,
                    current_tree_oid,
                    object_format=object_format,
                    expected_type="tree",
                )
                entries = self._parse_tree(tree_payload, oid_bytes=oid_bytes)
                component_bytes = component.encode("utf-8", errors="strict")
                matches = [
                    (mode, oid)
                    for mode, name, oid in entries
                    if name == component_bytes
                ]
                if not matches:
                    return self._gap(
                        binding,
                        PATH_ABSENT,
                        object_format=object_format,
                        commit_oid=head_sha,
                        root_tree_oid=root_tree_oid,
                        parent_tree_oid=current_tree_oid,
                        traversal=traversal,
                        evidence={
                            "resolver_schema": RESOLVER_SCHEMA,
                            "reason": "path_component_absent",
                            "absent_component_index": index,
                            "absent_component": component,
                        },
                    )
                if len(matches) != 1:
                    raise ResolutionIntegrityError(
                        "Git tree contains duplicate exact entry names"
                    )
                selected_mode, selected_oid = matches[0]
                selected_type = self._mode_type(selected_mode)
                traversal.append(
                    {
                        "component_index": index,
                        "component": component,
                        "tree_oid": current_tree_oid,
                        "selected_oid": selected_oid,
                        "mode": selected_mode,
                        "object_type": selected_type,
                    }
                )
                is_final = index == len(components) - 1
                if not is_final:
                    if selected_type != "tree":
                        return self._gap(
                            binding,
                            PATH_ABSENT,
                            object_format=object_format,
                            commit_oid=head_sha,
                            root_tree_oid=root_tree_oid,
                            parent_tree_oid=current_tree_oid,
                            object_oid=selected_oid,
                            mode=selected_mode,
                            object_type=selected_type,
                            traversal=traversal,
                            evidence={
                                "resolver_schema": RESOLVER_SCHEMA,
                                "reason": "non_tree_intermediate_component",
                                "component_index": index,
                            },
                        )
                    parent_tree_oid = current_tree_oid
                    current_tree_oid = selected_oid
                else:
                    parent_tree_oid = current_tree_oid

            assert selected_oid is not None
            assert selected_mode is not None
            assert selected_type is not None
            if selected_type not in {"blob", "symlink"}:
                return self._gap(
                    binding,
                    UNSUPPORTED_OBJECT,
                    object_format=object_format,
                    commit_oid=head_sha,
                    root_tree_oid=root_tree_oid,
                    parent_tree_oid=parent_tree_oid,
                    object_oid=selected_oid,
                    mode=selected_mode,
                    object_type=selected_type,
                    traversal=traversal,
                    evidence={
                        "resolver_schema": RESOLVER_SCHEMA,
                        "reason": "path_names_non_blob_git_object",
                        "dereferenced": False,
                    },
                )
            content = self._read_object(
                mirror,
                selected_oid,
                object_format=object_format,
                expected_type="blob",
            )
            content_sha256 = _sha256_bytes(content)
            content_kind, lfs_oid, lfs_size = self._classify_blob(
                content,
                mode_type=selected_type,
            )
            return GitResolution(
                repository=repository,
                head_sha=head_sha,
                source_path=source_path,
                status=RESOLVED,
                object_format=object_format,
                commit_oid=head_sha,
                root_tree_oid=root_tree_oid,
                parent_tree_oid=parent_tree_oid,
                object_oid=selected_oid,
                blob_oid=selected_oid,
                mode=selected_mode,
                object_type=selected_type,
                content_kind=content_kind,
                content_sha256=content_sha256,
                content_size=len(content),
                lfs_oid_sha256=lfs_oid,
                lfs_size=lfs_size,
                traversal=tuple(traversal),
                evidence=self._join_evidence(
                    binding,
                    {
                        "resolver_schema": RESOLVER_SCHEMA,
                        "git_object_id_verified": True,
                        "content_sha256_verified": True,
                        "dereferenced": False,
                    },
                ),
                content=content,
            )
        except _GitPermissionDenied:
            return self._gap(
                binding,
                PERMISSION_DENIED,
                evidence={
                    "resolver_schema": RESOLVER_SCHEMA,
                    "reason": "permission_denied_reading_local_mirror",
                },
            )
        except PermissionError:
            return self._gap(
                binding,
                PERMISSION_DENIED,
                evidence={
                    "resolver_schema": RESOLVER_SCHEMA,
                    "reason": "permission_denied_reading_local_mirror",
                },
            )
        except UnicodeError as exc:
            raise ResolutionIntegrityError(
                "local Git metadata has invalid encoding"
            ) from exc


class SourceSidecarStore:
    """Crash-safe exact-dedup CAS for repository Git blob bytes."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        occurrence_set_sha256: str | None = None,
        upstream_fetch_receipt_sha256: str | None = None,
        binding_inventory_sha256: str | None = None,
        input_binding_count: int | None = None,
        max_pack_bytes: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / _SQLITE_NAME
        database_existed = self.db_path.exists()
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            self.db_path,
            isolation_level=None,
            timeout=60,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = DELETE")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute("PRAGMA busy_timeout = 60000")
        try:
            self._initialize(
                occurrence_set_sha256=occurrence_set_sha256,
                upstream_fetch_receipt_sha256=upstream_fetch_receipt_sha256,
                binding_inventory_sha256=binding_inventory_sha256,
                input_binding_count=input_binding_count,
                max_pack_bytes=max_pack_bytes,
            )
            if not database_existed:
                _fsync_file(self.db_path)
                _fsync_directory(self.root)
            self._recover()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    def _initialize(
        self,
        *,
        occurrence_set_sha256: str | None,
        upstream_fetch_receipt_sha256: str | None,
        binding_inventory_sha256: str | None,
        input_binding_count: int | None,
        max_pack_bytes: int | None,
    ) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS settings(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS packs(
                pack_id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL UNIQUE,
                committed_end INTEGER NOT NULL CHECK(committed_end >= 8),
                blob_count INTEGER NOT NULL CHECK(blob_count >= 0)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS blobs(
                content_sha256 TEXT PRIMARY KEY
                    CHECK(length(content_sha256) = 64),
                size INTEGER NOT NULL CHECK(size >= 0),
                pack_id INTEGER NOT NULL REFERENCES packs(pack_id),
                offset INTEGER NOT NULL CHECK(offset >= 8),
                frame_size INTEGER NOT NULL CHECK(frame_size > 0),
                UNIQUE(pack_id, offset)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS git_objects(
                repository TEXT NOT NULL,
                object_format TEXT NOT NULL
                    CHECK(object_format IN ('sha1', 'sha256')),
                blob_oid TEXT NOT NULL,
                content_sha256 TEXT NOT NULL REFERENCES blobs(content_sha256),
                size INTEGER NOT NULL CHECK(size >= 0),
                PRIMARY KEY(repository, object_format, blob_oid)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bindings(
                repository TEXT NOT NULL,
                head_sha TEXT NOT NULL,
                source_path TEXT NOT NULL,
                status TEXT NOT NULL,
                object_format TEXT,
                commit_oid TEXT,
                root_tree_oid TEXT,
                parent_tree_oid TEXT,
                object_oid TEXT,
                blob_oid TEXT,
                mode TEXT,
                object_type TEXT,
                content_kind TEXT,
                content_sha256 TEXT REFERENCES blobs(content_sha256),
                content_size INTEGER CHECK(content_size >= 0),
                lfs_oid_sha256 TEXT,
                lfs_size INTEGER CHECK(lfs_size >= 0),
                traversal_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                record_sha256 TEXT NOT NULL CHECK(length(record_sha256) = 64),
                PRIMARY KEY(repository, head_sha, source_path),
                CHECK(
                    (status = 'resolved'
                     AND blob_oid IS NOT NULL
                     AND content_sha256 IS NOT NULL
                     AND content_size IS NOT NULL)
                    OR
                    (status != 'resolved'
                     AND blob_oid IS NULL
                     AND content_sha256 IS NULL
                     AND content_size IS NULL)
                )
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS bindings_content_idx
            ON bindings(content_sha256)
            """,
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in statements:
                self._connection.execute(statement)
            existing = dict(self._connection.execute("SELECT key, value FROM settings"))
            if not existing:
                values = (
                    occurrence_set_sha256,
                    upstream_fetch_receipt_sha256,
                    binding_inventory_sha256,
                )
                if any(value is None for value in values):
                    raise SourceStoreError(
                        "new source store requires all upstream binding hashes"
                    )
                assert occurrence_set_sha256 is not None
                assert upstream_fetch_receipt_sha256 is not None
                assert binding_inventory_sha256 is not None
                for name, value in (
                    ("occurrence_set_sha256", occurrence_set_sha256),
                    (
                        "upstream_fetch_receipt_sha256",
                        upstream_fetch_receipt_sha256,
                    ),
                    ("binding_inventory_sha256", binding_inventory_sha256),
                ):
                    if _HEX64_RE.fullmatch(value) is None:
                        raise SourceStoreError(f"{name} is not a lowercase SHA-256")
                if (
                    isinstance(input_binding_count, bool)
                    or not isinstance(input_binding_count, int)
                    or input_binding_count < 0
                ):
                    raise SourceStoreError(
                        "new source store requires non-negative input_binding_count"
                    )
                limit = (
                    DEFAULT_MAX_PACK_BYTES if max_pack_bytes is None else max_pack_bytes
                )
                if (
                    isinstance(limit, bool)
                    or not isinstance(limit, int)
                    or limit < len(_PACK_MAGIC) + _FRAME_HEADER.size
                ):
                    raise SourceStoreError("max_pack_bytes is too small")
                creator_script_sha256 = _script_sha256()
                schema_sha256 = _sqlite_schema_sha256(self._connection)
                resolver_sha256 = _hash_records(
                    "cppmega-ci-source-resolver-contract-v1",
                    (
                        {
                            "resolver_schema": RESOLVER_SCHEMA,
                            "normalization_schema": NORMALIZATION_SCHEMA,
                            "content_semantics": CONTENT_SEMANTICS,
                            "network_access": False,
                            "recursive_tree_walk": False,
                            "symlink_dereference": False,
                            "submodule_dereference": False,
                            "lfs_dereference": False,
                        },
                    ),
                )
                settings = {
                    "schema": STORE_SCHEMA,
                    "pack_schema": PACK_SCHEMA,
                    "receipt_schema": RECEIPT_SCHEMA,
                    "occurrence_set_sha256": occurrence_set_sha256,
                    "upstream_fetch_receipt_sha256": (upstream_fetch_receipt_sha256),
                    "binding_inventory_sha256": binding_inventory_sha256,
                    "input_binding_count": str(input_binding_count),
                    "max_pack_bytes": str(limit),
                    "creator_script_sha256": creator_script_sha256,
                    "sqlite_schema_sha256": schema_sha256,
                    "resolver_sha256": resolver_sha256,
                }
                self._connection.executemany(
                    "INSERT INTO settings(key, value) VALUES (?, ?)",
                    sorted(settings.items()),
                )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        settings = dict(self._connection.execute("SELECT key, value FROM settings"))
        required = {
            "schema",
            "pack_schema",
            "receipt_schema",
            "occurrence_set_sha256",
            "upstream_fetch_receipt_sha256",
            "binding_inventory_sha256",
            "input_binding_count",
            "max_pack_bytes",
            "creator_script_sha256",
            "sqlite_schema_sha256",
            "resolver_sha256",
        }
        if set(settings) != required or settings.get("schema") != STORE_SCHEMA:
            raise SourceStoreError("source store settings are incomplete or stale")
        if settings["pack_schema"] != PACK_SCHEMA:
            raise SourceStoreError("source store pack schema is stale")
        if settings["receipt_schema"] != RECEIPT_SCHEMA:
            raise SourceStoreError("source store receipt schema is stale")
        for name in (
            "occurrence_set_sha256",
            "upstream_fetch_receipt_sha256",
            "binding_inventory_sha256",
            "creator_script_sha256",
            "sqlite_schema_sha256",
            "resolver_sha256",
        ):
            if _HEX64_RE.fullmatch(settings[name]) is None:
                raise SourceStoreError(f"stored {name} is invalid")
        actual_schema_sha256 = _sqlite_schema_sha256(self._connection)
        if actual_schema_sha256 != settings["sqlite_schema_sha256"]:
            raise SourceStoreError("source store SQLite schema hash differs")
        requested = {
            "occurrence_set_sha256": occurrence_set_sha256,
            "upstream_fetch_receipt_sha256": upstream_fetch_receipt_sha256,
            "binding_inventory_sha256": binding_inventory_sha256,
            "input_binding_count": (
                None if input_binding_count is None else str(input_binding_count)
            ),
            "max_pack_bytes": (None if max_pack_bytes is None else str(max_pack_bytes)),
        }
        for name, value in requested.items():
            if value is not None and settings[name] != value:
                raise SourceStoreError(f"requested {name} differs from durable store")
        self._settings = settings

    @property
    def max_pack_bytes(self) -> int:
        return int(self._settings["max_pack_bytes"])

    @property
    def input_binding_count(self) -> int:
        return int(self._settings["input_binding_count"])

    def _orphan_dir(self) -> Path:
        path = self.root / _ORPHAN_DIRECTORY
        path.mkdir(exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise SourceStoreError("unsafe orphan quarantine directory")
        _fsync_directory(self.root)
        return path

    def _quarantine_bytes(
        self,
        source: Path,
        *,
        offset: int,
        reason: str,
    ) -> dict[str, object]:
        size = source.stat().st_size - offset
        if size <= 0:
            raise SourceStoreError("cannot quarantine an empty pack range")
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            handle.seek(offset)
            remaining = size
            while remaining:
                block = handle.read(min(1024 * 1024, remaining))
                if not block:
                    raise SourceStoreError("orphan pack range changed during recovery")
                digest.update(block)
                remaining -= len(block)
        sha256 = digest.hexdigest()
        directory = self._orphan_dir()
        artifact_name = f"{source.stem}-{offset}-{sha256}.orphan"
        artifact = directory / artifact_name
        if not artifact.exists():
            temporary = artifact.with_name(f".{artifact.name}.tmp-{os.getpid()}")
            try:
                with (
                    source.open("rb") as source_handle,
                    temporary.open("xb") as target_handle,
                ):
                    source_handle.seek(offset)
                    remaining = size
                    while remaining:
                        block = source_handle.read(min(1024 * 1024, remaining))
                        if not block:
                            raise SourceStoreError(
                                "orphan pack range changed during quarantine"
                            )
                        target_handle.write(block)
                        remaining -= len(block)
                    target_handle.flush()
                    os.fsync(target_handle.fileno())
                os.replace(temporary, artifact)
                _fsync_directory(directory)
            finally:
                if temporary.exists():
                    temporary.unlink()
        if artifact.stat().st_size != size or _sha256_file(artifact) != sha256:
            raise SourceStoreError("orphan quarantine verification failed")
        record = {
            "schema": RECOVERY_SCHEMA,
            "original_filename": source.name,
            "source_offset": offset,
            "byte_size": size,
            "sha256": sha256,
            "quarantined_filename": artifact_name,
            "reason": reason,
        }
        metadata = directory / f"{artifact_name}.recovery.json"
        if not metadata.exists():
            atomic_write_json(metadata, record)
        elif json.loads(metadata.read_text(encoding="utf-8")) != record:
            raise SourceStoreError("orphan recovery metadata conflict")
        return record

    def recovery_records(self) -> list[dict[str, object]]:
        directory = self.root / _ORPHAN_DIRECTORY
        if not directory.exists():
            return []
        if directory.is_symlink() or not directory.is_dir():
            raise SourceStoreError("unsafe orphan quarantine directory")
        records: list[dict[str, object]] = []
        referenced: set[str] = set()
        for metadata in sorted(directory.glob("*.recovery.json")):
            try:
                encoded = metadata.read_bytes()
                record = json.loads(encoded.decode("utf-8", errors="strict"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise SourceStoreError("invalid recovery metadata") from exc
            if not isinstance(record, dict) or record.get("schema") != RECOVERY_SCHEMA:
                raise SourceStoreError("stale recovery metadata schema")
            filename = record.get("quarantined_filename")
            size = record.get("byte_size")
            digest = record.get("sha256")
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or not isinstance(digest, str)
                or _HEX64_RE.fullmatch(digest) is None
            ):
                raise SourceStoreError("invalid recovery metadata fields")
            artifact = directory / filename
            if (
                not artifact.is_file()
                or artifact.is_symlink()
                or artifact.stat().st_size != size
                or _sha256_file(artifact) != digest
            ):
                raise SourceStoreError("recovery artifact differs from metadata")
            referenced.add(filename)
            records.append(record)
        for artifact in directory.iterdir():
            if (
                artifact.is_file()
                and not artifact.name.startswith(".")
                and not artifact.name.endswith(".recovery.json")
                and artifact.name not in referenced
            ):
                raise SourceStoreError("unmanifested orphan recovery artifact")
        return records

    def _recover(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            rows = self._connection.execute(
                "SELECT pack_id, filename, committed_end FROM packs ORDER BY pack_id"
            ).fetchall()
            known = {str(row["filename"]) for row in rows}
            for row in rows:
                filename = str(row["filename"])
                expected = f"source-pack-{int(row['pack_id']):08d}.cissp"
                if filename != expected:
                    raise SourceStoreError("unsafe indexed source pack filename")
                path = self.root / filename
                if path.is_symlink() or not path.is_file():
                    raise SourceStoreError(
                        f"indexed source pack is missing: {filename}"
                    )
                committed_end = int(row["committed_end"])
                size = path.stat().st_size
                if size < committed_end:
                    raise SourceStoreError(
                        f"{filename} is shorter than its committed boundary"
                    )
                with path.open("r+b") as handle:
                    if handle.read(len(_PACK_MAGIC)) != _PACK_MAGIC:
                        raise SourceStoreError(f"{filename} has invalid pack magic")
                    if size > committed_end:
                        self._quarantine_bytes(
                            path,
                            offset=committed_end,
                            reason="uncommitted_pack_tail",
                        )
                        handle.truncate(committed_end)
                        handle.flush()
                        os.fsync(handle.fileno())
            for path in sorted(self.root.glob(_PACK_GLOB)):
                if path.name not in known:
                    if path.is_symlink() or not path.is_file():
                        raise SourceStoreError("unsafe unindexed source pack")
                    with path.open("rb") as handle:
                        if handle.read(len(_PACK_MAGIC)) != _PACK_MAGIC:
                            raise SourceStoreError(
                                "unindexed source pack has invalid magic"
                            )
                    self._quarantine_bytes(
                        path,
                        offset=0,
                        reason="unindexed_pack",
                    )
                    path.unlink()
                    _fsync_directory(self.root)
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def _new_pack(self) -> sqlite3.Row:
        next_id = int(
            self._connection.execute(
                "SELECT COALESCE(MAX(pack_id), 0) + 1 FROM packs"
            ).fetchone()[0]
        )
        filename = f"source-pack-{next_id:08d}.cissp"
        path = self.root / filename
        with path.open("xb") as handle:
            handle.write(_PACK_MAGIC)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.root)
        self._connection.execute(
            """
            INSERT INTO packs(pack_id, filename, committed_end, blob_count)
            VALUES (?, ?, ?, 0)
            """,
            (next_id, filename, len(_PACK_MAGIC)),
        )
        row = self._connection.execute(
            "SELECT * FROM packs WHERE pack_id = ?",
            (next_id,),
        ).fetchone()
        assert row is not None
        return row

    def _store_blob(self, content: bytes) -> str:
        content_sha256 = _sha256_bytes(content)
        existing = self._connection.execute(
            "SELECT size FROM blobs WHERE content_sha256 = ?",
            (content_sha256,),
        ).fetchone()
        if existing is not None:
            if int(existing["size"]) != len(content):
                raise SourceStoreError("SHA-256 identity has conflicting blob size")
            if self.read_blob(content_sha256) != content:
                raise SourceStoreError("SHA-256 identity has conflicting blob bytes")
            return content_sha256

        frame = (
            _FRAME_HEADER.pack(
                _FRAME_MAGIC,
                bytes.fromhex(content_sha256),
                len(content),
            )
            + content
        )
        pack = self._connection.execute(
            "SELECT * FROM packs ORDER BY pack_id DESC LIMIT 1"
        ).fetchone()
        if pack is None or (
            int(pack["blob_count"]) > 0
            and int(pack["committed_end"]) + len(frame) > self.max_pack_bytes
        ):
            pack = self._new_pack()
        pack_id = int(pack["pack_id"])
        offset = int(pack["committed_end"])
        path = self.root / str(pack["filename"])
        with path.open("r+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() != offset:
                raise SourceStoreError("source pack tail is not at committed boundary")
            handle.write(frame)
            handle.flush()
            os.fsync(handle.fileno())
        self._connection.execute(
            """
            INSERT INTO blobs(content_sha256, size, pack_id, offset, frame_size)
            VALUES (?, ?, ?, ?, ?)
            """,
            (content_sha256, len(content), pack_id, offset, len(frame)),
        )
        self._connection.execute(
            """
            UPDATE packs
            SET committed_end = ?, blob_count = blob_count + 1
            WHERE pack_id = ?
            """,
            (offset + len(frame), pack_id),
        )
        return content_sha256

    def read_blob(self, content_sha256: str) -> bytes:
        row = self._connection.execute(
            """
            SELECT blobs.*, packs.filename, packs.committed_end
            FROM blobs JOIN packs USING(pack_id)
            WHERE content_sha256 = ?
            """,
            (content_sha256,),
        ).fetchone()
        if row is None:
            raise KeyError(content_sha256)
        offset = int(row["offset"])
        frame_size = int(row["frame_size"])
        if offset + frame_size > int(row["committed_end"]):
            raise SourceStoreError("blob frame exceeds committed pack boundary")
        path = self.root / str(row["filename"])
        with path.open("rb") as handle:
            handle.seek(offset)
            header = handle.read(_FRAME_HEADER.size)
            if len(header) != _FRAME_HEADER.size:
                raise SourceStoreError("source blob frame header is truncated")
            magic, raw_digest, size = _FRAME_HEADER.unpack(header)
            content = handle.read(size)
            if len(content) != size:
                raise SourceStoreError("source blob frame payload is truncated")
        if (
            magic != _FRAME_MAGIC
            or frame_size != _FRAME_HEADER.size + size
            or raw_digest.hex() != content_sha256
            or int(row["size"]) != size
            or _sha256_bytes(content) != content_sha256
        ):
            raise SourceStoreError("source blob frame verification failed")
        return content

    def add_resolution(self, resolution: GitResolution) -> bool:
        """Commit one resolution; return ``False`` for an identical replay."""

        if resolution.status not in ALL_STATUSES:
            raise ValueError(f"unsupported binding status {resolution.status!r}")
        durable = resolution.durable_dict()
        record_sha256 = _sha256_bytes(_canonical_json_bytes(durable))
        traversal_json = _canonical_json(durable["traversal"])
        evidence_json = _canonical_json(durable["evidence"])
        key = (
            resolution.repository,
            resolution.head_sha,
            resolution.source_path,
        )
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._connection.execute(
                    """
                    SELECT record_sha256 FROM bindings
                    WHERE repository = ? AND head_sha = ? AND source_path = ?
                    """,
                    key,
                ).fetchone()
                if existing is not None:
                    if str(existing["record_sha256"]) != record_sha256:
                        raise BindingConflictError(
                            "conflicting replay for source binding"
                        )
                    self._connection.execute("COMMIT")
                    return False
                if resolution.status == RESOLVED:
                    content = resolution.content
                    if (
                        content is None
                        or resolution.content_sha256 != _sha256_bytes(content)
                        or resolution.content_size != len(content)
                        or resolution.object_format not in {"sha1", "sha256"}
                        or resolution.blob_oid is None
                    ):
                        raise SourceStoreError(
                            "resolved binding lacks verified Git blob bytes"
                        )
                    actual_git_oid = LocalGitResolver._git_oid(
                        resolution.object_format,
                        "blob",
                        content,
                    )
                    if actual_git_oid != resolution.blob_oid:
                        raise ResolutionIntegrityError(
                            "resolved bytes disagree with recorded Git blob OID"
                        )
                    stored_sha = self._store_blob(content)
                    if stored_sha != resolution.content_sha256:
                        raise SourceStoreError("stored content SHA-256 changed")
                    object_row = self._connection.execute(
                        """
                        SELECT content_sha256, size FROM git_objects
                        WHERE repository = ? AND object_format = ? AND blob_oid = ?
                        """,
                        (
                            resolution.repository,
                            resolution.object_format,
                            resolution.blob_oid,
                        ),
                    ).fetchone()
                    if object_row is None:
                        self._connection.execute(
                            """
                            INSERT INTO git_objects(
                                repository, object_format, blob_oid,
                                content_sha256, size
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                resolution.repository,
                                resolution.object_format,
                                resolution.blob_oid,
                                resolution.content_sha256,
                                resolution.content_size,
                            ),
                        )
                    elif (
                        str(object_row["content_sha256"]) != resolution.content_sha256
                        or int(object_row["size"]) != resolution.content_size
                    ):
                        raise ResolutionIntegrityError(
                            "Git blob OID maps to conflicting content"
                        )
                elif resolution.content is not None:
                    raise SourceStoreError("gap binding unexpectedly has content")

                self._connection.execute(
                    """
                    INSERT INTO bindings(
                        repository, head_sha, source_path, status,
                        object_format, commit_oid, root_tree_oid,
                        parent_tree_oid, object_oid, blob_oid, mode,
                        object_type, content_kind, content_sha256,
                        content_size, lfs_oid_sha256, lfs_size,
                        traversal_json, evidence_json, record_sha256
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        resolution.repository,
                        resolution.head_sha,
                        resolution.source_path,
                        resolution.status,
                        resolution.object_format,
                        resolution.commit_oid,
                        resolution.root_tree_oid,
                        resolution.parent_tree_oid,
                        resolution.object_oid,
                        resolution.blob_oid,
                        resolution.mode,
                        resolution.object_type,
                        resolution.content_kind,
                        resolution.content_sha256,
                        resolution.content_size,
                        resolution.lfs_oid_sha256,
                        resolution.lfs_size,
                        traversal_json,
                        evidence_json,
                        record_sha256,
                    ),
                )
                self._connection.execute("COMMIT")
                return True
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                # A failed transaction can leave only an uncommitted pack tail.
                self._recover()
                raise

    def iter_binding_sidecars(self) -> Iterator[dict[str, object]]:
        for row in self._connection.execute(
            """
            SELECT * FROM bindings
            ORDER BY repository, head_sha, source_path
            """
        ):
            yield {
                "schema": SIDECAR_SCHEMA,
                "repository": str(row["repository"]),
                "head_sha": str(row["head_sha"]),
                "source_path": str(row["source_path"]),
                "status": str(row["status"]),
                "content_semantics": CONTENT_SEMANTICS,
                "object_format": row["object_format"],
                "commit_oid": row["commit_oid"],
                "root_tree_oid": row["root_tree_oid"],
                "parent_tree_oid": row["parent_tree_oid"],
                "object_oid": row["object_oid"],
                "blob_oid": row["blob_oid"],
                "mode": row["mode"],
                "object_type": row["object_type"],
                "content_kind": row["content_kind"],
                "content_sha256": row["content_sha256"],
                "content_size": row["content_size"],
                "lfs_oid_sha256": row["lfs_oid_sha256"],
                "lfs_size": row["lfs_size"],
                "traversal": json.loads(str(row["traversal_json"])),
                "evidence": json.loads(str(row["evidence_json"])),
                "record_sha256": str(row["record_sha256"]),
            }

    def reference_ledger(self) -> dict[str, object]:
        """Return build-action references to CAS hashes, never source bodies."""

        entries: list[dict[str, object]] = []
        evidence_fields = (
            "occurrence_key",
            "occurrence_content_sha256",
            "occurrence_provenance_sha256",
            "action_index",
            "action_entity_id",
            "action_shape_sha256",
            "command_sha256",
            "source_input_index",
            "source_input",
            "cwd",
            "normalization",
            "discarded_heuristic_bindings_sha256",
        )
        for sidecar in self.iter_binding_sidecars():
            resolver_evidence = sidecar["evidence"]
            assert isinstance(resolver_evidence, Mapping)
            binding_evidence = resolver_evidence.get(
                "binding_inventory_evidence",
                [],
            )
            if not isinstance(binding_evidence, list):
                raise SourceStoreError("binding inventory evidence is not a list")
            for evidence in binding_evidence:
                if not isinstance(evidence, Mapping):
                    raise SourceStoreError(
                        "binding inventory action evidence is invalid"
                    )
                entries.append(
                    {
                        "repository": sidecar["repository"],
                        "head_sha": sidecar["head_sha"],
                        "source_path": sidecar["source_path"],
                        "status": sidecar["status"],
                        "content_semantics": CONTENT_SEMANTICS,
                        "content_sha256": sidecar["content_sha256"],
                        "content_size": sidecar["content_size"],
                        "object_format": sidecar["object_format"],
                        "blob_oid": sidecar["blob_oid"],
                        "mode": sidecar["mode"],
                        "object_type": sidecar["object_type"],
                        "content_kind": sidecar["content_kind"],
                        "action_evidence": {
                            field: evidence.get(field)
                            for field in evidence_fields
                            if field in evidence
                        },
                    }
                )
        entries.sort(key=_canonical_json)
        ledger_sha256 = _hash_records(
            "cppmega-ci-source-binding-reference-ledger-v1",
            entries,
        )
        return {
            "schema": REFERENCE_LEDGER_SCHEMA,
            "content_semantics": CONTENT_SEMANTICS,
            "occurrence_set_sha256": self._settings["occurrence_set_sha256"],
            "input_binding_inventory_sha256": self._settings[
                "binding_inventory_sha256"
            ],
            "reference_count": len(entries),
            "ledger_sha256": ledger_sha256,
            "entries": entries,
        }

    def _blob_set_sha256(self) -> str:
        return _hash_records(
            "cppmega-ci-source-blob-set-v1",
            (
                {
                    "content_sha256": str(row["content_sha256"]),
                    "size": int(row["size"]),
                }
                for row in self._connection.execute(
                    "SELECT content_sha256, size FROM blobs ORDER BY content_sha256"
                )
            ),
        )

    def _git_object_set_sha256(self) -> str:
        return _hash_records(
            "cppmega-ci-source-git-object-set-v1",
            (
                {
                    "repository": str(row["repository"]),
                    "object_format": str(row["object_format"]),
                    "blob_oid": str(row["blob_oid"]),
                    "content_sha256": str(row["content_sha256"]),
                    "size": int(row["size"]),
                }
                for row in self._connection.execute(
                    """
                    SELECT repository, object_format, blob_oid,
                           content_sha256, size
                    FROM git_objects
                    ORDER BY repository, object_format, blob_oid
                    """
                )
            ),
        )

    def _binding_set_sha256(self) -> str:
        return _hash_records(
            "cppmega-ci-source-binding-set-v1",
            (sidecar for sidecar in self.iter_binding_sidecars()),
        )

    def verify(self) -> dict[str, object]:
        """Verify SQLite, every pack frame, Git identities, and logical sets."""

        with self._lock:
            integrity = [
                str(row[0])
                for row in self._connection.execute("PRAGMA integrity_check").fetchall()
            ]
            if integrity != ["ok"]:
                raise SourceStoreError(f"SQLite integrity_check failed: {integrity}")
            if self._connection.execute("PRAGMA foreign_key_check").fetchall():
                raise SourceStoreError("SQLite foreign_key_check failed")
            pack_rows = self._connection.execute(
                "SELECT * FROM packs ORDER BY pack_id"
            ).fetchall()
            actual_names = {path.name for path in self.root.glob(_PACK_GLOB)}
            expected_names = {str(row["filename"]) for row in pack_rows}
            if actual_names != expected_names:
                raise SourceStoreError("source pack file set differs from SQLite")
            pack_hashes: list[dict[str, object]] = []
            verified_blobs = 0
            for pack in pack_rows:
                path = self.root / str(pack["filename"])
                committed_end = int(pack["committed_end"])
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != committed_end
                ):
                    raise SourceStoreError("source pack committed size mismatch")
                with path.open("rb") as handle:
                    if handle.read(len(_PACK_MAGIC)) != _PACK_MAGIC:
                        raise SourceStoreError("source pack magic mismatch")
                expected_offset = len(_PACK_MAGIC)
                count = 0
                for blob in self._connection.execute(
                    "SELECT * FROM blobs WHERE pack_id = ? ORDER BY offset",
                    (int(pack["pack_id"]),),
                ):
                    if int(blob["offset"]) != expected_offset:
                        raise SourceStoreError("source pack has frame gap or overlap")
                    self.read_blob(str(blob["content_sha256"]))
                    expected_offset += int(blob["frame_size"])
                    count += 1
                    verified_blobs += 1
                if expected_offset != committed_end or count != int(pack["blob_count"]):
                    raise SourceStoreError("source pack accounting mismatch")
                pack_hashes.append(
                    {
                        "filename": str(pack["filename"]),
                        "committed_end": committed_end,
                        "blob_count": count,
                        "sha256": _sha256_file(path, limit=committed_end),
                    }
                )
            blob_count = int(
                self._connection.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
            )
            if blob_count != verified_blobs:
                raise SourceStoreError("not every source blob frame was verified")
            for row in self._connection.execute(
                """
                SELECT repository, object_format, blob_oid, content_sha256, size
                FROM git_objects ORDER BY repository, object_format, blob_oid
                """
            ):
                content = self.read_blob(str(row["content_sha256"]))
                if len(content) != int(row["size"]):
                    raise SourceStoreError("Git object size differs from source blob")
                actual_oid = LocalGitResolver._git_oid(
                    str(row["object_format"]),
                    "blob",
                    content,
                )
                if actual_oid != str(row["blob_oid"]):
                    raise ResolutionIntegrityError(
                        "stored source blob differs from Git object ID"
                    )
            sidecars = list(self.iter_binding_sidecars())
            status_counts = Counter(str(item["status"]) for item in sidecars)
            if any(status not in ALL_STATUSES for status in status_counts):
                raise SourceStoreError("stored binding has unknown status")
            inventory_records: list[dict[str, object]] = []
            for sidecar in sidecars:
                durable = {
                    key: value
                    for key, value in sidecar.items()
                    if key not in {"schema", "record_sha256"}
                }
                if (
                    _sha256_bytes(_canonical_json_bytes(durable))
                    != sidecar["record_sha256"]
                ):
                    raise SourceStoreError(
                        "stored binding record SHA-256 differs from its fields"
                    )
                if sidecar["content_semantics"] != CONTENT_SEMANTICS:
                    raise SourceStoreError(
                        "stored binding has unsupported content semantics"
                    )
                resolver_evidence = sidecar["evidence"]
                if not isinstance(resolver_evidence, Mapping):
                    raise SourceStoreError("stored resolver evidence is invalid")
                inventory_record = resolver_evidence.get("binding_inventory_record")
                if not isinstance(inventory_record, Mapping):
                    raise SourceStoreError(
                        "stored binding lacks inventory membership evidence"
                    )
                durable_inventory_record = dict(inventory_record)
                if (
                    durable_inventory_record.get("repository"),
                    durable_inventory_record.get("head_sha"),
                    durable_inventory_record.get("source_path"),
                ) != (
                    sidecar["repository"],
                    sidecar["head_sha"],
                    sidecar["source_path"],
                ):
                    raise SourceStoreError(
                        "stored binding key disagrees with inventory membership"
                    )
                inventory_records.append(durable_inventory_record)
                if sidecar["status"] == RESOLVED:
                    object_row = self._connection.execute(
                        """
                        SELECT content_sha256, size
                        FROM git_objects
                        WHERE repository = ?
                          AND object_format = ?
                          AND blob_oid = ?
                        """,
                        (
                            sidecar["repository"],
                            sidecar["object_format"],
                            sidecar["blob_oid"],
                        ),
                    ).fetchone()
                    if (
                        object_row is None
                        or str(object_row["content_sha256"])
                        != sidecar["content_sha256"]
                        or int(object_row["size"]) != sidecar["content_size"]
                    ):
                        raise SourceStoreError(
                            "resolved binding disagrees with Git-object index"
                        )
            if len(sidecars) > self.input_binding_count:
                raise SourceStoreError("stored binding count exceeds input inventory")
            partial_inventory_sha256 = _hash_records(
                "cppmega-ci-source-binding-inventory-v1",
                _inventory_binding_hash_records(inventory_records),
            )
            partial_inventory = {
                "schema": INVENTORY_SCHEMA,
                "occurrence_set_sha256": self._settings["occurrence_set_sha256"],
                "upstream_fetch_receipt_sha256": self._settings[
                    "upstream_fetch_receipt_sha256"
                ],
                "binding_count": len(inventory_records),
                "binding_inventory_sha256": partial_inventory_sha256,
                "bindings": inventory_records,
            }
            try:
                verify_binding_inventory(partial_inventory)
            except ExtractionError as exc:
                raise SourceStoreError(
                    f"stored inventory membership is invalid: {exc}"
                ) from exc
            if (
                len(sidecars) == self.input_binding_count
                and partial_inventory_sha256
                != self._settings["binding_inventory_sha256"]
            ):
                raise SourceStoreError(
                    "complete binding set differs from frozen input inventory"
                )
            recovery = self.recovery_records()
            return {
                "ok": True,
                "schema": STORE_SCHEMA,
                "binding_count": len(sidecars),
                "missing_binding_count": self.input_binding_count - len(sidecars),
                "blob_count": blob_count,
                "git_object_count": int(
                    self._connection.execute(
                        "SELECT COUNT(*) FROM git_objects"
                    ).fetchone()[0]
                ),
                "status_counts": dict(sorted(status_counts.items())),
                "logical_blob_set_sha256": self._blob_set_sha256(),
                "logical_git_object_set_sha256": self._git_object_set_sha256(),
                "logical_binding_set_sha256": self._binding_set_sha256(),
                "pack_hashes": pack_hashes,
                "recovery": {
                    "orphan_count": len(recovery),
                    "records_sha256": _hash_records(
                        "cppmega-ci-source-recovery-records-v1",
                        recovery,
                    ),
                    "records": recovery,
                },
            }

    def receipt(self) -> dict[str, object]:
        verification = self.verify()
        status_counts = dict(verification["status_counts"])
        resolved_count = int(status_counts.get(RESOLVED, 0))
        gap_counts = {
            status: int(status_counts.get(status, 0))
            for status in sorted(GAP_STATUSES)
            if status_counts.get(status, 0)
        }
        exhaustive_status_counts = {
            status: int(status_counts.get(status, 0)) for status in sorted(ALL_STATUSES)
        }
        missing_count = int(verification["missing_binding_count"])
        complete = (
            missing_count == 0
            and resolved_count == self.input_binding_count
            and not gap_counts
        )
        ledger = self.reference_ledger()
        return {
            "schema": RECEIPT_SCHEMA,
            "status": "complete" if complete else "incomplete",
            "content_semantics": CONTENT_SEMANTICS,
            "input_binding_count": self.input_binding_count,
            "input_binding_inventory_sha256": self._settings[
                "binding_inventory_sha256"
            ],
            "occurrence_set_sha256": self._settings["occurrence_set_sha256"],
            "upstream_fetch_receipt_sha256": self._settings[
                "upstream_fetch_receipt_sha256"
            ],
            "resolved_binding_count": resolved_count,
            "missing_binding_count": missing_count,
            "gap_status_counts": gap_counts,
            "resolution_status_counts": exhaustive_status_counts,
            "blob_count": verification["blob_count"],
            "git_object_count": verification["git_object_count"],
            "logical_blob_set_sha256": verification["logical_blob_set_sha256"],
            "logical_git_object_set_sha256": verification[
                "logical_git_object_set_sha256"
            ],
            "logical_binding_set_sha256": verification["logical_binding_set_sha256"],
            "build_action_reference_count": ledger["reference_count"],
            "binding_reference_ledger_sha256": ledger["ledger_sha256"],
            "pack_hashes": verification["pack_hashes"],
            "resolver_schema": RESOLVER_SCHEMA,
            "resolver_sha256": self._settings["resolver_sha256"],
            "store_schema": STORE_SCHEMA,
            "pack_schema": PACK_SCHEMA,
            "binding_sidecar_schema": SIDECAR_SCHEMA,
            "reference_ledger_schema": REFERENCE_LEDGER_SCHEMA,
            "normalization_schema": NORMALIZATION_SCHEMA,
            "sqlite_schema_sha256": self._settings["sqlite_schema_sha256"],
            "script_sha256": self._settings["creator_script_sha256"],
            "recovery": verification["recovery"],
            "verification": {"mode": "full", "ok": True},
        }

    completion_receipt = receipt
    build_receipt = receipt
    create_receipt = receipt

    def write_receipt(
        self,
        path: str | os.PathLike[str],
    ) -> dict[str, object]:
        receipt = self.receipt()
        atomic_write_json(path, receipt)
        return receipt

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> SourceSidecarStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def materialize_inventory(
    inventory: Mapping[str, Any],
    mirror_mapping: Mapping[str, str | os.PathLike[str] | Mapping[str, object] | None],
    store_root: str | os.PathLike[str],
    *,
    max_pack_bytes: int | None = None,
) -> dict[str, object]:
    """Resolve and durably store every inventory tuple in deterministic order."""

    verify_binding_inventory(inventory)
    resolver = LocalGitResolver(mirror_mapping)
    bindings = inventory["bindings"]
    assert isinstance(bindings, list)
    with SourceSidecarStore(
        store_root,
        occurrence_set_sha256=str(inventory["occurrence_set_sha256"]),
        upstream_fetch_receipt_sha256=str(inventory["upstream_fetch_receipt_sha256"]),
        binding_inventory_sha256=str(inventory["binding_inventory_sha256"]),
        input_binding_count=int(inventory["binding_count"]),
        max_pack_bytes=max_pack_bytes,
    ) as store:
        for binding in sorted(
            bindings,
            key=lambda item: (
                str(item["repository"]),
                str(item["head_sha"]),
                str(item["source_path"]),
            ),
        ):
            store.add_resolution(resolver.resolve(binding))
        return store.receipt()


def _load_mirror_mapping(path: Path) -> Mapping[str, object]:
    value, _raw = _read_json_object(path, where="mirror mapping")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser(
        "inventory",
        help="extract a binding inventory from a frozen CI content store",
    )
    inventory.add_argument("--ci-store", type=Path, required=True)
    inventory.add_argument("--fetch-receipt", type=Path, required=True)
    inventory.add_argument("--content-store-receipt", type=Path)
    inventory.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser(
        "build",
        help="resolve an inventory using explicit local bare mirrors",
    )
    build.add_argument("--inventory", type=Path, required=True)
    build.add_argument("--mirrors", type=Path, required=True)
    build.add_argument("--store", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument(
        "--ledger",
        type=Path,
        help="optional body-free build-action reference ledger",
    )
    build.add_argument("--max-pack-bytes", type=int)

    verify = subparsers.add_parser("verify", help="fully verify a source CAS")
    verify.add_argument("--store", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "inventory":
            result = extract_binding_inventory(
                args.ci_store,
                args.fetch_receipt,
                content_store_receipt_path=args.content_store_receipt,
            )
            atomic_write_json(args.output, result)
        elif args.command == "build":
            inventory, _raw = _read_json_object(
                args.inventory,
                where="binding inventory",
            )
            mirrors = _load_mirror_mapping(args.mirrors)
            result = materialize_inventory(
                inventory,
                mirrors,
                args.store,
                max_pack_bytes=args.max_pack_bytes,
            )
            atomic_write_json(args.receipt, result)
            if args.ledger is not None:
                with SourceSidecarStore(args.store) as store:
                    atomic_write_json(args.ledger, store.reference_ledger())
        else:
            with SourceSidecarStore(args.store) as store:
                result = store.verify()
        json.dump(
            result,
            sys.stdout,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0
    except (
        SourceSidecarError,
        ValueError,
        OSError,
        sqlite3.Error,
    ) as exc:
        print(f"ci_source_sidecars: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
