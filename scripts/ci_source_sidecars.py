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
import codecs
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self
from urllib.parse import quote

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci_source_binding_projection import (
    MAX_SOURCE_BINDING_PROJECTION_RECORD_BYTES,
    SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
    SOURCE_BINDING_PROJECTION_SCHEMA,
    SourceBindingProjectionError,
    SourceBindingProjector,
    projection_record_key,
    projection_script_sha256,
    target_parser_script_sha256,
)

OCCURRENCE_SCHEMA = "cppmega_ci_chunk_occurrence_v3"
TRAINING_SIDECAR_SCHEMA = "cppmega_ci_chunk_training_sidecars_v2"
CONTENT_STORE_SCHEMA = "cppmega_ci_content_store_v1"
CONTENT_STORE_RECEIPT_SCHEMA = "cppmega_ci_content_store_receipt_v1"
CONTENT_STORE_PACK_SCHEMA = "cppmega_ci_content_pack_v1"
FETCH_RECEIPT_SCHEMA = "cppmega_ci_stream_fetch_receipt_v3"
CASE5_EXPORT_SCHEMA = "cppmega_ci_content_store_case5_export_v2"
REPRESENTATIVE_LEDGER_SCHEMA = "cppmega_ci_token_sequence_representative_ledger_v1"
INVENTORY_SCHEMA = "cppmega_ci_source_binding_inventory_v3"
STORE_SCHEMA = "cppmega_ci_source_sidecar_store_v2"
PACK_SCHEMA = "cppmega_ci_source_blob_pack_v1"
RECEIPT_SCHEMA = "cppmega_ci_source_sidecar_receipt_v2"
SIDECAR_SCHEMA = "cppmega_ci_source_binding_sidecar_v1"
REFERENCE_LEDGER_SCHEMA = "cppmega_ci_source_reference_ledger_v2"
RESOLVER_SCHEMA = "cppmega_ci_local_git_blob_resolver_v2"
NORMALIZATION_SCHEMA = "cppmega_ci_source_path_normalization_v2"
RECOVERY_SCHEMA = "cppmega_ci_source_sidecar_recovery_v1"
CONTENT_STORE_RECOVERY_SCHEMA = "cppmega_ci_content_store_recovery_v1"
CONTENT_SEMANTICS = "repository_blob_content"
TOKEN_SEQUENCE_ENCODING = "cppmega-token-sequence-u32be-v1"

RESOLVED = "resolved"
PATH_ABSENT = "path_absent"
COMMIT_ABSENT = "commit_absent"
REPO_UNAVAILABLE = "repo_unavailable"
PERMISSION_DENIED = "permission_denied"
DELETED_FORK = "deleted_fork"
UNSUPPORTED_OBJECT = "unsupported_object"
AMBIGUOUS_PATH = "ambiguous_path"
GENERATED_OR_MUTATED_UNRESOLVABLE = "generated_or_mutated_unresolvable"
CHECKOUT_PROVENANCE_UNRESOLVABLE = "checkout_provenance_unresolvable"

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
        CHECKOUT_PROVENANCE_UNRESOLVABLE,
    }
)
ALL_STATUSES = GAP_STATUSES | {RESOLVED}

DEFAULT_MAX_PACK_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_GIT_OBJECT_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JSONL_RECORD_BYTES = MAX_SOURCE_BINDING_PROJECTION_RECORD_BYTES
MAX_PROVENANCE_BYTES = 64 * 1024 * 1024
MAX_PACK_RECORDS = 100_000
MAX_RECOVERY_RECORDS = 100_000
MAX_SOURCE_SETTINGS = 64
DEFAULT_TRANSACTION_BATCH_SIZE = 128
_PACK_MAGIC = b"CISSPK1\n"
_FRAME_MAGIC = b"CISSFRM1"
_FRAME_HEADER = struct.Struct(">8s32sQ")
_CONTENT_PACK_MAGIC = b"CICSPK1\n"
_CONTENT_FRAME_MAGIC = b"CICSFRM1"
_CONTENT_FRAME_HEADER = struct.Struct(">8s32sQQ")
_PACK_GLOB = "source-pack-*.cissp"
_SQLITE_NAME = "index.sqlite3"
_ORPHAN_DIRECTORY = "orphaned"
_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_DRIVE_RELATIVE_RE = re.compile(r"^[A-Za-z]:(?![\\/])")
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"
_LFS_OID_RE = re.compile(rb"^oid sha256:([0-9a-f]{64})$", re.MULTILINE)
_LFS_SIZE_RE = re.compile(rb"^size ([0-9]+)$", re.MULTILINE)
_OCCURRENCE_FIELDS = (
    "repo",
    "run_attempt",
    "job",
    "step",
    "chunk_ordinal",
)
_REPRESENTATIVE_SELECTION = (
    "one-per-eligible-token-sequence; "
    "content-sha256-then-eligible-occurrence-key"
)
_SOURCE_BINDING_PROJECTION_ARTIFACT = "source_binding_projection.jsonl"
_SOURCE_BINDING_PROJECTION_ORDER = (
    "occurrence-key-then-action-index-then-source-input-index"
)
_SOURCE_BINDING_PROJECTION_CLAIM_BOUNDARY = (
    "derived source-binding semantics only; upstream parser sidecars, parser "
    "hashes, occurrence provenance, payload bytes, token IDs, token counts "
    "and CAS receipts are unchanged"
)
_SOURCE_BINDING_PROJECTION_VERIFIED_GAP = (
    "source_binding_projection_verified_unbound_input"
)
_SOURCE_BINDING_PROJECTION_REASONS = {
    "unchanged": {
        "current_binding_verified",
        "legacy_binding_already_current",
    },
    "added": {"binding_added_by_current_semantics"},
    "dropped": {"unsafe_or_unresolvable_binding_dropped"},
    "modified": {
        "repository_and_source_path_corrected",
        "pull_request_repository_corrected",
        "runner_cwd_relative_path_normalized",
        "binding_semantics_corrected",
    },
}
_NO_PROJECTED_BINDING = object()


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


def _sqlite_schema_records(
    connection: sqlite3.Connection,
) -> Iterator[list[str | None]]:
    for row in connection.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ):
        yield [
            str(row["type"]),
            str(row["name"]),
            str(row["tbl_name"]),
            None if row["sql"] is None else str(row["sql"]),
        ]


def _sqlite_schema_sha256(connection: sqlite3.Connection) -> str:
    return _hash_records(
        "cppmega-ci-source-sqlite-schema-v1",
        _sqlite_schema_records(connection),
    )


def _source_store_settings(
    connection: sqlite3.Connection,
) -> dict[str, str]:
    settings: dict[str, str] = {}
    for row in connection.execute(
        "SELECT key, value FROM settings ORDER BY key"
    ):
        key = str(row["key"])
        if key in settings or len(settings) >= MAX_SOURCE_SETTINGS:
            raise SourceStoreError(
                "source store settings exceed their bounded schema"
            )
        settings[key] = str(row["value"])
    return settings


def _content_store_sqlite_schema_sha256(
    connection: sqlite3.Connection,
) -> str:
    return _hash_records(
        "cppmega-ci-sqlite-schema-v1",
        _sqlite_schema_records(connection),
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


def _temporary_sibling(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for ordinal in range(1_000):
        candidate = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-"
            f"{threading.get_ident()}-{ordinal}"
        )
        if not candidate.exists():
            return candidate
    raise OSError(f"cannot allocate temporary artifact for {destination}")


def _publish_temporary(
    temporary: Path,
    destination: Path,
    *,
    force: bool,
    expected_existing_sha256: str | None,
) -> None:
    if expected_existing_sha256 is not None:
        _require_hex64(
            expected_existing_sha256,
            where=f"expected existing digest for {destination}",
        )
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise SourceStoreError(
                f"refusing to overwrite unsafe artifact {destination}"
            )
        current = _stable_file_sha256(destination)
        if not force and current != expected_existing_sha256:
            raise SourceStoreError(
                f"refusing to overwrite existing artifact {destination}; "
                "use --force or its exact expected SHA-256"
            )
    elif expected_existing_sha256 is not None:
        raise SourceStoreError(
            f"expected existing artifact is absent: {destination}"
        )
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _preflight_publication(
    destination: Path,
    *,
    force: bool,
    expected_existing_sha256: str | None,
) -> None:
    if expected_existing_sha256 is not None:
        _require_hex64(
            expected_existing_sha256,
            where=f"expected existing digest for {destination}",
        )
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise SourceStoreError(
                f"refusing to overwrite unsafe artifact {destination}"
            )
        current = _stable_file_sha256(destination)
        if not force and current != expected_existing_sha256:
            raise SourceStoreError(
                f"refusing to overwrite existing artifact {destination}; "
                "use --force or its exact expected SHA-256"
            )
    elif expected_existing_sha256 is not None:
        raise SourceStoreError(
            f"expected existing artifact is absent: {destination}"
        )


def publish_json(
    path: str | os.PathLike[str],
    value: object,
    *,
    force: bool = False,
    expected_existing_sha256: str | None = None,
) -> str:
    destination = Path(path)
    _preflight_publication(
        destination,
        force=force,
        expected_existing_sha256=expected_existing_sha256,
    )
    temporary = _temporary_sibling(destination)
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary(
            temporary,
            destination,
            force=force,
            expected_existing_sha256=expected_existing_sha256,
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_bytes(raw)


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


def _path_platform(*values: str | None) -> str | None:
    """Infer separators only from an absolute path, never from punctuation."""

    for raw in values:
        if not isinstance(raw, str):
            continue
        value = raw.strip()
        if _WINDOWS_ABSOLUTE_RE.match(value) is not None or value.startswith("\\\\"):
            return "windows"
        if value.startswith("/"):
            return "posix"
    return None


def _platform_path(value: str, *, platform: str) -> str:
    if platform == "windows":
        return value.replace("\\", "/")
    if platform == "posix":
        return value
    raise ValueError(f"unsupported path platform {platform!r}")


def _is_absolute(value: str, *, platform: str) -> bool:
    if platform == "windows":
        return (
            _WINDOWS_ABSOLUTE_RE.match(value) is not None
            or value.startswith("//")
        )
    return value.startswith("/")


def _safe_components(value: str, *, allow_empty: bool = False) -> tuple[str, ...] | None:
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
    if not components and not allow_empty:
        return None
    return tuple(components)


def _hosted_checkout_split(
    absolute_path: str,
    *,
    platform: str,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Split an exact GitHub-hosted checkout root from its relative suffix.

    The structural prefixes are fixed.  Workspace and checkout directory names
    are opaque and are never inferred from a repository basename.
    """

    value = _platform_path(absolute_path, platform=platform)
    if platform == "windows":
        if _WINDOWS_ABSOLUTE_RE.match(value) is None:
            return None
        drive, remainder = value[:2], value[3:]
        normalized = _safe_components(remainder, allow_empty=True)
        if normalized is None or len(normalized) < 3:
            return None
        if normalized[0].casefold() != "a":
            return None
        root = (drive.casefold(), *normalized[:3])
        return root, normalized[3:]

    normalized = _safe_components(value.removeprefix("/"), allow_empty=True)
    if normalized is None:
        return None
    root_length: int | None = None
    if len(normalized) >= 5 and tuple(item.casefold() for item in normalized[:3]) in {
        ("home", "runner", "work"),
        ("users", "runner", "work"),
    }:
        root_length = 5
    elif len(normalized) >= 3 and normalized[0].casefold() == "__w":
        root_length = 3
    if root_length is None:
        return None
    return normalized[:root_length], normalized[root_length:]


def _absolute_join(
    cwd: str,
    source: str,
    *,
    platform: str,
) -> tuple[str, ...] | None:
    normalized_cwd = _platform_path(cwd, platform=platform)
    normalized_source = _platform_path(source, platform=platform)
    if _WINDOWS_DRIVE_RELATIVE_RE.match(normalized_source):
        return None
    if _is_absolute(normalized_source, platform=platform):
        candidate = normalized_source
    else:
        candidate = normalized_cwd.rstrip("/") + "/" + normalized_source
    if platform == "windows":
        if _WINDOWS_ABSOLUTE_RE.match(candidate) is None:
            return None
        drive = candidate[:2].casefold()
        components = _safe_components(candidate[3:], allow_empty=True)
        return None if components is None else (drive, *components)
    components = _safe_components(candidate.removeprefix("/"), allow_empty=True)
    return components


def normalize_source_candidates(
    source_input: str,
    cwd: str | None,
    *,
    repository: str | None = None,
    platform: str | None = None,
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
    del repository  # Repository basenames are deliberately not path evidence.
    if _WINDOWS_DRIVE_RELATIVE_RE.match(raw):
        return PathNormalization(
            GENERATED_OR_MUTATED_UNRESOLVABLE,
            (),
            source_input,
            cwd,
            "windows_drive_relative_path",
        )
    selected_platform = platform or _path_platform(cwd, raw)
    if selected_platform not in {"posix", "windows"}:
        # Pure relative normalization remains available as a lexical utility,
        # but extraction never uses it because checkout provenance requires an
        # absolute, structurally recognized cwd.
        if cwd is None or not cwd.strip() or cwd.strip() == ".":
            combined = raw
        else:
            combined = f"{cwd.strip()}/{raw}"
        normalized = _safe_components(combined)
        if normalized is None:
            return PathNormalization(
                GENERATED_OR_MUTATED_UNRESOLVABLE,
                (),
                source_input,
                cwd,
                "relative_path_escapes_unknown_checkout",
            )
        return PathNormalization(
            RESOLVED,
            ("/".join(normalized),),
            source_input,
            cwd,
            None,
        )
    if cwd is None or not cwd.strip():
        return PathNormalization(
            CHECKOUT_PROVENANCE_UNRESOLVABLE,
            (),
            source_input,
            cwd,
            "missing_checkout_cwd",
        )
    cwd_value = _platform_path(cwd.strip(), platform=selected_platform)
    source_value = _platform_path(raw, platform=selected_platform)
    cwd_split = _hosted_checkout_split(cwd_value, platform=selected_platform)
    absolute = _absolute_join(
        cwd_value,
        source_value,
        platform=selected_platform,
    )
    if cwd_split is None or absolute is None:
        return PathNormalization(
            CHECKOUT_PROVENANCE_UNRESOLVABLE,
            (),
            source_input,
            cwd,
            "cwd_outside_exact_hosted_checkout",
        )
    checkout_root, _cwd_suffix = cwd_split
    absolute_root = tuple(absolute[: len(checkout_root)])
    root_matches = (
        tuple(item.casefold() for item in absolute_root)
        == tuple(item.casefold() for item in checkout_root)
        if selected_platform == "windows"
        else absolute_root == checkout_root
    )
    if not root_matches:
        return PathNormalization(
            GENERATED_OR_MUTATED_UNRESOLVABLE,
            (),
            source_input,
            cwd,
            "path_escapes_exact_checkout_root",
        )
    relative = absolute[len(checkout_root) :]
    if not relative:
        return PathNormalization(
            GENERATED_OR_MUTATED_UNRESOLVABLE,
            (),
            source_input,
            cwd,
            "path_names_checkout_root",
        )
    return PathNormalization(
        RESOLVED,
        ("/".join(relative),),
        source_input,
        cwd,
        None,
    )


def normalize_source_path(
    source_input: str,
    cwd: str | None,
    *,
    repository: str | None = None,
    platform: str | None = None,
) -> str:
    """Return one normalized path or raise when the join is not exact."""

    result = normalize_source_candidates(
        source_input,
        cwd,
        repository=repository,
        platform=platform,
    )
    if result.status != RESOLVED:
        raise ValueError(f"source path is not uniquely resolvable: {result.status}")
    return result.candidates[0]


def _decode_provenance(row: sqlite3.Row) -> dict[str, Any]:
    raw_size = int(row["provenance_raw_size"])
    if raw_size < 0 or raw_size > MAX_PROVENANCE_BYTES:
        raise ExtractionError("occurrence provenance raw size is outside policy")
    compressed = bytes(row["provenance_zlib"])
    if len(compressed) > MAX_PROVENANCE_BYTES:
        raise ExtractionError("occurrence provenance compressed size is outside policy")
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, raw_size + 1)
    except zlib.error as exc:
        raise ExtractionError("occurrence provenance zlib is invalid") from exc
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ExtractionError("occurrence provenance has a non-canonical zlib stream")
    if len(raw) != raw_size:
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


def _read_json_object(path: Path, *, where: str) -> tuple[dict[str, Any], bytes]:
    try:
        handle, before = _open_regular_no_follow(path)
        with handle:
            if before.st_size > MAX_JSON_BYTES:
                raise ExtractionError(f"{where} exceeds the JSON size limit")
            raw = handle.read(MAX_JSON_BYTES + 1)
            after = os.fstat(handle.fileno())
    except ExtractionError:
        raise
    except OSError as exc:
        raise ExtractionError(f"{where} is not valid JSON") from exc
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        len(raw) != before.st_size
        or len(raw) > MAX_JSON_BYTES
        or any(
            getattr(before, field) != getattr(after, field)
            for field in identity_fields
        )
    ):
        raise ExtractionError(f"{where} changed while reading")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"{where} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ExtractionError(f"{where} must be a JSON object")
    return value, raw


class _RecordHasher:
    def __init__(self, domain: str) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(domain.encode("ascii"))
        self._digest.update(b"\0")
        self.count = 0

    def update(self, record: object) -> None:
        encoded = _canonical_json_bytes(record)
        self._digest.update(struct.pack(">Q", len(encoded)))
        self._digest.update(encoded)
        self.count += 1

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _open_regular_no_follow(path: Path) -> tuple[io.BufferedReader, os.stat_result]:
    path_metadata = os.lstat(path)
    if stat.S_ISLNK(path_metadata.st_mode) or not stat.S_ISREG(
        path_metadata.st_mode
    ):
        raise ExtractionError(f"{path} is not a regular non-symlink file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_dev != path_metadata.st_dev
            or metadata.st_ino != path_metadata.st_ino
        ):
            raise ExtractionError(f"{path} is not a regular file")
        return os.fdopen(descriptor, "rb"), metadata
    except BaseException:
        os.close(descriptor)
        raise


def _stable_file_sha256(path: Path, *, expected_size: int | None = None) -> str:
    digest = hashlib.sha256()
    handle, before = _open_regular_no_follow(path)
    with handle:
        if expected_size is not None and before.st_size != expected_size:
            raise ExtractionError(f"{path.name} size differs from its receipt")
        remaining = before.st_size
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                raise ExtractionError(f"{path.name} changed while hashing")
            digest.update(block)
            remaining -= len(block)
        after = os.fstat(handle.fileno())
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise ExtractionError(f"{path.name} changed while hashing")
    return digest.hexdigest()


def _copy_stable_sqlite(source: Path, destination: Path) -> None:
    source_handle, before = _open_regular_no_follow(source)
    with source_handle, destination.open("xb") as target:
        shutil.copyfileobj(source_handle, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
        after = os.fstat(source_handle.fileno())
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise ExtractionError("content-store SQLite changed while snapshotting")


def _content_set_digest(connection: sqlite3.Connection) -> str:
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
            for row in connection.execute(
                """
                SELECT sha256, raw_size, token_count, tokenizer_fingerprint,
                       token_sequence_sha256
                FROM contents
                ORDER BY sha256
                """
            )
        ),
    )


def _token_sequence_set_digest(connection: sqlite3.Connection) -> str:
    return _hash_records(
        "cppmega-ci-token-sequence-set-v1",
        (
            {
                "token_sequence_sha256": str(row["token_sequence_sha256"]),
                "token_count": int(row["token_count"]),
                "tokenizer_fingerprint": str(row["tokenizer_fingerprint"]),
                "encoding": TOKEN_SEQUENCE_ENCODING,
            }
            for row in connection.execute(
                """
                SELECT token_sequence_sha256, token_count, tokenizer_fingerprint
                FROM token_sequences
                ORDER BY token_sequence_sha256
                """
            )
        ),
    )


def _content_store_sqlite_logical_sha256(
    connection: sqlite3.Connection,
) -> str:
    def records() -> Iterator[object]:
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
        row = connection.execute(
            "SELECT * FROM stats WHERE singleton = 1"
        ).fetchone()
        if row is not None:
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


def _content_store_occurrence_set_sha256(
    connection: sqlite3.Connection,
    *,
    projection_scope: Counter[str] | None = None,
) -> str:
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
            provenance = _decode_provenance(row)
            if projection_scope is not None:
                if provenance.get("schema") != OCCURRENCE_SCHEMA:
                    raise ExtractionError(
                        "source-binding projection occurrence schema is stale"
                    )
                chunk = provenance.get("chunk")
                training = (
                    chunk.get("training_sidecars")
                    if isinstance(chunk, Mapping)
                    else None
                )
                actions = (
                    training.get("build_actions")
                    if isinstance(training, Mapping)
                    and training.get("schema") == TRAINING_SIDECAR_SCHEMA
                    else None
                )
                if not isinstance(actions, list):
                    raise ExtractionError(
                        "source-binding projection build_actions is invalid"
                    )
                projection_scope["occurrences"] += 1
                projection_scope["actions"] += len(actions)
                for action in actions:
                    if not isinstance(action, Mapping):
                        raise ExtractionError(
                            "source-binding projection build action is invalid"
                        )
                    if action.get("cwd") is not None and not isinstance(
                        action.get("cwd"),
                        str,
                    ):
                        raise ExtractionError(
                            "source-binding projection action cwd is invalid"
                        )
                    source_inputs = action.get("source_inputs")
                    if not isinstance(source_inputs, list) or any(
                        not isinstance(item, str) or not item
                        for item in source_inputs
                    ):
                        raise ExtractionError(
                            "source-binding projection source_inputs is invalid"
                        )
                    declared_source_count = action.get("source_input_count")
                    if (
                        isinstance(declared_source_count, bool)
                        or not isinstance(declared_source_count, int)
                        or declared_source_count != len(source_inputs)
                    ):
                        raise ExtractionError(
                            "source-binding projection source_inputs are truncated"
                        )
                    stored_bindings = action.get("repository_source_bindings")
                    if not isinstance(stored_bindings, list) or any(
                        not isinstance(binding, Mapping)
                        for binding in stored_bindings
                    ):
                        raise ExtractionError(
                            "source-binding projection stored bindings are invalid"
                        )
                    declared_binding_count = action.get(
                        "repository_source_binding_count"
                    )
                    if (
                        isinstance(declared_binding_count, bool)
                        or not isinstance(declared_binding_count, int)
                        or declared_binding_count != len(stored_bindings)
                    ):
                        raise ExtractionError(
                            "source-binding projection stored bindings are truncated"
                        )
                    projection_scope["source_inputs"] += len(source_inputs)
                    projection_scope["old_bindings"] += len(stored_bindings)
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


def _verify_content_store_pack(
    connection: sqlite3.Connection,
    *,
    root: Path,
    pack_row: sqlite3.Row,
    receipt_record: Mapping[str, Any],
) -> None:
    filename = str(pack_row["filename"])
    committed_end = int(pack_row["committed_end"])
    expected_content_count = int(pack_row["content_count"])
    path = root / filename
    try:
        handle, before = _open_regular_no_follow(path)
    except (OSError, ExtractionError) as exc:
        raise ExtractionError(
            f"content-store pack verification failed: {filename}"
        ) from exc
    pack_digest = hashlib.sha256()
    try:
        with handle:
            if before.st_size != committed_end:
                raise ExtractionError(
                    f"content-store pack size differs: {filename}"
                )
            magic = handle.read(len(_CONTENT_PACK_MAGIC))
            pack_digest.update(magic)
            if magic != _CONTENT_PACK_MAGIC:
                raise ExtractionError(
                    f"content-store pack magic differs: {filename}"
                )
            expected_offset = len(_CONTENT_PACK_MAGIC)
            content_count = 0
            for content_row in connection.execute(
                """
                SELECT sha256, raw_size, offset, frame_size, compressed_size
                FROM contents
                WHERE pack_id = ?
                ORDER BY offset
                """,
                (int(pack_row["pack_id"]),),
            ):
                digest = _require_hex64(
                    content_row["sha256"],
                    where=f"content-store pack {filename} content SHA-256",
                )
                raw_size = int(content_row["raw_size"])
                offset = int(content_row["offset"])
                frame_size = int(content_row["frame_size"])
                compressed_size = int(content_row["compressed_size"])
                if (
                    raw_size < 0
                    or compressed_size < 0
                    or offset != expected_offset
                    or frame_size != _CONTENT_FRAME_HEADER.size + compressed_size
                    or frame_size > committed_end - offset
                ):
                    raise ExtractionError(
                        f"content-store frame metadata differs: {digest}"
                    )
                header = handle.read(_CONTENT_FRAME_HEADER.size)
                pack_digest.update(header)
                if len(header) != _CONTENT_FRAME_HEADER.size:
                    raise ExtractionError(
                        f"content-store frame header is truncated: {digest}"
                    )
                (
                    frame_magic,
                    frame_digest,
                    header_raw_size,
                    header_compressed_size,
                ) = _CONTENT_FRAME_HEADER.unpack(header)
                if (
                    frame_magic != _CONTENT_FRAME_MAGIC
                    or frame_digest.hex() != digest
                    or header_raw_size != raw_size
                    or header_compressed_size != compressed_size
                ):
                    raise ExtractionError(
                        f"content-store frame header differs: {digest}"
                    )

                decompressor = zlib.decompressobj()
                decoder = codecs.getincrementaldecoder("utf-8")(
                    errors="strict"
                )
                content_digest = hashlib.sha256()
                decompressed_size = 0
                compressed_remaining = compressed_size
                while compressed_remaining:
                    block = handle.read(
                        min(1024 * 1024, compressed_remaining)
                    )
                    if not block:
                        raise ExtractionError(
                            f"content-store frame payload is truncated: {digest}"
                        )
                    pack_digest.update(block)
                    compressed_remaining -= len(block)
                    pending = block
                    while pending:
                        budget = min(
                            1024 * 1024,
                            raw_size - decompressed_size + 1,
                        )
                        if budget < 1:
                            raise ExtractionError(
                                f"content-store frame expands past raw size: {digest}"
                            )
                        previous_pending_size = len(pending)
                        output = decompressor.decompress(pending, budget)
                        pending = decompressor.unconsumed_tail
                        if output:
                            decompressed_size += len(output)
                            if decompressed_size > raw_size:
                                raise ExtractionError(
                                    "content-store frame expands past raw size: "
                                    f"{digest}"
                                )
                            content_digest.update(output)
                            decoder.decode(output, final=False)
                        if (
                            pending
                            and not output
                            and len(pending) >= previous_pending_size
                        ):
                            raise ExtractionError(
                                f"content-store zlib stream made no progress: {digest}"
                            )
                    if decompressor.eof and compressed_remaining:
                        raise ExtractionError(
                            f"content-store frame has trailing zlib data: {digest}"
                        )

                while not decompressor.eof:
                    budget = min(
                        1024 * 1024,
                        raw_size - decompressed_size + 1,
                    )
                    if budget < 1:
                        raise ExtractionError(
                            f"content-store frame expands past raw size: {digest}"
                        )
                    output = decompressor.decompress(b"", budget)
                    if not output:
                        break
                    decompressed_size += len(output)
                    if decompressed_size > raw_size:
                        raise ExtractionError(
                            f"content-store frame expands past raw size: {digest}"
                        )
                    content_digest.update(output)
                    decoder.decode(output, final=False)
                decoder.decode(b"", final=True)
                if (
                    not decompressor.eof
                    or decompressor.unused_data
                    or decompressor.unconsumed_tail
                    or decompressed_size != raw_size
                    or content_digest.hexdigest() != digest
                ):
                    raise ExtractionError(
                        f"content-store frame verification failed: {digest}"
                    )
                expected_offset += frame_size
                content_count += 1
            after = os.fstat(handle.fileno())
    except (UnicodeError, zlib.error) as exc:
        raise ExtractionError(
            f"content-store frame encoding is invalid: {filename}"
        ) from exc
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(before, field) != getattr(after, field)
        for field in identity_fields
    ):
        raise ExtractionError(
            f"content-store pack changed while verifying: {filename}"
        )
    if (
        expected_offset != committed_end
        or content_count != expected_content_count
        or pack_digest.hexdigest() != receipt_record.get("sha256")
    ):
        raise ExtractionError(
            f"content-store pack receipt differs: {filename}"
        )


def _verify_content_store_relations(
    connection: sqlite3.Connection,
) -> None:
    for sequence in connection.execute(
        """
        SELECT token_sequence_sha256, token_count, tokenizer_fingerprint
        FROM token_sequences
        ORDER BY token_sequence_sha256
        """
    ):
        _require_hex64(
            sequence["token_sequence_sha256"],
            where="content-store token-sequence SHA-256",
        )
        if (
            int(sequence["token_count"]) < 0
            or not str(sequence["tokenizer_fingerprint"])
        ):
            raise ExtractionError(
                "content-store token-sequence metadata is invalid"
            )
    aggregate = connection.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM contents) AS unique_content_count,
            (SELECT COALESCE(SUM(raw_size), 0) FROM contents)
                AS unique_bytes,
            (SELECT COUNT(*) FROM occurrences) AS occurrence_count,
            (
                SELECT COALESCE(SUM(contents.raw_size), 0)
                FROM occurrences
                JOIN contents
                  ON contents.sha256 = occurrences.content_sha256
            ) AS raw_occurrence_bytes,
            (
                SELECT COUNT(*) FROM contents
                WHERE token_sequence_sha256 IS NOT NULL
            ) AS tokenized_unique_content_count,
            (SELECT COUNT(*) FROM token_sequences)
                AS unique_token_sequence_count,
            (SELECT COALESCE(SUM(token_count), 0) FROM token_sequences)
                AS exact_unique_payload_tokens
        """
    ).fetchone()
    stats = connection.execute(
        "SELECT * FROM stats WHERE singleton = 1"
    ).fetchone()
    if aggregate is None or stats is None:
        raise ExtractionError(
            "content-store aggregate or stats row is missing"
        )
    expected_stats = {
        "raw_occurrence_bytes": int(aggregate["raw_occurrence_bytes"]),
        "unique_bytes": int(aggregate["unique_bytes"]),
        "duplicate_bytes": (
            int(aggregate["raw_occurrence_bytes"])
            - int(aggregate["unique_bytes"])
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
    if expected_stats["duplicate_bytes"] < 0:
        raise ExtractionError(
            "content-store occurrence bytes are below unique bytes"
        )
    for field, expected in expected_stats.items():
        if int(stats[field]) != expected:
            raise ExtractionError(
                f"content-store counter mismatch for {field}"
            )

    fingerprints = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT DISTINCT tokenizer_fingerprint
            FROM contents
            WHERE tokenizer_fingerprint IS NOT NULL
            ORDER BY tokenizer_fingerprint
            LIMIT 2
            """
        )
    ]
    if len(fingerprints) > 1:
        raise ExtractionError(
            "content-store contents use multiple tokenizer fingerprints"
        )
    expected_fingerprint = fingerprints[0] if fingerprints else None
    if stats["tokenizer_fingerprint"] != expected_fingerprint:
        raise ExtractionError(
            "content-store stats tokenizer fingerprint differs"
        )
    if (
        connection.execute(
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
        is not None
    ):
        raise ExtractionError(
            "content-store content token metadata differs"
        )
    if (
        connection.execute(
            """
            SELECT token_sequence_sha256
            FROM token_sequences
            WHERE NOT EXISTS (
                SELECT 1 FROM contents
                WHERE contents.token_sequence_sha256 =
                      token_sequences.token_sequence_sha256
            )
            LIMIT 1
            """
        ).fetchone()
        is not None
    ):
        raise ExtractionError(
            "content-store has an unreferenced token sequence"
        )


def _content_store_policy_max_pack_bytes(
    policy: Mapping[str, Any],
) -> int:
    if set(policy) != {
        "compression",
        "content_digest",
        "content_encoding",
        "frame_schema",
        "max_pack_bytes",
        "occurrence_key",
        "pack_commit_protocol",
        "provenance_storage",
        "token_count_semantics",
        "token_sequence_encoding",
    }:
        raise ExtractionError("content-store policy shape is stale")
    compression = policy.get("compression")
    if not isinstance(compression, Mapping):
        raise ExtractionError("content-store compression policy is invalid")
    compression_level = compression.get("level")
    max_pack_bytes = policy.get("max_pack_bytes")
    minimum_pack_bytes = (
        len(_CONTENT_PACK_MAGIC) + _CONTENT_FRAME_HEADER.size + 8
    )
    if (
        set(compression) != {"algorithm", "level"}
        or compression.get("algorithm") != "zlib"
        or isinstance(compression_level, bool)
        or not isinstance(compression_level, int)
        or not 0 <= compression_level <= 9
        or isinstance(max_pack_bytes, bool)
        or not isinstance(max_pack_bytes, int)
        or max_pack_bytes < minimum_pack_bytes
        or policy.get("content_digest") != "sha256"
        or policy.get("content_encoding") != "utf-8-strict"
        or policy.get("frame_schema") != CONTENT_STORE_PACK_SCHEMA
        or policy.get("occurrence_key") != list(_OCCURRENCE_FIELDS)
        or policy.get("pack_commit_protocol")
        != "fsync-pack-then-sqlite-full-commit"
        or policy.get("token_count_semantics")
        != "exact-canonical-payload-only-no-framing"
        or policy.get("token_sequence_encoding") != TOKEN_SEQUENCE_ENCODING
        or policy.get("provenance_storage")
        != {
            "canonical_encoding": "json-sort-keys-utf8-v1",
            "compression": "zlib",
            "compression_level": compression_level,
            "digest": "sha256",
        }
    ):
        raise ExtractionError("content-store policy is unsupported")
    return max_pack_bytes


def _content_store_counters(connection: sqlite3.Connection) -> dict[str, object]:
    row = connection.execute("SELECT * FROM stats WHERE singleton = 1").fetchone()
    if row is None:
        raise ExtractionError("content-store stats row is missing")
    unique_count = int(row["unique_content_count"])
    tokenized_count = int(row["tokenized_unique_content_count"])
    fingerprint = row["tokenizer_fingerprint"]
    all_tokenized = (
        unique_count > 0
        and tokenized_count == unique_count
        and fingerprint is not None
    )
    return {
        "raw_occurrence_bytes": int(row["raw_occurrence_bytes"]),
        "unique_bytes": int(row["unique_bytes"]),
        "duplicate_bytes": int(row["duplicate_bytes"]),
        "unique_content_count": unique_count,
        "occurrence_count": int(row["occurrence_count"]),
        "tokenized_unique_content_count": tokenized_count,
        "unique_token_sequence_count": int(row["unique_token_sequence_count"]),
        "tokenizer_fingerprint": fingerprint,
        "exact_unique_payload_tokens": (
            int(row["exact_unique_payload_tokens"]) if all_tokenized else None
        ),
    }


def _verify_content_store_recovery(
    root: Path,
    recovery: Mapping[str, Any],
) -> None:
    if set(recovery) != {
        "quarantined_orphan_count",
        "records_sha256",
        "records",
    }:
        raise ExtractionError("content-store recovery receipt differs")
    records = recovery.get("records")
    if (
        not isinstance(records, list)
        or len(records) > MAX_RECOVERY_RECORDS
        or recovery.get("quarantined_orphan_count") != len(records)
        or recovery.get("records_sha256")
        != _hash_records("cppmega-ci-recovery-records-v1", iter(records))
    ):
        raise ExtractionError("content-store recovery receipt differs")
    quarantine = root / "orphaned"
    try:
        quarantine_metadata = quarantine.lstat()
    except FileNotFoundError:
        if records:
            raise ExtractionError(
                "content-store recovery artifacts are missing"
            )
        return
    if stat.S_ISLNK(quarantine_metadata.st_mode) or not stat.S_ISDIR(
        quarantine_metadata.st_mode
    ):
        raise ExtractionError("content-store recovery directory is unsafe")

    allowed_names: set[str] = set()
    previous_metadata_name: str | None = None
    required_record_keys = {
        "schema",
        "kind",
        "reason",
        "original_filename",
        "source_offset",
        "byte_size",
        "sha256",
        "quarantined_filename",
    }
    for index, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or set(record) != required_record_keys
            or record.get("schema") != CONTENT_STORE_RECOVERY_SCHEMA
            or record.get("kind") not in {"whole-pack", "pack-tail"}
        ):
            raise ExtractionError(
                f"content-store recovery record {index} is invalid"
            )
        reason = record.get("reason")
        original = record.get("original_filename")
        source_offset = record.get("source_offset")
        byte_size = record.get("byte_size")
        filename = record.get("quarantined_filename")
        digest = record.get("sha256")
        if (
            not isinstance(reason, str)
            or not reason
            or not isinstance(original, str)
            or Path(original).name != original
            or isinstance(source_offset, bool)
            or not isinstance(source_offset, int)
            or source_offset < 0
            or isinstance(byte_size, bool)
            or not isinstance(byte_size, int)
            or byte_size < 0
            or not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ExtractionError(
                f"content-store recovery record {index} has invalid fields"
            )
        _require_hex64(
            digest,
            where=f"content-store recovery record {index} SHA-256",
        )
        metadata_name = f"{filename}.recovery.json"
        if (
            previous_metadata_name is not None
            and metadata_name <= previous_metadata_name
        ):
            raise ExtractionError(
                "content-store recovery records are not sorted and unique"
            )
        previous_metadata_name = metadata_name
        metadata_value, metadata_raw = _read_json_object(
            quarantine / metadata_name,
            where=f"content-store recovery metadata {metadata_name}",
        )
        if (
            metadata_value != dict(record)
            or metadata_raw != _canonical_json_bytes(record) + b"\n"
        ):
            raise ExtractionError(
                f"content-store recovery metadata differs: {metadata_name}"
            )
        try:
            artifact_sha = _stable_file_sha256(
                quarantine / filename,
                expected_size=byte_size,
            )
        except (OSError, ExtractionError) as exc:
            raise ExtractionError(
                f"content-store recovery artifact differs: {filename}"
            ) from exc
        if artifact_sha != digest:
            raise ExtractionError(
                f"content-store recovery artifact differs: {filename}"
            )
        allowed_names.update({filename, metadata_name})
    for artifact in quarantine.iterdir():
        if artifact.name.startswith("."):
            continue
        if artifact.name not in allowed_names:
            raise ExtractionError(
                "content-store recovery directory has unmanifested artifacts"
            )


def _verify_frozen_content_store(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    snapshot_db: Path,
) -> tuple[sqlite3.Connection, dict[str, int]]:
    required_receipt_keys = {
        "schema",
        "status",
        "store_schema",
        "policy",
        "policy_sha256",
        "script_sha256",
        "sqlite_schema_sha256",
        "target_exact_unique_payload_tokens",
        "exact_unique_payload_tokens",
        "counters",
        "logical_content_set_sha256",
        "logical_token_sequence_set_sha256",
        "occurrence_set_sha256",
        "pack_hashes",
        "sqlite_logical_sha256",
        "recovery",
        "verification",
    }
    if set(receipt) not in (
        required_receipt_keys,
        required_receipt_keys | {"emitted_valid_training_tokens"},
    ):
        raise ExtractionError("content-store receipt shape is incomplete or stale")
    if (
        receipt.get("schema") != CONTENT_STORE_RECEIPT_SCHEMA
        or receipt.get("store_schema") != CONTENT_STORE_SCHEMA
    ):
        raise ExtractionError("content-store receipt schema is missing or stale")
    if receipt.get("status") != "complete":
        raise ExtractionError("content-store receipt is not complete")
    verification = receipt.get("verification")
    if (
        not isinstance(verification, Mapping)
        or verification.get("ok") is not True
        or verification.get("mode") != "full"
    ):
        raise ExtractionError("content-store receipt lacks full verification")
    expected_occurrence_set = _require_hex64(
        receipt.get("occurrence_set_sha256"),
        where="content-store receipt occurrence_set_sha256",
    )
    expected_script = _require_hex64(
        receipt.get("script_sha256"),
        where="content-store receipt script_sha256",
    )
    expected_policy = _require_hex64(
        receipt.get("policy_sha256"),
        where="content-store receipt policy_sha256",
    )
    expected_sqlite_schema = _require_hex64(
        receipt.get("sqlite_schema_sha256"),
        where="content-store receipt sqlite_schema_sha256",
    )
    expected_sqlite_logical = _require_hex64(
        receipt.get("sqlite_logical_sha256"),
        where="content-store receipt sqlite_logical_sha256",
    )
    expected_content_set = _require_hex64(
        receipt.get("logical_content_set_sha256"),
        where="content-store receipt logical_content_set_sha256",
    )
    expected_token_set = _require_hex64(
        receipt.get("logical_token_sequence_set_sha256"),
        where="content-store receipt logical_token_sequence_set_sha256",
    )
    db_path = root / _SQLITE_NAME
    if root.is_symlink() or not root.is_dir():
        raise ExtractionError("content-store root is not a safe directory")
    if not db_path.is_file() or db_path.is_symlink():
        raise ExtractionError("frozen content-store SQLite file is missing")
    for suffix in ("-wal", "-journal"):
        if Path(f"{db_path}{suffix}").exists():
            raise ExtractionError(
                f"frozen content store has a mutable SQLite {suffix} file"
            )

    pack_receipts = receipt.get("pack_hashes")
    if (
        not isinstance(pack_receipts, list)
        or len(pack_receipts) > MAX_PACK_RECORDS
    ):
        raise ExtractionError("content-store receipt pack_hashes is invalid")
    receipt_names: set[str] = set()
    for index, record in enumerate(pack_receipts):
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {"filename", "committed_end", "content_count", "sha256"}
        ):
            raise ExtractionError(f"pack_hashes[{index}] is not an object")
        filename = record.get("filename")
        committed_end = record.get("committed_end")
        content_count = record.get("content_count")
        digest = record.get("sha256")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or isinstance(committed_end, bool)
            or not isinstance(committed_end, int)
            or committed_end < 0
            or isinstance(content_count, bool)
            or not isinstance(content_count, int)
            or content_count < 0
        ):
            raise ExtractionError(f"pack_hashes[{index}] has invalid metadata")
        _require_hex64(digest, where=f"pack_hashes[{index}].sha256")
        if filename in receipt_names:
            raise ExtractionError(f"content-store pack verification failed: {filename}")
        receipt_names.add(filename)
    actual_pack_count = 0
    for path in root.glob("pack-*.cicp"):
        actual_pack_count += 1
        if path.name not in receipt_names:
            raise ExtractionError("content-store pack set differs from its receipt")
    if actual_pack_count != len(receipt_names):
        raise ExtractionError("content-store pack set differs from its receipt")
    _copy_stable_sqlite(db_path, snapshot_db)
    uri = f"file:{quote(str(snapshot_db.resolve()), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA temp_store = FILE")
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or str(integrity[0]) != "ok":
            message = None if integrity is None else str(integrity[0])
            raise ExtractionError(
                f"content-store integrity_check failed: {message}"
            )
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ExtractionError("content-store foreign_key_check failed")
        settings: dict[str, str] = {}
        for row in connection.execute(
            "SELECT key, value FROM settings ORDER BY key"
        ):
            key = str(row["key"])
            if key in settings or len(settings) >= 4:
                raise ExtractionError(
                    "content-store SQLite settings are incomplete or stale"
                )
            settings[key] = str(row["value"])
        if (
            set(settings)
            != {
                "schema",
                "policy",
                "creator_script_sha256",
                "sqlite_schema_sha256",
            }
            or settings.get("schema") != CONTENT_STORE_SCHEMA
        ):
            raise ExtractionError("content-store SQLite schema is missing or stale")
        if expected_script != settings.get("creator_script_sha256"):
            raise ExtractionError(
                "content-store script SHA-256 differs from durable settings"
            )
        if (
            receipt.get("sqlite_schema_sha256")
            != settings.get("sqlite_schema_sha256")
        ):
            raise ExtractionError(
                "content-store SQLite schema SHA-256 differs from durable settings"
            )
        policy = receipt.get("policy")
        try:
            durable_policy = json.loads(settings["policy"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ExtractionError(
                "content-store durable policy is invalid"
            ) from exc
        if (
            not isinstance(policy, Mapping)
            or not isinstance(durable_policy, Mapping)
            or dict(policy) != durable_policy
            or _sha256_bytes(_canonical_json_bytes(policy)) != expected_policy
        ):
            raise ExtractionError("content-store policy digest differs")
        max_pack_bytes = _content_store_policy_max_pack_bytes(
            durable_policy
        )
        if (
            connection.execute(
                "SELECT 1 FROM packs WHERE committed_end > ? LIMIT 1",
                (max_pack_bytes,),
            ).fetchone()
            is not None
        ):
            raise ExtractionError(
                "content-store pack exceeds its durable policy"
            )
        sqlite_packs = connection.execute(
            """
            SELECT pack_id, filename, committed_end, content_count
            FROM packs ORDER BY pack_id
            """
        )
        for index, row in enumerate(sqlite_packs):
            if index >= len(pack_receipts):
                raise ExtractionError("content-store receipt omits an indexed pack")
            record = pack_receipts[index]
            assert isinstance(record, Mapping)
            if (
                record.get("filename") != str(row["filename"])
                or record.get("committed_end") != int(row["committed_end"])
                or record.get("content_count") != int(row["content_count"])
            ):
                raise ExtractionError(
                    "content-store pack receipt differs from SQLite"
                )
            _verify_content_store_pack(
                connection,
                root=root,
                pack_row=row,
                receipt_record=record,
            )
        sqlite_pack_count = int(
            connection.execute("SELECT COUNT(*) FROM packs").fetchone()[0]
        )
        if sqlite_pack_count != len(pack_receipts):
            raise ExtractionError("content-store receipt has extra pack records")
        _verify_content_store_relations(connection)
        actual_schema = _content_store_sqlite_schema_sha256(connection)
        if actual_schema != expected_sqlite_schema:
            raise ExtractionError("content-store SQLite schema SHA-256 differs")
        if _content_store_sqlite_logical_sha256(connection) != expected_sqlite_logical:
            raise ExtractionError("content-store SQLite logical SHA-256 differs")
        projection_scope: Counter[str] = Counter()
        if (
            _content_store_occurrence_set_sha256(
                connection,
                projection_scope=projection_scope,
            )
            != expected_occurrence_set
        ):
            raise ExtractionError("content-store occurrence set differs")
        if _content_set_digest(connection) != expected_content_set:
            raise ExtractionError("content-store logical content set differs")
        if _token_sequence_set_digest(connection) != expected_token_set:
            raise ExtractionError("content-store logical token sequence set differs")
        counters = receipt.get("counters")
        actual_counters = _content_store_counters(connection)
        if not isinstance(counters, Mapping) or dict(counters) != actual_counters:
            raise ExtractionError("content-store counters differ from SQLite")
        target_tokens = receipt.get("target_exact_unique_payload_tokens")
        exact_tokens = receipt.get("exact_unique_payload_tokens")
        if (
            isinstance(target_tokens, bool)
            or not isinstance(target_tokens, int)
            or target_tokens < 0
            or isinstance(exact_tokens, bool)
            or not isinstance(exact_tokens, int)
            or exact_tokens < target_tokens
            or exact_tokens != actual_counters["exact_unique_payload_tokens"]
        ):
            raise ExtractionError("content-store completion threshold differs")
        if "emitted_valid_training_tokens" in receipt:
            emitted_tokens = receipt["emitted_valid_training_tokens"]
            if (
                isinstance(emitted_tokens, bool)
                or not isinstance(emitted_tokens, int)
                or emitted_tokens < 0
            ):
                raise ExtractionError(
                    "content-store emitted training token count is invalid"
                )
        recovery = receipt.get("recovery")
        if not isinstance(recovery, Mapping):
            raise ExtractionError("content-store recovery receipt is missing")
        _verify_content_store_recovery(root, recovery)
    except BaseException:
        connection.close()
        raise
    return connection, {
        "occurrence_count": int(projection_scope["occurrences"]),
        "action_count": int(projection_scope["actions"]),
        "source_input_count": int(projection_scope["source_inputs"]),
        "old_binding_count": int(projection_scope["old_bindings"]),
    }


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


def _iter_canonical_jsonl(
    path: Path,
    *,
    where: str,
    allow_empty: bool = False,
) -> Iterator[tuple[dict[str, Any], bytes]]:
    handle, _metadata = _open_regular_no_follow(path)
    with handle:
        line_number = 0
        while True:
            raw = handle.readline(MAX_JSONL_RECORD_BYTES + 1)
            if not raw:
                break
            line_number += 1
            if len(raw) > MAX_JSONL_RECORD_BYTES:
                raise ExtractionError(
                    f"{where} line {line_number} exceeds the record size limit"
                )
            if not raw.endswith(b"\n") or raw == b"\n":
                raise ExtractionError(
                    f"{where} line {line_number} is not final-newline canonical JSONL"
                )
            try:
                value = json.loads(raw[:-1].decode("utf-8", errors="strict"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ExtractionError(
                    f"{where} line {line_number} is invalid JSON"
                ) from exc
            if (
                not isinstance(value, dict)
                or _canonical_json_bytes(value) + b"\n" != raw
            ):
                raise ExtractionError(
                    f"{where} line {line_number} is not canonical JSON"
                )
            yield value, raw
        if line_number == 0 and not allow_empty:
            raise ExtractionError(f"{where} is empty")


def _require_occurrence_key(value: object, *, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(_OCCURRENCE_FIELDS):
        raise ExtractionError(
            f"{where} must contain exactly {list(_OCCURRENCE_FIELDS)}"
        )
    output: dict[str, object] = {}
    for field in _OCCURRENCE_FIELDS[:-1]:
        item = value[field]
        if not isinstance(item, str) or not item:
            raise ExtractionError(f"{where}.{field} must be non-empty")
        output[field] = item
    ordinal = value["chunk_ordinal"]
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ExtractionError(f"{where}.chunk_ordinal is invalid")
    output["chunk_ordinal"] = ordinal
    return output


def _prepare_representatives(
    connection: sqlite3.Connection,
    *,
    ledger_path: Path,
    export_receipt: Mapping[str, Any],
) -> dict[str, object]:
    representatives = export_receipt.get("representatives")
    if (
        export_receipt.get("schema") != CASE5_EXPORT_SCHEMA
        or export_receipt.get("status") != "complete"
        or not isinstance(representatives, Mapping)
        or representatives.get("schema") != REPRESENTATIVE_LEDGER_SCHEMA
        or representatives.get("selection") != _REPRESENTATIVE_SELECTION
        or representatives.get("ledger_artifact") != ledger_path.name
    ):
        raise ExtractionError("CASE5 representative receipt is missing or stale")
    expected_count = representatives.get("count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        raise ExtractionError("CASE5 representative count is invalid")
    expected_logical = _require_hex64(
        representatives.get("ledger_sha256"),
        where="CASE5 representative ledger_sha256",
    )
    expected_artifact = _require_hex64(
        representatives.get("ledger_artifact_sha256"),
        where="CASE5 representative ledger_artifact_sha256",
    )
    artifacts = export_receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise ExtractionError("CASE5 export artifacts are invalid")
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == "representative_ledger"
    ]
    if len(matches) != 1:
        raise ExtractionError("CASE5 export must bind one representative ledger")
    artifact = matches[0]
    expected_artifact_size = artifact.get("byte_size")
    if (
        artifact.get("path") != ledger_path.name
        or artifact.get("rows") != expected_count
        or isinstance(expected_artifact_size, bool)
        or not isinstance(expected_artifact_size, int)
        or expected_artifact_size < 0
        or artifact.get("sha256") != expected_artifact
    ):
        raise ExtractionError("CASE5 representative artifact metadata differs")

    connection.execute(
        """
        CREATE TEMP TABLE selected_representatives(
            token_sequence_sha256 TEXT PRIMARY KEY,
            token_count INTEGER NOT NULL,
            representative_content_sha256 TEXT NOT NULL,
            repo TEXT NOT NULL,
            run_attempt TEXT NOT NULL,
            job TEXT NOT NULL,
            step TEXT NOT NULL,
            chunk_ordinal INTEGER NOT NULL,
            representative_provenance_sha256 TEXT NOT NULL,
            record_json TEXT NOT NULL
        )
        """
    )
    required_keys = {
        "schema",
        "token_sequence_sha256",
        "token_count",
        "candidate_content_count",
        "candidate_occurrence_count",
        "candidate_content_sha256_sequence_sha256",
        "representative_content_sha256",
        "representative_occurrence_key",
        "representative_provenance_sha256",
    }
    logical = _RecordHasher("cppmega-ci-case5-representative-ledger-v1")
    physical = hashlib.sha256()
    physical_size = 0
    previous_sequence: str | None = None
    for index, (record, raw) in enumerate(
        _iter_canonical_jsonl(
            ledger_path,
            where="CASE5 representative ledger",
            allow_empty=expected_count == 0,
        )
    ):
        physical.update(raw)
        physical_size += len(raw)
        if set(record) != required_keys or record.get("schema") != (
            REPRESENTATIVE_LEDGER_SCHEMA
        ):
            raise ExtractionError(f"representative ledger record {index} is invalid")
        sequence = _require_hex64(
            record.get("token_sequence_sha256"),
            where=f"representative ledger record {index} token sequence",
        )
        content_sha = _require_hex64(
            record.get("representative_content_sha256"),
            where=f"representative ledger record {index} content",
        )
        provenance_sha = _require_hex64(
            record.get("representative_provenance_sha256"),
            where=f"representative ledger record {index} provenance",
        )
        _require_hex64(
            record.get("candidate_content_sha256_sequence_sha256"),
            where=f"representative ledger record {index} candidate sequence",
        )
        for name in (
            "token_count",
            "candidate_content_count",
            "candidate_occurrence_count",
        ):
            number = record.get(name)
            minimum = 0 if name == "token_count" else 1
            if (
                isinstance(number, bool)
                or not isinstance(number, int)
                or number < minimum
            ):
                raise ExtractionError(
                    f"representative ledger record {index} {name} is invalid"
                )
        if previous_sequence is not None and sequence <= previous_sequence:
            raise ExtractionError(
                "representative ledger token sequences are not sorted and unique"
            )
        previous_sequence = sequence
        key = _require_occurrence_key(
            record.get("representative_occurrence_key"),
            where=f"representative ledger record {index} occurrence key",
        )
        logical.update(record)
        connection.execute(
            """
            INSERT INTO selected_representatives(
                token_sequence_sha256, token_count,
                representative_content_sha256,
                repo, run_attempt, job, step, chunk_ordinal,
                representative_provenance_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sequence,
                int(record["token_count"]),
                content_sha,
                key["repo"],
                key["run_attempt"],
                key["job"],
                key["step"],
                key["chunk_ordinal"],
                provenance_sha,
                _canonical_json(record),
            ),
        )
    if (
        logical.count != expected_count
        or logical.hexdigest != expected_logical
        or physical_size != expected_artifact_size
        or physical.hexdigest() != expected_artifact
    ):
        raise ExtractionError("CASE5 representative ledger receipt differs")
    return {
        "representative_count": expected_count,
        "representative_ledger_sha256": expected_logical,
        "representative_ledger_artifact_sha256": expected_artifact,
    }


def _validate_projection_binding(
    value: object,
    *,
    where: str,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ExtractionError(f"{where} must be a binding object or null")
    _repository_name(value.get("repository"), where=f"{where}.repository")
    _require_git_oid(value.get("head_sha"), where=f"{where}.head_sha")
    source_path = value.get("source_path")
    confidence = value.get("confidence")
    if (
        not isinstance(source_path, str)
        or not source_path
        or not isinstance(confidence, Mapping)
    ):
        raise ExtractionError(f"{where} fields are invalid")
    return value


def _validate_projection_record(
    record: Mapping[str, Any],
    *,
    index: int,
    section: Mapping[str, Any],
) -> tuple[str, str, str, str, int, int, int]:
    required = {
        "schema",
        "mode",
        "input_parser_sha256",
        "target_parser_sha256",
        "occurrence_key",
        "provenance_sha256",
        "action_index",
        "source_index",
        "source_input",
        "source_input_sha256",
        "cwd",
        "cwd_sha256",
        "action_sha256",
        "old_binding",
        "projected_binding",
        "change_kind",
        "reason",
    }
    where = f"source-binding projection record {index}"
    if set(record) != required or record.get("schema") != (
        SOURCE_BINDING_PROJECTION_SCHEMA
    ):
        raise ExtractionError(f"{where} shape is invalid")
    try:
        key = projection_record_key(record)
    except SourceBindingProjectionError as exc:
        raise ExtractionError(f"{where} is invalid: {exc}") from exc
    if (
        record.get("mode") != section["mode"]
        or record.get("input_parser_sha256")
        != section["input_parser_script_sha256"]
        or record.get("target_parser_sha256")
        != section["target_parser_script_sha256"]
    ):
        raise ExtractionError(f"{where} parser lineage differs")
    _require_hex64(
        record.get("provenance_sha256"),
        where=f"{where}.provenance_sha256",
    )
    source_input = record.get("source_input")
    if not isinstance(source_input, str):
        raise ExtractionError(f"{where}.source_input is invalid")
    if record.get("source_input_sha256") != _sha256_bytes(
        source_input.encode("utf-8")
    ):
        raise ExtractionError(f"{where}.source_input_sha256 differs")
    cwd = record.get("cwd")
    cwd_sha256 = record.get("cwd_sha256")
    if cwd is None:
        if cwd_sha256 is not None:
            raise ExtractionError(f"{where}.cwd_sha256 must be null")
    elif not isinstance(cwd, str) or cwd_sha256 != _sha256_bytes(
        cwd.encode("utf-8")
    ):
        raise ExtractionError(f"{where}.cwd binding differs")
    _require_hex64(record.get("action_sha256"), where=f"{where}.action_sha256")
    old_binding = _validate_projection_binding(
        record.get("old_binding"),
        where=f"{where}.old_binding",
    )
    projected_binding = _validate_projection_binding(
        record.get("projected_binding"),
        where=f"{where}.projected_binding",
    )
    change_kind = record.get("change_kind")
    reason = record.get("reason")
    if (
        not isinstance(change_kind, str)
        or not change_kind
        or (reason is not None and (not isinstance(reason, str) or not reason))
        or (projected_binding is None and reason is None)
        or (change_kind == "unchanged" and old_binding != projected_binding)
    ):
        raise ExtractionError(f"{where} change evidence is invalid")
    return key


def _prepare_source_binding_projection(
    connection: sqlite3.Connection,
    *,
    ledger_path: Path,
    export_receipt: Mapping[str, Any],
    content_receipt: Mapping[str, Any],
    frozen_fetch_state: Mapping[str, Any],
    projection_scope: Mapping[str, int],
) -> tuple[dict[str, object], SourceBindingProjector]:
    section = export_receipt.get("source_binding_projection")
    required_section = {
        "schema",
        "mode",
        "projection_script_sha256",
        "input_parser_script_sha256",
        "target_parser_script_sha256",
        "input_occurrence_set_sha256",
        "input_fetch_state_sqlite_logical_sha256",
        "input_fetch_state_sidecar_set_sha256",
        "coverage",
        "change_counts",
        "ledger_artifact",
        "ledger_record_count",
        "ledger_sha256",
        "ledger_artifact_sha256",
        "claim_boundary",
    }
    if (
        not isinstance(section, Mapping)
        or set(section) != required_section
        or section.get("schema") != SOURCE_BINDING_PROJECTION_SCHEMA
        or section.get("mode") not in {"legacy_projection", "current_audit"}
        or section.get("ledger_artifact")
        != _SOURCE_BINDING_PROJECTION_ARTIFACT
        or ledger_path.name != _SOURCE_BINDING_PROJECTION_ARTIFACT
        or section.get("claim_boundary")
        != _SOURCE_BINDING_PROJECTION_CLAIM_BOUNDARY
    ):
        raise ExtractionError(
            "CASE5 source-binding projection receipt is missing or stale"
        )
    projection_script = _require_hex64(
        section.get("projection_script_sha256"),
        where="source-binding projection script SHA-256",
    )
    input_parser = _require_hex64(
        section.get("input_parser_script_sha256"),
        where="source-binding projection input parser SHA-256",
    )
    target_parser = _require_hex64(
        section.get("target_parser_script_sha256"),
        where="source-binding projection target parser SHA-256",
    )
    expected_logical = _require_hex64(
        section.get("ledger_sha256"),
        where="source-binding projection ledger SHA-256",
    )
    expected_artifact = _require_hex64(
        section.get("ledger_artifact_sha256"),
        where="source-binding projection artifact SHA-256",
    )
    if (
        projection_script != projection_script_sha256()
        or target_parser != target_parser_script_sha256()
    ):
        raise ExtractionError(
            "source-binding projection implementation lineage differs"
        )
    try:
        projector = SourceBindingProjector(
            input_parser,
            authorized_legacy_sha256=(
                input_parser
                if section["mode"] == "legacy_projection"
                else None
            ),
        )
    except SourceBindingProjectionError as exc:
        raise ExtractionError(
            f"source-binding projection parser lineage is unsupported: {exc}"
        ) from exc
    if (
        projector.mode != section["mode"]
        or projector.target_parser_sha256 != target_parser
        or projector.implementation_sha256 != projection_script
    ):
        raise ExtractionError(
            "source-binding projection descriptor differs from implementation"
        )
    settings = frozen_fetch_state.get("settings")
    if (
        not isinstance(settings, Mapping)
        or settings.get("parser_script_sha256") != input_parser
        or section.get("input_occurrence_set_sha256")
        != content_receipt.get("occurrence_set_sha256")
        or section.get("input_fetch_state_sqlite_logical_sha256")
        != frozen_fetch_state.get("sqlite_logical_sha256")
        or section.get("input_fetch_state_sidecar_set_sha256")
        != frozen_fetch_state.get("sidecar_set_sha256")
    ):
        raise ExtractionError(
            "source-binding projection input scope differs"
        )
    for name in (
        "input_occurrence_set_sha256",
        "input_fetch_state_sqlite_logical_sha256",
        "input_fetch_state_sidecar_set_sha256",
    ):
        _require_hex64(section.get(name), where=f"source-binding projection {name}")

    coverage = section.get("coverage")
    if (
        not isinstance(coverage, Mapping)
        or set(coverage)
        != {
            "order",
            "occurrence_count",
            "action_count",
            "source_input_count",
            "old_binding_count",
            "projected_binding_count",
        }
        or coverage.get("order") != _SOURCE_BINDING_PROJECTION_ORDER
        or any(
            isinstance(coverage.get(name), bool)
            or not isinstance(coverage.get(name), int)
            or int(coverage[name]) < 0
            for name in (
                "occurrence_count",
                "action_count",
                "source_input_count",
                "old_binding_count",
                "projected_binding_count",
            )
        )
        or any(
            int(coverage[name]) != int(projection_scope[name])
            for name in (
                "occurrence_count",
                "action_count",
                "source_input_count",
                "old_binding_count",
            )
        )
    ):
        raise ExtractionError(
            "source-binding projection coverage differs from the frozen store"
        )
    expected_count = section.get("ledger_record_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
        or expected_count != coverage["source_input_count"]
    ):
        raise ExtractionError(
            "source-binding projection ledger count differs from coverage"
        )
    expected_change_counts = section.get("change_counts")
    if not isinstance(expected_change_counts, Mapping) or any(
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for name, value in expected_change_counts.items()
    ):
        raise ExtractionError("source-binding projection change counts are invalid")

    artifacts = export_receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise ExtractionError("CASE5 export artifacts are invalid")
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == "source_binding_projection"
    ]
    if len(matches) != 1:
        raise ExtractionError(
            "CASE5 export must bind one source-binding projection artifact"
        )
    artifact = matches[0]
    artifact_size = artifact.get("byte_size")
    if (
        set(artifact) != {"path", "kind", "rows", "byte_size", "sha256"}
        or artifact.get("path") != _SOURCE_BINDING_PROJECTION_ARTIFACT
        or artifact.get("rows") != expected_count
        or isinstance(artifact_size, bool)
        or not isinstance(artifact_size, int)
        or artifact_size < 0
        or artifact.get("sha256") != expected_artifact
    ):
        raise ExtractionError(
            "CASE5 source-binding projection artifact metadata differs"
        )

    connection.executescript(
        """
        CREATE TEMP TABLE selected_source_binding_projection(
            repo TEXT NOT NULL,
            run_attempt TEXT NOT NULL,
            job TEXT NOT NULL,
            step TEXT NOT NULL,
            chunk_ordinal INTEGER NOT NULL,
            action_index INTEGER NOT NULL,
            source_index INTEGER NOT NULL,
            record_json TEXT NOT NULL,
            PRIMARY KEY(
                repo, run_attempt, job, step, chunk_ordinal,
                action_index, source_index
            )
        );
        """
    )
    selected_cursor = iter(
        connection.execute(
            """
            SELECT repo, run_attempt, job, step, chunk_ordinal
            FROM selected_representatives
            ORDER BY repo, run_attempt, job, step, chunk_ordinal
            """
        )
    )
    selected_row = next(selected_cursor, None)
    occurrence_cursor = iter(
        connection.execute(
            """
            SELECT repo, run_attempt, job, step, chunk_ordinal,
                   provenance_sha256, provenance_raw_size, provenance_zlib
            FROM occurrences
            ORDER BY repo, run_attempt, job, step, chunk_ordinal
            """
        )
    )
    occurrence_row = next(occurrence_cursor, None)

    def row_key(row: sqlite3.Row) -> tuple[str, str, str, str, int]:
        return (
            str(row["repo"]),
            str(row["run_attempt"]),
            str(row["job"]),
            str(row["step"]),
            int(row["chunk_ordinal"]),
        )

    logical = _RecordHasher(SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN)
    physical = hashlib.sha256()
    physical_size = 0
    previous_key: tuple[str, str, str, str, int, int, int] | None = None
    previous_record_occurrence: tuple[str, str, str, str, int] | None = None
    previous_record_action: tuple[str, str, str, str, int, int] | None = None
    record_occurrence_count = 0
    record_action_count = 0
    change_counts: Counter[str] = Counter()
    projected_binding_count = 0
    old_binding_count = 0
    decoded_occurrence_key: tuple[str, str, str, str, int] | None = None
    decoded_provenance: dict[str, Any] | None = None
    decoded_actions: list[Any] | None = None
    decoded_action_index: int | None = None
    decoded_action_projection: tuple[dict[str, object], ...] | None = None
    for index, (record, raw) in enumerate(
        _iter_canonical_jsonl(
            ledger_path,
            where="source-binding projection ledger",
            allow_empty=expected_count == 0,
        )
    ):
        key = _validate_projection_record(record, index=index, section=section)
        if previous_key is not None and key <= previous_key:
            raise ExtractionError(
                "source-binding projection records are not sorted and unique"
            )
        if (
            previous_key is None
            or key[:6] != previous_key[:6]
        ):
            if key[6] != 0:
                raise ExtractionError(
                    "source-binding projection source indexes are not contiguous"
                )
        elif key[6] != previous_key[6] + 1:
            raise ExtractionError(
                "source-binding projection source indexes are not contiguous"
            )
        previous_key = key
        occurrence_key = key[:5]
        while occurrence_row is not None and row_key(occurrence_row) < occurrence_key:
            occurrence_row = next(occurrence_cursor, None)
        if (
            occurrence_row is None
            or row_key(occurrence_row) != occurrence_key
            or str(occurrence_row["provenance_sha256"])
            != record["provenance_sha256"]
        ):
            raise ExtractionError(
                "source-binding projection record is outside the frozen occurrence set"
            )
        if decoded_occurrence_key != occurrence_key:
            decoded_occurrence_key = occurrence_key
            decoded_provenance = _decode_provenance(occurrence_row)
            chunk = decoded_provenance.get("chunk")
            training = (
                chunk.get("training_sidecars")
                if isinstance(chunk, Mapping)
                else None
            )
            decoded_actions = (
                training.get("build_actions")
                if isinstance(training, Mapping)
                else None
            )
            if not isinstance(decoded_actions, list):
                raise ExtractionError(
                    "source-binding projection occurrence actions are invalid"
                )
            decoded_action_index = None
            decoded_action_projection = None
        action_index = key[5]
        source_index = key[6]
        if decoded_provenance is None or decoded_actions is None:
            raise ExtractionError(
                "source-binding projection occurrence state is incomplete"
            )
        if action_index >= len(decoded_actions) or not isinstance(
            decoded_actions[action_index],
            Mapping,
        ):
            raise ExtractionError(
                "source-binding projection action index is outside provenance"
            )
        if decoded_action_index != action_index:
            try:
                action_projection = projector.project_action(
                    occurrence_key=record["occurrence_key"],
                    provenance_sha256=str(record["provenance_sha256"]),
                    provenance=decoded_provenance,
                    action=decoded_actions[action_index],
                    action_index=action_index,
                )
            except SourceBindingProjectionError as exc:
                raise ExtractionError(
                    f"source-binding projection cannot be recomputed: {exc}"
                ) from exc
            decoded_action_index = action_index
            decoded_action_projection = action_projection.records
        if decoded_action_projection is None:
            raise ExtractionError(
                "source-binding projection action state is incomplete"
            )
        if (
            source_index >= len(decoded_action_projection)
            or record != decoded_action_projection[source_index]
        ):
            raise ExtractionError(
                "source-binding projection record differs from provenance"
            )
        while selected_row is not None and row_key(selected_row) < occurrence_key:
            selected_row = next(selected_cursor, None)
        if selected_row is not None and row_key(selected_row) == occurrence_key:
            connection.execute(
                """
                INSERT INTO selected_source_binding_projection(
                    repo, run_attempt, job, step, chunk_ordinal,
                    action_index, source_index, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*key, _canonical_json(record)),
            )
        logical.update(record)
        physical.update(raw)
        physical_size += len(raw)
        change_counts[str(record["change_kind"])] += 1
        old_binding_count += record["old_binding"] is not None
        projected_binding_count += record["projected_binding"] is not None
        if occurrence_key != previous_record_occurrence:
            record_occurrence_count += 1
            previous_record_occurrence = occurrence_key
        if key[:6] != previous_record_action:
            record_action_count += 1
            previous_record_action = key[:6]
    if (
        logical.count != expected_count
        or logical.hexdigest != expected_logical
        or physical_size != artifact_size
        or physical.hexdigest() != expected_artifact
        or dict(sorted(change_counts.items()))
        != dict(sorted(expected_change_counts.items()))
        or old_binding_count != coverage["old_binding_count"]
        or projected_binding_count != coverage["projected_binding_count"]
        or record_occurrence_count > coverage["occurrence_count"]
        or record_action_count > coverage["action_count"]
    ):
        raise ExtractionError(
            "source-binding projection ledger receipt differs"
        )
    return (
        {
            "source_binding_projection_schema": SOURCE_BINDING_PROJECTION_SCHEMA,
            "source_binding_projection_mode": section["mode"],
            "source_binding_projection_script_sha256": projection_script,
            "source_binding_projection_input_parser_sha256": input_parser,
            "source_binding_projection_target_parser_sha256": target_parser,
            "source_binding_projection_ledger_record_count": expected_count,
            "source_binding_projection_ledger_sha256": expected_logical,
            "source_binding_projection_ledger_artifact_sha256": expected_artifact,
        },
        projector,
    )


def _repository_name(value: object, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or value.count("/") != 1
        or any(not component for component in value.split("/"))
    ):
        raise ExtractionError(f"{where} must be an exact owner/name")
    return value


def _runner_platform(provenance: Mapping[str, Any], cwd: str | None) -> str | None:
    job = provenance.get("job")
    labels: list[str] = []
    if isinstance(job, Mapping):
        raw_labels = job.get("labels", [])
        if isinstance(raw_labels, list) and all(
            isinstance(item, str) for item in raw_labels
        ):
            labels = [item.casefold() for item in raw_labels]
    windows = any(label.startswith("windows") for label in labels)
    posix = any(
        label.startswith(("ubuntu", "macos")) or label in {"linux", "darwin"}
        for label in labels
    )
    if windows and posix:
        return None
    if windows:
        return "windows"
    if posix:
        return "posix"
    return _path_platform(cwd)


def _checkout_binding(
    provenance: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    source_index: int,
    source_input: str,
    cwd: str | None,
    projected_binding: object = _NO_PROJECTED_BINDING,
    projection_gap_reason: str | None = None,
) -> tuple[str, str, PathNormalization, dict[str, object]]:
    workflow = provenance.get("workflow")
    if not isinstance(workflow, Mapping):
        raise ExtractionError("v3 occurrence workflow provenance is invalid")
    event = workflow.get("event")
    if not isinstance(event, str) or not event:
        event = ""
    head_sha = _require_git_oid(
        workflow.get("head_sha"),
        where="v3 workflow.head_sha",
    )
    canonical_repository = _repository_name(
        provenance.get("repository"),
        where="v3 repository",
    )
    source_repository = _repository_name(
        provenance.get("source_repository"),
        where="v3 source_repository",
    )
    checkout_reason: str | None = None
    selected: Mapping[str, Any] | None = None
    uses_projection = projected_binding is not _NO_PROJECTED_BINDING
    bindings = action.get("repository_source_bindings")
    if uses_projection:
        if projected_binding is None:
            checkout_reason = (
                projection_gap_reason or "source_binding_projection_typed_gap"
            )
        elif isinstance(projected_binding, Mapping):
            selected = projected_binding
        else:
            checkout_reason = "source_binding_projection_record_is_invalid"
    else:
        binding_count = action.get("repository_source_binding_count")
        if (
            not isinstance(bindings, list)
            or binding_count != len(bindings)
            or len(bindings) != len(action.get("source_inputs", []))
            or source_index >= len(bindings)
            or not isinstance(bindings[source_index], Mapping)
        ):
            checkout_reason = "missing_or_ambiguous_repository_source_binding"
        else:
            selected = bindings[source_index]

    if event in {"pull_request", "pull_request_target"}:
        expected_repository = canonical_repository
        checkout_kind = (
            "pull_request_merge"
            if event == "pull_request"
            else "pull_request_target_base"
        )
    elif event and canonical_repository == source_repository:
        expected_repository = canonical_repository
        checkout_kind = "same_repository_head"
    else:
        expected_repository = canonical_repository
        checkout_kind = "unproven_head_or_fork_checkout"
        checkout_reason = checkout_reason or "workflow_event_cannot_prove_checkout_tuple"

    if not uses_projection and isinstance(bindings, list):
        for candidate in bindings:
            if not isinstance(candidate, Mapping):
                checkout_reason = checkout_reason or (
                    "repository_source_binding_is_invalid"
                )
                break
            try:
                candidate_repository = _repository_name(
                    candidate.get("repository"),
                    where="repository source binding repository",
                )
                candidate_head = _require_git_oid(
                    candidate.get("head_sha"),
                    where="repository source binding head_sha",
                )
            except ExtractionError:
                checkout_reason = checkout_reason or (
                    "repository_source_binding_is_invalid"
                )
                break
            if (
                candidate_repository != expected_repository
                or candidate_head != head_sha
            ):
                checkout_reason = checkout_reason or (
                    "multiple_or_noncanonical_checkout_tuples"
                )
                break

    selected_path: str | None = None
    if selected is not None:
        try:
            selected_repository = _repository_name(
                selected.get("repository"),
                where="repository source binding repository",
            )
            selected_head = _require_git_oid(
                selected.get("head_sha"),
                where="repository source binding head_sha",
            )
        except ExtractionError:
            checkout_reason = checkout_reason or "repository_source_binding_is_invalid"
        else:
            if (
                selected_repository != expected_repository
                or selected_head != head_sha
            ):
                checkout_reason = checkout_reason or (
                    "repository_source_binding_tuple_differs_from_workflow_checkout"
                )
            raw_selected_path = selected.get("source_path")
            if isinstance(raw_selected_path, str) and raw_selected_path:
                selected_path = raw_selected_path
            else:
                checkout_reason = checkout_reason or (
                    "repository_source_binding_path_is_invalid"
                )
            confidence = selected.get("confidence")
            if (
                not isinstance(confidence, Mapping)
                or confidence.get("source") != "relative_source_path_v1"
            ):
                checkout_reason = checkout_reason or (
                    "repository_source_binding_uses_untrusted_path_heuristic"
                )

    platform = _runner_platform(provenance, cwd)
    if (
        platform not in {"posix", "windows"}
        or not isinstance(cwd, str)
        or _hosted_checkout_split(
            _platform_path(cwd, platform=platform),
            platform=platform,
        )
        is None
    ):
        checkout_reason = checkout_reason or "unknown_or_non_hosted_checkout_cwd"
    normalization = normalize_source_candidates(
        source_input,
        cwd,
        platform=platform,
    )
    if normalization.status != RESOLVED:
        checkout_reason = checkout_reason or normalization.reason
    elif selected_path != normalization.candidates[0]:
        checkout_reason = checkout_reason or (
            "repository_source_binding_path_differs_from_exact_checkout_path"
        )
    checkout_evidence: dict[str, object] = {
        "workflow_event": event or None,
        "workflow_head_sha": head_sha,
        "canonical_repository": canonical_repository,
        "head_repository": source_repository,
        "expected_checkout_repository": expected_repository,
        "checkout_kind": checkout_kind,
        "platform": platform,
        "repository_source_binding": (
            None if selected is None else dict(selected)
        ),
        "source_binding_projection_applied": uses_projection,
        "source_binding_projection_gap_reason": (
            projection_gap_reason if uses_projection else None
        ),
        "reason": checkout_reason,
    }
    if checkout_reason is not None:
        normalization = PathNormalization(
            CHECKOUT_PROVENANCE_UNRESOLVABLE,
            (),
            source_input,
            cwd,
            checkout_reason,
        )
    return expected_repository, head_sha, normalization, checkout_evidence


def _create_inventory_spool(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TEMP TABLE inventory_bindings(
            repository TEXT NOT NULL,
            head_sha TEXT NOT NULL,
            source_path TEXT NOT NULL,
            record_json TEXT NOT NULL,
            PRIMARY KEY(repository, head_sha, source_path)
        );
        CREATE TEMP TABLE inventory_references(
            repository TEXT NOT NULL,
            head_sha TEXT NOT NULL,
            source_path TEXT NOT NULL,
            record_sha256 TEXT NOT NULL,
            record_json TEXT NOT NULL,
            PRIMARY KEY(
                repository, head_sha, source_path, record_sha256
            )
        );
        """
    )


def _insert_inventory_record(
    connection: sqlite3.Connection,
    *,
    binding: Mapping[str, object],
    reference: Mapping[str, object],
) -> None:
    key = (
        str(binding["repository"]),
        str(binding["head_sha"]),
        str(binding["source_path"]),
    )
    binding_json = _canonical_json(binding)
    existing = connection.execute(
        """
        SELECT record_json FROM inventory_bindings
        WHERE repository = ? AND head_sha = ? AND source_path = ?
        """,
        key,
    ).fetchone()
    if existing is None:
        connection.execute(
            """
            INSERT INTO inventory_bindings(
                repository, head_sha, source_path, record_json
            ) VALUES (?, ?, ?, ?)
            """,
            (*key, binding_json),
        )
    elif str(existing["record_json"]) != binding_json:
        raise ExtractionError("one source binding has conflicting frozen metadata")
    reference_json = _canonical_json(reference)
    connection.execute(
        """
        INSERT OR IGNORE INTO inventory_references(
            repository, head_sha, source_path, record_sha256, record_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (*key, _sha256_bytes(reference_json.encode("utf-8")), reference_json),
    )


def _spool_selected_inventory(
    connection: sqlite3.Connection,
    *,
    projector: SourceBindingProjector,
) -> None:
    _create_inventory_spool(connection)
    selected_count = 0
    for row in connection.execute(
        """
        SELECT
            selected.token_sequence_sha256 AS selected_token_sequence_sha256,
            selected.token_count AS selected_token_count,
            selected.representative_content_sha256,
            selected.representative_provenance_sha256,
            selected.record_json AS representative_record_json,
            selected.repo, selected.run_attempt, selected.job, selected.step,
            selected.chunk_ordinal,
            occurrences.content_sha256,
            occurrences.provenance_sha256,
            occurrences.provenance_raw_size,
            occurrences.provenance_zlib,
            contents.token_sequence_sha256,
            contents.token_count
        FROM selected_representatives AS selected
        LEFT JOIN occurrences
          ON occurrences.repo = selected.repo
         AND occurrences.run_attempt = selected.run_attempt
         AND occurrences.job = selected.job
         AND occurrences.step = selected.step
         AND occurrences.chunk_ordinal = selected.chunk_ordinal
        LEFT JOIN contents
          ON contents.sha256 = occurrences.content_sha256
        ORDER BY selected.token_sequence_sha256
        """
    ):
        selected_count += 1
        if (
            row["content_sha256"] is None
            or str(row["content_sha256"])
            != str(row["representative_content_sha256"])
            or str(row["provenance_sha256"])
            != str(row["representative_provenance_sha256"])
            or str(row["token_sequence_sha256"])
            != str(row["selected_token_sequence_sha256"])
            or int(row["token_count"]) != int(row["selected_token_count"])
        ):
            raise ExtractionError(
                "CASE5 representative is not an exact member of the frozen store"
            )
        provenance = _decode_provenance(row)
        if provenance.get("schema") != OCCURRENCE_SCHEMA:
            raise ExtractionError("representative occurrence schema is stale")
        run_evidence = provenance.get("run_metadata_evidence")
        if (
            not isinstance(run_evidence, Mapping)
            or run_evidence.get("exact_attempt_match") is not True
        ):
            raise ExtractionError(
                "representative occurrence lacks exact-attempt metadata"
            )
        chunk = provenance.get("chunk")
        training = (
            chunk.get("training_sidecars")
            if isinstance(chunk, Mapping)
            else None
        )
        if (
            not isinstance(training, Mapping)
            or training.get("schema") != TRAINING_SIDECAR_SCHEMA
        ):
            raise ExtractionError("representative training sidecar schema is stale")
        actions = training.get("build_actions")
        if not isinstance(actions, list):
            raise ExtractionError("representative build_actions is not a list")
        occurrence_key = {
            "repo": str(row["repo"]),
            "run_attempt": str(row["run_attempt"]),
            "job": str(row["job"]),
            "step": str(row["step"]),
            "chunk_ordinal": int(row["chunk_ordinal"]),
        }
        representative_record = json.loads(str(row["representative_record_json"]))
        projected_rows = {
            (int(projected["action_index"]), int(projected["source_index"])): (
                json.loads(str(projected["record_json"]))
            )
            for projected in connection.execute(
                """
                SELECT action_index, source_index, record_json
                FROM selected_source_binding_projection
                WHERE repo = ? AND run_attempt = ? AND job = ?
                  AND step = ? AND chunk_ordinal = ?
                ORDER BY action_index, source_index
                """,
                (
                    occurrence_key["repo"],
                    occurrence_key["run_attempt"],
                    occurrence_key["job"],
                    occurrence_key["step"],
                    occurrence_key["chunk_ordinal"],
                ),
            )
        }
        consumed_projection_records = 0
        for action_index, action in enumerate(actions):
            if not isinstance(action, Mapping):
                raise ExtractionError("representative build action is invalid")
            try:
                expected_projection = projector.project_action(
                    occurrence_key=occurrence_key,
                    provenance_sha256=str(row["provenance_sha256"]),
                    provenance=provenance,
                    action=action,
                    action_index=action_index,
                )
            except SourceBindingProjectionError as exc:
                raise ExtractionError(
                    f"representative source projection cannot be recomputed: {exc}"
                ) from exc
            source_inputs = action.get("source_inputs")
            if not isinstance(source_inputs, list) or any(
                not isinstance(item, str) for item in source_inputs
            ):
                raise ExtractionError("representative source_inputs is invalid")
            cwd = action.get("cwd")
            if cwd is not None and not isinstance(cwd, str):
                raise ExtractionError("representative build action cwd is invalid")
            for source_index, source_input in enumerate(source_inputs):
                projection_record = projected_rows.get(
                    (action_index, source_index)
                )
                if projection_record is None:
                    raise ExtractionError(
                        "representative source input lacks its exact projection record"
                    )
                consumed_projection_records += 1
                if (
                    projection_record
                    != expected_projection.records[source_index]
                    or projection_record.get("provenance_sha256")
                    != str(row["provenance_sha256"])
                    or projection_record.get("occurrence_key")
                    != occurrence_key
                    or projection_record.get("action_sha256")
                    != _sha256_bytes(_canonical_json_bytes(action))
                    or projection_record.get("source_input") != source_input
                    or projection_record.get("cwd") != cwd
                ):
                    raise ExtractionError(
                        "representative source projection differs from provenance"
                    )
                projected_binding = projection_record.get("projected_binding")
                projection_reason = projection_record.get("reason")
                projection_gap_reason: str | None = None
                if projected_binding is None:
                    projection_gap_reason = (
                        projection_reason
                        if projection_record.get("change_kind") == "dropped"
                        and isinstance(projection_reason, str)
                        else _SOURCE_BINDING_PROJECTION_VERIFIED_GAP
                    )
                repository, head_sha, normalization, checkout_evidence = (
                    _checkout_binding(
                        provenance,
                        action,
                        source_index=source_index,
                        source_input=source_input,
                        cwd=cwd,
                        projected_binding=projected_binding,
                        projection_gap_reason=projection_gap_reason,
                    )
                )
                projection_record_sha256 = _sha256_bytes(
                    _canonical_json_bytes(projection_record)
                )
                reference_core = {
                    "schema": INVENTORY_SCHEMA,
                    "record_type": "reference",
                    "token_sequence_sha256": str(
                        row["selected_token_sequence_sha256"]
                    ),
                    "representative_occurrence_key": occurrence_key,
                    "representative_content_sha256": str(row["content_sha256"]),
                    "representative_provenance_sha256": str(
                        row["provenance_sha256"]
                    ),
                    "representative_selection_record_sha256": _sha256_bytes(
                        _canonical_json_bytes(representative_record)
                    ),
                    "action_index": action_index,
                    "action_entity_id": action.get("action_entity_id"),
                    "action_shape_sha256": action.get("action_shape_sha256"),
                    "command_sha256": action.get("command_sha256"),
                    "source_input_index": source_index,
                    "source_input": source_input,
                    "cwd": cwd,
                    "source_binding_projection": {
                        "schema": SOURCE_BINDING_PROJECTION_SCHEMA,
                        "mode": projection_record["mode"],
                        "record_sha256": projection_record_sha256,
                        "change_kind": projection_record["change_kind"],
                        "reason": projection_reason,
                    },
                    "normalization": normalization.as_dict(),
                    "checkout_evidence": checkout_evidence,
                }
                if normalization.status == RESOLVED:
                    source_path = normalization.candidates[0]
                else:
                    source_path = _unresolved_source_path(
                        repository,
                        head_sha,
                        reference_core,
                    )
                binding = {
                    "schema": INVENTORY_SCHEMA,
                    "record_type": "binding",
                    "repository": repository,
                    "head_sha": head_sha,
                    "source_path": source_path,
                    "normalization_status": normalization.status,
                    "normalized_candidates": list(normalization.candidates),
                }
                reference = {
                    **reference_core,
                    "repository": repository,
                    "head_sha": head_sha,
                    "source_path": source_path,
                }
                _insert_inventory_record(
                    connection,
                    binding=binding,
                    reference=reference,
                )
        if consumed_projection_records != len(projected_rows):
            raise ExtractionError(
                "representative occurrence has extra projection records"
            )
    expected = int(
        connection.execute(
            "SELECT COUNT(*) FROM selected_representatives"
        ).fetchone()[0]
    )
    if selected_count != expected:
        raise ExtractionError("not every CASE5 representative was scanned")


def _spool_record_digest(
    connection: sqlite3.Connection,
    *,
    table: str,
    domain: str,
) -> tuple[int, str]:
    if table not in {"inventory_bindings", "inventory_references"}:
        raise ValueError("unsupported inventory spool table")
    hasher = _RecordHasher(domain)
    for row in connection.execute(
        f"""
        SELECT record_json FROM {table}
        ORDER BY repository, head_sha, source_path
        {", record_sha256" if table == "inventory_references" else ""}
        """
    ):
        hasher.update(json.loads(str(row["record_json"])))
    return hasher.count, hasher.hexdigest


def _inventory_logical_sha256(header: Mapping[str, object]) -> str:
    semantic_header = {
        key: value
        for key, value in header.items()
        if key != "inventory_logical_sha256"
    }
    return _hash_records(
        "cppmega-ci-source-binding-inventory-v3",
        (semantic_header,),
    )


def _write_inventory_jsonl(
    connection: sqlite3.Connection,
    *,
    header: Mapping[str, object],
    destination: Path,
    force: bool,
    expected_existing_sha256: str | None,
) -> str:
    _preflight_publication(
        destination,
        force=force,
        expected_existing_sha256=expected_existing_sha256,
    )
    temporary = _temporary_sibling(destination)
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as handle:
            for value in (header,):
                raw = _canonical_json_bytes(value) + b"\n"
                if len(raw) > MAX_JSONL_RECORD_BYTES:
                    raise ExtractionError(
                        "source binding inventory header exceeds the record size limit"
                    )
                handle.write(raw)
                digest.update(raw)
            for table in ("inventory_bindings", "inventory_references"):
                suffix = (
                    ", record_sha256"
                    if table == "inventory_references"
                    else ""
                )
                for row in connection.execute(
                    f"""
                    SELECT record_json{suffix} FROM {table}
                    ORDER BY repository, head_sha, source_path{suffix}
                    """
                ):
                    raw = str(row["record_json"]).encode("utf-8") + b"\n"
                    if len(raw) > MAX_JSONL_RECORD_BYTES:
                        raise ExtractionError(
                            "source binding inventory record exceeds the "
                            "record size limit"
                        )
                    handle.write(raw)
                    digest.update(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary(
            temporary,
            destination,
            force=force,
            expected_existing_sha256=expected_existing_sha256,
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedInventory:
    path: Path
    header: Mapping[str, object]
    artifact_sha256: str

    def iter_bindings(self) -> Iterator[dict[str, Any]]:
        for index, (record, _raw) in enumerate(
            _iter_canonical_jsonl(self.path, where="source binding inventory")
        ):
            if index == 0:
                continue
            if record.get("record_type") == "binding":
                yield record
            elif record.get("record_type") != "reference":
                raise ExtractionError("inventory contains an unknown record type")

    def iter_references(self) -> Iterator[dict[str, Any]]:
        for index, (record, _raw) in enumerate(
            _iter_canonical_jsonl(self.path, where="source binding inventory")
        ):
            if index and record.get("record_type") == "reference":
                yield record


def _validate_inventory_binding(
    record: Mapping[str, Any],
    *,
    index: int,
) -> tuple[str, str, str]:
    required = {
        "schema",
        "record_type",
        "repository",
        "head_sha",
        "source_path",
        "normalization_status",
        "normalized_candidates",
    }
    if (
        set(record) != required
        or record.get("schema") != INVENTORY_SCHEMA
        or record.get("record_type") != "binding"
    ):
        raise ExtractionError(f"inventory binding record {index} is invalid")
    repository = _repository_name(
        record.get("repository"),
        where=f"inventory binding record {index} repository",
    )
    head_sha = _require_git_oid(
        record.get("head_sha"),
        where=f"inventory binding record {index} head_sha",
    )
    source_path = record.get("source_path")
    status = record.get("normalization_status")
    candidates = record.get("normalized_candidates")
    if (
        not isinstance(source_path, str)
        or not source_path
        or status not in {
            RESOLVED,
            AMBIGUOUS_PATH,
            GENERATED_OR_MUTATED_UNRESOLVABLE,
            CHECKOUT_PROVENANCE_UNRESOLVABLE,
        }
        or not isinstance(candidates, list)
        or any(not isinstance(item, str) or not item for item in candidates)
        or (status == RESOLVED and candidates != [source_path])
        or (status != RESOLVED and not source_path.startswith("!unresolved/"))
    ):
        raise ExtractionError(f"inventory binding record {index} fields are invalid")
    return repository, head_sha, source_path


def _validate_inventory_reference(
    record: Mapping[str, Any],
    *,
    index: int,
) -> tuple[str, str, str, str]:
    required = {
        "schema",
        "record_type",
        "repository",
        "head_sha",
        "source_path",
        "token_sequence_sha256",
        "representative_occurrence_key",
        "representative_content_sha256",
        "representative_provenance_sha256",
        "representative_selection_record_sha256",
        "action_index",
        "action_entity_id",
        "action_shape_sha256",
        "command_sha256",
        "source_input_index",
        "source_input",
        "cwd",
        "source_binding_projection",
        "normalization",
        "checkout_evidence",
    }
    if (
        set(record) != required
        or record.get("schema") != INVENTORY_SCHEMA
        or record.get("record_type") != "reference"
    ):
        raise ExtractionError(f"inventory reference record {index} is invalid")
    repository = _repository_name(
        record.get("repository"),
        where=f"inventory reference record {index} repository",
    )
    head_sha = _require_git_oid(
        record.get("head_sha"),
        where=f"inventory reference record {index} head_sha",
    )
    source_path = record.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise ExtractionError(f"inventory reference record {index} path is invalid")
    _require_hex64(
        record.get("token_sequence_sha256"),
        where=f"inventory reference record {index} token sequence",
    )
    _require_hex64(
        record.get("representative_content_sha256"),
        where=f"inventory reference record {index} content",
    )
    _require_hex64(
        record.get("representative_provenance_sha256"),
        where=f"inventory reference record {index} provenance",
    )
    _require_hex64(
        record.get("representative_selection_record_sha256"),
        where=f"inventory reference record {index} selection record",
    )
    _require_occurrence_key(
        record.get("representative_occurrence_key"),
        where=f"inventory reference record {index} occurrence key",
    )
    for name in ("action_index", "source_input_index"):
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExtractionError(
                f"inventory reference record {index} {name} is invalid"
            )
    if (
        not isinstance(record.get("source_input"), str)
        or not isinstance(record.get("source_binding_projection"), Mapping)
        or not isinstance(record.get("normalization"), Mapping)
        or not isinstance(record.get("checkout_evidence"), Mapping)
    ):
        raise ExtractionError(
            f"inventory reference record {index} evidence is invalid"
        )
    projection = record["source_binding_projection"]
    projection_change_kind = projection.get("change_kind")
    projection_reason = projection.get("reason")
    if (
        set(projection)
        != {"schema", "mode", "record_sha256", "change_kind", "reason"}
        or projection.get("schema") != SOURCE_BINDING_PROJECTION_SCHEMA
        or projection.get("mode") not in {"legacy_projection", "current_audit"}
        or projection_change_kind not in _SOURCE_BINDING_PROJECTION_REASONS
        or projection_reason
        not in _SOURCE_BINDING_PROJECTION_REASONS.get(
            str(projection_change_kind),
            set(),
        )
        or (
            projection.get("mode") == "current_audit"
            and (
                projection_change_kind != "unchanged"
                or projection_reason != "current_binding_verified"
            )
        )
    ):
        raise ExtractionError(
            f"inventory reference record {index} projection is invalid"
        )
    _require_hex64(
        projection.get("record_sha256"),
        where=f"inventory reference record {index} projection record",
    )
    return (
        repository,
        head_sha,
        source_path,
        _sha256_bytes(_canonical_json_bytes(record)),
    )


def _verify_inventory_reference_membership(path: Path) -> None:
    bindings = VerifiedInventory(path, {}, "").iter_bindings()
    references = VerifiedInventory(path, {}, "").iter_references()
    try:
        binding = next(bindings)
    except StopIteration:
        binding = None
    previous_reference_binding: tuple[str, str, str] | None = None
    for reference in references:
        reference_key = (
            str(reference["repository"]),
            str(reference["head_sha"]),
            str(reference["source_path"]),
        )
        while binding is not None:
            binding_key = (
                str(binding["repository"]),
                str(binding["head_sha"]),
                str(binding["source_path"]),
            )
            if binding_key >= reference_key:
                break
            try:
                binding = next(bindings)
            except StopIteration:
                binding = None
        if binding is None or binding_key != reference_key:
            raise ExtractionError(
                "inventory reference is not a member of its binding inventory"
            )
        previous_reference_binding = reference_key
    if previous_reference_binding is None:
        # A zero-reference inventory must also have no orphan binding records.
        if binding is not None:
            raise ExtractionError("inventory binding has no representative reference")
        return
    # Every binding is produced from at least one representative reference.
    referenced_keys = (
        (
            str(record["repository"]),
            str(record["head_sha"]),
            str(record["source_path"]),
        )
        for record in VerifiedInventory(path, {}, "").iter_references()
    )
    references_iter = iter(referenced_keys)
    current_reference = next(references_iter, None)
    for binding_record in VerifiedInventory(path, {}, "").iter_bindings():
        binding_key = (
            str(binding_record["repository"]),
            str(binding_record["head_sha"]),
            str(binding_record["source_path"]),
        )
        while current_reference is not None and current_reference < binding_key:
            current_reference = next(references_iter, None)
        if current_reference != binding_key:
            raise ExtractionError(
                "inventory binding has no representative reference"
            )


def verify_binding_inventory(
    inventory_path: str | os.PathLike[str],
) -> VerifiedInventory:
    """Stream-verify a canonical v3 JSONL inventory without materializing it."""

    path = Path(inventory_path)
    artifact = hashlib.sha256()
    header: dict[str, Any] | None = None
    binding_hasher = _RecordHasher("cppmega-ci-source-binding-records-v3")
    reference_hasher = _RecordHasher("cppmega-ci-source-reference-records-v3")
    previous_binding: tuple[str, str, str] | None = None
    previous_reference: tuple[str, str, str, str] | None = None
    phase = "header"
    for index, (record, raw) in enumerate(
        _iter_canonical_jsonl(path, where="source binding inventory")
    ):
        artifact.update(raw)
        if index == 0:
            if (
                record.get("schema") != INVENTORY_SCHEMA
                or record.get("record_type") != "header"
            ):
                raise ExtractionError("inventory header schema is missing or stale")
            header = record
            phase = "bindings"
            continue
        record_type = record.get("record_type")
        if record_type == "binding" and phase == "bindings":
            key = _validate_inventory_binding(record, index=index)
            if previous_binding is not None and key <= previous_binding:
                raise ExtractionError("inventory bindings are not sorted and unique")
            previous_binding = key
            binding_hasher.update(record)
        elif record_type == "reference":
            phase = "references"
            key = _validate_inventory_reference(record, index=index)
            if previous_reference is not None and key <= previous_reference:
                raise ExtractionError("inventory references are not sorted and unique")
            previous_reference = key
            reference_hasher.update(record)
        else:
            raise ExtractionError("inventory record order or type is invalid")
    if header is None:
        raise ExtractionError("inventory header is missing")
    required_header_keys = {
        "schema",
        "record_type",
        "occurrence_schema",
        "training_sidecar_schema",
        "normalization_schema",
        "content_semantics",
        "occurrence_set_sha256",
        "upstream_fetch_receipt_sha256",
        "frozen_fetch_state",
        "frozen_fetch_state_sha256",
        "content_store_receipt_sha256",
        "content_store_sqlite_schema_sha256",
        "content_store_sqlite_logical_sha256",
        "case5_export_receipt_sha256",
        "representative_ledger_schema",
        "representative_count",
        "representative_ledger_sha256",
        "representative_ledger_artifact_sha256",
        "source_binding_projection_schema",
        "source_binding_projection_mode",
        "source_binding_projection_script_sha256",
        "source_binding_projection_input_parser_sha256",
        "source_binding_projection_target_parser_sha256",
        "source_binding_projection_ledger_record_count",
        "source_binding_projection_ledger_sha256",
        "source_binding_projection_ledger_artifact_sha256",
        "binding_count",
        "reference_count",
        "binding_records_sha256",
        "reference_records_sha256",
        "inventory_logical_sha256",
    }
    if set(header) != required_header_keys:
        raise ExtractionError("inventory header fields are incomplete or stale")
    for name in (
        "occurrence_set_sha256",
        "upstream_fetch_receipt_sha256",
        "frozen_fetch_state_sha256",
        "content_store_receipt_sha256",
        "content_store_sqlite_schema_sha256",
        "content_store_sqlite_logical_sha256",
        "case5_export_receipt_sha256",
        "representative_ledger_sha256",
        "representative_ledger_artifact_sha256",
        "source_binding_projection_script_sha256",
        "source_binding_projection_input_parser_sha256",
        "source_binding_projection_target_parser_sha256",
        "source_binding_projection_ledger_sha256",
        "source_binding_projection_ledger_artifact_sha256",
        "binding_records_sha256",
        "reference_records_sha256",
        "inventory_logical_sha256",
    ):
        _require_hex64(header.get(name), where=f"inventory header {name}")
    expected_shape = {
        "schema": INVENTORY_SCHEMA,
        "record_type": "header",
        "occurrence_schema": OCCURRENCE_SCHEMA,
        "training_sidecar_schema": TRAINING_SIDECAR_SCHEMA,
        "normalization_schema": NORMALIZATION_SCHEMA,
        "content_semantics": CONTENT_SEMANTICS,
        "representative_ledger_schema": REPRESENTATIVE_LEDGER_SCHEMA,
        "source_binding_projection_schema": SOURCE_BINDING_PROJECTION_SCHEMA,
    }
    if any(header.get(key) != value for key, value in expected_shape.items()):
        raise ExtractionError("inventory header contract is stale")
    frozen_fetch_state = header.get("frozen_fetch_state")
    if (
        not isinstance(frozen_fetch_state, Mapping)
        or header.get("frozen_fetch_state_sha256")
        != _sha256_bytes(_canonical_json_bytes(frozen_fetch_state))
    ):
        raise ExtractionError("inventory frozen fetch-state binding differs")
    for name, actual in (
        ("binding_count", binding_hasher.count),
        ("reference_count", reference_hasher.count),
    ):
        if header.get(name) != actual:
            raise ExtractionError(f"inventory header {name} differs")
    representative_count = header.get("representative_count")
    if (
        isinstance(representative_count, bool)
        or not isinstance(representative_count, int)
        or representative_count < 0
    ):
        raise ExtractionError("inventory representative_count is invalid")
    projection_record_count = header.get(
        "source_binding_projection_ledger_record_count"
    )
    if (
        isinstance(projection_record_count, bool)
        or not isinstance(projection_record_count, int)
        or projection_record_count < 0
        or header.get("source_binding_projection_mode")
        not in {"legacy_projection", "current_audit"}
    ):
        raise ExtractionError("inventory source-binding projection is invalid")
    if (
        header.get("binding_records_sha256") != binding_hasher.hexdigest
        or header.get("reference_records_sha256") != reference_hasher.hexdigest
        or header.get("inventory_logical_sha256")
        != _inventory_logical_sha256(header)
    ):
        raise ExtractionError("inventory logical digest differs")
    _verify_inventory_reference_membership(path)
    return VerifiedInventory(path, header, artifact.hexdigest())


def extract_binding_inventory(
    content_store_root: str | os.PathLike[str],
    upstream_fetch_receipt_path: str | os.PathLike[str],
    *,
    content_store_receipt_path: str | os.PathLike[str],
    case5_export_receipt_path: str | os.PathLike[str],
    representative_ledger_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    force: bool = False,
    expected_output_sha256: str | None = None,
) -> dict[str, Any]:
    """Write a representative-only canonical JSONL binding inventory."""

    root = Path(content_store_root)
    fetch_receipt, fetch_raw = _read_json_object(
        Path(upstream_fetch_receipt_path),
        where="upstream fetch receipt",
    )
    content_receipt, content_receipt_raw = _read_json_object(
        Path(content_store_receipt_path),
        where="content-store receipt",
    )
    export_receipt, export_receipt_raw = _read_json_object(
        Path(case5_export_receipt_path),
        where="CASE5 export receipt",
    )
    if fetch_receipt.get("schema") != FETCH_RECEIPT_SCHEMA:
        raise ExtractionError("upstream fetch receipt schema is missing or stale")
    nested = fetch_receipt.get("content_store_receipt")
    if not isinstance(nested, Mapping) or dict(nested) != content_receipt:
        raise ExtractionError("fetch and standalone content-store receipts differ")
    if export_receipt.get("schema") != CASE5_EXPORT_SCHEMA:
        raise ExtractionError("CASE5 export receipt schema is missing or stale")
    frozen_fetch_state = fetch_receipt.get("frozen_fetch_state")
    input_fetch_state = export_receipt.get("input_fetch_state")
    if not isinstance(frozen_fetch_state, Mapping) or not isinstance(
        input_fetch_state,
        Mapping,
    ):
        raise ExtractionError("frozen fetch-state lineage binding is missing")
    if dict(frozen_fetch_state) != dict(input_fetch_state):
        raise ExtractionError(
            "upstream and CASE5 frozen fetch-state bindings differ"
        )
    frozen_fetch_state_binding = dict(frozen_fetch_state)
    frozen_fetch_state_sha256 = _sha256_bytes(
        _canonical_json_bytes(frozen_fetch_state_binding)
    )
    input_store = export_receipt.get("input_store")
    content_receipt_sha256 = _sha256_bytes(content_receipt_raw)
    if (
        not isinstance(input_store, Mapping)
        or input_store.get("receipt_sha256") != content_receipt_sha256
    ):
        raise ExtractionError("CASE5 export is not bound to this store receipt")
    for export_name, receipt_name in (
        ("sqlite_schema_sha256", "sqlite_schema_sha256"),
        ("sqlite_logical_sha256", "sqlite_logical_sha256"),
        ("logical_content_set_sha256", "logical_content_set_sha256"),
        (
            "logical_token_sequence_set_sha256",
            "logical_token_sequence_set_sha256",
        ),
        ("occurrence_set_sha256", "occurrence_set_sha256"),
        ("pack_hashes", "pack_hashes"),
    ):
        if input_store.get(export_name) != content_receipt.get(receipt_name):
            raise ExtractionError(
                f"CASE5 export input_store {export_name} differs from receipt"
            )

    output = Path(output_path)
    with tempfile.TemporaryDirectory(prefix="ci-source-inventory-") as temporary:
        snapshot_db = Path(temporary) / "snapshot.sqlite3"
        connection, projection_scope = _verify_frozen_content_store(
            root,
            content_receipt,
            snapshot_db=snapshot_db,
        )
        try:
            representative_binding = _prepare_representatives(
                connection,
                ledger_path=Path(representative_ledger_path),
                export_receipt=export_receipt,
            )
            projection_binding, projector = _prepare_source_binding_projection(
                connection,
                ledger_path=(
                    Path(case5_export_receipt_path).parent
                    / _SOURCE_BINDING_PROJECTION_ARTIFACT
                ),
                export_receipt=export_receipt,
                content_receipt=content_receipt,
                frozen_fetch_state=frozen_fetch_state_binding,
                projection_scope=projection_scope,
            )
            _spool_selected_inventory(connection, projector=projector)
            binding_count, binding_sha = _spool_record_digest(
                connection,
                table="inventory_bindings",
                domain="cppmega-ci-source-binding-records-v3",
            )
            reference_count, reference_sha = _spool_record_digest(
                connection,
                table="inventory_references",
                domain="cppmega-ci-source-reference-records-v3",
            )
            if (
                projection_binding[
                    "source_binding_projection_script_sha256"
                ]
                != projection_script_sha256()
                or projection_binding[
                    "source_binding_projection_target_parser_sha256"
                ]
                != target_parser_script_sha256()
            ):
                raise ExtractionError(
                    "source-binding projection implementation changed during extraction"
                )
            header: dict[str, object] = {
                "schema": INVENTORY_SCHEMA,
                "record_type": "header",
                "occurrence_schema": OCCURRENCE_SCHEMA,
                "training_sidecar_schema": TRAINING_SIDECAR_SCHEMA,
                "normalization_schema": NORMALIZATION_SCHEMA,
                "content_semantics": CONTENT_SEMANTICS,
                "occurrence_set_sha256": content_receipt[
                    "occurrence_set_sha256"
                ],
                "upstream_fetch_receipt_sha256": _sha256_bytes(fetch_raw),
                "frozen_fetch_state": frozen_fetch_state_binding,
                "frozen_fetch_state_sha256": frozen_fetch_state_sha256,
                "content_store_receipt_sha256": content_receipt_sha256,
                "content_store_sqlite_schema_sha256": content_receipt[
                    "sqlite_schema_sha256"
                ],
                "content_store_sqlite_logical_sha256": content_receipt[
                    "sqlite_logical_sha256"
                ],
                "case5_export_receipt_sha256": _sha256_bytes(export_receipt_raw),
                "representative_ledger_schema": REPRESENTATIVE_LEDGER_SCHEMA,
                **representative_binding,
                **projection_binding,
                "binding_count": binding_count,
                "reference_count": reference_count,
                "binding_records_sha256": binding_sha,
                "reference_records_sha256": reference_sha,
            }
            header["inventory_logical_sha256"] = _inventory_logical_sha256(
                header
            )
            artifact_sha = _write_inventory_jsonl(
                connection,
                header=header,
                destination=output,
                force=force,
                expected_existing_sha256=expected_output_sha256,
            )
        finally:
            connection.close()
    verified = verify_binding_inventory(output)
    if verified.artifact_sha256 != artifact_sha:
        raise ExtractionError("published inventory artifact SHA-256 changed")
    return {
        "schema": INVENTORY_SCHEMA,
        "status": "complete",
        "inventory_logical_sha256": header["inventory_logical_sha256"],
        "inventory_artifact_sha256": artifact_sha,
        "binding_count": header["binding_count"],
        "reference_count": header["reference_count"],
        "representative_count": header["representative_count"],
        "source_binding_projection_ledger_record_count": header[
            "source_binding_projection_ledger_record_count"
        ],
        "source_binding_projection_ledger_sha256": header[
            "source_binding_projection_ledger_sha256"
        ],
        "source_binding_projection_ledger_artifact_sha256": header[
            "source_binding_projection_ledger_artifact_sha256"
        ],
    }


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


class _GitObjectTooLarge(Exception):
    def __init__(self, oid: str, size: int) -> None:
        super().__init__(oid, size)
        self.oid = oid
        self.size = size


class LocalGitResolver:
    """Resolve exact commit paths using only explicitly mapped local mirrors."""

    def __init__(
        self,
        mirror_mapping: Mapping[
            str, str | os.PathLike[str] | Mapping[str, object] | None
        ],
        *,
        max_git_object_bytes: int = DEFAULT_MAX_GIT_OBJECT_BYTES,
    ) -> None:
        if (
            isinstance(max_git_object_bytes, bool)
            or not isinstance(max_git_object_bytes, int)
            or max_git_object_bytes < 1
        ):
            raise ValueError("max_git_object_bytes must be a positive integer")
        self._mapping = _normalize_mirror_mapping(mirror_mapping)
        self.max_git_object_bytes = max_git_object_bytes

    @staticmethod
    def _git_oid(object_format: str, object_type: str, payload: bytes) -> str:
        if object_format not in {"sha1", "sha256"}:
            raise ResolutionIntegrityError(
                f"unsupported Git object format {object_format!r}"
            )
        header = f"{object_type} {len(payload)}\0".encode("ascii")
        constructor = hashlib.sha1 if object_format == "sha1" else hashlib.sha256
        digest = constructor()
        digest.update(header)
        digest.update(payload)
        return digest.hexdigest()

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
                capture_output=True,
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
            raw_size = (
                self._run_git(
                    mirror,
                    ["cat-file", "-s", oid],
                    absent_ok=absent_ok,
                )
                .decode("ascii", errors="strict")
                .strip()
            )
            size = int(raw_size)
        except ValueError as exc:
            raise ResolutionIntegrityError("Git object size is invalid") from exc
        if size < 0:
            raise ResolutionIntegrityError("Git object size is negative")
        if size > self.max_git_object_bytes:
            raise _GitObjectTooLarge(oid, size)
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
        if len(payload) != size:
            raise ResolutionIntegrityError(
                f"Git object {oid} size changed while reading"
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
    ) -> Iterator[tuple[str, bytes, str]]:
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
            yield mode, name, raw_oid.hex()
            cursor = nul + 1 + oid_bytes

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
        inventory_record = {
            "schema": binding.get("schema"),
            "record_type": binding.get("record_type"),
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
        }
        _validate_inventory_binding(inventory_record, index=0)
        inventory_record_sha256 = _sha256_bytes(
            _canonical_json_bytes(inventory_record)
        )
        return {
            **resolver_evidence,
            "inventory_binding_record_sha256": inventory_record_sha256,
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
                    CHECKOUT_PROVENANCE_UNRESOLVABLE,
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
                component_bytes = component.encode("utf-8", errors="strict")
                matches: list[tuple[str, str]] = []
                for mode, name, oid in self._parse_tree(
                    tree_payload,
                    oid_bytes=oid_bytes,
                ):
                    if name == component_bytes:
                        matches.append((mode, oid))
                        if len(matches) > 1:
                            break
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
                        status = (
                            UNSUPPORTED_OBJECT
                            if selected_type == "submodule"
                            else PATH_ABSENT
                        )
                        return self._gap(
                            binding,
                            status,
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
        except _GitObjectTooLarge as exc:
            return self._gap(
                binding,
                UNSUPPORTED_OBJECT,
                evidence={
                    "resolver_schema": RESOLVER_SCHEMA,
                    "reason": "git_object_exceeds_bounded_read_policy",
                    "object_oid": exc.oid,
                    "object_size": exc.size,
                    "max_git_object_bytes": self.max_git_object_bytes,
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


def _resolver_contract_sha256() -> str:
    return _hash_records(
        "cppmega-ci-source-resolver-contract-v2",
        (
            {
                "resolver_schema": RESOLVER_SCHEMA,
                "normalization_schema": NORMALIZATION_SCHEMA,
                "content_semantics": CONTENT_SEMANTICS,
                "network_access": False,
                "max_git_object_bytes": DEFAULT_MAX_GIT_OBJECT_BYTES,
                "recursive_tree_walk": False,
                "symlink_dereference": False,
                "submodule_dereference": False,
                "lfs_dereference": False,
                "checkout_root_inference": "exact-github-hosted-structure-only",
            },
        ),
    )


class SourceSidecarStore:
    """Crash-safe exact-dedup CAS for repository Git blob bytes."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        inventory: VerifiedInventory | None = None,
        max_pack_bytes: int | None = None,
    ) -> None:
        self.root = Path(root)
        if self.root.exists():
            root_metadata = self.root.lstat()
            if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
                root_metadata.st_mode
            ):
                raise SourceStoreError("source store root is not a safe directory")
        else:
            self.root.mkdir(parents=True)
        self.db_path = self.root / _SQLITE_NAME
        database_existed = self.db_path.exists()
        if database_existed:
            database_metadata = self.db_path.lstat()
            if stat.S_ISLNK(database_metadata.st_mode) or not stat.S_ISREG(
                database_metadata.st_mode
            ):
                raise SourceStoreError("source store SQLite path is unsafe")
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
                inventory=inventory,
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
        inventory: VerifiedInventory | None,
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
            """
            CREATE TABLE IF NOT EXISTS inventory_bindings(
                repository TEXT NOT NULL,
                head_sha TEXT NOT NULL,
                source_path TEXT NOT NULL,
                normalization_status TEXT NOT NULL,
                record_json TEXT NOT NULL,
                record_sha256 TEXT NOT NULL CHECK(length(record_sha256) = 64),
                PRIMARY KEY(repository, head_sha, source_path)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS inventory_references(
                record_sha256 TEXT PRIMARY KEY CHECK(length(record_sha256) = 64),
                repository TEXT NOT NULL,
                head_sha TEXT NOT NULL,
                source_path TEXT NOT NULL,
                token_sequence_sha256 TEXT NOT NULL
                    CHECK(length(token_sequence_sha256) = 64),
                repo TEXT NOT NULL,
                run_attempt TEXT NOT NULL,
                job TEXT NOT NULL,
                step TEXT NOT NULL,
                chunk_ordinal INTEGER NOT NULL CHECK(chunk_ordinal >= 0),
                record_json TEXT NOT NULL,
                FOREIGN KEY(repository, head_sha, source_path)
                    REFERENCES inventory_bindings(
                        repository, head_sha, source_path
                    )
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS inventory_references_binding_idx
            ON inventory_references(repository, head_sha, source_path)
            """,
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            settings_table = self._connection.execute(
                """
                SELECT 1 FROM sqlite_schema
                WHERE type = 'table' AND name = 'settings'
                """
            ).fetchone()
            existing = (
                {}
                if settings_table is None
                else _source_store_settings(self._connection)
            )
            if not existing:
                if inventory is None:
                    raise SourceStoreError(
                        "new source store requires a verified v3 inventory"
                    )
                for statement in statements:
                    self._connection.execute(statement)
                header = inventory.header
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
                resolver_sha256 = _resolver_contract_sha256()
                settings = {
                    "schema": STORE_SCHEMA,
                    "pack_schema": PACK_SCHEMA,
                    "receipt_schema": RECEIPT_SCHEMA,
                    "inventory_schema": INVENTORY_SCHEMA,
                    "reference_ledger_schema": REFERENCE_LEDGER_SCHEMA,
                    "occurrence_set_sha256": str(
                        header["occurrence_set_sha256"]
                    ),
                    "upstream_fetch_receipt_sha256": str(
                        header["upstream_fetch_receipt_sha256"]
                    ),
                    "frozen_fetch_state_sha256": str(
                        header["frozen_fetch_state_sha256"]
                    ),
                    "content_store_receipt_sha256": str(
                        header["content_store_receipt_sha256"]
                    ),
                    "content_store_sqlite_schema_sha256": str(
                        header["content_store_sqlite_schema_sha256"]
                    ),
                    "content_store_sqlite_logical_sha256": str(
                        header["content_store_sqlite_logical_sha256"]
                    ),
                    "case5_export_receipt_sha256": str(
                        header["case5_export_receipt_sha256"]
                    ),
                    "representative_ledger_sha256": str(
                        header["representative_ledger_sha256"]
                    ),
                    "representative_ledger_artifact_sha256": str(
                        header["representative_ledger_artifact_sha256"]
                    ),
                    "representative_count": str(header["representative_count"]),
                    "inventory_logical_sha256": str(
                        header["inventory_logical_sha256"]
                    ),
                    "inventory_artifact_sha256": inventory.artifact_sha256,
                    "binding_records_sha256": str(
                        header["binding_records_sha256"]
                    ),
                    "reference_records_sha256": str(
                        header["reference_records_sha256"]
                    ),
                    "input_binding_count": str(header["binding_count"]),
                    "input_reference_count": str(header["reference_count"]),
                    "max_pack_bytes": str(limit),
                    "max_git_object_bytes": str(DEFAULT_MAX_GIT_OBJECT_BYTES),
                    "creator_script_sha256": creator_script_sha256,
                    "sqlite_schema_sha256": schema_sha256,
                    "resolver_sha256": resolver_sha256,
                }
                self._connection.executemany(
                    "INSERT INTO settings(key, value) VALUES (?, ?)",
                    sorted(settings.items()),
                )
                binding_batch: list[tuple[object, ...]] = []
                for binding in inventory.iter_bindings():
                    encoded = _canonical_json(binding)
                    binding_batch.append(
                        (
                            binding["repository"],
                            binding["head_sha"],
                            binding["source_path"],
                            binding["normalization_status"],
                            encoded,
                            _sha256_bytes(encoded.encode("utf-8")),
                        )
                    )
                    if len(binding_batch) >= DEFAULT_TRANSACTION_BATCH_SIZE:
                        self._connection.executemany(
                            """
                            INSERT INTO inventory_bindings(
                                repository, head_sha, source_path,
                                normalization_status, record_json, record_sha256
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            binding_batch,
                        )
                        binding_batch.clear()
                if binding_batch:
                    self._connection.executemany(
                        """
                        INSERT INTO inventory_bindings(
                            repository, head_sha, source_path,
                            normalization_status, record_json, record_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        binding_batch,
                    )
                reference_batch: list[tuple[object, ...]] = []
                for reference in inventory.iter_references():
                    key = _require_occurrence_key(
                        reference["representative_occurrence_key"],
                        where="inventory reference occurrence key",
                    )
                    encoded = _canonical_json(reference)
                    reference_batch.append(
                        (
                            _sha256_bytes(encoded.encode("utf-8")),
                            reference["repository"],
                            reference["head_sha"],
                            reference["source_path"],
                            reference["token_sequence_sha256"],
                            key["repo"],
                            key["run_attempt"],
                            key["job"],
                            key["step"],
                            key["chunk_ordinal"],
                            encoded,
                        )
                    )
                    if len(reference_batch) >= DEFAULT_TRANSACTION_BATCH_SIZE:
                        self._connection.executemany(
                            """
                            INSERT INTO inventory_references(
                                record_sha256, repository, head_sha, source_path,
                                token_sequence_sha256, repo, run_attempt, job,
                                step, chunk_ordinal, record_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            reference_batch,
                        )
                        reference_batch.clear()
                if reference_batch:
                    self._connection.executemany(
                        """
                        INSERT INTO inventory_references(
                            record_sha256, repository, head_sha, source_path,
                            token_sequence_sha256, repo, run_attempt, job,
                            step, chunk_ordinal, record_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        reference_batch,
                    )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        settings = _source_store_settings(self._connection)
        required = {
            "schema",
            "pack_schema",
            "receipt_schema",
            "inventory_schema",
            "reference_ledger_schema",
            "occurrence_set_sha256",
            "upstream_fetch_receipt_sha256",
            "frozen_fetch_state_sha256",
            "content_store_receipt_sha256",
            "content_store_sqlite_schema_sha256",
            "content_store_sqlite_logical_sha256",
            "case5_export_receipt_sha256",
            "representative_ledger_sha256",
            "representative_ledger_artifact_sha256",
            "representative_count",
            "inventory_logical_sha256",
            "inventory_artifact_sha256",
            "binding_records_sha256",
            "reference_records_sha256",
            "input_binding_count",
            "input_reference_count",
            "max_pack_bytes",
            "max_git_object_bytes",
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
        if settings["inventory_schema"] != INVENTORY_SCHEMA:
            raise SourceStoreError("source store inventory schema is stale")
        if settings["reference_ledger_schema"] != REFERENCE_LEDGER_SCHEMA:
            raise SourceStoreError("source store reference ledger schema is stale")
        for name in (
            "occurrence_set_sha256",
            "upstream_fetch_receipt_sha256",
            "frozen_fetch_state_sha256",
            "content_store_receipt_sha256",
            "content_store_sqlite_schema_sha256",
            "content_store_sqlite_logical_sha256",
            "case5_export_receipt_sha256",
            "representative_ledger_sha256",
            "representative_ledger_artifact_sha256",
            "inventory_logical_sha256",
            "inventory_artifact_sha256",
            "binding_records_sha256",
            "reference_records_sha256",
            "creator_script_sha256",
            "sqlite_schema_sha256",
            "resolver_sha256",
        ):
            if _HEX64_RE.fullmatch(settings[name]) is None:
                raise SourceStoreError(f"stored {name} is invalid")
        actual_schema_sha256 = _sqlite_schema_sha256(self._connection)
        if actual_schema_sha256 != settings["sqlite_schema_sha256"]:
            raise SourceStoreError("source store SQLite schema hash differs")
        if settings["creator_script_sha256"] != _script_sha256():
            raise SourceStoreError(
                "source store creator script differs; audited migration required"
            )
        if settings["resolver_sha256"] != _resolver_contract_sha256():
            raise SourceStoreError(
                "source store resolver differs; audited migration required"
            )
        for name in (
            "input_binding_count",
            "input_reference_count",
            "representative_count",
            "max_pack_bytes",
            "max_git_object_bytes",
        ):
            try:
                value = int(settings[name])
            except ValueError as exc:
                raise SourceStoreError(f"stored {name} is invalid") from exc
            if value < 0:
                raise SourceStoreError(f"stored {name} is invalid")
        if int(settings["max_git_object_bytes"]) != DEFAULT_MAX_GIT_OBJECT_BYTES:
            raise SourceStoreError("source store Git object limit is stale")
        binding_count = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM inventory_bindings"
            ).fetchone()[0]
        )
        reference_count = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM inventory_references"
            ).fetchone()[0]
        )
        if (
            binding_count != int(settings["input_binding_count"])
            or reference_count != int(settings["input_reference_count"])
        ):
            raise SourceStoreError("stored inventory membership count differs")
        binding_digest = _hash_records(
            "cppmega-ci-source-binding-records-v3",
            (
                json.loads(str(row["record_json"]))
                for row in self._connection.execute(
                    """
                    SELECT record_json FROM inventory_bindings
                    ORDER BY repository, head_sha, source_path
                    """
                )
            ),
        )
        reference_digest = _hash_records(
            "cppmega-ci-source-reference-records-v3",
            (
                json.loads(str(row["record_json"]))
                for row in self._connection.execute(
                    """
                    SELECT record_json FROM inventory_references
                    ORDER BY repository, head_sha, source_path, record_sha256
                    """
                )
            ),
        )
        if (
            binding_digest != settings["binding_records_sha256"]
            or reference_digest != settings["reference_records_sha256"]
        ):
            raise SourceStoreError("stored inventory logical membership differs")
        if (
            inventory is not None
            and _stable_file_sha256(inventory.path)
            != inventory.artifact_sha256
        ):
            raise SourceStoreError(
                "verified inventory artifact changed while importing"
            )
        requested = {
            "max_pack_bytes": (None if max_pack_bytes is None else str(max_pack_bytes)),
        }
        if inventory is not None:
            requested.update(
                {
                    "occurrence_set_sha256": str(
                        inventory.header["occurrence_set_sha256"]
                    ),
                    "upstream_fetch_receipt_sha256": str(
                        inventory.header["upstream_fetch_receipt_sha256"]
                    ),
                    "frozen_fetch_state_sha256": str(
                        inventory.header["frozen_fetch_state_sha256"]
                    ),
                    "inventory_logical_sha256": str(
                        inventory.header["inventory_logical_sha256"]
                    ),
                    "inventory_artifact_sha256": inventory.artifact_sha256,
                    "input_binding_count": str(
                        inventory.header["binding_count"]
                    ),
                    "input_reference_count": str(
                        inventory.header["reference_count"]
                    ),
                }
            )
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
        if _stable_file_sha256(artifact, expected_size=size) != sha256:
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
        else:
            existing, _raw = _read_json_object(
                metadata,
                where="source recovery metadata",
            )
            if existing != record:
                raise SourceStoreError("orphan recovery metadata conflict")
        return record

    def recovery_records(self) -> list[dict[str, object]]:
        directory = self.root / _ORPHAN_DIRECTORY
        try:
            directory_metadata = directory.lstat()
        except FileNotFoundError:
            return []
        if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(
            directory_metadata.st_mode
        ):
            raise SourceStoreError("unsafe orphan quarantine directory")
        records: list[dict[str, object]] = []
        referenced: set[str] = set()
        metadata_paths: list[Path] = []
        for candidate in directory.iterdir():
            if candidate.name.endswith(".recovery.json"):
                if len(metadata_paths) >= MAX_RECOVERY_RECORDS:
                    raise SourceStoreError(
                        "recovery record count exceeds policy"
                    )
                metadata_paths.append(candidate)
        for metadata in sorted(metadata_paths):
            try:
                record, _raw = _read_json_object(
                    metadata,
                    where="source recovery metadata",
                )
            except (OSError, ExtractionError) as exc:
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
            try:
                artifact_digest = _stable_file_sha256(
                    artifact,
                    expected_size=size,
                )
            except (OSError, ExtractionError) as exc:
                raise SourceStoreError(
                    "recovery artifact differs from metadata"
                ) from exc
            if artifact_digest != digest:
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
            pack_count = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM packs"
                ).fetchone()[0]
            )
            if pack_count > MAX_PACK_RECORDS:
                raise SourceStoreError("source pack count exceeds policy")
            rows = self._connection.execute(
                "SELECT pack_id, filename, committed_end FROM packs ORDER BY pack_id"
            )
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
            for path in self.root.glob(_PACK_GLOB):
                indexed = self._connection.execute(
                    "SELECT 1 FROM packs WHERE filename = ?",
                    (path.name,),
                ).fetchone()
                if indexed is None:
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
        if next_id > MAX_PACK_RECORDS:
            raise SourceStoreError("source pack count exceeds policy")
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
        if len(content) > int(self._settings["max_git_object_bytes"]):
            raise SourceStoreError("source blob exceeds bounded object policy")
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

        header = _FRAME_HEADER.pack(
            _FRAME_MAGIC,
            bytes.fromhex(content_sha256),
            len(content),
        )
        frame_size = len(header) + len(content)
        if frame_size + len(_PACK_MAGIC) > self.max_pack_bytes:
            raise SourceStoreError("source blob frame exceeds pack size policy")
        pack = self._connection.execute(
            "SELECT * FROM packs ORDER BY pack_id DESC LIMIT 1"
        ).fetchone()
        if pack is None or (
            int(pack["blob_count"]) > 0
            and int(pack["committed_end"]) + frame_size > self.max_pack_bytes
        ):
            pack = self._new_pack()
        pack_id = int(pack["pack_id"])
        offset = int(pack["committed_end"])
        path = self.root / str(pack["filename"])
        with path.open("r+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() != offset:
                raise SourceStoreError("source pack tail is not at committed boundary")
            handle.write(header)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        self._connection.execute(
            """
            INSERT INTO blobs(content_sha256, size, pack_id, offset, frame_size)
            VALUES (?, ?, ?, ?, ?)
            """,
            (content_sha256, len(content), pack_id, offset, frame_size),
        )
        self._connection.execute(
            """
            UPDATE packs
            SET committed_end = ?, blob_count = blob_count + 1
            WHERE pack_id = ?
            """,
            (offset + frame_size, pack_id),
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
        committed_end = int(row["committed_end"])
        indexed_size = int(row["size"])
        if (
            offset < len(_PACK_MAGIC)
            or frame_size < _FRAME_HEADER.size
            or frame_size > self.max_pack_bytes
            or indexed_size < 0
            or indexed_size > int(self._settings["max_git_object_bytes"])
            or frame_size != _FRAME_HEADER.size + indexed_size
            or offset > committed_end
            or frame_size > committed_end - offset
        ):
            raise SourceStoreError("blob frame exceeds committed pack boundary")
        path = self.root / str(row["filename"])
        try:
            handle, metadata = _open_regular_no_follow(path)
        except (OSError, ExtractionError) as exc:
            raise SourceStoreError("source blob pack is unsafe") from exc
        with handle:
            if metadata.st_size != committed_end:
                raise SourceStoreError("source blob pack size differs")
            handle.seek(offset)
            header = handle.read(_FRAME_HEADER.size)
            if len(header) != _FRAME_HEADER.size:
                raise SourceStoreError("source blob frame header is truncated")
            magic, raw_digest, size = _FRAME_HEADER.unpack(header)
            if (
                size != indexed_size
                or size > frame_size - _FRAME_HEADER.size
                or size > int(self._settings["max_git_object_bytes"])
            ):
                raise SourceStoreError("source blob frame header is invalid")
            content = handle.read(size)
            if len(content) != size:
                raise SourceStoreError("source blob frame payload is truncated")
        if (
            magic != _FRAME_MAGIC
            or frame_size != _FRAME_HEADER.size + size
            or raw_digest.hex() != content_sha256
            or indexed_size != size
            or _sha256_bytes(content) != content_sha256
        ):
            raise SourceStoreError("source blob frame verification failed")
        return content

    def _add_resolution_locked(self, resolution: GitResolution) -> bool:
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
        inventory_record_sha = resolution.evidence.get(
            "inventory_binding_record_sha256"
        )
        if (
            not isinstance(inventory_record_sha, str)
            or _HEX64_RE.fullmatch(inventory_record_sha) is None
        ):
            raise SourceStoreError(
                "resolution lacks frozen inventory membership digest"
            )
        membership = self._connection.execute(
            """
            SELECT record_sha256 FROM inventory_bindings
            WHERE repository = ? AND head_sha = ? AND source_path = ?
            """,
            key,
        ).fetchone()
        if (
            membership is None
            or str(membership["record_sha256"]) != inventory_record_sha
        ):
            raise SourceStoreError(
                "resolution is not a member of the frozen inventory"
            )
        existing = self._connection.execute(
            """
            SELECT record_sha256 FROM bindings
            WHERE repository = ? AND head_sha = ? AND source_path = ?
            """,
            key,
        ).fetchone()
        if existing is not None:
            if str(existing["record_sha256"]) != record_sha256:
                raise BindingConflictError("conflicting replay for source binding")
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
        return True

    def add_resolutions(
        self,
        resolutions: Iterable[GitResolution],
        *,
        batch_size: int = DEFAULT_TRANSACTION_BATCH_SIZE,
    ) -> int:
        """Commit bounded transaction batches and return the number inserted."""

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        added = 0
        pending = 0
        with self._lock:
            try:
                for resolution in resolutions:
                    if not self._connection.in_transaction:
                        self._connection.execute("BEGIN IMMEDIATE")
                    added += int(self._add_resolution_locked(resolution))
                    pending += 1
                    if pending >= batch_size:
                        self._connection.execute("COMMIT")
                        pending = 0
                if self._connection.in_transaction:
                    self._connection.execute("COMMIT")
                return added
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                # A failed batch can leave only an uncommitted durable pack tail.
                self._recover()
                raise

    def add_resolution(self, resolution: GitResolution) -> bool:
        """Commit one resolution; return ``False`` for an identical replay."""

        return bool(self.add_resolutions((resolution,), batch_size=1))

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

    def _iter_reference_entries_locked(self) -> Iterator[dict[str, object]]:
        for row in self._connection.execute(
            """
            SELECT bindings.*, inventory_references.record_json
                   AS reference_record_json,
                   inventory_references.record_sha256
                   AS reference_record_sha256
            FROM inventory_references
            JOIN bindings
              ON bindings.repository = inventory_references.repository
             AND bindings.head_sha = inventory_references.head_sha
             AND bindings.source_path = inventory_references.source_path
            ORDER BY inventory_references.repository,
                     inventory_references.head_sha,
                     inventory_references.source_path,
                     inventory_references.record_sha256
            """
        ):
            reference = json.loads(str(row["reference_record_json"]))
            if (
                not isinstance(reference, dict)
                or _sha256_bytes(_canonical_json_bytes(reference))
                != str(row["reference_record_sha256"])
            ):
                raise SourceStoreError(
                    "stored representative reference digest differs"
                )
            yield {
                "schema": REFERENCE_LEDGER_SCHEMA,
                "record_type": "reference",
                "repository": str(row["repository"]),
                "head_sha": str(row["head_sha"]),
                "source_path": str(row["source_path"]),
                "status": str(row["status"]),
                "content_semantics": CONTENT_SEMANTICS,
                "content_sha256": row["content_sha256"],
                "content_size": row["content_size"],
                "object_format": row["object_format"],
                "blob_oid": row["blob_oid"],
                "mode": row["mode"],
                "object_type": row["object_type"],
                "content_kind": row["content_kind"],
                "representative_reference": reference,
            }

    def _reference_ledger_summary_locked(self) -> dict[str, object]:
        hasher = _RecordHasher(
            "cppmega-ci-source-binding-reference-ledger-v2"
        )
        for entry in self._iter_reference_entries_locked():
            hasher.update(entry)
        return {
            "schema": REFERENCE_LEDGER_SCHEMA,
            "record_type": "header",
            "content_semantics": CONTENT_SEMANTICS,
            "occurrence_set_sha256": self._settings["occurrence_set_sha256"],
            "input_inventory_logical_sha256": self._settings[
                "inventory_logical_sha256"
            ],
            "input_inventory_artifact_sha256": self._settings[
                "inventory_artifact_sha256"
            ],
            "representative_ledger_sha256": self._settings[
                "representative_ledger_sha256"
            ],
            "representative_ledger_artifact_sha256": self._settings[
                "representative_ledger_artifact_sha256"
            ],
            "reference_count": hasher.count,
            "ledger_sha256": hasher.hexdigest,
        }

    def reference_ledger(self) -> dict[str, object]:
        """Return the bounded reference-ledger summary, never all entries."""

        with self._lock:
            self._connection.execute("BEGIN")
            try:
                result = self._reference_ledger_summary_locked()
                self._connection.execute("COMMIT")
                return result
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def write_reference_ledger(
        self,
        path: str | os.PathLike[str],
        *,
        force: bool = False,
        expected_existing_sha256: str | None = None,
    ) -> dict[str, object]:
        """Stream one canonical JSONL ledger from one SQLite read transaction."""

        destination = Path(path)
        _preflight_publication(
            destination,
            force=force,
            expected_existing_sha256=expected_existing_sha256,
        )
        temporary = _temporary_sibling(destination)
        artifact = hashlib.sha256()
        with self._lock:
            self._connection.execute("BEGIN")
            try:
                summary = self._reference_ledger_summary_locked()
                with temporary.open("xb") as handle:
                    raw = _canonical_json_bytes(summary) + b"\n"
                    handle.write(raw)
                    artifact.update(raw)
                    for entry in self._iter_reference_entries_locked():
                        raw = _canonical_json_bytes(entry) + b"\n"
                        handle.write(raw)
                        artifact.update(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                if temporary.exists():
                    temporary.unlink()
                raise
        try:
            _publish_temporary(
                temporary,
                destination,
                force=force,
                expected_existing_sha256=expected_existing_sha256,
            )
        finally:
            if temporary.exists():
                temporary.unlink()
        return {
            **summary,
            "artifact_sha256": artifact.hexdigest(),
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

    def _sqlite_logical_sha256_locked(self) -> str:
        def records() -> Iterator[object]:
            for table, columns, order in (
                ("settings", "key, value", "key"),
                (
                    "packs",
                    "pack_id, filename, committed_end, blob_count",
                    "pack_id",
                ),
                (
                    "blobs",
                    "content_sha256, size, pack_id, offset, frame_size",
                    "content_sha256",
                ),
                (
                    "git_objects",
                    "repository, object_format, blob_oid, content_sha256, size",
                    "repository, object_format, blob_oid",
                ),
                (
                    "inventory_bindings",
                    (
                        "repository, head_sha, source_path, "
                        "normalization_status, record_sha256"
                    ),
                    "repository, head_sha, source_path",
                ),
                (
                    "inventory_references",
                    (
                        "record_sha256, repository, head_sha, source_path, "
                        "token_sequence_sha256, repo, run_attempt, job, step, "
                        "chunk_ordinal"
                    ),
                    "repository, head_sha, source_path, record_sha256",
                ),
                (
                    "bindings",
                    (
                        "repository, head_sha, source_path, status, "
                        "object_format, commit_oid, root_tree_oid, "
                        "parent_tree_oid, object_oid, blob_oid, mode, "
                        "object_type, content_kind, content_sha256, "
                        "content_size, lfs_oid_sha256, lfs_size, record_sha256"
                    ),
                    "repository, head_sha, source_path",
                ),
            ):
                for row in self._connection.execute(
                    f"SELECT {columns} FROM {table} ORDER BY {order}"
                ):
                    yield [table, *list(row)]

        return _hash_records("cppmega-ci-source-sqlite-logical-v2", records())

    def _verify_locked(self) -> dict[str, object]:
        integrity = self._connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()
        if integrity is None or str(integrity[0]) != "ok":
            message = None if integrity is None else str(integrity[0])
            raise SourceStoreError(f"SQLite integrity_check failed: {message}")
        if (
            self._connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchone()
            is not None
        ):
            raise SourceStoreError("SQLite foreign_key_check failed")
        if (
            self._settings["creator_script_sha256"] != _script_sha256()
            or self._settings["resolver_sha256"] != _resolver_contract_sha256()
        ):
            raise SourceStoreError(
                "resume requires the exact creator script and resolver"
            )

        actual_pack_count = 0
        for path in self.root.glob(_PACK_GLOB):
            actual_pack_count += 1
            if (
                self._connection.execute(
                    "SELECT 1 FROM packs WHERE filename = ?",
                    (path.name,),
                ).fetchone()
                is None
            ):
                raise SourceStoreError("source pack file set differs from SQLite")
        indexed_pack_count = int(
            self._connection.execute("SELECT COUNT(*) FROM packs").fetchone()[0]
        )
        if actual_pack_count != indexed_pack_count:
            raise SourceStoreError("source pack file set differs from SQLite")
        pack_hashes: list[dict[str, object]] = []
        verified_blobs = 0
        for pack in self._connection.execute("SELECT * FROM packs ORDER BY pack_id"):
            path = self.root / str(pack["filename"])
            committed_end = int(pack["committed_end"])
            try:
                pack_sha256 = _stable_file_sha256(
                    path,
                    expected_size=committed_end,
                )
            except (ExtractionError, OSError) as exc:
                raise SourceStoreError(
                    "source pack committed file verification failed"
                ) from exc
            handle, _metadata = _open_regular_no_follow(path)
            with handle:
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
                    "sha256": pack_sha256,
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

        status_counts: Counter[str] = Counter()
        binding_count = 0
        for sidecar in self.iter_binding_sidecars():
            binding_count += 1
            status = str(sidecar["status"])
            status_counts[status] += 1
            if status not in ALL_STATUSES:
                raise SourceStoreError("stored binding has unknown status")
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
            evidence = sidecar["evidence"]
            if not isinstance(evidence, Mapping):
                raise SourceStoreError("stored resolver evidence is invalid")
            inventory_sha = evidence.get("inventory_binding_record_sha256")
            membership = self._connection.execute(
                """
                SELECT record_sha256 FROM inventory_bindings
                WHERE repository = ? AND head_sha = ? AND source_path = ?
                """,
                (
                    sidecar["repository"],
                    sidecar["head_sha"],
                    sidecar["source_path"],
                ),
            ).fetchone()
            if (
                membership is None
                or inventory_sha != str(membership["record_sha256"])
            ):
                raise SourceStoreError(
                    "stored partial binding is outside frozen inventory"
                )
            if status == RESOLVED:
                object_row = self._connection.execute(
                    """
                    SELECT content_sha256, size FROM git_objects
                    WHERE repository = ? AND object_format = ? AND blob_oid = ?
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
        if binding_count > self.input_binding_count:
            raise SourceStoreError("stored binding count exceeds frozen inventory")
        missing_binding_count = self.input_binding_count - binding_count
        reference_summary = self._reference_ledger_summary_locked()
        input_reference_count = int(self._settings["input_reference_count"])
        reference_count = int(reference_summary["reference_count"])
        if reference_count > input_reference_count:
            raise SourceStoreError("stored references exceed frozen inventory")
        missing_reference_count = input_reference_count - reference_count
        complete = (
            missing_binding_count == 0
            and missing_reference_count == 0
            and int(status_counts.get(RESOLVED, 0)) == self.input_binding_count
            and not any(status_counts.get(status, 0) for status in GAP_STATUSES)
        )
        recovery = self.recovery_records()
        return {
            "ok": True,
            "status": "complete" if complete else "incomplete",
            "schema": STORE_SCHEMA,
            "binding_count": binding_count,
            "missing_binding_count": missing_binding_count,
            "reference_count": reference_count,
            "missing_reference_count": missing_reference_count,
            "reference_ledger": reference_summary,
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
            "sqlite_logical_sha256": self._sqlite_logical_sha256_locked(),
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

    def verify(self) -> dict[str, object]:
        """Verify all physical/logical state from one locked transaction."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                result = self._verify_locked()
                self._connection.execute("COMMIT")
                return result
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def _receipt_locked(self) -> dict[str, object]:
        verification = self._verify_locked()
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
        complete = verification["status"] == "complete"
        ledger = verification["reference_ledger"]
        assert isinstance(ledger, Mapping)
        return {
            "schema": RECEIPT_SCHEMA,
            "status": "complete" if complete else "incomplete",
            "content_semantics": CONTENT_SEMANTICS,
            "input_binding_count": self.input_binding_count,
            "input_reference_count": int(self._settings["input_reference_count"]),
            "input_inventory_logical_sha256": self._settings[
                "inventory_logical_sha256"
            ],
            "input_inventory_artifact_sha256": self._settings[
                "inventory_artifact_sha256"
            ],
            "occurrence_set_sha256": self._settings["occurrence_set_sha256"],
            "upstream_fetch_receipt_sha256": self._settings[
                "upstream_fetch_receipt_sha256"
            ],
            "frozen_fetch_state_sha256": self._settings[
                "frozen_fetch_state_sha256"
            ],
            "content_store_receipt_sha256": self._settings[
                "content_store_receipt_sha256"
            ],
            "content_store_sqlite_schema_sha256": self._settings[
                "content_store_sqlite_schema_sha256"
            ],
            "content_store_sqlite_logical_sha256": self._settings[
                "content_store_sqlite_logical_sha256"
            ],
            "case5_export_receipt_sha256": self._settings[
                "case5_export_receipt_sha256"
            ],
            "representative_count": int(self._settings["representative_count"]),
            "representative_ledger_sha256": self._settings[
                "representative_ledger_sha256"
            ],
            "representative_ledger_artifact_sha256": self._settings[
                "representative_ledger_artifact_sha256"
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
            "missing_reference_count": verification["missing_reference_count"],
            "pack_hashes": verification["pack_hashes"],
            "resolver_schema": RESOLVER_SCHEMA,
            "resolver_sha256": self._settings["resolver_sha256"],
            "store_schema": STORE_SCHEMA,
            "pack_schema": PACK_SCHEMA,
            "binding_sidecar_schema": SIDECAR_SCHEMA,
            "reference_ledger_schema": REFERENCE_LEDGER_SCHEMA,
            "normalization_schema": NORMALIZATION_SCHEMA,
            "sqlite_schema_sha256": self._settings["sqlite_schema_sha256"],
            "sqlite_logical_sha256": verification["sqlite_logical_sha256"],
            "script_sha256": self._settings["creator_script_sha256"],
            "recovery": verification["recovery"],
            "verification": {"mode": "full", "ok": True},
        }

    def receipt(self) -> dict[str, object]:
        """Build the receipt from one lock and one SQLite transaction."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                result = self._receipt_locked()
                self._connection.execute("COMMIT")
                return result
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    completion_receipt = receipt
    build_receipt = receipt
    create_receipt = receipt

    def write_receipt(
        self,
        path: str | os.PathLike[str],
        *,
        force: bool = False,
        expected_existing_sha256: str | None = None,
    ) -> dict[str, object]:
        receipt = self.receipt()
        publish_json(
            path,
            receipt,
            force=force,
            expected_existing_sha256=expected_existing_sha256,
        )
        return receipt

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def materialize_inventory(
    inventory_path: str | os.PathLike[str],
    mirror_mapping: Mapping[str, str | os.PathLike[str] | Mapping[str, object] | None],
    store_root: str | os.PathLike[str],
    *,
    max_pack_bytes: int | None = None,
    receipt_path: str | os.PathLike[str] | None = None,
    ledger_path: str | os.PathLike[str] | None = None,
    force: bool = False,
    expected_receipt_sha256: str | None = None,
    expected_ledger_sha256: str | None = None,
) -> dict[str, object]:
    """Stream, resolve, and batch-commit every frozen inventory binding."""

    inventory = verify_binding_inventory(inventory_path)
    resolver = LocalGitResolver(mirror_mapping)
    with SourceSidecarStore(
        store_root,
        inventory=inventory,
        max_pack_bytes=max_pack_bytes,
    ) as store:
        store.add_resolutions(
            resolver.resolve(binding) for binding in inventory.iter_bindings()
        )
        if ledger_path is not None:
            store.write_reference_ledger(
                ledger_path,
                force=force,
                expected_existing_sha256=expected_ledger_sha256,
            )
        if receipt_path is not None:
            return store.write_receipt(
                receipt_path,
                force=force,
                expected_existing_sha256=expected_receipt_sha256,
            )
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
    inventory.add_argument("--content-store-receipt", type=Path, required=True)
    inventory.add_argument("--case5-export-receipt", type=Path, required=True)
    inventory.add_argument("--representative-ledger", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument("--force", action="store_true")
    inventory.add_argument("--expected-output-sha256")

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
    build.add_argument("--force", action="store_true")
    build.add_argument("--expected-receipt-sha256")
    build.add_argument("--expected-ledger-sha256")

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
                case5_export_receipt_path=args.case5_export_receipt,
                representative_ledger_path=args.representative_ledger,
                output_path=args.output,
                force=args.force,
                expected_output_sha256=args.expected_output_sha256,
            )
        elif args.command == "build":
            _preflight_publication(
                args.receipt,
                force=args.force,
                expected_existing_sha256=args.expected_receipt_sha256,
            )
            if args.ledger is not None:
                _preflight_publication(
                    args.ledger,
                    force=args.force,
                    expected_existing_sha256=args.expected_ledger_sha256,
                )
            elif args.expected_ledger_sha256 is not None:
                raise ValueError("--expected-ledger-sha256 requires --ledger")
            mirrors = _load_mirror_mapping(args.mirrors)
            result = materialize_inventory(
                args.inventory,
                mirrors,
                args.store,
                max_pack_bytes=args.max_pack_bytes,
                receipt_path=args.receipt,
                ledger_path=args.ledger,
                force=args.force,
                expected_receipt_sha256=args.expected_receipt_sha256,
                expected_ledger_sha256=args.expected_ledger_sha256,
            )
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
        return 3 if result.get("status") == "incomplete" else 0
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
