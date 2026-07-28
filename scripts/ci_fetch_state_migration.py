#!/usr/bin/env python3
"""Immutable projection of supported CI fetch-state v3 snapshots to v4.

The source is never opened writable.  Projection accepts only the two exact
historical v3 contracts:

* the legacy layout without ``attempts.archive_zlib``; and
* the transitional archive-bearing layout that was still labelled v3.

Every other settings/schema pair fails closed.  A destination is built under a
private temporary name, verified as a current v4 snapshot, and then published
without replacing an existing path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import tempfile
from typing import Iterable, Mapping, Sequence
import zipfile


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci_content_store import _sqlite_schema_sha256  # noqa: E402
from scripts.ci_stream_fetch import (  # noqa: E402
    ArchiveError,
    MalformedResponseError,
    SCHEMA_VERSION as CURRENT_FETCH_STATE_SCHEMA,
    _RUN_ATTEMPT_STATES,
    _STATE_SCHEMA,
    _validate_empty_zip_bytes,
    _validate_run_metadata_identity,
)
from scripts.ci_zlib_evidence import (  # noqa: E402
    MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES,
    MAX_EMPTY_ARCHIVE_EVIDENCE_COMPRESSED_BYTES,
    MAX_JOBS_EVIDENCE_BYTES,
    MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
    MAX_RUN_METADATA_BYTES,
    MAX_RUN_METADATA_COMPRESSED_BYTES,
    MAX_STATE_JSON_EVIDENCE_BYTES,
    MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES,
    ZlibEvidenceError,
    constrain_sqlite_evidence_rows,
    strict_bounded_zlib_decode,
)


LEGACY_FETCH_STATE_SCHEMA = "cppmega_ci_stream_fetch_v3"
EXPECTED_CURRENT_FETCH_STATE_SCHEMA = "cppmega_ci_stream_fetch_v4"
LEGACY_V3_SQLITE_SCHEMA_SHA256 = (
    "a4de343203ccdd9769af8120f698649aa54053c5c6629bd5b6d9026dee5fae2c"
)
CURRENT_V4_SQLITE_SCHEMA_SHA256 = (
    "90b968106ae2bedef4da52f0d8e81ea935ef4ff9c224ff5d8275aaa1b0e3e0b3"
)
PROJECTION_SCHEMA = "cppmega_ci_fetch_state_v3_to_v4_projection_v1"
PROJECTION_LEDGER_RECORD_SCHEMA = (
    "cppmega_ci_fetch_state_v3_to_v4_projection_attempt_v1"
)
PROJECTION_LEDGER_DOMAIN = "cppmega-ci-fetch-state-v3-to-v4-ledger-v1"
LEGACY_REQUEUE_REASON = (
    "legacy-zero-member-empty-lacks-replayable-archive-bytes"
)

LEGACY_V3_LAYOUT = "legacy-v3-without-archive-zlib"
TRANSITIONAL_V3_LAYOUT = "transitional-v3-with-archive-zlib"
CURRENT_V4_LAYOUT = "current-v4-with-archive-zlib"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SETTINGS_KEYS = frozenset(
    {
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
)
_EXACT_COPY_TABLES = (
    ("members", "repo,run_id,attempt,archive_member"),
    ("request_ledger", "id"),
    ("binding_upgrades", "id"),
)
_TERMINAL_ERROR_FIELDS = (
    "terminal_http_status",
    "terminal_body_sha256",
    "error_class",
    "error_message",
)
_EXPECTED_INTERNAL_SQLITE_SCHEMA = (
    ("index", "sqlite_autoindex_attempts_1", "attempts", None),
    (
        "index",
        "sqlite_autoindex_binding_upgrades_1",
        "binding_upgrades",
        None,
    ),
    ("index", "sqlite_autoindex_members_1", "members", None),
    ("index", "sqlite_autoindex_settings_1", "settings", None),
    (
        "table",
        "sqlite_sequence",
        "sqlite_sequence",
        "CREATE TABLE sqlite_sequence(name,seq)",
    ),
)


class FetchStateMigrationError(RuntimeError):
    """A source snapshot or projected destination violates the contract."""


def _require_runtime_v4_contract() -> None:
    if CURRENT_FETCH_STATE_SCHEMA != EXPECTED_CURRENT_FETCH_STATE_SCHEMA:
        raise FetchStateMigrationError(
            "runtime fetch-state version is not the projector's frozen v4"
        )


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str

    def receipt_fields(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "bytes": self.size,
            "mtime_ns": self.mtime_ns,
            "inode": self.inode,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class FetchStateInspection:
    path: Path
    layout: str
    settings: Mapping[str, str]
    sqlite_schema_sha256: str
    snapshot: FileSnapshot


@dataclass(frozen=True)
class FetchStateProjectionResult:
    source: FileSnapshot
    destination: FileSnapshot
    source_layout: str
    source_sqlite_schema_sha256: str
    destination_sqlite_schema_sha256: str
    attempts: int
    requeued_attempts: int
    ledger_records: tuple[Mapping[str, object], ...]
    ledger_sha256: str

    def receipt_fields(self) -> dict[str, object]:
        """Return deterministic fields suitable for a merge/finalization receipt."""

        return {
            "schema": PROJECTION_SCHEMA,
            "source": {
                **self.source.receipt_fields(),
                "layout": self.source_layout,
                "settings_schema": LEGACY_FETCH_STATE_SCHEMA,
                "sqlite_schema_sha256": self.source_sqlite_schema_sha256,
            },
            "destination": {
                **self.destination.receipt_fields(),
                "settings_schema": CURRENT_FETCH_STATE_SCHEMA,
                "sqlite_schema_sha256": (
                    self.destination_sqlite_schema_sha256
                ),
            },
            "attempts": self.attempts,
            "requeued_attempts": self.requeued_attempts,
            "ledger_sha256": self.ledger_sha256,
            "ledger": [dict(record) for record in self.ledger_records],
        }


@dataclass
class _OpenedInspection:
    inspection: FetchStateInspection
    connection: sqlite3.Connection
    guard_descriptor: int

    def close(self) -> None:
        try:
            self.connection.close()
        finally:
            os.close(self.guard_descriptor)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FetchStateMigrationError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(descriptor, 1024 * 1024, offset)
        if not block:
            return digest.hexdigest()
        digest.update(block)
        offset += len(block)


def _row_sha256(row: Mapping[str, object]) -> str:
    """Hash a SQLite row with explicit column, type, and byte framing."""

    digest = hashlib.sha256()
    for name in sorted(row):
        value = row[name]
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        if value is None:
            tag, payload = b"n", b""
        elif isinstance(value, bool):
            tag, payload = b"t", b"1" if value else b"0"
        elif isinstance(value, int):
            tag, payload = b"i", str(value).encode("ascii")
        elif isinstance(value, float):
            tag, payload = b"f", value.hex().encode("ascii")
        elif isinstance(value, str):
            tag, payload = b"s", value.encode("utf-8")
        elif isinstance(value, (bytes, bytearray, memoryview)):
            tag, payload = b"b", bytes(value)
        else:
            raise FetchStateMigrationError(
                f"SQLite row column {name!r} has unsupported type"
            )
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _ledger_sha256(records: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    domain = PROJECTION_LEDGER_DOMAIN.encode("utf-8")
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    for record in records:
        payload = _canonical_json_bytes(record)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _require_hex64(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise FetchStateMigrationError(
            f"{where} must be one lowercase hexadecimal SHA-256"
        )
    return value


def _require_int(
    value: object,
    *,
    where: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FetchStateMigrationError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise FetchStateMigrationError(f"{where} must be >= {minimum}")
    return value


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _check_sqlite_sidecars(path: Path, *, where: str) -> None:
    for suffix in ("-wal", "-journal"):
        sidecar = Path(f"{path}{suffix}")
        try:
            sidecar_stat = os.lstat(sidecar)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise FetchStateMigrationError(
                f"{where} SQLite sidecar cannot be inspected: {sidecar}"
            ) from exc
        if stat.S_ISLNK(sidecar_stat.st_mode) or not stat.S_ISREG(
            sidecar_stat.st_mode
        ):
            raise FetchStateMigrationError(
                f"{where} SQLite has an unsafe sidecar: {sidecar.name}"
            )
        if sidecar_stat.st_size != 0:
            raise FetchStateMigrationError(
                f"{where} SQLite is not frozen; found nonempty {sidecar.name}"
            )
    shm = Path(f"{path}-shm")
    try:
        shm_stat = os.lstat(shm)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise FetchStateMigrationError(
            f"{where} SQLite shared-memory sidecar cannot be inspected"
        ) from exc
    raise FetchStateMigrationError(
        f"{where} SQLite is not frozen; found {shm.name}"
    )


def _require_no_sqlite_sidecars(path: Path, *, where: str) -> None:
    for suffix in ("-wal", "-journal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        try:
            os.lstat(sidecar)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise FetchStateMigrationError(
                f"{where} SQLite sidecar cannot be inspected: {sidecar}"
            ) from exc
        raise FetchStateMigrationError(
            f"{where} SQLite has an unexpected sidecar: {sidecar.name}"
        )


def _file_snapshot(
    path: Path,
    descriptor: int,
    *,
    where: str,
) -> FileSnapshot:
    try:
        path_stat_before = os.lstat(path)
        guard_stat_before = os.fstat(descriptor)
    except OSError as exc:
        raise FetchStateMigrationError(f"{where} SQLite disappeared") from exc
    if (
        stat.S_ISLNK(path_stat_before.st_mode)
        or not stat.S_ISREG(path_stat_before.st_mode)
        or not stat.S_ISREG(guard_stat_before.st_mode)
        or (path_stat_before.st_dev, path_stat_before.st_ino)
        != (guard_stat_before.st_dev, guard_stat_before.st_ino)
    ):
        raise FetchStateMigrationError(
            f"{where} SQLite is not a stable regular non-symlink"
        )
    digest = _hash_descriptor(descriptor)
    try:
        path_stat_after = os.lstat(path)
        guard_stat_after = os.fstat(descriptor)
    except OSError as exc:
        raise FetchStateMigrationError(
            f"{where} SQLite disappeared while it was hashed"
        ) from exc
    before = (
        path_stat_before.st_dev,
        path_stat_before.st_ino,
        path_stat_before.st_size,
        path_stat_before.st_mtime_ns,
    )
    after = (
        path_stat_after.st_dev,
        path_stat_after.st_ino,
        path_stat_after.st_size,
        path_stat_after.st_mtime_ns,
    )
    guarded = (
        guard_stat_after.st_dev,
        guard_stat_after.st_ino,
        guard_stat_after.st_size,
        guard_stat_after.st_mtime_ns,
    )
    if before != after or after != guarded:
        raise FetchStateMigrationError(
            f"{where} SQLite changed while it was hashed"
        )
    return FileSnapshot(
        path=path,
        device=path_stat_after.st_dev,
        inode=path_stat_after.st_ino,
        size=path_stat_after.st_size,
        mtime_ns=path_stat_after.st_mtime_ns,
        sha256=digest,
    )


def _assert_same_snapshot(
    expected: FileSnapshot,
    descriptor: int,
    *,
    where: str,
) -> None:
    actual = _file_snapshot(expected.path, descriptor, where=where)
    if actual != expected:
        raise FetchStateMigrationError(
            f"{where} SQLite changed during immutable inspection"
        )


def _dispatch_layout(
    *,
    settings_schema: str,
    sqlite_schema_sha256: str,
    allow_current: bool,
) -> str:
    pair = (settings_schema, sqlite_schema_sha256)
    if pair == (
        LEGACY_FETCH_STATE_SCHEMA,
        LEGACY_V3_SQLITE_SCHEMA_SHA256,
    ):
        return LEGACY_V3_LAYOUT
    if pair == (
        LEGACY_FETCH_STATE_SCHEMA,
        CURRENT_V4_SQLITE_SCHEMA_SHA256,
    ):
        return TRANSITIONAL_V3_LAYOUT
    if allow_current and pair == (
        CURRENT_FETCH_STATE_SCHEMA,
        CURRENT_V4_SQLITE_SCHEMA_SHA256,
    ):
        return CURRENT_V4_LAYOUT
    raise FetchStateMigrationError(
        "fetch-state settings/schema pair is unsupported: "
        f"settings={settings_schema!r}, sqlite={sqlite_schema_sha256}"
    )


def _validate_internal_sqlite_schema(
    connection: sqlite3.Connection,
    *,
    where: str,
) -> None:
    actual = tuple(
        (
            str(row["type"]),
            str(row["name"]),
            str(row["tbl_name"]),
            None if row["sql"] is None else str(row["sql"]),
        )
        for row in connection.execute(
            """
            SELECT type,name,tbl_name,sql
            FROM sqlite_schema
            WHERE name LIKE 'sqlite_%'
            ORDER BY type,name
            """
        )
    )
    if actual != _EXPECTED_INTERNAL_SQLITE_SCHEMA:
        raise FetchStateMigrationError(
            f"{where} has unexpected internal SQLite schema artifacts"
        )


def classify_fetch_state_contract(
    *,
    settings_schema: str,
    sqlite_schema_sha256: str,
) -> str:
    """Classify one exact legacy, transitional, or current contract pair."""

    _require_runtime_v4_contract()
    if not isinstance(settings_schema, str):
        raise FetchStateMigrationError(
            "fetch-state settings schema must be a string"
        )
    _require_hex64(
        sqlite_schema_sha256,
        where="fetch-state SQLite schema SHA-256",
    )
    return _dispatch_layout(
        settings_schema=settings_schema,
        sqlite_schema_sha256=sqlite_schema_sha256,
        allow_current=True,
    )


def _validate_settings_values(
    settings: Mapping[str, str],
    *,
    where: str,
) -> None:
    for key in ("inventory_path", "content_store_path"):
        value = settings[key]
        if not value or not Path(value).is_absolute():
            raise FetchStateMigrationError(
                f"{where} setting {key} must be an absolute path"
            )
    try:
        tokenizer_contract = json.loads(settings["tokenizer_contract"])
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FetchStateMigrationError(
            f"{where} tokenizer contract is invalid JSON"
        ) from exc
    if (
        not isinstance(tokenizer_contract, Mapping)
        or settings["tokenizer_contract"]
        != _canonical_json_bytes(tokenizer_contract).decode("utf-8")
    ):
        raise FetchStateMigrationError(
            f"{where} tokenizer contract is not a canonical JSON object"
        )
    _require_hex64(
        settings["tokenizer_fingerprint"],
        where=f"{where} tokenizer fingerprint",
    )
    for key in (
        "fetcher_script_sha256",
        "parser_script_sha256",
        "content_store_script_sha256",
    ):
        _require_hex64(settings[key], where=f"{where} setting {key}")
    if settings["chunk_semantics"] != (
        "parser-dedup-text-cppmega-training-tokenizer-"
        "payload-only-no-framing-v2"
    ):
        raise FetchStateMigrationError(
            f"{where} chunk semantics are unsupported"
        )
    if not settings["created_at"]:
        raise FetchStateMigrationError(
            f"{where} created_at setting is empty"
        )


def _open_inspection(
    path: str | os.PathLike[str],
    *,
    allow_current: bool,
    where: str,
) -> _OpenedInspection:
    source = _absolute_path(path)
    try:
        initial = os.lstat(source)
    except OSError as exc:
        raise FetchStateMigrationError(
            f"{where} SQLite does not exist safely: {source}"
        ) from exc
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise FetchStateMigrationError(
            f"{where} SQLite must be a regular non-symlink: {source}"
        )
    _check_sqlite_sidecars(source, where=where)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        guard_descriptor = os.open(source, flags)
    except OSError as exc:
        raise FetchStateMigrationError(
            f"{where} SQLite cannot be opened without following links"
        ) from exc
    connection: sqlite3.Connection | None = None
    try:
        guard_stat = os.fstat(guard_descriptor)
        initial_identity = (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
        )
        guarded_identity = (
            guard_stat.st_dev,
            guard_stat.st_ino,
            guard_stat.st_size,
            guard_stat.st_mtime_ns,
        )
        if (
            not stat.S_ISREG(guard_stat.st_mode)
            or guarded_identity != initial_identity
        ):
            raise FetchStateMigrationError(
                f"{where} SQLite changed during no-follow open"
            )
        snapshot = _file_snapshot(
            source,
            guard_descriptor,
            where=where,
        )
        connection = sqlite3.connect(
            f"file:/dev/fd/{guard_descriptor}?mode=ro&immutable=1",
            uri=True,
        )
        constrain_sqlite_evidence_rows(connection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA foreign_keys=ON")
        if int(connection.execute("PRAGMA query_only").fetchone()[0]) != 1:
            raise FetchStateMigrationError(
                f"{where} SQLite query_only could not be enabled"
            )
        integrity = tuple(
            str(row[0])
            for row in connection.execute("PRAGMA integrity_check")
        )
        if integrity != ("ok",):
            raise FetchStateMigrationError(
                f"{where} SQLite integrity_check failed: {integrity}"
            )
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise FetchStateMigrationError(
                f"{where} SQLite foreign_key_check failed"
            )
        sqlite_schema_sha256 = _sqlite_schema_sha256(connection)
        _validate_internal_sqlite_schema(connection, where=where)
        try:
            settings_rows = connection.execute(
                "SELECT key,value FROM settings ORDER BY key"
            ).fetchall()
        except sqlite3.Error as exc:
            raise FetchStateMigrationError(
                f"{where} SQLite settings table is unreadable"
            ) from exc
        settings = {
            str(row["key"]): str(row["value"])
            for row in settings_rows
        }
        if len(settings) != len(settings_rows) or set(settings) != _SETTINGS_KEYS:
            raise FetchStateMigrationError(
                f"{where} settings do not match the exact fetch-state contract"
            )
        settings_schema = settings.get("schema", "")
        layout = _dispatch_layout(
            settings_schema=settings_schema,
            sqlite_schema_sha256=sqlite_schema_sha256,
            allow_current=allow_current,
        )
        _validate_settings_values(settings, where=where)
        _check_sqlite_sidecars(source, where=where)
        _assert_same_snapshot(
            snapshot,
            guard_descriptor,
            where=where,
        )
        return _OpenedInspection(
            inspection=FetchStateInspection(
                path=source,
                layout=layout,
                settings=settings,
                sqlite_schema_sha256=sqlite_schema_sha256,
                snapshot=snapshot,
            ),
            connection=connection,
            guard_descriptor=guard_descriptor,
        )
    except BaseException:
        if connection is not None:
            connection.close()
        os.close(guard_descriptor)
        raise


def inspect_fetch_state(
    path: str | os.PathLike[str],
) -> FetchStateInspection:
    """Inspect one supported frozen v3 snapshot without creating SQLite state."""

    _require_runtime_v4_contract()
    opened = _open_inspection(
        path,
        allow_current=False,
        where="source fetch-state",
    )
    try:
        _validate_source_rows(
            opened.connection,
            layout=opened.inspection.layout,
        )
        _check_sqlite_sidecars(
            opened.inspection.path,
            where="source fetch-state",
        )
        _assert_same_snapshot(
            opened.inspection.snapshot,
            opened.guard_descriptor,
            where="source fetch-state",
        )
        return opened.inspection
    finally:
        opened.close()


def _decode_canonical_zlib(
    compressed: object,
    *,
    expected_raw_size: object,
    expected_sha256: object,
    max_raw_size: int,
    max_compressed_size: int,
    where: str,
) -> object:
    size = _require_int(
        expected_raw_size,
        where=f"{where} raw size",
        minimum=0,
    )
    digest = _require_hex64(
        expected_sha256,
        where=f"{where} SHA-256",
    )
    if not isinstance(compressed, (bytes, bytearray, memoryview)):
        raise FetchStateMigrationError(f"{where} compressed value is not a BLOB")
    try:
        raw = strict_bounded_zlib_decode(
            compressed,
            expected_raw_size=size,
            expected_sha256=digest,
            max_raw_size=max_raw_size,
            max_compressed_size=max_compressed_size,
            where=where,
        )
    except ZlibEvidenceError as exc:
        raise FetchStateMigrationError(
            f"{where} is not exact bounded zlib evidence"
        ) from exc
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FetchStateMigrationError(f"{where} is not valid JSON") from exc
    if raw != _canonical_json_bytes(value):
        raise FetchStateMigrationError(f"{where} is not canonical JSON")
    return value


def _validate_bounded_columns(
    connection: sqlite3.Connection,
    *,
    has_archive_zlib: bool,
) -> None:
    archive_case = ""
    archive_predicate = ""
    parameters: list[int] = [
        MAX_RUN_METADATA_BYTES,
        MAX_RUN_METADATA_COMPRESSED_BYTES,
        MAX_JOBS_EVIDENCE_BYTES,
        MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
    ]
    if has_archive_zlib:
        archive_case = """
            WHEN archive_zlib IS NOT NULL
              AND (
                typeof(archive_zlib)!='blob'
                OR archive_size IS NULL
                OR archive_size<0
                OR archive_size>?
                OR length(archive_zlib)>?
              )
              THEN 'archive_zlib'
        """
        archive_predicate = """
            OR (
              archive_zlib IS NOT NULL
              AND (
                typeof(archive_zlib)!='blob'
                OR archive_size IS NULL
                OR archive_size<0
                OR archive_size>?
                OR length(archive_zlib)>?
              )
            )
        """
        parameters.extend(
            (
                MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES,
                MAX_EMPTY_ARCHIVE_EVIDENCE_COMPRESSED_BYTES,
            )
        )
    parameters.extend(
        (
            MAX_RUN_METADATA_BYTES,
            MAX_RUN_METADATA_COMPRESSED_BYTES,
            MAX_JOBS_EVIDENCE_BYTES,
            MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
        )
    )
    if has_archive_zlib:
        parameters.extend(
            (
                MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES,
                MAX_EMPTY_ARCHIVE_EVIDENCE_COMPRESSED_BYTES,
            )
        )
    row = connection.execute(
        f"""
        SELECT repo,run_id,attempt,
          CASE
            WHEN typeof(run_metadata_zlib)!='blob'
              OR run_metadata_raw_size<0
              OR run_metadata_raw_size>?
              OR length(run_metadata_zlib)>?
              THEN 'run_metadata_zlib'
            WHEN jobs_raw_size IS NOT NULL
              AND (jobs_raw_size<0 OR jobs_raw_size>?)
              THEN 'jobs_raw_size'
            WHEN jobs_zlib IS NOT NULL
              AND (
                typeof(jobs_zlib)!='blob'
                OR length(jobs_zlib)>?
              )
              THEN 'jobs_zlib'
            {archive_case}
          END AS evidence_field
        FROM attempts
        WHERE typeof(run_metadata_zlib)!='blob'
           OR run_metadata_raw_size<0
           OR run_metadata_raw_size>?
           OR length(run_metadata_zlib)>?
           OR (
             jobs_raw_size IS NOT NULL
             AND (jobs_raw_size<0 OR jobs_raw_size>?)
           )
           OR (
             jobs_zlib IS NOT NULL
             AND (
               typeof(jobs_zlib)!='blob'
               OR length(jobs_zlib)>?
             )
           )
           {archive_predicate}
        LIMIT 1
        """,
        tuple(parameters),
    ).fetchone()
    if row is not None:
        raise FetchStateMigrationError(
            "fetch-state attempt evidence exceeds its bounded contract: "
            f"{row['repo']}#{row['run_id']}/{row['attempt']} "
            f"{row['evidence_field']}"
        )
    member = connection.execute(
        """
        SELECT repo,run_id,attempt,archive_member
        FROM members
        WHERE typeof(sidecar_zlib)!='blob'
           OR sidecar_raw_size<0
           OR sidecar_raw_size>?
           OR length(sidecar_zlib)>?
        LIMIT 1
        """,
        (
            MAX_STATE_JSON_EVIDENCE_BYTES,
            MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES,
        ),
    ).fetchone()
    if member is not None:
        raise FetchStateMigrationError(
            "fetch-state member sidecar exceeds its bounded contract: "
            f"{member['repo']}#{member['run_id']}/{member['attempt']}/"
            f"{member['archive_member']}"
        )


def _decode_jobs(
    row: Mapping[str, object],
    *,
    key: tuple[str, int, int],
) -> tuple[Mapping[str, object], ...] | None:
    fields = (
        row["jobs_sha256"],
        row["jobs_raw_size"],
        row["jobs_zlib"],
    )
    present = tuple(value is not None for value in fields)
    if not any(present):
        return None
    if not all(present):
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} has partial jobs evidence"
        )
    value = _decode_canonical_zlib(
        row["jobs_zlib"],
        expected_raw_size=row["jobs_raw_size"],
        expected_sha256=row["jobs_sha256"],
        max_raw_size=MAX_JOBS_EVIDENCE_BYTES,
        max_compressed_size=MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
        where=f"fetch-state attempt {key} jobs",
    )
    if not isinstance(value, list):
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} jobs must be a JSON list"
        )
    jobs: list[Mapping[str, object]] = []
    ids: set[int] = set()
    for index, job in enumerate(value):
        if not isinstance(job, Mapping):
            raise FetchStateMigrationError(
                f"fetch-state attempt {key} jobs[{index}] is not an object"
            )
        job_id = _require_int(
            job.get("id"),
            where=f"fetch-state attempt {key} jobs[{index}].id",
            minimum=1,
        )
        if job_id in ids:
            raise FetchStateMigrationError(
                f"fetch-state attempt {key} jobs has duplicate id {job_id}"
            )
        ids.add(job_id)
        jobs.append(dict(job))
    return tuple(jobs)


def _validate_run_metadata(
    row: Mapping[str, object],
    *,
    key: tuple[str, int, int],
) -> None:
    value = _decode_canonical_zlib(
        row["run_metadata_zlib"],
        expected_raw_size=row["run_metadata_raw_size"],
        expected_sha256=row["run_metadata_sha256"],
        max_raw_size=MAX_RUN_METADATA_BYTES,
        max_compressed_size=MAX_RUN_METADATA_COMPRESSED_BYTES,
        where=f"fetch-state attempt {key} run metadata",
    )
    if not isinstance(value, Mapping):
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} run metadata is not an object"
        )
    source = row["run_metadata_source"]
    source_attempt = _require_int(
        row["run_metadata_source_attempt"],
        where=f"fetch-state attempt {key} metadata source attempt",
        minimum=1,
    )
    exact = _require_int(
        row["run_metadata_exact"],
        where=f"fetch-state attempt {key} metadata exact flag",
        minimum=0,
    )
    if exact not in {0, 1}:
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} metadata exact flag is invalid"
        )
    seed_attempt = _require_int(
        row["inventory_seed_attempt"],
        where=f"fetch-state attempt {key} inventory seed attempt",
        minimum=1,
    )
    seed_sha256 = _require_hex64(
        row["inventory_seed_metadata_sha256"],
        where=f"fetch-state attempt {key} inventory seed SHA-256",
    )
    metadata_sha256 = _require_hex64(
        row["run_metadata_sha256"],
        where=f"fetch-state attempt {key} metadata SHA-256",
    )
    if source not in {
        "inventory-run-list",
        "github-workflow-run-attempt-api",
    }:
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} metadata source is unsupported"
        )
    if exact != int(source_attempt == key[2]):
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} metadata exactness is inconsistent"
        )
    if seed_attempt < key[2]:
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} inventory seed predates the attempt"
        )
    if source == "inventory-run-list" and (
        source_attempt != seed_attempt
        or seed_sha256 != metadata_sha256
    ):
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} inventory metadata binding is inconsistent"
        )
    if source == "github-workflow-run-attempt-api" and (
        not exact or seed_attempt <= key[2]
    ):
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} API metadata binding is inconsistent"
        )
    try:
        _validate_run_metadata_identity(
            value,
            run_id=key[1],
            attempt=source_attempt,
        )
    except MalformedResponseError as exc:
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} run metadata identity is inconsistent"
        ) from exc


def _archive_identity(
    row: Mapping[str, object],
    *,
    key: tuple[str, int, int],
) -> dict[str, object] | None:
    fields = (
        row["archive_source"],
        row["archive_sha256"],
        row["archive_size"],
    )
    present = tuple(value is not None for value in fields)
    if not any(present):
        return None
    if not all(present):
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} has partial archive identity"
        )
    source = fields[0]
    if not isinstance(source, str) or not source:
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} archive source is invalid"
        )
    sha256 = _require_hex64(
        fields[1],
        where=f"fetch-state attempt {key} archive SHA-256",
    )
    size = _require_int(
        fields[2],
        where=f"fetch-state attempt {key} archive size",
        minimum=1,
    )
    return {
        "source": source,
        "sha256": sha256,
        "bytes": size,
    }


def _validate_empty_archive(
    row: Mapping[str, object],
    *,
    key: tuple[str, int, int],
) -> None:
    compressed = row["archive_zlib"]
    if not isinstance(compressed, (bytes, bytearray, memoryview)):
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} replayable empty archive is not a BLOB"
        )
    try:
        raw = strict_bounded_zlib_decode(
            compressed,
            expected_raw_size=_require_int(
                row["archive_size"],
                where=f"fetch-state attempt {key} archive size",
                minimum=1,
            ),
            expected_sha256=_require_hex64(
                row["archive_sha256"],
                where=f"fetch-state attempt {key} archive SHA-256",
            ),
            max_raw_size=MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES,
            max_compressed_size=MAX_EMPTY_ARCHIVE_EVIDENCE_COMPRESSED_BYTES,
            where=f"fetch-state attempt {key} empty archive",
        )
        _validate_empty_zip_bytes(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise FetchStateMigrationError(
                f"fetch-state attempt {key} empty archive has a bad CRC"
            )
    except (ZlibEvidenceError, ArchiveError, OSError, zipfile.BadZipFile) as exc:
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} lacks an exact valid empty ZIP"
        ) from exc


def _validate_members(connection: sqlite3.Connection) -> None:
    invalid_counter = connection.execute(
        """
        SELECT repo,run_id,attempt,archive_member
        FROM members
        WHERE raw_size<0 OR sidecar_raw_size<0
           OR chunk_count<0 OR occurrence_tokens<0
        LIMIT 1
        """
    ).fetchone()
    if invalid_counter is not None:
        raise FetchStateMigrationError(
            "fetch-state member has a negative size/counter: "
            f"{invalid_counter['repo']}#{invalid_counter['run_id']}/"
            f"{invalid_counter['attempt']}/{invalid_counter['archive_member']}"
        )
    for row in connection.execute(
        """
        SELECT * FROM members
        ORDER BY repo,run_id,attempt,archive_member
        """
    ):
        key = (
            str(row["repo"]),
            int(row["run_id"]),
            int(row["attempt"]),
            str(row["archive_member"]),
        )
        for field in (
            "raw_sha256",
            "canonical_sha256",
            "dedup_sha256",
            "sidecar_sha256",
        ):
            _require_hex64(
                row[field],
                where=f"fetch-state member {key} {field}",
            )
        sidecar = _decode_canonical_zlib(
            row["sidecar_zlib"],
            expected_raw_size=row["sidecar_raw_size"],
            expected_sha256=row["sidecar_sha256"],
            max_raw_size=MAX_STATE_JSON_EVIDENCE_BYTES,
            max_compressed_size=MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES,
            where=f"fetch-state member {key} sidecar",
        )
        if not isinstance(sidecar, Mapping):
            raise FetchStateMigrationError(
                f"fetch-state member {key} sidecar is not an object"
            )


def _validate_attempt_accounting(connection: sqlite3.Connection) -> None:
    for row in connection.execute(
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
        GROUP BY
          attempts.repo,attempts.run_id,attempts.attempt,
          attempts.member_count,attempts.chunk_count,
          attempts.occurrence_tokens
        ORDER BY attempts.repo,attempts.run_id,attempts.attempt
        """
    ):
        key = (
            str(row["repo"]),
            int(row["run_id"]),
            int(row["attempt"]),
        )
        declared = (
            _require_int(
                row["member_count"],
                where=f"fetch-state attempt {key} member_count",
                minimum=0,
            ),
            _require_int(
                row["chunk_count"],
                where=f"fetch-state attempt {key} chunk_count",
                minimum=0,
            ),
            _require_int(
                row["occurrence_tokens"],
                where=f"fetch-state attempt {key} occurrence_tokens",
                minimum=0,
            ),
        )
        actual = (
            int(row["actual_member_count"]),
            int(row["actual_chunk_count"]),
            int(row["actual_occurrence_tokens"]),
        )
        if declared != actual:
            raise FetchStateMigrationError(
                f"fetch-state attempt {key} member accounting is inconsistent"
            )


def _project_attempt_row(
    row: sqlite3.Row,
    *,
    layout: str,
) -> tuple[dict[str, object], Mapping[str, object] | None]:
    source = dict(row)
    key = (
        str(source["repo"]),
        _require_int(source["run_id"], where="attempt run_id", minimum=1),
        _require_int(source["attempt"], where="attempt number", minimum=1),
    )
    status_value = source["status"]
    if not isinstance(status_value, str) or status_value not in _RUN_ATTEMPT_STATES:
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} status is unsupported"
        )
    status = status_value
    _require_int(
        source["tries"],
        where=f"fetch-state attempt {key} tries",
        minimum=0,
    )
    member_count = _require_int(
        source["member_count"],
        where=f"fetch-state attempt {key} member_count",
        minimum=0,
    )
    chunk_count = _require_int(
        source["chunk_count"],
        where=f"fetch-state attempt {key} chunk_count",
        minimum=0,
    )
    occurrence_tokens = _require_int(
        source["occurrence_tokens"],
        where=f"fetch-state attempt {key} occurrence_tokens",
        minimum=0,
    )
    for timestamp in ("created_at", "discovered_at", "updated_at"):
        if not isinstance(source[timestamp], str) or not source[timestamp]:
            raise FetchStateMigrationError(
                f"fetch-state attempt {key} {timestamp} is invalid"
            )
    _validate_run_metadata(source, key=key)
    jobs = _decode_jobs(source, key=key)
    archive_identity = _archive_identity(source, key=key)
    has_archive_zlib = layout != LEGACY_V3_LAYOUT
    archive_zlib = source.get("archive_zlib")
    if archive_zlib is not None and (
        not has_archive_zlib
        or status != "empty"
        or member_count != 0
    ):
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} has archive bytes outside zero-member empty"
        )
    if (status in {"done", "empty"} or member_count > 0) and jobs is None:
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} lacks required jobs evidence"
        )
    if status in {"done", "empty"} and archive_identity is None:
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} lacks required archive identity"
        )
    if status in {"done", "empty"} and int(source["run_metadata_exact"]) != 1:
        raise FetchStateMigrationError(
            f"fetch-state attempt {key} lacks exact run metadata"
        )
    if status == "done" and (
        member_count < 1
        or chunk_count < 1
        or occurrence_tokens <= 0
    ):
        raise FetchStateMigrationError(
            f"fetch-state done attempt {key} lacks positive durable content"
        )
    if status in {"terminal_404", "terminal_410"}:
        expected_http_status = 404 if status == "terminal_404" else 410
        terminal_http_status = _require_int(
            source["terminal_http_status"],
            where=f"fetch-state terminal attempt {key} HTTP status",
        )
        if (
            terminal_http_status != expected_http_status
            or member_count != 0
            or chunk_count != 0
            or occurrence_tokens != 0
        ):
            raise FetchStateMigrationError(
                f"fetch-state terminal attempt {key} has inconsistent status "
                "or durable-content counters"
            )
        _require_hex64(
            source["terminal_body_sha256"],
            where=f"fetch-state terminal attempt {key} body SHA-256",
        )
        if (
            source["error_class"] != "TerminalHTTP"
            or not isinstance(source["error_message"], str)
            or not source["error_message"]
        ):
            raise FetchStateMigrationError(
                f"fetch-state terminal attempt {key} lacks terminal error "
                "evidence"
            )
    ledger: Mapping[str, object] | None = None
    projected = dict(source)
    if not has_archive_zlib:
        projected["archive_zlib"] = None
    if status == "empty":
        if chunk_count != 0 or occurrence_tokens != 0:
            raise FetchStateMigrationError(
                f"fetch-state empty attempt {key} retains chunks or tokens"
            )
        if any(source[field] is not None for field in _TERMINAL_ERROR_FIELDS):
            raise FetchStateMigrationError(
                f"fetch-state empty attempt {key} retains terminal/error fields"
            )
        if member_count == 0:
            if layout == LEGACY_V3_LAYOUT:
                legacy_sha256 = _row_sha256(source)
                projected.update(
                    {
                        "status": "retry",
                        "tries": 0,
                        "terminal_http_status": None,
                        "terminal_body_sha256": None,
                        "error_class": None,
                        "error_message": None,
                    }
                )
                projected_sha256 = _row_sha256(projected)
                assert archive_identity is not None
                ledger = {
                    "schema": PROJECTION_LEDGER_RECORD_SCHEMA,
                    "key": {
                        "repo": key[0],
                        "run_id": key[1],
                        "attempt": key[2],
                    },
                    "legacy_row_sha256": legacy_sha256,
                    "projected_row_sha256": projected_sha256,
                    "archive_identity": archive_identity,
                    "jobs_sha256": _require_hex64(
                        source["jobs_sha256"],
                        where=f"fetch-state attempt {key} jobs SHA-256",
                    ),
                    "reason": LEGACY_REQUEUE_REASON,
                    "action": "requeue",
                }
            else:
                if archive_zlib is None:
                    raise FetchStateMigrationError(
                        f"fetch-state attempt {key} lacks replayable empty archive"
                    )
                _validate_empty_archive(source, key=key)
        elif archive_zlib is not None:
            raise FetchStateMigrationError(
                f"fetch-state parsed-empty attempt {key} retains archive bytes"
            )
    return projected, ledger


def _sqlite_sequence_rows(
    connection: sqlite3.Connection,
    *,
    where: str,
) -> tuple[tuple[str, int], ...]:
    output: list[tuple[str, int]] = []
    names: set[str] = set()
    for row in connection.execute(
        "SELECT name,seq FROM sqlite_sequence ORDER BY name"
    ):
        name = str(row["name"])
        if name not in {"request_ledger", "binding_upgrades"}:
            raise FetchStateMigrationError(
                f"{where} sqlite_sequence contains unexpected table {name!r}"
            )
        if name in names:
            raise FetchStateMigrationError(
                f"{where} sqlite_sequence repeats table {name!r}"
            )
        names.add(name)
        sequence = _require_int(
            row["seq"],
            where=f"{where} sqlite_sequence {name}",
            minimum=0,
        )
        output.append((name, sequence))
    return tuple(output)


def _validate_source_rows(
    connection: sqlite3.Connection,
    *,
    layout: str,
) -> int:
    _sqlite_sequence_rows(connection, where="source")
    _validate_bounded_columns(
        connection,
        has_archive_zlib=layout != LEGACY_V3_LAYOUT,
    )
    _validate_members(connection)
    _validate_attempt_accounting(connection)
    attempts = 0
    for row in connection.execute(
        "SELECT * FROM attempts ORDER BY repo,run_id,attempt"
    ):
        _project_attempt_row(row, layout=layout)
        attempts += 1
    return attempts


def _table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> tuple[str, ...]:
    return tuple(
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table})")
    )


def _copy_table(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    *,
    table: str,
    order_by: str,
) -> int:
    columns = _table_columns(source, table)
    destination_columns = _table_columns(destination, table)
    if columns != destination_columns:
        raise FetchStateMigrationError(
            f"cannot byte-copy {table}; source/destination columns differ"
        )
    quoted = ",".join(f'"{column}"' for column in columns)
    placeholders = ",".join("?" for _column in columns)
    count = 0
    for row in source.execute(
        f"SELECT {quoted} FROM {table} ORDER BY {order_by}"
    ):
        destination.execute(
            f"INSERT INTO {table}({quoted}) VALUES ({placeholders})",
            tuple(row[column] for column in columns),
        )
        count += 1
    return count


def _table_sha256(
    connection: sqlite3.Connection,
    *,
    table: str,
    order_by: str,
) -> str:
    digest = hashlib.sha256()
    domain = f"cppmega-ci-fetch-state-table-v1:{table}".encode("utf-8")
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order_by}"):
        row_digest = bytes.fromhex(_row_sha256(dict(row)))
        digest.update(row_digest)
    return digest.hexdigest()


def _copy_sqlite_sequences(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
) -> None:
    source_rows = _sqlite_sequence_rows(
        source,
        where="source",
    )
    for name, sequence in source_rows:
        maximum = int(
            destination.execute(
                f"SELECT COALESCE(MAX(id),0) FROM {name}"
            ).fetchone()[0]
        )
        if sequence < maximum:
            raise FetchStateMigrationError(
                f"source sqlite_sequence for {name} is below its local IDs"
            )
        destination.execute(
            "UPDATE sqlite_sequence SET seq=? WHERE name=?",
            (sequence, name),
        )
        if destination.execute(
            "SELECT changes()"
        ).fetchone()[0] == 0:
            destination.execute(
                "INSERT INTO sqlite_sequence(name,seq) VALUES (?,?)",
                (name, sequence),
            )
    destination_rows = _sqlite_sequence_rows(
        destination,
        where="projected",
    )
    if destination_rows != source_rows:
        raise FetchStateMigrationError(
            "projected sqlite_sequence differs from source local IDs"
        )


def _build_destination(
    source: sqlite3.Connection,
    *,
    source_layout: str,
    temporary_descriptor: int,
) -> tuple[int, tuple[Mapping[str, object], ...]]:
    destination = sqlite3.connect(
        f"file:/dev/fd/{temporary_descriptor}?mode=rw",
        uri=True,
    )
    try:
        constrain_sqlite_evidence_rows(destination)
        destination.row_factory = sqlite3.Row
        destination.execute("PRAGMA foreign_keys=ON")
        # The unpublished database is disposable until its guarded descriptor
        # is fsynced and linked into place.  Disabling the rollback journal
        # avoids reopening any pathname derived from /dev/fd while every
        # database write remains bound to the mkstemp inode.
        if str(
            destination.execute("PRAGMA journal_mode=OFF").fetchone()[0]
        ).lower() != "off":
            raise FetchStateMigrationError(
                "projection SQLite journal_mode could not be disabled"
            )
        destination.execute("PRAGMA synchronous=OFF")
        destination.executescript(_STATE_SCHEMA)
        if _sqlite_schema_sha256(destination) != CURRENT_V4_SQLITE_SCHEMA_SHA256:
            raise FetchStateMigrationError(
                "runtime v4 fetch-state SQL does not match its frozen hash"
            )
        _validate_internal_sqlite_schema(
            destination,
            where="projected v4 SQLite",
        )
        destination.execute("BEGIN IMMEDIATE")
        try:
            for row in source.execute(
                "SELECT key,value FROM settings ORDER BY key"
            ):
                value = (
                    CURRENT_FETCH_STATE_SCHEMA
                    if str(row["key"]) == "schema"
                    else row["value"]
                )
                destination.execute(
                    "INSERT INTO settings(key,value) VALUES (?,?)",
                    (row["key"], value),
                )
            destination_attempt_columns = _table_columns(
                destination,
                "attempts",
            )
            quoted = ",".join(
                f'"{column}"' for column in destination_attempt_columns
            )
            placeholders = ",".join(
                "?" for _column in destination_attempt_columns
            )
            ledger: list[Mapping[str, object]] = []
            attempts = 0
            expected_attempt_digest = hashlib.sha256()
            attempt_domain = (
                "cppmega-ci-fetch-state-table-v1:attempts"
            ).encode("utf-8")
            expected_attempt_digest.update(
                len(attempt_domain).to_bytes(4, "big")
            )
            expected_attempt_digest.update(attempt_domain)
            for row in source.execute(
                "SELECT * FROM attempts ORDER BY repo,run_id,attempt"
            ):
                projected, record = _project_attempt_row(
                    row,
                    layout=source_layout,
                )
                if set(projected) != set(destination_attempt_columns):
                    raise FetchStateMigrationError(
                        "projected attempt columns do not match v4"
                    )
                destination.execute(
                    f"INSERT INTO attempts({quoted}) VALUES ({placeholders})",
                    tuple(
                        projected[column]
                        for column in destination_attempt_columns
                    ),
                )
                if record is not None:
                    ledger.append(record)
                expected_attempt_digest.update(
                    bytes.fromhex(_row_sha256(projected))
                )
                attempts += 1
            for table, order_by in _EXACT_COPY_TABLES:
                _copy_table(
                    source,
                    destination,
                    table=table,
                    order_by=order_by,
                )
            _copy_sqlite_sequences(source, destination)
            destination.execute("COMMIT")
        except BaseException:
            if destination.in_transaction:
                destination.execute("ROLLBACK")
            raise
        integrity = tuple(
            str(row[0])
            for row in destination.execute("PRAGMA integrity_check")
        )
        if integrity != ("ok",):
            raise FetchStateMigrationError(
                f"projected v4 integrity_check failed: {integrity}"
            )
        if destination.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise FetchStateMigrationError(
                "projected v4 foreign_key_check failed"
            )
        if expected_attempt_digest.hexdigest() != _table_sha256(
            destination,
            table="attempts",
            order_by="repo,run_id,attempt",
        ):
            raise FetchStateMigrationError(
                "projected v4 attempts differ from the intended row bytes"
            )
        for table, order_by in _EXACT_COPY_TABLES:
            if _table_sha256(
                source,
                table=table,
                order_by=order_by,
            ) != _table_sha256(
                destination,
                table=table,
                order_by=order_by,
            ):
                raise FetchStateMigrationError(
                    f"projected v4 {table} differs from its source bytes"
                )
        _validate_source_rows(destination, layout=CURRENT_V4_LAYOUT)
        return attempts, tuple(ledger)
    finally:
        destination.close()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_private_destination_parent(path: Path) -> None:
    parent = path.parent
    try:
        parent_stat = os.lstat(parent)
    except OSError as exc:
        raise FetchStateMigrationError(
            f"destination parent must already exist safely: {parent}"
        ) from exc
    if (
        stat.S_ISLNK(parent_stat.st_mode)
        or not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.geteuid()
        or stat.S_IMODE(parent_stat.st_mode) & 0o077
    ):
        raise FetchStateMigrationError(
            "destination parent must be an owned non-symlink directory "
            f"without group/world permissions: {parent}"
        )


def _require_absent(path: Path, *, where: str) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise FetchStateMigrationError(f"{where} cannot be inspected") from exc
    raise FetchStateMigrationError(f"{where} already exists: {path}")


def _assert_path_matches_descriptor(
    path: Path,
    descriptor: int,
    *,
    where: str,
) -> tuple[int, int]:
    try:
        path_stat = os.lstat(path)
        descriptor_stat = os.fstat(descriptor)
    except OSError as exc:
        raise FetchStateMigrationError(f"{where} disappeared") from exc
    if (
        stat.S_ISLNK(path_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or not stat.S_ISREG(descriptor_stat.st_mode)
        or (path_stat.st_dev, path_stat.st_ino)
        != (descriptor_stat.st_dev, descriptor_stat.st_ino)
    ):
        raise FetchStateMigrationError(f"{where} path identity changed")
    return descriptor_stat.st_dev, descriptor_stat.st_ino


def _unlink_if_guarded(path: Path, descriptor: int) -> bool:
    try:
        path_stat = os.lstat(path)
        descriptor_stat = os.fstat(descriptor)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    if (
        stat.S_ISREG(path_stat.st_mode)
        and not stat.S_ISLNK(path_stat.st_mode)
        and (path_stat.st_dev, path_stat.st_ino)
        == (descriptor_stat.st_dev, descriptor_stat.st_ino)
    ):
        path.unlink()
        _fsync_directory(path.parent)
        return True
    return False


def _publish_no_replace(
    temporary: Path,
    destination: Path,
    *,
    guard_descriptor: int,
) -> None:
    expected_identity = _assert_path_matches_descriptor(
        temporary,
        guard_descriptor,
        where="projection temporary SQLite",
    )
    _require_absent(destination, where="destination")
    try:
        os.link(
            temporary,
            destination,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise FetchStateMigrationError(
            f"destination already exists: {destination}"
        ) from exc
    except OSError as exc:
        raise FetchStateMigrationError(
            f"cannot atomically publish projected fetch-state: {exc}"
        ) from exc
    try:
        if _assert_path_matches_descriptor(
            destination,
            guard_descriptor,
            where="published projection SQLite",
        ) != expected_identity:
            raise FetchStateMigrationError(
                "published projection SQLite identity is inconsistent"
            )
        temporary.unlink()
        _fsync_directory(destination.parent)
    except BaseException:
        _unlink_if_guarded(destination, guard_descriptor)
        raise


def project_fetch_state_v3_to_v4(
    source_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
) -> FetchStateProjectionResult:
    """Project one exact frozen v3 state into a new, independently verified v4.

    The source and destination must differ.  The destination is never replaced.
    Requeued legacy zero-member empties are returned as deterministic ledger
    records in the result.
    """

    _require_runtime_v4_contract()
    source = _absolute_path(source_path)
    destination = _absolute_path(destination_path)
    if source == destination:
        raise FetchStateMigrationError(
            "source and destination fetch-state paths must differ"
        )
    opened = _open_inspection(
        source,
        allow_current=False,
        where="source fetch-state",
    )
    source_inspection = opened.inspection
    temporary: Path | None = None
    temporary_descriptor = -1
    published = False
    try:
        _validate_source_rows(
            opened.connection,
            layout=source_inspection.layout,
        )
        _check_sqlite_sidecars(
            source_inspection.path,
            where="source fetch-state",
        )
        _assert_same_snapshot(
            source_inspection.snapshot,
            opened.guard_descriptor,
            where="source fetch-state",
        )
        _require_private_destination_parent(destination)
        _require_absent(destination, where="destination")
        try:
            temporary_descriptor, raw_temporary = tempfile.mkstemp(
                prefix=f".{destination.name}.project-v4-",
                suffix=".sqlite",
                dir=destination.parent,
            )
        except OSError as exc:
            raise FetchStateMigrationError(
                "cannot create projection temporary SQLite safely"
            ) from exc
        temporary = Path(raw_temporary)
        os.fchmod(temporary_descriptor, 0o600)
        _assert_path_matches_descriptor(
            temporary,
            temporary_descriptor,
            where="projection temporary SQLite",
        )
        attempts, ledger = _build_destination(
            opened.connection,
            source_layout=source_inspection.layout,
            temporary_descriptor=temporary_descriptor,
        )
        _require_no_sqlite_sidecars(
            temporary,
            where="projection temporary",
        )
        _assert_path_matches_descriptor(
            temporary,
            temporary_descriptor,
            where="projection temporary SQLite",
        )
        os.fsync(temporary_descriptor)
        _check_sqlite_sidecars(
            source_inspection.path,
            where="source fetch-state",
        )
        _assert_same_snapshot(
            source_inspection.snapshot,
            opened.guard_descriptor,
            where="source fetch-state",
        )
        _publish_no_replace(
            temporary,
            destination,
            guard_descriptor=temporary_descriptor,
        )
        published = True
    finally:
        opened.close()
        if (
            not published
            and temporary is not None
            and temporary_descriptor >= 0
        ):
            _unlink_if_guarded(temporary, temporary_descriptor)
        if not published and temporary_descriptor >= 0:
            os.close(temporary_descriptor)
            temporary_descriptor = -1
    if not published:
        raise AssertionError("unreachable projection publication state")

    projected: _OpenedInspection | None = None
    try:
        projected = _open_inspection(
            destination,
            allow_current=True,
            where="projected fetch-state",
        )
        _require_no_sqlite_sidecars(
            destination,
            where="projected fetch-state",
        )
        if projected.inspection.layout != CURRENT_V4_LAYOUT:
            raise FetchStateMigrationError(
                "published fetch-state is not the exact current v4 contract"
            )
        _validate_source_rows(
            projected.connection,
            layout=projected.inspection.layout,
        )
        destination_snapshot = projected.inspection.snapshot
        guarded_destination = _file_snapshot(
            destination,
            temporary_descriptor,
            where="published projection SQLite",
        )
        if guarded_destination != destination_snapshot:
            raise FetchStateMigrationError(
                "published projection changed during final verification"
            )
        destination_sqlite_schema_sha256 = (
            projected.inspection.sqlite_schema_sha256
        )
    except BaseException:
        _unlink_if_guarded(destination, temporary_descriptor)
        raise
    finally:
        if projected is not None:
            projected.close()
        os.close(temporary_descriptor)
    return FetchStateProjectionResult(
        source=source_inspection.snapshot,
        destination=destination_snapshot,
        source_layout=source_inspection.layout,
        source_sqlite_schema_sha256=(
            source_inspection.sqlite_schema_sha256
        ),
        destination_sqlite_schema_sha256=(
            destination_sqlite_schema_sha256
        ),
        attempts=attempts,
        requeued_attempts=len(ledger),
        ledger_records=ledger,
        ledger_sha256=_ledger_sha256(ledger),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = project_fetch_state_v3_to_v4(
        args.source,
        args.destination,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(result.receipt_fields()) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
