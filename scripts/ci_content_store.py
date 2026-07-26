#!/usr/bin/env python3
"""Crash-safe, content-addressed storage for streamed CI text chunks.

The pack files contain only compressed, canonical UTF-8 content.  SQLite keeps
the content index, every occurrence/provenance record, durable pack boundaries,
and exact counters.  Pack bytes are fsynced before their indexing transaction
commits.  Consequently, bytes past a database-bound ``committed_end`` are
always safe to truncate during recovery.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys
import threading
from typing import Iterable, Iterator, Mapping, Sequence
import zlib


STORE_SCHEMA = "cppmega_ci_content_store_v1"
RECEIPT_SCHEMA = "cppmega_ci_content_store_receipt_v1"
PACK_SCHEMA = "cppmega_ci_content_pack_v1"
DEFAULT_MAX_PACK_BYTES = 256 * 1024 * 1024
DEFAULT_COMPRESSION_LEVEL = 6
PRODUCTION_TARGET_UNIQUE_TOKENS = 20_000_000_000
TOKEN_SEQUENCE_ENCODING = "cppmega-token-sequence-u32be-v1"
RECOVERY_SCHEMA = "cppmega_ci_content_store_recovery_v1"

_PACK_MAGIC = b"CICSPK1\n"
_FRAME_MAGIC = b"CICSFRM1"
_FRAME_HEADER = struct.Struct(">8s32sQQ")
_TOKEN_SEQUENCE_MAGIC = TOKEN_SEQUENCE_ENCODING.encode("ascii") + b"\0"
_PACK_GLOB = "pack-*.cicp"
_SQLITE_NAME = "index.sqlite3"
_ORPHAN_DIRECTORY = "orphaned"
_OCCURRENCE_FIELDS = (
    "repo",
    "run_attempt",
    "job",
    "step",
    "chunk_ordinal",
)


class ContentStoreError(RuntimeError):
    """Base exception for content-store failures."""


class StorePolicyError(ContentStoreError):
    """The requested policy differs from the durable store policy."""


class HashCollisionError(ContentStoreError):
    """A SHA-256 identity resolved to different canonical bytes."""


class OccurrenceConflictError(ContentStoreError):
    """A resumed occurrence key disagrees with its committed record."""


class ContentMetadataConflictError(ContentStoreError):
    """Exact tokenizer metadata disagrees for the same store or content."""


class VerificationError(ContentStoreError):
    """The store failed a full physical or logical verification."""


class ThresholdNotMetError(VerificationError):
    """A completion receipt cannot prove its token threshold."""


@dataclass(frozen=True)
class OccurrenceKey:
    """Stable identity of one chunk occurrence in one CI step."""

    repo: str
    run_attempt: str
    job: str
    step: str
    chunk_ordinal: int

    def as_dict(self) -> dict[str, object]:
        return {
            "repo": self.repo,
            "run_attempt": self.run_attempt,
            "job": self.job,
            "step": self.step,
            "chunk_ordinal": self.chunk_ordinal,
        }

    def as_tuple(self) -> tuple[str, str, str, str, int]:
        return (
            self.repo,
            self.run_attempt,
            self.job,
            self.step,
            self.chunk_ordinal,
        )


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


def _hash_records(domain: str, records: Iterator[object]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    for record in records:
        encoded = _canonical_json_bytes(record)
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _sha256_prefix(path: Path, byte_count: int) -> str:
    digest = hashlib.sha256()
    remaining = byte_count
    with path.open("rb") as handle:
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                raise VerificationError(
                    f"{path.name} ended before committed byte {byte_count}"
                )
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sqlite_schema_sha256(connection: sqlite3.Connection) -> str:
    cursor = connection.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    )
    return _hash_records(
        "cppmega-ci-sqlite-schema-v1",
        (
            [
                str(row["type"]),
                str(row["name"]),
                str(row["tbl_name"]),
                None if row["sql"] is None else str(row["sql"]),
            ]
            for row in cursor
        ),
    )


def hash_token_sequence(token_ids: Sequence[int]) -> str:
    """Hash token IDs using the store's versioned canonical uint32 encoding."""

    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError("token_ids must be a sequence of uint32 integers")
    for token_id in token_ids:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id <= 0xFFFFFFFF
        ):
            raise ValueError("every token ID must be a uint32 integer")
    packed = array("I", token_ids)
    if packed.itemsize != 4:
        raise RuntimeError("this platform does not provide uint32 array storage")
    if sys.byteorder == "little":
        packed.byteswap()
    digest = hashlib.sha256()
    digest.update(_TOKEN_SEQUENCE_MAGIC)
    digest.update(struct.pack(">Q", len(token_ids)))
    digest.update(packed)
    return digest.hexdigest()


class CIContentStore:
    """A durable exact-dedup store for canonical UTF-8 CI chunks."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_pack_bytes: int | None = None,
        pack_size_limit: int | None = None,
        compression_level: int | None = None,
    ) -> None:
        if max_pack_bytes is not None and pack_size_limit is not None:
            if max_pack_bytes != pack_size_limit:
                raise ValueError(
                    "max_pack_bytes and pack_size_limit disagree"
                )
        requested_pack_bytes = (
            max_pack_bytes if max_pack_bytes is not None else pack_size_limit
        )
        if requested_pack_bytes is not None:
            self._validate_pack_limit(requested_pack_bytes)
        if compression_level is not None:
            self._validate_compression_level(compression_level)

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / _SQLITE_NAME
        self._runtime_script_sha256 = _script_sha256()
        database_existed = self.db_path.exists()
        self._lock = threading.RLock()
        self._closed = False
        self._recovery_events: list[dict[str, object]] = []
        self._connection = sqlite3.connect(
            self.db_path,
            isolation_level=None,
            timeout=60.0,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 60000")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = DELETE")
        self._connection.execute("PRAGMA synchronous = FULL")

        try:
            self._initialize_schema(
                requested_pack_bytes=requested_pack_bytes,
                requested_compression_level=compression_level,
            )
            if not database_existed:
                _fsync_file(self.db_path)
                _fsync_directory(self.root)
            self._recover_pack_tails()
        except BaseException:
            self._connection.close()
            self._closed = True
            raise

    @staticmethod
    def _validate_pack_limit(value: int) -> None:
        minimum = len(_PACK_MAGIC) + _FRAME_HEADER.size + 8
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("max_pack_bytes must be an integer")
        if value < minimum:
            raise ValueError(
                f"max_pack_bytes must be at least {minimum} bytes"
            )

    @staticmethod
    def _validate_compression_level(value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("compression_level must be an integer")
        if not 0 <= value <= 9:
            raise ValueError("compression_level must be between 0 and 9")

    @property
    def policy(self) -> dict[str, object]:
        return json.loads(_canonical_json(self._policy))

    @property
    def max_pack_bytes(self) -> int:
        return int(self._policy["max_pack_bytes"])

    @property
    def compression_level(self) -> int:
        compression = self._policy["compression"]
        assert isinstance(compression, dict)
        return int(compression["level"])

    @property
    def script_sha256(self) -> str:
        return self._creator_script_sha256

    def _initialize_schema(
        self,
        *,
        requested_pack_bytes: int | None,
        requested_compression_level: int | None,
    ) -> None:
        schema_statements = (
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS packs (
                pack_id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL UNIQUE,
                committed_end INTEGER NOT NULL
                    CHECK (committed_end >= 8),
                content_count INTEGER NOT NULL
                    CHECK (content_count >= 0)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS token_sequences (
                token_sequence_sha256 TEXT PRIMARY KEY
                    CHECK (length(token_sequence_sha256) = 64),
                token_count INTEGER NOT NULL CHECK (token_count >= 0),
                tokenizer_fingerprint TEXT NOT NULL
                    CHECK (length(tokenizer_fingerprint) > 0)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS contents (
                sha256 TEXT PRIMARY KEY
                    CHECK (length(sha256) = 64),
                raw_size INTEGER NOT NULL CHECK (raw_size >= 0),
                pack_id INTEGER NOT NULL REFERENCES packs(pack_id),
                offset INTEGER NOT NULL CHECK (offset >= 8),
                frame_size INTEGER NOT NULL CHECK (frame_size > 0),
                compressed_size INTEGER NOT NULL
                    CHECK (compressed_size >= 0),
                token_count INTEGER CHECK (token_count >= 0),
                tokenizer_fingerprint TEXT,
                token_sequence_sha256 TEXT
                    REFERENCES token_sequences(token_sequence_sha256),
                CHECK (
                    (token_count IS NULL
                     AND tokenizer_fingerprint IS NULL
                     AND token_sequence_sha256 IS NULL)
                    OR
                    (token_count IS NOT NULL
                     AND tokenizer_fingerprint IS NOT NULL
                     AND length(tokenizer_fingerprint) > 0
                     AND token_sequence_sha256 IS NOT NULL
                     AND length(token_sequence_sha256) = 64)
                ),
                UNIQUE (pack_id, offset)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS occurrences (
                repo TEXT NOT NULL CHECK (length(repo) > 0),
                run_attempt TEXT NOT NULL CHECK (length(run_attempt) > 0),
                job TEXT NOT NULL CHECK (length(job) > 0),
                step TEXT NOT NULL CHECK (length(step) > 0),
                chunk_ordinal INTEGER NOT NULL CHECK (chunk_ordinal >= 0),
                content_sha256 TEXT NOT NULL
                    REFERENCES contents(sha256),
                provenance_sha256 TEXT NOT NULL
                    CHECK (length(provenance_sha256) = 64),
                provenance_raw_size INTEGER NOT NULL
                    CHECK (provenance_raw_size >= 0),
                provenance_zlib BLOB NOT NULL,
                PRIMARY KEY (
                    repo, run_attempt, job, step, chunk_ordinal
                )
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS occurrences_content_idx
            ON occurrences(content_sha256)
            """,
            """
            CREATE TABLE IF NOT EXISTS stats (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                raw_occurrence_bytes INTEGER NOT NULL
                    CHECK (raw_occurrence_bytes >= 0),
                unique_bytes INTEGER NOT NULL CHECK (unique_bytes >= 0),
                duplicate_bytes INTEGER NOT NULL
                    CHECK (duplicate_bytes >= 0),
                unique_content_count INTEGER NOT NULL
                    CHECK (unique_content_count >= 0),
                occurrence_count INTEGER NOT NULL
                    CHECK (occurrence_count >= 0),
                tokenized_unique_content_count INTEGER NOT NULL
                    CHECK (tokenized_unique_content_count >= 0),
                unique_token_sequence_count INTEGER NOT NULL
                    CHECK (unique_token_sequence_count >= 0),
                exact_unique_payload_tokens INTEGER NOT NULL
                    CHECK (exact_unique_payload_tokens >= 0),
                tokenizer_fingerprint TEXT
            )
            """,
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in schema_statements:
                self._connection.execute(statement)
            schema_row = self._connection.execute(
                "SELECT value FROM settings WHERE key = 'schema'"
            ).fetchone()
            if schema_row is None:
                pack_limit = (
                    DEFAULT_MAX_PACK_BYTES
                    if requested_pack_bytes is None
                    else requested_pack_bytes
                )
                level = (
                    DEFAULT_COMPRESSION_LEVEL
                    if requested_compression_level is None
                    else requested_compression_level
                )
                self._validate_pack_limit(pack_limit)
                self._validate_compression_level(level)
                policy = {
                    "compression": {
                        "algorithm": "zlib",
                        "level": level,
                    },
                    "content_digest": "sha256",
                    "content_encoding": "utf-8-strict",
                    "frame_schema": PACK_SCHEMA,
                    "max_pack_bytes": pack_limit,
                    "occurrence_key": list(_OCCURRENCE_FIELDS),
                    "pack_commit_protocol": (
                        "fsync-pack-then-sqlite-full-commit"
                    ),
                    "provenance_storage": {
                        "canonical_encoding": "json-sort-keys-utf8-v1",
                        "compression": "zlib",
                        "compression_level": level,
                        "digest": "sha256",
                    },
                    "token_count_semantics": (
                        "exact-canonical-payload-only-no-framing"
                    ),
                    "token_sequence_encoding": TOKEN_SEQUENCE_ENCODING,
                }
                creator_hash = self._runtime_script_sha256
                sqlite_schema_hash = _sqlite_schema_sha256(self._connection)
                self._connection.executemany(
                    "INSERT INTO settings(key, value) VALUES (?, ?)",
                    (
                        ("schema", STORE_SCHEMA),
                        ("policy", _canonical_json(policy)),
                        ("creator_script_sha256", creator_hash),
                        ("sqlite_schema_sha256", sqlite_schema_hash),
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO stats(
                        singleton,
                        raw_occurrence_bytes,
                        unique_bytes,
                        duplicate_bytes,
                        unique_content_count,
                        occurrence_count,
                        tokenized_unique_content_count,
                        unique_token_sequence_count,
                        exact_unique_payload_tokens,
                        tokenizer_fingerprint
                    ) VALUES (1, 0, 0, 0, 0, 0, 0, 0, 0, NULL)
                    """
                )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

        settings = dict(
            self._connection.execute(
                "SELECT key, value FROM settings"
            ).fetchall()
        )
        if settings.get("schema") != STORE_SCHEMA:
            raise StorePolicyError(
                "unsupported content-store schema "
                f"{settings.get('schema')!r}"
            )
        try:
            policy = json.loads(settings["policy"])
            creator_hash = settings["creator_script_sha256"]
            stored_sqlite_schema_hash = settings["sqlite_schema_sha256"]
        except (KeyError, json.JSONDecodeError) as exc:
            raise StorePolicyError("store settings are incomplete") from exc
        if not isinstance(policy, dict):
            raise StorePolicyError("stored policy is not an object")
        expected_policy_keys = {
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
        }
        if set(policy) != expected_policy_keys:
            raise StorePolicyError("stored policy has an unsupported shape")
        if policy.get("content_digest") != "sha256":
            raise StorePolicyError("stored content digest is unsupported")
        if policy.get("content_encoding") != "utf-8-strict":
            raise StorePolicyError("stored content encoding is unsupported")
        if policy.get("frame_schema") != PACK_SCHEMA:
            raise StorePolicyError("stored frame schema is unsupported")
        if policy.get("occurrence_key") != list(_OCCURRENCE_FIELDS):
            raise StorePolicyError("stored occurrence key is unsupported")
        if (
            policy.get("token_count_semantics")
            != "exact-canonical-payload-only-no-framing"
        ):
            raise StorePolicyError("stored token-count semantics are unsupported")
        if policy.get("token_sequence_encoding") != TOKEN_SEQUENCE_ENCODING:
            raise StorePolicyError(
                "stored token-sequence encoding is unsupported"
            )
        compression = policy.get("compression")
        if not isinstance(compression, dict):
            raise StorePolicyError("stored compression policy is invalid")
        if compression.get("algorithm") != "zlib":
            raise StorePolicyError("stored compression algorithm is unsupported")
        stored_level = compression.get("level")
        provenance_storage = policy.get("provenance_storage")
        if provenance_storage != {
            "canonical_encoding": "json-sort-keys-utf8-v1",
            "compression": "zlib",
            "compression_level": stored_level,
            "digest": "sha256",
        }:
            raise StorePolicyError("stored provenance policy is unsupported")
        stored_pack_limit = policy.get("max_pack_bytes")
        self._validate_compression_level(stored_level)
        self._validate_pack_limit(stored_pack_limit)
        if (
            requested_pack_bytes is not None
            and requested_pack_bytes != stored_pack_limit
        ):
            raise StorePolicyError(
                "requested max_pack_bytes differs from durable store policy"
            )
        if (
            requested_compression_level is not None
            and requested_compression_level != stored_level
        ):
            raise StorePolicyError(
                "requested compression_level differs from durable store policy"
            )
        if (
            not isinstance(creator_hash, str)
            or len(creator_hash) != 64
            or any(character not in "0123456789abcdef" for character in creator_hash)
        ):
            raise StorePolicyError("creator script SHA-256 is invalid")
        actual_sqlite_schema_hash = _sqlite_schema_sha256(self._connection)
        if stored_sqlite_schema_hash != actual_sqlite_schema_hash:
            raise StorePolicyError(
                "SQLite schema differs from its durable schema binding"
            )
        self._policy = policy
        self._creator_script_sha256 = creator_hash
        self._sqlite_schema_sha256 = actual_sqlite_schema_hash

    def _orphan_directory(self) -> Path:
        path = self.root / _ORPHAN_DIRECTORY
        if not path.exists():
            path.mkdir()
            _fsync_directory(self.root)
        if path.is_symlink() or not path.is_dir():
            raise VerificationError(
                f"{_ORPHAN_DIRECTORY} is not a safe quarantine directory"
            )
        return path

    @staticmethod
    def _load_recovery_metadata(path: Path) -> dict[str, object]:
        try:
            encoded = path.read_bytes()
            value = json.loads(encoded.decode("utf-8", errors="strict"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise VerificationError(
                f"invalid recovery metadata: {path.name}"
            ) from exc
        if not isinstance(value, dict):
            raise VerificationError(
                f"recovery metadata is not an object: {path.name}"
            )
        if _canonical_json_bytes(value) + b"\n" != encoded:
            raise VerificationError(
                f"recovery metadata is not canonical: {path.name}"
            )
        return value

    def _quarantine_paths(
        self,
        *,
        original_filename: str,
        kind: str,
        reason: str,
        sha256: str,
        byte_size: int,
        source_offset: int,
        reuse_complete: bool,
    ) -> tuple[Path, Path, dict[str, object], bool]:
        quarantine = self._orphan_directory()
        base = f"{original_filename}.{kind}-{sha256[:16]}"
        for ordinal in range(1_000_000):
            suffix = "" if ordinal == 0 else f"-{ordinal:03d}"
            data_name = f"{base}{suffix}.bin"
            data_path = quarantine / data_name
            metadata_path = quarantine / f"{data_name}.recovery.json"
            record: dict[str, object] = {
                "schema": RECOVERY_SCHEMA,
                "kind": kind,
                "reason": reason,
                "original_filename": original_filename,
                "source_offset": source_offset,
                "byte_size": byte_size,
                "sha256": sha256,
                "quarantined_filename": data_name,
            }
            data_exists = data_path.exists()
            metadata_exists = metadata_path.exists()
            if not data_exists and not metadata_exists:
                return data_path, metadata_path, record, False
            if metadata_exists:
                existing = self._load_recovery_metadata(metadata_path)
                if existing != record:
                    raise VerificationError(
                        f"conflicting recovery metadata: {metadata_path.name}"
                    )
                if not data_exists:
                    return data_path, metadata_path, record, False
                if reuse_complete:
                    if data_path.is_symlink() or not data_path.is_file():
                        raise VerificationError(
                            f"unsafe quarantined artifact: {data_name}"
                        )
                    if (
                        data_path.stat().st_size != byte_size
                        or _sha256_prefix(data_path, byte_size) != sha256
                    ):
                        raise VerificationError(
                            f"quarantined artifact hash mismatch: {data_name}"
                        )
                    return data_path, metadata_path, record, True
                continue
            if data_exists:
                raise VerificationError(
                    f"quarantined artifact lacks metadata: {data_name}"
                )
        raise VerificationError("quarantine filename space is exhausted")

    def _write_recovery_metadata(
        self,
        path: Path,
        record: dict[str, object],
    ) -> None:
        if path.exists():
            if self._load_recovery_metadata(path) != record:
                raise VerificationError(
                    f"conflicting recovery metadata: {path.name}"
                )
            return
        encoded = _canonical_json_bytes(record) + b"\n"
        temporary: Path | None = None
        for ordinal in range(1_000):
            candidate = path.with_name(
                f".{path.name}.tmp-{os.getpid()}-"
                f"{threading.get_ident()}-{ordinal}"
            )
            try:
                with candidate.open("xb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary = candidate
                break
            except FileExistsError:
                continue
        if temporary is None:
            raise VerificationError(
                "cannot allocate recovery metadata temporary file"
            )
        try:
            if path.exists():
                raise VerificationError(
                    f"recovery metadata appeared concurrently: {path.name}"
                )
            os.link(temporary, path)
            _fsync_directory(path.parent)
        finally:
            # This is only a second directory entry for already-durable
            # metadata; it never contains the quarantined pack bytes.
            if temporary.exists():
                temporary.unlink()

    def _quarantine_whole_pack(
        self,
        path: Path,
        *,
        reason: str,
    ) -> dict[str, object]:
        if path.is_symlink() or not path.is_file():
            raise VerificationError(
                f"unknown pack is not a safe regular file: {path.name}"
            )
        size = path.stat().st_size
        if size < len(_PACK_MAGIC):
            raise VerificationError(
                f"unknown pack is too short to quarantine: {path.name}"
            )
        with path.open("rb") as handle:
            if handle.read(len(_PACK_MAGIC)) != _PACK_MAGIC:
                raise VerificationError(
                    f"unknown pack has an invalid header: {path.name}"
                )
            os.fsync(handle.fileno())
        digest = _sha256_prefix(path, size)
        destination, metadata_path, record, complete = (
            self._quarantine_paths(
                original_filename=path.name,
                kind="whole-pack",
                reason=reason,
                sha256=digest,
                byte_size=size,
                source_offset=0,
                reuse_complete=False,
            )
        )
        if complete:
            raise AssertionError("whole-pack quarantine cannot be pre-complete")
        self._write_recovery_metadata(metadata_path, record)
        os.replace(path, destination)
        _fsync_directory(destination.parent)
        _fsync_directory(self.root)
        self._recovery_events.append(record)
        return record

    def _quarantine_pack_tail(
        self,
        path: Path,
        *,
        committed_end: int,
        reason: str,
    ) -> dict[str, object]:
        size = path.stat().st_size
        tail_size = size - committed_end
        if tail_size <= 0:
            raise ValueError("pack does not have an orphan tail")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            source.seek(committed_end)
            remaining = tail_size
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    raise VerificationError(
                        f"{path.name} orphan tail is truncated"
                    )
                digest.update(block)
                remaining -= len(block)
        tail_digest = digest.hexdigest()
        destination, metadata_path, record, complete = (
            self._quarantine_paths(
                original_filename=path.name,
                kind="pack-tail",
                reason=reason,
                sha256=tail_digest,
                byte_size=tail_size,
                source_offset=committed_end,
                reuse_complete=True,
            )
        )
        self._write_recovery_metadata(metadata_path, record)
        if not complete:
            temporary: Path | None = None
            for ordinal in range(1_000):
                candidate = destination.with_name(
                    f".{destination.name}.tmp-{os.getpid()}-"
                    f"{threading.get_ident()}-{ordinal}"
                )
                if not candidate.exists():
                    temporary = candidate
                    break
            if temporary is None:
                raise VerificationError(
                    "cannot allocate orphan-tail temporary file"
                )
            try:
                with path.open("rb") as source, temporary.open("xb") as target:
                    source.seek(committed_end)
                    remaining = tail_size
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            raise VerificationError(
                                f"{path.name} orphan tail changed during recovery"
                            )
                        target.write(block)
                        remaining -= len(block)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, destination)
                _fsync_directory(destination.parent)
            finally:
                # The authoritative tail still exists in the source pack until
                # after the durable quarantine copy is complete.
                if temporary.exists():
                    temporary.unlink()
        if (
            destination.stat().st_size != tail_size
            or _sha256_prefix(destination, tail_size) != tail_digest
        ):
            raise VerificationError(
                f"quarantined tail verification failed: {destination.name}"
            )
        self._recovery_events.append(record)
        return record

    def recovery_records(self) -> list[dict[str, object]]:
        quarantine = self.root / _ORPHAN_DIRECTORY
        if not quarantine.exists():
            return []
        if quarantine.is_symlink() or not quarantine.is_dir():
            raise VerificationError("unsafe orphan quarantine directory")
        records: list[dict[str, object]] = []
        referenced: set[str] = set()
        for metadata_path in sorted(
            quarantine.glob("*.recovery.json")
        ):
            record = self._load_recovery_metadata(metadata_path)
            if record.get("schema") != RECOVERY_SCHEMA:
                raise VerificationError(
                    f"unsupported recovery schema: {metadata_path.name}"
                )
            filename = record.get("quarantined_filename")
            byte_size = record.get("byte_size")
            digest = record.get("sha256")
            kind = record.get("kind")
            reason = record.get("reason")
            original_filename = record.get("original_filename")
            source_offset = record.get("source_offset")
            if (
                kind not in {"whole-pack", "pack-tail"}
                or not isinstance(reason, str)
                or not reason
                or not isinstance(original_filename, str)
                or Path(original_filename).name != original_filename
                or isinstance(source_offset, bool)
                or not isinstance(source_offset, int)
                or source_offset < 0
                or not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or isinstance(byte_size, bool)
                or not isinstance(byte_size, int)
                or byte_size < 0
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
            ):
                raise VerificationError(
                    f"invalid recovery record: {metadata_path.name}"
                )
            artifact = quarantine / filename
            if artifact.is_symlink() or not artifact.is_file():
                raise VerificationError(
                    f"recovery artifact is missing: {filename}"
                )
            if (
                artifact.stat().st_size != byte_size
                or _sha256_prefix(artifact, byte_size) != digest
            ):
                raise VerificationError(
                    f"recovery artifact hash mismatch: {filename}"
                )
            referenced.add(filename)
            records.append(record)
        for artifact in quarantine.iterdir():
            if (
                artifact.is_file()
                and not artifact.name.startswith(".")
                and not artifact.name.endswith(".recovery.json")
                and artifact.name not in referenced
            ):
                raise VerificationError(
                    f"unmanifested recovery artifact: {artifact.name}"
                )
        return records

    def _recover_pack_tails(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            rows = self._connection.execute(
                """
                SELECT pack_id, filename, committed_end
                FROM packs
                ORDER BY pack_id
                """
            ).fetchall()
            known_names = {str(row["filename"]) for row in rows}
            for row in rows:
                filename = str(row["filename"])
                expected_filename = f"pack-{int(row['pack_id']):08d}.cicp"
                if filename != expected_filename:
                    raise VerificationError(
                        f"pack {row['pack_id']} has unsafe filename {filename!r}"
                    )
                path = self.root / filename
                if path.is_symlink() or not path.is_file():
                    raise VerificationError(
                        f"committed pack is missing: {path.name}"
                    )
                size = path.stat().st_size
                committed_end = int(row["committed_end"])
                if size < committed_end:
                    raise VerificationError(
                        f"{path.name} is shorter than committed_end "
                        f"({size} < {committed_end})"
                    )
                with path.open("r+b") as handle:
                    if handle.read(len(_PACK_MAGIC)) != _PACK_MAGIC:
                        raise VerificationError(
                            f"{path.name} has an invalid pack header"
                        )
                    if size > committed_end:
                        self._quarantine_pack_tail(
                            path,
                            committed_end=committed_end,
                            reason="uncommitted-tail-found-on-open",
                        )
                        handle.truncate(committed_end)
                        handle.flush()
                        os.fsync(handle.fileno())
            for path in sorted(self.root.glob(_PACK_GLOB)):
                if path.name not in known_names:
                    self._quarantine_whole_pack(
                        path,
                        reason="unindexed-pack-found-on-open",
                    )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _canonical_content(content: bytes | str) -> bytes:
        if isinstance(content, str):
            return content.encode("utf-8", errors="strict")
        if isinstance(content, bytes):
            # Strict decode rejects overlong encodings, surrogates, and all
            # other byte sequences that are not canonical UTF-8.
            return content.decode("utf-8", errors="strict").encode("utf-8")
        raise TypeError("content must be bytes or str")

    @staticmethod
    def _content_sha256(content: bytes) -> str:
        """Production digest seam; subclasses may force a collision in tests."""

        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _normalize_occurrence_key(
        occurrence_key: OccurrenceKey
        | Mapping[str, object]
        | Sequence[object],
    ) -> OccurrenceKey:
        if isinstance(occurrence_key, OccurrenceKey):
            values = occurrence_key.as_dict()
        elif isinstance(occurrence_key, Mapping):
            missing = set(_OCCURRENCE_FIELDS) - set(occurrence_key)
            extra = set(occurrence_key) - set(_OCCURRENCE_FIELDS)
            if missing or extra:
                raise ValueError(
                    "occurrence_key must contain exactly "
                    f"{', '.join(_OCCURRENCE_FIELDS)}"
                )
            values = dict(occurrence_key)
        elif (
            isinstance(occurrence_key, Sequence)
            and not isinstance(occurrence_key, (str, bytes))
            and len(occurrence_key) == len(_OCCURRENCE_FIELDS)
        ):
            values = dict(zip(_OCCURRENCE_FIELDS, occurrence_key, strict=True))
        else:
            raise TypeError(
                "occurrence_key must be an OccurrenceKey, mapping, or 5-tuple"
            )

        normalized_strings: dict[str, str] = {}
        for field in ("repo", "run_attempt", "job", "step"):
            value = values[field]
            if field == "run_attempt" and isinstance(value, int) and not isinstance(
                value, bool
            ):
                value = str(value)
            if not isinstance(value, str) or not value:
                raise ValueError(f"occurrence_key.{field} must be non-empty")
            normalized_strings[field] = value
        ordinal = values["chunk_ordinal"]
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
        ):
            raise ValueError(
                "occurrence_key.chunk_ordinal must be a non-negative integer"
            )
        return OccurrenceKey(
            repo=normalized_strings["repo"],
            run_attempt=normalized_strings["run_attempt"],
            job=normalized_strings["job"],
            step=normalized_strings["step"],
            chunk_ordinal=ordinal,
        )

    @staticmethod
    def _normalize_token_metadata(
        token_count: int | None,
        tokenizer_fingerprint: str | None,
        token_sequence_sha256: str | None,
    ) -> tuple[int | None, str | None, str | None]:
        supplied = (
            token_count is not None,
            tokenizer_fingerprint is not None,
            token_sequence_sha256 is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError(
                "token_count, tokenizer_fingerprint, and "
                "token_sequence_sha256 must be supplied together"
            )
        if token_count is None:
            return None, None, None
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError("token_count must be a non-negative integer")
        if (
            not isinstance(tokenizer_fingerprint, str)
            or not tokenizer_fingerprint
        ):
            raise ValueError("tokenizer_fingerprint must be non-empty")
        if (
            not isinstance(token_sequence_sha256, str)
            or len(token_sequence_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in token_sequence_sha256
            )
        ):
            raise ValueError(
                "token_sequence_sha256 must be 64 lowercase hexadecimal "
                "characters"
            )
        return token_count, tokenizer_fingerprint, token_sequence_sha256

    def _begin_write(self) -> None:
        if self._closed:
            raise ContentStoreError("content store is closed")
        self._connection.execute("BEGIN IMMEDIATE")

    def _durable_commit(self) -> None:
        self._connection.execute("COMMIT")
        _fsync_file(self.db_path)
        _fsync_directory(self.root)

    def _rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.execute("ROLLBACK")

    def _new_pack(self) -> tuple[int, str]:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(pack_id), 0) + 1 AS next_id FROM packs"
        ).fetchone()
        assert row is not None
        pack_id = int(row["next_id"])
        filename = f"pack-{pack_id:08d}.cicp"
        path = self.root / filename
        if path.exists():
            raise VerificationError(
                f"unindexed pack path already exists: {filename}"
            )
        with path.open("xb") as handle:
            handle.write(_PACK_MAGIC)
            handle.flush()
        try:
            self._connection.execute(
                """
                INSERT INTO packs(
                    pack_id, filename, committed_end, content_count
                ) VALUES (?, ?, ?, 0)
                """,
                (pack_id, filename, len(_PACK_MAGIC)),
            )
        except BaseException:
            # SQLite rejected the pack row before any frame could be appended;
            # this path contains only the newly-created eight-byte header.
            path.unlink()
            raise
        return pack_id, filename

    def _append_frame(
        self,
        *,
        filename: str,
        committed_end: int,
        digest: str,
        content: bytes,
        compressed: bytes,
        fsync: bool = True,
    ) -> int:
        header = _FRAME_HEADER.pack(
            _FRAME_MAGIC,
            bytes.fromhex(digest),
            len(content),
            len(compressed),
        )
        path = self.root / filename
        with path.open("r+b") as handle:
            handle.seek(0, os.SEEK_END)
            actual_end = handle.tell()
            if actual_end != committed_end:
                raise VerificationError(
                    f"{filename} append boundary changed "
                    f"({actual_end} != {committed_end})"
                )
            handle.write(header)
            handle.write(compressed)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        return len(header) + len(compressed)

    def _restore_pack_after_failed_transaction(
        self,
        *,
        pack_id: int,
        filename: str,
    ) -> None:
        row = self._connection.execute(
            "SELECT committed_end FROM packs WHERE pack_id = ?",
            (pack_id,),
        ).fetchone()
        path = self.root / filename
        if row is None:
            if path.exists():
                self._quarantine_whole_pack(
                    path,
                    reason="pack-from-rolled-back-transaction",
                )
            return
        committed_end = int(row["committed_end"])
        if path.is_symlink() or not path.is_file():
            raise VerificationError(
                f"cannot restore unsafe pack path: {filename}"
            )
        size = path.stat().st_size
        if size > committed_end:
            self._quarantine_pack_tail(
                path,
                committed_end=committed_end,
                reason="tail-from-rolled-back-transaction",
            )
            with path.open("r+b") as handle:
                handle.truncate(committed_end)
                handle.flush()
                os.fsync(handle.fileno())

    def _read_content_row(self, row: sqlite3.Row) -> bytes:
        pack_row = self._connection.execute(
            "SELECT filename, committed_end FROM packs WHERE pack_id = ?",
            (int(row["pack_id"]),),
        ).fetchone()
        if pack_row is None:
            raise VerificationError(
                f"content {row['sha256']} references a missing pack"
            )
        offset = int(row["offset"])
        compressed_size = int(row["compressed_size"])
        frame_size = int(row["frame_size"])
        expected_frame_size = _FRAME_HEADER.size + compressed_size
        if frame_size != expected_frame_size:
            raise VerificationError(
                f"content {row['sha256']} has an invalid frame size"
            )
        if offset + frame_size > int(pack_row["committed_end"]):
            raise VerificationError(
                f"content {row['sha256']} exceeds its committed pack boundary"
            )
        path = self.root / str(pack_row["filename"])
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                header = handle.read(_FRAME_HEADER.size)
                compressed = handle.read(compressed_size)
        except OSError as exc:
            raise VerificationError(
                f"cannot read content {row['sha256']}: {exc}"
            ) from exc
        if len(header) != _FRAME_HEADER.size or len(compressed) != compressed_size:
            raise VerificationError(
                f"content {row['sha256']} has a truncated frame"
            )
        magic, header_digest, raw_size, header_compressed_size = (
            _FRAME_HEADER.unpack(header)
        )
        if magic != _FRAME_MAGIC:
            raise VerificationError(
                f"content {row['sha256']} has an invalid frame header"
            )
        if header_digest.hex() != str(row["sha256"]):
            raise VerificationError(
                f"content {row['sha256']} frame digest disagrees with SQLite"
            )
        if raw_size != int(row["raw_size"]):
            raise VerificationError(
                f"content {row['sha256']} frame size disagrees with SQLite"
            )
        if header_compressed_size != compressed_size:
            raise VerificationError(
                f"content {row['sha256']} compressed size disagrees with SQLite"
            )
        try:
            decompressor = zlib.decompressobj()
            content = decompressor.decompress(compressed)
            content += decompressor.flush()
        except zlib.error as exc:
            raise VerificationError(
                f"content {row['sha256']} cannot be decompressed"
            ) from exc
        if (
            not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            raise VerificationError(
                f"content {row['sha256']} has an invalid zlib stream"
            )
        if len(content) != int(row["raw_size"]):
            raise VerificationError(
                f"content {row['sha256']} decompressed size mismatch"
            )
        try:
            canonical = self._canonical_content(content)
        except UnicodeError as exc:
            raise VerificationError(
                f"content {row['sha256']} is not canonical UTF-8"
            ) from exc
        if canonical != content:
            raise VerificationError(
                f"content {row['sha256']} is not canonical UTF-8"
            )
        actual_digest = self._content_sha256(content)
        if actual_digest != str(row["sha256"]):
            raise VerificationError(
                f"content hash mismatch for {row['sha256']}: {actual_digest}"
            )
        return content

    def _assert_same_content(
        self,
        row: sqlite3.Row,
        candidate: bytes,
        digest: str,
    ) -> None:
        if int(row["raw_size"]) != len(candidate):
            raise HashCollisionError(
                f"SHA-256 collision detected for {digest}: size differs"
            )
        if self._read_content_row(row) != candidate:
            raise HashCollisionError(
                f"SHA-256 collision detected for {digest}: bytes differ"
            )

    @staticmethod
    def _decode_provenance_row(
        row: sqlite3.Row,
    ) -> tuple[dict[str, object], bytes]:
        provenance_digest = str(row["provenance_sha256"])
        if (
            len(provenance_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in provenance_digest
            )
        ):
            raise VerificationError(
                "invalid provenance SHA-256 in SQLite"
            )
        compressed = bytes(row["provenance_zlib"])
        try:
            decompressor = zlib.decompressobj()
            encoded = decompressor.decompress(compressed)
            encoded += decompressor.flush()
        except zlib.error as exc:
            raise VerificationError(
                "occurrence provenance cannot be decompressed"
            ) from exc
        if (
            not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            raise VerificationError(
                "occurrence provenance has an invalid zlib stream"
            )
        if len(encoded) != int(row["provenance_raw_size"]):
            raise VerificationError(
                "occurrence provenance decompressed size mismatch"
            )
        if hashlib.sha256(encoded).hexdigest() != provenance_digest:
            raise VerificationError(
                "occurrence provenance SHA-256 mismatch"
            )
        try:
            provenance = json.loads(encoded.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise VerificationError(
                "occurrence provenance is not canonical UTF-8 JSON"
            ) from exc
        if not isinstance(provenance, dict):
            raise VerificationError(
                "occurrence provenance is not a JSON object"
            )
        if _canonical_json_bytes(provenance) != encoded:
            raise VerificationError(
                "occurrence provenance is not canonical JSON"
            )
        return provenance, encoded

    def _validate_store_tokenizer(self, fingerprint: str) -> None:
        stats = self._connection.execute(
            "SELECT tokenizer_fingerprint FROM stats WHERE singleton = 1"
        ).fetchone()
        if stats is None:
            raise VerificationError("stats row is missing")
        stored = stats["tokenizer_fingerprint"]
        if stored is not None and stored != fingerprint:
            raise ContentMetadataConflictError(
                "tokenizer fingerprint mismatch: "
                f"store is bound to {stored!r}, got {fingerprint!r}"
            )

    def _ensure_token_sequence(
        self,
        *,
        token_sequence_sha256: str,
        token_count: int,
        tokenizer_fingerprint: str,
    ) -> bool:
        """Bind one canonical token sequence, returning whether it was new."""

        self._validate_store_tokenizer(tokenizer_fingerprint)
        row = self._connection.execute(
            """
            SELECT token_count, tokenizer_fingerprint
            FROM token_sequences
            WHERE token_sequence_sha256 = ?
            """,
            (token_sequence_sha256,),
        ).fetchone()
        if row is not None:
            if (
                int(row["token_count"]) != token_count
                or row["tokenizer_fingerprint"] != tokenizer_fingerprint
            ):
                raise ContentMetadataConflictError(
                    "token-sequence hash has conflicting length or tokenizer "
                    f"binding: {token_sequence_sha256}"
                )
            return False
        self._connection.execute(
            """
            INSERT INTO token_sequences(
                token_sequence_sha256, token_count, tokenizer_fingerprint
            ) VALUES (?, ?, ?)
            """,
            (
                token_sequence_sha256,
                token_count,
                tokenizer_fingerprint,
            ),
        )
        return True

    def _reconcile_token_metadata(
        self,
        row: sqlite3.Row,
        token_count: int | None,
        tokenizer_fingerprint: str | None,
        token_sequence_sha256: str | None,
    ) -> bool:
        if token_count is None:
            return False
        assert tokenizer_fingerprint is not None
        assert token_sequence_sha256 is not None
        stored_count = row["token_count"]
        stored_fingerprint = row["tokenizer_fingerprint"]
        stored_sequence_sha256 = row["token_sequence_sha256"]
        if stored_count is not None:
            if (
                int(stored_count) != token_count
                or stored_fingerprint != tokenizer_fingerprint
                or stored_sequence_sha256 != token_sequence_sha256
            ):
                raise ContentMetadataConflictError(
                    f"token metadata conflict for content {row['sha256']}"
                )
            return False
        sequence_added = self._ensure_token_sequence(
            token_sequence_sha256=token_sequence_sha256,
            token_count=token_count,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        self._connection.execute(
            """
            UPDATE contents
            SET token_count = ?,
                tokenizer_fingerprint = ?,
                token_sequence_sha256 = ?
            WHERE sha256 = ?
            """,
            (
                token_count,
                tokenizer_fingerprint,
                token_sequence_sha256,
                str(row["sha256"]),
            ),
        )
        self._connection.execute(
            """
            UPDATE stats
            SET tokenized_unique_content_count =
                    tokenized_unique_content_count + 1,
                unique_token_sequence_count =
                    unique_token_sequence_count + ?,
                exact_unique_payload_tokens =
                    exact_unique_payload_tokens + ?,
                tokenizer_fingerprint =
                    COALESCE(tokenizer_fingerprint, ?)
            WHERE singleton = 1
            """,
            (
                1 if sequence_added else 0,
                token_count if sequence_added else 0,
                tokenizer_fingerprint,
            ),
        )
        return True

    def _add_chunk_in_transaction(
        self,
        content: bytes | str,
        provenance: dict[str, object],
        occurrence_key: OccurrenceKey
        | Mapping[str, object]
        | Sequence[object],
        *,
        token_count: int | None = None,
        tokenizer_fingerprint: str | None = None,
        token_sequence_sha256: str | None = None,
        touched_packs: dict[int, str],
    ) -> dict[str, object]:
        canonical_content = self._canonical_content(content)
        digest = self._content_sha256(canonical_content)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("content digest implementation returned invalid SHA-256")
        if not isinstance(provenance, dict):
            raise TypeError("provenance must be a dict")
        provenance_bytes = _canonical_json_bytes(provenance)
        provenance_digest = hashlib.sha256(provenance_bytes).hexdigest()
        key = self._normalize_occurrence_key(occurrence_key)
        (
            token_count,
            tokenizer_fingerprint,
            token_sequence_sha256,
        ) = self._normalize_token_metadata(
            token_count,
            tokenizer_fingerprint,
            token_sequence_sha256,
        )

        occurrence = self._connection.execute(
            """
            SELECT content_sha256, provenance_sha256,
                   provenance_raw_size, provenance_zlib
            FROM occurrences
            WHERE repo = ? AND run_attempt = ? AND job = ?
              AND step = ? AND chunk_ordinal = ?
            """,
            key.as_tuple(),
        ).fetchone()
        content_row = self._connection.execute(
            "SELECT * FROM contents WHERE sha256 = ?",
            (digest,),
        ).fetchone()

        if occurrence is not None:
            if (
                occurrence["content_sha256"] != digest
                or occurrence["provenance_sha256"] != provenance_digest
                or int(occurrence["provenance_raw_size"])
                != len(provenance_bytes)
            ):
                raise OccurrenceConflictError(
                    f"conflicting replay for occurrence {key.as_dict()}"
                )
            _, stored_provenance_bytes = self._decode_provenance_row(occurrence)
            if stored_provenance_bytes != provenance_bytes:
                raise HashCollisionError(
                    "SHA-256 collision detected for occurrence provenance "
                    f"{provenance_digest}"
                )
            if content_row is None:
                raise VerificationError(
                    "occurrence references content missing from the index"
                )
            self._assert_same_content(
                content_row,
                canonical_content,
                digest,
            )
            metadata_added = self._reconcile_token_metadata(
                content_row,
                token_count,
                tokenizer_fingerprint,
                token_sequence_sha256,
            )
            return {
                "sha256": digest,
                "raw_size": len(canonical_content),
                "content_added": False,
                "occurrence_added": False,
                "token_metadata_added": metadata_added,
                "occurrence_key": key.as_dict(),
            }

        provenance_compressed = zlib.compress(
            provenance_bytes,
            level=self.compression_level,
        )
        if content_row is not None:
            self._assert_same_content(
                content_row,
                canonical_content,
                digest,
            )
            metadata_added = self._reconcile_token_metadata(
                content_row,
                token_count,
                tokenizer_fingerprint,
                token_sequence_sha256,
            )
            self._connection.execute(
                """
                INSERT INTO occurrences(
                    repo, run_attempt, job, step, chunk_ordinal,
                    content_sha256, provenance_sha256,
                    provenance_raw_size, provenance_zlib
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    *key.as_tuple(),
                    digest,
                    provenance_digest,
                    len(provenance_bytes),
                    sqlite3.Binary(provenance_compressed),
                ),
            )
            self._connection.execute(
                """
                UPDATE stats
                SET raw_occurrence_bytes = raw_occurrence_bytes + ?,
                    duplicate_bytes = duplicate_bytes + ?,
                    occurrence_count = occurrence_count + 1
                WHERE singleton = 1
                """,
                (len(canonical_content), len(canonical_content)),
            )
            return {
                "sha256": digest,
                "raw_size": len(canonical_content),
                "content_added": False,
                "occurrence_added": True,
                "token_metadata_added": metadata_added,
                "occurrence_key": key.as_dict(),
            }

        if token_count is not None:
            assert tokenizer_fingerprint is not None
            assert token_sequence_sha256 is not None
            token_sequence_added = self._ensure_token_sequence(
                token_sequence_sha256=token_sequence_sha256,
                token_count=token_count,
                tokenizer_fingerprint=tokenizer_fingerprint,
            )
        else:
            token_sequence_added = False
        compressed = zlib.compress(
            canonical_content,
            level=self.compression_level,
        )
        frame_size = _FRAME_HEADER.size + len(compressed)
        if len(_PACK_MAGIC) + frame_size > self.max_pack_bytes:
            raise ValueError(
                "compressed chunk frame exceeds max_pack_bytes; "
                "split the chunk before storing it"
            )
        pack = self._connection.execute(
            """
            SELECT pack_id, filename, committed_end
            FROM packs
            ORDER BY pack_id DESC
            LIMIT 1
            """
        ).fetchone()
        if (
            pack is None
            or int(pack["committed_end"]) + frame_size
            > self.max_pack_bytes
        ):
            pack_id, filename = self._new_pack()
            committed_end = len(_PACK_MAGIC)
        else:
            pack_id = int(pack["pack_id"])
            filename = str(pack["filename"])
            committed_end = int(pack["committed_end"])
        touched_packs[pack_id] = filename
        written_frame_size = self._append_frame(
            filename=filename,
            committed_end=committed_end,
            digest=digest,
            content=canonical_content,
            compressed=compressed,
            fsync=False,
        )
        self._connection.execute(
            """
            INSERT INTO contents(
                sha256, raw_size, pack_id, offset, frame_size,
                compressed_size, token_count, tokenizer_fingerprint,
                token_sequence_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                digest,
                len(canonical_content),
                pack_id,
                committed_end,
                written_frame_size,
                len(compressed),
                token_count,
                tokenizer_fingerprint,
                token_sequence_sha256,
            ),
        )
        self._connection.execute(
            """
            INSERT INTO occurrences(
                repo, run_attempt, job, step, chunk_ordinal,
                content_sha256, provenance_sha256,
                provenance_raw_size, provenance_zlib
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                *key.as_tuple(),
                digest,
                provenance_digest,
                len(provenance_bytes),
                sqlite3.Binary(provenance_compressed),
            ),
        )
        self._connection.execute(
            """
            UPDATE packs
            SET committed_end = ?,
                content_count = content_count + 1
            WHERE pack_id = ?
            """,
            (committed_end + written_frame_size, pack_id),
        )
        tokenized_increment = 1 if token_count is not None else 0
        token_increment = token_count if token_count is not None else 0
        self._connection.execute(
            """
            UPDATE stats
            SET raw_occurrence_bytes = raw_occurrence_bytes + ?,
                unique_bytes = unique_bytes + ?,
                unique_content_count = unique_content_count + 1,
                occurrence_count = occurrence_count + 1,
                tokenized_unique_content_count =
                    tokenized_unique_content_count + ?,
                unique_token_sequence_count =
                    unique_token_sequence_count + ?,
                exact_unique_payload_tokens =
                    exact_unique_payload_tokens + ?,
                tokenizer_fingerprint = CASE
                    WHEN ? IS NULL THEN tokenizer_fingerprint
                    ELSE COALESCE(tokenizer_fingerprint, ?)
                END
            WHERE singleton = 1
            """,
            (
                len(canonical_content),
                len(canonical_content),
                tokenized_increment,
                1 if token_sequence_added else 0,
                token_increment if token_sequence_added else 0,
                tokenizer_fingerprint,
                tokenizer_fingerprint,
            ),
        )
        return {
            "sha256": digest,
            "raw_size": len(canonical_content),
            "content_added": True,
            "occurrence_added": True,
            "token_metadata_added": token_count is not None,
            "occurrence_key": key.as_dict(),
        }

    def add_chunks(
        self,
        records: Iterable[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Atomically add a job/step batch with one commit and pack fsync set.

        Each record must contain ``content``, ``provenance``, and
        ``occurrence_key``.  It may also contain ``token_count``,
        ``tokenizer_fingerprint``, and ``token_sequence_sha256``.  A conflict
        rolls back the entire batch.  Pack files are each fsynced once before
        the single SQLite FULL commit.
        """

        if self._runtime_script_sha256 != self._creator_script_sha256:
            raise StorePolicyError(
                "writer script differs from the script bound at store "
                "creation; start a new store to avoid mixed producer logic"
            )
        required = {"content", "provenance", "occurrence_key"}
        allowed = required | {
            "token_count",
            "tokenizer_fingerprint",
            "token_sequence_sha256",
        }
        touched_packs: dict[int, str] = {}
        results: list[dict[str, object]] = []
        with self._lock:
            self._begin_write()
            try:
                for record in records:
                    if not isinstance(record, Mapping):
                        raise TypeError("every batch record must be a mapping")
                    missing = required - set(record)
                    extra = set(record) - allowed
                    if missing or extra:
                        raise ValueError(
                            "batch records require content, provenance, and "
                            "occurrence_key, with only documented token fields"
                        )
                    results.append(
                        self._add_chunk_in_transaction(
                            record["content"],  # type: ignore[arg-type]
                            record["provenance"],  # type: ignore[arg-type]
                            record["occurrence_key"],  # type: ignore[arg-type]
                            token_count=record.get("token_count"),  # type: ignore[arg-type]
                            tokenizer_fingerprint=record.get(  # type: ignore[arg-type]
                                "tokenizer_fingerprint"
                            ),
                            token_sequence_sha256=record.get(  # type: ignore[arg-type]
                                "token_sequence_sha256"
                            ),
                            touched_packs=touched_packs,
                        )
                    )
                for pack_id in sorted(touched_packs):
                    _fsync_file(self.root / touched_packs[pack_id])
                if touched_packs:
                    _fsync_directory(self.root)
                self._durable_commit()
                return results
            except BaseException:
                self._rollback()
                for pack_id, filename in sorted(touched_packs.items()):
                    self._restore_pack_after_failed_transaction(
                        pack_id=pack_id,
                        filename=filename,
                    )
                raise

    def add_chunk(
        self,
        content: bytes | str,
        provenance: dict[str, object],
        occurrence_key: OccurrenceKey
        | Mapping[str, object]
        | Sequence[object],
        *,
        token_count: int | None = None,
        tokenizer_fingerprint: str | None = None,
        token_sequence_sha256: str | None = None,
    ) -> dict[str, object]:
        """Add one occurrence via the same durable path as a one-record batch.

        ``token_count`` counts only the canonical payload.  The corresponding
        ``token_sequence_sha256`` must use :func:`hash_token_sequence`; no BOS,
        domain delimiter, padding, packing separator, or downstream framing is
        part of this count.
        """

        return self.add_chunks(
            (
                {
                    "content": content,
                    "provenance": provenance,
                    "occurrence_key": occurrence_key,
                    "token_count": token_count,
                    "tokenizer_fingerprint": tokenizer_fingerprint,
                    "token_sequence_sha256": token_sequence_sha256,
                },
            )
        )[0]

    def read_chunk(self, sha256: str, *, as_text: bool = False) -> bytes | str:
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        with self._lock:
            if self._closed:
                raise ContentStoreError("content store is closed")
            row = self._connection.execute(
                "SELECT * FROM contents WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
            if row is None:
                raise KeyError(sha256)
            content = self._read_content_row(row)
        return content.decode("utf-8") if as_text else content

    read = read_chunk

    def iter_contents(
        self,
        *,
        include_content: bool = False,
    ) -> Iterator[dict[str, object]]:
        with self._lock:
            if self._closed:
                raise ContentStoreError("content store is closed")
            cursor = self._connection.execute(
                """
                SELECT *
                FROM contents
                ORDER BY sha256
                """
            )
            for row in cursor:
                record: dict[str, object] = {
                    "sha256": str(row["sha256"]),
                    "raw_size": int(row["raw_size"]),
                    "pack_id": int(row["pack_id"]),
                    "offset": int(row["offset"]),
                    "frame_size": int(row["frame_size"]),
                    "compressed_size": int(row["compressed_size"]),
                    "token_count": (
                        None
                        if row["token_count"] is None
                        else int(row["token_count"])
                    ),
                    "tokenizer_fingerprint": row["tokenizer_fingerprint"],
                    "token_sequence_sha256": row["token_sequence_sha256"],
                }
                if include_content:
                    record["content"] = self._read_content_row(row)
                yield record

    def iter_chunks(self) -> Iterator[dict[str, object]]:
        yield from self.iter_contents(include_content=True)

    def iter_occurrences(self) -> Iterator[dict[str, object]]:
        with self._lock:
            if self._closed:
                raise ContentStoreError("content store is closed")
            cursor = self._connection.execute(
                """
                SELECT repo, run_attempt, job, step, chunk_ordinal,
                       content_sha256, provenance_sha256,
                       provenance_raw_size, provenance_zlib
                FROM occurrences
                ORDER BY repo, run_attempt, job, step, chunk_ordinal
                """
            )
            for row in cursor:
                provenance, _ = self._decode_provenance_row(row)
                yield {
                    "occurrence_key": {
                        "repo": str(row["repo"]),
                        "run_attempt": str(row["run_attempt"]),
                        "job": str(row["job"]),
                        "step": str(row["step"]),
                        "chunk_ordinal": int(row["chunk_ordinal"]),
                    },
                    "content_sha256": str(row["content_sha256"]),
                    "provenance_sha256": str(row["provenance_sha256"]),
                    "provenance": provenance,
                }

    @staticmethod
    def _counters_from_row(row: sqlite3.Row) -> dict[str, object]:
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
            "unique_token_sequence_count": int(
                row["unique_token_sequence_count"]
            ),
            "tokenizer_fingerprint": fingerprint,
            "exact_unique_payload_tokens": (
                int(row["exact_unique_payload_tokens"])
                if all_tokenized
                else None
            ),
        }

    def status(self) -> dict[str, object]:
        with self._lock:
            if self._closed:
                raise ContentStoreError("content store is closed")
            row = self._connection.execute(
                "SELECT * FROM stats WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise VerificationError("stats row is missing")
            pack_row = self._connection.execute(
                """
                SELECT COUNT(*) AS pack_count,
                       COALESCE(SUM(committed_end), 0) AS committed_pack_bytes
                FROM packs
                """
            ).fetchone()
            assert pack_row is not None
            quarantine = self.root / _ORPHAN_DIRECTORY
            quarantined_count = (
                len(list(quarantine.glob("*.recovery.json")))
                if quarantine.is_dir() and not quarantine.is_symlink()
                else 0
            )
            return {
                "schema": STORE_SCHEMA,
                "policy": self.policy,
                "script_sha256": self.script_sha256,
                "counters": self._counters_from_row(row),
                "pack_count": int(pack_row["pack_count"]),
                "committed_pack_bytes": int(pack_row["committed_pack_bytes"]),
                "quarantined_orphan_count": quarantined_count,
                "recovery_events_on_open": list(self._recovery_events),
            }

    def _content_set_digest(self) -> str:
        cursor = self._connection.execute(
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
                        None
                        if row["token_count"] is None
                        else int(row["token_count"])
                    ),
                    "tokenizer_fingerprint": row["tokenizer_fingerprint"],
                    "token_sequence_sha256": row["token_sequence_sha256"],
                }
                for row in cursor
            ),
        )

    def _token_sequence_set_digest(self) -> str:
        cursor = self._connection.execute(
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
                    "token_sequence_sha256": str(
                        row["token_sequence_sha256"]
                    ),
                    "token_count": int(row["token_count"]),
                    "tokenizer_fingerprint": str(
                        row["tokenizer_fingerprint"]
                    ),
                    "encoding": TOKEN_SEQUENCE_ENCODING,
                }
                for row in cursor
            ),
        )

    def _occurrence_set_digest(self) -> str:
        cursor = self._connection.execute(
            """
            SELECT repo, run_attempt, job, step, chunk_ordinal,
                   content_sha256, provenance_sha256,
                   provenance_raw_size, provenance_zlib
            FROM occurrences
            ORDER BY repo, run_attempt, job, step, chunk_ordinal
            """
        )

        def records() -> Iterator[object]:
            for row in cursor:
                provenance, _ = self._decode_provenance_row(row)
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

        return _hash_records(
            "cppmega-ci-occurrence-set-v1",
            records(),
        )

    def _sqlite_logical_digest(self) -> str:
        def records() -> Iterator[object]:
            for row in self._connection.execute(
                "SELECT key, value FROM settings ORDER BY key"
            ):
                yield ["settings", str(row["key"]), str(row["value"])]
            for row in self._connection.execute(
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
            for row in self._connection.execute(
                """
                SELECT token_sequence_sha256, token_count,
                       tokenizer_fingerprint
                FROM token_sequences
                ORDER BY token_sequence_sha256
                """
            ):
                yield [
                    "token_sequences",
                    str(row["token_sequence_sha256"]),
                    int(row["token_count"]),
                    str(row["tokenizer_fingerprint"]),
                ]
            for row in self._connection.execute(
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
                    (
                        None
                        if row["token_count"] is None
                        else int(row["token_count"])
                    ),
                    row["tokenizer_fingerprint"],
                    row["token_sequence_sha256"],
                ]
            for row in self._connection.execute(
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
                    hashlib.sha256(bytes(row["provenance_zlib"])).hexdigest(),
                ]
            row = self._connection.execute(
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

    def _verify_locked(self) -> dict[str, object]:
        integrity = self._connection.execute(
            "PRAGMA integrity_check"
        ).fetchall()
        integrity_messages = [str(row[0]) for row in integrity]
        if integrity_messages != ["ok"]:
            raise VerificationError(
                "SQLite integrity_check failed: "
                + "; ".join(integrity_messages)
            )
        foreign_key_rows = self._connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
        if foreign_key_rows:
            raise VerificationError("SQLite foreign_key_check failed")

        pack_rows = self._connection.execute(
            """
            SELECT pack_id, filename, committed_end, content_count
            FROM packs
            ORDER BY pack_id
            """
        ).fetchall()
        known_names = {str(row["filename"]) for row in pack_rows}
        actual_names = {path.name for path in self.root.glob(_PACK_GLOB)}
        if actual_names != known_names:
            missing = sorted(known_names - actual_names)
            unindexed = sorted(actual_names - known_names)
            raise VerificationError(
                f"pack file set mismatch; missing={missing}, unindexed={unindexed}"
            )

        verified_content_count = 0
        pack_receipts: list[dict[str, object]] = []
        for pack in pack_rows:
            filename = str(pack["filename"])
            expected_filename = f"pack-{int(pack['pack_id']):08d}.cicp"
            if filename != expected_filename:
                raise VerificationError(
                    f"pack {pack['pack_id']} has unsafe filename {filename!r}"
                )
            path = self.root / filename
            committed_end = int(pack["committed_end"])
            if path.is_symlink() or not path.is_file():
                raise VerificationError(f"committed pack is missing: {filename}")
            if path.stat().st_size != committed_end:
                raise VerificationError(
                    f"{filename} size differs from committed_end"
                )
            with path.open("rb") as handle:
                if handle.read(len(_PACK_MAGIC)) != _PACK_MAGIC:
                    raise VerificationError(
                        f"{filename} has an invalid pack header"
                    )
            expected_offset = len(_PACK_MAGIC)
            pack_content_count = 0
            content_rows = self._connection.execute(
                """
                SELECT *
                FROM contents
                WHERE pack_id = ?
                ORDER BY offset
                """,
                (int(pack["pack_id"]),),
            )
            for content_row in content_rows:
                if int(content_row["offset"]) != expected_offset:
                    raise VerificationError(
                        f"{filename} has a frame gap or overlap at "
                        f"{content_row['sha256']}"
                    )
                self._read_content_row(content_row)
                expected_offset += int(content_row["frame_size"])
                pack_content_count += 1
                verified_content_count += 1
            if expected_offset != committed_end:
                raise VerificationError(
                    f"{filename} committed boundary is not frame-aligned"
                )
            if pack_content_count != int(pack["content_count"]):
                raise VerificationError(
                    f"{filename} content_count disagrees with its frames"
                )
            pack_receipts.append(
                {
                    "filename": filename,
                    "committed_end": committed_end,
                    "content_count": pack_content_count,
                    "sha256": _sha256_prefix(path, committed_end),
                }
            )

        for sequence in self._connection.execute(
            """
            SELECT token_sequence_sha256, token_count,
                   tokenizer_fingerprint
            FROM token_sequences
            ORDER BY token_sequence_sha256
            """
        ):
            sequence_digest = str(sequence["token_sequence_sha256"])
            if (
                len(sequence_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in sequence_digest
                )
            ):
                raise VerificationError(
                    "invalid token-sequence SHA-256 in SQLite"
                )
            if (
                int(sequence["token_count"]) < 0
                or not str(sequence["tokenizer_fingerprint"])
            ):
                raise VerificationError(
                    f"invalid token-sequence metadata for {sequence_digest}"
                )
        aggregate = self._connection.execute(
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
                (
                    SELECT COUNT(*) FROM token_sequences
                ) AS unique_token_sequence_count,
                (
                    SELECT COALESCE(SUM(token_count), 0)
                    FROM token_sequences
                ) AS exact_unique_payload_tokens
            """
        ).fetchone()
        stats = self._connection.execute(
            "SELECT * FROM stats WHERE singleton = 1"
        ).fetchone()
        if aggregate is None or stats is None:
            raise VerificationError("aggregate or stats row is missing")
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
            raise VerificationError(
                "raw occurrence bytes are lower than unique bytes"
            )
        for field, expected in expected_stats.items():
            if int(stats[field]) != expected:
                raise VerificationError(
                    f"counter mismatch for {field}: "
                    f"{stats[field]} != {expected}"
                )
        if verified_content_count != expected_stats["unique_content_count"]:
            raise VerificationError("not every indexed content frame was verified")

        fingerprints = [
            str(row[0])
            for row in self._connection.execute(
                """
                SELECT DISTINCT tokenizer_fingerprint
                FROM contents
                WHERE tokenizer_fingerprint IS NOT NULL
                ORDER BY tokenizer_fingerprint
                """
            )
        ]
        stats_fingerprint = stats["tokenizer_fingerprint"]
        if len(fingerprints) > 1:
            raise VerificationError(
                "unique chunks use more than one tokenizer fingerprint"
            )
        expected_fingerprint = fingerprints[0] if fingerprints else None
        if stats_fingerprint != expected_fingerprint:
            raise VerificationError(
                "stats tokenizer fingerprint disagrees with unique content"
            )
        inconsistent_token_bindings = self._connection.execute(
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
        if inconsistent_token_bindings is not None:
            raise VerificationError(
                "content token metadata disagrees with its token sequence"
            )
        orphan_sequences = self._connection.execute(
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
        if orphan_sequences is not None:
            raise VerificationError("unreferenced token sequence is indexed")

        policy_sha256 = hashlib.sha256(
            _canonical_json_bytes(self._policy)
        ).hexdigest()
        recovery_records = self.recovery_records()
        recovery_digest = _hash_records(
            "cppmega-ci-recovery-records-v1",
            iter(recovery_records),
        )
        return {
            "ok": True,
            "schema": STORE_SCHEMA,
            "policy_sha256": policy_sha256,
            "script_sha256": self.script_sha256,
            "sqlite_schema_sha256": self._sqlite_schema_sha256,
            "counters": self._counters_from_row(stats),
            "logical_content_set_sha256": self._content_set_digest(),
            "logical_token_sequence_set_sha256": (
                self._token_sequence_set_digest()
            ),
            "occurrence_set_sha256": self._occurrence_set_digest(),
            "sqlite_logical_sha256": self._sqlite_logical_digest(),
            "packs": pack_receipts,
            "recovery": {
                "quarantined_orphan_count": len(recovery_records),
                "records_sha256": recovery_digest,
                "records": recovery_records,
            },
        }

    def verify(self, *, raise_on_error: bool = True) -> dict[str, object]:
        """Fully verify SQLite, every committed frame, hashes, and counters."""

        try:
            with self._lock:
                self._begin_write()
                try:
                    report = self._verify_locked()
                    self._connection.execute("COMMIT")
                    return report
                except BaseException:
                    self._rollback()
                    raise
        except (
            ContentStoreError,
            sqlite3.Error,
            OSError,
            UnicodeError,
            ValueError,
        ) as exc:
            if raise_on_error:
                if isinstance(exc, VerificationError):
                    raise
                raise VerificationError(str(exc)) from exc
            return {
                "ok": False,
                "schema": STORE_SCHEMA,
                "error": str(exc),
            }

    def completion_receipt(
        self,
        *,
        target_unique_tokens: int = PRODUCTION_TARGET_UNIQUE_TOKENS,
        emitted_valid_training_tokens: int | None = None,
    ) -> dict[str, object]:
        if (
            isinstance(target_unique_tokens, bool)
            or not isinstance(target_unique_tokens, int)
            or target_unique_tokens < 0
        ):
            raise ValueError(
                "target_unique_tokens must be a non-negative integer"
            )
        if (
            emitted_valid_training_tokens is not None
            and (
                isinstance(emitted_valid_training_tokens, bool)
                or not isinstance(emitted_valid_training_tokens, int)
                or emitted_valid_training_tokens < 0
            )
        ):
            raise ValueError(
                "emitted_valid_training_tokens must be a non-negative integer"
            )
        verification = self.verify()
        counters = verification["counters"]
        assert isinstance(counters, dict)
        exact_unique_payload_tokens = counters["exact_unique_payload_tokens"]
        if exact_unique_payload_tokens is None:
            raise ThresholdNotMetError(
                "completion receipt refused: every unique chunk must have an "
                "exact token count under one tokenizer fingerprint"
            )
        if int(exact_unique_payload_tokens) < target_unique_tokens:
            raise ThresholdNotMetError(
                "completion receipt refused: exact_unique_payload_tokens "
                f"{exact_unique_payload_tokens} is below target "
                f"{target_unique_tokens}"
            )
        receipt: dict[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "status": "complete",
            "store_schema": STORE_SCHEMA,
            "policy": self.policy,
            "policy_sha256": verification["policy_sha256"],
            "script_sha256": verification["script_sha256"],
            "sqlite_schema_sha256": verification["sqlite_schema_sha256"],
            "target_exact_unique_payload_tokens": target_unique_tokens,
            "exact_unique_payload_tokens": exact_unique_payload_tokens,
            "counters": counters,
            "logical_content_set_sha256": verification[
                "logical_content_set_sha256"
            ],
            "logical_token_sequence_set_sha256": verification[
                "logical_token_sequence_set_sha256"
            ],
            "occurrence_set_sha256": verification["occurrence_set_sha256"],
            "pack_hashes": verification["packs"],
            "sqlite_logical_sha256": verification["sqlite_logical_sha256"],
            "recovery": verification["recovery"],
            "verification": {
                "mode": "full",
                "ok": True,
            },
        }
        if emitted_valid_training_tokens is not None:
            receipt["emitted_valid_training_tokens"] = (
                emitted_valid_training_tokens
            )
        return receipt

    build_receipt = completion_receipt
    create_receipt = completion_receipt

    def write_completion_receipt(
        self,
        path: str | os.PathLike[str],
        *,
        target_unique_tokens: int = PRODUCTION_TARGET_UNIQUE_TOKENS,
        emitted_valid_training_tokens: int | None = None,
    ) -> dict[str, object]:
        receipt = self.completion_receipt(
            target_unique_tokens=target_unique_tokens,
            emitted_valid_training_tokens=emitted_valid_training_tokens,
        )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}"
        )
        encoded = (
            json.dumps(
                receipt,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
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
        return receipt

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "CIContentStore":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


ContentAddressedCIStore = CIContentStore


def _print_json(value: object) -> None:
    print(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and inspect a cppmega CI content store."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser(
        "status",
        help="show durable counters without claiming full verification",
    )
    status_parser.add_argument("store", type=Path)

    verify_parser = subparsers.add_parser(
        "verify",
        help="verify SQLite, every committed content frame, and all counters",
    )
    verify_parser.add_argument("store", type=Path)

    receipt_parser = subparsers.add_parser(
        "receipt",
        help="emit a fully verified threshold/completion receipt",
    )
    receipt_parser.add_argument("store", type=Path)
    receipt_parser.add_argument(
        "--target-exact-unique-payload-tokens",
        "--target-unique-tokens",
        "--target",
        dest="target_unique_tokens",
        type=int,
        default=PRODUCTION_TARGET_UNIQUE_TOKENS,
    )
    receipt_parser.add_argument(
        "--emitted-valid-training-tokens",
        type=int,
        help=(
            "optional downstream count, recorded separately and never used "
            "for the exact unique-payload threshold"
        ),
    )
    receipt_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        with CIContentStore(args.store) as store:
            if args.command == "status":
                result = store.status()
            elif args.command == "verify":
                result = store.verify()
            else:
                if args.output is None:
                    result = store.completion_receipt(
                        target_unique_tokens=args.target_unique_tokens,
                        emitted_valid_training_tokens=(
                            args.emitted_valid_training_tokens
                        ),
                    )
                else:
                    result = store.write_completion_receipt(
                        args.output,
                        target_unique_tokens=args.target_unique_tokens,
                        emitted_valid_training_tokens=(
                            args.emitted_valid_training_tokens
                        ),
                    )
        _print_json(result)
        return 0
    except (
        ContentStoreError,
        OSError,
        sqlite3.Error,
        UnicodeError,
        ValueError,
    ) as exc:
        print(f"ci-content-store: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
