#!/usr/bin/env python3
"""Checkpoint and receipt-bind a completed global source dedup SQLite store."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cppmega.data.source_conveyor_composition import (  # noqa: E402
    GLOBAL_DEDUP_RECEIPT_SCHEMA,
)
from tools.clang_indexer.dedup_store import DedupStore  # noqa: E402


LOGICAL_HASH_ALGORITHM = "cppmega_sqlite_rows_lenprefixed_v1"
VERIFIER_PATH = "scripts/data/verify_global_dedup_store.py"
_STAGED_TABLES = frozenset(
    {
        "dedup_stages",
        "exact_stage",
        "minhash_stage",
        "lsh_stage",
        "chunk_claims_stage",
    }
)
_TABLE_QUERIES = {
    "exact": "SELECT hash FROM exact ORDER BY hash",
    "lsh": "SELECT band_id, band_hash, doc_id FROM lsh "
    "ORDER BY band_id, band_hash, doc_id",
    "minhash": "SELECT doc_id, sig FROM minhash ORDER BY doc_id",
    "dedup_meta": "SELECT key, val FROM dedup_meta ORDER BY key",
    "chunk_claims": "SELECT namespace, hash, claim_count FROM chunk_claims "
    "ORDER BY namespace, hash",
    "dedup_stages": "SELECT stage_id, created_at, next_doc_id FROM dedup_stages "
    "ORDER BY stage_id",
    "exact_stage": "SELECT stage_id, hash FROM exact_stage ORDER BY stage_id, hash",
    "minhash_stage": "SELECT stage_id, stage_doc_id, sig FROM minhash_stage "
    "ORDER BY stage_id, stage_doc_id",
    "lsh_stage": "SELECT stage_id, band_id, band_hash, stage_doc_id FROM lsh_stage "
    "ORDER BY stage_id, band_id, band_hash, stage_doc_id",
    "chunk_claims_stage": "SELECT stage_id, namespace, hash, claim_count "
    "FROM chunk_claims_stage ORDER BY stage_id, namespace, hash",
}
_TABLE_COLUMNS = {
    "exact": (("hash", "BLOB", 0, 1),),
    "lsh": (
        ("band_id", "INTEGER", 1, 0),
        ("band_hash", "BLOB", 1, 0),
        ("doc_id", "INTEGER", 1, 0),
    ),
    "minhash": (
        ("doc_id", "INTEGER", 0, 1),
        ("sig", "BLOB", 1, 0),
    ),
    "dedup_meta": (
        ("key", "TEXT", 0, 1),
        ("val", "INTEGER", 0, 0),
    ),
    "chunk_claims": (
        ("namespace", "TEXT", 1, 1),
        ("hash", "BLOB", 1, 2),
        ("claim_count", "INTEGER", 1, 0),
    ),
    "dedup_stages": (
        ("stage_id", "TEXT", 0, 1),
        ("created_at", "REAL", 1, 0),
        ("next_doc_id", "INTEGER", 1, 0),
    ),
    "exact_stage": (
        ("stage_id", "TEXT", 1, 1),
        ("hash", "BLOB", 1, 2),
    ),
    "minhash_stage": (
        ("stage_id", "TEXT", 1, 1),
        ("stage_doc_id", "INTEGER", 1, 2),
        ("sig", "BLOB", 1, 0),
    ),
    "lsh_stage": (
        ("stage_id", "TEXT", 1, 0),
        ("band_id", "INTEGER", 1, 0),
        ("band_hash", "BLOB", 1, 0),
        ("stage_doc_id", "INTEGER", 1, 0),
    ),
    "chunk_claims_stage": (
        ("stage_id", "TEXT", 1, 1),
        ("namespace", "TEXT", 1, 2),
        ("hash", "BLOB", 1, 3),
        ("claim_count", "INTEGER", 1, 0),
    ),
}
_EXPLICIT_INDEXES = frozenset({"lsh_band", "lsh_stage_band"})
_PRODUCTION_TABLES = frozenset(
    {"exact", "lsh", "minhash", "dedup_meta", "chunk_claims"}
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_regular_file(path: Path, *, where: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{where} must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _encode_scalar(value: object) -> bytes:
    if value is None:
        tag, payload = b"n", b""
    elif isinstance(value, int):
        tag, payload = b"i", str(value).encode("ascii")
    elif isinstance(value, float):
        tag, payload = b"f", struct.pack(">d", value)
    elif isinstance(value, str):
        tag, payload = b"s", value.encode("utf-8")
    elif isinstance(value, bytes):
        tag, payload = b"b", value
    else:  # pragma: no cover - sqlite only returns the types above
        raise TypeError(f"unsupported SQLite scalar type: {type(value)!r}")
    return tag + len(payload).to_bytes(8, "big") + payload


def _logical_table_receipt(
    connection: sqlite3.Connection,
    *,
    table: str,
) -> dict[str, object]:
    digest = hashlib.sha256()
    rows = 0
    for row in connection.execute(_TABLE_QUERIES[table]):
        digest.update(b"R")
        digest.update(len(row).to_bytes(4, "big"))
        for value in row:
            digest.update(_encode_scalar(value))
        rows += 1
    return {"rows": rows, "logical_sha256": digest.hexdigest()}


def _validate_schema(connection: sqlite3.Connection) -> str:
    table_names = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    }
    if table_names != set(_TABLE_COLUMNS):
        raise RuntimeError(
            "dedup table schema drifted: "
            f"missing={sorted(set(_TABLE_COLUMNS) - table_names)} "
            f"extra={sorted(table_names - set(_TABLE_COLUMNS))}"
        )
    for table, expected in _TABLE_COLUMNS.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            raise RuntimeError(
                f"dedup table {table} columns drifted: "
                f"actual={actual} expected={expected}"
            )
    explicit_indexes = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        )
    }
    if explicit_indexes != _EXPLICIT_INDEXES:
        raise RuntimeError(
            "dedup explicit indexes drifted: "
            f"actual={sorted(explicit_indexes)} expected={sorted(_EXPLICIT_INDEXES)}"
        )
    unsupported_objects = list(
        connection.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE type IN ('trigger', 'view') ORDER BY type, name"
        )
    )
    if unsupported_objects:
        raise RuntimeError(f"dedup database has unsupported objects: {unsupported_objects}")
    schema = [
        {
            "type": str(row[0]),
            "name": str(row[1]),
            "table": str(row[2]),
            "sql": str(row[3]),
        }
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL ORDER BY type, name"
        )
    ]
    return _canonical_sha256(schema)


def _validate_relational_content(
    connection: sqlite3.Connection,
    *,
    tables: dict[str, dict[str, object]],
) -> None:
    empty = sorted(
        table for table in _PRODUCTION_TABLES if int(tables[table]["rows"]) < 1
    )
    if empty:
        raise RuntimeError(f"dedup production tables are empty: {empty}")
    scalar_checks = {
        "exact hash width": (
            "SELECT COUNT(*) FROM exact WHERE length(hash) != 20",
            0,
        ),
        "minhash signature width": (
            f"SELECT COUNT(*) FROM minhash WHERE length(sig) != "
            f"{DedupStore.NUM_PERM * 8}",
            0,
        ),
        "LSH band hash width": (
            "SELECT COUNT(*) FROM lsh WHERE length(band_hash) != 20",
            0,
        ),
        "chunk claim hash width": (
            "SELECT COUNT(*) FROM chunk_claims WHERE length(hash) != 20",
            0,
        ),
        "chunk claim namespace": (
            "SELECT COUNT(*) FROM chunk_claims "
            "WHERE namespace != 'semantic_chunk:v1'",
            0,
        ),
        "chunk claim count": (
            "SELECT COUNT(*) FROM chunk_claims WHERE claim_count < 1",
            0,
        ),
        "orphan LSH document": (
            "SELECT COUNT(*) FROM lsh "
            "LEFT JOIN minhash ON minhash.doc_id = lsh.doc_id "
            "WHERE minhash.doc_id IS NULL",
            0,
        ),
    }
    for label, (query, expected) in scalar_checks.items():
        actual = int(connection.execute(query).fetchone()[0])
        if actual != expected:
            raise RuntimeError(f"dedup {label} integrity failed: {actual}")
    minhash_rows = int(tables["minhash"]["rows"])
    linked_docs = int(
        connection.execute("SELECT COUNT(DISTINCT doc_id) FROM lsh").fetchone()[0]
    )
    if linked_docs != minhash_rows:
        raise RuntimeError(
            "dedup LSH document coverage differs from minhash: "
            f"{linked_docs} != {minhash_rows}"
        )
    band_min, band_max = connection.execute(
        "SELECT MIN(bands), MAX(bands) FROM "
        "(SELECT COUNT(*) AS bands FROM lsh GROUP BY doc_id)"
    ).fetchone()
    if (
        not isinstance(band_min, int)
        or not isinstance(band_max, int)
        or band_min < 1
        or band_min != band_max
    ):
        raise RuntimeError(
            f"dedup LSH band cardinality drifted: min={band_min} max={band_max}"
        )
    meta = list(connection.execute("SELECT key, val FROM dedup_meta ORDER BY key"))
    max_doc_id = int(connection.execute("SELECT MAX(doc_id) FROM minhash").fetchone()[0])
    if meta != [("next_doc_id", max_doc_id + 1)]:
        raise RuntimeError(
            "dedup next_doc_id metadata drifted: "
            f"actual={meta} expected={max_doc_id + 1}"
        )
    if int(tables["exact"]["rows"]) < minhash_rows:
        raise RuntimeError("dedup exact inventory is smaller than its near inventory")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"receipt output must not be a symlink: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def verify_global_dedup_store(
    database_path: Path,
    *,
    output_path: Path,
    busy_timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Produce a receipt only for an idle, fully promoted production dedup DB."""

    database_path = _resolve_regular_file(database_path, where="dedup database")
    output_path = output_path.expanduser().resolve()
    if output_path == database_path:
        raise ValueError("dedup receipt output must differ from the database")
    connection = sqlite3.connect(
        database_path.as_uri() + "?mode=rw",
        timeout=busy_timeout_seconds,
        isolation_level=None,
        uri=True,
    )
    transaction_open = False
    try:
        connection.execute(f"PRAGMA busy_timeout={int(busy_timeout_seconds * 1000)}")
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
        if journal_mode.lower() != "wal":
            raise RuntimeError(
                f"dedup database journal mode is not WAL: {journal_mode}"
            )
        checkpoint_row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint_row is None or len(checkpoint_row) != 3:
            raise RuntimeError("dedup WAL checkpoint returned an invalid result")
        busy, log_frames, checkpointed_frames = map(int, checkpoint_row)
        wal_path = Path(str(database_path) + "-wal")
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0
        checkpoint = {
            "mode": "TRUNCATE",
            "busy": busy,
            "log_frames": log_frames,
            "checkpointed_frames": checkpointed_frames,
            "wal_size_bytes": wal_size,
        }
        if checkpoint != {
            "mode": "TRUNCATE",
            "busy": 0,
            "log_frames": 0,
            "checkpointed_frames": 0,
            "wal_size_bytes": 0,
        }:
            raise RuntimeError(f"dedup WAL is not fully checkpointed: {checkpoint}")

        connection.execute("BEGIN EXCLUSIVE")
        transaction_open = True
        integrity_rows = [
            str(row[0]) for row in connection.execute("PRAGMA quick_check")
        ]
        if integrity_rows != ["ok"]:
            raise RuntimeError(f"dedup SQLite quick_check failed: {integrity_rows}")
        sqlite_schema_sha256 = _validate_schema(connection)
        tables = {
            table: _logical_table_receipt(connection, table=table)
            for table in _TABLE_QUERIES
        }
        staged = {
            table: int(tables[table]["rows"])
            for table in sorted(_STAGED_TABLES)
            if int(tables[table]["rows"])
        }
        if staged:
            raise RuntimeError(f"dedup database has unpromoted staged rows: {staged}")
        _validate_relational_content(connection, tables=tables)

        stat_before = database_path.stat()
        database_sha256 = _sha256(database_path)
        stat_after = database_path.stat()
        stat_identity_before = (
            stat_before.st_dev,
            stat_before.st_ino,
            stat_before.st_size,
            stat_before.st_mtime_ns,
        )
        stat_identity_after = (
            stat_after.st_dev,
            stat_after.st_ino,
            stat_after.st_size,
            stat_after.st_mtime_ns,
        )
        if stat_identity_before != stat_identity_after:
            raise RuntimeError("dedup database changed while hashing")
        receipt: dict[str, object] = {
            "schema": GLOBAL_DEDUP_RECEIPT_SCHEMA,
            "status": "verified",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "database": {
                "path": str(database_path),
                "size_bytes": stat_after.st_size,
                "sha256": database_sha256,
            },
            "checkpoint": checkpoint,
            "integrity_check": "ok",
            "sqlite_schema_sha256": sqlite_schema_sha256,
            "logical_hash_algorithm": LOGICAL_HASH_ALGORITHM,
            "logical_sha256": _canonical_sha256(tables),
            "tables": tables,
            "policy": {
                "exact": "sha1_token_ids_v1",
                "chunk": "tokenized_chunk_claims_v1",
                "near": {
                    "enabled": True,
                    "threshold": DedupStore.THRESHOLD,
                    "num_perm": DedupStore.NUM_PERM,
                    "shingle_k": DedupStore.SHINGLE_K,
                },
            },
            "verifier": {
                "repository_identity": "cppmega",
                "script": VERIFIER_PATH,
                "script_sha256": _sha256(Path(__file__).resolve()),
            },
        }
        _write_json_atomic(output_path, receipt)
        connection.execute("COMMIT")
        transaction_open = False
        return receipt
    finally:
        if transaction_open:
            connection.execute("ROLLBACK")
        connection.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--busy-timeout-seconds", type=float, default=300.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.busy_timeout_seconds <= 0:
        raise SystemExit("--busy-timeout-seconds must be positive")
    receipt = verify_global_dedup_store(
        args.database,
        output_path=args.output,
        busy_timeout_seconds=args.busy_timeout_seconds,
    )
    print(
        json.dumps(
            {
                "receipt": str(args.output.resolve()),
                "database_sha256": receipt["database"]["sha256"],
                "logical_sha256": receipt["logical_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
