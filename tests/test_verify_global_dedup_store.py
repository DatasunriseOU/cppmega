from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from scripts.data.verify_global_dedup_store import verify_global_dedup_store


def _write_dedup_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.executescript(
            """
            CREATE TABLE exact (hash BLOB PRIMARY KEY);
            CREATE TABLE lsh (
                band_id INTEGER NOT NULL,
                band_hash BLOB NOT NULL,
                doc_id INTEGER NOT NULL
            );
            CREATE INDEX lsh_band ON lsh (band_id, band_hash);
            CREATE TABLE minhash (
                doc_id INTEGER PRIMARY KEY,
                sig BLOB NOT NULL
            );
            CREATE TABLE dedup_meta (key TEXT PRIMARY KEY, val INTEGER);
            CREATE TABLE chunk_claims (
                namespace TEXT NOT NULL,
                hash BLOB NOT NULL,
                claim_count INTEGER NOT NULL,
                PRIMARY KEY(namespace, hash)
            );
            CREATE TABLE dedup_stages (
                stage_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                next_doc_id INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE exact_stage (
                stage_id TEXT NOT NULL,
                hash BLOB NOT NULL,
                PRIMARY KEY(stage_id, hash)
            );
            CREATE TABLE minhash_stage (
                stage_id TEXT NOT NULL,
                stage_doc_id INTEGER NOT NULL,
                sig BLOB NOT NULL,
                PRIMARY KEY(stage_id, stage_doc_id)
            );
            CREATE TABLE lsh_stage (
                stage_id TEXT NOT NULL,
                band_id INTEGER NOT NULL,
                band_hash BLOB NOT NULL,
                stage_doc_id INTEGER NOT NULL
            );
            CREATE INDEX lsh_stage_band
                ON lsh_stage (stage_id, band_id, band_hash);
            CREATE TABLE chunk_claims_stage (
                stage_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                hash BLOB NOT NULL,
                claim_count INTEGER NOT NULL,
                PRIMARY KEY(stage_id, namespace, hash)
            );
            """
        )
        connection.execute("INSERT INTO exact(hash) VALUES (?)", (b"e" * 20,))
        connection.execute(
            "INSERT INTO minhash(doc_id, sig) VALUES (?, ?)",
            (0, b"m" * 2048),
        )
        connection.execute(
            "INSERT INTO lsh(band_id, band_hash, doc_id) VALUES (?, ?, ?)",
            (0, b"l" * 20, 0),
        )
        connection.execute(
            "INSERT INTO chunk_claims(namespace, hash, claim_count) VALUES (?, ?, ?)",
            ("semantic_chunk:v1", b"c" * 20, 1),
        )
        connection.execute(
            "INSERT INTO dedup_meta(key, val) VALUES (?, ?)",
            ("next_doc_id", 1),
        )
        connection.commit()
    finally:
        connection.close()


def test_global_dedup_receipt_checkpoints_and_hashes_exact_schema(
    tmp_path: Path,
) -> None:
    database = tmp_path / "dedup.sqlite"
    output = tmp_path / "dedup_receipt.json"
    _write_dedup_database(database)

    receipt = verify_global_dedup_store(database, output_path=output)

    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    assert receipt["status"] == "verified"
    assert receipt["checkpoint"] == {
        "mode": "TRUNCATE",
        "busy": 0,
        "log_frames": 0,
        "checkpointed_frames": 0,
        "wal_size_bytes": 0,
    }
    assert receipt["database"] == {
        "path": str(database.resolve()),
        "size_bytes": database.stat().st_size,
        "sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
    }
    assert receipt["tables"]["exact"]["rows"] == 1
    assert receipt["tables"]["minhash"]["rows"] == 1
    assert receipt["tables"]["lsh"]["rows"] == 1
    assert receipt["tables"]["dedup_meta"]["rows"] == 1
    assert receipt["tables"]["chunk_claims"]["rows"] == 1
    assert all(
        receipt["tables"][name]["rows"] == 0
        for name in (
            "dedup_stages",
            "exact_stage",
            "minhash_stage",
            "lsh_stage",
            "chunk_claims_stage",
        )
    )


def test_global_dedup_receipt_rejects_unpromoted_stage(tmp_path: Path) -> None:
    database = tmp_path / "dedup.sqlite"
    output = tmp_path / "dedup_receipt.json"
    _write_dedup_database(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "INSERT INTO dedup_stages(stage_id, created_at, next_doc_id) "
            "VALUES (?, ?, ?)",
            ("unfinished", 1.0, 0),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="unpromoted staged rows"):
        verify_global_dedup_store(database, output_path=output)

    assert not output.exists()


def test_global_dedup_receipt_rejects_schema_extension(tmp_path: Path) -> None:
    database = tmp_path / "dedup.sqlite"
    output = tmp_path / "dedup_receipt.json"
    _write_dedup_database(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE ambient_fallback(value TEXT)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="dedup table schema drifted"):
        verify_global_dedup_store(database, output_path=output)

    assert not output.exists()


def test_global_dedup_receipt_rejects_empty_near_index(tmp_path: Path) -> None:
    database = tmp_path / "dedup.sqlite"
    output = tmp_path / "dedup_receipt.json"
    _write_dedup_database(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DELETE FROM lsh")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="production tables are empty"):
        verify_global_dedup_store(database, output_path=output)

    assert not output.exists()
