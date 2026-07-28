from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import threading
from typing import Callable
import zlib

import pytest

from scripts.ci_content_store import (
    CIContentStore,
    ContentMetadataConflictError,
    HashCollisionError,
    _ORPHAN_TOKEN_SEQUENCE_QUERY,
    OccurrenceConflictError,
    ThresholdNotMetError,
    TOKEN_SEQUENCE_ENCODING,
    VerificationError,
    hash_token_sequence,
)
from scripts.ci_zlib_evidence import (
    MAX_STATE_JSON_EVIDENCE_BYTES,
    MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES,
)


@lru_cache(maxsize=None)
def _compressed_repetition(raw_size: int) -> tuple[bytes, str]:
    compressor = zlib.compressobj(9)
    digest = hashlib.sha256()
    parts: list[bytes] = []
    chunk = b"x" * (1024 * 1024)
    remaining = raw_size
    while remaining:
        current = chunk[: min(len(chunk), remaining)]
        digest.update(current)
        parts.append(compressor.compress(current))
        remaining -= len(current)
    parts.append(compressor.flush())
    return b"".join(parts), digest.hexdigest()


def _key(ordinal: int, *, step: str = "collect") -> dict[str, object]:
    return {
        "repo": "owner/repo",
        "run_attempt": 2,
        "job": "linux-tests",
        "step": step,
        "chunk_ordinal": ordinal,
    }


def _sequence(*token_ids: int) -> str:
    return hash_token_sequence(token_ids)


class _CommitCountingStore(CIContentStore):
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.durable_commit_calls = 0
        super().__init__(*args, **kwargs)

    def _durable_commit(self) -> None:
        self.durable_commit_calls += 1
        super()._durable_commit()


def test_token_sequence_hash_matches_reference_u32_big_endian_encoding() -> None:
    token_ids = (0, 1, 0x12345678, 0xFFFFFFFF)
    reference = hashlib.sha256(
        TOKEN_SEQUENCE_ENCODING.encode("ascii")
        + b"\0"
        + struct.pack(">Q", len(token_ids))
        + struct.pack(">IIII", *token_ids)
    ).hexdigest()

    assert hash_token_sequence(token_ids) == reference
    with pytest.raises(ValueError, match="uint32"):
        hash_token_sequence((True,))
    with pytest.raises(ValueError, match="uint32"):
        hash_token_sequence((0x1_0000_0000,))


def test_duplicate_content_is_stored_once_but_every_occurrence_is_counted(
    tmp_path: Path,
) -> None:
    with CIContentStore(tmp_path / "store") as store:
        first = store.add_chunk(
            "same log\n",
            {"archive": "a.zip", "entry": "one.txt"},
            _key(0),
            token_count=3,
            tokenizer_fingerprint="tokenizer-v1",
            token_sequence_sha256=_sequence(10, 11, 12),
        )
        second = store.add_chunk(
            b"same log\n",
            {"archive": "b.zip", "entry": "two.txt"},
            _key(1),
            token_count=3,
            tokenizer_fingerprint="tokenizer-v1",
            token_sequence_sha256=_sequence(10, 11, 12),
        )

        assert first["content_added"] is True
        assert second["content_added"] is False
        assert store.read_chunk(first["sha256"]) == b"same log\n"
        assert [item["content"] for item in store.iter_chunks()] == [b"same log\n"]
        assert len(list(store.iter_occurrences())) == 2
        assert store.status()["counters"] == {
            "raw_occurrence_bytes": 18,
            "unique_bytes": 9,
            "duplicate_bytes": 9,
            "unique_content_count": 1,
            "occurrence_count": 2,
            "tokenized_unique_content_count": 1,
            "unique_token_sequence_count": 1,
            "tokenizer_fingerprint": "tokenizer-v1",
            "exact_unique_payload_tokens": 3,
        }
        assert store.verify()["ok"] is True


def test_one_store_connection_is_safely_serialized_across_fetch_threads(
    tmp_path: Path,
) -> None:
    with CIContentStore(tmp_path / "store") as store:
        def add(ordinal: int) -> None:
            store.add_chunk(
                f"threaded-{ordinal % 5}\n",
                {"worker_record": ordinal},
                _key(ordinal),
                token_count=2,
                tokenizer_fingerprint="tokenizer-v1",
                token_sequence_sha256=_sequence(100, ordinal % 5),
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(add, range(80)))

        counters = store.verify()["counters"]
        assert counters["occurrence_count"] == 80
        assert counters["unique_content_count"] == 5
        assert counters["unique_token_sequence_count"] == 5
        assert counters["exact_unique_payload_tokens"] == 10


def test_batch_is_atomic_and_whole_batch_replay_is_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    records = [
        {
            "content": f"chunk-{ordinal}",
            "provenance": {"entry": f"{ordinal}.txt"},
            "occurrence_key": _key(ordinal),
        }
        for ordinal in range(2)
    ]
    with _CommitCountingStore(root) as store:
        first = store.add_chunks(records)
        assert store.durable_commit_calls == 1
        committed_size = sum(
            pack["committed_end"] for pack in store.verify()["packs"]
        )
        replay = store.add_chunks(records)
        assert store.durable_commit_calls == 2
        assert all(item["occurrence_added"] for item in first)
        assert all(not item["occurrence_added"] for item in replay)
        assert store.status()["counters"]["occurrence_count"] == 2

        with pytest.raises(OccurrenceConflictError):
            store.add_chunks(
                [
                    {
                        "content": "would-be-orphan",
                        "provenance": {"entry": "new.txt"},
                        "occurrence_key": _key(2),
                    },
                    {
                        "content": "conflict",
                        "provenance": {"entry": "0.txt"},
                        "occurrence_key": _key(0),
                    },
                ]
            )
        verification = store.verify()
        assert verification["counters"]["occurrence_count"] == 2
        assert verification["counters"]["unique_content_count"] == 2
        assert sum(
            pack["committed_end"] for pack in verification["packs"]
        ) == committed_size

    with CIContentStore(root) as reopened:
        assert reopened.verify()["counters"]["occurrence_count"] == 2


def test_rich_provenance_is_canonical_compressed_and_round_trips(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    provenance = {
        "timestamp": "2026-07-26T12:34:56Z",
        "substitution_ledger": [
            {
                "source": "very-long-repeated-source-path/" * 8,
                "replacement": "very-long-repeated-replacement/" * 8,
            }
            for _ in range(50)
        ],
    }
    with CIContentStore(root) as store:
        store.add_chunk("payload", provenance, _key(0))
        occurrence = next(store.iter_occurrences())
        assert occurrence["provenance"] == provenance

    with sqlite3.connect(root / "index.sqlite3") as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(occurrences)")
        }
        storage = connection.execute(
            """
            SELECT typeof(provenance_zlib), length(provenance_zlib),
                   provenance_raw_size, length(provenance_sha256)
            FROM occurrences
            """
        ).fetchone()
    assert "provenance_json" not in columns
    assert storage is not None
    assert storage[0] == "blob"
    assert storage[1] < storage[2]
    assert storage[3] == 64


def test_corrupt_compressed_provenance_fails_full_verification(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    with CIContentStore(root) as store:
        store.add_chunk("payload", {"ledger": ["value"] * 20}, _key(0))
    with sqlite3.connect(root / "index.sqlite3") as connection:
        blob = connection.execute(
            "SELECT provenance_zlib FROM occurrences"
        ).fetchone()[0]
        corrupted = bytes(blob[:-1]) + bytes([blob[-1] ^ 0xFF])
        connection.execute(
            "UPDATE occurrences SET provenance_zlib = ?",
            (sqlite3.Binary(corrupted),),
        )

    with CIContentStore(root) as store:
        assert store.verify(raise_on_error=False)["ok"] is False
        with pytest.raises(VerificationError, match="provenance"):
            store.verify()


def test_provenance_bomb_is_preflighted_before_resume_replay_and_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    raw_size = MAX_STATE_JSON_EVIDENCE_BYTES + 1
    compressed, digest = _compressed_repetition(raw_size)
    with CIContentStore(root) as store:
        store.add_chunk("payload", {"entry": "job.log"}, _key(0))
        with store._connection:
            store._connection.execute(
                """
                UPDATE occurrences SET
                  provenance_sha256=?,
                  provenance_raw_size=?,
                  provenance_zlib=?
                """,
                (digest, raw_size, sqlite3.Binary(compressed)),
            )
        assert store.verify(raise_on_error=False)["ok"] is False
        with pytest.raises(VerificationError, match="byte bounds"):
            store.verify()
        with pytest.raises(VerificationError, match="byte bounds"):
            store.completion_receipt(target_unique_tokens=0)
        with pytest.raises(VerificationError, match="byte bounds"):
            store.add_chunk("payload", {"entry": "job.log"}, _key(0))
        with pytest.raises(VerificationError, match="byte bounds"):
            list(store.iter_occurrences())

    with pytest.raises(VerificationError, match="byte bounds"):
        CIContentStore(root)


def test_mutable_store_operations_pin_preflight_and_blob_read_to_one_snapshot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    provenance = {"entry": "job.log"}
    provenance_raw = json.dumps(
        provenance,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    provenance_digest = hashlib.sha256(provenance_raw).hexdigest()
    provenance_zlib = zlib.compress(provenance_raw, 6)

    with CIContentStore(root) as store:
        store.add_chunk("payload", provenance, _key(0))
        largest_materialized_blob = 0

        def guarded_row_factory(
            cursor: sqlite3.Cursor,
            values: tuple[object, ...],
        ) -> sqlite3.Row:
            nonlocal largest_materialized_blob
            largest_materialized_blob = max(
                (
                    len(value)
                    for value in values
                    if isinstance(value, bytes)
                ),
                default=largest_materialized_blob,
            )
            if (
                largest_materialized_blob
                > MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES
            ):
                raise AssertionError("oversized provenance BLOB was materialized")
            return sqlite3.Row(cursor, values)

        store._connection.row_factory = guarded_row_factory

        def restore_valid_provenance() -> None:
            with sqlite3.connect(store.db_path) as connection:
                connection.execute(
                    """
                    UPDATE occurrences SET
                      provenance_sha256=?,
                      provenance_raw_size=?,
                      provenance_zlib=?
                    WHERE repo='owner/repo' AND chunk_ordinal=0
                    """,
                    (
                        provenance_digest,
                        len(provenance_raw),
                        sqlite3.Binary(provenance_zlib),
                    ),
                )

        def assert_surface_rejects_before_materialization(
            invoke: Callable[[], object],
        ) -> None:
            trigger = threading.Event()
            finished = threading.Event()
            writer_errors: list[BaseException] = []
            trace_triggered = False

            def attacker() -> None:
                trigger.wait()
                try:
                    with sqlite3.connect(store.db_path, timeout=30.0) as connection:
                        connection.execute(
                            """
                            UPDATE occurrences
                            SET provenance_zlib=zeroblob(?)
                            WHERE repo='owner/repo' AND chunk_ordinal=0
                            """,
                            (MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES + 1,),
                        )
                except BaseException as exc:
                    writer_errors.append(exc)
                finally:
                    finished.set()

            writer = threading.Thread(target=attacker)
            writer.start()

            def trace(statement: str) -> None:
                nonlocal trace_triggered
                normalized = " ".join(statement.split()).upper()
                evidence_select = (
                    normalized.startswith("SELECT")
                    and "PROVENANCE_ZLIB" in normalized
                    and "FROM OCCURRENCES" in normalized
                    and "LENGTH(PROVENANCE_ZLIB)" not in normalized
                )
                if not trace_triggered and (
                    normalized.startswith("BEGIN") or evidence_select
                ):
                    trace_triggered = True
                    trigger.set()
                    finished.wait(30.0)

            store._connection.set_trace_callback(trace)
            try:
                with pytest.raises(
                    VerificationError,
                    match="occurrence provenance exceeds its versioned byte bounds",
                ):
                    invoke()
            finally:
                store._connection.set_trace_callback(None)
                trigger.set()
                writer.join(timeout=30.0)
            assert not writer.is_alive()
            assert trace_triggered
            assert writer_errors == []
            assert (
                largest_materialized_blob
                <= MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES
            )

        surfaces = (
            lambda: store.add_chunk("payload", provenance, _key(0)),
            lambda: list(store.iter_occurrences()),
            store.verify,
        )
        for invoke in surfaces:
            restore_valid_provenance()
            assert_surface_rejects_before_materialization(invoke)


def test_identical_replay_is_idempotent_and_conflicting_replay_fails(
    tmp_path: Path,
) -> None:
    provenance = {"entry": "job.log", "metadata": {"line": 1}}
    with CIContentStore(tmp_path / "store") as store:
        store.add_chunk("payload", provenance, _key(0))
        replay = store.add_chunk(
            "payload",
            {"metadata": {"line": 1}, "entry": "job.log"},
            _key(0),
        )
        assert replay["occurrence_added"] is False
        assert store.status()["counters"]["occurrence_count"] == 1

        with pytest.raises(OccurrenceConflictError, match="conflicting replay"):
            store.add_chunk("different", provenance, _key(0))
        with pytest.raises(OccurrenceConflictError, match="conflicting replay"):
            store.add_chunk("payload", {"entry": "other.log"}, _key(0))

        assert store.status()["counters"]["occurrence_count"] == 1
        assert store.verify()["ok"] is True


def test_reopen_truncates_crash_orphan_tail_to_database_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    with CIContentStore(root) as store:
        store.add_chunk("committed", {"entry": "one"}, _key(0))
        pack = store.verify()["packs"][0]
    pack_path = root / pack["filename"]
    committed_end = pack["committed_end"]
    with pack_path.open("ab") as handle:
        handle.write(b"orphan frame bytes from a killed writer")
        handle.flush()

    assert pack_path.stat().st_size > committed_end
    with CIContentStore(root) as recovered:
        assert pack_path.stat().st_size == committed_end
        verification = recovered.verify()
        assert verification["ok"] is True
        recovery_records = recovered.recovery_records()
        assert len(recovery_records) == 1
        recovery_record = recovery_records[0]
        assert recovery_record["kind"] == "pack-tail"
        quarantined_tail = (
            root
            / "orphaned"
            / recovery_record["quarantined_filename"]
        )
        assert quarantined_tail.read_bytes() == (
            b"orphan frame bytes from a killed writer"
        )
        assert recovered.read_chunk(hashlib.sha256(b"committed").hexdigest()) == (
            b"committed"
        )


def test_unknown_valid_pack_is_preserved_in_auditable_quarantine(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    with CIContentStore(root):
        pass
    orphan_bytes = b"CICSPK1\n" + b"recoverable-uncommitted-pack-bytes"
    unknown_pack = root / "pack-99999999.cicp"
    unknown_pack.write_bytes(orphan_bytes)

    with CIContentStore(root) as recovered:
        assert not unknown_pack.exists()
        records = recovered.recovery_records()
        assert len(records) == 1
        record = records[0]
        assert record == {
            "schema": "cppmega_ci_content_store_recovery_v1",
            "kind": "whole-pack",
            "reason": "unindexed-pack-found-on-open",
            "original_filename": "pack-99999999.cicp",
            "source_offset": 0,
            "byte_size": len(orphan_bytes),
            "sha256": hashlib.sha256(orphan_bytes).hexdigest(),
            "quarantined_filename": record["quarantined_filename"],
        }
        quarantined = root / "orphaned" / record["quarantined_filename"]
        assert quarantined.read_bytes() == orphan_bytes
        metadata = quarantined.with_name(
            f"{quarantined.name}.recovery.json"
        )
        assert json.loads(metadata.read_text(encoding="utf-8")) == record
        verification = recovered.verify()
        assert verification["recovery"]["quarantined_orphan_count"] == 1
        assert len(verification["recovery"]["records_sha256"]) == 64


def test_unknown_invalid_pack_fails_closed_without_deleting_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    with CIContentStore(root):
        pass
    invalid_bytes = b"not-a-valid-pack-but-still-not-ours-to-delete"
    unknown_pack = root / "pack-77777777.cicp"
    unknown_pack.write_bytes(invalid_bytes)

    with pytest.raises(VerificationError, match="invalid header"):
        CIContentStore(root)
    assert unknown_pack.read_bytes() == invalid_bytes


def test_small_pack_limit_rotates_without_exceeding_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    with CIContentStore(root, max_pack_bytes=125) as store:
        for ordinal in range(4):
            # These values are deliberately poor zlib matches with each other.
            content = bytes(range(32 + ordinal, 72 + ordinal)).decode("ascii")
            store.add_chunk(content, {"ordinal": ordinal}, _key(ordinal))
        verification = store.verify()

    assert len(verification["packs"]) >= 2
    assert all(pack["committed_end"] <= 125 for pack in verification["packs"])
    assert all(
        (root / pack["filename"]).stat().st_size <= 125
        for pack in verification["packs"]
    )


def test_corrupt_committed_pack_is_detected_and_receipt_is_refused(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    with CIContentStore(root) as store:
        store.add_chunk(
            "integrity matters",
            {"entry": "one"},
            _key(0),
            token_count=2,
            tokenizer_fingerprint="tokenizer-v1",
            token_sequence_sha256=_sequence(7, 8),
        )
        pack = store.verify()["packs"][0]
        pack_path = root / pack["filename"]
        with pack_path.open("r+b") as handle:
            handle.seek(pack["committed_end"] - 1)
            final_byte = handle.read(1)
            handle.seek(pack["committed_end"] - 1)
            handle.write(bytes([final_byte[0] ^ 0xFF]))
            handle.flush()

        report = store.verify(raise_on_error=False)
        assert report["ok"] is False
        with pytest.raises(VerificationError):
            store.verify()
        with pytest.raises(VerificationError):
            store.completion_receipt(target_unique_tokens=1)


class _CollidingStore(CIContentStore):
    @staticmethod
    def _content_sha256(content: bytes) -> str:
        del content
        return "0" * 64


def test_hash_collision_is_detected_by_exact_byte_comparison(
    tmp_path: Path,
) -> None:
    with _CollidingStore(tmp_path / "store") as store:
        store.add_chunk("first", {"entry": "one"}, _key(0))
        with pytest.raises(HashCollisionError, match="collision"):
            store.add_chunk("other", {"entry": "two"}, _key(1))
        assert store.status()["counters"]["unique_content_count"] == 1
        assert store.status()["counters"]["occurrence_count"] == 1


def test_token_counts_require_one_fingerprint_and_complete_coverage(
    tmp_path: Path,
) -> None:
    with CIContentStore(tmp_path / "store") as store:
        store.add_chunk(
            "counted",
            {"entry": "one"},
            _key(0),
            token_count=5,
            tokenizer_fingerprint="tokenizer-v1",
            token_sequence_sha256=_sequence(1, 2, 3, 4, 5),
        )
        with pytest.raises(
            ContentMetadataConflictError,
            match="fingerprint mismatch",
        ):
            store.add_chunk(
                "different",
                {"entry": "two"},
                _key(1),
                token_count=7,
                tokenizer_fingerprint="tokenizer-v2",
                token_sequence_sha256=_sequence(6, 7, 8, 9, 10, 11, 12),
            )
        store.add_chunk("not counted", {"entry": "three"}, _key(2))
        with pytest.raises(ValueError, match="must be supplied together"):
            store.add_chunk(
                "missing sequence binding",
                {"entry": "four"},
                _key(3),
                token_count=3,
                tokenizer_fingerprint="tokenizer-v1",
            )

        counters = store.status()["counters"]
        assert counters["tokenized_unique_content_count"] == 1
        assert counters["unique_token_sequence_count"] == 1
        assert counters["unique_content_count"] == 2
        assert counters["exact_unique_payload_tokens"] is None
        with pytest.raises(ThresholdNotMetError, match="every unique chunk"):
            store.completion_receipt(target_unique_tokens=1)

        enrichment = store.add_chunk(
            "not counted",
            {"entry": "three"},
            _key(2),
            token_count=4,
            tokenizer_fingerprint="tokenizer-v1",
            token_sequence_sha256=_sequence(20, 21, 22, 23),
        )
        assert enrichment["token_metadata_added"] is True
        assert store.status()["counters"]["exact_unique_payload_tokens"] == 9


def test_exact_token_threshold_deduplicates_normalized_token_sequences(
    tmp_path: Path,
) -> None:
    normalized_sequence = _sequence(101, 202, 303)
    with CIContentStore(tmp_path / "store") as store:
        store.add_chunk(
            "value = 1\n",
            {"entry": "compact"},
            _key(0),
            token_count=3,
            tokenizer_fingerprint="whitespace-normalizing-tokenizer",
            token_sequence_sha256=normalized_sequence,
        )
        store.add_chunk(
            "value  =  1\n",
            {"entry": "spaced"},
            _key(1),
            token_count=3,
            tokenizer_fingerprint="whitespace-normalizing-tokenizer",
            token_sequence_sha256=normalized_sequence,
        )

        counters = store.status()["counters"]
        assert counters["unique_content_count"] == 2
        assert counters["unique_token_sequence_count"] == 1
        assert counters["exact_unique_payload_tokens"] == 3
        with pytest.raises(ThresholdNotMetError, match="below target"):
            store.completion_receipt(target_unique_tokens=4)
        assert (
            store.completion_receipt(target_unique_tokens=3)[
                "exact_unique_payload_tokens"
            ]
            == 3
        )


def test_token_sequence_hash_replay_with_conflicting_length_fails(
    tmp_path: Path,
) -> None:
    sequence_hash = _sequence(1, 2, 3)
    with CIContentStore(tmp_path / "store") as store:
        store.add_chunk(
            "first spelling",
            {"entry": "one"},
            _key(0),
            token_count=3,
            tokenizer_fingerprint="tokenizer-v1",
            token_sequence_sha256=sequence_hash,
        )
        with pytest.raises(
            ContentMetadataConflictError,
            match="conflicting length",
        ):
            store.add_chunk(
                "second spelling",
                {"entry": "two"},
                _key(1),
                token_count=4,
                tokenizer_fingerprint="tokenizer-v1",
                token_sequence_sha256=sequence_hash,
            )
        assert store.status()["counters"]["unique_content_count"] == 1
        assert store.verify()["ok"] is True


def test_threshold_refusal_pass_and_receipt_are_deterministic_across_reopen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    with CIContentStore(root) as store:
        store.add_chunk(
            "alpha",
            {"entry": "one"},
            _key(0),
            token_count=6,
            tokenizer_fingerprint="tok-sha256:abc",
            token_sequence_sha256=_sequence(1, 2, 3, 4, 5, 6),
        )
        store.add_chunk(
            "beta",
            {"entry": "two"},
            _key(1),
            token_count=7,
            tokenizer_fingerprint="tok-sha256:abc",
            token_sequence_sha256=_sequence(7, 8, 9, 10, 11, 12, 13),
        )
        with pytest.raises(ThresholdNotMetError, match="below target"):
            store.completion_receipt(target_unique_tokens=14)
        first = store.completion_receipt(
            target_unique_tokens=13,
            emitted_valid_training_tokens=21,
        )

    with CIContentStore(root) as reopened:
        second = reopened.completion_receipt(
            target_unique_tokens=13,
            emitted_valid_training_tokens=21,
        )

    assert first == second
    assert first["status"] == "complete"
    assert first["exact_unique_payload_tokens"] == 13
    assert first["target_exact_unique_payload_tokens"] == 13
    assert "target_unique_tokens" not in first
    assert first["counters"]["exact_unique_payload_tokens"] == 13
    assert first["emitted_valid_training_tokens"] == 21
    assert len(first["logical_content_set_sha256"]) == 64
    assert len(first["logical_token_sequence_set_sha256"]) == 64
    assert len(first["occurrence_set_sha256"]) == 64
    assert len(first["sqlite_schema_sha256"]) == 64
    assert len(first["sqlite_logical_sha256"]) == 64
    assert first["pack_hashes"]
    assert "database_file_sha256" not in first


def test_orphan_token_sequence_check_is_set_based_and_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    with CIContentStore(root) as store:
        store.add_chunk(
            "referenced",
            {"entry": "one"},
            _key(0),
            token_count=2,
            tokenizer_fingerprint="tokenizer-v1",
            token_sequence_sha256=_sequence(1, 2),
        )
        plan = [
            str(row[3])
            for row in store._connection.execute(
                f"EXPLAIN QUERY PLAN {_ORPHAN_TOKEN_SEQUENCE_QUERY}"
            )
        ]

    assert any("EXCEPT" in step for step in plan)
    assert all("CORRELATED" not in step for step in plan)
    assert sum("SCAN token_sequences" in step for step in plan) == 1
    assert sum("SCAN contents" in step for step in plan) == 1

    with sqlite3.connect(root / "index.sqlite3") as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO token_sequences(
                token_sequence_sha256,token_count,tokenizer_fingerprint
            ) VALUES (?,?,?)
            """,
            (_sequence(90, 91, 92), 3, "tokenizer-v1"),
        )
        connection.execute(
            """
            UPDATE stats
            SET unique_token_sequence_count=unique_token_sequence_count+1,
                exact_unique_payload_tokens=exact_unique_payload_tokens+3
            WHERE singleton=1
            """
        )
        connection.commit()

    with CIContentStore(root) as store:
        with pytest.raises(
            VerificationError,
            match="unreferenced token sequence",
        ):
            store.verify()


def test_cli_status_verify_and_receipt(tmp_path: Path) -> None:
    root = tmp_path / "store"
    with CIContentStore(root) as store:
        store.add_chunk(
            "cli",
            {"entry": "one"},
            _key(0),
            token_count=2,
            tokenizer_fingerprint="tokenizer-v1",
            token_sequence_sha256=_sequence(90, 91),
        )

    script = Path(__file__).parents[1] / "scripts" / "ci_content_store.py"
    for command, expected in (
        (["status", str(root)], "counters"),
        (["verify", str(root)], '"ok": true'),
        (
            ["receipt", str(root), "--target-unique-tokens", "2"],
            '"status": "complete"',
        ),
    ):
        result = subprocess.run(
            [sys.executable, str(script), *command],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert expected in result.stdout
        json.loads(result.stdout)
