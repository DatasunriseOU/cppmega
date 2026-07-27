from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile
import zlib

import pytest

from scripts import ci_stream_fetch as ci
from scripts import recover_ci_preserved_archives as recovery


def _write_state(
    path: Path,
    *,
    member_payload: bytes,
) -> Path:
    metadata = json.dumps(
        {
            "id": 1,
            "run_attempt": 1,
            "status": "completed",
            "created_at": "2026-04-27T16:01:00Z",
            "repository": {"full_name": "owner/repo", "id": 1},
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    with sqlite3.connect(path) as connection:
        connection.executescript(ci._STATE_SCHEMA)
        connection.execute(
            "INSERT INTO settings(key,value) VALUES ('schema',?)",
            (ci.SCHEMA_VERSION,),
        )
        connection.execute(
            """
            INSERT INTO attempts(
              repo,run_id,attempt,created_at,
              run_metadata_sha256,run_metadata_raw_size,run_metadata_zlib,
              run_metadata_source,run_metadata_source_attempt,
              run_metadata_exact,inventory_seed_attempt,
              inventory_seed_metadata_sha256,status,tries,
              member_count,chunk_count,occurrence_tokens,
              terminal_http_status,terminal_body_sha256,
              error_class,error_message,discovered_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "owner/repo",
                1,
                1,
                "2026-04-27T16:01:00Z",
                hashlib.sha256(metadata).hexdigest(),
                len(metadata),
                sqlite3.Binary(zlib.compress(metadata, 6)),
                "inventory-run-list",
                1,
                1,
                1,
                "a" * 64,
                "terminal_410",
                2,
                0,
                0,
                0,
                410,
                "b" * 64,
                "TerminalHTTP",
                "GitHub HTTP 410",
                "2026-07-27T10:00:00Z",
                "2026-07-27T11:00:00Z",
            ),
        )
        sidecar = b"{}"
        connection.execute(
            """
            INSERT INTO members(
              repo,run_id,attempt,archive_member,job_key,
              raw_sha256,raw_size,canonical_sha256,dedup_sha256,
              sidecar_sha256,sidecar_raw_size,sidecar_zlib,
              chunk_count,occurrence_tokens
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "owner/repo",
                1,
                1,
                "0_build.txt",
                "10:0_build.txt",
                hashlib.sha256(member_payload).hexdigest(),
                len(member_payload),
                "c" * 64,
                "d" * 64,
                hashlib.sha256(sidecar).hexdigest(),
                len(sidecar),
                sqlite3.Binary(zlib.compress(sidecar, 6)),
                1,
                7,
            ),
        )
    return path


def _write_candidate(
    work_dir: Path,
    *,
    member_payload: bytes,
) -> Path:
    failed = work_dir / "failed"
    (work_dir / "tmp").mkdir(parents=True)
    failed.mkdir(parents=True)
    path = failed / "owner__repo--1--1--candidate.zip.partial"
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("0_build.txt", member_payload)
        archive.writestr("1_test.txt", b"tests passed\n")
    return path


def test_preserved_archive_recovery_is_dry_run_then_atomic_requeue(
    tmp_path: Path,
) -> None:
    payload = b"compile output\n"
    state = _write_state(
        tmp_path / "fetch.sqlite",
        member_payload=payload,
    )
    work = tmp_path / "work"
    candidate = _write_candidate(work, member_payload=payload)
    rescue = tmp_path / "rescue"
    rescue.mkdir()

    plans = recovery.build_plans(
        state_path=state,
        work_dir=work,
        rescue_spool=rescue,
        max_archive_bytes=1024 * 1024,
        max_member_bytes=1024 * 1024,
        max_uncompressed_bytes=1024 * 1024,
        max_members=10,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.identity == ("owner/repo", 1, 1)
    assert plan.archive.path == candidate.resolve()
    assert len(plan.witnesses) == 1
    with sqlite3.connect(state) as connection:
        assert connection.execute(
            "SELECT status FROM attempts"
        ).fetchone()[0] == "terminal_410"
    assert list(rescue.iterdir()) == []

    result = recovery.apply_plan(plan)

    assert result["status"] == "requeued"
    rescue_archive = Path(str(result["rescue_archive"]))
    assert rescue_archive.read_bytes() == candidate.read_bytes()
    receipt = json.loads(
        Path(str(result["receipt"])).read_text(encoding="utf-8")
    )
    assert receipt["schema"] == recovery.SCHEMA
    assert receipt["status"] == "verified"
    assert receipt["proof"]["durable_member_witness"] == {
        "count": 1,
        "chunk_count": 1,
        "occurrence_tokens": 7,
        "set_sha256": plan.witness_set_sha256,
    }
    assert receipt["proof"]["source_archive"]["sha256"] == (
        hashlib.sha256(candidate.read_bytes()).hexdigest()
    )
    manifest = (rescue / "manifest.tsv").read_text(encoding="utf-8")
    assert "owner/repo\t1\t1\t" in manifest
    assert "\tzip\t" in manifest

    with sqlite3.connect(state) as connection:
        row = connection.execute(
            """
            SELECT status,tries,archive_source,archive_sha256,archive_size,
                   terminal_http_status,terminal_body_sha256,error_class
            FROM attempts
            """
        ).fetchone()
        assert tuple(row) == (
            "retry",
            0,
            "preserved-local-archive",
            hashlib.sha256(candidate.read_bytes()).hexdigest(),
            candidate.stat().st_size,
            None,
            None,
            "PreservedArchiveRecovery",
        )
        audit = connection.execute(
            """
            SELECT endpoint,outcome,error_class,error_message
            FROM request_ledger
            WHERE endpoint='operator/preserved_archive_recovery'
            """
        ).fetchone()
        assert audit[0:3] == (
            "operator/preserved_archive_recovery",
            "operator/preserved_archive_recovery",
            "PreservedArchiveRecoveryReceipt",
        )
        assert receipt["recovery_id"] in audit[3]

    attempt = ci.Attempt(
        repo="owner/repo",
        run_id=1,
        attempt=1,
        created_at="2026-04-27T16:01:00Z",
        run_metadata={},
        run_metadata_sha256="e" * 64,
        run_metadata_source="inventory-run-list",
        run_metadata_source_attempt=1,
        run_metadata_exact=True,
        inventory_seed_attempt=1,
        inventory_seed_metadata_sha256="f" * 64,
    )
    source = ci.RescueSpool(rescue).locate(attempt)
    assert isinstance(source, ci.ArchiveSource)
    assert source.raw_sha256 == receipt["proof"]["source_archive"]["sha256"]

    assert recovery.build_plans(
        state_path=state,
        work_dir=work,
        rescue_spool=rescue,
    ) == ()


def test_preserved_archive_recovery_rejects_member_mismatch(
    tmp_path: Path,
) -> None:
    state = _write_state(
        tmp_path / "fetch.sqlite",
        member_payload=b"trusted member\n",
    )
    work = tmp_path / "work"
    _write_candidate(work, member_payload=b"different member\n")
    rescue = tmp_path / "rescue"
    rescue.mkdir()

    with pytest.raises(
        recovery.RecoveryError,
        match="no candidate satisfies",
    ):
        recovery.build_plans(
            state_path=state,
            work_dir=work,
            rescue_spool=rescue,
        )

    with sqlite3.connect(state) as connection:
        assert connection.execute(
            "SELECT status FROM attempts"
        ).fetchone()[0] == "terminal_410"
    assert list(rescue.iterdir()) == []
