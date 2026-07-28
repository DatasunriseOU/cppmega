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
        "members": [
            {
                "archive_member": "0_build.txt",
                "job_key": "10:0_build.txt",
                "raw_sha256": hashlib.sha256(payload).hexdigest(),
                "raw_size": len(payload),
                "chunk_count": 1,
                "occurrence_tokens": 7,
            }
        ],
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
        audit_evidence = json.loads(audit[3])
        assert audit[3] == recovery._canonical_json_bytes(
            audit_evidence
        ).decode("utf-8")
        assert audit_evidence == {
            "schema": ci.PRESERVED_RECOVERY_LEDGER_SCHEMA,
            "producer_lineage": ci._producer_lineage(
                recovery._producer_binding()
            ),
            "recovery_id": receipt["recovery_id"],
            "receipt": {
                "name": Path(str(result["receipt"])).name,
                "bytes": Path(str(result["receipt"])).stat().st_size,
                "sha256": result["receipt_sha256"],
            },
            "source_row_sha256": plan.row_sha256,
            "witness_set_sha256": plan.witness_set_sha256,
            "archive": {
                "sha256": result["archive_sha256"],
                "bytes": result["archive_bytes"],
            },
        }

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
    source = ci.RescueSpool(rescue).locate(
        attempt,
        audit=audit_evidence,
    )
    assert isinstance(source, ci.ArchiveSource)
    assert source.raw_sha256 == receipt["proof"]["source_archive"]["sha256"]
    assert source.source == "preserved-local-archive"
    with pytest.raises(
        ci.ArchiveError,
        match="audit sidecar|authorization",
    ):
        ci.RescueSpool(rescue).locate(attempt, audit=None)
    forged_audit = json.loads(json.dumps(audit_evidence))
    forged_audit["witness_set_sha256"] = "0" * 64
    with pytest.raises(
        ci.ArchiveError,
        match="recovery authorization",
    ):
        ci.RescueSpool(rescue).locate(
            attempt,
            audit=forged_audit,
        )
    receipt_path = Path(str(result["receipt"]))
    exact_receipt_bytes = receipt_path.read_bytes()
    forged_receipt = json.loads(exact_receipt_bytes)
    forged_receipt["proof"]["durable_member_witness"]["members"][0][
        "raw_sha256"
    ] = "0" * 64
    receipt_path.write_bytes(
        recovery._canonical_json_bytes(forged_receipt) + b"\n"
    )
    try:
        with pytest.raises(
            ci.ArchiveError,
            match="recovery authorization",
        ):
            ci.RescueSpool(rescue).locate(
                attempt,
                audit=audit_evidence,
            )
    finally:
        receipt_path.write_bytes(exact_receipt_bytes)

    assert recovery.build_plans(
        state_path=state,
        work_dir=work,
        rescue_spool=rescue,
    ) == ()


def test_legacy_preserved_recovery_audit_migrates_append_only(
    tmp_path: Path,
) -> None:
    payload = b"compile output\n"
    state = _write_state(
        tmp_path / "fetch.sqlite",
        member_payload=payload,
    )
    work = tmp_path / "work"
    _write_candidate(work, member_payload=payload)
    rescue = tmp_path / "rescue"
    rescue.mkdir()
    [plan] = recovery.build_plans(
        state_path=state,
        work_dir=work,
        rescue_spool=rescue,
    )
    result = recovery.apply_plan(plan)

    current_receipt_path = Path(str(result["receipt"]))
    receipt = json.loads(current_receipt_path.read_bytes())
    receipt.pop("producer_binding")
    prior_sha = "9d4d3f21" + "0" * 56
    receipt["proof"]["recovery_script_sha256"] = prior_sha
    receipt["recovery_id"] = recovery._sha256_bytes(
        recovery._canonical_json_bytes(receipt["proof"])
    )
    legacy_receipt_path = (
        rescue
        / (
            "owner__repo--1--attempt-1.preserved-recovery-"
            f"{receipt['recovery_id'][:16]}.json"
        )
    )
    legacy_receipt_raw = recovery._canonical_json_bytes(receipt) + b"\n"
    legacy_receipt_path.write_bytes(legacy_receipt_raw)
    current_receipt_path.unlink()
    legacy_receipt_sha = hashlib.sha256(legacy_receipt_raw).hexdigest()
    legacy_audit = (
        f"recovery_id={receipt['recovery_id']} "
        f"receipt_sha256={legacy_receipt_sha} "
        f"source_row_sha256={plan.row_sha256}"
    )
    with sqlite3.connect(state) as connection:
        connection.execute(
            """
            UPDATE request_ledger SET error_message=?
            WHERE endpoint='operator/preserved_archive_recovery'
              AND outcome='operator/preserved_archive_recovery'
              AND error_class='PreservedArchiveRecoveryReceipt'
            """,
            (legacy_audit,),
        )
        connection.execute(
            """
            UPDATE attempts SET error_message=?
            WHERE repo='owner/repo' AND run_id=1 AND attempt=1
            """,
            (legacy_audit,),
        )

    with pytest.raises(
        recovery.RecoveryError,
        match="explicit producer upgrade authorization",
    ):
        recovery.migrate_producer_lineage(
            state_path=state,
            rescue_spool=rescue,
            target=("owner/repo", 1, 1),
            allow_producer_upgrade_from_sha256=None,
            producer_upgrade_reason=None,
        )

    reason = "authorize audited migration of preserved recovery lineage"
    migrated = recovery.migrate_producer_lineage(
        state_path=state,
        rescue_spool=rescue,
        target=("owner/repo", 1, 1),
        allow_producer_upgrade_from_sha256=prior_sha,
        producer_upgrade_reason=reason,
    )
    assert migrated["status"] == "migrated"
    assert legacy_receipt_path.read_bytes() == legacy_receipt_raw

    with sqlite3.connect(state) as connection:
        rows = connection.execute(
            """
            SELECT error_message FROM request_ledger
            WHERE endpoint='operator/preserved_archive_recovery'
              AND outcome='operator/preserved_archive_recovery'
              AND error_class='PreservedArchiveRecoveryReceipt'
            ORDER BY id
            """
        ).fetchall()
        attempt_error = connection.execute(
            """
            SELECT error_message FROM attempts
            WHERE repo='owner/repo' AND run_id=1 AND attempt=1
            """
        ).fetchone()[0]
    assert len(rows) == 2
    assert rows[0][0] == legacy_audit
    audit = json.loads(rows[1][0])
    assert attempt_error == rows[1][0]
    lineage = audit["producer_lineage"]
    assert lineage["origin"] == {
        "script_sha256": prior_sha,
        "semantic_contract_sha256": (
            ci.PRESERVED_RECOVERY_LEGACY_SEMANTIC_CONTRACT_SHA256
        ),
    }
    assert lineage["current"] == recovery._producer_binding()
    assert lineage["upgrades"][-1]["reason"] == reason

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
    source = ci.RescueSpool(rescue).locate(
        attempt,
        audit=audit,
    )
    assert isinstance(source, ci.ArchiveSource)
    assert source.source == "preserved-local-archive"


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
