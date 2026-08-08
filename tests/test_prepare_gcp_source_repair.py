from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
)
from scripts.distributed_data_prep.source_manifest import build_source_manifest
from scripts.distributed_data_prep.source_work_queue import (
    assignment_claim_uri,
    assignment_outcome_uri,
    build_assignment_claim,
)
from scripts.distributed_data_prep.source_worker import assignment_completion_uri
from scripts.prepare_gcp_source_repair import load_repair_evidence
from scripts.prepare_gcp_source_pilot import render_runner


def _metadata(path: Path, *, kind: str) -> dict[str, object]:
    return {
        "kind": kind,
        "generation": "1",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _evidence_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    root = tmp_path / "evidence"
    root.mkdir()
    repositories = []
    for index in range(4):
        repositories.append(
            {
                "repo": f"project-{index}",
                "project_id": f"owner/project-{index}",
                "source": {
                    "kind": "git_mirror",
                    "remote_url": f"https://github.com/owner/project-{index}.git",
                    "expected_commit": f"{index + 1}" * 40,
                    "expected_tree": None,
                },
            }
        )
    manifest = build_source_manifest(
        repositories,
        worker_count=4,
        gcs_output_prefix="gs://repair-test/runs/base",
        code_revision="a" * 40,
        indexer_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        quarantine_manifest_sha256="d" * 64,
    )
    manifest_path = root / "source-manifest.json"
    _write_json(manifest_path, manifest)
    manifest_file_sha = sha256_file(manifest_path)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    completed_job, failed_job, active_job, _stale_job = jobs
    assert all(isinstance(job, dict) for job in jobs)
    failed_job = dict(failed_job)

    claim = build_assignment_claim(
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha,
        job=failed_job,
        attempt=0,
        executor={
            "physical_worker_index": 0,
            "physical_worker_count": 2,
            "slots_per_worker": 2,
            "slot_index": 0,
            "worker": "worker-0000",
        },
        scheduler_instance="repair-test.boot-a",
        now_unix_s=50,
        lease_seconds=900,
        heartbeat_seconds=120,
    )
    claim_sha = canonical_sha256(claim)
    assignment_fields = (
        "ordinal",
        "repo",
        "project_id",
        "worker",
        "assignment_sha256",
    )
    outcome = {
        "schema": "cppmega.distributed_source_assignment_attempt_outcome_v1",
        "status": "deterministic",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha,
        "assignment": {field: failed_job[field] for field in assignment_fields},
        "attempt": 0,
        "claim_sha256": claim_sha,
        "executor": claim["executor"],
        "scheduler_instance": claim["scheduler_instance"],
        "worker_exit_code": 2,
        "published_unix_s": 60,
        "training_ready": False,
    }
    manifest_sha = str(manifest["manifest_sha256"])
    assignment = str(failed_job["assignment_sha256"])
    claim_path = (
        root / "receipts" / "claims" / manifest_sha / assignment / "0000.claim.json"
    )
    outcome_path = (
        root
        / "receipts"
        / "outcomes"
        / manifest_sha
        / assignment
        / f"0000-{claim_sha}.outcome.json"
    )
    _write_json(claim_path, claim)
    _write_json(outcome_path, outcome)
    claim_uri = assignment_claim_uri(manifest, failed_job, 0)
    outcome_uri = assignment_outcome_uri(manifest, failed_job, 0, claim_sha)
    completed_uri = assignment_completion_uri(manifest, completed_job)
    outcome_sha = sha256_file(outcome_path)

    state = {
        "schema": "cppmega.gcp_source_run_monitor_state_v1",
        "run_id": "base",
        "updated_at_unix": 100,
        "heartbeat_ledger": {
            "schema": "cppmega.gcp_source_heartbeat_ledger_v1",
            "run_id": "base",
            "manifest_sha256": manifest_sha,
            "path": "original.sqlite3",
        },
        "diagnostics": {},
        "workers": {},
        "validated_receipts": {
            claim_uri: _metadata(claim_path, kind="claim"),
            outcome_uri: _metadata(outcome_path, kind="outcome"),
            completed_uri: {
                "kind": "assignment",
                "generation": "1",
                "sha256": "e" * 64,
                "size_bytes": 1,
            },
        },
    }
    report = {
        "schema": "cppmega.gcp_source_run_monitor_report_v1",
        "run_id": "base",
        "run_root": manifest["gcs_output_prefix"],
        "checked_at_unix": 100,
        "state": "blocked_deterministic",
        "counts": {
            "assignment_receipts": 1,
            "expected_assignment_receipts": 4,
            "assignment_outcome_receipts": 1,
            "terminal_assignment_outcomes": 1,
            "deterministic_assignment_outcomes": 1,
            "transient_assignment_outcomes": 0,
            "fresh_heartbeat_assignments": 1,
        },
        "workers": [
            {
                "fresh_assignment_heartbeats": [
                    {
                        "assignment_sha256": active_job["assignment_sha256"],
                        "repo": active_job["repo"],
                    }
                ]
            }
        ],
        "outcome_inventory_sha256": hashlib.sha256(
            canonical_json_bytes([outcome_sha])
        ).hexdigest(),
        "training_ready": False,
    }
    _write_json(root / "watchdog.state.json", state)
    _write_json(root / "watchdog.current.json", report)
    connection = sqlite3.connect(root / "watchdog.heartbeat.sqlite3")
    try:
        connection.execute("CREATE TABLE heartbeat (id INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()
    return root, manifest, outcome_path


def _load(root: Path, manifest: dict[str, object]):
    return load_repair_evidence(
        root,
        expected_base_manifest_file_sha256=sha256_file(root / "source-manifest.json"),
        expected_base_manifest_sha256=str(manifest["manifest_sha256"]),
        expected_base_repository_count=4,
        expected_deterministic_count=1,
    )


def test_receipt_bound_repair_evidence_classifies_every_assignment(
    tmp_path: Path,
) -> None:
    root, manifest, _outcome_path = _evidence_fixture(tmp_path)
    evidence = _load(root, manifest)

    assert evidence.evidence_receipt["counts"] == {
        "assignments": 4,
        "success": 1,
        "deterministic": 1,
        "active": 1,
        "stale": 1,
    }
    assert len(evidence.failed_jobs) == 1
    selected = evidence.evidence_receipt["selected_receipts"]
    assert isinstance(selected, list)
    assert selected[0]["worker_exit_code"] == 2
    assert evidence.evidence_receipt["training_ready"] is False


def test_runner_binds_optional_repair_contract_without_weakening_normal_runs() -> None:
    template = (
        'bundle="__CPPMEGA_BUNDLE_SHA256__"\n'
        'overlay="__CPPMEGA_OVERLAY_SHA256__"\n'
        'manifest="__CPPMEGA_MANIFEST_SHA256__"\n'
        'repair="__CPPMEGA_REPAIR_CONTRACT_SHA256__"\n'
    )
    hashes = {"bundle": "1" * 64, "overlay": "2" * 64, "manifest": "3" * 64}

    normal = render_runner(template, hashes)
    assert 'repair="none"' in normal
    repair = render_runner(template, hashes, repair_contract_sha256="4" * 64)
    assert 'repair="' + "4" * 64 + '"' in repair
    with pytest.raises(ContractError, match="repair contract SHA-256"):
        render_runner(template, hashes, repair_contract_sha256="unpinned")


def test_repair_evidence_rejects_mutated_outcome_bytes(tmp_path: Path) -> None:
    root, manifest, outcome_path = _evidence_fixture(tmp_path)
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["published_unix_s"] = 61
    _write_json(outcome_path, outcome)

    with pytest.raises(ContractError, match="receipt SHA-256 drifted"):
        _load(root, manifest)


def test_repair_evidence_rejects_active_failure_overlap(tmp_path: Path) -> None:
    root, manifest, _outcome_path = _evidence_fixture(tmp_path)
    report_path = root / "watchdog.current.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failed = manifest["repositories"][1]
    report["workers"][0]["fresh_assignment_heartbeats"][0]["assignment_sha256"] = (
        failed["assignment_sha256"]
    )
    _write_json(report_path, report)

    with pytest.raises(ContractError, match="classifications overlap"):
        _load(root, manifest)
