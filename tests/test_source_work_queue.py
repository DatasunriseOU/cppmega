from __future__ import annotations

from pathlib import Path

import pytest

from scripts.distributed_data_prep._common import ContractError, atomic_write_json
from scripts.distributed_data_prep.source_manifest import build_source_manifest
from scripts.distributed_data_prep.source_work_queue import (
    assignment_claim_uri,
    assignment_diagnostic_uri,
    build_assignment_claim,
    claim_assignment,
    publish_assignment_diagnostic,
    publish_assignment_heartbeat,
    publish_assignment_outcome,
    validate_assignment_claim,
)
from scripts.distributed_data_prep.source_worker import (
    LocalObjectStore,
    TransientTransportError,
    run_source_worker,
)


def _manifest() -> tuple[dict[str, object], dict[str, object]]:
    manifest = build_source_manifest(
        [
            {
                "repo": "project",
                "project_id": "owner/project",
                "source": {
                    "kind": "git_mirror",
                    "remote_url": "https://github.com/owner/project.git",
                    "expected_commit": "1" * 40,
                    "expected_tree": None,
                },
            }
        ],
        worker_count=2,
        gcs_output_prefix="gs://queue-test/source-run",
        code_revision="2" * 40,
        indexer_sha256="3" * 64,
        tokenizer_sha256="4" * 64,
        quarantine_manifest_sha256="5" * 64,
    )
    job = manifest["repositories"][0]
    assert isinstance(job, dict)
    return manifest, job


def _executor(slot_index: int) -> dict[str, object]:
    return {
        "physical_worker_index": 0,
        "physical_worker_count": 1,
        "slots_per_worker": 2,
        "slot_index": slot_index,
        "worker": f"worker-{slot_index:04d}",
    }


def _claim(
    *,
    manifest: dict[str, object],
    job: dict[str, object],
    executor: dict[str, object],
    instance: str,
    now: int,
    store: LocalObjectStore,
    root: Path,
):
    return claim_assignment(
        manifest=manifest,
        manifest_file_sha256="6" * 64,
        job=job,
        executor=executor,
        scheduler_instance=instance,
        now_unix_s=now,
        lease_seconds=30,
        heartbeat_seconds=5,
        max_attempts=10,
        object_store=store,
        verification_root=root,
    )


def test_immutable_heartbeat_keeps_a_live_claim_and_expiry_allows_takeover(
    tmp_path: Path,
) -> None:
    manifest, job = _manifest()
    store = LocalObjectStore(tmp_path / "objects")
    first = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(0),
        instance="vm-0.boot-a",
        now=1_000,
        store=store,
        root=tmp_path / "verify-a",
    )
    assert first.state == "claimed"
    assert first.lease is not None

    busy = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(1),
        instance="vm-0.boot-b",
        now=1_001,
        store=store,
        root=tmp_path / "verify-b",
    )
    assert busy.state == "busy"

    assert (
        publish_assignment_heartbeat(
            manifest=manifest,
            lease=first.lease,
            now_unix_s=1_030,
            object_store=store,
            verification_root=tmp_path / "heartbeat",
        )
        == 6
    )
    still_busy = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(1),
        instance="vm-0.boot-b",
        now=1_040,
        store=store,
        root=tmp_path / "verify-c",
    )
    assert still_busy.state == "busy"

    takeover = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(1),
        instance="vm-0.boot-b",
        now=1_061,
        store=store,
        root=tmp_path / "verify-d",
    )
    assert takeover.state == "claimed"
    assert takeover.lease is not None
    assert takeover.lease.claim["attempt"] == 1
    assert takeover.lease.claim["assignment"] == first.lease.claim["assignment"]


def test_exit_75_advances_the_attempt_but_exit_2_stops_all_retries(
    tmp_path: Path,
) -> None:
    manifest, job = _manifest()
    store = LocalObjectStore(tmp_path / "objects")
    first = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(0),
        instance="vm-0.boot-a",
        now=2_000,
        store=store,
        root=tmp_path / "verify-a",
    )
    assert first.lease is not None
    transient = publish_assignment_outcome(
        manifest=manifest,
        lease=first.lease,
        worker_exit_code=75,
        now_unix_s=2_001,
        object_store=store,
        verification_root=tmp_path / "transient",
    )
    assert transient["status"] == "transient"

    second = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(1),
        instance="vm-0.boot-b",
        now=2_002,
        store=store,
        root=tmp_path / "verify-b",
    )
    assert second.lease is not None
    assert second.lease.claim["attempt"] == 1
    deterministic = publish_assignment_outcome(
        manifest=manifest,
        lease=second.lease,
        worker_exit_code=2,
        now_unix_s=2_003,
        object_store=store,
        verification_root=tmp_path / "deterministic",
    )
    assert deterministic["status"] == "deterministic"

    stopped = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(0),
        instance="vm-0.boot-c",
        now=2_004,
        store=store,
        root=tmp_path / "verify-c",
    )
    assert stopped.state == "deterministic"
    assert stopped.outcome == deterministic


def test_failed_attempt_log_is_immutably_published_before_outcome(
    tmp_path: Path,
) -> None:
    manifest, job = _manifest()
    store = LocalObjectStore(tmp_path / "objects")
    decision = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(0),
        instance="vm-0.boot-diagnostic",
        now=3_000,
        store=store,
        root=tmp_path / "verify",
    )
    assert decision.lease is not None
    log_path = tmp_path / "attempt.log"
    log_path.write_bytes(b"exact deterministic parser failure\n")

    receipt = publish_assignment_diagnostic(
        manifest=manifest,
        lease=decision.lease,
        log_path=log_path,
        object_store=store,
    )
    expected_uri = assignment_diagnostic_uri(
        manifest,
        job,
        0,
        decision.lease.claim_sha256,
    )
    assert receipt["uri"] == expected_uri
    assert receipt["size_bytes"] == log_path.stat().st_size
    downloaded = tmp_path / "downloaded.log"
    store.download(expected_uri, downloaded, generation=str(receipt["generation"]))
    assert downloaded.read_bytes() == log_path.read_bytes()

    log_path.write_bytes(b"different bytes\n")
    with pytest.raises(ContractError, match="immutable object collision"):
        publish_assignment_diagnostic(
            manifest=manifest,
            lease=decision.lease,
            log_path=log_path,
            object_store=store,
        )


def test_expired_owner_cannot_publish_an_outcome_after_takeover_window(
    tmp_path: Path,
) -> None:
    manifest, job = _manifest()
    store = LocalObjectStore(tmp_path / "objects")
    first = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(0),
        instance="vm-0.boot-a",
        now=4_000,
        store=store,
        root=tmp_path / "verify-a",
    )
    assert first.lease is not None

    with pytest.raises(TransientTransportError, match="lease expired"):
        publish_assignment_outcome(
            manifest=manifest,
            lease=first.lease,
            worker_exit_code=2,
            now_unix_s=4_031,
            object_store=store,
            verification_root=tmp_path / "stale-outcome",
        )

    takeover = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(1),
        instance="vm-0.boot-b",
        now=4_031,
        store=store,
        root=tmp_path / "verify-b",
    )
    assert takeover.state == "claimed"
    assert takeover.lease is not None
    assert takeover.lease.claim["attempt"] == 1


def test_successor_claim_fences_late_heartbeat_and_outcome(
    tmp_path: Path,
) -> None:
    manifest, job = _manifest()
    store = LocalObjectStore(tmp_path / "objects")
    first = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(0),
        instance="vm-0.boot-a",
        now=5_000,
        store=store,
        root=tmp_path / "verify-a",
    )
    assert first.lease is not None

    successor = build_assignment_claim(
        manifest=manifest,
        manifest_file_sha256="6" * 64,
        job=job,
        attempt=1,
        executor=_executor(1),
        scheduler_instance="vm-0.boot-b",
        now_unix_s=5_001,
        lease_seconds=30,
        heartbeat_seconds=5,
    )
    successor_path = tmp_path / "successor.json"
    atomic_write_json(successor_path, successor)
    store.publish_if_absent(successor_path, assignment_claim_uri(manifest, job, 1))

    with pytest.raises(TransientTransportError, match="superseded"):
        publish_assignment_heartbeat(
            manifest=manifest,
            lease=first.lease,
            now_unix_s=5_005,
            object_store=store,
            verification_root=tmp_path / "late-heartbeat",
        )
    with pytest.raises(TransientTransportError, match="superseded"):
        publish_assignment_outcome(
            manifest=manifest,
            lease=first.lease,
            worker_exit_code=2,
            now_unix_s=5_005,
            object_store=store,
            verification_root=tmp_path / "late-outcome",
        )

    decision = _claim(
        manifest=manifest,
        job=job,
        executor=_executor(0),
        instance="vm-0.boot-a",
        now=5_006,
        store=store,
        root=tmp_path / "verify-b",
    )
    assert decision.state == "busy"


def test_claim_validation_rejects_assignment_and_topology_drift() -> None:
    manifest, job = _manifest()
    claim = build_assignment_claim(
        manifest=manifest,
        manifest_file_sha256="6" * 64,
        job=job,
        attempt=0,
        executor=_executor(0),
        scheduler_instance="vm-0.boot-a",
        now_unix_s=3_000,
        lease_seconds=30,
        heartbeat_seconds=5,
    )
    tampered = dict(claim)
    tampered_assignment = dict(claim["assignment"])
    tampered_assignment["worker"] = "worker-0001"
    tampered["assignment"] = tampered_assignment
    with pytest.raises(ContractError, match="assignment drifted"):
        validate_assignment_claim(
            tampered,
            manifest=manifest,
            manifest_file_sha256="6" * 64,
            job=job,
        )

    with pytest.raises(ContractError, match="topology"):
        build_assignment_claim(
            manifest=manifest,
            manifest_file_sha256="6" * 64,
            job=job,
            attempt=0,
            executor={**_executor(0), "worker": "worker-0001"},
            scheduler_instance="vm-0.boot-a",
            now_unix_s=3_000,
            lease_seconds=30,
            heartbeat_seconds=5,
        )


def test_single_assignment_selector_cannot_impersonate_manifest_owner(
    tmp_path: Path,
) -> None:
    manifest, job = _manifest()
    with pytest.raises(ContractError, match="not owned by the requested"):
        run_source_worker(
            manifest,
            manifest_file_sha256="6" * 64,
            worker="worker-0001",
            scratch_root=tmp_path / "scratch",
            receipt_root=tmp_path / "receipts",
            repo_root=tmp_path,
            python=Path("/usr/bin/python3"),
            indexer=tmp_path / "missing-indexer.py",
            tokenizer=tmp_path / "missing-tokenizer.json",
            quarantine_manifest=tmp_path / "missing-quarantine.json",
            object_store=LocalObjectStore(tmp_path / "objects"),
            assignment_sha256=str(job["assignment_sha256"]),
        )
