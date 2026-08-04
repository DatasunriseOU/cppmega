from __future__ import annotations

import fnmatch
import json
import subprocess
from pathlib import Path
from typing import Mapping

import pytest

from scripts.distributed_data_prep._common import atomic_write_json, sha256_file
from scripts.distributed_data_prep.source_manifest import (
    build_source_manifest,
    repositories_for_worker,
)
from scripts.distributed_data_prep.source_slot_scheduler import (
    SLOT_COMPLETION_RECEIPT_SCHEMA,
    slot_specs,
)
from scripts.distributed_data_prep.source_worker import (
    ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA,
    LocalObjectStore,
    assignment_completion_uri,
)
from scripts.gcp_source_run_monitor import (
    GcloudRunClient,
    MONITOR_SCHEMA,
    MonitorError,
    run_monitor,
)

RUN_ID = "source-prod-20260804-003"
RUN_ROOT = f"gs://test-cppmega/runs/{RUN_ID}"
PHYSICAL_WORKERS = [f"cppmega-corpus-{index:02d}-{RUN_ID}" for index in range(4)]
RESOURCES = {
    "parse_workers_per_slot": 6,
    "memory_limit_gb_per_slot": 24.0,
    "cpu_budget_vcpus": 16,
    "memory_budget_gb": 56.0,
}


class FakeRunClient:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[dict[str, object], bytes, dict[str, object]]] = {}
        self.instances = [
            {
                "name": worker,
                "id": str(index + 1),
                "status": "RUNNING",
                "zone": "zones/us-central1-a",
            }
            for index, worker in enumerate(PHYSICAL_WORKERS)
        ]
        self.serial = b"cppmega-source-worker stopped\n"
        self.serial_calls: list[str] = []
        self.serial_call_zones: list[tuple[str, str]] = []

    def add_json(
        self,
        uri: str,
        value: Mapping[str, object],
        *,
        generation: str | None = None,
        updated: str = "2026-08-04T11:00:00Z",
    ) -> None:
        raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        resolved_generation = generation or str(len(self.objects) + 1)
        metadata = {
            "uri": uri,
            "generation": resolved_generation,
            "size_bytes": len(raw),
            "updated": updated,
        }
        self.objects[uri] = (metadata, raw, dict(value))

    def list_objects(self, pattern: str) -> list[dict[str, object]]:
        return [
            dict(metadata)
            for uri, (metadata, _raw, _value) in sorted(self.objects.items())
            if fnmatch.fnmatchcase(uri, pattern)
        ]

    def read_json(
        self, metadata: Mapping[str, object]
    ) -> tuple[bytes, dict[str, object]]:
        stored, raw, value = self.objects[str(metadata["uri"])]
        assert stored["generation"] == metadata["generation"]
        return raw, dict(value)

    def list_instances(
        self, *, project_id: str, run_id: str
    ) -> list[dict[str, object]]:
        assert project_id == "test-project"
        assert run_id == RUN_ID
        return [dict(row) for row in self.instances]

    def serial_output(self, *, project_id: str, zone: str, instance: str) -> bytes:
        assert project_id == "test-project"
        self.serial_calls.append(instance)
        self.serial_call_zones.append((instance, zone))
        return self.serial


class FailingObjectStore:
    def publish_if_absent(self, source: Path, uri: str) -> Mapping[str, object]:
        raise RuntimeError("diagnostics upload unavailable")

    def download(
        self, uri: str, destination: Path, *, generation: str | None = None
    ) -> Mapping[str, object]:
        raise RuntimeError("diagnostics download unavailable")


class FailSecondPublishOnceStore:
    def __init__(self, root: Path) -> None:
        self.inner = LocalObjectStore(root)
        self.calls = 0
        self.failed = False

    def publish_if_absent(self, source: Path, uri: str) -> Mapping[str, object]:
        self.calls += 1
        if self.calls == 2 and not self.failed:
            self.failed = True
            raise RuntimeError("diagnostics receipt upload interrupted")
        return self.inner.publish_if_absent(source, uri)

    def download(
        self, uri: str, destination: Path, *, generation: str | None = None
    ) -> Mapping[str, object]:
        return self.inner.download(uri, destination, generation=generation)


def test_gcloud_empty_object_pattern_is_an_empty_inventory() -> None:
    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            argv,
            1,
            b"",
            b"ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n",
        )

    client = GcloudRunClient("gcloud", runner=runner)
    assert client.list_objects(f"{RUN_ROOT}/control/failed/*.json") == []


def test_gcloud_object_listing_does_not_hide_other_exit_one_errors() -> None:
    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 1, b"", b"permission denied\n")

    client = GcloudRunClient("gcloud", runner=runner)
    with pytest.raises(MonitorError, match="permission denied"):
        client.list_objects(f"{RUN_ROOT}/control/failed/*.json")


def test_ready_receipt_requires_the_configured_local_ssd_count(tmp_path: Path) -> None:
    manifest_path, _manifest_value = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    uri = next(uri for uri in client.objects if "/control/ready/" in uri)
    metadata, _raw, value = client.objects[uri]
    value["local_ssd_count"] = 1
    client.objects[uri] = (
        metadata,
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        value,
    )

    with pytest.raises(MonitorError, match="Local SSD count drifted"):
        run_monitor(
            config,
            client=client,
            object_store=LocalObjectStore(tmp_path / "gcs"),
            now=lambda: 100,
        )


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    repositories = [
        {
            "repo": f"repo-{index:02d}",
            "project_id": f"Org/repo-{index:02d}",
            "source": {
                "kind": "git_mirror",
                "remote_url": f"https://github.com/Org/repo-{index:02d}.git",
                "expected_commit": f"{index + 1:040x}",
                "expected_tree": None,
            },
        }
        for index in range(16)
    ]
    manifest = build_source_manifest(
        repositories,
        worker_count=8,
        gcs_output_prefix=RUN_ROOT,
        code_revision="1" * 40,
        indexer_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        quarantine_manifest_sha256="4" * 64,
    )
    path = tmp_path / "source-manifest.json"
    atomic_write_json(path, manifest)
    return path, manifest


def _config(tmp_path: Path, manifest_path: Path) -> dict[str, object]:
    return {
        "schema": MONITOR_SCHEMA,
        "run_id": RUN_ID,
        "run_root": RUN_ROOT,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "project_id": "test-project",
        "zone": "us-central1-a",
        "physical_workers": PHYSICAL_WORKERS,
        "slots_per_worker": 2,
        "expected_local_ssd_count": 2,
        "resources": RESOURCES,
        "state_path": str(tmp_path / "state.json"),
        "report_path": str(tmp_path / "report.json"),
        "terminal_receipt_path": str(tmp_path / "terminal.json"),
        "diagnostics_dir": str(tmp_path / "diagnostics"),
        "diagnostics_upload_prefix": f"{RUN_ROOT}/diagnostics/gcp-source-monitor",
        "stale_after_seconds": 1800,
        "gcloud": "/opt/homebrew/bin/gcloud",
    }


def _boot_id(index: int) -> str:
    return f"00000000-0000-4000-8000-{index + 1:012d}"


def _add_ready(client: FakeRunClient) -> None:
    for index, worker in enumerate(PHYSICAL_WORKERS):
        boot_id = _boot_id(index)
        client.add_json(
            f"{RUN_ROOT}/control/ready/{worker}.{boot_id}.json",
            {
                "schema_version": 1,
                "state": "ready",
                "run_id": RUN_ID,
                "worker_name": worker,
                "boot_id": boot_id,
                "created_at": "2026-08-04T11:00:00Z",
                "local_ssd_count": 2,
                "local_stage_bytes": 750_000_000_000,
            },
        )


def _add_failure(client: FakeRunClient, *, worker_index: int, exit_code: int) -> None:
    worker = PHYSICAL_WORKERS[worker_index]
    boot_id = _boot_id(worker_index)
    client.add_json(
        f"{RUN_ROOT}/control/failed/{worker}.{boot_id}.json",
        {
            "schema_version": 1,
            "state": "failed",
            "worker": f"worker-{worker_index:04d}",
            "worker_name": worker,
            "boot_id": boot_id,
            "created_at": "2026-08-04T11:30:00Z",
            "exit_code": exit_code,
        },
        updated="2026-08-04T11:30:00Z",
    )


def _add_claim(
    client: FakeRunClient,
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    job: Mapping[str, object],
    physical_worker_index: int,
    slot_index: int = 0,
    attempt: int = 0,
) -> None:
    logical_worker = f"worker-{physical_worker_index * 2 + slot_index:04d}"
    assignment_sha256 = str(job["assignment_sha256"])
    client.add_json(
        f"{RUN_ROOT}/source-assignment-claims/{manifest['manifest_sha256']}/"
        f"{assignment_sha256}/{attempt:04d}.claim.json",
        {
            "schema": "cppmega.distributed_source_assignment_claim_v1",
            "status": "claimed",
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": manifest_file_sha256,
            "assignment": {
                key: job[key]
                for key in (
                    "ordinal",
                    "repo",
                    "project_id",
                    "worker",
                    "assignment_sha256",
                )
            },
            "attempt": attempt,
            "executor": {
                "physical_worker_index": physical_worker_index,
                "physical_worker_count": 4,
                "slots_per_worker": 2,
                "slot_index": slot_index,
                "worker": logical_worker,
            },
            "scheduler_instance": f"{PHYSICAL_WORKERS[physical_worker_index]}.test",
            "created_unix_s": 10,
            "expires_unix_s": 910,
            "lease_seconds": 900,
            "heartbeat_seconds": 120,
            "training_ready": False,
        },
    )


def _source_receipt_entry(
    manifest: Mapping[str, object], job: Mapping[str, object]
) -> dict[str, object]:
    uri = (
        f"{RUN_ROOT}/source-receipts/{manifest['manifest_sha256']}/"
        f"{int(job['ordinal']):05d}-{job['repo']}/{'a' * 64}.receipt.json"
    )
    return {
        "uri": uri,
        "generation": "1",
        "size_bytes": 100,
        "sha256": "b" * 64,
    }


def _add_all_completions(
    client: FakeRunClient, manifest: dict[str, object], manifest_file_sha256: str
) -> None:
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    for job in jobs:
        client.add_json(
            assignment_completion_uri(manifest, job),
            {
                "schema": ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA,
                "status": "complete",
                "manifest_sha256": manifest["manifest_sha256"],
                "manifest_file_sha256": manifest_file_sha256,
                "assignment": {
                    key: job[key]
                    for key in (
                        "ordinal",
                        "repo",
                        "project_id",
                        "worker",
                        "assignment_sha256",
                    )
                },
                "source_receipt": _source_receipt_entry(manifest, job),
                "training_ready": False,
            },
        )
    for physical_index in range(4):
        specs = slot_specs(
            physical_worker_index=physical_index,
            physical_worker_count=4,
            slots_per_worker=2,
        )
        for spec in specs:
            source_receipts = []
            for job in repositories_for_worker(manifest, spec.worker):
                source_receipts.append(
                    {
                        **{
                            key: job[key]
                            for key in (
                                "ordinal",
                                "repo",
                                "project_id",
                                "worker",
                                "assignment_sha256",
                            )
                        },
                        **_source_receipt_entry(manifest, job),
                    }
                )
            client.add_json(
                f"{RUN_ROOT}/source-slot-receipts/{manifest['manifest_sha256']}/"
                f"{spec.worker}.complete.json",
                {
                    "schema": SLOT_COMPLETION_RECEIPT_SCHEMA,
                    "status": "complete",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "manifest_file_sha256": manifest_file_sha256,
                    "topology": {
                        "physical_worker_index": spec.physical_worker_index,
                        "physical_worker_count": spec.physical_worker_count,
                        "slots_per_worker": spec.slots_per_worker,
                        "slot_index": spec.slot_index,
                        "worker": spec.worker,
                    },
                    "resources": RESOURCES,
                    "source_receipts": source_receipts,
                    "training_ready": False,
                },
            )
        worker = PHYSICAL_WORKERS[physical_index]
        owned = [
            job for job in jobs if job["worker"] in {spec.worker for spec in specs}
        ]
        client.add_json(
            f"{RUN_ROOT}/control/completed/{worker}.{_boot_id(physical_index)}.json",
            {
                "schema_version": 1,
                "state": "complete",
                "worker": f"worker-{physical_index:04d}",
                "worker_name": worker,
                "boot_id": _boot_id(physical_index),
                "created_at": "2026-08-04T12:00:00Z",
                "manifest_file_sha256": manifest_file_sha256,
                "receipt_count": len(owned),
                "slots_per_worker": 2,
                "logical_worker_count": 8,
                "completed_slots": [spec.worker for spec in specs],
                "resumed_slots": [],
            },
            updated="2026-08-04T12:00:00Z",
        )


def test_running_run_becomes_idle_only_after_unchanged_stale_window(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest_value = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    store = LocalObjectStore(tmp_path / "gcs")

    first = run_monitor(config, client=client, object_store=store, now=lambda: 100)
    second = run_monitor(config, client=client, object_store=store, now=lambda: 2000)

    assert first["state"] == "running"
    assert {worker["state"] for worker in first["workers"]} == {"running"}
    assert second["state"] == "manual_review"
    assert {worker["state"] for worker in second["workers"]} == {
        "idle_suspected_manual_review"
    }
    assert second["training_ready"] is False


def test_dynamic_claims_are_counted_by_executor_not_manifest_home_worker(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    job = jobs[0]
    assert job["worker"] == "worker-0000"
    _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
        physical_worker_index=1,
    )
    client.add_json(
        assignment_completion_uri(manifest, job),
        {
            "schema": ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA,
            "status": "complete",
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": config["manifest_file_sha256"],
            "assignment": {
                key: job[key]
                for key in (
                    "ordinal",
                    "repo",
                    "project_id",
                    "worker",
                    "assignment_sha256",
                )
            },
            "source_receipt": _source_receipt_entry(manifest, job),
            "training_ready": False,
        },
    )

    result = run_monitor(
        config,
        client=client,
        object_store=LocalObjectStore(tmp_path / "gcs"),
        now=lambda: 100,
    )

    assert result["scheduler_mode"] == "dynamic_claim_queue"
    assert result["counts"]["assignment_receipts"] == 1
    assert result["counts"]["assignment_claim_receipts"] == 1
    assert result["counts"]["claimed_assignments"] == 1
    assert result["workers"][0]["assignment_receipts"] == 1
    assert result["workers"][0]["claim_receipts"] == 0
    assert result["workers"][1]["claim_receipts"] == 1
    assert result["workers"][1]["completed_claimed_assignments"] == 1


def test_exit_75_is_recoverable_only_after_diagnostics_publication(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest_value = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    _add_failure(client, worker_index=0, exit_code=75)
    store = LocalObjectStore(tmp_path / "gcs")

    first = run_monitor(config, client=client, object_store=store, now=lambda: 100)
    second = run_monitor(config, client=client, object_store=store, now=lambda: 200)

    failed = first["workers"][0]
    assert first["state"] == "recoverable_transient"
    assert failed["state"] == "transient_failure_diagnostics_preserved"
    assert failed["recovery_evidence"] == "exit_75"
    assert failed["replacement_permitted"] is True
    assert failed["diagnostics"]["status"] == "published"
    assert first["recovery_policy"]["automatic_replacement_performed"] is False
    assert client.serial_calls == [PHYSICAL_WORKERS[0]]
    assert second["workers"][0]["diagnostics"] == failed["diagnostics"]


def test_failure_diagnostics_use_each_instances_inventory_zone(tmp_path: Path) -> None:
    manifest_path, _manifest_value = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    worker = PHYSICAL_WORKERS[1]
    client.instances[1]["zone"] = (
        "https://www.googleapis.com/compute/v1/projects/test-project/"
        "zones/us-central1-f"
    )
    _add_ready(client)
    _add_failure(client, worker_index=1, exit_code=75)

    result = run_monitor(
        config,
        client=client,
        object_store=LocalObjectStore(tmp_path / "gcs"),
        now=lambda: 100,
    )

    assert result["workers"][1]["zone"] == "us-central1-f"
    assert client.serial_call_zones == [(worker, "us-central1-f")]


def test_missing_instance_reuses_its_last_confirmed_inventory_zone(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest_value = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    worker = PHYSICAL_WORKERS[1]
    client.instances[1]["zone"] = "zones/us-central1-f"
    _add_ready(client)
    store = LocalObjectStore(tmp_path / "gcs")
    run_monitor(config, client=client, object_store=store, now=lambda: 100)

    client.instances = [row for row in client.instances if row["name"] != worker]
    _add_failure(client, worker_index=1, exit_code=75)
    result = run_monitor(config, client=client, object_store=store, now=lambda: 200)

    assert result["workers"][1]["instance_status"] == "MISSING"
    assert result["workers"][1]["zone"] == "us-central1-f"
    assert client.serial_call_zones == [(worker, "us-central1-f")]


def test_exit_75_is_not_recoverable_when_diagnostics_publication_fails(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest_value = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    _add_failure(client, worker_index=0, exit_code=75)

    result = run_monitor(
        config,
        client=client,
        object_store=FailingObjectStore(),
        now=lambda: 100,
    )

    failed = result["workers"][0]
    assert result["state"] == "recovery_blocked_diagnostics"
    assert failed["state"] == "transient_failure_recovery_blocked"
    assert failed["replacement_permitted"] is False
    assert "diagnostics upload unavailable" in failed["diagnostics_error"]


def test_diagnostics_receipt_resume_reuses_frozen_serial_snapshot(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest_value = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    _add_failure(client, worker_index=0, exit_code=75)
    store = FailSecondPublishOnceStore(tmp_path / "gcs")

    first = run_monitor(config, client=client, object_store=store, now=lambda: 100)
    assert first["workers"][0]["state"] == "transient_failure_recovery_blocked"
    client.serial = b"later serial output containing HTTP 429\n"
    second = run_monitor(config, client=client, object_store=store, now=lambda: 200)

    diagnostics = second["workers"][0]["diagnostics"]
    assert second["workers"][0]["state"] == "transient_failure_diagnostics_preserved"
    assert diagnostics["confirmed_http_429"] is False
    assert client.serial_calls == [PHYSICAL_WORKERS[0]]


def test_newer_ready_boot_supersedes_old_transient_failure(tmp_path: Path) -> None:
    manifest_path, _manifest_value = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    _add_failure(client, worker_index=0, exit_code=75)
    worker = PHYSICAL_WORKERS[0]
    new_boot_id = _boot_id(20)
    client.add_json(
        f"{RUN_ROOT}/control/ready/{worker}.{new_boot_id}.json",
        {
            "schema_version": 1,
            "state": "ready",
            "run_id": RUN_ID,
            "worker_name": worker,
            "boot_id": new_boot_id,
            "created_at": "2026-08-04T12:00:00Z",
            "local_ssd_count": 2,
            "local_stage_bytes": 750_000_000_000,
        },
        updated="2026-08-04T12:00:00Z",
    )

    result = run_monitor(
        config,
        client=client,
        object_store=LocalObjectStore(tmp_path / "gcs"),
        now=lambda: 100,
    )

    worker_report = result["workers"][0]
    assert worker_report["state"] == "running"
    assert worker_report["replacement_permitted"] is False
    assert worker_report["superseded_failure"]["reason"] == "newer_ready_boot"
    assert client.serial_calls == []


def test_later_assignment_progress_recovers_same_boot_after_exit_75(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    _add_failure(client, worker_index=0, exit_code=75)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    job = next(job for job in jobs if job["worker"] == "worker-0000")
    client.add_json(
        assignment_completion_uri(manifest, job),
        {
            "schema": ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA,
            "status": "complete",
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": config["manifest_file_sha256"],
            "assignment": {
                key: job[key]
                for key in (
                    "ordinal",
                    "repo",
                    "project_id",
                    "worker",
                    "assignment_sha256",
                )
            },
            "source_receipt": _source_receipt_entry(manifest, job),
            "training_ready": False,
        },
        updated="2026-08-04T12:00:00Z",
    )

    result = run_monitor(
        config,
        client=client,
        object_store=LocalObjectStore(tmp_path / "gcs"),
        now=lambda: 100,
    )

    worker_report = result["workers"][0]
    assert worker_report["state"] == "running_recovered_after_failure"
    assert worker_report["superseded_failure"]["reason"] == "later_progress"
    assert worker_report["replacement_permitted"] is False
    assert client.serial_calls == []


def test_exit_2_never_becomes_retryable_even_when_serial_contains_429(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest_value = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    client.serial = b"HTTP 429 Too Many Requests\n"
    _add_ready(client)
    _add_failure(client, worker_index=1, exit_code=2)

    result = run_monitor(
        config,
        client=client,
        object_store=LocalObjectStore(tmp_path / "gcs"),
        now=lambda: 100,
    )

    failed = result["workers"][1]
    assert result["state"] == "blocked_deterministic"
    assert failed["state"] == "deterministic_failure_manual_review"
    assert failed["recovery_evidence"] == "exit_2"
    assert failed["diagnostics"]["confirmed_http_429"] is True
    assert failed["replacement_permitted"] is False


def test_complete_run_writes_local_verified_non_training_terminal_receipt(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    _add_all_completions(client, manifest, str(config["manifest_file_sha256"]))

    result = run_monitor(
        config,
        client=client,
        object_store=LocalObjectStore(tmp_path / "gcs"),
        now=lambda: 100,
    )

    assert result["state"] == "complete"
    assert result["counts"]["assignment_receipts"] == 16
    assert result["counts"]["slot_receipts"] == 8
    assert result["counts"]["completed_workers"] == 4
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["status"] == "verified"
    assert terminal["training_ready"] is False
    assert terminal["receipt_inventory_sha256"] == result["receipt_inventory_sha256"]
