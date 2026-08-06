from __future__ import annotations

import fnmatch
import hashlib
import http.client
import json
import subprocess
import urllib.error
from pathlib import Path
from typing import Mapping

import pytest

from scripts.distributed_data_prep._common import (
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from scripts.distributed_data_prep.source_manifest import (
    build_source_manifest,
    repositories_for_worker,
)
from scripts.distributed_data_prep.source_slot_scheduler import (
    SLOT_COMPLETION_RECEIPT_SCHEMA,
    slot_specs,
)
from scripts.distributed_data_prep.source_work_queue import (
    ASSIGNMENT_HEARTBEAT_SCHEMA,
    assignment_heartbeat_uri,
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
        self.batch_calls: list[list[str]] = []

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

    def read_json_many(
        self, metadata_rows: list[Mapping[str, object]]
    ) -> list[tuple[bytes, dict[str, object]]]:
        self.batch_calls.append([str(row["uri"]) for row in metadata_rows])
        return [self.read_json(metadata) for metadata in metadata_rows]

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
        return subprocess.CompletedProcess(argv, 0, b"test-access-token\n", b"")

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    client = GcloudRunClient(
        "gcloud", runner=runner, urlopen=lambda *_args, **_kwargs: Response()
    )
    assert client.list_objects(f"{RUN_ROOT}/control/failed/*.json") == []


def test_gcloud_object_listing_does_not_hide_other_exit_one_errors() -> None:
    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, b"test-access-token\n", b"")

    def urlopen(request: object, *, timeout: int) -> object:
        raise urllib.error.HTTPError(
            str(request.full_url), 403, "permission denied", {}, None
        )

    client = GcloudRunClient("gcloud", runner=runner, urlopen=urlopen)
    with pytest.raises(MonitorError, match="HTTP 403"):
        client.list_objects(f"{RUN_ROOT}/control/failed/*.json")


def test_gcloud_object_listing_retries_transient_http_statuses() -> None:
    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, b"test-access-token\n", b"")

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    calls = 0
    sleeps: list[float] = []

    def urlopen(request: object, *, timeout: int) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                str(request.full_url), 429, "too many requests", {}, None
            )
        return Response()

    client = GcloudRunClient(
        "gcloud",
        runner=runner,
        urlopen=urlopen,
        sleeper=sleeps.append,
    )
    assert client.list_objects(f"{RUN_ROOT}/control/failed/*.json") == []
    assert calls == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize(
    "transport_error",
    [
        TimeoutError("timed out"),
        ConnectionResetError("connection reset"),
        http.client.IncompleteRead(b"partial"),
        http.client.BadStatusLine("bad status"),
    ],
    ids=["timeout", "connection-reset", "incomplete-read", "bad-status"],
)
def test_gcloud_object_listing_retries_body_transport_errors(
    transport_error: BaseException,
) -> None:
    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, b"test-access-token\n", b"")

    class Response:
        status = 200

        def __init__(self, error: BaseException | None = None) -> None:
            self.error = error

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            if self.error is not None:
                raise self.error
            return b"{}"

    calls = 0

    def urlopen(_request: object, *, timeout: int) -> Response:
        nonlocal calls
        calls += 1
        return Response(transport_error if calls == 1 else None)

    client = GcloudRunClient("gcloud", runner=runner, urlopen=urlopen)
    assert client.list_objects(f"{RUN_ROOT}/control/failed/*.json") == []
    assert calls == 2


@pytest.mark.parametrize(
    "transport_error",
    [
        TimeoutError("timed out"),
        ConnectionResetError("connection reset"),
        http.client.IncompleteRead(b"partial"),
        http.client.BadStatusLine("bad status"),
    ],
    ids=["timeout", "connection-reset", "incomplete-read", "bad-status"],
)
def test_gcloud_batch_json_read_retries_body_transport_errors(
    transport_error: BaseException,
) -> None:
    raw = b'{"training_ready":false}\n'
    rows = [
        {
            "uri": f"{RUN_ROOT}/receipt.json",
            "generation": "303",
            "size_bytes": len(raw),
        }
    ]

    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, b"test-access-token\n", b"")

    class Response:
        status = 200

        def __init__(self, error: BaseException | None = None) -> None:
            self.error = error

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            if self.error is not None:
                raise self.error
            return raw

    calls = 0

    def urlopen(_request: object, *, timeout: int) -> Response:
        nonlocal calls
        calls += 1
        return Response(transport_error if calls == 1 else None)

    result = GcloudRunClient("gcloud", runner=runner, urlopen=urlopen).read_json_many(
        rows
    )
    assert result == [(raw, {"training_ready": False})]
    assert calls == 2


def test_gcloud_batch_json_read_preserves_generation_and_byte_boundaries() -> None:
    first = b'{"assignment_sha256":"' + b"a" * 64 + b'"}\n'
    second = b'{"heartbeat_index":17,"training_ready":false}\n'
    rows = [
        {
            "uri": f"{RUN_ROOT}/first.json",
            "generation": "101",
            "size_bytes": len(first),
        },
        {
            "uri": f"{RUN_ROOT}/second.json",
            "generation": "202",
            "size_bytes": len(second),
        },
    ]

    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        assert argv == ["gcloud", "auth", "print-access-token"]
        return subprocess.CompletedProcess(argv, 0, b"test-access-token\n", b"")

    class Response:
        status = 200

        def __init__(self, raw: bytes) -> None:
            self.raw = raw

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return self.raw

    requested: list[str] = []

    def urlopen(request: object, *, timeout: int) -> Response:
        assert timeout == 180
        assert request.get_header("Authorization") == "Bearer test-access-token"
        url = str(request.full_url)
        requested.append(url)
        return Response(first if "first.json" in url else second)

    result = GcloudRunClient("gcloud", runner=runner, urlopen=urlopen).read_json_many(
        rows
    )

    assert result[0] == (first, {"assignment_sha256": "a" * 64})
    assert result[1] == (second, {"heartbeat_index": 17, "training_ready": False})
    assert len(requested) == 2
    assert any("first.json?alt=media&generation=101" in url for url in requested)
    assert any("second.json?alt=media&generation=202" in url for url in requested)


def test_gcloud_batch_json_read_rejects_boundary_drift() -> None:
    raw = b'{"training_ready":false}\n'
    rows = [
        {
            "uri": f"{RUN_ROOT}/receipt.json",
            "generation": "303",
            "size_bytes": len(raw),
        }
    ]

    def runner(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, b"test-access-token\n", b"")

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return raw + b"x"

    def urlopen(_request: object, *, timeout: int) -> Response:
        assert timeout == 180
        return Response()

    with pytest.raises(MonitorError, match="generation size drifted"):
        GcloudRunClient("gcloud", runner=runner, urlopen=urlopen).read_json_many(rows)


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
    created_unix_s: int = 10,
    lease_seconds: int = 900,
    heartbeat_seconds: int = 120,
) -> tuple[dict[str, object], str]:
    logical_worker = f"worker-{physical_worker_index * 2 + slot_index:04d}"
    assignment_sha256 = str(job["assignment_sha256"])
    uri = (
        f"{RUN_ROOT}/source-assignment-claims/{manifest['manifest_sha256']}/"
        f"{assignment_sha256}/{attempt:04d}.claim.json"
    )
    claim: dict[str, object] = {
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
        "created_unix_s": created_unix_s,
        "expires_unix_s": created_unix_s + lease_seconds,
        "lease_seconds": lease_seconds,
        "heartbeat_seconds": heartbeat_seconds,
        "training_ready": False,
    }
    client.add_json(
        uri,
        claim,
    )
    _metadata, raw, _value = client.objects[uri]
    claim_sha256 = canonical_sha256(claim)
    assert claim_sha256 != hashlib.sha256(raw).hexdigest()
    return claim, claim_sha256


def _add_heartbeat(
    client: FakeRunClient,
    *,
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    claim: Mapping[str, object],
    claim_sha256: str,
    heartbeat_index: int,
    updated: str = "2026-08-04T12:00:00Z",
) -> str:
    scheduled = int(claim["created_unix_s"]) + heartbeat_index * int(
        claim["heartbeat_seconds"]
    )
    uri = assignment_heartbeat_uri(
        manifest,
        job,
        int(claim["attempt"]),
        claim_sha256,
        heartbeat_index,
    )
    client.add_json(
        uri,
        {
            "schema": ASSIGNMENT_HEARTBEAT_SCHEMA,
            "status": "active",
            "manifest_sha256": manifest["manifest_sha256"],
            "assignment_sha256": job["assignment_sha256"],
            "attempt": claim["attempt"],
            "claim_sha256": claim_sha256,
            "executor": claim["executor"],
            "scheduler_instance": claim["scheduler_instance"],
            "heartbeat_index": heartbeat_index,
            "scheduled_unix_s": scheduled,
            "lease_through_unix_s": scheduled + int(claim["lease_seconds"]),
            "training_ready": False,
        },
        updated=updated,
    )
    return uri


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


def _add_completion(
    client: FakeRunClient,
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    job: Mapping[str, object],
    updated: str = "2026-08-04T12:00:00Z",
) -> None:
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
        updated=updated,
    )


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


def test_historical_claim_on_another_worker_does_not_hide_a_stale_worker(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=jobs[0],
        physical_worker_index=1,
    )
    _add_completion(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=jobs[0],
    )
    store = LocalObjectStore(tmp_path / "gcs")

    run_monitor(config, client=client, object_store=store, now=lambda: 100)
    result = run_monitor(config, client=client, object_store=store, now=lambda: 2000)

    assert result["scheduler_mode"] == "dynamic_claim_queue"
    assert result["workers"][0]["claim_receipts"] == 0
    assert result["workers"][0]["state"] == "idle_suspected_manual_review"


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
    completion_uri = assignment_completion_uri(manifest, job)
    _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
        physical_worker_index=1,
    )
    client.add_json(
        completion_uri,
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
    assert [completion_uri] in client.batch_calls


def test_fresh_exact_assignment_heartbeat_keeps_executor_worker_live(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    config["stale_after_seconds"] = 60
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    job = jobs[0]
    claim, claim_sha256 = _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
        physical_worker_index=1,
    )
    _add_heartbeat(
        client,
        manifest=manifest,
        job=job,
        claim=claim,
        claim_sha256=claim_sha256,
        heartbeat_index=16,
    )
    store = LocalObjectStore(tmp_path / "gcs")

    first = run_monitor(config, client=client, object_store=store, now=lambda: 100)
    second = run_monitor(config, client=client, object_store=store, now=lambda: 2000)

    assert first["counts"]["assignment_heartbeat_receipts"] == 1
    assert first["counts"]["fresh_heartbeat_assignments"] == 0
    worker = second["workers"][1]
    assert worker["state"] == "running"

    assert worker["last_progress_at_unix"] == 1930
    assert worker["fresh_heartbeat_assignments"] == 1
    assert worker["fresh_assignment_heartbeats"] == [
        {
            "repo": job["repo"],
            "assignment_sha256": job["assignment_sha256"],
            "attempt": 0,
            "logical_worker": "worker-0002",
            "heartbeat_index": 16,
            "scheduled_unix_s": 1930,
            "lease_through_unix_s": 2830,
        }
    ]
    assert second["training_ready"] is False


def test_current_claim_lease_prevents_a_false_idle_report(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    config["stale_after_seconds"] = 60
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=jobs[0],
        physical_worker_index=1,
        created_unix_s=100,
        lease_seconds=900,
    )
    store = LocalObjectStore(tmp_path / "gcs")

    run_monitor(config, client=client, object_store=store, now=lambda: 100)
    leased = run_monitor(config, client=client, object_store=store, now=lambda: 500)
    expired = run_monitor(config, client=client, object_store=store, now=lambda: 1100)

    assert leased["workers"][1]["state"] == "running"
    assert leased["workers"][1]["current_claimed_assignments"] == 1
    assert expired["workers"][1]["state"] == "idle_suspected_manual_review"


def test_superseded_claim_heartbeat_does_not_keep_executor_worker_live(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    job = jobs[0]
    claim, claim_sha256 = _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
        physical_worker_index=1,
    )
    _add_heartbeat(
        client,
        manifest=manifest,
        job=job,
        claim=claim,
        claim_sha256=claim_sha256,
        heartbeat_index=16,
    )
    _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
        physical_worker_index=1,
        attempt=1,
        created_unix_s=50,
    )
    store = LocalObjectStore(tmp_path / "gcs")

    run_monitor(config, client=client, object_store=store, now=lambda: 100)
    result = run_monitor(config, client=client, object_store=store, now=lambda: 2000)

    worker = result["workers"][1]
    assert result["counts"]["assignment_heartbeat_receipts"] == 1
    assert result["counts"]["fresh_heartbeat_assignments"] == 0
    assert worker["current_claim_heartbeat_receipts"] == 0
    assert worker["state"] == "idle_suspected_manual_review"


def test_completed_assignment_does_not_make_executor_worker_look_idle(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    job = jobs[0]
    claim, claim_sha256 = _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
        physical_worker_index=1,
    )
    _add_heartbeat(
        client,
        manifest=manifest,
        job=job,
        claim=claim,
        claim_sha256=claim_sha256,
        heartbeat_index=16,
    )
    _add_completion(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
    )
    store = LocalObjectStore(tmp_path / "gcs")

    run_monitor(config, client=client, object_store=store, now=lambda: 100)
    result = run_monitor(config, client=client, object_store=store, now=lambda: 1000)

    worker = result["workers"][1]
    assert result["counts"]["fresh_heartbeat_assignments"] == 0
    assert worker["current_claim_heartbeat_receipts"] == 0
    assert worker["current_claimed_assignments"] == 0
    assert worker["state"] == "running"

    stale = run_monitor(config, client=client, object_store=store, now=lambda: 2000)
    assert stale["workers"][1]["state"] == "idle_suspected_manual_review"


def test_assignment_heartbeat_uri_is_bound_to_its_exact_index(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    job = jobs[0]
    claim, claim_sha256 = _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
        physical_worker_index=1,
    )
    uri = _add_heartbeat(
        client,
        manifest=manifest,
        job=job,
        claim=claim,
        claim_sha256=claim_sha256,
        heartbeat_index=16,
    )
    metadata, raw, value = client.objects.pop(uri)
    wrong_uri = uri.replace("00000016.heartbeat.json", "00000015.heartbeat.json")
    metadata["uri"] = wrong_uri
    client.objects[wrong_uri] = (metadata, raw, value)

    with pytest.raises(MonitorError, match="heartbeat URI binding drifted"):
        run_monitor(
            config,
            client=client,
            object_store=LocalObjectStore(tmp_path / "gcs"),
            now=lambda: 2000,
        )


def test_cached_assignment_heartbeat_is_rebound_to_the_exact_claim(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    job = jobs[0]
    claim, claim_sha256 = _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
        physical_worker_index=1,
    )
    uri = _add_heartbeat(
        client,
        manifest=manifest,
        job=job,
        claim=claim,
        claim_sha256=claim_sha256,
        heartbeat_index=16,
    )
    store = LocalObjectStore(tmp_path / "gcs")
    run_monitor(config, client=client, object_store=store, now=lambda: 2000)
    state_path = Path(str(config["state_path"]))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["validated_receipts"][uri]["summary"]["physical_worker_index"] = 0
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = run_monitor(config, client=client, object_store=store, now=lambda: 2100)

    assert result["workers"][0]["fresh_heartbeat_assignments"] == 0
    assert result["workers"][1]["fresh_heartbeat_assignments"] == 1
    repaired = json.loads(state_path.read_text(encoding="utf-8"))
    assert repaired["validated_receipts"][uri]["summary"]["physical_worker_index"] == 1


def test_assignment_heartbeat_generation_is_immutable(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    job = jobs[0]
    claim, claim_sha256 = _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
        physical_worker_index=1,
    )
    uri = _add_heartbeat(
        client,
        manifest=manifest,
        job=job,
        claim=claim,
        claim_sha256=claim_sha256,
        heartbeat_index=16,
    )
    store = LocalObjectStore(tmp_path / "gcs")
    run_monitor(config, client=client, object_store=store, now=lambda: 2000)
    _metadata, _raw, value = client.objects[uri]
    value["manifest_sha256"] = "f" * 64
    client.add_json(uri, value, generation="999")

    with pytest.raises(MonitorError, match="heartbeat receipt generation drifted"):
        run_monitor(config, client=client, object_store=store, now=lambda: 2100)


def test_validated_assignment_completion_cannot_disappear(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    job = jobs[0]
    uri = assignment_completion_uri(manifest, job)
    _add_completion(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
    )
    store = LocalObjectStore(tmp_path / "gcs")
    run_monitor(config, client=client, object_store=store, now=lambda: 100)
    client.objects.pop(uri)

    with pytest.raises(MonitorError, match="assignment receipt disappeared"):
        run_monitor(config, client=client, object_store=store, now=lambda: 200)


def test_fresh_heartbeat_cannot_override_a_later_deterministic_failure(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    job = jobs[0]
    claim, claim_sha256 = _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=job,
        physical_worker_index=1,
    )
    _add_heartbeat(
        client,
        manifest=manifest,
        job=job,
        claim=claim,
        claim_sha256=claim_sha256,
        heartbeat_index=16,
        updated="2026-08-04T11:20:00Z",
    )
    _add_failure(client, worker_index=1, exit_code=2)

    result = run_monitor(
        config,
        client=client,
        object_store=LocalObjectStore(tmp_path / "gcs"),
        now=lambda: 2000,
    )

    worker = result["workers"][1]
    assert worker["fresh_heartbeat_assignments"] == 1
    assert worker["state"] == "deterministic_failure_manual_review"
    assert worker["replacement_permitted"] is False
    assert result["state"] == "blocked_deterministic"


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


def test_cached_diagnostics_are_reverified_before_retry_is_permitted(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest_value = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    _add_failure(client, worker_index=0, exit_code=75)
    store = LocalObjectStore(tmp_path / "gcs")
    first = run_monitor(config, client=client, object_store=store, now=lambda: 100)

    second = run_monitor(
        config,
        client=client,
        object_store=FailingObjectStore(),
        now=lambda: 200,
    )

    assert first["workers"][0]["replacement_permitted"] is True
    assert second["workers"][0]["state"] == "transient_failure_recovery_blocked"
    assert second["workers"][0]["replacement_permitted"] is False
    assert "diagnostics upload unavailable" in second["workers"][0]["diagnostics_error"]
    assert client.serial_calls == [PHYSICAL_WORKERS[0]]


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


def test_stolen_completion_cannot_recover_a_different_worker_after_exit_75(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    _add_ready(client)
    jobs = manifest["repositories"]
    assert isinstance(jobs, list)
    home_worker_job = jobs[0]
    assert home_worker_job["worker"] == "worker-0000"
    _add_failure(client, worker_index=0, exit_code=75)
    _add_claim(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=home_worker_job,
        physical_worker_index=1,
    )
    _add_completion(
        client,
        manifest=manifest,
        manifest_file_sha256=str(config["manifest_file_sha256"]),
        job=home_worker_job,
    )

    result = run_monitor(
        config,
        client=client,
        object_store=LocalObjectStore(tmp_path / "gcs"),
        now=lambda: 100,
    )

    failed = result["workers"][0]
    assert result["scheduler_mode"] == "dynamic_claim_queue"
    assert failed["assignment_receipts"] == 1
    assert failed["claim_receipts"] == 0
    assert failed["state"] == "transient_failure_diagnostics_preserved"
    assert failed["recovery_evidence"] == "exit_75"
    assert failed["replacement_permitted"] is True


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


def test_exit_one_with_historical_serial_429_stays_manual_review(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest_value = _manifest(tmp_path)
    config = _config(tmp_path, manifest_path)
    client = FakeRunClient()
    client.serial = b"old HTTP 429 Too Many Requests\ncurrent contract failure\n"
    _add_ready(client)
    _add_failure(client, worker_index=1, exit_code=1)

    result = run_monitor(
        config,
        client=client,
        object_store=LocalObjectStore(tmp_path / "gcs"),
        now=lambda: 100,
    )

    failed = result["workers"][1]
    assert failed["diagnostics"]["confirmed_http_429"] is True
    assert failed["state"] == "unclassified_failure_manual_review"
    assert failed["replacement_permitted"] is False
    assert result["state"] == "manual_review"


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
