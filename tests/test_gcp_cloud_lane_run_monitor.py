from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import pytest

from scripts.distributed_data_prep._common import (
    ContractError,
    canonical_sha256,
)
from scripts.distributed_data_prep.cloud_lane_heartbeat import (
    build_worker_heartbeat,
    worker_heartbeat_uri,
)
from scripts.distributed_data_prep.cloud_lane_pool_worker import (
    POOL_COMPLETION_SCHEMA,
    POOL_FAILURE_SCHEMA,
    pool_completion_sha256,
    pool_failure_sha256,
)
from scripts.distributed_data_prep.source_worker import TransientTransportError
from scripts import gcp_cloud_lane_run_monitor as monitor


RUN_ID = "ci-case5-smoke-20260809-004"
RUN_ROOT = f"gs://fixture-bucket/runs/{RUN_ID}"
OUTPUT_PREFIX = f"{RUN_ROOT}/outputs"
MANIFEST_SHA256 = "1" * 64
MANIFEST_FILE_SHA256 = "2" * 64
CODE_REVISION = "3" * 40
ADAPTER_SHA256 = "4" * 64
PHYSICAL_WORKER = "physical-0000"
NOW = datetime(2026, 8, 9, 21, 0, tzinfo=timezone.utc)


def _config(
    tmp_path: Path,
    *,
    heartbeat_required: bool = True,
    deployed_worker_count: int = 1,
    assignment_pool_size: int = 1,
) -> dict[str, object]:
    return {
        "schema": monitor.CONFIG_SCHEMA,
        "project_id": "fixture-project",
        "gcloud_account": "fixture@example.invalid",
        "zone": "us-central1-a",
        "instance_name": "fixture-worker-0000",
        "run_id": RUN_ID,
        "run_root": RUN_ROOT,
        "output_prefix": OUTPUT_PREFIX,
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_file_sha256": MANIFEST_FILE_SHA256,
        "code_revision": CODE_REVISION,
        "physical_worker": PHYSICAL_WORKER,
        "deployed_worker_count": deployed_worker_count,
        "assignment_pool_size": assignment_pool_size,
        "heartbeat_required": heartbeat_required,
        "heartbeat_max_age_seconds": 600,
        "report_path": str(tmp_path / "report.json"),
    }


class FakeClient:
    def __init__(self) -> None:
        self.instance: Mapping[str, object] | None = {
            "name": "fixture-worker-0000",
            "status": "RUNNING",
            "labels": {"run-id": RUN_ID},
        }
        self.objects: dict[
            str, tuple[dict[str, object], Mapping[str, object], str]
        ] = {}
        self.reads: list[tuple[str, str]] = []

    def add(
        self,
        uri: str,
        value: Mapping[str, object],
        *,
        generation: str = "1",
        size_bytes: int | None = None,
    ) -> dict[str, object]:
        bucket_and_name = uri.removeprefix("gs://").split("/", 1)
        assert len(bucket_and_name) == 2
        encoded = json.dumps(value, sort_keys=True).encode("utf-8")
        metadata: dict[str, object] = {
            "name": bucket_and_name[1],
            "generation": generation,
            "size": len(encoded) if size_bytes is None else size_bytes,
        }
        self.objects[uri] = (
            metadata,
            dict(value),
            hashlib.sha256(encoded).hexdigest(),
        )
        return {
            "uri": uri,
            "generation": generation,
            "size_bytes": metadata["size"],
            "sha256": "5" * 64,
        }

    def describe_instance(
        self, *, project_id: str, zone: str, instance_name: str
    ) -> Mapping[str, object] | None:
        assert project_id == "fixture-project"
        assert zone == "us-central1-a"
        assert instance_name == "fixture-worker-0000"
        return self.instance

    def list_objects(self, prefix: str) -> Sequence[Mapping[str, object]]:
        canonical = prefix.rstrip("/") + "/"
        return [
            metadata
            for uri, (metadata, _value, _content_sha256) in sorted(
                self.objects.items()
            )
            if uri.startswith(canonical)
        ]

    def read_json(self, metadata: Mapping[str, object]) -> monitor.ExactJsonRead:
        name = str(metadata["name"])
        matches = [
            (uri, stored_metadata, value, content_sha256)
            for uri, (stored_metadata, value, content_sha256) in self.objects.items()
            if str(stored_metadata["name"]) == name
        ]
        assert len(matches) == 1
        uri, stored_metadata, value, content_sha256 = matches[0]
        assert metadata["generation"] == stored_metadata["generation"]
        self.reads.append((uri, str(metadata["generation"])))
        return monitor.ExactJsonRead(
            value=dict(value), content_sha256=content_sha256
        )


def _add_heartbeat(
    client: FakeClient,
    *,
    sequence: int = 0,
    emitted_at: str = "2026-08-09T20:59:00Z",
    generation: str = "7",
) -> dict[str, object]:
    value = build_worker_heartbeat(
        sequence=sequence,
        manifest_sha256=MANIFEST_SHA256,
        manifest_file_sha256=MANIFEST_FILE_SHA256,
        code_revision=CODE_REVISION,
        physical_worker=PHYSICAL_WORKER,
        physical_worker_count=1,
        emitted_at=emitted_at,
    )
    client.add(
        worker_heartbeat_uri(RUN_ROOT, value),
        value,
        generation=generation,
    )
    return value


def _pool_completion(publication: Mapping[str, object]) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": POOL_COMPLETION_SCHEMA,
        "status": "complete",
        "kind": "ci",
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_file_sha256": MANIFEST_FILE_SHA256,
        "code_revision": CODE_REVISION,
        "adapter_sha256": ADAPTER_SHA256,
        "physical_worker_index": 0,
        "physical_worker_count": 1,
        "logical_workers": ["ci-00000"],
        "logical_worker_completions": [
            {
                "worker": "ci-00000",
                "receipt_sha256": "6" * 64,
                "publication": dict(publication),
            }
        ],
        "totals": {
            "source_record_count": 10,
            "candidate_document_count": 9,
            "valid_tokens": 1234,
            "assignment_receipt_count": 1,
        },
        "training_ready": False,
    }
    value["receipt_sha256"] = pool_completion_sha256(value)
    return value


def _add_completion(client: FakeClient) -> dict[str, object]:
    logical_uri = (
        f"{OUTPUT_PREFIX}/worker-completions/ci/{MANIFEST_SHA256}/"
        "ci-00000/fixture.complete.json"
    )
    publication = client.add(logical_uri, {"status": "complete"}, generation="11")
    value = _pool_completion(publication)
    client.add(
        f"{RUN_ROOT}/control/cloud-lane-completed/{MANIFEST_SHA256}/"
        f"{PHYSICAL_WORKER}.complete.json",
        value,
        generation="12",
    )
    client.add(
        f"{RUN_ROOT}/control/cloud-lane-runner-completions/{MANIFEST_SHA256}/"
        f"{PHYSICAL_WORKER}/{value['receipt_sha256']}.complete.json",
        value,
        generation="13",
    )
    return value


def _pool_failure(*, confirmed_http_429: bool) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": POOL_FAILURE_SCHEMA,
        "status": "failed",
        "kind": "ci",
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_file_sha256": MANIFEST_FILE_SHA256,
        "physical_worker_index": 0,
        "physical_worker_count": 1,
        "diagnostics": [
            {
                "worker": "ci-00000",
                "error_type": "RuntimeError",
                "diagnostic_sha256": "7" * 64,
                "confirmed_http_429": confirmed_http_429,
            }
        ],
        "retry_exit_code": 75 if confirmed_http_429 else 2,
        "training_ready": False,
    }
    value["receipt_sha256"] = pool_failure_sha256(value)
    return value


def _runner_failure(*, confirmed_http_429: bool) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": monitor.RUNNER_FAILURE_SCHEMA,
        "status": "failed",
        "stage": "worker",
        "exit_code": 75 if confirmed_http_429 else 2,
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_file_sha256": MANIFEST_FILE_SHA256,
        "physical_worker": PHYSICAL_WORKER,
        "diagnostic_sha256": "8" * 64,
        "confirmed_http_429": confirmed_http_429,
        "training_ready": False,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _add_failure(
    client: FakeClient, *, pool_429: bool, runner_429: bool
) -> tuple[dict[str, object], dict[str, object]]:
    pool = _pool_failure(confirmed_http_429=pool_429)
    runner = _runner_failure(confirmed_http_429=runner_429)
    client.add(
        f"{RUN_ROOT}/control/cloud-lane-failures/{MANIFEST_SHA256}/"
        f"{PHYSICAL_WORKER}/{pool['receipt_sha256']}.failure.json",
        pool,
        generation="21",
    )
    client.add(
        f"{RUN_ROOT}/control/cloud-lane-runner-failures/{MANIFEST_SHA256}/"
        f"{PHYSICAL_WORKER}/{runner['receipt_sha256']}.failure.json",
        runner,
        generation="22",
    )
    return pool, runner


def test_fresh_heartbeat_is_exact_generation_read_and_never_training_ready(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    heartbeat = _add_heartbeat(client)

    report = monitor.monitor_cloud_lane_run(
        _config(tmp_path), client=client, now=NOW
    )

    assert report["state"] == "running"
    assert report["heartbeat"] == {
        "count": 1,
        "latest_sequence": 0,
        "latest_emitted_at": "2026-08-09T20:59:00Z",
        "latest_age_seconds": 60,
        "latest_generation": "7",
        "latest_receipt_sha256": heartbeat["receipt_sha256"],
        "fresh": True,
    }
    assert client.reads == [(worker_heartbeat_uri(RUN_ROOT, heartbeat), "7")]
    assert report["retry_eligible"] is False
    assert report["cleanup_authorized"] is False
    assert report["training_ready"] is False


def test_config_binds_scoped_account_and_separates_deployed_from_pool_size(
    tmp_path: Path,
) -> None:
    path = tmp_path / "monitor.json"
    value = _config(
        tmp_path,
        deployed_worker_count=1,
        assignment_pool_size=32,
    )
    path.write_text(json.dumps(value), encoding="utf-8")

    loaded = monitor.load_monitor_config(path)

    assert loaded["gcloud_account"] == "fixture@example.invalid"
    assert loaded["deployed_worker_count"] == 1
    assert loaded["assignment_pool_size"] == 32

    value["deployed_worker_count"] = 2
    value["assignment_pool_size"] = 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ContractError, match="smaller than deployed_worker_count"):
        monitor.load_monitor_config(path)


@pytest.mark.parametrize(
    ("heartbeat_required", "expected_state"),
    [
        (True, "running_missing_required_heartbeat"),
        (False, "running_without_heartbeat"),
    ],
)
def test_missing_heartbeat_supports_strict_and_legacy_runs(
    tmp_path: Path, heartbeat_required: bool, expected_state: str
) -> None:
    report = monitor.monitor_cloud_lane_run(
        _config(tmp_path, heartbeat_required=heartbeat_required),
        client=FakeClient(),
        now=NOW,
    )

    assert report["state"] == expected_state
    assert report["cleanup_authorized"] is False


def test_stale_or_implausibly_future_heartbeat_fails_closed(tmp_path: Path) -> None:
    stale = FakeClient()
    _add_heartbeat(stale, emitted_at="2026-08-09T20:00:00Z")
    report = monitor.monitor_cloud_lane_run(
        _config(tmp_path), client=stale, now=NOW
    )
    assert report["state"] == "running_stale_heartbeat"

    future = FakeClient()
    _add_heartbeat(future, emitted_at="2026-08-09T21:06:00Z")
    with pytest.raises(ContractError, match="far in the future"):
        monitor.monitor_cloud_lane_run(_config(tmp_path), client=future, now=NOW)


def test_completion_requires_matching_pool_runner_and_physical_outputs(
    tmp_path: Path,
) -> None:
    complete = FakeClient()
    value = _add_completion(complete)
    report = monitor.monitor_cloud_lane_run(
        _config(tmp_path), client=complete, now=NOW
    )
    assert report["state"] == "completed_verified"
    completion = report["completion"]
    counts = report["counts"]
    assert isinstance(completion, Mapping)
    assert isinstance(counts, Mapping)
    assert completion["receipt_sha256"] == value["receipt_sha256"]
    assert counts["output_objects"] == 1
    assert report["cleanup_authorized"] is True
    assert report["training_ready"] is False
    assert set(complete.reads) == {
        (
            f"{RUN_ROOT}/control/cloud-lane-completed/{MANIFEST_SHA256}/"
            f"{PHYSICAL_WORKER}.complete.json",
            "12",
        ),
        (
            f"{RUN_ROOT}/control/cloud-lane-runner-completions/{MANIFEST_SHA256}/"
            f"{PHYSICAL_WORKER}/{value['receipt_sha256']}.complete.json",
            "13",
        ),
    }

    partial = FakeClient()
    logical_uri = f"{OUTPUT_PREFIX}/worker-completions/ci/fixture.complete.json"
    publication = partial.add(logical_uri, {"status": "complete"}, generation="31")
    partial_value = _pool_completion(publication)
    partial.add(
        f"{RUN_ROOT}/control/cloud-lane-completed/{MANIFEST_SHA256}/"
        f"{PHYSICAL_WORKER}.complete.json",
        partial_value,
    )
    with pytest.raises(ContractError, match="partially published"):
        monitor.monitor_cloud_lane_run(_config(tmp_path), client=partial, now=NOW)

    missing_output = FakeClient()
    missing_value = _pool_completion(
        {
            "uri": f"{OUTPUT_PREFIX}/missing.complete.json",
            "generation": "41",
            "size_bytes": 10,
            "sha256": "9" * 64,
        }
    )
    missing_output.add(
        f"{RUN_ROOT}/control/cloud-lane-completed/{MANIFEST_SHA256}/"
        f"{PHYSICAL_WORKER}.complete.json",
        missing_value,
    )
    missing_output.add(
        f"{RUN_ROOT}/control/cloud-lane-runner-completions/{MANIFEST_SHA256}/"
        f"{PHYSICAL_WORKER}/{missing_value['receipt_sha256']}.complete.json",
        missing_value,
    )
    with pytest.raises(ContractError, match="absent from outputs"):
        monitor.monitor_cloud_lane_run(
            _config(tmp_path), client=missing_output, now=NOW
        )


def test_deterministic_confirmed_429_and_mixed_failures_are_distinct(
    tmp_path: Path,
) -> None:
    deterministic = FakeClient()
    _add_failure(deterministic, pool_429=False, runner_429=False)
    deterministic_report = monitor.monitor_cloud_lane_run(
        _config(tmp_path), client=deterministic, now=NOW
    )
    assert deterministic_report["state"] == "failed_deterministic"
    assert deterministic_report["retry_eligible"] is False
    assert deterministic_report["cleanup_authorized"] is False

    transient = FakeClient()
    _add_failure(transient, pool_429=True, runner_429=True)
    transient_report = monitor.monitor_cloud_lane_run(
        _config(tmp_path), client=transient, now=NOW
    )
    assert transient_report["state"] == "failed_confirmed_429"
    assert transient_report["retry_eligible"] is True
    assert transient_report["cleanup_authorized"] is False

    mixed = FakeClient()
    _add_failure(mixed, pool_429=False, runner_429=True)
    mixed_report = monitor.monitor_cloud_lane_run(
        _config(tmp_path), client=mixed, now=NOW
    )
    assert mixed_report["state"] == "failed_deterministic"
    assert mixed_report["retry_eligible"] is False


def test_completion_failure_contradiction_and_unknown_control_are_rejected(
    tmp_path: Path,
) -> None:
    contradictory = FakeClient()
    _add_completion(contradictory)
    _add_failure(contradictory, pool_429=False, runner_429=False)
    with pytest.raises(ContractError, match="both completion and failure"):
        monitor.monitor_cloud_lane_run(
            _config(tmp_path), client=contradictory, now=NOW
        )

    unknown = FakeClient()
    unknown.add(f"{RUN_ROOT}/control/unrecognized.json", {"unexpected": True})
    with pytest.raises(ContractError, match="unknown control object"):
        monitor.monitor_cloud_lane_run(_config(tmp_path), client=unknown, now=NOW)


def test_failure_receipt_path_digest_is_part_of_the_immutable_binding(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    pool, _runner = _add_failure(client, pool_429=False, runner_429=False)
    original = (
        f"{RUN_ROOT}/control/cloud-lane-failures/{MANIFEST_SHA256}/"
        f"{PHYSICAL_WORKER}/{pool['receipt_sha256']}.failure.json"
    )
    metadata, value, _content_sha256 = client.objects.pop(original)
    drifted = original.replace(str(pool["receipt_sha256"]), "a" * 64)
    size = metadata["size"]
    assert isinstance(size, int)
    client.add(
        drifted,
        value,
        generation=str(metadata["generation"]),
        size_bytes=size,
    )

    with pytest.raises(ContractError, match="path/content digest drifted"):
        monitor.monitor_cloud_lane_run(_config(tmp_path), client=client, now=NOW)


def test_runner_failure_accepts_only_verified_exact_file_or_receipt_digest(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    _pool, runner = _add_failure(client, pool_429=True, runner_429=True)
    original = (
        f"{RUN_ROOT}/control/cloud-lane-runner-failures/{MANIFEST_SHA256}/"
        f"{PHYSICAL_WORKER}/{runner['receipt_sha256']}.failure.json"
    )
    metadata, value, content_sha256 = client.objects.pop(original)
    raw_digest_uri = original.replace(str(runner["receipt_sha256"]), content_sha256)
    size = metadata["size"]
    assert isinstance(size, int)
    client.add(
        raw_digest_uri,
        value,
        generation=str(metadata["generation"]),
        size_bytes=size,
    )

    report = monitor.monitor_cloud_lane_run(_config(tmp_path), client=client, now=NOW)

    assert report["state"] == "failed_confirmed_429"
    failures = report["failures"]
    assert isinstance(failures, Mapping)
    runner_failures = failures["runner"]
    assert isinstance(runner_failures, list)
    assert runner_failures[0]["path_digest_kind"] == "content_sha256"
    assert report["retry_eligible"] is True


def test_multiworker_inventory_ignores_only_in_range_sibling_control(
    tmp_path: Path,
) -> None:
    sibling = FakeClient()
    heartbeat = build_worker_heartbeat(
        sequence=0,
        manifest_sha256=MANIFEST_SHA256,
        manifest_file_sha256=MANIFEST_FILE_SHA256,
        code_revision=CODE_REVISION,
        physical_worker="physical-0001",
        physical_worker_count=2,
        emitted_at="2026-08-09T20:59:00Z",
    )
    sibling.add(worker_heartbeat_uri(RUN_ROOT, heartbeat), heartbeat)
    report = monitor.monitor_cloud_lane_run(
        _config(
            tmp_path,
            heartbeat_required=False,
            deployed_worker_count=2,
            assignment_pool_size=2,
        ),
        client=sibling,
        now=NOW,
    )
    assert report["state"] == "running_without_heartbeat"
    counts = report["counts"]
    assert isinstance(counts, Mapping)
    assert counts["sibling_control_objects"] == 1

    outside = FakeClient()
    outside_uri = worker_heartbeat_uri(RUN_ROOT, heartbeat).replace(
        "physical-0001", "physical-0002"
    )
    outside.add(outside_uri, heartbeat)
    with pytest.raises(ContractError, match="outside physical_worker_count"):
        monitor.monitor_cloud_lane_run(
            _config(
                tmp_path,
                heartbeat_required=False,
                deployed_worker_count=2,
                assignment_pool_size=2,
            ),
            client=outside,
            now=NOW,
        )


def test_nonrunning_instance_without_terminal_receipt_is_not_healthy(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    assert client.instance is not None
    client.instance = dict(client.instance, status="TERMINATED")
    report = monitor.monitor_cloud_lane_run(
        _config(tmp_path, heartbeat_required=False), client=client, now=NOW
    )
    assert report["state"] == "instance_not_running_without_terminal_receipt"
    assert report["cleanup_authorized"] is False


def test_gcloud_classifier_never_treats_bare_or_mixed_status_as_pure_429() -> None:
    def runner(stderr: str):
        return lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=stderr
        )

    pure = monitor.GcloudCloudLaneMonitorClient(
        runner=runner("HTTPError 429: Too Many Requests")
    )
    with pytest.raises(TransientTransportError):
        pure.list_objects(RUN_ROOT)

    for detail in (
        "object path /429/retry failed",
        "HTTP 429 followed by HTTP 503",
        "HTTP 429 after connection reset",
        "HTTP 401 and HTTP 429",
    ):
        client = monitor.GcloudCloudLaneMonitorClient(runner=runner(detail))
        with pytest.raises(RuntimeError):
            client.list_objects(RUN_ROOT)


def test_gcloud_client_applies_the_scoped_account_to_every_command() -> None:
    observed: list[list[str]] = []

    def runner(argv, **_kwargs):
        observed.append(list(argv))
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="[]",
            stderr="",
        )

    client = monitor.GcloudCloudLaneMonitorClient(
        account="fixture@example.invalid",
        runner=runner,
    )
    assert client.list_objects(RUN_ROOT) == []
    assert observed
    assert all("--account=fixture@example.invalid" in argv for argv in observed)


def test_gcloud_exact_json_read_rejects_size_drift_and_duplicate_keys() -> None:
    raw = '{"status":"failed"}\n'

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=raw,
            stderr="",
        )

    client = monitor.GcloudCloudLaneMonitorClient(runner=runner)
    metadata = {
        "uri": f"{RUN_ROOT}/receipt.json",
        "generation": "123",
        "size_bytes": len(raw.encode("utf-8")),
    }
    observed = client.read_json(metadata)
    assert observed.value == {"status": "failed"}
    assert observed.content_sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()

    with pytest.raises(ContractError, match="size drifted"):
        client.read_json(dict(metadata, size_bytes=1))

    duplicate_raw = '{"status":"failed","status":"complete"}\n'

    def duplicate_runner(argv, **_kwargs):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=duplicate_raw,
            stderr="",
        )

    duplicate = monitor.GcloudCloudLaneMonitorClient(runner=duplicate_runner)
    with pytest.raises(ContractError, match="duplicate JSON key"):
        duplicate.read_json(
            dict(metadata, size_bytes=len(duplicate_raw.encode("utf-8")))
        )


@pytest.mark.parametrize(
    ("failure_429", "expected_exit"),
    [(True, 75), (False, 2)],
)
def test_cli_exit_code_exposes_only_receipt_proven_429_as_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_429: bool,
    expected_exit: int,
) -> None:
    client = FakeClient()
    _add_failure(client, pool_429=failure_429, runner_429=failure_429)
    config = _config(tmp_path)
    monkeypatch.setattr(monitor, "load_monitor_config", lambda _path: config)
    monkeypatch.setattr(
        monitor, "GcloudCloudLaneMonitorClient", lambda **_kwargs: client
    )

    assert monitor._main(["--config", str(tmp_path / "config.json")]) == expected_exit
    report = json.loads(Path(str(config["report_path"])).read_text(encoding="utf-8"))
    assert report["retry_eligible"] is failure_429
    assert report["training_ready"] is False


def test_cli_monitor_transport_429_is_75_but_other_defects_are_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(monitor, "load_monitor_config", lambda _path: config)

    class BrokenClient(FakeClient):
        error: BaseException = RuntimeError("HTTP 429 is not classified evidence")

        def describe_instance(self, **_kwargs):
            raise self.error

    broken = BrokenClient()
    monkeypatch.setattr(
        monitor, "GcloudCloudLaneMonitorClient", lambda **_kwargs: broken
    )
    assert monitor._main(["--config", str(tmp_path / "config.json")]) == 2

    broken.error = TransientTransportError("confirmed HTTP 429")
    assert monitor._main(["--config", str(tmp_path / "config.json")]) == 75
