#!/usr/bin/env python3
"""Fail-closed read-only monitor for one GCP cloud-lane worker run.

The monitor never restarts or destroys a VM.  It binds one instance, control
prefix, output prefix, manifest, and physical-worker identity; reads immutable
receipts at their exact GCS generations; and writes one local atomic report.
Only an unmixed, explicitly observed HTTP 429 maps to exit 75.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Mapping, Protocol, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    MAX_METADATA_BYTES,
    ContractError,
    atomic_write_json,
    canonical_sha256,
    load_json_object,
    require_exact_fields,
    require_git_object,
    require_int,
    require_nonempty,
    require_sha256,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.cloud_lane_heartbeat import (  # noqa: E402
    validate_worker_heartbeat,
)
from scripts.distributed_data_prep.cloud_lane_pool_worker import (  # noqa: E402
    POOL_COMPLETION_SCHEMA,
    POOL_FAILURE_SCHEMA,
    pool_completion_sha256,
    pool_failure_sha256,
)
from scripts.distributed_data_prep.source_worker import (  # noqa: E402
    TransientTransportError,
)


CONFIG_SCHEMA = "cppmega.gcp_cloud_lane_run_monitor_config_v1"
REPORT_SCHEMA = "cppmega.gcp_cloud_lane_run_monitor_report_v1"
RUNNER_FAILURE_SCHEMA = "cppmega.cloud_lane_bootstrap_failure_v1"
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_ZONE_RE = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_INSTANCE_RE = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_ACCOUNT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*@[a-z0-9.-]+$")
_PHYSICAL_WORKER_RE = re.compile(r"^physical-([0-9]{4})$")
_HEARTBEAT_PATH_RE = re.compile(
    r"^control/cloud-lane-heartbeats/([0-9a-f]{64})/"
    r"(physical-[0-9]{4})/([0-9]{6})-([0-9a-f]{64})\.heartbeat\.json$"
)
_POOL_COMPLETION_PATH_RE = re.compile(
    r"^control/cloud-lane-completed/([0-9a-f]{64})/"
    r"(physical-[0-9]{4})\.complete\.json$"
)
_POOL_FAILURE_PATH_RE = re.compile(
    r"^control/cloud-lane-failures/([0-9a-f]{64})/"
    r"(physical-[0-9]{4})/([0-9a-f]{64})\.failure\.json$"
)
_RUNNER_COMPLETION_PATH_RE = re.compile(
    r"^control/cloud-lane-runner-completions/([0-9a-f]{64})/"
    r"(physical-[0-9]{4})/([0-9a-f]{64})\.complete\.json$"
)
_RUNNER_FAILURE_PATH_RE = re.compile(
    r"^control/cloud-lane-runner-failures/([0-9a-f]{64})/"
    r"(physical-[0-9]{4})/([0-9a-f]{64})\.failure\.json$"
)
_READY_PATH_RE = re.compile(r"^control/ready/[a-z0-9.-]+\.json$")
_HTTP_STATUS_RE = re.compile(
    r"(?:\bHTTP(?:/[0-9.]+)?(?:\s*error)?\s*|"
    r"\breturned error:\s*|\bstatus(?:\s+code)?\s*[:=_-]\s*)"
    r"([1-5][0-9]{2})\b",
    re.IGNORECASE,
)
_DETERMINISTIC_MARKERS = (
    "access denied",
    "authentication failed",
    "connection refused",
    "connection reset",
    "could not resolve",
    "forbidden",
    "invalid credentials",
    "network is unreachable",
    "permission denied",
    "repository not found",
    "timed out",
    "timeout",
    "unauthorized",
)
_MAX_FUTURE_CLOCK_SKEW = timedelta(minutes=5)
_CONFIG_FIELDS = {
    "schema",
    "project_id",
    "gcloud_account",
    "zone",
    "instance_name",
    "run_id",
    "run_root",
    "output_prefix",
    "manifest_sha256",
    "manifest_file_sha256",
    "code_revision",
    "physical_worker",
    "deployed_worker_count",
    "assignment_pool_size",
    "heartbeat_required",
    "heartbeat_max_age_seconds",
    "report_path",
}


class CloudLaneMonitorClient(Protocol):
    def describe_instance(
        self, *, project_id: str, zone: str, instance_name: str
    ) -> Mapping[str, object] | None: ...

    def list_objects(self, prefix: str) -> Sequence[Mapping[str, object]]: ...

    def read_json(self, metadata: Mapping[str, object]) -> "ExactJsonRead": ...


@dataclass(frozen=True)
class ExactJsonRead:
    """One exact-generation JSON object and the SHA-256 of its stored bytes."""

    value: Mapping[str, object]
    content_sha256: str


def _confirmed_http_429(detail: str) -> bool:
    lowered = detail.lower()
    if any(marker in lowered for marker in _DETERMINISTIC_MARKERS):
        return False
    statuses = [int(value) for value in _HTTP_STATUS_RE.findall(lowered)]
    return bool(statuses) and all(status == 429 for status in statuses)


class GcloudCloudLaneMonitorClient:
    def __init__(
        self,
        *,
        executable: str = "gcloud",
        account: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.executable = executable
        self.account = account
        self.runner = runner

    def _run(self, argv: Sequence[str]) -> str:
        command = list(argv)
        if self.account is not None:
            command.append(f"--account={self.account}")
        completed = self.runner(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            return completed.stdout or ""
        detail = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        if _confirmed_http_429(detail):
            raise TransientTransportError(
                f"confirmed HTTP 429 from command {command!r}: {detail[-4000:]}"
            )
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command!r}\n"
            f"{detail[-8000:]}"
        )

    @staticmethod
    def _json(raw: str, *, where: str) -> object:
        if len(raw.encode("utf-8")) > MAX_METADATA_BYTES:
            raise ContractError(f"{where} exceeds the metadata bound")

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    raise ContractError(
                        f"{where} contains duplicate JSON key {key!r}"
                    )
                value[key] = item
            return value

        try:
            return json.loads(raw, object_pairs_hook=reject_duplicates)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{where} is not valid JSON") from exc

    def describe_instance(
        self, *, project_id: str, zone: str, instance_name: str
    ) -> Mapping[str, object] | None:
        command = [
            self.executable,
            "compute",
            "instances",
            "describe",
            instance_name,
            f"--project={project_id}",
            f"--zone={zone}",
            "--format=json(name,status,lastStartTimestamp,machineType,labels)",
        ]
        if self.account is not None:
            command.append(f"--account={self.account}")
        completed = self.runner(
            command, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            detail = f"{completed.stdout or ''}\n{completed.stderr or ''}"
            lowered = detail.lower()
            if "was not found" in lowered or (
                "could not fetch resource" in lowered and "not found" in lowered
            ):
                return None
            if _confirmed_http_429(detail):
                raise TransientTransportError(
                    f"confirmed HTTP 429 describing {instance_name}: {detail[-4000:]}"
                )
            raise RuntimeError(
                f"instance describe failed ({completed.returncode}): {detail[-8000:]}"
            )
        value = self._json(completed.stdout or "", where="instance describe")
        if not isinstance(value, dict):
            raise ContractError("instance describe must be a JSON object")
        return value

    def list_objects(self, prefix: str) -> Sequence[Mapping[str, object]]:
        raw = self._run(
            [
                self.executable,
                "storage",
                "objects",
                "list",
                f"{prefix}/**",
                "--format=json(name,size,generation)",
            ]
        )
        value = self._json(raw, where=f"object listing {prefix}")
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ContractError("GCS object listing must be a JSON array of objects")
        return value

    def read_json(self, metadata: Mapping[str, object]) -> ExactJsonRead:
        uri = require_nonempty(metadata.get("uri"), where="object metadata uri")
        generation = require_nonempty(
            metadata.get("generation"), where="object metadata generation"
        )
        raw = self._run(
            [self.executable, "storage", "cat", f"{uri}#{generation}"]
        )
        encoded = raw.encode("utf-8")
        expected_size = require_int(
            metadata.get("size_bytes"), where="object metadata size"
        )
        if len(encoded) != expected_size:
            raise ContractError(
                f"exact-generation object {uri}#{generation} size drifted"
            )
        value = self._json(raw, where=f"exact-generation object {uri}#{generation}")
        if not isinstance(value, dict):
            raise ContractError("immutable receipt must be a JSON object")
        return ExactJsonRead(
            value=value,
            content_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def _identifier(value: object, pattern: re.Pattern[str], *, where: str) -> str:
    text = require_nonempty(value, where=where)
    if pattern.fullmatch(text) is None:
        raise ContractError(f"{where} is not canonical")
    return text


def load_monitor_config(path: Path) -> dict[str, object]:
    _raw, value = load_json_object(path, where="cloud lane monitor config")
    require_exact_fields(value, _CONFIG_FIELDS, where="cloud lane monitor config")
    if value["schema"] != CONFIG_SCHEMA:
        raise ContractError("cloud lane monitor config schema drifted")
    project_id = _identifier(value["project_id"], _RUN_ID_RE, where="project_id")
    gcloud_account = _identifier(
        value["gcloud_account"], _ACCOUNT_RE, where="gcloud_account"
    )
    zone = _identifier(value["zone"], _ZONE_RE, where="zone")
    instance_name = _identifier(
        value["instance_name"], _INSTANCE_RE, where="instance_name"
    )
    run_id = _identifier(value["run_id"], _RUN_ID_RE, where="run_id")
    run_root = validate_gcs_uri(value["run_root"], where="run_root")
    output_prefix = validate_gcs_uri(value["output_prefix"], where="output_prefix")
    if not run_root.endswith(f"/runs/{run_id}"):
        raise ContractError("run_root is not bound to run_id")
    manifest_sha256 = require_sha256(
        value["manifest_sha256"], where="manifest_sha256"
    )
    manifest_file_sha256 = require_sha256(
        value["manifest_file_sha256"], where="manifest_file_sha256"
    )
    code_revision = require_git_object(value["code_revision"], where="code_revision")
    deployed_worker_count = require_int(
        value["deployed_worker_count"], where="deployed_worker_count", minimum=1
    )
    assignment_pool_size = require_int(
        value["assignment_pool_size"], where="assignment_pool_size", minimum=1
    )
    if assignment_pool_size < deployed_worker_count:
        raise ContractError("assignment_pool_size is smaller than deployed_worker_count")
    physical_worker = _identifier(
        value["physical_worker"], _PHYSICAL_WORKER_RE, where="physical_worker"
    )
    match = _PHYSICAL_WORKER_RE.fullmatch(physical_worker)
    assert match is not None
    if int(match.group(1)) >= deployed_worker_count:
        raise ContractError("physical_worker is outside deployed_worker_count")
    heartbeat_required = value["heartbeat_required"]
    if not isinstance(heartbeat_required, bool):
        raise ContractError("heartbeat_required must be boolean")
    heartbeat_max_age_seconds = require_int(
        value["heartbeat_max_age_seconds"],
        where="heartbeat_max_age_seconds",
        minimum=1,
    )
    report_path = Path(require_nonempty(value["report_path"], where="report_path"))
    if not report_path.is_absolute():
        raise ContractError("report_path must be absolute")
    return {
        "schema": CONFIG_SCHEMA,
        "project_id": project_id,
        "gcloud_account": gcloud_account,
        "zone": zone,
        "instance_name": instance_name,
        "run_id": run_id,
        "run_root": run_root,
        "output_prefix": output_prefix,
        "manifest_sha256": manifest_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "code_revision": code_revision,
        "physical_worker": physical_worker,
        "deployed_worker_count": deployed_worker_count,
        "assignment_pool_size": assignment_pool_size,
        "heartbeat_required": heartbeat_required,
        "heartbeat_max_age_seconds": heartbeat_max_age_seconds,
        "report_path": str(report_path),
    }


def _object_metadata(
    raw: Mapping[str, object], *, bucket: str
) -> dict[str, object]:
    name = require_nonempty(raw.get("name"), where="object name")
    if name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ContractError("object name is unsafe")
    generation = require_nonempty(raw.get("generation"), where=f"{name} generation")
    if not generation.isdigit() or int(generation) <= 0:
        raise ContractError(f"{name} generation is invalid")
    size_value = raw.get("size")
    if isinstance(size_value, str) and size_value.isdigit():
        size = int(size_value)
    else:
        size = require_int(size_value, where=f"{name} size")
    return {
        "name": name,
        "uri": f"gs://{bucket}/{name}",
        "generation": generation,
        "size_bytes": size,
    }


def _relative_control_path(name: str, *, run_id: str) -> str:
    prefix = f"runs/{run_id}/"
    if not name.startswith(prefix):
        raise ContractError(f"control object escaped run prefix: {name}")
    return name[len(prefix) :]


def _receipt_digest(value: Mapping[str, object], *, where: str) -> str:
    expected = require_sha256(value.get("receipt_sha256"), where=f"{where} receipt")
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    actual = canonical_sha256(payload)
    if actual != expected:
        raise ContractError(f"{where} self-digest drifted")
    return expected


def _validate_pool_completion(
    value: Mapping[str, object], config: Mapping[str, object]
) -> dict[str, object]:
    require_exact_fields(
        value,
        {
            "schema",
            "status",
            "kind",
            "manifest_sha256",
            "manifest_file_sha256",
            "code_revision",
            "adapter_sha256",
            "physical_worker_index",
            "physical_worker_count",
            "logical_workers",
            "logical_worker_completions",
            "totals",
            "training_ready",
            "receipt_sha256",
        },
        where="pool completion",
    )
    if value.get("schema") != POOL_COMPLETION_SCHEMA or value.get("status") != "complete":
        raise ContractError("pool completion schema/status drifted")
    if pool_completion_sha256(value) != value.get("receipt_sha256"):
        raise ContractError("pool completion self-digest drifted")
    for key in ("manifest_sha256", "manifest_file_sha256", "code_revision"):
        if value.get(key) != config[key]:
            raise ContractError(f"pool completion {key} drifted")
    if value.get("physical_worker_index") != int(str(config["physical_worker"])[-4:]):
        raise ContractError("pool completion physical worker drifted")
    if value.get("physical_worker_count") != config["assignment_pool_size"]:
        raise ContractError("pool completion assignment pool size drifted")
    if value.get("training_ready") is not False:
        raise ContractError("pool completion must remain training_ready=false")
    require_nonempty(value.get("kind"), where="pool completion kind")
    require_sha256(value.get("adapter_sha256"), where="pool completion adapter")
    raw_workers = value.get("logical_workers")
    if not isinstance(raw_workers, list) or not raw_workers:
        raise ContractError("pool completion logical workers are missing")
    logical_workers = [
        require_nonempty(item, where="pool completion logical worker")
        for item in raw_workers
    ]
    if len(set(logical_workers)) != len(logical_workers):
        raise ContractError("pool completion logical workers are not unique")
    raw_completions = value.get("logical_worker_completions")
    if not isinstance(raw_completions, list) or len(raw_completions) != len(
        logical_workers
    ):
        raise ContractError("pool completion logical receipt coverage drifted")
    logical_completions = []
    output_prefix = str(config["output_prefix"]).rstrip("/") + "/"
    for index, raw in enumerate(raw_completions):
        if not isinstance(raw, Mapping):
            raise ContractError("pool logical completion must be an object")
        require_exact_fields(
            raw,
            {"worker", "receipt_sha256", "publication"},
            where="pool logical completion",
        )
        worker = require_nonempty(
            raw.get("worker"), where="pool logical completion worker"
        )
        if worker != logical_workers[index]:
            raise ContractError("pool logical completion worker order drifted")
        receipt_sha256 = require_sha256(
            raw.get("receipt_sha256"),
            where="pool logical completion receipt",
        )
        publication = raw.get("publication")
        if not isinstance(publication, Mapping):
            raise ContractError("pool logical completion publication is missing")
        require_exact_fields(
            publication,
            {"uri", "generation", "size_bytes", "sha256"},
            where="pool logical completion publication",
        )
        uri = validate_gcs_uri(
            publication.get("uri"), where="pool logical completion publication URI"
        )
        if not uri.startswith(output_prefix):
            raise ContractError("pool logical completion publication escaped output prefix")
        generation = require_nonempty(
            publication.get("generation"),
            where="pool logical completion publication generation",
        )
        if not generation.isdecimal() or int(generation) < 1:
            raise ContractError("pool logical completion publication generation is invalid")
        size_bytes = require_int(
            publication.get("size_bytes"),
            where="pool logical completion publication size",
            minimum=1,
        )
        sha256 = require_sha256(
            publication.get("sha256"),
            where="pool logical completion publication SHA-256",
        )
        logical_completions.append(
            {
                "worker": worker,
                "receipt_sha256": receipt_sha256,
                "publication": {
                    "uri": uri,
                    "generation": generation,
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                },
            }
        )
    totals = value.get("totals")
    if not isinstance(totals, Mapping):
        raise ContractError("pool completion totals are missing")
    require_exact_fields(
        totals,
        {
            "source_record_count",
            "candidate_document_count",
            "valid_tokens",
            "assignment_receipt_count",
        },
        where="pool completion totals",
    )
    normalized_totals = {
        "source_record_count": require_int(
            totals.get("source_record_count"),
            where="pool completion source_record_count",
            minimum=1,
        ),
        "candidate_document_count": require_int(
            totals.get("candidate_document_count"),
            where="pool completion candidate_document_count",
        ),
        "valid_tokens": require_int(
            totals.get("valid_tokens"),
            where="pool completion valid_tokens",
        ),
        "assignment_receipt_count": require_int(
            totals.get("assignment_receipt_count"),
            where="pool completion assignment_receipt_count",
            minimum=len(logical_workers),
        ),
    }
    return {
        "receipt_sha256": value["receipt_sha256"],
        "totals": normalized_totals,
        "logical_worker_completions": logical_completions,
    }


def _validate_pool_failure(
    value: Mapping[str, object], config: Mapping[str, object]
) -> dict[str, object]:
    require_exact_fields(
        value,
        {
            "schema",
            "status",
            "kind",
            "manifest_sha256",
            "manifest_file_sha256",
            "physical_worker_index",
            "physical_worker_count",
            "diagnostics",
            "retry_exit_code",
            "training_ready",
            "receipt_sha256",
        },
        where="pool failure",
    )
    if value.get("schema") != POOL_FAILURE_SCHEMA or value.get("status") != "failed":
        raise ContractError("pool failure schema/status drifted")
    if pool_failure_sha256(value) != value.get("receipt_sha256"):
        raise ContractError("pool failure self-digest drifted")
    for key in ("manifest_sha256", "manifest_file_sha256"):
        if value.get(key) != config[key]:
            raise ContractError(f"pool failure {key} drifted")
    if value.get("physical_worker_index") != int(str(config["physical_worker"])[-4:]):
        raise ContractError("pool failure physical worker drifted")
    if value.get("physical_worker_count") != config["assignment_pool_size"]:
        raise ContractError("pool failure assignment pool size drifted")
    require_nonempty(value.get("kind"), where="pool failure kind")
    retry_exit_code = require_int(value.get("retry_exit_code"), where="retry_exit_code")
    if retry_exit_code not in {2, 75}:
        raise ContractError("pool failure retry_exit_code is invalid")
    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ContractError("pool failure diagnostics are missing")
    confirmed_values = []
    for item in diagnostics:
        if not isinstance(item, Mapping):
            raise ContractError("pool failure diagnostic must be an object")
        require_exact_fields(
            item,
            {
                "worker",
                "error_type",
                "diagnostic_sha256",
                "confirmed_http_429",
            },
            where="pool failure diagnostic",
        )
        require_nonempty(item.get("worker"), where="pool failure diagnostic worker")
        require_nonempty(
            item.get("error_type"), where="pool failure diagnostic error type"
        )
        require_sha256(
            item.get("diagnostic_sha256"),
            where="pool failure diagnostic SHA-256",
        )
        observed = item.get("confirmed_http_429")
        if not isinstance(observed, bool):
            raise ContractError("pool failure diagnostic 429 flag must be boolean")
        confirmed_values.append(observed)
    confirmed = all(confirmed_values)
    if (retry_exit_code == 75) != confirmed:
        raise ContractError("pool failure 429 classification drifted")
    if value.get("training_ready") is not False:
        raise ContractError("pool failure must remain training_ready=false")
    return {
        "receipt_sha256": value["receipt_sha256"],
        "exit_code": retry_exit_code,
        "confirmed_http_429": confirmed,
    }


def _validate_runner_failure(
    value: Mapping[str, object], config: Mapping[str, object]
) -> dict[str, object]:
    expected_fields = {
        "schema",
        "status",
        "stage",
        "exit_code",
        "manifest_sha256",
        "manifest_file_sha256",
        "physical_worker",
        "diagnostic_sha256",
        "confirmed_http_429",
        "training_ready",
        "receipt_sha256",
    }
    require_exact_fields(value, expected_fields, where="runner failure")
    if value["schema"] != RUNNER_FAILURE_SCHEMA or value["status"] != "failed":
        raise ContractError("runner failure schema/status drifted")
    _receipt_digest(value, where="runner failure")
    for key in ("manifest_sha256", "manifest_file_sha256", "physical_worker"):
        if value[key] != config[key]:
            raise ContractError(f"runner failure {key} drifted")
    require_sha256(value["diagnostic_sha256"], where="diagnostic_sha256")
    exit_code = require_int(value["exit_code"], where="runner failure exit_code")
    if exit_code not in {2, 75}:
        raise ContractError("runner failure exit code is invalid")
    confirmed = value["confirmed_http_429"]
    if not isinstance(confirmed, bool) or (exit_code == 75) != confirmed:
        raise ContractError("runner failure 429 classification drifted")
    if value["training_ready"] is not False:
        raise ContractError("runner failure must remain training_ready=false")
    return {
        "receipt_sha256": value["receipt_sha256"],
        "exit_code": exit_code,
        "confirmed_http_429": confirmed,
        "stage": require_nonempty(value["stage"], where="runner failure stage"),
    }


def _parse_timestamp(value: object, *, where: str) -> datetime:
    text = require_nonempty(value, where=where)
    if not text.endswith("Z"):
        raise ContractError(f"{where} must use canonical UTC")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{where} is invalid") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"{where} must be timezone-aware")
    return parsed


def _is_target_control_worker(
    worker: str, *, config: Mapping[str, object]
) -> bool:
    match = _PHYSICAL_WORKER_RE.fullmatch(worker)
    if match is None:
        raise ContractError("control physical worker is not canonical")
    worker_count = require_int(
        config["assignment_pool_size"],
        where="assignment_pool_size",
        minimum=1,
    )
    if int(match.group(1)) >= worker_count:
        raise ContractError("control physical worker is outside physical_worker_count")
    return worker == config["physical_worker"]


def monitor_cloud_lane_run(
    config: Mapping[str, object],
    *,
    client: CloudLaneMonitorClient,
    now: datetime | None = None,
) -> dict[str, object]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ContractError("monitor time must be timezone-aware")
    project_id = str(config["project_id"])
    zone = str(config["zone"])
    instance_name = str(config["instance_name"])
    run_id = str(config["run_id"])
    run_root = str(config["run_root"])
    output_prefix = str(config["output_prefix"])
    manifest_sha256 = str(config["manifest_sha256"])
    physical_worker = str(config["physical_worker"])
    bucket = run_root[len("gs://") :].split("/", 1)[0]

    instance = client.describe_instance(
        project_id=project_id, zone=zone, instance_name=instance_name
    )
    if instance is not None:
        if instance.get("name") != instance_name:
            raise ContractError("instance name drifted")
        labels = instance.get("labels")
        if not isinstance(labels, Mapping) or labels.get("run-id") != run_id:
            raise ContractError("instance run-id label drifted")

    control_raw = client.list_objects(f"{run_root}/control")
    output_raw = client.list_objects(output_prefix)
    control = [_object_metadata(item, bucket=bucket) for item in control_raw]
    output_bucket = output_prefix[len("gs://") :].split("/", 1)[0]
    outputs = [_object_metadata(item, bucket=output_bucket) for item in output_raw]
    output_object_prefix = output_prefix[len("gs://") :].split("/", 1)
    if len(output_object_prefix) != 2 or not output_object_prefix[1]:
        raise ContractError("output_prefix must include a non-empty object prefix")
    expected_output_name_prefix = output_object_prefix[1].rstrip("/") + "/"
    for item in outputs:
        if not str(item["name"]).startswith(expected_output_name_prefix):
            raise ContractError("output object escaped output_prefix")
        if require_int(item["size_bytes"], where="output object size") < 1:
            raise ContractError("output object must not be empty")
    output_by_uri = {str(item["uri"]): item for item in outputs}
    if len(output_by_uri) != len(outputs):
        raise ContractError("output object inventory contains duplicate URIs")

    heartbeats: list[tuple[int, str, dict[str, object]]] = []
    pool_completions: list[dict[str, object]] = []
    runner_completions: list[tuple[str, dict[str, object]]] = []
    pool_failures: list[tuple[str, dict[str, object]]] = []
    runner_failures: list[tuple[str, dict[str, object]]] = []
    ready_count = 0
    sibling_control_count = 0
    for metadata in control:
        relative = _relative_control_path(str(metadata["name"]), run_id=run_id)
        match = _HEARTBEAT_PATH_RE.fullmatch(relative)
        if match is not None:
            if match.group(1) != manifest_sha256:
                raise ContractError("heartbeat path binding drifted")
            if not _is_target_control_worker(match.group(2), config=config):
                sibling_control_count += 1
                continue
            heartbeats.append((int(match.group(3)), match.group(4), metadata))
            continue
        match = _POOL_COMPLETION_PATH_RE.fullmatch(relative)
        if match is not None:
            if match.group(1) != manifest_sha256:
                raise ContractError("pool completion path binding drifted")
            if not _is_target_control_worker(match.group(2), config=config):
                sibling_control_count += 1
                continue
            pool_completions.append(metadata)
            continue
        match = _POOL_FAILURE_PATH_RE.fullmatch(relative)
        if match is not None:
            if match.group(1) != manifest_sha256:
                raise ContractError("pool failure path binding drifted")
            if not _is_target_control_worker(match.group(2), config=config):
                sibling_control_count += 1
                continue
            pool_failures.append((match.group(3), metadata))
            continue
        match = _RUNNER_COMPLETION_PATH_RE.fullmatch(relative)
        if match is not None:
            if match.group(1) != manifest_sha256:
                raise ContractError("runner completion path binding drifted")
            if not _is_target_control_worker(match.group(2), config=config):
                sibling_control_count += 1
                continue
            runner_completions.append((match.group(3), metadata))
            continue
        match = _RUNNER_FAILURE_PATH_RE.fullmatch(relative)
        if match is not None:
            if match.group(1) != manifest_sha256:
                raise ContractError("runner failure path binding drifted")
            if not _is_target_control_worker(match.group(2), config=config):
                sibling_control_count += 1
                continue
            runner_failures.append((match.group(3), metadata))
            continue
        if _READY_PATH_RE.fullmatch(relative) is not None:
            ready_count += 1
            continue
        raise ContractError(f"unknown control object: {relative}")

    heartbeat_summary: dict[str, object] | None = None
    if heartbeats:
        sequences = sorted(sequence for sequence, _digest, _metadata in heartbeats)
        if sequences != list(range(sequences[-1] + 1)):
            raise ContractError("heartbeat sequence is not contiguous from zero")
        sequence, path_digest, latest_metadata = max(heartbeats, key=lambda item: item[0])
        heartbeat_read = client.read_json(latest_metadata)
        heartbeat = validate_worker_heartbeat(heartbeat_read.value)
        if heartbeat["receipt_sha256"] != path_digest or heartbeat["sequence"] != sequence:
            raise ContractError("latest heartbeat path/content drifted")
        for key in (
            "manifest_sha256",
            "manifest_file_sha256",
            "code_revision",
            "physical_worker",
        ):
            if heartbeat[key] != config[key]:
                raise ContractError(f"latest heartbeat {key} drifted")
        if heartbeat["physical_worker_count"] != config["assignment_pool_size"]:
            raise ContractError("latest heartbeat assignment_pool_size drifted")
        emitted = _parse_timestamp(heartbeat["emitted_at"], where="heartbeat emitted_at")
        if emitted - current > _MAX_FUTURE_CLOCK_SKEW:
            raise ContractError("heartbeat emitted_at is implausibly far in the future")
        age = max(0, int((current - emitted).total_seconds()))
        heartbeat_summary = {
            "count": len(heartbeats),
            "latest_sequence": sequence,
            "latest_emitted_at": heartbeat["emitted_at"],
            "latest_age_seconds": age,
            "latest_generation": latest_metadata["generation"],
            "latest_receipt_sha256": heartbeat["receipt_sha256"],
            "fresh": age
            <= require_int(
                config["heartbeat_max_age_seconds"],
                where="heartbeat_max_age_seconds",
                minimum=1,
            ),
        }

    if len(pool_completions) > 1 or len(runner_completions) > 1:
        raise ContractError("multiple completion receipts exist for one physical worker")
    completion_summary: dict[str, object] | None = None
    if pool_completions or runner_completions:
        if len(pool_completions) != 1 or len(runner_completions) != 1:
            raise ContractError("completion evidence is only partially published")
        pool_value = client.read_json(pool_completions[0]).value
        runner_path_digest, runner_metadata = runner_completions[0]
        runner_value = client.read_json(runner_metadata).value
        pool_summary = _validate_pool_completion(pool_value, config)
        runner_summary = _validate_pool_completion(runner_value, config)
        if runner_summary["receipt_sha256"] != runner_path_digest:
            raise ContractError("runner completion path/content digest drifted")
        if pool_value != runner_value or pool_summary != runner_summary:
            raise ContractError("pool and runner completion receipts differ")
        completion_summary = pool_summary
        logical_worker_completions = completion_summary.get(
            "logical_worker_completions"
        )
        if not isinstance(logical_worker_completions, list):
            raise ContractError("normalized logical completions are missing")
        for logical in logical_worker_completions:
            assert isinstance(logical, Mapping)
            publication = logical["publication"]
            assert isinstance(publication, Mapping)
            observed = output_by_uri.get(str(publication["uri"]))
            if observed is None:
                raise ContractError("logical completion publication is absent from outputs")
            if (
                str(observed["generation"]) != publication["generation"]
                or observed["size_bytes"] != publication["size_bytes"]
            ):
                raise ContractError("logical completion publication metadata drifted")

    validated_pool_failures = []
    for path_digest, metadata in pool_failures:
        failure = _validate_pool_failure(client.read_json(metadata).value, config)
        if failure["receipt_sha256"] != path_digest:
            raise ContractError("pool failure path/content digest drifted")
        validated_pool_failures.append(failure)
    validated_runner_failures = []
    for path_digest, metadata in runner_failures:
        failure_read = client.read_json(metadata)
        failure = _validate_runner_failure(failure_read.value, config)
        if path_digest not in {
            failure["receipt_sha256"],
            failure_read.content_sha256,
        }:
            raise ContractError("runner failure path/content digest drifted")
        failure["path_digest_kind"] = (
            "receipt_sha256"
            if path_digest == failure["receipt_sha256"]
            else "content_sha256"
        )
        validated_runner_failures.append(failure)
    if completion_summary is not None and (validated_pool_failures or validated_runner_failures):
        raise ContractError("run contains both completion and failure evidence")

    output_bytes = sum(
        require_int(item["size_bytes"], where="output object size")
        for item in outputs
    )
    instance_status = (
        None
        if instance is None
        else require_nonempty(instance.get("status"), where="instance status")
    )
    retry_eligible = bool(validated_runner_failures) and all(
        item["confirmed_http_429"] is True and item["exit_code"] == 75
        for item in (*validated_pool_failures, *validated_runner_failures)
    )
    cleanup_authorized = False
    if completion_summary is not None:
        if not outputs:
            raise ContractError("completed run has no physical output objects")
        state = "completed_verified"
        cleanup_authorized = True
    elif validated_runner_failures:
        state = "failed_confirmed_429" if retry_eligible else "failed_deterministic"
    elif validated_pool_failures:
        state = "failure_partial_without_runner_receipt"
    elif instance is None:
        state = "missing_instance_without_terminal_receipt"
    elif instance_status != "RUNNING":
        state = "instance_not_running_without_terminal_receipt"
    elif heartbeat_summary is None:
        state = "running_without_heartbeat"
        if config["heartbeat_required"] is True:
            state = "running_missing_required_heartbeat"
    elif heartbeat_summary["fresh"] is True:
        state = "running"
    else:
        state = "running_stale_heartbeat"

    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "checked_at": current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "run_root": run_root,
        "output_prefix": output_prefix,
        "manifest_sha256": manifest_sha256,
        "manifest_file_sha256": config["manifest_file_sha256"],
        "code_revision": config["code_revision"],
        "physical_worker": physical_worker,
        "deployed_worker_count": config["deployed_worker_count"],
        "assignment_pool_size": config["assignment_pool_size"],
        "state": state,
        "instance": {
            "name": instance_name,
            "status": instance_status,
            "present": instance is not None,
        },
        "counts": {
            "ready_objects": ready_count,
            "sibling_control_objects": sibling_control_count,
            "heartbeat_objects": len(heartbeats),
            "pool_completion_objects": len(pool_completions),
            "runner_completion_objects": len(runner_completions),
            "pool_failure_objects": len(pool_failures),
            "runner_failure_objects": len(runner_failures),
            "output_objects": len(outputs),
            "output_bytes": output_bytes,
        },
        "heartbeat": heartbeat_summary,
        "completion": completion_summary,
        "failures": {
            "pool": validated_pool_failures,
            "runner": validated_runner_failures,
        },
        "retry_eligible": retry_eligible,
        "cleanup_authorized": cleanup_authorized,
        "training_ready": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_monitor_config(args.config)
        report = monitor_cloud_lane_run(
            config,
            client=GcloudCloudLaneMonitorClient(
                account=str(config["gcloud_account"])
            ),
        )
        atomic_write_json(Path(str(config["report_path"])), report)
        print(json.dumps(report, sort_keys=True))
        if report["state"] == "failed_confirmed_429":
            return 75
        if report["state"] in {
            "failed_deterministic",
            "failure_partial_without_runner_receipt",
            "missing_instance_without_terminal_receipt",
            "instance_not_running_without_terminal_receipt",
            "running_missing_required_heartbeat",
            "running_stale_heartbeat",
        }:
            return 2
    except TransientTransportError as exc:
        print(f"GCP cloud lane monitor confirmed HTTP 429: {exc}", file=sys.stderr)
        return 75
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"GCP cloud lane monitor failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "CONFIG_SCHEMA",
    "ExactJsonRead",
    "GcloudCloudLaneMonitorClient",
    "REPORT_SCHEMA",
    "load_monitor_config",
    "monitor_cloud_lane_run",
]
