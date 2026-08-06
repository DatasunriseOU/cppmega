#!/usr/bin/env python3
"""Receipt-bound monitor for one distributed GCP source run.

The monitor is intentionally conservative.  It treats exit 75 as transient,
exit 2 as deterministic, and never performs a VM replacement.  A transient
failure is marked replacement-eligible only after serial diagnostics and a
diagnostics receipt have been published immutably and read back by the shared
GCS object-store transport.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Mapping, Protocol, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    gcs_join,
    load_json_object,
    require_exact_fields,
    require_int,
    require_nonempty,
    require_sha256,
    sha256_file,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.source_manifest import (  # noqa: E402
    validate_source_manifest,
)
from scripts.distributed_data_prep.source_slot_scheduler import (  # noqa: E402
    slot_specs,
    validate_slot_completion_receipt,
)
from scripts.distributed_data_prep.source_work_queue import (  # noqa: E402
    ASSIGNMENT_CLAIM_SCHEMA,
    ASSIGNMENT_HEARTBEAT_SCHEMA,
)
from scripts.distributed_data_prep.source_worker import (  # noqa: E402
    GcloudObjectStore,
    ObjectStore,
    assignment_completion_uri,
    validate_assignment_completion_receipt,
)

MONITOR_SCHEMA = "cppmega.gcp_source_run_monitor_v1"
STATE_SCHEMA = "cppmega.gcp_source_run_monitor_state_v1"
REPORT_SCHEMA = "cppmega.gcp_source_run_monitor_report_v1"
TERMINAL_SCHEMA = "cppmega.gcp_source_run_terminal_receipt_v1"
DIAGNOSTICS_SCHEMA = "cppmega.gcp_source_failure_diagnostics_v1"
TRANSIENT_EXIT_CODE = 75
DETERMINISTIC_EXIT_CODE = 2
_WORKER_NAME_RE = re.compile(r"[a-z][a-z0-9-]{0,62}")
_ZONE_NAME_RE = re.compile(r"[a-z][a-z0-9-]{0,62}")
_CLAIM_FILENAME_RE = re.compile(r"([0-9]{4})\.claim\.json")
_HEARTBEAT_CLAIM_DIR_RE = re.compile(r"([0-9]{4})-([0-9a-f]{64})")
_HEARTBEAT_FILENAME_RE = re.compile(r"([0-9]{8})\.heartbeat\.json")
_JSON_READ_BATCH_SIZE = 1_024
_JSON_READ_MAX_WORKERS = 64
_JSON_READ_MAX_RETRIES = 3
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_HTTP_429_RE = re.compile(
    r"(?:^|[^0-9])429(?:[^0-9]|$)|too many requests|resource_exhausted",
    re.IGNORECASE,
)


def _is_transient_http_status(status: int) -> bool:
    return status in {408, 429, 500, 502, 503, 504}


class MonitorError(ContractError):
    """The run inventory or monitor configuration is unsafe."""


class RunClient(Protocol):
    def list_objects(self, pattern: str) -> list[dict[str, object]]: ...

    def read_json(
        self, metadata: Mapping[str, object]
    ) -> tuple[bytes, dict[str, object]]: ...

    def list_instances(
        self, *, project_id: str, run_id: str
    ) -> list[dict[str, object]]: ...

    def serial_output(self, *, project_id: str, zone: str, instance: str) -> bytes: ...


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[bytes]]


def _default_command_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        timeout=180,
    )


class GcloudRunClient:
    """Small read-only GCP client used by the monitor."""

    def __init__(
        self,
        executable: str = "gcloud",
        *,
        runner: CommandRunner = _default_command_runner,
        urlopen: Callable[..., object] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.executable = executable
        self.runner = runner
        self.urlopen = urlopen
        self.sleeper = sleeper
        self._cached_access_token: str | None = None
        self._access_token_lock = threading.Lock()

    def _run(self, argv: Sequence[str], *, where: str) -> bytes:
        completed = self.runner([self.executable, *argv])
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-4000:]
            raise MonitorError(
                f"{where} failed with exit {completed.returncode}: {detail}"
            )
        return completed.stdout

    def _access_token(self) -> str:
        with self._access_token_lock:
            if self._cached_access_token is not None:
                return self._cached_access_token
            completed = self.runner([self.executable, "auth", "print-access-token"])
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace")[-4000:]
                raise MonitorError(
                    "GCS monitor access-token request failed with exit "
                    f"{completed.returncode}: {detail}"
                )
            try:
                token = completed.stdout.decode("ascii", errors="strict").strip()
            except UnicodeDecodeError as exc:
                raise MonitorError("GCS monitor access token is invalid") from exc
            if not token or any(character.isspace() for character in token):
                raise MonitorError("GCS monitor access token is invalid")
            self._cached_access_token = token
            return token

    def _expire_access_token(self, token: str) -> None:
        with self._access_token_lock:
            if self._cached_access_token == token:
                self._cached_access_token = None

    def list_objects(self, pattern: str) -> list[dict[str, object]]:
        validate_gcs_uri(
            pattern.replace("**", "object").replace("*", "object"),
            where="GCS list pattern",
        )
        bucket, object_pattern = pattern[len("gs://") :].split("/", 1)
        wildcard_at = min(
            (index for index in (object_pattern.find("*"),) if index >= 0),
            default=len(object_pattern),
        )
        prefix = object_pattern[:wildcard_at]
        result: list[dict[str, object]] = []
        seen: set[str] = set()
        params = {
            "prefix": prefix,
            "maxResults": "1000",
            "fields": "nextPageToken,items(name,generation,size,updated)",
        }
        page_tokens: set[str] = set()
        while True:
            endpoint = (
                "https://storage.googleapis.com/storage/v1/b/"
                f"{urllib.parse.quote(bucket, safe='')}/o?"
                + urllib.parse.urlencode(params)
            )
            page: Mapping[str, object] | None = None
            for attempt in range(_JSON_READ_MAX_RETRIES + 1):
                access_token = self._access_token()
                try:
                    request = urllib.request.Request(
                        endpoint,
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
                    with self.urlopen(request, timeout=180) as response:
                        status = getattr(response, "status", None)
                        raw = response.read()
                    if status != 200:
                        raise MonitorError(
                            f"GCS inventory returned HTTP {status}: {pattern}"
                        )
                    decoded = json.loads(raw)
                    if not isinstance(decoded, Mapping):
                        raise MonitorError("GCS inventory page is not an object")
                    page = decoded
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 401 and attempt < _JSON_READ_MAX_RETRIES:
                        self._expire_access_token(access_token)
                        continue
                    if (
                        not _is_transient_http_status(exc.code)
                        or attempt >= _JSON_READ_MAX_RETRIES
                    ):
                        raise MonitorError(
                            f"GCS inventory failed with HTTP {exc.code}: {pattern}"
                        ) from exc
                except urllib.error.URLError as exc:
                    if attempt >= _JSON_READ_MAX_RETRIES:
                        raise MonitorError(
                            f"GCS inventory transport failed: {pattern}"
                        ) from exc
                except json.JSONDecodeError as exc:
                    raise MonitorError("GCS inventory returned invalid JSON") from exc
                self.sleeper(float(2**attempt))
            if page is None:  # pragma: no cover - loop always returns or raises
                raise AssertionError("unreachable")
            items = page.get("items", [])
            if not isinstance(items, list):
                raise MonitorError("GCS inventory items are invalid")
            for index, item in enumerate(items):
                if not isinstance(item, Mapping):
                    raise MonitorError(f"GCS inventory item {index} is invalid")
                name = require_nonempty(
                    item.get("name"), where=f"GCS inventory item {index} name"
                )
                uri = validate_gcs_uri(
                    f"gs://{bucket}/{name}", where=f"GCS inventory item {index} URI"
                )
                if not _gcs_pattern_matches(uri, pattern):
                    continue
                generation = str(item.get("generation", ""))
                if not generation.isdecimal() or int(generation) < 1:
                    raise MonitorError(
                        f"GCS inventory item {index} has an invalid generation"
                    )
                try:
                    size_bytes = int(item.get("size"))
                except (TypeError, ValueError) as exc:
                    raise MonitorError(
                        f"GCS inventory item {index} has an invalid size"
                    ) from exc
                if size_bytes < 1 or uri in seen:
                    raise MonitorError(
                        f"GCS inventory item {index} is empty or duplicated"
                    )
                seen.add(uri)
                result.append(
                    {
                        "uri": uri,
                        "generation": generation,
                        "size_bytes": size_bytes,
                        "updated": str(item.get("updated", "")),
                    }
                )
            next_page_token = page.get("nextPageToken")
            if next_page_token is None:
                break
            token = require_nonempty(next_page_token, where="GCS inventory page token")
            if token in page_tokens:
                raise MonitorError("GCS inventory page token repeated")
            page_tokens.add(token)
            params["pageToken"] = token
        return sorted(result, key=lambda item: str(item["uri"]))

    def read_json(
        self, metadata: Mapping[str, object]
    ) -> tuple[bytes, dict[str, object]]:
        uri = validate_gcs_uri(metadata.get("uri"), where="GCS JSON URI")
        generation = str(metadata.get("generation", ""))
        if not generation.isdecimal():
            raise MonitorError(f"GCS JSON generation is invalid: {uri}")
        raw = self._run(
            ["storage", "cat", f"{uri}#{generation}"],
            where=f"reading {uri}#{generation}",
        )
        if len(raw) != int(metadata.get("size_bytes", -1)):
            raise MonitorError(f"GCS JSON size drifted: {uri}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MonitorError(f"GCS object is not valid JSON: {uri}") from exc
        if not isinstance(value, dict):
            raise MonitorError(f"GCS JSON object is not a mapping: {uri}")
        return raw, value

    def read_json_many(
        self, metadata_rows: Sequence[Mapping[str, object]]
    ) -> list[tuple[bytes, dict[str, object]]]:
        """Read a bounded batch through generation-pinned GCS media requests."""
        rows = list(metadata_rows)
        if not rows:
            return []
        requests: list[tuple[str, str, int, str]] = []
        for index, metadata in enumerate(rows):
            uri = validate_gcs_uri(
                metadata.get("uri"), where=f"GCS JSON batch URI {index}"
            )
            generation = str(metadata.get("generation", ""))
            if not generation.isdecimal() or int(generation) < 1:
                raise MonitorError(f"GCS JSON batch generation is invalid: {uri}")
            try:
                size_bytes = int(metadata["size_bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MonitorError(f"GCS JSON batch size is invalid: {uri}") from exc
            if size_bytes < 1:
                raise MonitorError(f"GCS JSON batch object is empty: {uri}")
            bucket, object_name = uri[len("gs://") :].split("/", 1)
            endpoint = (
                "https://storage.googleapis.com/download/storage/v1/b/"
                f"{urllib.parse.quote(bucket, safe='')}/o/"
                f"{urllib.parse.quote(object_name, safe='')}?"
                + urllib.parse.urlencode({"alt": "media", "generation": generation})
            )
            requests.append((uri, generation, size_bytes, endpoint))
        def read_one(request: tuple[str, str, int, str]) -> bytes:
            uri, generation, size_bytes, endpoint = request
            for attempt in range(_JSON_READ_MAX_RETRIES + 1):
                access_token = self._access_token()
                try:
                    http_request = urllib.request.Request(
                        endpoint,
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
                    with self.urlopen(http_request, timeout=180) as response:
                        status = getattr(response, "status", None)
                        raw = response.read()
                    if status != 200:
                        raise MonitorError(
                            f"GCS JSON generation read returned HTTP {status}: "
                            f"{uri}#{generation}"
                        )
                    if len(raw) != size_bytes:
                        raise MonitorError(
                            f"GCS JSON generation size drifted: {uri}#{generation}"
                        )
                    return raw
                except urllib.error.HTTPError as exc:
                    if exc.code == 401 and attempt < _JSON_READ_MAX_RETRIES:
                        self._expire_access_token(access_token)
                        continue
                    if (
                        not _is_transient_http_status(exc.code)
                        or attempt >= _JSON_READ_MAX_RETRIES
                    ):
                        raise MonitorError(
                            f"GCS JSON generation read failed with HTTP {exc.code}: "
                            f"{uri}#{generation}"
                        ) from exc
                except urllib.error.URLError as exc:
                    if attempt >= _JSON_READ_MAX_RETRIES:
                        raise MonitorError(
                            f"GCS JSON generation transport failed: {uri}#{generation}"
                        ) from exc
                self.sleeper(float(2**attempt))
            raise AssertionError("unreachable")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(_JSON_READ_MAX_WORKERS, len(requests))
        ) as executor:
            raw_rows = list(executor.map(read_one, requests))
        result: list[tuple[bytes, dict[str, object]]] = []
        for index, raw in enumerate(raw_rows):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise MonitorError(
                    f"GCS JSON batch object {index} is not valid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise MonitorError(f"GCS JSON batch object {index} is not a mapping")
            result.append((raw, value))
        return result

    def list_instances(
        self, *, project_id: str, run_id: str
    ) -> list[dict[str, object]]:
        raw = self._run(
            [
                "compute",
                "instances",
                "list",
                f"--project={project_id}",
                f"--filter=labels.run-id={run_id}",
                "--format=json(name,id,status,zone,lastStartTimestamp,labels)",
            ],
            where=f"listing instances for {run_id}",
        )
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MonitorError("gcloud returned invalid instance JSON") from exc
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise MonitorError("gcloud returned an invalid instance inventory")
        return [dict(row) for row in rows]

    def serial_output(self, *, project_id: str, zone: str, instance: str) -> bytes:
        return self._run(
            [
                "compute",
                "instances",
                "get-serial-port-output",
                instance,
                f"--project={project_id}",
                f"--zone={zone}",
                "--port=1",
                "--start=0",
            ],
            where=f"capturing serial diagnostics for {instance}",
        )


def _path(value: object, *, where: str) -> Path:
    return Path(require_nonempty(value, where=where))


def _positive_int(value: object, *, where: str) -> int:
    return require_int(value, where=where, minimum=1)


def _zone_name(value: object, *, where: str) -> str:
    raw = require_nonempty(value, where=where)
    zone = raw.rsplit("/", 1)[-1]
    if _ZONE_NAME_RE.fullmatch(zone) is None:
        raise MonitorError(f"{where} is invalid")
    return zone


def _string_list(value: object, *, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise MonitorError(f"{where} must be a non-empty list")
    result = [
        require_nonempty(item, where=f"{where}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise MonitorError(f"{where} must be unique")
    return result


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    value = dict(config)
    expected = {
        "schema",
        "run_id",
        "run_root",
        "manifest_path",
        "manifest_file_sha256",
        "project_id",
        "zone",
        "physical_workers",
        "slots_per_worker",
        "expected_local_ssd_count",
        "resources",
        "state_path",
        "report_path",
        "terminal_receipt_path",
        "diagnostics_dir",
        "diagnostics_upload_prefix",
        "stale_after_seconds",
        "gcloud",
    }
    require_exact_fields(value, expected, where="GCP source monitor config")
    if value["schema"] != MONITOR_SCHEMA:
        raise MonitorError("GCP source monitor config schema is unsupported")
    run_id = require_nonempty(value["run_id"], where="run_id")
    run_root = validate_gcs_uri(value["run_root"], where="run_root")
    if not run_root.endswith(f"/{run_id}"):
        raise MonitorError("run_root is not bound to run_id")
    _path(value["manifest_path"], where="manifest_path")
    require_sha256(value["manifest_file_sha256"], where="manifest_file_sha256")
    require_nonempty(value["project_id"], where="project_id")
    value["zone"] = _zone_name(value["zone"], where="zone")
    workers = _string_list(value["physical_workers"], where="physical_workers")
    if any(_WORKER_NAME_RE.fullmatch(worker) is None for worker in workers):
        raise MonitorError("physical_workers contains an invalid instance name")
    slots = _positive_int(value["slots_per_worker"], where="slots_per_worker")
    if slots > 2:
        raise MonitorError("slots_per_worker exceeds the production scheduler bound")
    _positive_int(value["expected_local_ssd_count"], where="expected_local_ssd_count")
    resources = value["resources"]
    if not isinstance(resources, Mapping):
        raise MonitorError("resources must be an object")
    require_exact_fields(
        resources,
        {
            "parse_workers_per_slot",
            "memory_limit_gb_per_slot",
            "cpu_budget_vcpus",
            "memory_budget_gb",
        },
        where="resources",
    )
    _positive_int(resources["parse_workers_per_slot"], where="parse_workers_per_slot")
    _positive_int(resources["cpu_budget_vcpus"], where="cpu_budget_vcpus")
    for field in ("memory_limit_gb_per_slot", "memory_budget_gb"):
        raw = resources[field]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or float(raw) <= 0
        ):
            raise MonitorError(f"{field} must be positive")
    for field in (
        "state_path",
        "report_path",
        "terminal_receipt_path",
        "diagnostics_dir",
    ):
        _path(value[field], where=field)
    diagnostics_prefix = validate_gcs_uri(
        value["diagnostics_upload_prefix"], where="diagnostics_upload_prefix"
    )
    if not diagnostics_prefix.startswith(f"{run_root}/diagnostics/"):
        raise MonitorError(
            "diagnostics_upload_prefix escaped the run diagnostics namespace"
        )
    _positive_int(value["stale_after_seconds"], where="stale_after_seconds")
    require_nonempty(value["gcloud"], where="gcloud")
    value["physical_workers"] = workers
    value["resources"] = dict(resources)
    return value


def load_config(path: Path) -> dict[str, object]:
    _raw, value = load_json_object(path, where="GCP source monitor config")
    return validate_config(value)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="ascii") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MonitorError("another GCP source monitor is already running") from exc
        yield


def _empty_state(run_id: str) -> dict[str, object]:
    return {
        "schema": STATE_SCHEMA,
        "run_id": run_id,
        "validated_receipts": {},
        "workers": {},
        "diagnostics": {},
    }


def _load_state(path: Path, *, run_id: str) -> dict[str, object]:
    if not path.exists():
        return _empty_state(run_id)
    _raw, state = load_json_object(path, where="GCP source monitor state")
    if state.get("schema") != STATE_SCHEMA or state.get("run_id") != run_id:
        raise MonitorError("GCP source monitor state binding drifted")
    for field in ("validated_receipts", "workers", "diagnostics"):
        if not isinstance(state.get(field), Mapping):
            raise MonitorError(f"GCP source monitor state {field} is invalid")
        state[field] = dict(state[field])
    return state


def _raw_manifest(path: Path, expected_sha256: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise MonitorError(f"manifest_path must be a regular file: {path}")
    if sha256_file(path) != expected_sha256:
        raise MonitorError("manifest file SHA-256 drifted")
    _raw, manifest = load_json_object(path, where="GCP source manifest")
    return validate_source_manifest(manifest)


def _metadata_map(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        uri = validate_gcs_uri(row.get("uri"), where="inventory URI")
        if uri in result:
            raise MonitorError(f"duplicate inventory URI: {uri}")
        result[uri] = dict(row)
    return result


def _require_immutable_inventory(
    *,
    kind: str,
    inventory: Mapping[str, Mapping[str, object]],
    state: Mapping[str, object],
) -> None:
    cache = state["validated_receipts"]
    assert isinstance(cache, Mapping)
    missing = sorted(
        uri
        for uri, entry in cache.items()
        if isinstance(uri, str)
        and isinstance(entry, Mapping)
        and entry.get("kind") == kind
        and uri not in inventory
    )
    if missing:
        raise MonitorError(
            f"previously validated {kind} receipt disappeared: {missing[0]}"
        )


def _gcs_pattern_matches(uri: str, pattern: str) -> bool:
    sentinel = "\x00"
    expression = re.escape(pattern)
    expression = expression.replace(r"\*\*", sentinel)
    expression = expression.replace(r"\*", "[^/]*")
    expression = expression.replace(sentinel, ".*")
    return re.fullmatch(expression, uri) is not None


def _read_json_rows(
    client: RunClient, metadata_rows: Sequence[Mapping[str, object]]
) -> list[tuple[bytes, dict[str, object]]]:
    """Read uncached JSON receipts in bounded batches when the client supports it."""
    rows = list(metadata_rows)
    if not rows:
        return []
    batch_reader = getattr(client, "read_json_many", None)
    if callable(batch_reader):
        result: list[tuple[bytes, dict[str, object]]] = []
        for offset in range(0, len(rows), _JSON_READ_BATCH_SIZE):
            batch = rows[offset : offset + _JSON_READ_BATCH_SIZE]
            values = batch_reader(batch)
            if len(values) != len(batch):
                raise MonitorError(
                    "GCS JSON batch reader returned an unexpected object count"
                )
            result.extend(values)
        return result
    return [client.read_json(metadata) for metadata in rows]


def _control_receipt(
    *,
    kind: str,
    raw: bytes,
    value: Mapping[str, object],
    metadata: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    receipt = dict(value)
    common = {"schema_version", "state", "worker_name", "boot_id", "created_at"}
    if kind == "ready":
        expected = common | {"run_id", "local_ssd_count", "local_stage_bytes"}
    elif kind == "failed":
        expected = common | {"worker", "exit_code"}
    elif kind == "completed":
        expected = common | {
            "worker",
            "manifest_file_sha256",
            "receipt_count",
            "slots_per_worker",
            "logical_worker_count",
            "completed_slots",
            "resumed_slots",
        }
    else:  # pragma: no cover - internal dispatch
        raise AssertionError(kind)
    require_exact_fields(receipt, expected, where=f"GCP {kind} control receipt")
    expected_state = "complete" if kind == "completed" else kind
    if receipt["schema_version"] != 1 or receipt["state"] != expected_state:
        raise MonitorError(f"GCP {kind} control receipt schema/state drifted")
    worker_name = require_nonempty(receipt["worker_name"], where="control worker_name")
    workers = config["physical_workers"]
    assert isinstance(workers, list)
    if worker_name not in workers:
        raise MonitorError(f"control receipt names an unexpected worker: {worker_name}")
    boot_id = require_nonempty(receipt["boot_id"], where="control boot_id")
    if _UUID_RE.fullmatch(boot_id) is None:
        raise MonitorError("control receipt boot_id is invalid")
    require_nonempty(receipt["created_at"], where="control created_at")
    worker_index = workers.index(worker_name)
    if kind == "ready":
        if receipt["run_id"] != config["run_id"]:
            raise MonitorError("ready receipt run_id drifted")
        if receipt["local_ssd_count"] != config["expected_local_ssd_count"]:
            raise MonitorError("ready receipt Local SSD count drifted")
        _positive_int(receipt["local_ssd_count"], where="ready local_ssd_count")
        _positive_int(receipt["local_stage_bytes"], where="ready local_stage_bytes")
    else:
        if receipt["worker"] != f"worker-{worker_index:04d}":
            raise MonitorError(f"{kind} receipt physical worker identity drifted")
    if kind == "failed":
        exit_code = receipt["exit_code"]
        if (
            isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or exit_code < 1
        ):
            raise MonitorError("failed receipt exit_code is invalid")
    if kind == "completed":
        slots = int(config["slots_per_worker"])
        expected_slots = [
            f"worker-{worker_index * slots + slot:04d}" for slot in range(slots)
        ]
        completed_slots = receipt["completed_slots"]
        resumed_slots = receipt["resumed_slots"]
        if (
            receipt["manifest_file_sha256"] != config["manifest_file_sha256"]
            or receipt["slots_per_worker"] != slots
            or receipt["logical_worker_count"] != len(workers) * slots
            or not isinstance(completed_slots, list)
            or completed_slots != expected_slots
            or not isinstance(resumed_slots, list)
            or any(slot not in expected_slots for slot in resumed_slots)
        ):
            raise MonitorError("completed control receipt binding drifted")
        _positive_int(receipt["receipt_count"], where="completed receipt_count")
    return {
        **receipt,
        "uri": metadata["uri"],
        "generation": metadata["generation"],
        "updated": metadata.get("updated", ""),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _assignment_claim(
    *,
    raw: bytes,
    value: Mapping[str, object],
    metadata: Mapping[str, object],
    config: Mapping[str, object],
    manifest: Mapping[str, object],
    jobs_by_sha256: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    claim = dict(value)
    require_exact_fields(
        claim,
        {
            "schema",
            "status",
            "manifest_sha256",
            "manifest_file_sha256",
            "assignment",
            "attempt",
            "executor",
            "scheduler_instance",
            "created_unix_s",
            "expires_unix_s",
            "lease_seconds",
            "heartbeat_seconds",
            "training_ready",
        },
        where="GCP source assignment claim",
    )
    if (
        claim["schema"] != ASSIGNMENT_CLAIM_SCHEMA
        or claim["status"] != "claimed"
        or claim["manifest_sha256"] != manifest["manifest_sha256"]
        or claim["manifest_file_sha256"] != config["manifest_file_sha256"]
        or claim["training_ready"] is not False
    ):
        raise MonitorError("GCP source assignment claim binding drifted")
    assignment = claim["assignment"]
    if not isinstance(assignment, Mapping):
        raise MonitorError("GCP source assignment claim assignment is invalid")
    assignment_sha256 = require_sha256(
        assignment.get("assignment_sha256"),
        where="GCP source assignment claim assignment SHA-256",
    )
    job = jobs_by_sha256.get(assignment_sha256)
    expected_assignment = (
        {
            key: job[key]
            for key in ("ordinal", "repo", "project_id", "worker", "assignment_sha256")
        }
        if job is not None
        else None
    )
    if expected_assignment is None or dict(assignment) != expected_assignment:
        raise MonitorError("GCP source assignment claim escaped the manifest")
    attempt = require_int(claim["attempt"], where="claim attempt")
    if attempt > 9_999:
        raise MonitorError("GCP source assignment claim attempt exceeds its bound")
    prefix = (
        f"{config['run_root']}/source-assignment-claims/"
        f"{manifest['manifest_sha256']}/"
    )
    uri = validate_gcs_uri(metadata.get("uri"), where="assignment claim URI")
    relative = uri[len(prefix) :] if uri.startswith(prefix) else ""
    parts = relative.split("/")
    filename_match = _CLAIM_FILENAME_RE.fullmatch(parts[1]) if len(parts) == 2 else None
    if (
        len(parts) != 2
        or parts[0] != assignment_sha256
        or filename_match is None
        or int(filename_match.group(1)) != attempt
    ):
        raise MonitorError("GCP source assignment claim URI binding drifted")
    executor = claim["executor"]
    if not isinstance(executor, Mapping):
        raise MonitorError("GCP source assignment claim executor is invalid")
    executor = dict(executor)
    require_exact_fields(
        executor,
        {
            "physical_worker_index",
            "physical_worker_count",
            "slots_per_worker",
            "slot_index",
            "worker",
        },
        where="GCP source assignment claim executor",
    )
    workers = config["physical_workers"]
    assert isinstance(workers, list)
    slots = int(config["slots_per_worker"])
    physical_index = require_int(
        executor["physical_worker_index"], where="claim physical worker index"
    )
    slot_index = require_int(executor["slot_index"], where="claim slot index")
    if (
        physical_index >= len(workers)
        or slot_index >= slots
        or executor["physical_worker_count"] != len(workers)
        or executor["slots_per_worker"] != slots
        or executor["worker"] != f"worker-{physical_index * slots + slot_index:04d}"
    ):
        raise MonitorError("GCP source assignment claim executor topology drifted")
    scheduler_instance = require_nonempty(
        claim["scheduler_instance"], where="claim scheduler instance"
    )
    if len(scheduler_instance) > 256 or not scheduler_instance.isascii():
        raise MonitorError("GCP source assignment claim scheduler instance is invalid")
    created = require_int(
        claim["created_unix_s"], where="claim creation time", minimum=1
    )
    expires = require_int(claim["expires_unix_s"], where="claim expiry time", minimum=1)
    lease = require_int(claim["lease_seconds"], where="claim lease", minimum=1)
    heartbeat = require_int(
        claim["heartbeat_seconds"], where="claim heartbeat", minimum=1
    )
    if heartbeat >= lease or expires != created + lease:
        raise MonitorError("GCP source assignment claim lease drifted")
    return {
        "assignment_sha256": assignment_sha256,
        "attempt": attempt,
        "claim_sha256": canonical_sha256(claim),
        "physical_worker_index": physical_index,
        "logical_worker": executor["worker"],
        "executor": executor,
        "scheduler_instance": scheduler_instance,
        "created_unix_s": created,
        "expires_unix_s": expires,
        "lease_seconds": lease,
        "heartbeat_seconds": heartbeat,
        "uri": uri,
        "generation": str(metadata["generation"]),
        "updated": metadata.get("updated", ""),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _cached_claim_summary(
    cached: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    config: Mapping[str, object],
    manifest: Mapping[str, object],
    jobs_by_sha256: Mapping[str, Mapping[str, object]],
) -> dict[str, object] | None:
    summary = cached.get("summary")
    if not isinstance(summary, Mapping):
        return None
    value = dict(summary)
    expected = {
        "assignment_sha256",
        "attempt",
        "claim_sha256",
        "physical_worker_index",
        "logical_worker",
        "executor",
        "scheduler_instance",
        "created_unix_s",
        "expires_unix_s",
        "lease_seconds",
        "heartbeat_seconds",
    }
    if set(value) != expected:
        return None
    try:
        assignment_sha256 = require_sha256(
            value["assignment_sha256"], where="cached claim assignment"
        )
        attempt = require_int(value["attempt"], where="cached claim attempt")
        job = jobs_by_sha256.get(assignment_sha256)
        if job is None or attempt > 9_999:
            return None
        claim_sha256 = require_sha256(
            value["claim_sha256"], where="cached canonical claim SHA-256"
        )
        physical_index = require_int(
            value["physical_worker_index"], where="cached claim physical worker"
        )
        logical_worker = require_nonempty(
            value["logical_worker"], where="cached claim logical worker"
        )
        if not isinstance(value["executor"], Mapping):
            return None
        executor = dict(value["executor"])
        require_exact_fields(
            executor,
            {
                "physical_worker_index",
                "physical_worker_count",
                "slots_per_worker",
                "slot_index",
                "worker",
            },
            where="cached claim executor",
        )
        scheduler_instance = require_nonempty(
            value["scheduler_instance"], where="cached claim scheduler instance"
        )
        created = require_int(
            value["created_unix_s"], where="cached claim creation", minimum=1
        )
        expires = require_int(
            value["expires_unix_s"], where="cached claim expiry", minimum=1
        )
        lease_seconds = require_int(
            value["lease_seconds"], where="cached claim lease", minimum=1
        )
        heartbeat_seconds = require_int(
            value["heartbeat_seconds"], where="cached claim heartbeat", minimum=1
        )
        workers = config["physical_workers"]
        assert isinstance(workers, list)
        slots = int(config["slots_per_worker"])
        slot_index = require_int(executor["slot_index"], where="cached claim slot")
        expected_logical_worker = f"worker-{physical_index * slots + slot_index:04d}"
        prefix = (
            f"{config['run_root']}/source-assignment-claims/"
            f"{manifest['manifest_sha256']}/"
        )
        uri = validate_gcs_uri(metadata.get("uri"), where="cached claim URI")
        relative = uri[len(prefix) :] if uri.startswith(prefix) else ""
        parts = relative.split("/")
        filename_match = (
            _CLAIM_FILENAME_RE.fullmatch(parts[1]) if len(parts) == 2 else None
        )
        if (
            physical_index >= len(workers)
            or slot_index >= slots
            or executor["physical_worker_index"] != physical_index
            or executor["physical_worker_count"] != len(workers)
            or executor["slots_per_worker"] != slots
            or executor["worker"] != logical_worker
            or logical_worker != expected_logical_worker
            or len(scheduler_instance) > 256
            or not scheduler_instance.isascii()
            or heartbeat_seconds >= lease_seconds
            or expires != created + lease_seconds
            or len(parts) != 2
            or parts[0] != assignment_sha256
            or filename_match is None
            or int(filename_match.group(1)) != attempt
        ):
            return None
        expected_claim = {
            "schema": ASSIGNMENT_CLAIM_SCHEMA,
            "status": "claimed",
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
            "attempt": attempt,
            "executor": executor,
            "scheduler_instance": scheduler_instance,
            "created_unix_s": created,
            "expires_unix_s": expires,
            "lease_seconds": lease_seconds,
            "heartbeat_seconds": heartbeat_seconds,
            "training_ready": False,
        }
        if claim_sha256 != canonical_sha256(expected_claim):
            return None
    except ContractError:
        return None
    return {
        **value,
        "executor": executor,
        "uri": metadata["uri"],
        "generation": str(metadata["generation"]),
        "updated": metadata.get("updated", ""),
        "sha256": cached["sha256"],
    }


def _heartbeat_binding(
    *,
    assignment_sha256: str,
    attempt: int,
    claim_sha256: str,
    heartbeat_index: int,
    metadata: Mapping[str, object],
    config: Mapping[str, object],
    manifest: Mapping[str, object],
    claims_by_identity: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> tuple[Mapping[str, object], int, int, str]:
    claim = claims_by_identity.get((assignment_sha256, attempt, claim_sha256))
    if claim is None:
        raise MonitorError("GCP source assignment heartbeat has no exact claim")
    scheduled = int(claim["created_unix_s"]) + heartbeat_index * int(
        claim["heartbeat_seconds"]
    )
    lease_through = scheduled + int(claim["lease_seconds"])
    prefix = (
        f"{config['run_root']}/source-assignment-heartbeats/"
        f"{manifest['manifest_sha256']}/"
    )
    uri = validate_gcs_uri(metadata.get("uri"), where="assignment heartbeat URI")
    relative = uri[len(prefix) :] if uri.startswith(prefix) else ""
    parts = relative.split("/")
    claim_dir_match = (
        _HEARTBEAT_CLAIM_DIR_RE.fullmatch(parts[1]) if len(parts) == 3 else None
    )
    filename_match = (
        _HEARTBEAT_FILENAME_RE.fullmatch(parts[2]) if len(parts) == 3 else None
    )
    if (
        len(parts) != 3
        or parts[0] != assignment_sha256
        or claim_dir_match is None
        or int(claim_dir_match.group(1)) != attempt
        or claim_dir_match.group(2) != claim_sha256
        or filename_match is None
        or int(filename_match.group(1)) != heartbeat_index
    ):
        raise MonitorError("GCP source assignment heartbeat URI binding drifted")
    return claim, scheduled, lease_through, uri


def _assignment_heartbeat(
    *,
    raw: bytes,
    value: Mapping[str, object],
    metadata: Mapping[str, object],
    config: Mapping[str, object],
    manifest: Mapping[str, object],
    claims_by_identity: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> dict[str, object]:
    heartbeat = dict(value)
    require_exact_fields(
        heartbeat,
        {
            "schema",
            "status",
            "manifest_sha256",
            "assignment_sha256",
            "attempt",
            "claim_sha256",
            "executor",
            "scheduler_instance",
            "heartbeat_index",
            "scheduled_unix_s",
            "lease_through_unix_s",
            "training_ready",
        },
        where="GCP source assignment heartbeat",
    )
    assignment_sha256 = require_sha256(
        heartbeat["assignment_sha256"],
        where="GCP source assignment heartbeat assignment SHA-256",
    )
    claim_sha256 = require_sha256(
        heartbeat["claim_sha256"],
        where="GCP source assignment heartbeat claim SHA-256",
    )
    attempt = require_int(heartbeat["attempt"], where="heartbeat attempt")
    heartbeat_index = require_int(
        heartbeat["heartbeat_index"], where="heartbeat index", minimum=1
    )
    claim, scheduled, lease_through, uri = _heartbeat_binding(
        assignment_sha256=assignment_sha256,
        attempt=attempt,
        claim_sha256=claim_sha256,
        heartbeat_index=heartbeat_index,
        metadata=metadata,
        config=config,
        manifest=manifest,
        claims_by_identity=claims_by_identity,
    )
    if (
        heartbeat["schema"] != ASSIGNMENT_HEARTBEAT_SCHEMA
        or heartbeat["status"] != "active"
        or heartbeat["manifest_sha256"] != manifest["manifest_sha256"]
        or heartbeat["executor"] != claim["executor"]
        or heartbeat["scheduler_instance"] != claim["scheduler_instance"]
        or heartbeat["scheduled_unix_s"] != scheduled
        or heartbeat["lease_through_unix_s"] != lease_through
        or heartbeat["training_ready"] is not False
    ):
        raise MonitorError("GCP source assignment heartbeat binding drifted")
    return {
        "assignment_sha256": assignment_sha256,
        "attempt": attempt,
        "claim_sha256": claim_sha256,
        "physical_worker_index": claim["physical_worker_index"],
        "logical_worker": claim["logical_worker"],
        "heartbeat_index": heartbeat_index,
        "scheduled_unix_s": scheduled,
        "lease_through_unix_s": lease_through,
        "uri": uri,
        "generation": str(metadata["generation"]),
        "updated": metadata.get("updated", ""),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _cached_heartbeat_summary(
    cached: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    config: Mapping[str, object],
    manifest: Mapping[str, object],
    claims_by_identity: Mapping[tuple[str, int, str], Mapping[str, object]],
) -> dict[str, object] | None:
    summary = cached.get("summary")
    if not isinstance(summary, Mapping):
        return None
    value = dict(summary)
    expected = {
        "assignment_sha256",
        "attempt",
        "claim_sha256",
        "physical_worker_index",
        "logical_worker",
        "heartbeat_index",
        "scheduled_unix_s",
        "lease_through_unix_s",
    }
    if set(value) != expected:
        return None
    try:
        assignment_sha256 = require_sha256(
            value["assignment_sha256"], where="cached heartbeat assignment"
        )
        attempt = require_int(value["attempt"], where="cached heartbeat attempt")
        claim_sha256 = require_sha256(
            value["claim_sha256"], where="cached heartbeat claim"
        )
        physical_worker_index = require_int(
            value["physical_worker_index"], where="cached heartbeat physical worker"
        )
        logical_worker = require_nonempty(
            value["logical_worker"], where="cached heartbeat logical worker"
        )
        heartbeat_index = require_int(
            value["heartbeat_index"], where="cached heartbeat index", minimum=1
        )
        scheduled_unix_s = require_int(
            value["scheduled_unix_s"], where="cached heartbeat schedule", minimum=1
        )
        lease_through_unix_s = require_int(
            value["lease_through_unix_s"],
            where="cached heartbeat lease-through",
            minimum=1,
        )
        claim, scheduled, lease_through, uri = _heartbeat_binding(
            assignment_sha256=assignment_sha256,
            attempt=attempt,
            claim_sha256=claim_sha256,
            heartbeat_index=heartbeat_index,
            metadata=metadata,
            config=config,
            manifest=manifest,
            claims_by_identity=claims_by_identity,
        )
        if (
            physical_worker_index != claim["physical_worker_index"]
            or logical_worker != claim["logical_worker"]
            or scheduled_unix_s != scheduled
            or lease_through_unix_s != lease_through
        ):
            return None
    except ContractError:
        return None
    return {
        **value,
        "uri": uri,
        "generation": str(metadata["generation"]),
        "updated": metadata.get("updated", ""),
        "sha256": cached["sha256"],
    }


def _latest(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    if not rows:
        return None
    return max(
        rows, key=lambda row: (str(row.get("updated", "")), int(row["generation"]))
    )


def _event_key(row: Mapping[str, object] | None) -> tuple[str, int]:
    if row is None:
        return ("", 0)
    return (str(row.get("updated", "")), int(row["generation"]))


def _cached_receipt(
    *,
    kind: str,
    metadata: Mapping[str, object],
    state: dict[str, object],
) -> dict[str, object] | None:
    cache = state["validated_receipts"]
    assert isinstance(cache, dict)
    raw = cache.get(str(metadata["uri"]))
    if not isinstance(raw, Mapping):
        return None
    if raw.get("kind") != kind:
        return None
    if str(raw.get("generation")) != str(metadata["generation"]):
        raise MonitorError(
            f"immutable {kind} receipt generation drifted: {metadata['uri']}"
        )
    if raw.get("size_bytes") != metadata.get("size_bytes"):
        raise MonitorError(
            f"immutable {kind} receipt size drifted: {metadata['uri']}"
        )
    try:
        require_sha256(raw.get("sha256"), where=f"cached {kind} SHA-256")
    except ContractError:
        return None
    return dict(raw)


def _remember_receipt(
    *,
    kind: str,
    metadata: Mapping[str, object],
    raw: bytes,
    state: dict[str, object],
) -> dict[str, object]:
    entry = {
        "kind": kind,
        "generation": str(metadata["generation"]),
        "size_bytes": int(metadata["size_bytes"]),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    cache = state["validated_receipts"]
    assert isinstance(cache, dict)
    cache[str(metadata["uri"])] = entry
    return entry


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise MonitorError(f"immutable local receipt collision: {path}")
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_readback_verified(
    object_store: ObjectStore,
    source: Path,
    uri: str,
) -> Mapping[str, object]:
    metadata = dict(object_store.publish_if_absent(source, uri))
    generation = str(metadata.get("generation", ""))
    if not generation.isdecimal() or int(generation) < 1:
        raise MonitorError(f"published GCS object has an invalid generation: {uri}")
    downloader = getattr(object_store, "download", None)
    if not callable(downloader):
        raise MonitorError(f"object store cannot read back published object: {uri}")
    with tempfile.TemporaryDirectory(prefix="cppmega-gcp-diagnostics-readback-") as raw:
        destination = Path(raw) / "object"
        readback_metadata = downloader(uri, destination, generation=generation)
        if not destination.is_file():
            raise MonitorError(f"published GCS object read-back is missing: {uri}")
        if sha256_file(destination) != sha256_file(source):
            raise MonitorError(f"published GCS object bytes drifted: {uri}")
        if destination.stat().st_size != source.stat().st_size:
            raise MonitorError(f"published GCS object size drifted: {uri}")
        if (
            not isinstance(readback_metadata, Mapping)
            or str(readback_metadata.get("generation", "")) != generation
        ):
            raise MonitorError(f"published GCS object generation drifted: {uri}")
    return metadata


def _preserve_diagnostics(
    *,
    failure: Mapping[str, object],
    config: Mapping[str, object],
    state: dict[str, object],
    client: RunClient,
    object_store: ObjectStore,
    zone: str,
    now: int,
) -> dict[str, object]:
    fingerprint = hashlib.sha256(
        f"{failure['uri']}#{failure['generation']}:{failure['sha256']}".encode("ascii")
    ).hexdigest()
    diagnostics = state["diagnostics"]
    assert isinstance(diagnostics, dict)
    worker = str(failure["worker_name"])
    local_root = (
        _path(config["diagnostics_dir"], where="diagnostics_dir") / worker / fingerprint
    )
    remote_root = gcs_join(
        str(config["diagnostics_upload_prefix"]),
        str(failure["boot_id"]),
        fingerprint,
    )
    receipt_path = local_root / f"{fingerprint}.diagnostics.json"
    receipt_uri = gcs_join(remote_root, f"{fingerprint}.diagnostics.json")
    if receipt_path.exists():
        receipt_bytes, receipt = load_json_object(
            receipt_path, where="local GCP failure diagnostics receipt"
        )
        require_exact_fields(
            receipt,
            {
                "schema",
                "status",
                "run_id",
                "worker_name",
                "boot_id",
                "failure",
                "serial",
                "captured_at_unix",
                "confirmed_http_429",
                "training_ready",
            },
            where="local GCP failure diagnostics receipt",
        )
        expected_failure = {
            key: failure[key]
            for key in ("uri", "generation", "sha256", "exit_code", "created_at")
        }
        serial_receipt = receipt["serial"]
        if (
            receipt["schema"] != DIAGNOSTICS_SCHEMA
            or receipt["status"] != "published"
            or receipt["run_id"] != config["run_id"]
            or receipt["worker_name"] != worker
            or receipt["boot_id"] != failure["boot_id"]
            or receipt["failure"] != expected_failure
            or not isinstance(serial_receipt, Mapping)
            or receipt["training_ready"] is not False
        ):
            raise MonitorError("local GCP failure diagnostics receipt binding drifted")
        require_exact_fields(
            serial_receipt,
            {"uri", "generation", "size_bytes", "sha256"},
            where="local GCP failure serial receipt",
        )
        serial_sha256 = require_sha256(
            serial_receipt["sha256"], where="local GCP failure serial SHA-256"
        )
        serial_path = local_root / f"{serial_sha256}.serial.log"
        if (
            serial_path.is_symlink()
            or not serial_path.is_file()
            or serial_path.stat().st_size != serial_receipt["size_bytes"]
            or sha256_file(serial_path) != serial_sha256
            or serial_receipt["uri"]
            != gcs_join(remote_root, f"{serial_sha256}.serial.log")
        ):
            raise MonitorError("local GCP failure serial diagnostics drifted")
        serial_metadata = dict(
            _publish_readback_verified(
                object_store, serial_path, str(serial_receipt["uri"])
            )
        )
        if str(serial_metadata.get("generation")) != str(serial_receipt["generation"]):
            raise MonitorError("GCP failure serial diagnostics generation drifted")
        receipt_metadata = dict(
            _publish_readback_verified(object_store, receipt_path, receipt_uri)
        )
        result = {
            "status": "published",
            "fingerprint": fingerprint,
            "local_path": str(receipt_path),
            "uri": receipt_uri,
            "generation": str(receipt_metadata["generation"]),
            "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "confirmed_http_429": receipt["confirmed_http_429"],
        }
        diagnostics[fingerprint] = result
        return result
    serial = client.serial_output(
        project_id=str(config["project_id"]),
        zone=zone,
        instance=worker,
    )
    serial_sha256 = hashlib.sha256(serial).hexdigest()
    serial_path = local_root / f"{serial_sha256}.serial.log"
    _write_immutable(serial_path, serial)
    serial_uri = gcs_join(remote_root, f"{serial_sha256}.serial.log")
    serial_metadata = dict(
        _publish_readback_verified(object_store, serial_path, serial_uri)
    )
    receipt: dict[str, object] = {
        "schema": DIAGNOSTICS_SCHEMA,
        "status": "published",
        "run_id": config["run_id"],
        "worker_name": worker,
        "boot_id": failure["boot_id"],
        "failure": {
            key: failure[key]
            for key in ("uri", "generation", "sha256", "exit_code", "created_at")
        },
        "serial": {
            "uri": serial_uri,
            "generation": str(serial_metadata["generation"]),
            "size_bytes": len(serial),
            "sha256": serial_sha256,
        },
        "captured_at_unix": now,
        "confirmed_http_429": bool(
            _HTTP_429_RE.search(serial.decode("utf-8", errors="replace"))
        ),
        "training_ready": False,
    }
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    _write_immutable(receipt_path, receipt_bytes)
    receipt_metadata = dict(
        _publish_readback_verified(object_store, receipt_path, receipt_uri)
    )
    result = {
        "status": "published",
        "fingerprint": fingerprint,
        "local_path": str(receipt_path),
        "uri": receipt_uri,
        "generation": str(receipt_metadata["generation"]),
        "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "confirmed_http_429": receipt["confirmed_http_429"],
    }
    diagnostics[fingerprint] = result
    return result


def _terminal_receipt(
    *,
    path: Path,
    config: Mapping[str, object],
    manifest: Mapping[str, object],
    report: Mapping[str, object],
    now: int,
) -> dict[str, object]:
    if path.exists():
        _raw, existing = load_json_object(path, where="GCP source terminal receipt")
        if (
            existing.get("schema") != TERMINAL_SCHEMA
            or existing.get("status") != "verified"
            or existing.get("run_id") != config["run_id"]
            or existing.get("manifest_sha256") != manifest["manifest_sha256"]
            or existing.get("training_ready") is not False
        ):
            raise MonitorError("GCP source terminal receipt binding drifted")
        return existing
    receipt = {
        "schema": TERMINAL_SCHEMA,
        "status": "verified",
        "run_id": config["run_id"],
        "run_root": config["run_root"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": config["manifest_file_sha256"],
        "verified_at_unix": now,
        "counts": report["counts"],
        "receipt_inventory_sha256": report["receipt_inventory_sha256"],
        "training_ready": False,
    }
    _write_immutable(path, canonical_json_bytes(receipt) + b"\n")
    return receipt


def run_monitor(
    config: Mapping[str, object],
    *,
    client: RunClient | None = None,
    object_store: ObjectStore | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, object]:
    """Inspect one run and publish no recovery action."""

    checked = validate_config(config)
    run_id = str(checked["run_id"])
    checked_at = int(now())
    state_path = _path(checked["state_path"], where="state_path")
    report_path = _path(checked["report_path"], where="report_path")
    lock_path = state_path.with_name(f".{state_path.name}.lock")
    run_client = client or GcloudRunClient(str(checked["gcloud"]))
    store = object_store or GcloudObjectStore(str(checked["gcloud"]))
    with _exclusive_lock(lock_path):
        state = _load_state(state_path, run_id=run_id)
        manifest = _raw_manifest(
            _path(checked["manifest_path"], where="manifest_path"),
            str(checked["manifest_file_sha256"]),
        )
        if manifest["gcs_output_prefix"] != checked["run_root"]:
            raise MonitorError("manifest output prefix does not match run_root")
        physical_workers = checked["physical_workers"]
        assert isinstance(physical_workers, list)
        slots_per_worker = int(checked["slots_per_worker"])
        logical_specs = tuple(
            spec
            for physical_index in range(len(physical_workers))
            for spec in slot_specs(
                physical_worker_index=physical_index,
                physical_worker_count=len(physical_workers),
                slots_per_worker=slots_per_worker,
            )
        )
        if manifest["workers"] != [spec.worker for spec in logical_specs]:
            raise MonitorError(
                "manifest logical workers do not match configured topology"
            )

        controls: dict[str, list[dict[str, object]]] = {}
        for kind in ("ready", "failed", "completed"):
            rows: list[dict[str, object]] = []
            for metadata in run_client.list_objects(
                f"{checked['run_root']}/control/{kind}/*.json"
            ):
                raw, value = run_client.read_json(metadata)
                rows.append(
                    _control_receipt(
                        kind=kind,
                        raw=raw,
                        value=value,
                        metadata=metadata,
                        config=checked,
                    )
                )
            controls[kind] = rows

        jobs = manifest["repositories"]
        assert isinstance(jobs, list)
        jobs_by_sha256 = {str(job["assignment_sha256"]): job for job in jobs}
        claim_inventory = _metadata_map(
            run_client.list_objects(
                f"{checked['run_root']}/source-assignment-claims/"
                f"{manifest['manifest_sha256']}/*/*.claim.json"
            )
        )
        _require_immutable_inventory(
            kind="claim", inventory=claim_inventory, state=state
        )
        claim_records: list[dict[str, object]] = []
        claim_sha256: list[str] = []
        uncached_claim_metadata: list[Mapping[str, object]] = []
        claim_summary_fields = (
            "assignment_sha256",
            "attempt",
            "claim_sha256",
            "physical_worker_index",
            "logical_worker",
            "executor",
            "scheduler_instance",
            "created_unix_s",
            "expires_unix_s",
            "lease_seconds",
            "heartbeat_seconds",
        )
        for metadata in claim_inventory.values():
            cached = _cached_receipt(kind="claim", metadata=metadata, state=state)
            record = (
                _cached_claim_summary(
                    cached,
                    metadata,
                    config=checked,
                    manifest=manifest,
                    jobs_by_sha256=jobs_by_sha256,
                )
                if cached is not None
                else None
            )
            if record is None:
                uncached_claim_metadata.append(metadata)
                continue
            claim_records.append(record)
            claim_sha256.append(str(record["sha256"]))
        for metadata, (raw, value) in zip(
            uncached_claim_metadata,
            _read_json_rows(run_client, uncached_claim_metadata),
        ):
            record = _assignment_claim(
                raw=raw,
                value=value,
                metadata=metadata,
                config=checked,
                manifest=manifest,
                jobs_by_sha256=jobs_by_sha256,
            )
            cached = _remember_receipt(
                kind="claim", metadata=metadata, raw=raw, state=state
            )
            cached["summary"] = {key: record[key] for key in claim_summary_fields}
            claim_records.append(record)
            claim_sha256.append(str(record["sha256"]))
        latest_claims: dict[str, dict[str, object]] = {}
        for record in claim_records:
            assignment_sha256 = str(record["assignment_sha256"])
            previous = latest_claims.get(assignment_sha256)
            if previous is None or (
                int(record["attempt"]),
                str(record["updated"]),
                int(record["generation"]),
            ) > (
                int(previous["attempt"]),
                str(previous["updated"]),
                int(previous["generation"]),
            ):
                latest_claims[assignment_sha256] = record
        claims_by_identity = {
            (
                str(record["assignment_sha256"]),
                int(record["attempt"]),
                str(record["claim_sha256"]),
            ): record
            for record in claim_records
        }
        heartbeat_inventory = _metadata_map(
            run_client.list_objects(
                f"{checked['run_root']}/source-assignment-heartbeats/"
                f"{manifest['manifest_sha256']}/*/*/*.heartbeat.json"
            )
        )
        _require_immutable_inventory(
            kind="heartbeat", inventory=heartbeat_inventory, state=state
        )
        heartbeat_records: list[dict[str, object]] = []
        heartbeat_sha256: list[str] = []
        uncached_heartbeat_metadata: list[Mapping[str, object]] = []
        heartbeat_summary_fields = (
            "assignment_sha256",
            "attempt",
            "claim_sha256",
            "physical_worker_index",
            "logical_worker",
            "heartbeat_index",
            "scheduled_unix_s",
            "lease_through_unix_s",
        )
        for metadata in heartbeat_inventory.values():
            cached = _cached_receipt(kind="heartbeat", metadata=metadata, state=state)
            record = (
                _cached_heartbeat_summary(
                    cached,
                    metadata,
                    config=checked,
                    manifest=manifest,
                    claims_by_identity=claims_by_identity,
                )
                if cached is not None
                else None
            )
            if record is None:
                uncached_heartbeat_metadata.append(metadata)
                continue
            heartbeat_records.append(record)
            heartbeat_sha256.append(str(record["sha256"]))
        for metadata, (raw, value) in zip(
            uncached_heartbeat_metadata,
            _read_json_rows(run_client, uncached_heartbeat_metadata),
        ):
            record = _assignment_heartbeat(
                raw=raw,
                value=value,
                metadata=metadata,
                config=checked,
                manifest=manifest,
                claims_by_identity=claims_by_identity,
            )
            cached = _remember_receipt(
                kind="heartbeat", metadata=metadata, raw=raw, state=state
            )
            cached["summary"] = {key: record[key] for key in heartbeat_summary_fields}
            heartbeat_records.append(record)
            heartbeat_sha256.append(str(record["sha256"]))
        jobs_by_uri = {assignment_completion_uri(manifest, job): job for job in jobs}
        assignment_inventory = _metadata_map(
            run_client.list_objects(
                f"{checked['run_root']}/source-assignment-completions/"
                f"{manifest['manifest_sha256']}/*.complete.json"
            )
        )
        _require_immutable_inventory(
            kind="assignment", inventory=assignment_inventory, state=state
        )
        unexpected_assignments = sorted(set(assignment_inventory) - set(jobs_by_uri))
        if unexpected_assignments:
            raise MonitorError(
                f"unexpected assignment completion receipt: {unexpected_assignments[0]}"
            )
        valid_assignment_uris: set[str] = set()
        assignment_sha256: list[str] = []
        uncached_assignment_items: list[tuple[str, Mapping[str, object]]] = []
        for uri, metadata in assignment_inventory.items():
            cached = _cached_receipt(kind="assignment", metadata=metadata, state=state)
            if cached is None:
                uncached_assignment_items.append((uri, metadata))
                continue
            valid_assignment_uris.add(uri)
            assignment_sha256.append(str(cached["sha256"]))
        for (uri, metadata), (raw, value) in zip(
            uncached_assignment_items,
            _read_json_rows(
                run_client,
                [metadata for _uri, metadata in uncached_assignment_items],
            ),
        ):
            validate_assignment_completion_receipt(
                value,
                manifest=manifest,
                manifest_file_sha256=str(checked["manifest_file_sha256"]),
                job=jobs_by_uri[uri],
            )
            cached = _remember_receipt(
                kind="assignment", metadata=metadata, raw=raw, state=state
            )
            valid_assignment_uris.add(uri)
            assignment_sha256.append(str(cached["sha256"]))
        completed_assignment_sha256 = {
            str(jobs_by_uri[uri]["assignment_sha256"]) for uri in valid_assignment_uris
        }
        current_claims = {
            assignment_sha256_value: record
            for assignment_sha256_value, record in latest_claims.items()
            if assignment_sha256_value not in completed_assignment_sha256
        }
        current_heartbeat_records = [
            record
            for record in heartbeat_records
            if (latest_claim := current_claims.get(str(record["assignment_sha256"])))
            is not None
            and int(record["attempt"]) == int(latest_claim["attempt"])
            and str(record["claim_sha256"])
            == str(latest_claim["claim_sha256"])
            and int(record["scheduled_unix_s"]) <= checked_at
        ]
        latest_current_heartbeats: dict[str, dict[str, object]] = {}
        for record in current_heartbeat_records:
            assignment_sha256_value = str(record["assignment_sha256"])
            previous = latest_current_heartbeats.get(assignment_sha256_value)
            if previous is None or (
                int(record["heartbeat_index"]),
                str(record["updated"]),
                int(record["generation"]),
            ) > (
                int(previous["heartbeat_index"]),
                str(previous["updated"]),
                int(previous["generation"]),
            ):
                latest_current_heartbeats[assignment_sha256_value] = record
        fresh_current_heartbeats = {
            assignment_sha256_value: record
            for assignment_sha256_value, record in latest_current_heartbeats.items()
            if checked_at
            < max(
                int(record["lease_through_unix_s"]),
                int(record["scheduled_unix_s"])
                + int(checked["stale_after_seconds"]),
            )
        }

        specs_by_uri = {
            gcs_join(
                str(checked["run_root"]),
                "source-slot-receipts",
                str(manifest["manifest_sha256"]),
                f"{spec.worker}.complete.json",
            ): spec
            for spec in logical_specs
        }
        slot_inventory = _metadata_map(
            run_client.list_objects(
                f"{checked['run_root']}/source-slot-receipts/"
                f"{manifest['manifest_sha256']}/*.complete.json"
            )
        )
        _require_immutable_inventory(kind="slot", inventory=slot_inventory, state=state)
        unexpected_slots = sorted(set(slot_inventory) - set(specs_by_uri))
        if unexpected_slots:
            raise MonitorError(
                f"unexpected slot completion receipt: {unexpected_slots[0]}"
            )
        valid_slot_uris: set[str] = set()
        slot_sha256: list[str] = []
        resources = checked["resources"]
        assert isinstance(resources, Mapping)
        for uri, metadata in slot_inventory.items():
            cached = _cached_receipt(kind="slot", metadata=metadata, state=state)
            if cached is None:
                raw, value = run_client.read_json(metadata)
                validate_slot_completion_receipt(
                    value,
                    manifest=manifest,
                    manifest_file_sha256=str(checked["manifest_file_sha256"]),
                    spec=specs_by_uri[uri],
                    resources=resources,
                )
                cached = _remember_receipt(
                    kind="slot", metadata=metadata, raw=raw, state=state
                )
            valid_slot_uris.add(uri)
            slot_sha256.append(str(cached["sha256"]))

        instance_rows = run_client.list_instances(
            project_id=str(checked["project_id"]), run_id=run_id
        )
        instances: dict[str, dict[str, object]] = {}
        for row in instance_rows:
            name = require_nonempty(row.get("name"), where="instance name")
            if name in instances:
                raise MonitorError(f"duplicate instance: {name}")
            instances[name] = dict(row)
        unexpected_instances = sorted(set(instances) - set(physical_workers))

        ready_by_worker = {
            worker: [row for row in controls["ready"] if row["worker_name"] == worker]
            for worker in physical_workers
        }
        failed_by_worker = {
            worker: [row for row in controls["failed"] if row["worker_name"] == worker]
            for worker in physical_workers
        }
        completed_by_worker = {
            worker: [
                row for row in controls["completed"] if row["worker_name"] == worker
            ]
            for worker in physical_workers
        }
        worker_state = state["workers"]
        assert isinstance(worker_state, dict)
        worker_reports: list[dict[str, object]] = []
        for physical_index, worker in enumerate(physical_workers):
            owned_specs = logical_specs[
                physical_index
                * slots_per_worker : (physical_index + 1)
                * slots_per_worker
            ]
            owned_workers = {spec.worker for spec in owned_specs}
            completed_assignments = sum(
                assignment_completion_uri(manifest, job) in valid_assignment_uris
                for job in jobs
                if job["worker"] in owned_workers
            )
            expected_assignments = sum(job["worker"] in owned_workers for job in jobs)
            completed_slots = sum(
                uri in valid_slot_uris
                for uri, spec in specs_by_uri.items()
                if spec.worker in owned_workers
            )
            worker_claims = [
                record
                for record in claim_records
                if record["physical_worker_index"] == physical_index
            ]
            worker_latest_claims = [
                record
                for record in latest_claims.values()
                if record["physical_worker_index"] == physical_index
            ]
            worker_current_claims = [
                record
                for record in current_claims.values()
                if record["physical_worker_index"] == physical_index
            ]
            worker_heartbeats = [
                record
                for record in heartbeat_records
                if record["physical_worker_index"] == physical_index
            ]
            worker_current_heartbeats = [
                record
                for record in latest_current_heartbeats.values()
                if record["physical_worker_index"] == physical_index
            ]
            worker_fresh_heartbeats = [
                record
                for record in fresh_current_heartbeats.values()
                if record["physical_worker_index"] == physical_index
            ]
            completed_claimed_assignments = sum(
                str(record["assignment_sha256"]) in completed_assignment_sha256
                for record in worker_latest_claims
            )
            progress_events = (
                [
                    metadata
                    for uri, metadata in assignment_inventory.items()
                    if jobs_by_uri[uri]["worker"] in owned_workers
                ]
                + [
                    metadata
                    for uri, metadata in slot_inventory.items()
                    if specs_by_uri[uri].worker in owned_workers
                ]
                + worker_claims
                + worker_current_heartbeats
            )
            latest_progress_event = _latest(progress_events)
            signature = (
                f"{completed_assignments}/{expected_assignments}:"
                f"{completed_slots}/{slots_per_worker}:"
                f"{len(worker_claims)}/{len(worker_latest_claims)}/"
                f"{completed_claimed_assignments}"
            )
            prior = worker_state.get(worker)
            if (
                not isinstance(prior, Mapping)
                or prior.get("progress_signature") != signature
            ):
                progress_at = checked_at
            else:
                progress_at = require_int(
                    prior.get("progress_at_unix"),
                    where="worker progress time",
                    minimum=0,
                )
            if worker_current_heartbeats:
                progress_at = max(
                    progress_at,
                    max(
                        int(record["scheduled_unix_s"])
                        for record in worker_current_heartbeats
                    ),
                )
            latest_ready = _latest(ready_by_worker[worker])
            latest_failed = _latest(failed_by_worker[worker])
            latest_completed = _latest(completed_by_worker[worker])
            failure_matches_latest_boot = (
                latest_failed is not None
                and latest_ready is not None
                and latest_failed["boot_id"] == latest_ready["boot_id"]
            )
            progress_after_failure = failure_matches_latest_boot and _event_key(
                latest_progress_event
            ) > _event_key(latest_failed)
            active_failure = (
                latest_failed
                if failure_matches_latest_boot
                and _event_key(latest_failed) >= _event_key(latest_ready)
                and not progress_after_failure
                else None
            )
            instance = instances.get(worker)
            instance_status = (
                str(instance.get("status", "MISSING")) if instance else "MISSING"
            )
            prior_zone = prior.get("zone") if isinstance(prior, Mapping) else None
            instance_zone = _zone_name(
                (instance.get("zone") if instance else prior_zone or checked["zone"]),
                where=f"instance {worker} zone",
            )
            worker_state[worker] = {
                "progress_signature": signature,
                "progress_at_unix": progress_at,
                "zone": instance_zone,
            }
            report: dict[str, object] = {
                "name": worker,
                "instance_status": instance_status,
                "zone": instance_zone,
                "ready_receipts": len(ready_by_worker[worker]),
                "failed_receipts": len(failed_by_worker[worker]),
                "completed_receipts": len(completed_by_worker[worker]),
                "assignment_receipts": completed_assignments,
                "expected_assignments": expected_assignments,
                "assignment_accounting": "manifest_home_shard",
                "claim_receipts": len(worker_claims),
                "claimed_assignments": len(worker_latest_claims),
                "current_claimed_assignments": len(worker_current_claims),
                "completed_claimed_assignments": completed_claimed_assignments,
                "heartbeat_receipts": len(worker_heartbeats),
                "current_claim_heartbeat_receipts": sum(
                    record["physical_worker_index"] == physical_index
                    for record in current_heartbeat_records
                ),
                "fresh_heartbeat_assignments": len(worker_fresh_heartbeats),
                "fresh_assignment_heartbeats": [
                    {
                        "repo": jobs_by_sha256[str(record["assignment_sha256"])][
                            "repo"
                        ],
                        "assignment_sha256": record["assignment_sha256"],
                        "attempt": record["attempt"],
                        "logical_worker": record["logical_worker"],
                        "heartbeat_index": record["heartbeat_index"],
                        "scheduled_unix_s": record["scheduled_unix_s"],
                        "lease_through_unix_s": record["lease_through_unix_s"],
                    }
                    for record in sorted(
                        worker_fresh_heartbeats,
                        key=lambda item: str(item["assignment_sha256"]),
                    )
                ],
                "slot_receipts": completed_slots,
                "expected_slots": slots_per_worker,
                "last_progress_at_unix": progress_at,
                "replacement_permitted": False,
            }
            if latest_failed is not None and active_failure is None:
                report["superseded_failure"] = {
                    "boot_id": latest_failed["boot_id"],
                    "exit_code": latest_failed["exit_code"],
                    "reason": (
                        "later_progress"
                        if progress_after_failure
                        else "newer_ready_boot"
                    ),
                }
            if latest_completed is not None:
                if (
                    completed_assignments != expected_assignments
                    or completed_slots != slots_per_worker
                ):
                    report["state"] = "completed_control_missing_receipts"
                else:
                    report["state"] = "complete"
            elif active_failure is not None:
                exit_code = int(active_failure["exit_code"])
                report["exit_code"] = exit_code
                try:
                    diagnostics = _preserve_diagnostics(
                        failure=active_failure,
                        config=checked,
                        state=state,
                        client=run_client,
                        object_store=store,
                        zone=instance_zone,
                        now=checked_at,
                    )
                except (ContractError, OSError, RuntimeError, ValueError) as exc:
                    report["diagnostics_error"] = str(exc)
                    diagnostics = None
                if diagnostics is not None:
                    report["diagnostics"] = diagnostics
                if exit_code == DETERMINISTIC_EXIT_CODE:
                    report["state"] = "deterministic_failure_manual_review"
                    report["recovery_evidence"] = "exit_2"
                elif exit_code == TRANSIENT_EXIT_CODE:
                    report["state"] = (
                        "transient_failure_diagnostics_preserved"
                        if diagnostics is not None
                        else "transient_failure_recovery_blocked"
                    )
                    report["recovery_evidence"] = "exit_75"
                    report["replacement_permitted"] = diagnostics is not None
                elif (
                    diagnostics is not None
                    and diagnostics.get("confirmed_http_429") is True
                ):
                    report["state"] = "transient_failure_diagnostics_preserved"
                    report["recovery_evidence"] = "confirmed_http_429"
                    report["replacement_permitted"] = True
                else:
                    report["state"] = "unclassified_failure_manual_review"
            elif instance_status != "RUNNING":
                report["state"] = "instance_not_running"
            elif latest_ready is None:
                report["state"] = "awaiting_ready"
            elif progress_after_failure:
                report["state"] = "running_recovered_after_failure"
            elif (
                completed_assignments == expected_assignments
                and completed_slots == slots_per_worker
            ):
                report["state"] = "finalizing"
            elif claim_records and not worker_current_claims:
                report["state"] = "running"
            elif any(
                int(record["expires_unix_s"]) > checked_at
                for record in worker_current_claims
            ) or worker_fresh_heartbeats:
                report["state"] = "running"
            elif checked_at - progress_at >= int(checked["stale_after_seconds"]):
                report["state"] = "idle_suspected_manual_review"
            else:
                report["state"] = "running"
            worker_reports.append(report)

        counts = {
            "physical_workers": len(physical_workers),
            "ready_workers": sum(
                bool(ready_by_worker[worker]) for worker in physical_workers
            ),
            "failed_control_receipts": len(controls["failed"]),
            "completed_workers": sum(
                report["state"] == "complete" for report in worker_reports
            ),
            "assignment_receipts": len(valid_assignment_uris),
            "expected_assignment_receipts": len(jobs),
            "assignment_claim_receipts": len(claim_records),
            "claimed_assignments": len(latest_claims),
            "assignment_heartbeat_receipts": len(heartbeat_records),
            "fresh_heartbeat_assignments": len(fresh_current_heartbeats),
            "slot_receipts": len(valid_slot_uris),
            "expected_slot_receipts": len(logical_specs),
        }
        receipt_inventory_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "assignments": sorted(assignment_sha256),
                    "slots": sorted(slot_sha256),
                    "completed": sorted(
                        str(row["sha256"]) for row in controls["completed"]
                    ),
                }
            )
        ).hexdigest()
        claim_inventory_sha256 = hashlib.sha256(
            canonical_json_bytes(sorted(claim_sha256))
        ).hexdigest()
        heartbeat_inventory_sha256 = hashlib.sha256(
            canonical_json_bytes(sorted(heartbeat_sha256))
        ).hexdigest()
        if counts["completed_workers"] == len(physical_workers):
            run_state = "complete"
        elif any(
            report["state"] == "deterministic_failure_manual_review"
            for report in worker_reports
        ):
            run_state = "blocked_deterministic"
        elif any(
            report["state"] == "transient_failure_recovery_blocked"
            for report in worker_reports
        ):
            run_state = "recovery_blocked_diagnostics"
        elif any(
            report["state"] == "transient_failure_diagnostics_preserved"
            for report in worker_reports
        ):
            run_state = "recoverable_transient"
        elif any("manual_review" in str(report["state"]) for report in worker_reports):
            run_state = "manual_review"
        else:
            run_state = "running"
        report_payload: dict[str, object] = {
            "schema": REPORT_SCHEMA,
            "run_id": run_id,
            "run_root": checked["run_root"],
            "checked_at_unix": checked_at,
            "state": run_state,
            "counts": counts,
            "workers": worker_reports,
            "unexpected_instances": unexpected_instances,
            "receipt_inventory_sha256": receipt_inventory_sha256,
            "claim_inventory_sha256": claim_inventory_sha256,
            "heartbeat_inventory_sha256": heartbeat_inventory_sha256,
            "scheduler_mode": (
                "dynamic_claim_queue" if claim_records else "manifest_home_shards"
            ),
            "recovery_policy": {
                "transient_exit_code": TRANSIENT_EXIT_CODE,
                "deterministic_exit_code": DETERMINISTIC_EXIT_CODE,
                "diagnostics_required_before_replacement": True,
                "automatic_replacement_performed": False,
            },
            "training_ready": False,
        }
        if run_state == "complete":
            terminal_path = _path(
                checked["terminal_receipt_path"], where="terminal_receipt_path"
            )
            terminal = _terminal_receipt(
                path=terminal_path,
                config=checked,
                manifest=manifest,
                report=report_payload,
                now=checked_at,
            )
            report_payload["terminal_receipt"] = {
                "path": str(terminal_path),
                "status": terminal["status"],
                "sha256": sha256_file(terminal_path),
            }
        state["updated_at_unix"] = checked_at
        atomic_write_json(state_path, state)
        atomic_write_json(report_path, report_payload)
        return report_payload


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_monitor(load_config(args.config))
    except (MonitorError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"cppmega GCP source run monitor failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "DETERMINISTIC_EXIT_CODE",
    "DIAGNOSTICS_SCHEMA",
    "GcloudRunClient",
    "MONITOR_SCHEMA",
    "MonitorError",
    "REPORT_SCHEMA",
    "STATE_SCHEMA",
    "TERMINAL_SCHEMA",
    "TRANSIENT_EXIT_CODE",
    "load_config",
    "run_monitor",
    "validate_config",
]
