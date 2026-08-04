#!/usr/bin/env python3
"""Run independent source-manifest workers in bounded VM-local slots.

The source manifest still owns the logical assignment.  A VM only derives the
contiguous logical worker IDs assigned to it and starts one source worker per
ID.  Every slot has its own checkout, scratch directory, logs, and receipts;
the only shared inputs are immutable bootstrap files.  A slot becomes
resumable only after all of its source receipts have been read back and an
immutable slot completion receipt has been published to GCS.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    gcs_join,
    load_json_object,
    require_exact_fields,
    require_int,
    require_sha256,
    run_checked,
    sha256_file,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.source_manifest import (  # noqa: E402
    load_source_manifest,
    repositories_for_worker,
    validate_source_manifest,
)
from scripts.distributed_data_prep.source_worker import (  # noqa: E402
    GcloudObjectStore,
    ObjectStore,
    TransientTransportError,
    validate_worker_receipt,
)


SLOT_COMPLETION_RECEIPT_SCHEMA = (
    "cppmega.distributed_source_slot_completion_receipt_v1"
)
SCHEDULER_RECEIPT_SCHEMA = "cppmega.distributed_source_slot_scheduler_receipt_v1"


@dataclass(frozen=True)
class SlotSpec:
    physical_worker_index: int
    physical_worker_count: int
    slots_per_worker: int
    slot_index: int
    worker: str


def _positive_int(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(f"{where} must be a positive integer")
    return value


def _positive_float(value: object, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ContractError(f"{where} must be a positive finite number")
    return result


def logical_worker_count(physical_worker_count: int, slots_per_worker: int) -> int:
    physical = _positive_int(physical_worker_count, where="physical worker count")
    slots = _positive_int(slots_per_worker, where="slots per worker")
    result = physical * slots
    if result > 10_000:
        raise ContractError("logical worker count exceeds the four-digit worker-id bound")
    return result


def slot_specs(
    *, physical_worker_index: int, physical_worker_count: int, slots_per_worker: int
) -> tuple[SlotSpec, ...]:
    physical = _positive_int(physical_worker_count, where="physical worker count")
    index = physical_worker_index
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= physical:
        raise ContractError("physical worker index is outside the worker pool")
    slots = _positive_int(slots_per_worker, where="slots per worker")
    logical_worker_count(physical, slots)
    return tuple(
        SlotSpec(
            physical_worker_index=index,
            physical_worker_count=physical,
            slots_per_worker=slots,
            slot_index=slot,
            worker=f"worker-{index * slots + slot:04d}",
        )
        for slot in range(slots)
    )


def validate_slot_resources(
    *,
    slots_per_worker: int,
    parse_workers_per_slot: int,
    memory_limit_gb_per_slot: float,
    cpu_budget_vcpus: int,
    memory_budget_gb: float,
) -> dict[str, int | float]:
    """Validate aggregate slot limits before starting any child process."""

    slots = _positive_int(slots_per_worker, where="slots per worker")
    parse_workers = _positive_int(
        parse_workers_per_slot, where="parse workers per slot"
    )
    cpu_budget = _positive_int(cpu_budget_vcpus, where="CPU budget")
    memory_limit = _positive_float(
        memory_limit_gb_per_slot, where="memory limit per slot"
    )
    memory_budget = _positive_float(memory_budget_gb, where="memory budget")
    if slots * parse_workers > cpu_budget:
        raise ContractError(
            "aggregate parser workers exceed the VM CPU budget: "
            f"{slots} * {parse_workers} > {cpu_budget}"
        )
    if slots * memory_limit > memory_budget + 1e-9:
        raise ContractError(
            "aggregate indexer memory limits exceed the VM memory budget: "
            f"{slots} * {memory_limit:g} > {memory_budget:g}"
        )
    return {
        "parse_workers_per_slot": parse_workers,
        "memory_limit_gb_per_slot": memory_limit,
        "cpu_budget_vcpus": cpu_budget,
        "memory_budget_gb": memory_budget,
    }


def host_capacity() -> tuple[int | None, float | None]:
    """Return visible CPU and GiB capacity, tolerating non-Linux unit tests."""

    cpu = os.cpu_count()
    memory_gb: float | None = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                memory_kib = int(line.split()[1])
                memory_gb = memory_kib / (1024 * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    return cpu, memory_gb


def validate_host_capacity(resources: Mapping[str, int | float]) -> None:
    cpu, memory_gb = host_capacity()
    if cpu is not None and cpu < int(resources["cpu_budget_vcpus"]):
        raise ContractError(
            f"visible host CPU capacity {cpu} is below configured budget "
            f"{resources['cpu_budget_vcpus']}"
        )
    if memory_gb is not None:
        # Keep a small OS/GCS/clone reserve outside the configured worker
        # budget.  N2-standard-16 reports slightly under 64 GiB to Linux.
        available = memory_gb - 4.0
        if available + 1e-9 < float(resources["memory_budget_gb"]):
            raise ContractError(
                f"visible host memory {memory_gb:.2f} GiB leaves only "
                f"{available:.2f} GiB after reserve, below configured budget "
                f"{resources['memory_budget_gb']:g} GiB"
            )


def validate_manifest_topology(
    manifest: Mapping[str, object],
    *,
    physical_worker_index: int,
    physical_worker_count: int,
    slots_per_worker: int,
) -> tuple[dict[str, object], tuple[SlotSpec, ...]]:
    plan = validate_source_manifest(manifest)
    specs = slot_specs(
        physical_worker_index=physical_worker_index,
        physical_worker_count=physical_worker_count,
        slots_per_worker=slots_per_worker,
    )
    expected_workers = [
        f"worker-{index:04d}"
        for index in range(logical_worker_count(physical_worker_count, slots_per_worker))
    ]
    if plan["workers"] != expected_workers:
        raise ContractError(
            "source manifest logical workers do not match VM/slot topology: "
            f"expected {len(expected_workers)}, got {len(plan['workers'])}"
        )
    return plan, specs


def slot_completion_uri(manifest: Mapping[str, object], worker: str) -> str:
    return gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-slot-receipts",
        str(manifest["manifest_sha256"]),
        f"{worker}.complete.json",
    )


def source_receipt_uri(
    manifest: Mapping[str, object], job: Mapping[str, object], worker_receipt: Mapping[str, object]
) -> str:
    artifact = worker_receipt.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ContractError("source worker receipt has no artifact")
    compression = artifact.get("compression")
    if not isinstance(compression, Mapping):
        raise ContractError("source worker receipt has no compression metadata")
    digest = require_sha256(compression.get("sha256"), where="source artifact sha256")
    return gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-receipts",
        str(manifest["manifest_sha256"]),
        f"{int(job['ordinal']):05d}-{job['repo']}",
        f"{digest}.receipt.json",
    )


def _slot_topology(spec: SlotSpec) -> dict[str, object]:
    return {
        "physical_worker_index": spec.physical_worker_index,
        "physical_worker_count": spec.physical_worker_count,
        "slots_per_worker": spec.slots_per_worker,
        "slot_index": spec.slot_index,
        "worker": spec.worker,
    }


def _validate_slot_receipt_shape(
    receipt: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    spec: SlotSpec,
    resources: Mapping[str, int | float],
) -> list[dict[str, object]]:
    value = dict(receipt)
    require_exact_fields(
        value,
        {
            "schema",
            "status",
            "manifest_sha256",
            "manifest_file_sha256",
            "topology",
            "resources",
            "source_receipts",
            "training_ready",
        },
        where="source slot completion receipt",
    )
    if (
        value["schema"] != SLOT_COMPLETION_RECEIPT_SCHEMA
        or value["status"] != "complete"
        or value["manifest_sha256"] != manifest["manifest_sha256"]
        or value["manifest_file_sha256"] != manifest_file_sha256
        or value["training_ready"] is not False
    ):
        raise ContractError("source slot completion receipt binding drifted")
    topology = value["topology"]
    if not isinstance(topology, Mapping) or dict(topology) != _slot_topology(spec):
        raise ContractError("source slot completion topology drifted")
    raw_resources = value["resources"]
    if not isinstance(raw_resources, Mapping) or dict(raw_resources) != dict(resources):
        raise ContractError("source slot completion resource binding drifted")
    entries = value["source_receipts"]
    if not isinstance(entries, list):
        raise ContractError("source slot completion source_receipts must be a list")
    expected_jobs = repositories_for_worker(manifest, spec.worker)
    if len(entries) != len(expected_jobs):
        raise ContractError("source slot completion receipt count drifted")
    expected_fields = {
        "ordinal",
        "repo",
        "project_id",
        "worker",
        "assignment_sha256",
        "uri",
        "generation",
        "size_bytes",
        "sha256",
    }
    seen: set[int] = set()
    for raw, job in zip(entries, expected_jobs):
        if not isinstance(raw, Mapping):
            raise ContractError("source slot completion entry is not an object")
        entry = dict(raw)
        require_exact_fields(entry, expected_fields, where="source slot completion entry")
        if dict(entry)["ordinal"] != job["ordinal"] or {
            key: entry[key]
            for key in ("repo", "project_id", "worker", "assignment_sha256")
        } != {
            key: job[key]
            for key in ("repo", "project_id", "worker", "assignment_sha256")
        }:
            raise ContractError("source slot completion assignment drifted")
        ordinal = require_int(entry["ordinal"], where="source slot completion ordinal")
        if ordinal in seen:
            raise ContractError("source slot completion contains duplicate assignments")
        seen.add(ordinal)
        validate_gcs_uri(entry["uri"], where="source slot completion source receipt URI")
        generation = str(entry["generation"])
        if not generation.isdecimal() or int(generation) < 1:
            raise ContractError("source slot completion receipt generation is invalid")
        require_int(entry["size_bytes"], where="source slot completion receipt size", minimum=1)
        require_sha256(entry["sha256"], where="source slot completion receipt sha256")
    return [dict(entry) for entry in entries]


def _download_and_validate_source_receipt(
    *,
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    entry: Mapping[str, object],
    object_store: ObjectStore,
    destination: Path,
) -> dict[str, object]:
    uri = validate_gcs_uri(entry["uri"], where="source receipt URI")
    generation = str(entry["generation"])
    metadata = object_store.describe_if_present(uri, generation=generation)
    if metadata is None:
        raise ContractError(f"confirmed source receipt is missing: {uri}")
    downloaded = object_store.download(uri, destination, generation=generation)
    if (
        str(downloaded.get("uri")) != uri
        or str(downloaded.get("generation")) != generation
        or int(downloaded.get("size_bytes", -1)) != int(entry["size_bytes"])
        or destination.stat().st_size != int(entry["size_bytes"])
        or sha256_file(destination) != entry["sha256"]
    ):
        raise ContractError(f"source receipt readback drifted: {uri}")
    raw, receipt = load_json_object(destination, where="source worker receipt")
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ContractError(f"source receipt digest drifted: {uri}")
    validate_worker_receipt(receipt, manifest=manifest, job=job)
    if source_receipt_uri(manifest, job, receipt) != uri:
        raise ContractError(f"source receipt URI is not bound to its artifact: {uri}")
    return receipt


def validate_slot_completion_receipt(
    receipt: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    spec: SlotSpec,
    resources: Mapping[str, int | float],
    object_store: ObjectStore | None = None,
    verification_root: Path | None = None,
) -> dict[str, object]:
    plan = validate_source_manifest(manifest)
    require_sha256(manifest_file_sha256, where="manifest_file_sha256")
    entries = _validate_slot_receipt_shape(
        receipt,
        manifest=plan,
        manifest_file_sha256=manifest_file_sha256,
        spec=spec,
        resources=resources,
    )
    if object_store is not None:
        if verification_root is None:
            raise ValueError("verification_root is required when object_store is used")
        verification_root.mkdir(parents=True, exist_ok=True)
        jobs = repositories_for_worker(plan, spec.worker)
        with tempfile.TemporaryDirectory(prefix="slot-receipt-", dir=verification_root) as raw_tmp:
            temp_root = Path(raw_tmp)
            for entry, job in zip(entries, jobs):
                _download_and_validate_source_receipt(
                    manifest=plan,
                    job=job,
                    entry=entry,
                    object_store=object_store,
                    destination=temp_root / f"{int(job['ordinal']):05d}.json",
                )
    return dict(receipt)


def load_resumable_slot_receipt(
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    spec: SlotSpec,
    resources: Mapping[str, int | float],
    object_store: ObjectStore,
    verification_root: Path,
) -> dict[str, object] | None:
    """Load a slot receipt only when all referenced receipts read back exactly."""

    uri = slot_completion_uri(manifest, spec.worker)
    metadata = object_store.describe_if_present(uri)
    if metadata is None:
        return None
    generation = str(metadata.get("generation", ""))
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError(f"slot completion object has invalid generation: {uri}")
    verification_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="slot-completion-", dir=verification_root) as raw_tmp:
        path = Path(raw_tmp) / "completion.json"
        downloaded = object_store.download(uri, path, generation=generation)
        if (
            str(downloaded.get("uri")) != uri
            or str(downloaded.get("generation")) != generation
            or path.stat().st_size != int(metadata.get("size_bytes", -1))
        ):
            raise ContractError(f"slot completion readback metadata drifted: {uri}")
        _raw, receipt = load_json_object(path, where="source slot completion receipt")
        validate_slot_completion_receipt(
            receipt,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            spec=spec,
            resources=resources,
            object_store=object_store,
            verification_root=verification_root,
        )
        return receipt


def _extract_overlay(overlay: Path, destination: Path) -> None:
    zstd = subprocess.Popen(
        ["zstd", "-dc", "--", str(overlay)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert zstd.stdout is not None
    tar = subprocess.run(
        ["tar", "-xf", "-", "-C", str(destination), "--no-same-owner"],
        stdin=zstd.stdout,
        capture_output=True,
        check=False,
    )
    zstd.stdout.close()
    stderr = zstd.stderr.read() if zstd.stderr is not None else b""
    zstd_return = zstd.wait()
    if tar.returncode != 0 or zstd_return != 0:
        raise RuntimeError(
            "slot code overlay extraction failed: "
            f"tar={tar.stderr[-4000:].decode(errors='replace')}; "
            f"zstd={stderr[-4000:].decode(errors='replace')}"
        )


def _prepare_slot_repository(
    *,
    bundle: Path,
    overlay: Path,
    code_revision: str,
    slot_root: Path,
) -> tuple[Path, Path, Path]:
    attempt_root = Path(tempfile.mkdtemp(prefix="attempt-", dir=slot_root))
    repo_root = attempt_root / "cppmega"
    scratch_root = attempt_root / "scratch"
    receipt_root = attempt_root / "receipts"
    scratch_root.mkdir()
    receipt_root.mkdir()
    run_checked(["git", "clone", "--no-checkout", str(bundle), str(repo_root)])
    run_checked(["git", "-C", str(repo_root), "bundle", "verify", str(bundle)])
    run_checked(["git", "-C", str(repo_root), "checkout", "--detach", code_revision])
    _extract_overlay(overlay, repo_root)
    return attempt_root, repo_root, scratch_root


def _build_worker_command(
    *,
    source_worker: Path,
    manifest: Path,
    spec: SlotSpec,
    repo_root: Path,
    scratch_root: Path,
    receipt_root: Path,
    python: Path,
    resources: Mapping[str, int | float],
) -> list[str]:
    return [
        str(python),
        str(source_worker),
        "--manifest",
        str(manifest),
        "--worker",
        spec.worker,
        "--scratch-root",
        str(scratch_root),
        "--receipt-root",
        str(receipt_root),
        "--repo-root",
        str(repo_root),
        "--python",
        str(python),
        "--parse-workers",
        str(resources["parse_workers_per_slot"]),
        "--memory-limit-gb",
        str(resources["memory_limit_gb_per_slot"]),
    ]


def _build_slot_completion(
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    spec: SlotSpec,
    resources: Mapping[str, int | float],
    receipt_root: Path,
    object_store: ObjectStore,
    verification_root: Path,
) -> dict[str, object]:
    plan = validate_source_manifest(manifest)
    jobs = repositories_for_worker(plan, spec.worker)
    entries: list[dict[str, object]] = []
    verification_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="source-receipt-check-", dir=verification_root) as raw_tmp:
        temp_root = Path(raw_tmp)
        for job in jobs:
            local = receipt_root / f"{int(job['ordinal']):05d}-{job['repo']}.json"
            if not local.is_file() or local.is_symlink():
                raise ContractError(
                    f"source worker did not emit the expected receipt: {local.name}"
                )
            raw, worker_receipt = load_json_object(local, where="source worker receipt")
            validate_worker_receipt(worker_receipt, manifest=plan, job=job)
            local_sha = hashlib.sha256(raw).hexdigest()
            uri = source_receipt_uri(plan, job, worker_receipt)
            metadata = object_store.describe_if_present(uri)
            if metadata is None:
                raise ContractError(f"source worker receipt was not published: {uri}")
            generation = str(metadata.get("generation", ""))
            if not generation.isdecimal() or int(generation) < 1:
                raise ContractError(f"source worker receipt generation is invalid: {uri}")
            checked = _download_and_validate_source_receipt(
                manifest=plan,
                job=job,
                entry={
                    "uri": uri,
                    "generation": generation,
                    "size_bytes": int(metadata.get("size_bytes", -1)),
                    "sha256": local_sha,
                },
                object_store=object_store,
                destination=temp_root / f"{int(job['ordinal']):05d}.json",
            )
            if checked != worker_receipt:
                raise ContractError(f"local and published source receipt differ: {uri}")
            entries.append(
                {
                    "ordinal": job["ordinal"],
                    "repo": job["repo"],
                    "project_id": job["project_id"],
                    "worker": job["worker"],
                    "assignment_sha256": job["assignment_sha256"],
                    "uri": uri,
                    "generation": generation,
                    "size_bytes": int(metadata["size_bytes"]),
                    "sha256": local_sha,
                }
            )
    receipt: dict[str, object] = {
        "schema": SLOT_COMPLETION_RECEIPT_SCHEMA,
        "status": "complete",
        "manifest_sha256": plan["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha256,
        "topology": _slot_topology(spec),
        "resources": dict(resources),
        "source_receipts": entries,
        "training_ready": False,
    }
    validate_slot_completion_receipt(
        receipt,
        manifest=plan,
        manifest_file_sha256=manifest_file_sha256,
        spec=spec,
        resources=resources,
    )
    return receipt


def _publish_slot_completion(
    *,
    receipt: Mapping[str, object],
    manifest: Mapping[str, object],
    spec: SlotSpec,
    slot_root: Path,
    object_store: ObjectStore,
) -> Mapping[str, object]:
    local = slot_root / "completion.json"
    atomic_write_json(local, receipt)
    uri = slot_completion_uri(manifest, spec.worker)
    metadata = object_store.publish_if_absent(local, uri)
    generation = str(metadata.get("generation", ""))
    if str(metadata.get("uri")) != uri or not generation.isdecimal():
        raise ContractError(f"slot completion publication metadata drifted: {uri}")
    with tempfile.TemporaryDirectory(prefix="slot-publish-check-", dir=slot_root) as raw_tmp:
        path = Path(raw_tmp) / "completion.json"
        downloaded = object_store.download(uri, path, generation=generation)
        if (
            str(downloaded.get("generation")) != generation
            or path.stat().st_size != local.stat().st_size
            or sha256_file(path) != sha256_file(local)
        ):
            raise ContractError(f"slot completion publication readback drifted: {uri}")
    return metadata


def _terminate_processes(active: Sequence[dict[str, object]]) -> None:
    processes = [entry["process"] for entry in active]
    for process in processes:
        assert isinstance(process, subprocess.Popen)
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 30.0
    for process in processes:
        assert isinstance(process, subprocess.Popen)
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _scheduler_receipt(
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    specs: Sequence[SlotSpec],
    resources: Mapping[str, int | float],
    completed: Sequence[str],
    resumed: Sequence[str],
    source_receipt_count: int,
) -> dict[str, object]:
    return {
        "schema": SCHEDULER_RECEIPT_SCHEMA,
        "status": "complete",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha256,
        "physical_worker_index": specs[0].physical_worker_index,
        "physical_worker_count": specs[0].physical_worker_count,
        "slots_per_worker": specs[0].slots_per_worker,
        "workers": [spec.worker for spec in specs],
        "completed_slots": sorted(completed),
        "resumed_slots": sorted(resumed),
        "source_receipt_count": source_receipt_count,
        "resources": dict(resources),
        "training_ready": False,
    }


def run_source_slot_scheduler(
    *,
    manifest_path: Path,
    manifest_file_sha256: str,
    run_root: str,
    bundle: Path,
    overlay: Path,
    stage_root: Path,
    scheduler_receipt_path: Path,
    physical_worker_index: int,
    physical_worker_count: int,
    slots_per_worker: int = 1,
    parse_workers_per_slot: int = 8,
    memory_limit_gb_per_slot: float = 48.0,
    cpu_budget_vcpus: int = 16,
    memory_budget_gb: float = 56.0,
    python: Path = Path(sys.executable),
    source_worker: Path | None = None,
    object_store: ObjectStore | None = None,
    check_host: bool = True,
    poll_interval: float = 0.2,
) -> dict[str, object]:
    """Run all slots owned by one VM and publish exact completion receipts."""

    manifest, raw_sha256 = load_source_manifest(manifest_path)
    if raw_sha256 != require_sha256(
        manifest_file_sha256, where="manifest_file_sha256"
    ):
        raise ContractError("manifest file SHA-256 drifted")
    if validate_gcs_uri(run_root, where="run_root") != manifest["gcs_output_prefix"]:
        raise ContractError("worker run root does not match manifest output prefix")
    manifest, specs = validate_manifest_topology(
        manifest,
        physical_worker_index=physical_worker_index,
        physical_worker_count=physical_worker_count,
        slots_per_worker=slots_per_worker,
    )
    resources = validate_slot_resources(
        slots_per_worker=slots_per_worker,
        parse_workers_per_slot=parse_workers_per_slot,
        memory_limit_gb_per_slot=memory_limit_gb_per_slot,
        cpu_budget_vcpus=cpu_budget_vcpus,
        memory_budget_gb=memory_budget_gb,
    )
    if check_host:
        validate_host_capacity(resources)
    if not bundle.is_file() or bundle.is_symlink():
        raise ContractError(f"source bundle is not a regular file: {bundle}")
    if not overlay.is_file() or overlay.is_symlink():
        raise ContractError(f"source overlay is not a regular file: {overlay}")
    if poll_interval <= 0 or not math.isfinite(poll_interval):
        raise ValueError("poll_interval must be positive and finite")
    store = object_store or GcloudObjectStore()
    source_worker_path = (source_worker or Path(__file__).with_name("source_worker.py")).resolve()
    if not source_worker_path.is_file() or source_worker_path.is_symlink():
        raise ContractError(f"source worker script is not a regular file: {source_worker_path}")
    stage_root = stage_root.resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    scheduler_receipt_path = scheduler_receipt_path.resolve()
    scheduler_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = stage_root / ".source-slot-scheduler.lock"
    with lock_path.open("a+", encoding="ascii") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("another source slot scheduler is already running") from exc
        slots_root = stage_root / "slots"
        slots_root.mkdir(parents=True, exist_ok=True)
        resumed: list[str] = []
        completed: list[str] = []
        transient_failures: list[str] = []
        deterministic_failures: list[tuple[str, int, Path]] = []
        source_receipt_count = 0
        pending: list[tuple[SlotSpec, Path]] = []
        for spec in specs:
            slot_root = slots_root / spec.worker
            slot_root.mkdir(parents=True, exist_ok=True)
            try:
                existing = load_resumable_slot_receipt(
                    manifest=manifest,
                    manifest_file_sha256=manifest_file_sha256,
                    spec=spec,
                    resources=resources,
                    object_store=store,
                    verification_root=slot_root / "verify",
                )
            except ContractError:
                # A present but invalid immutable receipt is not equivalent to
                # an absent receipt.  Refuse to overwrite it or silently rerun.
                raise
            if existing is not None:
                resumed.append(spec.worker)
                source_receipt_count += len(existing["source_receipts"])
            else:
                pending.append((spec, slot_root))

        active: list[dict[str, object]] = []
        try:
            for spec, slot_root in pending:
                attempt_root, repo_root, scratch_root = _prepare_slot_repository(
                    bundle=bundle,
                    overlay=overlay,
                    code_revision=str(manifest["code_revision"]),
                    slot_root=slot_root,
                )
                receipt_root = attempt_root / "receipts"
                log_path = attempt_root / "source-worker.log"
                log = log_path.open("ab")
                command = _build_worker_command(
                    source_worker=source_worker_path,
                    manifest=manifest_path.resolve(),
                    spec=spec,
                    repo_root=repo_root,
                    scratch_root=scratch_root,
                    receipt_root=receipt_root,
                    python=python.resolve(),
                    resources=resources,
                )
                environment = os.environ.copy()
                environment.update(
                    {
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "GIT_OPTIONAL_LOCKS": "0",
                        "CPPMEGA_SOURCE_SLOT": spec.worker,
                    }
                )
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                active.append(
                    {
                        "spec": spec,
                        "slot_root": slot_root,
                        "attempt_root": attempt_root,
                        "receipt_root": receipt_root,
                        "process": process,
                        "log": log,
                    }
                )

            while active:
                for entry in list(active):
                    process = entry["process"]
                    assert isinstance(process, subprocess.Popen)
                    code = process.poll()
                    if code is None:
                        continue
                    log = entry["log"]
                    assert hasattr(log, "close")
                    log.close()
                    active.remove(entry)
                    spec = entry["spec"]
                    assert isinstance(spec, SlotSpec)
                    if code == 75:
                        # A transiently exhausted slot has no completion receipt.
                        # Keep other slots running so their immutable receipts are
                        # preserved; the outer systemd service will resume only
                        # this slot on its next attempt.
                        transient_failures.append(spec.worker)
                        continue
                    if code != 0:
                        attempt_root = entry["attempt_root"]
                        assert isinstance(attempt_root, Path)
                        deterministic_failures.append(
                            (spec.worker, code, attempt_root / "source-worker.log")
                        )
                        continue
                    slot_receipt = _build_slot_completion(
                        manifest=manifest,
                        manifest_file_sha256=manifest_file_sha256,
                        spec=spec,
                        resources=resources,
                        receipt_root=entry["receipt_root"],
                        object_store=store,
                        verification_root=entry["slot_root"] / "verify",
                    )
                    _publish_slot_completion(
                        receipt=slot_receipt,
                        manifest=manifest,
                        spec=spec,
                        slot_root=entry["slot_root"],
                        object_store=store,
                    )
                    completed.append(spec.worker)
                    source_receipt_count += len(slot_receipt["source_receipts"])
                if active:
                    time.sleep(poll_interval)
        except BaseException:
            _terminate_processes(active)
            for entry in active:
                log = entry["log"]
                if hasattr(log, "close"):
                    log.close()
            raise

        if deterministic_failures:
            details = "; ".join(
                f"{worker} exit={code} log={log_path}"
                for worker, code, log_path in sorted(deterministic_failures)
            )
            transient_detail = (
                "; transient slots pending: "
                f"{', '.join(sorted(transient_failures))}"
                if transient_failures
                else ""
            )
            raise RuntimeError(
                "source slots failed deterministically: "
                f"{details}; completed slots preserved: "
                f"{', '.join(sorted(completed)) or 'none'}"
                f"{transient_detail}"
            )

        if transient_failures:
            raise TransientTransportError(
                "source slot transport retry budget exhausted for "
                f"{', '.join(sorted(transient_failures))}; completed slots preserved: "
                f"{', '.join(sorted(completed)) or 'none'}"
            )

        receipt = _scheduler_receipt(
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            specs=specs,
            resources=resources,
            completed=completed + resumed,
            resumed=resumed,
            source_receipt_count=source_receipt_count,
        )
        atomic_write_json(scheduler_receipt_path, receipt)
        return receipt


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--scheduler-receipt", required=True, type=Path)
    parser.add_argument("--physical-worker-index", required=True, type=int)
    parser.add_argument("--physical-worker-count", required=True, type=int)
    parser.add_argument("--slots-per-worker", type=int, default=1)
    parser.add_argument("--parse-workers-per-slot", type=int, default=8)
    parser.add_argument("--memory-limit-gb-per-slot", type=float, default=48.0)
    parser.add_argument("--cpu-budget-vcpus", type=int, default=16)
    parser.add_argument("--memory-budget-gb", type=float, default=56.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source-worker", type=Path)
    args = parser.parse_args(argv)
    try:
        run_source_slot_scheduler(
            manifest_path=args.manifest,
            manifest_file_sha256=args.manifest_file_sha256,
            run_root=args.run_root,
            bundle=args.bundle,
            overlay=args.overlay,
            stage_root=args.stage_root,
            scheduler_receipt_path=args.scheduler_receipt,
            physical_worker_index=args.physical_worker_index,
            physical_worker_count=args.physical_worker_count,
            slots_per_worker=args.slots_per_worker,
            parse_workers_per_slot=args.parse_workers_per_slot,
            memory_limit_gb_per_slot=args.memory_limit_gb_per_slot,
            cpu_budget_vcpus=args.cpu_budget_vcpus,
            memory_budget_gb=args.memory_budget_gb,
            python=args.python,
            source_worker=args.source_worker,
        )
    except TransientTransportError as exc:
        parser.exit(75, f"distributed source slot scheduler transient failure: {exc}\n")
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"distributed source slot scheduler failed: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "SCHEDULER_RECEIPT_SCHEMA",
    "SLOT_COMPLETION_RECEIPT_SCHEMA",
    "SlotSpec",
    "logical_worker_count",
    "load_resumable_slot_receipt",
    "run_source_slot_scheduler",
    "slot_completion_uri",
    "slot_specs",
    "source_receipt_uri",
    "validate_host_capacity",
    "validate_manifest_topology",
    "validate_slot_completion_receipt",
    "validate_slot_resources",
]
