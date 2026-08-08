#!/usr/bin/env python3
"""Immutable GCP worker-lane contracts for PR, MR, and CI candidates.

This module is deliberately a transport and resume boundary, not a dataset
materializer.  It binds already-verified store snapshots, assigns contiguous
primary-record ranges deterministically, publishes content-addressed candidate
segments and checkpoints with exact-generation readback, and emits a completion
receipt that can become ``source_receipt_sha256`` in the existing output sealing
contract.  Candidate receipts always remain ``training_ready = false`` until a
separate lossless Parquet/Megatron materializer and ``seal_outputs`` succeed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    gcs_join,
    iter_jsonl_bytes,
    load_json_object,
    require_exact_fields,
    require_git_object,
    require_int,
    require_nonempty,
    require_sha256,
    sha256_file,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.seal_outputs import (
    BUCKET_AUDIT_SCHEMA,
    OUTPUT_MANIFEST_SCHEMA,
    TARGET_LENGTHS,
)

CLOUD_LANE_MANIFEST_SCHEMA = "cppmega.distributed_cloud_lane_manifest_v1"
CLOUD_LANE_CHECKPOINT_SCHEMA = "cppmega.distributed_cloud_lane_checkpoint_v1"
CLOUD_LANE_COMPLETION_SCHEMA = "cppmega.distributed_cloud_lane_completion_v1"
CLOUD_LANE_AGGREGATE_SCHEMA = "cppmega.distributed_cloud_lane_aggregate_v1"
IMMUTABLE_OBJECT_DESCRIPTOR_SCHEMA = "cppmega.immutable_gcs_object_v1"

LANE_KINDS = ("github_pr", "gitlab_mr", "ci")
ASSIGNMENT_ALGORITHM = "contiguous_primary_ranges_round_robin_v1"
CANDIDATE_FORMAT = "canonical_jsonl"
CANDIDATE_COMPRESSION = "zstd"
CANDIDATE_DOCUMENT_ORDER = "canonical_source_record_then_document_v1"
CANDIDATE_ENVELOPE_SCHEMA = "cppmega.distributed_cloud_lane_candidate_v1"

_KIND_SET = frozenset(LANE_KINDS)
_SNAPSHOT_ROLES = frozenset({"primary", "membership", "ancillary"})
_WORKER_RE = re.compile(r"^worker-[0-9]{4}$")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_FORMAT_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,63}$")
_ITEM_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+%:@=-]{0,127}$")


class ObjectStore(Protocol):
    """The generation-aware subset shared with the source GCS transport."""

    def publish_if_absent(self, source: Path, uri: str) -> Mapping[str, object]: ...

    def download(
        self, uri: str, destination: Path, *, generation: str | None = None
    ) -> Mapping[str, object]: ...


def _without_digest(value: Mapping[str, object], field: str) -> dict[str, object]:
    result = copy.deepcopy(dict(value))
    result.pop(field, None)
    return result


def cloud_lane_manifest_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "manifest_sha256"))


def cloud_lane_checkpoint_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "checkpoint_sha256"))


def cloud_lane_completion_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "receipt_sha256"))


def cloud_lane_aggregate_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "receipt_sha256"))


def _positive_generation(value: object, *, where: str) -> str:
    generation = require_nonempty(value, where=where)
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError(f"{where} must be a positive decimal generation")
    return generation


def _canonical_item_id(value: object, *, where: str) -> str:
    item_id = require_nonempty(value, where=where)
    parts = item_id.split("/")
    if any(_ITEM_PART_RE.fullmatch(part) is None for part in parts):
        raise ContractError(f"{where} is not a canonical item id")
    return item_id


def _validate_snapshot(value: object, *, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    require_exact_fields(
        value,
        {
            "name",
            "role",
            "uri",
            "generation",
            "size_bytes",
            "sha256",
            "content_set_sha256",
            "schema_sha256",
            "format",
            "record_count",
        },
        where=where,
    )
    name = require_nonempty(value["name"], where=f"{where}.name")
    if _NAME_RE.fullmatch(name) is None:
        raise ContractError(f"{where}.name is not canonical")
    role = require_nonempty(value["role"], where=f"{where}.role")
    if role not in _SNAPSHOT_ROLES:
        raise ContractError(f"{where}.role is unsupported: {role!r}")
    snapshot_format = require_nonempty(value["format"], where=f"{where}.format")
    if _FORMAT_RE.fullmatch(snapshot_format) is None:
        raise ContractError(f"{where}.format is not canonical")
    record_count = require_int(
        value["record_count"], where=f"{where}.record_count", minimum=0
    )
    if role in {"primary", "membership"} and record_count < 1:
        raise ContractError(f"{where}.{role} snapshot must contain records")
    return {
        "name": name,
        "role": role,
        "uri": validate_gcs_uri(value["uri"], where=f"{where}.uri"),
        "generation": _positive_generation(
            value["generation"], where=f"{where}.generation"
        ),
        "size_bytes": require_int(
            value["size_bytes"], where=f"{where}.size_bytes", minimum=1
        ),
        "sha256": require_sha256(value["sha256"], where=f"{where}.sha256"),
        "content_set_sha256": require_sha256(
            value["content_set_sha256"], where=f"{where}.content_set_sha256"
        ),
        "schema_sha256": require_sha256(
            value["schema_sha256"], where=f"{where}.schema_sha256"
        ),
        "format": snapshot_format,
        "record_count": record_count,
    }


def _normalize_snapshots(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ContractError("input_snapshots must be a non-empty list")
    snapshots = [
        _validate_snapshot(raw, where=f"input_snapshots[{index}]")
        for index, raw in enumerate(value)
    ]
    names = [str(snapshot["name"]) for snapshot in snapshots]
    if len(names) != len(set(names)):
        raise ContractError("input snapshot names must be unique")
    if snapshots != sorted(snapshots, key=lambda item: str(item["name"])):
        raise ContractError("input snapshots must be name-sorted")
    roles = [str(snapshot["role"]) for snapshot in snapshots]
    if roles.count("primary") != 1 or roles.count("membership") != 1:
        raise ContractError(
            "input snapshots require exactly one primary and one membership snapshot"
        )
    return snapshots


def _snapshot_set_sha256(snapshots: Sequence[Mapping[str, object]]) -> str:
    return canonical_sha256([dict(snapshot) for snapshot in snapshots])


def _producer_sha256(
    *,
    code_revision: str,
    runner_sha256: str,
    membership_policy_sha256: str,
    candidate_schema_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "code_revision": code_revision,
            "runner_sha256": runner_sha256,
            "membership_policy_sha256": membership_policy_sha256,
            "candidate_schema_sha256": candidate_schema_sha256,
            "candidate_format": CANDIDATE_FORMAT,
            "candidate_compression": CANDIDATE_COMPRESSION,
            "document_order": CANDIDATE_DOCUMENT_ORDER,
        }
    )


def _normalize_work_items(
    value: object, *, primary_record_count: int
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ContractError("work_items must be a non-empty list")
    items: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        where = f"work_items[{index}]"
        if not isinstance(raw, Mapping):
            raise ContractError(f"{where} must be an object")
        require_exact_fields(
            raw,
            {"item_id", "record_start", "record_count", "partition_sha256"},
            where=where,
        )
        items.append(
            {
                "item_id": _canonical_item_id(raw["item_id"], where=f"{where}.item_id"),
                "record_start": require_int(
                    raw["record_start"], where=f"{where}.record_start", minimum=0
                ),
                "record_count": require_int(
                    raw["record_count"], where=f"{where}.record_count", minimum=1
                ),
                "partition_sha256": require_sha256(
                    raw["partition_sha256"], where=f"{where}.partition_sha256"
                ),
            }
        )
    items.sort(key=lambda item: (int(item["record_start"]), str(item["item_id"])))
    ids = [str(item["item_id"]) for item in items]
    if len(ids) != len(set(ids)):
        raise ContractError("work item ids must be unique")
    cursor = 0
    for item in items:
        if item["record_start"] != cursor:
            raise ContractError(
                "work items must cover contiguous primary record ranges"
            )
        cursor += int(item["record_count"])
    if cursor != primary_record_count:
        raise ContractError("work items do not exactly cover the primary snapshot")
    return items


def _assignment_payload(
    *,
    kind: str,
    ordinal: int,
    item: Mapping[str, object],
    worker: str,
    input_snapshot_set_sha256: str,
    producer_sha256: str,
) -> dict[str, object]:
    return {
        "kind": kind,
        "ordinal": ordinal,
        "item_id": item["item_id"],
        "record_start": item["record_start"],
        "record_count": item["record_count"],
        "partition_sha256": item["partition_sha256"],
        "worker": worker,
        "input_snapshot_set_sha256": input_snapshot_set_sha256,
        "producer_sha256": producer_sha256,
        "target_lengths": list(TARGET_LENGTHS),
    }


def _assignment_record(
    *,
    kind: str,
    ordinal: int,
    item: Mapping[str, object],
    worker: str,
    input_snapshot_set_sha256: str,
    producer_sha256: str,
) -> dict[str, object]:
    payload = _assignment_payload(
        kind=kind,
        ordinal=ordinal,
        item=item,
        worker=worker,
        input_snapshot_set_sha256=input_snapshot_set_sha256,
        producer_sha256=producer_sha256,
    )
    return {
        "ordinal": ordinal,
        "item_id": item["item_id"],
        "record_start": item["record_start"],
        "record_count": item["record_count"],
        "partition_sha256": item["partition_sha256"],
        "worker": worker,
        "assignment_sha256": canonical_sha256(payload),
    }


def build_cloud_lane_manifest(
    *,
    kind: str,
    input_snapshots: Sequence[Mapping[str, object]],
    work_items: Sequence[Mapping[str, object]],
    worker_count: int,
    gcs_output_prefix: str,
    code_revision: str,
    runner_sha256: str,
    tokenizer_sha256: str,
    dataset_schema_sha256: str,
    membership_policy_sha256: str,
    candidate_schema_sha256: str,
) -> dict[str, object]:
    """Build a deterministic manifest for one PR/MR/CI snapshot set."""

    if kind not in _KIND_SET:
        raise ContractError(f"unsupported cloud lane kind: {kind!r}")
    if (
        isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or worker_count < 1
    ):
        raise ContractError("worker_count must be a positive integer")
    workers = [f"worker-{index:04d}" for index in range(worker_count)]
    prefix = validate_gcs_uri(gcs_output_prefix.rstrip("/"), where="gcs_output_prefix")
    snapshots = sorted(
        [dict(snapshot) for snapshot in input_snapshots],
        key=lambda item: str(item.get("name", "")),
    )
    snapshots = _normalize_snapshots(snapshots)
    primary = next(snapshot for snapshot in snapshots if snapshot["role"] == "primary")
    normalized_items = _normalize_work_items(
        [dict(item) for item in work_items],
        primary_record_count=int(primary["record_count"]),
    )
    revision = require_git_object(code_revision, where="code_revision")
    runner = require_sha256(runner_sha256, where="runner_sha256")
    tokenizer = require_sha256(tokenizer_sha256, where="tokenizer_sha256")
    dataset_schema = require_sha256(
        dataset_schema_sha256, where="dataset_schema_sha256"
    )
    membership_policy = require_sha256(
        membership_policy_sha256, where="membership_policy_sha256"
    )
    candidate_schema = require_sha256(
        candidate_schema_sha256, where="candidate_schema_sha256"
    )
    snapshot_set = _snapshot_set_sha256(snapshots)
    producer = _producer_sha256(
        code_revision=revision,
        runner_sha256=runner,
        membership_policy_sha256=membership_policy,
        candidate_schema_sha256=candidate_schema,
    )
    assignments = [
        _assignment_record(
            kind=kind,
            ordinal=ordinal,
            item=item,
            worker=workers[ordinal % len(workers)],
            input_snapshot_set_sha256=snapshot_set,
            producer_sha256=producer,
        )
        for ordinal, item in enumerate(normalized_items)
    ]
    manifest: dict[str, object] = {
        "schema": CLOUD_LANE_MANIFEST_SCHEMA,
        "status": "ready",
        "kind": kind,
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "workers": workers,
        "gcs_output_prefix": prefix,
        "input_snapshots": snapshots,
        "input_snapshot_set_sha256": snapshot_set,
        "primary_record_count": int(primary["record_count"]),
        "work_item_count": len(assignments),
        "work_item_order_sha256": canonical_sha256(
            [
                {
                    key: assignment[key]
                    for key in (
                        "item_id",
                        "record_start",
                        "record_count",
                        "partition_sha256",
                    )
                }
                for assignment in assignments
            ]
        ),
        "pipeline": {
            "code_revision": revision,
            "runner_sha256": runner,
            "producer_sha256": producer,
            "tokenizer_sha256": tokenizer,
            "dataset_schema_sha256": dataset_schema,
            "membership_policy_sha256": membership_policy,
            "candidate_schema_sha256": candidate_schema,
            "candidate_format": CANDIDATE_FORMAT,
            "candidate_compression": CANDIDATE_COMPRESSION,
            "document_order": CANDIDATE_DOCUMENT_ORDER,
            "global_dedup_applied": False,
            "target_lengths": list(TARGET_LENGTHS),
            "downstream_output_manifest_schema": OUTPUT_MANIFEST_SCHEMA,
            "training_ready": False,
        },
        "assignments": assignments,
    }
    manifest["manifest_sha256"] = cloud_lane_manifest_sha256(manifest)
    return validate_cloud_lane_manifest(manifest)


def validate_cloud_lane_manifest(value: Mapping[str, object]) -> dict[str, object]:
    manifest = copy.deepcopy(dict(value))
    require_exact_fields(
        manifest,
        {
            "schema",
            "status",
            "kind",
            "manifest_sha256",
            "assignment_algorithm",
            "workers",
            "gcs_output_prefix",
            "input_snapshots",
            "input_snapshot_set_sha256",
            "primary_record_count",
            "work_item_count",
            "work_item_order_sha256",
            "pipeline",
            "assignments",
        },
        where="cloud lane manifest",
    )
    if (
        manifest["schema"] != CLOUD_LANE_MANIFEST_SCHEMA
        or manifest["status"] != "ready"
    ):
        raise ContractError("cloud lane manifest schema/status is unsupported")
    kind = require_nonempty(manifest["kind"], where="cloud lane kind")
    if kind not in _KIND_SET:
        raise ContractError(f"unsupported cloud lane kind: {kind!r}")
    if manifest["assignment_algorithm"] != ASSIGNMENT_ALGORITHM:
        raise ContractError("cloud lane assignment algorithm drifted")
    digest = require_sha256(manifest["manifest_sha256"], where="manifest_sha256")
    if cloud_lane_manifest_sha256(manifest) != digest:
        raise ContractError("cloud lane manifest logical digest is invalid")
    validate_gcs_uri(manifest["gcs_output_prefix"], where="gcs_output_prefix")
    workers = manifest["workers"]
    if (
        not isinstance(workers, list)
        or not workers
        or workers != [f"worker-{index:04d}" for index in range(len(workers))]
        or any(
            not isinstance(worker, str) or _WORKER_RE.fullmatch(worker) is None
            for worker in workers
        )
    ):
        raise ContractError("cloud lane workers are not canonical")
    snapshots = _normalize_snapshots(manifest["input_snapshots"])
    snapshot_set = require_sha256(
        manifest["input_snapshot_set_sha256"], where="input_snapshot_set_sha256"
    )
    if _snapshot_set_sha256(snapshots) != snapshot_set:
        raise ContractError("input snapshot set digest drifted")
    primary = next(snapshot for snapshot in snapshots if snapshot["role"] == "primary")
    primary_count = require_int(
        manifest["primary_record_count"], where="primary_record_count", minimum=1
    )
    if primary_count != primary["record_count"]:
        raise ContractError("primary record count drifted from the snapshot")

    pipeline = manifest["pipeline"]
    if not isinstance(pipeline, Mapping):
        raise ContractError("cloud lane pipeline must be an object")
    require_exact_fields(
        pipeline,
        {
            "code_revision",
            "runner_sha256",
            "producer_sha256",
            "tokenizer_sha256",
            "dataset_schema_sha256",
            "membership_policy_sha256",
            "candidate_schema_sha256",
            "candidate_format",
            "candidate_compression",
            "document_order",
            "global_dedup_applied",
            "target_lengths",
            "downstream_output_manifest_schema",
            "training_ready",
        },
        where="cloud lane pipeline",
    )
    revision = require_git_object(
        pipeline["code_revision"], where="pipeline.code_revision"
    )
    runner = require_sha256(pipeline["runner_sha256"], where="pipeline.runner_sha256")
    for field in (
        "producer_sha256",
        "tokenizer_sha256",
        "dataset_schema_sha256",
        "membership_policy_sha256",
        "candidate_schema_sha256",
    ):
        require_sha256(pipeline[field], where=f"pipeline.{field}")
    expected_producer = _producer_sha256(
        code_revision=revision,
        runner_sha256=runner,
        membership_policy_sha256=str(pipeline["membership_policy_sha256"]),
        candidate_schema_sha256=str(pipeline["candidate_schema_sha256"]),
    )
    if pipeline["producer_sha256"] != expected_producer:
        raise ContractError("cloud lane producer digest drifted")
    if (
        pipeline["candidate_format"] != CANDIDATE_FORMAT
        or pipeline["candidate_compression"] != CANDIDATE_COMPRESSION
        or pipeline["document_order"] != CANDIDATE_DOCUMENT_ORDER
        or pipeline["global_dedup_applied"] is not False
        or pipeline["target_lengths"] != list(TARGET_LENGTHS)
        or pipeline["downstream_output_manifest_schema"] != OUTPUT_MANIFEST_SCHEMA
        or pipeline["training_ready"] is not False
    ):
        raise ContractError(
            "cloud lane pipeline is not a pre-sealing candidate contract"
        )

    raw_assignments = manifest["assignments"]
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ContractError("cloud lane assignments must be a non-empty list")
    expected_count = require_int(
        manifest["work_item_count"], where="work_item_count", minimum=1
    )
    if len(raw_assignments) != expected_count:
        raise ContractError("cloud lane work_item_count drifted")
    assignments: list[dict[str, object]] = []
    cursor = 0
    ids: set[str] = set()
    for ordinal, raw in enumerate(raw_assignments):
        where = f"assignments[{ordinal}]"
        if not isinstance(raw, Mapping):
            raise ContractError(f"{where} must be an object")
        require_exact_fields(
            raw,
            {
                "ordinal",
                "item_id",
                "record_start",
                "record_count",
                "partition_sha256",
                "worker",
                "assignment_sha256",
            },
            where=where,
        )
        if raw["ordinal"] != ordinal:
            raise ContractError("cloud lane assignment ordinals are not contiguous")
        item_id = _canonical_item_id(raw["item_id"], where=f"{where}.item_id")
        if item_id in ids:
            raise ContractError("cloud lane assignment item ids are not unique")
        ids.add(item_id)
        record_start = require_int(raw["record_start"], where=f"{where}.record_start")
        record_count = require_int(
            raw["record_count"], where=f"{where}.record_count", minimum=1
        )
        if record_start != cursor:
            raise ContractError("cloud lane assignments do not cover contiguous ranges")
        cursor += record_count
        partition_sha = require_sha256(
            raw["partition_sha256"], where=f"{where}.partition_sha256"
        )
        worker = raw["worker"]
        if worker != workers[ordinal % len(workers)]:
            raise ContractError(f"{where}.worker assignment drifted")
        expected_assignment = _assignment_record(
            kind=kind,
            ordinal=ordinal,
            item={
                "item_id": item_id,
                "record_start": record_start,
                "record_count": record_count,
                "partition_sha256": partition_sha,
            },
            worker=str(worker),
            input_snapshot_set_sha256=snapshot_set,
            producer_sha256=expected_producer,
        )
        if dict(raw) != expected_assignment:
            raise ContractError(f"{where} digest or fields drifted")
        assignments.append(expected_assignment)
    if cursor != primary_count:
        raise ContractError("cloud lane assignment ranges do not close")
    order_digest = require_sha256(
        manifest["work_item_order_sha256"], where="work_item_order_sha256"
    )
    expected_order = canonical_sha256(
        [
            {
                key: assignment[key]
                for key in (
                    "item_id",
                    "record_start",
                    "record_count",
                    "partition_sha256",
                )
            }
            for assignment in assignments
        ]
    )
    if order_digest != expected_order:
        raise ContractError("cloud lane work item order digest drifted")
    manifest["input_snapshots"] = snapshots
    manifest["assignments"] = assignments
    manifest["pipeline"] = dict(pipeline)
    return manifest


def load_cloud_lane_manifest(path: Path) -> tuple[dict[str, object], str]:
    raw, payload = load_json_object(path, where="cloud lane manifest")
    return validate_cloud_lane_manifest(payload), hashlib.sha256(raw).hexdigest()


def assignments_for_worker(
    manifest: Mapping[str, object], worker: str
) -> tuple[dict[str, object], ...]:
    plan = validate_cloud_lane_manifest(manifest)
    if worker not in plan["workers"]:
        raise ContractError(f"worker is not assigned by the manifest: {worker}")
    return tuple(
        dict(assignment)
        for assignment in plan["assignments"]
        if assignment["worker"] == worker
    )


def _job_from_manifest(
    manifest: Mapping[str, object], assignment: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    plan = validate_cloud_lane_manifest(manifest)
    ordinal = require_int(assignment.get("ordinal"), where="assignment.ordinal")
    jobs = plan["assignments"]
    assert isinstance(jobs, list)
    if ordinal >= len(jobs) or dict(assignment) != jobs[ordinal]:
        raise ContractError("assignment is not an exact member of the manifest")
    return plan, dict(jobs[ordinal])


def _assignment_namespace(assignment: Mapping[str, object]) -> str:
    return (
        f"{int(assignment['ordinal']):05d}-{str(assignment['assignment_sha256'])[:16]}"
    )


def segment_uri(
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
    *,
    segment_ordinal: int,
    sha256: str,
) -> str:
    plan, job = _job_from_manifest(manifest, assignment)
    digest = require_sha256(sha256, where="segment sha256")
    ordinal = require_int(segment_ordinal, where="segment ordinal")
    return gcs_join(
        str(plan["gcs_output_prefix"]),
        "lane-segments",
        str(plan["kind"]),
        str(plan["manifest_sha256"]),
        _assignment_namespace(job),
        f"{ordinal:05d}-{digest}.jsonl.zst",
    )


def _stable_local_file(path: Path, *, where: str) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{where} must be a regular file: {path}")
    before = path.stat()
    if before.st_size < 1:
        raise ContractError(f"{where} must not be empty")
    digest = sha256_file(path)
    after = path.stat()
    identity = lambda stat: (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise ContractError(f"{where} changed while hashing")
    return before.st_size, digest


def _validate_candidate_segment_file(
    path: Path,
    *,
    kind: str,
    source_record_start: int,
    source_record_count: int,
    candidate_document_count: int,
    valid_tokens: int,
    scratch_root: Path,
) -> None:
    """Verify ZSTD integrity and the generic canonical candidate envelope."""

    before = _stable_local_file(path, where="candidate segment")
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lane-candidate-", dir=scratch_root) as raw:
        decompressed = Path(raw) / "candidate.jsonl"
        with decompressed.open("wb") as output:
            completed = subprocess.run(
                ["zstd", "--quiet", "--decompress", "--stdout", str(path)],
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
            )
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace")[-2000:]
            raise ContractError(f"candidate segment is not valid ZSTD: {message}")

        observed_count = 0
        observed_tokens = 0
        previous_key: tuple[int, int, str] | None = None
        expected_document_ordinal: dict[int, int] = {}

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ContractError(
                        f"candidate segment contains duplicate JSON key {key!r}"
                    )
                result[key] = value
            return result

        for line_number, encoded in enumerate(iter_jsonl_bytes(decompressed), 1):
            try:
                document = json.loads(
                    encoded,
                    object_pairs_hook=reject_duplicates,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError(
                    f"candidate segment line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(document, Mapping):
                raise ContractError(
                    f"candidate segment line {line_number} must be an object"
                )
            require_exact_fields(
                document,
                {
                    "schema",
                    "kind",
                    "source_record_ordinal",
                    "document_ordinal",
                    "valid_tokens",
                    "payload",
                    "payload_sha256",
                },
                where=f"candidate segment line {line_number}",
            )
            if document["schema"] != CANDIDATE_ENVELOPE_SCHEMA:
                raise ContractError("candidate segment envelope schema drifted")
            if document["kind"] != kind:
                raise ContractError("candidate segment kind drifted")
            record_ordinal = require_int(
                document["source_record_ordinal"],
                where=f"candidate segment line {line_number} source_record_ordinal",
            )
            if not (
                source_record_start
                <= record_ordinal
                < source_record_start + source_record_count
            ):
                raise ContractError("candidate segment record escaped its source range")
            document_ordinal = require_int(
                document["document_ordinal"],
                where=f"candidate segment line {line_number} document_ordinal",
            )
            expected_ordinal = expected_document_ordinal.get(record_ordinal, 0)
            if document_ordinal != expected_ordinal:
                raise ContractError(
                    "candidate segment document ordinals are not contiguous"
                )
            expected_document_ordinal[record_ordinal] = expected_ordinal + 1
            token_count = require_int(
                document["valid_tokens"],
                where=f"candidate segment line {line_number} valid_tokens",
            )
            payload = document["payload"]
            if not isinstance(payload, Mapping):
                raise ContractError("candidate segment payload must be an object")
            payload_sha = require_sha256(
                document["payload_sha256"],
                where=f"candidate segment line {line_number} payload_sha256",
            )
            if canonical_sha256(payload) != payload_sha:
                raise ContractError("candidate segment payload digest drifted")
            if encoded != canonical_json_bytes(document) + b"\n":
                raise ContractError("candidate segment JSONL is not canonical")
            key = (record_ordinal, document_ordinal, payload_sha)
            if previous_key is not None and key <= previous_key:
                raise ContractError("candidate segment document order drifted")
            previous_key = key
            observed_count += 1
            observed_tokens += token_count
        if observed_count != candidate_document_count:
            raise ContractError("candidate segment document count drifted")
        if observed_tokens != valid_tokens:
            raise ContractError("candidate segment valid token count drifted")
    if _stable_local_file(path, where="candidate segment") != before:
        raise ContractError("candidate segment changed during content validation")


def _verified_publish(
    source: Path,
    uri: str,
    *,
    object_store: ObjectStore,
    scratch_root: Path,
) -> dict[str, object]:
    validate_gcs_uri(uri, where="immutable publication URI")
    size, digest = _stable_local_file(source, where="immutable publication source")
    metadata = dict(object_store.publish_if_absent(source, uri))
    if metadata.get("uri") != uri:
        raise ContractError("published object URI metadata drifted")
    generation = _positive_generation(
        metadata.get("generation"), where="published object generation"
    )
    if (
        require_int(
            metadata.get("size_bytes"), where="published object size", minimum=1
        )
        != size
    ):
        raise ContractError("published object size metadata drifted")
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lane-readback-", dir=scratch_root) as raw:
        downloaded = Path(raw) / "object"
        verified = dict(object_store.download(uri, downloaded, generation=generation))
        if (
            str(verified.get("generation")) != generation
            or require_int(
                verified.get("size_bytes"), where="readback object size", minimum=1
            )
            != size
            or _stable_local_file(downloaded, where="readback object") != (size, digest)
        ):
            raise ContractError("exact-generation object readback failed")
    return {
        "uri": uri,
        "generation": generation,
        "size_bytes": size,
        "sha256": digest,
    }


def verify_input_snapshots(
    manifest: Mapping[str, object],
    *,
    object_store: ObjectStore,
    scratch_root: Path,
) -> tuple[dict[str, object], ...]:
    """Download every input at its exact generation and verify physical bytes."""

    plan = validate_cloud_lane_manifest(manifest)
    scratch_root.mkdir(parents=True, exist_ok=True)
    verified: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="lane-inputs-", dir=scratch_root) as raw:
        root = Path(raw)
        for index, snapshot in enumerate(plan["input_snapshots"]):
            assert isinstance(snapshot, Mapping)
            destination = root / f"{index:04d}.snapshot"
            metadata = object_store.download(
                str(snapshot["uri"]),
                destination,
                generation=str(snapshot["generation"]),
            )
            if (
                str(metadata.get("generation")) != snapshot["generation"]
                or require_int(
                    metadata.get("size_bytes"),
                    where=f"snapshot {snapshot['name']} readback size",
                    minimum=1,
                )
                != snapshot["size_bytes"]
                or _stable_local_file(
                    destination, where=f"snapshot {snapshot['name']} readback"
                )
                != (snapshot["size_bytes"], snapshot["sha256"])
            ):
                raise ContractError(
                    f"input snapshot {snapshot['name']} exact-generation verification failed"
                )
            verified.append(dict(snapshot))
    return tuple(verified)


def initial_checkpoint(
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
    *,
    manifest_file_sha256: str,
) -> dict[str, object]:
    plan, job = _job_from_manifest(manifest, assignment)
    raw_manifest_sha = require_sha256(
        manifest_file_sha256, where="manifest_file_sha256"
    )
    checkpoint: dict[str, object] = {
        "schema": CLOUD_LANE_CHECKPOINT_SCHEMA,
        "status": "in_progress",
        "manifest_sha256": plan["manifest_sha256"],
        "manifest_file_sha256": raw_manifest_sha,
        "kind": plan["kind"],
        "assignment": job,
        "input_snapshot_set_sha256": plan["input_snapshot_set_sha256"],
        "producer_sha256": plan["pipeline"]["producer_sha256"],
        "target_lengths": list(TARGET_LENGTHS),
        "checkpoint_sequence": 0,
        "next_record_ordinal": job["record_start"],
        "segments": [],
        "segment_set_sha256": canonical_sha256([]),
        "previous_checkpoint_sha256": None,
        "training_ready": False,
    }
    checkpoint["checkpoint_sha256"] = cloud_lane_checkpoint_sha256(checkpoint)
    return validate_checkpoint(checkpoint, manifest=plan, assignment=job)


def _validate_segment(
    value: object,
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
    expected_ordinal: int,
    expected_start: int,
) -> dict[str, object]:
    where = f"segment[{expected_ordinal}]"
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    require_exact_fields(
        value,
        {
            "ordinal",
            "source_record_start",
            "source_record_count",
            "candidate_document_count",
            "valid_tokens",
            "uri",
            "generation",
            "size_bytes",
            "sha256",
            "format",
            "compression",
        },
        where=where,
    )
    if value["ordinal"] != expected_ordinal:
        raise ContractError(f"{where}.ordinal drifted")
    start = require_int(
        value["source_record_start"], where=f"{where}.source_record_start"
    )
    count = require_int(
        value["source_record_count"], where=f"{where}.source_record_count", minimum=1
    )
    if start != expected_start:
        raise ContractError("checkpoint segments do not cover contiguous source ranges")
    documents = require_int(
        value["candidate_document_count"],
        where=f"{where}.candidate_document_count",
        minimum=0,
    )
    valid_tokens = require_int(
        value["valid_tokens"], where=f"{where}.valid_tokens", minimum=0
    )
    digest = require_sha256(value["sha256"], where=f"{where}.sha256")
    uri = validate_gcs_uri(value["uri"], where=f"{where}.uri")
    expected_uri = segment_uri(
        manifest,
        assignment,
        segment_ordinal=expected_ordinal,
        sha256=digest,
    )
    if uri != expected_uri:
        raise ContractError(f"{where}.uri escaped its manifest namespace")
    if (
        value["format"] != CANDIDATE_FORMAT
        or value["compression"] != CANDIDATE_COMPRESSION
    ):
        raise ContractError(f"{where} format/compression drifted")
    return {
        "ordinal": expected_ordinal,
        "source_record_start": start,
        "source_record_count": count,
        "candidate_document_count": documents,
        "valid_tokens": valid_tokens,
        "uri": uri,
        "generation": _positive_generation(
            value["generation"], where=f"{where}.generation"
        ),
        "size_bytes": require_int(
            value["size_bytes"], where=f"{where}.size_bytes", minimum=1
        ),
        "sha256": digest,
        "format": CANDIDATE_FORMAT,
        "compression": CANDIDATE_COMPRESSION,
    }


def validate_checkpoint(
    value: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
) -> dict[str, object]:
    plan, job = _job_from_manifest(manifest, assignment)
    checkpoint = copy.deepcopy(dict(value))
    require_exact_fields(
        checkpoint,
        {
            "schema",
            "status",
            "manifest_sha256",
            "manifest_file_sha256",
            "kind",
            "assignment",
            "input_snapshot_set_sha256",
            "producer_sha256",
            "target_lengths",
            "checkpoint_sequence",
            "next_record_ordinal",
            "segments",
            "segment_set_sha256",
            "previous_checkpoint_sha256",
            "checkpoint_sha256",
            "training_ready",
        },
        where="cloud lane checkpoint",
    )
    if checkpoint["schema"] != CLOUD_LANE_CHECKPOINT_SCHEMA:
        raise ContractError("cloud lane checkpoint schema is unsupported")
    if (
        checkpoint["manifest_sha256"] != plan["manifest_sha256"]
        or checkpoint["kind"] != plan["kind"]
        or checkpoint["assignment"] != job
        or checkpoint["input_snapshot_set_sha256"] != plan["input_snapshot_set_sha256"]
        or checkpoint["producer_sha256"] != plan["pipeline"]["producer_sha256"]
        or checkpoint["target_lengths"] != list(TARGET_LENGTHS)
        or checkpoint["training_ready"] is not False
    ):
        raise ContractError("cloud lane checkpoint bindings drifted")
    require_sha256(checkpoint["manifest_file_sha256"], where="manifest_file_sha256")
    sequence = require_int(
        checkpoint["checkpoint_sequence"], where="checkpoint_sequence"
    )
    raw_segments = checkpoint["segments"]
    if not isinstance(raw_segments, list) or len(raw_segments) != sequence:
        raise ContractError("checkpoint sequence does not match its segment ledger")
    cursor = int(job["record_start"])
    segments: list[dict[str, object]] = []
    end = cursor + int(job["record_count"])
    for ordinal, raw in enumerate(raw_segments):
        segment = _validate_segment(
            raw,
            manifest=plan,
            assignment=job,
            expected_ordinal=ordinal,
            expected_start=cursor,
        )
        cursor += int(segment["source_record_count"])
        if cursor > end:
            raise ContractError("checkpoint segment ledger exceeds its assignment")
        segments.append(segment)
    if checkpoint["next_record_ordinal"] != cursor:
        raise ContractError("checkpoint cursor drifted from its segment ledger")
    expected_status = "complete" if cursor == end else "in_progress"
    if checkpoint["status"] != expected_status:
        raise ContractError("checkpoint status does not match source coverage")
    previous = checkpoint["previous_checkpoint_sha256"]
    if sequence == 0:
        if previous is not None:
            raise ContractError("initial checkpoint cannot name a previous checkpoint")
    else:
        require_sha256(previous, where="previous_checkpoint_sha256")
    segment_set = require_sha256(
        checkpoint["segment_set_sha256"], where="segment_set_sha256"
    )
    if segment_set != canonical_sha256(segments):
        raise ContractError("checkpoint segment set digest drifted")
    checkpoint_digest = require_sha256(
        checkpoint["checkpoint_sha256"], where="checkpoint_sha256"
    )
    if cloud_lane_checkpoint_sha256(checkpoint) != checkpoint_digest:
        raise ContractError("cloud lane checkpoint logical digest is invalid")
    checkpoint["segments"] = segments
    return checkpoint


def publish_segment(
    source: Path,
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
    checkpoint: Mapping[str, object],
    source_record_count: int,
    candidate_document_count: int,
    valid_tokens: int,
    object_store: ObjectStore,
    scratch_root: Path,
) -> dict[str, object]:
    plan, job = _job_from_manifest(manifest, assignment)
    current = validate_checkpoint(checkpoint, manifest=plan, assignment=job)
    if current["status"] != "in_progress":
        raise ContractError("cannot append a segment to a complete checkpoint")
    count = require_int(source_record_count, where="source_record_count", minimum=1)
    next_record = int(current["next_record_ordinal"])
    assignment_end = int(job["record_start"]) + int(job["record_count"])
    if next_record + count > assignment_end:
        raise ContractError("segment exceeds its assignment source range")
    documents = require_int(
        candidate_document_count, where="candidate_document_count", minimum=0
    )
    tokens = require_int(valid_tokens, where="valid_tokens", minimum=0)
    _validate_candidate_segment_file(
        source,
        kind=str(plan["kind"]),
        source_record_start=next_record,
        source_record_count=count,
        candidate_document_count=documents,
        valid_tokens=tokens,
        scratch_root=scratch_root,
    )
    _size, digest = _stable_local_file(source, where="candidate segment")
    ordinal = len(current["segments"])
    uri = segment_uri(plan, job, segment_ordinal=ordinal, sha256=digest)
    published = _verified_publish(
        source, uri, object_store=object_store, scratch_root=scratch_root
    )
    segment = {
        "ordinal": ordinal,
        "source_record_start": next_record,
        "source_record_count": count,
        "candidate_document_count": documents,
        "valid_tokens": tokens,
        **published,
        "format": CANDIDATE_FORMAT,
        "compression": CANDIDATE_COMPRESSION,
    }
    return _validate_segment(
        segment,
        manifest=plan,
        assignment=job,
        expected_ordinal=ordinal,
        expected_start=next_record,
    )


def advance_checkpoint(
    checkpoint: Mapping[str, object],
    segment: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
) -> dict[str, object]:
    plan, job = _job_from_manifest(manifest, assignment)
    previous = validate_checkpoint(checkpoint, manifest=plan, assignment=job)
    if previous["status"] != "in_progress":
        raise ContractError("cannot advance a complete checkpoint")
    ordinal = len(previous["segments"])
    normalized_segment = _validate_segment(
        segment,
        manifest=plan,
        assignment=job,
        expected_ordinal=ordinal,
        expected_start=int(previous["next_record_ordinal"]),
    )
    segments = [*previous["segments"], normalized_segment]
    cursor = int(previous["next_record_ordinal"]) + int(
        normalized_segment["source_record_count"]
    )
    end = int(job["record_start"]) + int(job["record_count"])
    if cursor > end:
        raise ContractError("advanced checkpoint exceeds its assignment")
    advanced: dict[str, object] = {
        **{
            key: previous[key]
            for key in (
                "schema",
                "manifest_sha256",
                "manifest_file_sha256",
                "kind",
                "assignment",
                "input_snapshot_set_sha256",
                "producer_sha256",
                "target_lengths",
                "training_ready",
            )
        },
        "status": "complete" if cursor == end else "in_progress",
        "checkpoint_sequence": len(segments),
        "next_record_ordinal": cursor,
        "segments": segments,
        "segment_set_sha256": canonical_sha256(segments),
        "previous_checkpoint_sha256": previous["checkpoint_sha256"],
    }
    advanced["checkpoint_sha256"] = cloud_lane_checkpoint_sha256(advanced)
    return validate_checkpoint(advanced, manifest=plan, assignment=job)


def _json_publication_descriptor(
    published: Mapping[str, object], *, logical_sha256: str
) -> dict[str, object]:
    return {
        "schema": IMMUTABLE_OBJECT_DESCRIPTOR_SCHEMA,
        "uri": published["uri"],
        "generation": published["generation"],
        "size_bytes": published["size_bytes"],
        "sha256": published["sha256"],
        "logical_sha256": require_sha256(
            logical_sha256, where="published logical_sha256"
        ),
    }


def _validate_json_publication_descriptor(
    value: object, *, where: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    require_exact_fields(
        value,
        {"schema", "uri", "generation", "size_bytes", "sha256", "logical_sha256"},
        where=where,
    )
    if value["schema"] != IMMUTABLE_OBJECT_DESCRIPTOR_SCHEMA:
        raise ContractError(f"{where} schema is unsupported")
    return {
        "schema": IMMUTABLE_OBJECT_DESCRIPTOR_SCHEMA,
        "uri": validate_gcs_uri(value["uri"], where=f"{where}.uri"),
        "generation": _positive_generation(
            value["generation"], where=f"{where}.generation"
        ),
        "size_bytes": require_int(
            value["size_bytes"], where=f"{where}.size_bytes", minimum=1
        ),
        "sha256": require_sha256(value["sha256"], where=f"{where}.sha256"),
        "logical_sha256": require_sha256(
            value["logical_sha256"], where=f"{where}.logical_sha256"
        ),
    }


def publish_manifest(
    path: Path,
    *,
    object_store: ObjectStore,
    scratch_root: Path,
) -> dict[str, object]:
    manifest, raw_sha = load_cloud_lane_manifest(path)
    uri = gcs_join(
        str(manifest["gcs_output_prefix"]),
        "lane-manifests",
        str(manifest["kind"]),
        str(manifest["manifest_sha256"]),
        f"{raw_sha}.manifest.json",
    )
    published = _verified_publish(
        path, uri, object_store=object_store, scratch_root=scratch_root
    )
    return _json_publication_descriptor(
        published, logical_sha256=str(manifest["manifest_sha256"])
    )


def publish_checkpoint(
    path: Path,
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
    object_store: ObjectStore,
    scratch_root: Path,
) -> dict[str, object]:
    raw, payload = load_json_object(path, where="cloud lane checkpoint")
    plan, job = _job_from_manifest(manifest, assignment)
    checkpoint = validate_checkpoint(payload, manifest=plan, assignment=job)
    raw_sha = hashlib.sha256(raw).hexdigest()
    uri = gcs_join(
        str(plan["gcs_output_prefix"]),
        "lane-checkpoints",
        str(plan["kind"]),
        str(plan["manifest_sha256"]),
        _assignment_namespace(job),
        f"{raw_sha}.checkpoint.json",
    )
    published = _verified_publish(
        path, uri, object_store=object_store, scratch_root=scratch_root
    )
    return _json_publication_descriptor(
        published, logical_sha256=str(checkpoint["checkpoint_sha256"])
    )


def resume_checkpoint(
    descriptor: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
    object_store: ObjectStore,
    scratch_root: Path,
) -> dict[str, object]:
    plan, job = _job_from_manifest(manifest, assignment)
    record = _validate_json_publication_descriptor(
        descriptor, where="checkpoint publication descriptor"
    )
    expected_prefix = (
        gcs_join(
            str(plan["gcs_output_prefix"]),
            "lane-checkpoints",
            str(plan["kind"]),
            str(plan["manifest_sha256"]),
            _assignment_namespace(job),
        )
        + "/"
    )
    if not str(record["uri"]).startswith(expected_prefix):
        raise ContractError("checkpoint descriptor escaped its manifest namespace")
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lane-resume-", dir=scratch_root) as raw:
        path = Path(raw) / "checkpoint.json"
        metadata = object_store.download(
            str(record["uri"]), path, generation=str(record["generation"])
        )
        if str(metadata.get("generation")) != record[
            "generation"
        ] or _stable_local_file(path, where="checkpoint readback") != (
            record["size_bytes"],
            record["sha256"],
        ):
            raise ContractError("checkpoint exact-generation readback failed")
        _raw, payload = load_json_object(path, where="resumed checkpoint")
        checkpoint = validate_checkpoint(payload, manifest=plan, assignment=job)
    if checkpoint["checkpoint_sha256"] != record["logical_sha256"]:
        raise ContractError("checkpoint descriptor logical digest drifted")
    return checkpoint


def build_completion_receipt(
    checkpoint: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
    checkpoint_publication: Mapping[str, object],
) -> dict[str, object]:
    plan, job = _job_from_manifest(manifest, assignment)
    complete = validate_checkpoint(checkpoint, manifest=plan, assignment=job)
    if complete["status"] != "complete":
        raise ContractError("completion receipt requires a complete checkpoint")
    segments = complete["segments"]
    assert isinstance(segments, list)
    pipeline = plan["pipeline"]
    assert isinstance(pipeline, Mapping)
    checkpoint_descriptor = _validate_json_publication_descriptor(
        checkpoint_publication, where="completion checkpoint publication"
    )
    expected_checkpoint_prefix = (
        gcs_join(
            str(plan["gcs_output_prefix"]),
            "lane-checkpoints",
            str(plan["kind"]),
            str(plan["manifest_sha256"]),
            _assignment_namespace(job),
        )
        + "/"
    )
    if (
        not str(checkpoint_descriptor["uri"]).startswith(expected_checkpoint_prefix)
        or checkpoint_descriptor["logical_sha256"] != complete["checkpoint_sha256"]
    ):
        raise ContractError("completion checkpoint publication binding drifted")
    receipt: dict[str, object] = {
        "schema": CLOUD_LANE_COMPLETION_SCHEMA,
        "status": "complete",
        "kind": plan["kind"],
        "manifest_sha256": plan["manifest_sha256"],
        "manifest_file_sha256": complete["manifest_file_sha256"],
        "assignment": job,
        "input_snapshot_set_sha256": plan["input_snapshot_set_sha256"],
        "checkpoint_sha256": complete["checkpoint_sha256"],
        "checkpoint_publication": checkpoint_descriptor,
        "segment_set_sha256": complete["segment_set_sha256"],
        "segments": segments,
        "totals": {
            "source_record_count": sum(
                int(segment["source_record_count"]) for segment in segments
            ),
            "candidate_document_count": sum(
                int(segment["candidate_document_count"]) for segment in segments
            ),
            "valid_tokens": sum(int(segment["valid_tokens"]) for segment in segments),
            "segment_count": len(segments),
        },
        "target_lengths": list(TARGET_LENGTHS),
        "bindings": {
            "producer_sha256": pipeline["producer_sha256"],
            "tokenizer_sha256": pipeline["tokenizer_sha256"],
            "dataset_schema_sha256": pipeline["dataset_schema_sha256"],
            "membership_policy_sha256": pipeline["membership_policy_sha256"],
        },
        "downstream_contract": {
            "output_manifest_schema": OUTPUT_MANIFEST_SCHEMA,
            "bucket_audit_schema": BUCKET_AUDIT_SCHEMA,
            "source_receipt_binding": "receipt_sha256",
            "requires_global_dedup": True,
            "requires_lossless_parquet": True,
            "requires_megatron_materialization": True,
            "requires_seal_outputs": True,
        },
        "training_ready": False,
    }
    receipt["receipt_sha256"] = cloud_lane_completion_sha256(receipt)
    return validate_completion_receipt(receipt, manifest=plan, assignment=job)


def validate_completion_receipt(
    value: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
) -> dict[str, object]:
    plan, job = _job_from_manifest(manifest, assignment)
    receipt = copy.deepcopy(dict(value))
    require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "kind",
            "manifest_sha256",
            "manifest_file_sha256",
            "assignment",
            "input_snapshot_set_sha256",
            "checkpoint_sha256",
            "checkpoint_publication",
            "segment_set_sha256",
            "segments",
            "totals",
            "target_lengths",
            "bindings",
            "downstream_contract",
            "training_ready",
            "receipt_sha256",
        },
        where="cloud lane completion receipt",
    )
    if (
        receipt["schema"] != CLOUD_LANE_COMPLETION_SCHEMA
        or receipt["status"] != "complete"
    ):
        raise ContractError("cloud lane completion schema/status is unsupported")
    if (
        receipt["kind"] != plan["kind"]
        or receipt["manifest_sha256"] != plan["manifest_sha256"]
        or receipt["assignment"] != job
        or receipt["input_snapshot_set_sha256"] != plan["input_snapshot_set_sha256"]
        or receipt["target_lengths"] != list(TARGET_LENGTHS)
        or receipt["training_ready"] is not False
    ):
        raise ContractError("cloud lane completion receipt bindings drifted")
    require_sha256(receipt["manifest_file_sha256"], where="manifest_file_sha256")
    checkpoint_sha = require_sha256(
        receipt["checkpoint_sha256"], where="checkpoint_sha256"
    )
    checkpoint_descriptor = _validate_json_publication_descriptor(
        receipt["checkpoint_publication"],
        where="completion checkpoint publication",
    )
    expected_checkpoint_prefix = (
        gcs_join(
            str(plan["gcs_output_prefix"]),
            "lane-checkpoints",
            str(plan["kind"]),
            str(plan["manifest_sha256"]),
            _assignment_namespace(job),
        )
        + "/"
    )
    if (
        not str(checkpoint_descriptor["uri"]).startswith(expected_checkpoint_prefix)
        or checkpoint_descriptor["logical_sha256"] != checkpoint_sha
    ):
        raise ContractError("completion checkpoint publication binding drifted")
    raw_segments = receipt["segments"]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ContractError("completion receipt must contain segments")
    cursor = int(job["record_start"])
    segments: list[dict[str, object]] = []
    for ordinal, raw in enumerate(raw_segments):
        segment = _validate_segment(
            raw,
            manifest=plan,
            assignment=job,
            expected_ordinal=ordinal,
            expected_start=cursor,
        )
        cursor += int(segment["source_record_count"])
        segments.append(segment)
    if cursor != int(job["record_start"]) + int(job["record_count"]):
        raise ContractError("completion receipt does not cover the full assignment")
    segment_set = require_sha256(
        receipt["segment_set_sha256"], where="segment_set_sha256"
    )
    if segment_set != canonical_sha256(segments):
        raise ContractError("completion receipt segment set digest drifted")
    totals = receipt["totals"]
    if not isinstance(totals, Mapping):
        raise ContractError("completion receipt totals must be an object")
    require_exact_fields(
        totals,
        {
            "source_record_count",
            "candidate_document_count",
            "valid_tokens",
            "segment_count",
        },
        where="completion receipt totals",
    )
    expected_totals = {
        "source_record_count": sum(
            int(item["source_record_count"]) for item in segments
        ),
        "candidate_document_count": sum(
            int(item["candidate_document_count"]) for item in segments
        ),
        "valid_tokens": sum(int(item["valid_tokens"]) for item in segments),
        "segment_count": len(segments),
    }
    for field, expected in expected_totals.items():
        if require_int(totals[field], where=f"totals.{field}") != expected:
            raise ContractError(f"completion receipt totals.{field} drifted")
    pipeline = plan["pipeline"]
    assert isinstance(pipeline, Mapping)
    expected_bindings = {
        "producer_sha256": pipeline["producer_sha256"],
        "tokenizer_sha256": pipeline["tokenizer_sha256"],
        "dataset_schema_sha256": pipeline["dataset_schema_sha256"],
        "membership_policy_sha256": pipeline["membership_policy_sha256"],
    }
    if receipt["bindings"] != expected_bindings:
        raise ContractError("completion receipt producer/schema bindings drifted")
    expected_downstream = {
        "output_manifest_schema": OUTPUT_MANIFEST_SCHEMA,
        "bucket_audit_schema": BUCKET_AUDIT_SCHEMA,
        "source_receipt_binding": "receipt_sha256",
        "requires_global_dedup": True,
        "requires_lossless_parquet": True,
        "requires_megatron_materialization": True,
        "requires_seal_outputs": True,
    }
    if receipt["downstream_contract"] != expected_downstream:
        raise ContractError("completion receipt downstream contract drifted")
    receipt_digest = require_sha256(receipt["receipt_sha256"], where="receipt_sha256")
    if cloud_lane_completion_sha256(receipt) != receipt_digest:
        raise ContractError("cloud lane completion receipt logical digest is invalid")
    receipt["segments"] = segments
    receipt["totals"] = expected_totals
    receipt["checkpoint_publication"] = checkpoint_descriptor
    return receipt


def _assignment_completion_summary(
    receipt: Mapping[str, object],
    publication: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
) -> dict[str, object]:
    plan, job = _job_from_manifest(manifest, assignment)
    complete = validate_completion_receipt(receipt, manifest=plan, assignment=job)
    descriptor = _validate_json_publication_descriptor(
        publication, where="assignment completion publication"
    )
    expected_prefix = (
        gcs_join(
            str(plan["gcs_output_prefix"]),
            "lane-receipts",
            str(plan["kind"]),
            str(plan["manifest_sha256"]),
            _assignment_namespace(job),
        )
        + "/"
    )
    if (
        not str(descriptor["uri"]).startswith(expected_prefix)
        or descriptor["logical_sha256"] != complete["receipt_sha256"]
    ):
        raise ContractError("assignment completion publication binding drifted")
    return {
        "ordinal": job["ordinal"],
        "assignment_sha256": job["assignment_sha256"],
        "receipt_sha256": complete["receipt_sha256"],
        "publication": descriptor,
        "checkpoint_sha256": complete["checkpoint_sha256"],
        "segment_set_sha256": complete["segment_set_sha256"],
        "totals": dict(complete["totals"]),
    }


def build_lane_completion_receipt(
    assignment_completions: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
) -> dict[str, object]:
    """Close one lane only after every manifest assignment has a receipt."""

    plan = validate_cloud_lane_manifest(manifest)
    raw_manifest_sha = require_sha256(
        manifest_file_sha256, where="manifest_file_sha256"
    )
    summaries: list[dict[str, object]] = []
    for receipt, publication in assignment_completions:
        if not isinstance(receipt, Mapping):
            raise ContractError("assignment completion receipt must be an object")
        assignment = receipt.get("assignment")
        if not isinstance(assignment, Mapping):
            raise ContractError("assignment completion receipt has no assignment")
        summary = _assignment_completion_summary(
            receipt,
            publication,
            manifest=plan,
            assignment=assignment,
        )
        if receipt.get("manifest_file_sha256") != raw_manifest_sha:
            raise ContractError("assignment completion raw manifest binding drifted")
        summaries.append(summary)
    summaries.sort(key=lambda item: int(item["ordinal"]))
    assignments = plan["assignments"]
    assert isinstance(assignments, list)
    if len(summaries) != len(assignments):
        raise ContractError("lane completion does not cover every assignment")
    for expected, summary in zip(assignments, summaries, strict=True):
        if (
            summary["ordinal"] != expected["ordinal"]
            or summary["assignment_sha256"] != expected["assignment_sha256"]
        ):
            raise ContractError("lane completion assignment coverage drifted")
    pipeline = plan["pipeline"]
    assert isinstance(pipeline, Mapping)
    totals = {
        "source_record_count": sum(
            int(summary["totals"]["source_record_count"]) for summary in summaries
        ),
        "candidate_document_count": sum(
            int(summary["totals"]["candidate_document_count"]) for summary in summaries
        ),
        "valid_tokens": sum(
            int(summary["totals"]["valid_tokens"]) for summary in summaries
        ),
        "segment_count": sum(
            int(summary["totals"]["segment_count"]) for summary in summaries
        ),
        "assignment_receipt_count": len(summaries),
    }
    receipt: dict[str, object] = {
        "schema": CLOUD_LANE_AGGREGATE_SCHEMA,
        "status": "complete",
        "kind": plan["kind"],
        "manifest_sha256": plan["manifest_sha256"],
        "manifest_file_sha256": raw_manifest_sha,
        "input_snapshot_set_sha256": plan["input_snapshot_set_sha256"],
        "assignment_receipt_set_sha256": canonical_sha256(summaries),
        "assignment_receipts": summaries,
        "totals": totals,
        "target_lengths": list(TARGET_LENGTHS),
        "bindings": {
            "producer_sha256": pipeline["producer_sha256"],
            "tokenizer_sha256": pipeline["tokenizer_sha256"],
            "dataset_schema_sha256": pipeline["dataset_schema_sha256"],
            "membership_policy_sha256": pipeline["membership_policy_sha256"],
        },
        "downstream_contract": {
            "output_manifest_schema": OUTPUT_MANIFEST_SCHEMA,
            "bucket_audit_schema": BUCKET_AUDIT_SCHEMA,
            "source_receipt_binding": "receipt_sha256",
            "requires_global_dedup": True,
            "requires_lossless_parquet": True,
            "requires_megatron_materialization": True,
            "requires_seal_outputs": True,
        },
        "training_ready": False,
    }
    receipt["receipt_sha256"] = cloud_lane_aggregate_sha256(receipt)
    return validate_lane_completion_receipt(receipt, manifest=plan)


def validate_lane_completion_receipt(
    value: Mapping[str, object], *, manifest: Mapping[str, object]
) -> dict[str, object]:
    plan = validate_cloud_lane_manifest(manifest)
    receipt = copy.deepcopy(dict(value))
    require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "kind",
            "manifest_sha256",
            "manifest_file_sha256",
            "input_snapshot_set_sha256",
            "assignment_receipt_set_sha256",
            "assignment_receipts",
            "totals",
            "target_lengths",
            "bindings",
            "downstream_contract",
            "training_ready",
            "receipt_sha256",
        },
        where="cloud lane aggregate completion receipt",
    )
    if (
        receipt["schema"] != CLOUD_LANE_AGGREGATE_SCHEMA
        or receipt["status"] != "complete"
    ):
        raise ContractError("cloud lane aggregate schema/status is unsupported")
    if (
        receipt["kind"] != plan["kind"]
        or receipt["manifest_sha256"] != plan["manifest_sha256"]
        or receipt["input_snapshot_set_sha256"] != plan["input_snapshot_set_sha256"]
        or receipt["target_lengths"] != list(TARGET_LENGTHS)
        or receipt["training_ready"] is not False
    ):
        raise ContractError("cloud lane aggregate bindings drifted")
    require_sha256(receipt["manifest_file_sha256"], where="manifest_file_sha256")
    raw_summaries = receipt["assignment_receipts"]
    assignments = plan["assignments"]
    assert isinstance(assignments, list)
    if not isinstance(raw_summaries, list) or len(raw_summaries) != len(assignments):
        raise ContractError("cloud lane aggregate does not cover every assignment")
    summaries: list[dict[str, object]] = []
    for expected, raw in zip(assignments, raw_summaries, strict=True):
        where = f"assignment_receipts[{expected['ordinal']}]"
        if not isinstance(raw, Mapping):
            raise ContractError(f"{where} must be an object")
        require_exact_fields(
            raw,
            {
                "ordinal",
                "assignment_sha256",
                "receipt_sha256",
                "publication",
                "checkpoint_sha256",
                "segment_set_sha256",
                "totals",
            },
            where=where,
        )
        if (
            raw["ordinal"] != expected["ordinal"]
            or raw["assignment_sha256"] != expected["assignment_sha256"]
        ):
            raise ContractError(f"{where} assignment binding drifted")
        for field in (
            "assignment_sha256",
            "receipt_sha256",
            "checkpoint_sha256",
            "segment_set_sha256",
        ):
            require_sha256(raw[field], where=f"{where}.{field}")
        descriptor = _validate_json_publication_descriptor(
            raw["publication"], where=f"{where}.publication"
        )
        expected_prefix = (
            gcs_join(
                str(plan["gcs_output_prefix"]),
                "lane-receipts",
                str(plan["kind"]),
                str(plan["manifest_sha256"]),
                _assignment_namespace(expected),
            )
            + "/"
        )
        if (
            not str(descriptor["uri"]).startswith(expected_prefix)
            or descriptor["logical_sha256"] != raw["receipt_sha256"]
        ):
            raise ContractError(f"{where} publication binding drifted")
        totals = raw["totals"]
        if not isinstance(totals, Mapping):
            raise ContractError(f"{where}.totals must be an object")
        require_exact_fields(
            totals,
            {
                "source_record_count",
                "candidate_document_count",
                "valid_tokens",
                "segment_count",
            },
            where=f"{where}.totals",
        )
        normalized_totals = {
            field: require_int(totals[field], where=f"{where}.totals.{field}")
            for field in (
                "source_record_count",
                "candidate_document_count",
                "valid_tokens",
                "segment_count",
            )
        }
        if normalized_totals["source_record_count"] != expected["record_count"]:
            raise ContractError(f"{where} source record count drifted")
        summaries.append(
            {
                "ordinal": expected["ordinal"],
                "assignment_sha256": expected["assignment_sha256"],
                "receipt_sha256": raw["receipt_sha256"],
                "publication": descriptor,
                "checkpoint_sha256": raw["checkpoint_sha256"],
                "segment_set_sha256": raw["segment_set_sha256"],
                "totals": normalized_totals,
            }
        )
    assignment_set = require_sha256(
        receipt["assignment_receipt_set_sha256"],
        where="assignment_receipt_set_sha256",
    )
    if assignment_set != canonical_sha256(summaries):
        raise ContractError("aggregate assignment receipt set digest drifted")
    totals = receipt["totals"]
    if not isinstance(totals, Mapping):
        raise ContractError("cloud lane aggregate totals must be an object")
    require_exact_fields(
        totals,
        {
            "source_record_count",
            "candidate_document_count",
            "valid_tokens",
            "segment_count",
            "assignment_receipt_count",
        },
        where="cloud lane aggregate totals",
    )
    expected_totals = {
        "source_record_count": sum(
            int(summary["totals"]["source_record_count"]) for summary in summaries
        ),
        "candidate_document_count": sum(
            int(summary["totals"]["candidate_document_count"]) for summary in summaries
        ),
        "valid_tokens": sum(
            int(summary["totals"]["valid_tokens"]) for summary in summaries
        ),
        "segment_count": sum(
            int(summary["totals"]["segment_count"]) for summary in summaries
        ),
        "assignment_receipt_count": len(summaries),
    }
    for field, expected in expected_totals.items():
        if require_int(totals[field], where=f"aggregate totals.{field}") != expected:
            raise ContractError(f"aggregate totals.{field} drifted")
    if expected_totals["source_record_count"] != plan["primary_record_count"]:
        raise ContractError("aggregate source record coverage does not close")
    pipeline = plan["pipeline"]
    assert isinstance(pipeline, Mapping)
    expected_bindings = {
        "producer_sha256": pipeline["producer_sha256"],
        "tokenizer_sha256": pipeline["tokenizer_sha256"],
        "dataset_schema_sha256": pipeline["dataset_schema_sha256"],
        "membership_policy_sha256": pipeline["membership_policy_sha256"],
    }
    if receipt["bindings"] != expected_bindings:
        raise ContractError("aggregate producer/schema bindings drifted")
    expected_downstream = {
        "output_manifest_schema": OUTPUT_MANIFEST_SCHEMA,
        "bucket_audit_schema": BUCKET_AUDIT_SCHEMA,
        "source_receipt_binding": "receipt_sha256",
        "requires_global_dedup": True,
        "requires_lossless_parquet": True,
        "requires_megatron_materialization": True,
        "requires_seal_outputs": True,
    }
    if receipt["downstream_contract"] != expected_downstream:
        raise ContractError("aggregate downstream contract drifted")
    receipt_digest = require_sha256(receipt["receipt_sha256"], where="receipt_sha256")
    if cloud_lane_aggregate_sha256(receipt) != receipt_digest:
        raise ContractError("cloud lane aggregate logical digest is invalid")
    receipt["assignment_receipts"] = summaries
    receipt["totals"] = expected_totals
    return receipt


def sealing_bindings_from_completion(
    receipt: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
) -> dict[str, str]:
    """Return the exact binding shape consumed by output manifests."""

    complete = validate_lane_completion_receipt(receipt, manifest=manifest)
    bindings = complete["bindings"]
    assert isinstance(bindings, Mapping)
    return {
        "source_receipt_sha256": str(complete["receipt_sha256"]),
        "producer_sha256": str(bindings["producer_sha256"]),
        "tokenizer_sha256": str(bindings["tokenizer_sha256"]),
        "dataset_schema_sha256": str(bindings["dataset_schema_sha256"]),
    }


def _verify_segment_objects(
    receipt: Mapping[str, object],
    *,
    object_store: ObjectStore,
    scratch_root: Path,
) -> None:
    scratch_root.mkdir(parents=True, exist_ok=True)
    segments = receipt["segments"]
    assert isinstance(segments, list)
    with tempfile.TemporaryDirectory(prefix="lane-segments-", dir=scratch_root) as raw:
        root = Path(raw)
        for segment in segments:
            assert isinstance(segment, Mapping)
            destination = root / f"{int(segment['ordinal']):05d}.jsonl.zst"
            metadata = object_store.download(
                str(segment["uri"]),
                destination,
                generation=str(segment["generation"]),
            )
            if str(metadata.get("generation")) != segment[
                "generation"
            ] or _stable_local_file(
                destination, where="completion segment readback"
            ) != (
                segment["size_bytes"],
                segment["sha256"],
            ):
                raise ContractError(
                    "completion segment exact-generation readback failed"
                )


def publish_completion_receipt(
    path: Path,
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
    object_store: ObjectStore,
    scratch_root: Path,
) -> dict[str, object]:
    """Verify every segment, then publish the immutable completion receipt last."""

    raw, payload = load_json_object(path, where="cloud lane completion receipt")
    plan, job = _job_from_manifest(manifest, assignment)
    receipt = validate_completion_receipt(payload, manifest=plan, assignment=job)
    checkpoint = resume_checkpoint(
        receipt["checkpoint_publication"],
        manifest=plan,
        assignment=job,
        object_store=object_store,
        scratch_root=scratch_root,
    )
    if (
        checkpoint["status"] != "complete"
        or checkpoint["checkpoint_sha256"] != receipt["checkpoint_sha256"]
        or checkpoint["manifest_file_sha256"] != receipt["manifest_file_sha256"]
        or checkpoint["segments"] != receipt["segments"]
    ):
        raise ContractError("published completion checkpoint does not match receipt")
    _verify_segment_objects(
        receipt, object_store=object_store, scratch_root=scratch_root
    )
    raw_sha = hashlib.sha256(raw).hexdigest()
    uri = gcs_join(
        str(plan["gcs_output_prefix"]),
        "lane-receipts",
        str(plan["kind"]),
        str(plan["manifest_sha256"]),
        _assignment_namespace(job),
        f"{raw_sha}.receipt.json",
    )
    published = _verified_publish(
        path, uri, object_store=object_store, scratch_root=scratch_root
    )
    return _json_publication_descriptor(
        published, logical_sha256=str(receipt["receipt_sha256"])
    )


def publish_lane_completion_receipt(
    path: Path,
    *,
    manifest: Mapping[str, object],
    object_store: ObjectStore,
    scratch_root: Path,
) -> dict[str, object]:
    """Revalidate all assignment artifacts, then publish the lane root last."""

    raw, payload = load_json_object(path, where="cloud lane aggregate receipt")
    plan = validate_cloud_lane_manifest(manifest)
    receipt = validate_lane_completion_receipt(payload, manifest=plan)
    summaries = receipt["assignment_receipts"]
    assignments = plan["assignments"]
    assert isinstance(summaries, list)
    assert isinstance(assignments, list)
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="lane-aggregate-", dir=scratch_root
    ) as raw_tmp:
        root = Path(raw_tmp)
        for assignment, summary in zip(assignments, summaries, strict=True):
            assert isinstance(assignment, Mapping)
            assert isinstance(summary, Mapping)
            descriptor = _validate_json_publication_descriptor(
                summary["publication"], where="aggregate assignment publication"
            )
            assignment_receipt_path = root / (
                f"{int(assignment['ordinal']):05d}.receipt.json"
            )
            metadata = object_store.download(
                str(descriptor["uri"]),
                assignment_receipt_path,
                generation=str(descriptor["generation"]),
            )
            if str(metadata.get("generation")) != descriptor[
                "generation"
            ] or _stable_local_file(
                assignment_receipt_path,
                where="aggregate assignment receipt readback",
            ) != (
                descriptor["size_bytes"],
                descriptor["sha256"],
            ):
                raise ContractError(
                    "aggregate assignment receipt exact-generation readback failed"
                )
            _assignment_raw, assignment_payload = load_json_object(
                assignment_receipt_path,
                where="aggregate assignment receipt",
            )
            assignment_receipt = validate_completion_receipt(
                assignment_payload,
                manifest=plan,
                assignment=assignment,
            )
            observed_summary = _assignment_completion_summary(
                assignment_receipt,
                descriptor,
                manifest=plan,
                assignment=assignment,
            )
            if observed_summary != summary:
                raise ContractError("aggregate assignment summary drifted")
            checkpoint = resume_checkpoint(
                assignment_receipt["checkpoint_publication"],
                manifest=plan,
                assignment=assignment,
                object_store=object_store,
                scratch_root=scratch_root,
            )
            if (
                checkpoint["status"] != "complete"
                or checkpoint["checkpoint_sha256"]
                != assignment_receipt["checkpoint_sha256"]
                or checkpoint["segments"] != assignment_receipt["segments"]
            ):
                raise ContractError("aggregate assignment checkpoint drifted")
            _verify_segment_objects(
                assignment_receipt,
                object_store=object_store,
                scratch_root=scratch_root,
            )
    raw_sha = hashlib.sha256(raw).hexdigest()
    uri = gcs_join(
        str(plan["gcs_output_prefix"]),
        "lane-completions",
        str(plan["kind"]),
        str(plan["manifest_sha256"]),
        f"{raw_sha}.completion.json",
    )
    published = _verified_publish(
        path, uri, object_store=object_store, scratch_root=scratch_root
    )
    return _json_publication_descriptor(
        published, logical_sha256=str(receipt["receipt_sha256"])
    )


__all__ = [
    "ASSIGNMENT_ALGORITHM",
    "CANDIDATE_COMPRESSION",
    "CANDIDATE_DOCUMENT_ORDER",
    "CANDIDATE_ENVELOPE_SCHEMA",
    "CANDIDATE_FORMAT",
    "CLOUD_LANE_CHECKPOINT_SCHEMA",
    "CLOUD_LANE_COMPLETION_SCHEMA",
    "CLOUD_LANE_AGGREGATE_SCHEMA",
    "CLOUD_LANE_MANIFEST_SCHEMA",
    "IMMUTABLE_OBJECT_DESCRIPTOR_SCHEMA",
    "LANE_KINDS",
    "ObjectStore",
    "advance_checkpoint",
    "assignments_for_worker",
    "build_cloud_lane_manifest",
    "build_completion_receipt",
    "build_lane_completion_receipt",
    "cloud_lane_aggregate_sha256",
    "cloud_lane_checkpoint_sha256",
    "cloud_lane_completion_sha256",
    "cloud_lane_manifest_sha256",
    "initial_checkpoint",
    "load_cloud_lane_manifest",
    "publish_checkpoint",
    "publish_completion_receipt",
    "publish_lane_completion_receipt",
    "publish_manifest",
    "publish_segment",
    "resume_checkpoint",
    "sealing_bindings_from_completion",
    "segment_uri",
    "validate_checkpoint",
    "validate_cloud_lane_manifest",
    "validate_completion_receipt",
    "validate_lane_completion_receipt",
    "verify_input_snapshots",
]
