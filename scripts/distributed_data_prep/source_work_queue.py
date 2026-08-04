#!/usr/bin/env python3
"""Coordinate receipt-safe source assignments with immutable GCS leases."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    canonical_sha256,
    gcs_join,
    load_json_object,
    require_exact_fields,
    require_int,
    require_sha256,
)
from scripts.distributed_data_prep.source_worker import (
    ObjectStore,
    TransientTransportError,
    assignment_completion_uri,
)

ASSIGNMENT_CLAIM_SCHEMA = "cppmega.distributed_source_assignment_claim_v1"
ASSIGNMENT_HEARTBEAT_SCHEMA = "cppmega.distributed_source_assignment_claim_heartbeat_v1"
ASSIGNMENT_OUTCOME_SCHEMA = "cppmega.distributed_source_assignment_attempt_outcome_v1"
_EXECUTOR_FIELDS = {
    "physical_worker_index",
    "physical_worker_count",
    "slots_per_worker",
    "slot_index",
    "worker",
}
_ASSIGNMENT_FIELDS = {
    "ordinal",
    "repo",
    "project_id",
    "worker",
    "assignment_sha256",
}


@dataclass(frozen=True)
class AssignmentLease:
    job: dict[str, object]
    claim: dict[str, object]
    claim_sha256: str


@dataclass(frozen=True)
class ClaimDecision:
    state: str
    lease: AssignmentLease | None = None
    outcome: dict[str, object] | None = None


def _assignment(job: Mapping[str, object]) -> dict[str, object]:
    return {key: job[key] for key in _ASSIGNMENT_FIELDS}


def _validate_executor(value: object, *, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    executor = dict(value)
    require_exact_fields(executor, _EXECUTOR_FIELDS, where=where)
    physical_count = require_int(
        executor["physical_worker_count"],
        where=f"{where}.physical_worker_count",
        minimum=1,
    )
    physical_index = require_int(
        executor["physical_worker_index"], where=f"{where}.physical_worker_index"
    )
    slots = require_int(
        executor["slots_per_worker"], where=f"{where}.slots_per_worker", minimum=1
    )
    slot_index = require_int(executor["slot_index"], where=f"{where}.slot_index")
    if physical_index >= physical_count or slot_index >= slots:
        raise ContractError(f"{where} is outside its declared topology")
    expected_worker = f"worker-{physical_index * slots + slot_index:04d}"
    if executor["worker"] != expected_worker:
        raise ContractError(f"{where}.worker drifted from its topology")
    return executor


def _positive_seconds(value: object, *, where: str) -> int:
    return require_int(value, where=where, minimum=1)


def _stored_int(value: object, *, where: str, minimum: int = 0) -> int:
    return require_int(value, where=where, minimum=minimum)


def assignment_claim_uri(
    manifest: Mapping[str, object], job: Mapping[str, object], attempt: int
) -> str:
    attempt_index = require_int(attempt, where="claim attempt")
    if attempt_index > 9_999:
        raise ContractError("claim attempt exceeds the four-digit bound")
    return gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-assignment-claims",
        str(manifest["manifest_sha256"]),
        require_sha256(job["assignment_sha256"], where="assignment_sha256"),
        f"{attempt_index:04d}.claim.json",
    )


def assignment_heartbeat_uri(
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    attempt: int,
    claim_sha256: str,
    heartbeat_index: int,
) -> str:
    index = require_int(heartbeat_index, where="heartbeat index", minimum=1)
    return gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-assignment-heartbeats",
        str(manifest["manifest_sha256"]),
        require_sha256(job["assignment_sha256"], where="assignment_sha256"),
        f"{attempt:04d}-{require_sha256(claim_sha256, where='claim_sha256')}",
        f"{index:08d}.heartbeat.json",
    )


def assignment_outcome_uri(
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    attempt: int,
    claim_sha256: str,
) -> str:
    return gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-assignment-attempt-outcomes",
        str(manifest["manifest_sha256"]),
        require_sha256(job["assignment_sha256"], where="assignment_sha256"),
        f"{attempt:04d}-{require_sha256(claim_sha256, where='claim_sha256')}.outcome.json",
    )


def build_assignment_claim(
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    job: Mapping[str, object],
    attempt: int,
    executor: Mapping[str, object],
    scheduler_instance: str,
    now_unix_s: int,
    lease_seconds: int,
    heartbeat_seconds: int,
) -> dict[str, object]:
    created = require_int(now_unix_s, where="claim creation time", minimum=1)
    lease = _positive_seconds(lease_seconds, where="claim lease seconds")
    heartbeat = _positive_seconds(heartbeat_seconds, where="claim heartbeat seconds")
    if heartbeat >= lease:
        raise ContractError("claim heartbeat interval must be shorter than its lease")
    if (
        not scheduler_instance
        or len(scheduler_instance) > 256
        or not scheduler_instance.isascii()
    ):
        raise ContractError("scheduler instance must be 1-256 ASCII characters")
    claim: dict[str, object] = {
        "schema": ASSIGNMENT_CLAIM_SCHEMA,
        "status": "claimed",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": require_sha256(
            manifest_file_sha256, where="manifest_file_sha256"
        ),
        "assignment": _assignment(job),
        "attempt": require_int(attempt, where="claim attempt"),
        "executor": _validate_executor(executor, where="claim executor"),
        "scheduler_instance": scheduler_instance,
        "created_unix_s": created,
        "expires_unix_s": created + lease,
        "lease_seconds": lease,
        "heartbeat_seconds": heartbeat,
        "training_ready": False,
    }
    return validate_assignment_claim(
        claim,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        job=job,
    )


def validate_assignment_claim(
    claim: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    job: Mapping[str, object],
) -> dict[str, object]:
    value = dict(claim)
    require_exact_fields(
        value,
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
        where="source assignment claim",
    )
    if (
        value["schema"] != ASSIGNMENT_CLAIM_SCHEMA
        or value["status"] != "claimed"
        or value["manifest_sha256"] != manifest["manifest_sha256"]
        or value["manifest_file_sha256"] != manifest_file_sha256
        or value["training_ready"] is not False
    ):
        raise ContractError("source assignment claim binding drifted")
    if value["assignment"] != _assignment(job):
        raise ContractError("source assignment claim assignment drifted")
    require_int(value["attempt"], where="claim attempt")
    _validate_executor(value["executor"], where="claim executor")
    instance = value["scheduler_instance"]
    if (
        not isinstance(instance, str)
        or not instance
        or len(instance) > 256
        or not instance.isascii()
    ):
        raise ContractError("source assignment claim scheduler instance is invalid")
    created = require_int(
        value["created_unix_s"], where="claim creation time", minimum=1
    )
    expires = require_int(value["expires_unix_s"], where="claim expiry time", minimum=1)
    lease = _positive_seconds(value["lease_seconds"], where="claim lease seconds")
    heartbeat = _positive_seconds(
        value["heartbeat_seconds"], where="claim heartbeat seconds"
    )
    if heartbeat >= lease or expires != created + lease:
        raise ContractError("source assignment claim lease drifted")
    return value


def _read_json_if_present(
    *, object_store: ObjectStore, uri: str, verification_root: Path, where: str
) -> dict[str, object] | None:
    metadata = object_store.describe_if_present(uri)
    if metadata is None:
        return None
    generation = str(metadata.get("generation", ""))
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError(f"{where} has an invalid generation: {uri}")
    verification_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="queue-object-", dir=verification_root
    ) as raw_tmp:
        path = Path(raw_tmp) / "object.json"
        downloaded = object_store.download(uri, path, generation=generation)
        if (
            str(downloaded.get("uri")) != uri
            or str(downloaded.get("generation")) != generation
            or path.stat().st_size
            != _stored_int(metadata.get("size_bytes"), where=f"{where} object size")
        ):
            raise ContractError(f"{where} readback drifted: {uri}")
        _raw, value = load_json_object(path, where=where)
        return value


def _publish_or_read_existing(
    *,
    value: Mapping[str, object],
    uri: str,
    object_store: ObjectStore,
    verification_root: Path,
    where: str,
) -> tuple[dict[str, object], bool]:
    verification_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="queue-publish-", dir=verification_root
    ) as raw_tmp:
        local = Path(raw_tmp) / "object.json"
        atomic_write_json(local, value)
        try:
            object_store.publish_if_absent(local, uri)
        except ContractError as collision:
            existing = _read_json_if_present(
                object_store=object_store,
                uri=uri,
                verification_root=verification_root,
                where=where,
            )
            if existing is None:
                raise collision
            return existing, existing == dict(value)
    existing = _read_json_if_present(
        object_store=object_store,
        uri=uri,
        verification_root=verification_root,
        where=where,
    )
    if existing is None:
        raise ContractError(f"published {where} disappeared: {uri}")
    return existing, existing == dict(value)


def _validate_heartbeat(
    heartbeat: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    claim: Mapping[str, object],
    claim_sha256: str,
    heartbeat_index: int,
) -> dict[str, object]:
    value = dict(heartbeat)
    require_exact_fields(
        value,
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
        where="source assignment heartbeat",
    )
    interval = _stored_int(
        claim["heartbeat_seconds"], where="claim heartbeat seconds", minimum=1
    )
    scheduled = (
        _stored_int(claim["created_unix_s"], where="claim creation time", minimum=1)
        + heartbeat_index * interval
    )
    if (
        value["schema"] != ASSIGNMENT_HEARTBEAT_SCHEMA
        or value["status"] != "active"
        or value["manifest_sha256"] != manifest["manifest_sha256"]
        or value["assignment_sha256"] != job["assignment_sha256"]
        or value["attempt"] != claim["attempt"]
        or value["claim_sha256"] != claim_sha256
        or value["executor"] != claim["executor"]
        or value["scheduler_instance"] != claim["scheduler_instance"]
        or value["heartbeat_index"] != heartbeat_index
        or value["scheduled_unix_s"] != scheduled
        or value["lease_through_unix_s"]
        != scheduled
        + _stored_int(claim["lease_seconds"], where="claim lease seconds", minimum=1)
        or value["training_ready"] is not False
    ):
        raise ContractError("source assignment heartbeat binding drifted")
    return value


def publish_assignment_heartbeat(
    *,
    manifest: Mapping[str, object],
    lease: AssignmentLease,
    now_unix_s: int,
    object_store: ObjectStore,
    verification_root: Path,
) -> int | None:
    claim = lease.claim
    if _claim_has_successor(
        manifest=manifest,
        job=lease.job,
        claim=claim,
        object_store=object_store,
    ):
        raise TransientTransportError(
            "assignment claim was superseded before heartbeat publication: "
            f"{lease.job['assignment_sha256']}"
        )
    created = _stored_int(
        claim["created_unix_s"], where="claim creation time", minimum=1
    )
    interval = _stored_int(
        claim["heartbeat_seconds"], where="claim heartbeat seconds", minimum=1
    )
    lease_seconds = _stored_int(
        claim["lease_seconds"], where="claim lease seconds", minimum=1
    )
    elapsed = require_int(now_unix_s, where="heartbeat time", minimum=1) - created
    if elapsed < interval:
        return None
    index = max(1, elapsed // interval)
    scheduled = created + index * interval
    heartbeat: dict[str, object] = {
        "schema": ASSIGNMENT_HEARTBEAT_SCHEMA,
        "status": "active",
        "manifest_sha256": manifest["manifest_sha256"],
        "assignment_sha256": lease.job["assignment_sha256"],
        "attempt": claim["attempt"],
        "claim_sha256": lease.claim_sha256,
        "executor": claim["executor"],
        "scheduler_instance": claim["scheduler_instance"],
        "heartbeat_index": index,
        "scheduled_unix_s": scheduled,
        "lease_through_unix_s": scheduled + lease_seconds,
        "training_ready": False,
    }
    uri = assignment_heartbeat_uri(
        manifest,
        lease.job,
        _stored_int(claim["attempt"], where="claim attempt"),
        lease.claim_sha256,
        index,
    )
    published, identical = _publish_or_read_existing(
        value=heartbeat,
        uri=uri,
        object_store=object_store,
        verification_root=verification_root,
        where="source assignment heartbeat",
    )
    _validate_heartbeat(
        published,
        manifest=manifest,
        job=lease.job,
        claim=claim,
        claim_sha256=lease.claim_sha256,
        heartbeat_index=index,
    )
    if not identical:
        raise ContractError("source assignment heartbeat collision")
    return index


def _claim_is_live(
    *,
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    claim: Mapping[str, object],
    claim_sha256: str,
    now_unix_s: int,
    object_store: ObjectStore,
    verification_root: Path,
) -> bool:
    if (
        _stored_int(claim["expires_unix_s"], where="claim expiry time", minimum=1)
        > now_unix_s
    ):
        return True
    interval = _stored_int(
        claim["heartbeat_seconds"], where="claim heartbeat seconds", minimum=1
    )
    lease_seconds = _stored_int(
        claim["lease_seconds"], where="claim lease seconds", minimum=1
    )
    elapsed = max(
        0,
        now_unix_s
        - _stored_int(claim["created_unix_s"], where="claim creation time", minimum=1),
    )
    latest = max(1, elapsed // interval)
    window = math.ceil(lease_seconds / interval) + 1
    for index in range(latest, max(0, latest - window), -1):
        uri = assignment_heartbeat_uri(
            manifest,
            job,
            _stored_int(claim["attempt"], where="claim attempt"),
            claim_sha256,
            index,
        )
        heartbeat = _read_json_if_present(
            object_store=object_store,
            uri=uri,
            verification_root=verification_root,
            where="source assignment heartbeat",
        )
        if heartbeat is None:
            continue
        validated = _validate_heartbeat(
            heartbeat,
            manifest=manifest,
            job=job,
            claim=claim,
            claim_sha256=claim_sha256,
            heartbeat_index=index,
        )
        if (
            _stored_int(
                validated["lease_through_unix_s"],
                where="heartbeat lease-through time",
                minimum=1,
            )
            > now_unix_s
        ):
            return True
    return False


def _claim_has_successor(
    *,
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    claim: Mapping[str, object],
    object_store: ObjectStore,
) -> bool:
    attempt = _stored_int(claim["attempt"], where="claim attempt")
    if attempt >= 9_999:
        return False
    return (
        object_store.describe_if_present(
            assignment_claim_uri(manifest, job, attempt + 1)
        )
        is not None
    )


def _validate_outcome(
    outcome: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    claim: Mapping[str, object],
    claim_sha256: str,
) -> dict[str, object]:
    value = dict(outcome)
    require_exact_fields(
        value,
        {
            "schema",
            "status",
            "manifest_sha256",
            "manifest_file_sha256",
            "assignment",
            "attempt",
            "claim_sha256",
            "executor",
            "scheduler_instance",
            "worker_exit_code",
            "published_unix_s",
            "training_ready",
        },
        where="source assignment attempt outcome",
    )
    exit_code = require_int(
        value["worker_exit_code"],
        where="assignment attempt worker exit code",
        minimum=1,
    )
    if value["status"] == "transient":
        if exit_code != 75:
            raise ContractError("transient assignment outcome must bind worker exit 75")
    elif value["status"] == "deterministic":
        if exit_code == 75:
            raise ContractError(
                "deterministic assignment outcome cannot bind worker exit 75"
            )
    else:
        raise ContractError("source assignment attempt outcome status is invalid")
    if (
        value["schema"] != ASSIGNMENT_OUTCOME_SCHEMA
        or value["manifest_sha256"] != manifest["manifest_sha256"]
        or value["manifest_file_sha256"] != claim["manifest_file_sha256"]
        or value["assignment"] != _assignment(job)
        or value["attempt"] != claim["attempt"]
        or value["claim_sha256"] != claim_sha256
        or value["executor"] != claim["executor"]
        or value["scheduler_instance"] != claim["scheduler_instance"]
        or value["training_ready"] is not False
    ):
        raise ContractError("source assignment attempt outcome binding drifted")
    require_int(value["published_unix_s"], where="outcome publication time", minimum=1)
    return value


def publish_assignment_outcome(
    *,
    manifest: Mapping[str, object],
    lease: AssignmentLease,
    worker_exit_code: int,
    now_unix_s: int,
    object_store: ObjectStore,
    verification_root: Path,
) -> dict[str, object]:
    if worker_exit_code == 0:
        raise ValueError(
            "successful assignments use their completion pointer, not an outcome"
        )
    if _claim_has_successor(
        manifest=manifest,
        job=lease.job,
        claim=lease.claim,
        object_store=object_store,
    ):
        raise TransientTransportError(
            "assignment claim was superseded before attempt outcome publication: "
            f"{lease.job['assignment_sha256']}"
        )
    # A worker that lost its lease is no longer allowed to fence a newer
    # attempt with a terminal outcome.  The caller will retry the assignment
    # and retain the stale process log as local diagnostics.
    if not _claim_is_live(
        manifest=manifest,
        job=lease.job,
        claim=lease.claim,
        claim_sha256=lease.claim_sha256,
        now_unix_s=now_unix_s,
        object_store=object_store,
        verification_root=verification_root,
    ):
        raise TransientTransportError(
            "assignment lease expired before attempt outcome publication: "
            f"{lease.job['assignment_sha256']}"
        )
    status = "transient" if worker_exit_code == 75 else "deterministic"
    outcome: dict[str, object] = {
        "schema": ASSIGNMENT_OUTCOME_SCHEMA,
        "status": status,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": lease.claim["manifest_file_sha256"],
        "assignment": _assignment(lease.job),
        "attempt": lease.claim["attempt"],
        "claim_sha256": lease.claim_sha256,
        "executor": lease.claim["executor"],
        "scheduler_instance": lease.claim["scheduler_instance"],
        "worker_exit_code": require_int(
            worker_exit_code, where="assignment attempt worker exit code", minimum=1
        ),
        "published_unix_s": require_int(
            now_unix_s, where="outcome publication time", minimum=1
        ),
        "training_ready": False,
    }
    uri = assignment_outcome_uri(
        manifest,
        lease.job,
        _stored_int(lease.claim["attempt"], where="claim attempt"),
        lease.claim_sha256,
    )
    published, identical = _publish_or_read_existing(
        value=outcome,
        uri=uri,
        object_store=object_store,
        verification_root=verification_root,
        where="source assignment attempt outcome",
    )
    validated = _validate_outcome(
        published,
        manifest=manifest,
        job=lease.job,
        claim=lease.claim,
        claim_sha256=lease.claim_sha256,
    )
    if not identical:
        raise ContractError("source assignment attempt outcome collision")
    return validated


def claim_assignment(
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    job: Mapping[str, object],
    executor: Mapping[str, object],
    scheduler_instance: str,
    now_unix_s: int,
    lease_seconds: int,
    heartbeat_seconds: int,
    max_attempts: int,
    object_store: ObjectStore,
    verification_root: Path,
) -> ClaimDecision:
    """Claim one assignment, or report that it is complete or currently busy."""

    if (
        object_store.describe_if_present(assignment_completion_uri(manifest, job))
        is not None
    ):
        return ClaimDecision("complete")
    attempts = require_int(max_attempts, where="maximum claim attempts", minimum=1)
    if attempts > 10_000:
        raise ContractError("maximum claim attempts exceeds the four-digit bound")
    validated_executor = _validate_executor(executor, where="assignment executor")
    for attempt in range(attempts):
        uri = assignment_claim_uri(manifest, job, attempt)
        claim = _read_json_if_present(
            object_store=object_store,
            uri=uri,
            verification_root=verification_root,
            where="source assignment claim",
        )
        if claim is None:
            proposed = build_assignment_claim(
                manifest=manifest,
                manifest_file_sha256=manifest_file_sha256,
                job=job,
                attempt=attempt,
                executor=validated_executor,
                scheduler_instance=scheduler_instance,
                now_unix_s=now_unix_s,
                lease_seconds=lease_seconds,
                heartbeat_seconds=heartbeat_seconds,
            )
            claim, _identical = _publish_or_read_existing(
                value=proposed,
                uri=uri,
                object_store=object_store,
                verification_root=verification_root,
                where="source assignment claim",
            )
        validated_claim = validate_assignment_claim(
            claim,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            job=job,
        )
        if _stored_int(validated_claim["attempt"], where="claim attempt") != attempt:
            raise ContractError("source assignment claim URI attempt drifted")
        claim_sha256 = canonical_sha256(validated_claim)
        outcome_uri = assignment_outcome_uri(manifest, job, attempt, claim_sha256)
        outcome = _read_json_if_present(
            object_store=object_store,
            uri=outcome_uri,
            verification_root=verification_root,
            where="source assignment attempt outcome",
        )
        if outcome is not None:
            validated_outcome = _validate_outcome(
                outcome,
                manifest=manifest,
                job=job,
                claim=validated_claim,
                claim_sha256=claim_sha256,
            )
            if _claim_has_successor(
                manifest=manifest,
                job=job,
                claim=validated_claim,
                object_store=object_store,
            ):
                continue
            if validated_outcome["status"] == "deterministic":
                return ClaimDecision("deterministic", outcome=validated_outcome)
            continue
        if (
            object_store.describe_if_present(assignment_completion_uri(manifest, job))
            is not None
        ):
            return ClaimDecision("complete")
        if _claim_has_successor(
            manifest=manifest,
            job=job,
            claim=validated_claim,
            object_store=object_store,
        ):
            continue
        if (
            validated_claim["scheduler_instance"] == scheduler_instance
            and validated_claim["executor"] == validated_executor
        ):
            return ClaimDecision(
                "claimed",
                AssignmentLease(dict(job), validated_claim, claim_sha256),
            )
        if _claim_is_live(
            manifest=manifest,
            job=job,
            claim=validated_claim,
            claim_sha256=claim_sha256,
            now_unix_s=now_unix_s,
            object_store=object_store,
            verification_root=verification_root,
        ):
            return ClaimDecision("busy")
    raise ContractError(
        f"assignment exhausted {attempts} immutable claim attempts: "
        f"{job['assignment_sha256']}"
    )


__all__ = [
    "ASSIGNMENT_CLAIM_SCHEMA",
    "ASSIGNMENT_HEARTBEAT_SCHEMA",
    "ASSIGNMENT_OUTCOME_SCHEMA",
    "AssignmentLease",
    "ClaimDecision",
    "assignment_claim_uri",
    "assignment_heartbeat_uri",
    "assignment_outcome_uri",
    "build_assignment_claim",
    "claim_assignment",
    "publish_assignment_heartbeat",
    "publish_assignment_outcome",
    "validate_assignment_claim",
]
