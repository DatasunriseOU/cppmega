#!/usr/bin/env python3
"""Build one receipt-bound GCP source repair payload.

The repair input is not a hand-written repository list.  It is derived from a
frozen monitor report, monitor state, the original manifest, and the exact
terminal outcome/claim receipts.  Production preparation is intentionally
pinned to the 54 deterministic failures in source-prod-20260804-005.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    require_git_object,
    require_int,
    require_sha256,
    sha256_file,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.source_manifest import (  # noqa: E402
    DEFAULT_TARGET_LENGTHS,
    validate_source_manifest,
)
from scripts.distributed_data_prep.source_work_queue import (  # noqa: E402
    assignment_claim_uri,
    assignment_outcome_uri,
    validate_assignment_claim,
)
from scripts.prepare_gcp_source_pilot import (  # noqa: E402
    prepare_pilot,
    render_runner,
)

REPAIR_EVIDENCE_SCHEMA = "cppmega.gcp_source_repair_evidence_v1"
REPAIR_CONTRACT_SCHEMA = "cppmega.gcp_source_repair_contract_v1"
REPAIR_PAYLOAD_SCHEMA = "cppmega.gcp_source_repair_payload_v1"
REPORT_SCHEMA = "cppmega.gcp_source_run_monitor_report_v1"
STATE_SCHEMA = "cppmega.gcp_source_run_monitor_state_v1"
PRODUCTION_BASE_REPOSITORY_COUNT = 482
PRODUCTION_DETERMINISTIC_COUNT = 54
PRODUCTION_TARGET_LENGTHS = tuple(DEFAULT_TARGET_LENGTHS)
_OUTCOME_FIELDS = {
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
}
_ASSIGNMENT_FIELDS = {
    "ordinal",
    "repo",
    "project_id",
    "worker",
    "assignment_sha256",
}


@dataclass(frozen=True)
class RepairEvidence:
    base_manifest: dict[str, object]
    failed_jobs: tuple[dict[str, object], ...]
    classification: dict[str, tuple[str, ...]]
    evidence_receipt: dict[str, object]


def _regular_file(path: Path, *, where: str) -> Path:
    if path.is_symlink():
        raise ContractError(f"{where} must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ContractError(f"{where} must be a regular file: {path}")
    return resolved


def _json_file(path: Path, *, where: str) -> tuple[bytes, dict[str, object]]:
    resolved = _regular_file(path, where=where)
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{where} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be a JSON object: {path}")
    return raw, value


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _state_metadata(
    state: Mapping[str, object], *, uri: str, kind: str, path: Path
) -> dict[str, object]:
    inventory = state.get("validated_receipts")
    if not isinstance(inventory, Mapping):
        raise ContractError("monitor state validated_receipts must be an object")
    raw = inventory.get(uri)
    if not isinstance(raw, Mapping):
        raise ContractError(f"receipt is absent from pinned monitor state: {uri}")
    metadata = dict(raw)
    if metadata.get("kind") != kind:
        raise ContractError(f"receipt kind drifted for {uri}")
    generation = metadata.get("generation")
    if (
        not isinstance(generation, str)
        or not generation.isdecimal()
        or int(generation) < 1
    ):
        raise ContractError(f"receipt generation is invalid for {uri}")
    expected_sha = require_sha256(metadata.get("sha256"), where=f"{kind} SHA-256")
    if sha256_file(path) != expected_sha:
        raise ContractError(f"pinned {kind} receipt SHA-256 drifted: {uri}")
    if (
        require_int(metadata.get("size_bytes"), where=f"{kind} size", minimum=1)
        != path.stat().st_size
    ):
        raise ContractError(f"pinned {kind} receipt size drifted: {uri}")
    return {
        "uri": uri,
        "generation": generation,
        "sha256": expected_sha,
        "size_bytes": path.stat().st_size,
    }


def _completion_assignments(
    state: Mapping[str, object], *, manifest_sha256: str
) -> set[str]:
    inventory = state.get("validated_receipts")
    if not isinstance(inventory, Mapping):
        raise ContractError("monitor state validated_receipts must be an object")
    marker = f"/source-assignment-completions/{manifest_sha256}/"
    completed: set[str] = set()
    for uri, raw in inventory.items():
        if not isinstance(uri, str) or not isinstance(raw, Mapping):
            raise ContractError("monitor state receipt inventory is malformed")
        if raw.get("kind") != "assignment":
            continue
        if marker not in uri or not uri.endswith(".complete.json"):
            raise ContractError(f"assignment receipt URI drifted: {uri}")
        assignment = uri.rsplit("/", 1)[-1].removesuffix(".complete.json")
        require_sha256(assignment, where="completion assignment SHA-256")
        completed.add(assignment)
    return completed


def _fresh_assignments(report: Mapping[str, object]) -> set[str]:
    workers = report.get("workers")
    if not isinstance(workers, list):
        raise ContractError("monitor report workers must be a list")
    fresh: set[str] = set()
    for worker in workers:
        if not isinstance(worker, Mapping):
            raise ContractError("monitor worker report must be an object")
        rows = worker.get("fresh_assignment_heartbeats")
        if not isinstance(rows, list):
            raise ContractError("fresh_assignment_heartbeats must be a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ContractError("fresh assignment heartbeat must be an object")
            assignment = require_sha256(
                row.get("assignment_sha256"), where="fresh assignment SHA-256"
            )
            if assignment in fresh:
                raise ContractError("fresh assignment heartbeat is duplicated")
            fresh.add(assignment)
    return fresh


def _validate_sqlite(path: Path) -> dict[str, object]:
    resolved = _regular_file(path, where="heartbeat SQLite snapshot")
    uri = f"file:{resolved.as_posix()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            rows = connection.execute("PRAGMA integrity_check").fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise ContractError("heartbeat SQLite snapshot cannot be read") from exc
    if rows != [("ok",)]:
        raise ContractError(f"heartbeat SQLite integrity check failed: {rows[:3]}")
    return {
        "path": resolved.name,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "integrity_check": "ok",
    }


def load_repair_evidence(
    evidence_root: Path,
    *,
    expected_base_manifest_file_sha256: str,
    expected_base_manifest_sha256: str,
    expected_base_repository_count: int = PRODUCTION_BASE_REPOSITORY_COUNT,
    expected_deterministic_count: int = PRODUCTION_DETERMINISTIC_COUNT,
) -> RepairEvidence:
    """Validate and classify one frozen repair-evidence directory."""

    root = evidence_root.resolve()
    if evidence_root.is_symlink() or not root.is_dir():
        raise ContractError("repair evidence root must be a non-symlink directory")
    manifest_raw, manifest_value = _json_file(
        root / "source-manifest.json", where="base source manifest"
    )
    manifest_file_sha = _sha256_bytes(manifest_raw)
    if manifest_file_sha != require_sha256(
        expected_base_manifest_file_sha256,
        where="expected base manifest file SHA-256",
    ):
        raise ContractError("base manifest file SHA-256 differs from its pin")
    manifest = validate_source_manifest(manifest_value)
    manifest_sha = str(manifest["manifest_sha256"])
    if manifest_sha != require_sha256(
        expected_base_manifest_sha256,
        where="expected base manifest logical SHA-256",
    ):
        raise ContractError("base manifest logical SHA-256 differs from its pin")
    if int(manifest["repository_count"]) != expected_base_repository_count:
        raise ContractError("base manifest repository count differs from its pin")
    if tuple(manifest["pipeline"]["target_lengths"]) != PRODUCTION_TARGET_LENGTHS:  # type: ignore[index]
        raise ContractError("base manifest target lengths are not production-lossless")

    report_raw, report = _json_file(
        root / "watchdog.current.json", where="pinned monitor report"
    )
    state_raw, state = _json_file(
        root / "watchdog.state.json", where="pinned monitor state"
    )
    if report.get("schema") != REPORT_SCHEMA or state.get("schema") != STATE_SCHEMA:
        raise ContractError("monitor report/state schema drifted")
    if report.get("run_id") != state.get("run_id"):
        raise ContractError("monitor report/state run binding drifted")
    if report.get("checked_at_unix") != state.get("updated_at_unix"):
        raise ContractError("monitor report/state snapshot timestamps differ")
    if report.get("run_root") != manifest.get("gcs_output_prefix"):
        raise ContractError("monitor run root differs from the base manifest")
    if (
        report.get("state") != "blocked_deterministic"
        or report.get("training_ready") is not False
    ):
        raise ContractError(
            "monitor report is not a non-training-ready deterministic block"
        )
    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        raise ContractError("monitor report counts must be an object")
    exact_counts = {
        "expected_assignment_receipts": expected_base_repository_count,
        "deterministic_assignment_outcomes": expected_deterministic_count,
        "terminal_assignment_outcomes": expected_deterministic_count,
        "assignment_outcome_receipts": expected_deterministic_count,
        "transient_assignment_outcomes": 0,
    }
    for field, expected in exact_counts.items():
        if counts.get(field) != expected:
            raise ContractError(f"monitor count {field} differs from its repair pin")

    sqlite_receipt = _validate_sqlite(root / "watchdog.heartbeat.sqlite3")
    jobs = [dict(job) for job in manifest["repositories"]]  # type: ignore[index]
    jobs_by_assignment = {str(job["assignment_sha256"]): job for job in jobs}
    outcome_root = root / "receipts" / "outcomes"
    outcome_files = sorted(outcome_root.rglob("*.outcome.json"))
    if len(outcome_files) != expected_deterministic_count:
        raise ContractError(
            "local deterministic outcome receipt count differs from its pin"
        )
    all_outcome_files = [path for path in outcome_root.rglob("*") if path.is_file()]
    if outcome_files != sorted(all_outcome_files):
        raise ContractError("outcome evidence contains an unexpected file")

    failed: dict[str, dict[str, object]] = {}
    selected_receipts: list[dict[str, object]] = []
    outcome_hashes: list[str] = []
    for path in outcome_files:
        _raw, outcome = _json_file(path, where="deterministic outcome receipt")
        if set(outcome) != _OUTCOME_FIELDS:
            raise ContractError(f"deterministic outcome fields drifted: {path}")
        if (
            outcome.get("schema")
            != "cppmega.distributed_source_assignment_attempt_outcome_v1"
            or outcome.get("status") != "deterministic"
            or outcome.get("worker_exit_code") != 2
            or outcome.get("training_ready") is not False
            or outcome.get("manifest_sha256") != manifest_sha
            or outcome.get("manifest_file_sha256") != manifest_file_sha
        ):
            raise ContractError(f"deterministic outcome binding drifted: {path}")
        assignment_value = outcome.get("assignment")
        if (
            not isinstance(assignment_value, Mapping)
            or set(assignment_value) != _ASSIGNMENT_FIELDS
        ):
            raise ContractError(f"deterministic outcome assignment is invalid: {path}")
        assignment = require_sha256(
            assignment_value.get("assignment_sha256"), where="failed assignment SHA-256"
        )
        job = jobs_by_assignment.get(assignment)
        if job is None or dict(assignment_value) != {
            field: job[field] for field in _ASSIGNMENT_FIELDS
        }:
            raise ContractError(
                f"deterministic outcome names a foreign assignment: {path}"
            )
        if assignment in failed:
            raise ContractError("deterministic outcome assignments are not unique")
        attempt = require_int(outcome.get("attempt"), where="failed attempt")
        claim_sha = require_sha256(outcome.get("claim_sha256"), where="claim SHA-256")
        expected_name = f"{attempt:04d}-{claim_sha}.outcome.json"
        if path.name != expected_name or path.parent.name != assignment:
            raise ContractError(f"deterministic outcome path drifted: {path}")
        outcome_uri = assignment_outcome_uri(manifest, job, attempt, claim_sha)
        outcome_metadata = _state_metadata(
            state, uri=outcome_uri, kind="outcome", path=path
        )
        outcome_hashes.append(str(outcome_metadata["sha256"]))

        claim_uri = assignment_claim_uri(manifest, job, attempt)
        claim_path = (
            root
            / "receipts"
            / "claims"
            / manifest_sha
            / assignment
            / f"{attempt:04d}.claim.json"
        )
        _claim_raw, claim = _json_file(claim_path, where="deterministic claim receipt")
        claim_metadata = _state_metadata(
            state, uri=claim_uri, kind="claim", path=claim_path
        )
        if canonical_sha256(claim) != claim_sha:
            raise ContractError(f"claim logical SHA-256 drifted: {claim_uri}")
        validated_claim = validate_assignment_claim(
            claim,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha,
            job=job,
        )
        if outcome.get("executor") != validated_claim.get("executor") or outcome.get(
            "scheduler_instance"
        ) != validated_claim.get("scheduler_instance"):
            raise ContractError(
                f"outcome/claim executor binding drifted: {outcome_uri}"
            )
        failed[assignment] = job
        selected_receipts.append(
            {
                "assignment_sha256": assignment,
                "repo": job["repo"],
                "outcome": outcome_metadata,
                "claim": claim_metadata,
                "claim_sha256": claim_sha,
                "attempt": attempt,
                "worker_exit_code": 2,
            }
        )

    expected_inventory = hashlib.sha256(
        canonical_json_bytes(sorted(outcome_hashes))
    ).hexdigest()
    if report.get("outcome_inventory_sha256") != expected_inventory:
        raise ContractError("pinned outcome inventory digest drifted")

    completed = _completion_assignments(state, manifest_sha256=manifest_sha)
    fresh = _fresh_assignments(report)
    failed_set = set(failed)
    all_assignments = set(jobs_by_assignment)
    if not (completed | failed_set | fresh) <= all_assignments:
        raise ContractError("monitor classification contains a foreign assignment")
    if completed & failed_set or completed & fresh or failed_set & fresh:
        raise ContractError("monitor assignment classifications overlap")
    stale = all_assignments - completed - failed_set - fresh
    report_completed = counts.get("assignment_receipts")
    report_fresh = counts.get("fresh_heartbeat_assignments")
    if report_completed != len(completed) or report_fresh != len(fresh):
        raise ContractError("monitor classification counts drifted")
    if len(completed) + len(failed_set) + len(fresh) + len(stale) != len(
        all_assignments
    ):
        raise ContractError("monitor classification is not exhaustive")

    classification = {
        "success": tuple(sorted(completed)),
        "deterministic": tuple(sorted(failed_set)),
        "active": tuple(sorted(fresh)),
        "stale": tuple(sorted(stale)),
    }
    classification_repositories = {
        category: [
            {
                "assignment_sha256": assignment,
                "repo": jobs_by_assignment[assignment]["repo"],
                "project_id": jobs_by_assignment[assignment]["project_id"],
            }
            for assignment in assignments
        ]
        for category, assignments in classification.items()
    }
    evidence_receipt: dict[str, object] = {
        "schema": REPAIR_EVIDENCE_SCHEMA,
        "status": "verified",
        "training_ready": False,
        "base_run_id": report["run_id"],
        "base_run_root": report["run_root"],
        "base_manifest_sha256": manifest_sha,
        "base_manifest_file_sha256": manifest_file_sha,
        "base_code_revision": manifest["code_revision"],
        "checked_at_unix": report["checked_at_unix"],
        "snapshot": {
            "source_manifest": {
                "sha256": manifest_file_sha,
                "size_bytes": len(manifest_raw),
            },
            "watchdog_report": {
                "sha256": _sha256_bytes(report_raw),
                "size_bytes": len(report_raw),
            },
            "watchdog_state": {
                "sha256": _sha256_bytes(state_raw),
                "size_bytes": len(state_raw),
            },
            "heartbeat_sqlite": sqlite_receipt,
        },
        "counts": {
            "assignments": len(all_assignments),
            "success": len(completed),
            "deterministic": len(failed_set),
            "active": len(fresh),
            "stale": len(stale),
        },
        "classification": {key: list(value) for key, value in classification.items()},
        "classification_repositories": classification_repositories,
        "selected_receipts": sorted(
            selected_receipts, key=lambda row: str(row["assignment_sha256"])
        ),
    }
    evidence_receipt["evidence_sha256"] = canonical_sha256(evidence_receipt)
    return RepairEvidence(
        base_manifest=manifest,
        failed_jobs=tuple(failed[key] for key in sorted(failed)),
        classification=classification,
        evidence_receipt=evidence_receipt,
    )


def prepare_repair(
    *,
    repo_root: Path,
    evidence_root: Path,
    runner_template: Path,
    output_root: Path,
    gcs_output_prefix: str,
    expected_base_manifest_file_sha256: str,
    expected_base_manifest_sha256: str,
    worker_count: int,
    slots_per_worker: int,
    parse_workers_per_slot: int,
    memory_limit_gb_per_slot: float,
    cpu_budget_vcpus: int,
    memory_budget_gb: float,
) -> dict[str, object]:
    """Build an atomic source payload containing only receipt-proven failures."""

    evidence = load_repair_evidence(
        evidence_root,
        expected_base_manifest_file_sha256=expected_base_manifest_file_sha256,
        expected_base_manifest_sha256=expected_base_manifest_sha256,
    )
    new_prefix = validate_gcs_uri(
        gcs_output_prefix.rstrip("/"), where="repair GCS prefix"
    )
    base_prefix = str(evidence.base_manifest["gcs_output_prefix"]).rstrip("/")
    if (
        new_prefix == base_prefix
        or new_prefix.startswith(base_prefix + "/")
        or base_prefix.startswith(new_prefix + "/")
    ):
        raise ContractError("repair GCS prefix must be a separate namespace")
    if output_root.exists() or output_root.is_symlink():
        raise ContractError(f"repair output already exists: {output_root}")

    repositories = [
        {
            "repo": job["repo"],
            "project_id": job["project_id"],
            "source": job["source"],
        }
        for job in evidence.failed_jobs
    ]
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.", dir=output_root.parent
    ) as raw_stage:
        stage = Path(raw_stage)
        repositories_path = stage / "repair-repositories.json"
        atomic_write_json(repositories_path, {"repositories": repositories})
        payload_root = stage / "payload"
        prepare_pilot(
            repo_root=repo_root,
            repositories_path=repositories_path,
            runner_template=runner_template,
            output_root=payload_root,
            gcs_output_prefix=new_prefix,
            worker_count=worker_count,
            slots_per_worker=slots_per_worker,
            parse_workers_per_slot=parse_workers_per_slot,
            memory_limit_gb_per_slot=memory_limit_gb_per_slot,
            cpu_budget_vcpus=cpu_budget_vcpus,
            memory_budget_gb=memory_budget_gb,
        )
        _new_raw, new_manifest_value = _json_file(
            payload_root / "manifests" / "source-manifest.json",
            where="generated repair source manifest",
        )
        new_manifest = validate_source_manifest(new_manifest_value)
        if tuple(new_manifest["pipeline"]["target_lengths"]) != PRODUCTION_TARGET_LENGTHS:  # type: ignore[index]
            raise ContractError("repair target lengths drifted")
        if int(new_manifest["repository_count"]) != PRODUCTION_DETERMINISTIC_COUNT:
            raise ContractError("repair manifest is not scoped to exactly 54 failures")
        old_assignments = {
            str(job["assignment_sha256"])
            for job in evidence.base_manifest["repositories"]  # type: ignore[index]
        }
        new_assignments = {
            str(job["assignment_sha256"])
            for job in new_manifest["repositories"]  # type: ignore[index]
        }
        reused = sorted(old_assignments & new_assignments)
        if reused:
            raise ContractError(
                f"repair reused an old assignment identity: {reused[0]}"
            )
        if (
            new_manifest["gcs_output_prefix"]
            == evidence.base_manifest["gcs_output_prefix"]
        ):
            raise ContractError("repair reused the old output identity")

        old_by_project = {str(job["project_id"]): job for job in evidence.failed_jobs}
        mappings = []
        for new_job in new_manifest["repositories"]:  # type: ignore[index]
            old_job = old_by_project[str(new_job["project_id"])]
            mappings.append(
                {
                    "project_id": new_job["project_id"],
                    "repo": new_job["repo"],
                    "old_assignment_sha256": old_job["assignment_sha256"],
                    "new_assignment_sha256": new_job["assignment_sha256"],
                }
            )
        contract: dict[str, object] = {
            "schema": REPAIR_CONTRACT_SCHEMA,
            "status": "preflighted",
            "training_ready": False,
            "evidence": evidence.evidence_receipt,
            "repair": {
                "gcs_output_prefix": new_prefix,
                "manifest_sha256": new_manifest["manifest_sha256"],
                "manifest_file_sha256": sha256_file(
                    payload_root / "manifests" / "source-manifest.json"
                ),
                "code_revision": require_git_object(
                    new_manifest["code_revision"], where="repair code revision"
                ),
                "repository_count": new_manifest["repository_count"],
                "target_lengths": list(PRODUCTION_TARGET_LENGTHS),
                "old_assignment_identity_reuse": False,
                "old_output_identity_reuse": False,
                "assignment_mapping": mappings,
            },
        }
        contract["contract_sha256"] = canonical_sha256(contract)
        contract_path = payload_root / "manifests" / "repair-contract.json"
        atomic_write_json(contract_path, contract)
        contract_file_sha = sha256_file(contract_path)

        receipt_path = payload_root / "payload-receipt.json"
        _receipt_raw, receipt = _json_file(receipt_path, where="repair payload receipt")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, list):
            raise ContractError("repair payload artifacts must be a list")
        artifact_hashes = {
            str(row.get("path")): str(row.get("sha256"))
            for row in artifacts
            if isinstance(row, Mapping)
        }
        rendered = render_runner(
            _regular_file(runner_template, where="repair runner template").read_text(
                encoding="utf-8"
            ),
            {
                "bundle": artifact_hashes.get("bootstrap/cppmega.bundle", ""),
                "overlay": artifact_hashes.get(
                    "bootstrap/distributed-data-prep.tar.zst", ""
                ),
                "manifest": artifact_hashes.get("manifests/source-manifest.json", ""),
            },
            repair_contract_sha256=contract_file_sha,
        )
        runner_path = payload_root / "bootstrap" / "source-worker-runner"
        runner_path.chmod(0o755)
        runner_path.write_text(rendered, encoding="utf-8", newline="\n")
        runner_path.chmod(0o555)
        for row in artifacts:
            if (
                isinstance(row, dict)
                and row.get("path") == "bootstrap/source-worker-runner"
            ):
                row["sha256"] = sha256_file(runner_path)
                row["size_bytes"] = runner_path.stat().st_size
                break
        else:
            raise ContractError("repair payload omitted its runner artifact")
        artifacts.append(
            {
                "path": "manifests/repair-contract.json",
                "size_bytes": contract_path.stat().st_size,
                "sha256": contract_file_sha,
            }
        )
        receipt.update(
            {
                "schema": REPAIR_PAYLOAD_SCHEMA,
                "status": "ready",
                "training_ready": False,
                "base_manifest_sha256": evidence.base_manifest["manifest_sha256"],
                "base_manifest_file_sha256": expected_base_manifest_file_sha256,
                "repair_contract_sha256": contract_file_sha,
                "repair_contract_logical_sha256": contract["contract_sha256"],
                "old_assignment_identity_reuse": False,
                "old_output_identity_reuse": False,
            }
        )
        receipt["artifacts"] = sorted(artifacts, key=lambda row: str(row["path"]))
        atomic_write_json(receipt_path, receipt)
        os.replace(payload_root, output_root)
    return receipt


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--runner-template", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gcs-output-prefix")
    parser.add_argument("--expected-base-manifest-file-sha256", required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--slots-per-worker", type=int, default=2)
    parser.add_argument("--parse-workers-per-slot", type=int, default=6)
    parser.add_argument("--memory-limit-gb-per-slot", type=float, default=24.0)
    parser.add_argument("--cpu-budget-vcpus", type=int, default=16)
    parser.add_argument("--memory-budget-gb", type=float, default=56.0)
    args = parser.parse_args(argv)
    try:
        if args.preflight_only:
            if args.output_root is not None or args.gcs_output_prefix is not None:
                raise ContractError(
                    "preflight-only does not accept repair output options"
                )
            evidence = load_repair_evidence(
                args.evidence_root,
                expected_base_manifest_file_sha256=args.expected_base_manifest_file_sha256,
                expected_base_manifest_sha256=args.expected_base_manifest_sha256,
            )
            if args.preflight_receipt is not None:
                atomic_write_json(args.preflight_receipt, evidence.evidence_receipt)
            else:
                print(json.dumps(evidence.evidence_receipt, indent=2, sort_keys=True))
            return 0
        if args.output_root is None or args.gcs_output_prefix is None:
            raise ContractError(
                "repair build requires --output-root and --gcs-output-prefix"
            )
        if args.preflight_receipt is not None:
            raise ContractError("--preflight-receipt requires --preflight-only")
        prepare_repair(
            repo_root=args.repo_root,
            evidence_root=args.evidence_root,
            runner_template=args.runner_template,
            output_root=args.output_root,
            gcs_output_prefix=args.gcs_output_prefix,
            expected_base_manifest_file_sha256=args.expected_base_manifest_file_sha256,
            expected_base_manifest_sha256=args.expected_base_manifest_sha256,
            worker_count=args.worker_count,
            slots_per_worker=args.slots_per_worker,
            parse_workers_per_slot=args.parse_workers_per_slot,
            memory_limit_gb_per_slot=args.memory_limit_gb_per_slot,
            cpu_budget_vcpus=args.cpu_budget_vcpus,
            memory_budget_gb=args.memory_budget_gb,
        )
    except (ContractError, OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        parser.exit(2, f"GCP source repair preparation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "RepairEvidence",
    "load_repair_evidence",
    "prepare_repair",
]
