#!/usr/bin/env python3
"""Execute receipt-bound PR, MR, or CI assignments against immutable snapshots.

The lane-specific adapter owns record decoding and rendering.  This worker owns
the transport boundary: it verifies every input snapshot at its exact GCS
generation, pins the adapter bytes to the lane manifest, canonicalizes adapter
output, publishes one content-addressed segment per assignment, and publishes
the checkpoint and completion receipt last.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    load_json_object,
    require_exact_fields,
    require_int,
    require_nonempty,
    require_sha256,
    sha256_file,
)
from scripts.distributed_data_prep.cloud_lane import (  # noqa: E402
    CANDIDATE_ENVELOPE_SCHEMA,
    ObjectStore,
    advance_checkpoint,
    assignments_for_worker,
    build_completion_receipt,
    initial_checkpoint,
    load_cloud_lane_manifest,
    publish_checkpoint,
    publish_completion_receipt,
    publish_segment,
    validate_cloud_lane_manifest,
    validate_completion_receipt,
)
from scripts.distributed_data_prep.source_worker import (  # noqa: E402
    GcloudObjectStore,
    compress_zstd,
)


ADAPTER_REQUEST_SCHEMA = "cppmega.distributed_cloud_lane_adapter_request_v1"
ADAPTER_OUTPUT_SCHEMA = "cppmega.distributed_cloud_lane_adapter_output_v1"
WORKER_LEDGER_SCHEMA = "cppmega.distributed_cloud_lane_worker_ledger_v1"
WORKER_COMPLETION_SCHEMA = "cppmega.distributed_cloud_lane_worker_completion_v1"


def _without_digest(value: Mapping[str, object], field: str) -> dict[str, object]:
    result = copy.deepcopy(dict(value))
    result.pop(field, None)
    return result


def worker_ledger_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "ledger_sha256"))


def worker_completion_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "receipt_sha256"))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"adapter output contains duplicate key {key!r}")
        result[key] = value
    return result


def _adapter_entrypoint(command: Sequence[str]) -> Path:
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise ContractError("adapter command must contain non-empty arguments")
    candidates: list[Path] = []
    if len(command) > 1:
        candidates.append(Path(command[1]))
    candidates.append(Path(command[0]))
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve()
    raise ContractError("adapter command must execute a regular entrypoint directly or via an interpreter")


def _validate_adapter(
    command: Sequence[str], *, expected_sha256: str
) -> tuple[tuple[str, ...], Path]:
    expected = require_sha256(expected_sha256, where="adapter SHA-256")
    normalized = tuple(str(value) for value in command)
    entrypoint = _adapter_entrypoint(normalized)
    if sha256_file(entrypoint) != expected:
        raise ContractError("adapter entrypoint differs from manifest runner_sha256")
    return normalized, entrypoint


def _download_snapshots(
    manifest: Mapping[str, object],
    *,
    object_store: ObjectStore,
    input_root: Path,
) -> list[dict[str, object]]:
    import fcntl

    plan = validate_cloud_lane_manifest(manifest)
    input_root.mkdir(parents=True, exist_ok=True)
    if input_root.is_symlink() or not input_root.is_dir():
        raise ContractError("cloud lane snapshot cache must be a regular directory")
    lock_path = input_root / ".snapshot-cache.lock"
    if lock_path.is_symlink():
        raise ContractError("cloud lane snapshot cache lock must not be a symlink")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        results: list[dict[str, object]] = []
        for index, raw in enumerate(plan["input_snapshots"]):
            assert isinstance(raw, Mapping)
            snapshot = dict(raw)
            destination = input_root / f"{index:04d}-{snapshot['name']}.snapshot"
            cached = False
            if destination.exists() and not destination.is_symlink():
                before = destination.stat()
                observed_sha256 = sha256_file(destination)
                after = destination.stat()
                cached = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                ) == (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ) and (
                    after.st_size == snapshot["size_bytes"]
                    and observed_sha256 == snapshot["sha256"]
                )
            if not cached:
                descriptor, raw_stage = tempfile.mkstemp(
                    prefix=f".{destination.name}.", suffix=".download", dir=input_root
                )
                os.close(descriptor)
                stage = Path(raw_stage)
                try:
                    metadata = object_store.download(
                        str(snapshot["uri"]),
                        stage,
                        generation=str(snapshot["generation"]),
                    )
                    if (
                        str(metadata.get("generation")) != snapshot["generation"]
                        or int(metadata.get("size_bytes", -1))
                        != snapshot["size_bytes"]
                        or stage.stat().st_size != snapshot["size_bytes"]
                        or sha256_file(stage) != snapshot["sha256"]
                    ):
                        raise ContractError(
                            f"input snapshot {snapshot['name']} exact-generation verification failed"
                        )
                    os.replace(stage, destination)
                finally:
                    stage.unlink(missing_ok=True)
            results.append({**snapshot, "local_path": str(destination)})
        return results


def _snapshot_identities(snapshots: Sequence[Mapping[str, object]]) -> dict[str, tuple[int, str]]:
    identities: dict[str, tuple[int, str]] = {}
    for snapshot in snapshots:
        path = Path(str(snapshot["local_path"]))
        identities[str(snapshot["name"])] = (path.stat().st_size, sha256_file(path))
    return identities


def _build_adapter_request(
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
    snapshots: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema": ADAPTER_REQUEST_SCHEMA,
        "kind": manifest["kind"],
        "manifest_sha256": manifest["manifest_sha256"],
        "input_snapshot_set_sha256": manifest["input_snapshot_set_sha256"],
        "assignment": dict(assignment),
        "snapshots": [dict(snapshot) for snapshot in snapshots],
        "output_schema": ADAPTER_OUTPUT_SCHEMA,
        "training_ready": False,
    }


def _canonicalize_adapter_output(
    source: Path,
    destination: Path,
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
) -> tuple[int, int]:
    if source.is_symlink() or not source.is_file():
        raise ContractError("lane adapter did not emit a regular JSONL output")
    start = int(assignment["record_start"])
    end = start + int(assignment["record_count"])
    documents = 0
    valid_tokens = 0
    previous_key: tuple[int, int, str] | None = None
    next_document_ordinal: dict[int, int] = {}
    with source.open("rb") as stream, destination.open("wb") as output:
        for line_number, encoded in enumerate(stream, 1):
            if not encoded.endswith(b"\n"):
                raise ContractError(
                    f"adapter output line {line_number} is not newline terminated"
                )
            try:
                raw = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError(
                    f"adapter output line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(raw, Mapping):
                raise ContractError(f"adapter output line {line_number} is not an object")
            require_exact_fields(
                raw,
                {
                    "schema",
                    "source_record_ordinal",
                    "document_ordinal",
                    "valid_tokens",
                    "payload",
                },
                where=f"adapter output line {line_number}",
            )
            if raw["schema"] != ADAPTER_OUTPUT_SCHEMA:
                raise ContractError("adapter output schema drifted")
            record_ordinal = require_int(
                raw["source_record_ordinal"],
                where=f"adapter output line {line_number} source_record_ordinal",
            )
            if not start <= record_ordinal < end:
                raise ContractError("adapter output escaped its assigned source range")
            document_ordinal = require_int(
                raw["document_ordinal"],
                where=f"adapter output line {line_number} document_ordinal",
            )
            if document_ordinal != next_document_ordinal.get(record_ordinal, 0):
                raise ContractError("adapter document ordinals are not contiguous")
            next_document_ordinal[record_ordinal] = document_ordinal + 1
            token_count = require_int(
                raw["valid_tokens"],
                where=f"adapter output line {line_number} valid_tokens",
                minimum=1,
            )
            payload = raw["payload"]
            if not isinstance(payload, Mapping):
                raise ContractError("adapter payload must be an object")
            payload_sha = canonical_sha256(payload)
            envelope = {
                "schema": CANDIDATE_ENVELOPE_SCHEMA,
                "kind": manifest["kind"],
                "source_record_ordinal": record_ordinal,
                "document_ordinal": document_ordinal,
                "valid_tokens": token_count,
                "payload": dict(payload),
                "payload_sha256": payload_sha,
            }
            key = (record_ordinal, document_ordinal, payload_sha)
            if previous_key is not None and key <= previous_key:
                raise ContractError("adapter output is not in canonical document order")
            previous_key = key
            output.write(canonical_json_bytes(envelope) + b"\n")
            documents += 1
            valid_tokens += token_count
        output.flush()
        os.fsync(output.fileno())
    return documents, valid_tokens


def _run_adapter(
    command: Sequence[str],
    *,
    request_path: Path,
    output_path: Path,
    cwd: Path,
    env: Mapping[str, str] | None,
) -> None:
    environment = os.environ.copy()
    if env is not None:
        environment.update(env)
    completed = subprocess.run(
        [*command, "--request", str(request_path), "--output", str(output_path)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"lane adapter failed with exit code {completed.returncode}: "
            f"{completed.stderr[-8000:]}"
        )


def _initial_ledger(
    manifest: Mapping[str, object], *, manifest_file_sha256: str, worker: str
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": WORKER_LEDGER_SCHEMA,
        "status": "in_progress",
        "kind": manifest["kind"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha256,
        "worker": worker,
        "assignments": [],
        "training_ready": False,
    }
    value["ledger_sha256"] = worker_ledger_sha256(value)
    return value


def _validate_ledger(
    value: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    worker: str,
) -> dict[str, object]:
    ledger = copy.deepcopy(dict(value))
    require_exact_fields(
        ledger,
        {
            "schema",
            "status",
            "kind",
            "manifest_sha256",
            "manifest_file_sha256",
            "worker",
            "assignments",
            "training_ready",
            "ledger_sha256",
        },
        where="cloud lane worker ledger",
    )
    if (
        ledger["schema"] != WORKER_LEDGER_SCHEMA
        or ledger["status"] not in {"in_progress", "complete"}
        or ledger["kind"] != manifest["kind"]
        or ledger["manifest_sha256"] != manifest["manifest_sha256"]
        or ledger["manifest_file_sha256"] != manifest_file_sha256
        or ledger["worker"] != worker
        or ledger["training_ready"] is not False
    ):
        raise ContractError("cloud lane worker ledger binding drifted")
    if worker_ledger_sha256(ledger) != require_sha256(
        ledger["ledger_sha256"], where="worker ledger SHA-256"
    ):
        raise ContractError("cloud lane worker ledger digest drifted")
    assignments = ledger["assignments"]
    if not isinstance(assignments, list):
        raise ContractError("worker ledger assignments must be a list")
    expected = assignments_for_worker(manifest, worker)
    if len(assignments) > len(expected):
        raise ContractError("worker ledger contains too many assignments")
    normalized: list[dict[str, object]] = []
    for index, (entry, assignment) in enumerate(zip(assignments, expected, strict=False)):
        if not isinstance(entry, Mapping):
            raise ContractError(f"worker ledger assignment {index} is malformed")
        require_exact_fields(
            entry,
            {"assignment_sha256", "receipt", "publication"},
            where=f"worker ledger assignment {index}",
        )
        if entry["assignment_sha256"] != assignment["assignment_sha256"]:
            raise ContractError("worker ledger assignment order drifted")
        receipt = validate_completion_receipt(
            entry["receipt"], manifest=manifest, assignment=assignment
        )
        publication = entry["publication"]
        if not isinstance(publication, Mapping):
            raise ContractError("worker ledger publication is malformed")
        normalized.append(
            {
                "assignment_sha256": assignment["assignment_sha256"],
                "receipt": receipt,
                "publication": dict(publication),
            }
        )
    expected_status = "complete" if len(normalized) == len(expected) else "in_progress"
    if ledger["status"] != expected_status:
        raise ContractError("worker ledger status does not match assignment coverage")
    ledger["assignments"] = normalized
    return ledger


def _write_ledger(path: Path, ledger: Mapping[str, object]) -> None:
    value = copy.deepcopy(dict(ledger))
    value["ledger_sha256"] = worker_ledger_sha256(value)
    atomic_write_json(path, value)


def _load_or_create_ledger(
    path: Path,
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    worker: str,
) -> dict[str, object]:
    if not path.exists() and not path.is_symlink():
        ledger = _initial_ledger(
            manifest, manifest_file_sha256=manifest_file_sha256, worker=worker
        )
        _write_ledger(path, ledger)
        return ledger
    _raw, payload = load_json_object(path, where="cloud lane worker ledger")
    return _validate_ledger(
        payload,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        worker=worker,
    )


def _revalidate_completed_entry(
    entry: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    assignment: Mapping[str, object],
    object_store: ObjectStore,
    scratch_root: Path,
) -> None:
    receipt = entry["receipt"]
    assert isinstance(receipt, Mapping)
    with tempfile.TemporaryDirectory(prefix="lane-resume-", dir=scratch_root) as raw:
        path = Path(raw) / "receipt.json"
        atomic_write_json(path, receipt)
        observed = publish_completion_receipt(
            path,
            manifest=manifest,
            assignment=assignment,
            object_store=object_store,
            scratch_root=scratch_root,
        )
    if dict(observed) != dict(entry["publication"]):
        raise ContractError("resumed assignment publication descriptor drifted")


def run_cloud_lane_worker(
    *,
    manifest_path: Path,
    worker: str,
    adapter_command: Sequence[str],
    adapter_sha256: str,
    scratch_root: Path,
    receipt_root: Path,
    ledger_path: Path,
    object_store: ObjectStore,
    adapter_env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    manifest, manifest_file_sha256 = load_cloud_lane_manifest(manifest_path)
    expected_adapter_sha = str(manifest["pipeline"]["runner_sha256"])
    if require_sha256(adapter_sha256, where="adapter SHA-256") != expected_adapter_sha:
        raise ContractError("adapter SHA-256 differs from manifest runner_sha256")
    command, adapter_entrypoint = _validate_adapter(
        adapter_command, expected_sha256=expected_adapter_sha
    )
    assignments = assignments_for_worker(manifest, worker)
    if not assignments:
        raise ContractError(f"manifest assigns no work to {worker}")
    scratch_root = scratch_root.resolve()
    receipt_root = receipt_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    receipt_root.mkdir(parents=True, exist_ok=True)
    snapshots = _download_snapshots(
        manifest,
        object_store=object_store,
        input_root=scratch_root / "inputs" / str(manifest["manifest_sha256"]),
    )
    snapshot_identities = _snapshot_identities(snapshots)
    ledger = _load_or_create_ledger(
        ledger_path.resolve(),
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        worker=worker,
    )
    completed_entries = ledger["assignments"]
    assert isinstance(completed_entries, list)
    for entry, assignment in zip(completed_entries, assignments, strict=False):
        assert isinstance(entry, Mapping)
        _revalidate_completed_entry(
            entry,
            manifest=manifest,
            assignment=assignment,
            object_store=object_store,
            scratch_root=scratch_root,
        )

    for assignment in assignments[len(completed_entries) :]:
        with tempfile.TemporaryDirectory(prefix="lane-assignment-", dir=scratch_root) as raw:
            attempt = Path(raw)
            request_path = attempt / "request.json"
            adapter_output = attempt / "adapter-output.jsonl"
            canonical_output = attempt / "candidate.jsonl"
            compressed_output = attempt / "candidate.jsonl.zst"
            atomic_write_json(
                request_path,
                _build_adapter_request(
                    manifest=manifest, assignment=assignment, snapshots=snapshots
                ),
            )
            _run_adapter(
                command,
                request_path=request_path,
                output_path=adapter_output,
                cwd=attempt,
                env=adapter_env,
            )
            if sha256_file(adapter_entrypoint) != expected_adapter_sha:
                raise ContractError("adapter entrypoint changed during execution")
            if _snapshot_identities(snapshots) != snapshot_identities:
                raise ContractError("adapter modified an immutable input snapshot")
            document_count, valid_tokens = _canonicalize_adapter_output(
                adapter_output,
                canonical_output,
                manifest=manifest,
                assignment=assignment,
            )
            compress_zstd(canonical_output, compressed_output)
            checkpoint = initial_checkpoint(
                manifest,
                assignment,
                manifest_file_sha256=manifest_file_sha256,
            )
            segment = publish_segment(
                compressed_output,
                manifest=manifest,
                assignment=assignment,
                checkpoint=checkpoint,
                source_record_count=int(assignment["record_count"]),
                candidate_document_count=document_count,
                valid_tokens=valid_tokens,
                object_store=object_store,
                scratch_root=scratch_root,
            )
            checkpoint = advance_checkpoint(
                checkpoint, segment, manifest=manifest, assignment=assignment
            )
            checkpoint_path = attempt / "checkpoint.json"
            atomic_write_json(checkpoint_path, checkpoint)
            checkpoint_publication = publish_checkpoint(
                checkpoint_path,
                manifest=manifest,
                assignment=assignment,
                object_store=object_store,
                scratch_root=scratch_root,
            )
            receipt = build_completion_receipt(
                checkpoint,
                manifest=manifest,
                assignment=assignment,
                checkpoint_publication=checkpoint_publication,
            )
            receipt_path = attempt / "receipt.json"
            atomic_write_json(receipt_path, receipt)
            publication = publish_completion_receipt(
                receipt_path,
                manifest=manifest,
                assignment=assignment,
                object_store=object_store,
                scratch_root=scratch_root,
            )
        completed_entries.append(
            {
                "assignment_sha256": assignment["assignment_sha256"],
                "receipt": receipt,
                "publication": publication,
            }
        )
        ledger = {
            **ledger,
            "status": (
                "complete"
                if len(completed_entries) == len(assignments)
                else "in_progress"
            ),
            "assignments": completed_entries,
        }
        _write_ledger(ledger_path, ledger)
        _raw, ledger = load_json_object(ledger_path, where="cloud lane worker ledger")
        ledger = _validate_ledger(
            ledger,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            worker=worker,
        )
        completed_entries = ledger["assignments"]
        assert isinstance(completed_entries, list)

    totals = {
        "source_record_count": 0,
        "candidate_document_count": 0,
        "valid_tokens": 0,
        "assignment_receipt_count": len(completed_entries),
    }
    summaries: list[dict[str, object]] = []
    for entry in completed_entries:
        assert isinstance(entry, Mapping)
        receipt = entry["receipt"]
        assert isinstance(receipt, Mapping)
        receipt_totals = receipt["totals"]
        assert isinstance(receipt_totals, Mapping)
        for field in ("source_record_count", "candidate_document_count", "valid_tokens"):
            totals[field] += int(receipt_totals[field])
        summaries.append(
            {
                "assignment_sha256": entry["assignment_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
                "publication": entry["publication"],
            }
        )
    completion: dict[str, object] = {
        "schema": WORKER_COMPLETION_SCHEMA,
        "status": "complete",
        "kind": manifest["kind"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha256,
        "worker": worker,
        "adapter_sha256": expected_adapter_sha,
        "ledger_sha256": ledger["ledger_sha256"],
        "assignment_receipts": summaries,
        "totals": totals,
        "training_ready": False,
    }
    completion["receipt_sha256"] = worker_completion_sha256(completion)
    atomic_write_json(receipt_root / f"{worker}.complete.json", completion)
    return completion


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--receipt-root", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("adapter_command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.adapter_command)
    if command[:1] == ["--"]:
        command.pop(0)
    try:
        run_cloud_lane_worker(
            manifest_path=args.manifest,
            worker=require_nonempty(args.worker, where="worker"),
            adapter_command=command,
            adapter_sha256=args.adapter_sha256,
            scratch_root=args.scratch_root,
            receipt_root=args.receipt_root,
            ledger_path=args.ledger,
            object_store=GcloudObjectStore(),
        )
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"cloud lane worker failed: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())
