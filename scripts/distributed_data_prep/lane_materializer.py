#!/usr/bin/env python3
"""Receipt-bound bridge from cloud-lane candidates to sealed lane artifacts.

``cloud_lane`` deliberately stops after publishing immutable candidate segments.
The candidate envelope is shared by GitHub PR, GitLab MR, and CI, while the
payload schemas and their lossless materializers are intentionally different.
This module is the narrow execution boundary between those two facts:

* every primary/membership snapshot, assignment receipt, checkpoint, and
  candidate segment is read at its recorded GCS generation before use;
* canonical candidate envelopes and payload JSONL are reconstructed in a
  stable manifest/assignment/segment order;
* a SHA-256-pinned, lane-specific adapter receives only that snapshot and a
  private output directory; and
* existing ``seal_outputs.validate_output_manifest`` code verifies every
  Parquet, Megatron prefix, sidecar, audit, zero receipt, and target length
  before an immutable local materialization receipt is published.

The adapter is intentionally explicit.  Treating the generic ``payload`` as
clang-enriched source would silently discard PR/MR/CI-specific semantics.  A
production adapter must therefore use the matching existing exporter,
materializer, packer, and Megatron converter for its payload schema.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

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
    resume_checkpoint,
    sealing_bindings_from_completion,
    validate_cloud_lane_manifest,
    validate_completion_receipt,
    validate_lane_completion_receipt,
    verify_input_snapshots,
)
from scripts.distributed_data_prep.seal_outputs import (  # noqa: E402
    OUTPUT_MANIFEST_SCHEMA,
    TARGET_LENGTHS,
    artifact_set_sha256,
    validate_output_manifest,
)


LANE_CANDIDATE_SNAPSHOT_SCHEMA = "cppmega.distributed_lane_candidate_snapshot_v1"
LANE_MATERIALIZATION_REQUEST_SCHEMA = (
    "cppmega.distributed_lane_materialization_request_v1"
)
LANE_MATERIALIZATION_CHECKPOINT_SCHEMA = (
    "cppmega.distributed_lane_materialization_checkpoint_v1"
)
LANE_MATERIALIZATION_RECEIPT_SCHEMA = (
    "cppmega.distributed_lane_materialization_receipt_v1"
)

_ADAPTER_ID_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_ATTEMPT_RE = re.compile(r"^attempt-[0-9]{6}$")
_CHECKPOINT_STATUSES = frozenset(
    {"snapshot_ready", "adapter_running", "adapter_complete", "published"}
)


def _without_digest(value: Mapping[str, object], field: str) -> dict[str, object]:
    result = dict(value)
    result.pop(field, None)
    return result


def candidate_snapshot_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "snapshot_sha256"))


def materialization_checkpoint_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "checkpoint_sha256"))


def materialization_receipt_sha256(value: Mapping[str, object]) -> str:
    return canonical_sha256(_without_digest(value, "receipt_sha256"))


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _stable_file_descriptor(
    path: Path,
    *,
    where: str,
    allow_empty: bool = False,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{where} must be a regular file: {path}")
    before = _stat_identity(path)
    if not allow_empty and before[2] < 1:
        raise ContractError(f"{where} must not be empty: {path}")
    digest = sha256_file(path)
    if _stat_identity(path) != before:
        raise ContractError(f"{where} changed while hashing: {path}")
    return {"size_bytes": before[2], "sha256": digest}


def _validate_descriptor(
    value: object,
    *,
    root: Path,
    expected_path: str,
    where: str,
    allow_empty: bool = False,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    require_exact_fields(value, {"path", "size_bytes", "sha256"}, where=where)
    relative = require_nonempty(value["path"], where=f"{where}.path")
    parsed = PurePosixPath(relative)
    if (
        relative != expected_path
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ContractError(f"{where}.path drifted")
    size = require_int(value["size_bytes"], where=f"{where}.size_bytes")
    digest = require_sha256(value["sha256"], where=f"{where}.sha256")
    path = root / relative
    observed = _stable_file_descriptor(path, where=where, allow_empty=allow_empty)
    if observed != {"size_bytes": size, "sha256": digest}:
        raise ContractError(f"{where} descriptor differs from local bytes")
    return {"path": relative, "size_bytes": size, "sha256": digest}


def _validate_output_manifest_descriptor(
    value: object, *, root: Path, where: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    require_exact_fields(
        value,
        {"path", "size_bytes", "sha256", "logical_sha256"},
        where=where,
    )
    descriptor = _validate_descriptor(
        {
            key: value[key]
            for key in ("path", "size_bytes", "sha256")
        },
        root=root,
        expected_path="output-manifest.json",
        where=where,
    )
    return {
        **descriptor,
        "logical_sha256": require_sha256(
            value["logical_sha256"], where=f"{where}.logical_sha256"
        ),
    }


def _ensure_regular_directory(path: Path, *, where: str, create: bool = False) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"{where} must be a regular directory: {path}")
    return path.resolve()


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Take a non-blocking single-writer lock for one materialization root."""

    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ContractError(f"lane materializer lock must not be a symlink: {path}")
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ContractError(
                f"lane materialization is already active for {path.parent}"
            ) from error
        yield


@contextmanager
def _new_atomic_directory(target: Path) -> Iterator[Path]:
    """Yield a sibling stage and publish only when the target is absent."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise ContractError(f"immutable output target already exists: {target}")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.", suffix=".staged", dir=target.parent
        )
    )
    published = False
    try:
        yield stage
        if target.exists() or target.is_symlink():
            raise ContractError(f"immutable output target raced: {target}")
        os.replace(stage, target)
        published = True
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"candidate JSON contains duplicate key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class AdapterSpec:
    """A pinned executable adapter invoked in a private attempt directory."""

    adapter_id: str
    argv: tuple[str, ...]
    entrypoint: Path
    entrypoint_sha256: str


def validate_adapter_spec(spec: AdapterSpec) -> dict[str, object]:
    adapter_id = require_nonempty(spec.adapter_id, where="adapter_id")
    if _ADAPTER_ID_RE.fullmatch(adapter_id) is None:
        raise ContractError("adapter_id is not canonical")
    if not spec.argv or any(
        not isinstance(argument, str) or not argument or "\x00" in argument
        for argument in spec.argv
    ):
        raise ContractError("adapter argv must be a non-empty NUL-free command")
    entrypoint = Path(spec.entrypoint).expanduser()
    if entrypoint.is_symlink() or not entrypoint.is_file():
        raise ContractError(f"adapter entrypoint must be a regular file: {entrypoint}")
    entrypoint = entrypoint.resolve()
    expected_sha = require_sha256(
        spec.entrypoint_sha256, where="adapter entrypoint_sha256"
    )
    if sha256_file(entrypoint) != expected_sha:
        raise ContractError("adapter entrypoint SHA-256 drifted")
    entrypoint_position = (
        tuple(spec.argv).index(str(entrypoint))
        if str(entrypoint) in spec.argv
        else -1
    )
    if entrypoint_position not in {0, 1}:
        raise ContractError(
            "adapter argv must execute its pinned entrypoint directly or through "
            "one interpreter argument"
        )
    command_sha = canonical_sha256(
        {
            "adapter_id": adapter_id,
            "argv": list(spec.argv),
            "entrypoint_sha256": expected_sha,
        }
    )
    return {
        "adapter_id": adapter_id,
        "argv": list(spec.argv),
        "entrypoint": entrypoint,
        "entrypoint_sha256": expected_sha,
        "command_sha256": command_sha,
    }


def make_adapter_spec(
    *, adapter_id: str, argv: Sequence[str], entrypoint: Path
) -> AdapterSpec:
    """Create a pinned adapter spec from local bytes for a new run."""

    raw_entrypoint = Path(entrypoint).expanduser()
    if raw_entrypoint.is_symlink() or not raw_entrypoint.is_file():
        raise ContractError(
            f"adapter entrypoint must be a regular file: {raw_entrypoint}"
        )
    resolved = raw_entrypoint.resolve()
    return AdapterSpec(
        adapter_id=adapter_id,
        argv=tuple(argv),
        entrypoint=resolved,
        entrypoint_sha256=sha256_file(resolved),
    )


def _load_lane_context(
    *, manifest_path: Path, completion_path: Path
) -> dict[str, object]:
    manifest_raw, manifest_payload = load_json_object(
        manifest_path, where="cloud lane materializer manifest"
    )
    manifest = validate_cloud_lane_manifest(manifest_payload)
    completion_raw, completion_payload = load_json_object(
        completion_path, where="cloud lane aggregate completion receipt"
    )
    completion = validate_lane_completion_receipt(completion_payload, manifest=manifest)
    manifest_file_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    completion_file_sha256 = hashlib.sha256(completion_raw).hexdigest()
    if completion["manifest_file_sha256"] != manifest_file_sha256:
        raise ContractError("lane completion binds different manifest bytes")
    return {
        "manifest": manifest,
        "completion": completion,
        "manifest_file_sha256": manifest_file_sha256,
        "completion_file_sha256": completion_file_sha256,
        "bindings": sealing_bindings_from_completion(completion, manifest=manifest),
    }


def _read_exact_object(
    descriptor: Mapping[str, object],
    *,
    object_store: ObjectStore,
    destination: Path,
    where: str,
) -> None:
    uri = require_nonempty(descriptor.get("uri"), where=f"{where}.uri")
    generation = require_nonempty(
        descriptor.get("generation"), where=f"{where}.generation"
    )
    expected_size = require_int(
        descriptor.get("size_bytes"), where=f"{where}.size_bytes", minimum=1
    )
    expected_sha = require_sha256(descriptor.get("sha256"), where=f"{where}.sha256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = object_store.download(uri, destination, generation=generation)
    if str(metadata.get("generation")) != generation:
        raise ContractError(f"{where} exact-generation readback drifted")
    observed = _stable_file_descriptor(destination, where=where)
    if observed != {"size_bytes": expected_size, "sha256": expected_sha}:
        raise ContractError(f"{where} exact-generation bytes drifted")


def _read_assignment_receipt(
    summary: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    completion: Mapping[str, object],
    object_store: ObjectStore,
    scratch_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    ordinal = require_int(summary.get("ordinal"), where="assignment summary ordinal")
    assignments = manifest["assignments"]
    assert isinstance(assignments, list)
    if ordinal >= len(assignments):
        raise ContractError("assignment receipt ordinal is outside its manifest")
    assignment = assignments[ordinal]
    assert isinstance(assignment, Mapping)
    publication = summary.get("publication")
    if not isinstance(publication, Mapping):
        raise ContractError("assignment receipt publication is missing")
    with tempfile.TemporaryDirectory(prefix="lane-assignment-", dir=scratch_root) as raw:
        path = Path(raw) / "receipt.json"
        _read_exact_object(
            publication,
            object_store=object_store,
            destination=path,
            where=f"assignment {ordinal} receipt",
        )
        _raw, payload = load_json_object(path, where=f"assignment {ordinal} receipt")
    receipt = validate_completion_receipt(
        payload, manifest=manifest, assignment=assignment
    )
    expected = {
        "ordinal": assignment["ordinal"],
        "assignment_sha256": assignment["assignment_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "segment_set_sha256": receipt["segment_set_sha256"],
        "totals": receipt["totals"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ContractError(f"assignment {ordinal} aggregate summary drifted")
    if receipt["manifest_file_sha256"] != completion["manifest_file_sha256"]:
        raise ContractError("assignment receipt raw manifest binding drifted")
    checkpoint = resume_checkpoint(
        receipt["checkpoint_publication"],
        manifest=manifest,
        assignment=assignment,
        object_store=object_store,
        scratch_root=scratch_root,
    )
    if (
        checkpoint["status"] != "complete"
        or checkpoint["checkpoint_sha256"] != receipt["checkpoint_sha256"]
        or checkpoint["segments"] != receipt["segments"]
    ):
        raise ContractError(f"assignment {ordinal} checkpoint drifted")
    return receipt, checkpoint


def _iter_verified_segment(
    path: Path,
    *,
    kind: str,
    source_record_start: int,
    source_record_count: int,
    expected_documents: int,
    expected_tokens: int,
) -> Iterator[tuple[bytes, Mapping[str, object], int]]:
    """Yield validated canonical candidate envelopes from one ZSTD segment."""

    before = _stable_file_descriptor(path, where="candidate segment")
    process = subprocess.Popen(
        ["zstd", "--quiet", "--decompress", "--stdout", "--", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    documents = 0
    tokens = 0
    previous_key: tuple[int, int, str] | None = None
    next_document_ordinal: dict[int, int] = {}
    try:
        for line_number, encoded in enumerate(process.stdout, 1):
            if not encoded.endswith(b"\n"):
                raise ContractError(
                    f"candidate segment line {line_number} is not newline terminated"
                )
            try:
                document = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ContractError(
                    f"candidate segment line {line_number} is invalid JSON"
                ) from error
            if not isinstance(document, Mapping):
                raise ContractError(f"candidate segment line {line_number} is not an object")
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
            if document["schema"] != CANDIDATE_ENVELOPE_SCHEMA or document["kind"] != kind:
                raise ContractError("candidate envelope schema/kind drifted")
            record_ordinal = require_int(
                document["source_record_ordinal"],
                where=f"candidate segment line {line_number} source_record_ordinal",
            )
            if not (
                source_record_start
                <= record_ordinal
                < source_record_start + source_record_count
            ):
                raise ContractError("candidate record escaped its assigned source range")
            document_ordinal = require_int(
                document["document_ordinal"],
                where=f"candidate segment line {line_number} document_ordinal",
            )
            if document_ordinal != next_document_ordinal.get(record_ordinal, 0):
                raise ContractError("candidate document ordinals are not contiguous")
            next_document_ordinal[record_ordinal] = document_ordinal + 1
            valid_tokens = require_int(
                document["valid_tokens"],
                where=f"candidate segment line {line_number} valid_tokens",
            )
            payload = document["payload"]
            if not isinstance(payload, Mapping):
                raise ContractError("candidate payload must be an object")
            payload_sha = require_sha256(
                document["payload_sha256"],
                where=f"candidate segment line {line_number} payload_sha256",
            )
            if canonical_sha256(payload) != payload_sha:
                raise ContractError("candidate payload SHA-256 drifted")
            if encoded != canonical_json_bytes(document) + b"\n":
                raise ContractError("candidate envelope JSON is not canonical")
            key = (record_ordinal, document_ordinal, payload_sha)
            if previous_key is not None and key <= previous_key:
                raise ContractError("candidate envelope order drifted")
            previous_key = key
            documents += 1
            tokens += valid_tokens
            yield encoded, payload, valid_tokens
    finally:
        process.stdout.close()
    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace")[-2000:]
        raise ContractError(f"candidate segment ZSTD decode failed: {message}")
    if documents != expected_documents or tokens != expected_tokens:
        raise ContractError("candidate segment count/token receipt drifted")
    if _stable_file_descriptor(path, where="candidate segment") != before:
        raise ContractError("candidate segment changed during decompression")


def _snapshot_directory(output_root: Path, *, kind: str, receipt_sha256: str) -> Path:
    return output_root / "candidate-snapshots" / f"{kind}-{receipt_sha256}"


def _run_directory(output_root: Path, *, kind: str, receipt_sha256: str) -> Path:
    return output_root / "runs" / f"{kind}-{receipt_sha256}"


def _result_directory(output_root: Path, *, kind: str, receipt_sha256: str) -> Path:
    return output_root / "materialized" / kind / receipt_sha256


def validate_candidate_snapshot(
    value: Mapping[str, object],
    *,
    snapshot_root: Path,
    context: Mapping[str, object],
) -> dict[str, object]:
    """Validate the local immutable canonical snapshot against a lane receipt."""

    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise ContractError(f"candidate snapshot root is invalid: {snapshot_root}")
    snapshot = dict(value)
    require_exact_fields(
        snapshot,
        {
            "schema",
            "status",
            "kind",
            "manifest_sha256",
            "manifest_file_sha256",
            "lane_completion_receipt_sha256",
            "lane_completion_file_sha256",
            "input_snapshot_set_sha256",
            "assignment_receipt_set_sha256",
            "target_lengths",
            "source_record_count",
            "candidate_document_count",
            "valid_tokens",
            "candidates",
            "payloads",
            "training_ready",
            "snapshot_sha256",
        },
        where="lane candidate snapshot",
    )
    manifest = context["manifest"]
    completion = context["completion"]
    assert isinstance(manifest, Mapping) and isinstance(completion, Mapping)
    expected_bindings = {
        "kind": manifest["kind"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": context["manifest_file_sha256"],
        "lane_completion_receipt_sha256": completion["receipt_sha256"],
        "lane_completion_file_sha256": context["completion_file_sha256"],
        "input_snapshot_set_sha256": manifest["input_snapshot_set_sha256"],
        "assignment_receipt_set_sha256": completion["assignment_receipt_set_sha256"],
        "target_lengths": list(TARGET_LENGTHS),
        "source_record_count": completion["totals"]["source_record_count"],
        "candidate_document_count": completion["totals"]["candidate_document_count"],
        "valid_tokens": completion["totals"]["valid_tokens"],
    }
    if (
        snapshot["schema"] != LANE_CANDIDATE_SNAPSHOT_SCHEMA
        or snapshot["status"] != "complete"
        or snapshot["training_ready"] is not False
    ):
        raise ContractError("candidate snapshot schema/status drifted")
    for key, expected in expected_bindings.items():
        if snapshot[key] != expected:
            raise ContractError(f"candidate snapshot {key} binding drifted")
    candidates = _validate_descriptor(
        snapshot["candidates"],
        root=snapshot_root,
        expected_path="candidates.jsonl",
        where="candidate snapshot candidates",
        allow_empty=True,
    )
    payloads = _validate_descriptor(
        snapshot["payloads"],
        root=snapshot_root,
        expected_path="payloads.jsonl",
        where="candidate snapshot payloads",
        allow_empty=True,
    )
    if candidate_snapshot_sha256(snapshot) != require_sha256(
        snapshot["snapshot_sha256"], where="candidate snapshot snapshot_sha256"
    ):
        raise ContractError("candidate snapshot logical SHA-256 drifted")
    return {**snapshot, "candidates": candidates, "payloads": payloads}


def load_candidate_snapshot(
    snapshot_root: Path, *, context: Mapping[str, object]
) -> dict[str, object]:
    raw, value = load_json_object(
        snapshot_root / "snapshot.json", where="lane candidate snapshot"
    )
    snapshot = validate_candidate_snapshot(
        value, snapshot_root=snapshot_root, context=context
    )
    descriptor = _stable_file_descriptor(
        snapshot_root / "snapshot.json", where="lane candidate snapshot manifest"
    )
    if descriptor != {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
        raise ContractError("candidate snapshot manifest changed during validation")
    return snapshot


def materialize_candidate_snapshot(
    *,
    context: Mapping[str, object],
    output_root: Path,
    scratch_root: Path,
    object_store: ObjectStore,
) -> tuple[Path, dict[str, object]]:
    """Reconstruct and validate a canonical candidate/payload JSONL snapshot."""

    manifest = context["manifest"]
    completion = context["completion"]
    assert isinstance(manifest, Mapping) and isinstance(completion, Mapping)
    kind = str(manifest["kind"])
    receipt_sha = str(completion["receipt_sha256"])
    target = _snapshot_directory(output_root, kind=kind, receipt_sha256=receipt_sha)
    if target.exists() or target.is_symlink():
        return target, load_candidate_snapshot(target, context=context)

    _ensure_regular_directory(scratch_root, where="lane materializer scratch", create=True)
    # A candidate receipt is not enough by itself: check the exact immutable
    # source/membership objects that produced it before accepting any segment.
    verify_input_snapshots(manifest, object_store=object_store, scratch_root=scratch_root)
    summaries = completion["assignment_receipts"]
    assert isinstance(summaries, list)
    with _new_atomic_directory(target) as stage:
        candidates_path = stage / "candidates.jsonl"
        payloads_path = stage / "payloads.jsonl"
        candidate_digest = hashlib.sha256()
        payload_digest = hashlib.sha256()
        candidates_size = 0
        payloads_size = 0
        documents = 0
        valid_tokens = 0
        previous_key: tuple[int, int, str] | None = None
        with candidates_path.open("wb") as candidates, payloads_path.open("wb") as payloads:
            with tempfile.TemporaryDirectory(prefix="lane-segments-", dir=scratch_root) as raw:
                temp = Path(raw)
                for summary in summaries:
                    if not isinstance(summary, Mapping):
                        raise ContractError("aggregate assignment summary is malformed")
                    receipt, _checkpoint = _read_assignment_receipt(
                        summary,
                        manifest=manifest,
                        completion=completion,
                        object_store=object_store,
                        scratch_root=scratch_root,
                    )
                    segments = receipt["segments"]
                    assert isinstance(segments, list)
                    for segment in segments:
                        assert isinstance(segment, Mapping)
                        segment_path = temp / (
                            f"{int(summary['ordinal']):05d}-{int(segment['ordinal']):05d}.jsonl.zst"
                        )
                        _read_exact_object(
                            segment,
                            object_store=object_store,
                            destination=segment_path,
                            where=(
                                f"assignment {summary['ordinal']} segment "
                                f"{segment['ordinal']}"
                            ),
                        )
                        for encoded, payload, row_tokens in _iter_verified_segment(
                            segment_path,
                            kind=kind,
                            source_record_start=int(segment["source_record_start"]),
                            source_record_count=int(segment["source_record_count"]),
                            expected_documents=int(segment["candidate_document_count"]),
                            expected_tokens=int(segment["valid_tokens"]),
                        ):
                            parsed = json.loads(encoded)
                            key = (
                                int(parsed["source_record_ordinal"]),
                                int(parsed["document_ordinal"]),
                                str(parsed["payload_sha256"]),
                            )
                            if previous_key is not None and key <= previous_key:
                                raise ContractError("global candidate document order drifted")
                            previous_key = key
                            candidates.write(encoded)
                            candidate_digest.update(encoded)
                            candidates_size += len(encoded)
                            payload_line = canonical_json_bytes(payload) + b"\n"
                            payloads.write(payload_line)
                            payload_digest.update(payload_line)
                            payloads_size += len(payload_line)
                            documents += 1
                            valid_tokens += row_tokens
                        segment_path.unlink(missing_ok=True)
            candidates.flush()
            os.fsync(candidates.fileno())
            payloads.flush()
            os.fsync(payloads.fileno())
        totals = completion["totals"]
        assert isinstance(totals, Mapping)
        if (
            documents != totals["candidate_document_count"]
            or valid_tokens != totals["valid_tokens"]
        ):
            raise ContractError("candidate snapshot does not close lane completion totals")
        snapshot: dict[str, object] = {
            "schema": LANE_CANDIDATE_SNAPSHOT_SCHEMA,
            "status": "complete",
            "kind": kind,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": context["manifest_file_sha256"],
            "lane_completion_receipt_sha256": completion["receipt_sha256"],
            "lane_completion_file_sha256": context["completion_file_sha256"],
            "input_snapshot_set_sha256": manifest["input_snapshot_set_sha256"],
            "assignment_receipt_set_sha256": completion[
                "assignment_receipt_set_sha256"
            ],
            "target_lengths": list(TARGET_LENGTHS),
            "source_record_count": totals["source_record_count"],
            "candidate_document_count": documents,
            "valid_tokens": valid_tokens,
            "candidates": {
                "path": "candidates.jsonl",
                "size_bytes": candidates_size,
                "sha256": candidate_digest.hexdigest(),
            },
            "payloads": {
                "path": "payloads.jsonl",
                "size_bytes": payloads_size,
                "sha256": payload_digest.hexdigest(),
            },
            "training_ready": False,
        }
        snapshot["snapshot_sha256"] = candidate_snapshot_sha256(snapshot)
        atomic_write_json(stage / "snapshot.json", snapshot)
        validate_candidate_snapshot(snapshot, snapshot_root=stage, context=context)
    return target, load_candidate_snapshot(target, context=context)


def _checkpoint_base(
    *, context: Mapping[str, object], snapshot: Mapping[str, object], adapter: Mapping[str, object]
) -> dict[str, object]:
    manifest = context["manifest"]
    completion = context["completion"]
    assert isinstance(manifest, Mapping) and isinstance(completion, Mapping)
    return {
        "kind": manifest["kind"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": context["manifest_file_sha256"],
        "lane_completion_receipt_sha256": completion["receipt_sha256"],
        "lane_completion_file_sha256": context["completion_file_sha256"],
        "candidate_snapshot_sha256": snapshot["snapshot_sha256"],
        "adapter_id": adapter["adapter_id"],
        "adapter_command_sha256": adapter["command_sha256"],
        "adapter_entrypoint_sha256": adapter["entrypoint_sha256"],
    }


def _build_checkpoint(
    *,
    context: Mapping[str, object],
    snapshot: Mapping[str, object],
    adapter: Mapping[str, object],
    status: str,
    attempts_started: int,
    completed_attempt: int | None,
    output: Mapping[str, object] | None,
    previous_checkpoint_sha256: str | None,
) -> dict[str, object]:
    if status not in _CHECKPOINT_STATUSES:
        raise ContractError("unsupported materialization checkpoint status")
    if attempts_started < 0:
        raise ContractError("materialization attempts_started must be non-negative")
    if completed_attempt is not None and (
        completed_attempt < 1 or completed_attempt > attempts_started
    ):
        raise ContractError("materialization completed attempt is invalid")
    if status in {"adapter_complete", "published"}:
        if completed_attempt is None or output is None:
            raise ContractError("completed materialization checkpoint lacks output")
    elif completed_attempt is not None or output is not None:
        raise ContractError("incomplete materialization checkpoint has output")
    if status == "snapshot_ready" and attempts_started != 0:
        raise ContractError("initial materialization checkpoint cannot have attempts")
    checkpoint: dict[str, object] = {
        "schema": LANE_MATERIALIZATION_CHECKPOINT_SCHEMA,
        "status": status,
        **_checkpoint_base(context=context, snapshot=snapshot, adapter=adapter),
        "attempts_started": attempts_started,
        "completed_attempt": completed_attempt,
        "output": None if output is None else dict(output),
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "training_ready": False,
    }
    checkpoint["checkpoint_sha256"] = materialization_checkpoint_sha256(checkpoint)
    return validate_materialization_checkpoint(
        checkpoint, context=context, snapshot=snapshot, adapter=adapter
    )


def validate_materialization_checkpoint(
    value: Mapping[str, object],
    *,
    context: Mapping[str, object],
    snapshot: Mapping[str, object],
    adapter: Mapping[str, object],
) -> dict[str, object]:
    checkpoint = dict(value)
    require_exact_fields(
        checkpoint,
        {
            "schema",
            "status",
            "kind",
            "manifest_sha256",
            "manifest_file_sha256",
            "lane_completion_receipt_sha256",
            "lane_completion_file_sha256",
            "candidate_snapshot_sha256",
            "adapter_id",
            "adapter_command_sha256",
            "adapter_entrypoint_sha256",
            "attempts_started",
            "completed_attempt",
            "output",
            "previous_checkpoint_sha256",
            "training_ready",
            "checkpoint_sha256",
        },
        where="lane materialization checkpoint",
    )
    if checkpoint["schema"] != LANE_MATERIALIZATION_CHECKPOINT_SCHEMA:
        raise ContractError("materialization checkpoint schema drifted")
    status = require_nonempty(checkpoint["status"], where="checkpoint status")
    if status not in _CHECKPOINT_STATUSES or checkpoint["training_ready"] is not False:
        raise ContractError("materialization checkpoint status/training binding drifted")
    if {
        key: checkpoint[key]
        for key in _checkpoint_base(context=context, snapshot=snapshot, adapter=adapter)
    } != _checkpoint_base(context=context, snapshot=snapshot, adapter=adapter):
        raise ContractError("materialization checkpoint input/adapter binding drifted")
    attempts = require_int(
        checkpoint["attempts_started"], where="checkpoint attempts_started"
    )
    completed = checkpoint["completed_attempt"]
    if completed is not None:
        completed = require_int(completed, where="checkpoint completed_attempt", minimum=1)
    output = checkpoint["output"]
    if status in {"adapter_complete", "published"}:
        if completed is None or completed > attempts or not isinstance(output, Mapping):
            raise ContractError("completed materialization checkpoint output drifted")
        require_exact_fields(
            output,
            {
                "output_manifest_sha256",
                "output_manifest_logical_sha256",
                "artifact_set_sha256",
                "artifact_count",
                "artifact_bytes",
            },
            where="checkpoint output",
        )
        for key in (
            "output_manifest_sha256",
            "output_manifest_logical_sha256",
            "artifact_set_sha256",
        ):
            require_sha256(output[key], where=f"checkpoint output {key}")
        require_int(output["artifact_count"], where="checkpoint output artifact_count")
        require_int(output["artifact_bytes"], where="checkpoint output artifact_bytes")
        output = dict(output)
    elif completed is not None or output is not None:
        raise ContractError("incomplete materialization checkpoint has output")
    if status == "snapshot_ready" and attempts != 0:
        raise ContractError("snapshot-ready checkpoint has adapter attempts")
    previous = checkpoint["previous_checkpoint_sha256"]
    if previous is not None:
        require_sha256(previous, where="previous_checkpoint_sha256")
    if materialization_checkpoint_sha256(checkpoint) != require_sha256(
        checkpoint["checkpoint_sha256"], where="checkpoint_sha256"
    ):
        raise ContractError("materialization checkpoint logical SHA-256 drifted")
    return {**checkpoint, "completed_attempt": completed, "output": output}


def _load_or_initialize_checkpoint(
    path: Path,
    *,
    context: Mapping[str, object],
    snapshot: Mapping[str, object],
    adapter: Mapping[str, object],
) -> dict[str, object]:
    if not path.exists() and not path.is_symlink():
        checkpoint = _build_checkpoint(
            context=context,
            snapshot=snapshot,
            adapter=adapter,
            status="snapshot_ready",
            attempts_started=0,
            completed_attempt=None,
            output=None,
            previous_checkpoint_sha256=None,
        )
        atomic_write_json(path, checkpoint)
        return checkpoint
    _raw, value = load_json_object(path, where="lane materialization checkpoint")
    return validate_materialization_checkpoint(
        value, context=context, snapshot=snapshot, adapter=adapter
    )


def _write_next_checkpoint(
    path: Path,
    *,
    current: Mapping[str, object],
    context: Mapping[str, object],
    snapshot: Mapping[str, object],
    adapter: Mapping[str, object],
    status: str,
    attempts_started: int,
    completed_attempt: int | None,
    output: Mapping[str, object] | None,
) -> dict[str, object]:
    checkpoint = _build_checkpoint(
        context=context,
        snapshot=snapshot,
        adapter=adapter,
        status=status,
        attempts_started=attempts_started,
        completed_attempt=completed_attempt,
        output=output,
        previous_checkpoint_sha256=str(current["checkpoint_sha256"]),
    )
    atomic_write_json(path, checkpoint)
    return checkpoint


def _inventory_regular_files(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"adapter artifact root is invalid: {root}")
    files: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ContractError(f"adapter artifact tree contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContractError(f"adapter artifact tree contains a special file: {relative}")
        files.add(relative)
    return files


def _validated_adapter_output(
    attempt_root: Path,
    *,
    context: Mapping[str, object],
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    artifacts = attempt_root / "artifacts"
    manifest_path = attempt_root / "output-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ContractError("adapter did not emit output-manifest.json")
    raw, output_value = load_json_object(
        manifest_path, where="adapter output manifest"
    )
    output_manifest, records = validate_output_manifest(
        output_value, artifact_root=artifacts
    )
    raw_sha = hashlib.sha256(raw).hexdigest()
    manifest = context["manifest"]
    assert isinstance(manifest, Mapping)
    if output_manifest["kind"] != manifest["kind"]:
        raise ContractError("adapter output manifest kind differs from cloud lane")
    bindings = context["bindings"]
    assert isinstance(bindings, Mapping)
    if output_manifest["bindings"] != bindings:
        raise ContractError("adapter output manifest receipt/tokenizer/schema bindings drifted")
    buckets = output_manifest["buckets"]
    assert isinstance(buckets, list)
    output_valid_tokens = sum(
        int(bucket["counts"]["valid_tokens"])
        for bucket in buckets
        if bucket["status"] == "materialized"
    )
    if output_valid_tokens != snapshot["valid_tokens"]:
        raise ContractError(
            "adapter output valid token count does not losslessly close "
            "the immutable candidate snapshot"
        )
    expected_files = {str(record["path"]) for record in records}
    if _inventory_regular_files(artifacts) != expected_files:
        raise ContractError("adapter artifact tree has unreceipted or missing files")
    manifest_descriptor = _stable_file_descriptor(
        manifest_path, where="adapter output manifest"
    )
    if manifest_descriptor["sha256"] != raw_sha:
        raise ContractError("adapter output manifest changed during verification")
    artifact_digest = artifact_set_sha256(records)
    return {
        "manifest": output_manifest,
        "manifest_descriptor": {
            "path": "output-manifest.json",
            **manifest_descriptor,
            "logical_sha256": output_manifest["manifest_sha256"],
        },
        "artifact_set_sha256": artifact_digest,
        "artifact_count": len(records),
        "artifact_bytes": sum(int(record["size"]) for record in records),
    }


def _checkpoint_output(validated: Mapping[str, object]) -> dict[str, object]:
    descriptor = validated["manifest_descriptor"]
    assert isinstance(descriptor, Mapping)
    return {
        "output_manifest_sha256": descriptor["sha256"],
        "output_manifest_logical_sha256": descriptor["logical_sha256"],
        "artifact_set_sha256": validated["artifact_set_sha256"],
        "artifact_count": validated["artifact_count"],
        "artifact_bytes": validated["artifact_bytes"],
    }


def _assert_checkpoint_output(
    checkpoint: Mapping[str, object], validated: Mapping[str, object]
) -> None:
    output = checkpoint["output"]
    if not isinstance(output, Mapping) or dict(output) != _checkpoint_output(validated):
        raise ContractError("completed materialization checkpoint output drifted")


def _build_adapter_request(
    *,
    context: Mapping[str, object],
    snapshot_root: Path,
    snapshot: Mapping[str, object],
    adapter: Mapping[str, object],
) -> dict[str, object]:
    manifest = context["manifest"]
    completion = context["completion"]
    assert isinstance(manifest, Mapping) and isinstance(completion, Mapping)
    return {
        "schema": LANE_MATERIALIZATION_REQUEST_SCHEMA,
        "kind": manifest["kind"],
        "target_lengths": list(TARGET_LENGTHS),
        "lane": {
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": context["manifest_file_sha256"],
            "completion_receipt_sha256": completion["receipt_sha256"],
            "completion_file_sha256": context["completion_file_sha256"],
            "input_snapshot_set_sha256": manifest["input_snapshot_set_sha256"],
        },
        "bindings": dict(context["bindings"]),
        "candidate_snapshot": {
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "root": str(snapshot_root),
            "candidates": dict(snapshot["candidates"]),
            "payloads": dict(snapshot["payloads"]),
            "source_record_count": snapshot["source_record_count"],
            "candidate_document_count": snapshot["candidate_document_count"],
            "valid_tokens": snapshot["valid_tokens"],
        },
        "adapter": {
            "adapter_id": adapter["adapter_id"],
            "command_sha256": adapter["command_sha256"],
            "entrypoint_sha256": adapter["entrypoint_sha256"],
        },
        "output_contract": {
            "output_manifest_schema": OUTPUT_MANIFEST_SCHEMA,
            "artifact_root": "artifacts",
            "output_manifest": "output-manifest.json",
            "training_ready": False,
            "requires_global_seal_outputs": True,
        },
    }


def _run_adapter(
    attempt_root: Path,
    *,
    request: Mapping[str, object],
    snapshot_root: Path,
    adapter: Mapping[str, object],
) -> None:
    request_path = attempt_root / "materialization-request.json"
    atomic_write_json(request_path, request)
    artifacts = attempt_root / "artifacts"
    artifacts.mkdir()
    stdout_path = attempt_root / "adapter.stdout.log"
    stderr_path = attempt_root / "adapter.stderr.log"
    environment = dict(os.environ)
    environment.update(
        {
            "CPPMEGA_LANE_REQUEST": str(request_path),
            "CPPMEGA_LANE_CANDIDATES": str(snapshot_root / "candidates.jsonl"),
            "CPPMEGA_LANE_PAYLOADS": str(snapshot_root / "payloads.jsonl"),
            "CPPMEGA_LANE_ARTIFACT_ROOT": str(artifacts),
            "CPPMEGA_LANE_OUTPUT_MANIFEST": str(attempt_root / "output-manifest.json"),
        }
    )
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            list(adapter["argv"]),
            cwd=attempt_root,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
        stdout.flush()
        os.fsync(stdout.fileno())
        stderr.flush()
        os.fsync(stderr.fileno())
    if completed.returncode != 0:
        raise RuntimeError(
            f"lane adapter {adapter['adapter_id']} failed with exit {completed.returncode}; "
            f"see {stderr_path}"
        )


def _build_materialization_receipt(
    *,
    context: Mapping[str, object],
    snapshot: Mapping[str, object],
    adapter: Mapping[str, object],
    validated: Mapping[str, object],
) -> dict[str, object]:
    manifest = context["manifest"]
    completion = context["completion"]
    descriptor = validated["manifest_descriptor"]
    assert isinstance(manifest, Mapping) and isinstance(completion, Mapping)
    assert isinstance(descriptor, Mapping)
    receipt: dict[str, object] = {
        "schema": LANE_MATERIALIZATION_RECEIPT_SCHEMA,
        "status": "verified",
        "kind": manifest["kind"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": context["manifest_file_sha256"],
        "lane_completion_receipt_sha256": completion["receipt_sha256"],
        "lane_completion_file_sha256": context["completion_file_sha256"],
        "candidate_snapshot_sha256": snapshot["snapshot_sha256"],
        "input_snapshot_set_sha256": manifest["input_snapshot_set_sha256"],
        "target_lengths": list(TARGET_LENGTHS),
        "bindings": dict(context["bindings"]),
        "adapter": {
            "adapter_id": adapter["adapter_id"],
            "command_sha256": adapter["command_sha256"],
            "entrypoint_sha256": adapter["entrypoint_sha256"],
        },
        "output_manifest": dict(descriptor),
        "artifact_root": "artifacts",
        "artifact_set_sha256": validated["artifact_set_sha256"],
        "artifact_count": validated["artifact_count"],
        "artifact_bytes": validated["artifact_bytes"],
        "global_seal_required": True,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = materialization_receipt_sha256(receipt)
    return receipt


def validate_materialization_receipt(
    value: Mapping[str, object],
    *,
    result_root: Path,
    context: Mapping[str, object],
    snapshot: Mapping[str, object],
    adapter: Mapping[str, object],
) -> dict[str, object]:
    """Validate a locally published lane output and its full artifact graph."""

    receipt = dict(value)
    require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "kind",
            "manifest_sha256",
            "manifest_file_sha256",
            "lane_completion_receipt_sha256",
            "lane_completion_file_sha256",
            "candidate_snapshot_sha256",
            "input_snapshot_set_sha256",
            "target_lengths",
            "bindings",
            "adapter",
            "output_manifest",
            "artifact_root",
            "artifact_set_sha256",
            "artifact_count",
            "artifact_bytes",
            "global_seal_required",
            "training_ready",
            "receipt_sha256",
        },
        where="lane materialization receipt",
    )
    if (
        receipt["schema"] != LANE_MATERIALIZATION_RECEIPT_SCHEMA
        or receipt["status"] != "verified"
        or receipt["global_seal_required"] is not True
        or receipt["training_ready"] is not False
        or receipt["target_lengths"] != list(TARGET_LENGTHS)
        or receipt["artifact_root"] != "artifacts"
    ):
        raise ContractError("lane materialization receipt schema/status drifted")
    manifest = context["manifest"]
    completion = context["completion"]
    assert isinstance(manifest, Mapping) and isinstance(completion, Mapping)
    expected = {
        "kind": manifest["kind"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": context["manifest_file_sha256"],
        "lane_completion_receipt_sha256": completion["receipt_sha256"],
        "lane_completion_file_sha256": context["completion_file_sha256"],
        "candidate_snapshot_sha256": snapshot["snapshot_sha256"],
        "input_snapshot_set_sha256": manifest["input_snapshot_set_sha256"],
        "bindings": context["bindings"],
        "adapter": {
            "adapter_id": adapter["adapter_id"],
            "command_sha256": adapter["command_sha256"],
            "entrypoint_sha256": adapter["entrypoint_sha256"],
        },
    }
    for key, desired in expected.items():
        if receipt[key] != desired:
            raise ContractError(f"lane materialization receipt {key} binding drifted")
    if materialization_receipt_sha256(receipt) != require_sha256(
        receipt["receipt_sha256"], where="lane materialization receipt_sha256"
    ):
        raise ContractError("lane materialization receipt logical SHA-256 drifted")
    descriptor = _validate_output_manifest_descriptor(
        receipt["output_manifest"],
        root=result_root,
        where="lane materialization output manifest",
    )
    logical_sha = str(descriptor["logical_sha256"])
    raw, output_value = load_json_object(
        result_root / "output-manifest.json", where="lane materialization output manifest"
    )
    if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ContractError("lane materialization output manifest raw SHA-256 drifted")
    output_manifest, records = validate_output_manifest(
        output_value, artifact_root=result_root / "artifacts"
    )
    if output_manifest["manifest_sha256"] != logical_sha:
        raise ContractError("lane materialization output logical manifest SHA-256 drifted")
    if output_manifest["kind"] != receipt["kind"]:
        raise ContractError("lane materialization output kind drifted")
    if output_manifest["bindings"] != context["bindings"]:
        raise ContractError("lane materialization output bindings drifted")
    if _inventory_regular_files(result_root / "artifacts") != {
        str(record["path"]) for record in records
    }:
        raise ContractError("lane materialization artifact inventory drifted")
    digest = artifact_set_sha256(records)
    if digest != require_sha256(receipt["artifact_set_sha256"], where="artifact_set_sha256"):
        raise ContractError("lane materialization artifact set SHA-256 drifted")
    count = require_int(receipt["artifact_count"], where="artifact_count")
    byte_count = require_int(receipt["artifact_bytes"], where="artifact_bytes")
    if count != len(records) or byte_count != sum(int(record["size"]) for record in records):
        raise ContractError("lane materialization artifact totals drifted")
    return {
        **receipt,
        "output_manifest": {**descriptor, "logical_sha256": logical_sha},
    }


def _load_result_receipt(
    result_root: Path,
    *,
    context: Mapping[str, object],
    snapshot: Mapping[str, object],
    adapter: Mapping[str, object],
) -> dict[str, object]:
    raw, value = load_json_object(
        result_root / "materialization-receipt.json",
        where="lane materialization receipt",
    )
    receipt = validate_materialization_receipt(
        value,
        result_root=result_root,
        context=context,
        snapshot=snapshot,
        adapter=adapter,
    )
    descriptor = _stable_file_descriptor(
        result_root / "materialization-receipt.json", where="lane materialization receipt"
    )
    if descriptor != {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
        raise ContractError("lane materialization receipt changed during validation")
    return receipt


def _publish_completed_attempt(
    *,
    attempt_root: Path,
    result_root: Path,
    context: Mapping[str, object],
    snapshot: Mapping[str, object],
    adapter: Mapping[str, object],
) -> dict[str, object]:
    if result_root.exists() or result_root.is_symlink():
        return _load_result_receipt(
            result_root, context=context, snapshot=snapshot, adapter=adapter
        )
    validated = _validated_adapter_output(
        attempt_root, context=context, snapshot=snapshot
    )
    receipt = _build_materialization_receipt(
        context=context, snapshot=snapshot, adapter=adapter, validated=validated
    )
    atomic_write_json(attempt_root / "materialization-receipt.json", receipt)
    validate_materialization_receipt(
        receipt,
        result_root=attempt_root,
        context=context,
        snapshot=snapshot,
        adapter=adapter,
    )
    result_root.parent.mkdir(parents=True, exist_ok=True)
    if result_root.exists() or result_root.is_symlink():
        raise ContractError("immutable materialization result raced during publication")
    os.replace(attempt_root, result_root)
    directory_fd = os.open(result_root.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return _load_result_receipt(
        result_root, context=context, snapshot=snapshot, adapter=adapter
    )


def run_lane_materializer(
    *,
    manifest_path: Path,
    completion_path: Path,
    output_root: Path,
    scratch_root: Path,
    object_store: ObjectStore,
    adapter_spec: AdapterSpec,
) -> dict[str, object]:
    """Materialize one complete cloud lane with crash-safe adapter resume.

    A failed adapter leaves only its private attempt directory.  A completed
    adapter is validated and checkpointed before the result directory is moved
    atomically, so a restart after that point never re-runs the adapter.
    """

    output_root = _ensure_regular_directory(
        Path(output_root), where="lane materializer output root", create=True
    )
    scratch_root = _ensure_regular_directory(
        Path(scratch_root), where="lane materializer scratch root", create=True
    )
    adapter = validate_adapter_spec(adapter_spec)
    with _exclusive_lock(output_root / ".lane-materializer.lock"):
        context = _load_lane_context(
            manifest_path=Path(manifest_path), completion_path=Path(completion_path)
        )
        manifest = context["manifest"]
        completion = context["completion"]
        assert isinstance(manifest, Mapping) and isinstance(completion, Mapping)
        kind = str(manifest["kind"])
        receipt_sha = str(completion["receipt_sha256"])
        snapshot_root, snapshot = materialize_candidate_snapshot(
            context=context,
            output_root=output_root,
            scratch_root=scratch_root,
            object_store=object_store,
        )
        result_root = _result_directory(
            output_root, kind=kind, receipt_sha256=receipt_sha
        )
        if result_root.exists() or result_root.is_symlink():
            return _load_result_receipt(
                result_root, context=context, snapshot=snapshot, adapter=adapter
            )
        run_root = _run_directory(output_root, kind=kind, receipt_sha256=receipt_sha)
        _ensure_regular_directory(run_root, where="lane materializer run root", create=True)
        checkpoint_path = run_root / "checkpoint.json"
        checkpoint = _load_or_initialize_checkpoint(
            checkpoint_path, context=context, snapshot=snapshot, adapter=adapter
        )
        attempts_root = _ensure_regular_directory(
            run_root / "attempts", where="lane materializer attempts root", create=True
        )
        if checkpoint["status"] == "adapter_running":
            running_attempt = attempts_root / (
                f"attempt-{int(checkpoint['attempts_started']):06d}"
            )
            output_manifest = running_attempt / "output-manifest.json"
            if output_manifest.exists() or output_manifest.is_symlink():
                if running_attempt.is_symlink() or not running_attempt.is_dir():
                    raise ContractError(
                        "running materialization attempt is not a regular directory"
                    )
                # A host can die after the adapter exits but before the next
                # checkpoint write.  Revalidate complete output before deciding
                # whether it is safe to avoid an expensive replay.
                validated = _validated_adapter_output(
                    running_attempt, context=context, snapshot=snapshot
                )
                checkpoint = _write_next_checkpoint(
                    checkpoint_path,
                    current=checkpoint,
                    context=context,
                    snapshot=snapshot,
                    adapter=adapter,
                    status="adapter_complete",
                    attempts_started=int(checkpoint["attempts_started"]),
                    completed_attempt=int(checkpoint["attempts_started"]),
                    output=_checkpoint_output(validated),
                )
        if checkpoint["status"] in {"adapter_complete", "published"}:
            completed = checkpoint["completed_attempt"]
            assert isinstance(completed, int)
            attempt_root = attempts_root / f"attempt-{completed:06d}"
            validated = _validated_adapter_output(
                attempt_root, context=context, snapshot=snapshot
            )
            _assert_checkpoint_output(checkpoint, validated)
            receipt = _publish_completed_attempt(
                attempt_root=attempt_root,
                result_root=result_root,
                context=context,
                snapshot=snapshot,
                adapter=adapter,
            )
            if checkpoint["status"] != "published":
                _write_next_checkpoint(
                    checkpoint_path,
                    current=checkpoint,
                    context=context,
                    snapshot=snapshot,
                    adapter=adapter,
                    status="published",
                    attempts_started=int(checkpoint["attempts_started"]),
                    completed_attempt=completed,
                    output=_checkpoint_output(validated),
                )
            return receipt

        attempt_number = int(checkpoint["attempts_started"]) + 1
        attempt_root = attempts_root / f"attempt-{attempt_number:06d}"
        if (
            attempt_root.exists()
            or attempt_root.is_symlink()
            or _ATTEMPT_RE.fullmatch(attempt_root.name) is None
        ):
            raise ContractError(f"lane materializer attempt target already exists: {attempt_root}")
        checkpoint = _write_next_checkpoint(
            checkpoint_path,
            current=checkpoint,
            context=context,
            snapshot=snapshot,
            adapter=adapter,
            status="adapter_running",
            attempts_started=attempt_number,
            completed_attempt=None,
            output=None,
        )
        attempt_root.mkdir()
        request = _build_adapter_request(
            context=context,
            snapshot_root=snapshot_root,
            snapshot=snapshot,
            adapter=adapter,
        )
        _run_adapter(
            attempt_root,
            request=request,
            snapshot_root=snapshot_root,
            adapter=adapter,
        )
        # A completed subprocess is not accepted until both its code and its
        # immutable snapshot still have the exact pinned bytes.
        validate_adapter_spec(adapter_spec)
        load_candidate_snapshot(snapshot_root, context=context)
        if (
            sha256_file(Path(manifest_path)) != context["manifest_file_sha256"]
            or sha256_file(Path(completion_path)) != context["completion_file_sha256"]
        ):
            raise ContractError("lane manifest or completion receipt changed during adapter run")
        validated = _validated_adapter_output(
            attempt_root, context=context, snapshot=snapshot
        )
        checkpoint = _write_next_checkpoint(
            checkpoint_path,
            current=checkpoint,
            context=context,
            snapshot=snapshot,
            adapter=adapter,
            status="adapter_complete",
            attempts_started=attempt_number,
            completed_attempt=attempt_number,
            output=_checkpoint_output(validated),
        )
        receipt = _publish_completed_attempt(
            attempt_root=attempt_root,
            result_root=result_root,
            context=context,
            snapshot=snapshot,
            adapter=adapter,
        )
        _write_next_checkpoint(
            checkpoint_path,
            current=checkpoint,
            context=context,
            snapshot=snapshot,
            adapter=adapter,
            status="published",
            attempts_started=attempt_number,
            completed_attempt=attempt_number,
            output=_checkpoint_output(validated),
        )
        return receipt


def _parse_adapter_command(value: Sequence[str]) -> tuple[str, ...]:
    argv = tuple(value)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise ContractError("provide an adapter command after '--'")
    return argv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--completion-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--adapter-entrypoint", required=True, type=Path)
    parser.add_argument(
        "adapter_command",
        nargs=argparse.REMAINDER,
        help="adapter command after '--'; it must include --adapter-entrypoint",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    command = _parse_adapter_command(args.adapter_command)
    adapter = make_adapter_spec(
        adapter_id=args.adapter_id,
        argv=command,
        entrypoint=args.adapter_entrypoint,
    )
    from scripts.distributed_data_prep.source_worker import GcloudObjectStore

    receipt = run_lane_materializer(
        manifest_path=args.manifest,
        completion_path=args.completion_receipt,
        output_root=args.output_root,
        scratch_root=args.scratch_root,
        object_store=GcloudObjectStore(),
        adapter_spec=adapter,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())


__all__ = [
    "AdapterSpec",
    "LANE_CANDIDATE_SNAPSHOT_SCHEMA",
    "LANE_MATERIALIZATION_CHECKPOINT_SCHEMA",
    "LANE_MATERIALIZATION_RECEIPT_SCHEMA",
    "LANE_MATERIALIZATION_REQUEST_SCHEMA",
    "candidate_snapshot_sha256",
    "load_candidate_snapshot",
    "make_adapter_spec",
    "materialization_checkpoint_sha256",
    "materialization_receipt_sha256",
    "materialize_candidate_snapshot",
    "run_lane_materializer",
    "validate_adapter_spec",
    "validate_candidate_snapshot",
    "validate_materialization_checkpoint",
    "validate_materialization_receipt",
]
