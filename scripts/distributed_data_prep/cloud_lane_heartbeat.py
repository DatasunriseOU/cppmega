#!/usr/bin/env python3
"""Publish and independently verify immutable cloud-lane worker heartbeats.

The heartbeat is deliberately outside the CASE5 adapter/session lifetime.  It
only reports physical-worker liveness and therefore can never make a dataset
training-ready.  Every object is content addressed, published create-only, and
read back at the exact generation before its local publication receipt exists.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import signal
import sys
import tempfile
import threading
from types import FrameType
from typing import Callable, Mapping, Sequence, cast

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_sha256,
    gcs_join,
    load_json_object,
    require_exact_fields,
    require_git_object,
    require_int,
    require_nonempty,
    require_sha256,
    sha256_file,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.cloud_lane import ObjectStore  # noqa: E402
from scripts.distributed_data_prep.source_worker import (  # noqa: E402
    GcloudObjectStore,
)


WORKER_HEARTBEAT_SCHEMA = "cppmega.cloud_lane_worker_heartbeat_v1"
WORKER_HEARTBEAT_PUBLICATION_SCHEMA = (
    "cppmega.cloud_lane_worker_heartbeat_publication_v1"
)
WORKER_HEARTBEAT_PHASE = "cloud-lane-pool-worker"
DEFAULT_HEARTBEAT_SECONDS = 300
_PHYSICAL_WORKER_RE = re.compile(r"^physical-([0-9]{4})$")
_RFC3339_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z$"
)
_HEARTBEAT_FIELDS = {
    "schema",
    "status",
    "phase",
    "sequence",
    "manifest_sha256",
    "manifest_file_sha256",
    "code_revision",
    "physical_worker",
    "physical_worker_count",
    "emitted_at",
    "training_ready",
    "receipt_sha256",
}
_PUBLICATION_FIELDS = {
    "schema",
    "uri",
    "generation",
    "size_bytes",
    "sha256",
    "receipt_sha256",
    "training_ready",
}


def worker_heartbeat_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return canonical_sha256(payload)


def _utc_timestamp(value: object, *, where: str) -> str:
    timestamp = require_nonempty(value, where=where)
    if _RFC3339_UTC_RE.fullmatch(timestamp) is None:
        raise ContractError(f"{where} must be a canonical RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{where} is not a real timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError(f"{where} must use UTC")
    return timestamp


def _physical_worker(value: object, *, worker_count: int, where: str) -> str:
    worker = require_nonempty(value, where=where)
    match = _PHYSICAL_WORKER_RE.fullmatch(worker)
    if match is None:
        raise ContractError(f"{where} is not canonical")
    if int(match.group(1)) >= worker_count:
        raise ContractError(f"{where} is outside physical_worker_count")
    return worker


def validate_worker_heartbeat(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the complete heartbeat schema and its self-digest."""

    require_exact_fields(value, _HEARTBEAT_FIELDS, where="worker heartbeat")
    if value["schema"] != WORKER_HEARTBEAT_SCHEMA:
        raise ContractError("worker heartbeat schema drifted")
    if value["status"] != "running" or value["phase"] != WORKER_HEARTBEAT_PHASE:
        raise ContractError("worker heartbeat liveness state drifted")
    sequence = require_int(
        value["sequence"], where="worker heartbeat sequence", minimum=0
    )
    worker_count = require_int(
        value["physical_worker_count"],
        where="worker heartbeat physical_worker_count",
        minimum=1,
    )
    worker = _physical_worker(
        value["physical_worker"],
        worker_count=worker_count,
        where="worker heartbeat physical_worker",
    )
    manifest_sha256 = require_sha256(
        value["manifest_sha256"], where="worker heartbeat manifest_sha256"
    )
    manifest_file_sha256 = require_sha256(
        value["manifest_file_sha256"],
        where="worker heartbeat manifest_file_sha256",
    )
    code_revision = require_git_object(
        value["code_revision"], where="worker heartbeat code_revision"
    )
    emitted_at = _utc_timestamp(
        value["emitted_at"], where="worker heartbeat emitted_at"
    )
    if value["training_ready"] is not False:
        raise ContractError("worker heartbeat must remain training_ready=false")
    receipt_sha256 = require_sha256(
        value["receipt_sha256"], where="worker heartbeat receipt_sha256"
    )
    if worker_heartbeat_sha256(value) != receipt_sha256:
        raise ContractError("worker heartbeat receipt_sha256 drifted")
    return {
        "schema": WORKER_HEARTBEAT_SCHEMA,
        "status": "running",
        "phase": WORKER_HEARTBEAT_PHASE,
        "sequence": sequence,
        "manifest_sha256": manifest_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "code_revision": code_revision,
        "physical_worker": worker,
        "physical_worker_count": worker_count,
        "emitted_at": emitted_at,
        "training_ready": False,
        "receipt_sha256": receipt_sha256,
    }


def build_worker_heartbeat(
    *,
    sequence: int,
    manifest_sha256: str,
    manifest_file_sha256: str,
    code_revision: str,
    physical_worker: str,
    physical_worker_count: int,
    emitted_at: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": WORKER_HEARTBEAT_SCHEMA,
        "status": "running",
        "phase": WORKER_HEARTBEAT_PHASE,
        "sequence": sequence,
        "manifest_sha256": manifest_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "code_revision": code_revision,
        "physical_worker": physical_worker,
        "physical_worker_count": physical_worker_count,
        "emitted_at": emitted_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_ready": False,
    }
    value["receipt_sha256"] = worker_heartbeat_sha256(value)
    return validate_worker_heartbeat(value)


def worker_heartbeat_uri(control_prefix: str, value: Mapping[str, object]) -> str:
    heartbeat = validate_worker_heartbeat(value)
    sequence = cast(int, heartbeat["sequence"])
    return gcs_join(
        control_prefix,
        "control",
        "cloud-lane-heartbeats",
        str(heartbeat["manifest_sha256"]),
        str(heartbeat["physical_worker"]),
        f"{sequence:06d}-"
        f"{heartbeat['receipt_sha256']}.heartbeat.json",
    )


def validate_worker_heartbeat_uri(
    uri: str, *, control_prefix: str, value: Mapping[str, object]
) -> str:
    observed = validate_gcs_uri(uri, where="worker heartbeat URI")
    expected = worker_heartbeat_uri(control_prefix, value)
    if observed != expected:
        raise ContractError("worker heartbeat URI binding drifted")
    return observed


def validate_worker_heartbeat_file(
    path: Path, *, uri: str, control_prefix: str
) -> dict[str, object]:
    _raw, value = load_json_object(path, where="worker heartbeat receipt")
    heartbeat = validate_worker_heartbeat(value)
    validate_worker_heartbeat_uri(uri, control_prefix=control_prefix, value=heartbeat)
    return heartbeat


def validate_worker_heartbeat_publication(
    value: Mapping[str, object],
    *,
    heartbeat: Mapping[str, object],
    control_prefix: str,
) -> dict[str, object]:
    """Validate the local descriptor proving exact-generation publication."""

    receipt = validate_worker_heartbeat(heartbeat)
    require_exact_fields(
        value, _PUBLICATION_FIELDS, where="worker heartbeat publication"
    )
    if value["schema"] != WORKER_HEARTBEAT_PUBLICATION_SCHEMA:
        raise ContractError("worker heartbeat publication schema drifted")
    uri = validate_worker_heartbeat_uri(
        require_nonempty(value["uri"], where="worker heartbeat publication URI"),
        control_prefix=control_prefix,
        value=receipt,
    )
    generation = _positive_generation(
        value["generation"], where="worker heartbeat publication generation"
    )
    size_bytes = require_int(
        value["size_bytes"],
        where="worker heartbeat publication size_bytes",
        minimum=1,
    )
    sha256 = require_sha256(
        value["sha256"], where="worker heartbeat publication SHA-256"
    )
    receipt_sha256 = require_sha256(
        value["receipt_sha256"],
        where="worker heartbeat publication receipt_sha256",
    )
    if receipt_sha256 != receipt["receipt_sha256"]:
        raise ContractError("worker heartbeat publication receipt binding drifted")
    if value["training_ready"] is not False:
        raise ContractError(
            "worker heartbeat publication must remain training_ready=false"
        )
    return {
        "schema": WORKER_HEARTBEAT_PUBLICATION_SCHEMA,
        "uri": uri,
        "generation": generation,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "receipt_sha256": receipt_sha256,
        "training_ready": False,
    }


def _positive_generation(value: object, *, where: str) -> str:
    generation = require_nonempty(value, where=where)
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError(f"{where} must be a positive decimal generation")
    return generation


def publish_worker_heartbeat(
    value: Mapping[str, object],
    *,
    control_prefix: str,
    receipt_root: Path,
    scratch_root: Path,
    object_store: ObjectStore,
) -> dict[str, object]:
    """Create-only publish one heartbeat and verify its exact generation."""

    heartbeat = validate_worker_heartbeat(value)
    sequence = cast(int, heartbeat["sequence"])
    uri = worker_heartbeat_uri(control_prefix, heartbeat)
    receipt_root.mkdir(parents=True, exist_ok=True)
    local = receipt_root / (
        f"{heartbeat['physical_worker']}.heartbeat."
        f"{sequence:06d}.json"
    )
    atomic_write_json(local, heartbeat)
    local_size = local.stat().st_size
    local_sha256 = sha256_file(local)
    published = dict(object_store.publish_if_absent(local, uri))
    if published.get("uri") != uri:
        raise ContractError("worker heartbeat publication URI metadata drifted")
    generation = _positive_generation(
        published.get("generation"), where="worker heartbeat generation"
    )
    if require_int(
        published.get("size_bytes"),
        where="worker heartbeat published size",
        minimum=1,
    ) != local_size:
        raise ContractError("worker heartbeat publication size metadata drifted")
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="heartbeat-readback-", dir=scratch_root
    ) as raw:
        readback = Path(raw) / "heartbeat.json"
        observed = dict(object_store.download(uri, readback, generation=generation))
        if (
            observed.get("uri") != uri
            or str(observed.get("generation")) != generation
            or require_int(
                observed.get("size_bytes"),
                where="worker heartbeat readback size",
                minimum=1,
            )
            != local_size
            or sha256_file(readback) != local_sha256
        ):
            raise ContractError("worker heartbeat exact-generation readback drifted")
        validate_worker_heartbeat_file(
            readback, uri=uri, control_prefix=control_prefix
        )
    publication = validate_worker_heartbeat_publication(
        {
            "schema": WORKER_HEARTBEAT_PUBLICATION_SCHEMA,
            "uri": uri,
            "generation": generation,
            "size_bytes": local_size,
            "sha256": local_sha256,
            "receipt_sha256": heartbeat["receipt_sha256"],
            "training_ready": False,
        },
        heartbeat=heartbeat,
        control_prefix=control_prefix,
    )
    atomic_write_json(
        local.with_name(f"{local.name}.publication.json"), publication
    )
    return publication


def verify_published_worker_heartbeat(
    *,
    uri: str,
    generation: str,
    control_prefix: str,
    object_store: ObjectStore,
    scratch_root: Path,
) -> dict[str, object]:
    """Consumer-side exact-generation download and contract validation."""

    validated_uri = validate_gcs_uri(uri, where="worker heartbeat URI")
    validated_generation = _positive_generation(
        generation, where="worker heartbeat generation"
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="heartbeat-verify-", dir=scratch_root
    ) as raw:
        receipt = Path(raw) / "heartbeat.json"
        metadata = dict(
            object_store.download(
                validated_uri, receipt, generation=validated_generation
            )
        )
        if (
            metadata.get("uri") != validated_uri
            or str(metadata.get("generation")) != validated_generation
            or require_int(
                metadata.get("size_bytes"),
                where="worker heartbeat downloaded size",
                minimum=1,
            )
            != receipt.stat().st_size
        ):
            raise ContractError("worker heartbeat download metadata drifted")
        return validate_worker_heartbeat_file(
            receipt, uri=validated_uri, control_prefix=control_prefix
        )


def run_worker_heartbeat_loop(
    *,
    control_prefix: str,
    receipt_root: Path,
    scratch_root: Path,
    manifest_sha256: str,
    manifest_file_sha256: str,
    code_revision: str,
    physical_worker: str,
    physical_worker_count: int,
    interval_seconds: float,
    object_store: ObjectStore,
    stop_event: threading.Event,
    on_publication: Callable[[Mapping[str, object]], None] | None = None,
) -> int:
    """Publish immediately, then periodically until a cooperative stop."""

    if interval_seconds <= 0:
        raise ContractError("heartbeat interval must be positive")
    sequence = 0
    pending: dict[str, object] | None = None
    while not stop_event.is_set():
        if pending is None:
            pending = build_worker_heartbeat(
                sequence=sequence,
                manifest_sha256=manifest_sha256,
                manifest_file_sha256=manifest_file_sha256,
                code_revision=code_revision,
                physical_worker=physical_worker,
                physical_worker_count=physical_worker_count,
            )
        try:
            publication = publish_worker_heartbeat(
                pending,
                control_prefix=control_prefix,
                receipt_root=receipt_root,
                scratch_root=scratch_root,
                object_store=object_store,
            )
        except Exception as exc:  # liveness is observational, never a data seal
            print(
                "cloud lane heartbeat publication failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        else:
            pending = None
            sequence += 1
            if on_publication is not None:
                on_publication(publication)
        if stop_event.wait(interval_seconds):
            break
    return sequence


def run_worker_heartbeat_service(
    *,
    control_prefix: str,
    receipt_root: Path,
    scratch_root: Path,
    manifest_sha256: str,
    manifest_file_sha256: str,
    code_revision: str,
    physical_worker: str,
    physical_worker_count: int,
    interval_seconds: float,
    object_store: ObjectStore,
    on_publication: Callable[[Mapping[str, object]], None] | None = None,
) -> int:
    """Run the loop with SIGINT/SIGTERM converted to cooperative cleanup."""

    stop_event = threading.Event()
    previous: dict[signal.Signals, object] = {}

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    try:
        run_worker_heartbeat_loop(
            control_prefix=control_prefix,
            receipt_root=receipt_root,
            scratch_root=scratch_root,
            manifest_sha256=manifest_sha256,
            manifest_file_sha256=manifest_file_sha256,
            code_revision=code_revision,
            physical_worker=physical_worker,
            physical_worker_count=physical_worker_count,
            interval_seconds=interval_seconds,
            object_store=object_store,
            stop_event=stop_event,
            on_publication=on_publication,
        )
        return 0
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)  # type: ignore[arg-type]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="publish worker heartbeats")
    run.add_argument("--control-prefix", required=True)
    run.add_argument("--receipt-root", required=True, type=Path)
    run.add_argument("--scratch-root", required=True, type=Path)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--manifest-file-sha256", required=True)
    run.add_argument("--code-revision", required=True)
    run.add_argument("--physical-worker", required=True)
    run.add_argument("--physical-worker-count", required=True, type=int)
    run.add_argument(
        "--interval-seconds", type=int, default=DEFAULT_HEARTBEAT_SECONDS
    )
    verify = subparsers.add_parser(
        "verify", help="verify one exact-generation published heartbeat"
    )
    verify.add_argument("--control-prefix", required=True)
    verify.add_argument("--uri", required=True)
    verify.add_argument("--generation", required=True)
    verify.add_argument("--scratch-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        heartbeat = verify_published_worker_heartbeat(
            uri=args.uri,
            generation=args.generation,
            control_prefix=args.control_prefix,
            object_store=GcloudObjectStore(),
            scratch_root=args.scratch_root,
        )
        print(json.dumps(heartbeat, sort_keys=True))
        return 0
    run_worker_heartbeat_service(
        control_prefix=args.control_prefix,
        receipt_root=args.receipt_root,
        scratch_root=args.scratch_root,
        manifest_sha256=args.manifest_sha256,
        manifest_file_sha256=args.manifest_file_sha256,
        code_revision=args.code_revision,
        physical_worker=args.physical_worker,
        physical_worker_count=args.physical_worker_count,
        interval_seconds=args.interval_seconds,
        object_store=GcloudObjectStore(),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        raise SystemExit(main())
    except (ContractError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


__all__ = [
    "DEFAULT_HEARTBEAT_SECONDS",
    "WORKER_HEARTBEAT_PUBLICATION_SCHEMA",
    "WORKER_HEARTBEAT_SCHEMA",
    "build_worker_heartbeat",
    "publish_worker_heartbeat",
    "run_worker_heartbeat_loop",
    "run_worker_heartbeat_service",
    "validate_worker_heartbeat",
    "validate_worker_heartbeat_file",
    "validate_worker_heartbeat_publication",
    "validate_worker_heartbeat_uri",
    "verify_published_worker_heartbeat",
    "worker_heartbeat_sha256",
    "worker_heartbeat_uri",
]
