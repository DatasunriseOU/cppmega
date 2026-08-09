#!/usr/bin/env python3
"""Run one physical VM's receipt-bound share of a logical cloud lane.

The lane manifest owns logical worker IDs.  This wrapper maps them to a smaller
physical pool by stable modulo, shares one verified snapshot cache per VM, and
publishes both logical-worker and physical-worker completion receipts.  It
returns exit 75 only when a diagnostic contains an explicit HTTP 429; every
other failure remains deterministic exit 2 for the supervising systemd unit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_sha256,
    gcs_join,
    require_int,
    require_sha256,
    sha256_file,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.cloud_lane import (  # noqa: E402
    ObjectStore,
    load_cloud_lane_manifest,
)
from scripts.distributed_data_prep.cloud_lane_worker import (  # noqa: E402
    AdapterSession,
    WORKER_COMPLETION_SCHEMA,
    prepare_verified_snapshot_cache,
    publish_deferred_worker_completion,
    run_cloud_lane_worker,
    worker_completion_sha256,
)
from scripts.distributed_data_prep.source_worker import GcloudObjectStore  # noqa: E402

POOL_COMPLETION_SCHEMA = "cppmega.distributed_cloud_lane_pool_completion_v1"
POOL_FAILURE_SCHEMA = "cppmega.distributed_cloud_lane_pool_failure_v1"
_HTTP_429_RE = re.compile(
    r"(?:"
    r"\bHTTP(?:Error)?(?:/\d(?:\.\d)?)?\s*(?:status(?:\s+code)?)?\s*[:=]?\s*429\b"
    r"|\bstatus(?:\s+code)?\s*[:=]?\s*429\b"
    r"|\b429\s+Too\s+Many\s+Requests\b"
    r")",
    re.IGNORECASE,
)


class ConfirmedHttp429(RuntimeError):
    """At least one logical worker failed with explicit HTTP 429 evidence."""


class _LogicalWorkerFailure(RuntimeError):
    def __init__(self, worker: str, error: BaseException) -> None:
        super().__init__(str(error))
        self.worker = worker
        self.error = error


def pool_completion_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return canonical_sha256(payload)


def pool_failure_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return canonical_sha256(payload)


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ContractError("cloud lane checkout has no readable Git HEAD")
    return completed.stdout.strip()


def _publish_exact(
    source: Path,
    uri: str,
    *,
    object_store: ObjectStore,
    scratch_root: Path,
) -> dict[str, object]:
    published = object_store.publish_if_absent(source, uri)
    generation = str(published.get("generation", ""))
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError("cloud lane publication has no exact GCS generation")
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pool-readback-", dir=scratch_root) as raw:
        readback = Path(raw) / "object"
        metadata = object_store.download(uri, readback, generation=generation)
        if (
            str(metadata.get("generation")) != generation
            or int(metadata.get("size_bytes", -1)) != source.stat().st_size
            or sha256_file(readback) != sha256_file(source)
        ):
            raise ContractError("cloud lane publication exact-generation readback failed")
    return {
        "uri": uri,
        "generation": generation,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _logical_workers(
    workers: Sequence[object], *, physical_index: int, physical_count: int
) -> list[str]:
    index = require_int(physical_index, where="physical_index")
    count = require_int(physical_count, where="physical_count", minimum=1)
    if index >= count:
        raise ContractError("physical_index must be smaller than physical_count")
    if count > len(workers):
        raise ContractError("physical pool cannot exceed logical worker count")
    assigned = [str(worker) for worker in workers[index::count]]
    if not assigned:
        raise ContractError("physical worker received no logical cloud lane workers")
    return assigned


def _worker_completion_uri(
    manifest: Mapping[str, object], completion: Mapping[str, object]
) -> str:
    return gcs_join(
        str(manifest["gcs_output_prefix"]),
        "worker-completions",
        str(manifest["kind"]),
        str(manifest["manifest_sha256"]),
        str(completion["worker"]),
        f"{completion['receipt_sha256']}.complete.json",
    )


def _run_logical_worker(
    worker: str,
    *,
    manifest_path: Path,
    adapter_path: Path,
    adapter_sha256: str,
    scratch_root: Path,
    receipt_root: Path,
    ledger_root: Path,
    publication_scratch: Path,
    object_store: ObjectStore,
    adapter_session: AdapterSession | None,
    verified_snapshots: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    try:
        completion = run_cloud_lane_worker(
            manifest_path=manifest_path,
            worker=worker,
            adapter_command=[sys.executable, str(adapter_path)],
            adapter_sha256=adapter_sha256,
            scratch_root=scratch_root,
            receipt_root=receipt_root,
            ledger_path=ledger_root / f"{worker}.ledger.json",
            object_store=object_store,
            adapter_session=adapter_session,
            verified_snapshots=verified_snapshots,
            defer_completion_publication=adapter_session is not None,
        )
        if adapter_session is not None:
            return {"prepared": completion}
        completion_path = receipt_root / f"{worker}.complete.json"
        publication = _publish_exact(
            completion_path,
            _worker_completion_uri(
                completion=completion, manifest=manifest_path_to_value(manifest_path)
            ),
            object_store=object_store,
            scratch_root=publication_scratch / worker,
        )
        return {"completion": completion, "publication": publication}
    except BaseException as error:
        raise _LogicalWorkerFailure(worker, error) from error


def manifest_path_to_value(path: Path) -> dict[str, object]:
    """Load through the strict public contract at every publication boundary."""

    value, _file_sha256 = load_cloud_lane_manifest(path)
    return value


def _diagnostic(error: BaseException) -> tuple[str, bool]:
    text = f"{type(error).__name__}: {str(error)[:4000]}"
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(), bool(
        _HTTP_429_RE.search(text)
    )


def _case5_adapter_session(
    *, adapter_path: Path, repo_root: Path, session_root: Path
) -> AdapterSession | None:
    expected = (
        repo_root / "scripts" / "distributed_data_prep" / "ci_case5_snapshot.py"
    ).resolve()
    if adapter_path != expected:
        return None
    from scripts.distributed_data_prep.ci_case5_snapshot import (
        CiCase5AdapterSession,
    )

    return CiCase5AdapterSession(session_root=session_root)


def _write_failure_receipt(
    *,
    path: Path,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    physical_index: int,
    physical_count: int,
    failures: Sequence[_LogicalWorkerFailure],
) -> dict[str, object]:
    diagnostics = []
    for failure in sorted(failures, key=lambda item: item.worker):
        digest, confirmed = _diagnostic(failure.error)
        diagnostics.append(
            {
                "worker": failure.worker,
                "error_type": type(failure.error).__name__,
                "diagnostic_sha256": digest,
                "confirmed_http_429": confirmed,
            }
        )
    receipt: dict[str, object] = {
        "schema": POOL_FAILURE_SCHEMA,
        "status": "failed",
        "kind": manifest["kind"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha256,
        "physical_worker_index": physical_index,
        "physical_worker_count": physical_count,
        "diagnostics": diagnostics,
        "retry_exit_code": 75 if all(item["confirmed_http_429"] for item in diagnostics) else 2,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = pool_failure_sha256(receipt)
    atomic_write_json(path, receipt)
    return receipt


def run_cloud_lane_pool_worker(
    *,
    manifest_path: Path,
    adapter_path: Path,
    repo_root: Path,
    physical_index: int,
    physical_count: int,
    slots: int,
    control_prefix: str,
    stage_root: Path,
    object_store: ObjectStore,
    adapter_session_factory: Callable[[Path], AdapterSession] | None = None,
    enable_case5_session: bool = False,
) -> dict[str, object]:
    manifest, manifest_file_sha256 = load_cloud_lane_manifest(manifest_path)
    pipeline = manifest["pipeline"]
    assert isinstance(pipeline, Mapping)
    if _git_head(repo_root.resolve()) != pipeline["code_revision"]:
        raise ContractError("cloud lane checkout differs from manifest code_revision")
    adapter_path = adapter_path.resolve()
    adapter_sha256 = require_sha256(
        pipeline["runner_sha256"], where="pipeline.runner_sha256"
    )
    if adapter_path.is_symlink() or sha256_file(adapter_path) != adapter_sha256:
        raise ContractError("cloud lane adapter differs from manifest runner_sha256")
    control = validate_gcs_uri(control_prefix.rstrip("/"), where="control_prefix")
    assigned = _logical_workers(
        manifest["workers"],
        physical_index=physical_index,
        physical_count=physical_count,
    )
    slot_count = require_int(slots, where="slots", minimum=1)
    if slot_count > min(16, len(assigned)):
        raise ContractError("slots exceed the bounded physical worker assignment")

    stage_root = stage_root.resolve()
    scratch_root = stage_root / "work"
    receipt_root = stage_root / "receipts" / "logical"
    ledger_root = stage_root / "receipts" / "ledgers"
    publication_scratch = stage_root / "work" / "publication"
    for directory in (scratch_root, receipt_root, ledger_root, publication_scratch):
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ContractError(
                "cloud lane stage directories must be regular directories"
            )

    session_root = scratch_root / "adapter-session"
    if adapter_session_factory is not None:
        adapter_session = adapter_session_factory(session_root)
    elif enable_case5_session:
        adapter_session = _case5_adapter_session(
            adapter_path=adapter_path,
            repo_root=repo_root.resolve(),
            session_root=session_root,
        )
        if adapter_session is None:
            raise ContractError(
                "persistent CASE5 session requires the canonical CASE5 adapter"
            )
    else:
        adapter_session = None
    verified_snapshots: list[dict[str, object]] | None = None
    if adapter_session is not None:
        verified_snapshots = prepare_verified_snapshot_cache(
            manifest,
            object_store=object_store,
            input_root=scratch_root / "inputs" / str(manifest["manifest_sha256"]),
        )

    results: dict[str, dict[str, object]] = {}
    failures: list[_LogicalWorkerFailure] = []
    with ThreadPoolExecutor(max_workers=slot_count) as executor:
        futures: dict[Future[dict[str, object]], str] = {
            executor.submit(
                _run_logical_worker,
                worker,
                manifest_path=manifest_path,
                adapter_path=adapter_path,
                adapter_sha256=adapter_sha256,
                scratch_root=scratch_root,
                receipt_root=receipt_root,
                ledger_root=ledger_root,
                publication_scratch=publication_scratch,
                object_store=object_store,
                adapter_session=adapter_session,
                verified_snapshots=verified_snapshots,
            ): worker
            for worker in assigned
        }
        for future in as_completed(futures):
            worker = futures[future]
            try:
                results[worker] = future.result()
            except _LogicalWorkerFailure as error:
                failures.append(error)
    if adapter_session is not None:
        try:
            adapter_session.close()
        except BaseException as error:
            failures.append(
                _LogicalWorkerFailure(
                    f"physical-{physical_index:04d}-adapter-session", error
                )
            )

    if not failures and adapter_session is not None:
        for worker in assigned:
            try:
                prepared = results[worker]["prepared"]
                if not isinstance(prepared, Mapping):
                    raise ContractError("deferred logical worker result is malformed")
                completion = publish_deferred_worker_completion(
                    prepared,
                    manifest_path=manifest_path,
                    ledger_path=ledger_root / f"{worker}.ledger.json",
                    receipt_root=receipt_root,
                    object_store=object_store,
                    scratch_root=publication_scratch / worker / "deferred",
                )
                completion_path = receipt_root / f"{worker}.complete.json"
                publication = _publish_exact(
                    completion_path,
                    _worker_completion_uri(
                        completion=completion, manifest=manifest
                    ),
                    object_store=object_store,
                    scratch_root=publication_scratch / worker,
                )
                results[worker] = {
                    "completion": completion,
                    "publication": publication,
                }
            except BaseException as error:
                failures.append(_LogicalWorkerFailure(worker, error))

    if failures:
        failure_path = (
            stage_root / "receipts" / f"physical-{physical_index:04d}.failed.json"
        )
        failure = _write_failure_receipt(
            path=failure_path,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            physical_index=physical_index,
            physical_count=physical_count,
            failures=failures,
        )
        failure_uri = gcs_join(
            control,
            "control",
            "cloud-lane-failures",
            str(manifest["manifest_sha256"]),
            f"physical-{physical_index:04d}",
            f"{failure['receipt_sha256']}.failure.json",
        )
        try:
            _publish_exact(
                failure_path,
                failure_uri,
                object_store=object_store,
                scratch_root=publication_scratch / "failure",
            )
        except Exception:
            pass
        if int(failure["retry_exit_code"]) == 75:
            raise ConfirmedHttp429(
                f"physical worker {physical_index} has confirmed HTTP 429 diagnostics "
                f"{failure['receipt_sha256']}"
            )
        raise ContractError(
            f"physical worker {physical_index} has deterministic diagnostics "
            f"{failure['receipt_sha256']}"
        )

    logical_receipts = []
    totals = {
        "source_record_count": 0,
        "candidate_document_count": 0,
        "valid_tokens": 0,
        "assignment_receipt_count": 0,
    }
    for worker in assigned:
        result = results[worker]
        completion = result["completion"]
        assert isinstance(completion, Mapping)
        if (
            completion["schema"] != WORKER_COMPLETION_SCHEMA
            or worker_completion_sha256(completion) != completion["receipt_sha256"]
        ):
            raise ContractError("logical worker completion receipt drifted")
        worker_totals = completion["totals"]
        assert isinstance(worker_totals, Mapping)
        for field in totals:
            totals[field] += int(worker_totals[field])
        logical_receipts.append(
            {
                "worker": worker,
                "receipt_sha256": completion["receipt_sha256"],
                "publication": result["publication"],
            }
        )
    receipt: dict[str, object] = {
        "schema": POOL_COMPLETION_SCHEMA,
        "status": "complete",
        "kind": manifest["kind"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha256,
        "code_revision": pipeline["code_revision"],
        "adapter_sha256": adapter_sha256,
        "physical_worker_index": physical_index,
        "physical_worker_count": physical_count,
        "logical_workers": assigned,
        "logical_worker_completions": logical_receipts,
        "totals": totals,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = pool_completion_sha256(receipt)
    completion_path = (
        stage_root / "receipts" / f"physical-{physical_index:04d}.complete.json"
    )
    atomic_write_json(completion_path, receipt)
    publication = _publish_exact(
        completion_path,
        gcs_join(
            control,
            "control",
            "cloud-lane-completed",
            str(manifest["manifest_sha256"]),
            f"physical-{physical_index:04d}.complete.json",
        ),
        object_store=object_store,
        scratch_root=publication_scratch / "physical",
    )
    return {"receipt": receipt, "publication": publication}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--physical-index", required=True, type=int)
    parser.add_argument("--physical-count", required=True, type=int)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--persistent-case5-session", action="store_true")
    parser.add_argument("--control-prefix", required=True)
    parser.add_argument("--stage-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        run_cloud_lane_pool_worker(
            manifest_path=args.manifest,
            adapter_path=args.adapter,
            repo_root=args.repo_root,
            physical_index=args.physical_index,
            physical_count=args.physical_count,
            slots=args.slots,
            control_prefix=args.control_prefix,
            stage_root=args.stage_root,
            object_store=GcloudObjectStore(),
            enable_case5_session=args.persistent_case5_session,
        )
    except ConfirmedHttp429 as exc:
        print(f"cloud lane pool worker retryable failure: {exc}", file=sys.stderr)
        return 75
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"cloud lane pool worker failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "ConfirmedHttp429",
    "POOL_COMPLETION_SCHEMA",
    "POOL_FAILURE_SCHEMA",
    "pool_completion_sha256",
    "pool_failure_sha256",
    "run_cloud_lane_pool_worker",
]
