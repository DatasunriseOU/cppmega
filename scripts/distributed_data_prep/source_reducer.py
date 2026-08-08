#!/usr/bin/env python3
"""Canonical single-writer reducer for distributed source candidates.

Worker completion order is ignored.  This reducer requires exactly one verified
receipt for every manifest job, consumes repositories by manifest ordinal and
documents by the worker's canonical order, applies token-id exact+near dedup in
one SQLite writer, and starts materialize/route/pack only after global selection
has finished.  The whole reducer tree is staged and atomically published; a
crash leaves no resumable-but-ambiguous global dedup ledger.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.data.atomic_publish import atomic_output_directory  # noqa: E402
from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    load_json_object,
    require_exact_fields,
    require_int,
    require_sha256,
    sha256_file,
)
from scripts.distributed_data_prep.source_manifest import (  # noqa: E402
    PRE_GLOBAL_SCHEMA,
    load_source_manifest,
    validate_source_manifest,
)
from scripts.distributed_data_prep.source_worker import (  # noqa: E402
    CANONICAL_DOCUMENT_ORDER,
    GcloudObjectStore,
    ObjectStore,
    _canonical_sort_key,
    validate_quarantine_receipt_file,
    validate_worker_receipt,
)

SOURCE_REDUCER_RECEIPT_SCHEMA = "cppmega.distributed_source_reducer_receipt_v1"
DEDUP_POLICY = {
    "unit": "enriched_document_token_ids_v1",
    "exact": "sha1_token_ids_v1",
    "near": {
        "enabled": True,
        "threshold": 0.7,
        "num_perm": 256,
        "shingle_k": 5,
    },
    "order": "manifest_ordinal_then_canonical_enriched_json_v1",
    "writers": 1,
}


class DedupWriter(Protocol):
    def seen_exact_tokens(self, token_ids: Sequence[int]) -> bool: ...

    def seen_near_tokens(self, token_ids: Sequence[int]) -> bool: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


DedupFactory = Callable[[Path], DedupWriter]
TokenizerFactory = Callable[[Path], Any]


def _default_dedup_factory(path: Path) -> DedupWriter:
    from tools.clang_indexer.dedup_store import DedupStore

    return DedupStore(str(path), near=True, commit_every=2000)


def _default_tokenizer_factory(path: Path) -> Any:
    from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer

    return load_cppmega_tokenizer(path)


def load_worker_receipts(
    manifest: Mapping[str, object],
    receipt_paths: Sequence[Path],
    *,
    manifest_file_sha256: str | None = None,
) -> tuple[tuple[dict[str, object], str], ...]:
    """Validate exact manifest coverage and return receipts by ordinal."""

    plan = validate_source_manifest(manifest)
    jobs = plan["repositories"]
    assert isinstance(jobs, list)
    by_ordinal: dict[int, tuple[dict[str, object], str]] = {}
    for path in receipt_paths:
        raw, receipt = load_json_object(path, where=f"source worker receipt {path}")
        assignment = receipt.get("assignment")
        if not isinstance(assignment, Mapping):
            raise ContractError(f"source worker receipt has no assignment: {path}")
        ordinal = require_int(
            assignment.get("ordinal"), where=f"source worker receipt {path} ordinal"
        )
        if ordinal >= len(jobs):
            raise ContractError(f"source worker receipt ordinal is out of range: {path}")
        if ordinal in by_ordinal:
            raise ContractError(f"multiple source worker receipts for ordinal {ordinal}")
        validated = validate_worker_receipt(receipt, manifest=plan, job=jobs[ordinal])
        if (
            manifest_file_sha256 is not None
            and validated["manifest_file_sha256"] != manifest_file_sha256
        ):
            raise ContractError(
                f"source worker receipt binds different manifest bytes: {path}"
            )
        by_ordinal[ordinal] = (validated, hashlib.sha256(raw).hexdigest())
    missing = sorted(set(range(len(jobs))) - set(by_ordinal))
    if missing:
        raise ContractError(f"source worker receipt coverage is incomplete: {missing[:20]}")
    return tuple(by_ordinal[ordinal] for ordinal in range(len(jobs)))


def _validate_source_snapshot(
    job: Mapping[str, object], receipt: Mapping[str, object]
) -> None:
    source = job["source"]
    snapshot = receipt["source_snapshot"]
    if not isinstance(source, Mapping) or not isinstance(snapshot, Mapping):
        raise ContractError("source snapshot binding is malformed")
    if source["kind"] == "git_mirror":
        if (
            snapshot.get("kind") != "git_mirror"
            or snapshot.get("remote_url") != source["remote_url"]
            or snapshot.get("expected_commit") != source["expected_commit"]
            or snapshot.get("resolved_commit") != source["expected_commit"]
        ):
            raise ContractError(f"Git source snapshot drifted for {job['repo']}")
        if source.get("expected_tree") is not None and snapshot.get("tree") != source["expected_tree"]:
            raise ContractError(f"Git source tree drifted for {job['repo']}")
        refs = snapshot.get("refs")
        objects = snapshot.get("objects")
        if not isinstance(refs, Mapping) or not isinstance(objects, Mapping):
            raise ContractError(f"Git source inventory is missing for {job['repo']}")
        require_sha256(refs.get("sha256"), where=f"{job['repo']} refs sha256")
        require_sha256(
            objects.get("inventory_sha256"),
            where=f"{job['repo']} object inventory sha256",
        )
        require_int(refs.get("count"), where=f"{job['repo']} refs count")
        require_int(objects.get("count"), where=f"{job['repo']} object count", minimum=1)
    elif source["kind"] == "immutable_gcs_tar":
        obj = snapshot.get("object")
        if (
            snapshot.get("kind") != "immutable_gcs_tar"
            or not isinstance(obj, Mapping)
            or obj.get("uri") != source["uri"]
            or str(obj.get("generation")) != str(source["generation"])
            or obj.get("sha256") != source["sha256"]
        ):
            raise ContractError(f"immutable GCS source snapshot drifted for {job['repo']}")
    else:
        raise ContractError(f"unsupported source kind in reducer: {source['kind']}")


def _iter_zstd_jsonl(path: Path):
    process = subprocess.Popen(
        ["zstd", "-dc", "--", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        for line_number, raw in enumerate(process.stdout, 1):
            if not raw.endswith(b"\n"):
                raise ContractError(f"candidate line {line_number} is truncated")
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError(f"candidate line {line_number} is invalid JSON") from exc
            if not isinstance(value, dict):
                raise ContractError(f"candidate line {line_number} is not an object")
            canonical = canonical_json_bytes(value)
            if raw != canonical + b"\n":
                raise ContractError(f"candidate line {line_number} is not canonical JSON")
            yield line_number, value, canonical
    finally:
        process.stdout.close()
    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"zstd candidate decode failed: {stderr[-8000:].decode(errors='replace')}"
        )


def _write_deduped_repository(
    artifact: Path,
    destination: Path,
    *,
    receipt: Mapping[str, object],
    tokenizer: Any,
    dedup: DedupWriter,
) -> dict[str, object]:
    candidate = receipt["candidate"]
    assert isinstance(candidate, Mapping)
    expected_documents = int(candidate["documents"])
    expected_stream_sha = str(candidate["canonical_stream_sha256"])
    stream_digest = hashlib.sha256()
    accepted_digest = hashlib.sha256()
    documents = 0
    accepted = 0
    dropped_exact = 0
    dropped_near = 0
    previous_key: str | None = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as output:
            for _line_number, document, canonical in _iter_zstd_jsonl(artifact):
                payload_sha = hashlib.sha256(canonical).hexdigest()
                sort_key = _canonical_sort_key(document, payload_sha)
                if previous_key is not None and sort_key < previous_key:
                    raise ContractError("worker candidate document order drifted")
                previous_key = sort_key
                stream_digest.update(canonical)
                stream_digest.update(b"\n")
                documents += 1
                text = document.get("text")
                if not isinstance(text, str) or not text:
                    raise ContractError("candidate document has no non-empty text")
                token_ids = tokenizer.encode(text)
                if not token_ids:
                    raise ContractError("candidate document tokenized to an empty sequence")
                if dedup.seen_exact_tokens(token_ids):
                    dropped_exact += 1
                    continue
                if dedup.seen_near_tokens(token_ids):
                    dropped_near += 1
                    continue
                output.write(canonical)
                output.write(b"\n")
                accepted_digest.update(canonical)
                accepted_digest.update(b"\n")
                accepted += 1
    if documents != expected_documents or stream_digest.hexdigest() != expected_stream_sha:
        raise ContractError("worker candidate count or canonical stream digest drifted")
    dedup.commit()
    return {
        "documents": documents,
        "accepted": accepted,
        "dropped_exact": dropped_exact,
        "dropped_near": dropped_near,
        "accepted_stream_sha256": accepted_digest.hexdigest(),
        "accepted_gzip_sha256": sha256_file(destination),
        "accepted_gzip_bytes": destination.stat().st_size,
    }


def _dedup_database_receipt(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(
        f"file:{path}?mode=rw",
        timeout=300.0,
        isolation_level=None,
        uri=True,
    )
    transaction_open = False
    try:
        connection.execute("PRAGMA busy_timeout=300000")
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
        if journal_mode.lower() != "wal":
            raise ContractError(
                f"global dedup database journal mode is not WAL: {journal_mode}"
            )
        checkpoint_row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint_row is None or len(checkpoint_row) != 3:
            raise ContractError("global dedup WAL checkpoint returned an invalid result")
        busy, log_frames, checkpointed_frames = map(int, checkpoint_row)
        wal_path = Path(str(path) + "-wal")
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0
        checkpoint = {
            "mode": "TRUNCATE",
            "busy": busy,
            "log_frames": log_frames,
            "checkpointed_frames": checkpointed_frames,
            "wal_size_bytes": wal_size,
        }
        expected_checkpoint = {
            "mode": "TRUNCATE",
            "busy": 0,
            "log_frames": 0,
            "checkpointed_frames": 0,
            "wal_size_bytes": 0,
        }
        if checkpoint != expected_checkpoint:
            raise ContractError(
                f"global dedup WAL is not fully checkpointed: {checkpoint}"
            )

        # Keep an exclusive read transaction across integrity validation and the
        # byte hash so no second SQLite connection can mutate the DB between the
        # two.  The reducer owns this freshly staged database, so contention is a
        # contract violation rather than something to paper over.
        connection.execute("BEGIN EXCLUSIVE")
        transaction_open = True
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise ContractError(f"global dedup integrity_check failed: {integrity!r}")
        tables = {
            str(name): int(connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            for name in (
                "exact",
                "minhash",
                "lsh",
                "dedup_meta",
                "chunk_claims",
                "dedup_stages",
                "exact_stage",
                "minhash_stage",
                "lsh_stage",
                "chunk_claims_stage",
            )
        }
        stat_before = path.stat()
        database_sha256 = sha256_file(path)
        stat_after = path.stat()
        if (
            stat_before.st_dev,
            stat_before.st_ino,
            stat_before.st_size,
            stat_before.st_mtime_ns,
        ) != (
            stat_after.st_dev,
            stat_after.st_ino,
            stat_after.st_size,
            stat_after.st_mtime_ns,
        ):
            raise ContractError("global dedup database changed while hashing")
        connection.execute("COMMIT")
        transaction_open = False
    finally:
        if transaction_open:
            connection.execute("ROLLBACK")
        connection.close()
    sidecars = [
        sidecar.name
        for sidecar in (Path(str(path) + "-wal"), Path(str(path) + "-shm"))
        if sidecar.exists()
    ]
    if sidecars:
        raise ContractError(
            f"global dedup database retains SQLite sidecars after close: {sidecars}"
        )
    if any(
        tables[name]
        for name in (
            "dedup_stages",
            "exact_stage",
            "minhash_stage",
            "lsh_stage",
            "chunk_claims_stage",
        )
    ):
        raise ContractError("global reducer dedup database retains staged rows")
    return {
        "path": path.name,
        "size_bytes": stat_after.st_size,
        "sha256": database_sha256,
        "checkpoint": checkpoint,
        "sidecars": [],
        "integrity_check": "ok",
        "tables": tables,
        "policy": DEDUP_POLICY,
    }


def _pack_reduced_repositories(
    accepted: Sequence[tuple[Mapping[str, object], Path, Mapping[str, object]]],
    *,
    packed_root: Path,
    target_lengths: Sequence[int],
    memory_limit_gb: float,
) -> list[dict[str, object]]:
    """Reuse the proven materialize/route/pack stages after global selection."""

    from scripts import streaming_reindex as sr
    from scripts import streaming_reindex_commits as src

    lengths = sorted({int(length) for length in target_lengths})
    if not lengths or lengths[0] <= 0:
        raise ContractError("target lengths must be positive")
    materialize_budget = sr.lossless_materialize_budget(lengths)
    _bucket_for, route_by_fit = sr._route_by_fit_impl()
    packed_receipts: list[dict[str, object]] = []
    for job, accepted_path, selection in accepted:
        repo = str(job["repo"])
        project_id = str(job["project_id"])
        if int(selection["accepted"]) == 0:
            packed_receipts.append(
                {
                    "ordinal": job["ordinal"],
                    "repo": repo,
                    "project_id": project_id,
                    "empty_after_global_dedup": True,
                    "lengths": {},
                }
            )
            continue
        work = packed_root / ".work" / f"{int(job['ordinal']):05d}-{repo}"
        work.mkdir(parents=True, exist_ok=False)
        tokenized = sr.stage_materialize(
            repo,
            accepted_path,
            work,
            memory_limit_gb,
            project_id=project_id,
            max_tokens=materialize_budget,
            fixed_shape_max_tokens=lengths[-1],
        )
        materialize_stats = sr.read_materialize_stats(
            tokenized, fixed_shape_max_tokens=lengths[-1]
        )
        routed = route_by_fit(tokenized, lengths, work / "routed", repo=repo)
        if not routed:
            raise ContractError(f"no globally deduped documents routed for {repo}")
        length_receipts: dict[str, object] = {}
        for length, routed_path in sorted(routed.items()):
            packed = sr.stage_pack(repo, routed_path, length, work)
            src.recompress_zstd_max(packed)
            filename = sr.code_output_filename(repo)
            destination = packed_root / str(length) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(packed, destination)
            stats = sr._parquet_stats(destination, length)
            length_receipts[str(length)] = {
                **stats,
                "path": destination.relative_to(packed_root.parent).as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "compression": "ZSTD",
            }
        packed_receipts.append(
            {
                "ordinal": job["ordinal"],
                "repo": repo,
                "project_id": project_id,
                "empty_after_global_dedup": False,
                "lengths": length_receipts,
                "materialize_stats": materialize_stats,
            }
        )
        shutil.rmtree(work)
    work_root = packed_root / ".work"
    if work_root.exists():
        work_root.rmdir()
    return packed_receipts


def reduce_source_candidates(
    manifest: Mapping[str, object],
    receipt_paths: Sequence[Path],
    *,
    manifest_file_sha256: str,
    output_root: Path,
    scratch_root: Path,
    tokenizer_path: Path,
    object_store: ObjectStore,
    dedup_factory: DedupFactory = _default_dedup_factory,
    tokenizer_factory: TokenizerFactory = _default_tokenizer_factory,
    pack: bool = True,
    memory_limit_gb: float = 24.0,
) -> dict[str, object]:
    """Reduce an exact worker receipt set and atomically publish the result."""

    plan = validate_source_manifest(manifest)
    require_sha256(manifest_file_sha256, where="manifest_file_sha256")
    if output_root.exists():
        raise ContractError(
            f"reducer output already exists; use a new run root or verify it: {output_root}"
        )
    pipeline = plan["pipeline"]
    assert isinstance(pipeline, Mapping)
    if tokenizer_path.is_symlink() or not tokenizer_path.is_file():
        raise ContractError(f"tokenizer is not a regular file: {tokenizer_path}")
    if sha256_file(tokenizer_path) != pipeline["tokenizer_sha256"]:
        raise ContractError("reducer tokenizer hash drifted from the source manifest")
    receipts = load_worker_receipts(
        plan,
        receipt_paths,
        manifest_file_sha256=manifest_file_sha256,
    )
    jobs = plan["repositories"]
    assert isinstance(jobs, list)
    scratch_root.mkdir(parents=True, exist_ok=True)

    with atomic_output_directory(output_root) as staged_root:
        accepted_root = staged_root / "accepted"
        packed_root = staged_root / "packed"
        dedup_path = staged_root / "global_dedup.sqlite"
        tokenizer = tokenizer_factory(tokenizer_path)
        dedup = dedup_factory(dedup_path)
        selection_records: list[dict[str, object]] = []
        accepted_for_pack: list[
            tuple[Mapping[str, object], Path, Mapping[str, object]]
        ] = []
        receipt_bindings: list[dict[str, object]] = []
        try:
            for ordinal, ((receipt, receipt_sha), job) in enumerate(zip(receipts, jobs)):
                if int(job["ordinal"]) != ordinal:
                    raise ContractError("reducer manifest order drifted")
                _validate_source_snapshot(job, receipt)
                artifact = receipt["artifact"]
                assert isinstance(artifact, Mapping)
                quarantine_artifact = receipt["quarantine_artifact"]
                assert isinstance(quarantine_artifact, Mapping)
                with tempfile.TemporaryDirectory(
                    prefix=f"reduce-{ordinal:05d}-{job['repo']}-", dir=scratch_root
                ) as raw_tmp:
                    candidate_path = Path(raw_tmp) / "candidate.jsonl.zst"
                    quarantine_path = Path(raw_tmp) / "source-quarantine.json"
                    object_store.download(
                        str(quarantine_artifact["uri"]),
                        quarantine_path,
                        generation=str(quarantine_artifact["generation"]),
                    )
                    if (
                        quarantine_path.stat().st_size
                        != int(quarantine_artifact["size_bytes"])
                        or sha256_file(quarantine_path)
                        != quarantine_artifact["sha256"]
                    ):
                        raise ContractError(
                            f"downloaded quarantine receipt drifted for {job['repo']}"
                        )
                    quarantine_payload = validate_quarantine_receipt_file(
                        quarantine_path,
                        project_id=str(job["project_id"]),
                        manifest_sha256=str(pipeline["quarantine_manifest_sha256"]),
                    )
                    object_store.download(
                        str(artifact["uri"]),
                        candidate_path,
                        generation=str(artifact["generation"]),
                    )
                    if (
                        candidate_path.stat().st_size != int(artifact["size_bytes"])
                        or sha256_file(candidate_path) != artifact["sha256"]
                    ):
                        raise ContractError(
                            f"downloaded candidate bytes drifted for {job['repo']}"
                        )
                    accepted_path = accepted_root / f"{ordinal:05d}-{job['repo']}.jsonl.gz"
                    selection = _write_deduped_repository(
                        candidate_path,
                        accepted_path,
                        receipt=receipt,
                        tokenizer=tokenizer,
                        dedup=dedup,
                    )
                record = {
                    "ordinal": ordinal,
                    "repo": job["repo"],
                    "project_id": job["project_id"],
                    **selection,
                }
                selection_records.append(record)
                accepted_for_pack.append((job, accepted_path, selection))
                receipt_bindings.append(
                    {
                        "ordinal": ordinal,
                        "repo": job["repo"],
                        "receipt_sha256": receipt_sha,
                        "artifact_sha256": artifact["sha256"],
                        "artifact_generation": str(artifact["generation"]),
                        "quarantine_sha256": quarantine_artifact["sha256"],
                        "quarantine_generation": str(
                            quarantine_artifact["generation"]
                        ),
                        "quarantine_summary_sha256": canonical_sha256(
                            {
                                key: quarantine_payload[key]
                                for key in (
                                    "project_id",
                                    "manifest_sha256",
                                    "candidate_count_before_quarantine",
                                    "candidate_count_after_quarantine",
                                    "quarantined_count",
                                    "external_reference_omissions",
                                    "parse_recovery",
                                )
                            }
                        ),
                    }
                )
        finally:
            dedup.close()

        dedup_receipt = _dedup_database_receipt(dedup_path)
        packed = (
            _pack_reduced_repositories(
                accepted_for_pack,
                packed_root=packed_root,
                target_lengths=pipeline["target_lengths"],
                memory_limit_gb=memory_limit_gb,
            )
            if pack
            else []
        )
        totals = {
            "candidate_documents": sum(int(row["documents"]) for row in selection_records),
            "accepted_documents": sum(int(row["accepted"]) for row in selection_records),
            "dropped_exact": sum(int(row["dropped_exact"]) for row in selection_records),
            "dropped_near": sum(int(row["dropped_near"]) for row in selection_records),
        }
        reducer_receipt: dict[str, object] = {
            "schema": SOURCE_REDUCER_RECEIPT_SCHEMA,
            "status": "complete",
            "manifest_sha256": plan["manifest_sha256"],
            "manifest_file_sha256": manifest_file_sha256,
            "repository_order_sha256": plan["repository_order_sha256"],
            "worker_receipts": receipt_bindings,
            "worker_receipts_sha256": canonical_sha256(receipt_bindings),
            "selection": selection_records,
            "selection_sha256": canonical_sha256(selection_records),
            "totals": totals,
            "dedup": dedup_receipt,
            "packing": {
                "executed": pack,
                "target_lengths": pipeline["target_lengths"],
                "repositories": packed,
                "repositories_sha256": canonical_sha256(packed),
            },
            # Global document exact+near is complete and packing is downstream of
            # it.  Function/chunk parity with the old in-indexer policy remains an
            # explicit release gate instead of being silently mislabeled ready.
            "training_ready": False,
            "blocking_gates": [
                "semantic_function_and_chunk_dedup_parity",
                "packed_sidecar_validation",
                "megatron_sealing",
            ],
        }
        atomic_write_json(staged_root / "reducer_receipt.json", reducer_receipt)
    return reducer_receipt


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("cppmega/tokenizer/tokenizer.json")
    )
    parser.add_argument("--memory-limit-gb", type=float, default=24.0)
    parser.add_argument("--validate-and-dedup-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest, raw_sha = load_source_manifest(args.manifest)
        reduce_source_candidates(
            manifest,
            args.receipt,
            manifest_file_sha256=raw_sha,
            output_root=args.output_root,
            scratch_root=args.scratch_root,
            tokenizer_path=args.tokenizer,
            object_store=GcloudObjectStore(),
            pack=not args.validate_and_dedup_only,
            memory_limit_gb=args.memory_limit_gb,
        )
    except (ContractError, RuntimeError, OSError, ValueError) as exc:
        parser.exit(2, f"distributed source reducer failed: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "DEDUP_POLICY",
    "SOURCE_REDUCER_RECEIPT_SCHEMA",
    "load_worker_receipts",
    "reduce_source_candidates",
]
