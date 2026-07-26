#!/usr/bin/env python3
"""Build an audited, immutable cppmega macro-routes Megatron bundle.

The live parquet conveyor keeps adding/replacing shards.  This builder first
reflinks or copies stable shards into a private run-local snapshot, audits that
snapshot, then converts every requested sequence bucket with the full token and
graph sidecar contract.  Bucket directories and the final bundle are published
by rename only after validation succeeds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from typing import Iterable, TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from data_prep_parquet_to_megatron import (  # noqa: E402
    DEFAULT_CPPMEGA_GRAPH_SIDECARS,
    DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS,
    _current_generation_directory,
    convert_parquet_to_megatron,
)
from data.publish_megatron_bundle_to_nebius_s3 import (  # noqa: E402
    EXPECTED_BUNDLE_TOKENIZER_CONTRACT,
    EXPECTED_VOCAB_SIZE,
    _objective_source_snapshot_summary,
    _validate_bundle,
    _validate_prefix_manifest_contract,
    _validate_tokenizer_directory,
)
from cppmega.megatron.objective_contract import (  # noqa: E402
    load_objective_materialization_artifact,
    validate_materialized_objective_contract,
)
from cppmega.receipt_binding import build_data_producer_binding  # noqa: E402


DEFAULT_BUCKETS = (1024, 2048, 4096, 8192, 16384)
BUILD_PLAN_SCHEMA = "cppmega_macro_routes_build_plan_v1"
CI_MANIFEST_SCHEMA = "cppmega_ci_fixed_buckets_manifest_v3"
CI_BUCKET_MANIFEST_SCHEMA = "cppmega_ci_fixed_bucket_v2"
CI_LOG_COMPLETION_SCHEMA = "cppmega_ci_log_extraction_v1"
BUNDLE_KNOWN_LIMITATIONS = (
    "the source snapshot is the manifest-complete subset; failed or live "
    "conveyor units are excluded",
)
DTYPE_SIZES = {
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "int32": 4,
    "int64": 8,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _copy_private(source: Path, target: Path) -> None:
    """Create a copy-on-write clone when available, otherwise a private copy."""

    clone_command = (
        ["cp", "-c", str(source), str(target)]
        if sys.platform == "darwin"
        else ["cp", "--reflink=auto", "--", str(source), str(target)]
    )
    try:
        cloned = subprocess.run(clone_command, capture_output=True, check=False)
    except OSError:
        cloned = None
    if cloned is None or cloned.returncode != 0:
        target.unlink(missing_ok=True)
        shutil.copy2(source, target)


def _copy_stable_snapshot_file(source: Path, target: Path) -> dict[str, object]:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"snapshot source is not a regular file: {source}")
    if target.exists():
        raise RuntimeError(f"snapshot target already exists: {target}")

    staged = target.with_name(
        f".{target.name}.staging-{os.getpid()}-{threading.get_ident()}"
    )
    staged.unlink(missing_ok=True)
    try:
        before = source.stat()
        before_sha256 = _sha256(source)
        after_initial_hash = source.stat()
        if _stat_signature(before) != _stat_signature(after_initial_hash):
            raise RuntimeError(f"source changed during pre-copy hashing: {source}")

        _copy_private(source, staged)
        after_copy = source.stat()
        after_sha256 = _sha256(source)
        after_final_hash = source.stat()
        staged_before_hash = staged.stat()
        staged_sha256 = _sha256(staged)
        staged_after_hash = staged.stat()

        source_signatures = {
            _stat_signature(before),
            _stat_signature(after_initial_hash),
            _stat_signature(after_copy),
            _stat_signature(after_final_hash),
        }
        if len(source_signatures) != 1 or before_sha256 != after_sha256:
            raise RuntimeError(f"source changed while snapshotting: {source}")
        if (
            _stat_signature(staged_before_hash) != _stat_signature(staged_after_hash)
            or staged_before_hash.st_size != before.st_size
            or staged_sha256 != before_sha256
        ):
            raise RuntimeError(f"private snapshot copy is unstable: {source}")
        if (
            staged_after_hash.st_dev == after_final_hash.st_dev
            and staged_after_hash.st_ino == after_final_hash.st_ino
        ):
            raise RuntimeError(f"snapshot copy unexpectedly shares an inode: {source}")
        os.replace(staged, target)
        return {
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "sha256": before_sha256,
        }
    finally:
        staged.unlink(missing_ok=True)


def _acquire_build_lock(output_dir: Path) -> TextIO:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f".{output_dir.name}.build.lock")
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"bundle build already active: {output_dir}") from error
    return handle


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _git_sha(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _producer_binding_from_conveyor(
    conveyor_manifest: dict[str, object],
    *,
    cppmega_commit: str,
    cppmega_tree_sha256: str,
    cppmega_mlx_commit: str,
    cppmega_mlx_tree_sha256: str,
) -> dict[str, object]:
    revision = conveyor_manifest.get("code_revision")
    if not isinstance(revision, dict):
        raise RuntimeError("conveyor manifest lacks a code_revision receipt")
    if int(revision.get("schema_version", 0)) < 2 or revision.get("dirty") is not False:
        raise RuntimeError(
            "conveyor manifest requires a clean code revision schema v2 receipt"
        )
    if revision.get("producer_role") != "canonical_source_conveyor":
        raise RuntimeError(
            "conveyor manifest code revision has an unsupported producer role"
        )
    if revision.get("repository_identity") != "cppmega":
        raise RuntimeError(
            "conveyor manifest code revision is not bound to cppmega"
        )
    if revision.get("git_commit") != cppmega_commit:
        raise RuntimeError(
            "conveyor manifest cppmega commit does not match the reviewed "
            "bundle-builder commit"
        )
    if revision.get("source_tree_sha256") != cppmega_tree_sha256:
        raise RuntimeError(
            "conveyor manifest cppmega source tree does not match the reviewed "
            "bundle-builder tree"
        )
    indexer = revision.get("indexer_provenance")
    if not isinstance(indexer, dict):
        raise RuntimeError("conveyor manifest lacks clang indexer provenance")
    if indexer.get("schema") != "cppmega_indexer_dependency_binding_v1":
        raise RuntimeError("conveyor manifest clang indexer provenance is unsupported")
    closure = indexer.get("dependency_closure_sha256")
    if closure != revision.get("indexer_dependency_closure_sha256"):
        raise RuntimeError(
            "conveyor manifest clang indexer dependency closure is inconsistent"
        )
    return build_data_producer_binding(
        cppmega_commit=cppmega_commit,
        cppmega_tree_sha256=cppmega_tree_sha256,
        cppmega_mlx_commit=cppmega_mlx_commit,
        cppmega_mlx_tree_sha256=cppmega_mlx_tree_sha256,
        clang_indexer_sha256=str(indexer.get("source_sha256", "")),
        clang_indexer_dependency_closure_sha256=str(closure or ""),
    )


def _stable_parquets(
    root: Path,
    bucket: int,
    min_age_seconds: float,
    allowed_names: set[str],
) -> list[Path]:
    bucket_root = root / str(bucket)
    if not bucket_root.is_dir():
        raise FileNotFoundError(f"missing parquet bucket: {bucket_root}")
    cutoff_ns = time.time_ns() - int(min_age_seconds * 1_000_000_000)
    return [
        path
        for path in sorted(bucket_root.glob("*.parquet"))
        if path.name in allowed_names and path.stat().st_mtime_ns <= cutoff_ns
    ]


def _load_manifest_allowlist(
    manifest_path: Path,
    buckets: tuple[int, ...],
) -> tuple[dict[tuple[str, int], dict[str, int]], dict[str, object]]:
    manifest_bytes = manifest_path.read_bytes()
    blob = json.loads(manifest_bytes)
    done = blob.get("done")
    if not isinstance(done, dict):
        raise ValueError(f"{manifest_path}: missing done map")
    failed = blob.get("failed")
    if not isinstance(failed, dict):
        raise ValueError(f"{manifest_path}: missing failed map")
    if failed:
        sample = sorted(str(key) for key in failed)[:10]
        raise RuntimeError(
            f"{manifest_path}: refusing to freeze a conveyor with "
            f"{len(failed)} failed units; sample={sample}"
        )
    allowed: dict[tuple[str, int], dict[str, int]] = {
        (kind, bucket): {} for kind in ("code", "commits") for bucket in buckets
    }
    for key, info in done.items():
        if not isinstance(info, dict):
            continue
        lengths = info.get("lengths")
        if not isinstance(lengths, dict) or not lengths:
            continue
        if key.endswith("::code"):
            kind = "code"
            repo = key[: -len("::code")]
            filename = f"{repo}.parquet"
        elif "::r" in key:
            kind = "commits"
            repo, start = key.rsplit("::r", 1)
            try:
                int(start)
            except ValueError:
                continue
            filename = f"{repo}_r{start}.parquet"
        else:
            continue
        for bucket in buckets:
            length_info = lengths.get(str(bucket))
            if length_info is None:
                continue
            if not isinstance(length_info, dict):
                raise RuntimeError(
                    f"manifest has malformed length metadata for {key}/{bucket}"
                )
            rows = length_info.get("rows")
            if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
                raise RuntimeError(
                    f"manifest has invalid row count for {key}/{bucket}: {rows!r}"
                )
            bucket_rows = allowed[(kind, bucket)]
            if filename in bucket_rows:
                raise RuntimeError(
                    f"manifest maps duplicate shard {kind}/{bucket}/{filename}"
                )
            bucket_rows[filename] = rows
    if any(not names for names in allowed.values()):
        empty = [
            f"{kind}/{bucket}" for (kind, bucket), names in allowed.items() if not names
        ]
        raise RuntimeError(f"manifest has no completed files for: {', '.join(empty)}")
    metadata = {
        "path": str(manifest_path.resolve()),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "done_units": len(done),
        "failed_units": 0,
        "allowlist_counts": {
            f"{kind}/{bucket}": len(names)
            for (kind, bucket), names in sorted(allowed.items())
        },
    }
    return allowed, metadata


def _load_ci_manifest_allowlist(
    manifest_path: Path,
    ci_root: Path,
    buckets: tuple[int, ...],
    *,
    cppmega_mlx_commit: str,
    cppmega_mlx_tree_sha256: str,
) -> tuple[dict[tuple[str, int], dict[str, int]], dict[str, object]]:
    """Validate the immutable CI generation and return its exact shard allowlist."""

    manifest_path = manifest_path.resolve()
    ci_root = ci_root.resolve()
    if manifest_path != ci_root / "manifest.json":
        raise RuntimeError(
            "CI manifest must be the generation-root manifest: "
            f"{ci_root / 'manifest.json'}"
        )
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid CI manifest {manifest_path}: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != CI_MANIFEST_SCHEMA
        or manifest.get("kind") != "ci"
        or manifest.get("seq_lengths") != list(buckets)
    ):
        raise RuntimeError(
            f"{manifest_path}: unsupported CI manifest or bucket ladder"
        )
    verification = manifest.get("verification")
    if (
        not isinstance(verification, dict)
        or verification.get("fixed_width_all_rows") is not True
        or verification.get("source_tokens_equal_fragment_tokens") is not True
        or verification.get("unexpected_rejects") != 0
        or verification.get("packing_overflow_docs") != 0
    ):
        raise RuntimeError(f"{manifest_path}: CI verification is not green")
    counters = manifest.get("counters")
    if not isinstance(counters, dict):
        raise RuntimeError(f"{manifest_path}: CI counters are missing")
    zero_counters = (
        "malformed_json_rows",
        "empty_text_docs",
        "zero_token_docs",
        "normalization_rejects",
        "packing_overflow_docs",
        "unexpected_rejects",
    )
    if any(
        not isinstance(counters.get(name), int)
        or isinstance(counters.get(name), bool)
        or int(counters[name]) != 0
        for name in zero_counters
    ):
        raise RuntimeError(f"{manifest_path}: CI reject counters are not zero")
    for name in (
        "input_docs",
        "tokenized_docs",
        "source_tokens",
        "fragment_tokens",
        "fragments",
    ):
        value = counters.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise RuntimeError(f"{manifest_path}: invalid CI counter {name}={value!r}")
    if (
        counters["input_docs"] != counters["tokenized_docs"]
        or counters["source_tokens"] != counters["fragment_tokens"]
    ):
        raise RuntimeError(f"{manifest_path}: CI token/document conservation failed")
    split_policy = manifest.get("split_policy")
    if (
        not isinstance(split_policy, dict)
        or split_policy.get("schema")
        != "cppmega_ci_lossless_token_fragmentation_v1"
        or split_policy.get("token_loss") != 0
        or split_policy.get("cross_boundary_edges_are_counted") is not True
    ):
        raise RuntimeError(f"{manifest_path}: CI split policy is not lossless")
    producer = manifest.get("producer")
    revision = producer.get("code_revision") if isinstance(producer, dict) else None
    if (
        not isinstance(producer, dict)
        or producer.get("script") != "tokenize_ci_enriched.py"
        or not isinstance(producer.get("script_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", producer["script_sha256"])
        or not isinstance(revision, dict)
        or revision.get("schema") != "cppmega_ci_code_revision_v2"
        or revision.get("schema_version") != 2
        or revision.get("repository_identity") != "cppmega.mlx"
        or revision.get("dirty") is not False
        or not isinstance(revision.get("git_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision["git_commit"])
        or not isinstance(revision.get("source_tree_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", revision["source_tree_sha256"]
        )
        or revision.get("status_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise RuntimeError(f"{manifest_path}: CI producer revision is not clean/bound")
    if revision["git_commit"] != cppmega_mlx_commit:
        raise RuntimeError(
            f"{manifest_path}: CI producer commit does not match the reviewed "
            "cppmega.mlx commit"
        )
    if revision["source_tree_sha256"] != cppmega_mlx_tree_sha256:
        raise RuntimeError(
            f"{manifest_path}: CI producer source tree does not match the "
            "reviewed cppmega.mlx tree"
        )

    source_inventory = manifest.get("source_inventory")
    if (
        not isinstance(source_inventory, list)
        or len(source_inventory) != 2
    ):
        raise RuntimeError(f"{manifest_path}: CI source inventory is missing")
    inventory_digest = _canonical_sha256(source_inventory)
    if manifest.get("source_inventory_sha256") != inventory_digest:
        raise RuntimeError(f"{manifest_path}: CI source inventory digest drifted")
    for record in source_inventory:
        if (
            not isinstance(record, dict)
            or set(record) != {"name", "path", "size", "mtime_ns", "sha256"}
            or not isinstance(record["name"], str)
            or not record["name"]
            or not isinstance(record["path"], str)
            or not record["path"]
            or not isinstance(record["size"], int)
            or isinstance(record["size"], bool)
            or record["size"] < 1
            or not isinstance(record["mtime_ns"], int)
            or isinstance(record["mtime_ns"], bool)
            or record["mtime_ns"] < 1
            or not isinstance(record["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        ):
            raise RuntimeError(f"{manifest_path}: malformed CI source inventory")
    inventory_names = [str(record["name"]) for record in source_inventory]
    if inventory_names != [
        "ci_logs_enriched.jsonl",
        "ci_paired_enriched.jsonl",
    ]:
        raise RuntimeError(
            f"{manifest_path}: CI source inventory is not the canonical ordered "
            f"pair: {inventory_names}"
        )
    source_completion = manifest.get("source_completion")
    if (
        not isinstance(source_completion, dict)
        or source_completion.get("schema") != CI_LOG_COMPLETION_SCHEMA
        or source_completion.get("status") != "complete"
        or source_completion.get("unresolved_count") != 0
    ):
        raise RuntimeError(
            f"{manifest_path}: CI log extraction completion is missing or incomplete"
        )
    completion_counts: dict[str, int] = {}
    for name in (
        "unique_job_count",
        "fetched_count",
        "expired_count",
        "too_short_count",
    ):
        value = source_completion.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise RuntimeError(
                f"{manifest_path}: invalid CI extraction counter {name}"
            )
        completion_counts[name] = value
    if completion_counts["unique_job_count"] != (
        completion_counts["fetched_count"]
        + completion_counts["expired_count"]
        + completion_counts["too_short_count"]
    ):
        raise RuntimeError(
            f"{manifest_path}: CI extraction job accounting drifted"
        )
    if counters["input_docs"] < completion_counts["fetched_count"]:
        raise RuntimeError(
            f"{manifest_path}: tokenized CI docs omit fetched log documents"
        )
    completion_output = source_completion.get("output")
    completion_state = source_completion.get("state")
    logs_inventory = source_inventory[0]
    if (
        not isinstance(completion_output, dict)
        or completion_output.get("row_count")
        != completion_counts["fetched_count"]
        or completion_output.get("size") != logs_inventory["size"]
        or completion_output.get("sha256") != logs_inventory["sha256"]
        or not isinstance(completion_state, dict)
        or completion_state.get("row_count")
        != completion_counts["unique_job_count"]
        or not isinstance(completion_state.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", completion_state["sha256"])
        or not isinstance(source_completion.get("receipt_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            source_completion["receipt_sha256"],
        )
    ):
        raise RuntimeError(
            f"{manifest_path}: CI extraction artifact binding drifted"
        )
    expired_jobs = source_completion.get("expired_jobs")
    if (
        not isinstance(expired_jobs, list)
        or len(expired_jobs) != completion_counts["expired_count"]
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("job_id"), int)
            or "HTTP 410" not in str(item.get("detail", ""))
            for item in expired_jobs
        )
    ):
        raise RuntimeError(
            f"{manifest_path}: CI expired-job accounting lacks HTTP 410 evidence"
        )

    raw_buckets = manifest.get("buckets")
    if not isinstance(raw_buckets, dict) or set(raw_buckets) != {
        str(bucket) for bucket in buckets
    }:
        raise RuntimeError(f"{manifest_path}: CI bucket receipts are incomplete")
    allowed: dict[tuple[str, int], dict[str, int]] = {}
    total_fragments = 0
    total_valid_tokens = 0
    allowed_relative_parquets: set[Path] = set()
    for bucket in buckets:
        receipt = raw_buckets[str(bucket)]
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != CI_BUCKET_MANIFEST_SCHEMA
            or receipt.get("kind") != "ci"
            or receipt.get("bucket_seq_length") != bucket
            or receipt.get("fixed_width_verified") is not True
            or receipt.get("packing_overflow_docs") != 0
        ):
            raise RuntimeError(
                f"{manifest_path}: invalid CI bucket receipt for {bucket}"
            )
        fragments = receipt.get("fragments")
        rows = receipt.get("packed_rows")
        valid_tokens = receipt.get("valid_tokens")
        capacity_tokens = receipt.get("capacity_tokens")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in (fragments, rows, valid_tokens, capacity_tokens)
        ):
            raise RuntimeError(
                f"{manifest_path}: CI bucket {bucket} has invalid counts"
            )
        if capacity_tokens != rows * bucket or valid_tokens > capacity_tokens:
            raise RuntimeError(
                f"{manifest_path}: CI bucket {bucket} capacity drifted"
            )
        parquet = receipt.get("parquet")
        bucket_manifest = receipt.get("manifest")
        if not isinstance(parquet, dict) or not isinstance(bucket_manifest, dict):
            raise RuntimeError(
                f"{manifest_path}: CI bucket {bucket} lacks file bindings"
            )
        expected_parquet = Path(str(bucket)) / f"ci_packed_{bucket}.parquet"
        expected_bucket_manifest = Path(str(bucket)) / "manifest.json"
        if (
            Path(str(parquet.get("path"))) != expected_parquet
            or Path(str(bucket_manifest.get("path"))) != expected_bucket_manifest
        ):
            raise RuntimeError(
                f"{manifest_path}: CI bucket {bucket} paths are not canonical"
            )
        parquet_path = ci_root / expected_parquet
        bucket_manifest_path = ci_root / expected_bucket_manifest
        if (
            parquet_path.is_symlink()
            or not parquet_path.is_file()
            or parquet_path.stat().st_size != parquet.get("size")
            or _sha256(parquet_path) != parquet.get("sha256")
            or bucket_manifest_path.is_symlink()
            or not bucket_manifest_path.is_file()
            or _sha256(bucket_manifest_path) != bucket_manifest.get("sha256")
        ):
            raise RuntimeError(
                f"{manifest_path}: CI bucket {bucket} artifact binding drifted"
            )
        persisted_bucket = json.loads(
            bucket_manifest_path.read_text(encoding="utf-8")
        )
        expected_persisted = {
            key: value
            for key, value in receipt.items()
            if key != "manifest"
        }
        expected_persisted["parquet"] = {
            **receipt["parquet"],
            "path": parquet_path.name,
        }
        if persisted_bucket != expected_persisted:
            raise RuntimeError(
                f"{manifest_path}: CI bucket {bucket} manifest drifted"
            )
        allowed[("ci", bucket)] = {parquet_path.name: int(rows)}
        allowed_relative_parquets.add(expected_parquet)
        total_fragments += int(fragments)
        total_valid_tokens += int(valid_tokens)

    actual_relative_parquets = {
        path.relative_to(ci_root)
        for path in ci_root.glob("*/*.parquet")
        if path.is_file()
    }
    if actual_relative_parquets != allowed_relative_parquets:
        raise RuntimeError(
            f"{manifest_path}: CI parquet inventory differs from manifest: "
            f"extra={sorted(actual_relative_parquets - allowed_relative_parquets)} "
            f"missing={sorted(allowed_relative_parquets - actual_relative_parquets)}"
        )
    if (
        total_fragments != counters["fragments"]
        or total_valid_tokens != counters["fragment_tokens"]
    ):
        raise RuntimeError(f"{manifest_path}: CI bucket totals drifted")
    metadata = {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "schema": CI_MANIFEST_SCHEMA,
        "source_inventory_sha256": inventory_digest,
        "producer_revision": revision,
        "producer_script_sha256": producer["script_sha256"],
        "source_completion": source_completion,
        "input_docs": counters["input_docs"],
        "fragments": counters["fragments"],
        "valid_tokens": counters["fragment_tokens"],
        "cross_boundary_chunk_edges": counters.get(
            "cross_boundary_chunk_edges", 0
        ),
        "cross_boundary_token_edges": counters.get(
            "cross_boundary_token_edges", 0
        ),
        "allowlist_counts": {
            f"ci/{bucket}": len(allowed[("ci", bucket)])
            for bucket in buckets
        },
    }
    return allowed, metadata


def _snapshot_sources(
    *,
    code_root: Path,
    commit_root: Path,
    ci_root: Path | None = None,
    snapshot_root: Path,
    buckets: tuple[int, ...],
    min_age_seconds: float,
    hash_jobs: int,
    allowed: dict[tuple[str, int], dict[str, int]],
    conveyor_manifest: dict[str, object],
    ci_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest_path = snapshot_root / "source_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema") != "cppmega_parquet_snapshot_v1"
            or existing.get("conveyor_manifest") != conveyor_manifest
            or existing.get("ci_manifest") != ci_manifest
        ):
            raise RuntimeError("existing source snapshot does not match build inputs")
        return existing
    if snapshot_root.exists():
        raise RuntimeError(
            f"incomplete source snapshot exists without manifest: {snapshot_root}"
        )

    candidates: list[tuple[str, int, Path, Path, int]] = []
    source_roots = [("code", code_root), ("commits", commit_root)]
    if ci_root is not None:
        if ci_manifest is None:
            raise RuntimeError("CI snapshot root requires a validated CI manifest")
        source_roots.append(("ci", ci_root))
    elif ci_manifest is not None:
        raise RuntimeError("CI manifest supplied without a CI snapshot root")
    for kind, source_root in source_roots:
        for bucket in buckets:
            source_paths = _stable_parquets(
                source_root,
                bucket,
                0.0 if kind == "ci" else min_age_seconds,
                set(allowed[(kind, bucket)]),
            )
            if not source_paths:
                raise RuntimeError(f"no stable {kind}/{bucket} parquet shards")
            missing = set(allowed[(kind, bucket)]) - {
                path.name for path in source_paths
            }
            if missing:
                raise RuntimeError(
                    f"{kind}/{bucket}: {len(missing)} manifest-backed shards are "
                    f"missing or younger than min-age: {sorted(missing)[:10]}"
                )
            kind_dir = snapshot_root / kind / str(bucket)
            kind_dir.mkdir(parents=True, exist_ok=True)
            for source in source_paths:
                target = kind_dir / source.name
                candidates.append(
                    (
                        kind,
                        bucket,
                        source,
                        target,
                        allowed[(kind, bucket)][source.name],
                    )
                )

    def copy_candidate(
        candidate: tuple[str, int, Path, Path, int],
    ) -> dict[str, object]:
        kind, bucket, source, target, rows = candidate
        copied = _copy_stable_snapshot_file(source, target)
        return {
            "kind": kind,
            "bucket": bucket,
            "source": str(source.resolve()),
            "snapshot": str(target.relative_to(snapshot_root)),
            "size": copied["size"],
            "mtime_ns": copied["mtime_ns"],
            "rows": rows,
            "sha256": copied["sha256"],
        }

    with ThreadPoolExecutor(max_workers=max(1, hash_jobs)) as pool:
        records = list(pool.map(copy_candidate, candidates))

    by_kind_bucket: dict[str, int] = {}
    for record in records:
        key = f"{record['kind']}/{record['bucket']}"
        by_kind_bucket[key] = by_kind_bucket.get(key, 0) + 1
    expected_counts = {
        f"{kind}/{bucket}": len(allowed[(kind, bucket)])
        for kind, _source_root in source_roots
        for bucket in buckets
    }
    if by_kind_bucket != expected_counts:
        raise RuntimeError(
            "source snapshot lost files during private-copy publication: "
            f"actual={by_kind_bucket} expected={expected_counts}"
        )
    payload: dict[str, object] = {
        "schema": "cppmega_parquet_snapshot_v1",
        "created_at": _utc_now(),
        "min_age_seconds": min_age_seconds,
        "file_count": len(records),
        "by_kind_bucket": by_kind_bucket,
        "conveyor_manifest": conveyor_manifest,
        "ci_manifest": ci_manifest,
        "files": records,
    }
    _write_json_atomic(manifest_path, payload)
    return payload


def _write_repaired_snapshot_manifest(
    *,
    snapshot_root: Path,
    source_manifest: dict[str, object],
    repair_receipt: dict[str, object],
    hash_jobs: int,
) -> dict[str, object]:
    output_path = snapshot_root / "repaired_manifest.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("source_manifest_sha256") != _canonical_sha256(
            source_manifest
        ):
            raise RuntimeError("existing repaired snapshot source binding mismatch")
        existing_records = existing.get("files")
        if not isinstance(existing_records, list):
            raise RuntimeError("existing repaired snapshot has no files list")
        for record in existing_records:
            if not isinstance(record, dict):
                raise RuntimeError("existing repaired snapshot file record is invalid")
            path = snapshot_root / str(record["snapshot"])
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != int(record["size"])
                or _sha256(path) != str(record["snapshot_sha256"])
            ):
                raise RuntimeError(f"existing repaired snapshot file drifted: {path}")
        return existing
    scan_by_path = {
        str(Path(str(record["path"])).resolve()): record
        for record in repair_receipt.get("file_scans", [])
        if isinstance(record, dict) and record.get("path")
    }
    changed_paths = set(scan_by_path)
    source_records = source_manifest.get("files")
    if not isinstance(source_records, list):
        raise RuntimeError("source snapshot manifest has no files list")

    def add_repaired_hash(record: dict[str, object]) -> dict[str, object]:
        snapshot_path = (snapshot_root / str(record["snapshot"])).resolve()
        changed = str(snapshot_path) in changed_paths
        snapshot_size = snapshot_path.stat().st_size
        snapshot_sha256 = _sha256(snapshot_path)
        actually_changed = (
            snapshot_size != int(record["size"])
            or snapshot_sha256 != str(record["sha256"])
        )
        if changed != actually_changed:
            raise RuntimeError(
                "boundary repair receipt does not match snapshot bytes: "
                f"{snapshot_path}"
            )
        if changed and int(scan_by_path[str(snapshot_path)].get("rows", -1)) != int(
            record["rows"]
        ):
            raise RuntimeError(
                f"boundary repair row count drifted for snapshot: {snapshot_path}"
            )
        return {
            "kind": record["kind"],
            "bucket": record["bucket"],
            "snapshot": record["snapshot"],
            "size": snapshot_size,
            "rows": record["rows"],
            "source_sha256": record["sha256"],
            "snapshot_sha256": snapshot_sha256,
            "boundary_repaired": changed,
        }

    with ThreadPoolExecutor(max_workers=max(1, hash_jobs)) as pool:
        records = list(pool.map(add_repaired_hash, source_records))
    known_paths = {
        str((snapshot_root / str(record["snapshot"])).resolve())
        for record in source_records
        if isinstance(record, dict)
    }
    if changed_paths - known_paths:
        raise RuntimeError(
            "boundary repair receipt references files outside the source snapshot"
        )
    payload: dict[str, object] = {
        "schema": "cppmega_repaired_parquet_snapshot_v1",
        "created_at": _utc_now(),
        "source_manifest_sha256": _canonical_sha256(source_manifest),
        "file_count": len(records),
        "changed_files": len(changed_paths),
        "files": records,
    }
    _write_json_atomic(output_path, payload)
    return payload


def _run_boundary_repair(
    *,
    snapshot_root: Path,
    repair_script: Path,
    repair_root: Path,
    buckets: tuple[int, ...],
    workers: int,
) -> dict[str, object]:
    receipt_path = repair_root / "packed_document_boundary_repair.json"
    if not receipt_path.exists():
        roots = [snapshot_root / "code", snapshot_root / "commits"]
        if (snapshot_root / "ci").is_dir():
            roots.append(snapshot_root / "ci")
        root_args = [
            argument
            for root in roots
            for argument in ("--root", str(root))
        ]
        subprocess.run(
            [
                sys.executable,
                str(repair_script),
                *root_args,
                "--buckets",
                ",".join(str(bucket) for bucket in buckets),
                "--workers",
                str(max(1, workers)),
                "--receipt",
                str(receipt_path),
            ],
            check=True,
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "cppmega_packed_document_boundary_repair_v1":
        raise RuntimeError("unsupported boundary repair receipt")
    return receipt


def _run_snapshot_audit(
    *,
    snapshot_root: Path,
    audit_script: Path,
    audit_root: Path,
    buckets: tuple[int, ...],
    workers: int,
    snapshot_manifest_sha256: str,
) -> dict[str, object]:
    receipt_path = audit_root / "sidecar_parquet_audit.json"
    binding_path = audit_root / "snapshot_binding.json"
    if receipt_path.exists():
        if not binding_path.exists():
            raise RuntimeError(
                "existing audit receipt is not bound to the repaired snapshot manifest"
            )
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if (
            binding.get("schema") != "cppmega_snapshot_audit_binding_v1"
            or binding.get("snapshot_manifest_sha256") != snapshot_manifest_sha256
            or binding.get("audit_receipt_sha256") != _sha256(receipt_path)
        ):
            raise RuntimeError("existing audit receipt snapshot binding mismatch")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    else:
        empty_pr_root = audit_root / "empty_standalone_pr_root"
        empty_pr_root.mkdir(parents=True, exist_ok=True)
        ci_args = (
            ["--ci-root", str(snapshot_root / "ci")]
            if (snapshot_root / "ci").is_dir()
            else []
        )
        subprocess.run(
            [
                sys.executable,
                str(audit_script),
                "--code-root",
                str(snapshot_root / "code"),
                "--commit-root",
                str(snapshot_root / "commits"),
                "--pr-root",
                str(empty_pr_root),
                *ci_args,
                "--buckets",
                ",".join(str(bucket) for bucket in buckets),
                "--workers",
                str(max(1, workers)),
                "--vocab-size",
                "65536",
                "--out-dir",
                str(audit_root),
            ],
            check=True,
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _write_json_atomic(
            binding_path,
            {
                "schema": "cppmega_snapshot_audit_binding_v1",
                "created_at": _utc_now(),
                "snapshot_manifest_sha256": snapshot_manifest_sha256,
                "audit_receipt_sha256": _sha256(receipt_path),
            },
        )

    total = receipt.get("total", {})
    if int(total.get("bad_files", -1)) or int(total.get("bad_rows", -1)):
        raise RuntimeError(
            f"snapshot parquet audit is not green: bad_files={total.get('bad_files')} "
            f"bad_rows={total.get('bad_rows')}"
        )
    if receipt.get("bad_files"):
        raise RuntimeError("snapshot audit contains bad_files entries")
    return receipt


def _parse_objective_artifacts(
    values: list[str], buckets: tuple[int, ...]
) -> dict[int, Path]:
    artifacts: dict[int, Path] = {}
    for value in values:
        bucket_text, separator, path_text = value.partition("=")
        if not separator or not bucket_text or not path_text:
            raise ValueError(
                "--objective-artifact must use BUCKET=/path/to/"
                "objective_materialization.json"
            )
        try:
            bucket = int(bucket_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid objective artifact bucket {bucket_text!r}"
            ) from exc
        if bucket in artifacts:
            raise ValueError(f"duplicate objective artifact for bucket {bucket}")
        artifacts[bucket] = Path(path_text).resolve()
    expected = set(buckets)
    actual = set(artifacts)
    if actual != expected:
        raise ValueError(
            "objective artifact buckets must exactly match --buckets: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    for path in artifacts.values():
        load_objective_materialization_artifact(path)
    return artifacts


def _objective_expected_counts(path: Path) -> dict[str, int]:
    artifact = load_objective_materialization_artifact(path)
    totals = artifact.contract.payload["totals"]
    samples = int(totals["samples"])
    return {
        "rows": samples,
        # shifted_lm_document_v1 appends one zero-loss sentinel per sample.
        "valid_tokens": int(totals["input_tokens"]) + samples,
        "trained_tokens": int(totals["loss_tokens"]),
    }


def _validate_objective_source_binding(
    *,
    objective_artifact_path: Path,
    repaired_snapshot_manifest: dict[str, object],
    bucket: int,
) -> dict[str, object]:
    artifact = load_objective_materialization_artifact(objective_artifact_path)
    binding = artifact.contract.payload.get("source_snapshot")
    if not isinstance(binding, dict):
        raise RuntimeError(
            f"bucket {bucket}: objective contract has no source_snapshot binding"
        )
    try:
        source_summary = _objective_source_snapshot_summary(binding, bucket=bucket)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    files = binding["files"]
    objective_records: list[dict[str, object]] = []
    for record in files:
        relative = Path(str(record["path"]))
        if (
            relative.is_absolute()
            or len(relative.parts) != 3
            or relative.parts[0] not in {"code", "commits", "ci"}
            or relative.parts[1] != str(bucket)
            or relative.name in {"", ".", ".."}
        ):
            raise RuntimeError(
                f"bucket {bucket}: objective source path is not canonical: {relative}"
            )
        objective_records.append(
            {
                "kind": relative.parts[0],
                "bucket": bucket,
                "path": relative.as_posix(),
                "size": int(record["size_bytes"]),
                "sha256": str(record["sha256"]),
                "rows": int(record["rows"]),
            }
        )

    repaired_files = repaired_snapshot_manifest.get("files")
    if not isinstance(repaired_files, list):
        raise RuntimeError("repaired snapshot manifest has no files list")
    snapshot_records = [
        {
            "kind": str(record["kind"]),
            "bucket": int(record["bucket"]),
            "path": str(record["snapshot"]),
            "size": int(record["size"]),
            "sha256": str(record["snapshot_sha256"]),
            "rows": int(record["rows"]),
        }
        for record in repaired_files
        if isinstance(record, dict) and int(record.get("bucket", -1)) == bucket
    ]
    if objective_records != snapshot_records:
        mismatch_index = next(
            (
                index
                for index, pair in enumerate(
                    zip(objective_records, snapshot_records, strict=False)
                )
                if pair[0] != pair[1]
            ),
            min(len(objective_records), len(snapshot_records)),
        )
        raise RuntimeError(
            f"bucket {bucket}: objective sources do not match repaired snapshot "
            f"at ordered record {mismatch_index}; "
            f"objective_count={len(objective_records)} "
            f"snapshot_count={len(snapshot_records)}"
        )
    return source_summary


def _read_mmididx(idx_path: Path) -> dict[str, int]:
    with idx_path.open("rb") as fh:
        if fh.read(9) != b"MMIDIDX\x00\x00":
            raise RuntimeError(f"invalid MMIDIDX header: {idx_path}")
        version = struct.unpack("<Q", fh.read(8))[0]
        dtype_code = struct.unpack("<B", fh.read(1))[0]
        sequences = struct.unpack("<Q", fh.read(8))[0]
        documents = struct.unpack("<Q", fh.read(8))[0]
        sizes = fh.read(sequences * 4)
        if len(sizes) != sequences * 4:
            raise RuntimeError(f"truncated MMIDIDX sizes: {idx_path}")
        import numpy as np

        sizes_array = np.frombuffer(sizes, dtype=np.int32)
        expected_size = 34 + sequences * 4 + sequences * 8 + documents * 8
        actual_size = idx_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"MMIDIDX size mismatch for {idx_path}: {actual_size} != {expected_size}"
            )
        return {
            "version": version,
            "dtype_code": dtype_code,
            "sequences": sequences,
            "documents": documents,
            "tokens": int(sizes_array.sum(dtype=np.int64)),
        }


def _verify_prefix(prefix: Path, expected: dict[str, int]) -> dict[str, object]:
    manifest_path = prefix.with_suffix(".json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    token_count = int(data["token_count"])
    document_count = int(data["document_count"])
    trained_tokens = int(data["trained_token_count"])
    if token_count != expected["valid_tokens"]:
        raise RuntimeError(
            f"{prefix}: token_count {token_count} != {expected['valid_tokens']}"
        )
    if document_count != expected["rows"]:
        raise RuntimeError(
            f"{prefix}: document_count {document_count} != {expected['rows']}"
        )
    if trained_tokens != expected["trained_tokens"]:
        raise RuntimeError(
            f"{prefix}: trained_token_count {trained_tokens} != {expected['trained_tokens']}"
        )
    token_dtype_size = 2 if data["dtype"] == "uint16" else DTYPE_SIZES[data["dtype"]]
    if prefix.with_suffix(".bin").stat().st_size != token_count * token_dtype_size:
        raise RuntimeError(f"{prefix}: token binary size mismatch")

    index = _read_mmididx(prefix.with_suffix(".idx"))
    if index["tokens"] != token_count or index["sequences"] != document_count:
        raise RuntimeError(f"{prefix}: MMIDIDX token/document count mismatch")
    if index["documents"] != document_count + 1:
        raise RuntimeError(f"{prefix}: MMIDIDX document sentinel mismatch")

    prefix_dir = prefix.parent
    side_paths = data.get("side_channel_paths", {})
    expected_side_names = {name for name, _dtype in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS}
    if set(side_paths) != expected_side_names:
        raise RuntimeError(f"{prefix}: incomplete token sidecar profile")
    for name, spec in side_paths.items():
        side_path = prefix_dir / spec["path"]
        expected_bytes = token_count * DTYPE_SIZES[spec["dtype"]]
        if side_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"{prefix}: {name} size {side_path.stat().st_size} != {expected_bytes}"
            )

    graph_paths = data.get("graph_sidecar_paths", {})
    expected_graph_names = {
        name for name, _kind, _dtype in DEFAULT_CPPMEGA_GRAPH_SIDECARS
    }
    if set(graph_paths) != expected_graph_names:
        raise RuntimeError(f"{prefix}: incomplete graph sidecar profile")
    for name, spec in graph_paths.items():
        offsets_path = prefix_dir / spec["offsets_path"]
        data_path = prefix_dir / spec["data_path"]
        if offsets_path.stat().st_size != (document_count + 1) * 8:
            raise RuntimeError(f"{prefix}: {name} offsets size mismatch")
        tail = int(spec.get("shape_tail", [1])[0])
        expected_bytes = int(spec["item_count"]) * tail * DTYPE_SIZES[spec["dtype"]]
        if data_path.stat().st_size != expected_bytes:
            raise RuntimeError(f"{prefix}: {name} data size mismatch")

    source_platform = data.get("source_platform_sidecar")
    if not isinstance(source_platform, dict):
        raise RuntimeError(f"{prefix}: compact source platform sidecar missing")
    if source_platform.get("schema") != "cppmega_source_platform_v1":
        raise RuntimeError(f"{prefix}: unsupported source platform sidecar schema")
    sequence_offsets = prefix_dir / str(source_platform["sequence_doc_offsets_path"])
    doc_offsets = prefix_dir / str(source_platform["doc_platform_offsets_path"])
    platform_ids = prefix_dir / str(source_platform["platform_ids_path"])
    if sequence_offsets.stat().st_size != (document_count + 1) * 8:
        raise RuntimeError(f"{prefix}: source platform sequence offsets size mismatch")
    source_document_count = int(source_platform["source_document_count"])
    if doc_offsets.stat().st_size != (source_document_count + 1) * 8:
        raise RuntimeError(f"{prefix}: source platform document offsets size mismatch")
    if platform_ids.stat().st_size != int(source_platform["platform_id_count"]) * 2:
        raise RuntimeError(f"{prefix}: source platform IDs size mismatch")
    objective = data.get("objective_contract")
    if objective is None:
        raise RuntimeError(f"{prefix}: objective_contract is required")
    validated_objective = validate_materialized_objective_contract(
        objective,
        base_dir=str(prefix.parent),
        document_count=document_count,
    )
    if validated_objective.payload["totals"]["samples"] != document_count:
        raise RuntimeError(
            f"{prefix}: objective sample count does not match document_count"
        )
    _validate_prefix_manifest_contract(prefix)
    return data


def _build_bucket(
    *,
    bucket: int,
    data_root: Path,
    objective_artifact_path: Path,
) -> dict[str, object]:
    artifact = load_objective_materialization_artifact(objective_artifact_path)
    expected = _objective_expected_counts(objective_artifact_path)
    final_dir = data_root / f"seq_{bucket}"
    building_dir = data_root / f".seq_{bucket}.building"
    prefix_name = f"cppmega_macro_routes_seq{bucket}_train"
    final_prefix = final_dir / prefix_name
    if final_dir.exists():
        if building_dir.exists():
            shutil.rmtree(building_dir)
        manifest = _verify_prefix(final_prefix, expected)
        _require_bucket_objective_binding(final_prefix, manifest, artifact)
        return {"bucket": bucket, "prefix": str(final_prefix), "manifest": manifest}

    if building_dir.exists():
        shutil.rmtree(building_dir)
    building_dir.mkdir(parents=True)
    building_prefix = building_dir / prefix_name
    convert_parquet_to_megatron(
        input_dir=None,
        output_prefix=str(building_prefix),
        split="all",
        token_column="input_ids",
        length_column="valid_token_count",
        dtype_str="uint16",
        vocab_size=65536,
        writer_backend="mmididx",
        source_platform_sidecar=True,
        objective_artifact_path=str(objective_artifact_path.resolve()),
    )

    generation_dir = _current_generation_directory(building_prefix)
    if generation_dir is None:
        raise RuntimeError(f"{building_prefix}: converter published no generation")
    generation_prefix = generation_dir / prefix_name
    manifest = _verify_prefix(generation_prefix, expected)
    _require_bucket_objective_binding(generation_prefix, manifest, artifact)

    # The converter publishes through an immutable pointer so concurrent readers
    # never observe a mixed generation. A transport bundle must instead contain
    # only regular files. Moving the already-validated generation directory gives
    # the bundle both properties without weakening the publisher's symlink gate.
    os.replace(generation_dir, final_dir)
    shutil.rmtree(building_dir)
    manifest = _verify_prefix(final_prefix, expected)
    _require_bucket_objective_binding(final_prefix, manifest, artifact)
    return {"bucket": bucket, "prefix": str(final_prefix), "manifest": manifest}


def _require_bucket_objective_binding(
    prefix: Path, manifest: dict[str, object], artifact: object
) -> None:
    materialization = manifest.get("objective_materialization")
    if (
        not isinstance(materialization, dict)
        or materialization.get("artifact_set_sha256")
        != getattr(artifact, "artifact_set_sha256")
        or materialization.get("artifact_file_sha256")
        != getattr(artifact, "file_sha256")
    ):
        raise RuntimeError(f"{prefix}: converted objective artifact binding drifted")


def _artifact_records(root: Path, hash_jobs: int) -> list[dict[str, object]]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "snapshot" not in path.relative_to(root).parts
        and path.name != "manifest.json"
    )

    def record(path: Path) -> dict[str, object]:
        return {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }

    with ThreadPoolExecutor(max_workers=max(1, hash_jobs)) as pool:
        return list(pool.map(record, paths))


def _artifact_set_sha256(records: list[dict[str, object]]) -> str:
    canonical = [
        {
            "path": str(record["path"]),
            "size": int(record["size"]),
            "sha256": str(record["sha256"]),
        }
        for record in sorted(records, key=lambda item: str(item["path"]))
    ]
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _stage_tokenizer(tokenizer_dir: Path, bundle_root: Path) -> dict[str, object]:
    source_files = sorted(_validate_tokenizer_directory(tokenizer_dir))
    source_bindings = {
        path.name: (path.stat().st_size, _sha256(path)) for path in source_files
    }

    target = bundle_root / "tokenizer"
    if target.exists() and (target.is_symlink() or not target.is_dir()):
        raise RuntimeError(f"tokenizer staging target is invalid: {target}")
    target.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    for path in source_files:
        staged = target / path.name
        if staged.exists():
            if staged.is_symlink() or not staged.is_file():
                raise RuntimeError(f"staged tokenizer file is invalid: {staged}")
        else:
            shutil.copy2(path, staged)
        expected_size, expected_sha256 = source_bindings[path.name]
        if (
            staged.stat().st_size != expected_size
            or _sha256(staged) != expected_sha256
        ):
            raise RuntimeError(f"staged tokenizer file does not match source: {staged}")
        records.append(
            {
                "path": staged.relative_to(bundle_root).as_posix(),
                "size": expected_size,
                "sha256": expected_sha256,
            }
        )
    if {path.name for path in target.iterdir()} != set(source_bindings):
        raise RuntimeError(f"staged tokenizer contains unexpected files: {target}")
    return {
        "path": "tokenizer",
        "contract": EXPECTED_BUNDLE_TOKENIZER_CONTRACT,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "files": records,
        "artifact_set_sha256": _artifact_set_sha256(records),
    }


def _stage_data_contracts(bundle_root: Path) -> dict[str, dict[str, object]]:
    target = bundle_root / "contracts"
    if target.exists() and (target.is_symlink() or not target.is_dir()):
        raise RuntimeError(f"data-contract staging target is invalid: {target}")
    target.mkdir(exist_ok=True)
    sources = {
        "domain_schema": REPO_ROOT / "data/domain_schema_v1.json",
        "tokenizer_contract": (
            REPO_ROOT / "data/tokenizer_v2/tokenizer_contract_v1.json"
        ),
    }
    descriptors: dict[str, dict[str, object]] = {}
    for name, source in sources.items():
        staged = target / source.name
        expected_size = source.stat().st_size
        expected_sha256 = _sha256(source)
        if staged.exists():
            if staged.is_symlink() or not staged.is_file():
                raise RuntimeError(f"staged data contract is invalid: {staged}")
        else:
            shutil.copy2(source, staged)
        if (
            staged.stat().st_size != expected_size
            or _sha256(staged) != expected_sha256
        ):
            raise RuntimeError(f"staged data contract does not match source: {staged}")
        descriptors[name] = {
            "path": staged.relative_to(bundle_root).as_posix(),
            "size": expected_size,
            "sha256": expected_sha256,
        }
    if {path.name for path in target.iterdir()} != {
        source.name for source in sources.values()
    }:
        raise RuntimeError(f"staged data contracts contain unexpected files: {target}")
    return descriptors


def _portable_bucket_results(
    bundle_root: Path, results: list[dict[str, object]]
) -> list[dict[str, object]]:
    portable: list[dict[str, object]] = []
    for result in results:
        prefix = Path(str(result["prefix"]))
        try:
            relative_prefix = prefix.relative_to(bundle_root)
        except ValueError as error:
            raise RuntimeError(
                f"bucket prefix escapes bundle root: {prefix}"
            ) from error
        normalized = dict(result)
        normalized["prefix"] = relative_prefix.as_posix()
        portable.append(normalized)
    return portable


def _objective_build_records(
    objective_artifacts: dict[int, Path], buckets: tuple[int, ...]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for bucket in buckets:
        artifact = load_objective_materialization_artifact(
            objective_artifacts[bucket]
        )
        records.append(
            {
                "bucket": bucket,
                "artifact_path": str(artifact.path),
                "artifact_set_sha256": artifact.artifact_set_sha256,
                "artifact_file_sha256": artifact.file_sha256,
                "contract_path": str(artifact.contract_path),
                "contract_sha256": artifact.contract.sha256,
                "contract_file_sha256": _sha256(artifact.contract_path),
            }
        )
    return records


def _source_file_records(paths: Iterable[Path], root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(paths):
        records.append(
            {
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _create_build_plan(
    *,
    args: argparse.Namespace,
    buckets: tuple[int, ...],
    output_dir: Path,
    objective_artifacts: dict[int, Path],
    conveyor_manifest: dict[str, object],
    ci_manifest: dict[str, object],
) -> dict[str, object]:
    objective_records = _objective_build_records(objective_artifacts, buckets)
    tokenizer_dir = args.tokenizer_dir.resolve()
    tokenizer_records = _source_file_records(
        _validate_tokenizer_directory(tokenizer_dir), tokenizer_dir
    )
    contract_paths = (
        REPO_ROOT / "data/domain_schema_v1.json",
        REPO_ROOT / "data/tokenizer_v2/tokenizer_contract_v1.json",
    )
    converter_path = REPO_ROOT / "scripts/data_prep_parquet_to_megatron.py"
    plan = {
        "output_dir": str(output_dir),
        "source_roots": {
            "code": str(args.code_root.resolve()),
            "commits": str(args.commit_root.resolve()),
            "ci": str(args.ci_root.resolve()),
        },
        "conveyor_manifest": conveyor_manifest,
        "ci_manifest": ci_manifest,
        "implementation": _producer_binding_from_conveyor(
            conveyor_manifest,
            cppmega_commit=args.cppmega_commit,
            cppmega_tree_sha256=args.cppmega_tree_sha256,
            cppmega_mlx_commit=args.cppmega_mlx_commit,
            cppmega_mlx_tree_sha256=args.cppmega_mlx_tree_sha256,
        ),
        "buckets": list(buckets),
        "min_age_seconds": args.min_age_seconds,
        "keep_snapshot": bool(args.keep_snapshot),
        "objective_artifacts": objective_records,
        "tokenizer": {
            "path": str(tokenizer_dir),
            "files": tokenizer_records,
            "artifact_set_sha256": _artifact_set_sha256(tokenizer_records),
        },
        "data_contracts": _source_file_records(contract_paths, REPO_ROOT),
        "implementation_sha256": {
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "converter": {
                "path": str(converter_path.resolve()),
                "sha256": _sha256(converter_path),
            },
            "audit": {
                "path": str(args.audit_script.resolve()),
                "sha256": _sha256(args.audit_script.resolve()),
            },
            "boundary_repair": {
                "path": str(args.repair_script.resolve()),
                "sha256": _sha256(args.repair_script.resolve()),
            },
        },
    }
    return {
        "schema": BUILD_PLAN_SCHEMA,
        "objective_artifacts_sha256": _canonical_sha256(objective_records),
        "build_plan_sha256": _canonical_sha256(plan),
        "plan": plan,
    }


def _ensure_partial_build_plan(
    partial_dir: Path, expected: dict[str, object]
) -> None:
    plan_path = partial_dir / "build_plan.json"
    if partial_dir.exists():
        if partial_dir.is_symlink() or not partial_dir.is_dir():
            raise RuntimeError(f"partial build path is not a directory: {partial_dir}")
        if plan_path.is_symlink() or not plan_path.is_file():
            raise RuntimeError(
                f"stale partial build has no canonical build plan: {partial_dir}"
            )
        try:
            existing = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"stale partial build has an unreadable build plan: {partial_dir}"
            ) from error
        if not isinstance(existing, dict):
            raise RuntimeError(
                f"stale partial build has an invalid build plan: {partial_dir}"
            )
        if existing != expected:
            raise RuntimeError(
                "stale partial build plan mismatch: "
                f"existing={existing.get('build_plan_sha256')} "
                f"expected={expected.get('build_plan_sha256')} "
                f"existing_objectives={existing.get('objective_artifacts_sha256')} "
                f"expected_objectives={expected.get('objective_artifacts_sha256')}"
            )
        return
    partial_dir.mkdir(parents=True)
    _write_json_atomic(plan_path, expected)


def _assert_build_plan_inputs(
    *,
    args: argparse.Namespace,
    objective_artifacts: dict[int, Path],
    buckets: tuple[int, ...],
    output_dir: Path,
    build_plan: dict[str, object],
) -> None:
    _allowed, conveyor_manifest = _load_manifest_allowlist(
        args.conveyor_manifest.resolve(), buckets
    )
    _ci_allowed, ci_manifest = _load_ci_manifest_allowlist(
        args.ci_manifest.resolve(),
        args.ci_root.resolve(),
        buckets,
        cppmega_mlx_commit=args.cppmega_mlx_commit,
        cppmega_mlx_tree_sha256=args.cppmega_mlx_tree_sha256,
    )
    actual = _create_build_plan(
        args=args,
        buckets=buckets,
        output_dir=output_dir,
        objective_artifacts=objective_artifacts,
        conveyor_manifest=conveyor_manifest,
        ci_manifest=ci_manifest,
    )
    if actual == build_plan:
        return
    if actual.get("objective_artifacts_sha256") != build_plan.get(
        "objective_artifacts_sha256"
    ):
        raise RuntimeError("objective artifacts changed after build plan creation")
    raise RuntimeError("build inputs changed after build plan creation")


def _publish_validated_bundle(
    partial_dir: Path, output_dir: Path, *, hash_jobs: int
) -> None:
    _validate_bundle(partial_dir, hash_jobs)
    os.replace(partial_dir, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-root",
        type=Path,
        default=None,
        help="explicit root containing bucketed code parquet shards",
    )
    parser.add_argument(
        "--commit-root",
        type=Path,
        default=None,
        help="explicit root containing bucketed commit parquet shards",
    )
    parser.add_argument(
        "--ci-root",
        type=Path,
        default=None,
        help="explicit immutable root containing five-bucket CI parquet shards",
    )
    parser.add_argument(
        "--ci-manifest",
        type=Path,
        default=None,
        help=(
            "explicit cppmega_ci_fixed_buckets_manifest_v3 binding the CI "
            "source inventory, lossless splits, shard hashes and reject counters"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/megatron_ready/macro_routes_v1_20260713",
    )
    parser.add_argument(
        "--audit-script",
        type=Path,
        default=REPO_ROOT / "scripts/audit_sidecar_parquet.py",
    )
    parser.add_argument(
        "--repair-script",
        type=Path,
        default=REPO_ROOT / "scripts/repair_packed_document_boundaries.py",
    )
    parser.add_argument(
        "--conveyor-manifest",
        type=Path,
        default=None,
        help="explicit completed conveyor manifest binding the source shards",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=REPO_ROOT / "data/tokenizer_v2",
    )
    parser.add_argument(
        "--cppmega-commit",
        default=None,
        help="exact clean cppmega Git commit used by the bundle builder",
    )
    parser.add_argument(
        "--cppmega-tree-sha256",
        default=None,
        help="SHA-256 of the reviewed cppmega tracked source tree",
    )
    parser.add_argument(
        "--cppmega-mlx-commit",
        default=None,
        help="exact clean cppmega.mlx commit used to produce objective sidecars",
    )
    parser.add_argument(
        "--cppmega-mlx-tree-sha256",
        default=None,
        help="SHA-256 of the reviewed cppmega.mlx tracked source tree",
    )
    parser.add_argument(
        "--objective-artifact",
        action="append",
        default=[],
        metavar="BUCKET=PATH",
        help=(
            "Canonical shard-hashed CASE1 objective artifact for one sequence "
            "bucket. Repeat once for every --buckets entry."
        ),
    )
    parser.add_argument("--buckets", default=",".join(map(str, DEFAULT_BUCKETS)))
    parser.add_argument("--min-age-seconds", type=float, default=120.0)
    parser.add_argument("--audit-workers", type=int, default=8)
    parser.add_argument("--repair-workers", type=int, default=4)
    parser.add_argument("--hash-jobs", type=int, default=4)
    parser.add_argument(
        "--keep-snapshot",
        action="store_true",
        help="retain repaired private parquet snapshot after a successful build",
    )
    return parser


def _require_explicit_source_inputs(args: argparse.Namespace) -> None:
    missing = [
        option
        for attribute, option in (
            ("code_root", "--code-root"),
            ("commit_root", "--commit-root"),
            ("ci_root", "--ci-root"),
            ("ci_manifest", "--ci-manifest"),
            ("conveyor_manifest", "--conveyor-manifest"),
            ("cppmega_commit", "--cppmega-commit"),
            ("cppmega_tree_sha256", "--cppmega-tree-sha256"),
            ("cppmega_mlx_commit", "--cppmega-mlx-commit"),
            ("cppmega_mlx_tree_sha256", "--cppmega-mlx-tree-sha256"),
        )
        if getattr(args, attribute) is None
    ]
    if missing:
        raise SystemExit(
            "bundle source inputs must be explicit: " + ", ".join(missing)
        )


def _run_build(
    *,
    args: argparse.Namespace,
    buckets: tuple[int, ...],
    objective_artifacts: dict[int, Path],
    output_dir: Path,
) -> int:
    partial_dir = output_dir.with_name(f".{output_dir.name}.partial")
    if output_dir.exists():
        raise SystemExit(f"final bundle already exists: {output_dir}")

    allowlist, conveyor_manifest = _load_manifest_allowlist(
        args.conveyor_manifest.resolve(), buckets
    )
    ci_allowlist, ci_manifest = _load_ci_manifest_allowlist(
        args.ci_manifest.resolve(),
        args.ci_root.resolve(),
        buckets,
        cppmega_mlx_commit=args.cppmega_mlx_commit,
        cppmega_mlx_tree_sha256=args.cppmega_mlx_tree_sha256,
    )
    overlap = set(allowlist).intersection(ci_allowlist)
    if overlap:
        raise RuntimeError(f"CI allowlist collides with conveyor kinds: {overlap}")
    allowlist.update(ci_allowlist)
    build_plan = _create_build_plan(
        args=args,
        buckets=buckets,
        output_dir=output_dir,
        objective_artifacts=objective_artifacts,
        conveyor_manifest=conveyor_manifest,
        ci_manifest=ci_manifest,
    )
    _ensure_partial_build_plan(partial_dir, build_plan)

    snapshot_root = partial_dir / "snapshot"
    source_manifest = _snapshot_sources(
        code_root=args.code_root.resolve(),
        commit_root=args.commit_root.resolve(),
        ci_root=args.ci_root.resolve(),
        snapshot_root=snapshot_root,
        buckets=buckets,
        min_age_seconds=args.min_age_seconds,
        hash_jobs=args.hash_jobs,
        allowed=allowlist,
        conveyor_manifest=conveyor_manifest,
        ci_manifest=ci_manifest,
    )
    repair_receipt = _run_boundary_repair(
        snapshot_root=snapshot_root,
        repair_script=args.repair_script.resolve(),
        repair_root=partial_dir / "repair",
        buckets=buckets,
        workers=args.repair_workers,
    )
    repaired_snapshot_manifest = _write_repaired_snapshot_manifest(
        snapshot_root=snapshot_root,
        source_manifest=source_manifest,
        repair_receipt=repair_receipt,
        hash_jobs=args.hash_jobs,
    )
    objective_source_bindings = {
        bucket: _validate_objective_source_binding(
            objective_artifact_path=objective_artifacts[bucket],
            repaired_snapshot_manifest=repaired_snapshot_manifest,
            bucket=bucket,
        )
        for bucket in buckets
    }
    audit_receipt = _run_snapshot_audit(
        snapshot_root=snapshot_root,
        audit_script=args.audit_script.resolve(),
        audit_root=partial_dir / "audit",
        buckets=buckets,
        workers=args.audit_workers,
        snapshot_manifest_sha256=_sha256(snapshot_root / "repaired_manifest.json"),
    )
    _assert_build_plan_inputs(
        args=args,
        objective_artifacts=objective_artifacts,
        buckets=buckets,
        output_dir=output_dir,
        build_plan=build_plan,
    )
    data_root = partial_dir / "data"
    data_root.mkdir(exist_ok=True)
    bucket_results = _portable_bucket_results(
        partial_dir,
        [
            _build_bucket(
                bucket=bucket,
                data_root=data_root,
                objective_artifact_path=objective_artifacts[bucket],
            )
            for bucket in buckets
        ],
    )
    provenance_root = partial_dir / "provenance"
    provenance_root.mkdir(exist_ok=True)
    _write_json_atomic(provenance_root / "source_manifest.json", source_manifest)
    _write_json_atomic(
        provenance_root / "repaired_snapshot_manifest.json",
        repaired_snapshot_manifest,
    )
    staged_ci_manifest = provenance_root / "ci_manifest.json"
    shutil.copy2(args.ci_manifest.resolve(), staged_ci_manifest)
    for bucket in buckets:
        staged_bucket_manifest = provenance_root / "ci" / str(bucket) / "manifest.json"
        staged_bucket_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            args.ci_root.resolve() / str(bucket) / "manifest.json",
            staged_bucket_manifest,
        )
    objective_descriptors: dict[str, dict[str, object]] = {}
    for bucket in buckets:
        artifact = load_objective_materialization_artifact(objective_artifacts[bucket])
        staged_artifact = provenance_root / f"objective_artifact_seq{bucket}.json"
        staged_contract = provenance_root / f"objective_contract_seq{bucket}.json"
        shutil.copy2(artifact.path, staged_artifact)
        shutil.copy2(artifact.contract_path, staged_contract)
        objective_descriptors[str(bucket)] = {
            "artifact_path": staged_artifact.relative_to(partial_dir).as_posix(),
            "artifact_schema": artifact.payload["schema"],
            "artifact_set_sha256": artifact.artifact_set_sha256,
            "artifact_file_sha256": artifact.file_sha256,
            "contract_path": staged_contract.relative_to(partial_dir).as_posix(),
            "contract_schema": artifact.contract.payload["schema"],
            "contract_sha256": artifact.contract.sha256,
            "contract_file_sha256": _sha256(staged_contract),
            "source_snapshot": objective_source_bindings[bucket],
        }
    tokenizer = _stage_tokenizer(args.tokenizer_dir, partial_dir)
    data_contracts = _stage_data_contracts(partial_dir)
    _assert_build_plan_inputs(
        args=args,
        objective_artifacts=objective_artifacts,
        buckets=buckets,
        output_dir=output_dir,
        build_plan=build_plan,
    )
    _ensure_partial_build_plan(partial_dir, build_plan)
    if not args.keep_snapshot:
        shutil.rmtree(snapshot_root)
    artifacts = _artifact_records(partial_dir, args.hash_jobs)
    artifact_set_sha256 = _artifact_set_sha256(artifacts)
    audit_total = audit_receipt["total"]
    manifest = {
        "schema": "cppmega_megatron_bundle_v2",
        "bundle_id": f"{output_dir.name}-{artifact_set_sha256[:16]}",
        "created_at": _utc_now(),
        "tokenizer_contract": EXPECTED_BUNDLE_TOKENIZER_CONTRACT,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "tokenizer": tokenizer,
        "data_contracts": data_contracts,
        "token_column": "input_ids",
        "length_column": "valid_token_count",
        "writer_backend": "mmididx",
        "training_contract": "objective_materialized",
        "build_plan": {
            "path": "build_plan.json",
            "build_plan_sha256": build_plan["build_plan_sha256"],
            "objective_artifacts_sha256": build_plan[
                "objective_artifacts_sha256"
            ],
        },
        "objective_materialization": {
            "schema": "cppmega_bucketed_objective_materializations_v1",
            "buckets": objective_descriptors,
        },
        "buckets": list(buckets),
        "known_limitations": list(BUNDLE_KNOWN_LIMITATIONS),
        "source_snapshot": {
            "file_count": source_manifest["file_count"],
            "manifest": "provenance/source_manifest.json",
            "repaired_manifest": "provenance/repaired_snapshot_manifest.json",
            "ci_manifest": {
                "path": "provenance/ci_manifest.json",
                "sha256": _sha256(staged_ci_manifest),
                "source_inventory_sha256": ci_manifest[
                    "source_inventory_sha256"
                ],
                "input_docs": ci_manifest["input_docs"],
                "fragments": ci_manifest["fragments"],
                "valid_tokens": ci_manifest["valid_tokens"],
                "cross_boundary_chunk_edges": ci_manifest[
                    "cross_boundary_chunk_edges"
                ],
                "cross_boundary_token_edges": ci_manifest[
                    "cross_boundary_token_edges"
                ],
            },
            "local_snapshot_retained": bool(args.keep_snapshot),
        },
        "boundary_repair": {
            "receipt": "repair/packed_document_boundary_repair.json",
            "changed_files": repair_receipt["changed_files"],
            "changed_rows": repair_receipt["changed_rows"],
            "restored_boundaries": repair_receipt["restored_boundaries"],
            "old_trained_tokens": repair_receipt["old_trained_tokens"],
            "new_trained_tokens": repair_receipt["new_trained_tokens"],
        },
        "audit": {
            "receipt": "audit/sidecar_parquet_audit.json",
            "snapshot_binding": "audit/snapshot_binding.json",
            "files": audit_total["files"],
            "rows": audit_total["rows"],
            "valid_tokens": audit_total["valid_tokens"],
            "trained_tokens": audit_total["trained_tokens"],
            "bad_files": audit_total["bad_files"],
            "bad_rows": audit_total["bad_rows"],
        },
        "git": {
            "cppmega": _git_sha(REPO_ROOT),
        },
        "implementation": _producer_binding_from_conveyor(
            conveyor_manifest,
            cppmega_commit=args.cppmega_commit,
            cppmega_tree_sha256=args.cppmega_tree_sha256,
            cppmega_mlx_commit=args.cppmega_mlx_commit,
            cppmega_mlx_tree_sha256=args.cppmega_mlx_tree_sha256,
        ),
        "implementation_sha256": {
            "builder": _sha256(Path(__file__).resolve()),
            "converter": _sha256(
                REPO_ROOT / "scripts/data_prep_parquet_to_megatron.py"
            ),
            "audit": _sha256(args.audit_script.resolve()),
            "boundary_repair": _sha256(args.repair_script.resolve()),
        },
        "bucket_results": bucket_results,
        "artifacts": artifacts,
        "artifact_set_sha256": artifact_set_sha256,
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(int(record["size"]) for record in artifacts),
    }
    _write_json_atomic(partial_dir / "manifest.json", manifest)
    _publish_validated_bundle(partial_dir, output_dir, hash_jobs=args.hash_jobs)
    print(json.dumps({"bundle": str(output_dir), "audit": manifest["audit"]}, indent=2))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _require_explicit_source_inputs(args)
    buckets = tuple(int(value) for value in args.buckets.split(",") if value)
    if not buckets:
        raise SystemExit("at least one bucket is required")
    if len(buckets) != len(set(buckets)):
        raise SystemExit("--buckets must not contain duplicates")
    output_dir = args.output_dir.resolve()
    lock = _acquire_build_lock(output_dir)
    try:
        objective_artifacts = _parse_objective_artifacts(
            args.objective_artifact, buckets
        )
        return _run_build(
            args=args,
            buckets=buckets,
            objective_artifacts=objective_artifacts,
            output_dir=output_dir,
        )
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
