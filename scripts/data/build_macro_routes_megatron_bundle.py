#!/usr/bin/env python3
"""Build an audited, immutable cppmega macro-routes Megatron bundle.

The live parquet conveyor keeps adding/replacing shards.  This builder first
hardlinks only stable shards into a run-local snapshot, audits that snapshot,
then converts every requested sequence bucket with the full token and graph
sidecar contract.  Bucket directories and the final bundle are published by
rename only after validation succeeds.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from data_prep_parquet_to_megatron import (  # noqa: E402
    DEFAULT_CPPMEGA_GRAPH_SIDECARS,
    DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS,
    convert_parquet_to_megatron,
)
from data.publish_megatron_bundle_to_nebius_s3 import (  # noqa: E402
    EXPECTED_BUNDLE_TOKENIZER_CONTRACT,
    EXPECTED_VOCAB_SIZE,
    _validate_prefix_manifest_contract,
    _validate_tokenizer_directory,
)
from cppmega.megatron.objective_contract import (  # noqa: E402
    load_objective_materialization_artifact,
    validate_materialized_objective_contract,
)


DEFAULT_BUCKETS = (1024, 2048, 4096, 8192, 16384)
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
) -> tuple[dict[tuple[str, int], set[str]], dict[str, object]]:
    blob = json.loads(manifest_path.read_text(encoding="utf-8"))
    done = blob.get("done")
    if not isinstance(done, dict):
        raise ValueError(f"{manifest_path}: missing done map")
    allowed = {
        (kind, bucket): set() for kind in ("code", "commits") for bucket in buckets
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
            if str(bucket) in lengths:
                allowed[(kind, bucket)].add(filename)
    if any(not names for names in allowed.values()):
        empty = [
            f"{kind}/{bucket}" for (kind, bucket), names in allowed.items() if not names
        ]
        raise RuntimeError(f"manifest has no completed files for: {', '.join(empty)}")
    metadata = {
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path),
        "done_units": len(done),
        "failed_units": len(blob.get("failed") or {}),
        "allowlist_counts": {
            f"{kind}/{bucket}": len(names)
            for (kind, bucket), names in sorted(allowed.items())
        },
    }
    return allowed, metadata


def _snapshot_sources(
    *,
    code_root: Path,
    commit_root: Path,
    snapshot_root: Path,
    buckets: tuple[int, ...],
    min_age_seconds: float,
    hash_jobs: int,
    allowed: dict[tuple[str, int], set[str]],
    conveyor_manifest: dict[str, object],
) -> dict[str, object]:
    manifest_path = snapshot_root / "source_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if snapshot_root.exists():
        raise RuntimeError(
            f"incomplete source snapshot exists without manifest: {snapshot_root}"
        )

    records: list[dict[str, object]] = []
    for kind, source_root in (("code", code_root), ("commits", commit_root)):
        for bucket in buckets:
            source_paths = _stable_parquets(
                source_root,
                bucket,
                min_age_seconds,
                allowed[(kind, bucket)],
            )
            if not source_paths:
                raise RuntimeError(f"no stable {kind}/{bucket} parquet shards")
            missing = allowed[(kind, bucket)] - {path.name for path in source_paths}
            if missing:
                raise RuntimeError(
                    f"{kind}/{bucket}: {len(missing)} manifest-backed shards are "
                    f"missing or younger than min-age: {sorted(missing)[:10]}"
                )
            kind_dir = snapshot_root / kind / str(bucket)
            kind_dir.mkdir(parents=True, exist_ok=True)
            for source in source_paths:
                before = source.stat()
                target = kind_dir / source.name
                os.link(source, target)
                after = source.stat()
                if (
                    before.st_ino != after.st_ino
                    or before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns
                ):
                    target.unlink(missing_ok=True)
                    continue
                records.append(
                    {
                        "kind": kind,
                        "bucket": bucket,
                        "source": str(source.resolve()),
                        "snapshot": str(target.relative_to(snapshot_root)),
                        "size": before.st_size,
                        "mtime_ns": before.st_mtime_ns,
                    }
                )

    def add_hash(record: dict[str, object]) -> dict[str, object]:
        out = dict(record)
        out["sha256"] = _sha256(snapshot_root / str(record["snapshot"]))
        return out

    with ThreadPoolExecutor(max_workers=max(1, hash_jobs)) as pool:
        records = list(pool.map(add_hash, records))

    by_kind_bucket: dict[str, int] = {}
    for record in records:
        key = f"{record['kind']}/{record['bucket']}"
        by_kind_bucket[key] = by_kind_bucket.get(key, 0) + 1
    expected_counts = {
        f"{kind}/{bucket}": len(allowed[(kind, bucket)])
        for kind in ("code", "commits")
        for bucket in buckets
    }
    if by_kind_bucket != expected_counts:
        raise RuntimeError(
            "source snapshot lost files during hardlink publication: "
            f"actual={by_kind_bucket} expected={expected_counts}"
        )
    payload: dict[str, object] = {
        "schema": "cppmega_parquet_snapshot_v1",
        "created_at": _utc_now(),
        "min_age_seconds": min_age_seconds,
        "file_count": len(records),
        "by_kind_bucket": by_kind_bucket,
        "conveyor_manifest": conveyor_manifest,
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
        return json.loads(output_path.read_text(encoding="utf-8"))
    changed_paths = {
        str(Path(str(record["path"])).resolve())
        for record in repair_receipt.get("file_scans", [])
        if isinstance(record, dict) and record.get("path")
    }
    source_records = source_manifest.get("files")
    if not isinstance(source_records, list):
        raise RuntimeError("source snapshot manifest has no files list")

    def add_repaired_hash(record: dict[str, object]) -> dict[str, object]:
        snapshot_path = (snapshot_root / str(record["snapshot"])).resolve()
        changed = str(snapshot_path) in changed_paths
        return {
            "kind": record["kind"],
            "bucket": record["bucket"],
            "snapshot": record["snapshot"],
            "size": snapshot_path.stat().st_size,
            "source_sha256": record["sha256"],
            "snapshot_sha256": _sha256(snapshot_path) if changed else record["sha256"],
            "boundary_repaired": changed,
        }

    with ThreadPoolExecutor(max_workers=max(1, hash_jobs)) as pool:
        records = list(pool.map(add_repaired_hash, source_records))
    payload: dict[str, object] = {
        "schema": "cppmega_repaired_parquet_snapshot_v1",
        "created_at": _utc_now(),
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
        subprocess.run(
            [
                sys.executable,
                str(repair_script),
                "--root",
                str(snapshot_root / "code"),
                "--root",
                str(snapshot_root / "commits"),
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
    expected_keys = {
        "schema",
        "sequence_length",
        "file_count",
        "row_count",
        "files",
        "sampling",
        "artifact_set_sha256",
    }
    if set(binding) != expected_keys:
        raise RuntimeError(
            f"bucket {bucket}: objective source_snapshot keys drifted: "
            f"{sorted(set(binding) ^ expected_keys)}"
        )
    if binding["schema"] != "cppmega_objective_source_snapshot_v1":
        raise RuntimeError(f"bucket {bucket}: unsupported objective source schema")
    if int(binding["sequence_length"]) != bucket:
        raise RuntimeError(
            f"bucket {bucket}: objective source sequence_length mismatch"
        )
    files = binding["files"]
    if not isinstance(files, list) or int(binding["file_count"]) != len(files):
        raise RuntimeError(f"bucket {bucket}: objective source file_count mismatch")
    source_counter: Counter[tuple[int, str]] = Counter()
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "path", "size_bytes", "sha256", "rows"
        }:
            raise RuntimeError(
                f"bucket {bucket}: malformed objective source file record"
            )
        source_counter[(int(record["size_bytes"]), str(record["sha256"]))] += 1

    repaired_files = repaired_snapshot_manifest.get("files")
    if not isinstance(repaired_files, list):
        raise RuntimeError("repaired snapshot manifest has no files list")
    snapshot_records = [
        record
        for record in repaired_files
        if isinstance(record, dict) and int(record.get("bucket", -1)) == bucket
    ]
    snapshot_counter = Counter(
        (int(record["size"]), str(record["snapshot_sha256"]))
        for record in snapshot_records
    )
    if source_counter != snapshot_counter:
        raise RuntimeError(
            f"bucket {bucket}: objective sources do not match repaired snapshot; "
            f"objective_only={list((source_counter - snapshot_counter).elements())[:5]} "
            f"snapshot_only={list((snapshot_counter - source_counter).elements())[:5]}"
        )

    digest_payload = dict(binding)
    recorded_digest = str(digest_payload.pop("artifact_set_sha256"))
    actual_digest = _artifact_set_sha256(
        [
            {
                "path": str(record["path"]),
                "size": int(record["size_bytes"]),
                "sha256": str(record["sha256"]),
            }
            for record in files
        ]
    )
    if recorded_digest != actual_digest:
        raise RuntimeError(
            f"bucket {bucket}: objective source artifact_set_sha256 mismatch"
        )
    if int(binding["row_count"]) < 1:
        raise RuntimeError(f"bucket {bucket}: objective source row_count must be positive")
    sampling = binding["sampling"]
    if not isinstance(sampling, dict) or sampling.get("mode") != (
        "deterministic_epoch_shuffle_v1"
    ):
        raise RuntimeError(f"bucket {bucket}: unsupported objective source sampling")
    return {
        "schema": binding["schema"],
        "artifact_set_sha256": recorded_digest,
        "file_count": len(files),
        "row_count": int(binding["row_count"]),
        "sampling": sampling,
    }


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
    prefix_name = f"cppmega_macro_routes_seq{bucket}_train"
    final_prefix = final_dir / prefix_name
    if final_dir.exists():
        manifest = _verify_prefix(final_prefix, expected)
        return {"bucket": bucket, "prefix": str(final_prefix), "manifest": manifest}

    building_dir = data_root / f".seq_{bucket}.building"
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
    manifest = _verify_prefix(building_prefix, expected)
    materialization = manifest.get("objective_materialization")
    if (
        not isinstance(materialization, dict)
        or materialization.get("artifact_set_sha256") != artifact.artifact_set_sha256
        or materialization.get("artifact_file_sha256") != artifact.file_sha256
    ):
        raise RuntimeError(
            f"{building_prefix}: converted objective artifact binding drifted"
        )
    os.replace(building_dir, final_dir)
    return {"bucket": bucket, "prefix": str(final_prefix), "manifest": manifest}


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

    target = bundle_root / "tokenizer"
    if target.exists():
        raise RuntimeError(f"tokenizer staging target already exists: {target}")
    target.mkdir()
    records: list[dict[str, object]] = []
    for path in source_files:
        staged = target / path.name
        shutil.copy2(path, staged)
        records.append(
            {
                "path": staged.relative_to(bundle_root).as_posix(),
                "size": staged.stat().st_size,
                "sha256": _sha256(staged),
            }
        )
    return {
        "path": "tokenizer",
        "contract": EXPECTED_BUNDLE_TOKENIZER_CONTRACT,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "files": records,
        "artifact_set_sha256": _artifact_set_sha256(records),
    }


def _stage_data_contracts(bundle_root: Path) -> dict[str, dict[str, object]]:
    target = bundle_root / "contracts"
    target.mkdir()
    sources = {
        "domain_schema": REPO_ROOT / "data/domain_schema_v1.json",
        "tokenizer_contract": (
            REPO_ROOT / "data/tokenizer_v2/tokenizer_contract_v1.json"
        ),
    }
    descriptors: dict[str, dict[str, object]] = {}
    for name, source in sources.items():
        staged = target / source.name
        shutil.copy2(source, staged)
        descriptors[name] = {
            "path": staged.relative_to(bundle_root).as_posix(),
            "size": staged.stat().st_size,
            "sha256": _sha256(staged),
        }
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


def build_arg_parser() -> argparse.ArgumentParser:
    sibling = REPO_ROOT.parent / "cppmega.mlx"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-root",
        type=Path,
        default=sibling / "outputs/reindexed_macro_routes_v1_20260710_135335_code",
    )
    parser.add_argument(
        "--commit-root",
        type=Path,
        default=sibling / "outputs/reindexed_macro_routes_v1_20260710_135335_commits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=sibling / "outputs/megatron_ready/macro_routes_v1_20260713",
    )
    parser.add_argument(
        "--audit-script",
        type=Path,
        default=sibling / "scripts/audit_sidecar_parquet.py",
    )
    parser.add_argument(
        "--repair-script",
        type=Path,
        default=sibling / "scripts/repair_packed_document_boundaries.py",
    )
    parser.add_argument(
        "--conveyor-manifest",
        type=Path,
        default=sibling / "outputs/conveyor/macro_routes_v1_20260710_135335/_done.json",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=REPO_ROOT / "data/tokenizer_v2",
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
        help="retain repaired parquet hardlink/copy snapshot after a successful build",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    buckets = tuple(int(value) for value in args.buckets.split(",") if value)
    if not buckets:
        raise SystemExit("at least one bucket is required")
    objective_artifacts = _parse_objective_artifacts(args.objective_artifact, buckets)
    output_dir = args.output_dir.resolve()
    partial_dir = output_dir.with_name(f".{output_dir.name}.partial")
    if output_dir.exists():
        raise SystemExit(f"final bundle already exists: {output_dir}")
    partial_dir.mkdir(parents=True, exist_ok=True)

    allowlist, conveyor_manifest = _load_manifest_allowlist(
        args.conveyor_manifest.resolve(), buckets
    )

    snapshot_root = partial_dir / "snapshot"
    source_manifest = _snapshot_sources(
        code_root=args.code_root.resolve(),
        commit_root=args.commit_root.resolve(),
        snapshot_root=snapshot_root,
        buckets=buckets,
        min_age_seconds=args.min_age_seconds,
        hash_jobs=args.hash_jobs,
        allowed=allowlist,
        conveyor_manifest=conveyor_manifest,
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
    if not args.keep_snapshot:
        shutil.rmtree(snapshot_root)
    artifacts = _artifact_records(partial_dir, args.hash_jobs)
    artifact_set_sha256 = _artifact_set_sha256(artifacts)
    audit_total = audit_receipt["total"]
    manifest = {
        "schema": "cppmega_megatron_bundle_v1",
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
        "objective_materialization": {
            "schema": "cppmega_bucketed_objective_materializations_v1",
            "buckets": objective_descriptors,
        },
        "buckets": list(buckets),
        "known_limitations": [
            "semantic symbol IDs and some local/global lookups are qname-based; "
            "overloaded or same-qname symbols can collapse until clang USR identity is adopted",
            "this frozen generation has no observed shell, diagnostic, or cross-domain graph edges",
            "the source snapshot is the manifest-complete subset; failed/live conveyor units are excluded",
        ],
        "source_snapshot": {
            "file_count": source_manifest["file_count"],
            "manifest": "provenance/source_manifest.json",
            "repaired_manifest": "provenance/repaired_snapshot_manifest.json",
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
            "cppmega_mlx": _git_sha(REPO_ROOT.parent / "cppmega.mlx"),
        },
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
    os.replace(partial_dir, output_dir)
    print(json.dumps({"bundle": str(output_dir), "audit": manifest["audit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
