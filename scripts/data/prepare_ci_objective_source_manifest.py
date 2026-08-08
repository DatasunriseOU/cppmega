#!/usr/bin/env python3
"""Freeze receipt-bound CI and objective-seed Parquet pools for materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Mapping

import pyarrow.parquet as pq

from scripts.data.build_macro_routes_megatron_bundle import (
    CI_CONTENT_STORE_EXPORT_SCHEMA,
    DEFAULT_BUCKETS,
    PRODUCTION_CI_CONTENT_STORE_EXPORT_SCHEMA,
    REPO_ROOT,
    SUPPORTED_CI_BUCKETS,
    _inventory_regular_files_fail_closed,
    _load_content_store_ci_export_allowlist,
    _sha256,
    _stat_signature,
    _write_json_atomic,
)

MANIFEST_SCHEMA = "cppmega_ci_objective_pool_manifest_v1"
SCHEDULE = "alternate_primary_seed_v1"
_REPAIRED_SNAPSHOT_SCHEMA = "cppmega_repaired_parquet_snapshot_v1"
_CASE5_SHARD_NAME = re.compile(
    r"ci-case5-(train|validation|test)-([0-9]+)-([0-9]{6})\.parquet"
)


def _validate_buckets(buckets: tuple[int, ...]) -> tuple[int, ...]:
    """Accept only the immutable live or large-context CI bucket profile."""

    if buckets not in {DEFAULT_BUCKETS, SUPPORTED_CI_BUCKETS}:
        raise ValueError(
            "objective source manifest requires exactly one of "
            f"{DEFAULT_BUCKETS} or {SUPPORTED_CI_BUCKETS}"
        )
    return buckets


def _stable_seed_record(path: Path, *, root: Path) -> dict[str, object]:
    before = path.stat()
    digest = _sha256(path)
    parquet = pq.ParquetFile(path)
    rows = int(parquet.metadata.num_rows)
    codecs = {
        str(parquet.metadata.row_group(row_group).column(column).compression)
        for row_group in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.num_columns)
    }
    after = path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or rows < 1
        or codecs != {"ZSTD"}
    ):
        raise RuntimeError(f"objective seed parquet is unstable or not ZSTD: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "rows": rows,
        "size_bytes": after.st_size,
        "sha256": digest,
    }


def _load_repaired_ci_records(
    path: Path,
    *,
    ci_root: Path,
    artifacts: Mapping[str, object],
    buckets: tuple[int, ...],
) -> dict[str, dict[str, object]]:
    absolute_path = path.absolute()
    if (
        absolute_path.is_symlink()
        or absolute_path.resolve() != absolute_path
        or not absolute_path.is_file()
    ):
        raise RuntimeError(f"repaired snapshot manifest is not a regular file: {path}")
    path = absolute_path
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if _stat_signature(before) != _stat_signature(after):
        raise RuntimeError(f"repaired snapshot manifest changed while reading: {path}")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid repaired snapshot manifest: {path}") from exc
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema",
            "created_at",
            "source_manifest_sha256",
            "file_count",
            "changed_files",
            "files",
        }
        or payload.get("schema") != _REPAIRED_SNAPSHOT_SCHEMA
        or not isinstance(payload.get("files"), list)
        or payload.get("file_count") != len(payload["files"])
        or not isinstance(payload.get("source_manifest_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(payload["source_manifest_sha256"])
        )
    ):
        raise RuntimeError("repaired snapshot manifest contract is invalid")

    ci_root = ci_root.resolve()
    actual_files = _inventory_regular_files_fail_closed(ci_root)
    expected_files = {
        Path(relative)
        for relative, artifact in artifacts.items()
        if isinstance(artifact, Mapping) and artifact.get("kind") == "case5_parquet"
    }
    repaired: dict[str, dict[str, object]] = {}
    changed_files = 0
    for index, raw_record in enumerate(payload["files"]):
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "kind",
            "bucket",
            "snapshot",
            "size",
            "rows",
            "source_sha256",
            "snapshot_sha256",
            "boundary_repaired",
        }:
            raise RuntimeError(
                f"repaired snapshot file[{index}] contract is invalid"
            )
        if not isinstance(raw_record["boundary_repaired"], bool):
            raise RuntimeError(
                f"repaired snapshot file[{index}] repair flag is invalid"
            )
        changed_files += int(raw_record["boundary_repaired"])
        if raw_record["kind"] != "ci":
            continue
        snapshot = raw_record["snapshot"]
        bucket = raw_record["bucket"]
        if (
            not isinstance(snapshot, str)
            or Path(snapshot).as_posix() != snapshot
            or not isinstance(bucket, int)
            or isinstance(bucket, bool)
            or bucket not in buckets
        ):
            raise RuntimeError(
                f"repaired CI snapshot file[{index}] path/bucket is invalid"
            )
        snapshot_path = Path(snapshot)
        match = _CASE5_SHARD_NAME.fullmatch(snapshot_path.name)
        if (
            len(snapshot_path.parts) != 3
            or snapshot_path.parts[:2] != ("ci", str(bucket))
            or match is None
            or int(match.group(2)) != bucket
        ):
            raise RuntimeError(
                f"repaired CI snapshot file[{index}] is not a canonical CASE5 shard"
            )
        relative = Path(*snapshot_path.parts[1:])
        relative_text = relative.as_posix()
        artifact = artifacts.get(relative_text)
        size = raw_record["size"]
        rows = raw_record["rows"]
        source_sha256 = raw_record["source_sha256"]
        snapshot_sha256 = raw_record["snapshot_sha256"]
        boundary_repaired = raw_record["boundary_repaired"]
        if (
            relative_text in repaired
            or not isinstance(artifact, Mapping)
            or artifact.get("kind") != "case5_parquet"
            or artifact.get("bucket") != bucket
            or artifact.get("split") != match.group(1)
            or artifact.get("rows") != rows
            or artifact.get("sha256") != source_sha256
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            or not isinstance(rows, int)
            or isinstance(rows, bool)
            or rows < 1
            or not isinstance(snapshot_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", snapshot_sha256)
            or not isinstance(boundary_repaired, bool)
        ):
            raise RuntimeError(
                f"repaired CI snapshot binding drifted: {relative_text}"
            )
        actual = ci_root / relative
        if actual.is_symlink() or not actual.is_file() or actual.resolve() != actual:
            raise RuntimeError(
                f"repaired CI snapshot is not a regular canonical file: {actual}"
            )
        actual_before = actual.stat()
        actual_sha256 = _sha256(actual)
        parquet = pq.ParquetFile(actual)
        actual_rows = int(parquet.metadata.num_rows)
        codecs = {
            str(parquet.metadata.row_group(row_group).column(column).compression)
            for row_group in range(parquet.metadata.num_row_groups)
            for column in range(parquet.metadata.num_columns)
        }
        actual_after = actual.stat()
        if (
            _stat_signature(actual_before) != _stat_signature(actual_after)
            or actual_after.st_size != size
            or actual_sha256 != snapshot_sha256
            or actual_rows != rows
            or codecs != {"ZSTD"}
            or boundary_repaired
            != (
                size != artifact.get("byte_size")
                or snapshot_sha256 != source_sha256
            )
        ):
            raise RuntimeError(
                f"repaired CI snapshot bytes drifted: {relative_text}"
            )
        repaired[relative_text] = {
            "path": relative_text,
            "rows": rows,
            "size_bytes": size,
            "sha256": snapshot_sha256,
        }

    if (
        set(repaired) != {path.as_posix() for path in expected_files}
        or actual_files != expected_files
        or payload.get("changed_files") != changed_files
    ):
        raise RuntimeError("repaired CI snapshot inventory differs from CASE5 receipt")
    return repaired


def build_source_pool_manifest(
    *,
    ci_root: Path,
    ci_receipt_path: Path,
    repaired_manifest_path: Path,
    objective_seed_root: Path,
    seed_globs: tuple[str, ...],
    buckets: tuple[int, ...],
    producer: Mapping[str, object],
) -> dict[str, object]:
    buckets = _validate_buckets(buckets)
    ci_root = ci_root.absolute()
    ci_receipt_path = ci_receipt_path.absolute()
    objective_seed_root = objective_seed_root.absolute()
    if (
        ci_root.is_symlink()
        or ci_root.resolve() != ci_root
        or not ci_root.is_dir()
        or ci_receipt_path.is_symlink()
        or ci_receipt_path.resolve() != ci_receipt_path
        or not ci_receipt_path.is_file()
        or objective_seed_root.is_symlink()
        or objective_seed_root.resolve() != objective_seed_root
        or not objective_seed_root.is_dir()
    ):
        raise RuntimeError("objective source inputs must be canonical regular paths")
    receipt_raw = ci_receipt_path.read_bytes()
    try:
        receipt = json.loads(receipt_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid CI export receipt: {ci_receipt_path}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema")
        not in {
            CI_CONTENT_STORE_EXPORT_SCHEMA,
            PRODUCTION_CI_CONTENT_STORE_EXPORT_SCHEMA,
        }
    ):
        raise RuntimeError("objective source manifest requires a CASE5 export receipt")
    allowed, ci_metadata = _load_content_store_ci_export_allowlist(
        manifest_path=ci_receipt_path,
        manifest_bytes=receipt_raw,
        manifest=receipt,
        ci_root=ci_receipt_path.parent,
        buckets=buckets,
    )
    raw_artifacts = receipt.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise RuntimeError("CI export receipt has no artifact inventory")
    artifacts = {
        str(record["path"]): record
        for record in raw_artifacts
        if isinstance(record, dict) and record.get("kind") == "case5_parquet"
    }
    repaired = _load_repaired_ci_records(
        repaired_manifest_path,
        ci_root=ci_root,
        artifacts=artifacts,
        buckets=buckets,
    )
    files_by_sequence_length: dict[str, list[dict[str, object]]] = {}
    for bucket in buckets:
        names = allowed[("ci", bucket)]
        records: list[dict[str, object]] = []
        for name in sorted(names):
            match = _CASE5_SHARD_NAME.fullmatch(name)
            if match is None or int(match.group(2)) != bucket:
                raise RuntimeError(f"CI allowlist has a non-canonical shard: {name}")
            if match.group(1) != "train":
                continue
            relative = f"{bucket}/{name}"
            record = repaired.get(relative)
            if not isinstance(record, dict) or record.get("rows") != names[name]:
                raise RuntimeError(f"CI allowlist record drifted: {relative}")
            records.append(record)
        if not records:
            raise RuntimeError(f"CI objective pool has no train shards for {bucket}")
        files_by_sequence_length[str(bucket)] = records

    if not objective_seed_root.is_dir() or not seed_globs:
        raise RuntimeError("objective seed root and globs must be non-empty")
    seed_paths = sorted(
        {
            path.absolute()
            for pattern in seed_globs
            for path in objective_seed_root.glob(pattern)
            if path.is_file()
        }
    )
    if not seed_paths:
        raise RuntimeError("objective seed globs matched no files")
    if any(
        path.is_symlink()
        or path.resolve() != path
        or path.suffix != ".parquet"
        or objective_seed_root not in path.parents
        for path in seed_paths
    ):
        raise RuntimeError("objective seed paths must be Parquet files under their root")
    seed_records = [
        _stable_seed_record(path, root=objective_seed_root) for path in seed_paths
    ]

    source_completion = ci_metadata.get("source_completion")
    if not isinstance(source_completion, dict):
        raise RuntimeError("validated CI export has no source completion binding")
    return {
        "schema": MANIFEST_SCHEMA,
        "algorithm": SCHEDULE,
        "sequence_lengths": list(buckets),
        "ci_export": {
            "path": "export_receipt.json",
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "schema": receipt["schema"],
            "status": "complete",
            "source_completion": source_completion,
        },
        "primary_ci": {
            "files_by_sequence_length": files_by_sequence_length,
        },
        "objective_seed": {
            "files": seed_records,
        },
        "producer": dict(producer),
    }


def _producer_binding() -> dict[str, object]:
    script = Path(__file__).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout:
        raise RuntimeError("cppmega tracked worktree must be clean")
    return {
        "repository": "cppmega",
        "git_commit": commit,
        "script": script.relative_to(REPO_ROOT).as_posix(),
        "script_sha256": _sha256(script),
    }


def _parse_buckets(raw: str) -> tuple[int, ...]:
    try:
        buckets = tuple(int(value) for value in raw.split(",") if value)
    except ValueError as exc:
        raise ValueError("--buckets must be comma-separated integers") from exc
    return _validate_buckets(buckets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci-root",
        type=Path,
        required=True,
        help="Boundary-repaired private snapshot/ci directory",
    )
    parser.add_argument(
        "--ci-receipt",
        type=Path,
        required=True,
        help="Original CASE5 export_receipt.json",
    )
    parser.add_argument(
        "--repaired-manifest",
        type=Path,
        required=True,
        help="Private snapshot/repaired_manifest.json",
    )
    parser.add_argument("--objective-seed-root", type=Path, required=True)
    parser.add_argument("--seed-glob", action="append", required=True)
    parser.add_argument(
        "--buckets",
        default=",".join(str(bucket) for bucket in DEFAULT_BUCKETS),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    roots = (args.ci_root.resolve(), args.objective_seed_root.resolve())
    if any(output == root or root in output.parents for root in roots):
        raise ValueError("--output must be outside immutable source roots")
    payload = build_source_pool_manifest(
        ci_root=args.ci_root,
        ci_receipt_path=args.ci_receipt,
        repaired_manifest_path=args.repaired_manifest,
        objective_seed_root=args.objective_seed_root,
        seed_globs=tuple(args.seed_glob),
        buckets=_parse_buckets(args.buckets),
        producer=_producer_binding(),
    )
    _write_json_atomic(output, payload)
    if json.loads(output.read_bytes()) != payload:
        raise RuntimeError("published source pool manifest differs from memory")
    print(
        json.dumps(
            {
                "manifest": str(output),
                "sha256": _sha256(output),
                "primary_files": sum(
                    len(files)
                    for files in payload["primary_ci"][
                        "files_by_sequence_length"
                    ].values()
                ),
                "objective_seed_files": len(
                    payload["objective_seed"]["files"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
