#!/usr/bin/env python3
"""Freeze receipt-bound CI and objective-seed Parquet pools for materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping

import pyarrow.parquet as pq

from scripts.data.build_macro_routes_megatron_bundle import (
    CI_CONTENT_STORE_EXPORT_SCHEMA,
    DEFAULT_BUCKETS,
    PRODUCTION_CI_CONTENT_STORE_EXPORT_SCHEMA,
    REPO_ROOT,
    _load_content_store_ci_export_allowlist,
    _sha256,
    _write_json_atomic,
)

MANIFEST_SCHEMA = "cppmega_ci_objective_pool_manifest_v1"
SCHEDULE = "alternate_primary_seed_v1"


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


def build_source_pool_manifest(
    *,
    ci_root: Path,
    ci_receipt_path: Path,
    objective_seed_root: Path,
    seed_globs: tuple[str, ...],
    buckets: tuple[int, ...],
    producer: Mapping[str, object],
) -> dict[str, object]:
    if buckets != DEFAULT_BUCKETS:
        raise ValueError(f"objective source manifest requires {DEFAULT_BUCKETS}")
    ci_root = ci_root.resolve()
    ci_receipt_path = ci_receipt_path.resolve()
    objective_seed_root = objective_seed_root.resolve()
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
        ci_root=ci_root,
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
    files_by_sequence_length: dict[str, list[dict[str, object]]] = {}
    for bucket in buckets:
        names = allowed[("ci", bucket)]
        records: list[dict[str, object]] = []
        for name in sorted(names):
            relative = f"{bucket}/{name}"
            record = artifacts.get(relative)
            if (
                not isinstance(record, dict)
                or record.get("rows") != names[name]
                or not isinstance(record.get("byte_size"), int)
                or not isinstance(record.get("sha256"), str)
            ):
                raise RuntimeError(f"CI allowlist record drifted: {relative}")
            records.append(
                {
                    "path": relative,
                    "rows": int(record["rows"]),
                    "size_bytes": int(record["byte_size"]),
                    "sha256": str(record["sha256"]),
                }
            )
        files_by_sequence_length[str(bucket)] = records

    if not objective_seed_root.is_dir() or not seed_globs:
        raise RuntimeError("objective seed root and globs must be non-empty")
    seed_paths = sorted(
        {
            path.resolve()
            for pattern in seed_globs
            for path in objective_seed_root.glob(pattern)
            if path.is_file()
        }
    )
    if not seed_paths:
        raise RuntimeError("objective seed globs matched no files")
    if any(
        path.suffix != ".parquet" or objective_seed_root not in path.parents
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
    buckets = tuple(int(value) for value in raw.split(",") if value)
    if buckets != DEFAULT_BUCKETS:
        raise ValueError(f"--buckets must be exactly {DEFAULT_BUCKETS}")
    return buckets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-root", type=Path, required=True)
    parser.add_argument("--ci-receipt", type=Path, required=True)
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
