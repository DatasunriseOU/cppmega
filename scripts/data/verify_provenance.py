#!/usr/bin/env python3
"""Verify provenance columns are populated for a megacpp dataset.

For the configured dataset kind, assert the required provenance columns are
present in the schema AND non-empty for every scanned row. Fail-closed
(RULE #1): the first empty value raises with WHERE (shard + row + column) and
WHAT.

Required provenance per kind:
  * static_code : repo, filepath (filepath only enforced when in schema)
  * commits     : repo, commit_hash/commit (whichever the schema carries)
  * other       : repo

This catches the v10 defect where ``repo``/``commit``/``timestamp`` were stamped
as empty strings.

Usage:
    .venv/bin/python verify_provenance.py \
        --dataset-dir /Users/dave/sources/parquet/clang_commits_4k_v1 --kind commits

    .venv/bin/python verify_provenance.py \
        --dataset-dir /Users/dave/sources/parquet/clang_semantic_4k_v10 \
        --kind static_code
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


# Per-kind required provenance. Tuples-of-aliases mean "at least one must exist
# in the schema and be non-empty"; bare strings are hard-required when present.
REQUIRED_PROVENANCE = {
    "static_code": (("repo",), ("filepath",)),
    "commits": (("repo",), ("commit_hash", "commit")),
    "other": (("repo",),),
}
# filepath is enforced only when present in the schema for static_code (some
# packed/aggregated rows carry constituent_provenance instead of a top filepath).
SCHEMA_OPTIONAL_WHEN_ABSENT = {"filepath"}


class ProvenanceError(RuntimeError):
    """Raised when a required provenance value is missing/empty."""


def _fail(where: str, what: str) -> None:
    raise ProvenanceError(f"WHERE={where} WHAT={what}")


def _empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def find_shards(dataset_dir: Path) -> list[Path]:
    shards = sorted(dataset_dir.glob("*.parquet"))
    if not shards:
        _fail(str(dataset_dir), "no *.parquet shards found")
    return shards


def _resolve_required_columns(
    dataset_dir: Path,
    present: set[str],
    kind: str,
) -> list[str]:
    resolved: list[str] = []
    for alias_group in REQUIRED_PROVENANCE[kind]:
        chosen = next((a for a in alias_group if a in present), None)
        if chosen is None:
            # Allow schema-optional columns (e.g. filepath) to be skipped.
            if all(a in SCHEMA_OPTIONAL_WHEN_ABSENT for a in alias_group):
                continue
            _fail(
                str(dataset_dir),
                f"kind={kind} requires one of {alias_group} but none present in schema",
            )
        resolved.append(chosen)
    if not resolved:
        _fail(str(dataset_dir), f"kind={kind} resolved no provenance columns to verify")
    return resolved


def verify(*, dataset_dir: Path, kind: str, sample: int) -> dict[str, object]:
    shards = find_shards(dataset_dir)
    present = set(pq.ParquetFile(shards[0]).schema_arrow.names)
    required = _resolve_required_columns(dataset_dir, present, kind)

    checked_rows = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        remaining = sample if sample > 0 else None
        for batch in pf.iter_batches(
            batch_size=1024 if remaining is None else min(remaining, 1024),
            columns=required,
        ):
            if remaining is not None and remaining <= 0:
                break
            cols = {c: batch.column(c).to_pylist() for c in required}
            n = batch.num_rows if remaining is None else min(remaining, batch.num_rows)
            for row in range(n):
                for c in required:
                    if _empty(cols[c][row]):
                        _fail(
                            f"{shard.name}#row{checked_rows + row}#col{c}",
                            f"required provenance column {c!r} is empty",
                        )
                checked_rows += 1
            if remaining is not None:
                remaining -= n

    return {
        "dataset_dir": str(dataset_dir),
        "kind": kind,
        "shards": len(shards),
        "provenance_columns_verified": required,
        "rows_checked": checked_rows,
        "status": "OK",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--kind", required=True, choices=["static_code", "commits", "other"]
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Rows to sample per shard; 0 = all rows (default).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 2

    try:
        result = verify(dataset_dir=dataset_dir, kind=args.kind, sample=args.sample)
    except ProvenanceError as exc:
        print(f"PROVENANCE CHECK FAILED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
