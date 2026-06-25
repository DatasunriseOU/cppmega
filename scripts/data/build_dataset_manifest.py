#!/usr/bin/env python3
"""Build a per-dataset manifest JSON for a megacpp tokenized parquet dataset.

The manifest records the immutable facts a downstream stage needs to trust a
dataset before consuming it:

  * shard list (name + byte size + rows)
  * total rows, real tokens (sum actual_token_count), padded tokens @ seq
  * total on-disk size
  * distinct repos (count + sample)
  * full schema column list (name -> arrow type)
  * tokenizer-contract + tokenizer-artifact fingerprints (sha256 of the files)

Fail-closed (RULE #1): missing dataset dir, no shards, schema drift across
shards, or a missing required token column all raise with WHERE + WHAT.

Usage:
    .venv/bin/python build_dataset_manifest.py \
        --dataset-dir /Users/dave/sources/parquet/clang_semantic_4k_v10 \
        --contract /Volumes/external/sources/cppmega.mlx/cppmega_mlx/tokenizer/tokenizer_contract_v1.json \
        --tokenizer /Volumes/external/sources/cppmega.mlx/cppmega_mlx/tokenizer/tokenizer.json \
        --out /tmp/manifest_clang_semantic_4k_v10.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


TOKEN_COLUMN = "token_ids"
ACTUAL_TOKEN_COUNT_COLUMN = "actual_token_count"


class ManifestError(RuntimeError):
    """Raised on a fail-closed manifest invariant violation."""


def _fail(where: str, what: str) -> None:
    raise ManifestError(f"WHERE={where} WHAT={what}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_shards(dataset_dir: Path) -> list[Path]:
    shards = sorted(dataset_dir.glob("*.parquet"))
    if not shards:
        _fail(str(dataset_dir), "no *.parquet shards found")
    return shards


def _schema_columns(schema) -> dict[str, str]:
    return {name: str(schema.field(name).type) for name in schema.names}


def build(
    *,
    dataset_dir: Path,
    seq_len: int,
    contract: Path | None,
    tokenizer: Path | None,
    repo_sample: int,
    batch_size: int,
) -> dict[str, object]:
    if contract is None:
        _fail("manifest fingerprints", "contract fingerprint required")
    if tokenizer is None:
        _fail("manifest fingerprints", "tokenizer fingerprint required")

    import pyarrow.parquet as pq

    shards = find_shards(dataset_dir)
    base_schema = pq.ParquetFile(shards[0]).schema_arrow
    base_columns = _schema_columns(base_schema)
    if TOKEN_COLUMN not in base_columns:
        _fail(str(dataset_dir), f"required token column {TOKEN_COLUMN!r} absent")

    has_actual = ACTUAL_TOKEN_COUNT_COLUMN in base_columns
    has_repo = "repo" in base_columns

    shard_entries: list[dict[str, object]] = []
    total_rows = 0
    real_tokens = 0
    total_size = 0
    repo_counter: Counter[str] = Counter()

    read_cols = [TOKEN_COLUMN]
    if has_actual:
        read_cols.append(ACTUAL_TOKEN_COUNT_COLUMN)
    if has_repo:
        read_cols.append("repo")

    for shard in shards:
        pf = pq.ParquetFile(shard)
        shard_columns = _schema_columns(pf.schema_arrow)
        if shard_columns != base_columns:
            added = sorted(set(shard_columns) - set(base_columns))
            removed = sorted(set(base_columns) - set(shard_columns))
            _fail(
                shard.name,
                f"schema drift vs {shards[0].name}: added={added} removed={removed}",
            )
        rows = pf.metadata.num_rows
        size = shard.stat().st_size
        shard_rows = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
            tokens = batch.column(TOKEN_COLUMN).to_pylist()
            actual = (
                batch.column(ACTUAL_TOKEN_COUNT_COLUMN).to_pylist()
                if has_actual
                else None
            )
            repos = batch.column("repo").to_pylist() if has_repo else None
            for i in range(batch.num_rows):
                if actual is not None and actual[i] is not None:
                    real_tokens += int(actual[i])
                else:
                    real_tokens += len(tokens[i]) if tokens[i] else 0
                if repos is not None:
                    rv = repos[i]
                    repo_counter[rv if isinstance(rv, str) and rv else "<empty>"] += 1
            shard_rows += batch.num_rows
        shard_entries.append(
            {"name": shard.name, "rows": rows, "bytes": size}
        )
        total_rows += shard_rows
        total_size += size

    fingerprints: dict[str, object] = {}
    if not contract.is_file():
        _fail(str(contract), "contract file not found")
    fingerprints["contract"] = {
        "path": str(contract),
        "sha256": _sha256(contract),
    }
    if not tokenizer.is_file():
        _fail(str(tokenizer), "tokenizer artifact not found")
    fingerprints["tokenizer"] = {
        "path": str(tokenizer),
        "sha256": _sha256(tokenizer),
    }

    distinct_repos = [r for r in repo_counter if r != "<empty>"]
    manifest = {
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_dir.name,
        "seq_len": seq_len,
        "shards": shard_entries,
        "shard_count": len(shard_entries),
        "rows": total_rows,
        "real_tokens": real_tokens,
        "padded_tokens_at_seq": total_rows * seq_len,
        "total_bytes": total_size,
        "distinct_repos": len(distinct_repos),
        "repo_sample": sorted(distinct_repos)[:repo_sample],
        "schema_columns": base_columns,
        "fingerprints": fingerprints,
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--contract", required=True, help="tokenizer_contract_v1.json")
    parser.add_argument("--tokenizer", required=True, help="tokenizer.json artifact")
    parser.add_argument("--repo-sample", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--out", default=None, help="Manifest path (default: print to stdout)."
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 2

    try:
        manifest = build(
            dataset_dir=dataset_dir,
            seq_len=args.seq_len,
            contract=Path(args.contract).resolve(),
            tokenizer=Path(args.tokenizer).resolve(),
            repo_sample=args.repo_sample,
            batch_size=args.batch_size,
        )
    except ManifestError as exc:
        print(f"MANIFEST BUILD FAILED: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(manifest, indent=2)
    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
