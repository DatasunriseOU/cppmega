#!/usr/bin/env python3
"""Verify token-aligned side-channel columns are length-aligned with token_ids.

For every token-aligned side-channel column present in the dataset, this asserts
``len(side_channel_row) == len(token_ids_row)`` for a sample of N rows per shard.
It is fail-closed (RULE #1): the FIRST mismatch raises with WHERE (shard + row +
column) and WHAT (the two lengths). There is no silent truncation/padding here —
that auto-alignment lives (wrongly) in the converter; this guard exists to catch
it before it can paper over a real bug.

Usage:
    .venv/bin/python verify_side_channel_shapes.py \
        --dataset-dir /Users/dave/sources/parquet/clang_semantic_4k_v10 \
        --sample 256

    .venv/bin/python verify_side_channel_shapes.py \
        --dataset-dir /Users/dave/sources/parquet/clang_commits_4k_v1 \
        --channels token_structure_ids,token_dep_levels --sample 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq


TOKEN_COLUMN = "token_ids"

# Token-aligned side channels in the modern v12 schema (one value per token).
# NOTE: row-level columns such as ``platform_ids`` (the deduplicated row
# signature) are deliberately excluded — only genuinely per-token columns belong
# here. The token-level platform column is ``token_platform_ids``.
DEFAULT_TOKEN_ALIGNED_CHANNELS = (
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
    "token_def_use",
    "token_symbol_ids",
    "token_call_targets",
    "token_type_refs",
    "token_change_mask_pre",
    "token_change_mask_post",
    "token_platform_ids",
)


class ShapeError(RuntimeError):
    """Raised when a side-channel row length != token_ids length."""


def _fail(where: str, what: str) -> None:
    raise ShapeError(f"WHERE={where} WHAT={what}")


def find_shards(dataset_dir: Path) -> list[Path]:
    shards = sorted(dataset_dir.glob("*.parquet"))
    if not shards:
        _fail(str(dataset_dir), "no *.parquet shards found")
    return shards


def verify(
    *,
    dataset_dir: Path,
    channels: tuple[str, ...] | None,
    sample: int,
) -> dict[str, object]:
    shards = find_shards(dataset_dir)
    present = set(pq.ParquetFile(shards[0]).schema_arrow.names)
    if TOKEN_COLUMN not in present:
        _fail(str(dataset_dir), f"required token column {TOKEN_COLUMN!r} absent")

    if channels is None:
        check_channels = tuple(c for c in DEFAULT_TOKEN_ALIGNED_CHANNELS if c in present)
    else:
        check_channels = channels
        missing = [c for c in check_channels if c not in present]
        if missing:
            _fail(
                str(dataset_dir),
                f"requested side-channels absent from schema: {', '.join(missing)}",
            )

    if not check_channels:
        _fail(
            str(dataset_dir),
            "no token-aligned side-channel columns present to verify",
        )

    read_cols = [TOKEN_COLUMN, *check_channels]
    checked_rows = 0
    per_channel_checked = {c: 0 for c in check_channels}

    for shard in shards:
        pf = pq.ParquetFile(shard)
        remaining = sample
        for batch in pf.iter_batches(batch_size=min(sample, 1024), columns=read_cols):
            if remaining <= 0:
                break
            tokens = batch.column(TOKEN_COLUMN).to_pylist()
            channel_cols = {c: batch.column(c).to_pylist() for c in check_channels}
            n = min(remaining, batch.num_rows)
            for row in range(n):
                tok = tokens[row]
                if tok is None:
                    _fail(f"{shard.name}#row{row}", f"{TOKEN_COLUMN} is null")
                tlen = len(tok)
                for c in check_channels:
                    val = channel_cols[c][row]
                    if val is None:
                        _fail(
                            f"{shard.name}#row{row}#col{c}",
                            f"side-channel is null while token_ids has length {tlen}",
                        )
                    if len(val) != tlen:
                        _fail(
                            f"{shard.name}#row{row}#col{c}",
                            f"len(side_channel)={len(val)} != len(token_ids)={tlen}",
                        )
                    per_channel_checked[c] += 1
                checked_rows += 1
            remaining -= n

    return {
        "dataset_dir": str(dataset_dir),
        "shards": len(shards),
        "channels_verified": list(check_channels),
        "rows_checked": checked_rows,
        "per_channel_checked": per_channel_checked,
        "status": "OK",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channels to check (default: all present token-aligned).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=256,
        help="Rows to sample per shard (default 256).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 2

    channels = (
        tuple(c.strip() for c in args.channels.split(",") if c.strip())
        if args.channels
        else None
    )
    try:
        result = verify(dataset_dir=dataset_dir, channels=channels, sample=args.sample)
    except ShapeError as exc:
        print(f"SIDE-CHANNEL SHAPE CHECK FAILED: {exc}", file=sys.stderr)
        return 1

    import json

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
