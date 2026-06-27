#!/usr/bin/env python3
"""Convert nanochat-style tokenized parquet to Megatron indexed binary format.

Reads ``token_ids`` (uint32) from parquet shards and writes ``.bin`` + ``.idx``
files that Megatron's GPTDataset / MMapIndexedDataset can consume directly.

This script must run on the H200 machine where megatron-core is installed.

Usage:
    python data_prep_parquet_to_megatron.py \
        --input-dir /home/dave/cppmega-root/data/parquet/clang_semantic_4k_v10 \
        --output-prefix /home/dave/cppmega-root/data/megatron/clang_semantic_4k_v10 \
        --split train

    python data_prep_parquet_to_megatron.py \
        --input-dir /home/dave/cppmega-root/data/parquet/clang_commits_4k_v1 \
        --output-prefix /home/dave/cppmega-root/data/megatron/clang_commits_4k_v1 \
        --split train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


_OUTPUT_DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "int32": np.int32,
    "int64": np.int64,
    # Historical CLI compatibility. Megatron's MMIDIDX dtype enum has no
    # uint32 token dtype, so positive token IDs that need more than uint16 must
    # be stored as int32.
    "uint32": np.int32,
}

_MEGATRON_DTYPE_CODE_MAP = {
    np.uint8: 1,
    np.int32: 4,
    np.int64: 5,
    np.uint16: 8,
}


def _resolve_output_dtype(dtype_str: str) -> type[np.generic]:
    try:
        dtype = _OUTPUT_DTYPE_MAP[dtype_str]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype_str}") from exc
    if dtype_str == "uint32":
        print(
            "WARNING: Megatron MMIDIDX has no uint32 dtype code; "
            "writing token IDs as int32 instead",
            file=sys.stderr,
        )
    return dtype


def _megatron_dtype_code(dtype: type[np.generic] | np.dtype) -> int:
    dtype_type = np.dtype(dtype).type
    if dtype_type not in _MEGATRON_DTYPE_CODE_MAP:
        supported = ", ".join(
            sorted(
                np.dtype(supported_dtype).name
                for supported_dtype in _MEGATRON_DTYPE_CODE_MAP
            )
        )
        raise ValueError(
            f"unsupported Megatron MMIDIDX dtype {np.dtype(dtype).name}; "
            f"supported dtypes: {supported}"
        )
    return _MEGATRON_DTYPE_CODE_MAP[dtype_type]


def find_parquet_shards(input_dir: str, split: str) -> list[str]:
    """Find all parquet shard files for a given split.

    Convention from nanochat:
    - train shards: shard_00000.parquet ... shard_NNNNN.parquet (all except last)
    - val shard: val_shard.parquet or last shard
    """
    input_path = Path(input_dir)
    all_parquets = sorted(input_path.glob("*.parquet"))
    if not all_parquets:
        raise FileNotFoundError(f"no parquet files in {input_dir}")

    val_shard = input_path / "val_shard.parquet"
    has_explicit_val = val_shard.exists()

    if split == "train":
        if has_explicit_val:
            return [str(p) for p in all_parquets if p.name != "val_shard.parquet"]
        # Last shard is val by convention
        return [str(p) for p in all_parquets[:-1]] if len(all_parquets) > 1 else [str(all_parquets[0])]
    elif split == "val":
        if has_explicit_val:
            return [str(val_shard)]
        return [str(all_parquets[-1])] if len(all_parquets) > 1 else [str(all_parquets[0])]
    elif split == "all":
        return [str(p) for p in all_parquets]
    else:
        raise ValueError(f"unknown split: {split}")


def _require_token_aligned_side_channel(
    column: str,
    side_val: list[int] | None,
    token_ids: list[int],
    *,
    shard_path: str,
    row_idx: int,
) -> list[int]:
    """Return a side-channel row only if it is exactly token-aligned."""

    if side_val is None:
        raise ValueError(
            f"side-channel {column} is null at {shard_path}#row{row_idx}; "
            f"token_ids length {len(token_ids)}"
        )
    if len(side_val) != len(token_ids):
        raise ValueError(
            f"side-channel {column} length {len(side_val)} != "
            f"token_ids length {len(token_ids)} at {shard_path}#row{row_idx}"
        )
    return side_val


def _convert_parquet_to_numpy(
    input_dir: str,
    output_prefix: str,
    split: str,
    token_column: str,
    dtype_str: str,
    side_channels: list[str] | None = None,
    side_channel_dtypes: list[str] | None = None,
    vocab_size: int = 65536,
) -> None:
    """Fallback: write Megatron-compatible .bin + .idx using raw numpy.

    Format:
    - .bin: contiguous flat array of all token IDs
    - .idx: magic(9), version(1), dtype_code(1), num_sequences(8), num_documents(8),
            then sizes[num_docs] as int32, then pointers[num_docs] as int64
    """
    import pyarrow.parquet as pq
    import struct

    dtype = _resolve_output_dtype(dtype_str)
    dtype_code = _megatron_dtype_code(dtype)

    shards = find_parquet_shards(input_dir, split)
    print(f"found {len(shards)} {split} shards")

    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    side_dtypes = {col: np.dtype(dt) for col, dt in zip(side_channels or [], side_channel_dtypes or [], strict=True)}
    t0 = time.time()

    columns_to_read = [token_column] + (side_channels or [])

    bin_path = output_prefix + ".bin"
    side_writers = {
        col: open(f"{output_prefix}_{col}.bin", "wb")
        for col in (side_channels or [])
    }
    sizes: list[int] = []
    pointers: list[int] = []
    total_tokens = 0

    try:
        with open(bin_path, "wb") as bin_fh:
            for shard_idx, shard_path in enumerate(shards):
                pf = pq.ParquetFile(shard_path)
                for rg_idx in range(pf.metadata.num_row_groups):
                    table = pf.read_row_group(rg_idx, columns=columns_to_read)
                    token_col = table.column(token_column)
                    side_cols = {col: table.column(col) for col in (side_channels or [])}
                    for row_idx in range(len(token_col)):
                        token_ids = token_col[row_idx].as_py()
                        if not token_ids:
                            continue
                        arr = np.array(token_ids, dtype=dtype)
                        pointers.append(total_tokens * dtype().itemsize)
                        sizes.append(len(arr))
                        arr.tofile(bin_fh)

                        for col in (side_channels or []):
                            side_val = _require_token_aligned_side_channel(
                                col,
                                side_cols[col][row_idx].as_py(),
                                token_ids,
                                shard_path=shard_path,
                                row_idx=row_idx,
                            )
                            np.array(side_val, dtype=side_dtypes[col]).tofile(side_writers[col])

                        total_tokens += len(arr)
                if (shard_idx + 1) % 10 == 0 or shard_idx + 1 == len(shards):
                    print(
                        f"  read {shard_idx + 1}/{len(shards)} shards, "
                        f"{len(sizes):,} docs, {total_tokens:,} tokens"
                    )
    finally:
        for writer in side_writers.values():
            writer.close()

    print(f"total: {len(sizes)} documents")

    sizes_arr = np.array(sizes, dtype=np.int32)
    pointers_arr = np.array(pointers, dtype=np.int64)

    # Write .idx (Megatron MMapIndexedDataset format)
    idx_path = output_prefix + ".idx"
    MAGIC = b"MMIDIDX\x00\x00"  # 9 bytes
    VERSION = 1
    with open(idx_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", VERSION))
        f.write(struct.pack("<B", dtype_code))
        f.write(struct.pack("<Q", len(sizes_arr)))  # num sequences
        f.write(struct.pack("<Q", len(sizes_arr) + 1))  # num documents (includes sentinel)
        sizes_arr.tofile(f)
        pointers_arr.tofile(f)
        # Document indices (each doc is one sequence)
        doc_idx = np.arange(len(sizes_arr) + 1, dtype=np.int64)
        doc_idx.tofile(f)

    # Write JSON sidecar
    json_path = output_prefix + ".json"
    sidecar_data = {
        "vocab_size": vocab_size,
        "tokenizer_contract": "megacpp",
        "dtype": dtype_str,
        "token_count": total_tokens,
    }
    if side_channels:
        side_channel_paths = {}
        for col, dt_str in zip(side_channels, side_channel_dtypes or [], strict=True):
            side_channel_paths[col] = {
                "path": f"{os.path.basename(output_prefix)}_{col}.bin",
                "dtype": dt_str,
            }
        sidecar_data["side_channel_paths"] = side_channel_paths

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(sidecar_data, jf, indent=4)

    elapsed = time.time() - t0
    bin_size = os.path.getsize(bin_path) / (1024**3)
    print(f"\n{split}: {len(sizes_arr)} docs, {total_tokens:,} tokens, {bin_size:.2f} GiB in {elapsed:.1f}s")
    print(f"output: {bin_path} + {idx_path}")
    return


def convert_parquet_to_megatron(
    input_dir: str,
    output_prefix: str,
    split: str = "train",
    token_column: str = "token_ids",
    dtype_str: str = "uint16",
    side_channels: list[str] | None = None,
    side_channel_dtypes: list[str] | None = None,
    vocab_size: int = 65536,
) -> None:
    """Convert parquet token_ids to Megatron MMapIndexedDataset format."""
    import pyarrow.parquet as pq
    import json

    # Import Megatron's dataset builder
    try:
        from megatron.core.datasets.indexed_dataset import (  # pyright: ignore[reportMissingImports]
            IndexedDatasetBuilder as MMapIndexedDatasetBuilder,
        )
    except (ImportError, Exception) as e:
        print(f"WARNING: megatron import failed ({e}), using fallback writer", file=sys.stderr)
        MMapIndexedDatasetBuilder = None

    if MMapIndexedDatasetBuilder is None:
        # Fallback: write raw numpy binary + simple index
        _convert_parquet_to_numpy(
            input_dir,
            output_prefix,
            split,
            token_column,
            dtype_str,
            side_channels=side_channels,
            side_channel_dtypes=side_channel_dtypes,
            vocab_size=vocab_size,
        )
        return

    shards = find_parquet_shards(input_dir, split)
    print(f"found {len(shards)} {split} shards in {input_dir}")

    # Determine dtype
    dtype = _resolve_output_dtype(dtype_str)

    # Create output directory
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    builder = MMapIndexedDatasetBuilder(output_prefix + ".bin", dtype=dtype)

    # Open side channel writers
    side_writers = {}
    side_dtypes = {}
    if side_channels:
        for col, dt_str in zip(side_channels, side_channel_dtypes or [], strict=True):
            side_bin_path = f"{output_prefix}_{col}.bin"
            side_writers[col] = open(side_bin_path, "wb")
            side_dtypes[col] = np.dtype(dt_str)

    total_docs = 0
    total_tokens = 0
    t0 = time.time()

    columns_to_read = [token_column] + (side_channels or [])

    for shard_idx, shard_path in enumerate(shards):
        pf = pq.ParquetFile(shard_path)
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=columns_to_read)
            token_col = table.column(token_column)
            side_cols = {col: table.column(col) for col in (side_channels or [])}
            for row_idx in range(len(token_col)):
                token_ids = token_col[row_idx].as_py()
                if not token_ids:
                    continue
                arr = np.array(token_ids, dtype=dtype)
                builder.add_document(arr, [len(arr)])

                # Write aligned side channel values
                for col in (side_channels or []):
                    side_val = _require_token_aligned_side_channel(
                        col,
                        side_cols[col][row_idx].as_py(),
                        token_ids,
                        shard_path=shard_path,
                        row_idx=row_idx,
                    )
                    arr_side = np.array(side_val, dtype=side_dtypes[col])
                    arr_side.tofile(side_writers[col])

                total_docs += 1
                total_tokens += len(arr)

        elapsed = time.time() - t0
        print(
            f"  shard {shard_idx + 1}/{len(shards)}: "
            f"{total_docs:,} docs, {total_tokens:,} tokens "
            f"({elapsed:.1f}s)"
        )

    builder.finalize(output_prefix + ".idx")

    # Close side channel writers
    for writer in side_writers.values():
        writer.close()

    # Write JSON sidecar
    json_path = output_prefix + ".json"
    sidecar_data = {
        "vocab_size": vocab_size,
        "tokenizer_contract": "megacpp",
        "dtype": dtype_str,
        "token_count": total_tokens,
    }
    if side_channels:
        side_channel_paths = {}
        for col, dt_str in zip(side_channels, side_channel_dtypes or [], strict=True):
            side_channel_paths[col] = {
                "path": f"{os.path.basename(output_prefix)}_{col}.bin",
                "dtype": dt_str,
            }
        sidecar_data["side_channel_paths"] = side_channel_paths

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(sidecar_data, jf, indent=4)

    elapsed = time.time() - t0
    print(
        f"\n{split} conversion complete: "
        f"{total_docs:,} documents, {total_tokens:,} tokens "
        f"in {elapsed:.1f}s"
    )
    print(f"output: {output_prefix}.bin + {output_prefix}.idx")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert nanochat tokenized parquet to Megatron indexed binary"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing parquet shards",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix for .bin and .idx files",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="train",
        help="Which split to convert (default: train)",
    )
    parser.add_argument(
        "--token-column",
        default="token_ids",
        help="Parquet column containing token IDs (default: token_ids)",
    )
    parser.add_argument(
        "--dtype",
        choices=["uint8", "uint16", "int32", "int64", "uint32"],
        default="uint16",
        help=(
            "Output dtype for token IDs. uint32 is accepted as a deprecated "
            "alias that writes int32, because Megatron MMIDIDX has no uint32 "
            "dtype code."
        ),
    )
    parser.add_argument(
        "--side-channels",
        default=None,
        help="Comma-separated list of side-channel columns to convert (e.g. structure_ids,dep_levels)",
    )
    parser.add_argument(
        "--side-channel-dtypes",
        default=None,
        help="Comma-separated list of dtypes for side-channels (e.g. uint16,uint8)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=65536,
        help="Tokenizer vocab size to write to the JSON sidecar (default: 65536)",
    )
    args = parser.parse_args()

    side_channels_list = [x.strip() for x in args.side_channels.split(",") if x.strip()] if args.side_channels else None
    side_channel_dtypes_list = [x.strip() for x in args.side_channel_dtypes.split(",") if x.strip()] if args.side_channel_dtypes else None

    if side_channels_list and not side_channel_dtypes_list:
        raise ValueError("--side-channel-dtypes must be specified when --side-channels is provided")
    if side_channels_list and side_channel_dtypes_list and len(side_channels_list) != len(side_channel_dtypes_list):
        raise ValueError("Number of --side-channels must match number of --side-channel-dtypes")

    convert_parquet_to_megatron(
        input_dir=args.input_dir,
        output_prefix=args.output_prefix,
        split=args.split,
        token_column=args.token_column,
        dtype_str=args.dtype,
        side_channels=side_channels_list,
        side_channel_dtypes=side_channel_dtypes_list,
        vocab_size=args.vocab_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
