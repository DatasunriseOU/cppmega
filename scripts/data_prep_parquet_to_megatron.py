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
import os
import sys
import time
from pathlib import Path

import numpy as np


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
    else:
        raise ValueError(f"unknown split: {split}")


def _convert_parquet_to_numpy(
    input_dir: str,
    output_prefix: str,
    split: str,
    token_column: str,
    dtype_str: str,
) -> None:
    """Fallback: write Megatron-compatible .bin + .idx using raw numpy.

    Format:
    - .bin: contiguous flat array of all token IDs
    - .idx: magic(9), version(1), dtype_code(1), num_sequences(8), num_documents(8),
            then sizes[num_docs] as int32, then pointers[num_docs] as int64
    """
    import pyarrow.parquet as pq
    import struct

    dtype_map = {"uint16": np.uint16, "uint32": np.uint32, "int32": np.int32}
    dtype_code_map = {np.uint16: 1, np.uint32: 4, np.int32: 5}  # Megatron codes
    dtype = dtype_map[dtype_str]

    shards = find_parquet_shards(input_dir, split)
    print(f"found {len(shards)} {split} shards")

    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Collect all documents
    all_docs: list[np.ndarray] = []
    t0 = time.time()

    for shard_idx, shard_path in enumerate(shards):
        pf = pq.ParquetFile(shard_path)
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=[token_column])
            column = table.column(token_column)
            for row_idx in range(len(column)):
                token_ids = column[row_idx].as_py()
                if token_ids:
                    all_docs.append(np.array(token_ids, dtype=dtype))
        if (shard_idx + 1) % 10 == 0:
            print(f"  read {shard_idx + 1}/{len(shards)} shards, {len(all_docs)} docs")

    print(f"total: {len(all_docs)} documents")

    # Write .bin
    bin_path = output_prefix + ".bin"
    total_tokens = sum(len(d) for d in all_docs)
    flat = np.empty(total_tokens, dtype=dtype)
    offset = 0
    sizes = np.empty(len(all_docs), dtype=np.int32)
    pointers = np.empty(len(all_docs), dtype=np.int64)

    for i, doc in enumerate(all_docs):
        n = len(doc)
        flat[offset:offset + n] = doc
        sizes[i] = n
        pointers[i] = offset * dtype().itemsize
        offset += n

    flat.tofile(bin_path)

    # Write .idx (Megatron MMapIndexedDataset format)
    idx_path = output_prefix + ".idx"
    MAGIC = b"MMIDIDX\x00\x00"  # 9 bytes
    VERSION = 1
    with open(idx_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", VERSION))
        f.write(struct.pack("<B", dtype_code_map.get(dtype, 1)))
        f.write(struct.pack("<Q", len(all_docs)))  # num sequences
        f.write(struct.pack("<Q", len(all_docs) + 1))  # num documents (includes sentinel)
        sizes.tofile(f)
        pointers.tofile(f)
        # Document indices (each doc is one sequence)
        doc_idx = np.arange(len(all_docs) + 1, dtype=np.int64)
        doc_idx.tofile(f)

    elapsed = time.time() - t0
    bin_size = os.path.getsize(bin_path) / (1024**3)
    print(f"\n{split}: {len(all_docs)} docs, {total_tokens:,} tokens, {bin_size:.2f} GiB in {elapsed:.1f}s")
    print(f"output: {bin_path} + {idx_path}")


def convert_parquet_to_megatron(
    input_dir: str,
    output_prefix: str,
    split: str = "train",
    token_column: str = "token_ids",
    dtype_str: str = "uint16",
) -> None:
    """Convert parquet token_ids to Megatron MMapIndexedDataset format."""
    import pyarrow.parquet as pq

    # Import Megatron's dataset builder
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder as MMapIndexedDatasetBuilder
    except (ImportError, Exception) as e:
        print(f"WARNING: megatron import failed ({e}), using fallback writer", file=sys.stderr)
        MMapIndexedDatasetBuilder = None

    if MMapIndexedDatasetBuilder is None:
        # Fallback: write raw numpy binary + simple index
        _convert_parquet_to_numpy(input_dir, output_prefix, split, token_column, dtype_str)
        return

    shards = find_parquet_shards(input_dir, split)
    print(f"found {len(shards)} {split} shards in {input_dir}")

    # Determine dtype
    dtype_map = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
    }
    dtype = dtype_map.get(dtype_str)
    if dtype is None:
        raise ValueError(f"unsupported dtype: {dtype_str}")

    # Create output directory
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    builder = MMapIndexedDatasetBuilder(output_prefix + ".bin", dtype=dtype)

    total_docs = 0
    total_tokens = 0
    t0 = time.time()

    for shard_idx, shard_path in enumerate(shards):
        pf = pq.ParquetFile(shard_path)
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=[token_column])
            column = table.column(token_column)
            for row_idx in range(len(column)):
                token_ids = column[row_idx].as_py()
                if not token_ids:
                    continue
                arr = np.array(token_ids, dtype=dtype)
                builder.add_document(arr)
                total_docs += 1
                total_tokens += len(arr)

        elapsed = time.time() - t0
        print(
            f"  shard {shard_idx + 1}/{len(shards)}: "
            f"{total_docs:,} docs, {total_tokens:,} tokens "
            f"({elapsed:.1f}s)"
        )

    builder.finalize(output_prefix + ".idx")

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
        choices=["train", "val"],
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
        choices=["uint16", "uint32", "int32"],
        default="uint16",
        help="Output dtype for token IDs (default: uint16, sufficient for vocab < 65536)",
    )
    args = parser.parse_args()

    convert_parquet_to_megatron(
        input_dir=args.input_dir,
        output_prefix=args.output_prefix,
        split=args.split,
        token_column=args.token_column,
        dtype_str=args.dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
