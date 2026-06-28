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

_SIDECAR_DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}

_MEGATRON_DTYPE_CODE_MAP = {
    np.uint8: 1,
    np.int32: 4,
    np.int64: 5,
    np.uint16: 8,
}

DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS: tuple[tuple[str, str], ...] = (
    ("token_structure_ids", "uint8"),
    ("token_dep_levels", "uint16"),
    ("token_ast_depth", "uint16"),
    ("token_sibling_index", "uint16"),
    ("token_ast_node_type", "uint16"),
    ("token_symbol_ids", "uint32"),
    ("token_call_targets", "uint32"),
    ("token_type_refs", "uint32"),
    ("token_def_use", "uint8"),
    ("token_change_mask_pre", "uint8"),
    ("token_change_mask_post", "uint8"),
    ("token_platform_ids", "uint16"),
)

DEFAULT_CPPMEGA_GRAPH_SIDECARS: tuple[tuple[str, str, str], ...] = (
    ("token_call_edges", "edge_pairs", "int32"),
    ("token_type_edges", "edge_pairs", "int32"),
    ("token_chunk_starts", "ragged_1d", "uint32"),
    ("token_chunk_ends", "ragged_1d", "uint32"),
    ("token_chunk_kinds", "ragged_1d", "uint16"),
    ("token_chunk_dep_levels", "ragged_1d", "uint16"),
)


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


def _resolve_sidecar_dtype(dtype_str: str) -> np.dtype:
    try:
        return np.dtype(_SIDECAR_DTYPE_MAP[dtype_str])
    except KeyError as exc:
        raise ValueError(f"unsupported sidecar dtype: {dtype_str}") from exc


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


def _unique_columns(*groups: list[str] | tuple[str, ...] | None) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for column in group or ():
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def _normalize_edge_pairs(
    value: object,
    *,
    column: str,
    shard_path: str,
    row_idx: int,
) -> np.ndarray:
    """Normalize a parquet edge-list cell to an ``(E, 2)`` int32 array."""

    if value is None:
        return np.zeros((0, 2), dtype=np.int32)
    if isinstance(value, np.ndarray) and value.dtype.fields:
        fields = value.dtype.fields
        if "from" not in fields or "to" not in fields:
            raise ValueError(
                f"{column} structured edge array must contain from/to fields at "
                f"{shard_path}#row{row_idx}"
            )
        pairs = np.stack([value["from"], value["to"]], axis=1)
    else:
        pairs_list: list[tuple[int, int]] = []
        for edge in value:  # type: ignore[union-attr]
            if hasattr(edge, "as_py"):
                edge = edge.as_py()
            if isinstance(edge, dict):
                src = edge.get("from")
                dst = edge.get("to")
            elif isinstance(edge, (list, tuple, np.ndarray)) and len(edge) >= 2:
                src = edge[0]
                dst = edge[1]
            else:
                raise ValueError(
                    f"{column} edge must be {{from,to}} or pair at "
                    f"{shard_path}#row{row_idx}; got {type(edge).__name__}"
                )
            if src is None or dst is None:
                raise ValueError(
                    f"{column} edge has missing endpoint at {shard_path}#row{row_idx}"
                )
            src_i = int(src)
            dst_i = int(dst)
            if src_i < 0 or dst_i < 0:
                raise ValueError(
                    f"{column} edge endpoints must be non-negative at "
                    f"{shard_path}#row{row_idx}: ({src_i}, {dst_i})"
                )
            pairs_list.append((src_i, dst_i))
        pairs = np.asarray(pairs_list, dtype=np.int64)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            f"{column} must normalize to (E, 2), got shape {tuple(pairs.shape)} "
            f"at {shard_path}#row{row_idx}"
        )
    if np.any(pairs < 0):
        raise ValueError(f"{column} contains negative edge endpoint at {shard_path}#row{row_idx}")
    if np.any(pairs > np.iinfo(np.int32).max):
        raise ValueError(f"{column} edge endpoint exceeds int32 at {shard_path}#row{row_idx}")
    return pairs.astype(np.int32, copy=False)


def _normalize_ragged_int_vector(
    value: object,
    *,
    dtype: np.dtype,
    column: str,
    shard_path: str,
    row_idx: int,
) -> np.ndarray:
    """Normalize a ragged parquet scalar to a 1-D integer array."""

    if value is None:
        return np.zeros((0,), dtype=dtype)
    arr = np.asarray(value)
    if arr.size == 0:
        return np.zeros((0,), dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(
            f"{column} must be a 1-D ragged integer vector, got shape "
            f"{tuple(arr.shape)} at {shard_path}#row{row_idx}"
        )
    if np.any(arr < 0):
        raise ValueError(f"{column} contains negative value at {shard_path}#row{row_idx}")
    info = np.iinfo(dtype)
    if np.any(arr > info.max):
        raise ValueError(
            f"{column} value exceeds {dtype.name} max {info.max} at {shard_path}#row{row_idx}"
        )
    return arr.astype(dtype, copy=False)


class _GraphSidecarWriters:
    """Write document-aligned CSR-style graph sidecars next to .bin/.idx."""

    def __init__(
        self,
        output_prefix: str,
        specs: tuple[tuple[str, str, str], ...] = DEFAULT_CPPMEGA_GRAPH_SIDECARS,
    ) -> None:
        self._output_prefix = output_prefix
        self._specs = specs
        self._data_files = {
            column: open(f"{output_prefix}_{column}_data.bin", "wb")
            for column, _, _ in specs
        }
        self._offsets = {column: [0] for column, _, _ in specs}
        self._item_counts = {column: 0 for column, _, _ in specs}
        self._closed = False

    @property
    def columns(self) -> list[str]:
        return [column for column, _, _ in self._specs]

    def append(
        self,
        values: dict[str, object],
        *,
        shard_path: str,
        row_idx: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("graph sidecar writers are already closed")
        for column, kind, dtype_str in self._specs:
            dtype = _resolve_sidecar_dtype(dtype_str)
            if kind == "edge_pairs":
                arr = _normalize_edge_pairs(
                    values.get(column),
                    column=column,
                    shard_path=shard_path,
                    row_idx=row_idx,
                )
                arr.astype(dtype, copy=False).tofile(self._data_files[column])
                item_count = int(arr.shape[0])
            elif kind == "ragged_1d":
                arr = _normalize_ragged_int_vector(
                    values.get(column),
                    dtype=dtype,
                    column=column,
                    shard_path=shard_path,
                    row_idx=row_idx,
                )
                arr.tofile(self._data_files[column])
                item_count = int(arr.shape[0])
            else:
                raise ValueError(f"unsupported graph sidecar kind {kind!r} for {column}")
            self._item_counts[column] += item_count
            self._offsets[column].append(self._item_counts[column])

    def close(self) -> dict[str, dict[str, object]]:
        if self._closed:
            raise RuntimeError("graph sidecar writers are already closed")
        for fh in self._data_files.values():
            fh.close()
        manifest: dict[str, dict[str, object]] = {}
        base = os.path.basename(self._output_prefix)
        for column, kind, dtype_str in self._specs:
            offsets_path = f"{self._output_prefix}_{column}_offsets.bin"
            np.asarray(self._offsets[column], dtype=np.int64).tofile(offsets_path)
            entry: dict[str, object] = {
                "kind": kind,
                "offsets_path": f"{base}_{column}_offsets.bin",
                "data_path": f"{base}_{column}_data.bin",
                "offset_dtype": "int64",
                "dtype": dtype_str,
                "item_count": self._item_counts[column],
            }
            if kind == "edge_pairs":
                entry["shape_tail"] = [2]
            manifest[column] = entry
        self._closed = True
        return manifest

    def abort_close(self) -> None:
        if self._closed:
            return
        for fh in self._data_files.values():
            fh.close()
        self._closed = True


def _graph_sidecar_values(
    graph_cols: dict[str, object],
    row_idx: int,
) -> dict[str, object]:
    return {column: graph_cols[column][row_idx].as_py() for column in graph_cols}  # type: ignore[index]


def _add_graph_manifest(
    sidecar_data: dict[str, object],
    graph_sidecar_paths: dict[str, dict[str, object]] | None,
) -> None:
    if not graph_sidecar_paths:
        return
    sidecar_data["graph_sidecar_schema"] = "cppmega_graph_routes_v1"
    sidecar_data["graph_sidecar_paths"] = graph_sidecar_paths


def _convert_parquet_to_numpy(
    input_dir: str,
    output_prefix: str,
    split: str,
    token_column: str,
    dtype_str: str,
    side_channels: list[str] | None = None,
    side_channel_dtypes: list[str] | None = None,
    graph_sidecars: tuple[tuple[str, str, str], ...] | None = DEFAULT_CPPMEGA_GRAPH_SIDECARS,
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

    graph_writers = _GraphSidecarWriters(output_prefix, graph_sidecars) if graph_sidecars else None
    graph_columns = graph_writers.columns if graph_writers is not None else []
    columns_to_read = _unique_columns([token_column], side_channels, graph_columns)

    bin_path = output_prefix + ".bin"
    side_writers = {
        col: open(f"{output_prefix}_{col}.bin", "wb")
        for col in (side_channels or [])
    }
    sizes: list[int] = []
    pointers: list[int] = []
    total_tokens = 0
    graph_sidecar_paths: dict[str, dict[str, object]] | None = None

    try:
        with open(bin_path, "wb") as bin_fh:
            for shard_idx, shard_path in enumerate(shards):
                pf = pq.ParquetFile(shard_path)
                for rg_idx in range(pf.metadata.num_row_groups):
                    table = pf.read_row_group(rg_idx, columns=columns_to_read)
                    token_col = table.column(token_column)
                    side_cols = {col: table.column(col) for col in (side_channels or [])}
                    graph_cols = {col: table.column(col) for col in graph_columns}
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
                        if graph_writers is not None:
                            graph_writers.append(
                                _graph_sidecar_values(graph_cols, row_idx),
                                shard_path=shard_path,
                                row_idx=row_idx,
                            )

                        total_tokens += len(arr)
                if (shard_idx + 1) % 10 == 0 or shard_idx + 1 == len(shards):
                    print(
                        f"  read {shard_idx + 1}/{len(shards)} shards, "
                        f"{len(sizes):,} docs, {total_tokens:,} tokens"
                    )
        if graph_writers is not None:
            graph_sidecar_paths = graph_writers.close()
    except Exception:
        if graph_writers is not None:
            graph_writers.abort_close()
        raise
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
        "document_count": len(sizes_arr),
    }
    if side_channels:
        side_channel_paths = {}
        for col, dt_str in zip(side_channels, side_channel_dtypes or [], strict=True):
            side_channel_paths[col] = {
                "path": f"{os.path.basename(output_prefix)}_{col}.bin",
                "dtype": dt_str,
            }
        sidecar_data["side_channel_paths"] = side_channel_paths
    _add_graph_manifest(sidecar_data, graph_sidecar_paths)

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
    graph_sidecars: tuple[tuple[str, str, str], ...] | None = DEFAULT_CPPMEGA_GRAPH_SIDECARS,
    vocab_size: int = 65536,
) -> None:
    """Convert parquet token_ids to Megatron MMapIndexedDataset format."""
    import pyarrow.parquet as pq
    import json

    if side_channels is None and side_channel_dtypes is None:
        side_channels = [name for name, _ in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]
        side_channel_dtypes = [dtype for _, dtype in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]

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
            graph_sidecars=graph_sidecars,
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

    graph_writers = _GraphSidecarWriters(output_prefix, graph_sidecars) if graph_sidecars else None
    graph_columns = graph_writers.columns if graph_writers is not None else []
    graph_sidecar_paths: dict[str, dict[str, object]] | None = None

    total_docs = 0
    total_tokens = 0
    t0 = time.time()

    columns_to_read = _unique_columns([token_column], side_channels, graph_columns)

    try:
        for shard_idx, shard_path in enumerate(shards):
            pf = pq.ParquetFile(shard_path)
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=columns_to_read)
                token_col = table.column(token_column)
                side_cols = {col: table.column(col) for col in (side_channels or [])}
                graph_cols = {col: table.column(col) for col in graph_columns}
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
                    if graph_writers is not None:
                        graph_writers.append(
                            _graph_sidecar_values(graph_cols, row_idx),
                            shard_path=shard_path,
                            row_idx=row_idx,
                        )

                    total_docs += 1
                    total_tokens += len(arr)

            elapsed = time.time() - t0
            print(
                f"  shard {shard_idx + 1}/{len(shards)}: "
                f"{total_docs:,} docs, {total_tokens:,} tokens "
                f"({elapsed:.1f}s)"
            )

        builder.finalize(output_prefix + ".idx")
        if graph_writers is not None:
            graph_sidecar_paths = graph_writers.close()
    except Exception:
        if graph_writers is not None:
            graph_writers.abort_close()
        raise
    finally:
        for writer in side_writers.values():
            writer.close()

    # Write JSON sidecar
    json_path = output_prefix + ".json"
    sidecar_data = {
        "vocab_size": vocab_size,
        "tokenizer_contract": "megacpp",
        "dtype": dtype_str,
        "token_count": total_tokens,
        "document_count": total_docs,
    }
    if side_channels:
        side_channel_paths = {}
        for col, dt_str in zip(side_channels, side_channel_dtypes or [], strict=True):
            side_channel_paths[col] = {
                "path": f"{os.path.basename(output_prefix)}_{col}.bin",
                "dtype": dt_str,
            }
        sidecar_data["side_channel_paths"] = side_channel_paths
    _add_graph_manifest(sidecar_data, graph_sidecar_paths)

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
        help=(
            "Comma-separated side-channel columns to convert. Overrides the "
            "default cppmega-full token-aligned sidecar profile."
        ),
    )
    parser.add_argument(
        "--side-channel-dtypes",
        default=None,
        help=(
            "Comma-separated side-channel dtypes. Required when "
            "--side-channels is explicitly provided."
        ),
    )
    parser.add_argument(
        "--no-side-channels",
        action="store_true",
        help=(
            "Write only token .bin/.idx. This also disables graph sidecars. "
            "It is legacy flat-LM behavior and must be explicit; cppmega-full "
            "sidecars are the default."
        ),
    )
    parser.add_argument(
        "--no-graph-sidecars",
        action="store_true",
        help=(
            "Disable document-aligned graph route sidecars while keeping any "
            "token-aligned side channels. Default writes token_call_edges, "
            "token_type_edges, and token_chunk_* CSR sidecars."
        ),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=65536,
        help="Tokenizer vocab size to write to the JSON sidecar (default: 65536)",
    )
    args = parser.parse_args()

    if args.no_side_channels and args.side_channels:
        raise ValueError("--no-side-channels cannot be combined with --side-channels")
    if args.no_side_channels and args.side_channel_dtypes:
        raise ValueError("--no-side-channels cannot be combined with --side-channel-dtypes")

    if args.no_side_channels:
        side_channels_list = []
        side_channel_dtypes_list = []
    elif args.side_channels:
        side_channels_list = [x.strip() for x in args.side_channels.split(",") if x.strip()]
        side_channel_dtypes_list = (
            [x.strip() for x in args.side_channel_dtypes.split(",") if x.strip()]
            if args.side_channel_dtypes
            else None
        )
    else:
        side_channels_list = [name for name, _ in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]
        side_channel_dtypes_list = [dtype for _, dtype in DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS]

    if side_channels_list and not side_channel_dtypes_list:
        raise ValueError("--side-channel-dtypes must be specified when --side-channels is provided")
    if side_channels_list and side_channel_dtypes_list and len(side_channels_list) != len(side_channel_dtypes_list):
        raise ValueError("Number of --side-channels must match number of --side-channel-dtypes")
    graph_sidecars = (
        None
        if args.no_side_channels or args.no_graph_sidecars
        else DEFAULT_CPPMEGA_GRAPH_SIDECARS
    )

    convert_parquet_to_megatron(
        input_dir=args.input_dir,
        output_prefix=args.output_prefix,
        split=args.split,
        token_column=args.token_column,
        dtype_str=args.dtype,
        side_channels=side_channels_list,
        side_channel_dtypes=side_channel_dtypes_list,
        graph_sidecars=graph_sidecars,
        vocab_size=args.vocab_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
