#!/usr/bin/env python3
"""Create a small valid Megatron+cppmega sidecar prefix from an existing prefix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np


_IDX_MAGIC = b"MMIDIDX\x00\x00"
_DTYPE_BY_CODE = {
    1: np.dtype("uint8"),
    4: np.dtype("int32"),
    5: np.dtype("int64"),
    8: np.dtype("uint16"),
}


def _read_mmididx(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        magic = f.read(len(_IDX_MAGIC))
        if magic != _IDX_MAGIC:
            raise ValueError(f"{path} is not an MMIDIDX file")
        version = struct.unpack("<Q", f.read(8))[0]
        if version != 1:
            raise ValueError(f"unsupported MMIDIDX version {version} in {path}")
        dtype_code = struct.unpack("<B", f.read(1))[0]
        num_sequences = struct.unpack("<Q", f.read(8))[0]
        num_documents = struct.unpack("<Q", f.read(8))[0]
        sizes = np.fromfile(f, dtype=np.int32, count=num_sequences)
        pointers = np.fromfile(f, dtype=np.int64, count=num_sequences)
        doc_idx = np.fromfile(f, dtype=np.int64, count=num_documents)
    if len(sizes) != num_sequences or len(pointers) != num_sequences or len(doc_idx) != num_documents:
        raise ValueError(f"truncated MMIDIDX arrays in {path}")
    return dtype_code, sizes, pointers, doc_idx


def _write_mmididx(path: Path, dtype_code: int, sizes: np.ndarray) -> None:
    dtype = _DTYPE_BY_CODE[dtype_code]
    pointers = np.zeros(len(sizes), dtype=np.int64)
    if len(sizes) > 1:
        offsets = np.cumsum(sizes[:-1], dtype=np.int64) * dtype.itemsize
        pointers[1:] = offsets
    with path.open("wb") as f:
        f.write(_IDX_MAGIC)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", dtype_code))
        f.write(struct.pack("<Q", len(sizes)))
        f.write(struct.pack("<Q", len(sizes) + 1))
        sizes.astype(np.int32, copy=False).tofile(f)
        pointers.tofile(f)
        np.arange(len(sizes) + 1, dtype=np.int64).tofile(f)


def _copy_token_aligned_file(src: Path, dst: Path, dtype: np.dtype, items: int) -> None:
    bytes_to_copy = int(items) * dtype.itemsize
    remaining = bytes_to_copy
    with src.open("rb") as fin, dst.open("wb") as fout:
        while remaining:
            chunk = fin.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{src} ended at {fout.tell()} bytes, needed {bytes_to_copy}")
            fout.write(chunk)
            remaining -= len(chunk)


def _copy_graph_sidecar(
    *,
    src_base: Path,
    dst_base: Path,
    src_manifest_entry: dict[str, object],
    dst_manifest_entry: dict[str, object],
    document_count: int,
) -> None:
    offset_dtype = np.dtype(str(src_manifest_entry.get("offset_dtype", "int64")))
    dtype = np.dtype(str(src_manifest_entry["dtype"]))
    shape_tail = tuple(int(x) for x in src_manifest_entry.get("shape_tail", []))

    src_offsets = np.memmap(src_base.parent / str(src_manifest_entry["offsets_path"]), mode="r", dtype=offset_dtype)
    if len(src_offsets) < document_count + 1:
        raise ValueError(
            f"{src_manifest_entry['offsets_path']} has {len(src_offsets)} offsets, "
            f"need {document_count + 1}"
        )
    new_offsets = np.asarray(src_offsets[: document_count + 1], dtype=offset_dtype).copy()
    item_count = int(new_offsets[-1])

    src_data_path = src_base.parent / str(src_manifest_entry["data_path"])
    if shape_tail:
        data = np.memmap(src_data_path, mode="r", dtype=dtype, shape=(int(src_manifest_entry["item_count"]),) + shape_tail)
        sliced = np.asarray(data[:item_count])
    else:
        data = np.memmap(src_data_path, mode="r", dtype=dtype)
        sliced = np.asarray(data[:item_count])

    offsets_path = dst_base.parent / str(dst_manifest_entry["offsets_path"])
    data_path = dst_base.parent / str(dst_manifest_entry["data_path"])
    new_offsets.tofile(offsets_path)
    sliced.astype(dtype, copy=False).tofile(data_path)
    dst_manifest_entry["item_count"] = item_count


def create_subset(src_prefix: Path, dst_prefix: Path, *, max_tokens: int, max_docs: int | None) -> None:
    src_manifest = json.loads(src_prefix.with_suffix(".json").read_text())
    dtype_code, sizes, _, _ = _read_mmididx(src_prefix.with_suffix(".idx"))
    token_dtype = _DTYPE_BY_CODE[dtype_code]

    cumulative = np.cumsum(sizes, dtype=np.int64)
    doc_count = int(np.searchsorted(cumulative, max_tokens, side="right") + 1)
    if max_docs is not None:
        doc_count = min(doc_count, max_docs)
    doc_count = min(doc_count, len(sizes))
    if doc_count <= 0:
        raise ValueError("subset would be empty")
    subset_sizes = sizes[:doc_count].copy()
    token_count = int(subset_sizes.sum())

    dst_prefix.parent.mkdir(parents=True, exist_ok=True)
    _copy_token_aligned_file(
        src_prefix.with_suffix(".bin"),
        dst_prefix.with_suffix(".bin"),
        token_dtype,
        token_count,
    )
    _write_mmididx(dst_prefix.with_suffix(".idx"), dtype_code, subset_sizes)

    dst_manifest = dict(src_manifest)
    dst_manifest["token_count"] = token_count
    dst_manifest["document_count"] = doc_count
    dst_manifest["source_prefix"] = str(src_prefix)
    dst_manifest["subset_max_tokens"] = max_tokens

    side_paths: dict[str, dict[str, object]] = {}
    for column, entry in (src_manifest.get("side_channel_paths") or {}).items():
        dtype = np.dtype(str(entry["dtype"]))
        src_name = str(entry["path"])
        dst_name = f"{dst_prefix.name}_{column}.bin"
        _copy_token_aligned_file(src_prefix.parent / src_name, dst_prefix.parent / dst_name, dtype, token_count)
        side_paths[column] = {"path": dst_name, "dtype": str(entry["dtype"])}
    dst_manifest["side_channel_paths"] = side_paths

    graph_paths: dict[str, dict[str, object]] = {}
    for column, entry in (src_manifest.get("graph_sidecar_paths") or {}).items():
        copied = dict(entry)
        copied["offsets_path"] = f"{dst_prefix.name}_{column}_offsets.bin"
        copied["data_path"] = f"{dst_prefix.name}_{column}_data.bin"
        _copy_graph_sidecar(
            src_base=src_prefix,
            dst_base=dst_prefix,
            src_manifest_entry=entry,
            dst_manifest_entry=copied,
            document_count=doc_count,
        )
        graph_paths[column] = copied
    if graph_paths:
        dst_manifest["graph_sidecar_schema"] = "cppmega_graph_routes_v1"
        dst_manifest["graph_sidecar_paths"] = graph_paths

    dst_prefix.with_suffix(".json").write_text(json.dumps(dst_manifest, indent=2), encoding="utf-8")
    print(
        f"wrote {dst_prefix}: docs={doc_count:,} tokens={token_count:,} "
        f"token_bytes={os.path.getsize(dst_prefix.with_suffix('.bin')):,}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-prefix", type=Path, required=True)
    parser.add_argument("--dst-prefix", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8_000_000)
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    create_subset(args.src_prefix, args.dst_prefix, max_tokens=args.max_tokens, max_docs=args.max_docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
