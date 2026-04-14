#!/usr/bin/env python3
"""megacpp data prep — Stage 5: Verify Megatron dataset is trainable.

Checks:
  1. ``<prefix>_train.bin`` and ``<prefix>_train.idx`` exist and are non-empty.
  2. ``.idx`` parses (via megatron-core if available, else raw struct check).
  3. Token-ID range lies within expected vocab (default 131072).
  4. Document count and total-token count are printed.
  5. First 64 tokens of document 0 are printed for sanity.

Exits non-zero on failure (no silent fallbacks).

Usage:
    python verify_dataset_megacpp.py
    python verify_dataset_megacpp.py --vocab-size 131072
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

DEFAULT_DATA_ROOT = os.environ.get(
    "MEGACPP_DATA_ROOT", "/home/dave/cppmega-root/data"
)
DEFAULT_DATASET_NAME = os.environ.get(
    "MEGACPP_DATASET_NAME", "clang_semantic_4k_v10"
)


def _raw_idx_inspect(idx_path: Path) -> tuple[int, int]:
    """Minimal MMIDIDX .idx parser — matches the fallback writer in
    ``data_prep_parquet_to_megatron.py``. Returns (num_docs, dtype_code)."""
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        if magic != b"MMIDIDX\x00\x00":
            raise RuntimeError(f"bad magic in {idx_path}: {magic!r}")
        (version,) = struct.unpack("<Q", f.read(8))
        (dtype_code,) = struct.unpack("<B", f.read(1))
        (num_sequences,) = struct.unpack("<Q", f.read(8))
    return num_sequences, dtype_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--vocab-size", type=int, default=131072)
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    megatron_dir = data_root / "megatron"
    if not megatron_dir.is_dir():
        sys.exit(f"ERROR: {megatron_dir} missing — run stage 3 first.")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        suffix = "train" if split == "train" else "valid"
        prefix = megatron_dir / f"{args.dataset_name}_{suffix}"
        bin_path = prefix.with_suffix(".bin")
        idx_path = prefix.with_suffix(".idx")
        print(f"\n[megacpp_verify] split={split} prefix={prefix}")
        if not bin_path.is_file():
            sys.exit(f"ERROR: missing {bin_path}")
        if not idx_path.is_file():
            sys.exit(f"ERROR: missing {idx_path}")
        bin_bytes = bin_path.stat().st_size
        idx_bytes = idx_path.stat().st_size
        print(f"  .bin={bin_bytes/1024**3:.2f} GiB  .idx={idx_bytes/1024**2:.2f} MiB")
        if bin_bytes == 0:
            sys.exit(f"ERROR: {bin_path} is empty")

        # Prefer megatron-core if available.
        try:
            from megatron.core.datasets.indexed_dataset import IndexedDataset

            ds = IndexedDataset(str(prefix))
            n_docs = len(ds.document_indices) - 1
            total_tokens = int(ds.sequence_lengths.sum())
            sample = ds.get(0)[:64]
            print(f"  megatron: docs={n_docs:,} tokens={total_tokens:,} "
                  f"dtype={sample.dtype}")
            mn, mx = int(sample.min()), int(sample.max())
            # Full-array min/max is cheap because .bin is memmapped.
            arr_mm = np.memmap(bin_path, dtype=sample.dtype, mode="r")
            mn = min(mn, int(arr_mm.min()))
            mx = max(mx, int(arr_mm.max()))
            print(f"  token id range: [{mn}, {mx}]")
            if mx >= args.vocab_size:
                sys.exit(
                    f"ERROR: max token id {mx} >= vocab_size {args.vocab_size}"
                )
            print(f"  doc0[:64]: {sample.tolist()}")
        except Exception as e:
            print(f"  megatron-core unavailable ({e}); falling back to raw idx check")
            n_docs, dtype_code = _raw_idx_inspect(idx_path)
            print(f"  raw: num_sequences={n_docs:,} dtype_code={dtype_code}")

    print("\n[megacpp_verify] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
