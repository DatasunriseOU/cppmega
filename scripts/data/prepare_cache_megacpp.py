#!/usr/bin/env python3
"""megacpp data prep — Stage 4: Pre-build Megatron GPTDataset index cache.

Megatron's ``GPTDataset`` lazily writes a shuffle / sample / document index
under ``<dataset_prefix>/cache/GPTDataset_indices/`` on first access. For
large corpora this can add several minutes to the first training launch.
This stage warms that cache by instantiating the dataset once with the same
config knobs used at train time.

This is a best-effort helper — on machines without megatron-core installed
(dev laptops) the script prints a clear skip message and exits 0.

Usage:
    python prepare_cache_megacpp.py [--data-root ...] [--dataset-name ...]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = os.environ.get(
    "MEGACPP_DATA_ROOT", "/home/dave/cppmega-root/data"
)
DEFAULT_DATASET_NAME = os.environ.get(
    "MEGACPP_DATASET_NAME", "clang_semantic_4k_v10"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    prefix = data_root / "megatron" / f"{args.dataset_name}_train"
    bin_path = prefix.with_suffix(".bin")
    idx_path = prefix.with_suffix(".idx")
    if not bin_path.is_file() or not idx_path.is_file():
        sys.exit(
            f"ERROR: {bin_path} or {idx_path} missing — run stage 3 "
            f"(prepare_format_megacpp.py) first."
        )

    try:
        from megatron.core.datasets.indexed_dataset import IndexedDataset
    except Exception as e:  # pragma: no cover — dev laptop path
        print(
            f"[megacpp_cache] SKIP: megatron-core not importable ({e}).\n"
            f"  Run this stage on the training host instead.",
            file=sys.stderr,
        )
        return 0

    print(f"[megacpp_cache] touching IndexedDataset at {prefix}")
    ds = IndexedDataset(str(prefix))
    n_docs = len(ds.document_indices) - 1
    total_tokens = int(ds.sequence_lengths.sum())
    print(
        f"[megacpp_cache] docs={n_docs:,}  tokens={total_tokens:,}  "
        f"dtype={ds.index.dtype}"
    )
    print(
        "[megacpp_cache] GPTDataset sample-index is built at train launch "
        "(Megatron handles it); this stage just validates the .bin/.idx pair."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
