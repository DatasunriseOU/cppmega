#!/usr/bin/env python3
"""megacpp data prep — Stage 3: Convert tokenized parquet to Megatron ``.bin``/``.idx``.

Thin wrapper over ``scripts/data_prep_parquet_to_megatron.py`` that applies
megacpp naming/path conventions:

  input  = ${MEGACPP_DATA_ROOT}/parquet/${MEGACPP_DATASET_NAME}
  output = ${MEGACPP_DATA_ROOT}/megatron/${MEGACPP_DATASET_NAME}_train.{bin,idx}
           ${MEGACPP_DATA_ROOT}/megatron/${MEGACPP_DATASET_NAME}_valid.{bin,idx}

Writes both train and val splits. Defaults match the production
``clang_semantic_4k_v10`` dataset consumed by
``scripts/remote_smoke_h200_dsa_9_4_m.sh``.

Usage:
    python prepare_format_megacpp.py             # all defaults
    python prepare_format_megacpp.py --dataset-name clang_semantic_4k_v10
    python prepare_format_megacpp.py --data-root /mnt/data/cppmega-root/data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Reuse the existing parquet→Megatron converter shipped in scripts/.
_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from data_prep_parquet_to_megatron import convert_parquet_to_megatron  # noqa: E402


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
    parser.add_argument(
        "--dtype",
        choices=["uint16", "uint32", "int32"],
        default="uint32",
        help=(
            "Token-ID dtype. Use uint32 for the megacpp tokenizer (vocab=131072 > "
            "65535). uint16 only valid for vocab < 65536."
        ),
    )
    parser.add_argument(
        "--token-column",
        default="token_ids",
        help="Parquet column containing token IDs",
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated splits to emit (default: train,val)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    input_dir = data_root / "parquet" / args.dataset_name
    if not input_dir.is_dir():
        sys.exit(
            f"ERROR: input parquet dir not found: {input_dir}\n"
            "  Run prepare_tokenize_megacpp.py first (stage 2)."
        )
    output_root = data_root / "megatron"
    output_root.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        if split not in ("train", "val"):
            sys.exit(f"ERROR: unknown split '{split}'")
        # Megatron --data-path convention: <dataset_name>_train / _valid
        suffix = "train" if split == "train" else "valid"
        output_prefix = output_root / f"{args.dataset_name}_{suffix}"
        print(
            f"[megacpp_format] split={split} "
            f"input={input_dir} output_prefix={output_prefix}"
        )
        convert_parquet_to_megatron(
            input_dir=str(input_dir),
            output_prefix=str(output_prefix),
            split=split,
            token_column=args.token_column,
            dtype_str=args.dtype,
        )

    print(f"[megacpp_format] done. Megatron dataset at {output_root}")
    print(
        f"  point training at: "
        f"--data-path 1.0 {output_root}/{args.dataset_name}_train"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
