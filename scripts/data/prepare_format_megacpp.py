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

import pyarrow.parquet as pq


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

# Default token-aligned side channels (column -> Megatron .bin dtype) carried
# alongside the token stream. These match the modern v12 parquet schema. Any
# requested channel that is absent from the parquet schema is a hard error
# unless --allow-missing-side-channels is passed (RULE #1: fail loud).
DEFAULT_SIDE_CHANNELS: tuple[tuple[str, str], ...] = (
    ("token_structure_ids", "uint8"),
    ("token_dep_levels", "uint16"),
    ("token_ast_depth", "uint16"),
    ("token_sibling_index", "uint16"),
    ("token_ast_node_type", "uint16"),
    ("token_def_use", "uint8"),
)


def _parse_side_channels(spec: str) -> list[tuple[str, str]]:
    """Parse a 'col:dtype,col:dtype' spec into ordered (column, dtype) pairs."""

    pairs: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"--side-channels entry {item!r} must be 'column:dtype'"
            )
        column, dtype = item.split(":", 1)
        column, dtype = column.strip(), dtype.strip()
        if not column or not dtype:
            raise ValueError(
                f"--side-channels entry {item!r} must be 'column:dtype'"
            )
        pairs.append((column, dtype))
    return pairs


def _resolve_side_channels(
    input_dir: Path,
    requested: list[tuple[str, str]],
    *,
    allow_missing: bool,
) -> list[tuple[str, str]]:
    """Filter requested side channels against the actual parquet schema.

    Raises when a requested channel is absent unless allow_missing is set, in
    which case the absent channel is dropped (and reported on stderr).
    """

    shards = sorted(input_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards in {input_dir}")
    present = set(pq.ParquetFile(shards[0]).schema_arrow.names)

    resolved: list[tuple[str, str]] = []
    missing: list[str] = []
    for column, dtype in requested:
        if column in present:
            resolved.append((column, dtype))
        else:
            missing.append(column)

    if missing and not allow_missing:
        raise ValueError(
            "requested side-channel(s) absent from parquet schema "
            f"{input_dir.name}: {', '.join(missing)} "
            "(pass --allow-missing-side-channels to drop them)"
        )
    if missing:
        print(
            f"[megacpp_format] WARNING dropping absent side-channels: "
            f"{', '.join(missing)}",
            file=sys.stderr,
        )
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--dtype",
        choices=["uint8", "uint16", "int32", "int64", "uint32"],
        default="int32",
        help=(
            "Token-ID dtype. int32 is the safe default for any vocab. The canonical "
            "tokenizer is 65536 (ids fit uint16); uint32 is accepted as a deprecated "
            "alias that writes int32 because Megatron MMIDIDX has no uint32 dtype code."
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
    default_side_channels = ",".join(
        f"{col}:{dtype}" for col, dtype in DEFAULT_SIDE_CHANNELS
    )
    parser.add_argument(
        "--side-channels",
        default=default_side_channels,
        help=(
            "Comma-separated token-aligned side channels as 'column:dtype'. "
            f"Default: {default_side_channels}. Empty string disables side channels."
        ),
    )
    parser.add_argument(
        "--allow-missing-side-channels",
        action="store_true",
        help=(
            "Drop requested side channels that are absent from the parquet "
            "schema instead of raising (default: raise / fail loud)."
        ),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=65536,
        help=(
            "Tokenizer vocab size written to the JSON sidecar. Default 65536 = the "
            "canonical fixed mlx tokenizer (tokenizer_contract_v1.json). Existing "
            "parquet token_ids fit this (max observed 65529)."
        ),
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

    requested_side_channels = _parse_side_channels(args.side_channels)
    resolved_side_channels = (
        _resolve_side_channels(
            input_dir,
            requested_side_channels,
            allow_missing=args.allow_missing_side_channels,
        )
        if requested_side_channels
        else []
    )
    side_channels = [col for col, _ in resolved_side_channels] or None
    side_channel_dtypes = [dtype for _, dtype in resolved_side_channels] or None
    if side_channels:
        print(
            "[megacpp_format] side_channels="
            + ", ".join(f"{c}:{d}" for c, d in resolved_side_channels)
        )

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
            side_channels=side_channels,
            side_channel_dtypes=side_channel_dtypes,
            vocab_size=args.vocab_size,
        )

    print(f"[megacpp_format] done. Megatron dataset at {output_root}")
    print(
        f"  point training at: "
        f"--data-path 1.0 {output_root}/{args.dataset_name}_train"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
