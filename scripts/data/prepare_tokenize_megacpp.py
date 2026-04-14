#!/usr/bin/env python3
"""megacpp data prep — Stage 2: Tokenize raw C/C++ source into parquet shards.

This stage runs the Clang semantic indexer over the raw corpus, emits JSONL
with per-chunk token IDs, and streams it into parquet shards of the form
``shard_NNNNN.parquet`` plus an optional ``val_shard.parquet``.

DEPENDENCY — nanochat source tree required
-------------------------------------------
The semantic indexer and the hybrid C++ BPE tokenizer live in the upstream
nanochat repo and are too heavyweight to vendor verbatim here:

  * ``nanochat/tools/clang_indexer/index_project.py`` — libclang-based
    cross-file semantic indexer (emits enriched JSONL).
  * ``nanochat/nanochat/cpp_tokenizer.py`` — hybrid fixed-vocab + BPE
    tokenizer (131072 tokens) backed by ``nanochat/tokenizer.json``.
  * ``nanochat/scripts/data/clang_enriched_to_4k_parquet.py`` — 4K-budgeted
    parquet converter; imports several ``nanochat.*`` modules.
  * ``nanochat/scripts/data/stream_jsonl_to_parquet.py`` — simple streaming
    variant (no nanochat.* deps) — a standalone copy is shipped alongside as
    ``prepare_stream_jsonl_to_parquet_megacpp.py``.

To run the full tokenize stage, clone nanochat next to cppmega:

    git clone <nanochat-url> /path/to/nanochat
    export MEGACPP_NANOCHAT_ROOT=/path/to/nanochat

Then this script dispatches to nanochat's ``run_clang_pipeline.sh`` with the
megacpp output layout. If ``MEGACPP_NANOCHAT_ROOT`` is not set, the script
exits with a clear error pointing here.

Output layout (under ``${MEGACPP_DATA_ROOT}``):

    data/
      cpp_raw/                        # stage 1 clones
      parquet/
        clang_semantic_4k_v10/
          shard_00000.parquet
          shard_00001.parquet
          ...
          val_shard.parquet
          _COMPLETE
      tokenizer/
        tokenizer.json                # copied/linked from nanochat
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA_ROOT = os.environ.get(
    "MEGACPP_DATA_ROOT", "/home/dave/cppmega-root/data"
)
DEFAULT_DATASET_NAME = "clang_semantic_4k_v10"


def _require_nanochat_root() -> Path:
    root = os.environ.get("MEGACPP_NANOCHAT_ROOT")
    if not root:
        sys.exit(
            "ERROR: MEGACPP_NANOCHAT_ROOT is not set.\n"
            "  The tokenize stage depends on the nanochat repo (clang indexer +\n"
            "  cpp_tokenizer). Clone nanochat and point this env var at it:\n"
            "    export MEGACPP_NANOCHAT_ROOT=/path/to/nanochat\n"
            "  See docs/data_preparation.md for details."
        )
    p = Path(root)
    if not p.is_dir():
        sys.exit(f"ERROR: MEGACPP_NANOCHAT_ROOT={p} does not exist")
    pipeline = p / "scripts" / "data" / "run_clang_pipeline.sh"
    if not pipeline.is_file():
        sys.exit(f"ERROR: {pipeline} not found — is MEGACPP_NANOCHAT_ROOT correct?")
    return p


def _copy_tokenizer(nanochat_root: Path, data_root: Path) -> None:
    src = nanochat_root / "tokenizer.json"
    if not src.is_file():
        sys.exit(f"ERROR: tokenizer artifact not found at {src}")
    dst_dir = data_root / "tokenizer"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "tokenizer.json"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
        print(f"[megacpp] copied tokenizer: {src} -> {dst}")
    else:
        print(f"[megacpp] tokenizer already up to date at {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"megacpp data root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help=f"output parquet dataset subdir (default: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--raw-subdir",
        default="cpp_raw",
        help="subdir of --data-root holding stage-1 raw clones",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=48,
        help="parse-workers passed to clang indexer",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="hard token budget per sample (matches the 4k dataset family)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    raw_dir = data_root / args.raw_subdir
    if not raw_dir.is_dir():
        sys.exit(
            f"ERROR: raw corpus not found at {raw_dir}.\n"
            f"  Run prepare_download_megacpp.sh first."
        )

    nanochat_root = _require_nanochat_root()
    _copy_tokenizer(nanochat_root, data_root)

    parquet_root = data_root / "parquet" / args.dataset_name
    parquet_root.mkdir(parents=True, exist_ok=True)

    pipeline = nanochat_root / "scripts" / "data" / "run_clang_pipeline.sh"
    env = os.environ.copy()
    env["TOKENIZER_PATH"] = str(data_root / "tokenizer" / "tokenizer.json")
    env["MAX_TOKENS"] = str(args.max_tokens)
    # Output layout — run_clang_pipeline.sh writes:
    #   $OUTPUT_DIR/cpp_clang_crossfile_16k.jsonl  (default 16k label)
    #   $OUTPUT_DIR/parquet/cpp_clang_crossfile_16k/
    # We honor that layout; the caller renames / symlinks to
    # ${dataset_name} afterwards.
    staging = data_root / "staging" / args.dataset_name
    staging.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash",
        str(pipeline),
        str(raw_dir),
        str(staging),
        str(args.workers),
    ]
    print("[megacpp] running nanochat clang pipeline:")
    print("   ", " ".join(cmd))
    ret = subprocess.call(cmd, env=env)
    if ret != 0:
        sys.exit(f"ERROR: nanochat clang pipeline exited with {ret}")

    # Publish: move/symlink staging/parquet/* into data/parquet/<dataset_name>/
    produced = staging / "parquet"
    if produced.is_dir():
        # Find the single sub-dataset dir
        subs = [p for p in produced.iterdir() if p.is_dir()]
        if len(subs) != 1:
            sys.exit(f"ERROR: expected 1 sub-dataset dir in {produced}, got {subs}")
        src = subs[0]
        for shard in src.iterdir():
            dst = parquet_root / shard.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(shard), str(dst))
        print(f"[megacpp] published parquet shards to {parquet_root}")
    else:
        sys.exit(f"ERROR: pipeline did not produce {produced}")

    print(f"[megacpp] stage 2 done: {parquet_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
