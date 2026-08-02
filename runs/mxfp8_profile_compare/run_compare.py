#!/usr/bin/env python3
"""Subprocess-isolated GB10 MXFP8 profile comparison driver.

Runs the existing batch-4 and batch-16 matrices in separate OS processes so no CUDA
context, cuBLAS heuristic state, or TileLang JIT cache is shared across
configs. Back-to-back configs in one process tree previously crashed with
`cuBLAS Error: an internal operation failed` (see RESULTS.md "Methodology
note").

Fail-fast: the first config that exits non-zero stops the sweep and this
script exits with the same code.

Usage:
    python runs/mxfp8_profile_compare/run_compare.py
    python runs/mxfp8_profile_compare/run_compare.py --runner /path/to/fake_runner.sh  # tests
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = Path(__file__).resolve().parent

# Same configs as the original run_compare.sh (batch 4) and run_batch16.sh
# (batch 16); process isolation must not silently shrink the comparison.
CONFIGS: list[tuple[str, list[str]]] = [
    ("bf16", ["--train-iters", "20", "--fp8-recipe", "off"]),
    (
        "mxfp8_gemm_ready",
        [
            "--train-iters",
            "20",
            "--fp8-recipe",
            "mxfp8",
            "--mxfp8-linear-kernel-contract",
            "gemm_ready_v1",
        ],
    ),
    (
        "mxfp8_legacy",
        [
            "--train-iters",
            "20",
            "--fp8-recipe",
            "mxfp8",
            "--mxfp8-linear-kernel-contract",
            "legacy",
        ],
    ),
    (
        "bf16_b16",
        [
            "--train-iters",
            "20",
            "--micro-batch-size",
            "16",
            "--global-batch-size",
            "16",
            "--fp8-recipe",
            "off",
        ],
    ),
    (
        "mxfp8_gemm_ready_b16",
        [
            "--train-iters",
            "20",
            "--micro-batch-size",
            "16",
            "--global-batch-size",
            "16",
            "--fp8-recipe",
            "mxfp8",
            "--mxfp8-linear-kernel-contract",
            "gemm_ready_v1",
        ],
    ),
]


def run_config(
    name: str,
    args: list[str],
    runner: Path,
    out_dir: Path,
    base_env: dict[str, str] | None = None,
) -> int:
    """Run one config as a fresh subprocess; return its exit code."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"profile_{name}_{ts}"
    log = out_dir / f"{run_id}.log"
    env = dict(os.environ if base_env is None else base_env)
    env.update(
        {
            "ROOT": str(_REPO_ROOT),
            "RUN_ID": run_id,
            "LOG": str(log),
            "NVSMI_LOG": str(out_dir / f"{run_id}.nvsmi.log"),
        }
    )
    print(f"=== running {name} -> {log}", flush=True)
    proc = subprocess.run([str(runner), *args], env=env, check=False)
    print(f"  done {name} rc={proc.returncode}", flush=True)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner",
        type=Path,
        default=_REPO_ROOT / "scripts" / "local_gb10_quarter_train.sh",
        help="per-config training launcher (executed once per config in a fresh process)",
    )
    parser.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    parser.add_argument(
        "--suite",
        choices=("all", "b4", "b16"),
        default="all",
        help="config matrix slice: b4 = batch-4 pair, b16 = batch-16 pair, all = both",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="comma-separated subset of config names (overrides --suite)",
    )
    args = parser.parse_args(argv)

    if args.configs:
        wanted = {item.strip() for item in args.configs.split(",") if item.strip()}
        selected = [cfg for cfg in CONFIGS if cfg[0] in wanted]
        unknown = wanted - {name for name, _ in CONFIGS}
        if unknown:
            print(f"FATAL: unknown configs {sorted(unknown)}", file=sys.stderr)
            return 2
    elif args.suite == "b4":
        selected = [cfg for cfg in CONFIGS if not cfg[0].endswith("_b16")]
    elif args.suite == "b16":
        selected = [cfg for cfg in CONFIGS if cfg[0].endswith("_b16")]
    else:
        selected = CONFIGS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg_args in selected:
        rc = run_config(name, cfg_args, args.runner, args.out_dir)
        if rc != 0:
            remaining = [n for n, _ in selected][[n for n, _ in selected].index(name) + 1 :]
            print(
                f"FATAL: {name} exited {rc}; fail-fast stop, "
                f"skipped remaining configs: {remaining}",
                file=sys.stderr,
            )
            return rc
    print("=== all configs completed (each in its own process)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
