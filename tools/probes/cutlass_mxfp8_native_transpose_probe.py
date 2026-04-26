#!/usr/bin/env python3
"""Build and run the scoped CUTLASS SM121 MXFP8 native transpose probe."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


PROBE_CU = Path(__file__).with_suffix(".cu")
DEFAULT_CUTLASS = Path("/home/dave/vllm/.deps/cutlass-src")


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def build(args: argparse.Namespace) -> Path:
    cutlass = args.cutlass.resolve()
    if not (cutlass / "include" / "cutlass" / "cutlass.h").exists():
        raise SystemExit(f"CUTLASS include tree not found under {cutlass}")
    build_dir = args.build_dir.resolve() if args.build_dir else Path(tempfile.gettempdir())
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "cutlass_mxfp8_native_transpose_probe"
    cmd = [
        args.nvcc,
        "-std=c++17",
        "-O2",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        f"-arch={args.arch}",
        f"-I{cutlass / 'include'}",
        f"-I{cutlass / 'tools' / 'util' / 'include'}",
        str(PROBE_CU),
        "-o",
        str(binary),
    ]
    if args.verbose:
        cmd.insert(3, "-Xptxas=-v")
    run(cmd)
    return binary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutlass", type=Path, default=DEFAULT_CUTLASS)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument(
        "--arch",
        default="sm_121a",
        help="nvcc real architecture. SM121 block-scaled MMA needs an accelerated arch such as sm_121a.",
    )
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument(
        "--attempt-te-compact",
        action="store_true",
        help="Run the direct TE compact scale layout attempt. This currently aborts in TMA descriptor creation and is opt-in.",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    binary = build(args)
    if args.compile_only:
        print(f"built={binary}")
        return 0

    cmd = [str(binary), f"--m={args.m}", f"--n={args.n}", f"--k={args.k}"]
    if args.layout_only:
        cmd.append("--layout-only")
    if args.attempt_te_compact:
        cmd.append("--attempt-te-compact")
    env = dict(os.environ)
    run(cmd, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
