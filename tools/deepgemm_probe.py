#!/usr/bin/env python3
"""Non-invasive DeepGEMM/GB10 probe.

This script does not import deep_gemm and does not launch GPU kernels.  It is
intended to capture the local CUDA/Torch/GPU state and scan cloned DeepGEMM
source trees for architecture dispatch evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any


DEFAULT_SOURCE_PATHS = (
    pathlib.Path("/tmp/deepgemm-gb10-check/upstream"),
    pathlib.Path("/tmp/deepgemm-gb10-check/medmekk-hf"),
)


PATTERNS = {
    "exact_arch_major_9": re.compile(r"arch_major\s*==\s*9"),
    "exact_arch_major_10": re.compile(r"arch_major\s*==\s*10"),
    "cuda_arch_900_guard": re.compile(r"__CUDA_ARCH__\s*>=\s*900"),
    "cuda_arch_1000_guard": re.compile(r"__CUDA_ARCH__\s*>=\s*1000"),
    "tcgen05": re.compile(r"tcgen05\."),
    "wgmma": re.compile(r"wgmma|GMMA|mma_sm90_gmma"),
    "sm121": re.compile(r"sm_?121|12\.1|SM121", re.IGNORECASE),
}


def run(argv: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return {"returncode": proc.returncode, "output": proc.stdout.strip()}
    except Exception as exc:  # pragma: no cover - diagnostic tool
        return {"error": repr(exc)}


def torch_state() -> dict[str, Any]:
    try:
        import torch

        state: dict[str, Any] = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            state["device_name"] = torch.cuda.get_device_name(0)
            state["device_capability"] = torch.cuda.get_device_capability(0)
        return state
    except Exception as exc:  # pragma: no cover - diagnostic tool
        return {"error": repr(exc)}


def package_specs() -> dict[str, str | None]:
    names = ["deep_gemm", "triton", "tilelang", "transformer_engine"]
    result: dict[str, str | None] = {}
    for name in names:
        spec = importlib.util.find_spec(name)
        result[name] = None if spec is None else spec.origin
    return result


def git_state(path: pathlib.Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {}
    return {
        "head": run(["git", "-C", str(path), "rev-parse", "HEAD"]),
        "last_commit": run(["git", "-C", str(path), "log", "-1", "--format=%H%x09%ci%x09%s"]),
    }


def scan_source(path: pathlib.Path) -> dict[str, Any]:
    matches: dict[str, list[str]] = {key: [] for key in PATTERNS}
    if not path.exists():
        return {"exists": False}
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in {".py", ".hpp", ".h", ".cuh", ".cu", ".cpp", ".toml", ".md", ".txt"}:
            continue
        try:
            text = file_path.read_text(errors="ignore")
        except OSError:
            continue
        rel = file_path.relative_to(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for key, pattern in PATTERNS.items():
                if pattern.search(line):
                    if len(matches[key]) < 20:
                        matches[key].append(f"{rel}:{line_no}: {line.strip()}")
    return {
        "exists": True,
        "git": git_state(path),
        "matches": {key: value for key, value in matches.items() if value},
    }


def compile_empty_arch(arch: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="deepgemm-probe-") as tmp:
        src = pathlib.Path(tmp) / "empty.cu"
        out = pathlib.Path(tmp) / "empty.cubin"
        src.touch()
        return run(["nvcc", str(src), "-cubin", "-o", str(out), f"--gpu-architecture={arch}"], timeout=20)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        type=pathlib.Path,
        help="DeepGEMM source tree to scan; may be passed multiple times.",
    )
    parser.add_argument(
        "--compile-empty",
        action="store_true",
        help="Also compile an empty CUDA file for sm_121a/sm_120f/sm_100a. This does not launch GPU kernels.",
    )
    args = parser.parse_args()

    sources = tuple(args.source) if args.source else DEFAULT_SOURCE_PATHS
    report: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "torch": torch_state(),
        "packages": package_specs(),
        "nvidia_smi": run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv",
            ]
        ),
        "nvcc_version": run(["nvcc", "--version"]),
        "nvcc_arches": run(["nvcc", "--list-gpu-arch"]),
        "sources": {str(path): scan_source(path) for path in sources},
    }
    if args.compile_empty:
        report["empty_arch_compile"] = {
            arch: compile_empty_arch(arch)
            for arch in ("sm_121a", "sm_120f", "sm_100a")
        }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
