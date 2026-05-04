"""Wave4 CUDA microbench for the Mamba3 bwd_bwd R x R diagonal path."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rr_diag_cuda_extension import rr_cuda_kernel_metadata, rr_specialized_cuda
from rr_diag_specialization import (
    Shape,
    _dtype,
    _time_cuda,
    _time_wall,
    full_fused_reference,
    make_inputs,
    max_diffs,
    rr_specialized_torch,
    rr_specialized_triton,
)
from rr_diag_wave3_microbench import PRESETS, _cta_model


def _shape_from_args(args: argparse.Namespace) -> Shape:
    if args.shape:
        values = PRESETS[args.shape]
        return Shape(
            B=values["B"],
            S=values["S"],
            H=values["H"],
            N=values["N"],
            P=values["P"],
            R=args.R,
            chunk=args.chunk,
        )
    return Shape(B=args.B, S=args.S, H=args.H, N=args.N, P=args.P, R=args.R, chunk=args.chunk)


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = _shape_from_args(args)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    inputs = make_inputs(shape, dtype=dtype, device=device, seed=args.seed)

    ref = full_fused_reference(inputs, shape)
    torch_rr = rr_specialized_torch(inputs, shape)
    correctness: dict[str, Any] = {"torch_rr_vs_full": max_diffs(ref, torch_rr)}

    timer = _time_cuda if device.type == "cuda" else _time_wall
    timings: dict[str, Any] = {
        "full_fused_torch_reference": timer(
            lambda: full_fused_reference(inputs, shape),
            warmup=args.warmup,
            iters=args.iters,
        ),
        "rr_torch_oracle": timer(
            lambda: rr_specialized_torch(inputs, shape),
            warmup=args.warmup,
            iters=args.iters,
        ),
    }

    metadata: dict[str, Any] = {}
    if device.type == "cuda":
        tri = rr_specialized_triton(inputs, shape, num_warps=args.num_warps)
        cuda = rr_specialized_cuda(inputs)
        torch.cuda.synchronize()
        correctness["triton_rr_vs_full"] = max_diffs(ref, tri)
        correctness["cuda_rr_vs_full"] = max_diffs(ref, cuda)
        correctness["cuda_rr_vs_triton_rr"] = max_diffs(tri, cuda)
        metadata["cuda_kernel"] = rr_cuda_kernel_metadata(inputs)
        timings["rr_triton_timestep_cta_wave3"] = _time_cuda(
            lambda: rr_specialized_triton(inputs, shape, num_warps=args.num_warps),
            warmup=args.warmup,
            iters=args.iters,
        )
        timings["rr_cuda_timestep_cta_wave4"] = _time_cuda(
            lambda: rr_specialized_cuda(inputs),
            warmup=args.warmup,
            iters=args.iters,
        )

    base = timings["full_fused_torch_reference"]["mean_ms"]
    triton_ms = timings.get("rr_triton_timestep_cta_wave3", {}).get("mean_ms")
    for row in timings.values():
        row["speedup_vs_full_fused_torch_reference"] = base / row["mean_ms"] if row["mean_ms"] else None
        row["speedup_vs_wave3_triton"] = triton_ms / row["mean_ms"] if triton_ms and row["mean_ms"] else None

    return {
        "shape_name": args.shape or "custom",
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cta_model": _cta_model(shape),
        "metadata": metadata,
        "correctness": correctness,
        "timings": timings,
        "integration_plan": [
            "Do not use a host-side post-kernel full-chain split in wave5.",
            "If CUDA beats wave3 Triton, port this block-owned R=4 path into the bwd_bwd launch boundary or call it as a device-side/CuTe helper from the fused kernel.",
            "If CUDA does not beat wave3 Triton, keep the Triton microbench as the performance target and spend wave5 on fusing the timestep-owned algorithm into bwd_bwd rather than further standalone tuning.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(PRESETS), default=None)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=256)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--P", type=int, default=128)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--num-warps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
