"""Wave3 standalone microbench for the Mamba3 bwd_bwd R x R diagonal path.

This is intentionally not a stage2 post-kernel variant.  It isolates the
algorithmic question for Lane A: can the same-time R x R diagonal users run
faster than the current full ``[chunk * R, chunk * R]`` product when given
enough CTA parallelism?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


PRESETS: dict[str, dict[str, int]] = {
    "smoke": {"B": 1, "S": 256, "H": 4, "N": 64, "P": 64},
    "representative": {"B": 2, "S": 1024, "H": 16, "N": 64, "P": 64},
    "productionish": {"B": 4, "S": 4096, "H": 32, "N": 64, "P": 128},
}


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


def _cta_model(shape: Shape) -> dict[str, Any]:
    timestep_ctas = shape.tiles * shape.chunk
    return {
        "tile_ctas_if_one_per_chunk": shape.tiles,
        "timestep_ctas": timestep_ctas,
        "ctas_per_sm_at_132_sms": timestep_ctas / 132.0,
        "work_per_timestep": {
            "dqk_dot_flops": 2 * shape.R * shape.R * shape.P,
            "dk_dq_consumer_flops": 2 * 2 * shape.R * shape.R * shape.N,
        },
    }


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

    if device.type == "cuda":
        tri = rr_specialized_triton(inputs, shape, num_warps=args.num_warps)
        torch.cuda.synchronize()
        correctness["triton_rr_vs_full"] = max_diffs(ref, tri)
        timings["rr_triton_timestep_cta"] = _time_cuda(
            lambda: rr_specialized_triton(inputs, shape, num_warps=args.num_warps),
            warmup=args.warmup,
            iters=args.iters,
        )

    base = timings["full_fused_torch_reference"]["mean_ms"]
    for row in timings.values():
        row["speedup_vs_full_fused_torch_reference"] = base / row["mean_ms"] if row["mean_ms"] else None

    return {
        "shape_name": args.shape or "custom",
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cta_model": _cta_model(shape),
        "correctness": correctness,
        "timings": timings,
        "integration_plan": [
            "Do not use a host-side post-kernel. Move this timestep-owned R x R code into the bwd_bwd launch boundary.",
            "Custom CUDA/CuTe path: assign one CTA per (B,H,timestep), compute the 4x4 dPhiO@PsiV.T block, then immediately apply DGAMMA_DIAG, DK, and DQ diagonal consumers before the CTA exits.",
            "Keep the existing full reverse-causal off-time dk_intrachunk/dq_intrachunk path unchanged; only replace the same-time dqk_from_diag users.",
            "Use the stage2 bwd_fwd WS/TMA kernel as-is and replace only bwd_bwd with a custom kernel or a TileLang extern call once the full bwd_bwd body is ported around this microkernel.",
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
