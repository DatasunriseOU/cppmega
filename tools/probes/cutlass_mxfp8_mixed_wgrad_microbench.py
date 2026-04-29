#!/usr/bin/env python3
"""Profile cppmega's mixed MXFP8 wgrad path.

This isolates the full-model slow case where ``dy`` is still the original TE
compact-columnwise tensor and ``x.T`` has already been saved as compact rowwise
MXFP8.  The timed kernel computes:

    dy.T[out_n, reduction] @ x[reduction, out_k]

without materializing ``dy.T``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import torch
import transformer_engine  # noqa: F401 - loads TE common libs before TE torch extension
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor import MXFP8Quantizer

from cppmega.megatron import cutlass_mxfp8_gemm as cutlass


def _quantize_mxfp8(tensor: torch.Tensor) -> Any:
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    quantizer.internal = True
    quantizer.optimize_for_gemm = False
    return quantizer(tensor)


def _time_call(fn: Any) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    reduction = args.reduction
    out_n = args.out_n
    out_k = args.out_k
    if reduction % 128 or out_n % 128 or out_k % 128:
        raise SystemExit("reduction, out-n, and out-k must be multiples of 128")

    x = (torch.randn((reduction, out_k), device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    dy = (torch.randn((reduction, out_n), device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    torch.cuda.synchronize()

    xq = _quantize_mxfp8(x)
    dyq = _quantize_mxfp8(dy)
    x_t_data = xq._columnwise_data.t().contiguous()
    x_t_scale = xq._columnwise_scale_inv.t().contiguous()
    out = torch.empty((out_n, out_k), device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()

    def mixed_wgrad() -> torch.Tensor:
        a_columnwise_smem = args.backend == "a_col_smem_scalar"
        return cutlass._tn_gemm_compact_direct(
            dyq._columnwise_data,
            dyq._columnwise_scale_inv,
            x_t_data,
            x_t_scale,
            m=out_n,
            n=out_k,
            k=reduction,
            a_source=cutlass._SOURCE_COLUMNWISE_TRANSPOSE,
            a_data_ld=int(dyq._columnwise_data.shape[1]),
            a_scale_ld=int(dyq._columnwise_scale_inv.shape[1]),
            b_source=cutlass._SOURCE_ROWWISE,
            b_data_ld=int(x_t_data.shape[1]),
            b_scale_ld=int(x_t_scale.shape[1]),
            out=out,
            asymmetric=True,
            a_columnwise_smem=a_columnwise_smem,
        )

    for _ in range(args.warmup):
        mixed_wgrad()
    torch.cuda.synchronize()

    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    times = [_time_call(mixed_wgrad) for _ in range(args.iters)]
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()

    finite = bool(torch.isfinite(out).all().item())
    return {
        "timestamp_unix": time.time(),
        "device": torch.cuda.get_device_name(),
        "shape": {
            "dy": [reduction, out_n],
            "x_t": [out_k, reduction],
            "logical_gemm": [out_n, out_k, reduction],
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "backend": args.backend,
        "finite": finite,
        "elapsed_ms": {
            "min": min(times),
            "median": statistics.median(times),
            "mean": statistics.fmean(times),
            "max": max(times),
            "samples": times if args.samples else [],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduction", type=int, default=16384)
    parser.add_argument("--out-n", type=int, default=3584)
    parser.add_argument("--out-k", type=int, default=3584)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--backend", choices=("legacy", "a_col_smem_scalar"), default="legacy")
    parser.add_argument("--samples", action="store_true")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Wrap only the timed calls in cudaProfilerStart/Stop for ncu/nsys capture.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be >= 0 and --iters must be > 0")

    print(json.dumps(_run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
