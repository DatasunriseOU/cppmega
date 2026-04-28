#!/usr/bin/env python3
"""Benchmark FlashInfer/CUTLASS SM120 MXFP8 GEMM runner choices for one shape."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor import MXFP8Quantizer

from cppmega.megatron import flashinfer_mxfp8_gemm


def _time_cuda(fn: Any, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    den = torch.linalg.vector_norm(b.float().reshape(-1)).clamp_min(1e-12)
    return float((torch.linalg.vector_norm((a.float() - b.float()).reshape(-1)) / den).item())


def _quantize_rowwise(tensor: torch.Tensor) -> Any:
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    quantizer.internal = True
    quantizer.optimize_for_gemm = False
    return quantizer(tensor)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(args.seed)

    x = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    weight = torch.randn((args.n, args.k), device=device, dtype=torch.bfloat16)
    xq = _quantize_rowwise(x)
    wq = _quantize_rowwise(weight)

    a_data = xq._rowwise_data
    b_data = wq._rowwise_data.t()
    a_scale = flashinfer_mxfp8_gemm.swizzle_rowwise_scale(xq._rowwise_scale_inv, args.m, args.k)
    b_scale = flashinfer_mxfp8_gemm.swizzle_rowwise_scale(wq._rowwise_scale_inv, args.n, args.k)
    out = torch.empty((args.m, args.n), device=device, dtype=torch.bfloat16)

    modes: list[dict[str, Any]] = []
    baseline: torch.Tensor | None = None
    for mode in ("mm_mxfp8", "direct_tactic"):
        config = flashinfer_mxfp8_gemm.runner_config(mode, args.tactic)
        row: dict[str, Any] = {"mode": config.mode, "tactic": config.tactic}
        try:
            row["elapsed_ms"] = _time_cuda(
                lambda: flashinfer_mxfp8_gemm._mm_mxfp8(  # noqa: SLF001 - probe-only
                    a_data,
                    b_data,
                    a_scale,
                    b_scale,
                    out=out,
                    out_dtype=torch.bfloat16,
                    config=config,
                ),
                warmup=args.warmup,
                iters=args.iters,
            )
            current = out.detach().clone()
            row["finite"] = bool(torch.isfinite(current).all().item())
            if baseline is None:
                baseline = current
                row["max_abs_vs_baseline"] = 0.0
                row["rel_l2_vs_baseline"] = 0.0
            else:
                row["max_abs_vs_baseline"] = float((current - baseline).abs().max().item())
                row["rel_l2_vs_baseline"] = _rel_l2(current, baseline)
        except Exception as exc:  # pragma: no cover - host/backend-specific probe path
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc).splitlines()[0]
        modes.append(row)

    tactics: list[dict[str, Any]] = []
    if args.try_all_tactics:
        from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, _get_cache_buf

        major, _minor = torch.cuda.get_device_capability(device)
        runner = flashinfer_mxfp8_gemm._load_flashinfer_cutlass_runner(major)  # noqa: SLF001
        workspace = _get_cache_buf(
            "cppmega_flashinfer_mxfp8_shape_probe_workspace",
            DEFAULT_WORKSPACE_SIZE,
            device,
        )
        a = a_data.view(torch.float8_e4m3fn)
        b = b_data.view(torch.float8_e4m3fn)
        runner_inputs = [a, b, a_scale, b_scale, torch.bfloat16, out, workspace]
        for tactic in runner.get_valid_tactics(runner_inputs, None):
            row = {"tactic": tactic}
            try:
                row["elapsed_ms"] = _time_cuda(
                    lambda tactic=tactic: runner.forward(runner_inputs, tactic=tactic),
                    warmup=args.warmup,
                    iters=args.iters,
                )
            except Exception as exc:  # pragma: no cover - host/backend-specific probe path
                row["error_type"] = type(exc).__name__
                row["error"] = str(exc).splitlines()[0]
            tactics.append(row)

    bf16_ref: dict[str, Any] | None = None
    if args.include_bf16_ref and baseline is not None:
        ref = x @ weight.t()
        bf16_ref = {
            "max_abs": float((baseline - ref).abs().max().item()),
            "rel_l2": _rel_l2(baseline, ref),
        }

    return {
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "device": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "iters": args.iters,
        "modes": modes,
        "tactics": tactics,
        "bf16_ref": bf16_ref,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tactic", type=int, default=0)
    parser.add_argument("--try-all-tactics", action="store_true")
    parser.add_argument("--include-bf16-ref", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise SystemExit("--m, --n, and --k must be positive")
    if args.k % 32 or args.n % 32:
        raise SystemExit("--k and --n must be multiples of 32 for SM120 FlashInfer MXFP8")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be >= 0 and --iters must be > 0")
    if args.tactic < 0:
        raise SystemExit("--tactic must be non-negative")

    print(json.dumps(_run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
