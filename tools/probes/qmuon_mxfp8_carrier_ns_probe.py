#!/usr/bin/env python3
"""Microbench qMuon MXFP8 carrier emission and first NS Gram GEMM.

This probe is deliberately narrower than the production Megatron optimizer:
it measures one 2D Muon tensor where q8 momentum emits rowwise MXFP8 carrier
storage, then CUTLASS consumes that carrier for the Newton-Schulz ``X @ X.T``
GEMM.  It is intended to answer whether the carrier/GEMM building block is
worth integrating into the full optimizer.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cppmega.megatron import cutlass_mxfp8_gemm as cutlass  # noqa: E402
from cppmega.megatron.quantized_muon_momentum import (  # noqa: E402
    dequantize_mxfp8_carrier,
    empty_mxfp8_carrier_like,
    empty_quantized_momentum_like,
    quantize_momentum_,
    quantized_muon_momentum_update_mxfp8_carrier_,
)


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) / iters


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.linalg.vector_norm((a.float() - b.float()).reshape(-1))
    den = torch.linalg.vector_norm(b.float().reshape(-1)).clamp_min(1e-12)
    return float((num / den).item())


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def run_probe(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rows % 128 or args.cols % 128:
        raise SystemExit("--rows and --cols must be multiples of 128 for CUTLASS MXFP8")
    if args.rows <= 0 or args.cols <= 0:
        raise SystemExit("--rows and --cols must be positive")

    device = torch.device("cuda")
    dtype = _dtype_from_name(args.grad_dtype)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.rows, args.cols)
    old_m = (args.scale * torch.randn(shape, device=device, dtype=dtype, generator=gen)).contiguous()
    grad = (args.scale * torch.randn(shape, device=device, dtype=dtype, generator=gen)).contiguous()
    grad_bf16 = grad.to(torch.bfloat16)

    q_state = empty_quantized_momentum_like(old_m, storage_dtype=torch.int8)
    quantize_momentum_(q_state, old_m)
    carrier = empty_mxfp8_carrier_like(grad)
    mxfp8_gram = torch.empty((args.rows, args.rows), device=device, dtype=torch.bfloat16)

    bf16_m = old_m.to(torch.bfloat16).contiguous()
    bf16_x = torch.empty_like(bf16_m)
    bf16_gram = torch.empty((args.rows, args.rows), device=device, dtype=torch.bfloat16)

    inv_norm = quantized_muon_momentum_update_mxfp8_carrier_(
        q_state,
        grad,
        carrier,
        beta=args.beta,
    )
    cutlass.tn_gemm_direct_rowwise(
        carrier.rowwise_data,
        carrier.rowwise_scale_inv,
        carrier.rowwise_data,
        carrier.rowwise_scale_inv,
        out=mxfp8_gram,
    )
    carrier_dequant = dequantize_mxfp8_carrier(carrier)
    reference_gram = carrier_dequant.float() @ carrier_dequant.float().mT
    torch.cuda.synchronize()

    def q_update_emit():
        quantized_muon_momentum_update_mxfp8_carrier_(
            q_state,
            grad,
            carrier,
            beta=args.beta,
        )

    def mxfp8_gram_only():
        cutlass.tn_gemm_direct_rowwise(
            carrier.rowwise_data,
            carrier.rowwise_scale_inv,
            carrier.rowwise_data,
            carrier.rowwise_scale_inv,
            out=mxfp8_gram,
        )

    def q_update_emit_plus_gram():
        quantized_muon_momentum_update_mxfp8_carrier_(
            q_state,
            grad,
            carrier,
            beta=args.beta,
        )
        cutlass.tn_gemm_direct_rowwise(
            carrier.rowwise_data,
            carrier.rowwise_scale_inv,
            carrier.rowwise_data,
            carrier.rowwise_scale_inv,
            out=mxfp8_gram,
        )

    def bf16_update_norm_gram():
        bf16_m.lerp_(grad_bf16, 1.0 - args.beta)
        inv = bf16_m.float().square().sum().clamp_min(args.eps * args.eps).rsqrt()
        bf16_x.copy_(bf16_m)
        bf16_x.mul_(inv)
        torch.mm(bf16_x, bf16_x.mT, out=bf16_gram)

    q_update_ms = _time_cuda(q_update_emit, warmup=args.warmup, iters=args.iters)
    mxfp8_gram_ms = _time_cuda(mxfp8_gram_only, warmup=args.warmup, iters=args.iters)
    q_total_ms = _time_cuda(q_update_emit_plus_gram, warmup=args.warmup, iters=args.iters)
    bf16_total_ms = _time_cuda(bf16_update_norm_gram, warmup=args.warmup, iters=args.iters)

    inv_norm = quantized_muon_momentum_update_mxfp8_carrier_(
        q_state,
        grad,
        carrier,
        beta=args.beta,
    )
    cutlass.tn_gemm_direct_rowwise(
        carrier.rowwise_data,
        carrier.rowwise_scale_inv,
        carrier.rowwise_data,
        carrier.rowwise_scale_inv,
        out=mxfp8_gram,
    )
    carrier_dequant = dequantize_mxfp8_carrier(carrier)
    reference_gram = carrier_dequant.float() @ carrier_dequant.float().mT
    torch.cuda.synchronize()

    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    print(
        "config: "
        f"shape={shape} numel={old_m.numel():,} grad_dtype={grad.dtype} "
        f"beta={args.beta} warmup={args.warmup} iters={args.iters}"
    )
    print(
        "carrier: "
        f"data_dtype={carrier.rowwise_data.dtype} data_shape={tuple(carrier.rowwise_data.shape)} "
        f"scale_dtype={carrier.rowwise_scale_inv.dtype} "
        f"scale_shape={tuple(carrier.rowwise_scale_inv.shape)} "
        f"inv_norm={float(inv_norm.detach().cpu()):.8e}"
    )
    print(
        "correctness: "
        f"carrier_gram_rel_l2={_rel_l2(mxfp8_gram, reference_gram):.6g} "
        f"carrier_finite={bool(torch.isfinite(carrier_dequant).all().item())} "
        f"gram_finite={bool(torch.isfinite(mxfp8_gram).all().item())}"
    )
    print(
        "perf_ms: "
        f"q_update_emit_mxfp8={q_update_ms:.4f} "
        f"mxfp8_gram={mxfp8_gram_ms:.4f} "
        f"q_update_emit_plus_mxfp8_gram={q_total_ms:.4f} "
        f"bf16_update_norm_gram={bf16_total_ms:.4f} "
        f"speedup_vs_bf16={bf16_total_ms / q_total_ms:.3f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--grad-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260430)
    run_probe(parser.parse_args())


if __name__ == "__main__":
    main()
