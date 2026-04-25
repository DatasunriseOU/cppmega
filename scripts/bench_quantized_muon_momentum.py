#!/usr/bin/env python3
"""Correctness and timing smoke test for quantized Muon momentum."""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cppmega.megatron.quantized_muon_momentum import (  # noqa: E402
    TRITON_AVAILABLE,
    dequantize_momentum,
    empty_quantized_momentum_like,
    quantize_momentum_,
    quantized_muon_momentum_update_multi_and_normalize_,
    quantized_muon_momentum_update_multi_,
    quantized_muon_momentum_update_,
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


def _run_one(
    *,
    shape: tuple[int, int],
    storage_dtype: torch.dtype,
    beta: float,
    warmup: int,
    iters: int,
) -> None:
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(20260425)
    old_m = 0.2 * torch.randn(shape, device=device, dtype=torch.bfloat16, generator=gen)
    grad = 0.2 * torch.randn(shape, device=device, dtype=torch.bfloat16, generator=gen)

    state = empty_quantized_momentum_like(old_m, storage_dtype=storage_dtype)
    quantize_momentum_(state, old_m)
    old_m_dequant = dequantize_momentum(state)
    scratch = quantized_muon_momentum_update_(state, grad, beta=beta)

    exact = (beta * old_m_dequant + (1.0 - beta) * grad.float()).to(torch.bfloat16)
    bf16_ref = (beta * old_m.float() + (1.0 - beta) * grad.float()).to(torch.bfloat16)
    dequant_ref_diff = (scratch.float() - exact.float()).abs()
    bf16_diff = (scratch.float() - bf16_ref.float()).abs()
    dequant_diff = (dequantize_momentum(state) - bf16_ref.float()).abs()

    # Fresh states for timing so correctness work does not bias either path.
    q_state = empty_quantized_momentum_like(old_m, storage_dtype=storage_dtype)
    quantize_momentum_(q_state, old_m)
    q_scratch = torch.empty_like(old_m, dtype=torch.bfloat16)
    bf16_m = old_m.clone()

    q_ms = _time_cuda(
        lambda: quantized_muon_momentum_update_(q_state, grad, beta=beta, scratch=q_scratch),
        warmup=warmup,
        iters=iters,
    )
    q_native_state = empty_quantized_momentum_like(old_m, storage_dtype=storage_dtype)
    quantize_momentum_(q_native_state, old_m)
    native_grad = grad.clone()
    q_native_ms = _time_cuda(
        lambda: quantized_muon_momentum_update_multi_([q_native_state], [native_grad], beta=beta),
        warmup=warmup,
        iters=iters,
    )
    q_norm_state = empty_quantized_momentum_like(old_m, storage_dtype=storage_dtype)
    quantize_momentum_(q_norm_state, old_m)
    norm_grad = grad.clone()
    q_norm_ms = _time_cuda(
        lambda: quantized_muon_momentum_update_multi_and_normalize_([q_norm_state], [norm_grad], beta=beta),
        warmup=warmup,
        iters=iters,
    )
    q_torch_norm_state = empty_quantized_momentum_like(old_m, storage_dtype=storage_dtype)
    quantize_momentum_(q_torch_norm_state, old_m)
    torch_norm_grad = grad.clone()

    def _native_update_then_torch_norm():
        quantized_muon_momentum_update_multi_([q_torch_norm_state], [torch_norm_grad], beta=beta)
        _ = torch_norm_grad / torch_norm_grad.norm().clamp_min(1e-7)

    q_torch_norm_ms = _time_cuda(_native_update_then_torch_norm, warmup=warmup, iters=iters)
    bf16_ms = _time_cuda(lambda: bf16_m.lerp_(grad, 1.0 - beta), warmup=warmup, iters=iters)

    print(f"storage={storage_dtype} shape={shape} numel={old_m.numel():,}")
    print(
        "  correctness: "
        f"dequant_ref_max_abs={dequant_ref_diff.max().item():.6g} "
        f"vs_bf16_max_abs={bf16_diff.max().item():.6g} "
        f"vs_bf16_mean_abs={bf16_diff.mean().item():.6g} "
        f"dequant_vs_bf16_max_abs={dequant_diff.max().item():.6g}"
    )
    print(
        "  perf: "
        f"quantized_update={q_ms:.4f} ms "
        f"cuda_multi_inplace={q_native_ms:.4f} ms "
        f"cuda_multi_update_norm={q_norm_ms:.4f} ms "
        f"cuda_multi_plus_torch_norm={q_torch_norm_ms:.4f} ms "
        f"bf16_lerp={bf16_ms:.4f} ms "
        f"ratio={q_native_ms / bf16_ms:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--storage",
        choices=("int8", "uint8", "both"),
        default="both",
        help="Quantized momentum storage dtype to test.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not TRITON_AVAILABLE:
        raise SystemExit("Triton is required")

    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    shape = (args.rows, args.cols)
    storage_dtypes = [torch.int8, torch.uint8] if args.storage == "both" else [getattr(torch, args.storage)]
    for storage_dtype in storage_dtypes:
        _run_one(
            shape=shape,
            storage_dtype=storage_dtype,
            beta=args.beta,
            warmup=args.warmup,
            iters=args.iters,
        )


if __name__ == "__main__":
    main()
