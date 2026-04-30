"""Wave6 chunk-owner CUDA prototype for the Mamba3 bwd_bwd R x R diagonal path.

This keeps the wave4 diagonal math but changes the ownership model to match the
real bwd_bwd launch boundary more closely: one CTA owns a production
``(B, H, chunk)`` tile and computes all same-time ``R x R`` diagonal consumers
for that chunk in one kernel.  The prototype uses the stage2 tensor layouts and
writes DK/DQ diagonal contributions directly, avoiding the wave5 post-kernel
store/reload/add envelope.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rr_diag_cuda_extension import (  # noqa: E402
    stage2_rr_diag_chunk_owner_cuda,
    stage2_rr_diag_chunk_owner_cuda_metadata,
    stage2_rr_diag_chunk_owner_cuda_out,
    stage2_rr_diag_chunk_warp_owner_cuda,
    stage2_rr_diag_chunk_warp_owner_cuda_metadata,
    stage2_rr_diag_chunk_warp_owner_cuda_out,
    stage2_rr_diag_post_cuda,
    stage2_rr_diag_post_cuda_metadata,
)


@dataclass(frozen=True)
class Shape:
    B: int
    S: int
    H: int
    G: int
    N: int
    P: int
    R: int = 4
    chunk: int = 16

    @property
    def nchunks(self) -> int:
        return math.ceil(self.S / self.chunk)


PRESETS: dict[str, dict[str, int]] = {
    "smoke": {"B": 1, "S": 256, "H": 4, "G": 1, "N": 64, "P": 64},
    "representative": {"B": 2, "S": 1024, "H": 16, "G": 1, "N": 64, "P": 64},
    "productionish": {"B": 4, "S": 4096, "H": 32, "G": 1, "N": 64, "P": 128},
}

COMPARISON_CONTEXT: dict[str, Any] = {
    "wave4_standalone_cuda_diag_productionish_ms": 2.0560,
    "wave5_stage2_bf1_bb0_productionish_bwd_bwd_ms": 3.6971,
    "wave5_stage2_bf1_bb0_productionish_chain_ms": 5.4528,
    "wave5_stage2_rr_diag_cuda_split_productionish_bwd_bwd_ms": 6.5335,
    "wave5_stage2_rr_diag_cuda_split_productionish_chain_ms": 8.2905,
}


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _shape_from_args(args: argparse.Namespace) -> Shape:
    if args.shape:
        values = PRESETS[args.shape]
        return Shape(
            B=values["B"],
            S=values["S"],
            H=values["H"],
            G=values["G"],
            N=values["N"],
            P=values["P"],
            R=args.R,
            chunk=args.chunk,
        )
    return Shape(B=args.B, S=args.S, H=args.H, G=args.G, N=args.N, P=args.P, R=args.R, chunk=args.chunk)


def _randn(
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    *size: int,
    scale: float = 0.01,
) -> torch.Tensor:
    return (torch.randn(size, device=device, dtype=dtype, generator=generator) * scale).contiguous()


def make_inputs(shape: Shape, *, dtype: torch.dtype, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = _randn(generator, device, dtype, shape.B, shape.S, shape.R, shape.G, shape.N)
    k = _randn(generator, device, dtype, shape.B, shape.S, shape.R, shape.G, shape.N)
    return {
        "q": q,
        "k": k,
        "q_flat": q.view(shape.B, shape.S * shape.R, shape.G, shape.N),
        "k_flat": k.view(shape.B, shape.S * shape.R, shape.G, shape.N),
        "v": _randn(generator, device, dtype, shape.B, shape.S, shape.H, shape.P),
        "dout": _randn(generator, device, dtype, shape.B, shape.S, shape.H, shape.P),
        "q_bias": _randn(generator, device, torch.float32, shape.H, shape.R, shape.N),
        "k_bias": _randn(generator, device, torch.float32, shape.H, shape.R, shape.N),
        "mimo_v": _randn(generator, device, torch.float32, shape.H, shape.R, shape.P),
        "mimo_o": _randn(generator, device, torch.float32, shape.H, shape.R, shape.P),
        "qk_dot": _randn(generator, device, dtype, shape.B, shape.H, shape.S, shape.R * shape.R),
        "dt": _randn(generator, device, torch.float32, shape.B, shape.H, shape.S),
        "trap": _randn(generator, device, dtype, shape.B, shape.H, shape.S),
    }


def _empty_outputs(shape: Shape, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dgamma_diag = torch.empty(shape.B, shape.H, shape.S, device=device, dtype=torch.float32)
    dk_delta = torch.empty(shape.B, shape.S * shape.R, shape.H, shape.N, device=device, dtype=dtype)
    dq_delta = torch.empty_like(dk_delta)
    return dgamma_diag, dk_delta, dq_delta


def _zero_outputs(shape: Shape, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dgamma_diag = torch.zeros(shape.B, shape.H, shape.S, device=device, dtype=torch.float32)
    dk_delta = torch.zeros(shape.B, shape.S * shape.R, shape.H, shape.N, device=device, dtype=dtype)
    dq_delta = torch.zeros_like(dk_delta)
    return dgamma_diag, dk_delta, dq_delta


def stage2_post_cuda_reference(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dgamma_diag, dk_delta, dq_delta = _zero_outputs(
        shape, dtype=inputs["dout"].dtype, device=inputs["dout"].device
    )
    stage2_rr_diag_post_cuda(
        dout=inputs["dout"],
        q_flat=inputs["q_flat"],
        k_flat=inputs["k_flat"],
        v=inputs["v"],
        q_bias=inputs["q_bias"],
        k_bias=inputs["k_bias"],
        mimo_v=inputs["mimo_v"],
        mimo_o=inputs["mimo_o"],
        qk_dot=inputs["qk_dot"],
        dt=inputs["dt"],
        trap=inputs["trap"],
        dk=dk_delta,
        dq=dq_delta,
        dgamma_diag=dgamma_diag,
    )
    return dgamma_diag, dk_delta, dq_delta


def chunk_owner_cuda(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return stage2_rr_diag_chunk_owner_cuda(
        dout=inputs["dout"],
        q_flat=inputs["q_flat"],
        k_flat=inputs["k_flat"],
        v=inputs["v"],
        q_bias=inputs["q_bias"],
        k_bias=inputs["k_bias"],
        mimo_v=inputs["mimo_v"],
        mimo_o=inputs["mimo_o"],
        qk_dot=inputs["qk_dot"],
        dt=inputs["dt"],
        trap=inputs["trap"],
        chunk_size=shape.chunk,
    )


def chunk_warp_owner_cuda(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return stage2_rr_diag_chunk_warp_owner_cuda(
        dout=inputs["dout"],
        q_flat=inputs["q_flat"],
        k_flat=inputs["k_flat"],
        v=inputs["v"],
        q_bias=inputs["q_bias"],
        k_bias=inputs["k_bias"],
        mimo_v=inputs["mimo_v"],
        mimo_o=inputs["mimo_o"],
        qk_dot=inputs["qk_dot"],
        dt=inputs["dt"],
        trap=inputs["trap"],
        chunk_size=shape.chunk,
    )


def torch_reference(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout = inputs["dout"].float().permute(0, 2, 1, 3).contiguous()
    v = inputs["v"].float().permute(0, 2, 1, 3).contiguous()
    dphi = dout[:, :, :, None, :] * inputs["mimo_o"].float()[None, :, None, :, :]
    psiv = v[:, :, :, None, :] * inputs["mimo_v"].float()[None, :, None, :, :]
    dqk = torch.einsum("bhsrp,bhsip->bhsri", dphi, psiv)
    qk_dot = inputs["qk_dot"].float().view(shape.B, shape.H, shape.S, shape.R, shape.R)
    dgamma_diag = (qk_dot * dqk).sum(dim=(-1, -2)).contiguous()
    gamma = inputs["dt"].float() * torch.sigmoid(inputs["trap"].float())

    dk_delta = torch.empty(
        shape.B,
        shape.S,
        shape.R,
        shape.H,
        shape.N,
        device=inputs["dout"].device,
        dtype=torch.float32,
    )
    dq_delta = torch.empty_like(dk_delta)
    for h in range(shape.H):
        h_qk = h // (shape.H // shape.G)
        q_pre = inputs["q"][:, :, :, h_qk, :].float() + inputs["q_bias"][h].float()[None, None, :, :]
        k_pre = inputs["k"][:, :, :, h_qk, :].float() + inputs["k_bias"][h].float()[None, None, :, :]
        scaled = dqk[:, h] * gamma[:, h, :, None, None]
        dk_delta[:, :, :, h, :] = torch.einsum("bsri,bsrn->bsin", scaled, q_pre)
        dq_delta[:, :, :, h, :] = torch.einsum("bsri,bsin->bsrn", scaled, k_pre)

    return (
        dgamma_diag,
        dk_delta.reshape(shape.B, shape.S * shape.R, shape.H, shape.N).to(inputs["dout"].dtype).contiguous(),
        dq_delta.reshape(shape.B, shape.S * shape.R, shape.H, shape.N).to(inputs["dout"].dtype).contiguous(),
    )


def max_diffs(
    ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    got: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    names = ("dgamma_diag", "dk_delta", "dq_delta")
    return {
        name: float((lhs.float() - rhs.float()).abs().max().item())
        for name, lhs, rhs in zip(names, ref, got)
    }


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    mean = sum(ordered) / len(ordered)
    var = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean_ms": mean,
        "min_ms": ordered[0],
        "p50_ms": ordered[len(ordered) // 2],
        "max_ms": ordered[-1],
        "std_ms": math.sqrt(var),
        "samples_ms": values,
    }


def _time_cuda(fn: Callable[[], None], *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return _stats(samples)


def _time_wall(fn: Callable[[], Any], *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return _stats(samples)


def _cta_model(shape: Shape) -> dict[str, Any]:
    chunk_ctas = shape.B * shape.H * shape.nchunks
    timestep_ctas = shape.B * shape.H * shape.S
    return {
        "chunk_owner_ctas": chunk_ctas,
        "wave4_timestep_ctas": timestep_ctas,
        "cta_reduction_vs_timestep_owner": timestep_ctas / chunk_ctas,
        "chunk_owner_ctas_per_sm_at_132_sms": chunk_ctas / 132.0,
        "timesteps_per_cta": shape.chunk,
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

    correctness: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    timer = _time_cuda if device.type == "cuda" else _time_wall

    if device.type == "cuda":
        post_ref = stage2_post_cuda_reference(inputs, shape)
        chunk = chunk_owner_cuda(inputs, shape)
        chunk_warp = chunk_warp_owner_cuda(inputs, shape)
        torch.cuda.synchronize()
        correctness["wave6_chunk_owner_vs_wave5_timestep_post_cuda"] = max_diffs(post_ref, chunk)
        correctness["wave6_chunk_warp_owner_vs_wave5_timestep_post_cuda"] = max_diffs(post_ref, chunk_warp)
        metadata["stage2_post_timestep_cta"] = stage2_rr_diag_post_cuda_metadata(inputs["dout"])
        metadata["wave6_chunk_owner"] = stage2_rr_diag_chunk_owner_cuda_metadata(inputs["dout"])
        metadata["wave6_chunk_warp_owner"] = stage2_rr_diag_chunk_warp_owner_cuda_metadata(inputs["dout"])

        post_dgamma, post_dk, post_dq = _zero_outputs(shape, dtype=dtype, device=device)
        chunk_dgamma, chunk_dk, chunk_dq = _empty_outputs(shape, dtype=dtype, device=device)
        chunk_warp_dgamma, chunk_warp_dk, chunk_warp_dq = _empty_outputs(shape, dtype=dtype, device=device)

        def run_post() -> None:
            stage2_rr_diag_post_cuda(
                dout=inputs["dout"],
                q_flat=inputs["q_flat"],
                k_flat=inputs["k_flat"],
                v=inputs["v"],
                q_bias=inputs["q_bias"],
                k_bias=inputs["k_bias"],
                mimo_v=inputs["mimo_v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                dk=post_dk,
                dq=post_dq,
                dgamma_diag=post_dgamma,
            )

        def run_chunk_owner() -> None:
            stage2_rr_diag_chunk_owner_cuda_out(
                dout=inputs["dout"],
                q_flat=inputs["q_flat"],
                k_flat=inputs["k_flat"],
                v=inputs["v"],
                q_bias=inputs["q_bias"],
                k_bias=inputs["k_bias"],
                mimo_v=inputs["mimo_v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                dgamma_diag=chunk_dgamma,
                dk_delta=chunk_dk,
                dq_delta=chunk_dq,
                chunk_size=shape.chunk,
            )

        def run_chunk_warp_owner() -> None:
            stage2_rr_diag_chunk_warp_owner_cuda_out(
                dout=inputs["dout"],
                q_flat=inputs["q_flat"],
                k_flat=inputs["k_flat"],
                v=inputs["v"],
                q_bias=inputs["q_bias"],
                k_bias=inputs["k_bias"],
                mimo_v=inputs["mimo_v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                dgamma_diag=chunk_warp_dgamma,
                dk_delta=chunk_warp_dk,
                dq_delta=chunk_warp_dq,
                chunk_size=shape.chunk,
            )

        timings["wave5_stage2_cuda_post_timestep_cta_slice"] = timer(
            run_post, warmup=args.warmup, iters=args.iters
        )
        timings["wave6_chunk_owner_inlaunch_cuda_slice"] = timer(
            run_chunk_owner, warmup=args.warmup, iters=args.iters
        )
        timings["wave6_chunk_warp_owner_inlaunch_cuda_slice"] = timer(
            run_chunk_warp_owner, warmup=args.warmup, iters=args.iters
        )
    else:
        ref = torch_reference(inputs, shape)
        correctness["torch_reference_self"] = max_diffs(ref, ref)
        timings["torch_reference"] = timer(lambda: torch_reference(inputs, shape), warmup=args.warmup, iters=args.iters)

    if args.torch_reference and device.type == "cuda":
        torch_ref = torch_reference(inputs, shape)
        chunk = chunk_owner_cuda(inputs, shape)
        chunk_warp = chunk_warp_owner_cuda(inputs, shape)
        torch.cuda.synchronize()
        correctness["wave6_chunk_owner_vs_torch_reference"] = max_diffs(torch_ref, chunk)
        correctness["wave6_chunk_warp_owner_vs_torch_reference"] = max_diffs(torch_ref, chunk_warp)

    post_ms = timings.get("wave5_stage2_cuda_post_timestep_cta_slice", {}).get("mean_ms")
    for timing_name in (
        "wave6_chunk_owner_inlaunch_cuda_slice",
        "wave6_chunk_warp_owner_inlaunch_cuda_slice",
    ):
        chunk_ms = timings.get(timing_name, {}).get("mean_ms")
        if post_ms and chunk_ms:
            timings[timing_name]["speedup_vs_wave5_timestep_post_slice"] = post_ms / chunk_ms
            timings[timing_name]["ratio_vs_wave4_standalone_cuda_diag_prod"] = (
                chunk_ms / COMPARISON_CONTEXT["wave4_standalone_cuda_diag_productionish_ms"]
                if args.shape == "productionish"
                else None
            )
            timings[timing_name]["ratio_vs_stage2_bf1_bb0_bwd_bwd_prod"] = (
                chunk_ms / COMPARISON_CONTEXT["wave5_stage2_bf1_bb0_productionish_bwd_bwd_ms"]
                if args.shape == "productionish"
                else None
            )

    return {
        "shape_name": args.shape or "custom",
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "comparison_context": COMPARISON_CONTEXT,
        "cta_model": _cta_model(shape),
        "metadata": metadata,
        "correctness": correctness,
        "timings": timings,
        "read": [
            "This is a chunk-level in-launch slice prototype, not a full bwd_bwd replacement.",
            "It uses production stage2 layouts and avoids the extra post-kernel output reload/add/store envelope.",
            "A production integration still needs the surrounding DK/DQ/DGAMMA non-diagonal work ported or a TileLang device-helper hook.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(PRESETS), default=None)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=256)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--G", type=int, default=1)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--P", type=int, default=128)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--torch-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
