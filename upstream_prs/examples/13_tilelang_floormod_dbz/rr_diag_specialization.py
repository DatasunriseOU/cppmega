"""Isolated Mamba3 bwd_bwd diagonal R x R replacement harness.

This validates the part of ``mamba_mimo_bwd_bwd`` where the full
``(chunk_size * R, chunk_size * R)`` product

    dqk_from_diag = dPhiO @ PsiV.T

is only consumed on the per-time diagonal blocks.  The replacement computes
``chunk_size`` independent ``(R, R)`` products and feeds the same three users:
``DGAMMA_DIAG``, the diagonal contribution to ``DK``, and the diagonal
contribution to ``DQ``.

The script intentionally does not replace the off-time reverse-causal
``dk_intrachunk`` / ``dq_intrachunk`` path; those entries are semantically local
within a chunk but not diagonal, so the full masked local matrix is still
required unless a separate triangular/local algorithm is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Shape:
    B: int
    S: int
    H: int
    N: int
    P: int
    R: int = 4
    chunk: int = 16

    @property
    def nchunks(self) -> int:
        return math.ceil(self.S / self.chunk)

    @property
    def tiles(self) -> int:
        return self.B * self.H * self.nchunks

    @property
    def fused(self) -> int:
        return self.chunk * self.R


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def make_inputs(shape: Shape, *, dtype: torch.dtype, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scale = 0.25

    def randn(*size: int) -> torch.Tensor:
        return torch.randn(size, device=device, dtype=dtype, generator=generator) * scale

    return {
        "dphi": randn(shape.tiles, shape.chunk, shape.R, shape.P).contiguous(),
        "psiv": randn(shape.tiles, shape.chunk, shape.R, shape.P).contiguous(),
        "q_pre_rot": randn(shape.tiles, shape.chunk, shape.R, shape.N).contiguous(),
        "k_pre_rot": randn(shape.tiles, shape.chunk, shape.R, shape.N).contiguous(),
        "qk_dot": randn(shape.tiles, shape.chunk, shape.R, shape.R).contiguous(),
        "gamma": torch.randn(
            shape.tiles,
            shape.chunk,
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).contiguous()
        * scale,
    }


def full_fused_reference(inputs: dict[str, torch.Tensor], shape: Shape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Current full-matrix algorithm, reduced to the diagonal users."""

    dphi = inputs["dphi"].float().reshape(shape.tiles, shape.fused, shape.P)
    psiv = inputs["psiv"].float().reshape(shape.tiles, shape.fused, shape.P)
    full = torch.bmm(dphi, psiv.transpose(1, 2))
    diag = full.view(shape.tiles, shape.chunk, shape.R, shape.chunk, shape.R).diagonal(dim1=1, dim2=3)
    dqk = diag.permute(0, 3, 1, 2).contiguous()
    return consume_dqk_diag(dqk, inputs)


def rr_specialized_torch(inputs: dict[str, torch.Tensor], shape: Shape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """R x R specialization using torch/einsum as a portable oracle."""

    dqk = torch.einsum("tcrp,tcsp->tcrs", inputs["dphi"].float(), inputs["psiv"].float())
    return consume_dqk_diag(dqk, inputs)


def consume_dqk_diag(
    dqk: torch.Tensor,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same diagonal dqk block to DGAMMA_DIAG, DK, and DQ."""

    dgamma = (inputs["qk_dot"].float() * dqk).sum(dim=(-1, -2))
    scaled = dqk * inputs["gamma"][:, :, None, None]
    dk_delta = torch.einsum("tcri,tcrn->tcin", scaled, inputs["q_pre_rot"].float()).contiguous()
    dq_delta = torch.einsum("tcri,tcin->tcrn", scaled, inputs["k_pre_rot"].float()).contiguous()
    return dgamma.contiguous(), dk_delta, dq_delta


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        return False
    return True


if _has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _rr_diag_kernel(
        dphi,
        psiv,
        q_pre_rot,
        k_pre_rot,
        qk_dot,
        gamma,
        dgamma,
        dk_delta,
        dq_delta,
        total_programs: tl.constexpr,
        C: tl.constexpr,
        R: tl.constexpr,
        P: tl.constexpr,
        N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_p = tl.arange(0, BLOCK_P)
        offs_n = tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)

        tile = pid // C
        cs = pid - tile * C

        dphi_base = ((tile * C + cs) * R) * P
        psiv_base = dphi_base
        q_base = ((tile * C + cs) * R) * N
        k_base = q_base
        qk_base = ((tile * C + cs) * R) * R

        dphi_mat = tl.load(
            dphi + dphi_base + offs_r[:, None] * P + offs_p[None, :],
            mask=(offs_r[:, None] < R) & (offs_p[None, :] < P),
            other=0.0,
        )
        psiv_mat = tl.load(
            psiv + psiv_base + offs_p[:, None] + offs_r[None, :] * P,
            mask=(offs_p[:, None] < P) & (offs_r[None, :] < R),
            other=0.0,
        )
        dqk = tl.dot(dphi_mat, psiv_mat, input_precision="ieee", out_dtype=tl.float32)

        qk = tl.load(
            qk_dot + qk_base + offs_r[:, None] * R + offs_r[None, :],
            mask=(offs_r[:, None] < R) & (offs_r[None, :] < R),
            other=0.0,
        )
        dg = tl.sum(dqk * qk, axis=0)
        dg = tl.sum(dg, axis=0)
        tl.store(dgamma + tile * C + cs, dg)

        g = tl.load(gamma + tile * C + cs)
        scaled = dqk * g
        q_mat = tl.load(
            q_pre_rot + q_base + offs_r[:, None] * N + offs_n[None, :],
            mask=(offs_r[:, None] < R) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        k_mat = tl.load(
            k_pre_rot + k_base + offs_r[:, None] * N + offs_n[None, :],
            mask=(offs_r[:, None] < R) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        dk = tl.dot(tl.trans(scaled), q_mat, input_precision="ieee", out_dtype=tl.float32)
        dq = tl.dot(scaled, k_mat, input_precision="ieee", out_dtype=tl.float32)
        out_base = ((tile * C + cs) * R) * N
        tl.store(
            dk_delta + out_base + offs_r[:, None] * N + offs_n[None, :],
            dk,
            mask=(offs_r[:, None] < R) & (offs_n[None, :] < N),
        )
        tl.store(
            dq_delta + out_base + offs_r[:, None] * N + offs_n[None, :],
            dq,
            mask=(offs_r[:, None] < R) & (offs_n[None, :] < N),
        )


def rr_specialized_triton(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
    *,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _has_triton():
        raise RuntimeError("triton is not importable")
    if not inputs["dphi"].is_cuda:
        raise RuntimeError("triton path requires CUDA tensors")

    import triton

    block_p = triton.next_power_of_2(shape.P)
    block_n = triton.next_power_of_2(shape.N)
    dgamma = torch.empty(shape.tiles, shape.chunk, device=inputs["dphi"].device, dtype=torch.float32)
    dk_delta = torch.empty(shape.tiles, shape.chunk, shape.R, shape.N, device=inputs["dphi"].device, dtype=torch.float32)
    dq_delta = torch.empty_like(dk_delta)
    grid = (shape.tiles * shape.chunk,)
    _rr_diag_kernel[grid](
        inputs["dphi"],
        inputs["psiv"],
        inputs["q_pre_rot"],
        inputs["k_pre_rot"],
        inputs["qk_dot"],
        inputs["gamma"],
        dgamma,
        dk_delta,
        dq_delta,
        shape.tiles * shape.chunk,
        shape.chunk,
        shape.R,
        shape.P,
        shape.N,
        16,
        block_p,
        block_n,
        num_warps=num_warps,
    )
    return dgamma, dk_delta, dq_delta


def max_diffs(
    ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    got: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    names = ("dgamma", "dk_delta", "dq_delta")
    return {
        f"{name}_max_abs": float((lhs.float() - rhs.float()).abs().max().item())
        for name, lhs, rhs in zip(names, ref, got, strict=True)
    }


def _time_cuda(fn: Any, *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return _stats(samples)


def _time_wall(fn: Any, *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return _stats(samples)


def _stats(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    mean = sum(samples) / len(samples)
    return {
        "mean_ms": mean,
        "min_ms": ordered[0],
        "p50_ms": ordered[len(ordered) // 2],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def flop_model(shape: Shape) -> dict[str, float]:
    full_dqk = shape.tiles * (shape.fused * shape.fused) * shape.P * 2
    rr_dqk = shape.tiles * shape.chunk * shape.R * shape.R * shape.P * 2
    diag_consumers = shape.tiles * shape.chunk * (shape.R * shape.R + 2 * shape.R * shape.R * shape.N) * 2
    return {
        "full_dqk_flops": float(full_dqk),
        "rr_dqk_flops": float(rr_dqk),
        "dqk_reduction": float(full_dqk / rr_dqk),
        "rr_plus_dq_dk_flops": float(rr_dqk + diag_consumers),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = Shape(B=args.B, S=args.S, H=args.H, N=args.N, P=args.P, R=args.R, chunk=args.chunk)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    inputs = make_inputs(shape, dtype=dtype, device=device, seed=args.seed)

    ref = full_fused_reference(inputs, shape)
    specialized = rr_specialized_torch(inputs, shape)
    correctness: dict[str, Any] = {"torch_rr_vs_full": max_diffs(ref, specialized)}

    timings: dict[str, Any] = {}
    timer = _time_cuda if device.type == "cuda" else _time_wall
    timings["full_fused_torch"] = timer(lambda: full_fused_reference(inputs, shape), warmup=args.warmup, iters=args.iters)
    timings["rr_specialized_torch"] = timer(lambda: rr_specialized_torch(inputs, shape), warmup=args.warmup, iters=args.iters)

    if args.triton and device.type == "cuda":
        tri = rr_specialized_triton(inputs, shape, num_warps=args.num_warps)
        torch.cuda.synchronize()
        correctness["triton_rr_vs_full"] = max_diffs(ref, tri)
        timings["rr_specialized_triton"] = _time_cuda(
            lambda: rr_specialized_triton(inputs, shape, num_warps=args.num_warps),
            warmup=args.warmup,
            iters=args.iters,
        )

    out = {
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "correctness": correctness,
        "timings": timings,
        "flop_model": flop_model(shape),
    }
    for name, row in timings.items():
        base = timings["full_fused_torch"]["mean_ms"]
        row["speedup_vs_full_fused_torch"] = base / row["mean_ms"] if row["mean_ms"] else None
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--triton", action="store_true")
    parser.add_argument("--num-warps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
