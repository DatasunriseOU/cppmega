"""Benchmark M2RNN Triton backward memory/recompute knobs.

This is intentionally narrower than ``tools/bench_m2rnn_kernels_nam56r.py``:
it calls ``m2rnn_scan_triton`` directly with the production head topology
(``q/k`` one-head, ``v/xf/W`` full-head by default) so recurrent-state
checkpointing, ``SAVE_HNEW`` and backward chunk temporaries are visible without
projection/conv/norm noise from the full mixer.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Variant:
    save_hnew: int
    chunk_size: int


@dataclass
class Result:
    save_hnew: int
    chunk_size: int
    fwd_ms: float
    fwdbwd_ms: float
    stdev_ms: float
    peak_alloc_mib: float
    peak_reserved_mib: float
    active_peak_mib: float
    allocated_recurrent_mib: float
    finite: bool


def _make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    device = "cuda"
    dtype = torch.bfloat16
    g = torch.Generator(device=device).manual_seed(args.seed)

    q = torch.randn(
        args.batch, args.seq, args.q_heads, args.k_dim,
        device=device, dtype=dtype, generator=g,
    ).requires_grad_(True)
    k = torch.randn(
        args.batch, args.seq, args.k_heads, args.k_dim,
        device=device, dtype=dtype, generator=g,
    ).requires_grad_(True)
    v = torch.randn(
        args.batch, args.seq, args.v_heads, args.v_dim,
        device=device, dtype=dtype, generator=g,
    ).requires_grad_(True)
    W = (
        torch.eye(args.v_dim, device=device, dtype=dtype)
        .unsqueeze(0)
        .expand(args.w_heads, -1, -1)
        .contiguous()
        .clone()
    )
    W += 0.05 * torch.randn(W.shape, device=device, dtype=dtype, generator=g)
    W.requires_grad_(True)
    xf = torch.sigmoid(
        torch.randn(
            args.batch, args.seq, args.xf_heads,
            device=device, dtype=dtype, generator=g,
        )
    ).requires_grad_(True)
    return q, k, v, W, xf


def _zero_grads(tensors: tuple[torch.Tensor, ...]) -> None:
    for tensor in tensors:
        tensor.grad = None


def _recurrent_alloc_mib(args: argparse.Namespace, variant: Variant) -> float:
    common_heads = max(args.q_heads, args.k_heads, args.v_heads, args.w_heads, args.xf_heads)
    elems_per_state = args.batch * common_heads * args.k_dim * args.v_dim
    num_chunks = (args.seq + variant.chunk_size - 1) // variant.chunk_size
    checkpoint_bytes = elems_per_state * (num_chunks + 1) * 4
    y_chunk_bytes = elems_per_state * (min(args.seq, variant.chunk_size) + 1) * 4
    hnew_bytes = (
        elems_per_state * args.seq * torch.finfo(torch.bfloat16).bits // 8
        if variant.save_hnew
        else 0
    )
    dW_slab_bytes = args.batch * common_heads * args.v_dim * args.v_dim * 4
    return (checkpoint_bytes + y_chunk_bytes + hnew_bytes + dW_slab_bytes) / (1024 * 1024)


def _run_variant(args: argparse.Namespace, variant: Variant) -> Result:
    os.environ["CPPMEGA_M2RNN_SAVE_HNEW"] = str(variant.save_hnew)
    os.environ["CPPMEGA_M2RNN_BWD_CHUNK_SIZE"] = str(variant.chunk_size)
    os.environ.setdefault("CPPMEGA_M2RNN_BWD_REDUCE_BROADCAST_QK", "1")

    from cppmega.megatron.m2rnn_triton import (
        m2rnn_scan_triton,
        reset_m2rnn_runtime_config_cache,
    )

    reset_m2rnn_runtime_config_cache()
    tensors = _make_inputs(args)

    for _ in range(args.warmup):
        _zero_grads(tensors)
        out, _ = m2rnn_scan_triton(*tensors)
        out.float().pow(2).mean().backward()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    fwd_times: list[float] = []
    finite = True
    for _ in range(args.iters):
        _zero_grads(tensors)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, _ = m2rnn_scan_triton(*tensors)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = out.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1e3)
        times.append((t2 - t0) * 1e3)
        finite = finite and bool(torch.isfinite(out).all().item())
        finite = finite and all(
            tensor.grad is None or bool(torch.isfinite(tensor.grad).all().item())
            for tensor in tensors
        )

    stats = torch.cuda.memory_stats()
    result = Result(
        save_hnew=variant.save_hnew,
        chunk_size=variant.chunk_size,
        fwd_ms=statistics.mean(fwd_times),
        fwdbwd_ms=statistics.mean(times),
        stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        peak_alloc_mib=torch.cuda.max_memory_allocated() / (1024 * 1024),
        peak_reserved_mib=torch.cuda.max_memory_reserved() / (1024 * 1024),
        active_peak_mib=stats.get("active_bytes.all.peak", 0) / (1024 * 1024),
        allocated_recurrent_mib=_recurrent_alloc_mib(args, variant),
        finite=finite,
    )
    del tensors, out, loss
    torch.cuda.empty_cache()
    return result


def _profile_variant(args: argparse.Namespace, variant: Variant) -> str:
    os.environ["CPPMEGA_M2RNN_SAVE_HNEW"] = str(variant.save_hnew)
    os.environ["CPPMEGA_M2RNN_BWD_CHUNK_SIZE"] = str(variant.chunk_size)
    os.environ.setdefault("CPPMEGA_M2RNN_BWD_REDUCE_BROADCAST_QK", "1")

    from cppmega.megatron.m2rnn_triton import (
        m2rnn_scan_triton,
        reset_m2rnn_runtime_config_cache,
    )

    reset_m2rnn_runtime_config_cache()
    tensors = _make_inputs(args)
    _zero_grads(tensors)
    out, _ = m2rnn_scan_triton(*tensors)
    out.float().pow(2).mean().backward()
    torch.cuda.synchronize()

    _zero_grads(tensors)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        out, _ = m2rnn_scan_triton(*tensors)
        out.float().pow(2).mean().backward()
    torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.profile_rows)
    if args.profile_out:
        path = Path(args.profile_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table)
    del tensors, out
    torch.cuda.empty_cache()
    return table


def _format_results(results: list[Result]) -> str:
    headers = [
        "save",
        "chunk",
        "fwd ms",
        "fwd+bwd ms",
        "stdev",
        "peak alloc MiB",
        "peak reserved MiB",
        "active peak MiB",
        "recurrent MiB",
        "finite",
    ]
    rows = [
        [
            str(r.save_hnew),
            str(r.chunk_size),
            f"{r.fwd_ms:.2f}",
            f"{r.fwdbwd_ms:.2f}",
            f"{r.stdev_ms:.2f}",
            f"{r.peak_alloc_mib:.1f}",
            f"{r.peak_reserved_mib:.1f}",
            f"{r.active_peak_mib:.1f}",
            f"{r.allocated_recurrent_mib:.1f}",
            "yes" if r.finite else "NO",
        ]
        for r in results
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"

    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [sep, fmt(headers), sep]
    lines.extend(fmt(row) for row in rows)
    lines.append(sep)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=44)
    parser.add_argument("--k-dim", type=int, default=64)
    parser.add_argument("--v-dim", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=1)
    parser.add_argument("--k-heads", type=int, default=1)
    parser.add_argument("--v-heads", type=int, default=None)
    parser.add_argument("--w-heads", type=int, default=None)
    parser.add_argument("--xf-heads", type=int, default=None)
    parser.add_argument("--save-hnew", type=int, nargs="+", default=[0, 1], choices=[0, 1])
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-save-hnew", type=int, default=0, choices=[0, 1])
    parser.add_argument("--profile-chunk-size", type=int, default=64)
    parser.add_argument("--profile-rows", type=int, default=30)
    parser.add_argument("--profile-out", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; M2RNN Triton benchmark requires a GPU.", file=sys.stderr)
        return 1

    args.v_heads = args.heads if args.v_heads is None else args.v_heads
    args.w_heads = args.heads if args.w_heads is None else args.w_heads
    args.xf_heads = args.heads if args.xf_heads is None else args.xf_heads

    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(
        "Shape: "
        f"B={args.batch}, S={args.seq}, H={args.heads}, K={args.k_dim}, V={args.v_dim}, "
        f"q_heads={args.q_heads}, k_heads={args.k_heads}, "
        f"v_heads={args.v_heads}, xf_heads={args.xf_heads}, w_heads={args.w_heads}, dtype=bf16"
    )
    print(f"Iters: warmup={args.warmup}, timed={args.iters}")

    variants = [
        Variant(save_hnew=save_hnew, chunk_size=chunk_size)
        for save_hnew in args.save_hnew
        for chunk_size in args.chunk_sizes
    ]
    results = [_run_variant(args, variant) for variant in variants]
    print()
    print(_format_results(results))

    if args.profile:
        print()
        print(
            f"Torch profiler table for save={args.profile_save_hnew}, "
            f"chunk={args.profile_chunk_size}:"
        )
        print(_profile_variant(args, Variant(args.profile_save_hnew, args.profile_chunk_size)))
        if args.profile_out:
            print(f"Profiler table written to {args.profile_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
