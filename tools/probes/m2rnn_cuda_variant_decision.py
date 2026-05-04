#!/usr/bin/env python3
"""Compare M2RNN CUDA diagnostic variants against Triton on one shape."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cppmega.megatron.m2rnn_pararnn_tiled_cuda import (  # noqa: E402
    TiledCudaPararnnConfig,
    m2rnn_pararnn_tiled_cuda_forward,
)
from cppmega.megatron.m2rnn_triton import m2rnn_scan_triton  # noqa: E402


@contextmanager
def _temporary_env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _make_inputs(B: int, S: int, H: int, K: int, V: int, dtype: torch.dtype, seed: int):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, K, device=device, dtype=dtype, generator=g)
    k = torch.randn(B, S, H, K, device=device, dtype=dtype, generator=g)
    v = torch.randn(B, S, H, V, device=device, dtype=dtype, generator=g)
    W = torch.eye(V, device=device, dtype=dtype).unsqueeze(0).expand(H, -1, -1).contiguous().clone()
    W += 0.05 * torch.randn(W.shape, device=device, dtype=dtype, generator=g)
    xf = torch.sigmoid(torch.randn(B, S, H, device=device, dtype=dtype, generator=g))
    return q, k, v, W, xf


def _active_compute_processes() -> list[dict[str, str]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if completed.returncode != 0:
        return []

    current_pid = str(os.getpid())
    processes = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=2)]
        if len(parts) != 3 or parts[0] == current_pid:
            continue
        processes.append({"pid": parts[0], "process_name": parts[1], "used_memory_mib": parts[2]})
    return processes


def _flatten_output(out):
    if isinstance(out, tuple):
        return out[0]
    return out


def _time_cuda_fn(fn: Callable[[], object], *, warmup: int, iters: int) -> dict[str, object]:
    last = None
    for _ in range(warmup):
        last = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_t0 = time.perf_counter()
    for _ in range(iters):
        last = fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_t0) * 1000.0 / iters
    event_ms = float(start.elapsed_time(end)) / iters
    checksum = float(_flatten_output(last).float().sum().detach().cpu())
    return {
        "event_ms_per_iter": event_ms,
        "wall_ms_per_iter": wall_ms,
        "checksum": checksum,
    }


def _variant_decision(default_ms: float, candidate_ms: float, threshold: float) -> str:
    if candidate_ms < default_ms * (1.0 - threshold):
        return "continue_candidate"
    return "diagnostic_only"


def _format_markdown(result: dict[str, object]) -> str:
    shape = result["shape"]
    rows = []
    for variant in result["variants"]:
        rows.append(
            "| {name} | {event:.3f} | {wall:.3f} | {speedup:.2f}x | {decision} |".format(
                name=variant["name"],
                event=variant["event_ms_per_iter"],
                wall=variant["wall_ms_per_iter"],
                speedup=variant["speedup_vs_default_event"],
                decision=variant["decision"],
            )
        )

    return "\n".join(
        [
            "# M2RNN CUDA Variant Decision - 2026-04-28",
            "",
            "Status: evidence",
            "Canonical: docs/status/m2rnn_tiled_cuda_2026_04_28.md",
            "Date: 2026-04-28",
            "Scope: Cycle 5 CUDA/Triton variant decision for M2RNN tiled CUDA branch.",
            "",
            f"Device: `{result['device']}`, capability `{tuple(result['capability'])}`, torch `{result['torch']}`.",
            "",
            "Shape:",
            "",
            (
                f"- `B={shape['B']}, S={shape['S']}, H={shape['H']}, K={shape['K']}, "
                f"V={shape['V']}, tile_size={shape['tile_size']}, max_its={shape['max_its']}, "
                f"dtype={shape['dtype']}`"
            ),
            "",
            "Decision rule:",
            "",
            (
                f"- A CUDA candidate must beat default CUDA by more than "
                f"{result['threshold_pct']:.0f}% on CUDA-event timing before it becomes worth "
                "production follow-up."
            ),
            "- Otherwise this CUDA branch remains resource/diagnostic, with Triton as the active path.",
            "- Discard runs if `nvidia-smi` shows unrelated compute processes on the same GPU.",
            "",
            "| Variant | CUDA event ms/iter | Wall ms/iter | Speedup vs default event | Decision |",
            "| --- | ---: | ---: | ---: | --- |",
            *rows,
            "",
            "Recommendation:",
            "",
            f"- `{result['recommendation']}`",
            "",
            "Row-block prototype decision:",
            "",
            "- Not implemented as a kept variant in this cycle.",
            "- A one-block-per-`(Be,tile,row)` summary row would need each row block to see the full previous `d[16]` and all rows of `M[16,16]` at every token step.",
            "- CUDA blocks cannot synchronize or exchange shared state within a tile, so a safe row-block split would require extra global intermediate state or one kernel launch per token step.",
            "- That would directly add either the large prefix storage this branch removed or thousands of fine-grained launches, so it is de-prioritized until a different matrix-composition strategy is designed.",
            "",
            "Sources used:",
            "",
            "- NVIDIA CUDA Occupancy Calculator: register and shared-memory use constrain active blocks/warps per SM. https://docs.nvidia.com/cuda/archive/11.7.1/cuda-occupancy-calculator/index.html",
            "- CUDA C++ Programming Guide occupancy APIs and profiler occupancy guidance. https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html",
            "- CUDA C++ Best Practices Guide occupancy discussion. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html",
            "- CCCL/CUB WarpScan documentation for warp-wide scan primitives. https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpScan.html",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=1024)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--V", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--max-its", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--allow-gpu-contention", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.V != 16:
        raise ValueError("decision probe currently compares the V=16 default and warprow CUDA variants")
    active_processes = _active_compute_processes()
    if active_processes and not args.allow_gpu_contention:
        raise RuntimeError(
            "GPU has active compute processes; rerun on an idle device or pass "
            f"--allow-gpu-contention for a diagnostic-only run: {active_processes}"
        )

    dtype = torch.bfloat16
    q, k, v, W, xf = _make_inputs(args.B, args.S, args.H, args.K, args.V, dtype, args.seed)
    config = TiledCudaPararnnConfig(max_its=args.max_its, tile_size=args.tile_size)

    with _temporary_env("CPPMEGA_M2RNN_WARPROW_V16", None), _temporary_env("CPPMEGA_M2RNN_APPROX_TANH", None):
        default = _time_cuda_fn(
            lambda: m2rnn_pararnn_tiled_cuda_forward(q, k, v, W, xf, config=config),
            warmup=args.warmup,
            iters=args.iters,
        )
    with _temporary_env("CPPMEGA_M2RNN_WARPROW_V16", None), _temporary_env("CPPMEGA_M2RNN_APPROX_TANH", "1"):
        approx_tanh = _time_cuda_fn(
            lambda: m2rnn_pararnn_tiled_cuda_forward(q, k, v, W, xf, config=config),
            warmup=args.warmup,
            iters=args.iters,
        )
    with _temporary_env("CPPMEGA_M2RNN_WARPROW_V16", "1"), _temporary_env("CPPMEGA_M2RNN_APPROX_TANH", None):
        warprow = _time_cuda_fn(
            lambda: m2rnn_pararnn_tiled_cuda_forward(q, k, v, W, xf, config=config),
            warmup=args.warmup,
            iters=args.iters,
        )
    with _temporary_env("CPPMEGA_M2RNN_WARPROW_V16", "1"), _temporary_env("CPPMEGA_M2RNN_APPROX_TANH", "1"):
        warprow_approx_tanh = _time_cuda_fn(
            lambda: m2rnn_pararnn_tiled_cuda_forward(q, k, v, W, xf, config=config),
            warmup=args.warmup,
            iters=args.iters,
        )

    triton = _time_cuda_fn(
        lambda: m2rnn_scan_triton(q, k, v, W, xf),
        warmup=args.warmup,
        iters=args.iters,
    )

    default_event = float(default["event_ms_per_iter"])
    variants = []
    for name, values in (
        ("cuda_default", default),
        ("cuda_approx_tanh_opt_in", approx_tanh),
        ("cuda_v16_warprow_opt_in", warprow),
        ("cuda_v16_warprow_approx_tanh_opt_in", warprow_approx_tanh),
        ("triton_reference", triton),
    ):
        event_ms = float(values["event_ms_per_iter"])
        decision = "active_reference" if name == "triton_reference" else "default"
        if name.startswith("cuda_") and name != "cuda_default":
            decision = _variant_decision(default_event, event_ms, args.threshold)
        variants.append(
            {
                "name": name,
                **values,
                "speedup_vs_default_event": default_event / event_ms if event_ms > 0 else 0.0,
                "decision": decision,
            }
        )

    best_cuda = min(
        [row for row in variants if str(row["name"]).startswith("cuda_")],
        key=lambda row: float(row["event_ms_per_iter"]),
    )
    recommendation = "pause_cuda_production_work_keep_resource_diagnostic_branch"
    if best_cuda["name"] != "cuda_default" and best_cuda["decision"] == "continue_candidate":
        recommendation = f"continue_cuda_candidate_{best_cuda['name']}"

    result = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "shape": {
            "B": args.B,
            "S": args.S,
            "H": args.H,
            "K": args.K,
            "V": args.V,
            "tile_size": args.tile_size,
            "max_its": args.max_its,
            "dtype": "bf16",
        },
        "threshold_pct": args.threshold * 100.0,
        "variants": variants,
        "recommendation": recommendation,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.write_text(_format_markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
