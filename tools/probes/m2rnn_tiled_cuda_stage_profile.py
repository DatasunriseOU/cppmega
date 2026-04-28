#!/usr/bin/env python3
"""Stage-level profiler for the M2RNN tiled CUDA ParaRNN path."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cppmega.megatron.m2rnn_pararnn_tiled_cuda import (  # noqa: E402
    _broadcast_heads,
    _load_cuda_ext,
    _make_h0_row,
    _use_warprow_v16,
    m2rnn_pararnn_tiled_cuda_forward,
)


def _make_inputs(B: int, S: int, H: int, K: int, V: int, dtype: torch.dtype, seed: int):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, K, generator=g, device=device, dtype=torch.float32) * 0.5
    k = torch.randn(B, S, H, K, generator=g, device=device, dtype=torch.float32) * 0.5
    v = torch.randn(B, S, H, V, generator=g, device=device, dtype=torch.float32) * 0.5
    W = torch.randn(H, V, V, generator=g, device=device, dtype=torch.float32) * (0.45 / math.sqrt(V))
    xf = torch.sigmoid(torch.randn(B, S, H, generator=g, device=device, dtype=torch.float32) - 0.5)
    return q.to(dtype), k.to(dtype), v.to(dtype), W.to(dtype), xf.to(dtype)


def _event_time_ms(fn: Callable[[], object]) -> tuple[object, float, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    t0 = time.perf_counter()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return out, float(start.elapsed_time(end)), wall_ms


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _profile_once(args: argparse.Namespace, q, k, v, W, xf, ext) -> dict[str, object]:
    stages: dict[str, list[float]] = {
        "prep_cast_contiguous_gpu_ms": [],
        "h_alloc_gpu_ms": [],
        "workspace_alloc_gpu_ms": [],
        "summary_gpu_ms": [],
        "scan_gpu_ms": [],
        "apply_gpu_ms": [],
        "update_gpu_ms": [],
        "final_layout_gpu_ms": [],
        "final_einsum_gpu_ms": [],
    }
    wall: dict[str, list[float]] = {name.replace("_gpu_ms", "_wall_ms"): [] for name in stages}

    (prepared, gpu_ms, wall_ms) = _event_time_ms(
        lambda: tuple(t.to(dtype=torch.float32).contiguous() for t in _broadcast_heads(q, k, v, W, xf)[:5])
    )
    stages["prep_cast_contiguous_gpu_ms"].append(gpu_ms)
    wall["prep_cast_contiguous_wall_ms"].append(wall_ms)
    qf, kf, vf, Wf, xff = prepared
    B, S, H, K = qf.shape
    V = vf.size(-1)
    h0_row = _make_h0_row(None, B=B, H=H, K=K, V=V, device=q.device, dtype=torch.float32)
    Be = B * H * K
    n_tiles = (S + args.tile_size - 1) // args.tile_size
    if _use_warprow_v16(V, args.tile_size):
        summary_out = ext.tile_summaries_v16_warprow_out
        apply_out = ext.apply_tile_prefixes_v16_warprow_out
        kernel_variant = "v16_warprow"
    else:
        summary_out = ext.tile_summaries_out
        apply_out = ext.apply_tile_prefixes_out
        kernel_variant = "baseline"

    (h, gpu_ms, wall_ms) = _event_time_ms(lambda: torch.zeros(Be, S, V, device=q.device, dtype=torch.float32))
    stages["h_alloc_gpu_ms"].append(gpu_ms)
    wall["h_alloc_wall_ms"].append(wall_ms)

    ((tile_A, tile_b, tile_inputs, delta), gpu_ms, wall_ms) = _event_time_ms(
        lambda: (
            torch.empty(Be, n_tiles, V, V, device=q.device, dtype=torch.float32),
            torch.empty(Be, n_tiles, V, device=q.device, dtype=torch.float32),
            torch.empty(Be, n_tiles, V, device=q.device, dtype=torch.float32),
            torch.empty(Be, S, V, device=q.device, dtype=torch.float32),
        )
    )
    stages["workspace_alloc_gpu_ms"].append(gpu_ms)
    wall["workspace_alloc_wall_ms"].append(wall_ms)

    for _ in range(args.max_its):
        (_, gpu_ms, wall_ms) = _event_time_ms(
            lambda: summary_out(qf, kf, vf, Wf, xff, h.contiguous(), h0_row, tile_A, tile_b, args.tile_size)
        )
        stages["summary_gpu_ms"].append(gpu_ms)
        wall["summary_wall_ms"].append(wall_ms)

        (_, gpu_ms, wall_ms) = _event_time_ms(
            lambda: ext.scan_tile_summaries_out(tile_A, tile_b, tile_inputs)
        )
        stages["scan_gpu_ms"].append(gpu_ms)
        wall["scan_wall_ms"].append(wall_ms)

        (_, gpu_ms, wall_ms) = _event_time_ms(
            lambda: apply_out(
                qf,
                kf,
                vf,
                Wf,
                xff,
                h.contiguous(),
                h0_row,
                tile_inputs,
                delta,
                args.tile_size,
            )
        )
        stages["apply_gpu_ms"].append(gpu_ms)
        wall["apply_wall_ms"].append(wall_ms)

        (h, gpu_ms, wall_ms) = _event_time_ms(lambda: h + float(args.omega_sor) * delta)
        stages["update_gpu_ms"].append(gpu_ms)
        wall["update_wall_ms"].append(wall_ms)

    (h_btehv, gpu_ms, wall_ms) = _event_time_ms(
        lambda: h.view(B, H, K, S, V).permute(0, 3, 1, 2, 4).contiguous()
    )
    stages["final_layout_gpu_ms"].append(gpu_ms)
    wall["final_layout_wall_ms"].append(wall_ms)

    ((out, h_final), gpu_ms, wall_ms) = _event_time_ms(
        lambda: (torch.einsum("bshk,bshkv->bshv", qf, h_btehv), h_btehv[:, -1].contiguous())
    )
    stages["final_einsum_gpu_ms"].append(gpu_ms)
    wall["final_einsum_wall_ms"].append(wall_ms)

    return {
        "stage_gpu_ms": {name: _stats(values) for name, values in stages.items()},
        "stage_wall_ms": {name: _stats(values) for name, values in wall.items()},
        "kernel_variant": kernel_variant,
        "output_checksum": float(out.float().sum().detach().cpu()),
        "h_final_checksum": float(h_final.float().sum().detach().cpu()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=1024)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--V", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--max-its", type=int, default=3)
    parser.add_argument("--omega-sor", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    q, k, v, W, xf = _make_inputs(args.B, args.S, args.H, args.K, args.V, dtype, args.seed)

    t0 = time.perf_counter()
    ext = _load_cuda_ext()
    torch.cuda.synchronize()
    extension_load_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(args.warmup):
        m2rnn_pararnn_tiled_cuda_forward(
            q,
            k,
            v,
            W,
            xf,
            config=type("Cfg", (), {
                "max_its": args.max_its,
                "omega_sor": args.omega_sor,
                "tile_size": args.tile_size,
            })(),
        )
    torch.cuda.synchronize()

    results = [_profile_once(args, q, k, v, W, xf, ext) for _ in range(args.iters)]
    whole_values: list[float] = []
    for _ in range(args.iters):
        (_, _, wall_ms) = _event_time_ms(
            lambda: m2rnn_pararnn_tiled_cuda_forward(
                q,
                k,
                v,
                W,
                xf,
                config=type("Cfg", (), {
                    "max_its": args.max_its,
                    "omega_sor": args.omega_sor,
                    "tile_size": args.tile_size,
                })(),
            )
        )
        whole_values.append(wall_ms)

    stage_names = results[0]["stage_gpu_ms"].keys()
    wall_names = results[0]["stage_wall_ms"].keys()
    out = {
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
            "dtype": args.dtype,
        },
        "kernel_variant": results[0]["kernel_variant"],
        "extension_load_ms": extension_load_ms,
        "whole_forward_wall_ms": _stats(whole_values),
        "stage_gpu_ms": {
            name: _stats([float(r["stage_gpu_ms"][name]["mean"]) for r in results]) for name in stage_names
        },
        "stage_wall_ms": {
            name: _stats([float(r["stage_wall_ms"][name]["mean"]) for r in results]) for name in wall_names
        },
        "checksums": {
            "out": [r["output_checksum"] for r in results],
            "h_final": [r["h_final_checksum"] for r in results],
        },
        "notes": [
            "summary/scan/apply/update stats are per Newton iteration.",
            "gpu_ms uses CUDA events; wall_ms includes Python/C++ dispatch and allocation around each stage.",
        ],
    }

    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
