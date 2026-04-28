#!/usr/bin/env python3
"""Tile-size benchmark for the tiled Triton M2RNN ParaRNN prototype."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cppmega.megatron.m2rnn_pararnn import PararnnConfig, m2rnn_pararnn_forward
from cppmega.megatron.m2rnn_pararnn_tiled_triton import (
    TRITON_AVAILABLE,
    TiledTritonConfig,
    estimate_tiled_solve_memory,
    m2rnn_pararnn_tiled_triton_forward,
)


PROFILES = {
    "gb10-small": {"B": 1, "S": 512, "H": 4, "K": 16, "V": 16, "tiles": "16,32,64,128"},
    "h200-small": {"B": 2, "S": 1024, "H": 8, "K": 16, "V": 16, "tiles": "32,64,128"},
}


def _bytes(x: int) -> str:
    val = float(x)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if val < 1024 or unit == "GiB":
            return f"{val:.2f} {unit}"
        val /= 1024
    return f"{val:.2f} B"


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp64":
        return torch.float64
    return torch.float32


def _make_inputs(B: int, S: int, H: int, K: int, V: int, *, device: str, dtype: torch.dtype, seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, K, generator=g, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, S, H, K, generator=g, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, S, H, V, generator=g, device=device, dtype=dtype) * 0.5
    W = torch.randn(H, V, V, generator=g, device=device, dtype=dtype) * (0.35 / math.sqrt(V))
    xf = torch.sigmoid(torch.randn(B, S, H, generator=g, device=device, dtype=dtype) - 0.5)
    return q, k, v, W, xf


def _measure(fn, *, device: str, warmup: int, repeat: int) -> tuple[float, int]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / repeat, torch.cuda.max_memory_allocated()

        peak_bytes = 0
        start_t = time.perf_counter()
        for _ in range(repeat):
            fn()
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0 / repeat
        return elapsed_ms, peak_bytes


def _parse_tiles(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="gb10-small")
    parser.add_argument("--B", type=int)
    parser.add_argument("--S", type=int)
    parser.add_argument("--H", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--V", type=int)
    parser.add_argument("--tiles")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp64"), default="fp32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-torch", action="store_true")
    parser.add_argument("--check-dense", action="store_true")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    B = args.B if args.B is not None else profile["B"]
    S = args.S if args.S is not None else profile["S"]
    H = args.H if args.H is not None else profile["H"]
    K = args.K if args.K is not None else profile["K"]
    V = args.V if args.V is not None else profile["V"]
    tiles = _parse_tiles(args.tiles if args.tiles is not None else profile["tiles"])
    dtype = _dtype(args.dtype)
    prefer_triton = not args.force_torch

    q, k, v, W, xf = _make_inputs(B, S, H, K, V, device=args.device, dtype=dtype, seed=args.seed)
    print(
        f"profile={args.profile} device={args.device} dtype={dtype} "
        f"triton_available={TRITON_AVAILABLE} prefer_triton={prefer_triton}"
    )
    print(f"shape B={B} S={S} H={H} K={K} V={V} iters={args.iters} tiles={tiles}")
    print("tile,num_tiles,latency_ms,peak_alloc,full_A,peak_tile_A,summary,ratio")

    dense_ref = None
    if args.check_dense:
        dense_ref = m2rnn_pararnn_forward(
            q,
            k,
            v,
            W,
            xf,
            config=PararnnConfig(max_its=args.iters, init_strategy="zero", chunk_size=0),
        )

    for tile in tiles:
        cfg = TiledTritonConfig(
            max_its=args.iters,
            init_strategy="zero",
            tile_size=tile,
            prefer_triton=prefer_triton,
        )

        def run_once():
            return m2rnn_pararnn_tiled_triton_forward(q, k, v, W, xf, config=cfg)

        elapsed_ms, peak_alloc = _measure(run_once, device=args.device, warmup=args.warmup, repeat=args.repeat)
        compute_dtype = torch.promote_types(torch.float32, dtype)
        stats = estimate_tiled_solve_memory(B=B, S=S, H=H, K=K, V=V, tile_size=tile, dtype=compute_dtype)
        line = (
            f"{tile},{stats.num_tiles},{elapsed_ms:.3f},{_bytes(peak_alloc)},"
            f"{_bytes(stats.full_A_bytes)},{_bytes(stats.peak_tile_A_bytes)},"
            f"{_bytes(stats.summary_bytes)},{stats.full_A_to_tile_ratio:.2f}x"
        )
        if dense_ref is not None:
            out, h = run_once()
            max_out = (out.float() - dense_ref[0].float()).abs().max().item()
            max_h = (h.float() - dense_ref[1].float()).abs().max().item()
            line += f",max_out={max_out:.6e},max_h={max_h:.6e}"
        print(line)


if __name__ == "__main__":
    main()
