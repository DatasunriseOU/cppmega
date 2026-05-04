#!/usr/bin/env python3
"""Probe tiled/streaming M2RNN ParaRNN memory and parity."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cppmega.megatron.m2rnn_pararnn import PararnnConfig, m2rnn_pararnn_forward
from cppmega.megatron.m2rnn_pararnn_tiled_triton import (
    TRITON_AVAILABLE,
    TiledTritonConfig,
    estimate_tiled_solve_memory,
    m2rnn_pararnn_tiled_triton_forward,
)


def _bytes(x: int) -> str:
    val = float(x)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if val < 1024 or unit == "GiB":
            return f"{val:.2f} {unit}"
        val /= 1024
    return f"{val:.2f} B"


def _make_inputs(B, S, H, K, V, *, device, dtype, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, K, generator=g, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, S, H, K, generator=g, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, S, H, V, generator=g, device=device, dtype=dtype) * 0.5
    W = torch.randn(H, V, V, generator=g, device=device, dtype=dtype) * (0.35 / math.sqrt(V))
    xf = torch.sigmoid(torch.randn(B, S, H, generator=g, device=device, dtype=dtype) - 0.5)
    return q, k, v, W, xf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=64)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--V", type=int, default=4)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp64"), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-torch", action="store_true")
    args = parser.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp64": torch.float64}[args.dtype]
    q, k, v, W, xf = _make_inputs(
        args.B, args.S, args.H, args.K, args.V, device=args.device, dtype=dtype, seed=args.seed
    )
    prefer_triton = not args.force_torch
    out_tiled, h_tiled, stats = m2rnn_pararnn_tiled_triton_forward(
        q,
        k,
        v,
        W,
        xf,
        config=TiledTritonConfig(max_its=args.iters, tile_size=args.tile, prefer_triton=prefer_triton),
        return_stats=True,
    )
    out_dense, h_dense = m2rnn_pararnn_forward(
        q,
        k,
        v,
        W,
        xf,
        config=PararnnConfig(max_its=args.iters, chunk_size=0),
    )
    max_out = (out_tiled.float() - out_dense.float()).abs().max().item()
    max_h = (h_tiled.float() - h_dense.float()).abs().max().item()

    prod = estimate_tiled_solve_memory(B=2, S=4096, H=8, K=64, V=16, tile_size=64)
    print(f"device={args.device} dtype={dtype} triton_available={TRITON_AVAILABLE} prefer_triton={prefer_triton}")
    print(f"shape B={args.B} S={args.S} H={args.H} K={args.K} V={args.V} tile={args.tile}")
    print(f"parity max_out={max_out:.6e} max_h={max_h:.6e}")
    print(
        "probe memory: "
        f"full_A={_bytes(stats.full_A_bytes)} peak_tile_A={_bytes(stats.peak_tile_A_bytes)} "
        f"summary={_bytes(stats.summary_bytes)} ratio={stats.full_A_to_tile_ratio:.2f}x"
    )
    print(
        "NAM56R-like memory: "
        f"full_A={_bytes(prod.full_A_bytes)} peak_tile_A={_bytes(prod.peak_tile_A_bytes)} "
        f"summary={_bytes(prod.summary_bytes)} ratio={prod.full_A_to_tile_ratio:.2f}x"
    )


if __name__ == "__main__":
    main()
