#!/usr/bin/env python3
"""Build and parity probe for the M2RNN tiled CUDA ParaRNN prototype."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cppmega.megatron.m2rnn_pararnn import PararnnConfig, m2rnn_pararnn_forward
from cppmega.megatron.m2rnn_pararnn_tiled_cuda import (
    TiledCudaPararnnConfig,
    memory_accounting_bytes,
    m2rnn_pararnn_tiled_cuda_forward,
)


def _torch_m2rnn_forward(q, k, v, W, xf, *, h0=None):
    batch, seq, n_q, k_dim = q.shape
    n_k = k.size(-2)
    n_v = v.size(-2)
    n_w = W.size(0)
    n_f = xf.size(-1)
    v_dim = v.size(-1)
    n = max(n_q, n_k, n_v, n_w, n_f)

    if h0 is None:
        h = torch.zeros(batch, n, k_dim, v_dim, device=q.device, dtype=q.dtype)
    else:
        h = h0

    if n_q != n:
        q = q.repeat_interleave(n // n_q, dim=-2)
    if n_k != n:
        k = k.repeat_interleave(n // n_k, dim=-2)
    if n_v != n:
        v = v.repeat_interleave(n // n_v, dim=-2)
    if n_w != n:
        W = W.repeat_interleave(n // n_w, dim=0)
    if n_f != n:
        xf = xf.repeat_interleave(n // n_f, dim=-1)

    x = k[..., None] * v[..., None, :]
    W_expanded = W[None, ...]
    y = torch.empty(batch, seq, n, k_dim, v_dim, device=q.device, dtype=q.dtype)
    for s in range(seq):
        f = xf[:, s, :, None, None]
        h_new = torch.tanh(h @ W_expanded + x[:, s])
        h = f * h + (1 - f) * h_new
        y[:, s] = h
    out = (q[..., None, :] @ y).squeeze(-2)
    return out, h


def _make_inputs(B: int, S: int, H: int, K: int, V: int, seed: int):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, K, generator=g, device=device, dtype=torch.float32) * 0.5
    k = torch.randn(B, S, H, K, generator=g, device=device, dtype=torch.float32) * 0.5
    v = torch.randn(B, S, H, V, generator=g, device=device, dtype=torch.float32) * 0.5
    W = torch.randn(H, V, V, generator=g, device=device, dtype=torch.float32) * (0.45 / math.sqrt(V))
    xf = torch.sigmoid(torch.randn(B, S, H, generator=g, device=device, dtype=torch.float32) - 0.5)
    return q, k, v, W, xf


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().detach().cpu())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=33)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--V", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--max-its", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    os.environ.setdefault("CPPMEGA_VERBOSE_EXT_BUILD", "1")
    q, k, v, W, xf = _make_inputs(args.B, args.S, args.H, args.K, args.V, args.seed)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_cuda, h_cuda = m2rnn_pararnn_tiled_cuda_forward(
        q,
        k,
        v,
        W,
        xf,
        config=TiledCudaPararnnConfig(max_its=args.max_its, tile_size=args.tile_size),
    )
    torch.cuda.synchronize()
    cuda_ms = (time.perf_counter() - t0) * 1000.0

    out_seq, h_seq = _torch_m2rnn_forward(q, k, v, W, xf)
    out_par, h_par = m2rnn_pararnn_forward(
        q, k, v, W, xf, config=PararnnConfig(max_its=args.max_its, chunk_size=args.tile_size)
    )
    torch.cuda.synchronize()

    result = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "shape": {
            "B": args.B,
            "S": args.S,
            "H": args.H,
            "K": args.K,
            "V": args.V,
            "tile_size": args.tile_size,
            "max_its": args.max_its,
        },
        "cuda_ms": cuda_ms,
        "max_abs_vs_sequential": {
            "out": _max_abs(out_cuda, out_seq),
            "h_final": _max_abs(h_cuda, h_seq),
        },
        "max_abs_pararnn_vs_sequential": {
            "out": _max_abs(out_par, out_seq),
            "h_final": _max_abs(h_par, h_seq),
        },
        "memory_bytes": memory_accounting_bytes(args.B, args.S, args.H, args.K, args.V, args.tile_size),
        "notes": [
            "CUDA kernel does not store per-token Jacobian A[B,S,H,K,V,V].",
            "Production path does not allocate local_prefix[Be,S,V,V]; apply kernel recomputes within-tile prefixes.",
            "Tile summary scan over tile_A/tile_b runs on CUDA; the Python loop is retained only as a test reference.",
            "Run with CPPMEGA_VERBOSE_EXT_BUILD=1 after clearing the torch extension cache to see ptxas register/spill lines.",
        ],
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tol = 5e-4
    if result["max_abs_vs_sequential"]["out"] > tol or result["max_abs_vs_sequential"]["h_final"] > tol:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
