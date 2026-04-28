#!/usr/bin/env python3
"""Decision-table benchmark for M2RNN ParaRNN implementations.

This complements ``bench_m2rnn_tiled_triton.py`` by comparing dense ParaRNN,
explicit tiled Triton tile sizes, and an optional external TileLang/shared-old
callable if one is available in the environment.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp64":
        return torch.float64
    return torch.float32


def _parse_ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def _make_inputs(B: int, S: int, H: int, K: int, V: int, *, device: str, dtype: torch.dtype, seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, K, generator=g, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, S, H, K, generator=g, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, S, H, V, generator=g, device=device, dtype=dtype) * 0.5
    W = torch.randn(H, V, V, generator=g, device=device, dtype=dtype) * (0.35 / math.sqrt(V))
    xf = torch.sigmoid(torch.randn(B, S, H, generator=g, device=device, dtype=dtype) - 0.5)
    return q, k, v, W, xf


def _measure(fn: Callable[[], tuple[torch.Tensor, torch.Tensor]], *, device: str, warmup: int, repeat: int):
    with torch.inference_mode():
        out: tuple[torch.Tensor, torch.Tensor] | None = None
        for _ in range(warmup):
            out = fn()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                out = fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / repeat, torch.cuda.max_memory_allocated(), out

        start_t = time.perf_counter()
        for _ in range(repeat):
            out = fn()
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0 / repeat
        return elapsed_ms, 0, out


def _load_callable(spec: str | None) -> tuple[str, Callable[..., Any] | None, str]:
    if not spec:
        return "tilelang_shared_old", None, "not_requested"
    if ":" not in spec:
        return "tilelang_shared_old", None, "expected MODULE:FUNC"
    module_name, func_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
    except Exception as exc:  # pragma: no cover - depends on optional installs.
        return "tilelang_shared_old", None, f"unavailable:{type(exc).__name__}:{exc}"
    return "tilelang_shared_old", func, "ok"


def _max_errors(
    out: tuple[torch.Tensor, torch.Tensor] | None,
    ref: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[str, str]:
    if out is None or ref is None:
        return "", ""
    max_out = (out[0].float() - ref[0].float()).abs().max().item()
    max_h = (out[1].float() - ref[1].float()).abs().max().item()
    return f"{max_out:.6e}", f"{max_h:.6e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--V", type=int, default=16)
    parser.add_argument("--sizes", default="16,32,512,4096,8192")
    parser.add_argument("--tiles", default="16,32,64")
    parser.add_argument("--dtypes", default="fp32,bf16")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dense-max-S",
        type=int,
        default=32,
        help="Run dense ParaRNN only up to this sequence length; use -1 to disable.",
    )
    parser.add_argument(
        "--tilelang-callable",
        default=None,
        help="Optional MODULE:FUNC with signature fn(q, k, v, W, xf) -> (out, h_final).",
    )
    args = parser.parse_args()

    sizes = _parse_ints(args.sizes)
    tiles = _parse_ints(args.tiles)
    dtype_names = [part.strip() for part in args.dtypes.split(",") if part.strip()]
    tilelang_name, tilelang_fn, tilelang_status = _load_callable(args.tilelang_callable)

    fieldnames = [
        "impl",
        "dtype",
        "B",
        "S",
        "H",
        "K",
        "V",
        "tile",
        "num_tiles",
        "latency_ms",
        "peak_alloc_bytes",
        "summary_bytes",
        "max_out",
        "max_h",
        "status",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()

    print(
        f"# device={args.device} triton_available={TRITON_AVAILABLE} "
        f"tilelang_status={tilelang_status}",
        file=sys.stderr,
    )

    for dtype_name in dtype_names:
        dtype = _dtype(dtype_name)
        for S in sizes:
            q, k, v, W, xf = _make_inputs(
                args.B, S, args.H, args.K, args.V, device=args.device, dtype=dtype, seed=args.seed
            )

            dense_ref: tuple[torch.Tensor, torch.Tensor] | None = None
            if args.dense_max_S >= 0 and S <= args.dense_max_S:
                dense_cfg = PararnnConfig(max_its=args.iters, init_strategy="zero", chunk_size=0)

                def dense_once():
                    return m2rnn_pararnn_forward(q, k, v, W, xf, config=dense_cfg)

                dense_ms, dense_peak, dense_ref = _measure(
                    dense_once, device=args.device, warmup=args.warmup, repeat=args.repeat
                )
                writer.writerow(
                    {
                        "impl": "dense_pararnn",
                        "dtype": dtype_name,
                        "B": args.B,
                        "S": S,
                        "H": args.H,
                        "K": args.K,
                        "V": args.V,
                        "tile": "",
                        "num_tiles": "",
                        "latency_ms": f"{dense_ms:.6f}",
                        "peak_alloc_bytes": dense_peak,
                        "summary_bytes": "",
                        "max_out": "0.000000e+00",
                        "max_h": "0.000000e+00",
                        "status": "ok",
                    }
                )

            best_tile: tuple[int, float] | None = None
            for tile in tiles:
                cfg = TiledTritonConfig(
                    max_its=args.iters,
                    init_strategy="zero",
                    tile_size=tile,
                    prefer_triton=True,
                )

                def tiled_once():
                    return m2rnn_pararnn_tiled_triton_forward(q, k, v, W, xf, config=cfg)

                try:
                    tiled_ms, tiled_peak, tiled_out = _measure(
                        tiled_once, device=args.device, warmup=args.warmup, repeat=args.repeat
                    )
                    max_out, max_h = _max_errors(tiled_out, dense_ref)
                    stats = estimate_tiled_solve_memory(
                        B=args.B,
                        S=S,
                        H=args.H,
                        K=args.K,
                        V=args.V,
                        tile_size=tile,
                        dtype=torch.promote_types(torch.float32, dtype),
                    )
                    status = "ok"
                    if best_tile is None or tiled_ms < best_tile[1]:
                        best_tile = (tile, tiled_ms)
                except Exception as exc:
                    tiled_ms = float("nan")
                    tiled_peak = 0
                    max_out = ""
                    max_h = ""
                    stats = estimate_tiled_solve_memory(
                        B=args.B,
                        S=S,
                        H=args.H,
                        K=args.K,
                        V=args.V,
                        tile_size=tile,
                        dtype=torch.promote_types(torch.float32, dtype),
                    )
                    status = f"error:{type(exc).__name__}:{exc}"
                writer.writerow(
                    {
                        "impl": "tiled_triton",
                        "dtype": dtype_name,
                        "B": args.B,
                        "S": S,
                        "H": args.H,
                        "K": args.K,
                        "V": args.V,
                        "tile": tile,
                        "num_tiles": stats.num_tiles,
                        "latency_ms": f"{tiled_ms:.6f}",
                        "peak_alloc_bytes": tiled_peak,
                        "summary_bytes": stats.summary_bytes,
                        "max_out": max_out,
                        "max_h": max_h,
                        "status": status,
                    }
                )

            if tilelang_fn is not None:

                def tilelang_once():
                    return tilelang_fn(q, k, v, W, xf)

                try:
                    tl_ms, tl_peak, tl_out = _measure(
                        tilelang_once, device=args.device, warmup=args.warmup, repeat=args.repeat
                    )
                    max_out, max_h = _max_errors(tl_out, dense_ref)
                    status = "ok"
                except Exception as exc:
                    tl_ms = float("nan")
                    tl_peak = 0
                    max_out = ""
                    max_h = ""
                    status = f"error:{type(exc).__name__}:{exc}"
                writer.writerow(
                    {
                        "impl": tilelang_name,
                        "dtype": dtype_name,
                        "B": args.B,
                        "S": S,
                        "H": args.H,
                        "K": args.K,
                        "V": args.V,
                        "tile": "",
                        "num_tiles": "",
                        "latency_ms": f"{tl_ms:.6f}",
                        "peak_alloc_bytes": tl_peak,
                        "summary_bytes": "",
                        "max_out": max_out,
                        "max_h": max_h,
                        "status": status,
                    }
                )

            if best_tile is not None:
                sys.stdout.flush()
                print(
                    f"# decision dtype={dtype_name} S={S} best_tiled_tile={best_tile[0]} "
                    f"latency_ms={best_tile[1]:.6f}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
