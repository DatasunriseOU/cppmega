#!/usr/bin/env python3
"""Probe the M2RNN tiled TileLang ParaRNN path."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cppmega.megatron.m2rnn_pararnn import PararnnConfig, m2rnn_pararnn_forward
from cppmega.megatron import m2rnn_pararnn_tiled_tilelang as tiled_impl
from cppmega.megatron.m2rnn_pararnn_tiled_tilelang import (
    TiledTileLangConfig,
    m2rnn_pararnn_tiled_tilelang_forward,
)


def _make_inputs(B, S, H, k_dim, v_dim, *, device, dtype, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, k_dim, generator=g, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, S, H, k_dim, generator=g, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, S, H, v_dim, generator=g, device=device, dtype=dtype) * 0.5
    W = torch.randn(H, v_dim, v_dim, generator=g, device=device, dtype=dtype) * (
        0.5 / math.sqrt(v_dim)
    )
    xf = torch.sigmoid(torch.randn(B, S, H, generator=g, device=device, dtype=dtype) - 0.5)
    return q, k, v, W, xf


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _time_forward(fn, *, device: str, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / repeats


def _time_cuda_kernel(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def _print_stage_breakdown(args, q, k, v, W, xf, *, tile_len: int, device: str) -> None:
    if device != "cuda":
        print("stage_breakdown_unavailable=requires_cuda")
        return

    (
        _qf,
        x_proj,
        f_t,
        W_be,
        h0_row,
        _q_broadcast,
        _B,
        S,
        _H,
        _k_dim,
    ) = tiled_impl._prepare_flat_problem(q, k, v, W, xf, None)
    h = tiled_impl._initial_guess(
        x_proj,
        f_t,
        W_be,
        h0_row,
        init_strategy="zero",
    )
    be, _seq, v_dim = x_proj.shape
    n_tiles = math.ceil(S / tile_len)
    summary_A = torch.empty(be, n_tiles, v_dim, v_dim, device=device, dtype=x_proj.dtype)
    summary_b = torch.empty(be, n_tiles, v_dim, device=device, dtype=x_proj.dtype)
    carries = torch.empty(be, n_tiles, v_dim, device=device, dtype=x_proj.dtype)
    delta = torch.empty_like(h)

    ok, log = tiled_impl._try_tilelang_summary(
        h,
        x_proj,
        f_t,
        W_be,
        h0_row,
        summary_A,
        summary_b,
        tile_len,
    )
    if not ok:
        print("stage_breakdown_unavailable=tilelang_summary_failed")
        print(log.rstrip())
        return
    tiled_impl._try_triton_scan(summary_A, summary_b, carries)
    tiled_impl._try_tilelang_apply(h, x_proj, f_t, W_be, h0_row, carries, delta, tile_len)
    torch.cuda.synchronize()

    summary_ms = _time_cuda_kernel(
        lambda: tiled_impl._try_tilelang_summary(
            h,
            x_proj,
            f_t,
            W_be,
            h0_row,
            summary_A,
            summary_b,
            tile_len,
        ),
        warmup=args.stage_warmup,
        repeats=args.stage_repeats,
    )
    scan_ms = _time_cuda_kernel(
        lambda: tiled_impl._try_triton_scan(summary_A, summary_b, carries),
        warmup=args.stage_warmup,
        repeats=args.stage_repeats,
    )
    apply_ms = _time_cuda_kernel(
        lambda: tiled_impl._try_tilelang_apply(
            h,
            x_proj,
            f_t,
            W_be,
            h0_row,
            carries,
            delta,
            tile_len,
        ),
        warmup=args.stage_warmup,
        repeats=args.stage_repeats,
    )
    print(
        "stage_breakdown "
        f"tile_len={tile_len} n_tiles={n_tiles} "
        f"summary_gpu_ms={summary_ms:.4f} "
        f"scan_triton_gpu_ms={scan_ms:.4f} "
        f"apply_gpu_ms={apply_ms:.4f}"
    )


def _run_case(args, *, tile_len: int, dtype: torch.dtype, device: str) -> None:
    q, k, v, W, xf = _make_inputs(
        args.B,
        args.S,
        args.H,
        args.K,
        args.V,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    if args.stage_breakdown:
        _print_stage_breakdown(args, q, k, v, W, xf, tile_len=tile_len, device=device)

    config = TiledTileLangConfig(
        max_its=args.max_its,
        tile_len=tile_len,
        backend=args.backend,
        allow_tilelang_fallback=not args.no_fallback,
    )
    out_tiled, h_tiled, stats = m2rnn_pararnn_tiled_tilelang_forward(
        q,
        k,
        v,
        W,
        xf,
        config=config,
        return_stats=True,
    )
    out_full, h_full = m2rnn_pararnn_forward(
        q,
        k,
        v,
        W,
        xf,
        config=PararnnConfig(max_its=args.max_its, chunk_size=0),
    )

    if device == "cuda":
        torch.cuda.synchronize()

    print(f"tile_len={tile_len}")
    print(f"backend_requested={stats.backend_requested}")
    print(f"backend_used={stats.backend_used}")
    print(f"tilelang_attempted={stats.tilelang_attempted}")
    print(f"tilelang_used={stats.tilelang_used}")
    print(f"tilelang_summary_used={stats.tilelang_summary_used}")
    print(f"triton_scan_used={stats.triton_scan_used}")
    print(f"tilelang_scan_used={stats.tilelang_scan_used}")
    print(f"tilelang_apply_used={stats.tilelang_apply_used}")
    print(f"out_max_diff_vs_full_pararnn={_max_diff(out_tiled, out_full):.6e}")
    print(f"h_max_diff_vs_full_pararnn={_max_diff(h_tiled, h_full):.6e}")
    print(f"be={stats.be} s={stats.s} v_dim={stats.v_dim} n_tiles={stats.n_tiles}")
    print(f"max_tile_jac_elements={stats.max_tile_jac_elements}")
    print(f"torch_materialized_tile_jac_elements={stats.torch_materialized_tile_jac_elements}")
    print(f"full_jac_elements_avoided={stats.full_jac_elements_avoided}")
    print(f"max_tile_jac_bytes_fp32={stats.max_tile_jac_bytes_fp32}")
    print(f"full_jac_bytes_fp32={stats.full_jac_bytes_fp32}")
    print(f"summary_a_elements={stats.summary_a_elements}")
    print(f"summary_b_elements={stats.summary_b_elements}")

    if args.benchmark:
        tiled_ms = _time_forward(
            lambda: m2rnn_pararnn_tiled_tilelang_forward(q, k, v, W, xf, config=config),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        full_ms = _time_forward(
            lambda: m2rnn_pararnn_forward(
                q,
                k,
                v,
                W,
                xf,
                config=PararnnConfig(max_its=args.max_its, chunk_size=0),
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        print(f"tiled_ms={tiled_ms:.3f}")
        print(f"full_pararnn_ms={full_ms:.3f}")

    if stats.tilelang_compile_log:
        print("tilelang_compile_log_begin")
        print(stats.tilelang_compile_log.rstrip())
        print("tilelang_compile_log_end")
    print("")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "torch", "tilelang"], default="auto")
    parser.add_argument("--tile-len", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--sweep-tile-lens", action="store_true")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=64)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--V", type=int, default=16)
    parser.add_argument("--max-its", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--stage-breakdown", action="store_true")
    parser.add_argument("--stage-warmup", type=int, default=10)
    parser.add_argument("--stage-repeats", type=int, default=50)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    print(
        f"device={device} dtype={args.dtype} torch={torch.__version__} "
        f"cuda_available={torch.cuda.is_available()}"
    )
    try:
        import tilelang

        print(f"tilelang={getattr(tilelang, '__version__', 'unknown')} path={tilelang.__file__}")
    except Exception as exc:
        print(f"tilelang_import_failed={type(exc).__name__}: {exc}")

    tile_lens = [16, 32, 64] if args.sweep_tile_lens else [args.tile_len]
    for tile_len in tile_lens:
        _run_case(args, tile_len=tile_len, dtype=dtype, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
