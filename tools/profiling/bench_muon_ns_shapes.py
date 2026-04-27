#!/usr/bin/env python3
"""Benchmark Megatron Muon Newton-Schulz cost for profiler-hot shapes."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
from collections.abc import Iterable

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MEGATRON_ROOT = pathlib.Path("/home/dave/megatron-lm")
DEFAULT_SHAPES = (
    "3584x3584",
    "896x3584",
    "896x896",
    "3584x7168",
)


def _parse_steps(value: str) -> list[int]:
    steps = [int(part) for part in value.split(",") if part]
    if not steps or any(step < 1 for step in steps):
        raise argparse.ArgumentTypeError("--steps must contain positive integers")
    return steps


def _parse_shape(value: str) -> tuple[int, int, int]:
    shape_part, _, count_part = value.partition(":")
    rows_part, sep, cols_part = shape_part.lower().partition("x")
    if sep != "x":
        raise argparse.ArgumentTypeError(f"shape must be ROWSxCOLS[:COUNT], got {value!r}")
    rows = int(rows_part)
    cols = int(cols_part)
    count = int(count_part) if count_part else 1
    if rows <= 0 or cols <= 0 or count <= 0:
        raise argparse.ArgumentTypeError(f"shape values must be positive, got {value!r}")
    return rows, cols, count


def _load_megatron_ns(megatron_root: pathlib.Path):
    for path in (str(REPO_ROOT), str(megatron_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from megatron.core.optimizer.emerging_optimizers import _newton_schulz_lowmem
    except Exception as exc:  # pragma: no cover - exercised by local tool use.
        raise SystemExit(
            "Could not import Megatron _newton_schulz_lowmem. "
            f"Use --megatron-root if it is not at {DEFAULT_MEGATRON_ROOT}."
        ) from exc
    return _newton_schulz_lowmem


def _time_cuda(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.mean(samples), statistics.pstdev(samples) if len(samples) > 1 else 0.0


def _make_input(rows: int, cols: int, *, dtype: torch.dtype, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, cols), device="cuda", dtype=dtype, generator=gen)
    return x / x.float().norm().clamp_min(1e-7).to(dtype)


def _bench_shape(
    ns_fn,
    *,
    rows: int,
    cols: int,
    count: int,
    steps_list: Iterable[int],
    coefficient_type: str,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> list[dict[str, float | int | str]]:
    x = _make_input(rows, cols, dtype=dtype, seed=20260427 + rows + cols)
    results = []
    with torch.no_grad():
        for steps in steps_list:
            def _run() -> None:
                with torch.cuda.nvtx.range(f"muon_ns_{rows}x{cols}_steps{steps}"):
                    y = ns_fn(
                        x,
                        steps=steps,
                        coefficient_type=coefficient_type,
                        already_normalized=True,
                    )
                    # Keep the result live so eager execution cannot drop work.
                    y.detach()

            mean_ms, std_ms = _time_cuda(_run, warmup=warmup, iters=iters)
            results.append(
                {
                    "shape": f"{rows}x{cols}",
                    "count": count,
                    "steps": steps,
                    "ms": mean_ms,
                    "std_ms": std_ms,
                    "estimated_ms": mean_ms * count,
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=pathlib.Path, default=DEFAULT_MEGATRON_ROOT)
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        default=None,
        help="Matrix shape as ROWSxCOLS[:COUNT]. May be repeated.",
    )
    parser.add_argument("--steps", type=_parse_steps, default=_parse_steps("5,3"))
    parser.add_argument("--coefficient-type", default="quintic")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 0 or args.iters < 1:
        raise SystemExit("--warmup must be >= 0 and --iters must be >= 1")

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    shapes = args.shape if args.shape is not None else [_parse_shape(s) for s in DEFAULT_SHAPES]
    ns_fn = _load_megatron_ns(args.megatron_root)

    torch.set_float32_matmul_precision("medium")
    print(
        f"device={torch.cuda.get_device_name(0)} dtype={args.dtype} "
        f"coefficient_type={args.coefficient_type} warmup={args.warmup} iters={args.iters}"
    )
    print("shape,count,steps,ms_per_call,std_ms,estimated_ms_for_count")

    all_results = []
    for rows, cols, count in shapes:
        all_results.extend(
            _bench_shape(
                ns_fn,
                rows=rows,
                cols=cols,
                count=count,
                steps_list=args.steps,
                coefficient_type=args.coefficient_type,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    for row in all_results:
        print(
            f"{row['shape']},{row['count']},{row['steps']},"
            f"{row['ms']:.4f},{row['std_ms']:.4f},{row['estimated_ms']:.4f}"
        )

    max_steps = max(args.steps)
    by_shape = {}
    for row in all_results:
        by_shape.setdefault(row["shape"], {})[row["steps"]] = row
    if len(args.steps) > 1:
        print("shape,baseline_steps,target_steps,per_call_saved_ms,estimated_saved_ms_for_count")
        for shape, rows_by_steps in by_shape.items():
            baseline = rows_by_steps.get(max_steps)
            if baseline is None:
                continue
            for steps in args.steps:
                if steps == max_steps or steps not in rows_by_steps:
                    continue
                target = rows_by_steps[steps]
                saved = float(baseline["ms"]) - float(target["ms"])
                print(
                    f"{shape},{max_steps},{steps},{saved:.4f},"
                    f"{saved * int(target['count']):.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
