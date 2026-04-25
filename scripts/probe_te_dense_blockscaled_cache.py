#!/usr/bin/env python3
"""Probe cached dense MXFP8/NVFP4 weight reuse in Transformer Engine.

This is intentionally a low-level timing probe, not a replacement training
kernel.  It exercises the same TE exposure used by high-level Linear modules:

* block-scaled quantizers (MXFP8Quantizer / NVFP4Quantizer)
* cpp_extensions.general_gemm, which calls tex.generic_gemm
* TransformerEngineBaseModule.get_weight_workspace(cache_name="weight")

The important comparison is whether the weight is quantized every call or
prequantized/cached and reused while only activations are quantized per call.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class BenchResult:
    format: str
    mode: str
    ms: float
    rel_l2: float | None = None
    max_abs_err: float | None = None
    note: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=256, help="Activation rows/tokens.")
    parser.add_argument("--n", type=int, default=4096, help="Output features.")
    parser.add_argument("--k", type=int, default=4096, help="Input features.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--formats",
        default="mxfp8,nvfp4",
        help="Comma-separated formats to probe: mxfp8,nvfp4.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=("bf16", "fp16"),
        help="High-precision input/output dtype.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--no-module-workspace",
        action="store_true",
        help="Skip TE Linear.get_weight_workspace cache timing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a table.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print inspected TE API signatures.",
    )
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def _cuda_time_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iters)


def _error_stats(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    diff = out.float() - ref.float()
    denom = torch.linalg.vector_norm(ref.float()).clamp_min(1e-12)
    rel_l2 = torch.linalg.vector_norm(diff) / denom
    max_abs = diff.abs().amax()
    return float(rel_l2.item()), float(max_abs.item())


def _make_quantizers(fmt: str):
    import transformer_engine  # noqa: F401 - loads transformer_engine_torch wheel lib
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor import MXFP8Quantizer, NVFP4Quantizer

    fmt = fmt.lower()
    if fmt == "mxfp8":
        weight_q = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        act_q = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    elif fmt == "nvfp4":
        weight_q = NVFP4Quantizer(rowwise=True, columnwise=True)
        act_q = NVFP4Quantizer(rowwise=True, columnwise=False)
    else:
        raise ValueError(f"unsupported format {fmt!r}")

    weight_q.set_usage(rowwise=True, columnwise=True)
    act_q.set_usage(rowwise=True, columnwise=False)
    return weight_q, act_q


def _bench_format(
    *,
    fmt: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    ref: torch.Tensor,
    warmup: int,
    iters: int,
    include_module_workspace: bool,
) -> list[BenchResult]:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    weight_q, act_q = _make_quantizers(fmt)
    results: list[BenchResult] = []

    def quantize_weight_and_gemm() -> torch.Tensor:
        wq = weight_q(weight)
        xq = act_q(x)
        out, *_ = general_gemm(wq, xq, out_dtype=x.dtype)
        return out

    ms = _cuda_time_ms(quantize_weight_and_gemm, warmup=warmup, iters=iters)
    out = quantize_weight_and_gemm()
    torch.cuda.synchronize()
    rel_l2, max_abs = _error_stats(out, ref)
    results.append(
        BenchResult(
            format=fmt,
            mode="quantize_weight_each_call",
            ms=ms,
            rel_l2=rel_l2,
            max_abs_err=max_abs,
            note="weight + activation quantized inside timed loop",
        )
    )

    cached_weight = weight_q(weight)
    torch.cuda.synchronize()

    def prequantized_weight_gemm() -> torch.Tensor:
        xq = act_q(x)
        out, *_ = general_gemm(cached_weight, xq, out_dtype=x.dtype)
        return out

    ms = _cuda_time_ms(prequantized_weight_gemm, warmup=warmup, iters=iters)
    out = prequantized_weight_gemm()
    torch.cuda.synchronize()
    rel_l2, max_abs = _error_stats(out, ref)
    results.append(
        BenchResult(
            format=fmt,
            mode="prequantized_weight_gemm",
            ms=ms,
            rel_l2=rel_l2,
            max_abs_err=max_abs,
            note="activation quantized each call; weight quantized once",
        )
    )

    if not include_module_workspace:
        return results

    layer = te.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        params_dtype=x.dtype,
        device=x.device,
    )
    with torch.no_grad():
        layer.weight.copy_(weight)

    workspace_q, workspace_act_q = _make_quantizers(fmt)
    workspace_q.internal = True
    workspace = layer.get_weight_workspace(
        tensor=layer.weight,
        quantizer=workspace_q,
        cache_name="weight",
        update_workspace=True,
        workspace_dtype=x.dtype,
    )
    reused_workspace = layer.get_weight_workspace(
        tensor=layer.weight,
        quantizer=workspace_q,
        cache_name="weight",
        update_workspace=False,
        workspace_dtype=x.dtype,
    )
    same_workspace = workspace is reused_workspace

    def te_workspace_update_each_call() -> torch.Tensor:
        wq = layer.get_weight_workspace(
            tensor=layer.weight,
            quantizer=workspace_q,
            cache_name="weight",
            update_workspace=True,
            workspace_dtype=x.dtype,
        )
        xq = workspace_act_q(x)
        out, *_ = general_gemm(wq, xq, out_dtype=x.dtype)
        return out

    ms = _cuda_time_ms(te_workspace_update_each_call, warmup=warmup, iters=iters)
    out = te_workspace_update_each_call()
    torch.cuda.synchronize()
    rel_l2, max_abs = _error_stats(out, ref)
    results.append(
        BenchResult(
            format=fmt,
            mode="te_workspace_update_each_call",
            ms=ms,
            rel_l2=rel_l2,
            max_abs_err=max_abs,
            note="get_weight_workspace(cache_name='weight', update_workspace=True)",
        )
    )

    def te_workspace_cached_gemm() -> torch.Tensor:
        wq = layer.get_weight_workspace(
            tensor=layer.weight,
            quantizer=workspace_q,
            cache_name="weight",
            update_workspace=False,
            workspace_dtype=x.dtype,
        )
        xq = workspace_act_q(x)
        out, *_ = general_gemm(wq, xq, out_dtype=x.dtype)
        return out

    ms = _cuda_time_ms(te_workspace_cached_gemm, warmup=warmup, iters=iters)
    out = te_workspace_cached_gemm()
    torch.cuda.synchronize()
    rel_l2, max_abs = _error_stats(out, ref)
    results.append(
        BenchResult(
            format=fmt,
            mode="te_workspace_cached_gemm",
            ms=ms,
            rel_l2=rel_l2,
            max_abs_err=max_abs,
            note=(
                "get_weight_workspace(cache_name='weight', update_workspace=False); "
                f"same_object={same_workspace}"
            ),
        )
    )
    return results


def _api_summary() -> dict[str, str]:
    import transformer_engine  # noqa: F401
    import transformer_engine_torch as tex  # noqa: F401
    from transformer_engine.pytorch.cpp_extensions import general_gemm
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    from transformer_engine.pytorch.tensor import (
        Float8CurrentScalingQuantizer,
        MXFP8Quantizer,
        NVFP4Quantizer,
    )

    return {
        "Float8CurrentScalingQuantizer": str(inspect.signature(Float8CurrentScalingQuantizer)),
        "MXFP8Quantizer": str(inspect.signature(MXFP8Quantizer)),
        "NVFP4Quantizer": str(inspect.signature(NVFP4Quantizer)),
        "general_gemm": str(inspect.signature(general_gemm)),
        "get_weight_workspace": str(
            inspect.signature(TransformerEngineBaseModule.get_weight_workspace)
        ),
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TE dense block-scaled cache probe")

    import transformer_engine
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    x = torch.randn(args.m, args.k, device=device, dtype=dtype)
    weight = torch.randn(args.n, args.k, device=device, dtype=dtype)
    ref = torch.matmul(x, weight.t())
    torch.cuda.synchronize()

    results: list[BenchResult] = []
    bf16_ms = _cuda_time_ms(
        lambda: torch.matmul(x, weight.t()),
        warmup=args.warmup,
        iters=args.iters,
    )
    results.append(BenchResult(format="bf16", mode="torch_matmul", ms=bf16_ms, note="x @ w.T"))

    formats = [fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()]
    for fmt in formats:
        results.extend(
            _bench_format(
                fmt=fmt,
                x=x,
                weight=weight,
                ref=ref,
                warmup=args.warmup,
                iters=args.iters,
                include_module_workspace=not args.no_module_workspace,
            )
        )

    meta = {
        "torch": torch.__version__,
        "transformer_engine": getattr(transformer_engine, "__version__", "unknown"),
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "dtype": str(dtype),
        "warmup": args.warmup,
        "iters": args.iters,
        "general_gemm_module": getattr(general_gemm, "__module__", ""),
        "env": {
            key: os.environ[key]
            for key in ("NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT", "NVTE_BACKWARD_OVERRIDE")
            if key in os.environ
        },
    }
    if args.verbose:
        meta["api"] = _api_summary()

    if args.json:
        print(json.dumps({"meta": meta, "results": [asdict(r) for r in results]}, indent=2))
        return

    print(
        "TE dense block-scaled cache probe: "
        f"device={meta['device']} sm={meta['capability']} "
        f"torch={meta['torch']} TE={meta['transformer_engine']}"
    )
    print(
        f"shape: x=({args.m}, {args.k}) weight=({args.n}, {args.k}) "
        f"dtype={dtype} warmup={args.warmup} iters={args.iters}"
    )
    if args.verbose:
        print("\nInspected TE API signatures:")
        for name, sig in meta["api"].items():
            print(f"  {name}{sig}")

    print("\n| Format | Mode | ms/iter | rel_l2 | max_abs_err | Note |")
    print("| --- | --- | ---: | ---: | ---: | --- |")
    for result in results:
        rel = "" if result.rel_l2 is None else f"{result.rel_l2:.6f}"
        max_abs = "" if result.max_abs_err is None else f"{result.max_abs_err:.6f}"
        print(
            f"| {result.format} | {result.mode} | {result.ms:.4f} | "
            f"{rel} | {max_abs} | {result.note} |"
        )


if __name__ == "__main__":
    main()
