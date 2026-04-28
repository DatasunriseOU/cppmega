#!/usr/bin/env python3
"""Compare logical grouped MXFP8 transpose fallback against direct CUTLASS.

This probe is intentionally opt-in.  It builds a small list of per-expert
MXFP8 backward GEMMs, times the current copy-based TN adapter loop, and compares
it against the direct CUTLASS dgrad/wgrad helpers when that backend can be
imported and executed.  If CUDA, TE, or the direct backend is unavailable, the
probe reports ``skip`` as JSON instead of failing the accepted-path validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _emit(report: dict[str, Any]) -> int:
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report.get("status") == "fail" else 0


def _skip(reason: str, **data: Any) -> int:
    return _emit({"status": "skip", "reason": reason, **data})


def _load_direct_backend() -> Any | None:
    try:
        from cppmega.megatron import cutlass_mxfp8_gemm as cutlass
    except Exception as exc:
        raise RuntimeError(f"could not import cutlass_mxfp8_gemm: {type(exc).__name__}: {exc}") from exc
    missing = [
        name
        for name in ("dgrad_nn_gemm", "wgrad_nt_gemm", "is_supported_shape")
        if not hasattr(cutlass, name)
    ]
    if missing:
        raise RuntimeError(f"missing direct backend API: {missing}")
    return cutlass


def _run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"status": "skip", "reason": f"torch unavailable: {type(exc).__name__}: {exc}"}
    if not torch.cuda.is_available():
        return {"status": "skip", "reason": "CUDA unavailable"}

    try:
        cutlass = _load_direct_backend()
    except RuntimeError as exc:
        return {"status": "skip", "reason": str(exc)}
    if not cutlass.is_supported_shape(args.m, args.n, args.k):
        return {
            "status": "skip",
            "reason": "direct backend does not support requested shape",
            "shape": {"m": args.m, "n": args.n, "k": args.k},
        }

    try:
        from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

        from tools.probes.te_blockscaled_backward_probe import (
            _mxfp8_columnwise_as_rowwise_transpose,
            _mxfp8_quantize,
            _record_success,
            _time_repeated_tensor_call_with_memory,
        )
    except Exception as exc:
        return {"status": "skip", "reason": f"TE probe helpers unavailable: {type(exc).__name__}: {exc}"}

    torch.manual_seed(args.seed)
    xs = [
        torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
        for _ in range(args.experts)
    ]
    weights = [
        torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
        for _ in range(args.experts)
    ]
    dys = [
        torch.randn(args.m, args.n, device="cuda", dtype=torch.bfloat16)
        for _ in range(args.experts)
    ]
    refs = {
        "dgrad": torch.stack([dy @ weight for dy, weight in zip(dys, weights)]),
        "wgrad": torch.stack([dy.t() @ x for dy, x in zip(dys, xs)]),
    }
    xqs = [_mxfp8_quantize(x, columnwise=True) for x in xs]
    wqs = [_mxfp8_quantize(weight, columnwise=True) for weight in weights]
    dyqs = [_mxfp8_quantize(dy, columnwise=True) for dy in dys]
    weight_ts = [_mxfp8_columnwise_as_rowwise_transpose(wq) for wq in wqs]
    x_ts = [_mxfp8_columnwise_as_rowwise_transpose(xq) for xq in xqs]
    dy_ts = [_mxfp8_columnwise_as_rowwise_transpose(dyq) for dyq in dyqs]

    adapter_dgrad_out = torch.empty_like(refs["dgrad"], dtype=torch.bfloat16)
    adapter_wgrad_out = torch.empty_like(refs["wgrad"], dtype=torch.bfloat16)
    direct_dgrad_out = torch.empty_like(refs["dgrad"], dtype=torch.bfloat16)
    direct_wgrad_out = torch.empty_like(refs["wgrad"], dtype=torch.bfloat16)

    def _adapter_dgrad() -> torch.Tensor:
        for idx, (weight_t, dyq) in enumerate(zip(weight_ts, dyqs)):
            general_gemm(
                weight_t,
                dyq,
                out=adapter_dgrad_out[idx],
                out_dtype=torch.bfloat16,
                layout="TN",
                grad=True,
                use_split_accumulator=False,
            )
        return adapter_dgrad_out

    def _adapter_wgrad() -> torch.Tensor:
        for idx, (x_t, dy_t) in enumerate(zip(x_ts, dy_ts)):
            general_gemm(
                x_t,
                dy_t,
                out=adapter_wgrad_out[idx],
                out_dtype=torch.bfloat16,
                layout="TN",
                grad=True,
                use_split_accumulator=False,
            )
        return adapter_wgrad_out

    def _direct_dgrad() -> torch.Tensor:
        for idx, (wq, dyq) in enumerate(zip(wqs, dyqs)):
            cutlass.dgrad_nn_gemm(
                dyq._rowwise_data,
                dyq._rowwise_scale_inv,
                wq._columnwise_data,
                wq._columnwise_scale_inv,
                out=direct_dgrad_out[idx],
            )
        return direct_dgrad_out

    def _direct_wgrad() -> torch.Tensor:
        for idx, (xq, dyq) in enumerate(zip(xqs, dyqs)):
            cutlass.wgrad_nt_gemm(
                dyq._columnwise_data,
                dyq._columnwise_scale_inv,
                xq._columnwise_data,
                xq._columnwise_scale_inv,
                out=direct_wgrad_out[idx],
            )
        return direct_wgrad_out

    try:
        adapter_dgrad, adapter_dgrad_ms, adapter_dgrad_mem = _time_repeated_tensor_call_with_memory(
            _adapter_dgrad, warmup=args.warmup, iters=args.iters
        )
        adapter_wgrad, adapter_wgrad_ms, adapter_wgrad_mem = _time_repeated_tensor_call_with_memory(
            _adapter_wgrad, warmup=args.warmup, iters=args.iters
        )
        direct_dgrad, direct_dgrad_ms, direct_dgrad_mem = _time_repeated_tensor_call_with_memory(
            _direct_dgrad, warmup=args.warmup, iters=args.iters
        )
        direct_wgrad, direct_wgrad_ms, direct_wgrad_mem = _time_repeated_tensor_call_with_memory(
            _direct_wgrad, warmup=args.warmup, iters=args.iters
        )
    except Exception as exc:
        return {
            "status": "skip",
            "reason": f"direct/current grouped microprobe unavailable: {type(exc).__name__}: {exc}",
        }

    results = {
        "adapter_dgrad": _record_success(
            "logical_grouped_adapter_dgrad", adapter_dgrad, refs["dgrad"], rel_l2_limit=0.15
        ),
        "adapter_wgrad": _record_success(
            "logical_grouped_adapter_wgrad", adapter_wgrad, refs["wgrad"], rel_l2_limit=0.15
        ),
        "direct_dgrad": _record_success(
            "logical_grouped_direct_dgrad", direct_dgrad, refs["dgrad"], rel_l2_limit=0.15
        ),
        "direct_wgrad": _record_success(
            "logical_grouped_direct_wgrad", direct_wgrad, refs["wgrad"], rel_l2_limit=0.15
        ),
    }
    failed = any(row.get("status") != "pass" for row in results.values())
    copy_bytes_per_group = {
        "dgrad_payload_and_scale_bytes": sum(
            int(wq._columnwise_data.numel() + wq._columnwise_scale_inv.numel()) for wq in wqs
        ),
        "wgrad_payload_and_scale_bytes": sum(
            int(xq._columnwise_data.numel() + xq._columnwise_scale_inv.numel())
            + int(dyq._columnwise_data.numel() + dyq._columnwise_scale_inv.numel())
            for xq, dyq in zip(xqs, dyqs)
        ),
    }
    return {
        "status": "fail" if failed else "pass",
        "backend": "cutlass_native_dense_direct_loop",
        "shape": {"experts": args.experts, "m": args.m, "n": args.n, "k": args.k},
        "warmup": args.warmup,
        "iters": args.iters,
        "copy_bytes_avoided_by_direct": copy_bytes_per_group,
        "elapsed_ms": {
            "adapter_dgrad": adapter_dgrad_ms,
            "adapter_wgrad": adapter_wgrad_ms,
            "direct_dgrad": direct_dgrad_ms,
            "direct_wgrad": direct_wgrad_ms,
        },
        "memory": {
            "adapter_dgrad": adapter_dgrad_mem,
            "adapter_wgrad": adapter_wgrad_mem,
            "direct_dgrad": direct_dgrad_mem,
            "direct_wgrad": direct_wgrad_mem,
        },
        "results": results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.experts <= 0:
        return _skip("--experts must be > 0")
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        return _skip("--m, --n, and --k must be > 0")
    if args.warmup < 0 or args.iters <= 0:
        return _skip("--warmup must be >= 0 and --iters must be > 0")
    return _emit(_run(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
