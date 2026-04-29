#!/usr/bin/env python3
"""Profile cppmega's mixed MXFP8 wgrad paths.

The default direct path computes:

    dy.T[out_n, reduction] @ x[reduction, out_k]

from the original TE compact-columnwise ``dy`` plus saved rowwise ``x.T``.
The ``te_emit_swizzled_stock`` backend is the fast stock-CUTLASS sidecar
control: it materializes full ``dy.T`` rowwise payload/scales plus a full
GEMM-swizzled ``x.T`` scale tensor.

The ``streaming_swizzled_stock`` backend reuses the stock swizzled-scale GEMM,
but only prepares tile-local GEMM-ready scratch: ``dy.T`` payload+scale for one
M tile and ``x.T`` scale for one N tile.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import torch
import transformer_engine  # noqa: F401 - loads TE common libs before TE torch extension
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor import MXFP8Quantizer

from cppmega.megatron import cutlass_mxfp8_gemm as cutlass


def _quantize_mxfp8(tensor: torch.Tensor) -> Any:
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    quantizer.internal = True
    quantizer.optimize_for_gemm = False
    return quantizer(tensor)


def _time_call(fn: Any) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.linalg.vector_norm((a.float() - b.float()).reshape(-1))
    den = torch.linalg.vector_norm(b.float().reshape(-1)).clamp_min(1e-12)
    return float((num / den).item())


def _swizzle_rowwise_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if rows % 128 or cols % 128:
        raise ValueError("rowwise MXFP8 scale swizzle requires rows/cols multiples of 128")
    if scale.dim() != 2 or tuple(scale.shape) != (rows, cols // 32):
        raise ValueError(
            f"scale shape must be {(rows, cols // 32)} for rowwise matrix {(rows, cols)}, "
            f"got {tuple(scale.shape)}"
        )
    return (
        scale.view(rows // 128, 4, 32, cols // 128, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(rows, cols // 32)
    )


def _emit_transposed_dy_swizzled(
    dy: torch.Tensor,
    dy_colwise_scale_inv: torch.Tensor,
) -> Any:
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    quantizer.internal = True
    quantizer.optimize_for_gemm = False
    if not hasattr(quantizer, "quantize_rowwise_transpose"):
        raise RuntimeError("MXFP8Quantizer.quantize_rowwise_transpose is unavailable")
    return quantizer.quantize_rowwise_transpose(
        dy,
        dy_colwise_scale_inv,
        with_gemm_swizzled_scales=True,
    )


def _make_streaming_scratch(
    tile_m: int,
    tile_n: int,
    k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((tile_m, k), device=device, dtype=torch.uint8),
        torch.empty((tile_m, k // 32), device=device, dtype=torch.uint8),
        torch.empty((tile_n, k // 32), device=device, dtype=torch.uint8),
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    reduction = args.reduction
    out_n = args.out_n
    out_k = args.out_k
    if reduction % 128 or out_n % 128 or out_k % 128:
        raise SystemExit("reduction, out-n, and out-k must be multiples of 128")
    if args.tile_m % 128 or args.tile_n % 128:
        raise SystemExit("--tile-m and --tile-n must be multiples of 128")

    device = torch.device("cuda")
    x = (torch.randn((reduction, out_k), device=device, dtype=torch.bfloat16) * 0.01).contiguous()
    dy = (torch.randn((reduction, out_n), device=device, dtype=torch.bfloat16) * 0.01).contiguous()
    torch.cuda.synchronize()

    xq = _quantize_mxfp8(x)
    dyq = _quantize_mxfp8(dy)
    x_t_data = xq._columnwise_data.t().contiguous()
    x_t_scale = xq._columnwise_scale_inv.t().contiguous()
    out = torch.empty((out_n, out_k), device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    cutlass._load_cuda_ext()
    torch.cuda.synchronize()

    base_allocated_bytes = int(torch.cuda.memory_allocated())
    backend_setup_elapsed_ms = 0.0
    backend_extra_tensors: list[tuple[str, torch.Tensor]] = []
    streaming_scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    if args.backend == "te_emit_swizzled_stock":
        setup_start = torch.cuda.Event(enable_timing=True)
        setup_end = torch.cuda.Event(enable_timing=True)
        setup_start.record()
        dy_t = _emit_transposed_dy_swizzled(dy, dyq._columnwise_scale_inv)
        x_t_scale_swizzled = _swizzle_rowwise_scale(x_t_scale, out_k, reduction)
        setup_end.record()
        torch.cuda.synchronize()
        backend_setup_elapsed_ms = float(setup_start.elapsed_time(setup_end))
        backend_extra_tensors = [
            ("dy_t_rowwise_data", dy_t._rowwise_data),
            ("dy_t_gemm_scale_inv", dy_t._rowwise_scale_inv),
            ("x_t_gemm_scale_inv", x_t_scale_swizzled),
        ]
    else:
        dy_t = None
        x_t_scale_swizzled = None

    if args.backend == "streaming_swizzled_stock":
        max_tile_m = min(args.tile_m, out_n)
        max_tile_n = min(args.tile_n, out_k)
        streaming_scratch = _make_streaming_scratch(max_tile_m, max_tile_n, reduction, device)
        backend_extra_tensors = [
            ("a_tile_rowwise_data_scratch", streaming_scratch[0]),
            ("a_tile_gemm_scale_inv_scratch", streaming_scratch[1]),
            ("b_tile_gemm_scale_inv_scratch", streaming_scratch[2]),
        ]

    allocated_after_setup_bytes = int(torch.cuda.memory_allocated())

    def mixed_wgrad() -> torch.Tensor:
        if args.backend == "te_emit_swizzled_stock":
            assert dy_t is not None
            assert x_t_scale_swizzled is not None
            return cutlass.tn_gemm_swizzled_scale(
                dy_t._rowwise_data,
                dy_t._rowwise_scale_inv,
                x_t_data,
                x_t_scale_swizzled,
                out=out,
            )

        if args.backend == "streaming_swizzled_stock":
            assert streaming_scratch is not None
            return cutlass.wgrad_nt_gemm_streaming_swizzled_stock(
                dyq._columnwise_data,
                dyq._columnwise_scale_inv,
                x_t_data,
                x_t_scale,
                out=out,
                tile_m=args.tile_m,
                tile_n=args.tile_n,
                scratch=streaming_scratch,
            )

        a_columnwise_smem = args.backend == "a_col_smem_scalar"
        a_columnwise_smem_b_tma_early = args.backend == "a_col_smem_b_tma_early"
        return cutlass._tn_gemm_compact_direct(
            dyq._columnwise_data,
            dyq._columnwise_scale_inv,
            x_t_data,
            x_t_scale,
            m=out_n,
            n=out_k,
            k=reduction,
            a_source=cutlass._SOURCE_COLUMNWISE_TRANSPOSE,
            a_data_ld=int(dyq._columnwise_data.shape[1]),
            a_scale_ld=int(dyq._columnwise_scale_inv.shape[1]),
            b_source=cutlass._SOURCE_ROWWISE,
            b_data_ld=int(x_t_data.shape[1]),
            b_scale_ld=int(x_t_scale.shape[1]),
            out=out,
            asymmetric=True,
            a_columnwise_smem=a_columnwise_smem,
            a_columnwise_smem_b_tma_early=a_columnwise_smem_b_tma_early,
        )

    parity: dict[str, Any] | None = None
    if args.check_parity:
        mixed_wgrad()
        candidate = out.detach().clone()
        direct = cutlass.wgrad_nt_gemm_x_rowwise_transpose(
            dyq._columnwise_data,
            dyq._columnwise_scale_inv,
            x_t_data,
            x_t_scale,
        )
        torch.cuda.synchronize()
        parity = {
            "exact": bool(torch.equal(candidate, direct)),
            "max_abs": float((candidate.float() - direct.float()).abs().max().item()),
            "rel_l2": _rel_l2(candidate, direct),
        }
        del candidate, direct

    for _ in range(args.warmup):
        mixed_wgrad()
    torch.cuda.synchronize()
    allocated_after_warmup_bytes = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()

    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    times = [_time_call(mixed_wgrad) for _ in range(args.iters)]
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    timed_peak_allocated_bytes = int(torch.cuda.max_memory_allocated())

    finite = bool(torch.isfinite(out).all().item())
    extra = {name: _tensor_nbytes(tensor) for name, tensor in backend_extra_tensors}
    full_sidecar_names = {"dy_t_rowwise_data", "dy_t_gemm_scale_inv", "x_t_gemm_scale_inv"}
    return {
        "timestamp_unix": time.time(),
        "device": torch.cuda.get_device_name(),
        "shape": {
            "dy": [reduction, out_n],
            "x_t": [out_k, reduction],
            "logical_gemm": [out_n, out_k, reduction],
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "backend": args.backend,
        "tile": {"m": args.tile_m, "n": args.tile_n},
        "finite": finite,
        "parity_vs_direct": parity,
        "backend_setup_elapsed_ms": backend_setup_elapsed_ms,
        "memory": {
            "base_allocated_bytes": base_allocated_bytes,
            "allocated_after_backend_setup_bytes": allocated_after_setup_bytes,
            "backend_setup_alloc_delta_bytes": allocated_after_setup_bytes - base_allocated_bytes,
            "allocated_after_warmup_bytes": allocated_after_warmup_bytes,
            "timed_peak_alloc_delta_bytes": timed_peak_allocated_bytes - allocated_after_warmup_bytes,
            "extra_tensor_bytes": extra,
            "extra_tensor_total_bytes": sum(extra.values()),
            "full_materialized_sidecar_bytes": {
                name: value for name, value in extra.items() if name in full_sidecar_names
            },
            "full_materialized_sidecar_total_bytes": sum(
                value for name, value in extra.items() if name in full_sidecar_names
            ),
        },
        "elapsed_ms": {
            "min": min(times),
            "median": statistics.median(times),
            "mean": statistics.fmean(times),
            "max": max(times),
            "samples": times if args.samples else [],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduction", type=int, default=16384)
    parser.add_argument("--out-n", type=int, default=3584)
    parser.add_argument("--out-k", type=int, default=3584)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--tile-m", type=int, default=1024)
    parser.add_argument("--tile-n", type=int, default=2048)
    parser.add_argument(
        "--backend",
        choices=(
            "legacy",
            "a_col_smem_scalar",
            "a_col_smem_b_tma_early",
            "te_emit_swizzled_stock",
            "streaming_swizzled_stock",
        ),
        default="legacy",
    )
    parser.add_argument("--check-parity", action="store_true")
    parser.add_argument("--samples", action="store_true")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Wrap only the timed calls in cudaProfilerStart/Stop for ncu/nsys capture.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be >= 0 and --iters must be > 0")

    print(json.dumps(_run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
