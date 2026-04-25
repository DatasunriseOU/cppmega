#!/usr/bin/env python3
"""Probe TE block-scaled Linear backward GEMM layouts on GB10.

This is intentionally small and standalone.  It exercises the low-level
Transformer Engine ``general_gemm`` calls used by Linear backward:

* dgrad: ``layout="NN"``, ``grad=True``
* wgrad: ``layout="NT"``, ``grad=True``

For MXFP8 it also prototypes an adapter that retargets compact columnwise
payloads as rowwise transposed operands and calls the supported ``TN`` GEMM.
The adapter only copies uint8 FP8 payloads and uint8 scale payloads; it does
not dequantize to BF16.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.tensor import MXFP8Quantizer, NVFP4Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor


def _load_cppmega_fp8_shim() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    shim_path = repo_root / "scripts" / "cppmega_fp8_shim.py"
    spec = importlib.util.spec_from_file_location("cppmega_fp8_shim_probe", shim_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load shim from {shim_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rel_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float(((out.float() - ref.float()).norm() / ref.float().norm()).item())


def _max_abs(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref.float()).abs().max().item())


def _record_success(name: str, out: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass",
        "shape": list(out.shape),
        "max_abs": _max_abs(out, ref),
        "rel_l2": _rel_l2(out, ref),
    }


def _record_failure(name: str, exc: BaseException) -> dict[str, Any]:
    return {
        "name": name,
        "status": "fail",
        "error_type": type(exc).__name__,
        "error": str(exc).splitlines()[0],
    }


def _try_gemm(
    name: str,
    ref: torch.Tensor,
    gemm_fn: Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        out, *_ = gemm_fn(*args, **kwargs)
        torch.cuda.synchronize()
        return _record_success(name, out, ref)
    except Exception as exc:  # pragma: no cover - this is a probe
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return _record_failure(name, exc)


def _mxfp8_quantize(tensor: torch.Tensor, *, columnwise: bool) -> Any:
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=columnwise)
    quantizer.internal = True
    # Compact scales are required so columnwise scales can be retargeted by
    # transposing the scale tensor. GEMM-swizzled scales are not a simple
    # transpose and produced bad numerics in the prototype.
    quantizer.optimize_for_gemm = False
    return quantizer(tensor)


def _nvfp4_quantize(tensor: torch.Tensor, *, columnwise: bool, is_weight: bool) -> Any:
    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=columnwise,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=is_weight,
        stochastic_rounding=False,
    )
    quantizer.internal = True
    quantizer.optimize_for_gemm = False
    return quantizer(tensor)


def _mxfp8_columnwise_as_rowwise_transpose(tensor: Any) -> MXFP8Tensor:
    """Build a rowwise MXFP8 tensor for ``tensor.T`` from columnwise payloads."""

    if tensor._with_gemm_swizzled_scales:
        raise ValueError("MXFP8 transpose adapter requires compact, non-swizzled scales")
    if tensor._columnwise_data is None or tensor._columnwise_scale_inv is None:
        raise ValueError("MXFP8 tensor is missing columnwise payloads")

    if tensor._columnwise_data.dim() < 2:
        raise ValueError("MXFP8 transpose adapter requires matrix-like data")
    if tensor._columnwise_scale_inv.dim() != 2:
        raise ValueError("MXFP8 transpose adapter requires 2D compact scales")
    data_2d = tensor._columnwise_data.reshape(-1, tensor._columnwise_data.shape[-1])
    rowwise_data = data_2d.t().contiguous()
    rowwise_scale_inv = tensor._columnwise_scale_inv.t().contiguous()
    quantizer = MXFP8Quantizer(tensor._fp8_dtype, rowwise=True, columnwise=False)
    quantizer.internal = True
    quantizer.optimize_for_gemm = False
    return MXFP8Tensor(
        shape=rowwise_data.shape,
        dtype=tensor._dtype,
        fp8_dtype=tensor._fp8_dtype,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=None,
        columnwise_scale_inv=None,
        quantizer=quantizer,
        requires_grad=False,
        with_gemm_swizzled_scales=False,
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    shim_module = None
    wrapped_general_gemm = None
    if args.use_shim:
        os.environ.setdefault("CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER", "1")
        os.environ.setdefault("CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK", "0")
        os.environ.setdefault("CPPMEGA_TE_MXFP8_BWD_DEBUG", "1")
        shim_module = _load_cppmega_fp8_shim()
        from transformer_engine.pytorch.module import linear as te_linear_module

        wrapped_general_gemm = te_linear_module.general_gemm

    torch.manual_seed(args.seed)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(args.m, args.n, device="cuda", dtype=torch.bfloat16)

    refs = {
        "fprop": x @ weight.t(),
        "dgrad": dy @ weight,
        "wgrad": dy.t() @ x,
    }
    results: list[dict[str, Any]] = []

    if args.format in ("mxfp8", "both"):
        xq = _mxfp8_quantize(x, columnwise=True)
        wq = _mxfp8_quantize(weight, columnwise=True)
        dyq = _mxfp8_quantize(dy, columnwise=True)
        results.append(
            _try_gemm(
                "mxfp8_fprop_native_TN",
                refs["fprop"],
                general_gemm,
                wq,
                xq,
                out_dtype=torch.bfloat16,
                layout="TN",
                use_split_accumulator=False,
            )
        )
        results.append(
            _try_gemm(
                "mxfp8_dgrad_native_NN",
                refs["dgrad"],
                general_gemm,
                wq,
                dyq,
                out_dtype=torch.bfloat16,
                layout="NN",
                grad=True,
                use_split_accumulator=False,
            )
        )
        results.append(
            _try_gemm(
                "mxfp8_wgrad_native_NT",
                refs["wgrad"],
                general_gemm,
                xq,
                dyq,
                out_dtype=torch.bfloat16,
                layout="NT",
                grad=True,
                use_split_accumulator=False,
            )
        )

        weight_t = _mxfp8_columnwise_as_rowwise_transpose(wq)
        x_t = _mxfp8_columnwise_as_rowwise_transpose(xq)
        dy_t = _mxfp8_columnwise_as_rowwise_transpose(dyq)
        results.append(
            _try_gemm(
                "mxfp8_dgrad_adapter_TN",
                refs["dgrad"],
                general_gemm,
                weight_t,
                dyq,
                out_dtype=torch.bfloat16,
                layout="TN",
                grad=True,
                use_split_accumulator=False,
            )
        )
        results.append(
            _try_gemm(
                "mxfp8_wgrad_adapter_TN",
                refs["wgrad"],
                general_gemm,
                x_t,
                dy_t,
                out_dtype=torch.bfloat16,
                layout="TN",
                grad=True,
                use_split_accumulator=False,
            )
        )

        if wrapped_general_gemm is not None:
            results.append(
                _try_gemm(
                    "mxfp8_dgrad_shim_NN_to_TN",
                    refs["dgrad"],
                    wrapped_general_gemm,
                    wq,
                    dyq,
                    out_dtype=torch.bfloat16,
                    layout="NN",
                    grad=True,
                    use_split_accumulator=False,
                )
            )
            results.append(
                _try_gemm(
                    "mxfp8_wgrad_shim_NT_to_TN",
                    refs["wgrad"],
                    wrapped_general_gemm,
                    xq,
                    dyq,
                    out_dtype=torch.bfloat16,
                    layout="NT",
                    grad=True,
                    use_split_accumulator=False,
                )
            )

    if args.format in ("nvfp4", "both"):
        xq = _nvfp4_quantize(x, columnwise=True, is_weight=False)
        wq = _nvfp4_quantize(weight, columnwise=True, is_weight=True)
        dyq = _nvfp4_quantize(dy, columnwise=True, is_weight=False)
        results.append(
            _try_gemm(
                "nvfp4_fprop_native_TN_rht_off",
                refs["fprop"],
                general_gemm,
                wq,
                xq,
                out_dtype=torch.bfloat16,
                layout="TN",
                use_split_accumulator=False,
            )
        )
        results.append(
            _try_gemm(
                "nvfp4_dgrad_native_NN_rht_off",
                refs["dgrad"],
                general_gemm,
                wq,
                dyq,
                out_dtype=torch.bfloat16,
                layout="NN",
                grad=True,
                use_split_accumulator=False,
            )
        )
        results.append(
            _try_gemm(
                "nvfp4_wgrad_native_NT_rht_off",
                refs["wgrad"],
                general_gemm,
                xq,
                dyq,
                out_dtype=torch.bfloat16,
                layout="NT",
                grad=True,
                use_split_accumulator=False,
            )
        )

    props = torch.cuda.get_device_properties(0)
    report = {
        "timestamp_unix": time.time(),
        "device": {"name": props.name, "sm": f"{props.major}.{props.minor}"},
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "format": args.format,
        "use_shim": args.use_shim,
        "results": results,
    }
    if shim_module is not None and hasattr(shim_module, "cppmega_te_mxfp8_bwd_stats_snapshot"):
        report["shim_stats"] = shim_module.cppmega_te_mxfp8_bwd_stats_snapshot()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["mxfp8", "nvfp4", "both"], default="both")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--use-shim",
        action="store_true",
        help="Load scripts/cppmega_fp8_shim.py and verify NN/NT calls are routed by the shim.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.m % 32 or args.n % 32 or args.k % 32:
        raise SystemExit("m, n, and k must be multiples of 32 for this MXFP8 probe")

    print(json.dumps(_run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
