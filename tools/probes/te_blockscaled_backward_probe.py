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
from functools import lru_cache
from pathlib import Path
import time
from typing import Any, Literal

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


@lru_cache(maxsize=1)
def _load_mxfp8_transpose_emit_ext() -> Any:
    ext_path = Path(__file__).with_name("te_mxfp8_transpose_emit_ext.py")
    spec = importlib.util.spec_from_file_location("te_mxfp8_transpose_emit_ext_probe", ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load transpose-emission extension from {ext_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rel_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float(((out.float() - ref.float()).norm() / ref.float().norm()).item())


def _max_abs(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref.float()).abs().max().item())


def _record_success(
    name: str,
    out: torch.Tensor,
    ref: torch.Tensor,
    *,
    rel_l2_limit: float | None = None,
) -> dict[str, Any]:
    rel_l2 = _rel_l2(out, ref)
    status = "pass"
    if rel_l2_limit is not None and rel_l2 > rel_l2_limit:
        status = "bad_math"
    return {
        "name": name,
        "status": status,
        "shape": list(out.shape),
        "max_abs": _max_abs(out, ref),
        "rel_l2": rel_l2,
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
    rel_l2_limit: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    row, _ = _try_gemm_capture(
        name,
        ref,
        gemm_fn,
        *args,
        rel_l2_limit=rel_l2_limit,
        **kwargs,
    )
    return row


def _try_gemm_capture(
    name: str,
    ref: torch.Tensor,
    gemm_fn: Any,
    *args: Any,
    rel_l2_limit: float | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out, *_ = gemm_fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        row = _record_success(name, out, ref, rel_l2_limit=rel_l2_limit)
        row["elapsed_ms"] = float(start.elapsed_time(end))
        return row, out
    except Exception as exc:  # pragma: no cover - this is a probe
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return _record_failure(name, exc), None


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


_RetargetMode = Literal["copy", "view", "reshape"]


def _mxfp8_retarget_columnwise_as_rowwise_transpose(
    tensor: Any,
    *,
    data_mode: _RetargetMode = "copy",
    scale_mode: _RetargetMode = "copy",
) -> MXFP8Tensor:
    """Build a rowwise MXFP8 tensor for ``tensor.T`` from columnwise payloads.

    ``copy`` is the accepted adapter path. ``view`` and ``reshape`` are no-copy
    probes that keep the same storage and change only tensor metadata. TE's
    current MXFP8 C++ converter ignores PyTorch strides, so these modes are
    expected to demonstrate wrong math rather than provide a production path.
    """

    if tensor._with_gemm_swizzled_scales:
        raise ValueError("MXFP8 transpose adapter requires compact, non-swizzled scales")
    if tensor._columnwise_data is None or tensor._columnwise_scale_inv is None:
        raise ValueError("MXFP8 tensor is missing columnwise payloads")

    if tensor._columnwise_data.dim() < 2:
        raise ValueError("MXFP8 transpose adapter requires matrix-like data")
    if tensor._columnwise_scale_inv.dim() != 2:
        raise ValueError("MXFP8 transpose adapter requires 2D compact scales")
    data_2d = tensor._columnwise_data.reshape(-1, tensor._columnwise_data.shape[-1])
    scale = tensor._columnwise_scale_inv

    def _retarget_2d(source: torch.Tensor, mode: _RetargetMode) -> torch.Tensor:
        if mode == "copy":
            return source.t().contiguous()
        if mode == "view":
            return source.t()
        if mode == "reshape":
            return source.reshape(source.shape[1], source.shape[0])
        raise ValueError(f"unsupported retarget mode: {mode}")

    rowwise_data = _retarget_2d(data_2d, data_mode)
    rowwise_scale_inv = _retarget_2d(scale, scale_mode)
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


def _mxfp8_columnwise_as_rowwise_transpose(tensor: Any) -> MXFP8Tensor:
    """Build a rowwise MXFP8 tensor for ``tensor.T`` from copied columnwise payloads."""

    return _mxfp8_retarget_columnwise_as_rowwise_transpose(tensor)


def _timed_mxfp8_columnwise_as_rowwise_transpose(tensor: Any) -> tuple[MXFP8Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    transposed = _mxfp8_columnwise_as_rowwise_transpose(tensor)
    end.record()
    torch.cuda.synchronize()
    return transposed, float(start.elapsed_time(end))


def _mxfp8_emit_rowwise_transpose_from_bf16(source: torch.Tensor, tensor: Any) -> MXFP8Tensor:
    """Emit rowwise MXFP8 storage for ``source.T`` using TE columnwise scales."""

    if tensor._with_gemm_swizzled_scales:
        raise ValueError("MXFP8 transpose emission requires compact, non-swizzled scales")
    if tensor._columnwise_scale_inv is None:
        raise ValueError("MXFP8 tensor is missing compact columnwise scales")
    if source.dim() != 2:
        raise ValueError("MXFP8 transpose emission probe requires a 2D BF16 source")

    ext = _load_mxfp8_transpose_emit_ext()
    rowwise_data, rowwise_scale_inv = ext.emit_transpose_from_bf16(
        source,
        tensor._columnwise_scale_inv,
    )
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


def _timed_mxfp8_emit_rowwise_transpose_from_bf16(
    source: torch.Tensor,
    tensor: Any,
) -> tuple[MXFP8Tensor, float]:
    _mxfp8_emit_rowwise_transpose_from_bf16(source, tensor)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    emitted = _mxfp8_emit_rowwise_transpose_from_bf16(source, tensor)
    end.record()
    torch.cuda.synchronize()
    return emitted, float(start.elapsed_time(end))


def _mxfp8_transpose_copy_bytes(tensor: Any) -> int:
    if tensor._columnwise_data is None or tensor._columnwise_scale_inv is None:
        return 0
    return int(tensor._columnwise_data.numel() + tensor._columnwise_scale_inv.numel())


def _mxfp8_transpose_emit_bytes(source: torch.Tensor, tensor: Any) -> dict[str, int]:
    if tensor._columnwise_scale_inv is None:
        return {
            "bf16_source_read_bytes": int(source.numel() * source.element_size()),
            "emitted_payload_bytes": int(source.numel()),
            "scale_transpose_bytes": 0,
            "existing_mxfp8_payload_copy_bytes": 0,
        }
    return {
        "bf16_source_read_bytes": int(source.numel() * source.element_size()),
        "emitted_payload_bytes": int(source.numel()),
        "scale_transpose_bytes": int(tensor._columnwise_scale_inv.numel()),
        "existing_mxfp8_payload_copy_bytes": 0,
    }


def _append_mxfp8_nocopy_probe_results(
    results: list[dict[str, Any]],
    refs: dict[str, torch.Tensor],
    xq: Any,
    wq: Any,
    dyq: Any,
    *,
    rel_l2_limit: float,
) -> None:
    """Exercise no-copy retargeting variants that should fail accuracy today."""

    variants: tuple[tuple[str, _RetargetMode, _RetargetMode], ...] = (
        ("copy_data_copy_scale", "copy", "copy"),
        ("view_data_copy_scale", "view", "copy"),
        ("copy_data_view_scale", "copy", "view"),
        ("view_data_view_scale", "view", "view"),
        ("reshape_data_reshape_scale", "reshape", "reshape"),
    )
    for variant_name, data_mode, scale_mode in variants:
        weight_t = _mxfp8_retarget_columnwise_as_rowwise_transpose(
            wq, data_mode=data_mode, scale_mode=scale_mode
        )
        results.append(
            _try_gemm(
                f"mxfp8_dgrad_adapter_TN_{variant_name}",
                refs["dgrad"],
                general_gemm,
                weight_t,
                dyq,
                out_dtype=torch.bfloat16,
                layout="TN",
                grad=True,
                use_split_accumulator=False,
                rel_l2_limit=rel_l2_limit,
            )
        )

        x_t = _mxfp8_retarget_columnwise_as_rowwise_transpose(
            xq, data_mode=data_mode, scale_mode=scale_mode
        )
        dy_t = _mxfp8_retarget_columnwise_as_rowwise_transpose(
            dyq, data_mode=data_mode, scale_mode=scale_mode
        )
        results.append(
            _try_gemm(
                f"mxfp8_wgrad_adapter_TN_{variant_name}",
                refs["wgrad"],
                general_gemm,
                x_t,
                dy_t,
                out_dtype=torch.bfloat16,
                layout="TN",
                grad=True,
                use_split_accumulator=False,
                rel_l2_limit=rel_l2_limit,
            )
        )


def _append_mxfp8_transpose_emit_probe_results(
    results: list[dict[str, Any]],
    refs: dict[str, torch.Tensor],
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    xq: Any,
    wq: Any,
    dyq: Any,
) -> dict[str, Any]:
    """Compare direct BF16->rowwise-transposed MXFP8 emission with copied TN adapter."""

    emit_bytes = {
        "dgrad": _mxfp8_transpose_emit_bytes(weight, wq),
        "wgrad_x": _mxfp8_transpose_emit_bytes(x, xq),
        "wgrad_dy": _mxfp8_transpose_emit_bytes(dy, dyq),
    }

    try:
        weight_t_copy, copy_ms = _timed_mxfp8_columnwise_as_rowwise_transpose(wq)
        emit_bytes["dgrad"]["copy_adapter_transpose_elapsed_ms"] = copy_ms
        weight_t_emit, emit_ms = _timed_mxfp8_emit_rowwise_transpose_from_bf16(weight, wq)
        emit_bytes["dgrad"]["emit_elapsed_ms"] = emit_ms
        copy_row, copy_out = _try_gemm_capture(
            "mxfp8_dgrad_adapter_TN_copy_for_emit_compare",
            refs["dgrad"],
            general_gemm,
            weight_t_copy,
            dyq,
            out_dtype=torch.bfloat16,
            layout="TN",
            grad=True,
            use_split_accumulator=False,
        )
        emit_row, emit_out = _try_gemm_capture(
            "mxfp8_dgrad_transpose_emit_TN",
            refs["dgrad"],
            general_gemm,
            weight_t_emit,
            dyq,
            out_dtype=torch.bfloat16,
            layout="TN",
            grad=True,
            use_split_accumulator=False,
        )
        if copy_out is not None and emit_out is not None:
            emit_row["max_abs_vs_copy_adapter"] = _max_abs(emit_out, copy_out)
            emit_row["rel_l2_vs_copy_adapter"] = _rel_l2(emit_out, copy_out)
        results.append(copy_row)
        results.append(emit_row)
    except Exception as exc:  # pragma: no cover - this is a probe
        results.append(_record_failure("mxfp8_dgrad_transpose_emit_TN", exc))

    try:
        x_t_copy, x_copy_ms = _timed_mxfp8_columnwise_as_rowwise_transpose(xq)
        dy_t_copy, dy_copy_ms = _timed_mxfp8_columnwise_as_rowwise_transpose(dyq)
        emit_bytes["wgrad_x"]["copy_adapter_transpose_elapsed_ms"] = x_copy_ms
        emit_bytes["wgrad_dy"]["copy_adapter_transpose_elapsed_ms"] = dy_copy_ms
        x_t_emit, x_emit_ms = _timed_mxfp8_emit_rowwise_transpose_from_bf16(x, xq)
        dy_t_emit, dy_emit_ms = _timed_mxfp8_emit_rowwise_transpose_from_bf16(dy, dyq)
        emit_bytes["wgrad_x"]["emit_elapsed_ms"] = x_emit_ms
        emit_bytes["wgrad_dy"]["emit_elapsed_ms"] = dy_emit_ms
        copy_row, copy_out = _try_gemm_capture(
            "mxfp8_wgrad_adapter_TN_copy_for_emit_compare",
            refs["wgrad"],
            general_gemm,
            x_t_copy,
            dy_t_copy,
            out_dtype=torch.bfloat16,
            layout="TN",
            grad=True,
            use_split_accumulator=False,
        )
        emit_row, emit_out = _try_gemm_capture(
            "mxfp8_wgrad_transpose_emit_TN",
            refs["wgrad"],
            general_gemm,
            x_t_emit,
            dy_t_emit,
            out_dtype=torch.bfloat16,
            layout="TN",
            grad=True,
            use_split_accumulator=False,
        )
        if copy_out is not None and emit_out is not None:
            emit_row["max_abs_vs_copy_adapter"] = _max_abs(emit_out, copy_out)
            emit_row["rel_l2_vs_copy_adapter"] = _rel_l2(emit_out, copy_out)
        results.append(copy_row)
        results.append(emit_row)
    except Exception as exc:  # pragma: no cover - this is a probe
        results.append(_record_failure("mxfp8_wgrad_transpose_emit_TN", exc))

    return emit_bytes


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
    mxfp8_emit_bytes: dict[str, Any] | None = None

    if args.format in ("mxfp8", "both"):
        xq = _mxfp8_quantize(x, columnwise=True)
        wq = _mxfp8_quantize(weight, columnwise=True)
        dyq = _mxfp8_quantize(dy, columnwise=True)
        mxfp8_copy_bytes = {
            "dgrad_payload_and_scale_bytes": _mxfp8_transpose_copy_bytes(wq),
            "wgrad_payload_and_scale_bytes": (
                _mxfp8_transpose_copy_bytes(xq) + _mxfp8_transpose_copy_bytes(dyq)
            ),
        }
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

        if args.probe_nocopy:
            _append_mxfp8_nocopy_probe_results(
                results,
                refs,
                xq,
                wq,
                dyq,
                rel_l2_limit=args.nocopy_rel_l2_limit,
            )
        if args.prototype_transpose_emit:
            mxfp8_emit_bytes = _append_mxfp8_transpose_emit_probe_results(
                results,
                refs,
                x,
                weight,
                dy,
                xq,
                wq,
                dyq,
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
    if args.format in ("mxfp8", "both"):
        report["mxfp8_adapter_copy_bytes"] = mxfp8_copy_bytes
    if mxfp8_emit_bytes is not None:
        report["mxfp8_transpose_emit_prototype_bytes"] = mxfp8_emit_bytes
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
    parser.add_argument(
        "--probe-nocopy",
        action="store_true",
        help=(
            "Also run experimental no-copy MXFP8 transpose retargeting variants. "
            "These are expected to report bad_math on current TE."
        ),
    )
    parser.add_argument(
        "--nocopy-rel-l2-limit",
        type=float,
        default=0.15,
        help="Relative L2 threshold used to mark no-copy retargeting variants as bad_math.",
    )
    parser.add_argument(
        "--prototype-transpose-emit",
        action="store_true",
        help=(
            "Build a local CUDA extension that emits rowwise-transposed MXFP8 operands "
            "directly from BF16 plus TE columnwise scales and compares against the "
            "copy-based TN adapter. This is a probe, not a TE runtime patch."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.m % 32 or args.n % 32 or args.k % 32:
        raise SystemExit("m, n, and k must be multiples of 32 for this MXFP8 probe")

    print(json.dumps(_run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
