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
import sys
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
    sys.path.insert(0, str(repo_root))
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
        "finite": torch.isfinite(out).all().item(),
        "max_abs": _max_abs(out, ref),
        "rel_l2": rel_l2,
    }


def _record_failure(name: str, exc: BaseException) -> dict[str, Any]:
    return {
        "name": name,
        "status": "fail",
        "finite": None,
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
        warmup=True,
        **kwargs,
    )
    return row


def _try_gemm_capture(
    name: str,
    ref: torch.Tensor,
    gemm_fn: Any,
    *args: Any,
    rel_l2_limit: float | None = None,
    warmup: bool = True,
    **kwargs: Any,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    try:
        if warmup:
            _warmup_out, *_ = gemm_fn(*args, **kwargs)
            torch.cuda.synchronize()
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


def _mxfp8_emit_rowwise_transpose_from_bf16(
    source: torch.Tensor,
    tensor: Any,
    *,
    with_gemm_swizzled_scales: bool = False,
) -> MXFP8Tensor:
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
        fp8_dtype=int(tensor._fp8_dtype),
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
    )
    quantizer = MXFP8Quantizer(tensor._fp8_dtype, rowwise=True, columnwise=False)
    quantizer.internal = True
    quantizer.optimize_for_gemm = bool(with_gemm_swizzled_scales)
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
        with_gemm_swizzled_scales=bool(with_gemm_swizzled_scales),
    )


def _timed_mxfp8_emit_rowwise_transpose_from_bf16(
    source: torch.Tensor,
    tensor: Any,
    *,
    with_gemm_swizzled_scales: bool = False,
) -> tuple[MXFP8Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    emitted = _mxfp8_emit_rowwise_transpose_from_bf16(
        source,
        tensor,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
    )
    end.record()
    torch.cuda.synchronize()
    return emitted, float(start.elapsed_time(end))


def _mxfp8_prepack_existing_columnwise_as_swizzled_rowwise_transpose(tensor: Any) -> MXFP8Tensor:
    """Emit GEMM-ready rowwise-transposed storage from an existing columnwise MXFP8 tensor."""

    if tensor._with_gemm_swizzled_scales:
        raise ValueError("MXFP8 producer prepack requires compact, non-swizzled source scales")
    if tensor._columnwise_data is None or tensor._columnwise_scale_inv is None:
        raise ValueError("MXFP8 tensor is missing columnwise payload/scales")
    if tensor._columnwise_data.dim() < 2:
        raise ValueError("MXFP8 producer prepack requires matrix-like columnwise data")
    if tensor._columnwise_scale_inv.dim() != 2:
        raise ValueError("MXFP8 producer prepack requires 2D compact columnwise scales")

    ext = _load_mxfp8_transpose_emit_ext()
    rowwise_data, rowwise_scale_inv = ext.transpose_payload_swizzle_scale(
        tensor._columnwise_data.reshape(-1, tensor._columnwise_data.shape[-1]),
        tensor._columnwise_scale_inv,
    )
    quantizer = MXFP8Quantizer(tensor._fp8_dtype, rowwise=True, columnwise=False)
    quantizer.internal = True
    quantizer.optimize_for_gemm = True
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
        with_gemm_swizzled_scales=True,
    )


def _timed_mxfp8_prepack_existing_columnwise_as_swizzled_rowwise_transpose(
    tensor: Any,
) -> tuple[MXFP8Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    emitted = _mxfp8_prepack_existing_columnwise_as_swizzled_rowwise_transpose(tensor)
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


def _compare_emit_variants(
    results: list[dict[str, Any]],
    emit_bytes: dict[str, Any],
    refs: dict[str, torch.Tensor],
    name_prefix: str,
    a_bf16: torch.Tensor | None,
    a_q: Any,
    b_bf16: torch.Tensor | None,
    b_q: Any,
    ref_key: str,
    emit_bytes_keys: tuple[str, str] | tuple[str],
) -> None:
    """Compare direct BF16->rowwise-transposed MXFP8 emission with copied TN adapter.

    Handle one operand-pair (e.g. dgrad where only operand A emits from BF16) or two
    operand-pairs (e.g. wgrad where both A and B emit from BF16).

    When *bf16 is ``None`` for an operand the corresponding q-tensor is fed directly
    to the GEMM without any copy / emit transform.
    """

    try:
        # ---- copy adapter transpose ----
        a_t_copy, a_copy_ms = _timed_mxfp8_columnwise_as_rowwise_transpose(a_q)
        emit_bytes[emit_bytes_keys[0]]["copy_adapter_transpose_elapsed_ms"] = a_copy_ms

        if b_bf16 is not None:
            b_t_copy, b_copy_ms = _timed_mxfp8_columnwise_as_rowwise_transpose(b_q)
            emit_bytes[emit_bytes_keys[1]]["copy_adapter_transpose_elapsed_ms"] = b_copy_ms
        else:
            b_t_copy = b_q

        # ---- prepack transpose (swizzled from existing payload) ----
        a_t_producer, a_producer_ms = (
            _timed_mxfp8_prepack_existing_columnwise_as_swizzled_rowwise_transpose(a_q)
        )
        emit_bytes[emit_bytes_keys[0]][
            "existing_payload_swizzled_prepack_elapsed_ms"
        ] = a_producer_ms

        if b_bf16 is not None:
            b_t_producer, b_producer_ms = (
                _timed_mxfp8_prepack_existing_columnwise_as_swizzled_rowwise_transpose(b_q)
            )
            emit_bytes[emit_bytes_keys[1]][
                "existing_payload_swizzled_prepack_elapsed_ms"
            ] = b_producer_ms
        else:
            b_t_producer = b_q

        # ---- emit transpose from BF16 ----
        if a_bf16 is not None:
            a_t_emit, a_emit_ms = _timed_mxfp8_emit_rowwise_transpose_from_bf16(a_bf16, a_q)
            emit_bytes[emit_bytes_keys[0]]["emit_elapsed_ms"] = a_emit_ms
            a_t_emit_swizzled, a_swizzled_emit_ms = _timed_mxfp8_emit_rowwise_transpose_from_bf16(
                a_bf16,
                a_q,
                with_gemm_swizzled_scales=True,
            )
            emit_bytes[emit_bytes_keys[0]]["swizzled_emit_elapsed_ms"] = a_swizzled_emit_ms
        else:
            a_t_emit = a_t_copy
            a_t_emit_swizzled = a_t_copy

        if b_bf16 is not None:
            b_t_emit, b_emit_ms = _timed_mxfp8_emit_rowwise_transpose_from_bf16(b_bf16, b_q)
            emit_bytes[emit_bytes_keys[1]]["emit_elapsed_ms"] = b_emit_ms
            b_t_emit_swizzled, b_swizzled_emit_ms = _timed_mxfp8_emit_rowwise_transpose_from_bf16(
                b_bf16,
                b_q,
                with_gemm_swizzled_scales=True,
            )
            emit_bytes[emit_bytes_keys[1]]["swizzled_emit_elapsed_ms"] = b_swizzled_emit_ms
        else:
            b_t_emit = b_q
            b_t_emit_swizzled = b_q

        # ---- GEMM captures ----
        copy_row, copy_out = _try_gemm_capture(
            f"mxfp8_{name_prefix}_adapter_TN_copy_for_emit_compare",
            refs[ref_key],
            general_gemm,
            a_t_copy,
            b_t_copy,
            out_dtype=torch.bfloat16,
            layout="TN",
            grad=True,
            use_split_accumulator=False,
        )
        emit_row, emit_out = _try_gemm_capture(
            f"mxfp8_{name_prefix}_transpose_emit_TN",
            refs[ref_key],
            general_gemm,
            a_t_emit,
            b_t_emit,
            out_dtype=torch.bfloat16,
            layout="TN",
            grad=True,
            use_split_accumulator=False,
        )
        if copy_out is not None and emit_out is not None:
            emit_row["max_abs_vs_copy_adapter"] = _max_abs(emit_out, copy_out)
            emit_row["rel_l2_vs_copy_adapter"] = _rel_l2(emit_out, copy_out)
        producer_row, producer_out = _try_gemm_capture(
            f"mxfp8_{name_prefix}_existing_payload_swizzled_prepack_TN",
            refs[ref_key],
            general_gemm,
            a_t_producer,
            b_t_producer,
            out_dtype=torch.bfloat16,
            layout="TN",
            grad=True,
            use_split_accumulator=False,
        )
        if copy_out is not None and producer_out is not None:
            producer_row["max_abs_vs_copy_adapter"] = _max_abs(producer_out, copy_out)
            producer_row["rel_l2_vs_copy_adapter"] = _rel_l2(producer_out, copy_out)
        swizzled_emit_row, swizzled_emit_out = _try_gemm_capture(
            f"mxfp8_{name_prefix}_transpose_emit_swizzled_TN",
            refs[ref_key],
            general_gemm,
            a_t_emit_swizzled,
            b_t_emit_swizzled,
            out_dtype=torch.bfloat16,
            layout="TN",
            grad=True,
            use_split_accumulator=False,
        )
        if copy_out is not None and swizzled_emit_out is not None:
            swizzled_emit_row["max_abs_vs_copy_adapter"] = _max_abs(swizzled_emit_out, copy_out)
            swizzled_emit_row["rel_l2_vs_copy_adapter"] = _rel_l2(swizzled_emit_out, copy_out)
        results.append(copy_row)
        results.append(emit_row)
        results.append(producer_row)
        results.append(swizzled_emit_row)
    except Exception as exc:  # pragma: no cover - this is a probe
        results.append(_record_failure(f"mxfp8_{name_prefix}_transpose_emit_TN", exc))


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

    _compare_emit_variants(
        results,
        emit_bytes,
        refs,
        name_prefix="dgrad",
        a_bf16=weight,
        a_q=wq,
        b_bf16=None,
        b_q=dyq,
        ref_key="dgrad",
        emit_bytes_keys=("dgrad",),
    )

    _compare_emit_variants(
        results,
        emit_bytes,
        refs,
        name_prefix="wgrad",
        a_bf16=x,
        a_q=xq,
        b_bf16=dy,
        b_q=dyq,
        ref_key="wgrad",
        emit_bytes_keys=("wgrad_x", "wgrad_dy"),
    )

    return emit_bytes


def _numeric_stats_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for key, after_value in after.items():
        before_value = before.get(key, 0)
        if isinstance(after_value, int) and isinstance(before_value, int):
            delta[key] = after_value - before_value
    return delta


def _time_repeated_gemm(
    gemm_fn: Any,
    *args: Any,
    warmup: int,
    iters: int,
    **kwargs: Any,
) -> tuple[torch.Tensor, float]:
    out: torch.Tensor | None = None
    for _ in range(warmup):
        out, *_ = gemm_fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out, *_ = gemm_fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    if out is None:
        raise RuntimeError("timed GEMM did not produce an output")
    return out, float(start.elapsed_time(end) / max(iters, 1))


def _time_repeated_tensor_call(fn: Any, *, warmup: int, iters: int) -> tuple[torch.Tensor, float]:
    out: torch.Tensor | None = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    if out is None:
        raise RuntimeError("timed call did not produce an output")
    return out, float(start.elapsed_time(end) / max(iters, 1))


def _time_repeated_tensor_call_with_memory(
    fn: Any,
    *,
    warmup: int,
    iters: int,
) -> tuple[Any, float, dict[str, int]]:
    out: Any = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    before = int(torch.cuda.memory_allocated())
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    after = int(torch.cuda.memory_allocated())
    peak = int(torch.cuda.max_memory_allocated())
    if out is None:
        raise RuntimeError("timed call did not produce an output")
    return (
        out,
        float(start.elapsed_time(end) / max(iters, 1)),
        {
            "memory_allocated_before_bytes": before,
            "memory_allocated_after_bytes": after,
            "memory_allocated_delta_bytes": after - before,
            "max_memory_allocated_delta_bytes": peak - before,
        },
    )


def _run_cutlass_direct_microbench(
    wrapped_general_gemm: Any,
    shim_module: Any | None,
    refs: dict[str, torch.Tensor],
    xq: Any,
    wq: Any,
    dyq: Any,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if os.environ.get("CPPMEGA_TE_MXFP8_BWD_BACKEND") != "cutlass_native":
        raise ValueError(
            "cutlass direct microbench requires CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native"
        )

    out_dgrad = torch.empty_like(refs["dgrad"], dtype=torch.bfloat16)
    out_wgrad = torch.empty_like(refs["wgrad"], dtype=torch.bfloat16)
    stats_before = (
        shim_module.cppmega_te_mxfp8_bwd_stats_snapshot()
        if shim_module is not None and hasattr(shim_module, "cppmega_te_mxfp8_bwd_stats_snapshot")
        else {}
    )

    dgrad_out, dgrad_ms = _time_repeated_gemm(
        wrapped_general_gemm,
        wq,
        dyq,
        warmup=warmup,
        iters=iters,
        out=out_dgrad,
        out_dtype=torch.bfloat16,
        layout="NN",
        grad=True,
        use_split_accumulator=False,
    )
    wgrad_out, wgrad_ms = _time_repeated_gemm(
        wrapped_general_gemm,
        xq,
        dyq,
        warmup=warmup,
        iters=iters,
        out=out_wgrad,
        out_dtype=torch.bfloat16,
        layout="NT",
        grad=True,
        use_split_accumulator=False,
    )

    stats_after = (
        shim_module.cppmega_te_mxfp8_bwd_stats_snapshot()
        if shim_module is not None and hasattr(shim_module, "cppmega_te_mxfp8_bwd_stats_snapshot")
        else {}
    )
    return {
        "backend": os.environ.get("CPPMEGA_TE_MXFP8_BWD_BACKEND"),
        "scale_backend": os.environ.get("CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND", "compact"),
        "warmup": warmup,
        "iters": iters,
        "dgrad_elapsed_ms": dgrad_ms,
        "wgrad_elapsed_ms": wgrad_ms,
        "dgrad": _record_success(
            "mxfp8_cutlass_direct_microbench_dgrad",
            dgrad_out,
            refs["dgrad"],
            rel_l2_limit=0.15,
        ),
        "wgrad": _record_success(
            "mxfp8_cutlass_direct_microbench_wgrad",
            wgrad_out,
            refs["wgrad"],
            rel_l2_limit=0.15,
        ),
        "shim_stats_delta": _numeric_stats_delta(stats_before, stats_after),
    }


def _run_cutlass_direct_api_microbench(
    refs: dict[str, torch.Tensor],
    xq: Any,
    wq: Any,
    dyq: Any,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    from cppmega.megatron import cutlass_mxfp8_gemm as cutlass

    out_dgrad_base = torch.empty_like(refs["dgrad"], dtype=torch.bfloat16)
    out_wgrad_base = torch.empty_like(refs["wgrad"], dtype=torch.bfloat16)
    out_dgrad_asym = torch.empty_like(refs["dgrad"], dtype=torch.bfloat16)
    out_wgrad_asym = torch.empty_like(refs["wgrad"], dtype=torch.bfloat16)

    def _dgrad(*, asymmetric: bool, out: torch.Tensor) -> torch.Tensor:
        return cutlass.dgrad_nn_gemm(
            dyq._rowwise_data,
            dyq._rowwise_scale_inv,
            wq._columnwise_data,
            wq._columnwise_scale_inv,
            out=out,
            asymmetric=asymmetric,
        )

    def _wgrad(*, asymmetric: bool, out: torch.Tensor) -> torch.Tensor:
        return cutlass.wgrad_nt_gemm(
            dyq._columnwise_data,
            dyq._columnwise_scale_inv,
            xq._columnwise_data,
            xq._columnwise_scale_inv,
            out=out,
            asymmetric=asymmetric,
        )

    dgrad_base, dgrad_base_ms = _time_repeated_tensor_call(
        lambda: _dgrad(asymmetric=False, out=out_dgrad_base),
        warmup=warmup,
        iters=iters,
    )
    wgrad_base, wgrad_base_ms = _time_repeated_tensor_call(
        lambda: _wgrad(asymmetric=False, out=out_wgrad_base),
        warmup=warmup,
        iters=iters,
    )
    dgrad_asym, dgrad_asym_ms = _time_repeated_tensor_call(
        lambda: _dgrad(asymmetric=True, out=out_dgrad_asym),
        warmup=warmup,
        iters=iters,
    )
    wgrad_asym, wgrad_asym_ms = _time_repeated_tensor_call(
        lambda: _wgrad(asymmetric=True, out=out_wgrad_asym),
        warmup=warmup,
        iters=iters,
    )

    return {
        "backend": "cutlass_native_direct_api",
        "scale_backend": os.environ.get("CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND", "compact"),
        "warmup": warmup,
        "iters": iters,
        "base": {
            "dgrad_elapsed_ms": dgrad_base_ms,
            "wgrad_elapsed_ms": wgrad_base_ms,
            "dgrad": _record_success(
                "mxfp8_cutlass_direct_api_base_dgrad",
                dgrad_base,
                refs["dgrad"],
                rel_l2_limit=0.15,
            ),
            "wgrad": _record_success(
                "mxfp8_cutlass_direct_api_base_wgrad",
                wgrad_base,
                refs["wgrad"],
                rel_l2_limit=0.15,
            ),
        },
        "asymmetric": {
            "dgrad_elapsed_ms": dgrad_asym_ms,
            "wgrad_elapsed_ms": wgrad_asym_ms,
            "dgrad": _record_success(
                "mxfp8_cutlass_direct_api_asymmetric_dgrad",
                dgrad_asym,
                refs["dgrad"],
                rel_l2_limit=0.15,
            ),
            "wgrad": _record_success(
                "mxfp8_cutlass_direct_api_asymmetric_wgrad",
                wgrad_asym,
                refs["wgrad"],
                rel_l2_limit=0.15,
            ),
            "dgrad_max_abs_vs_base": _max_abs(dgrad_asym, dgrad_base),
            "wgrad_max_abs_vs_base": _max_abs(wgrad_asym, wgrad_base),
        },
    }


def _run_mxfp8_adapter_microbench(
    refs: dict[str, torch.Tensor],
    weight_t: Any,
    x_t: Any,
    dy_t: Any,
    dyq: Any,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    out_dgrad = torch.empty_like(refs["dgrad"], dtype=torch.bfloat16)
    out_wgrad = torch.empty_like(refs["wgrad"], dtype=torch.bfloat16)
    dgrad_out, dgrad_ms = _time_repeated_gemm(
        general_gemm,
        weight_t,
        dyq,
        warmup=warmup,
        iters=iters,
        out=out_dgrad,
        out_dtype=torch.bfloat16,
        layout="TN",
        grad=True,
        use_split_accumulator=False,
    )
    wgrad_out, wgrad_ms = _time_repeated_gemm(
        general_gemm,
        x_t,
        dy_t,
        warmup=warmup,
        iters=iters,
        out=out_wgrad,
        out_dtype=torch.bfloat16,
        layout="TN",
        grad=True,
        use_split_accumulator=False,
    )
    return {
        "backend": "te_tn_adapter_pretransposed",
        "warmup": warmup,
        "iters": iters,
        "dgrad_elapsed_ms": dgrad_ms,
        "wgrad_elapsed_ms": wgrad_ms,
        "dgrad": _record_success(
            "mxfp8_adapter_microbench_dgrad",
            dgrad_out,
            refs["dgrad"],
            rel_l2_limit=0.15,
        ),
        "wgrad": _record_success(
            "mxfp8_adapter_microbench_wgrad",
            wgrad_out,
            refs["wgrad"],
            rel_l2_limit=0.15,
        ),
    }


def _run_flashinfer_mxfp8_microbench(
    refs: dict[str, torch.Tensor],
    x: torch.Tensor,
    weight: torch.Tensor,
    xq: Any,
    wq: Any,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Time FlashInfer/CUTLASS MXFP8 GEMM using TE payloads and swizzled TE scales."""

    from flashinfer import SfLayout, mm_mxfp8, mxfp8_quantize
    from cppmega.megatron import flashinfer_mxfp8_gemm as cppmega_flashinfer

    if xq._rowwise_data is None or xq._rowwise_scale_inv is None:
        raise ValueError("xq is missing rowwise MXFP8 payload/scales")
    if wq._rowwise_data is None or wq._rowwise_scale_inv is None:
        raise ValueError("wq is missing rowwise MXFP8 payload/scales")

    m, k = xq._rowwise_data.shape
    n, wk = wq._rowwise_data.shape
    if wk != k:
        raise ValueError(f"weight K mismatch: x K={k}, weight K={wk}")

    # TE stores MXFP8 payload bytes as uint8. PyTorch dtype view is zero-copy and
    # lets FlashInfer accept the same storage as float8_e4m3fn.
    a = xq._rowwise_data.view(torch.float8_e4m3fn)
    b = wq._rowwise_data.view(torch.float8_e4m3fn).t()
    a_descale, scale_swizzle_ms, scale_swizzle_memory = _time_repeated_tensor_call_with_memory(
        lambda: cppmega_flashinfer.swizzle_rowwise_scale(xq._rowwise_scale_inv, m, k),
        warmup=warmup,
        iters=iters,
    )
    b_descale, weight_scale_swizzle_ms, weight_scale_swizzle_memory = (
        _time_repeated_tensor_call_with_memory(
            lambda: cppmega_flashinfer.swizzle_rowwise_scale(wq._rowwise_scale_inv, n, k),
            warmup=warmup,
            iters=iters,
        )
    )

    def _plain_mxfp8(out: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        return mm_mxfp8(
            a,
            b,
            a_descale,
            b_descale,
            out=out,
            out_dtype=out_dtype,
            # FlashInfer's use_8x4_sf_layout=False selects the 1x1 kernel path
            # for consuming already-swizzled 128x4 scales; the scales themselves
            # ARE in 128x4 layout (swizzled above). The flag name is misleading.
            use_8x4_sf_layout=False,
            backend="cutlass",
        )

    out = torch.empty_like(refs["fprop"], dtype=torch.bfloat16)
    te_payload_out, te_payload_gemm_ms, te_payload_gemm_memory = (
        _time_repeated_tensor_call_with_memory(
            lambda: _plain_mxfp8(out, torch.bfloat16),
            warmup=warmup,
            iters=iters,
        )
    )
    out_fp16 = torch.empty_like(refs["fprop"], dtype=torch.float16)
    te_payload_fp16_out, te_payload_fp16_gemm_ms, te_payload_fp16_gemm_memory = (
        _time_repeated_tensor_call_with_memory(
            lambda: _plain_mxfp8(out_fp16, torch.float16),
            warmup=warmup,
            iters=iters,
        )
    )

    (native_a, native_a_descale), native_x_quantize_ms, native_x_quantize_memory = (
        _time_repeated_tensor_call_with_memory(
            lambda: mxfp8_quantize(x, sf_swizzle_layout=SfLayout.layout_128x4, backend="cuda"),
            warmup=warmup,
            iters=iters,
        )
    )
    (native_w, native_w_descale), native_w_quantize_ms, native_w_quantize_memory = (
        _time_repeated_tensor_call_with_memory(
            lambda: mxfp8_quantize(weight, sf_swizzle_layout=SfLayout.layout_128x4, backend="cuda"),
            warmup=warmup,
            iters=iters,
        )
    )
    native_out_buffer = torch.empty_like(refs["fprop"], dtype=torch.bfloat16)
    native_out, native_gemm_ms, native_gemm_memory = _time_repeated_tensor_call_with_memory(
        lambda: mm_mxfp8(
            native_a,
            native_w.t(),
            native_a_descale,
            native_w_descale,
            out=native_out_buffer,
            out_dtype=torch.bfloat16,
            use_8x4_sf_layout=False,
            backend="cutlass",
        ),
        warmup=warmup,
        iters=iters,
    )

    row = _record_success(
        "mxfp8_flashinfer_te_payload_cutlass",
        te_payload_out,
        refs["fprop"],
        rel_l2_limit=0.15,
    )
    row["max_abs_vs_flashinfer_native_quantize"] = _max_abs(te_payload_out, native_out)
    row["rel_l2_vs_flashinfer_native_quantize"] = _rel_l2(te_payload_out, native_out)
    fp16_row = _record_success(
        "mxfp8_flashinfer_te_payload_cutlass_fp16_out",
        te_payload_fp16_out,
        refs["fprop"],
        rel_l2_limit=0.15,
    )

    return {
        "backend": "flashinfer_cutlass_sm120",
        "scale_layout": "SfLayout.layout_128x4",
        "payload_source": "TE rowwise MXFP8 uint8 storage viewed as torch.float8_e4m3fn",
        "epilogue_capabilities": cppmega_flashinfer.epilogue_capability_report(),
        "warmup": warmup,
        "iters": iters,
        "te_payload_gemm_elapsed_ms": te_payload_gemm_ms,
        "te_payload_gemm_memory": te_payload_gemm_memory,
        "te_payload_fp16_out_gemm_elapsed_ms": te_payload_fp16_gemm_ms,
        "te_payload_fp16_out_gemm_memory": te_payload_fp16_gemm_memory,
        "te_x_scale_swizzle_elapsed_ms": scale_swizzle_ms,
        "te_x_scale_swizzle_memory": scale_swizzle_memory,
        "te_w_scale_swizzle_elapsed_ms": weight_scale_swizzle_ms,
        "te_w_scale_swizzle_memory": weight_scale_swizzle_memory,
        "te_scale_swizzle_2x_elapsed_ms": scale_swizzle_ms + weight_scale_swizzle_ms,
        "flashinfer_native_x_quantize_elapsed_ms": native_x_quantize_ms,
        "flashinfer_native_x_quantize_memory": native_x_quantize_memory,
        "flashinfer_native_w_quantize_elapsed_ms": native_w_quantize_ms,
        "flashinfer_native_w_quantize_memory": native_w_quantize_memory,
        "flashinfer_native_quantize_2x_elapsed_ms": native_x_quantize_ms
        + native_w_quantize_ms,
        "flashinfer_native_gemm_elapsed_ms": native_gemm_ms,
        "flashinfer_native_gemm_memory": native_gemm_memory,
        "result": row,
        "fp16_out_result": fp16_row,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    shim_module = None
    wrapped_general_gemm = None
    if args.use_shim:
        os.environ["CPPMEGA_TE_MXFP8_BWD_BACKEND"] = args.mxfp8_bwd_backend
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
    cutlass_direct_microbench: dict[str, Any] | None = None
    cutlass_direct_api_microbench: dict[str, Any] | None = None
    mxfp8_adapter_microbench: dict[str, Any] | None = None
    mxfp8_flashinfer_microbench: dict[str, Any] | None = None

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
            row, _ = _try_gemm_capture(
                "mxfp8_dgrad_shim_NN_to_TN",
                refs["dgrad"],
                wrapped_general_gemm,
                wq,
                dyq,
                out_dtype=torch.bfloat16,
                layout="NN",
                grad=True,
                use_split_accumulator=False,
                warmup=False,
            )
            results.append(row)
            row, _ = _try_gemm_capture(
                "mxfp8_wgrad_shim_NT_to_TN",
                refs["wgrad"],
                wrapped_general_gemm,
                xq,
                dyq,
                out_dtype=torch.bfloat16,
                layout="NT",
                grad=True,
                use_split_accumulator=False,
                warmup=False,
            )
            results.append(row)
            if args.microbench_cutlass_direct:
                try:
                    cutlass_direct_microbench = _run_cutlass_direct_microbench(
                        wrapped_general_gemm,
                        shim_module,
                        refs,
                        xq,
                        wq,
                        dyq,
                        warmup=args.microbench_warmup,
                        iters=args.microbench_iters,
                    )
                except Exception as exc:  # pragma: no cover - this is a probe
                    cutlass_direct_microbench = _record_failure(
                        "mxfp8_cutlass_direct_microbench",
                        exc,
                    )
        if args.microbench_cutlass_direct_asymmetric:
            try:
                cutlass_direct_api_microbench = _run_cutlass_direct_api_microbench(
                    refs,
                    xq,
                    wq,
                    dyq,
                    warmup=args.microbench_warmup,
                    iters=args.microbench_iters,
                )
            except Exception as exc:  # pragma: no cover - this is a probe
                cutlass_direct_api_microbench = _record_failure(
                    "mxfp8_cutlass_direct_api_microbench",
                    exc,
                )
        if args.microbench_adapter:
            try:
                mxfp8_adapter_microbench = _run_mxfp8_adapter_microbench(
                    refs,
                    weight_t,
                    x_t,
                    dy_t,
                    dyq,
                    warmup=args.microbench_warmup,
                    iters=args.microbench_iters,
                )
            except Exception as exc:  # pragma: no cover - this is a probe
                mxfp8_adapter_microbench = _record_failure(
                    "mxfp8_adapter_microbench",
                    exc,
                )
        if args.microbench_flashinfer:
            try:
                mxfp8_flashinfer_microbench = _run_flashinfer_mxfp8_microbench(
                    refs,
                    x,
                    weight,
                    xq,
                    wq,
                    warmup=args.microbench_warmup,
                    iters=args.microbench_iters,
                )
            except Exception as exc:  # pragma: no cover - this is a probe
                mxfp8_flashinfer_microbench = _record_failure(
                    "mxfp8_flashinfer_microbench",
                    exc,
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
        "transpose_emit_backend": os.environ.get(
            "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND", "auto"
        )
        if args.prototype_transpose_emit
        else None,
        "results": results,
    }
    if args.format in ("mxfp8", "both"):
        report["mxfp8_adapter_copy_bytes"] = mxfp8_copy_bytes
    if mxfp8_emit_bytes is not None:
        report["mxfp8_transpose_emit_prototype_bytes"] = mxfp8_emit_bytes
    if cutlass_direct_microbench is not None:
        report["mxfp8_cutlass_direct_microbench"] = cutlass_direct_microbench
    if cutlass_direct_api_microbench is not None:
        report["mxfp8_cutlass_direct_api_microbench"] = cutlass_direct_api_microbench
    if mxfp8_adapter_microbench is not None:
        report["mxfp8_adapter_microbench"] = mxfp8_adapter_microbench
    if mxfp8_flashinfer_microbench is not None:
        report["mxfp8_flashinfer_microbench"] = mxfp8_flashinfer_microbench
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
        "--mxfp8-bwd-backend",
        choices=("te_tn_adapter", "flashinfer_cutlass", "cutlass_native"),
        default="te_tn_adapter",
        help=(
            "MXFP8 backward backend to install before loading the cppmega shim. "
            "This replaces the legacy CPPMEGA_TE_MXFP8_BWD_BACKEND probe override."
        ),
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
            "Emit rowwise-transposed MXFP8 operands directly from BF16 plus TE "
            "columnwise scales and compare against the copy-based TN adapter. "
            "Uses a patched TE op when available, otherwise falls back to the "
            "local probe extension."
        ),
    )
    parser.add_argument(
        "--require-te-transpose-emit",
        action="store_true",
        help=(
            "Require transformer_engine_torch.mxfp8_scaling_transpose_cast for "
            "--prototype-transpose-emit instead of falling back to the probe extension."
        ),
    )
    parser.add_argument(
        "--microbench-cutlass-direct",
        action="store_true",
        help=(
            "When --use-shim and CPPMEGA_TE_MXFP8_BWD_BACKEND=cutlass_native are set, "
            "time repeated direct-loader dgrad/wgrad calls with preallocated outputs "
            "and report shim counter deltas."
        ),
    )
    parser.add_argument(
        "--microbench-cutlass-direct-asymmetric",
        action="store_true",
        help=(
            "Time the low-level CUTLASS direct API with base and split MK/NK "
            "asymmetric entrypoints side by side. This bypasses the shim and "
            "uses explicit Python parameters rather than environment toggles."
        ),
    )
    parser.add_argument(
        "--microbench-adapter",
        action="store_true",
        help="Time repeated pretransposed TE TN adapter calls with preallocated outputs.",
    )
    parser.add_argument(
        "--microbench-flashinfer",
        action="store_true",
        help=(
            "Time FlashInfer/CUTLASS MXFP8 GEMM using TE rowwise payloads plus "
            "a producer scale-swizzle kernel, and compare against FlashInfer's "
            "native swizzled quantizer."
        ),
    )
    parser.add_argument("--microbench-warmup", type=int, default=10)
    parser.add_argument("--microbench-iters", type=int, default=100)
    args = parser.parse_args()

    if args.require_te_transpose_emit:
        args.prototype_transpose_emit = True
        os.environ["CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND"] = "te"
        os.environ["CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_STRICT"] = "1"

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.m % 32 or args.n % 32 or args.k % 32:
        raise SystemExit("m, n, and k must be multiples of 32 for this MXFP8 probe")
    if args.microbench_cutlass_direct and not args.use_shim:
        raise SystemExit("--microbench-cutlass-direct requires --use-shim")
    if args.microbench_warmup < 0 or args.microbench_iters <= 0:
        raise SystemExit("--microbench-warmup must be >= 0 and --microbench-iters must be > 0")

    print(json.dumps(_run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
