"""Prototype quantized Muon momentum storage.

This module intentionally keeps production qMuon behavior explicit.  The
standard optimizer path can feed the returned low-precision scratch into the
existing Newton-Schulz path, while the MXFP8 carrier helpers are opt-in probe
APIs for measuring a non-BF16 Newton-Schulz carrier.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - importable on CPU-only hosts
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


BLOCK_SIZE = 256
QMAX = 127.0
_CUDA_EXT = None
_CUDA_EXT_ERROR: BaseException | None = None


@dataclass(frozen=True)
class QuantizedMuonMomentum:
    """Blockwise symmetric 8-bit Muon momentum state."""

    data: torch.Tensor
    absmax: torch.Tensor
    block_size: int = BLOCK_SIZE

    @property
    def unsigned(self) -> bool:
        return self.data.dtype == torch.uint8


@dataclass(frozen=True)
class QuantizedMuonMxfp8Carrier:
    """Rowwise MXFP8 payload emitted from quantized Muon momentum.

    `rowwise_data` contains raw E4M3 bytes with the same logical 2D shape as the
    source tensor. `rowwise_scale_inv` contains compact E8M0 block-scale bytes,
    one per 32 consecutive values along the last dimension.  The byte stores the
    scale exponent used for dequantization by MXFP8 GEMM kernels.
    """

    rowwise_data: torch.Tensor
    rowwise_scale_inv: torch.Tensor
    rows: int
    cols: int


@dataclass(frozen=True)
class QuantizedMuonNormSegment:
    """A flattened tensor segment that contributes to one normalization group.

    `group_id` is the logical normalization group. For fused QKV tensors, use
    the same `group_id` for every repeated Q, K, or V segment across query
    groups, so the norm is accumulated per slice type rather than over the
    whole fused QKV tensor.
    """

    tensor_index: int
    start: int
    length: int
    group_id: int


@dataclass(frozen=True)
class QuantizedMuonNormPlan:
    """Reusable block-to-normalization-group metadata.

    `block_group_ids` has one int64 entry per quantization block across the
    multi-tensor update list. Reusing this plan avoids rebuilding QKV/shard
    metadata every optimizer step.
    """

    block_group_ids: torch.Tensor
    num_groups: int


def _num_blocks(n_elements: int, block_size: int = BLOCK_SIZE) -> int:
    return (n_elements + block_size - 1) // block_size


def build_quantized_muon_norm_plan(
    states: Sequence[QuantizedMuonMomentum],
    segments: Sequence[QuantizedMuonNormSegment],
    *,
    num_groups: int | None = None,
    require_full_coverage: bool = True,
) -> QuantizedMuonNormPlan:
    """Build reusable grouped-normalization metadata for the CUDA extension.

    The plan is intentionally block-based: each 256-value quantization block
    maps to exactly one normalization group. That makes the fused update kernel
    cheap: after computing a block-local sumsq it performs one atomic add to the
    appropriate group. Segment boundaries therefore must align with quantization
    block boundaries, except for a segment ending at the tensor's final partial
    block.

    For tensor-parallel sharding, build the same logical group order on every
    rank and pass `tp_group` to
    `quantized_muon_momentum_update_multi_and_normalize_groups_`; local shard
    sums are all-reduced before scaling.
    """

    if not states:
        raise ValueError("states must not be empty")
    if not segments:
        raise ValueError("segments must not be empty")
    device = states[0].data.device
    if any(state.data.device != device for state in states):
        raise ValueError("all states must be on the same device")
    if any(state.block_size != BLOCK_SIZE for state in states):
        raise ValueError(f"only block_size={BLOCK_SIZE} is implemented")

    inferred_groups = max(segment.group_id for segment in segments) + 1
    if num_groups is None:
        num_groups = inferred_groups
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    if inferred_groups > num_groups:
        raise ValueError(f"segments refer to group {inferred_groups - 1}, num_groups={num_groups}")

    block_offsets: list[int] = [0]
    for state in states:
        block_offsets.append(block_offsets[-1] + _num_blocks(state.data.numel(), state.block_size))

    block_group_ids = [-1] * block_offsets[-1]
    for segment in segments:
        if segment.tensor_index < 0 or segment.tensor_index >= len(states):
            raise ValueError(f"invalid tensor_index {segment.tensor_index}")
        if segment.group_id < 0 or segment.group_id >= num_groups:
            raise ValueError(f"invalid group_id {segment.group_id}")
        if segment.start < 0 or segment.length <= 0:
            raise ValueError(f"invalid segment start/length: {segment}")

        n = states[segment.tensor_index].data.numel()
        end = segment.start + segment.length
        if end > n:
            raise ValueError(f"segment exceeds tensor {segment.tensor_index} numel={n}: {segment}")
        if segment.start % BLOCK_SIZE != 0:
            raise ValueError(f"segment start must align to block size {BLOCK_SIZE}: {segment}")
        if end % BLOCK_SIZE != 0 and end != n:
            raise ValueError(
                f"segment end must align to block size {BLOCK_SIZE} or tensor end: {segment}"
            )

        first_block = segment.start // BLOCK_SIZE
        last_block = _num_blocks(end, BLOCK_SIZE)
        global_first = block_offsets[segment.tensor_index] + first_block
        global_last = block_offsets[segment.tensor_index] + last_block
        for block_idx in range(global_first, global_last):
            previous = block_group_ids[block_idx]
            if previous != -1 and previous != segment.group_id:
                raise ValueError(
                    f"overlapping normalization segments for global block {block_idx}: "
                    f"{previous} vs {segment.group_id}"
                )
            block_group_ids[block_idx] = segment.group_id

    if require_full_coverage:
        try:
            missing = block_group_ids.index(-1)
        except ValueError:
            missing = -1
        if missing >= 0:
            raise ValueError(f"normalization segments do not cover global block {missing}")

    return QuantizedMuonNormPlan(
        block_group_ids=torch.tensor(block_group_ids, device=device, dtype=torch.long),
        num_groups=num_groups,
    )


def empty_quantized_momentum_like(
    tensor: torch.Tensor,
    *,
    storage_dtype: torch.dtype = torch.int8,
    block_size: int = BLOCK_SIZE,
) -> QuantizedMuonMomentum:
    """Allocate zero-initialized quantized momentum state for ``tensor``."""

    if storage_dtype not in (torch.int8, torch.uint8):
        raise ValueError(f"storage_dtype must be torch.int8 or torch.uint8, got {storage_dtype}")
    if block_size != BLOCK_SIZE:
        raise ValueError(f"only block_size={BLOCK_SIZE} is implemented, got {block_size}")

    data = torch.empty(tensor.shape, device=tensor.device, dtype=storage_dtype)
    if storage_dtype == torch.uint8:
        data.fill_(128)
    else:
        data.zero_()
    absmax = torch.zeros(
        (_num_blocks(tensor.numel(), block_size),),
        device=tensor.device,
        dtype=torch.float32,
    )
    return QuantizedMuonMomentum(data=data, absmax=absmax, block_size=block_size)


def empty_mxfp8_carrier_like(tensor: torch.Tensor) -> QuantizedMuonMxfp8Carrier:
    """Allocate rowwise MXFP8 carrier storage for a 2D Muon tensor."""

    if tensor.ndim != 2:
        raise ValueError(f"MXFP8 Muon carrier requires a 2D tensor, got {tensor.ndim}D")
    rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
    if cols % 32 != 0:
        raise ValueError(f"MXFP8 Muon carrier requires cols divisible by 32, got {cols}")
    rowwise_data = torch.empty((rows, cols), device=tensor.device, dtype=torch.uint8)
    rowwise_scale_inv = torch.empty((rows, cols // 32), device=tensor.device, dtype=torch.uint8)
    return QuantizedMuonMxfp8Carrier(
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        rows=rows,
        cols=cols,
    )


def dequantize_momentum(
    state: QuantizedMuonMomentum,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a quantized momentum state with torch ops."""

    _validate_state(state)
    flat = state.data.reshape(-1)
    if flat.numel() == 0:
        return torch.empty_like(state.data, dtype=dtype)

    q = flat.to(torch.int32)
    if state.unsigned:
        q = q - 128
    block_ids = torch.arange(flat.numel(), device=flat.device, dtype=torch.long) // state.block_size
    values = q.to(torch.float32) * (state.absmax[block_ids] / QMAX)
    return values.reshape_as(state.data).to(dtype)


def dequantize_mxfp8_carrier(
    carrier: QuantizedMuonMxfp8Carrier,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a rowwise MXFP8 carrier with torch ops for tests/probes."""

    _validate_mxfp8_carrier(carrier)
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required to dequantize MXFP8 carriers")
    fp8_values = carrier.rowwise_data.view(torch.float8_e4m3fn).to(torch.float32)
    scale_exp = carrier.rowwise_scale_inv.to(torch.int16)
    scales = torch.where(
        scale_exp == 0,
        torch.zeros_like(scale_exp, dtype=torch.float32),
        torch.pow(
            torch.full_like(scale_exp, 2, dtype=torch.float32),
            scale_exp.to(torch.float32) - 127.0,
        ),
    )
    scale_values = torch.repeat_interleave(scales, 32, dim=1)[:, : carrier.cols]
    return (fp8_values * scale_values).to(dtype)


def quantize_momentum_(
    state: QuantizedMuonMomentum,
    src: torch.Tensor,
    *,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize ``src`` into ``state`` and return bf16 ``scratch`` containing ``src``."""

    return quantized_muon_momentum_update_(state, src, beta=0.0, scratch=scratch)


def quantized_muon_momentum_update_(
    state: QuantizedMuonMomentum,
    grad: torch.Tensor,
    *,
    beta: float,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update quantized Muon momentum and emit bf16 scratch.

    The update is:

    ``m = beta * dequantized_m + (1 - beta) * grad``

    ``state`` is rewritten with nearest-rounded symmetric 8-bit values and
    per-block absmax metadata.  ``scratch`` receives ``m`` in bfloat16.
    """

    _validate_update_inputs(state, grad, beta, scratch)
    if scratch is None:
        scratch = torch.empty_like(grad, dtype=torch.bfloat16)
    if grad.numel() == 0:
        return scratch

    if scratch is grad and _can_use_cuda_ext([state], [grad]):
        quantized_muon_momentum_update_multi_([state], [grad], beta=beta)
        return grad

    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is required for quantized_muon_momentum_update_")
    if not grad.is_cuda:
        raise RuntimeError("quantized_muon_momentum_update_ currently requires CUDA tensors")

    grid = (_num_blocks(grad.numel(), state.block_size),)
    _quantized_muon_momentum_update_kernel[grid](
        state.data,
        state.absmax,
        grad,
        scratch,
        grad.numel(),
        float(beta),
        UNSIGNED=state.unsigned,
        BLOCK=state.block_size,
        num_warps=8,
    )
    return scratch


def quantized_muon_momentum_update_multi_(
    states: Sequence[QuantizedMuonMomentum],
    grads: Sequence[torch.Tensor],
    *,
    beta: float,
) -> Sequence[torch.Tensor]:
    """Update many quantized Muon momentum tensors and overwrite grads in place.

    This is the production-shaped path: persistent momentum is stored as int8 or
    uint8 plus per-block absmax, while each ``grad`` tensor is rewritten with the
    updated momentum in its own dtype.  For bf16 grads, that rewritten grad is the
    low-memory Newton-Schulz input; no separate bf16 scratch tensor is allocated.
    """

    if len(states) != len(grads):
        raise ValueError(f"states and grads length mismatch: {len(states)} != {len(grads)}")
    if not states:
        return grads
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    for state, grad in zip(states, grads):
        _validate_update_inputs(state, grad, beta, scratch=grad)

    unsigned = states[0].unsigned
    if any(state.unsigned != unsigned for state in states):
        raise ValueError("all quantized momentum states in one multi update must share storage dtype")
    grad_dtype = grads[0].dtype
    if any(grad.dtype != grad_dtype for grad in grads):
        raise ValueError("all grad tensors in one multi update must share dtype")

    if _can_use_cuda_ext(states, grads):
        ext = _load_cuda_ext()
        ext.update_multi_(
            [state.data for state in states],
            [state.absmax for state in states],
            list(grads),
            float(beta),
            bool(unsigned),
        )
        return grads

    for state, grad in zip(states, grads):
        quantized_muon_momentum_update_(state, grad, beta=beta, scratch=grad)
    return grads


def quantized_muon_momentum_update_mxfp8_carrier_(
    state: QuantizedMuonMomentum,
    grad: torch.Tensor,
    carrier: QuantizedMuonMxfp8Carrier,
    *,
    beta: float,
    eps: float = 1e-7,
    normalize: bool = True,
) -> torch.Tensor:
    """Update qMuon state and emit normalized rowwise MXFP8 carrier storage.

    This probe path does not write the updated momentum into ``grad`` and does
    not allocate a BF16 scratch tensor.  It uses FP32 per-256-value block sums to
    compute the Frobenius norm, then emits the quantized state as an MXFP8
    carrier scaled by the reciprocal norm.  The returned scalar tensor is the
    inverse norm used for carrier emission.
    """

    _validate_update_inputs(state, grad, beta, scratch=None)
    _validate_mxfp8_carrier(carrier)
    if grad.shape != carrier.rowwise_data.shape:
        raise ValueError(
            f"grad shape {tuple(grad.shape)} must match carrier shape "
            f"{tuple(carrier.rowwise_data.shape)}"
        )
    if grad.ndim != 2:
        raise ValueError(f"MXFP8 Muon carrier update requires a 2D grad, got {grad.ndim}D")
    if not grad.is_cuda:
        raise RuntimeError("MXFP8 Muon carrier update currently requires CUDA tensors")
    if not _can_use_cuda_ext([state], [grad]):
        raise RuntimeError("MXFP8 Muon carrier update requires the CUDA extension")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    ext = _load_cuda_ext()
    partial_sumsq = ext.update_with_sumsq_no_grad_(
        state.data,
        state.absmax,
        grad,
        float(beta),
        bool(state.unsigned),
    )
    if normalize:
        inv_norm = partial_sumsq.sum().clamp_min(eps * eps).rsqrt()
    else:
        inv_norm = torch.ones((), device=grad.device, dtype=torch.float32)
    ext.emit_mxfp8_carrier_from_quantized_(
        state.data,
        state.absmax,
        carrier.rowwise_data,
        carrier.rowwise_scale_inv,
        inv_norm,
        bool(state.unsigned),
    )
    return inv_norm


def quantized_muon_momentum_update_multi_with_sumsq_(
    states: Sequence[QuantizedMuonMomentum],
    grads: Sequence[torch.Tensor],
    *,
    beta: float,
) -> torch.Tensor:
    """Update many quantized momentum tensors and return per-block sumsq.

    ``grads`` are overwritten in place with the updated momentum.  The returned
    vector contains one FP32 sum of squares per 256-value quantization block,
    computed before BF16/FP16 rounding on store.  Summing it gives the Frobenius
    norm input needed by Newton-Schulz normalization without rereading the full
    momentum tensor.
    """

    if len(states) != len(grads):
        raise ValueError(f"states and grads length mismatch: {len(states)} != {len(grads)}")
    if not states:
        device = grads[0].device if grads else torch.device("cpu")
        return torch.empty((0,), device=device, dtype=torch.float32)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    for state, grad in zip(states, grads):
        _validate_update_inputs(state, grad, beta, scratch=grad)

    unsigned = states[0].unsigned
    if any(state.unsigned != unsigned for state in states):
        raise ValueError("all quantized momentum states in one multi update must share storage dtype")
    grad_dtype = grads[0].dtype
    if any(grad.dtype != grad_dtype for grad in grads):
        raise ValueError("all grad tensors in one multi update must share dtype")

    if _can_use_cuda_ext(states, grads):
        ext = _load_cuda_ext()
        return ext.update_multi_with_sumsq_(
            [state.data for state in states],
            [state.absmax for state in states],
            list(grads),
            float(beta),
            bool(unsigned),
        )

    quantized_muon_momentum_update_multi_(states, grads, beta=beta)
    return torch.cat([grad.float().square().reshape(-1).sum().reshape(1) for grad in grads])


def quantized_muon_momentum_update_multi_with_group_sumsq_(
    states: Sequence[QuantizedMuonMomentum],
    grads: Sequence[torch.Tensor],
    norm_plan: QuantizedMuonNormPlan,
    *,
    beta: float,
) -> torch.Tensor:
    """Update quantized momentum and return sumsq per normalization group.

    This is the QKV/sharded variant of
    `quantized_muon_momentum_update_multi_with_sumsq_`. Each 256-value
    quantization block contributes to one logical norm group, so fused QKV can
    accumulate Q, K, and V norms independently while still updating the
    quantized momentum in one pass.
    """

    _validate_multi_update(states, grads, beta)
    _validate_norm_plan(states, norm_plan)

    unsigned = states[0].unsigned
    grad_dtype = grads[0].dtype
    if _can_use_cuda_ext(states, grads):
        ext = _load_cuda_ext()
        return ext.update_multi_with_group_sumsq_(
            [state.data for state in states],
            [state.absmax for state in states],
            list(grads),
            norm_plan.block_group_ids,
            int(norm_plan.num_groups),
            float(beta),
            bool(unsigned),
        )

    quantized_muon_momentum_update_multi_(states, grads, beta=beta)
    flat_group_ids = norm_plan.block_group_ids.cpu().tolist()
    sums = torch.zeros((norm_plan.num_groups,), device=grads[0].device, dtype=torch.float32)
    block_cursor = 0
    for grad in grads:
        flat = grad.float().reshape(-1)
        for start in range(0, flat.numel(), BLOCK_SIZE):
            group_id = flat_group_ids[block_cursor]
            if group_id >= 0:
                sums[group_id] += flat[start : start + BLOCK_SIZE].square().sum()
            block_cursor += 1
    return sums


def quantized_muon_momentum_scale_multi_by_group_(
    grads: Sequence[torch.Tensor],
    norm_plan: QuantizedMuonNormPlan,
    inv_norms: torch.Tensor,
) -> Sequence[torch.Tensor]:
    """Scale grad tensors in place by per-block group inverse norm."""

    if not grads:
        return grads
    _validate_norm_plan_for_grads(grads, norm_plan)
    if inv_norms.shape != (norm_plan.num_groups,):
        raise ValueError(
            f"inv_norms must have shape ({norm_plan.num_groups},), got {tuple(inv_norms.shape)}"
        )
    if inv_norms.device != grads[0].device:
        raise ValueError("inv_norms must be on grad device")
    if inv_norms.dtype != torch.float32:
        raise ValueError(f"inv_norms must be float32, got {inv_norms.dtype}")
    if not inv_norms.is_contiguous():
        inv_norms = inv_norms.contiguous()

    if _CUDA_EXT_ERROR is None and all(grad.is_cuda for grad in grads):
        ext = _load_cuda_ext()
        ext.scale_multi_by_group_(list(grads), norm_plan.block_group_ids, inv_norms)
        return grads

    flat_group_ids = norm_plan.block_group_ids.cpu().tolist()
    block_cursor = 0
    for grad in grads:
        flat = grad.reshape(-1)
        for start in range(0, flat.numel(), BLOCK_SIZE):
            group_id = flat_group_ids[block_cursor]
            if group_id >= 0:
                flat[start : start + BLOCK_SIZE].mul_(inv_norms[group_id].to(grad.dtype))
            block_cursor += 1
    return grads


def quantized_muon_momentum_scale_multi_by_group_from_sumsq_(
    grads: Sequence[torch.Tensor],
    norm_plan: QuantizedMuonNormPlan,
    group_sumsq: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> Sequence[torch.Tensor]:
    """Scale grad tensors in place by `rsqrt(max(group_sumsq, eps**2))`."""

    if not grads:
        return grads
    _validate_norm_plan_for_grads(grads, norm_plan)
    if group_sumsq.shape != (norm_plan.num_groups,):
        raise ValueError(
            f"group_sumsq must have shape ({norm_plan.num_groups},), got {tuple(group_sumsq.shape)}"
        )
    if group_sumsq.device != grads[0].device:
        raise ValueError("group_sumsq must be on grad device")
    if group_sumsq.dtype != torch.float32:
        raise ValueError(f"group_sumsq must be float32, got {group_sumsq.dtype}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if not group_sumsq.is_contiguous():
        group_sumsq = group_sumsq.contiguous()

    if _CUDA_EXT_ERROR is None and all(grad.is_cuda for grad in grads):
        ext = _load_cuda_ext()
        ext.scale_multi_by_group_from_sumsq_(
            list(grads),
            norm_plan.block_group_ids,
            group_sumsq,
            float(eps),
        )
        return grads

    inv_norms = group_sumsq.clamp_min(eps * eps).rsqrt()
    return quantized_muon_momentum_scale_multi_by_group_(grads, norm_plan, inv_norms)


def quantized_muon_momentum_update_multi_and_normalize_groups_(
    states: Sequence[QuantizedMuonMomentum],
    grads: Sequence[torch.Tensor],
    norm_plan: QuantizedMuonNormPlan,
    *,
    beta: float,
    eps: float = 1e-7,
    tp_group: torch.distributed.ProcessGroup | None = None,
    return_inv_norms: bool = True,
) -> torch.Tensor:
    """Update quantized momentum and normalize each group in place.

    This is the production path for fused QKV and sharded Muon tensors:
    local group sums are computed inside the momentum update kernel, optionally
    all-reduced across tensor-parallel ranks, then a grouped scale kernel turns
    the updated BF16 grads into Newton-Schulz input.
    """

    group_sumsq = quantized_muon_momentum_update_multi_with_group_sumsq_(
        states,
        grads,
        norm_plan,
        beta=beta,
    )
    if tp_group is not None:
        torch.distributed.all_reduce(group_sumsq, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    quantized_muon_momentum_scale_multi_by_group_from_sumsq_(
        grads,
        norm_plan,
        group_sumsq,
        eps=eps,
    )
    if return_inv_norms:
        return group_sumsq.clamp_min(eps * eps).rsqrt()
    return group_sumsq


def quantized_muon_momentum_update_multi_and_normalize_(
    states: Sequence[QuantizedMuonMomentum],
    grads: Sequence[torch.Tensor],
    *,
    beta: float,
    eps: float = 1e-7,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Update quantized momentum, then normalize grads in place for Newton-Schulz.

    This fuses Muon's momentum update with the sum-of-squares half of
    Newton-Schulz normalization.  It then scales ``grads`` in place by the
    reciprocal Frobenius norm.  If ``tp_group`` is provided, the scalar sumsq is
    all-reduced before scaling, matching ``distributed_normalize_p2`` semantics.

    Returns the reciprocal norm scalar used for scaling.
    """

    partial_sumsq = quantized_muon_momentum_update_multi_with_sumsq_(
        states,
        grads,
        beta=beta,
    )
    total_sumsq = partial_sumsq.sum()
    if tp_group is not None:
        torch.distributed.all_reduce(total_sumsq, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    inv_norm = total_sumsq.clamp_min(eps * eps).rsqrt()
    if len(grads) == 1:
        grads[0].mul_(inv_norm)
    else:
        torch._foreach_mul_(list(grads), inv_norm)
    return inv_norm


def _validate_state(state: QuantizedMuonMomentum) -> None:
    if state.block_size != BLOCK_SIZE:
        raise ValueError(f"only block_size={BLOCK_SIZE} is implemented, got {state.block_size}")
    if state.data.dtype not in (torch.int8, torch.uint8):
        raise ValueError(f"state.data must be int8 or uint8, got {state.data.dtype}")
    expected_blocks = _num_blocks(state.data.numel(), state.block_size)
    if state.absmax.shape != (expected_blocks,):
        raise ValueError(
            f"state.absmax must have shape ({expected_blocks},), got {tuple(state.absmax.shape)}"
        )
    if state.absmax.dtype != torch.float32:
        raise ValueError(f"state.absmax must be float32, got {state.absmax.dtype}")
    if state.absmax.device != state.data.device:
        raise ValueError("state.data and state.absmax must be on the same device")


def _validate_update_inputs(
    state: QuantizedMuonMomentum,
    grad: torch.Tensor,
    beta: float,
    scratch: torch.Tensor | None,
) -> None:
    _validate_state(state)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if grad.shape != state.data.shape:
        raise ValueError(f"grad shape {tuple(grad.shape)} must match state shape {tuple(state.data.shape)}")
    if grad.device != state.data.device:
        raise ValueError("grad and state must be on the same device")
    if grad.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"grad must be bf16, fp16, or fp32, got {grad.dtype}")
    if not state.data.is_contiguous() or not grad.is_contiguous():
        raise ValueError("state.data and grad must be contiguous for the prototype Triton kernel")
    if scratch is not None:
        if scratch.shape != grad.shape:
            raise ValueError(f"scratch shape {tuple(scratch.shape)} must match grad shape {tuple(grad.shape)}")
        if scratch.device != grad.device:
            raise ValueError("scratch and grad must be on the same device")
        if scratch.dtype != torch.bfloat16:
            raise ValueError(f"scratch must be bfloat16, got {scratch.dtype}")
        if not scratch.is_contiguous():
            raise ValueError("scratch must be contiguous for the prototype Triton kernel")


def _validate_mxfp8_carrier(carrier: QuantizedMuonMxfp8Carrier) -> None:
    if carrier.rows <= 0 or carrier.cols <= 0:
        raise ValueError(f"carrier rows/cols must be positive, got {carrier.rows}x{carrier.cols}")
    if carrier.cols % 32 != 0:
        raise ValueError(f"carrier cols must be divisible by 32, got {carrier.cols}")
    if carrier.rowwise_data.shape != (carrier.rows, carrier.cols):
        raise ValueError(
            "carrier.rowwise_data must have shape "
            f"({carrier.rows}, {carrier.cols}), got {tuple(carrier.rowwise_data.shape)}"
        )
    expected_scale_shape = (carrier.rows, carrier.cols // 32)
    if carrier.rowwise_scale_inv.shape != expected_scale_shape:
        raise ValueError(
            f"carrier.rowwise_scale_inv must have shape {expected_scale_shape}, "
            f"got {tuple(carrier.rowwise_scale_inv.shape)}"
        )
    if carrier.rowwise_data.dtype != torch.uint8:
        raise ValueError(f"carrier.rowwise_data must be uint8, got {carrier.rowwise_data.dtype}")
    if carrier.rowwise_scale_inv.dtype != torch.uint8:
        raise ValueError(
            f"carrier.rowwise_scale_inv must be uint8, got {carrier.rowwise_scale_inv.dtype}"
        )
    if carrier.rowwise_data.device != carrier.rowwise_scale_inv.device:
        raise ValueError("carrier payload and scale tensors must be on the same device")
    if not carrier.rowwise_data.is_contiguous() or not carrier.rowwise_scale_inv.is_contiguous():
        raise ValueError("carrier payload and scale tensors must be contiguous")


def _validate_multi_update(
    states: Sequence[QuantizedMuonMomentum],
    grads: Sequence[torch.Tensor],
    beta: float,
) -> None:
    if len(states) != len(grads):
        raise ValueError(f"states and grads length mismatch: {len(states)} != {len(grads)}")
    if not states:
        raise ValueError("states must not be empty")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    for state, grad in zip(states, grads):
        _validate_update_inputs(state, grad, beta, scratch=grad)

    unsigned = states[0].unsigned
    if any(state.unsigned != unsigned for state in states):
        raise ValueError("all quantized momentum states in one multi update must share storage dtype")
    grad_dtype = grads[0].dtype
    if any(grad.dtype != grad_dtype for grad in grads):
        raise ValueError("all grad tensors in one multi update must share dtype")


def _validate_norm_plan(
    states: Sequence[QuantizedMuonMomentum],
    norm_plan: QuantizedMuonNormPlan,
) -> None:
    if norm_plan.num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {norm_plan.num_groups}")
    if norm_plan.block_group_ids.dtype != torch.long:
        raise ValueError("norm_plan.block_group_ids must be int64")
    if not norm_plan.block_group_ids.is_contiguous():
        raise ValueError("norm_plan.block_group_ids must be contiguous")
    if norm_plan.block_group_ids.device != states[0].data.device:
        raise ValueError("norm_plan.block_group_ids must be on state device")
    expected_blocks = sum(_num_blocks(state.data.numel(), state.block_size) for state in states)
    if norm_plan.block_group_ids.shape != (expected_blocks,):
        raise ValueError(
            f"norm_plan.block_group_ids must have shape ({expected_blocks},), "
            f"got {tuple(norm_plan.block_group_ids.shape)}"
        )


def _validate_norm_plan_for_grads(
    grads: Sequence[torch.Tensor],
    norm_plan: QuantizedMuonNormPlan,
) -> None:
    if norm_plan.num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {norm_plan.num_groups}")
    if norm_plan.block_group_ids.dtype != torch.long:
        raise ValueError("norm_plan.block_group_ids must be int64")
    if not norm_plan.block_group_ids.is_contiguous():
        raise ValueError("norm_plan.block_group_ids must be contiguous")
    if norm_plan.block_group_ids.device != grads[0].device:
        raise ValueError("norm_plan.block_group_ids must be on grad device")
    expected_blocks = sum(_num_blocks(grad.numel(), BLOCK_SIZE) for grad in grads)
    if norm_plan.block_group_ids.shape != (expected_blocks,):
        raise ValueError(
            f"norm_plan.block_group_ids must have shape ({expected_blocks},), "
            f"got {tuple(norm_plan.block_group_ids.shape)}"
        )


def _can_use_cuda_ext(
    states: Sequence[QuantizedMuonMomentum],
    grads: Sequence[torch.Tensor],
) -> bool:
    if not states:
        return False
    if _CUDA_EXT_ERROR is not None:
        return False
    if any(state.block_size != BLOCK_SIZE for state in states):
        return False
    if any(not grad.is_cuda for grad in grads):
        return False
    return True


def _load_cuda_ext():
    global _CUDA_EXT, _CUDA_EXT_ERROR
    if _CUDA_EXT is not None:
        return _CUDA_EXT
    if _CUDA_EXT_ERROR is not None:
        raise RuntimeError("quantized Muon CUDA extension failed to load") from _CUDA_EXT_ERROR
    try:
        from torch.utils.cpp_extension import load

        src_dir = Path(__file__).resolve().parent / "cuda_ext"
        verbose = os.environ.get("CPPMEGA_VERBOSE_EXT_BUILD", "0") == "1"
        _CUDA_EXT = load(
            name="cppmega_quantized_muon_momentum_cuda",
            sources=[
                str(src_dir / "quantized_muon_momentum.cpp"),
                str(src_dir / "quantized_muon_momentum.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=verbose,
        )
        return _CUDA_EXT
    except BaseException as exc:  # pragma: no cover - build failures are host-specific
        _CUDA_EXT_ERROR = exc
        raise


if TRITON_AVAILABLE:

    @triton.jit
    def _nearest_i8(x):
        rounded = tl.where(x >= 0.0, tl.floor(x + 0.5), tl.ceil(x - 0.5))
        rounded = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
        return rounded.to(tl.int32)

    @triton.jit
    def _quantized_muon_momentum_update_kernel(
        q_ptr,
        absmax_ptr,
        grad_ptr,
        scratch_ptr,
        n_elements,
        beta,
        UNSIGNED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements

        q_raw = tl.load(q_ptr + offsets, mask=mask, other=128 if UNSIGNED else 0)
        q_i32 = q_raw.to(tl.int32)
        if UNSIGNED:
            q_i32 = q_i32 - 128

        old_absmax = tl.load(absmax_ptr + pid)
        old_m = q_i32.to(tl.float32) * old_absmax / 127.0
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        updated = beta * old_m + (1.0 - beta) * grad

        tl.store(scratch_ptr + offsets, updated, mask=mask)

        new_absmax = tl.max(tl.abs(tl.where(mask, updated, 0.0)), axis=0)
        inv_scale = tl.where(new_absmax > 0.0, 127.0 / new_absmax, 0.0)
        q_i32_new = _nearest_i8(updated * inv_scale)

        if UNSIGNED:
            q_store = (q_i32_new + 128).to(tl.uint8)
        else:
            q_store = q_i32_new.to(tl.int8)

        tl.store(q_ptr + offsets, q_store, mask=mask)
        tl.store(absmax_ptr + pid, new_absmax)
