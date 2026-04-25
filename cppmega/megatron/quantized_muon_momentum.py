"""Prototype quantized Muon momentum storage.

This module intentionally stops at the momentum-buffer boundary.  It does not
wire into Megatron optimizers; callers can feed the returned bf16 scratch into
the existing low-memory Newton-Schulz path.
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


def _num_blocks(n_elements: int, block_size: int = BLOCK_SIZE) -> int:
    return (n_elements + block_size - 1) // block_size


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
