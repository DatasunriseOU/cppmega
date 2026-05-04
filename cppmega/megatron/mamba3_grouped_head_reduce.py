"""Guarded grouped-head reduction helpers for Mamba3 MIMO backward.

The production Mamba3 TileLang backward can return expanded ``dq`` / ``dk``
with shape ``[B, S, R, H, N]``. Intermediate grouped-head configs then reduce
that to ``[B, S, R, G, N]``. The default path stays PyTorch; the Triton helper
is opt-in so it can be benchmarked without changing production semantics.
"""

from __future__ import annotations

import os
from typing import Literal

import torch

Backend = Literal["torch", "triton"]

_BACKEND_ENV = "CPPMEGA_MAMBA3_GROUPED_HEAD_REDUCE_BACKEND"


def _validate_pair(
    dq_expanded: torch.Tensor,
    dk_expanded: torch.Tensor,
    groups: int,
) -> tuple[int, int, int, int, int, int]:
    if dq_expanded.shape != dk_expanded.shape:
        raise ValueError(
            "dq_expanded and dk_expanded must have identical shapes, got "
            f"{tuple(dq_expanded.shape)} and {tuple(dk_expanded.shape)}"
        )
    if dq_expanded.ndim != 5:
        raise ValueError(
            f"grouped-head reduction expects [B, S, R, H, N], got {tuple(dq_expanded.shape)}"
        )
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")

    B, S, R, H, N = (int(dim) for dim in dq_expanded.shape)
    if H % groups != 0:
        raise ValueError(f"H must be divisible by groups, got H={H}, groups={groups}")
    return B, S, R, H, N, H // groups


def reduce_grouped_heads_torch(
    dq_expanded: torch.Tensor,
    dk_expanded: torch.Tensor,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce expanded Mamba3 ``dq`` / ``dk`` with the current PyTorch contract."""

    B, S, R, H, N, heads_per_group = _validate_pair(dq_expanded, dk_expanded, groups)
    dq = dq_expanded.view(B, S, R, groups, heads_per_group, N).sum(dim=4)
    dk = dk_expanded.view(B, S, R, groups, heads_per_group, N).sum(dim=4)
    return dq, dk


def reduce_grouped_heads_triton(
    dq_expanded: torch.Tensor,
    dk_expanded: torch.Tensor,
    groups: int,
    *,
    block_m: int = 8,
    block_n: int = 64,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce expanded Mamba3 ``dq`` / ``dk`` using one fused Triton launch."""

    B, S, R, H, N, heads_per_group = _validate_pair(dq_expanded, dk_expanded, groups)
    if not dq_expanded.is_cuda or not dk_expanded.is_cuda:
        raise RuntimeError("Triton grouped-head reduction requires CUDA tensors")
    if not dq_expanded.is_contiguous() or not dk_expanded.is_contiguous():
        raise ValueError("Triton grouped-head reduction requires contiguous inputs")
    if dq_expanded.dtype != dk_expanded.dtype:
        raise ValueError(
            f"dq_expanded and dk_expanded must have the same dtype, got "
            f"{dq_expanded.dtype} and {dk_expanded.dtype}"
        )

    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("triton is required for Triton grouped-head reduction") from exc

    @triton.jit
    def _reduce_pair_kernel(
        dq_ptr,
        dk_ptr,
        dq_out_ptr,
        dk_out_ptr,
        total_m: tl.constexpr,
        S_: tl.constexpr,
        R_: tl.constexpr,
        H_: tl.constexpr,
        G_: tl.constexpr,
        N_: tl.constexpr,
        HPG_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        m = tl.program_id(0) * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        n = tl.program_id(1) * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        m_mask = m < total_m
        n_mask = n < N_

        g = m % G_
        tmp = m // G_
        r = tmp % R_
        tmp = tmp // R_
        s = tmp % S_
        b = tmp // S_

        in_base = (((b[:, None] * S_ + s[:, None]) * R_ + r[:, None]) * H_ + g[:, None] * HPG_) * N_ + n[None, :]
        mask = m_mask[:, None] & n_mask[None, :]
        acc_dq = tl.zeros((BLOCK_M_, BLOCK_N_), tl.float32)
        acc_dk = tl.zeros((BLOCK_M_, BLOCK_N_), tl.float32)
        for h in tl.static_range(0, HPG_):
            in_offsets = in_base + h * N_
            acc_dq += tl.load(dq_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
            acc_dk += tl.load(dk_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)

        out_offsets = m[:, None] * N_ + n[None, :]
        tl.store(dq_out_ptr + out_offsets, acc_dq, mask=mask)
        tl.store(dk_out_ptr + out_offsets, acc_dk, mask=mask)

    dq_out = torch.empty((B, S, R, groups, N), device=dq_expanded.device, dtype=dq_expanded.dtype)
    dk_out = torch.empty_like(dq_out)
    total_m = B * S * R * groups
    block_n_eff = triton.next_power_of_2(min(N, block_n))
    grid = (triton.cdiv(total_m, block_m), triton.cdiv(N, block_n_eff))
    _reduce_pair_kernel[grid](
        dq_expanded,
        dk_expanded,
        dq_out,
        dk_out,
        total_m,
        S,
        R,
        H,
        groups,
        N,
        heads_per_group,
        BLOCK_M_=block_m,
        BLOCK_N_=block_n_eff,
        num_warps=num_warps,
    )
    return dq_out, dk_out


def reduce_grouped_heads(
    dq_expanded: torch.Tensor,
    dk_expanded: torch.Tensor,
    groups: int,
    *,
    backend: Backend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce expanded grouped heads with a default-off backend selector."""

    selected = (backend or os.environ.get(_BACKEND_ENV, "torch")).strip().lower()
    if selected in ("", "0", "off", "none", "torch"):
        return reduce_grouped_heads_torch(dq_expanded, dk_expanded, groups)
    if selected == "triton":
        return reduce_grouped_heads_triton(dq_expanded, dk_expanded, groups)
    raise ValueError(
        f"unsupported grouped-head reduction backend {selected!r}; "
        f"set {_BACKEND_ENV}=torch or triton"
    )


__all__ = [
    "reduce_grouped_heads",
    "reduce_grouped_heads_torch",
    "reduce_grouped_heads_triton",
]
