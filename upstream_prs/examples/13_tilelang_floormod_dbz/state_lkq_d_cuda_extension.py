"""CUDA extension wrapper for Wave9 state/LKQ/D bwd_bwd experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


_EXTENSION: Any | None = None


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    this_dir = Path(__file__).resolve().parent
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    _EXTENSION = load(
        name="state_lkq_d_cuda_ext_wave9",
        sources=[str(this_dir / "state_lkq_d_cuda_kernel.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "--ptxas-options=-v",
        ],
        extra_cflags=["-O3"],
        verbose=bool(int(os.environ.get("STATE_LKQ_D_CUDA_VERBOSE_BUILD", "0"))),
    )
    return _EXTENSION


def state_lkq_d_dv_dd_chunk_owner_cuda(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    dstates: torch.Tensor,
    dphi: torch.Tensor,
    v: torch.Tensor,
    mimo_v: torch.Tensor,
    exp_rev: torch.Tensor,
    segsum: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not q.is_cuda:
        raise RuntimeError("state/LKQ/D CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.state_lkq_d_dv_dd_chunk_owner(
        q, k, dstates, dphi, v, mimo_v, exp_rev, segsum, D, chunk_size
    )


def state_lkq_d_dv_dd_chunk_owner_cuda_out(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    dstates: torch.Tensor,
    dphi: torch.Tensor,
    v: torch.Tensor,
    mimo_v: torch.Tensor,
    exp_rev: torch.Tensor,
    segsum: torch.Tensor,
    D: torch.Tensor,
    dv: torch.Tensor,
    dd: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not q.is_cuda:
        raise RuntimeError("state/LKQ/D CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.state_lkq_d_dv_dd_chunk_owner_out(
        q, k, dstates, dphi, v, mimo_v, exp_rev, segsum, D, dv, dd, chunk_size
    )


def state_lkq_d_dv_dd_dmimov_partials_chunk_owner_cuda(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    dstates: torch.Tensor,
    dphi: torch.Tensor,
    v: torch.Tensor,
    mimo_v: torch.Tensor,
    exp_rev: torch.Tensor,
    segsum: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not q.is_cuda:
        raise RuntimeError("state/LKQ/D CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.state_lkq_d_dv_dd_dmimov_partials_chunk_owner(
        q, k, dstates, dphi, v, mimo_v, exp_rev, segsum, D, chunk_size
    )


def state_lkq_d_dv_dd_dmimov_partials_chunk_owner_cuda_out(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    dstates: torch.Tensor,
    dphi: torch.Tensor,
    v: torch.Tensor,
    mimo_v: torch.Tensor,
    exp_rev: torch.Tensor,
    segsum: torch.Tensor,
    D: torch.Tensor,
    dv: torch.Tensor,
    dd: torch.Tensor,
    dmimo_partials: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not q.is_cuda:
        raise RuntimeError("state/LKQ/D CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.state_lkq_d_dv_dd_dmimov_partials_chunk_owner_out(
        q, k, dstates, dphi, v, mimo_v, exp_rev, segsum, D, dv, dd, dmimo_partials, chunk_size
    )


def state_lkq_d_reduce_dmimov_partials_cuda(partials: torch.Tensor) -> torch.Tensor:
    if not partials.is_cuda:
        raise RuntimeError("state/LKQ/D DMIMO_V reducer requires CUDA tensors")
    ext = _load_extension()
    return ext.state_lkq_d_reduce_dmimov_partials(partials)


def state_lkq_d_reduce_dmimov_partials_cuda_out(
    *,
    partials: torch.Tensor,
    dmimo_v: torch.Tensor,
) -> None:
    if not partials.is_cuda:
        raise RuntimeError("state/LKQ/D DMIMO_V reducer requires CUDA tensors")
    ext = _load_extension()
    ext.state_lkq_d_reduce_dmimov_partials_out(partials, dmimo_v)


def state_lkq_d_chunk_owner_cuda_metadata(q: torch.Tensor, *, with_partials: bool) -> dict[str, Any]:
    if not q.is_cuda:
        return {}
    ext = _load_extension()
    return ext.state_lkq_d_chunk_owner_metadata(q, with_partials)


def state_lkq_d_reduce_dmimov_partials_cuda_metadata() -> dict[str, Any]:
    ext = _load_extension()
    return ext.state_lkq_d_reduce_dmimov_partials_metadata()
