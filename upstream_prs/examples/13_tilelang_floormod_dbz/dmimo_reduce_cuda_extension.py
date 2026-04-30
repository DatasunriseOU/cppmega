"""CUDA extension wrapper for Wave8 DMIMO_V reduction experiments."""

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
        name="dmimo_reduce_cuda_ext_wave8",
        sources=[str(this_dir / "dmimo_reduce_cuda_kernel.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "--ptxas-options=-v",
        ],
        extra_cflags=["-O3"],
        verbose=bool(int(os.environ.get("DMIMO_REDUCE_CUDA_VERBOSE_BUILD", "0"))),
    )
    return _EXTENSION


def qk_dmimov_atomic_chunk_cuda(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    if not dout.is_cuda:
        raise RuntimeError("DMIMO_V atomic CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.qk_dmimov_atomic_chunk(dout, v, mimo_o, qk_dot, dt, trap, chunk_size)


def qk_dmimov_atomic_chunk_cuda_out(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dmimo_v: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("DMIMO_V atomic CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.qk_dmimov_atomic_chunk_out(dout, v, mimo_o, qk_dot, dt, trap, dmimo_v, chunk_size)


def qk_dmimov_partials_chunk_cuda(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    if not dout.is_cuda:
        raise RuntimeError("DMIMO_V partial CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.qk_dmimov_partials_chunk(dout, v, mimo_o, qk_dot, dt, trap, chunk_size)


def qk_dmimov_partials_chunk_cuda_out(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    partials: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("DMIMO_V partial CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.qk_dmimov_partials_chunk_out(dout, v, mimo_o, qk_dot, dt, trap, partials, chunk_size)


def qk_dmimov_reduce_partials_cuda(partials: torch.Tensor) -> torch.Tensor:
    if not partials.is_cuda:
        raise RuntimeError("DMIMO_V partial reducer requires CUDA tensors")
    ext = _load_extension()
    return ext.qk_dmimov_reduce_partials(partials)


def qk_dmimov_reduce_partials_cuda_out(*, partials: torch.Tensor, dmimo_v: torch.Tensor) -> None:
    if not partials.is_cuda:
        raise RuntimeError("DMIMO_V partial reducer requires CUDA tensors")
    ext = _load_extension()
    ext.qk_dmimov_reduce_partials_out(partials, dmimo_v)


def qk_dmimov_two_pass_cuda(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    if not dout.is_cuda:
        raise RuntimeError("DMIMO_V two-pass CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.qk_dmimov_two_pass(dout, v, mimo_o, qk_dot, dt, trap, chunk_size)


def qk_dmimov_output_owner_cuda(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    if not dout.is_cuda:
        raise RuntimeError("DMIMO_V output-owner CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.qk_dmimov_output_owner(dout, v, mimo_o, qk_dot, dt, trap, chunk_size)


def qk_dmimov_output_owner_cuda_out(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dmimo_v: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("DMIMO_V output-owner CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.qk_dmimov_output_owner_out(dout, v, mimo_o, qk_dot, dt, trap, dmimo_v, chunk_size)


def qk_dmimov_output_owner_rvec_cuda(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    if not dout.is_cuda:
        raise RuntimeError("DMIMO_V output-owner all-R CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.qk_dmimov_output_owner_rvec(dout, v, mimo_o, qk_dot, dt, trap, chunk_size)


def qk_dmimov_output_owner_rvec_cuda_out(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dmimo_v: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("DMIMO_V output-owner all-R CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.qk_dmimov_output_owner_rvec_out(dout, v, mimo_o, qk_dot, dt, trap, dmimo_v, chunk_size)


def qk_dmimov_atomic_chunk_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.qk_dmimov_atomic_chunk_metadata(dout)


def qk_dmimov_partials_chunk_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.qk_dmimov_partials_chunk_metadata(dout)


def qk_dmimov_reduce_partials_cuda_metadata() -> dict[str, Any]:
    ext = _load_extension()
    return ext.qk_dmimov_reduce_partials_metadata()


def qk_dmimov_output_owner_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.qk_dmimov_output_owner_metadata(dout)


def qk_dmimov_output_owner_rvec_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.qk_dmimov_output_owner_rvec_metadata(dout)
