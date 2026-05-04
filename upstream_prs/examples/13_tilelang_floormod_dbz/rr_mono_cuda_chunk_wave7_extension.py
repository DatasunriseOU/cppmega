"""Minimal CUDA extension wrapper for the Wave 7 row-stream owner kernel."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


_EXTENSION: Any | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive, got {value}")
    return value


def _safe_suffix(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value)


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    this_dir = Path(__file__).resolve().parent
    threads = _env_int("RR_DIAG_THREADS", 256)
    if threads % 32 != 0:
        raise RuntimeError(f"RR_DIAG_THREADS must be a warp multiple, got {threads}")
    suffix = _safe_suffix(os.environ.get("RR_DIAG_CUDA_EXT_SUFFIX", ""))
    name = f"rr_mono_cuda_chunk_wave7_ext_t{threads}"
    if suffix:
        name = f"{name}_{suffix}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    _EXTENSION = load(
        name=name,
        sources=[str(this_dir / "rr_mono_cuda_chunk_wave7_kernel.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "--ptxas-options=-v",
            f"-DRR_DIAG_THREADS={threads}",
        ],
        extra_cflags=["-O3"],
        verbose=bool(int(os.environ.get("RR_DIAG_CUDA_VERBOSE_BUILD", "0"))),
    )
    return _EXTENSION


def stage2_mono_row_stream_chunk_owner_cuda(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    dstates: torch.Tensor,
    da_cs_rev: torch.Tensor,
    segsum: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not dout.is_cuda:
        raise RuntimeError("Wave 7 row-stream owner CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.stage2_mono_row_stream_chunk_owner(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        dstates,
        da_cs_rev,
        segsum,
        D,
        chunk_size,
    )


def stage2_mono_row_stream_chunk_owner_cuda_out(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    dstates: torch.Tensor,
    da_cs_rev: torch.Tensor,
    segsum: torch.Tensor,
    D: torch.Tensor,
    dv_delta: torch.Tensor,
    dmimo_v_delta: torch.Tensor,
    dmimo_v_chunk_delta: torch.Tensor,
    dssda_delta: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("Wave 7 row-stream owner CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_mono_row_stream_chunk_owner_out(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        dstates,
        da_cs_rev,
        segsum,
        D,
        dv_delta,
        dmimo_v_delta,
        dmimo_v_chunk_delta,
        dssda_delta,
        chunk_size,
    )


def stage2_mono_row_stream_chunk_owner_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_mono_row_stream_chunk_owner_metadata(dout)
