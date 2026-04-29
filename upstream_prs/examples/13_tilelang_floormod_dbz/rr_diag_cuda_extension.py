"""CUDA extension wrapper for the standalone R x R diagonal microbench."""

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
        name="rr_diag_cuda_ext_wave4",
        sources=[str(this_dir / "rr_diag_cuda_kernel.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "--ptxas-options=-v",
        ],
        extra_cflags=["-O3"],
        verbose=bool(int(os.environ.get("RR_DIAG_CUDA_VERBOSE_BUILD", "0"))),
    )
    return _EXTENSION


def rr_specialized_cuda(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not inputs["dphi"].is_cuda:
        raise RuntimeError("CUDA extension path requires CUDA tensors")
    ext = _load_extension()
    return ext.rr_diag_forward(
        inputs["dphi"],
        inputs["psiv"],
        inputs["q_pre_rot"],
        inputs["k_pre_rot"],
        inputs["qk_dot"],
        inputs["gamma"],
    )


def rr_cuda_kernel_metadata(inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    if not inputs["dphi"].is_cuda:
        return {}
    ext = _load_extension()
    return ext.rr_diag_kernel_metadata(inputs["dphi"])


def stage2_rr_diag_post_cuda(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dk: torch.Tensor,
    dq: torch.Tensor,
    dgamma_diag: torch.Tensor,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("stage2 CUDA post path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_rr_diag_post(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        dk,
        dq,
        dgamma_diag,
    )


def stage2_rr_diag_post_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_rr_diag_post_metadata(dout)
