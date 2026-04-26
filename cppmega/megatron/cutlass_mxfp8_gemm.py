"""GB10 CUTLASS MXFP8 GEMM integration helpers.

This module is intentionally narrow: it exposes the SM120/SM121 native MXFP8
TN GEMM path behind a Python function that accepts TE-style MXFP8 rowwise
payloads and compact rowwise E8M0 scales.  The scale tensors are repacked to
CUTLASS' native SM1xx scale layout inside the extension.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


_CUDA_EXT: Any | None = None
_CUDA_EXT_ERROR: BaseException | None = None
_CUTLASS_ROOT = Path(os.environ.get("CPPMEGA_CUTLASS_ROOT", "/home/dave/vllm/.deps/cutlass-src"))


def _as_uint8_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise TypeError(f"expected uint8 tensor, got {tensor.dtype}")
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _load_cuda_ext() -> Any:
    global _CUDA_EXT, _CUDA_EXT_ERROR
    if _CUDA_EXT is not None:
        return _CUDA_EXT
    if _CUDA_EXT_ERROR is not None:
        raise RuntimeError("CUTLASS MXFP8 CUDA extension failed to load") from _CUDA_EXT_ERROR
    try:
        from torch.utils.cpp_extension import load

        include_dir = _CUTLASS_ROOT / "include"
        util_include_dir = _CUTLASS_ROOT / "tools" / "util" / "include"
        if not (include_dir / "cutlass" / "cutlass.h").exists():
            raise FileNotFoundError(f"CUTLASS include tree not found under {_CUTLASS_ROOT}")

        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if (major, minor) == (12, 1):
                os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
            elif (major, minor) == (12, 0):
                os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

        src_dir = Path(__file__).resolve().parent / "cuda_ext"
        verbose = os.environ.get("CPPMEGA_VERBOSE_EXT_BUILD", "0") == "1"
        _CUDA_EXT = load(
            name="cppmega_cutlass_mxfp8_gemm_cuda",
            sources=[
                str(src_dir / "cutlass_mxfp8_gemm.cpp"),
                str(src_dir / "cutlass_mxfp8_gemm.cu"),
            ],
            extra_include_paths=[str(include_dir), str(util_include_dir)],
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=[
                "-O2",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ],
            verbose=verbose,
        )
        return _CUDA_EXT
    except BaseException as exc:  # pragma: no cover - build failures are host-specific
        _CUDA_EXT_ERROR = exc
        raise


def is_supported_shape(m: int, n: int, k: int) -> bool:
    """Return whether the current minimal GB10 CUTLASS path accepts this GEMM."""

    return m > 0 and n > 0 and k > 0 and m % 128 == 0 and n % 128 == 0 and k % 128 == 0


def tn_gemm(
    a_rowwise_data: torch.Tensor,
    a_rowwise_scale_inv: torch.Tensor,
    b_rowwise_data: torch.Tensor,
    b_rowwise_scale_inv: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float | None = None,
) -> torch.Tensor:
    """Run native CUTLASS MXFP8 TN GEMM.

    Inputs are logical `A[M, K]` and `B[N, K]`; the returned tensor is
    `A @ B.T` in BF16.  Payloads and scales must be compact MXFP8 uint8
    rowwise tensors.
    """

    if a_rowwise_data.dim() != 2 or b_rowwise_data.dim() != 2:
        raise ValueError("CUTLASS MXFP8 TN GEMM requires 2D payload tensors")
    if a_rowwise_scale_inv.dim() != 2 or b_rowwise_scale_inv.dim() != 2:
        raise ValueError("CUTLASS MXFP8 TN GEMM requires 2D scale tensors")

    m = int(a_rowwise_data.shape[0])
    k = int(a_rowwise_data.shape[1])
    n = int(b_rowwise_data.shape[0])
    if int(b_rowwise_data.shape[1]) != k:
        raise ValueError(f"K mismatch: A is {tuple(a_rowwise_data.shape)}, B is {tuple(b_rowwise_data.shape)}")
    if not is_supported_shape(m, n, k):
        raise ValueError(f"unsupported CUTLASS MXFP8 GB10 shape {m}x{n}x{k}; require multiples of 128")

    if out is None:
        out_arg = torch.empty(0, device=a_rowwise_data.device, dtype=torch.bfloat16)
        use_out = False
    else:
        if out.dtype != torch.bfloat16:
            raise TypeError(f"CUTLASS MXFP8 backend currently requires BF16 out, got {out.dtype}")
        if out.numel() < m * n:
            raise ValueError(f"out is too small for {m}x{n}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        out_arg = out
        use_out = True

    ext = _load_cuda_ext()
    return ext.tn_gemm(
        _as_uint8_contiguous(a_rowwise_data),
        _as_uint8_contiguous(a_rowwise_scale_inv),
        _as_uint8_contiguous(b_rowwise_data),
        _as_uint8_contiguous(b_rowwise_scale_inv),
        m,
        n,
        k,
        out_arg,
        use_out,
        bool(accumulate),
        float(alpha),
        float(1.0 if beta is None and accumulate else (0.0 if beta is None else beta)),
    )
