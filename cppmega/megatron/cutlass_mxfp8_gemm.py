"""GB10 CUTLASS MXFP8 GEMM integration helpers.

This module exposes the SM120/SM121 native MXFP8 TN GEMM path behind a Python
function that accepts TE-style MXFP8 rowwise payloads and compact rowwise E8M0
scales.  The CUTLASS extension uses a cppmega mainloop fork that reads compact
scales directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


_CUDA_EXT: Any | None = None
_CUDA_EXT_ERROR: BaseException | None = None
_CUTLASS_ROOT = Path(os.environ.get("CPPMEGA_CUTLASS_ROOT", "/home/dave/vllm/.deps/cutlass-src"))

_SOURCE_ROWWISE = 0
_SOURCE_COLUMNWISE_TRANSPOSE = 1


def _as_uint8_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise TypeError(f"expected uint8 tensor, got {tensor.dtype}")
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _resolve_beta(beta: float | None, accumulate: bool) -> float:
    """Validate ``beta`` against ``accumulate`` and return the C++ beta value.

    The C++ extension uses beta as passed.  For overwrite calls, reject nonzero
    beta instead of silently dropping it; for accumulate calls, default to 1.0.
    """

    if accumulate:
        return 1.0 if beta is None else float(beta)
    if beta is None or float(beta) == 0.0:
        return 0.0
    raise ValueError(
        "beta is meaningful only when accumulate=True; "
        f"got beta={beta!r} with accumulate=False"
    )


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

    beta_f = _resolve_beta(beta, bool(accumulate))

    ext = _load_cuda_ext()
    return ext.tn_gemm_compact_scale(
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
        beta_f,
    )


def _tn_gemm_compact_direct(
    a_data: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b_data: torch.Tensor,
    b_scale_inv: torch.Tensor,
    *,
    m: int,
    n: int,
    k: int,
    a_source: int,
    a_data_ld: int,
    a_scale_ld: int,
    b_source: int,
    b_data_ld: int,
    b_scale_ld: int,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float | None = None,
    asymmetric: bool = True,
    a_columnwise_smem: bool = False,
) -> torch.Tensor:
    if not is_supported_shape(m, n, k):
        raise ValueError(f"unsupported CUTLASS MXFP8 GB10 shape {m}x{n}x{k}; require multiples of 128")
    if a_source not in (_SOURCE_ROWWISE, _SOURCE_COLUMNWISE_TRANSPOSE):
        raise ValueError(f"unsupported A source {a_source}")
    if b_source not in (_SOURCE_ROWWISE, _SOURCE_COLUMNWISE_TRANSPOSE):
        raise ValueError(f"unsupported B source {b_source}")

    if out is None:
        out_arg = torch.empty(0, device=a_data.device, dtype=torch.bfloat16)
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

    beta_f = _resolve_beta(beta, bool(accumulate))

    ext = _load_cuda_ext()
    if a_columnwise_smem:
        if not asymmetric:
            raise ValueError("A-columnwise-smem direct path requires asymmetric=True")
        if a_source != _SOURCE_COLUMNWISE_TRANSPOSE:
            raise ValueError("A-columnwise-smem direct path requires A columnwise-transpose source")
        entrypoint = ext.tn_gemm_compact_direct_a_col_smem_asym
    else:
        entrypoint = ext.tn_gemm_compact_direct_asym if asymmetric else ext.tn_gemm_compact_direct
    return entrypoint(
        _as_uint8_contiguous(a_data),
        _as_uint8_contiguous(a_scale_inv),
        _as_uint8_contiguous(b_data),
        _as_uint8_contiguous(b_scale_inv),
        m,
        n,
        k,
        int(a_source),
        int(a_data_ld),
        int(a_scale_ld),
        int(b_source),
        int(b_data_ld),
        int(b_scale_ld),
        out_arg,
        use_out,
        bool(accumulate),
        float(alpha),
        beta_f,
    )


def dgrad_nn_gemm(
    dy_rowwise_data: torch.Tensor,
    dy_rowwise_scale_inv: torch.Tensor,
    weight_colwise_data: torch.Tensor,
    weight_colwise_scale_inv: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float | None = None,
    asymmetric: bool = True,
) -> torch.Tensor:
    """Run dgrad NN as TN without materializing ``weight.T``.

    Computes ``dy[M, N] @ weight[N, K]``.  ``dy`` must be TE compact rowwise;
    ``weight`` must be the original TE compact columnwise payload/scales.
    """

    if dy_rowwise_data.dim() != 2 or dy_rowwise_scale_inv.dim() != 2:
        raise ValueError("dy rowwise payload/scales must be 2D")
    if weight_colwise_data.dim() != 2 or weight_colwise_scale_inv.dim() != 2:
        raise ValueError("weight columnwise payload/scales must be 2D")
    m = int(dy_rowwise_data.shape[0])
    k = int(dy_rowwise_data.shape[1])
    if int(weight_colwise_data.shape[0]) != k:
        raise ValueError(
            "dgrad reduction mismatch: "
            f"dy is {tuple(dy_rowwise_data.shape)}, weight is {tuple(weight_colwise_data.shape)}"
        )
    n = int(weight_colwise_data.shape[1])
    return _tn_gemm_compact_direct(
        dy_rowwise_data,
        dy_rowwise_scale_inv,
        weight_colwise_data,
        weight_colwise_scale_inv,
        m=m,
        n=n,
        k=k,
        a_source=_SOURCE_ROWWISE,
        a_data_ld=int(dy_rowwise_data.shape[1]),
        a_scale_ld=int(dy_rowwise_scale_inv.shape[1]),
        b_source=_SOURCE_COLUMNWISE_TRANSPOSE,
        b_data_ld=int(weight_colwise_data.shape[1]),
        b_scale_ld=int(weight_colwise_scale_inv.shape[1]),
        out=out,
        accumulate=accumulate,
        alpha=alpha,
        beta=beta,
        asymmetric=asymmetric,
    )


def tn_gemm_direct_rowwise(
    a_rowwise_data: torch.Tensor,
    a_rowwise_scale_inv: torch.Tensor,
    b_rowwise_data: torch.Tensor,
    b_rowwise_scale_inv: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float | None = None,
    asymmetric: bool = True,
) -> torch.Tensor:
    """Run TN GEMM through the compact-direct asymmetric scheduler.

    Inputs are logical ``A[M, K]`` and ``B[N, K]`` in compact rowwise MXFP8
    storage; the returned BF16 tensor is ``A @ B.T``. This shares the same
    stable GB10 scheduler as mixed compact-columnwise backward routes.
    """

    if a_rowwise_data.dim() != 2 or a_rowwise_scale_inv.dim() != 2:
        raise ValueError("A rowwise payload/scales must be 2D")
    if b_rowwise_data.dim() != 2 or b_rowwise_scale_inv.dim() != 2:
        raise ValueError("B rowwise payload/scales must be 2D")
    m = int(a_rowwise_data.shape[0])
    k = int(a_rowwise_data.shape[1])
    n = int(b_rowwise_data.shape[0])
    if int(b_rowwise_data.shape[1]) != k:
        raise ValueError(
            f"K mismatch: A is {tuple(a_rowwise_data.shape)}, B is {tuple(b_rowwise_data.shape)}"
        )
    return _tn_gemm_compact_direct(
        a_rowwise_data,
        a_rowwise_scale_inv,
        b_rowwise_data,
        b_rowwise_scale_inv,
        m=m,
        n=n,
        k=k,
        a_source=_SOURCE_ROWWISE,
        a_data_ld=int(a_rowwise_data.shape[1]),
        a_scale_ld=int(a_rowwise_scale_inv.shape[1]),
        b_source=_SOURCE_ROWWISE,
        b_data_ld=int(b_rowwise_data.shape[1]),
        b_scale_ld=int(b_rowwise_scale_inv.shape[1]),
        out=out,
        accumulate=accumulate,
        alpha=alpha,
        beta=beta,
        asymmetric=asymmetric,
    )


def tn_gemm_swizzled_scale(
    a_rowwise_data: torch.Tensor,
    a_gemm_scale_inv: torch.Tensor,
    b_rowwise_data: torch.Tensor,
    b_gemm_scale_inv: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float | None = None,
) -> torch.Tensor:
    """Probe TN GEMM with rowwise payloads and GEMM-swizzled MXFP8 scales.

    This bypasses cppmega's compact-scale producer and lets the stock CUTLASS
    block-scaled mainloop consume scale factors in the hardware layout emitted
    by TE or by the local swizzle probe.
    """

    if a_rowwise_data.dim() != 2 or b_rowwise_data.dim() != 2:
        raise ValueError("rowwise payload tensors must be 2D")
    if a_gemm_scale_inv.dtype != torch.uint8 or b_gemm_scale_inv.dtype != torch.uint8:
        raise TypeError("GEMM-swizzled MXFP8 scales must be uint8 tensors")

    m = int(a_rowwise_data.shape[0])
    k = int(a_rowwise_data.shape[1])
    n = int(b_rowwise_data.shape[0])
    if int(b_rowwise_data.shape[1]) != k:
        raise ValueError(
            f"K mismatch: A is {tuple(a_rowwise_data.shape)}, B is {tuple(b_rowwise_data.shape)}"
        )
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

    beta_f = _resolve_beta(beta, bool(accumulate))
    ext = _load_cuda_ext()
    return ext.tn_gemm_swizzled_scale(
        _as_uint8_contiguous(a_rowwise_data),
        _as_uint8_contiguous(a_gemm_scale_inv),
        _as_uint8_contiguous(b_rowwise_data),
        _as_uint8_contiguous(b_gemm_scale_inv),
        m,
        n,
        k,
        out_arg,
        use_out,
        bool(accumulate),
        float(alpha),
        beta_f,
    )


def wgrad_nt_gemm(
    dy_colwise_data: torch.Tensor,
    dy_colwise_scale_inv: torch.Tensor,
    x_colwise_data: torch.Tensor,
    x_colwise_scale_inv: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float | None = None,
    asymmetric: bool = True,
) -> torch.Tensor:
    """Run wgrad NT as TN without materializing ``dy.T`` or ``x.T``.

    Computes ``dy.T[N, M] @ x[M, K]`` from original TE compact columnwise
    payloads/scales for both operands.
    """

    if dy_colwise_data.dim() != 2 or dy_colwise_scale_inv.dim() != 2:
        raise ValueError("dy columnwise payload/scales must be 2D")
    if x_colwise_data.dim() != 2 or x_colwise_scale_inv.dim() != 2:
        raise ValueError("x columnwise payload/scales must be 2D")
    if int(dy_colwise_data.shape[0]) != int(x_colwise_data.shape[0]):
        raise ValueError(
            "wgrad reduction mismatch: "
            f"dy is {tuple(dy_colwise_data.shape)}, x is {tuple(x_colwise_data.shape)}"
        )
    m = int(dy_colwise_data.shape[1])
    n = int(x_colwise_data.shape[1])
    k = int(dy_colwise_data.shape[0])
    return _tn_gemm_compact_direct(
        dy_colwise_data,
        dy_colwise_scale_inv,
        x_colwise_data,
        x_colwise_scale_inv,
        m=m,
        n=n,
        k=k,
        a_source=_SOURCE_COLUMNWISE_TRANSPOSE,
        a_data_ld=int(dy_colwise_data.shape[1]),
        a_scale_ld=int(dy_colwise_scale_inv.shape[1]),
        b_source=_SOURCE_COLUMNWISE_TRANSPOSE,
        b_data_ld=int(x_colwise_data.shape[1]),
        b_scale_ld=int(x_colwise_scale_inv.shape[1]),
        out=out,
        accumulate=accumulate,
        alpha=alpha,
        beta=beta,
        asymmetric=asymmetric,
    )


def wgrad_nt_gemm_x_rowwise_transpose(
    dy_colwise_data: torch.Tensor,
    dy_colwise_scale_inv: torch.Tensor,
    x_t_rowwise_data: torch.Tensor,
    x_t_rowwise_scale_inv: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float | None = None,
    asymmetric: bool = True,
) -> torch.Tensor:
    """Run wgrad NT when ``x.T`` is already saved as rowwise MXFP8.

    Computes ``dy.T[N, M] @ x[M, K]``. ``dy`` is the original compact
    columnwise tensor, while ``x_t`` is the logical transpose ``[K, M]`` in
    rowwise storage. This avoids materializing ``dy.T`` or re-reading ``x`` as
    compact columnwise.
    """

    if dy_colwise_data.dim() != 2 or dy_colwise_scale_inv.dim() != 2:
        raise ValueError("dy columnwise payload/scales must be 2D")
    if x_t_rowwise_data.dim() != 2 or x_t_rowwise_scale_inv.dim() != 2:
        raise ValueError("x.T rowwise payload/scales must be 2D")
    if int(dy_colwise_data.shape[0]) != int(x_t_rowwise_data.shape[1]):
        raise ValueError(
            "wgrad reduction mismatch: "
            f"dy is {tuple(dy_colwise_data.shape)}, x.T is {tuple(x_t_rowwise_data.shape)}"
        )
    m = int(dy_colwise_data.shape[1])
    n = int(x_t_rowwise_data.shape[0])
    k = int(dy_colwise_data.shape[0])
    return _tn_gemm_compact_direct(
        dy_colwise_data,
        dy_colwise_scale_inv,
        x_t_rowwise_data,
        x_t_rowwise_scale_inv,
        m=m,
        n=n,
        k=k,
        a_source=_SOURCE_COLUMNWISE_TRANSPOSE,
        a_data_ld=int(dy_colwise_data.shape[1]),
        a_scale_ld=int(dy_colwise_scale_inv.shape[1]),
        b_source=_SOURCE_ROWWISE,
        b_data_ld=int(x_t_rowwise_data.shape[1]),
        b_scale_ld=int(x_t_rowwise_scale_inv.shape[1]),
        out=out,
        accumulate=accumulate,
        alpha=alpha,
        beta=beta,
        asymmetric=asymmetric,
        a_columnwise_smem=True,
    )
