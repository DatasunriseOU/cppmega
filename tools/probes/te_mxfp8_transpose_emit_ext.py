"""MXFP8 transposed-emission helper for probes.

Patched TransformerEngine builds expose this as
``transformer_engine_torch.mxfp8_scaling_transpose_cast``.  This module prefers
that real TE op and keeps the inline runtime extension as a fallback for
comparing older/unpatched wheels.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline


_BACKEND_ENV = "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND"


_CPP_SOURCE = r"""
#include <torch/extension.h>

void mxfp8_transpose_emit_cuda(
    torch::Tensor input,
    torch::Tensor columnwise_scale_inv,
    torch::Tensor output_rowwise_data,
    torch::Tensor output_rowwise_scale_inv);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "mxfp8_transpose_emit",
      &mxfp8_transpose_emit_cuda,
      "Emit rowwise MXFP8 storage for input.T from BF16 input and compact columnwise scales");
}
"""


_CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_UINT8(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Byte, #x " must be uint8")

__device__ __forceinline__ float e8m0_exp2_rcp(const uint8_t biased_exp) {
  return (biased_exp == 0)
             ? 1.0f
             : __int_as_float(static_cast<int32_t>(254 - biased_exp) << 23);
}

__global__ void transpose_scale_kernel(
    const uint8_t* __restrict__ in_scale,
    uint8_t* __restrict__ out_scale,
    const int64_t in_rows,
    const int64_t in_cols,
    const int64_t out_rows,
    const int64_t out_cols,
    const int64_t total) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t out_r = idx / out_cols;
  const int64_t out_c = idx - out_r * out_cols;
  uint8_t value = 0;
  if (out_c < in_rows && out_r < in_cols) {
    value = in_scale[out_c * in_cols + out_r];
  }
  out_scale[idx] = value;
}

constexpr int kTileDim = 16;

__global__ void emit_data_tiled_kernel(
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ columnwise_scale_inv,
    uint8_t* __restrict__ output_rowwise_data,
    const int64_t rows,
    const int64_t cols,
    const int64_t col_scale_stride) {
  __shared__ uint8_t tile[kTileDim][kTileDim + 1];

  const int64_t c = blockIdx.x * kTileDim + threadIdx.x;
  const int64_t r = blockIdx.y * kTileDim + threadIdx.y;
  if (r < rows && c < cols) {
    const uint8_t scale = columnwise_scale_inv[(r / 32) * col_scale_stride + c];
    const float scaled = __bfloat162float(input[r * cols + c]) * e8m0_exp2_rcp(scale);
    const __nv_fp8_e4m3 fp8_value(scaled);
    tile[threadIdx.y][threadIdx.x] = *reinterpret_cast<const uint8_t*>(&fp8_value);
  }

  __syncthreads();

  const int64_t out_r = blockIdx.x * kTileDim + threadIdx.y;
  const int64_t out_c = blockIdx.y * kTileDim + threadIdx.x;
  if (out_r < cols && out_c < rows) {
    output_rowwise_data[out_r * rows + out_c] = tile[threadIdx.x][threadIdx.y];
  }
}

}  // namespace

void mxfp8_transpose_emit_cuda(
    torch::Tensor input,
    torch::Tensor columnwise_scale_inv,
    torch::Tensor output_rowwise_data,
    torch::Tensor output_rowwise_scale_inv) {
  CHECK_CUDA(input);
  CHECK_CUDA(columnwise_scale_inv);
  CHECK_CUDA(output_rowwise_data);
  CHECK_CUDA(output_rowwise_scale_inv);
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(columnwise_scale_inv);
  CHECK_CONTIGUOUS(output_rowwise_data);
  CHECK_CONTIGUOUS(output_rowwise_scale_inv);
  CHECK_UINT8(columnwise_scale_inv);
  CHECK_UINT8(output_rowwise_data);
  CHECK_UINT8(output_rowwise_scale_inv);
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16, "input must be bfloat16");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(columnwise_scale_inv.dim() == 2, "columnwise_scale_inv must be 2D");
  TORCH_CHECK(output_rowwise_data.dim() == 2, "output_rowwise_data must be 2D");
  TORCH_CHECK(output_rowwise_scale_inv.dim() == 2, "output_rowwise_scale_inv must be 2D");

  const int64_t rows = input.size(0);
  const int64_t cols = input.size(1);
  TORCH_CHECK(rows % 32 == 0, "input rows must be divisible by 32");
  TORCH_CHECK(cols % 32 == 0, "input cols must be divisible by 32");
  TORCH_CHECK(output_rowwise_data.size(0) == cols, "output data dim0 must equal input cols");
  TORCH_CHECK(output_rowwise_data.size(1) == rows, "output data dim1 must equal input rows");
  TORCH_CHECK(columnwise_scale_inv.size(0) >= rows / 32, "columnwise scales dim0 too small");
  TORCH_CHECK(columnwise_scale_inv.size(1) >= cols, "columnwise scales dim1 too small");
  TORCH_CHECK(
      output_rowwise_scale_inv.size(0) == columnwise_scale_inv.size(1),
      "output scales dim0 must equal columnwise scales dim1");
  TORCH_CHECK(
      output_rowwise_scale_inv.size(1) == columnwise_scale_inv.size(0),
      "output scales dim1 must equal columnwise scales dim0");

  const c10::cuda::CUDAGuard device_guard(input.device());
  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();

  const int64_t scale_total = output_rowwise_scale_inv.numel();
  if (scale_total > 0) {
    const int blocks = static_cast<int>((scale_total + threads - 1) / threads);
    transpose_scale_kernel<<<blocks, threads, 0, stream>>>(
        columnwise_scale_inv.data_ptr<uint8_t>(),
        output_rowwise_scale_inv.data_ptr<uint8_t>(),
        columnwise_scale_inv.size(0),
        columnwise_scale_inv.size(1),
        output_rowwise_scale_inv.size(0),
        output_rowwise_scale_inv.size(1),
        scale_total);
  }

  if (input.numel() > 0) {
    const dim3 data_block(kTileDim, kTileDim);
    const dim3 data_grid(
        static_cast<unsigned int>((cols + kTileDim - 1) / kTileDim),
        static_cast<unsigned int>((rows + kTileDim - 1) / kTileDim));
    emit_data_tiled_kernel<<<data_grid, data_block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        columnwise_scale_inv.data_ptr<uint8_t>(),
        output_rowwise_data.data_ptr<uint8_t>(),
        rows,
        cols,
        columnwise_scale_inv.size(1));
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def load_extension() -> Any:
    """Compile and load the local probe extension."""

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1")
    return load_inline(
        name="cppmega_te_mxfp8_transpose_emit_probe",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=bool(int(os.environ.get("CPPMEGA_MXFP8_EXT_VERBOSE", "0"))),
    )


def emit_transpose_from_bf16(
    source: torch.Tensor,
    columnwise_scale_inv: torch.Tensor,
    *,
    fp8_dtype: int | None = None,
    with_gemm_swizzled_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(rowwise_data, rowwise_scale_inv)`` for ``source.T``.

    ``columnwise_scale_inv`` must be the compact E8M0 scales produced by TE for
    ``source``.  The scale tensor is transposed exactly; the FP8 payload is
    emitted from BF16 source values using those same scale bytes.
    """

    if source.dtype != torch.bfloat16:
        raise TypeError(f"source must be bfloat16, got {source.dtype}")
    if source.dim() != 2:
        raise ValueError(f"source must be 2D, got shape={tuple(source.shape)}")
    if columnwise_scale_inv.dtype != torch.uint8:
        raise TypeError(f"columnwise_scale_inv must be uint8, got {columnwise_scale_inv.dtype}")
    if columnwise_scale_inv.dim() != 2:
        raise ValueError("columnwise_scale_inv must be 2D")

    source = source.contiguous()
    columnwise_scale_inv = columnwise_scale_inv.contiguous()
    rows, cols = source.shape
    rowwise_data = torch.empty((cols, rows), device=source.device, dtype=torch.uint8)
    rowwise_scale_inv = torch.empty(
        (columnwise_scale_inv.shape[1], columnwise_scale_inv.shape[0]),
        device=source.device,
        dtype=torch.uint8,
    )
    backend = os.environ.get(_BACKEND_ENV, "auto").lower()
    if backend not in {"auto", "te", "probe"}:
        raise ValueError(f"{_BACKEND_ENV} must be one of auto, te, probe; got {backend!r}")
    if backend in {"auto", "te"}:
        try:
            import transformer_engine.common as te_common

            te_common.load_framework_extension("torch")
            import transformer_engine_torch as tex
        except ImportError:
            tex = None
        te_op = getattr(tex, "mxfp8_scaling_transpose_cast", None) if tex is not None else None
        if te_op is not None:
            if fp8_dtype is None:
                fp8_dtype = int(tex.DType.kFloat8E4M3)
            te_op(
                source,
                columnwise_scale_inv,
                rowwise_data,
                rowwise_scale_inv,
                rows,
                cols,
                fp8_dtype,
                bool(with_gemm_swizzled_scales),
            )
            return rowwise_data, rowwise_scale_inv
        if backend == "te":
            raise RuntimeError(
                "patched TransformerEngine op mxfp8_scaling_transpose_cast is not available"
            )
    if with_gemm_swizzled_scales:
        raise RuntimeError("local probe fallback cannot emit GEMM-swizzled MXFP8 scales")

    load_extension().mxfp8_transpose_emit(
        source,
        columnwise_scale_inv,
        rowwise_data,
        rowwise_scale_inv,
    )
    return rowwise_data, rowwise_scale_inv
