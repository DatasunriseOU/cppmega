"""Batched MXFP8 saved-transpose emits for TE Linear backward.

This module deliberately targets the narrow cppmega GB10 MXFP8 bottleneck:
many TE Linear autograd saves each launch a small rowwise-transpose emit.  The
shim queues those saves and flushes them through one CUDA launch when a backward
GEMM needs the operands.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.utils.cpp_extension import load_inline


KIND_BF16_EMIT = 0
KIND_UINT8_TRANSPOSE = 1

_CUDA_EXT: Any | None = None


_CPP_SOURCE = r"""
#include <torch/extension.h>

#include <vector>

void cppmega_mxfp8_batched_transpose_cuda(
    std::vector<torch::Tensor> inputs,
    std::vector<torch::Tensor> columnwise_scales,
    std::vector<torch::Tensor> output_rowwise_data,
    std::vector<torch::Tensor> output_rowwise_scales,
    std::vector<int64_t> kinds,
    std::vector<int64_t> swizzled,
    std::vector<int64_t> rows,
    std::vector<int64_t> cols);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "batched_transpose",
      &cppmega_mxfp8_batched_transpose_cuda,
      "Batch MXFP8 BF16->FP8 transpose emits and uint8 transpose copies");
}
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int64_t kKindBf16Emit = 0;
constexpr int64_t kKindUint8Transpose = 1;
constexpr int kTransposeTileDim = 16;
constexpr uint32_t kFp32MantissaBits = 23;
constexpr int64_t kMetaFields = 11;
constexpr int64_t kInputPtr = 0;
constexpr int64_t kScalePtr = 1;
constexpr int64_t kOutputDataPtr = 2;
constexpr int64_t kOutputScalePtr = 3;
constexpr int64_t kRows = 4;
constexpr int64_t kCols = 5;
constexpr int64_t kScaleRows = 6;
constexpr int64_t kScaleCols = 7;
constexpr int64_t kOutputScaleStride = 8;
constexpr int64_t kKind = 9;
constexpr int64_t kSwizzled = 10;

__device__ __forceinline__ float cppmega_exp2f_rcp(uint8_t biased_exp) {
  if (biased_exp == 255) {
    return __int_as_float(0x7fffffff);
  }
  if (biased_exp == 254) {
    return __int_as_float(0x00400000);
  }
  return __int_as_float((254 - static_cast<uint32_t>(biased_exp)) << kFp32MantissaBits);
}

__device__ __forceinline__ size_t cppmega_gemm_swizzled_scale_idx(
    size_t i, size_t j, size_t num_tiles_x) {
  constexpr size_t kTileDimX = 4;
  constexpr size_t kTileDimY = 128;
  constexpr size_t kTileSize = kTileDimX * kTileDimY;
  const size_t tile_idx_x = j / kTileDimX;
  const size_t tile_idx_y = i / kTileDimY;
  const size_t idx_in_tile_x = j % kTileDimX;
  const size_t idx_in_tile_y = i % kTileDimY;
  size_t idx = (tile_idx_y * num_tiles_x + tile_idx_x) * kTileSize;
  idx += (idx_in_tile_y % 32) * 16 + (idx_in_tile_y / 32) * 4 + idx_in_tile_x;
  return idx;
}

__global__ void __launch_bounds__(kTransposeTileDim * kTransposeTileDim)
    cppmega_mxfp8_batched_transpose_kernel(
        const int64_t* meta,
        int64_t num_entries) {
  const int64_t entry = static_cast<int64_t>(blockIdx.z);
  if (entry >= num_entries) {
    return;
  }

  __shared__ uint8_t tile[kTransposeTileDim][kTransposeTileDim + 1];

  const int64_t row_count = meta[kRows * num_entries + entry];
  const int64_t col_count = meta[kCols * num_entries + entry];
  const int64_t scale_row_count = meta[kScaleRows * num_entries + entry];
  const int64_t scale_col_count = meta[kScaleCols * num_entries + entry];
  const int64_t output_scale_stride = meta[kOutputScaleStride * num_entries + entry];
  const int64_t kind = meta[kKind * num_entries + entry];
  const bool use_swizzled = meta[kSwizzled * num_entries + entry] != 0;

  const uint8_t* scale =
      reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(meta[kScalePtr * num_entries + entry]));
  uint8_t* output_data =
      reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(meta[kOutputDataPtr * num_entries + entry]));
  uint8_t* output_scale =
      reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(meta[kOutputScalePtr * num_entries + entry]));

  const int64_t c = static_cast<int64_t>(blockIdx.x) * kTransposeTileDim + threadIdx.x;
  const int64_t r = static_cast<int64_t>(blockIdx.y) * kTransposeTileDim + threadIdx.y;

  if (threadIdx.y == 0 && (blockIdx.y % 2 == 0)) {
    const int64_t out_r = c;
    const int64_t out_c = (static_cast<int64_t>(blockIdx.y) * kTransposeTileDim) / 32;
    if (out_r < scale_col_count && out_c < scale_row_count) {
      size_t output_idx = static_cast<size_t>(out_r * output_scale_stride + out_c);
      uint8_t scale_value = 0;
      if (!use_swizzled || (out_r < col_count && out_c < row_count / 32)) {
        scale_value = scale[out_c * scale_col_count + out_r];
      }
      if (use_swizzled) {
        const size_t num_tiles_x = (static_cast<size_t>(row_count) + 127) / 128;
        output_idx = cppmega_gemm_swizzled_scale_idx(
            static_cast<size_t>(out_r),
            static_cast<size_t>(out_c),
            num_tiles_x);
      }
      output_scale[output_idx] = scale_value;
    }
  }

  if (kind == kKindBf16Emit) {
    const __nv_bfloat16* input =
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(meta[kInputPtr * num_entries + entry]));
    if (r < row_count && c < col_count) {
      const uint8_t biased_exponent = scale[(r / 32) * scale_col_count + c];
      const float block_scale_inverse = cppmega_exp2f_rcp(biased_exponent);
      const float value = __bfloat162float(input[r * col_count + c]) * block_scale_inverse;
      tile[threadIdx.y][threadIdx.x] =
          __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    }
  } else {
    const uint8_t* input =
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(meta[kInputPtr * num_entries + entry]));
    if (r < row_count && c < col_count) {
      tile[threadIdx.y][threadIdx.x] = input[r * col_count + c];
    }
  }

  __syncthreads();

  const int64_t out_r = static_cast<int64_t>(blockIdx.x) * kTransposeTileDim + threadIdx.y;
  const int64_t out_c = static_cast<int64_t>(blockIdx.y) * kTransposeTileDim + threadIdx.x;
  if (out_r < col_count && out_c < row_count) {
    output_data[out_r * row_count + out_c] = tile[threadIdx.x][threadIdx.y];
  }
}

void check_cuda_byte_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kByte, name, " must be uint8");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void copy_i64_vector_to_device(
    torch::Tensor& dst, const std::vector<int64_t>& src, cudaStream_t stream) {
  C10_CUDA_CHECK(cudaMemcpyAsync(
      dst.data_ptr<int64_t>(),
      src.data(),
      src.size() * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      stream));
}

}  // namespace

void cppmega_mxfp8_batched_transpose_cuda(
    std::vector<torch::Tensor> inputs,
    std::vector<torch::Tensor> columnwise_scales,
    std::vector<torch::Tensor> output_rowwise_data,
    std::vector<torch::Tensor> output_rowwise_scales,
    std::vector<int64_t> kinds,
    std::vector<int64_t> swizzled,
    std::vector<int64_t> rows,
    std::vector<int64_t> cols) {
  const int64_t n = static_cast<int64_t>(inputs.size());
  TORCH_CHECK(n > 0, "batched_transpose requires at least one entry");
  TORCH_CHECK(static_cast<int64_t>(columnwise_scales.size()) == n, "scale count mismatch");
  TORCH_CHECK(static_cast<int64_t>(output_rowwise_data.size()) == n, "output data count mismatch");
  TORCH_CHECK(static_cast<int64_t>(output_rowwise_scales.size()) == n, "output scale count mismatch");
  TORCH_CHECK(static_cast<int64_t>(kinds.size()) == n, "kind count mismatch");
  TORCH_CHECK(static_cast<int64_t>(swizzled.size()) == n, "swizzled count mismatch");
  TORCH_CHECK(static_cast<int64_t>(rows.size()) == n, "row count mismatch");
  TORCH_CHECK(static_cast<int64_t>(cols.size()) == n, "col count mismatch");

  const auto device = inputs[0].device();
  c10::cuda::CUDAGuard device_guard(device);

  std::vector<int64_t> meta(kMetaFields * n);
  auto set_meta = [&meta, n](int64_t field, int64_t entry, int64_t value) {
    meta[field * n + entry] = value;
  };
  int64_t grid_x = 1;
  int64_t grid_y = 1;

  for (int64_t i = 0; i < n; ++i) {
    TORCH_CHECK(inputs[i].device() == device, "all inputs must be on one CUDA device");
    TORCH_CHECK(columnwise_scales[i].device() == device, "all scales must be on one CUDA device");
    TORCH_CHECK(output_rowwise_data[i].device() == device, "all output data must be on one CUDA device");
    TORCH_CHECK(output_rowwise_scales[i].device() == device, "all output scales must be on one CUDA device");
    TORCH_CHECK(inputs[i].is_contiguous(), "input must be contiguous");
    TORCH_CHECK(inputs[i].dim() == 2, "input must be 2D");
    check_cuda_byte_contiguous(columnwise_scales[i], "columnwise_scale");
    check_cuda_byte_contiguous(output_rowwise_data[i], "output_rowwise_data");
    check_cuda_byte_contiguous(output_rowwise_scales[i], "output_rowwise_scale");
    TORCH_CHECK(columnwise_scales[i].dim() == 2, "columnwise_scale must be 2D");
    TORCH_CHECK(output_rowwise_data[i].dim() == 2, "output_rowwise_data must be 2D");
    TORCH_CHECK(output_rowwise_scales[i].dim() == 2, "output_rowwise_scale must be 2D");
    TORCH_CHECK(rows[i] == inputs[i].size(0), "rows must match input dim0");
    TORCH_CHECK(cols[i] == inputs[i].size(1), "cols must match input dim1");
    TORCH_CHECK(rows[i] % 32 == 0, "rows must be divisible by 32");
    TORCH_CHECK(cols[i] % 32 == 0, "cols must be divisible by 32");
    TORCH_CHECK(output_rowwise_data[i].size(0) == cols[i], "output data dim0 must be cols");
    TORCH_CHECK(output_rowwise_data[i].size(1) == rows[i], "output data dim1 must be rows");
    TORCH_CHECK(output_rowwise_scales[i].size(0) == columnwise_scales[i].size(1),
                "output scale dim0 must match scale dim1");
    TORCH_CHECK(output_rowwise_scales[i].size(1) == columnwise_scales[i].size(0),
                "output scale dim1 must match scale dim0");
    TORCH_CHECK(columnwise_scales[i].size(0) >= rows[i] / 32, "scale dim0 is too small");
    TORCH_CHECK(columnwise_scales[i].size(1) >= cols[i], "scale dim1 is too small");

    if (kinds[i] == kKindBf16Emit) {
      TORCH_CHECK(inputs[i].scalar_type() == at::kBFloat16, "BF16 emit input must be bfloat16");
    } else if (kinds[i] == kKindUint8Transpose) {
      TORCH_CHECK(inputs[i].scalar_type() == at::kByte, "uint8 transpose input must be uint8");
      TORCH_CHECK(swizzled[i] == 0, "uint8 transpose currently supports compact scales only");
    } else {
      TORCH_CHECK(false, "unsupported batched transpose kind: ", kinds[i]);
    }

    set_meta(kInputPtr, i, static_cast<int64_t>(reinterpret_cast<uintptr_t>(inputs[i].data_ptr())));
    set_meta(kScalePtr, i, static_cast<int64_t>(reinterpret_cast<uintptr_t>(columnwise_scales[i].data_ptr())));
    set_meta(kOutputDataPtr, i,
             static_cast<int64_t>(reinterpret_cast<uintptr_t>(output_rowwise_data[i].data_ptr())));
    set_meta(kOutputScalePtr, i,
             static_cast<int64_t>(reinterpret_cast<uintptr_t>(output_rowwise_scales[i].data_ptr())));
    set_meta(kRows, i, rows[i]);
    set_meta(kCols, i, cols[i]);
    set_meta(kScaleRows, i, columnwise_scales[i].size(0));
    set_meta(kScaleCols, i, columnwise_scales[i].size(1));
    set_meta(kOutputScaleStride, i, output_rowwise_scales[i].stride(0));
    set_meta(kKind, i, kinds[i]);
    set_meta(kSwizzled, i, swizzled[i]);

    const int64_t entry_grid_x = std::max(
        (cols[i] + kTransposeTileDim - 1) / kTransposeTileDim,
        (columnwise_scales[i].size(1) + kTransposeTileDim - 1) / kTransposeTileDim);
    const int64_t entry_grid_y = std::max(
        (rows[i] + kTransposeTileDim - 1) / kTransposeTileDim,
        columnwise_scales[i].size(0) * 2);
    grid_x = std::max(grid_x, entry_grid_x);
    grid_y = std::max(grid_y, entry_grid_y);
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto d_meta = torch::empty({kMetaFields, n}, opts);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  copy_i64_vector_to_device(d_meta, meta, stream);

  dim3 block(kTransposeTileDim, kTransposeTileDim);
  dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y),
            static_cast<unsigned int>(n));
  cppmega_mxfp8_batched_transpose_kernel<<<grid, block, 0, stream>>>(
      d_meta.data_ptr<int64_t>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _load_cuda_ext() -> Any:
    global _CUDA_EXT
    if _CUDA_EXT is not None:
        return _CUDA_EXT
    _CUDA_EXT = load_inline(
        name="cppmega_mxfp8_batched_transpose",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return _CUDA_EXT


def _as_entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized = list(entries)
    if not materialized:
        return []
    return materialized


def batched_transpose(entries: Iterable[dict[str, Any]]) -> None:
    """Flush queued MXFP8 transpose emit/copy entries with one CUDA launch."""

    materialized = _as_entries(entries)
    if not materialized:
        return

    inputs: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    output_data: list[torch.Tensor] = []
    output_scales: list[torch.Tensor] = []
    kinds: list[int] = []
    swizzled: list[int] = []
    rows: list[int] = []
    cols: list[int] = []

    for entry in materialized:
        input_tensor = entry["input"]
        scale_tensor = entry["columnwise_scale_inv"]
        out_data = entry["output_rowwise_data"]
        out_scale = entry["output_rowwise_scale_inv"]
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("entry input must be a torch.Tensor")
        if input_tensor.dim() != 2:
            raise ValueError(f"entry input must be 2D, got {tuple(input_tensor.shape)}")
        inputs.append(input_tensor)
        scales.append(scale_tensor)
        output_data.append(out_data)
        output_scales.append(out_scale)
        kinds.append(int(entry["kind"]))
        swizzled.append(1 if entry.get("with_gemm_swizzled_scales", False) else 0)
        rows.append(int(input_tensor.shape[0]))
        cols.append(int(input_tensor.shape[1]))

    _load_cuda_ext().batched_transpose(
        inputs,
        scales,
        output_data,
        output_scales,
        kinds,
        swizzled,
        rows,
        cols,
    )
