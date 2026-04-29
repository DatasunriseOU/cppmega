#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <cstdint>

namespace cppmega_grouped_mxfp8 {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_UINT8(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Byte, #x " must be uint8")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::BFloat16, #x " must be bfloat16")

constexpr int64_t kNumExperts = 16;
constexpr int64_t kBlock = 32;

__device__ __forceinline__ float ue8m0_to_float(uint8_t x) {
  if (x == 0xff) {
    return __uint_as_float(0x7fffffff);
  }
  if (x == 0) {
    return __uint_as_float(0x00400000);
  }
  return __uint_as_float(static_cast<uint32_t>(x) << 23);
}

__device__ __forceinline__ float e4m3fn_to_float(uint8_t x) {
  uint8_t magnitude = x & 0x7f;
  if (magnitude == 0) {
    return (x & 0x80) ? -0.0f : 0.0f;
  }
  int exp = (magnitude >> 3) & 0x0f;
  int mant = magnitude & 0x07;
  if (exp == 0x0f && mant == 0x07) {
    return __uint_as_float(0x7fffffff);
  }
  float value;
  if (exp == 0) {
    value = ldexpf(static_cast<float>(mant), -9);
  } else {
    value = ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
  }
  return (x & 0x80) ? -value : value;
}

__device__ __forceinline__ int find_expert_for_row(
    int64_t row,
    int64_t const* __restrict__ expert_offsets) {
#pragma unroll
  for (int expert = 0; expert < kNumExperts; ++expert) {
    if (row >= expert_offsets[expert] && row < expert_offsets[expert + 1]) {
      return expert;
    }
  }
  return kNumExperts - 1;
}

__global__ void dgrad_nn_kernel(
    uint8_t const* __restrict__ dy,
    uint8_t const* __restrict__ sf_dy,
    uint8_t const* __restrict__ weight,
    uint8_t const* __restrict__ sf_weight,
    int64_t const* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ out,
    int64_t total_rows,
    int64_t n,
    int64_t k,
    int64_t n_blocks,
    float alpha,
    float beta,
    int64_t total_outputs) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_outputs) {
    return;
  }
  int64_t row = idx / k;
  int64_t col = idx - row * k;
  int expert = find_expert_for_row(row, expert_offsets);

  float accum = 0.0f;
  for (int64_t red = 0; red < n; ++red) {
    float lhs = e4m3fn_to_float(dy[row * n + red]) *
        ue8m0_to_float(sf_dy[row * n_blocks + red / kBlock]);
    int64_t weight_idx = (static_cast<int64_t>(expert) * n + red) * k + col;
    int64_t weight_sf_idx = (static_cast<int64_t>(expert) * n_blocks + red / kBlock) * k + col;
    float rhs = e4m3fn_to_float(weight[weight_idx]) * ue8m0_to_float(sf_weight[weight_sf_idx]);
    accum += lhs * rhs;
  }

  float result = alpha * accum;
  if (beta != 0.0f) {
    result += beta * __bfloat162float(out[idx]);
  }
  out[idx] = __float2bfloat16_rn(result);
}

__global__ void wgrad_nt_kernel(
    uint8_t const* __restrict__ dy,
    uint8_t const* __restrict__ sf_dy,
    uint8_t const* __restrict__ x,
    uint8_t const* __restrict__ sf_x,
    int64_t const* __restrict__ expert_offsets,
    int64_t const* __restrict__ scale_offsets,
    bool use_scale_offsets,
    __nv_bfloat16* __restrict__ out,
    int64_t total_rows,
    int64_t n,
    int64_t k,
    float alpha,
    float beta,
    int64_t total_outputs) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_outputs) {
    return;
  }
  int64_t col_k = idx % k;
  int64_t tmp = idx / k;
  int64_t col_n = tmp % n;
  int64_t expert = tmp / n;
  int64_t start = expert_offsets[expert];
  int64_t end = expert_offsets[expert + 1];
  int64_t scale_start = use_scale_offsets ? scale_offsets[expert] : 0;

  float accum = 0.0f;
  for (int64_t row = start; row < end; ++row) {
    int64_t row_block = use_scale_offsets ? scale_start + (row - start) / kBlock : row / kBlock;
    float lhs = e4m3fn_to_float(dy[row * n + col_n]) *
        ue8m0_to_float(sf_dy[row_block * n + col_n]);
    float rhs = e4m3fn_to_float(x[row * k + col_k]) *
        ue8m0_to_float(sf_x[row_block * k + col_k]);
    accum += lhs * rhs;
  }

  float result = alpha * accum;
  if (beta != 0.0f) {
    result += beta * __bfloat162float(out[idx]);
  }
  out[idx] = __float2bfloat16_rn(result);
}

__global__ void dgrad_nn_ptrs_kernel(
    uint64_t const* __restrict__ dy_ptrs,
    uint64_t const* __restrict__ sf_dy_ptrs,
    uint64_t const* __restrict__ weight_ptrs,
    uint64_t const* __restrict__ sf_weight_ptrs,
    int64_t const* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ out,
    int64_t total_rows,
    int64_t n,
    int64_t k,
    int64_t n_blocks,
    float alpha,
    float beta,
    int64_t total_outputs) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_outputs) {
    return;
  }
  int64_t row = idx / k;
  int64_t col = idx - row * k;
  int expert = find_expert_for_row(row, expert_offsets);
  int64_t local_row = row - expert_offsets[expert];

  auto const* dy = reinterpret_cast<uint8_t const*>(dy_ptrs[expert]);
  auto const* sf_dy = reinterpret_cast<uint8_t const*>(sf_dy_ptrs[expert]);
  auto const* weight = reinterpret_cast<uint8_t const*>(weight_ptrs[expert]);
  auto const* sf_weight = reinterpret_cast<uint8_t const*>(sf_weight_ptrs[expert]);

  float accum = 0.0f;
  for (int64_t red = 0; red < n; ++red) {
    float lhs = e4m3fn_to_float(dy[local_row * n + red]) *
        ue8m0_to_float(sf_dy[local_row * n_blocks + red / kBlock]);
    float rhs = e4m3fn_to_float(weight[red * k + col]) *
        ue8m0_to_float(sf_weight[(red / kBlock) * k + col]);
    accum += lhs * rhs;
  }

  float result = alpha * accum;
  if (beta != 0.0f) {
    result += beta * __bfloat162float(out[idx]);
  }
  out[idx] = __float2bfloat16_rn(result);
}

__global__ void wgrad_nt_ptrs_kernel(
    uint64_t const* __restrict__ dy_ptrs,
    uint64_t const* __restrict__ sf_dy_ptrs,
    uint64_t const* __restrict__ x_ptrs,
    uint64_t const* __restrict__ sf_x_ptrs,
    uint64_t const* __restrict__ out_ptrs,
    int64_t const* __restrict__ expert_offsets,
    int64_t n,
    int64_t k,
    float alpha,
    float beta,
    int64_t total_outputs) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_outputs) {
    return;
  }
  int64_t col_k = idx % k;
  int64_t tmp = idx / k;
  int64_t col_n = tmp % n;
  int64_t expert = tmp / n;
  int64_t start = expert_offsets[expert];
  int64_t end = expert_offsets[expert + 1];

  auto const* dy = reinterpret_cast<uint8_t const*>(dy_ptrs[expert]);
  auto const* sf_dy = reinterpret_cast<uint8_t const*>(sf_dy_ptrs[expert]);
  auto const* x = reinterpret_cast<uint8_t const*>(x_ptrs[expert]);
  auto const* sf_x = reinterpret_cast<uint8_t const*>(sf_x_ptrs[expert]);
  auto* out = reinterpret_cast<__nv_bfloat16*>(out_ptrs[expert]);

  float accum = 0.0f;
  for (int64_t row = start; row < end; ++row) {
    int64_t local_row = row - start;
    int64_t row_block = local_row / kBlock;
    float lhs = e4m3fn_to_float(dy[local_row * n + col_n]) *
        ue8m0_to_float(sf_dy[row_block * n + col_n]);
    float rhs = e4m3fn_to_float(x[local_row * k + col_k]) *
        ue8m0_to_float(sf_x[row_block * k + col_k]);
    accum += lhs * rhs;
  }

  float result = alpha * accum;
  int64_t out_idx = col_n * k + col_k;
  if (beta != 0.0f) {
    result += beta * __bfloat162float(out[out_idx]);
  }
  out[out_idx] = __float2bfloat16_rn(result);
}

int64_t ceil_div(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

void validate_offsets(at::Tensor const& expert_offsets) {
  CHECK_CUDA(expert_offsets);
  CHECK_CONTIGUOUS(expert_offsets);
  TORCH_CHECK(expert_offsets.scalar_type() == at::ScalarType::Long, "expert_offsets must be int64");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be 1D");
  TORCH_CHECK(expert_offsets.size(0) == kNumExperts + 1, "expert_offsets must have 17 entries");
}

void validate_ptrs(at::Tensor const& ptrs, char const* name) {
  CHECK_CUDA(ptrs);
  CHECK_CONTIGUOUS(ptrs);
  TORCH_CHECK(ptrs.scalar_type() == at::ScalarType::Long, name, " must be int64");
  TORCH_CHECK(ptrs.dim() == 1 && ptrs.size(0) == kNumExperts, name, " must have 16 entries");
}

void validate_common_shape(int64_t total_rows, int64_t n, int64_t k) {
  TORCH_CHECK(total_rows > 0 && n > 0 && k > 0, "total_rows, n, and k must be positive");
  TORCH_CHECK(n % kBlock == 0 && k % kBlock == 0,
              "grouped MXFP8 prototype requires feature dims divisible by 32");
  TORCH_CHECK(total_rows <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
              "reference grouped MXFP8 dimensions must fit int");
}

void validate_device_match(at::Tensor const& ref, at::Tensor const& other, char const* name) {
  TORCH_CHECK(other.device() == ref.device(), name, " must be on the same device as the first operand");
}

}  // namespace cppmega_grouped_mxfp8

using namespace cppmega_grouped_mxfp8;

at::Tensor grouped_mxfp8_dgrad_nn_cuda(
    at::Tensor dy_u8,
    at::Tensor sf_dy_u8,
    at::Tensor weight_u8,
    at::Tensor sf_weight_u8,
    at::Tensor expert_offsets,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta) {
  CHECK_CUDA(dy_u8);
  CHECK_CUDA(sf_dy_u8);
  CHECK_CUDA(weight_u8);
  CHECK_CUDA(sf_weight_u8);
  CHECK_CONTIGUOUS(dy_u8);
  CHECK_CONTIGUOUS(sf_dy_u8);
  CHECK_CONTIGUOUS(weight_u8);
  CHECK_CONTIGUOUS(sf_weight_u8);
  CHECK_UINT8(dy_u8);
  CHECK_UINT8(sf_dy_u8);
  CHECK_UINT8(weight_u8);
  CHECK_UINT8(sf_weight_u8);
  validate_offsets(expert_offsets);
  validate_device_match(dy_u8, sf_dy_u8, "sf_dy_u8");
  validate_device_match(dy_u8, weight_u8, "weight_u8");
  validate_device_match(dy_u8, sf_weight_u8, "sf_weight_u8");
  validate_device_match(dy_u8, expert_offsets, "expert_offsets");

  TORCH_CHECK(dy_u8.dim() == 2, "dy_u8 must be 2D [T,N]");
  TORCH_CHECK(sf_dy_u8.dim() == 2, "sf_dy_u8 must be 2D [T,N/32]");
  TORCH_CHECK(weight_u8.dim() == 3, "weight_u8 must be 3D [16,N,K]");
  TORCH_CHECK(sf_weight_u8.dim() == 3, "sf_weight_u8 must be 3D [16,N/32,K]");
  TORCH_CHECK(weight_u8.size(0) == kNumExperts, "weight_u8 dim0 must be 16 experts");

  int64_t total_rows = dy_u8.size(0);
  int64_t n = dy_u8.size(1);
  int64_t k = weight_u8.size(2);
  int64_t n_blocks = ceil_div(n, kBlock);
  validate_common_shape(total_rows, n, k);
  TORCH_CHECK(weight_u8.size(1) == n, "weight_u8 dim1 must match dy N");
  TORCH_CHECK(sf_dy_u8.size(0) == total_rows && sf_dy_u8.size(1) == n_blocks,
              "sf_dy_u8 must have shape [T,N/32]");
  TORCH_CHECK(sf_weight_u8.size(0) == kNumExperts &&
              sf_weight_u8.size(1) == n_blocks &&
              sf_weight_u8.size(2) == k,
              "sf_weight_u8 must have shape [16,N/32,K]");
  TORCH_CHECK(use_out || beta == 0.0, "nonzero beta requires out");
  if (use_out) {
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(out);
    CHECK_BF16(out);
    validate_device_match(dy_u8, out, "out");
    TORCH_CHECK(out.sizes() == at::IntArrayRef({total_rows, k}), "out must have shape [T,K]");
  }

  c10::cuda::CUDAGuard device_guard(dy_u8.device());
  at::Tensor result = use_out
      ? out
      : at::empty({total_rows, k}, at::TensorOptions().device(dy_u8.device()).dtype(at::kBFloat16));

  int64_t total_outputs = total_rows * k;
  constexpr int threads = 256;
  int blocks = static_cast<int>((total_outputs + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  dgrad_nn_kernel<<<blocks, threads, 0, stream>>>(
      dy_u8.data_ptr<uint8_t>(),
      sf_dy_u8.data_ptr<uint8_t>(),
      weight_u8.data_ptr<uint8_t>(),
      sf_weight_u8.data_ptr<uint8_t>(),
      expert_offsets.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(result.data_ptr<at::BFloat16>()),
      total_rows,
      n,
      k,
      n_blocks,
      static_cast<float>(alpha),
      static_cast<float>(beta),
      total_outputs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  (void)accumulate;
  return result;
}

at::Tensor grouped_mxfp8_wgrad_nt_cuda(
    at::Tensor dy_u8,
    at::Tensor sf_dy_u8,
    at::Tensor x_u8,
    at::Tensor sf_x_u8,
    at::Tensor expert_offsets,
    at::Tensor scale_offsets,
    bool use_scale_offsets,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta) {
  CHECK_CUDA(dy_u8);
  CHECK_CUDA(sf_dy_u8);
  CHECK_CUDA(x_u8);
  CHECK_CUDA(sf_x_u8);
  CHECK_CONTIGUOUS(dy_u8);
  CHECK_CONTIGUOUS(sf_dy_u8);
  CHECK_CONTIGUOUS(x_u8);
  CHECK_CONTIGUOUS(sf_x_u8);
  CHECK_UINT8(dy_u8);
  CHECK_UINT8(sf_dy_u8);
  CHECK_UINT8(x_u8);
  CHECK_UINT8(sf_x_u8);
  validate_offsets(expert_offsets);
  if (use_scale_offsets) {
    validate_offsets(scale_offsets);
  }
  validate_device_match(dy_u8, sf_dy_u8, "sf_dy_u8");
  validate_device_match(dy_u8, x_u8, "x_u8");
  validate_device_match(dy_u8, sf_x_u8, "sf_x_u8");
  validate_device_match(dy_u8, expert_offsets, "expert_offsets");
  if (use_scale_offsets) {
    validate_device_match(dy_u8, scale_offsets, "scale_offsets");
  }

  TORCH_CHECK(dy_u8.dim() == 2, "dy_u8 must be 2D [T,N]");
  TORCH_CHECK(x_u8.dim() == 2, "x_u8 must be 2D [T,K]");
  TORCH_CHECK(sf_dy_u8.dim() == 2, "sf_dy_u8 must be 2D [ceil(T/32),N]");
  TORCH_CHECK(sf_x_u8.dim() == 2, "sf_x_u8 must be 2D [ceil(T/32),K]");

  int64_t total_rows = dy_u8.size(0);
  int64_t n = dy_u8.size(1);
  int64_t k = x_u8.size(1);
  int64_t row_blocks = use_scale_offsets ? sf_dy_u8.size(0) : ceil_div(total_rows, kBlock);
  validate_common_shape(total_rows, n, k);
  TORCH_CHECK(x_u8.size(0) == total_rows, "x_u8 dim0 must match dy T");
  TORCH_CHECK(sf_dy_u8.size(0) == row_blocks && sf_dy_u8.size(1) == n,
              "sf_dy_u8 must have shape [ceil(T/32),N]");
  TORCH_CHECK(sf_x_u8.size(0) == row_blocks && sf_x_u8.size(1) == k,
              "sf_x_u8 must have shape [ceil(T/32),K]");
  TORCH_CHECK(use_out || beta == 0.0, "nonzero beta requires out");
  if (use_out) {
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(out);
    CHECK_BF16(out);
    validate_device_match(dy_u8, out, "out");
    TORCH_CHECK(out.sizes() == at::IntArrayRef({kNumExperts, n, k}), "out must have shape [16,N,K]");
  }

  c10::cuda::CUDAGuard device_guard(dy_u8.device());
  at::Tensor result = use_out
      ? out
      : at::empty({kNumExperts, n, k}, at::TensorOptions().device(dy_u8.device()).dtype(at::kBFloat16));

  int64_t total_outputs = kNumExperts * n * k;
  constexpr int threads = 256;
  int blocks = static_cast<int>((total_outputs + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  wgrad_nt_kernel<<<blocks, threads, 0, stream>>>(
      dy_u8.data_ptr<uint8_t>(),
      sf_dy_u8.data_ptr<uint8_t>(),
      x_u8.data_ptr<uint8_t>(),
      sf_x_u8.data_ptr<uint8_t>(),
      expert_offsets.data_ptr<int64_t>(),
      use_scale_offsets ? scale_offsets.data_ptr<int64_t>() : nullptr,
      use_scale_offsets,
      reinterpret_cast<__nv_bfloat16*>(result.data_ptr<at::BFloat16>()),
      total_rows,
      n,
      k,
      static_cast<float>(alpha),
      static_cast<float>(beta),
      total_outputs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  (void)accumulate;
  return result;
}

at::Tensor grouped_mxfp8_dgrad_nn_ptrs_cuda(
    at::Tensor dy_ptrs,
    at::Tensor sf_dy_ptrs,
    at::Tensor weight_ptrs,
    at::Tensor sf_weight_ptrs,
    at::Tensor expert_offsets,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta,
    int64_t total_rows,
    int64_t n,
    int64_t k) {
  validate_ptrs(dy_ptrs, "dy_ptrs");
  validate_ptrs(sf_dy_ptrs, "sf_dy_ptrs");
  validate_ptrs(weight_ptrs, "weight_ptrs");
  validate_ptrs(sf_weight_ptrs, "sf_weight_ptrs");
  validate_offsets(expert_offsets);
  validate_device_match(dy_ptrs, sf_dy_ptrs, "sf_dy_ptrs");
  validate_device_match(dy_ptrs, weight_ptrs, "weight_ptrs");
  validate_device_match(dy_ptrs, sf_weight_ptrs, "sf_weight_ptrs");
  validate_device_match(dy_ptrs, expert_offsets, "expert_offsets");
  validate_common_shape(total_rows, n, k);
  TORCH_CHECK(use_out || beta == 0.0, "nonzero beta requires out");
  if (use_out) {
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(out);
    CHECK_BF16(out);
    validate_device_match(dy_ptrs, out, "out");
    TORCH_CHECK(out.sizes() == at::IntArrayRef({total_rows, k}), "out must have shape [T,K]");
  }

  c10::cuda::CUDAGuard device_guard(dy_ptrs.device());
  at::Tensor result = use_out
      ? out
      : at::empty({total_rows, k}, at::TensorOptions().device(dy_ptrs.device()).dtype(at::kBFloat16));

  int64_t total_outputs = total_rows * k;
  constexpr int threads = 256;
  int blocks = static_cast<int>((total_outputs + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  dgrad_nn_ptrs_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<uint64_t const*>(dy_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<uint64_t const*>(sf_dy_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<uint64_t const*>(weight_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<uint64_t const*>(sf_weight_ptrs.data_ptr<int64_t>()),
      expert_offsets.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(result.data_ptr<at::BFloat16>()),
      total_rows,
      n,
      k,
      ceil_div(n, kBlock),
      static_cast<float>(alpha),
      static_cast<float>(beta),
      total_outputs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  (void)accumulate;
  return result;
}

void grouped_mxfp8_wgrad_nt_ptrs_cuda(
    at::Tensor dy_ptrs,
    at::Tensor sf_dy_ptrs,
    at::Tensor x_ptrs,
    at::Tensor sf_x_ptrs,
    at::Tensor out_ptrs,
    at::Tensor expert_offsets,
    bool accumulate,
    double alpha,
    double beta,
    int64_t total_rows,
    int64_t n,
    int64_t k) {
  validate_ptrs(dy_ptrs, "dy_ptrs");
  validate_ptrs(sf_dy_ptrs, "sf_dy_ptrs");
  validate_ptrs(x_ptrs, "x_ptrs");
  validate_ptrs(sf_x_ptrs, "sf_x_ptrs");
  validate_ptrs(out_ptrs, "out_ptrs");
  validate_offsets(expert_offsets);
  validate_device_match(dy_ptrs, sf_dy_ptrs, "sf_dy_ptrs");
  validate_device_match(dy_ptrs, x_ptrs, "x_ptrs");
  validate_device_match(dy_ptrs, sf_x_ptrs, "sf_x_ptrs");
  validate_device_match(dy_ptrs, out_ptrs, "out_ptrs");
  validate_device_match(dy_ptrs, expert_offsets, "expert_offsets");
  validate_common_shape(total_rows, n, k);

  c10::cuda::CUDAGuard device_guard(dy_ptrs.device());
  int64_t total_outputs = kNumExperts * n * k;
  constexpr int threads = 256;
  int blocks = static_cast<int>((total_outputs + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  wgrad_nt_ptrs_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<uint64_t const*>(dy_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<uint64_t const*>(sf_dy_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<uint64_t const*>(x_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<uint64_t const*>(sf_x_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<uint64_t const*>(out_ptrs.data_ptr<int64_t>()),
      expert_offsets.data_ptr<int64_t>(),
      n,
      k,
      static_cast<float>(alpha),
      static_cast<float>(beta),
      total_outputs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  (void)accumulate;
}
