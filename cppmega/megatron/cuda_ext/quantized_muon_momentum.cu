#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
// Four warps/block raises occupancy for the quantized Muon update while the
// 256-element logical block still tiles exactly with 2 items per thread.
constexpr int kThreads = 128;
constexpr int kItemsPerThread = kBlockSize / kThreads;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kThreads / kWarpSize;
constexpr float kQMax = 127.0f;
constexpr float kAbsmaxEps = 1e-30f;

template <typename T>
__device__ __forceinline__ float load_value(const T* ptr, int64_t offset);

template <>
__device__ __forceinline__ float load_value<float>(const float* ptr, int64_t offset) {
  return ptr[offset];
}

template <>
__device__ __forceinline__ float load_value<__half>(const __half* ptr, int64_t offset) {
  return __half2float(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_value<__nv_bfloat16>(
    const __nv_bfloat16* ptr,
    int64_t offset) {
  return __bfloat162float(ptr[offset]);
}

template <typename T>
__device__ __forceinline__ void store_value(T* ptr, int64_t offset, float value);

template <>
__device__ __forceinline__ void store_value<float>(float* ptr, int64_t offset, float value) {
  ptr[offset] = value;
}

template <>
__device__ __forceinline__ void store_value<__half>(__half* ptr, int64_t offset, float value) {
  ptr[offset] = __float2half(value);
}

template <>
__device__ __forceinline__ void store_value<__nv_bfloat16>(
    __nv_bfloat16* ptr,
    int64_t offset,
    float value) {
  ptr[offset] = __float2bfloat16(value);
}

__device__ __forceinline__ int nearest_i8(float x) {
  float rounded = x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f);
  rounded = fminf(fmaxf(rounded, -127.0f), 127.0f);
  return static_cast<int>(rounded);
}

__device__ __forceinline__ int tensor_for_block(
    int64_t global_block,
    const int64_t* block_offsets,
    int num_tensors) {
  int lo = 0;
  int hi = num_tensors;
  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    if (block_offsets[mid] <= global_block) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ __forceinline__ void block_reduce_max_sum(
    float local_max,
    float local_sum,
    float* scratch_max,
    float* scratch_sum,
    float& out_max,
    float& out_sum) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid >> 5;

#pragma unroll
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, mask));
    local_sum += __shfl_xor_sync(0xffffffffu, local_sum, mask);
  }

  if (lane == 0) {
    scratch_max[warp_id] = local_max;
    scratch_sum[warp_id] = local_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    float v_max = (lane < kNumWarps) ? scratch_max[lane] : 0.0f;
    float v_sum = (lane < kNumWarps) ? scratch_sum[lane] : 0.0f;
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
      v_max = fmaxf(v_max, __shfl_xor_sync(0xffffffffu, v_max, mask));
      v_sum += __shfl_xor_sync(0xffffffffu, v_sum, mask);
    }
    if (lane == 0) {
      scratch_max[0] = v_max;
      scratch_sum[0] = v_sum;
    }
  }
  __syncthreads();

  out_max = scratch_max[0];
  out_sum = scratch_sum[0];
}

template <typename GradT, typename QT, bool UnsignedStorage>
__global__ __launch_bounds__(kThreads, 2) void qmuon_update_multi_kernel(
    const uintptr_t* __restrict__ q_ptrs,
    const uintptr_t* __restrict__ absmax_ptrs,
    const uintptr_t* __restrict__ grad_ptrs,
    const int64_t* __restrict__ n_elements,
    const int64_t* __restrict__ block_offsets,
    float* __restrict__ sumsq_out,
    const int64_t* __restrict__ block_group_ids,
    float* __restrict__ group_sumsq_out,
    int num_tensors,
    float beta) {
  const int64_t global_block = static_cast<int64_t>(blockIdx.x);
  const int tensor_idx = tensor_for_block(global_block, block_offsets, num_tensors);
  const int64_t local_block = global_block - block_offsets[tensor_idx];
  const int tid = threadIdx.x;
  const int64_t n = n_elements[tensor_idx];

  QT* q = reinterpret_cast<QT*>(q_ptrs[tensor_idx]);
  float* absmax = reinterpret_cast<float*>(absmax_ptrs[tensor_idx]);
  GradT* grad = reinterpret_cast<GradT*>(grad_ptrs[tensor_idx]);

  const float old_absmax = absmax[local_block];
  float updated[kItemsPerThread];
  int64_t elements[kItemsPerThread];
  bool valid[kItemsPerThread];
  float local_absmax = 0.0f;
  float local_sumsq = 0.0f;

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    elements[item] = local_block * kBlockSize + tid + item * kThreads;
    valid[item] = elements[item] < n;
    updated[item] = 0.0f;
    if (valid[item]) {
      int qi = static_cast<int>(q[elements[item]]);
      if constexpr (UnsignedStorage) {
        qi -= 128;
      }
      const float old_m = static_cast<float>(qi) * old_absmax / kQMax;
      const float g = load_value<GradT>(grad, elements[item]);
      updated[item] = beta * old_m + (1.0f - beta) * g;
      local_absmax = fmaxf(local_absmax, fabsf(updated[item]));
      local_sumsq += updated[item] * updated[item];
    }
  }

  __shared__ float scratch_max[kNumWarps];
  __shared__ float scratch_sum[kNumWarps];
  float new_absmax = 0.0f;
  float block_sumsq = 0.0f;
  block_reduce_max_sum(
      local_absmax, local_sumsq, scratch_max, scratch_sum, new_absmax, block_sumsq);

  const float new_absmax_clamped = fmaxf(new_absmax, kAbsmaxEps);
  const float inv_scale = new_absmax > kAbsmaxEps ? kQMax / new_absmax_clamped : 0.0f;
#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    if (valid[item]) {
      const int q_new = nearest_i8(updated[item] * inv_scale);
      if constexpr (UnsignedStorage) {
        q[elements[item]] = static_cast<QT>(q_new + 128);
      } else {
        q[elements[item]] = static_cast<QT>(q_new);
      }
      store_value<GradT>(grad, elements[item], updated[item]);
    }
  }
  if (tid == 0) {
    absmax[local_block] = new_absmax;
    if (sumsq_out != nullptr) {
      sumsq_out[global_block] = block_sumsq;
    }
    if (group_sumsq_out != nullptr) {
      const int64_t group_id = block_group_ids[global_block];
      if (group_id >= 0) {
        atomicAdd(group_sumsq_out + group_id, block_sumsq);
      }
    }
  }
}

// TODO: fuse update + scale via grid-group sync to remove the second launch
// and the global-memory round trip of the gradient buffer.
template <typename GradT>
__global__ __launch_bounds__(kThreads, 2) void qmuon_scale_multi_by_group_kernel(
    const uintptr_t* __restrict__ grad_ptrs,
    const int64_t* __restrict__ n_elements,
    const int64_t* __restrict__ block_offsets,
    const int64_t* __restrict__ block_group_ids,
    const float* __restrict__ inv_norms,
    int num_tensors) {
  const int64_t global_block = static_cast<int64_t>(blockIdx.x);
  const int tensor_idx = tensor_for_block(global_block, block_offsets, num_tensors);
  const int64_t local_block = global_block - block_offsets[tensor_idx];
  const int64_t group_id = block_group_ids[global_block];
  if (group_id < 0) {
    return;
  }

  const int tid = threadIdx.x;
  const int64_t n = n_elements[tensor_idx];
  GradT* grad = reinterpret_cast<GradT*>(grad_ptrs[tensor_idx]);
  const float scale = inv_norms[group_id];

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int64_t element = local_block * kBlockSize + tid + item * kThreads;
    if (element < n) {
      const float value = load_value<GradT>(grad, element) * scale;
      store_value<GradT>(grad, element, value);
    }
  }
}

template <typename GradT>
__global__ __launch_bounds__(kThreads, 2) void qmuon_scale_multi_by_group_from_sumsq_kernel(
    const uintptr_t* __restrict__ grad_ptrs,
    const int64_t* __restrict__ n_elements,
    const int64_t* __restrict__ block_offsets,
    const int64_t* __restrict__ block_group_ids,
    const float* __restrict__ group_sumsq,
    float eps2,
    int num_tensors) {
  const int64_t global_block = static_cast<int64_t>(blockIdx.x);
  const int tensor_idx = tensor_for_block(global_block, block_offsets, num_tensors);
  const int64_t local_block = global_block - block_offsets[tensor_idx];
  const int64_t group_id = block_group_ids[global_block];
  if (group_id < 0) {
    return;
  }

  const int tid = threadIdx.x;
  const int64_t n = n_elements[tensor_idx];
  GradT* grad = reinterpret_cast<GradT*>(grad_ptrs[tensor_idx]);
  const float scale = rsqrtf(fmaxf(group_sumsq[group_id], eps2));

#pragma unroll
  for (int item = 0; item < kItemsPerThread; ++item) {
    const int64_t element = local_block * kBlockSize + tid + item * kThreads;
    if (element < n) {
      const float value = load_value<GradT>(grad, element) * scale;
      store_value<GradT>(grad, element, value);
    }
  }
}

int64_t div_up(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

void validate_tensor_lists(
    const std::vector<at::Tensor>& q_tensors,
    const std::vector<at::Tensor>& absmax_tensors,
    const std::vector<at::Tensor>& grad_tensors,
    double beta,
    bool unsigned_storage) {
  TORCH_CHECK(!q_tensors.empty(), "q_tensors must not be empty");
  TORCH_CHECK(
      q_tensors.size() == absmax_tensors.size() && q_tensors.size() == grad_tensors.size(),
      "q_tensors, absmax_tensors, and grad_tensors must have the same length");
  TORCH_CHECK(beta >= 0.0 && beta <= 1.0, "beta must be in [0, 1], got ", beta);

  const auto q_dtype = unsigned_storage ? at::kByte : at::kChar;
  const auto grad_dtype = grad_tensors[0].scalar_type();
  TORCH_CHECK(
      grad_dtype == at::kBFloat16 || grad_dtype == at::kHalf || grad_dtype == at::kFloat,
      "grad tensors must be bf16, fp16, or fp32");
  const auto device = q_tensors[0].device();

  for (size_t i = 0; i < q_tensors.size(); ++i) {
    const auto& q = q_tensors[i];
    const auto& absmax = absmax_tensors[i];
    const auto& grad = grad_tensors[i];
    TORCH_CHECK(q.is_cuda() && absmax.is_cuda() && grad.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(q.device() == device && absmax.device() == device && grad.device() == device,
                "all tensors must be on the same CUDA device");
    TORCH_CHECK(q.scalar_type() == q_dtype, "all q tensors must match unsigned_storage dtype");
    TORCH_CHECK(absmax.scalar_type() == at::kFloat, "all absmax tensors must be float32");
    TORCH_CHECK(grad.scalar_type() == grad_dtype, "all grad tensors must share one dtype");
    TORCH_CHECK(q.is_contiguous() && absmax.is_contiguous() && grad.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(q.sizes() == grad.sizes(), "q and grad tensor shapes must match");
    const int64_t expected_blocks = div_up(q.numel(), kBlockSize);
    TORCH_CHECK(absmax.numel() == expected_blocks,
                "absmax tensor ", i, " has ", absmax.numel(),
                " entries, expected ", expected_blocks);
  }
}

template <typename GradT, typename QT, bool UnsignedStorage>
void launch_update(
    const int64_t* d_q_ptrs,
    const int64_t* d_absmax_ptrs,
    const int64_t* d_grad_ptrs,
    const int64_t* d_n_elements,
    const int64_t* d_block_offsets,
    const at::Tensor& sumsq_out,
    const at::Tensor& block_group_ids,
    const at::Tensor& group_sumsq_out,
    int num_tensors,
    int64_t total_blocks,
    float beta,
    cudaStream_t stream) {
  qmuon_update_multi_kernel<GradT, QT, UnsignedStorage>
      <<<static_cast<unsigned int>(total_blocks), kThreads, 0, stream>>>(
          reinterpret_cast<const uintptr_t*>(d_q_ptrs),
          reinterpret_cast<const uintptr_t*>(d_absmax_ptrs),
          reinterpret_cast<const uintptr_t*>(d_grad_ptrs),
          d_n_elements,
          d_block_offsets,
          sumsq_out.defined() ? sumsq_out.data_ptr<float>() : nullptr,
          block_group_ids.defined() ? block_group_ids.data_ptr<int64_t>() : nullptr,
          group_sumsq_out.defined() ? group_sumsq_out.data_ptr<float>() : nullptr,
          num_tensors,
          beta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename GradT>
void launch_scale_by_group(
    const int64_t* d_grad_ptrs,
    const int64_t* d_n_elements,
    const int64_t* d_block_offsets,
    const at::Tensor& block_group_ids,
    const at::Tensor& inv_norms,
    int num_tensors,
    int64_t total_blocks,
    cudaStream_t stream) {
  qmuon_scale_multi_by_group_kernel<GradT>
      <<<static_cast<unsigned int>(total_blocks), kThreads, 0, stream>>>(
          reinterpret_cast<const uintptr_t*>(d_grad_ptrs),
          d_n_elements,
          d_block_offsets,
          block_group_ids.data_ptr<int64_t>(),
          inv_norms.data_ptr<float>(),
          num_tensors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename GradT>
void launch_scale_by_group_from_sumsq(
    const int64_t* d_grad_ptrs,
    const int64_t* d_n_elements,
    const int64_t* d_block_offsets,
    const at::Tensor& block_group_ids,
    const at::Tensor& group_sumsq,
    float eps,
    int num_tensors,
    int64_t total_blocks,
    cudaStream_t stream) {
  qmuon_scale_multi_by_group_from_sumsq_kernel<GradT>
      <<<static_cast<unsigned int>(total_blocks), kThreads, 0, stream>>>(
          reinterpret_cast<const uintptr_t*>(d_grad_ptrs),
          d_n_elements,
          d_block_offsets,
          block_group_ids.data_ptr<int64_t>(),
          group_sumsq.data_ptr<float>(),
          eps * eps,
          num_tensors);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

at::Tensor qmuon_update_multi_impl_(
    std::vector<at::Tensor> q_tensors,
    std::vector<at::Tensor> absmax_tensors,
    std::vector<at::Tensor> grad_tensors,
    double beta,
    bool unsigned_storage,
    bool return_sumsq,
    const at::Tensor& block_group_ids = at::Tensor(),
    int64_t num_groups = 0) {
  validate_tensor_lists(q_tensors, absmax_tensors, grad_tensors, beta, unsigned_storage);

  c10::cuda::CUDAGuard device_guard(q_tensors[0].device());
  const auto device = q_tensors[0].device();
  const int num_tensors = static_cast<int>(q_tensors.size());

  const int64_t q_ptrs_offset = 0;
  const int64_t absmax_ptrs_offset = q_ptrs_offset + num_tensors;
  const int64_t grad_ptrs_offset = absmax_ptrs_offset + num_tensors;
  const int64_t n_elements_offset = grad_ptrs_offset + num_tensors;
  const int64_t block_offsets_offset = n_elements_offset + num_tensors;
  const int64_t metadata_size = block_offsets_offset + num_tensors + 1;
  std::vector<int64_t> h_metadata(metadata_size, 0);

  for (int i = 0; i < num_tensors; ++i) {
    h_metadata[q_ptrs_offset + i] = reinterpret_cast<int64_t>(q_tensors[i].data_ptr());
    h_metadata[absmax_ptrs_offset + i] =
        reinterpret_cast<int64_t>(absmax_tensors[i].data_ptr<float>());
    h_metadata[grad_ptrs_offset + i] = reinterpret_cast<int64_t>(grad_tensors[i].data_ptr());
    h_metadata[n_elements_offset + i] = q_tensors[i].numel();
    h_metadata[block_offsets_offset + i + 1] =
        h_metadata[block_offsets_offset + i] +
        div_up(h_metadata[n_elements_offset + i], kBlockSize);
  }

  const int64_t total_blocks = h_metadata[metadata_size - 1];
  if (total_blocks == 0) {
    return at::empty({0}, at::TensorOptions().device(device).dtype(at::kFloat));
  }

  const auto long_opts = at::TensorOptions().device(device).dtype(at::kLong);
  const auto float_opts = at::TensorOptions().device(device).dtype(at::kFloat);
  at::Tensor d_metadata = at::empty({metadata_size}, long_opts);
  at::Tensor sumsq_out;
  if (return_sumsq) {
    sumsq_out = at::empty({total_blocks}, float_opts);
  }
  at::Tensor group_sumsq_out;
  if (block_group_ids.defined()) {
    TORCH_CHECK(num_groups > 0, "num_groups must be positive for grouped sumsq");
    TORCH_CHECK(block_group_ids.is_cuda(), "block_group_ids must be CUDA");
    TORCH_CHECK(block_group_ids.device() == device, "block_group_ids must be on the same device");
    TORCH_CHECK(block_group_ids.scalar_type() == at::kLong, "block_group_ids must be int64");
    TORCH_CHECK(block_group_ids.is_contiguous(), "block_group_ids must be contiguous");
    TORCH_CHECK(
        block_group_ids.numel() == total_blocks,
        "block_group_ids has ",
        block_group_ids.numel(),
        " entries, expected ",
        total_blocks);
    group_sumsq_out = at::empty({num_groups}, float_opts);
  }

  auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
  if (group_sumsq_out.defined()) {
    C10_CUDA_CHECK(cudaMemsetAsync(
        group_sumsq_out.data_ptr<float>(), 0, sizeof(float) * num_groups, stream));
  }
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_metadata.data_ptr<int64_t>(),
      h_metadata.data(),
      sizeof(int64_t) * h_metadata.size(),
      cudaMemcpyHostToDevice,
      stream));
  const int64_t* d_metadata_ptr = d_metadata.data_ptr<int64_t>();
  const int64_t* d_q_ptrs = d_metadata_ptr + q_ptrs_offset;
  const int64_t* d_absmax_ptrs = d_metadata_ptr + absmax_ptrs_offset;
  const int64_t* d_grad_ptrs = d_metadata_ptr + grad_ptrs_offset;
  const int64_t* d_n_elements = d_metadata_ptr + n_elements_offset;
  const int64_t* d_block_offsets = d_metadata_ptr + block_offsets_offset;

  const float beta_f = static_cast<float>(beta);
  const auto grad_dtype = grad_tensors[0].scalar_type();

  if (unsigned_storage) {
    if (grad_dtype == at::kBFloat16) {
      launch_update<__nv_bfloat16, uint8_t, true>(
          d_q_ptrs, d_absmax_ptrs, d_grad_ptrs, d_n_elements, d_block_offsets,
          sumsq_out, block_group_ids, group_sumsq_out, num_tensors, total_blocks, beta_f, stream);
    } else if (grad_dtype == at::kHalf) {
      launch_update<__half, uint8_t, true>(
          d_q_ptrs, d_absmax_ptrs, d_grad_ptrs, d_n_elements, d_block_offsets,
          sumsq_out, block_group_ids, group_sumsq_out, num_tensors, total_blocks, beta_f, stream);
    } else {
      launch_update<float, uint8_t, true>(
          d_q_ptrs, d_absmax_ptrs, d_grad_ptrs, d_n_elements, d_block_offsets,
          sumsq_out, block_group_ids, group_sumsq_out, num_tensors, total_blocks, beta_f, stream);
    }
  } else {
    if (grad_dtype == at::kBFloat16) {
      launch_update<__nv_bfloat16, int8_t, false>(
          d_q_ptrs, d_absmax_ptrs, d_grad_ptrs, d_n_elements, d_block_offsets,
          sumsq_out, block_group_ids, group_sumsq_out, num_tensors, total_blocks, beta_f, stream);
    } else if (grad_dtype == at::kHalf) {
      launch_update<__half, int8_t, false>(
          d_q_ptrs, d_absmax_ptrs, d_grad_ptrs, d_n_elements, d_block_offsets,
          sumsq_out, block_group_ids, group_sumsq_out, num_tensors, total_blocks, beta_f, stream);
    } else {
      launch_update<float, int8_t, false>(
          d_q_ptrs, d_absmax_ptrs, d_grad_ptrs, d_n_elements, d_block_offsets,
          sumsq_out, block_group_ids, group_sumsq_out, num_tensors, total_blocks, beta_f, stream);
    }
  }
  return group_sumsq_out.defined() ? group_sumsq_out : sumsq_out;
}

void qmuon_update_multi_cuda_(
    std::vector<at::Tensor> q_tensors,
    std::vector<at::Tensor> absmax_tensors,
    std::vector<at::Tensor> grad_tensors,
    double beta,
    bool unsigned_storage) {
  qmuon_update_multi_impl_(q_tensors, absmax_tensors, grad_tensors, beta, unsigned_storage, false);
}

at::Tensor qmuon_update_multi_with_sumsq_cuda_(
    std::vector<at::Tensor> q_tensors,
    std::vector<at::Tensor> absmax_tensors,
    std::vector<at::Tensor> grad_tensors,
    double beta,
    bool unsigned_storage) {
  return qmuon_update_multi_impl_(q_tensors, absmax_tensors, grad_tensors, beta, unsigned_storage, true);
}

at::Tensor qmuon_update_multi_with_group_sumsq_cuda_(
    std::vector<at::Tensor> q_tensors,
    std::vector<at::Tensor> absmax_tensors,
    std::vector<at::Tensor> grad_tensors,
    at::Tensor block_group_ids,
    int64_t num_groups,
    double beta,
    bool unsigned_storage) {
  return qmuon_update_multi_impl_(
      q_tensors,
      absmax_tensors,
      grad_tensors,
      beta,
      unsigned_storage,
      false,
      block_group_ids,
      num_groups);
}

void qmuon_scale_multi_by_group_cuda_(
    std::vector<at::Tensor> grad_tensors,
    at::Tensor block_group_ids,
    at::Tensor inv_norms) {
  TORCH_CHECK(!grad_tensors.empty(), "grad_tensors must not be empty");
  TORCH_CHECK(block_group_ids.is_cuda(), "block_group_ids must be CUDA");
  TORCH_CHECK(block_group_ids.scalar_type() == at::kLong, "block_group_ids must be int64");
  TORCH_CHECK(block_group_ids.is_contiguous(), "block_group_ids must be contiguous");
  TORCH_CHECK(inv_norms.is_cuda(), "inv_norms must be CUDA");
  TORCH_CHECK(inv_norms.scalar_type() == at::kFloat, "inv_norms must be float32");
  TORCH_CHECK(inv_norms.is_contiguous(), "inv_norms must be contiguous");

  const auto device = grad_tensors[0].device();
  const auto grad_dtype = grad_tensors[0].scalar_type();
  TORCH_CHECK(
      grad_dtype == at::kBFloat16 || grad_dtype == at::kHalf || grad_dtype == at::kFloat,
      "grad tensors must be bf16, fp16, or fp32");
  TORCH_CHECK(block_group_ids.device() == device, "block_group_ids must be on grad device");
  TORCH_CHECK(inv_norms.device() == device, "inv_norms must be on grad device");

  const int num_tensors = static_cast<int>(grad_tensors.size());
  int64_t total_blocks = 0;
  for (size_t i = 0; i < grad_tensors.size(); ++i) {
    const auto& grad = grad_tensors[i];
    TORCH_CHECK(grad.is_cuda(), "all grad tensors must be CUDA");
    TORCH_CHECK(grad.device() == device, "all grad tensors must be on the same CUDA device");
    TORCH_CHECK(grad.scalar_type() == grad_dtype, "all grad tensors must share one dtype");
    TORCH_CHECK(grad.is_contiguous(), "all grad tensors must be contiguous");
    total_blocks += div_up(grad.numel(), kBlockSize);
  }
  TORCH_CHECK(
      block_group_ids.numel() == total_blocks,
      "block_group_ids has ",
      block_group_ids.numel(),
      " entries, expected ",
      total_blocks);
  if (total_blocks == 0) {
    return;
  }

  c10::cuda::CUDAGuard device_guard(device);
  const int64_t grad_ptrs_offset = 0;
  const int64_t n_elements_offset = grad_ptrs_offset + num_tensors;
  const int64_t block_offsets_offset = n_elements_offset + num_tensors;
  const int64_t metadata_size = block_offsets_offset + num_tensors + 1;
  std::vector<int64_t> h_metadata(metadata_size, 0);
  for (int i = 0; i < num_tensors; ++i) {
    h_metadata[grad_ptrs_offset + i] = reinterpret_cast<int64_t>(grad_tensors[i].data_ptr());
    h_metadata[n_elements_offset + i] = grad_tensors[i].numel();
    h_metadata[block_offsets_offset + i + 1] =
        h_metadata[block_offsets_offset + i] +
        div_up(h_metadata[n_elements_offset + i], kBlockSize);
  }

  const auto long_opts = at::TensorOptions().device(device).dtype(at::kLong);
  at::Tensor d_metadata = at::empty({metadata_size}, long_opts);
  auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_metadata.data_ptr<int64_t>(),
      h_metadata.data(),
      sizeof(int64_t) * h_metadata.size(),
      cudaMemcpyHostToDevice,
      stream));
  const int64_t* d_metadata_ptr = d_metadata.data_ptr<int64_t>();
  const int64_t* d_grad_ptrs = d_metadata_ptr + grad_ptrs_offset;
  const int64_t* d_n_elements = d_metadata_ptr + n_elements_offset;
  const int64_t* d_block_offsets = d_metadata_ptr + block_offsets_offset;

  if (grad_dtype == at::kBFloat16) {
    launch_scale_by_group<__nv_bfloat16>(
        d_grad_ptrs, d_n_elements, d_block_offsets, block_group_ids, inv_norms,
        num_tensors, total_blocks, stream);
  } else if (grad_dtype == at::kHalf) {
    launch_scale_by_group<__half>(
        d_grad_ptrs, d_n_elements, d_block_offsets, block_group_ids, inv_norms,
        num_tensors, total_blocks, stream);
  } else {
    launch_scale_by_group<float>(
        d_grad_ptrs, d_n_elements, d_block_offsets, block_group_ids, inv_norms,
        num_tensors, total_blocks, stream);
  }
}

void qmuon_scale_multi_by_group_from_sumsq_cuda_(
    std::vector<at::Tensor> grad_tensors,
    at::Tensor block_group_ids,
    at::Tensor group_sumsq,
    double eps) {
  TORCH_CHECK(!grad_tensors.empty(), "grad_tensors must not be empty");
  TORCH_CHECK(block_group_ids.is_cuda(), "block_group_ids must be CUDA");
  TORCH_CHECK(block_group_ids.scalar_type() == at::kLong, "block_group_ids must be int64");
  TORCH_CHECK(block_group_ids.is_contiguous(), "block_group_ids must be contiguous");
  TORCH_CHECK(group_sumsq.is_cuda(), "group_sumsq must be CUDA");
  TORCH_CHECK(group_sumsq.scalar_type() == at::kFloat, "group_sumsq must be float32");
  TORCH_CHECK(group_sumsq.is_contiguous(), "group_sumsq must be contiguous");
  TORCH_CHECK(eps > 0.0, "eps must be positive, got ", eps);

  const auto device = grad_tensors[0].device();
  const auto grad_dtype = grad_tensors[0].scalar_type();
  TORCH_CHECK(
      grad_dtype == at::kBFloat16 || grad_dtype == at::kHalf || grad_dtype == at::kFloat,
      "grad tensors must be bf16, fp16, or fp32");
  TORCH_CHECK(block_group_ids.device() == device, "block_group_ids must be on grad device");
  TORCH_CHECK(group_sumsq.device() == device, "group_sumsq must be on grad device");

  const int num_tensors = static_cast<int>(grad_tensors.size());
  int64_t total_blocks = 0;
  for (size_t i = 0; i < grad_tensors.size(); ++i) {
    const auto& grad = grad_tensors[i];
    TORCH_CHECK(grad.is_cuda(), "all grad tensors must be CUDA");
    TORCH_CHECK(grad.device() == device, "all grad tensors must be on the same CUDA device");
    TORCH_CHECK(grad.scalar_type() == grad_dtype, "all grad tensors must share one dtype");
    TORCH_CHECK(grad.is_contiguous(), "all grad tensors must be contiguous");
    total_blocks += div_up(grad.numel(), kBlockSize);
  }
  TORCH_CHECK(
      block_group_ids.numel() == total_blocks,
      "block_group_ids has ",
      block_group_ids.numel(),
      " entries, expected ",
      total_blocks);
  if (total_blocks == 0) {
    return;
  }

  c10::cuda::CUDAGuard device_guard(device);
  const int64_t grad_ptrs_offset = 0;
  const int64_t n_elements_offset = grad_ptrs_offset + num_tensors;
  const int64_t block_offsets_offset = n_elements_offset + num_tensors;
  const int64_t metadata_size = block_offsets_offset + num_tensors + 1;
  std::vector<int64_t> h_metadata(metadata_size, 0);
  for (int i = 0; i < num_tensors; ++i) {
    h_metadata[grad_ptrs_offset + i] = reinterpret_cast<int64_t>(grad_tensors[i].data_ptr());
    h_metadata[n_elements_offset + i] = grad_tensors[i].numel();
    h_metadata[block_offsets_offset + i + 1] =
        h_metadata[block_offsets_offset + i] +
        div_up(h_metadata[n_elements_offset + i], kBlockSize);
  }

  const auto long_opts = at::TensorOptions().device(device).dtype(at::kLong);
  at::Tensor d_metadata = at::empty({metadata_size}, long_opts);
  auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      d_metadata.data_ptr<int64_t>(),
      h_metadata.data(),
      sizeof(int64_t) * h_metadata.size(),
      cudaMemcpyHostToDevice,
      stream));
  const int64_t* d_metadata_ptr = d_metadata.data_ptr<int64_t>();
  const int64_t* d_grad_ptrs = d_metadata_ptr + grad_ptrs_offset;
  const int64_t* d_n_elements = d_metadata_ptr + n_elements_offset;
  const int64_t* d_block_offsets = d_metadata_ptr + block_offsets_offset;
  const float eps_f = static_cast<float>(eps);

  if (grad_dtype == at::kBFloat16) {
    launch_scale_by_group_from_sumsq<__nv_bfloat16>(
        d_grad_ptrs, d_n_elements, d_block_offsets, block_group_ids, group_sumsq,
        eps_f, num_tensors, total_blocks, stream);
  } else if (grad_dtype == at::kHalf) {
    launch_scale_by_group_from_sumsq<__half>(
        d_grad_ptrs, d_n_elements, d_block_offsets, block_group_ids, group_sumsq,
        eps_f, num_tensors, total_blocks, stream);
  } else {
    launch_scale_by_group_from_sumsq<float>(
        d_grad_ptrs, d_n_elements, d_block_offsets, block_group_ids, group_sumsq,
        eps_f, num_tensors, total_blocks, stream);
  }
}
