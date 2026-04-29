#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kR = 4;
constexpr int kThreads = 128;
constexpr int kDqkElems = kR * kR;

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr, int64_t offset) {
  return static_cast<float>(ptr[offset]);
}

template <typename scalar_t>
__global__ void rr_diag_kernel(
    const scalar_t* __restrict__ dphi,
    const scalar_t* __restrict__ psiv,
    const scalar_t* __restrict__ q_pre_rot,
    const scalar_t* __restrict__ k_pre_rot,
    const scalar_t* __restrict__ qk_dot,
    const float* __restrict__ gamma,
    float* __restrict__ dgamma,
    float* __restrict__ dk_delta,
    float* __restrict__ dq_delta,
    int64_t total_programs,
    int C,
    int R,
    int P,
    int N) {
  extern __shared__ float smem[];
  float* partial = smem;
  float* dqk_s = smem + kDqkElems * blockDim.x;
  const int tid = threadIdx.x;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || R != kR) {
    return;
  }

  float acc[kDqkElems];
#pragma unroll
  for (int e = 0; e < kDqkElems; ++e) {
    acc[e] = 0.0f;
  }

  for (int p = tid; p < P; p += blockDim.x) {
    float d[kR];
    float v[kR];
#pragma unroll
    for (int r = 0; r < kR; ++r) {
      d[r] = load_as_float(dphi, ((pid * R + r) * static_cast<int64_t>(P)) + p);
      v[r] = load_as_float(psiv, ((pid * R + r) * static_cast<int64_t>(P)) + p);
    }
#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
      for (int s = 0; s < kR; ++s) {
        acc[r * kR + s] += d[r] * v[s];
      }
    }
  }

#pragma unroll
  for (int e = 0; e < kDqkElems; ++e) {
    partial[e * blockDim.x + tid] = acc[e];
  }
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
#pragma unroll
      for (int e = 0; e < kDqkElems; ++e) {
        partial[e * blockDim.x + tid] += partial[e * blockDim.x + tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid < kDqkElems) {
    dqk_s[tid] = partial[tid * blockDim.x];
  }
  __syncthreads();

  if (tid == 0) {
    float dg = 0.0f;
#pragma unroll
    for (int r = 0; r < kR; ++r) {
#pragma unroll
      for (int s = 0; s < kR; ++s) {
        const float qk = load_as_float(qk_dot, ((pid * R + r) * static_cast<int64_t>(R)) + s);
        dg += qk * dqk_s[r * kR + s];
      }
    }
    dgamma[pid] = dg;
  }

  const float g = gamma[pid];
  const int rn = R * N;
  for (int idx = tid; idx < rn; idx += blockDim.x) {
    const int i = idx / N;
    const int n = idx - i * N;
    float dk = 0.0f;
#pragma unroll
    for (int r = 0; r < kR; ++r) {
      const float scaled = dqk_s[r * kR + i] * g;
      const float q = load_as_float(q_pre_rot, ((pid * R + r) * static_cast<int64_t>(N)) + n);
      dk += scaled * q;
    }
    dk_delta[pid * static_cast<int64_t>(rn) + idx] = dk;
  }

  for (int idx = tid; idx < rn; idx += blockDim.x) {
    const int r = idx / N;
    const int n = idx - r * N;
    float dq = 0.0f;
#pragma unroll
    for (int i = 0; i < kR; ++i) {
      const float scaled = dqk_s[r * kR + i] * g;
      const float k = load_as_float(k_pre_rot, ((pid * R + i) * static_cast<int64_t>(N)) + n);
      dq += scaled * k;
    }
    dq_delta[pid * static_cast<int64_t>(rn) + idx] = dq;
  }
}

void check_input(const at::Tensor& tensor, const char* name, at::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " dtype mismatch");
}

std::vector<at::Tensor> rr_diag_forward(
    const at::Tensor& dphi,
    const at::Tensor& psiv,
    const at::Tensor& q_pre_rot,
    const at::Tensor& k_pre_rot,
    const at::Tensor& qk_dot,
    const at::Tensor& gamma) {
  check_input(dphi, "dphi", dphi.scalar_type());
  check_input(psiv, "psiv", dphi.scalar_type());
  check_input(q_pre_rot, "q_pre_rot", dphi.scalar_type());
  check_input(k_pre_rot, "k_pre_rot", dphi.scalar_type());
  check_input(qk_dot, "qk_dot", dphi.scalar_type());
  TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
  TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
  TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be fp32");
  TORCH_CHECK(dphi.dim() == 4, "dphi must have shape [tiles, C, R, P]");
  TORCH_CHECK(psiv.sizes() == dphi.sizes(), "psiv shape mismatch");
  TORCH_CHECK(qk_dot.size(0) == dphi.size(0), "qk_dot tiles mismatch");
  TORCH_CHECK(qk_dot.size(1) == dphi.size(1), "qk_dot C mismatch");
  TORCH_CHECK(qk_dot.size(2) == dphi.size(2), "qk_dot R mismatch");
  TORCH_CHECK(qk_dot.size(3) == dphi.size(2), "qk_dot R mismatch");
  TORCH_CHECK(q_pre_rot.size(0) == dphi.size(0), "q_pre_rot tiles mismatch");
  TORCH_CHECK(q_pre_rot.size(1) == dphi.size(1), "q_pre_rot C mismatch");
  TORCH_CHECK(q_pre_rot.size(2) == dphi.size(2), "q_pre_rot R mismatch");
  TORCH_CHECK(k_pre_rot.sizes() == q_pre_rot.sizes(), "k_pre_rot shape mismatch");
  TORCH_CHECK(gamma.sizes() == at::IntArrayRef({dphi.size(0), dphi.size(1)}), "gamma shape mismatch");

  const int64_t tiles = dphi.size(0);
  const int C = static_cast<int>(dphi.size(1));
  const int R = static_cast<int>(dphi.size(2));
  const int P = static_cast<int>(dphi.size(3));
  const int N = static_cast<int>(q_pre_rot.size(3));
  TORCH_CHECK(R == kR, "wave4 CUDA kernel currently specializes R=4, got R=", R);

  auto f32_opts = dphi.options().dtype(at::kFloat);
  at::Tensor dgamma = at::empty({tiles, C}, f32_opts);
  at::Tensor dk_delta = at::empty({tiles, C, R, N}, f32_opts);
  at::Tensor dq_delta = at::empty_like(dk_delta);

  const int64_t total_programs = tiles * static_cast<int64_t>(C);
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  const size_t smem_bytes = sizeof(float) * (kDqkElems * kThreads + kDqkElems);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dphi.scalar_type(), "rr_diag_forward", [&] {
    rr_diag_kernel<scalar_t><<<grid, block, smem_bytes, stream>>>(
        dphi.data_ptr<scalar_t>(),
        psiv.data_ptr<scalar_t>(),
        q_pre_rot.data_ptr<scalar_t>(),
        k_pre_rot.data_ptr<scalar_t>(),
        qk_dot.data_ptr<scalar_t>(),
        gamma.data_ptr<float>(),
        dgamma.data_ptr<float>(),
        dk_delta.data_ptr<float>(),
        dq_delta.data_ptr<float>(),
        total_programs,
        C,
        R,
        P,
        N);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dgamma, dk_delta, dq_delta};
}

template <typename scalar_t>
py::dict metadata_for_dtype() {
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, rr_diag_kernel<scalar_t>));

  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const size_t smem_bytes = sizeof(float) * (kDqkElems * kThreads + kDqkElems);
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, rr_diag_kernel<scalar_t>, kThreads, smem_bytes));

  py::dict out;
  out["threads_per_block"] = kThreads;
  out["dynamic_smem_bytes"] = static_cast<int64_t>(smem_bytes);
  out["num_regs"] = attrs.numRegs;
  out["static_smem_bytes"] = static_cast<int64_t>(attrs.sharedSizeBytes);
  out["const_bytes"] = static_cast<int64_t>(attrs.constSizeBytes);
  out["local_bytes"] = static_cast<int64_t>(attrs.localSizeBytes);
  out["max_threads_per_block"] = attrs.maxThreadsPerBlock;
  out["ptx_version"] = attrs.ptxVersion;
  out["binary_version"] = attrs.binaryVersion;
  out["active_blocks_per_sm"] = active_blocks;
  out["active_threads_per_sm"] = active_blocks * kThreads;
  out["max_threads_per_sm"] = prop.maxThreadsPerMultiProcessor;
  out["theoretical_occupancy_pct"] =
      100.0 * static_cast<double>(active_blocks * kThreads) /
      static_cast<double>(prop.maxThreadsPerMultiProcessor);
  return out;
}

py::dict rr_diag_kernel_metadata(const at::Tensor& dphi) {
  TORCH_CHECK(dphi.is_cuda(), "dphi must be CUDA");
  py::dict out;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dphi.scalar_type(), "rr_diag_kernel_metadata", [&] {
    out = metadata_for_dtype<scalar_t>();
  });
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rr_diag_forward", &rr_diag_forward, "R x R diagonal microkernel forward");
  m.def("rr_diag_kernel_metadata", &rr_diag_kernel_metadata, "R x R diagonal microkernel metadata");
}
