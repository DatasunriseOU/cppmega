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
constexpr int kChunk = 16;
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

template <typename scalar_t>
__global__ void stage2_rr_diag_post_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ q_flat,
    const scalar_t* __restrict__ k_flat,
    const scalar_t* __restrict__ v,
    const float* __restrict__ q_bias,
    const float* __restrict__ k_bias,
    const float* __restrict__ mimo_v,
    const float* __restrict__ mimo_o,
    const scalar_t* __restrict__ qk_dot,
    const float* __restrict__ dt,
    const scalar_t* __restrict__ trap,
    scalar_t* __restrict__ dk,
    scalar_t* __restrict__ dq,
    float* __restrict__ dgamma_diag,
    int64_t total_programs,
    int B,
    int S,
    int H,
    int G,
    int N,
    int P,
    int R) {
  extern __shared__ float smem[];
  float* partial = smem;
  float* dqk_s = smem + kDqkElems * blockDim.x;
  const int tid = threadIdx.x;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || R != kR) {
    return;
  }

  const int s = static_cast<int>(pid % S);
  const int64_t bh = pid / S;
  const int h = static_cast<int>(bh % H);
  const int b = static_cast<int>(bh / H);
  if (b >= B) {
    return;
  }
  const int h_per_group = H / G;
  const int h_qk = h / h_per_group;

  float acc[kDqkElems];
#pragma unroll
  for (int e = 0; e < kDqkElems; ++e) {
    acc[e] = 0.0f;
  }

  const int64_t dout_base = ((static_cast<int64_t>(b) * S + s) * H + h) * P;
  const int64_t mimo_base = static_cast<int64_t>(h) * R * P;
  for (int p = tid; p < P; p += blockDim.x) {
    const float dout_p = load_as_float(dout, dout_base + p);
    const float v_p = load_as_float(v, dout_base + p);
    float d[kR];
    float pv[kR];
#pragma unroll
    for (int r = 0; r < kR; ++r) {
      d[r] = dout_p * mimo_o[mimo_base + r * static_cast<int64_t>(P) + p];
      pv[r] = v_p * mimo_v[mimo_base + r * static_cast<int64_t>(P) + p];
    }
#pragma unroll
    for (int r_out = 0; r_out < kR; ++r_out) {
#pragma unroll
      for (int r_in = 0; r_in < kR; ++r_in) {
        acc[r_out * kR + r_in] += d[r_out] * pv[r_in];
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

  const int64_t bhs = (static_cast<int64_t>(b) * H + h) * S + s;
  if (tid == 0) {
    float dg = 0.0f;
    const int64_t qk_base = bhs * kDqkElems;
#pragma unroll
    for (int e = 0; e < kDqkElems; ++e) {
      dg += load_as_float(qk_dot, qk_base + e) * dqk_s[e];
    }
    dgamma_diag[bhs] = dg;
  }

  const float gamma = dt[bhs] / (1.0f + __expf(-load_as_float(trap, bhs)));
  const int rn = R * N;
  const int64_t q_base = ((static_cast<int64_t>(b) * S * R + s * R) * G + h_qk) * N;
  const int64_t out_base = ((static_cast<int64_t>(b) * S * R + s * R) * H + h) * N;
  const int64_t bias_base = static_cast<int64_t>(h) * R * N;

  for (int idx = tid; idx < rn; idx += blockDim.x) {
    const int r_in = idx / N;
    const int n = idx - r_in * N;
    float delta = 0.0f;
#pragma unroll
    for (int r_out = 0; r_out < kR; ++r_out) {
      const float q_pre =
          load_as_float(q_flat, q_base + (r_out * static_cast<int64_t>(G)) * N + n) +
          q_bias[bias_base + r_out * static_cast<int64_t>(N) + n];
      delta += dqk_s[r_out * kR + r_in] * gamma * q_pre;
    }
    const int64_t offset = out_base + (r_in * static_cast<int64_t>(H)) * N + n;
    const float old = load_as_float(dk, offset);
    dk[offset] = static_cast<scalar_t>(old + delta);
  }

  for (int idx = tid; idx < rn; idx += blockDim.x) {
    const int r_out = idx / N;
    const int n = idx - r_out * N;
    float delta = 0.0f;
#pragma unroll
    for (int r_in = 0; r_in < kR; ++r_in) {
      const float k_pre =
          load_as_float(k_flat, q_base + (r_in * static_cast<int64_t>(G)) * N + n) +
          k_bias[bias_base + r_in * static_cast<int64_t>(N) + n];
      delta += dqk_s[r_out * kR + r_in] * gamma * k_pre;
    }
    const int64_t offset = out_base + (r_out * static_cast<int64_t>(H)) * N + n;
    const float old = load_as_float(dq, offset);
    dq[offset] = static_cast<scalar_t>(old + delta);
  }
}

template <typename scalar_t>
__global__ void stage2_rr_diag_chunk_owner_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ q_flat,
    const scalar_t* __restrict__ k_flat,
    const scalar_t* __restrict__ v,
    const float* __restrict__ q_bias,
    const float* __restrict__ k_bias,
    const float* __restrict__ mimo_v,
    const float* __restrict__ mimo_o,
    const scalar_t* __restrict__ qk_dot,
    const float* __restrict__ dt,
    const scalar_t* __restrict__ trap,
    float* __restrict__ dgamma_diag,
    scalar_t* __restrict__ dk_delta,
    scalar_t* __restrict__ dq_delta,
    int64_t total_programs,
    int B,
    int S,
    int H,
    int G,
    int N,
    int P,
    int R,
    int nchunks,
    int chunk_size) {
  extern __shared__ float smem[];
  float* partial = smem;
  float* dqk_s = smem + kDqkElems * blockDim.x;
  const int tid = threadIdx.x;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || R != kR || chunk_size != kChunk) {
    return;
  }

  const int chunk = static_cast<int>(pid % nchunks);
  const int64_t bh = pid / nchunks;
  const int h = static_cast<int>(bh % H);
  const int b = static_cast<int>(bh / H);
  if (b >= B) {
    return;
  }
  const int h_per_group = H / G;
  const int h_qk = h / h_per_group;
  const int chunk_start = chunk * kChunk;
  const int64_t mimo_base = static_cast<int64_t>(h) * R * P;
  const int64_t bias_base = static_cast<int64_t>(h) * R * N;

#pragma unroll
  for (int local_cs = 0; local_cs < kChunk; ++local_cs) {
    const int s = chunk_start + local_cs;
    if (s >= S) {
      return;
    }

    float acc[kDqkElems];
#pragma unroll
    for (int e = 0; e < kDqkElems; ++e) {
      acc[e] = 0.0f;
    }

    const int64_t dout_base = ((static_cast<int64_t>(b) * S + s) * H + h) * P;
    for (int p = tid; p < P; p += blockDim.x) {
      const float dout_p = load_as_float(dout, dout_base + p);
      const float v_p = load_as_float(v, dout_base + p);
      float d[kR];
      float pv[kR];
#pragma unroll
      for (int r = 0; r < kR; ++r) {
        d[r] = dout_p * mimo_o[mimo_base + r * static_cast<int64_t>(P) + p];
        pv[r] = v_p * mimo_v[mimo_base + r * static_cast<int64_t>(P) + p];
      }
#pragma unroll
      for (int r_out = 0; r_out < kR; ++r_out) {
#pragma unroll
        for (int r_in = 0; r_in < kR; ++r_in) {
          acc[r_out * kR + r_in] += d[r_out] * pv[r_in];
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

    const int64_t bhs = (static_cast<int64_t>(b) * H + h) * S + s;
    if (tid == 0) {
      float dg = 0.0f;
      const int64_t qk_base = bhs * kDqkElems;
#pragma unroll
      for (int e = 0; e < kDqkElems; ++e) {
        dg += load_as_float(qk_dot, qk_base + e) * dqk_s[e];
      }
      dgamma_diag[bhs] = dg;
    }

    const float gamma = dt[bhs] / (1.0f + __expf(-load_as_float(trap, bhs)));
    const int rn = R * N;
    const int64_t q_base = ((static_cast<int64_t>(b) * S * R + s * R) * G + h_qk) * N;
    const int64_t out_base = ((static_cast<int64_t>(b) * S * R + s * R) * H + h) * N;

    for (int idx = tid; idx < rn; idx += blockDim.x) {
      const int r_in = idx / N;
      const int n = idx - r_in * N;
      float delta = 0.0f;
#pragma unroll
      for (int r_out = 0; r_out < kR; ++r_out) {
        const float q_pre =
            load_as_float(q_flat, q_base + (r_out * static_cast<int64_t>(G)) * N + n) +
            q_bias[bias_base + r_out * static_cast<int64_t>(N) + n];
        delta += dqk_s[r_out * kR + r_in] * gamma * q_pre;
      }
      const int64_t offset = out_base + (r_in * static_cast<int64_t>(H)) * N + n;
      dk_delta[offset] = static_cast<scalar_t>(delta);
    }

    for (int idx = tid; idx < rn; idx += blockDim.x) {
      const int r_out = idx / N;
      const int n = idx - r_out * N;
      float delta = 0.0f;
#pragma unroll
      for (int r_in = 0; r_in < kR; ++r_in) {
        const float k_pre =
            load_as_float(k_flat, q_base + (r_in * static_cast<int64_t>(G)) * N + n) +
            k_bias[bias_base + r_in * static_cast<int64_t>(N) + n];
        delta += dqk_s[r_out * kR + r_in] * gamma * k_pre;
      }
      const int64_t offset = out_base + (r_out * static_cast<int64_t>(H)) * N + n;
      dq_delta[offset] = static_cast<scalar_t>(delta);
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void stage2_rr_diag_chunk_warp_owner_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ q_flat,
    const scalar_t* __restrict__ k_flat,
    const scalar_t* __restrict__ v,
    const float* __restrict__ q_bias,
    const float* __restrict__ k_bias,
    const float* __restrict__ mimo_v,
    const float* __restrict__ mimo_o,
    const scalar_t* __restrict__ qk_dot,
    const float* __restrict__ dt,
    const scalar_t* __restrict__ trap,
    float* __restrict__ dgamma_diag,
    scalar_t* __restrict__ dk_delta,
    scalar_t* __restrict__ dq_delta,
    int64_t total_programs,
    int B,
    int S,
    int H,
    int G,
    int N,
    int P,
    int R,
    int nchunks,
    int chunk_size) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const unsigned mask = 0xffffffffu;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || R != kR || chunk_size != kChunk || blockDim.x != kThreads) {
    return;
  }

  const int chunk = static_cast<int>(pid % nchunks);
  const int64_t bh = pid / nchunks;
  const int h = static_cast<int>(bh % H);
  const int b = static_cast<int>(bh / H);
  if (b >= B) {
    return;
  }
  const int h_per_group = H / G;
  const int h_qk = h / h_per_group;
  const int chunk_start = chunk * kChunk;
  const int64_t mimo_base = static_cast<int64_t>(h) * R * P;
  const int64_t bias_base = static_cast<int64_t>(h) * R * N;

#pragma unroll
  for (int batch = 0; batch < kChunk; batch += 4) {
    const int local_cs = batch + warp;
    const int s = chunk_start + local_cs;
    if (local_cs >= kChunk || s >= S) {
      continue;
    }

    float acc[kDqkElems];
#pragma unroll
    for (int e = 0; e < kDqkElems; ++e) {
      acc[e] = 0.0f;
    }

    const int64_t dout_base = ((static_cast<int64_t>(b) * S + s) * H + h) * P;
    for (int p = lane; p < P; p += 32) {
      const float dout_p = load_as_float(dout, dout_base + p);
      const float v_p = load_as_float(v, dout_base + p);
      float d[kR];
      float pv[kR];
#pragma unroll
      for (int r = 0; r < kR; ++r) {
        d[r] = dout_p * mimo_o[mimo_base + r * static_cast<int64_t>(P) + p];
        pv[r] = v_p * mimo_v[mimo_base + r * static_cast<int64_t>(P) + p];
      }
#pragma unroll
      for (int r_out = 0; r_out < kR; ++r_out) {
#pragma unroll
        for (int r_in = 0; r_in < kR; ++r_in) {
          acc[r_out * kR + r_in] += d[r_out] * pv[r_in];
        }
      }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int e = 0; e < kDqkElems; ++e) {
        acc[e] += __shfl_down_sync(mask, acc[e], offset);
      }
    }
#pragma unroll
    for (int e = 0; e < kDqkElems; ++e) {
      acc[e] = __shfl_sync(mask, acc[e], 0);
    }

    const int64_t bhs = (static_cast<int64_t>(b) * H + h) * S + s;
    if (lane == 0) {
      float dg = 0.0f;
      const int64_t qk_base = bhs * kDqkElems;
#pragma unroll
      for (int e = 0; e < kDqkElems; ++e) {
        dg += load_as_float(qk_dot, qk_base + e) * acc[e];
      }
      dgamma_diag[bhs] = dg;
    }

    const float gamma = dt[bhs] / (1.0f + __expf(-load_as_float(trap, bhs)));
    const int rn = R * N;
    const int64_t q_base = ((static_cast<int64_t>(b) * S * R + s * R) * G + h_qk) * N;
    const int64_t out_base = ((static_cast<int64_t>(b) * S * R + s * R) * H + h) * N;

    for (int idx = lane; idx < rn; idx += 32) {
      const int r_in = idx / N;
      const int n = idx - r_in * N;
      float delta = 0.0f;
#pragma unroll
      for (int r_out = 0; r_out < kR; ++r_out) {
        const float q_pre =
            load_as_float(q_flat, q_base + (r_out * static_cast<int64_t>(G)) * N + n) +
            q_bias[bias_base + r_out * static_cast<int64_t>(N) + n];
        delta += acc[r_out * kR + r_in] * gamma * q_pre;
      }
      const int64_t offset = out_base + (r_in * static_cast<int64_t>(H)) * N + n;
      dk_delta[offset] = static_cast<scalar_t>(delta);
    }

    for (int idx = lane; idx < rn; idx += 32) {
      const int r_out = idx / N;
      const int n = idx - r_out * N;
      float delta = 0.0f;
#pragma unroll
      for (int r_in = 0; r_in < kR; ++r_in) {
        const float k_pre =
            load_as_float(k_flat, q_base + (r_in * static_cast<int64_t>(G)) * N + n) +
            k_bias[bias_base + r_in * static_cast<int64_t>(N) + n];
        delta += acc[r_out * kR + r_in] * gamma * k_pre;
      }
      const int64_t offset = out_base + (r_out * static_cast<int64_t>(H)) * N + n;
      dq_delta[offset] = static_cast<scalar_t>(delta);
    }
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

void stage2_rr_diag_post(
    const at::Tensor& dout,
    const at::Tensor& q_flat,
    const at::Tensor& k_flat,
    const at::Tensor& v,
    const at::Tensor& q_bias,
    const at::Tensor& k_bias,
    const at::Tensor& mimo_v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    const at::Tensor& dk,
    const at::Tensor& dq,
    const at::Tensor& dgamma_diag) {
  check_input(dout, "dout", dout.scalar_type());
  check_input(q_flat, "q_flat", dout.scalar_type());
  check_input(k_flat, "k_flat", dout.scalar_type());
  check_input(v, "v", dout.scalar_type());
  check_input(qk_dot, "qk_dot", dout.scalar_type());
  check_input(trap, "trap", dout.scalar_type());
  check_input(dk, "dk", dout.scalar_type());
  check_input(dq, "dq", dout.scalar_type());
  TORCH_CHECK(q_bias.is_cuda() && q_bias.is_contiguous(), "q_bias must be contiguous CUDA");
  TORCH_CHECK(k_bias.is_cuda() && k_bias.is_contiguous(), "k_bias must be contiguous CUDA");
  TORCH_CHECK(mimo_v.is_cuda() && mimo_v.is_contiguous(), "mimo_v must be contiguous CUDA");
  TORCH_CHECK(mimo_o.is_cuda() && mimo_o.is_contiguous(), "mimo_o must be contiguous CUDA");
  TORCH_CHECK(dt.is_cuda() && dt.is_contiguous(), "dt must be contiguous CUDA");
  TORCH_CHECK(dgamma_diag.is_cuda() && dgamma_diag.is_contiguous(), "dgamma_diag must be contiguous CUDA");
  TORCH_CHECK(q_bias.scalar_type() == at::kFloat, "q_bias must be fp32");
  TORCH_CHECK(k_bias.scalar_type() == at::kFloat, "k_bias must be fp32");
  TORCH_CHECK(mimo_v.scalar_type() == at::kFloat, "mimo_v must be fp32");
  TORCH_CHECK(mimo_o.scalar_type() == at::kFloat, "mimo_o must be fp32");
  TORCH_CHECK(dt.scalar_type() == at::kFloat, "dt must be fp32");
  TORCH_CHECK(dgamma_diag.scalar_type() == at::kFloat, "dgamma_diag must be fp32");

  TORCH_CHECK(dout.dim() == 4, "dout must have shape [B, S, H, P]");
  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  TORCH_CHECK(v.sizes() == dout.sizes(), "v shape mismatch");
  TORCH_CHECK(q_bias.dim() == 3, "q_bias must have shape [H, R, N]");
  TORCH_CHECK(k_bias.sizes() == q_bias.sizes(), "k_bias shape mismatch");
  TORCH_CHECK(mimo_v.sizes() == at::IntArrayRef({H, q_bias.size(1), P}), "mimo_v shape mismatch");
  TORCH_CHECK(mimo_o.sizes() == mimo_v.sizes(), "mimo_o shape mismatch");

  const int R = static_cast<int>(q_bias.size(1));
  const int N = static_cast<int>(q_bias.size(2));
  TORCH_CHECK(R == kR, "stage2 CUDA post kernel currently specializes R=4, got R=", R);
  TORCH_CHECK(q_flat.dim() == 4, "q_flat must have shape [B, S*R, G, N]");
  const int G = static_cast<int>(q_flat.size(2));
  TORCH_CHECK(G > 0 && H % G == 0, "H must be divisible by G");
  TORCH_CHECK(q_flat.sizes() == at::IntArrayRef({B, S * R, G, N}), "q_flat shape mismatch");
  TORCH_CHECK(k_flat.sizes() == q_flat.sizes(), "k_flat shape mismatch");
  TORCH_CHECK(qk_dot.sizes() == at::IntArrayRef({B, H, S, R * R}), "qk_dot must have shape [B, H, S, R*R]");
  TORCH_CHECK(dt.sizes() == at::IntArrayRef({B, H, S}), "dt shape mismatch");
  TORCH_CHECK(trap.sizes() == at::IntArrayRef({B, H, S}), "trap shape mismatch");
  TORCH_CHECK(dk.sizes() == at::IntArrayRef({B, S * R, H, N}), "dk shape mismatch");
  TORCH_CHECK(dq.sizes() == dk.sizes(), "dq shape mismatch");
  TORCH_CHECK(dgamma_diag.sizes() == at::IntArrayRef({B, H, S}), "dgamma_diag shape mismatch");

  const int64_t total_programs = static_cast<int64_t>(B) * H * S;
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  const size_t smem_bytes = sizeof(float) * (kDqkElems * kThreads + kDqkElems);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "stage2_rr_diag_post", [&] {
    stage2_rr_diag_post_kernel<scalar_t><<<grid, block, smem_bytes, stream>>>(
        dout.data_ptr<scalar_t>(),
        q_flat.data_ptr<scalar_t>(),
        k_flat.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        q_bias.data_ptr<float>(),
        k_bias.data_ptr<float>(),
        mimo_v.data_ptr<float>(),
        mimo_o.data_ptr<float>(),
        qk_dot.data_ptr<scalar_t>(),
        dt.data_ptr<float>(),
        trap.data_ptr<scalar_t>(),
        dk.data_ptr<scalar_t>(),
        dq.data_ptr<scalar_t>(),
        dgamma_diag.data_ptr<float>(),
        total_programs,
        B,
        S,
        H,
        G,
        N,
        P,
        R);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void validate_stage2_chunk_inputs(
    const at::Tensor& dout,
    const at::Tensor& q_flat,
    const at::Tensor& k_flat,
    const at::Tensor& v,
    const at::Tensor& q_bias,
    const at::Tensor& k_bias,
    const at::Tensor& mimo_v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    int chunk_size) {
  check_input(dout, "dout", dout.scalar_type());
  check_input(q_flat, "q_flat", dout.scalar_type());
  check_input(k_flat, "k_flat", dout.scalar_type());
  check_input(v, "v", dout.scalar_type());
  check_input(qk_dot, "qk_dot", dout.scalar_type());
  check_input(trap, "trap", dout.scalar_type());
  TORCH_CHECK(q_bias.is_cuda() && q_bias.is_contiguous(), "q_bias must be contiguous CUDA");
  TORCH_CHECK(k_bias.is_cuda() && k_bias.is_contiguous(), "k_bias must be contiguous CUDA");
  TORCH_CHECK(mimo_v.is_cuda() && mimo_v.is_contiguous(), "mimo_v must be contiguous CUDA");
  TORCH_CHECK(mimo_o.is_cuda() && mimo_o.is_contiguous(), "mimo_o must be contiguous CUDA");
  TORCH_CHECK(dt.is_cuda() && dt.is_contiguous(), "dt must be contiguous CUDA");
  TORCH_CHECK(q_bias.scalar_type() == at::kFloat, "q_bias must be fp32");
  TORCH_CHECK(k_bias.scalar_type() == at::kFloat, "k_bias must be fp32");
  TORCH_CHECK(mimo_v.scalar_type() == at::kFloat, "mimo_v must be fp32");
  TORCH_CHECK(mimo_o.scalar_type() == at::kFloat, "mimo_o must be fp32");
  TORCH_CHECK(dt.scalar_type() == at::kFloat, "dt must be fp32");
  TORCH_CHECK(chunk_size == kChunk, "chunk owner kernel currently specializes chunk_size=16, got ", chunk_size);

  TORCH_CHECK(dout.dim() == 4, "dout must have shape [B, S, H, P]");
  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  TORCH_CHECK(v.sizes() == dout.sizes(), "v shape mismatch");
  TORCH_CHECK(q_bias.dim() == 3, "q_bias must have shape [H, R, N]");
  TORCH_CHECK(k_bias.sizes() == q_bias.sizes(), "k_bias shape mismatch");
  TORCH_CHECK(mimo_v.sizes() == at::IntArrayRef({H, q_bias.size(1), P}), "mimo_v shape mismatch");
  TORCH_CHECK(mimo_o.sizes() == mimo_v.sizes(), "mimo_o shape mismatch");

  const int R = static_cast<int>(q_bias.size(1));
  const int N = static_cast<int>(q_bias.size(2));
  TORCH_CHECK(R == kR, "chunk owner CUDA kernel currently specializes R=4, got R=", R);
  TORCH_CHECK(q_flat.dim() == 4, "q_flat must have shape [B, S*R, G, N]");
  const int G = static_cast<int>(q_flat.size(2));
  TORCH_CHECK(G > 0 && H % G == 0, "H must be divisible by G");
  TORCH_CHECK(q_flat.sizes() == at::IntArrayRef({B, S * R, G, N}), "q_flat shape mismatch");
  TORCH_CHECK(k_flat.sizes() == q_flat.sizes(), "k_flat shape mismatch");
  TORCH_CHECK(qk_dot.sizes() == at::IntArrayRef({B, H, S, R * R}), "qk_dot must have shape [B, H, S, R*R]");
  TORCH_CHECK(dt.sizes() == at::IntArrayRef({B, H, S}), "dt shape mismatch");
  TORCH_CHECK(trap.sizes() == at::IntArrayRef({B, H, S}), "trap shape mismatch");
}

void stage2_rr_diag_chunk_owner_out(
    const at::Tensor& dout,
    const at::Tensor& q_flat,
    const at::Tensor& k_flat,
    const at::Tensor& v,
    const at::Tensor& q_bias,
    const at::Tensor& k_bias,
    const at::Tensor& mimo_v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    const at::Tensor& dgamma_diag,
    const at::Tensor& dk_delta,
    const at::Tensor& dq_delta,
    int chunk_size) {
  validate_stage2_chunk_inputs(
      dout, q_flat, k_flat, v, q_bias, k_bias, mimo_v, mimo_o, qk_dot, dt, trap, chunk_size);
  check_input(dk_delta, "dk_delta", dout.scalar_type());
  check_input(dq_delta, "dq_delta", dout.scalar_type());
  TORCH_CHECK(dgamma_diag.is_cuda() && dgamma_diag.is_contiguous(), "dgamma_diag must be contiguous CUDA");
  TORCH_CHECK(dgamma_diag.scalar_type() == at::kFloat, "dgamma_diag must be fp32");

  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  const int R = static_cast<int>(q_bias.size(1));
  const int N = static_cast<int>(q_bias.size(2));
  const int G = static_cast<int>(q_flat.size(2));
  TORCH_CHECK(dk_delta.sizes() == at::IntArrayRef({B, S * R, H, N}), "dk_delta shape mismatch");
  TORCH_CHECK(dq_delta.sizes() == dk_delta.sizes(), "dq_delta shape mismatch");
  TORCH_CHECK(dgamma_diag.sizes() == at::IntArrayRef({B, H, S}), "dgamma_diag shape mismatch");

  const int nchunks = (S + chunk_size - 1) / chunk_size;
  const int64_t total_programs = static_cast<int64_t>(B) * H * nchunks;
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  const size_t smem_bytes = sizeof(float) * (kDqkElems * kThreads + kDqkElems);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "stage2_rr_diag_chunk_owner_out", [&] {
    stage2_rr_diag_chunk_owner_kernel<scalar_t><<<grid, block, smem_bytes, stream>>>(
        dout.data_ptr<scalar_t>(),
        q_flat.data_ptr<scalar_t>(),
        k_flat.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        q_bias.data_ptr<float>(),
        k_bias.data_ptr<float>(),
        mimo_v.data_ptr<float>(),
        mimo_o.data_ptr<float>(),
        qk_dot.data_ptr<scalar_t>(),
        dt.data_ptr<float>(),
        trap.data_ptr<scalar_t>(),
        dgamma_diag.data_ptr<float>(),
        dk_delta.data_ptr<scalar_t>(),
        dq_delta.data_ptr<scalar_t>(),
        total_programs,
        B,
        S,
        H,
        G,
        N,
        P,
        R,
        nchunks,
        chunk_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<at::Tensor> stage2_rr_diag_chunk_owner(
    const at::Tensor& dout,
    const at::Tensor& q_flat,
    const at::Tensor& k_flat,
    const at::Tensor& v,
    const at::Tensor& q_bias,
    const at::Tensor& k_bias,
    const at::Tensor& mimo_v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    int chunk_size) {
  validate_stage2_chunk_inputs(
      dout, q_flat, k_flat, v, q_bias, k_bias, mimo_v, mimo_o, qk_dot, dt, trap, chunk_size);

  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int R = static_cast<int>(q_bias.size(1));
  const int N = static_cast<int>(q_bias.size(2));
  auto f32_opts = dout.options().dtype(at::kFloat);
  at::Tensor dgamma_diag = at::empty({B, H, S}, f32_opts);
  at::Tensor dk_delta = at::empty({B, S * R, H, N}, dout.options());
  at::Tensor dq_delta = at::empty_like(dk_delta);
  stage2_rr_diag_chunk_owner_out(
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
      dgamma_diag,
      dk_delta,
      dq_delta,
      chunk_size);
  return {dgamma_diag, dk_delta, dq_delta};
}

void stage2_rr_diag_chunk_warp_owner_out(
    const at::Tensor& dout,
    const at::Tensor& q_flat,
    const at::Tensor& k_flat,
    const at::Tensor& v,
    const at::Tensor& q_bias,
    const at::Tensor& k_bias,
    const at::Tensor& mimo_v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    const at::Tensor& dgamma_diag,
    const at::Tensor& dk_delta,
    const at::Tensor& dq_delta,
    int chunk_size) {
  validate_stage2_chunk_inputs(
      dout, q_flat, k_flat, v, q_bias, k_bias, mimo_v, mimo_o, qk_dot, dt, trap, chunk_size);
  check_input(dk_delta, "dk_delta", dout.scalar_type());
  check_input(dq_delta, "dq_delta", dout.scalar_type());
  TORCH_CHECK(dgamma_diag.is_cuda() && dgamma_diag.is_contiguous(), "dgamma_diag must be contiguous CUDA");
  TORCH_CHECK(dgamma_diag.scalar_type() == at::kFloat, "dgamma_diag must be fp32");

  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  const int R = static_cast<int>(q_bias.size(1));
  const int N = static_cast<int>(q_bias.size(2));
  const int G = static_cast<int>(q_flat.size(2));
  TORCH_CHECK(dk_delta.sizes() == at::IntArrayRef({B, S * R, H, N}), "dk_delta shape mismatch");
  TORCH_CHECK(dq_delta.sizes() == dk_delta.sizes(), "dq_delta shape mismatch");
  TORCH_CHECK(dgamma_diag.sizes() == at::IntArrayRef({B, H, S}), "dgamma_diag shape mismatch");

  const int nchunks = (S + chunk_size - 1) / chunk_size;
  const int64_t total_programs = static_cast<int64_t>(B) * H * nchunks;
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "stage2_rr_diag_chunk_warp_owner_out", [&] {
    stage2_rr_diag_chunk_warp_owner_kernel<scalar_t><<<grid, block, 0, stream>>>(
        dout.data_ptr<scalar_t>(),
        q_flat.data_ptr<scalar_t>(),
        k_flat.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        q_bias.data_ptr<float>(),
        k_bias.data_ptr<float>(),
        mimo_v.data_ptr<float>(),
        mimo_o.data_ptr<float>(),
        qk_dot.data_ptr<scalar_t>(),
        dt.data_ptr<float>(),
        trap.data_ptr<scalar_t>(),
        dgamma_diag.data_ptr<float>(),
        dk_delta.data_ptr<scalar_t>(),
        dq_delta.data_ptr<scalar_t>(),
        total_programs,
        B,
        S,
        H,
        G,
        N,
        P,
        R,
        nchunks,
        chunk_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<at::Tensor> stage2_rr_diag_chunk_warp_owner(
    const at::Tensor& dout,
    const at::Tensor& q_flat,
    const at::Tensor& k_flat,
    const at::Tensor& v,
    const at::Tensor& q_bias,
    const at::Tensor& k_bias,
    const at::Tensor& mimo_v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    int chunk_size) {
  validate_stage2_chunk_inputs(
      dout, q_flat, k_flat, v, q_bias, k_bias, mimo_v, mimo_o, qk_dot, dt, trap, chunk_size);

  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int R = static_cast<int>(q_bias.size(1));
  const int N = static_cast<int>(q_bias.size(2));
  auto f32_opts = dout.options().dtype(at::kFloat);
  at::Tensor dgamma_diag = at::empty({B, H, S}, f32_opts);
  at::Tensor dk_delta = at::empty({B, S * R, H, N}, dout.options());
  at::Tensor dq_delta = at::empty_like(dk_delta);
  stage2_rr_diag_chunk_warp_owner_out(
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
      dgamma_diag,
      dk_delta,
      dq_delta,
      chunk_size);
  return {dgamma_diag, dk_delta, dq_delta};
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

template <typename scalar_t>
py::dict stage2_metadata_for_dtype() {
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, stage2_rr_diag_post_kernel<scalar_t>));

  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const size_t smem_bytes = sizeof(float) * (kDqkElems * kThreads + kDqkElems);
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, stage2_rr_diag_post_kernel<scalar_t>, kThreads, smem_bytes));

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

template <typename scalar_t>
py::dict stage2_chunk_owner_metadata_for_dtype() {
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, stage2_rr_diag_chunk_owner_kernel<scalar_t>));

  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const size_t smem_bytes = sizeof(float) * (kDqkElems * kThreads + kDqkElems);
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, stage2_rr_diag_chunk_owner_kernel<scalar_t>, kThreads, smem_bytes));

  py::dict out;
  out["threads_per_block"] = kThreads;
  out["chunk_size"] = kChunk;
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

py::dict stage2_rr_diag_post_metadata(const at::Tensor& dout) {
  TORCH_CHECK(dout.is_cuda(), "dout must be CUDA");
  py::dict out;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "stage2_rr_diag_post_metadata", [&] {
    out = stage2_metadata_for_dtype<scalar_t>();
  });
  return out;
}

py::dict stage2_rr_diag_chunk_owner_metadata(const at::Tensor& dout) {
  TORCH_CHECK(dout.is_cuda(), "dout must be CUDA");
  py::dict out;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "stage2_rr_diag_chunk_owner_metadata", [&] {
    out = stage2_chunk_owner_metadata_for_dtype<scalar_t>();
  });
  return out;
}

template <typename scalar_t>
py::dict stage2_chunk_warp_owner_metadata_for_dtype() {
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, stage2_rr_diag_chunk_warp_owner_kernel<scalar_t>));

  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, stage2_rr_diag_chunk_warp_owner_kernel<scalar_t>, kThreads, 0));

  py::dict out;
  out["threads_per_block"] = kThreads;
  out["warps_per_block"] = kThreads / 32;
  out["timesteps_per_warp_batch"] = kThreads / 32;
  out["chunk_size"] = kChunk;
  out["dynamic_smem_bytes"] = 0;
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

py::dict stage2_rr_diag_chunk_warp_owner_metadata(const at::Tensor& dout) {
  TORCH_CHECK(dout.is_cuda(), "dout must be CUDA");
  py::dict out;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "stage2_rr_diag_chunk_warp_owner_metadata", [&] {
    out = stage2_chunk_warp_owner_metadata_for_dtype<scalar_t>();
  });
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rr_diag_forward", &rr_diag_forward, "R x R diagonal microkernel forward");
  m.def("rr_diag_kernel_metadata", &rr_diag_kernel_metadata, "R x R diagonal microkernel metadata");
  m.def("stage2_rr_diag_post", &stage2_rr_diag_post, "In-place stage2 R x R diagonal post kernel");
  m.def("stage2_rr_diag_post_metadata", &stage2_rr_diag_post_metadata, "Stage2 R x R diagonal post kernel metadata");
  m.def("stage2_rr_diag_chunk_owner", &stage2_rr_diag_chunk_owner, "Chunk-owner stage2 R x R diagonal kernel");
  m.def(
      "stage2_rr_diag_chunk_owner_out",
      &stage2_rr_diag_chunk_owner_out,
      "In-place-output chunk-owner stage2 R x R diagonal kernel");
  m.def(
      "stage2_rr_diag_chunk_owner_metadata",
      &stage2_rr_diag_chunk_owner_metadata,
      "Chunk-owner stage2 R x R diagonal kernel metadata");
  m.def(
      "stage2_rr_diag_chunk_warp_owner",
      &stage2_rr_diag_chunk_warp_owner,
      "Warp-per-timestep chunk-owner stage2 R x R diagonal kernel");
  m.def(
      "stage2_rr_diag_chunk_warp_owner_out",
      &stage2_rr_diag_chunk_warp_owner_out,
      "In-place-output warp-per-timestep chunk-owner stage2 R x R diagonal kernel");
  m.def(
      "stage2_rr_diag_chunk_warp_owner_metadata",
      &stage2_rr_diag_chunk_warp_owner_metadata,
      "Warp-per-timestep chunk-owner stage2 R x R diagonal kernel metadata");
}
