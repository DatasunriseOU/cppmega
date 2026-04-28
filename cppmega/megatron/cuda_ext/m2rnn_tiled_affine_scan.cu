#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kMaxV = 16;
constexpr int kThreads = 256;

__device__ __forceinline__ int64_t div_floor_i64(int64_t x, int64_t y) {
  return x / y;
}

__global__ __launch_bounds__(kThreads, 2) void m2rnn_tile_summary_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ W,
    const float* __restrict__ xf,
    const float* __restrict__ h_traj,
    const float* __restrict__ h0_row,
    float* __restrict__ tile_A,
    float* __restrict__ tile_b,
    int B,
    int S,
    int H,
    int K,
    int V,
    int tile_size,
    int n_tiles) {
  const int tid = threadIdx.x;
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  const int tile = static_cast<int>(block % n_tiles);
  const int64_t chain = div_floor_i64(block, n_tiles);

  const int k_idx = static_cast<int>(chain % K);
  const int h = static_cast<int>((chain / K) % H);
  const int b = static_cast<int>(chain / (static_cast<int64_t>(H) * K));
  const int start = tile * tile_size;
  const int end = min(start + tile_size, S);

  __shared__ float M[kMaxV * kMaxV];
  __shared__ float M_next[kMaxV * kMaxV];
  __shared__ float P[kMaxV * kMaxV];
  __shared__ float d[kMaxV];
  __shared__ float d_next[kMaxV];
  __shared__ float h_prev[kMaxV];
  __shared__ float z[kMaxV];
  __shared__ float h_new[kMaxV];
  __shared__ float rhs[kMaxV];

  if (tid < V * V) {
    const int row = tid / V;
    const int col = tid - row * V;
    M[tid] = row == col ? 1.0f : 0.0f;
  }
  if (tid < V) {
    d[tid] = 0.0f;
  }
  __syncthreads();

  for (int s = start; s < end; ++s) {
    if (tid < V) {
      const int i = tid;
      if (s == 0) {
        h_prev[i] = h0_row[(chain * V) + i];
      } else {
        h_prev[i] = h_traj[((chain * S + (s - 1)) * V) + i];
      }
    }
    __syncthreads();

    if (tid < V) {
      const int i = tid;
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < kMaxV; ++j) {
        if (j < V) {
          acc += h_prev[j] * W[((h * V + j) * V) + i];
        }
      }
      const float k_val = k[(((static_cast<int64_t>(b) * S + s) * H + h) * K) + k_idx];
      const float v_val = v[(((static_cast<int64_t>(b) * S + s) * H + h) * V) + i];
      z[i] = acc + k_val * v_val;
    }
    __syncthreads();

    if (tid < V) {
      const int i = tid;
      const float f = xf[(static_cast<int64_t>(b) * S + s) * H + h];
      const float h_t = h_traj[((chain * S + s) * V) + i];
      const float tanh_z = tanhf(z[i]);
      h_new[i] = tanh_z;
      rhs[i] = -h_t + f * h_prev[i] + (1.0f - f) * tanh_z;
    }
    __syncthreads();

    if (tid < V * V) {
      const int i = tid / V;
      const int j = tid - i * V;
      const float f = xf[(static_cast<int64_t>(b) * S + s) * H + h];
      const float sech2 = 1.0f - h_new[i] * h_new[i];
      const float wji = W[((h * V + j) * V) + i];
      P[tid] = (i == j ? f : 0.0f) + (1.0f - f) * sech2 * wji;
    }
    __syncthreads();

    if (tid < V) {
      const int i = tid;
      float acc = rhs[i];
#pragma unroll
      for (int j = 0; j < kMaxV; ++j) {
        if (j < V) {
          acc += P[i * V + j] * d[j];
        }
      }
      d_next[i] = acc;
    }

    if (tid < V * V) {
      const int i = tid / V;
      const int j = tid - i * V;
      float acc = 0.0f;
#pragma unroll
      for (int m = 0; m < kMaxV; ++m) {
        if (m < V) {
          acc += P[i * V + m] * M[m * V + j];
        }
      }
      M_next[tid] = acc;
    }
    __syncthreads();

    if (tid < V) {
      d[tid] = d_next[tid];
    }
    if (tid < V * V) {
      M[tid] = M_next[tid];
    }
    __syncthreads();
  }

  if (tid < V) {
    tile_b[((chain * n_tiles + tile) * V) + tid] = d[tid];
  }
  if (tid < V * V) {
    tile_A[(((chain * n_tiles + tile) * V * V) + tid)] = M[tid];
  }
}

__global__ __launch_bounds__(kThreads, 2) void m2rnn_apply_tile_prefix_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ W,
    const float* __restrict__ xf,
    const float* __restrict__ h_traj,
    const float* __restrict__ h0_row,
    const float* __restrict__ tile_inputs,
    float* __restrict__ delta,
    int B,
    int S,
    int H,
    int K,
    int V,
    int tile_size,
    int n_tiles) {
  const int tid = threadIdx.x;
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  const int tile = static_cast<int>(block % n_tiles);
  const int64_t chain = div_floor_i64(block, n_tiles);

  const int k_idx = static_cast<int>(chain % K);
  const int h = static_cast<int>((chain / K) % H);
  const int b = static_cast<int>(chain / (static_cast<int64_t>(H) * K));
  const int start = tile * tile_size;
  const int end = min(start + tile_size, S);

  __shared__ float M[kMaxV * kMaxV];
  __shared__ float M_next[kMaxV * kMaxV];
  __shared__ float P[kMaxV * kMaxV];
  __shared__ float d[kMaxV];
  __shared__ float d_next[kMaxV];
  __shared__ float carry[kMaxV];
  __shared__ float h_prev[kMaxV];
  __shared__ float z[kMaxV];
  __shared__ float h_new[kMaxV];
  __shared__ float rhs[kMaxV];

  if (tid < V * V) {
    const int row = tid / V;
    const int col = tid - row * V;
    M[tid] = row == col ? 1.0f : 0.0f;
  }
  if (tid < V) {
    d[tid] = 0.0f;
    carry[tid] = tile_inputs[((chain * n_tiles + tile) * V) + tid];
  }
  __syncthreads();

  for (int s = start; s < end; ++s) {
    if (tid < V) {
      const int i = tid;
      if (s == 0) {
        h_prev[i] = h0_row[(chain * V) + i];
      } else {
        h_prev[i] = h_traj[((chain * S + (s - 1)) * V) + i];
      }
    }
    __syncthreads();

    if (tid < V) {
      const int i = tid;
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < kMaxV; ++j) {
        if (j < V) {
          acc += h_prev[j] * W[((h * V + j) * V) + i];
        }
      }
      const float k_val = k[(((static_cast<int64_t>(b) * S + s) * H + h) * K) + k_idx];
      const float v_val = v[(((static_cast<int64_t>(b) * S + s) * H + h) * V) + i];
      z[i] = acc + k_val * v_val;
    }
    __syncthreads();

    if (tid < V) {
      const int i = tid;
      const float f = xf[(static_cast<int64_t>(b) * S + s) * H + h];
      const float h_t = h_traj[((chain * S + s) * V) + i];
      const float tanh_z = tanhf(z[i]);
      h_new[i] = tanh_z;
      rhs[i] = -h_t + f * h_prev[i] + (1.0f - f) * tanh_z;
    }
    __syncthreads();

    if (tid < V * V) {
      const int i = tid / V;
      const int j = tid - i * V;
      const float f = xf[(static_cast<int64_t>(b) * S + s) * H + h];
      const float sech2 = 1.0f - h_new[i] * h_new[i];
      const float wji = W[((h * V + j) * V) + i];
      P[tid] = (i == j ? f : 0.0f) + (1.0f - f) * sech2 * wji;
    }
    __syncthreads();

    if (tid < V) {
      const int i = tid;
      float acc = rhs[i];
#pragma unroll
      for (int j = 0; j < kMaxV; ++j) {
        if (j < V) {
          acc += P[i * V + j] * d[j];
        }
      }
      d_next[i] = acc;
    }

    if (tid < V * V) {
      const int i = tid / V;
      const int j = tid - i * V;
      float acc = 0.0f;
#pragma unroll
      for (int m = 0; m < kMaxV; ++m) {
        if (m < V) {
          acc += P[i * V + m] * M[m * V + j];
        }
      }
      M_next[tid] = acc;
    }
    __syncthreads();

    if (tid < V) {
      const int i = tid;
      float prefix = 0.0f;
#pragma unroll
      for (int j = 0; j < kMaxV; ++j) {
        if (j < V) {
          prefix += M_next[i * V + j] * carry[j];
        }
      }
      d[tid] = d_next[tid];
      delta[((chain * S + s) * V) + tid] = d_next[tid] + prefix;
    }
    if (tid < V * V) {
      M[tid] = M_next[tid];
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kThreads, 2) void m2rnn_scan_tile_summaries_kernel(
    const float* __restrict__ tile_A,
    const float* __restrict__ tile_b,
    float* __restrict__ tile_inputs,
    int V,
    int n_tiles) {
  const int tid = threadIdx.x;
  const int64_t chain = static_cast<int64_t>(blockIdx.x);

  __shared__ float carry[kMaxV];
  __shared__ float next[kMaxV];

  if (tid < V) {
    carry[tid] = 0.0f;
  }
  __syncthreads();

  for (int tile = 0; tile < n_tiles; ++tile) {
    const int64_t tile_base = (chain * static_cast<int64_t>(n_tiles) + tile);
    if (tid < V) {
      tile_inputs[tile_base * V + tid] = carry[tid];
    }
    __syncthreads();

    if (tid < V) {
      const int i = tid;
      float acc = tile_b[tile_base * V + i];
#pragma unroll
      for (int j = 0; j < kMaxV; ++j) {
        if (j < V) {
          acc += tile_A[(tile_base * V * V) + (i * V + j)] * carry[j];
        }
      }
      next[i] = acc;
    }
    __syncthreads();

    if (tid < V) {
      carry[tid] = next[tid];
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kThreads, 2) void m2rnn_local_tile_scan_debug_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ W,
    const float* __restrict__ xf,
    const float* __restrict__ h_traj,
    const float* __restrict__ h0_row,
    float* __restrict__ local_delta,
    float* __restrict__ local_prefix,
    float* __restrict__ tile_A,
    float* __restrict__ tile_b,
    int B,
    int S,
    int H,
    int K,
    int V,
    int tile_size,
    int n_tiles) {
  const int tid = threadIdx.x;
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  const int tile = static_cast<int>(block % n_tiles);
  const int64_t chain = div_floor_i64(block, n_tiles);

  const int k_idx = static_cast<int>(chain % K);
  const int h = static_cast<int>((chain / K) % H);
  const int b = static_cast<int>(chain / (static_cast<int64_t>(H) * K));
  const int start = tile * tile_size;
  const int end = min(start + tile_size, S);

  __shared__ float M[kMaxV * kMaxV];
  __shared__ float M_next[kMaxV * kMaxV];
  __shared__ float P[kMaxV * kMaxV];
  __shared__ float d[kMaxV];
  __shared__ float d_next[kMaxV];
  __shared__ float h_prev[kMaxV];
  __shared__ float z[kMaxV];
  __shared__ float h_new[kMaxV];
  __shared__ float rhs[kMaxV];

  if (tid < V * V) {
    const int row = tid / V;
    const int col = tid - row * V;
    M[tid] = row == col ? 1.0f : 0.0f;
  }
  if (tid < V) {
    d[tid] = 0.0f;
  }
  __syncthreads();

  for (int s = start; s < end; ++s) {
    if (tid < V) {
      const int i = tid;
      if (s == 0) {
        h_prev[i] = h0_row[(chain * V) + i];
      } else {
        h_prev[i] = h_traj[((chain * S + (s - 1)) * V) + i];
      }
    }
    __syncthreads();

    if (tid < V) {
      const int i = tid;
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < kMaxV; ++j) {
        if (j < V) {
          acc += h_prev[j] * W[((h * V + j) * V) + i];
        }
      }
      const float k_val = k[(((static_cast<int64_t>(b) * S + s) * H + h) * K) + k_idx];
      const float v_val = v[(((static_cast<int64_t>(b) * S + s) * H + h) * V) + i];
      z[i] = acc + k_val * v_val;
    }
    __syncthreads();

    if (tid < V) {
      const int i = tid;
      const float f = xf[(static_cast<int64_t>(b) * S + s) * H + h];
      const float h_t = h_traj[((chain * S + s) * V) + i];
      const float tanh_z = tanhf(z[i]);
      h_new[i] = tanh_z;
      rhs[i] = -h_t + f * h_prev[i] + (1.0f - f) * tanh_z;
    }
    __syncthreads();

    if (tid < V * V) {
      const int i = tid / V;
      const int j = tid - i * V;
      const float f = xf[(static_cast<int64_t>(b) * S + s) * H + h];
      const float sech2 = 1.0f - h_new[i] * h_new[i];
      const float wji = W[((h * V + j) * V) + i];
      P[tid] = (i == j ? f : 0.0f) + (1.0f - f) * sech2 * wji;
    }
    __syncthreads();

    if (tid < V) {
      const int i = tid;
      float acc = rhs[i];
#pragma unroll
      for (int j = 0; j < kMaxV; ++j) {
        if (j < V) {
          acc += P[i * V + j] * d[j];
        }
      }
      d_next[i] = acc;
    }

    if (tid < V * V) {
      const int i = tid / V;
      const int j = tid - i * V;
      float acc = 0.0f;
#pragma unroll
      for (int m = 0; m < kMaxV; ++m) {
        if (m < V) {
          acc += P[i * V + m] * M[m * V + j];
        }
      }
      M_next[tid] = acc;
    }
    __syncthreads();

    if (tid < V) {
      d[tid] = d_next[tid];
      local_delta[((chain * S + s) * V) + tid] = d_next[tid];
    }
    if (tid < V * V) {
      M[tid] = M_next[tid];
      local_prefix[(((chain * S + s) * V * V) + tid)] = M_next[tid];
    }
    __syncthreads();
  }

  if (tid < V) {
    tile_b[((chain * n_tiles + tile) * V) + tid] = d[tid];
  }
  if (tid < V * V) {
    tile_A[(((chain * n_tiles + tile) * V * V) + tid)] = M[tid];
  }
}

void check_float_cuda_contiguous(const at::Tensor& t, const char* name) {
  if (!t.is_cuda()) {
    throw std::invalid_argument(std::string(name) + " must be a CUDA tensor");
  }
  if (t.scalar_type() != at::kFloat) {
    throw std::invalid_argument(std::string(name) + " must be float32");
  }
  if (!t.is_contiguous()) {
    throw std::invalid_argument(std::string(name) + " must be contiguous");
  }
}

int64_t div_up(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

void check_problem_shapes(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& W,
    const at::Tensor& xf,
    const at::Tensor& h_traj,
    const at::Tensor& h0_row,
    int64_t tile_size) {
  check_float_cuda_contiguous(q, "q");
  check_float_cuda_contiguous(k, "k");
  check_float_cuda_contiguous(v, "v");
  check_float_cuda_contiguous(W, "W");
  check_float_cuda_contiguous(xf, "xf");
  check_float_cuda_contiguous(h_traj, "h_traj");
  check_float_cuda_contiguous(h0_row, "h0_row");

  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || W.dim() != 3 || xf.dim() != 3 ||
      h_traj.dim() != 3 || h0_row.dim() != 2) {
    throw std::invalid_argument("unexpected tensor rank for m2rnn local tile scan");
  }

  const int64_t B = q.size(0);
  const int64_t S = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K = q.size(3);
  const int64_t V = v.size(3);
  if (k.size(0) != B || k.size(1) != S || k.size(2) != H || k.size(3) != K) {
    throw std::invalid_argument("k shape must match q after head broadcast");
  }
  if (v.size(0) != B || v.size(1) != S || v.size(2) != H) {
    throw std::invalid_argument("v shape must match q batch/seq/head after head broadcast");
  }
  if (W.size(0) != H || W.size(1) != V || W.size(2) != V) {
    throw std::invalid_argument("W shape must be (H, V, V)");
  }
  if (xf.size(0) != B || xf.size(1) != S || xf.size(2) != H) {
    throw std::invalid_argument("xf shape must be (B, S, H)");
  }
  const int64_t Be = B * H * K;
  if (h_traj.size(0) != Be || h_traj.size(1) != S || h_traj.size(2) != V) {
    throw std::invalid_argument("h_traj shape must be (B*H*K, S, V)");
  }
  if (h0_row.size(0) != Be || h0_row.size(1) != V) {
    throw std::invalid_argument("h0_row shape must be (B*H*K, V)");
  }
  if (V < 1 || V > kMaxV) {
    throw std::invalid_argument("m2rnn tiled CUDA path requires 1 <= V <= 16");
  }
  if (tile_size < 1) {
    throw std::invalid_argument("tile_size must be positive");
  }
}

void check_tile_summary_shapes(
    const at::Tensor& tile_A,
    const at::Tensor& tile_b,
    int64_t Be,
    int64_t n_tiles,
    int64_t V) {
  check_float_cuda_contiguous(tile_A, "tile_A");
  check_float_cuda_contiguous(tile_b, "tile_b");
  if (tile_A.dim() != 4 || tile_A.size(0) != Be || tile_A.size(1) != n_tiles ||
      tile_A.size(2) != V || tile_A.size(3) != V) {
    throw std::invalid_argument("tile_A shape must be (B*H*K, n_tiles, V, V)");
  }
  if (tile_b.dim() != 3 || tile_b.size(0) != Be || tile_b.size(1) != n_tiles ||
      tile_b.size(2) != V) {
    throw std::invalid_argument("tile_b shape must be (B*H*K, n_tiles, V)");
  }
  if (tile_A.device() != tile_b.device()) {
    throw std::invalid_argument("tile_A and tile_b must be on the same CUDA device");
  }
}

void check_tile_inputs_shape(
    const at::Tensor& tile_inputs,
    int64_t Be,
    int64_t n_tiles,
    int64_t V) {
  check_float_cuda_contiguous(tile_inputs, "tile_inputs");
  if (tile_inputs.dim() != 3 || tile_inputs.size(0) != Be || tile_inputs.size(1) != n_tiles ||
      tile_inputs.size(2) != V) {
    throw std::invalid_argument("tile_inputs shape must be (B*H*K, n_tiles, V)");
  }
}

void check_delta_shape(const at::Tensor& delta, int64_t Be, int64_t S, int64_t V) {
  check_float_cuda_contiguous(delta, "delta");
  if (delta.dim() != 3 || delta.size(0) != Be || delta.size(1) != S || delta.size(2) != V) {
    throw std::invalid_argument("delta shape must be (B*H*K, S, V)");
  }
}

void tile_summaries_out(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& W,
    const at::Tensor& xf,
    const at::Tensor& h_traj,
    const at::Tensor& h0_row,
    const at::Tensor& tile_A,
    const at::Tensor& tile_b,
    int64_t tile_size) {
  check_problem_shapes(q, k, v, W, xf, h_traj, h0_row, tile_size);

  const int64_t B = q.size(0);
  const int64_t S = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K = q.size(3);
  const int64_t V = v.size(3);

  const int64_t n_tiles = div_up(S, tile_size);
  const int64_t Be = B * H * K;
  check_tile_summary_shapes(tile_A, tile_b, Be, n_tiles, V);

  const c10::cuda::CUDAGuard device_guard(q.device());
  const int64_t blocks = Be * n_tiles;
  m2rnn_tile_summary_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      W.data_ptr<float>(),
      xf.data_ptr<float>(),
      h_traj.data_ptr<float>(),
      h0_row.data_ptr<float>(),
      tile_A.data_ptr<float>(),
      tile_b.data_ptr<float>(),
      static_cast<int>(B),
      static_cast<int>(S),
      static_cast<int>(H),
      static_cast<int>(K),
      static_cast<int>(V),
      static_cast<int>(tile_size),
      static_cast<int>(n_tiles));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<at::Tensor> tile_summaries(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& W,
    const at::Tensor& xf,
    const at::Tensor& h_traj,
    const at::Tensor& h0_row,
    int64_t tile_size) {
  check_problem_shapes(q, k, v, W, xf, h_traj, h0_row, tile_size);

  const int64_t B = q.size(0);
  const int64_t S = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K = q.size(3);
  const int64_t V = v.size(3);
  const int64_t n_tiles = div_up(S, tile_size);
  const int64_t Be = B * H * K;
  auto opts = q.options();
  at::Tensor tile_A = at::empty({Be, n_tiles, V, V}, opts);
  at::Tensor tile_b = at::empty({Be, n_tiles, V}, opts);
  tile_summaries_out(q, k, v, W, xf, h_traj, h0_row, tile_A, tile_b, tile_size);

  return {tile_A, tile_b};
}

void apply_tile_prefixes_out(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& W,
    const at::Tensor& xf,
    const at::Tensor& h_traj,
    const at::Tensor& h0_row,
    const at::Tensor& tile_inputs,
    const at::Tensor& delta,
    int64_t tile_size) {
  check_problem_shapes(q, k, v, W, xf, h_traj, h0_row, tile_size);

  const int64_t B = q.size(0);
  const int64_t S = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K = q.size(3);
  const int64_t V = v.size(3);
  const int64_t Be = B * H * K;
  const int64_t n_tiles = div_up(S, tile_size);
  check_tile_inputs_shape(tile_inputs, Be, n_tiles, V);
  check_delta_shape(delta, Be, S, V);

  const c10::cuda::CUDAGuard device_guard(q.device());
  const int64_t blocks = Be * n_tiles;
  m2rnn_apply_tile_prefix_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      W.data_ptr<float>(),
      xf.data_ptr<float>(),
      h_traj.data_ptr<float>(),
      h0_row.data_ptr<float>(),
      tile_inputs.data_ptr<float>(),
      delta.data_ptr<float>(),
      static_cast<int>(B),
      static_cast<int>(S),
      static_cast<int>(H),
      static_cast<int>(K),
      static_cast<int>(V),
      static_cast<int>(tile_size),
      static_cast<int>(n_tiles));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor apply_tile_prefixes(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& W,
    const at::Tensor& xf,
    const at::Tensor& h_traj,
    const at::Tensor& h0_row,
    const at::Tensor& tile_inputs,
    int64_t tile_size) {
  check_problem_shapes(q, k, v, W, xf, h_traj, h0_row, tile_size);

  const int64_t B = q.size(0);
  const int64_t S = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K = q.size(3);
  const int64_t V = v.size(3);
  const int64_t Be = B * H * K;
  at::Tensor delta = at::empty({Be, S, V}, q.options());
  apply_tile_prefixes_out(q, k, v, W, xf, h_traj, h0_row, tile_inputs, delta, tile_size);

  return delta;
}

void scan_tile_summaries_out(
    const at::Tensor& tile_A,
    const at::Tensor& tile_b,
    const at::Tensor& tile_inputs) {
  check_float_cuda_contiguous(tile_A, "tile_A");
  check_float_cuda_contiguous(tile_b, "tile_b");
  if (tile_A.device() != tile_b.device()) {
    throw std::invalid_argument("tile_A and tile_b must be on the same CUDA device");
  }
  if (tile_A.dim() != 4 || tile_b.dim() != 3) {
    throw std::invalid_argument("tile_A/tile_b ranks must be (Be,n_tiles,V,V)/(Be,n_tiles,V)");
  }

  const int64_t Be = tile_b.size(0);
  const int64_t n_tiles = tile_b.size(1);
  const int64_t V = tile_b.size(2);
  if (tile_A.size(0) != Be || tile_A.size(1) != n_tiles || tile_A.size(2) != V ||
      tile_A.size(3) != V) {
    throw std::invalid_argument("tile_A shape must be (Be, n_tiles, V, V)");
  }
  if (V < 1 || V > kMaxV) {
    throw std::invalid_argument("m2rnn tiled CUDA path requires 1 <= V <= 16");
  }
  if (Be < 1 || n_tiles < 1) {
    throw std::invalid_argument("tile_A/tile_b must contain at least one chain and one tile");
  }
  check_tile_inputs_shape(tile_inputs, Be, n_tiles, V);

  const c10::cuda::CUDAGuard device_guard(tile_A.device());
  m2rnn_scan_tile_summaries_kernel<<<
      static_cast<unsigned int>(Be),
      kThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      tile_A.data_ptr<float>(),
      tile_b.data_ptr<float>(),
      tile_inputs.data_ptr<float>(),
      static_cast<int>(V),
      static_cast<int>(n_tiles));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor scan_tile_summaries(
    const at::Tensor& tile_A,
    const at::Tensor& tile_b) {
  check_float_cuda_contiguous(tile_b, "tile_b");
  at::Tensor tile_inputs = at::empty_like(tile_b);
  scan_tile_summaries_out(tile_A, tile_b, tile_inputs);

  return tile_inputs;
}

std::vector<at::Tensor> local_tile_scan_debug(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& W,
    const at::Tensor& xf,
    const at::Tensor& h_traj,
    const at::Tensor& h0_row,
    int64_t tile_size) {
  check_problem_shapes(q, k, v, W, xf, h_traj, h0_row, tile_size);

  const int64_t B = q.size(0);
  const int64_t S = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K = q.size(3);
  const int64_t V = v.size(3);
  const int64_t Be = B * H * K;
  const int64_t n_tiles = div_up(S, tile_size);
  auto opts = q.options();
  at::Tensor local_delta = at::empty({Be, S, V}, opts);
  at::Tensor local_prefix = at::empty({Be, S, V, V}, opts);
  at::Tensor tile_A = at::empty({Be, n_tiles, V, V}, opts);
  at::Tensor tile_b = at::empty({Be, n_tiles, V}, opts);

  const c10::cuda::CUDAGuard device_guard(q.device());
  const int64_t blocks = Be * n_tiles;
  m2rnn_local_tile_scan_debug_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      W.data_ptr<float>(),
      xf.data_ptr<float>(),
      h_traj.data_ptr<float>(),
      h0_row.data_ptr<float>(),
      local_delta.data_ptr<float>(),
      local_prefix.data_ptr<float>(),
      tile_A.data_ptr<float>(),
      tile_b.data_ptr<float>(),
      static_cast<int>(B),
      static_cast<int>(S),
      static_cast<int>(H),
      static_cast<int>(K),
      static_cast<int>(V),
      static_cast<int>(tile_size),
      static_cast<int>(n_tiles));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {local_delta, local_prefix, tile_A, tile_b};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tile_summaries", &tile_summaries, "M2RNN local tile summaries (CUDA)");
  m.def("tile_summaries_out", &tile_summaries_out, "M2RNN local tile summaries into preallocated outputs (CUDA)");
  m.def("scan_tile_summaries", &scan_tile_summaries, "M2RNN tile summary prefix scan (CUDA)");
  m.def("scan_tile_summaries_out", &scan_tile_summaries_out, "M2RNN tile summary prefix scan into preallocated output (CUDA)");
  m.def("apply_tile_prefixes", &apply_tile_prefixes, "M2RNN recompute tile-prefix apply (CUDA)");
  m.def("apply_tile_prefixes_out", &apply_tile_prefixes_out, "M2RNN recompute tile-prefix apply into preallocated output (CUDA)");
  m.def("local_tile_scan_debug", &local_tile_scan_debug, "M2RNN local tiled affine scan debug (CUDA)");
}
