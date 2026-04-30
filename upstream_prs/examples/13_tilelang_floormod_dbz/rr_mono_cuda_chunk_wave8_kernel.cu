#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kR = 4;
constexpr int kChunk = 16;
constexpr int kFusedChunk = kChunk * kR;
constexpr int kPPanel = 64;
#ifndef RR_DIAG_THREADS
#define RR_DIAG_THREADS 256
#endif
constexpr int kThreads = RR_DIAG_THREADS;
constexpr int kWarps = kThreads / 32;
static_assert(kThreads > 0 && kThreads % 32 == 0, "RR_DIAG_THREADS must be a positive warp multiple");

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float value) {
  return __float2bfloat16_rn(value);
}

__device__ void wmma_gemm_bf16_row_row(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    float* __restrict__ c,
    int m_tiles,
    int n_tiles,
    int k_tiles,
    int lda,
    int ldb,
    int ldc,
    bool accumulate) {
  using namespace nvcuda;
  constexpr int kWmma = 16;
  const int warp_id = threadIdx.x / 32;
  const int warp_count = blockDim.x / 32;
  const int total_tiles = m_tiles * n_tiles;
  for (int tile = warp_id; tile < total_tiles; tile += warp_count) {
    const int tile_m = tile / n_tiles;
    const int tile_n = tile - tile_m * n_tiles;
    wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float> acc_frag;
    if (accumulate) {
      wmma::load_matrix_sync(
          acc_frag,
          c + (tile_m * kWmma) * ldc + tile_n * kWmma,
          ldc,
          wmma::mem_row_major);
    } else {
      wmma::fill_fragment(acc_frag, 0.0f);
    }
    for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
      wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, __nv_bfloat16, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, __nv_bfloat16, wmma::row_major> b_frag;
      wmma::load_matrix_sync(a_frag, a + (tile_m * kWmma) * lda + tile_k * kWmma, lda);
      wmma::load_matrix_sync(b_frag, b + (tile_k * kWmma) * ldb + tile_n * kWmma, ldb);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    wmma::store_matrix_sync(
        c + (tile_m * kWmma) * ldc + tile_n * kWmma,
        acc_frag,
        ldc,
        wmma::mem_row_major);
  }
}

__device__ void wmma_gemm_bf16_row_col(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b_col_major,
    float* __restrict__ c,
    int m_tiles,
    int n_tiles,
    int k_tiles,
    int lda,
    int ldb_col_major,
    int ldc,
    bool accumulate) {
  using namespace nvcuda;
  constexpr int kWmma = 16;
  const int warp_id = threadIdx.x / 32;
  const int warp_count = blockDim.x / 32;
  const int total_tiles = m_tiles * n_tiles;
  for (int tile = warp_id; tile < total_tiles; tile += warp_count) {
    const int tile_m = tile / n_tiles;
    const int tile_n = tile - tile_m * n_tiles;
    wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float> acc_frag;
    if (accumulate) {
      wmma::load_matrix_sync(
          acc_frag,
          c + (tile_m * kWmma) * ldc + tile_n * kWmma,
          ldc,
          wmma::mem_row_major);
    } else {
      wmma::fill_fragment(acc_frag, 0.0f);
    }
    for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
      wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, __nv_bfloat16, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, __nv_bfloat16, wmma::col_major> b_frag;
      wmma::load_matrix_sync(a_frag, a + (tile_m * kWmma) * lda + tile_k * kWmma, lda);
      wmma::load_matrix_sync(
          b_frag,
          b_col_major + tile_k * kWmma + (tile_n * kWmma) * ldb_col_major,
          ldb_col_major);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    wmma::store_matrix_sync(
        c + (tile_m * kWmma) * ldc + tile_n * kWmma,
        acc_frag,
        ldc,
        wmma::mem_row_major);
  }
}

__global__ void stage2_mono_wmma_tile_stream_chunk_owner_kernel(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ q_flat,
    const __nv_bfloat16* __restrict__ k_flat,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ q_bias,
    const float* __restrict__ k_bias,
    const float* __restrict__ mimo_v,
    const float* __restrict__ mimo_o,
    const __nv_bfloat16* __restrict__ dstates,
    const float* __restrict__ da_cs_rev,
    const float* __restrict__ segsum,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dv_delta,
    float* __restrict__ dmimo_v_chunk_delta,
    float* __restrict__ dssda_delta,
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
  extern __shared__ int4 smem_aligned[];
  unsigned char* cursor = reinterpret_cast<unsigned char*>(smem_aligned);
  __nv_bfloat16* k_s = reinterpret_cast<__nv_bfloat16*>(cursor);
  cursor += static_cast<int64_t>(kFusedChunk) * N * sizeof(__nv_bfloat16);
  __nv_bfloat16* q_t_s = reinterpret_cast<__nv_bfloat16*>(cursor);
  cursor += static_cast<int64_t>(N) * kFusedChunk * sizeof(__nv_bfloat16);
  __nv_bfloat16* dphi_s = reinterpret_cast<__nv_bfloat16*>(cursor);
  cursor += static_cast<int64_t>(kFusedChunk) * kPPanel * sizeof(__nv_bfloat16);
  __nv_bfloat16* psi_s = reinterpret_cast<__nv_bfloat16*>(cursor);
  cursor += static_cast<int64_t>(kFusedChunk) * kPPanel * sizeof(__nv_bfloat16);
  cursor = reinterpret_cast<unsigned char*>(
      (reinterpret_cast<uintptr_t>(cursor) + alignof(float) - 1) &
      ~(static_cast<uintptr_t>(alignof(float) - 1)));
  float* accum_s = reinterpret_cast<float*>(cursor);
  cursor += static_cast<int64_t>(kFusedChunk) * kPPanel * sizeof(float);
  float* lkq_tile_s = reinterpret_cast<float*>(cursor);
  cursor += 16 * 16 * sizeof(float);
  __nv_bfloat16* lkq_masked_tile_s = reinterpret_cast<__nv_bfloat16*>(cursor);

  const int tid = threadIdx.x;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || N != 64 || R != kR || chunk_size != kChunk ||
      blockDim.x != kThreads || (P % kPPanel) != 0) {
    return;
  }

  const int p_panels = P / kPPanel;
  const int p_panel = static_cast<int>(pid % p_panels);
  const int p_base = p_panel * kPPanel;
  const int64_t chunk_pid = pid / p_panels;
  const int chunk = static_cast<int>(chunk_pid % nchunks);
  const int64_t bh = chunk_pid / nchunks;
  const int h = static_cast<int>(bh % H);
  const int b = static_cast<int>(bh / H);
  if (b >= B) {
    return;
  }

  const int h_per_group = H / G;
  const int h_qk = h / h_per_group;
  const int chunk_start = chunk * kChunk;
  constexpr int p_tiles = kPPanel / 16;
  const int64_t qk_flat_base = static_cast<int64_t>(b) * S * R * G * N + h_qk * N;
  const int64_t bias_base = static_cast<int64_t>(h) * R * N;
  const int64_t mimo_base = static_cast<int64_t>(h) * R * P;
  const int64_t dstates_base = (((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * N) * P;
  const int64_t seg_base = (((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * kChunk) * kChunk;
  const float d_value = D[h];

  for (int idx = tid; idx < kFusedChunk * N; idx += blockDim.x) {
    const int row = idx / N;
    const int n = idx - row * N;
    const int cs = row / kR;
    const int r = row - cs * kR;
    const int s = chunk_start + cs;
    float k_val = 0.0f;
    float q_val = 0.0f;
    if (s < S) {
      const int64_t qk_row = qk_flat_base + (static_cast<int64_t>(s) * R + r) * G * N;
      const int64_t bias_row = bias_base + static_cast<int64_t>(r) * N;
      k_val = bf16_to_float(k_flat[qk_row + n]) + k_bias[bias_row + n];
      q_val = bf16_to_float(q_flat[qk_row + n]) + q_bias[bias_row + n];
    }
    k_s[row * N + n] = float_to_bf16(k_val);
    q_t_s[n * kFusedChunk + row] = float_to_bf16(q_val);
  }

  for (int idx = tid; idx < kFusedChunk * kPPanel; idx += blockDim.x) {
    const int row = idx / kPPanel;
    const int p_local = idx - row * kPPanel;
    const int p = p_base + p_local;
    const int cs = row / kR;
    const int r = row - cs * kR;
    const int s = chunk_start + cs;
    float dphi = 0.0f;
    float psi = 0.0f;
    if (s < S) {
      const int64_t hp = ((static_cast<int64_t>(b) * S + s) * H + h) * P + p;
      const int64_t mimo = mimo_base + static_cast<int64_t>(r) * P + p;
      dphi = bf16_to_float(dout[hp]) * mimo_o[mimo];
      psi = bf16_to_float(v[hp]) * mimo_v[mimo];
    }
    dphi_s[idx] = float_to_bf16(dphi);
    psi_s[idx] = float_to_bf16(psi);
  }
  __syncthreads();

  wmma_gemm_bf16_row_col(psi_s, dphi_s, accum_s, 4, 4, p_tiles, kPPanel, kPPanel, kFusedChunk, false);
  __syncthreads();

  for (int tile_m = 0; tile_m < 4; ++tile_m) {
    for (int tile_n = 0; tile_n < 4; ++tile_n) {
      wmma_gemm_bf16_row_row(
          k_s + tile_m * 16 * N,
          q_t_s + tile_n * 16,
          lkq_tile_s,
          1,
          1,
          4,
          N,
          kFusedChunk,
          16,
          false);
      __syncthreads();

      for (int idx = tid; idx < 4 * 4; idx += blockDim.x) {
        const int local_cs_i = idx / 4;
        const int local_cs_j = idx - local_cs_i * 4;
        const int cs_i = tile_m * 4 + local_cs_i;
        const int cs_j = tile_n * 4 + local_cs_j;
        const int s_i = chunk_start + cs_i;
        const int s_j = chunk_start + cs_j;
        float acc = 0.0f;
        if (s_i < S && s_j < S) {
#pragma unroll
          for (int r_i = 0; r_i < kR; ++r_i) {
#pragma unroll
            for (int r_j = 0; r_j < kR; ++r_j) {
              const int local_row = local_cs_i * kR + r_i;
              const int local_col = local_cs_j * kR + r_j;
              const int row = cs_i * kR + r_i;
              const int col = cs_j * kR + r_j;
              acc += lkq_tile_s[local_row * 16 + local_col] *
                  accum_s[row * kFusedChunk + col];
            }
          }
        }
        atomicAdd(dssda_delta + seg_base + cs_i * kChunk + cs_j, acc);
      }
      __syncthreads();
    }
  }

  wmma_gemm_bf16_row_row(k_s, dstates + dstates_base + p_base, accum_s, 4, p_tiles, 4, N, P, kPPanel, false);
  __syncthreads();

  for (int idx = tid; idx < kFusedChunk * kPPanel; idx += blockDim.x) {
    const int row = idx / kPPanel;
    const int cs = row / kR;
    const int s = chunk_start + cs;
    if (s < S) {
      accum_s[idx] *= __expf(da_cs_rev[(static_cast<int64_t>(b) * H + h) * S + s]);
    } else {
      accum_s[idx] = 0.0f;
    }
  }
  __syncthreads();

  for (int tile_m = 0; tile_m < 4; ++tile_m) {
    for (int tile_n = 0; tile_n < 4; ++tile_n) {
      if (tile_n >= tile_m) {
        wmma_gemm_bf16_row_row(
            k_s + tile_m * 16 * N,
            q_t_s + tile_n * 16,
            lkq_tile_s,
            1,
            1,
            4,
            N,
            kFusedChunk,
            16,
            false);
        __syncthreads();

        for (int idx = tid; idx < 16 * 16; idx += blockDim.x) {
          const int local_row = idx / 16;
          const int local_col = idx - local_row * 16;
          const int row = tile_m * 16 + local_row;
          const int col = tile_n * 16 + local_col;
          const int cs_i = row / kR;
          const int cs_j = col / kR;
          float masked = 0.0f;
          if (cs_i < cs_j) {
            masked = lkq_tile_s[idx] * __expf(segsum[seg_base + cs_i * kChunk + cs_j]);
          }
          lkq_masked_tile_s[idx] = float_to_bf16(masked);
        }
        __syncthreads();

        wmma_gemm_bf16_row_row(
            lkq_masked_tile_s,
            dphi_s + tile_n * 16 * kPPanel,
            accum_s + tile_m * 16 * kPPanel,
            1,
            p_tiles,
            1,
            16,
            kPPanel,
            kPPanel,
            true);
        __syncthreads();
      }
    }
  }

  for (int idx = tid; idx < kFusedChunk * kPPanel; idx += blockDim.x) {
    accum_s[idx] += d_value * bf16_to_float(dphi_s[idx]);
  }
  __syncthreads();

  for (int idx = tid; idx < kChunk * kPPanel; idx += blockDim.x) {
    const int cs = idx / kPPanel;
    const int p_local = idx - cs * kPPanel;
    const int p = p_base + p_local;
    const int s = chunk_start + cs;
    if (s < S) {
      float dv = 0.0f;
#pragma unroll
      for (int r = 0; r < kR; ++r) {
        dv += accum_s[(cs * kR + r) * kPPanel + p_local] *
            mimo_v[mimo_base + static_cast<int64_t>(r) * P + p];
      }
      dv_delta[((static_cast<int64_t>(b) * S + s) * H + h) * P + p] = float_to_bf16(dv);
    }
  }

  for (int idx = tid; idx < kR * kPPanel; idx += blockDim.x) {
    const int r = idx / kPPanel;
    const int p_local = idx - r * kPPanel;
    const int p = p_base + p_local;
    float acc = 0.0f;
#pragma unroll
    for (int cs = 0; cs < kChunk; ++cs) {
      const int s = chunk_start + cs;
      if (s < S) {
        const float v_p = bf16_to_float(v[((static_cast<int64_t>(b) * S + s) * H + h) * P + p]);
        acc += accum_s[(cs * kR + r) * kPPanel + p_local] * v_p;
      }
    }
    dmimo_v_chunk_delta[(((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * R + r) * P + p] = acc;
  }
}

void check_input(const at::Tensor& tensor, const char* name, at::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " dtype mismatch");
}

__global__ void reduce_dmimo_chunks_kernel(
    const float* __restrict__ dmimo_v_chunk_delta,
    float* __restrict__ dmimo_v_delta,
    int64_t total_programs,
    int B,
    int H,
    int R,
    int P,
    int nchunks) {
  const int tid = threadIdx.x;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || R != kR) {
    return;
  }

  const int r = static_cast<int>(pid % R);
  const int h = static_cast<int>((pid / R) % H);
  const int b = static_cast<int>(pid / (static_cast<int64_t>(R) * H));
  if (b >= B) {
    return;
  }

  for (int p = tid; p < P; p += blockDim.x) {
    float acc = 0.0f;
    for (int chunk = 0; chunk < nchunks; ++chunk) {
      acc += dmimo_v_chunk_delta[((((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * R + r) * P) + p];
    }
    dmimo_v_delta[(((static_cast<int64_t>(b) * H + h) * R + r) * P) + p] = acc;
  }
}

size_t mono_wmma_tile_stream_smem_bytes(int P) {
  (void)P;
  const size_t bf16_bytes =
      (static_cast<size_t>(kFusedChunk) * 64 +
       static_cast<size_t>(64) * kFusedChunk +
       static_cast<size_t>(kFusedChunk) * kPPanel +
       static_cast<size_t>(kFusedChunk) * kPPanel +
       static_cast<size_t>(16) * 16) *
      sizeof(__nv_bfloat16);
  const size_t float_bytes =
      (static_cast<size_t>(kFusedChunk) * kPPanel +
       static_cast<size_t>(16) * 16) *
      sizeof(float);
  return bf16_bytes + float_bytes + alignof(float);
}

void validate_inputs(
    const at::Tensor& dout,
    const at::Tensor& q_flat,
    const at::Tensor& k_flat,
    const at::Tensor& v,
    const at::Tensor& q_bias,
    const at::Tensor& k_bias,
    const at::Tensor& mimo_v,
    const at::Tensor& mimo_o,
    const at::Tensor& dstates,
    const at::Tensor& da_cs_rev,
    const at::Tensor& segsum,
    const at::Tensor& D,
    int chunk_size) {
  check_input(dout, "dout", at::kBFloat16);
  check_input(q_flat, "q_flat", at::kBFloat16);
  check_input(k_flat, "k_flat", at::kBFloat16);
  check_input(v, "v", at::kBFloat16);
  check_input(dstates, "dstates", at::kBFloat16);
  TORCH_CHECK(q_bias.is_cuda() && q_bias.is_contiguous() && q_bias.scalar_type() == at::kFloat, "q_bias must be contiguous CUDA fp32");
  TORCH_CHECK(k_bias.is_cuda() && k_bias.is_contiguous() && k_bias.scalar_type() == at::kFloat, "k_bias must be contiguous CUDA fp32");
  TORCH_CHECK(mimo_v.is_cuda() && mimo_v.is_contiguous() && mimo_v.scalar_type() == at::kFloat, "mimo_v must be contiguous CUDA fp32");
  TORCH_CHECK(mimo_o.is_cuda() && mimo_o.is_contiguous() && mimo_o.scalar_type() == at::kFloat, "mimo_o must be contiguous CUDA fp32");
  TORCH_CHECK(da_cs_rev.is_cuda() && da_cs_rev.is_contiguous() && da_cs_rev.scalar_type() == at::kFloat, "da_cs_rev must be contiguous CUDA fp32");
  TORCH_CHECK(segsum.is_cuda() && segsum.is_contiguous() && segsum.scalar_type() == at::kFloat, "segsum must be contiguous CUDA fp32");
  TORCH_CHECK(D.is_cuda() && D.is_contiguous() && D.scalar_type() == at::kFloat, "D must be contiguous CUDA fp32");
  TORCH_CHECK(chunk_size == kChunk, "Wave 8 tile-stream WMMA path specializes chunk_size=16");

  TORCH_CHECK(dout.dim() == 4, "dout must have shape [B, S, H, P]");
  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  const int nchunks = (S + chunk_size - 1) / chunk_size;
  TORCH_CHECK(v.sizes() == dout.sizes(), "v shape mismatch");
  TORCH_CHECK(q_bias.dim() == 3, "q_bias must have shape [H, R, N]");
  TORCH_CHECK(k_bias.sizes() == q_bias.sizes(), "k_bias shape mismatch");
  const int R = static_cast<int>(q_bias.size(1));
  const int N = static_cast<int>(q_bias.size(2));
  TORCH_CHECK(R == kR, "Wave 8 tile-stream WMMA path specializes R=4");
  TORCH_CHECK(N == 64, "Wave 8 tile-stream WMMA path specializes N=64");
  TORCH_CHECK(P >= kPPanel && P % kPPanel == 0, "Wave 8 tile-stream WMMA path requires P to be a positive multiple of 64");
  TORCH_CHECK(q_flat.dim() == 4, "q_flat must have shape [B, S*R, G, N]");
  const int G = static_cast<int>(q_flat.size(2));
  TORCH_CHECK(G > 0 && H % G == 0, "H must be divisible by G");
  TORCH_CHECK(q_flat.sizes() == at::IntArrayRef({B, S * R, G, N}), "q_flat shape mismatch");
  TORCH_CHECK(k_flat.sizes() == q_flat.sizes(), "k_flat shape mismatch");
  TORCH_CHECK(q_bias.sizes() == at::IntArrayRef({H, R, N}), "q_bias shape mismatch");
  TORCH_CHECK(mimo_v.sizes() == at::IntArrayRef({H, R, P}), "mimo_v shape mismatch");
  TORCH_CHECK(mimo_o.sizes() == mimo_v.sizes(), "mimo_o shape mismatch");
  TORCH_CHECK(dstates.sizes() == at::IntArrayRef({B, H, nchunks, N, P}), "dstates shape mismatch");
  TORCH_CHECK(da_cs_rev.sizes() == at::IntArrayRef({B, H, S}), "da_cs_rev shape mismatch");
  TORCH_CHECK(segsum.sizes() == at::IntArrayRef({B, H, nchunks, chunk_size, chunk_size}), "segsum shape mismatch");
  TORCH_CHECK(D.sizes() == at::IntArrayRef({H}), "D shape mismatch");
}

void stage2_mono_wmma_tile_stream_chunk_owner_out(
    const at::Tensor& dout,
    const at::Tensor& q_flat,
    const at::Tensor& k_flat,
    const at::Tensor& v,
    const at::Tensor& q_bias,
    const at::Tensor& k_bias,
    const at::Tensor& mimo_v,
    const at::Tensor& mimo_o,
    const at::Tensor& dstates,
    const at::Tensor& da_cs_rev,
    const at::Tensor& segsum,
    const at::Tensor& D,
    const at::Tensor& dv_delta,
    const at::Tensor& dmimo_v_delta,
    const at::Tensor& dmimo_v_chunk_delta,
    const at::Tensor& dssda_delta,
    int chunk_size) {
  validate_inputs(dout, q_flat, k_flat, v, q_bias, k_bias, mimo_v, mimo_o, dstates, da_cs_rev, segsum, D, chunk_size);
  check_input(dv_delta, "dv_delta", at::kBFloat16);
  TORCH_CHECK(dmimo_v_delta.is_cuda() && dmimo_v_delta.is_contiguous(), "dmimo_v_delta must be contiguous CUDA");
  TORCH_CHECK(dmimo_v_chunk_delta.is_cuda() && dmimo_v_chunk_delta.is_contiguous(), "dmimo_v_chunk_delta must be contiguous CUDA");
  TORCH_CHECK(dssda_delta.is_cuda() && dssda_delta.is_contiguous(), "dssda_delta must be contiguous CUDA");
  TORCH_CHECK(dmimo_v_delta.scalar_type() == at::kFloat, "dmimo_v_delta must be fp32");
  TORCH_CHECK(dmimo_v_chunk_delta.scalar_type() == at::kFloat, "dmimo_v_chunk_delta must be fp32");
  TORCH_CHECK(dssda_delta.scalar_type() == at::kFloat, "dssda_delta must be fp32");

  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  const int R = static_cast<int>(q_bias.size(1));
  const int N = static_cast<int>(q_bias.size(2));
  const int G = static_cast<int>(q_flat.size(2));
  const int nchunks = (S + chunk_size - 1) / chunk_size;
  TORCH_CHECK(dv_delta.sizes() == at::IntArrayRef({B, S, H, P}), "dv_delta shape mismatch");
  TORCH_CHECK(dmimo_v_delta.sizes() == at::IntArrayRef({B, H, R, P}), "dmimo_v_delta shape mismatch");
  TORCH_CHECK(dmimo_v_chunk_delta.sizes() == at::IntArrayRef({B, H, nchunks, R, P}), "dmimo_v_chunk_delta shape mismatch");
  TORCH_CHECK(dssda_delta.sizes() == at::IntArrayRef({B, H, nchunks, chunk_size, chunk_size}), "dssda_delta shape mismatch");

  const int p_panels = P / kPPanel;
  dssda_delta.zero_();

  const int64_t total_programs = static_cast<int64_t>(B) * H * nchunks * p_panels;
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  const size_t smem_bytes = mono_wmma_tile_stream_smem_bytes(P);
  auto stream = at::cuda::getCurrentCUDAStream();

  C10_CUDA_CHECK(cudaFuncSetAttribute(
      stage2_mono_wmma_tile_stream_chunk_owner_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));

  stage2_mono_wmma_tile_stream_chunk_owner_kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(q_flat.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k_flat.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      q_bias.data_ptr<float>(),
      k_bias.data_ptr<float>(),
      mimo_v.data_ptr<float>(),
      mimo_o.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(dstates.data_ptr<at::BFloat16>()),
      da_cs_rev.data_ptr<float>(),
      segsum.data_ptr<float>(),
      D.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(dv_delta.data_ptr<at::BFloat16>()),
      dmimo_v_chunk_delta.data_ptr<float>(),
      dssda_delta.data_ptr<float>(),
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t reduce_programs = static_cast<int64_t>(B) * H * R;
  reduce_dmimo_chunks_kernel<<<
      dim3(static_cast<unsigned int>(reduce_programs)),
      block,
      0,
      stream>>>(
      dmimo_v_chunk_delta.data_ptr<float>(),
      dmimo_v_delta.data_ptr<float>(),
      reduce_programs,
      B,
      H,
      R,
      P,
      nchunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<at::Tensor> stage2_mono_wmma_tile_stream_chunk_owner(
    const at::Tensor& dout,
    const at::Tensor& q_flat,
    const at::Tensor& k_flat,
    const at::Tensor& v,
    const at::Tensor& q_bias,
    const at::Tensor& k_bias,
    const at::Tensor& mimo_v,
    const at::Tensor& mimo_o,
    const at::Tensor& dstates,
    const at::Tensor& da_cs_rev,
    const at::Tensor& segsum,
    const at::Tensor& D,
    int chunk_size) {
  validate_inputs(dout, q_flat, k_flat, v, q_bias, k_bias, mimo_v, mimo_o, dstates, da_cs_rev, segsum, D, chunk_size);
  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  const int R = static_cast<int>(q_bias.size(1));
  const int nchunks = (S + chunk_size - 1) / chunk_size;
  auto f32_opts = dout.options().dtype(at::kFloat);
  at::Tensor dv_delta = at::empty({B, S, H, P}, dout.options());
  at::Tensor dmimo_v_delta = at::empty({B, H, R, P}, f32_opts);
  at::Tensor dmimo_v_chunk_delta = at::empty({B, H, nchunks, R, P}, f32_opts);
  at::Tensor dssda_delta = at::empty({B, H, nchunks, chunk_size, chunk_size}, f32_opts);
  stage2_mono_wmma_tile_stream_chunk_owner_out(
      dout,
      q_flat,
      k_flat,
      v,
      q_bias,
      k_bias,
      mimo_v,
      mimo_o,
      dstates,
      da_cs_rev,
      segsum,
      D,
      dv_delta,
      dmimo_v_delta,
      dmimo_v_chunk_delta,
      dssda_delta,
      chunk_size);
  return {dv_delta, dmimo_v_delta, dssda_delta};
}

py::dict stage2_mono_wmma_tile_stream_chunk_owner_metadata(const at::Tensor& dout) {
  TORCH_CHECK(dout.is_cuda(), "dout must be CUDA");
  TORCH_CHECK(dout.scalar_type() == at::kBFloat16, "Wave 8 tile-stream WMMA metadata requires bf16");
  TORCH_CHECK(dout.dim() == 4, "dout must have shape [B, S, H, P]");

  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, stage2_mono_wmma_tile_stream_chunk_owner_kernel));

  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const int P = static_cast<int>(dout.size(3));
  const size_t smem_bytes = mono_wmma_tile_stream_smem_bytes(P);
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, stage2_mono_wmma_tile_stream_chunk_owner_kernel, kThreads, smem_bytes));

  py::dict out;
  out["threads_per_block"] = kThreads;
  out["warps_per_block"] = kWarps;
  out["chunk_size"] = kChunk;
  out["fused_chunk"] = kFusedChunk;
  out["p_panel"] = kPPanel;
  out["p_panels"] = P / kPPanel;
  out["owner"] = "B,H,chunk,P64-panel";
  out["outputs"] = "DV, final DMIMO_V, DSSDA";
  out["tensor_core_gemms"] =
      "per P64 panel: DKI=PsiV@dPhi^T, tile-streamed LKQ=K@Q^T, state=K@dstates[:,P64], dPsi+=masked(LKQ tile)@dPhi[:,P64]";
  out["masked_lkq_smem"] = "one 16x16 bf16 tile";
  out["dssda_accumulation"] = "DSSDA is zeroed before launch and atomically accumulated by P64 panels";
  out["dynamic_smem_bytes"] = static_cast<int64_t>(smem_bytes);
  out["lkq_full_tile_elements"] = 0;
  out["lkq_stream_tile_elements"] = 16 * 16;
  out["dpsi_shared_bytes"] = static_cast<int64_t>(kFusedChunk) * kPPanel * static_cast<int64_t>(sizeof(float));
  out["dki_dpsi_reused_shared_bytes"] = static_cast<int64_t>(kFusedChunk) * kPPanel * static_cast<int64_t>(sizeof(float));
  out["dmimo_chunk_scratch_note"] = "per-chunk scratch is reduced to final DMIMO_V by a second kernel";
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

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "stage2_mono_wmma_tile_stream_chunk_owner",
      &stage2_mono_wmma_tile_stream_chunk_owner,
      "Wave 8 tile-stream tensor-core WMMA state/LKQ/D chunk-owner kernel");
  m.def(
      "stage2_mono_wmma_tile_stream_chunk_owner_out",
      &stage2_mono_wmma_tile_stream_chunk_owner_out,
      "Wave 8 tile-stream tensor-core WMMA state/LKQ/D chunk-owner kernel into preallocated outputs");
  m.def(
      "stage2_mono_wmma_tile_stream_chunk_owner_metadata",
      &stage2_mono_wmma_tile_stream_chunk_owner_metadata,
      "Wave 8 tile-stream tensor-core WMMA state/LKQ/D chunk-owner kernel metadata");
}
