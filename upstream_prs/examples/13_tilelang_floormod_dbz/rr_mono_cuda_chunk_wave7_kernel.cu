#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kR = 4;
constexpr int kChunk = 16;
constexpr int kFusedChunk = kChunk * kR;
constexpr int kN = 64;
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

__global__ void stage2_mono_row_stream_chunk_owner_kernel(
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
  cursor += static_cast<int64_t>(kFusedChunk) * kN * sizeof(__nv_bfloat16);
  __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(cursor);
  cursor += static_cast<int64_t>(kFusedChunk) * kN * sizeof(__nv_bfloat16);
  __nv_bfloat16* dphi_s = reinterpret_cast<__nv_bfloat16*>(cursor);
  cursor += static_cast<int64_t>(kFusedChunk) * kPPanel * sizeof(__nv_bfloat16);
  cursor = reinterpret_cast<unsigned char*>(
      (reinterpret_cast<uintptr_t>(cursor) + alignof(float) - 1) &
      ~(static_cast<uintptr_t>(alignof(float) - 1)));
  float* accum_s = reinterpret_cast<float*>(cursor);
  cursor += static_cast<int64_t>(kFusedChunk) * kPPanel * sizeof(float);
  float* dssda_s = reinterpret_cast<float*>(cursor);
  cursor += static_cast<int64_t>(kChunk) * kChunk * sizeof(float);
  float* lkq_row_s = reinterpret_cast<float*>(cursor);

  const int tid = threadIdx.x;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || N != kN || R != kR || chunk_size != kChunk ||
      blockDim.x != kThreads || P < kPPanel || (P % kPPanel) != 0) {
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
  const int64_t qk_flat_base = static_cast<int64_t>(b) * S * R * G * N + h_qk * N;
  const int64_t bias_base = static_cast<int64_t>(h) * R * N;
  const int64_t mimo_base = static_cast<int64_t>(h) * R * P;
  const int64_t dstates_base = (((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * N) * P;
  const int64_t seg_base = (((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * kChunk) * kChunk;
  const float d_value = D[h];

  for (int idx = tid; idx < kFusedChunk * kN; idx += blockDim.x) {
    const int row = idx / kN;
    const int n = idx - row * kN;
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
    k_s[idx] = float_to_bf16(k_val);
    q_s[idx] = float_to_bf16(q_val);
  }

  for (int idx = tid; idx < kChunk * kChunk; idx += blockDim.x) {
    dssda_s[idx] = 0.0f;
  }
  __syncthreads();

  for (int p_base = 0; p_base < P; p_base += kPPanel) {
    for (int idx = tid; idx < kFusedChunk * kPPanel; idx += blockDim.x) {
      const int row = idx / kPPanel;
      const int p_local = idx - row * kPPanel;
      const int p = p_base + p_local;
      const int cs = row / kR;
      const int r = row - cs * kR;
      const int s = chunk_start + cs;
      float dphi = 0.0f;
      float state = 0.0f;
      if (s < S) {
        const int64_t hp = ((static_cast<int64_t>(b) * S + s) * H + h) * P + p;
        dphi = bf16_to_float(dout[hp]) * mimo_o[mimo_base + static_cast<int64_t>(r) * P + p];
        for (int n = 0; n < kN; ++n) {
          state += bf16_to_float(k_s[row * kN + n]) *
              bf16_to_float(dstates[dstates_base + static_cast<int64_t>(n) * P + p]);
        }
        state *= __expf(da_cs_rev[(static_cast<int64_t>(b) * H + h) * S + s]);
      }
      const __nv_bfloat16 dphi_b = float_to_bf16(dphi);
      dphi_s[idx] = dphi_b;
      accum_s[idx] = state + d_value * bf16_to_float(dphi_b);
    }
    __syncthreads();

    for (int row = 0; row < kFusedChunk; ++row) {
      const int cs_i = row / kR;
      const int r_i = row - cs_i * kR;
      const int s_i = chunk_start + cs_i;

      for (int col = tid; col < kFusedChunk; col += blockDim.x) {
        float lkq = 0.0f;
        if (s_i < S) {
          for (int n = 0; n < kN; ++n) {
            lkq += bf16_to_float(k_s[row * kN + n]) * bf16_to_float(q_s[col * kN + n]);
          }
        }
        lkq_row_s[col] = lkq;
      }
      __syncthreads();

      for (int p_local = tid; p_local < kPPanel; p_local += blockDim.x) {
        float add = 0.0f;
#pragma unroll
        for (int col = 0; col < kFusedChunk; ++col) {
          const int cs_j = col / kR;
          if (cs_i < cs_j) {
            const float masked = lkq_row_s[col] * __expf(segsum[seg_base + cs_i * kChunk + cs_j]);
            add += bf16_to_float(float_to_bf16(masked)) * bf16_to_float(dphi_s[col * kPPanel + p_local]);
          }
        }
        accum_s[row * kPPanel + p_local] += add;
      }

      for (int col = tid; col < kFusedChunk; col += blockDim.x) {
        const int cs_j = col / kR;
        const int s_j = chunk_start + cs_j;
        float dki = 0.0f;
        if (s_i < S && s_j < S) {
          for (int p_local = 0; p_local < kPPanel; ++p_local) {
            const int p = p_base + p_local;
            const int64_t hp_i = ((static_cast<int64_t>(b) * S + s_i) * H + h) * P + p;
            const float psi = bf16_to_float(v[hp_i]) *
                mimo_v[mimo_base + static_cast<int64_t>(r_i) * P + p];
            dki += bf16_to_float(float_to_bf16(psi)) *
                bf16_to_float(dphi_s[col * kPPanel + p_local]);
          }
        }
        atomicAdd(&dssda_s[cs_i * kChunk + cs_j], lkq_row_s[col] * dki);
      }
      __syncthreads();
    }

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
      dmimo_v_chunk_delta[(((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * kR + r) * P + p] = acc;
    }
    __syncthreads();
  }

  for (int idx = tid; idx < kChunk * kChunk; idx += blockDim.x) {
    dssda_delta[seg_base + idx] = dssda_s[idx];
  }
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

void check_input(const at::Tensor& tensor, const char* name, at::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " dtype mismatch");
}

size_t mono_row_stream_smem_bytes() {
  const size_t bf16_bytes =
      (static_cast<size_t>(kFusedChunk) * kN +
       static_cast<size_t>(kFusedChunk) * kN +
       static_cast<size_t>(kFusedChunk) * kPPanel) *
      sizeof(__nv_bfloat16);
  const size_t float_bytes =
      (static_cast<size_t>(kFusedChunk) * kPPanel +
       static_cast<size_t>(kChunk) * kChunk +
       static_cast<size_t>(kFusedChunk)) *
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
  TORCH_CHECK(chunk_size == kChunk, "Wave 7 row-stream owner specializes chunk_size=16");

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
  TORCH_CHECK(R == kR, "Wave 7 row-stream owner specializes R=4");
  TORCH_CHECK(N == kN, "Wave 7 row-stream owner specializes N=64");
  TORCH_CHECK(P >= kPPanel && P % kPPanel == 0, "Wave 7 row-stream owner requires P to be a positive multiple of 64");
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

void stage2_mono_row_stream_chunk_owner_out(
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
  TORCH_CHECK(
      dmimo_v_chunk_delta.sizes() == at::IntArrayRef({B, H, nchunks, R, P}),
      "dmimo_v_chunk_delta shape mismatch");
  TORCH_CHECK(dssda_delta.sizes() == at::IntArrayRef({B, H, nchunks, chunk_size, chunk_size}), "dssda_delta shape mismatch");

  const int64_t total_programs = static_cast<int64_t>(B) * H * nchunks;
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  const size_t smem_bytes = mono_row_stream_smem_bytes();
  auto stream = at::cuda::getCurrentCUDAStream();

  C10_CUDA_CHECK(cudaFuncSetAttribute(
      stage2_mono_row_stream_chunk_owner_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));

  stage2_mono_row_stream_chunk_owner_kernel<<<grid, block, smem_bytes, stream>>>(
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

std::vector<at::Tensor> stage2_mono_row_stream_chunk_owner(
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
  stage2_mono_row_stream_chunk_owner_out(
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

py::dict stage2_mono_row_stream_chunk_owner_metadata(const at::Tensor& dout) {
  TORCH_CHECK(dout.is_cuda(), "dout must be CUDA");
  TORCH_CHECK(dout.scalar_type() == at::kBFloat16, "Wave 7 row-stream metadata requires bf16");
  TORCH_CHECK(dout.dim() == 4, "dout must have shape [B, S, H, P]");

  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, stage2_mono_row_stream_chunk_owner_kernel));

  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  const int nchunks = (S + kChunk - 1) / kChunk;
  const size_t smem_bytes = mono_row_stream_smem_bytes();
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, stage2_mono_row_stream_chunk_owner_kernel, kThreads, smem_bytes));

  py::dict out;
  out["threads_per_block"] = kThreads;
  out["warps_per_block"] = kWarps;
  out["chunk_size"] = kChunk;
  out["fused_chunk"] = kFusedChunk;
  out["p_panel"] = kPPanel;
  out["p_panels"] = P / kPPanel;
  out["owner"] = "B,H,chunk row-stream";
  out["chunk_owner_ctas"] = B * H * nchunks;
  out["reduction_ctas"] = B * H * kR;
  out["outputs"] = "DV, final DMIMO_V, DSSDA";
  out["local_reuse"] =
      "K/Q staged once per chunk; dPhi/dPsi one P64 panel at a time; LKQ is streamed one row at a time instead of materialized as a full 64x64 tile";
  out["dynamic_smem_bytes"] = static_cast<int64_t>(smem_bytes);
  out["k_shared_bytes"] = static_cast<int64_t>(kFusedChunk) * kN * static_cast<int64_t>(sizeof(__nv_bfloat16));
  out["q_shared_bytes"] = static_cast<int64_t>(kFusedChunk) * kN * static_cast<int64_t>(sizeof(__nv_bfloat16));
  out["dphi_shared_bytes"] = static_cast<int64_t>(kFusedChunk) * kPPanel * static_cast<int64_t>(sizeof(__nv_bfloat16));
  out["dpsi_shared_bytes"] = static_cast<int64_t>(kFusedChunk) * kPPanel * static_cast<int64_t>(sizeof(float));
  out["dssda_shared_bytes"] = static_cast<int64_t>(kChunk) * kChunk * static_cast<int64_t>(sizeof(float));
  out["lkq_row_shared_bytes"] = static_cast<int64_t>(kFusedChunk) * static_cast<int64_t>(sizeof(float));
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
      "stage2_mono_row_stream_chunk_owner",
      &stage2_mono_row_stream_chunk_owner,
      "Wave 7 row-stream chunk-owner state/LKQ/D kernel");
  m.def(
      "stage2_mono_row_stream_chunk_owner_out",
      &stage2_mono_row_stream_chunk_owner_out,
      "Wave 7 row-stream chunk-owner state/LKQ/D kernel into preallocated outputs");
  m.def(
      "stage2_mono_row_stream_chunk_owner_metadata",
      &stage2_mono_row_stream_chunk_owner_metadata,
      "Wave 7 row-stream chunk-owner state/LKQ/D kernel metadata");
}
