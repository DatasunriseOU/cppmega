#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace cppmega_mamba3_mono {

constexpr int kChunk = 16;
constexpr int kRank = 4;
constexpr int kFcs = kChunk * kRank;
constexpr int kN = 64;
constexpr int kThreads = 256;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Half, #x " must be float16")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")

__device__ __forceinline__ float h2f(__half x) {
  return __half2float(x);
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void mono_chunk_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ dout,
    const __half* __restrict__ v,
    const __half* __restrict__ mimo_v,
    const __half* __restrict__ mimo_o,
    const float* __restrict__ qk_dot,
    const float* __restrict__ dt,
    const float* __restrict__ trap,
    const __half* __restrict__ dstates,
    float* __restrict__ dv,
    float* __restrict__ dmimo_v,
    float* __restrict__ dk_diag,
    float* __restrict__ dq_diag,
    float* __restrict__ lkq_checksum,
    int B,
    int S,
    int H,
    int P,
    int nchunks) {
  extern __shared__ unsigned char smem_raw[];
  unsigned char* cursor = smem_raw;

  __half* s_q = reinterpret_cast<__half*>(cursor);
  cursor += kFcs * kN * sizeof(__half);
  __half* s_k_t = reinterpret_cast<__half*>(cursor);
  cursor += kN * kFcs * sizeof(__half);
  __half* s_dphi = reinterpret_cast<__half*>(cursor);
  cursor += kFcs * P * sizeof(__half);
  __half* s_psi = reinterpret_cast<__half*>(cursor);
  cursor += kFcs * P * sizeof(__half);
  float* s_dpsi = reinterpret_cast<float*>(cursor);
  cursor += kFcs * P * sizeof(float);
  float* s_lkq = reinterpret_cast<float*>(cursor);

  const int chunk = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int chunk_s0 = chunk * kChunk;

  for (int idx = tid; idx < kFcs * kN; idx += blockDim.x) {
    int f = idx / kN;
    int n = idx - f * kN;
    int t = f / kRank;
    int r = f - t * kRank;
    int s = chunk_s0 + t;
    int64_t q_idx = (((static_cast<int64_t>(b) * S + s) * H + h) * kRank + r) * kN + n;
    s_q[f * kN + n] = q[q_idx];
    s_k_t[n * kFcs + f] = k[q_idx];
  }

  for (int idx = tid; idx < kFcs * P; idx += blockDim.x) {
    int f = idx / P;
    int p = idx - f * P;
    int t = f / kRank;
    int r = f - t * kRank;
    int s = chunk_s0 + t;
    int64_t seq_idx = (static_cast<int64_t>(b) * S + s) * H + h;
    int64_t hp_idx = seq_idx * P + p;
    int64_t mimo_idx = (static_cast<int64_t>(h) * kRank + r) * P + p;
    float dphi = h2f(dout[hp_idx]) * h2f(mimo_o[mimo_idx]);
    float psi = h2f(v[hp_idx]) * h2f(mimo_v[mimo_idx]);
    s_dphi[idx] = __float2half_rn(dphi);
    s_psi[idx] = __float2half_rn(psi);
    s_dpsi[idx] = 0.0f;
  }

  __syncthreads();

  using namespace nvcuda;
  constexpr int kWmma = 16;
  int warp_id = tid / 32;
  int warp_count = blockDim.x / 32;
  for (int tile = warp_id; tile < 16; tile += warp_count) {
    int tile_m = tile / 4;
    int tile_n = tile - tile_m * 4;
    wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    for (int n0 = 0; n0 < kN; n0 += kWmma) {
      wmma::load_matrix_sync(a_frag, s_q + (tile_m * kWmma) * kN + n0, kN);
      wmma::load_matrix_sync(b_frag, s_k_t + n0 * kFcs + tile_n * kWmma, kFcs);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    wmma::store_matrix_sync(
        s_lkq + (tile_m * kWmma) * kFcs + tile_n * kWmma,
        acc_frag,
        kFcs,
        wmma::mem_row_major);
  }

  __syncthreads();

  for (int idx = tid; idx < kFcs * P; idx += blockDim.x) {
    int f = idx / P;
    int p = idx - f * P;
    int t = f / kRank;
    int r_in = f - t * kRank;
    int s = chunk_s0 + t;
    int64_t bh = static_cast<int64_t>(b) * H + h;
    float gamma = dt[bh * S + s] * sigmoidf_fast(trap[bh * S + s]);

    float acc = 0.0f;

    // State consumer: K[f, :] @ dstates[:, p].
    for (int n = 0; n < kN; ++n) {
      int64_t ds_idx = ((static_cast<int64_t>(b) * H + h) * kN + n) * P + p;
      acc += h2f(s_k_t[n * kFcs + f]) * h2f(dstates[ds_idx]);
    }

    // Intra-chunk LKQ consumer.  The WMMA tile is reused here before any
    // global write, which is the behavior this skeleton is meant to exercise.
    for (int fq = 0; fq < kFcs; ++fq) {
      int tq = fq / kRank;
      if (t < tq) {
        acc += s_lkq[f * kFcs + fq] * h2f(s_dphi[fq * P + p]);
      }
    }

    // Same-time qk_dot diagonal consumer.
    for (int r_out = 0; r_out < kRank; ++r_out) {
      int64_t qk_idx =
          ((((static_cast<int64_t>(b) * S + s) * H + h) * kRank + r_out) * kRank + r_in);
      acc += gamma * qk_dot[qk_idx] * h2f(s_dphi[(t * kRank + r_out) * P + p]);
    }
    s_dpsi[idx] = acc;
  }

  __syncthreads();

  for (int idx = tid; idx < kChunk * P; idx += blockDim.x) {
    int t = idx / P;
    int p = idx - t * P;
    int s = chunk_s0 + t;
    float acc = 0.0f;
    for (int r = 0; r < kRank; ++r) {
      int64_t mimo_idx = (static_cast<int64_t>(h) * kRank + r) * P + p;
      acc += s_dpsi[(t * kRank + r) * P + p] * h2f(mimo_v[mimo_idx]);
    }
    int64_t out_idx = ((static_cast<int64_t>(b) * S + s) * H + h) * P + p;
    dv[out_idx] = acc;
  }

  for (int idx = tid; idx < kFcs * P; idx += blockDim.x) {
    int f = idx / P;
    int p = idx - f * P;
    int t = f / kRank;
    int r = f - t * kRank;
    int s = chunk_s0 + t;
    int64_t hp_idx = ((static_cast<int64_t>(b) * S + s) * H + h) * P + p;
    int64_t out_idx = ((static_cast<int64_t>(b) * H + h) * kRank + r) * P + p;
    atomicAdd(dmimo_v + out_idx, s_dpsi[idx] * h2f(v[hp_idx]));
  }

  for (int idx = tid; idx < kChunk * kRank * kRank * kN; idx += blockDim.x) {
    int n = idx % kN;
    int tmp = idx / kN;
    int r_in = tmp % kRank;
    tmp /= kRank;
    int r_out = tmp % kRank;
    int t = tmp / kRank;
    int s = chunk_s0 + t;
    int64_t bh = static_cast<int64_t>(b) * H + h;
    float gamma = dt[bh * S + s] * sigmoidf_fast(trap[bh * S + s]);

    float dot = 0.0f;
    int f_out = t * kRank + r_out;
    int f_in = t * kRank + r_in;
    for (int p = 0; p < P; ++p) {
      dot += h2f(s_dphi[f_out * P + p]) * h2f(s_psi[f_in * P + p]);
    }
    float grad_qk = gamma * dot;

    int64_t base = (((static_cast<int64_t>(b) * S + s) * H + h) * kRank);
    int64_t dq_idx = (base + r_out) * kN + n;
    int64_t dk_idx = (base + r_in) * kN + n;
    atomicAdd(dq_diag + dq_idx, grad_qk * h2f(s_k_t[n * kFcs + f_in]));
    atomicAdd(dk_diag + dk_idx, grad_qk * h2f(s_q[f_out * kN + n]));
  }

  __syncthreads();

  if (tid == 0) {
    float sum = 0.0f;
    for (int i = 0; i < kFcs * kFcs; ++i) {
      sum += s_lkq[i];
    }
    lkq_checksum[(static_cast<int64_t>(b) * H + h) * nchunks + chunk] = sum;
  }
}

size_t shared_storage_bytes(int P) {
  return kFcs * kN * sizeof(__half)      // Q
      + kN * kFcs * sizeof(__half)       // K^T
      + kFcs * P * sizeof(__half)        // dPhi
      + kFcs * P * sizeof(__half)        // Psi
      + kFcs * P * sizeof(float)         // dPsi
      + kFcs * kFcs * sizeof(float);     // LKQ
}

void validate_inputs(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& dout,
    const at::Tensor& v,
    const at::Tensor& mimo_v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    const at::Tensor& dstates,
    int64_t chunk_size) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(dout);
  CHECK_CUDA(v);
  CHECK_CUDA(mimo_v);
  CHECK_CUDA(mimo_o);
  CHECK_CUDA(qk_dot);
  CHECK_CUDA(dt);
  CHECK_CUDA(trap);
  CHECK_CUDA(dstates);
  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(dout);
  CHECK_CONTIGUOUS(v);
  CHECK_CONTIGUOUS(mimo_v);
  CHECK_CONTIGUOUS(mimo_o);
  CHECK_CONTIGUOUS(qk_dot);
  CHECK_CONTIGUOUS(dt);
  CHECK_CONTIGUOUS(trap);
  CHECK_CONTIGUOUS(dstates);
  CHECK_HALF(q);
  CHECK_HALF(k);
  CHECK_HALF(dout);
  CHECK_HALF(v);
  CHECK_HALF(mimo_v);
  CHECK_HALF(mimo_o);
  CHECK_HALF(dstates);
  CHECK_FLOAT(qk_dot);
  CHECK_FLOAT(dt);
  CHECK_FLOAT(trap);

  TORCH_CHECK(chunk_size == kChunk, "skeleton currently requires chunk_size=16");
  TORCH_CHECK(q.dim() == 5, "q must have shape (B,S,H,R,N)");
  TORCH_CHECK(k.sizes() == q.sizes(), "k shape must match q");
  TORCH_CHECK(dout.dim() == 4, "dout must have shape (B,S,H,P)");
  TORCH_CHECK(v.sizes() == dout.sizes(), "v shape must match dout");
  TORCH_CHECK(mimo_v.dim() == 3, "mimo_v must have shape (H,R,P)");
  TORCH_CHECK(mimo_o.sizes() == mimo_v.sizes(), "mimo_o shape must match mimo_v");
  TORCH_CHECK(qk_dot.dim() == 5, "qk_dot must have shape (B,S,H,R,R)");
  TORCH_CHECK(dt.dim() == 3, "dt must have shape (B,H,S)");
  TORCH_CHECK(trap.sizes() == dt.sizes(), "trap shape must match dt");
  TORCH_CHECK(dstates.dim() == 4, "dstates must have shape (B,H,N,P)");

  int64_t B = q.size(0);
  int64_t S = q.size(1);
  int64_t H = q.size(2);
  int64_t R = q.size(3);
  int64_t N = q.size(4);
  int64_t P = dout.size(3);
  TORCH_CHECK(q.sizes() == at::IntArrayRef({B, S, H, R, N}), "invalid q shape");
  TORCH_CHECK(dout.size(0) == B && dout.size(1) == S && dout.size(2) == H, "dout B/S/H mismatch");
  TORCH_CHECK(mimo_v.size(0) == H && mimo_v.size(1) == R && mimo_v.size(2) == P, "mimo_v shape mismatch");
  TORCH_CHECK(qk_dot.sizes() == at::IntArrayRef({B, S, H, R, R}), "qk_dot shape mismatch");
  TORCH_CHECK(dt.sizes() == at::IntArrayRef({B, H, S}), "dt shape mismatch");
  TORCH_CHECK(dstates.sizes() == at::IntArrayRef({B, H, N, P}), "dstates shape mismatch");
  TORCH_CHECK(S % kChunk == 0, "S must be divisible by 16");
  TORCH_CHECK(R == kRank, "skeleton currently requires R=4");
  TORCH_CHECK(N == kN, "skeleton currently requires N=64");
  TORCH_CHECK(P > 0 && P <= 128, "skeleton requires 1 <= P <= 128");
}

}  // namespace cppmega_mamba3_mono

std::vector<at::Tensor> mamba3_mono_chunk_skeleton_cuda(
    at::Tensor q,
    at::Tensor k,
    at::Tensor dout,
    at::Tensor v,
    at::Tensor mimo_v,
    at::Tensor mimo_o,
    at::Tensor qk_dot,
    at::Tensor dt,
    at::Tensor trap,
    at::Tensor dstates,
    int64_t chunk_size) {
  using namespace cppmega_mamba3_mono;
  validate_inputs(q, k, dout, v, mimo_v, mimo_o, qk_dot, dt, trap, dstates, chunk_size);
  c10::cuda::CUDAGuard device_guard(q.device());

  int B = static_cast<int>(q.size(0));
  int S = static_cast<int>(q.size(1));
  int H = static_cast<int>(q.size(2));
  int P = static_cast<int>(dout.size(3));
  int nchunks = S / kChunk;

  auto opts_f = q.options().dtype(at::kFloat);
  at::Tensor dv = at::zeros({B, S, H, P}, opts_f);
  at::Tensor dmimo_v = at::zeros({B, H, kRank, P}, opts_f);
  at::Tensor dk_diag = at::zeros({B, S, H, kRank, kN}, opts_f);
  at::Tensor dq_diag = at::zeros({B, S, H, kRank, kN}, opts_f);
  at::Tensor lkq_checksum = at::zeros({B, H, nchunks}, opts_f);

  size_t smem_bytes = shared_storage_bytes(P);
  cudaError_t attr_status = cudaFuncSetAttribute(
      mono_chunk_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  TORCH_CHECK(
      attr_status == cudaSuccess,
      "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=",
      smem_bytes,
      ") failed: ",
      cudaGetErrorString(attr_status));

  dim3 grid(nchunks, H, B);
  auto stream = at::cuda::getCurrentCUDAStream();
  mono_chunk_kernel<<<grid, kThreads, smem_bytes, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(dout.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(mimo_v.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(mimo_o.data_ptr<at::Half>()),
      qk_dot.data_ptr<float>(),
      dt.data_ptr<float>(),
      trap.data_ptr<float>(),
      reinterpret_cast<const __half*>(dstates.data_ptr<at::Half>()),
      dv.data_ptr<float>(),
      dmimo_v.data_ptr<float>(),
      dk_diag.data_ptr<float>(),
      dq_diag.data_ptr<float>(),
      lkq_checksum.data_ptr<float>(),
      B,
      S,
      H,
      P,
      nchunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dv, dmimo_v, dk_diag, dq_diag, lkq_checksum};
}

py::dict mamba3_mono_chunk_skeleton_metadata() {
  using namespace cppmega_mamba3_mono;
  py::dict d;
  d["chunk_size"] = kChunk;
  d["rank"] = kRank;
  d["fused_chunk_sequence"] = kFcs;
  d["n_tile"] = kN;
  d["threads"] = kThreads;
  d["cta_mapping"] = "(chunk, head, batch)";
  d["tensor_core_tile"] = "WMMA 16x16x16 tiles over LKQ=(64x64) = K @ Q^T";
  d["shared_storage_p64_bytes"] = static_cast<int64_t>(shared_storage_bytes(64));
  d["shared_storage_p128_bytes"] = static_cast<int64_t>(shared_storage_bytes(128));
  d["outputs"] = "DV, DMIMO_V, DK_diag, DQ_diag, LKQ checksum";
  return d;
}
