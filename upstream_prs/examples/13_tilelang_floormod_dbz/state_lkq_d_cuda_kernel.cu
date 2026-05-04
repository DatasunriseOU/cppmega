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
constexpr int kFcs = kChunk * kR;
constexpr int kThreads = 128;
constexpr int kMaxP = 128;

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr, int64_t offset) {
  return static_cast<float>(ptr[offset]);
}

template <typename scalar_t>
__device__ __forceinline__ float round_to_scalar_float(float x) {
  return static_cast<float>(static_cast<scalar_t>(x));
}

template <>
__device__ __forceinline__ float round_to_scalar_float<float>(float x) {
  return x;
}

template <typename scalar_t, bool kWriteDmimoPartials>
__global__ void state_lkq_d_dv_dd_chunk_owner_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ dstates,
    const scalar_t* __restrict__ dphi,
    const scalar_t* __restrict__ v,
    const float* __restrict__ mimo_v,
    const float* __restrict__ exp_rev,
    const float* __restrict__ segsum,
    const float* __restrict__ D,
    scalar_t* __restrict__ dv,
    float* __restrict__ dd,
    float* __restrict__ dmimo_partials,
    int64_t total_programs,
    int B,
    int S,
    int H,
    int N,
    int P,
    int R,
    int nchunks,
    int chunk_size) {
  __shared__ float lkq[kFcs * kFcs];
  __shared__ float dmimo_s[kR * kMaxP];
  __shared__ float dd_s[kThreads];

  const int tid = threadIdx.x;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || R != kR || chunk_size != kChunk || P > kMaxP) {
    return;
  }

  const int chunk = static_cast<int>(pid % nchunks);
  const int64_t bh = pid / nchunks;
  const int h = static_cast<int>(bh % H);
  const int b = static_cast<int>(bh / H);
  if (b >= B) {
    return;
  }

  const int chunk_start = chunk * kChunk;
  const int64_t qk_base = (((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * kFcs) * N;
  const int64_t dstate_base = (((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * N) * P;
  const int64_t dphi_base = (((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * kFcs) * P;
  const int64_t exp_base = (((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * kChunk);
  const int64_t seg_base = ((((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * kChunk) * kChunk);
  const int64_t mimo_base = static_cast<int64_t>(h) * R * P;

  if constexpr (kWriteDmimoPartials) {
    for (int idx = tid; idx < R * P; idx += blockDim.x) {
      dmimo_s[idx] = 0.0f;
    }
  }

  float dd_local = 0.0f;
  for (int idx = tid; idx < kFcs * P; idx += blockDim.x) {
    dd_local += load_as_float(dphi, dphi_base + idx);
  }
  dd_s[tid] = dd_local;

  for (int elem = tid; elem < kFcs * kFcs; elem += blockDim.x) {
    const int row = elem / kFcs;
    const int col = elem - row * kFcs;
    const int ci = row / kR;
    const int cj = col / kR;
    float acc = 0.0f;
    if (ci < cj) {
      for (int n = 0; n < N; ++n) {
        acc += load_as_float(k, qk_base + row * static_cast<int64_t>(N) + n) *
               load_as_float(q, qk_base + col * static_cast<int64_t>(N) + n);
      }
      // Matches full_bwd_bwd_epilogue.py:
      // seg_c[:, :, ci_idx.unsqueeze(0), ci_idx.unsqueeze(1)].
      acc *= __expf(segsum[seg_base + cj * kChunk + ci]);
    }
    lkq[elem] = acc;
  }
  __syncthreads();

  for (int idx = tid; idx < kChunk * P; idx += blockDim.x) {
    const int ci = idx / P;
    const int p = idx - ci * P;
    const int s = chunk_start + ci;
    if (s >= S) {
      continue;
    }

    float dv_acc = 0.0f;
    const float v_bp = load_as_float(v, ((static_cast<int64_t>(b) * S + s) * H + h) * P + p);
    for (int r = 0; r < kR; ++r) {
      const int row = ci * kR + r;
      float state_acc = 0.0f;
      for (int n = 0; n < N; ++n) {
        state_acc += load_as_float(k, qk_base + row * static_cast<int64_t>(N) + n) *
                     load_as_float(dstates, dstate_base + n * static_cast<int64_t>(P) + p);
      }
      state_acc *= exp_rev[exp_base + ci];

      float lkq_acc = 0.0f;
      for (int cj = ci + 1; cj < kChunk; ++cj) {
#pragma unroll
        for (int rj = 0; rj < kR; ++rj) {
          const int col = cj * kR + rj;
          lkq_acc += lkq[row * kFcs + col] *
                     load_as_float(dphi, dphi_base + col * static_cast<int64_t>(P) + p);
        }
      }

      const float d_direct =
          D[h] * load_as_float(dphi, dphi_base + row * static_cast<int64_t>(P) + p);
      const float dpsi = round_to_scalar_float<scalar_t>(state_acc + lkq_acc + d_direct);
      dv_acc += dpsi * mimo_v[mimo_base + r * static_cast<int64_t>(P) + p];
      if constexpr (kWriteDmimoPartials) {
        atomicAdd(&dmimo_s[r * kMaxP + p], dpsi * v_bp);
      }
    }

    dv[((static_cast<int64_t>(b) * S + s) * H + h) * P + p] =
        static_cast<scalar_t>(dv_acc);
  }

  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      dd_s[tid] += dd_s[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(&dd[static_cast<int64_t>(b) * H + h], dd_s[0]);
  }

  if constexpr (kWriteDmimoPartials) {
    __syncthreads();
    for (int idx = tid; idx < R * P; idx += blockDim.x) {
      const int r = idx / P;
      const int p = idx - r * P;
      dmimo_partials[((((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * R + r) * P) + p] =
          dmimo_s[r * kMaxP + p];
    }
  }
}

__global__ void reduce_dmimo_partials_kernel(
    const float* __restrict__ partials,
    float* __restrict__ dmimo_v,
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
  const int b = static_cast<int>(pid / (R * static_cast<int64_t>(H)));
  if (b >= B) {
    return;
  }

  for (int p = tid; p < P; p += blockDim.x) {
    float acc = 0.0f;
    for (int chunk = 0; chunk < nchunks; ++chunk) {
      acc += partials[((((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * R + r) * P) + p];
    }
    dmimo_v[((static_cast<int64_t>(b) * H + h) * R + r) * P + p] = acc;
  }
}

void check_input(const at::Tensor& tensor, const char* name, at::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " dtype mismatch");
}

void check_cuda_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void validate_inputs(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& dstates,
    const at::Tensor& dphi,
    const at::Tensor& v,
    const at::Tensor& mimo_v,
    const at::Tensor& exp_rev,
    const at::Tensor& segsum,
    const at::Tensor& D,
    int chunk_size) {
  check_input(q, "q", q.scalar_type());
  check_input(k, "k", q.scalar_type());
  check_input(dstates, "dstates", q.scalar_type());
  check_input(dphi, "dphi", q.scalar_type());
  check_input(v, "v", q.scalar_type());
  check_cuda_contiguous(mimo_v, "mimo_v");
  check_cuda_contiguous(exp_rev, "exp_rev");
  check_cuda_contiguous(segsum, "segsum");
  check_cuda_contiguous(D, "D");
  TORCH_CHECK(mimo_v.scalar_type() == at::kFloat, "mimo_v must be fp32");
  TORCH_CHECK(exp_rev.scalar_type() == at::kFloat, "exp_rev must be fp32");
  TORCH_CHECK(segsum.scalar_type() == at::kFloat, "segsum must be fp32");
  TORCH_CHECK(D.scalar_type() == at::kFloat, "D must be fp32");
  TORCH_CHECK(chunk_size == kChunk, "state/LKQ/D kernels specialize chunk_size=16, got ", chunk_size);

  TORCH_CHECK(q.dim() == 5, "q must have shape [B, H, nchunks, fcs, N]");
  const int B = static_cast<int>(q.size(0));
  const int H = static_cast<int>(q.size(1));
  const int nchunks = static_cast<int>(q.size(2));
  const int fcs = static_cast<int>(q.size(3));
  const int N = static_cast<int>(q.size(4));
  TORCH_CHECK(fcs == kFcs, "q fcs must be chunk_size*R=64");
  TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
  TORCH_CHECK(dstates.sizes() == at::IntArrayRef({B, H, nchunks, N, dphi.size(4)}), "dstates shape mismatch");
  TORCH_CHECK(dphi.sizes() == at::IntArrayRef({B, H, nchunks, fcs, dphi.size(4)}), "dphi shape mismatch");
  const int P = static_cast<int>(dphi.size(4));
  TORCH_CHECK(P <= kMaxP, "P must be <=128 for this skeleton, got ", P);
  TORCH_CHECK(v.sizes() == at::IntArrayRef({B, nchunks * chunk_size, H, P}), "v shape mismatch");
  TORCH_CHECK(mimo_v.sizes() == at::IntArrayRef({H, kR, P}), "mimo_v shape mismatch");
  TORCH_CHECK(exp_rev.sizes() == at::IntArrayRef({B, H, nchunks, chunk_size}), "exp_rev shape mismatch");
  TORCH_CHECK(segsum.sizes() == at::IntArrayRef({B, H, nchunks, chunk_size, chunk_size}), "segsum shape mismatch");
  TORCH_CHECK(D.sizes() == at::IntArrayRef({H}), "D shape mismatch");
}

void validate_outputs(
    const at::Tensor& q,
    const at::Tensor& dphi,
    const at::Tensor& dv,
    const at::Tensor& dd,
    int chunk_size) {
  check_input(dv, "dv", q.scalar_type());
  check_cuda_contiguous(dd, "dd");
  TORCH_CHECK(dd.scalar_type() == at::kFloat, "dd must be fp32");
  const int B = static_cast<int>(q.size(0));
  const int H = static_cast<int>(q.size(1));
  const int nchunks = static_cast<int>(q.size(2));
  const int P = static_cast<int>(dphi.size(4));
  TORCH_CHECK(dv.sizes() == at::IntArrayRef({B, nchunks * chunk_size, H, P}), "dv shape mismatch");
  TORCH_CHECK(dd.sizes() == at::IntArrayRef({B, H}), "dd shape mismatch");
}

template <bool kWriteDmimoPartials>
void launch_state_lkq_d_kernel(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& dstates,
    const at::Tensor& dphi,
    const at::Tensor& v,
    const at::Tensor& mimo_v,
    const at::Tensor& exp_rev,
    const at::Tensor& segsum,
    const at::Tensor& D,
    const at::Tensor& dv,
    const at::Tensor& dd,
    const at::Tensor& dmimo_partials,
    int chunk_size) {
  validate_inputs(q, k, dstates, dphi, v, mimo_v, exp_rev, segsum, D, chunk_size);
  validate_outputs(q, dphi, dv, dd, chunk_size);
  const int B = static_cast<int>(q.size(0));
  const int H = static_cast<int>(q.size(1));
  const int nchunks = static_cast<int>(q.size(2));
  const int N = static_cast<int>(q.size(4));
  const int P = static_cast<int>(dphi.size(4));
  const int S = nchunks * chunk_size;
  const int R = kR;
  if constexpr (kWriteDmimoPartials) {
    check_cuda_contiguous(dmimo_partials, "dmimo_partials");
    TORCH_CHECK(dmimo_partials.scalar_type() == at::kFloat, "dmimo_partials must be fp32");
    TORCH_CHECK(
        dmimo_partials.sizes() == at::IntArrayRef({B, H, nchunks, R, P}),
        "dmimo_partials shape mismatch");
  }

  const int64_t total_programs = static_cast<int64_t>(B) * H * nchunks;
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "state_lkq_d_dv_dd_chunk_owner", [&] {
    state_lkq_d_dv_dd_chunk_owner_kernel<scalar_t, kWriteDmimoPartials><<<grid, block, 0, stream>>>(
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        dstates.data_ptr<scalar_t>(),
        dphi.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        mimo_v.data_ptr<float>(),
        exp_rev.data_ptr<float>(),
        segsum.data_ptr<float>(),
        D.data_ptr<float>(),
        dv.data_ptr<scalar_t>(),
        dd.data_ptr<float>(),
        kWriteDmimoPartials ? dmimo_partials.data_ptr<float>() : nullptr,
        total_programs,
        B,
        S,
        H,
        N,
        P,
        R,
        nchunks,
        chunk_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void state_lkq_d_dv_dd_chunk_owner_out(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& dstates,
    const at::Tensor& dphi,
    const at::Tensor& v,
    const at::Tensor& mimo_v,
    const at::Tensor& exp_rev,
    const at::Tensor& segsum,
    const at::Tensor& D,
    const at::Tensor& dv,
    const at::Tensor& dd,
    int chunk_size) {
  launch_state_lkq_d_kernel<false>(
      q, k, dstates, dphi, v, mimo_v, exp_rev, segsum, D, dv, dd, at::Tensor(), chunk_size);
}

std::vector<at::Tensor> state_lkq_d_dv_dd_chunk_owner(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& dstates,
    const at::Tensor& dphi,
    const at::Tensor& v,
    const at::Tensor& mimo_v,
    const at::Tensor& exp_rev,
    const at::Tensor& segsum,
    const at::Tensor& D,
    int chunk_size) {
  validate_inputs(q, k, dstates, dphi, v, mimo_v, exp_rev, segsum, D, chunk_size);
  const int B = static_cast<int>(q.size(0));
  const int H = static_cast<int>(q.size(1));
  const int nchunks = static_cast<int>(q.size(2));
  const int P = static_cast<int>(dphi.size(4));
  at::Tensor dv = at::empty({B, nchunks * chunk_size, H, P}, q.options());
  at::Tensor dd = at::zeros({B, H}, q.options().dtype(at::kFloat));
  state_lkq_d_dv_dd_chunk_owner_out(q, k, dstates, dphi, v, mimo_v, exp_rev, segsum, D, dv, dd, chunk_size);
  return {dv, dd};
}

void state_lkq_d_dv_dd_dmimov_partials_chunk_owner_out(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& dstates,
    const at::Tensor& dphi,
    const at::Tensor& v,
    const at::Tensor& mimo_v,
    const at::Tensor& exp_rev,
    const at::Tensor& segsum,
    const at::Tensor& D,
    const at::Tensor& dv,
    const at::Tensor& dd,
    const at::Tensor& dmimo_partials,
    int chunk_size) {
  launch_state_lkq_d_kernel<true>(
      q, k, dstates, dphi, v, mimo_v, exp_rev, segsum, D, dv, dd, dmimo_partials, chunk_size);
}

std::vector<at::Tensor> state_lkq_d_dv_dd_dmimov_partials_chunk_owner(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& dstates,
    const at::Tensor& dphi,
    const at::Tensor& v,
    const at::Tensor& mimo_v,
    const at::Tensor& exp_rev,
    const at::Tensor& segsum,
    const at::Tensor& D,
    int chunk_size) {
  validate_inputs(q, k, dstates, dphi, v, mimo_v, exp_rev, segsum, D, chunk_size);
  const int B = static_cast<int>(q.size(0));
  const int H = static_cast<int>(q.size(1));
  const int nchunks = static_cast<int>(q.size(2));
  const int P = static_cast<int>(dphi.size(4));
  at::Tensor dv = at::empty({B, nchunks * chunk_size, H, P}, q.options());
  at::Tensor dd = at::zeros({B, H}, q.options().dtype(at::kFloat));
  at::Tensor partials = at::empty({B, H, nchunks, kR, P}, q.options().dtype(at::kFloat));
  state_lkq_d_dv_dd_dmimov_partials_chunk_owner_out(
      q, k, dstates, dphi, v, mimo_v, exp_rev, segsum, D, dv, dd, partials, chunk_size);
  return {dv, dd, partials};
}

void state_lkq_d_reduce_dmimov_partials_out(const at::Tensor& partials, const at::Tensor& dmimo_v) {
  check_cuda_contiguous(partials, "partials");
  check_cuda_contiguous(dmimo_v, "dmimo_v");
  TORCH_CHECK(partials.scalar_type() == at::kFloat, "partials must be fp32");
  TORCH_CHECK(dmimo_v.scalar_type() == at::kFloat, "dmimo_v must be fp32");
  TORCH_CHECK(partials.dim() == 5, "partials must have shape [B,H,nchunks,R,P]");
  const int B = static_cast<int>(partials.size(0));
  const int H = static_cast<int>(partials.size(1));
  const int nchunks = static_cast<int>(partials.size(2));
  const int R = static_cast<int>(partials.size(3));
  const int P = static_cast<int>(partials.size(4));
  TORCH_CHECK(R == kR, "partials R mismatch");
  TORCH_CHECK(dmimo_v.sizes() == at::IntArrayRef({B, H, R, P}), "dmimo_v shape mismatch");

  const int64_t total_programs = static_cast<int64_t>(B) * H * R;
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  reduce_dmimo_partials_kernel<<<grid, block, 0, stream>>>(
      partials.data_ptr<float>(),
      dmimo_v.data_ptr<float>(),
      total_programs,
      B,
      H,
      R,
      P,
      nchunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor state_lkq_d_reduce_dmimov_partials(const at::Tensor& partials) {
  check_cuda_contiguous(partials, "partials");
  TORCH_CHECK(partials.scalar_type() == at::kFloat, "partials must be fp32");
  TORCH_CHECK(partials.dim() == 5, "partials must have shape [B,H,nchunks,R,P]");
  at::Tensor out = at::empty(
      {partials.size(0), partials.size(1), partials.size(3), partials.size(4)},
      partials.options());
  state_lkq_d_reduce_dmimov_partials_out(partials, out);
  return out;
}

py::dict chunk_owner_metadata(const at::Tensor& q, bool with_partials) {
  py::dict result;
  if (!q.is_cuda()) {
    return result;
  }
  cudaFuncAttributes attr;
  if (with_partials) {
    C10_CUDA_CHECK(cudaFuncGetAttributes(
        &attr,
        reinterpret_cast<const void*>(
            state_lkq_d_dv_dd_chunk_owner_kernel<at::BFloat16, true>)));
  } else {
    C10_CUDA_CHECK(cudaFuncGetAttributes(
        &attr,
        reinterpret_cast<const void*>(
            state_lkq_d_dv_dd_chunk_owner_kernel<at::BFloat16, false>)));
  }
  int device = q.get_device();
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      with_partials
          ? reinterpret_cast<const void*>(state_lkq_d_dv_dd_chunk_owner_kernel<at::BFloat16, true>)
          : reinterpret_cast<const void*>(state_lkq_d_dv_dd_chunk_owner_kernel<at::BFloat16, false>),
      kThreads,
      0));
  cudaDeviceProp prop;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  result["num_regs"] = attr.numRegs;
  result["shared_bytes"] = attr.sharedSizeBytes;
  result["local_bytes"] = attr.localSizeBytes;
  result["max_threads_per_block"] = attr.maxThreadsPerBlock;
  result["active_blocks_per_sm"] = active_blocks;
  result["theoretical_occupancy"] = static_cast<double>(active_blocks * kThreads) /
                                    static_cast<double>(prop.maxThreadsPerMultiProcessor);
  result["sm_count"] = prop.multiProcessorCount;
  return result;
}

py::dict reduce_metadata() {
  py::dict result;
  cudaFuncAttributes attr;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(reduce_dmimo_partials_kernel)));
  int device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, reinterpret_cast<const void*>(reduce_dmimo_partials_kernel), kThreads, 0));
  cudaDeviceProp prop;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  result["num_regs"] = attr.numRegs;
  result["shared_bytes"] = attr.sharedSizeBytes;
  result["local_bytes"] = attr.localSizeBytes;
  result["max_threads_per_block"] = attr.maxThreadsPerBlock;
  result["active_blocks_per_sm"] = active_blocks;
  result["theoretical_occupancy"] = static_cast<double>(active_blocks * kThreads) /
                                    static_cast<double>(prop.maxThreadsPerMultiProcessor);
  result["sm_count"] = prop.multiProcessorCount;
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "state_lkq_d_dv_dd_chunk_owner",
      &state_lkq_d_dv_dd_chunk_owner,
      "state+LKQ+D contribution to DV/DD");
  m.def(
      "state_lkq_d_dv_dd_chunk_owner_out",
      &state_lkq_d_dv_dd_chunk_owner_out,
      "state+LKQ+D contribution to existing DV/DD");
  m.def(
      "state_lkq_d_dv_dd_dmimov_partials_chunk_owner",
      &state_lkq_d_dv_dd_dmimov_partials_chunk_owner,
      "state+LKQ+D contribution to DV/DD plus per-chunk DMIMO_V partials");
  m.def(
      "state_lkq_d_dv_dd_dmimov_partials_chunk_owner_out",
      &state_lkq_d_dv_dd_dmimov_partials_chunk_owner_out,
      "state+LKQ+D contribution to existing DV/DD/DMIMO_V partials");
  m.def(
      "state_lkq_d_reduce_dmimov_partials",
      &state_lkq_d_reduce_dmimov_partials,
      "reduce state+LKQ+D DMIMO_V partials");
  m.def(
      "state_lkq_d_reduce_dmimov_partials_out",
      &state_lkq_d_reduce_dmimov_partials_out,
      "reduce state+LKQ+D DMIMO_V partials into existing output");
  m.def(
      "state_lkq_d_chunk_owner_metadata",
      &chunk_owner_metadata,
      "state+LKQ+D chunk-owner metadata");
  m.def(
      "state_lkq_d_reduce_dmimov_partials_metadata",
      &reduce_metadata,
      "state+LKQ+D DMIMO_V reducer metadata");
}
