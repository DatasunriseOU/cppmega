#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace py = pybind11;

namespace {

constexpr int kR = 4;
constexpr int kChunk = 16;
constexpr int kThreads = 128;
constexpr int kPtile = 32;
constexpr int kDqkElems = kR * kR;

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(const scalar_t* ptr, int64_t offset) {
  return static_cast<float>(ptr[offset]);
}

template <typename scalar_t>
__global__ void qk_dmimov_atomic_chunk_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ v,
    const float* __restrict__ mimo_o,
    const scalar_t* __restrict__ qk_dot,
    const float* __restrict__ dt,
    const scalar_t* __restrict__ trap,
    float* __restrict__ dmimo_v,
    int64_t total_programs,
    int B,
    int S,
    int H,
    int P,
    int R,
    int nchunks,
    int chunk_size) {
  const int tid = threadIdx.x;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || R != kR || chunk_size != kChunk) {
    return;
  }

  const int r_in = static_cast<int>(pid % R);
  const int chunk = static_cast<int>((pid / R) % nchunks);
  const int h = static_cast<int>((pid / (R * static_cast<int64_t>(nchunks))) % H);
  const int b = static_cast<int>(pid / (R * static_cast<int64_t>(nchunks) * H));
  if (b >= B) {
    return;
  }

  const int chunk_start = chunk * kChunk;
  for (int p = tid; p < P; p += blockDim.x) {
    float acc = 0.0f;
#pragma unroll
    for (int local_cs = 0; local_cs < kChunk; ++local_cs) {
      const int s = chunk_start + local_cs;
      if (s >= S) {
        continue;
      }
      const int64_t bhs = (static_cast<int64_t>(b) * H + h) * S + s;
      const float gamma = dt[bhs] / (1.0f + __expf(-load_as_float(trap, bhs)));
      const int64_t shp = ((static_cast<int64_t>(b) * S + s) * H + h) * P + p;
      const float base = load_as_float(dout, shp) * load_as_float(v, shp) * gamma;
      const int64_t qk_base = bhs * kDqkElems;
#pragma unroll
      for (int r_out = 0; r_out < kR; ++r_out) {
        acc += base * mimo_o[(static_cast<int64_t>(h) * R + r_out) * P + p] *
               load_as_float(qk_dot, qk_base + r_out * kR + r_in);
      }
    }
    atomicAdd(&dmimo_v[((static_cast<int64_t>(b) * H + h) * R + r_in) * P + p], acc);
  }
}

template <typename scalar_t>
__global__ void qk_dmimov_partials_chunk_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ v,
    const float* __restrict__ mimo_o,
    const scalar_t* __restrict__ qk_dot,
    const float* __restrict__ dt,
    const scalar_t* __restrict__ trap,
    float* __restrict__ partials,
    int64_t total_programs,
    int B,
    int S,
    int H,
    int P,
    int R,
    int nchunks,
    int chunk_size) {
  const int tid = threadIdx.x;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || R != kR || chunk_size != kChunk) {
    return;
  }

  const int r_in = static_cast<int>(pid % R);
  const int chunk = static_cast<int>((pid / R) % nchunks);
  const int h = static_cast<int>((pid / (R * static_cast<int64_t>(nchunks))) % H);
  const int b = static_cast<int>(pid / (R * static_cast<int64_t>(nchunks) * H));
  if (b >= B) {
    return;
  }

  const int chunk_start = chunk * kChunk;
  for (int p = tid; p < P; p += blockDim.x) {
    float acc = 0.0f;
#pragma unroll
    for (int local_cs = 0; local_cs < kChunk; ++local_cs) {
      const int s = chunk_start + local_cs;
      if (s >= S) {
        continue;
      }
      const int64_t bhs = (static_cast<int64_t>(b) * H + h) * S + s;
      const float gamma = dt[bhs] / (1.0f + __expf(-load_as_float(trap, bhs)));
      const int64_t shp = ((static_cast<int64_t>(b) * S + s) * H + h) * P + p;
      const float base = load_as_float(dout, shp) * load_as_float(v, shp) * gamma;
      const int64_t qk_base = bhs * kDqkElems;
#pragma unroll
      for (int r_out = 0; r_out < kR; ++r_out) {
        acc += base * mimo_o[(static_cast<int64_t>(h) * R + r_out) * P + p] *
               load_as_float(qk_dot, qk_base + r_out * kR + r_in);
      }
    }
    partials[((((static_cast<int64_t>(b) * H + h) * nchunks + chunk) * R + r_in) * P) + p] = acc;
  }
}

__global__ void qk_dmimov_reduce_partials_kernel(
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

template <typename scalar_t>
__global__ void qk_dmimov_output_owner_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ v,
    const float* __restrict__ mimo_o,
    const scalar_t* __restrict__ qk_dot,
    const float* __restrict__ dt,
    const scalar_t* __restrict__ trap,
    float* __restrict__ dmimo_v,
    int64_t total_programs,
    int B,
    int S,
    int H,
    int P,
    int R,
    int ptiles) {
  __shared__ float smem[kThreads];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int group = tid >> 5;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || R != kR) {
    return;
  }

  const int ptile = static_cast<int>(pid % ptiles);
  const int r_in = static_cast<int>((pid / ptiles) % R);
  const int h = static_cast<int>((pid / (ptiles * static_cast<int64_t>(R))) % H);
  const int b = static_cast<int>(pid / (ptiles * static_cast<int64_t>(R) * H));
  const int p = ptile * kPtile + lane;

  float acc = 0.0f;
  if (b < B && p < P) {
    for (int s = group; s < S; s += 4) {
      const int64_t bhs = (static_cast<int64_t>(b) * H + h) * S + s;
      const float gamma = dt[bhs] / (1.0f + __expf(-load_as_float(trap, bhs)));
      const int64_t shp = ((static_cast<int64_t>(b) * S + s) * H + h) * P + p;
      const float base = load_as_float(dout, shp) * load_as_float(v, shp) * gamma;
      const int64_t qk_base = bhs * kDqkElems;
#pragma unroll
      for (int r_out = 0; r_out < kR; ++r_out) {
        acc += base * mimo_o[(static_cast<int64_t>(h) * R + r_out) * P + p] *
               load_as_float(qk_dot, qk_base + r_out * kR + r_in);
      }
    }
  }

  smem[tid] = acc;
  __syncthreads();

  if (group == 0 && b < B && p < P) {
    const float total = smem[lane] + smem[32 + lane] + smem[64 + lane] + smem[96 + lane];
    dmimo_v[((static_cast<int64_t>(b) * H + h) * R + r_in) * P + p] = total;
  }
}

template <typename scalar_t>
__global__ void qk_dmimov_output_owner_rvec_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ v,
    const float* __restrict__ mimo_o,
    const scalar_t* __restrict__ qk_dot,
    const float* __restrict__ dt,
    const scalar_t* __restrict__ trap,
    float* __restrict__ dmimo_v,
    int64_t total_programs,
    int B,
    int S,
    int H,
    int P,
    int R,
    int ptiles) {
  __shared__ float smem[kR * kThreads];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int group = tid >> 5;
  const int64_t pid = static_cast<int64_t>(blockIdx.x);
  if (pid >= total_programs || R != kR) {
    return;
  }

  const int ptile = static_cast<int>(pid % ptiles);
  const int h = static_cast<int>((pid / ptiles) % H);
  const int b = static_cast<int>(pid / (ptiles * static_cast<int64_t>(H)));
  const int p = ptile * kPtile + lane;

  float acc[kR];
#pragma unroll
  for (int r = 0; r < kR; ++r) {
    acc[r] = 0.0f;
  }

  if (b < B && p < P) {
    for (int s = group; s < S; s += 4) {
      const int64_t bhs = (static_cast<int64_t>(b) * H + h) * S + s;
      const float gamma = dt[bhs] / (1.0f + __expf(-load_as_float(trap, bhs)));
      const int64_t shp = ((static_cast<int64_t>(b) * S + s) * H + h) * P + p;
      const float base = load_as_float(dout, shp) * load_as_float(v, shp) * gamma;
      const int64_t qk_base = bhs * kDqkElems;
#pragma unroll
      for (int r_out = 0; r_out < kR; ++r_out) {
        const float dphi = base * mimo_o[(static_cast<int64_t>(h) * R + r_out) * P + p];
#pragma unroll
        for (int r_in = 0; r_in < kR; ++r_in) {
          acc[r_in] += dphi * load_as_float(qk_dot, qk_base + r_out * kR + r_in);
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < kR; ++r) {
    smem[r * kThreads + tid] = acc[r];
  }
  __syncthreads();

  if (group == 0 && b < B && p < P) {
#pragma unroll
    for (int r = 0; r < kR; ++r) {
      const float total =
          smem[r * kThreads + lane] + smem[r * kThreads + 32 + lane] +
          smem[r * kThreads + 64 + lane] + smem[r * kThreads + 96 + lane];
      dmimo_v[((static_cast<int64_t>(b) * H + h) * R + r) * P + p] = total;
    }
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

void validate_qk_dmimov_inputs(
    const at::Tensor& dout,
    const at::Tensor& v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    int chunk_size) {
  check_input(dout, "dout", dout.scalar_type());
  check_input(v, "v", dout.scalar_type());
  check_input(qk_dot, "qk_dot", dout.scalar_type());
  check_input(trap, "trap", dout.scalar_type());
  check_cuda_contiguous(mimo_o, "mimo_o");
  check_cuda_contiguous(dt, "dt");
  TORCH_CHECK(mimo_o.scalar_type() == at::kFloat, "mimo_o must be fp32");
  TORCH_CHECK(dt.scalar_type() == at::kFloat, "dt must be fp32");
  TORCH_CHECK(chunk_size == kChunk, "DMIMO_V kernels currently specialize chunk_size=16, got ", chunk_size);

  TORCH_CHECK(dout.dim() == 4, "dout must have shape [B, S, H, P]");
  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  TORCH_CHECK(v.sizes() == dout.sizes(), "v shape mismatch");
  TORCH_CHECK(mimo_o.dim() == 3, "mimo_o must have shape [H, R, P]");
  const int R = static_cast<int>(mimo_o.size(1));
  TORCH_CHECK(R == kR, "DMIMO_V CUDA kernels currently specialize R=4, got R=", R);
  TORCH_CHECK(mimo_o.sizes() == at::IntArrayRef({H, R, P}), "mimo_o shape mismatch");
  TORCH_CHECK(qk_dot.sizes() == at::IntArrayRef({B, H, S, R * R}), "qk_dot must have shape [B, H, S, R*R]");
  TORCH_CHECK(dt.sizes() == at::IntArrayRef({B, H, S}), "dt shape mismatch");
  TORCH_CHECK(trap.sizes() == at::IntArrayRef({B, H, S}), "trap shape mismatch");
}

void validate_dmimov_output(const at::Tensor& out, int B, int H, int R, int P, const char* name) {
  check_cuda_contiguous(out, name);
  TORCH_CHECK(out.scalar_type() == at::kFloat, name, " must be fp32");
  TORCH_CHECK(out.sizes() == at::IntArrayRef({B, H, R, P}), name, " shape mismatch");
}

void qk_dmimov_atomic_chunk_out(
    const at::Tensor& dout,
    const at::Tensor& v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    const at::Tensor& dmimo_v,
    int chunk_size) {
  validate_qk_dmimov_inputs(dout, v, mimo_o, qk_dot, dt, trap, chunk_size);
  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  const int R = static_cast<int>(mimo_o.size(1));
  validate_dmimov_output(dmimo_v, B, H, R, P, "dmimo_v");

  const int nchunks = (S + chunk_size - 1) / chunk_size;
  const int64_t total_programs = static_cast<int64_t>(B) * H * nchunks * R;
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "qk_dmimov_atomic_chunk_out", [&] {
    qk_dmimov_atomic_chunk_kernel<scalar_t><<<grid, block, 0, stream>>>(
        dout.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        mimo_o.data_ptr<float>(),
        qk_dot.data_ptr<scalar_t>(),
        dt.data_ptr<float>(),
        trap.data_ptr<scalar_t>(),
        dmimo_v.data_ptr<float>(),
        total_programs,
        B,
        S,
        H,
        P,
        R,
        nchunks,
        chunk_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor qk_dmimov_atomic_chunk(
    const at::Tensor& dout,
    const at::Tensor& v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    int chunk_size) {
  validate_qk_dmimov_inputs(dout, v, mimo_o, qk_dot, dt, trap, chunk_size);
  at::Tensor out = at::zeros({dout.size(0), dout.size(2), mimo_o.size(1), dout.size(3)}, dout.options().dtype(at::kFloat));
  qk_dmimov_atomic_chunk_out(dout, v, mimo_o, qk_dot, dt, trap, out, chunk_size);
  return out;
}

void qk_dmimov_partials_chunk_out(
    const at::Tensor& dout,
    const at::Tensor& v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    const at::Tensor& partials,
    int chunk_size) {
  validate_qk_dmimov_inputs(dout, v, mimo_o, qk_dot, dt, trap, chunk_size);
  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  const int R = static_cast<int>(mimo_o.size(1));
  const int nchunks = (S + chunk_size - 1) / chunk_size;
  check_cuda_contiguous(partials, "partials");
  TORCH_CHECK(partials.scalar_type() == at::kFloat, "partials must be fp32");
  TORCH_CHECK(partials.sizes() == at::IntArrayRef({B, H, nchunks, R, P}), "partials shape mismatch");

  const int64_t total_programs = static_cast<int64_t>(B) * H * nchunks * R;
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "qk_dmimov_partials_chunk_out", [&] {
    qk_dmimov_partials_chunk_kernel<scalar_t><<<grid, block, 0, stream>>>(
        dout.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        mimo_o.data_ptr<float>(),
        qk_dot.data_ptr<scalar_t>(),
        dt.data_ptr<float>(),
        trap.data_ptr<scalar_t>(),
        partials.data_ptr<float>(),
        total_programs,
        B,
        S,
        H,
        P,
        R,
        nchunks,
        chunk_size);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor qk_dmimov_partials_chunk(
    const at::Tensor& dout,
    const at::Tensor& v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    int chunk_size) {
  validate_qk_dmimov_inputs(dout, v, mimo_o, qk_dot, dt, trap, chunk_size);
  const int S = static_cast<int>(dout.size(1));
  const int nchunks = (S + chunk_size - 1) / chunk_size;
  at::Tensor partials = at::empty(
      {dout.size(0), dout.size(2), nchunks, mimo_o.size(1), dout.size(3)},
      dout.options().dtype(at::kFloat));
  qk_dmimov_partials_chunk_out(dout, v, mimo_o, qk_dot, dt, trap, partials, chunk_size);
  return partials;
}

void qk_dmimov_reduce_partials_out(const at::Tensor& partials, const at::Tensor& dmimo_v) {
  check_cuda_contiguous(partials, "partials");
  check_cuda_contiguous(dmimo_v, "dmimo_v");
  TORCH_CHECK(partials.scalar_type() == at::kFloat, "partials must be fp32");
  TORCH_CHECK(dmimo_v.scalar_type() == at::kFloat, "dmimo_v must be fp32");
  TORCH_CHECK(partials.dim() == 5, "partials must have shape [B, H, nchunks, R, P]");
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
  qk_dmimov_reduce_partials_kernel<<<grid, block, 0, stream>>>(
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

at::Tensor qk_dmimov_reduce_partials(const at::Tensor& partials) {
  check_cuda_contiguous(partials, "partials");
  TORCH_CHECK(partials.scalar_type() == at::kFloat, "partials must be fp32");
  TORCH_CHECK(partials.dim() == 5, "partials must have shape [B, H, nchunks, R, P]");
  at::Tensor out = at::empty({partials.size(0), partials.size(1), partials.size(3), partials.size(4)}, partials.options());
  qk_dmimov_reduce_partials_out(partials, out);
  return out;
}

at::Tensor qk_dmimov_two_pass(
    const at::Tensor& dout,
    const at::Tensor& v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    int chunk_size) {
  at::Tensor partials = qk_dmimov_partials_chunk(dout, v, mimo_o, qk_dot, dt, trap, chunk_size);
  return qk_dmimov_reduce_partials(partials);
}

void qk_dmimov_output_owner_out(
    const at::Tensor& dout,
    const at::Tensor& v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    const at::Tensor& dmimo_v,
    int chunk_size) {
  validate_qk_dmimov_inputs(dout, v, mimo_o, qk_dot, dt, trap, chunk_size);
  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  const int R = static_cast<int>(mimo_o.size(1));
  validate_dmimov_output(dmimo_v, B, H, R, P, "dmimo_v");

  const int ptiles = (P + kPtile - 1) / kPtile;
  const int64_t total_programs = static_cast<int64_t>(B) * H * R * ptiles;
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "qk_dmimov_output_owner_out", [&] {
    qk_dmimov_output_owner_kernel<scalar_t><<<grid, block, 0, stream>>>(
        dout.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        mimo_o.data_ptr<float>(),
        qk_dot.data_ptr<scalar_t>(),
        dt.data_ptr<float>(),
        trap.data_ptr<scalar_t>(),
        dmimo_v.data_ptr<float>(),
        total_programs,
        B,
        S,
        H,
        P,
        R,
        ptiles);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor qk_dmimov_output_owner(
    const at::Tensor& dout,
    const at::Tensor& v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    int chunk_size) {
  validate_qk_dmimov_inputs(dout, v, mimo_o, qk_dot, dt, trap, chunk_size);
  at::Tensor out = at::empty({dout.size(0), dout.size(2), mimo_o.size(1), dout.size(3)}, dout.options().dtype(at::kFloat));
  qk_dmimov_output_owner_out(dout, v, mimo_o, qk_dot, dt, trap, out, chunk_size);
  return out;
}

void qk_dmimov_output_owner_rvec_out(
    const at::Tensor& dout,
    const at::Tensor& v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    const at::Tensor& dmimo_v,
    int chunk_size) {
  validate_qk_dmimov_inputs(dout, v, mimo_o, qk_dot, dt, trap, chunk_size);
  const int B = static_cast<int>(dout.size(0));
  const int S = static_cast<int>(dout.size(1));
  const int H = static_cast<int>(dout.size(2));
  const int P = static_cast<int>(dout.size(3));
  const int R = static_cast<int>(mimo_o.size(1));
  validate_dmimov_output(dmimo_v, B, H, R, P, "dmimo_v");

  const int ptiles = (P + kPtile - 1) / kPtile;
  const int64_t total_programs = static_cast<int64_t>(B) * H * ptiles;
  const dim3 grid(static_cast<unsigned int>(total_programs));
  const dim3 block(kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "qk_dmimov_output_owner_rvec_out", [&] {
    qk_dmimov_output_owner_rvec_kernel<scalar_t><<<grid, block, 0, stream>>>(
        dout.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        mimo_o.data_ptr<float>(),
        qk_dot.data_ptr<scalar_t>(),
        dt.data_ptr<float>(),
        trap.data_ptr<scalar_t>(),
        dmimo_v.data_ptr<float>(),
        total_programs,
        B,
        S,
        H,
        P,
        R,
        ptiles);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor qk_dmimov_output_owner_rvec(
    const at::Tensor& dout,
    const at::Tensor& v,
    const at::Tensor& mimo_o,
    const at::Tensor& qk_dot,
    const at::Tensor& dt,
    const at::Tensor& trap,
    int chunk_size) {
  validate_qk_dmimov_inputs(dout, v, mimo_o, qk_dot, dt, trap, chunk_size);
  at::Tensor out = at::empty({dout.size(0), dout.size(2), mimo_o.size(1), dout.size(3)}, dout.options().dtype(at::kFloat));
  qk_dmimov_output_owner_rvec_out(dout, v, mimo_o, qk_dot, dt, trap, out, chunk_size);
  return out;
}

py::dict make_metadata(
    const cudaFuncAttributes& attrs,
    const cudaDeviceProp& prop,
    int active_blocks,
    int dynamic_smem_bytes,
    const char* owner) {
  py::dict out;
  out["owner"] = owner;
  out["threads_per_block"] = kThreads;
  out["chunk_size"] = kChunk;
  out["p_tile"] = kPtile;
  out["dynamic_smem_bytes"] = dynamic_smem_bytes;
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
py::dict atomic_metadata_for_dtype() {
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, qk_dmimov_atomic_chunk_kernel<scalar_t>));
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, qk_dmimov_atomic_chunk_kernel<scalar_t>, kThreads, 0));
  return make_metadata(attrs, prop, active_blocks, 0, "(B,H,chunk,R) atomic chunk partial");
}

template <typename scalar_t>
py::dict partials_metadata_for_dtype() {
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, qk_dmimov_partials_chunk_kernel<scalar_t>));
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, qk_dmimov_partials_chunk_kernel<scalar_t>, kThreads, 0));
  return make_metadata(attrs, prop, active_blocks, 0, "(B,H,chunk,R) partial writer");
}

py::dict reduce_partials_metadata() {
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, qk_dmimov_reduce_partials_kernel));
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, qk_dmimov_reduce_partials_kernel, kThreads, 0));
  return make_metadata(attrs, prop, active_blocks, 0, "(B,H,R,P) partial reducer");
}

template <typename scalar_t>
py::dict output_owner_metadata_for_dtype() {
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, qk_dmimov_output_owner_kernel<scalar_t>));
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, qk_dmimov_output_owner_kernel<scalar_t>, kThreads, 0));
  return make_metadata(attrs, prop, active_blocks, 0, "(B,H,R,P-tile) output owner");
}

template <typename scalar_t>
py::dict output_owner_rvec_metadata_for_dtype() {
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, qk_dmimov_output_owner_rvec_kernel<scalar_t>));
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, qk_dmimov_output_owner_rvec_kernel<scalar_t>, kThreads, 0));
  return make_metadata(attrs, prop, active_blocks, 0, "(B,H,P-tile) output owner, all R");
}

py::dict qk_dmimov_atomic_chunk_metadata(const at::Tensor& dout) {
  TORCH_CHECK(dout.is_cuda(), "dout must be CUDA");
  py::dict out;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "qk_dmimov_atomic_chunk_metadata", [&] {
    out = atomic_metadata_for_dtype<scalar_t>();
  });
  return out;
}

py::dict qk_dmimov_partials_chunk_metadata(const at::Tensor& dout) {
  TORCH_CHECK(dout.is_cuda(), "dout must be CUDA");
  py::dict out;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "qk_dmimov_partials_chunk_metadata", [&] {
    out = partials_metadata_for_dtype<scalar_t>();
  });
  return out;
}

py::dict qk_dmimov_output_owner_metadata(const at::Tensor& dout) {
  TORCH_CHECK(dout.is_cuda(), "dout must be CUDA");
  py::dict out;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "qk_dmimov_output_owner_metadata", [&] {
    out = output_owner_metadata_for_dtype<scalar_t>();
  });
  return out;
}

py::dict qk_dmimov_output_owner_rvec_metadata(const at::Tensor& dout) {
  TORCH_CHECK(dout.is_cuda(), "dout must be CUDA");
  py::dict out;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, dout.scalar_type(), "qk_dmimov_output_owner_rvec_metadata", [&] {
    out = output_owner_rvec_metadata_for_dtype<scalar_t>();
  });
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_dmimov_atomic_chunk", &qk_dmimov_atomic_chunk, "QK DMIMO_V atomic chunk reduction");
  m.def("qk_dmimov_atomic_chunk_out", &qk_dmimov_atomic_chunk_out, "QK DMIMO_V atomic chunk reduction into existing output");
  m.def("qk_dmimov_partials_chunk", &qk_dmimov_partials_chunk, "QK DMIMO_V per-chunk partial writer");
  m.def("qk_dmimov_partials_chunk_out", &qk_dmimov_partials_chunk_out, "QK DMIMO_V per-chunk partial writer into existing output");
  m.def("qk_dmimov_reduce_partials", &qk_dmimov_reduce_partials, "QK DMIMO_V final partial reduction");
  m.def("qk_dmimov_reduce_partials_out", &qk_dmimov_reduce_partials_out, "QK DMIMO_V final partial reduction into existing output");
  m.def("qk_dmimov_two_pass", &qk_dmimov_two_pass, "QK DMIMO_V two-pass partial plus reduce path");
  m.def("qk_dmimov_output_owner", &qk_dmimov_output_owner, "QK DMIMO_V output-owner reduction");
  m.def("qk_dmimov_output_owner_out", &qk_dmimov_output_owner_out, "QK DMIMO_V output-owner reduction into existing output");
  m.def("qk_dmimov_output_owner_rvec", &qk_dmimov_output_owner_rvec, "QK DMIMO_V output-owner reduction for all R lanes");
  m.def(
      "qk_dmimov_output_owner_rvec_out",
      &qk_dmimov_output_owner_rvec_out,
      "QK DMIMO_V output-owner all-R reduction into existing output");
  m.def("qk_dmimov_atomic_chunk_metadata", &qk_dmimov_atomic_chunk_metadata, "QK DMIMO_V atomic chunk metadata");
  m.def("qk_dmimov_partials_chunk_metadata", &qk_dmimov_partials_chunk_metadata, "QK DMIMO_V partial writer metadata");
  m.def("qk_dmimov_reduce_partials_metadata", &reduce_partials_metadata, "QK DMIMO_V final reducer metadata");
  m.def("qk_dmimov_output_owner_metadata", &qk_dmimov_output_owner_metadata, "QK DMIMO_V output-owner metadata");
  m.def(
      "qk_dmimov_output_owner_rvec_metadata",
      &qk_dmimov_output_owner_rvec_metadata,
      "QK DMIMO_V output-owner all-R metadata");
}
