#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

namespace cppmega_cutlass_mxfp8 {

using namespace cute;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_UINT8(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Byte, #x " must be uint8")

const char* cutlass_status_name(cutlass::Status status) {
  switch (status) {
    case cutlass::Status::kSuccess:
      return "success";
    case cutlass::Status::kErrorMisalignedOperand:
      return "misaligned_operand";
    case cutlass::Status::kErrorInvalidProblem:
      return "invalid_problem";
    case cutlass::Status::kErrorNotSupported:
      return "not_supported";
    case cutlass::Status::kErrorWorkspaceNull:
      return "workspace_null";
    case cutlass::Status::kErrorInternal:
      return "internal";
    case cutlass::Status::kErrorArchMismatch:
      return "arch_mismatch";
    case cutlass::Status::kErrorInsufficientDriver:
      return "insufficient_driver";
    default:
      return "unknown";
  }
}

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using ElementAData = typename ElementA::DataType;
using ElementBData = typename ElementB::DataType;
using ElementSF = typename ElementA::ScaleFactorType;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = AlignmentC;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ThreadBlockShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutCTag,
    AlignmentC,
    ElementD,
    LayoutDTag,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutATag,
    AlignmentA,
    ElementB,
    LayoutBTag,
    AlignmentB,
    ElementAccumulator,
    ThreadBlockShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

template <class Layout>
__global__ void prepack_rowwise_scale_kernel(
    uint8_t const* __restrict__ compact,
    uint8_t* __restrict__ native,
    Layout native_layout,
    int rows,
    int k_blocks,
    int compact_ld) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * k_blocks;
  if (idx >= total) {
    return;
  }
  int row = idx / k_blocks;
  int kb = idx - row * k_blocks;
  native[native_layout(row, kb * 32, 0)] = compact[row * compact_ld + kb];
}

template <class LayoutA, class LayoutB>
__global__ void prepack_two_rowwise_scale_kernel(
    uint8_t const* __restrict__ compact_a,
    uint8_t* __restrict__ native_a,
    LayoutA native_layout_a,
    int rows_a,
    int k_blocks_a,
    int compact_ld_a,
    uint8_t const* __restrict__ compact_b,
    uint8_t* __restrict__ native_b,
    LayoutB native_layout_b,
    int rows_b,
    int k_blocks_b,
    int compact_ld_b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_a = rows_a * k_blocks_a;
  int total_b = rows_b * k_blocks_b;
  if (idx < total_a) {
    int row = idx / k_blocks_a;
    int kb = idx - row * k_blocks_a;
    native_a[native_layout_a(row, kb * 32, 0)] = compact_a[row * compact_ld_a + kb];
    return;
  }
  idx -= total_a;
  if (idx < total_b) {
    int row = idx / k_blocks_b;
    int kb = idx - row * k_blocks_b;
    native_b[native_layout_b(row, kb * 32, 0)] = compact_b[row * compact_ld_b + kb];
  }
}

void validate_inputs(
    at::Tensor const& A_u8,
    at::Tensor const& SFA_u8,
    at::Tensor const& B_u8,
    at::Tensor const& SFB_u8,
    int m,
    int n,
    int k) {
  CHECK_CUDA(A_u8);
  CHECK_CUDA(SFA_u8);
  CHECK_CUDA(B_u8);
  CHECK_CUDA(SFB_u8);
  CHECK_CONTIGUOUS(A_u8);
  CHECK_CONTIGUOUS(SFA_u8);
  CHECK_CONTIGUOUS(B_u8);
  CHECK_CONTIGUOUS(SFB_u8);
  CHECK_UINT8(A_u8);
  CHECK_UINT8(SFA_u8);
  CHECK_UINT8(B_u8);
  CHECK_UINT8(SFB_u8);
  TORCH_CHECK(m > 0 && n > 0 && k > 0, "m, n, k must be positive");
  TORCH_CHECK(m % 128 == 0 && n % 128 == 0 && k % 128 == 0,
              "CUTLASS MXFP8 GB10 backend currently requires M/N/K multiples of 128, got ",
              m, "x", n, "x", k);
  TORCH_CHECK(A_u8.numel() >= static_cast<int64_t>(m) * k, "A_u8 is too small");
  TORCH_CHECK(B_u8.numel() >= static_cast<int64_t>(n) * k, "B_u8 is too small");
  TORCH_CHECK(SFA_u8.dim() == 2, "SFA_u8 must be 2D [padded_M, K/32]");
  TORCH_CHECK(SFB_u8.dim() == 2, "SFB_u8 must be 2D [padded_N, K/32]");
  int k_blocks = k / 32;
  TORCH_CHECK(SFA_u8.size(0) >= m, "SFA_u8 dim0 is smaller than M");
  TORCH_CHECK(SFA_u8.size(1) >= k_blocks, "SFA_u8 dim1 is smaller than K/32");
  TORCH_CHECK(SFB_u8.size(0) >= n, "SFB_u8 dim0 is smaller than N");
  TORCH_CHECK(SFB_u8.size(1) >= k_blocks, "SFB_u8 dim1 is smaller than K/32");
}

struct CachedTensors {
  at::Tensor native_SFA;
  at::Tensor native_SFB;
  at::Tensor workspace;
};

using CacheKey = std::tuple<int64_t, uintptr_t, int64_t, int64_t, int64_t>;

struct CacheKeyHash {
  std::size_t operator()(CacheKey const& key) const {
    auto h = std::hash<int64_t>{}(std::get<0>(key));
    h ^= std::hash<uintptr_t>{}(std::get<1>(key)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(std::get<2>(key)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(std::get<3>(key)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(std::get<4>(key)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

std::mutex& cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<CacheKey, CachedTensors, CacheKeyHash>& tensor_cache() {
  static std::unordered_map<CacheKey, CachedTensors, CacheKeyHash> cache;
  return cache;
}

bool cache_enabled() {
  char const* value = std::getenv("CPPMEGA_CUTLASS_MXFP8_CACHE");
  return value == nullptr || std::string(value) != "0";
}

CachedTensors get_cached_tensors(
    int64_t device_index,
    cudaStream_t stream,
    int64_t native_sfa_elems,
    int64_t native_sfb_elems,
    int64_t workspace_size,
    at::TensorOptions const& byte_options) {
  auto allocate = [&]() {
    CachedTensors tensors;
    tensors.native_SFA = at::empty({native_sfa_elems}, byte_options);
    tensors.native_SFB = at::empty({native_sfb_elems}, byte_options);
    if (workspace_size > 0) {
      tensors.workspace = at::empty({workspace_size}, byte_options);
    }
    return tensors;
  };

  if (!cache_enabled()) {
    return allocate();
  }

  CacheKey key{
      device_index,
      reinterpret_cast<uintptr_t>(stream),
      native_sfa_elems,
      native_sfb_elems,
      workspace_size};
  std::lock_guard<std::mutex> lock(cache_mutex());
  auto& cache = tensor_cache();
  auto it = cache.find(key);
  if (it == cache.end()) {
    it = cache.emplace(key, allocate()).first;
  }
  return it->second;
}

#endif

}  // namespace cppmega_cutlass_mxfp8

using namespace cppmega_cutlass_mxfp8;

at::Tensor cutlass_mxfp8_tn_gemm_cuda(
    at::Tensor A_u8,
    at::Tensor SFA_u8,
    at::Tensor B_u8,
    at::Tensor SFB_u8,
    int64_t m64,
    int64_t n64,
    int64_t k64,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  TORCH_CHECK(false, "CUTLASS SM120/SM121 MXFP8 backend was not compiled for this architecture");
#else
  TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "M/N/K exceed int");
  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);
  int k_blocks = k / 32;
  int l = 1;

  validate_inputs(A_u8, SFA_u8, B_u8, SFB_u8, m, n, k);
  if (use_out) {
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(out);
    TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bfloat16");
    TORCH_CHECK(out.numel() >= static_cast<int64_t>(m) * n, "out is too small");
  }

  c10::cuda::CUDAGuard device_guard(A_u8.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, l));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, l));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, l));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, l));
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, l));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, l));

  int64_t native_sfa_elems = static_cast<int64_t>(size(filter_zeros(layout_SFA)));
  int64_t native_sfb_elems = static_cast<int64_t>(size(filter_zeros(layout_SFB)));
  auto byte_options = at::TensorOptions().device(A_u8.device()).dtype(at::kByte);

  at::Tensor D = use_out
      ? out.view({m, n})
      : at::empty({m, n}, at::TensorOptions().device(A_u8.device()).dtype(at::kBFloat16));

  auto ptr_A = reinterpret_cast<ElementAData const*>(A_u8.data_ptr<uint8_t>());
  auto ptr_B = reinterpret_cast<ElementBData const*>(B_u8.data_ptr<uint8_t>());
  auto ptr_D = reinterpret_cast<ElementD*>(D.data_ptr<at::BFloat16>());
  auto ptr_C = use_out && accumulate ? ptr_D : ptr_D;
  float beta_f = accumulate ? static_cast<float>(beta) : 0.0f;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {ptr_A, stride_A, ptr_B, stride_B, nullptr, layout_SFA, nullptr, layout_SFB},
      {{static_cast<float>(alpha), beta_f}, ptr_C, stride_C, ptr_D, stride_D}};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto cached = get_cached_tensors(
      A_u8.device().index(),
      stream,
      native_sfa_elems,
      native_sfb_elems,
      static_cast<int64_t>(workspace_size),
      byte_options);
  at::Tensor native_SFA = cached.native_SFA;
  at::Tensor native_SFB = cached.native_SFB;

  constexpr int threads = 256;
  int total_scale_elems = (m + n) * k_blocks;
  int scale_blocks = (total_scale_elems + threads - 1) / threads;
  prepack_two_rowwise_scale_kernel<<<scale_blocks, threads, 0, stream>>>(
      SFA_u8.data_ptr<uint8_t>(),
      native_SFA.data_ptr<uint8_t>(),
      layout_SFA,
      m,
      k_blocks,
      static_cast<int>(SFA_u8.size(1)),
      SFB_u8.data_ptr<uint8_t>(),
      native_SFB.data_ptr<uint8_t>(),
      layout_SFB,
      n,
      k_blocks,
      static_cast<int>(SFB_u8.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto ptr_SFA = reinterpret_cast<ElementSF const*>(native_SFA.data_ptr<uint8_t>());
  auto ptr_SFB = reinterpret_cast<ElementSF const*>(native_SFB.data_ptr<uint8_t>());
  arguments.mainloop.ptr_SFA = ptr_SFA;
  arguments.mainloop.ptr_SFB = ptr_SFB;

  cutlass::Status can_status = gemm.can_implement(arguments);
  TORCH_CHECK(can_status == cutlass::Status::kSuccess,
              "CUTLASS MXFP8 can_implement failed: ", cutlass_status_name(can_status));

  void* workspace_ptr = workspace_size > 0 ? cached.workspace.data_ptr<uint8_t>() : nullptr;
  cutlass::Status init_status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(init_status == cutlass::Status::kSuccess,
              "CUTLASS MXFP8 initialize failed: ", cutlass_status_name(init_status));
  cutlass::Status run_status = gemm.run(stream);
  TORCH_CHECK(run_status == cutlass::Status::kSuccess,
              "CUTLASS MXFP8 run failed: ", cutlass_status_name(run_status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return D;
#endif
}
