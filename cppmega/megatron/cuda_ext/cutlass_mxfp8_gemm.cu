#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <cstdlib>
#include <string>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include "cppmega_sm120_blockscaled_mma_tma_compact_scale.hpp"

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
using KernelElementA = typename CollectiveMainloop::ElementA;
using KernelElementB = typename CollectiveMainloop::ElementB;
using KernelElementSF = typename CollectiveMainloop::ElementSF;
constexpr int64_t kOperandRowwise =
    static_cast<int64_t>(cutlass::gemm::collective::CppMegaCompactOperandSource::kRowwise);
constexpr int64_t kOperandColumnwiseTranspose =
    static_cast<int64_t>(
        cutlass::gemm::collective::CppMegaCompactOperandSource::kColumnwiseTranspose);

// Split the manual scale+payload fill across the CUTLASS main producer and
// auxiliary producer roles. The direct compact-scale backend is still opt-in,
// but this avoids making one producer warp serialize both operands.
using CompactScaleDispatchPolicy =
    cutlass::gemm::collective::MainloopSm120TmaWarpSpecializedBlockScaledCompactScale<
        CollectiveMainloop::DispatchPolicy::Stages,
        CollectiveMainloop::DispatchPolicy::SchedulerPipelineStageCount,
        typename CollectiveMainloop::DispatchPolicy::ClusterShape,
        typename CollectiveMainloop::DispatchPolicy::Schedule,
        true>;

using CompactScaleAsymmetricDispatchPolicy =
    cutlass::gemm::collective::MainloopSm120TmaWarpSpecializedBlockScaledCompactScale<
        CollectiveMainloop::DispatchPolicy::Stages,
        CollectiveMainloop::DispatchPolicy::SchedulerPipelineStageCount,
        typename CollectiveMainloop::DispatchPolicy::ClusterShape,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseBlockScaledSm120<
            CollectiveMainloop::DispatchPolicy::SchedulerPipelineStageCount,
            false>>;

using CompactScaleCollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    CompactScaleDispatchPolicy,
    ThreadBlockShape,
    cute::tuple<KernelElementA, KernelElementSF>,
    cute::tuple<StrideA, LayoutSFA>,
    cute::tuple<KernelElementB, KernelElementSF>,
    cute::tuple<StrideB, LayoutSFB>,
    typename CollectiveMainloop::TiledMma,
    typename CollectiveMainloop::GmemTiledCopyPairA,
    typename CollectiveMainloop::SmemLayoutAtomsA,
    typename CollectiveMainloop::SmemCopyAtomsA,
    cute::identity,
    typename CollectiveMainloop::GmemTiledCopyPairB,
    typename CollectiveMainloop::SmemLayoutAtomsB,
    typename CollectiveMainloop::SmemCopyAtomsB,
    cute::identity>;

using CompactScaleAsymmetricCollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    CompactScaleAsymmetricDispatchPolicy,
    ThreadBlockShape,
    cute::tuple<KernelElementA, KernelElementSF>,
    cute::tuple<StrideA, LayoutSFA>,
    cute::tuple<KernelElementB, KernelElementSF>,
    cute::tuple<StrideB, LayoutSFB>,
    typename CollectiveMainloop::TiledMma,
    typename CollectiveMainloop::GmemTiledCopyPairA,
    typename CollectiveMainloop::SmemLayoutAtomsA,
    typename CollectiveMainloop::SmemCopyAtomsA,
    cute::identity,
    typename CollectiveMainloop::GmemTiledCopyPairB,
    typename CollectiveMainloop::SmemLayoutAtomsB,
    typename CollectiveMainloop::SmemCopyAtomsB,
    cute::identity>;

using CompactScaleGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CompactScaleCollectiveMainloop,
    CollectiveEpilogue,
    void>;

using CompactScaleGemm = cutlass::gemm::device::GemmUniversalAdapter<CompactScaleGemmKernel>;
using CompactScaleAsymmetricGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CompactScaleAsymmetricCollectiveMainloop,
    CollectiveEpilogue,
    void>;

using CompactScaleAsymmetricGemm = cutlass::gemm::device::GemmUniversalAdapter<CompactScaleAsymmetricGemmKernel>;
using CompactScaleStrideA = typename CompactScaleGemm::GemmKernel::StrideA;
using CompactScaleStrideB = typename CompactScaleGemm::GemmKernel::StrideB;
using CompactScaleStrideC = typename CompactScaleGemm::GemmKernel::StrideC;
using CompactScaleStrideD = typename CompactScaleGemm::GemmKernel::StrideD;

// =======================================================================
// Validation helpers
// =======================================================================

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

void validate_direct_operand(
    at::Tensor const& data,
    at::Tensor const& scale,
    int logical_rows,
    int k,
    int64_t source,
    int64_t data_ld,
    int64_t scale_ld,
    char const* name) {
  CHECK_CUDA(data);
  CHECK_CUDA(scale);
  CHECK_CONTIGUOUS(data);
  CHECK_CONTIGUOUS(scale);
  CHECK_UINT8(data);
  CHECK_UINT8(scale);
  TORCH_CHECK(data.dim() == 2, name, " payload must be 2D");
  TORCH_CHECK(scale.dim() == 2, name, " scale must be 2D");
  TORCH_CHECK(source == kOperandRowwise || source == kOperandColumnwiseTranspose,
              name, " source must be 0 rowwise or 1 columnwise-transpose");
  TORCH_CHECK(data_ld > 0 && data_ld <= INT_MAX, name, " data_ld must fit int");
  TORCH_CHECK(scale_ld > 0 && scale_ld <= INT_MAX, name, " scale_ld must fit int");
  int k_blocks = k / 32;
  if (source == kOperandRowwise) {
    TORCH_CHECK(data.size(0) >= logical_rows, name, " rowwise payload dim0 is smaller than logical rows");
    TORCH_CHECK(data.size(1) >= k, name, " rowwise payload dim1 is smaller than K");
    TORCH_CHECK(data_ld >= k, name, " rowwise data_ld is smaller than K");
    TORCH_CHECK(scale.size(0) >= logical_rows, name, " rowwise scale dim0 is smaller than logical rows");
    TORCH_CHECK(scale.size(1) >= k_blocks, name, " rowwise scale dim1 is smaller than K/32");
    TORCH_CHECK(scale_ld >= k_blocks, name, " rowwise scale_ld is smaller than K/32");
  } else {
    TORCH_CHECK(data.size(0) >= k, name, " columnwise payload dim0 is smaller than K");
    TORCH_CHECK(data.size(1) >= logical_rows, name, " columnwise payload dim1 is smaller than logical rows");
    TORCH_CHECK(data_ld >= logical_rows, name, " columnwise data_ld is smaller than logical rows");
    TORCH_CHECK(scale.size(0) >= k_blocks, name, " columnwise scale dim0 is smaller than K/32");
    TORCH_CHECK(scale.size(1) >= logical_rows, name, " columnwise scale dim1 is smaller than logical rows");
    TORCH_CHECK(scale_ld >= logical_rows, name, " columnwise scale_ld is smaller than logical rows");
  }
}

// =======================================================================
// Templated GEMM runners using compact-scale mainloop
// =======================================================================

template <class Gemm>
at::Tensor run_compact_scale_gemm_impl(
    at::Tensor const& A_u8,
    at::Tensor const& SFA_u8,
    at::Tensor const& B_u8,
    at::Tensor const& SFB_u8,
    int m, int n, int k,
    at::Tensor& D,
    float alpha, float beta,
    cudaStream_t stream) {

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  int l = 1;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, l));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, l));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, l));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, l));

  auto ptr_A = reinterpret_cast<KernelElementA const*>(A_u8.data_ptr<uint8_t>());
  auto ptr_B = reinterpret_cast<KernelElementB const*>(B_u8.data_ptr<uint8_t>());
  auto ptr_SFA = reinterpret_cast<KernelElementSF const*>(SFA_u8.data_ptr<uint8_t>());
  auto ptr_SFB = reinterpret_cast<KernelElementSF const*>(SFB_u8.data_ptr<uint8_t>());
  auto ptr_D = reinterpret_cast<ElementD*>(D.data_ptr<at::BFloat16>());
  // When beta == 0 the C operand is logically unused. Pass nullptr so the
  // CUTLASS epilogue skips the C load entirely; otherwise an uninitialized
  // out= tensor holding NaN could propagate as NaN * 0 = NaN.
  ElementD* ptr_C = (beta == 0.0f) ? nullptr : ptr_D;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {
          ptr_A, stride_A, ptr_B, stride_B,
          ptr_SFA, static_cast<int64_t>(SFA_u8.size(1)),
          ptr_SFB, static_cast<int64_t>(SFB_u8.size(1)),
          false,
          static_cast<int32_t>(kOperandRowwise), static_cast<int64_t>(A_u8.size(1)),
          static_cast<int32_t>(kOperandRowwise), static_cast<int64_t>(B_u8.size(1)),
      },
      {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D}};

  Gemm gemm;
  cutlass::Status can_status = gemm.can_implement(arguments);
  TORCH_CHECK(can_status == cutlass::Status::kSuccess,
              "CUTLASS MXFP8 compact-scale can_implement failed: ",
              cutlass_status_name(can_status));

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor ws;
  if (workspace_size > 0) {
    ws = at::empty({static_cast<int64_t>(workspace_size)},
                   at::TensorOptions().device(A_u8.device()).dtype(at::kByte));
  }
  void* workspace_ptr = workspace_size > 0 ? ws.data_ptr<uint8_t>() : nullptr;
  cutlass::Status init_status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(init_status == cutlass::Status::kSuccess,
              "CUTLASS MXFP8 compact-scale initialize failed: ",
              cutlass_status_name(init_status));
  cutlass::Status run_status = gemm.run(stream);
  TORCH_CHECK(run_status == cutlass::Status::kSuccess,
              "CUTLASS MXFP8 compact-scale run failed: ",
              cutlass_status_name(run_status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return D;
}

template <class Gemm>
at::Tensor run_compact_direct_gemm_impl(
    at::Tensor const& A_u8,
    at::Tensor const& SFA_u8,
    at::Tensor const& B_u8,
    at::Tensor const& SFB_u8,
    int m, int n, int k,
    int64_t a_source, int64_t a_data_ld, int64_t a_scale_ld,
    int64_t b_source, int64_t b_data_ld, int64_t b_scale_ld,
    at::Tensor& D,
    float alpha, float beta,
    cudaStream_t stream) {

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  int l = 1;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, l));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, l));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, l));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, l));

  auto ptr_A = reinterpret_cast<KernelElementA const*>(A_u8.data_ptr<uint8_t>());
  auto ptr_B = reinterpret_cast<KernelElementB const*>(B_u8.data_ptr<uint8_t>());
  auto ptr_SFA = reinterpret_cast<KernelElementSF const*>(SFA_u8.data_ptr<uint8_t>());
  auto ptr_SFB = reinterpret_cast<KernelElementSF const*>(SFB_u8.data_ptr<uint8_t>());
  auto ptr_D = reinterpret_cast<ElementD*>(D.data_ptr<at::BFloat16>());
  // See run_compact_scale_gemm_impl: skip the C load when beta == 0.
  ElementD* ptr_C = (beta == 0.0f) ? nullptr : ptr_D;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {
          ptr_A, stride_A, ptr_B, stride_B,
          ptr_SFA, a_scale_ld, ptr_SFB, b_scale_ld,
          true,
          static_cast<int32_t>(a_source), a_data_ld,
          static_cast<int32_t>(b_source), b_data_ld,
      },
      {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D}};

  Gemm gemm;
  cutlass::Status can_status = gemm.can_implement(arguments);
  TORCH_CHECK(can_status == cutlass::Status::kSuccess,
              "CUTLASS MXFP8 compact direct can_implement failed: ",
              cutlass_status_name(can_status));

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor ws;
  if (workspace_size > 0) {
    ws = at::empty({static_cast<int64_t>(workspace_size)},
                   at::TensorOptions().device(A_u8.device()).dtype(at::kByte));
  }
  void* workspace_ptr = workspace_size > 0 ? ws.data_ptr<uint8_t>() : nullptr;
  cutlass::Status init_status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(init_status == cutlass::Status::kSuccess,
              "CUTLASS MXFP8 compact direct initialize failed: ",
              cutlass_status_name(init_status));
  cutlass::Status run_status = gemm.run(stream);
  TORCH_CHECK(run_status == cutlass::Status::kSuccess,
              "CUTLASS MXFP8 compact direct run failed: ",
              cutlass_status_name(run_status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return D;
}

#endif

}  // namespace cppmega_cutlass_mxfp8

using namespace cppmega_cutlass_mxfp8;

at::Tensor cutlass_mxfp8_tn_gemm_compact_scale_cuda(
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
  TORCH_CHECK(false, "CUTLASS SM120/SM121 MXFP8 compact-scale backend was not compiled for this architecture");
#else
  TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "M/N/K exceed int");
  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);

  validate_inputs(A_u8, SFA_u8, B_u8, SFB_u8, m, n, k);
  if (use_out) {
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(out);
    TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bfloat16");
    TORCH_CHECK(out.numel() >= static_cast<int64_t>(m) * n, "out is too small");
  }

  c10::cuda::CUDAGuard device_guard(A_u8.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  at::Tensor D = use_out
      ? out.view({m, n})
      : at::empty({m, n}, at::TensorOptions().device(A_u8.device()).dtype(at::kBFloat16));

  // beta-gating against accumulate is enforced in the python wrapper; trust the
  // value as passed here.
  (void)accumulate;
  float alpha_f = static_cast<float>(alpha);
  float beta_f = static_cast<float>(beta);

  return run_compact_scale_gemm_impl<CompactScaleGemm>(
      A_u8, SFA_u8, B_u8, SFB_u8, m, n, k, D,
      alpha_f, beta_f, stream);
#endif
}

at::Tensor cutlass_mxfp8_tn_gemm_compact_direct_cuda(
    at::Tensor A_u8,
    at::Tensor SFA_u8,
    at::Tensor B_u8,
    at::Tensor SFB_u8,
    int64_t m64,
    int64_t n64,
    int64_t k64,
    int64_t a_source,
    int64_t a_data_ld,
    int64_t a_scale_ld,
    int64_t b_source,
    int64_t b_data_ld,
    int64_t b_scale_ld,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  TORCH_CHECK(false, "CUTLASS SM120/SM121 MXFP8 compact direct backend was not compiled for this architecture");
#else
  TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "M/N/K exceed int");
  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);

  TORCH_CHECK(m > 0 && n > 0 && k > 0, "m, n, k must be positive");
  TORCH_CHECK(m % 128 == 0 && n % 128 == 0 && k % 128 == 0,
              "CUTLASS MXFP8 compact direct backend currently requires M/N/K multiples of 128, got ",
              m, "x", n, "x", k);
  validate_direct_operand(A_u8, SFA_u8, m, k, a_source, a_data_ld, a_scale_ld, "A");
  validate_direct_operand(B_u8, SFB_u8, n, k, b_source, b_data_ld, b_scale_ld, "B");
  if (use_out) {
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(out);
    TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bfloat16");
    TORCH_CHECK(out.numel() >= static_cast<int64_t>(m) * n, "out is too small");
  }

  c10::cuda::CUDAGuard device_guard(A_u8.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  at::Tensor D = use_out
      ? out.view({m, n})
      : at::empty({m, n}, at::TensorOptions().device(A_u8.device()).dtype(at::kBFloat16));

  // beta-gating against accumulate is enforced in the python wrapper; trust the
  // value as passed here.
  (void)accumulate;
  float alpha_f = static_cast<float>(alpha);
  float beta_f = static_cast<float>(beta);

  return run_compact_direct_gemm_impl<CompactScaleGemm>(
      A_u8, SFA_u8, B_u8, SFB_u8, m, n, k,
      a_source, a_data_ld, a_scale_ld,
      b_source, b_data_ld, b_scale_ld,
      D, alpha_f, beta_f, stream);
#endif
}

at::Tensor cutlass_mxfp8_tn_gemm_compact_direct_asym_cuda(
    at::Tensor A_u8,
    at::Tensor SFA_u8,
    at::Tensor B_u8,
    at::Tensor SFB_u8,
    int64_t m64,
    int64_t n64,
    int64_t k64,
    int64_t a_source,
    int64_t a_data_ld,
    int64_t a_scale_ld,
    int64_t b_source,
    int64_t b_data_ld,
    int64_t b_scale_ld,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  TORCH_CHECK(false, "CUTLASS SM120/SM121 MXFP8 compact direct asymmetric backend was not compiled for this architecture");
#else
  TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "M/N/K exceed int");
  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);

  TORCH_CHECK(m > 0 && n > 0 && k > 0, "m, n, k must be positive");
  TORCH_CHECK(m % 128 == 0 && n % 128 == 0 && k % 128 == 0,
              "CUTLASS MXFP8 compact direct asymmetric backend currently requires M/N/K multiples of 128, got ",
              m, "x", n, "x", k);
  validate_direct_operand(A_u8, SFA_u8, m, k, a_source, a_data_ld, a_scale_ld, "A");
  validate_direct_operand(B_u8, SFB_u8, n, k, b_source, b_data_ld, b_scale_ld, "B");
  if (use_out) {
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(out);
    TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bfloat16");
    TORCH_CHECK(out.numel() >= static_cast<int64_t>(m) * n, "out is too small");
  }

  c10::cuda::CUDAGuard device_guard(A_u8.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  at::Tensor D = use_out
      ? out.view({m, n})
      : at::empty({m, n}, at::TensorOptions().device(A_u8.device()).dtype(at::kBFloat16));

  // beta-gating against accumulate is enforced in the python wrapper; trust the
  // value as passed here.
  (void)accumulate;
  float alpha_f = static_cast<float>(alpha);
  float beta_f = static_cast<float>(beta);

  return run_compact_direct_gemm_impl<CompactScaleAsymmetricGemm>(
      A_u8, SFA_u8, B_u8, SFB_u8, m, n, k,
      a_source, a_data_ld, a_scale_ld,
      b_source, b_data_ld, b_scale_ld,
      D, alpha_f, beta_f, stream);
#endif
}
