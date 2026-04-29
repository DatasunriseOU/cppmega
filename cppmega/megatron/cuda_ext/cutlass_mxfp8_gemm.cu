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

using CompactScaleNoCopyBDispatchPolicy =
    cutlass::gemm::collective::MainloopSm120TmaWarpSpecializedBlockScaledCompactScale<
        CollectiveMainloop::DispatchPolicy::Stages,
        CollectiveMainloop::DispatchPolicy::SchedulerPipelineStageCount,
        typename CollectiveMainloop::DispatchPolicy::ClusterShape,
        typename CollectiveMainloop::DispatchPolicy::Schedule,
        true,
        true>;

using CompactScaleAsymmetricNoCopyBDispatchPolicy =
    cutlass::gemm::collective::MainloopSm120TmaWarpSpecializedBlockScaledCompactScale<
        CollectiveMainloop::DispatchPolicy::Stages,
        CollectiveMainloop::DispatchPolicy::SchedulerPipelineStageCount,
        typename CollectiveMainloop::DispatchPolicy::ClusterShape,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseBlockScaledSm120<
            CollectiveMainloop::DispatchPolicy::SchedulerPipelineStageCount,
            false>,
        true,
        true>;

using CompactScaleAsymmetricAColumnwiseSmemDispatchPolicy =
    cutlass::gemm::collective::MainloopSm120TmaWarpSpecializedBlockScaledCompactScale<
        CollectiveMainloop::DispatchPolicy::Stages,
        CollectiveMainloop::DispatchPolicy::SchedulerPipelineStageCount,
        typename CollectiveMainloop::DispatchPolicy::ClusterShape,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseBlockScaledSm120<
            CollectiveMainloop::DispatchPolicy::SchedulerPipelineStageCount,
            false>,
        false,
        false,
        true>;

using CompactScaleAsymmetricAColumnwiseSmemBTmaEarlyDispatchPolicy =
    cutlass::gemm::collective::MainloopSm120TmaWarpSpecializedBlockScaledCompactScale<
        CollectiveMainloop::DispatchPolicy::Stages,
        CollectiveMainloop::DispatchPolicy::SchedulerPipelineStageCount,
        typename CollectiveMainloop::DispatchPolicy::ClusterShape,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseBlockScaledSm120<
            CollectiveMainloop::DispatchPolicy::SchedulerPipelineStageCount,
            false>,
        false,
        false,
        true,
        true>;

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
using CompactScaleDirectGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CompactScaleCollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler>;
using CompactScaleDirectGemm =
    cutlass::gemm::device::GemmUniversalAdapter<CompactScaleDirectGemmKernel>;
using CompactScaleAsymmetricDirectGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CompactScaleAsymmetricCollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler>;
using CompactScaleAsymmetricDirectGemm =
    cutlass::gemm::device::GemmUniversalAdapter<CompactScaleAsymmetricDirectGemmKernel>;
using CompactScaleStrideA = typename CompactScaleGemm::GemmKernel::StrideA;
using CompactScaleStrideB = typename CompactScaleGemm::GemmKernel::StrideB;
using CompactScaleStrideC = typename CompactScaleGemm::GemmKernel::StrideC;
using CompactScaleStrideD = typename CompactScaleGemm::GemmKernel::StrideD;
using NoCopyBStride = cutlass::gemm::collective::CppMegaNoCopyBStride;

using CompactScaleNoCopyBCollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    CompactScaleNoCopyBDispatchPolicy,
    ThreadBlockShape,
    cute::tuple<KernelElementA, KernelElementSF>,
    cute::tuple<StrideA, LayoutSFA>,
    cute::tuple<KernelElementB, KernelElementSF>,
    cute::tuple<NoCopyBStride, LayoutSFB>,
    typename CollectiveMainloop::TiledMma,
    typename CollectiveMainloop::GmemTiledCopyPairA,
    typename CollectiveMainloop::SmemLayoutAtomsA,
    typename CollectiveMainloop::SmemCopyAtomsA,
    cute::identity,
    typename CollectiveMainloop::GmemTiledCopyPairB,
    typename CollectiveMainloop::SmemLayoutAtomsB,
    typename CollectiveMainloop::SmemCopyAtomsB,
    cute::identity>;

using CompactScaleAsymmetricNoCopyBCollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    CompactScaleAsymmetricNoCopyBDispatchPolicy,
    ThreadBlockShape,
    cute::tuple<KernelElementA, KernelElementSF>,
    cute::tuple<StrideA, LayoutSFA>,
    cute::tuple<KernelElementB, KernelElementSF>,
    cute::tuple<NoCopyBStride, LayoutSFB>,
    typename CollectiveMainloop::TiledMma,
    typename CollectiveMainloop::GmemTiledCopyPairA,
    typename CollectiveMainloop::SmemLayoutAtomsA,
    typename CollectiveMainloop::SmemCopyAtomsA,
    cute::identity,
    typename CollectiveMainloop::GmemTiledCopyPairB,
    typename CollectiveMainloop::SmemLayoutAtomsB,
    typename CollectiveMainloop::SmemCopyAtomsB,
    cute::identity>;

using CompactScaleNoCopyBGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CompactScaleNoCopyBCollectiveMainloop,
    CollectiveEpilogue,
    void>;
using CompactScaleNoCopyBGemm = cutlass::gemm::device::GemmUniversalAdapter<CompactScaleNoCopyBGemmKernel>;

using CompactScaleAsymmetricNoCopyBGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CompactScaleAsymmetricNoCopyBCollectiveMainloop,
    CollectiveEpilogue,
    void>;
using CompactScaleAsymmetricNoCopyBGemm =
    cutlass::gemm::device::GemmUniversalAdapter<CompactScaleAsymmetricNoCopyBGemmKernel>;

using CompactScaleAsymmetricAColumnwiseSmemCollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    CompactScaleAsymmetricAColumnwiseSmemDispatchPolicy,
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

using CompactScaleAsymmetricAColumnwiseSmemBTmaEarlyCollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    CompactScaleAsymmetricAColumnwiseSmemBTmaEarlyDispatchPolicy,
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

using CompactScaleAsymmetricAColumnwiseSmemGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CompactScaleAsymmetricAColumnwiseSmemCollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler>;
using CompactScaleAsymmetricAColumnwiseSmemGemm =
    cutlass::gemm::device::GemmUniversalAdapter<CompactScaleAsymmetricAColumnwiseSmemGemmKernel>;

using CompactScaleAsymmetricAColumnwiseSmemBTmaEarlyGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CompactScaleAsymmetricAColumnwiseSmemBTmaEarlyCollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler>;
using CompactScaleAsymmetricAColumnwiseSmemBTmaEarlyGemm =
    cutlass::gemm::device::GemmUniversalAdapter<CompactScaleAsymmetricAColumnwiseSmemBTmaEarlyGemmKernel>;

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

void validate_swizzled_rowwise_inputs(
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
              "CUTLASS MXFP8 swizzled-scale backend currently requires M/N/K multiples of 128, got ",
              m, "x", n, "x", k);
  TORCH_CHECK(A_u8.dim() == 2 && A_u8.size(0) >= m && A_u8.size(1) >= k,
              "A rowwise payload must be at least [M, K]");
  TORCH_CHECK(B_u8.dim() == 2 && B_u8.size(0) >= n && B_u8.size(1) >= k,
              "B rowwise payload must be at least [N, K]");

  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
  int64_t expected_sfa = static_cast<int64_t>(size(filter_zeros(layout_SFA)));
  int64_t expected_sfb = static_cast<int64_t>(size(filter_zeros(layout_SFB)));
  TORCH_CHECK(SFA_u8.numel() >= expected_sfa,
              "SFA GEMM-swizzled scale storage is too small: got ",
              SFA_u8.numel(), ", need at least ", expected_sfa);
  TORCH_CHECK(SFB_u8.numel() >= expected_sfb,
              "SFB GEMM-swizzled scale storage is too small: got ",
              SFB_u8.numel(), ", need at least ", expected_sfb);
}

__device__ __forceinline__ int64_t rowwise_gemm_swizzled_scale_offset(
    int row,
    int k_block,
    int rows,
    int k) {
  // Matches the TE/CUTLASS MXFP8 rowwise scale transform:
  //   scale.view(rows/128, 4, 32, k/128, 4)
  //        .permute(0, 3, 2, 1, 4)
  //        .contiguous()
  int row_tile = row / 128;
  int row_in_tile = row - row_tile * 128;
  int row_group = row_in_tile / 32;
  int row_lane = row_in_tile - row_group * 32;
  int k_group = k_block / 4;
  int k_lane = k_block - k_group * 4;
  int k_groups = k / 128;
  (void)rows;
  return (((static_cast<int64_t>(row_tile) * k_groups + k_group) * 32 + row_lane) * 4 +
          row_group) * 4 + k_lane;
}

__global__ void prepare_wgrad_stock_a_tile_kernel(
    uint8_t const* __restrict__ dy_colwise,
    uint8_t const* __restrict__ dy_colwise_scale,
    uint8_t* __restrict__ a_tile,
    uint8_t* __restrict__ sfa_tile,
    int m_start,
    int tile_m,
    int k,
    int dy_data_ld,
    int dy_scale_ld) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  __shared__ uint8_t transpose_tile[kTile][kTile + 1];

  int row_base = blockIdx.x * kTile;
  int k_base = blockIdx.y * kTile;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  for (int j = 0; j < kTile; j += kBlockRows) {
    int row = row_base + tx;
    int kk = k_base + ty + j;
    transpose_tile[ty + j][tx] =
        dy_colwise[static_cast<int64_t>(kk) * dy_data_ld + (m_start + row)];
  }
  __syncthreads();

  for (int j = 0; j < kTile; j += kBlockRows) {
    int row = row_base + ty + j;
    int kk = k_base + tx;
    a_tile[static_cast<int64_t>(row) * k + kk] = transpose_tile[tx][ty + j];
  }

  int k_blocks = k / 32;
  int64_t thread =
      (static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
      threadIdx.y * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * blockDim.y * gridDim.x * gridDim.y;
  int64_t scale_elems = static_cast<int64_t>(tile_m) * k_blocks;
  for (int64_t idx = thread; idx < scale_elems; idx += stride) {
    int k_block = static_cast<int>(idx / tile_m);
    int row = static_cast<int>(idx - static_cast<int64_t>(k_block) * tile_m);
    int64_t src = static_cast<int64_t>(k_block) * dy_scale_ld + (m_start + row);
    int64_t dst = rowwise_gemm_swizzled_scale_offset(row, k_block, tile_m, k);
    sfa_tile[dst] = dy_colwise_scale[src];
  }
}

__global__ void prepare_wgrad_stock_b_scale_tile_kernel(
    uint8_t const* __restrict__ x_t_rowwise_scale,
    uint8_t* __restrict__ sfb_tile,
    int n_start,
    int tile_n,
    int k,
    int x_t_scale_ld) {
  int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int k_blocks = k / 32;
  int64_t scale_elems = static_cast<int64_t>(tile_n) * k_blocks;
  for (int64_t idx = thread; idx < scale_elems; idx += stride) {
    int row = static_cast<int>(idx / k_blocks);
    int k_block = static_cast<int>(idx - static_cast<int64_t>(row) * k_blocks);
    int64_t src = static_cast<int64_t>(n_start + row) * x_t_scale_ld + k_block;
    int64_t dst = rowwise_gemm_swizzled_scale_offset(row, k_block, tile_n, k);
    sfb_tile[dst] = x_t_rowwise_scale[src];
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
    int64_t d_offset,
    int64_t d_ld,
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
  if (d_ld > 0) {
    cute::get<0>(stride_C) = d_ld;
    cute::get<0>(stride_D) = d_ld;
  }

  auto ptr_A = reinterpret_cast<KernelElementA const*>(A_u8.data_ptr<uint8_t>());
  auto ptr_B = reinterpret_cast<KernelElementB const*>(B_u8.data_ptr<uint8_t>());
  auto ptr_SFA = reinterpret_cast<KernelElementSF const*>(SFA_u8.data_ptr<uint8_t>());
  auto ptr_SFB = reinterpret_cast<KernelElementSF const*>(SFB_u8.data_ptr<uint8_t>());
  auto ptr_D = reinterpret_cast<ElementD*>(D.data_ptr<at::BFloat16>()) + d_offset;
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
at::Tensor run_swizzled_scale_gemm_impl(
    at::Tensor const& A_u8,
    at::Tensor const& SFA_u8,
    at::Tensor const& B_u8,
    at::Tensor const& SFB_u8,
    int m, int n, int k,
    at::Tensor& D,
    int64_t d_offset,
    int64_t d_ld,
    float alpha, float beta,
    cudaStream_t stream) {

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  int l = 1;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, l));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, l));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, l));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, l));
  if (d_ld > 0) {
    cute::get<0>(stride_C) = d_ld;
    cute::get<0>(stride_D) = d_ld;
  }
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, l));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, l));

  auto ptr_A = reinterpret_cast<KernelElementA const*>(A_u8.data_ptr<uint8_t>());
  auto ptr_B = reinterpret_cast<KernelElementB const*>(B_u8.data_ptr<uint8_t>());
  auto ptr_SFA = reinterpret_cast<KernelElementSF const*>(SFA_u8.data_ptr<uint8_t>());
  auto ptr_SFB = reinterpret_cast<KernelElementSF const*>(SFB_u8.data_ptr<uint8_t>());
  auto ptr_D = reinterpret_cast<ElementD*>(D.data_ptr<at::BFloat16>()) + d_offset;
  ElementD* ptr_C = (beta == 0.0f) ? nullptr : ptr_D;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {
          ptr_A, stride_A,
          ptr_B, stride_B,
          ptr_SFA, layout_SFA,
          ptr_SFB, layout_SFB,
      },
      {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D}};

  Gemm gemm;
  cutlass::Status can_status = gemm.can_implement(arguments);
  TORCH_CHECK(can_status == cutlass::Status::kSuccess,
              "CUTLASS MXFP8 swizzled-scale can_implement failed: ",
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
              "CUTLASS MXFP8 swizzled-scale initialize failed: ",
              cutlass_status_name(init_status));
  cutlass::Status run_status = gemm.run(stream);
  TORCH_CHECK(run_status == cutlass::Status::kSuccess,
              "CUTLASS MXFP8 swizzled-scale run failed: ",
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
  if constexpr (cute::is_same_v<StrideB, NoCopyBStride>) {
    // TE compact columnwise payload is physically [K, N] row-major.  The
    // logical TN B operand is [N, K] column-major, so the same bytes can be
    // consumed by stock TMA with leading dimension N (or a padded b_data_ld).
    // Only A-columnwise still needs the manual transpose loader.
    if (b_source == kOperandColumnwiseTranspose) {
      cute::get<1>(stride_B) = b_data_ld;
    }
  }
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
      0, 0,
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

  return run_compact_direct_gemm_impl<CompactScaleDirectGemm>(
      A_u8, SFA_u8, B_u8, SFB_u8, m, n, k,
      a_source, a_data_ld, a_scale_ld,
      b_source, b_data_ld, b_scale_ld,
      D, alpha_f, beta_f, stream);
#endif
}

at::Tensor cutlass_mxfp8_tn_gemm_swizzled_scale_cuda(
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
  TORCH_CHECK(false, "CUTLASS SM120/SM121 MXFP8 swizzled-scale backend was not compiled for this architecture");
#else
  TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "M/N/K exceed int");
  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);

  validate_swizzled_rowwise_inputs(A_u8, SFA_u8, B_u8, SFB_u8, m, n, k);
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

  (void)accumulate;
  float alpha_f = static_cast<float>(alpha);
  float beta_f = static_cast<float>(beta);

  return run_swizzled_scale_gemm_impl<Gemm>(
      A_u8, SFA_u8, B_u8, SFB_u8, m, n, k, D,
      0, 0,
      alpha_f, beta_f, stream);
#endif
}

at::Tensor cutlass_mxfp8_tn_gemm_swizzled_scale_strided_cuda(
    at::Tensor A_u8,
    at::Tensor SFA_u8,
    at::Tensor B_u8,
    at::Tensor SFB_u8,
    int64_t m64,
    int64_t n64,
    int64_t k64,
    at::Tensor out,
    int64_t out_ld,
    int64_t out_offset,
    bool accumulate,
    double alpha,
    double beta) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  TORCH_CHECK(false, "CUTLASS SM120/SM121 MXFP8 swizzled-scale strided probe was not compiled for this architecture");
#else
  TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "M/N/K exceed int");
  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);

  validate_swizzled_rowwise_inputs(A_u8, SFA_u8, B_u8, SFB_u8, m, n, k);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(out);
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bfloat16");
  TORCH_CHECK(out_ld >= n, "out_ld must be >= tile N");
  TORCH_CHECK(out_offset >= 0, "out_offset must be non-negative");
  TORCH_CHECK(out.numel() >= out_offset + static_cast<int64_t>(m - 1) * out_ld + n,
              "out is too small for strided tile");

  c10::cuda::CUDAGuard device_guard(A_u8.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  at::Tensor D = out;

  (void)accumulate;
  float alpha_f = static_cast<float>(alpha);
  float beta_f = static_cast<float>(beta);

  return run_swizzled_scale_gemm_impl<Gemm>(
      A_u8, SFA_u8, B_u8, SFB_u8, m, n, k, D,
      out_offset, out_ld,
      alpha_f, beta_f, stream);
#endif
}

void cutlass_mxfp8_prepare_wgrad_stock_a_tile_cuda(
    at::Tensor dy_colwise_u8,
    at::Tensor dy_colwise_scale_u8,
    at::Tensor a_tile_u8,
    at::Tensor sfa_tile_u8,
    int64_t m_start64,
    int64_t tile_m64,
    int64_t k64,
    int64_t dy_data_ld64,
    int64_t dy_scale_ld64) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  TORCH_CHECK(false, "CUTLASS SM120/SM121 MXFP8 stock-tile prepare kernel was not compiled for this architecture");
#else
  TORCH_CHECK(m_start64 >= 0, "m_start must be non-negative");
  TORCH_CHECK(tile_m64 > 0 && k64 > 0, "tile_m and k must be positive");
  TORCH_CHECK(tile_m64 <= INT_MAX && k64 <= INT_MAX &&
              dy_data_ld64 <= INT_MAX && dy_scale_ld64 <= INT_MAX,
              "tile_m/k/leading dimensions exceed int");
  int m_start = static_cast<int>(m_start64);
  int tile_m = static_cast<int>(tile_m64);
  int k = static_cast<int>(k64);
  int dy_data_ld = static_cast<int>(dy_data_ld64);
  int dy_scale_ld = static_cast<int>(dy_scale_ld64);
  TORCH_CHECK(tile_m % 128 == 0 && k % 128 == 0,
              "A stock tile prepare requires tile_m and K multiples of 128");
  CHECK_CUDA(dy_colwise_u8);
  CHECK_CUDA(dy_colwise_scale_u8);
  CHECK_CUDA(a_tile_u8);
  CHECK_CUDA(sfa_tile_u8);
  CHECK_CONTIGUOUS(dy_colwise_u8);
  CHECK_CONTIGUOUS(dy_colwise_scale_u8);
  CHECK_CONTIGUOUS(a_tile_u8);
  CHECK_CONTIGUOUS(sfa_tile_u8);
  CHECK_UINT8(dy_colwise_u8);
  CHECK_UINT8(dy_colwise_scale_u8);
  CHECK_UINT8(a_tile_u8);
  CHECK_UINT8(sfa_tile_u8);
  TORCH_CHECK(dy_colwise_u8.numel() >= static_cast<int64_t>(k - 1) * dy_data_ld + m_start + tile_m,
              "dy_colwise_u8 is too small for requested tile");
  TORCH_CHECK(dy_colwise_scale_u8.numel() >= static_cast<int64_t>(k / 32 - 1) * dy_scale_ld + m_start + tile_m,
              "dy_colwise_scale_u8 is too small for requested tile");
  TORCH_CHECK(a_tile_u8.numel() >= static_cast<int64_t>(tile_m) * k,
              "a_tile_u8 is too small");
  TORCH_CHECK(sfa_tile_u8.numel() >= static_cast<int64_t>(tile_m) * (k / 32),
              "sfa_tile_u8 is too small");

  c10::cuda::CUDAGuard device_guard(dy_colwise_u8.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(32, 8, 1);
  dim3 grid(tile_m / 32, k / 32, 1);
  prepare_wgrad_stock_a_tile_kernel<<<grid, block, 0, stream>>>(
      dy_colwise_u8.data_ptr<uint8_t>(),
      dy_colwise_scale_u8.data_ptr<uint8_t>(),
      a_tile_u8.data_ptr<uint8_t>(),
      sfa_tile_u8.data_ptr<uint8_t>(),
      m_start,
      tile_m,
      k,
      dy_data_ld,
      dy_scale_ld);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

void cutlass_mxfp8_prepare_wgrad_stock_b_scale_tile_cuda(
    at::Tensor x_t_rowwise_scale_u8,
    at::Tensor sfb_tile_u8,
    int64_t n_start64,
    int64_t tile_n64,
    int64_t k64,
    int64_t x_t_scale_ld64) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  TORCH_CHECK(false, "CUTLASS SM120/SM121 MXFP8 stock B-scale prepare kernel was not compiled for this architecture");
#else
  TORCH_CHECK(n_start64 >= 0, "n_start must be non-negative");
  TORCH_CHECK(tile_n64 > 0 && k64 > 0, "tile_n and k must be positive");
  TORCH_CHECK(tile_n64 <= INT_MAX && k64 <= INT_MAX && x_t_scale_ld64 <= INT_MAX,
              "tile_n/k/leading dimension exceed int");
  int n_start = static_cast<int>(n_start64);
  int tile_n = static_cast<int>(tile_n64);
  int k = static_cast<int>(k64);
  int x_t_scale_ld = static_cast<int>(x_t_scale_ld64);
  TORCH_CHECK(tile_n % 128 == 0 && k % 128 == 0,
              "B stock scale tile prepare requires tile_n and K multiples of 128");
  CHECK_CUDA(x_t_rowwise_scale_u8);
  CHECK_CUDA(sfb_tile_u8);
  CHECK_CONTIGUOUS(x_t_rowwise_scale_u8);
  CHECK_CONTIGUOUS(sfb_tile_u8);
  CHECK_UINT8(x_t_rowwise_scale_u8);
  CHECK_UINT8(sfb_tile_u8);
  TORCH_CHECK(x_t_rowwise_scale_u8.numel() >=
                  static_cast<int64_t>(n_start + tile_n - 1) * x_t_scale_ld + (k / 32),
              "x_t_rowwise_scale_u8 is too small for requested tile");
  TORCH_CHECK(sfb_tile_u8.numel() >= static_cast<int64_t>(tile_n) * (k / 32),
              "sfb_tile_u8 is too small");

  c10::cuda::CUDAGuard device_guard(x_t_rowwise_scale_u8.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t work = static_cast<int64_t>(tile_n) * (k / 32);
  int threads = 256;
  int64_t blocks64 = (work + threads - 1) / threads;
  int blocks = static_cast<int>(blocks64 > 1024 ? 1024 : blocks64);
  prepare_wgrad_stock_b_scale_tile_kernel<<<blocks, threads, 0, stream>>>(
      x_t_rowwise_scale_u8.data_ptr<uint8_t>(),
      sfb_tile_u8.data_ptr<uint8_t>(),
      n_start,
      tile_n,
      k,
      x_t_scale_ld);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

  return run_compact_direct_gemm_impl<CompactScaleAsymmetricDirectGemm>(
      A_u8, SFA_u8, B_u8, SFB_u8, m, n, k,
      a_source, a_data_ld, a_scale_ld,
      b_source, b_data_ld, b_scale_ld,
      D, alpha_f, beta_f, stream);
#endif
}

at::Tensor cutlass_mxfp8_tn_gemm_compact_direct_a_col_smem_asym_cuda(
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
  TORCH_CHECK(false, "CUTLASS SM120/SM121 MXFP8 compact direct A-columnwise-smem asymmetric backend was not compiled for this architecture");
#else
  TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "M/N/K exceed int");
  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);

  TORCH_CHECK(m > 0 && n > 0 && k > 0, "m, n, k must be positive");
  TORCH_CHECK(m % 128 == 0 && n % 128 == 0 && k % 128 == 0,
              "CUTLASS MXFP8 compact direct A-columnwise-smem asymmetric backend currently requires M/N/K multiples of 128, got ",
              m, "x", n, "x", k);
  TORCH_CHECK(a_source == kOperandColumnwiseTranspose,
              "A-columnwise-smem backend requires A source to be columnwise-transpose");
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

  (void)accumulate;
  float alpha_f = static_cast<float>(alpha);
  float beta_f = static_cast<float>(beta);

  return run_compact_direct_gemm_impl<CompactScaleAsymmetricAColumnwiseSmemGemm>(
      A_u8, SFA_u8, B_u8, SFB_u8, m, n, k,
      a_source, a_data_ld, a_scale_ld,
      b_source, b_data_ld, b_scale_ld,
      D, alpha_f, beta_f, stream);
#endif
}

at::Tensor cutlass_mxfp8_tn_gemm_compact_direct_a_col_smem_b_tma_early_asym_cuda(
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
  TORCH_CHECK(false, "CUTLASS SM120/SM121 MXFP8 compact direct A-columnwise-smem B-TMA-early asymmetric backend was not compiled for this architecture");
#else
  TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "M/N/K exceed int");
  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);

  TORCH_CHECK(m > 0 && n > 0 && k > 0, "m, n, k must be positive");
  TORCH_CHECK(m % 128 == 0 && n % 128 == 0 && k % 128 == 0,
              "CUTLASS MXFP8 compact direct A-columnwise-smem B-TMA-early asymmetric backend currently requires M/N/K multiples of 128, got ",
              m, "x", n, "x", k);
  TORCH_CHECK(a_source == kOperandColumnwiseTranspose,
              "A-columnwise-smem B-TMA-early backend requires A source to be columnwise-transpose");
  TORCH_CHECK(b_source == kOperandRowwise,
              "A-columnwise-smem B-TMA-early backend requires B source to be rowwise");
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

  (void)accumulate;
  float alpha_f = static_cast<float>(alpha);
  float beta_f = static_cast<float>(beta);

  return run_compact_direct_gemm_impl<CompactScaleAsymmetricAColumnwiseSmemBTmaEarlyGemm>(
      A_u8, SFA_u8, B_u8, SFB_u8, m, n, k,
      a_source, a_data_ld, a_scale_ld,
      b_source, b_data_ld, b_scale_ld,
      D, alpha_f, beta_f, stream);
#endif
}
