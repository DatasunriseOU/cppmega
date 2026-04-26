/***************************************************************************************************
 * Scoped cppmega probe for CUTLASS/CuTe SM121 MXFP8 block-scaled GEMM layouts.
 *
 * This file intentionally lives outside the production build.  It checks two things:
 *
 * 1. The native CUTLASS SM120/SM121 block-scaled MXFP8 GEMM path can be
 *    instantiated and run on a small problem using CUTLASS-owned scale layouts.
 * 2. The native scale-factor layout offsets are not a simple alias of TE compact
 *    rowwise scales or TE compact columnwise scales for a logical transpose.
 **************************************************************************************************/

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <string>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

namespace {

using namespace cute;

struct Options {
  int m = 128;
  int n = 128;
  int k = 128;
  bool run_gemm = true;
  bool layout_only = false;
  bool attempt_te_compact = false;
};

template <class T>
bool parse_int_arg(char const* arg, char const* name, T& out) {
  std::string prefix = std::string("--") + name + "=";
  std::string value(arg);
  if (value.rfind(prefix, 0) != 0) {
    return false;
  }
  out = static_cast<T>(std::atoi(value.c_str() + prefix.size()));
  return true;
}

Options parse_options(int argc, char const** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--layout-only") {
      options.layout_only = true;
      options.run_gemm = false;
    } else if (arg == "--attempt-te-compact") {
      options.attempt_te_compact = true;
    } else if (arg == "--no-gemm") {
      options.run_gemm = false;
    } else if (!parse_int_arg(argv[i], "m", options.m) &&
               !parse_int_arg(argv[i], "n", options.n) &&
               !parse_int_arg(argv[i], "k", options.k)) {
      std::cerr << "unknown argument: " << arg << "\n";
      std::exit(2);
    }
  }
  return options;
}

void check_cuda(cudaError_t error, char const* expr, char const* file, int line) {
  if (error != cudaSuccess) {
    std::cerr << file << ":" << line << " CUDA error for " << expr << ": "
              << cudaGetErrorString(error) << "\n";
    std::exit(1);
  }
}

void check_cutlass(cutlass::Status status, char const* expr, char const* file, int line) {
  if (status != cutlass::Status::kSuccess) {
    std::cerr << file << ":" << line << " CUTLASS error for " << expr << ": "
              << cutlassGetStatusString(status) << "\n";
    std::exit(1);
  }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUTLASS_CHECK(expr) check_cutlass((expr), #expr, __FILE__, __LINE__)

template <class T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

long long te_compact_rowwise_offset(int logical_rows, int logical_k, int row, int k) {
  (void)logical_rows;
  return static_cast<long long>(row) * (logical_k / 32) + (k / 32);
}

long long te_columnwise_transpose_alias_offset(int logical_rows, int logical_k, int row, int k) {
  (void)logical_k;
  return static_cast<long long>(k / 32) * logical_rows + row;
}

void print_layout_probe(int logical_rows, int logical_k) {
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<32>;
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(logical_rows, 1, logical_k, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(1, logical_rows, logical_k, 1));
  auto te_columnwise_transpose_layout = make_layout(
      make_shape(logical_rows, make_shape(Int<32>{}, logical_k / 32), Int<1>{}),
      make_stride(Int<1>{}, make_stride(Int<0>{}, logical_rows), Int<0>{}));

  long long rowwise_mismatches = 0;
  long long columnwise_alias_mismatches = 0;
  long long cute_columnwise_alias_mismatches = 0;
  long long checked = 0;
  for (int row = 0; row < logical_rows; ++row) {
    for (int k = 0; k < logical_k; k += 32) {
      long long native = static_cast<long long>(layout_sfb(row, k, 0));
      long long rowwise = te_compact_rowwise_offset(logical_rows, logical_k, row, k);
      long long columnwise_alias =
          te_columnwise_transpose_alias_offset(logical_rows, logical_k, row, k);
      long long cute_columnwise_alias =
          static_cast<long long>(te_columnwise_transpose_layout(row, k, 0));
      rowwise_mismatches += native != rowwise;
      columnwise_alias_mismatches += native != columnwise_alias;
      cute_columnwise_alias_mismatches += cute_columnwise_alias != columnwise_alias;
      checked += 1;
    }
  }

  std::cout << "layout_probe\n";
  std::cout << "  logical_rows=" << logical_rows << " logical_k=" << logical_k
            << " sf_vec=32\n";
  std::cout << "  native_sfa_storage_elems=" << size(filter_zeros(layout_sfa))
            << " native_sfb_storage_elems=" << size(filter_zeros(layout_sfb)) << "\n";
  std::cout << "  te_compact_scale_elems=" << (logical_rows * (logical_k / 32)) << "\n";
  std::cout << "  checked_scale_slots=" << checked << "\n";
  std::cout << "  native_vs_te_rowwise_mismatches=" << rowwise_mismatches << "\n";
  std::cout << "  native_vs_te_columnwise_transpose_alias_mismatches="
            << columnwise_alias_mismatches << "\n";
  std::cout << "  cute_layout_vs_te_columnwise_transpose_alias_mismatches="
            << cute_columnwise_alias_mismatches << "\n";
  std::cout << "  sample_offsets row,k native te_rowwise te_colwise_alias cute_te_alias\n";
  int sample_rows[] = {0, 1, 31, 32, 33, 64, 96};
  for (int row : sample_rows) {
    if (row >= logical_rows) {
      continue;
    }
    for (int k = 0; k < logical_k && k < 128; k += 32) {
      std::cout << "    " << row << "," << k << " " << layout_sfb(row, k, 0) << " "
                << te_compact_rowwise_offset(logical_rows, logical_k, row, k) << " "
                << te_columnwise_transpose_alias_offset(logical_rows, logical_k, row, k)
                << " " << te_columnwise_transpose_layout(row, k, 0)
                << "\n";
    }
  }
}

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

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
using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

using ElementAData = typename ElementA::DataType;
using ElementBData = typename ElementB::DataType;
using ElementSF = typename ElementA::ScaleFactorType;

using TECompactRowwiseScaleLayout = decltype(make_layout(
    make_shape(int32_t{}, make_shape(Int<32>{}, int32_t{}), Int<1>{}),
    make_stride(int64_t{}, make_stride(Int<0>{}, Int<1>{}), Int<0>{})));

using TECompactColwiseTransposeScaleLayout = decltype(make_layout(
    make_shape(int32_t{}, make_shape(Int<32>{}, int32_t{}), Int<1>{}),
    make_stride(Int<1>{}, make_stride(Int<0>{}, int64_t{}), Int<0>{})));

TECompactRowwiseScaleLayout make_te_compact_rowwise_scale_layout(int rows, int k) {
  return make_layout(
      make_shape(rows, make_shape(Int<32>{}, k / 32), Int<1>{}),
      make_stride(static_cast<int64_t>(k / 32), make_stride(Int<0>{}, Int<1>{}), Int<0>{}));
}

TECompactColwiseTransposeScaleLayout make_te_compact_colwise_transpose_scale_layout(
    int rows,
    int k) {
  return make_layout(
      make_shape(rows, make_shape(Int<32>{}, k / 32), Int<1>{}),
      make_stride(Int<1>{}, make_stride(Int<0>{}, static_cast<int64_t>(rows)), Int<0>{}));
}

using TECompactCollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    typename CollectiveMainloop::DispatchPolicy,
    ThreadBlockShape,
    cute::tuple<ElementAData, ElementSF>,
    cute::tuple<StrideA, TECompactRowwiseScaleLayout>,
    cute::tuple<ElementBData, ElementSF>,
    cute::tuple<StrideB, TECompactColwiseTransposeScaleLayout>,
    typename CollectiveMainloop::TiledMma,
    typename CollectiveMainloop::GmemTiledCopyPairA,
    typename CollectiveMainloop::SmemLayoutAtomsA,
    typename CollectiveMainloop::SmemCopyAtomsA,
    cute::identity,
    typename CollectiveMainloop::GmemTiledCopyPairB,
    typename CollectiveMainloop::SmemLayoutAtomsB,
    typename CollectiveMainloop::SmemCopyAtomsB,
    cute::identity>;

using TECompactGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    TECompactCollectiveMainloop,
    CollectiveEpilogue,
    void>;

using TECompactGemm = cutlass::gemm::device::GemmUniversalAdapter<TECompactGemmKernel>;

template <typename Element, typename Layout>
bool initialize_block(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
  double scope_max = 1;
  double scope_min = -1;
  if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
    scope_max = 4;
    scope_min = 1;
  }
  cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);
  return true;
}

template <class DstLayout, class SrcLayout>
void materialize_scale_layout(
    ElementSF* dst,
    DstLayout const& dst_layout,
    ElementSF const* src,
    SrcLayout const& src_layout,
    int rows,
    int k) {
  for (int row = 0; row < rows; ++row) {
    for (int kk = 0; kk < k; kk += 32) {
      dst[dst_layout(row, kk, 0)] = src[src_layout(row, kk, 0)];
    }
  }
}

int run_native_gemm(Options const& options) {
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  LayoutA layout_A = make_layout(make_shape(options.m, options.k, 1), stride_A);
  LayoutB layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
  LayoutC layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C);
  LayoutD layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);
  LayoutSFA layout_SFA =
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(options.m, options.n, options.k, 1));
  LayoutSFB layout_SFB =
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(options.m, options.n, options.k, 1));

  cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
  cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
  cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
  cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;

  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

  initialize_block(block_A.host_view(), 2021);
  initialize_block(block_B.host_view(), 2022);
  initialize_block(block_C.host_view(), 2023);
  initialize_block(block_SFA.host_view(), 2024);
  initialize_block(block_SFB.host_view(), 2025);

  block_A.sync_device();
  block_B.sync_device();
  block_C.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, 1},
      {block_A.device_data(), stride_A, block_B.device_data(), stride_B, block_SFA.device_data(),
       layout_SFA, block_SFB.device_data(), layout_SFB},
      {{1.0f, 0.0f}, block_C.device_data(), stride_C, block_D.device_data(), stride_D}};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
  Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
  Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
  Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);
  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,
      decltype(tensor_A),
      decltype(tensor_SFA),
      decltype(tensor_B),
      decltype(tensor_SFB)>
      mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

  auto tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C);
  auto tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);
  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator,
      ElementAccumulator,
      ElementAccumulator,
      decltype(tensor_C),
      decltype(tensor_D)>
      epilogue_params{1.0f, 0.0f, tensor_C, tensor_D};
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  block_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D.host_view());
  passed &= cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0;
  passed &= cutlass::reference::host::TensorNorm(block_D.host_view()) > 0;

  std::cout << "native_gemm\n";
  std::cout << "  problem=" << options.m << "x" << options.n << "x" << options.k << "\n";
  std::cout << "  layout=TN A=row-major B=column-major\n";
  std::cout << "  scale_layout=native_sm1xx_blockscaled\n";
  std::cout << "  disposition=" << (passed ? "passed" : "failed") << "\n";
  return passed ? 0 : 1;
}

int run_te_compact_gemm(Options const& options) {
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  LayoutA layout_A = make_layout(make_shape(options.m, options.k, 1), stride_A);
  LayoutB layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
  LayoutC layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C);
  LayoutD layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);
  LayoutSFA native_layout_SFA =
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(options.m, options.n, options.k, 1));
  LayoutSFB native_layout_SFB =
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(options.m, options.n, options.k, 1));
  auto te_layout_SFA = make_te_compact_rowwise_scale_layout(options.m, options.k);
  auto te_layout_SFB = make_te_compact_colwise_transpose_scale_layout(options.n, options.k);

  cutlass::HostTensor<ElementAData, cutlass::layout::PackedVectorLayout> block_A;
  cutlass::HostTensor<ElementBData, cutlass::layout::PackedVectorLayout> block_B;
  cutlass::HostTensor<ElementSF, cutlass::layout::PackedVectorLayout> block_te_SFA;
  cutlass::HostTensor<ElementSF, cutlass::layout::PackedVectorLayout> block_te_SFB;
  cutlass::HostTensor<ElementSF, cutlass::layout::PackedVectorLayout> block_native_SFA;
  cutlass::HostTensor<ElementSF, cutlass::layout::PackedVectorLayout> block_native_SFB;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D_te_compact;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D_native_control;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;

  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D_te_compact.reset(cutlass::make_Coord(size(layout_D)));
  block_D_native_control.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
  block_te_SFA.reset(cutlass::make_Coord(size(filter_zeros(te_layout_SFA))));
  block_te_SFB.reset(cutlass::make_Coord(size(filter_zeros(te_layout_SFB))));
  block_native_SFA.reset(cutlass::make_Coord(size(filter_zeros(native_layout_SFA))));
  block_native_SFB.reset(cutlass::make_Coord(size(filter_zeros(native_layout_SFB))));

  initialize_block(block_A.host_view(), 3021);
  initialize_block(block_B.host_view(), 3022);
  initialize_block(block_C.host_view(), 3023);
  initialize_block(block_te_SFA.host_view(), 3024);
  initialize_block(block_te_SFB.host_view(), 3025);
  cutlass::reference::host::TensorFill(block_native_SFA.host_view(), ElementSF{});
  cutlass::reference::host::TensorFill(block_native_SFB.host_view(), ElementSF{});
  materialize_scale_layout(
      block_native_SFA.host_data(),
      native_layout_SFA,
      block_te_SFA.host_data(),
      te_layout_SFA,
      options.m,
      options.k);
  materialize_scale_layout(
      block_native_SFB.host_data(),
      native_layout_SFB,
      block_te_SFB.host_data(),
      te_layout_SFB,
      options.n,
      options.k);

  block_A.sync_device();
  block_B.sync_device();
  block_C.sync_device();
  block_te_SFA.sync_device();
  block_te_SFB.sync_device();
  block_native_SFA.sync_device();
  block_native_SFB.sync_device();

  typename TECompactGemm::Arguments te_arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, 1},
      {block_A.device_data(), stride_A, block_B.device_data(), stride_B,
       block_te_SFA.device_data(), te_layout_SFA, block_te_SFB.device_data(), te_layout_SFB},
      {{1.0f, 0.0f},
       block_C.device_data(),
       stride_C,
       block_D_te_compact.device_data(),
       stride_D}};

  TECompactGemm te_gemm;
  size_t te_workspace_size = TECompactGemm::get_workspace_size(te_arguments);
  cutlass::device_memory::allocation<uint8_t> te_workspace(te_workspace_size);
  CUTLASS_CHECK(te_gemm.can_implement(te_arguments));
  CUTLASS_CHECK(te_gemm.initialize(te_arguments, te_workspace.get()));
  CUTLASS_CHECK(te_gemm.run());

  typename Gemm::Arguments native_arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, 1},
      {block_A.device_data(), stride_A, block_B.device_data(), stride_B,
       block_native_SFA.device_data(), native_layout_SFA, block_native_SFB.device_data(),
       native_layout_SFB},
      {{1.0f, 0.0f},
       block_C.device_data(),
       stride_C,
       block_D_native_control.device_data(),
       stride_D}};

  Gemm native_gemm;
  size_t native_workspace_size = Gemm::get_workspace_size(native_arguments);
  cutlass::device_memory::allocation<uint8_t> native_workspace(native_workspace_size);
  CUTLASS_CHECK(native_gemm.can_implement(native_arguments));
  CUTLASS_CHECK(native_gemm.initialize(native_arguments, native_workspace.get()));
  CUTLASS_CHECK(native_gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
  Tensor tensor_SFA = make_tensor(block_te_SFA.host_data(), te_layout_SFA);
  Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
  Tensor tensor_SFB = make_tensor(block_te_SFB.host_data(), te_layout_SFB);
  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,
      decltype(tensor_A),
      decltype(tensor_SFA),
      decltype(tensor_B),
      decltype(tensor_SFB)>
      mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

  auto tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C);
  auto tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);
  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator,
      ElementAccumulator,
      ElementAccumulator,
      decltype(tensor_C),
      decltype(tensor_D)>
      epilogue_params{1.0f, 0.0f, tensor_C, tensor_D};
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  block_D_te_compact.sync_host();
  block_D_native_control.sync_host();

  bool compact_matches_reference =
      cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D_te_compact.host_view());
  bool compact_matches_native_control =
      cutlass::reference::host::TensorEquals(block_D_native_control.host_view(), block_D_te_compact.host_view());
  bool nonzero = cutlass::reference::host::TensorNorm(block_D_te_compact.host_view()) > 0;
  bool passed = compact_matches_reference && compact_matches_native_control && nonzero;

  std::cout << "te_compact_gemm\n";
  std::cout << "  problem=" << options.m << "x" << options.n << "x" << options.k << "\n";
  std::cout << "  layout=TN A=row-major B=column-major\n";
  std::cout << "  scale_layout=A_te_compact_rowwise,B_te_columnwise_transpose_alias\n";
  std::cout << "  compact_scale_elems_A=" << size(filter_zeros(te_layout_SFA))
            << " compact_scale_elems_B=" << size(filter_zeros(te_layout_SFB)) << "\n";
  std::cout << "  native_control_scale_elems_A=" << size(filter_zeros(native_layout_SFA))
            << " native_control_scale_elems_B=" << size(filter_zeros(native_layout_SFB)) << "\n";
  std::cout << "  compact_matches_cutlass_host_reference="
            << (compact_matches_reference ? "yes" : "no") << "\n";
  std::cout << "  compact_matches_materialized_native_scale_control="
            << (compact_matches_native_control ? "yes" : "no") << "\n";
  std::cout << "  disposition=" << (passed ? "passed" : "failed") << "\n";
  return passed ? 0 : 1;
}

#else

int run_native_gemm(Options const&) {
  std::cout << "native_gemm\n";
  std::cout << "  disposition=skipped\n";
  std::cout << "  reason=CUTLASS_ARCH_MMA_SM120_SUPPORTED/SM121_SUPPORTED not defined\n";
  return 0;
}

int run_te_compact_gemm(Options const&) {
  std::cout << "te_compact_gemm\n";
  std::cout << "  disposition=skipped\n";
  std::cout << "  reason=CUTLASS_ARCH_MMA_SM120_SUPPORTED/SM121_SUPPORTED not defined\n";
  return 0;
}

#endif

}  // namespace

int main(int argc, char const** argv) {
  Options options = parse_options(argc, argv);
  if (options.m % 128 != 0 || options.n % 128 != 0 || options.k % 128 != 0) {
    std::cerr << "--m, --n, and --k must be multiples of 128 for this minimal probe\n";
    return 2;
  }

  print_layout_probe(options.n, options.k);
  if (!options.run_gemm) {
    return 0;
  }

  cudaDeviceProp props{};
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  std::cout << "device\n";
  std::cout << "  name=" << props.name << "\n";
  std::cout << "  compute_capability=" << props.major << "." << props.minor << "\n";
  if (!(props.major == 12 && (props.minor == 0 || props.minor == 1))) {
    std::cout << "native_gemm\n";
    std::cout << "  disposition=skipped\n";
    std::cout << "  reason=requires SM120 or SM121\n";
    return 0;
  }

  int native_status = run_native_gemm(options);
  if (native_status != 0) {
    return native_status;
  }
  if (options.attempt_te_compact) {
    return run_te_compact_gemm(options);
  }
  std::cout << "te_compact_gemm\n";
  std::cout << "  disposition=not_run\n";
  std::cout << "  reason=direct compact scale TMA attempt aborts on current CUTLASS/CUDA; "
               "pass --attempt-te-compact to reproduce\n";
  return 0;
}
