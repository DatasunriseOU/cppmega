/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

enum class CppMegaCompactOperandSource : int32_t {
  kRowwise = 0,
  kColumnwiseTranspose = 1,
};

// TE compact columnwise B payload is physically [K, N] row-major. For the
// logical TN operand [N, K], those bytes alias a column-major B tensor, but the
// SM120 TMA descriptor still needs the K-contiguous shared-memory basis. This
// tag lets cppmega select that descriptor path without changing normal CUTLASS
// row/column-major instantiations.
using CppMegaNoCopyBStride = cute::Stride<cute::Int<1>, int64_t, int64_t>;

template <class Stride>
struct CppMegaUsesNoCopyBStride : cute::false_type {};

template <>
struct CppMegaUsesNoCopyBStride<CppMegaNoCopyBStride> : cute::true_type {};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  int Stages_,
  int SchedulerPipelineStageCount_,
  class ClusterShape_,
  class KernelSchedule_,
  bool UseAuxiliaryLoad_ = false,
  bool UseNoCopyBTma_ = false,
  bool UseAColumnwiseSmemLayout_ = false,
  bool UseBTmaEarly_ = false
>
struct MainloopSm120TmaWarpSpecializedBlockScaledCompactScale {
  constexpr static int Stages = Stages_;
  constexpr static int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  constexpr static bool UseAuxiliaryLoad = UseAuxiliaryLoad_;
  constexpr static bool UseNoCopyBTma = UseNoCopyBTma_;
  constexpr static bool UseAColumnwiseSmemLayout = UseAColumnwiseSmemLayout_;
  constexpr static bool UseBTmaEarly = UseBTmaEarly_;
  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm120;
};

} // namespace cutlass::gemm::collective

namespace cutlass::gemm::kernel::detail {

template<
  int Stages,
  int SchedulerPipelineStageCount,
  class ClusterShape,
  class KernelSchedule,
  bool UseAuxiliaryLoad,
  bool UseNoCopyBTma,
  bool UseAColumnwiseSmemLayout,
  bool UseBTmaEarly
>
struct HasAuxiliaryLoad<
  cutlass::gemm::collective::MainloopSm120TmaWarpSpecializedBlockScaledCompactScale<
    Stages,
    SchedulerPipelineStageCount,
    ClusterShape,
    KernelSchedule,
    UseAuxiliaryLoad,
    UseNoCopyBTma,
    UseAColumnwiseSmemLayout,
    UseBTmaEarly
  >
> : cute::bool_constant<UseAuxiliaryLoad>{};

} // namespace cutlass::gemm::kernel::detail

namespace cutlass::gemm::collective {
using namespace cute;

template <
  int Stages,
  int SchedulerPipelineStageCount,
  class ClusterShape,
  class KernelScheduleType,
  bool UseAuxiliaryLoad,
  bool UseNoCopyBTma,
  bool UseAColumnwiseSmemLayout,
  bool UseBTmaEarly,
  class TileShape_,
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomsA_,
  class SmemCopyAtomsA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomsB_,
  class SmemCopyAtomsB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm120TmaWarpSpecializedBlockScaledCompactScale<
        Stages,
        SchedulerPipelineStageCount,
        ClusterShape,
        KernelScheduleType,
        UseAuxiliaryLoad,
        UseNoCopyBTma,
        UseAColumnwiseSmemLayout,
        UseBTmaEarly>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomsA_,
    SmemCopyAtomsA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomsB_,
    SmemCopyAtomsB_,
    TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm120TmaWarpSpecializedBlockScaledCompactScale<
      Stages,
      SchedulerPipelineStageCount,
      ClusterShape,
      KernelScheduleType,
      UseAuxiliaryLoad,
      UseNoCopyBTma,
      UseAColumnwiseSmemLayout,
      UseBTmaEarly>;
  using TileShape = TileShape_;
  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;

  static_assert(cute::is_same_v<remove_cvref_t<decltype(get<1>(ElementPairA{}))>,
                                remove_cvref_t<decltype(get<1>(ElementPairB{}))>>, "SFA and SFB data types should be the same");

  using RuntimeDataTypeA = void*;
  using RuntimeDataTypeB = void*;

   // A and B matrices
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using StrideA  = remove_cvref_t<decltype(get<0>(StridePairA{}))>;

  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB  = remove_cvref_t<decltype(get<0>(StridePairB{}))>;

  // SFA and SFB
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using LayoutSFA = remove_cvref_t<decltype(get<1>(StridePairA{}))>;
  using LayoutSFB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;

  using ArrayElementA = ElementA;
  using ArrayElementB = ElementB;

  using TiledMma = TiledMma_;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using ElementAccumulator = typename TiledMma::ValTypeC;

  static constexpr int SFVecSize = TiledMma::Traits::SFVecSize;
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;

  // Gmem copies
  using GmemTiledCopyPairA = GmemTiledCopyPairA_;
  using GmemTiledCopyPairB = GmemTiledCopyPairB_;
  using GmemTiledCopyA    = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopySFA  = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopyB    = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairB{}))>;
  using GmemTiledCopySFB  = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairB{}))>;

  // Smem copies
  using SmemLayoutAtomsA = SmemLayoutAtomsA_;
  using SmemLayoutAtomsB = SmemLayoutAtomsB_;

  using SmemLayoutAtomA   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsA{}))>;
  using SmemLayoutAtomSFA = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsA{}))>;
  using SmemLayoutAtomB   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsB{}))>;
  using SmemLayoutAtomSFB = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsB{}))>;

  using SmemCopyAtomsA =  SmemCopyAtomsA_;
  using SmemCopyAtomsB =  SmemCopyAtomsB_;

  using SmemCopyAtomA   = remove_cvref_t<decltype(get<0>(SmemCopyAtomsA{}))>;
  using SmemCopyAtomSFA = remove_cvref_t<decltype(get<1>(SmemCopyAtomsA{}))>;

  using SmemCopyAtomB   = remove_cvref_t<decltype(get<0>(SmemCopyAtomsB{}))>;
  using SmemCopyAtomSFB = remove_cvref_t<decltype(get<1>(SmemCopyAtomsB{}))>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;

  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int ThreadCount = size(TiledMma{});
  // SM120 cooperative TMA schedule uses one producer warp-group.  The compact
  // direct path keeps synchronization warp-local; full CTA sync would deadlock
  // because consumer warp-groups do not execute this producer collective.
  static constexpr int LoadThreadCount = 128;

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using MainloopPipelineMK = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using MainloopPipelineNK = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<DispatchPolicy::Stages>;
  using PipelineStateMK = typename cutlass::PipelineState<DispatchPolicy::Stages>;
  using PipelineStateNK = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  // Generic cooperative split mode uses one TMA leader from the main producer
  // warp plus 32 auxiliary warp arrivals for the B/SFB half. The default direct
  // backend keeps the original single-producer path.
  static constexpr int NumProducerThreadEvents = DispatchPolicy::UseAuxiliaryLoad ? 33 : 1;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(not cute::is_void_v<SmemCopyAtomA>,
    "SM120 mainloop must specify a copy atom for A operand smem->rmem reads.");
  static_assert(not cute::is_void_v<SmemCopyAtomB>,
    "SM120 mainloop must specify a copy atom for B operand smem->rmem reads.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t<
          DispatchPolicy::UseAColumnwiseSmemLayout,
          Step<_2,_1,_3>,
          conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t<
          DispatchPolicy::UseNoCopyBTma,
          Step<_1,_2,_3>,
          conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>>{}));

  // SmemLayoutAtomSFA and SmemLayoutAtomSFB are for whole CTA tiles. We add the number of pipeline stages here.
  // The number of pipeline stages is the same as the number of pipeline stages from AB Load <-> MainLoop
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));

  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));

  static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
  static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operands from rmem for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>, "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>, "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  static constexpr bool IsF8F6F4 = detail::is_sm120_f8f6f4<TiledMma, ElementA, ElementB>();

  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  using TmaInternalElementA = cute::conditional_t<not IsF8F6F4,
                                                  ElementA,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m1_t>,
                                                  cutlass::detail::float_e2m1_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m3_t>,
                                                cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e3m2_t>,
                                                cutlass::detail::float_e3m2_unpacksmem_t,
                                                uint_bit_t<sizeof_bits_v<ElementA>>>>>>;

  using TmaInternalElementB = cute::conditional_t<not IsF8F6F4,
                                                  ElementB,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m1_t>,
                                                  cutlass::detail::float_e2m1_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m3_t>,
                                                cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e3m2_t>,
                                                cutlass::detail::float_e3m2_unpacksmem_t,
                                                uint_bit_t<sizeof_bits_v<ElementB>>>>>>;

  using TmaInternalElementSF = ElementSF;

  using SmemAllocTypeA = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeA>;
  using SmemAllocTypeB = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeB>;

  static constexpr uint32_t ScaleTransactionBytesMK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>));

  static constexpr uint32_t ScaleTransactionBytesNK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>));

  static constexpr uint32_t PayloadTransactionBytesMK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));

  static constexpr uint32_t PayloadTransactionBytesNK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutB{})) * sizeof_bits<ElementB>::value));

  // The CUTLASS pipeline barrier must cover every byte written into the stage,
  // even when scales or payloads are produced by warp stores instead of TMA.
  static constexpr uint32_t TmaTransactionBytesMK =
      ScaleTransactionBytesMK + PayloadTransactionBytesMK;
  static constexpr uint32_t TmaTransactionBytesNK =
      ScaleTransactionBytesNK + PayloadTransactionBytesNK;
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
      alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;
    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) PipelineStorage pipeline_storage;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;
  using PipelineStorageMK = typename MainloopPipelineMK::SharedStorage;
  using PipelineStorageNK = typename MainloopPipelineNK::SharedStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementSF const* ptr_SFA{nullptr};
    int64_t sfa_ld{0};
    ElementSF const* ptr_SFB{nullptr};
    int64_t sfb_ld{0};
    bool manual_payload_load{false};
    int32_t a_source{static_cast<int32_t>(CppMegaCompactOperandSource::kRowwise)};
    int64_t a_data_ld{0};
    int32_t b_source{static_cast<int32_t>(CppMegaCompactOperandSource::kRowwise)};
    int64_t b_data_ld{0};
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast

    alignas(64) TMA_A tma_load_a;
    alignas(64) TMA_B tma_load_b;
    ElementA const* ptr_A{nullptr};
    ElementB const* ptr_B{nullptr};
    ElementSF const* ptr_SFA{nullptr};
    int64_t sfa_ld{0};
    ElementSF const* ptr_SFB{nullptr};
    int64_t sfb_ld{0};
    bool manual_payload_load{false};
    int32_t a_source{static_cast<int32_t>(CppMegaCompactOperandSource::kRowwise)};
    int64_t a_data_ld{0};
    int32_t b_source{static_cast<int32_t>(CppMegaCompactOperandSource::kRowwise)};
    int64_t b_data_ld{0};
    int64_t problem_k{0};
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t payload_transaction_bytes_mk = PayloadTransactionBytesMK;
    uint32_t payload_transaction_bytes_nk = PayloadTransactionBytesNK;
    uint32_t scale_transaction_bytes_mk = ScaleTransactionBytesMK;
    uint32_t scale_transaction_bytes_nk = ScaleTransactionBytesNK;
    uint32_t scale_transaction_bytes = ScaleTransactionBytesMK + ScaleTransactionBytesNK;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));

    typename Params::TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}); // No programmatic multicast
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        _1{}); // No programmatic multicast

    return {
      tma_load_a,
      tma_load_b,
      args.ptr_A,
      args.ptr_B,
      args.ptr_SFA,
      args.sfa_ld,
      args.ptr_SFB,
      args.sfb_ld,
      args.manual_payload_load,
      args.a_source,
      args.a_data_ld,
      args.b_source,
      args.b_data_ld,
      static_cast<int64_t>(K),
      TmaTransactionBytes,
      TmaTransactionBytesMK,
      TmaTransactionBytesNK,
      PayloadTransactionBytesMK,
      PayloadTransactionBytesNK,
      ScaleTransactionBytesMK,
      ScaleTransactionBytesNK,
      ScaleTransactionBytesMK + ScaleTransactionBytesNK
    };
  }

  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
  }

  // Temporary adhoc partitioning for scaling factors.
  template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
  {
    CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(permutation_mnk),
                            get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfatensor, t_tile);                 // (PermM,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
  }

  template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
  {
    CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<1>(permutation_mnk),
                            get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfbtensor, t_tile);                 // (PermN,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomN,AtomK),(RestN,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{},_);           // ((ThrV,FrgV),(RestN,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
    return thr_tensor;
  }

  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma)
  {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFA =  thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFA);
  }

  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma)
  {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFB =  thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFB);
  }

  template<class TiledMma>
  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutSFA_TV(TiledMma& mma)
  {
    // (M,K) -> (M,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto atile = make_tile(_,
                          make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                make_stride(               Int<1>{} ,                Int<0>{} )),
                                    _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
  }

  template<class TiledMma>
  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutSFB_TV(TiledMma& mma)
  {
    // (N,K) -> (N,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto btile = make_tile(_,
                          make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                make_stride(               Int<0>{} ,                Int<1>{} )),
                                    _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  /// The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                          // (m,k,l)
    Tensor mB_nkl = params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                          // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class LoadInputs,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      LoadInputs const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});  // (BLK_N,BLK_K,PIPE)

    //
    // Prepare the TMA loads for A and B. Scale factors are loaded from compact
    // rowwise tensors directly into the native shared-memory scale layout.
    //

    auto [gA_mkl, gB_nkl] = load_inputs;

    auto block_tma_a = params.tma_load_a.get_slice(0);
    auto block_tma_b = params.tma_load_b.get_slice(0);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

    Tensor gA =   gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
    Tensor gB =   gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

    // Partition source and destination tensors for tma copies
    Tensor tAgA = block_tma_a.partition_S(gA);                                              // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);                                              // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

    int m_coord_i = int(m_coord);
    int n_coord_i = int(n_coord);

    auto load_compact_scale_factors = [&](int write_stage, int k_tile, bool load_a, bool load_b) {
      constexpr int TileM = int(size<0>(TileShape{}));
      constexpr int TileN = int(size<1>(TileShape{}));
      constexpr int TileK = int(size<2>(TileShape{}));
      constexpr int Vec = SFVecSize;
      constexpr int KBlocks = TileK / Vec;
      constexpr int ScaleVecBytes = 4;
      static_assert(KBlocks % ScaleVecBytes == 0, "KBlocks must be divisible by scale vector width");
      int m_base = m_coord_i * TileM;
      int n_base = n_coord_i * TileN;
      int k_base_block = k_tile * KBlocks;
      int lane = thread_idx & 31;
      auto ptr_SFA_u8 = reinterpret_cast<uint8_t const*>(params.ptr_SFA);
      auto ptr_SFB_u8 = reinterpret_cast<uint8_t const*>(params.ptr_SFB);

      auto store_sfa_m_contiguous = [&](int row_base, int kb, uint4 chunk) {
        uint32_t word = chunk.x;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.y;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + 4 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.z;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + 8 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.w;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + 12 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
      };

      auto store_sfb_n_contiguous = [&](int row_base, int kb, uint4 chunk) {
        uint32_t word = chunk.x;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFB(row_base + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.y;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFB(row_base + 4 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.z;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFB(row_base + 8 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.w;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFB(row_base + 12 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
      };

      auto store_sfa_k_contiguous = [&](int row, int kb_base, uint32_t word) {
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < ScaleVecBytes; ++byte) {
          sSFA(row, (kb_base + byte) * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
      };

      auto store_sfb_k_contiguous = [&](int row, int kb_base, uint32_t word) {
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < ScaleVecBytes; ++byte) {
          sSFB(row, (kb_base + byte) * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
      };

      if (load_a) {
        if (params.a_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose)) {
          // K-block major order makes each warp transaction read contiguous TE columnwise scale bytes.
          constexpr int RowVectors = TileM / 16;
          for (int idx = lane; idx < KBlocks * RowVectors; idx += 32) {
            int kb = idx / RowVectors;
            int row_vec = idx - kb * RowVectors;
            int row_base = row_vec * 16;
            int64_t scale_offset =
                static_cast<int64_t>(k_base_block + kb) * params.sfa_ld + (m_base + row_base);
            uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_SFA_u8 + scale_offset);
            store_sfa_m_contiguous(row_base, kb, chunk);
          }
        } else {
          constexpr int KBlockVectors = KBlocks / ScaleVecBytes;
          for (int idx = lane; idx < TileM * KBlockVectors; idx += 32) {
            int row = idx / KBlockVectors;
            int kb_vec = idx - row * KBlockVectors;
            int kb_base = kb_vec * ScaleVecBytes;
            int64_t scale_offset =
                static_cast<int64_t>(m_base + row) * params.sfa_ld + k_base_block + kb_base;
            uint32_t word = *reinterpret_cast<uint32_t const*>(ptr_SFA_u8 + scale_offset);
            store_sfa_k_contiguous(row, kb_base, word);
          }
        }
      }

      if (load_b) {
        if (params.b_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose)) {
          // K-block major order makes each warp transaction read contiguous TE columnwise scale bytes.
          constexpr int RowVectors = TileN / 16;
          for (int idx = lane; idx < KBlocks * RowVectors; idx += 32) {
            int kb = idx / RowVectors;
            int row_vec = idx - kb * RowVectors;
            int row_base = row_vec * 16;
            int64_t scale_offset =
                static_cast<int64_t>(k_base_block + kb) * params.sfb_ld + (n_base + row_base);
            uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_SFB_u8 + scale_offset);
            store_sfb_n_contiguous(row_base, kb, chunk);
          }
        } else {
          constexpr int KBlockVectors = KBlocks / ScaleVecBytes;
          for (int idx = lane; idx < TileN * KBlockVectors; idx += 32) {
            int row = idx / KBlockVectors;
            int kb_vec = idx - row * KBlockVectors;
            int kb_base = kb_vec * ScaleVecBytes;
            int64_t scale_offset =
                static_cast<int64_t>(n_base + row) * params.sfb_ld + k_base_block + kb_base;
            uint32_t word = *reinterpret_cast<uint32_t const*>(ptr_SFB_u8 + scale_offset);
            store_sfb_k_contiguous(row, kb_base, word);
          }
        }
      }
    };

    auto load_payload_tile_a = [&](int write_stage, int k_tile) {
      static_assert(IsF8F6F4, "cppmega compact direct payload loader only supports byte-packed SM120 f8/f6/f4 operands");
      constexpr int TileM = int(size<0>(TileShape{}));
      constexpr int TileK = int(size<2>(TileShape{}));
      constexpr int VecBytes = 16;
      static_assert(TileM % VecBytes == 0, "TileM must be divisible by the vector width");
      static_assert(TileK % VecBytes == 0, "TileK must be divisible by the vector width");
      int m_base = m_coord_i * TileM;
      int k_base = k_tile * TileK;
      int lane = thread_idx & 31;
      auto ptr_A = reinterpret_cast<uint8_t const*>(params.ptr_A);

      auto store_a_m_word = [&](int row_base, int kk, int byte_base, uint32_t word) {
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sA(row_base + byte_base + byte, kk, write_stage) =
              static_cast<SmemAllocTypeA>((word >> (8 * byte)) & 0xffu);
        }
      };

      auto store_a_m_contiguous = [&](int row_base, int kk, uint4 chunk) {
        store_a_m_word(row_base, kk, 0, chunk.x);
        store_a_m_word(row_base, kk, 4, chunk.y);
        store_a_m_word(row_base, kk, 8, chunk.z);
        store_a_m_word(row_base, kk, 12, chunk.w);
      };

      auto store_a_k_word = [&](int row, int kk_base, int byte_base, uint32_t word) {
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sA(row, kk_base + byte_base + byte, write_stage) =
              static_cast<SmemAllocTypeA>((word >> (8 * byte)) & 0xffu);
        }
      };

      auto store_a_k_contiguous = [&](int row, int kk_base, uint4 chunk) {
        store_a_k_word(row, kk_base, 0, chunk.x);
        store_a_k_word(row, kk_base, 4, chunk.y);
        store_a_k_word(row, kk_base, 8, chunk.z);
        store_a_k_word(row, kk_base, 12, chunk.w);
      };

      if (params.a_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose)) {
        constexpr int RowVectors = TileM / VecBytes;
        for (int idx = lane; idx < TileK * RowVectors; idx += 32) {
          int kk = idx / RowVectors;
          int row_vec = idx - kk * RowVectors;
          int row_base = row_vec * VecBytes;
          int64_t payload_offset =
              static_cast<int64_t>(k_base + kk) * params.a_data_ld + (m_base + row_base);
          uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_A + payload_offset);
          store_a_m_contiguous(row_base, kk, chunk);
        }
      } else {
        constexpr int KVectors = TileK / VecBytes;
        for (int idx = lane; idx < TileM * KVectors; idx += 32) {
          int row = idx / KVectors;
          int k_vec = idx - row * KVectors;
          int kk_base = k_vec * VecBytes;
          int64_t payload_offset =
              static_cast<int64_t>(m_base + row) * params.a_data_ld + k_base + kk_base;
          uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_A + payload_offset);
          store_a_k_contiguous(row, kk_base, chunk);
        }
      }
    };

    auto load_payload_tile_b = [&](int write_stage, int k_tile) {
      static_assert(IsF8F6F4, "cppmega compact direct payload loader only supports byte-packed SM120 f8/f6/f4 operands");
      constexpr int TileN = int(size<1>(TileShape{}));
      constexpr int TileK = int(size<2>(TileShape{}));
      constexpr int VecBytes = 16;
      static_assert(TileN % VecBytes == 0, "TileN must be divisible by the vector width");
      static_assert(TileK % VecBytes == 0, "TileK must be divisible by the vector width");
      int n_base = n_coord_i * TileN;
      int k_base = k_tile * TileK;
      int lane = thread_idx & 31;
      auto ptr_B = reinterpret_cast<uint8_t const*>(params.ptr_B);

      auto store_b_n_word = [&](int row_base, int kk, int byte_base, uint32_t word) {
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sB(row_base + byte_base + byte, kk, write_stage) =
              static_cast<SmemAllocTypeB>((word >> (8 * byte)) & 0xffu);
        }
      };

      auto store_b_n_contiguous = [&](int row_base, int kk, uint4 chunk) {
        store_b_n_word(row_base, kk, 0, chunk.x);
        store_b_n_word(row_base, kk, 4, chunk.y);
        store_b_n_word(row_base, kk, 8, chunk.z);
        store_b_n_word(row_base, kk, 12, chunk.w);
      };

      auto store_b_k_word = [&](int row, int kk_base, int byte_base, uint32_t word) {
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sB(row, kk_base + byte_base + byte, write_stage) =
              static_cast<SmemAllocTypeB>((word >> (8 * byte)) & 0xffu);
        }
      };

      auto store_b_k_contiguous = [&](int row, int kk_base, uint4 chunk) {
        store_b_k_word(row, kk_base, 0, chunk.x);
        store_b_k_word(row, kk_base, 4, chunk.y);
        store_b_k_word(row, kk_base, 8, chunk.z);
        store_b_k_word(row, kk_base, 12, chunk.w);
      };

      if (params.b_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose)) {
        constexpr int RowVectors = TileN / VecBytes;
        for (int idx = lane; idx < TileK * RowVectors; idx += 32) {
          int kk = idx / RowVectors;
          int row_vec = idx - kk * RowVectors;
          int row_base = row_vec * VecBytes;
          int64_t payload_offset =
              static_cast<int64_t>(k_base + kk) * params.b_data_ld + (n_base + row_base);
          uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_B + payload_offset);
          store_b_n_contiguous(row_base, kk, chunk);
        }
      } else {
        constexpr int KVectors = TileK / VecBytes;
        for (int idx = lane; idx < TileN * KVectors; idx += 32) {
          int row = idx / KVectors;
          int k_vec = idx - row * KVectors;
          int kk_base = k_vec * VecBytes;
          int64_t payload_offset =
              static_cast<int64_t>(n_base + row) * params.b_data_ld + k_base + kk_base;
          uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_B + payload_offset);
          store_b_k_contiguous(row, kk_base, chunk);
        }
      }
    };

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      if (lane_predicate) {
        pipeline.producer_acquire(smem_pipe_write);
      }
      __syncwarp();

      int write_stage = smem_pipe_write.index();
      load_compact_scale_factors(
          write_stage,
          *k_tile_iter,
          true,
          !DispatchPolicy::UseAuxiliaryLoad);
      __syncwarp();

      //
      // Copy gmem to smem for *k_tile_iter
      //
      bool manual_payload_a =
          params.manual_payload_load &&
          params.a_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose);
      bool manual_payload_b =
          !DispatchPolicy::UseNoCopyBTma &&
          !DispatchPolicy::UseAuxiliaryLoad &&
          params.manual_payload_load &&
          params.b_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose);

      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
      uint32_t manual_transaction_bytes = DispatchPolicy::UseAuxiliaryLoad
          ? params.scale_transaction_bytes_mk
          : params.scale_transaction_bytes;
      if (manual_payload_a) {
        load_payload_tile_a(write_stage, *k_tile_iter);
        manual_transaction_bytes += params.payload_transaction_bytes_mk;
      } else if (lane_predicate) {
        copy(params.tma_load_a.with(*tma_barrier), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
      }
      if (manual_payload_b) {
        load_payload_tile_b(write_stage, *k_tile_iter);
        manual_transaction_bytes += params.payload_transaction_bytes_nk;
      } else if constexpr (!DispatchPolicy::UseAuxiliaryLoad) {
        if (lane_predicate) {
          copy(params.tma_load_b.with(*tma_barrier), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        }
      }
      __syncwarp();
      cutlass::arch::fence_view_async_shared();
      __syncwarp();
      if (lane_predicate) {
        cutlass::arch::ClusterTransactionBarrier::complete_transaction(
            tma_barrier, block_rank_in_cluster, manual_transaction_bytes, 1);
      }

      // Advance k tile
      ++k_tile_iter;
      ++smem_pipe_write;
    }
    __syncwarp();
  }

  // A/SFA half of the asymmetric producer path.  This mirrors CUTLASS'
  // LoadMK role but keeps cppmega's compact TE scale/payload contract.
  template <
    class LoadInputs,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load_MK(
      Params const& params,
      MainloopPipelineMK pipeline,
      PipelineStateMK smem_pipe_write,
      LoadInputs const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});

    auto [gA_mkl, gB_nkl] = load_inputs;
    (void)gB_nkl;

    auto block_tma_a = params.tma_load_a.get_slice(0);

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    (void)n_coord;
    (void)k_coord;
    Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);

    Tensor tAgA = block_tma_a.partition_S(gA);
    Tensor tAsA = block_tma_a.partition_D(sA);

    int m_coord_i = int(m_coord);

    auto load_sfa = [&](int write_stage, int k_tile, bool split_a_producer) {
      constexpr int TileM = int(size<0>(TileShape{}));
      constexpr int TileK = int(size<2>(TileShape{}));
      constexpr int Vec = SFVecSize;
      constexpr int KBlocks = TileK / Vec;
      constexpr int ScaleVecBytes = 4;
      static_assert(KBlocks % ScaleVecBytes == 0, "KBlocks must be divisible by scale vector width");
      int m_base = m_coord_i * TileM;
      int k_base_block = k_tile * KBlocks;
      int lane = thread_idx & 31;
      auto ptr_SFA_u8 = reinterpret_cast<uint8_t const*>(params.ptr_SFA);

      auto store_m_contiguous = [&](int row_base, int kb, uint4 chunk) {
        uint32_t word = chunk.x;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.y;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + 4 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.z;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + 8 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.w;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + 12 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
      };

      auto store_k_contiguous = [&](int row, int kb_base, uint32_t word) {
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < ScaleVecBytes; ++byte) {
          sSFA(row, (kb_base + byte) * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
      };

      if (params.a_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose)) {
        constexpr int RowVectors = TileM / 16;
        if constexpr (DispatchPolicy::UseAColumnwiseSmemLayout) {
          static_assert(RowVectors % 2 == 0, "split A producer expects an even row-vector count");
          if (split_a_producer) {
            constexpr int SplitRowVectors = RowVectors / 2;
            for (int idx = lane; idx < SplitRowVectors * KBlocks; idx += 32) {
              int row_vec = (idx / KBlocks) * 2;
              int kb = idx - (idx / KBlocks) * KBlocks;
              int row_base = row_vec * 16;
              int64_t scale_offset =
                  static_cast<int64_t>(k_base_block + kb) * params.sfa_ld + (m_base + row_base);
              uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_SFA_u8 + scale_offset);
              store_m_contiguous(row_base, kb, chunk);
            }
          } else {
            for (int idx = lane; idx < KBlocks * RowVectors; idx += 32) {
              int kb = idx / RowVectors;
              int row_vec = idx - kb * RowVectors;
              int row_base = row_vec * 16;
              int64_t scale_offset =
                  static_cast<int64_t>(k_base_block + kb) * params.sfa_ld + (m_base + row_base);
              uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_SFA_u8 + scale_offset);
              store_m_contiguous(row_base, kb, chunk);
            }
          }
        } else {
          for (int idx = lane; idx < KBlocks * RowVectors; idx += 32) {
            int kb = idx / RowVectors;
            int row_vec = idx - kb * RowVectors;
            int row_base = row_vec * 16;
            int64_t scale_offset =
                static_cast<int64_t>(k_base_block + kb) * params.sfa_ld + (m_base + row_base);
            uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_SFA_u8 + scale_offset);
            store_m_contiguous(row_base, kb, chunk);
          }
        }
      } else {
        constexpr int KBlockVectors = KBlocks / ScaleVecBytes;
        for (int idx = lane; idx < TileM * KBlockVectors; idx += 32) {
          int row = idx / KBlockVectors;
          int kb_vec = idx - row * KBlockVectors;
          int kb_base = kb_vec * ScaleVecBytes;
          int64_t scale_offset =
              static_cast<int64_t>(m_base + row) * params.sfa_ld + k_base_block + kb_base;
          uint32_t word = *reinterpret_cast<uint32_t const*>(ptr_SFA_u8 + scale_offset);
          store_k_contiguous(row, kb_base, word);
        }
      }
    };

    auto load_payload_a = [&](int write_stage, int k_tile, bool split_a_producer) {
      static_assert(IsF8F6F4, "cppmega compact direct payload loader only supports byte-packed SM120 f8/f6/f4 operands");
      constexpr int TileM = int(size<0>(TileShape{}));
      constexpr int TileK = int(size<2>(TileShape{}));
      constexpr int VecBytes = 16;
      int m_base = m_coord_i * TileM;
      int k_base = k_tile * TileK;
      int lane = thread_idx & 31;
      auto ptr_A = reinterpret_cast<uint8_t const*>(params.ptr_A);

      auto store_m_word = [&](int row_base, int kk, int byte_base, uint32_t word) {
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sA(row_base + byte_base + byte, kk, write_stage) =
              static_cast<SmemAllocTypeA>((word >> (8 * byte)) & 0xffu);
        }
      };
      auto store_m_contiguous = [&](int row_base, int kk, uint4 chunk) {
        store_m_word(row_base, kk, 0, chunk.x);
        store_m_word(row_base, kk, 4, chunk.y);
        store_m_word(row_base, kk, 8, chunk.z);
        store_m_word(row_base, kk, 12, chunk.w);
      };

      constexpr int RowVectors = TileM / VecBytes;
      if constexpr (DispatchPolicy::UseAColumnwiseSmemLayout) {
        static_assert(RowVectors % 2 == 0, "split A producer expects an even row-vector count");
        if (split_a_producer) {
          constexpr int SplitRowVectors = RowVectors / 2;
          for (int idx = lane; idx < SplitRowVectors * TileK; idx += 32) {
            int row_vec = (idx / TileK) * 2;
            int kk = idx - (idx / TileK) * TileK;
            int row_base = row_vec * VecBytes;
            int64_t payload_offset =
                static_cast<int64_t>(k_base + kk) * params.a_data_ld + (m_base + row_base);
            uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_A + payload_offset);
            store_m_contiguous(row_base, kk, chunk);
          }
        } else {
          for (int idx = lane; idx < RowVectors * TileK; idx += 32) {
            int row_vec = idx / TileK;
            int kk = idx - row_vec * TileK;
            int row_base = row_vec * VecBytes;
            int64_t payload_offset =
                static_cast<int64_t>(k_base + kk) * params.a_data_ld + (m_base + row_base);
            uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_A + payload_offset);
            store_m_contiguous(row_base, kk, chunk);
          }
        }
      } else {
        for (int idx = lane; idx < TileK * RowVectors; idx += 32) {
          int kk = idx / RowVectors;
          int row_vec = idx - kk * RowVectors;
          int row_base = row_vec * VecBytes;
          int64_t payload_offset =
              static_cast<int64_t>(k_base + kk) * params.a_data_ld + (m_base + row_base);
          uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_A + payload_offset);
          store_m_contiguous(row_base, kk, chunk);
        }
      }
    };

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      pipeline.producer_acquire(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      bool split_a_producer =
          DispatchPolicy::UseAColumnwiseSmemLayout &&
          params.a_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose) &&
          params.b_source == static_cast<int32_t>(CppMegaCompactOperandSource::kRowwise);
      load_sfa(write_stage, *k_tile_iter, split_a_producer);
      if (!split_a_producer) {
        __syncwarp();
      }

      bool manual_payload =
          params.manual_payload_load &&
          params.a_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose);
      using BarrierType = typename MainloopPipelineMK::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
      uint32_t manual_transaction_bytes = params.scale_transaction_bytes_mk;
      if (manual_payload) {
        load_payload_a(write_stage, *k_tile_iter, split_a_producer);
        manual_transaction_bytes += params.payload_transaction_bytes_mk;
      } else if (lane_predicate) {
        copy(params.tma_load_a.with(*tma_barrier), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
      }
      __syncwarp();
      cutlass::arch::fence_view_async_shared();
      __syncwarp();
      if (lane_predicate) {
        cutlass::arch::ClusterTransactionBarrier::complete_transaction(
            tma_barrier, block_rank_in_cluster, manual_transaction_bytes, 1);
      }

      ++k_tile_iter;
      ++smem_pipe_write;
    }
  }

  // B/SFB half of the asymmetric producer path.
  template <
    class LoadInputs,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load_NK(
      Params const& params,
      MainloopPipelineNK pipeline,
      PipelineStateNK smem_pipe_write,
      LoadInputs const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors,
      bool commit_aux_arrivals = false) {
    int lane_predicate = cute::elect_one_sync();

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

    auto [gA_mkl, gB_nkl] = load_inputs;
    (void)gA_mkl;

    auto block_tma_b = params.tma_load_b.get_slice(0);

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    (void)k_coord;
    Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);

    Tensor tBgB = block_tma_b.partition_S(gB);
    Tensor tBsB = block_tma_b.partition_D(sB);

    int m_coord_i = int(m_coord);
    int n_coord_i = int(n_coord);

    auto load_sfa_a_odd_rows = [&](int write_stage, int k_tile) {
      constexpr int TileM = int(size<0>(TileShape{}));
      constexpr int TileK = int(size<2>(TileShape{}));
      constexpr int Vec = SFVecSize;
      constexpr int KBlocks = TileK / Vec;
      constexpr int RowVectors = TileM / 16;
      static_assert(RowVectors % 2 == 0, "split A producer expects an even row-vector count");
      constexpr int SplitRowVectors = RowVectors / 2;
      int m_base = m_coord_i * TileM;
      int k_base_block = k_tile * KBlocks;
      int lane = thread_idx & 31;
      auto ptr_SFA_u8 = reinterpret_cast<uint8_t const*>(params.ptr_SFA);

      auto store_m_contiguous = [&](int row_base, int kb, uint4 chunk) {
        uint32_t word = chunk.x;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.y;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + 4 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.z;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + 8 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.w;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFA(row_base + 12 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
      };

      for (int idx = lane; idx < SplitRowVectors * KBlocks; idx += 32) {
        int split_row_vec = idx / KBlocks;
        int row_vec = split_row_vec * 2 + 1;
        int kb = idx - split_row_vec * KBlocks;
        int row_base = row_vec * 16;
        int64_t scale_offset =
            static_cast<int64_t>(k_base_block + kb) * params.sfa_ld + (m_base + row_base);
        uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_SFA_u8 + scale_offset);
        store_m_contiguous(row_base, kb, chunk);
      }
    };

    auto load_payload_a_odd_rows = [&](int write_stage, int k_tile) {
      static_assert(IsF8F6F4, "cppmega compact direct payload loader only supports byte-packed SM120 f8/f6/f4 operands");
      constexpr int TileM = int(size<0>(TileShape{}));
      constexpr int TileK = int(size<2>(TileShape{}));
      constexpr int VecBytes = 16;
      constexpr int RowVectors = TileM / VecBytes;
      static_assert(RowVectors % 2 == 0, "split A producer expects an even row-vector count");
      constexpr int SplitRowVectors = RowVectors / 2;
      int m_base = m_coord_i * TileM;
      int k_base = k_tile * TileK;
      int lane = thread_idx & 31;
      auto ptr_A = reinterpret_cast<uint8_t const*>(params.ptr_A);

      auto store_m_word = [&](int row_base, int kk, int byte_base, uint32_t word) {
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sA(row_base + byte_base + byte, kk, write_stage) =
              static_cast<SmemAllocTypeA>((word >> (8 * byte)) & 0xffu);
        }
      };
      auto store_m_contiguous = [&](int row_base, int kk, uint4 chunk) {
        store_m_word(row_base, kk, 0, chunk.x);
        store_m_word(row_base, kk, 4, chunk.y);
        store_m_word(row_base, kk, 8, chunk.z);
        store_m_word(row_base, kk, 12, chunk.w);
      };

      for (int idx = lane; idx < SplitRowVectors * TileK; idx += 32) {
        int split_row_vec = idx / TileK;
        int row_vec = split_row_vec * 2 + 1;
        int kk = idx - split_row_vec * TileK;
        int row_base = row_vec * VecBytes;
        int64_t payload_offset =
            static_cast<int64_t>(k_base + kk) * params.a_data_ld + (m_base + row_base);
        uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_A + payload_offset);
        store_m_contiguous(row_base, kk, chunk);
      }
    };

    auto load_sfb = [&](int write_stage, int k_tile) {
      constexpr int TileN = int(size<1>(TileShape{}));
      constexpr int TileK = int(size<2>(TileShape{}));
      constexpr int Vec = SFVecSize;
      constexpr int KBlocks = TileK / Vec;
      constexpr int ScaleVecBytes = 4;
      static_assert(KBlocks % ScaleVecBytes == 0, "KBlocks must be divisible by scale vector width");
      int n_base = n_coord_i * TileN;
      int k_base_block = k_tile * KBlocks;
      int lane = thread_idx & 31;
      auto ptr_SFB_u8 = reinterpret_cast<uint8_t const*>(params.ptr_SFB);

      auto store_n_contiguous = [&](int row_base, int kb, uint4 chunk) {
        uint32_t word = chunk.x;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFB(row_base + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.y;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFB(row_base + 4 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.z;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFB(row_base + 8 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
        word = chunk.w;
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sSFB(row_base + 12 + byte, kb * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
      };

      auto store_k_contiguous = [&](int row, int kb_base, uint32_t word) {
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < ScaleVecBytes; ++byte) {
          sSFB(row, (kb_base + byte) * Vec, write_stage) =
              ElementSF::bitcast(static_cast<uint8_t>((word >> (8 * byte)) & 0xffu));
        }
      };

      if (params.b_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose)) {
        constexpr int RowVectors = TileN / 16;
        for (int idx = lane; idx < KBlocks * RowVectors; idx += 32) {
          int kb = idx / RowVectors;
          int row_vec = idx - kb * RowVectors;
          int row_base = row_vec * 16;
          int64_t scale_offset =
              static_cast<int64_t>(k_base_block + kb) * params.sfb_ld + (n_base + row_base);
          uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_SFB_u8 + scale_offset);
          store_n_contiguous(row_base, kb, chunk);
        }
      } else {
        constexpr int KBlockVectors = KBlocks / ScaleVecBytes;
        for (int idx = lane; idx < TileN * KBlockVectors; idx += 32) {
          int row = idx / KBlockVectors;
          int kb_vec = idx - row * KBlockVectors;
          int kb_base = kb_vec * ScaleVecBytes;
          int64_t scale_offset =
              static_cast<int64_t>(n_base + row) * params.sfb_ld + k_base_block + kb_base;
          uint32_t word = *reinterpret_cast<uint32_t const*>(ptr_SFB_u8 + scale_offset);
          store_k_contiguous(row, kb_base, word);
        }
      }
    };

    auto load_payload_b = [&](int write_stage, int k_tile) {
      static_assert(IsF8F6F4, "cppmega compact direct payload loader only supports byte-packed SM120 f8/f6/f4 operands");
      constexpr int TileN = int(size<1>(TileShape{}));
      constexpr int TileK = int(size<2>(TileShape{}));
      constexpr int VecBytes = 16;
      int n_base = n_coord_i * TileN;
      int k_base = k_tile * TileK;
      int lane = thread_idx & 31;
      auto ptr_B = reinterpret_cast<uint8_t const*>(params.ptr_B);

      auto store_n_word = [&](int row_base, int kk, int byte_base, uint32_t word) {
        CUTLASS_PRAGMA_UNROLL
        for (int byte = 0; byte < 4; ++byte) {
          sB(row_base + byte_base + byte, kk, write_stage) =
              static_cast<SmemAllocTypeB>((word >> (8 * byte)) & 0xffu);
        }
      };
      auto store_n_contiguous = [&](int row_base, int kk, uint4 chunk) {
        store_n_word(row_base, kk, 0, chunk.x);
        store_n_word(row_base, kk, 4, chunk.y);
        store_n_word(row_base, kk, 8, chunk.z);
        store_n_word(row_base, kk, 12, chunk.w);
      };

      constexpr int RowVectors = TileN / VecBytes;
      for (int idx = lane; idx < TileK * RowVectors; idx += 32) {
        int kk = idx / RowVectors;
        int row_vec = idx - kk * RowVectors;
        int row_base = row_vec * VecBytes;
        int64_t payload_offset =
            static_cast<int64_t>(k_base + kk) * params.b_data_ld + (n_base + row_base);
        uint4 chunk = *reinterpret_cast<uint4 const*>(ptr_B + payload_offset);
        store_n_contiguous(row_base, kk, chunk);
      }
    };

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      pipeline.producer_acquire(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      bool split_a_producer =
          DispatchPolicy::UseAColumnwiseSmemLayout &&
          params.a_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose) &&
          params.b_source == static_cast<int32_t>(CppMegaCompactOperandSource::kRowwise);
      bool manual_payload =
          !DispatchPolicy::UseNoCopyBTma &&
          params.manual_payload_load &&
          params.b_source == static_cast<int32_t>(CppMegaCompactOperandSource::kColumnwiseTranspose);
      using BarrierType = typename MainloopPipelineNK::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
      bool issue_b_tma_early =
          DispatchPolicy::UseBTmaEarly &&
          split_a_producer &&
          !manual_payload;
      if (split_a_producer) {
        load_sfa_a_odd_rows(write_stage, *k_tile_iter);
      }
      if (issue_b_tma_early && lane_predicate) {
        copy(params.tma_load_b.with(*tma_barrier), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
      }
      load_sfb(write_stage, *k_tile_iter);
      if (!split_a_producer) {
        __syncwarp();
      }

      uint32_t manual_transaction_bytes = params.scale_transaction_bytes_nk;
      if (split_a_producer) {
        load_payload_a_odd_rows(write_stage, *k_tile_iter);
      }
      if (manual_payload) {
        load_payload_b(write_stage, *k_tile_iter);
        manual_transaction_bytes += params.payload_transaction_bytes_nk;
      } else if (!issue_b_tma_early && lane_predicate) {
        copy(params.tma_load_b.with(*tma_barrier), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
      }
      __syncwarp();
      cutlass::arch::fence_view_async_shared();
      __syncwarp();
      if (lane_predicate) {
        cutlass::arch::ClusterTransactionBarrier::complete_transaction(
            tma_barrier, block_rank_in_cluster, manual_transaction_bytes, 1);
      }
      if (commit_aux_arrivals) {
        pipeline.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive_noinc);
      }

      ++k_tile_iter;
      ++smem_pipe_write;
    }
  }

  template <
    class LoadInputs,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load_auxiliary(
      Params const& params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      LoadInputs const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    load_NK(
      params,
      pipeline,
      smem_pipe_write,
      load_inputs,
      blk_coord,
      k_tile_iter,
      k_tile_count,
      thread_idx,
      block_rank_in_cluster,
      shared_tensors,
      true);
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      [[maybe_unused]] Params const& params) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});         // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});  // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    // Allocate fragments and descriptors
    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));                         // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));                         // (MMA,MMA_N,MMA_K)

    Tensor tCrSFA = partition_fragment_SFA(sSFA(_,_,Int<0>{}), thread_mma);                  // (MMA,MMA_M,MMA_K)
    Tensor tCrSFB = partition_fragment_SFB(sSFB(_,_,Int<0>{}), thread_mma);                  // (MMA,MMA_N,MMA_K)

    //
    // Copy from smem to registers
    //

    // A
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(
      as_position_independent_swizzle_tensor(sA));                                      // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                            //      (CPY,CPY_M,CPY_K)

    // B
    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(
      as_position_independent_swizzle_tensor(sB));                                      // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                            //      (CPY,CPY_M,CPY_K)

    // SFA
    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{},
                                                    get_layoutSFA_TV(tiled_mma),
                                                    make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                  );
    auto smem_thr_copy_SFA   = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tCsSFA            = smem_thr_copy_SFA.partition_S(
        as_position_independent_swizzle_tensor(sSFA));                                      // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrSFA_copy_view  = smem_thr_copy_SFA.retile_D(tCrSFA);                          //      (CPY,CPY_M,CPY_K)

    // SFB
    auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                    get_layoutSFB_TV(tiled_mma),
                                                    make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                  );
    auto smem_thr_copy_SFB   = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB            = smem_thr_copy_SFB.partition_S(
      as_position_independent_swizzle_tensor(sSFB));                                       // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrSFB_copy_view  = smem_thr_copy_SFB.retile_D(tCrSFB);                         //      (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));                        // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));                        // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                  // CPY_K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                  // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                    // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                    // PIPE

    CUTE_STATIC_ASSERT_V(size<1>(tCsSFA) == size<1>(tCrSFA_copy_view));                    // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCrSFA_copy_view));                    // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFA) == size<1>(accum));                               // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFB) == size<2>(accum));                               // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCsSFB));                              // CPY_K
    CUTE_STATIC_ASSERT_V(size<3>(tCsSFA) == size<3>(tCsSFB));                              // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(sA) == size<2>(sSFA));                                    // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(sB) == size<2>(sSFA));                                    // PIPE

    //
    // PIPELINED MAIN LOOP
    //

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    int read_stage = smem_pipe_read.index();
    auto tCsA_stage   = tCsA(_,_,_,read_stage);
    auto tCsB_stage   = tCsB(_,_,_,read_stage);
    auto tCsSFA_stage = tCsSFA(_,_,_,read_stage);
    auto tCsSFB_stage = tCsSFB(_,_,_,read_stage);

    auto copy_kblock = [&](auto k_block) {
        // copy smem->rmem for A/B operand
      copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
      copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));

      // Left shift A,B for FP4
      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));


      // Copy smem->rmem for SFA/SFB operand
      copy(tCsSFA_stage(_,_,k_block), tCrSFA_copy_view(_,_,k_block));
      copy(tCsSFB_stage(_,_,k_block), tCrSFB_copy_view(_,_,k_block));
    };

    auto gemm_kblock = [&](auto k_block) {
      // (V,M) x (V,N) => (V,M,N)
      cute::gemm(tiled_mma, make_zip_tensor(tCrA(_,_,k_block), tCrSFA(_,_,k_block)), make_zip_tensor(tCrB(_,_,k_block), tCrSFB(_,_,k_block)), accum);
    };

    pipeline.consumer_wait(smem_pipe_read);

    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {
      //
      // Compute on k_tile
      //
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {

        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(
          thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          // UNLOCK smem_pipe_read, done _computing_ on it
          pipeline.consumer_release(smem_pipe_read);
          ++smem_pipe_read;
          read_stage = smem_pipe_read.index();
          tCsA_stage   = tCsA(_,_,_,read_stage);
          tCsB_stage   = tCsB(_,_,_,read_stage);
          tCsSFA_stage = tCsSFA(_,_,_,read_stage);
          tCsSFB_stage = tCsSFB(_,_,_,read_stage);
          pipeline.consumer_wait(smem_pipe_read);
        }

        copy_kblock(k_block_next);
        gemm_kblock(k_block);

      });
    } // k_tile_count

    //
    // Hoist out last k_tile
    //
    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {

      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

      if (k_block == K_BLOCK_MAX - 1) {
        cutlass::arch::NamedBarrier::sync(
        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        // UNLOCK smem_pipe_read, done _computing_ on it
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }

      if (k_block_next > 0) {
        copy_kblock(k_block_next);
      }
      gemm_kblock(k_block);

    });
}

  /// Consumer path for CUTLASS' asymmetric DMA kernel wrapper.
  template <
    class FrgTensorC,
    class KTileIterator,
    class CtaTileCoord,
    class ProblemShape_MNKL
  >
  CUTLASS_DEVICE void
  mma(MainloopPipelineMK pipeline_mk,
      PipelineStateMK smem_pipe_read_mk,
      MainloopPipelineNK pipeline_nk,
      PipelineStateNK smem_pipe_read_nk,
      FrgTensorC& accum,
      KTileIterator k_tile_iter,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      [[maybe_unused]] Params const& params,
      [[maybe_unused]] CtaTileCoord const& cta_tile_coord,
      [[maybe_unused]] ProblemShape_MNKL const& problem_shape_MNKL) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));
    Tensor tCrSFA = partition_fragment_SFA(sSFA(_,_,Int<0>{}), thread_mma);
    Tensor tCrSFB = partition_fragment_SFB(sSFB(_,_,Int<0>{}), thread_mma);

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA = smem_thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sA));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_SFA = make_tiled_copy_impl(
        SmemCopyAtomSFA{},
        get_layoutSFA_TV(tiled_mma),
        make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tCsSFA = smem_thr_copy_SFA.partition_S(as_position_independent_swizzle_tensor(sSFA));
    Tensor tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(tCrSFA);

    auto smem_tiled_copy_SFB = make_tiled_copy_impl(
        SmemCopyAtomSFB{},
        get_layoutSFB_TV(tiled_mma),
        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB = smem_thr_copy_SFB.partition_S(as_position_independent_swizzle_tensor(sSFB));
    Tensor tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(tCrSFB);

    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == Int<DispatchPolicy::Stages>{});
    CUTE_STATIC_ASSERT_V(size<3>(tCsB) == Int<DispatchPolicy::Stages>{});

    auto K_BLOCK_MAX = size<2>(tCrA);

    int read_stage_mk = smem_pipe_read_mk.index();
    int read_stage_nk = smem_pipe_read_nk.index();
    auto tCsA_stage = tCsA(_,_,_,read_stage_mk);
    auto tCsB_stage = tCsB(_,_,_,read_stage_nk);
    auto tCsSFA_stage = tCsSFA(_,_,_,read_stage_mk);
    auto tCsSFB_stage = tCsSFB(_,_,_,read_stage_nk);

    auto wait_both = [&]() {
      pipeline_mk.consumer_wait(smem_pipe_read_mk);
      pipeline_nk.consumer_wait(smem_pipe_read_nk);
    };

    auto release_advance_both = [&]() {
      pipeline_mk.consumer_release(smem_pipe_read_mk);
      pipeline_nk.consumer_release(smem_pipe_read_nk);
      ++smem_pipe_read_mk;
      ++smem_pipe_read_nk;
      read_stage_mk = smem_pipe_read_mk.index();
      read_stage_nk = smem_pipe_read_nk.index();
      tCsA_stage = tCsA(_,_,_,read_stage_mk);
      tCsB_stage = tCsB(_,_,_,read_stage_nk);
      tCsSFA_stage = tCsSFA(_,_,_,read_stage_mk);
      tCsSFB_stage = tCsSFB(_,_,_,read_stage_nk);
    };

    auto copy_kblock = [&](auto k_block) {
      copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
      copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));

      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));

      copy(tCsSFA_stage(_,_,k_block), tCrSFA_copy_view(_,_,k_block));
      copy(tCsSFB_stage(_,_,k_block), tCrSFB_copy_view(_,_,k_block));
    };

    auto gemm_kblock = [&](auto k_block) {
      cute::gemm(
          tiled_mma,
          make_zip_tensor(tCrA(_,_,k_block), tCrSFA(_,_,k_block)),
          make_zip_tensor(tCrB(_,_,k_block), tCrSFB(_,_,k_block)),
          accum);
    };

    wait_both();
    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(
              thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          release_advance_both();
          wait_both();
        }

        copy_kblock(k_block_next);
        gemm_kblock(k_block);
      });
    }

    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

      if (k_block == K_BLOCK_MAX - 1) {
        cutlass::arch::NamedBarrier::sync(
            thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        release_advance_both();
      }

      if (k_block_next > 0) {
        copy_kblock(k_block_next);
      }
      gemm_kblock(k_block);
    });
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline, PipelineState, int) {
  }

  CUTLASS_DEVICE void
  mma_tail(MainloopPipelineMK, PipelineStateMK, MainloopPipelineNK, PipelineStateNK, int) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
