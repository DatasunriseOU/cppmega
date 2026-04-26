#!/usr/bin/env python3
"""Prototype CUTLASS SM121 MXFP8 GEMM for TE compact transpose operands.

This is a probe-only extension.  It consumes TE-like compact MXFP8 tensors:

* A payload is logical rowwise [M, K] with compact rowwise scales [M, K / 32].
* B payload is the original TE columnwise storage [K, N].  The opt-in manual
  path tries to expose it as logical B with stride_N=1,stride_K=N so B(n, k)
  aliases source[k, n] without a payload transpose.
* B scales are compact TE columnwise [K / 32, N].

The direct custom-global-scale-layout mainloop is blocked by TMA descriptor
constraints for compact scale layouts.  This probe keeps the payload no-copy
attempt opt-in because it currently aborts in TMA descriptor creation too.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


CUTLASS_ROOT = Path("/home/dave/vllm/.deps/cutlass-src")
BLOCK_K = 32


_CPP_SOURCE = r"""
#include <torch/extension.h>

pybind11::dict cutlass_mxfp8_te_compact_versions();
pybind11::dict run_cutlass_mxfp8_te_compact_gemm(
    torch::Tensor A_u8,
    torch::Tensor SFA_u8,
    torch::Tensor B_colwise_u8,
    torch::Tensor SFB_colwise_u8,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t warmup,
    int64_t iters);
"""


_CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

namespace py = pybind11;
using namespace cute;

namespace cppmega_mxfp8_te_compact_probe {

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

const char* cuda_status_name(cudaError_t status) {
  return cudaGetErrorName(status);
}

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_UINT8(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Byte, #x " must be uint8")

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
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using NoCopyBStride = cute::Stride<cute::Int<1>, int64_t, int64_t>;

using NoCopyBCollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    typename CollectiveMainloop::DispatchPolicy,
    ThreadBlockShape,
    cute::tuple<ElementAData, ElementSF>,
    cute::tuple<StrideA, LayoutSFA>,
    cute::tuple<ElementBData, ElementSF>,
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

using NoCopyBGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    NoCopyBCollectiveMainloop,
    CollectiveEpilogue,
    void>;

using NoCopyBGemm = cutlass::gemm::device::GemmUniversalAdapter<NoCopyBGemmKernel>;

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

template <class Layout>
__global__ void prepack_colwise_transpose_scale_kernel(
    uint8_t const* __restrict__ compact_colwise,
    uint8_t* __restrict__ native,
    Layout native_layout,
    int cols,
    int k_blocks,
    int compact_ld) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = cols * k_blocks;
  if (idx >= total) {
    return;
  }
  int col = idx / k_blocks;
  int kb = idx - col * k_blocks;
  native[native_layout(col, kb * 32, 0)] = compact_colwise[kb * compact_ld + col];
}

void check_inputs(
    torch::Tensor const& A_u8,
    torch::Tensor const& SFA_u8,
    torch::Tensor const& B_colwise_u8,
    torch::Tensor const& SFB_colwise_u8,
    int m,
    int n,
    int k) {
  CHECK_CUDA(A_u8);
  CHECK_CUDA(SFA_u8);
  CHECK_CUDA(B_colwise_u8);
  CHECK_CUDA(SFB_colwise_u8);
  CHECK_CONTIGUOUS(A_u8);
  CHECK_CONTIGUOUS(SFA_u8);
  CHECK_CONTIGUOUS(B_colwise_u8);
  CHECK_CONTIGUOUS(SFB_colwise_u8);
  CHECK_UINT8(A_u8);
  CHECK_UINT8(SFA_u8);
  CHECK_UINT8(B_colwise_u8);
  CHECK_UINT8(SFB_colwise_u8);
  TORCH_CHECK(m > 0 && n > 0 && k > 0, "m, n, k must be positive");
  TORCH_CHECK(m % 128 == 0 && n % 128 == 0 && k % 128 == 0,
              "this minimal prototype requires m, n, k multiples of 128");
  TORCH_CHECK(A_u8.numel() >= static_cast<int64_t>(m) * k, "A_u8 is too small");
  TORCH_CHECK(B_colwise_u8.numel() >= static_cast<int64_t>(k) * n,
              "B_colwise_u8 is too small");
  TORCH_CHECK(SFA_u8.dim() == 2, "SFA_u8 must be [M, K/32] or padded");
  TORCH_CHECK(SFB_colwise_u8.dim() == 2,
              "SFB_colwise_u8 must be [K/32, N] or padded");
  int k_blocks = k / 32;
  TORCH_CHECK(SFA_u8.size(0) >= m, "SFA_u8 dim0 is smaller than M");
  TORCH_CHECK(SFA_u8.size(1) >= k_blocks, "SFA_u8 dim1 is smaller than K/32");
  TORCH_CHECK(SFB_colwise_u8.size(0) >= k_blocks,
              "SFB_colwise_u8 dim0 is smaller than K/32");
  TORCH_CHECK(SFB_colwise_u8.size(1) >= n, "SFB_colwise_u8 dim1 is smaller than N");
}

#endif

}  // namespace cppmega_mxfp8_te_compact_probe

using namespace cppmega_mxfp8_te_compact_probe;

py::dict cutlass_mxfp8_te_compact_versions() {
  py::dict out;
  out["compile_time_cudart_version"] = CUDART_VERSION;
  int runtime_version = 0;
  cudaError_t status = cudaRuntimeGetVersion(&runtime_version);
  out["runtime_cudart_status"] = cuda_status_name(status);
  out["runtime_cudart_version"] = runtime_version;
  int driver_version = 0;
  status = cudaDriverGetVersion(&driver_version);
  out["driver_status"] = cuda_status_name(status);
  out["driver_version"] = driver_version;
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  out["cutlass_sm120_supported_macro"] = true;
#else
  out["cutlass_sm120_supported_macro"] = false;
#endif
#if defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  out["cutlass_sm121_supported_macro"] = true;
#else
  out["cutlass_sm121_supported_macro"] = false;
#endif
  return out;
}

py::dict run_cutlass_mxfp8_te_compact_gemm(
    torch::Tensor A_u8,
    torch::Tensor SFA_u8,
    torch::Tensor B_colwise_u8,
    torch::Tensor SFB_colwise_u8,
    int64_t m64,
    int64_t n64,
    int64_t k64,
    int64_t warmup,
    int64_t iters) {
  py::dict out;

#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  out["status"] = "unsupported_compile_arch";
  return out;
#else
  TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "m, n, k exceed int");
  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);
  int k_blocks = k / 32;
  int l = 1;
  check_inputs(A_u8, SFA_u8, B_colwise_u8, SFB_colwise_u8, m, n, k);

  c10::cuda::CUDAGuard device_guard(A_u8.device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, l));
  NoCopyBStride stride_B =
      make_stride(Int<1>{}, static_cast<int64_t>(n), static_cast<int64_t>(n) * k);
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, l));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, l));
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, l));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, l));

  int64_t native_sfa_elems = static_cast<int64_t>(size(filter_zeros(layout_SFA)));
  int64_t native_sfb_elems = static_cast<int64_t>(size(filter_zeros(layout_SFB)));
  auto byte_options = torch::TensorOptions().device(A_u8.device()).dtype(torch::kUInt8);
  torch::Tensor native_SFA = torch::zeros({native_sfa_elems}, byte_options);
  torch::Tensor native_SFB = torch::zeros({native_sfb_elems}, byte_options);

  constexpr int threads = 256;
  int a_blocks = (m * k_blocks + threads - 1) / threads;
  int b_blocks = (n * k_blocks + threads - 1) / threads;
  prepack_rowwise_scale_kernel<<<a_blocks, threads, 0, stream>>>(
      SFA_u8.data_ptr<uint8_t>(),
      native_SFA.data_ptr<uint8_t>(),
      layout_SFA,
      m,
      k_blocks,
      static_cast<int>(SFA_u8.size(1)));
  prepack_colwise_transpose_scale_kernel<<<b_blocks, threads, 0, stream>>>(
      SFB_colwise_u8.data_ptr<uint8_t>(),
      native_SFB.data_ptr<uint8_t>(),
      layout_SFB,
      n,
      k_blocks,
      static_cast<int>(SFB_colwise_u8.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto out_options = torch::TensorOptions().device(A_u8.device()).dtype(torch::kBFloat16);
  torch::Tensor D = torch::empty({m, n}, out_options);

  auto ptr_A = reinterpret_cast<ElementAData const*>(A_u8.data_ptr<uint8_t>());
  auto ptr_B = reinterpret_cast<ElementBData const*>(B_colwise_u8.data_ptr<uint8_t>());
  auto ptr_SFA = reinterpret_cast<ElementSF const*>(native_SFA.data_ptr<uint8_t>());
  auto ptr_SFB = reinterpret_cast<ElementSF const*>(native_SFB.data_ptr<uint8_t>());
  auto ptr_D = reinterpret_cast<ElementD*>(D.data_ptr<at::BFloat16>());

  typename NoCopyBGemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {ptr_A, stride_A, ptr_B, stride_B, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB},
      {{1.0f, 0.0f}, ptr_D, stride_C, ptr_D, stride_D}};

  NoCopyBGemm gemm;
  cutlass::Status can_status = gemm.can_implement(arguments);
  out["can_implement"] = cutlass_status_name(can_status);
  if (can_status != cutlass::Status::kSuccess) {
    out["status"] = "can_implement_failed";
    return out;
  }

  size_t workspace_size = NoCopyBGemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status init_status = gemm.initialize(arguments, workspace.get(), stream);
  out["initialize"] = cutlass_status_name(init_status);
  if (init_status != cutlass::Status::kSuccess) {
    out["status"] = "initialize_failed";
    return out;
  }

  auto run_once = [&]() {
    cutlass::Status run_status = gemm.run(stream);
    TORCH_CHECK(run_status == cutlass::Status::kSuccess,
                "CUTLASS run failed: ", cutlass_status_name(run_status));
  };

  for (int64_t i = 0; i < warmup; ++i) {
    run_once();
  }

  float elapsed_ms = 0.0f;
  if (iters > 0) {
    cudaEvent_t start;
    cudaEvent_t stop;
    C10_CUDA_CHECK(cudaEventCreate(&start));
    C10_CUDA_CHECK(cudaEventCreate(&stop));
    C10_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int64_t i = 0; i < iters; ++i) {
      run_once();
    }
    C10_CUDA_CHECK(cudaEventRecord(stop, stream));
    C10_CUDA_CHECK(cudaEventSynchronize(stop));
    C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    C10_CUDA_CHECK(cudaEventDestroy(start));
    C10_CUDA_CHECK(cudaEventDestroy(stop));
  } else {
    run_once();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  out["status"] = "success";
  out["out"] = D;
  out["native_sfa_elems"] = native_sfa_elems;
  out["native_sfb_elems"] = native_sfb_elems;
  out["compact_sfa_elems"] = static_cast<int64_t>(SFA_u8.numel());
  out["compact_sfb_elems"] = static_cast<int64_t>(SFB_colwise_u8.numel());
  out["workspace_size"] = static_cast<unsigned long long>(workspace_size);
  out["elapsed_ms_total"] = elapsed_ms;
  out["elapsed_ms_per_iter"] = iters > 0 ? elapsed_ms / static_cast<float>(iters) : 0.0f;
  return out;
#endif
}
"""


def _load_extension(args: argparse.Namespace) -> Any:
    import torch
    from torch.utils.cpp_extension import load_inline

    cutlass = Path(args.cutlass_root)
    include_dir = cutlass / "include"
    util_include_dir = cutlass / "tools" / "util" / "include"
    if not (include_dir / "cutlass" / "cutlass.h").exists():
        raise FileNotFoundError(f"CUTLASS include tree not found under {cutlass}")

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability(args.device)
        if (major, minor) == (12, 1):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
        elif (major, minor) == (12, 0):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    build_dir = Path(args.build_dir) if args.build_dir else Path(tempfile.gettempdir())
    build_dir.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name="cppmega_cutlass_mxfp8_te_compact_gemm_probe",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        functions=[
            "cutlass_mxfp8_te_compact_versions",
            "run_cutlass_mxfp8_te_compact_gemm",
        ],
        extra_include_paths=[str(include_dir), str(util_include_dir)],
        extra_cflags=["-O2", "-std=c++17"],
        extra_cuda_cflags=[
            "-O2",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
        build_directory=str(build_dir),
        verbose=args.verbose_build,
        with_cuda=True,
    )


def _rel_l2(a: Any, b: Any) -> float:
    import torch

    num = torch.linalg.vector_norm((a.float() - b.float()).reshape(-1))
    den = torch.linalg.vector_norm(b.float().reshape(-1)).clamp_min(1e-12)
    return float((num / den).item())


def _max_abs(a: Any, b: Any) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not args.attempt_no_copy_payload:
        return {
            "status": "not_run",
            "reason": (
                "manual no-copy B payload stride currently aborts in CUTLASS/CuTe TMA "
                "descriptor creation; pass --attempt-no-copy-payload to reproduce"
            ),
            "shape": {"m": args.m, "n": args.n, "k": args.k},
            "notes": [
                "The failed mode uses B stride_N=1,stride_K=N to alias TE [K,N] payload.",
                "Observed blocker: TMA asserts gmem_prob_stride[0] == 1 / majorness mismatch.",
            ],
        }

    if not torch.cuda.is_available():
        return {"status": "skip", "reason": "CUDA is not available"}
    if not hasattr(torch, "float8_e4m3fn"):
        return {"status": "skip", "reason": "torch.float8_e4m3fn is unavailable"}

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    ext = _load_extension(args)
    torch.manual_seed(args.seed)

    m, n, k = args.m, args.n, args.k
    k_blocks = k // BLOCK_K
    a = torch.randn((m, k), device=device, dtype=torch.float32).clamp(-args.clamp, args.clamp)
    b = torch.randn((k, n), device=device, dtype=torch.float32).clamp(-args.clamp, args.clamp)
    a_fp8 = a.to(torch.float8_e4m3fn).contiguous()
    b_colwise_fp8 = b.to(torch.float8_e4m3fn).contiguous()

    sfa = torch.full((m, k_blocks), args.scale_byte, device=device, dtype=torch.uint8)
    sfb = torch.full((k_blocks, n), args.scale_byte, device=device, dtype=torch.uint8)
    result = dict(
        ext.run_cutlass_mxfp8_te_compact_gemm(
            a_fp8.view(torch.uint8),
            sfa,
            b_colwise_fp8.view(torch.uint8),
            sfb,
            m,
            n,
            k,
            args.warmup,
            args.iters,
        )
    )
    out = result.pop("out", None)
    if result.get("status") == "success" and out is not None:
        ref = (a_fp8.float() @ b_colwise_fp8.float()).to(torch.bfloat16)
        result["max_abs_vs_identity_bf16_ref"] = _max_abs(out, ref)
        result["rel_l2_vs_identity_bf16_ref"] = _rel_l2(out, ref)
        result["out_norm"] = float(out.float().norm().item())
        result["ref_norm"] = float(ref.float().norm().item())
        if m == n == k:
            alt_refs = {
                "A@B.T": a_fp8.float() @ b_colwise_fp8.float().t(),
                "A.T@B": a_fp8.float().t() @ b_colwise_fp8.float(),
                "A.T@B.T": a_fp8.float().t() @ b_colwise_fp8.float().t(),
            }
            result["square_alt_rel_l2"] = {
                name: _rel_l2(out, alt.to(torch.bfloat16))
                for name, alt in alt_refs.items()
            }
        result["math_status"] = (
            "pass"
            if result["rel_l2_vs_identity_bf16_ref"] <= args.rel_l2_limit
            else "bad_math"
        )

    result["torch"] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }
    result["cutlass"] = {
        "root": str(Path(args.cutlass_root)),
        "versions": dict(ext.cutlass_mxfp8_te_compact_versions()),
    }
    result["shape"] = {"m": m, "n": n, "k": k}
    result["scale_byte"] = args.scale_byte
    result["notes"] = [
        "B payload aliases original TE columnwise [K,N] storage as CUTLASS column-major [N,K].",
        "Only compact scale bytes are prepacked to native SM1xx scale layout.",
        "scale_byte=127 is the identity UE8M0 scale for the BF16 reference check.",
    ]
    return result


def _validate(args: argparse.Namespace) -> None:
    if args.m % 128 or args.n % 128 or args.k % 128:
        raise SystemExit("--m, --n, and --k must be multiples of 128")
    if args.k <= 0 or args.m <= 0 or args.n <= 0:
        raise SystemExit("--m, --n, and --k must be positive")
    if not 0 <= args.scale_byte <= 255:
        raise SystemExit("--scale-byte must be in [0, 255]")
    if args.rel_l2_limit <= 0:
        raise SystemExit("--rel-l2-limit must be positive")
    if args.warmup < 0 or args.iters < 0:
        raise SystemExit("--warmup and --iters must be non-negative")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--clamp", type=float, default=2.0)
    parser.add_argument("--scale-byte", type=int, default=127)
    parser.add_argument("--rel-l2-limit", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument(
        "--attempt-no-copy-payload",
        action="store_true",
        help="Run the current manual no-copy B payload attempt. This is expected to abort in TMA descriptor creation.",
    )
    parser.add_argument("--cutlass-root", default=str(CUTLASS_ROOT))
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    _validate(args)
    report = _run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("status") == "not_run":
        return 0
    if report.get("status") != "success":
        return 1
    if report.get("math_status") != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
