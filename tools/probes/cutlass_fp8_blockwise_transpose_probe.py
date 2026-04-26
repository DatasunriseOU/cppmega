#!/usr/bin/env python3
"""Probe CUTLASS SM120/SM121 FP8 blockwise GEMM with TE colwise layout.

This checks the custom-layout idea for a no-payload-transpose backward GEMM:

* B uses the original TE columnwise payload bytes with source shape [K, N].
  The desired logical transpose would be B(n, k) -> source[k, n], i.e.
  stride_N=1,stride_K=N.
* B scales use compact TE columnwise indexing.  CUTLASS sees SFB as logical
  [N, K / 32], but the runtime stride over K-blocks is padded_N, so
  SFB(n, kb) maps to columnwise_scale_inv[kb, n].
* The implemented GEMM is the CUTLASS software-scale path: FP8 E4M3 data plus
  FP32 scale factors.  Direct MXFP8 E8M0 scale consumption still needs a
  custom mainloop scale-load/decode hook.

The important SM120 finding is that the stock builder only accepts TN/K-major
operand layouts.  For B this means stride_N=K,stride_K=1, which requires a
materialized [N, K] FP8 payload.  The probe keeps the failing no-copy variant
and a materialized-transpose control so the boundary is explicit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


CUTLASS_ROOT = Path("/home/dave/vllm/.deps/cutlass-src")
BLOCK_K = 32


_CPP_BINDINGS = r"""
#include <torch/extension.h>

namespace py = pybind11;

py::dict cutlass_probe_versions();
py::dict run_cutlass_fp8_blockwise_colwise(
    torch::Tensor A_u8,
    torch::Tensor SFA,
    torch::Tensor B_colwise_u8,
    torch::Tensor SFB_colwise,
    int64_t m,
    int64_t n,
    int64_t k);
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

namespace py = pybind11;
using namespace cute;

namespace {

const char *cutlass_status_name(cutlass::Status status) {
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

const char *cuda_status_name(cudaError_t status) {
  return cudaGetErrorName(status);
}

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute = float;

using MmaTileShape_MNK = Shape<_128, _128, _32>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

constexpr int ScaleGranularityM = 1;
constexpr int ScaleGranularityN = 1;
constexpr int ScaleGranularityK = 32;

using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
    ScaleGranularityM,
    ScaleGranularityN,
    ScaleGranularityK,
    UMMA::Major::K,
    UMMA::Major::MN>;

using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutC,
    AlignmentC,
    ElementD,
    LayoutD,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassTensorOp,
    ElementA,
    cute::tuple<LayoutA, LayoutSFA>,
    AlignmentA,
    ElementB,
    cute::tuple<LayoutB, LayoutSFB>,
    AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

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

LayoutSFA make_sfa_layout(int m, int k, int l, int sfa_ld, int sfa_batch_stride) {
  int k_blocks = cute::ceil_div(k, ScaleGranularityK);
  auto mk_layout = make_layout(
      make_shape(make_shape(Int<ScaleGranularityM>{}, m),
                 make_shape(Int<ScaleGranularityK>{}, k_blocks)),
      make_stride(make_stride(_0{}, sfa_ld), make_stride(_0{}, _1{})));
  return make_layout(append(shape(mk_layout), l), append(stride(mk_layout), sfa_batch_stride));
}

LayoutSFB make_sfb_layout(int n, int k, int l, int sfb_ld, int sfb_batch_stride) {
  int k_blocks = cute::ceil_div(k, ScaleGranularityK);
  auto nk_layout = make_layout(
      make_shape(make_shape(Int<ScaleGranularityN>{}, n),
                 make_shape(Int<ScaleGranularityK>{}, k_blocks)),
      make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, sfb_ld)));
  return make_layout(append(shape(nk_layout), l), append(stride(nk_layout), sfb_batch_stride));
}

#endif

}  // namespace

py::dict cutlass_probe_versions() {
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

py::dict run_cutlass_fp8_blockwise_colwise(
    torch::Tensor A_u8,
    torch::Tensor SFA,
    torch::Tensor B_colwise_u8,
    torch::Tensor SFB_colwise,
    int64_t m64,
    int64_t n64,
    int64_t k64) {
  py::dict out;

#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && !defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  out["status"] = "unsupported_compile_arch";
  return out;
#else
  TORCH_CHECK(A_u8.is_cuda(), "A_u8 must be CUDA");
  TORCH_CHECK(B_colwise_u8.is_cuda(), "B_colwise_u8 must be CUDA");
  TORCH_CHECK(SFA.is_cuda(), "SFA must be CUDA");
  TORCH_CHECK(SFB_colwise.is_cuda(), "SFB_colwise must be CUDA");
  TORCH_CHECK(A_u8.scalar_type() == at::ScalarType::Byte, "A_u8 must be uint8");
  TORCH_CHECK(B_colwise_u8.scalar_type() == at::ScalarType::Byte, "B_colwise_u8 must be uint8");
  TORCH_CHECK(SFA.scalar_type() == at::ScalarType::Float, "SFA must be float32");
  TORCH_CHECK(SFB_colwise.scalar_type() == at::ScalarType::Float, "SFB_colwise must be float32");
  TORCH_CHECK(A_u8.is_contiguous(), "A_u8 must be contiguous");
  TORCH_CHECK(B_colwise_u8.is_contiguous(), "B_colwise_u8 must be contiguous");
  TORCH_CHECK(SFA.is_contiguous(), "SFA must be contiguous");
  TORCH_CHECK(SFB_colwise.is_contiguous(), "SFB_colwise must be contiguous");
  TORCH_CHECK(m64 > 0 && n64 > 0 && k64 > 0, "m, n, k must be positive");
  TORCH_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, "m, n, k exceed int");

  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);
  int l = 1;
  int k_blocks = cute::ceil_div(k, ScaleGranularityK);

  TORCH_CHECK(k % ScaleGranularityK == 0, "k must be divisible by 32");
  TORCH_CHECK(A_u8.numel() >= static_cast<int64_t>(m) * k, "A_u8 is too small");
  TORCH_CHECK(B_colwise_u8.numel() >= static_cast<int64_t>(k) * n, "B_colwise_u8 is too small");
  TORCH_CHECK(SFA.dim() == 2, "SFA must be [M_padded, Kblocks_padded]");
  TORCH_CHECK(SFB_colwise.dim() == 2, "SFB_colwise must be [Kblocks_padded, N_padded]");
  TORCH_CHECK(SFA.size(0) >= m, "SFA first dim is smaller than m");
  TORCH_CHECK(SFA.size(1) >= k_blocks, "SFA second dim is smaller than k/32");
  TORCH_CHECK(SFB_colwise.size(0) >= k_blocks, "SFB first dim is smaller than k/32");
  TORCH_CHECK(SFB_colwise.size(1) >= n, "SFB second dim is smaller than n");

  c10::cuda::CUDAGuard device_guard(A_u8.device());
  auto D = torch::empty(
      {m, n},
      torch::TensorOptions().device(A_u8.device()).dtype(torch::kBFloat16));

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, l));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, l));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, l));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, l));

  int sfa_ld = static_cast<int>(SFA.size(1));
  int sfb_ld = static_cast<int>(SFB_colwise.size(1));
  int sfa_batch_stride = static_cast<int>(SFA.size(0) * SFA.size(1));
  int sfb_batch_stride = static_cast<int>(SFB_colwise.size(0) * SFB_colwise.size(1));
  LayoutSFA layout_SFA = make_sfa_layout(m, k, l, sfa_ld, sfa_batch_stride);
  LayoutSFB layout_SFB = make_sfb_layout(n, k, l, sfb_ld, sfb_batch_stride);

  auto ptr_A = reinterpret_cast<ElementA const*>(A_u8.data_ptr<uint8_t>());
  auto ptr_B = reinterpret_cast<ElementB const*>(B_colwise_u8.data_ptr<uint8_t>());
  auto ptr_D = reinterpret_cast<ElementD*>(D.data_ptr<at::BFloat16>());

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {ptr_A,
       stride_A,
       ptr_B,
       stride_B,
       SFA.data_ptr<float>(),
       layout_SFA,
       SFB_colwise.data_ptr<float>(),
       layout_SFB},
      {{}, ptr_D, stride_C, ptr_D, stride_D}};

  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  Gemm gemm;
  cutlass::Status can_status = gemm.can_implement(arguments);
  out["can_implement"] = cutlass_status_name(can_status);
  if (can_status != cutlass::Status::kSuccess) {
    out["status"] = "can_implement_failed";
    return out;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  cutlass::Status init_status = gemm.initialize(arguments, workspace.get(), stream);
  out["initialize"] = cutlass_status_name(init_status);
  if (init_status != cutlass::Status::kSuccess) {
    out["status"] = "initialize_failed";
    return out;
  }

  cutlass::Status run_status = gemm.run(stream);
  out["run"] = cutlass_status_name(run_status);
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (run_status != cutlass::Status::kSuccess) {
    out["status"] = "run_failed";
    return out;
  }

  out["status"] = "success";
  out["out"] = D;
  out["workspace_size"] = static_cast<unsigned long long>(workspace_size);
  out["tile_shape"] = py::make_tuple(128, 128, 32);
  out["scale_granularity"] = py::make_tuple(1, 1, 32);
  out["sfa_ld"] = sfa_ld;
  out["sfb_ld"] = sfb_ld;
  return out;
#endif
}
"""


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _layout_report(m: int, n: int, k: int) -> dict[str, Any]:
    k_blocks = k // BLOCK_K
    sfa_rows = _round_up(m, 128)
    sfa_cols = _round_up(k_blocks, 4)
    sfb_rows = _round_up(k_blocks, 4)
    sfb_cols = _round_up(n, 128)
    samples = []
    for mi in (0, min(m - 1, 7)):
        for ni in (0, min(n - 1, 11)):
            for ki in (0, min(k - 1, 63)):
                kb = ki // BLOCK_K
                samples.append(
                    {
                        "m": mi,
                        "n": ni,
                        "k": ki,
                        "b_source_offset": ki * n + ni,
                        "b_desired_direct_offset": ni + ki * n,
                        "b_stock_sm120_k_major_offset_on_original": ni * k + ki,
                        "sfa_rowwise_offset": mi * sfa_cols + kb,
                        "sfb_colwise_source_offset": kb * sfb_cols + ni,
                        "sfb_cutlass_offset": ni + kb * sfb_cols,
                    }
                )
    direct_b_ok = all(row["b_source_offset"] == row["b_desired_direct_offset"] for row in samples)
    stock_b_mismatch = any(
        row["b_source_offset"] != row["b_stock_sm120_k_major_offset_on_original"]
        for row in samples
    )
    sfb_ok = all(
        row["sfb_colwise_source_offset"] == row["sfb_cutlass_offset"] for row in samples
    )
    return {
        "status": "pass" if direct_b_ok and sfb_ok and stock_b_mismatch else "fail",
        "shape": {"m": m, "n": n, "k": k, "k_blocks": k_blocks},
        "te_compact_scale_shapes": {
            "a_rowwise_float_or_decoded": [sfa_rows, sfa_cols],
            "b_columnwise_float_or_decoded": [sfb_rows, sfb_cols],
        },
        "cutlass_layouts": {
            "B_desired_no_copy": "logical (N,K), stride_N=1,stride_K=N over source [K,N]",
            "B_stock_sm120": "TN/K-major B requires stride_N=K,stride_K=1, so source [K,N] is wrong without transpose",
            "SFA": "major-K layout offset=m*padded_Kblocks+kb",
            "SFB": "major-MN layout offset=n+kb*padded_N over compact columnwise scales",
        },
        "samples": samples,
    }


def _load_extension(args: argparse.Namespace):
    import torch
    from torch.utils.cpp_extension import load_inline

    include_root = Path(args.cutlass_root)
    include_dir = include_root / "include"
    util_include_dir = include_root / "tools" / "util" / "include"
    if not include_dir.exists():
        raise FileNotFoundError(f"CUTLASS include dir not found: {include_dir}")

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
        name="cppmega_cutlass_fp8_blockwise_transpose_probe",
        cpp_sources=[_CPP_BINDINGS],
        cuda_sources=[_CUDA_SOURCE],
        functions=["cutlass_probe_versions", "run_cutlass_fp8_blockwise_colwise"],
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


def _rel_l2(a, b) -> float:
    import torch

    num = torch.linalg.vector_norm((a.float() - b.float()).reshape(-1))
    den = torch.linalg.vector_norm(b.float().reshape(-1)).clamp_min(1e-12)
    return float((num / den).item())


def _max_abs(a, b) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _run_gemm(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        if args.require_gemm:
            raise RuntimeError("CUDA is not available")
        return {"status": "skip", "reason": "CUDA is not available"}
    if not hasattr(torch, "float8_e4m3fn"):
        if args.require_gemm:
            raise RuntimeError("torch.float8_e4m3fn is unavailable")
        return {"status": "skip", "reason": "torch.float8_e4m3fn is unavailable"}

    torch.cuda.set_device(args.device)
    ext = _load_extension(args)

    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    m, n, k = args.m, args.n, args.k
    k_blocks = k // BLOCK_K

    a = torch.randn((m, k), device=device, dtype=torch.float32).clamp(-args.clamp, args.clamp)
    b_source = torch.randn((k, n), device=device, dtype=torch.float32).clamp(
        -args.clamp, args.clamp
    )
    a_fp8 = a.to(torch.float8_e4m3fn).contiguous()
    b_colwise_fp8 = b_source.to(torch.float8_e4m3fn).contiguous()

    sfa_rows = _round_up(m, 128)
    sfa_cols = _round_up(k_blocks, 4)
    sfb_rows = _round_up(k_blocks, 4)
    sfb_cols = _round_up(n, 128)
    sfa = torch.ones((sfa_rows, sfa_cols), device=device, dtype=torch.float32)
    sfb = torch.ones((sfb_rows, sfb_cols), device=device, dtype=torch.float32)
    if not args.identity_scales:
        sfa[:m, :k_blocks] = torch.rand((m, k_blocks), device=device) * 0.5 + 0.75
        sfb[:k_blocks, :n] = torch.rand((k_blocks, n), device=device) * 0.5 + 0.75

    a_scale = sfa[:m, :k_blocks].repeat_interleave(BLOCK_K, dim=1)[:, :k]
    b_scale = sfb[:k_blocks, :n].repeat_interleave(BLOCK_K, dim=0)[:k, :]
    a_deq = a_fp8.to(torch.float32) * a_scale
    b_deq = b_colwise_fp8.to(torch.float32) * b_scale
    ref = a_deq @ b_deq
    ref_bf16 = ref.to(torch.bfloat16)

    b_materialized_t_fp8 = b_colwise_fp8.t().contiguous()

    def run_variant(name: str, b_payload: Any, intent: str) -> dict[str, Any]:
        row = dict(
            ext.run_cutlass_fp8_blockwise_colwise(
                a_fp8.view(torch.uint8),
                sfa,
                b_payload.view(torch.uint8),
                sfb,
                m,
                n,
                k,
            )
        )
        cleaned = {key: value for key, value in row.items() if key != "out"}
        cleaned["name"] = name
        cleaned["intent"] = intent
        out_tensor = row.get("out")
        if cleaned.get("status") == "success" and out_tensor is not None:
            cleaned["max_abs_vs_ref_bf16"] = _max_abs(out_tensor, ref_bf16)
            cleaned["rel_l2_vs_ref_bf16"] = _rel_l2(out_tensor, ref_bf16)
            cleaned["max_abs_vs_ref_fp32"] = _max_abs(out_tensor, ref)
            cleaned["rel_l2_vs_ref_fp32"] = _rel_l2(out_tensor, ref)
            if m == n == k:
                alt_refs = {
                    "A@source.T": a_deq @ b_deq.t(),
                    "A.T@source": a_deq.t() @ b_deq,
                    "A.T@source.T": a_deq.t() @ b_deq.t(),
                }
                cleaned["square_alt_rel_l2"] = {
                    alt_name: _rel_l2(out_tensor, alt_ref.to(torch.bfloat16))
                    for alt_name, alt_ref in alt_refs.items()
                }
            cleaned["math_status"] = (
                "pass" if cleaned["rel_l2_vs_ref_bf16"] <= args.rel_l2_limit else "bad_math"
            )
        return cleaned

    variants = [
        run_variant(
            "original_te_columnwise_payload_as_b",
            b_colwise_fp8,
            "no-copy target: source is [K,N], but stock SM120 B is K-major [N,K]",
        ),
        run_variant(
            "materialized_transpose_control",
            b_materialized_t_fp8,
            "control: physical [N,K] B payload required by stock SM120 TN/K-major path",
        ),
    ]
    variant_by_name = {row["name"]: row for row in variants}
    original_status = variant_by_name["original_te_columnwise_payload_as_b"].get("math_status")
    control_status = variant_by_name["materialized_transpose_control"].get("math_status")
    if control_status == "pass" and original_status == "bad_math":
        status = "stock_sm120_requires_payload_transpose"
    elif control_status == "pass":
        status = "inconclusive"
    else:
        status = "control_failed"
    return {
        "status": status,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        },
        "cutlass": {
            "root": str(Path(args.cutlass_root)),
            "versions": dict(ext.cutlass_probe_versions()),
        },
        "shape": {"m": m, "n": n, "k": k},
        "variants": variants,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise SystemExit("--m, --n, and --k must be positive")
    if args.k % BLOCK_K:
        raise SystemExit(f"--k must be divisible by {BLOCK_K}")
    if args.m % 16 or args.n % 16:
        raise SystemExit("--m and --n must be multiples of 16 for TMA alignment")
    if args.rel_l2_limit <= 0:
        raise SystemExit("--rel-l2-limit must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--clamp", type=float, default=2.0)
    parser.add_argument("--rel-l2-limit", type=float, default=1e-5)
    parser.add_argument("--identity-scales", action="store_true")
    parser.add_argument("--cutlass-root", default=str(CUTLASS_ROOT))
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument("--require-gemm", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    _validate_args(args)

    report: dict[str, Any] = {
        "layout": _layout_report(args.m, args.n, args.k),
        "notes": [
            "SM120 CUTLASS blockwise software-scale mainloop requires FP32 scale tensors.",
            "MXFP8 uint8 E8M0 columnwise_scale_inv can use these offsets, but needs decode before rescale.",
            "SM120 builder requires ScaleGranularityK == TileShape_K, so this probe uses K tile 32.",
            "Stock SM120 blockwise builder supports only TN/K-major A and B; direct original B payload is the expected failing variant.",
        ],
    }
    if not args.layout_only:
        report["gemm"] = _run_gemm(args)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
