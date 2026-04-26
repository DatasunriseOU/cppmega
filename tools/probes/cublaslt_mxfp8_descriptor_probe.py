#!/usr/bin/env python3
"""Probe local cuBLASLt MXFP8 descriptor/scale layout support.

The probe answers two local-environment questions:

* Which CUDA/cuBLASLt headers and CUBLASLT_MATMUL_DESC_*SCALE* attributes are
  present?
* Can a small E4M3 + VEC32_UE8M0 block-scaled GEMM be made correct by only
  changing cuBLASLt descriptors when a transposed/columnwise payload is needed?

The GEMM uses constant E8M0 scale bytes by default, so the tested "fixed scale
layout" is deliberately benign.  If descriptor-only transposition still fails
or produces bad math under identity scales, it cannot be a valid MXFP8 adapter
for real per-block scale layouts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cublas_v2.h>

namespace {

const char *status_name(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "NOT_SUPPORTED";
    default:
      return "UNKNOWN";
  }
}

const char *cuda_status_name(cudaError_t status) {
  return cudaGetErrorName(status);
}

void destroy_if(cublasLtMatmulPreference_t desc) {
  if (desc != nullptr) {
    cublasLtMatmulPreferenceDestroy(desc);
  }
}

void destroy_if(cublasLtMatrixLayout_t desc) {
  if (desc != nullptr) {
    cublasLtMatrixLayoutDestroy(desc);
  }
}

void destroy_if(cublasLtMatmulDesc_t desc) {
  if (desc != nullptr) {
    cublasLtMatmulDescDestroy(desc);
  }
}

void destroy_if(cublasLtHandle_t handle) {
  if (handle != nullptr) {
    cublasLtDestroy(handle);
  }
}

cublasStatus_t set_row_major(cublasLtMatrixLayout_t desc) {
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  return cublasLtMatrixLayoutSetAttribute(
      desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
}

}  // namespace

py::dict versions() {
  py::dict out;
  out["compile_time_cudart_version"] = CUDART_VERSION;
  out["compile_time_cublas_version"] = CUBLAS_VERSION;
  out["runtime_cublaslt_version"] = static_cast<unsigned long long>(cublasLtGetVersion());

  int runtime_version = 0;
  cudaError_t cuda_status = cudaRuntimeGetVersion(&runtime_version);
  out["runtime_cudart_status"] = cuda_status_name(cuda_status);
  out["runtime_cudart_version"] = runtime_version;

  int driver_version = 0;
  cuda_status = cudaDriverGetVersion(&driver_version);
  out["driver_status"] = cuda_status_name(cuda_status);
  out["driver_version"] = driver_version;

  int device_count = 0;
  cuda_status = cudaGetDeviceCount(&device_count);
  out["cuda_get_device_count_status"] = cuda_status_name(cuda_status);
  out["device_count"] = device_count;

  py::list devices;
  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop;
    cuda_status = cudaGetDeviceProperties(&prop, device);
    py::dict row;
    row["device"] = device;
    row["status"] = cuda_status_name(cuda_status);
    if (cuda_status == cudaSuccess) {
      row["name"] = prop.name;
      row["sm"] = py::make_tuple(prop.major, prop.minor);
      row["multi_processor_count"] = prop.multiProcessorCount;
    }
    devices.append(row);
  }
  out["devices"] = devices;
  return out;
}

py::dict run_mxfp8_matmul(
    torch::Tensor A,
    torch::Tensor A_scale,
    torch::Tensor B,
    torch::Tensor B_scale,
    int64_t m,
    int64_t n,
    int64_t k,
    bool transa,
    bool transb,
    int64_t a_rows,
    int64_t a_cols,
    int64_t a_ld,
    int64_t b_rows,
    int64_t b_cols,
    int64_t b_ld,
    int64_t workspace_bytes) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(A_scale.is_cuda(), "A_scale must be a CUDA tensor");
  TORCH_CHECK(B_scale.is_cuda(), "B_scale must be a CUDA tensor");
  TORCH_CHECK(A.scalar_type() == at::ScalarType::Byte, "A must be uint8");
  TORCH_CHECK(B.scalar_type() == at::ScalarType::Byte, "B must be uint8");
  TORCH_CHECK(A_scale.scalar_type() == at::ScalarType::Byte, "A_scale must be uint8");
  TORCH_CHECK(B_scale.scalar_type() == at::ScalarType::Byte, "B_scale must be uint8");

  c10::cuda::CUDAGuard device_guard(A.device());
  auto out = torch::empty(
      {m, n},
      torch::TensorOptions().device(A.device()).dtype(torch::kBFloat16));
  out.zero_();

  py::dict row;
  row["output_dtype"] = "bfloat16";
  row["scale_mode"] = "CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0";
  row["a_dtype"] = "CUDA_R_8F_E4M3";
  row["b_dtype"] = "CUDA_R_8F_E4M3";

  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  cublasStatus_t status = cublasLtCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    row["create_handle_status"] = status_name(status);
    return row;
  }

  status = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    row["matmul_desc_create_status"] = status_name(status);
    destroy_if(handle);
    return row;
  }

  cublasOperation_t op_a = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_b = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  row["transa"] = transa ? "T" : "N";
  row["transb"] = transb ? "T" : "N";

  status = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
  row["set_transa_status"] = status_name(status);
  status = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
  row["set_transb_status"] = status_name(status);

  int8_t fast_accum = 0;
  status = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum));
  row["set_fast_accum_status"] = status_name(status);

  cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  auto a_scale_ptr = A_scale.data_ptr<uint8_t>();
  auto b_scale_ptr = B_scale.data_ptr<uint8_t>();
  status = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
  row["set_a_scale_mode_status"] = status_name(status);
  status = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
  row["set_b_scale_mode_status"] = status_name(status);
  status = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr));
  row["set_a_scale_pointer_status"] = status_name(status);
  status = cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr));
  row["set_b_scale_pointer_status"] = status_name(status);

  status = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, a_rows, a_cols, a_ld);
  row["a_layout_create_status"] = status_name(status);
  if (status == CUBLAS_STATUS_SUCCESS) {
    row["a_order_status"] = status_name(set_row_major(a_desc));
  }
  status = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, b_rows, b_cols, b_ld);
  row["b_layout_create_status"] = status_name(status);
  if (status == CUBLAS_STATUS_SUCCESS) {
    row["b_order_status"] = status_name(set_row_major(b_desc));
  }
  status = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, m, n, n);
  row["c_layout_create_status"] = status_name(status);
  if (status == CUBLAS_STATUS_SUCCESS) {
    row["c_order_status"] = status_name(set_row_major(c_desc));
  }
  status = cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, m, n, n);
  row["d_layout_create_status"] = status_name(status);
  if (status == CUBLAS_STATUS_SUCCESS) {
    row["d_order_status"] = status_name(set_row_major(d_desc));
  }

  if (a_desc == nullptr || b_desc == nullptr || c_desc == nullptr || d_desc == nullptr) {
    row["status"] = "layout_fail";
    destroy_if(preference);
    destroy_if(a_desc);
    destroy_if(b_desc);
    destroy_if(c_desc);
    destroy_if(d_desc);
    destroy_if(op_desc);
    destroy_if(handle);
    return row;
  }

  status = cublasLtMatmulPreferenceCreate(&preference);
  row["preference_create_status"] = status_name(status);
  if (status == CUBLAS_STATUS_SUCCESS) {
    size_t workspace_size = static_cast<size_t>(workspace_bytes);
    status = cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size));
    row["set_workspace_status"] = status_name(status);
  }

  auto workspace = torch::empty(
      {workspace_bytes},
      torch::TensorOptions().device(A.device()).dtype(torch::kUInt8));
  cublasLtMatmulHeuristicResult_t heuristic = {};
  int returned_results = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      op_desc,
      a_desc,
      b_desc,
      c_desc,
      d_desc,
      preference,
      1,
      &heuristic,
      &returned_results);
  row["heuristic_status"] = status_name(status);
  row["returned_results"] = returned_results;
  if (returned_results > 0) {
    row["heuristic_result_status"] = status_name(heuristic.state);
    row["workspace_size"] = static_cast<unsigned long long>(heuristic.workspaceSize);
    row["waves_count"] = heuristic.wavesCount;
  }

  if (status == CUBLAS_STATUS_SUCCESS && returned_results > 0 &&
      heuristic.state == CUBLAS_STATUS_SUCCESS) {
    float alpha = 1.0f;
    float beta = 0.0f;
    status = cublasLtMatmul(
        handle,
        op_desc,
        &alpha,
        A.data_ptr<uint8_t>(),
        a_desc,
        B.data_ptr<uint8_t>(),
        b_desc,
        &beta,
        out.data_ptr<at::BFloat16>(),
        c_desc,
        out.data_ptr<at::BFloat16>(),
        d_desc,
        &heuristic.algo,
        workspace.data_ptr<uint8_t>(),
        static_cast<size_t>(workspace_bytes),
        at::cuda::getCurrentCUDAStream());
    row["matmul_status"] = status_name(status);
    cudaError_t cuda_status = cudaGetLastError();
    row["cuda_last_error"] = cuda_status_name(cuda_status);
  }

  row["out"] = out;
  destroy_if(preference);
  destroy_if(a_desc);
  destroy_if(b_desc);
  destroy_if(c_desc);
  destroy_if(d_desc);
  destroy_if(op_desc);
  destroy_if(handle);
  return row;
}
"""


_DESC_SCALE_RE = re.compile(
    r"^\s*(CUBLASLT_MATMUL_DESC_[A-Z0-9_]*SCALE[A-Z0-9_]*)\s*=\s*([0-9]+)"
)
_MATRIX_SCALE_RE = re.compile(
    r"^\s*(CUBLASLT_MATMUL_MATRIX_SCALE_[A-Z0-9x_]+)\s*(?:=\s*([0-9]+))?"
)
_DEFINE_RE = re.compile(r"^\s*#define\s+([A-Z0-9_]+)\s+([0-9]+)\b")


def _run_text(cmd: list[str], *, timeout_s: int = 10) -> str:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - environment probe
        return f"{type(exc).__name__}: {exc}"
    return (proc.stdout + proc.stderr).strip()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _candidate_include_dirs(root: Path) -> list[Path]:
    return [
        root / "include",
        root / "targets" / "sbsa-linux" / "include",
        root / "targets" / "x86_64-linux" / "include",
    ]


def _cuda_roots() -> list[Path]:
    roots: list[Path] = []
    env_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env_home:
        roots.append(Path(env_home))
    nvcc = shutil.which("nvcc")
    if nvcc:
        roots.append(Path(nvcc).resolve().parents[1])
    roots.append(Path("/usr/local/cuda"))
    roots.extend(sorted(Path("/usr/local").glob("cuda-*")))

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen and root.exists():
            seen.add(key)
            unique.append(root)
    return unique


def _first_existing_include_dir(root: Path) -> Path | None:
    for include_dir in _candidate_include_dirs(root):
        if (include_dir / "cublasLt.h").exists():
            return include_dir
    return None


def _parse_defines(header: Path, names: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in _read_text(header).splitlines():
        match = _DEFINE_RE.match(line)
        if match and match.group(1) in names:
            out[match.group(1)] = int(match.group(2))
    return out


def _parse_cuda_installations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in _cuda_roots():
        include_dir = _first_existing_include_dir(root)
        row: dict[str, Any] = {
            "root": str(root),
            "realpath": str(root.resolve()) if root.exists() else None,
            "include_dir": str(include_dir) if include_dir is not None else None,
            "has_cublasLt_h": include_dir is not None and (include_dir / "cublasLt.h").exists(),
        }
        nvcc = root / "bin" / "nvcc"
        if nvcc.exists():
            row["nvcc_version"] = _run_text([str(nvcc), "--version"])
        if include_dir is not None:
            cuda_runtime = include_dir / "cuda_runtime_api.h"
            cublas_api = include_dir / "cublas_api.h"
            row.update(_parse_defines(cuda_runtime, {"CUDART_VERSION"}))
            row.update(
                _parse_defines(
                    cublas_api,
                    {"CUBLAS_VER_MAJOR", "CUBLAS_VER_MINOR", "CUBLAS_VER_PATCH", "CUBLAS_VER_BUILD"},
                )
            )
            if {
                "CUBLAS_VER_MAJOR",
                "CUBLAS_VER_MINOR",
                "CUBLAS_VER_PATCH",
            }.issubset(row):
                row["CUBLAS_VERSION"] = (
                    row["CUBLAS_VER_MAJOR"] * 10000
                    + row["CUBLAS_VER_MINOR"] * 100
                    + row["CUBLAS_VER_PATCH"]
                )
        rows.append(row)
    return rows


def _active_include_dir() -> Path | None:
    roots = _cuda_roots()
    for root in roots:
        include_dir = _first_existing_include_dir(root)
        if include_dir is not None:
            return include_dir
    return None


def _active_lib_dir(include_dir: Path | None) -> Path | None:
    if include_dir is None:
        return None
    if include_dir.name == "include" and include_dir.parent.name != "targets":
        cuda_root = include_dir.parent
    else:
        cuda_root = include_dir.parents[2] if "targets" in include_dir.parts else include_dir.parent
    candidates = [
        cuda_root / "lib64",
        cuda_root / "targets" / "sbsa-linux" / "lib",
        cuda_root / "targets" / "x86_64-linux" / "lib",
    ]
    for candidate in candidates:
        if (candidate / "libcublasLt.so").exists():
            return candidate
    return None


def _parse_cublaslt_attrs(include_dir: Path | None) -> dict[str, Any]:
    if include_dir is None:
        return {"error": "no cublasLt.h found"}
    header = include_dir / "cublasLt.h"
    desc_attrs: list[dict[str, Any]] = []
    matrix_scale_modes: list[dict[str, Any]] = []
    implicit_value = 0
    in_matrix_scale_enum = False

    for line in _read_text(header).splitlines():
        match = _DESC_SCALE_RE.match(line)
        if match:
            desc_attrs.append({"name": match.group(1), "value": int(match.group(2))})

        if "typedef enum" in line:
            in_matrix_scale_enum = False
        if "CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F" in line:
            in_matrix_scale_enum = True
            implicit_value = 0
        if in_matrix_scale_enum:
            match = _MATRIX_SCALE_RE.match(line)
            if match:
                value = int(match.group(2)) if match.group(2) is not None else implicit_value
                matrix_scale_modes.append({"name": match.group(1), "value": value})
                implicit_value = value + 1
            if "CUBLASLT_MATMUL_MATRIX_SCALE_END" in line:
                in_matrix_scale_enum = False

    return {
        "header": str(header),
        "desc_scale_attrs": desc_attrs,
        "matrix_scale_modes": matrix_scale_modes,
    }


def _load_extension(include_dir: Path, lib_dir: Path | None, *, verbose: bool) -> Any:
    from torch.utils.cpp_extension import load_inline

    extra_ldflags = ["-lcublasLt", "-lcublas", "-lcudart"]
    if lib_dir is not None:
        extra_ldflags.insert(0, f"-L{lib_dir}")
    return load_inline(
        name="cppmega_cublaslt_mxfp8_descriptor_probe",
        cpp_sources=[_CPP_SOURCE],
        functions=["versions", "run_mxfp8_matmul"],
        extra_cflags=["-O2"],
        extra_include_paths=[str(include_dir)],
        extra_ldflags=extra_ldflags,
        verbose=verbose,
    )


def _rel_l2(out: Any, ref: Any) -> float:
    return float(((out.float() - ref.float()).norm() / ref.float().norm()).item())


def _max_abs(out: Any, ref: Any) -> float:
    return float((out.float() - ref.float()).abs().max().item())


def _clean_ext_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "out"}


def _classify(row: dict[str, Any], *, rel_l2_limit: float) -> str:
    if row.get("matmul_status") == "SUCCESS":
        return "pass" if float(row.get("rel_l2", 0.0)) <= rel_l2_limit else "bad_math"
    if row.get("returned_results") == 0:
        return "no_algorithm"
    if row.get("heuristic_status") not in (None, "SUCCESS"):
        return "heuristic_fail"
    if row.get("matmul_status"):
        return "matmul_fail"
    return "fail"


def _run_gemm_probe(args: argparse.Namespace, include_dir: Path, lib_dir: Path | None) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        if args.require_gemm:
            raise RuntimeError("CUDA is required for GEMM probe")
        return {"status": "skip", "reason": "torch.cuda.is_available() is false"}
    if not hasattr(torch, "float8_e4m3fn"):
        if args.require_gemm:
            raise RuntimeError("torch.float8_e4m3fn is required for GEMM probe")
        return {"status": "skip", "reason": "torch.float8_e4m3fn is unavailable"}

    ext = _load_extension(include_dir, lib_dir, verbose=args.verbose_build)

    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    m, n, k = args.m, args.n, args.k
    a = torch.randn((m, k), device=device, dtype=torch.float32).clamp(-args.clamp, args.clamp)
    b = torch.randn((k, n), device=device, dtype=torch.float32).clamp(-args.clamp, args.clamp)
    a_fp8 = a.to(torch.float8_e4m3fn).contiguous()
    b_fp8 = b.to(torch.float8_e4m3fn).contiguous()
    b_phys_t = b_fp8.t().contiguous()
    a_phys_t = a_fp8.t().contiguous()
    a_u8 = a_fp8.view(torch.uint8)
    b_u8 = b_fp8.view(torch.uint8)
    b_phys_t_u8 = b_phys_t.view(torch.uint8)
    a_phys_t_u8 = a_phys_t.view(torch.uint8)

    scale_elems = max(args.scale_elems, 1 << 20)
    scale = torch.full((scale_elems,), args.scale_byte, device=device, dtype=torch.uint8)
    ref = a_fp8.to(torch.float32) @ b_fp8.to(torch.float32)
    workspace_bytes = args.workspace_mib * 1024 * 1024

    def run_variant(
        name: str,
        a_data: Any,
        b_data: Any,
        *,
        transa: bool,
        transb: bool,
        a_shape: tuple[int, int],
        b_shape: tuple[int, int],
        intent: str,
    ) -> dict[str, Any]:
        row = dict(
            ext.run_mxfp8_matmul(
                a_data,
                scale,
                b_data,
                scale,
                m,
                n,
                k,
                transa,
                transb,
                a_shape[0],
                a_shape[1],
                a_shape[1],
                b_shape[0],
                b_shape[1],
                b_shape[1],
                workspace_bytes,
            )
        )
        out = row.get("out")
        cleaned = _clean_ext_row(row)
        cleaned["name"] = name
        cleaned["intent"] = intent
        cleaned["a_physical_shape"] = list(a_shape)
        cleaned["b_physical_shape"] = list(b_shape)
        if cleaned.get("matmul_status") == "SUCCESS" and out is not None:
            torch.cuda.synchronize()
            cleaned["max_abs"] = _max_abs(out, ref)
            cleaned["rel_l2"] = _rel_l2(out, ref)
        cleaned["status"] = _classify(cleaned, rel_l2_limit=args.rel_l2_limit)
        return cleaned

    variants = [
        run_variant(
            "nn_original_payloads",
            a_u8,
            b_u8,
            transa=False,
            transb=False,
            a_shape=(m, k),
            b_shape=(k, n),
            intent="direct row-major A@B; useful when cuBLASLt supports NN block-scaled FP8",
        ),
        run_variant(
            "nt_physical_transpose_b",
            a_u8,
            b_phys_t_u8,
            transa=False,
            transb=True,
            a_shape=(m, k),
            b_shape=(n, k),
            intent="control: physically transpose B payload, then use descriptor transB=T",
        ),
        run_variant(
            "nt_descriptor_only_b_original_payload_fixed_scale",
            a_u8,
            b_u8,
            transa=False,
            transb=True,
            a_shape=(m, k),
            b_shape=(n, k),
            intent="descriptor-only: pretend original B payload is already B.T; scale layout unchanged",
        ),
        run_variant(
            "tn_physical_transpose_a",
            a_phys_t_u8,
            b_u8,
            transa=True,
            transb=False,
            a_shape=(k, m),
            b_shape=(k, n),
            intent="control for A-side transpose if local cuBLASLt supports TN",
        ),
        run_variant(
            "tn_descriptor_only_a_original_payload_fixed_scale",
            a_u8,
            b_u8,
            transa=True,
            transb=False,
            a_shape=(k, m),
            b_shape=(k, n),
            intent="descriptor-only: pretend original A payload is already A.T; scale layout unchanged",
        ),
    ]

    evidence = [
        row
        for row in variants
        if row["name"].endswith("_original_payload_fixed_scale")
        and row["status"] in {"bad_math", "no_algorithm", "heuristic_fail", "matmul_fail"}
    ]
    return {
        "status": "pass" if evidence else "inconclusive",
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "shape": {"m": m, "n": n, "k": k},
        "scale": {
            "mode": "CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0",
            "constant_e8m0_byte": args.scale_byte,
            "elements_allocated": scale_elems,
        },
        "runtime_versions": dict(ext.versions()),
        "variants": variants,
        "descriptor_only_failure_evidence": [row["name"] for row in evidence],
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.m % 32 or args.n % 32 or args.k % 32:
        raise SystemExit("--m, --n, and --k must be multiples of 32 for MXFP8/VEC32 probing")
    if args.scale_byte < 0 or args.scale_byte > 255:
        raise SystemExit("--scale-byte must be in [0, 255]")
    if args.workspace_mib < 0:
        raise SystemExit("--workspace-mib must be non-negative")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--clamp", type=float, default=2.0)
    parser.add_argument("--scale-byte", type=int, default=127)
    parser.add_argument("--scale-elems", type=int, default=1 << 20)
    parser.add_argument("--workspace-mib", type=int, default=32)
    parser.add_argument("--rel-l2-limit", type=float, default=0.05)
    parser.add_argument("--skip-gemm", action="store_true")
    parser.add_argument("--require-gemm", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    _validate_args(args)

    include_dir = _active_include_dir()
    lib_dir = _active_lib_dir(include_dir)
    report: dict[str, Any] = {
        "cuda_installations": _parse_cuda_installations(),
        "active_include_dir": str(include_dir) if include_dir is not None else None,
        "active_lib_dir": str(lib_dir) if lib_dir is not None else None,
        "cublaslt_header_attrs": _parse_cublaslt_attrs(include_dir),
    }

    if args.skip_gemm:
        report["gemm_probe"] = {"status": "skip", "reason": "skipped by --skip-gemm"}
    elif include_dir is None:
        report["gemm_probe"] = {"status": "skip", "reason": "no cublasLt.h include dir found"}
        if args.require_gemm:
            print(json.dumps(report, indent=2, sort_keys=True))
            raise SystemExit(2)
    else:
        try:
            report["gemm_probe"] = _run_gemm_probe(args, include_dir, lib_dir)
        except Exception as exc:  # pragma: no cover - environment probe
            report["gemm_probe"] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc).splitlines()[0] if str(exc) else repr(exc),
            }
            if args.require_gemm:
                print(json.dumps(report, indent=2, sort_keys=True))
                raise

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_gemm and report.get("gemm_probe", {}).get("status") not in {"pass", "inconclusive"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
