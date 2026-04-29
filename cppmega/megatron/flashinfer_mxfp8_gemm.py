"""FlashInfer/CUTLASS MXFP8 GEMM helpers for GB10.

This module keeps Transformer Engine as the owner of MXFP8 payload storage and
only converts TE compact rowwise scale tensors into the 1D ``layout_128x4``
scale layout consumed by FlashInfer's SM120 CUTLASS backend.
"""

from __future__ import annotations

import os
import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import torch
from torch.utils.cpp_extension import load_inline

_SUPPORTED_OUT_DTYPES = (torch.bfloat16, torch.float16)
_RUNNER_ENV = "CPPMEGA_FLASHINFER_MXFP8_RUNNER"
_TACTIC_ENV = "CPPMEGA_FLASHINFER_MXFP8_TACTIC"
_DEFAULT_RUNNER_MODE = "mm_mxfp8"

RunnerMode = Literal["mm_mxfp8", "direct_tactic"]
CompactColumnwiseOp = Literal["dgrad_nn", "wgrad_nt"]
CompactColumnwiseReason = Literal[
    "accepted",
    "missing_rowwise_data",
    "missing_columnwise_data",
    "swizzled_scales",
    "bad_fp8_dtype",
    "bad_payload_dtype",
    "bad_payload_rank",
    "bad_scale_rank",
    "shape_mismatch",
    "unsupported_shape",
    "unsupported_out_dtype",
    "bad_out",
    "flashinfer_mm_mxfp8_no_compact_columnwise",
]

_COMPACT_COLUMNWISE_BACKEND = "cutlass_native_compact_direct"
_PURE_FLASHINFER_COMPACT_REASON: CompactColumnwiseReason = (
    "flashinfer_mm_mxfp8_no_compact_columnwise"
)


@dataclass(frozen=True)
class FlashinferMxfp8RunnerConfig:
    """Runtime selection for FlashInfer MXFP8 GEMM dispatch."""

    mode: RunnerMode = _DEFAULT_RUNNER_MODE
    tactic: int = 0


@dataclass(frozen=True)
class CompactColumnwiseBackendStatus:
    """Feature-detection result for dense compact-columnwise backward GEMMs."""

    op: CompactColumnwiseOp
    accepted: bool
    reason: CompactColumnwiseReason
    detail: str = ""
    backend: str = _COMPACT_COLUMNWISE_BACKEND


class CompactColumnwiseUnsupportedError(RuntimeError):
    """Raised when compact-columnwise dense backward cannot run without a copy."""

    def __init__(self, status: CompactColumnwiseBackendStatus):
        self.status = status
        self.reason = status.reason
        message = (
            f"MXFP8 compact-columnwise {status.op} is unsupported by "
            f"{status.backend}: {status.reason}"
        )
        if status.detail:
            message = f"{message}: {status.detail}"
        super().__init__(message)


def runner_config(
    mode: str | None = None,
    tactic: int | str | None = None,
) -> FlashinferMxfp8RunnerConfig:
    """Parse an explicit FlashInfer MXFP8 runner config."""

    raw_mode = _DEFAULT_RUNNER_MODE if mode is None else str(mode).strip().lower()
    raw_mode = raw_mode.replace("-", "_")
    aliases = {
        "default": "mm_mxfp8",
        "flashinfer": "mm_mxfp8",
        "mm": "mm_mxfp8",
        "direct": "direct_tactic",
        "direct_tactic0": "direct_tactic",
        "direct_runner": "direct_tactic",
    }
    parsed_mode = aliases.get(raw_mode, raw_mode)
    if parsed_mode not in ("mm_mxfp8", "direct_tactic"):
        raise ValueError(
            "FlashInfer MXFP8 runner mode must be one of "
            f"mm_mxfp8,direct_tactic; got {raw_mode!r}"
        )

    raw_tactic = 0 if tactic is None else tactic
    try:
        parsed_tactic = int(raw_tactic)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FlashInfer MXFP8 tactic must be an integer, got {raw_tactic!r}") from exc
    if parsed_tactic < 0:
        raise ValueError(f"FlashInfer MXFP8 tactic must be non-negative, got {parsed_tactic}")
    return FlashinferMxfp8RunnerConfig(mode=parsed_mode, tactic=parsed_tactic)  # type: ignore[arg-type]


_CPP_SOURCE = r"""
#include <torch/extension.h>

void mxfp8_swizzle_rowwise_scale_cuda(
    torch::Tensor rowwise_scale_inv,
    torch::Tensor output_swizzled_scale_inv,
    int64_t rows,
    int64_t cols);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "mxfp8_swizzle_rowwise_scale",
      &mxfp8_swizzle_rowwise_scale_cuda,
      "Convert compact rowwise MXFP8 scales to FlashInfer/CUTLASS layout_128x4");
}
"""


_CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_UINT8(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Byte, #x " must be uint8")

// GEMM-swizzled scale factor index for SM120 TMA 128x4 scale tiles.
// This replicates CUTLASS's TMA swizzle descriptor layout. If CUTLASS
// changes its swizzle pattern for a new SM architecture, this function must be
// updated in lockstep or scale factors will be silently misindexed.
__device__ __forceinline__ int64_t gemm_swizzled_scale_idx(
    const int64_t i,
    const int64_t j,
    const int64_t num_tiles_x) {
  constexpr int64_t tile_dim_x = 4;
  constexpr int64_t tile_dim_y = 128;
  constexpr int64_t tile_size = tile_dim_x * tile_dim_y;
  const int64_t tile_idx_x = j / tile_dim_x;
  const int64_t tile_idx_y = i / tile_dim_y;
  const int64_t idx_in_tile_x = j % tile_dim_x;
  const int64_t idx_in_tile_y = i % tile_dim_y;
  int64_t idx = (tile_idx_y * num_tiles_x + tile_idx_x) * tile_size;
  idx += (idx_in_tile_y % 32) * 16 + (idx_in_tile_y / 32) * 4 + idx_in_tile_x;
  return idx;
}

__global__ void swizzle_rowwise_scale_kernel(
    const uint8_t* __restrict__ rowwise_scale_inv,
    uint8_t* __restrict__ output_swizzled_scale_inv,
    const int64_t rows,
    const int64_t cols,
    const int64_t compact_stride,
    const int64_t total) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t k_blocks = (cols + 31) / 32;
  const int64_t padded_k_blocks = ((k_blocks + 3) / 4) * 4;
  const int64_t row = idx / padded_k_blocks;
  const int64_t k_block = idx - row * padded_k_blocks;
  uint8_t value = 0;
  if (row < rows && k_block < k_blocks) {
    value = rowwise_scale_inv[row * compact_stride + k_block];
  }
  const int64_t num_tiles_x = (cols + 127) / 128;
  const int64_t swizzled_idx = gemm_swizzled_scale_idx(row, k_block, num_tiles_x);
  output_swizzled_scale_inv[swizzled_idx] = value;
}

void mxfp8_swizzle_rowwise_scale_cuda(
    torch::Tensor rowwise_scale_inv,
    torch::Tensor output_swizzled_scale_inv,
    int64_t rows,
    int64_t cols) {
  CHECK_CUDA(rowwise_scale_inv);
  CHECK_CUDA(output_swizzled_scale_inv);
  CHECK_CONTIGUOUS(rowwise_scale_inv);
  CHECK_CONTIGUOUS(output_swizzled_scale_inv);
  CHECK_UINT8(rowwise_scale_inv);
  CHECK_UINT8(output_swizzled_scale_inv);
  TORCH_CHECK(rowwise_scale_inv.dim() == 2, "rowwise_scale_inv must be 2D");
  TORCH_CHECK(output_swizzled_scale_inv.dim() == 1, "output_swizzled_scale_inv must be 1D");
  TORCH_CHECK(rows > 0 && cols > 0, "rows/cols must be positive");
  TORCH_CHECK(rows % 32 == 0, "rows must be divisible by 32");
  TORCH_CHECK(cols % 32 == 0, "cols must be divisible by 32");

  const int64_t k_blocks = (cols + 31) / 32;
  const int64_t padded_rows = ((rows + 127) / 128) * 128;
  const int64_t padded_k_blocks = ((k_blocks + 3) / 4) * 4;
  const int64_t expected = padded_rows * padded_k_blocks;
  TORCH_CHECK(rowwise_scale_inv.size(0) >= rows, "rowwise_scale_inv dim0 too small");
  TORCH_CHECK(rowwise_scale_inv.size(1) >= k_blocks, "rowwise_scale_inv dim1 too small");
  TORCH_CHECK(output_swizzled_scale_inv.numel() == expected, "bad output scale size");

  const c10::cuda::CUDAGuard device_guard(rowwise_scale_inv.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  const int blocks = static_cast<int>((expected + threads - 1) / threads);
  swizzle_rowwise_scale_kernel<<<blocks, threads, 0, stream>>>(
      rowwise_scale_inv.data_ptr<uint8_t>(),
      output_swizzled_scale_inv.data_ptr<uint8_t>(),
      rows,
      cols,
      rowwise_scale_inv.size(1),
      expected);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


@lru_cache(maxsize=1)
def _load_cuda_ext() -> Any:
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if (major, minor) == (12, 1):
                os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
            elif (major, minor) == (12, 0):
                os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
    return load_inline(
        name="cppmega_flashinfer_mxfp8_scale_cuda",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=bool(int(os.environ.get("CPPMEGA_FLASHINFER_MXFP8_EXT_VERBOSE", "0"))),
    )


@lru_cache(maxsize=1)
def _load_flashinfer_mm() -> Any:
    from flashinfer import mm_mxfp8

    return mm_mxfp8


@lru_cache(maxsize=None)
def _load_flashinfer_cutlass_runner(major: int) -> Any:
    from flashinfer.gemm.gemm_base import get_cutlass_mxfp8_gemm_module

    return get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()


def runner_config_from_env() -> FlashinferMxfp8RunnerConfig:
    """Return the launch-profile bridge FlashInfer MXFP8 runner mode.

    Launchers should set these through ``RunProfile.precision`` rather than
    ad-hoc shell state.  This helper exists because the TE monkey patch is still
    imported before Megatron passes a normal Python config object around.
    """

    return runner_config(
        os.environ.get(_RUNNER_ENV, _DEFAULT_RUNNER_MODE),
        os.environ.get(_TACTIC_ENV, "0"),
    )


def _compact_status(
    op: CompactColumnwiseOp,
    accepted: bool,
    reason: CompactColumnwiseReason = "accepted",
    detail: str = "",
    *,
    backend: str = _COMPACT_COLUMNWISE_BACKEND,
) -> CompactColumnwiseBackendStatus:
    return CompactColumnwiseBackendStatus(
        op=op,
        accepted=accepted,
        reason=reason,
        detail=detail,
        backend=backend,
    )


def _unsupported_status(
    op: CompactColumnwiseOp,
    reason: CompactColumnwiseReason,
    detail: str = "",
    *,
    backend: str = _COMPACT_COLUMNWISE_BACKEND,
) -> CompactColumnwiseBackendStatus:
    return _compact_status(op, False, reason, detail, backend=backend)


def _validate_e4m3_fp8_dtype(tensor: Any) -> None:
    fp8_dtype = getattr(tensor, "_fp8_dtype", None)
    try:
        import transformer_engine_torch as tex

        if fp8_dtype != tex.DType.kFloat8E4M3:
            raise ValueError(
                f"FlashInfer MXFP8 backend expects E4M3 payloads, got {fp8_dtype}"
            )
    except ImportError:
        if "E4M3" not in str(fp8_dtype):
            raise ValueError(
                f"FlashInfer MXFP8 backend expects E4M3 payloads, got {fp8_dtype}"
            )


def _fp8_dtype_status(
    op: CompactColumnwiseOp,
    tensor: Any,
    operand_name: str,
) -> CompactColumnwiseBackendStatus | None:
    try:
        _validate_e4m3_fp8_dtype(tensor)
    except ValueError as exc:
        return _unsupported_status(op, "bad_fp8_dtype", f"{operand_name}: {exc}")
    return None


def _rowwise_matrix(tensor: Any) -> tuple[torch.Tensor, torch.Tensor, bool]:
    _validate_e4m3_fp8_dtype(tensor)
    data = getattr(tensor, "_rowwise_data", None)
    scale = getattr(tensor, "_rowwise_scale_inv", None)
    if not isinstance(data, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise ValueError("MXFP8 FlashInfer backend requires rowwise data and scales")
    if data.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise ValueError("MXFP8 FlashInfer backend requires uint8 payloads/scales")
    if data.dim() < 2 or scale.dim() != 2:
        raise ValueError("MXFP8 FlashInfer backend requires matrix-like rowwise data and 2D scales")
    if data.dim() > 2:
        data = data.reshape(-1, data.shape[-1])
    return data, scale, bool(getattr(tensor, "_with_gemm_swizzled_scales", False))


def _columnwise_matrix(tensor: Any) -> tuple[torch.Tensor, torch.Tensor, bool]:
    _validate_e4m3_fp8_dtype(tensor)
    data = getattr(tensor, "_columnwise_data", None)
    scale = getattr(tensor, "_columnwise_scale_inv", None)
    if not isinstance(data, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise ValueError("MXFP8 FlashInfer backend requires columnwise data and scales")
    if data.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise ValueError("MXFP8 FlashInfer backend requires uint8 payloads/scales")
    if data.dim() < 2:
        raise ValueError("MXFP8 FlashInfer backend requires matrix-like columnwise data")
    if scale.dim() != 2:
        raise ValueError("MXFP8 FlashInfer backend requires 2D columnwise scales")
    if data.dim() > 2:
        data = data.reshape(-1, data.shape[-1])
    return data, scale, bool(getattr(tensor, "_with_gemm_swizzled_scales", False))


def _as_fp8_payload(data: torch.Tensor) -> torch.Tensor:
    if data.dtype != torch.uint8:
        raise TypeError(f"MXFP8 payload must be uint8, got {data.dtype}")
    return data.view(torch.float8_e4m3fn)


def _resolve_out_dtype(out: torch.Tensor | None, out_dtype: torch.dtype | None) -> torch.dtype:
    effective_out_dtype = out_dtype
    if effective_out_dtype is None:
        effective_out_dtype = out.dtype if out is not None else torch.bfloat16
    if effective_out_dtype not in _SUPPORTED_OUT_DTYPES:
        raise ValueError(
            "FlashInfer MXFP8 backend requires BF16/FP16 out_dtype, "
            f"got {effective_out_dtype}"
        )
    if out is not None and out.dtype != effective_out_dtype:
        raise ValueError(
            "FlashInfer MXFP8 backend requires out dtype to match out_dtype; "
            f"got out={out.dtype}, out_dtype={effective_out_dtype}"
        )
    return effective_out_dtype


def swizzle_rowwise_scale(
    rowwise_scale_inv: torch.Tensor,
    rows: int,
    cols: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return FlashInfer/CUTLASS ``layout_128x4`` scales for a rowwise matrix."""

    if rowwise_scale_inv.dtype != torch.uint8:
        raise TypeError(f"rowwise_scale_inv must be uint8, got {rowwise_scale_inv.dtype}")
    if rowwise_scale_inv.dim() != 2:
        raise ValueError("rowwise_scale_inv must be 2D")
    rowwise_scale_inv = rowwise_scale_inv.contiguous()
    k_blocks = (cols + 31) // 32
    padded_rows = ((rows + 127) // 128) * 128
    padded_k_blocks = ((k_blocks + 3) // 4) * 4
    expected = padded_rows * padded_k_blocks
    if out is None:
        out = torch.empty((expected,), device=rowwise_scale_inv.device, dtype=torch.uint8)
    elif out.dtype != torch.uint8:
        raise TypeError(f"out must be uint8, got {out.dtype}")
    elif out.device != rowwise_scale_inv.device:
        raise ValueError("out must be on the same device as rowwise_scale_inv")
    elif out.dim() != 1 or out.numel() != expected:
        raise ValueError(f"out must have shape ({expected},), got {tuple(out.shape)}")
    _load_cuda_ext().mxfp8_swizzle_rowwise_scale(rowwise_scale_inv, out, int(rows), int(cols))
    return out


def _scale_for_rowwise_matrix(
    scale: torch.Tensor,
    rows: int,
    cols: int,
    *,
    with_gemm_swizzled_scales: bool,
) -> torch.Tensor:
    if with_gemm_swizzled_scales:
        k_blocks = (cols + 31) // 32
        padded_rows = ((rows + 127) // 128) * 128
        padded_k_blocks = ((k_blocks + 3) // 4) * 4
        expected = padded_rows * padded_k_blocks
        if scale.numel() != expected:
            raise ValueError(
                f"pre-swizzled scale has {scale.numel()} elements, "
                f"expected {expected} (padded_rows={padded_rows}, "
                f"padded_k_blocks={padded_k_blocks})"
            )
        return scale.reshape(-1)
    return swizzle_rowwise_scale(scale, rows, cols)


def normalize_gemm_kwargs(
    *,
    out_dtype: torch.dtype | None,
    out: torch.Tensor | None = None,
    bias: Any,
    gelu: bool,
    gelu_in: Any,
    quantization_params: Any,
    accumulate: bool,
    alpha: float,
    beta: Any,
) -> dict[str, Any]:
    effective_out_dtype = _resolve_out_dtype(out, out_dtype)
    if out is not None:
        if not out.is_contiguous():
            raise ValueError("FlashInfer MXFP8 backend requires contiguous out tensor")
    if bias is not None:
        raise ValueError(
            "FlashInfer MXFP8 backend does not fuse bias/bgrad: "
            "mm_mxfp8 and the SM120 CUTLASS binding expose no bias operand"
        )
    if gelu or gelu_in is not None:
        raise ValueError(
            "FlashInfer MXFP8 backend does not fuse GELU: "
            "mm_mxfp8 only stores the raw GEMM output"
        )
    if quantization_params is not None:
        raise ValueError(
            "FlashInfer MXFP8 backend does not quantize GEMM outputs: "
            "mm_mxfp8 only writes BF16/FP16 tensors"
        )
    if accumulate:
        if beta not in (0, 0.0):
            raise ValueError(
                "FlashInfer MXFP8 backend does not implement accumulate=True "
                "with a C/beta source; only beta=0.0 overwrite is safe"
            )
    if alpha != 1.0:
        raise ValueError(f"FlashInfer MXFP8 backend requires alpha=1.0, got {alpha}")
    if not accumulate and beta not in (None, 0, 0.0):
        raise ValueError(f"FlashInfer MXFP8 backend requires beta unset/0, got {beta}")
    return {"out": out, "out_dtype": effective_out_dtype}


def check_plain_gemm_kwargs(
    *,
    out_dtype: torch.dtype | None,
    bias: Any,
    gelu: bool,
    gelu_in: Any,
    quantization_params: Any,
    accumulate: bool,
    alpha: float,
    beta: Any,
) -> None:
    normalize_gemm_kwargs(
        out_dtype=out_dtype,
        out=None,
        bias=bias,
        gelu=gelu,
        gelu_in=gelu_in,
        quantization_params=quantization_params,
        accumulate=accumulate,
        alpha=alpha,
        beta=beta,
    )


def _has_compact_columnwise_metadata(tensor: Any) -> bool:
    return isinstance(getattr(tensor, "_columnwise_data", None), torch.Tensor) or isinstance(
        getattr(tensor, "_columnwise_scale_inv", None),
        torch.Tensor,
    )


def _is_rowwise_transpose_operand(tensor: Any) -> bool:
    return bool(
        getattr(tensor, "_te_rowwise_transpose_for_backward_operand", False)
        or getattr(tensor, "_cppmega_mxfp8_rowwise_transpose_operand", False)
    )


def _rowwise_matrix_for_compact_status(
    op: CompactColumnwiseOp,
    tensor: Any,
    operand_name: str,
) -> tuple[torch.Tensor, torch.Tensor, CompactColumnwiseBackendStatus | None]:
    dtype_status = _fp8_dtype_status(op, tensor, operand_name)
    if dtype_status is not None:
        return torch.empty(0), torch.empty(0), dtype_status
    if bool(getattr(tensor, "_with_gemm_swizzled_scales", False)):
        return (
            torch.empty(0),
            torch.empty(0),
            _unsupported_status(
                op,
                "swizzled_scales",
                f"{operand_name} uses GEMM-swizzled scales; compact direct needs 2D TE scales",
            ),
        )
    data = getattr(tensor, "_rowwise_data", None)
    scale = getattr(tensor, "_rowwise_scale_inv", None)
    if not isinstance(data, torch.Tensor) or not isinstance(scale, torch.Tensor):
        return (
            torch.empty(0),
            torch.empty(0),
            _unsupported_status(op, "missing_rowwise_data", f"{operand_name} lacks rowwise data/scales"),
        )
    if data.dtype != torch.uint8 or scale.dtype != torch.uint8:
        return (
            torch.empty(0),
            torch.empty(0),
            _unsupported_status(
                op,
                "bad_payload_dtype",
                f"{operand_name} payload/scales must be uint8, got {data.dtype}/{scale.dtype}",
            ),
        )
    if data.dim() < 2:
        return (
            torch.empty(0),
            torch.empty(0),
            _unsupported_status(op, "bad_payload_rank", f"{operand_name} payload must be matrix-like"),
        )
    if scale.dim() != 2:
        return (
            torch.empty(0),
            torch.empty(0),
            _unsupported_status(op, "bad_scale_rank", f"{operand_name} scales must be 2D compact scales"),
        )
    if data.dim() > 2:
        data = data.reshape(-1, data.shape[-1])
    return data, scale, None


def _columnwise_matrix_for_compact_status(
    op: CompactColumnwiseOp,
    tensor: Any,
    operand_name: str,
) -> tuple[torch.Tensor, torch.Tensor, CompactColumnwiseBackendStatus | None]:
    dtype_status = _fp8_dtype_status(op, tensor, operand_name)
    if dtype_status is not None:
        return torch.empty(0), torch.empty(0), dtype_status
    if bool(getattr(tensor, "_with_gemm_swizzled_scales", False)):
        return (
            torch.empty(0),
            torch.empty(0),
            _unsupported_status(
                op,
                "swizzled_scales",
                f"{operand_name} uses GEMM-swizzled scales; compact direct needs 2D TE scales",
            ),
        )
    data = getattr(tensor, "_columnwise_data", None)
    scale = getattr(tensor, "_columnwise_scale_inv", None)
    if not isinstance(data, torch.Tensor) or not isinstance(scale, torch.Tensor):
        return (
            torch.empty(0),
            torch.empty(0),
            _unsupported_status(
                op,
                "missing_columnwise_data",
                f"{operand_name} lacks compact columnwise data/scales",
            ),
        )
    if data.dtype != torch.uint8 or scale.dtype != torch.uint8:
        return (
            torch.empty(0),
            torch.empty(0),
            _unsupported_status(
                op,
                "bad_payload_dtype",
                f"{operand_name} payload/scales must be uint8, got {data.dtype}/{scale.dtype}",
            ),
        )
    if data.dim() < 2:
        return (
            torch.empty(0),
            torch.empty(0),
            _unsupported_status(op, "bad_payload_rank", f"{operand_name} payload must be matrix-like"),
        )
    if scale.dim() != 2:
        return (
            torch.empty(0),
            torch.empty(0),
            _unsupported_status(op, "bad_scale_rank", f"{operand_name} scales must be 2D compact scales"),
        )
    if data.dim() > 2:
        data = data.reshape(-1, data.shape[-1])
    return data, scale, None


def _check_compact_out_status(
    op: CompactColumnwiseOp,
    out: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    rows: int,
    cols: int,
) -> CompactColumnwiseBackendStatus | None:
    try:
        effective_out_dtype = _resolve_out_dtype(out, out_dtype)
    except ValueError as exc:
        return _unsupported_status(op, "unsupported_out_dtype", str(exc))
    if effective_out_dtype != torch.bfloat16:
        return _unsupported_status(
            op,
            "unsupported_out_dtype",
            "compact-columnwise direct backend currently writes BF16 only",
        )
    if out is not None:
        if not out.is_contiguous():
            return _unsupported_status(op, "bad_out", "out must be contiguous")
        if out.numel() < rows * cols:
            return _unsupported_status(op, "bad_out", f"out is too small for {rows}x{cols}")
    return None


def dense_compact_columnwise_dgrad_status(
    weight: Any,
    dy: Any,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> CompactColumnwiseBackendStatus:
    """Return whether dgrad NN can run from original compact columnwise weight."""

    op: CompactColumnwiseOp = "dgrad_nn"
    weight_data, weight_scale, status = _columnwise_matrix_for_compact_status(
        op,
        weight,
        "weight",
    )
    if status is not None:
        return status
    dy_data, dy_scale, status = _rowwise_matrix_for_compact_status(op, dy, "dy")
    if status is not None:
        return status

    m, reduction_n = (int(dy_data.shape[0]), int(dy_data.shape[1]))
    if int(weight_data.shape[0]) != reduction_n:
        return _unsupported_status(
            op,
            "shape_mismatch",
            f"dy is {tuple(dy_data.shape)}, weight columnwise is {tuple(weight_data.shape)}",
        )
    k = int(weight_data.shape[1])
    out_status = _check_compact_out_status(op, out, out_dtype, m, k)
    if out_status is not None:
        return out_status

    from cppmega.megatron import cutlass_mxfp8_gemm as cutlass

    if not cutlass.is_supported_shape(m, k, reduction_n):
        return _unsupported_status(
            op,
            "unsupported_shape",
            f"requires M/N/K multiples of 128, got {m}x{k}x{reduction_n}",
        )
    reduction_blocks = reduction_n // 32
    if int(dy_scale.shape[0]) < m or int(dy_scale.shape[1]) < reduction_blocks:
        return _unsupported_status(
            op,
            "shape_mismatch",
            f"dy scale {tuple(dy_scale.shape)} is too small for {m}x{reduction_n}",
        )
    if int(weight_scale.shape[0]) < reduction_blocks or int(weight_scale.shape[1]) < k:
        return _unsupported_status(
            op,
            "shape_mismatch",
            f"weight columnwise scale {tuple(weight_scale.shape)} is too small for {reduction_n}x{k}",
        )
    return _compact_status(op, True)


def dense_compact_columnwise_wgrad_status(
    x: Any,
    dy: Any,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> CompactColumnwiseBackendStatus:
    """Return whether wgrad NT can run from original compact columnwise operands."""

    op: CompactColumnwiseOp = "wgrad_nt"
    x_data, x_scale, status = _columnwise_matrix_for_compact_status(op, x, "x")
    if status is not None:
        return status
    dy_data, dy_scale, status = _columnwise_matrix_for_compact_status(op, dy, "dy")
    if status is not None:
        return status

    if int(dy_data.shape[0]) != int(x_data.shape[0]):
        return _unsupported_status(
            op,
            "shape_mismatch",
            f"dy columnwise is {tuple(dy_data.shape)}, x columnwise is {tuple(x_data.shape)}",
        )
    m = int(dy_data.shape[1])
    n = int(x_data.shape[1])
    reduction_k = int(dy_data.shape[0])
    out_status = _check_compact_out_status(op, out, out_dtype, m, n)
    if out_status is not None:
        return out_status

    from cppmega.megatron import cutlass_mxfp8_gemm as cutlass

    if not cutlass.is_supported_shape(m, n, reduction_k):
        return _unsupported_status(
            op,
            "unsupported_shape",
            f"requires M/N/K multiples of 128, got {m}x{n}x{reduction_k}",
        )
    reduction_blocks = reduction_k // 32
    if int(dy_scale.shape[0]) < reduction_blocks or int(dy_scale.shape[1]) < m:
        return _unsupported_status(
            op,
            "shape_mismatch",
            f"dy columnwise scale {tuple(dy_scale.shape)} is too small for {reduction_k}x{m}",
        )
    if int(x_scale.shape[0]) < reduction_blocks or int(x_scale.shape[1]) < n:
        return _unsupported_status(
            op,
            "shape_mismatch",
            f"x columnwise scale {tuple(x_scale.shape)} is too small for {reduction_k}x{n}",
        )
    return _compact_status(op, True)


def compact_columnwise_dgrad_status(
    weight: Any,
    dy: Any,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> CompactColumnwiseBackendStatus:
    """Backward-compatible alias for dense compact-columnwise dgrad probing."""

    return dense_compact_columnwise_dgrad_status(weight, dy, out=out, out_dtype=out_dtype)


def compact_columnwise_wgrad_status(
    x: Any,
    dy: Any,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> CompactColumnwiseBackendStatus:
    """Backward-compatible alias for dense compact-columnwise wgrad probing."""

    return dense_compact_columnwise_wgrad_status(x, dy, out=out, out_dtype=out_dtype)


def _raise_if_compact_unsupported(status: CompactColumnwiseBackendStatus) -> None:
    if not status.accepted:
        raise CompactColumnwiseUnsupportedError(status)


def _mm_mxfp8_direct(
    a_data: torch.Tensor,
    b_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    out: torch.Tensor | None,
    out_dtype: torch.dtype,
    config: FlashinferMxfp8RunnerConfig | None = None,
) -> torch.Tensor:
    config = runner_config() if config is None else config
    a = _as_fp8_payload(a_data)
    b = _as_fp8_payload(b_data)
    if out is None:
        out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype)

    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, _get_cache_buf

    major, _minor = torch.cuda.get_device_capability(a.device)
    workspace = _get_cache_buf(
        "cppmega_flashinfer_mxfp8_direct_workspace",
        DEFAULT_WORKSPACE_SIZE,
        a.device,
    )
    runner = _load_flashinfer_cutlass_runner(major)
    return runner.forward(
        [a, b, a_scale, b_scale, out_dtype, out, workspace],
        tactic=config.tactic,
    )


def _mm_mxfp8(
    a_data: torch.Tensor,
    b_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    out: torch.Tensor | None,
    out_dtype: torch.dtype,
    config: FlashinferMxfp8RunnerConfig | None = None,
) -> torch.Tensor:
    config = runner_config_from_env() if config is None else config
    a = _as_fp8_payload(a_data)
    b = _as_fp8_payload(b_data)
    if config.mode == "direct_tactic":
        if out is None:
            out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype)
        return _mm_mxfp8_direct(
            a_data,
            b_data,
            a_scale,
            b_scale,
            out=out,
            out_dtype=out_dtype,
            config=config,
        )
    return _load_flashinfer_mm()(
        a,
        b,
        a_scale,
        b_scale,
        out=out,
        out_dtype=out_dtype,
        use_8x4_sf_layout=False,
        backend="cutlass",
    )


def fprop_tn_gemm(
    weight: Any,
    x: Any,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    config: FlashinferMxfp8RunnerConfig | None = None,
) -> torch.Tensor:
    """Compute TE ``layout='TN'`` fprop: ``x @ weight.T``."""

    x_data, x_scale, x_swizzled = _rowwise_matrix(x)
    w_data, w_scale, w_swizzled = _rowwise_matrix(weight)
    m, k = x_data.shape
    n, wk = w_data.shape
    if wk != k:
        raise ValueError(f"MXFP8 fprop K mismatch: x K={k}, weight K={wk}")
    return _mm_mxfp8(
        x_data,
        w_data.t(),
        _scale_for_rowwise_matrix(x_scale, m, k, with_gemm_swizzled_scales=x_swizzled),
        _scale_for_rowwise_matrix(w_scale, n, k, with_gemm_swizzled_scales=w_swizzled),
        out=out,
        out_dtype=_resolve_out_dtype(out, out_dtype),
        config=config,
    )


def dgrad_nn_gemm_compact_columnwise(
    weight: Any,
    dy: Any,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    config: FlashinferMxfp8RunnerConfig | None = None,
) -> torch.Tensor:
    """Compute dgrad NN from original TE compact columnwise weight metadata.

    This path does not materialize ``weight.T``.  It delegates to cppmega's
    compact-direct CUTLASS backend, which aliases TE columnwise payload/scales
    as the logical transposed B operand.
    """

    del config  # Signature compatibility; compact-direct CUTLASS has no tactic.
    status = dense_compact_columnwise_dgrad_status(
        weight,
        dy,
        out=out,
        out_dtype=out_dtype,
    )
    _raise_if_compact_unsupported(status)
    _resolve_out_dtype(out, out_dtype)
    dy_data, dy_scale, _dy_swizzled = _rowwise_matrix(dy)
    weight_data, weight_scale, _weight_swizzled = _columnwise_matrix(weight)

    from cppmega.megatron import cutlass_mxfp8_gemm as cutlass

    return cutlass.dgrad_nn_gemm(
        dy_data,
        dy_scale,
        weight_data,
        weight_scale,
        out=out,
        # The cooperative auxiliary-load scheduler can deadlock on GB10 once
        # dense Linear dgrad spans more than the first persistent tile wave.
        # The split MK/NK asymmetric mainloop keeps the no-sidecar compact
        # loader path stable for full-model dense shapes.
        asymmetric=True,
    )


def wgrad_nt_gemm_compact_columnwise(
    x: Any,
    dy: Any,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    config: FlashinferMxfp8RunnerConfig | None = None,
) -> torch.Tensor:
    """Compute wgrad NT from original TE compact columnwise x/dy metadata.

    This path does not materialize ``x.T`` or ``dy.T``.  It delegates to
    cppmega's compact-direct CUTLASS backend, which reads both operands through
    the TE columnwise compact representation.
    """

    del config  # Signature compatibility; compact-direct CUTLASS has no tactic.
    status = dense_compact_columnwise_wgrad_status(
        x,
        dy,
        out=out,
        out_dtype=out_dtype,
    )
    _raise_if_compact_unsupported(status)
    _resolve_out_dtype(out, out_dtype)
    x_data, x_scale, _x_swizzled = _columnwise_matrix(x)
    dy_data, dy_scale, _dy_swizzled = _columnwise_matrix(dy)

    from cppmega.megatron import cutlass_mxfp8_gemm as cutlass

    return cutlass.wgrad_nt_gemm(
        dy_data,
        dy_scale,
        x_data,
        x_scale,
        out=out,
        asymmetric=True,
    )


def dgrad_nn_gemm(
    weight_t: Any,
    dy: Any,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    config: FlashinferMxfp8RunnerConfig | None = None,
) -> torch.Tensor:
    """Compute TE backward dgrad from ``weight.T`` sidecar and ``dy``."""

    if _has_compact_columnwise_metadata(weight_t) and not _is_rowwise_transpose_operand(weight_t):
        status = dense_compact_columnwise_dgrad_status(
            weight_t,
            dy,
            out=out,
            out_dtype=out_dtype,
        )
        if status.accepted:
            return dgrad_nn_gemm_compact_columnwise(
                weight_t,
                dy,
                out=out,
                out_dtype=out_dtype,
                config=config,
            )
        raise CompactColumnwiseUnsupportedError(status)

    dy_data, dy_scale, dy_swizzled = _rowwise_matrix(dy)
    wt_data, wt_scale, wt_swizzled = _rowwise_matrix(weight_t)
    m, n = dy_data.shape
    k, wn = wt_data.shape
    if wn != n:
        raise ValueError(f"MXFP8 dgrad N mismatch: dy N={n}, weight.T N={wn}")
    return _mm_mxfp8(
        dy_data,
        wt_data.t(),
        _scale_for_rowwise_matrix(dy_scale, m, n, with_gemm_swizzled_scales=dy_swizzled),
        _scale_for_rowwise_matrix(wt_scale, k, n, with_gemm_swizzled_scales=wt_swizzled),
        out=out,
        out_dtype=_resolve_out_dtype(out, out_dtype),
        config=config,
    )


def wgrad_nt_gemm(
    x_t: Any,
    dy_t: Any,
    *,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    config: FlashinferMxfp8RunnerConfig | None = None,
) -> torch.Tensor:
    """Compute TE backward wgrad from ``dy.T`` and ``x.T`` sidecars."""

    if (
        (_has_compact_columnwise_metadata(x_t) or _has_compact_columnwise_metadata(dy_t))
        and not _is_rowwise_transpose_operand(x_t)
        and not _is_rowwise_transpose_operand(dy_t)
    ):
        status = dense_compact_columnwise_wgrad_status(
            x_t,
            dy_t,
            out=out,
            out_dtype=out_dtype,
        )
        if status.accepted:
            return wgrad_nt_gemm_compact_columnwise(
                x_t,
                dy_t,
                out=out,
                out_dtype=out_dtype,
                config=config,
            )
        raise CompactColumnwiseUnsupportedError(status)

    dy_t_data, dy_t_scale, dy_t_swizzled = _rowwise_matrix(dy_t)
    x_t_data, x_t_scale, x_t_swizzled = _rowwise_matrix(x_t)
    n, m = dy_t_data.shape
    k, xm = x_t_data.shape
    if xm != m:
        raise ValueError(f"MXFP8 wgrad M mismatch: dy.T M={m}, x.T M={xm}")
    return _mm_mxfp8(
        dy_t_data,
        x_t_data.t(),
        _scale_for_rowwise_matrix(dy_t_scale, n, m, with_gemm_swizzled_scales=dy_t_swizzled),
        _scale_for_rowwise_matrix(x_t_scale, k, m, with_gemm_swizzled_scales=x_t_swizzled),
        out=out,
        out_dtype=_resolve_out_dtype(out, out_dtype),
        config=config,
    )


def epilogue_capability_report() -> dict[str, Any]:
    """Return the fused-epilogue surface currently exposed by FlashInfer MXFP8."""

    try:
        from flashinfer import mm_mxfp8

        mm_mxfp8_signature = str(inspect.signature(mm_mxfp8))
    except Exception as exc:  # pragma: no cover - host/package dependent
        mm_mxfp8_signature = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        runner_config = runner_config_from_env()
        runner_config_report: dict[str, Any] | str = {
            "mode": runner_config.mode,
            "tactic": runner_config.tactic,
        }
    except Exception as exc:
        runner_config_report = f"invalid: {type(exc).__name__}: {exc}"

    return {
        "mm_mxfp8_signature": mm_mxfp8_signature,
        "runner_config": runner_config_report,
        "cutlass_runner_inputs": [
            "a",
            "b",
            "a_descale",
            "b_descale",
            "out_dtype",
            "out",
            "workspace_buffer",
        ],
        "sm120_binding": "mxfp8_gemm(mat1, mat2, mat1Scale, mat2Scale, out, workspace_buffer, tactic)",
        "dense_compact_columnwise": {
            "backend": _COMPACT_COLUMNWISE_BACKEND,
            "direct_apis": [
                "dgrad_nn_gemm_compact_columnwise",
                "wgrad_nt_gemm_compact_columnwise",
            ],
            "status_apis": [
                "dense_compact_columnwise_dgrad_status",
                "dense_compact_columnwise_wgrad_status",
            ],
            "flashinfer_mm_mxfp8": {
                "accepted": False,
                "reason": _PURE_FLASHINFER_COMPACT_REASON,
                "detail": "FlashInfer mm_mxfp8 accepts rowwise operands plus layout_128x4 scales, not TE compact columnwise operands.",
            },
        },
        "supported": {
            "preallocated_out": "passes out directly to mm_mxfp8",
            "out_dtype": "BF16 and FP16",
            "accumulate_beta0": "treated as overwrite; no C source is read",
            "dense_compact_columnwise_dgrad_wgrad": (
                "BF16 dgrad NN and wgrad NT through cppmega CUTLASS compact-direct "
                "without materializing x.T/dy.T/weight.T"
            ),
        },
        "unsupported": {
            "accumulate_beta_nonzero": "FlashInfer SM120 MXFP8 instantiates ElementC=void / no beta source",
            "bias": "no bias operand in mm_mxfp8 or mxfp8_gemm binding",
            "gelu": "no activation or GELU input capture in mm_mxfp8",
            "output_quantization": "mm_mxfp8 only writes BF16/FP16 out tensors",
            "comm_overlap": "no ub/ub_type/extra_output/bulk_overlap contract in runner inputs",
            _PURE_FLASHINFER_COMPACT_REASON: (
                "pure FlashInfer mm_mxfp8 has no compact-columnwise operand source; "
                "use the exposed compact-direct CUTLASS APIs or reject explicitly"
            ),
        },
    }
