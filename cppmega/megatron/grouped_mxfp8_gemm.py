"""Grouped direct MXFP8 GEMM prototype for GB10 MoE backward paths.

The backend reads TE compact MXFP8 payloads/scales directly:

* dgrad NN reads rowwise ``dy`` and columnwise expert weights.
* wgrad NT reads columnwise ``dy`` and columnwise ``x``.

This is a correctness-first CUDA extension with one launch per grouped
operation.  It deliberately avoids bridge transposes and large hidden copies.
The CUDA kernel is scalar/reference code; replacing it with a grouped
CUTLASS/FlashInfer mainloop is the next performance step.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


_CUDA_EXT: Any | None = None
_CUDA_EXT_ERROR: BaseException | None = None
_EXPECTED_NUM_EXPERTS = 16
_MXFP8_BLOCK = 32


@dataclass(frozen=True)
class GroupedMxfp8MicrobenchConfig:
    """Small probe config for the reference grouped backend."""

    total_tokens: int = 512
    n: int = 128
    k: int = 128
    num_experts: int = _EXPECTED_NUM_EXPERTS
    warmup: int = 5
    iters: int = 20
    seed: int = 123


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _resolve_beta(beta: float | None, accumulate: bool) -> float:
    if accumulate:
        return 1.0 if beta is None else float(beta)
    if beta is None or float(beta) == 0.0:
        return 0.0
    raise ValueError(
        "beta is meaningful only when accumulate=True; "
        f"got beta={beta!r} with accumulate=False"
    )


def _check_uint8_cuda_contiguous(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.uint8:
        raise TypeError(f"{name} must be uint8, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be CUDA")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous; this backend does not make payload copies")


def _prepare_expert_offsets(
    expert_offsets: torch.Tensor | Sequence[int],
    *,
    total_rows: int,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(expert_offsets, torch.Tensor):
        if expert_offsets.dim() != 1:
            raise ValueError("expert_offsets must be 1D")
        offsets_cpu = expert_offsets.detach().to(device="cpu", dtype=torch.int64)
    else:
        offsets_cpu = torch.tensor(list(expert_offsets), dtype=torch.int64)

    expected_len = num_experts + 1
    if int(offsets_cpu.numel()) != expected_len:
        raise ValueError(f"expert_offsets must have {expected_len} entries")
    if int(offsets_cpu[0].item()) != 0:
        raise ValueError("expert_offsets[0] must be 0")
    if int(offsets_cpu[-1].item()) != int(total_rows):
        raise ValueError(
            f"expert_offsets[-1] must equal total rows {total_rows}, "
            f"got {int(offsets_cpu[-1].item())}"
        )
    deltas = offsets_cpu[1:] - offsets_cpu[:-1]
    if bool((deltas < 0).any().item()):
        raise ValueError("expert_offsets must be nondecreasing")

    if isinstance(expert_offsets, torch.Tensor) and expert_offsets.is_cuda:
        if expert_offsets.dtype == torch.int64 and expert_offsets.device == device and expert_offsets.is_contiguous():
            return expert_offsets
    return offsets_cpu.to(device=device, non_blocking=True)


def _prepare_scale_offsets(
    scale_offsets: torch.Tensor | Sequence[int],
    *,
    scale_rows: int,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(scale_offsets, torch.Tensor):
        if scale_offsets.dim() != 1:
            raise ValueError("scale_offsets must be 1D")
        offsets_cpu = scale_offsets.detach().to(device="cpu", dtype=torch.int64)
    else:
        offsets_cpu = torch.tensor(list(scale_offsets), dtype=torch.int64)

    expected_len = num_experts + 1
    if int(offsets_cpu.numel()) != expected_len:
        raise ValueError(f"scale_offsets must have {expected_len} entries")
    if int(offsets_cpu[0].item()) != 0:
        raise ValueError("scale_offsets[0] must be 0")
    if int(offsets_cpu[-1].item()) != int(scale_rows):
        raise ValueError(
            f"scale_offsets[-1] must equal scale rows {scale_rows}, "
            f"got {int(offsets_cpu[-1].item())}"
        )
    deltas = offsets_cpu[1:] - offsets_cpu[:-1]
    if bool((deltas < 0).any().item()):
        raise ValueError("scale_offsets must be nondecreasing")

    if isinstance(scale_offsets, torch.Tensor) and scale_offsets.is_cuda:
        if scale_offsets.dtype == torch.int64 and scale_offsets.device == device and scale_offsets.is_contiguous():
            return scale_offsets
    return offsets_cpu.to(device=device, non_blocking=True)


def _prepare_out(
    out: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    accumulate: bool,
    beta: float,
) -> tuple[torch.Tensor, bool]:
    if out is None:
        if accumulate and beta != 0.0:
            raise ValueError("out is required when accumulate=True uses a nonzero beta")
        return torch.empty(0, device=device, dtype=torch.bfloat16), False

    if out.dtype != torch.bfloat16:
        raise TypeError(f"grouped MXFP8 backend currently requires BF16 out, got {out.dtype}")
    if out.device != device:
        raise ValueError("out must be on the same CUDA device as the inputs")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if tuple(out.shape) != shape:
        raise ValueError(f"out must have shape {shape}, got {tuple(out.shape)}")
    return out, True


def _validate_16_expert_shape(total_rows: int, n: int, k: int, num_experts: int) -> None:
    if num_experts != _EXPECTED_NUM_EXPERTS:
        raise ValueError(
            f"grouped MXFP8 prototype is restricted to {_EXPECTED_NUM_EXPERTS} experts, "
            f"got {num_experts}"
        )
    if total_rows <= 0 or n <= 0 or k <= 0:
        raise ValueError("total_rows, n, and k must be positive")
    if n % _MXFP8_BLOCK != 0 or k % _MXFP8_BLOCK != 0:
        raise ValueError("grouped MXFP8 prototype requires feature dims divisible by 32")


def is_supported_shape(total_rows: int, n: int, k: int, num_experts: int = _EXPECTED_NUM_EXPERTS) -> bool:
    """Return whether the reference grouped backend accepts the shape."""

    return (
        int(num_experts) == _EXPECTED_NUM_EXPERTS
        and int(total_rows) > 0
        and int(n) > 0
        and int(k) > 0
        and int(n) % _MXFP8_BLOCK == 0
        and int(k) % _MXFP8_BLOCK == 0
    )


def _tensor_attr(item: Any, attr: str, name: str) -> torch.Tensor:
    tensor = getattr(item, attr, None)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} is missing {attr}")
    if tensor.dtype != torch.uint8:
        raise TypeError(f"{name}.{attr} must be uint8, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name}.{attr} must be contiguous")
    return tensor


def _shared_contiguous_view(
    tensors: Sequence[torch.Tensor],
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Return a larger zero-copy view over a list of contiguous storage slices."""

    if len(tensors) == 0:
        raise ValueError(f"{name} must not be empty")
    first = tensors[0]
    storage_ptr = first.untyped_storage().data_ptr()
    device = first.device
    dtype = first.dtype
    expected_offset = int(first.storage_offset())
    for idx, tensor in enumerate(tensors):
        if tensor.device != device:
            raise ValueError(f"{name}[{idx}] is on {tensor.device}, expected {device}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name}[{idx}] has dtype {tensor.dtype}, expected {dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name}[{idx}] must be contiguous")
        if tensor.untyped_storage().data_ptr() != storage_ptr:
            raise ValueError(f"{name}[{idx}] does not share storage with {name}[0]")
        if int(tensor.storage_offset()) != expected_offset:
            raise ValueError(
                f"{name}[{idx}] storage offset {int(tensor.storage_offset())} "
                f"does not match expected contiguous offset {expected_offset}"
            )
        expected_offset += int(tensor.numel())

    total_numel = 1
    for dim in shape:
        total_numel *= int(dim)
    if total_numel != expected_offset - int(first.storage_offset()):
        raise ValueError(f"{name} shape {shape} does not match source numel {total_numel}")
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return first.as_strided(shape, tuple(reversed(stride)))


def _compact_view_from_attr_list(
    items: Sequence[Any],
    attr: str,
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    tensors = [_tensor_attr(item, attr, f"{name}[{idx}]") for idx, item in enumerate(items)]
    return _shared_contiguous_view(tensors, name=f"{name}.{attr}", shape=shape)


def _out_view_from_group_list(
    out: Any,
    *,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(out, (list, tuple)):
        raise ValueError("grouped direct output must be a list/tuple")
    if len(out) != _EXPECTED_NUM_EXPERTS:
        raise ValueError(f"grouped direct wgrad output must have {_EXPECTED_NUM_EXPERTS} tensors")
    tensors = []
    for idx, tensor in enumerate(out):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"out[{idx}] must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"out[{idx}] must be BF16, got {tensor.dtype}")
        tensors.append(tensor)
    return _shared_contiguous_view(tensors, name="out", shape=shape)


def _single_out_tensor(out: Any, *, shape: tuple[int, ...]) -> torch.Tensor | None:
    if not isinstance(out, (list, tuple)) or len(out) != 1:
        raise ValueError("grouped direct dgrad output must be a single-output list")
    tensor = out[0]
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("grouped direct dgrad output must be a torch.Tensor")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"grouped direct dgrad output must be BF16, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError("grouped direct dgrad output must be contiguous")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"grouped direct dgrad output must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor


def _m_splits_from_kwargs(kwargs: dict[str, Any], *, num_groups: int) -> list[int]:
    m_splits = kwargs.get("m_splits")
    if m_splits is None:
        raise ValueError("grouped direct requires m_splits")
    if isinstance(m_splits, torch.Tensor):
        values = [int(v) for v in m_splits.detach().cpu().tolist()]
    else:
        values = [int(v) for v in m_splits]
    if len(values) != num_groups:
        raise ValueError(f"m_splits must have {num_groups} entries, got {len(values)}")
    if any(v < 0 for v in values):
        raise ValueError("m_splits must be non-negative")
    return values


def _offsets_from_splits(m_splits: Sequence[int]) -> list[int]:
    offsets = [0]
    for split in m_splits:
        offsets.append(offsets[-1] + int(split))
    return offsets


def _scale_offsets_from_splits(m_splits: Sequence[int]) -> list[int]:
    offsets = [0]
    for split in m_splits:
        offsets.append(offsets[-1] + _ceil_div(int(split), _MXFP8_BLOCK))
    return offsets


def _out_dtype_from_grouped_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.dtype:
    out_dtype = kwargs.get("out_dtype", None)
    if out_dtype is None and len(args) >= 2 and isinstance(args[1], torch.dtype):
        out_dtype = args[1]
    if out_dtype is None:
        out_dtype = torch.bfloat16
    if out_dtype != torch.bfloat16:
        raise TypeError(f"grouped MXFP8 backend currently requires BF16 out_dtype, got {out_dtype}")
    return out_dtype


def _ensure_no_grouped_epilogue(kwargs: dict[str, Any]) -> None:
    if bool(kwargs.get("use_bias", False)):
        raise ValueError("grouped MXFP8 direct backend does not fuse bias")
    if kwargs.get("gelu", False) or kwargs.get("gelu_in", None) is not None:
        raise ValueError("grouped MXFP8 direct backend does not fuse GELU")


def _ptr_tensor(tensors: Sequence[torch.Tensor], *, device: torch.device, name: str) -> torch.Tensor:
    if len(tensors) != _EXPECTED_NUM_EXPERTS:
        raise ValueError(f"{name} must have {_EXPECTED_NUM_EXPERTS} tensors")
    ptrs = [int(tensor.data_ptr()) for tensor in tensors]
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


def _attrs_from_list(items: Sequence[Any], attr: str, *, name: str) -> list[torch.Tensor]:
    return [_tensor_attr(item, attr, f"{name}[{idx}]") for idx, item in enumerate(items)]


def _check_same_device(tensors: Sequence[torch.Tensor], *, device: torch.device, name: str) -> None:
    for idx, tensor in enumerate(tensors):
        if tensor.device != device:
            raise ValueError(f"{name}[{idx}] is on {tensor.device}, expected {device}")


def _check_dgrad_list_shapes(
    *,
    dy: Sequence[torch.Tensor],
    dy_scale: Sequence[torch.Tensor],
    weight: Sequence[torch.Tensor],
    weight_scale: Sequence[torch.Tensor],
    m_splits: Sequence[int],
) -> tuple[int, int, int]:
    dy0 = dy[0]
    weight0 = weight[0]
    if dy0.dim() != 2 or weight0.dim() != 2:
        raise ValueError("dgrad list operands must be 2D")
    n = int(dy0.shape[1])
    k = int(weight0.shape[1])
    n_blocks = _ceil_div(n, _MXFP8_BLOCK)
    for idx, split in enumerate(m_splits):
        m_i = int(split)
        if tuple(dy[idx].shape) != (m_i, n):
            raise ValueError(f"dy[{idx}] must have shape {(m_i, n)}, got {tuple(dy[idx].shape)}")
        if int(dy_scale[idx].shape[0]) < m_i or int(dy_scale[idx].shape[1]) < n_blocks:
            raise ValueError(
                f"dy_scale[{idx}] is too small for {(m_i, n_blocks)}, got {tuple(dy_scale[idx].shape)}"
            )
        if tuple(weight[idx].shape) != (n, k):
            raise ValueError(f"weight[{idx}] must have shape {(n, k)}, got {tuple(weight[idx].shape)}")
        if tuple(weight_scale[idx].shape) != (n_blocks, k):
            raise ValueError(
                f"weight_scale[{idx}] must have shape {(n_blocks, k)}, got {tuple(weight_scale[idx].shape)}"
            )
    return sum(int(v) for v in m_splits), n, k


def _check_wgrad_list_shapes(
    *,
    dy: Sequence[torch.Tensor],
    dy_scale: Sequence[torch.Tensor],
    x: Sequence[torch.Tensor],
    x_scale: Sequence[torch.Tensor],
    out: Sequence[torch.Tensor],
    m_splits: Sequence[int],
) -> tuple[int, int, int]:
    dy0 = dy[0]
    x0 = x[0]
    if dy0.dim() != 2 or x0.dim() != 2:
        raise ValueError("wgrad list operands must be 2D")
    n = int(dy0.shape[1])
    k = int(x0.shape[1])
    for idx, split in enumerate(m_splits):
        m_i = int(split)
        row_blocks_i = _ceil_div(m_i, _MXFP8_BLOCK)
        if tuple(dy[idx].shape) != (m_i, n):
            raise ValueError(f"dy[{idx}] must have shape {(m_i, n)}, got {tuple(dy[idx].shape)}")
        if tuple(x[idx].shape) != (m_i, k):
            raise ValueError(f"x[{idx}] must have shape {(m_i, k)}, got {tuple(x[idx].shape)}")
        if int(dy_scale[idx].shape[0]) < row_blocks_i or int(dy_scale[idx].shape[1]) < n:
            raise ValueError(
                f"dy_scale[{idx}] is too small for {(row_blocks_i, n)}, got {tuple(dy_scale[idx].shape)}"
            )
        if int(x_scale[idx].shape[0]) < row_blocks_i or int(x_scale[idx].shape[1]) < k:
            raise ValueError(
                f"x_scale[{idx}] is too small for {(row_blocks_i, k)}, got {tuple(x_scale[idx].shape)}"
            )
        if tuple(out[idx].shape) != (n, k):
            raise ValueError(f"out[{idx}] must have shape {(n, k)}, got {tuple(out[idx].shape)}")
    return sum(int(v) for v in m_splits), n, k


def try_grouped_direct(
    A: Sequence[Any],
    B: Sequence[Any],
    out: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[bool, torch.Tensor | str]:
    """Try the zero-transpose grouped MXFP8 backend for TE grouped backward.

    The function is intentionally fail-closed: it only accepts operands whose
    compact payloads/scales can be viewed as one contiguous grouped storage.
    It never calls the rowwise-transpose bridge and never stacks/cats payloads.
    """

    try:
        layout = kwargs.get("layout", "TN")
        if layout not in ("NN", "NT"):
            return False, f"unsupported_layout:{layout}"
        if len(A) != _EXPECTED_NUM_EXPERTS or len(B) != _EXPECTED_NUM_EXPERTS:
            return False, f"expected_{_EXPECTED_NUM_EXPERTS}_experts"
        _out_dtype_from_grouped_args(args, kwargs)
        _ensure_no_grouped_epilogue(kwargs)
        m_splits = _m_splits_from_kwargs(kwargs, num_groups=_EXPECTED_NUM_EXPERTS)
        offsets = _offsets_from_splits(m_splits)
        total_rows = int(offsets[-1])
        if total_rows <= 0:
            return False, "empty_grouped_token_batch"
        accumulate = bool(kwargs.get("accumulate", False))
        alpha = float(kwargs.get("alpha", 1.0))
        beta = kwargs.get("beta", None)

        if layout == "NN":
            dy_items = _attrs_from_list(B, "_rowwise_data", name="dy")
            dy_scale_items = _attrs_from_list(B, "_rowwise_scale_inv", name="dy_scale")
            weight_items = _attrs_from_list(A, "_columnwise_data", name="weight")
            weight_scale_items = _attrs_from_list(A, "_columnwise_scale_inv", name="weight_scale")
            total_rows_checked, n, k = _check_dgrad_list_shapes(
                dy=dy_items,
                dy_scale=dy_scale_items,
                weight=weight_items,
                weight_scale=weight_scale_items,
                m_splits=m_splits,
            )
            if total_rows_checked != total_rows:
                return False, "dgrad_m_splits_total_changed"
            _validate_16_expert_shape(total_rows, n, k, _EXPECTED_NUM_EXPERTS)
            out_tensor = _single_out_tensor(out, shape=(total_rows, k))
            result = dgrad_nn_gemm_list(
                dy_items,
                dy_scale_items,
                weight_items,
                weight_scale_items,
                offsets,
                out=out_tensor,
                accumulate=accumulate,
                alpha=alpha,
                beta=beta,
            )
            return True, result

        if not isinstance(out, (list, tuple)):
            raise ValueError("grouped direct wgrad output must be a list/tuple")
        dy_items = _attrs_from_list(B, "_columnwise_data", name="dy")
        dy_scale_items = _attrs_from_list(B, "_columnwise_scale_inv", name="dy_scale")
        x_items = _attrs_from_list(A, "_columnwise_data", name="x")
        x_scale_items = _attrs_from_list(A, "_columnwise_scale_inv", name="x_scale")
        out_items = list(out)
        for idx, tensor in enumerate(out_items):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"out[{idx}] must be a torch.Tensor")
            if tensor.dtype != torch.bfloat16:
                raise TypeError(f"out[{idx}] must be BF16, got {tensor.dtype}")
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"out[{idx}] must be CUDA contiguous")
        total_rows_checked, n, k = _check_wgrad_list_shapes(
            dy=dy_items,
            dy_scale=dy_scale_items,
            x=x_items,
            x_scale=x_scale_items,
            out=out_items,
            m_splits=m_splits,
        )
        if total_rows_checked != total_rows:
            return False, "wgrad_m_splits_total_changed"
        _validate_16_expert_shape(total_rows, n, k, _EXPECTED_NUM_EXPERTS)
        result = wgrad_nt_gemm_list(
            dy_items,
            dy_scale_items,
            x_items,
            x_scale_items,
            offsets,
            out=out_items,
            accumulate=accumulate,
            alpha=alpha,
            beta=beta,
        )
        try:
            return True, _out_view_from_group_list(out, shape=(_EXPECTED_NUM_EXPERTS, n, k))
        except Exception:
            pass
        return True, result
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


try_grouped_mxfp8_direct = try_grouped_direct


def _load_cuda_ext() -> Any:
    global _CUDA_EXT, _CUDA_EXT_ERROR
    if _CUDA_EXT is not None:
        return _CUDA_EXT
    if _CUDA_EXT_ERROR is not None:
        raise RuntimeError("grouped MXFP8 CUDA extension failed to load") from _CUDA_EXT_ERROR

    try:
        from torch.utils.cpp_extension import load

        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if (major, minor) == (12, 1):
                os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
            elif (major, minor) == (12, 0):
                os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

        src_dir = Path(__file__).resolve().parent / "cuda_ext"
        verbose = os.environ.get("CPPMEGA_VERBOSE_EXT_BUILD", "0") == "1"
        _CUDA_EXT = load(
            name="cppmega_grouped_mxfp8_gemm_cuda",
            sources=[
                str(src_dir / "grouped_mxfp8_gemm.cpp"),
                str(src_dir / "grouped_mxfp8_gemm.cu"),
            ],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            verbose=verbose,
        )
        return _CUDA_EXT
    except BaseException as exc:  # pragma: no cover - build failures are host-specific
        _CUDA_EXT_ERROR = exc
        raise


def dgrad_nn_gemm(
    dy_rowwise_data: torch.Tensor,
    dy_rowwise_scale_inv: torch.Tensor,
    weight_colwise_data: torch.Tensor,
    weight_colwise_scale_inv: torch.Tensor,
    expert_offsets: torch.Tensor | Sequence[int],
    *,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float | None = None,
) -> torch.Tensor:
    """Compute grouped dgrad NN from direct compact MXFP8 operands.

    ``dy_rowwise_*`` describe concatenated routed-token rows ``[T, N]``.
    ``weight_colwise_*`` describe expert weights ``[16, N, K]`` with compact
    columnwise scales ``[16, N/32, K]``.  The returned BF16 tensor is
    ``[T, K]``.
    """

    _check_uint8_cuda_contiguous(dy_rowwise_data, "dy_rowwise_data")
    _check_uint8_cuda_contiguous(dy_rowwise_scale_inv, "dy_rowwise_scale_inv")
    _check_uint8_cuda_contiguous(weight_colwise_data, "weight_colwise_data")
    _check_uint8_cuda_contiguous(weight_colwise_scale_inv, "weight_colwise_scale_inv")
    if dy_rowwise_data.dim() != 2 or dy_rowwise_scale_inv.dim() != 2:
        raise ValueError("dy rowwise payload/scales must be 2D")
    if weight_colwise_data.dim() != 3 or weight_colwise_scale_inv.dim() != 3:
        raise ValueError("weight columnwise payload/scales must be 3D")
    if weight_colwise_data.device != dy_rowwise_data.device or weight_colwise_scale_inv.device != dy_rowwise_data.device:
        raise ValueError("all dgrad tensors must be on the same CUDA device")

    total_rows = int(dy_rowwise_data.shape[0])
    n = int(dy_rowwise_data.shape[1])
    num_experts = int(weight_colwise_data.shape[0])
    k = int(weight_colwise_data.shape[2])
    _validate_16_expert_shape(total_rows, n, k, num_experts)
    if int(weight_colwise_data.shape[1]) != n:
        raise ValueError(
            f"weight_colwise_data must have shape [16, {n}, K], "
            f"got {tuple(weight_colwise_data.shape)}"
        )
    if tuple(dy_rowwise_scale_inv.shape) != (total_rows, _ceil_div(n, _MXFP8_BLOCK)):
        raise ValueError(
            "dy_rowwise_scale_inv must have compact rowwise shape "
            f"({total_rows}, {_ceil_div(n, _MXFP8_BLOCK)}), got {tuple(dy_rowwise_scale_inv.shape)}"
        )
    if tuple(weight_colwise_scale_inv.shape) != (num_experts, _ceil_div(n, _MXFP8_BLOCK), k):
        raise ValueError(
            "weight_colwise_scale_inv must have compact columnwise shape "
            f"({num_experts}, {_ceil_div(n, _MXFP8_BLOCK)}, {k}), "
            f"got {tuple(weight_colwise_scale_inv.shape)}"
        )

    offsets = _prepare_expert_offsets(
        expert_offsets,
        total_rows=total_rows,
        num_experts=num_experts,
        device=dy_rowwise_data.device,
    )
    beta_f = _resolve_beta(beta, bool(accumulate))
    out_arg, use_out = _prepare_out(
        out,
        shape=(total_rows, k),
        device=dy_rowwise_data.device,
        accumulate=bool(accumulate),
        beta=beta_f,
    )

    return _load_cuda_ext().dgrad_nn(
        dy_rowwise_data,
        dy_rowwise_scale_inv,
        weight_colwise_data,
        weight_colwise_scale_inv,
        offsets,
        out_arg,
        use_out,
        bool(accumulate),
        float(alpha),
        beta_f,
    )


def dgrad_nn_gemm_list(
    dy_rowwise_data: Sequence[torch.Tensor],
    dy_rowwise_scale_inv: Sequence[torch.Tensor],
    weight_colwise_data: Sequence[torch.Tensor],
    weight_colwise_scale_inv: Sequence[torch.Tensor],
    expert_offsets: torch.Tensor | Sequence[int],
    *,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float | None = None,
) -> torch.Tensor:
    """Compute grouped dgrad from per-expert compact MXFP8 tensors in one launch."""

    dy = list(dy_rowwise_data)
    dy_scale = list(dy_rowwise_scale_inv)
    weight = list(weight_colwise_data)
    weight_scale = list(weight_colwise_scale_inv)
    if len(dy) != _EXPECTED_NUM_EXPERTS:
        raise ValueError(f"expected {_EXPECTED_NUM_EXPERTS} dy tensors")
    device = dy[0].device
    for name, tensors in (
        ("dy_rowwise_data", dy),
        ("dy_rowwise_scale_inv", dy_scale),
        ("weight_colwise_data", weight),
        ("weight_colwise_scale_inv", weight_scale),
    ):
        _check_same_device(tensors, device=device, name=name)
        for tensor in tensors:
            _check_uint8_cuda_contiguous(tensor, name)

    if isinstance(expert_offsets, torch.Tensor):
        offsets_cpu = expert_offsets.detach().to(device="cpu", dtype=torch.int64)
        offsets_values = [int(v) for v in offsets_cpu.tolist()]
    else:
        offsets_values = [int(v) for v in expert_offsets]
    m_splits = [offsets_values[i + 1] - offsets_values[i] for i in range(_EXPECTED_NUM_EXPERTS)]
    total_rows, n, k = _check_dgrad_list_shapes(
        dy=dy,
        dy_scale=dy_scale,
        weight=weight,
        weight_scale=weight_scale,
        m_splits=m_splits,
    )
    _validate_16_expert_shape(total_rows, n, k, _EXPECTED_NUM_EXPERTS)
    offsets = _prepare_expert_offsets(
        offsets_values,
        total_rows=total_rows,
        num_experts=_EXPECTED_NUM_EXPERTS,
        device=device,
    )
    beta_f = _resolve_beta(beta, bool(accumulate))
    out_arg, use_out = _prepare_out(
        out,
        shape=(total_rows, k),
        device=device,
        accumulate=bool(accumulate),
        beta=beta_f,
    )
    return _load_cuda_ext().dgrad_nn_ptrs(
        _ptr_tensor(dy, device=device, name="dy_rowwise_data"),
        _ptr_tensor(dy_scale, device=device, name="dy_rowwise_scale_inv"),
        _ptr_tensor(weight, device=device, name="weight_colwise_data"),
        _ptr_tensor(weight_scale, device=device, name="weight_colwise_scale_inv"),
        offsets,
        out_arg,
        use_out,
        bool(accumulate),
        float(alpha),
        beta_f,
        total_rows,
        n,
        k,
    )


def wgrad_nt_gemm(
    dy_colwise_data: torch.Tensor,
    dy_colwise_scale_inv: torch.Tensor,
    x_colwise_data: torch.Tensor,
    x_colwise_scale_inv: torch.Tensor,
    expert_offsets: torch.Tensor | Sequence[int],
    *,
    scale_offsets: torch.Tensor | Sequence[int] | None = None,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float | None = None,
) -> torch.Tensor:
    """Compute grouped wgrad NT from direct compact MXFP8 operands.

    ``dy_colwise_*`` and ``x_colwise_*`` describe the same routed-token rows.
    The returned BF16 tensor is ``[16, N, K]``.
    """

    _check_uint8_cuda_contiguous(dy_colwise_data, "dy_colwise_data")
    _check_uint8_cuda_contiguous(dy_colwise_scale_inv, "dy_colwise_scale_inv")
    _check_uint8_cuda_contiguous(x_colwise_data, "x_colwise_data")
    _check_uint8_cuda_contiguous(x_colwise_scale_inv, "x_colwise_scale_inv")
    if dy_colwise_data.dim() != 2 or x_colwise_data.dim() != 2:
        raise ValueError("dy/x columnwise payloads must be 2D")
    if dy_colwise_scale_inv.dim() != 2 or x_colwise_scale_inv.dim() != 2:
        raise ValueError("dy/x columnwise scales must be 2D")
    if x_colwise_data.device != dy_colwise_data.device or dy_colwise_scale_inv.device != dy_colwise_data.device:
        raise ValueError("all wgrad tensors must be on the same CUDA device")
    if x_colwise_scale_inv.device != dy_colwise_data.device:
        raise ValueError("all wgrad tensors must be on the same CUDA device")

    total_rows = int(dy_colwise_data.shape[0])
    n = int(dy_colwise_data.shape[1])
    k = int(x_colwise_data.shape[1])
    num_experts = _EXPECTED_NUM_EXPERTS
    _validate_16_expert_shape(total_rows, n, k, num_experts)
    if int(x_colwise_data.shape[0]) != total_rows:
        raise ValueError(
            f"x_colwise_data must have {total_rows} rows, got {tuple(x_colwise_data.shape)}"
        )
    if scale_offsets is None:
        row_blocks = _ceil_div(total_rows, _MXFP8_BLOCK)
        scale_offsets_tensor = torch.empty(0, dtype=torch.int64, device=dy_colwise_data.device)
        use_scale_offsets = False
    else:
        if isinstance(scale_offsets, torch.Tensor):
            scale_offsets_cpu = scale_offsets.detach().to(device="cpu", dtype=torch.int64)
            scale_offsets_values = [int(v) for v in scale_offsets_cpu.tolist()]
        else:
            scale_offsets_values = [int(v) for v in scale_offsets]
        if len(scale_offsets_values) != num_experts + 1:
            raise ValueError(f"scale_offsets must have {num_experts + 1} entries")
        row_blocks = int(scale_offsets_values[-1])
        scale_offsets_tensor = _prepare_scale_offsets(
            scale_offsets_values,
            scale_rows=row_blocks,
            num_experts=num_experts,
            device=dy_colwise_data.device,
        )
        use_scale_offsets = True
    if tuple(dy_colwise_scale_inv.shape) != (row_blocks, n):
        raise ValueError(
            "dy_colwise_scale_inv must have compact columnwise shape "
            f"({row_blocks}, {n}), got {tuple(dy_colwise_scale_inv.shape)}"
        )
    if tuple(x_colwise_scale_inv.shape) != (row_blocks, k):
        raise ValueError(
            "x_colwise_scale_inv must have compact columnwise shape "
            f"({row_blocks}, {k}), got {tuple(x_colwise_scale_inv.shape)}"
        )

    offsets = _prepare_expert_offsets(
        expert_offsets,
        total_rows=total_rows,
        num_experts=num_experts,
        device=dy_colwise_data.device,
    )
    beta_f = _resolve_beta(beta, bool(accumulate))
    out_arg, use_out = _prepare_out(
        out,
        shape=(num_experts, n, k),
        device=dy_colwise_data.device,
        accumulate=bool(accumulate),
        beta=beta_f,
    )

    return _load_cuda_ext().wgrad_nt(
        dy_colwise_data,
        dy_colwise_scale_inv,
        x_colwise_data,
        x_colwise_scale_inv,
        offsets,
        scale_offsets_tensor,
        use_scale_offsets,
        out_arg,
        use_out,
        bool(accumulate),
        float(alpha),
        beta_f,
    )


def wgrad_nt_gemm_list(
    dy_colwise_data: Sequence[torch.Tensor],
    dy_colwise_scale_inv: Sequence[torch.Tensor],
    x_colwise_data: Sequence[torch.Tensor],
    x_colwise_scale_inv: Sequence[torch.Tensor],
    expert_offsets: torch.Tensor | Sequence[int],
    *,
    out: Sequence[torch.Tensor],
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float | None = None,
) -> list[torch.Tensor]:
    """Compute grouped wgrad from per-expert compact MXFP8 tensors in one launch."""

    dy = list(dy_colwise_data)
    dy_scale = list(dy_colwise_scale_inv)
    x = list(x_colwise_data)
    x_scale = list(x_colwise_scale_inv)
    out_list = list(out)
    if len(dy) != _EXPECTED_NUM_EXPERTS or len(out_list) != _EXPECTED_NUM_EXPERTS:
        raise ValueError(f"expected {_EXPECTED_NUM_EXPERTS} tensors per grouped operand")
    device = dy[0].device
    for name, tensors in (
        ("dy_colwise_data", dy),
        ("dy_colwise_scale_inv", dy_scale),
        ("x_colwise_data", x),
        ("x_colwise_scale_inv", x_scale),
    ):
        _check_same_device(tensors, device=device, name=name)
        for tensor in tensors:
            _check_uint8_cuda_contiguous(tensor, name)
    for idx, tensor in enumerate(out_list):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"out[{idx}] must be a torch.Tensor")
        if tensor.device != device or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError(f"out[{idx}] must be CUDA contiguous BF16 on {device}")

    if isinstance(expert_offsets, torch.Tensor):
        offsets_cpu = expert_offsets.detach().to(device="cpu", dtype=torch.int64)
        offsets_values = [int(v) for v in offsets_cpu.tolist()]
    else:
        offsets_values = [int(v) for v in expert_offsets]
    m_splits = [offsets_values[i + 1] - offsets_values[i] for i in range(_EXPECTED_NUM_EXPERTS)]
    total_rows, n, k = _check_wgrad_list_shapes(
        dy=dy,
        dy_scale=dy_scale,
        x=x,
        x_scale=x_scale,
        out=out_list,
        m_splits=m_splits,
    )
    _validate_16_expert_shape(total_rows, n, k, _EXPECTED_NUM_EXPERTS)
    offsets = _prepare_expert_offsets(
        offsets_values,
        total_rows=total_rows,
        num_experts=_EXPECTED_NUM_EXPERTS,
        device=device,
    )
    beta_f = _resolve_beta(beta, bool(accumulate))
    _load_cuda_ext().wgrad_nt_ptrs(
        _ptr_tensor(dy, device=device, name="dy_colwise_data"),
        _ptr_tensor(dy_scale, device=device, name="dy_colwise_scale_inv"),
        _ptr_tensor(x, device=device, name="x_colwise_data"),
        _ptr_tensor(x_scale, device=device, name="x_colwise_scale_inv"),
        _ptr_tensor(out_list, device=device, name="out"),
        offsets,
        bool(accumulate),
        float(alpha),
        beta_f,
        n,
        k,
    )
    return out_list


def backend_capability_report() -> dict[str, Any]:
    """Return the current grouped direct backend surface and restrictions."""

    return {
        "name": "cppmega_grouped_mxfp8_gemm",
        "kind": "single-launch CUDA reference kernel",
        "num_experts": _EXPECTED_NUM_EXPERTS,
        "apis": {
            "dgrad_nn_gemm": "dy rowwise [T,N] + weight columnwise [16,N,K] -> BF16 [T,K]",
            "wgrad_nt_gemm": "dy columnwise [T,N] + x columnwise [T,K] -> BF16 [16,N,K]",
        },
        "launches_per_call": 1,
        "restrictions": {
            "feature_alignment": "N and K must be divisible by 32",
            "experts": "exactly 16 experts",
            "inputs": "CUDA, contiguous, uint8 compact MXFP8 payloads/scales",
            "output": "BF16 only",
            "epilogue": "alpha/beta only; no bias, GELU, output quantization, or comm overlap",
        },
        "avoids": [
            "columnwise-to-rowwise bridge calls",
            "large transpose payload materialization",
            "Python loop over experts",
        ],
        "todo": [
            "replace scalar reference GEMM with grouped CUTLASS/FlashInfer SM120 mainloop",
            "support production epilogue contracts once a grouped kernel backend exposes them",
        ],
    }


def _uniform_offsets(total_tokens: int, num_experts: int, device: torch.device) -> torch.Tensor:
    base = total_tokens // num_experts
    rem = total_tokens % num_experts
    counts = [base + (1 if i < rem else 0) for i in range(num_experts)]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return torch.tensor(offsets, dtype=torch.int64, device=device)


def _random_fp8_payload(shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    values = torch.tensor([0, 48, 50, 54, 56, 58, 176, 178, 182, 184, 186], dtype=torch.uint8, device=device)
    indices = torch.randint(0, int(values.numel()), shape, device=device)
    return values[indices].contiguous()


def _time_cuda(fn: Any, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iters)


def microbench(config: GroupedMxfp8MicrobenchConfig | None = None) -> dict[str, Any]:
    """Run a small local CUDA timing probe for the reference grouped backend."""

    config = GroupedMxfp8MicrobenchConfig() if config is None else config
    if not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "CUDA unavailable",
            "config": asdict(config),
            "capability": backend_capability_report(),
        }
    if not is_supported_shape(config.total_tokens, config.n, config.k, config.num_experts):
        raise ValueError(f"unsupported grouped MXFP8 microbench config: {config}")

    torch.manual_seed(config.seed)
    device = torch.device("cuda")
    row_blocks = _ceil_div(config.total_tokens, _MXFP8_BLOCK)
    offsets = _uniform_offsets(config.total_tokens, config.num_experts, device)
    scale_rowwise = torch.full(
        (config.total_tokens, _ceil_div(config.n, _MXFP8_BLOCK)),
        127,
        dtype=torch.uint8,
        device=device,
    )
    scale_col_dy = torch.full((row_blocks, config.n), 127, dtype=torch.uint8, device=device)
    scale_col_x = torch.full((row_blocks, config.k), 127, dtype=torch.uint8, device=device)
    scale_weight = torch.full(
        (config.num_experts, _ceil_div(config.n, _MXFP8_BLOCK), config.k),
        127,
        dtype=torch.uint8,
        device=device,
    )
    dy = _random_fp8_payload((config.total_tokens, config.n), device=device)
    x = _random_fp8_payload((config.total_tokens, config.k), device=device)
    weight = _random_fp8_payload((config.num_experts, config.n, config.k), device=device)
    dgrad_out = torch.empty((config.total_tokens, config.k), dtype=torch.bfloat16, device=device)
    wgrad_out = torch.empty((config.num_experts, config.n, config.k), dtype=torch.bfloat16, device=device)

    dgrad_ms = _time_cuda(
        lambda: dgrad_nn_gemm(dy, scale_rowwise, weight, scale_weight, offsets, out=dgrad_out),
        warmup=config.warmup,
        iters=config.iters,
    )
    wgrad_ms = _time_cuda(
        lambda: wgrad_nt_gemm(dy, scale_col_dy, x, scale_col_x, offsets, out=wgrad_out),
        warmup=config.warmup,
        iters=config.iters,
    )
    return {
        "available": True,
        "config": asdict(config),
        "dgrad_ms": dgrad_ms,
        "wgrad_ms": wgrad_ms,
        "capability": backend_capability_report(),
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbench", action="store_true", help="run the CUDA microbench instead of printing capability")
    parser.add_argument("--total-tokens", type=int, default=512)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args(argv)
    if args.microbench:
        report = microbench(
            GroupedMxfp8MicrobenchConfig(
                total_tokens=args.total_tokens,
                n=args.n,
                k=args.k,
                warmup=args.warmup,
                iters=args.iters,
            )
        )
    else:
        report = backend_capability_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual probe entry point
    raise SystemExit(_main())
