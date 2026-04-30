"""Opt-in frozen MXFP8 storage for non-TE parameter islands.

Megatron's ``--mxfp8-param-storage`` covers Transformer Engine modules, but
the token embedding, LM head, and cppmega feature embeddings are ordinary
Megatron/PyTorch modules.  This file provides an explicit profiling lane for
those remaining storage islands: quantize their parameters to TE MXFP8 tensors
and freeze them so no BF16 master parameter is kept alive.

This is intentionally not a training-quality replacement for BF16 embedding or
LM-head weights.  It is a memory upper-bound probe until those modules have a
real trainable MXFP8 optimizer/update path.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn as nn

Mxfp8StorageIslandMode = Literal["off", "frozen_mxfp8"]


@dataclass(frozen=True)
class Mxfp8StorageIslandConfig:
    """Runtime config rendered from the typed run profile."""

    mode: Mxfp8StorageIslandMode = "off"
    embedding: bool = True
    output_layer: bool = True
    ngram_table: bool = True
    ngram_out_proj: bool = False
    structure_table: bool = False
    structure_up_proj: bool = False
    pad_rows: bool = True
    pad_columns: bool = True
    columnwise: bool = False


class Mxfp8StorageIslandResult(NamedTuple):
    path: str
    status: str
    reason: str
    old_shape: tuple[int, ...] | None = None
    new_shape: tuple[int, ...] | None = None
    old_nbytes: int = 0
    new_nbytes: int = 0


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be a boolean value; got {raw!r}")


def mxfp8_storage_island_config_from_env() -> Mxfp8StorageIslandConfig:
    """Build storage-island config from profile-rendered env vars."""

    mode = os.environ.get("CPPMEGA_MXFP8_STORAGE_ISLANDS", "off").strip().lower()
    if mode not in ("off", "frozen_mxfp8"):
        raise ValueError(
            "CPPMEGA_MXFP8_STORAGE_ISLANDS must be off|frozen_mxfp8; "
            f"got {mode!r}"
        )
    return Mxfp8StorageIslandConfig(
        mode=mode,  # type: ignore[arg-type]
        embedding=_env_bool("CPPMEGA_MXFP8_STORAGE_ISLAND_EMBEDDING", True),
        output_layer=_env_bool("CPPMEGA_MXFP8_STORAGE_ISLAND_OUTPUT_LAYER", True),
        ngram_table=_env_bool("CPPMEGA_MXFP8_STORAGE_ISLAND_NGRAM_TABLE", True),
        ngram_out_proj=_env_bool("CPPMEGA_MXFP8_STORAGE_ISLAND_NGRAM_OUT_PROJ", False),
        structure_table=_env_bool("CPPMEGA_MXFP8_STORAGE_ISLAND_STRUCTURE_TABLE", False),
        structure_up_proj=_env_bool("CPPMEGA_MXFP8_STORAGE_ISLAND_STRUCTURE_UP_PROJ", False),
        pad_rows=_env_bool("CPPMEGA_MXFP8_STORAGE_ISLAND_PAD_ROWS", True),
        pad_columns=_env_bool("CPPMEGA_MXFP8_STORAGE_ISLAND_PAD_COLUMNS", True),
        columnwise=_env_bool("CPPMEGA_MXFP8_STORAGE_ISLAND_COLUMNWISE", False),
    )


def selected_mxfp8_storage_island_paths(
    config: Mxfp8StorageIslandConfig,
) -> tuple[str, ...]:
    """Return parameter paths selected by the config."""

    paths: list[str] = []
    if config.embedding:
        paths.append("embedding.word_embeddings.weight")
    if config.output_layer:
        paths.append("output_layer.weight")
    if config.ngram_table:
        paths.append("embedding.cppmega_ngram_hash.unified_table.weight")
    if config.ngram_out_proj:
        paths.append("embedding.cppmega_ngram_hash.out_proj.weight")
    if config.structure_table:
        paths.append("embedding.cppmega_structure.stacked_emb.weight")
    if config.structure_up_proj:
        paths.append("embedding.cppmega_structure.up_proj.weight")
    return tuple(paths)


def is_te_quantized_tensor(tensor: object) -> bool:
    """Return True for TE QuantizedTensor/MXFP8Tensor without hard dependency."""

    try:
        from transformer_engine.pytorch.tensor import QuantizedTensor

        if isinstance(tensor, QuantizedTensor):
            return True
    except Exception:
        pass
    return any(
        getattr(tensor, attr, None) is not None
        for attr in (
            "_rowwise_data",
            "_columnwise_data",
            "_data",
        )
    )


def maybe_dequantize_te_tensor(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a dense tensor for TE quantized tensors, otherwise ``tensor``."""

    if not is_te_quantized_tensor(tensor):
        return tensor
    dequantize = getattr(tensor, "dequantize", None)
    if dequantize is None:
        raise TypeError(f"quantized tensor {type(tensor)!r} does not expose dequantize()")
    kwargs = {"dtype": dtype} if dtype is not None else {}
    return dequantize(**kwargs)


def _rank0_print(message: str, *, stderr: bool = False) -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    print(message, file=sys.stderr if stderr else None, flush=True)


def _get_parent_and_attr(root: nn.Module, path: str) -> tuple[object, str] | None:
    parts = path.split(".")
    parent: object = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            return None
        parent = getattr(parent, part)
        if parent is None:
            return None
    return parent, parts[-1]


def _param_storage_nbytes(param: torch.Tensor) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    for attr in (
        "_rowwise_data",
        "_rowwise_scale_inv",
        "_columnwise_data",
        "_columnwise_scale_inv",
        "_data",
        "_scale_inv",
    ):
        tensor = getattr(param, attr, None)
        if tensor is None:
            continue
        try:
            storage = tensor.untyped_storage()
            key = (int(storage.data_ptr()), int(storage.nbytes()))
            nbytes = int(storage.nbytes())
        except Exception:
            key = (int(tensor.data_ptr()), int(tensor.numel() * tensor.element_size()))
            nbytes = int(tensor.numel() * tensor.element_size())
        if key in seen:
            continue
        seen.add(key)
        total += nbytes
    if total:
        return total
    return int(param.numel() * param.element_size())


def _copy_parameter_attrs(src: torch.nn.Parameter, dst: torch.nn.Parameter) -> None:
    for key, value in getattr(src, "__dict__", {}).items():
        if key in {"grad", "main_grad"}:
            continue
        try:
            setattr(dst, key, value)
        except Exception:
            pass
    dst.is_embedding_or_output_parameter = getattr(
        src, "is_embedding_or_output_parameter", True
    )
    dst._cppmega_mxfp8_frozen_storage = True


def _padded_source_tensor(
    param: torch.nn.Parameter,
    *,
    pad_rows: bool,
    pad_columns: bool,
    allow_column_padding: bool,
) -> tuple[torch.Tensor | None, str, tuple[int, ...]]:
    if is_te_quantized_tensor(param):
        return None, "already_quantized", tuple(param.shape)
    if param.ndim != 2:
        return None, "requires_2d_tensor", tuple(param.shape)
    rows, cols = (int(param.shape[0]), int(param.shape[1]))
    col_pad = (-cols) % 32
    if col_pad and (not pad_columns or not allow_column_padding):
        return None, "last_dim_not_divisible_by_32", tuple(param.shape)
    pad = (-rows) % 32
    if pad and not pad_rows:
        return None, "rows_not_divisible_by_32", tuple(param.shape)

    source = param.detach()
    if source.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        source = source.to(torch.bfloat16)
    if pad:
        padding = torch.zeros(
            (pad, cols),
            device=source.device,
            dtype=source.dtype,
        )
        source = torch.cat((source, padding), dim=0)
    if col_pad:
        padding = torch.zeros(
            (source.shape[0], col_pad),
            device=source.device,
            dtype=source.dtype,
        )
        source = torch.cat((source, padding), dim=1)
    return source.contiguous(), "ok", tuple(source.shape)


def _quantize_param_to_frozen_mxfp8(
    parent: object,
    attr: str,
    path: str,
    *,
    config: Mxfp8StorageIslandConfig,
) -> Mxfp8StorageIslandResult:
    current = getattr(parent, attr, None)
    if not isinstance(current, torch.nn.Parameter):
        return Mxfp8StorageIslandResult(path, "skipped", "not_a_parameter")
    if not current.is_cuda:
        return Mxfp8StorageIslandResult(
            path,
            "skipped",
            "parameter_not_cuda",
            tuple(current.shape),
            tuple(current.shape),
            _param_storage_nbytes(current),
            _param_storage_nbytes(current),
        )

    source, reason, new_shape = _padded_source_tensor(
        current,
        pad_rows=config.pad_rows,
        pad_columns=config.pad_columns,
        allow_column_padding=isinstance(parent, nn.Embedding),
    )
    old_shape = tuple(current.shape)
    old_nbytes = _param_storage_nbytes(current)
    if source is None:
        return Mxfp8StorageIslandResult(
            path, "skipped", reason, old_shape, new_shape, old_nbytes, old_nbytes
        )

    try:
        from transformer_engine.pytorch.tensor import MXFP8Quantizer
        import transformer_engine_torch as tex
    except Exception as exc:
        return Mxfp8StorageIslandResult(
            path,
            "skipped",
            f"te_import_failed:{type(exc).__name__}:{exc}",
            old_shape,
            new_shape,
            old_nbytes,
            old_nbytes,
        )

    quantizer = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=config.columnwise,
    )
    with torch.no_grad():
        q_tensor = quantizer(source.to(torch.bfloat16))
    q_param = torch.nn.Parameter(q_tensor, requires_grad=False)
    _copy_parameter_attrs(current, q_param)
    q_param._cppmega_original_shape = old_shape
    q_param._cppmega_mxfp8_padded_rows = int(new_shape[0] - old_shape[0])

    setattr(parent, attr, q_param)
    if isinstance(parent, nn.Embedding):
        parent.num_embeddings = int(new_shape[0])
        parent.embedding_dim = int(new_shape[1])

    new_nbytes = _param_storage_nbytes(q_param)
    return Mxfp8StorageIslandResult(
        path,
        "converted",
        "frozen_mxfp8",
        old_shape,
        new_shape,
        old_nbytes,
        new_nbytes,
    )


def apply_mxfp8_storage_islands(
    model: nn.Module,
    config: Mxfp8StorageIslandConfig,
) -> tuple[Mxfp8StorageIslandResult, ...]:
    """Convert selected BF16 storage islands to frozen MXFP8 parameters."""

    if config.mode == "off":
        return ()
    if config.mode != "frozen_mxfp8":
        raise ValueError(f"unsupported MXFP8 storage island mode: {config.mode!r}")
    if not torch.cuda.is_available():
        result = Mxfp8StorageIslandResult(
            "*", "skipped", "cuda_not_available", None, None, 0, 0
        )
        _rank0_print("[cppmega] MXFP8 storage islands skipped: CUDA unavailable")
        return (result,)

    results: list[Mxfp8StorageIslandResult] = []
    for path in selected_mxfp8_storage_island_paths(config):
        resolved = _get_parent_and_attr(model, path)
        if resolved is None:
            results.append(Mxfp8StorageIslandResult(path, "skipped", "missing_path"))
            continue
        parent, attr = resolved
        results.append(
            _quantize_param_to_frozen_mxfp8(parent, attr, path, config=config)
        )

    converted = [r for r in results if r.status == "converted"]
    if converted:
        old_total = sum(r.old_nbytes for r in converted)
        new_total = sum(r.new_nbytes for r in converted)
        _rank0_print(
            "[cppmega] MXFP8 frozen storage islands converted "
            f"{len(converted)} params: {old_total / (1024**3):.3f} GiB -> "
            f"{new_total / (1024**3):.3f} GiB"
        )
    skipped = [r for r in results if r.status != "converted"]
    if skipped:
        details = ", ".join(f"{r.path}:{r.reason}" for r in skipped)
        _rank0_print(
            f"[cppmega] MXFP8 frozen storage islands skipped: {details}",
            stderr=True,
        )
    return tuple(results)


def apply_mxfp8_storage_islands_from_env(
    model: nn.Module,
) -> tuple[Mxfp8StorageIslandResult, ...]:
    """Apply storage-island conversion using profile-rendered env vars."""

    config = mxfp8_storage_island_config_from_env()
    return apply_mxfp8_storage_islands(model, config)
