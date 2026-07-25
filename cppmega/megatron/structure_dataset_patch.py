"""Dynamic monkey-patching for Megatron-LM dataset structure ingress.

Dynamically overrides GPTDataset.__getitem__, Megatron's TP batch bridge, and
MambaModel/GPTModel forward passes to stream token-aligned binary MMap metadata
columns through coalesced tensor-parallel transport.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
import warnings
from collections.abc import Mapping
from functools import wraps
from math import prod
from typing import Dict, Any, Optional

import torch  # type: ignore[import-not-found]
import numpy as np

from cppmega.megatron.domain_route_contract import (
    CASE5_RECEIPT_KEY,
    CASE5_SCHEMA_VERSION,
    DOMAIN_SCHEMA_SHA256,  # noqa: F401 - compatibility export for loader callers
    GRAPH_ROUTE_COLUMNS,
    GRAPH_ROUTE_COORDINATE_SPACES,
    SOURCE_IDENTITY_REGISTRY_SCHEMA,
    TOKENIZER_CONTRACT_SHA256,  # noqa: F401 - compatibility export for loader callers
    is_accepted_case5_contract_hash_triple,
)
from cppmega.megatron.objective_contract import OBJECTIVE_IDS

# Thread-local storage to safely pass the current batch's structure inputs to model forward
_local_storage = threading.local()


def _env_flag(
    name: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environment is None else environment
    raw = source.get(name, "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} has invalid boolean value {raw!r}")


def _graph_capacity(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise RuntimeError(
            f"[cppmega-patch] {name} is required when graph routes are enabled; "
            "derive it from the bound CSR sidecars"
        )
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"[cppmega-patch] {name} must be an integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise RuntimeError(f"[cppmega-patch] {name} must be positive, got {value}")
    return value


def _set_current_structure_batch(batch: Dict[str, torch.Tensor] | None) -> None:
    _local_storage.current_structure_batch = batch


def _get_current_structure_batch() -> Dict[str, torch.Tensor] | None:
    return getattr(_local_storage, "current_structure_batch", None)


_TOKEN_BATCH_COLS = (
    "domain_ids",
    "role_ids",
    "entity_ids",
    "scope_ids",
    "source_doc_ids",
    "source_identity_ids",
    "confidence_ids",
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
    "platform_ids",
    "symbol_ids",
    "call_targets",
    "type_refs",
    "def_use",
    "change_mask_pre",
    "change_mask_post",
)

_REQUIRED_STRUCTURE_TOKEN_COLS = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)

_REQUIRED_DOMAIN_TOKEN_COLS = (
    "domain_ids",
    "role_ids",
    "entity_ids",
    "scope_ids",
    "source_doc_ids",
    "source_identity_ids",
    "confidence_ids",
)

_GRAPH_BATCH_COLS = (
    "graph_call_edges",
    "graph_call_edge_counts",
    "graph_type_edges",
    "graph_type_edge_counts",
    "graph_domain_edges",
    "graph_domain_edge_counts",
    "graph_build_edges",
    "graph_build_edge_counts",
    "graph_shell_edges",
    "graph_shell_edge_counts",
    "graph_diagnostic_edges",
    "graph_diagnostic_edge_counts",
    "graph_cross_domain_edges",
    "graph_cross_domain_edge_counts",
    "graph_chunk_starts",
    "graph_chunk_ends",
    "graph_chunk_kinds",
    "graph_chunk_dep_levels",
    "graph_chunk_counts",
    "graph_document_ids",
)

_OBJECTIVE_BATCH_COLS = ("objective_ids",)
_OBJECTIVE_ID_TO_TASK = {value: task for task, value in OBJECTIVE_IDS.items()}

_TOKEN_COL_ALIASES = {
    "domain_ids": ("token_domain_ids", "domain_ids"),
    "role_ids": ("token_role_ids", "role_ids"),
    "entity_ids": ("token_entity_ids", "entity_ids"),
    "scope_ids": ("token_scope_ids", "scope_ids"),
    "source_doc_ids": ("token_source_doc_ids", "source_doc_ids"),
    "source_identity_ids": (
        "token_source_identity_ids",
        "source_identity_ids",
    ),
    "confidence_ids": ("token_confidence_ids", "confidence_ids"),
    "structure_ids": ("token_structure_ids", "structure_ids"),
    "dep_levels": ("token_dep_levels", "dep_levels"),
    "ast_depth_ids": ("token_ast_depth", "ast_depth_ids", "token_ast_depth_ids"),
    "sibling_index_ids": (
        "token_sibling_index",
        "sibling_index_ids",
        "token_sibling_index_ids",
    ),
    "node_type_ids": (
        "token_ast_node_type",
        "node_type_ids",
        "token_ast_node_type_ids",
    ),
    "platform_ids": ("token_platform_ids", "platform_ids"),
    "symbol_ids": ("token_symbol_ids", "symbol_ids"),
    "call_targets": ("token_call_targets", "call_targets"),
    "type_refs": ("token_type_refs", "type_refs"),
    "def_use": ("token_def_use", "def_use"),
    "change_mask_pre": ("token_change_mask_pre", "change_mask_pre"),
    "change_mask_post": ("token_change_mask_post", "change_mask_post"),
}

_OPAQUE_SYMBOL_ID_COLS = frozenset(("symbol_ids", "call_targets", "type_refs"))
_OPAQUE_SYMBOL_ID_ALIASES = frozenset(
    alias for column in _OPAQUE_SYMBOL_ID_COLS for alias in _TOKEN_COL_ALIASES[column]
)
_SYMBOL_IDENTITY_SCHEMA_VERSION = 3
_OPAQUE_UINT64_ID_COLS = frozenset(
    ("source_identity_ids", "symbol_ids", "call_targets", "type_refs")
)
_OPAQUE_UINT64_ID_ALIASES = frozenset(
    alias for column in _OPAQUE_UINT64_ID_COLS for alias in _TOKEN_COL_ALIASES[column]
)
_CASE5_DOMAIN_ID_ALIASES = {
    column: frozenset(_TOKEN_COL_ALIASES[column])
    for column in (
        "domain_ids",
        "role_ids",
        "entity_ids",
        "scope_ids",
        "source_doc_ids",
        "source_identity_ids",
        "confidence_ids",
    )
}

_LOSS_MASK_ALIASES = ("loss_mask", "token_loss_mask")
_LOSS_MASK_ALIGNMENT = "source_token_predicts_next_v1"

_GRAPH_ROUTE_COLS = GRAPH_ROUTE_COLUMNS

_CPPMEGA_BATCH_COLS = _TOKEN_BATCH_COLS + _GRAPH_BATCH_COLS + _OBJECTIVE_BATCH_COLS
_TP_SIDECAR_MAX_DIMS = 4
_TP_BRIDGE_OK = 0
_TP_BRIDGE_MISSING = 1
_TP_BRIDGE_NOT_TENSOR = 2
_TP_BRIDGE_DTYPE = 3
_TP_BRIDGE_SHAPE = 4
_TP_BRIDGE_DEVICE = 5
_TPBridgeIssue = tuple[int, int, BaseException | None]


def _production_objective_required(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Require objective transport for the training graph path, not eval routes."""

    if _env_flag(
        "CPPMEGA_OBJECTIVE_CONTRACT_REQUIRED",
        environment=environment,
    ):
        return True
    return _env_flag(
        "CPPMEGA_GRAPH_ROUTES_ENABLED",
        environment=environment,
    ) and _env_flag(
        "CPPMEGA_DSA_GRAPH_AUX_ENABLED",
        environment=environment,
    )


def _validate_production_objective_batch(
    batch: Mapping[str, Any],
    structure_batch: Mapping[str, Any],
) -> None:
    """Require every valid token to belong to one materialized objective."""

    missing = sorted({"tokens", "labels", "loss_mask"} - set(batch))
    if missing:
        raise ValueError(
            f"production objective batch is missing core fields: {missing}"
        )
    objective_ids = structure_batch.get("objective_ids")
    if not isinstance(objective_ids, torch.Tensor):
        raise ValueError("production objective batch requires objective_ids tensor")
    tokens = batch["tokens"]
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    if not all(isinstance(value, torch.Tensor) for value in (tokens, labels, loss_mask)):
        raise TypeError("production objective batch core fields must be tensors")
    token_shape = tuple(tokens.shape)
    for name, value in (("labels", labels), ("loss_mask", loss_mask)):
        if tuple(value.shape) != token_shape:
            raise ValueError(
                f"production {name} shape {tuple(value.shape)} != tokens {token_shape}"
            )
    if tuple(objective_ids.shape) != token_shape:
        raise ValueError(
            "production objective_ids shape "
            f"{tuple(objective_ids.shape)} != tokens {token_shape}"
        )
    if objective_ids.is_floating_point() or objective_ids.dtype == torch.bool:
        raise ValueError("production objective_ids must use an integer dtype")
    for name in ("structure_ids", "source_doc_ids", "graph_document_ids"):
        marker = structure_batch.get(name)
        if marker is None:
            continue
        if not isinstance(marker, torch.Tensor):
            raise TypeError(f"production {name} must be a tensor")
        if tuple(marker.shape) != token_shape:
            raise ValueError(
                f"production {name} shape {tuple(marker.shape)} != tokens {token_shape}"
            )
    if not bool(torch.isfinite(loss_mask).all().item()):
        raise ValueError("production loss_mask must be finite")
    if bool(((loss_mask != 0) & (loss_mask != 1)).any().item()):
        raise ValueError("production loss_mask must contain only 0/1 values")

    ids = objective_ids.to(dtype=torch.long)
    known = ids == 0
    for objective_id in _OBJECTIVE_ID_TO_TASK:
        known |= ids == objective_id
    if bool((~known).any().item()):
        unknown = sorted(int(value) for value in torch.unique(ids[~known]).tolist())
        raise ValueError(f"production objective_ids contain unknown objective IDs: {unknown}")

    valid_tokens = tokens.ne(0) | labels.ne(0) | loss_mask.ne(0)
    for name in ("structure_ids", "source_doc_ids", "graph_document_ids"):
        marker = structure_batch.get(name)
        if isinstance(marker, torch.Tensor):
            valid_tokens |= marker.ne(0)
    if bool((valid_tokens & (ids <= 0)).any().item()):
        raise ValueError(
            "production objective_ids must contain a positive objective ID for "
            "every loss token and valid token"
        )


def _required_token_batch_cols() -> set[str]:
    """Return only sidecars consumed by enabled input embeddings."""

    required: set[str] = set(_REQUIRED_STRUCTURE_TOKEN_COLS)
    if os.environ.get("CPPMEGA_DOMAIN_EMBEDDING_ENABLED", "0") == "1":
        required.update(_REQUIRED_DOMAIN_TOKEN_COLS)
    return required


def _expected_sidecar_dtype(col: str) -> torch.dtype:
    if col in _OPAQUE_UINT64_ID_COLS:
        uint64 = getattr(torch, "uint64", None)
        if uint64 is None:
            raise RuntimeError(
                "[cppmega-patch] this PyTorch build has no torch.uint64; "
                f"cannot represent opaque sidecar {col!r} without narrowing"
            )
        return uint64
    return torch.long


def _padded_token_sidecar_tensor(tokens: torch.Tensor, *, col: str) -> torch.Tensor:
    """Build one zero sidecar for a Megatron batch-padding sample."""
    return torch.zeros(
        tokens.shape,
        dtype=_expected_sidecar_dtype(col),
        device=tokens.device,
    )


def _pop_structure_batch(
    batch: Dict[str, torch.Tensor] | None,
) -> Dict[str, torch.Tensor] | None:
    """Remove cppmega sidecar tensors from a Megatron batch and stash them."""
    if batch is None:
        _set_current_structure_batch(None)
        return None
    structure_batch = {
        col: batch.pop(col) for col in _CPPMEGA_BATCH_COLS if col in batch
    }
    production_required = _production_objective_required()
    if production_required and "objective_ids" not in structure_batch:
        raise RuntimeError(
            "[cppmega-patch] production objective contract requires objective_ids "
            "in every Megatron batch"
        )
    if production_required:
        _validate_production_objective_batch(batch, structure_batch)
    if structure_batch:
        receipt_path = os.environ.get("CPPMEGA_H200_BATCH_RECEIPT")
        if receipt_path:
            from cppmega.megatron.h200_preflight import observe_production_batch

            observe_production_batch(
                batch=batch,
                structure_batch=structure_batch,
                receipt_path=receipt_path,
            )
        _set_current_structure_batch(structure_batch)
        return structure_batch
    _set_current_structure_batch(None)
    return None


def _take_cppmega_sidecars(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove cppmega-owned tensors before Megatron filters the core batch."""
    return {col: batch.pop(col) for col in _CPPMEGA_BATCH_COLS if col in batch}


def _batch_transport_device(batch: Dict[str, torch.Tensor] | None) -> torch.device:
    if batch is not None:
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _tp_backend_name(group: Any) -> str:
    try:
        return str(torch.distributed.get_backend(group))
    except Exception:
        return "unknown"


def _uint64_transport_error(
    col: str,
    group: Any,
    *,
    device: torch.device,
    exc: BaseException | None = None,
) -> RuntimeError:
    detail = f"; transport error: {exc}" if exc is not None else ""
    return RuntimeError(
        "[cppmega-patch] tensor-parallel backend "
        f"{_tp_backend_name(group)!r} cannot transport opaque uint64 sidecar "
        f"{col!r} on {device} with torch {torch.__version__}; refusing "
        f"to narrow opaque IDs{detail}"
    )


def _broadcast_tp_tensor(
    tensor: torch.Tensor,
    *,
    col: str,
    broadcast_src_rank: int,
    broadcast_group: Any,
) -> None:
    try:
        torch.distributed.broadcast(
            tensor,
            broadcast_src_rank,
            group=broadcast_group,
        )
    except (RuntimeError, TypeError, NotImplementedError) as exc:
        if tensor.dtype == getattr(torch, "uint64", None):
            raise _uint64_transport_error(
                col,
                broadcast_group,
                device=tensor.device,
                exc=exc,
            ) from exc
        raise RuntimeError(
            "[cppmega-patch] tensor-parallel broadcast failed for cppmega "
            f"sidecar protocol tensor {col!r} with dtype={tensor.dtype}, "
            f"device={tensor.device}, backend={_tp_backend_name(broadcast_group)!r}: {exc}"
        ) from exc


def _payload_columns(present: set[str], *, opaque: bool) -> list[str]:
    return [
        col
        for col in _CPPMEGA_BATCH_COLS
        if col in present and (col in _OPAQUE_UINT64_ID_COLS) == opaque
    ]


def _prepare_source_payloads(
    sidecars: Dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], Dict[bool, torch.Tensor], _TPBridgeIssue | None]:
    if not sidecars:
        return {}, {}, (_TP_BRIDGE_MISSING, -1, None)

    prepared: Dict[str, torch.Tensor] = {}
    for col_index, col in enumerate(_CPPMEGA_BATCH_COLS):
        if col not in sidecars:
            continue
        tensor = sidecars[col]
        if not isinstance(tensor, torch.Tensor):
            return {}, {}, (_TP_BRIDGE_NOT_TENSOR, col_index, None)
        try:
            expected_dtype = _expected_sidecar_dtype(col)
        except RuntimeError as exc:
            return {}, {}, (_TP_BRIDGE_DEVICE, col_index, exc)
        if tensor.dtype != expected_dtype:
            return {}, {}, (_TP_BRIDGE_DTYPE, col_index, None)
        if tensor.ndim > _TP_SIDECAR_MAX_DIMS:
            return {}, {}, (_TP_BRIDGE_SHAPE, col_index, None)
        prepared[col] = tensor

    payloads: Dict[bool, torch.Tensor] = {}
    for opaque in (False, True):
        cols = _payload_columns(set(prepared), opaque=opaque)
        if not cols:
            continue
        try:
            payloads[opaque] = (
                torch.cat([prepared[col].reshape(-1) for col in cols])
                .to(device=device, non_blocking=True)
                .contiguous()
            )
        except (RuntimeError, TypeError, NotImplementedError) as exc:
            col_index = _CPPMEGA_BATCH_COLS.index(cols[0])
            return {}, {}, (_TP_BRIDGE_DEVICE, col_index, exc)
    return prepared, payloads, None


def _bridge_protocol_error(
    code: int,
    col_index: int,
    *,
    device: torch.device,
    broadcast_group: Any,
    cause: BaseException | None = None,
) -> RuntimeError:
    if 0 <= col_index < len(_CPPMEGA_BATCH_COLS):
        col = _CPPMEGA_BATCH_COLS[col_index]
    else:
        col = "<batch>"
    if code == _TP_BRIDGE_MISSING:
        message = (
            "TP source batch contains no cppmega sidecars; the GPTDataset/DataLoader "
            "bridge is not installed"
        )
    elif code == _TP_BRIDGE_NOT_TENSOR:
        message = f"sidecar {col!r} is not a torch.Tensor after DataLoader collation"
    elif code == _TP_BRIDGE_DTYPE:
        message = (
            f"sidecar {col!r} must use {_expected_sidecar_dtype(col)} after "
            "DataLoader collation"
        )
    elif code == _TP_BRIDGE_SHAPE:
        message = (
            f"sidecar {col!r} exceeds the supported {_TP_SIDECAR_MAX_DIMS}-D "
            "TP bridge shape"
        )
    elif code == _TP_BRIDGE_DEVICE and col in _OPAQUE_UINT64_ID_COLS:
        return _uint64_transport_error(
            col,
            broadcast_group,
            device=device,
            exc=cause,
        )
    elif code == _TP_BRIDGE_DEVICE:
        message = f"sidecar {col!r} cannot be moved to TP device {device}"
    else:
        message = f"unknown TP sidecar bridge control code {code} for {col!r}"
    if cause is not None:
        message += f"; transport error: {cause}"
    return RuntimeError(f"[cppmega-patch] {message}")


def _broadcast_cppmega_sidecars(
    source_sidecars: Dict[str, torch.Tensor],
    *,
    batch: Dict[str, torch.Tensor] | None,
    tp_rank: int,
    broadcast_src_rank: int,
    broadcast_group: Any,
    transport_device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """Broadcast sidecars without depending on Megatron retaining custom keys."""
    if broadcast_group is None or (
        hasattr(broadcast_group, "size") and broadcast_group.size() == 1
    ):
        return source_sidecars
    device = transport_device or _batch_transport_device(batch)
    prepared: Dict[str, torch.Tensor] = {}
    payloads: Dict[bool, torch.Tensor] = {}
    issue: _TPBridgeIssue | None = None
    if tp_rank == 0:
        prepared, payloads, issue = _prepare_source_payloads(
            source_sidecars,
            device=device,
        )

    # Row zero is the source preflight result. Remaining rows carry exact shapes.
    header = torch.full(
        (len(_CPPMEGA_BATCH_COLS) + 1, _TP_SIDECAR_MAX_DIMS + 1),
        -1,
        dtype=torch.int64,
        device="cpu" if tp_rank == 0 else device,
    )
    if tp_rank == 0:
        header[0, 0] = _TP_BRIDGE_OK if issue is None else issue[0]
        header[0, 1] = -1 if issue is None else issue[1]
        if issue is None:
            for col_index, col in enumerate(_CPPMEGA_BATCH_COLS, start=1):
                tensor = prepared.get(col)
                if tensor is None:
                    continue
                header[col_index, 0] = tensor.ndim
                if tensor.ndim:
                    header[col_index, 1 : tensor.ndim + 1] = torch.tensor(
                        tensor.shape,
                        dtype=torch.int64,
                    )
        header = header.to(device=device, non_blocking=True)
    _broadcast_tp_tensor(
        header,
        col="<cppmega-sidecar-header>",
        broadcast_src_rank=broadcast_src_rank,
        broadcast_group=broadcast_group,
    )

    header_rows = header.tolist()
    error_code = int(header_rows[0][0])
    error_col_index = int(header_rows[0][1])
    if error_code != _TP_BRIDGE_OK:
        cause = issue[2] if issue is not None else None
        raise _bridge_protocol_error(
            error_code,
            error_col_index,
            device=device,
            broadcast_group=broadcast_group,
            cause=cause,
        )

    shapes: Dict[str, tuple[int, ...]] = {}
    for col, row in zip(_CPPMEGA_BATCH_COLS, header_rows[1:], strict=True):
        ndim = int(row[0])
        if ndim >= 0:
            shapes[col] = tuple(int(value) for value in row[1 : ndim + 1])

    received: Dict[str, torch.Tensor] = {}
    for opaque in (False, True):
        cols = _payload_columns(set(shapes), opaque=opaque)
        if not cols:
            continue
        dtype = _expected_sidecar_dtype(cols[0])
        if tp_rank == 0:
            payload = payloads[opaque]
        else:
            payload_numel = sum(prod(shapes[col]) for col in cols)
            try:
                payload = torch.empty(payload_numel, dtype=dtype, device=device)
            except (RuntimeError, TypeError, NotImplementedError) as exc:
                if opaque:
                    raise _uint64_transport_error(
                        cols[0],
                        broadcast_group,
                        device=device,
                        exc=exc,
                    ) from exc
                raise
        _broadcast_tp_tensor(
            payload,
            col=cols[0],
            broadcast_src_rank=broadcast_src_rank,
            broadcast_group=broadcast_group,
        )
        offset = 0
        for col in cols:
            numel = prod(shapes[col])
            received[col] = payload[offset : offset + numel].view(shapes[col])
            offset += numel
    return received


def _make_get_batch_on_this_tp_rank_bridge(original_get_batch):
    """Wrap the pinned helper while owning cppmega sidecar extraction."""
    signature = inspect.signature(original_get_batch)
    required = {"batch", "tp_rank", "broadcast_src_rank", "broadcast_group"}
    missing = required - set(signature.parameters)

    @wraps(original_get_batch)
    def bridged_get_batch(*args, **kwargs):
        _set_current_structure_batch(None)
        if missing:
            if os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0") == "1":
                raise RuntimeError(
                    "[cppmega-patch] Megatron get_batch_on_this_tp_rank has an "
                    f"unsupported signature missing {sorted(missing)}; cppmega requires "
                    "the STACK.lock core_v0.18.0 batch bridge"
                )
            return original_get_batch(*args, **kwargs)

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        input_batch = bound.arguments["batch"]
        if not isinstance(input_batch, dict):
            raise TypeError(
                "[cppmega-patch] Megatron get_batch_on_this_tp_rank batch must be a dict, "
                f"got {type(input_batch).__name__}"
            )
        if os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0") != "1":
            return original_get_batch(*args, **kwargs)

        tp_rank = int(bound.arguments["tp_rank"])
        source_sidecars = _take_cppmega_sidecars(input_batch)
        if tp_rank == 0 and _production_objective_required():
            if "objective_ids" not in source_sidecars:
                raise RuntimeError(
                    "[cppmega-patch] production objective contract requires the "
                    "document-aligned objective_ids sidecar before TP broadcast"
                )
            _validate_production_objective_batch(input_batch, source_sidecars)
        source_device = _batch_transport_device(input_batch) if tp_rank == 0 else None
        batch = original_get_batch(*args, **kwargs)
        if batch is not None and not isinstance(batch, dict):
            raise TypeError(
                "[cppmega-patch] Megatron get_batch_on_this_tp_rank must return a dict "
                f"or None, got {type(batch).__name__}"
            )
        sidecars = _broadcast_cppmega_sidecars(
            source_sidecars,
            batch=batch,
            tp_rank=tp_rank,
            broadcast_src_rank=int(bound.arguments["broadcast_src_rank"]),
            broadcast_group=bound.arguments["broadcast_group"],
            transport_device=source_device,
        )
        _set_current_structure_batch(sidecars)
        return batch

    return bridged_get_batch


def _sidecar_json_path(dataset: Any) -> str:
    bin_path = getattr(dataset.dataset, "bin_path", None)
    if bin_path is None:
        path_prefix = getattr(dataset.dataset, "path_prefix", None)
        if path_prefix is not None:
            bin_path = f"{path_prefix}.bin"
        else:
            bin_reader = getattr(dataset.dataset, "bin_reader", None)
            if bin_reader is not None:
                bin_path = getattr(bin_reader, "_bin_path", None)

    if bin_path is None:
        raise RuntimeError(
            "[cppmega-patch] bin_path is None, cannot initialize side channels"
        )

    prefix = os.path.splitext(str(bin_path))[0]
    json_path = prefix + ".json"
    if not os.path.exists(json_path):
        json_path = prefix + ".idx.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"[cppmega-patch] sidecar manifest not found next to {bin_path!r}; "
            f"tried {prefix + '.json'!r} and {prefix + '.idx.json'!r}"
        )
    return json_path


def _load_sidecar_manifest(dataset: Any) -> tuple[str, dict[str, Any]]:
    if (
        hasattr(dataset, "_cppmega_sidecar_manifest")
        and dataset._cppmega_sidecar_manifest is not None
    ):
        return dataset._cppmega_sidecar_manifest
    json_path = _sidecar_json_path(dataset)
    with open(json_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    side_paths = sidecar.get("side_channel_paths")
    if isinstance(side_paths, dict) and any(
        alias in side_paths for alias in _LOSS_MASK_ALIASES
    ):
        alignment = sidecar.get("loss_mask_alignment")
        if alignment != _LOSS_MASK_ALIGNMENT:
            raise ValueError(
                f"[cppmega-patch] loss_mask sidecar in {json_path!r} requires "
                f"loss_mask_alignment={_LOSS_MASK_ALIGNMENT!r}; got {alignment!r}"
            )
    objective_contract = sidecar.get("objective_contract")
    objective_materialization = sidecar.get("objective_materialization")
    production_objectives = (
        objective_contract is not None or objective_materialization is not None
    )
    if production_objectives:
        if objective_contract is None or objective_materialization is None:
            raise KeyError(
                f"[cppmega-patch] production objective data in {json_path!r} "
                "requires both objective_contract and objective_materialization"
            )
        if not _env_flag("CPPMEGA_STRUCTURE_ENABLED"):
            raise RuntimeError(
                "production objective data requires CPPMEGA_STRUCTURE_ENABLED=1"
            )
        if not _env_flag("CPPMEGA_GRAPH_ROUTES_ENABLED"):
            raise RuntimeError(
                "production objective data requires CPPMEGA_GRAPH_ROUTES_ENABLED=1"
            )
    if _env_flag("CPPMEGA_GRAPH_ROUTES_ENABLED"):
        if objective_contract is None and _env_flag("CPPMEGA_DSA_GRAPH_AUX_ENABLED"):
            raise KeyError(
                f"[cppmega-patch] objective_contract missing in {json_path!r} "
                "while CPPMEGA_DSA_GRAPH_AUX_ENABLED=1; DSA indexer auxiliary "
                "loss requires pre-materialized objective data"
            )
        if objective_contract is not None:
            from cppmega.megatron.objective_contract import (
                validate_materialized_objective_artifact,
                validate_materialized_objective_contract,
            )

            document_count = sidecar.get("document_count")
            if not isinstance(document_count, int) or document_count < 1:
                raise ValueError(
                    f"[cppmega-patch] document_count must be a positive integer in "
                    f"{json_path!r}"
                )
            validated_objectives = validate_materialized_objective_contract(
                objective_contract,
                base_dir=os.path.dirname(json_path),
                document_count=document_count,
                require_schedule_receipt=True,
            )
            validate_materialized_objective_artifact(
                objective_materialization,
                objective_contract=validated_objectives,
                document_count=document_count,
            )
            from cppmega.megatron.graph_objective_loss import (
                validate_runtime_graph_contract,
            )

            validate_runtime_graph_contract(validated_objectives.payload["graph_auxiliary"])
    dataset._cppmega_sidecar_manifest = (json_path, sidecar)
    return json_path, sidecar


def _lazy_init_objective_ids(dataset: Any) -> np.ndarray | None:
    """Map the document-aligned objective IDs required by production mixing."""

    if hasattr(dataset, "_cppmega_objective_ids"):
        return dataset._cppmega_objective_ids
    if not _env_flag("CPPMEGA_STRUCTURE_ENABLED"):
        return None

    json_path, sidecar = _load_sidecar_manifest(dataset)
    wrapper = sidecar.get("objective_contract")
    if wrapper is None:
        if _env_flag("CPPMEGA_OBJECTIVE_CONTRACT_REQUIRED"):
            raise RuntimeError(
                f"[cppmega-patch] production objective contract missing in "
                f"{json_path!r}"
            )
        dataset._cppmega_objective_ids = None
        return None
    if not isinstance(wrapper, dict):
        raise ValueError(
            f"[cppmega-patch] objective_contract must be an object in {json_path!r}"
        )
    binding = wrapper.get("objective_id_sidecar")
    if not isinstance(binding, dict):
        raise KeyError(
            f"[cppmega-patch] objective_contract.objective_id_sidecar missing in "
            f"{json_path!r}"
        )
    if binding.get("dtype") != "uint8" or binding.get("document_aligned") is not True:
        raise ValueError(
            f"[cppmega-patch] objective ID sidecar binding is not document-aligned "
            f"uint8 in {json_path!r}"
        )
    rel_path = binding.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError(
            f"[cppmega-patch] objective ID sidecar path is invalid in {json_path!r}"
        )
    path = _safe_sidecar_path(
        os.path.dirname(json_path),
        rel_path,
        col="objective_ids",
        field="objective_contract.objective_id_sidecar.path",
        json_path=json_path,
    )
    objective_ids = np.memmap(path, mode="r", dtype=np.uint8)
    document_count = sidecar.get("document_count")
    if not isinstance(document_count, int) or document_count <= 0:
        raise ValueError(
            f"[cppmega-patch] document_count must be positive in {json_path!r}"
        )
    if objective_ids.size != document_count:
        raise ValueError(
            f"[cppmega-patch] objective ID sidecar has {objective_ids.size} entries, "
            f"expected {document_count} in {json_path!r}"
        )
    unknown = sorted(
        int(value)
        for value in np.unique(objective_ids)
        if int(value) not in _OBJECTIVE_ID_TO_TASK
    )
    if unknown:
        raise ValueError(
            f"[cppmega-patch] objective ID sidecar contains unknown IDs {unknown} "
            f"in {json_path!r}"
        )
    dataset._cppmega_objective_ids = objective_ids
    return objective_ids


def _safe_sidecar_path(
    base_dir: str, rel_path: str, *, col: str, field: str, json_path: str
) -> str:
    """Join a manifest-supplied relative sidecar path, refusing to escape base_dir.

    RULE #1 / security: a manifest is data, not code. An absolute path or a ``..``
    traversal in it must fail loud, not silently read outside the dataset dir.
    """
    if os.path.isabs(rel_path):
        raise ValueError(
            f"[cppmega-patch] {field} for {col!r} must be a relative filename, got "
            f"absolute {rel_path!r} in {json_path!r}"
        )
    joined = os.path.normpath(os.path.join(base_dir, rel_path))
    base_norm = os.path.normpath(base_dir)
    if joined != base_norm and not joined.startswith(base_norm + os.sep):
        raise ValueError(
            f"[cppmega-patch] {field} for {col!r} escapes the sidecar dir: "
            f"{rel_path!r} -> {joined!r} (base {base_norm!r}) in {json_path!r}"
        )
    # Symlink-safe containment: a symlink INSIDE base_dir could still point outside
    # it, bypassing the lexical check above. Resolve real paths and re-verify.
    real = os.path.realpath(joined)
    real_base = os.path.realpath(base_dir)
    if real != real_base and not real.startswith(real_base + os.sep):
        raise ValueError(
            f"[cppmega-patch] {field} for {col!r} resolves outside the sidecar dir via a "
            f"symlink: {rel_path!r} -> {real!r} (base {real_base!r}) in {json_path!r}"
        )
    return joined


def _lazy_init_side_channels(dataset: Any) -> Dict[str, Dict[str, Any]]:
    """Load JSON sidecar and initialize numpy.memmap for all defined side-channel columns."""
    if (
        hasattr(dataset, "_side_channels_cache")
        and dataset._side_channels_cache is not None
    ):
        return dataset._side_channels_cache

    dataset._side_channels_cache = {}

    if os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0") != "1":
        return dataset._side_channels_cache

    json_path, sidecar = _load_sidecar_manifest(dataset)

    side_paths = sidecar.get("side_channel_paths")
    if not side_paths or not isinstance(side_paths, dict):
        raise KeyError(
            f"[cppmega-patch] side_channel_paths missing in {json_path!r} while "
            "CPPMEGA_STRUCTURE_ENABLED=1"
        )

    present_symbol_columns = _OPAQUE_SYMBOL_ID_ALIASES & set(side_paths)
    if present_symbol_columns:
        if (
            sidecar.get("symbol_identity_schema_version")
            != _SYMBOL_IDENTITY_SCHEMA_VERSION
        ):
            raise ValueError(
                "[cppmega-patch] semantic symbol sidecars require "
                f"symbol_identity_schema_version={_SYMBOL_IDENTITY_SCHEMA_VERSION} "
                f"in {json_path!r}; got "
                f"{sidecar.get('symbol_identity_schema_version')!r}"
            )
        alias_groups = {
            column: set(_TOKEN_COL_ALIASES[column]) & set(side_paths)
            for column in _OPAQUE_SYMBOL_ID_COLS
        }
        invalid_groups = {
            column: sorted(matches)
            for column, matches in alias_groups.items()
            if len(matches) != 1
        }
        if invalid_groups:
            raise ValueError(
                "[cppmega-patch] semantic symbol sidecars require exactly one alias "
                f"for each identity channel in {json_path!r}; got {invalid_groups}"
            )

    base_dir = os.path.dirname(json_path)
    present_case5_aliases = set(side_paths) & set().union(
        *_CASE5_DOMAIN_ID_ALIASES.values()
    )
    if present_case5_aliases:
        # Only validate columns that have at least one alias present.
        # Old data (pre-2026-07-14) may lack source_doc_ids/source_identity_ids.
        invalid_alias_groups = {
            column: sorted(aliases & set(side_paths))
            for column, aliases in _CASE5_DOMAIN_ID_ALIASES.items()
            if (aliases & set(side_paths)) and len(aliases & set(side_paths)) != 1
        }
        if invalid_alias_groups:
            raise ValueError(
                "[cppmega-patch] CASE5 requires exactly one sidecar alias for "
                f"every domain route column in {json_path!r}; got "
                f"{invalid_alias_groups}"
            )
        case5_version = sidecar.get("case5_schema_version", 0)
        receipt = sidecar.get(CASE5_RECEIPT_KEY)
        if case5_version >= 2 and receipt is None:
            raise ValueError(
                "[cppmega-patch] CASE5 schema v2+ requires "
                f"case5_domain_ingestion_receipt in {json_path!r}"
            )
        if receipt is None and case5_version < 2:
            warnings.warn(
                "[cppmega-patch] CASE5 sidecar in "
                f"{json_path!r} lacks case5_domain_ingestion_receipt "
                "(tolerated for pre-v2 data)",
                stacklevel=2,
            )
        if receipt is not None:
            if not isinstance(receipt, dict) or receipt.get("status") != "success":
                raise ValueError(
                    f"[cppmega-patch] successful {CASE5_RECEIPT_KEY} missing from "
                    f"{json_path!r}"
                )
            if receipt.get("schema") != CASE5_SCHEMA_VERSION or not (
                is_accepted_case5_contract_hash_triple(
                    receipt.get("delimiter_contract_sha256"),
                    receipt.get("domain_schema_sha256"),
                    receipt.get("tokenizer_contract_sha256"),
                )
            ):
                raise ValueError(
                    f"[cppmega-patch] stale CASE5 schema or delimiter receipt in "
                    f"{json_path!r}: {receipt}"
                )
        registry = sidecar.get("source_identity_registry")
        if case5_version >= 2 and registry is None:
            raise ValueError(
                "[cppmega-patch] CASE5 schema v2+ requires "
                f"source_identity_registry in {json_path!r}"
            )
        if registry is None and case5_version < 2:
            warnings.warn(
                "[cppmega-patch] CASE5 sidecar in "
                f"{json_path!r} lacks source_identity_registry "
                "(tolerated for pre-v2 data)",
                stacklevel=2,
            )
        if registry is not None:
            if (
                not isinstance(registry, dict)
                or registry.get("schema") != SOURCE_IDENTITY_REGISTRY_SCHEMA
                or not registry.get("path")
            ):
                raise ValueError(
                    f"[cppmega-patch] CASE5 source identity registry receipt is "
                    f"missing or invalid in {json_path!r}"
                )
            registry_path = _safe_sidecar_path(
                base_dir,
                registry["path"],
                col="source_identity_registry",
                field="path",
                json_path=json_path,
            )
            if not os.path.exists(registry_path):
                raise FileNotFoundError(
                    f"[cppmega-patch] CASE5 source identity registry not found: "
                    f"{registry_path}"
                )
    for col, entry in side_paths.items():
        rel_path = entry.get("path")
        dtype_str = entry.get("dtype", "uint16")
        if col in _OPAQUE_UINT64_ID_ALIASES and dtype_str not in ("uint64", "uint32"):
            raise ValueError(
                f"[cppmega-patch] opaque identity sidecar {col!r} must use "
                f"uint64 or uint32, got {dtype_str!r} in {json_path!r}"
            )
        if not rel_path:
            raise ValueError(
                f"[cppmega-patch] side-channel {col!r} has no path in {json_path!r}"
            )
        path = _safe_sidecar_path(
            base_dir, rel_path, col=col, field="path", json_path=json_path
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[cppmega-patch] side-channel file for {col!r} not found: {path}"
            )
        mmap = np.memmap(path, mode="r", dtype=dtype_str)
        needs_widen = col in _OPAQUE_UINT64_ID_ALIASES and dtype_str == "uint32"
        dataset._side_channels_cache[col] = {
            "mmap": mmap,
            "dtype": np.dtype("uint64") if col in _OPAQUE_UINT64_ID_ALIASES else np.dtype(dtype_str),
            "widen_to_uint64": needs_widen,
        }
        print(
            f"[cppmega-patch] Mapped side-channel {col} from {path} with dtype {dtype_str}",
            flush=True,
        )

    return dataset._side_channels_cache


def _lazy_init_graph_sidecars(dataset: Any) -> Dict[str, Dict[str, Any]]:
    if (
        hasattr(dataset, "_graph_sidecars_cache")
        and dataset._graph_sidecars_cache is not None
    ):
        return dataset._graph_sidecars_cache

    dataset._graph_sidecars_cache = {}
    if os.environ.get("CPPMEGA_GRAPH_ROUTES_ENABLED", "0") != "1":
        return dataset._graph_sidecars_cache

    json_path, sidecar = _load_sidecar_manifest(dataset)
    if sidecar.get("graph_sidecar_schema") != "cppmega_graph_routes_v2":
        raise ValueError(
            f"[cppmega-patch] graph_sidecar_schema must be cppmega_graph_routes_v2 in {json_path!r}; "
            f"got {sidecar.get('graph_sidecar_schema')!r}"
        )
    graph_paths = sidecar.get("graph_sidecar_paths")
    if not graph_paths or not isinstance(graph_paths, dict):
        raise KeyError(
            f"[cppmega-patch] graph_sidecar_paths missing in {json_path!r} while "
            "CPPMEGA_GRAPH_ROUTES_ENABLED=1"
        )

    missing = sorted(set(_GRAPH_ROUTE_COLS) - set(graph_paths))
    if missing:
        raise KeyError(
            f"[cppmega-patch] graph sidecars missing required columns: {missing}"
        )

    document_count = int(
        sidecar.get("document_count", len(dataset.dataset.index.sequence_lengths))
    )
    base_dir = os.path.dirname(json_path)
    for col, entry in graph_paths.items():
        coordinate_space = entry.get("coordinate_space")
        if coordinate_space != GRAPH_ROUTE_COORDINATE_SPACES[col]:
            raise ValueError(
                f"[cppmega-patch] graph sidecar {col!r} coordinate_space "
                f"{coordinate_space!r} != {GRAPH_ROUTE_COORDINATE_SPACES[col]!r} in {json_path!r}"
            )
        offsets_path = _safe_sidecar_path(
            base_dir,
            entry["offsets_path"],
            col=col,
            field="offsets_path",
            json_path=json_path,
        )
        data_path = _safe_sidecar_path(
            base_dir,
            entry["data_path"],
            col=col,
            field="data_path",
            json_path=json_path,
        )
        if not os.path.exists(offsets_path):
            raise FileNotFoundError(
                f"[cppmega-patch] graph offsets file for {col!r} not found: {offsets_path}"
            )
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"[cppmega-patch] graph data file for {col!r} not found: {data_path}"
            )

        offset_dtype = np.dtype(entry.get("offset_dtype", "int64"))
        dtype = np.dtype(entry.get("dtype", "int32"))
        offsets = np.memmap(
            offsets_path, mode="r", dtype=offset_dtype, shape=(document_count + 1,)
        )
        if int(offsets[0]) != 0:
            raise ValueError(
                f"[cppmega-patch] graph offsets for {col!r} must start at 0"
            )
        if np.any(np.diff(offsets) < 0):
            raise ValueError(
                f"[cppmega-patch] graph offsets for {col!r} are not monotonic"
            )
        item_count = int(entry.get("item_count", int(offsets[-1])))
        if int(offsets[-1]) != item_count:
            raise ValueError(
                f"[cppmega-patch] graph offsets for {col!r} end at {int(offsets[-1])}, "
                f"manifest item_count={item_count}"
            )
        shape_tail = tuple(int(x) for x in entry.get("shape_tail", []))
        data_shape = (item_count,) + shape_tail
        if item_count == 0 or os.path.getsize(data_path) == 0:
            data = np.zeros(data_shape, dtype=dtype)
        else:
            data = np.memmap(data_path, mode="r", dtype=dtype, shape=data_shape)
        dataset._graph_sidecars_cache[col] = {
            "offsets": offsets,
            "data": data,
            "dtype": dtype,
            "shape_tail": shape_tail,
            "kind": entry.get("kind"),
            "coordinate_space": coordinate_space,
        }
        print(
            f"[cppmega-patch] Mapped graph sidecar {col} from {data_path} "
            f"with dtype {dtype} shape {data_shape}",
            flush=True,
        )

    return dataset._graph_sidecars_cache


def _ensure_gpt_indexes(dataset: Any) -> None:
    if dataset.shuffle_index is None:
        dataset.shuffle_index = np.load(
            dataset.path_to_shuffle_index, allow_pickle=True, mmap_mode="r"
        )
        dataset.sample_index = np.load(
            dataset.path_to_sample_index, allow_pickle=True, mmap_mode="r"
        )
        dataset.document_index = np.load(
            dataset.path_to_document_index, allow_pickle=True, mmap_mode="r"
        )


def _get_sample_token_spans(dataset: Any, idx: int) -> list[dict[str, int]]:
    """Return spans mapping a Megatron sample window back to source documents."""
    _ensure_gpt_indexes(dataset)

    shuffled_idx = int(dataset.shuffle_index[idx])
    doc_index_beg, doc_index_beg_offset = dataset.sample_index[shuffled_idx]
    doc_index_end, doc_index_end_offset = dataset.sample_index[shuffled_idx + 1]
    doc_index_beg = int(doc_index_beg)
    doc_index_end = int(doc_index_end)
    doc_index_beg_offset = int(doc_index_beg_offset)
    doc_index_end_offset = int(doc_index_end_offset)

    token_itemsize = np.dtype(dataset.dataset.index.dtype).itemsize
    seq_ptrs = dataset.dataset.index.sequence_pointers
    seq_lens = dataset.dataset.index.sequence_lengths
    docmap = dataset.document_index
    add_extra = int(dataset.config.add_extra_token_to_sequence)

    spans: list[dict[str, int]] = []
    target_start = 0
    for i in range(doc_index_beg, doc_index_end + 1):
        real_doc = int(docmap[i])
        doc_start_token = int(seq_ptrs[real_doc] // token_itemsize)
        if i == doc_index_beg:
            src_start = doc_index_beg_offset
            length = int(seq_lens[real_doc]) - src_start
        elif i == doc_index_end:
            src_start = 0
            length = doc_index_end_offset + add_extra
        else:
            src_start = 0
            length = int(seq_lens[real_doc])
        if doc_index_beg == doc_index_end:
            src_start = doc_index_beg_offset
            length = doc_index_end_offset - doc_index_beg_offset + add_extra
        if length <= 0:
            continue
        spans.append(
            {
                "real_doc": real_doc,
                "doc_start_token": doc_start_token,
                "source_start": src_start,
                "source_end": src_start + length,
                "target_start": target_start,
            }
        )
        target_start += length
    return spans


def _get_absolute_token_indices(dataset: Any, idx: int) -> np.ndarray:
    """Reconstruct absolute token-level indices inside the flat bin file for the given sequence."""
    parts = []
    for span in _get_sample_token_spans(dataset, idx):
        start = span["doc_start_token"] + span["source_start"]
        length = span["source_end"] - span["source_start"]
        parts.append(np.arange(start, start + length, dtype=np.int64))
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)


def _align_token_sidecar_tensor(
    tensor: torch.Tensor,
    *,
    target_len: int,
    col: str,
    idx: int,
    pad_value: int | float = 0,
) -> torch.Tensor:
    """Align a token sidecar to Megatron's possibly padded sample length."""

    if tensor.shape[0] > target_len:
        raise ValueError(
            f"[cppmega-patch] sidecar col {col!r} len {tensor.shape[0]} > "
            f"token len {target_len} (idx {idx}); index reconstruction bug"
        )
    if tensor.shape[0] < target_len:
        pad = torch.full(
            (target_len - tensor.shape[0],),
            pad_value,
            dtype=tensor.dtype,
        )
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor.contiguous()


def _sample_objective_ids(
    objective_ids: np.ndarray,
    spans: list[dict[str, int]],
    *,
    target_len: int,
) -> torch.Tensor:
    """Expand document-aligned objective IDs over one packed Megatron sample."""

    values = np.asarray(objective_ids).reshape(-1)
    if target_len <= 0:
        raise ValueError(f"objective ID target_len must be positive, got {target_len}")
    result = torch.zeros((target_len,), dtype=torch.long)
    expected_target_start = 0
    for span_index, span in enumerate(spans):
        real_doc = int(span["real_doc"])
        source_start = int(span["source_start"])
        source_end = int(span["source_end"])
        target_start = int(span["target_start"])
        length = source_end - source_start
        if (
            real_doc < 0
            or real_doc >= values.size
            or source_start < 0
            or length <= 0
            or target_start < 0
            or target_start != expected_target_start
        ):
            raise ValueError(
                "objective ID sample span is invalid: "
                f"index={span_index} real_doc={real_doc} source=({source_start},{source_end}) "
                f"target_start={target_start} target_len={target_len}"
            )
        objective_id = int(values[real_doc])
        if objective_id not in _OBJECTIVE_ID_TO_TASK:
            raise ValueError(
                f"objective ID sample span {span_index} has unknown ID {objective_id}"
            )
        available = min(length, target_len - target_start)
        if available <= 0:
            raise ValueError(
                "objective ID sample span exceeds target length: "
                f"index={span_index} target_start={target_start} length={length} "
                f"target_len={target_len}"
            )
        if available < length and span_index != len(spans) - 1:
            raise ValueError(
                "objective ID sample span was truncated before the final span: "
                f"index={span_index} target_start={target_start} length={length} "
                f"target_len={target_len}"
            )
        result[target_start : target_start + available] = objective_id
        expected_target_start = target_start + available
    return result


def _token_sidecar_tensor(values: np.ndarray, *, col: str) -> torch.Tensor:
    """Preserve opaque identity bits instead of narrowing them to int64."""

    array_values = np.asarray(values)
    if col in _OPAQUE_UINT64_ID_COLS:
        if array_values.dtype != np.dtype(np.uint64):
            raise ValueError(
                f"[cppmega-patch] {col} must arrive as uint64, got "
                f"{array_values.dtype.name}"
            )
        return torch.from_numpy(np.array(array_values, dtype=np.uint64, copy=True))
    return torch.from_numpy(array_values).long()


def _sample_document_ids(
    raw_document_ids: torch.Tensor,
    spans: list[dict[str, int]],
    *,
    target_len: int,
) -> torch.Tensor:
    """Remap packed row-local IDs to unique sample-local graph segments."""

    values = raw_document_ids.reshape(-1).to(dtype=torch.long)
    if target_len < 1:
        raise ValueError(
            f"graph document target_len must be positive, got {target_len}"
        )
    result = torch.zeros((target_len,), dtype=torch.long, device=values.device)
    cursor = 0
    next_document_id = 1
    expected_target_start = 0
    for span_index, span in enumerate(spans):
        target_start = int(span["target_start"])
        source_length = int(span["source_end"]) - int(span["source_start"])
        if source_length <= 0 or target_start != expected_target_start:
            raise ValueError(
                f"sample span {span_index} has invalid length/start: "
                f"length={source_length} target_start={target_start} "
                f"expected={expected_target_start}"
            )
        available = min(source_length, int(values.numel()) - cursor)
        if available < 0:
            raise ValueError("sample document-id cursor exceeded source sidecar")
        if target_start + available > target_len:
            raise ValueError(
                f"sample span {span_index} exceeds graph document target length"
            )
        segment = values[cursor : cursor + available]
        if bool((segment <= 0).any().item()):
            raise ValueError(
                "packed doc_ids must be positive throughout every sampled token"
            )
        previous_raw_id: int | None = None
        current_document_id = 0
        for offset, raw_id in enumerate(segment.tolist()):
            raw_id = int(raw_id)
            if previous_raw_id is None or raw_id != previous_raw_id:
                current_document_id = next_document_id
                next_document_id += 1
            result[target_start + offset] = current_document_id
            previous_raw_id = raw_id
        cursor += available
        expected_target_start = target_start + available
        if available < source_length:
            if cursor != int(values.numel()) or span_index != len(spans) - 1:
                raise ValueError(
                    "sample document-id sidecar ended before a non-final source span"
                )
            break
    if cursor != int(values.numel()):
        raise ValueError(
            f"sample document-id spans consumed {cursor} values, have "
            f"{int(values.numel())}"
        )
    return result


def _slice_graph_doc(cache_entry: dict[str, Any], real_doc: int) -> np.ndarray:
    offsets = cache_entry["offsets"]
    start = int(offsets[real_doc])
    end = int(offsets[real_doc + 1])
    return np.asarray(cache_entry["data"][start:end])


def _cap_2d(
    values: list[tuple[int, int]], *, max_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.full((max_rows, 2), -1, dtype=torch.long)
    count = min(len(values), max_rows)
    if count:
        out[:count] = torch.tensor(values[:count], dtype=torch.long)
    return out, torch.tensor(count, dtype=torch.long)


def _cap_3d(
    values: list[tuple[int, int, int]], *, max_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.full((max_rows, 3), -1, dtype=torch.long)
    count = min(len(values), max_rows)
    if count:
        out[:count] = torch.tensor(values[:count], dtype=torch.long)
    return out, torch.tensor(count, dtype=torch.long)


def _cap_1d(
    values: list[int], *, max_rows: int, pad: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.full((max_rows,), pad, dtype=torch.long)
    count = min(len(values), max_rows)
    if count:
        out[:count] = torch.tensor(values[:count], dtype=torch.long)
    return out, torch.tensor(count, dtype=torch.long)


def _build_graph_route_tensors(
    graph_sidecars: Dict[str, Dict[str, Any]],
    spans: list[dict[str, int]],
    *,
    target_len: int,
    max_edges: int,
    max_chunks: int,
) -> Dict[str, torch.Tensor]:
    call_edges: list[tuple[int, int]] = []
    type_edges: list[tuple[int, int]] = []
    domain_edges: list[tuple[int, int, int]] = []
    build_edges: list[tuple[int, int, int]] = []
    shell_edges: list[tuple[int, int, int]] = []
    diagnostic_edges: list[tuple[int, int, int]] = []
    cross_domain_edges: list[tuple[int, int, int]] = []
    chunk_starts: list[int] = []
    chunk_ends: list[int] = []
    chunk_kinds: list[int] = []
    chunk_dep_levels: list[int] = []

    for span in spans:
        real_doc = span["real_doc"]
        source_start = span["source_start"]
        source_end = span["source_end"]
        target_start = span["target_start"]

        for source_name, sink in (
            ("token_domain_edges", domain_edges),
            ("token_build_edges", build_edges),
            ("token_shell_edges", shell_edges),
            ("token_diagnostic_edges", diagnostic_edges),
            ("token_cross_domain_edges", cross_domain_edges),
        ):
            rows = _slice_graph_doc(graph_sidecars[source_name], real_doc)
            for src, dst, kind in rows:
                src_i = int(src)
                dst_i = int(dst)
                if (
                    source_start <= src_i < source_end
                    and source_start <= dst_i < source_end
                ):
                    adj_src = target_start + src_i - source_start
                    adj_dst = target_start + dst_i - source_start
                    if 0 <= adj_src < target_len and 0 <= adj_dst < target_len:
                        sink.append((adj_src, adj_dst, int(kind)))

        starts = _slice_graph_doc(graph_sidecars["token_chunk_starts"], real_doc)
        ends = _slice_graph_doc(graph_sidecars["token_chunk_ends"], real_doc)
        kinds = _slice_graph_doc(graph_sidecars["token_chunk_kinds"], real_doc)
        dep_levels = _slice_graph_doc(
            graph_sidecars["token_chunk_dep_levels"], real_doc
        )
        if not (len(starts) == len(ends) == len(kinds) == len(dep_levels)):
            raise ValueError(
                f"[cppmega-patch] chunk graph sidecar lengths disagree for document {real_doc}: "
                f"starts={len(starts)} ends={len(ends)} kinds={len(kinds)} dep_levels={len(dep_levels)}"
            )
        doc_chunk_to_sample: dict[int, int] = {}
        for doc_chunk_index, (start, end, kind, dep_level) in enumerate(
            zip(starts, ends, kinds, dep_levels, strict=True)
        ):
            start_i = int(start)
            end_i = int(end)
            overlap_start = max(start_i, source_start)
            overlap_end = min(end_i, source_end)
            if overlap_start >= overlap_end:
                continue
            adj_start = target_start + overlap_start - source_start
            adj_end = target_start + overlap_end - source_start
            if adj_start >= target_len:
                continue
            adj_end = min(adj_end, target_len)
            if adj_start < adj_end:
                doc_chunk_to_sample[doc_chunk_index] = len(chunk_starts)
                chunk_starts.append(adj_start)
                chunk_ends.append(adj_end)
                chunk_kinds.append(int(kind))
                chunk_dep_levels.append(int(dep_level))

        # call/type pairs are chunk-index routes, not token offsets. Remap the
        # document-local chunk ids through the chunks that overlap this sample;
        # the attention/indexer layer expands them to token-span blocks.
        for source_name, sink in (
            ("token_call_edges", call_edges),
            ("token_type_edges", type_edges),
        ):
            rows = _slice_graph_doc(graph_sidecars[source_name], real_doc)
            for src, dst in rows:
                src_i = int(src)
                dst_i = int(dst)
                if (
                    src_i < 0
                    or dst_i < 0
                    or src_i >= len(starts)
                    or dst_i >= len(starts)
                ):
                    raise ValueError(
                        f"[cppmega-patch] {source_name} chunk endpoint out of range "
                        f"for document {real_doc} with {len(starts)} chunks: "
                        f"({src_i}, {dst_i})"
                    )
                if src_i in doc_chunk_to_sample and dst_i in doc_chunk_to_sample:
                    sink.append(
                        (doc_chunk_to_sample[src_i], doc_chunk_to_sample[dst_i])
                    )

    if len(chunk_starts) > max_chunks:
        raise ValueError(
            "[cppmega-patch] graph route chunk capacity exceeded: "
            f"required={len(chunk_starts)} configured={max_chunks}; increase "
            "CPPMEGA_GRAPH_MAX_CHUNKS rather than truncating routes"
        )
    edge_families = {
        "call": call_edges,
        "type": type_edges,
        "domain": domain_edges,
        "build": build_edges,
        "shell": shell_edges,
        "diagnostic": diagnostic_edges,
        "cross_domain": cross_domain_edges,
    }
    overflow = {
        name: len(edges)
        for name, edges in edge_families.items()
        if len(edges) > max_edges
    }
    if overflow:
        raise ValueError(
            "[cppmega-patch] graph route edge capacity exceeded: "
            f"configured={max_edges} required={overflow}; increase "
            "CPPMEGA_GRAPH_MAX_EDGES rather than truncating routes"
        )

    graph_call_edges, graph_call_edge_counts = _cap_2d(call_edges, max_rows=max_edges)
    graph_type_edges, graph_type_edge_counts = _cap_2d(type_edges, max_rows=max_edges)
    graph_domain_edges, graph_domain_edge_counts = _cap_3d(
        domain_edges, max_rows=max_edges
    )
    graph_build_edges, graph_build_edge_counts = _cap_3d(
        build_edges, max_rows=max_edges
    )
    graph_shell_edges, graph_shell_edge_counts = _cap_3d(
        shell_edges, max_rows=max_edges
    )
    graph_diagnostic_edges, graph_diagnostic_edge_counts = _cap_3d(
        diagnostic_edges, max_rows=max_edges
    )
    graph_cross_domain_edges, graph_cross_domain_edge_counts = _cap_3d(
        cross_domain_edges, max_rows=max_edges
    )
    graph_chunk_starts, graph_chunk_counts = _cap_1d(chunk_starts, max_rows=max_chunks)
    graph_chunk_ends, _ = _cap_1d(chunk_ends, max_rows=max_chunks)
    graph_chunk_kinds, _ = _cap_1d(chunk_kinds, max_rows=max_chunks)
    graph_chunk_dep_levels, _ = _cap_1d(chunk_dep_levels, max_rows=max_chunks)

    return {
        "graph_call_edges": graph_call_edges,
        "graph_call_edge_counts": graph_call_edge_counts,
        "graph_type_edges": graph_type_edges,
        "graph_type_edge_counts": graph_type_edge_counts,
        "graph_domain_edges": graph_domain_edges,
        "graph_domain_edge_counts": graph_domain_edge_counts,
        "graph_build_edges": graph_build_edges,
        "graph_build_edge_counts": graph_build_edge_counts,
        "graph_shell_edges": graph_shell_edges,
        "graph_shell_edge_counts": graph_shell_edge_counts,
        "graph_diagnostic_edges": graph_diagnostic_edges,
        "graph_diagnostic_edge_counts": graph_diagnostic_edge_counts,
        "graph_cross_domain_edges": graph_cross_domain_edges,
        "graph_cross_domain_edge_counts": graph_cross_domain_edge_counts,
        "graph_chunk_starts": graph_chunk_starts,
        "graph_chunk_ends": graph_chunk_ends,
        "graph_chunk_kinds": graph_chunk_kinds,
        "graph_chunk_dep_levels": graph_chunk_dep_levels,
        "graph_chunk_counts": graph_chunk_counts,
    }


# --- 1. Monkey-patch GPTDataset.__getitem__ ---
try:
    from megatron.core.datasets.gpt_dataset import (  # pyright: ignore[reportMissingImports]
        GPTDataset,
    )

    orig_getitem = GPTDataset.__getitem__

    def patched_getitem(
        self: GPTDataset, idx: Optional[int]
    ) -> Dict[str, torch.Tensor]:
        sample = orig_getitem(self, idx)

        structure_enabled = _env_flag("CPPMEGA_STRUCTURE_ENABLED")
        graph_enabled = _env_flag("CPPMEGA_GRAPH_ROUTES_ENABLED")
        if graph_enabled and not structure_enabled:
            raise RuntimeError(
                "CPPMEGA_GRAPH_ROUTES_ENABLED=1 requires CPPMEGA_STRUCTURE_ENABLED=1"
            )
        if not structure_enabled:
            try:
                _load_sidecar_manifest(self)
            except FileNotFoundError:
                pass
            return sample

        objective_id_mmap = _lazy_init_objective_ids(self)

        if idx is None:
            # Padded sequence: return zero tensors matching the tokens shape
            for col in _TOKEN_BATCH_COLS:
                sample[col] = _padded_token_sidecar_tensor(sample["tokens"], col=col)
            if os.environ.get("CPPMEGA_GRAPH_ROUTES_ENABLED", "0") == "1":
                max_edges = _graph_capacity("CPPMEGA_GRAPH_MAX_EDGES")
                max_chunks = _graph_capacity("CPPMEGA_GRAPH_MAX_CHUNKS")
                graph = _build_graph_route_tensors(
                    {
                        "token_call_edges": {
                            "offsets": np.array([0, 0]),
                            "data": np.empty((0, 2), dtype=np.int32),
                        },
                        "token_type_edges": {
                            "offsets": np.array([0, 0]),
                            "data": np.empty((0, 2), dtype=np.int32),
                        },
                        "token_domain_edges": {
                            "offsets": np.array([0, 0]),
                            "data": np.empty((0, 3), dtype=np.int32),
                        },
                        "token_build_edges": {
                            "offsets": np.array([0, 0]),
                            "data": np.empty((0, 3), dtype=np.int32),
                        },
                        "token_shell_edges": {
                            "offsets": np.array([0, 0]),
                            "data": np.empty((0, 3), dtype=np.int32),
                        },
                        "token_diagnostic_edges": {
                            "offsets": np.array([0, 0]),
                            "data": np.empty((0, 3), dtype=np.int32),
                        },
                        "token_cross_domain_edges": {
                            "offsets": np.array([0, 0]),
                            "data": np.empty((0, 3), dtype=np.int32),
                        },
                        "token_chunk_starts": {
                            "offsets": np.array([0, 0]),
                            "data": np.empty((0,), dtype=np.uint32),
                        },
                        "token_chunk_ends": {
                            "offsets": np.array([0, 0]),
                            "data": np.empty((0,), dtype=np.uint32),
                        },
                        "token_chunk_kinds": {
                            "offsets": np.array([0, 0]),
                            "data": np.empty((0,), dtype=np.uint8),
                        },
                        "token_chunk_dep_levels": {
                            "offsets": np.array([0, 0]),
                            "data": np.empty((0,), dtype=np.uint16),
                        },
                    },
                    [],
                    target_len=int(sample["tokens"].shape[-1]),
                    max_edges=max_edges,
                    max_chunks=max_chunks,
                )
                graph["graph_document_ids"] = torch.zeros(
                    sample["tokens"].shape,
                    dtype=torch.long,
                    device=sample["tokens"].device,
                )
                sample.update(graph)
            if objective_id_mmap is not None:
                sample["objective_ids"] = torch.zeros(
                    sample["tokens"].shape,
                    dtype=torch.long,
                    device=sample["tokens"].device,
                )
            return sample

        # Initialize and fetch side-channels
        side_channels = _lazy_init_side_channels(self)
        if not side_channels:
            raise RuntimeError(
                "[cppmega-patch] no side channels loaded while CPPMEGA_STRUCTURE_ENABLED=1"
            )

        # RULE #1: no try/except->zeros. Resolve each canonical column from the
        # sidecar under any known alias; an unresolved column while structure is
        # enabled is a real misconfiguration and RAISES with WHERE+WHAT.
        indices = _get_absolute_token_indices(self, idx)
        target_len = int(sample["tokens"].shape[-1])
        spans = (
            _get_sample_token_spans(self, idx)
            if objective_id_mmap is not None
            or _env_flag("CPPMEGA_GRAPH_ROUTES_ENABLED")
            else None
        )
        if objective_id_mmap is not None:
            if spans is None:
                raise RuntimeError(
                    "[cppmega-patch] objective IDs require sampled document spans"
                )
            sample["objective_ids"] = _sample_objective_ids(
                objective_id_mmap,
                spans,
                target_len=target_len,
            ).to(device=sample["tokens"].device)

        # Packed cppmega parquet carries loss_mask that suppresses pad and
        # inter-document boundary labels. Megatron's default GPTDataset mask is
        # not enough for our packed multi-document rows, so require and apply it
        # here before the batch bridge sees the sample.
        loss_mask_source = next(
            (a for a in _LOSS_MASK_ALIASES if a in side_channels), None
        )
        if loss_mask_source is None:
            raise KeyError(
                "[cppmega-patch] parquet loss_mask sidecar missing from dataset "
                f"side-channels (tried {_LOSS_MASK_ALIASES}; have "
                f"{sorted(side_channels)}) while CPPMEGA_STRUCTURE_ENABLED=1"
            )
        loss_mask_entry = side_channels[loss_mask_source]
        loss_vals = loss_mask_entry["mmap"][indices]
        if loss_mask_entry.get("widen_to_uint64"):
            loss_vals = loss_vals.astype(np.uint64)
        loss_tensor = torch.from_numpy(loss_vals).float()
        if self.config.add_extra_token_to_sequence:
            loss_tensor = loss_tensor[:-1]
        sample["loss_mask"] = _align_token_sidecar_tensor(
            loss_tensor,
            target_len=target_len,
            col=loss_mask_source,
            idx=idx,
            pad_value=0.0,
        )
        if idx == 0:
            print(
                f"[cppmega-patch] Mapped side-channel {loss_mask_source} -> loss_mask",
                flush=True,
            )

        required_token_cols = _required_token_batch_cols()
        for col in _TOKEN_BATCH_COLS:
            source = next(
                (a for a in _TOKEN_COL_ALIASES[col] if a in side_channels), None
            )
            if source is None:
                if col in required_token_cols:
                    raise KeyError(
                        f"[cppmega-patch] required token sidecar column {col!r} "
                        f"missing from dataset side-channels (tried "
                        f"{_TOKEN_COL_ALIASES[col]}; have {sorted(side_channels)})"
                    )
                continue
            entry = side_channels[source]
            vals = entry["mmap"][indices]
            if entry.get("widen_to_uint64"):
                vals = vals.astype(np.uint64)
            tensor = _token_sidecar_tensor(vals, col=col)
            if self.config.add_extra_token_to_sequence:
                tensor = tensor[:-1]
            tensor = tensor.contiguous()
            # Align to the (possibly pad-extended) token length. Megatron pads a
            # short trailing sample's tokens up to sequence_length; mirror that by
            # zero-padding the structure tail -- those are genuine pad positions
            # (loss-masked), so zeros are correct, NOT a silent data fallback.
            # RULE #1: a structure run LONGER than the token window means the index
            # reconstruction is wrong -> RAISE rather than silently truncate.
            sample[col] = _align_token_sidecar_tensor(
                tensor,
                target_len=target_len,
                col=col,
                idx=idx,
                pad_value=0,
            )
            if idx == 0:
                print(
                    f"[cppmega-patch] Mapped side-channel {source} -> {col}",
                    flush=True,
                )

        if _env_flag("CPPMEGA_GRAPH_ROUTES_ENABLED"):
            graph_sidecars = _lazy_init_graph_sidecars(self)
            if not graph_sidecars:
                raise RuntimeError(
                    "[cppmega-patch] no graph sidecars loaded while CPPMEGA_GRAPH_ROUTES_ENABLED=1"
                )
            max_edges = _graph_capacity("CPPMEGA_GRAPH_MAX_EDGES")
            max_chunks = _graph_capacity("CPPMEGA_GRAPH_MAX_CHUNKS")
            spans = _get_sample_token_spans(self, idx)
            if spans is None:
                raise RuntimeError(
                    "[cppmega-patch] graph routes require sampled document spans"
                )
            graph = _build_graph_route_tensors(
                graph_sidecars,
                spans,
                target_len=int(sample["tokens"].shape[-1]),
                max_edges=max_edges,
                max_chunks=max_chunks,
            )
            document_entry = side_channels.get("doc_ids")
            if document_entry is None:
                raise KeyError(
                    "[cppmega-patch] graph auxiliary loss requires the packed "
                    "doc_ids sidecar; token_source_doc_ids is provenance and "
                    "cannot substitute for segment boundaries"
                )
            doc_ids_vals = document_entry["mmap"][indices]
            if document_entry.get("widen_to_uint64"):
                doc_ids_vals = doc_ids_vals.astype(np.uint64)
            raw_document_ids = torch.from_numpy(doc_ids_vals).long()
            if self.config.add_extra_token_to_sequence:
                raw_document_ids = raw_document_ids[:-1]
            graph["graph_document_ids"] = _sample_document_ids(
                raw_document_ids,
                spans,
                target_len=int(sample["tokens"].shape[-1]),
            )
            sample.update(graph)

        return sample

    GPTDataset.__getitem__ = patched_getitem
    print("[cppmega-patch] Successfully patched GPTDataset.__getitem__", flush=True)
except ImportError:
    # Megatron not installed (local unit-test path) -- nothing to patch.
    pass
except Exception as e:
    # RULE #1: Megatron present but patching failed -> a training run would
    # silently proceed without structure ingress. Fail loud.
    raise RuntimeError(
        f"[cppmega-patch] failed to patch GPTDataset.__getitem__: {e}"
    ) from e


# --- 2. Monkey-patch get_batch_on_this_tp_rank ---
try:
    try:
        # Megatron core_v0.18.0 moved this helper here; pretrain_gpt.py imports
        # from this module directly, so this patch must land before runpy enters
        # upstream pretrain_mamba/pretrain_hybrid.
        import megatron.core.utils as batch_utils  # type: ignore[import-not-found]
        if not hasattr(batch_utils, "get_batch_on_this_tp_rank"):
            raise ImportError(
                "megatron.core.utils has no get_batch_on_this_tp_rank"
            )
    except (ImportError, AttributeError):
        # Older cppmega H200 trees used the training.utils location.
        import megatron.training.utils as batch_utils  # type: ignore[import-not-found]

    batch_utils.get_batch_on_this_tp_rank = _make_get_batch_on_this_tp_rank_bridge(
        batch_utils.get_batch_on_this_tp_rank
    )
    print(
        f"[cppmega-patch] Successfully patched {batch_utils.__name__}.get_batch_on_this_tp_rank",
        flush=True,
    )
except ImportError:
    # Megatron not installed (local unit-test path) -- nothing to patch.
    pass
except Exception as e:
    raise RuntimeError(
        f"[cppmega-patch] failed to patch get_batch_on_this_tp_rank: {e}"
    ) from e


# --- 3. Monkey-patch MambaModel / GPTModel forward passes ---
try:
    from megatron.core.models.mamba import (  # pyright: ignore[reportMissingImports]
        MambaModel,
    )

    orig_mamba_forward = MambaModel.forward

    def patched_mamba_forward(self: MambaModel, *args, **kwargs) -> Any:
        structure_batch = _get_current_structure_batch()
        if structure_batch:
            from cppmega.megatron.structure_batch import maybe_set_structure_inputs

            maybe_set_structure_inputs(self, structure_batch)
        return orig_mamba_forward(self, *args, **kwargs)

    MambaModel.forward = patched_mamba_forward
    print("[cppmega-patch] Successfully patched MambaModel.forward", flush=True)
except ImportError:
    pass
except Exception as e:
    raise RuntimeError(
        f"[cppmega-patch] failed to patch MambaModel.forward: {e}"
    ) from e


try:
    from megatron.core.models.gpt import (  # pyright: ignore[reportMissingImports]
        GPTModel,
    )

    orig_gpt_forward = GPTModel.forward

    def patched_gpt_forward(self: GPTModel, *args, **kwargs) -> Any:
        structure_batch = _get_current_structure_batch()
        if structure_batch:
            from cppmega.megatron.structure_batch import maybe_set_structure_inputs

            maybe_set_structure_inputs(self, structure_batch)
        return orig_gpt_forward(self, *args, **kwargs)

    GPTModel.forward = patched_gpt_forward
    print("[cppmega-patch] Successfully patched GPTModel.forward", flush=True)
except ImportError:
    pass
except Exception as e:
    raise RuntimeError(f"[cppmega-patch] failed to patch GPTModel.forward: {e}") from e
