"""Fail-closed production-batch observation for the H200 preflight."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum
import json
import os
from pathlib import Path
from typing import Mapping

from cppmega.receipt_binding import (
    validate_binding_shape,
    validate_receipt_binding,
)
from cppmega.megatron.graph_objective_loss import graph_bias_beta_binding
from cppmega.megatron.objective_contract import OBJECTIVE_IDS


class GraphChunkKind(IntEnum):
    OTHER = 0
    PREAMBLE = 1
    FUNCTION_SIGNATURE = 2
    FUNCTION_BODY = 3
    CLASS_DECLARATION = 4
    CLASS_MEMBER = 5
    COMMENT = 6
    TYPEDEF = 7
    NAMESPACE = 8
    BUILD = 9
    HEADER_FRAGMENT = 10
    MACRO = 11


GRAPH_CHUNK_KIND_COUNT = len(GraphChunkKind)
if tuple(int(kind) for kind in GraphChunkKind) != tuple(
    range(GRAPH_CHUNK_KIND_COUNT)
):
    raise RuntimeError("GraphChunkKind values must remain contiguous from zero")


REQUIRED_BATCH_FIELDS = ("tokens", "labels", "loss_mask")
REQUIRED_GRAPH_BATCH_FIELDS = (
    "source_doc_ids",
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
)
_ROUTE_FAMILIES = (
    ("call", 2, "chunk"),
    ("type", 2, "chunk"),
    ("domain", 3, "token"),
    ("build", 3, "token"),
    ("shell", 3, "token"),
    ("diagnostic", 3, "token"),
    ("cross_domain", 3, "token"),
)
_OBJECTIVE_ID_TO_TASK = {value: task for task, value in OBJECTIVE_IDS.items()}


def _tensor_summary(value: object) -> dict[str, object]:
    import torch

    shape = getattr(value, "shape", None)
    if shape is None or not hasattr(value, "numel"):
        raise RuntimeError(f"preflight batch value is not a tensor: {type(value)!r}")
    detached = value.detach()
    return {
        "shape": [int(size) for size in shape],
        "dtype": str(getattr(value, "dtype", "unknown")),
        "device": str(getattr(value, "device", "unknown")),
        "numel": int(value.numel()),
        "nonzero": int(detached.count_nonzero().item()),
        "sum": float(detached.to(dtype=torch.float64).sum().item()),
    }


def _objective_mix_summary(
    *,
    batch: Mapping[str, object],
    structure_batch: Mapping[str, object],
    valid_tokens: object,
) -> dict[str, object]:
    import torch

    objective_ids = structure_batch.get("objective_ids")
    if not isinstance(objective_ids, torch.Tensor):
        raise RuntimeError("production objective batch is missing objective_ids")
    token_shape = tuple(batch["tokens"].shape)
    if tuple(objective_ids.shape) != token_shape:
        raise RuntimeError(
            "production objective_ids shape "
            f"{tuple(objective_ids.shape)} != tokens {token_shape}"
        )
    ids = objective_ids.detach().to(device="cpu", dtype=torch.int64)
    valid = valid_tokens.detach().to(device="cpu", dtype=torch.bool)
    nonzero_ids = ids[ids > 0]
    unknown = sorted(
        int(value)
        for value in torch.unique(nonzero_ids).tolist()
        if int(value) not in _OBJECTIVE_ID_TO_TASK
    )
    if unknown:
        raise RuntimeError(f"production objective_ids contain unknown IDs: {unknown}")
    if torch.any(ids[valid] <= 0):
        raise RuntimeError(
            "production objective_ids must be positive for every valid token"
        )
    loss_mask = batch["loss_mask"].detach().to(device="cpu")
    trained = valid & loss_mask.ne(0)
    observed_ids = sorted(int(value) for value in torch.unique(ids[valid]).tolist())
    return {
        "input_tokens_by_objective": {
            _OBJECTIVE_ID_TO_TASK[objective_id]: int(
                ((ids == objective_id) & valid).sum().item()
            )
            for objective_id in observed_ids
        },
        "loss_tokens_by_objective": {
            _OBJECTIVE_ID_TO_TASK[objective_id]: int(
                ((ids == objective_id) & trained).sum().item()
            )
            for objective_id in observed_ids
        },
        "observed_objective_ids": observed_ids,
    }


def _validate_active_graph(
    batch: Mapping[str, object], structure_batch: Mapping[str, object]
) -> dict[str, int]:
    import torch

    token_shape = tuple(batch["tokens"].shape)
    if len(token_shape) != 2 or token_shape[0] <= 0 or token_shape[1] <= 0:
        raise RuntimeError(
            f"production tokens must be a nonempty 2D batch, got {token_shape}"
        )
    for name in ("labels", "loss_mask"):
        if tuple(batch[name].shape) != token_shape:
            raise RuntimeError(
                f"production {name} shape {tuple(batch[name].shape)} != tokens {token_shape}"
            )
    if tuple(structure_batch["structure_ids"].shape) != token_shape:
        raise RuntimeError(
            "production structure_ids shape "
            f"{tuple(structure_batch['structure_ids'].shape)} != tokens {token_shape}"
        )

    chunk_names = (
        "graph_chunk_starts",
        "graph_chunk_ends",
        "graph_chunk_kinds",
        "graph_chunk_dep_levels",
    )
    chunks = {
        name: structure_batch[name].detach().to(device="cpu") for name in chunk_names
    }
    chunk_shape = tuple(chunks["graph_chunk_starts"].shape)
    if len(chunk_shape) != 2 or chunk_shape[0] != token_shape[0]:
        raise RuntimeError(
            f"production graph chunk tensors must be [batch, capacity], got {chunk_shape}"
        )
    for name, tensor in chunks.items():
        if tuple(tensor.shape) != chunk_shape:
            raise RuntimeError(
                f"production {name} shape {tuple(tensor.shape)} != {chunk_shape}"
            )
    counts = (
        structure_batch["graph_chunk_counts"]
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .reshape(-1)
    )
    if counts.numel() != token_shape[0]:
        raise RuntimeError(
            "production graph_chunk_counts must contain one count per batch row"
        )

    source_doc_ids = (
        structure_batch["source_doc_ids"]
        .detach()
        .to(device="cpu", dtype=torch.int64)
    )
    if tuple(source_doc_ids.shape) != token_shape:
        raise RuntimeError(
            "production source_doc_ids shape must match tokens"
        )
    valid_tokens = (
        batch["tokens"].detach().to(device="cpu").ne(0)
        | batch["labels"].detach().to(device="cpu").ne(0)
        | batch["loss_mask"].detach().to(device="cpu").ne(0)
        | structure_batch["structure_ids"].detach().to(device="cpu").ne(0)
    )
    if torch.any(source_doc_ids[valid_tokens] <= 0):
        raise RuntimeError(
            "production source_doc_ids must be positive for every valid token"
        )

    total_chunks = 0
    max_end = 0
    chunk_docs_by_row: list[list[int]] = []
    for row, raw_count in enumerate(counts.tolist()):
        count = int(raw_count)
        if count < 0 or count > chunk_shape[1]:
            raise RuntimeError(
                f"production graph chunk count {count} exceeds capacity {chunk_shape[1]}"
            )
        if count == 0:
            chunk_docs_by_row.append([])
            continue
        starts = chunks["graph_chunk_starts"][row, :count].to(dtype=torch.int64)
        ends = chunks["graph_chunk_ends"][row, :count].to(dtype=torch.int64)
        kinds = chunks["graph_chunk_kinds"][row, :count].to(dtype=torch.int64)
        levels = chunks["graph_chunk_dep_levels"][row, :count].to(dtype=torch.int64)
        if (
            torch.any(starts < 0)
            or torch.any(ends <= starts)
            or torch.any(ends > token_shape[1])
            or torch.any(levels < 0)
        ):
            raise RuntimeError("production active graph chunk spans are invalid")
        if torch.any(kinds < int(GraphChunkKind.OTHER)) or torch.any(
            kinds >= GRAPH_CHUNK_KIND_COUNT
        ):
            raise RuntimeError(
                "production graph chunk kind is outside the canonical range "
                f"[0,{GRAPH_CHUNK_KIND_COUNT})"
            )
        if count > 1 and torch.any(starts[1:] < ends[:-1]):
            raise RuntimeError(
                "production graph chunks must be ordered and nonoverlapping"
            )
        total_chunks += count
        max_end = max(max_end, int(ends.max().item()))
        chunk_docs: list[int] = []
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            docs = source_doc_ids[row, int(start) : int(end)]
            if docs.numel() == 0 or torch.any(docs != docs[0]):
                raise RuntimeError(
                    "production graph chunk crosses source-document provenance"
                )
            chunk_docs.append(int(docs[0].item()))
        chunk_docs_by_row.append(chunk_docs)
    if total_chunks <= 0:
        raise RuntimeError("production graph batch has no active chunks")

    route_edge_counts: dict[str, int] = {}
    for family, width, coordinate_space in _ROUTE_FAMILIES:
        edge_key = f"graph_{family}_edges"
        count_key = f"graph_{family}_edge_counts"
        edges = structure_batch[edge_key].detach().to(
            device="cpu", dtype=torch.int64
        )
        edge_counts = structure_batch[count_key].detach().to(
            device="cpu", dtype=torch.int64
        ).reshape(-1)
        if (
            edges.ndim != 3
            or edges.shape[0] != token_shape[0]
            or edges.shape[2] != width
            or edge_counts.numel() != token_shape[0]
        ):
            raise RuntimeError(
                f"production {family} route tensors have invalid shape"
            )
        family_total = 0
        for row, raw_count in enumerate(edge_counts.tolist()):
            count = int(raw_count)
            if count < 0 or count > edges.shape[1]:
                raise RuntimeError(
                    f"production {family} edge count exceeds capacity"
                )
            active_edges = edges[row, :count]
            family_total += count
            for edge in active_edges.tolist():
                source, target = int(edge[0]), int(edge[1])
                if coordinate_space == "chunk":
                    chunk_docs = chunk_docs_by_row[row]
                    if (
                        source < 0
                        or target < 0
                        or source >= len(chunk_docs)
                        or target >= len(chunk_docs)
                    ):
                        raise RuntimeError(
                            f"production {family} route endpoint exceeds chunks"
                        )
                    source_doc, target_doc = (
                        chunk_docs[source],
                        chunk_docs[target],
                    )
                else:
                    if (
                        source < 0
                        or target < 0
                        or source >= token_shape[1]
                        or target >= token_shape[1]
                        or int(edge[2]) <= 0
                    ):
                        raise RuntimeError(
                            f"production {family} route endpoint/kind is invalid"
                        )
                    source_doc = int(source_doc_ids[row, source].item())
                    target_doc = int(source_doc_ids[row, target].item())
                if source_doc <= 0 or source_doc != target_doc:
                    raise RuntimeError(
                        f"production {family} route endpoint provenance mismatch"
                    )
        if family_total:
            route_edge_counts[family] = family_total
    route_edge_count = sum(route_edge_counts.values())
    if route_edge_count <= 0:
        raise RuntimeError("production graph batch has no active route edges")
    return {
        "chunk_count": total_chunks,
        "max_chunk_end": max_end,
        "route_edge_count": route_edge_count,
        "route_edge_counts": route_edge_counts,
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _binding_from_environment() -> dict[str, object] | None:
    raw = os.environ.get("CPPMEGA_H200_RECEIPT_BINDING")
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "CPPMEGA_H200_RECEIPT_BINDING must be JSON"
        ) from error
    return validate_binding_shape(value, where="H200 environment")


def observe_production_batch(
    *,
    batch: Mapping[str, object],
    structure_batch: Mapping[str, object],
    receipt_path: str | Path,
    receipt_binding: Mapping[str, object] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Record the first real Megatron batch after sidecar materialization."""

    environment = os.environ if environment is None else environment
    if receipt_binding is None:
        receipt_binding = _binding_from_environment()
    output = Path(receipt_path)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (
            existing.get("schema") != "cppmega_h200_production_batch_v1"
            or existing.get("status") != "verified"
        ):
            raise RuntimeError(f"invalid existing H200 batch receipt: {output}")
        if receipt_binding is not None:
            validate_receipt_binding(
                existing.get("binding"),
                expected=receipt_binding,
                where="H200 batch receipt",
            )
        return existing

    missing_batch = sorted(set(REQUIRED_BATCH_FIELDS) - set(batch))
    if missing_batch:
        raise RuntimeError(f"production Megatron batch fields missing: {missing_batch}")
    missing_graph = sorted(set(REQUIRED_GRAPH_BATCH_FIELDS) - set(structure_batch))
    if missing_graph:
        raise RuntimeError(f"missing graph batch fields: {missing_graph}")
    if "structure_ids" not in structure_batch:
        raise RuntimeError("production structure batch is missing structure_ids")
    if (
        environment.get("CPPMEGA_OBJECTIVE_CONTRACT_REQUIRED", "0") == "1"
        and "objective_ids" not in structure_batch
    ):
        raise RuntimeError(
            "production objective contract requires objective_ids in the batch"
        )

    batch_summary = {
        name: _tensor_summary(batch[name]) for name in REQUIRED_BATCH_FIELDS
    }
    structure_names = ("structure_ids", *REQUIRED_GRAPH_BATCH_FIELDS)
    if "objective_ids" in structure_batch:
        structure_names = (*structure_names, "objective_ids")
    structure_summary = {
        name: _tensor_summary(structure_batch[name])
        for name in structure_names
    }
    if batch_summary["tokens"]["numel"] <= 0:
        raise RuntimeError("production tokens batch is empty")
    if batch_summary["loss_mask"]["nonzero"] <= 0:
        raise RuntimeError("production loss_mask must contain nonzero values")
    if structure_summary["structure_ids"]["nonzero"] <= 0:
        raise RuntimeError("production structure_ids must contain nonzero values")
    if structure_summary["graph_chunk_counts"]["sum"] <= 0:
        raise RuntimeError("production graph_chunk_counts must be nonzero")
    if structure_summary["graph_chunk_ends"]["nonzero"] <= 0:
        raise RuntimeError("production graph_chunk_ends must contain nonzero values")
    active_graph = _validate_active_graph(batch, structure_batch)
    valid_tokens = (
        batch["tokens"].detach().to(device="cpu").ne(0)
        | batch["labels"].detach().to(device="cpu").ne(0)
        | batch["loss_mask"].detach().to(device="cpu").ne(0)
        | structure_batch["structure_ids"].detach().to(device="cpu").ne(0)
    )
    objective_mix = (
        _objective_mix_summary(
            batch=batch,
            structure_batch=structure_batch,
            valid_tokens=valid_tokens,
        )
        if "objective_ids" in structure_batch
        else None
    )
    source_doc_ids = structure_batch["source_doc_ids"].detach().to(device="cpu")
    positive_source_ids = source_doc_ids[source_doc_ids > 0]

    receipt: dict[str, object] = {
        "schema": "cppmega_h200_production_batch_v1",
        "status": "verified",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "batch": batch_summary,
        "structure": structure_summary,
        "active_graph": active_graph,
        "source_provenance": {
            "positive_token_count": int(positive_source_ids.numel()),
            "minimum_source_doc_id": int(positive_source_ids.min().item()),
        },
    }
    if objective_mix is not None:
        receipt["objective_mix"] = objective_mix
    if receipt_binding is not None:
        receipt["binding"] = validate_binding_shape(
            receipt_binding, where="H200 batch receipt"
        )
    _write_json_atomic(output, receipt)
    return receipt


def observe_graph_prior(
    *,
    prior: object,
    consumer: str,
    receipt_path: str | Path,
    receipt_binding: Mapping[str, object] | None = None,
    bias_beta: float | None = None,
) -> dict[str, object]:
    """Prove that the selected graph consumer received a nonzero prior."""

    import torch

    beta_binding = graph_bias_beta_binding(bias_beta)
    if receipt_binding is None:
        receipt_binding = _binding_from_environment()
    if consumer not in {"dense_attention", "dsa_indexer"}:
        raise RuntimeError(f"unsupported graph-prior consumer {consumer!r}")
    output = Path(receipt_path)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (
            existing.get("schema") != "cppmega_h200_graph_prior_v1"
            or existing.get("status") != "verified"
            or existing.get("consumer") != consumer
        ):
            raise RuntimeError(f"invalid existing H200 graph-prior receipt: {output}")
        if existing.get("bias_beta") != beta_binding:
            raise RuntimeError(
                f"existing H200 graph-prior receipt beta differs from runtime: {output}"
            )
        if receipt_binding is not None:
            validate_receipt_binding(
                existing.get("binding"),
                expected=receipt_binding,
                where="H200 graph-prior receipt",
            )
        return existing
    if not isinstance(prior, torch.Tensor):
        raise RuntimeError("graph prior must be a tensor")
    detached = prior.detach()
    if detached.numel() <= 0 or not torch.isfinite(detached).all():
        raise RuntimeError("graph prior must be nonempty and finite")
    nonzero = int(torch.count_nonzero(detached).item())
    if nonzero <= 0:
        raise RuntimeError("graph prior consumer input must be nonzero")
    payload: dict[str, object] = {
        "schema": "cppmega_h200_graph_prior_v1",
        "status": "verified",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "rank": int(os.environ.get("RANK", "0")),
        "consumer": consumer,
        "bias_beta": beta_binding,
        "prior": {
            "shape": [int(value) for value in detached.shape],
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "nonzero": nonzero,
            "sum": float(detached.to(dtype=torch.float64).sum().item()),
            "minimum": float(detached.min().item()),
            "maximum": float(detached.max().item()),
        },
    }
    if receipt_binding is not None:
        payload["binding"] = validate_binding_shape(
            receipt_binding, where="H200 graph-prior receipt"
        )
    _write_json_atomic(output, payload)
    return payload
