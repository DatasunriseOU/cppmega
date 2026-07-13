"""Fail-closed production-batch observation for the H200 preflight."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Mapping

REQUIRED_BATCH_FIELDS = ("tokens", "labels", "loss_mask")
REQUIRED_GRAPH_BATCH_FIELDS = (
    "graph_chunk_starts",
    "graph_chunk_ends",
    "graph_chunk_kinds",
    "graph_chunk_dep_levels",
    "graph_chunk_counts",
)


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

    total_chunks = 0
    max_end = 0
    for row, raw_count in enumerate(counts.tolist()):
        count = int(raw_count)
        if count < 0 or count > chunk_shape[1]:
            raise RuntimeError(
                f"production graph chunk count {count} exceeds capacity {chunk_shape[1]}"
            )
        if count == 0:
            continue
        starts = chunks["graph_chunk_starts"][row, :count].to(dtype=torch.int64)
        ends = chunks["graph_chunk_ends"][row, :count].to(dtype=torch.int64)
        kinds = chunks["graph_chunk_kinds"][row, :count].to(dtype=torch.int64)
        levels = chunks["graph_chunk_dep_levels"][row, :count].to(dtype=torch.int64)
        if (
            torch.any(starts < 0)
            or torch.any(ends <= starts)
            or torch.any(ends > token_shape[1])
            or torch.any(kinds <= 0)
            or torch.any(levels < 0)
        ):
            raise RuntimeError("production active graph chunk spans/kinds are invalid")
        total_chunks += count
        max_end = max(max_end, int(ends.max().item()))
    if total_chunks <= 0:
        raise RuntimeError("production graph batch has no active chunks")
    return {"chunk_count": total_chunks, "max_chunk_end": max_end}


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def observe_production_batch(
    *,
    batch: Mapping[str, object],
    structure_batch: Mapping[str, object],
    receipt_path: str | Path,
) -> dict[str, object]:
    """Record the first real Megatron batch after sidecar materialization."""

    output = Path(receipt_path)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (
            existing.get("schema") != "cppmega_h200_production_batch_v1"
            or existing.get("status") != "verified"
        ):
            raise RuntimeError(f"invalid existing H200 batch receipt: {output}")
        return existing

    missing_batch = sorted(set(REQUIRED_BATCH_FIELDS) - set(batch))
    if missing_batch:
        raise RuntimeError(f"production Megatron batch fields missing: {missing_batch}")
    missing_graph = sorted(set(REQUIRED_GRAPH_BATCH_FIELDS) - set(structure_batch))
    if missing_graph:
        raise RuntimeError(f"missing graph batch fields: {missing_graph}")
    if "structure_ids" not in structure_batch:
        raise RuntimeError("production structure batch is missing structure_ids")

    batch_summary = {
        name: _tensor_summary(batch[name]) for name in REQUIRED_BATCH_FIELDS
    }
    structure_summary = {
        name: _tensor_summary(structure_batch[name])
        for name in ("structure_ids", *REQUIRED_GRAPH_BATCH_FIELDS)
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
    if structure_summary["graph_chunk_kinds"]["nonzero"] <= 0:
        raise RuntimeError("production graph_chunk_kinds must contain nonzero values")
    active_graph = _validate_active_graph(batch, structure_batch)

    receipt: dict[str, object] = {
        "schema": "cppmega_h200_production_batch_v1",
        "status": "verified",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "batch": batch_summary,
        "structure": structure_summary,
        "active_graph": active_graph,
    }
    _write_json_atomic(output, receipt)
    return receipt
