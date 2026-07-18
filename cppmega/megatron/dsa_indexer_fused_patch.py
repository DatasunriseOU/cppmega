"""Per-head fused accumulation for Megatron DSA ``_compute_index_scores``.

Upstream Megatron's ``_compute_index_scores``
(``megatron/core/transformer/experimental_attention_variant/dsa.py``) is::

    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())
    if use_relu: index_scores = torch.relu(index_scores)
    index_scores = index_scores * weights.unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)
    index_scores = index_scores.transpose(0, 1)

The intermediate ``[sq, b, h, sk]`` FP32 tensor is ``sq*b*h*sk*4`` bytes.
For NAM56R (``sq=sk=4096``, ``h=index_n_heads=32``, ``b=MBS=8``) that is
**16 GiB live** inside every indexer call. It is allocated, reduced over
``h``, then discarded — a textbook fuse-reduction opportunity.

This patch replaces the implementation with a per-head loop that
accumulates directly into a ``[b, sq, sk]`` FP32 buffer (268 MiB at the
same shape, a ~60x reduction) and **never materialises the full
``[sq, b, h, sk]`` tensor**. Math is identical to the upstream einsum
except for FP32 reduction order (head-wise instead of vectorised) which
is exact up to associative FP32 reorder (< 1e-6 relative error on
bounded inputs).

The fused score path is numerically equivalent to upstream and has no
runtime penalty on H200 (the per-head bmm lowers to a single cuBLAS GEMM
per head, same FLOP count). It replaces the dead
``dsa_fp8_patch.py`` (deleted 2026-04-13 in commit ``b6fb886``) for
memory-bound configurations like DSA 9+4 with MBS >= 8.

Applies to ALL callers of ``_compute_index_scores``:

* ``fused_qk_topk_naive`` (fwd, feeds topk selection)
* ``bwd_fused_indexer_loss_naive`` (bwd, indexer recompute)
* ``_LemyxFusedDSAIndexerLoss.apply`` (fwd via ``fused_qk_topk_naive``)
* ``IndexCache`` Full-layer ``topk`` re-derivation
  (``fused_qk_topk_naive`` in ``index_cache_patch.py``)

Usage::

    from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
    apply_dsa_indexer_fused_patch()  # call after Megatron imports, before training

Applied unconditionally. This patch is mandatory for production at
MBS=10 NAM56R — upstream ``_compute_index_scores`` materialises
``[sq, b, h, sk]`` fp32 = ~5 GiB/layer held by autograd for backward;
9 DSA layers => ~45 GiB resident.  Fused per-head ``[b, sq, sk]`` =
640 MiB/layer => ~5.7 GiB resident.  Net save ~40 GiB; without the fused
path MBS=10 is impossible, period.  No env gate — install or raise.

When the graph auxiliary objective is enabled, the patch also composes
graph loss into Megatron's fused indexer loss and recomputes its gradients
inside the custom backward. The exact forward attention mask and graph
sidecars are captured per microbatch and reused during backward.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
from contextvars import ContextVar, Token

import torch

log = logging.getLogger(__name__)

__all__ = [
    "apply_dsa_indexer_fused_patch",
    "build_graph_objective_tensors",
    "build_graph_route_bias_from_structure_batch",
    "compute_index_scores_fused_bf16",
]

_PATCH_MARKER = "__cppmega_dsa_indexer_fused_patched__"
_AUTOGRAD_PATCH_MARKER = "__cppmega_graph_microbatch_patched__"
_GRAPH_OBJECTIVE_PATCH_MARKER = "__cppmega_graph_objective_patched__"
_RUNTIME_RECEIPT_PATCH_MARKER = "__cppmega_dsa_runtime_receipts_patched__"
_NO_GRAPH_BATCH = object()
_NO_UPSTREAM_MASK = object()
_NO_DSA_LAYER = object()
_GRAPH_BATCH_OVERRIDE: ContextVar[object] = ContextVar(
    "cppmega_dsa_graph_batch_override", default=_NO_GRAPH_BATCH
)
_UPSTREAM_MASK_OVERRIDE: ContextVar[object] = ContextVar(
    "cppmega_dsa_upstream_mask_override", default=_NO_UPSTREAM_MASK
)
_DSA_LAYER_CONTEXT: ContextVar[object] = ContextVar(
    "cppmega_dsa_layer_context", default=_NO_DSA_LAYER
)
_GRAPH_OBJECTIVE_RECEIPT_LAYERS: set[int] = set()
_GRAPH_ROUTE_BATCH_KEYS = (
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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _qualified_name(value: object) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _emit_runtime_receipt(prefix: str, payload: dict[str, object]) -> None:
    print(
        f"{prefix} {json.dumps(payload, sort_keys=True, separators=(',', ':'))}",
        flush=True,
    )


def _as_batched_edges(
    structure_batch: dict[str, torch.Tensor],
    *,
    edge_key: str,
    count_key: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    edges = structure_batch.get(edge_key)
    counts = structure_batch.get(count_key)
    if edges is None and counts is None:
        return None
    if edges is None or counts is None:
        raise KeyError(
            f"graph route sidecar must provide both {edge_key!r} and "
            f"{count_key!r}; got edges={edges is not None} counts={counts is not None}"
        )
    if not isinstance(edges, torch.Tensor) or not isinstance(counts, torch.Tensor):
        raise TypeError(
            f"graph route sidecars {edge_key!r}/{count_key!r} must be torch.Tensor, "
            f"got {type(edges).__name__}/{type(counts).__name__}"
        )
    if edges.dim() == 2:
        edges = edges.unsqueeze(0)
    if edges.dim() != 3 or int(edges.shape[-1]) != 2:
        raise ValueError(
            f"{edge_key} must have shape [B,max_edges,2] or [max_edges,2], "
            f"got {tuple(edges.shape)}"
        )
    counts = counts.reshape(-1)
    if int(edges.shape[0]) not in (1, batch_size):
        raise ValueError(
            f"{edge_key} batch {int(edges.shape[0])} must be 1 or {batch_size}"
        )
    if int(counts.shape[0]) not in (1, batch_size):
        raise ValueError(
            f"{count_key} batch {int(counts.shape[0])} must be 1 or {batch_size}"
        )
    return edges.to(device=device, dtype=torch.long), counts.to(device=device, dtype=torch.long)


def _as_batched_edge_triples(
    structure_batch: dict[str, torch.Tensor],
    *,
    edge_key: str,
    count_key: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    edges = structure_batch.get(edge_key)
    counts = structure_batch.get(count_key)
    if edges is None and counts is None:
        return None
    if edges is None or counts is None:
        raise KeyError(
            f"domain graph route sidecar must provide both {edge_key!r} and "
            f"{count_key!r}; got edges={edges is not None} counts={counts is not None}"
        )
    if not isinstance(edges, torch.Tensor) or not isinstance(counts, torch.Tensor):
        raise TypeError(
            f"domain graph route sidecars {edge_key!r}/{count_key!r} must be "
            f"torch.Tensor, got {type(edges).__name__}/{type(counts).__name__}"
        )
    if edges.dim() == 2:
        edges = edges.unsqueeze(0)
    if edges.dim() != 3 or int(edges.shape[-1]) != 3:
        raise ValueError(
            f"{edge_key} must have shape [B,max_edges,3] or [max_edges,3], "
            f"got {tuple(edges.shape)}"
        )
    counts = counts.reshape(-1)
    if int(edges.shape[0]) not in (1, batch_size):
        raise ValueError(
            f"{edge_key} batch {int(edges.shape[0])} must be 1 or {batch_size}"
        )
    if int(counts.shape[0]) not in (1, batch_size):
        raise ValueError(
            f"{count_key} batch {int(counts.shape[0])} must be 1 or {batch_size}"
        )
    return edges.to(device=device, dtype=torch.long), counts.to(device=device, dtype=torch.long)


def _scatter_edges_(
    bias: torch.Tensor,
    edges: torch.Tensor,
    counts: torch.Tensor,
    *,
    weight: float,
    sq: int,
    sk: int,
    require_kind: bool,
) -> None:
    """Add ``weight`` at each declared edge's ``(src,dst)`` in ``bias`` [B,Sq,Sk].

    Fully vectorized: no per-sample Python loop and a SINGLE device sync (the
    validity check), instead of ~2*B host syncs per relation-kind per layer — real
    runs use micro-batch up to 192, so the old loop stalled the GPU thousands of
    times per step. ``counts[b]`` is the number of declared edges for sample ``b``
    (rest is padding); ``require_kind`` also requires ``edges[...,2] >= 0`` (triples).
    Duplicate ``(b,src,dst)`` accumulate via index_add_, identical to the old
    per-sample ``+=`` for the normal unique-edge case.
    """
    if weight == 0.0:
        return
    batch = int(bias.shape[0])
    max_edges = int(edges.shape[1])
    if int(edges.shape[0]) == 1 and batch > 1:
        edges = edges.expand(batch, -1, -1)
    if int(counts.shape[0]) == 1 and batch > 1:
        counts = counts.expand(batch)
    counts = counts.to(torch.long)  # non-inplace; not mutated below, so caller's tensor is safe
    # RULE #1: a declared count outside [0, max_edges] is corrupt sidecar metadata
    # (the padded edge tensor has exactly max_edges slots) -> raise, don't clamp it
    # into a valid-looking-but-wrong graph.
    if bool(((counts < 0) | (counts > max_edges)).any().item()):
        bad = counts[(counts < 0) | (counts > max_edges)][:8].tolist()
        raise ValueError(
            f"[cppmega-graph] edge counts out of range [0,{max_edges}]: {bad}"
        )
    if max_edges == 0:
        return  # validated above: with 0 edge slots every declared count must be 0
    active = torch.arange(max_edges, device=edges.device)[None, :] < counts[:, None]  # [B,max_edges]
    src = edges[..., 0]
    dst = edges[..., 1]
    in_range = (src >= 0) & (src < sq) & (dst >= 0) & (dst < sk)
    if require_kind:
        in_range = in_range & (edges[..., 2] >= 0)
    bad = active & ~in_range
    # RULE #1: a declared (active) edge out of range is corrupt graph metadata,
    # not padding -> raise, don't drop. One sync for the whole batch.
    if bool(bad.any().item()):
        bi, ei = torch.nonzero(bad, as_tuple=True)
        sample = edges[bi[:8], ei[:8]].tolist()
        raise ValueError(
            f"[cppmega-graph] {int(bad.sum().item())} declared graph edges out of "
            f"range for (sq={sq}, sk={sk}): {sample}"
        )
    bidx = torch.nonzero(active, as_tuple=True)[0]
    s = src[active]
    d = dst[active]
    lin = (bidx * sq + s) * sk + d  # int64 (bidx is long) -> safe for large B*Sq*Sk
    bias.view(-1).index_add_(0, lin, bias.new_full((int(lin.numel()),), float(weight)))


def _scatter_relation_edges_(
    bias: torch.Tensor,
    edges: torch.Tensor,
    counts: torch.Tensor,
    *,
    weight: float,
    sq: int,
    sk: int,
) -> None:
    _scatter_edges_(bias, edges, counts, weight=weight, sq=sq, sk=sk, require_kind=False)


def _as_batched_chunks(
    structure_batch: dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    starts = structure_batch.get("graph_chunk_starts")
    ends = structure_batch.get("graph_chunk_ends")
    counts = structure_batch.get("graph_chunk_counts")
    if starts is None or ends is None or counts is None:
        raise KeyError(
            "chunk-index call/type routes require graph_chunk_starts, "
            "graph_chunk_ends, and graph_chunk_counts"
        )
    if not all(isinstance(value, torch.Tensor) for value in (starts, ends, counts)):
        raise TypeError("graph chunk sidecars must be torch.Tensor")
    if starts.dim() == 1:
        starts = starts.unsqueeze(0)
    if ends.dim() == 1:
        ends = ends.unsqueeze(0)
    if starts.dim() != 2 or ends.shape != starts.shape:
        raise ValueError(
            f"graph chunk starts/ends must have matching [B,C] shape, got "
            f"{tuple(starts.shape)}/{tuple(ends.shape)}"
        )
    counts = counts.reshape(-1)
    if int(starts.shape[0]) not in (1, batch_size):
        raise ValueError(f"graph chunk batch {int(starts.shape[0])} must be 1 or {batch_size}")
    if int(counts.shape[0]) not in (1, batch_size):
        raise ValueError(f"graph chunk counts batch must be 1 or {batch_size}")
    if int(starts.shape[0]) == 1 and batch_size > 1:
        starts = starts.expand(batch_size, -1)
        ends = ends.expand(batch_size, -1)
    if int(counts.shape[0]) == 1 and batch_size > 1:
        counts = counts.expand(batch_size)
    starts = starts.to(device=device, dtype=torch.long)
    ends = ends.to(device=device, dtype=torch.long)
    counts = counts.to(device=device, dtype=torch.long)
    max_chunks = int(starts.shape[1])
    if bool(((counts < 0) | (counts > max_chunks)).any().item()):
        raise ValueError(f"graph chunk counts out of range [0,{max_chunks}]")
    return starts, ends, counts


def _token_chunk_map(
    starts: torch.Tensor,
    ends: torch.Tensor,
    counts: torch.Tensor,
    *,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, max_chunks = starts.shape
    if max_chunks == 0:
        return (
            torch.zeros((batch, length), dtype=torch.long, device=starts.device),
            torch.zeros((batch, length), dtype=torch.bool, device=starts.device),
        )
    slots = torch.arange(max_chunks, device=starts.device).unsqueeze(0)
    active = slots < counts.unsqueeze(1)
    invalid = active & ((starts < 0) | (ends <= starts) | (ends > length))
    if bool(invalid.any().item()):
        raise ValueError("active graph chunk span is outside the sample token range")
    if max_chunks > 1:
        ordered = active[:, 1:] & active[:, :-1] & (starts[:, 1:] < ends[:, :-1])
        if bool(ordered.any().item()):
            raise ValueError("active graph chunk spans overlap or are out of order")

    searchable = torch.where(active, starts, torch.full_like(starts, length + 1))
    positions = torch.arange(length, device=starts.device).unsqueeze(0).expand(batch, -1)
    chunk_ids = torch.searchsorted(searchable.contiguous(), positions.contiguous(), right=True) - 1
    safe_ids = chunk_ids.clamp(0, max_chunks - 1)
    selected_starts = starts.gather(1, safe_ids)
    selected_ends = ends.gather(1, safe_ids)
    valid = (
        (chunk_ids >= 0)
        & (chunk_ids < counts.unsqueeze(1))
        & (positions >= selected_starts)
        & (positions < selected_ends)
    )
    return safe_ids, valid


def _scatter_chunk_relation_edges_(
    bias: torch.Tensor,
    edges: torch.Tensor,
    counts: torch.Tensor,
    *,
    starts: torch.Tensor,
    ends: torch.Tensor,
    chunk_counts: torch.Tensor,
    weight: float,
    sq: int,
    sk: int,
) -> None:
    """Expand chunk-index relations into token-span blocks in ``bias``."""

    if weight == 0.0:
        return
    batch = int(bias.shape[0])
    max_edges = int(edges.shape[1])
    max_chunks = int(starts.shape[1])
    if int(edges.shape[0]) == 1 and batch > 1:
        edges = edges.expand(batch, -1, -1)
    if int(counts.shape[0]) == 1 and batch > 1:
        counts = counts.expand(batch)
    if bool(((counts < 0) | (counts > max_edges)).any().item()):
        raise ValueError(f"graph edge counts out of range [0,{max_edges}]")
    active = torch.arange(max_edges, device=edges.device).unsqueeze(0) < counts.unsqueeze(1)
    src = edges[..., 0]
    dst = edges[..., 1]
    valid_endpoint = (
        (src >= 0)
        & (dst >= 0)
        & (src < chunk_counts.unsqueeze(1))
        & (dst < chunk_counts.unsqueeze(1))
    )
    if bool((active & ~valid_endpoint).any().item()):
        raise ValueError("declared call/type edge references an unavailable chunk")
    if not bool(active.any().item()):
        return

    adjacency = bias.new_zeros((batch, max_chunks, max_chunks))
    bidx = torch.nonzero(active, as_tuple=True)[0]
    lin = (bidx * max_chunks + src[active]) * max_chunks + dst[active]
    adjacency.view(-1).index_add_(
        0, lin, adjacency.new_full((int(lin.numel()),), float(weight))
    )
    q_chunks, q_valid = _token_chunk_map(starts, ends, chunk_counts, length=sq)
    k_chunks, k_valid = _token_chunk_map(starts, ends, chunk_counts, length=sk)
    rows = adjacency.gather(
        1, q_chunks.unsqueeze(-1).expand(-1, -1, max_chunks)
    )
    block_bias = rows.gather(2, k_chunks.unsqueeze(1).expand(-1, sq, -1))
    block_bias.masked_fill_(~(q_valid.unsqueeze(2) & k_valid.unsqueeze(1)), 0)
    bias.add_(block_bias)


def _scatter_relation_edge_triples_(
    bias: torch.Tensor,
    edges: torch.Tensor,
    counts: torch.Tensor,
    *,
    weight: float,
    sq: int,
    sk: int,
) -> None:
    _scatter_edges_(bias, edges, counts, weight=weight, sq=sq, sk=sk, require_kind=True)


def build_graph_route_bias_from_structure_batch(
    structure_batch: dict[str, torch.Tensor] | None,
    *,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    call_weight: float = 1.0,
    type_weight: float = 1.0,
    domain_weight: float = 1.0,
    build_weight: float = 1.0,
    shell_weight: float = 1.0,
    diagnostic_weight: float = 1.0,
    cross_domain_weight: float = 1.0,
    consumer: str | None = None,
) -> torch.Tensor:
    """Build ``S_graph[b,t,s]`` from cppmega graph route sidecars.

    ``graph_call_edges`` / ``graph_type_edges`` are chunk-index pairs.  They are
    expanded through ``graph_chunk_starts/ends`` into token-span blocks. Domain,
    build, shell, diagnostic, and cross-domain triples are token-position edges.
    It intentionally raises on missing/malformed route sidecars: graph-routed
    cppmega models must not silently become token-only.
    """

    if structure_batch is None:
        raise RuntimeError(
            "CPPMEGA_GRAPH_ROUTES_ENABLED=1 but no current cppmega structure "
            "batch is available; refusing token-only DSA indexer"
        )
    if batch_size <= 0 or seqlen_q <= 0 or seqlen_k <= 0:
        raise ValueError(
            "build_graph_route_bias_from_structure_batch: batch/seqlen must be "
            f"positive, got B={batch_size} Sq={seqlen_q} Sk={seqlen_k}"
        )

    bias = torch.zeros(
        (batch_size, seqlen_q, seqlen_k), device=device, dtype=dtype
    )
    seen_relation = False
    chunk_layout: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for edge_key, count_key, weight in (
        ("graph_call_edges", "graph_call_edge_counts", call_weight),
        ("graph_type_edges", "graph_type_edge_counts", type_weight),
    ):
        relation = _as_batched_edges(
            structure_batch,
            edge_key=edge_key,
            count_key=count_key,
            batch_size=batch_size,
            device=device,
        )
        if relation is None:
            continue
        seen_relation = True
        edges, counts = relation
        if chunk_layout is None:
            chunk_layout = _as_batched_chunks(
                structure_batch, batch_size=batch_size, device=device
            )
        starts, ends, chunk_counts = chunk_layout
        _scatter_chunk_relation_edges_(
            bias,
            edges,
            counts,
            starts=starts,
            ends=ends,
            chunk_counts=chunk_counts,
            weight=weight,
            sq=seqlen_q,
            sk=seqlen_k,
        )
    for edge_key, count_key, weight in (
        ("graph_domain_edges", "graph_domain_edge_counts", domain_weight),
        ("graph_build_edges", "graph_build_edge_counts", build_weight),
        ("graph_shell_edges", "graph_shell_edge_counts", shell_weight),
        ("graph_diagnostic_edges", "graph_diagnostic_edge_counts", diagnostic_weight),
        ("graph_cross_domain_edges", "graph_cross_domain_edge_counts", cross_domain_weight),
    ):
        relation = _as_batched_edge_triples(
            structure_batch,
            edge_key=edge_key,
            count_key=count_key,
            batch_size=batch_size,
            device=device,
        )
        if relation is None:
            continue
        seen_relation = True
        edges, counts = relation
        _scatter_relation_edge_triples_(
            bias,
            edges,
            counts,
            weight=weight,
            sq=seqlen_q,
            sk=seqlen_k,
        )
    if not seen_relation:
        raise KeyError(
            "CPPMEGA_GRAPH_ROUTES_ENABLED=1 but structure batch has no graph "
            "route edge tensors (expected graph_call_edges/type_edges or "
            "domain/build/shell/diagnostic/cross-domain edges)"
        )
    receipt_path = os.environ.get("CPPMEGA_H200_GRAPH_PRIOR_RECEIPT")
    if receipt_path and consumer is not None:
        from cppmega.megatron.h200_preflight import observe_graph_prior

        observe_graph_prior(
            prior=bias,
            consumer=consumer,
            receipt_path=receipt_path,
        )
    return bias


def _current_graph_route_bias(
    *,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
) -> torch.Tensor:
    override = _GRAPH_BATCH_OVERRIDE.get()
    if override is _NO_GRAPH_BATCH:
        try:
            from cppmega.megatron.structure_dataset_patch import (
                _get_current_structure_batch,
            )
        except Exception as exc:  # pragma: no cover - remote Megatron environment
            raise RuntimeError(
                "CPPMEGA_GRAPH_ROUTES_ENABLED=1 but structure_dataset_patch is not "
                "importable; import it before applying the DSA indexer patch"
            ) from exc
        structure_batch = _get_current_structure_batch()
    else:
        structure_batch = override
    return build_graph_route_bias_from_structure_batch(
        structure_batch,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        device=device,
        dtype=torch.float32,
        call_weight=_env_float("CPPMEGA_DSA_GRAPH_CALL_WEIGHT", 1.0),
        type_weight=_env_float("CPPMEGA_DSA_GRAPH_TYPE_WEIGHT", 1.0),
        domain_weight=_env_float("CPPMEGA_DSA_GRAPH_DOMAIN_WEIGHT", 1.0),
        build_weight=_env_float("CPPMEGA_DSA_GRAPH_BUILD_WEIGHT", 1.0),
        shell_weight=_env_float("CPPMEGA_DSA_GRAPH_SHELL_WEIGHT", 1.0),
        diagnostic_weight=_env_float("CPPMEGA_DSA_GRAPH_DIAGNOSTIC_WEIGHT", 1.0),
        cross_domain_weight=_env_float("CPPMEGA_DSA_GRAPH_CROSS_DOMAIN_WEIGHT", 1.0),
        consumer="dsa_indexer",
    )


def _active_graph_structure_batch() -> dict[str, torch.Tensor]:
    override = _GRAPH_BATCH_OVERRIDE.get()
    if override is _NO_GRAPH_BATCH:
        try:
            from cppmega.megatron.structure_dataset_patch import (
                _get_current_structure_batch,
            )
        except Exception as exc:  # pragma: no cover - remote Megatron environment
            raise RuntimeError(
                "CPPMEGA_GRAPH_ROUTES_ENABLED=1 but structure_dataset_patch is not "
                "available"
            ) from exc
        current = _get_current_structure_batch()
    else:
        current = override
    if current is None or not isinstance(current, dict):
        raise RuntimeError(
            "CPPMEGA_GRAPH_ROUTES_ENABLED=1 but no active structure batch is "
            "available for graph auxiliary loss"
        )
    return current


def _graph_target_bias(
    structure_batch: dict[str, torch.Tensor],
    *,
    relations: tuple[str, ...],
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
) -> torch.Tensor:
    relation_keys = {
        "call": ("graph_call_edges", "graph_call_edge_counts"),
        "type": ("graph_type_edges", "graph_type_edge_counts"),
        "domain": ("graph_domain_edges", "graph_domain_edge_counts"),
        "build": ("graph_build_edges", "graph_build_edge_counts"),
        "shell": ("graph_shell_edges", "graph_shell_edge_counts"),
        "diagnostic": (
            "graph_diagnostic_edges",
            "graph_diagnostic_edge_counts",
        ),
        "cross_domain": (
            "graph_cross_domain_edges",
            "graph_cross_domain_edge_counts",
        ),
    }
    unknown = sorted(set(relations) - set(relation_keys))
    if unknown:
        raise ValueError(f"unsupported graph auxiliary relations: {unknown}")
    missing_relations = {
        relation: [
            key for key in relation_keys[relation] if key not in structure_batch
        ]
        for relation in relations
    }
    missing_relations = {
        relation: keys for relation, keys in missing_relations.items() if keys
    }
    if missing_relations:
        raise KeyError(
            "graph auxiliary loss is missing required relation sidecars: "
            f"{missing_relations}"
        )
    selected: dict[str, torch.Tensor] = {}
    for relation in relations:
        for key in relation_keys[relation]:
            if key in structure_batch:
                selected[key] = structure_batch[key]
    required_chunk_keys = (
        "graph_chunk_starts",
        "graph_chunk_ends",
        "graph_chunk_counts",
    )
    if set(relations) & {"call", "type"}:
        missing_chunks = [
            key for key in required_chunk_keys if key not in structure_batch
        ]
        if missing_chunks:
            raise KeyError(
                "call/type graph auxiliary loss is missing required chunk "
                f"sidecars: {missing_chunks}"
            )
    for key in required_chunk_keys:
        if key in structure_batch:
            selected[key] = structure_batch[key]
    return build_graph_route_bias_from_structure_batch(
        selected,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        device=device,
        dtype=torch.float32,
    )


def build_graph_objective_tensors(
    structure_batch: dict[str, torch.Tensor],
    *,
    relations: tuple[str, ...],
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
    upstream_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build token-expanded targets and the exact trainable graph pair mask."""

    targets = _graph_target_bias(
        structure_batch,
        relations=relations,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        device=device,
    ).gt(0)
    document_ids = structure_batch.get("graph_document_ids")
    if not isinstance(document_ids, torch.Tensor):
        raise KeyError(
            "graph auxiliary loss requires graph_document_ids derived from "
            "packed document boundaries"
        )
    if document_ids.dim() == 1:
        document_ids = document_ids.unsqueeze(0)
    if document_ids.dim() != 2:
        raise ValueError(
            "graph_document_ids must have shape [B,S] or [S], got "
            f"{tuple(document_ids.shape)}"
        )
    if int(document_ids.shape[0]) == 1 and batch_size > 1:
        document_ids = document_ids.expand(batch_size, -1)
    if int(document_ids.shape[0]) != batch_size:
        raise ValueError(
            f"graph_document_ids batch {int(document_ids.shape[0])} != {batch_size}"
        )
    if int(document_ids.shape[1]) < max(seqlen_q, seqlen_k):
        raise ValueError(
            f"graph_document_ids length {int(document_ids.shape[1])} is shorter "
            f"than graph score shape ({seqlen_q}, {seqlen_k})"
        )
    document_ids = document_ids.to(device=device, dtype=torch.long)
    query_docs = document_ids[:, :seqlen_q]
    key_docs = document_ids[:, :seqlen_k]
    positive_docs = (query_docs > 0).unsqueeze(2) & (key_docs > 0).unsqueeze(1)
    same_document = query_docs.unsqueeze(2) == key_docs.unsqueeze(1)

    query_positions = torch.arange(seqlen_q, device=device)[:, None]
    key_positions = torch.arange(seqlen_k, device=device)[None, :]
    pair_mask = (key_positions <= query_positions).unsqueeze(0)
    pair_mask = pair_mask.expand(batch_size, -1, -1)
    pair_mask = pair_mask & positive_docs & same_document

    if upstream_mask is not None:
        if not isinstance(upstream_mask, torch.Tensor):
            raise TypeError(
                f"DSA upstream mask must be torch.Tensor, got "
                f"{type(upstream_mask).__name__}"
            )
        mask = upstream_mask.to(device=device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3 or tuple(mask.shape[-2:]) != (seqlen_q, seqlen_k):
            raise ValueError(
                "DSA upstream mask must have shape [Q,K] or [B,Q,K], got "
                f"{tuple(mask.shape)}"
            )
        if int(mask.shape[0]) == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        if int(mask.shape[0]) != batch_size:
            raise ValueError(
                f"DSA upstream mask batch {int(mask.shape[0])} != {batch_size}"
            )
        if mask.dtype == torch.bool:
            upstream_valid = mask
        else:
            upstream_valid = torch.isfinite(mask) & (mask >= 0)
        pair_mask = pair_mask & upstream_valid

    targets = targets & pair_mask
    return targets, pair_mask


def _graph_objective_from_index_scores(
    index_scores: torch.Tensor,
    *,
    upstream_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute graph BCE/coverage on neural DSA scores, excluding fixed bias."""

    from cppmega.megatron.graph_objective_loss import (
        GraphAuxiliaryLossConfig,
        graph_auxiliary_loss,
    )

    if index_scores.ndim != 3:
        raise ValueError(
            f"DSA graph objective requires (B,Q,K) index scores, got "
            f"{tuple(index_scores.shape)}"
        )
    config = GraphAuxiliaryLossConfig.from_env()
    batch, seqlen_q, seqlen_k = (int(value) for value in index_scores.shape)
    structure_batch = _active_graph_structure_batch()
    targets, pair_mask = build_graph_objective_tensors(
        structure_batch,
        relations=config.relations,
        batch_size=batch,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        device=index_scores.device,
        upstream_mask=upstream_mask,
    )
    route_bias = _current_graph_route_bias(
        batch_size=batch,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        device=index_scores.device,
    )
    neural_scores = index_scores.float() - config.bias_beta * route_bias
    pair_mask = pair_mask & torch.isfinite(neural_scores)
    graph_loss, _components = graph_auxiliary_loss(
        neural_scores,
        targets.to(dtype=neural_scores.dtype),
        pair_mask=pair_mask,
        config=config,
    )
    return graph_loss


def _bind_upstream_mask(callable_obj, *args, **kwargs) -> torch.Tensor | None:
    try:
        bound = inspect.signature(callable_obj).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    mask = bound.arguments.get("mask")
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            f"Megatron FusedDSAIndexerLoss mask must be torch.Tensor, got "
            f"{type(mask).__name__}"
        )
    return mask.detach()


def _patch_dsa_graph_objective(dsa_mod) -> None:
    """Compose graph supervision into Megatron's autograd-carried DSA loss."""

    existing = getattr(dsa_mod, "compute_dsa_indexer_loss", None)
    if existing is None:
        raise RuntimeError("Megatron DSA compute_dsa_indexer_loss is unavailable")
    if getattr(existing, _GRAPH_OBJECTIVE_PATCH_MARKER, False):
        return

    def compute_dsa_indexer_loss_with_graph(index_scores, *args, **kwargs):
        indexer_loss = existing(index_scores, *args, **kwargs)
        if not _env_flag("CPPMEGA_DSA_GRAPH_AUX_ENABLED"):
            return indexer_loss
        if not _env_flag("CPPMEGA_GRAPH_ROUTES_ENABLED"):
            raise RuntimeError(
                "DSA graph auxiliary objective requires graph routes to be enabled"
            )
        upstream_mask = kwargs.get("mask")
        if upstream_mask is None:
            try:
                bound = inspect.signature(existing).bind_partial(
                    index_scores, *args, **kwargs
                )
            except (TypeError, ValueError):
                bound = None
            if bound is not None:
                upstream_mask = bound.arguments.get("mask")
        if upstream_mask is None:
            captured_mask = _UPSTREAM_MASK_OVERRIDE.get()
            if captured_mask is not _NO_UPSTREAM_MASK:
                upstream_mask = captured_mask
        return indexer_loss + _graph_objective_from_index_scores(
            index_scores, upstream_mask=upstream_mask
        )

    setattr(
        compute_dsa_indexer_loss_with_graph,
        _GRAPH_OBJECTIVE_PATCH_MARKER,
        True,
    )
    dsa_mod.compute_dsa_indexer_loss = compute_dsa_indexer_loss_with_graph

    sparse_existing = getattr(dsa_mod, "compute_dsa_indexer_loss_topk_sparse", None)
    if sparse_existing is not None and not getattr(
        sparse_existing, _GRAPH_OBJECTIVE_PATCH_MARKER, False
    ):
        def sparse_loss_requires_dense_graph_scores(*args, **kwargs):
            if _env_flag("CPPMEGA_DSA_GRAPH_AUX_ENABLED"):
                raise RuntimeError(
                    "graph auxiliary loss requires full DSA indexer scores; "
                    "top-k-only sparse indexer loss cannot satisfy the objective "
                    "contract"
                )
            return sparse_existing(*args, **kwargs)

        setattr(
            sparse_loss_requires_dense_graph_scores,
            _GRAPH_OBJECTIVE_PATCH_MARKER,
            True,
        )
        dsa_mod.compute_dsa_indexer_loss_topk_sparse = (
            sparse_loss_requires_dense_graph_scores
        )


def _capture_current_graph_batch() -> dict[str, torch.Tensor]:
    try:
        from cppmega.megatron.structure_dataset_patch import _get_current_structure_batch
    except Exception as exc:  # pragma: no cover - remote Megatron environment
        raise RuntimeError(
            "graph-routed DSA autograd requires structure_dataset_patch"
        ) from exc
    current = _get_current_structure_batch()
    if current is None:
        raise RuntimeError(
            "CPPMEGA_GRAPH_ROUTES_ENABLED=1 but no structure batch is active "
            "while entering FusedDSAIndexerLoss"
        )
    captured = {
        key: value.detach().clone()
        for key in _GRAPH_ROUTE_BATCH_KEYS
        if (value := current.get(key)) is not None
    }
    if not captured:
        raise KeyError("active structure batch contains no graph route tensors")
    return captured


def _set_graph_batch_override(batch: dict[str, torch.Tensor]) -> Token:
    return _GRAPH_BATCH_OVERRIDE.set(batch)


def _reset_graph_batch_override(token: Token) -> None:
    _GRAPH_BATCH_OVERRIDE.reset(token)


def _install_indexer_gradient_receipts(indexer) -> None:
    if not _env_flag("CPPMEGA_H200_DSA_GRAPH_RECEIPTS"):
        return
    if getattr(indexer, _RUNTIME_RECEIPT_PATCH_MARKER, False):
        return
    parameters = tuple(
        (name, parameter)
        for name, parameter in indexer.named_parameters()
        if parameter.requires_grad
    )
    if not parameters:
        raise RuntimeError("DSAIndexer exposes no trainable parameters for receipts")

    seen_norms: dict[str, float] = {}
    handles = []

    def hook_for(name: str):
        def record_gradient(gradient: torch.Tensor) -> torch.Tensor:
            if getattr(indexer, "cppmega_dsa_gradient_receipt_emitted", False):
                return gradient
            norm = float(
                torch.linalg.vector_norm(gradient.detach().float()).cpu().item()
            )
            seen_norms[name] = norm
            if len(seen_norms) == len(parameters):
                total_norm = math.sqrt(
                    sum(value * value for value in seen_norms.values())
                )
                _emit_runtime_receipt(
                    "CPPMEGA_DSA_INDEXER_GRAD",
                    {
                        "layer_number": getattr(
                            indexer, "cppmega_dsa_layer_number", None
                        ),
                        "actual_indexer_module": _qualified_name(indexer),
                        "grad_norm": total_norm,
                        "parameter_grad_norms": dict(sorted(seen_norms.items())),
                    },
                )
                indexer.cppmega_dsa_gradient_receipt_emitted = True
                seen_norms.clear()
            return gradient

        return record_gradient

    for name, parameter in parameters:
        handles.append(parameter.register_hook(hook_for(name)))
    indexer.cppmega_dsa_gradient_receipt_handles = tuple(handles)
    setattr(indexer, _RUNTIME_RECEIPT_PATCH_MARKER, True)


def _patch_dsa_runtime_receipts(dsa_mod) -> None:
    dsa_attention = getattr(dsa_mod, "DSAttention", None)
    dsa_indexer = getattr(dsa_mod, "DSAIndexer", None)
    if dsa_attention is None or dsa_indexer is None:
        raise RuntimeError("Megatron DSA runtime modules are unavailable")
    if getattr(dsa_attention, _RUNTIME_RECEIPT_PATCH_MARKER, False):
        return

    original_attention_init = dsa_attention.__init__
    original_attention_forward = dsa_attention.forward
    original_indexer_init = dsa_indexer.__init__

    def indexer_init(self, *args, **kwargs):
        original_indexer_init(self, *args, **kwargs)
        _install_indexer_gradient_receipts(self)

    def attention_init(self, *args, **kwargs):
        original_attention_init(self, *args, **kwargs)
        self.indexer.cppmega_dsa_layer_number = int(self.layer_number)

    def attention_forward(self, *args, **kwargs):
        token = _DSA_LAYER_CONTEXT.set(int(self.layer_number))
        try:
            return original_attention_forward(self, *args, **kwargs)
        finally:
            _DSA_LAYER_CONTEXT.reset(token)

    dsa_indexer.__init__ = indexer_init
    dsa_attention.__init__ = attention_init
    dsa_attention.forward = attention_forward
    setattr(dsa_attention, _RUNTIME_RECEIPT_PATCH_MARKER, True)


def _patch_fused_dsa_autograd(dsa_mod) -> None:
    fused_loss = getattr(dsa_mod, "FusedDSAIndexerLoss", None)
    if fused_loss is None:
        raise RuntimeError("Megatron DSA FusedDSAIndexerLoss is unavailable")
    if getattr(fused_loss, _AUTOGRAD_PATCH_MARKER, False):
        return
    original_forward = fused_loss.forward
    original_backward = fused_loss.backward

    def forward(ctx, *args, **kwargs):
        captured = None
        token = None
        upstream_mask = None
        mask_token = None
        if _env_flag("CPPMEGA_GRAPH_ROUTES_ENABLED"):
            captured = _capture_current_graph_batch()
            token = _set_graph_batch_override(captured)
        if _env_flag("CPPMEGA_DSA_GRAPH_AUX_ENABLED"):
            upstream_mask = _bind_upstream_mask(
                original_forward,
                ctx,
                *args,
                **kwargs,
            )
            mask_token = _UPSTREAM_MASK_OVERRIDE.set(upstream_mask)
        ctx.cppmega_graph_route_batch = captured
        ctx.cppmega_dsa_upstream_mask = upstream_mask
        layer_number = _DSA_LAYER_CONTEXT.get()
        ctx.cppmega_dsa_layer_number = (
            None if layer_number is _NO_DSA_LAYER else int(layer_number)
        )
        try:
            return original_forward(ctx, *args, **kwargs)
        finally:
            if mask_token is not None:
                _UPSTREAM_MASK_OVERRIDE.reset(mask_token)
            if token is not None:
                _reset_graph_batch_override(token)

    def backward(ctx, *args, **kwargs):
        captured = getattr(ctx, "cppmega_graph_route_batch", None)
        token = None
        if captured is not None:
            token = _set_graph_batch_override(captured)
        try:
            original_grads = original_backward(ctx, *args, **kwargs)
            if not _env_flag("CPPMEGA_DSA_GRAPH_AUX_ENABLED"):
                return original_grads
            grad_loss = args[1] if len(args) > 1 else kwargs.get("grad_loss")
            if grad_loss is None:
                return original_grads
            if not isinstance(original_grads, tuple) or len(original_grads) < 3:
                raise RuntimeError(
                    "Megatron FusedDSAIndexerLoss backward returned an incompatible "
                    "gradient tuple"
                )

            q, weights, k = ctx.saved_tensors[:3]
            with torch.enable_grad():
                q_recompute = q.detach().requires_grad_(True)
                weights_recompute = weights.detach().requires_grad_(True)
                k_recompute = k.detach().requires_grad_(True)
                neural_and_route_scores = dsa_mod._compute_index_scores(
                    q_recompute,
                    weights_recompute,
                    k_recompute,
                    use_relu=bool(getattr(ctx, "use_relu", True)),
                )
                graph_loss = _graph_objective_from_index_scores(
                    neural_and_route_scores,
                    upstream_mask=getattr(ctx, "cppmega_dsa_upstream_mask", None),
                )
                graph_grads = torch.autograd.grad(
                    graph_loss,
                    (q_recompute, weights_recompute, k_recompute),
                    grad_outputs=grad_loss,
                    allow_unused=True,
                )

            layer_number = getattr(ctx, "cppmega_dsa_layer_number", None)
            if (
                _env_flag("CPPMEGA_H200_DSA_GRAPH_RECEIPTS")
                and isinstance(layer_number, int)
                and layer_number not in _GRAPH_OBJECTIVE_RECEIPT_LAYERS
            ):
                from cppmega.megatron.graph_objective_loss import (
                    GraphAuxiliaryLossConfig,
                )

                _emit_runtime_receipt(
                    "CPPMEGA_DSA_GRAPH_OBJECTIVE",
                    {
                        "layer_number": layer_number,
                        "actual_dsa_module": _qualified_name(dsa_mod.DSAttention),
                        "effective_coefficient": (
                            GraphAuxiliaryLossConfig.from_env().indexer_weight
                        ),
                        "graph_loss": float(graph_loss.detach().float().item()),
                    },
                )
                _GRAPH_OBJECTIVE_RECEIPT_LAYERS.add(layer_number)

            combined_grads = list(original_grads)
            for index, graph_grad in enumerate(graph_grads):
                if graph_grad is None or not ctx.needs_input_grad[index]:
                    continue
                original_grad = combined_grads[index]
                if original_grad is None:
                    combined_grads[index] = graph_grad.to(
                        dtype=(q, weights, k)[index].dtype
                    )
                else:
                    combined_grads[index] = original_grad + graph_grad.to(
                        dtype=original_grad.dtype
                    )
            return tuple(combined_grads)
        finally:
            if token is not None:
                _reset_graph_batch_override(token)

    fused_loss.forward = staticmethod(forward)
    fused_loss.backward = staticmethod(backward)
    setattr(fused_loss, _AUTOGRAD_PATCH_MARKER, True)


def compute_index_scores_fused_bf16(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    use_relu: bool = True,
    *,
    graph_bias: torch.Tensor | None = None,
    graph_beta: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Drop-in replacement for Megatron DSA ``_compute_index_scores`` (BF16).

    Per-head fused accumulation: never materialises the ``[sq, b, h, sk]``
    FP32 intermediate.  Math is identical to the upstream einsum modulo
    FP32 associative reorder (head-wise).

    Args:
        q: ``[seqlen_q, batch, index_n_heads, index_head_dim]``.
        weights: ``[seqlen_q, batch, index_n_heads]``.
        k: ``[seqlen_k, batch, index_head_dim]``.
        use_relu: match upstream's ``use_relu`` flag.
        graph_bias: optional dense ``S_graph`` prior ``[batch, seqlen_q, seqlen_k]``.
        graph_beta: scalar multiplier for ``graph_bias``.

    Returns:
        ``[batch, seqlen_q, seqlen_k]`` FP32 index scores.
    """

    assert q.dim() == 4, f"q must be [sq,b,h,d], got {tuple(q.shape)}"
    assert k.dim() == 3, f"k must be [sk,b,d], got {tuple(k.shape)}"
    assert weights.dim() == 3, f"weights must be [sq,b,h], got {tuple(weights.shape)}"

    sq, b, h, d = q.shape
    sk, bk, dk = k.shape
    assert bk == b and dk == d, (
        f"shape mismatch q={tuple(q.shape)} k={tuple(k.shape)}"
    )

    # Accumulator: [b, sq, sk] fp32.  This is the final output shape; we
    # write into it head by head instead of building [sq, b, h, sk] first.
    index_scores = torch.zeros(
        (b, sq, sk), dtype=torch.float32, device=q.device
    )

    # Per-batch permutation of k: [sk, b, d] -> [b, d, sk] fp32 for bmm.
    # Reused across all heads of the same batch; cost = 1x [b, d, sk] fp32
    # (= ~4 MiB at production shape).
    k_f32 = k.float()  # [sk, b, d]
    k_bds = k_f32.permute(1, 2, 0).contiguous()  # [b, d, sk]
    del k_f32

    # Per-head: do one [b, sq, d] @ [b, d, sk] bmm -> [b, sq, sk] fp32.
    # Apply relu + weight + accumulate in place.  Working buffer ~ 268 MiB
    # at production shape (b=8, sq=sk=4096) vs 16 GiB for the full
    # [sq, b, h, sk] upstream intermediate.
    for hi in range(h):
        # [sq, b, d] -> [b, sq, d] fp32.
        q_h = q[:, :, hi, :].float().permute(1, 0, 2).contiguous()  # [b, sq, d]
        logits_h = torch.bmm(q_h, k_bds)  # [b, sq, sk] fp32
        del q_h
        if use_relu:
            logits_h = torch.relu(logits_h)
        # weights[:, :, hi] is [sq, b]; broadcast to [b, sq, 1].
        w_h = weights[:, :, hi].float().transpose(0, 1).unsqueeze(-1)  # [b, sq, 1]
        index_scores.add_(logits_h * w_h)
        del logits_h, w_h

    del k_bds
    if graph_bias is not None:
        if tuple(graph_bias.shape) != (b, sq, sk):
            raise ValueError(
                f"graph_bias must be ({b},{sq},{sk}), got {tuple(graph_bias.shape)}"
            )
        beta = (
            graph_beta.to(device=q.device, dtype=torch.float32)
            if isinstance(graph_beta, torch.Tensor)
            else torch.tensor(float(graph_beta), device=q.device, dtype=torch.float32)
        )
        if beta.numel() != 1:
            raise ValueError(f"graph_beta must be scalar, got {tuple(beta.shape)}")
        if not bool(torch.isfinite(beta).item()) or float(beta.item()) <= 0.0:
            raise ValueError(
                f"graph_beta must be a finite positive scalar, got {float(beta.item())}"
            )
        index_scores.add_(graph_bias.to(device=q.device, dtype=torch.float32) * beta)
    return index_scores


def apply_dsa_indexer_fused_patch(*, force: bool = False) -> bool:
    """Monkey-patch ``dsa._compute_index_scores`` with the fused variant.

    Idempotent.  Always installs — no env gate.  Raises on any failure;
    production MBS=10 is impossible without this patch (see module
    docstring for the ~40 GiB memory accounting).
    """

    from megatron.core.transformer.experimental_attention_variant import dsa as dsa_mod

    existing = getattr(dsa_mod, "_compute_index_scores", None)
    if existing is None:
        raise RuntimeError(
            "megatron.core.transformer.experimental_attention_variant.dsa."
            "_compute_index_scores not found — Megatron version mismatch?"
        )
    already_patched = getattr(existing, _PATCH_MARKER, False) and not force

    def _compute_index_scores_fused(q, weights, k, use_relu: bool = True, **kwargs):
        # Accept **kwargs to be forward-compatible with new upstream args
        # (e.g. PR #3674 added ``mask=``).  Unused kwargs are ignored by the
        # fused math — they only affect downstream masking in the caller.
        graph_bias = None
        graph_beta = 1.0
        if _env_flag("CPPMEGA_GRAPH_ROUTES_ENABLED"):
            sq, b, _h, _d = q.shape
            sk = k.shape[0]
            graph_bias = _current_graph_route_bias(
                batch_size=int(b),
                seqlen_q=int(sq),
                seqlen_k=int(sk),
                device=q.device,
            )
            graph_beta = _env_float("CPPMEGA_DSA_GRAPH_BIAS_BETA", 1.0)
        return compute_index_scores_fused_bf16(
            q,
            weights,
            k,
            use_relu=use_relu,
            graph_bias=graph_bias,
            graph_beta=graph_beta,
        )

    if not already_patched:
        setattr(_compute_index_scores_fused, _PATCH_MARKER, True)
        dsa_mod._compute_index_scores = _compute_index_scores_fused
    _patch_dsa_graph_objective(dsa_mod)
    _patch_fused_dsa_autograd(dsa_mod)
    _patch_dsa_runtime_receipts(dsa_mod)

    if already_patched:
        log.info("cppmega DSA indexer fused patch already applied")
        return True

    log.info(
        "cppmega DSA indexer fused patch applied: per-head accumulation, "
        "never materialises [sq, b, h, sk] intermediate"
    )
    print(
        "[cppmega] DSA indexer fused patch applied "
        "(per-head accumulation, [sq,b,h,sk] intermediate eliminated)"
    )
    return True
