"""FA4 score_mod graph-route attention backend for cppmega.

Replaces the dense ``[B,1,Sq,Sk]`` TE attention bias with a sparse CSR edge
representation consumed by FlashAttention-4's ``score_mod`` + ``aux_tensors``
interface.  This eliminates the O(B*Sq*Sk) memory blow-up and keeps FA4's
pipelined Hopper/Blackwell kernels on the fast path.

Graph routes are an ADDITIVE PRIOR (``score += bias``), not a mask of
allowed/forbidden connections.  FA4's ``block_sparse_tensors`` REMOVES
computation of blocks (a mask), which would destroy normal attention, so it
is NOT used here.  The full causal tile schedule is used instead
(``causal=True``, ``block_sparse_tensors=None``, ``mask_mod=None``); the
graph prior is applied purely through ``score_mod``.

The module is importable and testable WITHOUT a GPU: ``flash_attn.cute`` is
imported lazily inside the forward pass so unit tests can mock it.

Enable with ``CPPMEGA_FA4_GRAPH_ATTENTION=1`` (default ``"0"``).

See ``docs/fa4_score_mod_design.md`` for the full design rationale.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any

import torch

from cppmega.megatron.dsa_indexer_fused_patch import (
    _as_batched_chunks,
    _as_batched_edges,
    _as_batched_edge_triples,
    require_graph_routes_for_production,
)
from cppmega.megatron.graph_objective_loss import (
    graph_routes_active,
    resolve_graph_bias_beta,
    validate_graph_bias_beta,
)

log = logging.getLogger(__name__)

__all__ = [
    "CppMegaFA4DotProductAttention",
    "FA4GraphRouteAux",
    "GraphEdgeCSR",
    "build_fa4_graph_route_aux",
    "build_graph_edge_csr_from_structure_batch",
    "fa4_graph_attention_enabled",
    "graph_route_score_mod_bwd_ref",
    "graph_route_score_mod_ref",
    "graph_score_mod",
    "graph_score_mod_bwd",
]

# ---------------------------------------------------------------------------
# Env-flag guard
# ---------------------------------------------------------------------------

_FA4_GRAPH_ATTENTION_ENV = "CPPMEGA_FA4_GRAPH_ATTENTION"


def _env_flag(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1,true,yes,on,0,false,no,off; got {raw!r}"
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def fa4_graph_attention_enabled() -> bool:
    """Return True when the FA4 graph-route attention backend is active."""
    require_graph_routes_for_production()
    if not graph_routes_active():
        return False
    return _env_flag(_FA4_GRAPH_ATTENTION_ENV, "0")


# ---------------------------------------------------------------------------
# GraphEdgeCSR dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphEdgeCSR:
    """Batched CSR representation of sparse graph edges.

    All tensors are on the target device.  Integer tensors use ``int32`` to
    match FA4 aux_tensors metadata expectations; weights use the Q dtype
    (typically bf16) with beta pre-folded.

    Attributes:
        row_offsets: ``[B, Sq + 1]`` int32.  ``row_offsets[b, q:q+2]``
            brackets row ``q``'s edges in the flat arrays.
        col_idx: ``[B, max_nnz]`` int32.  Key indices sorted ascending within
            each row.  Padding slots after ``nnz[b]`` are zero.
        weights: ``[B, max_nnz]`` (Q dtype).  Pre-multiplied by
            ``beta * relation_weight``.  Bias is added to already-scaled
            scores, matching TE post_scale_bias semantics.  Padding zeros.
        batch_size: Number of batch elements.
        seqlen_q: Query sequence length (local, i.e. active query window).
        seqlen_k: Key sequence length.
        max_nnz: High-water mark for per-batch edge count (padded width).
        nnz_per_batch: ``[B]`` int32 actual edge counts per batch element.
        query_start: Global offset for the query axis (0 in prefill).
    """

    row_offsets: torch.Tensor
    col_idx: torch.Tensor
    weights: torch.Tensor
    batch_size: int
    seqlen_q: int
    seqlen_k: int
    max_nnz: int
    nnz_per_batch: torch.Tensor
    query_start: int

    def __post_init__(self) -> None:
        if self.row_offsets.dim() != 2:
            raise ValueError(
                f"row_offsets must be 2-D [B, Sq+1], got {tuple(self.row_offsets.shape)}"
            )
        if self.col_idx.dim() != 2:
            raise ValueError(
                f"col_idx must be 2-D [B, max_nnz], got {tuple(self.col_idx.shape)}"
            )
        if self.weights.dim() != 2:
            raise ValueError(
                f"weights must be 2-D [B, max_nnz], got {tuple(self.weights.shape)}"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seqlen_q <= 0 or self.seqlen_k <= 0:
            raise ValueError("seqlen_q and seqlen_k must be positive")
        if self.max_nnz < 0:
            raise ValueError("max_nnz must be non-negative")
        expected_offsets_shape = (self.batch_size, self.seqlen_q + 1)
        if tuple(self.row_offsets.shape) != expected_offsets_shape:
            raise ValueError(
                f"row_offsets shape {tuple(self.row_offsets.shape)} != "
                f"expected {expected_offsets_shape}"
            )
        expected_flat_shape = (self.batch_size, self.max_nnz)
        if tuple(self.col_idx.shape) != expected_flat_shape:
            raise ValueError(
                f"col_idx shape {tuple(self.col_idx.shape)} != "
                f"expected {expected_flat_shape}"
            )
        if tuple(self.weights.shape) != expected_flat_shape:
            raise ValueError(
                f"weights shape {tuple(self.weights.shape)} != "
                f"expected {expected_flat_shape}"
            )
        if self.query_start < 0:
            raise ValueError("query_start must be non-negative")


# ---------------------------------------------------------------------------
# CSR builder
# ---------------------------------------------------------------------------


def build_graph_edge_csr_from_structure_batch(
    structure_batch: dict[str, torch.Tensor] | None,
    *,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
    q_dtype: torch.dtype = torch.bfloat16,
    query_start: int = 0,
    beta: float | None = None,
    softmax_scale: float | None = None,
    max_nnz_per_batch: int | None = None,
    call_weight: float = 1.0,
    type_weight: float = 1.0,
    domain_weight: float = 1.0,
    build_weight: float = 1.0,
    shell_weight: float = 1.0,
    diagnostic_weight: float = 1.0,
    cross_domain_weight: float = 1.0,
) -> GraphEdgeCSR:
    """Convert the existing structure_batch edge format into batched CSR.

    Reuses the same ``_as_batched_edges`` / ``_as_batched_edge_triples`` /
    ``_as_batched_chunks`` helpers and fail-closed validation as the dense
    path.  Weights are pre-multiplied by ``beta * relation_weight`` so the
    in-kernel score_mod only needs an additive lookup.  Bias is added to
    already-scaled scores, matching TE post_scale_bias semantics.

    Args:
        structure_batch: The cppmega graph sidecar dict.
        batch_size: Microbatch size B.
        seqlen_q: Local query length (active window).
        seqlen_k: Full key length.
        device: Target CUDA device.
        q_dtype: Dtype for weight tensor (matches Q, typically bf16).
        query_start: Global query offset (0 for prefill, >0 for decode).
        beta: Graph bias beta; resolved from env if None.
        softmax_scale: Deprecated and ignored.  FA4 applies scaling internally
            before calling score_mod, so bias must NOT include softmax_scale.
        max_nnz_per_batch: High-water mark for per-batch nnz.  If None, uses
            ``CPPMEGA_FA4_MAX_NNZ_PER_BATCH`` env (default 4096).
        call_weight: Weight for call edges.
        type_weight: Weight for type edges.
        domain_weight: Weight for domain edges.
        build_weight: Weight for build edges.
        shell_weight: Weight for shell edges.
        diagnostic_weight: Weight for diagnostic edges.
        cross_domain_weight: Weight for cross-domain edges.

    Returns:
        A frozen ``GraphEdgeCSR`` with sorted, deduplicated edges.

    Raises:
        RuntimeError: If structure_batch is None or no route tensors found.
        ValueError: On corrupt sidecar metadata or nnz overflow.
    """
    if structure_batch is None:
        raise RuntimeError(
            "FA4 graph-route attention requires a structure batch; "
            "refusing token-only fallback"
        )
    if not isinstance(structure_batch, dict):
        raise TypeError(
            f"structure_batch must be a dict, got {type(structure_batch).__name__}"
        )
    if batch_size <= 0 or seqlen_q <= 0 or seqlen_k <= 0:
        raise ValueError(
            f"dimensions must be positive: B={batch_size} Sq={seqlen_q} Sk={seqlen_k}"
        )
    if query_start < 0 or query_start + seqlen_q > seqlen_k:
        raise ValueError(
            f"query window [{query_start}, {query_start + seqlen_q}) must be "
            f"contained in [0, {seqlen_k})"
        )

    effective_beta = (
        resolve_graph_bias_beta() if beta is None else validate_graph_bias_beta(beta)
    )
    if softmax_scale is not None:
        warnings.warn(
            "softmax_scale is deprecated and ignored in "
            "build_graph_edge_csr_from_structure_batch. FA4 applies scaling "
            "internally before calling score_mod; bias is added to "
            "already-scaled scores (TE post_scale_bias semantics).",
            DeprecationWarning,
            stacklevel=2,
        )
    weight_multiplier = effective_beta

    if max_nnz_per_batch is None:
        max_nnz_per_batch = _env_int("CPPMEGA_FA4_MAX_NNZ_PER_BATCH", 4096)
    if max_nnz_per_batch <= 0:
        raise ValueError("max_nnz_per_batch must be positive")

    # Per-batch edge accumulation lists.
    batch_src: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    batch_dst: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    batch_w: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

    seen_relation = False
    chunk_layout: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    query_end = query_start + seqlen_q

    # --- Chunk-index relations (call, type) ---
    for edge_key, count_key, relation_weight in (
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
        if int(edges.shape[0]) == 1 and batch_size > 1:
            edges = edges.expand(batch_size, -1, -1)
        if int(counts.shape[0]) == 1 and batch_size > 1:
            counts = counts.expand(batch_size)

        for b in range(batch_size):
            n_edges = int(counts[b].item())
            available = int(chunk_counts[b].item())
            for ei in range(n_edges):
                src_chunk = int(edges[b, ei, 0].item())
                dst_chunk = int(edges[b, ei, 1].item())
                if not (0 <= src_chunk < available and 0 <= dst_chunk < available):
                    raise ValueError(
                        f"declared call/type edge ({src_chunk},{dst_chunk}) "
                        f"references an unavailable chunk (available={available})"
                    )
                # Expand chunk pair into token-span rectangle, clipped to query window.
                src_start = max(query_start, int(starts[b, src_chunk].item()))
                src_end = min(query_end, int(ends[b, src_chunk].item()))
                dst_start = max(0, int(starts[b, dst_chunk].item()))
                dst_end = min(seqlen_k, int(ends[b, dst_chunk].item()))
                if src_start < src_end and dst_start < dst_end:
                    nq = src_end - src_start
                    nk = dst_end - dst_start
                    src_indices = (
                        torch.arange(src_start, src_end, device=device)
                        .unsqueeze(1)
                        .expand(nq, nk)
                        .reshape(-1)
                        - query_start
                    )
                    dst_indices = (
                        torch.arange(dst_start, dst_end, device=device)
                        .unsqueeze(0)
                        .expand(nq, nk)
                        .reshape(-1)
                    )
                    w_tensor = torch.full(
                        (nq * nk,),
                        relation_weight * weight_multiplier,
                        device=device,
                        dtype=q_dtype,
                    )
                    batch_src[b].append(src_indices.to(torch.int32))
                    batch_dst[b].append(dst_indices.to(torch.int32))
                    batch_w[b].append(w_tensor)

    # --- Token-position triple relations ---
    for edge_key, count_key, relation_weight in (
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
        if int(edges.shape[0]) == 1 and batch_size > 1:
            edges = edges.expand(batch_size, -1, -1)
        if int(counts.shape[0]) == 1 and batch_size > 1:
            counts = counts.expand(batch_size)
        max_edges = int(edges.shape[1])
        if bool(((counts < 0) | (counts > max_edges)).any().item()):
            raise ValueError(f"graph edge counts out of range [0,{max_edges}]")

        for b in range(batch_size):
            n_edges = int(counts[b].item())
            for ei in range(n_edges):
                src = int(edges[b, ei, 0].item())
                dst = int(edges[b, ei, 1].item())
                kind = int(edges[b, ei, 2].item())
                if kind < 0:
                    continue  # inactive triple
                if not (0 <= src < seqlen_k and 0 <= dst < seqlen_k):
                    raise ValueError(
                        f"graph token edge ({src},{dst}) is outside "
                        f"sequence bounds [0, {seqlen_k})"
                    )
                local_q = src - query_start
                if not (0 <= local_q < seqlen_q):
                    continue  # edge src outside active query window (decode)
                batch_src[b].append(
                    torch.tensor([local_q], device=device, dtype=torch.int32)
                )
                batch_dst[b].append(
                    torch.tensor([dst], device=device, dtype=torch.int32)
                )
                batch_w[b].append(
                    torch.tensor(
                        [relation_weight * weight_multiplier],
                        device=device,
                        dtype=q_dtype,
                    )
                )

    # --- Generated query edges overlay (weight=1.0, matches dense path) ---
    generated = _as_batched_edges(
        structure_batch,
        edge_key="graph_generated_query_edges",
        count_key="graph_generated_query_edge_counts",
        batch_size=batch_size,
        device=device,
    )
    if generated is not None:
        seen_relation = True
        gen_edges, gen_counts = generated
        if int(gen_edges.shape[0]) == 1 and batch_size > 1:
            gen_edges = gen_edges.expand(batch_size, -1, -1)
        if int(gen_counts.shape[0]) == 1 and batch_size > 1:
            gen_counts = gen_counts.expand(batch_size)
        max_gen = int(gen_edges.shape[1])
        if bool(((gen_counts < 0) | (gen_counts > max_gen)).any().item()):
            raise ValueError(f"generated query edge counts out of range [0,{max_gen}]")
        for b in range(batch_size):
            n_gen = int(gen_counts[b].item())
            for ei in range(n_gen):
                src = int(gen_edges[b, ei, 0].item())
                dst = int(gen_edges[b, ei, 1].item())
                if not (0 <= src < seqlen_k and 0 <= dst < seqlen_k):
                    raise ValueError(
                        f"generated query edge ({src},{dst}) outside "
                        f"sequence bounds [0, {seqlen_k})"
                    )
                local_q = src - query_start
                if not (0 <= local_q < seqlen_q):
                    raise ValueError(
                        f"generated query edge src={src} outside query window "
                        f"[{query_start}, {query_end})"
                    )
                batch_src[b].append(
                    torch.tensor([local_q], device=device, dtype=torch.int32)
                )
                batch_dst[b].append(
                    torch.tensor([dst], device=device, dtype=torch.int32)
                )
                batch_w[b].append(
                    torch.tensor(
                        [1.0 * weight_multiplier], device=device, dtype=q_dtype
                    )
                )

    if not seen_relation:
        raise RuntimeError(
            "FA4 graph-route attention: structure batch contains no route "
            "tensors (expected graph_call_edges/type_edges or "
            "domain/build/shell/diagnostic/cross-domain edges)"
        )

    # --- Build CSR per batch, summing duplicate (q, k) pairs ---
    row_offsets = torch.zeros(
        (batch_size, seqlen_q + 1), device=device, dtype=torch.int32
    )
    col_idx = torch.zeros(
        (batch_size, max_nnz_per_batch), device=device, dtype=torch.int32
    )
    weights_out = torch.zeros(
        (batch_size, max_nnz_per_batch), device=device, dtype=q_dtype
    )
    nnz_per_batch = torch.zeros((batch_size,), device=device, dtype=torch.int32)

    for b in range(batch_size):
        if not batch_src[b]:
            continue
        src_cat = torch.cat(batch_src[b])  # [total_edges_b] int32
        dst_cat = torch.cat(batch_dst[b])  # [total_edges_b] int32
        w_cat = torch.cat(batch_w[b])  # [total_edges_b] q_dtype

        # Sum duplicate (q, k) pairs via unique + index_add_ (matches the
        # dense path's index_add_ semantics).
        linear = src_cat.to(torch.long) * seqlen_k + dst_cat.to(torch.long)
        unique_linear, inverse = torch.unique(linear, return_inverse=True)
        n_unique = int(unique_linear.numel())
        summed_w = torch.zeros(n_unique, device=device, dtype=q_dtype)
        summed_w.index_add_(0, inverse, w_cat)

        # Recover (q, k) from unique linear indices.
        unique_q = (unique_linear // seqlen_k).to(torch.int32)
        unique_k = (unique_linear % seqlen_k).to(torch.int32)

        if n_unique > max_nnz_per_batch:
            raise ValueError(
                f"FA4 graph-route CSR overflow: batch {b} has {n_unique} unique "
                f"edges, exceeding max_nnz_per_batch={max_nnz_per_batch}. "
                f"Raise CPPMEGA_FA4_MAX_NNZ_PER_BATCH or reduce edge density."
            )

        # Sort by (q, k) for CSR layout: primary sort by q, secondary by k.
        sort_key = unique_q.to(torch.long) * seqlen_k + unique_k.to(torch.long)
        sort_order = torch.argsort(sort_key)
        sorted_q = unique_q[sort_order]
        sorted_k = unique_k[sort_order]
        sorted_w = summed_w[sort_order]

        # Build row offsets via bincount on q indices.
        row_counts = torch.zeros(seqlen_q, device=device, dtype=torch.int32)
        row_counts.scatter_add_(
            0,
            sorted_q.to(torch.long),
            torch.ones(n_unique, device=device, dtype=torch.int32),
        )
        offsets_b = torch.zeros(seqlen_q + 1, device=device, dtype=torch.int32)
        offsets_b[1:] = torch.cumsum(row_counts, dim=0).to(torch.int32)
        row_offsets[b] = offsets_b

        # Write sorted edges into padded arrays.
        col_idx[b, :n_unique] = sorted_k
        weights_out[b, :n_unique] = sorted_w
        nnz_per_batch[b] = n_unique

    return GraphEdgeCSR(
        row_offsets=row_offsets,
        col_idx=col_idx,
        weights=weights_out,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        max_nnz=max_nnz_per_batch,
        nnz_per_batch=nnz_per_batch,
        query_start=query_start,
    )


# ---------------------------------------------------------------------------
# FA4GraphRouteAux – test-facing aux wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FA4GraphRouteAux:
    """Auxiliary tensors for FA4 score_mod, shaped for direct kernel consumption.

    This is the test-facing wrapper around ``GraphEdgeCSR`` that exposes the
    flat tensor layout expected by the score_mod reference functions.

    Attributes:
        csr_row_offsets: ``[B, Sq+1]`` int32.
        csr_col_idx: ``[B, max_nnz]`` int32, sorted within each row.
        csr_weight: ``[B, max_nnz]`` (Q dtype), beta pre-folded.
        csr_meta: ``[4]`` int32 – ``[Sq, Sk, max_nnz, flags]``.
    """

    csr_row_offsets: torch.Tensor
    csr_col_idx: torch.Tensor
    csr_weight: torch.Tensor
    csr_meta: torch.Tensor


def build_fa4_graph_route_aux(
    structure_batch: dict[str, torch.Tensor] | None,
    *,
    batch_size: int,
    query_start: int = 0,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
    q_dtype: torch.dtype = torch.bfloat16,
    beta: float = 1.0,
    call_weight: float = 1.0,
    type_weight: float = 1.0,
    domain_weight: float = 1.0,
    build_weight: float = 1.0,
    shell_weight: float = 1.0,
    diagnostic_weight: float = 1.0,
    cross_domain_weight: float = 1.0,
) -> FA4GraphRouteAux:
    """Build FA4GraphRouteAux from a structure_batch.

    Thin wrapper around ``build_graph_edge_csr_from_structure_batch`` that
    returns the flat tensor layout expected by score_mod reference functions.
    """
    csr = build_graph_edge_csr_from_structure_batch(
        structure_batch,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        device=device,
        q_dtype=q_dtype,
        query_start=query_start,
        beta=beta,
        call_weight=call_weight,
        type_weight=type_weight,
        domain_weight=domain_weight,
        build_weight=build_weight,
        shell_weight=shell_weight,
        diagnostic_weight=diagnostic_weight,
        cross_domain_weight=cross_domain_weight,
    )
    flags = 1 if query_start > 0 else 0
    csr_meta = torch.tensor(
        [csr.seqlen_q, csr.seqlen_k, csr.max_nnz, flags],
        device=device,
        dtype=torch.int32,
    )
    return FA4GraphRouteAux(
        csr_row_offsets=csr.row_offsets,
        csr_col_idx=csr.col_idx,
        csr_weight=csr.weights,
        csr_meta=csr_meta,
    )


# ---------------------------------------------------------------------------
# score_mod reference functions (Python-level, for unit tests)
# ---------------------------------------------------------------------------


def graph_route_score_mod_ref(
    score: float,
    *,
    batch_idx: int,
    q_idx: int,
    kv_idx: int,
    aux: FA4GraphRouteAux,
) -> float:
    """Python reference for the FA4 score_mod: additive CSR bias lookup.

    Returns ``score + weight[batch_idx, q_idx, kv_idx]`` where the weight is
    found via binary search in the CSR structure.
    """
    row_offsets = aux.csr_row_offsets
    col_idx = aux.csr_col_idx
    weight = aux.csr_weight

    lo = int(row_offsets[batch_idx, q_idx].item())
    hi = int(row_offsets[batch_idx, q_idx + 1].item())

    left, right = lo, hi
    while left < right:
        mid = (left + right) >> 1
        c = int(col_idx[batch_idx, mid].item())
        if c < kv_idx:
            left = mid + 1
        else:
            right = mid
    if left < hi and int(col_idx[batch_idx, left].item()) == kv_idx:
        return score + float(weight[batch_idx, left].item())
    return score


def graph_route_score_mod_bwd_ref(
    grad_out: float,
    *,
    score: float = 0.0,
    batch_idx: int = 0,
    q_idx: int = 0,
    kv_idx: int = 0,
    aux: FA4GraphRouteAux | None = None,
) -> float:
    """Python reference for the FA4 score_mod backward: identity.

    ``score' = score + bias`` => ``d(score')/d(score) = 1``.
    """
    return grad_out


# ---------------------------------------------------------------------------
# score_mod callables (CuTe-DSL compatible reference implementation)
# ---------------------------------------------------------------------------


def graph_score_mod(
    acc_S_SSA: Any,
    batch_idx: Any,
    head_idx: Any,
    q_idx: Any,
    kv_idx: Any,
    seqlen_info: Any,
    aux_tensors: list[Any],
) -> Any:
    """FA4 score_mod: binary-search CSR for (query, key) edge weight.

    This is the reference implementation matching the CuTe-DSL pseudocode in
    the design doc.  When running under FA4's ``cute.jit``, this function is
    compiled to GPU code; for unit tests it can be called directly with scalar
    or tensor arguments.

    The score modification is::

        score'[b, h, q, k] = score[b, h, q, k] + weight[b, q, k]

    where weight already has ``beta * relation_weight``
    folded in on the host.  Bias is added to already-scaled scores,
    matching TE post_scale_bias semantics.
    """
    row_offsets, col_idx, weight, _meta = aux_tensors

    # q_idx is in local coordinates (query_start already accounted for by the
    # caller via aux_scalars in the real FA4 path).
    q_local = q_idx

    lo = row_offsets[batch_idx, q_local]
    hi = row_offsets[batch_idx, q_local + 1]

    # Binary search the sorted column list for kv_idx.
    # Edge counts per row are tiny (<<32), so this is a bounded loop the
    # CuTe DSL compiler unrolls into a register scan.
    found_w = 0.0
    left, right = lo, hi
    while left < right:
        mid = (left + right) >> 1
        c = col_idx[batch_idx, mid]
        if c < kv_idx:
            left = mid + 1
        else:
            right = mid
    if left < hi and col_idx[batch_idx, left] == kv_idx:
        found_w = weight[batch_idx, left]

    return acc_S_SSA + found_w


def graph_score_mod_bwd(
    grad_out_SSA: Any,
    score_SSA: Any,
    batch_idx: Any,
    head_idx: Any,
    q_idx: Any,
    kv_idx: Any,
    seqlen_info: Any,
    aux_tensors: list[Any],
) -> Any:
    """FA4 score_mod_bwd: identity.

    ``score' = score + bias`` => ``d(score')/d(score) = 1``.  The graph bias
    is non-learnable (built fresh each step from compiler edges), so no
    gradient flows to the CSR weight tensors.
    """
    return grad_out_SSA


# ---------------------------------------------------------------------------
# CppMegaFA4DotProductAttention module
# ---------------------------------------------------------------------------


class CppMegaFA4DotProductAttention(torch.nn.Module):
    """FA4-backed core attention with sparse graph-route score_mod.

    Drop-in replacement for ``TEDotProductAttention`` in Megatron's
    ``ModuleSpec`` wiring.  QKV projections, FP8 GEMMs, RoPE, and output
    projection stay in TE; only the dot-product/softmax kernel changes.

    Graph routes are additive ``score_mod`` bias, not an attention mask.
    A full causal tile schedule is used (``causal=True``,
    ``block_sparse_tensors=None``, ``mask_mod=None``); the graph prior is
    applied purely through ``score_mod``.

    The module refuses dense ``attention_bias`` tensors by contract (the
    whole point is to avoid materializing them).  Pass ``GraphEdgeCSR`` via
    the ``attention_bias`` argument or let the patch wrapper build it.

    Enable with ``CPPMEGA_FA4_GRAPH_ATTENTION=1``.
    """

    def __init__(
        self,
        config: Any = None,
        layer_number: int | None = None,
        attention_type: str | None = None,
        num_attention_heads: int | None = None,
        attention_dropout: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = True,
        window_size: tuple[int | None, int | None] = (None, None),
        deterministic: bool = False,
        max_nnz_per_batch: int | None = None,
        beta: float | None = None,
        call_weight: float = 1.0,
        type_weight: float = 1.0,
        domain_weight: float = 1.0,
        build_weight: float = 1.0,
        shell_weight: float = 1.0,
        diagnostic_weight: float = 1.0,
        cross_domain_weight: float = 1.0,
        **_ignored_te_kwargs: Any,
    ) -> None:
        super().__init__()
        if attention_dropout != 0.0:
            raise ValueError(
                "CppMegaFA4DotProductAttention does not support attention "
                f"dropout (got {attention_dropout}); FA4 score_mod path has "
                "no dropout support. Production cppmega uses dropout=0."
            )
        if config is not None and getattr(config, "attention_dropout", 0) > 0:
            raise ValueError(
                "CppMegaFA4DotProductAttention does not support attention "
                f"dropout (config.attention_dropout={config.attention_dropout}); "
                "FA4 score_mod path has no dropout support. Set "
                "attention_dropout=0 in TransformerConfig."
            )
        self.config = config
        self.layer_number = layer_number
        self.attention_type = attention_type
        self.num_attention_heads = num_attention_heads
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.deterministic = deterministic
        self.max_nnz_per_batch = max_nnz_per_batch
        self.beta = beta
        self.call_weight = call_weight
        self.type_weight = type_weight
        self.domain_weight = domain_weight
        self.build_weight = build_weight
        self.shell_weight = shell_weight
        self.diagnostic_weight = diagnostic_weight
        self.cross_domain_weight = cross_domain_weight
        self._first_forward_logged = False

    def _log_first_use(self) -> None:
        if not self._first_forward_logged:
            self._first_forward_logged = True
            log.info(
                "[cppmega] FA4 score_mod graph-route attention active "
                "(layer=%s, causal=%s, deterministic=%s)",
                self.layer_number,
                self.causal,
                self.deterministic,
            )
            print(
                f"[cppmega] FA4 score_mod graph-route attention active "
                f"(layer={self.layer_number}, causal={self.causal})",
                flush=True,
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Any = None,
        attn_mask_type: Any = None,
        attention_bias: Any = None,
        packed_seq_params: Any = None,
        inference_context: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run FA4 attention with optional graph-route score_mod.

        ABI contract: Megatron passes QKV as ``[S, B, H, D]`` (sequence-first).
        This module transposes to ``[B, S, H, D]`` for FA4, then reshapes the
        output to ``[S, B, H*D]`` (3-D) as expected by Megatron's ``linear_proj``.

        Args:
            query: ``[S, B, H, D]`` projected queries (Megatron SBHD layout).
            key: ``[S, B, Hk, D]`` projected keys (Megatron SBHD layout).
            value: ``[S, B, Hk, D]`` projected values (Megatron SBHD layout).
            attention_mask: Ignored (FA4 handles masking via causal/window).
            attn_mask_type: Ignored. A full causal tile schedule is always
                used (graph routes are additive score_mod bias, not a mask).
            attention_bias: One of:
                - ``None``: plain FA4 attention (no score_mod).
                - ``GraphEdgeCSR``: sparse graph-route bias via score_mod.
                - ``torch.Tensor``: RAISES (dense bias refused by contract).
            packed_seq_params: Must be None (not supported).
            inference_context: Optional; used for decode geometry.

        Returns:
            Context tensor ``[S, B, H*D]`` (3-D, Megatron linear_proj input).
        """
        # --- Fail-closed validation ---
        if not isinstance(query, torch.Tensor):
            raise TypeError(f"query must be a Tensor, got {type(query).__name__}")
        if not isinstance(key, torch.Tensor):
            raise TypeError(f"key must be a Tensor, got {type(key).__name__}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"value must be a Tensor, got {type(value).__name__}")

        if query.dim() != 4:
            raise ValueError(
                f"query must be [S, B, H, D], got shape {tuple(query.shape)}"
            )
        if key.dim() != 4:
            raise ValueError(
                f"key must be [S, B, Hk, D], got shape {tuple(key.shape)}"
            )
        if value.dim() != 4:
            raise ValueError(
                f"value must be [S, B, Hk, D], got shape {tuple(value.shape)}"
            )

        # --- SBHD → BSHD transpose (Megatron ABI: sequence-first) ---
        query = query.transpose(0, 1)  # [S,B,H,D] -> [B,S,H,D]
        key = key.transpose(0, 1)      # [S,B,Hk,D] -> [B,S,Hk,D]
        value = value.transpose(0, 1)  # [S,B,Hk,D] -> [B,S,Hk,D]

        batch_size, seqlen_q, num_heads, head_dim = query.shape
        _, seqlen_k, num_kv_heads, _ = key.shape

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads}) for GQA"
            )
        if packed_seq_params is not None:
            raise ValueError(
                "CppMegaFA4DotProductAttention does not support packed_seq_params"
            )

        # Refuse dense tensor bias by contract.
        if isinstance(attention_bias, torch.Tensor):
            raise TypeError(
                "CppMegaFA4DotProductAttention refuses dense attention_bias "
                "tensors by contract. Pass a GraphEdgeCSR or None. The FA4 "
                "backend exists precisely to avoid O(B*Sq*Sk) materialization."
            )

        # Sequence parallel / context parallel guard.
        if self.config is not None:
            if getattr(self.config, "sequence_parallel", False):
                raise RuntimeError(
                    "FA4 graph-route attention does not support sequence_parallel"
                )
            if int(getattr(self.config, "context_parallel_size", 1)) != 1:
                raise RuntimeError(
                    "FA4 graph-route attention does not support "
                    "context_parallel_size > 1"
                )

        self._log_first_use()

        # --- Resolve CSR ---
        csr: GraphEdgeCSR | None = None
        fa4_aux: FA4GraphRouteAux | None = None
        if isinstance(attention_bias, GraphEdgeCSR):
            csr = attention_bias
        elif isinstance(attention_bias, FA4GraphRouteAux):
            fa4_aux = attention_bias
        elif attention_bias is not None:
            raise TypeError(
                f"attention_bias must be None, GraphEdgeCSR, or FA4GraphRouteAux, "
                f"got {type(attention_bias).__name__}"
            )

        # --- Resolve softmax scale ---
        scale = self.softmax_scale
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # --- Call FA4 (lazy import for GPU-free testability) ---
        from flash_attn.cute.interface import flash_attn_func

        score_mod_fn = None
        score_mod_bwd_fn = None
        aux_tensors_arg = None
        aux_scalars_arg = None

        if fa4_aux is not None:
            # FA4GraphRouteAux path: tensors are pre-built.
            aux_tensors_arg = [
                fa4_aux.csr_row_offsets,
                fa4_aux.csr_col_idx,
                fa4_aux.csr_weight,
                fa4_aux.csr_meta,
            ]
            query_start = int(fa4_aux.csr_meta[3].item())
            aux_scalars_arg = (query_start,)
            score_mod_fn = graph_score_mod
            score_mod_bwd_fn = graph_score_mod_bwd

        elif csr is not None:
            # Meta tensor [Sq, Sk, max_nnz, flags] for in-kernel bounds checks.
            flags = 1 if csr.query_start > 0 else 0
            csr_meta = torch.tensor(
                [csr.seqlen_q, csr.seqlen_k, csr.max_nnz, flags],
                device=query.device,
                dtype=torch.int32,
            )

            aux_tensors_arg = [
                csr.row_offsets,
                csr.col_idx,
                csr.weights,
                csr_meta,
            ]
            aux_scalars_arg = (csr.query_start,)
            score_mod_fn = graph_score_mod
            score_mod_bwd_fn = graph_score_mod_bwd

        # Graph routes are an additive score_mod bias, not an attention
        # mask.  Use the full causal tile schedule: block_sparse_tensors=None
        # and mask_mod=None so no KV blocks are skipped.
        out = flash_attn_func(
            q=query,
            k=key,
            v=value,
            softmax_scale=scale,
            causal=True,
            window_size=self.window_size,
            deterministic=self.deterministic,
            score_mod=score_mod_fn,
            score_mod_bwd=score_mod_bwd_fn,
            aux_tensors=aux_tensors_arg,
            aux_scalars=aux_scalars_arg,
            block_sparse_tensors=None,
            mask_mod=None,
            return_lse=False,
        )

        # flash_attn_func returns (out, lse) when return_lse=True, else just out.
        if isinstance(out, tuple):
            out = out[0]

        # --- BSHD → S,B,H*D (Megatron expects 3-D for linear_proj) ---
        # FA4 returns [B, S, H, D]; Megatron expects [S, B, H*D].
        out = out.transpose(0, 1)  # [B,S,H,D] -> [S,B,H,D]
        out = out.reshape(out.size(0), out.size(1), -1)  # [S,B,H,D] -> [S,B,H*D]

        return out
