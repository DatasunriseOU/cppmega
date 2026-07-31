"""FA4 beta23 chunk-native score_mod adapter for cppmega.

Implements the chunk-native graph-route attention bias via FlashAttention-4's
``score_mod`` + ``aux_tensors`` interface (beta23 API surface).  Instead of
materializing a dense ``[B,1,Sq,Sk]`` bias, the graph prior is kept in chunk
space: a tiny ``[B, C+1, C+1]`` bias plus a ``[B, S]`` token-to-chunk map,
with a sparse CSR overlay for rare point edges that don't align to chunk
boundaries.

This yields ~89x-1600x HBM reduction vs the dense path while keeping FA4's
pipelined Hopper/Blackwell kernels on the fast path.

The module is importable without a GPU because ``flash_attn.cute`` is imported
lazily. Kernel integration is exercised against the real FA4 runtime on H200.

Enable with ``CPPMEGA_FA4_SCORE_MOD=1`` (default ``"0"``).

See ``docs/fa4_beta23_score_mod_poc.md`` for the full design rationale.
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
    _token_chunk_map,
    require_graph_routes_for_production,
)
from cppmega.megatron.graph_objective_loss import (
    graph_routes_active,
    resolve_graph_bias_beta,
    validate_graph_bias_beta,
)

log = logging.getLogger(__name__)

__all__ = [
    "ChunkNativeGraphBias",
    "CppMegaFA4ScoreModAttention",
    "build_chunk_native_graph_bias",
    "build_fa4_attention_bias_from_structure_batch",
    "chunk_native_score_mod_bwd_ref",
    "chunk_native_score_mod_ref",
    "fa4_score_mod_enabled",
    "_build_document_mask_aux",
    "_make_document_causal_mask_mod",
    "_make_graph_score_mod",
    "_make_graph_score_mod_bwd",
]

# ---------------------------------------------------------------------------
# Env-flag guard
# ---------------------------------------------------------------------------

_FA4_SCORE_MOD_ENV = "CPPMEGA_FA4_SCORE_MOD"
_FA4_MAX_RARE_ENV = "CPPMEGA_FA4_MAX_RARE_PER_ROW"
_FA4_MAX_RARE_DEFAULT = 256


def _fa4_max_rare_per_row() -> int:
    """Fixed high-water mark for rare-edge slots per batch item.

    Using a fixed allocation (instead of per-batch max) ensures stable aux
    tensor shapes across steps, preventing FA4 recompilation.

    The default of 256 covers the 95th percentile of real rows (avg 4.7
    edges/row, p95 ~40).  With real data, 1.16% of samples have >64 rare
    edges (max observed 952 was an outlier in old data); at microbatch 192
    the old default of 64 gave an 89% chance of overflow per step.  Raising
    to 256 provides ample headroom while keeping JIT compile time bounded.

    If the RuntimeError fires, either increase the limit via the
    CPPMEGA_FA4_MAX_RARE_PER_ROW env var, or investigate why so many
    token-level edges exist (possible sidecar bug or unexpected graph shape).
    """
    raw = os.environ.get(_FA4_MAX_RARE_ENV, str(_FA4_MAX_RARE_DEFAULT))
    val = int(raw)
    if val < 1:
        raise ValueError(f"{_FA4_MAX_RARE_ENV} must be >= 1, got {val}")
    return val


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


def fa4_score_mod_enabled() -> bool:
    """Return True when the FA4 chunk-native score_mod backend is active."""
    require_graph_routes_for_production()
    if not graph_routes_active():
        return False
    return _env_flag(_FA4_SCORE_MOD_ENV, "0")


# ---------------------------------------------------------------------------
# ChunkNativeGraphBias dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkNativeGraphBias:
    """Chunk-native graph-route bias for FA4 score_mod.

    The graph prior is represented as:
    - Token-to-chunk maps ``[B, Sq]`` and ``[B, Sk]`` (each token maps to
      chunk 0..C-1, or C as sentinel for inter-chunk gaps).
    - A chunk-pair bias matrix ``[B, C+1, C+1]`` carrying summed weights
      for each (src_chunk, dst_chunk) pair.  The sentinel row/column [C,*]
      and [*,C] are identically zero so gap tokens contribute no bias.
    - A per-row CSR overlay of rare point edges (token-position pairs that
      don't align to chunk boundaries).

    Integer aux tensors use int32 per the FA4 design contract.

    Attributes:
        token_to_chunk_q: ``[B, Sq]`` IntTensor (int32). Maps query token
            index to chunk index in ``[0, C]`` where C is the sentinel
            "no chunk".
        token_to_chunk_k: ``[B, Sk]`` IntTensor (int32). Maps key token
            index to chunk index in ``[0, C]`` where C is the sentinel
            "no chunk".
        chunk_bias: ``[B, C+1, C+1]`` FloatTensor. Additive bias per chunk
            pair, pre-multiplied by beta * relation_weight.
        rare_row_offsets: ``[B, Sq+1]`` IntTensor (int32). Per-row CSR
            offsets.  ``rare_row_offsets[b, q]`` to
            ``rare_row_offsets[b, q+1]`` brackets the rare edges for
            batch b, query row q.
        rare_q: ``[B, max_rare]`` IntTensor (int32). Query positions
            (padded).
        rare_k: ``[B, max_rare]`` IntTensor (int32). Key positions, sorted
            ascending within each query row.  Unused slots are padded with
            -1 (sentinel that never matches a valid key >= 0).
        rare_w: ``[B, max_rare]`` FloatTensor. Weights for rare edges,
            pre-multiplied by beta * relation_weight.
        max_chunks: int. Number of real chunks C (sentinel index is C).
        beta: float. Global scaling factor (informational; already folded
            into chunk_bias and rare_w).
    """

    token_to_chunk_q: torch.Tensor
    token_to_chunk_k: torch.Tensor
    chunk_bias: torch.Tensor
    rare_row_offsets: torch.Tensor
    rare_q: torch.Tensor
    rare_k: torch.Tensor
    rare_w: torch.Tensor
    max_chunks: int
    beta: float

    def __post_init__(self) -> None:
        if self.token_to_chunk_q.dim() != 2:
            raise ValueError(
                f"token_to_chunk_q must be 2-D [B, Sq], got "
                f"{tuple(self.token_to_chunk_q.shape)}"
            )
        if self.token_to_chunk_k.dim() != 2:
            raise ValueError(
                f"token_to_chunk_k must be 2-D [B, Sk], got "
                f"{tuple(self.token_to_chunk_k.shape)}"
            )
        if self.chunk_bias.dim() != 3:
            raise ValueError(
                f"chunk_bias must be 3-D [B, C+1, C+1], got "
                f"{tuple(self.chunk_bias.shape)}"
            )
        if self.rare_row_offsets.dim() != 2:
            raise ValueError(
                f"rare_row_offsets must be 2-D [B, Sq+1], got "
                f"{tuple(self.rare_row_offsets.shape)}"
            )
        batch_size = self.token_to_chunk_q.shape[0]
        if self.chunk_bias.shape[0] != batch_size:
            raise ValueError(
                f"chunk_bias batch dim {self.chunk_bias.shape[0]} != "
                f"token_to_chunk_q batch dim {batch_size}"
            )
        if self.chunk_bias.shape[1] != self.chunk_bias.shape[2]:
            raise ValueError(
                f"chunk_bias must be square [B, C+1, C+1], got "
                f"{tuple(self.chunk_bias.shape)}"
            )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_chunk_native_graph_bias(
    structure_batch: dict[str, torch.Tensor] | None,
    *,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    beta: float | None = None,
    softmax_scale: float | None = None,
    call_weight: float = 1.0,
    type_weight: float = 1.0,
    domain_weight: float = 1.0,
    build_weight: float = 1.0,
    shell_weight: float = 1.0,
    diagnostic_weight: float = 1.0,
    cross_domain_weight: float = 1.0,
    generated_query_weight: float = 1.0,
) -> ChunkNativeGraphBias:
    """Build chunk-native graph bias (vectorized, no per-edge GPU syncs).

    Extracts chunk layout and edge relations, builds the token-to-chunk maps,
    the chunk-pair bias matrix via scatter_add_, and the rare point-edge CSR
    overlay via vectorized torch operations.

    Falls back to ``_build_bias_slow`` if ``CPPMEGA_FA4_BIAS_SLOW_FALLBACK=1``.

    Args:
        structure_batch: The cppmega graph sidecar dict containing chunk
            layout and edge tensors.
        batch_size: Microbatch size B.
        seqlen_q: Query sequence length Sq.
        seqlen_k: Key sequence length Sk.
        device: Target device.
        dtype: Dtype for bias/weight tensors (typically float32 or bfloat16).
        beta: Graph bias beta; resolved from env if None.
        softmax_scale: Deprecated and ignored.
        call_weight: Weight for call edges.
        type_weight: Weight for type edges.
        domain_weight: Weight for domain edges.
        build_weight: Weight for build edges.
        shell_weight: Weight for shell edges.
        diagnostic_weight: Weight for diagnostic edges.
        cross_domain_weight: Weight for cross-domain edges.
        generated_query_weight: Weight for generated query edges.

    Returns:
        A frozen ``ChunkNativeGraphBias`` ready for score_mod consumption.

    Raises:
        RuntimeError: If structure_batch is None or no route tensors found.
        ValueError: On corrupt sidecar metadata or dimension violations.
    """
    if _env_flag("CPPMEGA_FA4_BIAS_SLOW_FALLBACK", "0"):
        return _build_bias_slow(
            structure_batch,
            batch_size=batch_size,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            device=device,
            dtype=dtype,
            beta=beta,
            softmax_scale=softmax_scale,
            call_weight=call_weight,
            type_weight=type_weight,
            domain_weight=domain_weight,
            build_weight=build_weight,
            shell_weight=shell_weight,
            diagnostic_weight=diagnostic_weight,
            cross_domain_weight=cross_domain_weight,
            generated_query_weight=generated_query_weight,
        )

    if structure_batch is None:
        raise RuntimeError(
            "FA4 chunk-native score_mod requires a structure batch; "
            "refusing token-only fallback"
        )
    if not isinstance(structure_batch, dict):
        raise TypeError(
            f"structure_batch must be a dict, got {type(structure_batch).__name__}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if seqlen_q <= 0:
        raise ValueError(f"seqlen_q must be positive, got {seqlen_q}")
    if seqlen_k <= 0:
        raise ValueError(f"seqlen_k must be positive, got {seqlen_k}")

    effective_beta = (
        resolve_graph_bias_beta() if beta is None else validate_graph_bias_beta(beta)
    )
    if softmax_scale is not None:
        warnings.warn(
            "softmax_scale is deprecated and ignored in "
            "build_chunk_native_graph_bias. FA4 applies scaling internally "
            "before calling score_mod; bias is added to already-scaled "
            "scores (TE post_scale_bias semantics).",
            DeprecationWarning,
            stacklevel=2,
        )
    weight_multiplier = effective_beta

    # --- Build token-to-chunk mappings ---
    chunk_layout = _as_batched_chunks(
        structure_batch, batch_size=batch_size, device=device
    )
    starts, ends, chunk_counts = chunk_layout
    max_chunks = int(starts.shape[1])

    chunk_ids_q, valid_q = _token_chunk_map(starts, ends, chunk_counts, length=seqlen_q)
    if seqlen_k == seqlen_q:
        chunk_ids_k, valid_k = chunk_ids_q, valid_q
    else:
        chunk_ids_k, valid_k = _token_chunk_map(starts, ends, chunk_counts, length=seqlen_k)

    sentinel = max_chunks
    token_to_chunk_q = torch.where(
        valid_q, chunk_ids_q, torch.full_like(chunk_ids_q, sentinel)
    ).to(torch.int32)
    token_to_chunk_k = torch.where(
        valid_k, chunk_ids_k, torch.full_like(chunk_ids_k, sentinel)
    ).to(torch.int32)

    # --- Build chunk_bias [B, C+1, C+1] via vectorized scatter_add_ ---
    c_plus_1 = max_chunks + 1
    chunk_bias = torch.zeros(
        (batch_size, c_plus_1, c_plus_1), device=device, dtype=dtype
    )
    # Flatten to [B, (C+1)*(C+1)] for scatter_add_ with linear indices.
    chunk_bias_flat = chunk_bias.view(batch_size, c_plus_1 * c_plus_1)

    seen_relation = False

    # Chunk-index relations: call, type — vectorized via scatter_add_.
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
        if int(edges.shape[0]) == 1 and batch_size > 1:
            edges = edges.expand(batch_size, -1, -1)
        if int(counts.shape[0]) == 1 and batch_size > 1:
            counts = counts.expand(batch_size)

        max_edges = int(edges.shape[1])
        # Validate counts are in valid range [0, max_edges].
        if bool((counts < 0).any()) or bool((counts > max_edges).any()):
            raise ValueError(
                f"{count_key} contains values outside [0, {max_edges}]; "
                f"got min={int(counts.min())}, max={int(counts.max())}"
            )
        # Build per-batch edge validity mask using counts (no .item() per edge).
        edge_idx = torch.arange(max_edges, device=device).unsqueeze(0)  # [1, E]
        valid_mask = edge_idx < counts.unsqueeze(1)  # [B, E]

        src_chunks = edges[:, :max_edges, 0].long()  # [B, E]
        dst_chunks = edges[:, :max_edges, 1].long()  # [B, E]

        # Validate: all active edges must reference available chunks.
        available = chunk_counts.unsqueeze(1).long()  # [B, 1]
        src_ok = (src_chunks >= 0) & (src_chunks < available)
        dst_ok = (dst_chunks >= 0) & (dst_chunks < available)
        invalid = valid_mask & ~(src_ok & dst_ok)
        if bool(invalid.any()):
            raise ValueError(
                f"declared edge references an unavailable chunk "
                f"(relation={edge_key})"
            )

        # Linear index into [C+1, C+1] flattened. Inactive/padded slots may
        # hold negative sentinels; clamp them into the sentinel row/column so
        # scatter_add_ stays in-bounds (their weight is zero anyway).
        src_safe = src_chunks.clamp(min=0, max=max_chunks)
        dst_safe = dst_chunks.clamp(min=0, max=max_chunks)
        flat_idx = src_safe * c_plus_1 + dst_safe  # [B, E]
        # Weight tensor: relation_weight * beta for valid edges, 0 otherwise.
        w = torch.where(
            valid_mask,
            torch.tensor(relation_weight * weight_multiplier, device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype),
        )  # [B, E]
        chunk_bias_flat.scatter_add_(1, flat_idx, w)

    # --- Rare point edges (token-position relations) — vectorized collection ---
    # Collect all rare edges as tensors per batch using vectorized ops.
    batch_rare_q: list[torch.Tensor] = []
    batch_rare_k: list[torch.Tensor] = []
    batch_rare_w: list[torch.Tensor] = []
    for _ in range(batch_size):
        batch_rare_q.append(torch.empty(0, device=device, dtype=torch.long))
        batch_rare_k.append(torch.empty(0, device=device, dtype=torch.long))
        batch_rare_w.append(torch.empty(0, device=device, dtype=dtype))

    # Token-triple relations: domain, build, shell, diagnostic, cross-domain.
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
        if bool(((counts < 0) | (counts > max_edges)).any()):
            raise ValueError(
                f"graph edge counts out of range [0,{max_edges}] for {edge_key}"
            )

        # Vectorized: build validity mask, filter active edges per batch.
        edge_idx = torch.arange(max_edges, device=device).unsqueeze(0)  # [1, E]
        valid_mask = edge_idx < counts.unsqueeze(1)  # [B, E]

        src = edges[:, :max_edges, 0].long()  # [B, E]
        dst = edges[:, :max_edges, 1].long()  # [B, E]
        kind = edges[:, :max_edges, 2].long()  # [B, E]

        # Active: valid slot AND kind >= 0.
        active = valid_mask & (kind >= 0)

        # Bounds check on active edges.
        src_ok = (src >= 0) & (src < seqlen_q)
        dst_ok = (dst >= 0) & (dst < seqlen_k)
        invalid = active & ~(src_ok & dst_ok)
        if bool(invalid.any()):
            raise ValueError(
                f"graph token edge outside sequence bounds "
                f"[0, {seqlen_q})x[0, {seqlen_k}) in {edge_key}"
            )

        # Collect per-batch using masked selection (no .item() per edge).
        w_val = relation_weight * weight_multiplier
        for b in range(batch_size):
            mask_b = active[b]  # [E]
            if not bool(mask_b.any()):
                continue
            batch_rare_q[b] = torch.cat([batch_rare_q[b], src[b][mask_b]])
            batch_rare_k[b] = torch.cat([batch_rare_k[b], dst[b][mask_b]])
            batch_rare_w[b] = torch.cat([
                batch_rare_w[b],
                torch.full(
                    (int(mask_b.sum()),), w_val, device=device, dtype=dtype
                ),
            ])

    # Generated query edges (token pairs, not triples) — vectorized.
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
        if bool(((gen_counts < 0) | (gen_counts > max_gen)).any()):
            raise ValueError(
                f"generated query edge counts out of range [0,{max_gen}]"
            )

        edge_idx = torch.arange(max_gen, device=device).unsqueeze(0)
        valid_mask = edge_idx < gen_counts.unsqueeze(1)

        src = gen_edges[:, :max_gen, 0].long()
        dst = gen_edges[:, :max_gen, 1].long()

        src_ok = (src >= 0) & (src < seqlen_q)
        dst_ok = (dst >= 0) & (dst < seqlen_k)
        invalid = valid_mask & ~(src_ok & dst_ok)
        if bool(invalid.any()):
            raise ValueError(
                f"generated query edge outside "
                f"sequence bounds [0, {seqlen_q})x[0, {seqlen_k})"
            )

        w_val = generated_query_weight * weight_multiplier
        for b in range(batch_size):
            mask_b = valid_mask[b]
            if not bool(mask_b.any()):
                continue
            batch_rare_q[b] = torch.cat([batch_rare_q[b], src[b][mask_b]])
            batch_rare_k[b] = torch.cat([batch_rare_k[b], dst[b][mask_b]])
            batch_rare_w[b] = torch.cat([
                batch_rare_w[b],
                torch.full(
                    (int(mask_b.sum()),), w_val, device=device, dtype=dtype
                ),
            ])

    if not seen_relation:
        raise RuntimeError(
            "FA4 chunk-native score_mod: structure batch contains no route "
            "tensors (expected graph_call_edges/type_edges or "
            "domain/build/shell/diagnostic/cross-domain/generated_query edges)"
        )

    # --- Build per-row CSR for rare edges (vectorized dedup + sort) ---
    max_rare = _fa4_max_rare_per_row()

    rare_q_padded = torch.zeros((batch_size, max_rare), device=device, dtype=torch.int32)
    rare_k_padded = torch.full((batch_size, max_rare), -1, device=device, dtype=torch.int32)
    rare_w_padded = torch.zeros((batch_size, max_rare), device=device, dtype=dtype)
    rare_row_offsets = torch.zeros(
        (batch_size, seqlen_q + 1), device=device, dtype=torch.int32
    )

    total_unique_declared = 0
    for b in range(batch_size):
        n_raw = batch_rare_q[b].numel()
        if n_raw == 0:
            continue

        q_arr = batch_rare_q[b]
        k_arr = batch_rare_k[b]
        w_arr = batch_rare_w[b]

        # Deduplicate: linear index = q * seqlen_k + k.
        linear = q_arr * seqlen_k + k_arr
        unique_linear, inverse = torch.unique(linear, return_inverse=True)
        n_unique = int(unique_linear.numel())
        total_unique_declared += n_unique
        summed_w = torch.zeros(n_unique, device=device, dtype=dtype)
        summed_w.index_add_(0, inverse, w_arr)

        unique_q = (unique_linear // seqlen_k).to(torch.int32)
        unique_k = (unique_linear % seqlen_k).to(torch.int32)

        # Sort by (q, k) ascending.
        sort_key = unique_q.long() * seqlen_k + unique_k.long()
        sort_order = torch.argsort(sort_key)
        sorted_q = unique_q[sort_order]
        sorted_k = unique_k[sort_order]
        sorted_w = summed_w[sort_order]

        if n_unique > max_rare:
            raise ValueError(
                f"rare edge overflow: batch element {b} has {n_unique} unique "
                f"rare edges, exceeding max_rare={max_rare}. Increase "
                f"CPPMEGA_FA4_MAX_RARE_PER_ROW (current limit: {max_rare}) or "
                f"investigate why so many token-level edges exist. Silent "
                f"truncation is not permitted: dropped edges change model "
                f"supervision without notification."
            )

        rare_q_padded[b, :n_unique] = sorted_q
        rare_k_padded[b, :n_unique] = sorted_k
        rare_w_padded[b, :n_unique] = sorted_w

        # Build row offsets via scatter + cumsum (no Python loop over edges).
        row_counts = torch.zeros(seqlen_q, device=device, dtype=torch.int32)
        row_counts.scatter_add_(
            0, sorted_q.long(),
            torch.ones(n_unique, device=device, dtype=torch.int32),
        )
        rare_row_offsets[b, 1:] = torch.cumsum(row_counts, dim=0)

    # Fail-closed invariant check: retained CSR entries must equal the number
    # of unique rare edges (duplicates are merged by weight summation above).
    total_retained = int(rare_row_offsets[:, -1].sum())
    assert total_retained == total_unique_declared, (
        f"rare edge overflow: {total_unique_declared} unique declared, "
        f"{total_retained} retained"
    )

    return ChunkNativeGraphBias(
        token_to_chunk_q=token_to_chunk_q,
        token_to_chunk_k=token_to_chunk_k,
        chunk_bias=chunk_bias_flat.view(batch_size, c_plus_1, c_plus_1),
        rare_row_offsets=rare_row_offsets,
        rare_q=rare_q_padded,
        rare_k=rare_k_padded,
        rare_w=rare_w_padded,
        max_chunks=max_chunks,
        beta=effective_beta,
    )


def _build_bias_slow(
    structure_batch: dict[str, torch.Tensor] | None,
    *,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    beta: float | None = None,
    softmax_scale: float | None = None,
    call_weight: float = 1.0,
    type_weight: float = 1.0,
    domain_weight: float = 1.0,
    build_weight: float = 1.0,
    shell_weight: float = 1.0,
    diagnostic_weight: float = 1.0,
    cross_domain_weight: float = 1.0,
    generated_query_weight: float = 1.0,
) -> ChunkNativeGraphBias:
    """SLOW FALLBACK: Build chunk-native graph bias with per-edge .item() syncs.

    Retained as a correctness reference.  The production path is the vectorized
    ``build_chunk_native_graph_bias`` which eliminates all GPU syncs.
    """
    if structure_batch is None:
        raise RuntimeError(
            "FA4 chunk-native score_mod requires a structure batch; "
            "refusing token-only fallback"
        )
    if not isinstance(structure_batch, dict):
        raise TypeError(
            f"structure_batch must be a dict, got {type(structure_batch).__name__}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if seqlen_q <= 0:
        raise ValueError(f"seqlen_q must be positive, got {seqlen_q}")
    if seqlen_k <= 0:
        raise ValueError(f"seqlen_k must be positive, got {seqlen_k}")

    effective_beta = (
        resolve_graph_bias_beta() if beta is None else validate_graph_bias_beta(beta)
    )
    if softmax_scale is not None:
        warnings.warn(
            "softmax_scale is deprecated and ignored in "
            "build_chunk_native_graph_bias. FA4 applies scaling internally "
            "before calling score_mod; bias is added to already-scaled "
            "scores (TE post_scale_bias semantics).",
            DeprecationWarning,
            stacklevel=2,
        )
    weight_multiplier = effective_beta

    # --- Build token-to-chunk mappings ---
    chunk_layout = _as_batched_chunks(
        structure_batch, batch_size=batch_size, device=device
    )
    starts, ends, chunk_counts = chunk_layout
    max_chunks = int(starts.shape[1])

    # _token_chunk_map returns (chunk_ids [B, S], valid [B, S])
    chunk_ids_q, valid_q = _token_chunk_map(starts, ends, chunk_counts, length=seqlen_q)
    if seqlen_k == seqlen_q:
        chunk_ids_k, valid_k = chunk_ids_q, valid_q
    else:
        chunk_ids_k, valid_k = _token_chunk_map(starts, ends, chunk_counts, length=seqlen_k)

    # Sentinel: tokens in inter-chunk gaps get chunk id = max_chunks (the C slot).
    sentinel = max_chunks
    token_to_chunk_q = torch.where(
        valid_q, chunk_ids_q, torch.full_like(chunk_ids_q, sentinel)
    ).to(torch.int32)
    token_to_chunk_k = torch.where(
        valid_k, chunk_ids_k, torch.full_like(chunk_ids_k, sentinel)
    ).to(torch.int32)

    # --- Build chunk_bias [B, C+1, C+1] ---
    c_plus_1 = max_chunks + 1
    chunk_bias = torch.zeros(
        (batch_size, c_plus_1, c_plus_1), device=device, dtype=dtype
    )

    seen_relation = False

    # Chunk-index relations: call, type.
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
                        f"declared edge ({src_chunk},{dst_chunk}) references "
                        f"an unavailable chunk (available={available}, "
                        f"batch={b}, relation={edge_key})"
                    )
                chunk_bias[b, src_chunk, dst_chunk] += (
                    relation_weight * weight_multiplier
                )

    # --- Rare point edges (token-position relations) ---
    # Collect all rare edges as flat lists per batch, then build per-row CSR.
    # Use seqlen_k for key bounds since rare edges index into key positions.
    batch_rare_q: list[list[int]] = [[] for _ in range(batch_size)]
    batch_rare_k: list[list[int]] = [[] for _ in range(batch_size)]
    batch_rare_w: list[list[float]] = [[] for _ in range(batch_size)]

    # Token-triple relations: domain, build, shell, diagnostic, cross-domain.
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
            raise ValueError(
                f"graph edge counts out of range [0,{max_edges}] for {edge_key}"
            )

        for b in range(batch_size):
            n_edges = int(counts[b].item())
            for ei in range(n_edges):
                src = int(edges[b, ei, 0].item())
                dst = int(edges[b, ei, 1].item())
                kind = int(edges[b, ei, 2].item())
                if kind < 0:
                    continue  # inactive triple
                if not (0 <= src < seqlen_q and 0 <= dst < seqlen_k):
                    raise ValueError(
                        f"graph token edge ({src},{dst}) is outside "
                        f"sequence bounds [0, {seqlen_q})x[0, {seqlen_k}) "
                        f"in {edge_key}"
                    )
                batch_rare_q[b].append(src)
                batch_rare_k[b].append(dst)
                batch_rare_w[b].append(relation_weight * weight_multiplier)

    # Generated query edges (token pairs, not triples).
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
            raise ValueError(
                f"generated query edge counts out of range [0,{max_gen}]"
            )
        for b in range(batch_size):
            n_gen = int(gen_counts[b].item())
            for ei in range(n_gen):
                src = int(gen_edges[b, ei, 0].item())
                dst = int(gen_edges[b, ei, 1].item())
                if not (0 <= src < seqlen_q and 0 <= dst < seqlen_k):
                    raise ValueError(
                        f"generated query edge ({src},{dst}) outside "
                        f"sequence bounds [0, {seqlen_q})x[0, {seqlen_k})"
                    )
                batch_rare_q[b].append(src)
                batch_rare_k[b].append(dst)
                batch_rare_w[b].append(generated_query_weight * weight_multiplier)

    if not seen_relation:
        raise RuntimeError(
            "FA4 chunk-native score_mod: structure batch contains no route "
            "tensors (expected graph_call_edges/type_edges or "
            "domain/build/shell/diagnostic/cross-domain/generated_query edges)"
        )

    # --- Build per-row CSR for rare edges ---
    # Deduplicate (q, k) pairs per batch by summing weights (matches the
    # dense path's index_add_ semantics).
    # Format: rare_row_offsets [B, Sq+1], rare_k [B, max_rare], rare_w [B, max_rare]
    # Within each row, keys are sorted ascending.
    #
    # max_rare is a FIXED high-water mark (env CPPMEGA_FA4_MAX_RARE_PER_ROW,
    # default 256) so that aux tensor shapes are stable across batches.  This
    # prevents FA4 recompilation due to shape changes between steps.  Unused
    # slots are padded with k=-1 (sentinel that never matches a valid key).

    max_rare = _fa4_max_rare_per_row()

    # First pass: deduplicate and sort per batch.
    batch_sorted_q: list[torch.Tensor] = []
    batch_sorted_k: list[torch.Tensor] = []
    batch_sorted_w: list[torch.Tensor] = []

    for b in range(batch_size):
        if not batch_rare_q[b]:
            batch_sorted_q.append(torch.empty(0, device=device, dtype=torch.int32))
            batch_sorted_k.append(torch.empty(0, device=device, dtype=torch.int32))
            batch_sorted_w.append(torch.empty(0, device=device, dtype=dtype))
            continue

        q_arr = torch.tensor(batch_rare_q[b], device=device, dtype=torch.long)
        k_arr = torch.tensor(batch_rare_k[b], device=device, dtype=torch.long)
        w_arr = torch.tensor(batch_rare_w[b], device=device, dtype=dtype)

        # Deduplicate: linear index = q * seqlen_k + k.
        linear = q_arr * seqlen_k + k_arr
        unique_linear, inverse = torch.unique(linear, return_inverse=True)
        n_unique = int(unique_linear.numel())
        summed_w = torch.zeros(n_unique, device=device, dtype=dtype)
        summed_w.index_add_(0, inverse, w_arr)

        unique_q = (unique_linear // seqlen_k).to(torch.int32)
        unique_k = (unique_linear % seqlen_k).to(torch.int32)

        # Sort by (q, k) ascending.
        sort_key = unique_q.long() * seqlen_k + unique_k.long()
        sort_order = torch.argsort(sort_key)
        sorted_q = unique_q[sort_order]
        sorted_k = unique_k[sort_order]
        sorted_w = summed_w[sort_order]

        if n_unique > max_rare:
            raise ValueError(
                f"rare edge overflow: batch element {b} has {n_unique} unique "
                f"rare edges, exceeding max_rare={max_rare}. Increase "
                f"CPPMEGA_FA4_MAX_RARE_PER_ROW (current limit: {max_rare}) or "
                f"investigate why so many token-level edges exist. Silent "
                f"truncation is not permitted: dropped edges change model "
                f"supervision without notification."
            )

        batch_sorted_q.append(sorted_q)
        batch_sorted_k.append(sorted_k)
        batch_sorted_w.append(sorted_w)

    # Build padded [B, max_rare] tensors and [B, Sq+1] row offsets.
    # Pad rare_k with -1 sentinel (never matches a valid key position >= 0).
    rare_q_padded = torch.zeros((batch_size, max_rare), device=device, dtype=torch.int32)
    rare_k_padded = torch.full((batch_size, max_rare), -1, device=device, dtype=torch.int32)
    rare_w_padded = torch.zeros((batch_size, max_rare), device=device, dtype=dtype)
    rare_row_offsets = torch.zeros(
        (batch_size, seqlen_q + 1), device=device, dtype=torch.int32
    )

    for b in range(batch_size):
        n = int(batch_sorted_q[b].numel())
        if n == 0:
            continue
        rare_q_padded[b, :n] = batch_sorted_q[b]
        rare_k_padded[b, :n] = batch_sorted_k[b]
        rare_w_padded[b, :n] = batch_sorted_w[b]

        # Build row offsets: for each query row q, count edges with that q.
        sorted_q_list = batch_sorted_q[b].long()
        # Use scatter to count per-row edges.
        row_counts = torch.zeros(seqlen_q, device=device, dtype=torch.int32)
        row_counts.scatter_add_(
            0, sorted_q_list, torch.ones(n, device=device, dtype=torch.int32)
        )
        # Per-row overflow check: catch individual rows exceeding max_rare.
        max_row_count = int(row_counts.max().item())
        if max_row_count > max_rare:
            worst_row = int(row_counts.argmax().item())
            raise ValueError(
                f"rare edge overflow: batch element {b}, query row {worst_row} "
                f"has {max_row_count} rare edges, exceeding max_rare={max_rare}. "
                f"Increase CPPMEGA_FA4_MAX_RARE_PER_ROW (current limit: "
                f"{max_rare}) or investigate the graph structure."
            )
        # Prefix sum to get offsets.
        rare_row_offsets[b, 1:] = torch.cumsum(row_counts, dim=0)

    # Fail-closed invariant: every declared rare edge must be present in the
    # CSR overlay.  If this fires, something silently dropped edges above.
    total_declared = sum(int(batch_sorted_q[b].numel()) for b in range(batch_size))
    total_retained = int(rare_row_offsets[:, -1].sum().item())
    assert total_retained == total_declared, (
        f"rare edge overflow: {total_declared} declared, {total_retained} retained"
    )

    return ChunkNativeGraphBias(
        token_to_chunk_q=token_to_chunk_q,
        token_to_chunk_k=token_to_chunk_k,
        chunk_bias=chunk_bias,
        rare_row_offsets=rare_row_offsets,
        rare_q=rare_q_padded,
        rare_k=rare_k_padded,
        rare_w=rare_w_padded,
        max_chunks=max_chunks,
        beta=effective_beta,
    )


# ---------------------------------------------------------------------------
# Production wiring entry point
# ---------------------------------------------------------------------------


def build_fa4_attention_bias_from_structure_batch(
    structure_batch: dict[str, torch.Tensor] | None,
    *,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    beta: float | None = None,
    softmax_scale: float | None = None,
    call_weight: float = 1.0,
    type_weight: float = 1.0,
    domain_weight: float = 1.0,
    build_weight: float = 1.0,
    shell_weight: float = 1.0,
    diagnostic_weight: float = 1.0,
    cross_domain_weight: float = 1.0,
    generated_query_weight: float = 1.0,
) -> ChunkNativeGraphBias:
    """Production wiring: build ChunkNativeGraphBias from the structure batch.

    Called by ``graph_route_attention_bias_patch`` when FA4 score_mod is
    enabled instead of building a dense ``[B,1,Sq,Sk]`` tensor.  Delegates
    to :func:`build_chunk_native_graph_bias`.

    Args:
        structure_batch: The cppmega graph sidecar dict.
        batch_size: Microbatch size B.
        seqlen_q: Query sequence length Sq.
        seqlen_k: Key sequence length Sk.
        device: Target device.
        dtype: Dtype for bias/weight tensors.
        beta: Graph bias beta; resolved from env if None.
        softmax_scale: Attention softmax scale (1/sqrt(head_dim)).
        call_weight: Weight for call edges.
        type_weight: Weight for type edges.
        domain_weight: Weight for domain edges.
        build_weight: Weight for build edges.
        shell_weight: Weight for shell edges.
        diagnostic_weight: Weight for diagnostic edges.
        cross_domain_weight: Weight for cross-domain edges.
        generated_query_weight: Weight for generated query edges.

    Returns:
        A frozen ``ChunkNativeGraphBias`` ready for the FA4 attention module.
    """
    return build_chunk_native_graph_bias(
        structure_batch,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        device=device,
        dtype=dtype,
        beta=beta,
        softmax_scale=softmax_scale,
        call_weight=call_weight,
        type_weight=type_weight,
        domain_weight=domain_weight,
        build_weight=build_weight,
        shell_weight=shell_weight,
        diagnostic_weight=diagnostic_weight,
        cross_domain_weight=cross_domain_weight,
        generated_query_weight=generated_query_weight,
    )


# ---------------------------------------------------------------------------
# Python reference score_mod (for testing without GPU)
# ---------------------------------------------------------------------------


def chunk_native_score_mod_ref(
    bias: ChunkNativeGraphBias,
    *,
    score: float,
    batch: int,
    head: int,
    seqlen_info: Any = None,
    q: int,
    k: int,
) -> float:
    """Python reference score_mod: score + chunk_bias + rare_edge_weight.

    Mirrors the FA4 kernel logic for unit testing on CPU.

    Args:
        bias: The ChunkNativeGraphBias state.
        score: Pre-computed attention score.
        batch: Batch index.
        head: Head index.
        seqlen_info: FA4 beta23 ABI parameter (seqlen_q, seqlen_k); unused
            in the reference but accepted for signature parity.
        q: Query position index.
        k: Key position index.
    """
    qc = int(bias.token_to_chunk_q[batch, q].item())
    kc = int(bias.token_to_chunk_k[batch, k].item())
    chunk_val = float(bias.chunk_bias[batch, qc, kc].item())

    # Rare edge lookup: scan row q's CSR entries.
    rare_val = 0.0
    lo = int(bias.rare_row_offsets[batch, q].item())
    hi = int(bias.rare_row_offsets[batch, q + 1].item())
    for i in range(lo, hi):
        k_i = int(bias.rare_k[batch, i].item())
        if k_i == k:
            rare_val = float(bias.rare_w[batch, i].item())
            break
        if k_i > k:
            break

    return score + chunk_val + rare_val


def chunk_native_score_mod_bwd_ref(
    bias: ChunkNativeGraphBias,
    *,
    grad_out: Any,
    batch: int,
    head: int,
    seqlen_info: Any = None,
    q: int,
    k: int,
) -> Any:
    """Python reference score_mod_bwd: identity (additive bias has grad=1).

    Since score' = score + const, d(score')/d(score) = 1, so grad passes
    through unchanged.

    Args:
        bias: The ChunkNativeGraphBias state.
        grad_out: Upstream gradient.
        batch: Batch index.
        head: Head index.
        seqlen_info: FA4 beta23 ABI parameter (seqlen_q, seqlen_k); unused
            in the reference but accepted for signature parity.
        q: Query position index.
        k: Key position index.
    """
    if isinstance(grad_out, torch.Tensor):
        if grad_out.dim() == 4:
            return float(grad_out[batch, head, q, k].item())
        return float(grad_out.item())
    return grad_out


# ---------------------------------------------------------------------------
# Factory: _make_graph_score_mod / _make_graph_score_mod_bwd
# ---------------------------------------------------------------------------
# c_plus_1 is captured as a compile-time Python constant in the closure.
# max_rare is passed for tensor allocation sizing only; the inner rare-edge
# loop uses dynamic per-row bounds from rare_row_offsets (CSR pointers),
# NOT a static range(max_rare).  This eliminates O(max_rare) work per score
# element for rows with few or zero rare edges.
#
# FA4 b19 and beta23 both use the same kwargs ABI (verified on H200):
#     score_mod(score, batch_idx, head_idx, *, q_idx, kv_idx, seqlen_info, aux_tensors)


def _make_graph_score_mod(c_plus_1: int, max_rare: int) -> Any:
    """Create a beta23 score_mod with c_plus_1 and max_rare captured in closure.

    The returned callable expects 6 flat aux_tensors:
        [0] token_to_chunk_q  [B, S] int32
        [1] token_to_chunk_k  [B, S] int32
        [2] chunk_bias_flat   [B, (C+1)*(C+1)] float32
        [3] rare_row_offsets  [B, S+1] int32  (CSR row pointers)
        [4] rare_k            [B, max_rare] int32
        [5] rare_w            [B, max_rare] float32

    The rare-edge overlay uses CSR row_offsets to dynamically bound the
    inner loop to only the edges belonging to the current query row:
    range(row_offsets[b, q], row_offsets[b, q+1]).  Rows with 0 rare
    edges execute 0 iterations; rows with N edges execute exactly N.

    Args:
        c_plus_1: Number of chunks + 1 (sentinel dimension of chunk_bias).
        max_rare: Fixed high-water mark for rare edges per batch item
            (used for tensor allocation; the inner loop uses dynamic
            per-row bounds from rare_row_offsets, not this constant).

    Returns:
        A score_mod callable with the beta23 keyword-only ABI.
    """

    def _score_mod(
        score: Any,
        batch_idx: Any,
        head_idx: Any,
        *,
        q_idx: Any,
        kv_idx: Any,
        seqlen_info: Any,
        aux_tensors: list[Any],
    ) -> Any:
        token_to_chunk_q = aux_tensors[0]
        token_to_chunk_k = aux_tensors[1]
        chunk_bias_flat = aux_tensors[2]
        rare_row_offsets = aux_tensors[3]
        rare_k = aux_tensors[4]
        rare_w = aux_tensors[5]

        # Extract scalars from vector<1xi32> (CuTe DSL requirement)
        b = batch_idx[0]
        qi = q_idx[0]
        ki = kv_idx[0]

        # Chunk-pair gather via flat indexing
        qc = token_to_chunk_q[b, qi]
        kc = token_to_chunk_k[b, ki]
        flat_idx = qc * c_plus_1 + kc
        bias_val = chunk_bias_flat[b, flat_idx]
        out = score + bias_val

        # Rare token-edge overlay: static unrolled scan over all slots.
        # max_rare is a Python int (compile-time constant captured in the
        # closure), so range(max_rare) is a static loop that the CuTe DSL
        # unrolls at JIT compile time.  Dynamic loops (cutlass.range /
        # cutlass.range_dynamic) require @cutlass.jit AST preprocessing which
        # is unavailable for score_mod callbacks called from FA4's kernel.
        # Masking with (i >= lo) & (i < hi) ensures only valid CSR entries
        # for the current query row contribute to the output.
        lo = rare_row_offsets[b, qi]
        hi = rare_row_offsets[b, qi + 1]
        for i in range(max_rare):
            in_range = (i >= lo) & (i < hi)
            k_match = rare_k[b, i] == ki
            out = out + in_range * k_match * rare_w[b, i]

        return out

    return _score_mod


def _make_graph_score_mod_bwd(c_plus_1: int) -> Any:
    """Create a beta23 score_mod_bwd with c_plus_1 captured in closure.

    Identity backward: additive bias has d(score')/d(score) = 1.

    Args:
        c_plus_1: Number of chunks + 1 (unused in backward, kept for
            symmetry with _make_graph_score_mod).

    Returns:
        A score_mod_bwd callable with the beta23 keyword-only ABI.
    """

    def _score_mod_bwd(
        grad_out: Any,
        score: Any,
        batch_idx: Any,
        head_idx: Any,
        *,
        q_idx: Any,
        kv_idx: Any,
        seqlen_info: Any,
        aux_tensors: list[Any],
    ) -> Any:
        return grad_out

    return _score_mod_bwd


def _build_document_mask_aux(
    *,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return beta23 aux tensors for packed-document attention masking."""

    if seqlen_q > seqlen_k:
        raise ValueError(
            f"FA4 document masking requires Sq <= Sk, got Sq={seqlen_q} Sk={seqlen_k}"
        )

    from cppmega.megatron.document_isolation import document_layout

    document_ids, _spans, multiple_documents = document_layout(
        batch_size=batch_size,
        sequence_length=seqlen_k,
        device=device,
    )
    if not multiple_documents:
        return None
    if document_ids is None:
        raise RuntimeError("validated document layout did not return document_ids")

    query_start = seqlen_k - seqlen_q
    document_ids_k = document_ids.to(dtype=torch.int32).contiguous()
    document_ids_q = document_ids_k[
        :, query_start : query_start + seqlen_q
    ].contiguous()
    # Keep a singleton head axis so the decorated CuTe callback can use the
    # official beta23 [batch, head, token] indexing pattern without copying
    # the same document IDs for every attention head.
    return document_ids_q[:, None, :], document_ids_k[:, None, :]


def _make_document_causal_mask_mod(
    document_ids_q_index: int,
    document_ids_k_index: int,
    *,
    causal: bool,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
) -> Any:
    """Create a beta23 ``mask_mod`` that keeps one causal document at a time."""

    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    # Do not use FA4's fast_sampling decorator here. Its five-point tile
    # classifier is only sound for simple geometric masks; arbitrary packed
    # document runs can contain valid pairs even when all sampled points miss.
    @cute.jit
    def _mask_mod(
        batch_idx: Any,
        head_idx: Any,
        q_idx: Any,
        kv_idx: Any,
        seqlen_info: Any,
        aux_tensors: list[Any],
    ) -> Any:
        document_ids_q = aux_tensors[document_ids_q_index]
        document_ids_k = aux_tensors[document_ids_k_index]
        q_read_idx = cutlass.min(q_idx[0], seqlen_info.seqlen_q - 1)
        kv_read_idx = cutlass.min(kv_idx[0], seqlen_info.seqlen_k - 1)
        q_document = utils.scalar_to_ssa(
            document_ids_q[batch_idx[0], 0, q_read_idx], cutlass.Int32
        )
        kv_document = utils.scalar_to_ssa(
            document_ids_k[batch_idx[0], 0, kv_read_idx], cutlass.Int32
        )
        query_start = utils.scalar_to_ssa(
            seqlen_info.seqlen_k - seqlen_info.seqlen_q,
            cutlass.Int32,
        )
        q_absolute = q_idx + query_start

        # doc_id=0 is the trailing padding segment. Keeping 0 -> 0 causal
        # attention avoids fully-masked softmax rows; its validated loss_mask
        # is zero, and equality still isolates it from every real document.
        keep = q_document == kv_document
        if cutlass.const_expr(causal):
            keep = keep & (kv_idx <= q_absolute)
        if cutlass.const_expr(
            window_size_left is not None and window_size_left >= 0
        ):
            window_left = utils.scalar_to_ssa(window_size_left, cutlass.Int32)
            keep = keep & (kv_idx >= q_absolute - window_left)
        if cutlass.const_expr(
            window_size_right is not None and window_size_right >= 0
        ):
            window_right = utils.scalar_to_ssa(window_size_right, cutlass.Int32)
            keep = keep & (kv_idx <= q_absolute + window_right)
        return keep

    return _mask_mod


# ---------------------------------------------------------------------------
# CppMegaFA4ScoreModAttention module
# ---------------------------------------------------------------------------


class CppMegaFA4ScoreModAttention(torch.nn.Module):
    """FA4-backed core attention with chunk-native graph-route score_mod.

    Drop-in replacement for ``TEDotProductAttention`` in Megatron's
    ``ModuleSpec`` wiring.  QKV projections, FP8 GEMMs, RoPE, and output
    projection stay in TE; only the dot-product/softmax kernel changes.

    The module refuses dense ``attention_bias`` tensors by contract (the
    whole point is to avoid materializing them).  Pass ``ChunkNativeGraphBias``
    via the ``attention_bias`` argument or let the patch wrapper build it.

    Enable with ``CPPMEGA_FA4_SCORE_MOD=1``.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        num_attention_heads: int | None = None,
        head_dim: int | None = None,
        attention_dropout: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = True,
        deterministic: bool = False,
        beta: float | None = None,
        call_weight: float = 1.0,
        type_weight: float = 1.0,
        domain_weight: float = 1.0,
        build_weight: float = 1.0,
        shell_weight: float = 1.0,
        diagnostic_weight: float = 1.0,
        cross_domain_weight: float = 1.0,
        generated_query_weight: float = 1.0,
        **_ignored_te_kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None and getattr(config, "attention_dropout", 0) > 0:
            raise ValueError(
                "CppMegaFA4ScoreModAttention does not support attention "
                f"dropout (config.attention_dropout={config.attention_dropout}); "
                "FA4 score_mod path has no dropout support. Set "
                "attention_dropout=0 in TransformerConfig."
            )
        if not causal:
            raise ValueError(
                "CppMegaFA4ScoreModAttention is causal-only (POC constraint)"
            )
        self.config = config
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.deterministic = deterministic
        self.beta = beta
        self.call_weight = call_weight
        self.type_weight = type_weight
        self.domain_weight = domain_weight
        self.build_weight = build_weight
        self.shell_weight = shell_weight
        self.diagnostic_weight = diagnostic_weight
        self.cross_domain_weight = cross_domain_weight
        self.generated_query_weight = generated_query_weight
        self._first_forward_logged = False

    def _log_first_use(self) -> None:
        if not self._first_forward_logged:
            self._first_forward_logged = True
            log.info(
                "[cppmega] FA4 chunk-native score_mod attention active "
                "(causal=%s, deterministic=%s)",
                self.causal,
                self.deterministic,
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
        """Run FA4 attention with chunk-native graph-route score_mod.

        ABI contract: Megatron calls core_attention as::

            core_attention(query, key, value, attention_mask,
                           attn_mask_type=..., attention_bias=...,
                           packed_seq_params=...)

        QKV are ``[S, B, H, D]`` (sequence-first).  This module transposes to
        ``[B, S, H, D]`` for FA4, then reshapes the output to ``[S, B, H*D]``
        (3-D) as expected by Megatron's ``linear_proj``.

        Args:
            query: ``[S, B, H, D]`` projected queries (Megatron SBHD layout).
            key: ``[S, B, Hk, D]`` projected keys (Megatron SBHD layout).
            value: ``[S, B, Hk, D]`` projected values (Megatron SBHD layout).
            attention_mask: Ignored (FA4 handles masking via causal).
            attn_mask_type: Megatron AttnMaskType enum. If AttnMaskType.causal,
                causal masking is enabled. If AttnMaskType.no_mask, disabled.
            attention_bias: One of:
                - ``None``: plain FA4 attention (no score_mod).
                - ``ChunkNativeGraphBias``: chunk-native graph-route bias.
                - ``torch.Tensor``: RAISES (dense bias refused by contract).
            packed_seq_params: Must be None (not supported).
            inference_context: Optional; unused in POC.

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
        q = query.transpose(0, 1)  # [S,B,H,D] -> [B,S,H,D]
        k = key.transpose(0, 1)    # [S,B,Hk,D] -> [B,S,Hk,D]
        v = value.transpose(0, 1)  # [S,B,Hk,D] -> [B,S,Hk,D]

        batch_size, seqlen_q, num_heads, head_dim = q.shape
        _, seqlen_k, num_kv_heads, _ = k.shape
        document_mask_aux = _build_document_mask_aux(
            batch_size=batch_size,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            device=q.device,
        )

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads}) for GQA"
            )
        if packed_seq_params is not None:
            raise ValueError(
                "CppMegaFA4ScoreModAttention does not support packed_seq_params"
            )

        # Refuse dense tensor bias by contract.
        if isinstance(attention_bias, torch.Tensor):
            raise TypeError(
                "CppMegaFA4ScoreModAttention refuses dense attention_bias "
                "tensors by contract. Pass a ChunkNativeGraphBias or None. "
                "The FA4 backend exists precisely to avoid O(B*Sq*Sk) "
                "materialization."
            )

        # Sequence parallel / context parallel guard.
        if self.config is not None:
            if getattr(self.config, "sequence_parallel", False):
                raise RuntimeError(
                    "FA4 chunk-native score_mod does not support "
                    "sequence_parallel"
                )
            if int(getattr(self.config, "context_parallel_size", 1)) != 1:
                raise RuntimeError(
                    "FA4 chunk-native score_mod does not support "
                    "context_parallel_size > 1"
                )

        self._log_first_use()

        # --- Resolve causal from attn_mask_type (Megatron ABI) ---
        causal = self.causal
        if attn_mask_type is not None:
            try:
                from megatron.core.transformer.enums import AttnMaskType

                if attn_mask_type == AttnMaskType.causal:
                    causal = True
                elif attn_mask_type == AttnMaskType.no_mask:
                    causal = False
            except ImportError:
                pass  # If megatron enums unavailable, fall back to self.causal

        # --- Resolve ChunkNativeGraphBias from attention_bias kwarg ---
        bias_state: ChunkNativeGraphBias | None = None
        if isinstance(attention_bias, ChunkNativeGraphBias):
            bias_state = attention_bias
        elif attention_bias is not None:
            raise TypeError(
                f"attention_bias must be None or ChunkNativeGraphBias, got "
                f"{type(attention_bias).__name__}"
            )

        # --- Resolve softmax scale ---
        scale = self.softmax_scale
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # --- Call FA4 (lazy import for GPU-free testability) ---
        try:
            from flash_attn.cute.interface import flash_attn_func
        except ImportError as exc:
            raise RuntimeError(
                "FA4 chunk-native score_mod requires flash_attn.cute "
                "(flash-attn-4 beta23+). Install with: "
                "pip install 'flash-attn-4[cu13]>=4.0.0b23'. "
                f"Import error: {exc}"
            ) from exc

        score_mod_fn = None
        score_mod_bwd_fn = None
        aux_tensors_arg = None
        mask_mod_fn = None

        if bias_state is not None:
            # Pack aux_tensors from ChunkNativeGraphBias.
            # Order: [token_to_chunk_q, token_to_chunk_k, chunk_bias_flat,
            #         rare_row_offsets, rare_k, rare_w]
            # chunk_bias is flattened from [B, C+1, C+1] to [B, (C+1)*(C+1)]
            # because the score_mod closure uses flat_idx = qc * c_plus_1 + kc.
            # c_plus_1 is captured as a compile-time closure constant (NOT an
            # aux_tensors element) since FA4's to_cute_aux_tensor() cannot
            # convert Python int scalars.
            chunk_bias = bias_state.chunk_bias  # [B, C+1, C+1]
            c_plus_1 = chunk_bias.shape[1]
            batch_size = chunk_bias.shape[0]
            chunk_bias_flat = chunk_bias.reshape(batch_size, -1).contiguous()

            aux_tensors_arg = [
                bias_state.token_to_chunk_q,
                bias_state.token_to_chunk_k,
                chunk_bias_flat,
                bias_state.rare_row_offsets,
                bias_state.rare_k,
                bias_state.rare_w,
            ]
            # FA4 b19 and beta23 both use the same kwargs ABI (verified on H200)
            # max_rare sizes the aux tensors; the inner loop uses dynamic CSR bounds
            max_rare = int(bias_state.rare_k.shape[1])
            score_mod_fn = _make_graph_score_mod(c_plus_1, max_rare)
            score_mod_bwd_fn = _make_graph_score_mod_bwd(c_plus_1)
            if not self._first_forward_logged:
                log.info(
                    "[cppmega] FA4 score_mod ABI: kwargs (b19/beta23 unified)"
                )

        if document_mask_aux is not None:
            if aux_tensors_arg is None:
                aux_tensors_arg = []
            document_ids_q_index = len(aux_tensors_arg)
            aux_tensors_arg.extend(document_mask_aux)
            mask_mod_fn = _make_document_causal_mask_mod(
                document_ids_q_index,
                document_ids_q_index + 1,
                causal=causal,
            )

        # In FA4 4.0.0b23's SM90 dispatch, the native causal/local branch does
        # not apply mask_mod. For packed rows the callback therefore owns the
        # complete document + causal predicate; single-document rows retain
        # the native causal fast path.
        native_causal = causal if mask_mod_fn is None else False
        out = flash_attn_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=scale,
            causal=native_causal,
            score_mod=score_mod_fn,
            score_mod_bwd=score_mod_bwd_fn,
            aux_tensors=aux_tensors_arg,
            block_sparse_tensors=None,
            mask_mod=mask_mod_fn,
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
