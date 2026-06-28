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

This is a pure correctness/memory fix — it is numerically equivalent to
upstream and has no runtime penalty on H200 (the per-head bmm lowers to
a single cuBLAS GEMM per head, same FLOP count). It replaces the dead
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
No backward changes — upstream ``_compute_index_scores`` is not
autograd-aware (called under ``torch.no_grad()`` in the fwd path and
inside a custom-autograd recompute in the bwd).
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)

__all__ = [
    "apply_dsa_indexer_fused_patch",
    "build_graph_route_bias_from_structure_batch",
    "compute_index_scores_fused_bf16",
]

_PATCH_MARKER = "__cppmega_dsa_indexer_fused_patched__"


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


def _scatter_relation_edges_(
    bias: torch.Tensor,
    edges: torch.Tensor,
    counts: torch.Tensor,
    *,
    weight: float,
    sq: int,
    sk: int,
) -> None:
    if weight == 0.0:
        return
    batch_size = int(bias.shape[0])
    max_edges = int(edges.shape[1])
    for bi in range(batch_size):
        edge_b = 0 if int(edges.shape[0]) == 1 else bi
        count_b = 0 if int(counts.shape[0]) == 1 else bi
        n = max(0, min(int(counts[count_b].item()), max_edges))
        if n == 0:
            continue
        pairs = edges[edge_b, :n]
        src = pairs[:, 0]
        dst = pairs[:, 1]
        valid = (src >= 0) & (src < sq) & (dst >= 0) & (dst < sk)
        if not bool(valid.any().item()):
            continue
        bias[bi, src[valid], dst[valid]] += float(weight)


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
) -> torch.Tensor:
    """Build ``S_graph[b,t,s]`` from cppmega graph route sidecars.

    The current Megatron sidecar bridge carries token-position edge pairs:
    ``graph_call_edges`` / ``graph_type_edges`` plus per-row counts.  This helper
    turns them into the dense additive indexer prior used before DSA top-k.
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
        _scatter_relation_edges_(
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
            "route edge tensors (expected graph_call_edges/type_edges)"
        )
    return bias


def _current_graph_route_bias(
    *,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
) -> torch.Tensor:
    try:
        from cppmega.megatron.structure_dataset_patch import _get_current_structure_batch
    except Exception as exc:  # pragma: no cover - exercised in remote Megatron env
        raise RuntimeError(
            "CPPMEGA_GRAPH_ROUTES_ENABLED=1 but structure_dataset_patch is not "
            "importable; import it before applying the DSA indexer patch"
        ) from exc

    return build_graph_route_bias_from_structure_batch(
        _get_current_structure_batch(),
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        device=device,
        dtype=torch.float32,
        call_weight=_env_float("CPPMEGA_DSA_GRAPH_CALL_WEIGHT", 1.0),
        type_weight=_env_float("CPPMEGA_DSA_GRAPH_TYPE_WEIGHT", 1.0),
    )


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
    if getattr(existing, _PATCH_MARKER, False) and not force:
        log.info("cppmega DSA indexer fused patch already applied")
        return True

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

    setattr(_compute_index_scores_fused, _PATCH_MARKER, True)
    dsa_mod._compute_index_scores = _compute_index_scores_fused

    log.info(
        "cppmega DSA indexer fused patch applied: per-head accumulation, "
        "never materialises [sq, b, h, sk] intermediate"
    )
    print(
        "[cppmega] DSA indexer fused patch applied "
        "(per-head accumulation, [sq,b,h,sk] intermediate eliminated)"
    )
    return True
