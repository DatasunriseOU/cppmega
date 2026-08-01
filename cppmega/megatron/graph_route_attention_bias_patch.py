"""Graph-route attention bias wiring for dense Megatron/TE attention.

The sidecar dataset already carries compiler-derived token routes
(``callsite -> callee``, ``type-use -> declaration``).  DSA consumes those
routes before sparse top-k selection via ``dsa_indexer_fused_patch``.  Dense
GQA/TE attention needs a separate path: Megatron's ``TransformerLayer`` accepts
``attention_bias`` and ``TEDotProductAttention`` forwards it to Transformer
Engine as a post-scale bias.  This module installs that path fail-closed.
"""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

from cppmega.megatron.dsa_indexer_fused_patch import (
    _as_batched_chunks,
    _as_batched_edges,
    _as_batched_edge_triples,
    _scatter_edges_,
    build_graph_route_bias_from_structure_batch,
    require_graph_routes_for_production,
)
from cppmega.megatron.graph_objective_loss import (
    graph_routes_active,
    resolve_graph_bias_beta,
    validate_graph_bias_beta,
)
from cppmega.megatron.fa4_score_mod_adapter import (
    ChunkNativeGraphBias,
    CppMegaFA4ScoreModAttention,
    build_fa4_attention_bias_from_structure_batch,
    fa4_score_mod_enabled,
)

log = logging.getLogger(__name__)

__all__ = [
    "apply_graph_route_attention_bias_patch",
    "attention_layer_route_kind",
    "build_dense_graph_attention_bias_from_structure_batch",
    "build_rectangular_graph_attention_bias_from_structure_batch",
    "graph_dense_bias_enabled",
    "invalidate_bias_cache",
    "PromptGraphInferenceState",
    "set_prompt_graph_inference_state",
]

_PATCH_MARKER = "__cppmega_graph_route_attention_bias_patched__"
_ORIGINAL_FORWARD_ATTRIBUTE = "__cppmega_graph_route_attention_bias_original__"
_INFERENCE_STATE_ATTRIBUTE = "_cppmega_prompt_graph_inference_state"

# ---------------------------------------------------------------------------
# Content-based bias cache.
#
# The same structure_batch flows through every TransformerLayer for one
# microbatch.  Rebuilding scatter/sort/CSR bias at each layer is wasteful.
# We cache the LAST result and validate by (data_ptr, shape) of the first
# available edge tensor plus geometry.  data_ptr changes when new storage is
# allocated for a different batch, unlike id() which CPython may reuse.
# ---------------------------------------------------------------------------
_BIAS_CACHE_KEYS = (
    "graph_call_edges",
    "graph_type_edges",
    "graph_domain_edges",
    "graph_build_edges",
    "graph_shell_edges",
    "graph_diagnostic_edges",
    "graph_cross_domain_edges",
    "graph_generated_query_edges",
)
_bias_cache: dict[str, Any] = {"key": None, "result": None}


def _structure_batch_cache_key(
    structure_batch: dict[str, torch.Tensor] | None,
) -> tuple | None:
    """Derive a content-based identity tuple from a structure_batch.

    Returns None when the batch is empty or has no recognised edge tensors
    (caller should skip caching in that case).
    """
    if not structure_batch:
        return None
    for key in _BIAS_CACHE_KEYS:
        tensor = structure_batch.get(key)
        if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
            return (tensor.data_ptr(), tuple(tensor.shape))
    return None


def invalidate_bias_cache() -> None:
    """Explicitly flush the dense/FA4 bias cache.

    Call this when new data is loaded into the structure batch tensors (e.g.
    from the dataloader hook) to guarantee the next layer forward rebuilds the
    bias from fresh data.  In practice the content-based key (data_ptr + shape)
    already detects new allocations, but this provides a deterministic flush
    point for callers that reuse tensor storage in-place.
    """
    _bias_cache["key"] = None
    _bias_cache["result"] = None


# FA4 bias is built fresh each call when caching is not applicable.
_PINNED_TRANSFORMER_PARAMETERS = (
    "self",
    "hidden_states",
    "attention_mask",
    "context",
    "context_mask",
    "rotary_pos_emb",
    "rotary_pos_cos",
    "rotary_pos_sin",
    "rotary_pos_cos_sin",
    "attention_bias",
    "inference_context",
    "packed_seq_params",
    "sequence_len_offset",
    "padding_mask",
    "inference_params",
)


@dataclass(frozen=True)
class PromptGraphInferenceState:
    structure_batch: dict[str, torch.Tensor]
    query_start: int
    key_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.structure_batch, dict) or not self.structure_batch:
            raise ValueError("prompt graph inference state requires a structure batch")
        if self.query_start < 0:
            raise ValueError("prompt graph query_start must be non-negative")
        if self.key_length <= self.query_start:
            raise ValueError(
                "prompt graph key_length must include at least one query token"
            )


def set_prompt_graph_inference_state(
    inference_context: Any,
    state: PromptGraphInferenceState,
) -> None:
    if inference_context is None:
        raise ValueError("cannot attach prompt graph state to a null inference context")
    setattr(inference_context, _INFERENCE_STATE_ATTRIBUTE, state)


def _prompt_graph_inference_state(inference_context: Any) -> PromptGraphInferenceState:
    state = getattr(inference_context, _INFERENCE_STATE_ATTRIBUTE, None)
    if not isinstance(state, PromptGraphInferenceState):
        raise RuntimeError(
            "incremental dense graph decode requires explicit prompt graph inference state"
        )
    return state


def _env_flag(name: str, default: str = "0") -> bool:
    raw = os.environ[name] if name in os.environ else default
    if not isinstance(raw, str):
        raise TypeError(f"{name} must be a string boolean value, got {type(raw).__name__}")
    normalized = raw.strip().lower()
    if not normalized:
        raise ValueError(
            f"{name} must be one of 1,true,yes,on,0,false,no,off; empty values are invalid"
        )
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1,true,yes,on,0,false,no,off; got {raw!r}"
    )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got {raw!r}") from exc


def _require_pinned_transformer_signature(
    value: object,
    *,
    qualified_name: str,
) -> inspect.Signature:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot inspect pinned Megatron seam {qualified_name}"
        ) from exc
    actual = tuple(signature.parameters)
    if actual != _PINNED_TRANSFORMER_PARAMETERS:
        raise RuntimeError(
            f"pinned Megatron seam {qualified_name} has parameters {actual}, "
            "expected the core_v0.18.0 TransformerLayer attention signature "
            f"{_PINNED_TRANSFORMER_PARAMETERS}"
        )
    return signature


def graph_dense_bias_enabled() -> bool:
    """Default dense graph bias on whenever graph routes are enabled."""

    require_graph_routes_for_production()
    if not graph_routes_active():
        return False
    return _env_flag("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", "1")


def build_dense_graph_attention_bias_from_structure_batch(
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
    beta: float | None = None,
) -> torch.Tensor:
    """Build TE-compatible dense attention bias ``[B,1,Sq,Sk]``.

    ``build_graph_route_bias_from_structure_batch`` returns ``S_graph`` as
    ``[B,Sq,Sk]`` for DSA indexer scoring.  Transformer Engine expects an
    attention bias broadcastable to ``[B,H,Sq,Sk]``, so we insert the singleton
    head dimension here.
    """

    effective_beta = (
        resolve_graph_bias_beta()
        if beta is None
        else validate_graph_bias_beta(beta)
    )
    # ponytail: the real fix for long context is a block-sparse bias carrying
    # only the handful of edges; this cap is the fail-loud guard against a
    # silent multi-GiB dense [B,1,Sq,Sk] blowup (bf16 is 4 GiB at B=8,S=16384).
    # Default cap = the repo's documented max validated seq (16384). The DSA model
    # never hits this dense path (route_kind=="dsa" returns None), so this only
    # guards a dense TE/GQA variant against an unbounded blowup beyond 16k.
    max_seq = _env_int("CPPMEGA_GRAPH_DENSE_MAX_SEQ", 16384)
    if seqlen_q > max_seq or seqlen_k > max_seq:
        elem = torch.empty((), dtype=dtype).element_size()
        gib = batch_size * seqlen_q * seqlen_k * elem / (1024 ** 3)
        raise RuntimeError(
            "dense graph-route attention bias would materialize a "
            f"[B={batch_size},1,Sq={seqlen_q},Sk={seqlen_k}] {dtype} tensor "
            f"(~{gib:.2f} GiB); Sq/Sk exceed CPPMEGA_GRAPH_DENSE_MAX_SEQ={max_seq}. "
            "Dense O(B*Sq*Sk) bias does not scale to long context — use DSA graph "
            "top-k / block-sparse bias for longer sequences, or raise the cap if "
            "you have the memory."
        )

    graph = build_graph_route_bias_from_structure_batch(
        structure_batch,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        device=device,
        dtype=dtype,
        call_weight=call_weight,
        type_weight=type_weight,
        domain_weight=domain_weight,
        build_weight=build_weight,
        shell_weight=shell_weight,
        diagnostic_weight=diagnostic_weight,
        cross_domain_weight=cross_domain_weight,
    )
    generated = _as_batched_edges(
        structure_batch,
        edge_key="graph_generated_query_edges",
        count_key="graph_generated_query_edge_counts",
        batch_size=batch_size,
        device=device,
    )
    if generated is not None:
        generated_edges, generated_counts = generated
        _scatter_edges_(
            graph,
            generated_edges,
            generated_counts,
            weight=1.0,
            sq=seqlen_q,
            sk=seqlen_k,
            require_kind=False,
        )
    if effective_beta != 1.0:
        graph = graph * effective_beta
    bias = graph.unsqueeze(1).contiguous()
    receipt_path = os.environ.get("CPPMEGA_H200_GRAPH_PRIOR_RECEIPT")
    if receipt_path:
        from cppmega.megatron.h200_preflight import observe_graph_prior

        observe_graph_prior(
            prior=bias,
            consumer="dense_attention",
            receipt_path=receipt_path,
            bias_beta=effective_beta,
        )
    return bias


def build_rectangular_graph_attention_bias_from_structure_batch(
    structure_batch: dict[str, torch.Tensor] | None,
    *,
    batch_size: int,
    query_start: int,
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
    beta: float | None = None,
) -> torch.Tensor:
    """Build ``[B,1,new-query,cached-key]`` graph bias in global token space."""

    effective_beta = (
        resolve_graph_bias_beta()
        if beta is None
        else validate_graph_bias_beta(beta)
    )
    if structure_batch is None:
        raise RuntimeError(
            "incremental graph decode has no prompt graph structure batch"
        )
    if batch_size <= 0 or seqlen_q <= 0 or seqlen_k <= 0:
        raise ValueError("rectangular graph bias dimensions must be positive")
    if query_start < 0 or query_start + seqlen_q > seqlen_k:
        raise ValueError(
            "rectangular graph query range must be contained in cached keys: "
            f"query=[{query_start},{query_start + seqlen_q}) keys={seqlen_k}"
        )
    max_seq = _env_int("CPPMEGA_GRAPH_DENSE_MAX_SEQ", 16384)
    if seqlen_q > max_seq or seqlen_k > max_seq:
        raise RuntimeError(
            "rectangular graph-route attention bias exceeds "
            f"CPPMEGA_GRAPH_DENSE_MAX_SEQ={max_seq}: Sq={seqlen_q} Sk={seqlen_k}"
        )

    bias = torch.zeros(
        (batch_size, seqlen_q, seqlen_k),
        device=device,
        dtype=dtype,
    )
    seen_relation = False
    chunk_layout: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    query_end = query_start + seqlen_q

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
                structure_batch,
                batch_size=batch_size,
                device=device,
            )
        starts, ends, chunk_counts = chunk_layout
        if int(edges.shape[0]) == 1 and batch_size > 1:
            edges = edges.expand(batch_size, -1, -1)
        if int(counts.shape[0]) == 1 and batch_size > 1:
            counts = counts.expand(batch_size)
        for batch_index in range(batch_size):
            for edge_index in range(int(counts[batch_index].item())):
                source_chunk = int(edges[batch_index, edge_index, 0].item())
                target_chunk = int(edges[batch_index, edge_index, 1].item())
                available = int(chunk_counts[batch_index].item())
                if not (
                    0 <= source_chunk < available
                    and 0 <= target_chunk < available
                ):
                    raise ValueError(
                        "declared call/type edge references an unavailable chunk"
                    )
                source_start = max(
                    query_start,
                    int(starts[batch_index, source_chunk].item()),
                )
                source_end = min(
                    query_end,
                    int(ends[batch_index, source_chunk].item()),
                )
                target_start = max(
                    0,
                    int(starts[batch_index, target_chunk].item()),
                )
                target_end = min(
                    seqlen_k,
                    int(ends[batch_index, target_chunk].item()),
                )
                if source_start < source_end and target_start < target_end:
                    bias[
                        batch_index,
                        source_start - query_start : source_end - query_start,
                        target_start:target_end,
                    ] += float(weight)

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
        _scatter_rectangular_token_edges_(
            bias,
            edges,
            counts,
            query_start=query_start,
            weight=weight,
        )

    generated = _as_batched_edges(
        structure_batch,
        edge_key="graph_generated_query_edges",
        count_key="graph_generated_query_edge_counts",
        batch_size=batch_size,
        device=device,
    )
    if generated is not None:
        seen_relation = True
        edges, counts = generated
        _scatter_rectangular_token_edges_(
            bias,
            edges,
            counts,
            query_start=query_start,
            weight=1.0,
        )
    if not seen_relation:
        raise KeyError("prompt graph inference state contains no route tensors")
    if effective_beta != 1.0:
        bias.mul_(effective_beta)
    return bias.unsqueeze(1).contiguous()


def _scatter_rectangular_token_edges_(
    bias: torch.Tensor,
    edges: torch.Tensor,
    counts: torch.Tensor,
    *,
    query_start: int,
    weight: float,
) -> None:
    batch_size, seqlen_q, seqlen_k = bias.shape
    if int(edges.shape[0]) == 1 and batch_size > 1:
        edges = edges.expand(batch_size, -1, -1)
    if int(counts.shape[0]) == 1 and batch_size > 1:
        counts = counts.expand(batch_size)
    max_edges = int(edges.shape[1])
    if bool(((counts < 0) | (counts > max_edges)).any().item()):
        raise ValueError(f"graph edge counts out of range [0,{max_edges}]")
    for batch_index in range(batch_size):
        for edge_index in range(int(counts[batch_index].item())):
            source = int(edges[batch_index, edge_index, 0].item())
            target = int(edges[batch_index, edge_index, 1].item())
            local_query = source - query_start
            if not (0 <= source < query_start + seqlen_q and 0 <= target < seqlen_k):
                raise ValueError(
                    f"graph token edge ({source},{target}) is outside decode bounds"
                )
            if local_query >= 0:
                bias[batch_index, local_query, target] += float(weight)


def attention_layer_route_kind(layer: Any) -> str:
    """Classify the TransformerLayer attention implementation for graph bias."""

    self_attention = getattr(layer, "self_attention", None)
    if self_attention is None:
        return "none"
    name = type(self_attention).__name__
    module = type(self_attention).__module__
    ident = f"{module}.{name}".lower()
    if "identity" in ident:
        return "none"
    if "mla" in ident or "multi_latent_attention" in ident:
        return "mla"
    if "dsa" in ident or "dsattention" in ident:
        return "dsa"
    if hasattr(self_attention, "core_attention"):
        return "dense"
    return "none"


def _hidden_shape_tensor(hidden_states: Any) -> torch.Tensor:
    tensor = hidden_states
    if not isinstance(tensor, torch.Tensor) and hasattr(tensor, "unwrap"):
        tensor = tensor.unwrap()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "cannot build graph attention bias: hidden_states must be a Tensor "
            f"or WrappedTensor, got {type(hidden_states).__name__}"
        )
    if tensor.dim() < 3:
        raise ValueError(
            "cannot build graph attention bias: hidden_states must be [S,B,H], "
            f"got shape {tuple(tensor.shape)}"
        )
    return tensor


def _forward_inference_context(param_names: list[str], args: tuple, kwargs: dict) -> Any:
    """Return Megatron's inference/decode context from a TransformerLayer.forward
    call (passed by keyword OR positionally), or ``None`` during training.

    Megatron passes ``inference_context`` (or the deprecated ``inference_params``
    alias) during incremental decode / cached generation. When present, the seam
    only sees the query tokens as hidden_states, so the true KV length is not Sq and
    a square dense bias would be wrong. ``param_names`` is
    ``signature(TransformerLayer.forward).parameters`` including ``self`` at index 0;
    the wrapper receives ``self`` separately, so a parameter at signature index ``i``
    sits at ``args[i-1]`` when passed positionally.
    """

    for key in ("inference_context", "inference_params"):
        value = kwargs.get(key)
        if value is not None:
            return value
        if key in param_names:
            ai = param_names.index(key) - 1  # drop leading 'self'
            if 0 <= ai < len(args) and args[ai] is not None:
                return args[ai]
    return None


def _graph_attention_bias_for_layer(
    layer: Any, hidden_states: Any, inference_context: Any = None
) -> "torch.Tensor | ChunkNativeGraphBias | None":
    if not graph_dense_bias_enabled():
        return None

    kind = attention_layer_route_kind(layer)
    if kind == "none":
        return None
    if kind == "dsa":
        # DSA consumes graph routes before top-k selection in
        # dsa_indexer_fused_patch.  Passing dense attention_bias as well would
        # double-apply the same relation prior.
        return None
    if kind == "mla":
        raise RuntimeError(
            "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS=1 but this layer uses MLA; "
            "dense graph-route bias is only wired for TE/GQA attention. "
            "Use DSA graph top-k bias or add an MLA-specific route bias seam."
        )

    tensor = _hidden_shape_tensor(hidden_states)
    config = getattr(layer, "config", None)
    if getattr(config, "sequence_parallel", False):
        raise RuntimeError(
            "dense graph-route attention bias does not support sequence_parallel yet; "
            "disable sequence_parallel or use DSA graph top-k bias"
        )
    configured_cp_size = int(getattr(config, "context_parallel_size", 1))
    self_attention = getattr(layer, "self_attention", None)
    pg_collection = getattr(self_attention, "pg_collection", None)
    cp_group = getattr(pg_collection, "cp", None)
    if cp_group is None:
        actual_cp_size = 1
    else:
        size = getattr(cp_group, "size", None)
        if not callable(size):
            raise TypeError(
                "graph-route attention pg_collection.cp must expose size()"
            )
        actual_cp_size = int(size())
    if actual_cp_size != configured_cp_size:
        raise RuntimeError(
            "graph-route attention configured context_parallel_size="
            f"{configured_cp_size}, but self_attention.pg_collection.cp has "
            f"size {actual_cp_size}"
        )

    use_fa4 = fa4_score_mod_enabled()
    if actual_cp_size > 1:
        core_attention = getattr(self_attention, "core_attention", None)
        if not use_fa4 or not isinstance(
            core_attention,
            CppMegaFA4ScoreModAttention,
        ):
            raise RuntimeError(
                "context-parallel graph-route attention supports only FA4 "
                "chunk-native bias; dense TE/torch attention remains fail-closed"
            )
        if inference_context is not None:
            raise NotImplementedError(
                "context-parallel FA4 graph-route attention does not support "
                "incremental decode geometry"
            )
    beta = resolve_graph_bias_beta()

    # --- Resolve weights once (used in cache key and all build paths) ---
    w_call = _env_float("CPPMEGA_GRAPH_ATTENTION_CALL_WEIGHT", 1.0)
    w_type = _env_float("CPPMEGA_GRAPH_ATTENTION_TYPE_WEIGHT", 1.0)
    w_domain = _env_float("CPPMEGA_GRAPH_ATTENTION_DOMAIN_WEIGHT", 1.0)
    w_build = _env_float("CPPMEGA_GRAPH_ATTENTION_BUILD_WEIGHT", 1.0)
    w_shell = _env_float("CPPMEGA_GRAPH_ATTENTION_SHELL_WEIGHT", 1.0)
    w_diag = _env_float("CPPMEGA_GRAPH_ATTENTION_DIAGNOSTIC_WEIGHT", 1.0)
    w_cross = _env_float("CPPMEGA_GRAPH_ATTENTION_CROSS_DOMAIN_WEIGHT", 1.0)

    # --- Resolve structure_batch early for cache key ---
    from cppmega.megatron.structure_dataset_patch import _get_current_structure_batch

    if inference_context is not None:
        state = _prompt_graph_inference_state(inference_context)
        offset = getattr(inference_context, "sequence_len_offset", None)
        if offset is not None and int(offset) != state.query_start:
            raise RuntimeError(
                "prompt graph inference state is stale: "
                f"context.sequence_len_offset={int(offset)} "
                f"query_start={state.query_start}"
            )
        sb_for_key = state.structure_batch
    else:
        state = None
        sb_for_key = _get_current_structure_batch()

    # --- Content-based cache lookup ---
    sb_identity = _structure_batch_cache_key(sb_for_key)
    batch_sz = int(tensor.shape[1])
    local_seqlen_q = int(tensor.shape[0])
    seqlen_q = local_seqlen_q * actual_cp_size
    if actual_cp_size > 1:
        from cppmega.megatron.document_isolation import _raw_document_ids

        document_ids = _raw_document_ids(required=True)
        if document_ids is None:
            raise RuntimeError(
                "context-parallel FA4 graph-route attention requires document_ids"
            )
        if document_ids.dim() == 1:
            document_ids = document_ids.unsqueeze(0)
        expected_shape = (batch_sz, seqlen_q)
        if tuple(document_ids.shape) != expected_shape:
            raise ValueError(
                "context-parallel FA4 global sequence geometry "
                f"{expected_shape} does not match document_ids shape "
                f"{tuple(document_ids.shape)}"
            )

    # Geometry differs between decode (rectangular) and prefill/train (square).
    if state is not None:
        geom = (batch_sz, seqlen_q, state.key_length, state.query_start)
    else:
        geom = (batch_sz, seqlen_q, seqlen_q, 0)

    cache_key: tuple | None = None
    if sb_identity is not None:
        cache_key = (
            sb_identity,
            use_fa4,
            geom,
            beta,
            w_call, w_type, w_domain, w_build, w_shell, w_diag, w_cross,
        )
        if _bias_cache["key"] == cache_key:
            return _bias_cache["result"]

    # --- Cache miss: compute bias ---
    result = _build_bias_uncached(
        tensor=tensor,
        state=state,
        inference_context=inference_context,
        sb_for_key=sb_for_key,
        use_fa4=use_fa4,
        beta=beta,
        batch_sz=batch_sz,
        seqlen_q=seqlen_q,
        w_call=w_call,
        w_type=w_type,
        w_domain=w_domain,
        w_build=w_build,
        w_shell=w_shell,
        w_diag=w_diag,
        w_cross=w_cross,
    )

    # --- Store in cache ---
    if cache_key is not None:
        _bias_cache["key"] = cache_key
        _bias_cache["result"] = result

    return result


def _build_bias_uncached(
    *,
    tensor: torch.Tensor,
    state: "PromptGraphInferenceState | None",
    inference_context: Any,
    sb_for_key: "dict[str, torch.Tensor] | None",
    use_fa4: bool,
    beta: float,
    batch_sz: int,
    seqlen_q: int,
    w_call: float,
    w_type: float,
    w_domain: float,
    w_build: float,
    w_shell: float,
    w_diag: float,
    w_cross: float,
) -> "torch.Tensor | ChunkNativeGraphBias | None":
    """Compute the graph attention bias (cache-miss path)."""

    # --- FA4 chunk-native path: return ChunkNativeGraphBias instead of dense ---
    if use_fa4:
        from cppmega.megatron.fa4_score_mod_adapter import (
            build_chunk_native_graph_bias,
        )

        if state is not None:
            # Decode mode: rectangular geometry (Sq=new tokens, Sk=full cache).
            sq = seqlen_q
            sk = state.key_length
            if state.query_start + sq != sk:
                raise RuntimeError(
                    "prompt graph inference state does not match query/KV geometry: "
                    f"query_start={state.query_start} Sq={sq} Sk={sk}"
                )
            # Build full-length chunk bias then slice query map to the new
            # token window [query_start, query_start+sq).  Key map covers all
            # cached tokens [0, sk).  chunk_bias stays [B, C+1, C+1].
            full_bias = build_chunk_native_graph_bias(
                state.structure_batch,
                batch_size=batch_sz,
                seqlen_q=sk,
                seqlen_k=sk,
                device=tensor.device,
                dtype=tensor.dtype if tensor.is_floating_point() else torch.float32,
                beta=beta,
                call_weight=w_call,
                type_weight=w_type,
                domain_weight=w_domain,
                build_weight=w_build,
                shell_weight=w_shell,
                diagnostic_weight=w_diag,
                cross_domain_weight=w_cross,
            )
            # Slice the query token-to-chunk map to the decode window and
            # rebase rare_q positions to local indices [0, sq).
            from dataclasses import replace as _dc_replace

            max_rare = full_bias.rare_q.shape[1]
            new_rare_q = torch.full(
                (batch_sz, max_rare), -1, device=tensor.device, dtype=torch.int32
            )
            new_rare_k = torch.full(
                (batch_sz, max_rare), -1, device=tensor.device, dtype=torch.int32
            )
            new_rare_w = torch.zeros(
                (batch_sz, max_rare), device=tensor.device,
                dtype=full_bias.rare_w.dtype,
            )
            q_start = state.query_start
            q_end = state.query_start + sq
            for b_idx in range(batch_sz):
                # Filter rare edges whose query is in the decode window.
                mask = (full_bias.rare_q[b_idx] >= q_start) & (
                    full_bias.rare_q[b_idx] < q_end
                )
                n_keep = int(mask.sum().item())
                if n_keep > max_rare:
                    n_keep = max_rare
                if n_keep > 0:
                    sel_q = full_bias.rare_q[b_idx][mask][:n_keep]
                    sel_k = full_bias.rare_k[b_idx][mask][:n_keep]
                    sel_w = full_bias.rare_w[b_idx][mask][:n_keep]
                    new_rare_q[b_idx, :n_keep] = sel_q - q_start
                    new_rare_k[b_idx, :n_keep] = sel_k
                    new_rare_w[b_idx, :n_keep] = sel_w

            # Build local rare_row_offsets [B, sq+1] for the decode window.
            new_row_offsets = torch.zeros(
                (batch_sz, sq + 1), device=tensor.device, dtype=torch.int32
            )
            for b_idx in range(batch_sz):
                row_counts = torch.zeros(sq, device=tensor.device, dtype=torch.int32)
                valid_q = new_rare_q[b_idx][new_rare_q[b_idx] >= 0]
                if valid_q.numel() > 0:
                    row_counts.scatter_add_(
                        0, valid_q.long(),
                        torch.ones(valid_q.numel(), device=tensor.device, dtype=torch.int32),
                    )
                new_row_offsets[b_idx, 1:] = torch.cumsum(row_counts, dim=0)

            return _dc_replace(
                full_bias,
                token_to_chunk_q=full_bias.token_to_chunk_q[
                    :, q_start:q_end
                ].contiguous(),
                rare_row_offsets=new_row_offsets,
                rare_q=new_rare_q,
                rare_k=new_rare_k,
                rare_w=new_rare_w,
            )

        bias_state = build_fa4_attention_bias_from_structure_batch(
            sb_for_key,
            batch_size=batch_sz,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_q,
            device=tensor.device,
            dtype=tensor.dtype if tensor.is_floating_point() else torch.float32,
            beta=beta,
            call_weight=w_call,
            type_weight=w_type,
            domain_weight=w_domain,
            build_weight=w_build,
            shell_weight=w_shell,
            diagnostic_weight=w_diag,
            cross_domain_weight=w_cross,
        )
        return bias_state

    if state is not None:
        if state.query_start + seqlen_q != state.key_length:
            raise RuntimeError(
                "prompt graph inference state does not match query/KV geometry: "
                f"query_start={state.query_start} Sq={seqlen_q} "
                f"Sk={state.key_length}"
            )
        return build_rectangular_graph_attention_bias_from_structure_batch(
            state.structure_batch,
            batch_size=batch_sz,
            query_start=state.query_start,
            seqlen_q=seqlen_q,
            seqlen_k=state.key_length,
            device=tensor.device,
            dtype=tensor.dtype if tensor.is_floating_point() else torch.float32,
            call_weight=w_call,
            type_weight=w_type,
            domain_weight=w_domain,
            build_weight=w_build,
            shell_weight=w_shell,
            diagnostic_weight=w_diag,
            cross_domain_weight=w_cross,
            beta=beta,
        )

    return build_dense_graph_attention_bias_from_structure_batch(
        sb_for_key,
        batch_size=batch_sz,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_q,
        device=tensor.device,
        dtype=tensor.dtype if tensor.is_floating_point() else torch.float32,
        call_weight=w_call,
        type_weight=w_type,
        domain_weight=w_domain,
        build_weight=w_build,
        shell_weight=w_shell,
        diagnostic_weight=w_diag,
        cross_domain_weight=w_cross,
        beta=beta,
    )


def apply_graph_route_attention_bias_patch(*, force: bool = False) -> bool:
    """Patch Megatron ``TransformerLayer.forward`` to inject graph bias.

    The patch is intentionally narrow: dense TE/GQA attention gets
    ``attention_bias``; DSA is handled by the DSA indexer patch; MLA raises so
    we do not silently run a graph-routed cppmega model as token-only.
    """

    require_graph_routes_for_production()

    from megatron.core.transformer.transformer_layer import TransformerLayer

    installed_forward = getattr(TransformerLayer, "forward", None)
    if installed_forward is None:
        raise RuntimeError("Megatron TransformerLayer.forward not found")
    if getattr(installed_forward, _PATCH_MARKER, False) and not force:
        log.info("cppmega graph-route attention bias patch already applied")
        return True

    existing = installed_forward
    if getattr(existing, _PATCH_MARKER, False):
        existing = getattr(existing, _ORIGINAL_FORWARD_ATTRIBUTE, None)
        if existing is None:
            raise RuntimeError(
                "graph-route attention bias patch marker has no original pinned forward"
            )
    forward_signature = _require_pinned_transformer_signature(
        existing,
        qualified_name="TransformerLayer.forward",
    )
    _require_pinned_transformer_signature(
        TransformerLayer._forward_attention,
        qualified_name="TransformerLayer._forward_attention",
    )

    def _forward_with_graph_route_bias(self, *args, **kwargs):
        try:
            bound = forward_signature.bind(self, *args, **kwargs)
        except TypeError:
            # Preserve Megatron's native argument error and traceback for calls
            # that do not match the pinned signature.
            return existing(self, *args, **kwargs)
        if bound.arguments.get("attention_bias") is None and graph_dense_bias_enabled():
            hidden_states = bound.arguments.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError(
                    "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS=1 but TransformerLayer.forward "
                    "received no hidden_states"
                )
            inference_context = bound.arguments.get("inference_context")
            if inference_context is None:
                inference_context = bound.arguments.get("inference_params")
            bias = _graph_attention_bias_for_layer(
                self,
                hidden_states,
                inference_context,
            )
            if bias is not None:
                bound.arguments["attention_bias"] = bias
        return existing(*bound.args, **bound.kwargs)

    setattr(_forward_with_graph_route_bias, _PATCH_MARKER, True)
    setattr(_forward_with_graph_route_bias, _ORIGINAL_FORWARD_ATTRIBUTE, existing)
    TransformerLayer.forward = _forward_with_graph_route_bias

    log.info("cppmega graph-route attention bias patch applied")
    print(
        "[cppmega] graph-route attention bias patch applied "
        "(dense TE/GQA attention_bias from compiler routes)",
        flush=True,
    )
    return True
