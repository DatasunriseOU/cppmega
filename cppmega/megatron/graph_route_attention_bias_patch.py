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
)

log = logging.getLogger(__name__)

__all__ = [
    "apply_graph_route_attention_bias_patch",
    "attention_layer_route_kind",
    "build_dense_graph_attention_bias_from_structure_batch",
    "build_rectangular_graph_attention_bias_from_structure_batch",
    "graph_dense_bias_enabled",
    "PromptGraphInferenceState",
    "set_prompt_graph_inference_state",
]

_PATCH_MARKER = "__cppmega_graph_route_attention_bias_patched__"
_INFERENCE_STATE_ATTRIBUTE = "_cppmega_prompt_graph_inference_state"


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
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


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


def graph_dense_bias_enabled() -> bool:
    """Default dense graph bias on whenever graph routes are enabled."""

    if not _env_flag("CPPMEGA_GRAPH_ROUTES_ENABLED"):
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
    beta: float = 1.0,
) -> torch.Tensor:
    """Build TE-compatible dense attention bias ``[B,1,Sq,Sk]``.

    ``build_graph_route_bias_from_structure_batch`` returns ``S_graph`` as
    ``[B,Sq,Sk]`` for DSA indexer scoring.  Transformer Engine expects an
    attention bias broadcastable to ``[B,H,Sq,Sk]``, so we insert the singleton
    head dimension here.
    """

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
    if beta != 1.0:
        graph = graph * float(beta)
    bias = graph.unsqueeze(1).contiguous()
    receipt_path = os.environ.get("CPPMEGA_H200_GRAPH_PRIOR_RECEIPT")
    if receipt_path:
        from cppmega.megatron.h200_preflight import observe_graph_prior

        observe_graph_prior(
            prior=bias,
            consumer="dense_attention",
            receipt_path=receipt_path,
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
    beta: float = 1.0,
) -> torch.Tensor:
    """Build ``[B,1,new-query,cached-key]`` graph bias in global token space."""

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
    if beta != 1.0:
        bias.mul_(float(beta))
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
) -> torch.Tensor | None:
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
    if getattr(getattr(layer, "config", None), "sequence_parallel", False):
        raise RuntimeError(
            "dense graph-route attention bias does not support sequence_parallel yet; "
            "disable sequence_parallel or use DSA graph top-k bias"
        )
    if int(getattr(getattr(layer, "config", None), "context_parallel_size", 1)) != 1:
        raise RuntimeError(
            "dense graph-route attention bias does not support context_parallel_size > 1 yet"
        )
    if inference_context is not None:
        state = _prompt_graph_inference_state(inference_context)
        offset = getattr(inference_context, "sequence_len_offset", None)
        if offset is not None and int(offset) != state.query_start:
            raise RuntimeError(
                "prompt graph inference state is stale: "
                f"context.sequence_len_offset={int(offset)} "
                f"query_start={state.query_start}"
            )
        if state.query_start + int(tensor.shape[0]) != state.key_length:
            raise RuntimeError(
                "prompt graph inference state does not match query/KV geometry: "
                f"query_start={state.query_start} Sq={int(tensor.shape[0])} "
                f"Sk={state.key_length}"
            )
        return build_rectangular_graph_attention_bias_from_structure_batch(
            state.structure_batch,
            batch_size=int(tensor.shape[1]),
            query_start=state.query_start,
            seqlen_q=int(tensor.shape[0]),
            seqlen_k=state.key_length,
            device=tensor.device,
            dtype=tensor.dtype if tensor.is_floating_point() else torch.float32,
            call_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_CALL_WEIGHT", 1.0),
            type_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_TYPE_WEIGHT", 1.0),
            domain_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_DOMAIN_WEIGHT", 1.0),
            build_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_BUILD_WEIGHT", 1.0),
            shell_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_SHELL_WEIGHT", 1.0),
            diagnostic_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_DIAGNOSTIC_WEIGHT", 1.0),
            cross_domain_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_CROSS_DOMAIN_WEIGHT", 1.0),
            beta=_env_float("CPPMEGA_GRAPH_ATTENTION_BIAS_BETA", 1.0),
        )

    from cppmega.megatron.structure_dataset_patch import _get_current_structure_batch

    return build_dense_graph_attention_bias_from_structure_batch(
        _get_current_structure_batch(),
        batch_size=int(tensor.shape[1]),
        seqlen_q=int(tensor.shape[0]),
        seqlen_k=int(tensor.shape[0]),
        device=tensor.device,
        dtype=tensor.dtype if tensor.is_floating_point() else torch.float32,
        call_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_CALL_WEIGHT", 1.0),
        type_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_TYPE_WEIGHT", 1.0),
        domain_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_DOMAIN_WEIGHT", 1.0),
        build_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_BUILD_WEIGHT", 1.0),
        shell_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_SHELL_WEIGHT", 1.0),
        diagnostic_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_DIAGNOSTIC_WEIGHT", 1.0),
        cross_domain_weight=_env_float("CPPMEGA_GRAPH_ATTENTION_CROSS_DOMAIN_WEIGHT", 1.0),
        beta=_env_float("CPPMEGA_GRAPH_ATTENTION_BIAS_BETA", 1.0),
    )


def apply_graph_route_attention_bias_patch(*, force: bool = False) -> bool:
    """Patch Megatron ``TransformerLayer.forward`` to inject graph bias.

    The patch is intentionally narrow: dense TE/GQA attention gets
    ``attention_bias``; DSA is handled by the DSA indexer patch; MLA raises so
    we do not silently run a graph-routed cppmega model as token-only.
    """

    from megatron.core.transformer.transformer_layer import TransformerLayer

    existing = getattr(TransformerLayer, "forward", None)
    if existing is None:
        raise RuntimeError("Megatron TransformerLayer.forward not found")
    if getattr(existing, _PATCH_MARKER, False) and not force:
        log.info("cppmega graph-route attention bias patch already applied")
        return True

    _forward_param_names = list(inspect.signature(existing).parameters)

    def _forward_with_graph_route_bias(self, *args, **kwargs):
        if kwargs.get("attention_bias") is None and graph_dense_bias_enabled():
            if "hidden_states" in kwargs:
                hidden_states = kwargs["hidden_states"]
            elif args:
                hidden_states = args[0]
            else:
                hidden_states = None
            if hidden_states is None:
                raise RuntimeError(
                    "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS=1 but TransformerLayer.forward "
                    "received no hidden_states"
                )
            bias = _graph_attention_bias_for_layer(
                self, hidden_states, _forward_inference_context(_forward_param_names, args, kwargs)
            )
            if bias is not None:
                kwargs["attention_bias"] = bias
        return existing(self, *args, **kwargs)

    setattr(_forward_with_graph_route_bias, _PATCH_MARKER, True)
    TransformerLayer.forward = _forward_with_graph_route_bias

    log.info("cppmega graph-route attention bias patch applied")
    print(
        "[cppmega] graph-route attention bias patch applied "
        "(dense TE/GQA attention_bias from compiler routes)",
        flush=True,
    )
    return True
