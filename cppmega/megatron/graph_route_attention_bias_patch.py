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
from typing import Any

import torch

from cppmega.megatron.dsa_indexer_fused_patch import (
    build_graph_route_bias_from_structure_batch,
)

log = logging.getLogger(__name__)

__all__ = [
    "apply_graph_route_attention_bias_patch",
    "attention_layer_route_kind",
    "build_dense_graph_attention_bias_from_structure_batch",
    "graph_dense_bias_enabled",
]

_PATCH_MARKER = "__cppmega_graph_route_attention_bias_patched__"


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
    if beta != 1.0:
        graph = graph * float(beta)
    return graph.unsqueeze(1).contiguous()


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
        raise RuntimeError(
            "dense graph-route attention bias only supports square self-attention "
            "(Sq==Sk); TransformerLayer.forward received an inference/decode context "
            f"({type(inference_context).__name__}) so the KV length cannot be derived "
            f"from the query hidden_states [S={int(tensor.shape[0])},B={int(tensor.shape[1])}]"
            " — not supported in incremental decode; use DSA graph top-k bias"
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
