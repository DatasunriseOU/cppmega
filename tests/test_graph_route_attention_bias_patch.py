import os

import pytest
import torch

from cppmega.megatron.graph_route_attention_bias_patch import (
    _forward_inference_context,
    _graph_attention_bias_for_layer,
    attention_layer_route_kind,
    build_dense_graph_attention_bias_from_structure_batch,
)


class _DenseSelfAttention:
    core_attention = object()


class _DSASelfAttention:
    core_attention = object()


class _MLASelfAttention:
    core_attention = object()


class _IdentityOp:
    pass


class _Layer:
    def __init__(self, self_attention):
        self.self_attention = self_attention


def test_dense_graph_attention_bias_is_broadcastable_post_scale_bias():
    structure_batch = {
        "graph_call_edges": torch.tensor([[[0, 2], [1, 3], [-1, -1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([2], dtype=torch.long),
        "graph_type_edges": torch.tensor([[[2, 1], [-1, -1], [-1, -1]]], dtype=torch.long),
        "graph_type_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_build_edges": torch.tensor([[[3, 0, 20], [-1, -1, -1]]], dtype=torch.long),
        "graph_build_edge_counts": torch.tensor([1], dtype=torch.long),
    }

    bias = build_dense_graph_attention_bias_from_structure_batch(
        structure_batch,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        call_weight=2.0,
        type_weight=3.0,
        build_weight=4.0,
        beta=0.5,
    )

    assert tuple(bias.shape) == (1, 1, 4, 4)
    assert bias[0, 0, 0, 2].item() == 1.0
    assert bias[0, 0, 1, 3].item() == 1.0
    assert bias[0, 0, 2, 1].item() == 1.5
    assert bias[0, 0, 3, 0].item() == 2.0
    assert bias.sum().item() == 5.5


def test_attention_layer_route_kind_only_biases_dense_attention():
    assert attention_layer_route_kind(_Layer(_DenseSelfAttention())) == "dense"
    assert attention_layer_route_kind(_Layer(_DSASelfAttention())) == "dsa"
    assert attention_layer_route_kind(_Layer(_MLASelfAttention())) == "mla"
    assert attention_layer_route_kind(_Layer(_IdentityOp())) == "none"


def test_dense_graph_attention_bias_requires_route_edges():
    with pytest.raises(RuntimeError, match="no current cppmega structure batch"):
        build_dense_graph_attention_bias_from_structure_batch(
            None,
            batch_size=1,
            seqlen_q=4,
            seqlen_k=4,
            device=torch.device("cpu"),
        )


def test_dense_graph_attention_bias_raises_above_max_seq(monkeypatch):
    monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_MAX_SEQ", "4")
    structure_batch = {
        "graph_call_edges": torch.tensor([[[0, 2], [-1, -1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
    }
    with pytest.raises(RuntimeError, match="CPPMEGA_GRAPH_DENSE_MAX_SEQ"):
        build_dense_graph_attention_bias_from_structure_batch(
            structure_batch,
            batch_size=1,
            seqlen_q=5,
            seqlen_k=5,
            device=torch.device("cpu"),
        )


def test_dense_graph_attention_bias_builds_at_max_seq(monkeypatch):
    monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_MAX_SEQ", "4")
    structure_batch = {
        "graph_call_edges": torch.tensor([[[0, 2], [-1, -1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
    }
    bias = build_dense_graph_attention_bias_from_structure_batch(
        structure_batch,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device=torch.device("cpu"),
        call_weight=2.0,
    )
    assert tuple(bias.shape) == (1, 1, 4, 4)
    assert bias[0, 0, 0, 2].item() == 2.0


def test_forward_inference_context_detects_context_and_params():
    ctx = object()
    assert _forward_inference_context({}) is None
    assert _forward_inference_context({"inference_context": None}) is None
    assert _forward_inference_context({"inference_context": ctx}) is ctx
    assert _forward_inference_context({"inference_params": ctx}) is ctx


def test_dense_graph_attention_bias_raises_in_incremental_decode(monkeypatch):
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", "1")
    layer = _Layer(_DenseSelfAttention())
    hidden_states = torch.zeros(4, 2, 8)

    class _FakeInferenceContext:
        pass

    with pytest.raises(RuntimeError, match="incremental decode"):
        _graph_attention_bias_for_layer(
            layer, hidden_states, _FakeInferenceContext()
        )


def test_dense_graph_attention_bias_env_defaults_are_enabled_with_graph_routes():
    old_graph = os.environ.get("CPPMEGA_GRAPH_ROUTES_ENABLED")
    old_dense = os.environ.get("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS")
    try:
        os.environ["CPPMEGA_GRAPH_ROUTES_ENABLED"] = "1"
        os.environ.pop("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", None)
        from cppmega.megatron.graph_route_attention_bias_patch import graph_dense_bias_enabled

        assert graph_dense_bias_enabled()
    finally:
        if old_graph is None:
            os.environ.pop("CPPMEGA_GRAPH_ROUTES_ENABLED", None)
        else:
            os.environ["CPPMEGA_GRAPH_ROUTES_ENABLED"] = old_graph
        if old_dense is None:
            os.environ.pop("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", None)
        else:
            os.environ["CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS"] = old_dense
