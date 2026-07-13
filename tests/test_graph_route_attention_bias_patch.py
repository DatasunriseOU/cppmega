import json
import os

import pytest
import torch

from cppmega.megatron.graph_route_attention_bias_patch import (
    PromptGraphInferenceState,
    _forward_inference_context,
    _graph_attention_bias_for_layer,
    attention_layer_route_kind,
    build_dense_graph_attention_bias_from_structure_batch,
    set_prompt_graph_inference_state,
)
from cppmega.megatron.structure_dataset_patch import _set_current_structure_batch


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
        "graph_chunk_starts": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([4], dtype=torch.long),
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


def test_dense_attention_consumer_records_nonzero_graph_prior(tmp_path, monkeypatch):
    receipt_path = tmp_path / "dense-prior.json"
    monkeypatch.setenv("CPPMEGA_H200_GRAPH_PRIOR_RECEIPT", str(receipt_path))
    structure_batch = {
        "graph_domain_edges": torch.tensor([[[1, 3, 5]]], dtype=torch.long),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
    }

    bias = build_dense_graph_attention_bias_from_structure_batch(
        structure_batch,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device=torch.device("cpu"),
    )

    assert bias[0, 0, 1, 3].item() == 1.0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["consumer"] == "dense_attention"
    assert receipt["prior"]["nonzero"] == 1


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


def test_dense_graph_attention_bias_accepts_explicit_empty_prompt_graph():
    structure_batch = {
        "graph_call_edges": torch.zeros((1, 0, 2), dtype=torch.long),
        "graph_call_edge_counts": torch.zeros((1,), dtype=torch.long),
        "graph_chunk_starts": torch.zeros((1, 0), dtype=torch.long),
        "graph_chunk_ends": torch.zeros((1, 0), dtype=torch.long),
        "graph_chunk_counts": torch.zeros((1,), dtype=torch.long),
    }

    bias = build_dense_graph_attention_bias_from_structure_batch(
        structure_batch,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device=torch.device("cpu"),
    )

    assert tuple(bias.shape) == (1, 1, 4, 4)
    assert torch.count_nonzero(bias).item() == 0


def test_dense_graph_attention_bias_raises_above_max_seq(monkeypatch):
    monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_MAX_SEQ", "4")
    structure_batch = {
        "graph_call_edges": torch.tensor([[[0, 2], [-1, -1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([4], dtype=torch.long),
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
        "graph_chunk_starts": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([4], dtype=torch.long),
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
    names = ["self", "hidden_states", "attention_mask", "inference_context", "inference_params"]
    # keyword forms
    assert _forward_inference_context(names, (), {}) is None
    assert _forward_inference_context(names, (), {"inference_context": None}) is None
    assert _forward_inference_context(names, (), {"inference_context": ctx}) is ctx
    assert _forward_inference_context(names, (), {"inference_params": ctx}) is ctx
    # POSITIONAL form: inference_context is signature index 3 -> args index 2 (self dropped)
    assert _forward_inference_context(names, ("hs", "mask", ctx), {}) is ctx
    assert _forward_inference_context(names, ("hs", "mask"), {}) is None


def test_dense_graph_attention_bias_rectangular_decode_matches_full_logits(monkeypatch):
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", "1")
    layer = _Layer(_DenseSelfAttention())
    structure_batch = {
        "graph_call_edges": torch.tensor([[[0, 1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_type_edges": torch.zeros((1, 0, 2), dtype=torch.long),
        "graph_type_edge_counts": torch.tensor([0], dtype=torch.long),
        "graph_generated_query_edges": torch.tensor(
            [[[3, 0]]], dtype=torch.long
        ),
        "graph_generated_query_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 2, 3]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[2, 3, 4]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([3], dtype=torch.long),
    }
    full_hidden = torch.zeros(4, 1, 8)

    class _FakeInferenceContext:
        sequence_len_offset = 3

    context = _FakeInferenceContext()
    set_prompt_graph_inference_state(
        context,
        PromptGraphInferenceState(
            structure_batch=structure_batch,
            query_start=3,
            key_length=4,
        ),
    )

    _set_current_structure_batch(structure_batch)
    try:
        full_bias = _graph_attention_bias_for_layer(layer, full_hidden)
        rectangular_bias = _graph_attention_bias_for_layer(
            layer, full_hidden[-1:], context
        )
    finally:
        _set_current_structure_batch(None)

    assert tuple(rectangular_bias.shape) == (1, 1, 1, 4)
    torch.testing.assert_close(rectangular_bias, full_bias[:, :, -1:, :])
    assert torch.count_nonzero(rectangular_bias).item() > 0

    query_key_logits = torch.tensor([[[[0.3, -0.1, 0.2, 0.0]]]])
    values = torch.tensor([[[[1.0], [2.0], [4.0], [8.0]]]])
    full_probs = torch.softmax(query_key_logits + full_bias[:, :, -1:, :], dim=-1)
    cached_probs = torch.softmax(query_key_logits + rectangular_bias, dim=-1)
    torch.testing.assert_close(
        torch.matmul(full_probs, values),
        torch.matmul(cached_probs, values),
    )


def test_incremental_decode_requires_explicit_prompt_graph_state(monkeypatch):
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", "1")

    with pytest.raises(RuntimeError, match="prompt graph inference state"):
        _graph_attention_bias_for_layer(
            _Layer(_DenseSelfAttention()),
            torch.zeros(1, 1, 8),
            object(),
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
