import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from cppmega.megatron.graph_route_attention_bias_patch import (
    PromptGraphInferenceState,
    _forward_inference_context,
    _graph_attention_bias_for_layer,
    attention_layer_route_kind,
    build_dense_graph_attention_bias_from_structure_batch,
    _env_flag,
    graph_dense_bias_enabled,
    set_prompt_graph_inference_state,
)
from cppmega.megatron.fa4_score_mod_adapter import (
    ChunkNativeGraphBias,
    CppMegaFA4ScoreModAttention,
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


def test_dense_graph_edge_changes_attention_logits_and_value_output():
    structure_batch = {
        "graph_domain_edges": torch.tensor([[[0, 3, 5]]], dtype=torch.long),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
    }
    graph_bias = build_dense_graph_attention_bias_from_structure_batch(
        structure_batch,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device=torch.device("cpu"),
        beta=4.0,
    )
    logits = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
    values = torch.tensor([[[[1.0], [2.0], [4.0], [8.0]]]])

    plain_output = torch.matmul(torch.softmax(logits, dim=-1), values)
    routed_logits = logits + graph_bias[:, :, :1, :]
    routed_output = torch.matmul(torch.softmax(routed_logits, dim=-1), values)

    assert routed_logits[0, 0, 0, 3].item() == pytest.approx(4.0)
    assert not torch.equal(routed_output, plain_output)
    assert routed_output.item() > plain_output.item()


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
    assert receipt["bias_beta"] == {
        "canonical_env": "CPPMEGA_GRAPH_BIAS_BETA",
        "legacy_envs": [
            "CPPMEGA_DSA_GRAPH_BIAS_BETA",
            "CPPMEGA_GRAPH_ATTENTION_BIAS_BETA",
        ],
        "value": "1",
    }
    assert receipt["prior"]["nonzero"] == 1


def test_dense_graph_attention_bias_rejects_dsa_dense_beta_drift():
    structure_batch = {
        "graph_domain_edges": torch.tensor([[[1, 3, 5]]], dtype=torch.long),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
    }
    _set_current_structure_batch(structure_batch)
    try:
        with patch.dict(
            os.environ,
            {
                "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
                "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS": "1",
                "CPPMEGA_DSA_GRAPH_BIAS_BETA": "2",
                "CPPMEGA_GRAPH_ATTENTION_BIAS_BETA": "3",
            },
            clear=False,
        ):
            with pytest.raises(ValueError, match="beta.*differ"):
                _graph_attention_bias_for_layer(
                    _Layer(_DenseSelfAttention()),
                    torch.zeros(4, 1, 8),
                )
    finally:
        _set_current_structure_batch(None)


def test_dense_graph_attention_bias_uses_canonical_beta():
    structure_batch = {
        "graph_domain_edges": torch.tensor([[[1, 3, 5]]], dtype=torch.long),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
    }
    _set_current_structure_batch(structure_batch)
    try:
        with patch.dict(
            os.environ,
            {
                "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
                "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS": "1",
                "CPPMEGA_GRAPH_BIAS_BETA": "2",
            },
            clear=True,
        ):
            bias = _graph_attention_bias_for_layer(
                _Layer(_DenseSelfAttention()),
                torch.zeros(4, 1, 8),
            )
            direct_bias = build_dense_graph_attention_bias_from_structure_batch(
                structure_batch,
                batch_size=1,
                seqlen_q=4,
                seqlen_k=4,
                device=torch.device("cpu"),
            )
    finally:
        _set_current_structure_batch(None)

    assert bias is not None
    assert bias[0, 0, 1, 3].item() == 2.0
    assert direct_bias[0, 0, 1, 3].item() == 2.0


def test_attention_layer_route_kind_only_biases_dense_attention():
    assert attention_layer_route_kind(_Layer(_DenseSelfAttention())) == "dense"
    assert attention_layer_route_kind(_Layer(_DSASelfAttention())) == "dsa"
    assert attention_layer_route_kind(_Layer(_MLASelfAttention())) == "mla"
    assert attention_layer_route_kind(_Layer(_IdentityOp())) == "none"


def test_fa4_context_parallel_bias_uses_global_document_geometry():
    class FakeGroup:
        def size(self):
            return 2

    config = SimpleNamespace(
        attention_dropout=0.0,
        context_parallel_size=2,
        sequence_parallel=False,
    )
    pg_collection = SimpleNamespace(cp=FakeGroup())
    core_attention = CppMegaFA4ScoreModAttention(
        config=config,
        layer_number=1,
        pg_collection=pg_collection,
    )
    self_attention = SimpleNamespace(
        core_attention=core_attention,
        pg_collection=pg_collection,
    )
    layer = SimpleNamespace(self_attention=self_attention, config=config)
    structure_batch = {
        "document_ids": torch.tensor([[1, 1, 2, 2, 2, 2, 3, 3]]),
        "graph_call_edges": torch.tensor([[[0, 1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 2, 6]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[2, 6, 8]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([3], dtype=torch.long),
    }
    environment_names = (
        "CPPMEGA_GRAPH_ROUTES_ENABLED",
        "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS",
        "CPPMEGA_FA4_SCORE_MOD",
        "CPPMEGA_GRAPH_BIAS_BETA",
    )
    previous = {name: os.environ.get(name) for name in environment_names}
    os.environ["CPPMEGA_GRAPH_ROUTES_ENABLED"] = "1"
    os.environ["CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS"] = "1"
    os.environ["CPPMEGA_FA4_SCORE_MOD"] = "1"
    os.environ["CPPMEGA_GRAPH_BIAS_BETA"] = "1"
    _set_current_structure_batch(structure_batch)
    try:
        bias = _graph_attention_bias_for_layer(
            layer,
            torch.zeros(4, 1, 8),
        )
        assert isinstance(bias, ChunkNativeGraphBias)
        assert tuple(bias.token_to_chunk_q.shape) == (1, 8)
        assert tuple(bias.token_to_chunk_k.shape) == (1, 8)

        with pytest.raises(
            NotImplementedError,
            match="sequence/context-parallel.*incremental decode",
        ):
            _graph_attention_bias_for_layer(
                layer,
                torch.zeros(4, 1, 8),
                inference_context=object(),
            )

        _set_current_structure_batch(
            {**structure_batch, "document_ids": torch.tensor([[1, 1, 2, 2]])}
        )
        with pytest.raises(ValueError, match="global sequence"):
            _graph_attention_bias_for_layer(
                layer,
                torch.zeros(4, 1, 8),
            )

        os.environ["CPPMEGA_FA4_SCORE_MOD"] = "0"
        dense_layer = SimpleNamespace(
            self_attention=SimpleNamespace(
                core_attention=object(),
                pg_collection=pg_collection,
            ),
            config=config,
        )
        with pytest.raises(RuntimeError, match="only FA4"):
            _graph_attention_bias_for_layer(
                dense_layer,
                torch.zeros(4, 1, 8),
            )
    finally:
        _set_current_structure_batch(None)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_fa4_sequence_parallel_bias_uses_global_document_geometry():
    class FakeGroup:
        def size(self):
            return 2

    config = SimpleNamespace(
        attention_dropout=0.0,
        context_parallel_size=1,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
    )
    pg_collection = SimpleNamespace(tp=FakeGroup(), cp=None)
    core_attention = CppMegaFA4ScoreModAttention(
        config=config,
        layer_number=1,
        pg_collection=pg_collection,
    )
    layer = SimpleNamespace(
        self_attention=SimpleNamespace(
            core_attention=core_attention,
            pg_collection=pg_collection,
        ),
        config=config,
    )
    structure_batch = {
        "document_ids": torch.tensor([[1, 1, 2, 2, 2, 2, 3, 3]]),
        "graph_call_edges": torch.tensor([[[0, 1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 2, 6]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[2, 6, 8]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([3], dtype=torch.long),
    }
    environment_names = (
        "CPPMEGA_GRAPH_ROUTES_ENABLED",
        "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS",
        "CPPMEGA_FA4_SCORE_MOD",
        "CPPMEGA_GRAPH_BIAS_BETA",
    )
    previous = {name: os.environ.get(name) for name in environment_names}
    os.environ.update(
        {
            "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
            "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS": "1",
            "CPPMEGA_FA4_SCORE_MOD": "1",
            "CPPMEGA_GRAPH_BIAS_BETA": "1",
        }
    )
    _set_current_structure_batch(structure_batch)
    try:
        bias = _graph_attention_bias_for_layer(
            layer,
            torch.zeros(4, 1, 8),
        )
        assert isinstance(bias, ChunkNativeGraphBias)
        assert tuple(bias.token_to_chunk_q.shape) == (1, 8)
        assert tuple(bias.token_to_chunk_k.shape) == (1, 8)

        with pytest.raises(
            NotImplementedError,
            match="sequence/context-parallel.*incremental decode",
        ):
            _graph_attention_bias_for_layer(
                layer,
                torch.zeros(4, 1, 8),
                inference_context=object(),
            )

        _set_current_structure_batch(
            {**structure_batch, "document_ids": torch.tensor([[1, 1, 2, 2]])}
        )
        with pytest.raises(ValueError, match="global sequence"):
            _graph_attention_bias_for_layer(
                layer,
                torch.zeros(4, 1, 8),
            )
    finally:
        _set_current_structure_batch(None)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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


def test_dense_tensor_only_production_path_requires_explicit_ablation():
    production = {
        "CPPMEGA_GRAPH_ROUTES_ENABLED": "0",
        "CPPMEGA_H200_GRAPH_PRIOR_RECEIPT": "/tmp/graph-prior.json",
    }
    with patch.dict(os.environ, production, clear=True):
        with pytest.raises(RuntimeError, match="tensor-only"):
            graph_dense_bias_enabled()

    with patch.dict(
        os.environ,
        {**production, "CPPMEGA_GRAPH_ROUTES_ABLATION": "1"},
        clear=True,
    ):
        assert graph_dense_bias_enabled() is False


def test_dense_graph_bias_is_disabled_by_explicit_ablation(monkeypatch):
    structure_batch = {
        "graph_domain_edges": torch.tensor([[[1, 0, 5]]], dtype=torch.long),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
    }
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", "1")
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ABLATION", "1")
    _set_current_structure_batch(structure_batch)
    try:
        assert (
            _graph_attention_bias_for_layer(
                _Layer(_DenseSelfAttention()),
                torch.zeros(2, 1, 8),
            )
            is None
        )
    finally:
        _set_current_structure_batch(None)


def test_pinned_transformer_layer_preserves_positional_attention_bias(monkeypatch):
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from cppmega.megatron.graph_route_attention_bias_patch import (
        apply_graph_route_attention_bias_patch,
    )

    class DenseProbe(TransformerLayer):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.self_attention = _DenseSelfAttention()
            self.config = SimpleNamespace(
                sequence_parallel=False,
                context_parallel_size=1,
            )
            self.seen_attention_bias = None

        def _forward_attention(self, *args, **kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            self.seen_attention_bias = kwargs.get("attention_bias")
            if len(args) > 8:
                self.seen_attention_bias = args[8]
            return hidden_states, None

        def _forward_mlp(self, hidden_states, *_args, **_kwargs):
            return hidden_states

    original_forward = TransformerLayer.forward
    structure_batch = {
        "graph_domain_edges": torch.tensor([[[1, 0, 5]]], dtype=torch.long),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
    }
    supplied_bias = torch.full((1, 1, 2, 2), 7.0)
    hidden_states = torch.zeros(2, 1, 8)
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", "1")
    _set_current_structure_batch(structure_batch)
    try:
        apply_graph_route_attention_bias_patch(force=True)
        layer = DenseProbe()
        output, _context = layer.forward(
            hidden_states,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            supplied_bias,
        )
    finally:
        TransformerLayer.forward = original_forward
        _set_current_structure_batch(None)

    assert output is hidden_states
    assert layer.seen_attention_bias is supplied_bias


def test_pinned_transformer_layer_injects_sidecar_bias_at_dense_gqa_seam(monkeypatch):
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from cppmega.megatron.graph_route_attention_bias_patch import (
        apply_graph_route_attention_bias_patch,
    )

    class DenseProbe(TransformerLayer):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.self_attention = _DenseSelfAttention()
            self.config = SimpleNamespace(
                sequence_parallel=False,
                context_parallel_size=1,
            )
            self.seen_attention_bias = None

        def _forward_attention(self, *args, **kwargs):
            self.seen_attention_bias = kwargs.get("attention_bias")
            if self.seen_attention_bias is None and len(args) > 8:
                self.seen_attention_bias = args[8]
            return args[0] if args else kwargs["hidden_states"], None

        def _forward_mlp(self, hidden_states, *_args, **_kwargs):
            return hidden_states

    original_forward = TransformerLayer.forward
    structure_batch = {
        "graph_domain_edges": torch.tensor([[[1, 0, 5]]], dtype=torch.long),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
    }
    hidden_states = torch.zeros(2, 1, 8)
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", "1")
    monkeypatch.setenv("CPPMEGA_GRAPH_BIAS_BETA", "3")
    _set_current_structure_batch(structure_batch)
    try:
        apply_graph_route_attention_bias_patch(force=True)
        layer = DenseProbe()
        output, _context = layer.forward(hidden_states, attention_bias=None)
    finally:
        TransformerLayer.forward = original_forward
        _set_current_structure_batch(None)

    assert output is hidden_states
    assert layer.seen_attention_bias is not None
    assert layer.seen_attention_bias.shape == (1, 1, 2, 2)
    assert layer.seen_attention_bias[0, 0, 1, 0].item() == pytest.approx(3.0)


def test_env_flag_accepts_only_documented_values_and_fails_closed():
    name = "CPPMEGA_TEST_STRICT_GRAPH_FLAG"
    previous = os.environ.get(name)
    try:
        for value in ("1", "true", "TRUE", " yes ", "on"):
            os.environ[name] = value
            assert _env_flag(name) is True
        for value in ("0", "false", "FALSE", " no ", "off"):
            os.environ[name] = value
            assert _env_flag(name) is False

        for value in ("", "   ", "tru", "0x1", "maybe"):
            os.environ[name] = value
            with pytest.raises(ValueError, match=name):
                _env_flag(name)

        os.environ.pop(name)
        assert _env_flag(name, "yes") is True
        assert _env_flag(name, "off") is False
        with pytest.raises(ValueError, match=name):
            _env_flag(name, "")
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def test_graph_dense_bias_enabled_rejects_malformed_flag_cr02(monkeypatch):
    """CR-02 regression: malformed CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS must raise.

    Previously _env_flag mapped every unrecognized value to False, so a typo
    like 'tru' silently disabled the dense graph bias path instead of failing
    loudly.  This test exercises the full graph_dense_bias_enabled() entry
    point with routes enabled.
    """
    monkeypatch.setenv("CPPMEGA_GRAPH_ROUTES_ENABLED", "1")

    # Malformed values must raise ValueError, not silently return False.
    for bad_value in ("tru", "0x1", "  tru  ", " true1", " 0x1 ", ""):
        monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", bad_value)
        with pytest.raises(ValueError):
            graph_dense_bias_enabled()

    # Documented true spellings enable the bias.
    for good_true in ("1", "true", "TRUE", "True", "yes", "on", " true "):
        monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", good_true)
        assert graph_dense_bias_enabled() is True

    # Documented false spellings disable the bias.
    for good_false in ("0", "false", "FALSE", "False", "no", "off", " false "):
        monkeypatch.setenv("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", good_false)
        assert graph_dense_bias_enabled() is False

    # Default (env var unset) remains enabled when routes are enabled.
    monkeypatch.delenv("CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS", raising=False)
    assert graph_dense_bias_enabled() is True
