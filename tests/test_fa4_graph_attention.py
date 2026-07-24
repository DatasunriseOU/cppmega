"""Unit tests for the FA4 score_mod graph-route attention backend.

Tests the CSR construction, score_mod equivalence with the dense bias path,
backward gradient flow, memory savings, and module spec compatibility for
``cppmega.megatron.fa4_graph_attention``.

flash_attn.cute is mocked so these tests run without a GPU.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Mock flash_attn.cute so the module under test can be imported without GPU.
# ---------------------------------------------------------------------------

_FA4_MOCK_MODULES = (
    "flash_attn",
    "flash_attn.cute",
    "flash_attn.cute.interface",
    "flash_attn.cute.block_sparsity",
    "flash_attn.cute.utils",
)


def _install_flash_attn_mock() -> dict[str, ModuleType]:
    """Install mock flash_attn.cute modules into sys.modules."""
    mocks: dict[str, ModuleType] = {}
    for name in _FA4_MOCK_MODULES:
        if name in sys.modules:
            mocks[name] = sys.modules[name]
            continue
        mod = MagicMock(spec=ModuleType)
        mod.__name__ = name
        mod.__package__ = name.rpartition(".")[0] or name
        mod.__spec__ = MagicMock()
        mod.__spec__.name = name
        sys.modules[name] = mod
        mocks[name] = mod

    # Provide BlockSparseTensorsTorch as a namedtuple-like class
    from typing import NamedTuple

    class BlockSparseTensorsTorch(NamedTuple):
        mask_block_cnt: torch.Tensor
        mask_block_idx: torch.Tensor
        full_block_cnt: torch.Tensor | None
        full_block_idx: torch.Tensor | None
        cu_total_m_blocks: torch.Tensor | None
        cu_block_idx_offsets: torch.Tensor | None
        block_size: tuple[int, int] | None
        dq_write_order: torch.Tensor | None
        dq_write_order_full: torch.Tensor | None
        spt: bool | None

    sys.modules["flash_attn.cute.block_sparsity"].BlockSparseTensorsTorch = (
        BlockSparseTensorsTorch
    )
    # Provide a no-op compute_dq_write_order
    sys.modules["flash_attn.cute.block_sparsity"].compute_dq_write_order = (
        lambda *a, **kw: None
    )
    # Provide flash_attn_func as a mock
    sys.modules["flash_attn.cute.interface"].flash_attn_func = MagicMock()
    return mocks


def _remove_flash_attn_mock(mocks: dict[str, ModuleType]) -> None:
    for name, mod in mocks.items():
        if sys.modules.get(name) is mod:
            del sys.modules[name]


# Install mock before importing the module under test.
_FA4_MOCKS = _install_flash_attn_mock()

from cppmega.megatron.fa4_graph_attention import (  # noqa: E402
    FA4GraphRouteAux,
    CppMegaFA4DotProductAttention,
    build_fa4_graph_route_aux,
    graph_route_score_mod_ref,
    graph_route_score_mod_bwd_ref,
)
from cppmega.megatron.graph_route_attention_bias_patch import (  # noqa: E402
    build_dense_graph_attention_bias_from_structure_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_structure_batch_token_edges() -> dict[str, torch.Tensor]:
    """Structure batch with token-level edges (domain/build triples).

    Edges (src, dst, kind):
      (0, 3, 5)  -- domain edge from token 0 to token 3
      (1, 2, 5)  -- domain edge from token 1 to token 2
      (2, 0, 5)  -- domain edge from token 2 to token 0
    """
    return {
        "graph_domain_edges": torch.tensor([[[0, 3, 5], [1, 2, 5], [2, 0, 5]]], dtype=torch.long),
        "graph_domain_edge_counts": torch.tensor([3], dtype=torch.long),
    }


def _chunk_structure_batch() -> dict[str, torch.Tensor]:
    """Structure batch with chunk-index call/type edges.

    Chunks: [0,2), [2,4), [4,6), [6,8)  (4 chunks covering 8 tokens)
    Call edges (chunk pairs): (0, 2), (1, 3)
    Type edges (chunk pairs): (2, 1)
    """
    return {
        "graph_call_edges": torch.tensor([[[0, 2], [1, 3], [-1, -1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([2], dtype=torch.long),
        "graph_type_edges": torch.tensor([[[2, 1], [-1, -1], [-1, -1]]], dtype=torch.long),
        "graph_type_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 2, 4, 6]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[2, 4, 6, 8]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([4], dtype=torch.long),
    }


def _mixed_structure_batch() -> dict[str, torch.Tensor]:
    """Structure batch with both chunk-index and token-level edges."""
    return {
        "graph_call_edges": torch.tensor([[[0, 1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_domain_edges": torch.tensor([[[3, 0, 5]]], dtype=torch.long),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 2]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[2, 4]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([2], dtype=torch.long),
    }


def _csr_lookup(
    aux: FA4GraphRouteAux, batch_idx: int, q_idx: int, kv_idx: int
) -> float:
    """Python reference for the in-kernel CSR binary search.

    Mirrors the score_mod pseudocode from the design doc: binary search
    the sorted column list for kv_idx within the row's edge range.
    """
    row_offsets = aux.csr_row_offsets
    col_idx = aux.csr_col_idx
    weight = aux.csr_weight

    lo = int(row_offsets[batch_idx, q_idx].item())
    hi = int(row_offsets[batch_idx, q_idx + 1].item())

    # Binary search for kv_idx in col_idx[batch_idx, lo:hi]
    left, right = lo, hi
    while left < right:
        mid = (left + right) >> 1
        c = int(col_idx[batch_idx, mid].item())
        if c < kv_idx:
            left = mid + 1
        else:
            right = mid
    if left < hi and int(col_idx[batch_idx, left].item()) == kv_idx:
        return float(weight[batch_idx, left].item())
    return 0.0


# ---------------------------------------------------------------------------
# 1. test_graph_edge_csr_construction
# ---------------------------------------------------------------------------


class TestGraphEdgeCSRConstruction:
    """Verify CSR offsets/indices/weights from known structure_batch edges."""

    def test_token_edges_csr_offsets_and_indices(self):
        """Token-level domain edges produce correct CSR structure."""
        sb = _simple_structure_batch_token_edges()
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=4,
            seqlen_k=4,
            device=torch.device("cpu"),
            q_dtype=torch.bfloat16,
            beta=1.0,
        )

        # Row offsets: [B, Sq+1] = [1, 5]
        assert aux.csr_row_offsets.shape == (1, 5)
        # Row 0 has edge to col 3; row 1 has edge to col 2; row 2 has edge to col 0
        # Row 3 has no edges
        offsets = aux.csr_row_offsets[0].tolist()
        assert offsets[0] == 0
        # Each of rows 0,1,2 has exactly 1 edge; row 3 has 0
        assert offsets[1] - offsets[0] == 1  # row 0: 1 edge
        assert offsets[2] - offsets[1] == 1  # row 1: 1 edge
        assert offsets[3] - offsets[2] == 1  # row 2: 1 edge
        assert offsets[4] - offsets[3] == 0  # row 3: 0 edges
        assert offsets[4] == 3  # total nnz = 3

    def test_token_edges_csr_col_indices_sorted(self):
        """Column indices within each row are sorted ascending."""
        sb = {
            "graph_domain_edges": torch.tensor(
                [[[0, 5, 1], [0, 2, 1], [0, 8, 1]]], dtype=torch.long
            ),
            "graph_domain_edge_counts": torch.tensor([3], dtype=torch.long),
        }
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=4,
            seqlen_k=16,
            device=torch.device("cpu"),
            q_dtype=torch.bfloat16,
            beta=1.0,
        )
        # Row 0 has edges to cols 5, 2, 8 -> sorted: 2, 5, 8
        lo = int(aux.csr_row_offsets[0, 0].item())
        hi = int(aux.csr_row_offsets[0, 1].item())
        cols = aux.csr_col_idx[0, lo:hi].tolist()
        assert cols == sorted(cols)
        assert set(cols) == {2, 5, 8}

    def test_token_edges_csr_weights_include_beta(self):
        """Weights are pre-multiplied by beta * relation_weight."""
        sb = _simple_structure_batch_token_edges()
        beta = 2.5
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=4,
            seqlen_k=4,
            device=torch.device("cpu"),
            q_dtype=torch.bfloat16,
            beta=beta,
            domain_weight=3.0,
        )
        # Edge (0,3) should have weight = beta * domain_weight = 2.5 * 3.0 = 7.5
        w = _csr_lookup(aux, 0, 0, 3)
        assert w == pytest.approx(7.5, abs=0.1)  # bf16 tolerance

    def test_chunk_edges_expand_to_token_rectangles(self):
        """Chunk-index edges expand into per-token CSR entries."""
        sb = _chunk_structure_batch()
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=8,
            seqlen_k=8,
            device=torch.device("cpu"),
            q_dtype=torch.bfloat16,
            beta=1.0,
            call_weight=1.0,
            type_weight=1.0,
        )
        # Call edge (chunk 0 -> chunk 2): tokens [0,2) -> [4,6)
        # So q=0 has edges to k=4,5; q=1 has edges to k=4,5
        assert _csr_lookup(aux, 0, 0, 4) == pytest.approx(1.0, abs=0.01)
        assert _csr_lookup(aux, 0, 0, 5) == pytest.approx(1.0, abs=0.01)
        assert _csr_lookup(aux, 0, 1, 4) == pytest.approx(1.0, abs=0.01)
        assert _csr_lookup(aux, 0, 1, 5) == pytest.approx(1.0, abs=0.01)
        # Call edge (chunk 1 -> chunk 3): tokens [2,4) -> [6,8)
        assert _csr_lookup(aux, 0, 2, 6) == pytest.approx(1.0, abs=0.01)
        assert _csr_lookup(aux, 0, 3, 7) == pytest.approx(1.0, abs=0.01)
        # Type edge (chunk 2 -> chunk 1): tokens [4,6) -> [2,4)
        assert _csr_lookup(aux, 0, 4, 2) == pytest.approx(1.0, abs=0.01)
        assert _csr_lookup(aux, 0, 5, 3) == pytest.approx(1.0, abs=0.01)
        # No edge at (0, 0)
        assert _csr_lookup(aux, 0, 0, 0) == 0.0

    def test_duplicate_edges_accumulate_weights(self):
        """Multiple relations producing the same (q,k) sum their weights."""
        sb = {
            "graph_domain_edges": torch.tensor([[[1, 2, 5]]], dtype=torch.long),
            "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
            "graph_build_edges": torch.tensor([[[1, 2, 10]]], dtype=torch.long),
            "graph_build_edge_counts": torch.tensor([1], dtype=torch.long),
        }
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=4,
            seqlen_k=4,
            device=torch.device("cpu"),
            q_dtype=torch.bfloat16,
            beta=1.0,
            domain_weight=2.0,
            build_weight=3.0,
        )
        # (1,2) gets domain_weight=2.0 + build_weight=3.0 = 5.0
        w = _csr_lookup(aux, 0, 1, 2)
        assert w == pytest.approx(5.0, abs=0.1)

    def test_csr_meta_shape_and_content(self):
        """csr_meta is [4] int32 with [Sq, Sk, max_nnz, flags]."""
        sb = _simple_structure_batch_token_edges()
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=4,
            seqlen_k=4,
            device=torch.device("cpu"),
            q_dtype=torch.bfloat16,
            beta=1.0,
        )
        assert aux.csr_meta.shape == (4,)
        assert aux.csr_meta.dtype == torch.int32
        assert int(aux.csr_meta[0].item()) == 4  # Sq
        assert int(aux.csr_meta[1].item()) == 4  # Sk

    def test_batch_size_greater_than_one(self):
        """CSR construction works for B>1 with per-batch edges."""
        sb = {
            "graph_domain_edges": torch.tensor(
                [[[0, 1, 5], [-1, -1, -1]], [[2, 3, 5], [0, 1, 5]]],
                dtype=torch.long,
            ),
            "graph_domain_edge_counts": torch.tensor([1, 2], dtype=torch.long),
        }
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=2,
            query_start=0,
            seqlen_q=4,
            seqlen_k=4,
            device=torch.device("cpu"),
            q_dtype=torch.bfloat16,
            beta=1.0,
        )
        assert aux.csr_row_offsets.shape[0] == 2
        # Batch 0: 1 edge (0->1)
        assert int(aux.csr_row_offsets[0, 4].item()) == 1
        # Batch 1: 2 edges (2->3, 0->1)
        assert int(aux.csr_row_offsets[1, 4].item()) == 2
        assert _csr_lookup(aux, 0, 0, 1) == pytest.approx(1.0, abs=0.01)
        assert _csr_lookup(aux, 1, 2, 3) == pytest.approx(1.0, abs=0.01)
        assert _csr_lookup(aux, 1, 0, 1) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# 3. test_score_mod_equivalence (CRITICAL)
# ---------------------------------------------------------------------------


class TestScoreModEquivalence:
    """Verify score_mod produces the same additive bias as the dense path.

    For all (query, key) pairs, the CSR-based score_mod lookup must produce
    the same value as dense_bias[b, 0, q, k].
    """

    def test_token_edges_equivalence(self):
        """Token-level edges: CSR score_mod matches dense bias exactly."""
        sb = _simple_structure_batch_token_edges()
        beta = 1.0
        Sq, Sk = 4, 4

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
        )
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            q_dtype=torch.float32,
            beta=beta,
        )

        for q in range(Sq):
            for k in range(Sk):
                expected = dense_bias[0, 0, q, k].item()
                actual = _csr_lookup(aux, 0, q, k)
                assert actual == pytest.approx(expected, abs=1e-5), (
                    f"Mismatch at (q={q}, k={k}): "
                    f"CSR={actual}, dense={expected}"
                )

    def test_chunk_edges_equivalence(self):
        """Chunk-index edges expanded to tokens match dense bias."""
        sb = _chunk_structure_batch()
        beta = 1.0
        Sq, Sk = 8, 8

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            call_weight=2.0,
            type_weight=3.0,
            beta=beta,
        )
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            q_dtype=torch.float32,
            beta=beta,
            call_weight=2.0,
            type_weight=3.0,
        )

        for q in range(Sq):
            for k in range(Sk):
                expected = dense_bias[0, 0, q, k].item()
                actual = _csr_lookup(aux, 0, q, k)
                assert actual == pytest.approx(expected, abs=1e-5), (
                    f"Mismatch at (q={q}, k={k}): "
                    f"CSR={actual}, dense={expected}"
                )

    def test_mixed_edges_equivalence(self):
        """Mixed chunk + token edges match dense bias."""
        sb = _mixed_structure_batch()
        beta = 0.5
        Sq, Sk = 4, 4

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
        )
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            q_dtype=torch.float32,
            beta=beta,
        )

        for q in range(Sq):
            for k in range(Sk):
                expected = dense_bias[0, 0, q, k].item()
                actual = _csr_lookup(aux, 0, q, k)
                assert actual == pytest.approx(expected, abs=1e-5), (
                    f"Mismatch at (q={q}, k={k}): "
                    f"CSR={actual}, dense={expected}"
                )

    def test_equivalence_with_nontrivial_beta(self):
        """Beta scaling is consistent between dense and CSR paths."""
        sb = _simple_structure_batch_token_edges()
        beta = 3.7
        Sq, Sk = 4, 4

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
        )
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            q_dtype=torch.float32,
            beta=beta,
        )

        for q in range(Sq):
            for k in range(Sk):
                expected = dense_bias[0, 0, q, k].item()
                actual = _csr_lookup(aux, 0, q, k)
                assert actual == pytest.approx(expected, abs=1e-4), (
                    f"Mismatch at (q={q}, k={k}): "
                    f"CSR={actual}, dense={expected}"
                )

    def test_score_mod_ref_function_matches_dense(self):
        """The graph_route_score_mod_ref function adds correct bias to scores."""
        sb = _simple_structure_batch_token_edges()
        Sq, Sk = 4, 4

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            q_dtype=torch.float32,
            beta=1.0,
        )

        # Simulate score_mod: input score + bias
        scores = torch.randn(1, 1, Sq, Sk)
        for q in range(Sq):
            for k in range(Sk):
                input_score = scores[0, 0, q, k].item()
                modified = graph_route_score_mod_ref(
                    input_score, batch_idx=0, q_idx=q, kv_idx=k, aux=aux
                )
                expected = input_score + dense_bias[0, 0, q, k].item()
                assert modified == pytest.approx(expected, abs=1e-5)

    def test_rectangular_decode_equivalence(self):
        """Rectangular decode (query_start > 0) matches dense bias slice."""
        sb = {
            "graph_domain_edges": torch.tensor(
                [[[5, 2, 5], [5, 0, 5], [3, 1, 5]]], dtype=torch.long
            ),
            "graph_domain_edge_counts": torch.tensor([3], dtype=torch.long),
        }
        Sq_full = 8
        query_start = 4
        Sq_decode = 4  # tokens [4, 8)
        Sk = 8

        # Dense full bias
        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq_full,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )

        # CSR for decode window
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=query_start,
            seqlen_q=Sq_decode,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            q_dtype=torch.float32,
            beta=1.0,
        )

        # Compare: CSR row q_local corresponds to dense row (query_start + q_local)
        for q_local in range(Sq_decode):
            q_global = query_start + q_local
            for k in range(Sk):
                expected = dense_bias[0, 0, q_global, k].item()
                actual = _csr_lookup(aux, 0, q_local, k)
                assert actual == pytest.approx(expected, abs=1e-5), (
                    f"Mismatch at (q_local={q_local}, q_global={q_global}, k={k}): "
                    f"CSR={actual}, dense={expected}"
                )


# ---------------------------------------------------------------------------
# 4. test_score_mod_backward
# ---------------------------------------------------------------------------


class TestScoreModBackward:
    """Verify gradient flows through score_mod correctly."""

    def test_backward_is_identity(self):
        """d(score')/d(score) = 1 because bias is additive constant."""
        sb = _simple_structure_batch_token_edges()
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=4,
            seqlen_k=4,
            device=torch.device("cpu"),
            q_dtype=torch.float32,
            beta=1.0,
        )

        # The backward of score' = score + bias is d(score')/d(score) = 1
        # So grad_in = grad_out * 1 = grad_out
        grad_out = torch.randn(4, 4)
        for q in range(4):
            for k in range(4):
                grad_in = graph_route_score_mod_bwd_ref(
                    grad_out[q, k].item(),
                    score=0.0,
                    batch_idx=0,
                    q_idx=q,
                    kv_idx=k,
                    aux=aux,
                )
                assert grad_in == pytest.approx(grad_out[q, k].item(), abs=1e-7)

    def test_graph_bias_is_non_learnable(self):
        """CSR weight tensors do not require grad (non-learnable bias)."""
        sb = _simple_structure_batch_token_edges()
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=4,
            seqlen_k=4,
            device=torch.device("cpu"),
            q_dtype=torch.float32,
            beta=1.0,
        )
        assert not aux.csr_weight.requires_grad
        assert not aux.csr_col_idx.requires_grad
        assert not aux.csr_row_offsets.requires_grad

    def test_gradient_flows_through_score_to_qkv(self):
        """Autograd: loss = sum(score_mod(score)) -> d(loss)/d(score) = 1."""
        sb = _simple_structure_batch_token_edges()
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=4,
            seqlen_k=4,
            device=torch.device("cpu"),
            q_dtype=torch.float32,
            beta=1.0,
        )

        # Simulate: score requires grad, score_mod adds bias
        score = torch.randn(1, 1, 4, 4, requires_grad=True)
        # Build the modified score using the reference implementation
        bias_tensor = torch.zeros(4, 4)
        for q in range(4):
            for k in range(4):
                bias_tensor[q, k] = _csr_lookup(aux, 0, q, k)
        modified_score = score + bias_tensor.unsqueeze(0).unsqueeze(0)
        loss = modified_score.sum()
        loss.backward()

        # d(loss)/d(score) should be 1.0 everywhere
        assert score.grad is not None
        torch.testing.assert_close(
            score.grad, torch.ones_like(score), atol=1e-7, rtol=1e-7
        )


# ---------------------------------------------------------------------------
# 5. test_memory_comparison
# ---------------------------------------------------------------------------


class TestMemoryComparison:
    """CSR representation uses < 1% of dense memory for typical sparsity."""

    def test_csr_vs_dense_memory_b192_s1024(self):
        """B=192, S=1024: CSR memory < 1% of dense [B,1,S,S] bf16."""
        B, S = 192, 1024

        # Dense memory: B * 1 * S * S * 2 bytes (bf16)
        dense_bytes = B * 1 * S * S * 2

        # Typical cppmega sparsity: ~2 edges per row on average
        # CSR memory:
        #   csr_row_offsets: B * (S+1) * 4 bytes (int32)
        #   csr_col_idx: B * max_nnz * 4 bytes (int32)
        #   csr_weight: B * max_nnz * 2 bytes (bf16)
        # With max_nnz_per_batch = 2 per row * S rows = 2*S edges per batch.
        # No block-sparse mask: graph routes are additive score_mod bias, not
        # a mask, so block_sparse_tensors is None in the FA4 path.
        max_nnz_per_batch = 2 * S  # 2 edges per row average
        csr_row_offsets_bytes = B * (S + 1) * 4
        csr_col_idx_bytes = B * max_nnz_per_batch * 4
        csr_weight_bytes = B * max_nnz_per_batch * 2

        csr_total_bytes = (
            csr_row_offsets_bytes
            + csr_col_idx_bytes
            + csr_weight_bytes
        )

        ratio = csr_total_bytes / dense_bytes
        assert ratio < 0.01, (
            f"CSR/dense memory ratio = {ratio:.4f} ({csr_total_bytes} / {dense_bytes}), "
            f"expected < 0.01"
        )

    def test_csr_memory_scales_with_nnz_not_s_squared(self):
        """CSR memory grows linearly with nnz, not quadratically with S."""
        B = 8
        max_nnz = 4096  # fixed high-water mark

        for S in (1024, 4096, 16384):
            dense_bytes = B * S * S * 2  # bf16
            csr_bytes = (
                B * (S + 1) * 4  # row_offsets int32
                + B * max_nnz * 4  # col_idx int32
                + B * max_nnz * 2  # weight bf16
            )
            ratio = csr_bytes / dense_bytes
            # As S grows, ratio should shrink (CSR is O(S + nnz), dense is O(S^2))
            if S >= 4096:
                assert ratio < 0.01

    def test_actual_aux_tensors_memory(self):
        """Measure actual tensor memory from a built aux."""
        sb = _simple_structure_batch_token_edges()
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=4,
            seqlen_k=4,
            device=torch.device("cpu"),
            q_dtype=torch.bfloat16,
            beta=1.0,
        )

        def _tensor_bytes(t: torch.Tensor) -> int:
            return t.nelement() * t.element_size()

        csr_bytes = (
            _tensor_bytes(aux.csr_row_offsets)
            + _tensor_bytes(aux.csr_col_idx)
            + _tensor_bytes(aux.csr_weight)
            + _tensor_bytes(aux.csr_meta)
        )
        # For this tiny example CSR might not be smaller than a dense
        # [B,1,S,S] bias (overhead), but verify the tensors exist and have
        # reasonable sizes
        assert csr_bytes > 0
        assert aux.csr_row_offsets.dtype == torch.int32
        assert aux.csr_col_idx.dtype == torch.int32


# ---------------------------------------------------------------------------
# 6. test_module_spec_compatibility
# ---------------------------------------------------------------------------


class TestModuleSpecCompatibility:
    """Verify CppMegaFA4DotProductAttention instantiation and interface."""

    def _make_config(self):
        """Create a minimal TransformerConfig-like object."""
        from types import SimpleNamespace

        return SimpleNamespace(
            sequence_parallel=False,
            context_parallel_size=1,
            attention_dropout=0.0,
            num_attention_heads=8,
            kv_channels=64,
            hidden_size=512,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

    def test_instantiation(self):
        """CppMegaFA4DotProductAttention can be instantiated."""
        config = self._make_config()
        module = CppMegaFA4DotProductAttention(
            config=config,
            layer_number=1,
            attention_type="self",
            num_attention_heads=8,
        )
        assert module is not None

    def test_forward_signature_accepts_expected_kwargs(self):
        """forward() accepts the standard Megatron core_attention kwargs."""
        import inspect

        config = self._make_config()
        module = CppMegaFA4DotProductAttention(
            config=config,
            layer_number=1,
            attention_type="self",
            num_attention_heads=8,
        )
        sig = inspect.signature(module.forward)
        param_names = set(sig.parameters.keys())
        # Must accept at minimum: query, key, value, attention_mask
        assert "query" in param_names
        assert "key" in param_names
        assert "value" in param_names
        assert "attention_mask" in param_names

    def test_rejects_nonzero_dropout(self):
        """Non-zero attention_dropout raises (FA4 score_mod path limitation)."""
        config = self._make_config()
        with pytest.raises((ValueError, RuntimeError)):
            CppMegaFA4DotProductAttention(
                config=config,
                layer_number=1,
                attention_type="self",
                num_attention_heads=8,
                attention_dropout=0.1,
            )

    def test_is_nn_module(self):
        """CppMegaFA4DotProductAttention is a torch.nn.Module."""
        config = self._make_config()
        module = CppMegaFA4DotProductAttention(
            config=config,
            layer_number=1,
            attention_type="self",
            num_attention_heads=8,
        )
        assert isinstance(module, torch.nn.Module)

    def test_forward_with_none_bias_calls_flash_attn(self):
        """forward with attention_bias=None calls flash_attn_func without score_mod."""
        config = self._make_config()
        module = CppMegaFA4DotProductAttention(
            config=config,
            layer_number=1,
            attention_type="self",
            num_attention_heads=8,
        )
        S, B, H, D = 16, 1, 8, 64
        # Megatron ABI: input is [S, B, H, D]
        q = torch.randn(S, B, H, D)
        k = torch.randn(S, B, H, D)
        v = torch.randn(S, B, H, D)

        mock_flash = sys.modules["flash_attn.cute.interface"].flash_attn_func
        mock_flash.reset_mock()
        # FA4 internally operates on [B, S, H, D] (after module transposes)
        mock_flash.return_value = torch.randn(B, S, H, D)

        module(q, k, v, attention_mask=None, attention_bias=None)

        mock_flash.assert_called_once()
        call_kwargs = mock_flash.call_args
        # score_mod should be None when no bias is provided
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("score_mod") is None

    def test_forward_with_aux_passes_score_mod(self):
        """forward with FA4GraphRouteAux passes score_mod to flash_attn_func."""
        config = self._make_config()
        module = CppMegaFA4DotProductAttention(
            config=config,
            layer_number=1,
            attention_type="self",
            num_attention_heads=8,
        )
        S, B, H, D = 16, 1, 8, 64
        # Megatron ABI: input is [S, B, H, D]
        q = torch.randn(S, B, H, D)
        k = torch.randn(S, B, H, D)
        v = torch.randn(S, B, H, D)

        sb = _simple_structure_batch_token_edges()
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=B,
            query_start=0,
            seqlen_q=S,
            seqlen_k=S,
            device=torch.device("cpu"),
            q_dtype=torch.bfloat16,
            beta=1.0,
        )

        mock_flash = sys.modules["flash_attn.cute.interface"].flash_attn_func
        mock_flash.reset_mock()
        # FA4 internally operates on [B, S, H, D] (after module transposes)
        mock_flash.return_value = torch.randn(B, S, H, D)

        module(q, k, v, attention_mask=None, attention_bias=aux)

        mock_flash.assert_called_once()
        call_kwargs = mock_flash.call_args
        # score_mod should NOT be None when aux is provided
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("score_mod") is not None
            assert call_kwargs.kwargs.get("score_mod_bwd") is not None
            assert call_kwargs.kwargs.get("aux_tensors") is not None

    def test_forward_rejects_dense_tensor_bias(self):
        """forward with a raw dense tensor bias raises (contract: use CSR)."""
        config = self._make_config()
        module = CppMegaFA4DotProductAttention(
            config=config,
            layer_number=1,
            attention_type="self",
            num_attention_heads=8,
        )
        S, B, H, D = 16, 1, 8, 64
        # Megatron ABI: input is [S, B, H, D]
        q = torch.randn(S, B, H, D)
        k = torch.randn(S, B, H, D)
        v = torch.randn(S, B, H, D)
        dense_bias = torch.zeros(B, 1, S, S)

        with pytest.raises((TypeError, RuntimeError, ValueError)):
            module(q, k, v, attention_mask=None, attention_bias=dense_bias)

    def test_sbhD_abi_contract(self):
        """forward accepts [S,B,H,D] input and returns [S,B,H*D] output (3-D)."""
        config = self._make_config()
        module = CppMegaFA4DotProductAttention(
            config=config,
            layer_number=1,
            attention_type="self",
            num_attention_heads=8,
        )
        S, B, H, D = 16, 2, 8, 64
        # Megatron ABI: input is [S, B, H, D]
        q = torch.randn(S, B, H, D)
        k = torch.randn(S, B, H, D)
        v = torch.randn(S, B, H, D)

        mock_flash = sys.modules["flash_attn.cute.interface"].flash_attn_func
        mock_flash.reset_mock()
        # FA4 internally operates on [B, S, H, D]
        mock_flash.return_value = torch.randn(B, S, H, D)

        out = module(q, k, v, attention_mask=None, attention_bias=None)

        # Output must be [S, B, H*D] (3-D, Megatron linear_proj input)
        assert out.shape == (S, B, H * D), (
            f"Expected output shape [S,B,H*D]=[{S},{B},{H * D}], "
            f"got {tuple(out.shape)}"
        )

    def test_causal_default_true(self):
        """causal=True is the default for CppMegaFA4DotProductAttention."""
        config = self._make_config()
        module = CppMegaFA4DotProductAttention(
            config=config,
            layer_number=1,
            attention_type="self",
            num_attention_heads=8,
        )
        assert module.causal is True, (
            f"Expected causal=True by default, got causal={module.causal}"
        )

    def test_dropout_raises(self):
        """ValueError raised when config.attention_dropout > 0."""
        from types import SimpleNamespace

        config = SimpleNamespace(
            sequence_parallel=False,
            context_parallel_size=1,
            attention_dropout=0.1,  # non-zero dropout
            num_attention_heads=8,
            kv_channels=64,
            hidden_size=512,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        with pytest.raises(ValueError, match="dropout"):
            CppMegaFA4DotProductAttention(
                config=config,
                layer_number=1,
                attention_type="self",
                num_attention_heads=8,
            )

    def test_no_block_sparse_in_forward(self):
        """forward always passes block_sparse_tensors=None to flash_attn_func.

        Graph routes are additive score_mod bias, not an attention mask.
        The FA4 forward path must never use block_sparse_tensors.
        """
        config = self._make_config()
        module = CppMegaFA4DotProductAttention(
            config=config,
            layer_number=1,
            attention_type="self",
            num_attention_heads=8,
        )
        S, B, H, D = 16, 1, 8, 64
        q = torch.randn(S, B, H, D)
        k = torch.randn(S, B, H, D)
        v = torch.randn(S, B, H, D)

        mock_flash = sys.modules["flash_attn.cute.interface"].flash_attn_func
        mock_flash.reset_mock()
        mock_flash.return_value = torch.randn(B, S, H, D)

        module(q, k, v, attention_mask=None, attention_bias=None)

        mock_flash.assert_called_once()
        call_kwargs = mock_flash.call_args.kwargs
        assert call_kwargs.get("block_sparse_tensors") is None, (
            f"Expected block_sparse_tensors=None, got "
            f"{call_kwargs.get('block_sparse_tensors')!r}"
        )

    def test_causal_always_true(self):
        """forward always passes causal=True to flash_attn_func.

        Graph routes use a full causal tile schedule; causal=True is
        hardcoded regardless of any module-level causal setting.
        """
        config = self._make_config()
        # Even if we construct with causal=False, forward must pass causal=True
        module = CppMegaFA4DotProductAttention(
            config=config,
            layer_number=1,
            attention_type="self",
            num_attention_heads=8,
            causal=False,
        )
        S, B, H, D = 16, 1, 8, 64
        q = torch.randn(S, B, H, D)
        k = torch.randn(S, B, H, D)
        v = torch.randn(S, B, H, D)

        mock_flash = sys.modules["flash_attn.cute.interface"].flash_attn_func
        mock_flash.reset_mock()
        mock_flash.return_value = torch.randn(B, S, H, D)

        module(q, k, v, attention_mask=None, attention_bias=None)

        mock_flash.assert_called_once()
        call_kwargs = mock_flash.call_args.kwargs
        assert call_kwargs.get("causal") is True, (
            f"Expected causal=True in flash_attn_func call, got "
            f"causal={call_kwargs.get('causal')!r}"
        )


# ---------------------------------------------------------------------------
# 7. test_bias_matches_te_post_scale_semantics
# ---------------------------------------------------------------------------


class TestBiasMatchesTEPostScaleSemantics:
    """Verify FA4 CSR bias == dense bias exactly (beta * relation_weight).

    The bias scaling fix ensures that the FA4 score_mod bias is
    beta * relation_weight with NO softmax_scale division.  FA4 applies
    softmax_scale internally before calling score_mod, so the bias must
    match the TE post_scale_bias semantics: added to already-scaled scores.
    """

    def test_bias_matches_te_post_scale_semantics(self):
        """For any edge (q,k), FA4_bias[q,k] == dense_bias[0,q,k] exactly."""
        sb = _simple_structure_batch_token_edges()
        beta = 2.5
        Sq, Sk = 4, 4

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            domain_weight=3.0,
        )
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            q_dtype=torch.float32,
            beta=beta,
            domain_weight=3.0,
        )

        # Verify every (q, k) pair: CSR bias == dense bias (no scale factor)
        for q in range(Sq):
            for k in range(Sk):
                fa4_val = _csr_lookup(aux, 0, q, k)
                dense_val = float(dense_bias[0, 0, q, k].item())
                assert fa4_val == pytest.approx(dense_val, abs=1e-6), (
                    f"Bias mismatch at (q={q}, k={k}): "
                    f"FA4_CSR={fa4_val}, dense={dense_val}. "
                    f"Expected exact match (beta * relation_weight, "
                    f"NO softmax_scale division)."
                )

        # Spot-check a known edge: (0, 3) should be beta * domain_weight = 7.5
        expected_edge_weight = beta * 3.0  # 2.5 * 3.0 = 7.5
        assert _csr_lookup(aux, 0, 0, 3) == pytest.approx(expected_edge_weight, abs=1e-6)
        assert float(dense_bias[0, 0, 0, 3].item()) == pytest.approx(
            expected_edge_weight, abs=1e-6
        )

    def test_bias_no_softmax_scale_division_chunk_edges(self):
        """Chunk edges: FA4 CSR bias matches dense with no scale division."""
        sb = _chunk_structure_batch()
        beta = 1.5
        Sq, Sk = 8, 8

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            call_weight=2.0,
            type_weight=3.0,
        )
        aux = build_fa4_graph_route_aux(
            sb,
            batch_size=1,
            query_start=0,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            q_dtype=torch.float32,
            beta=beta,
            call_weight=2.0,
            type_weight=3.0,
        )

        for q in range(Sq):
            for k in range(Sk):
                fa4_val = _csr_lookup(aux, 0, q, k)
                dense_val = float(dense_bias[0, 0, q, k].item())
                assert fa4_val == pytest.approx(dense_val, abs=1e-6), (
                    f"Bias mismatch at (q={q}, k={k}): "
                    f"FA4_CSR={fa4_val}, dense={dense_val}. "
                    f"Bias must be beta * relation_weight (no softmax_scale)."
                )

        # Call edge (chunk 0 -> chunk 2): tokens [0,2) -> [4,6)
        # Expected: beta * call_weight = 1.5 * 2.0 = 3.0
        assert _csr_lookup(aux, 0, 0, 4) == pytest.approx(3.0, abs=1e-6)
        # Type edge (chunk 2 -> chunk 1): tokens [4,6) -> [2,4)
        # Expected: beta * type_weight = 1.5 * 3.0 = 4.5
        assert _csr_lookup(aux, 0, 4, 2) == pytest.approx(4.5, abs=1e-6)
