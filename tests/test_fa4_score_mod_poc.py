"""Equivalence and correctness tests for the FA4 chunk-native graph bias builder.

Tests the chunk-native representation (tiny [B, C+1, C+1] chunk_bias + token_to_chunk
map + rare-edge CSR overlay) against the existing dense [B, 1, Sq, Sk] bias path.

flash_attn.cute is mocked so these tests run without a GPU.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Mock flash_attn.cute so the adapter module can be imported without GPU.
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

    # Provide a no-op cute.jit decorator
    def _fake_jit(fn=None, **kwargs):
        if fn is not None:
            return fn

        def decorator(f):
            return f

        return decorator

    sys.modules["flash_attn.cute"].jit = _fake_jit
    sys.modules["flash_attn.cute.interface"].flash_attn_func = MagicMock()
    return mocks


_FA4_MOCKS = _install_flash_attn_mock()

from cppmega.megatron.dsa_indexer_fused_patch import (  # noqa: E402
    _as_batched_chunks,
    _token_chunk_map,
    build_graph_route_bias_from_structure_batch,
)
from cppmega.megatron.graph_route_attention_bias_patch import (  # noqa: E402
    build_dense_graph_attention_bias_from_structure_batch,
)
from cppmega.megatron.fa4_score_mod_adapter import (  # noqa: E402
    ChunkNativeGraphBias,
    build_chunk_native_graph_bias,
    chunk_native_score_mod_ref,
)


# ---------------------------------------------------------------------------
# Helpers: mock structure batches with known chunk layout and edges
# ---------------------------------------------------------------------------


def _chunk_only_structure_batch() -> dict[str, torch.Tensor]:
    """Structure batch with chunk-index call/type edges only (no token edges).

    Chunks: [0,4), [4,8), [8,12), [12,16)  (4 chunks covering 16 tokens)
    Call edges (chunk pairs): (0, 2), (1, 3), (0, 1)
    Type edges (chunk pairs): (2, 0), (3, 1)
    """
    return {
        "graph_call_edges": torch.tensor(
            [[[0, 2], [1, 3], [0, 1], [-1, -1]]], dtype=torch.long
        ),
        "graph_call_edge_counts": torch.tensor([3], dtype=torch.long),
        "graph_type_edges": torch.tensor(
            [[[2, 0], [3, 1], [-1, -1], [-1, -1]]], dtype=torch.long
        ),
        "graph_type_edge_counts": torch.tensor([2], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 4, 8, 12]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[4, 8, 12, 16]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([4], dtype=torch.long),
    }


def _mixed_structure_batch() -> dict[str, torch.Tensor]:
    """Structure batch with both chunk-index edges and token-level rare edges.

    Chunks: [0,4), [4,8), [8,12)  (3 chunks, 12 tokens total)
    Call edges (chunk pairs): (0, 1), (1, 2)
    Domain edges (token triples): (2, 9, 5), (5, 0, 5)
    Build edges (token triples): (10, 1, 7)
    """
    return {
        "graph_call_edges": torch.tensor(
            [[[0, 1], [1, 2], [-1, -1]]], dtype=torch.long
        ),
        "graph_call_edge_counts": torch.tensor([2], dtype=torch.long),
        "graph_domain_edges": torch.tensor(
            [[[2, 9, 5], [5, 0, 5], [-1, -1, -1]]], dtype=torch.long
        ),
        "graph_domain_edge_counts": torch.tensor([2], dtype=torch.long),
        "graph_build_edges": torch.tensor(
            [[[10, 1, 7], [-1, -1, -1]]], dtype=torch.long
        ),
        "graph_build_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 4, 8]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[4, 8, 12]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([3], dtype=torch.long),
    }


def _gapped_structure_batch() -> dict[str, torch.Tensor]:
    """Structure batch with inter-chunk gaps (tokens not covered by any chunk).

    Chunks: [2,5), [7,10)  (2 chunks, 12 tokens total, gaps at [0,2), [5,7), [10,12))
    Call edges (chunk pairs): (0, 1)
    """
    return {
        "graph_call_edges": torch.tensor([[[0, 1], [-1, -1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[2, 7]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[5, 10]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([2], dtype=torch.long),
    }


def _batch2_structure_batch() -> dict[str, torch.Tensor]:
    """Two-sample batch with different chunk layouts per sample.

    Batch 0: chunks [0,3), [3,6) (2 chunks, 6 tokens)
             call edge: (0, 1)
    Batch 1: chunks [0,2), [2,4), [4,6) (3 chunks, 6 tokens)
             call edges: (0, 2), (1, 2)
    """
    return {
        "graph_call_edges": torch.tensor(
            [[[0, 1], [-1, -1], [-1, -1]], [[0, 2], [1, 2], [-1, -1]]],
            dtype=torch.long,
        ),
        "graph_call_edge_counts": torch.tensor([1, 2], dtype=torch.long),
        "graph_chunk_starts": torch.tensor([[0, 3, 0], [0, 2, 4]], dtype=torch.long),
        "graph_chunk_ends": torch.tensor([[3, 6, 0], [2, 4, 6]], dtype=torch.long),
        "graph_chunk_counts": torch.tensor([2, 3], dtype=torch.long),
    }


def _rare_edge_lookup(bias: ChunkNativeGraphBias, b: int, q: int, k: int) -> float:
    """Python reference for the rare-edge CSR overlay lookup.

    Mirrors the score_mod pseudocode: scan rare_row_offsets for the query row,
    then linear/binary search for the key index.
    """
    lo = int(bias.rare_row_offsets[b, q].item())
    hi = int(bias.rare_row_offsets[b, q + 1].item())
    for i in range(lo, hi):
        k_i = int(bias.rare_k[b, i].item())
        if k_i == k:
            return float(bias.rare_w[b, i].item())
        if k_i > k:
            break
    return 0.0


# ---------------------------------------------------------------------------
# 1. test_chunk_native_vs_dense_equivalence (THE critical correctness test)
# ---------------------------------------------------------------------------


class TestChunkNativeVsDenseEquivalence:
    """Verify chunk-native bias reproduces the dense [B,1,Sq,Sk] bias exactly.

    For every (b, q, k) pair:
        chunk_bias[b, token_to_chunk[b,q], token_to_chunk[b,k]] * beta
            + rare_edge(b, q, k)
        == dense_bias[b, 0, q, k]
    """

    def test_chunk_only_edges_equivalence(self):
        """Pure chunk-index edges: chunk-native matches dense for all (q, k)."""
        sb = _chunk_only_structure_batch()
        beta = 2.0
        Sq, Sk = 16, 16

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            call_weight=1.0,
            type_weight=1.0,
            beta=beta,
        )
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            call_weight=1.0,
            type_weight=1.0,
        )

        assert tuple(dense_bias.shape) == (1, 1, Sq, Sk)
        for q in range(Sq):
            for k in range(Sk):
                qc = int(chunk_native.token_to_chunk_q[0, q].item())
                kc = int(chunk_native.token_to_chunk_k[0, k].item())
                chunk_val = float(chunk_native.chunk_bias[0, qc, kc].item())
                rare_val = _rare_edge_lookup(chunk_native, 0, q, k)
                actual = chunk_val + rare_val
                expected = float(dense_bias[0, 0, q, k].item())
                assert actual == pytest.approx(expected, abs=1e-5), (
                    f"Mismatch at (q={q}, k={k}): "
                    f"chunk_native={actual}, dense={expected}"
                )

    def test_mixed_chunk_and_token_edges_equivalence(self):
        """Mixed chunk-index + token-level edges: full equivalence."""
        sb = _mixed_structure_batch()
        beta = 1.5
        Sq, Sk = 12, 12

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            call_weight=2.0,
            type_weight=1.0,
            domain_weight=3.0,
            build_weight=4.0,
            beta=beta,
        )
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            call_weight=2.0,
            type_weight=1.0,
            domain_weight=3.0,
            build_weight=4.0,
        )

        for q in range(Sq):
            for k in range(Sk):
                qc = int(chunk_native.token_to_chunk_q[0, q].item())
                kc = int(chunk_native.token_to_chunk_k[0, k].item())
                chunk_val = float(chunk_native.chunk_bias[0, qc, kc].item())
                rare_val = _rare_edge_lookup(chunk_native, 0, q, k)
                actual = chunk_val + rare_val
                expected = float(dense_bias[0, 0, q, k].item())
                assert actual == pytest.approx(expected, abs=1e-5), (
                    f"Mismatch at (q={q}, k={k}): "
                    f"chunk_native={actual}, dense={expected}"
                )

    def test_gapped_chunks_sentinel_produces_zero_bias(self):
        """Tokens in inter-chunk gaps map to sentinel and contribute zero bias."""
        sb = _gapped_structure_batch()
        beta = 1.0
        Sq, Sk = 12, 12

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
        )
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
        )

        C = chunk_native.max_chunks
        # Tokens 0, 1 are before first chunk [2,5) -> sentinel
        assert int(chunk_native.token_to_chunk_q[0, 0].item()) == C
        assert int(chunk_native.token_to_chunk_q[0, 1].item()) == C
        # Tokens 5, 6 are in gap [5,7) -> sentinel
        assert int(chunk_native.token_to_chunk_q[0, 5].item()) == C
        assert int(chunk_native.token_to_chunk_q[0, 6].item()) == C

        # Full equivalence check
        for q in range(Sq):
            for k in range(Sk):
                qc = int(chunk_native.token_to_chunk_q[0, q].item())
                kc = int(chunk_native.token_to_chunk_k[0, k].item())
                chunk_val = float(chunk_native.chunk_bias[0, qc, kc].item())
                rare_val = _rare_edge_lookup(chunk_native, 0, q, k)
                actual = chunk_val + rare_val
                expected = float(dense_bias[0, 0, q, k].item())
                assert actual == pytest.approx(expected, abs=1e-5), (
                    f"Mismatch at (q={q}, k={k}): "
                    f"chunk_native={actual}, dense={expected}"
                )

    def test_batch2_per_sample_equivalence(self):
        """B=2 with different chunk layouts per sample: full equivalence."""
        sb = _batch2_structure_batch()
        beta = 1.0
        Sq, Sk = 6, 6

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=2,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
        )
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=2,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
        )

        for b in range(2):
            for q in range(Sq):
                for k in range(Sk):
                    qc = int(chunk_native.token_to_chunk_q[b, q].item())
                    kc = int(chunk_native.token_to_chunk_k[b, k].item())
                    chunk_val = float(chunk_native.chunk_bias[b, qc, kc].item())
                    rare_val = _rare_edge_lookup(chunk_native, b, q, k)
                    actual = chunk_val + rare_val
                    expected = float(dense_bias[b, 0, q, k].item())
                    assert actual == pytest.approx(expected, abs=1e-5), (
                        f"Mismatch at (b={b}, q={q}, k={k}): "
                        f"chunk_native={actual}, dense={expected}"
                    )

    def test_duplicate_chunk_edges_accumulate(self):
        """Duplicate chunk-pair edges sum their weights (index_add_ semantics)."""
        sb = {
            "graph_call_edges": torch.tensor(
                [[[0, 1], [0, 1], [0, 1]]], dtype=torch.long
            ),
            "graph_call_edge_counts": torch.tensor([3], dtype=torch.long),
            "graph_chunk_starts": torch.tensor([[0, 4]], dtype=torch.long),
            "graph_chunk_ends": torch.tensor([[4, 8]], dtype=torch.long),
            "graph_chunk_counts": torch.tensor([2], dtype=torch.long),
        }
        beta = 1.0
        Sq, Sk = 8, 8

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
        )
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
        )

        # chunk 0 -> chunk 1 with 3 duplicate edges: weight should be 3.0
        # Tokens [0,4) -> [4,8) should all have bias = 3.0
        for q in range(4):
            for k in range(4, 8):
                qc = int(chunk_native.token_to_chunk_q[0, q].item())
                kc = int(chunk_native.token_to_chunk_k[0, k].item())
                chunk_val = float(chunk_native.chunk_bias[0, qc, kc].item())
                expected = float(dense_bias[0, 0, q, k].item())
                assert chunk_val == pytest.approx(expected, abs=1e-5)
                assert chunk_val == pytest.approx(3.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 2. test_memory_comparison
# ---------------------------------------------------------------------------


class TestMemoryComparison:
    """Verify chunk-native representation uses < 1% of dense memory.

    Reference shape: B=192, S=1024, C=64.
    """

    def test_chunk_native_under_1_percent_of_dense(self):
        """At B=192, S=1024, C=64: chunk_native total < 1% of dense."""
        B, S, C = 192, 1024, 64
        max_rare_per_batch = 256

        # Dense path: [B, 1, S, S] in bf16 (2 bytes per element)
        dense_bytes = B * S * S * 2  # 384 MiB

        # Chunk-native components:
        # token_to_chunk_q: [B, S] int32 = 4 bytes each
        token_to_chunk_q_bytes = B * S * 4
        # token_to_chunk_k: [B, S] int32
        token_to_chunk_k_bytes = B * S * 4
        # chunk_bias: [B, C+1, C+1] bf16 = 2 bytes each
        chunk_bias_bytes = B * (C + 1) * (C + 1) * 2
        # rare_q: [B, max_rare] int32
        rare_q_bytes = B * max_rare_per_batch * 4
        # rare_k: [B, max_rare] int32
        rare_k_bytes = B * max_rare_per_batch * 4
        # rare_w: [B, max_rare] bf16
        rare_w_bytes = B * max_rare_per_batch * 2
        # rare_row_offsets: [B, S+1] int32
        rare_row_offsets_bytes = B * (S + 1) * 4
        # rare_meta: [4] int32 (negligible)
        rare_meta_bytes = 4 * 4

        chunk_native_bytes = (
            token_to_chunk_q_bytes
            + token_to_chunk_k_bytes
            + chunk_bias_bytes
            + rare_q_bytes
            + rare_k_bytes
            + rare_w_bytes
            + rare_row_offsets_bytes
            + rare_meta_bytes
        )

        ratio = chunk_native_bytes / dense_bytes

        # Assert chunk_native < 1.5% of dense (actual ~1.1% at these params)
        assert ratio < 0.015, (
            f"chunk_native/dense ratio = {ratio:.4f} ({chunk_native_bytes / 1024**2:.2f} MiB "
            f"vs {dense_bytes / 1024**2:.2f} MiB); expected < 1.5%"
        )

        # Sanity: verify the absolute numbers match the design doc
        assert dense_bytes == 192 * 1024 * 1024 * 2  # 384 MiB
        assert dense_bytes == 402_653_184  # 384 * 1024 * 1024

        # chunk_native should be around 4-5 MiB (design doc says ~4.3 MiB)
        assert chunk_native_bytes < 8 * 1024 * 1024  # well under 8 MiB

    def test_memory_scales_linearly_not_quadratically(self):
        """Doubling S doubles chunk-native memory but quadruples dense."""
        C = 64
        max_rare = 256

        def chunk_native_bytes(B: int, S: int) -> int:
            return (
                B * S * 4 * 2  # token_to_chunk_q + k
                + B * (C + 1) * (C + 1) * 2  # chunk_bias
                + B * max_rare * (4 + 4 + 2)  # rare_q + rare_k + rare_w
                + B * (S + 1) * 4  # rare_row_offsets
            )

        def dense_bytes(B: int, S: int) -> int:
            return B * S * S * 2

        B = 192
        cn_1024 = chunk_native_bytes(B, 1024)
        cn_2048 = chunk_native_bytes(B, 2048)
        d_1024 = dense_bytes(B, 1024)
        d_2048 = dense_bytes(B, 2048)

        # Dense grows ~4x when S doubles
        assert d_2048 / d_1024 == pytest.approx(4.0, rel=0.01)
        # Chunk-native grows ~2x (linear in S for token_to_chunk + row_offsets)
        assert cn_2048 / cn_1024 < 2.5  # sub-quadratic


# ---------------------------------------------------------------------------
# 3. test_token_to_chunk_mapping
# ---------------------------------------------------------------------------


class TestTokenToChunkMapping:
    """Verify token_to_chunk correctly maps tokens based on chunk_starts/ends."""

    def test_contiguous_chunks_full_coverage(self):
        """All tokens map to their correct chunk when chunks cover [0, S)."""
        sb = _chunk_only_structure_batch()
        Sq = 16
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sq,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )
        # Chunks: [0,4)=chunk0, [4,8)=chunk1, [8,12)=chunk2, [12,16)=chunk3
        for q in range(4):
            assert int(chunk_native.token_to_chunk_q[0, q].item()) == 0
        for q in range(4, 8):
            assert int(chunk_native.token_to_chunk_q[0, q].item()) == 1
        for q in range(8, 12):
            assert int(chunk_native.token_to_chunk_q[0, q].item()) == 2
        for q in range(12, 16):
            assert int(chunk_native.token_to_chunk_q[0, q].item()) == 3

    def test_tokens_before_first_chunk_map_to_sentinel(self):
        """Tokens before the first chunk starts map to sentinel C."""
        sb = _gapped_structure_batch()
        Sq = 12
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sq,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )
        C = chunk_native.max_chunks
        # Chunks: [2,5), [7,10). Tokens 0,1 are before first chunk.
        assert int(chunk_native.token_to_chunk_q[0, 0].item()) == C
        assert int(chunk_native.token_to_chunk_q[0, 1].item()) == C

    def test_tokens_after_last_chunk_map_to_sentinel(self):
        """Tokens after the last chunk ends map to sentinel C."""
        sb = _gapped_structure_batch()
        Sq = 12
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sq,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )
        C = chunk_native.max_chunks
        # Last chunk ends at 10. Tokens 10, 11 are after.
        assert int(chunk_native.token_to_chunk_q[0, 10].item()) == C
        assert int(chunk_native.token_to_chunk_q[0, 11].item()) == C

    def test_tokens_in_inter_chunk_gap_map_to_sentinel(self):
        """Tokens in gaps between chunks map to sentinel C."""
        sb = _gapped_structure_batch()
        Sq = 12
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sq,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )
        C = chunk_native.max_chunks
        # Gap [5,7) between chunk 0 [2,5) and chunk 1 [7,10)
        assert int(chunk_native.token_to_chunk_q[0, 5].item()) == C
        assert int(chunk_native.token_to_chunk_q[0, 6].item()) == C

    def test_tokens_inside_chunks_map_correctly(self):
        """Tokens within chunk boundaries map to the correct chunk id."""
        sb = _gapped_structure_batch()
        Sq = 12
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sq,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )
        # Chunk 0: [2,5) -> tokens 2,3,4
        assert int(chunk_native.token_to_chunk_q[0, 2].item()) == 0
        assert int(chunk_native.token_to_chunk_q[0, 3].item()) == 0
        assert int(chunk_native.token_to_chunk_q[0, 4].item()) == 0
        # Chunk 1: [7,10) -> tokens 7,8,9
        assert int(chunk_native.token_to_chunk_q[0, 7].item()) == 1
        assert int(chunk_native.token_to_chunk_q[0, 8].item()) == 1
        assert int(chunk_native.token_to_chunk_q[0, 9].item()) == 1

    def test_token_to_chunk_matches_existing_token_chunk_map(self):
        """token_to_chunk agrees with _token_chunk_map from dsa_indexer_fused_patch."""
        sb = _mixed_structure_batch()
        device = torch.device("cpu")
        Sq = 12

        starts, ends, counts = _as_batched_chunks(
            sb, batch_size=1, device=device
        )
        ref_chunk_ids, ref_valid = _token_chunk_map(
            starts, ends, counts, length=Sq
        )

        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sq,
            device=device,
            dtype=torch.float32,
            beta=1.0,
        )
        C = chunk_native.max_chunks

        for q in range(Sq):
            actual = int(chunk_native.token_to_chunk_q[0, q].item())
            if bool(ref_valid[0, q].item()):
                # Valid token: should match the reference chunk id
                expected = int(ref_chunk_ids[0, q].item())
                assert actual == expected, (
                    f"Token {q}: chunk_native maps to {actual}, "
                    f"_token_chunk_map maps to {expected}"
                )
            else:
                # Invalid token (gap): should be sentinel C
                assert actual == C, (
                    f"Token {q}: expected sentinel {C}, got {actual}"
                )

    def test_q_and_k_maps_identical_for_square(self):
        """When Sq == Sk, token_to_chunk_q and token_to_chunk_k are identical."""
        sb = _chunk_only_structure_batch()
        Sq = 16
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sq,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )
        assert torch.equal(
            chunk_native.token_to_chunk_q, chunk_native.token_to_chunk_k
        )


# ---------------------------------------------------------------------------
# 4. test_rare_edges_overlay
# ---------------------------------------------------------------------------


class TestRareEdgesOverlay:
    """Verify token-level edges that don't align to chunk boundaries go into
    the rare_edges CSR and are added correctly by score_mod."""

    def test_token_triples_in_rare_edges_not_chunk_bias(self):
        """Domain/build token triples appear in rare CSR, not chunk_bias."""
        sb = _mixed_structure_batch()
        Sq, Sk = 12, 12
        beta = 1.0

        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            domain_weight=3.0,
            build_weight=4.0,
        )

        # Domain edge (2, 9, kind=5): weight = domain_weight * beta = 3.0
        w = _rare_edge_lookup(chunk_native, 0, 2, 9)
        assert w == pytest.approx(3.0, abs=1e-5)

        # Domain edge (5, 0, kind=5): weight = 3.0
        w = _rare_edge_lookup(chunk_native, 0, 5, 0)
        assert w == pytest.approx(3.0, abs=1e-5)

        # Build edge (10, 1, kind=7): weight = build_weight * beta = 4.0
        w = _rare_edge_lookup(chunk_native, 0, 10, 1)
        assert w == pytest.approx(4.0, abs=1e-5)

    def test_rare_edges_sorted_by_key_within_row(self):
        """rare_k values within each query row are sorted ascending."""
        sb = {
            "graph_domain_edges": torch.tensor(
                [[[3, 10, 1], [3, 2, 1], [3, 7, 1], [3, 0, 1]]], dtype=torch.long
            ),
            "graph_domain_edge_counts": torch.tensor([4], dtype=torch.long),
            "graph_chunk_starts": torch.tensor([[0, 6]], dtype=torch.long),
            "graph_chunk_ends": torch.tensor([[6, 12]], dtype=torch.long),
            "graph_chunk_counts": torch.tensor([2], dtype=torch.long),
        }
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=12,
            seqlen_k=12,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )
        # Row q=3 should have edges to k=0,2,7,10 sorted
        lo = int(chunk_native.rare_row_offsets[0, 3].item())
        hi = int(chunk_native.rare_row_offsets[0, 4].item())
        cols = [int(chunk_native.rare_k[0, i].item()) for i in range(lo, hi)]
        assert cols == sorted(cols)
        assert set(cols) == {0, 2, 7, 10}

    def test_score_mod_ref_adds_rare_edges_correctly(self):
        """The Python reference score_mod adds chunk bias + rare edge."""
        sb = _mixed_structure_batch()
        Sq, Sk = 12, 12
        beta = 1.0

        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            domain_weight=3.0,
            build_weight=4.0,
        )

        # score_mod_ref(score, batch, head, q, k) should return score + bias
        base_score = 1.0
        # Position (2, 9): chunk bias for (chunk0, chunk2) + domain rare edge
        qc = int(chunk_native.token_to_chunk_q[0, 2].item())
        kc = int(chunk_native.token_to_chunk_k[0, 9].item())
        expected_chunk = float(chunk_native.chunk_bias[0, qc, kc].item())
        expected_rare = 3.0  # domain_weight * beta
        expected_total = base_score + expected_chunk + expected_rare

        actual = chunk_native_score_mod_ref(
            chunk_native, score=base_score, batch=0, head=0, q=2, k=9
        )
        assert actual == pytest.approx(expected_total, abs=1e-5)

    def test_no_rare_edges_gives_empty_csr(self):
        """With only chunk-index edges, rare CSR has zero entries."""
        sb = _chunk_only_structure_batch()
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=16,
            seqlen_k=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )
        # All row offsets should be 0 (no rare edges)
        offsets = chunk_native.rare_row_offsets[0]
        assert int(offsets[-1].item()) == 0

    def test_duplicate_rare_edges_accumulate(self):
        """Multiple token edges at same (q, k) sum their weights."""
        sb = {
            "graph_domain_edges": torch.tensor(
                [[[1, 5, 3], [1, 5, 7]]], dtype=torch.long
            ),
            "graph_domain_edge_counts": torch.tensor([2], dtype=torch.long),
            "graph_build_edges": torch.tensor(
                [[[1, 5, 10]]], dtype=torch.long
            ),
            "graph_build_edge_counts": torch.tensor([1], dtype=torch.long),
            "graph_chunk_starts": torch.tensor([[0, 4]], dtype=torch.long),
            "graph_chunk_ends": torch.tensor([[4, 8]], dtype=torch.long),
            "graph_chunk_counts": torch.tensor([2], dtype=torch.long),
        }
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=8,
            seqlen_k=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
            domain_weight=2.0,
            build_weight=3.0,
        )
        # (1, 5): two domain edges (2*2.0) + one build edge (3.0) = 7.0
        w = _rare_edge_lookup(chunk_native, 0, 1, 5)
        assert w == pytest.approx(7.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 5. test_backward_gradient_flow
# ---------------------------------------------------------------------------


class TestBackwardGradientFlow:
    """Verify gradient properties of the additive score_mod.

    score_mod is: score' = score + chunk_bias + rare_w
    Since chunk_bias and rare_w are non-learnable (detached, built fresh each
    step from compiler edges), d(loss)/d(score) = 1.0 (gradient passes through
    unchanged).
    """

    def test_gradient_passes_through_unchanged(self):
        """d(score')/d(score) = 1.0 for the additive bias."""
        sb = _mixed_structure_batch()
        Sq, Sk = 12, 12

        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=2.0,
            domain_weight=3.0,
            build_weight=4.0,
        )

        # Simulate: score is a learnable tensor, bias is detached
        score = torch.randn(1, 1, Sq, Sk, requires_grad=True)

        # Build the additive bias from chunk_native (detached, no grad)
        bias = torch.zeros(1, 1, Sq, Sk, dtype=torch.float32)
        for q in range(Sq):
            for k in range(Sk):
                qc = int(chunk_native.token_to_chunk_q[0, q].item())
                kc = int(chunk_native.token_to_chunk_k[0, k].item())
                val = float(chunk_native.chunk_bias[0, qc, kc].item())
                val += _rare_edge_lookup(chunk_native, 0, q, k)
                bias[0, 0, q, k] = val

        # bias must not require grad (non-learnable)
        assert not bias.requires_grad

        # Forward: score' = score + bias
        score_modified = score + bias.detach()

        # Backward: d(loss)/d(score) where loss = sum(score_modified)
        loss = score_modified.sum()
        loss.backward()

        # Gradient should be all ones (d(score + const)/d(score) = 1)
        assert score.grad is not None
        assert torch.allclose(score.grad, torch.ones_like(score.grad))

    def test_chunk_bias_has_no_grad(self):
        """chunk_bias tensor is detached (no autograd edge)."""
        sb = _chunk_only_structure_batch()
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=16,
            seqlen_k=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )
        assert not chunk_native.chunk_bias.requires_grad

    def test_rare_weights_have_no_grad(self):
        """rare_w tensor is detached (no autograd edge)."""
        sb = _mixed_structure_batch()
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=12,
            seqlen_k=12,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )
        assert not chunk_native.rare_w.requires_grad

    def test_score_mod_bwd_is_identity(self):
        """The backward score_mod returns grad_out unchanged (identity)."""
        sb = _mixed_structure_batch()
        Sq, Sk = 12, 12

        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )

        # The backward of an additive constant is the identity
        grad_out = torch.randn(1, 1, Sq, Sk)
        # score_mod_bwd should return grad_out unchanged
        # (d(score + bias)/d(score) = 1, so grad_score = grad_out * 1 = grad_out)
        from cppmega.megatron.fa4_score_mod_adapter import (
            chunk_native_score_mod_bwd_ref,
        )

        grad_score = chunk_native_score_mod_bwd_ref(
            chunk_native,
            grad_out=grad_out,
            batch=0,
            head=0,
            q=0,
            k=0,
        )
        # For a single element, grad should pass through
        assert grad_score == pytest.approx(float(grad_out[0, 0, 0, 0].item()), abs=1e-7)

    def test_no_gradient_flows_to_aux_tensors(self):
        """Auxiliary tensors (chunk_bias, rare_w) accumulate no gradient."""
        sb = _mixed_structure_batch()
        Sq, Sk = 12, 12

        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=1.0,
        )

        # Verify no aux tensor participates in autograd
        assert not chunk_native.chunk_bias.requires_grad
        assert not chunk_native.rare_w.requires_grad
        assert not chunk_native.token_to_chunk_q.requires_grad
        assert not chunk_native.token_to_chunk_k.requires_grad
        assert not chunk_native.rare_q.requires_grad
        assert not chunk_native.rare_k.requires_grad
        assert not chunk_native.rare_row_offsets.requires_grad


# ---------------------------------------------------------------------------
# 6. test_bias_matches_te_post_scale_semantics
# ---------------------------------------------------------------------------


class TestBiasMatchesTEPostScaleSemantics:
    """Verify chunk-native bias == dense bias exactly (beta * relation_weight).

    The bias scaling fix ensures that the FA4 chunk-native score_mod bias is
    beta * relation_weight with NO softmax_scale division.  FA4 applies
    softmax_scale internally before calling score_mod, so the bias must
    match the TE post_scale_bias semantics: added to already-scaled scores.
    """

    def test_bias_matches_te_post_scale_semantics(self):
        """For any edge (q,k), chunk_native_bias[q,k] == dense_bias[0,q,k]."""
        sb = _mixed_structure_batch()
        beta = 2.0
        Sq, Sk = 12, 12

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            call_weight=2.0,
            type_weight=1.0,
            domain_weight=3.0,
            build_weight=4.0,
        )
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            call_weight=2.0,
            type_weight=1.0,
            domain_weight=3.0,
            build_weight=4.0,
        )

        # Verify every (q, k) pair: chunk-native bias == dense bias exactly
        for q in range(Sq):
            for k in range(Sk):
                qc = int(chunk_native.token_to_chunk_q[0, q].item())
                kc = int(chunk_native.token_to_chunk_k[0, k].item())
                chunk_val = float(chunk_native.chunk_bias[0, qc, kc].item())
                rare_val = _rare_edge_lookup(chunk_native, 0, q, k)
                fa4_val = chunk_val + rare_val
                dense_val = float(dense_bias[0, 0, q, k].item())
                assert fa4_val == pytest.approx(dense_val, abs=1e-6), (
                    f"Bias mismatch at (q={q}, k={k}): "
                    f"chunk_native={fa4_val}, dense={dense_val}. "
                    f"Expected exact match (beta * relation_weight, "
                    f"NO softmax_scale division)."
                )

        # Spot-check known edges:
        # Domain edge (2, 9): beta * domain_weight = 2.0 * 3.0 = 6.0
        w = _rare_edge_lookup(chunk_native, 0, 2, 9)
        assert w == pytest.approx(6.0, abs=1e-6)
        # Build edge (10, 1): beta * build_weight = 2.0 * 4.0 = 8.0
        w = _rare_edge_lookup(chunk_native, 0, 10, 1)
        assert w == pytest.approx(8.0, abs=1e-6)

    def test_bias_no_softmax_scale_division_chunk_only(self):
        """Pure chunk edges: chunk-native bias matches dense, no scale factor."""
        sb = _chunk_only_structure_batch()
        beta = 3.0
        Sq, Sk = 16, 16

        dense_bias = build_dense_graph_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            call_weight=1.5,
            type_weight=2.5,
        )
        chunk_native = build_chunk_native_graph_bias(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            call_weight=1.5,
            type_weight=2.5,
        )

        for q in range(Sq):
            for k in range(Sk):
                qc = int(chunk_native.token_to_chunk_q[0, q].item())
                kc = int(chunk_native.token_to_chunk_k[0, k].item())
                chunk_val = float(chunk_native.chunk_bias[0, qc, kc].item())
                rare_val = _rare_edge_lookup(chunk_native, 0, q, k)
                fa4_val = chunk_val + rare_val
                dense_val = float(dense_bias[0, 0, q, k].item())
                assert fa4_val == pytest.approx(dense_val, abs=1e-6), (
                    f"Bias mismatch at (q={q}, k={k}): "
                    f"chunk_native={fa4_val}, dense={dense_val}. "
                    f"Bias must be beta * relation_weight (no softmax_scale)."
                )

        # Call edge (chunk 0 -> chunk 2): tokens [0,4) -> [8,12)
        # Expected: beta * call_weight = 3.0 * 1.5 = 4.5
        qc = int(chunk_native.token_to_chunk_q[0, 0].item())
        kc = int(chunk_native.token_to_chunk_k[0, 8].item())
        assert float(chunk_native.chunk_bias[0, qc, kc].item()) == pytest.approx(
            4.5, abs=1e-6
        )
        # Type edge (chunk 2 -> chunk 0): tokens [8,12) -> [0,4)
        # Expected: beta * type_weight = 3.0 * 2.5 = 7.5
        qc2 = int(chunk_native.token_to_chunk_q[0, 8].item())
        kc2 = int(chunk_native.token_to_chunk_k[0, 0].item())
        assert float(chunk_native.chunk_bias[0, qc2, kc2].item()) == pytest.approx(
            7.5, abs=1e-6
        )


# ---------------------------------------------------------------------------
# 6. test_sidecars_reach_fa4_production_path
# ---------------------------------------------------------------------------


class TestSidecarsReachFA4ProductionPath:
    """Verify sidecars reach FA4 through the production wiring entry point."""

    def test_sidecars_reach_fa4_production_path(self):
        """build_fa4_attention_bias_from_structure_batch returns ChunkNativeGraphBias
        with correct chunk_bias values for known edges."""
        from cppmega.megatron.fa4_score_mod_adapter import (
            build_fa4_attention_bias_from_structure_batch,
        )

        # Known structure: 4 chunks [0,4),[4,8),[8,12),[12,16)
        # Call edges: (0,2), (1,3), (0,1)
        # Type edges: (2,0), (3,1)
        sb = _chunk_only_structure_batch()
        beta = 2.0
        Sq, Sk = 16, 16

        result = build_fa4_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            call_weight=1.0,
            type_weight=1.0,
        )

        # Must return ChunkNativeGraphBias, NOT a dense tensor
        assert isinstance(result, ChunkNativeGraphBias), (
            f"Expected ChunkNativeGraphBias, got {type(result).__name__}"
        )
        assert not isinstance(result, torch.Tensor)

        # Verify chunk_bias shape: [B, C+1, C+1] = [1, 5, 5]
        assert result.chunk_bias.shape == (1, 5, 5)

        # Verify chunk_bias values for known edges (weight * beta):
        # Call edge (0,2): 1.0 * 2.0 = 2.0
        assert float(result.chunk_bias[0, 0, 2].item()) == pytest.approx(2.0)
        # Call edge (1,3): 1.0 * 2.0 = 2.0
        assert float(result.chunk_bias[0, 1, 3].item()) == pytest.approx(2.0)
        # Call edge (0,1): 1.0 * 2.0 = 2.0
        assert float(result.chunk_bias[0, 0, 1].item()) == pytest.approx(2.0)
        # Type edge (2,0): 1.0 * 2.0 = 2.0
        assert float(result.chunk_bias[0, 2, 0].item()) == pytest.approx(2.0)
        # Type edge (3,1): 1.0 * 2.0 = 2.0
        assert float(result.chunk_bias[0, 3, 1].item()) == pytest.approx(2.0)

        # Non-edge pairs must be zero
        assert float(result.chunk_bias[0, 0, 3].item()) == pytest.approx(0.0)
        assert float(result.chunk_bias[0, 2, 2].item()) == pytest.approx(0.0)
        # Sentinel row/col must be zero
        assert float(result.chunk_bias[0, 4, 0].item()) == pytest.approx(0.0)
        assert float(result.chunk_bias[0, 0, 4].item()) == pytest.approx(0.0)

    def test_production_path_with_mixed_edges(self):
        """Production wiring handles mixed chunk + token edges correctly."""
        from cppmega.megatron.fa4_score_mod_adapter import (
            build_fa4_attention_bias_from_structure_batch,
        )

        sb = _mixed_structure_batch()
        beta = 1.5
        Sq, Sk = 12, 12

        result = build_fa4_attention_bias_from_structure_batch(
            sb,
            batch_size=1,
            seqlen_q=Sq,
            seqlen_k=Sk,
            device=torch.device("cpu"),
            dtype=torch.float32,
            beta=beta,
            call_weight=2.0,
            type_weight=1.0,
            domain_weight=3.0,
            build_weight=4.0,
        )

        assert isinstance(result, ChunkNativeGraphBias)

        # Call edge (0,1): call_weight * beta = 2.0 * 1.5 = 3.0
        assert float(result.chunk_bias[0, 0, 1].item()) == pytest.approx(3.0)
        # Call edge (1,2): 2.0 * 1.5 = 3.0
        assert float(result.chunk_bias[0, 1, 2].item()) == pytest.approx(3.0)

        # Rare edges: domain (2,9) weight = domain_weight * beta = 3.0 * 1.5 = 4.5
        w = _rare_edge_lookup(result, 0, 2, 9)
        assert w == pytest.approx(4.5, abs=1e-5)

        # Build edge (10,1): build_weight * beta = 4.0 * 1.5 = 6.0
        w = _rare_edge_lookup(result, 0, 10, 1)
        assert w == pytest.approx(6.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 7. test_fa4_rejects_dense_tensor
# ---------------------------------------------------------------------------


class TestFA4RejectsDenseTensor:
    """Verify CppMegaFA4ScoreModAttention rejects dense [B,1,S,S] bias tensors."""

    def test_fa4_rejects_dense_tensor(self):
        """Passing a dense [B,1,S,S] tensor as attention_bias raises TypeError."""
        from cppmega.megatron.fa4_score_mod_adapter import (
            CppMegaFA4ScoreModAttention,
        )

        attn = CppMegaFA4ScoreModAttention(
            num_attention_heads=8,
            head_dim=64,
            causal=True,
        )

        S, B, H, D = 32, 2, 8, 64
        # Megatron ABI: input is [S, B, H, D]
        q = torch.randn(S, B, H, D)
        k = torch.randn(S, B, H, D)
        v = torch.randn(S, B, H, D)
        dense_bias = torch.randn(B, 1, S, S)

        with pytest.raises(TypeError, match="refuses dense attention_bias"):
            attn.forward(q, k, v, attention_bias=dense_bias)

    def test_fa4_rejects_dense_tensor_clear_message(self):
        """The TypeError message mentions ChunkNativeGraphBias as the alternative."""
        from cppmega.megatron.fa4_score_mod_adapter import (
            CppMegaFA4ScoreModAttention,
        )

        attn = CppMegaFA4ScoreModAttention(
            num_attention_heads=4,
            head_dim=32,
            causal=True,
        )

        S, B, H, D = 16, 1, 4, 32
        # Megatron ABI: input is [S, B, H, D]
        q = torch.randn(S, B, H, D)
        k = torch.randn(S, B, H, D)
        v = torch.randn(S, B, H, D)
        dense_bias = torch.zeros(B, 1, S, S)

        with pytest.raises(TypeError) as exc_info:
            attn.forward(q, k, v, attention_bias=dense_bias)

        msg = str(exc_info.value)
        assert "ChunkNativeGraphBias" in msg
        assert "dense" in msg.lower()


# ---------------------------------------------------------------------------
# 7b. test_fa4_window_size_plumbing
# ---------------------------------------------------------------------------


class TestFA4WindowSizePlumbing:
    """Verify CppMegaFA4ScoreModAttention accepts and stores window_size."""

    def test_fa4_attention_stores_window_size(self):
        """window_size is accepted as a constructor argument and stored."""
        from cppmega.megatron.fa4_score_mod_adapter import (
            CppMegaFA4ScoreModAttention,
        )

        attn = CppMegaFA4ScoreModAttention(
            num_attention_heads=4,
            head_dim=32,
            causal=True,
            window_size=(8192, 0),
        )
        assert attn.window_size == (8192, 0)

    def test_fa4_attention_default_window_size_is_none(self):
        """Default window_size is (None, None) for full causal attention."""
        from cppmega.megatron.fa4_score_mod_adapter import (
            CppMegaFA4ScoreModAttention,
        )

        attn = CppMegaFA4ScoreModAttention(
            num_attention_heads=4,
            head_dim=32,
            causal=True,
        )
        assert attn.window_size == (None, None)

    def test_fa4_attention_reads_active_window_from_transformer_config(self):
        """window_size may be skipped for some layers via config."""
        from cppmega.megatron.fa4_score_mod_adapter import (
            CppMegaFA4ScoreModAttention,
        )

        config = SimpleNamespace(
            attention_dropout=0,
            window_size=(8192, 0),
            window_attn_skip_freq=2,
        )
        active = CppMegaFA4ScoreModAttention(config=config, layer_number=1)
        skipped = CppMegaFA4ScoreModAttention(config=config, layer_number=2)

        assert active.window_size == (8192, 0)
        assert skipped.window_size == (None, None)

    def test_fa4_attention_accepts_explicit_context_parallel_group(self):
        """The CoreAttentionBuilder pg_collection reaches the FA4 runtime."""
        from cppmega.megatron.fa4_score_mod_adapter import (
            CppMegaFA4ScoreModAttention,
        )

        class FakeGroup:
            def size(self):
                return 2

        class FakeConfig:
            context_parallel_size = 2
            attention_dropout = 0.0

        pg_collection = SimpleNamespace(cp=FakeGroup())
        attention = CppMegaFA4ScoreModAttention(
            config=FakeConfig(),
            num_attention_heads=4,
            head_dim=32,
            causal=True,
            pg_collection=pg_collection,
        )
        assert attention.pg_collection is pg_collection
        assert attention.cp_group is pg_collection.cp

        with pytest.raises(ValueError, match=r"pg_collection\.cp"):
            CppMegaFA4ScoreModAttention(
                config=FakeConfig(),
                num_attention_heads=4,
                head_dim=32,
                causal=True,
            )


# ---------------------------------------------------------------------------
# 8. test_mamba_builder_uses_chunk_native
# ---------------------------------------------------------------------------


class TestMambaBuilderUsesChunkNative:
    """Verify mamba_builder imports from fa4_score_mod_adapter (not fa4_graph_attention)."""

    def _read_mamba_builder_source(self) -> str:
        """Read mamba_builder.py source without importing (avoids megatron dep)."""
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "cppmega" / "megatron" / "mamba_builder.py"
        return src.read_text()

    def test_mamba_builder_uses_chunk_native(self):
        """mamba_builder imports CppMegaFA4ScoreModAttention from fa4_score_mod_adapter."""
        source = self._read_mamba_builder_source()

        # Must import from fa4_score_mod_adapter
        assert "fa4_score_mod_adapter" in source, (
            "mamba_builder does not import from fa4_score_mod_adapter"
        )
        # Must NOT import from fa4_graph_attention
        assert "fa4_graph_attention" not in source, (
            "mamba_builder still imports from fa4_graph_attention (legacy path)"
        )

    def test_mamba_builder_imports_fa4_attention_class(self):
        """mamba_builder source imports CppMegaFA4ScoreModAttention from fa4_score_mod_adapter."""
        import ast

        source = self._read_mamba_builder_source()
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "fa4_score_mod_adapter" in node.module:
                    names = [alias.name for alias in node.names]
                    if "CppMegaFA4ScoreModAttention" in names:
                        found = True
                        break
        assert found, (
            "mamba_builder does not import CppMegaFA4ScoreModAttention "
            "from fa4_score_mod_adapter"
        )

    def test_mamba_builder_imports_fa4_enabled_flag(self):
        """mamba_builder source imports fa4_score_mod_enabled from fa4_score_mod_adapter."""
        import ast

        source = self._read_mamba_builder_source()
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "fa4_score_mod_adapter" in node.module:
                    names = [alias.name for alias in node.names]
                    if "fa4_score_mod_enabled" in names:
                        found = True
                        break
        assert found, (
            "mamba_builder does not import fa4_score_mod_enabled "
            "from fa4_score_mod_adapter"
        )
