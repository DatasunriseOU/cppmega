"""Correctness test for the DSA indexer fused per-head accumulation patch.

Verifies :func:`compute_index_scores_fused_bf16` matches the upstream
Megatron BF16 ``_compute_index_scores`` einsum implementation to within
FP32 associative-reorder tolerance.

Run on GB10 (or any CUDA GPU) with a small shape; no Megatron checkout
required — this test inlines the upstream reference.
"""

from __future__ import annotations

import pytest
import torch

from cppmega.megatron.dsa_indexer_fused_patch import (
    build_graph_route_bias_from_structure_batch,
    compute_index_scores_fused_bf16,
)


def _upstream_reference(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    use_relu: bool = True,
) -> torch.Tensor:
    """Byte-identical clone of upstream Megatron ``_compute_index_scores``."""
    index_scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    if use_relu:
        index_scores = torch.relu(index_scores)
    index_scores = index_scores * weights.unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)
    return index_scores.transpose(0, 1)  # [b, sq, sk]


def test_fused_matches_reference_bf16_relu():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sq, sk, b, h, d = 128, 128, 2, 8, 64

    q = torch.randn(sq, b, h, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(sk, b, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(sq, b, h, dtype=torch.bfloat16, device=device)

    ref = _upstream_reference(q, w, k, use_relu=True)
    fused = compute_index_scores_fused_bf16(q, w, k, use_relu=True)

    assert ref.shape == fused.shape == (b, sq, sk)
    assert ref.dtype == fused.dtype == torch.float32

    # FP32 associative reorder tolerance: per-head accum vs fused-sum over h.
    abs_err = (ref - fused).abs().max().item()
    ref_abs = ref.abs().max().item()
    rel_err = abs_err / max(ref_abs, 1e-6)
    print(f"relu=True abs_err={abs_err:.3e} rel_err={rel_err:.3e}")
    assert rel_err < 1e-3, f"rel_err {rel_err} too high"


def test_fused_matches_reference_bf16_no_relu():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sq, sk, b, h, d = 256, 256, 1, 16, 64

    q = torch.randn(sq, b, h, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(sk, b, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(sq, b, h, dtype=torch.bfloat16, device=device)

    ref = _upstream_reference(q, w, k, use_relu=False)
    fused = compute_index_scores_fused_bf16(q, w, k, use_relu=False)

    abs_err = (ref - fused).abs().max().item()
    rel_err = abs_err / max(ref.abs().max().item(), 1e-6)
    print(f"relu=False abs_err={abs_err:.3e} rel_err={rel_err:.3e}")
    assert rel_err < 1e-3, f"rel_err {rel_err} too high"


def test_graph_route_bias_from_structure_batch_scatter_edges():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structure_batch = {
        "graph_call_edges": torch.tensor(
            [[[0, 2], [1, 3], [-1, -1]]], dtype=torch.long
        ),
        "graph_call_edge_counts": torch.tensor([2], dtype=torch.long),
        "graph_type_edges": torch.tensor([[[2, 1], [-1, -1]]], dtype=torch.long),
        "graph_type_edge_counts": torch.tensor([1], dtype=torch.long),
    }

    bias = build_graph_route_bias_from_structure_batch(
        structure_batch,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device=device,
        call_weight=2.0,
        type_weight=3.0,
    )

    assert tuple(bias.shape) == (1, 4, 4)
    assert bias[0, 0, 2].item() == 2.0
    assert bias[0, 1, 3].item() == 2.0
    assert bias[0, 2, 1].item() == 3.0
    assert bias.sum().item() == 7.0


def test_graph_route_bias_from_structure_batch_scatter_domain_edge_triples():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structure_batch = {
        "graph_domain_edges": torch.tensor(
            [[[0, 2, 20], [-1, -1, -1]]], dtype=torch.long
        ),
        "graph_domain_edge_counts": torch.tensor([1], dtype=torch.long),
        "graph_diagnostic_edges": torch.tensor(
            [[[1, 3, 60], [-1, -1, -1]]], dtype=torch.long
        ),
        "graph_diagnostic_edge_counts": torch.tensor([1], dtype=torch.long),
    }

    bias = build_graph_route_bias_from_structure_batch(
        structure_batch,
        batch_size=1,
        seqlen_q=4,
        seqlen_k=4,
        device=device,
        domain_weight=2.0,
        diagnostic_weight=5.0,
    )

    assert bias[0, 0, 2].item() == 2.0
    assert bias[0, 1, 3].item() == 5.0
    assert bias.sum().item() == 7.0


def test_fused_scores_add_graph_route_bias_before_topk():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sq = sk = 4
    b, h, d = 1, 2, 8
    q = torch.zeros(sq, b, h, d, dtype=torch.bfloat16, device=device)
    k = torch.zeros(sk, b, d, dtype=torch.bfloat16, device=device)
    w = torch.ones(sq, b, h, dtype=torch.bfloat16, device=device)
    graph_bias = torch.zeros((b, sq, sk), dtype=torch.float32, device=device)
    graph_bias[0, 1, 3] = 1.0
    graph_bias[0, 2, 0] = 0.5

    scores = compute_index_scores_fused_bf16(
        q,
        w,
        k,
        use_relu=True,
        graph_bias=graph_bias,
        graph_beta=10.0,
    )

    assert scores[0, 1, 3].item() == 10.0
    assert scores[0, 2, 0].item() == 5.0
    assert scores[0, 0, 0].item() == 0.0


def test_scatter_edges_vectorized_matches_reference_loop():
    from cppmega.megatron.dsa_indexer_fused_patch import _scatter_edges_

    torch.manual_seed(0)
    B, sq, sk, maxE = 8, 32, 24, 6
    edges = torch.full((B, maxE, 2), -1, dtype=torch.int32)
    counts = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        n = int(torch.randint(0, maxE + 1, (1,)))
        counts[b] = n
        if n:
            flat = torch.randperm(sq * sk)[:n]  # unique (src,dst) per sample
            edges[b, :n, 0] = (flat // sk).to(torch.int32)
            edges[b, :n, 1] = (flat % sk).to(torch.int32)

    bias_vec = torch.zeros(B, sq, sk)
    _scatter_edges_(bias_vec, edges, counts, weight=2.0, sq=sq, sk=sk, require_kind=False)

    bias_ref = torch.zeros(B, sq, sk)  # naive per-sample reference (old semantics)
    for b in range(B):
        for e in range(int(counts[b])):
            bias_ref[b, int(edges[b, e, 0]), int(edges[b, e, 1])] += 2.0

    assert torch.equal(bias_vec, bias_ref)

    # shared single-row edges must broadcast across the batch identically
    shared = torch.tensor([[[0, 1], [2, 3], [-1, -1]]], dtype=torch.int32)
    bias_b = torch.zeros(4, sq, sk)
    _scatter_edges_(bias_b, shared, torch.tensor([2]), weight=1.0, sq=sq, sk=sk, require_kind=False)
    assert bias_b[:, 0, 1].tolist() == [1.0] * 4 and bias_b[:, 2, 3].tolist() == [1.0] * 4


def test_scatter_edges_raises_on_count_out_of_range():
    from cppmega.megatron.dsa_indexer_fused_patch import _scatter_edges_

    edges = torch.tensor([[[0, 1]]], dtype=torch.long)  # max_edges = 1
    with pytest.raises(ValueError, match="out of range"):
        _scatter_edges_(torch.zeros(1, 4, 4), edges, torch.tensor([5]), weight=1.0, sq=4, sk=4, require_kind=False)
    with pytest.raises(ValueError, match="out of range"):
        _scatter_edges_(torch.zeros(1, 4, 4), edges, torch.tensor([-1]), weight=1.0, sq=4, sk=4, require_kind=False)


def test_graph_route_bias_raises_on_out_of_range_edge():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structure_batch = {
        # count says 1 real edge, but dst=9 is out of range for seqlen 4.
        "graph_call_edges": torch.tensor([[[0, 9], [-1, -1]]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor([1], dtype=torch.long),
    }
    with pytest.raises(ValueError, match="out of range"):
        build_graph_route_bias_from_structure_batch(
            structure_batch,
            batch_size=1,
            seqlen_q=4,
            seqlen_k=4,
            device=device,
            call_weight=2.0,
        )


def test_graph_route_bias_requires_structure_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(RuntimeError, match="no current cppmega structure batch"):
        build_graph_route_bias_from_structure_batch(
            None,
            batch_size=1,
            seqlen_q=4,
            seqlen_k=4,
            device=device,
        )


def test_fused_nam56r_shape():
    """Test a realistic NAM56R indexer shape: h=32 d=64 sq=sk=4096 b=1."""
    if not torch.cuda.is_available():
        return
    torch.manual_seed(2)
    device = torch.device("cuda")
    sq, sk, b, h, d = 4096, 4096, 1, 32, 64

    q = torch.randn(sq, b, h, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(sk, b, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(sq, b, h, dtype=torch.bfloat16, device=device)

    ref = _upstream_reference(q, w, k, use_relu=True)
    fused = compute_index_scores_fused_bf16(q, w, k, use_relu=True)

    abs_err = (ref - fused).abs().max().item()
    rel_err = abs_err / max(ref.abs().max().item(), 1e-6)
    print(f"nam56r-shape abs_err={abs_err:.3e} rel_err={rel_err:.3e}")
    assert rel_err < 5e-3, f"rel_err {rel_err} too high"


if __name__ == "__main__":
    test_fused_matches_reference_bf16_relu()
    test_fused_matches_reference_bf16_no_relu()
    test_fused_nam56r_shape()
    print("All parity tests passed.")
