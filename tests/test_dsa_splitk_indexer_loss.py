"""Unit tests for :mod:`cppmega.megatron.dsa_splitk_indexer_loss`.

Tests compare the split-K Triton kernel output against a pure-PyTorch
reference implementation of ``compute_dsa_indexer_loss`` (the function it
replaces).  The reference is self-contained so the test suite runs
without Megatron installed.

The Triton kernel requires CUDA, so all tests are gated on
``torch.cuda.is_available()``.
"""

from __future__ import annotations

import math

import pytest
import torch

_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton split-K kernel requires CUDA",
)


# ---------------------------------------------------------------------------
# Pure-PyTorch reference: mirrors Megatron's compute_dsa_indexer_loss
# ---------------------------------------------------------------------------

def _compute_dsa_indexer_loss_reference(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
) -> torch.Tensor:
    """Pure-PyTorch reference for the DSA indexer KL loss.

    Shapes:
        index_scores: [b, sq, sk]  (float32)
        topk_indices: [b, sq, topk] (int64)
        query:        [sq, b, np, hn] (bf16)
        key:          [sk, b, np, hn] (bf16)

    Returns scalar float32 loss = mean(KL per position) * loss_coeff.
    """
    sq, b, np_, hn = query.shape
    sk = key.shape[0]

    # --- attention target ---
    # [sq, b, np, hn] -> [b, np, sq, hn]
    q = query.permute(1, 2, 0, 3).float()
    # [sk, b, np, hn] -> [b, np, hn, sk]
    k = key.permute(1, 2, 3, 0).float()
    # [b, np, sq, sk]
    attention_scores = torch.matmul(q, k) * softmax_scale

    if sparse_loss:
        # Build -inf mask, scatter 0 at topk positions
        sparse_mask = torch.full(
            (b, sq, sk), float("-inf"), dtype=torch.float32, device=query.device,
        )
        sparse_mask.scatter_(-1, topk_indices, 0.0)
        # Broadcast [b, 1, sq, sk] across heads
        attention_scores = attention_scores + sparse_mask.unsqueeze(1)

    # Causal mask
    causal = torch.triu(
        torch.full((sq, sk), float("-inf"), dtype=torch.float32, device=query.device),
        diagonal=1,
    )
    attention_scores = attention_scores + causal.unsqueeze(0).unsqueeze(0)

    # Softmax per head, then average across heads
    attention_probs = torch.softmax(attention_scores, dim=-1)  # [b, np, sq, sk]
    attention_target = attention_probs.mean(dim=1)  # [b, sq, sk]

    # --- index softmax ---
    if sparse_loss:
        idx_input = index_scores + sparse_mask
    else:
        idx_input = index_scores
    index_probs = torch.softmax(idx_input, dim=-1)  # [b, sq, sk]

    # --- KL divergence: sum_k p * (log p - log q) ---
    eps = 1e-10
    kl = attention_target * (
        torch.log(attention_target + eps) - torch.log(index_probs + eps)
    )
    per_position_loss = kl.sum(dim=-1)  # [b, sq]
    return per_position_loss.mean() * loss_coeff


# ---------------------------------------------------------------------------
# Stub pg_collection for the split-K API
# ---------------------------------------------------------------------------

class _FakeTP:
    def size(self):
        return 1

class _FakePG:
    tp = _FakeTP()


# ---------------------------------------------------------------------------
# Helper to generate test inputs
# ---------------------------------------------------------------------------

def _make_inputs(
    *,
    sq: int = 128,
    sk: int = 128,
    b: int = 1,
    np_: int = 4,
    hn: int = 64,
    topk: int = 32,
    seed: int = 42,
    device: str = "cuda",
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn(sq, b, np_, hn, generator=g, dtype=torch.bfloat16).to(device)
    key = torch.randn(sk, b, np_, hn, generator=g, dtype=torch.bfloat16).to(device)
    index_scores = torch.randn(b, sq, sk, generator=g, dtype=torch.float32).to(device)

    # Build valid topk indices (each row: topk distinct values in [0, sk))
    topk_indices = torch.stack([
        torch.randperm(sk, generator=g)[:topk]
        for _ in range(b * sq)
    ]).reshape(b, sq, topk).to(device)

    return query, key, index_scores, topk_indices


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_REQUIRES_CUDA
@pytest.mark.parametrize("sparse_loss", [False, True], ids=["dense", "sparse"])
def test_splitk_matches_reference_small(sparse_loss: bool):
    """Small shape: sq=128, sk=128, b=1, np=4, hn=64."""
    from cppmega.megatron.dsa_splitk_indexer_loss import compute_dsa_indexer_loss_splitk

    query, key, index_scores, topk_indices = _make_inputs(
        sq=128, sk=128, b=1, np_=4, hn=64, topk=32,
    )
    softmax_scale = 1.0 / math.sqrt(64)
    loss_coeff = 1.0

    ref = _compute_dsa_indexer_loss_reference(
        index_scores, topk_indices, query, key,
        softmax_scale, loss_coeff, sparse_loss,
    )
    splitk = compute_dsa_indexer_loss_splitk(
        index_scores, topk_indices, query, key,
        softmax_scale, loss_coeff, sparse_loss, _FakePG(),
    )

    torch.testing.assert_close(
        splitk, ref, atol=0.01, rtol=0.01,
        msg=f"Split-K vs reference mismatch (sparse_loss={sparse_loss})",
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("sparse_loss", [False, True], ids=["dense", "sparse"])
def test_splitk_matches_reference_multihead(sparse_loss: bool):
    """More heads: sq=128, sk=128, b=2, np=8, hn=64."""
    from cppmega.megatron.dsa_splitk_indexer_loss import compute_dsa_indexer_loss_splitk

    query, key, index_scores, topk_indices = _make_inputs(
        sq=128, sk=128, b=2, np_=8, hn=64, topk=32, seed=99,
    )
    softmax_scale = 1.0 / math.sqrt(64)
    loss_coeff = 0.5

    ref = _compute_dsa_indexer_loss_reference(
        index_scores, topk_indices, query, key,
        softmax_scale, loss_coeff, sparse_loss,
    )
    splitk = compute_dsa_indexer_loss_splitk(
        index_scores, topk_indices, query, key,
        softmax_scale, loss_coeff, sparse_loss, _FakePG(),
    )

    torch.testing.assert_close(
        splitk, ref, atol=0.01, rtol=0.01,
        msg=f"Split-K vs reference mismatch (multi-head, sparse_loss={sparse_loss})",
    )


@_REQUIRES_CUDA
def test_splitk_loss_coeff_scales_linearly():
    """Verify that doubling loss_coeff doubles the output."""
    from cppmega.megatron.dsa_splitk_indexer_loss import compute_dsa_indexer_loss_splitk

    query, key, index_scores, topk_indices = _make_inputs(
        sq=128, sk=128, b=1, np_=4, hn=64, topk=32,
    )
    softmax_scale = 1.0 / math.sqrt(64)

    loss1 = compute_dsa_indexer_loss_splitk(
        index_scores, topk_indices, query, key,
        softmax_scale, 1.0, False, _FakePG(),
    )
    loss2 = compute_dsa_indexer_loss_splitk(
        index_scores, topk_indices, query, key,
        softmax_scale, 2.0, False, _FakePG(),
    )

    torch.testing.assert_close(loss2, loss1 * 2.0, atol=1e-5, rtol=1e-5)


@_REQUIRES_CUDA
def test_splitk_output_dtype_and_shape():
    """Output must be a scalar fp32 tensor."""
    from cppmega.megatron.dsa_splitk_indexer_loss import compute_dsa_indexer_loss_splitk

    query, key, index_scores, topk_indices = _make_inputs(
        sq=128, sk=128, b=1, np_=4, hn=64, topk=32,
    )
    loss = compute_dsa_indexer_loss_splitk(
        index_scores, topk_indices, query, key,
        1.0 / math.sqrt(64), 1.0, False, _FakePG(),
    )
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.dtype == torch.float32, f"Expected float32, got {loss.dtype}"


@_REQUIRES_CUDA
def test_splitk_larger_seq():
    """Larger sequence: sq=256, sk=256 to exercise multiple sk blocks."""
    from cppmega.megatron.dsa_splitk_indexer_loss import compute_dsa_indexer_loss_splitk

    query, key, index_scores, topk_indices = _make_inputs(
        sq=256, sk=256, b=1, np_=4, hn=64, topk=64, seed=77,
    )
    softmax_scale = 1.0 / math.sqrt(64)
    loss_coeff = 1.0

    ref = _compute_dsa_indexer_loss_reference(
        index_scores, topk_indices, query, key,
        softmax_scale, loss_coeff, False,
    )
    splitk = compute_dsa_indexer_loss_splitk(
        index_scores, topk_indices, query, key,
        softmax_scale, loss_coeff, False, _FakePG(),
    )

    torch.testing.assert_close(
        splitk, ref, atol=0.01, rtol=0.01,
        msg="Split-K vs reference mismatch (larger seq)",
    )


@_REQUIRES_CUDA
def test_patch_routing_splitk(monkeypatch):
    """Verify that CPPMEGA_DSA_KL_MODE=splitk is resolved correctly."""
    from cppmega.megatron.dsa_fp8_patch import resolve_kl_mode

    monkeypatch.setenv("CPPMEGA_DSA_KL_MODE", "splitk")
    assert resolve_kl_mode() == "splitk"

    monkeypatch.setenv("CPPMEGA_DSA_KL_MODE", "split-k")
    assert resolve_kl_mode() == "splitk"

    monkeypatch.setenv("CPPMEGA_DSA_KL_MODE", "")
    assert resolve_kl_mode() == "head_streaming"

    monkeypatch.delenv("CPPMEGA_DSA_KL_MODE", raising=False)
    assert resolve_kl_mode() == "head_streaming"
