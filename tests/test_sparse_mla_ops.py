"""Unit tests for cppmega.megatron.sparse_mla_ops — SparseMLA autograd.Function.

Tests compare SparseMLA output (via the ``sparse_mla_as_unfused_dsa`` adapter)
against the reference PyTorch ``sparse_dsa_fn`` at small shapes.

If TileLang JIT is unavailable (no GPU, TileLang not installed, or JIT compile
error), tests are skipped with an informative message.

Test shapes are deliberately small to avoid OOM and keep CI fast:
  sq=128, sk=128, b=2, np=16, hn=576 (dim+tail_dim), hnv=512 (dim), topk=64
  (topk must be multiple of block_I=64 for the TileLang kernel)
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Check TileLang availability
# ---------------------------------------------------------------------------
_TILELANG_SKIP_REASON = None

try:
    import tilelang  # noqa: F401
except ImportError:
    _TILELANG_SKIP_REASON = "tilelang not installed"

if _TILELANG_SKIP_REASON is None and not torch.cuda.is_available():
    _TILELANG_SKIP_REASON = "CUDA not available"

requires_tilelang = pytest.mark.skipif(
    _TILELANG_SKIP_REASON is not None, reason=_TILELANG_SKIP_REASON or ""
)


# ---------------------------------------------------------------------------
# Reference implementation (PyTorch gather-scatter)
# ---------------------------------------------------------------------------
def reference_sparse_dsa(query, key, value, topk_indices, softmax_scale):
    """Reference sparse DSA using cppmega.megatron.dsa_sparse_attention.sparse_dsa_fn."""
    from cppmega.megatron.dsa_sparse_attention import sparse_dsa_fn

    return sparse_dsa_fn(query, key, value, topk_indices, softmax_scale)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _make_test_tensors(
    sq=128,
    sk=128,
    b=2,
    np_=16,
    hn=576,  # dim + tail_dim
    hnv=512,  # dim (value channels)
    topk=64,
    dtype=torch.bfloat16,
    device="cuda",
    seed=42,
):
    """Create test tensors in Megatron's unfused_dsa_fn layout."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Query: [sq, b, np, hn]
    query = torch.randn(sq, b, np_, hn, dtype=dtype, device=device) * 0.1
    # Key: [sk, b, np, hn] — full KV latent for MLA
    key = torch.randn(sk, b, np_, hn, dtype=dtype, device=device) * 0.1
    # Value: [sk, b, np, hnv] — first hnv channels of key for MLA
    # In real MLA, V is a prefix of K. We match that here.
    value = key[..., :hnv].clone()

    # topk_indices: [b, sq, topk] — valid KV positions to attend to
    # Generate random valid indices (causal: indices <= position)
    topk_indices = torch.zeros(b, sq, topk, dtype=torch.int64, device=device)
    for bi in range(b):
        for si in range(sq):
            max_idx = min(si + 1, sk)
            if max_idx >= topk:
                perm = torch.randperm(max_idx, device=device)[:topk]
            else:
                # Not enough valid positions: fill with valid ones + repeat
                perm = torch.arange(max_idx, device=device)
                # Pad with repeats of position 0
                perm = torch.cat(
                    [perm, torch.zeros(topk - max_idx, dtype=torch.int64, device=device)]
                )
            topk_indices[bi, si] = perm.sort().values

    softmax_scale = (hn) ** (-0.5)

    return query, key, value, topk_indices, softmax_scale


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@requires_tilelang
class TestSparseMLA:
    """Test SparseMLA against reference sparse_dsa_fn."""

    def test_forward_parity(self):
        """Forward output matches reference within bf16 tolerance."""
        query, key, value, topk_indices, softmax_scale = _make_test_tensors()

        # Reference (PyTorch gather-scatter)
        ref_out = reference_sparse_dsa(
            query.clone(), key.clone(), value.clone(), topk_indices.clone(), softmax_scale
        )

        # TileLang SparseMLA via adapter
        from cppmega.megatron.sparse_mla_ops.sparse_mla import sparse_mla_as_unfused_dsa

        tl_out = sparse_mla_as_unfused_dsa(
            query.clone(), key.clone(), value.clone(), topk_indices.clone(), softmax_scale
        )

        assert ref_out.shape == tl_out.shape, (
            f"Shape mismatch: ref={ref_out.shape}, tl={tl_out.shape}"
        )

        # bf16 tolerance: atol=0.05, rtol=0.05
        close = torch.allclose(ref_out.float(), tl_out.float(), atol=0.05, rtol=0.05)
        if not close:
            diff = (ref_out.float() - tl_out.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            pytest.fail(
                f"Forward parity failed: max_diff={max_diff:.6f}, "
                f"mean_diff={mean_diff:.6f} (atol=0.05, rtol=0.05)"
            )

    def test_backward_parity(self):
        """Backward gradients for Q, K match reference within bf16 tolerance."""
        query, key, value, topk_indices, softmax_scale = _make_test_tensors()

        # Reference backward
        q_ref = query.clone().requires_grad_(True)
        k_ref = key.clone().requires_grad_(True)
        v_ref = k_ref[..., :512]  # MLA: V is prefix of K
        ref_out = reference_sparse_dsa(q_ref, k_ref, v_ref, topk_indices.clone(), softmax_scale)
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)
        ref_grad_q = q_ref.grad.clone()
        ref_grad_k = k_ref.grad.clone()

        # TileLang backward
        q_tl = query.clone().requires_grad_(True)
        k_tl = key.clone().requires_grad_(True)
        v_tl = k_tl[..., :512]
        from cppmega.megatron.sparse_mla_ops.sparse_mla import sparse_mla_as_unfused_dsa

        tl_out = sparse_mla_as_unfused_dsa(
            q_tl, k_tl, v_tl, topk_indices.clone(), softmax_scale
        )
        tl_out.backward(grad_out)
        tl_grad_q = q_tl.grad.clone()
        tl_grad_k = k_tl.grad.clone()

        # Check Q grad
        close_q = torch.allclose(ref_grad_q.float(), tl_grad_q.float(), atol=0.05, rtol=0.05)
        if not close_q:
            diff_q = (ref_grad_q.float() - tl_grad_q.float()).abs()
            max_diff_q = diff_q.max().item()
            mean_diff_q = diff_q.mean().item()
            pytest.fail(
                f"Backward Q grad parity failed: max_diff={max_diff_q:.6f}, "
                f"mean_diff={mean_diff_q:.6f} (atol=0.05, rtol=0.05)"
            )

        # Check K grad
        close_k = torch.allclose(ref_grad_k.float(), tl_grad_k.float(), atol=0.05, rtol=0.05)
        if not close_k:
            diff_k = (ref_grad_k.float() - tl_grad_k.float()).abs()
            max_diff_k = diff_k.max().item()
            mean_diff_k = diff_k.mean().item()
            pytest.fail(
                f"Backward K grad parity failed: max_diff={max_diff_k:.6f}, "
                f"mean_diff={mean_diff_k:.6f} (atol=0.05, rtol=0.05)"
            )

    def test_sparse_mla_autograd_direct(self):
        """Test SparseMLA autograd.Function directly (not through adapter)."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        b, sq, heads, dim_plus_tail = 2, 128, 16, 576
        sk, kv_group = 128, 1
        dim = 512
        topk = 64

        q = torch.randn(b, sq, heads, dim_plus_tail, dtype=torch.bfloat16, device="cuda") * 0.1
        kv = torch.randn(b, sk, kv_group, dim_plus_tail, dtype=torch.bfloat16, device="cuda") * 0.1

        # Build valid causal indices with -1 sentinel for padding
        indices = torch.full((b, sq, kv_group, topk), -1, dtype=torch.int32, device="cuda")
        for bi in range(b):
            for si in range(sq):
                max_idx = min(si + 1, sk)
                n = min(max_idx, topk)
                if n > 0:
                    perm = torch.randperm(max_idx, device="cuda")[:n].sort().values
                    indices[bi, si, 0, :n] = perm.to(torch.int32)

        q.requires_grad_(True)
        kv.requires_grad_(True)

        from cppmega.megatron.sparse_mla_ops import SparseMLA

        scaling = dim_plus_tail ** (-0.5)
        out, lse = SparseMLA.apply(q, kv, indices, scaling)

        assert out.shape == (b, sq, heads, dim), f"Unexpected output shape: {out.shape}"
        assert lse.shape == (b, sq, heads), f"Unexpected lse shape: {lse.shape}"

        # Check backward runs without error
        loss = out.sum()
        loss.backward()

        assert q.grad is not None, "Q grad is None after backward"
        assert kv.grad is not None, "KV grad is None after backward"
        assert q.grad.shape == q.shape, f"Q grad shape mismatch: {q.grad.shape} vs {q.shape}"
        assert kv.grad.shape == kv.shape, f"KV grad shape mismatch: {kv.grad.shape} vs {kv.shape}"

    # NOTE: the ``test_monkey_patch_resolve`` test was removed along with the
    # ``dsa_fp8_patch`` module on 2026-04-13. The equivalent routing now lives
    # in ``dsa_indexer_fused_patch`` and is exercised by
    # ``tests/test_dsa_indexer_fused_patch.py``.
