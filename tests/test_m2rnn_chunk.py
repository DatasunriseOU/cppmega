"""Tests for chunked parallel M²RNN recurrence.

Validates that ``chunked_m2rnn_forward`` produces outputs numerically
identical (within tolerance) to the sequential reference.
"""

from __future__ import annotations

import math

import pytest
import torch

from cppmega.megatron.m2rnn_chunk import (
    DEFAULT_CHUNK_SIZE,
    _chunked_m2rnn_forward,
    chunked_m2rnn_forward,
)


# ---------------------------------------------------------------------------
# Standalone sequential reference (copied from m2rnn_spec._torch_m2rnn_forward
# to avoid pulling in the megatron dependency which is remote-only).
# ---------------------------------------------------------------------------


def _torch_m2rnn_forward(q, k, v, W, xf, *, h0=None):
    """Sequential M²RNN — ground truth for comparison."""
    batch, seq, n_q, k_dim = q.shape
    n_k = k.size(-2)
    n_v = v.size(-2)
    n_w = W.size(0)
    n_f = xf.size(-1)
    v_dim = v.size(-1)
    n = max(n_q, n_k, n_v, n_w, n_f)

    if h0 is None:
        h = torch.zeros(batch, n, k_dim, v_dim, device=q.device, dtype=q.dtype)
    else:
        h = h0

    if n_q != n:
        q = q.repeat_interleave(n // n_q, dim=-2)
    if n_k != n:
        k = k.repeat_interleave(n // n_k, dim=-2)
    if n_v != n:
        v = v.repeat_interleave(n // n_v, dim=-2)
    if n_w != n:
        W = W.repeat_interleave(n // n_w, dim=0)
    if n_f != n:
        xf = xf.repeat_interleave(n // n_f, dim=-1)

    x = k[..., None] * v[..., None, :]
    W_expanded = W[None, ...]
    y = torch.empty(batch, seq, n, k_dim, v_dim, device=q.device, dtype=q.dtype)
    for s in range(seq):
        f = xf[:, s, :, None, None]
        h_new = torch.tanh(h @ W_expanded + x[:, s])
        h = f * h + (1 - f) * h_new
        y[:, s] = h
    out = (q[..., None, :] @ y).squeeze(-2)
    return out, h


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_inputs(
    batch: int = 2,
    seq: int = 512,
    n_heads: int = 4,
    k_dim: int = 64,
    v_dim: int = 16,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
):
    """Generate random inputs matching the M²RNN interface."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, seq, n_heads, k_dim, device=device, dtype=dtype, generator=g)
    k = torch.randn(batch, seq, n_heads, k_dim, device=device, dtype=dtype, generator=g)
    v = torch.randn(batch, seq, n_heads, v_dim, device=device, dtype=dtype, generator=g)
    # W initialized close to identity (like the real model)
    W = (
        torch.eye(v_dim, device=device, dtype=dtype)
        .unsqueeze(0)
        .expand(n_heads, -1, -1)
        .clone()
    )
    W += 0.01 * torch.randn(W.shape, device=device, dtype=dtype, generator=g)
    # Forget gate: softplus produces values in (0, 1) when combined with A_log
    xf = torch.rand(batch, seq, n_heads, device=device, dtype=dtype, generator=g)
    return q, k, v, W, xf


def _reshape_sequential_output(out_seq: torch.Tensor) -> torch.Tensor:
    """Reshape sequential output (B, S, H, k_dim, v_dim) -> (B, S, H, v_dim).

    The sequential ``_torch_m2rnn_forward`` returns the full state ``y`` and
    the caller does ``(q[..., None, :] @ y).squeeze(-2)`` to get (B,S,H,v_dim).
    We replicate that here.
    """
    # out_seq is already (B, S, H, v_dim) from the q-projection in the
    # sequential code's  out = (q[..., None, :] @ y).squeeze(-2)
    return out_seq


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChunkedM2RNNMatchesSequential:
    """Ensure chunked output matches sequential output across configs."""

    @pytest.mark.parametrize("chunk_size", [16, 32, 64, 128, 256])
    def test_various_chunk_sizes(self, chunk_size: int):
        seq = 256
        q, k, v, W, xf = _random_inputs(batch=2, seq=seq, n_heads=2, k_dim=16, v_dim=8)

        out_seq, h_seq = _torch_m2rnn_forward(q, k, v, W, xf)
        out_chunk, h_chunk = _chunked_m2rnn_forward(q, k, v, W, xf, chunk_size=chunk_size)

        torch.testing.assert_close(out_chunk, out_seq, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(h_chunk, h_seq, atol=1e-4, rtol=1e-4)

    def test_seq_not_divisible_by_chunk(self):
        """Sequence length not a multiple of chunk size."""
        seq = 200  # not divisible by 128
        q, k, v, W, xf = _random_inputs(batch=1, seq=seq, n_heads=2, k_dim=16, v_dim=8)

        out_seq, h_seq = _torch_m2rnn_forward(q, k, v, W, xf)
        out_chunk, h_chunk = _chunked_m2rnn_forward(q, k, v, W, xf, chunk_size=64)

        torch.testing.assert_close(out_chunk, out_seq, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(h_chunk, h_seq, atol=1e-4, rtol=1e-4)

    def test_single_chunk_matches(self):
        """When chunk_size >= seq_len, should exactly match sequential."""
        seq = 100
        q, k, v, W, xf = _random_inputs(batch=1, seq=seq, n_heads=2, k_dim=16, v_dim=8)

        out_seq, h_seq = _torch_m2rnn_forward(q, k, v, W, xf)
        out_chunk, h_chunk = _chunked_m2rnn_forward(q, k, v, W, xf, chunk_size=256)

        torch.testing.assert_close(out_chunk, out_seq, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h_chunk, h_seq, atol=1e-5, rtol=1e-5)

    def test_with_initial_state(self):
        """Passing h0 should work identically in both."""
        seq = 128
        B, H, k_dim, v_dim = 2, 3, 16, 8
        q, k, v, W, xf = _random_inputs(batch=B, seq=seq, n_heads=H, k_dim=k_dim, v_dim=v_dim)
        h0 = 0.1 * torch.randn(B, H, k_dim, v_dim)

        out_seq, h_seq = _torch_m2rnn_forward(q, k, v, W, xf, h0=h0)
        out_chunk, h_chunk = _chunked_m2rnn_forward(q, k, v, W, xf, h0=h0, chunk_size=32)

        torch.testing.assert_close(out_chunk, out_seq, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(h_chunk, h_seq, atol=1e-4, rtol=1e-4)

    def test_chunk_size_one_matches(self):
        """chunk_size=1 should degenerate to fully sequential."""
        seq = 50
        q, k, v, W, xf = _random_inputs(batch=1, seq=seq, n_heads=1, k_dim=8, v_dim=4)

        out_seq, h_seq = _torch_m2rnn_forward(q, k, v, W, xf)
        out_chunk, h_chunk = _chunked_m2rnn_forward(q, k, v, W, xf, chunk_size=1)

        torch.testing.assert_close(out_chunk, out_seq, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h_chunk, h_seq, atol=1e-5, rtol=1e-5)

    def test_full_scale_dimensions(self):
        """Test with production-scale k_dim=64, v_dim=16."""
        seq = 256
        q, k, v, W, xf = _random_inputs(batch=1, seq=seq, n_heads=2, k_dim=64, v_dim=16)

        out_seq, h_seq = _torch_m2rnn_forward(q, k, v, W, xf)
        out_chunk, h_chunk = _chunked_m2rnn_forward(q, k, v, W, xf, chunk_size=64)

        torch.testing.assert_close(out_chunk, out_seq, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(h_chunk, h_seq, atol=1e-3, rtol=1e-3)


class TestChunkedM2RNNHeadBroadcast:
    """Test that head-count broadcasting works correctly."""

    def test_different_head_counts(self):
        """n_q=1, n_k=1, n_v=4, n_w=1 — broadcasting should match."""
        B, S = 1, 64
        n_q, n_k, n_v, n_w = 1, 1, 4, 1
        k_dim, v_dim = 16, 8

        g = torch.Generator().manual_seed(123)
        q = torch.randn(B, S, n_q, k_dim, generator=g)
        k = torch.randn(B, S, n_k, k_dim, generator=g)
        v = torch.randn(B, S, n_v, v_dim, generator=g)
        W = torch.eye(v_dim).unsqueeze(0).expand(n_w, -1, -1).clone()
        xf = torch.rand(B, S, n_v, generator=g)

        out_seq, h_seq = _torch_m2rnn_forward(q, k, v, W, xf)
        out_chunk, h_chunk = _chunked_m2rnn_forward(q, k, v, W, xf, chunk_size=16)

        torch.testing.assert_close(out_chunk, out_seq, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(h_chunk, h_seq, atol=1e-4, rtol=1e-4)


class TestChunkedM2RNNPublicAPI:
    """Test the public ``chunked_m2rnn_forward`` entry point."""

    def test_public_api_matches(self):
        q, k, v, W, xf = _random_inputs(batch=1, seq=128, n_heads=2, k_dim=16, v_dim=8)

        out_internal, h_internal = _chunked_m2rnn_forward(q, k, v, W, xf)
        out_public, h_public = chunked_m2rnn_forward(q, k, v, W, xf)

        torch.testing.assert_close(out_public, out_internal)
        torch.testing.assert_close(h_public, h_internal)

    def test_default_chunk_size(self):
        assert DEFAULT_CHUNK_SIZE == 128


class TestChunkedM2RNNEdgeCases:
    """Edge cases and boundary conditions."""

    def test_seq_len_one(self):
        q, k, v, W, xf = _random_inputs(batch=1, seq=1, n_heads=1, k_dim=8, v_dim=4)
        out_seq, h_seq = _torch_m2rnn_forward(q, k, v, W, xf)
        out_chunk, h_chunk = _chunked_m2rnn_forward(q, k, v, W, xf, chunk_size=128)
        torch.testing.assert_close(out_chunk, out_seq, atol=1e-5, rtol=1e-5)

    def test_seq_equals_chunk(self):
        q, k, v, W, xf = _random_inputs(batch=1, seq=128, n_heads=1, k_dim=8, v_dim=4)
        out_seq, h_seq = _torch_m2rnn_forward(q, k, v, W, xf)
        out_chunk, h_chunk = _chunked_m2rnn_forward(q, k, v, W, xf, chunk_size=128)
        torch.testing.assert_close(out_chunk, out_seq, atol=1e-5, rtol=1e-5)

    def test_zero_initial_state_is_default(self):
        """h0=None should behave like h0=zeros."""
        B, H, k_dim, v_dim = 1, 2, 8, 4
        q, k, v, W, xf = _random_inputs(batch=B, seq=64, n_heads=H, k_dim=k_dim, v_dim=v_dim)
        h0_zeros = torch.zeros(B, H, k_dim, v_dim)

        out_none, h_none = _chunked_m2rnn_forward(q, k, v, W, xf, h0=None, chunk_size=16)
        out_zero, h_zero = _chunked_m2rnn_forward(q, k, v, W, xf, h0=h0_zeros, chunk_size=16)

        torch.testing.assert_close(out_none, out_zero, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(h_none, h_zero, atol=1e-6, rtol=1e-6)
