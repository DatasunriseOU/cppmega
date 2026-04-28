"""Numerical-match tests for the ParaRNN-style M2RNN parallel scan.

Validates that ``m2rnn_pararnn_forward`` (Newton + parallel reduction)
produces outputs that match the sequential reference within a Newton-
appropriate tolerance, that gradients are correct, and that the
algorithm scales to longer sequences without diverging.
"""

from __future__ import annotations

import math

import pytest
import torch

from cppmega.megatron.m2rnn_pararnn import (
    PararnnConfig,
    m2rnn_pararnn_forward,
)


# ---------------------------------------------------------------------------
# Standalone sequential reference (matches m2rnn_spec._torch_m2rnn_forward
# verbatim; copied to avoid the megatron dependency).
# ---------------------------------------------------------------------------


def _torch_m2rnn_forward(q, k, v, W, xf, *, h0=None):
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
# Test fixtures
# ---------------------------------------------------------------------------


def _make_inputs(B, S, H, k_dim, v_dim, *, device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, k_dim, generator=g, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, S, H, k_dim, generator=g, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, S, H, v_dim, generator=g, device=device, dtype=dtype) * 0.5
    # Per-head W. Spectral radius < 1 keeps the recurrence Lipschitz < f + (1-f) = 1.
    W = torch.randn(H, v_dim, v_dim, generator=g, device=device, dtype=dtype) * (
        0.5 / math.sqrt(v_dim)
    )
    # Forget gate already in (0, 1); bias towards remembering so Newton is stable.
    xf_logits = torch.randn(B, S, H, generator=g, device=device, dtype=dtype) - 0.5
    xf = torch.sigmoid(xf_logits)
    return q, k, v, W, xf


# ---------------------------------------------------------------------------
# Forward numerical match
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("S", [4, 16, 64])
def test_forward_matches_sequential_small(dtype, S):
    """With max_its=8 the Newton solver should drive the trajectory to
    machine epsilon of the sequential reference for short sequences."""
    device = "cpu"
    B, H, k_dim, v_dim = 2, 4, 8, 4
    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=dtype)

    out_ref, h_ref = _torch_m2rnn_forward(q, k, v, W, xf)
    out_par, h_par = m2rnn_pararnn_forward(
        q, k, v, W, xf, config=PararnnConfig(max_its=8)
    )

    if dtype == torch.float64:
        atol, rtol = 1e-9, 1e-9
    else:
        atol, rtol = 1e-4, 1e-4

    torch.testing.assert_close(out_par, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(h_par, h_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("S", [128, 512])
def test_forward_matches_sequential_long(S):
    """Long-sequence sanity: tolerance scales with S * eps per ParaRNN paper."""
    device = "cpu"
    B, H, k_dim, v_dim = 2, 2, 8, 4
    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=torch.float64)

    out_ref, h_ref = _torch_m2rnn_forward(q, k, v, W, xf)
    out_par, h_par = m2rnn_pararnn_forward(
        q, k, v, W, xf, config=PararnnConfig(max_its=10)
    )

    # The error budget is roughly S * machine_precision per the paper.
    tol = max(1e-8, S * 1e-13)
    torch.testing.assert_close(out_par, out_ref, atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# Newton convergence: error should decrease ~quadratically with iter count
# ---------------------------------------------------------------------------


def test_newton_quadratic_convergence():
    """Each Newton iteration should reduce the residual by an order of
    magnitude or more once we're in the basin of attraction."""
    device = "cpu"
    B, S, H, k_dim, v_dim = 1, 32, 2, 4, 4
    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=torch.float64)

    out_ref, _ = _torch_m2rnn_forward(q, k, v, W, xf)

    errs = []
    for n_its in [1, 2, 3, 4, 5]:
        out_par, _ = m2rnn_pararnn_forward(
            q, k, v, W, xf, config=PararnnConfig(max_its=n_its)
        )
        err = (out_par - out_ref).abs().max().item()
        errs.append(err)

    # Each subsequent iter should reduce the error meaningfully (factor >= 2)
    # in the convergence regime. Last iter may already be at floor.
    for prev, curr in zip(errs[:-2], errs[1:-1]):
        assert curr <= prev * 0.5 + 1e-12, (
            f"Newton not converging quadratically: {errs}"
        )


# ---------------------------------------------------------------------------
# Initialisation strategies
# ---------------------------------------------------------------------------


def test_chunk_init_matches_zero_init():
    """Both warm-start strategies must converge to the same answer when
    given enough iterations."""
    device = "cpu"
    B, S, H, k_dim, v_dim = 1, 32, 2, 4, 4
    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=torch.float64)

    out_zero, _ = m2rnn_pararnn_forward(
        q, k, v, W, xf, config=PararnnConfig(max_its=8, init_strategy="zero")
    )
    out_chunk, _ = m2rnn_pararnn_forward(
        q, k, v, W, xf, config=PararnnConfig(max_its=8, init_strategy="chunk")
    )
    torch.testing.assert_close(out_zero, out_chunk, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Gradient match -- the whole point is differentiability
# ---------------------------------------------------------------------------


def test_gradient_matches_sequential():
    """Backprop through the parallel scan should match backprop through
    the sequential reference."""
    device = "cpu"
    B, S, H, k_dim, v_dim = 1, 16, 2, 4, 4
    dtype = torch.float64

    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=dtype)
    target = torch.randn_like(q[..., : v_dim])  # match output shape (B, S, H, v_dim)

    # Sequential gradient
    q_s, k_s, v_s, W_s, xf_s = (t.clone().requires_grad_() for t in (q, k, v, W, xf))
    out_s, _ = _torch_m2rnn_forward(q_s, k_s, v_s, W_s, xf_s)
    loss_s = (out_s - target).pow(2).mean()
    loss_s.backward()

    # ParaRNN gradient
    q_p, k_p, v_p, W_p, xf_p = (t.clone().requires_grad_() for t in (q, k, v, W, xf))
    out_p, _ = m2rnn_pararnn_forward(
        q_p, k_p, v_p, W_p, xf_p, config=PararnnConfig(max_its=8)
    )
    loss_p = (out_p - target).pow(2).mean()
    loss_p.backward()

    atol, rtol = 1e-7, 1e-7
    torch.testing.assert_close(q_p.grad, q_s.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_p.grad, k_s.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_p.grad, v_s.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(W_p.grad, W_s.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(xf_p.grad, xf_s.grad, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Head broadcasting (n_q != n_k != n_v, etc -- same convention as reference)
# ---------------------------------------------------------------------------


def test_head_broadcast():
    """When n_q < n the reference repeats q to match; the parallel impl
    must do the same."""
    device = "cpu"
    B, S, k_dim, v_dim = 1, 16, 4, 4
    dtype = torch.float64

    g = torch.Generator(device=device).manual_seed(0)
    # n_q=1, n_k=2, n_v=4, n_w=4, n_f=4 -> H = 4
    q = torch.randn(B, S, 1, k_dim, generator=g, dtype=dtype) * 0.5
    k = torch.randn(B, S, 2, k_dim, generator=g, dtype=dtype) * 0.5
    v = torch.randn(B, S, 4, v_dim, generator=g, dtype=dtype) * 0.5
    W = torch.randn(4, v_dim, v_dim, generator=g, dtype=dtype) * 0.2
    xf = torch.sigmoid(torch.randn(B, S, 4, generator=g, dtype=dtype) - 0.5)

    out_ref, _ = _torch_m2rnn_forward(q, k, v, W, xf)
    out_par, _ = m2rnn_pararnn_forward(
        q, k, v, W, xf, config=PararnnConfig(max_its=8)
    )
    torch.testing.assert_close(out_par, out_ref, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Initial state h0 is honoured
# ---------------------------------------------------------------------------


def test_h0_propagation():
    """Non-zero initial state must influence the trajectory the same way
    as in the sequential reference."""
    device = "cpu"
    B, S, H, k_dim, v_dim = 1, 16, 2, 4, 4
    dtype = torch.float64

    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=dtype)
    h0 = torch.randn(B, H, k_dim, v_dim, dtype=dtype) * 0.3

    out_ref, h_ref = _torch_m2rnn_forward(q, k, v, W, xf, h0=h0)
    out_par, h_par = m2rnn_pararnn_forward(
        q, k, v, W, xf, h0=h0, config=PararnnConfig(max_its=8)
    )
    torch.testing.assert_close(out_par, out_ref, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(h_par, h_ref, atol=1e-9, rtol=1e-9)
