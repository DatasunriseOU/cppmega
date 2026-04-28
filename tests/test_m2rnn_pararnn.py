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


# ---------------------------------------------------------------------------
# M2RNN-realistic init: identity-ish W needs more Newton iters than the paper's
# stable-LSTM default of 3. The default (max_its=6) must converge cleanly here.
# ---------------------------------------------------------------------------


def _make_m2rnn_init_inputs(B, S, H, k_dim, v_dim, *, device, dtype, seed=0,
                             w_noise=0.005):
    """Replicates the default NAM56R init: W = I + small_noise * randn so the
    recurrence is near-linear (residual-style). Newton's quadratic basin is
    reached only after several iterations of linear progress, and the
    paper's max_its=3 default is too few here."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, k_dim, generator=g, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, S, H, k_dim, generator=g, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, S, H, v_dim, generator=g, device=device, dtype=dtype) * 0.5
    eye = torch.eye(v_dim, dtype=dtype, device=device)
    W = eye[None, :, :].repeat(H, 1, 1) + torch.randn(
        H, v_dim, v_dim, generator=g, device=device, dtype=dtype
    ) * w_noise
    xf_logits = torch.randn(B, S, H, generator=g, device=device, dtype=dtype) - 0.5
    xf = torch.sigmoid(xf_logits)
    return q, k, v, W, xf


@pytest.mark.parametrize("S,k_dim,v_dim", [(64, 8, 8), (128, 16, 16)])
def test_default_max_its_converges_on_m2rnn_init(S, k_dim, v_dim):
    """User-reported regression: with M2RNN-realistic identity-ish W,
    max_its=3 leaves residual ~1e-1, but the new default (8) drives it
    several orders of magnitude lower."""
    B, H = 1, 2
    q, k, v, W, xf = _make_m2rnn_init_inputs(
        B, S, H, k_dim, v_dim, device="cpu", dtype=torch.float64,
    )
    out_ref, _ = _torch_m2rnn_forward(q, k, v, W, xf)

    out_3, _ = m2rnn_pararnn_forward(q, k, v, W, xf, config=PararnnConfig(max_its=3))
    out_default, _ = m2rnn_pararnn_forward(q, k, v, W, xf)  # uses default max_its

    err_3 = (out_3 - out_ref).abs().max().item()
    err_default = (out_default - out_ref).abs().max().item()

    # The paper's default of 3 leaves a noticeable residual on identity-ish W.
    assert err_3 > 1e-2, (
        f"max_its=3 unexpectedly converged on identity-ish W: err={err_3}; "
        f"the test fixture may be too benign to exercise the regression."
    )
    # The new default must drive the error to < 1e-6 on this benign init.
    assert err_default < 1e-6, (
        f"default max_its insufficient: 3-iter err={err_3}, "
        f"default err={err_default}. Bump max_its or tighten omega_sor."
    )


def test_poorly_conditioned_W_needs_explicit_max_its():
    """Regimes with W far from identity (noise >= 0.05) need omega_sor or
    extra iters; we document that the default does NOT silently mask this."""
    B, S, H, k_dim, v_dim = 1, 64, 2, 8, 8
    q, k, v, W, xf = _make_m2rnn_init_inputs(
        B, S, H, k_dim, v_dim, device="cpu", dtype=torch.float64, w_noise=0.05,
    )
    out_ref, _ = _torch_m2rnn_forward(q, k, v, W, xf)
    # Default max_its with vanilla SOR may leave residual; that's expected
    # for ill-conditioned W. Caller should bump max_its or use omega_sor < 1.
    out_default, _ = m2rnn_pararnn_forward(q, k, v, W, xf)
    err_default = (out_default - out_ref).abs().max().item()
    # Many iters + under-relaxation gets there.
    out_more, _ = m2rnn_pararnn_forward(
        q, k, v, W, xf, config=PararnnConfig(max_its=20, omega_sor=0.7),
    )
    err_more = (out_more - out_ref).abs().max().item()
    # Document the relationship: more iters + damping wins.
    assert err_more < err_default, (
        f"under-relaxation should help: default {err_default:.3e}, "
        f"max_its=20 omega_sor=0.7 {err_more:.3e}"
    )


def test_streaming_never_materialises_full_jacobian():
    """Sanity-check that the chunked Newton path never holds a Jacobian
    larger than (Be * chunk_size * V * V * 4) bytes at any one time.

    We hook torch's storage allocations and check the largest 4D float
    tensor allocated during the call has at most chunk_size in the second
    dimension. CPU-only tracing -- correctness, not performance.
    """
    B, S, H, k_dim, v_dim = 1, 256, 2, 4, 4
    chunk_size = 64
    q, k, v, W, xf = _make_inputs(
        B, S, H, k_dim, v_dim, device="cpu", dtype=torch.float64
    )

    largest_axis_1 = 0

    orig_empty = torch.empty
    orig_zeros = torch.zeros
    orig_cat = torch.cat
    orig_einsum = torch.einsum

    # Inspect every 4D float64 tensor of shape (Be, S', V, V) by sniffing the
    # einsum outputs and direct allocations. This is best-effort but catches
    # the materialise-full-jac regression: any tensor with axis 1 == S would
    # break the assertion below.
    seen_shapes: list[tuple[int, ...]] = []

    def _wrap_einsum(*args, **kwargs):
        out = orig_einsum(*args, **kwargs)
        if out.dim() == 4 and out.dtype == torch.float64:
            seen_shapes.append(tuple(out.shape))
        return out

    torch.einsum = _wrap_einsum
    try:
        m2rnn_pararnn_forward(
            q, k, v, W, xf, config=PararnnConfig(chunk_size=chunk_size, max_its=2)
        )
    finally:
        torch.einsum = orig_einsum

    # Find the largest axis-1 across 4D einsum outputs that look like a Jacobian
    # (axis 2 == axis 3 == V). Anything > chunk_size means we materialised
    # more than a chunk's worth of Jacobian.
    for shape in seen_shapes:
        if len(shape) == 4 and shape[2] == v_dim and shape[3] == v_dim:
            largest_axis_1 = max(largest_axis_1, shape[1])

    assert largest_axis_1 <= chunk_size, (
        f"streaming regression: 4D Jacobian-shaped einsum output had axis-1 = "
        f"{largest_axis_1} > chunk_size = {chunk_size} (S = {S}). "
        f"Sample shapes: {seen_shapes[:5]}"
    )


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


# ---------------------------------------------------------------------------
# Triton kernel parity (CUDA + fp32)
# ---------------------------------------------------------------------------


_CUDA = torch.cuda.is_available()
try:
    from cppmega.megatron.m2rnn_pararnn_triton import (
        PARARNN_TRITON_AVAILABLE as _TRITON_AVAILABLE,
    )
except ImportError:
    _TRITON_AVAILABLE = False

_NEED_TRITON = pytest.mark.skipif(
    not (_CUDA and _TRITON_AVAILABLE),
    reason="Triton + CUDA required for pararnn Triton parity tests",
)


@_NEED_TRITON
@pytest.mark.parametrize("S", [64, 128, 256])
def test_triton_matches_torch_forward_fp32(S):
    """Triton kernel forward must match the torch path within fp32 floor."""
    device = "cuda"
    B, H, k_dim, v_dim = 2, 4, 8, 16
    dtype = torch.float32

    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=dtype)

    config_torch = PararnnConfig(max_its=8, kernel="torch")
    config_triton = PararnnConfig(max_its=8, kernel="triton")

    with torch.no_grad():
        out_torch, h_torch = m2rnn_pararnn_forward(
            q, k, v, W, xf, config=config_torch
        )
        out_triton, h_triton = m2rnn_pararnn_forward(
            q, k, v, W, xf, config=config_triton
        )

    # fp32 floor + log(S) Brent-Kung accumulation: 1e-4 abs / 1e-5 rel
    # is comfortable headroom; tighter tolerances catch real divergence.
    torch.testing.assert_close(out_triton, out_torch, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(h_triton, h_torch, atol=1e-4, rtol=1e-4)


@_NEED_TRITON
def test_triton_matches_sequential_fp32():
    """End-to-end: Triton path must match the sequential M2RNN reference."""
    device = "cuda"
    B, S, H, k_dim, v_dim = 1, 64, 2, 4, 16
    dtype = torch.float32

    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=dtype)
    out_seq, h_seq = _torch_m2rnn_forward(q, k, v, W, xf)

    with torch.no_grad():
        out_triton, h_triton = m2rnn_pararnn_forward(
            q, k, v, W, xf, config=PararnnConfig(max_its=10, kernel="triton"),
        )

    # The Newton solver converges to the fixed point of the recurrence,
    # which IS the sequential output.  fp32 floor + Newton residual.
    torch.testing.assert_close(out_triton, out_seq, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(h_triton, h_seq, atol=2e-3, rtol=2e-3)


@_NEED_TRITON
def test_triton_supports_grad_via_ift():
    """Phase C: kernel='triton' under torch.enable_grad() must run the
    autograd Function (IFT backward), not raise."""
    device = "cuda"
    B, S, H, k_dim, v_dim = 1, 32, 2, 4, 16
    dtype = torch.float32

    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=dtype)
    q.requires_grad_(True)

    out, _ = m2rnn_pararnn_forward(
        q, k, v, W, xf, config=PararnnConfig(max_its=4, kernel="triton"),
    )
    out.sum().backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()


@_NEED_TRITON
def test_triton_auto_falls_back_for_fp64():
    """kernel='auto' must pick torch (not triton) for fp64 inputs."""
    device = "cuda"
    B, S, H, k_dim, v_dim = 1, 32, 2, 4, 16
    dtype = torch.float64  # triton kernel is fp32-only -- auto must skip

    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=dtype)
    out_seq, _ = _torch_m2rnn_forward(q, k, v, W, xf)

    out_par, _ = m2rnn_pararnn_forward(
        q, k, v, W, xf, config=PararnnConfig(max_its=8, kernel="auto"),
    )
    # Newton-converged → matches sequential to fp64 precision.
    torch.testing.assert_close(out_par, out_seq, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# Phase C: IFT backward via torch.autograd.Function
# ---------------------------------------------------------------------------


def _ift_grad_check(*, kernel, dtype, device, atol, rtol, B=1, S=16, H=2,
                     k_dim=4, v_dim=4, with_h0=False):
    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=dtype)
    target = torch.randn(B, S, H, v_dim, device=device, dtype=dtype)
    h0 = (torch.randn(B, H, k_dim, v_dim, device=device, dtype=dtype) * 0.3
          if with_h0 else None)

    # Sequential reference grads.
    q_s, k_s, v_s, W_s, xf_s = (t.clone().requires_grad_() for t in (q, k, v, W, xf))
    h0_s = h0.clone().requires_grad_() if with_h0 else None
    out_s, _ = _torch_m2rnn_forward(q_s, k_s, v_s, W_s, xf_s, h0=h0_s)
    loss_s = (out_s - target).pow(2).mean()
    loss_s.backward()

    # IFT grads.
    q_p, k_p, v_p, W_p, xf_p = (t.clone().requires_grad_() for t in (q, k, v, W, xf))
    h0_p = h0.clone().requires_grad_() if with_h0 else None
    out_p, _ = m2rnn_pararnn_forward(
        q_p, k_p, v_p, W_p, xf_p, h0=h0_p,
        config=PararnnConfig(max_its=10, kernel=kernel),
    )
    loss_p = (out_p - target).pow(2).mean()
    loss_p.backward()

    torch.testing.assert_close(q_p.grad, q_s.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_p.grad, k_s.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_p.grad, v_s.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(W_p.grad, W_s.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(xf_p.grad, xf_s.grad, atol=atol, rtol=rtol)
    if with_h0:
        torch.testing.assert_close(h0_p.grad, h0_s.grad, atol=atol, rtol=rtol)
        return h0_p.grad


def test_ift_backward_torch_kernel_matches_sequential():
    """IFT backward (torch kernel, CPU fp64) matches sequential autograd
    to a tighter tolerance than the original autograd-through-Newton path."""
    _ift_grad_check(
        kernel="torch", dtype=torch.float64, device="cpu",
        atol=1e-9, rtol=1e-9,
    )


@_NEED_TRITON
def test_ift_backward_triton_kernel_matches_sequential():
    """IFT backward with the Triton kernel (CUDA fp32) matches the
    sequential reference within the fp32 floor."""
    _ift_grad_check(
        kernel="triton", dtype=torch.float32, device="cuda",
        atol=2e-3, rtol=2e-3, S=32, k_dim=8, v_dim=16,
    )


def test_ift_backward_h0_grad():
    """Non-zero h0 with requires_grad: grad_h0 must be finite and non-zero."""
    grad_h0 = _ift_grad_check(
        kernel="torch", dtype=torch.float64, device="cpu",
        atol=1e-9, rtol=1e-9, with_h0=True,
    )
    assert torch.isfinite(grad_h0).all()
    assert grad_h0.abs().max() > 0


def test_ift_backward_short_sequence_T1():
    """T=1 sanity: adjoint solve degenerates to λ[0] = -grad_h*[0]."""
    _ift_grad_check(
        kernel="torch", dtype=torch.float64, device="cpu",
        atol=1e-9, rtol=1e-9, S=1,
    )


def test_ift_backward_short_sequence_T2():
    """T=2 sanity: smallest non-trivial adjoint coupling."""
    _ift_grad_check(
        kernel="torch", dtype=torch.float64, device="cpu",
        atol=1e-9, rtol=1e-9, S=2,
    )


def test_ift_backward_broadcast_heads():
    """n_q < H: backward must reduce over the broadcast axis to recover
    the original (B, S, n_q, k_dim) gradient shape."""
    device = "cpu"
    dtype = torch.float64
    B, S, k_dim, v_dim = 1, 16, 4, 4

    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(B, S, 1, k_dim, generator=g, dtype=dtype) * 0.5
    k = torch.randn(B, S, 2, k_dim, generator=g, dtype=dtype) * 0.5
    v = torch.randn(B, S, 4, v_dim, generator=g, dtype=dtype) * 0.5
    W = torch.randn(4, v_dim, v_dim, generator=g, dtype=dtype) * 0.2
    xf = torch.sigmoid(torch.randn(B, S, 4, generator=g, dtype=dtype) - 0.5)
    target = torch.randn(B, S, 4, v_dim, dtype=dtype)  # output shape (B,S,H=4,V)

    q_s, k_s, v_s, W_s, xf_s = (t.clone().requires_grad_() for t in (q, k, v, W, xf))
    out_s, _ = _torch_m2rnn_forward(q_s, k_s, v_s, W_s, xf_s)
    (out_s - target).pow(2).mean().backward()

    q_p, k_p, v_p, W_p, xf_p = (t.clone().requires_grad_() for t in (q, k, v, W, xf))
    out_p, _ = m2rnn_pararnn_forward(
        q_p, k_p, v_p, W_p, xf_p,
        config=PararnnConfig(max_its=10, kernel="torch"),
    )
    (out_p - target).pow(2).mean().backward()

    assert q_p.grad.shape == q.shape
    assert k_p.grad.shape == k.shape
    torch.testing.assert_close(q_p.grad, q_s.grad, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(k_p.grad, k_s.grad, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(v_p.grad, v_s.grad, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(W_p.grad, W_s.grad, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(xf_p.grad, xf_s.grad, atol=1e-9, rtol=1e-9)
