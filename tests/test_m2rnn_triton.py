"""Parity tests for the fused Triton M²RNN kernel.

Validates ``m2rnn_scan_triton`` against a standalone copy of
``_torch_m2rnn_forward`` (the slow sequential reference from
``cppmega/megatron/m2rnn_spec.py``).  A CUDA GPU with Triton installed
is required; tests skip cleanly on CPU-only hosts.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Standalone reference (copied verbatim from m2rnn_spec.py so the test
# doesn't pull in megatron).
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
# Fixtures / utilities
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton M²RNN kernel requires CUDA",
)


def _make_inputs(B, S, H, K, V, *, dtype=torch.bfloat16, device="cuda", seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, K, device=device, dtype=dtype, generator=g)
    k = torch.randn(B, S, H, K, device=device, dtype=dtype, generator=g)
    v = torch.randn(B, S, H, V, device=device, dtype=dtype, generator=g)
    W = torch.eye(V, device=device, dtype=dtype).unsqueeze(0).expand(H, -1, -1).contiguous().clone()
    W += 0.05 * torch.randn(W.shape, device=device, dtype=dtype, generator=g)
    xf = torch.sigmoid(torch.randn(B, S, H, device=device, dtype=dtype, generator=g))
    return q, k, v, W, xf


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------


class TestFwdParity:
    def _check(self, B, S, H, K, V, *, atol_out=5e-2, atol_h=5e-2):
        try:
            import triton  # noqa: F401
        except ImportError:
            pytest.skip("triton not available")
        from cppmega.megatron.m2rnn_triton import m2rnn_scan_triton

        q, k, v, W, xf = _make_inputs(B, S, H, K, V)

        out_ref, h_ref = _torch_m2rnn_forward(q, k, v, W, xf)
        out_tri, h_tri = m2rnn_scan_triton(q, k, v, W, xf)

        max_out = (out_ref.float() - out_tri.float()).abs().max().item()
        max_h = (h_ref.float() - h_tri.float()).abs().max().item()
        assert max_out < atol_out, f"max_out={max_out}"
        assert max_h < atol_h, f"max_h={max_h}"

    def test_smoke_small(self):
        # bf16 accumulation over 64 steps; the reference is *also* in bf16
        # so we're bounded by mutual bf16 noise, not kernel error.
        self._check(B=2, S=64, H=2, K=16, V=16, atol_out=1e-1, atol_h=5e-2)

    def test_smoke_medium(self):
        self._check(B=2, S=256, H=4, K=32, V=16, atol_out=2e-1, atol_h=1e-1)

    def test_production_shape(self):
        # Real NAM56R M²RNN dims: K=64, V=16, H≈8.  S=4096 bf16 recurrence.
        self._check(B=2, S=4096, H=8, K=64, V=16, atol_out=5e-1, atol_h=3e-1)

    def test_fp32_smoke(self):
        """fp32 path: checks the kernel math directly (no bf16 noise)."""
        try:
            import triton  # noqa: F401
        except ImportError:
            pytest.skip("triton not available")
        from cppmega.megatron.m2rnn_triton import m2rnn_scan_triton

        q, k, v, W, xf = _make_inputs(2, 128, 2, 16, 16, dtype=torch.float32)
        out_ref, h_ref = _torch_m2rnn_forward(q, k, v, W, xf)
        out_tri, h_tri = m2rnn_scan_triton(q, k, v, W, xf)
        max_out = (out_ref - out_tri).abs().max().item()
        max_h = (h_ref - h_tri).abs().max().item()
        assert max_out < 1e-2, f"fp32 out max_abs={max_out}"
        assert max_h < 1e-2, f"fp32 h max_abs={max_h}"


# ---------------------------------------------------------------------------
# Backward parity
# ---------------------------------------------------------------------------


class TestBwdParity:
    def _check_bwd(self, B, S, H, K, V, *, rtol=1e-2):
        """Bwd parity: Triton (bf16) vs Reference (fp32, bf16 inputs cast up).

        We deliberately compare the Triton bf16 kernel against an fp32
        reference because the reference's ``torch.matmul`` bwd accumulates
        dW in bf16 ``h_prev.T @ d_pre`` and rounds aggressively — for
        S=4096 that is an order of magnitude noisier than the Triton
        kernel which keeps the dW accumulator in fp32 registers for the
        entire sequence.  The fp32 vs fp32 parity test below shows the
        kernel's own math error is in the 1e-6 range.
        """
        try:
            import triton  # noqa: F401
        except ImportError:
            pytest.skip("triton not available")
        from cppmega.megatron.m2rnn_triton import m2rnn_scan_triton

        q0, k0, v0, W0, xf0 = _make_inputs(B, S, H, K, V, dtype=torch.bfloat16)

        def leaves(src, dtype=None):
            return [
                (x if dtype is None else x.to(dtype)).detach().clone().requires_grad_(True)
                for x in src
            ]

        # Triton (bf16)
        q2, k2, v2, W2, xf2 = leaves([q0, k0, v0, W0, xf0])
        out_tri, _ = m2rnn_scan_triton(q2, k2, v2, W2, xf2)

        # Reference (fp32 with bf16 inputs cast up)
        q3, k3, v3, W3, xf3 = leaves([q0, k0, v0, W0, xf0], dtype=torch.float32)
        out_ref32, _ = _torch_m2rnn_forward(q3, k3, v3, W3, xf3)

        g_bf16 = torch.randn_like(out_tri)
        (out_tri * g_bf16).sum().backward()
        (out_ref32 * g_bf16.float()).sum().backward()

        def rel(name, tri_grad, ref_grad):
            diff = (tri_grad.float() - ref_grad).abs().max().item()
            mag = ref_grad.abs().max().item() + 1e-12
            r = diff / mag
            assert r < rtol, f"{name}: rel={r:.4e} (abs={diff:.4e}, mag={mag:.4e})"

        rel("dq", q2.grad, q3.grad)
        rel("dk", k2.grad, k3.grad)
        rel("dv", v2.grad, v3.grad)
        rel("dW", W2.grad, W3.grad)
        rel("dxf", xf2.grad, xf3.grad)

    def test_bwd_smoke(self):
        self._check_bwd(B=2, S=64, H=2, K=16, V=16, rtol=1e-2)

    def test_bwd_medium(self):
        self._check_bwd(B=2, S=256, H=4, K=32, V=16, rtol=1e-2)

    def test_bwd_production_shape(self):
        # Real NAM56R dims.  Triton keeps fp32 accumulators so it beats
        # the bf16 reference — the relative error bound here is Triton's
        # own bf16 I/O rounding, not accumulation noise.
        self._check_bwd(B=2, S=4096, H=8, K=64, V=16, rtol=1e-2)

    def test_bwd_broadcast_heads(self):
        """Regression: when n_q < n_max (heads broadcast via repeat_interleave),
        the bwd must reduce dq back to n_q heads — not collapse the K axis.

        This reproduces the production NAM56R-half failure where
        ``n_q = num_q_heads = 1`` got broadcast to ``n = 14`` heads inside the
        kernel and ``_unbroadcast_heads`` summed over the wrong axis (using
        a negative ``dim + 1`` that landed on K, not the freshly-inserted
        group axis), so dq came back as ``[B, S, 1, n=14]`` instead of
        ``[B, S, 1, K=64]``.  Forward stayed correct because the kernel
        already produces ``out`` shaped ``[B, S, n, V]`` and out-broadcast
        is not exercised on the forward path.
        """
        try:
            import triton  # noqa: F401
        except ImportError:
            pytest.skip("triton not available")
        from cppmega.megatron.m2rnn_triton import m2rnn_scan_triton

        B, S, K, V = 1, 64, 64, 16
        n_q = 1   # production: num_q_heads = 1
        n_k = 1
        n_v = 14  # n_max comes from one of the multi-head fields
        n_w = 14
        n_f = 14
        device = "cuda"
        dtype = torch.float32  # fp32 → tight bounds, isolates the shape bug
        g = torch.Generator(device=device).manual_seed(7)
        q = torch.randn(B, S, n_q, K, device=device, dtype=dtype, generator=g, requires_grad=True)
        k = torch.randn(B, S, n_k, K, device=device, dtype=dtype, generator=g, requires_grad=True)
        v = torch.randn(B, S, n_v, V, device=device, dtype=dtype, generator=g, requires_grad=True)
        W = torch.eye(V, device=device, dtype=dtype).unsqueeze(0).expand(n_w, -1, -1).contiguous().clone()
        W += 0.05 * torch.randn(W.shape, device=device, dtype=dtype, generator=g)
        W.requires_grad_(True)
        xf = torch.sigmoid(torch.randn(B, S, n_f, device=device, dtype=dtype, generator=g))
        xf.requires_grad_(True)

        out, _ = m2rnn_scan_triton(q, k, v, W, xf)
        # Sanity: out shape uses n_max
        assert out.shape == (B, S, max(n_q, n_k, n_v, n_w, n_f), V)
        out.sum().backward()

        # The whole point of the regression: dq must keep the original K dim
        # and the original n_q head count.
        assert q.grad.shape == q.shape, f"dq shape {q.grad.shape} != q shape {q.shape}"
        assert k.grad.shape == k.shape, f"dk shape {k.grad.shape} != k shape {k.shape}"
        assert v.grad.shape == v.shape, f"dv shape {v.grad.shape} != v shape {v.shape}"
        assert W.grad.shape == W.shape, f"dW shape {W.grad.shape} != W shape {W.shape}"
        assert xf.grad.shape == xf.shape, f"dxf shape {xf.grad.shape} != xf shape {xf.shape}"

    def test_bwd_fp32(self):
        """fp32 bwd parity — tight bound showing the kernel math is correct."""
        try:
            import triton  # noqa: F401
        except ImportError:
            pytest.skip("triton not available")
        from cppmega.megatron.m2rnn_triton import m2rnn_scan_triton

        B, S, H, K, V = 2, 128, 2, 16, 16
        q0, k0, v0, W0, xf0 = _make_inputs(B, S, H, K, V, dtype=torch.float32)

        def leaves(src):
            return [x.detach().clone().requires_grad_(True) for x in src]

        q1, k1, v1, W1, xf1 = leaves([q0, k0, v0, W0, xf0])
        q2, k2, v2, W2, xf2 = leaves([q0, k0, v0, W0, xf0])

        out_ref, _ = _torch_m2rnn_forward(q1, k1, v1, W1, xf1)
        out_tri, _ = m2rnn_scan_triton(q2, k2, v2, W2, xf2)

        g = torch.randn_like(out_ref)
        (out_ref * g).sum().backward()
        (out_tri * g).sum().backward()

        def rel(a, b):
            denom = a.abs().max().item() + 1e-12
            return (a - b).abs().max().item() / denom

        assert rel(q1.grad, q2.grad) < 1e-4
        assert rel(k1.grad, k2.grad) < 1e-4
        assert rel(v1.grad, v2.grad) < 1e-4
        assert rel(W1.grad, W2.grad) < 1e-4
        assert rel(xf1.grad, xf2.grad) < 1e-4
