"""Tests for the chunked forward Triton M2RNN kernel.

Validates ``m2rnn_scan_triton_chunked`` against both the persistent
``m2rnn_scan_triton`` kernel and the sequential PyTorch reference.

A CUDA GPU with Triton installed is required; tests skip cleanly on
CPU-only hosts.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Standalone sequential reference (copied verbatim from m2rnn_spec.py).
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


def _make_inputs(B, S, H, K, V, *, dtype=torch.bfloat16, device="cuda", seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, K, device=device, dtype=dtype, generator=g)
    k = torch.randn(B, S, H, K, device=device, dtype=dtype, generator=g)
    v = torch.randn(B, S, H, V, device=device, dtype=dtype, generator=g)
    W = (
        torch.eye(V, device=device, dtype=dtype)
        .unsqueeze(0)
        .expand(H, -1, -1)
        .contiguous()
        .clone()
    )
    W += 0.05 * torch.randn(W.shape, device=device, dtype=dtype, generator=g)
    xf = torch.sigmoid(torch.randn(B, S, H, device=device, dtype=dtype, generator=g))
    return q, k, v, W, xf


def _triton_skip_if_not_available():
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("triton not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------------------------------------------------------------
# Forward parity: chunked vs persistent Triton kernel
# ---------------------------------------------------------------------------


class TestFwdChunkedVsPersistent:
    """Ensure the chunked forward produces the same output as the persistent
    forward kernel (same math, same checkpoints, same output)."""

    def _check(self, B, S, H, K, V, fwd_chunk_size, *, atol_out=5e-2, atol_h=5e-2):
        _triton_skip_if_not_available()
        import cppmega.megatron.m2rnn_triton as _mod

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        q, k, v, W, xf = _make_inputs(B, S, H, K, V)

        out_ref, h_ref = _mod.m2rnn_scan_triton(q, k, v, W, xf)

        import os
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", str(fwd_chunk_size))
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q, k, v, W, xf)
        finally:
            monkeypatch.undo()

        max_out = (out_ref.float() - out_chunked.float()).abs().max().item()
        max_h = (h_ref.float() - h_chunked.float()).abs().max().item()
        assert max_out < atol_out, f"max_out={max_out} for FWD_CHUNK_SIZE={fwd_chunk_size}"
        assert max_h < atol_h, f"max_h={max_h} for FWD_CHUNK_SIZE={fwd_chunk_size}"

    @pytest.mark.parametrize("fwd_chunk_size", [32, 64, 128, 256])
    def test_various_chunk_sizes_small(self, fwd_chunk_size):
        self._check(B=2, S=128, H=2, K=16, V=16, fwd_chunk_size=fwd_chunk_size,
                    atol_out=1e-1, atol_h=5e-2)

    @pytest.mark.parametrize("fwd_chunk_size", [32, 64, 128, 256])
    def test_various_chunk_sizes_medium(self, fwd_chunk_size):
        self._check(B=2, S=256, H=4, K=32, V=16, fwd_chunk_size=fwd_chunk_size,
                    atol_out=2e-1, atol_h=1e-1)

    def test_production_shape_chunked(self):
        self._check(B=2, S=4096, H=8, K=64, V=16, fwd_chunk_size=128,
                    atol_out=5e-1, atol_h=3e-1)

    def test_fp32_smoke(self):
        """fp32 path: checks the kernel math directly (no bf16 noise)."""
        _triton_skip_if_not_available()
        import cppmega.megatron.m2rnn_triton as _mod

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        q, k, v, W, xf = _make_inputs(2, 128, 2, 16, 16, dtype=torch.float32)

        out_ref, h_ref = _mod.m2rnn_scan_triton(q, k, v, W, xf)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "32")
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q, k, v, W, xf)
        finally:
            monkeypatch.undo()

        max_out = (out_ref - out_chunked).abs().max().item()
        max_h = (h_ref - h_chunked).abs().max().item()
        assert max_out < 1e-2, f"fp32 out max_abs={max_out}"
        assert max_h < 1e-2, f"fp32 h max_abs={max_h}"

    def test_seq_not_divisible_by_chunk_size(self):
        """Sequence length not a multiple of chunk size."""
        _triton_skip_if_not_available()
        import cppmega.megatron.m2rnn_triton as _mod

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        q, k, v, W, xf = _make_inputs(2, 100, 2, 16, 16)

        out_ref, h_ref = _mod.m2rnn_scan_triton(q, k, v, W, xf)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "32")
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q, k, v, W, xf)
        finally:
            monkeypatch.undo()

        max_out = (out_ref.float() - out_chunked.float()).abs().max().item()
        max_h = (h_ref.float() - h_chunked.float()).abs().max().item()
        assert max_out < 1e-1, f"max_out={max_out}"
        assert max_h < 5e-2, f"max_h={max_h}"


# ---------------------------------------------------------------------------
# Forward parity: chunked vs PyTorch sequential reference
# ---------------------------------------------------------------------------


class TestFwdChunkedVsReference:
    """Ensure the chunked Triton forward matches the sequential PyTorch reference."""

    def _check(self, B, S, H, K, V, fwd_chunk_size, *, atol_out=5e-2, atol_h=5e-2):
        _triton_skip_if_not_available()

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        q, k, v, W, xf = _make_inputs(B, S, H, K, V)

        out_ref, h_ref = _torch_m2rnn_forward(q, k, v, W, xf)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", str(fwd_chunk_size))
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q, k, v, W, xf)
        finally:
            monkeypatch.undo()

        max_out = (out_ref.float() - out_chunked.float()).abs().max().item()
        max_h = (h_ref.float() - h_chunked.float()).abs().max().item()
        assert max_out < atol_out, f"max_out={max_out} for FWD_CHUNK_SIZE={fwd_chunk_size}"
        assert max_h < atol_h, f"max_h={max_h} for FWD_CHUNK_SIZE={fwd_chunk_size}"

    def test_small_vs_reference(self):
        self._check(B=2, S=64, H=2, K=16, V=16, fwd_chunk_size=16,
                    atol_out=1e-1, atol_h=5e-2)

    def test_medium_vs_reference(self):
        self._check(B=2, S=256, H=4, K=32, V=16, fwd_chunk_size=64,
                    atol_out=2e-1, atol_h=1e-1)


# ---------------------------------------------------------------------------
# Forward: with and without initial state h0
# ---------------------------------------------------------------------------


class TestFwdChunkedWithH0:
    """Test that the chunked kernel handles initial state correctly."""

    def test_with_h0(self):
        _triton_skip_if_not_available()
        import cppmega.megatron.m2rnn_triton as _mod

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        B, S, H, K, V = 2, 64, 2, 16, 16
        q, k, v, W, xf = _make_inputs(B, S, H, K, V, dtype=torch.float32)
        g = torch.Generator(device="cuda").manual_seed(99)
        h0 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32, generator=g)

        out_ref, h_ref = _mod.m2rnn_scan_triton(q, k, v, W, xf, h0=h0)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "32")
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q, k, v, W, xf, h0=h0)
        finally:
            monkeypatch.undo()

        max_out = (out_ref - out_chunked).abs().max().item()
        max_h = (h_ref - h_chunked).abs().max().item()
        assert max_out < 1e-2, f"max_out={max_out}"
        assert max_h < 1e-2, f"max_h={max_h}"

    def test_without_h0(self):
        _triton_skip_if_not_available()
        import cppmega.megatron.m2rnn_triton as _mod

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        B, S, H, K, V = 2, 64, 2, 16, 16
        q, k, v, W, xf = _make_inputs(B, S, H, K, V, dtype=torch.float32)

        out_ref, h_ref = _mod.m2rnn_scan_triton(q, k, v, W, xf, h0=None)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "32")
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q, k, v, W, xf, h0=None)
        finally:
            monkeypatch.undo()

        max_out = (out_ref - out_chunked).abs().max().item()
        max_h = (h_ref - h_chunked).abs().max().item()
        assert max_out < 1e-2, f"max_out={max_out}"
        assert max_h < 1e-2, f"max_h={max_h}"

    def test_h0_zeros_vs_none_fp32(self):
        """h0=zeros should produce identical results to h0=None."""
        _triton_skip_if_not_available()

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        B, S, H, K, V = 1, 32, 2, 16, 16
        q, k, v, W, xf = _make_inputs(B, S, H, K, V, dtype=torch.float32)
        h0_zeros = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "8")
        try:
            out_none, h_none = m2rnn_scan_triton_chunked(q, k, v, W, xf, h0=None)
            out_zero, h_zero = m2rnn_scan_triton_chunked(q, k, v, W, xf, h0=h0_zeros)
        finally:
            monkeypatch.undo()

        torch.testing.assert_close(out_none, out_zero, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h_none, h_zero, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Backward parity: chunked vs sequential reference
# ---------------------------------------------------------------------------


class TestBwdChunkedParity:
    """Test that the chunked forward produces correct backward gradients."""

    def _check_bwd(self, B, S, H, K, V, fwd_chunk_size, *, rtol=1e-2):
        _triton_skip_if_not_available()

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        q0, k0, v0, W0, xf0 = _make_inputs(B, S, H, K, V, dtype=torch.bfloat16)

        def leaves(src, dtype=None):
            return [
                (x if dtype is None else x.to(dtype)).detach().clone().requires_grad_(True)
                for x in src
            ]

        # Chunked Triton (bf16)
        q1, k1, v1, W1, xf1 = leaves([q0, k0, v0, W0, xf0])

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", str(fwd_chunk_size))
        try:
            out_chunked, _ = m2rnn_scan_triton_chunked(q1, k1, v1, W1, xf1)
        finally:
            monkeypatch.undo()

        # Reference (fp32 with bf16 inputs cast up)
        q2, k2, v2, W2, xf2 = leaves([q0, k0, v0, W0, xf0], dtype=torch.float32)
        out_ref32, _ = _torch_m2rnn_forward(q2, k2, v2, W2, xf2)

        g_bf16 = torch.randn_like(out_chunked)
        (out_chunked * g_bf16).sum().backward()
        (out_ref32 * g_bf16.float()).sum().backward()

        def rel(name, tri_grad, ref_grad):
            diff = (tri_grad.float() - ref_grad).abs().max().item()
            mag = ref_grad.abs().max().item() + 1e-12
            r = diff / mag
            assert r < rtol, f"{name}: rel={r:.4e} (abs={diff:.4e}, mag={mag:.4e})"

        rel("dq", q1.grad, q2.grad)
        rel("dk", k1.grad, k2.grad)
        rel("dv", v1.grad, v2.grad)
        rel("dW", W1.grad, W2.grad)
        rel("dxf", xf1.grad, xf2.grad)

    def test_bwd_smoke(self):
        self._check_bwd(B=2, S=64, H=2, K=16, V=16, fwd_chunk_size=16, rtol=1e-2)

    def test_bwd_medium(self):
        self._check_bwd(B=2, S=256, H=4, K=32, V=16, fwd_chunk_size=64, rtol=1e-2)

    def test_bwd_with_h0(self):
        """Gradient flows back to h0 correctly."""
        _triton_skip_if_not_available()

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        B, S, H, K, V = 2, 64, 2, 16, 16
        q0, k0, v0, W0, xf0 = _make_inputs(B, S, H, K, V, dtype=torch.float32, seed=11)
        g = torch.Generator(device="cuda").manual_seed(12)
        h00 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32, generator=g)

        def leaves(src):
            return [x.detach().clone().requires_grad_(True) for x in src]

        # Reference
        q1, k1, v1, W1, xf1, h01 = leaves([q0, k0, v0, W0, xf0, h00])
        out_ref, h_ref = _torch_m2rnn_forward(q1, k1, v1, W1, xf1, h0=h01)

        # Chunked Triton
        q2, k2, v2, W2, xf2, h02 = leaves([q0, k0, v0, W0, xf0, h00])

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "32")
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q2, k2, v2, W2, xf2, h0=h02)
        finally:
            monkeypatch.undo()

        g_out = torch.randn(out_ref.shape, device=out_ref.device, dtype=out_ref.dtype,
                            generator=torch.Generator(device="cuda").manual_seed(13))
        g_h = torch.randn(h_ref.shape, device=h_ref.device, dtype=h_ref.dtype,
                          generator=torch.Generator(device="cuda").manual_seed(14))
        ((out_ref * g_out).sum() + (h_ref * g_h).sum()).backward()
        ((out_chunked * g_out).sum() + (h_chunked * g_h).sum()).backward()

        def rel(a, b):
            denom = a.abs().max().item() + 1e-12
            return (a - b).abs().max().item() / denom

        assert rel(q1.grad, q2.grad) < 1e-4, f"dq mismatch: {rel(q1.grad, q2.grad):.4e}"
        assert rel(k1.grad, k2.grad) < 1e-4, f"dk mismatch: {rel(k1.grad, k2.grad):.4e}"
        assert rel(v1.grad, v2.grad) < 1e-4, f"dv mismatch: {rel(v1.grad, v2.grad):.4e}"
        assert rel(W1.grad, W2.grad) < 1e-4, f"dW mismatch: {rel(W1.grad, W2.grad):.4e}"
        assert rel(xf1.grad, xf2.grad) < 1e-4, f"dxf mismatch: {rel(xf1.grad, xf2.grad):.4e}"
        assert rel(h01.grad, h02.grad) < 1e-4, f"dh0 mismatch: {rel(h01.grad, h02.grad):.4e}"


# ---------------------------------------------------------------------------
# Backward parity: chunked vs persistent Triton
# ---------------------------------------------------------------------------


class TestBwdChunkedVsPersistent:
    """Verify the chunked forward + existing backward produces the same
    gradients as the persistent forward + existing backward."""

    def test_bwd_chunked_vs_persistent(self):
        _triton_skip_if_not_available()
        import cppmega.megatron.m2rnn_triton as _mod

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        B, S, H, K, V = 2, 128, 2, 16, 16
        q0, k0, v0, W0, xf0 = _make_inputs(B, S, H, K, V, dtype=torch.float32, seed=42)

        def leaves(src):
            return [x.detach().clone().requires_grad_(True) for x in src]

        # Persistent
        q1, k1, v1, W1, xf1 = leaves([q0, k0, v0, W0, xf0])
        out_pers, _ = _mod.m2rnn_scan_triton(q1, k1, v1, W1, xf1)

        # Chunked
        q2, k2, v2, W2, xf2 = leaves([q0, k0, v0, W0, xf0])

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "32")
        try:
            out_chunked, _ = m2rnn_scan_triton_chunked(q2, k2, v2, W2, xf2)
        finally:
            monkeypatch.undo()

        g = torch.randn(out_pers.shape, device=out_pers.device, dtype=out_pers.dtype,
                        generator=torch.Generator(device="cuda").manual_seed(99))
        (out_pers * g).sum().backward()
        (out_chunked * g).sum().backward()

        def rel(a, b):
            denom = a.abs().max().item() + 1e-12
            return (a - b).abs().max().item() / denom

        assert rel(q1.grad, q2.grad) < 1e-4, f"dq mismatch: {rel(q1.grad, q2.grad):.4e}"
        assert rel(k1.grad, k2.grad) < 1e-4, f"dk mismatch: {rel(k1.grad, k2.grad):.4e}"
        assert rel(v1.grad, v2.grad) < 1e-4, f"dv mismatch: {rel(v1.grad, v2.grad):.4e}"
        assert rel(W1.grad, W2.grad) < 1e-4, f"dW mismatch: {rel(W1.grad, W2.grad):.4e}"
        assert rel(xf1.grad, xf2.grad) < 1e-4, f"dxf mismatch: {rel(xf1.grad, xf2.grad):.4e}"


# ---------------------------------------------------------------------------
# Manual gradient correctness (replaces torch.autograd.gradcheck which
# fails due to tanh.approx + fp32 internals with fp64 inputs).
# ---------------------------------------------------------------------------


class TestGradCorrectness:
    """Validate gradient correctness via manual comparison against the
    sequential fp32 reference (same methodology as test_m2rnn_triton.py)."""

    def test_grads_small_fp32(self):
        """Small fp32 test: all grads must match reference within tight bounds."""
        _triton_skip_if_not_available()

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        B, S, H, K, V = 2, 32, 2, 16, 16
        q0, k0, v0, W0, xf0 = _make_inputs(B, S, H, K, V, dtype=torch.float32, seed=1)

        def leaves(src):
            return [x.detach().clone().requires_grad_(True) for x in src]

        q1, k1, v1, W1, xf1 = leaves([q0, k0, v0, W0, xf0])
        q2, k2, v2, W2, xf2 = leaves([q0, k0, v0, W0, xf0])

        out_ref, _ = _torch_m2rnn_forward(q1, k1, v1, W1, xf1)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "8")
        try:
            out_chunked, _ = m2rnn_scan_triton_chunked(q2, k2, v2, W2, xf2)
        finally:
            monkeypatch.undo()

        g = torch.randn(out_ref.shape, device=out_ref.device, dtype=out_ref.dtype,
                        generator=torch.Generator(device="cuda").manual_seed(99))
        (out_ref * g).sum().backward()
        (out_chunked * g).sum().backward()

        def rel(a, b):
            denom = a.abs().max().item() + 1e-12
            return (a - b).abs().max().item() / denom

        assert rel(q1.grad, q2.grad) < 1e-4, f"dq mismatch: {rel(q1.grad, q2.grad):.4e}"
        assert rel(k1.grad, k2.grad) < 1e-4, f"dk mismatch: {rel(k1.grad, k2.grad):.4e}"
        assert rel(v1.grad, v2.grad) < 1e-4, f"dv mismatch: {rel(v1.grad, v2.grad):.4e}"
        assert rel(W1.grad, W2.grad) < 1e-4, f"dW mismatch: {rel(W1.grad, W2.grad):.4e}"
        assert rel(xf1.grad, xf2.grad) < 1e-4, f"dxf mismatch: {rel(xf1.grad, xf2.grad):.4e}"

    def test_grads_with_h0_fp32(self):
        """Gradient through initial state h0 must match reference."""
        _triton_skip_if_not_available()

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        B, S, H, K, V = 1, 16, 2, 16, 16
        q0, k0, v0, W0, xf0 = _make_inputs(B, S, H, K, V, dtype=torch.float32, seed=2)
        g = torch.Generator(device="cuda").manual_seed(3)
        h00 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32, generator=g)

        def leaves(src):
            return [x.detach().clone().requires_grad_(True) for x in src]

        q1, k1, v1, W1, xf1, h01 = leaves([q0, k0, v0, W0, xf0, h00])
        q2, k2, v2, W2, xf2, h02 = leaves([q0, k0, v0, W0, xf0, h00])

        out_ref, h_ref = _torch_m2rnn_forward(q1, k1, v1, W1, xf1, h0=h01)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "8")
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q2, k2, v2, W2, xf2, h0=h02)
        finally:
            monkeypatch.undo()

        g_out = torch.randn(out_ref.shape, device=out_ref.device, dtype=out_ref.dtype,
                            generator=torch.Generator(device="cuda").manual_seed(10))
        g_h = torch.randn(h_ref.shape, device=h_ref.device, dtype=h_ref.dtype,
                          generator=torch.Generator(device="cuda").manual_seed(11))
        ((out_ref * g_out).sum() + (h_ref * g_h).sum()).backward()
        ((out_chunked * g_out).sum() + (h_chunked * g_h).sum()).backward()

        def rel(a, b):
            denom = a.abs().max().item() + 1e-12
            return (a - b).abs().max().item() / denom

        assert rel(q1.grad, q2.grad) < 1e-4, f"dq mismatch: {rel(q1.grad, q2.grad):.4e}"
        assert rel(k1.grad, k2.grad) < 1e-4, f"dk mismatch: {rel(k1.grad, k2.grad):.4e}"
        assert rel(v1.grad, v2.grad) < 1e-4, f"dv mismatch: {rel(v1.grad, v2.grad):.4e}"
        assert rel(W1.grad, W2.grad) < 1e-4, f"dW mismatch: {rel(W1.grad, W2.grad):.4e}"
        assert rel(xf1.grad, xf2.grad) < 1e-4, f"dxf mismatch: {rel(xf1.grad, xf2.grad):.4e}"
        assert rel(h01.grad, h02.grad) < 1e-4, f"dh0 mismatch: {rel(h01.grad, h02.grad):.4e}"


# ---------------------------------------------------------------------------
# Checkpoint memory mode
# ---------------------------------------------------------------------------


class TestChunkedCheckpointedBackward:
    """Verify that the chunked forward allocates checkpoints (not full y)."""

    def test_forward_allocates_checkpoints_not_full_y(self):
        _triton_skip_if_not_available()

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked

        B, S, H, K, V = 1, 33, 2, 16, 16
        q, k, v, W, xf = _make_inputs(B, S, H, K, V, dtype=torch.float32, seed=21)

        allocations = []
        orig_empty = torch.empty

        def recording_empty(*args, **kwargs):
            if args:
                raw_shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)) else args
            else:
                raw_shape = kwargs.get("size", ())
            try:
                shape = tuple(int(x) for x in raw_shape)
            except TypeError:
                shape = None
            allocations.append((shape, kwargs.get("dtype")))
            return orig_empty(*args, **kwargs)

        import cppmega.megatron.m2rnn_chunk_triton
        _orig_empty = torch.empty
        torch.empty = recording_empty
        import cppmega.megatron.m2rnn_triton as _mod_tri
        _mod_tri_empty = torch.empty
        # Also patch in m2rnn_triton module
        # Actually the recording_empty is already set on torch.empty, so all calls go through it

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "16")
        monkeypatch.setenv("CPPMEGA_M2RNN_SAVE_HNEW", "0")
        monkeypatch.setenv("CPPMEGA_M2RNN_BWD_CHUNK_SIZE", "8")
        monkeypatch.setattr(torch, "empty", recording_empty)
        try:
            _mod_tri.reset_m2rnn_runtime_config_cache()
            out, h_final = m2rnn_scan_triton_chunked(q, k, v, W, xf)
            torch.cuda.synchronize()
        finally:
            monkeypatch.undo()
            torch.empty = orig_empty

        shapes = [shape for shape, _dtype in allocations]
        full_y_shape = (B, S, H, K, V)
        ckpt_shape = (B, (S + 7) // 8 + 1, H, K, V)
        assert full_y_shape not in shapes, (
            f"Full y tensor {full_y_shape} should not be allocated, got shapes: {shapes}"
        )
        assert ckpt_shape in shapes, (
            f"Checkpoint tensor {ckpt_shape} should be allocated, got shapes: {shapes}"
        )
        assert out.shape == (B, S, H, V)
        assert h_final.shape == (B, H, K, V)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestChunkedEdgeCases:
    """Edge cases and boundary conditions for the chunked kernel."""

    def test_seq_len_one(self):
        _triton_skip_if_not_available()

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked
        import cppmega.megatron.m2rnn_triton as _mod

        q, k, v, W, xf = _make_inputs(1, 1, 1, 16, 16, dtype=torch.float32)
        out_ref, h_ref = _mod.m2rnn_scan_triton(q, k, v, W, xf)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "128")
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q, k, v, W, xf)
        finally:
            monkeypatch.undo()

        torch.testing.assert_close(out_chunked, out_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h_chunked, h_ref, atol=1e-5, rtol=1e-5)

    def test_chunk_size_one(self):
        """FWD_CHUNK_SIZE=1 should degenerate to fully sequential."""
        _triton_skip_if_not_available()

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked
        import cppmega.megatron.m2rnn_triton as _mod

        q, k, v, W, xf = _make_inputs(1, 32, 1, 16, 16, dtype=torch.float32)
        out_ref, h_ref = _mod.m2rnn_scan_triton(q, k, v, W, xf)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "1")
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q, k, v, W, xf)
        finally:
            monkeypatch.undo()

        torch.testing.assert_close(out_chunked, out_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h_chunked, h_ref, atol=1e-5, rtol=1e-5)

    def test_chunk_size_equals_seq(self):
        """When FWD_CHUNK_SIZE >= SEQ, single chunk matches persistent."""
        _triton_skip_if_not_available()

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked
        import cppmega.megatron.m2rnn_triton as _mod

        q, k, v, W, xf = _make_inputs(1, 32, 1, 16, 16, dtype=torch.float32)
        out_ref, h_ref = _mod.m2rnn_scan_triton(q, k, v, W, xf)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "64")
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q, k, v, W, xf)
        finally:
            monkeypatch.undo()

        torch.testing.assert_close(out_chunked, out_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h_chunked, h_ref, atol=1e-5, rtol=1e-5)

    def test_batch_head_broadcast(self):
        """Test with different head counts (broadcasting)."""
        _triton_skip_if_not_available()

        from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked
        import cppmega.megatron.m2rnn_triton as _mod

        B, S = 1, 32
        n_q, n_k, n_v, n_w, n_f = 1, 1, 4, 1, 4
        K, V = 16, 16
        device = "cuda"
        dtype = torch.float32
        g = torch.Generator(device=device).manual_seed(123)

        q = torch.randn(B, S, n_q, K, device=device, dtype=dtype, generator=g)
        k = torch.randn(B, S, n_k, K, device=device, dtype=dtype, generator=g)
        v = torch.randn(B, S, n_v, V, device=device, dtype=dtype, generator=g)
        W = torch.eye(V, device=device, dtype=dtype).unsqueeze(0).expand(n_w, -1, -1).clone()
        xf = torch.rand(B, S, n_f, device=device, dtype=dtype, generator=g)

        out_ref, h_ref = _mod.m2rnn_scan_triton(q, k, v, W, xf)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("CPPMEGA_M2RNN_FWD_CHUNK_SIZE", "16")
        try:
            out_chunked, h_chunked = m2rnn_scan_triton_chunked(q, k, v, W, xf)
        finally:
            monkeypatch.undo()

        torch.testing.assert_close(out_chunked, out_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h_chunked, h_ref, atol=1e-5, rtol=1e-5)
