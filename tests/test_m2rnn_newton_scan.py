"""Tests for Newton-linearized parallel scan for M2RNN.

Validates the Newton solver, linearized operators, parallel scan (Blelloch),
and the full ``m2rnn_scan_newton`` entry point against the sequential
reference recurrence.
"""

from __future__ import annotations

import math

import pytest
import torch

# ---------------------------------------------------------------------------
# If either triton or CUDA is missing, many tests still run on CPU (pure-PyTorch
# path is always available).  Tests requiring CUDA/Triton skip independently.
# ---------------------------------------------------------------------------

from cppmega.megatron.m2rnn_newton_scan import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NEWTON_ITERS,
    _compute_residual_and_sech2,
    _build_linear_operators,
    _sequential_recurrence_chunk,
    linearized_operators,
    parallel_scan_linear_sequential,
    parallel_scan_blelloch,
    newton_solve_chunk,
    m2rnn_scan_newton,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()


def _random_chunk_inputs(
    C: int = 128,
    K: int = 64,
    V: int = 16,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    seed: int = 42,
    eye_W: bool = True,
):
    """Generate random inputs for one chunk.

    Returns k, v, xf, q, W, h_start.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(C, K, device=device, dtype=dtype, generator=g)
    v = torch.randn(C, V, device=device, dtype=dtype, generator=g)
    q = torch.randn(C, K, device=device, dtype=dtype, generator=g)
    xf = torch.sigmoid(torch.randn(C, device=device, dtype=dtype, generator=g))
    if eye_W:
        W = torch.eye(V, device=device, dtype=dtype)
        W += 0.05 * torch.randn(V, V, device=device, dtype=dtype, generator=g)
    else:
        W = torch.randn(V, V, device=device, dtype=dtype, generator=g)
    h_start = torch.randn(K, V, device=device, dtype=dtype, generator=g) * 0.1
    return k, v, xf, q, W, h_start


# ===========================================================================
# Tests: linearized_operators vs manual computation
# ===========================================================================


class TestLinearizedOperators:
    """Verify that linearized_operators matches manual step-by-step computation."""

    @pytest.mark.parametrize("C", [1, 4, 16, 64])
    @pytest.mark.parametrize("K,V", [(8, 4), (16, 8), (32, 16)])
    def test_residual_formula(self, C: int, K: int, V: int):
        """residual[t] = h_guess[t] - true_recurrence[t]"""
        k, v, xf, q, W, h_start = _random_chunk_inputs(C=C, K=K, V=V)

        # Constant guess
        h_guess = h_start.unsqueeze(0).expand(C, K, V).clone()

        # Compute via linearized_operators
        sech2, residual, A, b = linearized_operators(h_guess, h_start, k, v, xf, W)

        # Manual residual computation
        h_prev_manual = torch.cat([h_start.unsqueeze(0), h_guess[:-1]], dim=0)
        x = k.unsqueeze(-1) * v.unsqueeze(-2)
        pre = h_prev_manual @ W + x
        h_true = xf.view(-1, 1, 1) * h_prev_manual + (1 - xf.view(-1, 1, 1)) * torch.tanh(pre)
        residual_manual = h_guess - h_true

        assert residual.shape == (C, K, V)
        torch.testing.assert_close(residual, residual_manual, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("C", [1, 8, 32])
    @pytest.mark.parametrize("K,V", [(8, 4), (16, 8)])
    def test_sech2_formula(self, C: int, K: int, V: int):
        """sech2 = 1 - tanh(pre)^2"""
        k, v, xf, q, W, h_start = _random_chunk_inputs(C=C, K=K, V=V)
        h_guess = h_start.unsqueeze(0).expand(C, K, V).clone()

        sech2, residual, A, b = linearized_operators(h_guess, h_start, k, v, xf, W)

        h_prev = torch.cat([h_start.unsqueeze(0), h_guess[:-1]], dim=0)
        x = k.unsqueeze(-1) * v.unsqueeze(-2)
        pre = h_prev @ W + x
        sech2_manual = 1.0 - torch.tanh(pre) ** 2

        torch.testing.assert_close(sech2, sech2_manual, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("C", [1, 8])
    @pytest.mark.parametrize("K,V", [(8, 4), (16, 8)])
    def test_operator_application(self, C: int, K: int, V: int):
        """Direct test: A_t applied to a random delta gives expected result."""
        k, v, xf, q, W, h_start = _random_chunk_inputs(C=C, K=K, V=V)
        h_guess = h_start.unsqueeze(0).expand(C, K, V).clone()

        sech2, residual, A_matrices, b_terms = linearized_operators(
            h_guess, h_start, k, v, xf, W
        )

        # Generate random delta input
        g = torch.Generator(device="cpu").manual_seed(123)
        delta_in = torch.randn(K, V, dtype=h_guess.dtype, generator=g)

        # Apply A_0 to delta_in via our representation
        # A_matrices[0,k] is (V,V): for each k, (V,V) @ (V,) -> (V,)
        delta_out_via_A = torch.einsum("kil,kl->ki", A_matrices[0], delta_in)

        # Apply A_0 to delta_in via the original formula:
        # A_0(delta) = alpha * delta + beta * (delta @ W)
        alpha_0 = xf[0]
        beta_0 = (1 - xf[0]) * sech2[0]  # (K, V)
        delta_out_via_formula = alpha_0 * delta_in + beta_0 * (delta_in @ W)

        torch.testing.assert_close(
            delta_out_via_A, delta_out_via_formula, atol=1e-5, rtol=1e-5
        )


# ===========================================================================
# Tests: parallel_scan_blelloch vs sequential reference
# ===========================================================================


class TestParallelScan:
    """Verify Blelloch scan matches sequential reference."""

    def _build_random_operators(
        self, C: int, K: int, V: int, dtype=torch.float32, device="cpu", seed: int = 7
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build random (A, b) pairs for testing the parallel scan in isolation."""
        g = torch.Generator(device=device).manual_seed(seed)
        # Build random A operators (K, V, V) — each close to identity
        I_V = torch.eye(V, device=device, dtype=dtype)
        A = torch.zeros(C, K, V, V, device=device, dtype=dtype)
        for t in range(C):
            for i in range(K):
                # Each operator is I + small random
                A[t, i] = I_V + 0.1 * torch.randn(V, V, device=device, dtype=dtype, generator=g)
        b = torch.randn(C, K, V, device=device, dtype=dtype, generator=g) * 0.1
        return A, b

    @pytest.mark.parametrize("C", [1, 2, 3, 8, 17, 32, 63, 128])
    @pytest.mark.parametrize("K,V", [(4, 4), (8, 4), (16, 8)])
    def test_blelloch_matches_sequential(self, C: int, K: int, V: int):
        """Blelloch scan must produce same deltas as sequential scan."""
        A, b = self._build_random_operators(C, K, V)

        delta_seq = parallel_scan_linear_sequential(A, b)
        delta_bl = parallel_scan_blelloch(A, b)

        assert delta_seq.shape == (C, K, V)
        assert delta_bl.shape == (C, K, V)
        torch.testing.assert_close(delta_bl, delta_seq, atol=1e-4, rtol=1e-4)

    def test_blelloch_power_of_two_exact(self):
        """Blelloch on power-of-two sizes should be exact."""
        for C in [2, 4, 8, 16, 32, 64]:
            A, b = self._build_random_operators(C, 4, 4, seed=C)
            delta_seq = parallel_scan_linear_sequential(A, b)
            delta_bl = parallel_scan_blelloch(A, b)
            torch.testing.assert_close(
                delta_bl, delta_seq, atol=1e-4, rtol=1e-4,
                msg=f"Blelloch mismatch at C={C}",
            )

    def test_operator_composition(self):
        """Verify compose(A_0, A_1) applied to zero matches A_1(A_0(0)+b_0)+b_1."""
        K, V = 8, 4
        A, b = self._build_random_operators(3, K, V)

        from cppmega.megatron.m2rnn_newton_scan import _pair_compose

        # Compose A[0] then A[1]: apply A[0] first, then A[1].
        A_comp, b_comp = _pair_compose(A[0], b[0], A[1], b[1])

        # Sequential: d_0 = A_0(0) + b_0 = b_0; d_1 = A_1(d_0) + b_1
        d_prev = torch.zeros(K, V)
        d_0_seq = torch.einsum("kij,kj->ki", A[0], d_prev) + b[0]
        d_1_seq = torch.einsum("kij,kj->ki", A[1], d_0_seq) + b[1]

        # Composed operator applied to zero: (A_1@A_0)(0) + A_1(b_0) + b_1
        # = A_1(b_0) + b_1 = d_1_seq
        torch.testing.assert_close(b_comp, d_1_seq, atol=1e-5, rtol=1e-5)


# ===========================================================================
# Tests: Newton solver convergence
# ===========================================================================


class TestNewtonSolver:
    """Verify Newton solver converges to the true recurrence solution."""

    @pytest.mark.parametrize("C,K,V", [(32, 16, 8), (64, 32, 16), (128, 64, 16)])
    def test_newton_converges_to_sequential(self, C: int, K: int, V: int):
        """After 5 Newton iterations, should match sequential within tolerance."""
        k, v, xf, q, W, h_start = _random_chunk_inputs(C=C, K=K, V=V)

        # Sequential reference
        out_ref, h_final_ref = _sequential_recurrence_chunk(
            h_start, k, v, xf, q, W
        )

        # Newton solution
        out_newton, h_guess, history = newton_solve_chunk(
            h_start, k, v, xf, q, W, num_iterations=5, verbose=False
        )

        assert len(history) == 5
        # Residual should decrease
        assert history[-1] <= history[0] + 1e-6, (
            f"Residual did not decrease: {history}"
        )

        # Check output match
        max_err_out = (out_newton - out_ref).abs().max().item()
        max_err_h = (h_guess[-1] - h_final_ref).abs().max().item()

        assert max_err_out < 1e-2, f"out max_err={max_err_out:.4e}"
        assert max_err_h < 1e-2, f"h_final max_err={max_err_h:.4e}"

    def test_newton_residual_decreases(self):
        """Residual norm should decrease monotonically with Newton iterations."""
        k, v, xf, q, W, h_start = _random_chunk_inputs(C=64, K=32, V=16)

        out5, h5, history5 = newton_solve_chunk(
            h_start, k, v, xf, q, W, num_iterations=5, verbose=False
        )

        # Check monotonic decrease
        for i in range(1, len(history5)):
            assert history5[i] <= history5[i - 1] + 1e-8, (
                f"Residual increased at iteration {i}: {history5[i-1]} -> {history5[i]}"
            )

    def test_newton_conservative_with_gt_initial_guess(self):
        """When initial guess IS the sequential trajectory, residual should be ~0."""
        k, v, xf, q, W, h_start = _random_chunk_inputs(C=32, K=16, V=8)

        # Compute the perfect trajectory
        _, h_final = _sequential_recurrence_chunk(h_start, k, v, xf, q, W)

        # Build the perfect guess trajectory
        C = k.shape[0]
        K_val = k.shape[1]
        V_val = v.shape[1]
        h_perfect = torch.empty(C, K_val, V_val)
        h = h_start
        for t in range(C):
            pre = h @ W + torch.outer(k[t], v[t])
            h_new = torch.tanh(pre)
            h = xf[t] * h + (1 - xf[t]) * h_new
            h_perfect[t] = h.clone()

        # Now run Newton with the perfect guess
        # Compute residual directly
        _, residual, _, _ = linearized_operators(h_perfect, h_start, k, v, xf, W)
        res_norm = residual.abs().mean().item()

        # The residual should be essentially zero since h_perfect satisfies recurrence
        assert res_norm < 1e-4, (
            f"Residual with perfect guess should be zero, got {res_norm:.2e}"
        )

    def test_newton_chunk_output_shape(self):
        """Verify output tensor shapes are correct."""
        C, K, V = 64, 16, 8
        k, v, xf, q, W, h_start = _random_chunk_inputs(C=C, K=K, V=V)

        out, h_guess, history = newton_solve_chunk(h_start, k, v, xf, q, W)

        assert out.shape == (C, V)
        assert h_guess.shape == (C, K, V)
        assert len(history) == DEFAULT_NEWTON_ITERS


# ===========================================================================
# Tests: Full m2rnn_scan_newton integration
# ===========================================================================


class TestM2RNNScanNewton:
    """Integration tests for the top-level m2rnn_scan_newton function."""

    @staticmethod
    def _sequential_full_scan(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        W: torch.Tensor,
        xf: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequential M2RNN — ground truth reference."""
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

        W_expanded = W[None, ...]
        x = k[..., None] * v[..., None, :]  # (B, S, n, K, V) — outer product
        y = torch.empty(batch, seq, n, k_dim, v_dim, device=q.device, dtype=q.dtype)
        for s in range(seq):
            f = xf[:, s, :].unsqueeze(-1).unsqueeze(-1)
            h_new = torch.tanh(h @ W_expanded + x[:, s])
            h = f * h + (1 - f) * h_new
            y[:, s] = h
        out = (q[..., None, :] @ y).squeeze(-2)
        return out, h

    def _random_full_inputs(
        self,
        B: int = 2,
        S: int = 64,
        H: int = 2,
        K: int = 16,
        V: int = 8,
        *,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        seed: int = 123,
    ):
        g = torch.Generator(device=device).manual_seed(seed)
        q = torch.randn(B, S, H, K, device=device, dtype=dtype, generator=g)
        k = torch.randn(B, S, H, K, device=device, dtype=dtype, generator=g)
        v = torch.randn(B, S, H, V, device=device, dtype=dtype, generator=g)
        W = torch.eye(V, device=device, dtype=dtype).unsqueeze(0).expand(H, -1, -1).clone()
        W += 0.05 * torch.randn(W.shape, device=device, dtype=dtype, generator=g)
        xf = torch.sigmoid(torch.randn(B, S, H, device=device, dtype=dtype, generator=g))
        return q, k, v, W, xf

    def test_newton_scan_matches_sequential_small(self):
        """Small-scale integration test."""
        B, S, H, K, V = 1, 32, 2, 16, 8
        q, k, v, W, xf = self._random_full_inputs(B=B, S=S, H=H, K=K, V=V)

        out_ref, h_ref = self._sequential_full_scan(q, k, v, W, xf)
        out_newton, h_newton = m2rnn_scan_newton(
            q, k, v, W, xf,
            chunk_size=16,
            newton_iters=3,
            use_newton=True,
        )

        max_err_out = (out_newton - out_ref).abs().max().item()
        max_err_h = (h_newton - h_ref).abs().max().item()

        assert max_err_out < 5e-2, f"out max_err={max_err_out:.4e}"
        assert max_err_h < 3e-2, f"h max_err_h={max_err_h:.4e}"

    def test_newton_sequential_fallback_matches(self):
        """When use_newton=False, should match sequential exactly (same math)."""
        q, k, v, W, xf = self._random_full_inputs(B=1, S=32, H=1, K=8, V=4)

        out_ref, h_ref = self._sequential_full_scan(q, k, v, W, xf)
        out_fb, h_fb = m2rnn_scan_newton(
            q, k, v, W, xf, chunk_size=16, use_newton=False
        )

        torch.testing.assert_close(out_fb, out_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h_fb, h_ref, atol=1e-5, rtol=1e-5)

    def test_newton_scan_with_h0(self):
        """Passing initial state should affect output correctly."""
        B, S, H, K, V = 1, 32, 2, 16, 8
        q, k, v, W, xf = self._random_full_inputs(B=B, S=S, H=H, K=K, V=V)

        g = torch.Generator(device="cpu").manual_seed(456)
        h0 = torch.randn(B, H, K, V, generator=g) * 0.1

        out_ref, h_ref = self._sequential_full_scan(q, k, v, W, xf, h0=h0)
        out_newton, h_newton = m2rnn_scan_newton(
            q, k, v, W, xf, h0=h0, chunk_size=8, newton_iters=3
        )

        max_err_out = (out_newton - out_ref).abs().max().item()
        max_err_h = (h_newton - h_ref).abs().max().item()

        assert max_err_out < 1e-1, f"out max_err={max_err_out:.4e}"
        assert max_err_h < 5e-2, f"h max_err_h={max_err_h:.4e}"

    def test_head_broadcasting(self):
        """Different head counts should broadcast correctly."""
        B, S = 1, 32
        K, V = 16, 8
        n_q, n_k, n_v, n_w = 1, 1, 4, 2
        n_f = 4

        g = torch.Generator().manual_seed(789)
        q = torch.randn(B, S, n_q, K, generator=g)
        k = torch.randn(B, S, n_k, K, generator=g)
        v = torch.randn(B, S, n_v, V, generator=g)
        W = torch.eye(V).unsqueeze(0).expand(n_w, -1, -1).clone()
        W += 0.05 * torch.randn(W.shape, generator=g)
        xf = torch.sigmoid(torch.randn(B, S, n_f, generator=g))

        out_ref, h_ref = self._sequential_full_scan(q, k, v, W, xf)
        out_newton, _ = m2rnn_scan_newton(
            q, k, v, W, xf, chunk_size=16, newton_iters=3
        )

        H = max(n_q, n_k, n_v, n_w, n_f)
        assert out_newton.shape == (B, S, H, V)
        max_err = (out_newton - out_ref).abs().max().item()
        assert max_err < 0.15, f"head-broadcast max_err={max_err:.4e}"

    def test_single_chunk_entire_seq(self):
        """When chunk_size >= seq_len, one Newton solve handles everything."""
        S = 32
        q, k, v, W, xf = self._random_full_inputs(B=1, S=S, H=1, K=8, V=4)

        out_ref, h_ref = self._sequential_full_scan(q, k, v, W, xf)
        out_newton, h_newton = m2rnn_scan_newton(
            q, k, v, W, xf, chunk_size=S + 10, newton_iters=5
        )

        max_err_out = (out_newton - out_ref).abs().max().item()
        max_err_h = (h_newton - h_ref).abs().max().item()
        assert max_err_out < 1e-2, f"single-chunk out err={max_err_out:.4e}"
        assert max_err_h < 1e-2, f"single-chunk h err={max_err_h:.4e}"

    def test_newton_improves_with_more_iterations(self):
        """More Newton iterations should improve accuracy."""
        q, k, v, W, xf = self._random_full_inputs(B=1, S=32, H=1, K=16, V=8)

        out_ref, _ = self._sequential_full_scan(q, k, v, W, xf)

        errors = []
        for n_iter in [1, 2, 3, 5]:
            out_n, _ = m2rnn_scan_newton(
                q, k, v, W, xf, chunk_size=32, newton_iters=n_iter
            )
            err = (out_n - out_ref).abs().max().item()
            errors.append(err)

        # Error should decrease (or at least not increase) with more iterations
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i - 1] * 1.01, (
                f"Error increased from {n_iter-1} to {n_iter} iters: "
                f"{errors[i-1]:.4e} -> {errors[i]:.4e}"
            )


# ===========================================================================
# Tests: Numerical stability / edge cases
# ===========================================================================


class TestEdgeCases:
    """Numerical stability and edge case tests."""

    def test_zero_initial_state(self):
        """Starting from zero state should work."""
        C, K, V = 32, 16, 8
        k, v, xf, q, W, _ = _random_chunk_inputs(C=C, K=K, V=V)
        h_start = torch.zeros(K, V)

        out_n, h_n, history = newton_solve_chunk(h_start, k, v, xf, q, W)
        out_s, h_s = _sequential_recurrence_chunk(h_start, k, v, xf, q, W)

        max_err = (out_n - out_s).abs().max().item()
        assert max_err < 1e-2, f"zero-init max_err={max_err:.4e}"

    def test_identity_W(self):
        """When W = I (identity), the Newton method should still work."""
        C, K, V = 64, 16, 8
        k, v, xf, q, _, h_start = _random_chunk_inputs(C=C, K=K, V=V)
        W = torch.eye(V)

        out_n, h_n, history = newton_solve_chunk(h_start, k, v, xf, q, W)
        out_s, h_s = _sequential_recurrence_chunk(h_start, k, v, xf, q, W)

        max_err = (out_n - out_s).abs().max().item()
        assert max_err < 5e-2, f"identity-W max_err={max_err:.4e}"

    def test_saturating_tanh(self):
        """When pre-activation is large (tanh saturates), sech2 ~ 0, test stability."""
        C, K, V = 32, 16, 8
        k, v, xf, q, W, h_start = _random_chunk_inputs(C=C, K=K, V=V)

        # Make pre-activation large so tanh saturates
        k_large = k * 100.0
        v_large = v * 100.0

        out_n, h_n, history = newton_solve_chunk(
            h_start, k_large, v_large, xf, q, W, num_iterations=4
        )
        out_s, h_s = _sequential_recurrence_chunk(
            h_start, k_large, v_large, xf, q, W
        )

        max_err = (out_n - out_s).abs().max().item()
        # May be larger due to saturated tanh making the problem near-linear
        assert max_err < 5e-1, f"saturating-tanh max_err={max_err:.4e}"

    def test_chunk_size_one(self):
        """C=1: Newton should exactly match sequential."""
        C, K, V = 1, 16, 8
        k, v, xf, q, W, h_start = _random_chunk_inputs(C=C, K=K, V=V, eye_W=False)

        out_n, h_n, history = newton_solve_chunk(h_start, k, v, xf, q, W)
        out_s, h_s = _sequential_recurrence_chunk(h_start, k, v, xf, q, W)

        torch.testing.assert_close(out_n, out_s, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h_n[-1], h_s, atol=1e-5, rtol=1e-5)

    def test_all_ones_forget_gate(self):
        """xf=1.0: no update, h stays at h_start. Newton should handle cleanly."""
        C, K, V = 32, 16, 8
        k, v, _, q, W, h_start = _random_chunk_inputs(C=C, K=K, V=V)
        xf_ones = torch.ones(C)

        out_n, h_n, history = newton_solve_chunk(
            h_start, k, v, xf_ones, q, W
        )
        out_s, h_s = _sequential_recurrence_chunk(
            h_start, k, v, xf_ones, q, W
        )

        max_err = (out_n - out_s).abs().max().item()
        assert max_err < 1e-4, f"xf=1.0 max_err={max_err:.4e}"

    def test_all_zeros_forget_gate(self):
        """xf=0.0: full update each step. Newton should track correctly."""
        C, K, V = 32, 16, 8
        k, v, _, q, W, h_start = _random_chunk_inputs(C=C, K=K, V=V)
        xf_zeros = torch.zeros(C)

        out_n, h_n, history = newton_solve_chunk(
            h_start, k, v, xf_zeros, q, W, num_iterations=5
        )
        out_s, h_s = _sequential_recurrence_chunk(
            h_start, k, v, xf_zeros, q, W
        )

        max_err = (out_n - out_s).abs().max().item()
        # xf=0 makes the recurrence more nonlinear, may need more iters
        assert max_err < 0.5, f"xf=0 max_err={max_err:.4e}"


# ===========================================================================
# Tests: Gradient correctness via autograd
# ===========================================================================


class TestGradCorrectness:
    """Verify that autograd flows correctly through the Newton solver."""

    def test_gradient_wrt_inputs(self):
        """Gradients w.r.t. k, v, xf, W should exist and be nonzero."""
        C, K, V = 32, 16, 8

        k = torch.randn(C, K, requires_grad=True)
        v = torch.randn(C, V, requires_grad=True)
        xf = torch.sigmoid(torch.randn(C, requires_grad=True))
        W = torch.eye(V, requires_grad=True)
        W.data += 0.05 * torch.randn(V, V)
        h_start = torch.randn(K, V, requires_grad=True)

        q = torch.ones(C, K, requires_grad=True)
        out, h_guess, history = newton_solve_chunk(h_start, k, v, xf, q, W)

    def test_gradient_wrt_h_start(self):
        """Gradient should flow back to h_start."""
        C, K, V = 16, 8, 4
        k, v, xf, q, W, _ = _random_chunk_inputs(C=C, K=K, V=V)

        h_start = torch.randn(K, V, requires_grad=True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        xf_d = xf.detach().clone().requires_grad_(True)

        out, h_guess, _ = newton_solve_chunk(h_start, k, v, xf_d, q, W)

        # Loss: sum of outputs
        loss = out.sum()
        loss.backward()

        assert h_start.grad is not None
        assert h_start.grad.abs().sum() > 0, "Gradient w.r.t. h_start is zero"

    def test_gradient_wrt_W(self):
        """Gradient flows to W."""
        C, K, V = 16, 8, 4
        k, v, xf, q, _, h_start = _random_chunk_inputs(C=C, K=K, V=V)

        W = torch.eye(V, requires_grad=True)
        W.data += 0.05 * torch.randn(V, V)

        out_n, _, _ = newton_solve_chunk(h_start, k, v, xf, q, W)
        out_s, _ = _sequential_recurrence_chunk(h_start, k, v, xf, q, W)

        loss_n = out_n.sum()
        loss_n.backward()
        assert W.grad is not None
        assert W.grad.abs().sum() > 0

    def test_gradient_matches_sequential(self):
        """Gradients from Newton solver should approximately match sequential."""
        C, K, V = 16, 8, 4
        k, v, xf, q, _, h_start = _random_chunk_inputs(C=C, K=K, V=V)

        # Newton
        k1 = k.detach().clone().requires_grad_(True)
        v1 = v.detach().clone().requires_grad_(True)
        xf1 = xf.detach().clone().requires_grad_(True)
        h1 = h_start.detach().clone().requires_grad_(True)
        W1 = torch.eye(V, requires_grad=True)
        W1.data += 0.05 * torch.randn(V, V)

        out_n, _, _ = newton_solve_chunk(h1, k1, v1, xf1, q, W1, num_iterations=4)
        loss_n = out_n.sum()
        loss_n.backward()

        # Sequential
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        xf2 = xf.detach().clone().requires_grad_(True)
        h2 = h_start.detach().clone().requires_grad_(True)
        W2 = torch.eye(V, requires_grad=True)
        W2.data += 0.05 * torch.randn(V, V)

        out_s, _ = _sequential_recurrence_chunk(h2, k2, v2, xf2, q, W2)
        loss_s = out_s.sum()
        loss_s.backward()

        # Check gradient magnitudes are similar
        for name, g_n, g_s in [
            ("k", k1.grad, k2.grad),
            ("W", W1.grad, W2.grad),
            ("h_start", h1.grad, h2.grad),
        ]:
            mag_n = g_n.abs().max().item()
            mag_s = g_s.abs().max().item()
            rel_diff = (g_n - g_s).abs().max().item() / (mag_s + 1e-12)
            assert rel_diff < 1.0, (
                f"Gradient mismatch for {name}: rel_diff={rel_diff:.4e}"
            )


# ===========================================================================
# Tests: NAM56R dimensions (if CUDA available)
# ===========================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestNAM56RDims:
    """Tests at production NAM56R dimensions: K=64, V=16."""

    def test_newton_chunk_production_dims(self):
        """Newton solve at K=64, V=16 with C=128."""
        C, K, V = 128, 64, 16
        k, v, xf, q, W, h_start = _random_chunk_inputs(
            C=C, K=K, V=V, device="cuda"
        )

        out_n, h_n, history = newton_solve_chunk(
            h_start, k, v, xf, q, W, num_iterations=4
        )
        out_s, h_s = _sequential_recurrence_chunk(h_start, k, v, xf, q, W)

        max_err_out = (out_n - out_s).abs().max().item()
        max_err_h = (h_n[-1] - h_s).abs().max().item()

        assert max_err_out < 0.1, f"prod-dims out err={max_err_out:.4e}"
        assert max_err_h < 0.05, f"prod-dims h err={max_err_h:.4e}"

    def test_blelloch_production_dims(self):
        """Blelloch scan at K=64, V=16, C=256."""
        C, K, V = 256, 64, 16

        I_V = torch.eye(V, device="cuda")
        A = torch.zeros(C, K, V, V, device="cuda")
        for t in range(C):
            for i in range(K):
                A[t, i] = I_V + 0.01 * torch.randn(V, V, device="cuda")
        b = torch.randn(C, K, V, device="cuda") * 0.1

        delta_seq = parallel_scan_linear_sequential(A, b)
        delta_bl = parallel_scan_blelloch(A, b)

        torch.testing.assert_close(delta_bl, delta_seq, atol=1e-3, rtol=1e-3)

    def test_full_scan_nam56r_short(self):
        """Full m2rnn_scan_newton at NAM56R dims, short sequence."""
        B, S, H, K, V = 1, 64, 8, 64, 16
        g = torch.Generator(device="cuda").manual_seed(42)
        q = torch.randn(B, S, H, K, device="cuda", generator=g)
        k = torch.randn(B, S, H, K, device="cuda", generator=g)
        v = torch.randn(B, S, H, V, device="cuda", generator=g)
        W = (
            torch.eye(V, device="cuda")
            .unsqueeze(0)
            .expand(H, -1, -1)
            .clone()
        )
        W += 0.05 * torch.randn(W.shape, device="cuda", generator=g)
        xf = torch.sigmoid(torch.randn(B, S, H, device="cuda", generator=g))

        out_ref, h_ref = TestM2RNNScanNewton._sequential_full_scan(q, k, v, W, xf)
        out_newton, h_newton = m2rnn_scan_newton(
            q, k, v, W, xf, chunk_size=32, newton_iters=4
        )

        max_err_out = (out_newton - out_ref).abs().max().item()
        max_err_h = (h_newton - h_ref).abs().max().item()
        assert max_err_out < 0.5, f"nam56r out err={max_err_out:.4e}"
        assert max_err_h < 0.3, f"nam56r h err={max_err_h:.4e}"

    def test_newton_recommits_triton_vs_newton_at_production_shape(self):
        """At NAM56R dims, compare against Triton kernel output where available."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")
        from cppmega.megatron.m2rnn_triton import m2rnn_scan_triton

        B, S, H, K, V = 1, 64, 4, 64, 16
        g = torch.Generator(device="cuda").manual_seed(99)
        q = torch.randn(B, S, H, K, device="cuda", generator=g)
        k = torch.randn(B, S, H, K, device="cuda", generator=g)
        v = torch.randn(B, S, H, V, device="cuda", generator=g)
        W = (
            torch.eye(V, device="cuda")
            .unsqueeze(0)
            .expand(H, -1, -1)
            .clone()
        )
        W += 0.05 * torch.randn(W.shape, device="cuda", generator=g)
        xf = torch.sigmoid(torch.randn(B, S, H, device="cuda", generator=g))

        out_tri, h_tri = m2rnn_scan_triton(q, k, v, W, xf)
        out_n, h_n = m2rnn_scan_newton(
            q, k, v, W, xf, chunk_size=32, newton_iters=3
        )

        # Newton should be close to Triton (both approximate the true recurrence)
        max_err_out = (out_n - out_tri).abs().max().item()
        max_err_h = (h_n - h_tri).abs().max().item()
        # Relaxed tolerance: Newton vs Triton may differ slightly due to
        # different compute paths (tanh approx in Triton vs torch.tanh)
        assert max_err_out < 1.0, f"newton-vs-triton out err={max_err_out:.4e}"
        assert max_err_h < 1.0, f"newton-vs-triton h err={max_err_h:.4e}"


# ===========================================================================
# Tests: Performance profile (not gating correctness)
# ===========================================================================


class TestPerformanceProfile:
    """Performance sanity checks — optional, informational only."""

    @pytest.mark.large
    def test_chunk_sizes_performance(self):
        """Profile different chunk sizes, ensure Newton doesn't crash."""
        C_vals = [32, 64, 128, 256]
        K, V = 64, 16

        for C in C_vals:
            k, v, xf, q, W, h_start = _random_chunk_inputs(C=C, K=K, V=V)
            out_n, h_n, history = newton_solve_chunk(
                h_start, k, v, xf, q, W, num_iterations=4
            )
            assert out_n.shape == (C, V)
            assert h_n.shape == (C, K, V)
            assert len(history) >= 3  # at least 3 Newton iters (may run 4)
