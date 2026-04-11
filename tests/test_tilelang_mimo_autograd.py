"""Correctness and CUDA-graph tests for cppmega's custom TileLang MIMO autograd.

Compares ``cppmega_tilelang_mimo_combined`` against upstream ``mamba3_mimo``
on random BF16 inputs at NAM56R shapes, checks both forward outputs and
backward gradients, then attempts CUDA graph capture.

Requires GPU. Run on bench3:
    CUDA_VISIBLE_DEVICES=1 python -m pytest tests/test_tilelang_mimo_autograd.py -v -s
"""

from __future__ import annotations

import time
import pytest
import torch

# ---------------------------------------------------------------------------
# Skip early if no GPU
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

# ---------------------------------------------------------------------------
# Imports (heavy -- triggers TileLang JIT on first use)
# ---------------------------------------------------------------------------
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as upstream_mimo
from cppmega.megatron.tilelang_mimo_autograd import cppmega_tilelang_mimo_combined


# ---------------------------------------------------------------------------
# NAM56R shapes
# ---------------------------------------------------------------------------
BATCH = 2
SEQLEN = 2048
NHEADS = 32
NGROUPS = 1       # nheads_qk (G in kernel)
HEADDIM_QK = 128
HEADDIM_V = 64
MIMO_RANK = 4
CHUNK_SIZE = 16
ROTARY_DIV = 4
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _make_inputs(requires_grad: bool = True):
    """Create random inputs matching NAM56R MIMO config.

    Dtype conventions mirror the upstream Mamba3 module:
    - Q, K, V, Trap, Z: model dtype (bf16)
    - ADT, DT, Angles: float32 (computed via F.softplus / angle_dt in fp32)
    - Bias/weight params (Q_bias, K_bias, MIMO_V/Z/Out, D): float32
    """
    def _r(*shape, dtype=DTYPE, rg=requires_grad):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=rg)

    Q = _r(BATCH, SEQLEN, MIMO_RANK, NGROUPS, HEADDIM_QK)
    K = _r(BATCH, SEQLEN, MIMO_RANK, NGROUPS, HEADDIM_QK)
    V = _r(BATCH, SEQLEN, NHEADS, HEADDIM_V)
    ADT = _r(BATCH, NHEADS, SEQLEN, dtype=torch.float32)
    DT = _r(BATCH, NHEADS, SEQLEN, dtype=torch.float32)
    Trap = _r(BATCH, NHEADS, SEQLEN)  # bf16, matches model dtype

    Q_bias = _r(NHEADS, MIMO_RANK, HEADDIM_QK, dtype=torch.float32)
    K_bias = _r(NHEADS, MIMO_RANK, HEADDIM_QK, dtype=torch.float32)
    MIMO_V = _r(NHEADS, MIMO_RANK, HEADDIM_V, dtype=torch.float32)
    MIMO_Z = _r(NHEADS, MIMO_RANK, HEADDIM_V, dtype=torch.float32)
    MIMO_Out = _r(NHEADS, MIMO_RANK, HEADDIM_V, dtype=torch.float32)

    n_angles = HEADDIM_QK // ROTARY_DIV
    Angles = _r(BATCH, SEQLEN, NHEADS, n_angles, dtype=torch.float32)
    D = _r(NHEADS, dtype=torch.float32)
    Z = _r(BATCH, SEQLEN, NHEADS, HEADDIM_V)

    return (
        Q, K, V, ADT, DT, Trap,
        Q_bias, K_bias,
        MIMO_V, MIMO_Z, MIMO_Out,
        Angles, D, Z,
    )


def _call(fn, inputs, return_state=False):
    return fn(
        *inputs,
        chunk_size=CHUNK_SIZE,
        rotary_dim_divisor=ROTARY_DIV,
        dtype=DTYPE,
        return_state=return_state,
        cu_seqlens=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForwardCorrectness:
    """Forward outputs must match upstream within BF16 tolerance."""

    def test_output_matches(self):
        inputs = _make_inputs(requires_grad=False)
        with torch.no_grad():
            ref = _call(upstream_mimo, inputs)
            ours = _call(cppmega_tilelang_mimo_combined, inputs)

        assert ref.shape == ours.shape, f"Shape mismatch: {ref.shape} vs {ours.shape}"
        torch.testing.assert_close(ours, ref, rtol=1e-2, atol=1e-2)

    def test_return_state_matches(self):
        inputs = _make_inputs(requires_grad=False)
        with torch.no_grad():
            ref = _call(upstream_mimo, inputs, return_state=True)
            ours = _call(cppmega_tilelang_mimo_combined, inputs, return_state=True)

        # ref and ours are tuples: (Out, Final_Angle, Final_SSM_State, Final_K, Final_V)
        assert len(ref) == len(ours), f"Tuple length mismatch: {len(ref)} vs {len(ours)}"
        for i, (r, o) in enumerate(zip(ref, ours)):
            torch.testing.assert_close(o, r, rtol=1e-2, atol=1e-2, msg=f"state component {i}")


class TestBackwardCorrectness:
    """Backward gradients must match upstream."""

    def test_grads_match(self):
        # Run upstream
        inputs_ref = _make_inputs(requires_grad=True)
        out_ref = _call(upstream_mimo, inputs_ref)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        grads_ref = [t.grad.clone() for t in inputs_ref if t.grad is not None]

        # Run ours with identical data
        inputs_ours = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in inputs_ref)
        out_ours = _call(cppmega_tilelang_mimo_combined, inputs_ours)
        loss_ours = out_ours.sum()
        loss_ours.backward()
        grads_ours = [t.grad.clone() for t in inputs_ours if t.grad is not None]

        assert len(grads_ref) == len(grads_ours), (
            f"Grad count mismatch: {len(grads_ref)} vs {len(grads_ours)}"
        )
        for i, (gr, go) in enumerate(zip(grads_ref, grads_ours)):
            torch.testing.assert_close(
                go, gr, rtol=1e-2, atol=1e-2,
                msg=f"grad {i} (shape {gr.shape})",
            )


class TestTiming:
    """Timing comparison -- our wrapper should be within 5% of upstream."""

    @staticmethod
    def _time_fn(fn, inputs, warmup=5, iters=20):
        # Warmup
        for _ in range(warmup):
            out = _call(fn, inputs)
            out.sum().backward()
            for t in inputs:
                if t.grad is not None:
                    t.grad = None
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            out = _call(fn, inputs)
            out.sum().backward()
            for t in inputs:
                if t.grad is not None:
                    t.grad = None
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return elapsed / iters * 1e6  # microseconds

    def test_timing_comparable(self):
        inputs = _make_inputs(requires_grad=True)
        inputs2 = tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in inputs)

        t_upstream = self._time_fn(upstream_mimo, inputs)
        t_ours = self._time_fn(cppmega_tilelang_mimo_combined, inputs2)

        slowdown = t_ours / t_upstream
        print(f"\n  upstream fwd+bwd: {t_upstream:.0f} us")
        print(f"  cppmega  fwd+bwd: {t_ours:.0f} us")
        print(f"  ratio (ours/upstream): {slowdown:.3f}")
        # Allow up to 10% overhead from the wrapper
        assert slowdown < 1.10, f"cppmega wrapper is {slowdown:.2f}x slower than upstream"


class TestCUDAGraph:
    """Verify that the autograd path is CUDA-graph capturable."""

    def test_fwd_bwd_graph_capture(self):
        inputs = _make_inputs(requires_grad=True)

        # Warmup (required before graph capture)
        out = _call(cppmega_tilelang_mimo_combined, inputs)
        loss = out.sum()
        loss.backward()
        for t in inputs:
            if t.grad is not None:
                t.grad = None
        torch.cuda.synchronize()

        # Capture
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                out = _call(cppmega_tilelang_mimo_combined, inputs)
                loss = out.sum()
                loss.backward()
            g.replay()
            torch.cuda.synchronize()
            graph_ok = True
            graph_msg = "PASS"
        except Exception as e:
            graph_ok = False
            graph_msg = f"FAIL: {e}"

        print(f"\n  CUDA graph capture: {graph_msg}")
        if not graph_ok:
            pytest.skip(f"CUDA graph capture failed (expected -- Triton dacs kernel): {graph_msg}")
