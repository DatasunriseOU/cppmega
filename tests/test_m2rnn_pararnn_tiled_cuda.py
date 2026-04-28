"""CUDA probe tests for the tiled M2RNN ParaRNN local affine scan."""

from __future__ import annotations

import math

import pytest
import torch

from cppmega.megatron.m2rnn_pararnn import PararnnConfig, m2rnn_pararnn_forward
from cppmega.megatron.m2rnn_pararnn_tiled_cuda import (
    TiledCudaPararnnConfig,
    local_tile_scan_debug,
    memory_accounting_bytes,
    m2rnn_pararnn_tiled_cuda_forward,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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


def _make_inputs(B, S, H, K, V, *, seed=0):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, K, generator=g, device=device, dtype=torch.float32) * 0.5
    k = torch.randn(B, S, H, K, generator=g, device=device, dtype=torch.float32) * 0.5
    v = torch.randn(B, S, H, V, generator=g, device=device, dtype=torch.float32) * 0.5
    W = torch.randn(H, V, V, generator=g, device=device, dtype=torch.float32) * (0.45 / math.sqrt(V))
    xf = torch.sigmoid(torch.randn(B, S, H, generator=g, device=device, dtype=torch.float32) - 0.5)
    return q, k, v, W, xf


def test_local_tile_scan_matches_direct_sequential_delta_for_zero_guess():
    B, S, H, K, V = 1, 9, 2, 3, 16
    tile_size = 4
    q, k, v, W, xf = _make_inputs(B, S, H, K, V, seed=1)
    Be = B * H * K
    h = torch.zeros(Be, S, V, device="cuda", dtype=torch.float32)
    h0 = torch.zeros(Be, V, device="cuda", dtype=torch.float32)

    local_delta, local_prefix, tile_A, tile_b = local_tile_scan_debug(
        q, k, v, W, xf, h, h0, tile_size=tile_size
    )
    torch.cuda.synchronize()

    assert tuple(local_delta.shape) == (Be, S, V)
    assert tuple(local_prefix.shape) == (Be, S, V, V)
    assert tuple(tile_A.shape) == (Be, 3, V, V)
    assert tuple(tile_b.shape) == (Be, 3, V)

    expected_delta = torch.empty_like(local_delta)
    expected_prefix = torch.empty_like(local_prefix)
    expected_A = torch.empty_like(tile_A)
    expected_b = torch.empty_like(tile_b)
    eye = torch.eye(V, device="cuda")
    for b in range(B):
        for head in range(H):
            for kk in range(K):
                chain = (b * H + head) * K + kk
                for tile in range(3):
                    M = eye.clone()
                    d = torch.zeros(V, device="cuda")
                    start = tile * tile_size
                    end = min(start + tile_size, S)
                    for s in range(start, end):
                        h_prev = h0[chain] if s == 0 else h[chain, s - 1]
                        z = h_prev @ W[head] + k[b, s, head, kk] * v[b, s, head]
                        h_new = torch.tanh(z)
                        f = xf[b, s, head]
                        rhs = -h[chain, s] + f * h_prev + (1.0 - f) * h_new
                        P = f * eye + (1.0 - f) * (1.0 - h_new * h_new)[:, None] * W[head].t()
                        d = rhs + P @ d
                        M = P @ M
                        expected_delta[chain, s] = d
                        expected_prefix[chain, s] = M
                    expected_A[chain, tile] = M
                    expected_b[chain, tile] = d

    torch.testing.assert_close(local_delta, expected_delta, atol=8e-6, rtol=8e-6)
    torch.testing.assert_close(local_prefix, expected_prefix, atol=8e-6, rtol=8e-6)
    torch.testing.assert_close(tile_A, expected_A, atol=8e-6, rtol=8e-6)
    torch.testing.assert_close(tile_b, expected_b, atol=8e-6, rtol=8e-6)


def test_production_memory_accounting_excludes_local_prefix():
    bytes_by_tensor = memory_accounting_bytes(B=1, S=33, H=2, K=4, V=16, tile_size=8)

    assert "local_prefix" not in bytes_by_tensor
    assert bytes_by_tensor["debug_only_local_prefix"] == bytes_by_tensor["forbidden_full_jacobian"]
    assert bytes_by_tensor["delta"] == 1 * 2 * 4 * 33 * 16 * 4
    assert bytes_by_tensor["debug_only_local_delta"] == bytes_by_tensor["delta"]


@pytest.mark.parametrize("tile_size", [3, 8])
def test_tiled_cuda_forward_matches_pararnn_and_sequential(tile_size):
    B, S, H, K, V = 1, 17, 2, 4, 16
    q, k, v, W, xf = _make_inputs(B, S, H, K, V, seed=2)

    out_ref, h_ref = _torch_m2rnn_forward(q, k, v, W, xf)
    out_par, h_par = m2rnn_pararnn_forward(
        q, k, v, W, xf, config=PararnnConfig(max_its=6, chunk_size=8)
    )
    out_cuda, h_cuda = m2rnn_pararnn_tiled_cuda_forward(
        q, k, v, W, xf, config=TiledCudaPararnnConfig(max_its=6, tile_size=tile_size)
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_par, out_ref, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(h_par, h_ref, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(out_cuda, out_ref, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(h_cuda, h_ref, atol=3e-4, rtol=3e-4)


def test_tiled_cuda_forward_accepts_bfloat16_inputs_with_fp32_accumulators():
    B, S, H, K, V = 1, 10, 2, 3, 8
    q, k, v, W, xf = _make_inputs(B, S, H, K, V, seed=4)
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)
    v_bf16 = v.to(torch.bfloat16)
    W_bf16 = W.to(torch.bfloat16)
    xf_bf16 = xf.to(torch.bfloat16)

    out_ref, h_ref = _torch_m2rnn_forward(
        q_bf16.float(), k_bf16.float(), v_bf16.float(), W_bf16.float(), xf_bf16.float()
    )
    out_cuda, h_cuda = m2rnn_pararnn_tiled_cuda_forward(
        q_bf16,
        k_bf16,
        v_bf16,
        W_bf16,
        xf_bf16,
        config=TiledCudaPararnnConfig(max_its=7, tile_size=4),
    )
    torch.cuda.synchronize()

    assert out_cuda.dtype == torch.float32
    assert h_cuda.dtype == torch.float32
    torch.testing.assert_close(out_cuda, out_ref, atol=4e-4, rtol=4e-4)
    torch.testing.assert_close(h_cuda, h_ref, atol=4e-4, rtol=4e-4)


def test_tiled_cuda_forward_h0_and_head_broadcast():
    B, S, K, V = 1, 12, 3, 16
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(3)
    q = torch.randn(B, S, 1, K, generator=g, device=device, dtype=torch.float32) * 0.5
    k = torch.randn(B, S, 2, K, generator=g, device=device, dtype=torch.float32) * 0.5
    v = torch.randn(B, S, 4, V, generator=g, device=device, dtype=torch.float32) * 0.5
    W = torch.randn(4, V, V, generator=g, device=device, dtype=torch.float32) * 0.1
    xf = torch.sigmoid(torch.randn(B, S, 4, generator=g, device=device, dtype=torch.float32) - 0.5)
    h0 = torch.randn(B, 4, K, V, generator=g, device=device, dtype=torch.float32) * 0.2

    out_ref, h_ref = _torch_m2rnn_forward(q, k, v, W, xf, h0=h0)
    out_cuda, h_cuda = m2rnn_pararnn_tiled_cuda_forward(
        q, k, v, W, xf, h0=h0, config=TiledCudaPararnnConfig(max_its=7, tile_size=5)
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_cuda, out_ref, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(h_cuda, h_ref, atol=3e-4, rtol=3e-4)
