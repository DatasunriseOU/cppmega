"""Tests for the tiled/streaming TileLang M2RNN ParaRNN path."""

from __future__ import annotations

import math

import pytest
import torch

from cppmega.megatron.m2rnn_pararnn import PararnnConfig, m2rnn_pararnn_forward
from cppmega.megatron.m2rnn_pararnn_tiled_tilelang import (
    TiledTileLangConfig,
    m2rnn_pararnn_tiled_tilelang_forward,
)


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


def _make_inputs(B, S, H, k_dim, v_dim, *, device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, k_dim, generator=g, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, S, H, k_dim, generator=g, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, S, H, v_dim, generator=g, device=device, dtype=dtype) * 0.5
    W = torch.randn(H, v_dim, v_dim, generator=g, device=device, dtype=dtype) * (
        0.5 / math.sqrt(v_dim)
    )
    xf = torch.sigmoid(torch.randn(B, S, H, generator=g, device=device, dtype=dtype) - 0.5)
    return q, k, v, W, xf


def test_one_newton_step_matches_full_pararnn_scan():
    device = "cpu"
    dtype = torch.float64
    q, k, v, W, xf = _make_inputs(1, 33, 2, 4, 16, device=device, dtype=dtype)

    out_tiled, h_tiled, stats = m2rnn_pararnn_tiled_tilelang_forward(
        q,
        k,
        v,
        W,
        xf,
        config=TiledTileLangConfig(max_its=1, tile_len=16, backend="torch"),
        return_stats=True,
    )
    out_full, h_full = m2rnn_pararnn_forward(
        q,
        k,
        v,
        W,
        xf,
        config=PararnnConfig(max_its=1, chunk_size=0),
    )

    torch.testing.assert_close(out_tiled, out_full, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(h_tiled, h_full, atol=1e-10, rtol=1e-10)
    assert stats.full_jac_elements_avoided == stats.be * stats.s * 16 * 16


@pytest.mark.parametrize("tile_len", [16, 32, 64])
def test_tiled_forward_matches_sequential_reference(tile_len):
    device = "cpu"
    dtype = torch.float64
    q, k, v, W, xf = _make_inputs(1, 64, 2, 4, 16, device=device, dtype=dtype)

    out_ref, h_ref = _torch_m2rnn_forward(q, k, v, W, xf)
    out_tiled, h_tiled = m2rnn_pararnn_tiled_tilelang_forward(
        q,
        k,
        v,
        W,
        xf,
        config=TiledTileLangConfig(max_its=8, tile_len=tile_len, backend="torch"),
    )

    torch.testing.assert_close(out_tiled, out_ref, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(h_tiled, h_ref, atol=1e-9, rtol=1e-9)


def test_tiled_memory_contract_reports_tile_bounded_jacobian():
    device = "cpu"
    dtype = torch.float32
    B, S, H, k_dim, v_dim = 2, 65, 3, 5, 16
    tile_len = 32
    q, k, v, W, xf = _make_inputs(B, S, H, k_dim, v_dim, device=device, dtype=dtype)

    _out, _h, stats = m2rnn_pararnn_tiled_tilelang_forward(
        q,
        k,
        v,
        W,
        xf,
        config=TiledTileLangConfig(max_its=1, tile_len=tile_len, backend="torch"),
        return_stats=True,
    )

    be = B * H * k_dim
    assert stats.max_tile_jac_elements == be * tile_len * v_dim * v_dim
    assert stats.max_tile_jac_elements < stats.full_jac_elements_avoided
    assert stats.summary_a_elements == be * math.ceil(S / tile_len) * v_dim * v_dim
    assert stats.summary_b_elements == be * math.ceil(S / tile_len) * v_dim
    assert stats.torch_materialized_tile_jac_elements == be * tile_len * v_dim * v_dim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang kernels")
@pytest.mark.parametrize("tile_len", [16, 32, 64])
def test_tilelang_summary_apply_matches_full_pararnn_scan_cuda(tile_len):
    pytest.importorskip("tilelang")
    device = "cuda"
    dtype = torch.float32
    q, k, v, W, xf = _make_inputs(1, 65, 2, 4, 16, device=device, dtype=dtype)

    out_tiled, h_tiled, stats = m2rnn_pararnn_tiled_tilelang_forward(
        q,
        k,
        v,
        W,
        xf,
        config=TiledTileLangConfig(
            max_its=3,
            tile_len=tile_len,
            backend="tilelang",
            allow_tilelang_fallback=False,
        ),
        return_stats=True,
    )
    out_full, h_full = m2rnn_pararnn_forward(
        q,
        k,
        v,
        W,
        xf,
        config=PararnnConfig(max_its=3, chunk_size=0),
    )

    torch.testing.assert_close(out_tiled, out_full, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(h_tiled, h_full, atol=2e-6, rtol=2e-6)
    assert stats.backend_used == "tilelang-summary+tilelang-apply"
    assert stats.tilelang_summary_used
    assert stats.tilelang_apply_used
    assert stats.torch_materialized_tile_jac_elements == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TileLang kernels")
def test_tilelang_bf16_callers_use_fp32_solve_buffers_cuda():
    pytest.importorskip("tilelang")
    device = "cuda"
    dtype = torch.bfloat16
    q, k, v, W, xf = _make_inputs(1, 33, 1, 2, 16, device=device, dtype=dtype)

    out_tiled, h_tiled, stats = m2rnn_pararnn_tiled_tilelang_forward(
        q,
        k,
        v,
        W,
        xf,
        config=TiledTileLangConfig(
            max_its=2,
            tile_len=32,
            backend="tilelang",
            allow_tilelang_fallback=False,
        ),
        return_stats=True,
    )
    out_full, h_full = m2rnn_pararnn_forward(
        q,
        k,
        v,
        W,
        xf,
        config=PararnnConfig(max_its=2, chunk_size=0),
    )

    assert out_tiled.dtype == torch.bfloat16
    assert h_tiled.dtype == torch.bfloat16
    assert stats.backend_used == "tilelang-summary+tilelang-apply"
    torch.testing.assert_close(out_tiled.float(), out_full.float(), atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(h_tiled.float(), h_full.float(), atol=8e-3, rtol=8e-3)
