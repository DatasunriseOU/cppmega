"""Tests for the tiled/streaming Triton-first M2RNN ParaRNN prototype."""

from __future__ import annotations

import math

import pytest
import torch

from cppmega.megatron.m2rnn_pararnn import PararnnConfig, m2rnn_pararnn_forward
from cppmega.megatron.m2rnn_pararnn_tiled_triton import (
    TRITON_AVAILABLE,
    TiledTritonConfig,
    _scan_summaries_torch,
    _scan_summaries_triton,
    estimate_tiled_solve_memory,
    m2rnn_pararnn_tiled_triton_forward,
)


def _make_inputs(B, S, H, K, V, *, device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, S, H, K, generator=g, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, S, H, K, generator=g, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, S, H, V, generator=g, device=device, dtype=dtype) * 0.5
    W = torch.randn(H, V, V, generator=g, device=device, dtype=dtype) * (0.35 / math.sqrt(V))
    xf = torch.sigmoid(torch.randn(B, S, H, generator=g, device=device, dtype=dtype) - 0.5)
    return q, k, v, W, xf


def test_one_newton_iteration_matches_dense_pararnn_cpu():
    B, S, H, K, V = 1, 17, 2, 3, 4
    q, k, v, W, xf = _make_inputs(B, S, H, K, V, device="cpu", dtype=torch.float64)

    out_dense, h_dense = m2rnn_pararnn_forward(
        q,
        k,
        v,
        W,
        xf,
        config=PararnnConfig(max_its=1, init_strategy="zero", chunk_size=0),
    )
    out_tiled, h_tiled, stats = m2rnn_pararnn_tiled_triton_forward(
        q,
        k,
        v,
        W,
        xf,
        config=TiledTritonConfig(max_its=1, init_strategy="zero", tile_size=5, prefer_triton=False),
        return_stats=True,
    )

    torch.testing.assert_close(out_tiled, out_dense, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(h_tiled, h_dense, atol=1e-12, rtol=1e-12)
    assert stats.full_A_bytes == B * H * K * S * V * V * q.element_size()
    assert stats.peak_tile_A_bytes == B * H * K * 5 * V * V * q.element_size()
    assert stats.avoids_full_A


def test_multi_iteration_matches_dense_pararnn_and_sequential_floor_cpu():
    B, S, H, K, V = 1, 32, 2, 4, 4
    q, k, v, W, xf = _make_inputs(B, S, H, K, V, device="cpu", dtype=torch.float64, seed=1)

    out_dense, h_dense = m2rnn_pararnn_forward(
        q,
        k,
        v,
        W,
        xf,
        config=PararnnConfig(max_its=7, init_strategy="zero", chunk_size=0),
    )
    out_tiled, h_tiled = m2rnn_pararnn_tiled_triton_forward(
        q,
        k,
        v,
        W,
        xf,
        config=TiledTritonConfig(max_its=7, init_strategy="zero", tile_size=8, prefer_triton=False),
    )

    torch.testing.assert_close(out_tiled, out_dense, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(h_tiled, h_dense, atol=1e-10, rtol=1e-10)


def test_memory_accounting_bounds_full_A_for_production_like_shape():
    stats = estimate_tiled_solve_memory(B=2, S=4096, H=8, K=64, V=16, tile_size=64)

    assert stats.full_A_bytes == 2 * 8 * 64 * 4096 * 16 * 16 * 4
    assert stats.peak_tile_A_bytes == 2 * 8 * 64 * 64 * 16 * 16 * 4
    assert stats.summary_bytes == (
        2 * 8 * 64 * 64 * 16 * 16 * 4 + 2 * 8 * 64 * 64 * 16 * 4
    )
    assert stats.full_A_to_tile_ratio == pytest.approx(64.0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="Triton tiled prototype requires CUDA and triton",
)
def test_triton_path_matches_torch_streaming_one_iteration_cuda():
    B, S, H, K, V = 1, 19, 1, 2, 4
    q, k, v, W, xf = _make_inputs(B, S, H, K, V, device="cuda", dtype=torch.float32, seed=2)

    cfg_torch = TiledTritonConfig(max_its=1, tile_size=7, prefer_triton=False)
    cfg_triton = TiledTritonConfig(max_its=1, tile_size=7, prefer_triton=True)
    out_torch, h_torch = m2rnn_pararnn_tiled_triton_forward(q, k, v, W, xf, config=cfg_torch)
    out_tri, h_tri = m2rnn_pararnn_tiled_triton_forward(q, k, v, W, xf, config=cfg_triton)

    torch.testing.assert_close(out_tri, out_torch, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(h_tri, h_torch, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="Triton tiled prototype requires CUDA and triton",
)
def test_triton_one_tile_fast_path_matches_torch_streaming_cuda():
    B, S, H, K, V = 1, 16, 1, 2, 4
    q, k, v, W, xf = _make_inputs(B, S, H, K, V, device="cuda", dtype=torch.float32, seed=5)

    cfg_torch = TiledTritonConfig(max_its=3, tile_size=S, prefer_triton=False)
    cfg_triton = TiledTritonConfig(max_its=3, tile_size=S, prefer_triton=True)
    out_torch, h_torch = m2rnn_pararnn_tiled_triton_forward(q, k, v, W, xf, config=cfg_torch)
    out_tri, h_tri = m2rnn_pararnn_tiled_triton_forward(q, k, v, W, xf, config=cfg_triton)

    torch.testing.assert_close(out_tri, out_torch, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(h_tri, h_torch, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="Triton summary scan requires CUDA and triton",
)
def test_triton_summary_scan_matches_torch_cuda():
    Be, num_tiles, V = 5, 9, 4
    g = torch.Generator(device="cuda").manual_seed(3)
    summaries_M = torch.randn(Be, num_tiles, V, V, generator=g, device="cuda") * 0.05
    summaries_b = torch.randn(Be, num_tiles, V, generator=g, device="cuda") * 0.25

    carries_torch = _scan_summaries_torch(summaries_M, summaries_b)
    carries_triton = _scan_summaries_triton(summaries_M, summaries_b, block_v=16)

    torch.testing.assert_close(carries_triton, carries_torch, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="Triton tiled prototype requires CUDA and triton",
)
def test_bfloat16_inputs_match_dense_pararnn_cuda():
    B, S, H, K, V = 1, 23, 1, 2, 4
    q, k, v, W, xf = _make_inputs(B, S, H, K, V, device="cuda", dtype=torch.bfloat16, seed=4)

    cfg_dense = PararnnConfig(max_its=2, init_strategy="zero", chunk_size=0)
    cfg_triton = TiledTritonConfig(max_its=2, init_strategy="zero", tile_size=7, prefer_triton=True)
    out_dense, h_dense = m2rnn_pararnn_forward(q, k, v, W, xf, config=cfg_dense)
    out_tri, h_tri = m2rnn_pararnn_tiled_triton_forward(q, k, v, W, xf, config=cfg_triton)

    assert out_tri.dtype == torch.bfloat16
    assert h_tri.dtype == torch.bfloat16
    torch.testing.assert_close(out_tri.float(), out_dense.float(), atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(h_tri.float(), h_dense.float(), atol=8e-3, rtol=8e-3)
