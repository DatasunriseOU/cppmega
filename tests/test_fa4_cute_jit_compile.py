"""CuTe JIT compilation validation for the FA4 production score_mod callbacks.

Tests that the PRODUCTION ``_make_graph_score_mod`` factory from
``cppmega.megatron.fa4_score_mod_adapter`` compiles and runs correctly
through FlashAttention-4's CuTe DSL JIT when passed to ``flash_attn_func``.

The production factory captures ``c_plus_1`` in closure and expects 6 flat
aux_tensors (chunk_bias flattened to [B, (C+1)*(C+1)]).  This test verifies:

1. Forward pass succeeds with the production callback
2. Backward pass succeeds with _make_graph_score_mod_bwd
3. Output shapes and dtypes are correct

These tests require an H200/B200 GPU with ``flash-attn-4`` (beta23+)
installed.  On machines without a CUDA GPU or without ``flash_attn.cute``
the whole module is skipped.

Run with (on a GPU machine with flash-attn-4 installed)::

    pytest tests/test_fa4_cute_jit_compile.py -v
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch", reason="CuTe JIT validation requires torch")


def _flash_attn_cute_is_real() -> bool:
    """Return True only if a *real* (non-mocked) ``flash_attn.cute`` imports.

    Sibling test modules install ``MagicMock`` stand-ins for
    ``flash_attn.cute`` into ``sys.modules`` so they can run CPU-only.  If such
    a mock is active, JIT "compilation" would be meaningless, so we treat it as
    unavailable here.
    """
    try:
        import flash_attn.cute as _cute  # noqa: F401
    except Exception:
        return False
    if isinstance(_cute, MagicMock):
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CuTe JIT score_mod validation requires a CUDA GPU (H200/B200)",
    ),
    pytest.mark.skipif(
        not _flash_attn_cute_is_real(),
        reason="CuTe JIT score_mod validation requires flash_attn.cute "
        "(flash-attn-4 beta23+); not installed or mocked",
    ),
]

# Pinned GHCR image digest (immutable, reproducible)
_GHCR_IMAGE_DIGEST = (
    "sha256:08c5db7368d1037d930e0825281468927de9c85b12ba10373fe07e082150d983"
)


# ---------------------------------------------------------------------------
# Production imports (deferred to test time to avoid import errors on CPU)
# ---------------------------------------------------------------------------


def _import_production_factory():
    """Import the production _make_graph_score_mod factory."""
    from cppmega.megatron.fa4_score_mod_adapter import (
        _make_graph_score_mod,
        _make_graph_score_mod_bwd,
    )
    return _make_graph_score_mod, _make_graph_score_mod_bwd


# ---------------------------------------------------------------------------
# Aux tensor builder (matches production ChunkNativeGraphBias layout)
# ---------------------------------------------------------------------------


def _build_production_aux_tensors(
    batch: int,
    seqlen: int,
    num_chunks: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], int]:
    """Build production-layout aux tensors with flat chunk_bias.

    Returns (aux_tensors, c_plus_1) where aux_tensors has 6 elements:
        [0] token_to_chunk_q  [B, S] int32
        [1] token_to_chunk_k  [B, S] int32
        [2] chunk_bias_flat   [B, (C+1)*(C+1)] float32
        [3] rare_q            [B, max_rare] int32
        [4] rare_k            [B, max_rare] int32
        [5] rare_w            [B, max_rare] float32
    """
    c_plus_1 = num_chunks + 1
    max_rare = 4  # minimal for testing

    token_to_chunk_q = torch.zeros(batch, seqlen, dtype=torch.int32, device=device)
    token_to_chunk_k = torch.zeros(batch, seqlen, dtype=torch.int32, device=device)

    # Flat chunk_bias [B, (C+1)*(C+1)]
    chunk_bias_flat = torch.zeros(
        batch, c_plus_1 * c_plus_1, dtype=torch.float32, device=device
    )

    rare_q = torch.zeros(batch, max_rare, dtype=torch.int32, device=device)
    rare_k = torch.full((batch, max_rare), -1, dtype=torch.int32, device=device)
    rare_w = torch.zeros(batch, max_rare, dtype=torch.float32, device=device)

    aux_tensors = [
        token_to_chunk_q,
        token_to_chunk_k,
        chunk_bias_flat,
        rare_q,
        rare_k,
        rare_w,
    ]
    return aux_tensors, c_plus_1


# ---------------------------------------------------------------------------
# 1. Production factory forward pass through flash_attn_func
# ---------------------------------------------------------------------------


def test_production_score_mod_forward() -> None:
    """Production _make_graph_score_mod forward via flash_attn_func succeeds.

    This is the ultimate proof: ``flash_attn_func`` traces and JIT-compiles
    the production score_mod callback into a real Hopper/Blackwell kernel
    and runs it.  A callback with wrong aux_tensors layout or arity would
    raise here.
    """
    from flash_attn.cute.interface import flash_attn_func

    _make_graph_score_mod, _make_graph_score_mod_bwd = _import_production_factory()

    device = torch.device("cuda")
    batch, seqlen, heads, head_dim = 1, 128, 4, 64
    num_chunks = 4
    dtype = torch.bfloat16

    q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)

    aux_tensors, c_plus_1 = _build_production_aux_tensors(
        batch, seqlen, num_chunks, device
    )

    # Create score_mod via production factory
    score_mod = _make_graph_score_mod(c_plus_1)
    score_mod_bwd = _make_graph_score_mod_bwd(c_plus_1)

    out = flash_attn_func(
        q=q,
        k=k,
        v=v,
        softmax_scale=head_dim ** -0.5,
        causal=True,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        aux_tensors=aux_tensors,
        block_sparse_tensors=None,
        mask_mod=None,
        return_lse=False,
    )
    if isinstance(out, tuple):
        out = out[0]

    assert out.shape == q.shape, f"expected {tuple(q.shape)}, got {tuple(out.shape)}"
    assert out.dtype == dtype


# ---------------------------------------------------------------------------
# 2. Production factory backward pass through flash_attn_func
# ---------------------------------------------------------------------------


def test_production_score_mod_backward() -> None:
    """Production _make_graph_score_mod_bwd backward via flash_attn_func.

    Verifies that the backward pass (gradient computation) succeeds with
    the production score_mod_bwd callback.  The identity backward
    (d(score')/d(score) = 1) should produce valid gradients for Q, K, V.
    """
    from flash_attn.cute.interface import flash_attn_func

    _make_graph_score_mod, _make_graph_score_mod_bwd = _import_production_factory()

    device = torch.device("cuda")
    batch, seqlen, heads, head_dim = 1, 128, 4, 64
    num_chunks = 4
    dtype = torch.bfloat16

    q = torch.randn(
        batch, seqlen, heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch, seqlen, heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen, heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )

    aux_tensors, c_plus_1 = _build_production_aux_tensors(
        batch, seqlen, num_chunks, device
    )

    # Create score_mod via production factory
    score_mod = _make_graph_score_mod(c_plus_1)
    score_mod_bwd = _make_graph_score_mod_bwd(c_plus_1)

    out = flash_attn_func(
        q=q,
        k=k,
        v=v,
        softmax_scale=head_dim ** -0.5,
        causal=True,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        aux_tensors=aux_tensors,
        block_sparse_tensors=None,
        mask_mod=None,
        return_lse=False,
    )
    if isinstance(out, tuple):
        out = out[0]

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Verify gradients exist and have correct shape
    assert q.grad is not None, "dQ is None after backward"
    assert k.grad is not None, "dK is None after backward"
    assert v.grad is not None, "dV is None after backward"
    assert q.grad.shape == q.shape, f"dQ shape mismatch: {q.grad.shape} vs {q.shape}"
    assert k.grad.shape == k.shape, f"dK shape mismatch: {k.grad.shape} vs {k.shape}"
    assert v.grad.shape == v.shape, f"dV shape mismatch: {v.grad.shape} vs {v.shape}"

    # Gradients should be finite (no NaN/Inf from broken score_mod_bwd)
    assert torch.isfinite(q.grad).all(), "dQ contains NaN/Inf"
    assert torch.isfinite(k.grad).all(), "dK contains NaN/Inf"
    assert torch.isfinite(v.grad).all(), "dV contains NaN/Inf"


# ---------------------------------------------------------------------------
# 3. Production factory with non-trivial bias values
# ---------------------------------------------------------------------------


def test_production_score_mod_nonzero_bias() -> None:
    """Production score_mod with non-zero chunk_bias produces biased output.

    Verifies that the flat chunk_bias indexing (qc * c_plus_1 + kc) correctly
    applies additive bias by comparing output with and without bias.
    """
    from flash_attn.cute.interface import flash_attn_func

    _make_graph_score_mod, _make_graph_score_mod_bwd = _import_production_factory()

    device = torch.device("cuda")
    batch, seqlen, heads, head_dim = 1, 64, 2, 64
    num_chunks = 2
    dtype = torch.bfloat16

    torch.manual_seed(123)
    q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)

    # Zero bias
    aux_zero, c_plus_1 = _build_production_aux_tensors(
        batch, seqlen, num_chunks, device
    )

    # Non-zero bias: set chunk (0,0) pair to a large value
    aux_nonzero, _ = _build_production_aux_tensors(
        batch, seqlen, num_chunks, device
    )
    # chunk_bias_flat[b, qc * c_plus_1 + kc] for (qc=0, kc=0) -> index 0
    aux_nonzero[2] = aux_nonzero[2].clone()
    aux_nonzero[2][0, 0] = 5.0  # strong bias for chunk pair (0, 0)

    score_mod = _make_graph_score_mod(c_plus_1)
    score_mod_bwd = _make_graph_score_mod_bwd(c_plus_1)

    out_zero = flash_attn_func(
        q=q, k=k, v=v,
        softmax_scale=head_dim ** -0.5,
        causal=True,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        aux_tensors=aux_zero,
        block_sparse_tensors=None,
        mask_mod=None,
        return_lse=False,
    )
    if isinstance(out_zero, tuple):
        out_zero = out_zero[0]

    out_biased = flash_attn_func(
        q=q, k=k, v=v,
        softmax_scale=head_dim ** -0.5,
        causal=True,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        aux_tensors=aux_nonzero,
        block_sparse_tensors=None,
        mask_mod=None,
        return_lse=False,
    )
    if isinstance(out_biased, tuple):
        out_biased = out_biased[0]

    # Outputs should differ due to the bias
    diff = (out_biased - out_zero).abs().max().item()
    assert diff > 1e-4, (
        f"Non-zero chunk_bias did not affect output (max diff={diff}); "
        "flat indexing may be broken"
    )
