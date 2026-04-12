"""Unit tests for :mod:`cppmega.megatron.dsa_tilelang_fused_kl`.

Compares the one-pass online-softmax implementation
(``attention_target_fused_kl``) against the head-streaming reference
(``_attention_target_fp32`` from ``dsa_fp8_indexer``).

Tolerance: abs<=0.1, rel<=0.1.  The online softmax reorders floating-point
additions compared to the standard two-pass ``F.softmax``, so results are
NOT bit-identical.  The tolerance reflects the expected numerical drift for
bf16 inputs with Gaussian magnitudes.

The monkey-patch routing via ``CPPMEGA_DSA_KL_MODE=tilelang_fused`` is also
tested (``apply_dsa_kl_mode_patch`` in ``dsa_fp8_patch``).
"""

from __future__ import annotations

import os

import pytest
import torch

from cppmega.megatron.dsa_fp8_indexer import _attention_target_fp32
from cppmega.megatron.dsa_tilelang_fused_kl import attention_target_fused_kl


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_inputs(
    *,
    sq: int,
    b: int,
    np_: int,
    hn: int,
    sk: int,
    topk: int,
    seed: int = 42,
    device: str = "cpu",
):
    """Generate matching (query, key, topk_indices) tensors."""
    g = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.bfloat16
    query = torch.randn(sq, b, np_, hn, generator=g, device=device, dtype=dtype)
    key = torch.randn(sk, b, np_, hn, generator=g, device=device, dtype=dtype)
    topk_indices = torch.stack(
        [torch.randperm(sk, generator=g, device=device)[:topk] for _ in range(b * sq)]
    ).reshape(b, sq, topk)
    softmax_scale = 1.0 / (hn ** 0.5)
    return query, key, softmax_scale, topk_indices


# ---------------------------------------------------------------------------
# Parity tests: fused KL vs head-streaming reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sparse_loss", [False, True])
@pytest.mark.parametrize(
    "sq,b,np_,hn,sk,topk",
    [
        (16, 2, 4, 32, 16, 4),
        (32, 1, 8, 64, 32, 8),
        (8, 2, 2, 16, 12, 3),
        # Asymmetric sq != sk.
        (16, 1, 4, 32, 24, 6),
    ],
)
def test_fused_kl_matches_head_streaming(sq, b, np_, hn, sk, topk, sparse_loss):
    """Output parity: attention_target_fused_kl vs _attention_target_fp32."""
    query, key, softmax_scale, topk_indices = _make_inputs(
        sq=sq, b=b, np_=np_, hn=hn, sk=sk, topk=topk, seed=7
    )

    ref_norm, ref_mask = _attention_target_fp32(
        query, key, softmax_scale, topk_indices, sparse_loss, pg_collection=None
    )
    fused_norm, fused_mask = attention_target_fused_kl(
        query, key, softmax_scale, topk_indices, sparse_loss, pg_collection=None
    )

    # Shapes must match exactly.
    assert ref_norm.shape == fused_norm.shape == (b, sq, sk)
    assert ref_mask.shape == fused_mask.shape == (b, sq, sk)

    # Masks must be identical (they are computed the same way).
    torch.testing.assert_close(ref_mask, fused_mask, atol=0, rtol=0)

    # Normalised attention scores: allow tolerance for FP reordering.
    # The reference produces NaN for fully-masked rows (all logits -inf)
    # because F.softmax([-inf,...,-inf]) = [nan,...,nan].  The fused
    # implementation produces 0 for those rows (more correct).  We
    # replace NaN with 0 in the reference before comparing so the test
    # validates the non-degenerate rows.
    ref_norm_clean = torch.nan_to_num(ref_norm, nan=0.0)
    fused_norm_clean = torch.nan_to_num(fused_norm, nan=0.0)
    torch.testing.assert_close(
        fused_norm_clean, ref_norm_clean, atol=0.1, rtol=0.1,
    )


@pytest.mark.parametrize("sparse_loss", [False, True])
def test_fused_kl_output_is_normalised(sparse_loss):
    """The output must be a valid probability distribution (sums to 1 per row).

    With sparse_loss=True, some rows may be fully masked (no valid topk
    positions in the causal window), producing an all-zero row.  We only
    check rows that have at least one non-zero entry.
    """
    query, key, softmax_scale, topk_indices = _make_inputs(
        sq=16, b=2, np_=4, hn=32, sk=16, topk=4
    )
    norm, _ = attention_target_fused_kl(
        query, key, softmax_scale, topk_indices, sparse_loss, pg_collection=None
    )
    row_sums = norm.sum(dim=-1)
    # Mask out fully-masked rows (sum == 0) before checking normalisation.
    valid_rows = row_sums > 1e-8
    if valid_rows.any():
        torch.testing.assert_close(
            row_sums[valid_rows],
            torch.ones_like(row_sums[valid_rows]),
            atol=1e-5,
            rtol=1e-5,
        )


def test_fused_kl_dtype_and_device():
    """Output dtype must be fp32 regardless of input dtype."""
    query, key, softmax_scale, topk_indices = _make_inputs(
        sq=8, b=1, np_=2, hn=16, sk=8, topk=2
    )
    norm, mask = attention_target_fused_kl(
        query, key, softmax_scale, topk_indices, False, pg_collection=None
    )
    assert norm.dtype == torch.float32
    assert mask.dtype == torch.float32


# ---------------------------------------------------------------------------
# Monkey-patch routing test
# ---------------------------------------------------------------------------


def test_kl_mode_env_routing(monkeypatch):
    """CPPMEGA_DSA_KL_MODE=tilelang_fused must route through the patch."""
    from cppmega.megatron.dsa_fp8_patch import resolve_kl_mode

    monkeypatch.setenv("CPPMEGA_DSA_KL_MODE", "tilelang_fused")
    assert resolve_kl_mode() == "tilelang_fused"

    monkeypatch.setenv("CPPMEGA_DSA_KL_MODE", "tilelang-fused")
    assert resolve_kl_mode() == "tilelang_fused"

    monkeypatch.setenv("CPPMEGA_DSA_KL_MODE", "tilelang")
    assert resolve_kl_mode() == "tilelang_fused"

    monkeypatch.setenv("CPPMEGA_DSA_KL_MODE", "")
    assert resolve_kl_mode() == "head_streaming"

    monkeypatch.delenv("CPPMEGA_DSA_KL_MODE", raising=False)
    assert resolve_kl_mode() == "head_streaming"


def test_apply_kl_mode_patch_replaces_function(monkeypatch):
    """apply_dsa_kl_mode_patch must swap _attention_target_fp32 in the module."""
    import cppmega.megatron.dsa_fp8_indexer as indexer_mod
    from cppmega.megatron.dsa_fp8_patch import apply_dsa_kl_mode_patch

    # Save original so we can restore after test.
    original = indexer_mod._attention_target_fp32

    monkeypatch.setenv("CPPMEGA_DSA_KL_MODE", "tilelang_fused")
    try:
        result = apply_dsa_kl_mode_patch(force=True)
        assert result is True
        # The function should have been replaced.
        assert indexer_mod._attention_target_fp32 is not original
        # It should have the patch marker.
        assert getattr(
            indexer_mod._attention_target_fp32,
            "__cppmega_dsa_kl_tilelang_fused_patched__",
            False,
        )
    finally:
        # Restore to avoid polluting other tests.
        indexer_mod._attention_target_fp32 = original


def test_apply_kl_mode_patch_idempotent(monkeypatch):
    """Calling apply_dsa_kl_mode_patch twice without force must be a no-op."""
    import cppmega.megatron.dsa_fp8_indexer as indexer_mod
    from cppmega.megatron.dsa_fp8_patch import apply_dsa_kl_mode_patch

    original = indexer_mod._attention_target_fp32

    monkeypatch.setenv("CPPMEGA_DSA_KL_MODE", "tilelang_fused")
    try:
        apply_dsa_kl_mode_patch(force=True)
        first_replacement = indexer_mod._attention_target_fp32
        apply_dsa_kl_mode_patch(force=False)
        # Should be the same object (no double-wrap).
        assert indexer_mod._attention_target_fp32 is first_replacement
    finally:
        indexer_mod._attention_target_fp32 = original


def test_apply_kl_mode_patch_skips_head_streaming(monkeypatch):
    """When mode is head_streaming, apply_dsa_kl_mode_patch must return False."""
    from cppmega.megatron.dsa_fp8_patch import apply_dsa_kl_mode_patch

    monkeypatch.setenv("CPPMEGA_DSA_KL_MODE", "head_streaming")
    assert apply_dsa_kl_mode_patch() is False

    monkeypatch.delenv("CPPMEGA_DSA_KL_MODE", raising=False)
    assert apply_dsa_kl_mode_patch() is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_fused_kl_single_position():
    """sq=1, sk=1 edge case must not crash and must produce valid output."""
    query, key, softmax_scale, topk_indices = _make_inputs(
        sq=1, b=1, np_=2, hn=16, sk=1, topk=1
    )
    norm, mask = attention_target_fused_kl(
        query, key, softmax_scale, topk_indices, False, pg_collection=None
    )
    assert norm.shape == (1, 1, 1)
    # Single position: softmax of a single element is 1.0.
    # After L1 normalisation of np_ heads each contributing 1.0, result is 1.0.
    torch.testing.assert_close(norm, torch.ones(1, 1, 1), atol=1e-5, rtol=1e-5)


def test_fused_kl_sk_not_divisible_by_tile():
    """sk that is not a multiple of _TILE_SK (64) must work correctly."""
    # sk=100 is not divisible by 64.
    query, key, softmax_scale, topk_indices = _make_inputs(
        sq=16, b=1, np_=2, hn=16, sk=100, topk=4
    )
    ref_norm, ref_mask = _attention_target_fp32(
        query, key, softmax_scale, topk_indices, False, pg_collection=None
    )
    fused_norm, fused_mask = attention_target_fused_kl(
        query, key, softmax_scale, topk_indices, False, pg_collection=None
    )
    torch.testing.assert_close(fused_norm, ref_norm, atol=0.1, rtol=0.1)
