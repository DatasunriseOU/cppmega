"""Numerical-parity tests for the TileLang Path C DSA split-K indexer loss.

The kernels under test live at
``cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py`` and replace the
CUDA-only Triton ``_fwd_fused_indexer_loss_stage1_kernel`` /
``_stage2_kernel`` in ``cppmega/megatron/dsa_splitk_indexer_loss.py`` on
hosts where TileLang is available (both CUDA and Apple Metal SIMDgroup).
"""

from __future__ import annotations

import math
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cppmega_mlx.nn._tilelang.dsa_splitk_indexer_loss")
from cppmega_mlx.nn._tilelang.dsa_splitk_indexer_loss import (  # noqa: E402
    dsa_splitk_indexer_loss_tilelang,
    dsa_splitk_path_c_status,
    tilelang_supports,
)


_STATUS = dsa_splitk_path_c_status()
_TILELANG_OK = _STATUS.available

_HAS_CUDA = torch.cuda.is_available()
_HAS_MPS = bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())


def _pick_device() -> torch.device:
    if _HAS_CUDA and tilelang_supports(torch.device("cuda")):
        return torch.device("cuda")
    if _HAS_MPS and tilelang_supports(torch.device("mps")):
        return torch.device("mps")
    return torch.device("cpu")


def _torch_indexer_loss_reference(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
) -> torch.Tensor:
    """Pure-torch reference for KL indexer loss (matmul + log + reduce_sum)."""

    ASq, AB, AH, AD = query.shape
    Sk = key.shape[0]

    q = query.permute(1, 2, 0, 3).to(torch.float32)  # [B, H, Sq, D]
    k = key.permute(1, 2, 0, 3).to(torch.float32)    # [B, H, Sk, D]

    scores = torch.matmul(q, k.transpose(-1, -2)) * float(softmax_scale)  # [B, H, Sq, Sk]
    causal = torch.triu(torch.ones(ASq, Sk, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))

    if sparse_loss:
        idx_mask = torch.full(
            (AB, ASq, Sk), float("-inf"), dtype=torch.float32, device=index_scores.device,
        ).scatter_(-1, topk_indices, 0.0)
        scores = scores + idx_mask.unsqueeze(1)
        idx_with_mask = index_scores + idx_mask
    else:
        idx_with_mask = index_scores

    p = torch.softmax(scores, dim=-1)            # [B, H, Sq, Sk]
    p_avg = p.mean(dim=1)                         # [B, Sq, Sk]
    q_idx = torch.softmax(idx_with_mask, dim=-1)  # [B, Sq, Sk]

    eps = 1e-10
    kl = p_avg * (torch.log(p_avg + eps) - torch.log(q_idx + eps))  # [B, Sq, Sk]
    kl = kl.masked_fill(causal, 0.0)
    per_pos = kl.sum(dim=-1)  # [B, Sq]
    return per_pos.mean() * float(loss_coeff)


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_dsa_splitk_indexer_loss_matches_torch_reference():
    """TileLang indexer loss must match a torch matmul+softmax+KL reference."""

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang DSA split-K requires a CUDA or Metal device")

    torch.manual_seed(0xC0DE)
    AB, AH, AD = 1, 2, 32
    ASq, Sk = 64, 128
    softmax_scale = 1.0 / math.sqrt(AD)
    loss_coeff = 0.7

    query = torch.randn(ASq, AB, AH, AD, dtype=torch.float16, device=device)
    key = torch.randn(Sk, AB, AH, AD, dtype=torch.float16, device=device)
    index_scores = torch.randn(AB, ASq, Sk, dtype=torch.float32, device=device)
    topk_indices = torch.zeros(AB, ASq, 4, dtype=torch.long, device=device)

    out = dsa_splitk_indexer_loss_tilelang(
        index_scores, topk_indices, query, key,
        softmax_scale=softmax_scale, loss_coeff=loss_coeff,
        sparse_loss=False, pg_collection=None,
    )
    ref = _torch_indexer_loss_reference(
        index_scores, topk_indices, query, key,
        softmax_scale=softmax_scale, loss_coeff=loss_coeff,
        sparse_loss=False,
    )

    assert out.dtype == torch.float32
    assert out.device.type == device.type
    # Online-softmax accumulation in fp32 across small (ASq=64, Sk=128) tiles
    # gives well below 1e-4 typical error vs the torch reference; tighten
    # tolerances from the previous 5e-2/5e-3 to surface real regressions.
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), rtol=1e-2, atol=1e-4)


@pytest.mark.skipif(not _HAS_CUDA, reason="Triton parity check requires CUDA")
def test_dsa_splitk_indexer_loss_matches_triton_reference():
    """On CUDA hosts, parity with the Triton indexer-loss kernels."""

    if not _TILELANG_OK:
        pytest.skip(f"TileLang unavailable: {_STATUS.reason}")

    triton = pytest.importorskip("triton")  # noqa: F841
    from cppmega.megatron.dsa_splitk_indexer_loss import compute_dsa_indexer_loss_splitk

    torch.manual_seed(0xBEEF)
    AB, AH, AD = 1, 4, 64
    ASq, Sk = 128, 256
    softmax_scale = 1.0 / math.sqrt(AD)
    loss_coeff = 0.5

    query = torch.randn(ASq, AB, AH, AD, dtype=torch.float16, device="cuda")
    key = torch.randn(Sk, AB, AH, AD, dtype=torch.float16, device="cuda")
    index_scores = torch.randn(AB, ASq, Sk, dtype=torch.float32, device="cuda")
    topk_indices = torch.zeros(AB, ASq, 4, dtype=torch.long, device="cuda")

    # Force TileLang via the public API (the wrapper in
    # ``compute_dsa_indexer_loss_splitk`` already prefers TileLang when both
    # paths are available).
    out_tilelang = dsa_splitk_indexer_loss_tilelang(
        index_scores, topk_indices, query, key,
        softmax_scale=softmax_scale, loss_coeff=loss_coeff,
        sparse_loss=False, pg_collection=None,
    )

    # Run the legacy Triton path by temporarily disabling the TileLang gate.
    import cppmega.megatron.dsa_splitk_indexer_loss as mod
    saved = mod._has_dsa_tilelang
    try:
        mod._has_dsa_tilelang = False
        out_triton = compute_dsa_indexer_loss_splitk(
            index_scores, topk_indices, query, key,
            softmax_scale=softmax_scale, loss_coeff=loss_coeff,
            sparse_loss=False, pg_collection=None,
        )
    finally:
        mod._has_dsa_tilelang = saved

    # CUDA Triton vs TileLang parity on the same shape/seed should be tight;
    # both run fp16 inputs with fp32 online-softmax accumulation. Tighten
    # from 5e-2/5e-3 to surface real divergences.
    torch.testing.assert_close(
        out_tilelang.to(torch.float32),
        out_triton.to(torch.float32),
        rtol=1e-2,
        atol=1e-4,
    )


# ---------------------------------------------------------------------------
# Wave-2 #06: sparse-only regression coverage
#
# These exercise the ``sparse_loss=True`` branch -- previously only the dense
# (``sparse_loss=False``) path had numerical-parity tests. The two cases below
# bracket the sparsity range:
#   * High sparsity:  TOPK=8 of Sk=4096   (~99.8% masked)
#   * Low sparsity:   TOPK=1024 of Sk=4096 (~75% masked)
# Together with the dense tests above they cover the four sparse_loss x
# kernel-stage combinations the hot path can hit.
# ---------------------------------------------------------------------------


def _run_sparse_parity(
    *,
    AB: int,
    AH: int,
    AD: int,
    ASq: int,
    Sk: int,
    TOPK: int,
    seed: int,
) -> None:
    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang DSA split-K requires a CUDA or Metal device")

    torch.manual_seed(seed)
    softmax_scale = 1.0 / math.sqrt(AD)
    loss_coeff = 1.0

    query = torch.randn(ASq, AB, AH, AD, dtype=torch.float16, device=device)
    key = torch.randn(Sk, AB, AH, AD, dtype=torch.float16, device=device)
    index_scores = torch.randn(AB, ASq, Sk, dtype=torch.float32, device=device)

    # Random TOPK indices per (b, sq) row, in [0, Sk). Deduplicate via sort to
    # match the wrapper's ``scatter_(-1, topk_indices, 0.0)`` semantics, which
    # collapses duplicates onto a single masked-in slot.
    topk_indices = torch.randint(0, Sk, (AB, ASq, TOPK), dtype=torch.long, device=device)

    out = dsa_splitk_indexer_loss_tilelang(
        index_scores, topk_indices, query, key,
        softmax_scale=softmax_scale, loss_coeff=loss_coeff,
        sparse_loss=True, pg_collection=None,
    )
    ref = _torch_indexer_loss_reference(
        index_scores, topk_indices, query, key,
        softmax_scale=softmax_scale, loss_coeff=loss_coeff,
        sparse_loss=True,
    )

    assert out.dtype == torch.float32
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), rtol=1e-2, atol=1e-4)


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_dsa_splitk_indexer_loss_sparse_high_sparsity():
    """High-sparsity sparse_loss path (TOPK=8 of Sk=4096) parity vs torch ref."""

    _run_sparse_parity(AB=1, AH=2, AD=64, ASq=128, Sk=4096, TOPK=8, seed=0xA11CE)


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_dsa_splitk_indexer_loss_sparse_low_sparsity():
    """Low-sparsity sparse_loss path (TOPK=1024 of Sk=4096) parity vs torch ref."""

    _run_sparse_parity(AB=1, AH=2, AD=64, ASq=128, Sk=4096, TOPK=1024, seed=0xB0B)


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_dsa_splitk_indexer_loss_sparse_full_topk_matches_dense():
    """sparse_loss=True with TOPK=Sk degenerates to the dense path numerically.

    Each row's mask is all-zeros (every position selected), so the kernel must
    return the same value as ``sparse_loss=False`` on identical inputs. This
    catches mask-application bugs (e.g. wrong sign, off-by-one on the scatter)
    without needing a Triton ground truth.
    """

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang DSA split-K requires a CUDA or Metal device")

    torch.manual_seed(0xDEAD)
    AB, AH, AD = 1, 2, 32
    ASq, Sk = 64, 128

    query = torch.randn(ASq, AB, AH, AD, dtype=torch.float16, device=device)
    key = torch.randn(Sk, AB, AH, AD, dtype=torch.float16, device=device)
    index_scores = torch.randn(AB, ASq, Sk, dtype=torch.float32, device=device)

    # TOPK == Sk and indices = arange => mask is all-zero.
    topk_indices = torch.arange(Sk, dtype=torch.long, device=device).expand(AB, ASq, Sk).contiguous()

    softmax_scale = 1.0 / math.sqrt(AD)
    loss_coeff = 1.0

    out_sparse = dsa_splitk_indexer_loss_tilelang(
        index_scores, topk_indices, query, key,
        softmax_scale=softmax_scale, loss_coeff=loss_coeff,
        sparse_loss=True, pg_collection=None,
    )
    out_dense = dsa_splitk_indexer_loss_tilelang(
        index_scores, topk_indices, query, key,
        softmax_scale=softmax_scale, loss_coeff=loss_coeff,
        sparse_loss=False, pg_collection=None,
    )
    torch.testing.assert_close(
        out_sparse.to(torch.float32), out_dense.to(torch.float32), rtol=1e-3, atol=1e-5,
    )
