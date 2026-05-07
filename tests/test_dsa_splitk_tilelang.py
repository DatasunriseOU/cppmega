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


# ---------------------------------------------------------------------------
# Wave-3 self-audit: explicit sparse-mask sign-convention test.
#
# Catches the failure mode where the wrapper's scatter direction inverts
# (``scatter(0, indices, -inf)`` instead of ``scatter(-inf, indices, 0)``) or
# the kernel's ``s = s + IndexMask[..]`` becomes ``s - IndexMask[..]``. A
# hand-crafted topk + dense ref makes the bug obvious; the random tests above
# can mask it because random scores already produce a wide value range.
# Also covers the new int32-topk_indices acceptance path landed in this wave.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_dsa_splitk_indexer_loss_sparse_mask_sign_convention():
    """Hand-crafted sparse mask: positions in topk → 0, others → -inf."""

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang DSA split-K requires a CUDA or Metal device")

    torch.manual_seed(0x517A)
    AB, AH, AD = 1, 1, 32
    ASq, Sk = 8, 16

    query = torch.randn(ASq, AB, AH, AD, dtype=torch.float16, device=device)
    key = torch.randn(Sk, AB, AH, AD, dtype=torch.float16, device=device)
    index_scores = torch.randn(AB, ASq, Sk, dtype=torch.float32, device=device)

    # Per-row top-2 picks, hand-chosen so each row has a distinct sparsity
    # pattern. int32 dtype intentional -- exercises the int32→int64 promotion
    # branch added in the wave-3 wrapper validation.
    topk_indices = torch.tensor(
        [[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]],
        dtype=torch.int32, device=device,
    )

    softmax_scale = 1.0 / math.sqrt(AD)
    loss_coeff = 1.0

    out_kernel = dsa_splitk_indexer_loss_tilelang(
        index_scores, topk_indices, query, key,
        softmax_scale=softmax_scale, loss_coeff=loss_coeff,
        sparse_loss=True, pg_collection=None,
    )
    ref = _torch_indexer_loss_reference(
        index_scores, topk_indices.to(torch.long), query, key,
        softmax_scale=softmax_scale, loss_coeff=loss_coeff, sparse_loss=True,
    )

    torch.testing.assert_close(
        out_kernel.to(torch.float32), ref.to(torch.float32),
        rtol=1e-2, atol=1e-4,
    )


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_dsa_splitk_indexer_loss_topk_validation():
    """Wave-3 self-audit: wrapper raises clearly on bad topk_indices inputs."""

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang DSA split-K requires a CUDA or Metal device")

    AB, AH, AD = 1, 1, 32
    ASq, Sk, TOPK = 8, 16, 4

    query = torch.randn(ASq, AB, AH, AD, dtype=torch.float16, device=device)
    key = torch.randn(Sk, AB, AH, AD, dtype=torch.float16, device=device)
    index_scores = torch.randn(AB, ASq, Sk, dtype=torch.float32, device=device)

    # Wrong shape.
    bad_shape = torch.zeros((AB, ASq + 1, TOPK), dtype=torch.long, device=device)
    with pytest.raises(ValueError, match="topk_indices must have shape"):
        dsa_splitk_indexer_loss_tilelang(
            index_scores, bad_shape, query, key,
            softmax_scale=1.0, loss_coeff=1.0, sparse_loss=True,
        )

    # Wrong dtype.
    bad_dtype = torch.zeros((AB, ASq, TOPK), dtype=torch.float32, device=device)
    with pytest.raises(TypeError, match="topk_indices.dtype"):
        dsa_splitk_indexer_loss_tilelang(
            index_scores, bad_dtype, query, key,
            softmax_scale=1.0, loss_coeff=1.0, sparse_loss=True,
        )


def test_dsa_debug_env_gate_skips_sync_in_production():
    """Wave-4 perf #1: production path must NOT sync the GPU.

    The validation block (.item()/.all()) is gated behind
    CPPMEGA_MLX_DSA_DEBUG. With the env var unset (default), even an
    out-of-range topk index should reach the kernel layer without the
    pre-launch ValueError firing. With the env var set, the same input
    must raise ValueError as before.
    """

    import os
    from cppmega_mlx.nn._tilelang.dsa_splitk_indexer_loss import (
        _dsa_debug_enabled,
    )

    # The debug helper is the contract; verify both branches respond to
    # env. Don't actually launch a GPU kernel here -- this is a fast
    # CPU-side regression for the gating logic.
    saved = os.environ.pop("CPPMEGA_MLX_DSA_DEBUG", None)
    try:
        assert _dsa_debug_enabled() is False
        for falsey in ("", "0", "false", "no", "off", "FALSE"):
            os.environ["CPPMEGA_MLX_DSA_DEBUG"] = falsey
            assert _dsa_debug_enabled() is False, falsey
        for truthy in ("1", "true", "yes", "on", "TRUE", "Yes"):
            os.environ["CPPMEGA_MLX_DSA_DEBUG"] = truthy
            assert _dsa_debug_enabled() is True, truthy
    finally:
        os.environ.pop("CPPMEGA_MLX_DSA_DEBUG", None)
        if saved is not None:
            os.environ["CPPMEGA_MLX_DSA_DEBUG"] = saved


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_dsa_splitk_indexer_loss_split_stage1_kernels_match_unified():
    """Wave-4 perf #3: split stage-1 kernels (AH > 1 path) match the unified
    kernel's M1/D1 outputs.

    The wrapper auto-selects the split path when AH > 1 and the unified
    path when AH == 1. To verify equivalence we run the same input shape
    twice (with AH > 1 it goes through the split, then we slice and
    reduce to the same statistics in the AH==1 unified launch). This
    catches drift between ``make_dsa_splitk_stage1_idx_kernel`` and the
    legacy unified ``if h == 0`` block.
    """

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang DSA split-K requires a CUDA or Metal device")

    AB, AD = 1, 32
    ASq, Sk, TOPK = 16, 32, 4
    softmax_scale = 1.0 / math.sqrt(AD)

    torch.manual_seed(13)
    query_template = torch.randn(ASq, AB, 1, AD, dtype=torch.float16, device=device)
    key_template = torch.randn(Sk, AB, 1, AD, dtype=torch.float16, device=device)
    index_scores = torch.randn(AB, ASq, Sk, dtype=torch.float32, device=device)
    topk = torch.randint(0, Sk, (AB, ASq, TOPK), dtype=torch.long, device=device)

    # AH==1: exercises the unified kernel.
    loss_ah1 = dsa_splitk_indexer_loss_tilelang(
        index_scores, topk, query_template, key_template,
        softmax_scale=softmax_scale, loss_coeff=1.0, sparse_loss=True,
    )

    # AH==4: exercises the split path; M1/D1 must match (they're
    # AH-independent by construction). Build query/key with 4 heads but
    # use the same head-0 slice so attn statistics also match.
    AH = 4
    query_ah4 = query_template.expand(ASq, AB, AH, AD).contiguous()
    key_ah4 = key_template.expand(Sk, AB, AH, AD).contiguous()
    loss_ah4 = dsa_splitk_indexer_loss_tilelang(
        index_scores, topk, query_ah4, key_ah4,
        softmax_scale=softmax_scale, loss_coeff=1.0, sparse_loss=True,
    )

    # Both paths produce a per-(b, sq) loss; they should be numerically
    # close because the index-softmax statistics are head-independent
    # and the attention path averages over the (here-identical) heads.
    torch.testing.assert_close(
        loss_ah1.float(), loss_ah4.float(),
        rtol=1e-2, atol=1e-4,
    )


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_stage2_q_hoist_numerical_parity_wave5():
    """Wave-5: full Q hoist (use_q_cache_v5=True) must match wave-4 partial-hoist
    kernel numerically on a small case where both paths are buildable.

    Validates that the budget-gated Q-cache (AH, BLOCK_SQ, AD) shared
    fragment produces fp32-equivalent (within tolerance) loss vs. the
    wave-4 per-(sk_tile, h) HBM reload pattern. Skips when the cache
    doesn't fit the per-target budget.
    """

    from cppmega_mlx.nn._tilelang.dsa_splitk_indexer_loss import (
        _block_constants_for_target,
        _can_use_q_cache_v5,
        _resolve_target,
        _scale_to_bits,
        _stage1_kernel_for,
        _stage2_kernel_for,
    )

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("Wave-5 Q-hoist parity needs CUDA or Metal")

    AB, AH, AD = 1, 4, 64
    ASq, Sk = 128, 512
    in_dtype = "float16"
    target = _resolve_target(device)
    stage1_kw, stage2_kw = _block_constants_for_target(target, AH=AH)
    if not _can_use_q_cache_v5(
        BLOCK_SQ=stage2_kw["BLOCK_SQ"], AH=AH, AD=AD, in_dtype=in_dtype, target=target
    ):
        pytest.skip(
            f"Q-cache (AH={AH}, BLOCK_SQ={stage2_kw['BLOCK_SQ']}, AD={AD}) "
            f"does not fit target {target!r} budget"
        )

    softmax_scale = 1.0 / math.sqrt(AD)
    scale_bits = _scale_to_bits(softmax_scale)

    torch.manual_seed(7)
    query = torch.randn(ASq, AB, AH, AD, dtype=torch.float16, device=device)
    key = torch.randn(Sk, AB, AH, AD, dtype=torch.float16, device=device)
    index_scores = torch.randn(AB, ASq, Sk, dtype=torch.float32, device=device)
    index_mask = torch.zeros(AB, ASq, Sk, dtype=torch.float32, device=device)
    softmax_m = torch.zeros(AB, AH, ASq, dtype=torch.float32, device=device)
    softmax_d = torch.ones(AB, AH, ASq, dtype=torch.float32, device=device)
    softmax_m1 = torch.zeros(AB, ASq, dtype=torch.float32, device=device)
    softmax_d1 = torch.ones(AB, ASq, dtype=torch.float32, device=device)

    stage1 = _stage1_kernel_for(
        AB, AH, AD, Sk, ASq, False, scale_bits, in_dtype, target,
        stage1_kw["BLOCK_SQ"], stage1_kw["BLOCK_SK"], stage1_kw["BLOCK_D"],
        stage1_kw["threads"], stage1_kw["num_stages"],
    )
    stage1(query, key, index_scores, index_mask, softmax_m, softmax_d, softmax_m1, softmax_d1)

    out_v4 = torch.empty(AB, ASq, dtype=torch.float32, device=device)
    out_v5 = torch.empty(AB, ASq, dtype=torch.float32, device=device)

    stage2_v4 = _stage2_kernel_for(
        AB, AH, AD, Sk, ASq, False, scale_bits, in_dtype, target,
        stage2_kw["BLOCK_SQ"], stage2_kw["BLOCK_SK"], stage2_kw["BLOCK_D"],
        stage2_kw["threads"], stage2_kw["num_stages"],
        use_q_cache_v5=False,
    )
    stage2_v4(
        query, key, index_scores, index_mask,
        softmax_m, softmax_d, softmax_m1, softmax_d1, out_v4,
    )

    stage2_v5 = _stage2_kernel_for(
        AB, AH, AD, Sk, ASq, False, scale_bits, in_dtype, target,
        stage2_kw["BLOCK_SQ"], stage2_kw["BLOCK_SK"], stage2_kw["BLOCK_D"],
        stage2_kw["threads"], stage2_kw["num_stages"],
        use_q_cache_v5=True,
    )
    stage2_v5(
        query, key, index_scores, index_mask,
        softmax_m, softmax_d, softmax_m1, softmax_d1, out_v5,
    )

    # Same algorithm, different memory hierarchy -- expect fp32 round-off only.
    torch.testing.assert_close(out_v4, out_v5, rtol=1e-4, atol=1e-5)


def test_can_use_q_cache_v5_budget_logic():
    """_can_use_q_cache_v5 returns sane bool over budget edges (no device required)."""

    from cppmega_mlx.nn._tilelang.dsa_splitk_indexer_loss import _can_use_q_cache_v5

    # Tiny shape -- fits everywhere.
    assert _can_use_q_cache_v5(BLOCK_SQ=16, AH=2, AD=32, in_dtype="float16", target="metal")
    assert _can_use_q_cache_v5(BLOCK_SQ=16, AH=2, AD=32, in_dtype="float16", target="cuda")
    assert _can_use_q_cache_v5(BLOCK_SQ=16, AH=2, AD=32, in_dtype="float16", target="hip")

    # Pathological -- 128*128*64*2 = 2 MB doesn't fit anywhere.
    assert not _can_use_q_cache_v5(BLOCK_SQ=128, AH=128, AD=64, in_dtype="float16", target="metal")
    assert not _can_use_q_cache_v5(BLOCK_SQ=128, AH=128, AD=64, in_dtype="float16", target="cuda")

    # 32*16*64*2 = 64 KB: fits CUDA budget (64 KB), not Metal (16 KB).
    assert not _can_use_q_cache_v5(BLOCK_SQ=32, AH=16, AD=64, in_dtype="float16", target="metal")
    assert _can_use_q_cache_v5(BLOCK_SQ=32, AH=16, AD=64, in_dtype="float16", target="cuda")


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_wave9_tiled_block_sq_non_multiple_asq():
    """Wave-9 #2 regression for grok rev_38ff59759f HIGH finding.

    The wave-8 366b5be tiled Q-cache picks ``BLOCK_SQ`` from
    ``{64, 32, 16, 8}`` to fit the per-target shared-memory budget. AH=8 +
    AD=64 lands on ``BLOCK_SQ=16`` on Metal. ``ASq=100`` is not a multiple
    of 16 (last sq_block covers indices [96, 112), only 4 rows valid) ->
    exercises the ``sq_idx < ASq`` clip on every loop nest in stage 2.
    Off-by-one in any of {Q_full hoist, m1/d1 load, M_pre/D_pre prefetch,
    Q_all_heads cache, valid predicate, output store} would diverge from
    the torch reference by a noticeable margin.
    """

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang DSA split-K requires a CUDA or Metal device")

    torch.manual_seed(0xA59E)
    AB, AH, AD = 1, 8, 64
    ASq, Sk = 100, 2048  # ASq deliberately non-multiple of any BLOCK_SQ choice
    softmax_scale = 1.0 / math.sqrt(AD)
    loss_coeff = 1.0

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
    # AH=8 + AD=64 routes through BLOCK_SQ=16 on Metal (16 KB budget exact).
    # ASq=100 -> 7 sq_blocks; last block has 4 valid rows; off-by-one in any
    # clip predicate would spike well above this tolerance.
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), rtol=1e-2, atol=1e-4)


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_wave9_topk_oob_rejected():
    """Wave-9 #4 regression for grok rev_38ff59759f HIGH finding.

    Production callers used to skip the wave-3 ``CPPMEGA_MLX_DSA_DEBUG``-
    gated bounds check, leaving torch ``scatter_(-1, topk_idx, 0.0)`` free
    to silently corrupt adjacent memory on CUDA release builds when an
    upstream bug fed a ``topk_index >= Sk``. Wave-9 #4 (cppmega.mlx
    ``fcd7068``) makes the bounds check always-on with a single fused
    ``((idx < 0) | (idx >= Sk)).any().item()`` reduction. This test
    feeds an OOB index and asserts ``ValueError`` is raised before
    scatter_ touches memory.
    """

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang DSA split-K requires a CUDA or Metal device")

    torch.manual_seed(0xB0BD)
    AB, AH, AD = 1, 4, 32
    ASq, Sk = 16, 64
    softmax_scale = 1.0 / math.sqrt(AD)

    query = torch.randn(ASq, AB, AH, AD, dtype=torch.float16, device=device)
    key = torch.randn(Sk, AB, AH, AD, dtype=torch.float16, device=device)
    index_scores = torch.randn(AB, ASq, Sk, dtype=torch.float32, device=device)
    # Inject an OOB index: Sk=64 so anything >= 64 must be rejected.
    topk_indices = torch.zeros(AB, ASq, 4, dtype=torch.long, device=device)
    topk_indices[0, 0, 0] = Sk + 7  # 71, well past the upper bound

    with pytest.raises(ValueError, match="topk_indices out of range"):
        dsa_splitk_indexer_loss_tilelang(
            index_scores, topk_indices, query, key,
            softmax_scale=softmax_scale, loss_coeff=1.0,
            sparse_loss=True, pg_collection=None,
        )

    # Negative indices should also be rejected.
    topk_indices[0, 0, 0] = -1
    with pytest.raises(ValueError, match="topk_indices out of range"):
        dsa_splitk_indexer_loss_tilelang(
            index_scores, topk_indices, query, key,
            softmax_scale=softmax_scale, loss_coeff=1.0,
            sparse_loss=True, pg_collection=None,
        )


def test_wave9_tiled_block_sq_choice_for_production_shapes():
    """Wave-9 #2 budget probe: AH=8/16 + AD=64 shapes must land on the wave-5 path."""

    from cppmega_mlx.nn._tilelang.dsa_splitk_indexer_loss import _can_use_q_cache_v5_tiled

    # AH=8 + AD=64 + Metal (16 KB): 8*16*64*2 = 16384 B -> exactly fits at BLOCK_SQ=16.
    assert _can_use_q_cache_v5_tiled(AH=8, AD=64, in_dtype="float16", target="metal") == 16
    # AH=16 + AD=64 + Metal: 16*8*64*2 = 16384 B -> fits at BLOCK_SQ=8.
    assert _can_use_q_cache_v5_tiled(AH=16, AD=64, in_dtype="float16", target="metal") == 8
    # AH=4 + AD=64 + Metal: 4*32*64*2 = 16384 B -> fits at BLOCK_SQ=32.
    assert _can_use_q_cache_v5_tiled(AH=4, AD=64, in_dtype="float16", target="metal") == 32
    # AH=128 + AD=64 + Metal: 128*8*64*2 = 131072 B -> over budget at all tiles.
    assert _can_use_q_cache_v5_tiled(AH=128, AD=64, in_dtype="float16", target="metal") is None
    # CUDA budget 64 KB -> AH=16 lands on BLOCK_SQ=32 (16*32*64*2 = 65536 B exact).
    assert _can_use_q_cache_v5_tiled(AH=16, AD=64, in_dtype="float16", target="cuda") == 32


def test_wave9_sparse_loss_scratch_cache():
    """Wave-9 #5 perf: sparse_loss path reuses the scatter scratch buffer.

    Pre-fix the wrapper allocated a fresh ``torch.full((AB, ASq, Sk), -inf)``
    every forward, costing O(AB*ASq*Sk*4) bytes per step. This test asserts:

    1. Two forwards with identical (AB, ASq, Sk, device, dtype) reuse the
       SAME backing tensor (``id()`` parity, since ``_get_scatter_scratch``
       returns the cached tensor itself before the scatter writes into it).
    2. The cache size stays at 1 entry for repeated identical-shape calls.
    3. The cap (``_SCATTER_SCRATCH_LRU_MAX``) prevents unbounded growth.
    """

    pytest = __import__("pytest")
    torch_mod = __import__("torch")

    try:
        from cppmega_mlx.nn._tilelang.dsa_splitk_indexer_loss import (
            _SCATTER_SCRATCH_CACHE,
            _SCATTER_SCRATCH_LRU_MAX,
            _get_scatter_scratch,
        )
    except Exception as exc:
        pytest.skip(f"cppmega.mlx tilelang dsa not importable: {exc}")

    _SCATTER_SCRATCH_CACHE.clear()

    device = torch_mod.device("cpu")  # CPU avoids no-Metal/no-CUDA skip; logic is device-agnostic.
    shape = (1, 4, 8)

    a = _get_scatter_scratch(shape, device, torch_mod.float32)
    b = _get_scatter_scratch(shape, device, torch_mod.float32)

    # Same backing buffer on cache hit.
    assert id(a) == id(b), "sparse_loss scatter scratch was re-allocated"
    # Single-shape callers keep the cache at 1 entry.
    assert len(_SCATTER_SCRATCH_CACHE) == 1
    # Refilled to -inf for the next scatter_ to produce an identical mask.
    assert torch_mod.isinf(b).all() and (b < 0).all(), "scratch not refilled with -inf"

    # 5 forwards same shape -> still 1 entry, same id.
    ids_seen = {id(a)}
    for _ in range(5):
        c = _get_scatter_scratch(shape, device, torch_mod.float32)
        ids_seen.add(id(c))
    assert ids_seen == {id(a)}
    assert len(_SCATTER_SCRATCH_CACHE) == 1

    # FIFO eviction stays within cap when shapes vary.
    for n in range(_SCATTER_SCRATCH_LRU_MAX + 4):
        _get_scatter_scratch((1, 4, 8 + n), device, torch_mod.float32)
    assert len(_SCATTER_SCRATCH_CACHE) <= _SCATTER_SCRATCH_LRU_MAX, (
        f"scratch cache grew past cap: {len(_SCATTER_SCRATCH_CACHE)} > {_SCATTER_SCRATCH_LRU_MAX}"
    )


def test_wave10_zero_size_handled():
    """Wave-10 #1 (grok+meta HIGH): zero-size shapes must not div-by-zero.

    Without the early-out at the top of ``dsa_splitk_indexer_loss_tilelang``,
    ``ASq=0`` (or ``AB=0`` / ``AH=0`` / ``AD=0`` / ``Sk=0``) reaches
    ``T.ceildiv(N, BLOCK_*)`` inside stage-1 / stage-2 kernel build →
    div-by-zero → CUDA illegal memory access / Metal host crash. Attacker
    scenario: malicious ONNX with attention mask length 0.

    The wrapper must return a 0-d fp32 scalar (matching the documented
    return contract) without invoking any kernel.
    """
    device = torch.device("cpu")
    in_dtype = torch.float32

    # ASq=0 (most common attacker shape: empty sequence).
    q = torch.zeros((0, 1, 4, 16), device=device, dtype=in_dtype)
    k = torch.zeros((4, 1, 4, 16), device=device, dtype=in_dtype)
    idx_scores = torch.zeros((1, 0, 4), device=device, dtype=torch.float32)
    topk = torch.zeros((1, 0, 2), device=device, dtype=torch.int64)
    out = dsa_splitk_indexer_loss_tilelang(
        idx_scores, topk, q, k, 1.0, 1.0, sparse_loss=True
    )
    assert out.shape == (), f"expected 0-d scalar, got shape {tuple(out.shape)}"
    assert out.dtype == torch.float32
    assert float(out) == 0.0

    # Sk=0.
    q = torch.zeros((2, 1, 4, 16), device=device, dtype=in_dtype)
    k = torch.zeros((0, 1, 4, 16), device=device, dtype=in_dtype)
    idx_scores = torch.zeros((1, 2, 0), device=device, dtype=torch.float32)
    topk = torch.zeros((1, 2, 2), device=device, dtype=torch.int64)
    out = dsa_splitk_indexer_loss_tilelang(
        idx_scores, topk, q, k, 1.0, 1.0, sparse_loss=True
    )
    assert out.shape == ()
    assert float(out) == 0.0

    # AB=0.
    q = torch.zeros((2, 0, 4, 16), device=device, dtype=in_dtype)
    k = torch.zeros((4, 0, 4, 16), device=device, dtype=in_dtype)
    idx_scores = torch.zeros((0, 2, 4), device=device, dtype=torch.float32)
    topk = torch.zeros((0, 2, 2), device=device, dtype=torch.int64)
    out = dsa_splitk_indexer_loss_tilelang(
        idx_scores, topk, q, k, 1.0, 1.0, sparse_loss=False
    )
    assert out.shape == ()
    assert float(out) == 0.0


def test_wave11_dsa_topk_zero_handled():
    """Wave-11 #3 (grok wave-10 review HIGH): TOPK == 0 with sparse_loss=True
    must be rejected at the boundary, not silently return NaN loss.

    With TOPK == 0 the ``index_scores.scatter_(-1, topk_indices, 0.0)`` is a
    no-op so the mask stays ``-inf`` everywhere. The downstream index-softmax
    then produces NaN, and the KL loss propagates NaN into training. The
    wrapper must raise ``ValueError`` instead.
    """
    device = torch.device("cpu")
    in_dtype = torch.float32
    q = torch.zeros((2, 1, 4, 16), device=device, dtype=in_dtype)
    k = torch.zeros((4, 1, 4, 16), device=device, dtype=in_dtype)
    idx_scores = torch.zeros((1, 2, 4), device=device, dtype=torch.float32)
    topk_zero = torch.zeros((1, 2, 0), device=device, dtype=torch.int64)
    with pytest.raises(ValueError, match="TOPK"):
        dsa_splitk_indexer_loss_tilelang(
            idx_scores, topk_zero, q, k, 1.0, 1.0, sparse_loss=True
        )

    # sparse_loss=False with TOPK == 0 should NOT raise (topk is unused there).
    out = dsa_splitk_indexer_loss_tilelang(
        idx_scores, topk_zero, q, k, 1.0, 1.0, sparse_loss=False
    )
    assert out.shape == ()
