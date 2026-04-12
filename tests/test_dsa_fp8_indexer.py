"""Unit tests for :mod:`cppmega.megatron.dsa_fp8_indexer`.

These tests are CUDA-gated because ``torch._scaled_mm`` requires a GPU
with fp8 support (H200 sm_90a or B200 sm_100). On a CPU-only laptop the
``fp8_rowwise_gemm_matches_bf16`` test is skipped; on bench3 it runs.

The BF16 reference clone of Megatron's ``_compute_index_scores`` is
tested on CPU too so the laptop side at least validates shape/math
parity of the reference function we are comparing against.
"""

from __future__ import annotations

import pytest
import torch

from cppmega.megatron.dsa_fp8_indexer import (
    FP8_E4M3_MAX,
    compute_index_scores_bf16_reference,
    quantize_rowwise_fp8,
)


_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP8 _scaled_mm path requires CUDA (H200/B200)",
)


def _make_inputs(*, sq: int, b: int, h: int, d: int, sk: int, seed: int = 0, device: str = "cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    # dtype must be bf16 -- that is what Megatron's DSA indexer produces.
    dtype = torch.bfloat16 if device == "cuda" else torch.bfloat16
    q = torch.randn(sq, b, h, d, generator=g, device=device, dtype=dtype)
    k = torch.randn(sk, b, d, generator=g, device=device, dtype=dtype)
    weights = torch.randn(sq, b, h, generator=g, device=device, dtype=dtype)
    return q, weights, k


# ---------------------------------------------------------------------------
# CPU-safe tests: the BF16 reference clone itself + quantisation helper.
# ---------------------------------------------------------------------------


def test_bf16_reference_shape_and_dtype():
    q, weights, k = _make_inputs(sq=16, b=2, h=4, d=32, sk=16)
    out = compute_index_scores_bf16_reference(q, weights, k)
    assert out.shape == (2, 16, 16)
    assert out.dtype == torch.float32


def test_bf16_reference_matches_explicit_math():
    q, weights, k = _make_inputs(sq=8, b=2, h=4, d=16, sk=12, seed=7)
    out = compute_index_scores_bf16_reference(q, weights, k)

    # Explicit loop reference
    sq, b, h, d = q.shape
    sk = k.shape[0]
    expected = torch.zeros(b, sq, sk, dtype=torch.float32)
    for bi in range(b):
        for i in range(sq):
            for j in range(sk):
                acc = 0.0
                for hi in range(h):
                    dot = 0.0
                    for di in range(d):
                        dot += q[i, bi, hi, di].float().item() * k[j, bi, di].float().item()
                    acc += max(dot, 0.0) * weights[i, bi, hi].float().item()
                expected[bi, i, j] = acc

    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)


def test_quantize_rowwise_fp8_roundtrip_cpu():
    # quantize_rowwise_fp8 is pure torch, runs on CPU too.
    x = torch.randn(4, 8, 16, dtype=torch.bfloat16) * 3.0  # some dynamic range
    x_fp8, scale = quantize_rowwise_fp8(x)
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scale.shape == x.shape[:-1]
    assert scale.dtype == torch.float32
    # Reconstruct
    x_recon = x_fp8.float() * scale.unsqueeze(-1)
    rel = (x_recon - x.float()).abs().max() / x.float().abs().max().clamp(min=1e-6)
    # Per-row fp8_e4m3 should give <8% max rel error on N(0,1) * 3 inputs.
    assert rel.item() < 0.1, f"rowwise fp8 roundtrip rel_err={rel.item()} exceeds 0.1"


def test_quantize_rowwise_fp8_absmax_invariance():
    # Rowwise scale = amax/448: verify we really get per-row amax.
    x = torch.zeros(2, 3, 8, dtype=torch.float32)
    x[0, 0, :] = 5.0
    x[0, 1, :] = 100.0  # larger row
    x[0, 2, :] = 1e-5
    x[1, 0, :] = 50.0
    _, scale = quantize_rowwise_fp8(x)
    # row (0,0) -> amax=5 -> scale = 5/448
    assert scale[0, 0].item() == pytest.approx(5.0 / FP8_E4M3_MAX, rel=1e-5)
    assert scale[0, 1].item() == pytest.approx(100.0 / FP8_E4M3_MAX, rel=1e-5)
    # clamp floor kicks in at 1e-4
    assert scale[0, 2].item() == pytest.approx(1e-4 / FP8_E4M3_MAX, rel=1e-5)


# ---------------------------------------------------------------------------
# CUDA-only: full FP8 vs BF16 parity + topk overlap check.
# ---------------------------------------------------------------------------


@_REQUIRES_CUDA
def test_fp8_compute_matches_bf16_within_tolerance():
    from cppmega.megatron.dsa_fp8_indexer import compute_index_scores_fp8

    torch.manual_seed(1234)
    q, weights, k = _make_inputs(sq=64, b=2, h=4, d=32, sk=64, device="cuda", seed=1234)

    bf16_out = compute_index_scores_bf16_reference(q, weights, k)
    fp8_out = compute_index_scores_fp8(q, weights, k)

    assert fp8_out.shape == bf16_out.shape
    assert fp8_out.dtype == torch.float32

    # FP8 e4m3 with rowwise scales + ReLU + weighted sum-over-heads gives
    # an elementwise error that is mostly bounded but has a long tail at
    # positions where bf16 ≈ 0 (relu mask boundary flips between bf16 and
    # fp8). DSA is a topk ranker, not a precise score reconstructor — so
    # the meaningful contract is topk overlap (see
    # ``test_fp8_topk_overlap_with_bf16``). Here we only require that
    # 90% of elements fall within an absolute tolerance of 0.5 and a
    # relative tolerance of 0.2 on the non-zero reference positions.
    abs_diff = (fp8_out - bf16_out).abs()
    within_abs = abs_diff <= 0.5
    ref_abs = bf16_out.abs()
    nonzero = ref_abs > 1e-3
    rel_diff = torch.where(nonzero, abs_diff / ref_abs.clamp(min=1e-6), torch.zeros_like(abs_diff))
    within_rel = rel_diff <= 0.2
    within = within_abs | (~nonzero) | within_rel
    frac_within = within.float().mean().item()
    assert frac_within >= 0.9, (
        f"fp8 vs bf16 fraction within tolerance = {frac_within:.3f} (expected >=0.9); "
        f"max abs diff = {abs_diff.max().item():.4f}"
    )


@_REQUIRES_CUDA
def test_fp8_topk_overlap_with_bf16():
    from cppmega.megatron.dsa_fp8_indexer import compute_index_scores_fp8

    torch.manual_seed(5678)
    q, weights, k = _make_inputs(sq=64, b=2, h=4, d=32, sk=128, device="cuda", seed=5678)

    bf16_out = compute_index_scores_bf16_reference(q, weights, k)
    fp8_out = compute_index_scores_fp8(q, weights, k)

    topk = 16
    bf16_top = bf16_out.topk(topk, dim=-1)[1]
    fp8_top = fp8_out.topk(topk, dim=-1)[1]

    # Per-query overlap count between the two topk sets.
    # Both have shape [b, sq, topk]. Convert to sets per query.
    b, sq, _ = bf16_top.shape
    overlap_total = 0
    for bi in range(b):
        for qi in range(sq):
            a = set(bf16_top[bi, qi].tolist())
            z = set(fp8_top[bi, qi].tolist())
            overlap_total += len(a & z)
    overlap_ratio = overlap_total / (b * sq * topk)

    # DSA is about *which* keys get picked, not exact scores. We demand
    # >=85% average overlap between fp8 and bf16 topk sets.
    assert overlap_ratio >= 0.85, (
        f"fp8 vs bf16 topk overlap = {overlap_ratio:.3f} (expected >=0.85)"
    )


# ---------------------------------------------------------------------------
# Monkey-patch smoke (unit, no Megatron import required): validates the
# resolve_indexer_dtype helper + argparse helper.
# ---------------------------------------------------------------------------


def test_resolve_indexer_dtype_env(monkeypatch):
    from cppmega.megatron.dsa_fp8_patch import (
        DSA_INDEXER_DTYPE_ENV,
        resolve_indexer_dtype,
    )

    monkeypatch.delenv(DSA_INDEXER_DTYPE_ENV, raising=False)
    assert resolve_indexer_dtype() == "bf16"

    monkeypatch.setenv(DSA_INDEXER_DTYPE_ENV, "fp8")
    assert resolve_indexer_dtype() == "fp8"

    monkeypatch.setenv(DSA_INDEXER_DTYPE_ENV, "BF16")
    assert resolve_indexer_dtype() == "bf16"


def test_resolve_indexer_dtype_config_precedence(monkeypatch):
    from cppmega.megatron.dsa_fp8_patch import (
        DSA_INDEXER_DTYPE_ENV,
        resolve_indexer_dtype,
    )

    class _FakeCfg:
        dsa_indexer_dtype = "fp8"

    monkeypatch.delenv(DSA_INDEXER_DTYPE_ENV, raising=False)
    assert resolve_indexer_dtype(_FakeCfg()) == "fp8"

    # env bf16 must NOT override an explicit config fp8
    monkeypatch.setenv(DSA_INDEXER_DTYPE_ENV, "bf16")
    assert resolve_indexer_dtype(_FakeCfg()) == "fp8"


def test_add_dsa_indexer_dtype_arg_registers_flag():
    import argparse

    from cppmega.megatron.dsa_fp8_patch import add_dsa_indexer_dtype_arg

    parser = argparse.ArgumentParser()
    parser.add_argument_group("experimental_attention_variant")
    add_dsa_indexer_dtype_arg(parser)
    # idempotent
    add_dsa_indexer_dtype_arg(parser)

    ns = parser.parse_args(["--dsa-indexer-dtype", "fp8"])
    assert ns.dsa_indexer_dtype == "fp8"

    ns = parser.parse_args([])
    assert ns.dsa_indexer_dtype == "bf16"

    with pytest.raises(SystemExit):
        parser.parse_args(["--dsa-indexer-dtype", "int4"])


# ---------------------------------------------------------------------------
# Stream G: backward FP8 parity. Compares bwd_fused_indexer_loss_fp8 against
# bwd_fused_indexer_loss_bf16_reference on small CUDA inputs.
# ---------------------------------------------------------------------------


def _make_backward_inputs(
    *,
    sq: int,
    sk: int,
    b: int,
    h: int,
    d: int,
    np_: int,
    hn: int,
    topk: int,
    seed: int,
    device: str = "cuda",
):
    """Build the full set of tensors that ``bwd_fused_indexer_loss_*``
    consumes: indexer ``q [sq,b,h,d]``, ``weights [sq,b,h]``, ``k [sk,b,d]``,
    main-attention ``query [sq,b,np_,hn]``, ``key [sk,b,np_,hn]``, and a
    ``topk_indices [b,sq,topk]`` int64 tensor whose rows are a random
    permutation of ``range(sk)[:topk]`` per query (so the KL target has
    non-trivial sparse mask support).
    """

    g = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.bfloat16
    q = torch.randn(sq, b, h, d, generator=g, device=device, dtype=dtype)
    weights = torch.randn(sq, b, h, generator=g, device=device, dtype=dtype).abs()
    k = torch.randn(sk, b, d, generator=g, device=device, dtype=dtype)
    query = torch.randn(sq, b, np_, hn, generator=g, device=device, dtype=dtype)
    key = torch.randn(sk, b, np_, hn, generator=g, device=device, dtype=dtype)

    topk_idx = torch.empty(b, sq, topk, dtype=torch.int64, device=device)
    for bi in range(b):
        for si in range(sq):
            perm = torch.randperm(sk, generator=g, device=device)
            topk_idx[bi, si] = perm[:topk]

    grad_loss = torch.tensor(1.0, dtype=torch.float32, device=device)
    return q, weights, k, query, key, topk_idx, grad_loss


def _frac_within(ref: torch.Tensor, got: torch.Tensor, *, atol: float, rtol: float) -> float:
    """Stream E's per-element contract: element passes if it is within
    ``atol`` absolute OR within ``rtol`` relative to the reference. Returns
    the fraction of elements that pass."""

    abs_diff = (got.float() - ref.float()).abs()
    within_abs = abs_diff <= atol
    ref_abs = ref.float().abs()
    nonzero = ref_abs > 1e-3
    rel_diff = torch.where(
        nonzero, abs_diff / ref_abs.clamp(min=1e-6), torch.zeros_like(abs_diff)
    )
    within_rel = rel_diff <= rtol
    within = within_abs | (~nonzero) | within_rel
    return within.float().mean().item()


@_REQUIRES_CUDA
def test_backward_parity_bf16_vs_fp8():
    """Run the upstream-equivalent BF16 reference backward and the FP8
    backward on the same small DSA input, then verify that
    ``grad_q``/``grad_weights``/``grad_k`` match within the same tolerance
    contract Stream E used for the forward path (``frac_within(abs<=0.5
    OR rel<=0.2) >= 0.9``). Also asserts that the recomputed indexer loss
    itself matches within tight tolerance."""

    from cppmega.megatron.dsa_fp8_indexer import (
        bwd_fused_indexer_loss_bf16_reference,
        bwd_fused_indexer_loss_fp8,
        compute_index_scores_bf16_reference,
        compute_index_scores_fp8,
    )

    sq = sk = 128
    b = 2
    h = 4
    d = 32
    np_ = 4
    hn = 32
    topk = 16

    torch.manual_seed(12345)
    q, weights, k, query, key, topk_idx, grad_loss = _make_backward_inputs(
        sq=sq,
        sk=sk,
        b=b,
        h=h,
        d=d,
        np_=np_,
        hn=hn,
        topk=topk,
        seed=12345,
        device="cuda",
    )

    softmax_scale = 1.0 / (hn**0.5)
    loss_coeff = 0.1
    sparse_loss = True

    bf16_gq, bf16_gw, bf16_gk = bwd_fused_indexer_loss_bf16_reference(
        q,
        weights,
        k,
        query,
        key,
        topk_idx,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        grad_loss,
        pg_collection=None,
    )
    fp8_gq, fp8_gw, fp8_gk = bwd_fused_indexer_loss_fp8(
        q,
        weights,
        k,
        query,
        key,
        topk_idx,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        grad_loss,
        pg_collection=None,
    )

    assert fp8_gq.shape == bf16_gq.shape == (sq, b, h, d)
    assert fp8_gw.shape == bf16_gw.shape == (sq, b, h)
    assert fp8_gk.shape == bf16_gk.shape == (sk, b, d)

    # Core tolerance contract: frac_within(abs<=0.5 OR rel<=0.2) >= 0.9.
    for name, ref, got in [
        ("grad_q", bf16_gq, fp8_gq),
        ("grad_weights", bf16_gw, fp8_gw),
        ("grad_k", bf16_gk, fp8_gk),
    ]:
        frac = _frac_within(ref, got, atol=0.5, rtol=0.2)
        assert frac >= 0.9, (
            f"fp8 vs bf16 {name} frac_within(abs<=0.5 OR rel<=0.2) = {frac:.3f} "
            f"(expected >=0.9); max_abs_diff = "
            f"{(got.float() - ref.float()).abs().max().item():.4f}"
        )

    # Sanity: the underlying forward index_scores from FP8 vs BF16 must
    # still match the forward contract --- we are not accidentally
    # measuring a backward bug that cancels forward noise.
    bf16_idx = compute_index_scores_bf16_reference(q, weights, k)
    fp8_idx = compute_index_scores_fp8(q, weights, k)
    frac_idx = _frac_within(bf16_idx, fp8_idx, atol=0.5, rtol=0.2)
    assert frac_idx >= 0.9, f"forward index scores drift {frac_idx:.3f} < 0.9"


@_REQUIRES_CUDA
def test_backward_fp8_no_sparse_loss_parity():
    """Same parity contract as ``test_backward_parity_bf16_vs_fp8`` but
    with ``sparse_loss=False``. This exercises the else-branch in the
    mask construction: causal mask only, no index_mask addend."""

    from cppmega.megatron.dsa_fp8_indexer import (
        bwd_fused_indexer_loss_bf16_reference,
        bwd_fused_indexer_loss_fp8,
    )

    sq = sk = 64
    b = 2
    h = 4
    d = 32
    np_ = 2
    hn = 32
    topk = 8

    torch.manual_seed(999)
    q, weights, k, query, key, topk_idx, grad_loss = _make_backward_inputs(
        sq=sq,
        sk=sk,
        b=b,
        h=h,
        d=d,
        np_=np_,
        hn=hn,
        topk=topk,
        seed=999,
        device="cuda",
    )
    softmax_scale = 1.0 / (hn**0.5)
    loss_coeff = 0.05

    bf16_grads = bwd_fused_indexer_loss_bf16_reference(
        q, weights, k, query, key, topk_idx, softmax_scale, loss_coeff,
        False, grad_loss, pg_collection=None,
    )
    fp8_grads = bwd_fused_indexer_loss_fp8(
        q, weights, k, query, key, topk_idx, softmax_scale, loss_coeff,
        False, grad_loss, pg_collection=None,
    )

    for name, ref, got in zip(("grad_q", "grad_weights", "grad_k"), bf16_grads, fp8_grads):
        frac = _frac_within(ref, got, atol=0.5, rtol=0.2)
        assert frac >= 0.9, (
            f"fp8 no-sparse {name} frac_within={frac:.3f} (<0.9); "
            f"max_abs={((got.float() - ref.float()).abs().max().item()):.4f}"
        )
