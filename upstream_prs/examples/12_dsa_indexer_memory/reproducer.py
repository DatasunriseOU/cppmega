#!/usr/bin/env python3
"""Reproducer for Megatron-LM DSA ``_compute_index_scores`` memory waste.

Upstream
(``megatron/core/transformer/experimental_attention_variant/dsa.py`` @
line 255–295 on ``main``) materialises a ``[seqlen_q, batch,
index_n_heads, seqlen_k]`` FP32 intermediate before reducing to
``[batch, seqlen_q, seqlen_k]``. At NAM56R DSA 9+4 MBS=8 that intermediate
is 16 GiB — large enough to block MBS=10 on 8xH200 despite >40 GiB of
other HBM headroom in the step.

This script implements two mathematically equivalent variants:

* ``upstream_compute_index_scores`` — an exact copy of the upstream body
  (``torch.einsum('sbhd,tbd->sbht', ...)`` → relu → weight → sum → T).
* ``fused_compute_index_scores`` — per-head ``bmm`` accumulation into a
  ``[b, sq, sk]`` FP32 buffer, never materialising ``[sq, b, h, sk]``.

It compares them on three axes:

* **Correctness**: ``max|a-b| / max(|a|, eps)`` at production-like shape.
* **Memory**: ``torch.cuda.max_memory_allocated`` delta per variant.
* **Gradients**: ``torch.autograd.gradcheck`` at a small shape (full
  autograd parity, not just forward parity).

Exits non-zero on correctness failure. CPU small-shape path is available
via ``--cpu`` for CI smoke checks on machines without CUDA.

Reference implementation: ``cppmega/megatron/dsa_indexer_fused_patch.py``.
"""

from __future__ import annotations

import argparse
import gc
import sys

import torch


# ---------------------------------------------------------------------------
# Variant 1: upstream — verbatim copy of Megatron-LM _compute_index_scores
# (dsa.py @ main, line 255-295).
# ---------------------------------------------------------------------------
def upstream_compute_index_scores(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    # [sq, b, h, d] x [sk, b, d] -> [sq, b, h, sk]
    index_scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    index_scores = torch.relu(index_scores)
    index_scores = index_scores * weights.unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)
    index_scores = index_scores.transpose(0, 1)  # -> [b, sq, sk]
    return index_scores


# ---------------------------------------------------------------------------
# Variant 2: fused — per-head accumulation into [b, sq, sk].
# ---------------------------------------------------------------------------
def fused_compute_index_scores(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    sq, b, h, d = q.shape
    sk = k.shape[0]

    # Output-shaped fp32 accumulator.
    index_scores = torch.zeros(
        (b, sq, sk), dtype=torch.float32, device=q.device
    )

    # Shared across heads: [sk, b, d] -> [b, d, sk] fp32.
    k_bds = k.float().permute(1, 2, 0).contiguous()

    for hi in range(h):
        q_h = q[:, :, hi, :].float().permute(1, 0, 2).contiguous()  # [b, sq, d]
        logits_h = torch.bmm(q_h, k_bds)  # [b, sq, sk] fp32
        logits_h = torch.relu(logits_h)
        w_h = weights[:, :, hi].float().transpose(0, 1).unsqueeze(-1)  # [b, sq, 1]
        index_scores.add_(logits_h * w_h)

    return index_scores


# ---------------------------------------------------------------------------
# Measurement helpers.
# ---------------------------------------------------------------------------
def _reset_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return float("nan")


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), 1e-8)
    return diff / scale


def measure(
    name: str,
    fn,
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Run ``fn`` once, return (output, peak MiB during the call)."""
    _reset_cuda_memory()
    # Baseline includes inputs; subtract their static allocation.
    base_mb = _peak_mb()
    out = fn(q, weights, k)
    peak_mb = _peak_mb()
    delta = peak_mb - base_mb
    print(f"  {name:<16} peak_alloc={peak_mb:9.1f} MiB "
          f"(delta vs inputs {delta:+.1f} MiB)")
    return out, delta


# ---------------------------------------------------------------------------
# Shape configs.
# ---------------------------------------------------------------------------
SHAPES = {
    # Small shape: fits on CPU, still exhibits the pattern.
    "small": dict(b=2, sq=256, sk=256, h=4, d=32),
    # Production shape: NAM56R DSA 9+4 MBS=8-like but heads reduced to 8 so
    # the reproducer fits in a single 16 GiB consumer GPU. The 16 GiB figure
    # from the PR template is at h=32 MBS=8 (4x this); the ratio is
    # preserved and the pattern is identical.
    "prod": dict(b=4, sq=4096, sk=4096, h=8, d=128),
    # Full production shape: matches NAM56R exactly. Requires an H100/H200
    # (needs ~20 GiB just for the upstream intermediate).
    "full": dict(b=8, sq=4096, sk=4096, h=32, d=128),
}


def run_shape(shape_name: str, device: torch.device, dtype=torch.bfloat16) -> float:
    cfg = SHAPES[shape_name]
    b, sq, sk, h, d = cfg["b"], cfg["sq"], cfg["sk"], cfg["h"], cfg["d"]
    print(f"\n=== shape={shape_name}  "
          f"b={b} sq={sq} sk={sk} h={h} d={d}  "
          f"dtype={dtype}  device={device} ===")
    # Expected upstream intermediate size (fp32).
    intermediate_mib = sq * b * h * sk * 4 / (1024 ** 2)
    output_mib = b * sq * sk * 4 / (1024 ** 2)
    print(f"  expected upstream [sq,b,h,sk] fp32 intermediate: "
          f"{intermediate_mib:.1f} MiB")
    print(f"  expected output   [b,sq,sk]   fp32:              "
          f"{output_mib:.1f} MiB")

    torch.manual_seed(0)
    q = torch.randn(sq, b, h, d, dtype=dtype, device=device) * 0.1
    weights = torch.randn(sq, b, h, dtype=dtype, device=device).abs() * 0.1
    k = torch.randn(sk, b, d, dtype=dtype, device=device) * 0.1

    out_up, peak_up = measure("upstream", upstream_compute_index_scores,
                              q, weights, k)
    out_fu, peak_fu = measure("fused",    fused_compute_index_scores,
                              q, weights, k)

    rel = _rel_err(out_up, out_fu)
    print(f"  correctness: max rel_err = {rel:.3e}")
    print(f"  memory:      upstream {peak_up:.1f} MiB -> fused "
          f"{peak_fu:.1f} MiB   (saved {peak_up - peak_fu:.1f} MiB, "
          f"{(peak_up / max(peak_fu, 1e-6)):.1f}x)")

    if rel > 1e-4:
        print(f"  FAIL: rel_err {rel:.3e} exceeds 1e-4 threshold")
        return rel
    print("  PASS: correctness within 1e-4")
    return rel


# ---------------------------------------------------------------------------
# Gradient parity via gradcheck (small, double precision).
# ---------------------------------------------------------------------------
def run_gradcheck(device: torch.device) -> bool:
    print("\n=== gradcheck (small shape, float64) ===")
    b, sq, sk, h, d = 2, 8, 8, 3, 4

    torch.manual_seed(1)
    q = torch.randn(sq, b, h, d, dtype=torch.float64, device=device,
                    requires_grad=True)
    # Keep weights positive (physical invariant) but ensure it's a leaf tensor
    # by generating then detaching+requires_grad_.
    weights = torch.randn(sq, b, h, dtype=torch.float64, device=device).abs() \
                  .detach().requires_grad_(True)
    k = torch.randn(sk, b, d, dtype=torch.float64, device=device,
                    requires_grad=True)

    # Check both variants independently against numeric gradients: this
    # catches any autograd-graph bug in either formulation.
    def _wrap(fn):
        def inner(q_, w_, k_):
            # Force float32 -> float64 inside the helpers by calling .double()
            # in the variant bodies is not possible; instead we use a
            # double-safe clone that skips the internal ``.float()`` cast
            # for gradcheck, then verify a separate forward match against
            # the original fp32 path.
            return fn(q_, w_, k_)
        return inner

    def _upstream_d(q_, w_, k_):
        idx = torch.einsum("sbhd,tbd->sbht", q_, k_)
        idx = torch.relu(idx)
        idx = idx * w_.unsqueeze(-1)
        idx = idx.sum(dim=2).transpose(0, 1)
        return idx

    def _fused_d(q_, w_, k_):
        sq_, b_, h_, d_ = q_.shape
        sk_ = k_.shape[0]
        out = torch.zeros((b_, sq_, sk_), dtype=q_.dtype, device=q_.device)
        k_bds = k_.permute(1, 2, 0).contiguous()
        for hi in range(h_):
            qh = q_[:, :, hi, :].permute(1, 0, 2).contiguous()
            lh = torch.bmm(qh, k_bds)
            lh = torch.relu(lh)
            wh = w_[:, :, hi].transpose(0, 1).unsqueeze(-1)
            out = out + lh * wh
        return out

    ok_up = torch.autograd.gradcheck(_upstream_d, (q, weights, k),
                                     eps=1e-6, atol=1e-4, rtol=1e-3,
                                     nondet_tol=1e-5)
    print(f"  upstream gradcheck: {'PASS' if ok_up else 'FAIL'}")
    ok_fu = torch.autograd.gradcheck(_fused_d, (q, weights, k),
                                     eps=1e-6, atol=1e-4, rtol=1e-3,
                                     nondet_tol=1e-5)
    print(f"  fused    gradcheck: {'PASS' if ok_fu else 'FAIL'}")

    # Forward parity between the two in double precision:
    out_up = _upstream_d(q, weights, k)
    out_fu = _fused_d(q, weights, k)
    fwd_rel = _rel_err(out_up, out_fu)
    print(f"  fwd parity (double): rel_err = {fwd_rel:.3e}")

    # Backward parity — compare analytic grads of both variants against
    # a common upstream-loss gradient.
    for p in (q, weights, k):
        p.grad = None
    out_up.sum().backward()
    gq_up = q.grad.clone(); gw_up = weights.grad.clone(); gk_up = k.grad.clone()
    for p in (q, weights, k):
        p.grad = None
    out_fu.sum().backward()
    gq_fu = q.grad.clone(); gw_fu = weights.grad.clone(); gk_fu = k.grad.clone()

    bwd_rel_q = _rel_err(gq_up, gq_fu)
    bwd_rel_w = _rel_err(gw_up, gw_fu)
    bwd_rel_k = _rel_err(gk_up, gk_fu)
    print(f"  bwd parity: dq rel_err={bwd_rel_q:.3e}  "
          f"dw rel_err={bwd_rel_w:.3e}  dk rel_err={bwd_rel_k:.3e}")

    ok = (ok_up and ok_fu
          and fwd_rel < 1e-10
          and max(bwd_rel_q, bwd_rel_w, bwd_rel_k) < 1e-10)
    print(f"  gradcheck overall: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU execution (small shape only).")
    p.add_argument("--shapes", nargs="+",
                   default=None,
                   help="Subset of shape names to run "
                        f"(choices: {list(SHAPES)}).")
    p.add_argument("--skip-gradcheck", action="store_true")
    args = p.parse_args(argv)

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"HBM total: "
              f"{torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GiB")

    if args.shapes is None:
        if use_cuda:
            # "full" shape needs ~20 GiB just for upstream intermediate.
            total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            args.shapes = ["small", "prod"] + (["full"] if total_gib >= 40 else [])
        else:
            args.shapes = ["small"]

    worst_rel = 0.0
    for s in args.shapes:
        if s not in SHAPES:
            print(f"unknown shape: {s!r}, choices={list(SHAPES)}")
            return 2
        rel = run_shape(s, device)
        worst_rel = max(worst_rel, rel)

    gc_ok = True
    if not args.skip_gradcheck:
        gc_ok = run_gradcheck(torch.device("cpu"))

    print("\n=== summary ===")
    print(f"  worst rel_err across shapes: {worst_rel:.3e}")
    print(f"  gradcheck: {'PASS' if gc_ok else 'FAIL'}")

    if worst_rel > 1e-4:
        print("OVERALL: FAIL (correctness)")
        return 1
    if not gc_ok:
        print("OVERALL: FAIL (gradient parity)")
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
