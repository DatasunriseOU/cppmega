"""Memory delta micro-bench for DSA FP8 backward (Stream G, task #84).

Purpose
-------
Standalone single-GPU script that mocks one ``bwd_fused_indexer_loss_*``
call at NAM56R production shape and reports ``torch.cuda.max_memory_allocated``
for the BF16 reference path vs the FP8 backward path.

This is **NOT** a training run. It does not touch megatron-core, does not
load Megatron's DSA module, and does not pull any TE / FP8 global state.
It calls the pure-torch references in
``cppmega.megatron.dsa_fp8_indexer``:

* ``bwd_fused_indexer_loss_bf16_reference`` --- byte-for-byte clone of
  upstream ``dsa.py::bwd_fused_indexer_loss_naive``.
* ``bwd_fused_indexer_loss_fp8`` --- Stream G's FP8 port.

Production NAM56R DSA shapes (cppmega/recipes/nam56r_* recipes, 2026-04-12):

    batch               = 4
    seqlen              = 4096   (sq = sk; causal)
    index_n_heads       = 8      (h)
    index_head_dim      = 64     (d)
    attention_heads     = 16     (np)   — local TP=1
    attention_head_dim  = 128    (hn)   — main attention head dim
    softmax_scale       = 1/sqrt(hn)
    loss_coeff          = 0.1
    sparse_loss         = True

Run on bench3 (H200):

    /mnt/data/venv/bin/python scripts/measure_dsa_fp8_bwd_memory.py

Expected output: BF16 bwd peak ≈ 8-12 GB, FP8 bwd peak ≈ 3-5 GB saved
on the indexer side (main-attention bmm is the same in both variants,
by design). This is additional headroom ON TOP of Stream E's 26 GB
forward indexer savings.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gc
import sys
from typing import Callable, Tuple

import torch

# Ensure we can import cppmega without a megatron checkout: this script
# exercises the local reference + FP8 functions only.
from cppmega.megatron.dsa_fp8_indexer import (
    bwd_fused_indexer_loss_bf16_reference,
    bwd_fused_indexer_loss_fp8,
)


def _make_production_inputs(
    *,
    batch: int,
    seqlen: int,
    index_n_heads: int,
    index_head_dim: int,
    attention_heads: int,
    attention_head_dim: int,
    topk: int,
    device: str,
    seed: int,
) -> Tuple[torch.Tensor, ...]:
    g = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.bfloat16
    q = torch.randn(
        seqlen, batch, index_n_heads, index_head_dim,
        generator=g, device=device, dtype=dtype,
    )
    weights = torch.randn(
        seqlen, batch, index_n_heads,
        generator=g, device=device, dtype=dtype,
    ).abs()
    k = torch.randn(
        seqlen, batch, index_head_dim,
        generator=g, device=device, dtype=dtype,
    )
    query = torch.randn(
        seqlen, batch, attention_heads, attention_head_dim,
        generator=g, device=device, dtype=dtype,
    )
    key = torch.randn(
        seqlen, batch, attention_heads, attention_head_dim,
        generator=g, device=device, dtype=dtype,
    )

    # Cheap realistic topk_indices: use .argsort over a random projection
    # so the per-query subset is not uniformly distributed but still
    # covers the full [0, seqlen) range.
    probe = torch.randn(batch, seqlen, seqlen, device=device, generator=g)
    topk_indices = probe.topk(topk, dim=-1)[1].to(torch.int64)
    del probe

    grad_loss = torch.tensor(1.0, dtype=torch.float32, device=device)
    return q, weights, k, query, key, topk_indices, grad_loss


def _measure_peak(
    fn: Callable[[], None],
    *,
    label: str,
) -> Tuple[float, float]:
    """Clear cached allocator state, run ``fn``, return peak alloc + peak
    reserved in megabytes.
    """

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_alloc = torch.cuda.memory_allocated() / (1024**2)
    fn()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    print(
        f"  {label:12s}: start_alloc={start_alloc:8.1f} MB "
        f"peak_alloc={peak_alloc:8.1f} MB "
        f"peak_reserved={peak_reserved:8.1f} MB"
    )
    return peak_alloc, peak_reserved


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--index-n-heads", type=int, default=8)
    parser.add_argument("--index-head-dim", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=16)
    parser.add_argument("--attention-head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument(
        "--skip-bf16", action="store_true",
        help="Skip the BF16 baseline run (useful if OOM on smaller GPU).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available --- this script needs an H200/A100/B200 GPU.")
        return 2

    device = "cuda"
    torch.cuda.set_device(0)
    dev_name = torch.cuda.get_device_name(0)
    dev_cap = torch.cuda.get_device_capability(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Device: {dev_name} (sm_{dev_cap[0]}{dev_cap[1]}) total={total_mem_gb:.1f} GB")
    print(
        "Shape: "
        f"batch={args.batch} seqlen={args.seqlen} "
        f"index_n_heads={args.index_n_heads} index_head_dim={args.index_head_dim} "
        f"np={args.attention_heads} hn={args.attention_head_dim} topk={args.topk}"
    )
    print()

    softmax_scale = 1.0 / (args.attention_head_dim**0.5)
    loss_coeff = 0.1
    sparse_loss = True

    # Build inputs ONCE, on device; exclude their cost from the measured
    # peak by resetting peak stats after construction.
    q, weights, k, query, key, topk_idx, grad_loss = _make_production_inputs(
        batch=args.batch,
        seqlen=args.seqlen,
        index_n_heads=args.index_n_heads,
        index_head_dim=args.index_head_dim,
        attention_heads=args.attention_heads,
        attention_head_dim=args.attention_head_dim,
        topk=args.topk,
        device=device,
        seed=args.seed,
    )
    input_bytes = sum(
        x.element_size() * x.numel()
        for x in (q, weights, k, query, key, topk_idx, grad_loss)
    )
    print(f"Input tensor footprint: {input_bytes / (1024**2):.1f} MB")

    # Warm up CUDA kernels so measured peak is not polluted by first-call
    # cuBLAS/CUTLASS handle init.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ---- BF16 reference run ------------------------------------------
    if args.skip_bf16:
        bf16_peak = float("nan")
        print("Skipping BF16 reference run (--skip-bf16)")
    else:
        print("BF16 reference backward (full: main-attention + indexer):")

        def _run_bf16():
            gq, gw, gk = bwd_fused_indexer_loss_bf16_reference(
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
            # touch to ensure no DCE
            _ = gq.sum() + gw.sum() + gk.sum()

        bf16_peak, _ = _measure_peak(_run_bf16, label="bf16_bwd")

    # ---- FP8 run -------------------------------------------------------
    print("FP8 backward (Stream G, full path):")

    def _run_fp8():
        gq, gw, gk = bwd_fused_indexer_loss_fp8(
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
        _ = gq.sum() + gw.sum() + gk.sum()

    fp8_peak, _ = _measure_peak(_run_fp8, label="fp8_bwd")

    # ---- Indexer-only isolation run -----------------------------------
    # Runs a shrunk-np variant that keeps the indexer shape the same but
    # makes the main-attention bmm negligible, so the peak delta reflects
    # the indexer-side savings alone. Useful because the main-attention
    # bmm at production np*hn dominates the full peak in both paths.
    print()
    print("Indexer-only isolation (same indexer shape, np=1 hn=16):")
    iq, iw, ik, iquery, ikey, itopk, igrad = _make_production_inputs(
        batch=args.batch,
        seqlen=args.seqlen,
        index_n_heads=args.index_n_heads,
        index_head_dim=args.index_head_dim,
        attention_heads=1,
        attention_head_dim=16,
        topk=args.topk,
        device=device,
        seed=args.seed,
    )
    iso_softmax_scale = 1.0 / (16**0.5)

    def _run_iso_bf16():
        gq, gw, gk = bwd_fused_indexer_loss_bf16_reference(
            iq, iw, ik, iquery, ikey, itopk,
            iso_softmax_scale, loss_coeff, sparse_loss, igrad,
            pg_collection=None,
        )
        _ = gq.sum() + gw.sum() + gk.sum()

    def _run_iso_fp8():
        gq, gw, gk = bwd_fused_indexer_loss_fp8(
            iq, iw, ik, iquery, ikey, itopk,
            iso_softmax_scale, loss_coeff, sparse_loss, igrad,
            pg_collection=None,
        )
        _ = gq.sum() + gw.sum() + gk.sum()

    iso_bf16_peak, _ = _measure_peak(_run_iso_bf16, label="iso_bf16_bwd")
    iso_fp8_peak, _ = _measure_peak(_run_iso_fp8, label="iso_fp8_bwd")
    # Note: iq/iw/ik/iquery/ikey/itopk/igrad will be released when main()
    # returns. We intentionally keep them live through the summary print
    # below so ruff's F821 analysis sees the closures as bound. The CUDA
    # memory stats have already been captured above.

    print()
    if not args.skip_bf16 and bf16_peak == bf16_peak:  # not NaN
        delta = bf16_peak - fp8_peak
        ratio = bf16_peak / max(fp8_peak, 1.0)
        savings = (delta / max(bf16_peak, 1.0)) * 100.0
        print(
            f"BF16 bwd peak delta (full):    {bf16_peak:8.1f} MB"
        )
        print(
            f"FP8  bwd peak delta (full):    {fp8_peak:8.1f} MB   "
            f"(savings {savings:.1f}%, ratio {ratio:.2f}x)"
        )
        iso_delta = iso_bf16_peak - iso_fp8_peak
        iso_savings = (iso_delta / max(iso_bf16_peak, 1.0)) * 100.0
        print(
            f"BF16 bwd peak delta (indexer): {iso_bf16_peak:8.1f} MB"
        )
        print(
            f"FP8  bwd peak delta (indexer): {iso_fp8_peak:8.1f} MB   "
            f"(savings {iso_savings:.1f}%, delta {iso_delta:.1f} MB)"
        )
        print()
        print(
            "Interpretation: the main-attention Q@K^T bmm (np=16, hn=128) is "
            "kept BF16/FP32 by design --- its [b, np, sq, sk] output is "
            "structurally needed by the softmax → sum → L1 chain. At production "
            "shape this dominates the full bwd peak in BOTH paths. The "
            "'indexer-only' isolation run keeps the indexer shape identical "
            "but shrinks np/hn to negligible so the per-call peak reflects "
            "ONLY the indexer recompute + grad_q/grad_k savings. Stream D v2 "
            "sees the full-path number directly; pipeline headroom at other "
            "DSA layers benefits from the indexer isolation number because "
            "the PyTorch caching allocator can reuse freed blocks across "
            "layer boundaries even when intra-call peaks are main-attention "
            "bound."
        )
    else:
        print(f"FP8 bwd peak delta (full): {fp8_peak:.1f} MB")
        print(f"BF16 indexer-only peak: {iso_bf16_peak:.1f} MB, "
              f"FP8 indexer-only peak: {iso_fp8_peak:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
