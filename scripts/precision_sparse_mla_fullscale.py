#!/usr/bin/env python3
"""Precision comparison: TileLang SparseMLA vs PyTorch reference at FULL training scale.

Compares forward output, dq, and dkv between TileLang fused kernels and a
PyTorch dense-attention-masked-to-topk reference implementation.

Scale: S=4096, H=28, d_total=192, d_v=64, topk=256, kv_group=1, batch=1
"""

import sys
import time

import torch
import torch.nn.functional as F


def pytorch_sparse_mla_fwd(q, kv, indices, sm_scale, d_v):
    """PyTorch reference: dense attention masked to topk positions.

    Args:
        q:       [B, S, H, D_total]  bf16
        kv:      [B, S_kv, G, D_total]  bf16
        indices: [B, S, G, topk]  int32, -1 = invalid
        sm_scale: float
        d_v:     int, value head dimension (first d_v channels of kv used as V)

    Returns:
        out: [B, S, H, d_v]  bf16
        lse: [B, S, H]  fp32
    """
    B, S, H, D_total = q.shape
    _, S_kv, G, _ = kv.shape
    topk = indices.shape[-1]
    D_tail = D_total - d_v

    out = torch.zeros(B, S, H, d_v, dtype=q.dtype, device=q.device)
    lse = torch.zeros(B, S, H, dtype=torch.float32, device=q.device)

    for b in range(B):
        for g in range(G):
            heads_per_group = H // G
            for h_local in range(heads_per_group):
                h = g * heads_per_group + h_local
                for s in range(S):
                    # Gather topk KV positions
                    idx = indices[b, s, g]  # [topk]
                    valid_mask = idx >= 0  # [topk]

                    # q vector: [D_total]
                    q_vec = q[b, s, h].float()  # [D_total]

                    # Gather K vectors at topk positions: [topk, D_total]
                    safe_idx = idx.clamp(min=0).long()
                    k_gathered = kv[b, safe_idx, g].float()  # [topk, D_total]

                    # QK scores: [topk]
                    scores = (q_vec.unsqueeze(0) * k_gathered).sum(-1) * sm_scale  # [topk]
                    scores = scores.where(valid_mask, torch.tensor(float('-inf'), device=q.device))

                    # Softmax
                    max_score = scores.max()
                    exp_scores = torch.exp(scores - max_score)
                    sum_exp = exp_scores.sum()
                    attn_w = exp_scores / sum_exp  # [topk]

                    # Gather V at topk positions: [topk, d_v]
                    v_gathered = kv[b, safe_idx, g, :d_v].float()  # [topk, d_v]

                    # Weighted sum
                    out_vec = (attn_w.unsqueeze(-1) * v_gathered).sum(0)  # [d_v]
                    out[b, s, h] = out_vec.to(q.dtype)
                    lse[b, s, h] = torch.log(sum_exp) + max_score

    return out, lse


def pytorch_sparse_mla_fwd_batched(q, kv, indices, sm_scale, d_v):
    """Vectorized PyTorch reference: sparse attention over topk positions.

    Same semantics as pytorch_sparse_mla_fwd but vectorized over batch/seq/heads.
    """
    B, S, H, D_total = q.shape
    _, S_kv, G, _ = kv.shape
    topk = indices.shape[-1]
    heads_per_group = H // G

    out = torch.zeros(B, S, H, d_v, dtype=q.dtype, device=q.device)
    lse = torch.zeros(B, S, H, dtype=torch.float32, device=q.device)

    for b in range(B):
        for g in range(G):
            # indices for this batch/group: [S, topk]
            idx = indices[b, :, g]  # [S, topk]
            valid_mask = (idx >= 0)  # [S, topk]
            safe_idx = idx.clamp(min=0).long()  # [S, topk]

            # Gather KV: [S, topk, D_total]
            kv_gathered = kv[b, :, g][safe_idx]  # [S, topk, D_total]

            for h_local in range(heads_per_group):
                h = g * heads_per_group + h_local
                # Q: [S, D_total]
                q_h = q[b, :, h].float()  # [S, D_total]

                # Scores: [S, topk]
                scores = torch.einsum('sd,std->st', q_h, kv_gathered.float()) * sm_scale

                # Mask invalid
                scores = scores.masked_fill(~valid_mask, float('-inf'))

                # Stable softmax
                max_scores = scores.max(dim=-1, keepdim=True).values  # [S, 1]
                exp_scores = torch.exp(scores - max_scores)  # [S, topk]
                sum_exp = exp_scores.sum(dim=-1, keepdim=True)  # [S, 1]
                attn_w = exp_scores / sum_exp  # [S, topk]

                # V gather: [S, topk, d_v]
                v_gathered = kv_gathered[:, :, :d_v].float()  # [S, topk, d_v]

                # Output: [S, d_v]
                out_h = torch.einsum('st,std->sd', attn_w, v_gathered)
                out[b, :, h] = out_h.to(q.dtype)
                lse[b, :, h] = torch.log(sum_exp.squeeze(-1)) + max_scores.squeeze(-1)

    return out, lse


def main():
    print("=" * 80)
    print("PRECISION COMPARISON: TileLang SparseMLA vs PyTorch Reference")
    print("FULL TRAINING SCALE")
    print("=" * 80)

    # ---- Parameters ----
    B = 1
    S = 4096
    H = 28       # heads
    D_total = 192  # dim + tail_dim
    d_v = 64     # value dimension
    D_tail = D_total - d_v  # = 128
    topk = 256
    G = 1        # kv_group
    seed = 42
    sm_scale = D_total ** (-0.5)

    print(f"\nParameters:")
    print(f"  batch={B}, seq={S}, heads={H}, d_total={D_total}, d_v={d_v}, d_tail={D_tail}")
    print(f"  topk={topk}, kv_group={G}, sm_scale={sm_scale:.6f}")
    print(f"  seed={seed}")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  dtype: bf16")

    # ---- Create random inputs ----
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    q = torch.randn(B, S, H, D_total, dtype=torch.bfloat16, device="cuda") * 0.1
    kv = torch.randn(B, S, G, D_total, dtype=torch.bfloat16, device="cuda") * 0.1

    # Build valid causal indices with -1 sentinel for padding
    indices = torch.full((B, S, G, topk), -1, dtype=torch.int32, device="cuda")
    print(f"\nBuilding causal topk indices...")
    for b in range(B):
        for s in range(S):
            max_idx = min(s + 1, S)
            n = min(max_idx, topk)
            if n > 0:
                perm = torch.randperm(max_idx, device="cuda")[:n].sort().values
                indices[b, s, 0, :n] = perm.to(torch.int32)
    print(f"  Done. Valid indices range: [{indices[indices >= 0].min().item()}, {indices[indices >= 0].max().item()}]")

    # Count how many positions have < topk valid entries (early positions)
    valid_counts = (indices[0, :, 0] >= 0).sum(dim=-1)  # [S]
    n_partial = (valid_counts < topk).sum().item()
    print(f"  Positions with < topk valid entries: {n_partial} (first {n_partial} positions)")

    # ---- PyTorch reference forward + backward ----
    print(f"\n{'='*80}")
    print("PYTORCH REFERENCE forward + backward")
    print(f"{'='*80}")

    q_ref = q.clone().detach().requires_grad_(True)
    kv_ref = kv.clone().detach().requires_grad_(True)

    t0 = time.time()
    ref_out, ref_lse = pytorch_sparse_mla_fwd_batched(q_ref, kv_ref, indices, sm_scale, d_v)
    torch.cuda.synchronize()
    t_ref_fwd = time.time() - t0
    print(f"  Forward: {t_ref_fwd:.2f}s")
    print(f"  out shape: {ref_out.shape}, lse shape: {ref_lse.shape}")
    print(f"  out stats: mean={ref_out.float().mean():.6f}, std={ref_out.float().std():.6f}")

    # Backward
    grad_out = torch.randn_like(ref_out) * 0.01
    t0 = time.time()
    ref_out.backward(grad_out)
    torch.cuda.synchronize()
    t_ref_bwd = time.time() - t0
    print(f"  Backward: {t_ref_bwd:.2f}s")

    ref_dq = q_ref.grad.clone()
    ref_dkv = kv_ref.grad.clone()
    print(f"  dq shape: {ref_dq.shape}, dkv shape: {ref_dkv.shape}")
    print(f"  dq stats: mean={ref_dq.float().mean():.8f}, std={ref_dq.float().std():.8f}, max_abs={ref_dq.float().abs().max():.6f}")
    print(f"  dkv stats: mean={ref_dkv.float().mean():.8f}, std={ref_dkv.float().std():.8f}, max_abs={ref_dkv.float().abs().max():.6f}")

    # ---- TileLang SparseMLA forward + backward ----
    print(f"\n{'='*80}")
    print("TILELANG SparseMLA forward + backward")
    print(f"{'='*80}")

    q_tl = q.clone().detach().requires_grad_(True)
    kv_tl = kv.clone().detach().requires_grad_(True)

    from cppmega.megatron.sparse_mla_ops.sparse_mla import SparseMLA

    t0 = time.time()
    tl_out, tl_lse = SparseMLA.apply(q_tl, kv_tl, indices, sm_scale, d_v)
    torch.cuda.synchronize()
    t_tl_fwd = time.time() - t0
    print(f"  Forward (incl JIT if first): {t_tl_fwd:.2f}s")
    print(f"  out shape: {tl_out.shape}, lse shape: {tl_lse.shape}")
    print(f"  out stats: mean={tl_out.float().mean():.6f}, std={tl_out.float().std():.6f}")

    # Backward
    t0 = time.time()
    tl_out.backward(grad_out[:, :, :, :d_v] if grad_out.shape[-1] != d_v else grad_out)
    torch.cuda.synchronize()
    t_tl_bwd = time.time() - t0
    print(f"  Backward (incl JIT if first): {t_tl_bwd:.2f}s")

    tl_dq = q_tl.grad.clone()
    tl_dkv = kv_tl.grad.clone()
    print(f"  dq shape: {tl_dq.shape}, dkv shape: {tl_dkv.shape}")
    print(f"  dq stats: mean={tl_dq.float().mean():.8f}, std={tl_dq.float().std():.8f}, max_abs={tl_dq.float().abs().max():.6f}")
    print(f"  dkv stats: mean={tl_dkv.float().mean():.8f}, std={tl_dkv.float().std():.8f}, max_abs={tl_dkv.float().abs().max():.6f}")

    # ---- Comparison: Forward ----
    print(f"\n{'='*80}")
    print("FORWARD COMPARISON")
    print(f"{'='*80}")

    fwd_diff = (ref_out.float() - tl_out.float()).abs()
    fwd_rel = fwd_diff / (ref_out.float().abs() + 1e-8)
    print(f"  max_abs_err:  {fwd_diff.max().item():.6f}")
    print(f"  mean_abs_err: {fwd_diff.mean().item():.6f}")
    print(f"  max_rel_err:  {fwd_rel.max().item():.6f}")
    print(f"  mean_rel_err: {fwd_rel.mean().item():.6f}")

    # Where are the large forward errors?
    threshold_fwd = 0.01
    large_fwd = fwd_diff > threshold_fwd
    n_large_fwd = large_fwd.sum().item()
    total_fwd = fwd_diff.numel()
    print(f"  elements > {threshold_fwd}: {n_large_fwd}/{total_fwd} ({100*n_large_fwd/total_fwd:.4f}%)")

    # LSE comparison
    lse_diff = (ref_lse - tl_lse).abs()
    print(f"\n  LSE max_abs_err:  {lse_diff.max().item():.6f}")
    print(f"  LSE mean_abs_err: {lse_diff.mean().item():.6f}")

    # ---- Comparison: dQ ----
    print(f"\n{'='*80}")
    print("dQ COMPARISON")
    print(f"{'='*80}")

    dq_diff = (ref_dq.float() - tl_dq.float()).abs()
    dq_rel = dq_diff / (ref_dq.float().abs() + 1e-8)
    print(f"  max_abs_err:  {dq_diff.max().item():.6f}")
    print(f"  mean_abs_err: {dq_diff.mean().item():.8f}")
    print(f"  max_rel_err:  {dq_rel.max().item():.6f}")
    print(f"  mean_rel_err: {dq_rel.mean().item():.6f}")

    for threshold in [0.01, 0.1, 1.0]:
        n_large = (dq_diff > threshold).sum().item()
        print(f"  elements > {threshold}: {n_large}/{dq_diff.numel()} ({100*n_large/dq_diff.numel():.4f}%)")

    # ---- Comparison: dKV ----
    print(f"\n{'='*80}")
    print("dKV COMPARISON")
    print(f"{'='*80}")

    dkv_diff = (ref_dkv.float() - tl_dkv.float()).abs()
    dkv_rel = dkv_diff / (ref_dkv.float().abs() + 1e-8)
    print(f"  max_abs_err:  {dkv_diff.max().item():.6f}")
    print(f"  mean_abs_err: {dkv_diff.mean().item():.8f}")
    print(f"  max_rel_err:  {dkv_rel.max().item():.6f}")
    print(f"  mean_rel_err: {dkv_rel.mean().item():.6f}")

    for threshold in [0.01, 0.1, 1.0, 5.0, 10.0, 44.0]:
        n_large = (dkv_diff > threshold).sum().item()
        print(f"  elements > {threshold}: {n_large}/{dkv_diff.numel()} ({100*n_large/dkv_diff.numel():.4f}%)")

    # ---- Spatial analysis of large dKV errors ----
    print(f"\n{'='*80}")
    print("SPATIAL ANALYSIS OF LARGE dKV ERRORS")
    print(f"{'='*80}")

    # Analyze errors > 1.0
    large_mask = dkv_diff > 1.0
    if large_mask.any():
        # Find positions of large errors
        # dkv shape: [B, S, G, D_total]
        large_positions = large_mask.nonzero(as_tuple=False)  # [N, 4] = (b, s, g, d)
        n_large_err = large_positions.shape[0]
        print(f"  Total elements with |err| > 1.0: {n_large_err}")

        # Sequence position distribution
        seq_positions = large_positions[:, 1]  # s dimension
        print(f"\n  Sequence position of large errors:")
        print(f"    min seq: {seq_positions.min().item()}")
        print(f"    max seq: {seq_positions.max().item()}")
        print(f"    mean seq: {seq_positions.float().mean().item():.1f}")
        print(f"    median seq: {seq_positions.float().median().item():.1f}")

        # Histogram by sequence region (8 bins)
        n_bins = 8
        bin_size = S // n_bins
        print(f"\n  Distribution by sequence region (bin_size={bin_size}):")
        for i in range(n_bins):
            lo, hi = i * bin_size, (i + 1) * bin_size
            count = ((seq_positions >= lo) & (seq_positions < hi)).sum().item()
            total_in_bin = bin_size * G * D_total * B
            print(f"    seq [{lo:4d}, {hi:4d}): {count:6d} large errors ({100*count/total_in_bin:.4f}% of bin)")

        # Head/group distribution
        group_positions = large_positions[:, 2]
        print(f"\n  KV group distribution: {torch.bincount(group_positions, minlength=G).tolist()}")

        # Dimension distribution
        dim_positions = large_positions[:, 3]
        print(f"\n  Dimension of large errors:")
        print(f"    min dim: {dim_positions.min().item()}")
        print(f"    max dim: {dim_positions.max().item()}")
        # V dims (0..d_v-1) vs K_tail dims (d_v..D_total-1)
        n_in_v = (dim_positions < d_v).sum().item()
        n_in_ktail = (dim_positions >= d_v).sum().item()
        print(f"    in V dims [0,{d_v}): {n_in_v} ({100*n_in_v/n_large_err:.1f}%)")
        print(f"    in K_tail dims [{d_v},{D_total}): {n_in_ktail} ({100*n_in_ktail/n_large_err:.1f}%)")

        # Per-position error magnitude
        # Max error per sequence position
        dkv_diff_per_seq = dkv_diff[0, :, 0].max(dim=-1).values  # [S]
        top_err_positions = dkv_diff_per_seq.topk(min(20, S))
        print(f"\n  Top 20 sequence positions by max |error|:")
        for val, pos in zip(top_err_positions.values, top_err_positions.indices):
            # Count how many times this position is referenced in indices
            ref_count = (indices[0, :, 0] == pos.item()).sum().item()
            print(f"    seq={pos.item():5d}  max_err={val.item():.4f}  referenced_by={ref_count} queries")

        # Correlation with reference count
        ref_counts = torch.zeros(S, device=q.device)
        flat_idx = indices[0, :, 0].long()  # [S, topk]
        valid = flat_idx >= 0
        for s in range(S):
            for t in range(topk):
                if valid[s, t]:
                    ref_counts[flat_idx[s, t]] += 1

        # Correlation between reference count and error
        err_per_pos = dkv_diff[0, :, 0].max(dim=-1).values  # [S]
        correlation = torch.corrcoef(torch.stack([ref_counts, err_per_pos]))[0, 1].item()
        print(f"\n  Pearson correlation(reference_count, max_error): {correlation:.4f}")
        print(f"  Reference count stats: min={ref_counts.min():.0f}, max={ref_counts.max():.0f}, mean={ref_counts.mean():.1f}")

        # Top referenced positions and their errors
        top_refs = ref_counts.topk(10)
        print(f"\n  Top 10 most-referenced KV positions:")
        for val, pos in zip(top_refs.values, top_refs.indices):
            err = err_per_pos[pos].item()
            print(f"    seq={pos.item():5d}  ref_count={val.item():.0f}  max_err={err:.4f}")

    else:
        print("  No elements with |err| > 1.0 -- great!")

    # ---- Comparison: errors at boundary positions (first topk positions) ----
    print(f"\n{'='*80}")
    print("BOUNDARY ANALYSIS: First 256 seq positions (partial valid indices)")
    print(f"{'='*80}")

    boundary_dkv_diff = dkv_diff[0, :256, 0]
    interior_dkv_diff = dkv_diff[0, 256:, 0]
    print(f"  Boundary (seq 0-255):")
    print(f"    max_abs_err:  {boundary_dkv_diff.max().item():.6f}")
    print(f"    mean_abs_err: {boundary_dkv_diff.mean().item():.8f}")
    print(f"  Interior (seq 256+):")
    print(f"    max_abs_err:  {interior_dkv_diff.max().item():.6f}")
    print(f"    mean_abs_err: {interior_dkv_diff.mean().item():.8f}")

    # Same for dQ
    boundary_dq_diff = dq_diff[0, :256]
    interior_dq_diff = dq_diff[0, 256:]
    print(f"\n  dQ Boundary (seq 0-255):")
    print(f"    max_abs_err:  {boundary_dq_diff.max().item():.6f}")
    print(f"    mean_abs_err: {boundary_dq_diff.mean().item():.8f}")
    print(f"  dQ Interior (seq 256+):")
    print(f"    max_abs_err:  {interior_dq_diff.max().item():.6f}")
    print(f"    mean_abs_err: {interior_dq_diff.mean().item():.8f}")

    # ---- Scale comparison with unit test ----
    print(f"\n{'='*80}")
    print("SCALE COMPARISON vs UNIT TEST (max_err=44 at S=128)")
    print(f"{'='*80}")
    print(f"  Full scale (S={S}) dKV max_abs_err: {dkv_diff.max().item():.4f}")
    print(f"  Full scale (S={S}) dQ  max_abs_err: {dq_diff.max().item():.4f}")
    print(f"  Full scale (S={S}) fwd max_abs_err: {fwd_diff.max().item():.4f}")
    unit_test_max_err = 44.0
    growth = dkv_diff.max().item() / unit_test_max_err if dkv_diff.max().item() > 0 else 0
    print(f"  Growth factor vs unit test max_err=44: {growth:.2f}x")

    if dkv_diff.max().item() > unit_test_max_err:
        print(f"  ** WARNING: Error GREW at full scale! **")
    elif dkv_diff.max().item() > 1.0:
        print(f"  ** NOTE: Error > 1.0 but smaller than unit test -- better at full scale **")
    else:
        print(f"  ** GOOD: Error < 1.0 at full scale **")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
