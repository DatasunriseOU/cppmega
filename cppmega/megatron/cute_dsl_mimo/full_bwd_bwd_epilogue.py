"""Full bwd_bwd PyTorch implementation matching TileLang exactly.

Vectorized (minimal Python for-loops). Produces all 14 outputs.
"""
import math
import torch


def full_bwd_bwd_pytorch(
    dout,           # (B, S, H, P) if reduceO
    q_raw,          # (B, S, R, G, N)
    k_raw,          # (B, S, R, G, N)
    v,              # (B, S, H, P)
    q_bias,         # (H, R, N)
    k_bias,         # (H, R, N)
    mimo_v,         # (H, R, P) -- Psi
    mimo_o,         # (H, R, P) -- Phi, or None
    angles,         # (B, S, H, N//rotary_dim_divisor)
    dA_cs,          # (B, H, S)
    dA_cs_rev,      # (B, H, S)
    dt,             # (B, H, S)
    trap,           # (B, H, S)
    D,              # (H,) or None
    segsum,         # (B, H, nchunks, cs, cs)
    states,         # (B, H, nchunks, N, P)
    qk_dot,         # (B, H, S, R, R)
    chunk_size=16,
    R=4,
    rotary_dim_divisor=4,
):
    B, S, R_dim, G, N = q_raw.shape
    H = mimo_v.shape[0]
    P = mimo_v.shape[2]
    cs = chunk_size
    nchunks = S // cs
    fcs = cs * R_dim
    dev = q_raw.device
    dtype = q_raw.dtype
    rdim = N // rotary_dim_divisor
    reduceO = mimo_o is not None

    # GQA mapping: i_h_qk = i_h // (H // G)
    heads_per_group = H // G

    # Process per head to match TileLang's per-head kernel
    # q_raw: (B, S, R, G, N) -- index [i_b, s, r, i_h_qk, n]
    # q_bias: (H, R, N) -- index [i_h, r, n]

    # Build per-head Q/K with bias + rotary + trap
    # Output shape: (B, H, nchunks, fcs, N) for Q and K (post-processing)
    q_all = torch.zeros(B, H, nchunks, fcs, N, dtype=torch.float32, device=dev)
    k_all = torch.zeros(B, H, nchunks, fcs, N, dtype=torch.float32, device=dev)
    q_pre_rot_all = torch.zeros(B, H, nchunks, fcs, N, dtype=torch.float32, device=dev)
    k_pre_rot_all = torch.zeros(B, H, nchunks, fcs, N, dtype=torch.float32, device=dev)
    k_pre_trap_all = torch.zeros(B, H, nchunks, fcs, N, dtype=torch.float32, device=dev)

    for ih in range(H):
        ih_qk = ih // heads_per_group
        # q_h: (B, S, R, N)
        q_h = q_raw[:, :, :, ih_qk, :].float() + q_bias[ih].unsqueeze(0).unsqueeze(0).float()  # (B, S, R, N)
        k_h = k_raw[:, :, :, ih_qk, :].float() + k_bias[ih].unsqueeze(0).unsqueeze(0).float()

        # Reshape to (B, nchunks, cs, R, N) -> (B, nchunks, fcs, N)
        q_h = q_h.view(B, nchunks, cs, R_dim, N)
        k_h = k_h.view(B, nchunks, cs, R_dim, N)

        # Save pre-rotated
        q_pre_rot_all[:, ih] = q_h.reshape(B, nchunks, fcs, N)
        k_pre_rot_all[:, ih] = k_h.reshape(B, nchunks, fcs, N)

        # Apply rotary
        # angles: (B, S, H, rdim) -> for this head: (B, S, rdim)
        ang_h = angles[:, :, ih, :].view(B, nchunks, cs, rdim)  # (B, nchunks, cs, rdim)
        cos_a = torch.cos(ang_h).unsqueeze(3)  # (B,nc,cs,1,rdim)
        sin_a = torch.sin(ang_h).unsqueeze(3)

        qf = q_h[..., :rdim]
        qs = q_h[..., N//2:N//2+rdim]
        q_rot = q_h.clone()
        q_rot[..., :rdim] = cos_a * qf - sin_a * qs
        q_rot[..., N//2:N//2+rdim] = sin_a * qf + cos_a * qs

        kf = k_h[..., :rdim]
        ks = k_h[..., N//2:N//2+rdim]
        k_rot = k_h.clone()
        k_rot[..., :rdim] = cos_a * kf - sin_a * ks
        k_rot[..., N//2:N//2+rdim] = sin_a * kf + cos_a * ks

        k_pre_trap_all[:, ih] = k_rot.reshape(B, nchunks, fcs, N)

        # Trap scaling
        gamma_h = dt[:, ih, :].float() * torch.sigmoid(trap[:, ih, :].float())  # (B, S)
        shifted_gamma_h = torch.zeros_like(gamma_h)
        if S > 1:
            shifted_gamma_h[:, :-1] = dt[:, ih, 1:].float() * torch.sigmoid(-trap[:, ih, 1:].float())
        ts_h = (gamma_h + shifted_gamma_h).view(B, nchunks, cs)  # (B, nc, cs)

        k_trap = k_rot * ts_h.unsqueeze(-1).unsqueeze(-1)  # (B, nc, cs, R, N)

        q_all[:, ih] = q_rot.reshape(B, nchunks, fcs, N)
        k_all[:, ih] = k_trap.reshape(B, nchunks, fcs, N)

    # Compute gamma and trap_scale for all heads
    gamma_all = dt.float() * torch.sigmoid(trap.float())  # (B, H, S)
    shifted_gamma_all = torch.zeros_like(gamma_all)
    if S > 1:
        shifted_gamma_all[..., :-1] = dt[..., 1:].float() * torch.sigmoid(-trap[..., 1:].float())
    trap_scale_all = gamma_all + shifted_gamma_all
    gamma_ch = gamma_all.view(B, H, nchunks, cs)
    trap_scale_ch = trap_scale_all.view(B, H, nchunks, cs)

    # dPhiO
    if reduceO:
        dout_ch = dout.view(B, nchunks, cs, H, P).permute(0, 3, 1, 2, 4).float()  # (B,H,nc,cs,P)
        dPhiO = dout_ch.unsqueeze(4) * mimo_o[None, :, None, None, :, :].float()  # (B,H,nc,cs,R,P)
    else:
        dPhiO = dout.view(B, nchunks, cs, R_dim, H, P).permute(0, 4, 1, 2, 3, 5).float()
    dPhiO_flat = dPhiO.reshape(B, H, nchunks, fcs, P)

    # PsiV
    v_ch = v.view(B, nchunks, cs, H, P).permute(0, 3, 1, 2, 4).float()  # (B,H,nc,cs,P)
    PsiV = v_ch.unsqueeze(4) * mimo_v[None, :, None, None, :, :].float()
    PsiV_flat = PsiV.reshape(B, H, nchunks, fcs, P)

    # Angles per head, chunked
    ang_all = angles.view(B, nchunks, cs, H, rdim).permute(0, 3, 1, 2, 4).float()  # (B,H,nc,cs,rdim)

    # Segsum fcs-level causal mask indices
    idx = torch.arange(fcs, device=dev)
    ci_idx = idx // R_dim
    causal = ci_idx.unsqueeze(1) < ci_idx.unsqueeze(0)  # (fcs, fcs)

    # dA_cs chunked
    dA_cs_ch = dA_cs.view(B, H, nchunks, cs)
    dA_cs_rev_ch = dA_cs_rev.view(B, H, nchunks, cs)
    qk_dot_ch = qk_dot.view(B, H, nchunks, cs, R_dim, R_dim)

    # Allocate outputs
    DK_o = torch.zeros(B, H, nchunks, fcs, N, dtype=torch.float32, device=dev)
    DQ_o = torch.zeros(B, H, nchunks, fcs, N, dtype=torch.float32, device=dev)
    DV_o = torch.zeros(B, H, nchunks, cs, P, dtype=torch.float32, device=dev)
    DPsi_acc = torch.zeros(B, H, R_dim, P, dtype=torch.float32, device=dev)
    DDA_CS_o = torch.zeros(B, H, nchunks, cs, dtype=torch.float32, device=dev)
    DDA_CS_REV_o = torch.zeros(B, H, nchunks, cs, dtype=torch.float32, device=dev)
    DFACTOR_o = torch.zeros(B, H, nchunks, cs, dtype=torch.float32, device=dev)
    DGAMMA_o = torch.zeros(B, H, nchunks, cs, dtype=torch.float32, device=dev)
    DSSDA_o = torch.zeros(B, H, nchunks, cs, cs, dtype=torch.float32, device=dev)
    DDA_o = torch.zeros(B, H, nchunks, cs, dtype=torch.float32, device=dev)
    DANG_o = torch.zeros(B, H, nchunks, cs, rdim, dtype=torch.float32, device=dev)

    dstates = torch.zeros(B, H, N, P, dtype=torch.float32, device=dev)

    for crev in range(nchunks):
        ci = nchunks - 1 - crev

        q_c = q_all[:, :, ci]         # (B,H,fcs,N)
        k_c = k_all[:, :, ci]         # trap-scaled, rotated
        dPh_c = dPhiO_flat[:, :, ci]   # (B,H,fcs,P)
        PsV_c = PsiV_flat[:, :, ci]    # (B,H,fcs,P)
        v_c = v_ch[:, :, ci]           # (B,H,cs,P)
        st_c = states[:, :, ci].float()  # (B,H,N,P)
        dacr_c = dA_cs_rev_ch[:, :, ci]
        dac_c = dA_cs_ch[:, :, ci]
        seg_c = segsum[:, :, ci]
        gam_c = gamma_ch[:, :, ci]
        qkd_c = qk_dot_ch[:, :, ci]
        kpt_c = k_pre_trap_all[:, :, ci]  # (B,H,fcs,N)
        qpr_c = q_pre_rot_all[:, :, ci]
        kpr_c = k_pre_rot_all[:, :, ci]
        ang_c = ang_all[:, :, ci]         # (B,H,cs,rdim)

        exp_rev = torch.exp(dacr_c)  # (B,H,cs)
        exp_rev_fcs = exp_rev.repeat_interleave(R_dim, dim=-1).unsqueeze(-1)  # (B,H,fcs,1)

        # TileLang uses bf16-truncated dstates_shared for GEMMs
        dstates_bf = dstates.to(dtype).float()

        # dPsiV = K @ dstates^T * exp(dA_cs_rev)
        dPsiV = torch.matmul(k_c, dstates_bf.transpose(-1,-2)) * exp_rev_fcs

        # LKQ + masking
        lkq = torch.matmul(k_c, q_c.transpose(-1,-2))
        lkq_save = lkq.clone()
        seg_exp = seg_c[:, :, ci_idx.unsqueeze(0), ci_idx.unsqueeze(1)]
        lkq_m = torch.where(causal[None, None], lkq * torch.exp(seg_exp), torch.zeros_like(lkq))
        dPsiV = dPsiV + torch.matmul(lkq_m, dPh_c)

        # Diagonal: D + qk_dot
        dPsiV_D = dPsiV.clone()
        if D is not None:
            dPsiV_D = dPsiV_D + dPh_c * D[None, :, None, None].float()

        # qk_dot contribution via einsum
        dPh_r = dPh_c.view(B, H, cs, R_dim, P)
        qkd_T = qkd_c.transpose(-1,-2).float()
        qkd_contrib = torch.einsum('bhcio,bhcop->bhcip', qkd_T, dPh_r)
        qkd_contrib = qkd_contrib * gam_c.unsqueeze(-1).unsqueeze(-1)
        dPsiV_D = dPsiV_D + qkd_contrib.reshape(B, H, fcs, P)

        # TileLang truncates dPsiV_D to dtype (bf16) via shared memory before DV/DPsi
        dPsiV_combined_bf = dPsiV_D.to(dtype).float()

        # DV
        dPsD_r = dPsiV_combined_bf.view(B, H, cs, R_dim, P)
        DV_o[:, :, ci] = (dPsD_r * mimo_v[None, :, None, :, :].float()).sum(dim=3)

        # DPsi
        DPsi_acc += torch.einsum('bhcrp,bhcp->bhrp', dPsD_r, v_c)

        # dqk_from_diag
        dqk = torch.matmul(dPh_c, PsV_c.transpose(-1,-2))

        # DGAMMA_DIAG: block-diagonal product of qk_dot * dqk
        dqk_diag = torch.zeros(B, H, cs, R_dim, R_dim, dtype=torch.float32, device=dev)
        for s in range(cs):
            dqk_diag[:, :, s] = dqk[:, :, s*R_dim:(s+1)*R_dim, s*R_dim:(s+1)*R_dim]
        DGAMMA_o[:, :, ci] = (qkd_c.float() * dqk_diag).sum(dim=(-1,-2))

        # Apply gamma to dqk
        gam_fcs = gam_c.repeat_interleave(R_dim, dim=-1).unsqueeze(-1)
        dqk = dqk * gam_fcs

        # dk path (uses bf16 dstates_shared in TileLang)
        dk = torch.matmul(PsV_c, dstates_bf.transpose(-1,-2))
        DDA_CS_REV_o[:, :, ci] = (k_c * dk).view(B, H, cs, R_dim*N).sum(dim=-1)
        dk = dk * exp_rev_fcs

        dk_intra = torch.matmul(PsV_c, dPh_c.transpose(-1,-2))

        # DSSDA
        kq_prod = lkq_save * dk_intra
        kq_r1 = kq_prod.view(B, H, fcs, cs, R_dim).sum(dim=-1)
        DSSDA_o[:, :, ci] = kq_r1.view(B, H, cs, R_dim, cs).sum(dim=3)

        # Mask dk_intra
        dk_intra_m = torch.where(causal[None,None], dk_intra * torch.exp(seg_exp), torch.zeros_like(dk_intra))

        dk_nodiag = dk + torch.matmul(dk_intra_m, q_c)

        # DFACTOR
        DFACTOR_o[:, :, ci] = (kpt_c * dk_nodiag).view(B, H, cs, R_dim*N).sum(dim=-1)

        # Trap scale
        ts_fcs = trap_scale_ch[:, :, ci].repeat_interleave(R_dim, dim=-1).unsqueeze(-1)
        dk_nodiag = dk_nodiag * ts_fcs

        # DDA
        da_cs_sum = dA_cs[:, :, ci*cs + cs - 1]
        ddA_sp = (st_c * dstates * torch.exp(da_cs_sum).unsqueeze(-1).unsqueeze(-1)).sum(dim=(-1,-2))
        DDA_o[:, :, ci] = ddA_sp.unsqueeze(-1).expand(-1, -1, cs)

        # dq path
        dq = torch.matmul(dPh_c, st_c.transpose(-1,-2))
        DDA_CS_o[:, :, ci] = (q_c * dq).view(B, H, cs, R_dim*N).sum(dim=-1)

        exp_cs = torch.exp(dac_c)
        exp_cs_fcs = exp_cs.repeat_interleave(R_dim, dim=-1).unsqueeze(-1)
        dq = dq * exp_cs_fcs
        dq = dq + torch.matmul(dk_intra_m.transpose(-1,-2), k_c)

        # Rotary inverse + dangle
        cos_ac = torch.cos(ang_c).unsqueeze(-2)
        sin_ac = torch.sin(ang_c).unsqueeze(-2)

        # dk inverse rotary + dangle
        dk_r = dk_nodiag.view(B, H, cs, R_dim, N)
        dk_f = dk_r[..., :rdim].clone()
        dk_s = dk_r[..., N//2:N//2+rdim].clone()
        kpr_r = kpr_c.view(B, H, cs, R_dim, N)
        kpr_f = kpr_r[..., :rdim]
        kpr_s = kpr_r[..., N//2:N//2+rdim]

        dang_dk = (
            dk_f * (-kpr_f * sin_ac - kpr_s * cos_ac) +
            dk_s * (kpr_f * cos_ac - kpr_s * sin_ac)
        )

        dk_inv = dk_nodiag.clone().view(B, H, cs, R_dim, N)
        dk_inv[..., :rdim] = cos_ac * dk_f + sin_ac * dk_s
        dk_inv[..., N//2:N//2+rdim] = -sin_ac * dk_f + cos_ac * dk_s
        dk_inv = dk_inv.reshape(B, H, fcs, N)

        # dqk_from_diag contribution to dk
        qpr_r = qpr_c.view(B, H, cs, R_dim, N)
        for s in range(cs):
            blk = dqk[:, :, s*R_dim:(s+1)*R_dim, s*R_dim:(s+1)*R_dim]
            q_blk = qpr_r[:, :, s]
            dk_inv[:, :, s*R_dim:(s+1)*R_dim, :] += torch.einsum('bhoi,bhon->bhin', blk, q_blk)

        DK_o[:, :, ci] = dk_inv

        # dq inverse rotary + dangle
        dq_r = dq.view(B, H, cs, R_dim, N)
        dq_f = dq_r[..., :rdim].clone()
        dq_s = dq_r[..., N//2:N//2+rdim].clone()
        qpr_f = qpr_r[..., :rdim]
        qpr_s = qpr_r[..., N//2:N//2+rdim]

        dang_dq = (
            dq_f * (-qpr_f * sin_ac - qpr_s * cos_ac) +
            dq_s * (qpr_f * cos_ac - qpr_s * sin_ac)
        )

        DANG_o[:, :, ci] = (dang_dk + dang_dq).sum(dim=-2)

        dq_inv = dq.clone().view(B, H, cs, R_dim, N)
        dq_inv[..., :rdim] = cos_ac * dq_f + sin_ac * dq_s
        dq_inv[..., N//2:N//2+rdim] = -sin_ac * dq_f + cos_ac * dq_s
        dq_inv = dq_inv.reshape(B, H, fcs, N)

        kpr_r2 = kpr_c.view(B, H, cs, R_dim, N)
        for s in range(cs):
            blk = dqk[:, :, s*R_dim:(s+1)*R_dim, s*R_dim:(s+1)*R_dim]
            k_blk = kpr_r2[:, :, s]
            dq_inv[:, :, s*R_dim:(s+1)*R_dim, :] += torch.einsum('bhoi,bhin->bhon', blk, k_blk)

        DQ_o[:, :, ci] = dq_inv

        # Update dstates
        dstates = dstates * torch.exp(da_cs_sum).unsqueeze(-1).unsqueeze(-1)
        dPhiO_scaled = dPh_c * exp_cs_fcs
        dstates = dstates + torch.matmul(q_c.transpose(-1,-2), dPhiO_scaled)

    # Format outputs
    # DK TileLang shape: (B, S*R, H, N)
    # Our shape: (B, H, nchunks, fcs, N) where fcs = cs*R
    # TileLang stores DK[i_b, fused_chunk_start:fused_chunk_start+fcs, i_h, :]
    # fused_chunk_start = chunk_idx * chunk_size * R
    # So the order is: for each chunk, fcs entries, per head
    # Our (B,H,nc,fcs,N) -> permute to (B,nc,fcs,H,N) -> reshape to (B, S*R, H, N)
    DK_f = DK_o.permute(0, 2, 3, 1, 4).reshape(B, S*R_dim, H, N).to(dtype)
    DQ_f = DQ_o.permute(0, 2, 3, 1, 4).reshape(B, S*R_dim, H, N).to(dtype)
    DV_f = DV_o.permute(0, 2, 3, 1, 4).reshape(B, S, H, P).to(dtype)
    DANG_f = DANG_o.permute(0, 2, 3, 1, 4).reshape(B, S, H, rdim)

    return {
        'DK': DK_f,
        'DQ': DQ_f,
        'DV': DV_f,
        'DMIMO_V': DPsi_acc,
        'DDA_CS': DDA_CS_o.view(B, H, S),
        'DDA_CS_REV': DDA_CS_REV_o.view(B, H, S),
        'DFACTOR': DFACTOR_o.view(B, H, S),
        'DGAMMA_DIAG': DGAMMA_o.view(B, H, S),
        'DSSDA': DSSDA_o,
        'DDA': DDA_o.view(B, H, S),
        'DANGLES': DANG_f,
    }
