"""CuTe DSL MIMO forward kernel for Hopper (sm_90a).

Phase 1: PyTorch reference implementation matching TileLang's mamba_mimo_fwd
algorithm exactly. This serves as the correctness baseline for the CuTe DSL
WGMMA kernel port (Phase 2, task #72).

The algorithm implements the chunked Mamba-3 MIMO forward scan:
  - Per-chunk QK matmul with bias + rotary embeddings
  - Trapezoidal discretization + causal masking
  - Inter-chunk contribution via carried recurrent state
  - Intra-chunk contribution via masked QK @ PsiV
  - Diagonal (same-step) QK·PsiV contribution
  - Optional D skip connection, Z gating, and MIMO output projection
  - Recurrent state update for next chunk

All matmuls use torch.matmul (which lowers to WGMMA on H200 via cuBLAS).
"""

import math
import torch
from torch import Tensor
from typing import Optional, Tuple


def cute_dsl_mimo_fwd(
    q: Tensor,          # [B, S, R, G, N]
    k: Tensor,          # [B, S, R, G, N]
    v: Tensor,          # [B, S, H, P]
    q_bias: Tensor,     # [H, R, N] fp32
    k_bias: Tensor,     # [H, R, N] fp32
    mimo_v: Tensor,     # [H, R, P] fp32  (Psi)
    mimo_o: Optional[Tensor],  # [H, R, P] fp32 (Phi) or None
    z: Optional[Tensor],       # [B, S, H, P]
    D: Optional[Tensor],       # [H] fp32
    mimo_z: Optional[Tensor],  # [H, R, P] fp32 (Zeta)
    angles: Tensor,     # [B, S, H, N//(rotary_dim_divisor)] fp32
    dA_cs: Tensor,      # [B, H, S] fp32
    dA_cs_rev: Tensor,  # [B, H, S] fp32
    dt: Tensor,         # [B, H, S] fp32
    trap: Tensor,       # [B, H, S]
    segsum: Tensor,     # [B, H, nchunks, C, C] fp32
    chunk_size: int,
    rotary_dim_divisor: int,
    dtype: torch.dtype,
    return_state: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Forward pass matching TileLang's mamba_mimo_fwd algorithm.

    Returns (output, final_state, final_k) where final_state and final_k
    are None unless return_state=True.
    """
    B, S, R, G, N = q.shape
    H = v.shape[2]
    P = v.shape[3]
    nchunks = math.ceil(S / chunk_size)
    tail_len = S % chunk_size
    fused_chunk_size = chunk_size * R
    reduceO = mimo_o is not None
    hasZ = z is not None
    hasD = D is not None

    device = q.device

    # Output tensor
    if reduceO:
        out = torch.empty(B, S, H, P, device=device, dtype=dtype)
    else:
        out = torch.empty(B, S, R, H, P, device=device, dtype=dtype)

    final_state = torch.empty(B, H, N, P, device=device, dtype=torch.float32) if return_state else None
    final_k = torch.empty(B, R, H, N, device=device, dtype=dtype) if return_state else None

    # Process each (batch, head) pair
    for i_b in range(B):
        for i_h in range(H):
            i_h_qk = i_h // (H // G)

            # Running recurrent state [N, P] in fp32
            states = torch.zeros(N, P, device=device, dtype=torch.float32)

            # Per-head Psi and Phi
            psi = mimo_v[i_h]  # [R, P] fp32
            phi = mimo_o[i_h] if reduceO else None  # [R, P] fp32

            # Per-head biases
            qb = q_bias[i_h]  # [R, N] fp32
            kb = k_bias[i_h]  # [R, N] fp32

            for i_chunk in range(nchunks):
                cs = i_chunk * chunk_size
                ce = min(cs + chunk_size, S)
                clen = ce - cs

                # --- Load Q, K, V for this chunk ---
                # q_chunk: [clen, R, N], k_chunk: [clen, R, N]
                q_chunk = q[i_b, cs:ce, :, i_h_qk, :].to(torch.float32)  # [clen, R, N]
                k_chunk = k[i_b, cs:ce, :, i_h_qk, :].to(torch.float32)  # [clen, R, N]
                v_chunk = v[i_b, cs:ce, i_h, :].to(torch.float32)        # [clen, P]

                # Add biases
                q_chunk = q_chunk + qb.unsqueeze(0)  # broadcast [1, R, N]
                k_chunk = k_chunk + kb.unsqueeze(0)

                # --- Compute trap_scale (trapezoidal discretization) ---
                # shifted gamma: dt[t+1] * sigmoid(-trap[t+1])
                if cs + chunk_size < S:
                    trap_shifted = trap[i_b, i_h, cs+1:cs+chunk_size+1].float()
                    dt_shifted = dt[i_b, i_h, cs+1:cs+chunk_size+1].float()
                else:
                    # Last chunk: pad with zeros
                    trap_shifted = torch.zeros(chunk_size, device=device, dtype=torch.float32)
                    dt_shifted = torch.zeros(chunk_size, device=device, dtype=torch.float32)
                    valid = min(chunk_size, S - 1 - cs)
                    if valid > 0:
                        trap_shifted[:valid] = trap[i_b, i_h, cs+1:cs+1+valid].float()
                        dt_shifted[:valid] = dt[i_b, i_h, cs+1:cs+1+valid].float()

                shifted_gamma = torch.zeros(chunk_size, device=device, dtype=torch.float32)
                for t in range(chunk_size):
                    if cs + t < S - 1:
                        shifted_gamma[t] = dt_shifted[t] * torch.sigmoid(-trap_shifted[t])

                trap_frag = trap[i_b, i_h, cs:cs+chunk_size].float()
                dt_frag = dt[i_b, i_h, cs:cs+chunk_size].float()
                gamma = dt_frag[:clen] * torch.sigmoid(trap_frag[:clen])
                trap_scale = gamma + shifted_gamma[:clen]

                # --- QK dot product for diagonal terms ---
                # Reshape to [clen*R, N] for matmul
                q_flat = q_chunk.reshape(clen * R, N)  # [clen*R, N]
                k_flat = k_chunk.reshape(clen * R, N)  # [clen*R, N]

                # Full QK dot: [clen*R, clen*R]
                qk_dot_full = q_flat @ k_flat.T

                # --- PsiV: up-project V with Psi ---
                # psi_v[t, r, p] = v[t, p] * psi[r, p]
                psi_v = v_chunk.unsqueeze(1) * psi.unsqueeze(0)  # [clen, R, P]
                psi_v_flat = psi_v.reshape(clen * R, P)  # [clen*R, P]

                # --- Rotary embeddings on Q ---
                angles_chunk = angles[i_b, cs:ce, i_h, :]  # [clen, N//rotary_dim_divisor]
                half_n = N // rotary_dim_divisor

                # Apply rotary to Q
                q_first = q_flat.reshape(clen, R, N)[:, :, :half_n]   # [clen, R, half_n]
                q_second = q_flat.reshape(clen, R, N)[:, :, N//2:N//2+half_n]  # [clen, R, half_n]
                cos_a = torch.cos(angles_chunk).unsqueeze(1)  # [clen, 1, half_n]
                sin_a = torch.sin(angles_chunk).unsqueeze(1)  # [clen, 1, half_n]

                q_rot = q_flat.reshape(clen, R, N).clone()
                q_rot[:, :, :half_n] = cos_a * q_first - sin_a * q_second
                q_rot[:, :, N//2:N//2+half_n] = sin_a * q_first + cos_a * q_second
                q_rot_flat = q_rot.reshape(clen * R, N)

                # --- Inter-chunk contribution: Q_rot @ states ---
                # o_interchunk: [clen*R, P]
                o_mimo = (q_rot_flat @ states.to(torch.float32))  # [clen*R, P]

                # --- Rotary embeddings on K ---
                k_first = k_flat.reshape(clen, R, N)[:, :, :half_n]
                k_second = k_flat.reshape(clen, R, N)[:, :, N//2:N//2+half_n]

                k_rot = k_flat.reshape(clen, R, N).clone()
                k_rot[:, :, :half_n] = cos_a * k_first - sin_a * k_second
                k_rot[:, :, N//2:N//2+half_n] = sin_a * k_first + cos_a * k_second
                k_rot_flat = k_rot.reshape(clen * R, N)

                # Save final_k if needed
                if return_state and i_chunk == nchunks - 1:
                    last_t = clen - 1
                    for r in range(R):
                        final_k[i_b, r, i_h, :] = k_rot_flat[last_t * R + r].to(dtype)

                # --- Trap-scale K for intrachunk ---
                k_trap = k_rot_flat.clone()
                for t in range(clen):
                    for r in range(R):
                        k_trap[t * R + r] *= trap_scale[t]

                # --- Intrachunk QK matmul ---
                qk_intra = q_rot_flat @ k_trap.T  # [clen*R, clen*R]

                # --- Causal mask + segsum ---
                da_cs_chunk = dA_cs[i_b, i_h, cs:ce]  # [clen]
                segsum_chunk = segsum[i_b, i_h, i_chunk, :clen, :clen]  # [clen, clen]

                qk_masked = torch.zeros_like(qk_intra)
                for i_row in range(clen * R):
                    for j_col in range(clen * R):
                        t_i = i_row // R
                        t_j = j_col // R
                        if t_i > t_j:  # Strictly causal (exclude diagonal)
                            qk_masked[i_row, j_col] = qk_intra[i_row, j_col] * torch.exp(segsum_chunk[t_i, t_j])

                # Scale inter-chunk by exp(dA_cs)
                exp_da_cs = torch.exp(da_cs_chunk)  # [clen]
                for t in range(clen):
                    for r in range(R):
                        o_mimo[t * R + r] *= exp_da_cs[t]

                # Add intrachunk contribution
                o_mimo = o_mimo + qk_masked @ psi_v_flat  # [clen*R, P]

                # --- Diagonal terms: qk_dot * psi_v ---
                diag_contrib = torch.zeros(clen, R, P, device=device, dtype=torch.float32)
                for t in range(clen):
                    for r_out in range(R):
                        for r_in in range(R):
                            diag_contrib[t, r_out] += qk_dot_full[t*R+r_out, t*R+r_in] * psi_v_flat[t*R+r_in]
                        diag_contrib[t, r_out] *= gamma[t]

                # Add D skip connection
                if hasD:
                    D_val = D[i_h].item()
                    for t in range(clen):
                        for r_out in range(R):
                            diag_contrib[t, r_out] += D_val * psi_v_flat[t*R+r_out]

                o_mimo = o_mimo + diag_contrib.reshape(clen * R, P)

                # --- Z gating + output projection ---
                if reduceO:
                    o_reshaped = o_mimo.reshape(clen, R, P)
                    if hasZ:
                        z_chunk = z[i_b, cs:ce, i_h, :].float()  # [clen, P]
                        zeta = mimo_z[i_h]  # [R, P] fp32
                        z_exp = z_chunk.unsqueeze(1) * zeta.unsqueeze(0) * 0.5  # [clen, R, P]
                        z_gate = z_exp * torch.tanh(z_exp) + z_exp  # SiLU-ish
                        o_reshaped = o_reshaped * phi.unsqueeze(0) * z_gate
                    else:
                        o_reshaped = o_reshaped * phi.unsqueeze(0)
                    # Sum over R dimension
                    o_out = o_reshaped.sum(dim=1)  # [clen, P]
                    out[i_b, cs:ce, i_h, :] = o_out.to(dtype)
                else:
                    o_reshaped = o_mimo.reshape(clen, R, P)
                    if hasZ:
                        z_chunk = z[i_b, cs:ce, i_h, :].float()
                        zeta = mimo_z[i_h]
                        z_exp = z_chunk.unsqueeze(1) * zeta.unsqueeze(0) * 0.5
                        z_gate = z_exp * torch.tanh(z_exp) + z_exp
                        o_reshaped = o_reshaped * z_gate
                    out[i_b, cs:ce, :, i_h, :] = o_reshaped.to(dtype)

                # --- State update ---
                dA_cs_rev_chunk = dA_cs_rev[i_b, i_h, cs:ce]  # [clen]
                k_state = k_rot_flat.clone()
                for t in range(clen):
                    for r in range(R):
                        k_state[t*R+r] *= torch.exp(dA_cs_rev_chunk[t])

                # Zero out beyond tail for last chunk
                if tail_len > 0 and i_chunk == nchunks - 1 and return_state:
                    for t in range(clen):
                        if t >= tail_len:
                            k_state[t*R:(t+1)*R] = 0.0

                # Chunk-level decay
                if tail_len > 0 and i_chunk == nchunks - 1:
                    da_cs_last = dA_cs[i_b, i_h, S-1]
                else:
                    da_cs_last = dA_cs[i_b, i_h, cs + chunk_size - 1]
                states *= torch.exp(da_cs_last)

                # State += K_state^T @ PsiV
                states += k_state.T @ psi_v_flat  # [N, P]

            # Save final state
            if return_state:
                final_state[i_b, i_h] = states

    return out, final_state, final_k
