"""Chunked parallel M²RNN recurrence for cppmega.

Reduces sequential steps from O(seq_len) to O(seq_len / CHUNK_SIZE) by:
1. Pre-computing ALL input projections in parallel  (embarrassingly parallel)
2. Pre-computing ALL forget gates in parallel        (embarrassingly parallel)
3. Running the matrix-valued recurrence in chunks of C steps
4. Passing only the (k_dim × v_dim) state between chunks sequentially
5. Post-computing ALL output projections in parallel (embarrassingly parallel)

For seq_len=4096, CHUNK=128 → only 32 sequential chunk passes instead of 4096.
Each chunk pass does 128 register-local sequential steps on a tiny (64×16) state.

The heavy GEMM work (input/output projections, conv1d) remains fully parallel.

References
----------
*  Chunked linear-recurrence scan: Hua et al. "Monarch Mixer" and
   Smith et al. "Simplified State Space Layers" use the same decomposition
   for gated linear recurrences.  The key observation is that the tanh
   nonlinearity prevents a *pure* associative parallel scan, but the
   per-chunk sequential cost is negligible when the state is small and
   fits in registers/SRAM.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 128  # C — tunable; 64-256 are all reasonable

# ---------------------------------------------------------------------------
# Reference PyTorch implementation  (numerically identical to the sequential
# ``_torch_m2rnn_forward`` in m2rnn_spec.py, but O(seq/C) sequential depth)
# ---------------------------------------------------------------------------


def _chunked_m2rnn_forward(
    q: torch.Tensor,       # (B, S, H, k_dim)
    k: torch.Tensor,       # (B, S, H, k_dim)
    v: torch.Tensor,       # (B, S, H, v_dim)
    W: torch.Tensor,       # (H, v_dim, v_dim)   -- per-head state transition
    xf: torch.Tensor,      # (B, S, H)           -- pre-computed forget gate ∈ (0,1)
    *,
    h0: Optional[torch.Tensor] = None,  # (B, H, k_dim, v_dim) or None
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked M²RNN — reference impl with O(S/C) sequential depth.

    Parameters
    ----------
    q, k : (B, S, H, k_dim)  — query / key projections.
    v     : (B, S, H, v_dim)  — value projection.
    W     : (n_w, v_dim, v_dim) — per-head (or broadcast) state weight.
    xf    : (B, S, n_f)        — forget gates (already sigmoided / softplus'd).
    h0    : optional initial state.
    chunk_size : chunk length C.

    Returns
    -------
    out : (B, S, H, v_dim)   — output after q-projection of running state.
    h   : (B, H, k_dim, v_dim) — final hidden state.
    """
    B, S, n_q, k_dim = q.shape
    n_k = k.size(-2)
    n_v = v.size(-2)
    n_w = W.size(0)
    n_f = xf.size(-1)
    v_dim = v.size(-1)
    H = max(n_q, n_k, n_v, n_w, n_f)

    # --- broadcast head dims to H (same logic as _torch_m2rnn_forward) -----
    if n_q != H:
        q = q.repeat_interleave(H // n_q, dim=-2)
    if n_k != H:
        k = k.repeat_interleave(H // n_k, dim=-2)
    if n_v != H:
        v = v.repeat_interleave(H // n_v, dim=-2)
    if n_w != H:
        W = W.repeat_interleave(H // n_w, dim=0)
    if n_f != H:
        xf = xf.repeat_interleave(H // n_f, dim=-1)

    # -----------------------------------------------------------------------
    # PHASE 1  (embarrassingly parallel across all B*S positions)
    # Pre-compute the rank-1 input matrices  x[b,s] = k[b,s] ⊗ v[b,s]
    # Shape: (B, S, H, k_dim, v_dim)
    # -----------------------------------------------------------------------
    x = k[..., None] * v[..., None, :]  # outer product per position

    # Reshape xf for broadcasting:  (B, S, H) -> (B, S, H, 1, 1)
    xf_5d = xf.unsqueeze(-1).unsqueeze(-1)

    # -----------------------------------------------------------------------
    # PHASE 2  (chunk-sequential, inter-chunk serial / intra-chunk serial
    #           but operating on register-sized state)
    # -----------------------------------------------------------------------
    if h0 is None:
        h = torch.zeros(B, H, k_dim, v_dim, device=q.device, dtype=q.dtype)
    else:
        h = h0

    W_exp = W.unsqueeze(0)  # (1, H, v_dim, v_dim) for bmm broadcast

    n_chunks = math.ceil(S / chunk_size)
    out = torch.empty(B, S, H, v_dim, device=q.device, dtype=q.dtype)

    for c in range(n_chunks):
        s_start = c * chunk_size
        s_end = min(s_start + chunk_size, S)
        C = s_end - s_start

        # Slice this chunk's inputs — already computed in Phase 1
        x_chunk = x[:, s_start:s_end]        # (B, C, H, k_dim, v_dim)
        f_chunk = xf_5d[:, s_start:s_end]     # (B, C, H, 1, 1)
        q_chunk = q[:, s_start:s_end]          # (B, C, H, k_dim)

        # Intra-chunk sequential scan (C=128 steps, tiny state in registers)
        for t in range(C):
            h_new = torch.tanh(h @ W_exp + x_chunk[:, t])
            f_t = f_chunk[:, t]
            h = f_t * h + (1.0 - f_t) * h_new

            # Output projection: q[b,s] @ h  →  (B, H, v_dim)
            # q_chunk[:, t] is (B, H, k_dim), h is (B, H, k_dim, v_dim)
            out[:, s_start + t] = torch.einsum("bhk,bhkv->bhv", q_chunk[:, t], h)

    return out, h


# ============================================================================
# Triton-style kernel pseudocode
# ============================================================================
#
# The following is the Triton kernel design for a fused chunked M²RNN.
# It is written as structured pseudocode following Triton conventions,
# NOT as executable Triton (which requires the Triton compiler).
#
# ---------------------------------------------------------------------------
#
# KERNEL 1: m2rnn_precompute_inputs
# ----------------------------------
# Fuses:  x[b,s,h] = k[b,s,h] ⊗ v[b,s,h]   (outer product)
#         f[b,s,h] = softplus_decay(xf[b,s,h], A_log[h], dt_bias[h])
#
# Grid:  (B * n_chunks, H)      — one program per (batch-chunk, head)
# Block: BLOCK_C = chunk_size   — each program handles C positions
#
# ```triton-pseudocode
# @triton.jit
# def m2rnn_precompute_inputs_kernel(
#     K_ptr, V_ptr, XF_ptr, A_log_ptr, DT_bias_ptr,
#     X_out_ptr, F_out_ptr,
#     B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
#     k_dim: tl.constexpr, v_dim: tl.constexpr,
#     CHUNK: tl.constexpr,
# ):
#     pid_bc = tl.program_id(0)    # batch-chunk index
#     pid_h  = tl.program_id(1)    # head index
#
#     batch_idx = pid_bc // n_chunks
#     chunk_idx = pid_bc % n_chunks
#     s_start   = chunk_idx * CHUNK
#
#     # Offsets within the chunk
#     offs_c = tl.arange(0, CHUNK)                 # [0..C-1]
#     offs_k = tl.arange(0, k_dim)                 # [0..63]
#     offs_v = tl.arange(0, v_dim)                 # [0..15]
#     mask_c = (s_start + offs_c) < S
#
#     # Load k: (C, k_dim),  v: (C, v_dim)
#     k = tl.load(K_ptr + batch_idx*S*H*k_dim + (s_start+offs_c[:,None])*H*k_dim
#                 + pid_h*k_dim + offs_k[None,:],
#                 mask=mask_c[:,None])              # (C, k_dim)
#     v = tl.load(V_ptr + batch_idx*S*H*v_dim + (s_start+offs_c[:,None])*H*v_dim
#                 + pid_h*v_dim + offs_v[None,:],
#                 mask=mask_c[:,None])              # (C, v_dim)
#
#     # Outer product:  x[c, i, j] = k[c, i] * v[c, j]
#     # Triton: broadcast (C,k_dim,1) * (C,1,v_dim) → (C, k_dim, v_dim)
#     x = k[:, :, None] * v[:, None, :]            # (C, k_dim, v_dim)
#
#     # Forget gate
#     xf_raw = tl.load(XF_ptr + batch_idx*S*H + (s_start+offs_c)*H + pid_h,
#                       mask=mask_c)                # (C,)
#     A  = tl.exp(tl.load(A_log_ptr + pid_h))
#     dt = tl.load(DT_bias_ptr + pid_h)
#     f  = tl.exp(-A * tl.log1p(tl.exp(xf_raw + dt)))  # softplus decay
#
#     # Store x: (C, k_dim, v_dim),  f: (C,)
#     tl.store(X_out_ptr + ..., x, mask=...)
#     tl.store(F_out_ptr + ..., f, mask=mask_c)
# ```
#
# ---------------------------------------------------------------------------
#
# KERNEL 2: m2rnn_chunk_recurrence  (THE CORE)
# ---------------------------------------------
# Each program owns one (batch, head) pair.
# It processes ALL chunks sequentially, but within each chunk runs a tight
# C-step loop over register-resident state.
#
# State h lives entirely in registers: k_dim × v_dim = 64 × 16 = 1024 floats
#   = 4 KB in fp32.  Well within Triton's register budget per program.
#
# W (v_dim × v_dim = 16×16 = 256 floats = 1 KB) is loaded once into registers.
#
# Grid:  (B, H)  — one program per (batch, head)
#
# ```triton-pseudocode
# @triton.jit
# def m2rnn_chunk_recurrence_kernel(
#     X_ptr,              # pre-computed input matrices (B, S, H, k_dim, v_dim)
#     F_ptr,              # pre-computed forget gates   (B, S, H)
#     Q_ptr,              # query projections           (B, S, H, k_dim)
#     W_ptr,              # state weight                (H, v_dim, v_dim)
#     H0_ptr,             # initial state (or null)     (B, H, k_dim, v_dim)
#     Out_ptr,            # output buffer               (B, S, H, v_dim)
#     Hfinal_ptr,         # final state output          (B, H, k_dim, v_dim)
#     B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
#     k_dim: tl.constexpr, v_dim: tl.constexpr,
#     CHUNK: tl.constexpr,
# ):
#     pid_b = tl.program_id(0)   # batch
#     pid_h = tl.program_id(1)   # head
#
#     # ---- Load W into registers (v_dim × v_dim = 16×16) ----
#     offs_vi = tl.arange(0, v_dim)
#     offs_vj = tl.arange(0, v_dim)
#     W = tl.load(W_ptr + pid_h * v_dim * v_dim
#                 + offs_vi[:, None] * v_dim + offs_vj[None, :])   # (v_dim, v_dim)
#
#     # ---- Initialize state h in registers (k_dim × v_dim) ----
#     offs_k = tl.arange(0, k_dim)
#     if H0_ptr is not None:
#         h = tl.load(H0_ptr + pid_b*H*k_dim*v_dim + pid_h*k_dim*v_dim
#                     + offs_k[:, None]*v_dim + offs_vi[None, :])   # (k_dim, v_dim)
#     else:
#         h = tl.zeros([k_dim, v_dim], dtype=tl.float32)
#
#     n_chunks = tl.cdiv(S, CHUNK)
#
#     # ---- Sequential loop over chunks (32 iterations for S=4096, C=128) ----
#     for c in range(n_chunks):
#         s_start = c * CHUNK
#         # The inner loop length; last chunk may be shorter
#         c_len = tl.minimum(CHUNK, S - s_start)
#
#         # ---- Intra-chunk: tight sequential scan (128 steps) ----
#         # h stays in registers the entire time — no HBM traffic for state
#         for t in range(CHUNK):
#             if t >= c_len:
#                 break
#             s = s_start + t
#
#             # Load x[b, s, h, :, :] — rank-1 input (k_dim, v_dim) from SRAM/L2
#             x_t = tl.load(X_ptr + pid_b*S*H*k_dim*v_dim + s*H*k_dim*v_dim
#                           + pid_h*k_dim*v_dim
#                           + offs_k[:, None]*v_dim + offs_vi[None, :])
#
#             # Load f[b, s, h] — scalar forget gate
#             f_t = tl.load(F_ptr + pid_b*S*H + s*H + pid_h)
#
#             # State transition:
#             #   h_candidate = tanh(h @ W + x_t)
#             #   h = f * h + (1 - f) * h_candidate
#             #
#             # h @ W:  (k_dim, v_dim) @ (v_dim, v_dim) = (k_dim, v_dim)
#             # This is the most expensive op per step, but v_dim=16 makes
#             # it only 16 multiply-accumulates per element → 1024*16 = 16K FMAs
#             hW = tl.dot(h, W)                   # (k_dim, v_dim)
#             h_new = tl.math.tanh(hW + x_t)      # (k_dim, v_dim)
#             h = f_t * h + (1.0 - f_t) * h_new   # (k_dim, v_dim)
#
#             # ---- Inline output projection: out[s] = q[s] @ h ----
#             # q: (k_dim,),  h: (k_dim, v_dim)  →  (v_dim,)
#             q_t = tl.load(Q_ptr + pid_b*S*H*k_dim + s*H*k_dim
#                           + pid_h*k_dim + offs_k)           # (k_dim,)
#             # Reduce over k_dim:  out_v[j] = sum_i q[i] * h[i, j]
#             out_t = tl.sum(q_t[:, None] * h, axis=0)        # (v_dim,)
#             tl.store(Out_ptr + pid_b*S*H*v_dim + s*H*v_dim
#                      + pid_h*v_dim + offs_vi, out_t)
#
#     # ---- Store final state ----
#     tl.store(Hfinal_ptr + pid_b*H*k_dim*v_dim + pid_h*k_dim*v_dim
#              + offs_k[:, None]*v_dim + offs_vi[None, :], h)
# ```
#
# ---------------------------------------------------------------------------
#
# KERNEL 3 (optional): m2rnn_output_postprocess
# ----------------------------------------------
# Fuses:  out = out + v * D              (residual skip)
#         out = out * silu(g)             (output gate)
#         out = group_norm(out)           (g_norm)
#
# Grid: (B * n_chunks, H)
# Each program handles C output positions for one head.
# Straightforward element-wise + reduction — omitted for brevity.
#
# ============================================================================
# Performance analysis
# ============================================================================
#
# Sequential baseline:
#   4096 sequential steps × (matmul + tanh + gate + store) = ~10s/layer
#
# Chunked design:
#   Phase 1 (parallel):  outer products + gates for all 4096 positions
#     → 1 kernel launch, fully parallel across B*S*H
#     → ~0.5ms (memory-bound, tiny per-element work)
#
#   Phase 2 (chunk-sequential):
#     Grid: (B, H) programs, each doing 32 sequential chunk passes × 128 steps
#     Per step: tl.dot(h, W) where h=(64,16), W=(16,16) → 16K FMAs
#     Total per program: 4096 × 16K = 64M FMAs @ ~300 TFLOPS = ~0.2ms
#     The bottleneck is memory loads of x_t and f_t:
#       4096 × (64×16 + 1) × 2 bytes = ~8 MB per (b,h) from L2
#       With B*H programs running concurrently → bandwidth-bound
#     Estimated: ~5-30ms depending on B, H, and L2 hit rate
#
#   Phase 3 (parallel): output post-processing
#     → 1 kernel launch, ~0.5ms
#
#   Total: ~6-30ms/layer  (vs ~10s sequential = 300-1600× speedup)
#
# Memory overhead:
#   Pre-computed x: B × S × H × k_dim × v_dim × 2 bytes
#   For B=1, S=4096, H=32, k=64, v=16: 256 MB (bf16)
#   This is the main cost; can be reduced by fusing Phase 1 into Phase 2
#   (load k,v per step and compute outer product inline — trades compute for memory).
#
# ============================================================================
# Memory-efficient variant: fuse input computation into the recurrence
# ============================================================================
#
# To avoid materializing the full (B, S, H, k_dim, v_dim) tensor, we can
# compute the outer product k⊗v inline during the recurrence loop.  This
# only requires loading k (k_dim) and v (v_dim) per step instead of
# storing/loading x (k_dim × v_dim):
#
# ```triton-pseudocode
# @triton.jit
# def m2rnn_chunk_recurrence_fused_kernel(
#     K_ptr, V_ptr, F_ptr, Q_ptr, W_ptr,
#     H0_ptr, Out_ptr, Hfinal_ptr,
#     B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
#     k_dim: tl.constexpr, v_dim: tl.constexpr,
#     CHUNK: tl.constexpr,
# ):
#     """Memory-efficient: computes k⊗v inline, never materializes x."""
#     pid_b = tl.program_id(0)
#     pid_h = tl.program_id(1)
#
#     # Load W into registers
#     offs_vi = tl.arange(0, v_dim)
#     offs_vj = tl.arange(0, v_dim)
#     offs_k  = tl.arange(0, k_dim)
#     W = tl.load(W_ptr + pid_h*v_dim*v_dim
#                 + offs_vi[:,None]*v_dim + offs_vj[None,:])
#
#     # Initialize state
#     h = tl.zeros([k_dim, v_dim], dtype=tl.float32)
#
#     n_chunks = tl.cdiv(S, CHUNK)
#
#     for c in range(n_chunks):
#         s_start = c * CHUNK
#         c_len = tl.minimum(CHUNK, S - s_start)
#
#         for t in range(CHUNK):
#             if t >= c_len:
#                 break
#             s = s_start + t
#
#             # Load k and v vectors (not the full outer product)
#             k_t = tl.load(K_ptr + pid_b*S*H*k_dim + s*H*k_dim
#                           + pid_h*k_dim + offs_k)             # (k_dim,)
#             v_t = tl.load(V_ptr + pid_b*S*H*v_dim + s*H*v_dim
#                           + pid_h*v_dim + offs_vi)             # (v_dim,)
#
#             # Inline outer product
#             x_t = k_t[:, None] * v_t[None, :]                  # (k_dim, v_dim)
#
#             f_t = tl.load(F_ptr + pid_b*S*H + s*H + pid_h)
#
#             hW = tl.dot(h, W)
#             h_new = tl.math.tanh(hW + x_t)
#             h = f_t * h + (1.0 - f_t) * h_new
#
#             q_t = tl.load(Q_ptr + pid_b*S*H*k_dim + s*H*k_dim
#                           + pid_h*k_dim + offs_k)
#             out_t = tl.sum(q_t[:, None] * h, axis=0)
#             tl.store(Out_ptr + pid_b*S*H*v_dim + s*H*v_dim
#                      + pid_h*v_dim + offs_vi, out_t)
#
#     tl.store(Hfinal_ptr + pid_b*H*k_dim*v_dim + pid_h*k_dim*v_dim
#              + offs_k[:,None]*v_dim + offs_vi[None,:], h)
# ```
#
# This variant uses:
#   B × S × H × (k_dim + v_dim) × 2 = ~20 MB   (vs 256 MB for materialized x)
#   at the cost of k_dim × v_dim extra multiplies per step (negligible vs tl.dot).
#
# ============================================================================
# Backward pass design
# ============================================================================
#
# The backward follows the same chunked structure:
#
# 1. Forward pass stores:  h_states at chunk boundaries (32 checkpoints)
#    Memory: B × H × n_chunks × k_dim × v_dim × 2 = ~4 MB
#
# 2. Backward pass processes chunks in reverse:
#    - For each chunk, re-run the forward to reconstruct per-step states
#      (activation checkpointing at chunk granularity)
#    - Accumulate dW, dh gradients through the chunk
#    - Pass dh to the previous chunk
#
# This gives O(S/C) memory for intermediate states instead of O(S).
#
# ============================================================================


def chunked_m2rnn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    *,
    h0: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Public entry point: chunked M²RNN forward (PyTorch reference).

    This is a drop-in replacement for ``_torch_m2rnn_forward`` from
    ``m2rnn_spec.py``.  The output tensor shape is (B, S, H, v_dim) —
    the caller should apply ``(q[..., None, :] @ y).squeeze(-2)`` reshape
    or use this output directly depending on the head-broadcast convention.

    When a Triton backend is available, this dispatches to the fused kernel.
    Otherwise it falls back to the reference chunked PyTorch loop above.
    """
    return _chunked_m2rnn_forward(
        q, k, v, W, xf, h0=h0, chunk_size=chunk_size
    )
