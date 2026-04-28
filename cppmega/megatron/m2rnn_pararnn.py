"""ParaRNN-style parallel scan for the M2RNN nonlinear gated recurrence.

Adapts the Newton-iteration + parallel-reduction algorithm from
Apple's ParaRNN (arXiv:2510.21450, github.com/apple/ml-pararnn) to the
M2RNN step

    h_t = f_t * h_{t-1} + (1 - f_t) * tanh(h_{t-1} @ W + x_t)

where the state ``h`` has shape ``(k_dim, v_dim)``, ``W`` is
``(v_dim, v_dim)``, ``f_t`` is a scalar per (batch, head), and
``x_t = k_t kron v_t`` is rank-1.

Each (batch, head, k_idx) row of ``h`` evolves independently with a
v_dim-sized state and a (v_dim, v_dim) Jacobian, so we reshape the
problem into ``Be = B * H * k_dim`` independent chains of length S
with N = v_dim, then run Newton iterations whose linearised solve is
a Brent-Kung-style parallel reduction with log2(S) depth.

This module is the Phase-A pure-PyTorch reference. It is intentionally
tensor-correct first, performance second; the Triton kernel comes in
Phase B once numerical match against ``_torch_m2rnn_forward`` and
gradient match are validated.

References
----------
Danieli, Rodriguez, Sarabia, Suau, Zappella -- ParaRNN, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class PararnnConfig:
    """Newton solver knobs.

    max_its
        Newton iteration count. ParaRNN paper uses 3 for stable
        gated RNNs; sech^2 <= 1 in our recurrence keeps the Jacobian
        spectrally bounded by ``f + (1-f) * ||W||``, so 3 is safe
        for the typical NAM56R init.

    omega_sor
        Successive-over-relaxation coefficient. ``< 1`` damps when
        the model is poorly conditioned (early training), ``= 1`` is
        vanilla Newton, ``> 1`` accelerates if the recurrence is
        well behaved. Default vanilla.

    init_strategy
        ``"zero"`` -- start every Newton solve from ``h_t = 0``
            (cheap, paper default).
        ``"chunk"`` -- run a small sequential sweep of length C
            to seed; reduces Newton iterations needed at the cost
            of one O(C) sequential pass.
    """

    max_its: int = 3
    omega_sor: float = 1.0
    init_strategy: str = "zero"
    # Chunk size for the streaming reduction. S > chunk_size triggers the
    # chunked path which never materialises the full (Be, S, V, V) Jacobian.
    # 0 disables chunking (use only when S * Be * V^2 fits comfortably in
    # device memory). 128 keeps peak Jacobian under ~150 MB at NAM56R dims.
    chunk_size: int = 128


# ---------------------------------------------------------------------------
# Brent-Kung parallel reduction over a bidiagonal block system.
#
# System: I * dh_t + A_t * dh_{t-1} = rhs_t,  with A_t shape (N, N).
# After log2(S) steps of pairwise combination, rhs[t] holds the solution.
# Mirrors apple/ml-pararnn _reduction_step_dense.
# ---------------------------------------------------------------------------


def _reduction_step_dense(
    jac: torch.Tensor,  # (..., S, N, N)
    rhs: torch.Tensor,  # (..., S, N)
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = 1 << step  # 2**step
    rhs[..., idx:, :] -= torch.einsum(
        "...tij,...tj->...ti", jac[..., idx:, :, :], rhs[..., :-idx, :]
    )
    jac[..., idx:, :, :] = torch.einsum(
        "...tij,...tjk->...tik",
        -jac[..., idx:, :, :],
        jac[..., :-idx, :, :],
    )
    # Zero the now-unused subdiagonal blocks of the first 2^step rows.
    # First ones must be zeroed because jac[0] can be != 0.
    jac[..., :idx, :, :] = 0
    return jac, rhs


def _parallel_reduce_dense(
    jac: torch.Tensor,  # (Be, S, N, N)
    rhs: torch.Tensor,  # (Be, S, N)
) -> torch.Tensor:
    """Solve the bidiagonal system in O(log S) depth, in place on jac/rhs."""
    num_steps = (rhs.shape[-2] - 1).bit_length()  # ceil(log2(S))
    for step in range(num_steps):
        jac, rhs = _reduction_step_dense(jac, rhs, step)
    return rhs


def _chunked_reduce_dense(
    jac: torch.Tensor,    # (Be, S, V, V)
    rhs: torch.Tensor,    # (Be, S, V)
    chunk_size: int,
) -> torch.Tensor:
    """Streaming chunked reduction: solves the same bidiagonal system as
    ``_parallel_reduce_dense`` but processes the sequence in chunks of
    ``chunk_size`` so peak Jacobian memory is O(Be * chunk * V^2) instead
    of O(Be * S * V^2).

    Algorithm
    ---------
    For each chunk c covering [c*C, (c+1)*C):
      1. The chunk-local first row's subdiagonal block gives the link to
         the previous chunk. We absorb the propagated previous-chunk
         delta into rhs[c*C] via -= jac[c*C] @ prev_delta, then zero
         jac[c*C] so the within-chunk scan treats the chunk as
         starting from zero state.
      2. Run the standard Brent-Kung scan inside the chunk.
      3. The chunk's last delta becomes prev_delta for the next chunk.

    This is sequential across chunks (n_chunks steps) but parallel
    inside each chunk (log2(C) steps), giving O(n_chunks + log C)
    sequential depth versus O(log S) for the full-S scan. The win is
    memory: peak working tensor is one chunk's (Be, C, V, V), not the
    whole (Be, S, V, V).

    Output is exact to the full-S scan (no approximation).
    """
    Be, S, V, _ = jac.shape
    out = torch.empty_like(rhs)
    prev_delta = torch.zeros(Be, V, device=jac.device, dtype=jac.dtype)

    for c_start in range(0, S, chunk_size):
        c_end = min(c_start + chunk_size, S)
        # clone() because the Brent-Kung step mutates in place.
        jac_c = jac[:, c_start:c_end].clone()
        rhs_c = rhs[:, c_start:c_end].clone()

        # Absorb the previous chunk's tail delta into the first row.
        if c_start > 0:
            rhs_c[:, 0] -= torch.einsum("bij,bj->bi", jac_c[:, 0], prev_delta)
        jac_c[:, 0] = 0  # the within-chunk scan now sees a zero left boundary

        delta_c = _parallel_reduce_dense(jac_c, rhs_c)
        out[:, c_start:c_end] = delta_c
        prev_delta = delta_c[:, -1]

    return out


# ---------------------------------------------------------------------------
# m2rnn-specific Newton residual + Jacobian assembly
# ---------------------------------------------------------------------------


def _m2rnn_residual_and_jacobian(
    h_traj: torch.Tensor,  # (Be, S, V)
    x_proj: torch.Tensor,  # (Be, S, V)
    f: torch.Tensor,  # (Be, S)
    W: torch.Tensor,  # (V, V)
    h0: torch.Tensor,  # (Be, V)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute F_t = h_t - f_t * h_{t-1} - (1 - f_t) * tanh(h_{t-1} @ W + x_t)
    and the subdiagonal Jacobian A_t = dF_t / dh_{t-1} for every t.

    Returns
    -------
    residual: (Be, S, V)
    jac:      (Be, S, V, V)
    h_new:    (Be, S, V) -- the candidate value tanh(...), kept so the
                            caller can reuse it in initialisation logic.
    """
    Be, S, V = h_traj.shape

    # h_{t-1} for t=0..S-1; h_{-1} = h0 (or zeros).
    h_prev = torch.cat([h0[:, None, :], h_traj[:, :-1, :]], dim=1)  # (Be, S, V)

    z = h_prev @ W + x_proj  # (Be, S, V)
    h_new = torch.tanh(z)

    f_b = f[..., None]  # (Be, S, 1) -- broadcast over V
    residual = h_traj - f_b * h_prev - (1.0 - f_b) * h_new  # (Be, S, V)

    # A_t[i, j] = -f_t * delta_ij - (1 - f_t) * sech^2(z_i) * W[j, i]
    #          = -f_t * I - (1 - f_t) * diag(sech^2(z)) @ W^T
    sech2 = 1.0 - h_new * h_new  # (Be, S, V) -- 1 - tanh^2 = sech^2
    eye_v = torch.eye(V, device=W.device, dtype=W.dtype)
    f_bb = f[..., None, None]  # (Be, S, 1, 1)

    # outer: sech2[..., i, None] * W.T[None, None, i, j] gives (Be, S, V, V)
    # broadcasting carefully:
    #   sech2 reshape (Be, S, V, 1)  *  W.T reshape (1, 1, V, V) -> (Be, S, V, V)
    # producing entry [i, j] = sech2_i * W.T[i, j] = sech2_i * W[j, i].
    nonlin_block = sech2[..., :, None] * W.t()[None, None, :, :]  # (Be, S, V, V)

    jac = -f_bb * eye_v[None, None, :, :] - (1.0 - f_bb) * nonlin_block

    return residual, jac, h_new


# ---------------------------------------------------------------------------
# Public reference forward
# ---------------------------------------------------------------------------


def m2rnn_pararnn_forward(
    q: torch.Tensor,   # (B, S, n_q, k_dim)
    k: torch.Tensor,   # (B, S, n_k, k_dim)
    v: torch.Tensor,   # (B, S, n_v, v_dim)
    W: torch.Tensor,   # (n_w, v_dim, v_dim)
    xf: torch.Tensor,  # (B, S, n_f) -- pre-sigmoided / pre-decayed forget gate
    *,
    h0: Optional[torch.Tensor] = None,  # (B, H, k_dim, v_dim)
    config: PararnnConfig = PararnnConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Newton + parallel-scan forward. Drop-in shape-compatible with
    ``_torch_m2rnn_forward`` from ``m2rnn_spec.py``.

    Returns
    -------
    out: (B, S, H, v_dim)
    h_final: (B, H, k_dim, v_dim)
    """
    B, S, n_q, k_dim = q.shape
    n_k = k.size(-2)
    n_v = v.size(-2)
    n_w = W.size(0)
    n_f = xf.size(-1)
    v_dim = v.size(-1)
    H = max(n_q, n_k, n_v, n_w, n_f)

    # Broadcast head dims to the union -- same convention as the reference.
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

    # Newton/scan math runs in at least fp32: we follow ParaRNN's CUDA
    # wrappers, which up-cast jac+rhs to fp32 for bf16/fp16 inputs. fp64
    # callers (tests, debugging) keep fp64 -- otherwise we'd cap precision
    # at the fp32 floor (~1e-7), defeating the convergence guarantee.
    compute_dtype = torch.promote_types(torch.float32, q.dtype)
    out_dtype = q.dtype

    qf = q.to(compute_dtype)
    kf = k.to(compute_dtype)
    vf = v.to(compute_dtype)
    Wf = W.to(compute_dtype)
    xff = xf.to(compute_dtype)

    Be = B * H * k_dim

    # ----- pre-compute x_proj = k * v  for every (b, s, h, k_idx)
    # m2rnn step: h_new = tanh(h_{t-1} W + x_t) where x_t[k_idx, :] = k_t[k_idx] * v_t[:]
    # -> for fixed (b, h, k_idx), x_proj[b, h, k_idx, s, v] = k[b, s, h, k_idx] * v[b, s, h, v].
    # Permute to (Be, S, V).
    x_proj = (
        kf[..., :, None] * vf[..., None, :]
    )  # (B, S, H, k_dim, v_dim)
    x_proj = x_proj.permute(0, 2, 3, 1, 4).reshape(Be, S, v_dim).contiguous()

    # ----- broadcast forget to per-(b, h, k_idx, t)
    # xf is (B, S, H); each k_idx of a head shares the same gate.
    f_t = (
        xff.permute(0, 2, 1)  # (B, H, S)
        .unsqueeze(2)         # (B, H, 1, S)
        .expand(B, H, k_dim, S)
        .reshape(Be, S)
        .contiguous()
    )

    # ----- per-row weight matrix; W is (H, V, V). For chain (b, h, k_idx)
    # the weight is W[h] regardless of k_idx, so flatten to (Be, V, V).
    W_be = (
        Wf.unsqueeze(0)        # (1, H, V, V)
        .unsqueeze(2)          # (1, H, 1, V, V)
        .expand(B, H, k_dim, v_dim, v_dim)
        .reshape(Be, v_dim, v_dim)
        .contiguous()
    )

    # ----- initial h0 broadcast to per-row state (Be, V)
    if h0 is None:
        h0_row = torch.zeros(Be, v_dim, device=q.device, dtype=compute_dtype)
    else:
        # h0: (B, H, k_dim, v_dim) -> (Be, V)
        h0_row = h0.to(compute_dtype).reshape(Be, v_dim).contiguous()

    # ----- initial guess for the trajectory
    if config.init_strategy == "zero":
        h = torch.zeros(Be, S, v_dim, device=q.device, dtype=compute_dtype)
    elif config.init_strategy == "chunk":
        # cheap sequential sweep, useful when Newton needs a warm start.
        h = torch.empty(Be, S, v_dim, device=q.device, dtype=compute_dtype)
        h_cur = h0_row
        for t in range(S):
            z = torch.einsum("bv,bvj->bj", h_cur, W_be) + x_proj[:, t]
            h_new = torch.tanh(z)
            f_bcast = f_t[:, t, None]
            h_cur = f_bcast * h_cur + (1.0 - f_bcast) * h_new
            h[:, t] = h_cur
    else:
        raise ValueError(f"unknown init_strategy: {config.init_strategy}")

    # ----- Newton iterations -----
    # System per chain (size S):  I * dh_t + A_t * dh_{t-1} = -F_t,  dh_0 driven by h0_row.
    # Each Newton step assembles A_t and -F_t from the current trajectory and
    # runs _parallel_reduce_dense to recover dh in O(log S) depth.
    use_chunked = config.chunk_size > 0 and S > config.chunk_size
    for _ in range(config.max_its):
        residual, jac, _h_new = _m2rnn_residual_and_jacobian_be(
            h_traj=h,
            x_proj=x_proj,
            f=f_t,
            W_be=W_be,
            h0_row=h0_row,
        )
        rhs = -residual  # (Be, S, V)
        if use_chunked:
            delta = _chunked_reduce_dense(jac, rhs, config.chunk_size)
        else:
            delta = _parallel_reduce_dense(jac.contiguous(), rhs.contiguous())
        h = h + config.omega_sor * delta

    # ----- output projection: out_t = q_t @ h_t (per (b, h, k_idx, t))
    # h is (Be=B*H*k_dim, S, V); q is (B, S, H, k_dim).
    h_btehv = h.view(B, H, k_dim, S, v_dim).permute(0, 3, 1, 2, 4)  # (B, S, H, k_dim, V)
    out = torch.einsum("bshk,bshkv->bshv", qf, h_btehv)  # (B, S, H, V)

    # Final hidden state at t=S-1 in the (B, H, k_dim, V) layout.
    h_final = h_btehv[:, -1].contiguous()  # (B, H, k_dim, V)

    return out.to(out_dtype), h_final.to(out_dtype)


def _m2rnn_residual_and_jacobian_be(
    h_traj: torch.Tensor,   # (Be, S, V)
    x_proj: torch.Tensor,   # (Be, S, V)
    f: torch.Tensor,        # (Be, S)
    W_be: torch.Tensor,     # (Be, V, V) -- per-row weight (heads broadcast)
    h0_row: torch.Tensor,   # (Be, V)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row variant: each chain has its own W (so we batch-mm)."""
    Be, S, V = h_traj.shape

    h_prev = torch.cat([h0_row[:, None, :], h_traj[:, :-1, :]], dim=1)  # (Be, S, V)

    # z[t] = h_{t-1} @ W_be   per row (Be, V) @ (Be, V, V) -> (Be, V), all t.
    z = torch.einsum("btv,bvw->btw", h_prev, W_be) + x_proj  # (Be, S, V)
    h_new = torch.tanh(z)

    f_b = f[..., None]
    residual = h_traj - f_b * h_prev - (1.0 - f_b) * h_new

    sech2 = 1.0 - h_new * h_new                    # (Be, S, V)
    eye_v = torch.eye(V, device=W_be.device, dtype=W_be.dtype)
    f_bb = f[..., None, None]                      # (Be, S, 1, 1)

    # Per-chain W^T: (Be, V, V)
    Wt_be = W_be.transpose(-1, -2)                 # (Be, V, V)
    # nonlin_block[t, i, j] = sech2[t, i] * Wt_be[i, j] = sech2[t, i] * W[j, i]
    nonlin_block = sech2[..., :, None] * Wt_be[:, None, :, :]  # (Be, S, V, V)

    jac = -f_bb * eye_v[None, None, :, :] - (1.0 - f_bb) * nonlin_block
    return residual, jac, h_new
