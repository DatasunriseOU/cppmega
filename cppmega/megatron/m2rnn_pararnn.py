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

import os
from dataclasses import dataclass
from typing import Optional

import torch

try:
    from cppmega.megatron.m2rnn_pararnn_triton import (
        PARARNN_TRITON_AVAILABLE as _PARARNN_TRITON_AVAILABLE,
        pararnn_brent_kung_scan_triton as _scan_triton,
        pararnn_residual_jac_chunk_triton as _residual_jac_triton,
    )
except ImportError:  # pragma: no cover -- triton-less envs
    _PARARNN_TRITON_AVAILABLE = False
    _residual_jac_triton = None  # type: ignore[assignment]
    _scan_triton = None  # type: ignore[assignment]


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

    # ParaRNN paper uses 3 for stable LSTM/GRU; with the M2RNN identity-ish
    # init (W ~ I + small perturbation) Newton needs more iterations to
    # drive the residual to the fp32 floor. Empirically (S=128, k=V=16,
    # W = eye + noise*randn, fp64):
    #   noise=0.001: iter 6 -> 1.18e-5,  iter 8 -> 9e-16  (converges)
    #   noise=0.010: iter 6 -> 1.54e-3,  iter 8 -> 2e-11   (converges, slower)
    #   noise=0.050: iter 6 -> 2.64,     iter 10 -> 4.2e-1 (poorly conditioned)
    # Default 8 covers realistic M2RNN init (noise ~ 0.01 from the residual
    # scheme); poorly-conditioned regimes (noise ~ 0.05) need omega_sor < 1
    # or explicit abs_tol/rel_tol convergence flags. ParaRNN's stable-LSTM
    # default of 3 is wrong for identity-ish recurrences.
    max_its: int = 8
    omega_sor: float = 1.0
    init_strategy: str = "zero"
    # Chunk size for the streaming Newton/reduce path. Each Newton iteration
    # processes the sequence chunk-by-chunk, building the (Be, chunk_size, V, V)
    # Jacobian on the fly so peak Jacobian memory is O(Be * chunk_size * V^2)
    # rather than O(Be * S * V^2). 0 disables chunking and rebuilds the
    # full-S Jacobian per iteration -- only safe when Be * S * V^2 * 4 bytes
    # fits comfortably in device memory.
    chunk_size: int = 128
    # Optional residual-norm convergence checks. ``abs_tol > 0`` triggers
    # an extra forward pass per Newton iteration; the iteration breaks as
    # soon as max_t ||F_t||_inf < abs_tol or that norm divided by the
    # iteration-0 norm < rel_tol. Default disabled -- callers fall back
    # to ``max_its``.
    abs_tol: float = 0.0
    rel_tol: float = 0.0
    # Inner-loop kernel:
    #   "auto"   -- use Triton kernels when CUDA + fp32 + Triton is
    #               available, else fall back to ``torch``.  Selected by
    #               default; can be overridden via the
    #               ``CPPMEGA_M2RNN_PARARNN_KERNEL`` env var.
    #   "torch"  -- pure PyTorch path (slower, but works on CPU and for
    #               fp64 reference parity tests).
    #   "triton" -- force the Triton path; raises if unavailable.
    kernel: str = "auto"
    # BLOCK_C tile for the Triton residual+Jacobian kernel.  Lower values
    # reduce per-program register pressure on small smem GPUs; the default
    # 16 keeps the (BLOCK_C, V, V) intermediate at ~16 KiB for V=16.
    triton_block_c: int = 16


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




# ---------------------------------------------------------------------------
# m2rnn-specific Newton residual + Jacobian assembly
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Kernel selection
# ---------------------------------------------------------------------------


def _select_kernel(config: "PararnnConfig", q_is_cuda: bool, q_is_fp32: bool) -> str:
    """Pick the inner-loop kernel.  ``CPPMEGA_M2RNN_PARARNN_KERNEL`` overrides
    ``config.kernel``; ``"auto"`` resolves to ``triton`` only when CUDA + fp32
    + the Triton extension is importable.

    The Triton kernels are fp32-only — autograd wrapping plus dtype dispatch
    is Phase C work.  fp64 callers (parity tests, debugging) always get the
    torch path.
    """
    env = os.environ.get("CPPMEGA_M2RNN_PARARNN_KERNEL")
    choice = env if env else config.kernel
    if choice == "torch":
        return "torch"
    if choice == "triton":
        if not _PARARNN_TRITON_AVAILABLE:
            raise RuntimeError(
                "kernel='triton' requested but cppmega.megatron.m2rnn_pararnn_triton "
                "is not importable (Triton missing?)"
            )
        if not q_is_cuda:
            raise RuntimeError("kernel='triton' requires CUDA tensors")
        if not q_is_fp32:
            raise RuntimeError(
                "kernel='triton' currently only supports fp32 compute; got non-fp32 "
                "(use kernel='auto' or 'torch' for bf16/fp16/fp64 callers)"
            )
        return "triton"
    if choice == "auto":
        if (
            _PARARNN_TRITON_AVAILABLE
            and q_is_cuda
            and q_is_fp32
            and torch.cuda.is_available()
        ):
            return "triton"
        return "torch"
    raise ValueError(f"unknown kernel choice: {choice}")


# ---------------------------------------------------------------------------
# Inner solver (works on already-broadcast Be-shaped tensors)
# ---------------------------------------------------------------------------


def _solve_fixed_point(
    x_proj: torch.Tensor,   # (Be, S, V)
    f_t: torch.Tensor,      # (Be, S)
    W_be: torch.Tensor,     # (Be, V, V)
    h0_row: torch.Tensor,   # (Be, V)
    *,
    config: PararnnConfig,
    use_triton: bool,
) -> torch.Tensor:
    """Streaming chunked Newton solve. Returns ``h`` of shape (Be, S, V).

    The caller is responsible for kernel selection and for running this under
    ``torch.no_grad()`` when wrapping in an autograd.Function.
    """
    Be, S, V = x_proj.shape
    chunk_size = config.chunk_size if config.chunk_size > 0 else S
    n_chunks = (S + chunk_size - 1) // chunk_size

    if config.init_strategy == "zero":
        h = torch.zeros(Be, S, V, device=x_proj.device, dtype=x_proj.dtype)
    elif config.init_strategy == "chunk":
        h = torch.empty(Be, S, V, device=x_proj.device, dtype=x_proj.dtype)
        h_cur = h0_row
        for t in range(S):
            z = torch.einsum("bv,bvj->bj", h_cur, W_be) + x_proj[:, t]
            h_new = torch.tanh(z)
            f_bcast = f_t[:, t, None]
            h_cur = f_bcast * h_cur + (1.0 - f_bcast) * h_new
            h[:, t] = h_cur
    else:
        raise ValueError(f"unknown init_strategy: {config.init_strategy}")

    iter0_residual = 0.0
    for newton_iter in range(config.max_its):
        prev_h_last = h0_row
        prev_delta_last = torch.zeros_like(h0_row)
        max_residual = 0.0

        for c_idx in range(n_chunks):
            c_start = c_idx * chunk_size
            c_end = min(c_start + chunk_size, S)
            h_c = h[:, c_start:c_end]
            x_c = x_proj[:, c_start:c_end]
            f_c = f_t[:, c_start:c_end]

            if use_triton:
                residual_c, jac_c = _residual_jac_triton(
                    h_c.contiguous(), x_c.contiguous(), f_c.contiguous(),
                    W_be, prev_h_last.contiguous(),
                    block_c=config.triton_block_c,
                )
            else:
                residual_c, jac_c = _residual_jac_chunk(
                    h_chunk=h_c, x_chunk=x_c, f_chunk=f_c,
                    W_be=W_be, prev_h_last=prev_h_last,
                )
            rhs_c = -residual_c

            if config.abs_tol > 0.0 or config.rel_tol > 0.0:
                with torch.no_grad():
                    chunk_max = residual_c.abs().amax().item()
                if chunk_max > max_residual:
                    max_residual = chunk_max

            if c_start > 0:
                rhs_c = rhs_c.clone()
                rhs_c[:, 0] = rhs_c[:, 0] - torch.einsum(
                    "bij,bj->bi", jac_c[:, 0], prev_delta_last,
                )
                jac_c = jac_c.clone()
                jac_c[:, 0] = 0
            else:
                jac_c = jac_c.clone()
                jac_c[:, 0] = 0

            if use_triton:
                delta_c = _scan_triton(jac_c, rhs_c.contiguous())
            else:
                delta_c = _parallel_reduce_dense(jac_c, rhs_c.contiguous())
            h = torch.cat(
                [h[:, :c_start], h_c + config.omega_sor * delta_c, h[:, c_end:]],
                dim=1,
            )

            prev_h_last = h[:, c_end - 1]
            prev_delta_last = delta_c[:, -1]

        if config.abs_tol > 0.0 and max_residual < config.abs_tol:
            break
        if newton_iter == 0:
            iter0_residual = max_residual
        elif (
            config.rel_tol > 0.0
            and iter0_residual > 0.0
            and max_residual / iter0_residual < config.rel_tol
        ):
            break

    return h


# ---------------------------------------------------------------------------
# Adjoint solve for the IFT backward
# ---------------------------------------------------------------------------


def _solve_adjoint_chunked(
    grad_h_star: torch.Tensor,  # (Be, S, V)
    h_star: torch.Tensor,       # (Be, S, V) -- detached
    x_proj: torch.Tensor,       # (Be, S, V)
    f_t: torch.Tensor,          # (Be, S)
    W_be: torch.Tensor,         # (Be, V, V)
    h0_row: torch.Tensor,       # (Be, V)
    *,
    config: PararnnConfig,
    use_triton: bool,
) -> torch.Tensor:
    """Adjoint solve for the IFT backward.

    Solves ``(∂F/∂h)^T λ = -grad_h_star`` where ``∂F/∂h`` is the lower
    bidiagonal block matrix produced by the same residual+Jacobian kernel
    used in the forward Newton iteration. The transpose is upper bidiagonal,
    so we time-reverse + transpose-jac and reuse the existing Brent-Kung
    scan, walking chunks right-to-left.
    """
    Be, S, V = h_star.shape
    chunk_size = config.chunk_size if config.chunk_size > 0 else S
    n_chunks = (S + chunk_size - 1) // chunk_size

    lam = torch.empty_like(h_star)

    # next_lam_first: λ at the first time step of the chunk to the right (i.e.
    # the chunk we just processed). For the last chunk it's zero.
    next_lam_first = torch.zeros_like(h0_row)
    # next_jac_first_T: jac[c_end]^T from the next chunk's first row, used to
    # absorb cross-chunk coupling λ[t] = ... - jac[t+1]^T λ[t+1] at the
    # boundary. For the last chunk it's zero (no row to its right).
    next_jac_first_T = torch.zeros(Be, V, V, device=h_star.device, dtype=h_star.dtype)

    for c_idx in range(n_chunks - 1, -1, -1):
        c_start = c_idx * chunk_size
        c_end = min(c_start + chunk_size, S)
        C = c_end - c_start

        # prev_h_last for this chunk: h_star[c_start - 1] or h0_row.
        if c_start > 0:
            prev_h_last_c = h_star[:, c_start - 1].contiguous()
        else:
            prev_h_last_c = h0_row

        h_c = h_star[:, c_start:c_end].contiguous()
        x_c = x_proj[:, c_start:c_end].contiguous()
        f_c = f_t[:, c_start:c_end].contiguous()

        if use_triton:
            _, jac_c = _residual_jac_triton(
                h_c, x_c, f_c, W_be, prev_h_last_c,
                block_c=config.triton_block_c,
            )
        else:
            _, jac_c = _residual_jac_chunk(
                h_chunk=h_c, x_chunk=x_c, f_chunk=f_c,
                W_be=W_be, prev_h_last=prev_h_last_c,
            )

        # Build the time-reversed transposed Jacobian system.
        # Forward chunk: row t has identity on diagonal and jac[t] on the
        # subdiagonal (column t-1). Transposed: row t has identity on
        # diagonal and jac[t]^T on the superdiagonal (column t+1).
        # After reversing time within the chunk (rev_t = C-1 - t), the
        # superdiagonal becomes a subdiagonal, so the same Brent-Kung scan
        # applies. The reversed-system "subdiagonal" at position rev_t is
        # jac[rev_t + 1]^T (i.e. one step to the right in original indexing).
        jac_T = jac_c.transpose(-1, -2)  # (Be, C, V, V)
        # Build the reversed-jac tensor: for rev_t in [0, C-1], the
        # subdiagonal block is the original jac at position (C-1 - rev_t) + 1
        # = C - rev_t, transposed. For rev_t == 0 (last original timestep)
        # there is no original t+1 within the chunk; this corresponds to the
        # boundary with the chunk to the right and is handled via
        # ``next_jac_first_T`` absorption below.
        jac_rev = torch.empty_like(jac_T)
        # rev_t = 0 corresponds to the last original timestep; its
        # superdiagonal in the original system would be jac[C]^T (out of chunk
        # bounds, lives in the next chunk). We zero this row so the scan sees
        # a clean left boundary, then fold the cross-chunk coupling into rhs.
        jac_rev[:, 0] = 0
        if C > 1:
            # rev_t in [1, C-1] -> original t = C-1 - rev_t, "next" original
            # t+1 = C - rev_t. We need jac[C - rev_t]^T for rev_t in [1, C-1],
            # i.e. jac_T at indices [C-1, C-2, ..., 1] -> flip jac_T[:, 1:C].
            jac_rev[:, 1:C] = torch.flip(jac_T[:, 1:C], dims=[1])

        # rhs in the reversed frame: rhs[rev_t] = -grad_h_star[c_start + (C-1-rev_t)].
        rhs_rev = torch.flip(-grad_h_star[:, c_start:c_end], dims=[1]).contiguous()

        # Cross-chunk coupling: at rev_t = 0 (= original t = c_end - 1) the
        # superdiagonal is jac[c_end]^T · λ[c_end], which lives in the next
        # chunk. Subtract it from the rhs at rev_t = 0.
        if c_idx < n_chunks - 1:
            rhs_rev = rhs_rev.clone()
            rhs_rev[:, 0] = rhs_rev[:, 0] - torch.einsum(
                "bij,bj->bi", next_jac_first_T, next_lam_first,
            )

        if use_triton:
            lam_rev = _scan_triton(jac_rev, rhs_rev.contiguous())
        else:
            lam_rev = _parallel_reduce_dense(jac_rev, rhs_rev.contiguous())

        # Un-reverse and store.
        lam_c = torch.flip(lam_rev, dims=[1])
        lam[:, c_start:c_end] = lam_c

        # Carry to the next (left-of-current) chunk.
        next_lam_first = lam_c[:, 0].contiguous()
        next_jac_first_T = jac_T[:, 0].contiguous()

    return lam


# ---------------------------------------------------------------------------
# Autograd Function with IFT backward
# ---------------------------------------------------------------------------


class _M2RNNPararnnFn(torch.autograd.Function):
    """``m2rnn_pararnn_forward`` with an implicit-function-theorem backward.

    Forward runs the existing Newton/scan solver under ``torch.no_grad()``;
    backward solves the adjoint system ``(∂F/∂h)^T λ = -grad_h*`` using the
    same Brent-Kung primitive (time-reversed, jac-transposed) and computes
    parameter VJPs by differentiating the residual ``F`` once via
    ``torch.autograd.grad``.
    """

    @staticmethod
    def forward(ctx, q, k, v, W, xf, h0, config_obj):
        B, S, n_q, k_dim = q.shape
        n_k = k.size(-2)
        n_v = v.size(-2)
        n_w = W.size(0)
        n_f = xf.size(-1)
        v_dim = v.size(-1)
        H = max(n_q, n_k, n_v, n_w, n_f)

        # Broadcast head dims to the union.
        q_b = q.repeat_interleave(H // n_q, dim=-2) if n_q != H else q
        k_b = k.repeat_interleave(H // n_k, dim=-2) if n_k != H else k
        v_b = v.repeat_interleave(H // n_v, dim=-2) if n_v != H else v
        W_b = W.repeat_interleave(H // n_w, dim=0) if n_w != H else W
        xf_b = xf.repeat_interleave(H // n_f, dim=-1) if n_f != H else xf

        compute_dtype = torch.promote_types(torch.float32, q.dtype)
        out_dtype = q.dtype

        qf = q_b.to(compute_dtype)
        kf = k_b.to(compute_dtype)
        vf = v_b.to(compute_dtype)
        Wf = W_b.to(compute_dtype)
        xff = xf_b.to(compute_dtype)

        Be = B * H * k_dim

        x_proj = (kf[..., :, None] * vf[..., None, :])
        x_proj = x_proj.permute(0, 2, 3, 1, 4).reshape(Be, S, v_dim).contiguous()
        f_t = (
            xff.permute(0, 2, 1)
            .unsqueeze(2)
            .expand(B, H, k_dim, S)
            .reshape(Be, S)
            .contiguous()
        )
        W_be = (
            Wf.unsqueeze(0).unsqueeze(2)
            .expand(B, H, k_dim, v_dim, v_dim)
            .reshape(Be, v_dim, v_dim)
            .contiguous()
        )
        if h0 is None:
            h0_row = torch.zeros(Be, v_dim, device=q.device, dtype=compute_dtype)
        else:
            h0_row = h0.to(compute_dtype).reshape(Be, v_dim).contiguous()

        kernel = _select_kernel(
            config_obj,
            q_is_cuda=q.is_cuda,
            q_is_fp32=(compute_dtype == torch.float32),
        )
        use_triton = kernel == "triton"

        with torch.no_grad():
            h_star = _solve_fixed_point(
                x_proj=x_proj, f_t=f_t, W_be=W_be, h0_row=h0_row,
                config=config_obj, use_triton=use_triton,
            )

        h_btehv = h_star.view(B, H, k_dim, S, v_dim).permute(0, 3, 1, 2, 4)
        out = torch.einsum("bshk,bshkv->bshv", qf, h_btehv)
        h_final = h_btehv[:, -1].contiguous()

        # Save for backward.
        ctx.save_for_backward(q, k, v, W, xf, h0 if h0 is not None else torch.empty(0),
                              h_star, qf)
        ctx.h0_is_none = h0 is None
        ctx.shapes = (B, S, n_q, n_k, n_v, n_w, n_f, H, k_dim, v_dim, Be)
        ctx.dtypes = (compute_dtype, out_dtype)
        ctx.config = config_obj
        ctx.use_triton = use_triton

        return out.to(out_dtype), h_final.to(out_dtype)

    @staticmethod
    def backward(ctx, grad_out, grad_h_final):
        q, k, v, W, xf, h0_saved, h_star, qf = ctx.saved_tensors
        h0 = None if ctx.h0_is_none else h0_saved
        B, S, n_q, n_k, n_v, n_w, n_f, H, k_dim, v_dim = ctx.shapes[:10]
        Be = ctx.shapes[10]
        compute_dtype, _ = ctx.dtypes
        config_obj = ctx.config
        use_triton = ctx.use_triton

        grad_out_f = grad_out.to(compute_dtype)
        grad_h_final_f = (
            grad_h_final.to(compute_dtype)
            if grad_h_final is not None else None
        )

        # ----- output projection backward ------------------------------------
        # out[b,s,h,v] = sum_k qf[b,s,h,k] * h_star_view[b,s,h,k,v]
        # grad_qf_full[b,s,h,k] = sum_v grad_out[b,s,h,v] * h_star_view[b,s,h,k,v]
        # grad_h*_full[b,s,h,k,v] = grad_out[b,s,h,v] * qf[b,s,h,k]
        h_star_view = h_star.view(B, H, k_dim, S, v_dim).permute(0, 3, 1, 2, 4)
        # (B, S, H, k_dim, V)
        grad_qf_full = torch.einsum("bshv,bshkv->bshk", grad_out_f, h_star_view)
        grad_h_star_full = torch.einsum(
            "bshv,bshk->bshkv", grad_out_f, qf,
        )  # (B, S, H, k_dim, V)
        if grad_h_final_f is not None:
            # grad_h_final has shape (B, H, k_dim, V); add to t = S-1.
            grad_h_star_full[:, -1] = grad_h_star_full[:, -1] + grad_h_final_f.to(
                compute_dtype
            )
        # Reshape to (Be, S, V) matching h_star.
        # h_star_view permutation was (0, 3, 1, 2, 4); inverse gets us back to
        # (B, H, k_dim, S, V) -> (Be, S, V).
        grad_h_star = (
            grad_h_star_full.permute(0, 2, 3, 1, 4)  # (B, H, k_dim, S, V)
            .reshape(Be, S, v_dim).contiguous()
        )

        # ----- recompute the broadcast inputs we need for adjoint solve ------
        q_b = q.repeat_interleave(H // n_q, dim=-2) if n_q != H else q
        k_b = k.repeat_interleave(H // n_k, dim=-2) if n_k != H else k
        v_b = v.repeat_interleave(H // n_v, dim=-2) if n_v != H else v
        W_b = W.repeat_interleave(H // n_w, dim=0) if n_w != H else W
        xf_b = xf.repeat_interleave(H // n_f, dim=-1) if n_f != H else xf

        with torch.no_grad():
            kf_d = k_b.to(compute_dtype)
            vf_d = v_b.to(compute_dtype)
            Wf_d = W_b.to(compute_dtype)
            xff_d = xf_b.to(compute_dtype)
            x_proj = (kf_d[..., :, None] * vf_d[..., None, :])
            x_proj = x_proj.permute(0, 2, 3, 1, 4).reshape(Be, S, v_dim).contiguous()
            f_t = (
                xff_d.permute(0, 2, 1).unsqueeze(2)
                .expand(B, H, k_dim, S).reshape(Be, S).contiguous()
            )
            W_be = (
                Wf_d.unsqueeze(0).unsqueeze(2)
                .expand(B, H, k_dim, v_dim, v_dim)
                .reshape(Be, v_dim, v_dim).contiguous()
            )
            if h0 is None:
                h0_row = torch.zeros(Be, v_dim, device=q.device, dtype=compute_dtype)
            else:
                h0_row = h0.to(compute_dtype).reshape(Be, v_dim).contiguous()

            lam = _solve_adjoint_chunked(
                grad_h_star=grad_h_star, h_star=h_star.detach(),
                x_proj=x_proj, f_t=f_t, W_be=W_be, h0_row=h0_row,
                config=config_obj, use_triton=use_triton,
            )

        # ----- parameter VJPs via autograd.grad against F --------------------
        # Build F = h_star - f * h_prev - (1-f) * tanh(h_prev @ W + x_proj)
        # with the parameters as differentiable inputs.
        with torch.enable_grad():
            k_var = k.detach().to(compute_dtype).requires_grad_(True)
            v_var = v.detach().to(compute_dtype).requires_grad_(True)
            W_var = W.detach().to(compute_dtype).requires_grad_(True)
            xf_var = xf.detach().to(compute_dtype).requires_grad_(True)
            if h0 is None:
                h0_var = None
            else:
                h0_var = h0.detach().to(compute_dtype).requires_grad_(True)

            # Broadcast within the autograd graph.
            kb = k_var.repeat_interleave(H // n_k, dim=-2) if n_k != H else k_var
            vb = v_var.repeat_interleave(H // n_v, dim=-2) if n_v != H else v_var
            Wb = W_var.repeat_interleave(H // n_w, dim=0) if n_w != H else W_var
            xfb = xf_var.repeat_interleave(H // n_f, dim=-1) if n_f != H else xf_var

            x_proj_g = (kb[..., :, None] * vb[..., None, :])
            x_proj_g = x_proj_g.permute(0, 2, 3, 1, 4).reshape(Be, S, v_dim)
            f_t_g = (
                xfb.permute(0, 2, 1).unsqueeze(2)
                .expand(B, H, k_dim, S).reshape(Be, S)
            )
            W_be_g = (
                Wb.unsqueeze(0).unsqueeze(2)
                .expand(B, H, k_dim, v_dim, v_dim)
                .reshape(Be, v_dim, v_dim)
            )
            if h0_var is None:
                h0_row_g = torch.zeros(Be, v_dim, device=q.device, dtype=compute_dtype)
            else:
                h0_row_g = h0_var.reshape(Be, v_dim)

            h_star_d = h_star.detach()
            h_prev_g = torch.cat([h0_row_g[:, None, :], h_star_d[:, :-1, :]], dim=1)
            z_g = torch.einsum("btv,bvw->btw", h_prev_g, W_be_g) + x_proj_g
            h_new_g = torch.tanh(z_g)
            f_b_g = f_t_g[..., None]
            F = h_star_d - f_b_g * h_prev_g - (1.0 - f_b_g) * h_new_g

            inputs = [k_var, v_var, W_var, xf_var]
            if h0_var is not None:
                inputs.append(h0_var)
            grads = torch.autograd.grad(
                F, inputs, grad_outputs=lam, allow_unused=False, retain_graph=False,
            )

        grad_k_b = grads[0]
        grad_v_b = grads[1]
        grad_W_b = grads[2]
        grad_xf_b = grads[3]
        grad_h0_b = grads[4] if h0_var is not None else None

        # ----- grad_q (no IFT, direct from output projection) ----------------
        # grad_qf_full lives at the broadcast shape (B, S, H, k_dim); reduce to
        # the original (B, S, n_q, k_dim) by undoing repeat_interleave.
        grad_q_b = grad_qf_full

        def _undo_repeat_interleave(g, n_orig, full_dim):
            """Undo ``repeat_interleave(H // n_orig, dim=full_dim)`` by summing
            over the inner repeat groups.
            """
            if n_orig == H:
                return g
            r = H // n_orig
            shape = list(g.shape)
            assert shape[full_dim] == H
            new_shape = shape[:full_dim] + [n_orig, r] + shape[full_dim + 1:]
            return g.view(new_shape).sum(dim=full_dim + 1)

        # grad_k/v/W/xf come from torch.autograd.grad against the un-broadcast
        # variables -- broadcast reduction is already absorbed by autograd.
        grad_q = _undo_repeat_interleave(grad_q_b, n_q, full_dim=2)
        grad_k = grad_k_b
        grad_v = grad_v_b
        grad_W = grad_W_b
        grad_xf = grad_xf_b

        grad_q = grad_q.to(q.dtype)
        grad_k = grad_k.to(k.dtype)
        grad_v = grad_v.to(v.dtype)
        grad_W = grad_W.to(W.dtype)
        grad_xf = grad_xf.to(xf.dtype)
        grad_h0 = grad_h0_b.to(h0.dtype) if grad_h0_b is not None else None

        return grad_q, grad_k, grad_v, grad_W, grad_xf, grad_h0, None


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

    Routes through :class:`_M2RNNPararnnFn` (autograd Function with an IFT
    backward) whenever any input has ``requires_grad`` and grad is enabled;
    otherwise runs the solver directly under ``no_grad``.

    Returns
    -------
    out: (B, S, H, v_dim)
    h_final: (B, H, k_dim, v_dim)
    """
    grad_enabled = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (q, k, v, W, xf, h0)
    )
    if grad_enabled:
        return _M2RNNPararnnFn.apply(q, k, v, W, xf, h0, config)

    # No-grad path: run solver directly without the autograd machinery.
    B, S, n_q, k_dim = q.shape
    n_k = k.size(-2)
    n_v = v.size(-2)
    n_w = W.size(0)
    n_f = xf.size(-1)
    v_dim = v.size(-1)
    H = max(n_q, n_k, n_v, n_w, n_f)

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

    compute_dtype = torch.promote_types(torch.float32, q.dtype)
    out_dtype = q.dtype

    qf = q.to(compute_dtype)
    kf = k.to(compute_dtype)
    vf = v.to(compute_dtype)
    Wf = W.to(compute_dtype)
    xff = xf.to(compute_dtype)

    Be = B * H * k_dim

    x_proj = (kf[..., :, None] * vf[..., None, :])
    x_proj = x_proj.permute(0, 2, 3, 1, 4).reshape(Be, S, v_dim).contiguous()
    f_t = (
        xff.permute(0, 2, 1).unsqueeze(2)
        .expand(B, H, k_dim, S).reshape(Be, S).contiguous()
    )
    W_be = (
        Wf.unsqueeze(0).unsqueeze(2)
        .expand(B, H, k_dim, v_dim, v_dim)
        .reshape(Be, v_dim, v_dim).contiguous()
    )
    if h0 is None:
        h0_row = torch.zeros(Be, v_dim, device=q.device, dtype=compute_dtype)
    else:
        h0_row = h0.to(compute_dtype).reshape(Be, v_dim).contiguous()

    kernel = _select_kernel(
        config,
        q_is_cuda=q.is_cuda,
        q_is_fp32=(compute_dtype == torch.float32),
    )
    use_triton = kernel == "triton"

    h = _solve_fixed_point(
        x_proj=x_proj, f_t=f_t, W_be=W_be, h0_row=h0_row,
        config=config, use_triton=use_triton,
    )

    h_btehv = h.view(B, H, k_dim, S, v_dim).permute(0, 3, 1, 2, 4)
    out = torch.einsum("bshk,bshkv->bshv", qf, h_btehv)
    h_final = h_btehv[:, -1].contiguous()

    return out.to(out_dtype), h_final.to(out_dtype)


def _residual_jac_chunk(
    h_chunk: torch.Tensor,    # (Be, C, V) -- current Newton iterate over chunk
    x_chunk: torch.Tensor,    # (Be, C, V)
    f_chunk: torch.Tensor,    # (Be, C)
    W_be: torch.Tensor,       # (Be, V, V) -- per-chain weight
    prev_h_last: torch.Tensor,  # (Be, V) -- h_{c_start - 1} from previous chunk
                                #            (or h0_row for the first chunk)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-chunk Newton residual + subdiagonal Jacobian.

    Identical math to the full-S assembly, but operates on one chunk at a
    time given the previous chunk's tail state. This is the streaming
    primitive that lets us run Newton without materialising the full
    (Be, S, V, V) Jacobian.

    Returns
    -------
    residual: (Be, C, V)
    jac:      (Be, C, V, V)
    """
    Be, C, V = h_chunk.shape

    # h_{t-1} for t in [c_start, c_end): chunk's own [:-1] preceded by prev_h_last.
    h_prev = torch.cat(
        [prev_h_last[:, None, :], h_chunk[:, :-1, :]], dim=1,
    )                                                              # (Be, C, V)

    # z[t] = h_{t-1} @ W_be (per-chain bmm)
    z = torch.einsum("btv,bvw->btw", h_prev, W_be) + x_chunk       # (Be, C, V)
    h_new = torch.tanh(z)

    f_b = f_chunk[..., None]                                       # (Be, C, 1)
    residual = h_chunk - f_b * h_prev - (1.0 - f_b) * h_new        # (Be, C, V)

    # A_t[i, j] = -f_t * delta_ij - (1 - f_t) * sech^2(z_i) * W[j, i]
    #          = -f_t * I - (1 - f_t) * diag(sech^2(z)) @ W^T
    sech2 = 1.0 - h_new * h_new                                    # (Be, C, V)
    eye_v = torch.eye(V, device=W_be.device, dtype=W_be.dtype)
    f_bb = f_chunk[..., None, None]                                # (Be, C, 1, 1)

    Wt_be = W_be.transpose(-1, -2)                                 # (Be, V, V)
    nonlin_block = sech2[..., :, None] * Wt_be[:, None, :, :]      # (Be, C, V, V)

    jac = -f_bb * eye_v[None, None, :, :] - (1.0 - f_bb) * nonlin_block
    return residual, jac
