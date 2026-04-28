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

    # ----- Newton iterations (streaming chunked) ----------------------------
    # System per chain (size S):  I * dh_t + A_t * dh_{t-1} = -F_t.
    # We never materialise the full (Be, S, V, V) Jacobian: each Newton iter
    # walks the sequence chunk-by-chunk, building only the current chunk's
    # residual + Jacobian, running within-chunk Brent-Kung in place, and
    # propagating ``h_last`` (for the next chunk's residual) and
    # ``delta_last`` (for the next chunk's row-0 absorption).
    #
    # Peak working memory:
    #     residual_chunk + jac_chunk + h_chunk + rhs_chunk + delta_chunk
    #   ~ 3 * Be * C * V (state-like) + Be * C * V * V (Jacobian)
    # For B=4, S=4096, H=44, k=64, V=16, C=128: jac chunk = 1.4 GiB, state
    # chunks ~85 MiB each. The previous full-S path materialised a 47 GiB
    # Jacobian, which OOMs every GPU we ship to.
    #
    # Cross-chunk dependency makes this O(n_chunks + log C) sequential depth
    # rather than O(log S). For S=4096 / C=128 that is 32 + 7 = 39 versus 12
    # full-parallel; the trade is worth the memory saving and is what makes
    # the Triton port (Phase B.2) viable on a single SM's smem budget.
    chunk_size = config.chunk_size if config.chunk_size > 0 else S
    n_chunks = (S + chunk_size - 1) // chunk_size

    kernel = _select_kernel(
        config,
        q_is_cuda=q.is_cuda,
        q_is_fp32=(compute_dtype == torch.float32),
    )
    grad_enabled = torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v, W, xf)
    )
    if kernel == "triton" and grad_enabled:
        # Triton kernels do not have autograd registered; gradient flow would
        # silently break.  Phase C wraps this in an autograd.Function with an
        # IFT backward.  Until then: explicit "triton" raises (so users know
        # they asked for the unsupported config); "auto" falls back to torch.
        env = os.environ.get("CPPMEGA_M2RNN_PARARNN_KERNEL")
        explicit = (env == "triton") or (config.kernel == "triton" and env != "torch")
        if explicit:
            raise RuntimeError(
                "kernel='triton' is only supported under torch.no_grad() — "
                "wrap your call site or use kernel='torch'/'auto' for "
                "differentiable forward"
            )
        kernel = "torch"
    use_triton = kernel == "triton"

    for newton_iter in range(config.max_its):
        prev_h_last = h0_row                                           # (Be, V)
        prev_delta_last = torch.zeros_like(h0_row)                     # (Be, V)
        max_residual = 0.0  # for optional convergence check

        for c_idx in range(n_chunks):
            c_start = c_idx * chunk_size
            c_end = min(c_start + chunk_size, S)
            h_c = h[:, c_start:c_end]                                  # (Be, C, V)
            x_c = x_proj[:, c_start:c_end]
            f_c = f_t[:, c_start:c_end]

            if use_triton:
                # Triton kernel needs contiguous fp32 inputs.  ``h_c``,
                # ``x_c``, ``f_c`` are slices of larger tensors so they
                # need ``.contiguous()`` to satisfy the kernel's stride
                # assumptions.
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
            rhs_c = -residual_c                                        # (Be, C, V)

            if config.abs_tol > 0.0 or config.rel_tol > 0.0:
                # Track max ||F_t||_inf across all chunks for the iter-level
                # convergence check.
                with torch.no_grad():
                    chunk_max = residual_c.abs().amax().item()
                if chunk_max > max_residual:
                    max_residual = chunk_max

            if c_start > 0:
                # Absorb the previous chunk's tail delta into the first row's RHS.
                rhs_c = rhs_c.clone()
                rhs_c[:, 0] = rhs_c[:, 0] - torch.einsum(
                    "bij,bj->bi", jac_c[:, 0], prev_delta_last,
                )
                # Zero the row-0 subdiagonal so the within-chunk scan sees a
                # left boundary of zero (the absorbed delta is already in rhs).
                jac_c = jac_c.clone()
                jac_c[:, 0] = 0
            else:
                # First chunk: row-0 already references h0_row in the residual,
                # so no delta to absorb. But the Brent-Kung scan still needs
                # jac[0] zeroed for safety (matches apple's _reduction_step_dense).
                jac_c = jac_c.clone()
                jac_c[:, 0] = 0

            if use_triton:
                delta_c = _scan_triton(jac_c, rhs_c.contiguous())
            else:
                delta_c = _parallel_reduce_dense(jac_c, rhs_c.contiguous())
            # Update h_traj for this chunk; out-of-place to keep autograd happy.
            h = torch.cat(
                [h[:, :c_start], h_c + config.omega_sor * delta_c, h[:, c_end:]],
                dim=1,
            )

            # Carry forward for the next chunk.
            prev_h_last = h[:, c_end - 1]                              # (Be, V)
            prev_delta_last = delta_c[:, -1]                           # (Be, V)

        # Iteration-level convergence check.
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

    # ----- output projection: out_t = q_t @ h_t (per (b, h, k_idx, t))
    # h is (Be=B*H*k_dim, S, V); q is (B, S, H, k_dim).
    h_btehv = h.view(B, H, k_dim, S, v_dim).permute(0, 3, 1, 2, 4)  # (B, S, H, k_dim, V)
    out = torch.einsum("bshk,bshkv->bshv", qf, h_btehv)  # (B, S, H, V)

    # Final hidden state at t=S-1 in the (B, H, k_dim, V) layout.
    h_final = h_btehv[:, -1].contiguous()  # (B, H, k_dim, V)

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
