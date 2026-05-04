"""Triton kernels for the M2RNN ParaRNN forward inner loop.

Replaces the per-chunk PyTorch path in :mod:`cppmega.megatron.m2rnn_pararnn`
with two fused kernels:

* :func:`pararnn_residual_jac_chunk_triton` — fuses 14 elementwise / einsum
  ops in ``_residual_jac_chunk`` (h_prev gather, ``h@W``, tanh, residual,
  sech^2, jacobian assembly) into a single launch per chunk.
* :func:`pararnn_reduction_step_triton` — one Brent-Kung step, fusing the
  jac and rhs combine operations into a single launch per scan step.

Both kernels are pure forward primitives; the autograd backward path for
M2RNN remains the existing implicit-function-theorem solver (Phase A) since
the Jacobian we already compute here is what the IFT backward needs.

Layout conventions (matching ``m2rnn_pararnn.py``)
--------------------------------------------------
* ``Be = B * H * k_dim`` independent chains.
* Per-chunk tensors: ``h_chunk``, ``x_chunk`` are ``(Be, C, V)`` fp32;
  ``f_chunk`` is ``(Be, C)`` fp32; ``W_be`` is ``(Be, V, V)`` fp32;
  ``prev_h_last`` is ``(Be, V)`` fp32.
* Jacobian axes are ``(Be, C, i, j)`` where ``i`` is the output index of
  the residual and ``j`` is the input index (``∂F_t,i/∂h_{t-1,j}``). Same
  convention as the dense PyTorch path so the existing
  ``_reduction_step_dense`` keeps its einsum signatures.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    PARARNN_TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover -- triton-less envs
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    PARARNN_TRITON_AVAILABLE = False


if PARARNN_TRITON_AVAILABLE:

    @triton.jit
    def _pararnn_residual_jac_kernel(
        h_chunk_ptr,        # (Be, C, V) fp32
        x_chunk_ptr,        # (Be, C, V) fp32
        f_chunk_ptr,        # (Be, C)    fp32
        W_ptr,              # (Be, V, V) fp32
        prev_h_ptr,         # (Be, V)    fp32
        residual_ptr,       # (Be, C, V) fp32  -- output
        jac_ptr,            # (Be, C, V, V) fp32  -- output
        Be, C,
        h_sb, h_sc, h_sv,
        x_sb, x_sc, x_sv,
        f_sb, f_sc,
        W_sb, W_si, W_sj,
        ph_sb, ph_sv,
        res_sb, res_sc, res_sv,
        jac_sb, jac_sc, jac_si, jac_sj,
        V: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Build residual and Jacobian for one chunk on one chain.

        Grid = (Be, ceil(C / BLOCK_C)).  Each program handles BLOCK_C
        timesteps for one chain.  Produces:

            residual[t, k] = h_chunk[t, k]
                             - f_t * h_prev[t, k]
                             - (1 - f_t) * tanh(z_t)[k]
            jac[t, i, j]   = -f_t * δ_{ij}
                             - (1 - f_t) * sech^2(z_t)[i] * W[j, i]

        with ``h_prev[t]`` = ``prev_h`` if t == 0 (global), else
        ``h_chunk[t-1]``; ``z_t = h_prev[t] @ W + x_chunk[t]``.
        """
        pid_chain = tl.program_id(0)
        pid_cblk = tl.program_id(1)

        offs_c = pid_cblk * BLOCK_C + tl.arange(0, BLOCK_C)  # (BLOCK_C,)
        offs_v = tl.arange(0, V)                              # (V,)
        c_mask = offs_c < C

        # --- W (V, V) -- shared by all timesteps in this program -------
        W = tl.load(
            W_ptr
            + pid_chain * W_sb
            + offs_v[:, None] * W_si
            + offs_v[None, :] * W_sj,
        )  # (V, V) fp32

        # --- chunk loads (BLOCK_C, V) ---------------------------------
        h_chunk = tl.load(
            h_chunk_ptr
            + pid_chain * h_sb
            + offs_c[:, None] * h_sc
            + offs_v[None, :] * h_sv,
            mask=c_mask[:, None],
            other=0.0,
        )
        x_chunk = tl.load(
            x_chunk_ptr
            + pid_chain * x_sb
            + offs_c[:, None] * x_sc
            + offs_v[None, :] * x_sv,
            mask=c_mask[:, None],
            other=0.0,
        )
        f_chunk = tl.load(
            f_chunk_ptr + pid_chain * f_sb + offs_c * f_sc,
            mask=c_mask,
            other=0.0,
        )  # (BLOCK_C,)

        # --- h_prev: row 0 (global) -> prev_h, else h_chunk[t-1] --------
        # The shifted load reads h_chunk[t-1] for global t > 0; for global
        # t == 0 (first chunk's row 0) the result is masked to 0 and we
        # overwrite with prev_h via tl.where.
        h_prev_shift = tl.load(
            h_chunk_ptr
            + pid_chain * h_sb
            + (offs_c[:, None] - 1) * h_sc
            + offs_v[None, :] * h_sv,
            mask=(offs_c[:, None] > 0) & c_mask[:, None],
            other=0.0,
        )
        prev_h = tl.load(
            prev_h_ptr + pid_chain * ph_sb + offs_v * ph_sv,
        )  # (V,)
        h_prev = tl.where(
            offs_c[:, None] == 0,
            prev_h[None, :],
            h_prev_shift,
        )  # (BLOCK_C, V)

        # --- z = h_prev @ W + x_chunk; h_new = tanh(z) -----------------
        # input_precision="ieee" matches the existing m2rnn_triton kernel
        # (no TF32 — both operands are fp32 and tests assume fp32 dot).
        z = tl.dot(h_prev, W, out_dtype=tl.float32, input_precision="ieee")
        z = z + x_chunk  # (BLOCK_C, V)

        # Hardware tanh.approx.f32 — same pattern as m2rnn_triton.
        h_new = tl.inline_asm_elementwise(
            asm="tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[z],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )  # (BLOCK_C, V)

        # --- residual --------------------------------------------------
        f_b = f_chunk[:, None]  # (BLOCK_C, 1)
        residual = h_chunk - f_b * h_prev - (1.0 - f_b) * h_new

        tl.store(
            residual_ptr
            + pid_chain * res_sb
            + offs_c[:, None] * res_sc
            + offs_v[None, :] * res_sv,
            residual,
            mask=c_mask[:, None],
        )

        # --- Jacobian: jac[t, i, j] = -f δ_{ij} - (1-f) sech^2[t,i] W[j,i]
        # Build column-by-column over j to avoid 3D tiles.
        sech2 = 1.0 - h_new * h_new  # (BLOCK_C, V) -- indexed by i

        for j in tl.static_range(V):
            # W[j, :] indexed by i — row j of W
            W_row_j = tl.load(
                W_ptr
                + pid_chain * W_sb
                + j * W_si
                + offs_v * W_sj,
            )  # (V,) indexed by i

            # nonlin[t, i] = sech^2[t, i] * W[j, i]
            nonlin_col = sech2 * W_row_j[None, :]  # (BLOCK_C, V)

            # one_hot[i] = 1 if i == j else 0 (the δ_{ij} term, fixing j=j_const)
            one_hot_i = (offs_v == j).to(tl.float32)  # (V,)

            jac_slice = (
                -f_b * one_hot_i[None, :]
                - (1.0 - f_b) * nonlin_col
            )  # (BLOCK_C, V)

            tl.store(
                jac_ptr
                + pid_chain * jac_sb
                + offs_c[:, None] * jac_sc
                + offs_v[None, :] * jac_si
                + j * jac_sj,
                jac_slice,
                mask=c_mask[:, None],
            )

    @triton.jit
    def _pararnn_reduction_step_kernel(
        jac_in_ptr,     # (Be, C, V, V) fp32 -- input snapshot
        rhs_in_ptr,     # (Be, C, V)    fp32 -- input snapshot
        jac_out_ptr,    # (Be, C, V, V) fp32 -- output buffer
        rhs_out_ptr,    # (Be, C, V)    fp32 -- output buffer
        Be, C,
        offset,         # 2**step
        jac_sb, jac_sc, jac_si, jac_sj,
        rhs_sb, rhs_sc, rhs_sv,
        V: tl.constexpr,
    ):
        """One Brent-Kung step on the bidiagonal block system.

        For every ``t``:
            * t <  offset : copy through       jac_out[t] = jac_in[t], rhs_out[t] = rhs_in[t]
            * t >= offset : combine            rhs_out[t] = rhs_in[t] - jac_in[t] @ rhs_in[t-offset]
                                               jac_out[t] = -jac_in[t] @ jac_in[t-offset]

        Out-of-place to avoid a read-after-write race between programs at
        the same step: program ``t`` reads jac_in[t-offset], which would be
        concurrently written by program ``t-offset`` if both halves wrote
        to the same buffer.  The Python wrapper double-buffers and swaps
        after each step.

        Grid = (Be, C).  ``_reduction_step_dense`` (the torch reference)
        computes the einsum out-of-place into a temporary then assigns,
        which has the same effect.
        """
        pid_chain = tl.program_id(0)
        t_curr = tl.program_id(1)

        offs_v = tl.arange(0, V)

        jac_curr = tl.load(
            jac_in_ptr
            + pid_chain * jac_sb
            + t_curr * jac_sc
            + offs_v[:, None] * jac_si
            + offs_v[None, :] * jac_sj,
        )  # (V, V)
        rhs_curr = tl.load(
            rhs_in_ptr
            + pid_chain * rhs_sb
            + t_curr * rhs_sc
            + offs_v * rhs_sv,
        )  # (V,)

        if t_curr >= offset:
            t_prev = t_curr - offset
            jac_prev = tl.load(
                jac_in_ptr
                + pid_chain * jac_sb
                + t_prev * jac_sc
                + offs_v[:, None] * jac_si
                + offs_v[None, :] * jac_sj,
            )
            rhs_prev = tl.load(
                rhs_in_ptr
                + pid_chain * rhs_sb
                + t_prev * rhs_sc
                + offs_v * rhs_sv,
            )

            # rhs[t] -= jac[t] @ rhs[t-offset]   (V,V) @ (V,)
            delta_rhs = tl.sum(jac_curr * rhs_prev[None, :], axis=1)  # (V,)
            rhs_new = rhs_curr - delta_rhs

            # jac[t] = - jac[t] @ jac[t-offset]   (V,V) @ (V,V)
            jac_new = -tl.dot(
                jac_curr, jac_prev,
                out_dtype=tl.float32,
                input_precision="ieee",
            )  # (V, V)
        else:
            # Below the step's offset: copy through.
            rhs_new = rhs_curr
            jac_new = jac_curr

        tl.store(
            rhs_out_ptr
            + pid_chain * rhs_sb
            + t_curr * rhs_sc
            + offs_v * rhs_sv,
            rhs_new,
        )
        tl.store(
            jac_out_ptr
            + pid_chain * jac_sb
            + t_curr * jac_sc
            + offs_v[:, None] * jac_si
            + offs_v[None, :] * jac_sj,
            jac_new,
        )


def pararnn_residual_jac_chunk_triton(
    h_chunk: torch.Tensor,    # (Be, C, V)
    x_chunk: torch.Tensor,    # (Be, C, V)
    f_chunk: torch.Tensor,    # (Be, C)
    W_be: torch.Tensor,       # (Be, V, V)
    prev_h_last: torch.Tensor,  # (Be, V)
    *,
    block_c: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton replacement for :func:`m2rnn_pararnn._residual_jac_chunk`.

    Returns
    -------
    residual : (Be, C, V) fp32
    jac      : (Be, C, V, V) fp32
    """
    assert PARARNN_TRITON_AVAILABLE, "Triton is required for the pararnn kernel"
    assert h_chunk.is_cuda and h_chunk.dtype == torch.float32, (
        "Triton pararnn kernel needs CUDA fp32 inputs"
    )
    Be, C, V = h_chunk.shape
    assert x_chunk.shape == (Be, C, V)
    assert f_chunk.shape == (Be, C)
    assert W_be.shape == (Be, V, V)
    assert prev_h_last.shape == (Be, V)

    # All operands must be contiguous and fp32 for ``tl.dot`` precision="ieee".
    h_chunk = h_chunk.contiguous()
    x_chunk = x_chunk.contiguous()
    f_chunk = f_chunk.contiguous()
    W_be = W_be.contiguous()
    prev_h_last = prev_h_last.contiguous()

    residual = torch.empty_like(h_chunk)
    jac = torch.empty(Be, C, V, V, dtype=torch.float32, device=h_chunk.device)

    # tl.dot requires K >= 16; V_DIM in our M2RNN configs is always 16.
    assert V >= 16, f"pararnn triton kernel requires V >= 16, got V={V}"

    grid = (Be, triton.cdiv(C, block_c))
    _pararnn_residual_jac_kernel[grid](
        h_chunk, x_chunk, f_chunk, W_be, prev_h_last,
        residual, jac,
        Be, C,
        h_chunk.stride(0), h_chunk.stride(1), h_chunk.stride(2),
        x_chunk.stride(0), x_chunk.stride(1), x_chunk.stride(2),
        f_chunk.stride(0), f_chunk.stride(1),
        W_be.stride(0), W_be.stride(1), W_be.stride(2),
        prev_h_last.stride(0), prev_h_last.stride(1),
        residual.stride(0), residual.stride(1), residual.stride(2),
        jac.stride(0), jac.stride(1), jac.stride(2), jac.stride(3),
        V=V,
        BLOCK_C=block_c,
    )
    return residual, jac


def pararnn_brent_kung_scan_triton(
    jac: torch.Tensor,  # (Be, C, V, V) fp32
    rhs: torch.Tensor,  # (Be, C, V)    fp32
) -> torch.Tensor:
    """Triton replacement for :func:`m2rnn_pararnn._parallel_reduce_dense`.

    Returns a new tensor holding the solution of the bidiagonal block system
    ``I dh_t + jac_t dh_{t-1} = rhs_t``.  The input ``jac`` and ``rhs`` are
    not mutated (we double-buffer to avoid the same-step read-after-write
    race between programs).  Caller is responsible for the row-0 jac
    zeroing — already done by the caller in ``m2rnn_pararnn_forward``.
    """
    assert PARARNN_TRITON_AVAILABLE
    assert jac.is_cuda and jac.dtype == torch.float32
    assert rhs.is_cuda and rhs.dtype == torch.float32
    Be, C, V, V2 = jac.shape
    assert V == V2 == rhs.shape[-1]
    assert rhs.shape[:2] == (Be, C)

    jac_a = jac.contiguous()
    rhs_a = rhs.contiguous()
    jac_b = torch.empty_like(jac_a)
    rhs_b = torch.empty_like(rhs_a)

    num_steps = (C - 1).bit_length()
    for step in range(num_steps):
        offset = 1 << step
        if offset >= C:
            break
        grid = (Be, C)
        _pararnn_reduction_step_kernel[grid](
            jac_a, rhs_a, jac_b, rhs_b,
            Be, C, offset,
            jac_a.stride(0), jac_a.stride(1), jac_a.stride(2), jac_a.stride(3),
            rhs_a.stride(0), rhs_a.stride(1), rhs_a.stride(2),
            V=V,
        )
        jac_a, jac_b = jac_b, jac_a
        rhs_a, rhs_b = rhs_b, rhs_a
    # After the swap, the latest result is in (jac_a, rhs_a).
    return rhs_a
