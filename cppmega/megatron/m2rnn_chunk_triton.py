"""Chunked forward Triton kernel for M2RNN scan.

Replaces the single persistent ``for s in range(SEQ)`` loop in
``_m2rnn_fwd_kernel`` with an outer loop over forward chunks, reducing
register pressure per unrolled iteration and improving instruction-cache
behaviour on very long sequences.

The per-step math is identical to the existing persistent kernel.  The
backward path reuses the existing ``_m2rnn_recompute_chunk_kernel`` and
``_m2rnn_bwd_chunk_kernel`` from ``m2rnn_triton.py`` -- the checkpoint
format (saved every ``BWD_CHUNK_SIZE`` steps) is unchanged.

Shapes (matching ``_torch_m2rnn_forward`` after head broadcast):
    q:  (B, S, H, K)
    k:  (B, S, H, K)
    v:  (B, S, H, V)
    W:  (H, V, V)
    xf: (B, S, H)
    out: (B, S, H, V)
    h_final: (B, H, K, V)

Usage:
    from cppmega.megatron.m2rnn_chunk_triton import m2rnn_scan_triton_chunked
    out, h_final = m2rnn_scan_triton_chunked(q, k, v, W, xf, h0=None)
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from cppmega.megatron.m2rnn_triton import (
    TRITON_AVAILABLE,
    _broadcast_heads,
    _m2rnn_bwd_chunk_kernel,
    _m2rnn_recompute_chunk_kernel,
    _unbroadcast_heads,
    get_m2rnn_runtime_config,
)

_FWD_CHUNK_SIZE_ENV = "CPPMEGA_M2RNN_FWD_CHUNK_SIZE"
_DEFAULT_FWD_CHUNK_SIZE = 128


def _get_fwd_chunk_size() -> int:
    """Read CPPMEGA_M2RNN_FWD_CHUNK_SIZE from the environment."""
    raw = os.environ.get(_FWD_CHUNK_SIZE_ENV)
    if raw is None:
        return _DEFAULT_FWD_CHUNK_SIZE
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_FWD_CHUNK_SIZE


if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _m2rnn_fwd_chunk_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        W_ptr,
        xf_ptr,
        h0_ptr,
        ckpt_ptr,
        hnew_ptr,
        out_ptr,
        hfinal_ptr,
        HAS_H0: tl.constexpr,
        SAVE_HNEW: tl.constexpr,
        BWD_CHUNK_SIZE: tl.constexpr,
        FWD_CHUNK_SIZE: tl.constexpr,
        SEQ: tl.constexpr,
        NHEADS: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        # input strides are all in elements
        q_sb, q_ss, q_sh, q_sk,
        k_sb, k_ss, k_sh, k_sk,
        v_sb, v_ss, v_sh, v_sv,
        W_sh, W_sr, W_sc,
        xf_sb, xf_ss, xf_sh,
        h0_sb, h0_sh, h0_sk, h0_sv,
        ckpt_sb, ckpt_sc, ckpt_sh, ckpt_sk, ckpt_sv,
        hn_sb, hn_ss, hn_sh, hn_sk, hn_sv,
        out_sb, out_ss, out_sh, out_sv,
        hf_sb, hf_sh, hf_sk, hf_sv,
    ):
        """One program per (batch, head) pair.  Chunked forward.

        Maintains ``h`` (K_DIM x V_DIM) in registers.  Loops over forward
        chunks; within each chunk runs a tight sequential loop.  Saves
        backward checkpoints at the same boundaries as the persistent
        kernel so the existing backward path can be reused unchanged.

        Per-step math (identical to ``_m2rnn_fwd_kernel``):
            x_s   = k_s[:, None] * v_s[None, :]       # (K, V) outer product
            h_new = tanh(h @ W + x_s)                  # (K, V)
            h     = xf_s * h + (1 - xf_s) * h_new      # gated state
            out_s = sum_k(q_s[:, None] * h, axis=0)    # (V,)
        """
        b = tl.program_id(0)
        h_idx = tl.program_id(1)

        offs_k = tl.arange(0, K_DIM)
        offs_v = tl.arange(0, V_DIM)

        # Load W (V_DIM x V_DIM) once, cast to fp32.
        W_row = tl.arange(0, V_DIM)
        W_col = tl.arange(0, V_DIM)
        W = tl.load(
            W_ptr + h_idx * W_sh + W_row[:, None] * W_sr + W_col[None, :] * W_sc
        ).to(tl.float32)

        # Initialize h.
        if HAS_H0:
            h = tl.load(
                h0_ptr
                + b * h0_sb
                + h_idx * h0_sh
                + offs_k[:, None] * h0_sk
                + offs_v[None, :] * h0_sv,
            ).to(tl.float32)
        else:
            h = tl.zeros((K_DIM, V_DIM), dtype=tl.float32)

        # Checkpoint 0 is the state before the first token.
        tl.store(
            ckpt_ptr
            + b * ckpt_sb
            + 0 * ckpt_sc
            + h_idx * ckpt_sh
            + offs_k[:, None] * ckpt_sk
            + offs_v[None, :] * ckpt_sv,
            h,
        )

        n_fwd_chunks = (SEQ + FWD_CHUNK_SIZE - 1) // FWD_CHUNK_SIZE

        for c in range(n_fwd_chunks):
            start = c * FWD_CHUNK_SIZE
            # chunk_len for the last chunk may be shorter
            remaining = SEQ - start
            chunk_len = tl.minimum(FWD_CHUNK_SIZE, remaining)

            for t in range(FWD_CHUNK_SIZE):
                if t < chunk_len:
                    s = start + t

                    # Load per-step inputs
                    q_s = tl.load(
                        q_ptr + b * q_sb + s * q_ss + h_idx * q_sh + offs_k * q_sk,
                    ).to(tl.float32)
                    k_s = tl.load(
                        k_ptr + b * k_sb + s * k_ss + h_idx * k_sh + offs_k * k_sk,
                    ).to(tl.float32)
                    v_s = tl.load(
                        v_ptr + b * v_sb + s * v_ss + h_idx * v_sh + offs_v * v_sv,
                    ).to(tl.float32)
                    xf_s = tl.load(
                        xf_ptr + b * xf_sb + s * xf_ss + h_idx * xf_sh,
                    ).to(tl.float32)

                    # Rank-1 outer product x = k (x) v : (K_DIM, V_DIM)
                    x = k_s[:, None] * v_s[None, :]

                    # h @ W : (K_DIM, V_DIM) @ (V_DIM, V_DIM) -> (K_DIM, V_DIM)
                    hW = tl.dot(h, W, out_dtype=tl.float32, input_precision="ieee")

                    pre = hW + x

                    # Hardware-accelerated tanh via inline PTX
                    h_new = tl.inline_asm_elementwise(
                        asm="tanh.approx.f32 $0, $1;",
                        constraints="=f,f",
                        args=[pre],
                        dtype=tl.float32,
                        is_pure=True,
                        pack=1,
                    )

                    # Optionally store h_new for the backward kernel.
                    if SAVE_HNEW:
                        tl.store(
                            hnew_ptr
                            + b * hn_sb
                            + s * hn_ss
                            + h_idx * hn_sh
                            + offs_k[:, None] * hn_sk
                            + offs_v[None, :] * hn_sv,
                            h_new,
                        )

                    h = xf_s * h + (1.0 - xf_s) * h_new

                    # Store sparse checkpoint at backward-chunk boundaries
                    # and at the final position.
                    if ((s + 1) % BWD_CHUNK_SIZE == 0) or (s == SEQ - 1):
                        ckpt_idx = (s + 1 + BWD_CHUNK_SIZE - 1) // BWD_CHUNK_SIZE
                        tl.store(
                            ckpt_ptr
                            + b * ckpt_sb
                            + ckpt_idx * ckpt_sc
                            + h_idx * ckpt_sh
                            + offs_k[:, None] * ckpt_sk
                            + offs_v[None, :] * ckpt_sv,
                            h,
                        )

                    # out_s = q_s @ h  (reduce over k): (V_DIM,)
                    out_s = tl.sum(q_s[:, None] * h, axis=0)
                    tl.store(
                        out_ptr
                        + b * out_sb
                        + s * out_ss
                        + h_idx * out_sh
                        + offs_v * out_sv,
                        out_s,
                    )

        # Store final h
        tl.store(
            hfinal_ptr
            + b * hf_sb
            + h_idx * hf_sh
            + offs_k[:, None] * hf_sk
            + offs_v[None, :] * hf_sv,
            h,
        )


# ---------------------------------------------------------------------------
# Autograd function -- reuses the existing backward kernels
# ---------------------------------------------------------------------------


class _M2RNNChunkedFn(torch.autograd.Function):
    """Autograd Function for the chunked M2RNN forward.

    Forward uses the chunked kernel; backward reuses the existing
    checkpointed backward kernels from ``m2rnn_triton.py``.
    """

    @staticmethod
    def forward(ctx, q, k, v, W, xf, h0):
        assert TRITON_AVAILABLE, "Triton is required for m2rnn_scan_triton_chunked"
        assert q.is_cuda, "m2rnn_scan_triton_chunked requires CUDA tensors"

        q_b, k_b, v_b, W_b, xf_b, H = _broadcast_heads(q, k, v, W, xf)

        B, S, _, K_DIM = q_b.shape
        V_DIM = v_b.size(-1)

        # Require contiguous for predictable strides.
        q_c = q_b.contiguous()
        k_c = k_b.contiguous()
        v_c = v_b.contiguous()
        W_c = W_b.contiguous()
        xf_c = xf_b.contiguous()

        if h0 is None:
            h0_c = torch.empty(1, device=q.device, dtype=q.dtype)
            has_h0 = False
            h0_dtype = None
        else:
            assert h0.shape == (B, H, K_DIM, V_DIM)
            h0_c = h0.contiguous()
            has_h0 = True
            h0_dtype = h0.dtype

        runtime_config = get_m2rnn_runtime_config()
        save_hnew = runtime_config.save_hnew
        chunk_size = runtime_config.bwd_chunk_size
        fwd_chunk_size = _get_fwd_chunk_size()
        num_chunks = (S + chunk_size - 1) // chunk_size
        checkpoints = torch.empty(
            B, num_chunks + 1, H, K_DIM, V_DIM, device=q.device, dtype=torch.float32
        )
        if save_hnew:
            h_new_save = torch.empty(B, S, H, K_DIM, V_DIM, device=q.device, dtype=q.dtype)
        else:
            h_new_save = torch.empty(1, device=q.device, dtype=q.dtype)
        out = torch.empty(B, S, H, V_DIM, device=q.device, dtype=q.dtype)
        h_final = torch.empty(B, H, K_DIM, V_DIM, device=q.device, dtype=q.dtype)

        grid = (B, H)
        fwd_args = (
            q_c,
            k_c,
            v_c,
            W_c,
            xf_c,
            h0_c,
            checkpoints,
            h_new_save,
            out,
            h_final,
        )
        fwd_kwargs = dict(
            HAS_H0=has_h0,
            SAVE_HNEW=save_hnew,
            BWD_CHUNK_SIZE=chunk_size,
            FWD_CHUNK_SIZE=fwd_chunk_size,
            SEQ=S,
            NHEADS=H,
            K_DIM=K_DIM,
            V_DIM=V_DIM,
            q_sb=q_c.stride(0), q_ss=q_c.stride(1), q_sh=q_c.stride(2), q_sk=q_c.stride(3),
            k_sb=k_c.stride(0), k_ss=k_c.stride(1), k_sh=k_c.stride(2), k_sk=k_c.stride(3),
            v_sb=v_c.stride(0), v_ss=v_c.stride(1), v_sh=v_c.stride(2), v_sv=v_c.stride(3),
            W_sh=W_c.stride(0), W_sr=W_c.stride(1), W_sc=W_c.stride(2),
            xf_sb=xf_c.stride(0), xf_ss=xf_c.stride(1), xf_sh=xf_c.stride(2),
            h0_sb=h0_c.stride(0) if has_h0 else 0,
            h0_sh=h0_c.stride(1) if has_h0 else 0,
            h0_sk=h0_c.stride(2) if has_h0 else 0,
            h0_sv=h0_c.stride(3) if has_h0 else 0,
            ckpt_sb=checkpoints.stride(0), ckpt_sc=checkpoints.stride(1),
            ckpt_sh=checkpoints.stride(2), ckpt_sk=checkpoints.stride(3),
            ckpt_sv=checkpoints.stride(4),
            hn_sb=h_new_save.stride(0) if save_hnew else 0,
            hn_ss=h_new_save.stride(1) if save_hnew else 0,
            hn_sh=h_new_save.stride(2) if save_hnew else 0,
            hn_sk=h_new_save.stride(3) if save_hnew else 0,
            hn_sv=h_new_save.stride(4) if save_hnew else 0,
            out_sb=out.stride(0), out_ss=out.stride(1), out_sh=out.stride(2), out_sv=out.stride(3),
            hf_sb=h_final.stride(0), hf_sh=h_final.stride(1), hf_sk=h_final.stride(2), hf_sv=h_final.stride(3),
        )

        _m2rnn_fwd_chunk_kernel[grid](
            *fwd_args,
            **fwd_kwargs,
            num_warps=runtime_config.fwd_num_warps,
            num_stages=runtime_config.fwd_num_stages,
        )

        ctx.save_for_backward(q_c, k_c, v_c, W_c, xf_c, h0_c, checkpoints, h_new_save)
        ctx.has_h0 = has_h0
        ctx.h0_dtype = h0_dtype
        ctx.save_hnew = save_hnew
        ctx.bwd_chunk_size = chunk_size
        ctx.num_chunks = num_chunks
        ctx.orig_shapes = (q.shape, k.shape, v.shape, W.shape, xf.shape)
        return out, h_final

    @staticmethod
    def backward(ctx, dout, dh_final):
        """Reuses the existing chunked backward kernels from m2rnn_triton.

        The checkpoint format is identical regardless of whether the
        forward used the persistent kernel or the chunked kernel.
        """
        q_c, k_c, v_c, W_c, xf_c, _h0_c, checkpoints, h_new_save = ctx.saved_tensors
        has_h0 = ctx.has_h0
        save_hnew = ctx.save_hnew
        chunk_size = ctx.bwd_chunk_size
        num_chunks = ctx.num_chunks
        orig_q_shape, orig_k_shape, orig_v_shape, orig_W_shape, orig_xf_shape = ctx.orig_shapes

        B, S, H, K_DIM = q_c.shape
        V_DIM = v_c.size(-1)

        if dout is None:
            dout_c = torch.zeros(B, S, H, V_DIM, device=q_c.device, dtype=q_c.dtype)
        else:
            dout_c = dout.contiguous()

        if dh_final is None:
            dh_carry = torch.zeros(B, H, K_DIM, V_DIM, device=q_c.device, dtype=torch.float32)
        else:
            dh_carry = dh_final.to(torch.float32).contiguous().clone()

        dq = torch.empty_like(q_c)
        dk = torch.empty_like(k_c)
        dv = torch.empty_like(v_c)
        dxf = torch.empty_like(xf_c)
        dW_slabs = torch.zeros(B * H, V_DIM, V_DIM, device=q_c.device, dtype=torch.float32)
        max_chunk_len = min(chunk_size, S)
        y_chunk = torch.empty(
            B, max_chunk_len + 1, H, K_DIM, V_DIM, device=q_c.device, dtype=torch.float32
        )

        grid = (B, H)

        for chunk_idx in range(num_chunks - 1, -1, -1):
            start = chunk_idx * chunk_size
            chunk_len = min(chunk_size, S - start)

            _m2rnn_recompute_chunk_kernel[grid](
                k_c,
                v_c,
                W_c,
                xf_c,
                checkpoints,
                y_chunk,
                start,
                chunk_idx,
                CHUNK_LEN=chunk_len,
                K_DIM=K_DIM,
                V_DIM=V_DIM,
                k_sb=k_c.stride(0), k_ss=k_c.stride(1), k_sh=k_c.stride(2), k_sk=k_c.stride(3),
                v_sb=v_c.stride(0), v_ss=v_c.stride(1), v_sh=v_c.stride(2), v_sv=v_c.stride(3),
                W_sh=W_c.stride(0), W_sr=W_c.stride(1), W_sc=W_c.stride(2),
                xf_sb=xf_c.stride(0), xf_ss=xf_c.stride(1), xf_sh=xf_c.stride(2),
                ckpt_sb=checkpoints.stride(0), ckpt_sc=checkpoints.stride(1),
                ckpt_sh=checkpoints.stride(2), ckpt_sk=checkpoints.stride(3),
                ckpt_sv=checkpoints.stride(4),
                yc_sb=y_chunk.stride(0), yc_ss=y_chunk.stride(1),
                yc_sh=y_chunk.stride(2), yc_sk=y_chunk.stride(3), yc_sv=y_chunk.stride(4),
                num_warps=4,
                num_stages=3,
            )

            _m2rnn_bwd_chunk_kernel[grid](
                q_c,
                k_c,
                v_c,
                W_c,
                xf_c,
                y_chunk,
                h_new_save,
                dout_c,
                dh_carry,
                dW_slabs,
                dq,
                dk,
                dv,
                dxf,
                start,
                SAVE_HNEW=save_hnew,
                CHUNK_LEN=chunk_len,
                NHEADS=H,
                K_DIM=K_DIM,
                V_DIM=V_DIM,
                q_sb=q_c.stride(0), q_ss=q_c.stride(1), q_sh=q_c.stride(2), q_sk=q_c.stride(3),
                k_sb=k_c.stride(0), k_ss=k_c.stride(1), k_sh=k_c.stride(2), k_sk=k_c.stride(3),
                v_sb=v_c.stride(0), v_ss=v_c.stride(1), v_sh=v_c.stride(2), v_sv=v_c.stride(3),
                W_sh=W_c.stride(0), W_sr=W_c.stride(1), W_sc=W_c.stride(2),
                xf_sb=xf_c.stride(0), xf_ss=xf_c.stride(1), xf_sh=xf_c.stride(2),
                yc_sb=y_chunk.stride(0), yc_ss=y_chunk.stride(1),
                yc_sh=y_chunk.stride(2), yc_sk=y_chunk.stride(3), yc_sv=y_chunk.stride(4),
                hn_sb=h_new_save.stride(0) if save_hnew else 0,
                hn_ss=h_new_save.stride(1) if save_hnew else 0,
                hn_sh=h_new_save.stride(2) if save_hnew else 0,
                hn_sk=h_new_save.stride(3) if save_hnew else 0,
                hn_sv=h_new_save.stride(4) if save_hnew else 0,
                dout_sb=dout_c.stride(0), dout_ss=dout_c.stride(1),
                dout_sh=dout_c.stride(2), dout_sv=dout_c.stride(3),
                dhc_sb=dh_carry.stride(0), dhc_sh=dh_carry.stride(1),
                dhc_sk=dh_carry.stride(2), dhc_sv=dh_carry.stride(3),
                dW_sbh=dW_slabs.stride(0), dW_sr=dW_slabs.stride(1), dW_sc=dW_slabs.stride(2),
                dq_sb=dq.stride(0), dq_ss=dq.stride(1), dq_sh=dq.stride(2), dq_sk=dq.stride(3),
                dk_sb=dk.stride(0), dk_ss=dk.stride(1), dk_sh=dk.stride(2), dk_sk=dk.stride(3),
                dv_sb=dv.stride(0), dv_ss=dv.stride(1), dv_sh=dv.stride(2), dv_sv=dv.stride(3),
                dxf_sb=dxf.stride(0), dxf_ss=dxf.stride(1), dxf_sh=dxf.stride(2),
                num_warps=4,
                num_stages=3,
            )

        # Reduce dW slabs: (B*H, V, V) -> (H, V, V) by summing over batch.
        dW = dW_slabs.view(B, H, V_DIM, V_DIM).sum(dim=0).to(W_c.dtype)

        # Collapse broadcasted dims back to original shapes.
        dq_out = _unbroadcast_heads(dq, orig_q_shape[-2], dim=-2)
        dk_out = _unbroadcast_heads(dk, orig_k_shape[-2], dim=-2)
        dv_out = _unbroadcast_heads(dv, orig_v_shape[-2], dim=-2)
        dW_out = _unbroadcast_heads(dW, orig_W_shape[0], dim=0)
        dxf_out = _unbroadcast_heads(dxf, orig_xf_shape[-1], dim=-1)

        dh0_out = dh_carry.to(ctx.h0_dtype) if has_h0 else None

        return dq_out, dk_out, dv_out, dW_out, dxf_out, dh0_out


def m2rnn_scan_triton_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    *,
    h0: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked Triton M2RNN scan -- same interface as ``m2rnn_scan_triton``.

    Uses an outer loop over forward chunks (controlled by
    ``CPPMEGA_M2RNN_FWD_CHUNK_SIZE``, default 128) to reduce per-iteration
    register pressure compared to the persistent unrolled loop in the
    default kernel.

    Shapes:
        q : (B, S, n_q, K_DIM)
        k : (B, S, n_k, K_DIM)
        v : (B, S, n_v, V_DIM)
        W : (n_w, V_DIM, V_DIM)
        xf: (B, S, n_f)
    Returns:
        out: (B, S, H, V_DIM)
        h_final: (B, H, K_DIM, V_DIM)
        where H = max(n_q, n_k, n_v, n_w, n_f).
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available; cannot run m2rnn_scan_triton_chunked")
    return _M2RNNChunkedFn.apply(q, k, v, W, xf, h0)
