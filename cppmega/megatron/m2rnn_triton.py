"""Fused Triton kernel for M²RNN scan.

Replaces the pure-Python ``for s in range(seq)`` loop in
``cppmega/megatron/m2rnn_spec.py::_torch_m2rnn_forward`` with a persistent
per-(batch, head) kernel that keeps the hidden state in registers for the
entire sequence.

Shapes (matching ``_torch_m2rnn_forward`` after head broadcast):
    q:  (B, S, H, K)
    k:  (B, S, H, K)
    v:  (B, S, H, V)
    W:  (H, V, V)
    xf: (B, S, H)
    out: (B, S, H, V)
    h_final: (B, H, K, V)

Per-step math (fwd):
    x_s   = k_s[:, None] * v_s[None, :]                 # (K, V) rank-1 outer product
    h_new = tanh(h @ W + x_s)                           # (K, V)
    h     = xf_s * h + (1 - xf_s) * h_new               # gated state
    out_s = sum_k(q_s[:, None] * h, axis=0)             # (V,)  == q_s @ h

Per-step math (bwd, processing positions in reverse):
    dh += q_s[:, None] * dy_s[None, :]                  # flow from q-projection
    dh_new      = (1 - xf_s) * dh
    df_s        = sum((h_prev_gated - h_new) * dh)      # gradient on scalar gate
    d_pre_tanh  = dh_new * (1 - h_new**2)               # through tanh
    dh_from_mm  = d_pre_tanh @ W.T                      # through h@W
    dW_accum   += h_prev.T @ d_pre_tanh                 # W accumulator
    dh          = xf_s * dh + dh_from_mm
    dk_s        = sum_v(d_pre_tanh * v_s[None, :], axis=1)
    dv_s        = sum_k(d_pre_tanh * k_s[:, None], axis=0)
    dq_s        = sum_v(dy_s[None, :] * h_prev_gated, axis=1)

Where ``h_prev`` is the state *before* this step's update and
``h_prev_gated`` is the state *after* this step's update (what was stored
in ``y[s]`` during the forward pass).

Backward checkpoint strategy:
    We always save the full y history (shape (B, S, H, K, V)) from the
    forward.  The h_new candidate tensor (same shape) is OPTIONAL:
    - CPPMEGA_M2RNN_SAVE_HNEW=0 (default): bwd recomputes h_new from
      h_prev + k + v + W (one tl.dot + tanh per step, ~2 ms/layer extra).
      Saves (B*S*H*K*V*2) bytes per R-layer.  At NAM56R full dims
      (B=4, S=4096, H=44, K=64, V=16) that's ~1.4 GB/layer, ~5.6 GB total.
    - CPPMEGA_M2RNN_SAVE_HNEW=1: bwd loads h_new from saved tensor (faster
      bwd, but uses more memory).  Useful for debugging parity.

Usage:
    from cppmega.megatron.m2rnn_triton import m2rnn_scan_triton
    out, h_final = m2rnn_scan_triton(q, k, v, W, xf, h0=None)
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# When True, the fwd kernel saves h_new (the pre-gate tanh candidate) to global
# memory so the backward can load it directly.  When False (the default), the
# backward RECOMPUTES h_new from h_prev + k + v + W, saving
# (B * S * H * K * V * 2) bytes per R-layer call.  At NAM56R production dims
# (B=4, S=4096, H=44, K=64, V=16) that is ~1.4 GB/layer, ~5.6 GB across 4
# R-layers — the difference between OOM and fitting at MBS=5.
_SAVE_HNEW: bool = os.environ.get("CPPMEGA_M2RNN_SAVE_HNEW", "0") == "1"


# Lazy import of triton so importing this module on a CPU-only host still
# lets ``cppmega.megatron.m2rnn_spec`` load (the fallback to the torch scan
# can then be detected via the ``TRITON_AVAILABLE`` flag).
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - triton-less envs
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    # Triton autotune grid for both fwd and bwd.  The shape discriminators
    # (BATCH, SEQ, NHEADS, K_DIM, V_DIM) are passed as tl.constexpr so they
    # are the natural autotune key.  First call at a new shape does a cold
    # sweep (~5-30s), subsequent calls reuse the cached selection.
    #
    # Grid:
    #   num_warps in {1, 2, 4, 8, 16}
    #   num_stages in {1, 2, 3, 4}
    # 16-warp entries are kept — if the Triton compiler rejects them for
    # this shape it just drops them from the config list.
    _M2RNN_AUTOTUNE_CONFIGS = [
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [1, 2, 4, 8, 16]
        for ns in [1, 2, 3, 4]
    ]
    # BATCH is deliberately excluded from the autotune key — it only
    # affects the grid (program_id(0)), not the kernel body.  Including
    # it forces a full autotune sweep + Triton JIT recompilation for
    # each new batch size, which OOMs at MBS≥3 because the JIT compiler
    # needs ~10-20 GB GPU workspace while the model already fills 100+ GB.
    _M2RNN_AUTOTUNE_KEY = ["SEQ", "NHEADS", "K_DIM", "V_DIM"]

    @triton.autotune(configs=_M2RNN_AUTOTUNE_CONFIGS, key=_M2RNN_AUTOTUNE_KEY)
    @triton.jit
    def _m2rnn_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        W_ptr,
        xf_ptr,
        h0_ptr,
        y_ptr,
        hnew_ptr,
        out_ptr,
        hfinal_ptr,
        HAS_H0: tl.constexpr,
        SAVE_HNEW: tl.constexpr,
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
        y_sb, y_ss, y_sh, y_sk, y_sv,
        hn_sb, hn_ss, hn_sh, hn_sk, hn_sv,
        out_sb, out_ss, out_sh, out_sv,
        hf_sb, hf_sh, hf_sk, hf_sv,
    ):
        """One program per (batch, head) pair.

        Maintains ``h`` (K_DIM x V_DIM) in registers; loops over SEQ steps.
        For each step writes ``y[b, s, h]`` (the new state, used as saved
        tensor for bwd) and ``out[b, s, h]`` (q @ h).
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

        for s in range(SEQ):
            # Load q_s, k_s, v_s, xf_s for this (b, h_idx)
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

            # Rank-1 outer product x = k ⊗ v : (K_DIM, V_DIM)
            x = k_s[:, None] * v_s[None, :]

            # h @ W : (K_DIM, V_DIM) @ (V_DIM, V_DIM) -> (K_DIM, V_DIM)
            # tl.dot requires K dim >= 16 (V_DIM=16 hits the minimum exactly).
            # input_precision="ieee" forces full fp32 (no TF32 truncation)
            # since both h (carried in fp32) and W (loaded to fp32) are fp32.
            # (tf32 was tried and breaks the fp32-input parity test.)
            hW = tl.dot(h, W, out_dtype=tl.float32, input_precision="ieee")

            pre = hW + x
            # Hardware-accelerated tanh via inline PTX: ``tanh.approx.f32``
            # is a single-cycle SFU instruction (sm_75+).  Triton's
            # ``libdevice.tanh`` maps to the ``__nv_tanhf`` bitcode which
            # is a software polynomial (slower), and the cuda-backend
            # libdevice does not surface ``fast_tanhf``.  Drop to asm.
            h_new = tl.inline_asm_elementwise(
                asm="tanh.approx.f32 $0, $1;",
                constraints="=f,f",
                args=[pre],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )

            # Optionally store h_new (pre-gate candidate) for the bwd kernel.
            # When SAVE_HNEW=False, the bwd kernel recomputes h_new from
            # h_prev + k + v + W, saving (B*S*H*K*V*2) bytes of global memory.
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

            # Store y[b, s, h] = h  (the *new* gated state)
            tl.store(
                y_ptr
                + b * y_sb
                + s * y_ss
                + h_idx * y_sh
                + offs_k[:, None] * y_sk
                + offs_v[None, :] * y_sv,
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

    @triton.autotune(configs=_M2RNN_AUTOTUNE_CONFIGS, key=_M2RNN_AUTOTUNE_KEY)
    @triton.jit
    def _m2rnn_bwd_kernel(
        # inputs (fwd activations)
        q_ptr,
        k_ptr,
        v_ptr,
        W_ptr,
        xf_ptr,
        h0_ptr,
        y_ptr,         # saved fwd y (B, S, H, K, V)  — contains h *after* each step
        hnew_ptr,      # saved fwd h_new candidate (B, S, H, K, V) — skip recompute
        dout_ptr,      # grad wrt fwd out: (B, S, H, V)
        dhfinal_ptr,   # grad wrt final h: (B, H, K, V) — usually zero
        # outputs
        dq_ptr,
        dk_ptr,
        dv_ptr,
        dW_ptr,        # accumulated per (batch, head) slab, reduced on host
        dxf_ptr,
        dh0_ptr,
        HAS_H0: tl.constexpr,
        SAVE_HNEW: tl.constexpr,
        SEQ: tl.constexpr,
        NHEADS: tl.constexpr,
        K_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        q_sb, q_ss, q_sh, q_sk,
        k_sb, k_ss, k_sh, k_sk,
        v_sb, v_ss, v_sh, v_sv,
        W_sh, W_sr, W_sc,
        xf_sb, xf_ss, xf_sh,
        h0_sb, h0_sh, h0_sk, h0_sv,
        y_sb, y_ss, y_sh, y_sk, y_sv,
        hn_sb, hn_ss, hn_sh, hn_sk, hn_sv,
        dout_sb, dout_ss, dout_sh, dout_sv,
        dhf_sb, dhf_sh, dhf_sk, dhf_sv,
        dq_sb, dq_ss, dq_sh, dq_sk,
        dk_sb, dk_ss, dk_sh, dk_sk,
        dv_sb, dv_ss, dv_sh, dv_sv,
        # dW has an extra leading (batch*head) dim so each program writes to
        # its own slab; host-side reduce.sum(dim=0) on B to get (H, V, V).
        dW_sbh, dW_sr, dW_sc,
        dxf_sb, dxf_ss, dxf_sh,
        dh0_sb, dh0_sh, dh0_sk, dh0_sv,
    ):
        """One program per (batch, head) pair.  Backward sweep over seq.

        Reads y[s] (the gated state *after* step s).  Also needs
        h_prev (the state *before* step s) for dW; we get it from y[s-1]
        (or h0 if s == 0).

        When SAVE_HNEW=True, loads h_new directly from the saved tensor.
        When SAVE_HNEW=False, recomputes h_new = tanh(h_prev @ W + k⊗v)
        per step (one extra tl.dot + tanh) to avoid storing the full
        (B, S, H, K, V) tensor in global memory.
        """
        b = tl.program_id(0)
        h_idx = tl.program_id(1)

        offs_k = tl.arange(0, K_DIM)
        offs_v = tl.arange(0, V_DIM)
        offs_vv = tl.arange(0, V_DIM)

        # Load W (V, V) and W.T into registers.
        W_row = tl.arange(0, V_DIM)
        W_col = tl.arange(0, V_DIM)
        W = tl.load(
            W_ptr + h_idx * W_sh + W_row[:, None] * W_sr + W_col[None, :] * W_sc
        ).to(tl.float32)
        Wt = tl.trans(W)

        # Running gradient w.r.t. h at the *current* (backwards-traversed) step.
        # Seeded from dhfinal (usually zero from autograd — only nonzero if user
        # consumes the final state directly).
        dh = tl.load(
            dhfinal_ptr
            + b * dhf_sb
            + h_idx * dhf_sh
            + offs_k[:, None] * dhf_sk
            + offs_v[None, :] * dhf_sv,
        ).to(tl.float32)

        # dW accumulator (V, V).
        dW = tl.zeros((V_DIM, V_DIM), dtype=tl.float32)

        for s_rev in range(SEQ):
            s = SEQ - 1 - s_rev

            # Load h_prev: the state *before* step s.
            if HAS_H0:
                if s == 0:
                    h_prev = tl.load(
                        h0_ptr
                        + b * h0_sb
                        + h_idx * h0_sh
                        + offs_k[:, None] * h0_sk
                        + offs_v[None, :] * h0_sv,
                    ).to(tl.float32)
                else:
                    h_prev = tl.load(
                        y_ptr
                        + b * y_sb
                        + (s - 1) * y_ss
                        + h_idx * y_sh
                        + offs_k[:, None] * y_sk
                        + offs_v[None, :] * y_sv,
                    ).to(tl.float32)
            else:
                if s == 0:
                    h_prev = tl.zeros((K_DIM, V_DIM), dtype=tl.float32)
                else:
                    h_prev = tl.load(
                        y_ptr
                        + b * y_sb
                        + (s - 1) * y_ss
                        + h_idx * y_sh
                        + offs_k[:, None] * y_sk
                        + offs_v[None, :] * y_sv,
                    ).to(tl.float32)

            # Load h_curr (gated state after step s).
            h_curr = tl.load(
                y_ptr
                + b * y_sb
                + s * y_ss
                + h_idx * y_sh
                + offs_k[:, None] * y_sk
                + offs_v[None, :] * y_sv,
            ).to(tl.float32)

            # Load per-step inputs.
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
            dy_s = tl.load(
                dout_ptr + b * dout_sb + s * dout_ss + h_idx * dout_sh + offs_v * dout_sv,
            ).to(tl.float32)

            # Obtain h_new (pre-gate tanh candidate).
            if SAVE_HNEW:
                # Load from saved fwd tensor (fast, but costs memory).
                h_new = tl.load(
                    hnew_ptr
                    + b * hn_sb
                    + s * hn_ss
                    + h_idx * hn_sh
                    + offs_k[:, None] * hn_sk
                    + offs_v[None, :] * hn_sv,
                ).to(tl.float32)
            else:
                # Recompute h_new from h_prev + inputs (same math as fwd).
                # k_s and v_s are already loaded above.
                x = k_s[:, None] * v_s[None, :]
                hW = tl.dot(h_prev, W, out_dtype=tl.float32, input_precision="ieee")
                pre = hW + x
                h_new = tl.inline_asm_elementwise(
                    asm="tanh.approx.f32 $0, $1;",
                    constraints="=f,f",
                    args=[pre],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )

            # dq_s = dy_s @ h_curr.T  (reduce over v)
            dq_s = tl.sum(dy_s[None, :] * h_curr, axis=1)

            # Incoming dh picks up the contribution from q's output.
            # out_s = q_s @ h_curr, so dh_curr += q_s outer dy_s.
            dh = dh + q_s[:, None] * dy_s[None, :]

            # df_s: grad through the gate.
            # h_curr = xf_s * h_prev + (1 - xf_s) * h_new
            # dh_prev_from_gate = xf_s * dh
            # dh_new_from_gate  = (1 - xf_s) * dh
            # d_xf_s = sum((h_prev - h_new) * dh)
            df_s = tl.sum((h_prev - h_new) * dh)

            dh_new = (1.0 - xf_s) * dh

            # Through tanh: d/dpre tanh(pre) = 1 - h_new^2
            d_pre = dh_new * (1.0 - h_new * h_new)

            # Through h @ W + x:
            #   dh_prev_from_mm = d_pre @ W.T
            #   dW_accum       += h_prev.T @ d_pre
            dh_from_mm = tl.dot(d_pre, Wt, out_dtype=tl.float32, input_precision="ieee")
            dW += tl.dot(tl.trans(h_prev), d_pre, out_dtype=tl.float32, input_precision="ieee")

            # dk_s and dv_s via outer product:
            # x[i, j] = k[i] * v[j]
            # dk[i] = sum_j d_pre[i, j] * v[j]
            # dv[j] = sum_i d_pre[i, j] * k[i]
            dk_s = tl.sum(d_pre * v_s[None, :], axis=1)
            dv_s = tl.sum(d_pre * k_s[:, None], axis=0)

            # Update dh going backwards.
            dh = xf_s * dh + dh_from_mm

            # Store outputs for this step.
            tl.store(
                dq_ptr + b * dq_sb + s * dq_ss + h_idx * dq_sh + offs_k * dq_sk,
                dq_s,
            )
            tl.store(
                dk_ptr + b * dk_sb + s * dk_ss + h_idx * dk_sh + offs_k * dk_sk,
                dk_s,
            )
            tl.store(
                dv_ptr + b * dv_sb + s * dv_ss + h_idx * dv_sh + offs_v * dv_sv,
                dv_s,
            )
            tl.store(
                dxf_ptr + b * dxf_sb + s * dxf_ss + h_idx * dxf_sh,
                df_s,
            )

        # Store dW slab (one per (batch, head)); host reduces over batch.
        bh = b * NHEADS + h_idx
        tl.store(
            dW_ptr
            + bh * dW_sbh
            + W_row[:, None] * dW_sr
            + W_col[None, :] * dW_sc,
            dW,
        )

        # Store dh0 (grad wrt initial state) = dh after the final (reverse-time) update.
        if HAS_H0:
            tl.store(
                dh0_ptr
                + b * dh0_sb
                + h_idx * dh0_sh
                + offs_k[:, None] * dh0_sk
                + offs_v[None, :] * dh0_sv,
                dh,
            )


# ---------------------------------------------------------------------------
# Python wrappers + autograd.Function
# ---------------------------------------------------------------------------


def _broadcast_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Broadcast all head counts to the common maximum (same logic as
    ``_torch_m2rnn_forward``)."""
    n_q = q.size(-2)
    n_k = k.size(-2)
    n_v = v.size(-2)
    n_w = W.size(0)
    n_f = xf.size(-1)
    n = max(n_q, n_k, n_v, n_w, n_f)

    if n_q != n:
        q = q.repeat_interleave(n // n_q, dim=-2)
    if n_k != n:
        k = k.repeat_interleave(n // n_k, dim=-2)
    if n_v != n:
        v = v.repeat_interleave(n // n_v, dim=-2)
    if n_w != n:
        W = W.repeat_interleave(n // n_w, dim=0)
    if n_f != n:
        xf = xf.repeat_interleave(n // n_f, dim=-1)
    return q, k, v, W, xf, n


class _M2RNNFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, W, xf, h0):
        assert TRITON_AVAILABLE, "Triton is required for m2rnn_scan_triton"
        assert q.is_cuda, "m2rnn_scan_triton requires CUDA tensors"

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
            h0_c = torch.zeros(B, H, K_DIM, V_DIM, device=q.device, dtype=q.dtype)
            has_h0 = False
        else:
            assert h0.shape == (B, H, K_DIM, V_DIM)
            h0_c = h0.contiguous()
            has_h0 = True

        y = torch.empty(B, S, H, K_DIM, V_DIM, device=q.device, dtype=q.dtype)
        if _SAVE_HNEW:
            h_new_save = torch.empty(B, S, H, K_DIM, V_DIM, device=q.device, dtype=q.dtype)
        else:
            # Dummy 1-element tensor: Triton needs a valid pointer but won't
            # read/write it when SAVE_HNEW=False.
            h_new_save = torch.empty(1, device=q.device, dtype=q.dtype)
        out = torch.empty(B, S, H, V_DIM, device=q.device, dtype=q.dtype)
        h_final = torch.empty(B, H, K_DIM, V_DIM, device=q.device, dtype=q.dtype)

        grid = (B, H)

        _m2rnn_fwd_kernel[grid](
            q_c,
            k_c,
            v_c,
            W_c,
            xf_c,
            h0_c,
            y,
            h_new_save,
            out,
            h_final,
            HAS_H0=has_h0,
            SAVE_HNEW=_SAVE_HNEW,
            SEQ=S,
            NHEADS=H,
            K_DIM=K_DIM,
            V_DIM=V_DIM,
            q_sb=q_c.stride(0), q_ss=q_c.stride(1), q_sh=q_c.stride(2), q_sk=q_c.stride(3),
            k_sb=k_c.stride(0), k_ss=k_c.stride(1), k_sh=k_c.stride(2), k_sk=k_c.stride(3),
            v_sb=v_c.stride(0), v_ss=v_c.stride(1), v_sh=v_c.stride(2), v_sv=v_c.stride(3),
            W_sh=W_c.stride(0), W_sr=W_c.stride(1), W_sc=W_c.stride(2),
            xf_sb=xf_c.stride(0), xf_ss=xf_c.stride(1), xf_sh=xf_c.stride(2),
            h0_sb=h0_c.stride(0), h0_sh=h0_c.stride(1), h0_sk=h0_c.stride(2), h0_sv=h0_c.stride(3),
            y_sb=y.stride(0), y_ss=y.stride(1), y_sh=y.stride(2), y_sk=y.stride(3), y_sv=y.stride(4),
            hn_sb=h_new_save.stride(0) if _SAVE_HNEW else 0,
            hn_ss=h_new_save.stride(1) if _SAVE_HNEW else 0,
            hn_sh=h_new_save.stride(2) if _SAVE_HNEW else 0,
            hn_sk=h_new_save.stride(3) if _SAVE_HNEW else 0,
            hn_sv=h_new_save.stride(4) if _SAVE_HNEW else 0,
            out_sb=out.stride(0), out_ss=out.stride(1), out_sh=out.stride(2), out_sv=out.stride(3),
            hf_sb=h_final.stride(0), hf_sh=h_final.stride(1), hf_sk=h_final.stride(2), hf_sv=h_final.stride(3),
        )

        ctx.save_for_backward(q_c, k_c, v_c, W_c, xf_c, h0_c, y, h_new_save)
        ctx.has_h0 = has_h0
        ctx.save_hnew = _SAVE_HNEW
        ctx.orig_shapes = (q.shape, k.shape, v.shape, W.shape, xf.shape)
        return out, h_final

    @staticmethod
    def backward(ctx, dout, dh_final):
        q_c, k_c, v_c, W_c, xf_c, h0_c, y, h_new_save = ctx.saved_tensors
        has_h0 = ctx.has_h0
        save_hnew = ctx.save_hnew
        orig_q_shape, orig_k_shape, orig_v_shape, orig_W_shape, orig_xf_shape = ctx.orig_shapes

        B, S, H, K_DIM = q_c.shape
        V_DIM = v_c.size(-1)

        dout_c = dout.contiguous()
        dhf_c = dh_final.contiguous() if dh_final is not None else torch.zeros_like(h0_c)

        dq = torch.empty_like(q_c)
        dk = torch.empty_like(k_c)
        dv = torch.empty_like(v_c)
        dxf = torch.empty_like(xf_c)
        # Per-(batch*head) dW slabs; reduced across batch on host.
        dW_slabs = torch.empty(B * H, V_DIM, V_DIM, device=q_c.device, dtype=torch.float32)
        dh0 = torch.empty_like(h0_c) if has_h0 else torch.empty(1, device=q_c.device, dtype=q_c.dtype)

        grid = (B, H)

        _m2rnn_bwd_kernel[grid](
            q_c, k_c, v_c, W_c, xf_c, h0_c, y, h_new_save,
            dout_c, dhf_c,
            dq, dk, dv, dW_slabs, dxf, dh0,
            HAS_H0=has_h0,
            SAVE_HNEW=save_hnew,
            SEQ=S,
            NHEADS=H,
            K_DIM=K_DIM,
            V_DIM=V_DIM,
            q_sb=q_c.stride(0), q_ss=q_c.stride(1), q_sh=q_c.stride(2), q_sk=q_c.stride(3),
            k_sb=k_c.stride(0), k_ss=k_c.stride(1), k_sh=k_c.stride(2), k_sk=k_c.stride(3),
            v_sb=v_c.stride(0), v_ss=v_c.stride(1), v_sh=v_c.stride(2), v_sv=v_c.stride(3),
            W_sh=W_c.stride(0), W_sr=W_c.stride(1), W_sc=W_c.stride(2),
            xf_sb=xf_c.stride(0), xf_ss=xf_c.stride(1), xf_sh=xf_c.stride(2),
            h0_sb=h0_c.stride(0), h0_sh=h0_c.stride(1), h0_sk=h0_c.stride(2), h0_sv=h0_c.stride(3),
            y_sb=y.stride(0), y_ss=y.stride(1), y_sh=y.stride(2), y_sk=y.stride(3), y_sv=y.stride(4),
            hn_sb=h_new_save.stride(0) if save_hnew else 0,
            hn_ss=h_new_save.stride(1) if save_hnew else 0,
            hn_sh=h_new_save.stride(2) if save_hnew else 0,
            hn_sk=h_new_save.stride(3) if save_hnew else 0,
            hn_sv=h_new_save.stride(4) if save_hnew else 0,
            dout_sb=dout_c.stride(0), dout_ss=dout_c.stride(1), dout_sh=dout_c.stride(2), dout_sv=dout_c.stride(3),
            dhf_sb=dhf_c.stride(0), dhf_sh=dhf_c.stride(1), dhf_sk=dhf_c.stride(2), dhf_sv=dhf_c.stride(3),
            dq_sb=dq.stride(0), dq_ss=dq.stride(1), dq_sh=dq.stride(2), dq_sk=dq.stride(3),
            dk_sb=dk.stride(0), dk_ss=dk.stride(1), dk_sh=dk.stride(2), dk_sk=dk.stride(3),
            dv_sb=dv.stride(0), dv_ss=dv.stride(1), dv_sh=dv.stride(2), dv_sv=dv.stride(3),
            dW_sbh=dW_slabs.stride(0), dW_sr=dW_slabs.stride(1), dW_sc=dW_slabs.stride(2),
            dxf_sb=dxf.stride(0), dxf_ss=dxf.stride(1), dxf_sh=dxf.stride(2),
            dh0_sb=dh0.stride(0) if has_h0 else 0,
            dh0_sh=dh0.stride(1) if has_h0 else 0,
            dh0_sk=dh0.stride(2) if has_h0 else 0,
            dh0_sv=dh0.stride(3) if has_h0 else 0,
        )

        # Reduce dW slabs: (B*H, V, V) -> (H, V, V) by summing over batch.
        dW = dW_slabs.view(B, H, V_DIM, V_DIM).sum(dim=0).to(W_c.dtype)

        # Collapse broadcasted dims back to original shapes. If the original
        # tensor had fewer heads and was repeat_interleaved, sum the expanded
        # grad back to the original head count.
        dq_out = _unbroadcast_heads(dq, orig_q_shape[-2], dim=-2)
        dk_out = _unbroadcast_heads(dk, orig_k_shape[-2], dim=-2)
        dv_out = _unbroadcast_heads(dv, orig_v_shape[-2], dim=-2)
        dW_out = _unbroadcast_heads(dW, orig_W_shape[0], dim=0)
        dxf_out = _unbroadcast_heads(dxf, orig_xf_shape[-1], dim=-1)

        dh0_out = dh0 if has_h0 else None

        return dq_out, dk_out, dv_out, dW_out, dxf_out, dh0_out


def _unbroadcast_heads(grad: torch.Tensor, orig_n: int, dim: int) -> torch.Tensor:
    """Inverse of ``repeat_interleave``: reshape and sum consecutive groups.

    If ``grad`` has ``n`` heads along ``dim`` and ``orig_n`` divides ``n``,
    view as (..., orig_n, n // orig_n, ...) and sum over the group axis.
    """
    cur_n = grad.size(dim)
    if cur_n == orig_n:
        return grad
    assert cur_n % orig_n == 0, f"cannot unbroadcast {cur_n} heads to {orig_n}"
    # Resolve negative dim BEFORE building the new shape — using ``dim + 1``
    # with a negative ``dim`` (e.g. -2) collapses the last axis instead of
    # the freshly inserted group axis, silently producing the wrong shape.
    if dim < 0:
        dim += grad.ndim
    group = cur_n // orig_n
    shape = list(grad.shape)
    new_shape = shape[:dim] + [orig_n, group] + shape[dim + 1 :]
    return grad.view(*new_shape).sum(dim=dim + 1)


def m2rnn_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    *,
    h0: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton M²RNN scan, drop-in replacement for
    ``cppmega.megatron.m2rnn_spec._torch_m2rnn_forward``.

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
        raise RuntimeError("Triton is not available; cannot run m2rnn_scan_triton")
    return _M2RNNFn.apply(q, k, v, W, xf, h0)
