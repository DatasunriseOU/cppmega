"""Wave2 triangular/tile-pruned monolithic Triton model for Mamba3 MIMO bwd_bwd.

Wave1 measured a monolithic checksum owner that still paid full ``64x64``
masked applies.  This probe keeps the same checksum contract but changes the
Triton owner body to apply only causal 4-step tiles:

* ``LKQ`` and ``dk_intra`` are produced as ``16x16`` tiles so ``DSSDA`` still
  sees the full unmasked matrix products it requires.
* Causal consumers apply only tiles with ``row_time <= col_time``.  Strictly
  future off-diagonal 4-step tiles are applied without element masks; diagonal
  tiles keep an internal causal mask for correctness.
* The kernel stores one checksum per ``(B,H,chunk)`` program.  It is still a
  compute lower bound, not a drop-in output kernel.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import traceback
from dataclasses import asdict
from typing import Any, Callable

import torch

_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from rr_diag_wave1_mono_triton_reuse_model import (
    COMPARISON_CONTEXT as WAVE1_COMPARISON_CONTEXT,
    PRESETS,
    Shape,
    _dtype,
    _fma_model as _wave1_fma_model,
    _has_triton,
    _indices,
    _memory_model,
    _shape_from_args,
    _stats,
    make_prepared_inputs,
    mono_reuse_torch_checksum,
)


COMPARISON_CONTEXT: dict[str, Any] = {
    **WAVE1_COMPARISON_CONTEXT,
    "wave1_mono_reuse_full_mask_lower_bound_ms": 4.53881,
    "wave1_mono_reuse_full_mask_fma": 114_631_000_064,
    "wave1_ideal_causal_apply_fma": 96_381_583_360,
}


def _time_cuda(fn: Callable[[], object], *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return _stats(times)


def _time_wall(fn: Callable[[], object], *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return _stats(times)


@torch.no_grad()
def mono_reuse_torch_checksum_batched(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
    *,
    handoff_dtype: torch.dtype,
    batch_chunks: int,
) -> torch.Tensor:
    """Memory-capped torch checksum reference matching the Wave1 algebra."""

    total = shape.B * shape.H * shape.nchunks
    fcs = shape.fcs
    batch_chunks = max(1, int(batch_chunks))

    q_all = inputs["q"].reshape(total, fcs, shape.N)
    k_all = inputs["k"].reshape(total, fcs, shape.N)
    k_pre_all = inputs["k_pre_trap"].reshape(total, fcs, shape.N)
    dstates_all = inputs["dstates"].reshape(total, shape.N, shape.P)
    states_all = inputs["states"].reshape(total, shape.N, shape.P)
    dphi_all = inputs["dphi"].reshape(total, fcs, shape.P)
    psiv_all = inputs["psiv"].reshape(total, fcs, shape.P)
    seg_all = inputs["segsum"].reshape(total, shape.chunk, shape.chunk)
    exp_rev_all = inputs["exp_rev"].reshape(total, shape.chunk)
    exp_cs_all = inputs["exp_cs"].reshape(total, shape.chunk)
    qk_all = inputs["qk_dot"].reshape(total, shape.chunk, shape.R, shape.R)
    gamma_all = inputs["gamma"].reshape(total, shape.chunk)
    v_all = (
        inputs["v"]
        .reshape(shape.B, shape.nchunks, shape.chunk, shape.H, shape.P)
        .permute(0, 3, 1, 2, 4)
        .reshape(total, shape.chunk, shape.P)
    )
    h_ids_all = ((torch.arange(total, device=q_all.device) // shape.nchunks) % shape.H).long()

    ci, causal, _ = _indices(shape, q_all.device)
    ci_rows = ci[None, :].expand(fcs, fcs)
    ci_cols = ci[:, None].expand(fcs, fcs)
    out = torch.empty(total, device=q_all.device, dtype=torch.float32)

    for start in range(0, total, batch_chunks):
        end = min(start + batch_chunks, total)
        q = q_all[start:end]
        k = k_all[start:end]
        k_pre_trap = k_pre_all[start:end]
        dstates = dstates_all[start:end]
        states = states_all[start:end]
        dphi = dphi_all[start:end]
        psiv = psiv_all[start:end]

        seg_weight = seg_all[start:end, ci_rows, ci_cols]
        seg_exp = torch.exp(seg_weight)

        exp_rev = exp_rev_all[start:end].repeat_interleave(shape.R, dim=-1).unsqueeze(-1)
        exp_cs = exp_cs_all[start:end].repeat_interleave(shape.R, dim=-1).unsqueeze(-1)

        lkq = torch.bmm(k, q.transpose(1, 2))
        lkq_masked = torch.where(causal[None], lkq.float() * seg_exp, torch.zeros_like(lkq.float()))
        state_dpsi = torch.bmm(k, dstates).float() * exp_rev
        lkq_dpsi = torch.bmm(lkq_masked.to(handoff_dtype), dphi).float()

        h_ids = h_ids_all[start:end]
        d_h = inputs["D"][h_ids].float().view(end - start, 1, 1)
        dpsi = state_dpsi + lkq_dpsi + d_h * dphi.float()

        qk = qk_all[start:end]
        qk_t = qk.transpose(-1, -2).float()
        dphi_r = dphi.reshape(end - start, shape.chunk, shape.R, shape.P).float()
        qk_contrib = torch.einsum("bcio,bcop->bcip", qk_t, dphi_r)
        gamma = gamma_all[start:end].float()
        dpsi = dpsi + (qk_contrib * gamma[:, :, None, None]).reshape(end - start, fcs, shape.P)
        dpsi = dpsi.to(handoff_dtype).float()

        dpsi_r = dpsi.reshape(end - start, shape.chunk, shape.R, shape.P)
        mimo = inputs["mimo_v"][h_ids].float()
        dv_checksum = (dpsi_r * mimo[:, None, :, :]).sum(dim=2).sum(dim=(1, 2))

        v = v_all[start:end].float()
        dmimo_checksum = (dpsi_r * v[:, :, None, :]).sum(dim=1).sum(dim=(1, 2))
        dd_checksum = dphi.float().sum(dim=(1, 2))

        dk_state = torch.bmm(psiv, dstates.transpose(1, 2)).float()
        dda_cs_rev = (k.float() * dk_state).reshape(end - start, shape.chunk, shape.R, shape.N).sum(dim=(2, 3))
        dk_state = dk_state * exp_rev

        dk_intra = torch.bmm(psiv, dphi.transpose(1, 2))
        dqk_diag = dk_intra.transpose(1, 2).reshape(
            end - start,
            shape.chunk,
            shape.R,
            shape.chunk,
            shape.R,
        )
        dqk_diag = dqk_diag.diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2).contiguous()
        dgamma_diag = (qk.float() * dqk_diag.float()).sum(dim=(2, 3))
        dgamma_diag_checksum = dgamma_diag.sum(dim=1)

        dssda = (lkq.float() * dk_intra.float()).reshape(
            end - start,
            shape.chunk,
            shape.R,
            shape.chunk,
            shape.R,
        ).sum(dim=(2, 4))
        dk_intra_masked = torch.where(causal[None], dk_intra.float() * seg_exp, torch.zeros_like(dk_intra.float()))

        dk_nodiag = dk_state + torch.bmm(dk_intra_masked.to(handoff_dtype), q).float()
        dfactor = (k_pre_trap.float() * dk_nodiag).reshape(end - start, shape.chunk, shape.R, shape.N).sum(dim=(2, 3))

        dq_state = torch.bmm(dphi, states.transpose(1, 2)).float()
        dda_cs = (q.float() * dq_state).reshape(end - start, shape.chunk, shape.R, shape.N).sum(dim=(2, 3))
        dq = dq_state * exp_cs + torch.bmm(dk_intra_masked.transpose(1, 2).to(handoff_dtype), k).float()

        dda = (states.float() * dstates.float()).sum(dim=(1, 2))

        out[start:end] = (
            dv_checksum
            + dmimo_checksum
            + dd_checksum
            + dk_nodiag.sum(dim=(1, 2))
            + dq.sum(dim=(1, 2))
            + dda_cs_rev.sum(dim=1)
            + dssda.sum(dim=(1, 2))
            + dfactor.sum(dim=1)
            + dda_cs.sum(dim=1)
            + dda
            + dgamma_diag_checksum * 0.0
        )

    return out


if _has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _mono_pruned_checksum_kernel(
        Q,
        K,
        K_PRE_TRAP,
        DSTATES,
        STATES,
        DPHI,
        PSIV,
        V,
        MIMO_V,
        D,
        EXP_REV,
        EXP_CS,
        SEGSUM,
        GAMMA,
        QK_DOT,
        SINK,
        B: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        R: tl.constexpr,
        CHUNK: tl.constexpr,
        NCHUNKS: tl.constexpr,
        FCS: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        pid = tl.program_id(0)
        chunk = pid % NCHUNKS
        bh = pid // NCHUNKS
        h = bh % H
        b = bh // H

        offs_t = tl.arange(0, 16)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, BLOCK_P)
        offs_c = tl.arange(0, CHUNK)
        offs_r = tl.arange(0, R)

        qk_base = (((b * H + h) * NCHUNKS + chunk) * FCS) * N
        np_base = (((b * H + h) * NCHUNKS + chunk) * N) * P
        fp_base = (((b * H + h) * NCHUNKS + chunk) * FCS) * P
        exp_base = ((b * H + h) * NCHUNKS + chunk) * CHUNK
        seg_base = (((b * H + h) * NCHUNKS + chunk) * CHUNK) * CHUNK
        qkdot_base = ((((b * H + h) * NCHUNKS + chunk) * CHUNK) * R) * R
        mimo_base = h * R * P

        dstates = tl.load(
            DSTATES + np_base + offs_n[:, None] * P + offs_p[None, :],
            mask=(offs_n[:, None] < N) & (offs_p[None, :] < P),
            other=0.0,
        )
        states = tl.load(
            STATES + np_base + offs_n[:, None] * P + offs_p[None, :],
            mask=(offs_n[:, None] < N) & (offs_p[None, :] < P),
            other=0.0,
        )

        dda = tl.sum(tl.sum(states.to(tl.float32) * dstates.to(tl.float32), axis=0), axis=0)
        checksum = dda

        for rb in tl.static_range(0, 4):
            rows = rb * 16 + offs_t
            row_c = rows // R
            row_r = rows - row_c * R

            q_rows = tl.load(
                Q + qk_base + rows[:, None] * N + offs_n[None, :],
                mask=(rows[:, None] < FCS) & (offs_n[None, :] < N),
                other=0.0,
            )
            k_rows = tl.load(
                K + qk_base + rows[:, None] * N + offs_n[None, :],
                mask=(rows[:, None] < FCS) & (offs_n[None, :] < N),
                other=0.0,
            )
            k_pre_rows = tl.load(
                K_PRE_TRAP + qk_base + rows[:, None] * N + offs_n[None, :],
                mask=(rows[:, None] < FCS) & (offs_n[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            dphi_rows = tl.load(
                DPHI + fp_base + rows[:, None] * P + offs_p[None, :],
                mask=(rows[:, None] < FCS) & (offs_p[None, :] < P),
                other=0.0,
            )
            psiv_rows = tl.load(
                PSIV + fp_base + rows[:, None] * P + offs_p[None, :],
                mask=(rows[:, None] < FCS) & (offs_p[None, :] < P),
                other=0.0,
            )

            exp_rev_rows = tl.load(EXP_REV + exp_base + row_c, mask=rows < FCS, other=0.0)
            exp_cs_rows = tl.load(EXP_CS + exp_base + row_c, mask=rows < FCS, other=0.0)

            state_dpsi = tl.dot(k_rows, dstates, input_precision="tf32", out_dtype=tl.float32)
            dpsi_rows = state_dpsi * exp_rev_rows[:, None]
            dpsi_rows += tl.load(D + h) * dphi_rows.to(tl.float32)

            qk_contrib = tl.zeros((16, BLOCK_P), dtype=tl.float32)
            row_chunk = row_c
            row_lane = row_r
            for ro in tl.static_range(0, 4):
                coeff = tl.load(
                    QK_DOT + qkdot_base + row_chunk[:, None] * R * R + ro * R + row_lane[:, None],
                    mask=rows[:, None] < FCS,
                    other=0.0,
                )
                dphi_ro = tl.load(
                    DPHI + fp_base + (row_chunk[:, None] * R + ro) * P + offs_p[None, :],
                    mask=(rows[:, None] < FCS) & (offs_p[None, :] < P),
                    other=0.0,
                ).to(tl.float32)
                qk_contrib += coeff * dphi_ro
            gamma_rows = tl.load(GAMMA + exp_base + row_c, mask=rows < FCS, other=0.0)
            dpsi_rows += qk_contrib * gamma_rows[:, None]

            dk_state_raw = tl.dot(psiv_rows, tl.trans(dstates), input_precision="tf32", out_dtype=tl.float32)
            checksum += tl.sum(tl.sum(k_rows.to(tl.float32) * dk_state_raw, axis=0), axis=0)
            dk_rows = dk_state_raw * exp_rev_rows[:, None]

            dq_state_rows = tl.dot(dphi_rows, tl.trans(states), input_precision="tf32", out_dtype=tl.float32)
            checksum += tl.sum(tl.sum(q_rows.to(tl.float32) * dq_state_rows, axis=0), axis=0)
            checksum += tl.sum(tl.sum(dq_state_rows * exp_cs_rows[:, None], axis=0), axis=0)
            checksum += tl.sum(tl.sum(dphi_rows.to(tl.float32), axis=0), axis=0)

            for cb in tl.static_range(0, 4):
                cols = cb * 16 + offs_t
                col_c = cols // R

                q_cols = tl.load(
                    Q + qk_base + cols[:, None] * N + offs_n[None, :],
                    mask=(cols[:, None] < FCS) & (offs_n[None, :] < N),
                    other=0.0,
                )
                dphi_cols = tl.load(
                    DPHI + fp_base + cols[:, None] * P + offs_p[None, :],
                    mask=(cols[:, None] < FCS) & (offs_p[None, :] < P),
                    other=0.0,
                )

                lkq_tile = tl.dot(k_rows, tl.trans(q_cols), input_precision="tf32", out_dtype=tl.float32)
                dk_intra_tile = tl.dot(
                    psiv_rows,
                    tl.trans(dphi_cols),
                    input_precision="tf32",
                    out_dtype=tl.float32,
                )
                checksum += tl.sum(tl.sum(lkq_tile * dk_intra_tile, axis=0), axis=0)

                if cb >= rb:
                    seg = tl.load(
                        SEGSUM + seg_base + col_c[None, :] * CHUNK + row_c[:, None],
                        mask=(rows[:, None] < FCS) & (cols[None, :] < FCS),
                        other=0.0,
                    )
                    seg_exp = tl.exp(seg)
                    if cb == rb:
                        causal = row_c[:, None] < col_c[None, :]
                        seg_exp = tl.where(causal, seg_exp, 0.0)

                    lkq_masked = lkq_tile * seg_exp
                    dpsi_rows += tl.dot(
                        lkq_masked.to(dphi_cols.dtype),
                        dphi_cols,
                        input_precision="tf32",
                        out_dtype=tl.float32,
                    )

                    dk_masked = dk_intra_tile * seg_exp
                    dk_rows += tl.dot(
                        dk_masked.to(q_cols.dtype),
                        q_cols,
                        input_precision="tf32",
                        out_dtype=tl.float32,
                    )
                    dq_delta = tl.dot(
                        tl.trans(dk_masked).to(k_rows.dtype),
                        k_rows,
                        input_precision="tf32",
                        out_dtype=tl.float32,
                    )
                    checksum += tl.sum(tl.sum(dq_delta, axis=0), axis=0)

            dpsi_rows = dpsi_rows.to(dphi_rows.dtype).to(tl.float32)
            mimo = tl.load(
                MIMO_V + mimo_base + row_r[:, None] * P + offs_p[None, :],
                mask=(row_r[:, None] < R) & (offs_p[None, :] < P),
                other=0.0,
            ).to(tl.float32)
            s_idx = chunk * CHUNK + row_c
            v = tl.load(
                V + ((b * S + s_idx[:, None]) * H + h) * P + offs_p[None, :],
                mask=(s_idx[:, None] < S) & (offs_p[None, :] < P),
                other=0.0,
            ).to(tl.float32)
            checksum += tl.sum(tl.sum(dpsi_rows * (mimo + v), axis=0), axis=0)
            checksum += tl.sum(tl.sum(dk_rows, axis=0), axis=0)
            checksum += tl.sum(tl.sum(k_pre_rows * dk_rows, axis=0), axis=0)

        tl.store(SINK + pid, checksum)


def mono_pruned_triton_checksum(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
    *,
    block_p: int,
    num_warps: int,
) -> torch.Tensor:
    if not _has_triton():
        raise RuntimeError("triton is not importable")
    if not inputs["q"].is_cuda:
        raise RuntimeError("triton path requires CUDA tensors")
    if shape.chunk != 16 or shape.R != 4 or shape.N != 64:
        raise ValueError("prototype specializes chunk=16, R=4, N=64")
    if block_p != shape.P:
        raise ValueError("tile-pruned checksum requires BLOCK_P == P so dk_intra is not tiled")

    total = shape.B * shape.H * shape.nchunks
    sink = torch.empty(total, device=inputs["q"].device, dtype=torch.float32)
    _mono_pruned_checksum_kernel[(total,)](
        inputs["q"],
        inputs["k"],
        inputs["k_pre_trap"],
        inputs["dstates"],
        inputs["states"],
        inputs["dphi"],
        inputs["psiv"],
        inputs["v"],
        inputs["mimo_v"],
        inputs["D"],
        inputs["exp_rev"],
        inputs["exp_cs"],
        inputs["segsum"],
        inputs["gamma"],
        inputs["qk_dot"],
        sink,
        shape.B,
        shape.S,
        shape.H,
        shape.N,
        shape.P,
        shape.R,
        shape.chunk,
        shape.nchunks,
        shape.fcs,
        block_p,
        num_warps=num_warps,
    )
    return sink


def _fma_model(shape: Shape, *, block_p: int) -> dict[str, Any]:
    base = _wave1_fma_model(shape, block_p=block_p)
    chunks = shape.B * shape.H * shape.nchunks
    fcs = shape.fcs
    tile_steps = 4
    tile_f = tile_steps * shape.R
    ntiles = shape.chunk // tile_steps
    full_entries = fcs * fcs
    tile_pruned_entries = (
        (ntiles * (ntiles - 1) // 2) * tile_f * tile_f
        + ntiles * tile_f * tile_f
    )
    causal_entries = shape.chunk * (shape.chunk - 1) // 2 * shape.R * shape.R
    apply_p_tile_pruned = chunks * tile_pruned_entries * shape.P
    apply_n_tile_pruned = chunks * tile_pruned_entries * shape.N

    tile_pruned = dict(base["monolithic_full_mask_fma"])
    tile_pruned["lkq_apply_to_dphi_full_mask"] = apply_p_tile_pruned
    tile_pruned["dk_intra_apply_to_q_full_mask"] = apply_n_tile_pruned
    tile_pruned["dk_intra_transpose_apply_to_k_full_mask"] = apply_n_tile_pruned
    tile_pruned_total = sum(tile_pruned.values())

    return {
        **base,
        "tile_steps": tile_steps,
        "tile_fused_rows": tile_f,
        "full_entries_per_chunk": full_entries,
        "causal_entries_per_chunk": causal_entries,
        "tile_pruned_entries_per_chunk": tile_pruned_entries,
        "monolithic_tile_pruned_fma": tile_pruned,
        "monolithic_tile_pruned_total_fma": tile_pruned_total,
        "tile_pruned_savings_vs_full_mask_fma": base["monolithic_full_mask_total_fma"] - tile_pruned_total,
        "tile_pruned_savings_vs_full_mask_pct": (
            base["monolithic_full_mask_total_fma"] - tile_pruned_total
        )
        / base["monolithic_full_mask_total_fma"],
        "tile_pruned_over_ideal_causal_fma": tile_pruned_total - base["monolithic_causal_apply_total_fma"],
        "tile_pruned_over_ideal_causal_pct": (
            tile_pruned_total - base["monolithic_causal_apply_total_fma"]
        )
        / base["monolithic_causal_apply_total_fma"],
    }


def _add_rates(timings: dict[str, Any], fma_model: dict[str, Any]) -> None:
    tile_pruned = fma_model["monolithic_tile_pruned_total_fma"]
    ideal_causal = fma_model["monolithic_causal_apply_total_fma"]
    full_mask = fma_model["monolithic_full_mask_total_fma"]
    for item in timings.values():
        if not isinstance(item, dict):
            continue
        mean_ms = item.get("mean_ms")
        if not mean_ms:
            continue
        item["monolithic_tile_pruned_tfma_per_s"] = tile_pruned / (mean_ms / 1000.0) / 1e12
        item["ideal_causal_pruned_ms_at_measured_tile_pruned_rate"] = (
            ideal_causal / (item["monolithic_tile_pruned_tfma_per_s"] * 1e12) * 1000.0
        )
        item["full_mask_ms_at_measured_tile_pruned_rate"] = (
            full_mask / (item["monolithic_tile_pruned_tfma_per_s"] * 1e12) * 1000.0
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = _shape_from_args(args)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    handoff_dtype = _dtype(args.handoff_dtype)
    inputs = make_prepared_inputs(shape, dtype=dtype, device=device, seed=args.seed)
    timer = _time_cuda if device.type == "cuda" else _time_wall

    fma_model = _fma_model(shape, block_p=args.block_p)
    timings: dict[str, Any] = {}
    correctness: dict[str, Any] = {}
    errors: dict[str, str] = {}

    ref = None
    if args.check_torch or device.type != "cuda":
        total = shape.B * shape.H * shape.nchunks
        if total > args.torch_reference_batch:
            ref = mono_reuse_torch_checksum_batched(
                inputs,
                shape,
                handoff_dtype=handoff_dtype,
                batch_chunks=args.torch_reference_batch,
            )
            correctness["torch_checksum_reference"] = "batched"
            correctness["torch_reference_batch_chunks"] = args.torch_reference_batch
        else:
            ref = mono_reuse_torch_checksum(inputs, shape, handoff_dtype=handoff_dtype)
            correctness["torch_checksum_reference"] = "materialized"
        if device.type == "cuda":
            torch.cuda.synchronize()
        correctness["torch_checksum_finite"] = bool(torch.isfinite(ref).all().item())

    if device.type == "cuda":
        try:
            out = mono_pruned_triton_checksum(inputs, shape, block_p=args.block_p, num_warps=args.num_warps)
            torch.cuda.synchronize()
            correctness["triton_pruned_checksum_finite"] = bool(torch.isfinite(out).all().item())
            if ref is not None:
                correctness["triton_pruned_vs_torch_checksum"] = {
                    "max_abs_delta": float((ref.float() - out.float()).abs().max().item()),
                    "mean_abs_delta": float((ref.float() - out.float()).abs().mean().item()),
                    "max_ref_abs": float(ref.float().abs().max().item()),
                }

            timings["triton_mono_tile_pruned_checksum_compute_lower_bound"] = timer(
                lambda: mono_pruned_triton_checksum(inputs, shape, block_p=args.block_p, num_warps=args.num_warps),
                warmup=args.warmup,
                iters=args.iters,
            )
        except BaseException as exc:
            errors["triton_mono_tile_pruned_checksum"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    if args.bench_torch:
        timings["torch_materialized_mono_reuse_checksum"] = timer(
            lambda: mono_reuse_torch_checksum(inputs, shape, handoff_dtype=handoff_dtype),
            warmup=args.torch_warmup,
            iters=args.torch_iters,
        )

    _add_rates(timings, fma_model)

    return {
        "shape_name": args.shape or "custom",
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "handoff_dtype": args.handoff_dtype,
        "torch": torch.__version__,
        "triton_importable": _has_triton(),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "block_p": args.block_p,
        "num_warps": args.num_warps,
        "comparison_context": COMPARISON_CONTEXT,
        "fma_model": fma_model,
        "memory_model": _memory_model(shape),
        "correctness": correctness,
        "timings": timings,
        "errors": errors,
        "read": [
            "This Wave2 kernel uses 16x16 FCS tiles, corresponding to four chunk timesteps by four chunk timesteps.",
            "Tiles below the causal frontier feed only DSSDA; they are not applied to dPsiV, DK, or DQ.",
            "Diagonal tiles are internally masked for correctness, so the measured tile-pruned FMA is above the ideal fully split triangular model.",
            "The checksum timing still omits global DV/DK/DQ/scalar stores; use the memory model for output traffic.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(PRESETS), default=None)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=256)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--G", type=int, default=1)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--P", type=int, default=128)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--handoff-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--block-p", type=int, default=128)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--check-torch", action="store_true")
    parser.add_argument("--bench-torch", action="store_true")
    parser.add_argument("--torch-warmup", type=int, default=1)
    parser.add_argument("--torch-iters", type=int, default=3)
    parser.add_argument("--torch-reference-batch", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
