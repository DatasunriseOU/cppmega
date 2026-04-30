"""Wave1 monolithic reuse model for Mamba3 MIMO bwd_bwd.

This probe is intentionally owner-level, not another isolated output slice.
It models the chunk body that has to be fused if a CUDA/CuTe rewrite is going
to beat the current TileLang kernel:

* ``LKQ = K @ Q.T`` is built once and feeds both ``dPsiV`` and ``DSSDA``.
* ``dk_intra = PsiV @ dPhiO.T`` is built once and feeds ``DGAMMA_DIAG``,
  ``DSSDA``, ``DK`` and ``DQ``.
* state products feed ``DV``, ``DMIMO_V``, ``DK/DQ`` and scalar reductions in
  one owner model.

The Triton path is a checksum microkernel.  It executes the monolithic tensor
algebra in one program per ``(B, H, chunk)`` and stores one checksum per
program, so its timing is a compute/reuse lower bound rather than a complete
drop-in kernel.  The operation and memory model reports the missing output
traffic separately.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class Shape:
    B: int
    S: int
    H: int
    G: int
    N: int
    P: int
    R: int = 4
    chunk: int = 16

    @property
    def nchunks(self) -> int:
        return math.ceil(self.S / self.chunk)

    @property
    def fcs(self) -> int:
        return self.chunk * self.R


PRESETS: dict[str, dict[str, int]] = {
    "smoke": {"B": 1, "S": 256, "H": 4, "G": 1, "N": 64, "P": 128},
    "productionish": {"B": 4, "S": 4096, "H": 32, "G": 1, "N": 64, "P": 128},
}

COMPARISON_CONTEXT: dict[str, Any] = {
    "wave7_diag_qk_dv_ms": 1.91459,
    "wave8_qk_dmimov_output_owner_ms": 0.53634,
    "wave8_combined_before_state_lkq_d_ms": 2.45093,
    "wave10_state_lkq_d_best_ms": 2.86062,
    "wave10_state_lkq_d_best_executed_fma": 42_949_672_960,
    "wave10_state_lkq_d_useful_fma": 29_259_776_000,
    "tilelang_stage2_bf1_bb0_bwd_bwd_ms": 3.70674,
}


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _shape_from_args(args: argparse.Namespace) -> Shape:
    if args.shape:
        values = PRESETS[args.shape]
        return Shape(
            B=values["B"],
            S=values["S"],
            H=values["H"],
            G=values["G"],
            N=values["N"],
            P=values["P"],
            R=args.R,
            chunk=args.chunk,
        )
    return Shape(B=args.B, S=args.S, H=args.H, G=args.G, N=args.N, P=args.P, R=args.R, chunk=args.chunk)


def _stats(values_ms: list[float]) -> dict[str, Any]:
    ordered = sorted(values_ms)
    if not ordered:
        return {"count": 0}
    mean = sum(ordered) / len(ordered)
    var = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean_ms": mean,
        "min_ms": ordered[0],
        "p50_ms": ordered[len(ordered) // 2],
        "max_ms": ordered[-1],
        "std_ms": math.sqrt(var),
        "samples_ms": values_ms,
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


def _randn(
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    *size: int,
    scale: float = 0.01,
) -> torch.Tensor:
    return (torch.randn(size, device=device, dtype=dtype, generator=generator) * scale).contiguous()


def make_prepared_inputs(shape: Shape, *, dtype: torch.dtype, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    fcs = shape.fcs
    return {
        "q": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, fcs, shape.N),
        "k": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, fcs, shape.N),
        "k_pre_trap": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, fcs, shape.N),
        "dstates": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, shape.N, shape.P),
        "states": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, shape.N, shape.P),
        "dphi": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, fcs, shape.P),
        "psiv": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, fcs, shape.P),
        "v": _randn(generator, device, dtype, shape.B, shape.S, shape.H, shape.P),
        "mimo_v": _randn(generator, device, torch.float32, shape.H, shape.R, shape.P),
        "D": _randn(generator, device, torch.float32, shape.H),
        "exp_rev": torch.exp(
            _randn(generator, device, torch.float32, shape.B, shape.H, shape.nchunks, shape.chunk, scale=0.01)
        ).contiguous(),
        "exp_cs": torch.exp(
            _randn(generator, device, torch.float32, shape.B, shape.H, shape.nchunks, shape.chunk, scale=0.01)
        ).contiguous(),
        "segsum": _randn(
            generator, device, torch.float32, shape.B, shape.H, shape.nchunks, shape.chunk, shape.chunk, scale=0.01
        ),
        "gamma": _randn(generator, device, torch.float32, shape.B, shape.H, shape.nchunks, shape.chunk, scale=0.01),
        "qk_dot": _randn(generator, device, torch.float32, shape.B, shape.H, shape.nchunks, shape.chunk, shape.R, shape.R),
    }


def _indices(shape: Shape, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offs = torch.arange(shape.fcs, device=device)
    ci = offs // shape.R
    causal = ci[:, None] < ci[None, :]
    return ci, causal, offs


@torch.no_grad()
def mono_reuse_torch_checksum(inputs: dict[str, torch.Tensor], shape: Shape, *, handoff_dtype: torch.dtype) -> torch.Tensor:
    """Materialized torch model of the monolithic chunk-owner algebra.

    The output is one checksum per ``(B,H,chunk)`` program.  It is useful for
    correctness checks and for measuring how expensive global materialization
    would be, but it is not an implementation candidate.
    """

    total = shape.B * shape.H * shape.nchunks
    fcs = shape.fcs
    q = inputs["q"].reshape(total, fcs, shape.N)
    k = inputs["k"].reshape(total, fcs, shape.N)
    k_pre_trap = inputs["k_pre_trap"].reshape(total, fcs, shape.N)
    dstates = inputs["dstates"].reshape(total, shape.N, shape.P)
    states = inputs["states"].reshape(total, shape.N, shape.P)
    dphi = inputs["dphi"].reshape(total, fcs, shape.P)
    psiv = inputs["psiv"].reshape(total, fcs, shape.P)

    ci, causal, _ = _indices(shape, q.device)
    seg = inputs["segsum"].reshape(total, shape.chunk, shape.chunk)
    seg_weight = seg[:, ci[None, :].expand(fcs, fcs), ci[:, None].expand(fcs, fcs)]
    seg_exp = torch.exp(seg_weight)

    exp_rev = inputs["exp_rev"].reshape(total, shape.chunk).repeat_interleave(shape.R, dim=-1).unsqueeze(-1)
    exp_cs = inputs["exp_cs"].reshape(total, shape.chunk).repeat_interleave(shape.R, dim=-1).unsqueeze(-1)

    lkq = torch.bmm(k, q.transpose(1, 2))
    lkq_masked = torch.where(causal[None], lkq.float() * seg_exp, torch.zeros_like(lkq.float()))
    state_dpsi = torch.bmm(k, dstates).float() * exp_rev
    lkq_dpsi = torch.bmm(lkq_masked.to(handoff_dtype), dphi).float()

    h_ids = ((torch.arange(total, device=q.device) // shape.nchunks) % shape.H).long()
    d_h = inputs["D"][h_ids].float().view(total, 1, 1)
    dpsi = state_dpsi + lkq_dpsi + d_h * dphi.float()

    qk = inputs["qk_dot"].reshape(total, shape.chunk, shape.R, shape.R)
    qk_t = qk.transpose(-1, -2).float()
    dphi_r = dphi.reshape(total, shape.chunk, shape.R, shape.P).float()
    qk_contrib = torch.einsum("bcio,bcop->bcip", qk_t, dphi_r)
    gamma = inputs["gamma"].reshape(total, shape.chunk).float()
    dpsi = dpsi + (qk_contrib * gamma[:, :, None, None]).reshape(total, fcs, shape.P)
    dpsi = dpsi.to(handoff_dtype).float()

    dpsi_r = dpsi.reshape(total, shape.chunk, shape.R, shape.P)
    mimo = inputs["mimo_v"][h_ids].float()
    dv_checksum = (dpsi_r * mimo[:, None, :, :]).sum(dim=2).sum(dim=(1, 2))

    v = inputs["v"].reshape(shape.B, shape.nchunks, shape.chunk, shape.H, shape.P)
    v = v.permute(0, 3, 1, 2, 4).reshape(total, shape.chunk, shape.P).float()
    dmimo_checksum = (dpsi_r * v[:, :, None, :]).sum(dim=1).sum(dim=(1, 2))
    dd_checksum = dphi.float().sum(dim=(1, 2))

    dk_state = torch.bmm(psiv, dstates.transpose(1, 2)).float()
    dda_cs_rev = (k.float() * dk_state).reshape(total, shape.chunk, shape.R, shape.N).sum(dim=(2, 3))
    dk_state = dk_state * exp_rev

    dk_intra = torch.bmm(psiv, dphi.transpose(1, 2))
    dqk_diag = dk_intra.transpose(1, 2).reshape(total, shape.chunk, shape.R, shape.chunk, shape.R)
    dqk_diag = dqk_diag.diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2).contiguous()
    dgamma_diag = (inputs["qk_dot"].reshape(total, shape.chunk, shape.R, shape.R).float() * dqk_diag.float()).sum(
        dim=(2, 3)
    )
    # Keep this in the materialized path so the torch model exercises the
    # DGAMMA_DIAG reuse source; the Triton checksum omits this scalar from its
    # parity checksum while the FMA model accounts for it separately.
    dgamma_diag_checksum = dgamma_diag.sum(dim=1)

    dssda = (lkq.float() * dk_intra.float()).reshape(total, shape.chunk, shape.R, shape.chunk, shape.R).sum(dim=(2, 4))
    dk_intra_masked = torch.where(causal[None], dk_intra.float() * seg_exp, torch.zeros_like(dk_intra.float()))

    dk_nodiag = dk_state + torch.bmm(dk_intra_masked.to(handoff_dtype), q).float()
    dfactor = (k_pre_trap.float() * dk_nodiag).reshape(total, shape.chunk, shape.R, shape.N).sum(dim=(2, 3))

    dq_state = torch.bmm(dphi, states.transpose(1, 2)).float()
    dda_cs = (q.float() * dq_state).reshape(total, shape.chunk, shape.R, shape.N).sum(dim=(2, 3))
    dq = dq_state * exp_cs + torch.bmm(dk_intra_masked.transpose(1, 2).to(handoff_dtype), k).float()

    dda = (states.float() * dstates.float()).sum(dim=(1, 2))

    return (
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


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        return False
    return True


if _has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _mono_reuse_checksum_kernel(
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

        offs_f = tl.arange(0, FCS)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, BLOCK_P)
        offs_c = tl.arange(0, CHUNK)
        offs_r = tl.arange(0, R)
        row_c = offs_f // R
        col_c = row_c

        qk_base = (((b * H + h) * NCHUNKS + chunk) * FCS) * N
        np_base = (((b * H + h) * NCHUNKS + chunk) * N) * P
        fp_base = (((b * H + h) * NCHUNKS + chunk) * FCS) * P
        exp_base = ((b * H + h) * NCHUNKS + chunk) * CHUNK
        seg_base = (((b * H + h) * NCHUNKS + chunk) * CHUNK) * CHUNK
        qkdot_base = ((((b * H + h) * NCHUNKS + chunk) * CHUNK) * R) * R
        mimo_base = h * R * P

        q = tl.load(
            Q + qk_base + offs_f[:, None] * N + offs_n[None, :],
            mask=(offs_f[:, None] < FCS) & (offs_n[None, :] < N),
            other=0.0,
        )
        k = tl.load(
            K + qk_base + offs_f[:, None] * N + offs_n[None, :],
            mask=(offs_f[:, None] < FCS) & (offs_n[None, :] < N),
            other=0.0,
        )
        k_pre_trap = tl.load(
            K_PRE_TRAP + qk_base + offs_f[:, None] * N + offs_n[None, :],
            mask=(offs_f[:, None] < FCS) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
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
        dphi = tl.load(
            DPHI + fp_base + offs_f[:, None] * P + offs_p[None, :],
            mask=(offs_f[:, None] < FCS) & (offs_p[None, :] < P),
            other=0.0,
        )
        psiv = tl.load(
            PSIV + fp_base + offs_f[:, None] * P + offs_p[None, :],
            mask=(offs_f[:, None] < FCS) & (offs_p[None, :] < P),
            other=0.0,
        )

        exp_rev = tl.load(EXP_REV + exp_base + row_c, mask=offs_f < FCS, other=0.0)
        exp_cs = tl.load(EXP_CS + exp_base + row_c, mask=offs_f < FCS, other=0.0)
        causal = row_c[:, None] < col_c[None, :]
        seg = tl.load(
            SEGSUM + seg_base + col_c[None, :] * CHUNK + row_c[:, None],
            mask=causal,
            other=0.0,
        )
        seg_exp = tl.exp(seg)

        state_dpsi = tl.dot(k, dstates, input_precision="tf32", out_dtype=tl.float32) * exp_rev[:, None]
        lkq = tl.dot(k, tl.trans(q), input_precision="tf32", out_dtype=tl.float32)
        lkq_masked = tl.where(causal, lkq * seg_exp, 0.0)
        dpsi = state_dpsi + tl.dot(lkq_masked.to(dphi.dtype), dphi, input_precision="tf32", out_dtype=tl.float32)
        dpsi += tl.load(D + h) * dphi.to(tl.float32)

        qk_contrib = tl.zeros((CHUNK, R, BLOCK_P), dtype=tl.float32)
        for ro in tl.static_range(0, 4):
            coeff = tl.load(
                QK_DOT + qkdot_base + offs_c[:, None] * R * R + ro * R + offs_r[None, :],
                mask=(offs_c[:, None] < CHUNK) & (offs_r[None, :] < R),
                other=0.0,
            )
            dphi_ro = tl.load(
                DPHI + fp_base + (offs_c[:, None] * R + ro) * P + offs_p[None, :],
                mask=(offs_c[:, None] < CHUNK) & (offs_p[None, :] < P),
                other=0.0,
            ).to(tl.float32)
            qk_contrib += coeff[:, :, None] * dphi_ro[:, None, :]
        gamma = tl.load(GAMMA + exp_base + offs_c, mask=offs_c < CHUNK, other=0.0)
        dpsi += tl.reshape(qk_contrib * gamma[:, None, None], (FCS, BLOCK_P))
        dpsi = dpsi.to(dphi.dtype).to(tl.float32)

        dpsi_crp = tl.reshape(dpsi, (CHUNK, R, BLOCK_P))
        mimo = tl.load(
            MIMO_V + mimo_base + offs_r[None, :, None] * P + offs_p[None, None, :],
            mask=offs_p[None, None, :] < P,
            other=0.0,
        ).to(tl.float32)
        dv_tile = tl.sum(dpsi_crp * mimo, axis=1)

        s_idx = chunk * CHUNK + offs_c
        v = tl.load(
            V + ((b * S + s_idx[:, None]) * H + h) * P + offs_p[None, :],
            mask=(s_idx[:, None] < S) & (offs_p[None, :] < P),
            other=0.0,
        ).to(tl.float32)
        dmimo_tile = tl.sum(dpsi_crp * v[:, None, :], axis=0)
        dd = tl.sum(tl.sum(dphi.to(tl.float32), axis=0), axis=0)

        dk_state = tl.dot(psiv, tl.trans(dstates), input_precision="tf32", out_dtype=tl.float32)
        dda_cs_rev = tl.sum(tl.sum(k.to(tl.float32) * dk_state, axis=0), axis=0)
        dk_state = dk_state * exp_rev[:, None]

        dk_intra = tl.dot(psiv, tl.trans(dphi), input_precision="tf32", out_dtype=tl.float32)
        dssda = tl.sum(tl.sum(lkq * dk_intra, axis=0), axis=0)
        dk_intra_masked = tl.where(causal, dk_intra * seg_exp, 0.0)
        dk_nodiag = dk_state + tl.dot(dk_intra_masked.to(q.dtype), q, input_precision="tf32", out_dtype=tl.float32)
        dfactor = tl.sum(tl.sum(k_pre_trap * dk_nodiag, axis=0), axis=0)

        dq_state = tl.dot(dphi, tl.trans(states), input_precision="tf32", out_dtype=tl.float32)
        dda_cs = tl.sum(tl.sum(q.to(tl.float32) * dq_state, axis=0), axis=0)
        dq = dq_state * exp_cs[:, None] + tl.dot(
            tl.trans(dk_intra_masked).to(k.dtype),
            k,
            input_precision="tf32",
            out_dtype=tl.float32,
        )

        dda = tl.sum(tl.sum(states.to(tl.float32) * dstates.to(tl.float32), axis=0), axis=0)

        checksum = tl.sum(tl.sum(dv_tile, axis=0), axis=0)
        checksum += tl.sum(tl.sum(dmimo_tile, axis=0), axis=0)
        checksum += dd
        checksum += tl.sum(tl.sum(dk_nodiag, axis=0), axis=0)
        checksum += tl.sum(tl.sum(dq, axis=0), axis=0)
        checksum += dda_cs_rev + dssda + dfactor + dda_cs + dda
        tl.store(SINK + pid, checksum)


def mono_reuse_triton_checksum(
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
    if block_p < shape.P:
        raise ValueError("monolithic checksum requires one full P tile so LKQ is not recomputed")

    total = shape.B * shape.H * shape.nchunks
    sink = torch.empty(total, device=inputs["q"].device, dtype=torch.float32)
    _mono_reuse_checksum_kernel[(total,)](
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
    chunks = shape.B * shape.H * shape.nchunks
    fcs = shape.fcs
    causal_entries = shape.chunk * (shape.chunk - 1) // 2 * shape.R * shape.R
    pblocks = math.ceil(shape.P / block_p)

    state = chunks * fcs * shape.N * shape.P
    lkq_full = chunks * fcs * shape.N * fcs
    lkq_causal = chunks * causal_entries * shape.N
    apply_p_full = chunks * fcs * fcs * shape.P
    apply_p_causal = chunks * causal_entries * shape.P
    apply_n_full = chunks * fcs * fcs * shape.N
    apply_n_causal = chunks * causal_entries * shape.N
    dk_state = chunks * fcs * shape.P * shape.N
    dk_intra = chunks * fcs * shape.P * fcs
    dq_state = chunks * fcs * shape.P * shape.N
    qk_dpsi = chunks * shape.chunk * shape.R * shape.R * shape.P
    r_reduce = chunks * shape.chunk * shape.R * shape.P
    scalar_small = chunks * (
        3 * fcs * shape.N
        + 2 * fcs * shape.P
        + fcs * fcs
        + shape.N * shape.P
        + shape.chunk * shape.R * shape.R
    )

    wave10_style_state_lkq = state + pblocks * lkq_full + apply_p_full
    separate = {
        "wave10_style_state_lkq_d": wave10_style_state_lkq,
        "qk_dpsi_for_dv": qk_dpsi,
        "qk_dpsi_recomputed_for_dmimov": qk_dpsi,
        "dv_and_dmimov_r_reductions": 2 * r_reduce,
        "dk_state": dk_state,
        "dk_intra": dk_intra,
        "dk_intra_apply_to_q_full_mask": apply_n_full,
        "dq_state": dq_state,
        "dk_intra_transpose_apply_to_k_full_mask": apply_n_full,
        "lkq_recomputed_for_dssda_if_separate": lkq_full,
        "dqk_diag_recomputed_for_dgamma_if_separate": qk_dpsi,
        "scalar_elementwise_reductions": scalar_small,
    }
    mono_full_mask = {
        "state_dpsi": state,
        "lkq_once": lkq_full,
        "lkq_apply_to_dphi_full_mask": apply_p_full,
        "qk_dpsi_once_for_dv_and_dmimov": qk_dpsi,
        "dv_and_dmimov_r_reductions": 2 * r_reduce,
        "dk_state": dk_state,
        "dk_intra_once_for_dgamma_dssda_dk_dq": dk_intra,
        "dk_intra_apply_to_q_full_mask": apply_n_full,
        "dq_state": dq_state,
        "dk_intra_transpose_apply_to_k_full_mask": apply_n_full,
        "scalar_elementwise_reductions": scalar_small,
    }
    mono_causal_apply = dict(mono_full_mask)
    mono_causal_apply["lkq_apply_to_dphi_full_mask"] = apply_p_causal
    mono_causal_apply["dk_intra_apply_to_q_full_mask"] = apply_n_causal
    mono_causal_apply["dk_intra_transpose_apply_to_k_full_mask"] = apply_n_causal

    separate_total = sum(separate.values())
    mono_full_total = sum(mono_full_mask.values())
    mono_causal_total = sum(mono_causal_apply.values())
    return {
        "chunks": chunks,
        "fused_chunk_size": fcs,
        "causal_entries_per_chunk": causal_entries,
        "pblocks": pblocks,
        "block_p": block_p,
        "separate_recompute_fma": separate,
        "separate_recompute_total_fma": separate_total,
        "monolithic_full_mask_fma": mono_full_mask,
        "monolithic_full_mask_total_fma": mono_full_total,
        "monolithic_causal_apply_fma": mono_causal_apply,
        "monolithic_causal_apply_total_fma": mono_causal_total,
        "reuse_savings_full_mask_fma": separate_total - mono_full_total,
        "reuse_savings_full_mask_pct": (separate_total - mono_full_total) / separate_total,
        "causal_apply_savings_vs_separate_fma": separate_total - mono_causal_total,
        "causal_apply_savings_vs_separate_pct": (separate_total - mono_causal_total) / separate_total,
        "wave10_isolated_state_lkq_d_executed_fma": wave10_style_state_lkq,
        "wave10_isolated_state_lkq_d_useful_causal_fma": state + lkq_causal + apply_p_causal,
    }


def _memory_model(shape: Shape) -> dict[str, Any]:
    dtype_bytes = 2
    fp32 = 4
    B, S, H, P, N, R = shape.B, shape.S, shape.H, shape.P, shape.N, shape.R
    chunks = B * H * shape.nchunks
    fcs = shape.fcs

    dv_bytes = B * S * H * P * dtype_bytes
    dk_bytes = B * S * R * H * N * dtype_bytes
    dq_bytes = dk_bytes
    dmimo_partial_bytes = B * H * shape.nchunks * R * P * fp32
    dmimo_output_bytes = B * H * R * P * fp32
    scalar_bhs = B * H * S * fp32
    dssda_bytes = B * H * shape.nchunks * shape.chunk * shape.chunk * fp32
    dangles_bytes = B * S * H * (N // 4) * fp32

    global_outputs = (
        dv_bytes
        + dk_bytes
        + dq_bytes
        + dmimo_partial_bytes
        + dmimo_output_bytes
        + 5 * scalar_bhs
        + dssda_bytes
        + dangles_bytes
    )
    reducer_rw = 2 * dmimo_partial_bytes + dmimo_output_bytes

    dpsi_temp = chunks * fcs * P * dtype_bytes
    lkq_temp = chunks * fcs * fcs * fp32
    dk_intra_temp = lkq_temp
    state_temp = chunks * fcs * P * fp32
    dk_temp = chunks * fcs * N * fp32
    dq_temp = dk_temp

    return {
        "monolithic_required_output_write_mib": global_outputs / (1024**2),
        "dv_output_mib": dv_bytes / (1024**2),
        "dk_output_mib": dk_bytes / (1024**2),
        "dq_output_mib": dq_bytes / (1024**2),
        "dmimov_partial_mib": dmimo_partial_bytes / (1024**2),
        "dmimov_reduce_extra_rw_mib": reducer_rw / (1024**2),
        "scalar_outputs_mib_including_dssda_dangles": (5 * scalar_bhs + dssda_bytes + dangles_bytes) / (1024**2),
        "global_temps_avoided_by_cuda_owner_mib": {
            "dpsi_bf16_temp": dpsi_temp / (1024**2),
            "lkq_fp32_temp": lkq_temp / (1024**2),
            "dk_intra_fp32_temp": dk_intra_temp / (1024**2),
            "state_dpsi_fp32_temp": state_temp / (1024**2),
            "dk_fp32_temp": dk_temp / (1024**2),
            "dq_fp32_temp": dq_temp / (1024**2),
        },
    }


def _add_rates(timings: dict[str, Any], fma_model: dict[str, Any]) -> None:
    executed = fma_model["monolithic_full_mask_total_fma"]
    causal = fma_model["monolithic_causal_apply_total_fma"]
    for item in timings.values():
        if not isinstance(item, dict):
            continue
        mean_ms = item.get("mean_ms")
        if not mean_ms:
            continue
        item["monolithic_full_mask_tfma_per_s"] = executed / (mean_ms / 1000.0) / 1e12
        item["monolithic_causal_apply_tfma_per_s_if_pruned"] = causal / (mean_ms / 1000.0) / 1e12


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
        ref = mono_reuse_torch_checksum(inputs, shape, handoff_dtype=handoff_dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
        correctness["torch_checksum_finite"] = bool(torch.isfinite(ref).all().item())

    if device.type == "cuda":
        try:
            out = mono_reuse_triton_checksum(inputs, shape, block_p=args.block_p, num_warps=args.num_warps)
            torch.cuda.synchronize()
            correctness["triton_checksum_finite"] = bool(torch.isfinite(out).all().item())
            if ref is not None:
                correctness["triton_vs_torch_checksum"] = {
                    "max_abs_delta": float((ref.float() - out.float()).abs().max().item()),
                    "mean_abs_delta": float((ref.float() - out.float()).abs().mean().item()),
                    "max_ref_abs": float(ref.float().abs().max().item()),
                }

            timings["triton_mono_reuse_checksum_compute_lower_bound"] = timer(
                lambda: mono_reuse_triton_checksum(inputs, shape, block_p=args.block_p, num_warps=args.num_warps),
                warmup=args.warmup,
                iters=args.iters,
            )
        except BaseException as exc:  # Keep the cost model useful if Triton cannot compile this owner body.
            errors["triton_mono_reuse_checksum"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

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
            "The Triton checksum kernel owns one full P tile per (B,H,chunk), so LKQ is computed once rather than once per P tile.",
            "LKQ feeds dPsiV and DSSDA; dk_intra feeds DGAMMA_DIAG, DSSDA, DK, and DQ.",
            "The checksum timing omits global DV/DK/DQ/scalar stores; use the memory model for output traffic.",
            "The model still pays full masked dot products unless a CUDA/CuTe kernel prunes triangular work.",
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
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
