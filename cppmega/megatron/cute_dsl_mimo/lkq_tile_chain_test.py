"""CuTe LKQ/state chain probe built from the correct 64x64 BF16 GEMM tile.

This is a bounded scan-owner composition probe, not a production fused kernel.
It keeps Wave 4's scalar-copy CuTe GEMM as the correctness oracle and composes
the chunk-local pieces that the monolithic owner must eventually keep in one
CTA:

  1. state = K @ DStates
  2. lkq = K @ Q^T
  3. apply = future_mask(lkq) @ dPhi
  4. dpsi = state + apply
  5. DV / DMIMO_V scalar consumers from dpsi

The scalar correctness mode keeps the Wave 5 LKQ/masked-LKQ global tensors.
The fused mode uses a two-WGMMA CuTe tile for ``future_mask(lkq) @ dPhi`` so
LKQ is only spilled to swizzled shared memory inside the kernel.
The Wave 7 fused-consumer mode keeps state/apply on-chip as BF16 shared-memory
tiles and writes only DV/DMIMO_V to global memory.  The Wave 8 mode lifts that
same path into a bounded reverse multi-chunk scan owner with a loop-carried
state update.  Wave 9 adds the same-time qk diagonal contribution; Wave 10 adds
dA/segsum/carry scaling semantics to the fused multi-chunk consumers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cuda.bindings.driver as cuda
import torch

from cppmega.megatron.cute_dsl_mimo.masked_lkq_apply import run_masked_lkq_apply
from cppmega.megatron.cute_dsl_mimo.single_gemm_test import run_single_gemm
from cppmega.megatron.cute_dsl_mimo.state_apply_consumers import (
    WAVE10_UINT4_BF16_ELEMENTS,
    WAVE10_UINT4_COPY_BITS,
    WAVE10_UINT4_COPY_BYTES,
    multi_chunk_copy_bits,
    run_multi_chunk_state_apply_consumers,
    run_state_apply_consumers,
)


CHUNK_SIZE = 16
RANK = 4
FCS = CHUNK_SIZE * RANK
N = 64
P = 64


@dataclass(frozen=True)
class ChainInputs:
    name: str
    k: torch.Tensor
    q: torch.Tensor
    dstates: torch.Tensor
    dphi: torch.Tensor
    v: torch.Tensor
    mimo_v: torch.Tensor


@dataclass(frozen=True)
class MultiChunkInputs:
    name: str
    nchunks: int
    k: torch.Tensor
    q: torch.Tensor
    dphi: torch.Tensor
    dA_cs: torch.Tensor
    dA_cs_rev: torch.Tensor
    segsum: torch.Tensor
    qk_dot: torch.Tensor
    gamma: torch.Tensor
    v: torch.Tensor
    mimo_v: torch.Tensor


def _structured(rows: int, cols: int, *, a: int, b: int, mod: int, denom: float) -> torch.Tensor:
    row = torch.arange(rows, dtype=torch.float32, device="cuda")[:, None]
    col = torch.arange(cols, dtype=torch.float32, device="cuda")[None, :]
    return (((row * a + col * b) % mod) - (mod // 2)) / denom


def _structured3(
    chunks: int,
    rows: int,
    cols: int,
    *,
    a: int,
    b: int,
    c: int,
    mod: int,
    denom: float,
) -> torch.Tensor:
    chunk = torch.arange(chunks, dtype=torch.float32, device="cuda")[:, None, None]
    row = torch.arange(rows, dtype=torch.float32, device="cuda")[None, :, None]
    col = torch.arange(cols, dtype=torch.float32, device="cuda")[None, None, :]
    return (((chunk * a + row * b + col * c) % mod) - (mod // 2)) / denom


def _structured_qk(
    chunks: int,
    *,
    a: int,
    b: int,
    c: int,
    d: int,
    mod: int,
    denom: float,
) -> torch.Tensor:
    chunk = torch.arange(chunks, dtype=torch.float32, device="cuda")[:, None, None, None]
    t = torch.arange(CHUNK_SIZE, dtype=torch.float32, device="cuda")[None, :, None, None]
    r_out = torch.arange(RANK, dtype=torch.float32, device="cuda")[None, None, :, None]
    r_in = torch.arange(RANK, dtype=torch.float32, device="cuda")[None, None, None, :]
    return (((chunk * a + t * b + r_out * c + r_in * d) % mod) - (mod // 2)) / denom


def _make_cases(seed: int) -> list[ChainInputs]:
    torch.manual_seed(seed)
    structured = ChainInputs(
        name="structured_mod",
        k=_structured(FCS, N, a=3, b=5, mod=17, denom=8.0).to(torch.bfloat16),
        q=_structured(FCS, N, a=7, b=-2, mod=19, denom=9.0).to(torch.bfloat16),
        dstates=_structured(N, P, a=11, b=3, mod=23, denom=11.0).to(torch.bfloat16),
        dphi=_structured(FCS, P, a=-5, b=13, mod=29, denom=13.0).to(torch.bfloat16),
        v=_structured(CHUNK_SIZE, P, a=2, b=7, mod=31, denom=32.0).to(torch.bfloat16),
        mimo_v=_structured(RANK, P, a=17, b=-3, mod=37, denom=64.0).to(torch.bfloat16),
    )

    scale = 0.125
    random = ChainInputs(
        name="random_seed",
        k=(torch.randn(FCS, N, dtype=torch.bfloat16, device="cuda") * scale),
        q=(torch.randn(FCS, N, dtype=torch.bfloat16, device="cuda") * scale),
        dstates=(torch.randn(N, P, dtype=torch.bfloat16, device="cuda") * scale),
        dphi=(torch.randn(FCS, P, dtype=torch.bfloat16, device="cuda") * scale),
        v=(torch.randn(CHUNK_SIZE, P, dtype=torch.bfloat16, device="cuda") * scale),
        mimo_v=(torch.randn(RANK, P, dtype=torch.bfloat16, device="cuda") * scale),
    )
    return [structured, random]


def _make_multi_cases(seed: int, nchunks: int) -> list[MultiChunkInputs]:
    torch.manual_seed(seed + nchunks)
    structured = MultiChunkInputs(
        name=f"structured_mod_{nchunks}chunks",
        nchunks=nchunks,
        k=_structured3(nchunks, FCS, N, a=5, b=3, c=7, mod=23, denom=16.0).to(torch.bfloat16),
        q=_structured3(nchunks, FCS, N, a=-3, b=11, c=2, mod=29, denom=18.0).to(torch.bfloat16),
        dphi=_structured3(nchunks, FCS, P, a=13, b=-5, c=17, mod=31, denom=24.0).to(torch.bfloat16),
        dA_cs=_structured3(nchunks, CHUNK_SIZE, 1, a=3, b=5, c=0, mod=17, denom=160.0)
        .squeeze(-1)
        .contiguous(),
        dA_cs_rev=_structured3(nchunks, CHUNK_SIZE, 1, a=-5, b=7, c=0, mod=19, denom=192.0)
        .squeeze(-1)
        .contiguous(),
        segsum=_structured3(
            nchunks, CHUNK_SIZE, CHUNK_SIZE, a=7, b=-3, c=5, mod=23, denom=192.0
        ).contiguous(),
        qk_dot=_structured_qk(
            nchunks, a=3, b=-7, c=11, d=5, mod=41, denom=96.0
        ).to(torch.bfloat16),
        gamma=_structured3(nchunks, CHUNK_SIZE, 1, a=5, b=-3, c=0, mod=23, denom=128.0)
        .squeeze(-1)
        .contiguous(),
        v=_structured3(nchunks, CHUNK_SIZE, P, a=7, b=2, c=5, mod=37, denom=64.0).to(torch.bfloat16),
        mimo_v=_structured(RANK, P, a=17, b=-3, mod=37, denom=64.0).to(torch.bfloat16),
    )

    scale = 0.03125
    random = MultiChunkInputs(
        name=f"random_seed_{nchunks}chunks",
        nchunks=nchunks,
        k=(torch.randn(nchunks, FCS, N, dtype=torch.bfloat16, device="cuda") * scale),
        q=(torch.randn(nchunks, FCS, N, dtype=torch.bfloat16, device="cuda") * scale),
        dphi=(torch.randn(nchunks, FCS, P, dtype=torch.bfloat16, device="cuda") * scale),
        dA_cs=(torch.randn(nchunks, CHUNK_SIZE, dtype=torch.float32, device="cuda") * scale),
        dA_cs_rev=(torch.randn(nchunks, CHUNK_SIZE, dtype=torch.float32, device="cuda") * scale),
        segsum=(torch.randn(nchunks, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32, device="cuda") * scale),
        qk_dot=(torch.randn(nchunks, CHUNK_SIZE, RANK, RANK, dtype=torch.bfloat16, device="cuda") * scale),
        gamma=(torch.randn(nchunks, CHUNK_SIZE, dtype=torch.float32, device="cuda") * scale),
        v=(torch.randn(nchunks, CHUNK_SIZE, P, dtype=torch.bfloat16, device="cuda") * scale),
        mimo_v=(torch.randn(RANK, P, dtype=torch.bfloat16, device="cuda") * scale),
    )
    return [structured, random]


def _future_mask_bf16() -> torch.Tensor:
    f = torch.arange(FCS, device="cuda")
    t = f // RANK
    mask = t[:, None] < t[None, :]
    return mask.to(torch.bfloat16)


def _run_tile(a: torch.Tensor, b_nk: torch.Tensor, stream: cuda.CUstream) -> torch.Tensor:
    """Run C = A @ B^T through the Wave 4 scalar-copy CuTe tile."""

    m, k = a.shape
    n, b_k = b_nk.shape
    if k != b_k:
        raise ValueError(f"K mismatch: A={tuple(a.shape)} B={tuple(b_nk.shape)}")
    c = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    run_single_gemm(m, n, k, a.contiguous(), b_nk.contiguous(), c, stream)
    return c


def _scalar_consumers(
    dpsi: torch.Tensor,
    v: torch.Tensor,
    mimo_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dpsi_t = dpsi.float().view(CHUNK_SIZE, RANK, P)
    dv = (dpsi_t * mimo_v.float()[None, :, :]).sum(dim=1)
    dmimo_v = (dpsi_t * v.float()[:, None, :]).sum(dim=0)
    return dv, dmimo_v


def _qk_diag_contrib(
    qk_dot: torch.Tensor,
    gamma: torch.Tensor,
    dphi: torch.Tensor,
) -> torch.Tensor:
    dphi_r = dphi.float().view(CHUNK_SIZE, RANK, P)
    contrib = torch.einsum("toi,top->tip", qk_dot.float(), dphi_r)
    contrib = contrib * gamma.float()[:, None, None]
    return contrib.reshape(FCS, P)


def _scaled_future_lkq(
    k: torch.Tensor,
    q: torch.Tensor,
    segsum: torch.Tensor,
) -> torch.Tensor:
    idx = torch.arange(FCS, device=k.device)
    row_t = idx // RANK
    col_t = idx // RANK
    future = row_t[:, None] < col_t[None, :]
    seg_exp = torch.exp(segsum[col_t[None, :], row_t[:, None]])
    lkq = k.float() @ q.float().T
    return (lkq * future.float() * seg_exp).to(torch.bfloat16)


def _scaled_dphi_for_carry(
    dphi: torch.Tensor,
    dA_cs: torch.Tensor,
) -> torch.Tensor:
    exp_cs = torch.exp(dA_cs).repeat_interleave(RANK)[:, None]
    return (dphi.float() * exp_cs).to(torch.bfloat16).float()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _reference(inputs: ChainInputs, mask_bf16: torch.Tensor) -> dict[str, torch.Tensor]:
    state = (inputs.k.float() @ inputs.dstates.float()).to(torch.bfloat16)
    lkq = (inputs.k.float() @ inputs.q.float().T).to(torch.bfloat16)
    masked_lkq = (lkq.float() * mask_bf16.float()).to(torch.bfloat16)
    apply = (masked_lkq.float() @ inputs.dphi.float()).to(torch.bfloat16)
    dpsi = state.float() + apply.float()
    dv, dmimo_v = _scalar_consumers(dpsi, inputs.v, inputs.mimo_v)
    ideal_dpsi = inputs.k.float() @ inputs.dstates.float()
    ideal_dpsi += ((inputs.k.float() @ inputs.q.float().T) * mask_bf16.float()) @ inputs.dphi.float()
    return {
        "state": state,
        "lkq": lkq,
        "masked_lkq": masked_lkq,
        "apply": apply,
        "dpsi": dpsi,
        "dv": dv,
        "dmimo_v": dmimo_v,
        "ideal_dpsi": ideal_dpsi,
    }


def _multi_reference(
    inputs: MultiChunkInputs,
    mask_bf16: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dv = torch.empty((inputs.nchunks, CHUNK_SIZE, P), dtype=torch.float32, device="cuda")
    dmimo_v = torch.zeros((RANK, P), dtype=torch.float32, device="cuda")
    carry_t = torch.zeros((P, N), dtype=torch.float32, device="cuda")
    dpsi_checksums: list[float] = []

    for c in range(inputs.nchunks - 1, -1, -1):
        carry_t_bf = carry_t.to(torch.bfloat16).float()
        exp_rev = torch.exp(inputs.dA_cs_rev[c]).repeat_interleave(RANK)[:, None]
        state = ((inputs.k[c].float() @ carry_t_bf.T) * exp_rev).to(torch.bfloat16)
        masked_lkq = _scaled_future_lkq(inputs.k[c], inputs.q[c], inputs.segsum[c])
        apply = (masked_lkq.float() @ inputs.dphi[c].float()).to(torch.bfloat16)
        qk_contrib = _qk_diag_contrib(inputs.qk_dot[c], inputs.gamma[c], inputs.dphi[c])
        dpsi = state.float() + apply.float() + qk_contrib
        dv_c, dmimo_c = _scalar_consumers(dpsi, inputs.v[c], inputs.mimo_v)
        dv[c] = dv_c
        dmimo_v += dmimo_c
        dpsi_checksums.append(float(dpsi.float().sum().item()))

        # The CuTe prototype carries DStates.T directly so the next state GEMM
        # can consume it as the K-major B operand.
        carry_t = carry_t * torch.exp(inputs.dA_cs[c, -1])
        carry_t += _scaled_dphi_for_carry(inputs.dphi[c], inputs.dA_cs[c]).T @ inputs.q[c].float()

    return {
        "dv": dv,
        "dmimo_v": dmimo_v,
        "carry_t": carry_t,
        "dpsi_checksums_rev": dpsi_checksums,
        "qk_diag_checksum": float(
            sum(
                _qk_diag_contrib(inputs.qk_dot[c], inputs.gamma[c], inputs.dphi[c]).sum()
                for c in range(inputs.nchunks)
            ).item()
        ),
    }


def _run_case_scalar(
    inputs: ChainInputs,
    mask_bf16: torch.Tensor,
    stream: cuda.CUstream,
) -> dict[str, Any]:
    state = _run_tile(inputs.k, inputs.dstates.T.contiguous(), stream)
    lkq = _run_tile(inputs.k, inputs.q, stream)
    torch.cuda.synchronize()

    masked_lkq = (lkq.float() * mask_bf16.float()).to(torch.bfloat16)
    apply = _run_tile(masked_lkq, inputs.dphi.T.contiguous(), stream)
    torch.cuda.synchronize()

    dpsi = state.float() + apply.float()
    dv, dmimo_v = _scalar_consumers(dpsi, inputs.v, inputs.mimo_v)
    ref = _reference(inputs, mask_bf16)

    diffs = {
        "state": _max_abs(state, ref["state"]),
        "lkq": _max_abs(lkq, ref["lkq"]),
        "masked_lkq": _max_abs(masked_lkq, ref["masked_lkq"]),
        "apply": _max_abs(apply, ref["apply"]),
        "dpsi": _max_abs(dpsi, ref["dpsi"]),
        "dv": _max_abs(dv, ref["dv"]),
        "dmimo_v": _max_abs(dmimo_v, ref["dmimo_v"]),
    }
    return {
        "mode": "scalar_copy_cute_tiles_plus_torch_mask",
        "lkq_global_materialized": True,
        "diffs": diffs,
        "ideal_dpsi_max_abs": _max_abs(dpsi, ref["ideal_dpsi"]),
        "lkq_checksum": float(lkq.float().sum().item()),
        "dpsi_checksum": float(dpsi.float().sum().item()),
        "dpsi_row0": [float(x) for x in dpsi[0, :4].tolist()],
        "ref_dpsi_row0": [float(x) for x in ref["dpsi"][0, :4].tolist()],
    }


def _run_case_fused_masked_apply(
    inputs: ChainInputs,
    mask_bf16: torch.Tensor,
    stream: cuda.CUstream,
) -> dict[str, Any]:
    state = _run_tile(inputs.k, inputs.dstates.T.contiguous(), stream)
    apply = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    run_masked_lkq_apply(
        FCS,
        RANK,
        inputs.k.contiguous(),
        inputs.q.contiguous(),
        inputs.dphi.T.contiguous(),
        apply,
        stream,
    )
    torch.cuda.synchronize()

    dpsi = state.float() + apply.float()
    dv, dmimo_v = _scalar_consumers(dpsi, inputs.v, inputs.mimo_v)
    ref = _reference(inputs, mask_bf16)

    diffs = {
        "state": _max_abs(state, ref["state"]),
        "apply": _max_abs(apply, ref["apply"]),
        "dpsi": _max_abs(dpsi, ref["dpsi"]),
        "dv": _max_abs(dv, ref["dv"]),
        "dmimo_v": _max_abs(dmimo_v, ref["dmimo_v"]),
    }
    return {
        "mode": "state_scalar_plus_fused_masked_lkq_apply",
        "lkq_global_materialized": False,
        "remaining_global": [
            "state BF16 tile from scalar-copy CuTe GEMM",
            "apply BF16 tile output from fused masked-apply kernel",
            "dpsi/DV/DMIMO_V torch-side scalar correctness consumers",
        ],
        "diffs": diffs,
        "ideal_dpsi_max_abs": _max_abs(dpsi, ref["ideal_dpsi"]),
        "dpsi_checksum": float(dpsi.float().sum().item()),
        "dpsi_row0": [float(x) for x in dpsi[0, :4].tolist()],
        "ref_dpsi_row0": [float(x) for x in ref["dpsi"][0, :4].tolist()],
    }


def _run_case_fused_state_apply_consumers(
    inputs: ChainInputs,
    mask_bf16: torch.Tensor,
    stream: cuda.CUstream,
) -> dict[str, Any]:
    dv = torch.empty((CHUNK_SIZE, P), dtype=torch.float32, device="cuda")
    dmimo_v = torch.empty((RANK, P), dtype=torch.float32, device="cuda")
    run_state_apply_consumers(
        FCS,
        RANK,
        CHUNK_SIZE,
        inputs.k.contiguous(),
        inputs.q.contiguous(),
        inputs.dstates.T.contiguous(),
        inputs.dphi.T.contiguous(),
        inputs.v.contiguous(),
        inputs.mimo_v.contiguous(),
        dv,
        dmimo_v,
        stream,
    )
    torch.cuda.synchronize()

    ref = _reference(inputs, mask_bf16)
    diffs = {
        "dv": _max_abs(dv, ref["dv"]),
        "dmimo_v": _max_abs(dmimo_v, ref["dmimo_v"]),
    }
    return {
        "mode": "fused_state_apply_consumers",
        "lkq_global_materialized": False,
        "state_global_materialized": False,
        "apply_global_materialized": False,
        "remaining_global": [
            "DV FP32 output tile",
            "DMIMO_V FP32 output tile",
            "pre-transposed DStates.T/DPh.T input-layout tensors in the harness",
        ],
        "diffs": diffs,
        "dv_checksum": float(dv.float().sum().item()),
        "dmimo_v_checksum": float(dmimo_v.float().sum().item()),
        "dv_row0": [float(x) for x in dv[0, :4].tolist()],
        "ref_dv_row0": [float(x) for x in ref["dv"][0, :4].tolist()],
        "dmimo_v_row0": [float(x) for x in dmimo_v[0, :4].tolist()],
        "ref_dmimo_v_row0": [float(x) for x in ref["dmimo_v"][0, :4].tolist()],
    }


def _run_case_multi_chunk_fused(
    inputs: MultiChunkInputs,
    mask_bf16: torch.Tensor,
    stream: cuda.CUstream,
) -> dict[str, Any]:
    dv = torch.empty((inputs.nchunks, CHUNK_SIZE, P), dtype=torch.float32, device="cuda")
    dmimo_v = torch.empty((RANK, P), dtype=torch.float32, device="cuda")
    q_t = inputs.q.transpose(-1, -2).contiguous()
    dphi_t = inputs.dphi.transpose(-1, -2).contiguous()
    run_multi_chunk_state_apply_consumers(
        FCS,
        RANK,
        CHUNK_SIZE,
        inputs.k.contiguous(),
        inputs.q.contiguous(),
        q_t,
        dphi_t,
        inputs.dA_cs.contiguous(),
        inputs.dA_cs_rev.contiguous(),
        inputs.segsum.contiguous(),
        inputs.qk_dot.contiguous(),
        inputs.gamma.contiguous(),
        inputs.v.contiguous(),
        inputs.mimo_v.contiguous(),
        dv,
        dmimo_v,
        stream,
    )
    torch.cuda.synchronize()

    ref = _multi_reference(inputs, mask_bf16)
    diffs = {
        "dv": _max_abs(dv, ref["dv"]),
        "dmimo_v": _max_abs(dmimo_v, ref["dmimo_v"]),
    }
    return {
        "mode": "multi_chunk_fused_state_apply_consumers",
        "nchunks": inputs.nchunks,
        "lkq_global_materialized": False,
        "state_global_materialized": False,
        "apply_global_materialized": False,
        "dpsi_global_materialized": False,
        "loop_carried_state": "carry_t = exp(dA_last) * carry_t + BF16(dPhi * exp(dA)).T @ Q in FP32 registers",
        "dA_state_scaling": "state = BF16((K @ BF16(carry).T) * exp(dA_cs_rev[t]))",
        "segsum_lkq_scaling": "masked LKQ = BF16((K @ Q.T) * exp(segsum[col_t,row_t])) for row_t < col_t",
        "qk_diag_contribution": "dpsi[t,r,p] += gamma[t] * sum_o qk_dot[t,o,r] * dPhi[t,o,p]",
        "remaining_global": [
            "DV FP32 output tensor for all chunks",
            "DMIMO_V FP32 output tile",
            "Harness input-layout tensors: Q.T and DPh.T",
        ],
        "diffs": diffs,
        "dv_checksum": float(dv.float().sum().item()),
        "dmimo_v_checksum": float(dmimo_v.float().sum().item()),
        "ref_dv_checksum": float(ref["dv"].float().sum().item()),
        "ref_dmimo_v_checksum": float(ref["dmimo_v"].float().sum().item()),
        "dv_chunk0_row0": [float(x) for x in dv[0, 0, :4].tolist()],
        "ref_dv_chunk0_row0": [float(x) for x in ref["dv"][0, 0, :4].tolist()],
        "dmimo_v_row0": [float(x) for x in dmimo_v[0, :4].tolist()],
        "ref_dmimo_v_row0": [float(x) for x in ref["dmimo_v"][0, :4].tolist()],
        "dpsi_checksums_rev": ref["dpsi_checksums_rev"],
        "qk_diag_checksum": ref["qk_diag_checksum"],
    }


def _time_chain(
    inputs: ChainInputs,
    mask_bf16: torch.Tensor,
    stream: cuda.CUstream,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    state = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    lkq = torch.empty((FCS, FCS), dtype=torch.bfloat16, device="cuda")
    masked_lkq = torch.empty((FCS, FCS), dtype=torch.bfloat16, device="cuda")
    apply = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    dstates_t = inputs.dstates.T.contiguous()
    dphi_t = inputs.dphi.T.contiguous()

    def launch_once() -> None:
        run_single_gemm(FCS, P, N, inputs.k, dstates_t, state, stream)
        run_single_gemm(FCS, FCS, N, inputs.k, inputs.q, lkq, stream)
        # Keep this benchmark path conservative: the torch mask is a temporary
        # stand-in for an eventual in-kernel masked epilogue, so place explicit
        # CUDA barriers around the framework/CuTe handoff.
        torch.cuda.synchronize()
        torch.mul(lkq, mask_bf16, out=masked_lkq)
        torch.cuda.synchronize()
        run_single_gemm(FCS, P, FCS, masked_lkq, dphi_t, apply, stream)

    for _ in range(warmup):
        launch_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch_once()
    end.record()
    torch.cuda.synchronize()

    elapsed_us = start.elapsed_time(end) * 1000.0
    per_iter_us = elapsed_us / iters
    flops = 2 * FCS * P * N + 2 * FCS * FCS * N + 2 * FCS * P * FCS
    return {
        "mode": "three_scalar_copy_cute_tiles_plus_torch_mask",
        "warmup": warmup,
        "iters": iters,
        "chain_us": per_iter_us,
        "estimated_tile_flops": flops,
        "estimated_tile_tflops": float(flops / (per_iter_us * 1e-6) / 1e12),
    }


def _time_chain_fused_masked_apply(
    inputs: ChainInputs,
    stream: cuda.CUstream,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    state = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    apply = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    dstates_t = inputs.dstates.T.contiguous()
    dphi_t = inputs.dphi.T.contiguous()

    def launch_once() -> None:
        run_single_gemm(FCS, P, N, inputs.k, dstates_t, state, stream)
        run_masked_lkq_apply(FCS, RANK, inputs.k, inputs.q, dphi_t, apply, stream)

    for _ in range(warmup):
        launch_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch_once()
    end.record()
    torch.cuda.synchronize()

    elapsed_us = start.elapsed_time(end) * 1000.0
    per_iter_us = elapsed_us / iters
    flops = 2 * FCS * P * N + 2 * FCS * FCS * N + 2 * FCS * P * FCS
    return {
        "mode": "state_scalar_plus_fused_masked_lkq_apply",
        "warmup": warmup,
        "iters": iters,
        "chain_us": per_iter_us,
        "estimated_tile_flops": flops,
        "estimated_tile_tflops": float(flops / (per_iter_us * 1e-6) / 1e12),
        "lkq_global_materialized": False,
        "launches_per_chain": 2,
    }


def _time_chain_fused_state_apply_consumers(
    inputs: ChainInputs,
    stream: cuda.CUstream,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    dv = torch.empty((CHUNK_SIZE, P), dtype=torch.float32, device="cuda")
    dmimo_v = torch.empty((RANK, P), dtype=torch.float32, device="cuda")
    dstates_t = inputs.dstates.T.contiguous()
    dphi_t = inputs.dphi.T.contiguous()

    def launch_once() -> None:
        run_state_apply_consumers(
            FCS,
            RANK,
            CHUNK_SIZE,
            inputs.k,
            inputs.q,
            dstates_t,
            dphi_t,
            inputs.v,
            inputs.mimo_v,
            dv,
            dmimo_v,
            stream,
        )

    for _ in range(warmup):
        launch_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch_once()
    end.record()
    torch.cuda.synchronize()

    elapsed_us = start.elapsed_time(end) * 1000.0
    per_iter_us = elapsed_us / iters
    gemm_flops = 2 * FCS * P * N + 2 * FCS * FCS * N + 2 * FCS * P * FCS
    consumer_flops = 2 * CHUNK_SIZE * RANK * P + 2 * RANK * CHUNK_SIZE * P
    flops = gemm_flops + consumer_flops
    return {
        "mode": "fused_state_apply_consumers",
        "warmup": warmup,
        "iters": iters,
        "chain_us": per_iter_us,
        "estimated_tile_flops": flops,
        "estimated_tile_tflops": float(flops / (per_iter_us * 1e-6) / 1e12),
        "lkq_global_materialized": False,
        "state_global_materialized": False,
        "apply_global_materialized": False,
        "launches_per_chain": 1,
    }


def _time_chain_multi_chunk_fused(
    inputs: MultiChunkInputs,
    stream: cuda.CUstream,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    dv = torch.empty((inputs.nchunks, CHUNK_SIZE, P), dtype=torch.float32, device="cuda")
    dmimo_v = torch.empty((RANK, P), dtype=torch.float32, device="cuda")
    q_t = inputs.q.transpose(-1, -2).contiguous()
    dphi_t = inputs.dphi.transpose(-1, -2).contiguous()

    def launch_once() -> None:
        run_multi_chunk_state_apply_consumers(
            FCS,
            RANK,
            CHUNK_SIZE,
            inputs.k,
            inputs.q,
            q_t,
            dphi_t,
            inputs.dA_cs,
            inputs.dA_cs_rev,
            inputs.segsum,
            inputs.qk_dot,
            inputs.gamma,
            inputs.v,
            inputs.mimo_v,
            dv,
            dmimo_v,
            stream,
        )

    for _ in range(warmup):
        launch_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch_once()
    end.record()
    torch.cuda.synchronize()

    elapsed_us = start.elapsed_time(end) * 1000.0
    per_scan_us = elapsed_us / iters
    gemm_flops_per_chunk = 4 * (2 * FCS * P * N)
    consumer_flops_per_chunk = 2 * CHUNK_SIZE * RANK * P + 2 * RANK * CHUNK_SIZE * P
    flops = inputs.nchunks * (gemm_flops_per_chunk + consumer_flops_per_chunk)
    return {
        "mode": "multi_chunk_fused_state_apply_consumers",
        "nchunks": inputs.nchunks,
        "warmup": warmup,
        "iters": iters,
        "scan_us": per_scan_us,
        "per_chunk_us": per_scan_us / inputs.nchunks,
        "estimated_tile_flops": flops,
        "estimated_tile_tflops": float(flops / (per_scan_us * 1e-6) / 1e12),
        "lkq_global_materialized": False,
        "state_global_materialized": False,
        "apply_global_materialized": False,
        "dpsi_global_materialized": False,
        "launches_per_scan": 1,
    }


def run_lkq_tile_chain(
    *,
    seed: int = 20260430,
    atol: float = 1e-5,
    bench_iters: int = 100,
    bench_warmup: int = 10,
    multi_chunk_counts: tuple[int, ...] = (2, 4, 8),
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CuTe LKQ tile chain probe")

    print("Wave 10: CuTe LKQ/state chain with bounded multi-chunk scan owner + qk/dA scaling")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: chunk={CHUNK_SIZE} rank={RANK} fcs={FCS} N={N} P={P}")
    print(f"atol: {atol}")
    active_multi_copy_bits = multi_chunk_copy_bits()
    uint4_enabled = active_multi_copy_bits == WAVE10_UINT4_COPY_BITS
    copy_strategy = (
        "wave10_uint4_128bit_g2s_multi_chunk_opt_in"
        if uint4_enabled
        else "scalar_bf16_universal_g2s_s2g"
    )
    print(f"copy_strategy: {copy_strategy}")
    print(
        "wave10_copy_contract: "
        f"{WAVE10_UINT4_COPY_BITS} bits, {WAVE10_UINT4_COPY_BYTES} bytes, "
        f"{WAVE10_UINT4_BF16_ELEMENTS} BF16 elements per lane"
    )
    print("wave6_fused_path: LKQ is R2S-spilled to swizzled smem only, no LKQ gmem output")
    print("wave7_fused_path: state/apply stay in swizzled smem, DV/DMIMO_V computed in-kernel")
    print(f"wave8_multi_chunk_counts: {list(multi_chunk_counts)}")
    print("wave9_multi_chunk_semantics: same-time qk_dot/gamma contribution included in fused consumers")
    print("wave10_multi_chunk_semantics: dA_cs_rev state scale, segsum LKQ scale, scaled carry update")

    mask_bf16 = _future_mask_bf16()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Compile the shared 64x64x64 tile before collecting case diagnostics.
    t0 = time.time()
    first = _make_cases(seed)[0]
    warm = torch.empty((FCS, FCS), dtype=torch.bfloat16, device="cuda")
    run_single_gemm(FCS, FCS, N, first.k, first.q, warm, stream)
    fused_warm = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    run_masked_lkq_apply(
        FCS,
        RANK,
        first.k.contiguous(),
        first.q.contiguous(),
        first.dphi.T.contiguous(),
        fused_warm,
        stream,
    )
    warm_dv = torch.empty((CHUNK_SIZE, P), dtype=torch.float32, device="cuda")
    warm_dmimo_v = torch.empty((RANK, P), dtype=torch.float32, device="cuda")
    run_state_apply_consumers(
        FCS,
        RANK,
        CHUNK_SIZE,
        first.k.contiguous(),
        first.q.contiguous(),
        first.dstates.T.contiguous(),
        first.dphi.T.contiguous(),
        first.v.contiguous(),
        first.mimo_v.contiguous(),
        warm_dv,
        warm_dmimo_v,
        stream,
    )
    torch.cuda.synchronize()
    compile_s = time.time() - t0
    print(f"compile_plus_first_lkq_fused_apply_and_consumer_launch_s: {compile_s:.3f}")

    scalar_cases: dict[str, Any] = {}
    fused_cases: dict[str, Any] = {}
    fused_consumer_cases: dict[str, Any] = {}
    multi_chunk_cases: dict[str, Any] = {}
    passed = True
    for case in _make_cases(seed):
        scalar_result = _run_case_scalar(case, mask_bf16, stream)
        scalar_cases[case.name] = scalar_result
        scalar_pass = all(value <= atol for value in scalar_result["diffs"].values())
        passed = passed and scalar_pass
        print(f"Scalar case {case.name}: {'PASS' if scalar_pass else 'FAIL'}")
        for name, value in scalar_result["diffs"].items():
            print(f"  {name}: max_abs={value:.6f}")
        print(f"  dpsi_row0[:4]: {scalar_result['dpsi_row0']}")
        print(f"  ref_row0[:4]:  {scalar_result['ref_dpsi_row0']}")
        print(f"  ideal_dpsi_bf16_chain_delta={scalar_result['ideal_dpsi_max_abs']:.6f}")

        fused_result = _run_case_fused_masked_apply(case, mask_bf16, stream)
        fused_cases[case.name] = fused_result
        fused_pass = all(value <= atol for value in fused_result["diffs"].values())
        passed = passed and fused_pass
        print(f"Fused masked-apply case {case.name}: {'PASS' if fused_pass else 'FAIL'}")
        for name, value in fused_result["diffs"].items():
            print(f"  {name}: max_abs={value:.6f}")
        print(f"  lkq_global_materialized: {fused_result['lkq_global_materialized']}")
        print(f"  dpsi_row0[:4]: {fused_result['dpsi_row0']}")
        print(f"  ref_row0[:4]:  {fused_result['ref_dpsi_row0']}")
        print(f"  ideal_dpsi_bf16_chain_delta={fused_result['ideal_dpsi_max_abs']:.6f}")

        consumer_result = _run_case_fused_state_apply_consumers(case, mask_bf16, stream)
        fused_consumer_cases[case.name] = consumer_result
        consumer_pass = all(value <= atol for value in consumer_result["diffs"].values())
        passed = passed and consumer_pass
        print(f"Fused state/apply consumer case {case.name}: {'PASS' if consumer_pass else 'FAIL'}")
        for name, value in consumer_result["diffs"].items():
            print(f"  {name}: max_abs={value:.6f}")
        print(f"  lkq_global_materialized: {consumer_result['lkq_global_materialized']}")
        print(f"  state_global_materialized: {consumer_result['state_global_materialized']}")
        print(f"  apply_global_materialized: {consumer_result['apply_global_materialized']}")
        print(f"  dv_row0[:4]: {consumer_result['dv_row0']}")
        print(f"  ref_dv_row0[:4]:  {consumer_result['ref_dv_row0']}")

    multi_t0 = time.time()
    for nchunks in multi_chunk_counts:
        for case in _make_multi_cases(seed, nchunks):
            multi_result = _run_case_multi_chunk_fused(case, mask_bf16, stream)
            multi_chunk_cases[case.name] = multi_result
            multi_pass = all(value <= atol for value in multi_result["diffs"].values())
            passed = passed and multi_pass
            print(
                f"Multi-chunk fused scan case {case.name}: "
                f"{'PASS' if multi_pass else 'FAIL'}"
            )
            for name, value in multi_result["diffs"].items():
                print(f"  {name}: max_abs={value:.6f}")
            print(f"  loop_carried_state: {multi_result['loop_carried_state']}")
            print(f"  dA_state_scaling: {multi_result['dA_state_scaling']}")
            print(f"  segsum_lkq_scaling: {multi_result['segsum_lkq_scaling']}")
            print(f"  qk_diag_contribution: {multi_result['qk_diag_contribution']}")
            print(f"  dv_chunk0_row0[:4]: {multi_result['dv_chunk0_row0']}")
            print(f"  ref_chunk0_row0[:4]: {multi_result['ref_dv_chunk0_row0']}")
            print(f"  dmimo_v_row0[:4]: {multi_result['dmimo_v_row0']}")
            print(f"  ref_dmimo_v_row0[:4]: {multi_result['ref_dmimo_v_row0']}")
    multi_compile_s = time.time() - multi_t0
    print(f"multi_chunk_compile_plus_correctness_s: {multi_compile_s:.3f}")

    timings = None
    if bench_iters > 0:
        timing_case = _make_cases(seed)[-1]
        scalar_timing = _time_chain(
            timing_case,
            mask_bf16,
            stream,
            warmup=bench_warmup,
            iters=bench_iters,
        )
        fused_timing = _time_chain_fused_masked_apply(
            timing_case,
            stream,
            warmup=bench_warmup,
            iters=bench_iters,
        )
        fused_consumer_timing = _time_chain_fused_state_apply_consumers(
            timing_case,
            stream,
            warmup=bench_warmup,
            iters=bench_iters,
        )
        timings = {
            "scalar_copy": scalar_timing,
            "fused_masked_apply": fused_timing,
            "fused_state_apply_consumers": fused_consumer_timing,
            "multi_chunk_fused": {},
        }
        for nchunks in multi_chunk_counts:
            multi_timing_case = _make_multi_cases(seed, nchunks)[-1]
            multi_timing = _time_chain_multi_chunk_fused(
                multi_timing_case,
                stream,
                warmup=bench_warmup,
                iters=bench_iters,
            )
            timings["multi_chunk_fused"][nchunks] = multi_timing
            print(
                f"Multi-chunk fused timing ({nchunks} chunks): "
                f"{multi_timing['scan_us']:.3f} us/scan "
                f"{multi_timing['per_chunk_us']:.3f} us/chunk "
                f"({bench_iters} iters)"
            )
        print(
            f"Scalar timing: {scalar_timing['chain_us']:.3f} us/chain "
            f"({bench_iters} iters, includes torch mask)"
        )
        print(
            f"Fused masked-apply timing: {fused_timing['chain_us']:.3f} us/chain "
            f"({bench_iters} iters, no LKQ gmem/torch mask)"
        )
        print(
            f"Fused state/apply consumer timing: {fused_consumer_timing['chain_us']:.3f} us/chain "
            f"({bench_iters} iters, no LKQ/state/apply gmem outputs)"
        )
        print(
            "Estimated tile throughput: "
            f"scalar={scalar_timing['estimated_tile_tflops']:.4f} TFLOP/s "
            f"fused_apply={fused_timing['estimated_tile_tflops']:.4f} TFLOP/s "
            f"fused_consumers={fused_consumer_timing['estimated_tile_tflops']:.4f} TFLOP/s"
        )

    print(f"{'PASS' if passed else 'FAIL'}: LKQ/state chain correctness")
    return {
        "passed": bool(passed),
        "atol": atol,
        "compile_plus_first_lkq_fused_apply_and_consumer_launch_s": compile_s,
        "copy_strategy": copy_strategy,
        "wave10_copy_contract": {
            "enabled": uint4_enabled,
            "active_multi_chunk_copy_bits": active_multi_copy_bits,
            "copy_bits": WAVE10_UINT4_COPY_BITS,
            "copy_bytes": WAVE10_UINT4_COPY_BYTES,
            "bf16_elements_per_copy": WAVE10_UINT4_BF16_ELEMENTS,
            "tile_rows": FCS,
            "tile_cols": N,
            "row_bytes": N * 2,
            "vectors_per_row": (N * 2) // WAVE10_UINT4_COPY_BYTES,
            "vectors_per_tile": (FCS * N * 2) // WAVE10_UINT4_COPY_BYTES,
            "guard_source": "mamba3-mono-triton-model commit 65ef653 Wave10 copy evidence",
        },
        "lkq_global_materialized_for_tested_fused_path": False,
        "state_apply_global_materialized_for_fused_consumer_path": False,
        "fused_path_remaining_global": [
            "Wave6 path: state BF16 tile from scalar-copy CuTe GEMM",
            "Wave6 path: apply BF16 tile output from fused masked-apply kernel",
            "Wave7 path: DV and DMIMO_V final FP32 output tiles only",
            "Wave10 path: DV FP32 tensor for all chunks and accumulated DMIMO_V FP32 tile only",
            "Harness input-layout tensors: DStates.T and DPh.T",
            "Wave8 harness input-layout tensors: Q.T and DPh.T",
        ],
        "multi_chunk_counts_tested": list(multi_chunk_counts),
        "shape": {
            "chunk": CHUNK_SIZE,
            "rank": RANK,
            "fcs": FCS,
            "N": N,
            "P": P,
        },
        "scalar_cases": scalar_cases,
        "fused_cases": fused_cases,
        "fused_consumer_cases": fused_consumer_cases,
        "multi_chunk_cases": multi_chunk_cases,
        "multi_chunk_compile_plus_correctness_s": multi_compile_s,
        "timings": timings,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_lkq_tile_chain(), indent=2, sort_keys=True))
