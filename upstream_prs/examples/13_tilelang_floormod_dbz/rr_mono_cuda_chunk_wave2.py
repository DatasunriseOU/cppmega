"""Wave 2 tensor-core CUDA chunk-owner prototype for Mamba3 bwd_bwd.

This keeps the Wave 1 output contract but replaces the scalar matrix bodies
with CUDA WMMA tiles:

* LKQ = K @ Q.T, materialized once per chunk in the stage2 consumer orientation.
* dki = PsiV @ dPhi.T for DSSDA.
* state = K @ dstates.
* dPsi += masked(LKQ) @ dPhi, then reused for DV and per-chunk DMIMO_V.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rr_diag_cuda_extension import (  # noqa: E402
    stage2_mono_wmma_lkq_dphi_chunk_owner_cuda,
    stage2_mono_wmma_lkq_dphi_chunk_owner_cuda_metadata,
    stage2_mono_wmma_lkq_dphi_chunk_owner_cuda_out,
)
from rr_diag_wave6_inlaunch_cuda import (  # noqa: E402
    PRESETS,
    Shape,
    _dtype,
    _shape_from_args,
    _time_cuda,
    _time_wall,
)
from rr_mono_cuda_chunk_wave1 import (  # noqa: E402
    COMPARISON_CONTEXT,
    _empty_mono_outputs,
    _max_diff,
    make_mono_inputs,
    mono_state_lkq_d_torch_reference,
)


WAVE1_SCALAR_PRODUCTIONISH_MS = 89.02105560302735


def _stage_bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.bfloat16).float()


@torch.no_grad()
def mono_wmma_lkq_dphi_torch_reference(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the exact Wave 2 bf16-staged WMMA contract."""

    if shape.S % shape.chunk != 0:
        raise RuntimeError("Wave 2 reference currently expects S divisible by chunk")
    if shape.N != 64 or shape.R != 4 or shape.P % 16 != 0:
        raise RuntimeError("Wave 2 WMMA path specializes N=64, R=4, and P % 16 == 0")

    bsz, nchunks, chunk, rank = shape.B, shape.nchunks, shape.chunk, shape.R
    fused_chunk = chunk * rank
    device = inputs["dout"].device
    dtype = inputs["dout"].dtype
    row_step = torch.arange(fused_chunk, device=device) // rank
    causal = row_step[:, None] < row_step[None, :]

    q5 = inputs["q_flat"].view(shape.B, shape.S, rank, shape.G, shape.N)
    k5 = inputs["k_flat"].view(shape.B, shape.S, rank, shape.G, shape.N)
    dv = torch.empty(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype)
    dmimo_v_chunk = torch.empty(
        shape.B,
        shape.H,
        nchunks,
        rank,
        shape.P,
        device=device,
        dtype=torch.float32,
    )
    dssda = torch.empty(
        shape.B,
        shape.H,
        nchunks,
        chunk,
        chunk,
        device=device,
        dtype=torch.float32,
    )

    heads_per_group = shape.H // shape.G
    for h in range(shape.H):
        h_qk = h // heads_per_group
        qh = _stage_bf16(q5[:, :, :, h_qk, :].float() + inputs["q_bias"][h].float()[None, None, :, :])
        kh = _stage_bf16(k5[:, :, :, h_qk, :].float() + inputs["k_bias"][h].float()[None, None, :, :])
        q_c = qh.view(bsz, nchunks, chunk, rank, shape.N).reshape(bsz, nchunks, fused_chunk, shape.N)
        k_c = kh.view(bsz, nchunks, chunk, rank, shape.N).reshape(bsz, nchunks, fused_chunk, shape.N)

        dout_ch = inputs["dout"][:, :, h, :].float().view(bsz, nchunks, chunk, shape.P)
        v_ch = inputs["v"][:, :, h, :].float().view(bsz, nchunks, chunk, shape.P)
        dphi = _stage_bf16(
            dout_ch[:, :, :, None, :]
            * inputs["mimo_o"][h].float()[None, None, None, :, :]
        ).reshape(bsz, nchunks, fused_chunk, shape.P)
        psi = _stage_bf16(
            v_ch[:, :, :, None, :]
            * inputs["mimo_v"][h].float()[None, None, None, :, :]
        ).reshape(bsz, nchunks, fused_chunk, shape.P)

        lkq = torch.matmul(k_c, q_c.transpose(-1, -2))
        dst = _stage_bf16(inputs["dstates"][:, h])
        exp_rev = torch.exp(
            inputs["da_cs_rev"][:, h].float().view(bsz, nchunks, chunk).repeat_interleave(rank, dim=2)
        ).unsqueeze(-1)
        dpsi = torch.matmul(k_c, dst) * exp_rev

        seg = inputs["segsum"][:, h].float()[:, :, row_step[:, None], row_step[None, :]]
        lkq_masked = _stage_bf16(torch.where(causal[None, None], lkq * torch.exp(seg), torch.zeros_like(lkq)))
        dpsi = dpsi + torch.matmul(lkq_masked, dphi)
        dpsi = dpsi + inputs["D"][h].float() * dphi

        dpsi_r = dpsi.view(bsz, nchunks, chunk, rank, shape.P)
        dv[:, :, h, :] = (
            dpsi_r * inputs["mimo_v"][h].float()[None, None, None, :, :]
        ).sum(dim=3).reshape(bsz, shape.S, shape.P).to(dtype)
        dmimo_v_chunk[:, h] = (dpsi_r * v_ch[:, :, :, None, :]).sum(dim=2)

        dki = torch.matmul(psi, dphi.transpose(-1, -2))
        dssda[:, h] = (lkq * dki).view(bsz, nchunks, chunk, rank, chunk, rank).sum(dim=(3, 5))

    return dv.contiguous(), dmimo_v_chunk.contiguous(), dssda.contiguous()


def mono_wmma_lkq_dphi_cuda(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return stage2_mono_wmma_lkq_dphi_chunk_owner_cuda(
        dout=inputs["dout"],
        q_flat=inputs["q_flat"],
        k_flat=inputs["k_flat"],
        v=inputs["v"],
        q_bias=inputs["q_bias"],
        k_bias=inputs["k_bias"],
        mimo_v=inputs["mimo_v"],
        mimo_o=inputs["mimo_o"],
        dstates=inputs["dstates"],
        da_cs_rev=inputs["da_cs_rev"],
        segsum=inputs["segsum"],
        D=inputs["D"],
        chunk_size=shape.chunk,
    )


def _diffs(
    ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    got: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    return {
        "dv_delta": _max_diff(ref[0], got[0]),
        "dmimo_v_chunk_delta": _max_diff(ref[1], got[1]),
        "dssda_delta": _max_diff(ref[2], got[2]),
    }


def _cta_model(shape: Shape) -> dict[str, Any]:
    chunk_ctas = shape.B * shape.H * shape.nchunks
    fused_chunk = shape.chunk * shape.R
    return {
        "chunk_owner_ctas": chunk_ctas,
        "chunk_owner_ctas_per_sm_at_132_sms": chunk_ctas / 132.0,
        "fused_rows_per_cta": fused_chunk,
        "wmma_tiles_per_cta": {
            "lkq_k_qt": 16 * (shape.N // 16),
            "dki_psiv_dphi_t": 16 * (shape.P // 16),
            "state_k_dstates": 4 * (shape.P // 16) * (shape.N // 16),
            "lkq_dphi_apply": 4 * (shape.P // 16) * (fused_chunk // 16),
        },
        "outputs_in_one_chunk_cta": ["DV", "DMIMO_V_chunk_partial", "DSSDA"],
        "comparison_ms": {
            "wave1_scalar_mono_state_lkq_d": WAVE1_SCALAR_PRODUCTIONISH_MS,
            **COMPARISON_CONTEXT,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = _shape_from_args(args)
    if shape.S % shape.chunk != 0:
        raise RuntimeError("Wave 2 harness currently expects S divisible by chunk")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    inputs = make_mono_inputs(shape, dtype=dtype, device=device, seed=args.seed)

    correctness: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    timer: Callable[..., dict[str, Any]] = _time_cuda if device.type == "cuda" else _time_wall

    if device.type == "cuda":
        if dtype != torch.bfloat16:
            raise RuntimeError("Wave 2 WMMA CUDA path currently requires --dtype bf16")

        ref = mono_wmma_lkq_dphi_torch_reference(inputs, shape)
        got = mono_wmma_lkq_dphi_cuda(inputs, shape)
        torch.cuda.synchronize()
        correctness["wave2_wmma_lkq_dphi_vs_bf16_staged_torch_reference"] = _diffs(ref, got)

        fp32_ref = mono_state_lkq_d_torch_reference(inputs, shape)
        correctness["wave2_wmma_lkq_dphi_vs_wave1_fp32_reference"] = _diffs(fp32_ref, got)
        metadata["wave2_wmma_lkq_dphi_chunk_owner"] = (
            stage2_mono_wmma_lkq_dphi_chunk_owner_cuda_metadata(inputs["dout"])
        )

        dv_out, dmimo_out, dssda_out = _empty_mono_outputs(shape, dtype=dtype, device=device)

        def run_wmma() -> None:
            stage2_mono_wmma_lkq_dphi_chunk_owner_cuda_out(
                dout=inputs["dout"],
                q_flat=inputs["q_flat"],
                k_flat=inputs["k_flat"],
                v=inputs["v"],
                q_bias=inputs["q_bias"],
                k_bias=inputs["k_bias"],
                mimo_v=inputs["mimo_v"],
                mimo_o=inputs["mimo_o"],
                dstates=inputs["dstates"],
                da_cs_rev=inputs["da_cs_rev"],
                segsum=inputs["segsum"],
                D=inputs["D"],
                dv_delta=dv_out,
                dmimo_v_chunk_delta=dmimo_out,
                dssda_delta=dssda_out,
                chunk_size=shape.chunk,
            )

        timings["wave2_wmma_lkq_dphi_chunk_owner_slice"] = timer(
            run_wmma,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        ref = mono_wmma_lkq_dphi_torch_reference(inputs, shape)
        correctness["torch_bf16_staged_reference_self"] = _diffs(ref, ref)
        timings["torch_bf16_staged_reference_state_lkq_d_subset"] = timer(
            lambda: mono_wmma_lkq_dphi_torch_reference(inputs, shape),
            warmup=args.warmup,
            iters=args.iters,
        )

    wmma_ms = timings.get("wave2_wmma_lkq_dphi_chunk_owner_slice", {}).get("mean_ms")
    if wmma_ms and args.shape == "productionish":
        tilelang = COMPARISON_CONTEXT["tilelang_stage2_bf1_bb0_bwd_bwd_productionish_ms"]
        wave10 = COMPARISON_CONTEXT["wave10_two_launch_wave7_plus_output_owner_dmimo_v_productionish_ms"]
        timings["wave2_wmma_lkq_dphi_chunk_owner_slice"]["speedup_vs_wave1_scalar_mono_prod"] = (
            WAVE1_SCALAR_PRODUCTIONISH_MS / wmma_ms
        )
        timings["wave2_wmma_lkq_dphi_chunk_owner_slice"]["ratio_vs_tilelang_bwd_bwd_prod"] = (
            wmma_ms / tilelang
        )
        timings["wave10_two_launch_plus_wave2_wmma_projection"] = {
            "mean_ms": wave10 + wmma_ms,
            "ratio_vs_tilelang_bwd_bwd_prod": (wave10 + wmma_ms) / tilelang,
            "margin_ms_vs_tilelang_bwd_bwd_prod": tilelang - (wave10 + wmma_ms),
            "note": "Projection only: wave10 covered subset and wave2 state/LKQ/D subset are not fused together yet.",
        }

    return {
        "shape_name": args.shape or "custom",
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "comparison_context": {
            "wave1_scalar_mono_state_lkq_d_productionish_ms": WAVE1_SCALAR_PRODUCTIONISH_MS,
            **COMPARISON_CONTEXT,
        },
        "cta_model": _cta_model(shape),
        "metadata": metadata,
        "correctness": correctness,
        "timings": timings,
        "read": [
            "Wave 2 keeps the Wave 1 chunk-owner output contract but tensorizes LKQ, dki, state, and LKQ@dPhi with WMMA.",
            "The kernel stages Q/K, dPhi, and PsiV through bf16 shared memory; the primary correctness reference mirrors that staged contract.",
            "The stage2 orientation stores LKQ as K @ Q.T, which is the transpose of the shorthand Q @ K.T and matches the existing dPsi consumer.",
            "DMIMO_V remains per-chunk in this wave; a final reduction owner is still needed for full stage replacement.",
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
