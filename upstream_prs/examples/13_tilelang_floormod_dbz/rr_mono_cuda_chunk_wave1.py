"""Wave 1 monolithic chunk-owner CUDA prototype for Mamba3 bwd_bwd.

This harness starts the missing state/LKQ/D side of the custom CUDA path.  One
CTA owns one ``(B, H, chunk)`` tile, materializes ``LKQ = K @ Q.T`` once, builds
a state/LKQ/D ``dPsiV`` tile in shared memory, and reuses it for ``DV`` plus a
per-chunk ``DMIMO_V`` contribution.  The same live ``LKQ`` tile also feeds the
``DSSDA`` scalar-family output.
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
    stage2_mono_state_lkq_d_chunk_owner_cuda,
    stage2_mono_state_lkq_d_chunk_owner_cuda_metadata,
    stage2_mono_state_lkq_d_chunk_owner_cuda_out,
)
from rr_diag_wave6_inlaunch_cuda import (  # noqa: E402
    PRESETS,
    Shape,
    _dtype,
    _randn,
    _shape_from_args,
    _time_cuda,
    _time_wall,
    make_inputs,
)


COMPARISON_CONTEXT: dict[str, Any] = {
    "wave10_two_launch_wave7_plus_output_owner_dmimo_v_productionish_ms": 2.09673,
    "wave10_one_launch_combined_productionish_ms": 2.31212,
    "tilelang_stage2_bf1_bb0_bwd_bwd_productionish_ms": 3.70674,
}


def make_mono_inputs(
    shape: Shape,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    inputs = make_inputs(shape, dtype=dtype, device=device, seed=seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1009)
    inputs["dstates"] = _randn(
        generator,
        device,
        dtype,
        shape.B,
        shape.H,
        shape.nchunks,
        shape.N,
        shape.P,
        scale=0.01,
    )
    inputs["da_cs_rev"] = _randn(
        generator,
        device,
        torch.float32,
        shape.B,
        shape.H,
        shape.S,
        scale=0.02,
    )
    inputs["segsum"] = _randn(
        generator,
        device,
        torch.float32,
        shape.B,
        shape.H,
        shape.nchunks,
        shape.chunk,
        shape.chunk,
        scale=0.02,
    )
    inputs["D"] = _randn(generator, device, torch.float32, shape.H, scale=0.01)
    return inputs


def _max_diff(ref: torch.Tensor, got: torch.Tensor) -> float:
    return float((ref.float() - got.float()).abs().max().item())


@torch.no_grad()
def mono_state_lkq_d_torch_reference(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the Wave 1 state/LKQ/D subset.

    The reference intentionally mirrors the prototype's simplified preprocessed
    Q/K contract: Q and K are the stage2 flattened inputs plus per-head bias.
    Rotary/trap preprocessing is outside this first chunk-owner slice.
    """

    if shape.S % shape.chunk != 0:
        raise RuntimeError("Wave 1 reference currently expects S divisible by chunk")

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
        qh = q5[:, :, :, h_qk, :].float() + inputs["q_bias"][h].float()[None, None, :, :]
        kh = k5[:, :, :, h_qk, :].float() + inputs["k_bias"][h].float()[None, None, :, :]
        q_c = qh.view(bsz, nchunks, chunk, rank, shape.N).reshape(bsz, nchunks, fused_chunk, shape.N)
        k_c = kh.view(bsz, nchunks, chunk, rank, shape.N).reshape(bsz, nchunks, fused_chunk, shape.N)

        dout_ch = inputs["dout"][:, :, h, :].float().view(bsz, nchunks, chunk, shape.P)
        v_ch = inputs["v"][:, :, h, :].float().view(bsz, nchunks, chunk, shape.P)
        dphi = (
            dout_ch[:, :, :, None, :]
            * inputs["mimo_o"][h].float()[None, None, None, :, :]
        ).reshape(bsz, nchunks, fused_chunk, shape.P)
        psi = (
            v_ch[:, :, :, None, :]
            * inputs["mimo_v"][h].float()[None, None, None, :, :]
        ).reshape(bsz, nchunks, fused_chunk, shape.P)

        lkq = torch.matmul(k_c, q_c.transpose(-1, -2))
        dst = inputs["dstates"][:, h].float()
        exp_rev = torch.exp(
            inputs["da_cs_rev"][:, h].float().view(bsz, nchunks, chunk).repeat_interleave(rank, dim=2)
        ).unsqueeze(-1)
        dpsi = torch.matmul(k_c, dst) * exp_rev

        seg = inputs["segsum"][:, h].float()[:, :, row_step[:, None], row_step[None, :]]
        lkq_masked = torch.where(causal[None, None], lkq * torch.exp(seg), torch.zeros_like(lkq))
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


def mono_state_lkq_d_cuda(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return stage2_mono_state_lkq_d_chunk_owner_cuda(
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


def _empty_mono_outputs(
    shape: Shape,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dv_delta = torch.empty(shape.B, shape.S, shape.H, shape.P, dtype=dtype, device=device)
    dmimo_v_chunk_delta = torch.empty(
        shape.B,
        shape.H,
        shape.nchunks,
        shape.R,
        shape.P,
        dtype=torch.float32,
        device=device,
    )
    dssda_delta = torch.empty(
        shape.B,
        shape.H,
        shape.nchunks,
        shape.chunk,
        shape.chunk,
        dtype=torch.float32,
        device=device,
    )
    return dv_delta, dmimo_v_chunk_delta, dssda_delta


def _cta_model(shape: Shape) -> dict[str, Any]:
    chunk_ctas = shape.B * shape.H * shape.nchunks
    fused_chunk = shape.chunk * shape.R
    return {
        "chunk_owner_ctas": chunk_ctas,
        "chunk_owner_ctas_per_sm_at_132_sms": chunk_ctas / 132.0,
        "timesteps_per_cta": shape.chunk,
        "fused_rows_per_cta": fused_chunk,
        "live_intermediates": {
            "lkq_elements": fused_chunk * fused_chunk,
            "dpsi_tile_elements": fused_chunk * int(os.environ.get("RR_DIAG_MONO_P_TILE", "32")),
            "dssda_elements": shape.chunk * shape.chunk,
        },
        "outputs_in_one_chunk_cta": ["DV", "DMIMO_V_chunk_partial", "DSSDA"],
        "work_per_cta_flops_approx": {
            "lkq": 2 * fused_chunk * fused_chunk * shape.N,
            "state_dpsi_per_full_P": 2 * fused_chunk * shape.N * shape.P,
            "lkq_dpsi_per_full_P": 2 * fused_chunk * fused_chunk * shape.P,
            "dssda_dki": 2 * fused_chunk * fused_chunk * shape.P,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = _shape_from_args(args)
    if shape.S % shape.chunk != 0:
        raise RuntimeError("Wave 1 harness currently expects S divisible by chunk")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    inputs = make_mono_inputs(shape, dtype=dtype, device=device, seed=args.seed)

    correctness: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    timer: Callable[..., dict[str, Any]] = _time_cuda if device.type == "cuda" else _time_wall

    if device.type == "cuda":
        ref = mono_state_lkq_d_torch_reference(inputs, shape)
        got = mono_state_lkq_d_cuda(inputs, shape)
        torch.cuda.synchronize()
        correctness["wave1_mono_state_lkq_d_vs_torch_reference"] = {
            "dv_delta": _max_diff(ref[0], got[0]),
            "dmimo_v_chunk_delta": _max_diff(ref[1], got[1]),
            "dssda_delta": _max_diff(ref[2], got[2]),
        }
        metadata["wave1_mono_state_lkq_d_chunk_owner"] = (
            stage2_mono_state_lkq_d_chunk_owner_cuda_metadata(inputs["dout"])
        )

        dv_out, dmimo_out, dssda_out = _empty_mono_outputs(shape, dtype=dtype, device=device)

        def run_mono() -> None:
            stage2_mono_state_lkq_d_chunk_owner_cuda_out(
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

        timings["wave1_mono_state_lkq_d_chunk_owner_slice"] = timer(
            run_mono,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        ref = mono_state_lkq_d_torch_reference(inputs, shape)
        correctness["torch_reference_self"] = {
            "dv_delta": _max_diff(ref[0], ref[0]),
            "dmimo_v_chunk_delta": _max_diff(ref[1], ref[1]),
            "dssda_delta": _max_diff(ref[2], ref[2]),
        }
        timings["torch_reference_state_lkq_d_subset"] = timer(
            lambda: mono_state_lkq_d_torch_reference(inputs, shape),
            warmup=args.warmup,
            iters=args.iters,
        )

    mono_ms = timings.get("wave1_mono_state_lkq_d_chunk_owner_slice", {}).get("mean_ms")
    if mono_ms and args.shape == "productionish":
        base = COMPARISON_CONTEXT["tilelang_stage2_bf1_bb0_bwd_bwd_productionish_ms"]
        wave10 = COMPARISON_CONTEXT["wave10_two_launch_wave7_plus_output_owner_dmimo_v_productionish_ms"]
        timings["wave1_mono_state_lkq_d_chunk_owner_slice"]["ratio_vs_tilelang_bwd_bwd_prod"] = (
            mono_ms / base
        )
        timings["wave10_two_launch_plus_wave1_mono_projection"] = {
            "mean_ms": wave10 + mono_ms,
            "ratio_vs_tilelang_bwd_bwd_prod": (wave10 + mono_ms) / base,
            "margin_ms_vs_tilelang_bwd_bwd_prod": base - (wave10 + mono_ms),
            "note": "Projection only: wave10 covered subset and wave1 state/LKQ/D subset are not fused together yet.",
        }

    return {
        "shape_name": args.shape or "custom",
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "comparison_context": COMPARISON_CONTEXT,
        "cta_model": _cta_model(shape),
        "metadata": metadata,
        "correctness": correctness,
        "timings": timings,
        "read": [
            "Wave 1 adds a true chunk-owner state/LKQ/D CUDA prototype instead of a scalar timestep loop.",
            "The CTA materializes LKQ once, keeps a p-tiled dPsiV intermediate live in shared memory, and feeds DV plus per-chunk DMIMO_V from that same tile.",
            "The LKQ tile is also reused for DSSDA, giving a scalar-family consumer in the same CTA/launch.",
            "DMIMO_V is per-chunk in this wave; the next wave must choose a final reduction owner or fold the reduction into a wider monolithic schedule.",
            "Rotary/trap preprocessing is not in this slice yet; q_flat/k_flat plus bias stand in for preprocessed Q/K.",
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
