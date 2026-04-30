"""Wave8 chunk-owner CUDA prototype expansion for Mamba3 bwd_bwd.

Wave7 added the same-time ``qk_dot -> dPsiV -> DV`` consumer to the
chunk-warp owner.  Wave8 adds the matching ``qk_dot -> dPsiV -> DMIMO_V``
consumer without per-chunk global partials: the one-launch combined kernel
keeps the existing ``(B, H, chunk)`` CTAs for diagonal/DV outputs and appends
``(B, H, R)`` sequence-owner CTAs that reduce all timesteps for each
``DMIMO_V[b, h, r, :]`` row.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rr_diag_cuda_extension import (  # noqa: E402
    stage2_qk_dmimo_v_sequence_owner_cuda,
    stage2_qk_dmimo_v_sequence_owner_cuda_metadata,
    stage2_qk_dmimo_v_sequence_owner_cuda_out,
    stage2_qk_dv_chunk_warp_owner_cuda,
    stage2_qk_dv_chunk_warp_owner_cuda_metadata,
    stage2_qk_dv_chunk_warp_owner_cuda_out,
    stage2_rr_diag_chunk_warp_owner_cuda,
    stage2_rr_diag_chunk_warp_owner_cuda_metadata,
    stage2_rr_diag_chunk_warp_owner_cuda_out,
    stage2_rr_diag_qk_dv_chunk_warp_owner_cuda,
    stage2_rr_diag_qk_dv_chunk_warp_owner_cuda_metadata,
    stage2_rr_diag_qk_dv_chunk_warp_owner_cuda_out,
    stage2_rr_diag_qk_dv_dmimo_v_owner_cuda,
    stage2_rr_diag_qk_dv_dmimo_v_owner_cuda_metadata,
    stage2_rr_diag_qk_dv_dmimo_v_owner_cuda_out,
)
from rr_diag_wave6_inlaunch_cuda import (  # noqa: E402
    PRESETS,
    Shape,
    _dtype,
    _empty_outputs,
    _shape_from_args,
    _time_cuda,
    _time_wall,
    make_inputs,
    max_diffs,
    stage2_post_cuda_reference,
)
from rr_diag_wave7_chunk_owner_cuda import (  # noqa: E402
    COMPARISON_CONTEXT as WAVE7_COMPARISON_CONTEXT,
    qk_dv_torch_reference,
)


COMPARISON_CONTEXT: dict[str, Any] = {
    **WAVE7_COMPARISON_CONTEXT,
    "wave7_chunk_warp_diag_plus_qk_dv_productionish_ms": 1.91459,
    "wave7_chunk_warp_qk_dv_productionish_ms": 0.35417,
}


def _max_diff_tensor(ref: torch.Tensor, got: torch.Tensor) -> float:
    return float((ref.float() - got.float()).abs().max().item())


@torch.no_grad()
def qk_dmimo_v_torch_reference(inputs: dict[str, torch.Tensor], shape: Shape) -> torch.Tensor:
    """Reference for the qk_dot -> dPsiV -> DMIMO_V reduced consumer."""

    dout = inputs["dout"].float()
    v = inputs["v"].float()
    qk = inputs["qk_dot"].float().view(shape.B, shape.H, shape.S, shape.R, shape.R)
    qk = qk.permute(0, 2, 1, 3, 4).contiguous()
    gamma = (inputs["dt"].float() * torch.sigmoid(inputs["trap"].float())).permute(0, 2, 1).contiguous()
    dmimo_v = torch.zeros(
        shape.B,
        shape.H,
        shape.R,
        shape.P,
        device=dout.device,
        dtype=torch.float32,
    )

    for r_in in range(shape.R):
        dpsi = torch.zeros_like(dout, dtype=torch.float32)
        for r_out in range(shape.R):
            dphi = dout * inputs["mimo_o"].float()[None, None, :, r_out, :]
            dpsi += dphi * qk[:, :, :, r_out, r_in].unsqueeze(-1)
        dmimo_v[:, :, r_in, :] = (dpsi * gamma.unsqueeze(-1) * v).sum(dim=1)

    return dmimo_v.contiguous()


def qk_dmimo_v_sequence_cuda(inputs: dict[str, torch.Tensor], shape: Shape) -> torch.Tensor:
    return stage2_qk_dmimo_v_sequence_owner_cuda(
        dout=inputs["dout"],
        v=inputs["v"],
        mimo_o=inputs["mimo_o"],
        qk_dot=inputs["qk_dot"],
        dt=inputs["dt"],
        trap=inputs["trap"],
    )


def combined_wave7_cuda(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return stage2_rr_diag_qk_dv_chunk_warp_owner_cuda(
        dout=inputs["dout"],
        q_flat=inputs["q_flat"],
        k_flat=inputs["k_flat"],
        v=inputs["v"],
        q_bias=inputs["q_bias"],
        k_bias=inputs["k_bias"],
        mimo_v=inputs["mimo_v"],
        mimo_o=inputs["mimo_o"],
        qk_dot=inputs["qk_dot"],
        dt=inputs["dt"],
        trap=inputs["trap"],
        chunk_size=shape.chunk,
    )


def combined_wave8_cuda(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return stage2_rr_diag_qk_dv_dmimo_v_owner_cuda(
        dout=inputs["dout"],
        q_flat=inputs["q_flat"],
        k_flat=inputs["k_flat"],
        v=inputs["v"],
        q_bias=inputs["q_bias"],
        k_bias=inputs["k_bias"],
        mimo_v=inputs["mimo_v"],
        mimo_o=inputs["mimo_o"],
        qk_dot=inputs["qk_dot"],
        dt=inputs["dt"],
        trap=inputs["trap"],
        chunk_size=shape.chunk,
    )


def _cta_model(shape: Shape) -> dict[str, Any]:
    chunk_ctas = shape.B * shape.H * shape.nchunks
    dmimo_ctas = shape.B * shape.H * shape.R
    total_ctas = chunk_ctas + dmimo_ctas
    return {
        "chunk_owner_ctas": chunk_ctas,
        "dmimo_v_sequence_owner_ctas": dmimo_ctas,
        "wave8_total_ctas": total_ctas,
        "wave8_total_ctas_per_sm_at_132_sms": total_ctas / 132.0,
        "timesteps_per_chunk_cta": shape.chunk,
        "dmimo_v_timesteps_per_sequence_cta": shape.S,
        "work_per_timestep": {
            "dqk_dot_flops": 2 * shape.R * shape.R * shape.P,
            "diag_dk_dq_consumer_flops": 2 * 2 * shape.R * shape.R * shape.N,
            "qk_dpsi_dv_consumer_flops": 2 * shape.R * shape.R * shape.P + 2 * shape.R * shape.P,
            "qk_dpsi_dmimo_v_consumer_flops": 2 * shape.R * shape.R * shape.P,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = _shape_from_args(args)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    inputs = make_inputs(shape, dtype=dtype, device=device, seed=args.seed)

    correctness: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    timer: Callable[..., dict[str, Any]] = _time_cuda if device.type == "cuda" else _time_wall

    if device.type == "cuda":
        post_ref = stage2_post_cuda_reference(inputs, shape)
        diag = stage2_rr_diag_chunk_warp_owner_cuda(
            dout=inputs["dout"],
            q_flat=inputs["q_flat"],
            k_flat=inputs["k_flat"],
            v=inputs["v"],
            q_bias=inputs["q_bias"],
            k_bias=inputs["k_bias"],
            mimo_v=inputs["mimo_v"],
            mimo_o=inputs["mimo_o"],
            qk_dot=inputs["qk_dot"],
            dt=inputs["dt"],
            trap=inputs["trap"],
            chunk_size=shape.chunk,
        )
        qk_dv_ref = qk_dv_torch_reference(inputs, shape)
        qk_dv = stage2_qk_dv_chunk_warp_owner_cuda(
            dout=inputs["dout"],
            mimo_v=inputs["mimo_v"],
            mimo_o=inputs["mimo_o"],
            qk_dot=inputs["qk_dot"],
            dt=inputs["dt"],
            trap=inputs["trap"],
            chunk_size=shape.chunk,
        )
        qk_dmimo_v_ref = qk_dmimo_v_torch_reference(inputs, shape)
        qk_dmimo_v = qk_dmimo_v_sequence_cuda(inputs, shape)
        wave7_combined = combined_wave7_cuda(inputs, shape)
        wave8_combined = combined_wave8_cuda(inputs, shape)
        torch.cuda.synchronize()

        correctness["wave6_diag_vs_wave5_timestep_post_cuda"] = max_diffs(post_ref, diag)
        correctness["wave7_qk_dv_vs_torch_reference"] = {
            "dv_delta": _max_diff_tensor(qk_dv_ref, qk_dv)
        }
        correctness["wave8_qk_dmimo_v_vs_torch_reference"] = {
            "dmimo_v_delta": _max_diff_tensor(qk_dmimo_v_ref, qk_dmimo_v)
        }
        correctness["wave7_combined_diag_vs_wave5_timestep_post_cuda"] = max_diffs(
            post_ref, wave7_combined[:3]
        )
        correctness["wave7_combined_dv_vs_torch_reference"] = {
            "dv_delta": _max_diff_tensor(qk_dv_ref, wave7_combined[3])
        }
        correctness["wave8_combined_diag_vs_wave5_timestep_post_cuda"] = max_diffs(
            post_ref, wave8_combined[:3]
        )
        correctness["wave8_combined_dv_vs_torch_reference"] = {
            "dv_delta": _max_diff_tensor(qk_dv_ref, wave8_combined[3])
        }
        correctness["wave8_combined_dmimo_v_vs_torch_reference"] = {
            "dmimo_v_delta": _max_diff_tensor(qk_dmimo_v_ref, wave8_combined[4])
        }

        metadata["wave6_chunk_warp_owner_diag"] = stage2_rr_diag_chunk_warp_owner_cuda_metadata(inputs["dout"])
        metadata["wave7_qk_dv_chunk_warp_owner"] = stage2_qk_dv_chunk_warp_owner_cuda_metadata(inputs["dout"])
        metadata["wave7_diag_plus_qk_dv_chunk_warp_owner"] = (
            stage2_rr_diag_qk_dv_chunk_warp_owner_cuda_metadata(inputs["dout"])
        )
        metadata["wave8_qk_dmimo_v_sequence_owner"] = (
            stage2_qk_dmimo_v_sequence_owner_cuda_metadata(inputs["dout"])
        )
        metadata["wave8_diag_plus_qk_dv_plus_dmimo_v_owner"] = (
            stage2_rr_diag_qk_dv_dmimo_v_owner_cuda_metadata(inputs["dout"])
        )

        diag_dgamma, diag_dk, diag_dq = _empty_outputs(shape, dtype=dtype, device=device)
        qk_dv_out = torch.empty_like(inputs["dout"])
        qk_dmimo_v_out = torch.empty(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32)
        wave7_dgamma, wave7_dk, wave7_dq = _empty_outputs(shape, dtype=dtype, device=device)
        wave7_dv = torch.empty_like(inputs["dout"])
        wave8_dgamma, wave8_dk, wave8_dq = _empty_outputs(shape, dtype=dtype, device=device)
        wave8_dv = torch.empty_like(inputs["dout"])
        wave8_dmimo_v = torch.empty_like(qk_dmimo_v_out)

        def run_diag() -> None:
            stage2_rr_diag_chunk_warp_owner_cuda_out(
                dout=inputs["dout"],
                q_flat=inputs["q_flat"],
                k_flat=inputs["k_flat"],
                v=inputs["v"],
                q_bias=inputs["q_bias"],
                k_bias=inputs["k_bias"],
                mimo_v=inputs["mimo_v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                dgamma_diag=diag_dgamma,
                dk_delta=diag_dk,
                dq_delta=diag_dq,
                chunk_size=shape.chunk,
            )

        def run_qk_dv() -> None:
            stage2_qk_dv_chunk_warp_owner_cuda_out(
                dout=inputs["dout"],
                mimo_v=inputs["mimo_v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                dv_delta=qk_dv_out,
                chunk_size=shape.chunk,
            )

        def run_qk_dmimo_v() -> None:
            stage2_qk_dmimo_v_sequence_owner_cuda_out(
                dout=inputs["dout"],
                v=inputs["v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                dmimo_v_delta=qk_dmimo_v_out,
            )

        def run_wave7_combined() -> None:
            stage2_rr_diag_qk_dv_chunk_warp_owner_cuda_out(
                dout=inputs["dout"],
                q_flat=inputs["q_flat"],
                k_flat=inputs["k_flat"],
                v=inputs["v"],
                q_bias=inputs["q_bias"],
                k_bias=inputs["k_bias"],
                mimo_v=inputs["mimo_v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                dgamma_diag=wave7_dgamma,
                dk_delta=wave7_dk,
                dq_delta=wave7_dq,
                dv_delta=wave7_dv,
                chunk_size=shape.chunk,
            )

        def run_wave8_combined() -> None:
            stage2_rr_diag_qk_dv_dmimo_v_owner_cuda_out(
                dout=inputs["dout"],
                q_flat=inputs["q_flat"],
                k_flat=inputs["k_flat"],
                v=inputs["v"],
                q_bias=inputs["q_bias"],
                k_bias=inputs["k_bias"],
                mimo_v=inputs["mimo_v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                dgamma_diag=wave8_dgamma,
                dk_delta=wave8_dk,
                dq_delta=wave8_dq,
                dv_delta=wave8_dv,
                dmimo_v_delta=wave8_dmimo_v,
                chunk_size=shape.chunk,
            )

        timings["wave6_chunk_warp_owner_diag_slice"] = timer(run_diag, warmup=args.warmup, iters=args.iters)
        timings["wave7_chunk_warp_qk_dv_consumer_slice"] = timer(
            run_qk_dv, warmup=args.warmup, iters=args.iters
        )
        timings["wave7_chunk_warp_diag_plus_qk_dv_total_slice"] = timer(
            run_wave7_combined, warmup=args.warmup, iters=args.iters
        )
        timings["wave8_sequence_qk_dmimo_v_consumer_slice"] = timer(
            run_qk_dmimo_v, warmup=args.warmup, iters=args.iters
        )
        timings["wave8_diag_plus_qk_dv_plus_dmimo_v_total_slice"] = timer(
            run_wave8_combined, warmup=args.warmup, iters=args.iters
        )
    else:
        qk_dv_ref = qk_dv_torch_reference(inputs, shape)
        qk_dmimo_v_ref = qk_dmimo_v_torch_reference(inputs, shape)
        correctness["qk_dv_torch_reference_self"] = {"dv_delta": _max_diff_tensor(qk_dv_ref, qk_dv_ref)}
        correctness["qk_dmimo_v_torch_reference_self"] = {
            "dmimo_v_delta": _max_diff_tensor(qk_dmimo_v_ref, qk_dmimo_v_ref)
        }
        timings["qk_dmimo_v_torch_reference"] = timer(
            lambda: qk_dmimo_v_torch_reference(inputs, shape),
            warmup=args.warmup,
            iters=args.iters,
        )

    diag_ms = timings.get("wave6_chunk_warp_owner_diag_slice", {}).get("mean_ms")
    qk_dv_ms = timings.get("wave7_chunk_warp_qk_dv_consumer_slice", {}).get("mean_ms")
    wave7_ms = timings.get("wave7_chunk_warp_diag_plus_qk_dv_total_slice", {}).get("mean_ms")
    dmimo_ms = timings.get("wave8_sequence_qk_dmimo_v_consumer_slice", {}).get("mean_ms")
    wave8_ms = timings.get("wave8_diag_plus_qk_dv_plus_dmimo_v_total_slice", {}).get("mean_ms")
    if diag_ms and qk_dv_ms and wave7_ms and dmimo_ms and wave8_ms:
        timings["wave8_diag_plus_qk_dv_plus_dmimo_v_total_slice"]["incremental_ms_vs_wave7_combined"] = (
            wave8_ms - wave7_ms
        )
        timings["wave8_diag_plus_qk_dv_plus_dmimo_v_total_slice"]["incremental_ms_vs_wave6_diag"] = (
            wave8_ms - diag_ms
        )
        timings["wave8_diag_plus_qk_dv_plus_dmimo_v_total_slice"]["component_sum_ms"] = (
            diag_ms + qk_dv_ms + dmimo_ms
        )
        timings["wave8_diag_plus_qk_dv_plus_dmimo_v_total_slice"]["component_sum_over_total"] = (
            (diag_ms + qk_dv_ms + dmimo_ms) / wave8_ms
        )
        timings["wave8_sequence_qk_dmimo_v_consumer_slice"]["ratio_vs_wave7_qk_dv"] = dmimo_ms / qk_dv_ms
        if args.shape == "productionish":
            base = COMPARISON_CONTEXT["wave6_stage2_bf1_bb0_productionish_bwd_bwd_ms"]
            timings["wave8_diag_plus_qk_dv_plus_dmimo_v_total_slice"][
                "ratio_vs_stage2_bf1_bb0_bwd_bwd_prod"
            ] = wave8_ms / base

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
            "Wave8 adds qk_dot -> dPsiV -> DMIMO_V with sequence-owner CTAs, not per-chunk atomics.",
            "The combined variant is still one CUDA launch: chunk-warp CTAs for wave7 outputs plus B,H,R CTAs for DMIMO_V rows.",
            "The DMIMO_V slice covers the qk_dot same-time contribution; state/LKQ/D contributions are still outside this prototype.",
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
