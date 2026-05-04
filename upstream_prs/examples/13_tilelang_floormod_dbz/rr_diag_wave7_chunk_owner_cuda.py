"""Wave7 chunk-owner CUDA prototype expansion for Mamba3 bwd_bwd.

Wave6 proved the chunk-warp owner is the right launch shape for the same-time
``R x R`` diagonal DK/DQ/DGAMMA consumers.  This harness adds the next
production consumer visible in the TileLang body: the ``qk_dot`` contribution
to ``dPsiV`` and its direct ``dV`` consumer.  The combined CUDA variant keeps
both paths in one launch and keeps output ownership local to ``(B, H, chunk)``.
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
    stage2_qk_dv_chunk_warp_owner_cuda,
    stage2_qk_dv_chunk_warp_owner_cuda_metadata,
    stage2_qk_dv_chunk_warp_owner_cuda_out,
    stage2_rr_diag_chunk_warp_owner_cuda,
    stage2_rr_diag_chunk_warp_owner_cuda_metadata,
    stage2_rr_diag_chunk_warp_owner_cuda_out,
    stage2_rr_diag_qk_dv_chunk_warp_owner_cuda,
    stage2_rr_diag_qk_dv_chunk_warp_owner_cuda_metadata,
    stage2_rr_diag_qk_dv_chunk_warp_owner_cuda_out,
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


COMPARISON_CONTEXT: dict[str, Any] = {
    "wave6_stage2_bf1_bb0_productionish_bwd_bwd_ms": 3.70674,
    "wave6_stage2_bf1_bb0_productionish_chain_ms": 5.47613,
    "wave6_chunk_warp_owner_diag_productionish_ms": 1.77566,
    "wave6_wave5_timestep_post_diag_productionish_ms": 3.16204,
}


def _max_diff_tensor(ref: torch.Tensor, got: torch.Tensor) -> float:
    return float((ref.float() - got.float()).abs().max().item())


@torch.no_grad()
def qk_dv_torch_reference(inputs: dict[str, torch.Tensor], shape: Shape) -> torch.Tensor:
    """Reference for the qk_dot -> dPsiV -> dV same-time consumer."""

    dout = inputs["dout"].float()
    qk = inputs["qk_dot"].float().view(shape.B, shape.H, shape.S, shape.R, shape.R)
    qk = qk.permute(0, 2, 1, 3, 4).contiguous()
    gamma = (inputs["dt"].float() * torch.sigmoid(inputs["trap"].float())).permute(0, 2, 1).contiguous()
    dv = torch.zeros_like(dout, dtype=torch.float32)

    for r_in in range(shape.R):
        dpsi = torch.zeros_like(dout, dtype=torch.float32)
        for r_out in range(shape.R):
            dphi = dout * inputs["mimo_o"].float()[None, None, :, r_out, :]
            dpsi += dphi * qk[:, :, :, r_out, r_in].unsqueeze(-1)
        dv += dpsi * gamma.unsqueeze(-1) * inputs["mimo_v"].float()[None, None, :, r_in, :]

    return dv.to(inputs["dout"].dtype).contiguous()


def qk_dv_chunk_warp_cuda(inputs: dict[str, torch.Tensor], shape: Shape) -> torch.Tensor:
    return stage2_qk_dv_chunk_warp_owner_cuda(
        dout=inputs["dout"],
        mimo_v=inputs["mimo_v"],
        mimo_o=inputs["mimo_o"],
        qk_dot=inputs["qk_dot"],
        dt=inputs["dt"],
        trap=inputs["trap"],
        chunk_size=shape.chunk,
    )


def combined_chunk_warp_cuda(
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


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
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
        "samples_ms": values,
    }


def _cta_model(shape: Shape) -> dict[str, Any]:
    chunk_ctas = shape.B * shape.H * shape.nchunks
    return {
        "chunk_owner_ctas": chunk_ctas,
        "chunk_owner_ctas_per_sm_at_132_sms": chunk_ctas / 132.0,
        "timesteps_per_cta": shape.chunk,
        "work_per_timestep": {
            "dqk_dot_flops": 2 * shape.R * shape.R * shape.P,
            "diag_dk_dq_consumer_flops": 2 * 2 * shape.R * shape.R * shape.N,
            "qk_dpsi_dv_consumer_flops": 2 * shape.R * shape.R * shape.P + 2 * shape.R * shape.P,
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
        qk_dv = qk_dv_chunk_warp_cuda(inputs, shape)
        combined = combined_chunk_warp_cuda(inputs, shape)
        torch.cuda.synchronize()

        correctness["wave6_diag_vs_wave5_timestep_post_cuda"] = max_diffs(post_ref, diag)
        correctness["wave7_qk_dv_vs_torch_reference"] = {
            "dv_delta": _max_diff_tensor(qk_dv_ref, qk_dv)
        }
        correctness["wave7_combined_diag_vs_wave5_timestep_post_cuda"] = max_diffs(post_ref, combined[:3])
        correctness["wave7_combined_dv_vs_torch_reference"] = {
            "dv_delta": _max_diff_tensor(qk_dv_ref, combined[3])
        }

        metadata["wave6_chunk_warp_owner_diag"] = stage2_rr_diag_chunk_warp_owner_cuda_metadata(inputs["dout"])
        metadata["wave7_qk_dv_chunk_warp_owner"] = stage2_qk_dv_chunk_warp_owner_cuda_metadata(inputs["dout"])
        metadata["wave7_diag_plus_qk_dv_chunk_warp_owner"] = (
            stage2_rr_diag_qk_dv_chunk_warp_owner_cuda_metadata(inputs["dout"])
        )

        diag_dgamma, diag_dk, diag_dq = _empty_outputs(shape, dtype=dtype, device=device)
        qk_dv_out = torch.empty_like(inputs["dout"])
        combined_dgamma, combined_dk, combined_dq = _empty_outputs(shape, dtype=dtype, device=device)
        combined_dv = torch.empty_like(inputs["dout"])

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

        def run_combined() -> None:
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
                dgamma_diag=combined_dgamma,
                dk_delta=combined_dk,
                dq_delta=combined_dq,
                dv_delta=combined_dv,
                chunk_size=shape.chunk,
            )

        timings["wave6_chunk_warp_owner_diag_slice"] = timer(run_diag, warmup=args.warmup, iters=args.iters)
        timings["wave7_chunk_warp_qk_dv_consumer_slice"] = timer(
            run_qk_dv, warmup=args.warmup, iters=args.iters
        )
        timings["wave7_chunk_warp_diag_plus_qk_dv_total_slice"] = timer(
            run_combined, warmup=args.warmup, iters=args.iters
        )
    else:
        qk_dv_ref = qk_dv_torch_reference(inputs, shape)
        correctness["qk_dv_torch_reference_self"] = {"dv_delta": _max_diff_tensor(qk_dv_ref, qk_dv_ref)}
        timings["qk_dv_torch_reference"] = timer(
            lambda: qk_dv_torch_reference(inputs, shape),
            warmup=args.warmup,
            iters=args.iters,
        )

    diag_ms = timings.get("wave6_chunk_warp_owner_diag_slice", {}).get("mean_ms")
    qk_dv_ms = timings.get("wave7_chunk_warp_qk_dv_consumer_slice", {}).get("mean_ms")
    combined_ms = timings.get("wave7_chunk_warp_diag_plus_qk_dv_total_slice", {}).get("mean_ms")
    if diag_ms and qk_dv_ms and combined_ms:
        timings["wave7_chunk_warp_diag_plus_qk_dv_total_slice"]["incremental_ms_vs_wave6_diag"] = (
            combined_ms - diag_ms
        )
        timings["wave7_chunk_warp_diag_plus_qk_dv_total_slice"]["component_sum_ms"] = diag_ms + qk_dv_ms
        timings["wave7_chunk_warp_diag_plus_qk_dv_total_slice"]["component_sum_over_total"] = (
            (diag_ms + qk_dv_ms) / combined_ms
        )
        timings["wave7_chunk_warp_qk_dv_consumer_slice"]["ratio_vs_wave6_diag"] = qk_dv_ms / diag_ms
        if args.shape == "productionish":
            base = COMPARISON_CONTEXT["wave6_stage2_bf1_bb0_productionish_bwd_bwd_ms"]
            timings["wave7_chunk_warp_diag_plus_qk_dv_total_slice"]["ratio_vs_stage2_bf1_bb0_bwd_bwd_prod"] = (
                combined_ms / base
            )

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
            "Wave7 adds the production qk_dot -> dPsiV -> dV same-time consumer to the chunk-warp owner.",
            "The combined variant is one CUDA launch with local output ownership for DGAMMA_DIAG, DK, DQ, and DV.",
            "DMIMO_V is intentionally not included yet because this ownership model would need cross-chunk reduction.",
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
