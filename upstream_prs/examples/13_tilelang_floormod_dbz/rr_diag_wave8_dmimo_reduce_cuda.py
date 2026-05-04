"""Wave8 DMIMO_V cross-chunk reduction alternatives for Mamba3 bwd_bwd.

Wave7 kept one CTA on each ``(B, H, chunk)`` tile and covered the same-time
``qk_dot -> dPsiV -> dV`` path.  The matching ``DMIMO_V`` contribution reduces
over all chunks into ``[B, H, R, P]``.  This harness isolates that reduction
boundary and compares three ownership strategies:

* per-chunk accumulation with global atomics;
* two-pass per-chunk partials plus a final reduction;
* direct output ownership by ``(B, H, R, P-tile)`` with no atomics or partials.
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

from dmimo_reduce_cuda_extension import (  # noqa: E402
    qk_dmimov_atomic_chunk_cuda,
    qk_dmimov_atomic_chunk_cuda_metadata,
    qk_dmimov_atomic_chunk_cuda_out,
    qk_dmimov_output_owner_cuda,
    qk_dmimov_output_owner_cuda_metadata,
    qk_dmimov_output_owner_cuda_out,
    qk_dmimov_output_owner_rvec_cuda,
    qk_dmimov_output_owner_rvec_cuda_metadata,
    qk_dmimov_output_owner_rvec_cuda_out,
    qk_dmimov_partials_chunk_cuda,
    qk_dmimov_partials_chunk_cuda_metadata,
    qk_dmimov_partials_chunk_cuda_out,
    qk_dmimov_reduce_partials_cuda,
    qk_dmimov_reduce_partials_cuda_metadata,
    qk_dmimov_reduce_partials_cuda_out,
)
from rr_diag_cuda_extension import (  # noqa: E402
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
)


COMPARISON_CONTEXT: dict[str, Any] = {
    "wave7_diag_plus_qk_dv_productionish_ms": 1.91459,
    "wave7_qk_dv_only_productionish_ms": 0.35417,
    "wave7_diag_only_refreshed_productionish_ms": 1.76130,
    "tilelang_stage2_bf1_bb0_productionish_bwd_bwd_ms": 3.70674,
    "tilelang_bwd_bwd_approx_ms": 3.7,
}


def _max_diff(ref: torch.Tensor, got: torch.Tensor) -> float:
    return float((ref.float() - got.float()).abs().max().item())


@torch.no_grad()
def qk_dmimov_torch_reference(inputs: dict[str, torch.Tensor], shape: Shape) -> torch.Tensor:
    """Reference for the qk_dot contribution to DMIMO_V.

    This matches the TileLang epilogue relation:
    ``DMIMO_V += dPsiV_combined[b,h,s,r,p] * V[b,s,h,p]``.
    The wave8 slice uses only the same-time qk_dot contribution to dPsiV,
    matching the wave7 qk/dV component.
    """

    dout = inputs["dout"].float().permute(0, 2, 1, 3).contiguous()
    v = inputs["v"].float().permute(0, 2, 1, 3).contiguous()
    gamma = inputs["dt"].float() * torch.sigmoid(inputs["trap"].float())
    qk = inputs["qk_dot"].float().view(shape.B, shape.H, shape.S, shape.R, shape.R)
    base = dout * v * gamma.unsqueeze(-1)
    out = torch.zeros(shape.B, shape.H, shape.R, shape.P, device=dout.device, dtype=torch.float32)
    mimo_o = inputs["mimo_o"].float()

    for r_in in range(shape.R):
        acc = torch.zeros(shape.B, shape.H, shape.P, device=dout.device, dtype=torch.float32)
        for r_out in range(shape.R):
            by_time = base * qk[:, :, :, r_out, r_in].unsqueeze(-1)
            acc += by_time.sum(dim=2) * mimo_o[None, :, r_out, :]
        out[:, :, r_in, :] = acc
    return out.contiguous()


def _memory_model(shape: Shape) -> dict[str, Any]:
    output_bytes = shape.B * shape.H * shape.R * shape.P * 4
    partial_bytes = shape.B * shape.H * shape.nchunks * shape.R * shape.P * 4
    atomic_adds = shape.B * shape.H * shape.nchunks * shape.R * shape.P
    timestep_contributions = shape.B * shape.H * shape.S * shape.R * shape.P
    return {
        "dmimo_v_output_bytes": output_bytes,
        "dmimo_v_output_mib": output_bytes / (1024**2),
        "atomic_chunk_extra_temp_bytes": 0,
        "atomic_chunk_global_atomic_adds": atomic_adds,
        "two_pass_partial_bytes": partial_bytes,
        "two_pass_partial_mib": partial_bytes / (1024**2),
        "two_pass_extra_global_rw_bytes": partial_bytes * 2 + output_bytes,
        "two_pass_extra_global_rw_mib": (partial_bytes * 2 + output_bytes) / (1024**2),
        "output_owner_extra_temp_bytes": 0,
        "raw_timestep_dmimov_contributions": timestep_contributions,
    }


def _cta_model(shape: Shape) -> dict[str, Any]:
    ptiles = math.ceil(shape.P / 32)
    atomic_ctas = shape.B * shape.H * shape.nchunks * shape.R
    reduce_ctas = shape.B * shape.H * shape.R
    output_owner_ctas = shape.B * shape.H * shape.R * ptiles
    output_owner_rvec_ctas = shape.B * shape.H * ptiles
    return {
        "atomic_or_partial_writer_ctas": atomic_ctas,
        "partial_reducer_ctas": reduce_ctas,
        "output_owner_ctas": output_owner_ctas,
        "output_owner_rvec_ctas": output_owner_rvec_ctas,
        "output_owner_p_tile": 32,
        "ctas_per_sm_at_132_sms": {
            "atomic_or_partial_writer": atomic_ctas / 132.0,
            "partial_reducer": reduce_ctas / 132.0,
            "output_owner": output_owner_ctas / 132.0,
            "output_owner_rvec": output_owner_rvec_ctas / 132.0,
        },
        "work_per_output_element": {
            "timesteps": shape.S,
            "r_out_terms_per_timestep": shape.R,
        },
        "work_per_chunk_partial_element": {
            "timesteps": shape.chunk,
            "r_out_terms_per_timestep": shape.R,
        },
    }


def _project_totals(timings: dict[str, Any], args: argparse.Namespace) -> None:
    measured_wave7 = timings.get("wave7_chunk_warp_diag_plus_qk_dv_total_slice", {}).get("mean_ms")
    canonical_wave7 = COMPARISON_CONTEXT["wave7_diag_plus_qk_dv_productionish_ms"]
    tilelang = COMPARISON_CONTEXT["tilelang_stage2_bf1_bb0_productionish_bwd_bwd_ms"]
    for name in (
        "wave8_dmimov_atomic_chunk_zero_plus_kernel",
        "wave8_dmimov_two_pass_total",
        "wave8_dmimov_output_owner",
        "wave8_dmimov_output_owner_rvec",
    ):
        mean_ms = timings.get(name, {}).get("mean_ms")
        if not mean_ms:
            continue
        timings[name]["ratio_vs_wave7_canonical_prod"] = (
            mean_ms / canonical_wave7 if args.shape == "productionish" else None
        )
        timings[name]["projected_total_with_wave7_measured_ms"] = (
            measured_wave7 + mean_ms if measured_wave7 else None
        )
        timings[name]["projected_total_with_wave7_canonical_prod_ms"] = (
            canonical_wave7 + mean_ms if args.shape == "productionish" else None
        )
        timings[name]["projected_total_ratio_vs_tilelang_prod"] = (
            (canonical_wave7 + mean_ms) / tilelang if args.shape == "productionish" else None
        )


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
        ref = qk_dmimov_torch_reference(inputs, shape)
        atomic = qk_dmimov_atomic_chunk_cuda(
            dout=inputs["dout"],
            v=inputs["v"],
            mimo_o=inputs["mimo_o"],
            qk_dot=inputs["qk_dot"],
            dt=inputs["dt"],
            trap=inputs["trap"],
            chunk_size=shape.chunk,
        )
        partials = qk_dmimov_partials_chunk_cuda(
            dout=inputs["dout"],
            v=inputs["v"],
            mimo_o=inputs["mimo_o"],
            qk_dot=inputs["qk_dot"],
            dt=inputs["dt"],
            trap=inputs["trap"],
            chunk_size=shape.chunk,
        )
        two_pass = qk_dmimov_reduce_partials_cuda(partials)
        owner = qk_dmimov_output_owner_cuda(
            dout=inputs["dout"],
            v=inputs["v"],
            mimo_o=inputs["mimo_o"],
            qk_dot=inputs["qk_dot"],
            dt=inputs["dt"],
            trap=inputs["trap"],
            chunk_size=shape.chunk,
        )
        owner_rvec = qk_dmimov_output_owner_rvec_cuda(
            dout=inputs["dout"],
            v=inputs["v"],
            mimo_o=inputs["mimo_o"],
            qk_dot=inputs["qk_dot"],
            dt=inputs["dt"],
            trap=inputs["trap"],
            chunk_size=shape.chunk,
        )
        torch.cuda.synchronize()

        correctness["qk_dmimov_atomic_chunk_vs_torch_reference"] = {"dmimo_v_delta": _max_diff(ref, atomic)}
        correctness["qk_dmimov_two_pass_vs_torch_reference"] = {"dmimo_v_delta": _max_diff(ref, two_pass)}
        correctness["qk_dmimov_output_owner_vs_torch_reference"] = {"dmimo_v_delta": _max_diff(ref, owner)}
        correctness["qk_dmimov_output_owner_rvec_vs_torch_reference"] = {
            "dmimo_v_delta": _max_diff(ref, owner_rvec)
        }
        correctness["two_pass_vs_output_owner"] = {"dmimo_v_delta": _max_diff(two_pass, owner)}
        correctness["output_owner_vs_output_owner_rvec"] = {"dmimo_v_delta": _max_diff(owner, owner_rvec)}

        metadata["qk_dmimov_atomic_chunk"] = qk_dmimov_atomic_chunk_cuda_metadata(inputs["dout"])
        metadata["qk_dmimov_partials_chunk"] = qk_dmimov_partials_chunk_cuda_metadata(inputs["dout"])
        metadata["qk_dmimov_reduce_partials"] = qk_dmimov_reduce_partials_cuda_metadata()
        metadata["qk_dmimov_output_owner"] = qk_dmimov_output_owner_cuda_metadata(inputs["dout"])
        metadata["qk_dmimov_output_owner_rvec"] = qk_dmimov_output_owner_rvec_cuda_metadata(inputs["dout"])
        metadata["wave7_diag_plus_qk_dv_chunk_warp_owner"] = (
            stage2_rr_diag_qk_dv_chunk_warp_owner_cuda_metadata(inputs["dout"])
        )

        combined_dgamma, combined_dk, combined_dq = _empty_outputs(shape, dtype=dtype, device=device)
        combined_dv = torch.empty_like(inputs["dout"])
        atomic_out = torch.empty(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32)
        partials_out = torch.empty(
            shape.B,
            shape.H,
            shape.nchunks,
            shape.R,
            shape.P,
            device=device,
            dtype=torch.float32,
        )
        reduced_out = torch.empty(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32)
        owner_out = torch.empty_like(reduced_out)
        owner_rvec_out = torch.empty_like(reduced_out)

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
                dgamma_diag=combined_dgamma,
                dk_delta=combined_dk,
                dq_delta=combined_dq,
                dv_delta=combined_dv,
                chunk_size=shape.chunk,
            )

        def run_atomic() -> None:
            atomic_out.zero_()
            qk_dmimov_atomic_chunk_cuda_out(
                dout=inputs["dout"],
                v=inputs["v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                dmimo_v=atomic_out,
                chunk_size=shape.chunk,
            )

        def run_partials() -> None:
            qk_dmimov_partials_chunk_cuda_out(
                dout=inputs["dout"],
                v=inputs["v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                partials=partials_out,
                chunk_size=shape.chunk,
            )

        def run_reduce() -> None:
            qk_dmimov_reduce_partials_cuda_out(partials=partials_out, dmimo_v=reduced_out)

        def run_two_pass() -> None:
            run_partials()
            run_reduce()

        def run_owner() -> None:
            qk_dmimov_output_owner_cuda_out(
                dout=inputs["dout"],
                v=inputs["v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                dmimo_v=owner_out,
                chunk_size=shape.chunk,
            )

        def run_owner_rvec() -> None:
            qk_dmimov_output_owner_rvec_cuda_out(
                dout=inputs["dout"],
                v=inputs["v"],
                mimo_o=inputs["mimo_o"],
                qk_dot=inputs["qk_dot"],
                dt=inputs["dt"],
                trap=inputs["trap"],
                dmimo_v=owner_rvec_out,
                chunk_size=shape.chunk,
            )

        run_partials()
        torch.cuda.synchronize()

        timings["wave7_chunk_warp_diag_plus_qk_dv_total_slice"] = timer(
            run_wave7_combined, warmup=args.warmup, iters=args.iters
        )
        timings["wave8_dmimov_atomic_chunk_zero_plus_kernel"] = timer(
            run_atomic, warmup=args.warmup, iters=args.iters
        )
        timings["wave8_dmimov_two_pass_partial_writer"] = timer(
            run_partials, warmup=args.warmup, iters=args.iters
        )
        timings["wave8_dmimov_two_pass_final_reduce"] = timer(
            run_reduce, warmup=args.warmup, iters=args.iters
        )
        timings["wave8_dmimov_two_pass_total"] = timer(
            run_two_pass, warmup=args.warmup, iters=args.iters
        )
        timings["wave8_dmimov_output_owner"] = timer(run_owner, warmup=args.warmup, iters=args.iters)
        timings["wave8_dmimov_output_owner_rvec"] = timer(
            run_owner_rvec, warmup=args.warmup, iters=args.iters
        )
    else:
        ref = qk_dmimov_torch_reference(inputs, shape)
        correctness["qk_dmimov_torch_reference_self"] = {"dmimo_v_delta": _max_diff(ref, ref)}
        timings["qk_dmimov_torch_reference"] = timer(
            lambda: qk_dmimov_torch_reference(inputs, shape),
            warmup=args.warmup,
            iters=args.iters,
        )

    _project_totals(timings, args)

    return {
        "shape_name": args.shape or "custom",
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "comparison_context": COMPARISON_CONTEXT,
        "cta_model": _cta_model(shape),
        "memory_model": _memory_model(shape),
        "metadata": metadata,
        "correctness": correctness,
        "timings": timings,
        "read": [
            "This isolates the qk_dot DMIMO_V reduction hazard from the larger bwd_bwd rewrite.",
            "Atomic and two-pass paths preserve chunk ownership; output-owner remaps work to the final [B,H,R,P] tile.",
            "The all-R output owner reuses each loaded timestep value across the four R outputs for a P tile.",
            "Projected totals add the measured DMIMO_V slice to the wave7 diag+qk/dV chunk-warp owner baseline.",
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
