"""Wave9 state/LKQ/D contribution model for Mamba3 bwd_bwd.

Wave7/8 cover the same-time diagonal/qk path and the qk contribution to
``DMIMO_V``.  This harness isolates the largest remaining ``dPsiV`` producer:

* state: ``K @ dstates.T * exp(dA_cs_rev)``;
* LKQ: ``masked(K @ Q.T) @ dPhiO``;
* direct D: ``D[h] * dPhiO``.

The CUDA kernel owns one ``(B, H, chunk)`` tile, materializes only an in-block
``LKQ`` tile, writes ``DV`` and ``DD`` directly, and can optionally write
``DMIMO_V`` per-chunk partials for a second-pass reduction.  This is not the
final tensor-core kernel; it is a correct ownership/cost probe with no global
``dPsiV`` temporary.
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

from rr_diag_wave6_inlaunch_cuda import (  # noqa: E402
    PRESETS,
    Shape,
    _dtype,
    _shape_from_args,
    _stats,
    _time_cuda,
    _time_wall,
)
from state_lkq_d_cuda_extension import (  # noqa: E402
    state_lkq_d_chunk_owner_cuda_metadata,
    state_lkq_d_dv_dd_chunk_owner_cuda,
    state_lkq_d_dv_dd_chunk_owner_cuda_out,
    state_lkq_d_dv_dd_dmimov_partials_chunk_owner_cuda,
    state_lkq_d_dv_dd_dmimov_partials_chunk_owner_cuda_out,
    state_lkq_d_reduce_dmimov_partials_cuda,
    state_lkq_d_reduce_dmimov_partials_cuda_metadata,
    state_lkq_d_reduce_dmimov_partials_cuda_out,
)


COMPARISON_CONTEXT: dict[str, Any] = {
    "wave8_diag_qk_dv_qk_dmimov_target_ms": 2.45093,
    "wave7_diag_plus_qk_dv_productionish_ms": 1.91459,
    "wave8_qk_dmimov_output_owner_rvec_ms": 0.53634,
    "tilelang_stage2_bf1_bb0_productionish_bwd_bwd_ms": 3.70674,
}


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
    fcs = shape.chunk * shape.R
    return {
        "q": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, fcs, shape.N),
        "k": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, fcs, shape.N),
        "dstates": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, shape.N, shape.P),
        "dphi": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, fcs, shape.P),
        "v": _randn(generator, device, dtype, shape.B, shape.S, shape.H, shape.P),
        "mimo_v": _randn(generator, device, torch.float32, shape.H, shape.R, shape.P),
        "exp_rev": torch.exp(
            _randn(generator, device, torch.float32, shape.B, shape.H, shape.nchunks, shape.chunk, scale=0.01)
        ).contiguous(),
        "segsum": _randn(
            generator, device, torch.float32, shape.B, shape.H, shape.nchunks, shape.chunk, shape.chunk, scale=0.01
        ),
        "D": _randn(generator, device, torch.float32, shape.H),
    }


@torch.no_grad()
def state_lkq_d_torch_reference(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference matching the CUDA probe's isolated state/LKQ/D contribution."""

    dtype = inputs["q"].dtype
    fcs = shape.chunk * shape.R
    total = shape.B * shape.H * shape.nchunks
    q = inputs["q"].float().reshape(total, fcs, shape.N)
    k = inputs["k"].float().reshape(total, fcs, shape.N)
    dstates = inputs["dstates"].float().reshape(total, shape.N, shape.P)
    dphi = inputs["dphi"].float().reshape(total, fcs, shape.P)

    state = torch.bmm(k, dstates)
    exp_rev = inputs["exp_rev"].float().reshape(total, shape.chunk)
    state = state * exp_rev.repeat_interleave(shape.R, dim=-1).unsqueeze(-1)

    lkq = torch.bmm(k, q.transpose(1, 2))
    ci = torch.arange(fcs, device=q.device) // shape.R
    causal = ci[:, None] < ci[None, :]
    seg = inputs["segsum"].float().reshape(total, shape.chunk, shape.chunk)
    seg_weight = seg[
        :,
        ci[None, :].expand(fcs, fcs),
        ci[:, None].expand(fcs, fcs),
    ]
    lkq = torch.where(causal[None, :, :], lkq * torch.exp(seg_weight), torch.zeros_like(lkq))
    lkq_dpsi = torch.bmm(lkq, dphi)

    d_per_total = (
        inputs["D"].float()[None, :, None]
        .expand(shape.B, shape.H, shape.nchunks)
        .reshape(total)[:, None, None]
    )
    dpsi = (state + lkq_dpsi + d_per_total * dphi).to(dtype).float()

    dpsi_bh = dpsi.reshape(shape.B, shape.H, shape.nchunks, shape.chunk, shape.R, shape.P)
    dv = (
        dpsi_bh
        * inputs["mimo_v"].float()[None, :, None, None, :, :]
    ).sum(dim=4)
    dv = dv.permute(0, 2, 3, 1, 4).reshape(shape.B, shape.S, shape.H, shape.P).to(dtype).contiguous()

    v = inputs["v"].float().reshape(shape.B, shape.nchunks, shape.chunk, shape.H, shape.P)
    v = v.permute(0, 3, 1, 2, 4).contiguous()
    dmimo_v = (dpsi_bh * v[:, :, :, :, None, :]).sum(dim=(2, 3))

    dd = inputs["dphi"].float().sum(dim=(2, 3, 4))
    return dv, dd.contiguous(), dmimo_v.contiguous()


def _max_diff(ref: torch.Tensor, got: torch.Tensor) -> float:
    return float((ref.float() - got.float()).abs().max().item())


def _operation_model(shape: Shape) -> dict[str, Any]:
    fcs = shape.chunk * shape.R
    causal_time_pairs = shape.chunk * (shape.chunk - 1) // 2
    causal_entries = causal_time_pairs * shape.R * shape.R
    chunks = shape.B * shape.H * shape.nchunks
    per_chunk = {
        "state_fma": fcs * shape.P * shape.N,
        "lkq_fma": causal_entries * shape.N,
        "lkq_apply_fma": causal_entries * shape.P,
        "D_direct_ops": fcs * shape.P,
        "DV_reduce_ops": fcs * shape.P,
        "DD_reduce_ops": fcs * shape.P,
        "DMIMO_partial_ops_if_piggybacked": fcs * shape.P,
    }
    total = {name: value * chunks for name, value in per_chunk.items()}
    partial_bytes = shape.B * shape.H * shape.nchunks * shape.R * shape.P * 4
    return {
        "chunks": chunks,
        "fused_chunk_size": fcs,
        "causal_lkq_entries_per_chunk": causal_entries,
        "per_chunk": per_chunk,
        "total": total,
        "total_fma_state_lkq_only": total["state_fma"] + total["lkq_fma"] + total["lkq_apply_fma"],
        "dmimov_partial_bytes": partial_bytes,
        "dmimov_partial_mib": partial_bytes / (1024**2),
        "dmimov_partial_extra_global_rw_mib": (partial_bytes * 2 + shape.B * shape.H * shape.R * shape.P * 4)
        / (1024**2),
    }


def _projection(timings: dict[str, Any], args: argparse.Namespace) -> None:
    if args.shape != "productionish":
        return
    base = COMPARISON_CONTEXT["wave8_diag_qk_dv_qk_dmimov_target_ms"]
    tilelang = COMPARISON_CONTEXT["tilelang_stage2_bf1_bb0_productionish_bwd_bwd_ms"]
    for name in (
        "state_lkq_d_dv_dd_chunk_owner",
        "state_lkq_d_dv_dd_dmimov_partials_chunk_owner",
        "state_lkq_d_dmimov_reduce_partials",
        "state_lkq_d_dv_dd_dmimov_two_pass_total",
    ):
        mean_ms = timings.get(name, {}).get("mean_ms")
        if mean_ms is None:
            continue
        timings[name]["projected_total_with_wave8_target_ms"] = base + mean_ms
        timings[name]["projected_total_ratio_vs_tilelang"] = (base + mean_ms) / tilelang


def _time_reference(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if iters <= 0:
        return {"count": 0, "skipped": True}
    timer = _time_cuda if device.type == "cuda" else _time_wall
    return timer(fn, warmup=warmup, iters=iters)


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = _shape_from_args(args)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    inputs = make_prepared_inputs(shape, dtype=dtype, device=device, seed=args.seed)
    timer = _time_cuda if device.type == "cuda" else _time_wall

    correctness: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    ref_dv = ref_dd = ref_dmimo = None
    should_reference = device.type == "cuda" or args.device == "cpu"
    if args.skip_reference:
        should_reference = False
    if should_reference:
        ref_dv, ref_dd, ref_dmimo = state_lkq_d_torch_reference(inputs, shape)
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        dv, dd = state_lkq_d_dv_dd_chunk_owner_cuda(**inputs, chunk_size=shape.chunk)
        dv_p, dd_p, partials = state_lkq_d_dv_dd_dmimov_partials_chunk_owner_cuda(
            **inputs, chunk_size=shape.chunk
        )
        dmimo = state_lkq_d_reduce_dmimov_partials_cuda(partials)
        torch.cuda.synchronize()

        if ref_dv is not None and ref_dd is not None and ref_dmimo is not None:
            correctness["state_lkq_d_dv_dd_vs_torch_reference"] = {
                "dv_delta": _max_diff(ref_dv, dv),
                "dd_delta": _max_diff(ref_dd, dd),
            }
            correctness["state_lkq_d_dv_dd_partials_vs_torch_reference"] = {
                "dv_delta": _max_diff(ref_dv, dv_p),
                "dd_delta": _max_diff(ref_dd, dd_p),
                "dmimo_v_delta": _max_diff(ref_dmimo, dmimo),
            }
            correctness["dv_dd_kernel_vs_partials_kernel"] = {
                "dv_delta": _max_diff(dv, dv_p),
                "dd_delta": _max_diff(dd, dd_p),
            }

        metadata["state_lkq_d_dv_dd_chunk_owner"] = state_lkq_d_chunk_owner_cuda_metadata(
            inputs["q"], with_partials=False
        )
        metadata["state_lkq_d_dv_dd_dmimov_partials_chunk_owner"] = (
            state_lkq_d_chunk_owner_cuda_metadata(inputs["q"], with_partials=True)
        )
        metadata["state_lkq_d_dmimov_reduce_partials"] = state_lkq_d_reduce_dmimov_partials_cuda_metadata()

        dv_out = torch.empty_like(inputs["v"])
        dd_out = torch.empty(shape.B, shape.H, device=device, dtype=torch.float32)
        dv_p_out = torch.empty_like(inputs["v"])
        dd_p_out = torch.empty_like(dd_out)
        partials_out = torch.empty(
            shape.B, shape.H, shape.nchunks, shape.R, shape.P, device=device, dtype=torch.float32
        )
        dmimo_out = torch.empty(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32)

        def run_dv_dd() -> None:
            dd_out.zero_()
            state_lkq_d_dv_dd_chunk_owner_cuda_out(
                **inputs,
                dv=dv_out,
                dd=dd_out,
                chunk_size=shape.chunk,
            )

        def run_dv_dd_partials() -> None:
            dd_p_out.zero_()
            state_lkq_d_dv_dd_dmimov_partials_chunk_owner_cuda_out(
                **inputs,
                dv=dv_p_out,
                dd=dd_p_out,
                dmimo_partials=partials_out,
                chunk_size=shape.chunk,
            )

        def run_reduce() -> None:
            state_lkq_d_reduce_dmimov_partials_cuda_out(partials=partials_out, dmimo_v=dmimo_out)

        def run_two_pass() -> None:
            run_dv_dd_partials()
            run_reduce()

        run_dv_dd_partials()
        torch.cuda.synchronize()

        timings["state_lkq_d_dv_dd_chunk_owner"] = timer(run_dv_dd, warmup=args.warmup, iters=args.iters)
        timings["state_lkq_d_dv_dd_dmimov_partials_chunk_owner"] = timer(
            run_dv_dd_partials, warmup=args.warmup, iters=args.iters
        )
        timings["state_lkq_d_dmimov_reduce_partials"] = timer(run_reduce, warmup=args.warmup, iters=args.iters)
        timings["state_lkq_d_dv_dd_dmimov_two_pass_total"] = timer(
            run_two_pass, warmup=args.warmup, iters=args.iters
        )
    else:
        if ref_dv is not None:
            correctness["state_lkq_d_torch_reference_self"] = {
                "dv_delta": _max_diff(ref_dv, ref_dv),
                "dd_delta": _max_diff(ref_dd, ref_dd),
                "dmimo_v_delta": _max_diff(ref_dmimo, ref_dmimo),
            }

    if not args.skip_reference_timing:
        timings["state_lkq_d_torch_reference_cost_model"] = _time_reference(
            lambda: state_lkq_d_torch_reference(inputs, shape),
            device=device,
            warmup=args.ref_warmup,
            iters=args.ref_iters,
        )

    if (
        "state_lkq_d_dv_dd_chunk_owner" in timings
        and "state_lkq_d_dv_dd_dmimov_partials_chunk_owner" in timings
    ):
        timings["state_lkq_d_dmimov_partial_increment"] = {
            "mean_ms": timings["state_lkq_d_dv_dd_dmimov_partials_chunk_owner"]["mean_ms"]
            - timings["state_lkq_d_dv_dd_chunk_owner"]["mean_ms"],
            "count": 1,
        }

    _projection(timings, args)

    return {
        "shape_name": args.shape or "custom",
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "comparison_context": COMPARISON_CONTEXT,
        "operation_model": _operation_model(shape),
        "metadata": metadata,
        "correctness": correctness,
        "timings": timings,
        "read": [
            "CUDA path computes the state/LKQ/D dPsiV producer and its DV/DD consumers without a global dPsiV temp.",
            "Optional DMIMO_V path writes per-chunk [B,H,nchunks,R,P] partials and uses a final output-owner reduction.",
            "A no-temp output-owner DMIMO_V path would need to recompute LKQ/state per P/R output tile, so it is a poor ownership match for this producer.",
            "The remaining full bwd_bwd work after this slice is DK/DQ state+intra paths, DDA_CS/DDA_CS_REV/DFACTOR/DSSDA/DDA/DANGLES, and final rotary/trap plumbing.",
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
    parser.add_argument("--ref-warmup", type=int, default=1)
    parser.add_argument("--ref-iters", type=int, default=3)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-reference-timing", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
