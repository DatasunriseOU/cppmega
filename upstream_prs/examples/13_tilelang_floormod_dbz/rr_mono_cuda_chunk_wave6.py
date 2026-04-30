"""Wave 6 chunk-group split-owner CUDA prototype for Mamba3 bwd_bwd.

Wave 6 splits the Wave 5 scan owner across chunk groups.  One CTA owns
``(B, H, chunk_group)``, walks only that group's chunks in reverse order, reuses
each chunk's LKQ across P64 panels, writes a per-group ``DMIMO_V`` partial, and
then a small reduction kernel produces final ``DMIMO_V``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    _max_diff,
    make_mono_inputs,
    mono_state_lkq_d_torch_reference,
)
from rr_mono_cuda_chunk_wave2 import (  # noqa: E402
    WAVE1_SCALAR_PRODUCTIONISH_MS,
    mono_wmma_lkq_dphi_torch_reference,
)
from rr_mono_cuda_chunk_wave6_extension import (  # noqa: E402
    stage2_mono_chunk_group_owner_cuda,
    stage2_mono_chunk_group_owner_cuda_metadata,
    stage2_mono_chunk_group_owner_cuda_out,
)


WAVE2_WMMA_PRODUCTIONISH_MS = 8.919168281555176
WAVE3_WMMA_TRIANGULAR_PRODUCTIONISH_MS = 8.467136001586914
WAVE4_WMMA_P64_PRODUCTIONISH_MS = 8.784351921081543
WAVE5_SCAN_OWNER_PRODUCTIONISH_MS = 14.08131217956543
P_PANEL = 64
DEFAULT_CHUNK_GROUP_SIZE = 8


def _chunk_groups(shape: Shape, chunk_group_size: int) -> int:
    return (shape.nchunks + chunk_group_size - 1) // chunk_group_size


def _empty_chunk_group_outputs(
    shape: Shape,
    *,
    dtype: torch.dtype,
    device: torch.device,
    chunk_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = _chunk_groups(shape, chunk_group_size)
    dv_delta = torch.empty(shape.B, shape.S, shape.H, shape.P, dtype=dtype, device=device)
    dmimo_v_delta = torch.empty(shape.B, shape.H, shape.R, shape.P, dtype=torch.float32, device=device)
    dmimo_group_delta = torch.empty(
        shape.B,
        shape.H,
        groups,
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
    return dv_delta, dmimo_v_delta, dmimo_group_delta, dssda_delta


def _collapse_dmimo_chunk(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return outputs[0], outputs[1].sum(dim=2).contiguous(), outputs[2]


def mono_chunk_group_owner_cuda(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
    *,
    chunk_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return stage2_mono_chunk_group_owner_cuda(
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
        chunk_group_size=chunk_group_size,
    )


def _diffs(
    ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    got: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    return {
        "dv_delta": _max_diff(ref[0], got[0]),
        "dmimo_v_delta": _max_diff(ref[1], got[1]),
        "dssda_delta": _max_diff(ref[2], got[2]),
    }


def _cta_model(shape: Shape, chunk_group_size: int) -> dict[str, Any]:
    groups = _chunk_groups(shape, chunk_group_size)
    chunk_group_ctas = shape.B * shape.H * groups
    chunk_visits = shape.B * shape.H * shape.nchunks
    p_panels = shape.P // P_PANEL
    fused_chunk = shape.chunk * shape.R
    return {
        "owner": "B,H,chunk-group scan",
        "chunk_group_size": chunk_group_size,
        "chunk_groups": groups,
        "chunk_group_owner_ctas": chunk_group_ctas,
        "chunk_group_owner_ctas_per_sm_at_132_sms": chunk_group_ctas / 132.0,
        "wave5_scan_owner_ctas": shape.B * shape.H,
        "cta_multiplier_vs_wave5": groups,
        "chunks_per_cta": chunk_group_size,
        "last_group_chunks": shape.nchunks - (groups - 1) * chunk_group_size,
        "logical_chunk_visits": chunk_visits,
        "reduction_ctas": shape.B * shape.H * shape.R,
        "dmimo_group_scratch_elements": shape.B * shape.H * groups * shape.R * shape.P,
        "reverse_chunk_loop": True,
        "p_panel": P_PANEL,
        "p_panels_per_chunk": p_panels,
        "local_accumulators": {
            "dmimo_v_elements_per_cta": shape.R * shape.P,
            "dssda_elements_per_chunk": shape.chunk * shape.chunk,
            "lkq_elements_per_chunk": fused_chunk * fused_chunk,
        },
        "wmma_tiles_per_logical_chunk": {
            "lkq_k_qt": (fused_chunk // 16) * (fused_chunk // 16) * (shape.N // 16),
            "dki_psiv_dphi_t": p_panels * (fused_chunk // 16) * (fused_chunk // 16) * (P_PANEL // 16),
            "state_k_dstates": p_panels * (fused_chunk // 16) * (P_PANEL // 16) * (shape.N // 16),
            "lkq_dphi_apply_full_mask": p_panels * (fused_chunk // 16) * (P_PANEL // 16) * (fused_chunk // 16),
        },
        "outputs": ["DV", "DMIMO_V_final", "DSSDA"],
        "comparison_ms": {
            "wave1_scalar_mono_state_lkq_d": WAVE1_SCALAR_PRODUCTIONISH_MS,
            "wave2_wmma_chunk_owner": WAVE2_WMMA_PRODUCTIONISH_MS,
            "wave3_wmma_triangular_chunk_owner": WAVE3_WMMA_TRIANGULAR_PRODUCTIONISH_MS,
            "wave4_wmma_p64_panel_chunk_owner": WAVE4_WMMA_P64_PRODUCTIONISH_MS,
            "wave5_scan_owner": WAVE5_SCAN_OWNER_PRODUCTIONISH_MS,
            **COMPARISON_CONTEXT,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = _shape_from_args(args)
    if shape.S % shape.chunk != 0:
        raise RuntimeError("Wave 6 harness currently expects S divisible by chunk")
    if shape.P % P_PANEL != 0:
        raise RuntimeError("Wave 6 chunk-group owner prototype requires P to be a multiple of 64")
    if args.chunk_group_size <= 0:
        raise RuntimeError("--chunk-group-size must be positive")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    inputs = make_mono_inputs(shape, dtype=dtype, device=device, seed=args.seed)
    chunk_group_size = args.chunk_group_size

    correctness: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    timer: Callable[..., dict[str, Any]] = _time_cuda if device.type == "cuda" else _time_wall

    if device.type == "cuda":
        if dtype != torch.bfloat16:
            raise RuntimeError("Wave 6 chunk-group owner CUDA path currently requires --dtype bf16")

        ref = _collapse_dmimo_chunk(mono_wmma_lkq_dphi_torch_reference(inputs, shape))
        got = mono_chunk_group_owner_cuda(inputs, shape, chunk_group_size=chunk_group_size)
        torch.cuda.synchronize()
        correctness["wave6_chunk_group_owner_vs_bf16_staged_torch_reference"] = _diffs(ref, got)

        fp32_ref = _collapse_dmimo_chunk(mono_state_lkq_d_torch_reference(inputs, shape))
        correctness["wave6_chunk_group_owner_vs_wave1_fp32_reference"] = _diffs(fp32_ref, got)
        metadata["wave6_chunk_group_owner"] = stage2_mono_chunk_group_owner_cuda_metadata(
            inputs["dout"],
            chunk_group_size=chunk_group_size,
        )

        dv_out, dmimo_out, dmimo_group_out, dssda_out = _empty_chunk_group_outputs(
            shape,
            dtype=dtype,
            device=device,
            chunk_group_size=chunk_group_size,
        )

        def run_chunk_group_owner() -> None:
            stage2_mono_chunk_group_owner_cuda_out(
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
                dmimo_v_delta=dmimo_out,
                dmimo_group_delta=dmimo_group_out,
                dssda_delta=dssda_out,
                chunk_size=shape.chunk,
                chunk_group_size=chunk_group_size,
            )

        timings["wave6_chunk_group_owner_slice"] = timer(
            run_chunk_group_owner,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        ref = _collapse_dmimo_chunk(mono_wmma_lkq_dphi_torch_reference(inputs, shape))
        correctness["torch_bf16_staged_scan_reference_self"] = _diffs(ref, ref)
        timings["torch_bf16_staged_scan_reference_state_lkq_d_subset"] = timer(
            lambda: _collapse_dmimo_chunk(mono_wmma_lkq_dphi_torch_reference(inputs, shape)),
            warmup=args.warmup,
            iters=args.iters,
        )

    scan_ms = timings.get("wave6_chunk_group_owner_slice", {}).get("mean_ms")
    if scan_ms and args.shape == "productionish":
        tilelang = COMPARISON_CONTEXT["tilelang_stage2_bf1_bb0_bwd_bwd_productionish_ms"]
        timings["wave6_chunk_group_owner_slice"]["ratio_vs_tilelang_bwd_bwd_prod"] = scan_ms / tilelang
        timings["wave6_chunk_group_owner_slice"]["speedup_vs_wave4_wmma_p64_prod"] = (
            WAVE4_WMMA_P64_PRODUCTIONISH_MS / scan_ms
        )
        timings["wave6_chunk_group_owner_slice"]["speedup_vs_wave5_scan_owner_prod"] = (
            WAVE5_SCAN_OWNER_PRODUCTIONISH_MS / scan_ms
        )
        timings["wave6_chunk_group_owner_slice"]["delta_ms_vs_wave4_wmma_p64_prod"] = (
            scan_ms - WAVE4_WMMA_P64_PRODUCTIONISH_MS
        )
        timings["wave6_chunk_group_owner_slice"]["delta_ms_vs_wave5_scan_owner_prod"] = (
            scan_ms - WAVE5_SCAN_OWNER_PRODUCTIONISH_MS
        )
        timings["wave6_chunk_group_owner_slice"]["chunk_group_owner_ctas_per_sm_at_132_sms"] = (
            shape.B * shape.H * _chunk_groups(shape, chunk_group_size) / 132.0
        )

    return {
        "shape_name": args.shape or "custom",
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "chunk_group_size": chunk_group_size,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "comparison_context": {
            "wave1_scalar_mono_state_lkq_d_productionish_ms": WAVE1_SCALAR_PRODUCTIONISH_MS,
            "wave2_wmma_lkq_dphi_chunk_owner_productionish_ms": WAVE2_WMMA_PRODUCTIONISH_MS,
            "wave3_wmma_triangular_lkq_dphi_chunk_owner_productionish_ms": WAVE3_WMMA_TRIANGULAR_PRODUCTIONISH_MS,
            "wave4_wmma_p64_panel_lkq_dphi_chunk_owner_productionish_ms": WAVE4_WMMA_P64_PRODUCTIONISH_MS,
            "wave5_scan_owner_productionish_ms": WAVE5_SCAN_OWNER_PRODUCTIONISH_MS,
            **COMPARISON_CONTEXT,
        },
        "cta_model": _cta_model(shape, chunk_group_size),
        "metadata": metadata,
        "correctness": correctness,
        "timings": timings,
        "read": [
            "Wave 6 is a chunk-group split owner: one CTA owns one (B,H,chunk_group), emits DV/DSSDA for its chunks, and writes one DMIMO_V partial.",
            "A second reduction kernel sums [B,H,chunk_groups,R,P] partials into final [B,H,R,P].",
            "LKQ is still computed once per chunk and reused across all P64 panels inside each group; causal pruning is intentionally not applied yet.",
            "The split recovers CTA count without giving up chunk-local LKQ reuse, but it adds global scratch traffic and a reduction launch for DMIMO_V.",
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
    parser.add_argument("--chunk-group-size", type=int, default=DEFAULT_CHUNK_GROUP_SIZE)
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
