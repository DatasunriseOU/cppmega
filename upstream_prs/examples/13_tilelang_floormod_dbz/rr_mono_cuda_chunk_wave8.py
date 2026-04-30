"""Wave 8 tile-stream tensor-core CUDA prototype for Mamba3 bwd_bwd.

Wave 8 keeps the P64 panel chunk-owner shape from the tensor-core waves, but
streams one 16x16 LKQ tile through shared memory.  The kernel uses WMMA for
DKI=PsiV@dPhi.T, LKQ tiles, state=K@dstates, and masked(LKQ tile)@dPhi.  A
small reduction kernel produces final DMIMO_V from per-chunk partials.
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
from rr_mono_cuda_chunk_wave8_extension import (  # noqa: E402
    stage2_mono_wmma_tile_stream_chunk_owner_cuda,
    stage2_mono_wmma_tile_stream_chunk_owner_cuda_metadata,
    stage2_mono_wmma_tile_stream_chunk_owner_cuda_out,
)


WAVE2_WMMA_PRODUCTIONISH_MS = 8.919168281555176
WAVE4_P64_PANEL_PRODUCTIONISH_MS = 8.784351921081543
WAVE5_SCAN_OWNER_PRODUCTIONISH_MS = 14.08131217956543
WAVE6_CHUNK_GROUP_OWNER_PRODUCTIONISH_MS = 14.515359878540039
WAVE7_ROW_STREAM_PRODUCTIONISH_MS = 179.76535034179688
P_PANEL = 64


def _empty_tile_stream_outputs(
    shape: Shape,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dv_delta = torch.empty(shape.B, shape.S, shape.H, shape.P, dtype=dtype, device=device)
    dmimo_v_delta = torch.empty(shape.B, shape.H, shape.R, shape.P, dtype=torch.float32, device=device)
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
    return dv_delta, dmimo_v_delta, dmimo_v_chunk_delta, dssda_delta


def _collapse_dmimo_chunk(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return outputs[0], outputs[1].sum(dim=2).contiguous(), outputs[2]


def mono_wmma_tile_stream_chunk_owner_cuda(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return stage2_mono_wmma_tile_stream_chunk_owner_cuda(
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
        "dmimo_v_delta": _max_diff(ref[1], got[1]),
        "dssda_delta": _max_diff(ref[2], got[2]),
    }


def _cta_model(shape: Shape) -> dict[str, Any]:
    chunk_ctas = shape.B * shape.H * shape.nchunks
    fused_chunk = shape.chunk * shape.R
    p_panels = shape.P // P_PANEL
    panel_ctas = chunk_ctas * p_panels
    smem_model_bytes = {
        "k_shared": fused_chunk * shape.N * 2,
        "q_shared": fused_chunk * shape.N * 2,
        "dphi_shared": fused_chunk * P_PANEL * 2,
        "psi_shared": fused_chunk * P_PANEL * 2,
        "dki_or_dpsi_reused_shared": fused_chunk * P_PANEL * 4,
        "lkq_tile_shared": 16 * 16 * 4,
        "masked_lkq_tile_shared": 16 * 16 * 2,
    }
    return {
        "owner": "B,H,chunk,P64-panel tile-stream",
        "chunk_owner_ctas": chunk_ctas,
        "p64_panel_owner_ctas": panel_ctas,
        "p64_panel_owner_ctas_per_sm_at_132_sms": panel_ctas / 132.0,
        "reduction_ctas": shape.B * shape.H * shape.R,
        "p_panel": P_PANEL,
        "p_panels_per_chunk": p_panels,
        "dmimo_chunk_scratch_elements": shape.B * shape.H * shape.nchunks * shape.R * shape.P,
        "live_intermediates": {
            "full_lkq_tile_elements": 0,
            "streamed_lkq_tile_elements": 16 * 16,
            "masked_lkq_tile_elements": 16 * 16,
            "reused_dki_or_dpsi_panel_elements": fused_chunk * P_PANEL,
        },
        "smem_model_bytes": smem_model_bytes,
        "smem_model_total_bytes": sum(smem_model_bytes.values()),
        "logical_chunk_visits": chunk_ctas,
        "outputs": ["DV", "DMIMO_V_final", "DSSDA"],
        "tradeoff": "keeps tensor-core DKI/state/LKQ consumers while replacing full LKQ residency with one streamed 16x16 LKQ tile and reusing the 64x64 float workspace across DKI and dPsi phases",
        "comparison_ms": {
            "wave1_scalar_mono_state_lkq_d": WAVE1_SCALAR_PRODUCTIONISH_MS,
            "wave2_wmma_chunk_owner": WAVE2_WMMA_PRODUCTIONISH_MS,
            "wave4_p64_panel_wmma": WAVE4_P64_PANEL_PRODUCTIONISH_MS,
            "wave5_scan_owner": WAVE5_SCAN_OWNER_PRODUCTIONISH_MS,
            "wave6_chunk_group_owner_group8": WAVE6_CHUNK_GROUP_OWNER_PRODUCTIONISH_MS,
            "wave7_row_stream": WAVE7_ROW_STREAM_PRODUCTIONISH_MS,
            **COMPARISON_CONTEXT,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = _shape_from_args(args)
    if shape.S % shape.chunk != 0:
        raise RuntimeError("Wave 8 harness currently expects S divisible by chunk")
    if shape.P % P_PANEL != 0:
        raise RuntimeError("Wave 8 tile-stream owner requires P to be a multiple of 64")
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    inputs = make_mono_inputs(shape, dtype=dtype, device=device, seed=args.seed)

    correctness: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    timer: Callable[..., dict[str, Any]] = _time_cuda if device.type == "cuda" else _time_wall

    if device.type == "cuda":
        if dtype != torch.bfloat16:
            raise RuntimeError("Wave 8 tile-stream owner CUDA path currently requires --dtype bf16")

        ref = _collapse_dmimo_chunk(mono_wmma_lkq_dphi_torch_reference(inputs, shape))
        got = mono_wmma_tile_stream_chunk_owner_cuda(inputs, shape)
        torch.cuda.synchronize()
        correctness["wave8_tile_stream_vs_bf16_staged_torch_reference"] = _diffs(ref, got)

        fp32_ref = _collapse_dmimo_chunk(mono_state_lkq_d_torch_reference(inputs, shape))
        correctness["wave8_tile_stream_vs_wave1_fp32_reference"] = _diffs(fp32_ref, got)
        metadata["wave8_wmma_tile_stream_chunk_owner"] = (
            stage2_mono_wmma_tile_stream_chunk_owner_cuda_metadata(inputs["dout"])
        )

        dv_out, dmimo_out, dmimo_chunk_out, dssda_out = _empty_tile_stream_outputs(
            shape,
            dtype=dtype,
            device=device,
        )

        def run_tile_stream() -> None:
            stage2_mono_wmma_tile_stream_chunk_owner_cuda_out(
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
                dmimo_v_chunk_delta=dmimo_chunk_out,
                dssda_delta=dssda_out,
                chunk_size=shape.chunk,
            )

        timings["wave8_wmma_tile_stream_chunk_owner_slice"] = timer(
            run_tile_stream,
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

    tile_ms = timings.get("wave8_wmma_tile_stream_chunk_owner_slice", {}).get("mean_ms")
    if tile_ms and args.shape == "productionish":
        tilelang = COMPARISON_CONTEXT["tilelang_stage2_bf1_bb0_bwd_bwd_productionish_ms"]
        timings["wave8_wmma_tile_stream_chunk_owner_slice"]["ratio_vs_tilelang_bwd_bwd_prod"] = (
            tile_ms / tilelang
        )
        timings["wave8_wmma_tile_stream_chunk_owner_slice"]["speedup_vs_wave2_wmma_prod"] = (
            WAVE2_WMMA_PRODUCTIONISH_MS / tile_ms
        )
        timings["wave8_wmma_tile_stream_chunk_owner_slice"]["speedup_vs_wave4_p64_panel_prod"] = (
            WAVE4_P64_PANEL_PRODUCTIONISH_MS / tile_ms
        )
        timings["wave8_wmma_tile_stream_chunk_owner_slice"]["speedup_vs_wave5_scan_owner_prod"] = (
            WAVE5_SCAN_OWNER_PRODUCTIONISH_MS / tile_ms
        )
        timings["wave8_wmma_tile_stream_chunk_owner_slice"]["speedup_vs_wave7_row_stream_prod"] = (
            WAVE7_ROW_STREAM_PRODUCTIONISH_MS / tile_ms
        )
        timings["wave8_wmma_tile_stream_chunk_owner_slice"]["delta_ms_vs_wave5_scan_owner_prod"] = (
            tile_ms - WAVE5_SCAN_OWNER_PRODUCTIONISH_MS
        )
        timings["wave8_wmma_tile_stream_chunk_owner_slice"]["delta_ms_vs_wave7_row_stream_prod"] = (
            tile_ms - WAVE7_ROW_STREAM_PRODUCTIONISH_MS
        )
        timings["wave8_wmma_tile_stream_chunk_owner_slice"]["p64_panel_owner_ctas_per_sm_at_132_sms"] = (
            shape.B * shape.H * shape.nchunks * (shape.P // P_PANEL) / 132.0
        )

    return {
        "shape_name": args.shape or "custom",
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "comparison_context": {
            "wave1_scalar_mono_state_lkq_d_productionish_ms": WAVE1_SCALAR_PRODUCTIONISH_MS,
            "wave2_wmma_lkq_dphi_chunk_owner_productionish_ms": WAVE2_WMMA_PRODUCTIONISH_MS,
            "wave4_p64_panel_wmma_productionish_ms": WAVE4_P64_PANEL_PRODUCTIONISH_MS,
            "wave5_scan_owner_productionish_ms": WAVE5_SCAN_OWNER_PRODUCTIONISH_MS,
            "wave6_chunk_group_owner_group8_productionish_ms": WAVE6_CHUNK_GROUP_OWNER_PRODUCTIONISH_MS,
            "wave7_row_stream_productionish_ms": WAVE7_ROW_STREAM_PRODUCTIONISH_MS,
            **COMPARISON_CONTEXT,
        },
        "cta_model": _cta_model(shape),
        "metadata": metadata,
        "correctness": correctness,
        "timings": timings,
        "read": [
            "Wave 8 is a tile-stream tensor-core probe: one CTA owns one (B,H,chunk,P64-panel).",
            "The CTA stages K/Q, dPhi, and PsiV, computes DKI with WMMA, then streams one 16x16 LKQ tile for DSSDA.",
            "The same float workspace is reused for state/dPsi before a second LKQ tile stream feeds masked(LKQ)@dPhi WMMA.",
            "DV and DSSDA are emitted by the panel-owner kernel; DMIMO_V uses per-chunk scratch plus a small final reduction kernel.",
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
