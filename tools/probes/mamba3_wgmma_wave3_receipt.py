"""Emit the Wave3 WGMMA/CuTe design receipt for Mamba3 mono bwd_bwd.

This is a CPU-only ledger check.  It imports the Wave2 Triton cost model and
adds the Lane-D schedule metadata and A/B gate budgets used by the Wave3 doc.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "upstream_prs" / "examples" / "13_tilelang_floormod_dbz"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from rr_diag_wave2_mono_triton_pruned_model import Shape, _fma_model, _memory_model  # noqa: E402


def _shape() -> Shape:
    return Shape(B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16)


def _bytes_model(shape: Shape) -> dict[str, int]:
    bf16 = 2
    fp32 = 4
    chunks = shape.B * shape.H * shape.nchunks
    scalar_bhs = shape.B * shape.H * shape.S * fp32
    out = {
        "dv_output_bytes": shape.B * shape.S * shape.H * shape.P * bf16,
        "dk_output_bytes": shape.B * shape.S * shape.R * shape.H * shape.N * bf16,
        "dq_output_bytes": shape.B * shape.S * shape.R * shape.H * shape.N * bf16,
        "dmimov_partial_bytes": shape.B * shape.H * shape.nchunks * shape.R * shape.P * fp32,
        "dmimov_output_bytes": shape.B * shape.H * shape.R * shape.P * fp32,
        "five_scalar_bhs_bytes": 5 * scalar_bhs,
        "dssda_output_bytes": shape.B * shape.H * shape.nchunks * shape.chunk * shape.chunk * fp32,
        "dangles_output_bytes": shape.B * shape.S * shape.H * (shape.N // 4) * fp32,
    }
    out["scalar_outputs_bytes"] = (
        out["five_scalar_bhs_bytes"] + out["dssda_output_bytes"] + out["dangles_output_bytes"]
    )
    out["chunk_owner_required_output_write_bytes"] = (
        out["dv_output_bytes"]
        + out["dk_output_bytes"]
        + out["dq_output_bytes"]
        + out["dmimov_partial_bytes"]
        + out["dmimov_output_bytes"]
        + out["scalar_outputs_bytes"]
    )
    out["scan_owner_required_output_write_bytes"] = (
        out["dv_output_bytes"]
        + out["dk_output_bytes"]
        + out["dq_output_bytes"]
        + out["dmimov_output_bytes"]
        + out["scalar_outputs_bytes"]
    )
    out["dmimov_reducer_extra_rw_bytes"] = 2 * out["dmimov_partial_bytes"] + out["dmimov_output_bytes"]
    out["chunk_body_count"] = chunks
    return out


def _mib(value: int) -> float:
    return value / float(1024**2)


def build_receipt() -> dict[str, Any]:
    shape = _shape()
    fma = _fma_model(shape, block_p=shape.P)
    memory = _memory_model(shape)
    bytes_model = _bytes_model(shape)

    optional_dstates_update_fma = (
        shape.B * shape.H * shape.nchunks * shape.N * shape.fcs * shape.P
    )
    scan_owner_ideal_with_dstates = (
        fma["monolithic_causal_apply_total_fma"] + optional_dstates_update_fma
    )

    return {
        "receipt": "mamba3_mono_wgmma_plan_wave3_2026_04_30",
        "status": "design_receipt_for_lane_d",
        "date": "2026-04-30",
        "branch": "worker/mamba3-mono-triton-model",
        "source_models": {
            "wave1_doc": "docs/status/mamba3_mono_triton_reuse_wave1_2026_04_30.md",
            "wave2_doc": "docs/status/mamba3_mono_triton_pruned_wave2_2026_04_30.md",
            "wave1_model": "upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave1_mono_triton_reuse_model.py",
            "wave2_model": "upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave2_mono_triton_pruned_model.py",
        },
        "shape": {
            "B": shape.B,
            "S": shape.S,
            "H": shape.H,
            "G": shape.G,
            "N": shape.N,
            "P": shape.P,
            "R": shape.R,
            "chunk": shape.chunk,
            "nchunks": shape.nchunks,
            "fused_chunk_rows": shape.fcs,
            "chunk_bodies": bytes_model["chunk_body_count"],
        },
        "reference_timings_ms": {
            "tilelang_stage2_bf1_bb0_bwd_bwd": 3.70674,
            "wave1_triton_full_mask_checksum_lower_bound": 4.53881,
            "wave2_triton_tile_pruned_checksum_lower_bound": 8.79331,
            "wave8_cuda_diag_qk_dmimov_before_state_lkq_d": 2.45093,
        },
        "modeled_fma": {
            "separate_recompute": fma["separate_recompute_total_fma"],
            "monolithic_full_mask": fma["monolithic_full_mask_total_fma"],
            "wave2_4step_tile_pruned": fma["monolithic_tile_pruned_total_fma"],
            "ideal_triangular_apply": fma["monolithic_causal_apply_total_fma"],
            "optional_scan_owner_dstates_update": optional_dstates_update_fma,
            "scan_owner_ideal_plus_dstates_update": scan_owner_ideal_with_dstates,
            "reuse_savings_vs_separate": fma["reuse_savings_full_mask_fma"],
            "triangular_savings_vs_separate": fma["causal_apply_savings_vs_separate_fma"],
            "tile_pruned_over_ideal": fma["tile_pruned_over_ideal_causal_fma"],
            "components_ideal_triangular": fma["monolithic_causal_apply_fma"],
            "components_wave2_4step_tile_pruned": fma["monolithic_tile_pruned_fma"],
        },
        "modeled_bytes": {
            "dv_output_bytes": bytes_model["dv_output_bytes"],
            "dk_output_bytes": bytes_model["dk_output_bytes"],
            "dq_output_bytes": bytes_model["dq_output_bytes"],
            "dmimov_partial_bytes": bytes_model["dmimov_partial_bytes"],
            "dmimov_output_bytes": bytes_model["dmimov_output_bytes"],
            "scalar_outputs_bytes": bytes_model["scalar_outputs_bytes"],
            "chunk_owner_required_output_write_bytes": bytes_model[
                "chunk_owner_required_output_write_bytes"
            ],
            "scan_owner_required_output_write_bytes": bytes_model[
                "scan_owner_required_output_write_bytes"
            ],
            "dmimov_reducer_extra_rw_bytes": bytes_model["dmimov_reducer_extra_rw_bytes"],
            "chunk_owner_required_output_write_mib": _mib(
                bytes_model["chunk_owner_required_output_write_bytes"]
            ),
            "scan_owner_required_output_write_mib": _mib(
                bytes_model["scan_owner_required_output_write_bytes"]
            ),
            "dmimov_reducer_extra_rw_mib": _mib(bytes_model["dmimov_reducer_extra_rw_bytes"]),
            "global_temps_avoided_by_cuda_owner_mib": memory[
                "global_temps_avoided_by_cuda_owner_mib"
            ],
        },
        "wgmma_schedule": {
            "preferred_cta_owner": "one CTA owns one (B,H) stream and iterates chunks in reverse",
            "fallback_cta_owner": "one CTA owns one (B,H,chunk) body if dstates is precomputed upstream",
            "cta_count_preferred": shape.B * shape.H,
            "chunk_iterations_per_cta": shape.nchunks,
            "gmma_atom": "SM90 BF16 GMMA m64n64k16 -> fp32 accum",
            "p_tiling": "P=128 split into two n64 panels; do not use one resident m64n128 accumulator for all consumers",
            "k_tiling": "all hot products use K=64 or K=128 as four/eight k16 GMMA groups",
            "triangular_schedule": "4x4 timestep tiles; compute all LKQ/dk_intra for DSSDA, apply only tiles with row_time < col_time, split diagonal into 4x4 lane subtiles",
            "no_duplicate_fma_requirements": [
                "LKQ is built once per chunk body",
                "dk_intra is built once per chunk body",
                "dki.T for DQ is supplied by a transpose smem view/copy, not by a second GMMA",
                "below-frontier causal apply tiles issue no GMMA",
            ],
        },
        "required_outputs": [
            "DV[B,S,H,P] bf16",
            "DK[B,S*R,H,N] bf16",
            "DQ[B,S*R,H,N] bf16",
            "DMIMO_V[B,H,R,P] fp32",
            "DDA_CS[B,H,S] fp32",
            "DDA_CS_REV[B,H,S] fp32",
            "DFACTOR[B,H,S] fp32",
            "DGAMMA_DIAG[B,H,S] fp32",
            "DSSDA[B,H,nchunks,chunk,chunk] fp32",
            "DDA[B,H,S] fp32",
            "DANGLES[B,S,H,N/4] fp32",
        ],
        "ab_gate_budget": {
            "green_full_kernel_ms": 3.35,
            "yellow_full_kernel_ms": 3.70674,
            "red_full_kernel_ms": 3.70674,
            "chunk_owner_main_body_ms": 3.20,
            "chunk_owner_dmimov_reducer_ms": 0.05,
            "scan_owner_main_body_ms": 3.30,
            "notes": [
                "Green is roughly 10% faster than TileLang 3.70674 ms.",
                "Yellow means faster than TileLang but without enough margin to ship.",
                "Any duplicate LKQ/dk_intra/dki.T GMMA path fails the A gate regardless of timing.",
            ],
        },
    }


def _canonical(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=pathlib.Path, default=None)
    args = parser.parse_args()

    receipt = build_receipt()
    rendered = _canonical(receipt)
    if args.check is None:
        print(rendered, end="")
        return

    actual = args.check.read_text()
    if actual != rendered:
        print(f"{args.check} does not match generated receipt", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
