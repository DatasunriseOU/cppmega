"""Wave 4 P64-panel tensor-core CUDA prototype for Mamba3 bwd_bwd.

Wave 4 keeps the Wave 3 WMMA math but changes ownership from one CTA per
``(B,H,chunk)`` to one CTA per ``(B,H,chunk,P64-panel)``:

* the production P=128 shape launches two P64 panel CTAs per logical chunk;
* per-panel CTAs write their DV/DMIMO_V P slice directly;
* DSSDA is accumulated across P64 panels inside the kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rr_mono_cuda_chunk_wave2 as _wave2
from rr_mono_cuda_chunk_wave4_extension import (
    stage2_mono_wmma_lkq_dphi_chunk_owner_cuda,
    stage2_mono_wmma_lkq_dphi_chunk_owner_cuda_metadata,
    stage2_mono_wmma_lkq_dphi_chunk_owner_cuda_out,
)


_wave2.stage2_mono_wmma_lkq_dphi_chunk_owner_cuda = stage2_mono_wmma_lkq_dphi_chunk_owner_cuda
_wave2.stage2_mono_wmma_lkq_dphi_chunk_owner_cuda_metadata = (
    stage2_mono_wmma_lkq_dphi_chunk_owner_cuda_metadata
)
_wave2.stage2_mono_wmma_lkq_dphi_chunk_owner_cuda_out = stage2_mono_wmma_lkq_dphi_chunk_owner_cuda_out

COMPARISON_CONTEXT = _wave2.COMPARISON_CONTEXT
WAVE1_SCALAR_PRODUCTIONISH_MS = _wave2.WAVE1_SCALAR_PRODUCTIONISH_MS
parse_args = _wave2.parse_args
_run_wave2_contract = _wave2.run

WAVE2_WMMA_PRODUCTIONISH_MS = 8.919168281555176
WAVE3_WMMA_TRIANGULAR_PRODUCTIONISH_MS = 8.467136001586914
P_PANEL = 64


def _rename_key(mapping: dict[str, object], old: str, new: str) -> None:
    if old in mapping:
        mapping[new] = mapping.pop(old)


def run(args):
    result = _run_wave2_contract(args)
    result["comparison_context"]["wave2_wmma_lkq_dphi_chunk_owner_productionish_ms"] = (
        WAVE2_WMMA_PRODUCTIONISH_MS
    )
    result["comparison_context"]["wave3_wmma_triangular_lkq_dphi_chunk_owner_productionish_ms"] = (
        WAVE3_WMMA_TRIANGULAR_PRODUCTIONISH_MS
    )

    _rename_key(
        result["metadata"],
        "wave2_wmma_lkq_dphi_chunk_owner",
        "wave4_wmma_p64_panel_lkq_dphi_chunk_owner",
    )
    _rename_key(
        result["correctness"],
        "wave2_wmma_lkq_dphi_vs_bf16_staged_torch_reference",
        "wave4_wmma_p64_panel_vs_bf16_staged_torch_reference",
    )
    _rename_key(
        result["correctness"],
        "wave2_wmma_lkq_dphi_vs_wave1_fp32_reference",
        "wave4_wmma_p64_panel_vs_wave1_fp32_reference",
    )
    _rename_key(
        result["timings"],
        "wave2_wmma_lkq_dphi_chunk_owner_slice",
        "wave4_wmma_p64_panel_lkq_dphi_chunk_owner_slice",
    )
    _rename_key(
        result["timings"],
        "wave10_two_launch_plus_wave2_wmma_projection",
        "wave10_two_launch_plus_wave4_wmma_p64_panel_projection",
    )
    projection = result["timings"].get("wave10_two_launch_plus_wave4_wmma_p64_panel_projection")
    if projection:
        projection["note"] = (
            "Projection only: wave10 covered subset and Wave 4 P64-panel state/LKQ/D subset are not fused together yet."
        )

    shape = result["shape"]
    m_tiles = (shape["chunk"] * shape["R"]) // 16
    p_panels = shape["P"] // P_PANEL
    p_panel_tiles = P_PANEL // 16
    dense_lkq_apply_tiles_per_panel = m_tiles * m_tiles * p_panel_tiles
    triangular_lkq_apply_tiles_per_panel = (m_tiles * (m_tiles + 1) // 2) * p_panel_tiles
    chunk_ctas = shape["B"] * shape["H"] * ((shape["S"] + shape["chunk"] - 1) // shape["chunk"])
    result["cta_model"]["chunk_owner_ctas"] = chunk_ctas
    result["cta_model"]["p64_panel_owner_ctas"] = chunk_ctas * p_panels
    result["cta_model"]["p_panel"] = P_PANEL
    result["cta_model"]["p_panels_per_logical_chunk"] = p_panels
    result["cta_model"]["wmma_tiles_per_cta"] = {
        "lkq_k_qt": m_tiles * m_tiles * (shape["N"] // 16),
        "dki_psiv_dphi_t": m_tiles * m_tiles * p_panel_tiles,
        "state_k_dstates": m_tiles * p_panel_tiles * (shape["N"] // 16),
        "lkq_dphi_apply": triangular_lkq_apply_tiles_per_panel,
        "lkq_dphi_apply_dense_wave2_per_panel": dense_lkq_apply_tiles_per_panel,
        "lkq_dphi_apply_skipped_by_triangular_prune_per_panel": (
            dense_lkq_apply_tiles_per_panel - triangular_lkq_apply_tiles_per_panel
        ),
    }
    result["cta_model"]["wmma_tiles_per_logical_chunk"] = {
        name: value * p_panels for name, value in result["cta_model"]["wmma_tiles_per_cta"].items()
    }
    result["cta_model"]["dssda_cross_panel"] = "atomicAdd partial DSSDA when P has more than one P64 panel"

    timing = result["timings"].get("wave4_wmma_p64_panel_lkq_dphi_chunk_owner_slice")
    if timing and args.shape == "productionish":
        mean_ms = timing.get("mean_ms")
        if mean_ms:
            timing["speedup_vs_wave2_wmma_prod"] = WAVE2_WMMA_PRODUCTIONISH_MS / mean_ms
            timing["delta_ms_vs_wave2_wmma_prod"] = mean_ms - WAVE2_WMMA_PRODUCTIONISH_MS
            timing["speedup_vs_wave3_wmma_triangular_prod"] = (
                WAVE3_WMMA_TRIANGULAR_PRODUCTIONISH_MS / mean_ms
            )
            timing["delta_ms_vs_wave3_wmma_triangular_prod"] = (
                mean_ms - WAVE3_WMMA_TRIANGULAR_PRODUCTIONISH_MS
            )

    result["read"] = [
        "Wave 4 keeps the Wave 3 WMMA tensor-core math and output tensors.",
        "The schedule splits P into fixed P64 panels so production P=128 uses two panel-owner CTAs per logical chunk.",
        "Each panel CTA keeps the P64 dPhi/Psi/dPsi working set in shared memory and writes its DV/DMIMO_V slice directly.",
        "DSSDA is accumulated across panels in-kernel; P64 smoke stores directly, production P128 uses atomicAdd partials.",
    ]
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
