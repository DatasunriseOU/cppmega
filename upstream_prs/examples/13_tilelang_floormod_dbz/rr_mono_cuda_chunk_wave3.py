"""Wave 3 tensor-core CUDA chunk-owner prototype for Mamba3 bwd_bwd.

Wave 3 keeps the Wave 2 WMMA output contract and applies a schedule-only
change in the CUDA kernel:

* masked LKQ reuses the dead Q^T bf16 shared-memory tile storage;
* masked(LKQ) @ dPhi skips tile-k blocks that are fully below the causal
  triangle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rr_mono_cuda_chunk_wave2 as _wave2
from rr_mono_cuda_chunk_wave3_extension import (
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


def _rename_key(mapping: dict[str, object], old: str, new: str) -> None:
    if old in mapping:
        mapping[new] = mapping.pop(old)


def run(args):
    result = _run_wave2_contract(args)
    result["comparison_context"]["wave2_wmma_lkq_dphi_chunk_owner_productionish_ms"] = (
        WAVE2_WMMA_PRODUCTIONISH_MS
    )

    _rename_key(
        result["metadata"],
        "wave2_wmma_lkq_dphi_chunk_owner",
        "wave3_wmma_triangular_lkq_dphi_chunk_owner",
    )
    _rename_key(
        result["correctness"],
        "wave2_wmma_lkq_dphi_vs_bf16_staged_torch_reference",
        "wave3_wmma_triangular_vs_bf16_staged_torch_reference",
    )
    _rename_key(
        result["correctness"],
        "wave2_wmma_lkq_dphi_vs_wave1_fp32_reference",
        "wave3_wmma_triangular_vs_wave1_fp32_reference",
    )
    _rename_key(
        result["timings"],
        "wave2_wmma_lkq_dphi_chunk_owner_slice",
        "wave3_wmma_triangular_lkq_dphi_chunk_owner_slice",
    )

    shape = result["shape"]
    m_tiles = (shape["chunk"] * shape["R"]) // 16
    p_tiles = shape["P"] // 16
    dense_lkq_apply_tiles = m_tiles * m_tiles * p_tiles
    triangular_lkq_apply_tiles = (m_tiles * (m_tiles + 1) // 2) * p_tiles
    result["cta_model"]["wmma_tiles_per_cta"]["lkq_dphi_apply"] = triangular_lkq_apply_tiles
    result["cta_model"]["wmma_tiles_per_cta"]["lkq_dphi_apply_dense_wave2"] = dense_lkq_apply_tiles
    result["cta_model"]["wmma_tiles_per_cta"]["lkq_dphi_apply_skipped_by_wave3"] = (
        dense_lkq_apply_tiles - triangular_lkq_apply_tiles
    )

    timing = result["timings"].get("wave3_wmma_triangular_lkq_dphi_chunk_owner_slice")
    if timing and args.shape == "productionish":
        mean_ms = timing.get("mean_ms")
        if mean_ms:
            timing["speedup_vs_wave2_wmma_prod"] = WAVE2_WMMA_PRODUCTIONISH_MS / mean_ms
            timing["delta_ms_vs_wave2_wmma_prod"] = mean_ms - WAVE2_WMMA_PRODUCTIONISH_MS

    result["read"] = [
        "Wave 3 keeps the Wave 2 chunk-owner WMMA contract and output tensors.",
        "The CUDA schedule reuses dead Q^T shared memory for masked LKQ after LKQ is materialized.",
        "The masked LKQ @ dPhi WMMA apply skips tile-k blocks that are fully below the causal triangle.",
        "No scalar matrix-loop fallback is introduced; LKQ, dki, state, and masked LKQ apply remain tensor-core WMMA.",
    ]
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
