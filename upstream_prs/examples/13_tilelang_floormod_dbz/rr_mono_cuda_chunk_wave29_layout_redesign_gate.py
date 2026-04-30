"""Wave 29 layout redesign gate for the Mamba3 monolithic CUDA path.

This probe intentionally does not extend the dead Wave28 P64-global-scratch or
P128-all-P scratch layouts.  It models materially different ownership choices:

* Hopper cluster/DSM cross-P reduction between two P64 panel CTAs.
* split output owners with attempted on-chip reuse.
* diagonal-only DK/DQ avoidance.
* CuTe/Triton hybrid ownership.

The goal is to decide whether any redesign deserves a CUDA implementation
before spending GPU time.  The lower bound is the measured Wave8 tile-stream
subset: it already computes only DV/final-DMIMO_V/DSSDA and is 3.0x slower than
the TileLang full bwd_bwd baseline at the productionish shape.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


BF16 = 2
FP32 = 4
PANEL = 64

COMPARISON_MS = {
    "tilelang_full_bwd_bwd_h200": 3.70674,
    "wave8_tile_stream_subset_h200": 11.1550079981486,
    "wave10_one_launch_diag_qk_dmimo_subset_h200": 2.31212,
    "wave7_scalar_row_stream_subset_h200": 179.76535034179688,
}


@dataclass(frozen=True)
class Shape:
    B: int = 4
    S: int = 4096
    H: int = 32
    G: int = 1
    N: int = 64
    P: int = 128
    R: int = 4
    chunk: int = 16

    @property
    def nchunks(self) -> int:
        return self.S // self.chunk

    @property
    def fused_chunk(self) -> int:
        return self.chunk * self.R

    @property
    def logical_chunks(self) -> int:
        return self.B * self.H * self.nchunks

    @property
    def panels(self) -> int:
        return math.ceil(self.P / PANEL)


def gib(nbytes: int | float) -> float:
    return float(nbytes) / 1024.0 / 1024.0 / 1024.0


def mib(nbytes: int | float) -> float:
    return float(nbytes) / 1024.0 / 1024.0


def wave8_subset_smem(shape: Shape) -> dict[str, int]:
    fcs = shape.fused_chunk
    return {
        "k_post_shared_bf16": fcs * shape.N * BF16,
        "q_post_t_shared_bf16": shape.N * fcs * BF16,
        "dphi_panel_shared_bf16": fcs * PANEL * BF16,
        "psi_panel_shared_bf16": fcs * PANEL * BF16,
        "reused_dki_or_dpsi_panel_fp32": fcs * PANEL * FP32,
        "lkq_stream_tile_fp32": 16 * 16 * FP32,
        "masked_lkq_tile_bf16": 16 * 16 * BF16,
    }


def p64_global_partial_bytes(shape: Shape) -> dict[str, int]:
    dk_dq = shape.logical_chunks * shape.panels * shape.fused_chunk * shape.N * FP32 * 2
    six_scalars = shape.logical_chunks * shape.panels * shape.chunk * FP32 * 6
    dmimo = shape.logical_chunks * shape.panels * shape.R * shape.P * FP32
    return {
        "dk_dq_partial_fp32_bytes": dk_dq,
        "six_scalar_partial_fp32_bytes": six_scalars,
        "dmimo_partial_fp32_bytes": dmimo,
        "total_bytes": dk_dq + six_scalars + dmimo,
    }


def p128_all_p_smem(shape: Shape, *, keep_prerot: bool) -> dict[str, int]:
    fcs = shape.fused_chunk
    items = {
        "k_post_shared_bf16": fcs * shape.N * BF16,
        "q_post_shared_bf16": fcs * shape.N * BF16,
        "dphi_full_p_shared_bf16": fcs * shape.P * BF16,
        "psi_full_p_shared_bf16": fcs * shape.P * BF16,
        "states_full_p_shared_bf16": shape.N * shape.P * BF16,
        "dstates_full_p_shared_bf16": shape.N * shape.P * BF16,
        "dpsi_or_dv_workspace_fp32": fcs * shape.P * FP32,
        "lkq_or_dqk_workspace_fp32": fcs * fcs * FP32,
        "dk_or_dq_workspace_fp32": fcs * shape.N * FP32,
        "masked_lkq_tile_bf16": 16 * 16 * BF16,
        "scalar_scratch_fp32": shape.chunk * 8 * FP32,
    }
    if keep_prerot:
        items |= {
            "q_pre_rot_shared_bf16": fcs * shape.N * BF16,
            "k_pre_rot_shared_bf16": fcs * shape.N * BF16,
            "k_pre_trap_shared_bf16": fcs * shape.N * BF16,
        }
    return items


def cluster_cross_p_reduction(shape: Shape, h100_smem_limit: int) -> dict[str, Any]:
    """Model a Hopper cluster where each P64 CTA publishes DK/DQ partials in DSM."""

    per_cta = wave8_subset_smem(shape)
    per_cta_total = sum(per_cta.values())
    dk_or_dq_partial = shape.fused_chunk * shape.N * FP32
    cluster_dsm_for_one_output = shape.panels * dk_or_dq_partial
    cluster_dsm_for_two_outputs = 2 * cluster_dsm_for_one_output
    scalar_partials = shape.panels * shape.chunk * 6 * FP32

    # The existing reused float workspace can hold one [fused_chunk, N] partial
    # when N=64 and PANEL=64, but every panel must keep its partial live across a
    # cluster barrier while a leader reads peer DSM.  That preserves the Wave8
    # subset work and adds two publish/barrier/reduce phases for DK and DQ.
    return {
        "owner": "cluster(B,H,chunk) with one CTA per P64 panel",
        "cluster_size_ctas": shape.panels,
        "ctas": shape.logical_chunks * shape.panels,
        "per_cta_smem_bytes": per_cta_total,
        "per_cta_smem_kib": per_cta_total / 1024.0,
        "cluster_smem_bytes": per_cta_total * shape.panels,
        "cluster_smem_kib": per_cta_total * shape.panels / 1024.0,
        "fits_one_cta_per_sm_by_smem": per_cta_total <= h100_smem_limit,
        "dsm_partial_bytes_if_dk_then_dq": cluster_dsm_for_one_output + scalar_partials,
        "dsm_partial_bytes_if_dk_and_dq_together": cluster_dsm_for_two_outputs + scalar_partials,
        "dsm_one_output_mib_per_cluster": mib(cluster_dsm_for_one_output + scalar_partials),
        "global_partial_bytes_avoided": p64_global_partial_bytes(shape),
        "lower_bound_ms": COMPARISON_MS["wave8_tile_stream_subset_h200"],
        "required_speedup_to_match_tilelang": (
            COMPARISON_MS["wave8_tile_stream_subset_h200"] / COMPARISON_MS["tilelang_full_bwd_bwd_h200"]
        ),
        "new_work": [
            "two cluster barriers minimum for DK and DQ reductions",
            f"leader/peer DSM reads over {shape.panels} P panels for each [64,64] fp32 partial",
            "same Wave8 DV/DMIMO/DSSDA subset remains in the critical path",
        ],
        "verdict": "no_go",
        "reason": (
            f"DSM can replace the {gib(p64_global_partial_bytes(shape)['total_bytes']):.6g} GiB "
            "global partial roundtrip, but it cannot make the measured Wave8 subset "
            "disappear.  The redesign starts from 11.155 ms before adding DK/DQ/scalar "
            "reductions, already 3.009x over the 3.70674 ms TileLang full baseline."
        ),
    }


def split_output_owner(shape: Shape) -> dict[str, Any]:
    fcs = shape.fused_chunk
    lkq_tile_ops_per_chunk = (fcs // 16) * (fcs // 16)
    p_panel_tensor_ops_per_chunk = shape.panels * lkq_tile_ops_per_chunk
    return {
        "owner": "separate DV/DMIMO/DSSDA P-panel owner plus DK/DQ/scalar owner",
        "on_chip_reuse_possible_between_owners": False,
        "reason_on_chip_reuse_fails": (
            "CTA-local shared memory cannot be consumed by a different non-cluster "
            "output owner.  A split owner must either recompute LKQ/state/D or write "
            "intermediates to global memory."
        ),
        "extra_lkq_16x16_wmma_ops_per_logical_chunk_if_recomputed": p_panel_tensor_ops_per_chunk,
        "extra_lkq_16x16_wmma_ops_productionish": shape.logical_chunks * p_panel_tensor_ops_per_chunk,
        "global_lkq_if_materialized_bytes": shape.logical_chunks * fcs * fcs * FP32,
        "global_lkq_if_materialized_gib": gib(shape.logical_chunks * fcs * fcs * FP32),
        "lower_bound_ms": COMPARISON_MS["wave8_tile_stream_subset_h200"],
        "verdict": "no_go",
        "reason": (
            "A split owner is no longer monolithic on-chip reuse.  Recompute adds a "
            "second LKQ/state pass; materialization adds about 0.5 GiB of fp32 LKQ "
            "traffic for the productionish shape.  Both are additions to a subset "
            "that is already slower than TileLang full bwd_bwd."
        ),
    }


def diagonal_only_avoidance(shape: Shape) -> dict[str, Any]:
    diag = shape.logical_chunks * shape.chunk * shape.R * shape.R * FP32
    full = shape.logical_chunks * shape.fused_chunk * shape.fused_chunk * FP32
    return {
        "owner": "avoid full DK/DQ partials by keeping only per-token R x R diagonal qk_dot",
        "diag_qk_bytes": diag,
        "full_fused_qk_bytes": full,
        "diag_vs_full_fraction": diag / full,
        "wave10_subset_ms": COMPARISON_MS["wave10_one_launch_diag_qk_dmimo_subset_h200"],
        "verdict": "incomplete_contract",
        "reason": (
            "This is a good covered-subset direction, but it does not compute the "
            "non-diagonal DK/DQ terms required by the full bwd_bwd contract.  It "
            "should remain a separate diagonal/qk CUDA piece, not a monolithic full "
            "replacement."
        ),
    }


def cute_triton_hybrid(shape: Shape) -> dict[str, Any]:
    return {
        "owner": "CuTe/Triton full state-LKQ-D kernel plus CUDA diagonal/output helpers",
        "monolithic_cuda_chunk_owner": False,
        "fits_wave29_lane_d_request": False,
        "why_not_a_lane_d_cuda_owner": (
            "The hybrid can use WGMMA/TMA-friendly full-P layouts and global kernel "
            "boundaries.  Those are exactly the properties the monolithic CUDA owner "
            "was trying to avoid, so this belongs to the TileLang/CuTe production "
            "path rather than this CUDA chunk branch."
        ),
        "verdict": "redirect",
        "reason": (
            "This is the only direction with a credible full-contract performance "
            "story, but it is not a monolithic CUDA chunk-owner redesign."
        ),
    }


def report(shape: Shape, h100_smem_limit: int, gb10_smem_limit: int) -> dict[str, Any]:
    p128_recompute = p128_all_p_smem(shape, keep_prerot=False)
    p128_reuse = p128_all_p_smem(shape, keep_prerot=True)
    p64 = wave8_subset_smem(shape)
    partials = p64_global_partial_bytes(shape)
    return {
        "shape": asdict(shape)
        | {
            "nchunks": shape.nchunks,
            "fused_chunk": shape.fused_chunk,
            "logical_chunks": shape.logical_chunks,
            "p64_panels": shape.panels,
        },
        "comparison_ms": COMPARISON_MS
        | {
            "wave8_subset_ratio_vs_tilelang_full": (
                COMPARISON_MS["wave8_tile_stream_subset_h200"]
                / COMPARISON_MS["tilelang_full_bwd_bwd_h200"]
            )
        },
        "wave28_binding_facts": {
            "p64_panel_smem_bytes": sum(p64.values()),
            "p64_panel_smem_kib": sum(p64.values()) / 1024.0,
            "p64_global_partial_total_bytes": partials["total_bytes"],
            "p64_global_partial_total_gib": gib(partials["total_bytes"]),
            "p64_dk_dq_partial_gib": gib(partials["dk_dq_partial_fp32_bytes"]),
            "p128_recompute_prerot_smem_bytes": sum(p128_recompute.values()),
            "p128_recompute_prerot_smem_kib": sum(p128_recompute.values()) / 1024.0,
            "p128_reuse_prerot_smem_bytes": sum(p128_reuse.values()),
            "p128_reuse_prerot_smem_kib": sum(p128_reuse.values()) / 1024.0,
            "p128_reuse_active_blocks_per_sm_h100_smem": h100_smem_limit // sum(p128_reuse.values()),
            "p128_reuse_active_blocks_per_sm_gb10_smem": gb10_smem_limit // sum(p128_reuse.values()),
        },
        "redesigns": {
            "cluster_cross_p_reduction": cluster_cross_p_reduction(shape, h100_smem_limit),
            "split_output_owner": split_output_owner(shape),
            "diagonal_only_avoidance": diagonal_only_avoidance(shape),
            "cute_triton_hybrid": cute_triton_hybrid(shape),
        },
        "decision": {
            "verdict": "hard_no_go_for_monolithic_cuda_layout_redesign",
            "deserves_another_wave": False,
            "reason": (
                "Every full-contract monolithic CUDA redesign either contains the "
                "measured 11.155 ms Wave8 subset as a lower bound and adds work, or "
                "stops being a monolithic on-chip reuse owner.  The only credible "
                "continuation is the non-monolithic TileLang/CuTe/CUTLASS-class path "
                "with small CUDA covered-subset helpers."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--S", type=int, default=4096)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--G", type=int, default=1)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--P", type=int, default=128)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--h100-smem-limit", type=int, default=227 * 1024)
    parser.add_argument("--gb10-smem-limit", type=int, default=99 * 1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = Shape(B=args.B, S=args.S, H=args.H, G=args.G, N=args.N, P=args.P, R=args.R, chunk=args.chunk)
    if shape.S % shape.chunk:
        raise SystemExit("S must be divisible by chunk for this gate")
    print(json.dumps(report(shape, args.h100_smem_limit, args.gb10_smem_limit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
