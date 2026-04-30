"""Lane D resource gate for a full monolithic Mamba3 chunk owner.

This is not a CUDA kernel.  It captures the resource math that decides whether
the next monolithic CUDA chunk redesign is worth implementing.  The model uses
the NAM56R productionish shape and the measured Wave 8/Wave 10 CUDA evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


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


BF16 = 2
FP32 = 4
PANEL = 64

COMPARISON_MS = {
    "tilelang_full_bwd_bwd_h200": 3.70674,
    "wave8_tile_stream_subset_h200": 11.1550079981486,
    "wave10_two_launch_covered_subset_h200": 2.09673,
    "wave10_one_launch_covered_subset_h200": 2.31212,
    "wave7_row_stream_subset_h200": 179.76535034179688,
}


def gib(nbytes: int | float) -> float:
    return float(nbytes) / 1024.0 / 1024.0 / 1024.0


def smem_p64_panel_subset(shape: Shape) -> dict[str, int]:
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


def smem_p128_single_owner_all_consumers(shape: Shape, *, keep_prerot: bool) -> dict[str, int]:
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
        items.update(
            {
                "q_pre_rot_shared_bf16": fcs * shape.N * BF16,
                "k_pre_rot_shared_bf16": fcs * shape.N * BF16,
                "k_pre_trap_shared_bf16": fcs * shape.N * BF16,
            }
        )
    return items


def partial_panel_roundtrip_bytes(shape: Shape) -> dict[str, int]:
    panels = math.ceil(shape.P / PANEL)
    chunk_ctas = shape.B * shape.H * shape.nchunks
    fcs = shape.fused_chunk
    dk_dq = chunk_ctas * panels * fcs * shape.N * FP32 * 2
    scalar = chunk_ctas * panels * shape.chunk * FP32 * 6
    dmimo = chunk_ctas * panels * shape.R * shape.P * FP32
    return {
        "dk_dq_partial_fp32_bytes": dk_dq,
        "six_scalar_partial_fp32_bytes": scalar,
        "dmimo_partial_fp32_bytes": dmimo,
        "total_bytes": dk_dq + scalar + dmimo,
    }


def layout_report(shape: Shape, h100_smem_limit: int, gb10_smem_limit: int) -> dict[str, Any]:
    p64_smem = smem_p64_panel_subset(shape)
    p128_min_smem = smem_p128_single_owner_all_consumers(shape, keep_prerot=False)
    p128_reuse_smem = smem_p128_single_owner_all_consumers(shape, keep_prerot=True)
    p64_roundtrip = partial_panel_roundtrip_bytes(shape)
    return {
        "shape": asdict(shape) | {"nchunks": shape.nchunks, "fused_chunk": shape.fused_chunk},
        "comparison_ms": COMPARISON_MS,
        "p64_panel_owner": {
            "smem_bytes": p64_smem,
            "smem_total_bytes": sum(p64_smem.values()),
            "p_panels_per_chunk": math.ceil(shape.P / PANEL),
            "owner": "B,H,chunk,P64-panel",
            "hard_blocker": "DK/DQ and scalar consumers need a P reduction; avoiding a global roundtrip would require owning all P in one CTA.",
            "required_global_roundtrip_if_kept": p64_roundtrip
            | {
                "total_gib": gib(p64_roundtrip["total_bytes"]),
                "dk_dq_partial_gib": gib(p64_roundtrip["dk_dq_partial_fp32_bytes"]),
            },
        },
        "p128_single_chunk_owner": {
            "owner": "B,H,chunk,all-P",
            "minimal_recompute_prerot_smem_bytes": p128_min_smem,
            "minimal_recompute_prerot_total_bytes": sum(p128_min_smem.values()),
            "reuse_prerot_smem_bytes": p128_reuse_smem,
            "reuse_prerot_total_bytes": sum(p128_reuse_smem.values()),
            "minimal_total_kib": sum(p128_min_smem.values()) / 1024.0,
            "reuse_total_kib": sum(p128_reuse_smem.values()) / 1024.0,
            "fits_h100_dynamic_smem": sum(p128_reuse_smem.values()) <= h100_smem_limit,
            "fits_local_gb10_dynamic_smem": sum(p128_reuse_smem.values()) <= gb10_smem_limit,
            "active_blocks_per_sm_at_h100_smem_limit": h100_smem_limit // sum(p128_reuse_smem.values()),
            "hard_blocker": "The all-P owner can avoid the P roundtrip, but it is one block/SM and adds omitted DK/DQ/DGAMMA/DDA work on top of a measured subset that is already 3.0x slower than TileLang.",
        },
        "decision": {
            "verdict": "hard_no_go",
            "reason": (
                "No monolithic CUDA owner simultaneously preserves LKQ/state reuse, avoids global "
                "roundtrips for DK/DQ/scalars, and has a credible path below the 3.70674 ms "
                "TileLang full bwd_bwd baseline."
            ),
        },
    }


def main() -> None:
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
    args = parser.parse_args()
    shape = Shape(B=args.B, S=args.S, H=args.H, G=args.G, N=args.N, P=args.P, R=args.R, chunk=args.chunk)
    if shape.S % shape.chunk:
        raise SystemExit("S must be divisible by chunk for this gate")
    print(json.dumps(layout_report(shape, args.h100_smem_limit, args.gb10_smem_limit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
