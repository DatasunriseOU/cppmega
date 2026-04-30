"""Emit and validate the Wave4 executable WGMMA schedule receipt.

This is a CPU-only schedule generator for the Mamba3 mono ``bwd_bwd`` rewrite.
It turns the Wave3 plan into concrete CTA ownership, GMMA-equivalent counts,
shared-memory accounting, register-pressure estimates, output slots, and
pass/kill criteria for Lane A/B implementation work.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "upstream_prs" / "examples" / "13_tilelang_floormod_dbz"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from rr_diag_wave2_mono_triton_pruned_model import Shape, _fma_model, _memory_model  # noqa: E402


BF16_BYTES = 2
FP32_BYTES = 4
GMMA_M = 64
GMMA_N = 64
GMMA_K = 16
FMA_PER_M64N64K16 = GMMA_M * GMMA_N * GMMA_K
H200_SMEM_BUDGET_BYTES = 128 * 1024
H200_SMEM_KILL_BYTES = 160 * 1024


@dataclass(frozen=True)
class GmmaProduct:
    component: str
    shape: str
    dense_m64n64k16_ops_per_chunk: int
    ideal_m64n64k16_equiv_ops_per_chunk: float
    tile_pruned_m64n64k16_equiv_ops_per_chunk: float
    useful_fma_per_chunk_ideal: int
    notes: str


@dataclass(frozen=True)
class SmemBuffer:
    name: str
    shape: str
    dtype: str
    bytes: int
    lifetime: str
    alias_group: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class OutputSlot:
    name: str
    shape: str
    dtype: str
    bytes: int
    owner: str
    cadence: str


def _shape() -> Shape:
    return Shape(B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16)


def _mib(value: int) -> float:
    return round(value / float(1024**2), 6)


def _kib(value: int) -> float:
    return round(value / 1024.0, 6)


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 12)


def _gops(value: float) -> float:
    return round(value, 6)


def _tile_pruned_entries(shape: Shape) -> int:
    tile_steps = 4
    tile_fused = tile_steps * shape.R
    ntiles = shape.chunk // tile_steps
    return (ntiles * (ntiles - 1) // 2) * tile_fused * tile_fused + ntiles * tile_fused * tile_fused


def _bytes_model(shape: Shape) -> dict[str, int]:
    scalar_bhs = shape.B * shape.H * shape.S * FP32_BYTES
    out = {
        "dv_output_bytes": shape.B * shape.S * shape.H * shape.P * BF16_BYTES,
        "dk_output_bytes": shape.B * shape.S * shape.R * shape.H * shape.N * BF16_BYTES,
        "dq_output_bytes": shape.B * shape.S * shape.R * shape.H * shape.N * BF16_BYTES,
        "dmimov_partial_bytes": shape.B * shape.H * shape.nchunks * shape.R * shape.P * FP32_BYTES,
        "dmimov_output_bytes": shape.B * shape.H * shape.R * shape.P * FP32_BYTES,
        "five_scalar_bhs_bytes": 5 * scalar_bhs,
        "dssda_output_bytes": shape.B * shape.H * shape.nchunks * shape.chunk * shape.chunk * FP32_BYTES,
        "dangles_output_bytes": shape.B * shape.S * shape.H * (shape.N // 4) * FP32_BYTES,
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
    out["per_chunk_scan_output_bytes"] = (
        shape.chunk * shape.P * BF16_BYTES
        + 2 * shape.chunk * shape.R * shape.N * BF16_BYTES
        + shape.chunk * shape.chunk * FP32_BYTES
        + 5 * shape.chunk * FP32_BYTES
        + shape.chunk * (shape.N // 4) * FP32_BYTES
    )
    out["per_cta_stream_scan_output_bytes"] = (
        out["per_chunk_scan_output_bytes"] * shape.nchunks + shape.R * shape.P * FP32_BYTES
    )
    return out


def _gmma_products(shape: Shape) -> list[GmmaProduct]:
    fcs = shape.fcs
    p_panels = math.ceil(shape.P / GMMA_N)
    causal_entries = shape.chunk * (shape.chunk - 1) // 2 * shape.R * shape.R
    pruned_entries = _tile_pruned_entries(shape)

    return [
        GmmaProduct(
            component="state_dpsi",
            shape="K[64,64] @ dstates[64,64] for each P panel",
            dense_m64n64k16_ops_per_chunk=p_panels * (shape.N // GMMA_K),
            ideal_m64n64k16_equiv_ops_per_chunk=8.0,
            tile_pruned_m64n64k16_equiv_ops_per_chunk=8.0,
            useful_fma_per_chunk_ideal=fcs * shape.N * shape.P,
            notes="Run once per P panel; dstates is loop-carried by the scan CTA.",
        ),
        GmmaProduct(
            component="lkq_once",
            shape="K[64,64] @ Q_T[64,64] -> LKQ[64,64]",
            dense_m64n64k16_ops_per_chunk=shape.N // GMMA_K,
            ideal_m64n64k16_equiv_ops_per_chunk=4.0,
            tile_pruned_m64n64k16_equiv_ops_per_chunk=4.0,
            useful_fma_per_chunk_ideal=fcs * shape.N * fcs,
            notes="Build exactly once per chunk; DSSDA consumes the full unmasked product.",
        ),
        GmmaProduct(
            component="lkq_apply_to_dphi",
            shape="causal(LKQ)[64,64] @ dPhiO[64,64] for each P panel",
            dense_m64n64k16_ops_per_chunk=p_panels * (fcs // GMMA_K),
            ideal_m64n64k16_equiv_ops_per_chunk=(causal_entries * shape.P) / FMA_PER_M64N64K16,
            tile_pruned_m64n64k16_equiv_ops_per_chunk=(pruned_entries * shape.P) / FMA_PER_M64N64K16,
            useful_fma_per_chunk_ideal=causal_entries * shape.P,
            notes="Ideal path requires 4x4 timestep subtile apply; full dense fallback is 8 ops/chunk.",
        ),
        GmmaProduct(
            component="dk_state",
            shape="PsiV[64,128] @ dstates_T[128,64] -> DK state[64,64]",
            dense_m64n64k16_ops_per_chunk=shape.P // GMMA_K,
            ideal_m64n64k16_equiv_ops_per_chunk=8.0,
            tile_pruned_m64n64k16_equiv_ops_per_chunk=8.0,
            useful_fma_per_chunk_ideal=fcs * shape.P * shape.N,
            notes="Consumes both P panels and feeds DDA_CS_REV plus DK.",
        ),
        GmmaProduct(
            component="dk_intra_once",
            shape="PsiV[64,128] @ dPhiO_T[128,64] -> dk_intra[64,64]",
            dense_m64n64k16_ops_per_chunk=shape.P // GMMA_K,
            ideal_m64n64k16_equiv_ops_per_chunk=8.0,
            tile_pruned_m64n64k16_equiv_ops_per_chunk=8.0,
            useful_fma_per_chunk_ideal=fcs * shape.P * fcs,
            notes="Build exactly once; feeds DGAMMA_DIAG, DSSDA, DK, and DQ.",
        ),
        GmmaProduct(
            component="dk_intra_apply_to_q",
            shape="causal(dk_intra)[64,64] @ Q[64,64] -> DK intra[64,64]",
            dense_m64n64k16_ops_per_chunk=fcs // GMMA_K,
            ideal_m64n64k16_equiv_ops_per_chunk=(causal_entries * shape.N) / FMA_PER_M64N64K16,
            tile_pruned_m64n64k16_equiv_ops_per_chunk=(pruned_entries * shape.N) / FMA_PER_M64N64K16,
            useful_fma_per_chunk_ideal=causal_entries * shape.N,
            notes="Same causal frontier as LKQ apply; diagonal must split to claim ideal.",
        ),
        GmmaProduct(
            component="dq_state",
            shape="dPhiO[64,128] @ states_T[128,64] -> DQ state[64,64]",
            dense_m64n64k16_ops_per_chunk=shape.P // GMMA_K,
            ideal_m64n64k16_equiv_ops_per_chunk=8.0,
            tile_pruned_m64n64k16_equiv_ops_per_chunk=8.0,
            useful_fma_per_chunk_ideal=fcs * shape.P * shape.N,
            notes="Consumes both P panels and feeds DDA_CS plus DQ.",
        ),
        GmmaProduct(
            component="dk_intra_t_apply_to_k",
            shape="causal(dk_intra)_T[64,64] @ K[64,64] -> DQ intra[64,64]",
            dense_m64n64k16_ops_per_chunk=fcs // GMMA_K,
            ideal_m64n64k16_equiv_ops_per_chunk=(causal_entries * shape.N) / FMA_PER_M64N64K16,
            tile_pruned_m64n64k16_equiv_ops_per_chunk=(pruned_entries * shape.N) / FMA_PER_M64N64K16,
            useful_fma_per_chunk_ideal=causal_entries * shape.N,
            notes="Input must be an SMEM transpose view/copy of dk_intra, not a second GMMA build.",
        ),
        GmmaProduct(
            component="scan_dstates_update",
            shape="Q_T[64,64] @ dPhiO[64,64] for each P panel",
            dense_m64n64k16_ops_per_chunk=p_panels * (fcs // GMMA_K),
            ideal_m64n64k16_equiv_ops_per_chunk=8.0,
            tile_pruned_m64n64k16_equiv_ops_per_chunk=8.0,
            useful_fma_per_chunk_ideal=shape.N * fcs * shape.P,
            notes="Scan-owner only; updates CTA-local dstates for the next reverse chunk.",
        ),
    ]


def _smem_buffers(shape: Shape) -> list[SmemBuffer]:
    tile_bf16 = GMMA_M * GMMA_N * BF16_BYTES
    return [
        SmemBuffer("sK", "64x64", "bf16", tile_bf16, "chunk", "operand_a"),
        SmemBuffer("sQ", "64x64", "bf16", tile_bf16, "chunk", "operand_b"),
        SmemBuffer("sQ_T", "64x64", "bf16", tile_bf16, "chunk", "transpose_operand"),
        SmemBuffer("sK_T", "64x64", "bf16", tile_bf16, "chunk", "transpose_operand"),
        SmemBuffer("sDStatePanel", "64x64", "bf16", tile_bf16, "P panel", "state_panel"),
        SmemBuffer("sStatePanel", "64x64", "bf16", tile_bf16, "P panel", "state_panel"),
        SmemBuffer("sDPhPanel", "64x64", "bf16", tile_bf16, "P panel stream", "p_operand"),
        SmemBuffer("sPsiPanel", "64x64", "bf16", tile_bf16, "P panel stream", "p_operand"),
        SmemBuffer("sLKQ", "64x64", "bf16", tile_bf16, "after LKQ build through DSSDA and dPsiV", None),
        SmemBuffer("sDKI", "64x64", "bf16", tile_bf16, "after dk_intra build through DK/DQ", None),
        SmemBuffer(
            "sDKI_T",
            "64x64",
            "bf16",
            tile_bf16,
            "DQ transpose feed",
            "dki_transpose",
            "May be a view; if copied, alias over sOut/dead operand storage.",
        ),
        SmemBuffer("sOut", "64x64", "bf16", tile_bf16, "C-store staging", "out_alias", "Alias over dead sK when legal."),
        SmemBuffer("qk_scalar_scratch", "<=1024 fp32/bf16 scalars", "mixed", 4096, "chunk", None),
    ]


def _output_slots(shape: Shape, bytes_model: dict[str, int]) -> list[OutputSlot]:
    return [
        OutputSlot("DV", "[B,S,H,P]", "bf16", bytes_model["dv_output_bytes"], "(B,H,chunk) within scan CTA", "once per chunk/P panel"),
        OutputSlot("DK", "[B,S*R,H,N]", "bf16", bytes_model["dk_output_bytes"], "(B,H,chunk) within scan CTA", "once per chunk"),
        OutputSlot("DQ", "[B,S*R,H,N]", "bf16", bytes_model["dq_output_bytes"], "(B,H,chunk) within scan CTA", "once per chunk"),
        OutputSlot("DMIMO_V", "[B,H,R,P]", "fp32", bytes_model["dmimov_output_bytes"], "(B,H) scan CTA", "once after reverse scan"),
        OutputSlot("DDA_CS", "[B,H,S]", "fp32", shape.B * shape.H * shape.S * FP32_BYTES, "(B,H,chunk)", "once per chunk"),
        OutputSlot("DDA_CS_REV", "[B,H,S]", "fp32", shape.B * shape.H * shape.S * FP32_BYTES, "(B,H,chunk)", "once per chunk"),
        OutputSlot("DFACTOR", "[B,H,S]", "fp32", shape.B * shape.H * shape.S * FP32_BYTES, "(B,H,chunk)", "once per chunk"),
        OutputSlot("DGAMMA_DIAG", "[B,H,S]", "fp32", shape.B * shape.H * shape.S * FP32_BYTES, "(B,H,chunk)", "once per chunk"),
        OutputSlot("DSSDA", "[B,H,nchunks,chunk,chunk]", "fp32", bytes_model["dssda_output_bytes"], "(B,H,chunk)", "once per chunk"),
        OutputSlot("DDA", "[B,H,S]", "fp32", shape.B * shape.H * shape.S * FP32_BYTES, "(B,H,chunk)", "once per chunk"),
        OutputSlot("DANGLES", "[B,S,H,N/4]", "fp32", bytes_model["dangles_output_bytes"], "(B,H,chunk)", "once per chunk"),
    ]


def _component_receipt(shape: Shape, fma: dict[str, Any], optional_dstates_update_fma: int) -> list[dict[str, Any]]:
    ideal = fma["monolithic_causal_apply_fma"]
    return [
        {
            "component": "state_dpsi",
            "fma": ideal["state_dpsi"],
            "receipt": "K @ dstates_panel runs once per P panel and feeds dPsiV.",
        },
        {
            "component": "lkq_once",
            "fma": ideal["lkq_once"],
            "receipt": "LKQ is built once per chunk and reused for dPsiV and DSSDA.",
        },
        {
            "component": "lkq_apply_to_dphi",
            "fma": ideal["lkq_apply_to_dphi_full_mask"],
            "receipt": "Causal apply only; implementation must report full-mask, 4-step, or ideal diagonal-split mode.",
        },
        {
            "component": "qk_dpsi_once_for_dv_and_dmimov",
            "fma": ideal["qk_dpsi_once_for_dv_and_dmimov"],
            "receipt": "qk_dot same-time contribution is computed once and feeds DV plus local DMIMO_V.",
        },
        {
            "component": "dv_and_dmimov_r_reductions",
            "fma": ideal["dv_and_dmimov_r_reductions"],
            "receipt": "DV is stored per chunk; DMIMO_V[R,P] remains CTA-local and writes once.",
        },
        {
            "component": "dk_state",
            "fma": ideal["dk_state"],
            "receipt": "PsiV @ dstates.T feeds DK and DDA_CS_REV.",
        },
        {
            "component": "dk_intra_once",
            "fma": ideal["dk_intra_once_for_dgamma_dssda_dk_dq"],
            "receipt": "dk_intra is built once and reused for DGAMMA_DIAG, DSSDA, DK, and DQ.",
        },
        {
            "component": "dk_intra_apply_to_q",
            "fma": ideal["dk_intra_apply_to_q_full_mask"],
            "receipt": "Causal dk_intra @ Q apply feeds DK.",
        },
        {
            "component": "dq_state",
            "fma": ideal["dq_state"],
            "receipt": "dPhiO @ states.T feeds DQ and DDA_CS.",
        },
        {
            "component": "dk_intra_t_apply_to_k",
            "fma": ideal["dk_intra_transpose_apply_to_k_full_mask"],
            "receipt": "DQ intra uses transpose SMEM view/copy; no second dk_intra GMMA build.",
        },
        {
            "component": "scalar_elementwise_reductions",
            "fma": ideal["scalar_elementwise_reductions"],
            "receipt": "Trap, rotary, DSSDA, DANGLES, DDA, and vector reductions are not TMA-tiny copies.",
        },
        {
            "component": "scan_dstates_update",
            "fma": optional_dstates_update_fma,
            "receipt": "Preferred scan owner updates dstates locally with Q.T @ dPhiO before the next reverse chunk.",
        },
    ]


def _schedule_steps() -> list[dict[str, Any]]:
    return [
        {
            "order": 0,
            "name": "map_cta",
            "action": "CTA owns one (b,h) stream; chunk loop is for chunk_idx=nchunks-1 downto 0.",
            "requires": ["dstates local to CTA", "DMIMO_V[R,P] local to CTA"],
        },
        {
            "order": 1,
            "name": "load_chunk_operands",
            "action": "Load/copy K, Q, K_T, Q_T, dPhiO/PsiV/state panels into SMEM views.",
            "requires": ["K-major legal SMEM operands", "no TMA for tiny vector slices"],
        },
        {
            "order": 2,
            "name": "build_lkq_once",
            "action": "Issue K @ Q_T GMMA, keep fp32 for scalar products, downcast to sLKQ for causal apply.",
            "requires": ["exactly one LKQ build per chunk"],
        },
        {
            "order": 3,
            "name": "dpsi_dv_dmimov_panels",
            "action": "For P panels 0..1, run K @ dstates_panel, causal LKQ @ dPhiO_panel, qk contribution, DV store, and local DMIMO_V accumulation.",
            "requires": ["P split as two n64 panels", "DMIMO_V final output only"],
        },
        {
            "order": 4,
            "name": "build_dk_state",
            "action": "Run PsiV @ dstates.T, accumulate DDA_CS_REV and DK state contribution.",
            "requires": ["dstates still local"],
        },
        {
            "order": 5,
            "name": "build_dk_intra_once",
            "action": "Run PsiV @ dPhiO.T once, keep fp32 for scalar products, downcast to sDKI for DK/DQ applies.",
            "requires": ["exactly one dk_intra build per chunk"],
        },
        {
            "order": 6,
            "name": "dk_dq_outputs",
            "action": "Use causal dk_intra @ Q for DK and transpose SMEM view/copy of dk_intra for causal dk_intra.T @ K for DQ.",
            "requires": ["no duplicate dk_intra.T GMMA build", "causal schedule is declared"],
        },
        {
            "order": 7,
            "name": "scan_state_update_and_scalar_stores",
            "action": "Store scalar outputs, then run Q.T @ dPhiO panel updates for CTA-local dstates.",
            "requires": ["reverse scan state is ready for next chunk"],
        },
        {
            "order": 8,
            "name": "final_stream_store",
            "action": "After chunk loop, store DMIMO_V[B,H,R,P] once.",
            "requires": ["no [B,H,nchunks,R,P] partial tensor"],
        },
    ]


def _register_pressure_estimate(shape: Shape) -> dict[str, Any]:
    warpgroup_threads = 128
    accumulator_regs_per_tile = (GMMA_M * GMMA_N) // warpgroup_threads
    full_dstates_regs = (shape.N * shape.P) // warpgroup_threads
    panel_dstates_regs = (shape.N * GMMA_N) // warpgroup_threads
    dmimov_regs = (shape.R * shape.P) // warpgroup_threads
    non_accumulator_regs = 40
    live_accumulator_tiles = 2
    epilogue_regs = 8
    panelized = (
        non_accumulator_regs
        + live_accumulator_tiles * accumulator_regs_per_tile
        + panel_dstates_regs
        + dmimov_regs
        + epilogue_regs
    )
    full_local = (
        non_accumulator_regs
        + live_accumulator_tiles * accumulator_regs_per_tile
        + full_dstates_regs
        + dmimov_regs
        + epilogue_regs
    )
    danger = full_local + accumulator_regs_per_tile
    return {
        "warpgroup_threads": warpgroup_threads,
        "accumulator_regs_per_64x64_fp32_tile_per_thread": accumulator_regs_per_tile,
        "dstates_regs_per_thread_if_panelized": panel_dstates_regs,
        "dstates_regs_per_thread_if_full_np_local": full_dstates_regs,
        "dmimov_regs_per_thread_if_distributed": dmimov_regs,
        "estimated_regs_per_thread_panelized": panelized,
        "estimated_regs_per_thread_full_dstates": full_local,
        "danger_regs_per_thread_with_third_live_accumulator": danger,
        "pass_budget_regs_per_thread": 192,
        "kill_budget_regs_per_thread": 224,
        "receipt": "Panelize dstates or spill to SMEM if ptxas exceeds 192 regs/thread; any local-memory spill is a kill.",
    }


def _smem_plan(shape: Shape) -> dict[str, Any]:
    buffers = _smem_buffers(shape)
    logical_bytes = sum(buffer.bytes for buffer in buffers)
    aliased_bytes = logical_bytes - 2 * GMMA_M * GMMA_N * BF16_BYTES
    guarded_bytes = logical_bytes + 16 * 1024
    return {
        "buffers": [asdict(buffer) for buffer in buffers],
        "logical_unique_buffer_bytes": logical_bytes,
        "logical_unique_buffer_kib": _kib(logical_bytes),
        "aliased_peak_bytes": aliased_bytes,
        "aliased_peak_kib": _kib(aliased_bytes),
        "peak_with_alignment_guard_bytes": guarded_bytes,
        "peak_with_alignment_guard_kib": _kib(guarded_bytes),
        "pass_budget_bytes": H200_SMEM_BUDGET_BYTES,
        "kill_budget_bytes": H200_SMEM_KILL_BYTES,
        "receipt": "Target <=128 KiB dynamic SMEM. sOut aliases dead sK; sDKI_T is a view or aliases staging storage if copied.",
    }


def _gmma_counts(shape: Shape, products: list[GmmaProduct]) -> dict[str, Any]:
    chunks = shape.B * shape.H * shape.nchunks
    non_scan_products = [product for product in products if product.component != "scan_dstates_update"]
    scan_update = next(product for product in products if product.component == "scan_dstates_update")
    full_per_chunk = sum(product.dense_m64n64k16_ops_per_chunk for product in non_scan_products)
    ideal_per_chunk = sum(product.ideal_m64n64k16_equiv_ops_per_chunk for product in non_scan_products)
    tile_per_chunk = sum(product.tile_pruned_m64n64k16_equiv_ops_per_chunk for product in non_scan_products)
    scan_ops = scan_update.dense_m64n64k16_ops_per_chunk
    return {
        "atom": "SM90 BF16 GMMA m64n64k16 -> fp32 accumulator",
        "fma_per_m64n64k16_op": FMA_PER_M64N64K16,
        "products": [
            {
                **asdict(product),
                "dense_m64n64k16_ops_grid": product.dense_m64n64k16_ops_per_chunk * chunks,
                "ideal_m64n64k16_equiv_ops_grid": _gops(product.ideal_m64n64k16_equiv_ops_per_chunk * chunks),
                "tile_pruned_m64n64k16_equiv_ops_grid": _gops(product.tile_pruned_m64n64k16_equiv_ops_per_chunk * chunks),
            }
            for product in products
        ],
        "totals": {
            "full_mask_dense_m64n64k16_ops_per_chunk_excluding_scan_update": full_per_chunk,
            "full_mask_dense_m64n64k16_ops_per_cta_stream_excluding_scan_update": full_per_chunk * shape.nchunks,
            "full_mask_dense_m64n64k16_ops_grid_excluding_scan_update": full_per_chunk * chunks,
            "ideal_triangular_m64n64k16_equiv_ops_per_chunk_excluding_scan_update": _gops(ideal_per_chunk),
            "ideal_triangular_m64n64k16_equiv_ops_per_cta_stream_excluding_scan_update": _gops(
                ideal_per_chunk * shape.nchunks
            ),
            "ideal_triangular_m64n64k16_equiv_ops_grid_excluding_scan_update": _gops(ideal_per_chunk * chunks),
            "wave2_4step_m64n64k16_equiv_ops_per_chunk_excluding_scan_update": _gops(tile_per_chunk),
            "wave2_4step_m64n64k16_equiv_ops_grid_excluding_scan_update": _gops(tile_per_chunk * chunks),
            "scan_dstates_update_dense_m64n64k16_ops_per_chunk": scan_ops,
            "scan_owner_full_mask_dense_m64n64k16_ops_per_chunk_with_update": full_per_chunk + scan_ops,
            "scan_owner_ideal_m64n64k16_equiv_ops_per_chunk_with_update": _gops(ideal_per_chunk + scan_ops),
            "scan_owner_wave2_4step_m64n64k16_equiv_ops_per_chunk_with_update": _gops(tile_per_chunk + scan_ops),
            "scan_owner_full_mask_dense_m64n64k16_ops_grid_with_update": (full_per_chunk + scan_ops) * chunks,
            "scan_owner_ideal_m64n64k16_equiv_ops_grid_with_update": _gops((ideal_per_chunk + scan_ops) * chunks),
        },
    }


def _criteria() -> dict[str, Any]:
    return {
        "pass": [
            "Correctness smoke covers every output slot: DV, DK, DQ, DMIMO_V, five BHS scalars, DSSDA, DANGLES.",
            "Preferred owner is one CTA per (B,H) stream with reverse chunk loop and CTA-local dstates plus DMIMO_V[R,P].",
            "LKQ and dk_intra are each built once per chunk; dk_intra.T is fed from SMEM view/copy.",
            "No global dPsiV, LKQ, dk_intra, DK, DQ intermediate, or DMIMO_V partial tensor is materialized.",
            "Dynamic SMEM <=128 KiB including alignment guard; ptxas registers/thread <=192 with no local-memory spills.",
            "Reported timing is green at <=3.35 ms or at least yellow at <=3.70674 ms against TileLang.",
        ],
        "kill": [
            "Any duplicate LKQ, dk_intra, or dk_intra.T GMMA build.",
            "Any [B,H,nchunks,R,P] DMIMO_V partial output in the preferred scan-owner path.",
            "Any silent owner mix where dstates is treated both as a loop-carried local and as precomputed chunk input.",
            "Dynamic SMEM >160 KiB, ptxas registers/thread >224, or any local-memory spill in the hot CTA.",
            "Full-kernel timing >3.70674 ms on productionish H200 after correctness.",
            "Claiming ideal 96.38B/113.56B FMA without diagonal 4x4 causal split or equivalent no-work-lower-triangle proof.",
        ],
        "timing_ms": {
            "green_full_kernel": 3.35,
            "yellow_full_kernel": 3.70674,
            "red_full_kernel_above": 3.70674,
            "scan_owner_main_body": 3.30,
            "chunk_owner_main_body": 3.20,
            "chunk_owner_dmimov_reducer": 0.05,
        },
    }


def build_receipt() -> dict[str, Any]:
    shape = _shape()
    chunks = shape.B * shape.H * shape.nchunks
    fma = _fma_model(shape, block_p=shape.P)
    memory = _memory_model(shape)
    bytes_model = _bytes_model(shape)
    products = _gmma_products(shape)
    optional_dstates_update_fma = shape.B * shape.H * shape.nchunks * shape.N * shape.fcs * shape.P
    scan_owner_ideal_with_dstates = (
        fma["monolithic_causal_apply_total_fma"] + optional_dstates_update_fma
    )
    scan_owner_tile_with_dstates = (
        fma["monolithic_tile_pruned_total_fma"] + optional_dstates_update_fma
    )
    scan_owner_full_with_dstates = (
        fma["monolithic_full_mask_total_fma"] + optional_dstates_update_fma
    )

    receipt = {
        "receipt": "mamba3_mono_wgmma_plan_wave4_2026_04_30",
        "status": "executable_schedule_receipt_for_lane_a_b",
        "date": "2026-04-30",
        "branch": "worker/mamba3-mono-triton-model",
        "source_receipt": "docs/status/mamba3_mono_wgmma_plan_wave3_receipt_2026_04_30.json",
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
            "chunk_bodies": chunks,
            "p_panels": math.ceil(shape.P / GMMA_N),
        },
        "ownership": {
            "preferred": {
                "name": "scan_owner_bh_stream",
                "cta_count": shape.B * shape.H,
                "chunk_iterations_per_cta": shape.nchunks,
                "chunk_order": "reverse: 255 downto 0",
                "dstates": "CTA-local loop-carried N x P state",
                "dmimov": "CTA-local R x P fp32 accumulator, final store only",
            },
            "fallback": {
                "name": "chunk_owner_bh_chunk",
                "cta_count": chunks,
                "allowed_only_if": "upstream provides correct per-chunk dstates handoff",
                "extra_global_bytes": bytes_model["dmimov_reducer_extra_rw_bytes"],
                "extra_global_mib": _mib(bytes_model["dmimov_reducer_extra_rw_bytes"]),
            },
        },
        "schedule_steps": _schedule_steps(),
        "gmma_counts": _gmma_counts(shape, products),
        "smem_plan": _smem_plan(shape),
        "register_pressure_estimate": _register_pressure_estimate(shape),
        "output_slots": [asdict(slot) for slot in _output_slots(shape, bytes_model)],
        "output_bytes": {
            "per_chunk_scan_output_bytes_excluding_final_dmimov": bytes_model["per_chunk_scan_output_bytes"],
            "per_cta_stream_scan_output_bytes_including_final_dmimov": bytes_model[
                "per_cta_stream_scan_output_bytes"
            ],
            "scan_owner_required_output_write_bytes": bytes_model["scan_owner_required_output_write_bytes"],
            "scan_owner_required_output_write_mib": _mib(bytes_model["scan_owner_required_output_write_bytes"]),
            "chunk_owner_required_output_write_bytes": bytes_model["chunk_owner_required_output_write_bytes"],
            "chunk_owner_required_output_write_mib": _mib(bytes_model["chunk_owner_required_output_write_bytes"]),
            "dmimov_partial_bytes": bytes_model["dmimov_partial_bytes"],
            "dmimov_partial_mib": _mib(bytes_model["dmimov_partial_bytes"]),
            "dmimov_reducer_extra_rw_bytes": bytes_model["dmimov_reducer_extra_rw_bytes"],
            "dmimov_reducer_extra_rw_mib": _mib(bytes_model["dmimov_reducer_extra_rw_bytes"]),
        },
        "modeled_fma": {
            "separate_recompute": fma["separate_recompute_total_fma"],
            "monolithic_full_mask": fma["monolithic_full_mask_total_fma"],
            "wave2_4step_tile_pruned": fma["monolithic_tile_pruned_total_fma"],
            "ideal_triangular_apply": fma["monolithic_causal_apply_total_fma"],
            "optional_scan_owner_dstates_update": optional_dstates_update_fma,
            "scan_owner_full_mask_plus_dstates_update": scan_owner_full_with_dstates,
            "scan_owner_wave2_4step_plus_dstates_update": scan_owner_tile_with_dstates,
            "scan_owner_ideal_plus_dstates_update": scan_owner_ideal_with_dstates,
            "components_ideal_triangular": fma["monolithic_causal_apply_fma"],
            "global_temps_avoided_by_cuda_owner_mib": memory["global_temps_avoided_by_cuda_owner_mib"],
        },
        "bytes_per_fma": {
            "scan_owner_ideal_plus_dstates_output_write_bytes_per_fma": _ratio(
                bytes_model["scan_owner_required_output_write_bytes"], scan_owner_ideal_with_dstates
            ),
            "scan_owner_wave2_4step_plus_dstates_output_write_bytes_per_fma": _ratio(
                bytes_model["scan_owner_required_output_write_bytes"], scan_owner_tile_with_dstates
            ),
            "scan_owner_full_mask_plus_dstates_output_write_bytes_per_fma": _ratio(
                bytes_model["scan_owner_required_output_write_bytes"], scan_owner_full_with_dstates
            ),
            "chunk_owner_ideal_output_write_bytes_per_fma": _ratio(
                bytes_model["chunk_owner_required_output_write_bytes"],
                fma["monolithic_causal_apply_total_fma"],
            ),
        },
        "component_receipt": _component_receipt(shape, fma, optional_dstates_update_fma),
        "criteria": _criteria(),
        "validation": {
            "generated_by": "tools/probes/mamba3_wgmma_wave4_schedule.py",
            "check_command": "python tools/probes/mamba3_wgmma_wave4_schedule.py --check docs/status/mamba3_mono_wgmma_plan_wave4_receipt_2026_04_30.json",
        },
    }
    _validate_receipt(receipt)
    return receipt


def _validate_receipt(receipt: dict[str, Any]) -> None:
    shape = receipt["shape"]
    assert shape["B"] * shape["H"] == receipt["ownership"]["preferred"]["cta_count"]
    assert (
        shape["B"] * shape["H"] * shape["nchunks"]
        == shape["chunk_bodies"]
        == receipt["ownership"]["fallback"]["cta_count"]
    )
    gmma_totals = receipt["gmma_counts"]["totals"]
    assert gmma_totals["full_mask_dense_m64n64k16_ops_per_chunk_excluding_scan_update"] == 52
    assert (
        gmma_totals["scan_owner_full_mask_dense_m64n64k16_ops_per_chunk_with_update"]
        == 60
    )
    assert gmma_totals["ideal_triangular_m64n64k16_equiv_ops_per_chunk_excluding_scan_update"] == 43.5
    assert receipt["smem_plan"]["peak_with_alignment_guard_bytes"] <= H200_SMEM_BUDGET_BYTES
    assert receipt["register_pressure_estimate"]["estimated_regs_per_thread_full_dstates"] <= 192
    slot_sum = sum(slot["bytes"] for slot in receipt["output_slots"])
    assert slot_sum == receipt["output_bytes"]["scan_owner_required_output_write_bytes"]
    assert receipt["bytes_per_fma"]["scan_owner_ideal_plus_dstates_output_write_bytes_per_fma"] < 0.0067


def _canonical(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _render_skeleton(receipt: dict[str, Any]) -> str:
    shape = receipt["shape"]
    totals = receipt["gmma_counts"]["totals"]
    return f"""// Wave4 Mamba3 mono WGMMA executable skeleton
// Shape: B={shape["B"]}, S={shape["S"]}, H={shape["H"]}, N={shape["N"]}, P={shape["P"]}, R={shape["R"]}, chunk={shape["chunk"]}
// Preferred grid: <<<B*H={shape["B"] * shape["H"]} CTAs>>>; each CTA loops {shape["nchunks"]} chunks in reverse.
// Per chunk dense full-mask GMMA ops: {totals["full_mask_dense_m64n64k16_ops_per_chunk_excluding_scan_update"]}; scan update adds {totals["scan_dstates_update_dense_m64n64k16_ops_per_chunk"]}.
// Ideal triangular GMMA-equivalent ops with scan update: {totals["scan_owner_ideal_m64n64k16_equiv_ops_per_chunk_with_update"]}.

cta_bh = blockIdx.x;
b = cta_bh / H;
h = cta_bh % H;
init_local_dstates_NxP();
init_local_dmimov_RxP();

for (int chunk_idx = nchunks - 1; chunk_idx >= 0; --chunk_idx) {{
  load_smem_K_Q_and_transpose_views(b, h, chunk_idx);
  build_LKQ_once_with_gmma_m64n64k16();

  for (int p_panel = 0; p_panel < 2; ++p_panel) {{
    state_dpsi = gmma(K, dstates_panel[p_panel]);
    dpsi = state_dpsi + causal_apply_LKQ_to_dPhiO(p_panel);
    dpsi += qk_same_time_contribution();
    store_DV_panel();
    accumulate_local_DMIMO_V();
  }}

  dk_state = gmma(PsiV, transpose(dstates));
  build_dk_intra_once_with_gmma_m64n64k16();
  emit_DSSDA_and_DGAMMA_DIAG_from_LKQ_and_dk_intra();
  dk = dk_state + causal_apply_dk_intra_to_Q();
  store_DK();

  dq_state = gmma(dPhiO, transpose(states));
  dq = dq_state + causal_apply_transposed_smem_dk_intra_to_K();
  store_DQ_and_scalar_outputs();

  dstates += gmma(Q_T, dPhiO_panels);  // CTA-local reverse-scan update
}}

store_final_DMIMO_V();
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=pathlib.Path, default=None)
    parser.add_argument("--write", type=pathlib.Path, default=None)
    parser.add_argument("--format", choices=("json", "skeleton"), default="json")
    args = parser.parse_args()

    receipt = build_receipt()
    rendered = _canonical(receipt)
    if args.check is not None:
        actual = args.check.read_text()
        if actual != rendered:
            print(f"{args.check} does not match generated receipt", file=sys.stderr)
            sys.exit(1)
        return

    if args.write is not None:
        args.write.write_text(rendered)
        return

    if args.format == "skeleton":
        print(_render_skeleton(receipt), end="")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
