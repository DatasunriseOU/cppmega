"""Wave6 Lane C copy-path receipt for the Mamba3 mono WGMMA plan.

This probe is intentionally CPU-only.  It takes the Wave5 copy ledger from the
Wave4 schedule receipt and adds concrete gates for the two non-scalar copy
paths:

* ``narrow_vector_128b_safe_attempt``: 16-byte vector-copy alignment, no-tail
  proof, swizzled-SMEM compatibility, and ptxas resource metadata schema.
* ``tma_cp_async_target``: minimal descriptor/tensor-map prototype checklist,
  mbarrier byte accounting, and the stricter 128 KiB SMEM pass budget.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path = [entry for entry in sys.path if entry != str(ROOT)]
sys.path.insert(0, str(ROOT))

wave4 = importlib.import_module("tools.probes.mamba3_wgmma_wave4_schedule")


BF16_BYTES = wave4.BF16_BYTES
VECTOR_BYTES = wave4.CP_ASYNC_BULK_ALIGNMENT_BYTES
TILE_ROWS = wave4.GMMA_M
TILE_COLS = wave4.GMMA_N
TILE_BYTES = TILE_ROWS * TILE_COLS * BF16_BYTES
ROW_BYTES = TILE_COLS * BF16_BYTES
SMEM_SWIZZLE_ATOM_BYTES = 128
NARROW_VECTOR_EXPECTED_DYNAMIC_SMEM_BYTES = 118_784
TMA_TARGET_DYNAMIC_SMEM_BYTES = 131_072
PASS_REGS_PER_THREAD = 192
KILL_REGS_PER_THREAD = 224

PTXAS_REQUIRED_FIELDS = (
    "registers_per_thread",
    "static_smem_bytes",
    "dynamic_smem_bytes",
    "spill_stores_bytes",
    "spill_loads_bytes",
)


@dataclass(frozen=True)
class TileCopyProof:
    name: str
    source_space: str
    destination_space: str
    tma_candidate: bool
    source_expr: str
    destination_expr: str
    tile_shape: str
    dtype: str
    row_bytes: int
    tile_bytes: int
    vector_bytes: int
    vectors_per_row: int
    vectors_per_tile: int
    row_tail_bytes: int
    tile_tail_bytes: int
    requires_runtime_base_alignment_assert: bool
    source_base_alignment_required_bytes: int
    destination_base_alignment_required_bytes: int
    offset_alignment_proof: str
    smem_layout: str
    smem_swizzle_atom_bytes: int
    swizzled_smem_compatible: bool
    swizzled_smem_proof: str
    vector_copy_static_pass: bool
    vector_copy_fail_conditions: list[str]


@dataclass(frozen=True)
class TmaDescriptorPrototype:
    tile_name: str
    source_tensor: str
    destination_smem: str
    rank: int
    element_type: str
    box_dim_elements: list[int]
    box_bytes: int
    tensor_map_required: bool
    swizzle_mode: str
    expected_mbarrier_bytes: int
    descriptor_static_pass: bool
    pass_conditions: list[str]
    fail_conditions: list[str]


def _mib(value: int) -> float:
    return round(value / float(1024**2), 6)


def _tile_copy_specs() -> list[dict[str, Any]]:
    """Return the 12 large tile movements/chunk from the Wave5 ledger."""

    return [
        {
            "name": "K",
            "source_space": "global",
            "destination_space": "smem",
            "tma_candidate": True,
            "source_expr": "K[b,h,chunk_idx*16 + t, n] as a 64x64 fused-row tile",
            "destination_expr": "sK physical row-major/swizzled 64x64 tile",
        },
        {
            "name": "Q",
            "source_space": "global",
            "destination_space": "smem",
            "tma_candidate": True,
            "source_expr": "Q[b,h,chunk_idx*16 + t, n] as a 64x64 fused-row tile",
            "destination_expr": "sQ physical row-major/swizzled 64x64 tile",
        },
        {
            "name": "K_T",
            "source_space": "global",
            "destination_space": "smem",
            "tma_candidate": True,
            "source_expr": "K tile copied with a CuTe layout that materializes or views K_T in SMEM",
            "destination_expr": "sK_T physical 128B-row swizzled transpose operand",
        },
        {
            "name": "Q_T",
            "source_space": "global",
            "destination_space": "smem",
            "tma_candidate": True,
            "source_expr": "Q tile copied with a CuTe layout that materializes or views Q_T in SMEM",
            "destination_expr": "sQ_T physical 128B-row swizzled transpose operand",
        },
        {
            "name": "state_panel_p0",
            "source_space": "global",
            "destination_space": "smem",
            "tma_candidate": True,
            "source_expr": "states[b,h,n,p=0:64] panel",
            "destination_expr": "sStatePanel p0 physical row-major/swizzled 64x64 tile",
        },
        {
            "name": "state_panel_p1",
            "source_space": "global",
            "destination_space": "smem",
            "tma_candidate": True,
            "source_expr": "states[b,h,n,p=64:128] panel",
            "destination_expr": "sStatePanel p1 physical row-major/swizzled 64x64 tile",
        },
        {
            "name": "dPhiO_panel_p0",
            "source_space": "global",
            "destination_space": "smem",
            "tma_candidate": True,
            "source_expr": "dPhiO[b,h,chunk_idx*16 + t,p=0:64] panel",
            "destination_expr": "sDPhPanel p0 physical row-major/swizzled 64x64 tile",
        },
        {
            "name": "dPhiO_panel_p1",
            "source_space": "global",
            "destination_space": "smem",
            "tma_candidate": True,
            "source_expr": "dPhiO[b,h,chunk_idx*16 + t,p=64:128] panel",
            "destination_expr": "sDPhPanel p1 physical row-major/swizzled 64x64 tile",
        },
        {
            "name": "PsiV_panel_p0",
            "source_space": "global",
            "destination_space": "smem",
            "tma_candidate": True,
            "source_expr": "PsiV[b,h,chunk_idx*16 + t,p=0:64] panel",
            "destination_expr": "sPsiPanel p0 physical row-major/swizzled 64x64 tile",
        },
        {
            "name": "PsiV_panel_p1",
            "source_space": "global",
            "destination_space": "smem",
            "tma_candidate": True,
            "source_expr": "PsiV[b,h,chunk_idx*16 + t,p=64:128] panel",
            "destination_expr": "sPsiPanel p1 physical row-major/swizzled 64x64 tile",
        },
        {
            "name": "dstates_panel_p0",
            "source_space": "cta_local",
            "destination_space": "smem",
            "tma_candidate": False,
            "source_expr": "CTA-local dstates[n,p=0:64] loop-carried panel",
            "destination_expr": "sDStatePanel p0 physical row-major/swizzled 64x64 tile",
        },
        {
            "name": "dstates_panel_p1",
            "source_space": "cta_local",
            "destination_space": "smem",
            "tma_candidate": False,
            "source_expr": "CTA-local dstates[n,p=64:128] loop-carried panel",
            "destination_expr": "sDStatePanel p1 physical row-major/swizzled 64x64 tile",
        },
    ]


def _tile_copy_proofs() -> list[TileCopyProof]:
    row_tail = ROW_BYTES % VECTOR_BYTES
    tile_tail = TILE_BYTES % VECTOR_BYTES
    vectors_per_row = ROW_BYTES // VECTOR_BYTES
    vectors_per_tile = TILE_BYTES // VECTOR_BYTES
    swizzle_compatible = (
        ROW_BYTES == SMEM_SWIZZLE_ATOM_BYTES
        and SMEM_SWIZZLE_ATOM_BYTES % VECTOR_BYTES == 0
        and row_tail == 0
        and tile_tail == 0
    )

    proofs: list[TileCopyProof] = []
    for spec in _tile_copy_specs():
        vector_pass = swizzle_compatible and row_tail == 0 and tile_tail == 0
        fail_conditions = [
            "source or destination base pointer is not 16-byte aligned at runtime",
            "the generated CuTe layout maps one 16-byte vector lane across non-contiguous SMEM fragments",
            "a masked tail branch is emitted for this 64x64 BF16 tile",
        ]
        if spec["name"] in {"K_T", "Q_T"}:
            fail_conditions.append(
                "the transpose operand is implemented as per-column 2-byte SMEM scatter instead of a vector-compatible physical layout/view"
            )

        proofs.append(
            TileCopyProof(
                name=spec["name"],
                source_space=spec["source_space"],
                destination_space=spec["destination_space"],
                tma_candidate=bool(spec["tma_candidate"]),
                source_expr=spec["source_expr"],
                destination_expr=spec["destination_expr"],
                tile_shape="64x64 bf16",
                dtype="bf16",
                row_bytes=ROW_BYTES,
                tile_bytes=TILE_BYTES,
                vector_bytes=VECTOR_BYTES,
                vectors_per_row=vectors_per_row,
                vectors_per_tile=vectors_per_tile,
                row_tail_bytes=row_tail,
                tile_tail_bytes=tile_tail,
                requires_runtime_base_alignment_assert=True,
                source_base_alignment_required_bytes=VECTOR_BYTES,
                destination_base_alignment_required_bytes=VECTOR_BYTES,
                offset_alignment_proof=(
                    "row_bytes=128 and panel offsets are multiples of 64 bf16 elements, "
                    "so every row/panel offset is a multiple of 16 bytes once the base pointer passes ptr % 16 == 0"
                ),
                smem_layout="128B-row GMMA operand swizzle or a CuTe layout with equivalent 16B-contiguous physical lanes",
                smem_swizzle_atom_bytes=SMEM_SWIZZLE_ATOM_BYTES,
                swizzled_smem_compatible=swizzle_compatible,
                swizzled_smem_proof=(
                    "64 bf16 values per row is exactly 128 bytes; a 16-byte vector lane divides the 128B swizzle atom "
                    "and never crosses a row or swizzle-atom boundary"
                ),
                vector_copy_static_pass=vector_pass,
                vector_copy_fail_conditions=fail_conditions,
            )
        )
    return proofs


def _documentation_sources() -> list[dict[str, str]]:
    return [
        {
            "name": "CUDA Programming Guide - asynchronous data copies",
            "url": "https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html",
            "receipt_note": (
                "TMA handles one-dimensional and multidimensional bulk async copies; "
                "global-to-shared TMA read completion is tracked by shared-memory barriers, "
                "and low-level cp.async.bulk requires explicit expected-byte accounting."
            ),
        },
        {
            "name": "libcu++ PTX cp.async.bulk wrappers",
            "url": "https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk.html",
            "receipt_note": (
                "The wrapper documents the 16-byte source/destination alignment and "
                "16-byte size-multiple requirements used by this receipt."
            ),
        },
        {
            "name": "CUTLASS CuTe TMA tensors",
            "url": "https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/cute/0z_tma_tensors.html",
            "receipt_note": (
                "CuTe TMA uses tensor-map/descriptor coordinates; descriptor construction "
                "is therefore a separate pass/fail gate from ordinary pointer copies."
            ),
        },
        {
            "name": "CuTe DSL cpasync submodule",
            "url": "https://docs.nvidia.com/cutlass/4.2.1/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html",
            "receipt_note": (
                "The CuTe DSL exposes tiled TMA atom construction and descriptor copy helpers "
                "for the Wave7 implementation prototype."
            ),
        },
    ]


def _tma_descriptor_prototypes(proofs: list[TileCopyProof]) -> list[TmaDescriptorPrototype]:
    prototypes: list[TmaDescriptorPrototype] = []
    for proof in proofs:
        if not proof.tma_candidate:
            continue
        prototypes.append(
            TmaDescriptorPrototype(
                tile_name=proof.name,
                source_tensor=proof.source_expr,
                destination_smem=proof.destination_expr,
                rank=2,
                element_type="bf16",
                box_dim_elements=[TILE_ROWS, TILE_COLS],
                box_bytes=TILE_BYTES,
                tensor_map_required=True,
                swizzle_mode="CU_TENSOR_MAP_SWIZZLE_128B or CuTe equivalent 128B GMMA SMEM swizzle",
                expected_mbarrier_bytes=TILE_BYTES,
                descriptor_static_pass=proof.swizzled_smem_compatible and proof.tile_tail_bytes == 0,
                pass_conditions=[
                    "descriptor/tensor-map construction succeeds for this source tensor and SMEM layout",
                    "global base pointer, shared base pointer, and transfer size satisfy 16-byte alignment",
                    "mbarrier expected-byte count is exactly 8192 for this tile before the wait",
                    "WGMMA does not consume the destination SMEM tile before the barrier wait/fence completes",
                ],
                fail_conditions=[
                    "descriptor construction fails or silently falls back to scalar copies",
                    "descriptor emits a different SMEM physical layout than the GMMA operand expects",
                    "expected-byte accounting is missing or not equal to the copied tile bytes",
                    "this tile is replaced by a tiny vector/scalar TMA descriptor path",
                ],
            )
        )
    return prototypes


def evaluate_ptxas_metadata(
    metadata: dict[str, Any] | None,
    *,
    variant: str,
) -> dict[str, Any]:
    """Evaluate compiled resource metadata against the Wave6 gates."""

    if variant == "narrow_vector_128b_safe_attempt":
        max_total_smem = NARROW_VECTOR_EXPECTED_DYNAMIC_SMEM_BYTES
        expected_dynamic_smem = NARROW_VECTOR_EXPECTED_DYNAMIC_SMEM_BYTES
        max_regs = PASS_REGS_PER_THREAD
    elif variant == "tma_cp_async_target":
        max_total_smem = TMA_TARGET_DYNAMIC_SMEM_BYTES
        expected_dynamic_smem = TMA_TARGET_DYNAMIC_SMEM_BYTES
        max_regs = PASS_REGS_PER_THREAD
    else:
        raise ValueError(f"unsupported copy-path variant: {variant}")

    schema = {
        "required_fields": list(PTXAS_REQUIRED_FIELDS),
        "registers_per_thread_max": max_regs,
        "kill_registers_per_thread": KILL_REGS_PER_THREAD,
        "expected_dynamic_smem_bytes": expected_dynamic_smem,
        "total_smem_bytes_max": max_total_smem,
        "spill_stores_bytes_max": 0,
        "spill_loads_bytes_max": 0,
    }

    if metadata is None:
        return {
            "variant": variant,
            "status": "missing_ptxas_metadata",
            "pass": False,
            "schema": schema,
            "metadata": None,
            "failures": [
                "ptxas metadata must report registers_per_thread, static_smem_bytes, dynamic_smem_bytes, spill stores, and spill loads"
            ],
        }

    failures: list[str] = []
    normalized: dict[str, int] = {}
    for field in PTXAS_REQUIRED_FIELDS:
        if field not in metadata:
            failures.append(f"missing required ptxas field: {field}")
            continue
        try:
            normalized[field] = int(metadata[field])
        except (TypeError, ValueError):
            failures.append(f"ptxas field is not an integer: {field}")

    if not failures:
        total_smem = normalized["static_smem_bytes"] + normalized["dynamic_smem_bytes"]
        if normalized["registers_per_thread"] > max_regs:
            failures.append(
                f"registers_per_thread {normalized['registers_per_thread']} exceeds pass budget {max_regs}"
            )
        if total_smem > max_total_smem:
            failures.append(f"total_smem_bytes {total_smem} exceeds pass budget {max_total_smem}")
        if normalized["spill_stores_bytes"] != 0:
            failures.append(f"spill_stores_bytes must be zero, got {normalized['spill_stores_bytes']}")
        if normalized["spill_loads_bytes"] != 0:
            failures.append(f"spill_loads_bytes must be zero, got {normalized['spill_loads_bytes']}")
    else:
        total_smem = None

    return {
        "variant": variant,
        "status": "pass" if not failures else "fail",
        "pass": not failures,
        "schema": schema,
        "metadata": normalized if normalized else metadata,
        "total_smem_bytes": total_smem,
        "failures": failures,
    }


def _narrow_vector_attempt(
    proofs: list[TileCopyProof],
    ptxas_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    global_tiles = [proof for proof in proofs if proof.source_space == "global"]
    local_tiles = [proof for proof in proofs if proof.source_space == "cta_local"]
    static_failures = [
        f"{proof.name}: {condition}"
        for proof in proofs
        if not proof.vector_copy_static_pass
        for condition in proof.vector_copy_fail_conditions
    ]
    ptxas_eval = evaluate_ptxas_metadata(
        ptxas_metadata,
        variant="narrow_vector_128b_safe_attempt",
    )
    can_enter_timing = not static_failures and ptxas_eval["pass"]

    return {
        "variant": "narrow_vector_128b_safe_attempt",
        "status": "static_pass_compile_metadata_required" if not ptxas_eval["pass"] else "ready_for_timing_gate",
        "copy_instruction_unit_bytes": VECTOR_BYTES,
        "vectors_per_tile": TILE_BYTES // VECTOR_BYTES,
        "copy_instructions_per_chunk": len(proofs) * (TILE_BYTES // VECTOR_BYTES),
        "global_tile_vectors_per_chunk": len(global_tiles) * (TILE_BYTES // VECTOR_BYTES),
        "local_tile_vectors_per_chunk": len(local_tiles) * (TILE_BYTES // VECTOR_BYTES),
        "large_copy_bytes_per_chunk": len(proofs) * TILE_BYTES,
        "global_copy_bytes_per_chunk": len(global_tiles) * TILE_BYTES,
        "local_stage_bytes_per_chunk": len(local_tiles) * TILE_BYTES,
        "alignment_and_tail_static_pass": not static_failures,
        "can_enter_timing_gate": can_enter_timing,
        "runtime_alignment_guards": [
            "assert every vectorized global base pointer % 16 == 0",
            "assert every vectorized SMEM base pointer % 16 == 0",
            "assert emitted row/panel strides preserve 16-byte offset alignment",
            "assert no masked vector-tail path appears in generated code",
        ],
        "static_failures": static_failures,
        "tile_proofs": [asdict(proof) for proof in proofs],
        "ptxas_resource_check": ptxas_eval,
        "pass_conditions": [
            "all 12 large tile movements have row_bytes=128, tile_bytes=8192, and 16-byte vector lanes with no row/tile tail",
            "K_T and Q_T use a vector-compatible physical SMEM layout/view, not per-column 2-byte scatter",
            "runtime base-alignment guards are present for every vectorized source and destination",
            "ptxas registers/thread <=192, total SMEM <=118784 B for this variant, and spill loads/stores are zero",
        ],
        "fail_conditions": [
            "any source or destination pointer lacks a 16-byte alignment proof",
            "any tile emits a masked tail or scalar cleanup path",
            "any swizzled-SMEM layout maps a 16-byte vector lane across non-contiguous fragments",
            "ptxas metadata is missing, registers/thread >192, total SMEM >118784 B, or any spill byte is reported",
        ],
    }


def _tma_target(
    proofs: list[TileCopyProof],
    ptxas_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    prototypes = _tma_descriptor_prototypes(proofs)
    non_tma_tiles = [proof.name for proof in proofs if not proof.tma_candidate]
    ptxas_eval = evaluate_ptxas_metadata(ptxas_metadata, variant="tma_cp_async_target")
    static_descriptor_failures = [
        proto.tile_name for proto in prototypes if not proto.descriptor_static_pass
    ]

    return {
        "variant": "tma_cp_async_target",
        "status": "descriptor_static_plan_compile_smoke_required",
        "descriptor_count": len(prototypes),
        "non_tma_large_tiles": non_tma_tiles,
        "descriptor_bytes_per_chunk": sum(proto.box_bytes for proto in prototypes),
        "expected_mbarrier_bytes_per_chunk": sum(proto.expected_mbarrier_bytes for proto in prototypes),
        "dynamic_smem_target_bytes": TMA_TARGET_DYNAMIC_SMEM_BYTES,
        "dynamic_smem_target_mib": _mib(TMA_TARGET_DYNAMIC_SMEM_BYTES),
        "descriptor_static_pass": not static_descriptor_failures,
        "descriptor_static_failures": static_descriptor_failures,
        "can_claim_green_or_yellow": not static_descriptor_failures and ptxas_eval["pass"],
        "descriptor_prototypes": [asdict(proto) for proto in prototypes],
        "ptxas_resource_check": ptxas_eval,
        "pass_conditions": [
            "exactly 10 global 64x64 BF16 tiles have descriptor/tensor-map prototypes",
            "exactly 2 CTA-local dstates stages remain non-TMA",
            "mbarrier expected-byte accounting equals 81920 B/chunk across the 10 descriptors",
            "WGMMA waits on the async barrier/fence before consuming every TMA-fed SMEM tile",
            "ptxas registers/thread <=192, total SMEM <=131072 B, and spill loads/stores are zero",
        ],
        "fail_conditions": [
            "any descriptor/tensor-map build fails for K, Q, K_T, Q_T, state, dPhiO, or PsiV tiles",
            "any TMA descriptor is attempted for CTA-local dstates or tiny scalar/vector slices",
            "expected mbarrier bytes differ from copied bytes or a wait/fence is missing",
            "dynamic/resource metadata exceeds 131072 B SMEM, exceeds 192 registers/thread, or reports any spill",
        ],
    }


def build_receipt(ptxas_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    wave4_receipt = wave4.build_receipt()
    wave5_scope = wave4_receipt["copy_strategy_variants"]["scope"]
    proofs = _tile_copy_proofs()
    tma_candidates = [proof for proof in proofs if proof.tma_candidate]
    local_tiles = [proof for proof in proofs if not proof.tma_candidate]

    receipt = {
        "receipt": "mamba3_mono_wgmma_copy_path_wave6_2026_04_30",
        "status": "lane_c_copy_path_static_receipt_and_resource_gate",
        "date": "2026-04-30",
        "branch": "worker/mamba3-mono-triton-model",
        "source_receipt": "docs/status/mamba3_mono_wgmma_plan_wave4_receipt_2026_04_30.json",
        "wave5_ledger": {
            "large_copy_tiles_per_chunk": wave5_scope["large_copy_tiles_per_chunk"],
            "tma_eligible_global_tiles_per_chunk": wave5_scope[
                "tma_eligible_global_tiles_per_chunk"
            ],
            "local_or_smem_stage_tiles_per_chunk": wave5_scope[
                "local_or_smem_stage_tiles_per_chunk"
            ],
            "large_copy_bytes_per_chunk": wave5_scope["large_copy_bytes_per_chunk"],
            "tma_eligible_global_copy_bytes_per_chunk": wave5_scope[
                "tma_eligible_global_copy_bytes_per_chunk"
            ],
            "local_or_smem_stage_bytes_per_chunk": wave5_scope[
                "local_or_smem_stage_bytes_per_chunk"
            ],
        },
        "static_tile_model": {
            "tile_shape": "64x64 bf16",
            "tile_rows": TILE_ROWS,
            "tile_cols": TILE_COLS,
            "row_bytes": ROW_BYTES,
            "tile_bytes": TILE_BYTES,
            "vector_bytes": VECTOR_BYTES,
            "vectors_per_row": ROW_BYTES // VECTOR_BYTES,
            "vectors_per_tile": TILE_BYTES // VECTOR_BYTES,
            "row_tail_bytes": ROW_BYTES % VECTOR_BYTES,
            "tile_tail_bytes": TILE_BYTES % VECTOR_BYTES,
            "smem_swizzle_atom_bytes": SMEM_SWIZZLE_ATOM_BYTES,
            "swizzle_compatibility_condition": (
                "row_bytes == 128 and vector_bytes divides the 128B SMEM swizzle atom"
            ),
            "tma_candidate_tile_count": len(tma_candidates),
            "non_tma_large_tile_count": len(local_tiles),
        },
        "narrow_vector_128b_safe_attempt": _narrow_vector_attempt(proofs, ptxas_metadata),
        "tma_cp_async_target": _tma_target(proofs, ptxas_metadata),
        "updated_gates": {
            "narrow_vector_128b_safe_attempt": [
                "static alignment/tail proof must pass for all 12 large tile movements",
                "K_T/Q_T transpose operands must prove vector-compatible swizzled SMEM layout/view",
                "runtime ptr%16 guards are required before enabling the vector path",
                "ptxas resource metadata fields are mandatory: registers/thread, static SMEM, dynamic SMEM, spill stores, spill loads",
                "timing gate is blocked unless ptxas <=192 regs/thread, total SMEM <=118784 B, and spills are zero",
            ],
            "tma_cp_async_target": [
                "descriptor/tensor-map prototypes cover only the 10 global 64x64 BF16 tiles",
                "CTA-local dstates stages and tiny scalar/vector slices are explicitly non-TMA",
                "mbarrier expected bytes must equal 81920 B/chunk and waits/fences must precede WGMMA use",
                "green/yellow timing claims require ptxas <=192 regs/thread, total SMEM <=131072 B, and zero spills",
            ],
        },
        "documentation_sources": _documentation_sources(),
        "validation": {
            "generated_by": "tools/probes/mamba3_wgmma_wave6_copy_path.py",
            "check_command": "python tools/probes/mamba3_wgmma_wave6_copy_path.py --check docs/status/mamba3_mono_wgmma_copy_path_wave6_receipt_2026_04_30.json",
            "optional_ptxas_metadata": "python tools/probes/mamba3_wgmma_wave6_copy_path.py --ptxas-metadata ptxas.json",
        },
    }
    _validate_receipt(receipt)
    return receipt


def _validate_receipt(receipt: dict[str, Any]) -> None:
    ledger = receipt["wave5_ledger"]
    assert ledger["large_copy_tiles_per_chunk"] == 12
    assert ledger["tma_eligible_global_tiles_per_chunk"] == 10
    assert ledger["local_or_smem_stage_tiles_per_chunk"] == 2
    assert ledger["large_copy_bytes_per_chunk"] == 98_304
    static_tile = receipt["static_tile_model"]
    assert static_tile["row_bytes"] == 128
    assert static_tile["tile_bytes"] == 8192
    assert static_tile["vectors_per_tile"] == 512
    assert static_tile["row_tail_bytes"] == 0
    assert static_tile["tile_tail_bytes"] == 0
    narrow = receipt["narrow_vector_128b_safe_attempt"]
    assert narrow["copy_instructions_per_chunk"] == 6144
    assert narrow["alignment_and_tail_static_pass"]
    assert len(narrow["tile_proofs"]) == 12
    assert all(proof["swizzled_smem_compatible"] for proof in narrow["tile_proofs"])
    tma = receipt["tma_cp_async_target"]
    assert tma["descriptor_count"] == 10
    assert tma["non_tma_large_tiles"] == ["dstates_panel_p0", "dstates_panel_p1"]
    assert tma["expected_mbarrier_bytes_per_chunk"] == 81_920
    assert tma["dynamic_smem_target_bytes"] == TMA_TARGET_DYNAMIC_SMEM_BYTES


def _canonical(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=pathlib.Path, default=None)
    parser.add_argument("--ptxas-metadata", type=pathlib.Path, default=None)
    args = parser.parse_args()

    ptxas_metadata = _read_json(args.ptxas_metadata) if args.ptxas_metadata else None
    receipt = build_receipt(ptxas_metadata=ptxas_metadata)
    rendered = _canonical(receipt)

    if args.check is None:
        print(rendered, end="")
        return

    actual = args.check.read_text()
    if actual != rendered:
        raise SystemExit(f"{args.check} does not match generated receipt")


if __name__ == "__main__":
    main()
