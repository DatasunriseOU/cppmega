from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.probes import mamba3_wgmma_wave4_schedule as wave4


def test_wave4_schedule_core_counts() -> None:
    receipt = wave4.build_receipt()

    assert receipt["ownership"]["preferred"]["cta_count"] == 128
    assert receipt["ownership"]["preferred"]["chunk_iterations_per_cta"] == 256
    assert receipt["ownership"]["fallback"]["cta_count"] == 32768

    totals = receipt["gmma_counts"]["totals"]
    assert totals["full_mask_dense_m64n64k16_ops_per_chunk_excluding_scan_update"] == 52
    assert totals["scan_owner_full_mask_dense_m64n64k16_ops_per_chunk_with_update"] == 60
    assert totals["ideal_triangular_m64n64k16_equiv_ops_per_chunk_excluding_scan_update"] == 43.5
    assert totals["scan_owner_ideal_m64n64k16_equiv_ops_per_chunk_with_update"] == 51.5

    assert receipt["smem_plan"]["peak_with_alignment_guard_bytes"] == 118784
    assert receipt["smem_plan"]["peak_with_alignment_guard_bytes"] <= receipt["smem_plan"]["pass_budget_bytes"]
    assert receipt["register_pressure_estimate"]["estimated_regs_per_thread_full_dstates"] == 180
    assert receipt["register_pressure_estimate"]["estimated_regs_per_thread_full_dstates"] <= 192


def test_wave4_schedule_receipt_bytes_and_gates() -> None:
    receipt = wave4.build_receipt()

    assert receipt["output_bytes"]["scan_owner_required_output_write_bytes"] == 748945408
    assert receipt["output_bytes"]["chunk_owner_required_output_write_bytes"] == 816054272
    assert receipt["output_bytes"]["dmimov_reducer_extra_rw_bytes"] == 134479872
    assert receipt["bytes_per_fma"]["scan_owner_ideal_plus_dstates_output_write_bytes_per_fma"] == 0.006595349782
    assert receipt["bytes_per_fma"]["chunk_owner_ideal_output_write_bytes_per_fma"] == 0.008467338324

    kill = "\n".join(receipt["criteria"]["kill"])
    assert "duplicate LKQ" in kill
    assert "DMIMO_V partial" in kill
    assert "scalar BF16 copy baseline" in kill
    assert "TMA descriptor failure" in kill
    assert "Full-kernel timing >3.70674 ms" in kill


def test_wave4_copy_strategy_variants_and_resources() -> None:
    receipt = wave4.build_receipt()
    copy_receipt = receipt["copy_strategy_variants"]

    scope = copy_receipt["scope"]
    assert scope["large_copy_tiles_per_chunk"] == 12
    assert scope["tma_eligible_global_tiles_per_chunk"] == 10
    assert scope["local_or_smem_stage_tiles_per_chunk"] == 2
    assert scope["large_copy_bytes_per_chunk"] == 98304
    assert scope["tma_eligible_global_copy_bytes_per_chunk"] == 81920
    assert scope["local_or_smem_stage_bytes_per_chunk"] == 16384
    assert scope["large_copy_mib_grid"] == 3072.0

    variants = {variant["name"]: variant for variant in copy_receipt["variants"]}
    assert set(variants) == {
        "scalar_bf16_correct_baseline",
        "narrow_vector_128b_safe_attempt",
        "tma_cp_async_target",
    }

    assert variants["scalar_bf16_correct_baseline"]["copy_instructions_per_chunk"] == 49152
    assert variants["scalar_bf16_correct_baseline"]["production_gate"].startswith("fails performance gate")
    assert variants["narrow_vector_128b_safe_attempt"]["copy_instructions_per_chunk"] == 6144
    assert variants["narrow_vector_128b_safe_attempt"]["estimated_regs_per_thread_full_dstates"] == 192
    assert variants["tma_cp_async_target"]["copy_instructions_per_chunk"] == 12
    assert variants["tma_cp_async_target"]["tma_eligible_global_tile_ops_per_chunk"] == 10
    assert variants["tma_cp_async_target"]["local_or_smem_stage_tile_ops_per_chunk"] == 2
    assert variants["tma_cp_async_target"]["estimated_dynamic_smem_bytes"] == 131072

    budget = copy_receipt["resource_budget"]
    assert budget["tma_one_stage_dynamic_smem_bytes"] == budget["pass_dynamic_smem_bytes"]
    assert budget["tma_two_stage_dynamic_smem_bytes"] > budget["pass_dynamic_smem_bytes"]
    assert budget["tma_two_stage_dynamic_smem_bytes"] < budget["kill_dynamic_smem_bytes"]

    sources = {source["name"] for source in copy_receipt["documentation_sources"]}
    assert "CUDA Programming Guide - asynchronous data copies" in sources
    assert "CUTLASS CuTe TMA tensors" in sources


def test_wave4_schedule_check_mode(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(wave4._canonical(wave4.build_receipt()))

    result = subprocess.run(
        [
            sys.executable,
            "tools/probes/mamba3_wgmma_wave4_schedule.py",
            "--check",
            str(receipt_path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
