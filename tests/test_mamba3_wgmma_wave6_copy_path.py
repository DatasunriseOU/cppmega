from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.probes import mamba3_wgmma_wave6_copy_path as wave6


def test_wave6_narrow_vector_static_alignment_tail_proof() -> None:
    receipt = wave6.build_receipt()
    static_tile = receipt["static_tile_model"]
    narrow = receipt["narrow_vector_128b_safe_attempt"]

    assert static_tile["row_bytes"] == 128
    assert static_tile["tile_bytes"] == 8192
    assert static_tile["vector_bytes"] == 16
    assert static_tile["vectors_per_row"] == 8
    assert static_tile["vectors_per_tile"] == 512
    assert static_tile["row_tail_bytes"] == 0
    assert static_tile["tile_tail_bytes"] == 0

    assert narrow["copy_instructions_per_chunk"] == 6144
    assert narrow["global_tile_vectors_per_chunk"] == 5120
    assert narrow["local_tile_vectors_per_chunk"] == 1024
    assert narrow["alignment_and_tail_static_pass"] is True
    assert narrow["can_enter_timing_gate"] is False
    assert narrow["ptxas_resource_check"]["status"] == "missing_ptxas_metadata"

    tile_proofs = {proof["name"]: proof for proof in narrow["tile_proofs"]}
    assert set(tile_proofs) == {
        "K",
        "Q",
        "K_T",
        "Q_T",
        "state_panel_p0",
        "state_panel_p1",
        "dPhiO_panel_p0",
        "dPhiO_panel_p1",
        "PsiV_panel_p0",
        "PsiV_panel_p1",
        "dstates_panel_p0",
        "dstates_panel_p1",
    }
    assert all(proof["source_base_alignment_required_bytes"] == 16 for proof in tile_proofs.values())
    assert all(proof["destination_base_alignment_required_bytes"] == 16 for proof in tile_proofs.values())
    assert all(proof["row_tail_bytes"] == 0 for proof in tile_proofs.values())
    assert all(proof["tile_tail_bytes"] == 0 for proof in tile_proofs.values())
    assert all(proof["swizzled_smem_compatible"] for proof in tile_proofs.values())

    assert "per-column 2-byte SMEM scatter" in " ".join(
        tile_proofs["K_T"]["vector_copy_fail_conditions"]
    )
    assert "per-column 2-byte SMEM scatter" in " ".join(
        tile_proofs["Q_T"]["vector_copy_fail_conditions"]
    )


def test_wave6_tma_descriptor_scope_and_mbarrier_bytes() -> None:
    receipt = wave6.build_receipt()
    tma = receipt["tma_cp_async_target"]

    assert tma["descriptor_count"] == 10
    assert tma["non_tma_large_tiles"] == ["dstates_panel_p0", "dstates_panel_p1"]
    assert tma["descriptor_bytes_per_chunk"] == 81920
    assert tma["expected_mbarrier_bytes_per_chunk"] == 81920
    assert tma["dynamic_smem_target_bytes"] == 131072
    assert tma["descriptor_static_pass"] is True
    assert tma["can_claim_green_or_yellow"] is False

    descriptor_names = {descriptor["tile_name"] for descriptor in tma["descriptor_prototypes"]}
    assert descriptor_names == {
        "K",
        "Q",
        "K_T",
        "Q_T",
        "state_panel_p0",
        "state_panel_p1",
        "dPhiO_panel_p0",
        "dPhiO_panel_p1",
        "PsiV_panel_p0",
        "PsiV_panel_p1",
    }
    assert all(descriptor["rank"] == 2 for descriptor in tma["descriptor_prototypes"])
    assert all(descriptor["box_dim_elements"] == [64, 64] for descriptor in tma["descriptor_prototypes"])
    assert all(descriptor["expected_mbarrier_bytes"] == 8192 for descriptor in tma["descriptor_prototypes"])


def test_wave6_ptxas_resource_gate_pass_and_fail() -> None:
    narrow_pass = wave6.evaluate_ptxas_metadata(
        {
            "registers_per_thread": 192,
            "static_smem_bytes": 0,
            "dynamic_smem_bytes": 118784,
            "spill_stores_bytes": 0,
            "spill_loads_bytes": 0,
        },
        variant="narrow_vector_128b_safe_attempt",
    )
    assert narrow_pass["pass"] is True
    assert narrow_pass["total_smem_bytes"] == 118784

    tma_pass = wave6.evaluate_ptxas_metadata(
        {
            "registers_per_thread": 184,
            "static_smem_bytes": 4096,
            "dynamic_smem_bytes": 126976,
            "spill_stores_bytes": 0,
            "spill_loads_bytes": 0,
        },
        variant="tma_cp_async_target",
    )
    assert tma_pass["pass"] is True
    assert tma_pass["total_smem_bytes"] == 131072

    fail = wave6.evaluate_ptxas_metadata(
        {
            "registers_per_thread": 193,
            "static_smem_bytes": 0,
            "dynamic_smem_bytes": 118800,
            "spill_stores_bytes": 8,
            "spill_loads_bytes": 16,
        },
        variant="narrow_vector_128b_safe_attempt",
    )
    assert fail["pass"] is False
    failures = "\n".join(fail["failures"])
    assert "registers_per_thread 193 exceeds pass budget 192" in failures
    assert "total_smem_bytes 118800 exceeds pass budget 118784" in failures
    assert "spill_stores_bytes must be zero, got 8" in failures
    assert "spill_loads_bytes must be zero, got 16" in failures


def test_wave6_ptxas_metadata_unblocks_timing_gate() -> None:
    receipt = wave6.build_receipt(
        ptxas_metadata={
            "registers_per_thread": 192,
            "static_smem_bytes": 0,
            "dynamic_smem_bytes": 118784,
            "spill_stores_bytes": 0,
            "spill_loads_bytes": 0,
        }
    )

    narrow = receipt["narrow_vector_128b_safe_attempt"]
    assert narrow["can_enter_timing_gate"] is True
    assert narrow["status"] == "ready_for_timing_gate"


def test_wave6_copy_path_check_mode(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(wave6._canonical(wave6.build_receipt()))

    result = subprocess.run(
        [
            sys.executable,
            "tools/probes/mamba3_wgmma_wave6_copy_path.py",
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
