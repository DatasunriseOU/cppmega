from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

from tools.probes import mamba3_wgmma_wave8_copy_evidence as wave8


def _ptxas_ingest() -> dict[str, object]:
    return {
        "schema": "mamba3_wave8_ptxas_ingest_v1",
        "status": "parsed",
        "coverage": "unit-test ptxas ingest",
        "kernel_name": wave8.MULTI_TILE_COPY_PROBE_KERNEL,
        "dynamic_smem_bytes_from_probe_contract": wave8.MULTI_TILE_COPY_PROBE_DYNAMIC_SMEM_BYTES,
        "metadata": {
            "registers_per_thread": 40,
            "static_smem_bytes": 0,
            "dynamic_smem_bytes": wave8.MULTI_TILE_COPY_PROBE_DYNAMIC_SMEM_BYTES,
            "spill_stores_bytes": 0,
            "spill_loads_bytes": 0,
        },
        "failures": [],
    }


def test_wave8_generated_alignment_evidence_covers_all_12_tiles() -> None:
    evidence = wave8.build_alignment_layout_evidence()
    check = wave8.evaluate_alignment_layout_evidence(evidence)

    assert check["pass"] is True
    assert check["tile_count"] == 12
    assert evidence["copy_vectors_per_chunk"] == 6144
    assert evidence["copy_bytes_per_chunk"] == 98304
    assert evidence["global_tile_count"] == 10
    assert evidence["cta_local_tile_count"] == 2

    by_name = {tile["name"]: tile for tile in evidence["tile_evidence"]}
    assert set(by_name) == set(wave8.EXPECTED_TILE_NAMES)
    assert by_name["K_T"]["transpose_layout_proof"]["per_column_scalar_scatter"] is False
    assert by_name["Q_T"]["transpose_layout_proof"]["vector_compatible"] is True
    assert by_name["dstates_panel_p0"]["tma_candidate"] is False
    assert all(tile["vector_type"] == "uint4" for tile in evidence["tile_evidence"])


def test_wave8_alignment_evidence_fails_if_transpose_degrades_to_scalar_scatter() -> None:
    evidence = wave8.build_alignment_layout_evidence()
    degraded = copy.deepcopy(evidence)
    by_name = {tile["name"]: tile for tile in degraded["tile_evidence"]}
    by_name["Q_T"]["transpose_layout_proof"]["per_column_scalar_scatter"] = True

    check = wave8.evaluate_alignment_layout_evidence(degraded)

    assert check["pass"] is False
    assert "Q_T: transpose_layout_proof.per_column_scalar_scatter must be false" in "\n".join(
        check["failures"]
    )


def test_wave8_receipt_passes_with_alignment_and_ptxas_ingest() -> None:
    receipt = wave8.build_receipt(
        alignment_evidence=wave8.build_alignment_layout_evidence(),
        ptxas_ingest=_ptxas_ingest(),
    )
    narrow = receipt["narrow_vector_128b_safe_attempt"]

    assert receipt["status"] == "pass_narrow_vector_12tile_evidence_ready"
    assert narrow["can_promote_wave6_gate"] is True
    assert narrow["blockers"] == []
    assert narrow["alignment_layout_evidence_check"]["tile_count"] == 12
    assert narrow["ptxas_resource_check"]["total_smem_bytes"] == 98304


def test_wave8_receipt_without_evidence_fails_with_precise_blockers() -> None:
    receipt = wave8.build_receipt()
    blockers = "\n".join(receipt["narrow_vector_128b_safe_attempt"]["blockers"])

    assert receipt["status"] == "fail_missing_or_incomplete_12tile_copy_evidence"
    assert "alignment/layout: attach deterministic Wave8 alignment/layout evidence" in blockers
    assert "ptxas-ingest: attach ptxas-ingest JSON" in blockers


def test_wave8_normalizes_wave7_style_ptxas_json() -> None:
    ingest = wave8.normalize_ptxas_ingest(
        {
            "coverage": "Wave7 local ptxas JSON",
            "resource_metadata": {
                "registers_per_thread": 40,
                "static_smem_bytes": 0,
                "dynamic_smem_bytes": 8192,
                "spill_stores_bytes": 0,
                "spill_loads_bytes": 0,
            },
        }
    )

    assert ingest["status"] == "parsed"
    assert ingest["metadata"]["registers_per_thread"] == 40
    assert ingest["dynamic_smem_bytes_from_probe_contract"] == 8192


def test_wave8_copy_evidence_check_mode(tmp_path: Path) -> None:
    alignment_path = tmp_path / "alignment.json"
    ptxas_path = tmp_path / "ptxas.json"
    receipt_path = tmp_path / "receipt.json"

    alignment = wave8.build_alignment_layout_evidence()
    ptxas = _ptxas_ingest()
    receipt = wave8.build_receipt(alignment_evidence=alignment, ptxas_ingest=ptxas)
    alignment_path.write_text(wave8._canonical(alignment))
    ptxas_path.write_text(wave8._canonical(ptxas))
    receipt_path.write_text(wave8._canonical(receipt))

    result = subprocess.run(
        [
            sys.executable,
            "tools/probes/mamba3_wgmma_wave8_copy_evidence.py",
            "--alignment-evidence",
            str(alignment_path),
            "--ptxas-ingest",
            str(ptxas_path),
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
