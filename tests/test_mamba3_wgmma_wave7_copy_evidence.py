from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.probes import mamba3_wgmma_wave7_copy_evidence as wave7


PTXAS_SAMPLE = """\
ptxas info    : Compiling entry function 'mamba3_wave7_narrow_copy_probe' for 'sm_121'
ptxas info    : Function properties for mamba3_wave7_narrow_copy_probe
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, 0 bytes smem, 392 bytes cmem[0]
"""


def _passing_alignment_evidence() -> dict[str, object]:
    return {
        "vector_bytes": 16,
        "tile_rows": 64,
        "tile_cols": 64,
        "dtype_bytes": 2,
        "row_bytes": 128,
        "tile_bytes": 8192,
        "row_tail_bytes": 0,
        "tile_tail_bytes": 0,
        "global_base_alignment_bytes": 16,
        "smem_base_alignment_bytes": 16,
        "runtime_global_alignment_guard": True,
        "runtime_smem_alignment_guard": True,
        "row_stride_alignment_bytes": 128,
        "uses_16b_contiguous_vector_type": True,
        "masked_tail_path_present": False,
        "tiles_covered": wave7.EXPECTED_TILE_NAMES,
        "kt_qt_vector_compatible_layout": True,
    }


def test_wave7_parse_ptxas_verbose_extracts_wave6_metadata() -> None:
    parsed = wave7.parse_ptxas_verbose(
        PTXAS_SAMPLE,
        dynamic_smem_bytes=8192,
        kernel_name="narrow_copy",
    )

    assert parsed["status"] == "parsed"
    assert parsed["metadata"] == {
        "registers_per_thread": 40,
        "static_smem_bytes": 0,
        "dynamic_smem_bytes": 8192,
        "spill_stores_bytes": 0,
        "spill_loads_bytes": 0,
    }


def test_wave7_placeholder_receipt_fails_until_evidence_is_attached() -> None:
    receipt = wave7.build_receipt()
    narrow = receipt["narrow_vector_128b_safe_attempt"]

    assert receipt["status"] == "fail_missing_or_incomplete_copy_evidence"
    assert narrow["can_promote_wave6_gate"] is False
    assert narrow["ptxas_evidence"]["status"] == "missing_ptxas_log"
    assert narrow["alignment_evidence_check"]["status"] == "missing_alignment_evidence"


def test_wave7_narrow_vector_receipt_passes_with_ptxas_and_alignment_evidence() -> None:
    receipt = wave7.build_receipt(
        ptxas_log_text=PTXAS_SAMPLE,
        dynamic_smem_bytes=8192,
        kernel_name="narrow_copy",
        alignment_evidence=_passing_alignment_evidence(),
    )
    narrow = receipt["narrow_vector_128b_safe_attempt"]

    assert receipt["status"] == "pass_narrow_vector_evidence_ready"
    assert narrow["can_promote_wave6_gate"] is True
    assert narrow["ptxas_resource_check"]["pass"] is True
    assert narrow["alignment_evidence_check"]["pass"] is True
    assert narrow["ptxas_resource_check"]["total_smem_bytes"] == 8192


def test_wave7_alignment_evidence_fails_on_partial_tile_coverage() -> None:
    evidence = _passing_alignment_evidence()
    evidence["tiles_covered"] = ["K"]
    check = wave7.evaluate_alignment_evidence(evidence)

    assert check["pass"] is False
    assert "tiles_covered is missing Wave6 tiles" in "\n".join(check["failures"])


def test_wave7_tma_descriptor_smoke_checklist_is_concrete() -> None:
    checklist = wave7.tma_descriptor_smoke_checklist()
    by_id = {item["id"]: item for item in checklist}

    assert by_id["descriptor_scope"]["expected_tiles"] == wave7.TMA_TILE_NAMES
    assert by_id["descriptor_scope"]["forbidden_tiles"] == [
        "dstates_panel_p0",
        "dstates_panel_p1",
        "tiny scalar/vector slices",
    ]
    assert by_id["mbarrier_expected_bytes"]["expected_bytes_per_chunk"] == 81920
    assert by_id["resource_gate"]["max_total_smem_bytes"] == 131072


def test_wave7_copy_evidence_check_mode(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(wave7._canonical(wave7.build_receipt()))

    result = subprocess.run(
        [
            sys.executable,
            "tools/probes/mamba3_wgmma_wave7_copy_evidence.py",
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
