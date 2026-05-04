from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from tools.probes import mamba3_wgmma_wave8_copy_evidence as wave8
from tools.probes import mamba3_wgmma_wave9_copy_probe as wave9
from tools.probes import mamba3_wgmma_wave10_copy_integration as wave10


ROOT = Path(__file__).resolve().parents[1]


def _runtime_result() -> dict[str, object]:
    return json.loads((ROOT / wave9.RUNTIME_RESULT_PATH).read_text())


def _ptxas_ingest() -> dict[str, object]:
    return json.loads((ROOT / wave9.PTXAS_INGEST_PATH).read_text())


def test_wave10_guard_header_is_generated_and_consumed_by_probe() -> None:
    header_path = ROOT / wave10.GUARD_HEADER_PATH
    probe_path = ROOT / "tools/probes/mamba3_wgmma_wave9_copy_probe.cu"

    assert header_path.read_text() == wave10.render_guard_header()

    probe_source = probe_path.read_text()
    assert '#include "mamba3_wgmma_wave10_copy_guards.hpp"' in probe_source
    assert "mamba3_wave10_copy::runtime_guard_bits" in probe_source
    assert "mamba3_wave10_copy::dynamic_smem_optin_guard" in probe_source


def test_wave10_guard_contract_covers_cute_layout_requirements() -> None:
    contract = wave10.build_guard_contract()
    check = wave10.evaluate_guard_contract(contract)

    assert check["pass"] is True
    assert contract["constants"]["vectors_per_chunk"] == 6144
    assert contract["constants"]["dynamic_smem_bytes"] == 98304
    assert contract["expected_data_layouts"]["global_source_tiles_per_chunk"] == 10
    assert contract["expected_data_layouts"]["local_stage_tiles_per_chunk"] == 2
    assert contract["guard_header"]["consumed_by"] == (
        "tools/probes/mamba3_wgmma_wave9_copy_probe.cu"
    )

    by_name = {
        tile["name"]: tile for tile in contract["expected_data_layouts"]["tiles"]
    }
    assert by_name["K_T"]["transpose_layout_proof"]["per_column_scalar_scatter"] is False
    assert by_name["Q_T"]["transpose_layout_proof"]["vector_compatible"] is True


def test_wave10_receipt_recommends_vector_path_with_wave9_evidence() -> None:
    receipt = wave10.build_receipt(
        runtime_result=_runtime_result(),
        ptxas_ingest=_ptxas_ingest(),
    )

    assert receipt["status"] == "pass_cute_integration_ready_vector_first"
    assert receipt["decision"]["vector_path_ready"] is True
    assert receipt["decision"]["recommendation"] == "integrate_vector_path_next"
    assert receipt["blockers"] == []
    assert receipt["checks"]["runtime_probe"]["pass"] is True
    assert receipt["checks"]["ptxas_resource"]["metadata"]["registers_per_thread"] == 40


def test_wave10_receipt_blocks_if_transpose_degrades_to_scalar_scatter() -> None:
    alignment = wave8.build_alignment_layout_evidence()
    degraded = copy.deepcopy(alignment)
    by_name = {tile["name"]: tile for tile in degraded["tile_evidence"]}
    by_name["K_T"]["transpose_layout_proof"]["per_column_scalar_scatter"] = True

    receipt = wave10.build_receipt(
        runtime_result=_runtime_result(),
        ptxas_ingest=_ptxas_ingest(),
        alignment_evidence=degraded,
    )

    assert receipt["status"] == "fail_cute_integration_blocked"
    assert receipt["decision"]["vector_path_ready"] is False
    assert "K_T: transpose_layout_proof.per_column_scalar_scatter must be false" in "\n".join(
        receipt["blockers"]
    )


def test_wave10_retained_artifacts_match_generator() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/probes/mamba3_wgmma_wave10_copy_integration.py",
            "--runtime-result",
            wave9.RUNTIME_RESULT_PATH,
            "--ptxas-ingest",
            wave9.PTXAS_INGEST_PATH,
            "--check",
            wave10.RECEIPT_PATH,
            "--check-guard-header",
            wave10.GUARD_HEADER_PATH,
            "--check-guide",
            wave10.GUIDE_PATH,
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
