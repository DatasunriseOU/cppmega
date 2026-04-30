from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

from tools.probes import mamba3_wgmma_wave9_copy_probe as wave9


PTXAS_SAMPLE = """\
ptxas info    : Compiling entry function 'mamba3_wave9_uint4_copy_12tile_probe' for 'sm_121'
ptxas info    : Function properties for mamba3_wave9_uint4_copy_12tile_probe
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers
ptxas info    : Compile time = 8.000 ms
"""


def _ptxas_ingest() -> dict[str, object]:
    return wave9.build_ptxas_ingest_from_log(
        ptxas_log_text=PTXAS_SAMPLE,
        ptxas_log_path=None,
    )


def _runtime_result() -> dict[str, object]:
    return {
        "schema": wave9.RUNTIME_SCHEMA,
        "evidence": wave9.RUNTIME_RESULT_NAME,
        "date": wave9.DATE,
        "status": "pass",
        "kernel_name": wave9.PROBE_KERNEL,
        "scalar_reference_kernel": wave9.SCALAR_REFERENCE_KERNEL,
        "returncode": 0,
        "command": [
            "<tmp>/mamba3_wgmma_wave9_copy_probe",
            "--chunks=128",
            "--warmup=5",
            "--iters=40",
            "--block-threads=256",
        ],
        "stderr_tail": [],
        "device": {
            "name": "NVIDIA GB10",
            "compute_capability": "12.1",
            "multiprocessor_count": 20,
            "max_dynamic_smem_optin_bytes": 101376,
        },
        "constants": {
            "logical_tile_count": 12,
            "global_tile_count": 10,
            "local_stage_tile_count": 2,
            "tile_rows": 64,
            "tile_cols": 64,
            "dtype": "bf16",
            "dtype_bytes": 2,
            "vector_type": "uint4",
            "vector_bytes": 16,
            "vectors_per_tile": 512,
            "vectors_per_chunk": 6144,
            "copy_bytes_per_chunk": 98304,
            "dynamic_smem_bytes": 98304,
        },
        "launch": {
            "chunks": 128,
            "grid_blocks": 40,
            "block_threads": 256,
            "dynamic_smem_bytes": 98304,
        },
        "correctness": {
            "status": "pass",
            "comparison": "byte_equal_to_scalar_cuda_kernel",
            "status_word": 0,
            "mismatched_elements": 0,
            "first_mismatch_index": None,
            "vector_checksum_fnv1a64": 123456789,
            "scalar_checksum_fnv1a64": 123456789,
        },
        "timing": {
            "status": "measured",
            "warmup_iterations": 5,
            "timed_iterations": 40,
            "logical_payload_bytes_per_iteration": 12582912,
            "copy_stage_bytes_per_iteration": 25165824,
            "vector_avg_us": 72.5,
            "scalar_avg_us": 81.0,
            "speedup_vs_scalar_time": 1.117241,
            "vector_effective_gib_s_copy_stage_bytes": 323.276,
            "scalar_effective_gib_s_payload_bytes": 144.675,
        },
        "blockers": [],
    }


def test_wave9_ptxas_ingest_extracts_vector_kernel_metadata() -> None:
    ingest = _ptxas_ingest()

    assert ingest["status"] == "parsed"
    assert ingest["schema"] == wave9.PTXAS_SCHEMA
    assert ingest["kernel_name"] == wave9.PROBE_KERNEL
    assert ingest["metadata"] == {
        "registers_per_thread": 40,
        "static_smem_bytes": 0,
        "dynamic_smem_bytes": 98304,
        "spill_stores_bytes": 0,
        "spill_loads_bytes": 0,
    }


def test_wave9_runtime_result_passes_correctness_and_timing_checks() -> None:
    check = wave9.evaluate_runtime_result(_runtime_result())

    assert check["pass"] is True
    assert check["failures"] == []


def test_wave9_runtime_result_fails_on_scalar_mismatch() -> None:
    runtime = copy.deepcopy(_runtime_result())
    runtime["status"] = "fail"
    runtime["correctness"]["status"] = "fail"
    runtime["correctness"]["mismatched_elements"] = 1
    runtime["correctness"]["first_mismatch_index"] = 7

    check = wave9.evaluate_runtime_result(runtime)

    assert check["pass"] is False
    failures = "\n".join(check["failures"])
    assert "runtime status must be pass" in failures
    assert "correctness.mismatched_elements must be zero" in failures


def test_wave9_receipt_passes_with_runtime_and_ptxas_evidence() -> None:
    receipt = wave9.build_receipt(
        runtime_result=_runtime_result(),
        ptxas_ingest=_ptxas_ingest(),
    )
    narrow = receipt["narrow_vector_128b_safe_attempt"]

    assert receipt["status"] == "pass_vector_copy_correctness_timing_probe"
    assert narrow["can_integrate_wave10_vector_copy"] is True
    assert narrow["blockers"] == []
    assert narrow["runtime_probe_check"]["pass"] is True
    assert narrow["ptxas_resource_check"]["total_smem_bytes"] == 98304


def test_wave9_receipt_blocks_without_runtime_probe() -> None:
    receipt = wave9.build_receipt(ptxas_ingest=_ptxas_ingest())
    blockers = "\n".join(receipt["narrow_vector_128b_safe_attempt"]["blockers"])

    assert receipt["status"] == "fail_vector_copy_runtime_probe_blocked"
    assert "runtime:" in blockers
    assert "attach Wave9 runtime probe JSON" in blockers


def test_wave9_copy_probe_check_mode(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime.json"
    ptxas_path = tmp_path / "ptxas.json"
    receipt_path = tmp_path / "receipt.json"

    runtime = _runtime_result()
    ptxas = _ptxas_ingest()
    receipt = wave9.build_receipt(runtime_result=runtime, ptxas_ingest=ptxas)
    runtime_path.write_text(wave9._canonical(runtime))
    ptxas_path.write_text(wave9._canonical(ptxas))
    receipt_path.write_text(wave9._canonical(receipt))

    result = subprocess.run(
        [
            sys.executable,
            "tools/probes/mamba3_wgmma_wave9_copy_probe.py",
            "--runtime-result",
            str(runtime_path),
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


def test_wave9_retained_receipt_matches_committed_runtime_and_ptxas() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_path = root / wave9.RUNTIME_RESULT_PATH
    ptxas_path = root / wave9.PTXAS_INGEST_PATH
    receipt_path = root / wave9.RECEIPT_PATH

    result = subprocess.run(
        [
            sys.executable,
            "tools/probes/mamba3_wgmma_wave9_copy_probe.py",
            "--runtime-result",
            str(runtime_path),
            "--ptxas-ingest",
            str(ptxas_path),
            "--check",
            str(receipt_path),
        ],
        check=False,
        cwd=root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
