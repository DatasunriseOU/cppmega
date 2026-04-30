"""Wave9 runtime probe for the Mamba3 mono WGMMA 12-tile uint4 copy path.

Wave8 retained compile-only evidence for the representative 12-logical-tile
copy kernel.  Wave9 turns that into a standalone CUDA runtime probe: compile
the executable probe, parse ptxas for the vector kernel, run a scalar reference
kernel, compare byte-for-byte output, ingest timing, and emit a pass/fail
receipt with explicit blockers.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probes import mamba3_wgmma_wave6_copy_path as wave6
from tools.probes import mamba3_wgmma_wave7_copy_evidence as wave7
from tools.probes import mamba3_wgmma_wave8_copy_evidence as wave8


DATE = "2026-04-30"
RECEIPT_NAME = "mamba3_mono_wgmma_copy_path_wave9_runtime_probe_2026_04_30"
PTXAS_INGEST_NAME = "mamba3_mono_wgmma_copy_path_wave9_ptxas_ingest_2026_04_30"
RUNTIME_RESULT_NAME = "mamba3_mono_wgmma_copy_path_wave9_runtime_2026_04_30"
PROBE_KERNEL = "mamba3_wave9_uint4_copy_12tile_probe"
SCALAR_REFERENCE_KERNEL = "mamba3_wave9_scalar_copy_12tile_reference"
PROBE_SOURCE = ROOT / "tools" / "probes" / "mamba3_wgmma_wave9_copy_probe.cu"
PTXAS_SCHEMA = "mamba3_wave9_ptxas_ingest_v1"
RUNTIME_SCHEMA = "mamba3_wave9_runtime_probe_v1"
RECEIPT_PATH = "docs/status/mamba3_mono_wgmma_copy_path_wave9_receipt_2026_04_30.json"
PTXAS_INGEST_PATH = "docs/status/mamba3_mono_wgmma_copy_path_wave9_ptxas_ingest_2026_04_30.json"
RUNTIME_RESULT_PATH = "docs/status/mamba3_mono_wgmma_copy_path_wave9_runtime_2026_04_30.json"
DEFAULT_CHUNKS = 128
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_TIMED_ITERATIONS = 40
DEFAULT_BLOCK_THREADS = 256

EXTRA_CUTLASS_INCLUDE_CANDIDATES = (
    pathlib.Path("/home/dave/vllm/.deps/vllm-flash-attn-src/csrc/cutlass/include"),
)


def _canonical(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(data))


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_cutlass_include(explicit: pathlib.Path | None) -> pathlib.Path | None:
    candidates: list[pathlib.Path | None] = [explicit] if explicit else []
    for env_name in ("CUTLASS_INCLUDE", "CUTLASS_INCLUDE_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(pathlib.Path(env_value))
    candidates.extend(wave7.DEFAULT_CUTLASS_INCLUDE_CANDIDATES)
    candidates.extend(EXTRA_CUTLASS_INCLUDE_CANDIDATES)
    for candidate in candidates:
        if candidate and (candidate / "cute" / "tensor.hpp").exists():
            return candidate
    return None


def _ptxas_missing(reason: str) -> dict[str, Any]:
    return {
        "schema": PTXAS_SCHEMA,
        "evidence": PTXAS_INGEST_NAME,
        "date": DATE,
        "status": "missing_ptxas_ingest",
        "coverage": None,
        "kernel_name": PROBE_KERNEL,
        "dynamic_smem_bytes_from_probe_contract": wave8.MULTI_TILE_COPY_PROBE_DYNAMIC_SMEM_BYTES,
        "metadata": None,
        "ptxas_evidence": None,
        "failures": [reason],
    }


def _runtime_missing(reason: str) -> dict[str, Any]:
    return {
        "schema": RUNTIME_SCHEMA,
        "evidence": RUNTIME_RESULT_NAME,
        "date": DATE,
        "status": "missing_runtime_probe",
        "kernel_name": PROBE_KERNEL,
        "scalar_reference_kernel": SCALAR_REFERENCE_KERNEL,
        "correctness": {"status": "not_run"},
        "timing": {"status": "not_run"},
        "blockers": [reason],
    }


def normalize_ptxas_ingest(
    data: dict[str, Any] | None,
    *,
    source_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    if data is None:
        return _ptxas_missing("attach Wave9 ptxas ingest JSON")

    normalized = wave8.normalize_ptxas_ingest(data, source_path=source_path)
    normalized.update(
        {
            "schema": PTXAS_SCHEMA,
            "evidence": data.get("evidence", PTXAS_INGEST_NAME),
            "date": data.get("date", DATE),
            "kernel_name": data.get("kernel_name")
            or normalized.get("kernel_name")
            or PROBE_KERNEL,
            "coverage": data.get(
                "coverage",
                "standalone Wave9 runtime probe ptxas ingest for the 12-tile uint4 copy kernel",
            ),
        }
    )
    if normalized["evidence"] == wave8.PTXAS_INGEST_NAME:
        normalized["evidence"] = PTXAS_INGEST_NAME
    return normalized


def build_ptxas_ingest_from_log(
    *,
    ptxas_log_text: str,
    ptxas_log_path: pathlib.Path | None,
) -> dict[str, Any]:
    ingest = wave8.build_ptxas_ingest_from_log(
        ptxas_log_text=ptxas_log_text,
        ptxas_log_path=ptxas_log_path,
        dynamic_smem_bytes=wave8.MULTI_TILE_COPY_PROBE_DYNAMIC_SMEM_BYTES,
        kernel_name=PROBE_KERNEL,
        coverage=(
            "standalone Wave9 CUDA/CuTe runtime probe; ptxas resource evidence "
            "for the 12-logical-tile uint4 vector copy kernel"
        ),
    )
    ingest["schema"] = PTXAS_SCHEMA
    ingest["evidence"] = PTXAS_INGEST_NAME
    return ingest


def compile_runtime_probe(
    *,
    output: pathlib.Path,
    cuda_arch: str,
    cutlass_include: pathlib.Path | None,
) -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return {
            "status": "skipped",
            "reason": "nvcc was not found in PATH",
            "ptxas_ingest": _ptxas_missing("nvcc was not found in PATH"),
        }

    include_dir = _find_cutlass_include(cutlass_include)
    if include_dir is None:
        return {
            "status": "skipped",
            "reason": "no CUTLASS/CuTe include directory with cute/tensor.hpp was found",
            "ptxas_ingest": _ptxas_missing(
                "no CUTLASS/CuTe include directory with cute/tensor.hpp was found"
            ),
        }

    command = [
        nvcc,
        "-std=c++17",
        "-O3",
        f"-arch={cuda_arch}",
        f"-I{include_dir}",
        "--ptxas-options=-v",
        str(PROBE_SOURCE),
        "-o",
        str(output),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    ptxas_text = "\n".join(part for part in (result.stdout, result.stderr) if part)
    ptxas_ingest = build_ptxas_ingest_from_log(
        ptxas_log_text=ptxas_text,
        ptxas_log_path=None,
    )
    sanitized_command = [
        "<nvcc>" if part == nvcc else "<tmp>/mamba3_wgmma_wave9_copy_probe" if part == str(output) else part
        for part in command
    ]
    return {
        "status": "compiled" if result.returncode == 0 else "compile_failed",
        "returncode": result.returncode,
        "command": sanitized_command,
        "cuda_arch": cuda_arch,
        "cutlass_include": str(include_dir),
        "ptxas_ingest": ptxas_ingest,
        "compiler_output_tail": ptxas_text.splitlines()[-40:],
    }


def run_runtime_probe(
    *,
    binary: pathlib.Path,
    chunks: int,
    warmup_iterations: int,
    timed_iterations: int,
    block_threads: int,
    grid_blocks: int | None,
) -> dict[str, Any]:
    command = [
        str(binary),
        f"--chunks={chunks}",
        f"--warmup={warmup_iterations}",
        f"--iters={timed_iterations}",
        f"--block-threads={block_threads}",
    ]
    if grid_blocks is not None:
        command.append(f"--grid-blocks={grid_blocks}")

    result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=180)
    sanitized_command = [
        "<tmp>/mamba3_wgmma_wave9_copy_probe" if part == str(binary) else part
        for part in command
    ]
    try:
        runtime = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            **_runtime_missing("runtime probe did not emit parseable JSON"),
            "returncode": result.returncode,
            "command": sanitized_command,
            "stdout_tail": result.stdout.splitlines()[-40:],
            "stderr_tail": result.stderr.splitlines()[-40:],
        }

    runtime["evidence"] = runtime.get("evidence", RUNTIME_RESULT_NAME)
    runtime["date"] = runtime.get("date", DATE)
    runtime["returncode"] = result.returncode
    runtime["command"] = sanitized_command
    runtime["stderr_tail"] = result.stderr.splitlines()[-40:]
    if result.returncode != 0 and runtime.get("status") == "pass":
        runtime["status"] = "fail"
        runtime.setdefault("blockers", []).append(
            f"runtime probe returned non-zero exit status {result.returncode}"
        )
    return runtime


def evaluate_runtime_result(runtime: dict[str, Any] | None) -> dict[str, Any]:
    if runtime is None:
        return {
            "status": "missing_runtime_probe",
            "pass": False,
            "runtime": None,
            "failures": [
                "attach Wave9 runtime probe JSON with scalar correctness and timing results"
            ],
        }

    failures: list[str] = []
    if runtime.get("schema") != RUNTIME_SCHEMA:
        failures.append(f"runtime schema must be {RUNTIME_SCHEMA}, got {runtime.get('schema')}")
    if runtime.get("kernel_name") != PROBE_KERNEL:
        failures.append(f"kernel_name must be {PROBE_KERNEL}, got {runtime.get('kernel_name')}")
    if runtime.get("scalar_reference_kernel") != SCALAR_REFERENCE_KERNEL:
        failures.append(
            "scalar_reference_kernel must be "
            f"{SCALAR_REFERENCE_KERNEL}, got {runtime.get('scalar_reference_kernel')}"
        )
    if runtime.get("status") != "pass":
        failures.append(f"runtime status must be pass, got {runtime.get('status')}")
    if _as_int(runtime.get("returncode", 0)) != 0:
        failures.append(f"runtime returncode must be zero, got {runtime.get('returncode')}")

    constants = runtime.get("constants")
    if not isinstance(constants, dict):
        failures.append("runtime constants must be present")
        constants = {}
    expected_ints = {
        "logical_tile_count": 12,
        "global_tile_count": 10,
        "local_stage_tile_count": 2,
        "tile_rows": wave6.TILE_ROWS,
        "tile_cols": wave6.TILE_COLS,
        "dtype_bytes": wave6.BF16_BYTES,
        "vector_bytes": wave6.VECTOR_BYTES,
        "vectors_per_tile": wave6.TILE_BYTES // wave6.VECTOR_BYTES,
        "vectors_per_chunk": 6144,
        "copy_bytes_per_chunk": 98304,
        "dynamic_smem_bytes": 98304,
    }
    for field, expected in expected_ints.items():
        value = _as_int(constants.get(field))
        if value != expected:
            failures.append(f"constants.{field} must be {expected}, got {constants.get(field)}")
    if constants.get("vector_type") != "uint4":
        failures.append(f"constants.vector_type must be uint4, got {constants.get('vector_type')}")
    if constants.get("dtype") != "bf16":
        failures.append(f"constants.dtype must be bf16, got {constants.get('dtype')}")

    launch = runtime.get("launch")
    if not isinstance(launch, dict):
        failures.append("runtime launch must be present")
        launch = {}
    if _as_int(launch.get("chunks")) is None or _as_int(launch.get("chunks")) <= 0:
        failures.append("launch.chunks must be a positive integer")
    if _as_int(launch.get("block_threads")) is None or _as_int(launch.get("block_threads")) <= 0:
        failures.append("launch.block_threads must be a positive integer")
    if _as_int(launch.get("dynamic_smem_bytes")) != 98304:
        failures.append(
            f"launch.dynamic_smem_bytes must be 98304, got {launch.get('dynamic_smem_bytes')}"
        )

    correctness = runtime.get("correctness")
    if not isinstance(correctness, dict):
        failures.append("runtime correctness must be present")
        correctness = {}
    if correctness.get("status") != "pass":
        failures.append(f"correctness.status must be pass, got {correctness.get('status')}")
    if correctness.get("comparison") != "byte_equal_to_scalar_cuda_kernel":
        failures.append("correctness.comparison must be byte_equal_to_scalar_cuda_kernel")
    if _as_int(correctness.get("status_word")) != 0:
        failures.append(f"correctness.status_word must be zero, got {correctness.get('status_word')}")
    if _as_int(correctness.get("mismatched_elements")) != 0:
        failures.append(
            "correctness.mismatched_elements must be zero, "
            f"got {correctness.get('mismatched_elements')}"
        )
    if correctness.get("vector_checksum_fnv1a64") != correctness.get("scalar_checksum_fnv1a64"):
        failures.append("vector and scalar checksums must match")

    timing = runtime.get("timing")
    if not isinstance(timing, dict):
        failures.append("runtime timing must be present")
        timing = {}
    if timing.get("status") != "measured":
        failures.append(f"timing.status must be measured, got {timing.get('status')}")
    for field in ("vector_avg_us", "scalar_avg_us"):
        value = _as_float(timing.get(field))
        if value is None or value <= 0.0:
            failures.append(f"timing.{field} must be positive, got {timing.get(field)}")
    if _as_int(timing.get("timed_iterations")) is None or _as_int(timing.get("timed_iterations")) <= 0:
        failures.append("timing.timed_iterations must be a positive integer")
    if _as_float(timing.get("vector_effective_gib_s_copy_stage_bytes")) is None:
        failures.append("timing.vector_effective_gib_s_copy_stage_bytes must be present")

    blockers = runtime.get("blockers")
    if blockers:
        failures.extend(f"runtime blocker: {blocker}" for blocker in blockers)

    return {
        "status": "pass" if not failures else "fail",
        "pass": not failures,
        "runtime": runtime,
        "failures": failures,
    }


def _ptxas_resource_check(ptxas_ingest: dict[str, Any]) -> dict[str, Any]:
    return wave6.evaluate_ptxas_metadata(
        ptxas_ingest.get("metadata"),
        variant="narrow_vector_128b_safe_attempt",
    )


def _receipt_blockers(
    *,
    alignment_check: dict[str, Any],
    ptxas_ingest: dict[str, Any],
    ptxas_resource_check: dict[str, Any],
    runtime_check: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not alignment_check["pass"]:
        blockers.extend(f"alignment/layout: {failure}" for failure in alignment_check["failures"])
    if ptxas_ingest.get("status") != "parsed":
        blockers.extend(f"ptxas-ingest: {failure}" for failure in ptxas_ingest.get("failures", []))
    if ptxas_ingest.get("kernel_name") != PROBE_KERNEL:
        blockers.append(
            f"ptxas-ingest: kernel_name must be {PROBE_KERNEL}, got {ptxas_ingest.get('kernel_name')}"
        )
    if not ptxas_resource_check["pass"]:
        blockers.extend(f"ptxas-resource: {failure}" for failure in ptxas_resource_check["failures"])
    if not runtime_check["pass"]:
        blockers.extend(f"runtime: {failure}" for failure in runtime_check["failures"])
    return blockers


def _compile_probe_summary(ptxas_ingest: dict[str, Any]) -> dict[str, Any]:
    if ptxas_ingest.get("status") == "parsed" and ptxas_ingest.get("kernel_name") == PROBE_KERNEL:
        return {
            "status": "compiled_retained_ptxas_ingest",
            "command": (
                "python tools/probes/mamba3_wgmma_wave9_copy_probe.py "
                f"--compile-run --write-runtime {RUNTIME_RESULT_PATH} "
                f"--write-ptxas-ingest {PTXAS_INGEST_PATH} --write {RECEIPT_PATH}"
            ),
            "evidence": PTXAS_INGEST_PATH,
            "kernel_name": PROBE_KERNEL,
            "dynamic_smem_bytes_from_probe_contract": ptxas_ingest.get(
                "dynamic_smem_bytes_from_probe_contract"
            ),
            "metadata": ptxas_ingest.get("metadata"),
            "coverage": ptxas_ingest.get("coverage"),
        }
    return {
        "status": "not_available",
        "command": (
            "python tools/probes/mamba3_wgmma_wave9_copy_probe.py "
            f"--compile-run --write-runtime {RUNTIME_RESULT_PATH} "
            f"--write-ptxas-ingest {PTXAS_INGEST_PATH} --write {RECEIPT_PATH}"
        ),
    }


def _runtime_probe_summary(runtime_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": runtime_result.get("status"),
        "evidence": RUNTIME_RESULT_PATH,
        "command": runtime_result.get("command"),
        "device": runtime_result.get("device"),
        "launch": runtime_result.get("launch"),
        "correctness": runtime_result.get("correctness"),
        "timing": runtime_result.get("timing"),
    }


def _wave10_recommendation(pass_receipt: bool) -> list[str]:
    if pass_receipt:
        return [
            "Wave10 should integrate the uint4 vector copy into the CuTe lane before switching to TMA descriptor smoke.",
            "Carry the ptr%16, row-stride, panel-offset, dynamic-SMEM, and K_T/Q_T physical-layout guards forward.",
            "Keep the post-dst-write __syncthreads() when reusing the 12-tile SMEM staging buffer across chunks.",
            "Use TMA descriptor smoke only if the integrated CuTe lane regresses correctness, spills, or timing.",
        ]
    return [
        "Wave10 should not integrate the vector lane until the listed runtime or ptxas blockers are cleared.",
        "If those blockers are structural in generated CuTe code, switch Wave10 to TMA descriptor smoke.",
    ]


def build_receipt(
    *,
    runtime_result: dict[str, Any] | None = None,
    ptxas_ingest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_ptxas_ingest = normalize_ptxas_ingest(ptxas_ingest)
    effective_runtime_result = runtime_result or _runtime_missing("attach Wave9 runtime probe JSON")

    alignment_evidence = wave8.build_alignment_layout_evidence()
    alignment_check = wave8.evaluate_alignment_layout_evidence(alignment_evidence)
    ptxas_resource_check = _ptxas_resource_check(effective_ptxas_ingest)
    runtime_check = evaluate_runtime_result(effective_runtime_result)
    can_integrate = bool(
        alignment_check["pass"]
        and effective_ptxas_ingest.get("status") == "parsed"
        and effective_ptxas_ingest.get("kernel_name") == PROBE_KERNEL
        and ptxas_resource_check["pass"]
        and runtime_check["pass"]
    )
    blockers = _receipt_blockers(
        alignment_check=alignment_check,
        ptxas_ingest=effective_ptxas_ingest,
        ptxas_resource_check=ptxas_resource_check,
        runtime_check=runtime_check,
    )

    receipt = {
        "receipt": RECEIPT_NAME,
        "status": (
            "pass_vector_copy_correctness_timing_probe"
            if can_integrate
            else "fail_vector_copy_runtime_probe_blocked"
        ),
        "date": DATE,
        "branch": "worker/mamba3-mono-triton-model",
        "source_receipts": [
            "docs/status/mamba3_mono_wgmma_copy_path_wave8_receipt_2026_04_30.json",
            "docs/status/mamba3_mono_wgmma_copy_path_wave8_ptxas_ingest_2026_04_30.json",
        ],
        "scope": {
            "variant": "narrow_vector_128b_safe_attempt",
            "probe": "standalone representative 12-tile CUDA/CuTe uint4 copy runtime probe",
            "logical_tile_movements": len(wave8.EXPECTED_TILE_NAMES),
            "copy_vectors_per_chunk": len(wave8.EXPECTED_TILE_NAMES)
            * (wave6.TILE_BYTES // wave6.VECTOR_BYTES),
            "copy_bytes_per_chunk": len(wave8.EXPECTED_TILE_NAMES) * wave6.TILE_BYTES,
            "global_tile_count": 10,
            "cta_local_tile_count": 2,
            "correctness_rule": "uint4 vector copy output must match scalar CUDA copy byte-for-byte",
            "timing_rule": "timing must be measured and ingested; no speedup threshold is used as a correctness gate",
        },
        "narrow_vector_128b_safe_attempt": {
            "status": "pass" if can_integrate else "fail",
            "can_integrate_wave10_vector_copy": can_integrate,
            "blockers": blockers,
            "alignment_layout_evidence_check": alignment_check,
            "ptxas_ingest": effective_ptxas_ingest,
            "ptxas_resource_check": ptxas_resource_check,
            "runtime_probe_check": runtime_check,
            "required_evidence_fields": {
                "runtime": [
                    "correctness.status",
                    "correctness.mismatched_elements",
                    "correctness.vector_checksum_fnv1a64",
                    "correctness.scalar_checksum_fnv1a64",
                    "timing.vector_avg_us",
                    "timing.scalar_avg_us",
                    "timing.vector_effective_gib_s_copy_stage_bytes",
                ],
                "ptxas_ingest": [
                    f"metadata.{field}" for field in wave6.PTXAS_REQUIRED_FIELDS
                ],
            },
        },
        "compile_probe": _compile_probe_summary(effective_ptxas_ingest),
        "runtime_probe": _runtime_probe_summary(effective_runtime_result),
        "validation": {
            "generated_by": "tools/probes/mamba3_wgmma_wave9_copy_probe.py",
            "runtime_result": RUNTIME_RESULT_PATH,
            "ptxas_ingest": PTXAS_INGEST_PATH,
            "check_command": (
                "python tools/probes/mamba3_wgmma_wave9_copy_probe.py "
                f"--runtime-result {RUNTIME_RESULT_PATH} "
                f"--ptxas-ingest {PTXAS_INGEST_PATH} "
                f"--check {RECEIPT_PATH}"
            ),
            "compile_run_command": (
                "python tools/probes/mamba3_wgmma_wave9_copy_probe.py "
                f"--compile-run --chunks {DEFAULT_CHUNKS} "
                f"--warmup {DEFAULT_WARMUP_ITERATIONS} --iters {DEFAULT_TIMED_ITERATIONS} "
                f"--write-runtime {RUNTIME_RESULT_PATH} "
                f"--write-ptxas-ingest {PTXAS_INGEST_PATH} "
                f"--write {RECEIPT_PATH}"
            ),
        },
        "wave10_recommendation": _wave10_recommendation(can_integrate),
    }
    _validate_receipt(receipt)
    return receipt


def _validate_receipt(receipt: dict[str, Any]) -> None:
    narrow = receipt["narrow_vector_128b_safe_attempt"]
    assert receipt["scope"]["logical_tile_movements"] == 12
    assert receipt["scope"]["copy_vectors_per_chunk"] == 6144
    assert receipt["scope"]["copy_bytes_per_chunk"] == 98304
    assert narrow["ptxas_resource_check"]["variant"] == "narrow_vector_128b_safe_attempt"
    if narrow["can_integrate_wave10_vector_copy"]:
        assert receipt["status"] == "pass_vector_copy_correctness_timing_probe"
        assert narrow["alignment_layout_evidence_check"]["pass"]
        assert narrow["ptxas_ingest"]["status"] == "parsed"
        assert narrow["ptxas_resource_check"]["pass"]
        assert narrow["runtime_probe_check"]["pass"]
        assert not narrow["blockers"]
    else:
        assert narrow["blockers"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=pathlib.Path, default=None)
    parser.add_argument("--write", type=pathlib.Path, default=None)
    parser.add_argument("--compile-run", action="store_true")
    parser.add_argument("--runtime-result", type=pathlib.Path, default=None)
    parser.add_argument("--write-runtime", type=pathlib.Path, default=None)
    parser.add_argument("--ptxas-ingest", type=pathlib.Path, default=None)
    parser.add_argument("--ptxas-log", type=pathlib.Path, default=None)
    parser.add_argument("--write-ptxas-ingest", type=pathlib.Path, default=None)
    parser.add_argument("--cuda-arch", default="sm_121")
    parser.add_argument("--cutlass-include", type=pathlib.Path, default=None)
    parser.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument("--iters", type=int, default=DEFAULT_TIMED_ITERATIONS)
    parser.add_argument("--block-threads", type=int, default=DEFAULT_BLOCK_THREADS)
    parser.add_argument("--grid-blocks", type=int, default=None)
    args = parser.parse_args()

    runtime_result: dict[str, Any] | None = None
    ptxas_ingest: dict[str, Any] | None = None

    if args.compile_run:
        with tempfile.TemporaryDirectory(prefix="cppmega_wave9_copy_") as tmp:
            binary = pathlib.Path(tmp) / "mamba3_wgmma_wave9_copy_probe"
            compile_result = compile_runtime_probe(
                output=binary,
                cuda_arch=args.cuda_arch,
                cutlass_include=args.cutlass_include,
            )
            ptxas_ingest = compile_result["ptxas_ingest"]
            if compile_result["status"] == "compiled":
                runtime_result = run_runtime_probe(
                    binary=binary,
                    chunks=args.chunks,
                    warmup_iterations=args.warmup,
                    timed_iterations=args.iters,
                    block_threads=args.block_threads,
                    grid_blocks=args.grid_blocks,
                )
            else:
                runtime_result = _runtime_missing(
                    f"compile status was {compile_result['status']}: {compile_result.get('reason', 'see compiler output')}"
                )

    if args.runtime_result:
        runtime_result = _read_json(args.runtime_result)
    if args.ptxas_ingest:
        ptxas_ingest = _read_json(args.ptxas_ingest)
    elif args.ptxas_log and ptxas_ingest is None:
        ptxas_ingest = build_ptxas_ingest_from_log(
            ptxas_log_text=args.ptxas_log.read_text(),
            ptxas_log_path=args.ptxas_log,
        )

    if args.write_runtime:
        if runtime_result is None:
            raise SystemExit("--write-runtime requires --compile-run or --runtime-result")
        _write_json(args.write_runtime, runtime_result)

    if args.write_ptxas_ingest:
        if ptxas_ingest is None:
            raise SystemExit("--write-ptxas-ingest requires --compile-run, --ptxas-ingest, or --ptxas-log")
        _write_json(args.write_ptxas_ingest, normalize_ptxas_ingest(ptxas_ingest))

    receipt = build_receipt(
        runtime_result=runtime_result,
        ptxas_ingest=ptxas_ingest,
    )
    rendered = _canonical(receipt)

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(rendered)

    if args.check is not None:
        actual = args.check.read_text()
        if actual != rendered:
            raise SystemExit(f"{args.check} does not match generated receipt")
        return

    if args.write is None:
        print(rendered, end="")


if __name__ == "__main__":
    main()
