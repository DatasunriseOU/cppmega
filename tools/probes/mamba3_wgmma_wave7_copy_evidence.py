"""Wave7 evidence receipt for the Mamba3 mono WGMMA copy path.

Wave6 modeled the static 128-bit copy gates.  This helper ingests real
``ptxas -v`` output plus an explicit alignment evidence JSON and reports
whether the narrow-vector attempt can be promoted from modeled to compiled
evidence.  It also records the concrete TMA descriptor smoke checklist that
must pass before the cp.async/TMA path can claim a timing color.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path = [entry for entry in sys.path if entry != str(ROOT)]
sys.path.insert(0, str(ROOT))

wave6 = importlib.import_module("tools.probes.mamba3_wgmma_wave6_copy_path")


DATE = "2026-04-30"
RECEIPT_NAME = "mamba3_mono_wgmma_copy_path_wave7_evidence_2026_04_30"
EXPECTED_TILE_NAMES = [
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
]
TMA_TILE_NAMES = EXPECTED_TILE_NAMES[:10]
ALIGNMENT_REQUIRED_FIELDS = (
    "vector_bytes",
    "tile_rows",
    "tile_cols",
    "dtype_bytes",
    "row_bytes",
    "tile_bytes",
    "row_tail_bytes",
    "tile_tail_bytes",
    "global_base_alignment_bytes",
    "smem_base_alignment_bytes",
    "runtime_global_alignment_guard",
    "runtime_smem_alignment_guard",
    "row_stride_alignment_bytes",
    "uses_16b_contiguous_vector_type",
    "masked_tail_path_present",
    "tiles_covered",
    "kt_qt_vector_compatible_layout",
)
PTXAS_DYNAMIC_SMEM_NOTE = (
    "ptxas -v reports registers, spills, and static shared memory. "
    "dynamic_smem_bytes must come from the kernel launch/probe receipt."
)
MINIMAL_COPY_PROBE_DYNAMIC_SMEM_BYTES = wave6.TILE_BYTES
DEFAULT_CUTLASS_INCLUDE_CANDIDATES = (
    pathlib.Path("/home/dave/cppmega-venv/lib/python3.13/site-packages/flashinfer/data/cutlass/include"),
    pathlib.Path("/home/dave/flashinfer/flashinfer/data/cutlass/include"),
    pathlib.Path("/home/dave/TransformerEngine/3rdparty/cutlass/include"),
    pathlib.Path("/home/dave/tilelang-build/3rdparty/cutlass/include"),
)


def _canonical(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_int(pattern: str, text: str, default: int = 0) -> int:
    match = re.search(pattern, text)
    if not match:
        return default
    return int(match.group(1))


def _ptxas_function_entries(text: str) -> list[dict[str, Any]]:
    """Parse function-level records from a ``ptxas --verbose`` log."""

    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        function_match = re.search(r"Function properties for\s+(.+)$", line)
        if function_match:
            current = {
                "function": function_match.group(1).strip(),
                "stack_frame_bytes": None,
                "spill_stores_bytes": None,
                "spill_loads_bytes": None,
                "registers_per_thread": None,
                "static_smem_bytes": None,
            }
            entries.append(current)
            continue

        if current is None and ("Used " in line or "spill stores" in line):
            current = {
                "function": "<unknown>",
                "stack_frame_bytes": None,
                "spill_stores_bytes": None,
                "spill_loads_bytes": None,
                "registers_per_thread": None,
                "static_smem_bytes": None,
            }
            entries.append(current)

        if current is None:
            continue

        if "stack frame" in line and "spill stores" in line and "spill loads" in line:
            current["stack_frame_bytes"] = _extract_int(r"(\d+)\s+bytes\s+stack frame", line)
            current["spill_stores_bytes"] = _extract_int(r"(\d+)\s+bytes\s+spill stores", line)
            current["spill_loads_bytes"] = _extract_int(r"(\d+)\s+bytes\s+spill loads", line)
            continue

        registers_match = re.search(r"Used\s+(\d+)\s+registers", line)
        if registers_match:
            current["registers_per_thread"] = int(registers_match.group(1))
            current["static_smem_bytes"] = _extract_int(r"(\d+)\s+bytes\s+smem", line)

    return entries


def parse_ptxas_verbose(
    text: str,
    *,
    dynamic_smem_bytes: int | None,
    kernel_name: str | None = None,
) -> dict[str, Any]:
    """Return Wave6-compatible ptxas metadata from raw verbose output."""

    entries = _ptxas_function_entries(text)
    if kernel_name:
        selected = next((entry for entry in entries if kernel_name in entry["function"]), None)
    else:
        selected = entries[0] if entries else None

    if selected is None:
        return {
            "status": "parse_failed",
            "metadata": None,
            "entries": entries,
            "failures": ["no ptxas function record with register usage was found"],
        }

    missing = [
        field
        for field in (
            "registers_per_thread",
            "static_smem_bytes",
            "spill_stores_bytes",
            "spill_loads_bytes",
        )
        if selected.get(field) is None
    ]
    if dynamic_smem_bytes is None:
        missing.append("dynamic_smem_bytes")

    if missing:
        return {
            "status": "incomplete",
            "selected_function": selected["function"],
            "metadata": None,
            "entries": entries,
            "failures": [f"missing ptxas evidence field: {field}" for field in missing],
            "dynamic_smem_note": PTXAS_DYNAMIC_SMEM_NOTE,
        }

    metadata = {
        "registers_per_thread": int(selected["registers_per_thread"]),
        "static_smem_bytes": int(selected["static_smem_bytes"]),
        "dynamic_smem_bytes": int(dynamic_smem_bytes),
        "spill_stores_bytes": int(selected["spill_stores_bytes"]),
        "spill_loads_bytes": int(selected["spill_loads_bytes"]),
    }
    return {
        "status": "parsed",
        "selected_function": selected["function"],
        "metadata": metadata,
        "entries": entries,
        "failures": [],
        "dynamic_smem_note": PTXAS_DYNAMIC_SMEM_NOTE,
    }


def _missing_ptxas_evidence() -> dict[str, Any]:
    return {
        "status": "missing_ptxas_log",
        "metadata": None,
        "failures": [
            "attach raw ptxas --verbose output with registers, static smem, spill stores, and spill loads",
            "attach dynamic_smem_bytes from the kernel launch/probe receipt",
        ],
        "required_fields": [
            "raw_ptxas_output",
            "registers_per_thread",
            "static_smem_bytes",
            "dynamic_smem_bytes",
            "spill_stores_bytes",
            "spill_loads_bytes",
        ],
        "dynamic_smem_note": PTXAS_DYNAMIC_SMEM_NOTE,
    }


def build_ptxas_evidence(
    *,
    ptxas_log_text: str | None,
    ptxas_log_path: pathlib.Path | None,
    dynamic_smem_bytes: int | None,
    kernel_name: str | None,
) -> dict[str, Any]:
    if ptxas_log_text is None:
        return _missing_ptxas_evidence()

    parsed = parse_ptxas_verbose(
        ptxas_log_text,
        dynamic_smem_bytes=dynamic_smem_bytes,
        kernel_name=kernel_name,
    )
    parsed["raw_log_sha256"] = hashlib.sha256(ptxas_log_text.encode()).hexdigest()
    parsed["raw_log_line_count"] = len(ptxas_log_text.splitlines())
    parsed["raw_log_path"] = str(ptxas_log_path) if ptxas_log_path else None
    return parsed


def evaluate_alignment_evidence(evidence: dict[str, Any] | None) -> dict[str, Any]:
    if evidence is None:
        return {
            "status": "missing_alignment_evidence",
            "pass": False,
            "evidence": None,
            "required_fields": list(ALIGNMENT_REQUIRED_FIELDS),
            "failures": [
                "attach runtime/base alignment and generated-layout evidence for the 128-bit copy attempt"
            ],
        }

    failures: list[str] = []
    normalized: dict[str, Any] = {}

    for field in ALIGNMENT_REQUIRED_FIELDS:
        if field not in evidence:
            failures.append(f"missing alignment evidence field: {field}")

    int_fields = {
        "vector_bytes",
        "tile_rows",
        "tile_cols",
        "dtype_bytes",
        "row_bytes",
        "tile_bytes",
        "row_tail_bytes",
        "tile_tail_bytes",
        "global_base_alignment_bytes",
        "smem_base_alignment_bytes",
        "row_stride_alignment_bytes",
    }
    for field in int_fields:
        value = _as_int(evidence.get(field))
        if value is None:
            failures.append(f"alignment evidence field is not an integer: {field}")
        else:
            normalized[field] = value

    bool_fields = {
        "runtime_global_alignment_guard",
        "runtime_smem_alignment_guard",
        "uses_16b_contiguous_vector_type",
        "masked_tail_path_present",
        "kt_qt_vector_compatible_layout",
    }
    for field in bool_fields:
        value = _as_bool(evidence.get(field))
        if value is None:
            failures.append(f"alignment evidence field is not a boolean: {field}")
        else:
            normalized[field] = value

    tiles_covered = evidence.get("tiles_covered")
    if isinstance(tiles_covered, list) and all(isinstance(item, str) for item in tiles_covered):
        normalized["tiles_covered"] = sorted(tiles_covered)
    else:
        failures.append("alignment evidence field is not a string list: tiles_covered")

    if not failures:
        if normalized["vector_bytes"] != wave6.VECTOR_BYTES:
            failures.append(f"vector_bytes must be 16, got {normalized['vector_bytes']}")
        if normalized["tile_rows"] != wave6.TILE_ROWS:
            failures.append(f"tile_rows must be 64, got {normalized['tile_rows']}")
        if normalized["tile_cols"] != wave6.TILE_COLS:
            failures.append(f"tile_cols must be 64, got {normalized['tile_cols']}")
        if normalized["dtype_bytes"] != wave6.BF16_BYTES:
            failures.append(f"dtype_bytes must be 2 for BF16, got {normalized['dtype_bytes']}")
        if normalized["row_bytes"] != wave6.ROW_BYTES:
            failures.append(f"row_bytes must be 128, got {normalized['row_bytes']}")
        if normalized["tile_bytes"] != wave6.TILE_BYTES:
            failures.append(f"tile_bytes must be 8192, got {normalized['tile_bytes']}")
        if normalized["row_tail_bytes"] != 0:
            failures.append(f"row_tail_bytes must be zero, got {normalized['row_tail_bytes']}")
        if normalized["tile_tail_bytes"] != 0:
            failures.append(f"tile_tail_bytes must be zero, got {normalized['tile_tail_bytes']}")
        if normalized["global_base_alignment_bytes"] < wave6.VECTOR_BYTES:
            failures.append(
                "global_base_alignment_bytes must be at least 16, "
                f"got {normalized['global_base_alignment_bytes']}"
            )
        if normalized["smem_base_alignment_bytes"] < wave6.VECTOR_BYTES:
            failures.append(
                "smem_base_alignment_bytes must be at least 16, "
                f"got {normalized['smem_base_alignment_bytes']}"
            )
        if normalized["row_stride_alignment_bytes"] % wave6.VECTOR_BYTES != 0:
            failures.append(
                "row_stride_alignment_bytes must be a multiple of 16, "
                f"got {normalized['row_stride_alignment_bytes']}"
            )
        if not normalized["runtime_global_alignment_guard"]:
            failures.append("runtime_global_alignment_guard must be true")
        if not normalized["runtime_smem_alignment_guard"]:
            failures.append("runtime_smem_alignment_guard must be true")
        if not normalized["uses_16b_contiguous_vector_type"]:
            failures.append("uses_16b_contiguous_vector_type must be true")
        if normalized["masked_tail_path_present"]:
            failures.append("masked_tail_path_present must be false")
        if not normalized["kt_qt_vector_compatible_layout"]:
            failures.append("kt_qt_vector_compatible_layout must be true")

        missing_tiles = sorted(set(EXPECTED_TILE_NAMES) - set(normalized["tiles_covered"]))
        extra_tiles = sorted(set(normalized["tiles_covered"]) - set(EXPECTED_TILE_NAMES))
        if missing_tiles:
            failures.append(f"tiles_covered is missing Wave6 tiles: {', '.join(missing_tiles)}")
        if extra_tiles:
            failures.append(f"tiles_covered has unknown tiles: {', '.join(extra_tiles)}")

    return {
        "status": "pass" if not failures else "fail",
        "pass": not failures,
        "evidence": normalized if normalized else evidence,
        "required_fields": list(ALIGNMENT_REQUIRED_FIELDS),
        "failures": failures,
    }


def tma_descriptor_smoke_checklist() -> list[dict[str, Any]]:
    return [
        {
            "id": "descriptor_scope",
            "status": "required",
            "evidence_required": "descriptor/tensor-map construction succeeds for exactly the 10 global Wave6 tiles",
            "expected_tiles": TMA_TILE_NAMES,
            "forbidden_tiles": ["dstates_panel_p0", "dstates_panel_p1", "tiny scalar/vector slices"],
        },
        {
            "id": "descriptor_shape",
            "status": "required",
            "evidence_required": "each descriptor reports rank=2, box_dim_elements=[64, 64], element_type=bf16",
            "expected_box_bytes_per_tile": wave6.TILE_BYTES,
            "expected_descriptor_bytes_per_chunk": len(TMA_TILE_NAMES) * wave6.TILE_BYTES,
        },
        {
            "id": "alignment",
            "status": "required",
            "evidence_required": "global base pointer, shared base pointer, and transfer size are 16-byte aligned for every descriptor",
            "required_alignment_bytes": wave6.VECTOR_BYTES,
        },
        {
            "id": "mbarrier_expected_bytes",
            "status": "required",
            "evidence_required": "mbarrier expected-byte accounting is programmed before each TMA wait",
            "expected_bytes_per_tile": wave6.TILE_BYTES,
            "expected_bytes_per_chunk": len(TMA_TILE_NAMES) * wave6.TILE_BYTES,
        },
        {
            "id": "wait_fence_ordering",
            "status": "required",
            "evidence_required": "generated code waits/fences on the async barrier before WGMMA consumes every TMA-fed SMEM tile",
        },
        {
            "id": "resource_gate",
            "status": "required",
            "evidence_required": "ptxas metadata for the TMA variant has <=192 registers/thread, <=131072 total SMEM bytes, and zero spills",
            "max_registers_per_thread": wave6.PASS_REGS_PER_THREAD,
            "max_total_smem_bytes": wave6.TMA_TARGET_DYNAMIC_SMEM_BYTES,
        },
    ]


def _find_cutlass_include(explicit: pathlib.Path | None) -> pathlib.Path | None:
    candidates = [explicit] if explicit else []
    candidates.extend(DEFAULT_CUTLASS_INCLUDE_CANDIDATES)
    for candidate in candidates:
        if candidate and (candidate / "cute" / "tensor.hpp").exists():
            return candidate
    return None


def compile_minimal_copy_probe(
    *,
    cuda_arch: str,
    cutlass_include: pathlib.Path | None,
) -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return {
            "status": "skipped",
            "reason": "nvcc was not found in PATH",
            "metadata": None,
        }

    include_dir = _find_cutlass_include(cutlass_include)
    if include_dir is None:
        return {
            "status": "skipped",
            "reason": "no CUTLASS/CuTe include directory with cute/tensor.hpp was found",
            "metadata": None,
        }

    source = ROOT / "tools" / "probes" / "mamba3_wgmma_wave7_copy_probe.cu"
    if not source.exists():
        return {
            "status": "skipped",
            "reason": f"probe source is missing: {source}",
            "metadata": None,
        }

    with tempfile.TemporaryDirectory(prefix="cppmega_wave7_copy_") as tmp:
        output = pathlib.Path(tmp) / "mamba3_wgmma_wave7_copy_probe.o"
        command = [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_arch}",
            f"-I{include_dir}",
            "--ptxas-options=-v",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
        result = subprocess.run(command, check=False, text=True, capture_output=True)

    ptxas_text = "\n".join(part for part in (result.stdout, result.stderr) if part)
    evidence = build_ptxas_evidence(
        ptxas_log_text=ptxas_text,
        ptxas_log_path=None,
        dynamic_smem_bytes=MINIMAL_COPY_PROBE_DYNAMIC_SMEM_BYTES,
        kernel_name="mamba3_wave7_narrow_copy_probe",
    )
    return {
        "status": "compiled" if result.returncode == 0 else "compile_failed",
        "returncode": result.returncode,
        "command": command,
        "cuda_arch": cuda_arch,
        "cutlass_include": str(include_dir),
        "dynamic_smem_bytes_from_probe_contract": MINIMAL_COPY_PROBE_DYNAMIC_SMEM_BYTES,
        "ptxas_evidence": evidence,
        "compiler_output_tail": ptxas_text.splitlines()[-40:],
    }


def build_receipt(
    *,
    ptxas_log_text: str | None = None,
    ptxas_log_path: pathlib.Path | None = None,
    dynamic_smem_bytes: int | None = None,
    kernel_name: str | None = None,
    alignment_evidence: dict[str, Any] | None = None,
    compile_probe_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if compile_probe_result and compile_probe_result.get("ptxas_evidence", {}).get("metadata"):
        ptxas_evidence = compile_probe_result["ptxas_evidence"]
    else:
        ptxas_evidence = build_ptxas_evidence(
            ptxas_log_text=ptxas_log_text,
            ptxas_log_path=ptxas_log_path,
            dynamic_smem_bytes=dynamic_smem_bytes,
            kernel_name=kernel_name,
        )

    ptxas_metadata = ptxas_evidence.get("metadata")
    ptxas_resource_check = wave6.evaluate_ptxas_metadata(
        ptxas_metadata,
        variant="narrow_vector_128b_safe_attempt",
    )
    alignment_check = evaluate_alignment_evidence(alignment_evidence)
    can_promote_narrow_vector = bool(ptxas_resource_check["pass"] and alignment_check["pass"])

    receipt = {
        "receipt": RECEIPT_NAME,
        "status": "pass_narrow_vector_evidence_ready" if can_promote_narrow_vector else "fail_missing_or_incomplete_copy_evidence",
        "date": DATE,
        "branch": "worker/mamba3-mono-triton-model",
        "source_receipt": "docs/status/mamba3_mono_wgmma_copy_path_wave6_receipt_2026_04_30.json",
        "scope": {
            "variant": "narrow_vector_128b_safe_attempt",
            "copy_attempt": "128-bit vector copy for 64x64 BF16 Wave6 tiles",
            "promotion_rule": "ptxas resource check and full Wave6 alignment/layout evidence must both pass",
            "local_compile_probe_coverage": "minimal one-tile CUDA/CuTe uint4 copy probe only; not the full monolithic WGMMA kernel",
        },
        "narrow_vector_128b_safe_attempt": {
            "status": "pass" if can_promote_narrow_vector else "fail",
            "can_promote_wave6_gate": can_promote_narrow_vector,
            "ptxas_evidence": ptxas_evidence,
            "ptxas_resource_check": ptxas_resource_check,
            "alignment_evidence_check": alignment_check,
            "required_evidence_fields": {
                "ptxas": [
                    "raw ptxas --verbose output",
                    "registers_per_thread",
                    "static_smem_bytes",
                    "dynamic_smem_bytes",
                    "spill_stores_bytes",
                    "spill_loads_bytes",
                ],
                "alignment": list(ALIGNMENT_REQUIRED_FIELDS),
            },
        },
        "compile_probe": compile_probe_result
        or {
            "status": "not_run",
            "command": (
                "python tools/probes/mamba3_wgmma_wave7_copy_evidence.py "
                "--compile-probe --cuda-arch sm_121"
            ),
        },
        "tma_descriptor_smoke_checklist": tma_descriptor_smoke_checklist(),
        "wave8_recommendation": [
            "Promote the narrow-vector path only after this receipt carries both parsed ptxas output and full Wave6 tile alignment evidence.",
            "Keep the TMA path behind the descriptor smoke checklist until descriptor scope, mbarrier bytes, wait/fence ordering, and ptxas resources are receipted.",
            "Treat a minimal one-tile compile probe as compiler/toolchain evidence, not as full monolithic-kernel timing evidence.",
        ],
        "validation": {
            "generated_by": "tools/probes/mamba3_wgmma_wave7_copy_evidence.py",
            "placeholder_check_command": "python tools/probes/mamba3_wgmma_wave7_copy_evidence.py --check docs/status/mamba3_mono_wgmma_copy_path_wave7_receipt_2026_04_30.json",
            "ingest_command": (
                "python tools/probes/mamba3_wgmma_wave7_copy_evidence.py "
                "--ptxas-log ptxas.log --dynamic-smem-bytes 118784 --alignment-evidence alignment.json"
            ),
        },
    }
    _validate_receipt(receipt)
    return receipt


def _validate_receipt(receipt: dict[str, Any]) -> None:
    narrow = receipt["narrow_vector_128b_safe_attempt"]
    assert narrow["ptxas_resource_check"]["variant"] == "narrow_vector_128b_safe_attempt"
    assert len(receipt["tma_descriptor_smoke_checklist"]) == 6
    descriptor_scope = receipt["tma_descriptor_smoke_checklist"][0]
    assert descriptor_scope["expected_tiles"] == TMA_TILE_NAMES
    descriptor_shape = receipt["tma_descriptor_smoke_checklist"][1]
    assert descriptor_shape["expected_descriptor_bytes_per_chunk"] == 81_920
    if narrow["can_promote_wave6_gate"]:
        assert receipt["status"] == "pass_narrow_vector_evidence_ready"
        assert narrow["ptxas_resource_check"]["pass"]
        assert narrow["alignment_evidence_check"]["pass"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=pathlib.Path, default=None)
    parser.add_argument("--ptxas-log", type=pathlib.Path, default=None)
    parser.add_argument("--dynamic-smem-bytes", type=int, default=None)
    parser.add_argument("--kernel-name", default=None)
    parser.add_argument("--alignment-evidence", type=pathlib.Path, default=None)
    parser.add_argument("--compile-probe", action="store_true")
    parser.add_argument("--cuda-arch", default="sm_121")
    parser.add_argument("--cutlass-include", type=pathlib.Path, default=None)
    args = parser.parse_args()

    ptxas_log_text = args.ptxas_log.read_text() if args.ptxas_log else None
    alignment_evidence = _read_json(args.alignment_evidence) if args.alignment_evidence else None
    compile_probe_result = (
        compile_minimal_copy_probe(
            cuda_arch=args.cuda_arch,
            cutlass_include=args.cutlass_include,
        )
        if args.compile_probe
        else None
    )
    receipt = build_receipt(
        ptxas_log_text=ptxas_log_text,
        ptxas_log_path=args.ptxas_log,
        dynamic_smem_bytes=args.dynamic_smem_bytes,
        kernel_name=args.kernel_name,
        alignment_evidence=alignment_evidence,
        compile_probe_result=compile_probe_result,
    )
    rendered = _canonical(receipt)

    if args.check is None:
        print(rendered, end="")
        return

    actual = args.check.read_text()
    if actual != rendered:
        raise SystemExit(f"{args.check} does not match generated receipt")


if __name__ == "__main__":
    main()
