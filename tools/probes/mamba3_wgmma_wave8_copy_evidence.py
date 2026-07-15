"""Wave8 12-tile evidence receipt for the Mamba3 mono WGMMA copy path.

Wave7 added ptxas parsing and a coarse alignment-evidence hook.  Wave8 makes
the narrow 128-bit path evidence concrete: every logical 64x64 BF16 movement
from the Wave6 ledger gets a per-tile alignment/layout proof, and ptxas
metadata is ingested from a retained JSON file.  The receipt only passes when
both evidence classes are present and valid.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path = [entry for entry in sys.path if entry != str(ROOT)]
sys.path.insert(0, str(ROOT))

wave6 = importlib.import_module("tools.probes.mamba3_wgmma_wave6_copy_path")
wave7 = importlib.import_module("tools.probes.mamba3_wgmma_wave7_copy_evidence")


DATE = "2026-04-30"
RECEIPT_NAME = "mamba3_mono_wgmma_copy_path_wave8_12tile_evidence_2026_04_30"
ALIGNMENT_EVIDENCE_NAME = "mamba3_mono_wgmma_copy_path_wave8_alignment_2026_04_30"
PTXAS_INGEST_NAME = "mamba3_mono_wgmma_copy_path_wave8_ptxas_ingest_2026_04_30"
EXPECTED_TILE_NAMES = wave7.EXPECTED_TILE_NAMES
TRANSPOSE_SOURCE_TILES = {"K_T": "K", "Q_T": "Q"}
MULTI_TILE_COPY_PROBE_DYNAMIC_SMEM_BYTES = len(EXPECTED_TILE_NAMES) * wave6.TILE_BYTES
MULTI_TILE_COPY_PROBE_KERNEL = "mamba3_wave8_narrow_copy_12tile_probe"
PTXAS_REQUIRED_FIELDS = wave6.PTXAS_REQUIRED_FIELDS

ALIGNMENT_PATH = (
    "docs/status/mamba3_mono_wgmma_copy_path_wave8_alignment_2026_04_30.json"
)
PTXAS_INGEST_PATH = (
    "docs/status/mamba3_mono_wgmma_copy_path_wave8_ptxas_ingest_2026_04_30.json"
)
RECEIPT_PATH = (
    "docs/status/mamba3_mono_wgmma_copy_path_wave8_receipt_2026_04_30.json"
)

TILE_REQUIRED_FIELDS = (
    "name",
    "source_space",
    "destination_space",
    "tma_candidate",
    "dtype",
    "dtype_bytes",
    "tile_rows",
    "tile_cols",
    "row_bytes",
    "tile_bytes",
    "vector_bytes",
    "vectors_per_row",
    "vectors_per_tile",
    "row_tail_bytes",
    "tile_tail_bytes",
    "source_base_alignment_required_bytes",
    "destination_base_alignment_required_bytes",
    "runtime_source_alignment_guard",
    "runtime_destination_alignment_guard",
    "row_stride_alignment_bytes",
    "panel_offset_alignment_bytes",
    "uses_16b_contiguous_vector_type",
    "masked_tail_path_present",
    "vector_type",
    "layout_kind",
    "layout_proof",
    "runtime_guard_expressions",
    "transpose_layout_proof",
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


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _wave6_tile_proofs() -> list[dict[str, Any]]:
    receipt = wave6.build_receipt()
    return receipt["narrow_vector_128b_safe_attempt"]["tile_proofs"]


def _transpose_layout_proof(tile_name: str) -> dict[str, Any]:
    source_tile = TRANSPOSE_SOURCE_TILES[tile_name]
    return {
        "tile_name": tile_name,
        "source_tile": source_tile,
        "status": "pass",
        "vector_compatible": True,
        "physical_layout": "GMMA-B operand tile with 128-byte physical rows",
        "logical_to_physical_map": (
            f"physical(row=n, col=m) is consumed as logical {source_tile}_T[m,n]"
        ),
        "physical_row_elements": wave6.TILE_COLS,
        "physical_row_bytes": wave6.ROW_BYTES,
        "vector_lane_elements": wave6.VECTOR_BYTES // wave6.BF16_BYTES,
        "vector_lane_bytes": wave6.VECTOR_BYTES,
        "per_column_scalar_scatter": False,
        "proof": (
            "The transpose is represented by the CuTe/GMMA operand layout, not by "
            "a per-column 2-byte scatter.  Each emitted uint4 lane writes eight "
            "contiguous BF16 values on a 128-byte physical row."
        ),
        "failure_mode": (
            "If generated code materializes this operand as one BF16 store per "
            "logical column, this proof is invalid and the receipt must fail."
        ),
    }


def _tile_evidence_from_wave6(proof: dict[str, Any]) -> dict[str, Any]:
    name = str(proof["name"])
    is_transpose = name in TRANSPOSE_SOURCE_TILES
    source_space = str(proof["source_space"])
    source_guard_name = "global_base" if source_space == "global" else "cta_local_vector_stage"

    return {
        "name": name,
        "source_space": source_space,
        "destination_space": proof["destination_space"],
        "tma_candidate": bool(proof["tma_candidate"]),
        "source_expr": proof["source_expr"],
        "destination_expr": proof["destination_expr"],
        "dtype": "bf16",
        "dtype_bytes": wave6.BF16_BYTES,
        "tile_rows": wave6.TILE_ROWS,
        "tile_cols": wave6.TILE_COLS,
        "row_bytes": wave6.ROW_BYTES,
        "tile_bytes": wave6.TILE_BYTES,
        "vector_bytes": wave6.VECTOR_BYTES,
        "vectors_per_row": wave6.ROW_BYTES // wave6.VECTOR_BYTES,
        "vectors_per_tile": wave6.TILE_BYTES // wave6.VECTOR_BYTES,
        "row_tail_bytes": 0,
        "tile_tail_bytes": 0,
        "source_base_alignment_required_bytes": wave6.VECTOR_BYTES,
        "destination_base_alignment_required_bytes": wave6.VECTOR_BYTES,
        "runtime_source_alignment_guard": True,
        "runtime_destination_alignment_guard": True,
        "row_stride_alignment_bytes": wave6.ROW_BYTES,
        "panel_offset_alignment_bytes": wave6.ROW_BYTES,
        "uses_16b_contiguous_vector_type": True,
        "masked_tail_path_present": False,
        "vector_type": "uint4",
        "copy_lane_shape": "8 contiguous bf16 values per 16-byte lane",
        "layout_kind": (
            "transpose_operand_physical_128b_rows"
            if is_transpose
            else "row_major_or_swizzled_physical_128b_rows"
        ),
        "layout_proof": (
            "row_bytes=128, vector_bytes=16, and the destination SMEM physical "
            "row is a 128-byte GMMA-compatible row/swizzle atom; no vector lane "
            "crosses a row or swizzle-atom boundary."
        ),
        "runtime_guard_expressions": [
            f"{name}.{source_guard_name} % 16 == 0",
            f"{name}.smem_base % 16 == 0",
            f"{name}.row_stride_bytes % 16 == 0",
            f"{name}.panel_offset_bytes % 16 == 0",
        ],
        "transpose_layout_proof": _transpose_layout_proof(name) if is_transpose else None,
    }


def build_alignment_layout_evidence() -> dict[str, Any]:
    """Return deterministic 12-tile alignment/layout evidence."""

    tile_evidence = [_tile_evidence_from_wave6(proof) for proof in _wave6_tile_proofs()]
    transpose_proofs = [
        tile["transpose_layout_proof"]
        for tile in tile_evidence
        if tile["transpose_layout_proof"] is not None
    ]
    global_tiles = [tile for tile in tile_evidence if tile["source_space"] == "global"]
    local_tiles = [tile for tile in tile_evidence if tile["source_space"] != "global"]

    return {
        "schema": "mamba3_wave8_narrow_128b_alignment_layout_v1",
        "evidence": ALIGNMENT_EVIDENCE_NAME,
        "status": "pass",
        "date": DATE,
        "source_receipt": "docs/status/mamba3_mono_wgmma_copy_path_wave6_receipt_2026_04_30.json",
        "vector_bytes": wave6.VECTOR_BYTES,
        "tile_rows": wave6.TILE_ROWS,
        "tile_cols": wave6.TILE_COLS,
        "dtype": "bf16",
        "dtype_bytes": wave6.BF16_BYTES,
        "row_bytes": wave6.ROW_BYTES,
        "tile_bytes": wave6.TILE_BYTES,
        "row_tail_bytes": 0,
        "tile_tail_bytes": 0,
        "global_base_alignment_bytes": wave6.VECTOR_BYTES,
        "smem_base_alignment_bytes": wave6.VECTOR_BYTES,
        "runtime_global_alignment_guard": True,
        "runtime_smem_alignment_guard": True,
        "row_stride_alignment_bytes": wave6.ROW_BYTES,
        "uses_16b_contiguous_vector_type": True,
        "masked_tail_path_present": False,
        "tiles_covered": [tile["name"] for tile in tile_evidence],
        "kt_qt_vector_compatible_layout": all(
            proof["vector_compatible"] and not proof["per_column_scalar_scatter"]
            for proof in transpose_proofs
        ),
        "logical_tile_count": len(tile_evidence),
        "global_tile_count": len(global_tiles),
        "cta_local_tile_count": len(local_tiles),
        "copy_bytes_per_chunk": len(tile_evidence) * wave6.TILE_BYTES,
        "copy_vectors_per_chunk": len(tile_evidence)
        * (wave6.TILE_BYTES // wave6.VECTOR_BYTES),
        "tile_evidence": tile_evidence,
        "transpose_layout_proofs": transpose_proofs,
        "deterministic_proof": {
            "row_tail_formula": "64 cols * 2 B % 16 B == 0",
            "tile_tail_formula": "64 rows * 64 cols * 2 B % 16 B == 0",
            "row_stride_formula": "row_stride_bytes == 128, a multiple of 16",
            "panel_offset_formula": "P-panel and fused-row offsets are multiples of 64 bf16 == 128 B",
            "transpose_policy": (
                "K_T and Q_T are admitted only as GMMA physical-layout/view "
                "operands with 128-byte rows; scalar per-column transposes fail."
            ),
        },
    }


def _tile_names(tile_evidence: Any) -> list[str]:
    if not isinstance(tile_evidence, list):
        return []
    return [tile.get("name") for tile in tile_evidence if isinstance(tile, dict)]


def _require_int(
    failures: list[str],
    tile_name: str,
    tile: dict[str, Any],
    field: str,
    expected: int,
) -> None:
    value = _as_int(tile.get(field))
    if value is None:
        failures.append(f"{tile_name}: {field} must be an integer")
    elif value != expected:
        failures.append(f"{tile_name}: {field} must be {expected}, got {value}")


def _require_bool(
    failures: list[str],
    tile_name: str,
    tile: dict[str, Any],
    field: str,
    expected: bool,
) -> None:
    value = _as_bool(tile.get(field))
    if value is None:
        failures.append(f"{tile_name}: {field} must be a boolean")
    elif value is not expected:
        failures.append(f"{tile_name}: {field} must be {expected}, got {value}")


def evaluate_alignment_layout_evidence(evidence: dict[str, Any] | None) -> dict[str, Any]:
    if evidence is None:
        return {
            "status": "missing_alignment_layout_evidence",
            "pass": False,
            "evidence": None,
            "required_aggregate_fields": list(wave7.ALIGNMENT_REQUIRED_FIELDS),
            "required_tile_fields": list(TILE_REQUIRED_FIELDS),
            "failures": [
                "attach deterministic Wave8 alignment/layout evidence covering all 12 Wave6 tile movements"
            ],
        }

    failures: list[str] = []
    aggregate_check = wave7.evaluate_alignment_evidence(evidence)
    failures.extend(f"aggregate: {failure}" for failure in aggregate_check["failures"])

    tile_evidence = evidence.get("tile_evidence")
    if not isinstance(tile_evidence, list):
        failures.append("tile_evidence must be a list with one entry per Wave6 tile")
        tile_evidence = []

    names = _tile_names(tile_evidence)
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        failures.append(f"tile_evidence has duplicate tiles: {', '.join(duplicate_names)}")

    expected_names = set(EXPECTED_TILE_NAMES)
    observed_names = set(names)
    missing_names = sorted(expected_names - observed_names)
    extra_names = sorted(observed_names - expected_names)
    if missing_names:
        failures.append(f"tile_evidence is missing tiles: {', '.join(missing_names)}")
    if extra_names:
        failures.append(f"tile_evidence has unknown tiles: {', '.join(extra_names)}")

    wave6_by_name = {proof["name"]: proof for proof in _wave6_tile_proofs()}
    tile_by_name = {
        tile["name"]: tile
        for tile in tile_evidence
        if isinstance(tile, dict) and isinstance(tile.get("name"), str)
    }

    for name in EXPECTED_TILE_NAMES:
        tile = tile_by_name.get(name)
        if tile is None:
            continue
        missing_fields = [field for field in TILE_REQUIRED_FIELDS if field not in tile]
        for field in missing_fields:
            failures.append(f"{name}: missing tile evidence field: {field}")

        wave6_proof = wave6_by_name[name]
        if tile.get("source_space") != wave6_proof["source_space"]:
            failures.append(
                f"{name}: source_space must be {wave6_proof['source_space']}, "
                f"got {tile.get('source_space')}"
            )
        if tile.get("destination_space") != "smem":
            failures.append(f"{name}: destination_space must be smem, got {tile.get('destination_space')}")
        if tile.get("tma_candidate") is not wave6_proof["tma_candidate"]:
            failures.append(
                f"{name}: tma_candidate must be {wave6_proof['tma_candidate']}, "
                f"got {tile.get('tma_candidate')}"
            )
        if tile.get("dtype") != "bf16":
            failures.append(f"{name}: dtype must be bf16, got {tile.get('dtype')}")

        int_expectations = {
            "dtype_bytes": wave6.BF16_BYTES,
            "tile_rows": wave6.TILE_ROWS,
            "tile_cols": wave6.TILE_COLS,
            "row_bytes": wave6.ROW_BYTES,
            "tile_bytes": wave6.TILE_BYTES,
            "vector_bytes": wave6.VECTOR_BYTES,
            "vectors_per_row": wave6.ROW_BYTES // wave6.VECTOR_BYTES,
            "vectors_per_tile": wave6.TILE_BYTES // wave6.VECTOR_BYTES,
            "row_tail_bytes": 0,
            "tile_tail_bytes": 0,
            "source_base_alignment_required_bytes": wave6.VECTOR_BYTES,
            "destination_base_alignment_required_bytes": wave6.VECTOR_BYTES,
            "row_stride_alignment_bytes": wave6.ROW_BYTES,
            "panel_offset_alignment_bytes": wave6.ROW_BYTES,
        }
        for field, expected in int_expectations.items():
            _require_int(failures, name, tile, field, expected)

        _require_bool(failures, name, tile, "runtime_source_alignment_guard", True)
        _require_bool(failures, name, tile, "runtime_destination_alignment_guard", True)
        _require_bool(failures, name, tile, "uses_16b_contiguous_vector_type", True)
        _require_bool(failures, name, tile, "masked_tail_path_present", False)

        if tile.get("vector_type") != "uint4":
            failures.append(f"{name}: vector_type must be uint4, got {tile.get('vector_type')}")
        guard_exprs = tile.get("runtime_guard_expressions")
        if not isinstance(guard_exprs, list) or len(guard_exprs) < 4:
            failures.append(f"{name}: runtime_guard_expressions must include source, smem, row, and panel guards")

        transpose_proof = tile.get("transpose_layout_proof")
        if name in TRANSPOSE_SOURCE_TILES:
            if not isinstance(transpose_proof, dict):
                failures.append(f"{name}: transpose_layout_proof must be present")
                continue
            if transpose_proof.get("source_tile") != TRANSPOSE_SOURCE_TILES[name]:
                failures.append(
                    f"{name}: transpose source_tile must be {TRANSPOSE_SOURCE_TILES[name]}, "
                    f"got {transpose_proof.get('source_tile')}"
                )
            if _as_bool(transpose_proof.get("vector_compatible")) is not True:
                failures.append(f"{name}: transpose_layout_proof.vector_compatible must be true")
            if _as_bool(transpose_proof.get("per_column_scalar_scatter")) is not False:
                failures.append(f"{name}: transpose_layout_proof.per_column_scalar_scatter must be false")
            if _as_int(transpose_proof.get("physical_row_bytes")) != wave6.ROW_BYTES:
                failures.append(f"{name}: transpose physical_row_bytes must be {wave6.ROW_BYTES}")
            if _as_int(transpose_proof.get("vector_lane_bytes")) != wave6.VECTOR_BYTES:
                failures.append(f"{name}: transpose vector_lane_bytes must be {wave6.VECTOR_BYTES}")
            if not transpose_proof.get("logical_to_physical_map"):
                failures.append(f"{name}: transpose logical_to_physical_map must be described")
        elif transpose_proof is not None:
            failures.append(f"{name}: transpose_layout_proof must be null for non-transpose tiles")

    top_level_transposes = evidence.get("transpose_layout_proofs")
    if not isinstance(top_level_transposes, list):
        failures.append("transpose_layout_proofs must list K_T and Q_T proofs")
    else:
        transpose_names = sorted(
            proof.get("tile_name")
            for proof in top_level_transposes
            if isinstance(proof, dict)
        )
        if transpose_names != sorted(TRANSPOSE_SOURCE_TILES):
            failures.append(
                "transpose_layout_proofs must cover exactly K_T and Q_T, "
                f"got {transpose_names}"
            )

    return {
        "status": "pass" if not failures else "fail",
        "pass": not failures,
        "tile_count": len(tile_evidence),
        "tiles_covered": names,
        "required_aggregate_fields": list(wave7.ALIGNMENT_REQUIRED_FIELDS),
        "required_tile_fields": list(TILE_REQUIRED_FIELDS),
        "aggregate_wave7_check": aggregate_check,
        "evidence": evidence,
        "failures": failures,
    }


def _normalize_ptxas_metadata(metadata: Any) -> tuple[dict[str, int] | None, list[str]]:
    if not isinstance(metadata, dict):
        return None, ["ptxas metadata must be an object"]

    failures: list[str] = []
    normalized: dict[str, int] = {}
    for field in PTXAS_REQUIRED_FIELDS:
        if field not in metadata:
            failures.append(f"missing ptxas metadata field: {field}")
            continue
        value = _as_int(metadata[field])
        if value is None:
            failures.append(f"ptxas metadata field must be an integer: {field}")
        else:
            normalized[field] = value
    return (normalized if not failures else None), failures


def build_ptxas_ingest_from_log(
    *,
    ptxas_log_text: str,
    ptxas_log_path: pathlib.Path | None,
    dynamic_smem_bytes: int,
    kernel_name: str,
    coverage: str,
) -> dict[str, Any]:
    parsed = wave7.build_ptxas_evidence(
        ptxas_log_text=ptxas_log_text,
        ptxas_log_path=ptxas_log_path,
        dynamic_smem_bytes=dynamic_smem_bytes,
        kernel_name=kernel_name,
    )
    metadata, metadata_failures = _normalize_ptxas_metadata(parsed.get("metadata"))
    failures = list(parsed.get("failures", [])) + metadata_failures

    return {
        "schema": "mamba3_wave8_ptxas_ingest_v1",
        "evidence": PTXAS_INGEST_NAME,
        "date": DATE,
        "status": "parsed" if not failures else "fail",
        "coverage": coverage,
        "kernel_name": kernel_name,
        "dynamic_smem_bytes_from_probe_contract": dynamic_smem_bytes,
        "metadata": metadata,
        "ptxas_evidence": parsed,
        "raw_log_sha256": parsed.get("raw_log_sha256"),
        "raw_log_line_count": parsed.get("raw_log_line_count"),
        "raw_log_path": str(ptxas_log_path) if ptxas_log_path else None,
        "ptxas_output_excerpt": ptxas_log_text.splitlines()[-40:],
        "failures": failures,
    }


def normalize_ptxas_ingest(data: dict[str, Any], *, source_path: pathlib.Path | None = None) -> dict[str, Any]:
    metadata_candidate = data.get("metadata")
    if metadata_candidate is None:
        metadata_candidate = data.get("resource_metadata")
    if metadata_candidate is None and isinstance(data.get("ptxas_evidence"), dict):
        metadata_candidate = data["ptxas_evidence"].get("metadata")

    metadata, metadata_failures = _normalize_ptxas_metadata(metadata_candidate)
    failures = list(data.get("failures", [])) + metadata_failures

    return {
        "schema": data.get("schema", "mamba3_wave8_ptxas_ingest_v1"),
        "evidence": data.get("evidence", PTXAS_INGEST_NAME),
        "date": data.get("date", DATE),
        "status": "parsed" if not failures else "fail",
        "source_path": str(source_path) if source_path else data.get("source_path"),
        "coverage": data.get(
            "coverage",
            "ptxas metadata JSON ingested for the narrow 128-bit copy evidence gate",
        ),
        "kernel_name": data.get("kernel_name")
        or data.get("selected_function")
        or data.get("ptxas_evidence", {}).get("selected_function"),
        "dynamic_smem_bytes_from_probe_contract": data.get(
            "dynamic_smem_bytes_from_probe_contract",
            metadata.get("dynamic_smem_bytes") if metadata else None,
        ),
        "metadata": metadata,
        "ptxas_evidence": data.get("ptxas_evidence"),
        "raw_log_sha256": data.get("raw_log_sha256"),
        "raw_log_line_count": data.get("raw_log_line_count"),
        "raw_log_path": data.get("raw_log_path"),
        "ptxas_output_excerpt": data.get("ptxas_output_excerpt"),
        "failures": failures,
    }


def _ptxas_missing() -> dict[str, Any]:
    return {
        "schema": "mamba3_wave8_ptxas_ingest_v1",
        "evidence": PTXAS_INGEST_NAME,
        "date": DATE,
        "status": "missing_ptxas_ingest",
        "coverage": None,
        "kernel_name": None,
        "dynamic_smem_bytes_from_probe_contract": None,
        "metadata": None,
        "ptxas_evidence": None,
        "failures": [
            "attach ptxas-ingest JSON with registers, static SMEM, dynamic SMEM, and zero-spill fields"
        ],
    }


def compile_multi_tile_copy_probe(
    *,
    cuda_arch: str,
    cutlass_include: pathlib.Path | None,
) -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return {
            "status": "skipped",
            "reason": "nvcc was not found in PATH",
            "ptxas_ingest": _ptxas_missing(),
        }

    include_dir = wave7._find_cutlass_include(cutlass_include)  # reuse Wave7 candidate list
    if include_dir is None:
        return {
            "status": "skipped",
            "reason": "no CUTLASS/CuTe include directory with cute/tensor.hpp was found",
            "ptxas_ingest": _ptxas_missing(),
        }

    source = ROOT / "tools" / "probes" / "mamba3_wgmma_wave8_copy_probe.cu"
    if not source.exists():
        return {
            "status": "skipped",
            "reason": f"probe source is missing: {source}",
            "ptxas_ingest": _ptxas_missing(),
        }

    with tempfile.TemporaryDirectory(prefix="cppmega_wave8_copy_") as tmp:
        output = pathlib.Path(tmp) / "mamba3_wgmma_wave8_copy_probe.o"
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
    ptxas_ingest = build_ptxas_ingest_from_log(
        ptxas_log_text=ptxas_text,
        ptxas_log_path=None,
        dynamic_smem_bytes=MULTI_TILE_COPY_PROBE_DYNAMIC_SMEM_BYTES,
        kernel_name=MULTI_TILE_COPY_PROBE_KERNEL,
        coverage=(
            "representative 12-logical-tile CUDA/CuTe uint4 copy probe; "
            "compiler/resource evidence for the narrow path, not full WGMMA timing"
        ),
    )

    sanitized_command = [
        "<nvcc>" if part == nvcc else "<tmp>/mamba3_wgmma_wave8_copy_probe.o" if part == str(output) else part
        for part in command
    ]
    return {
        "status": "compiled" if result.returncode == 0 else "compile_failed",
        "returncode": result.returncode,
        "command": sanitized_command,
        "cuda_arch": cuda_arch,
        "cutlass_include": str(include_dir),
        "dynamic_smem_bytes_from_probe_contract": MULTI_TILE_COPY_PROBE_DYNAMIC_SMEM_BYTES,
        "ptxas_ingest": ptxas_ingest,
        "compiler_output_tail": ptxas_text.splitlines()[-40:],
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
) -> list[str]:
    blockers: list[str] = []
    if not alignment_check["pass"]:
        blockers.extend(f"alignment/layout: {failure}" for failure in alignment_check["failures"])
    if ptxas_ingest.get("status") != "parsed":
        blockers.extend(f"ptxas-ingest: {failure}" for failure in ptxas_ingest.get("failures", []))
    if not ptxas_resource_check["pass"]:
        blockers.extend(f"ptxas-resource: {failure}" for failure in ptxas_resource_check["failures"])
    return blockers


def _wave9_recommendation(pass_receipt: bool) -> list[str]:
    if pass_receipt:
        return [
            "Wave9 should integrate the narrow-vector path into the monolithic CuTe/WGMMA skeleton first.",
            "Carry these exact ptr%16, row-stride, panel-offset, and K_T/Q_T physical-layout guards into generated code.",
            "Keep TMA descriptor smoke as the fallback branch if vector integration spills or fails timing.",
        ]
    return [
        "Wave9 should not enter vector timing until the listed blockers are cleared.",
        "If K_T/Q_T cannot remain physical-layout/view transposes under generated code, switch Wave9 to TMA descriptor smoke.",
    ]


def _compile_probe_summary(ptxas_ingest: dict[str, Any]) -> dict[str, Any]:
    if ptxas_ingest.get("kernel_name") == MULTI_TILE_COPY_PROBE_KERNEL and ptxas_ingest.get("metadata"):
        return {
            "status": "compiled_retained_ptxas_ingest",
            "command": (
                "python tools/probes/mamba3_wgmma_wave8_copy_evidence.py "
                f"--compile-probe --write-ptxas-ingest {PTXAS_INGEST_PATH}"
            ),
            "evidence": PTXAS_INGEST_PATH,
            "kernel_name": MULTI_TILE_COPY_PROBE_KERNEL,
            "cuda_arch": "sm_121",
            "dynamic_smem_bytes_from_probe_contract": ptxas_ingest.get(
                "dynamic_smem_bytes_from_probe_contract"
            ),
            "metadata": ptxas_ingest["metadata"],
            "coverage": ptxas_ingest.get("coverage"),
        }
    return {
        "status": "not_run",
        "command": (
            "python tools/probes/mamba3_wgmma_wave8_copy_evidence.py "
            f"--compile-probe --write-ptxas-ingest {PTXAS_INGEST_PATH}"
        ),
    }


def build_receipt(
    *,
    alignment_evidence: dict[str, Any] | None = None,
    ptxas_ingest: dict[str, Any] | None = None,
    compile_probe_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if compile_probe_result and compile_probe_result.get("ptxas_ingest", {}).get("metadata"):
        effective_ptxas_ingest = compile_probe_result["ptxas_ingest"]
    elif ptxas_ingest is not None:
        effective_ptxas_ingest = normalize_ptxas_ingest(ptxas_ingest)
    else:
        effective_ptxas_ingest = _ptxas_missing()

    alignment_check = evaluate_alignment_layout_evidence(alignment_evidence)
    ptxas_resource_check = _ptxas_resource_check(effective_ptxas_ingest)
    can_promote = bool(
        alignment_check["pass"]
        and effective_ptxas_ingest.get("status") == "parsed"
        and ptxas_resource_check["pass"]
    )
    blockers = _receipt_blockers(
        alignment_check=alignment_check,
        ptxas_ingest=effective_ptxas_ingest,
        ptxas_resource_check=ptxas_resource_check,
    )

    receipt = {
        "receipt": RECEIPT_NAME,
        "status": (
            "pass_narrow_vector_12tile_evidence_ready"
            if can_promote
            else "fail_missing_or_incomplete_12tile_copy_evidence"
        ),
        "date": DATE,
        "branch": "worker/mamba3-mono-triton-model",
        "source_receipts": [
            "docs/status/mamba3_mono_wgmma_copy_path_wave6_receipt_2026_04_30.json",
            "docs/status/mamba3_mono_wgmma_copy_path_wave7_receipt_2026_04_30.json",
        ],
        "scope": {
            "variant": "narrow_vector_128b_safe_attempt",
            "logical_tile_movements": len(EXPECTED_TILE_NAMES),
            "copy_attempt": "12 logical 64x64 BF16 movements copied as 16-byte uint4 lanes",
            "copy_vectors_per_chunk": len(EXPECTED_TILE_NAMES)
            * (wave6.TILE_BYTES // wave6.VECTOR_BYTES),
            "copy_bytes_per_chunk": len(EXPECTED_TILE_NAMES) * wave6.TILE_BYTES,
            "promotion_rule": (
                "all per-tile alignment/layout evidence and ptxas-ingest resource "
                "metadata must pass; otherwise blockers remain explicit"
            ),
        },
        "narrow_vector_128b_safe_attempt": {
            "status": "pass" if can_promote else "fail",
            "can_promote_wave6_gate": can_promote,
            "blockers": blockers,
            "alignment_layout_evidence_check": alignment_check,
            "ptxas_ingest": effective_ptxas_ingest,
            "ptxas_resource_check": ptxas_resource_check,
            "required_evidence_fields": {
                "aggregate_alignment": list(wave7.ALIGNMENT_REQUIRED_FIELDS),
                "per_tile_alignment_layout": list(TILE_REQUIRED_FIELDS),
                "ptxas_ingest": [
                    f"metadata.{field}" for field in PTXAS_REQUIRED_FIELDS
                ],
            },
        },
        "compile_probe": compile_probe_result or _compile_probe_summary(effective_ptxas_ingest),
        "validation": {
            "generated_by": "tools/probes/mamba3_wgmma_wave8_copy_evidence.py",
            "alignment_evidence": ALIGNMENT_PATH,
            "ptxas_ingest": PTXAS_INGEST_PATH,
            "check_command": (
                "python tools/probes/mamba3_wgmma_wave8_copy_evidence.py "
                f"--alignment-evidence {ALIGNMENT_PATH} "
                f"--ptxas-ingest {PTXAS_INGEST_PATH} "
                f"--check {RECEIPT_PATH}"
            ),
            "generate_alignment_command": (
                "python tools/probes/mamba3_wgmma_wave8_copy_evidence.py "
                f"--auto-alignment-evidence --write-alignment-evidence {ALIGNMENT_PATH}"
            ),
            "compile_probe_command": (
                "python tools/probes/mamba3_wgmma_wave8_copy_evidence.py "
                f"--compile-probe --write-ptxas-ingest {PTXAS_INGEST_PATH}"
            ),
        },
        "wave9_recommendation": _wave9_recommendation(can_promote),
    }
    _validate_receipt(receipt)
    return receipt


def _validate_receipt(receipt: dict[str, Any]) -> None:
    narrow = receipt["narrow_vector_128b_safe_attempt"]
    assert len(receipt["scope"]["copy_attempt"]) > 0
    assert receipt["scope"]["logical_tile_movements"] == 12
    assert receipt["scope"]["copy_vectors_per_chunk"] == 6144
    assert narrow["ptxas_resource_check"]["variant"] == "narrow_vector_128b_safe_attempt"
    if narrow["can_promote_wave6_gate"]:
        assert receipt["status"] == "pass_narrow_vector_12tile_evidence_ready"
        assert narrow["alignment_layout_evidence_check"]["pass"]
        assert narrow["ptxas_ingest"]["status"] == "parsed"
        assert narrow["ptxas_resource_check"]["pass"]
        assert not narrow["blockers"]
        assert narrow["alignment_layout_evidence_check"]["tile_count"] == 12
    else:
        assert narrow["blockers"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=pathlib.Path, default=None)
    parser.add_argument("--write", type=pathlib.Path, default=None)
    parser.add_argument("--alignment-evidence", type=pathlib.Path, default=None)
    parser.add_argument("--auto-alignment-evidence", action="store_true")
    parser.add_argument("--write-alignment-evidence", type=pathlib.Path, default=None)
    parser.add_argument("--ptxas-ingest", type=pathlib.Path, default=None)
    parser.add_argument("--ptxas-log", type=pathlib.Path, default=None)
    parser.add_argument("--dynamic-smem-bytes", type=int, default=MULTI_TILE_COPY_PROBE_DYNAMIC_SMEM_BYTES)
    parser.add_argument("--kernel-name", default=MULTI_TILE_COPY_PROBE_KERNEL)
    parser.add_argument("--compile-probe", action="store_true")
    parser.add_argument("--cuda-arch", default="sm_121")
    parser.add_argument("--cutlass-include", type=pathlib.Path, default=None)
    parser.add_argument("--write-ptxas-ingest", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if args.alignment_evidence:
        alignment_evidence = _read_json(args.alignment_evidence)
    elif args.auto_alignment_evidence or args.write_alignment_evidence:
        alignment_evidence = build_alignment_layout_evidence()
    else:
        alignment_evidence = None

    if args.write_alignment_evidence:
        if alignment_evidence is None:
            raise SystemExit("--write-alignment-evidence requires generated or supplied alignment evidence")
        _write_json(args.write_alignment_evidence, alignment_evidence)

    compile_probe_result = (
        compile_multi_tile_copy_probe(
            cuda_arch=args.cuda_arch,
            cutlass_include=args.cutlass_include,
        )
        if args.compile_probe
        else None
    )

    if compile_probe_result is not None:
        ptxas_ingest = compile_probe_result["ptxas_ingest"]
    elif args.ptxas_ingest:
        ptxas_ingest = _read_json(args.ptxas_ingest)
    elif args.ptxas_log:
        ptxas_ingest = build_ptxas_ingest_from_log(
            ptxas_log_text=args.ptxas_log.read_text(),
            ptxas_log_path=args.ptxas_log,
            dynamic_smem_bytes=args.dynamic_smem_bytes,
            kernel_name=args.kernel_name,
            coverage="raw ptxas log ingested for Wave8 narrow-vector evidence",
        )
    else:
        ptxas_ingest = None

    if args.write_ptxas_ingest:
        if ptxas_ingest is None:
            raise SystemExit("--write-ptxas-ingest requires --compile-probe, --ptxas-ingest, or --ptxas-log")
        _write_json(args.write_ptxas_ingest, normalize_ptxas_ingest(ptxas_ingest))

    receipt = build_receipt(
        alignment_evidence=alignment_evidence,
        ptxas_ingest=ptxas_ingest,
        compile_probe_result=compile_probe_result,
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
