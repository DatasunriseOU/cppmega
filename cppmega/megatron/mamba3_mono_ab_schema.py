"""Shared schema for Mamba3 monolithic bwd_bwd production A/B reports."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "mamba3-mono-ab/v1"
MAIN_GUARDED_STAGE2_COMMIT = "bc8c3f9"
MAIN_GUARDED_STAGE2_DOC = "docs/status/mamba3_stage2_prod_control_2026_04_29.md"
MAIN_GUARDED_STAGE2_MODULE = (
    "cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches"
)

DTYPE_BYTES = {
    "bf16": 2,
    "fp32": 4,
}


@dataclass(frozen=True)
class Shape:
    name: str
    B: int
    S: int
    H: int
    G: int
    N: int
    P: int
    R: int
    chunk: int = 16
    rotary_dim_divisor: int = 4

    @property
    def nchunks(self) -> int:
        return math.ceil(self.S / self.chunk)

    @property
    def rotary_dim(self) -> int:
        return self.N // self.rotary_dim_divisor

    def to_dict(self) -> dict[str, int | str]:
        out = asdict(self)
        out["nchunks"] = self.nchunks
        out["rotary_dim"] = self.rotary_dim
        return out


SHAPES: dict[str, Shape] = {
    "smoke": Shape("smoke", B=1, S=256, H=4, G=1, N=64, P=64, R=4),
    "representative": Shape("representative", B=2, S=1024, H=16, G=1, N=64, P=64, R=4),
    "productionish": Shape("productionish", B=4, S=4096, H=32, G=1, N=64, P=128, R=4),
}


@dataclass(frozen=True)
class BoundarySlotSpec:
    name: str
    role: str
    dtype: str
    shape_expr: tuple[str, ...]
    producer: str
    required_for_monolithic_ab: bool = True
    note: str = ""


BWD_FWD_HANDOFF_SPECS: tuple[BoundarySlotSpec, ...] = (
    BoundarySlotSpec(
        "dmimo_o",
        "bwd_fwd_output",
        "fp32",
        ("B", "H", "R", "P"),
        "mamba_mimo_bwd_fwd",
        False,
        "Tracked as a chain correctness output; it is not written by bwd_bwd.",
    ),
    BoundarySlotSpec(
        "states",
        "bwd_fwd_to_bwd_bwd_handoff",
        "bf16",
        ("B", "H", "nchunks", "N", "P"),
        "mamba_mimo_bwd_fwd",
        False,
        "Input cache consumed by bwd_bwd.",
    ),
    BoundarySlotSpec(
        "qk_dot",
        "bwd_fwd_to_bwd_bwd_handoff",
        "bf16",
        ("B", "H", "S", "R*R"),
        "mamba_mimo_bwd_fwd",
        False,
        "Canonical report shape is flattened R*R; baseline may materialize R,R.",
    ),
)

BWD_BWD_OUTPUT_SPECS: tuple[BoundarySlotSpec, ...] = (
    BoundarySlotSpec("dk", "bwd_bwd_output", "bf16", ("B", "S*R", "H", "N"), "mamba_mimo_bwd_bwd"),
    BoundarySlotSpec("dv", "bwd_bwd_output", "bf16", ("B", "S", "H", "P"), "mamba_mimo_bwd_bwd"),
    BoundarySlotSpec("dmimo_v", "bwd_bwd_output", "fp32", ("B", "H", "R", "P"), "mamba_mimo_bwd_bwd"),
    BoundarySlotSpec("dq", "bwd_bwd_output", "bf16", ("B", "S*R", "H", "N"), "mamba_mimo_bwd_bwd"),
    BoundarySlotSpec("dfactor", "bwd_bwd_output", "fp32", ("B", "H", "S"), "mamba_mimo_bwd_bwd"),
    BoundarySlotSpec("dgamma_diag", "bwd_bwd_output", "fp32", ("B", "H", "S"), "mamba_mimo_bwd_bwd"),
    BoundarySlotSpec(
        "dangles",
        "bwd_bwd_output",
        "fp32",
        ("B", "S", "H", "rotary_dim"),
        "mamba_mimo_bwd_bwd",
    ),
    BoundarySlotSpec("dd", "bwd_bwd_output", "fp32", ("B", "H"), "mamba_mimo_bwd_bwd"),
    BoundarySlotSpec("dda", "bwd_bwd_output", "fp32", ("B", "H", "S"), "mamba_mimo_bwd_bwd"),
    BoundarySlotSpec(
        "dssda",
        "bwd_bwd_output",
        "fp32",
        ("B", "H", "nchunks", "chunk", "chunk"),
        "mamba_mimo_bwd_bwd",
    ),
    BoundarySlotSpec("dda_cs_rev", "bwd_bwd_output", "fp32", ("B", "H", "S"), "mamba_mimo_bwd_bwd"),
    BoundarySlotSpec("dda_cs", "bwd_bwd_output", "fp32", ("B", "H", "S"), "mamba_mimo_bwd_bwd"),
)

FULL_CHAIN_COMPARE_NAMES = tuple(
    spec.name for spec in (*BWD_FWD_HANDOFF_SPECS, *BWD_BWD_OUTPUT_SPECS)
)
BWD_BWD_OUTPUT_NAMES = tuple(spec.name for spec in BWD_BWD_OUTPUT_SPECS)


def coerce_shape(value: Shape | dict[str, Any]) -> Shape:
    if isinstance(value, Shape):
        return value
    return Shape(
        name=str(value.get("name", "custom")),
        B=int(value["B"]),
        S=int(value["S"]),
        H=int(value["H"]),
        G=int(value["G"]),
        N=int(value["N"]),
        P=int(value["P"]),
        R=int(value["R"]),
        chunk=int(value.get("chunk", 16)),
        rotary_dim_divisor=int(value.get("rotary_dim_divisor", 4)),
    )


def selected_shapes(shape_csv: str) -> list[Shape]:
    shapes: list[Shape] = []
    for raw_name in shape_csv.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in SHAPES:
            raise ValueError(f"unknown shape {name!r}; choose one of {sorted(SHAPES)}")
        shapes.append(SHAPES[name])
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def _dim_value(expr: str, shape: Shape) -> int:
    values = {
        "B": shape.B,
        "S": shape.S,
        "H": shape.H,
        "G": shape.G,
        "N": shape.N,
        "P": shape.P,
        "R": shape.R,
        "S*R": shape.S * shape.R,
        "R*R": shape.R * shape.R,
        "chunk": shape.chunk,
        "nchunks": shape.nchunks,
        "rotary_dim": shape.rotary_dim,
    }
    if expr not in values:
        raise ValueError(f"unsupported shape expression {expr!r}")
    return values[expr]


def _numel(dims: tuple[int, ...]) -> int:
    return math.prod(dims)


def slot_schema(
    shape_like: Shape | dict[str, Any],
    specs: tuple[BoundarySlotSpec, ...] = (*BWD_FWD_HANDOFF_SPECS, *BWD_BWD_OUTPUT_SPECS),
) -> list[dict[str, Any]]:
    shape = coerce_shape(shape_like)
    slots: list[dict[str, Any]] = []
    for spec in specs:
        dims = tuple(_dim_value(expr, shape) for expr in spec.shape_expr)
        bytes_per_element = DTYPE_BYTES[spec.dtype]
        total_bytes = _numel(dims) * bytes_per_element
        slots.append(
            {
                "name": spec.name,
                "role": spec.role,
                "producer": spec.producer,
                "dtype": spec.dtype,
                "shape_expr": list(spec.shape_expr),
                "shape": list(dims),
                "numel": _numel(dims),
                "bytes": total_bytes,
                "mib": total_bytes / (1024**2),
                "required_for_monolithic_ab": spec.required_for_monolithic_ab,
                "note": spec.note,
            }
        )
    return slots


def memory_accounting(shape_like: Shape | dict[str, Any]) -> dict[str, Any]:
    shape = coerce_shape(shape_like)
    handoff_slots = slot_schema(shape, BWD_FWD_HANDOFF_SPECS)
    bwd_bwd_slots = slot_schema(shape, BWD_BWD_OUTPUT_SPECS)
    handoff_bytes = sum(int(slot["bytes"]) for slot in handoff_slots)
    bwd_bwd_output_bytes = sum(int(slot["bytes"]) for slot in bwd_bwd_slots)
    qkv_bytes = (
        2 * shape.B * shape.S * shape.R * shape.G * shape.N * DTYPE_BYTES["bf16"]
        + 2 * shape.B * shape.S * shape.H * shape.P * DTYPE_BYTES["bf16"]
    )
    static_param_bytes = (
        2 * shape.H * shape.R * shape.N * DTYPE_BYTES["fp32"]
        + 2 * shape.H * shape.R * shape.P * DTYPE_BYTES["fp32"]
        + shape.H * DTYPE_BYTES["fp32"]
    )
    scalar_input_bytes = (
        3 * shape.B * shape.H * shape.S * DTYPE_BYTES["fp32"]
        + shape.B * shape.H * shape.S * DTYPE_BYTES["bf16"]
        + shape.B * shape.H * shape.nchunks * shape.chunk * shape.chunk * DTYPE_BYTES["fp32"]
    )
    estimated_live_floor_bytes = (
        qkv_bytes + static_param_bytes + scalar_input_bytes + handoff_bytes + bwd_bwd_output_bytes
    )
    return {
        "shape": shape.to_dict(),
        "handoff_cache_bytes": handoff_bytes,
        "handoff_cache_mib": handoff_bytes / (1024**2),
        "bwd_bwd_output_bytes": bwd_bwd_output_bytes,
        "bwd_bwd_output_mib": bwd_bwd_output_bytes / (1024**2),
        "comparison_duplicate_bwd_bwd_output_bytes": bwd_bwd_output_bytes,
        "comparison_duplicate_bwd_bwd_output_mib": bwd_bwd_output_bytes / (1024**2),
        "estimated_input_and_param_bytes": qkv_bytes + static_param_bytes + scalar_input_bytes,
        "estimated_live_floor_bytes": estimated_live_floor_bytes,
        "estimated_live_floor_mib": estimated_live_floor_bytes / (1024**2),
        "slots": {
            "handoff": handoff_slots,
            "bwd_bwd_outputs": bwd_bwd_slots,
        },
    }


def readiness_gates() -> list[dict[str, str]]:
    return [
        {
            "gate": "full_boundary_correctness",
            "requirement": "all bwd_bwd output slots match main_guarded_stage2 within declared tolerance",
        },
        {
            "gate": "no_missing_work",
            "requirement": "off-time/state work plus full DK/DQ/DV/DMIMO_V/scalar outputs are implemented in-kernel",
        },
        {
            "gate": "memory",
            "requirement": "integrated peak allocated/reserved memory is at or below main_guarded_stage2",
        },
        {
            "gate": "launch_count",
            "requirement": "one bwd_bwd replacement launch is the target; extra launches need measured chain speedup",
        },
        {
            "gate": "h200",
            "requirement": "H200 smoke and productionish A/B pass against main_guarded_stage2",
        },
        {
            "gate": "portability",
            "requirement": "H100 smoke or another agreed portability smoke passes",
        },
        {
            "gate": "training_ab",
            "requirement": "guarded production A/B confirms the microbench win survives workload variance",
        },
        {
            "gate": "modal_hygiene",
            "requirement": "runs use unique IDs, bounded timeouts, artifact receipts, and no left-running wave-owned apps",
        },
    ]


def candidate_configs(monolithic_candidate_csv: str | None = None) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [
        {
            "candidate_id": "main_guarded_stage2",
            "display_name": "main guarded stage2 bf1/bb0",
            "role": "production_reference",
            "implementation_class": "tilelang_guarded_stage2",
            "source": {
                "commit": MAIN_GUARDED_STAGE2_COMMIT,
                "module": MAIN_GUARDED_STAGE2_MODULE,
                "doc": MAIN_GUARDED_STAGE2_DOC,
            },
            "config": {
                "tilelang_variant": "stage2_bf1_bb0",
                "bf_num_stages": 1,
                "bb_num_stages": 0,
                "flattened_inputs": True,
                "flat_qk_dot": True,
                "env_gates": (
                    "CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1",
                    "MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1",
                ),
            },
            "boundary_contract": {
                "required_outputs": list(BWD_BWD_OUTPUT_NAMES),
                "comparison_reference": "baseline and future monolithic candidates",
            },
        },
        {
            "candidate_id": "cuda_covered_subset_wave9",
            "display_name": "prior CUDA covered subset",
            "role": "prior_component_floor",
            "implementation_class": "cuda_component_subset",
            "source": {
                "branches": (
                    "worker/mamba3-cuda-full-bwd-ab",
                    "worker/mamba3-cuda-dmimo-reduce",
                ),
                "docs": (
                    "docs/status/mamba3_cuda_full_bwd_ab_wave9_2026_04_30.md",
                    "docs/status/mamba3_cuda_bwd_bwd_10wave_summary_2026_04_30.md",
                ),
            },
            "config": {
                "scope": "component timing floor only",
                "launches": "wave7 combined same-time slice plus optional DMIMO_V sidecar",
            },
            "covered_slots": {
                "partial": ["dk", "dq", "dv", "dmimo_v", "dgamma_diag"],
                "missing": [
                    "dfactor",
                    "dangles",
                    "dd",
                    "dda",
                    "dssda",
                    "dda_cs_rev",
                    "dda_cs",
                ],
            },
        },
    ]

    names = [
        item.strip()
        for item in (monolithic_candidate_csv or "monolithic_chunk_candidate").split(",")
        if item.strip()
    ]
    for name in names:
        configs.append(
            {
                "candidate_id": name,
                "display_name": name.replace("_", " "),
                "role": "future_monolithic_candidate",
                "implementation_class": "monolithic_mamba_mimo_bwd_bwd",
                "source": {"status": "not_integrated_in_wave1"},
                "config": {
                    "expected_call_boundary": "mamba_mimo_bwd_bwd",
                    "expected_output_slots": list(BWD_BWD_OUTPUT_NAMES),
                },
                "boundary_contract": {
                    "required_outputs": list(BWD_BWD_OUTPUT_NAMES),
                    "readiness_gates": readiness_gates(),
                },
            }
        )
    return configs


def empty_slot_results(status: str, note: str = "") -> dict[str, dict[str, Any]]:
    return {
        name: {
            "status": status,
            "max_abs": None,
            "ref_absmax": None,
            "rel_to_ref_absmax": None,
            "full_boundary_pass": False,
            "note": note,
        }
        for name in BWD_BWD_OUTPUT_NAMES
    }


def slot_results_from_diffs(diffs: dict[str, Any], *, atol: float = 0.0) -> dict[str, dict[str, Any]]:
    results = empty_slot_results("not_reported")
    for name in BWD_BWD_OUTPUT_NAMES:
        diff = diffs.get(name)
        if diff is None:
            continue
        max_abs = diff.get("max_abs")
        passed = max_abs is not None and float(max_abs) <= atol
        results[name] = {
            "status": "pass" if passed else "fail",
            "max_abs": max_abs,
            "ref_absmax": diff.get("ref_absmax"),
            "rel_to_ref_absmax": diff.get("rel_to_ref_absmax"),
            "full_boundary_pass": passed,
            "note": f"atol={atol}",
        }
    return results


def cuda_subset_slot_results(
    cuda_correctness: dict[str, Any],
    *,
    dmimov_sidecar_receipt: bool,
) -> dict[str, dict[str, Any]]:
    results = empty_slot_results("missing", "not covered by the prior CUDA subset")
    diag = cuda_correctness.get("wave7_combined_diag_vs_wave5_timestep_post_cuda", {})
    qk_dv = cuda_correctness.get("wave7_combined_dv_vs_torch_reference", {})

    for slot_name, diff_name in (("dk", "dk_delta"), ("dq", "dq_delta"), ("dgamma_diag", "dgamma_diag")):
        max_abs = diag.get(diff_name)
        results[slot_name] = {
            "status": "partial_component_pass" if max_abs is not None else "partial_not_run",
            "max_abs": max_abs,
            "ref_absmax": None,
            "rel_to_ref_absmax": None,
            "full_boundary_pass": False,
            "coverage": "same-time diagonal/qk slice only",
            "note": "component parity does not imply full boundary parity",
        }

    dv_abs = qk_dv.get("dv_delta")
    results["dv"] = {
        "status": "partial_component_pass" if dv_abs is not None else "partial_not_run",
        "max_abs": dv_abs,
        "ref_absmax": None,
        "rel_to_ref_absmax": None,
        "full_boundary_pass": False,
        "coverage": "same-time qk_dot -> dPsiV -> DV slice only",
        "note": "component parity does not imply full DV parity",
    }
    results["dmimo_v"] = {
        "status": "external_partial_receipt" if dmimov_sidecar_receipt else "partial_not_run",
        "max_abs": None,
        "ref_absmax": None,
        "rel_to_ref_absmax": None,
        "full_boundary_pass": False,
        "coverage": "qk_dot same-time DMIMO_V sidecar only",
        "note": "must be remeasured inside the integrated monolithic boundary",
    }
    return results


def summarize_slot_results(slot_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passed = [
        name
        for name, result in slot_results.items()
        if result.get("full_boundary_pass") is True
    ]
    failed = [
        name
        for name, result in slot_results.items()
        if result.get("status") == "fail"
    ]
    missing = [
        name
        for name, result in slot_results.items()
        if result.get("status") in {"missing", "not_reported", "not_run", "partial_not_run"}
    ]
    partial = [
        name
        for name, result in slot_results.items()
        if str(result.get("status", "")).startswith("partial")
        or result.get("status") == "external_partial_receipt"
    ]
    return {
        "required_count": len(BWD_BWD_OUTPUT_NAMES),
        "full_boundary_pass_count": len(passed),
        "full_boundary_pass": len(passed) == len(BWD_BWD_OUTPUT_NAMES),
        "passed": passed,
        "failed": failed,
        "partial": partial,
        "missing": missing,
    }
