"""Shared schema for Mamba3 monolithic bwd_bwd production A/B reports."""

from __future__ import annotations

import math
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mamba3-mono-ab/v1"
TRAINING_AB_STUB_VERSION = "mamba3-guarded-stage2-training-ab/v1"
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
OPTIONAL_HARDWARE_TAGS = ("H100", "H200", "B200")
DEFAULT_H200_SM_COUNT = 132
PRODUCTION_RESOURCE_METADATA_FIELDS = (
    "registers_per_thread",
    "smem_bytes",
    "active_blocks_per_sm",
    "theoretical_occupancy",
)

_RESOURCE_METADATA_ALIASES = {
    "registers_per_thread": ("registers_per_thread", "regs_per_thread"),
    "smem_bytes": (
        "smem_bytes",
        "shared_memory_bytes",
        "dynamic_smem_bytes",
        "static_smem_bytes",
    ),
    "active_blocks_per_sm": ("active_blocks_per_sm", "blocks_per_sm"),
    "theoretical_occupancy": (
        "theoretical_occupancy",
        "occupancy",
        "theoretical_occupancy_pct",
        "occupancy_pct",
    ),
}
_CTA_METADATA_ALIASES = {
    "total_ctas": (
        "total_ctas",
        "cta_count",
        "grid_ctas",
        "total_CTA_count",
        "total_CTAs",
        "scan_owner_ctas",
        "chunk_owner_ctas",
        "total_blocks",
        "grid_blocks",
    ),
    "ctas_per_sm": (
        "ctas_per_sm",
        "cta_per_sm",
        "ctas_per_sm_at_132_sms",
        "scan_owner_ctas_per_sm_at_132_sms",
        "chunk_owner_ctas_per_sm_at_132_sms",
    ),
    "sm_count": ("sm_count", "sms", "gpu_sm_count", "h200_sm_count"),
}
_MICRO_GEMM_SCOPE_MARKERS = (
    "micro_gemm",
    "micro-gemm",
    "single_gemm",
    "single-gemm",
    "cute_gemm",
    "64x64x64",
)
_LOCAL_SPEEDUP_SCOPE_MARKERS = (
    "local_component_speedup",
    "component_speedup",
    "local_speedup",
    "tile_chain",
    "lkq_chain",
    "masked_lkq_apply",
)
_INTEGRATED_FULL_SLOT_TIMING_MARKERS = (
    "integrated_full_bwd_bwd",
    "integrated_full_slot_bwd_bwd",
    "full_slot_integrated_timing",
    "full_bwd_bwd_boundary",
    "monolithic_bwd_bwd_boundary",
)
_MODAL_HYGIENE_PASS_STATUSES = {"pass", "passed", "ok", "clean", "stopped"}

_COMPONENT_RECORD_PAYLOAD_KEYS = (
    "mamba3_mono_ab_component_records",
    "candidate_component_records",
    "component_records",
)
_COMPONENT_RECORD_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)\n```", re.DOTALL)


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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _as_str_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _slot_list(value: Any) -> list[str]:
    requested = [item.lower() for item in _as_str_list(value)]
    invalid = [item for item in requested if item not in BWD_BWD_OUTPUT_NAMES]
    if invalid:
        raise ValueError(
            f"unknown bwd_bwd output slot(s) {invalid!r}; "
            f"choose from {list(BWD_BWD_OUTPUT_NAMES)!r}"
        )
    return [name for name in BWD_BWD_OUTPUT_NAMES if name in requested]


def _float_field(mapping: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return float(mapping[name])
    return None


def _int_field(mapping: dict[str, Any], names: tuple[str, ...], default: int = 0) -> int:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return int(mapping[name])
    return default


def _shape_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Shape):
        return value.name
    if isinstance(value, dict):
        name = value.get("name")
        return str(name).strip() if name else None
    name = str(value).strip()
    return name or None


def _normalize_hardware_tags(raw: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for mapping in (raw, metadata):
        for key in ("hardware_tags", "gpu_tags", "devices", "device", "gpu"):
            if key in mapping:
                values.extend(_as_list(mapping[key]))

    found: set[str] = set()
    for value in values:
        text = str(value).upper()
        for tag in OPTIONAL_HARDWARE_TAGS:
            if tag in text:
                found.add(tag)
    return [tag for tag in OPTIONAL_HARDWARE_TAGS if tag in found]


def _lookup_alias(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for key in names:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _metadata_lookup(metadata: dict[str, Any], logical_field: str) -> Any:
    return _lookup_alias(metadata, _RESOURCE_METADATA_ALIASES[logical_field])


def _metadata_float(metadata: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    value = _lookup_alias(metadata, aliases)
    return float(value) if value is not None else None


def _metadata_int(metadata: dict[str, Any], aliases: tuple[str, ...]) -> int | None:
    value = _lookup_alias(metadata, aliases)
    return int(float(value)) if value is not None else None


def _receipt_scope_from_raw(
    raw: dict[str, Any],
    *,
    implementation_class: str,
    shape_name: str | None,
    components: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    explicit = raw.get("receipt_scope") or metadata.get("receipt_scope")
    if explicit:
        return str(explicit)

    haystack = " ".join(
        str(item)
        for item in (
            implementation_class,
            shape_name or "",
            raw.get("candidate_id") or raw.get("id") or raw.get("name") or "",
            *(component.get("component_id", "") for component in components),
        )
    ).lower()
    if any(marker in haystack for marker in _MICRO_GEMM_SCOPE_MARKERS):
        return "micro_gemm_only"
    return "bwd_bwd_component"


def _payload_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = []
        for key in _COMPONENT_RECORD_PAYLOAD_KEYS:
            if key in payload:
                records = payload[key]
                break
        if not records and any(key in payload for key in ("candidate_id", "id", "name")):
            records = [payload]
    else:
        records = []
    if not isinstance(records, list):
        raise ValueError("candidate component record payload must be a list or record object")
    return [record for record in records if isinstance(record, dict)]


def _components_from_raw(raw: dict[str, Any]) -> list[dict[str, Any]]:
    raw_components = (
        raw.get("components")
        or raw.get("component_records")
        or raw.get("timing_components")
        or []
    )
    if isinstance(raw_components, dict):
        raw_components = [
            {"component_id": key, **(value if isinstance(value, dict) else {"mean_ms": value})}
            for key, value in raw_components.items()
        ]
    if not raw_components:
        total_ms = _float_field(
            raw,
            (
                "projected_bwd_bwd_ms",
                "candidate_bwd_bwd_ms",
                "component_total_ms",
                "total_ms",
                "combined_ms",
                "mean_ms",
                "ms",
            ),
        )
        if total_ms is None:
            return []
        raw_components = [
            {
                "component_id": raw.get("component_id") or "total",
                "mean_ms": total_ms,
                "launches": raw.get("launches", 1),
                "covered_slots": raw.get("covered_slots"),
            }
        ]

    components: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_components):
        if not isinstance(item, dict):
            item = {"component_id": f"component_{idx}", "mean_ms": item}
        component_id = (
            item.get("component_id")
            or item.get("id")
            or item.get("name")
            or f"component_{idx}"
        )
        mean_ms = _float_field(item, ("mean_ms", "elapsed_ms", "ms", "time_ms"))
        components.append(
            {
                "component_id": str(component_id),
                "mean_ms": mean_ms,
                "launches": _int_field(item, ("launches", "launch_count"), default=1),
                "covered_slots": _slot_list(
                    item.get("covered_slots")
                    or item.get("slots")
                    or item.get("outputs")
                ),
                "include_in_projection": bool(item.get("include_in_projection", True)),
                "status": str(item.get("status", "reported")),
                "note": str(item.get("note", "")),
            }
        )
    return components


def _normalize_memory(raw: dict[str, Any]) -> dict[str, Any]:
    memory = raw.get("memory_peak_gib") or raw.get("memory") or {}
    if not isinstance(memory, dict):
        memory = {"max_memory_allocated_gib": memory}
    out = dict(memory)
    allocated = _float_field(
        raw,
        (
            "max_memory_allocated_gib",
            "peak_allocated_gib",
            "allocated_gib",
        ),
    )
    reserved = _float_field(
        raw,
        (
            "max_memory_reserved_gib",
            "peak_reserved_gib",
            "reserved_gib",
        ),
    )
    if allocated is not None:
        out["max_memory_allocated_gib"] = allocated
    if reserved is not None:
        out["max_memory_reserved_gib"] = reserved
    return out


def _normalize_reference(raw: dict[str, Any]) -> dict[str, Any]:
    reference = (
        raw.get("reference")
        or raw.get("stage2_reference")
        or raw.get("tilelang_reference")
        or {}
    )
    if not isinstance(reference, dict):
        reference = {}
    out = dict(reference)
    fields = {
        "stage2_bwd_fwd_ms": (
            "stage2_bwd_fwd_ms",
            "tilelang_stage2_bwd_fwd_ms",
            "reference_bwd_fwd_ms",
        ),
        "stage2_bwd_bwd_ms": (
            "stage2_bwd_bwd_ms",
            "tilelang_stage2_bwd_bwd_ms",
            "reference_bwd_bwd_ms",
        ),
        "stage2_chain_ms": (
            "stage2_chain_ms",
            "tilelang_stage2_chain_ms",
            "reference_chain_ms",
        ),
        "stage2_max_memory_allocated_gib": (
            "stage2_max_memory_allocated_gib",
            "tilelang_stage2_max_memory_allocated_gib",
        ),
        "stage2_max_memory_reserved_gib": (
            "stage2_max_memory_reserved_gib",
            "tilelang_stage2_max_memory_reserved_gib",
        ),
    }
    for out_name, names in fields.items():
        value = _float_field(raw, names)
        if value is not None:
            out[out_name] = value
    return out


def normalize_candidate_component_record(
    raw: dict[str, Any],
    *,
    source_path: str | None = None,
) -> dict[str, Any]:
    candidate_id = raw.get("candidate_id") or raw.get("id") or raw.get("name")
    if not candidate_id:
        raise ValueError("candidate component record requires candidate_id, id, or name")

    components = _components_from_raw(raw)
    implementation_class = str(raw.get("implementation_class", "lane_component_record"))
    shape_name = _shape_name(raw.get("shape"))
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    metadata = dict(metadata)
    modal_hygiene = raw.get("modal_hygiene") or metadata.get("modal_hygiene") or {}
    if not isinstance(modal_hygiene, dict):
        modal_hygiene = {"status": str(modal_hygiene)}

    covered_slots = set(
        _slot_list(raw.get("covered_slots") or raw.get("slots") or raw.get("outputs"))
    )
    for component in components:
        covered_slots.update(component["covered_slots"])
    ordered_covered = [name for name in BWD_BWD_OUTPUT_NAMES if name in covered_slots]
    explicit_missing = raw.get("missing_slots")
    missing_slots = (
        _slot_list(explicit_missing)
        if explicit_missing is not None
        else [name for name in BWD_BWD_OUTPUT_NAMES if name not in covered_slots]
    )

    source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    source = dict(source)
    if raw.get("lane") is not None:
        source.setdefault("lane", raw["lane"])
    if raw.get("doc") is not None:
        source.setdefault("doc", raw["doc"])
    if raw.get("commit") is not None:
        source.setdefault("commit", raw["commit"])
    if source_path:
        source.setdefault("doc", source_path)

    correctness = raw.get("correctness") if isinstance(raw.get("correctness"), dict) else {}
    correctness = dict(correctness)
    max_abs = _float_field(raw, ("correctness_max_abs", "max_abs"))
    if max_abs is not None:
        correctness.setdefault("max_abs", max_abs)

    gate_budget = (
        raw.get("gate_budget")
        or raw.get("ab_gate_budget")
        or raw.get("budget")
        or {}
    )
    if not isinstance(gate_budget, dict):
        gate_budget = {}

    return {
        "candidate_id": str(candidate_id),
        "display_name": str(raw.get("display_name") or str(candidate_id).replace("_", " ")),
        "role": str(raw.get("role", "external_component_candidate")),
        "implementation_class": implementation_class,
        "shape": shape_name,
        "receipt_scope": _receipt_scope_from_raw(
            raw,
            implementation_class=implementation_class,
            shape_name=shape_name,
            components=components,
            metadata=metadata,
        ),
        "status": str(raw.get("status", "reported")),
        "source": source,
        "components": components,
        "covered_slots": ordered_covered,
        "missing_slots": missing_slots,
        "projected_bwd_bwd_ms": _float_field(
            raw,
            (
                "projected_bwd_bwd_ms",
                "candidate_bwd_bwd_ms",
                "component_total_ms",
                "total_ms",
                "combined_ms",
            ),
        ),
        "launches": _int_field(raw, ("launches", "launch_count"), default=0),
        "memory_peak_gib": _normalize_memory(raw),
        "correctness": correctness,
        "reference": _normalize_reference(raw),
        "metadata": metadata,
        "hardware_tags": _normalize_hardware_tags(raw, metadata),
        "modal_hygiene": modal_hygiene,
        "gate_budget": gate_budget,
        "note": str(raw.get("note", "")),
    }


def normalize_candidate_component_records(records: Any) -> list[dict[str, Any]]:
    return [
        normalize_candidate_component_record(record)
        for record in _payload_records(records)
    ]


def candidate_component_records_from_json(
    value: str | dict[str, Any] | list[Any],
    *,
    source_path: str | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(value) if isinstance(value, str) else value
    return [
        normalize_candidate_component_record(record, source_path=source_path)
        for record in _payload_records(payload)
    ]


def candidate_component_records_from_markdown(
    text: str,
    *,
    source_path: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for info, body in _COMPONENT_RECORD_FENCE_RE.findall(text):
        info_l = info.lower()
        if not (
            "json" in info_l
            or "mamba3-mono-ab" in info_l
            or "candidate" in info_l
        ):
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue
        payload_records = _payload_records(payload)
        if payload_records:
            records.extend(
                normalize_candidate_component_record(record, source_path=source_path)
                for record in payload_records
            )
    return records


def load_candidate_component_records(
    path_csv: str | None = None,
    json_text: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if json_text and json_text.strip():
        records.extend(candidate_component_records_from_json(json_text))
    for raw_path in _as_str_list(path_csv):
        path = Path(raw_path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            path_records = candidate_component_records_from_json(text, source_path=str(path))
        else:
            try:
                path_records = candidate_component_records_from_json(text, source_path=str(path))
            except json.JSONDecodeError:
                path_records = candidate_component_records_from_markdown(
                    text,
                    source_path=str(path),
                )
        if not path_records:
            raise ValueError(
                f"{path} did not contain candidate component records; add a JSON "
                "payload or a fenced JSON block with candidate_component_records"
            )
        records.extend(path_records)
    return records


def filter_candidate_component_records_for_shape(
    records: list[dict[str, Any]],
    shape_like: Shape | dict[str, Any] | str,
) -> list[dict[str, Any]]:
    shape_name = (
        _shape_name(shape_like)
        if isinstance(shape_like, str)
        else coerce_shape(shape_like).name
    )
    return [
        record
        for record in records
        if record.get("shape") in (None, "", shape_name)
    ]


def _resource_metadata_gate(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    present: dict[str, Any] = {}
    missing: list[str] = []
    for field in PRODUCTION_RESOURCE_METADATA_FIELDS:
        value = _metadata_lookup(metadata, field)
        if value is None:
            missing.append(field)
        else:
            present[field] = value
    return {
        "status": "pass" if not missing else "missing",
        "required_fields": list(PRODUCTION_RESOURCE_METADATA_FIELDS),
        "present": present,
        "missing": missing,
    }


def _cta_count_occupancy_gate(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    total_ctas = _metadata_int(metadata, _CTA_METADATA_ALIASES["total_ctas"])
    sm_count = (
        _metadata_int(metadata, _CTA_METADATA_ALIASES["sm_count"])
        or DEFAULT_H200_SM_COUNT
    )
    ctas_per_sm = _metadata_float(metadata, _CTA_METADATA_ALIASES["ctas_per_sm"])
    if ctas_per_sm is None and total_ctas is not None and sm_count > 0:
        ctas_per_sm = total_ctas / sm_count

    occupancy = _metadata_lookup(metadata, "theoretical_occupancy")
    occupancy_fraction = float(occupancy) if occupancy is not None else None
    if occupancy_fraction is not None and occupancy_fraction > 1.0:
        occupancy_fraction /= 100.0

    if total_ctas is None:
        status = "missing"
    elif (
        total_ctas < sm_count
        or (ctas_per_sm is not None and ctas_per_sm < 1.0)
    ):
        status = "underfilled"
    else:
        status = "pass"

    return {
        "status": status,
        "total_ctas": total_ctas,
        "sm_count": sm_count,
        "ctas_per_sm": ctas_per_sm,
        "minimum_total_ctas": sm_count,
        "minimum_ctas_per_sm": 1.0,
        "theoretical_occupancy": occupancy_fraction,
        "required": "at least one CTA per H200 SM plus reported theoretical occupancy",
    }


def _modal_hygiene_gate(record: dict[str, Any]) -> dict[str, Any]:
    hygiene = record.get("modal_hygiene") or {}
    if not isinstance(hygiene, dict) or not hygiene:
        return {"status": "missing", "required": "clean final Modal app state"}

    status = str(
        hygiene.get("status")
        or hygiene.get("verdict")
        or hygiene.get("cleanup_status")
        or ""
    ).lower()
    active_count = None
    for key in (
        "active_same_campaign_count",
        "active_total",
        "active_cppmega_mamba3",
        "running_or_nonzero_tasks",
    ):
        if hygiene.get(key) is not None:
            active_count = int(hygiene[key])
            break

    if status in _MODAL_HYGIENE_PASS_STATUSES or active_count == 0:
        gate_status = "pass"
    elif active_count is not None and active_count > 0:
        gate_status = "fail"
    else:
        gate_status = "missing"
    return {
        "status": gate_status,
        "reported_status": status or None,
        "active_count": active_count,
        "required": "clean final Modal app state",
    }


def _is_micro_gemm_record(record: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(item)
        for item in (
            record.get("receipt_scope", ""),
            record.get("implementation_class", ""),
            record.get("shape", ""),
            record.get("candidate_id", ""),
            *(component.get("component_id", "") for component in record.get("components", [])),
        )
    ).lower()
    return any(marker in haystack for marker in _MICRO_GEMM_SCOPE_MARKERS)


def _is_local_speedup_record(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata") or {}
    haystack = " ".join(
        str(item)
        for item in (
            record.get("receipt_scope", ""),
            record.get("implementation_class", ""),
            record.get("shape", ""),
            record.get("candidate_id", ""),
            record.get("status", ""),
            metadata.get("timing_scope", ""),
            metadata.get("receipt_scope", ""),
            *(component.get("component_id", "") for component in record.get("components", [])),
            *(component.get("note", "") for component in record.get("components", [])),
        )
    ).lower()
    return any(marker in haystack for marker in _LOCAL_SPEEDUP_SCOPE_MARKERS)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _integrated_timing_gate(
    record: dict[str, Any],
    *,
    total_ms: float | None,
) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    timing_scope = str(
        metadata.get("timing_scope")
        or metadata.get("integrated_timing_scope")
        or record.get("timing_scope")
        or record.get("receipt_scope")
        or ""
    ).lower()
    explicit_integrated = (
        _truthy(metadata.get("integrated_full_slot_timing"))
        or _truthy(metadata.get("integrated_timing"))
        or any(marker in timing_scope for marker in _INTEGRATED_FULL_SLOT_TIMING_MARKERS)
    )
    local_only = _is_local_speedup_record(record) and not explicit_integrated
    if local_only:
        status = "local_component_only"
    elif total_ms is None:
        status = "missing_timing"
    elif explicit_integrated:
        status = "pass"
    else:
        status = "reported"
    return {
        "status": status,
        "timing_scope": timing_scope or None,
        "integrated_full_slot_timing": explicit_integrated,
        "local_component_speedup_only": local_only,
        "required": "full-slot integrated bwd_bwd timing for production credit",
    }


def production_candidate_gate(
    record: dict[str, Any],
    *,
    projection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the fail-closed production eligibility verdict for a receipt."""

    projection = projection or {}
    covered = set(record.get("covered_slots") or [])
    explicit_missing = set(record.get("missing_slots") or [])
    missing_slots = [
        name
        for name in BWD_BWD_OUTPUT_NAMES
        if name not in covered or name in explicit_missing
    ]
    correctness = record.get("correctness") or {}
    resource_gate = _resource_metadata_gate(record)
    cta_gate = _cta_count_occupancy_gate(record)
    modal_gate = _modal_hygiene_gate(record)
    hardware_tags = list(record.get("hardware_tags") or [])

    total_ms = projection.get("projected_bwd_bwd_ms")
    integrated_timing_gate = _integrated_timing_gate(record, total_ms=total_ms)
    reference = record.get("reference") or {}
    gate_budget = record.get("gate_budget") or {}
    budget_ms = (
        projection.get("stage2_bwd_bwd_ms")
        or reference.get("stage2_bwd_bwd_ms")
        or gate_budget.get("yellow_full_kernel_ms")
        or gate_budget.get("red_full_kernel_ms")
    )
    if total_ms is None:
        performance_status = "missing_timing"
    elif budget_ms is None:
        performance_status = "missing_budget"
    elif float(total_ms) <= float(budget_ms):
        performance_status = "pass"
    else:
        performance_status = "fail"

    rejection_reasons: list[str] = []
    if _is_micro_gemm_record(record):
        rejection_reasons.append("micro_gemm_only_receipt")
    if integrated_timing_gate["local_component_speedup_only"]:
        rejection_reasons.append("non_integrated_timing_receipt")
    if (
        integrated_timing_gate["status"] != "pass"
        and "non_integrated_timing_receipt" not in rejection_reasons
    ):
        rejection_reasons.append("non_integrated_timing_receipt")
    if missing_slots:
        rejection_reasons.append("missing_required_output_slots")
    if correctness.get("full_boundary_pass") is not True:
        rejection_reasons.append("full_boundary_correctness_not_reported")
    if resource_gate["status"] != "pass":
        rejection_reasons.append("resource_metadata_incomplete")
    if cta_gate["status"] == "underfilled":
        rejection_reasons.append("cta_count_underfilled")
    elif cta_gate["status"] != "pass":
        rejection_reasons.append("cta_count_missing")
    if "H200" not in hardware_tags:
        rejection_reasons.append("h200_hardware_tag_missing")
    if modal_gate["status"] != "pass":
        rejection_reasons.append("modal_hygiene_not_clean")
    if performance_status != "pass":
        rejection_reasons.append("performance_budget_not_met")

    production_credit = not rejection_reasons
    return {
        "production_credit": production_credit,
        "production_credit_ms": float(total_ms) if production_credit else 0.0,
        "credited_output_slots": list(BWD_BWD_OUTPUT_NAMES) if production_credit else [],
        "rejection_reasons": rejection_reasons,
        "receipt_scope": record.get("receipt_scope"),
        "required_output_slots": list(BWD_BWD_OUTPUT_NAMES),
        "missing_output_slots": missing_slots,
        "full_boundary_correctness": correctness.get("full_boundary_pass") is True,
        "resource_metadata": resource_gate,
        "cta_count_occupancy": cta_gate,
        "hardware_tags": {
            "reported": hardware_tags,
            "allowed_optional_tags": list(OPTIONAL_HARDWARE_TAGS),
            "h200_required_for_h200_budget": True,
        },
        "modal_hygiene": modal_gate,
        "integrated_timing": integrated_timing_gate,
        "performance_budget": {
            "status": performance_status,
            "projected_bwd_bwd_ms": total_ms,
            "budget_ms": budget_ms,
        },
    }


def component_record_projection(
    record: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ref = dict(record.get("reference") or {})
    if reference:
        ref.update({key: value for key, value in reference.items() if value is not None})

    component_sum = 0.0
    component_count = 0
    component_launches = 0
    missing_timing: list[str] = []
    for component in record.get("components", []):
        if not component.get("include_in_projection", True):
            continue
        component_launches += int(component.get("launches") or 0)
        mean_ms = component.get("mean_ms")
        if mean_ms is None:
            missing_timing.append(str(component.get("component_id")))
            continue
        component_sum += float(mean_ms)
        component_count += 1

    total_ms = record.get("projected_bwd_bwd_ms")
    if total_ms is None and component_count:
        total_ms = component_sum
    if total_ms is not None:
        total_ms = float(total_ms)

    launch_count = int(record.get("launches") or 0) or component_launches
    stage2_bwd_bwd_ms = ref.get("stage2_bwd_bwd_ms")
    stage2_bwd_fwd_ms = ref.get("stage2_bwd_fwd_ms")
    stage2_chain_ms = ref.get("stage2_chain_ms")

    ratio = None
    speedup = None
    remaining = None
    if total_ms is not None and stage2_bwd_bwd_ms:
        ratio = total_ms / float(stage2_bwd_bwd_ms)
        speedup = float(stage2_bwd_bwd_ms) / total_ms if total_ms > 0 else None
        remaining = float(stage2_bwd_bwd_ms) - total_ms

    chain_floor_ms = None
    chain_speedup = None
    if total_ms is not None and stage2_bwd_fwd_ms is not None:
        chain_floor_ms = float(stage2_bwd_fwd_ms) + total_ms
        if stage2_chain_ms and chain_floor_ms > 0:
            chain_speedup = float(stage2_chain_ms) / chain_floor_ms

    memory = record.get("memory_peak_gib") or {}
    candidate_allocated = memory.get("max_memory_allocated_gib")
    candidate_reserved = memory.get("max_memory_reserved_gib")
    ref_allocated = ref.get("stage2_max_memory_allocated_gib")
    ref_reserved = ref.get("stage2_max_memory_reserved_gib")

    projection = {
        "candidate_id": record.get("candidate_id"),
        "shape": record.get("shape"),
        "projection_status": "missing_timing" if total_ms is None else "projected",
        "component_total_ms": component_sum if component_count else None,
        "projected_bwd_bwd_ms": total_ms,
        "launch_count": launch_count,
        "covered_slots": list(record.get("covered_slots") or []),
        "missing_slots": list(record.get("missing_slots") or []),
        "coverage_fraction": (
            len(record.get("covered_slots") or []) / len(BWD_BWD_OUTPUT_NAMES)
        ),
        "stage2_bwd_bwd_ms": stage2_bwd_bwd_ms,
        "ratio_vs_stage2_bwd_bwd": ratio,
        "speedup_floor_vs_stage2_bwd_bwd": speedup,
        "remaining_budget_ms_to_equal_stage2_bwd_bwd": remaining,
        "stage2_bwd_fwd_ms": stage2_bwd_fwd_ms,
        "stage2_chain_ms": stage2_chain_ms,
        "stage2_chain_with_candidate_floor_ms": chain_floor_ms,
        "stage2_chain_speedup_floor": chain_speedup,
        "memory_peak_gib": memory,
        "memory_delta_vs_stage2_gib": {
            "allocated": (
                float(candidate_allocated) - float(ref_allocated)
                if candidate_allocated is not None and ref_allocated is not None
                else None
            ),
            "reserved": (
                float(candidate_reserved) - float(ref_reserved)
                if candidate_reserved is not None and ref_reserved is not None
                else None
            ),
        },
        "missing_component_timings": missing_timing,
        "gate_budget": record.get("gate_budget") or {},
        "full_boundary_ready": (
            not record.get("missing_slots")
            and bool(record.get("correctness", {}).get("full_boundary_pass"))
        ),
    }
    projection["production_gate"] = production_candidate_gate(
        record,
        projection=projection,
    )
    return projection


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
            "requirement": (
                "all real mamba_mimo_bwd_bwd output slots match "
                "main_guarded_stage2 within declared tolerance"
            ),
        },
        {
            "gate": "no_missing_work",
            "requirement": "off-time/state work plus full DK/DQ/DV/DMIMO_V/scalar outputs are implemented in-kernel",
        },
        {
            "gate": "reject_micro_gemm_only",
            "requirement": (
                "correct CuTe/CUDA micro-GEMM receipts are component receipts only; "
                "they earn zero production credit until mapped to the real output slots"
            ),
        },
        {
            "gate": "reject_local_component_speedup_only",
            "requirement": (
                "local tile/component speedups earn zero production credit until "
                "they become full-slot integrated bwd_bwd timing receipts"
            ),
        },
        {
            "gate": "resource_metadata",
            "requirement": (
                "registers/thread, shared-memory bytes, active blocks/SM, and "
                "theoretical occupancy are reported for the integrated candidate"
            ),
        },
        {
            "gate": "cta_count_occupancy",
            "requirement": (
                "integrated candidate reports total CTA count and can launch at "
                "least one CTA per H200 SM; theoretical occupancy is reported"
            ),
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
            "gate": "hardware_tags",
            "requirement": (
                "receipts may tag H100/H200/B200 evidence; H200 evidence is "
                "required before consuming the H200 production budget"
            ),
        },
        {
            "gate": "portability",
            "requirement": "H100 smoke, B200 smoke, or another agreed portability smoke passes",
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


def candidate_configs(
    monolithic_candidate_csv: str | None = None,
    component_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    component_records = normalize_candidate_component_records(component_records or [])
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

    for record in component_records:
        projection = component_record_projection(record)
        configs.append(
            {
                "candidate_id": record["candidate_id"],
                "display_name": record["display_name"],
                "role": record["role"],
                "implementation_class": record["implementation_class"],
                "receipt_scope": record["receipt_scope"],
                "source": record["source"],
                "shape": record["shape"],
                "status": record["status"],
                "config": {
                    "expected_call_boundary": "mamba_mimo_bwd_bwd",
                    "component_records": record["components"],
                    "covered_slots": record["covered_slots"],
                    "missing_slots": record["missing_slots"],
                },
                "component_projection": projection,
                "production_gate": projection["production_gate"],
                "boundary_contract": {
                    "required_outputs": list(BWD_BWD_OUTPUT_NAMES),
                    "readiness_gates": readiness_gates(),
                },
                "metadata": record.get("metadata") or {},
                "hardware_tags": record.get("hardware_tags") or [],
                "modal_hygiene": record.get("modal_hygiene") or {},
                "gate_budget": record.get("gate_budget") or {},
            }
        )

    component_candidate_ids = {record["candidate_id"] for record in component_records}
    names = [
        item.strip()
        for item in (monolithic_candidate_csv or "monolithic_chunk_candidate").split(",")
        if item.strip()
    ]
    for name in names:
        if name in component_candidate_ids:
            continue
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


def guarded_stage2_training_ab_stub(
    *,
    run_id: str = "mamba3_stage2_guarded_train_ab",
    train_iters: int = 100,
    remote_script: str = "scripts/remote_production_h200_nam56r_v1.sh",
) -> dict[str, Any]:
    """Return a command receipt for a default-off guarded stage2 training A/B."""

    run_id = run_id.strip() or "mamba3_stage2_guarded_train_ab"
    patch_module = MAIN_GUARDED_STAGE2_MODULE
    rollback = (
        "PYTHONPATH=. CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK=1 "
        f"python -m {patch_module}"
    )
    apply_stage2 = (
        "PYTHONPATH=. CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1 "
        "MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1 "
        f"python -m {patch_module}"
    )
    baseline = (
        f"RUN_ID={run_id}_baseline TRAIN_ITERS={train_iters} "
        f"VARIANT=tilelang bash {remote_script}"
    )
    candidate = (
        f"RUN_ID={run_id}_stage2_bf1bb0 TRAIN_ITERS={train_iters} "
        f"VARIANT=tilelang bash {remote_script}"
    )
    return {
        "schema_version": TRAINING_AB_STUB_VERSION,
        "run_id": run_id,
        "production_defaults_changed": False,
        "reference": {
            "candidate_id": "main_guarded_stage2",
            "commit": MAIN_GUARDED_STAGE2_COMMIT,
            "module": MAIN_GUARDED_STAGE2_MODULE,
            "doc": MAIN_GUARDED_STAGE2_DOC,
            "bf_num_stages": 1,
            "bb_num_stages": 0,
        },
        "checklist": [
            "serialize baseline and stage2 runs on the same host because the stage2 patch mutates installed mamba_ssm source",
            "rollback before the baseline leg and after the candidate leg",
            "apply stage2 only with CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1 and MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1",
            "record git commit, image tag, dataset path, tokenizer path, train/eval iterations, loss, tok/s, max memory, and any NaN/OOM",
            "treat the result as guarded A/B evidence only; it does not enable stage2 by default",
        ],
        "launcher_stub": {
            "baseline_leg": [
                rollback,
                baseline,
            ],
            "candidate_leg": [
                apply_stage2,
                candidate,
                rollback,
            ],
        },
        "comparison_fields": [
            "loss curve / final loss",
            "tokens_per_second",
            "max_memory_allocated_gib",
            "max_memory_reserved_gib",
            "failures_or_restarts",
        ],
    }
