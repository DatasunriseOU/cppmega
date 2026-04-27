"""Attribute GEMM-like torch profiler kernels to their launching CPU ops.

The torch profiler CUDA table is useful for ranking kernels, but it loses the
module/op context needed to decide whether a BF16 GEMM came from TE Linear,
cross entropy, optimizer math, routing, or a custom path.  This probe reads a
PyTorch chrome trace JSON and joins kernel events back to CPU ops via the
profiler ``External id`` field.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


GEMM_MARKERS = (
    "gemm",
    "nvjet",
    "block_scaled",
    "tensorop_bf16",
    "tensorop_s1688",
    "ue8m0xe4m3",
    "volta_sgemm",
    "ampere_sgemm",
)


@dataclass(frozen=True)
class CpuContext:
    name: str
    input_types: tuple[str, ...]
    input_dims: tuple[tuple[int | str, ...], ...]


@dataclass
class GemmGroup:
    kind: str
    cpu_op: str
    input_types: tuple[str, ...]
    input_dims: tuple[tuple[int | str, ...], ...]
    calls: int = 0
    total_us: float = 0.0
    sample_kernel: str = ""

    @property
    def total_ms(self) -> float:
        return self.total_us / 1000.0

    @property
    def avg_us(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.total_us / self.calls


def _external_id(event: dict[str, Any]) -> str | None:
    value = event.get("args", {}).get("External id")
    if value is None:
        return None
    return str(value)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, tuple):
        return value
    if value is None:
        return ()
    return (value,)


def _normalize_dim(dim: Any) -> tuple[int | str, ...]:
    if isinstance(dim, list):
        return tuple(dim)
    if isinstance(dim, tuple):
        return dim
    if dim in (None, ""):
        return ()
    return (str(dim),)


def _cpu_context(event: dict[str, Any]) -> CpuContext:
    args = event.get("args", {})
    return CpuContext(
        name=str(event.get("name") or "<unknown>"),
        input_types=tuple(str(item) for item in _as_tuple(args.get("Input type"))),
        input_dims=tuple(_normalize_dim(item) for item in _as_tuple(args.get("Input Dims"))),
    )


def _is_gemm_kernel(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in GEMM_MARKERS)


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if "devicegemmmxfp8gemmsm120" in lowered or "mxfp8" in lowered:
        return "flashinfer_mxfp8_cutlass"
    if "block_scaled" in lowered or "ue8m0xe4m3" in lowered:
        return "te_blockscaled_mxfp8"
    if "tensorop_bf16" in lowered:
        return "cutlass_bf16_tc"
    if "nvjet_sm121" in lowered:
        return "nvjet_sm121_tc"
    if "tensorop_s1688gemm" in lowered:
        return "cutlass_tc"
    if "cublas" in lowered:
        return "cublas_gemm"
    if "cutlass" in lowered:
        return "cutlass_gemm"
    return "gemm_kernel"


def _pick_cpu_context(events: Sequence[dict[str, Any]]) -> CpuContext:
    if not events:
        return CpuContext("<unlinked>", (), ())

    def score(event: dict[str, Any]) -> tuple[int, float]:
        args = event.get("args", {})
        has_inputs = int(bool(args.get("Input Dims") or args.get("Input type")))
        return has_inputs, float(event.get("dur") or 0.0)

    return _cpu_context(max(events, key=score))


def collect_gemm_groups(trace: dict[str, Any]) -> list[GemmGroup]:
    cpu_by_external_id: dict[str, list[dict[str, Any]]] = {}
    for event in trace.get("traceEvents", []):
        if event.get("cat") != "cpu_op":
            continue
        external_id = _external_id(event)
        if external_id is None:
            continue
        cpu_by_external_id.setdefault(external_id, []).append(event)

    groups: dict[tuple[Any, ...], GemmGroup] = {}
    for event in trace.get("traceEvents", []):
        if event.get("cat") != "kernel":
            continue
        name = str(event.get("name") or "")
        if not _is_gemm_kernel(name):
            continue

        external_id = _external_id(event)
        context = _pick_cpu_context(cpu_by_external_id.get(external_id or "", ()))
        kind = classify_kernel(name)
        key = (kind, context.name, context.input_types, context.input_dims)
        group = groups.get(key)
        if group is None:
            group = GemmGroup(
                kind=kind,
                cpu_op=context.name,
                input_types=context.input_types,
                input_dims=context.input_dims,
                sample_kernel=name,
            )
            groups[key] = group
        group.calls += 1
        group.total_us += float(event.get("dur") or 0.0)

    return sorted(groups.values(), key=lambda group: group.total_us, reverse=True)


def _shorten(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return f"{value[: width - 3]}..."


def _json_ready(groups: Iterable[GemmGroup]) -> list[dict[str, Any]]:
    return [
        {
            **asdict(group),
            "total_ms": round(group.total_ms, 6),
            "avg_us": round(group.avg_us, 3),
        }
        for group in groups
    ]


def render_table(groups: Sequence[GemmGroup], limit: int) -> str:
    rows = groups[:limit]
    header = (
        f"{'total_ms':>10}  {'calls':>5}  {'avg_us':>10}  "
        f"{'kind':<26}  {'cpu_op':<30}  {'input_types':<28}  {'input_dims'}"
    )
    lines = [header, "-" * len(header)]
    for group in rows:
        types = json.dumps(group.input_types, separators=(",", ":"))
        dims = json.dumps(group.input_dims, separators=(",", ":"))
        lines.append(
            f"{group.total_ms:10.3f}  {group.calls:5d}  {group.avg_us:10.1f}  "
            f"{_shorten(group.kind, 26):<26}  "
            f"{_shorten(group.cpu_op, 30):<30}  "
            f"{_shorten(types, 28):<28}  "
            f"{_shorten(dims, 96)}"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="PyTorch chrome trace JSON")
    parser.add_argument("--limit", type=int, default=40, help="Rows to print")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with args.trace.open(encoding="utf-8") as handle:
        trace = json.load(handle)
    groups = collect_gemm_groups(trace)
    if args.json:
        print(json.dumps(_json_ready(groups[: args.limit]), indent=2))
    else:
        print(render_table(groups, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
