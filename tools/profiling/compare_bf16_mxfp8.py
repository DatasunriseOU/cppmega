"""Summarize BF16 vs MXFP8 cppmega training/profiler logs.

The parser is intentionally tied to the typed ``local_gb10_quarter`` launcher
contracts: Megatron iteration lines, ``[mem_profile]`` memory hooks, and
``[torch_profile]`` artifact lines.  It also discovers sibling Nsight artifacts
created by the launcher and NCU range runs.

Profiler rule of thumb: collect torch profiler and nsys in separate runs because
CUPTI rejects multiple active subscribers.  For NCU, profile the Megatron
``cudaProfilerStart/Stop`` window by enabling the typed ``--cuda-profile`` run
profile flags rather than using env-only switches.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

try:
    from tools.probes.gb10_accepted_path_validation_helpers import (
        NUMBER_RE,
        parse_training_log as parse_accepted_path_training_log,
    )
except Exception:  # pragma: no cover - keeps the standalone script usable.
    NUMBER_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

    def parse_accepted_path_training_log(text: str) -> dict[str, Any]:
        return {"losses": {}, "counters": {}, "fallback_reasons": None}


GIB = 1024**3
MIB = 1024**2


@dataclass(frozen=True)
class LogInput:
    """All logs and explicit artifacts belonging to one precision lane."""

    label: str
    log: Path
    extra_logs: tuple[Path, ...] = ()
    nvsmi_log: Path | None = None

    def all_logs(self) -> tuple[Path, ...]:
        return (self.log, *self.extra_logs)


@dataclass(frozen=True)
class StepRecord:
    iteration: int
    total_iterations: int
    consumed_samples: int
    elapsed_ms: float
    global_batch_size: int
    lm_loss: float
    skipped_iterations: int
    nan_iterations: int
    mtp_losses: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryRecord:
    tag: str
    alloc_gib: float | None = None
    reserved_gib: float | None = None
    max_alloc_gib: float | None = None
    max_reserved_gib: float | None = None


@dataclass(frozen=True)
class Artifact:
    kind: str
    path: str
    exists: bool


@dataclass(frozen=True)
class RunSummary:
    label: str
    logs: list[str]
    hot_step_start: int
    hot_step_end: int | None
    hot_step_count: int
    hot_step_avg_ms: float | None
    tok_per_sec: float | None
    seq_length: int | None
    tokens_per_step: int | None
    setup_alloc_gib: float | None
    setup_alloc_bytes: int | None
    max_alloc_gib: float | None
    max_alloc_bytes: int | None
    param_total_numel: int | None
    param_bytes_gib: float | None
    param_bytes: int | None
    param_bytes_by_storage: dict[str, int]
    param_gib_by_storage: dict[str, float]
    final_train_iteration: int | None
    final_train_loss: float | None
    final_val_loss: float | None
    final_test_loss: float | None
    skipped_iterations: int | None
    nan_iterations: int | None
    mxfp8_counters: dict[str, int]
    artifact_paths: dict[str, list[str]]
    artifacts: list[Artifact]
    warnings: list[str]


@dataclass(frozen=True)
class ComparisonReport:
    bf16: RunSummary
    mxfp8: RunSummary
    deltas: dict[str, dict[str, float | None]]
    notes: list[str]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _existing_path(path: Path) -> str:
    return str(path.expanduser())


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _gib_to_bytes(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(value * GIB))


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return value / GIB


def _parse_int_with_commas(value: str) -> int:
    return int(value.replace(",", ""))


def _unique_artifacts(items: Iterable[Artifact]) -> list[Artifact]:
    seen: set[tuple[str, str]] = set()
    out: list[Artifact] = []
    for item in items:
        key = (item.kind, item.path)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return sorted(out, key=lambda item: (item.kind, item.path))


def _group_artifacts(artifacts: Iterable[Artifact]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for artifact in artifacts:
        grouped.setdefault(artifact.kind, []).append(artifact.path)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def _add_artifact(items: list[Artifact], kind: str, path: Path | str) -> None:
    path_obj = Path(path).expanduser()
    items.append(Artifact(kind=kind, path=str(path_obj), exists=path_obj.exists()))


def _infer_sibling_artifacts(log: Path) -> list[Artifact]:
    artifacts: list[Artifact] = []
    if log.suffix == ".log":
        nvsmi = log.with_name(f"{log.stem}.nvsmi.log")
        if nvsmi.exists():
            _add_artifact(artifacts, "nvsmi_log", nvsmi)

        torch_dir = log.with_name(f"{log.stem}_torch_profile")
        if torch_dir.is_dir():
            for trace in sorted(torch_dir.glob("*.json")):
                _add_artifact(artifacts, "torch_trace", trace)
            for table in sorted(torch_dir.glob("*_cuda_table.txt")):
                _add_artifact(artifacts, "torch_table", table)

        for suffix, kind in (
            ("_nsys.nsys-rep", "nsys_report"),
            ("_nsys.sqlite", "nsys_sqlite"),
            (".ncu-rep", "ncu_report"),
            ("_ncu.log", "ncu_log"),
        ):
            candidate = log.with_name(f"{log.stem}{suffix}")
            if candidate.exists():
                _add_artifact(artifacts, kind, candidate)
    return artifacts


def _resolve_maybe_relative(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path.strip())
    if path.is_absolute():
        return path
    return base_dir / path


def _discover_artifacts_from_text(path: Path, text: str) -> list[Artifact]:
    artifacts: list[Artifact] = []
    base_dir = path.parent

    for match in re.finditer(r"\[torch_profile\]\s+step=\d+\s+trace=(\S+)\s+table=(\S+)", text):
        _add_artifact(artifacts, "torch_trace", _resolve_maybe_relative(match.group(1), base_dir))
        _add_artifact(artifacts, "torch_table", _resolve_maybe_relative(match.group(2), base_dir))

    for match in re.finditer(r"(?<!\S)(\S+\.nsys-rep)\b", text):
        raw = match.group(1).strip().rstrip(".,)")
        _add_artifact(artifacts, "nsys_report", _resolve_maybe_relative(raw, base_dir))

    for match in re.finditer(r"(?<!\S)(\S+\.sqlite)\b", text):
        raw = match.group(1).strip().rstrip(".,)")
        if "nsys" in raw:
            _add_artifact(artifacts, "nsys_sqlite", _resolve_maybe_relative(raw, base_dir))

    for match in re.finditer(r"(?<!\S)(\S+\.ncu-rep)\b", text):
        raw = match.group(1).strip().rstrip(".,)")
        _add_artifact(artifacts, "ncu_report", _resolve_maybe_relative(raw, base_dir))

    return artifacts


def _parse_seq_length(text: str) -> int | None:
    patterns = (
        rf"^\s*seq_length\s+\.+\s+(\d+)\s*$",
        rf"^\s*max_position_embeddings\s+\.+\s+(\d+)\s*$",
        r"\[local-quarter\].*?\bseq(?:_length)?=(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return int(match.group(1))
    return None


def _parse_steps(text: str) -> list[StepRecord]:
    records: list[StepRecord] = []
    iter_re = re.compile(
        rf"iteration\s+(?P<iteration>\d+)\s*/\s*(?P<total>\d+).*?"
        rf"consumed samples:\s*(?P<samples>\d+).*?"
        rf"elapsed time per iteration \(ms\):\s*(?P<elapsed>{NUMBER_RE}).*?"
        rf"global batch size:\s*(?P<gbs>\d+).*?"
        rf"lm loss:\s*(?P<lm>{NUMBER_RE}).*?"
        rf"number of skipped iterations:\s*(?P<skipped>\d+).*?"
        rf"number of nan iterations:\s*(?P<nan>\d+)",
    )
    mtp_re = re.compile(rf"\b(?P<name>mtp_\d+)\s+loss:\s*(?P<loss>{NUMBER_RE})")
    for line in text.splitlines():
        match = iter_re.search(line)
        if not match:
            continue
        records.append(
            StepRecord(
                iteration=int(match.group("iteration")),
                total_iterations=int(match.group("total")),
                consumed_samples=int(match.group("samples")),
                elapsed_ms=float(match.group("elapsed")),
                global_batch_size=int(match.group("gbs")),
                lm_loss=float(match.group("lm")),
                skipped_iterations=int(match.group("skipped")),
                nan_iterations=int(match.group("nan")),
                mtp_losses={
                    mtp.group("name"): float(mtp.group("loss"))
                    for mtp in mtp_re.finditer(line)
                },
            )
        )
    return records


def _parse_memory_records(text: str) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    mem_re = re.compile(
        rf"\[mem_profile\]\s+(?P<tag>[^:]+):\s+"
        rf"alloc=(?P<alloc>{NUMBER_RE})\s+GiB\s+"
        rf"reserved=(?P<reserved>{NUMBER_RE})\s+GiB\s+"
        rf"max_alloc=(?P<max_alloc>{NUMBER_RE})\s+GiB\s+"
        rf"max_reserved=(?P<max_reserved>{NUMBER_RE})\s+GiB"
    )
    rank_re = re.compile(
        rf"\[Rank\s+\d+\].*?memory \(MB\) \| "
        rf"allocated:\s*(?P<alloc>{NUMBER_RE}) \| "
        rf"max allocated:\s*(?P<max_alloc>{NUMBER_RE}) \| "
        rf"reserved:\s*(?P<reserved>{NUMBER_RE}) \| "
        rf"max reserved:\s*(?P<max_reserved>{NUMBER_RE})"
    )
    for line in text.splitlines():
        match = mem_re.search(line)
        if match:
            records.append(
                MemoryRecord(
                    tag=match.group("tag"),
                    alloc_gib=float(match.group("alloc")),
                    reserved_gib=float(match.group("reserved")),
                    max_alloc_gib=float(match.group("max_alloc")),
                    max_reserved_gib=float(match.group("max_reserved")),
                )
            )
            continue
        match = rank_re.search(line)
        if match:
            records.append(
                MemoryRecord(
                    tag="rank_memory",
                    alloc_gib=float(match.group("alloc")) * MIB / GIB,
                    reserved_gib=float(match.group("reserved")) * MIB / GIB,
                    max_alloc_gib=float(match.group("max_alloc")) * MIB / GIB,
                    max_reserved_gib=float(match.group("max_reserved")) * MIB / GIB,
                )
            )
    return records


def _parse_param_breakdown(text: str) -> tuple[int | None, int | None, dict[str, int]]:
    total_numel: int | None = None
    param_bytes: int | None = None
    by_storage: dict[str, int] = {}

    total_match = re.search(
        rf"\[mem_profile\]\s+total_params=([\d,]+)\s+param_bytes=({NUMBER_RE})\s+GiB",
        text,
    )
    if total_match:
        total_numel = _parse_int_with_commas(total_match.group(1))
        param_bytes = _gib_to_bytes(float(total_match.group(2)))

    in_storage = False
    storage_re = re.compile(
        rf"\[mem_profile\]\s+(?P<storage>.+?)\s+"
        rf"(?P<numel>[\d,]+)\s+elems\s+(?P<gib>{NUMBER_RE})\s+GiB"
    )
    for line in text.splitlines():
        if line.startswith("[mem_profile] by_storage:"):
            in_storage = True
            continue
        if in_storage and line.startswith("[mem_profile] top_parameters:"):
            break
        if not in_storage:
            continue
        match = storage_re.search(line)
        if not match:
            continue
        by_storage[match.group("storage").strip()] = _gib_to_bytes(float(match.group("gib"))) or 0

    return total_numel, param_bytes, by_storage


def _parse_eval_loss(text: str, split: Literal["validation", "test"]) -> float | None:
    pattern = (
        rf"validation loss at iteration\s+\d+\s+on\s+{split}\s+set\s+\|\s+"
        rf"lm loss value:\s*({NUMBER_RE})"
    )
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def _parse_nvsmi_peak_mib(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    peak: int | None = None
    for line in _read_text(path).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            used = int(float(parts[1]))
        except ValueError:
            continue
        peak = used if peak is None else max(peak, used)
    return peak


def _build_warning_list(text: str, artifacts: Iterable[Artifact]) -> list[str]:
    warnings: list[str] = []
    if "CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED" in text:
        warnings.append(
            "CUPTI multiple subscribers detected: do not combine torch profiler "
            "and nsys in one run; collect them separately."
        )
    missing = [artifact for artifact in artifacts if not artifact.exists]
    if missing:
        warnings.append(f"{len(missing)} referenced profiler artifact path(s) do not exist")
    return warnings


def summarize_run(inputs: LogInput, hot_step_start: int, hot_step_end: int | None) -> RunSummary:
    log_texts: list[str] = []
    artifacts: list[Artifact] = []
    log_paths: list[str] = []

    for log_path in inputs.all_logs():
        log_paths.append(_existing_path(log_path))
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        text = _read_text(log_path)
        log_texts.append(text)
        _add_artifact(artifacts, "log", log_path)
        artifacts.extend(_infer_sibling_artifacts(log_path))
        artifacts.extend(_discover_artifacts_from_text(log_path, text))

    nvsmi_log = inputs.nvsmi_log
    if nvsmi_log is None and inputs.log.suffix == ".log":
        candidate = inputs.log.with_name(f"{inputs.log.stem}.nvsmi.log")
        if candidate.exists():
            nvsmi_log = candidate
    if nvsmi_log is not None:
        _add_artifact(artifacts, "nvsmi_log", nvsmi_log)

    text = "\n".join(log_texts)
    steps = _parse_steps(text)
    memory = _parse_memory_records(text)
    seq_length = _parse_seq_length(text)
    total_numel, param_bytes, by_storage = _parse_param_breakdown(text)
    accepted_path = parse_accepted_path_training_log(text)
    artifacts = _unique_artifacts(artifacts)

    final_step = steps[-1] if steps else None
    hot_steps = [
        record
        for record in steps
        if record.iteration >= hot_step_start
        and (hot_step_end is None or record.iteration <= hot_step_end)
    ]
    hot_avg_ms = statistics.fmean(record.elapsed_ms for record in hot_steps) if hot_steps else None
    final_gbs = final_step.global_batch_size if final_step is not None else None
    tokens_per_step = seq_length * final_gbs if seq_length is not None and final_gbs else None
    tok_per_sec = None
    if hot_steps and tokens_per_step is not None:
        tok_per_sec = statistics.fmean(
            tokens_per_step / (record.elapsed_ms / 1000.0) for record in hot_steps
        )

    setup_record = next(
        (record for record in memory if record.tag == "after_setup_model_and_optimizer"),
        None,
    )
    max_alloc_values = [record.max_alloc_gib for record in memory if record.max_alloc_gib is not None]
    max_alloc_gib = max(max_alloc_values) if max_alloc_values else None

    # nvidia-smi is not the allocator max, but keeping it as an artifact-derived
    # warning source makes missing monitor logs visible in JSON without adding a
    # separate top-level metric.
    _parse_nvsmi_peak_mib(nvsmi_log)

    warnings = _build_warning_list(text, artifacts)
    counters = accepted_path.get("counters", {})
    if not isinstance(counters, dict):
        counters = {}

    return RunSummary(
        label=inputs.label,
        logs=log_paths,
        hot_step_start=hot_step_start,
        hot_step_end=hot_step_end,
        hot_step_count=len(hot_steps),
        hot_step_avg_ms=_round_float(hot_avg_ms, 3),
        tok_per_sec=_round_float(tok_per_sec, 3),
        seq_length=seq_length,
        tokens_per_step=tokens_per_step,
        setup_alloc_gib=_round_float(setup_record.alloc_gib if setup_record else None, 3),
        setup_alloc_bytes=_gib_to_bytes(setup_record.alloc_gib if setup_record else None),
        max_alloc_gib=_round_float(max_alloc_gib, 3),
        max_alloc_bytes=_gib_to_bytes(max_alloc_gib),
        param_total_numel=total_numel,
        param_bytes_gib=_round_float(_bytes_to_gib(param_bytes), 3),
        param_bytes=param_bytes,
        param_bytes_by_storage=by_storage,
        param_gib_by_storage={
            key: _round_float(value / GIB, 3) or 0.0 for key, value in sorted(by_storage.items())
        },
        final_train_iteration=final_step.iteration if final_step else None,
        final_train_loss=_round_float(final_step.lm_loss if final_step else None, 6),
        final_val_loss=_round_float(_parse_eval_loss(text, "validation"), 6),
        final_test_loss=_round_float(_parse_eval_loss(text, "test"), 6),
        skipped_iterations=final_step.skipped_iterations if final_step else None,
        nan_iterations=final_step.nan_iterations if final_step else None,
        mxfp8_counters={str(key): int(value) for key, value in counters.items()},
        artifact_paths=_group_artifacts(artifacts),
        artifacts=artifacts,
        warnings=warnings,
    )


def _delta(new: float | int | None, base: float | int | None) -> dict[str, float | None]:
    if new is None or base is None:
        return {"abs": None, "pct": None}
    absolute = float(new) - float(base)
    pct = None if float(base) == 0.0 else absolute / float(base) * 100.0
    return {"abs": _round_float(absolute, 6), "pct": _round_float(pct, 6)}


def build_comparison_report(
    bf16: LogInput,
    mxfp8: LogInput,
    hot_step_start: int,
    hot_step_end: int | None,
) -> ComparisonReport:
    bf16_summary = summarize_run(bf16, hot_step_start, hot_step_end)
    mxfp8_summary = summarize_run(mxfp8, hot_step_start, hot_step_end)
    deltas = {
        "hot_step_avg_ms": _delta(mxfp8_summary.hot_step_avg_ms, bf16_summary.hot_step_avg_ms),
        "tok_per_sec": _delta(mxfp8_summary.tok_per_sec, bf16_summary.tok_per_sec),
        "setup_alloc_gib": _delta(mxfp8_summary.setup_alloc_gib, bf16_summary.setup_alloc_gib),
        "max_alloc_gib": _delta(mxfp8_summary.max_alloc_gib, bf16_summary.max_alloc_gib),
        "param_bytes_gib": _delta(mxfp8_summary.param_bytes_gib, bf16_summary.param_bytes_gib),
        "final_train_loss": _delta(mxfp8_summary.final_train_loss, bf16_summary.final_train_loss),
        "final_val_loss": _delta(mxfp8_summary.final_val_loss, bf16_summary.final_val_loss),
        "final_test_loss": _delta(mxfp8_summary.final_test_loss, bf16_summary.final_test_loss),
    }
    notes = [
        "torch profiler and nsys must be separate runs; CUPTI allows only one active subscriber",
        "NCU range runs should use Megatron cudaProfilerStart/Stop via typed cuda_profile fields",
    ]
    return ComparisonReport(bf16=bf16_summary, mxfp8=mxfp8_summary, deltas=deltas, notes=notes)


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def _format_delta(delta: dict[str, float | None], suffix: str = "") -> str:
    absolute = delta.get("abs")
    pct = delta.get("pct")
    if absolute is None:
        return "-"
    abs_str = f"{absolute:+.3f}{suffix}"
    if pct is None:
        return abs_str
    return f"{abs_str} ({pct:+.2f}%)"


def _table_row(name: str, bf16: str, mxfp8: str, delta: str) -> str:
    return f"{name:<28} {bf16:>16} {mxfp8:>16} {delta:>20}"


def render_table(report: ComparisonReport) -> str:
    bf16 = report.bf16
    mxfp8 = report.mxfp8
    lines = [
        _table_row("metric", "bf16", "mxfp8", "mxfp8-bf16"),
        _table_row("-" * 6, "-" * 4, "-" * 5, "-" * 10),
        _table_row(
            "hot_step_avg_ms",
            _format_float(bf16.hot_step_avg_ms),
            _format_float(mxfp8.hot_step_avg_ms),
            _format_delta(report.deltas["hot_step_avg_ms"], " ms"),
        ),
        _table_row(
            "tok_per_sec",
            _format_float(bf16.tok_per_sec, 1),
            _format_float(mxfp8.tok_per_sec, 1),
            _format_delta(report.deltas["tok_per_sec"], " tok/s"),
        ),
        _table_row(
            "setup_alloc_gib",
            _format_float(bf16.setup_alloc_gib),
            _format_float(mxfp8.setup_alloc_gib),
            _format_delta(report.deltas["setup_alloc_gib"], " GiB"),
        ),
        _table_row(
            "max_alloc_gib",
            _format_float(bf16.max_alloc_gib),
            _format_float(mxfp8.max_alloc_gib),
            _format_delta(report.deltas["max_alloc_gib"], " GiB"),
        ),
        _table_row(
            "param_bytes_gib",
            _format_float(bf16.param_bytes_gib),
            _format_float(mxfp8.param_bytes_gib),
            _format_delta(report.deltas["param_bytes_gib"], " GiB"),
        ),
        _table_row(
            "final_train_loss",
            _format_float(bf16.final_train_loss, 6),
            _format_float(mxfp8.final_train_loss, 6),
            _format_delta(report.deltas["final_train_loss"]),
        ),
        _table_row(
            "final_val_loss",
            _format_float(bf16.final_val_loss, 6),
            _format_float(mxfp8.final_val_loss, 6),
            _format_delta(report.deltas["final_val_loss"]),
        ),
        _table_row(
            "final_test_loss",
            _format_float(bf16.final_test_loss, 6),
            _format_float(mxfp8.final_test_loss, 6),
            _format_delta(report.deltas["final_test_loss"]),
        ),
        _table_row(
            "skipped_iterations",
            _format_int(bf16.skipped_iterations),
            _format_int(mxfp8.skipped_iterations),
            "-",
        ),
        _table_row(
            "nan_iterations",
            _format_int(bf16.nan_iterations),
            _format_int(mxfp8.nan_iterations),
            "-",
        ),
    ]

    storage_names = sorted(
        set(bf16.param_gib_by_storage) | set(mxfp8.param_gib_by_storage)
    )
    if storage_names:
        lines.extend(["", "param bytes by storage (GiB):"])
        for storage in storage_names:
            lines.append(
                _table_row(
                    f"  {storage}",
                    _format_float(bf16.param_gib_by_storage.get(storage)),
                    _format_float(mxfp8.param_gib_by_storage.get(storage)),
                    _format_delta(
                        _delta(
                            mxfp8.param_gib_by_storage.get(storage),
                            bf16.param_gib_by_storage.get(storage),
                        ),
                        " GiB",
                    ),
                )
            )

    lines.extend(["", "artifacts:"])
    for summary in (bf16, mxfp8):
        lines.append(f"  {summary.label}:")
        for kind, paths in summary.artifact_paths.items():
            if kind == "log":
                continue
            joined = ", ".join(paths) if paths else "-"
            lines.append(f"    {kind}: {joined}")

    warnings = [*bf16.warnings, *mxfp8.warnings]
    if warnings:
        lines.extend(["", "warnings:"])
        for warning in sorted(set(warnings)):
            lines.append(f"  - {warning}")

    lines.extend(["", "notes:"])
    for note in report.notes:
        lines.append(f"  - {note}")
    return "\n".join(lines)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def render_json(report: ComparisonReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True, default=_json_default)


def _path_list(values: list[str] | None) -> tuple[Path, ...]:
    return tuple(Path(value).expanduser() for value in (values or ()))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bf16-log", type=Path, required=True)
    parser.add_argument("--mxfp8-log", type=Path, required=True)
    parser.add_argument(
        "--bf16-extra-log",
        action="append",
        default=[],
        help="Additional BF16 profiler log to scan for artifact paths.",
    )
    parser.add_argument(
        "--mxfp8-extra-log",
        action="append",
        default=[],
        help="Additional MXFP8 profiler log to scan for artifact paths.",
    )
    parser.add_argument("--bf16-nvsmi-log", type=Path, default=None)
    parser.add_argument("--mxfp8-nvsmi-log", type=Path, default=None)
    parser.add_argument("--hot-step-start", type=int, default=3)
    parser.add_argument("--hot-step-end", type=int, default=None)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.hot_step_start < 1:
        raise SystemExit("--hot-step-start must be >= 1")
    if args.hot_step_end is not None and args.hot_step_end < args.hot_step_start:
        raise SystemExit("--hot-step-end must be >= --hot-step-start")

    report = build_comparison_report(
        bf16=LogInput(
            label="bf16",
            log=args.bf16_log.expanduser(),
            extra_logs=_path_list(args.bf16_extra_log),
            nvsmi_log=args.bf16_nvsmi_log.expanduser() if args.bf16_nvsmi_log else None,
        ),
        mxfp8=LogInput(
            label="mxfp8",
            log=args.mxfp8_log.expanduser(),
            extra_logs=_path_list(args.mxfp8_extra_log),
            nvsmi_log=args.mxfp8_nvsmi_log.expanduser() if args.mxfp8_nvsmi_log else None,
        ),
        hot_step_start=args.hot_step_start,
        hot_step_end=args.hot_step_end,
    )
    if args.format == "json":
        print(render_json(report))
    else:
        print(render_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
