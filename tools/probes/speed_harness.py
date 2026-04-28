"""Thin speed-comparison harness built on the existing cppmega log parser.

This module deliberately reuses ``tools.profiling.compare_bf16_mxfp8`` for
Megatron train-log parsing.  It adds only multi-run comparison glue and parsing
for Nsight Systems ``cuda_gpu_kern_sum`` CSV exports.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from cppmega.recipes.run_profiles import RunProfile
from tools.profiling.compare_bf16_mxfp8 import LogInput, RunSummary, summarize_run


@dataclass(frozen=True)
class SpeedRunInput:
    label: str
    log: Path
    extra_logs: tuple[Path, ...] = ()
    nvsmi_log: Path | None = None
    nsys_kernel_csv: Path | None = None


@dataclass(frozen=True)
class NsysKernelRecord:
    name: str
    time_pct: float
    total_time_ms: float
    instances: int
    avg_us: float | None = None
    med_us: float | None = None
    max_us: float | None = None


@dataclass(frozen=True)
class NsysKernelSummary:
    status: str
    csv_path: str | None = None
    row_count: int = 0
    top_kernels: list[NsysKernelRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SpeedRunSummary:
    label: str
    log: str
    hot_step_start: int
    hot_step_end: int | None
    hot_step_count: int
    hot_step_avg_ms: float | None
    ms_per_token: float | None
    tok_per_sec: float | None
    seq_length: int | None
    tokens_per_step: int | None
    expected_tokens_per_step: int | None
    final_train_iteration: int | None
    final_train_loss: float | None
    final_val_loss: float | None
    final_test_loss: float | None
    max_alloc_gib: float | None
    setup_alloc_gib: float | None
    skipped_iterations: int | None
    nan_iterations: int | None
    mxfp8_counters: dict[str, int]
    artifact_paths: dict[str, list[str]]
    nsys: NsysKernelSummary
    warnings: list[str]


@dataclass(frozen=True)
class SpeedDelta:
    hot_step_avg_ms_pct: float | None
    tok_per_sec_pct: float | None
    max_alloc_gib_pct: float | None
    final_train_loss_abs: float | None


@dataclass(frozen=True)
class SpeedComparisonReport:
    run_profile: str
    run_profile_description: str
    profile_tokens_per_step: int
    hot_step_start: int
    hot_step_end: int | None
    baseline_label: str
    runs: list[SpeedRunSummary]
    deltas_vs_baseline: dict[str, SpeedDelta]
    warnings: list[str]


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _pct_delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return _round((value - base) / base * 100.0, 6)


def _abs_delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return _round(value - base, 6)


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip().replace(",", "")
    if not stripped:
        return None
    return float(stripped)


def _parse_int(value: str | None) -> int:
    parsed = _parse_float(value)
    return int(parsed) if parsed is not None else 0


def parse_label_path(value: str) -> tuple[str, Path]:
    """Parse ``LABEL=PATH`` CLI values."""

    if "=" not in value:
        raise ValueError(f"expected LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label:
        raise ValueError(f"missing label in {value!r}")
    if not raw_path:
        raise ValueError(f"missing path in {value!r}")
    return label, Path(raw_path).expanduser()


def discover_nsys_kernel_csv(log: Path) -> Path | None:
    """Return the default sibling ``cuda_gpu_kern_sum`` CSV for a log, if any."""

    candidates = sorted(log.parent.glob(f"{log.stem}*cuda_gpu_kern_sum*.csv"))
    return candidates[0] if candidates else None


def parse_nsys_kernel_csv(csv_path: Path, *, top_n: int = 8) -> NsysKernelSummary:
    """Parse an Nsight Systems ``cuda_gpu_kern_sum`` CSV export."""

    path = csv_path.expanduser()
    if not path.exists():
        return NsysKernelSummary(
            status="csv_missing",
            csv_path=str(path),
            warnings=[f"nsys kernel CSV does not exist: {path}"],
        )

    rows: list[NsysKernelRecord] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            name = (raw.get("Name") or "").strip()
            if not name:
                continue
            total_ns = _parse_float(raw.get("Total Time (ns)")) or 0.0
            record = NsysKernelRecord(
                name=name,
                time_pct=_parse_float(raw.get("Time (%)")) or 0.0,
                total_time_ms=total_ns / 1_000_000.0,
                instances=_parse_int(raw.get("Instances")),
                avg_us=(
                    (_parse_float(raw.get("Avg (ns)")) or 0.0) / 1_000.0
                    if raw.get("Avg (ns)") is not None
                    else None
                ),
                med_us=(
                    (_parse_float(raw.get("Med (ns)")) or 0.0) / 1_000.0
                    if raw.get("Med (ns)") is not None
                    else None
                ),
                max_us=(
                    (_parse_float(raw.get("Max (ns)")) or 0.0) / 1_000.0
                    if raw.get("Max (ns)") is not None
                    else None
                ),
            )
            rows.append(record)

    rows.sort(key=lambda item: item.total_time_ms, reverse=True)
    if not rows:
        return NsysKernelSummary(
            status="no_kernel_rows",
            csv_path=str(path),
            warnings=[
                "nsys kernel CSV has no kernel rows; do not use this run as kernel evidence"
            ],
        )
    return NsysKernelSummary(
        status="ok",
        csv_path=str(path),
        row_count=len(rows),
        top_kernels=rows[:top_n],
    )


def _nsys_summary(input_run: SpeedRunInput, top_n: int) -> NsysKernelSummary:
    csv_path = input_run.nsys_kernel_csv or discover_nsys_kernel_csv(input_run.log)
    if csv_path is None:
        return NsysKernelSummary(status="not_requested")
    return parse_nsys_kernel_csv(csv_path, top_n=top_n)


def _speed_summary_from_run_summary(
    *,
    input_run: SpeedRunInput,
    parsed: RunSummary,
    profile: RunProfile,
    nsys: NsysKernelSummary,
) -> SpeedRunSummary:
    expected_tokens = profile.training.tokens_per_step
    tokens_per_step = parsed.tokens_per_step or expected_tokens
    hot_step_avg_ms = parsed.hot_step_avg_ms
    tok_per_sec = parsed.tok_per_sec
    if tok_per_sec is None and hot_step_avg_ms is not None and tokens_per_step:
        tok_per_sec = tokens_per_step / (hot_step_avg_ms / 1000.0)
    ms_per_token = None
    if hot_step_avg_ms is not None and tokens_per_step:
        ms_per_token = hot_step_avg_ms / tokens_per_step

    warnings = list(parsed.warnings) + list(nsys.warnings)
    if parsed.tokens_per_step is not None and parsed.tokens_per_step != expected_tokens:
        warnings.append(
            "log tokens_per_step does not match typed run profile: "
            f"log={parsed.tokens_per_step} profile={expected_tokens}"
        )
    if parsed.hot_step_count < 2:
        warnings.append(
            f"hot window has only {parsed.hot_step_count} parsed step(s); compare cautiously"
        )
    if parsed.skipped_iterations:
        warnings.append(f"skipped_iterations={parsed.skipped_iterations}")
    if parsed.nan_iterations:
        warnings.append(f"nan_iterations={parsed.nan_iterations}")

    return SpeedRunSummary(
        label=input_run.label,
        log=str(input_run.log),
        hot_step_start=parsed.hot_step_start,
        hot_step_end=parsed.hot_step_end,
        hot_step_count=parsed.hot_step_count,
        hot_step_avg_ms=parsed.hot_step_avg_ms,
        ms_per_token=_round(ms_per_token, 9),
        tok_per_sec=_round(tok_per_sec, 3),
        seq_length=parsed.seq_length,
        tokens_per_step=tokens_per_step,
        expected_tokens_per_step=expected_tokens,
        final_train_iteration=parsed.final_train_iteration,
        final_train_loss=parsed.final_train_loss,
        final_val_loss=parsed.final_val_loss,
        final_test_loss=parsed.final_test_loss,
        max_alloc_gib=parsed.max_alloc_gib,
        setup_alloc_gib=parsed.setup_alloc_gib,
        skipped_iterations=parsed.skipped_iterations,
        nan_iterations=parsed.nan_iterations,
        mxfp8_counters=parsed.mxfp8_counters,
        artifact_paths=parsed.artifact_paths,
        nsys=nsys,
        warnings=warnings,
    )


def summarize_speed_run(
    input_run: SpeedRunInput,
    *,
    profile: RunProfile,
    hot_step_start: int,
    hot_step_end: int | None,
    nsys_top_n: int = 8,
) -> SpeedRunSummary:
    parsed = summarize_run(
        LogInput(
            label=input_run.label,
            log=input_run.log,
            extra_logs=input_run.extra_logs,
            nvsmi_log=input_run.nvsmi_log,
        ),
        hot_step_start=hot_step_start,
        hot_step_end=hot_step_end,
    )
    return _speed_summary_from_run_summary(
        input_run=input_run,
        parsed=parsed,
        profile=profile,
        nsys=_nsys_summary(input_run, nsys_top_n),
    )


def build_speed_comparison_report(
    runs: Sequence[SpeedRunInput],
    *,
    profile: RunProfile,
    baseline_label: str | None = None,
    hot_step_start: int = 3,
    hot_step_end: int | None = None,
    nsys_top_n: int = 8,
) -> SpeedComparisonReport:
    if not runs:
        raise ValueError("at least one run is required")
    labels = [run.label for run in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("run labels must be unique")
    baseline = baseline_label or labels[0]
    if baseline not in labels:
        raise ValueError(f"baseline label {baseline!r} is not one of: {', '.join(labels)}")

    summaries = [
        summarize_speed_run(
            run,
            profile=profile,
            hot_step_start=hot_step_start,
            hot_step_end=hot_step_end,
            nsys_top_n=nsys_top_n,
        )
        for run in runs
    ]
    baseline_summary = next(run for run in summaries if run.label == baseline)
    deltas = {
        run.label: SpeedDelta(
            hot_step_avg_ms_pct=_pct_delta(
                run.hot_step_avg_ms, baseline_summary.hot_step_avg_ms
            ),
            tok_per_sec_pct=_pct_delta(run.tok_per_sec, baseline_summary.tok_per_sec),
            max_alloc_gib_pct=_pct_delta(run.max_alloc_gib, baseline_summary.max_alloc_gib),
            final_train_loss_abs=_abs_delta(
                run.final_train_loss, baseline_summary.final_train_loss
            ),
        )
        for run in summaries
    }
    warnings = sorted({warning for run in summaries for warning in run.warnings})
    return SpeedComparisonReport(
        run_profile=profile.name,
        run_profile_description=profile.description,
        profile_tokens_per_step=profile.training.tokens_per_step,
        hot_step_start=hot_step_start,
        hot_step_end=hot_step_end,
        baseline_label=baseline,
        runs=summaries,
        deltas_vs_baseline=deltas,
        warnings=warnings,
    )


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def _shorten(value: str, max_len: int = 88) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "..."


def render_speed_table(report: SpeedComparisonReport) -> str:
    lines = [
        f"run_profile: {report.run_profile} "
        f"(tokens_per_step={report.profile_tokens_per_step:,}, "
        f"hot_steps={report.hot_step_start}"
        f"{'' if report.hot_step_end is None else '-' + str(report.hot_step_end)})",
        f"baseline: {report.baseline_label}",
        "",
        (
            f"{'run':<14} {'steps':>5} {'hot_ms':>10} {'ms/tok':>10} "
            f"{'tok/s':>10} {'d_tok/s':>9} {'loss':>10} {'val':>10} "
            f"{'max_GiB':>9} {'nsys':>12}"
        ),
        (
            f"{'-' * 3:<14} {'-' * 5:>5} {'-' * 6:>10} {'-' * 6:>10} "
            f"{'-' * 5:>10} {'-' * 7:>9} {'-' * 4:>10} {'-' * 3:>10} "
            f"{'-' * 7:>9} {'-' * 4:>12}"
        ),
    ]
    for run in report.runs:
        delta = report.deltas_vs_baseline[run.label]
        lines.append(
            f"{run.label:<14} "
            f"{run.hot_step_count:>5} "
            f"{_fmt_float(run.hot_step_avg_ms, 1):>10} "
            f"{_fmt_float(run.ms_per_token, 6):>10} "
            f"{_fmt_float(run.tok_per_sec, 1):>10} "
            f"{_fmt_pct(delta.tok_per_sec_pct):>9} "
            f"{_fmt_float(run.final_train_loss, 6):>10} "
            f"{_fmt_float(run.final_val_loss, 6):>10} "
            f"{_fmt_float(run.max_alloc_gib, 3):>9} "
            f"{run.nsys.status:>12}"
        )

    for run in report.runs:
        if not run.nsys.top_kernels:
            continue
        lines.extend(["", f"nsys top kernels: {run.label}"])
        for kernel in run.nsys.top_kernels:
            lines.append(
                f"  {kernel.time_pct:>5.1f}% "
                f"{kernel.total_time_ms:>9.1f} ms "
                f"x{kernel.instances:<5} "
                f"{_shorten(kernel.name)}"
            )

    if report.warnings:
        lines.extend(["", "warnings:"])
        lines.extend(f"  - {warning}" for warning in report.warnings)
    return "\n".join(lines)


def render_speed_json(report: SpeedComparisonReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def build_run_inputs(
    run_values: Iterable[str],
    *,
    nsys_kernel_csv_values: Iterable[str] = (),
) -> list[SpeedRunInput]:
    csv_by_label = dict(parse_label_path(value) for value in nsys_kernel_csv_values)
    runs = []
    for value in run_values:
        label, path = parse_label_path(value)
        runs.append(
            SpeedRunInput(
                label=label,
                log=path,
                nsys_kernel_csv=csv_by_label.get(label),
            )
        )
    extra = sorted(set(csv_by_label) - {run.label for run in runs})
    if extra:
        raise ValueError(f"nsys CSV provided for unknown run label(s): {', '.join(extra)}")
    return runs
