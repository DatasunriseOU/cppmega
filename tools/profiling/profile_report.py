"""Build a concise cppmega training/profiler report.

The core training metrics come from the typed launcher log format parsed by
``compare_bf16_mxfp8``.  Profiler details are plain-text summaries that are
easy to archive with a 100-step run:

* torch profiler ``*_cuda_table.txt`` tables
* Nsight Systems ``cuda_gpu_kern_sum`` and ``cuda_gpu_mem_sum`` reports
* Nsight Compute detail exports with Speed Of Light metrics

Run example:

    python tools/profiling/profile_report.py \
      --log /home/dave/logs/gb10_quarter_e2e100_baseline_20260429.log
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cppmega.recipes.run_profiles import get_run_profile
from tools.profiling.compare_bf16_mxfp8 import LogInput, RunSummary, summarize_run


NUMBER_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
DECIMAL_MB = 1_000_000


@dataclass(frozen=True)
class TorchOp:
    name: str
    cuda_total_ms: float
    self_cuda_ms: float
    self_cuda_pct: float
    calls: int
    table: str


@dataclass(frozen=True)
class NsysKernel:
    name: str
    time_pct: float
    total_time_s: float
    instances: int
    avg_ms: float
    source: str


@dataclass(frozen=True)
class MemOp:
    operation: str
    total_bytes: int
    total_mb: float
    count: int
    source: str


@dataclass(frozen=True)
class NcuKernel:
    name: str
    count: int
    avg_duration_ms: float | None
    max_duration_ms: float | None
    avg_compute_pct: float | None
    avg_memory_pct: float | None
    source: str


@dataclass(frozen=True)
class ProfileReport:
    label: str
    run_profile: str
    training: RunSummary
    torch_ops: list[TorchOp]
    nsys_kernels: list[NsysKernel]
    memops: list[MemOp]
    ncu_kernels: list[NcuKernel]
    artifacts: list[str]
    warnings: list[str]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _round(value: float | None, digits: int = 3) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _parse_time_ms(raw: str) -> float | None:
    raw = raw.strip()
    match = re.fullmatch(rf"({NUMBER_RE})\s*(ns|us|ms|s)", raw)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "ns":
        return value / 1_000_000.0
    if unit == "us":
        return value / 1000.0
    if unit == "s":
        return value * 1000.0
    return value


def _parse_pct(raw: str) -> float | None:
    try:
        return float(raw.strip().rstrip("%"))
    except ValueError:
        return None


def _parse_int(raw: str) -> int | None:
    try:
        return int(raw.strip().replace(",", ""))
    except ValueError:
        return None


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        key = resolved.resolve() if resolved.exists() else resolved.absolute()
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _profile_group(log: Path) -> tuple[str, str] | None:
    stem = log.stem
    for marker in ("baseline", "torchprof", "nsys", "ncu"):
        token = f"_{marker}_"
        if token not in stem:
            continue
        prefix, suffix = stem.split(token, 1)
        date_match = re.search(r"(\d{8})(?:_\d{6})?", suffix)
        if date_match:
            return prefix, date_match.group(1)
    return None


def _discover_related(log: Path, explicit: Sequence[Path], globs: Sequence[str]) -> list[Path]:
    paths: list[Path] = [log, *explicit]

    for pattern in globs:
        paths.extend(Path(value) for value in glob.glob(pattern))

    if log.suffix == ".log":
        paths.append(log.with_name(f"{log.stem}.nvsmi.log"))
        paths.append(log.with_name(f"{log.stem}_cuda_gpu_kern_sum.txt"))
        paths.append(log.with_name(f"{log.stem}_cuda_gpu_mem_sum.txt"))
        paths.append(log.with_name(f"{log.stem}_details.txt"))
        paths.append(log.with_name(f"{log.stem}_torch_profile"))

    group = _profile_group(log)
    if group is not None:
        prefix, date = group
        for candidate in log.parent.glob(f"{prefix}*{date}*"):
            if candidate.is_dir() and "torch_profile" in candidate.name:
                paths.append(candidate)
                continue
            if not candidate.is_file():
                continue
            name = candidate.name
            if any(
                needle in name
                for needle in (
                    "torchprof",
                    "cuda_gpu_kern_sum",
                    "cuda_gpu_mem_sum",
                    "ncu",
                    "nsys",
                    "details",
                )
            ):
                paths.append(candidate)

    discovered: list[Path] = []
    for path in paths:
        if path.is_dir():
            discovered.append(path)
            discovered.extend(sorted(path.glob("*_cuda_table.txt")))
        elif path.exists():
            discovered.append(path)
    return _dedupe_paths(discovered)


def _torch_table_step(path: Path) -> tuple[int, str]:
    match = re.search(r"train_step_(\d+)_cuda_table", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def _parse_torch_table(path: Path) -> list[TorchOp]:
    rows: list[TorchOp] = []
    for line in _read_text(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("Name"):
            continue
        columns = re.split(r"\s{2,}", stripped)
        if len(columns) < 14:
            continue
        self_cuda_ms = _parse_time_ms(columns[6])
        self_cuda_pct = _parse_pct(columns[7])
        cuda_total_ms = _parse_time_ms(columns[8])
        calls = _parse_int(columns[-1])
        if cuda_total_ms is None or self_cuda_ms is None or self_cuda_pct is None or calls is None:
            continue
        if cuda_total_ms <= 0 and self_cuda_ms <= 0:
            continue
        rows.append(
            TorchOp(
                name=columns[0],
                cuda_total_ms=_round(cuda_total_ms, 3) or 0.0,
                self_cuda_ms=_round(self_cuda_ms, 3) or 0.0,
                self_cuda_pct=_round(self_cuda_pct, 3) or 0.0,
                calls=calls,
                table=str(path),
            )
        )
    return sorted(rows, key=lambda row: row.cuda_total_ms, reverse=True)


def _parse_nsys_kernels(path: Path) -> list[NsysKernel]:
    rows: list[NsysKernel] = []
    for line in _read_text(path).splitlines():
        columns = re.split(r"\s{2,}", line.strip(), maxsplit=8)
        if len(columns) != 9:
            continue
        try:
            time_pct = float(columns[0])
            total_time_ns = int(columns[1].replace(",", ""))
            instances = int(columns[2].replace(",", ""))
            avg_ns = float(columns[3].replace(",", ""))
        except ValueError:
            continue
        rows.append(
            NsysKernel(
                name=columns[8],
                time_pct=_round(time_pct, 3) or 0.0,
                total_time_s=_round(total_time_ns / 1_000_000_000.0, 6) or 0.0,
                instances=instances,
                avg_ms=_round(avg_ns / 1_000_000.0, 3) or 0.0,
                source=str(path),
            )
        )
    return sorted(rows, key=lambda row: row.total_time_s, reverse=True)


def _parse_memops(path: Path) -> list[MemOp]:
    rows: list[MemOp] = []
    in_size_table = False
    for line in _read_text(path).splitlines():
        if "CUDA GPU MemOps Summary (by Size)" in line:
            in_size_table = True
            continue
        if not in_size_table:
            continue
        columns = re.split(r"\s{2,}", line.strip(), maxsplit=7)
        if len(columns) != 8:
            continue
        try:
            total_mb = float(columns[0].replace(",", ""))
            count = int(columns[1].replace(",", ""))
        except ValueError:
            continue
        rows.append(
            MemOp(
                operation=columns[7],
                total_bytes=int(round(total_mb * DECIMAL_MB)),
                total_mb=_round(total_mb, 3) or 0.0,
                count=count,
                source=str(path),
            )
        )
    return sorted(rows, key=lambda row: row.total_bytes, reverse=True)


def _kernel_name_from_ncu_header(line: str) -> str | None:
    stripped = line.strip()
    if ", Context " not in stripped or ", Stream " not in stripped or ", Device " not in stripped:
        return None
    launch = stripped.split(", Context ", 1)[0]
    match = re.match(r"(?P<name>.+)\s+\([^()]*\)x\([^()]*\)$", launch)
    return match.group("name") if match else launch


def _duration_to_ms(unit: str, value: float) -> float:
    if unit == "ns":
        return value / 1_000_000.0
    if unit == "us":
        return value / 1000.0
    if unit == "s":
        return value * 1000.0
    return value


def _parse_ncu_details(path: Path) -> list[NcuKernel]:
    samples: dict[str, dict[str, list[float] | str]] = {}
    current: str | None = None
    duration_re = re.compile(rf"^Duration\s+(ns|us|ms|s)\s+({NUMBER_RE})$")
    pct_re = re.compile(rf"^(Memory Throughput|Compute \(SM\) Throughput)\s+%\s+({NUMBER_RE})$")

    for line in _read_text(path).splitlines():
        name = _kernel_name_from_ncu_header(line)
        if name is not None:
            current = name
            samples.setdefault(
                current,
                {"duration_ms": [], "compute_pct": [], "memory_pct": [], "source": str(path)},
            )
            continue
        if current is None:
            continue
        stripped = line.strip()
        duration_match = duration_re.match(stripped)
        if duration_match:
            bucket = samples[current]["duration_ms"]
            assert isinstance(bucket, list)
            bucket.append(_duration_to_ms(duration_match.group(1), float(duration_match.group(2))))
            continue
        pct_match = pct_re.match(stripped)
        if pct_match:
            key = "memory_pct" if pct_match.group(1) == "Memory Throughput" else "compute_pct"
            bucket = samples[current][key]
            assert isinstance(bucket, list)
            bucket.append(float(pct_match.group(2)))

    kernels: list[NcuKernel] = []
    for name, values in samples.items():
        durations = values["duration_ms"]
        compute = values["compute_pct"]
        memory = values["memory_pct"]
        source = values["source"]
        assert isinstance(durations, list)
        assert isinstance(compute, list)
        assert isinstance(memory, list)
        assert isinstance(source, str)
        kernels.append(
            NcuKernel(
                name=name,
                count=max(len(durations), len(compute), len(memory)),
                avg_duration_ms=_round(statistics.fmean(durations), 3) if durations else None,
                max_duration_ms=_round(max(durations), 3) if durations else None,
                avg_compute_pct=_round(statistics.fmean(compute), 3) if compute else None,
                avg_memory_pct=_round(statistics.fmean(memory), 3) if memory else None,
                source=source,
            )
        )
    return sorted(
        kernels,
        key=lambda row: ((row.avg_duration_ms or 0.0) * row.count, row.count),
        reverse=True,
    )


def _parse_artifacts(paths: Sequence[Path]) -> tuple[list[TorchOp], list[NsysKernel], list[MemOp], list[NcuKernel]]:
    torch_tables: list[Path] = []
    nsys_kernel_paths: list[Path] = []
    memop_paths: list[Path] = []
    ncu_paths: list[Path] = []

    for path in paths:
        if path.is_dir():
            torch_tables.extend(path.glob("*_cuda_table.txt"))
            continue
        name = path.name
        if name.endswith("_cuda_table.txt"):
            torch_tables.append(path)
        if "cuda_gpu_kern_sum" in name:
            nsys_kernel_paths.append(path)
        if "cuda_gpu_mem_sum" in name:
            memop_paths.append(path)
        if name.endswith("_details.txt"):
            ncu_paths.append(path)

    torch_ops: list[TorchOp] = []
    if torch_tables:
        torch_ops = _parse_torch_table(max(_dedupe_paths(torch_tables), key=_torch_table_step))

    nsys_kernels: list[NsysKernel] = []
    for path in _dedupe_paths(nsys_kernel_paths):
        nsys_kernels.extend(_parse_nsys_kernels(path))

    memops: list[MemOp] = []
    for path in _dedupe_paths(memop_paths):
        memops.extend(_parse_memops(path))

    ncu_kernels: list[NcuKernel] = []
    for path in _dedupe_paths(ncu_paths):
        ncu_kernels.extend(_parse_ncu_details(path))

    return (
        sorted(torch_ops, key=lambda row: row.cuda_total_ms, reverse=True),
        sorted(nsys_kernels, key=lambda row: row.total_time_s, reverse=True),
        sorted(memops, key=lambda row: row.total_bytes, reverse=True),
        sorted(
            ncu_kernels,
            key=lambda row: ((row.avg_duration_ms or 0.0) * row.count, row.count),
            reverse=True,
        ),
    )


def build_profile_report(
    log: Path,
    *,
    label: str | None = None,
    run_profile: str = "local_gb10_quarter",
    artifacts: Sequence[Path] = (),
    artifact_globs: Sequence[str] = (),
    auto_discover: bool = True,
    hot_step_start: int = 3,
    hot_step_end: int | None = None,
) -> ProfileReport:
    profile = get_run_profile(run_profile)
    discovered = (
        _discover_related(log, artifacts, artifact_globs)
        if auto_discover
        else _dedupe_paths(
            [
                log,
                *artifacts,
                *(
                    Path(value)
                    for pattern in artifact_globs
                    for value in glob.glob(pattern)
                ),
            ]
        )
    )
    training = summarize_run(
        LogInput(label=label or log.stem, log=log),
        hot_step_start=hot_step_start,
        hot_step_end=hot_step_end,
    )
    if training.tokens_per_step is None:
        tokens_per_step = profile.training.tokens_per_step
        tok_per_sec = None
        if training.hot_step_avg_ms:
            tok_per_sec = tokens_per_step / (training.hot_step_avg_ms / 1000.0)
        training = replace(
            training,
            seq_length=profile.training.seq_length,
            tokens_per_step=tokens_per_step,
            tok_per_sec=_round(tok_per_sec, 3),
        )

    torch_ops, nsys_kernels, memops, ncu_kernels = _parse_artifacts(discovered)
    warnings = list(training.warnings)
    if not torch_ops:
        warnings.append("no torch profiler CUDA table found")
    if not nsys_kernels:
        warnings.append("no nsys cuda_gpu_kern_sum text found")
    if not memops:
        warnings.append("no nsys cuda_gpu_mem_sum text found")
    if not ncu_kernels:
        warnings.append("no ncu detail text found")

    return ProfileReport(
        label=label or log.stem,
        run_profile=run_profile,
        training=training,
        torch_ops=torch_ops,
        nsys_kernels=nsys_kernels,
        memops=memops,
        ncu_kernels=ncu_kernels,
        artifacts=[str(path) for path in discovered],
        warnings=sorted(set(warnings)),
    )


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    tb = value / 1_000_000_000_000.0
    if tb >= 0.1:
        return f"{tb:.3f} TB"
    gb = value / 1_000_000_000.0
    if gb >= 0.1:
        return f"{gb:.3f} GB"
    mb = value / 1_000_000.0
    return f"{mb:.3f} MB"


def _short_name(name: str, width: int = 88) -> str:
    if len(name) <= width:
        return name
    return name[: width - 3] + "..."


def render_table(report: ProfileReport, *, top_n: int = 8) -> str:
    training = report.training
    d2d = next((row for row in report.memops if "Device-to-Device" in row.operation), None)
    lines = [
        f"profile report: {report.label} ({report.run_profile})",
        "",
        "metric                         value",
        "-----------------------------  ----------------",
        f"steady avg ms                  {_fmt(training.hot_step_avg_ms, 1)}",
        f"tokens/sec                     {_fmt(training.tok_per_sec, 1)}",
        f"final train loss               {_fmt(training.final_train_loss, 6)}",
        f"final val loss                 {_fmt(training.final_val_loss, 6)}",
        f"final test loss                {_fmt(training.final_test_loss, 6)}",
        f"max alloc                      {_fmt(training.max_alloc_gib, 3)} GiB",
        f"D2D copy bytes                 {_fmt_bytes(d2d.total_bytes if d2d else None)}",
        f"steady steps                   {training.hot_step_start}-{training.hot_step_end or training.final_train_iteration or '?'} ({training.hot_step_count})",
    ]

    if report.nsys_kernels:
        lines.extend(["", "nsys top kernels:", "time%   total_s  count      avg_ms   name"])
        for row in report.nsys_kernels[:top_n]:
            lines.append(
                f"{row.time_pct:5.1f}  {row.total_time_s:8.3f}  "
                f"{row.instances:7d}  {row.avg_ms:8.3f}  {_short_name(row.name)}"
            )

    if report.torch_ops:
        table_name = Path(report.torch_ops[0].table).name
        lines.extend(["", f"torch profiler top ops ({table_name}):", "cuda_ms  self%   calls   name"])
        for row in report.torch_ops[:top_n]:
            lines.append(
                f"{row.cuda_total_ms:7.1f}  {row.self_cuda_pct:5.1f}  "
                f"{row.calls:6d}  {_short_name(row.name)}"
            )

    if report.ncu_kernels:
        lines.extend(["", "ncu sampled kernels:", "samples  avg_ms   compute%  memory%   name"])
        for row in report.ncu_kernels[:top_n]:
            lines.append(
                f"{row.count:7d}  {_fmt(row.avg_duration_ms, 3):>7}  "
                f"{_fmt(row.avg_compute_pct, 1):>8}  {_fmt(row.avg_memory_pct, 1):>7}   "
                f"{_short_name(row.name)}"
            )

    if report.memops:
        lines.extend(["", "nsys memops by size:", "bytes       count     operation"])
        for row in report.memops[:top_n]:
            lines.append(f"{_fmt_bytes(row.total_bytes):>10}  {row.count:8d}  {row.operation}")

    if report.warnings:
        lines.extend(["", "warnings:"])
        for warning in report.warnings:
            lines.append(f"  - {warning}")

    return "\n".join(lines)


def render_json(report: ProfileReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)


def _path_list(values: Sequence[str]) -> list[Path]:
    return [Path(value).expanduser() for value in values]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log", type=Path, required=True, help="Training log to summarize.")
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--run-profile",
        default="local_gb10_quarter",
        help="Typed run profile used for fallback token settings.",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Profiler text file or torch_profile directory to include.",
    )
    parser.add_argument(
        "--artifact-glob",
        action="append",
        default=[],
        help="Additional glob of profiler artifacts to scan.",
    )
    parser.add_argument("--no-auto-discover", action="store_true")
    parser.add_argument("--hot-step-start", type=int, default=3)
    parser.add_argument("--hot-step-end", type=int, default=None)
    parser.add_argument("--top", type=int, default=8)
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

    report = build_profile_report(
        log=args.log.expanduser(),
        label=args.label,
        run_profile=args.run_profile,
        artifacts=_path_list(args.artifact),
        artifact_globs=args.artifact_glob,
        auto_discover=not args.no_auto_discover,
        hot_step_start=args.hot_step_start,
        hot_step_end=args.hot_step_end,
    )
    if args.format == "json":
        print(render_json(report))
    else:
        print(render_table(report, top_n=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
