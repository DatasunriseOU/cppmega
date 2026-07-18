#!/usr/bin/env python3
"""Report usable parquet token totals and training steps by context length.

This is intentionally narrow: it reads packed parquet counters
(`valid_token_count`, `trained_token_count`, `num_docs`) without loading token
or sidecar payload columns, then reports how many full optimizer steps the
current data supports for a batch-size schedule.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid


DEFAULT_BATCH_BY_LENGTH = {
    1024: 192,
    2048: 96,
    4096: 48,
    8192: 24,
    16384: 12,
}

DEFAULT_CODE_ROOT = Path("outputs/reindexed")
DEFAULT_COMMIT_ROOT = Path("outputs/reindexed_commits")
DEFAULT_PR_ROOT = Path("outputs/reindexed_pr")

COUNTER_COLUMNS = ("valid_token_count", "trained_token_count", "num_docs")
_CONCURRENT_READ_ERRORS = (FileNotFoundError, ArrowInvalid, OSError)


@dataclass
class TokenStats:
    files: int = 0
    skipped_files: int = 0
    rows: int = 0
    docs: int = 0
    valid_tokens: int = 0
    trained_tokens: int = 0

    def add(self, other: "TokenStats") -> None:
        self.files += other.files
        self.skipped_files += other.skipped_files
        self.rows += other.rows
        self.docs += other.docs
        self.valid_tokens += other.valid_tokens
        self.trained_tokens += other.trained_tokens


@dataclass(frozen=True)
class ReportRow:
    kind: str
    length: int
    batch_size: int
    tokens_per_step: int
    files: int
    skipped_files: int
    rows: int
    docs: int
    valid_tokens: int
    trained_tokens: int
    steps_by_trained_tokens: int
    steps_by_valid_tokens: int


def parse_batch_schedule(value: str) -> dict[int, int]:
    """Parse `1024=192,2048=96` into an int mapping."""
    if not value.strip():
        return dict(DEFAULT_BATCH_BY_LENGTH)
    result: dict[int, int] = {}
    for part in value.split(","):
        key, sep, val = part.strip().partition("=")
        if not sep:
            raise ValueError(f"invalid --batch-schedule item {part!r}; expected LENGTH=BATCH")
        length = int(key)
        batch = int(val)
        if length <= 0 or batch <= 0:
            raise ValueError(f"invalid non-positive schedule item {part!r}")
        result[length] = batch
    return result


def parse_required_metadata(items: Iterable[str] | None) -> dict[str, str]:
    """Parse repeated `KEY=VALUE` metadata requirements."""
    required: dict[str, str] = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(
                f"invalid --require-metadata item {item!r}; expected KEY=VALUE"
            )
        required[key] = value
    return required


def _length_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )


def _metadata_matches(
    parquet_file: pq.ParquetFile,
    required_metadata: Mapping[str, str],
) -> bool:
    if not required_metadata:
        return True
    metadata = parquet_file.schema_arrow.metadata or {}
    for key, expected in required_metadata.items():
        actual = metadata.get(key.encode("utf-8"))
        if actual != expected.encode("utf-8"):
            return False
    return True


def _read_file_stats(
    path: Path,
    *,
    allow_concurrent_skips: bool,
    required_metadata: Mapping[str, str],
) -> TokenStats:
    try:
        parquet_file = pq.ParquetFile(path)
        if not _metadata_matches(parquet_file, required_metadata):
            return TokenStats(skipped_files=1)
        names = set(parquet_file.schema_arrow.names)
        missing = [name for name in COUNTER_COLUMNS[:2] if name not in names]
        if missing:
            raise ValueError(f"{path} missing required counter columns: {missing}")
        columns = [name for name in COUNTER_COLUMNS if name in names]
        table = parquet_file.read(columns=columns)
    except _CONCURRENT_READ_ERRORS:
        if allow_concurrent_skips:
            return TokenStats(skipped_files=1)
        raise

    stats = TokenStats(files=1, rows=table.num_rows)
    stats.valid_tokens = int(table.column("valid_token_count").to_numpy(zero_copy_only=False).sum())
    stats.trained_tokens = int(table.column("trained_token_count").to_numpy(zero_copy_only=False).sum())
    if "num_docs" in table.column_names:
        stats.docs = int(table.column("num_docs").to_numpy(zero_copy_only=False).sum())
    return stats


def collect_length_stats(
    root: Path,
    *,
    max_workers: int,
    allow_concurrent_skips: bool,
    required_metadata: Mapping[str, str] | None = None,
) -> dict[int, TokenStats]:
    """Collect stats for every numeric bucket directory under `root`."""
    results: dict[int, TokenStats] = {}
    metadata_filter = required_metadata or {}
    for length_dir in _length_dirs(root):
        total = TokenStats()
        files = sorted(length_dir.glob("*.parquet"))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _read_file_stats,
                    path,
                    allow_concurrent_skips=allow_concurrent_skips,
                    required_metadata=metadata_filter,
                )
                for path in files
            ]
            for future in as_completed(futures):
                total.add(future.result())
        results[int(length_dir.name)] = total
    return results


def _row(kind: str, length: int, stats: TokenStats, batch_by_length: dict[int, int]) -> ReportRow:
    batch = batch_by_length.get(length, 0)
    tokens_per_step = batch * length if batch else 0
    return ReportRow(
        kind=kind,
        length=length,
        batch_size=batch,
        tokens_per_step=tokens_per_step,
        files=stats.files,
        skipped_files=stats.skipped_files,
        rows=stats.rows,
        docs=stats.docs,
        valid_tokens=stats.valid_tokens,
        trained_tokens=stats.trained_tokens,
        steps_by_trained_tokens=stats.trained_tokens // tokens_per_step if tokens_per_step else 0,
        steps_by_valid_tokens=stats.valid_tokens // tokens_per_step if tokens_per_step else 0,
    )


def combine_stats(*items: TokenStats | None) -> TokenStats:
    total = TokenStats()
    for item in items:
        if item is not None:
            total.add(item)
    return total


def build_report(
    *,
    code_root: Path,
    commit_root: Path,
    pr_root: Path,
    batch_by_length: dict[int, int],
    max_workers: int,
    allow_concurrent_skips: bool,
    required_metadata: Mapping[str, str] | None = None,
) -> list[ReportRow]:
    code = collect_length_stats(
        code_root,
        max_workers=max_workers,
        allow_concurrent_skips=allow_concurrent_skips,
        required_metadata=required_metadata,
    )
    commits = collect_length_stats(
        commit_root,
        max_workers=max_workers,
        allow_concurrent_skips=allow_concurrent_skips,
        required_metadata=required_metadata,
    )
    pr = collect_length_stats(
        pr_root,
        max_workers=max_workers,
        allow_concurrent_skips=allow_concurrent_skips,
        required_metadata=required_metadata,
    )

    lengths = sorted(set(code) | set(commits) | set(pr) | set(batch_by_length))
    rows: list[ReportRow] = []
    for length in lengths:
        if length in code:
            rows.append(_row("code_only", length, code[length], batch_by_length))
        if length in commits:
            rows.append(_row("commits_with_pr_docstring", length, commits[length], batch_by_length))
        main = combine_stats(code.get(length), commits.get(length))
        if main.files or main.skipped_files:
            rows.append(_row("main_code_plus_commits", length, main, batch_by_length))
        if length in pr:
            rows.append(_row("standalone_pr_side_stream", length, pr[length], batch_by_length))
            with_pr = combine_stats(main, pr[length])
            rows.append(_row("main_plus_standalone_pr", length, with_pr, batch_by_length))
    return rows


def _format_int(value: int) -> str:
    return f"{value:,}"


def _print_ascii_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def border(sep: str = "+", fill: str = "-") -> str:
        return sep + sep.join(fill * (width + 2) for width in widths) + sep

    def line(values: list[str]) -> str:
        cells = []
        for idx, value in enumerate(values):
            cells.append(f" {value:>{widths[idx]}} ")
        return "|" + "|".join(cells) + "|"

    print(border())
    print(line(headers))
    print(border())
    for row in rows:
        print(line(row))
    print(border())


def _rows_by_length_and_kind(rows: Iterable[ReportRow]) -> dict[int, dict[str, ReportRow]]:
    result: dict[int, dict[str, ReportRow]] = {}
    for row in rows:
        result.setdefault(row.length, {})[row.kind] = row
    return result


def _print_summary(rows: Iterable[ReportRow]) -> None:
    """Print one unambiguous fixed-width curriculum row per context length."""
    summary_rows: list[list[str]] = []
    for length, by_kind in sorted(_rows_by_length_and_kind(rows).items()):
        main = by_kind.get("main_code_plus_commits")
        if main is None:
            continue
        code = by_kind.get("code_only")
        commits = by_kind.get("commits_with_pr_docstring")
        pr = by_kind.get("standalone_pr_side_stream")
        main_pr = by_kind.get("main_plus_standalone_pr")
        skipped = sum(row.skipped_files for row in by_kind.values())
        summary_rows.append(
            [
                str(length),
                _format_int(main.batch_size),
                _format_int(main.tokens_per_step),
                _format_int(code.trained_tokens) if code else "-",
                _format_int(code.steps_by_trained_tokens) if code else "-",
                _format_int(commits.trained_tokens) if commits else "-",
                _format_int(commits.steps_by_trained_tokens) if commits else "-",
                _format_int(main.trained_tokens),
                _format_int(main.steps_by_trained_tokens),
                _format_int(pr.trained_tokens) if pr else "-",
                _format_int(main_pr.steps_by_trained_tokens) if main_pr else "-",
                _format_int(skipped),
            ]
        )
    _print_ascii_table(
        [
            "bucket_len",
            "batch_size",
            "tokens_per_step",
            "code_trained_tokens",
            "code_steps",
            "commit_pr_trained_tokens",
            "commit_pr_steps",
            "main_trained_tokens",
            "main_steps",
            "standalone_pr_trained_tokens",
            "main_plus_standalone_pr_steps",
            "skipped_files",
        ],
        summary_rows,
    )
    print()
    print(
        "steps = floor(trained_token_count / tokens_per_step), where "
        "tokens_per_step = bucket_len * batch_size."
    )
    print(
        "main = code_only + commits_with_pr_docstring. "
        "standalone_pr columns are comparison-only and are not mixed into main."
    )


def _print_markdown(rows: Iterable[ReportRow]) -> None:
    print(
        "| kind | length | bs | tokens/step | files | skipped | rows | docs | trained tokens | steps(trained) | valid tokens | steps(valid) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row.kind,
                    str(row.length),
                    str(row.batch_size),
                    _format_int(row.tokens_per_step),
                    _format_int(row.files),
                    _format_int(row.skipped_files),
                    _format_int(row.rows),
                    _format_int(row.docs),
                    _format_int(row.trained_tokens),
                    _format_int(row.steps_by_trained_tokens),
                    _format_int(row.valid_tokens),
                    _format_int(row.steps_by_valid_tokens),
                ]
            )
            + " |"
        )


def _print_text(rows: Iterable[ReportRow]) -> None:
    for row in rows:
        print(
            f"{row.kind:28s} len={row.length:5d} bs={row.batch_size:4d} "
            f"tokens/step={row.tokens_per_step:9d} files={row.files:5d} "
            f"skipped={row.skipped_files:3d} rows={row.rows:9d} docs={row.docs:9d} "
            f"trained={row.trained_tokens:14d} steps={row.steps_by_trained_tokens:7d} "
            f"valid={row.valid_tokens:14d} valid_steps={row.steps_by_valid_tokens:7d}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--commit-root", type=Path, default=DEFAULT_COMMIT_ROOT)
    parser.add_argument("--pr-root", type=Path, default=DEFAULT_PR_ROOT)
    parser.add_argument(
        "--batch-schedule",
        default=",".join(f"{k}={v}" for k, v in DEFAULT_BATCH_BY_LENGTH.items()),
        help="Comma-separated LENGTH=BATCH entries.",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--allow-concurrent-skips",
        action="store_true",
        help="Skip files that disappear or are unreadable during live conveyor writes, and report the skip count.",
    )
    parser.add_argument(
        "--require-metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Only count parquet files whose schema metadata contains KEY=VALUE. "
            "May be repeated; non-matching files are reported as skipped."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("summary", "text", "markdown", "json"),
        default="summary",
        help="summary prints one row per bucket; text/markdown print every source kind.",
    )
    args = parser.parse_args()
    required_metadata = parse_required_metadata(args.require_metadata)

    rows = build_report(
        code_root=args.code_root,
        commit_root=args.commit_root,
        pr_root=args.pr_root,
        batch_by_length=parse_batch_schedule(args.batch_schedule),
        max_workers=max(args.jobs, 1),
        allow_concurrent_skips=args.allow_concurrent_skips,
        required_metadata=required_metadata,
    )
    if args.format == "json":
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
    elif args.format == "summary":
        _print_summary(rows)
    elif args.format == "markdown":
        _print_markdown(rows)
    else:
        _print_text(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
