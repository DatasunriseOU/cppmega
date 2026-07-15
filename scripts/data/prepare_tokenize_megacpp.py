#!/usr/bin/env python3
"""Run the repository-local source-to-packed-parquet conveyor.

The heavy source processing lives in ``scripts/data/source_conveyor.py``. This
entrypoint keeps the public contract small and portable: callers choose the
source tree, packed output root, tokenizer artifact, and target sequence
lengths explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SOURCE_CONVEYOR = HERE / "source_conveyor.py"
DEFAULT_TOKENIZER = REPO_ROOT / "data" / "tokenizer_v2" / "tokenizer.json"
DEFAULT_TARGET_LENGTHS = (1024, 2048, 4096)


def _parse_target_lengths(value: str) -> tuple[int, ...]:
    lengths: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            length = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"target length must be an integer, got {item!r}"
            ) from exc
        if length <= 0:
            raise argparse.ArgumentTypeError("target lengths must be positive")
        lengths.add(length)
    if not lengths:
        raise argparse.ArgumentTypeError("at least one target length is required")
    return tuple(sorted(lengths))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing already-extracted source repositories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root; packed parquet is published under <root>/<length>/.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer JSON artifact (default: {DEFAULT_TOKENIZER}).",
    )
    parser.add_argument(
        "--target-lengths",
        type=_parse_target_lengths,
        default=DEFAULT_TARGET_LENGTHS,
        metavar="CSV",
        help="Comma-separated packed lengths (default: 1024,2048,4096).",
    )
    parser.add_argument(
        "--repo-list",
        type=Path,
        default=None,
        help="Optional bare-repository to owner/repo identity map.",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=None,
        help="Optional cap for a focused run; omit for the full source root.",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=None,
        help="Override the local conveyor free-disk preflight for a controlled run.",
    )
    parser.add_argument(
        "--expected-code-revision",
        default=None,
        help="Exact 40-character Git revision; defaults to the current HEAD.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the local conveyor command without processing source data.",
    )
    return parser


def build_conveyor_command(args: argparse.Namespace) -> list[str]:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    tokenizer = args.tokenizer.resolve()
    target_lengths = ",".join(str(length) for length in args.target_lengths)
    command = [
        sys.executable,
        str(SOURCE_CONVEYOR),
        "--source-root",
        str(source_root),
        "--output-root",
        str(output_root),
        "--tokenizer",
        str(tokenizer),
        "--target-lengths",
        target_lengths,
    ]
    if args.repo_list is not None:
        command.extend(("--repo-list", str(args.repo_list.resolve())))
    if args.max_repos is not None:
        command.extend(("--max-repos", str(args.max_repos)))
    if args.min_free_disk_gb is not None:
        command.extend(("--min-free-disk-gb", str(args.min_free_disk_gb)))
    if args.expected_code_revision is not None:
        command.extend(("--expected-code-revision", args.expected_code_revision))
    return command


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    tokenizer = args.tokenizer.resolve()
    if not source_root.is_dir():
        parser.error(f"--source-root is not a directory: {source_root}")
    if not tokenizer.is_file():
        parser.error(f"--tokenizer is not a file: {tokenizer}")

    command = build_conveyor_command(args)
    if args.dry_run:
        print(f"DRY-RUN {shlex.join(command)}")
        return 0

    if not SOURCE_CONVEYOR.is_file():
        parser.error(f"local source conveyor not found: {SOURCE_CONVEYOR}")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
