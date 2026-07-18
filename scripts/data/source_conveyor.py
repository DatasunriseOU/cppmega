#!/usr/bin/env python3
"""Portable code-only adapter for the root streaming conveyor.

``scripts/streaming_conveyor.py`` is the canonical orchestration engine. Its
historical CLI uses ``--source-dir-root`` and keeps the tokenizer in the
``streaming_reindex`` module, so this small adapter exposes the root data
entrypoint contract without importing a sibling checkout or changing the
canonical engine.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BACKEND_PATH = REPO_ROOT / "scripts" / "streaming_conveyor.py"
DEFAULT_TOKENIZER = REPO_ROOT / "data" / "tokenizer_v2" / "tokenizer.json"


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
        help="Directory whose immediate children are source repositories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root for packed CODE parquet buckets.",
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
        default=(1024, 2048, 4096),
        metavar="CSV",
        help="Comma-separated packed CODE lengths (default: 1024,2048,4096).",
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
        help="Override the backend free-disk preflight for a controlled run.",
    )
    parser.add_argument(
        "--expected-code-revision",
        default=None,
        help="Exact 40-character Git revision; defaults to the current HEAD.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the backend command and tokenizer binding without processing.",
    )
    return parser


def _load_backend():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("scripts.streaming_conveyor")


def _current_revision() -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "cannot determine current Git revision for the conveyor: "
            + completed.stderr.strip()
        )
    revision = completed.stdout.strip()
    if len(revision) != 40:
        raise RuntimeError(f"unexpected Git revision: {revision!r}")
    return revision


def build_backend_args(
    args: argparse.Namespace,
    *,
    expected_code_revision: str | None = None,
) -> list[str]:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    lengths = ",".join(str(length) for length in args.target_lengths)
    backend_args = [
        "--streams",
        "code",
        "--source-dir-root",
        str(source_root),
        "--code-output-root",
        str(output_root),
        "--commit-output-root",
        str(output_root / ".commits"),
        "--conveyor-root",
        str(output_root / ".conveyor"),
        "--target-lengths-code",
        lengths,
        # The unified backend validates both ladders while parsing, even for
        # code-only runs. Keep the unused commit ladder deterministic and local.
        "--target-lengths-commits",
        lengths,
        # Disable unrelated commit/dedup state unless the caller opts in.
        "--pr-store",
        "",
        "--dedup-db",
        "",
    ]
    if args.repo_list is not None:
        backend_args.extend(("--repo-list", str(args.repo_list.resolve())))
    else:
        backend_args.extend(("--repo-list", ""))
    if args.max_repos is not None:
        backend_args.extend(("--max-repos", str(args.max_repos)))
    if args.min_free_disk_gb is not None:
        backend_args.extend(("--min-free-disk-gb", str(args.min_free_disk_gb)))
    if expected_code_revision is not None:
        backend_args.extend(("--expected-code-revision", expected_code_revision))
    return backend_args


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    tokenizer = args.tokenizer.resolve()
    if not source_root.is_dir():
        parser.error(f"--source-root is not a directory: {source_root}")
    if not tokenizer.is_file():
        parser.error(f"--tokenizer is not a file: {tokenizer}")
    if not BACKEND_PATH.is_file():
        parser.error(f"canonical conveyor not found: {BACKEND_PATH}")

    revision = args.expected_code_revision
    if revision is not None and len(revision.strip()) != 40:
        parser.error("--expected-code-revision must be a 40-character Git revision")
    if not args.dry_run and revision is None:
        revision = _current_revision()

    backend_args = build_backend_args(args, expected_code_revision=revision)
    backend_command = [sys.executable, str(BACKEND_PATH), *backend_args]
    if args.dry_run:
        print(f"DRY-RUN-CONFIG tokenizer={shlex.quote(str(tokenizer))}")
        print(f"DRY-RUN {shlex.join(backend_command)}")
        return 0

    backend = _load_backend()
    # The canonical backend already exposes its stage modules as the runtime
    # seam. Bind the explicit CLI artifact before it constructs any workers.
    backend.sr.TOKENIZER_PATH = tokenizer
    return int(backend.main(backend_args))


if __name__ == "__main__":
    raise SystemExit(main())
