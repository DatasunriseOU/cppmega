#!/usr/bin/env python3
"""Backfill PR audit columns into packed commit parquet files.

The commit training text already carries PR discussions in the docstring:

    @pr N
    @discussion
    ...

Older packed parquet rows did not preserve an auditable sidecar flag telling us
which source documents actually contain that PR discussion. This script reads
packed rows, decodes the beginning of each source document with the cppmega
tokenizer, parses the docstring markers, and appends compact metadata columns.

It does not change tokens, targets, loss masks, doc ids, or graph sidecars.
Writes are atomic per parquet file: temp file then os.replace().
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega.data.nanochat_pipeline.packed_rows_schema import (  # noqa: E402
    DOC_IDS_COLUMN,
    INPUT_IDS_COLUMN,
    SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
    SOURCE_PR_DISCUSSION_CHARS_COLUMN,
    SOURCE_PR_DISCUSSION_LINES_COLUMN,
    SOURCE_PR_NUMBERS_COLUMN,
)
from cppmega.data.nanochat_pipeline.tokenized_enriched_schema import (  # noqa: E402
    HAS_PR_DISCUSSION_COLUMN,
    PR_DISCUSSION_CHARS_COLUMN,
    PR_DISCUSSION_LINES_COLUMN,
    PR_NUMBER_COLUMN,
)
from cppmega.tokenizer.cpp_tokenizer import load_cppmega_tokenizer  # noqa: E402


SOURCE_DOC_TOKEN_LENGTHS_COLUMN = "source_doc_token_lengths"
VALID_TOKEN_COUNT_COLUMN = "valid_token_count"
NUM_DOCS_COLUMN = "num_docs"

_PR_RE = re.compile(r"(?m)^[ \t]*\*?[ \t]*@pr[ \t]+([0-9]+)\b")
_DISCUSSION_RE = re.compile(r"(?m)^[ \t]*\*?[ \t]*@discussion\b")
_NEXT_DOCSTRING_FIELD_RE = re.compile(
    r"(?m)^[ \t]*\*?[ \t]*@[A-Za-z_][A-Za-z0-9_-]*\b"
)
_DOCSTRING_END_RE = re.compile(r"(?m)^[ \t]*\*/|^[ \t]*//[ \t]*===")

_TOKENIZER = None


@dataclass(frozen=True)
class PrDocMeta:
    pr_number: int | None
    has_discussion: bool
    discussion_chars: int
    discussion_lines: int


def _load_tokenizer_once(tokenizer_path: str):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = load_cppmega_tokenizer(tokenizer_path)
    return _TOKENIZER


def _extract_discussion_text(decoded_prefix: str) -> str:
    marker = _DISCUSSION_RE.search(decoded_prefix)
    if marker is None:
        return ""
    start = marker.end()
    field = _NEXT_DOCSTRING_FIELD_RE.search(decoded_prefix, start)
    end_marker = _DOCSTRING_END_RE.search(decoded_prefix, start)
    candidates = [
        match.start()
        for match in (field, end_marker)
        if match is not None and match.start() > start
    ]
    end = min(candidates) if candidates else len(decoded_prefix)
    raw_lines = decoded_prefix[start:end].splitlines()
    cleaned: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("*"):
            stripped = stripped[1:].lstrip()
        cleaned.append(stripped)
    return "\n".join(cleaned).strip()


def _parse_pr_doc_meta(decoded_prefix: str) -> PrDocMeta:
    pr_match = _PR_RE.search(decoded_prefix)
    pr_number = int(pr_match.group(1)) if pr_match else None
    discussion = _extract_discussion_text(decoded_prefix)
    return PrDocMeta(
        pr_number=pr_number,
        has_discussion=bool(discussion),
        discussion_chars=len(discussion),
        discussion_lines=0 if not discussion else discussion.count("\n") + 1,
    )


def _source_lengths_for_row(row: dict[str, Any]) -> list[int]:
    lengths = row.get(SOURCE_DOC_TOKEN_LENGTHS_COLUMN)
    if lengths:
        return [int(value) for value in lengths]

    valid = int(row.get(VALID_TOKEN_COUNT_COLUMN) or 0)
    doc_ids = row.get(DOC_IDS_COLUMN) or []
    if valid <= 0 or not doc_ids:
        return []
    out: list[int] = []
    prev = int(doc_ids[0])
    count = 0
    for raw in doc_ids[:valid]:
        doc_id = int(raw)
        if doc_id != prev:
            out.append(count)
            prev = doc_id
            count = 0
        count += 1
    if count:
        out.append(count)
    return out


def _decode_doc_prefixes(
    *,
    row: dict[str, Any],
    tokenizer_path: str,
    prefix_tokens: int,
) -> list[PrDocMeta]:
    tokenizer = _load_tokenizer_once(tokenizer_path)
    token_ids = [int(value) for value in (row.get(INPUT_IDS_COLUMN) or [])]
    lengths = _source_lengths_for_row(row)
    metas: list[PrDocMeta] = []
    offset = 0
    for length in lengths:
        end = offset + max(0, int(length))
        prefix_end = min(end, offset + max(1, int(prefix_tokens)))
        decoded = tokenizer.decode(token_ids[offset:prefix_end])
        metas.append(_parse_pr_doc_meta(decoded))
        offset = end
    return metas


def _same_or_none(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    first = present[0]
    return first if all(value == first for value in present) else None


def _replace_or_append(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    field = pa.field(name, array.type)
    if name in table.column_names:
        index = table.column_names.index(name)
        return table.set_column(index, field, array)
    return table.append_column(field, array)


def _compression_for(path: Path) -> str:
    pf = pq.ParquetFile(path)
    if pf.metadata.num_row_groups == 0 or pf.metadata.num_columns == 0:
        return "zstd"
    codec = pf.metadata.row_group(0).column(0).compression
    return str(codec).lower() if codec else "zstd"


def _process_file(args: tuple[str, str, int, bool]) -> dict[str, Any]:
    raw_path, tokenizer_path, prefix_tokens, force = args
    path = Path(raw_path)
    if path.name.endswith(".tmp.parquet"):
        return {"file": str(path), "skipped_tmp": True}

    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    new_columns = {
        SOURCE_PR_NUMBERS_COLUMN,
        SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
        SOURCE_PR_DISCUSSION_CHARS_COLUMN,
        SOURCE_PR_DISCUSSION_LINES_COLUMN,
        PR_NUMBER_COLUMN,
        HAS_PR_DISCUSSION_COLUMN,
        PR_DISCUSSION_CHARS_COLUMN,
        PR_DISCUSSION_LINES_COLUMN,
    }
    if not force and new_columns.issubset(names):
        return {"file": str(path), "skipped_existing": True}

    required = {INPUT_IDS_COLUMN}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    table = pf.read()
    source_pr_numbers: list[list[int | None]] = []
    source_has_discussions: list[list[bool]] = []
    source_discussion_chars: list[list[int]] = []
    source_discussion_lines: list[list[int]] = []
    row_pr_numbers: list[int | None] = []
    row_has_discussions: list[bool] = []
    row_discussion_chars: list[int] = []
    row_discussion_lines: list[int] = []

    rows = table.select(
        [
            column
            for column in (
                INPUT_IDS_COLUMN,
                DOC_IDS_COLUMN,
                SOURCE_DOC_TOKEN_LENGTHS_COLUMN,
                VALID_TOKEN_COUNT_COLUMN,
                NUM_DOCS_COLUMN,
            )
            if column in table.column_names
        ]
    ).to_pylist()
    for row in rows:
        metas = _decode_doc_prefixes(
            row=row,
            tokenizer_path=tokenizer_path,
            prefix_tokens=prefix_tokens,
        )
        pr_numbers = [meta.pr_number for meta in metas]
        has_discussions = [meta.has_discussion for meta in metas]
        chars = [meta.discussion_chars for meta in metas]
        lines = [meta.discussion_lines for meta in metas]
        source_pr_numbers.append(pr_numbers)
        source_has_discussions.append(has_discussions)
        source_discussion_chars.append(chars)
        source_discussion_lines.append(lines)
        row_pr_numbers.append(_same_or_none(pr_numbers))
        row_has_discussions.append(any(has_discussions))
        row_discussion_chars.append(sum(chars))
        row_discussion_lines.append(sum(lines))

    table = _replace_or_append(
        table,
        SOURCE_PR_NUMBERS_COLUMN,
        pa.array(source_pr_numbers, type=pa.list_(pa.int64())),
    )
    table = _replace_or_append(
        table,
        SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
        pa.array(source_has_discussions, type=pa.list_(pa.bool_())),
    )
    table = _replace_or_append(
        table,
        SOURCE_PR_DISCUSSION_CHARS_COLUMN,
        pa.array(source_discussion_chars, type=pa.list_(pa.int32())),
    )
    table = _replace_or_append(
        table,
        SOURCE_PR_DISCUSSION_LINES_COLUMN,
        pa.array(source_discussion_lines, type=pa.list_(pa.int32())),
    )
    table = _replace_or_append(
        table,
        PR_NUMBER_COLUMN,
        pa.array(row_pr_numbers, type=pa.int64()),
    )
    table = _replace_or_append(
        table,
        HAS_PR_DISCUSSION_COLUMN,
        pa.array(row_has_discussions, type=pa.bool_()),
    )
    table = _replace_or_append(
        table,
        PR_DISCUSSION_CHARS_COLUMN,
        pa.array(row_discussion_chars, type=pa.int32()),
    )
    table = _replace_or_append(
        table,
        PR_DISCUSSION_LINES_COLUMN,
        pa.array(row_discussion_lines, type=pa.int32()),
    )

    tmp = path.with_name(f"{path.name}.prmeta.tmp")
    pq.write_table(table, tmp, compression=_compression_for(path), row_group_size=1024)
    os.replace(tmp, path)
    source_doc_count = sum(len(values) for values in source_pr_numbers)
    return {
        "file": str(path),
        "rows": table.num_rows,
        "source_docs": source_doc_count,
        "rows_with_pr_number": sum(1 for value in row_pr_numbers if value is not None),
        "rows_with_pr_discussion": sum(1 for value in row_has_discussions if value),
        "source_docs_with_pr_number": sum(
            1 for values in source_pr_numbers for value in values if value is not None
        ),
        "source_docs_with_pr_discussion": sum(
            1 for values in source_has_discussions for value in values if value
        ),
    }


def _iter_parquet_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*.parquet")
        if path.is_file() and not path.name.endswith(".tmp.parquet")
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill PR audit metadata columns into packed commit parquet."
    )
    parser.add_argument("--root", required=True, help="Parquet file or directory.")
    parser.add_argument(
        "--tokenizer-path",
        default="cppmega/tokenizer/tokenizer.json",
        help="cppmega tokenizer.json path.",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=2048,
        help="Decode this many tokens from the start of each source doc.",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", help="Write aggregate summary JSON.")
    args = parser.parse_args()

    root = Path(args.root)
    files = _iter_parquet_files(root)
    if args.dry_run:
        print(json.dumps({"files": len(files), "dry_run": True}, indent=2))
        return 0

    totals: dict[str, Any] = {
        "files": 0,
        "skipped_existing": 0,
        "skipped_tmp": 0,
        "rows": 0,
        "source_docs": 0,
        "rows_with_pr_number": 0,
        "rows_with_pr_discussion": 0,
        "source_docs_with_pr_number": 0,
        "source_docs_with_pr_discussion": 0,
    }
    work = [
        (str(path), str(args.tokenizer_path), int(args.prefix_tokens), bool(args.force))
        for path in files
    ]
    if args.jobs <= 1:
        iterator = map(_process_file, work)
    else:
        executor = ProcessPoolExecutor(max_workers=int(args.jobs))
        iterator = (future.result() for future in as_completed(executor.submit(_process_file, item) for item in work))

    try:
        for index, result in enumerate(iterator, 1):
            if result.get("skipped_existing"):
                totals["skipped_existing"] += 1
            elif result.get("skipped_tmp"):
                totals["skipped_tmp"] += 1
            else:
                totals["files"] += 1
                for key in (
                    "rows",
                    "source_docs",
                    "rows_with_pr_number",
                    "rows_with_pr_discussion",
                    "source_docs_with_pr_number",
                    "source_docs_with_pr_discussion",
                ):
                    totals[key] += int(result.get(key, 0))
            if index % 250 == 0:
                print(json.dumps({"processed": index, **totals}, sort_keys=True), flush=True)
    finally:
        if "executor" in locals():
            executor.shutdown(wait=True, cancel_futures=False)

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(
            json.dumps(totals, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(totals, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
