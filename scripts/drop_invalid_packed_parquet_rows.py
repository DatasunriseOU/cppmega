#!/usr/bin/env python3
"""Drop invalid rows from packed cppmega parquet shards.

This is a destructive-but-backed-up repair pass for packed training parquet.
Rows are dropped only when they violate the same fixed-shape/sidecar invariants
checked by ``scripts/audit_sidecar_parquet.py``.  The main use case is cleaning
older packed shards that accidentally contain rows longer than their bucket
capacity, for example a row in ``16384/`` whose ``input_ids`` length is 56k.
Those rows cannot be repaired in place without re-packing from source docs, so
they are removed from the current trainable corpus instead of silently entering
training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


TOKEN_COLUMNS = (
    "input_ids",
    "target_ids",
    "loss_mask",
    "doc_ids",
)

TOKEN_ALIGNED_FIELDS = (
    "token_platform_ids",
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
    "token_symbol_ids",
    "token_call_targets",
    "token_type_refs",
    "token_def_use",
    "token_change_mask_pre",
    "token_change_mask_post",
    "hunk_id_per_token",
    "edit_op_per_token",
)

CHUNK_ALIGNED_FIELDS = (
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
)


@dataclass(frozen=True)
class FileResult:
    path: str
    bucket: str
    rows_before: int = 0
    rows_after: int = 0
    dropped_rows: int = 0
    changed: bool = False
    dry_run: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bucket": self.bucket,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "dropped_rows": self.dropped_rows,
            "changed": self.changed,
            "dry_run": self.dry_run,
            "error": self.error,
        }


def _list_lengths(column: Any) -> np.ndarray:
    lengths = pc.fill_null(pc.list_value_length(column), 0).combine_chunks()
    return np.asarray(lengths.to_numpy(zero_copy_only=False), dtype=np.int64)


def _flat_numpy(column: Any) -> np.ndarray:
    flat = pc.list_flatten(column).combine_chunks()
    if len(flat) == 0:
        return np.asarray([], dtype=np.int64)
    return np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.int64)


def _column_numpy(column: Any) -> np.ndarray:
    arr = column.combine_chunks()
    return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.int64)


def _rows_with_flat_mask(lengths: np.ndarray, flat_mask: np.ndarray) -> np.ndarray:
    rows = np.zeros(len(lengths), dtype=bool)
    if len(lengths) == 0 or len(flat_mask) == 0:
        return rows
    nonempty = lengths > 0
    if not np.any(nonempty):
        return rows
    starts = np.empty(len(lengths), dtype=np.int64)
    starts[0] = 0
    if len(lengths) > 1:
        np.cumsum(lengths[:-1], out=starts[1:])
    reduced = np.add.reduceat(flat_mask.astype(np.int8), starts[nonempty])
    rows[nonempty] = reduced > 0
    return rows


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_or_empty(values: Any) -> Any:
    if values is None:
        return ()
    return values


def _flatten_edge(edge: Any) -> tuple[int | None, int | None]:
    if isinstance(edge, dict):
        return _as_int(edge.get("from")), _as_int(edge.get("to"))
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        return _as_int(edge[0]), _as_int(edge[1])
    return None, None


def _row_mask(table: pa.Table, *, bucket: int, vocab_size: int | None) -> np.ndarray:
    names = set(table.schema.names)
    rows = table.num_rows
    bad = np.zeros(rows, dtype=bool)

    if "input_ids" not in names:
        bad[:] = True
        return bad

    input_lengths = _list_lengths(table.column("input_ids"))
    bad |= input_lengths != bucket

    if "valid_token_count" in names:
        valid = _column_numpy(table.column("valid_token_count"))
    else:
        valid = input_lengths
    if "trained_token_count" in names:
        trained = _column_numpy(table.column("trained_token_count"))
    else:
        trained = valid
    bad |= (valid < 0) | (valid > input_lengths) | (trained < 0) | (trained > input_lengths)

    if vocab_size is not None:
        flat = _flat_numpy(table.column("input_ids"))
        if len(flat):
            bad_vocab = (flat < 0) | (flat >= vocab_size)
            bad |= _rows_with_flat_mask(input_lengths, bad_vocab)

    for name in ("target_ids", "loss_mask", "doc_ids", *TOKEN_ALIGNED_FIELDS):
        if name not in names:
            continue
        bad |= _list_lengths(table.column(name)) != input_lengths

    if "token_chunk_starts" in names:
        n_chunks = _list_lengths(table.column("token_chunk_starts"))
    else:
        n_chunks = np.zeros(rows, dtype=np.int64)

    for name in CHUNK_ALIGNED_FIELDS:
        if name in names:
            bad |= _list_lengths(table.column(name)) != n_chunks

    if "token_chunk_starts" in names:
        starts = _flat_numpy(table.column("token_chunk_starts"))
        if len(starts):
            bad_starts = (starts < 0) | (starts > np.repeat(input_lengths, n_chunks))
            bad |= _rows_with_flat_mask(n_chunks, bad_starts)

    if "token_chunk_ends" in names:
        ends = _flat_numpy(table.column("token_chunk_ends"))
        if len(ends):
            bad_ends = (ends < 0) | (ends > np.repeat(input_lengths, n_chunks))
            bad |= _rows_with_flat_mask(n_chunks, bad_ends)

    if "changed_chunk_ids" in names:
        for row_idx, values in enumerate(table.column("changed_chunk_ids").to_pylist()):
            limit = int(n_chunks[row_idx]) if row_idx < len(n_chunks) else 0
            for value in _iter_or_empty(values):
                idx = _as_int(value)
                if idx is None or idx < 0 or idx >= limit:
                    bad[row_idx] = True
                    break

    if "changed_chunk_spans" in names:
        for row_idx, spans in enumerate(table.column("changed_chunk_spans").to_pylist()):
            for span in _iter_or_empty(spans):
                if isinstance(span, dict):
                    start = _as_int(span.get("start"))
                    end = _as_int(span.get("end"))
                elif isinstance(span, (list, tuple)) and len(span) >= 2:
                    start = _as_int(span[0])
                    end = _as_int(span[1])
                else:
                    start = end = None
                if start is None or end is None or start < 0 or end < start or end > input_lengths[row_idx]:
                    bad[row_idx] = True
                    break

    for name in ("token_call_edges", "token_type_edges"):
        if name not in names:
            continue
        for row_idx, edges in enumerate(table.column(name).to_pylist()):
            limit = int(n_chunks[row_idx]) if row_idx < len(n_chunks) else 0
            for edge in _iter_or_empty(edges):
                src, dst = _flatten_edge(edge)
                if src is None or dst is None or src < 0 or dst < 0 or src >= limit or dst >= limit:
                    bad[row_idx] = True
                    break

    return bad


def _write_table_atomic(
    table: pa.Table,
    path: Path,
    *,
    compression: str,
    compression_level: int | None,
) -> None:
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(
            table,
            tmp_path,
            compression=compression,
            compression_level=compression_level,
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _process_file(
    path_str: str,
    *,
    vocab_size: int | None,
    dry_run: bool,
    backup_suffix: str,
    compression: str,
    compression_level: int | None,
    continue_on_error: bool,
) -> FileResult:
    path = Path(path_str)
    bucket = path.parent.name
    try:
        bucket_int = int(bucket)
    except ValueError:
        return FileResult(path=str(path), bucket=bucket, dry_run=dry_run, error="parent directory is not numeric bucket")

    try:
        table = pq.read_table(path)
        bad = _row_mask(table, bucket=bucket_int, vocab_size=vocab_size)
        dropped = int(np.count_nonzero(bad))
        if dropped == 0:
            return FileResult(
                path=str(path),
                bucket=bucket,
                rows_before=table.num_rows,
                rows_after=table.num_rows,
                dry_run=dry_run,
            )
        if not dry_run:
            backup = path.with_name(path.name + backup_suffix)
            if not backup.exists():
                shutil.copy2(path, backup)
            keep = pa.array(~bad)
            filtered = table.filter(keep)
            _write_table_atomic(
                filtered,
                path,
                compression=compression,
                compression_level=compression_level,
            )
        return FileResult(
            path=str(path),
            bucket=bucket,
            rows_before=table.num_rows,
            rows_after=table.num_rows - dropped,
            dropped_rows=dropped,
            changed=not dry_run,
            dry_run=dry_run,
        )
    except Exception as exc:
        # RULE #1: this is a DESTRUCTIVE cleaner for training-critical packed
        # parquet. A read/schema/rewrite failure must be fatal -- swallowing it
        # into a report entry and letting the process exit 0 would let an
        # exit-code-keyed pipeline treat the corpus as cleaned and upload/train
        # with invalid rows still present. By default we re-raise with WHERE
        # (path) + WHAT (original error) so the process crashes loud. Recording
        # the error and continuing is only allowed under the explicit
        # --continue-on-error opt-in, and even then main() still exits non-zero
        # whenever any file failed (see exit logic below).
        if continue_on_error:
            return FileResult(
                path=str(path),
                bucket=bucket,
                dry_run=dry_run,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise RuntimeError(
            f"failed to process packed parquet {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _discover(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_file():
            if path.suffix == ".parquet":
                out.append(path)
            continue
        out.extend(sorted(path.glob("*/*.parquet")))
    return sorted(dict.fromkeys(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path, help="Parquet file(s) or roots with bucket subdirs")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--vocab-size", type=int, default=65536)
    ap.add_argument("--no-vocab-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backup-suffix", default=".pre_validity_fix")
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--compression-level", type=int, default=6)
    ap.add_argument("--report", type=Path, default=Path("outputs/drop_invalid_packed_parquet_rows_report.json"))
    ap.add_argument("--fail-on-remaining", action="store_true")
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Opt in to processing remaining files past a per-file "
            "read/schema/write failure instead of crashing on the first one. "
            "The failure is recorded in the report, but the process STILL exits "
            "non-zero whenever any file failed. Without this flag a per-file "
            "failure raises immediately (fail-loud, the default)."
        ),
    )
    args = ap.parse_args()

    files = _discover(args.paths)
    if not files:
        raise SystemExit("no parquet files selected")

    vocab_size = None if args.no_vocab_check else args.vocab_size
    results: list[FileResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _process_file,
                str(path),
                vocab_size=vocab_size,
                dry_run=args.dry_run,
                backup_suffix=args.backup_suffix,
                compression=args.compression,
                compression_level=args.compression_level,
                continue_on_error=args.continue_on_error,
            )
            for path in files
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if idx % 100 == 0 or idx == len(futures):
                dropped = sum(item.dropped_rows for item in results)
                errors = sum(1 for item in results if item.error)
                print(
                    f"processed {idx}/{len(futures)} parquet files; dropped_rows={dropped}; errors={errors}",
                    flush=True,
                )

    total = {
        "files": len(results),
        "rows_before": sum(item.rows_before for item in results),
        "rows_after": sum(item.rows_after for item in results),
        "dropped_rows": sum(item.dropped_rows for item in results),
        "changed_files": sum(1 for item in results if item.changed),
        "error_files": sum(1 for item in results if item.error),
        "dry_run": args.dry_run,
    }
    by_bucket: dict[str, dict[str, int]] = {}
    for item in results:
        slot = by_bucket.setdefault(
            item.bucket,
            {"files": 0, "rows_before": 0, "rows_after": 0, "dropped_rows": 0, "changed_files": 0, "error_files": 0},
        )
        slot["files"] += 1
        slot["rows_before"] += item.rows_before
        slot["rows_after"] += item.rows_after
        slot["dropped_rows"] += item.dropped_rows
        slot["changed_files"] += int(item.changed)
        slot["error_files"] += int(item.error is not None)

    report = {
        "total": total,
        "by_bucket": dict(sorted(by_bucket.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])),
        "bad_files": [item.as_dict() for item in results if item.error or item.dropped_rows],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(total, indent=2))

    # RULE #1: any per-file failure is fatal BY DEFAULT. In the default
    # (fail-loud) mode a read/schema/write error already propagated out of
    # future.result() and crashed this process before reaching here, so the only
    # way error_files can be non-zero now is the narrow non-numeric-bucket guard
    # or an explicit --continue-on-error run. Either way we MUST exit non-zero --
    # never report errors and exit 0. --fail-on-remaining stays opt-in only for
    # the (non-error) dry-run "rows would be dropped" signal.
    if total["error_files"]:
        return 2
    if args.fail_on_remaining and args.dry_run and total["dropped_rows"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
