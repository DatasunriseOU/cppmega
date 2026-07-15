#!/usr/bin/env python3
"""Repair packed-row document boundaries from logical source lengths.

Older packed shards reused a file-level stable ID for multiple logical
documents from the same source file.  Their ``doc_ids`` and derived
``loss_mask`` were internally consistent but trained across function/document
boundaries.  This tool treats ``source_doc_token_lengths`` as the independent
truth, rebuilds row-local ``doc_ids`` and ``loss_mask``, and updates
``trained_token_count``.

Files are scanned before any write.  Changed parquet files are rewritten one
row group at a time to a sibling temporary file and published with
``os.replace``.  This makes it safe to run against a hard-linked snapshot: the
live source inode is never modified in place.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


BOUNDARY_COLUMNS = (
    "input_ids",
    "doc_ids",
    "loss_mask",
    "source_doc_ids",
    "source_doc_token_lengths",
    "valid_token_count",
    "trained_token_count",
    "num_docs",
)


@dataclass(frozen=True)
class FileScan:
    path: str
    rows: int
    changed_rows: int
    restored_boundaries: int
    old_trained_tokens: int
    new_trained_tokens: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _canonical_boundary_values(row: dict[str, Any], *, where: str) -> tuple[list[int], list[int], int]:
    input_ids = row.get("input_ids")
    doc_ids = row.get("doc_ids")
    loss_mask = row.get("loss_mask")
    source_lengths = row.get("source_doc_token_lengths")
    if not isinstance(input_ids, list) or not isinstance(doc_ids, list) or not isinstance(loss_mask, list):
        raise ValueError(f"{where}: missing list input_ids/doc_ids/loss_mask")
    capacity = len(input_ids)
    if len(doc_ids) != capacity or len(loss_mask) != capacity:
        raise ValueError(
            f"{where}: token-aligned length mismatch: capacity={capacity} "
            f"doc_ids={len(doc_ids)} loss_mask={len(loss_mask)}"
        )
    if not isinstance(source_lengths, list):
        raise ValueError(f"{where}: missing source_doc_token_lengths")
    valid = int(row.get("valid_token_count", capacity))
    source_doc_ids = row.get("source_doc_ids")
    return _canonical_from_lengths(
        capacity=capacity,
        valid=valid,
        logical_lengths=[int(value) for value in source_lengths],
        source_doc_count=len(source_doc_ids) if isinstance(source_doc_ids, list) else None,
        num_docs=int(row["num_docs"]) if row.get("num_docs") is not None else None,
        where=where,
    )


def _canonical_from_lengths(
    *,
    capacity: int,
    valid: int,
    logical_lengths: list[int],
    source_doc_count: int | None,
    num_docs: int | None,
    where: str,
) -> tuple[list[int], list[int], int]:
    if valid < 0 or valid > capacity:
        raise ValueError(f"{where}: invalid valid_token_count={valid} capacity={capacity}")
    if (valid > 0 and not logical_lengths) or any(length <= 0 for length in logical_lengths):
        raise ValueError(f"{where}: invalid source_doc_token_lengths={logical_lengths!r}")
    if sum(logical_lengths) != valid:
        raise ValueError(
            f"{where}: source_doc_token_lengths sum={sum(logical_lengths)} "
            f"!= valid_token_count={valid}"
        )
    if source_doc_count is not None and source_doc_count != len(logical_lengths):
        raise ValueError(
            f"{where}: source_doc_ids count={source_doc_count} "
            f"!= source_doc_token_lengths count={len(logical_lengths)}"
        )
    if num_docs is not None and num_docs != len(logical_lengths):
        raise ValueError(
            f"{where}: num_docs={num_docs} != source_doc_token_lengths count={len(logical_lengths)}"
        )

    expected_doc_ids: list[int] = []
    expected_loss_mask: list[int] = []
    for local_doc_id, length in enumerate(logical_lengths, start=1):
        expected_doc_ids.extend([local_doc_id] * length)
        expected_loss_mask.extend([1] * (length - 1))
        expected_loss_mask.append(0)
    pad_doc_id = len(logical_lengths) if logical_lengths else 0
    expected_doc_ids.extend([pad_doc_id] * (capacity - valid))
    expected_loss_mask.extend([0] * (capacity - valid))
    return expected_doc_ids, expected_loss_mask, sum(expected_loss_mask)


def repair_row(row: dict[str, Any], *, where: str = "row") -> tuple[dict[str, Any], bool, int]:
    expected_doc_ids, expected_loss_mask, trained = _canonical_boundary_values(row, where=where)
    changed = (
        [int(value) for value in row["doc_ids"]] != expected_doc_ids
        or [int(value) for value in row["loss_mask"]] != expected_loss_mask
        or int(row.get("trained_token_count", -1)) != trained
    )
    old_mask = [int(value) for value in row["loss_mask"]]
    restored = sum(
        1
        for old, new in zip(old_mask, expected_loss_mask)
        if old != 0 and new == 0
    )
    if not changed:
        return row, False, 0
    repaired = dict(row)
    repaired["doc_ids"] = expected_doc_ids
    repaired["loss_mask"] = expected_loss_mask
    repaired["trained_token_count"] = trained
    return repaired, True, restored


def _scan_file(path_str: str) -> FileScan:
    path = Path(path_str)
    parquet = pq.ParquetFile(path)
    missing = [name for name in BOUNDARY_COLUMNS[:7] if name not in parquet.schema_arrow.names]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
    selected = [name for name in BOUNDARY_COLUMNS if name in parquet.schema_arrow.names]
    rows = changed_rows = restored_boundaries = 0
    old_trained = new_trained = 0
    bucket = int(path.parent.name) if path.parent.name.isdigit() else 16384
    batch_size = max(1, 131_072 // bucket)
    for batch in parquet.iter_batches(columns=selected, batch_size=batch_size):
        for row in batch.to_pylist():
            where = f"{path}:row={rows}"
            repaired, changed, restored = repair_row(row, where=where)
            rows += 1
            changed_rows += int(changed)
            restored_boundaries += restored
            old_trained += int(row.get("trained_token_count", sum(row["loss_mask"])))
            new_trained += int(repaired.get("trained_token_count", sum(repaired["loss_mask"])))
    return FileScan(
        path=str(path),
        rows=rows,
        changed_rows=changed_rows,
        restored_boundaries=restored_boundaries,
        old_trained_tokens=old_trained,
        new_trained_tokens=new_trained,
    )


def _rewrite_file(path_str: str, compression_level: int) -> None:
    path = Path(path_str)
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.repair-", suffix=".parquet", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            tmp,
            schema,
            compression="zstd",
            compression_level=compression_level,
            use_dictionary=True,
        )
        bucket = int(path.parent.name) if path.parent.name.isdigit() else 16384
        batch_size = max(1, 131_072 // bucket)
        absolute_row = 0
        indices = {name: schema.get_field_index(name) for name in BOUNDARY_COLUMNS}
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch], schema=schema)
            capacities = pc.list_value_length(batch.column(indices["input_ids"])).to_pylist()
            logical_lengths_rows = batch.column(
                indices["source_doc_token_lengths"]
            ).to_pylist()
            source_doc_ids_rows = batch.column(indices["source_doc_ids"]).to_pylist()
            valids = batch.column(indices["valid_token_count"]).to_pylist()
            num_docs_rows = (
                batch.column(indices["num_docs"]).to_pylist()
                if indices["num_docs"] >= 0
                else [None] * len(batch)
            )

            doc_id_rows: list[list[int]] = []
            loss_mask_rows: list[list[int]] = []
            trained_rows: list[int] = []
            for offset, (capacity, logical_lengths, source_doc_ids, valid, num_docs) in enumerate(
                zip(
                    capacities,
                    logical_lengths_rows,
                    source_doc_ids_rows,
                    valids,
                    num_docs_rows,
                    strict=True,
                )
            ):
                logical_lengths = [int(value) for value in logical_lengths or []]
                expected_doc_ids, expected_loss_mask, trained = _canonical_from_lengths(
                    capacity=int(capacity),
                    valid=int(valid),
                    logical_lengths=logical_lengths,
                    source_doc_count=(
                        len(source_doc_ids) if isinstance(source_doc_ids, list) else None
                    ),
                    num_docs=int(num_docs) if num_docs is not None else None,
                    where=f"{path}:row={absolute_row + offset}",
                )
                doc_id_rows.append(expected_doc_ids)
                loss_mask_rows.append(expected_loss_mask)
                trained_rows.append(trained)

            table = table.set_column(
                indices["doc_ids"],
                schema.field(indices["doc_ids"]),
                pa.array(doc_id_rows, type=schema.field(indices["doc_ids"]).type),
            )
            table = table.set_column(
                indices["loss_mask"],
                schema.field(indices["loss_mask"]),
                pa.array(loss_mask_rows, type=schema.field(indices["loss_mask"]).type),
            )
            table = table.set_column(
                indices["trained_token_count"],
                schema.field(indices["trained_token_count"]),
                pa.array(
                    trained_rows,
                    type=schema.field(indices["trained_token_count"]).type,
                ),
            )
            writer.write_table(table)
            absolute_row += len(batch)
        writer.close()
        writer = None
        check = pq.ParquetFile(tmp)
        if check.metadata.num_rows != parquet.metadata.num_rows or check.schema_arrow != schema:
            raise RuntimeError(f"{path}: rewritten parquet validation failed")
        os.replace(tmp, path)
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)


def _discover(roots: list[Path], buckets: tuple[int, ...]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        for bucket in buckets:
            bucket_root = root / str(bucket)
            if not bucket_root.is_dir():
                raise FileNotFoundError(bucket_root)
            paths.extend(sorted(bucket_root.glob("*.parquet")))
    if not paths:
        raise RuntimeError("no parquet files selected")
    return paths


def _parse_buckets(value: str) -> tuple[int, ...]:
    buckets = tuple(int(part) for part in value.split(",") if part.strip())
    if not buckets or any(bucket <= 0 for bucket in buckets):
        raise argparse.ArgumentTypeError("buckets must be a non-empty comma-separated list")
    return buckets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--buckets", type=_parse_buckets, default=(1024, 2048, 4096, 8192, 16384))
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = _discover([root.resolve() for root in args.root], args.buckets)
    scans: list[FileScan] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_scan_file, str(path)): path for path in paths}
        for future in as_completed(futures):
            scans.append(future.result())
    scans.sort(key=lambda item: item.path)

    changed = [scan for scan in scans if scan.changed_rows]
    if changed and not args.dry_run:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(_rewrite_file, scan.path, args.compression_level) for scan in changed]
            for future in as_completed(futures):
                future.result()

    payload = {
        "schema": "cppmega_packed_document_boundary_repair_v1",
        "created_at": _utc_now(),
        "dry_run": bool(args.dry_run),
        "files": len(scans),
        "rows": sum(scan.rows for scan in scans),
        "changed_files": len(changed),
        "changed_rows": sum(scan.changed_rows for scan in scans),
        "restored_boundaries": sum(scan.restored_boundaries for scan in scans),
        "old_trained_tokens": sum(scan.old_trained_tokens for scan in scans),
        "new_trained_tokens": sum(scan.new_trained_tokens for scan in scans),
        "file_scans": [asdict(scan) for scan in changed],
    }
    if args.receipt:
        _write_json_atomic(args.receipt, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "file_scans"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
