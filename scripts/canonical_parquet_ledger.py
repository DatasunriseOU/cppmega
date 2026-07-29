"""Streaming canonical JSON records inside receipt-friendly ZSTD Parquet.

The JSON payload preserves the existing canonical logical hash contract while
Parquet removes the need to materialize large uncompressed JSONL ledgers.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


LEDGER_SCHEMA = "cppmega_canonical_parquet_ledger_v1"
DEFAULT_ROW_GROUP_ROWS = 512
DEFAULT_BUFFER_BYTES = 16 * 1024 * 1024

_SCHEMA_KEY = b"cppmega.schema"
_DOMAIN_KEY = b"cppmega.logical_domain"


class CanonicalParquetLedgerError(RuntimeError):
    """The canonical or physical Parquet ledger contract was violated."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CanonicalParquetLedgerError(
            f"value is not canonical JSON: {exc}"
        ) from exc


def _ledger_schema(domain: str | None) -> pa.Schema:
    metadata = {
        _SCHEMA_KEY: LEDGER_SCHEMA.encode("ascii"),
        _DOMAIN_KEY: (domain or "").encode("ascii"),
    }
    return pa.schema(
        (
            pa.field("sequence_index", pa.int64(), nullable=False),
            pa.field("record_json", pa.large_string(), nullable=False),
            pa.field("record_sha256", pa.binary(32), nullable=False),
        ),
        metadata=metadata,
    )


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_physical_ledger(
    path: Path,
    *,
    expected_domain: str | None,
) -> pq.ParquetFile:
    if path.is_symlink() or not path.is_file():
        raise CanonicalParquetLedgerError(
            f"ledger is missing, non-regular, or symlinked: {path}"
        )
    try:
        parquet = pq.ParquetFile(path)
    except (OSError, pa.ArrowException) as exc:
        raise CanonicalParquetLedgerError(
            f"ledger is not readable Parquet: {path}"
        ) from exc
    expected_schema = _ledger_schema(expected_domain)
    if parquet.schema_arrow != expected_schema:
        raise CanonicalParquetLedgerError(
            f"ledger schema or logical domain differs: {path}"
        )
    metadata = parquet.metadata
    for row_group in range(metadata.num_row_groups):
        for column in range(metadata.num_columns):
            codec = str(metadata.row_group(row_group).column(column).compression)
            if codec != "ZSTD":
                raise CanonicalParquetLedgerError(
                    f"ledger column is not ZSTD-compressed: {path}: {codec}"
                )
    return parquet


class CanonicalParquetLedgerWriter:
    """Append canonical records with bounded memory and logical sequence hashing."""

    def __init__(
        self,
        path: Path,
        *,
        domain: str | None = None,
        max_record_bytes: int | None = None,
        row_group_rows: int = DEFAULT_ROW_GROUP_ROWS,
        max_buffer_bytes: int = DEFAULT_BUFFER_BYTES,
    ):
        if path.exists() or path.is_symlink():
            raise CanonicalParquetLedgerError(
                f"ledger output already exists or is unsafe: {path}"
            )
        if row_group_rows < 1 or max_buffer_bytes < 1:
            raise ValueError("Parquet ledger buffer bounds must be positive")
        self.path = path
        self.domain = domain
        self._schema = _ledger_schema(domain)
        self._writer = pq.ParquetWriter(
            path,
            self._schema,
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            write_statistics=True,
        )
        self._logical_digest = hashlib.sha256()
        if domain is not None:
            self._logical_digest.update(domain.encode("ascii"))
            self._logical_digest.update(b"\0")
        self._max_record_bytes = max_record_bytes
        self._row_group_rows = row_group_rows
        self._max_buffer_bytes = max_buffer_bytes
        self._sequence_indexes: list[int] = []
        self._record_json: list[str] = []
        self._record_sha256: list[bytes] = []
        self._buffer_bytes = 0
        self.count = 0
        self._closed = False

    def append(self, value: object) -> None:
        if self._closed:
            raise CanonicalParquetLedgerError(
                f"ledger is already closed: {self.path}"
            )
        encoded = canonical_json_bytes(value)
        if (
            self._max_record_bytes is not None
            and len(encoded) > self._max_record_bytes
        ):
            raise CanonicalParquetLedgerError(
                f"{self.path.name} record exceeds "
                f"{self._max_record_bytes} bytes"
            )
        self._logical_digest.update(len(encoded).to_bytes(8, "big"))
        self._logical_digest.update(encoded)
        self._sequence_indexes.append(self.count)
        self._record_json.append(encoded.decode("utf-8"))
        self._record_sha256.append(hashlib.sha256(encoded).digest())
        self._buffer_bytes += len(encoded)
        self.count += 1
        if (
            len(self._record_json) >= self._row_group_rows
            or self._buffer_bytes >= self._max_buffer_bytes
        ):
            self._flush()

    def _flush(self) -> None:
        if not self._record_json:
            return
        table = pa.Table.from_arrays(
            (
                pa.array(self._sequence_indexes, type=pa.int64()),
                pa.array(self._record_json, type=pa.large_string()),
                pa.array(self._record_sha256, type=pa.binary(32)),
            ),
            schema=self._schema,
        )
        self._writer.write_table(table, row_group_size=table.num_rows)
        self._sequence_indexes.clear()
        self._record_json.clear()
        self._record_sha256.clear()
        self._buffer_bytes = 0

    @property
    def logical_sha256(self) -> str:
        if not self._closed:
            raise CanonicalParquetLedgerError(
                f"ledger is not closed: {self.path}"
            )
        return self._logical_digest.hexdigest()

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        self._writer.close()
        _fsync_file(self.path)
        parquet = _validate_physical_ledger(
            self.path,
            expected_domain=self.domain,
        )
        if parquet.metadata.num_rows != self.count:
            raise CanonicalParquetLedgerError(
                f"ledger row count differs after close: {self.path}"
            )
        self._closed = True


def iter_canonical_parquet_ledger(
    path: Path,
    *,
    expected_domain: str | None,
    expected_record_schema: str | None = None,
    max_record_bytes: int | None = None,
    allow_empty: bool = False,
) -> Iterator[tuple[dict[str, Any], bytes]]:
    """Yield canonical records while verifying every row and physical codec."""

    parquet = _validate_physical_ledger(
        path,
        expected_domain=expected_domain,
    )
    expected_index = 0
    try:
        for batch in parquet.iter_batches(
            columns=["sequence_index", "record_json", "record_sha256"],
            batch_size=DEFAULT_ROW_GROUP_ROWS,
        ):
            indexes = batch.column(0).to_pylist()
            records = batch.column(1).to_pylist()
            digests = batch.column(2).to_pylist()
            for sequence_index, raw_record, raw_digest in zip(
                indexes,
                records,
                digests,
                strict=True,
            ):
                if sequence_index != expected_index:
                    raise CanonicalParquetLedgerError(
                        "ledger sequence is not contiguous at row "
                        f"{expected_index}"
                    )
                if not isinstance(raw_record, str):
                    raise CanonicalParquetLedgerError(
                        f"ledger row {expected_index} has no JSON string"
                    )
                encoded = raw_record.encode("utf-8")
                if (
                    max_record_bytes is not None
                    and len(encoded) > max_record_bytes
                ):
                    raise CanonicalParquetLedgerError(
                        f"ledger row {expected_index} exceeds the record size limit"
                    )
                if (
                    not isinstance(raw_digest, bytes)
                    or hashlib.sha256(encoded).digest() != raw_digest
                ):
                    raise CanonicalParquetLedgerError(
                        f"ledger row {expected_index} digest differs"
                    )
                try:
                    value = json.loads(raw_record)
                except json.JSONDecodeError as exc:
                    raise CanonicalParquetLedgerError(
                        f"ledger row {expected_index} is invalid JSON"
                    ) from exc
                if (
                    not isinstance(value, dict)
                    or canonical_json_bytes(value) != encoded
                    or (
                        expected_record_schema is not None
                        and value.get("schema") != expected_record_schema
                    )
                ):
                    raise CanonicalParquetLedgerError(
                        f"ledger row {expected_index} is not canonical"
                    )
                yield value, encoded
                expected_index += 1
    except (OSError, pa.ArrowException) as exc:
        raise CanonicalParquetLedgerError(
            f"ledger row groups are unreadable: {path}"
        ) from exc
    if expected_index != parquet.metadata.num_rows:
        raise CanonicalParquetLedgerError("ledger row accounting differs")
    if expected_index == 0 and not allow_empty:
        raise CanonicalParquetLedgerError(f"ledger is empty: {path}")
