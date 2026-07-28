#!/usr/bin/env python3
"""Strict, allocation-bounded decoding for durable CI zlib evidence."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from typing import Callable, Final
import zlib


EVIDENCE_LIMITS_SCHEMA: Final = "cppmega_ci_zlib_evidence_limits_v1"
MAX_RUN_METADATA_BYTES: Final = 16 * 1024 * 1024
MAX_RUN_METADATA_COMPRESSED_BYTES: Final = (
    MAX_RUN_METADATA_BYTES + 64 * 1024
)
MAX_JOBS_EVIDENCE_BYTES: Final = 128 * 1024 * 1024
MAX_JOBS_EVIDENCE_COMPRESSED_BYTES: Final = (
    MAX_JOBS_EVIDENCE_BYTES + 64 * 1024
)
MAX_STATE_JSON_EVIDENCE_BYTES: Final = 128 * 1024 * 1024
MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES: Final = (
    MAX_STATE_JSON_EVIDENCE_BYTES + 64 * 1024
)
MAX_CONTENT_FRAME_RAW_BYTES: Final = 128 * 1024 * 1024
MAX_CONTENT_FRAME_BYTES: Final = MAX_CONTENT_FRAME_RAW_BYTES
MAX_CONTENT_FRAME_COMPRESSED_BYTES: Final = (
    MAX_CONTENT_FRAME_BYTES + 64 * 1024
)
MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES: Final = 1024 * 1024
MAX_EMPTY_ARCHIVE_EVIDENCE_COMPRESSED_BYTES: Final = (
    MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES + 64 * 1024
)
MAX_SQLITE_EVIDENCE_ROW_BYTES: Final = (
    MAX_RUN_METADATA_COMPRESSED_BYTES
    + MAX_JOBS_EVIDENCE_COMPRESSED_BYTES
    + MAX_EMPTY_ARCHIVE_EVIDENCE_COMPRESSED_BYTES
    + 4 * 1024 * 1024
)

_INPUT_CHUNK_BYTES: Final = 64 * 1024
_OUTPUT_CHUNK_BYTES: Final = 64 * 1024
_HEX64_RE = re.compile(r"[0-9a-f]{64}")


class ZlibEvidenceError(ValueError):
    """Compressed evidence violates its declared bounded byte contract."""


def constrain_sqlite_evidence_rows(connection: sqlite3.Connection) -> int:
    """Bound SQLite row materialization before any evidence BLOB is selected."""

    current = connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
    if current > MAX_SQLITE_EVIDENCE_ROW_BYTES:
        connection.setlimit(
            sqlite3.SQLITE_LIMIT_LENGTH,
            MAX_SQLITE_EVIDENCE_ROW_BYTES,
        )
    configured = connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
    if configured > MAX_SQLITE_EVIDENCE_ROW_BYTES:
        raise ZlibEvidenceError(
            "SQLite evidence row length limit could not be constrained"
        )
    return configured


def fetch_state_evidence_bound_violation(
    connection: sqlite3.Connection,
) -> tuple[str, str, int, int, str] | None:
    """Return the first oversized fetch-state BLOB without selecting its bytes."""

    attempt = connection.execute(
        """
        SELECT
          'attempt' AS record_type,
          repo,
          run_id,
          attempt,
          CASE
            WHEN typeof(run_metadata_zlib)!='blob'
              OR run_metadata_raw_size<0
              OR run_metadata_raw_size>?
              OR length(run_metadata_zlib)>?
              THEN 'run_metadata_zlib'
            WHEN jobs_raw_size IS NOT NULL
              AND (jobs_raw_size<0 OR jobs_raw_size>?)
              THEN 'jobs_raw_size'
            WHEN jobs_zlib IS NOT NULL
              AND (
                typeof(jobs_zlib)!='blob'
                OR length(jobs_zlib)>?
              )
              THEN 'jobs_zlib'
            WHEN archive_zlib IS NOT NULL
              AND (
                typeof(archive_zlib)!='blob'
                OR archive_size IS NULL
                OR archive_size<0
                OR archive_size>?
                OR length(archive_zlib)>?
              )
              THEN 'archive_zlib'
          END AS evidence_field
        FROM attempts
        WHERE typeof(run_metadata_zlib)!='blob'
           OR run_metadata_raw_size<0
           OR run_metadata_raw_size>?
           OR length(run_metadata_zlib)>?
           OR (
             jobs_raw_size IS NOT NULL
             AND (jobs_raw_size<0 OR jobs_raw_size>?)
           )
           OR (
             jobs_zlib IS NOT NULL
             AND (
               typeof(jobs_zlib)!='blob'
               OR length(jobs_zlib)>?
             )
           )
           OR (
             archive_zlib IS NOT NULL
             AND (
               typeof(archive_zlib)!='blob'
               OR archive_size IS NULL
               OR archive_size<0
               OR archive_size>?
               OR length(archive_zlib)>?
             )
           )
        LIMIT 1
        """,
        (
            MAX_RUN_METADATA_BYTES,
            MAX_RUN_METADATA_COMPRESSED_BYTES,
            MAX_JOBS_EVIDENCE_BYTES,
            MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
            MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES,
            MAX_EMPTY_ARCHIVE_EVIDENCE_COMPRESSED_BYTES,
            MAX_RUN_METADATA_BYTES,
            MAX_RUN_METADATA_COMPRESSED_BYTES,
            MAX_JOBS_EVIDENCE_BYTES,
            MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
            MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES,
            MAX_EMPTY_ARCHIVE_EVIDENCE_COMPRESSED_BYTES,
        ),
    ).fetchone()
    if attempt is not None:
        return (
            str(attempt[0]),
            str(attempt[1]),
            int(attempt[2]),
            int(attempt[3]),
            str(attempt[4]),
        )
    member = connection.execute(
        """
        SELECT
          'member' AS record_type,
          repo,
          run_id,
          attempt,
          'sidecar_zlib' AS evidence_field
        FROM members
        WHERE typeof(sidecar_zlib)!='blob'
           OR sidecar_raw_size<0
           OR sidecar_raw_size>?
           OR length(sidecar_zlib)>?
        LIMIT 1
        """,
        (
            MAX_STATE_JSON_EVIDENCE_BYTES,
            MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES,
        ),
    ).fetchone()
    if member is None:
        return None
    return (
        str(member[0]),
        str(member[1]),
        int(member[2]),
        int(member[3]),
        str(member[4]),
    )


def fetch_state_attempt_evidence_bound_violation(
    connection: sqlite3.Connection,
    *,
    repo: str,
    run_id: int,
    attempt: int,
) -> str | None:
    """Check one attempt's evidence without selecting any of its BLOB bytes."""

    row = connection.execute(
        """
        SELECT CASE
          WHEN typeof(run_metadata_zlib)!='blob'
            OR run_metadata_raw_size<0
            OR run_metadata_raw_size>?
            OR length(run_metadata_zlib)>?
            THEN 'run_metadata_zlib'
          WHEN jobs_raw_size IS NOT NULL
            AND (jobs_raw_size<0 OR jobs_raw_size>?)
            THEN 'jobs_raw_size'
          WHEN jobs_zlib IS NOT NULL
            AND (
              typeof(jobs_zlib)!='blob'
              OR length(jobs_zlib)>?
            )
            THEN 'jobs_zlib'
          WHEN archive_zlib IS NOT NULL
            AND (
              typeof(archive_zlib)!='blob'
              OR archive_size IS NULL
              OR archive_size<0
              OR archive_size>?
              OR length(archive_zlib)>?
            )
            THEN 'archive_zlib'
        END AS evidence_field
        FROM attempts
        WHERE repo=? AND run_id=? AND attempt=?
          AND (
            typeof(run_metadata_zlib)!='blob'
            OR run_metadata_raw_size<0
            OR run_metadata_raw_size>?
            OR length(run_metadata_zlib)>?
            OR (
              jobs_raw_size IS NOT NULL
              AND (jobs_raw_size<0 OR jobs_raw_size>?)
            )
            OR (
              jobs_zlib IS NOT NULL
              AND (
                typeof(jobs_zlib)!='blob'
                OR length(jobs_zlib)>?
              )
            )
            OR (
              archive_zlib IS NOT NULL
              AND (
                typeof(archive_zlib)!='blob'
                OR archive_size IS NULL
                OR archive_size<0
                OR archive_size>?
                OR length(archive_zlib)>?
              )
            )
          )
        LIMIT 1
        """,
        (
            MAX_RUN_METADATA_BYTES,
            MAX_RUN_METADATA_COMPRESSED_BYTES,
            MAX_JOBS_EVIDENCE_BYTES,
            MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
            MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES,
            MAX_EMPTY_ARCHIVE_EVIDENCE_COMPRESSED_BYTES,
            repo,
            run_id,
            attempt,
            MAX_RUN_METADATA_BYTES,
            MAX_RUN_METADATA_COMPRESSED_BYTES,
            MAX_JOBS_EVIDENCE_BYTES,
            MAX_JOBS_EVIDENCE_COMPRESSED_BYTES,
            MAX_EMPTY_ARCHIVE_EVIDENCE_BYTES,
            MAX_EMPTY_ARCHIVE_EVIDENCE_COMPRESSED_BYTES,
        ),
    ).fetchone()
    return None if row is None else str(row[0])


def content_store_evidence_bound_violation(
    connection: sqlite3.Connection,
) -> tuple[str, str, str, str, int] | None:
    """Return an oversized occurrence provenance key without reading its BLOB."""

    row = connection.execute(
        """
        SELECT repo,run_attempt,job,step,chunk_ordinal
        FROM occurrences
        WHERE typeof(provenance_zlib)!='blob'
           OR provenance_raw_size<0
           OR provenance_raw_size>?
           OR length(provenance_zlib)>?
        LIMIT 1
        """,
        (
            MAX_STATE_JSON_EVIDENCE_BYTES,
            MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES,
        ),
    ).fetchone()
    if row is None:
        return None
    return (
        str(row[0]),
        str(row[1]),
        str(row[2]),
        str(row[3]),
        int(row[4]),
    )


def content_store_occurrence_evidence_bound_violation(
    connection: sqlite3.Connection,
    *,
    repo: str,
    run_attempt: str,
    job: str,
    step: str,
    chunk_ordinal: int,
) -> bool:
    """Check one occurrence's provenance without selecting its BLOB bytes."""

    row = connection.execute(
        """
        SELECT 1
        FROM occurrences
        WHERE repo=? AND run_attempt=? AND job=? AND step=? AND chunk_ordinal=?
          AND (
            typeof(provenance_zlib)!='blob'
            OR provenance_raw_size<0
            OR provenance_raw_size>?
            OR length(provenance_zlib)>?
          )
        LIMIT 1
        """,
        (
            repo,
            run_attempt,
            job,
            step,
            chunk_ordinal,
            MAX_STATE_JSON_EVIDENCE_BYTES,
            MAX_STATE_JSON_EVIDENCE_COMPRESSED_BYTES,
        ),
    ).fetchone()
    return row is not None


def strict_bounded_zlib_decode(
    compressed: bytes | bytearray | memoryview,
    *,
    expected_raw_size: int | None,
    expected_sha256: str,
    max_raw_size: int,
    max_compressed_size: int,
    where: str,
    digest_function: Callable[[bytes], str] | None = None,
) -> bytes:
    """Decode one exact zlib stream without exceeding explicit byte bounds.

    ``expected_raw_size=None`` is reserved for legacy stores that do not have a
    raw-size column.  Such callers still get the semantic ``max_raw_size`` cap.
    Input and output are both streamed in fixed-size pieces; no unbounded
    ``flush()`` is used.
    """

    if (
        isinstance(max_raw_size, bool)
        or not isinstance(max_raw_size, int)
        or max_raw_size < 0
        or isinstance(max_compressed_size, bool)
        or not isinstance(max_compressed_size, int)
        or max_compressed_size <= 0
    ):
        raise ZlibEvidenceError(f"{where} has invalid decoder bounds")
    if expected_raw_size is not None and (
        isinstance(expected_raw_size, bool)
        or not isinstance(expected_raw_size, int)
        or expected_raw_size < 0
        or expected_raw_size > max_raw_size
    ):
        raise ZlibEvidenceError(
            f"{where} declared raw size exceeds its semantic bound"
        )
    if (
        not isinstance(expected_sha256, str)
        or _HEX64_RE.fullmatch(expected_sha256) is None
    ):
        raise ZlibEvidenceError(f"{where} has an invalid SHA-256")
    if not isinstance(compressed, (bytes, bytearray, memoryview)):
        raise ZlibEvidenceError(f"{where} is not compressed bytes")

    compressed_view = memoryview(compressed)
    if len(compressed_view) > max_compressed_size:
        raise ZlibEvidenceError(
            f"{where} compressed bytes exceed their semantic bound"
        )

    output_limit = (
        max_raw_size if expected_raw_size is None else expected_raw_size
    )
    decompressor = zlib.decompressobj()
    decoded_parts: list[bytes] = []
    decoded_size = 0
    input_offset = 0
    try:
        while input_offset < len(compressed_view):
            input_end = min(
                input_offset + _INPUT_CHUNK_BYTES,
                len(compressed_view),
            )
            pending = compressed_view[input_offset:input_end].tobytes()
            input_offset = input_end
            while pending:
                remaining = output_limit - decoded_size
                produced = decompressor.decompress(
                    pending,
                    min(_OUTPUT_CHUNK_BYTES, remaining + 1),
                )
                if produced:
                    decoded_parts.append(produced)
                    decoded_size += len(produced)
                if decoded_size > output_limit:
                    raise ZlibEvidenceError(
                        f"{where} decoded bytes exceed their declared bound"
                    )
                pending = decompressor.unconsumed_tail
                if decompressor.eof:
                    if (
                        pending
                        or decompressor.unused_data
                        or input_offset != len(compressed_view)
                    ):
                        raise ZlibEvidenceError(
                            f"{where} has trailing compressed data"
                        )
                    break
                if not pending and not produced:
                    break
            if decompressor.eof:
                break
    except zlib.error as exc:
        raise ZlibEvidenceError(f"{where} is invalid zlib") from exc

    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or input_offset != len(compressed_view)
    ):
        raise ZlibEvidenceError(
            f"{where} is truncated, trailing, or exceeds its bound"
        )
    if expected_raw_size is not None and decoded_size != expected_raw_size:
        raise ZlibEvidenceError(f"{where} decoded size differs")

    raw = b"".join(decoded_parts)
    actual_sha256 = (
        hashlib.sha256(raw).hexdigest()
        if digest_function is None
        else digest_function(raw)
    )
    if actual_sha256 != expected_sha256:
        raise ZlibEvidenceError(f"{where} SHA-256 differs")
    return raw
