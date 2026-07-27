#!/usr/bin/env python3
"""Finalize immutable, merge-compatible CI stream receipts.

The fetch loop cannot honestly bind its SQLite artifact while its writer is
still open.  This module is deliberately pyarrow-free so the lightweight
fetch environment can close every writer first and only then create the
frozen store and fetch-state receipts consumed by the shard merger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Iterable, Mapping

from scripts.ci_content_store import (
    CIContentStore,
    _hash_records,
    _sqlite_schema_sha256,
)
from scripts.ci_stream_fetch import (
    RECEIPT_SCHEMA,
    SCHEMA_VERSION,
    _STATE_SCHEMA,
    ExactTokenizer,
    _sha256_bytes,
    _sha256_file,
    _utc_now,
    atomic_write_json,
)


_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_TERMINAL_RECEIPT_STATES = (
    "done",
    "empty",
    "terminal_404",
    "terminal_410",
)


class ReceiptFinalizationError(RuntimeError):
    """A mutable or inconsistent stream cannot publish a frozen receipt."""


def _require_frozen_sqlite(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ReceiptFinalizationError(f"{label} is missing or unsafe: {path}")
    for suffix in ("-wal", "-shm", "-journal"):
        pending = Path(f"{path}{suffix}")
        if pending.exists() or pending.is_symlink():
            raise ReceiptFinalizationError(
                f"{label} is not frozen; found {pending.name}"
            )


def _freeze_fetch_state_sqlite(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ReceiptFinalizationError(
            f"fetch state is missing or unsafe: {path}"
        )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            path,
            isolation_level=None,
            timeout=0.25,
        )
        connection.execute("PRAGMA busy_timeout=250")
        mode_row = connection.execute("PRAGMA journal_mode").fetchone()
        mode = "" if mode_row is None else str(mode_row[0]).lower()
        if mode == "wal":
            checkpoint = connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if (
                checkpoint is None
                or len(checkpoint) != 3
                or int(checkpoint[0]) != 0
            ):
                raise ReceiptFinalizationError(
                    f"fetch-state WAL checkpoint is busy: {checkpoint}"
                )
            mode_row = connection.execute(
                "PRAGMA journal_mode=DELETE"
            ).fetchone()
            mode = "" if mode_row is None else str(mode_row[0]).lower()
        if mode != "delete":
            raise ReceiptFinalizationError(
                f"fetch-state journal mode did not freeze: {mode!r}"
            )
    except sqlite3.Error as exc:
        raise ReceiptFinalizationError(
            f"fetch state could not be frozen: {exc}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    _require_frozen_sqlite(path, label="fetch state")


def _expected_fetch_state_schema_sha256() -> str:
    connection = sqlite3.connect(":memory:")
    try:
        connection.row_factory = sqlite3.Row
        connection.executescript(_STATE_SCHEMA)
        return _sqlite_schema_sha256(connection)
    finally:
        connection.close()


def _fetch_state_logical_digest(connection: sqlite3.Connection) -> str:
    table_order = (
        ("settings", "key"),
        ("attempts", "repo,run_id,attempt"),
        ("members", "repo,run_id,attempt,archive_member"),
        ("request_ledger", "id"),
        ("binding_upgrades", "id"),
    )

    def records() -> Iterable[list[object]]:
        for table, order_by in table_order:
            for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order_by}"):
                values: list[object] = [table]
                for value in row:
                    if isinstance(value, bytes):
                        values.append(
                            {
                                "byte_size": len(value),
                                "sha256": _sha256_bytes(value),
                            }
                        )
                    else:
                        values.append(value)
                yield values

    return _hash_records("cppmega-ci-fetch-state-logical-v1", records())


def _canonical_summary(connection: sqlite3.Connection) -> tuple[dict[str, object], str]:
    statuses = {
        str(row["status"]): int(row["count"])
        for row in connection.execute(
            """
            SELECT status,COUNT(*) AS count
            FROM attempts GROUP BY status ORDER BY status
            """
        )
    }
    placeholders = ",".join("?" for _ in _TERMINAL_RECEIPT_STATES)
    totals = connection.execute(
        f"""
        SELECT COUNT(*) AS attempts,
               COALESCE(SUM(member_count),0) AS members,
               COALESCE(SUM(chunk_count),0) AS chunks,
               COALESCE(SUM(occurrence_tokens),0) AS occurrence_tokens
        FROM attempts WHERE status IN ({placeholders})
        """,
        _TERMINAL_RECEIPT_STATES,
    ).fetchone()
    member_totals = connection.execute(
        """
        SELECT COUNT(*) AS members,
               COALESCE(SUM(chunk_count),0) AS chunks,
               COALESCE(SUM(occurrence_tokens),0) AS occurrence_tokens
        FROM members
        """
    ).fetchone()
    if totals is None or member_totals is None:
        raise ReceiptFinalizationError("fetch-state accounting is missing")
    if (
        int(member_totals["members"]) != int(totals["members"])
        or int(member_totals["chunks"]) != int(totals["chunks"])
        or int(member_totals["occurrence_tokens"]) != int(totals["occurrence_tokens"])
    ):
        raise ReceiptFinalizationError(
            "fetch-state attempt/member accounting is inconsistent"
        )

    sidecar_digest = hashlib.sha256()
    for row in connection.execute(
        """
        SELECT repo,run_id,attempt,archive_member,sidecar_sha256
        FROM members ORDER BY repo,run_id,attempt,archive_member
        """
    ):
        sidecar_digest.update(
            (
                f"{row['repo']}\t{row['run_id']}\t{row['attempt']}\t"
                f"{row['archive_member']}\t{row['sidecar_sha256']}\n"
            ).encode("utf-8")
        )
    sidecar_set_sha256 = sidecar_digest.hexdigest()
    return (
        {
            "attempt_statuses": statuses,
            "attempts_terminal": int(totals["attempts"]),
            "members": int(totals["members"]),
            "chunks": int(totals["chunks"]),
            "occurrence_tokens": int(totals["occurrence_tokens"]),
            "requests": int(
                connection.execute("SELECT COUNT(*) FROM request_ledger").fetchone()[0]
            ),
            "sidecar_set_sha256": sidecar_set_sha256,
        },
        sidecar_set_sha256,
    )


def _verify_cas_member_coverage(
    store: CIContentStore,
    state_path: Path,
) -> None:
    connection = sqlite3.connect(
        f"{state_path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        expected = {
            (
                str(row["repo"]),
                int(row["run_id"]),
                int(row["attempt"]),
                str(row["archive_member"]),
            ): (
                str(row["status"]),
                int(row["chunk_count"]),
                str(row["job_key"]),
            )
            for row in connection.execute(
                """
                SELECT members.repo,members.run_id,members.attempt,
                       members.archive_member,members.chunk_count,
                       members.job_key,
                       attempts.status
                FROM members
                JOIN attempts USING(repo,run_id,attempt)
                ORDER BY members.repo,members.run_id,members.attempt,
                         members.archive_member
                """
            )
        }
    finally:
        connection.close()

    observed: dict[tuple[str, int, int, str], set[int]] = {}
    for occurrence in store.iter_occurrences():
        provenance = occurrence.get("provenance")
        occurrence_key = occurrence.get("occurrence_key")
        if not isinstance(provenance, Mapping) or not isinstance(
            occurrence_key, Mapping
        ):
            raise ReceiptFinalizationError(
                "CAS occurrence lacks canonical provenance"
            )
        archive = provenance.get("archive")
        if not isinstance(archive, Mapping):
            raise ReceiptFinalizationError(
                "CAS occurrence lacks archive provenance"
            )
        repo = provenance.get("repository_scope_key")
        run_id = provenance.get("run_id")
        attempt = provenance.get("run_attempt")
        archive_member = archive.get("member")
        chunk = provenance.get("chunk")
        if (
            not isinstance(repo, str)
            or not repo
            or isinstance(run_id, bool)
            or not isinstance(run_id, int)
            or run_id <= 0
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt <= 0
            or not isinstance(archive_member, str)
            or not archive_member
            or not isinstance(chunk, Mapping)
        ):
            raise ReceiptFinalizationError(
                "CAS occurrence fetch identity is invalid"
            )
        ordinal = chunk.get("ordinal")
        section_id = str(chunk.get("section_id") or f"section:{ordinal}")
        step_ordinal = chunk.get("step_ordinal")
        expected_step = (
            f"{section_id}:"
            f"{step_ordinal if step_ordinal is not None else 'none'}"
        )
        if (
            provenance.get("schema") != "cppmega_ci_chunk_occurrence_v3"
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
            or occurrence_key.get("repo") != repo
            or occurrence_key.get("run_attempt") != f"{run_id}:{attempt}"
            or occurrence_key.get("chunk_ordinal") != ordinal
            or occurrence_key.get("step") != expected_step
        ):
            raise ReceiptFinalizationError(
                "CAS occurrence key differs from its provenance"
            )
        member_key = (repo, run_id, attempt, archive_member)
        state_member = expected.get(member_key)
        if state_member is None:
            raise ReceiptFinalizationError(
                f"CAS occurrence has no fetch-state member: {member_key}"
            )
        status, _expected_chunks, expected_job = state_member
        if status != "done":
            raise ReceiptFinalizationError(
                f"CAS occurrence belongs to non-done attempt: "
                f"{member_key + (status,)}"
            )
        if occurrence_key.get("job") != expected_job:
            raise ReceiptFinalizationError(
                f"CAS occurrence job differs from fetch state: {member_key}"
            )
        ordinals = observed.setdefault(member_key, set())
        if ordinal in ordinals:
            raise ReceiptFinalizationError(
                f"CAS contains duplicate member chunk ordinal: "
                f"{member_key + (ordinal,)}"
            )
        ordinals.add(ordinal)

    for member_key, (status, expected_chunks, _expected_job) in expected.items():
        actual_chunks = len(observed.get(member_key, set()))
        if status == "done" and actual_chunks != expected_chunks:
            raise ReceiptFinalizationError(
                "CAS/fetch-state member coverage differs: "
                f"{member_key + (expected_chunks, actual_chunks)}"
            )
        if status != "done" and actual_chunks:
            raise ReceiptFinalizationError(
                "non-done fetch-state member has CAS coverage: "
                f"{member_key + (status, actual_chunks)}"
            )


def _frozen_state_binding(
    state_path: Path,
    *,
    tokenizer: ExactTokenizer,
    store_receipt: Mapping[str, object],
    original_state_path: Path,
    original_store_path: Path,
    original_inventory_path: Path | None,
) -> tuple[dict[str, object], str]:
    _require_frozen_sqlite(state_path, label="fetch state")
    before = state_path.stat()
    connection = sqlite3.connect(
        f"{state_path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        integrity = [
            str(row[0])
            for row in connection.execute("PRAGMA integrity_check").fetchall()
        ]
        if integrity != ["ok"]:
            raise ReceiptFinalizationError(
                f"fetch-state integrity_check failed: {integrity}"
            )
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ReceiptFinalizationError("fetch-state foreign_key_check failed")
        schema_sha256 = _sqlite_schema_sha256(connection)
        if schema_sha256 != _expected_fetch_state_schema_sha256():
            raise ReceiptFinalizationError(
                "fetch-state SQLite schema is not the frozen v3 schema"
            )
        settings = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM settings ORDER BY key")
        }
        expected_setting_keys = {
            "schema",
            "inventory_path",
            "content_store_path",
            "tokenizer_contract",
            "tokenizer_fingerprint",
            "fetcher_script_sha256",
            "parser_script_sha256",
            "content_store_script_sha256",
            "chunk_semantics",
            "created_at",
        }
        if set(settings) != expected_setting_keys:
            raise ReceiptFinalizationError(
                "fetch-state settings do not match the frozen v3 contract"
            )
        if settings["schema"] != SCHEMA_VERSION:
            raise ReceiptFinalizationError("fetch-state schema is unsupported")
        try:
            tokenizer_contract = json.loads(settings["tokenizer_contract"])
        except json.JSONDecodeError as exc:
            raise ReceiptFinalizationError(
                "fetch-state tokenizer contract is invalid"
            ) from exc
        if (
            tokenizer_contract != tokenizer.contract
            or settings["tokenizer_fingerprint"] != tokenizer.fingerprint
        ):
            raise ReceiptFinalizationError("fetch-state tokenizer binding differs")
        if Path(settings["content_store_path"]).resolve() != original_store_path:
            raise ReceiptFinalizationError(
                "fetch-state content-store path differs from the original path"
            )
        if (
            original_inventory_path is not None
            and Path(settings["inventory_path"]).resolve() != original_inventory_path
        ):
            raise ReceiptFinalizationError(
                "fetch-state inventory path differs from the original path"
            )
        if settings["content_store_script_sha256"] != store_receipt.get(
            "script_sha256"
        ):
            raise ReceiptFinalizationError(
                "fetch-state content-store script binding differs"
            )
        for field in (
            "fetcher_script_sha256",
            "parser_script_sha256",
            "content_store_script_sha256",
        ):
            if _HEX64_RE.fullmatch(settings[field]) is None:
                raise ReceiptFinalizationError(
                    f"fetch-state setting {field} is not a SHA-256"
                )
        if settings["chunk_semantics"] != (
            "parser-dedup-text-cppmega-training-tokenizer-payload-only-no-framing-v2"
        ):
            raise ReceiptFinalizationError(
                "fetch-state chunk semantics are unsupported"
            )
        processing = connection.execute(
            "SELECT repo,run_id,attempt FROM attempts WHERE status='processing' LIMIT 1"
        ).fetchone()
        if processing is not None:
            raise ReceiptFinalizationError(
                f"processing attempt cannot be frozen: {tuple(processing)}"
            )
        cas_non_done = connection.execute(
            """
            SELECT repo,run_id,attempt,status FROM attempts
            WHERE status!='done' AND (chunk_count>0 OR occurrence_tokens>0)
            LIMIT 1
            """
        ).fetchone()
        if cas_non_done is not None:
            raise ReceiptFinalizationError(
                f"CAS-bearing non-done attempt cannot be frozen: {tuple(cas_non_done)}"
            )
        summary, sidecar_set_sha256 = _canonical_summary(connection)
        logical_sha256 = _fetch_state_logical_digest(connection)
    finally:
        connection.close()

    artifact_sha256 = _sha256_file(state_path)
    after = state_path.stat()
    before_identity = (before.st_size, before.st_mtime_ns, before.st_ino)
    after_identity = (after.st_size, after.st_mtime_ns, after.st_ino)
    if before_identity != after_identity:
        raise ReceiptFinalizationError(
            "fetch-state artifact changed during finalization"
        )
    return (
        {
            "schema": SCHEMA_VERSION,
            "artifact": {
                "path": str(original_state_path),
                "byte_size": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "inode": after.st_ino,
                "sha256": artifact_sha256,
            },
            "sqlite_schema_sha256": schema_sha256,
            "sqlite_logical_sha256": logical_sha256,
            "settings": dict(sorted(settings.items())),
            "summary": summary,
            "sidecar_set_sha256": sidecar_set_sha256,
        },
        settings["inventory_path"],
    )


def _default_store_receipt_path(fetch_receipt_path: Path) -> Path:
    return fetch_receipt_path.with_name(f"{fetch_receipt_path.stem}.store.json")


def _resolved_non_symlink_path(
    value: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ReceiptFinalizationError(f"{label} cannot be a symlink: {candidate}")
    return candidate.resolve()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def finalize_fetch_receipts(
    *,
    state_path: str | os.PathLike[str],
    content_store_path: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str],
    target_unique_tokens: int,
    fetch_receipt_path: str | os.PathLike[str],
    store_receipt_path: str | os.PathLike[str] | None = None,
    original_state_path: str | os.PathLike[str] | None = None,
    original_content_store_path: str | os.PathLike[str] | None = None,
    original_inventory_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Close-time finalization for one immutable fetch-state/CAS pair."""

    if (
        isinstance(target_unique_tokens, bool)
        or not isinstance(target_unique_tokens, int)
        or target_unique_tokens <= 0
    ):
        raise ValueError("target_unique_tokens must be a positive integer")
    state = _resolved_non_symlink_path(state_path, label="fetch state")
    store_root = _resolved_non_symlink_path(
        content_store_path,
        label="content store",
    )
    fetch_receipt = _resolved_non_symlink_path(
        fetch_receipt_path,
        label="fetch receipt",
    )
    store_receipt = (
        _resolved_non_symlink_path(
            _default_store_receipt_path(fetch_receipt),
            label="content-store receipt",
        )
        if store_receipt_path is None
        else _resolved_non_symlink_path(
            store_receipt_path,
            label="content-store receipt",
        )
    )
    original_state = (
        state
        if original_state_path is None
        else Path(original_state_path).expanduser().resolve()
    )
    original_store = (
        store_root
        if original_content_store_path is None
        else Path(original_content_store_path).expanduser().resolve()
    )
    original_inventory = (
        None
        if original_inventory_path is None
        else Path(original_inventory_path).expanduser().resolve()
    )
    index_path = store_root / "index.sqlite3"
    tokenizer_file = _resolved_non_symlink_path(
        tokenizer_path,
        label="tokenizer",
    )
    if fetch_receipt == store_receipt:
        raise ValueError("fetch and content-store receipt paths must differ")
    protected_inputs = {state, index_path, tokenizer_file}
    if fetch_receipt in protected_inputs or store_receipt in protected_inputs:
        raise ValueError("a receipt path collides with an immutable input")
    if _is_within(fetch_receipt, store_root) or _is_within(
        store_receipt,
        store_root,
    ):
        raise ValueError("receipt paths must be outside the content store")
    _freeze_fetch_state_sqlite(state)
    _require_frozen_sqlite(index_path, label="content store")
    initial_state = state.stat()
    initial_state_identity = (
        initial_state.st_size,
        initial_state.st_mtime_ns,
        initial_state.st_ino,
    )
    tokenizer = ExactTokenizer(tokenizer_file)

    store = CIContentStore(store_root)
    try:
        store_value = store.completion_receipt(
            target_unique_tokens=target_unique_tokens
        )
        _verify_cas_member_coverage(store, state)
    finally:
        store.close()
    _require_frozen_sqlite(index_path, label="content store")

    frozen_state, bound_inventory = _frozen_state_binding(
        state,
        tokenizer=tokenizer,
        store_receipt=store_value,
        original_state_path=original_state,
        original_store_path=original_store,
        original_inventory_path=original_inventory,
    )
    value = {
        "schema": RECEIPT_SCHEMA,
        "completed_at": _utc_now(),
        "target_exact_unique_payload_tokens": target_unique_tokens,
        "fetch_state": frozen_state["summary"],
        "frozen_fetch_state": frozen_state,
        "content_store_receipt": store_value,
        "inventory_path": bound_inventory,
        "tokenizer_contract": tokenizer.contract,
        "tokenizer_fingerprint": tokenizer.fingerprint,
    }
    final_state = state.stat()
    if (
        final_state.st_size,
        final_state.st_mtime_ns,
        final_state.st_ino,
    ) != initial_state_identity:
        raise ReceiptFinalizationError(
            "fetch-state artifact changed during receipt finalization"
        )
    atomic_write_json(store_receipt, store_value)
    atomic_write_json(fetch_receipt, value)
    _require_frozen_sqlite(state, label="fetch state")
    _require_frozen_sqlite(index_path, label="content store")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize frozen store/fetch receipts after every stream writer exits"
        )
    )
    parser.add_argument("--state", required=True)
    parser.add_argument("--content-store", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-exact-unique-payload-tokens", required=True, type=int)
    parser.add_argument("--fetch-receipt", required=True)
    parser.add_argument("--store-receipt")
    parser.add_argument("--original-state-path")
    parser.add_argument("--original-content-store-path")
    parser.add_argument("--original-inventory-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        value = finalize_fetch_receipts(
            state_path=args.state,
            content_store_path=args.content_store,
            tokenizer_path=args.tokenizer,
            target_unique_tokens=args.target_exact_unique_payload_tokens,
            fetch_receipt_path=args.fetch_receipt,
            store_receipt_path=args.store_receipt,
            original_state_path=args.original_state_path,
            original_content_store_path=args.original_content_store_path,
            original_inventory_path=args.original_inventory_path,
        )
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
        print(f"[ci-stream-receipts] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
