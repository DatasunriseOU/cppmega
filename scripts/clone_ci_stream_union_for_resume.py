#!/usr/bin/env python3
"""Deep-clone a verified CI union into mutable continuation state.

The published source bundle remains immutable. The destination inventory,
fetch state, and content store are byte copies (never hard links); only the
destination fetch-state path settings and copied inventory-receipt paths are
rewritten. Receipt-relative immutable controls preserve the base state, CAS
index, tokenizer, inventories, and receipts so inclusion remains verifiable
after relocation without the source tree. A seed receipt binds those controls
and proves the logical base rows/CAS were conserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci_content_store import (  # noqa: E402
    _hash_records,
    _script_sha256 as _content_store_script_sha256,
)
from scripts.ci_stream_inventory import (  # noqa: E402
    CompletionError as InventoryCompletionError,
    atomic_write_json,
    verify_inventory_completion_receipt,
)
from scripts.export_ci_content_store_case5 import (  # noqa: E402
    ExactTokenizer,
    ExportError,
    FrozenFetchState,
    FrozenStore,
    _fetch_state_logical_digest,
    _fsync_tree,
    _publish_directory_no_replace,
    _sha256_file,
)
from scripts.merge_ci_stream_shards import (  # noqa: E402
    MERGE_RECEIPT_SCHEMA,
    PRODUCTION_MERGE_RECEIPT_SCHEMA,
    MergeError,
    _canonical_json_bytes,
    _load_json as _load_merge_json,
    _require_complete_bundle_tree,
    _sha256_bytes,
    _utc_now,
)


SEED_RECEIPT_SCHEMA = "cppmega_ci_stream_continuation_seed_receipt_v3"
_INVENTORY_NAME = "inventory.sqlite3"
_INVENTORY_RECEIPT_NAME = "inventory_receipt.json"
_STORE_NAME = "content_store"
_STORE_RECEIPT_NAME = "store_receipt.json"
_STATE_NAME = "fetch_state.sqlite3"
_FETCH_RECEIPT_NAME = "fetch_receipt.json"
_MERGE_RECEIPT_NAME = "merge_receipt.json"
_SEED_RECEIPT_NAME = "continuation_seed_receipt.json"
_CONTROL_DIRECTORY_NAME = "continuation_seed_controls"
_CONTROL_BASE_NAME = "base_union"
_CONTROL_INVENTORY_NAME = "continuation_inventory"
_CONTROL_TOKENIZER_NAME = "tokenizer.json"
_CONTENT_STORE_INDEX_NAME = "index.sqlite3"
_RELATIVE_PATH_SEMANTICS = "seed-receipt-parent-relative-posix-v1"


class CloneError(RuntimeError):
    """A base union cannot safely seed mutable continuation state."""


def _load_json(path: Path, *, where: str) -> tuple[dict[str, Any], str]:
    try:
        value, raw = _load_merge_json(path, where=where)
    except MergeError as exc:
        raise CloneError(str(exc)) from exc
    return value, _sha256_bytes(raw)


def _iter_tree_files(root: Path) -> Iterable[Path]:
    if root.is_symlink() or not root.is_dir():
        raise CloneError(f"bundle is missing or unsafe: {root}")

    def walk(directory: Path) -> Iterable[Path]:
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_symlink():
                raise CloneError(f"bundle contains a symlink: {path}")
            if path.is_dir():
                yield from walk(path)
            elif path.is_file():
                yield path
            else:
                raise CloneError(
                    f"bundle contains an unsafe artifact: {path}"
                )

    yield from walk(root)


def _snapshot_tree(root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    digest.update(b"cppmega-ci-continuation-tree-v3\0")
    file_count = 0
    byte_size = 0
    for path in _iter_tree_files(root):
        relative = path.relative_to(root).as_posix()
        stat_before = path.stat()
        file_sha256 = _sha256_file(path)
        stat_after = path.stat()
        if (
            stat_before.st_size,
            stat_before.st_mtime_ns,
            stat_before.st_ino,
        ) != (
            stat_after.st_size,
            stat_after.st_mtime_ns,
            stat_after.st_ino,
        ):
            raise CloneError(
                f"bundle artifact changed while it was hashed: {path}"
            )
        record = _canonical_json_bytes(
            {
                "path": relative,
                "byte_size": stat_after.st_size,
                "sha256": file_sha256,
            }
        )
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
        file_count += 1
        byte_size += stat_after.st_size
    return {
        "file_count": file_count,
        "byte_size": byte_size,
        "artifact_set_sha256": digest.hexdigest(),
    }


def _source_artifact_set_from_merge_receipt(
    receipt: Mapping[str, Any],
    *,
    merge_receipt_byte_size: int,
    merge_receipt_sha256: str,
) -> dict[str, object]:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise CloneError("base merge receipt lacks its artifact manifest")
    records = [
        {
            "path": str(artifact["path"]),
            "byte_size": int(artifact["byte_size"]),
            "sha256": str(artifact["sha256"]),
        }
        for artifact in artifacts
        if isinstance(artifact, Mapping)
    ]
    if len(records) != len(artifacts):
        raise CloneError("base merge receipt artifact manifest is malformed")
    records.append(
        {
            "path": _MERGE_RECEIPT_NAME,
            "byte_size": merge_receipt_byte_size,
            "sha256": merge_receipt_sha256,
        }
    )
    digest = hashlib.sha256()
    digest.update(b"cppmega-ci-continuation-tree-v3\0")
    byte_size = 0
    for record in sorted(records, key=lambda item: str(item["path"])):
        encoded = _canonical_json_bytes(record)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        byte_size += int(record["byte_size"])
    return {
        "file_count": len(records),
        "byte_size": byte_size,
        "artifact_set_sha256": digest.hexdigest(),
    }


def _logical_table_digest(
    connection: sqlite3.Connection,
    table: str,
    order_by: str,
) -> tuple[int, str]:
    exists = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table,),
    ).fetchone()
    if exists is None:
        return 0, _hash_records(
            f"cppmega-ci-continuation-{table}-v1",
            iter(()),
        )

    def records() -> Iterable[list[object]]:
        for row in connection.execute(
            f"SELECT * FROM {table} ORDER BY {order_by}"
        ):
            values: list[object] = []
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

    count = int(
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    )
    return count, _hash_records(
        f"cppmega-ci-continuation-{table}-v1",
        records(),
    )


def _state_inclusion_projection(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        tables = {
            "attempts": "repo,run_id,attempt",
            "members": "repo,run_id,attempt,archive_member",
            "request_ledger": "id",
            "binding_upgrades": "id",
        }
        projection = {
            table: {
                "row_count": count,
                "logical_sha256": digest,
            }
            for table, order_by in tables.items()
            for count, digest in [
                _logical_table_digest(connection, table, order_by)
            ]
        }
        settings = {
            str(row["key"]): str(row["value"])
            for row in connection.execute(
                "SELECT key,value FROM settings ORDER BY key"
            )
        }
        return {
            "tables": projection,
            "settings_without_mutable_paths_sha256": _sha256_bytes(
                json.dumps(
                    {
                        key: value
                        for key, value in settings.items()
                        if key not in {"inventory_path", "content_store_path"}
                    },
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ),
            "full_logical_sha256": _fetch_state_logical_digest(connection),
        }
    finally:
        connection.close()


def _rewrite_state_paths(
    state_path: Path,
    *,
    inventory_path: Path,
    store_path: Path,
) -> None:
    connection = sqlite3.connect(state_path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE settings SET value=? WHERE key='inventory_path'",
            (str(inventory_path),),
        )
        connection.execute(
            "UPDATE settings SET value=? WHERE key='content_store_path'",
            (str(store_path),),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _rewrite_inventory_receipt(
    source_receipt: Mapping[str, Any],
    destination_receipt: Path,
    *,
    destination_inventory: Path,
) -> dict[str, Any]:
    value = json.loads(
        json.dumps(source_receipt, allow_nan=False, ensure_ascii=False)
    )
    if not isinstance(value, dict):
        raise CloneError("inventory receipt copy is invalid")
    artifact = value.get("database_artifact")
    if not isinstance(artifact, dict):
        raise CloneError("inventory receipt lacks database_artifact")
    value["database"] = str(destination_inventory)
    artifact["path"] = str(destination_inventory)
    atomic_write_json(destination_receipt, value)
    return value


def _inventory_scope_projection(
    receipt: Mapping[str, Any],
    *,
    where: str,
) -> dict[str, object]:
    repo_list = receipt.get("repo_list")
    interval = receipt.get("interval")
    if not isinstance(repo_list, Mapping) or not isinstance(
        interval,
        Mapping,
    ):
        raise CloneError(f"{where} lacks its repository scope or interval")
    try:
        return {
            "repo_list_sha256": str(repo_list["sha256"]),
            "repo_scope_sha256": str(repo_list["scope_sha256"]),
            "repos": int(repo_list["repos"]),
            "original_repos": int(repo_list["original_repos"]),
            "unresolved": int(repo_list["unresolved"]),
            "interval": {
                "start": str(interval["start"]),
                "end": str(interval["end"]),
                "semantics": str(interval["semantics"]),
            },
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise CloneError(
            f"{where} has a malformed repository scope or interval"
        ) from exc


def _verify_inventory_compatibility(
    *,
    base_inventory: Path,
    base_inventory_receipt: Mapping[str, Any],
    base_inventory_receipt_sha256: str,
    base_state: Path,
    selected_inventory: Path,
    selected_inventory_receipt: Mapping[str, Any],
) -> dict[str, object]:
    """Prove the selected production inventory can safely extend the base."""

    base_scope = _inventory_scope_projection(
        base_inventory_receipt,
        where="base inventory receipt",
    )
    selected_scope = _inventory_scope_projection(
        selected_inventory_receipt,
        where="continuation inventory receipt",
    )
    if selected_scope != base_scope:
        raise CloneError(
            "continuation inventory repository scope or declared interval "
            "differs from the base inventory"
        )

    connection = sqlite3.connect(
        f"{selected_inventory.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(
            "ATTACH DATABASE ? AS base_inventory",
            (f"{base_inventory.as_uri()}?mode=ro&immutable=1",),
        )
        connection.execute(
            "ATTACH DATABASE ? AS base_state",
            (f"{base_state.as_uri()}?mode=ro&immutable=1",),
        )

        repository_mismatch = connection.execute(
            """
            SELECT
              base.repo_key AS base_repo,
              selected.repo_key AS selected_repo
            FROM base_inventory.repos AS base
            LEFT JOIN main.repos AS selected
              ON selected.repo_key=base.repo_key
             AND selected.owner=base.owner
             AND selected.name=base.name
             AND selected.canonical=base.canonical
             AND selected.ordinal=base.ordinal
            WHERE selected.repo_key IS NULL
            UNION ALL
            SELECT
              base.repo_key AS base_repo,
              selected.repo_key AS selected_repo
            FROM main.repos AS selected
            LEFT JOIN base_inventory.repos AS base
              ON base.repo_key=selected.repo_key
             AND base.owner=selected.owner
             AND base.name=selected.name
             AND base.canonical=selected.canonical
             AND base.ordinal=selected.ordinal
            WHERE base.repo_key IS NULL
            LIMIT 1
            """
        ).fetchone()
        if repository_mismatch is not None:
            raise CloneError(
                "continuation inventory repository rows are missing, "
                "shrunk, or unrelated to the base scope"
            )

        base_run_mismatch = connection.execute(
            """
            SELECT
              base.repo_key,
              base.run_id,
              base.run_attempt AS base_run_attempt,
              selected.run_attempt AS selected_run_attempt
            FROM base_inventory.runs AS base
            LEFT JOIN main.runs AS selected
              ON selected.repo_key=base.repo_key
             AND selected.run_id=base.run_id
            WHERE selected.run_id IS NULL
               OR selected.run_attempt < base.run_attempt
            ORDER BY base.repo_key,base.run_id
            LIMIT 1
            """
        ).fetchone()
        if base_run_mismatch is not None:
            repo = str(base_run_mismatch["repo_key"])
            run_id = int(base_run_mismatch["run_id"])
            selected_ceiling = base_run_mismatch["selected_run_attempt"]
            if selected_ceiling is None:
                raise CloneError(
                    "continuation inventory is missing base run "
                    f"{repo}#{run_id}"
                )
            raise CloneError(
                "continuation inventory run_attempt ceiling shrank for "
                f"{repo}#{run_id}: "
                f"{int(selected_ceiling)} < "
                f"{int(base_run_mismatch['base_run_attempt'])}"
            )

        attempt_mismatch = connection.execute(
            """
            SELECT
              attempts.repo,
              attempts.run_id,
              attempts.max_attempt,
              attempts.max_inventory_seed_attempt,
              selected.run_attempt AS selected_run_attempt
            FROM (
              SELECT
                repo,
                run_id,
                MAX(attempt) AS max_attempt,
                MAX(inventory_seed_attempt)
                  AS max_inventory_seed_attempt
              FROM base_state.attempts
              GROUP BY repo,run_id
            ) AS attempts
            LEFT JOIN main.runs AS selected
              ON selected.repo_key=attempts.repo
             AND selected.run_id=attempts.run_id
            WHERE selected.run_id IS NULL
               OR selected.run_attempt < attempts.max_attempt
               OR selected.run_attempt
                    < attempts.max_inventory_seed_attempt
            ORDER BY attempts.repo,attempts.run_id
            LIMIT 1
            """
        ).fetchone()
        if attempt_mismatch is not None:
            repo = str(attempt_mismatch["repo"])
            run_id = int(attempt_mismatch["run_id"])
            selected_ceiling = attempt_mismatch["selected_run_attempt"]
            if selected_ceiling is None:
                raise CloneError(
                    "continuation inventory cannot anchor base fetch run "
                    f"{repo}#{run_id}"
                )
            required_ceiling = max(
                int(attempt_mismatch["max_attempt"]),
                int(attempt_mismatch["max_inventory_seed_attempt"]),
            )
            raise CloneError(
                "continuation inventory run_attempt ceiling cannot anchor "
                f"base fetch run {repo}#{run_id}: "
                f"{int(selected_ceiling)} < {required_ceiling}"
            )

        current_seed_mismatch = connection.execute(
            """
            SELECT
              attempts.repo,
              attempts.run_id,
              attempts.attempt
            FROM base_state.attempts AS attempts
            JOIN main.runs AS selected
              ON selected.repo_key=attempts.repo
             AND selected.run_id=attempts.run_id
            WHERE selected.run_attempt=attempts.inventory_seed_attempt
              AND selected.metadata_sha256
                    != attempts.inventory_seed_metadata_sha256
            ORDER BY attempts.repo,attempts.run_id,attempts.attempt
            LIMIT 1
            """
        ).fetchone()
        if current_seed_mismatch is not None:
            raise CloneError(
                "continuation inventory current-ceiling metadata cannot "
                "anchor base fetch attempt "
                f"{current_seed_mismatch['repo']}#"
                f"{int(current_seed_mismatch['run_id'])}/"
                f"{int(current_seed_mismatch['attempt'])}"
            )

        base_run_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM base_inventory.runs"
            ).fetchone()[0]
        )
        selected_run_count = int(
            connection.execute("SELECT COUNT(*) FROM main.runs").fetchone()[0]
        )
        base_fetch_run_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                  SELECT repo,run_id
                  FROM base_state.attempts
                  GROUP BY repo,run_id
                )
                """
            ).fetchone()[0]
        )
        base_fetch_attempt_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM base_state.attempts"
            ).fetchone()[0]
        )
    except sqlite3.Error as exc:
        raise CloneError(
            f"continuation inventory compatibility proof failed: {exc}"
        ) from exc
    finally:
        connection.close()

    return {
        "semantics": "exact-scope-interval-and-per-run-ceiling-superset-v1",
        "base_inventory_receipt_sha256": (
            base_inventory_receipt_sha256
        ),
        "repository_scope": base_scope,
        "base_run_count": base_run_count,
        "selected_run_count": selected_run_count,
        "base_fetch_run_count": base_fetch_run_count,
        "base_fetch_attempt_count": base_fetch_attempt_count,
        "repository_rows_equal": True,
        "base_runs_preserved": True,
        "base_fetch_attempts_anchored": True,
        "run_attempt_ceilings_not_shrunk": True,
        "current_ceiling_metadata_bindings_equal": True,
    }


def clone_union_for_resume(
    *,
    base_union: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str],
    inventory_path: str | os.PathLike[str] | None = None,
    inventory_receipt_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Verify, deep-copy, path-rewrite, and atomically publish one seed."""

    base_input = Path(base_union).expanduser()
    target_input = Path(destination).expanduser()
    if base_input.is_symlink() or not base_input.is_dir():
        raise CloneError(f"base union is missing or unsafe: {base_input}")
    if target_input.exists() or target_input.is_symlink():
        raise CloneError(f"destination already exists: {target_input}")
    base = base_input.resolve()
    target = target_input.resolve()
    tokenizer_input = Path(tokenizer_path).expanduser()
    if tokenizer_input.is_symlink() or not tokenizer_input.is_file():
        raise CloneError(
            f"tokenizer is missing or unsafe: {tokenizer_input}"
        )
    tokenizer_file = tokenizer_input.resolve()
    if target == base or base in target.parents or target in base.parents:
        raise CloneError("base and destination trees must not contain each other")
    merge_receipt_path = base / _MERGE_RECEIPT_NAME
    merge_receipt, merge_receipt_sha256 = _load_json(
        merge_receipt_path,
        where="base merge receipt",
    )
    merge_receipt_byte_size = merge_receipt_path.stat().st_size
    if (
        merge_receipt.get("schema")
        not in {MERGE_RECEIPT_SCHEMA, PRODUCTION_MERGE_RECEIPT_SCHEMA}
        or merge_receipt.get("status") != "complete"
    ):
        raise CloneError("base merge receipt is unsupported or incomplete")
    try:
        _require_complete_bundle_tree(base, merge_receipt)
    except MergeError as exc:
        raise CloneError(f"base union artifact verification failed: {exc}") from exc

    base_inventory = base / _INVENTORY_NAME
    base_inventory_receipt = base / _INVENTORY_RECEIPT_NAME
    try:
        (
            verified_base_inventory_receipt,
            base_inventory_receipt_sha256,
        ) = verify_inventory_completion_receipt(
            base_inventory,
            base_inventory_receipt,
            require_production=True,
            expected_original_database_path=base_inventory,
        )
    except InventoryCompletionError as exc:
        raise CloneError(
            f"base union production inventory refused: {exc}"
        ) from exc
    if (inventory_path is None) != (inventory_receipt_path is None):
        raise CloneError(
            "inventory_path and inventory_receipt_path must be supplied together"
        )
    selected_inventory_input = (
        base_inventory
        if inventory_path is None
        else Path(inventory_path).expanduser()
    )
    selected_inventory_receipt_input = (
        base_inventory_receipt
        if inventory_receipt_path is None
        else Path(inventory_receipt_path).expanduser()
    )
    try:
        selected_receipt, selected_receipt_sha256 = (
            verify_inventory_completion_receipt(
                selected_inventory_input,
                selected_inventory_receipt_input,
                require_production=True,
                expected_original_database_path=selected_inventory_input,
            )
        )
    except InventoryCompletionError as exc:
        raise CloneError(f"continuation inventory refused: {exc}") from exc
    selected_inventory = selected_inventory_input.resolve()
    selected_inventory_receipt = selected_inventory_receipt_input.resolve()

    base_store = base / _STORE_NAME
    base_store_receipt = base / _STORE_RECEIPT_NAME
    base_state = base / _STATE_NAME
    base_fetch_receipt = base / _FETCH_RECEIPT_NAME
    fetch_receipt, fetch_receipt_sha256 = _load_json(
        base_fetch_receipt,
        where="base fetch receipt",
    )
    tokenizer_before = {
        "byte_size": tokenizer_file.stat().st_size,
        "sha256": _sha256_file(tokenizer_file),
    }
    tokenizer = ExactTokenizer(tokenizer_file)
    with FrozenStore(base_store, base_store_receipt) as store:
        with FrozenFetchState(
            base_state,
            tokenizer=tokenizer,
            store=store,
        ) as state:
            if fetch_receipt.get("fetch_state") != state.summary:
                raise CloneError(
                    "base fetch receipt summary differs from frozen state"
                )
            if fetch_receipt.get("content_store_receipt") != store.receipt:
                raise CloneError(
                    "base fetch receipt differs from frozen content store"
                )
            base_state_projection = _state_inclusion_projection(base_state)
            base_store_binding = {
                "receipt_sha256": _sha256_file(base_store_receipt),
                "sqlite_logical_sha256": store.receipt[
                    "sqlite_logical_sha256"
                ],
                "logical_content_set_sha256": store.receipt[
                    "logical_content_set_sha256"
                ],
                "occurrence_set_sha256": store.receipt[
                    "occurrence_set_sha256"
                ],
                "logical_token_sequence_set_sha256": store.receipt[
                    "logical_token_sequence_set_sha256"
                ],
                "counters": store.receipt["counters"],
            }
            runtime_store_script_sha256 = _content_store_script_sha256()
            if (
                store.receipt.get("script_sha256")
                != runtime_store_script_sha256
            ):
                raise CloneError(
                    "base content store is not continuation-writable: "
                    "creator_script_sha256 differs from the runtime "
                    "ci_content_store.py hash; use an explicit logical "
                    "content-store migration before cloning"
                )
            state.require_unchanged()
        store.require_unchanged()

    inventory_compatibility = _verify_inventory_compatibility(
        base_inventory=base_inventory,
        base_inventory_receipt=verified_base_inventory_receipt,
        base_inventory_receipt_sha256=(
            base_inventory_receipt_sha256
        ),
        base_state=base_state,
        selected_inventory=selected_inventory,
        selected_inventory_receipt=selected_receipt,
    )
    source_before = _snapshot_tree(base)
    expected_source_artifact_set = (
        _source_artifact_set_from_merge_receipt(
            merge_receipt,
            merge_receipt_byte_size=merge_receipt_byte_size,
            merge_receipt_sha256=merge_receipt_sha256,
        )
    )
    if source_before != expected_source_artifact_set:
        raise CloneError(
            "base union content differs from its verified merge artifact "
            "manifest"
        )
    selected_inventory_before = {
        "byte_size": selected_inventory.stat().st_size,
        "sha256": _sha256_file(selected_inventory),
    }
    selected_receipt_before = {
        "byte_size": selected_inventory_receipt.stat().st_size,
        "sha256": selected_receipt_sha256,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.partial-",
            dir=target.parent,
        )
    )
    published = False
    try:
        destination_inventory = partial / _INVENTORY_NAME
        destination_inventory_receipt = partial / _INVENTORY_RECEIPT_NAME
        destination_store = partial / _STORE_NAME
        destination_store_receipt = partial / _STORE_RECEIPT_NAME
        destination_state = partial / _STATE_NAME
        controls = partial / _CONTROL_DIRECTORY_NAME
        control_base = controls / _CONTROL_BASE_NAME
        control_base_store = control_base / _STORE_NAME
        control_inventory = controls / _CONTROL_INVENTORY_NAME
        control_tokenizer = controls / _CONTROL_TOKENIZER_NAME
        control_base_store.mkdir(parents=True)
        control_inventory.mkdir()
        shutil.copy2(
            base_state,
            control_base / _STATE_NAME,
        )
        shutil.copy2(
            base_store / _CONTENT_STORE_INDEX_NAME,
            control_base_store / _CONTENT_STORE_INDEX_NAME,
        )
        for source, destination_name in (
            (merge_receipt_path, _MERGE_RECEIPT_NAME),
            (base_fetch_receipt, _FETCH_RECEIPT_NAME),
            (base_store_receipt, _STORE_RECEIPT_NAME),
            (base_inventory, _INVENTORY_NAME),
            (base_inventory_receipt, _INVENTORY_RECEIPT_NAME),
        ):
            shutil.copy2(source, control_base / destination_name)
        shutil.copy2(
            selected_inventory,
            control_inventory / _INVENTORY_NAME,
        )
        shutil.copy2(
            selected_inventory_receipt,
            control_inventory / _INVENTORY_RECEIPT_NAME,
        )
        shutil.copy2(tokenizer_file, control_tokenizer)

        shutil.copy2(selected_inventory, destination_inventory)
        _rewrite_inventory_receipt(
            selected_receipt,
            destination_inventory_receipt,
            destination_inventory=target / _INVENTORY_NAME,
        )
        shutil.copytree(
            base_store,
            destination_store,
            copy_function=shutil.copy2,
            symlinks=False,
        )
        shutil.copy2(base_store_receipt, destination_store_receipt)
        shutil.copy2(base_state, destination_state)
        _rewrite_state_paths(
            destination_state,
            inventory_path=target / _INVENTORY_NAME,
            store_path=target / _STORE_NAME,
        )

        destination_projection = _state_inclusion_projection(
            destination_state
        )
        if (
            destination_projection["tables"]
            != base_state_projection["tables"]
            or destination_projection[
                "settings_without_mutable_paths_sha256"
            ]
            != base_state_projection[
                "settings_without_mutable_paths_sha256"
            ]
        ):
            raise CloneError(
                "destination state does not conserve the logical base rows"
            )
        hardlink_failures: list[str] = []
        for relative, source, copied in (
            (_INVENTORY_NAME, selected_inventory, destination_inventory),
            (_STATE_NAME, base_state, destination_state),
            (
                _STORE_RECEIPT_NAME,
                base_store_receipt,
                destination_store_receipt,
            ),
            (
                f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_TOKENIZER_NAME}",
                tokenizer_file,
                control_tokenizer,
            ),
            (
                f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_BASE_NAME}/"
                f"{_STATE_NAME}",
                base_state,
                control_base / _STATE_NAME,
            ),
            (
                f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_BASE_NAME}/"
                f"{_STORE_NAME}/{_CONTENT_STORE_INDEX_NAME}",
                base_store / _CONTENT_STORE_INDEX_NAME,
                control_base_store / _CONTENT_STORE_INDEX_NAME,
            ),
            (
                f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_BASE_NAME}/"
                f"{_MERGE_RECEIPT_NAME}",
                merge_receipt_path,
                control_base / _MERGE_RECEIPT_NAME,
            ),
            (
                f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_BASE_NAME}/"
                f"{_FETCH_RECEIPT_NAME}",
                base_fetch_receipt,
                control_base / _FETCH_RECEIPT_NAME,
            ),
            (
                f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_BASE_NAME}/"
                f"{_STORE_RECEIPT_NAME}",
                base_store_receipt,
                control_base / _STORE_RECEIPT_NAME,
            ),
            (
                f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_BASE_NAME}/"
                f"{_INVENTORY_NAME}",
                base_inventory,
                control_base / _INVENTORY_NAME,
            ),
            (
                f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_BASE_NAME}/"
                f"{_INVENTORY_RECEIPT_NAME}",
                base_inventory_receipt,
                control_base / _INVENTORY_RECEIPT_NAME,
            ),
            (
                f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_INVENTORY_NAME}/"
                f"{_INVENTORY_NAME}",
                selected_inventory,
                control_inventory / _INVENTORY_NAME,
            ),
            (
                f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_INVENTORY_NAME}/"
                f"{_INVENTORY_RECEIPT_NAME}",
                selected_inventory_receipt,
                control_inventory / _INVENTORY_RECEIPT_NAME,
            ),
        ):
            if (
                source.stat().st_dev,
                source.stat().st_ino,
            ) == (
                copied.stat().st_dev,
                copied.stat().st_ino,
            ):
                hardlink_failures.append(relative)
        for source in _iter_tree_files(base_store):
            relative_path = source.relative_to(base_store)
            copied = destination_store / relative_path
            if (
                source.stat().st_dev,
                source.stat().st_ino,
            ) == (
                copied.stat().st_dev,
                copied.stat().st_ino,
            ):
                if len(hardlink_failures) < 10:
                    hardlink_failures.append(relative_path.as_posix())
        if hardlink_failures:
            raise CloneError(
                "mutable clone contains hard links to the source: "
                f"{hardlink_failures[:10]}"
            )

        try:
            destination_receipt, _destination_receipt_sha256 = (
                verify_inventory_completion_receipt(
                    destination_inventory,
                    destination_inventory_receipt,
                    require_production=True,
                    expected_original_database_path=(
                        target / _INVENTORY_NAME
                    ),
                )
            )
        except InventoryCompletionError as exc:
            raise CloneError(
                f"destination inventory reverification failed: {exc}"
            ) from exc
        if (
            destination_receipt["db_logical_sha256"]
            != selected_receipt["db_logical_sha256"]
        ):
            raise CloneError(
                "destination inventory logical digest differs from its source"
            )
        try:
            with FrozenStore(
                destination_store,
                destination_store_receipt,
            ) as destination_frozen_store:
                with FrozenFetchState(
                    destination_state,
                    tokenizer=tokenizer,
                    store=destination_frozen_store,
                    bound_store_path=target / _STORE_NAME,
                ) as destination_frozen_state:
                    if (
                        destination_frozen_state.summary
                        != fetch_receipt.get("fetch_state")
                    ):
                        raise CloneError(
                            "destination frozen fetch state differs from "
                            "the verified base summary"
                        )
                    destination_frozen_state.require_unchanged()
                destination_frozen_store.require_unchanged()
        except ExportError as exc:
            raise CloneError(
                f"destination store/state reverification failed: {exc}"
            ) from exc

        source_after = _snapshot_tree(base)
        if source_after != source_before:
            raise CloneError("base union changed while it was cloned")
        if {
            "byte_size": selected_inventory.stat().st_size,
            "sha256": _sha256_file(selected_inventory),
        } != selected_inventory_before:
            raise CloneError("selected inventory changed while it was cloned")
        if {
            "byte_size": selected_inventory_receipt.stat().st_size,
            "sha256": _sha256_file(selected_inventory_receipt),
        } != selected_receipt_before:
            raise CloneError(
                "selected inventory receipt changed while it was cloned"
            )
        if {
            "byte_size": tokenizer_file.stat().st_size,
            "sha256": _sha256_file(tokenizer_file),
        } != tokenizer_before:
            raise CloneError("tokenizer changed while continuation was cloned")
        if _content_store_script_sha256() != runtime_store_script_sha256:
            raise CloneError(
                "runtime ci_content_store.py changed while continuation "
                "writability was verified"
            )

        controls_before = _snapshot_tree(controls)
        control_base_relative = (
            f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_BASE_NAME}"
        )
        control_inventory_relative = (
            f"{_CONTROL_DIRECTORY_NAME}/{_CONTROL_INVENTORY_NAME}"
        )
        seed_receipt: dict[str, Any] = {
            "schema": SEED_RECEIPT_SCHEMA,
            "semantics": (
                "portable-content-bound-and-reverified-mutable-clone-v3"
            ),
            "path_semantics": _RELATIVE_PATH_SEMANTICS,
            "created_at": _utc_now(),
            "status": "complete",
            "controls": {
                "path": _CONTROL_DIRECTORY_NAME,
                "artifact_set": controls_before,
                "self_contained": True,
            },
            "base_union": {
                "path": control_base_relative,
                "merge_receipt": {
                    "path": (
                        f"{control_base_relative}/{_MERGE_RECEIPT_NAME}"
                    ),
                    "sha256": merge_receipt_sha256,
                    "schema": merge_receipt["schema"],
                },
                "fetch_receipt": {
                    "path": (
                        f"{control_base_relative}/{_FETCH_RECEIPT_NAME}"
                    ),
                    "sha256": fetch_receipt_sha256,
                    "schema": fetch_receipt.get("schema"),
                },
                "inventory_receipt": {
                    "path": (
                        f"{control_base_relative}/"
                        f"{_INVENTORY_RECEIPT_NAME}"
                    ),
                    "sha256": base_inventory_receipt_sha256,
                },
                "content_store_receipt": {
                    "path": (
                        f"{control_base_relative}/{_STORE_RECEIPT_NAME}"
                    ),
                    "sha256": _sha256_file(base_store_receipt),
                },
                "source_artifact_set": source_before,
                "original_hashes_unchanged": True,
            },
            "tokenizer": {
                "path": (
                    f"{_CONTROL_DIRECTORY_NAME}/"
                    f"{_CONTROL_TOKENIZER_NAME}"
                ),
                **tokenizer_before,
            },
            "continuation_inventory": {
                "source_path": (
                    f"{control_inventory_relative}/{_INVENTORY_NAME}"
                ),
                "source_receipt_path": (
                    f"{control_inventory_relative}/"
                    f"{_INVENTORY_RECEIPT_NAME}"
                ),
                "source_receipt_sha256": selected_receipt_sha256,
                "database_sha256": selected_receipt[
                    "database_artifact"
                ]["sha256"],
                "db_logical_sha256": selected_receipt[
                    "db_logical_sha256"
                ],
                "expected_attempt_set_sha256": selected_receipt[
                    "expected_attempt_set_sha256"
                ],
                "base_compatibility": inventory_compatibility,
            },
            "destination": {
                "path": ".",
                "inventory": _INVENTORY_NAME,
                "inventory_receipt": _INVENTORY_RECEIPT_NAME,
                "fetch_state": _STATE_NAME,
                "content_store": _STORE_NAME,
                "content_store_receipt": _STORE_RECEIPT_NAME,
            },
            "base_inclusion": {
                "fetch_state": {
                    "source": base_state_projection,
                    "destination_after_path_rewrite": (
                        destination_projection
                    ),
                    "logical_rows_equal": True,
                    "mutable_path_rewrite_only": True,
                    "discovery_sweep_reset_for_new_verified_inventory": True,
                },
                "content_store": {
                    **base_store_binding,
                    "runtime_creator_script_sha256": (
                        runtime_store_script_sha256
                    ),
                    "continuation_writable": True,
                },
                "cas_logical_sets_equal": True,
                "no_mutable_hardlinks": True,
                "destination_inventory_store_state_reverified": True,
            },
        }
        atomic_write_json(partial / _SEED_RECEIPT_NAME, seed_receipt)
        _fsync_tree(partial)
        if _snapshot_tree(controls) != controls_before:
            raise CloneError(
                "staged continuation controls changed after seed receipt "
                "construction"
            )
        if _snapshot_tree(base) != source_before:
            raise CloneError(
                "base union changed after seed receipt construction"
            )
        if {
            "byte_size": selected_inventory.stat().st_size,
            "sha256": _sha256_file(selected_inventory),
        } != selected_inventory_before:
            raise CloneError(
                "selected inventory changed after seed receipt construction"
            )
        if {
            "byte_size": selected_inventory_receipt.stat().st_size,
            "sha256": _sha256_file(selected_inventory_receipt),
        } != selected_receipt_before:
            raise CloneError(
                "selected inventory receipt changed after seed receipt "
                "construction"
            )
        if {
            "byte_size": tokenizer_file.stat().st_size,
            "sha256": _sha256_file(tokenizer_file),
        } != tokenizer_before:
            raise CloneError(
                "tokenizer changed after seed receipt construction"
            )
        if _content_store_script_sha256() != runtime_store_script_sha256:
            raise CloneError(
                "runtime ci_content_store.py changed after seed receipt "
                "construction"
            )
        try:
            _publish_directory_no_replace(partial, target)
        except ExportError as exc:
            raise CloneError(
                f"continuation publication failed: {exc}"
            ) from exc
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        published = True
        return seed_receipt
    finally:
        if not published and partial.exists():
            shutil.rmtree(partial)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-union", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = clone_union_for_resume(
            base_union=args.base_union,
            destination=args.destination,
            tokenizer_path=args.tokenizer,
            inventory_path=args.inventory,
            inventory_receipt_path=args.inventory_receipt,
        )
    except (CloneError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"[clone-ci-stream-union] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
