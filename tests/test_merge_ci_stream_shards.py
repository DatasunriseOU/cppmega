from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, cast
import zlib

import pytest

from scripts import ci_stream_inventory as ci_inventory
from scripts.ci_content_store import CIContentStore
from scripts.ci_content_store import _sqlite_schema_sha256
from scripts.ci_fetch_state_migration import (
    CURRENT_V4_SQLITE_SCHEMA_SHA256,
    LEGACY_FETCH_STATE_SCHEMA,
    LEGACY_V3_SQLITE_SCHEMA_SHA256,
)
from scripts.ci_stream_fetch import (
    CIStreamFetcher,
    COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
    COMPLETION_MODE_THRESHOLD,
    EXHAUSTIVE_DISCOVERY_SCHEMA,
    RECEIPT_SCHEMA as FETCH_RECEIPT_SCHEMA,
    ExhaustiveInventoryBinding,
    ExactTokenizer,
    FetchState,
    _STATE_SCHEMA,
    _script_sha256 as _current_fetcher_script_sha256,
    exhaustive_discovery_sidecar_path,
)
from scripts.ci_stream_receipts import finalize_fetch_receipts
from scripts.ci_log_sidecars import _repo_source_binding
from scripts.ci_source_binding_projection import (
    LEGACY_PARSER_SHA256,
    SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
    SOURCE_BINDING_PROJECTION_SCHEMA,
    target_parser_script_sha256,
)
from scripts.canonical_parquet_ledger import iter_canonical_parquet_ledger
from scripts.clone_ci_stream_union_for_resume import (
    CloneError,
    _snapshot_tree,
    clone_union_for_resume,
)
from scripts.ci_stream_receipts import (
    ReceiptFinalizationError,
    verify_continuation_seed_inclusion,
)
from scripts.ci_stream_inventory import (
    METADATA_ENCODING,
    SCHEMA_VERSION as INVENTORY_SCHEMA,
    _SCHEMA_SQL as INVENTORY_SQL,
    _hash_lines,
    InventoryDB,
)
from scripts.ci_source_sidecars import (
    SourceSidecarStore,
    extract_binding_inventory,
    materialize_inventory,
    verify_binding_inventory,
)
from scripts.data.build_macro_routes_megatron_bundle import (
    _load_ci_manifest_allowlist,
)
from scripts.export_ci_content_store_case5 import (
    FrozenFetchState,
    FrozenStore,
    _fetch_state_logical_digest,
)
from scripts.export_ci_content_store_case5 import ExportError, export_store
from scripts.merge_ci_stream_shards import (
    MANIFEST_SCHEMA,
    JOURNAL_SCHEMA,
    LEGACY_JOURNAL_SCHEMA,
    LEGACY_JOURNAL_V2_SQLITE_SCHEMA_SHA256,
    MIGRATION_MODE,
    MIGRATION_SCHEMA,
    PRODUCTION_MANIFEST_SCHEMA,
    MergeError,
    MergePaused,
    TIME_SHARD_INVENTORY_SCHEMA,
    _INVENTORY_RUN_COLUMNS,
    _JOURNAL_SQL,
    _TIME_SHARD_SQL,
    _acquire_merge_lock,
    _append_source_drift_notes,
    _attempt_evidence_rank,
    _canonical_json_bytes,
    _merge_attempt_row,
    _release_merge_lock,
    _require_complete_bundle_tree,
    _verify_state_inventory_join,
    build_canonical_manifest,
    frozen_store_artifact_set_sha256,
    merge_shards,
)
from tests.test_export_ci_content_store_case5 import (
    TOKENIZER_JSON,
    _build_store,
    _provenance,
    _run_metadata,
)
from tests.test_ci_source_sidecars import _git_fixture
from tests.test_ci_stream_fetch import FakeGitHub, _zip_bytes
from tests.test_ci_stream_inventory import (
    DatasetAPI,
    END as INVENTORY_END,
    START as INVENTORY_START,
    _run as _inventory_run,
)

_UNKNOWN_PARSER_SHA256 = "1" * 64


@dataclass(frozen=True)
class BuiltShard:
    root: Path
    inventory: Path
    inventory_receipt: Path | None
    store: Path
    store_receipt: Path
    state: Path
    fetch_receipt: Path
    original_inventory: str
    original_store: str
    original_state: str


@pytest.fixture(scope="module")
def exact_tokenizer() -> ExactTokenizer:
    return ExactTokenizer(TOKENIZER_JSON)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _empty_inventory_template(
    tmp_path: Path,
    *,
    repo_key: str = "owner/repo",
) -> tuple[Path, dict[str, Any]]:
    owner, name = repo_key.split("/", 1)
    path = tmp_path / "inventory-template.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.executescript(INVENTORY_SQL)
        run_rows: list[tuple[int, bytes, str, dict[str, Any]]] = []
        for run_id in (100, 200, 300):
            _text, provenance = _record(
                f"inventory seed {run_id}",
                run_id=run_id,
                archive_member=f"{run_id}.txt",
            )
            run_metadata = _run_metadata(provenance)
            raw = _canonical_json_bytes(run_metadata)
            run_rows.append(
                (
                    run_id,
                    raw,
                    hashlib.sha256(raw).hexdigest(),
                    run_metadata,
                )
            )
        run_keys_sha256 = _hash_lines(
            f"{repo_key}\t{run_id}\t1\t{metadata_sha256}"
            for run_id, _raw, metadata_sha256, _metadata in run_rows
        )
        metadata = {
            "schema": INVENTORY_SCHEMA,
            "repo_list_path": "/frozen/repositories.json",
            "repo_list_sha256": "1" * 64,
            "repo_scope_sha256": _hash_lines((repo_key,)),
            "repo_count": "1",
            "original_repo_count": "1",
            "unresolved_count": "0",
            "start_epoch": "1780272000",
            "end_epoch": "1785542400",
            "start_utc": "2026-06-01T00:00:00Z",
            "end_utc": "2026-08-01T00:00:00Z",
            "script_sha256": "2" * 64,
            "metadata_encoding": METADATA_ENCODING,
            "smoke": "0",
            "max_repos": "",
            "created_at": "2026-07-26T09:00:00Z",
        }
        connection.executemany(
            "INSERT INTO inventory_meta(key,value) VALUES (?,?)",
            sorted(metadata.items()),
        )
        connection.execute(
            """
            INSERT INTO repos(repo_key,owner,name,canonical,ordinal)
            VALUES (?,?,?,?,0)
            """,
            (repo_key, owner, name, repo_key),
        )
        window_id = int(
            connection.execute(
                """
                INSERT INTO search_windows(
                  repo_key,start_epoch,end_epoch,parent_id,depth,status,
                  expected_total,expected_pages,pages_done,raw_items,
                  distinct_items,duplicate_items,run_keys_sha256,
                  created_at,updated_at
                ) VALUES (
                  ?,1780272000,1785542400,NULL,0,'done',
                  3,1,1,3,3,0,?,
                  '2026-07-26T09:00:00Z','2026-07-26T09:00:00Z'
                )
                """,
                (repo_key, run_keys_sha256),
            ).lastrowid
        )
        for run_id, raw, metadata_sha256, run_metadata in run_rows:
            connection.execute(
                """
                INSERT INTO runs(
                  repo_key,run_id,run_attempt,created_at,updated_at,
                  run_started_at,status,conclusion,workflow_id,workflow_name,
                  event,head_branch,head_sha,run_number,html_url,api_url,
                  metadata_blob,metadata_sha256,first_seen_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    repo_key,
                    run_id,
                    1,
                    run_metadata["created_at"],
                    run_metadata["updated_at"],
                    run_metadata["run_started_at"],
                    run_metadata["status"],
                    run_metadata["conclusion"],
                    run_metadata["workflow_id"],
                    run_metadata["name"],
                    run_metadata["event"],
                    run_metadata["head_branch"],
                    run_metadata["head_sha"],
                    run_metadata["run_number"],
                    None,
                    None,
                    sqlite3.Binary(zlib.compress(raw, 6)),
                    metadata_sha256,
                    "2026-07-26T09:00:00Z",
                ),
            )
            connection.execute(
                """
                INSERT INTO window_runs(
                  window_id,repo_key,run_id,run_attempt,metadata_sha256
                ) VALUES (?,?,?,?,?)
                """,
                (window_id, repo_key, run_id, 1, metadata_sha256),
            )
        connection.execute(
            """
            INSERT INTO window_pages(
              window_id,page_no,total_count,item_count,distinct_item_count,
              duplicate_item_count,payload_sha256,run_keys_sha256,fetched_at
            ) VALUES (?,1,3,3,3,0,?,?,?)
            """,
            (
                window_id,
                "3" * 64,
                run_keys_sha256,
                "2026-07-26T09:00:00Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO request_ledger(
              requested_at,repo_key,window_id,endpoint,page_no,per_page,
              attempt,http_status,outcome,latency_ms
            ) VALUES (?,?,?,?,1,100,1,200,'success',1)
            """,
            (
                "2026-07-26T09:00:00Z",
                repo_key,
                window_id,
                f"/repos/{repo_key}/actions/runs",
            ),
        )
        connection.commit()
    finally:
        connection.close()
    receipt = InventoryDB(path).completion_receipt()
    for suffix in ("-wal", "-shm", "-journal"):
        assert not Path(f"{path}{suffix}").exists()
    return path, receipt


def _inventory_with_new_attempt(
    tmp_path: Path,
    source: Path,
    *,
    run_id: int,
    run_attempt: int,
) -> tuple[Path, Path, dict[str, Any]]:
    path = tmp_path / "rerun-inventory.sqlite3"
    shutil.copyfile(source, path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        row = connection.execute(
            """
            SELECT metadata_blob FROM runs
            WHERE repo_key='owner/repo' AND run_id=?
            """,
            (run_id,),
        ).fetchone()
        assert row is not None
        metadata = json.loads(zlib.decompress(bytes(row["metadata_blob"])))
        metadata["run_attempt"] = run_attempt
        raw = _canonical_json_bytes(metadata)
        metadata_sha256 = hashlib.sha256(raw).hexdigest()
        connection.execute(
            """
            UPDATE runs
            SET run_attempt=?,metadata_blob=?,metadata_sha256=?
            WHERE repo_key='owner/repo' AND run_id=?
            """,
            (
                run_attempt,
                sqlite3.Binary(zlib.compress(raw, 6)),
                metadata_sha256,
                run_id,
            ),
        )
        connection.execute(
            """
            UPDATE window_runs
            SET run_attempt=?,metadata_sha256=?
            WHERE repo_key='owner/repo' AND run_id=?
            """,
            (run_attempt, metadata_sha256, run_id),
        )
        run_keys_sha256 = _hash_lines(
            f"{item['repo_key']}\t{item['run_id']}\t"
            f"{item['run_attempt']}\t{item['metadata_sha256']}"
            for item in connection.execute(
                """
                SELECT repo_key,run_id,run_attempt,metadata_sha256
                FROM window_runs
                ORDER BY repo_key,run_id,run_attempt
                """
            )
        )
        connection.execute(
            "UPDATE search_windows SET run_keys_sha256=?",
            (run_keys_sha256,),
        )
        connection.execute(
            "UPDATE window_pages SET run_keys_sha256=?",
            (run_keys_sha256,),
        )
        connection.commit()
    finally:
        connection.close()
    receipt_path = tmp_path / "rerun-inventory-receipt.json"
    receipt = InventoryDB(
        path,
        initialize_schema=False,
    ).completion_receipt()
    _write_json(receipt_path, receipt)
    return path, receipt_path, receipt


def _set_original_bindings(
    state_path: Path,
    *,
    inventory_path: str,
    store_path: str,
) -> None:
    connection = sqlite3.connect(state_path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            "UPDATE settings SET value=? WHERE key='inventory_path'",
            (inventory_path,),
        )
        connection.execute(
            "UPDATE settings SET value=? WHERE key='content_store_path'",
            (store_path,),
        )
        connection.commit()
    finally:
        connection.close()


def _materialize_genuine_done_evidence(state_path: Path) -> None:
    connection = sqlite3.connect(state_path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        for row in connection.execute(
            """
            SELECT repo,run_id,attempt,run_metadata_zlib
            FROM attempts WHERE status='done'
            ORDER BY repo,run_id,attempt
            """
        ).fetchall():
            metadata = json.loads(zlib.decompress(row["run_metadata_zlib"]))
            repository = metadata.get("repository")
            canonical_repo = (
                str(row["repo"])
                if not isinstance(repository, Mapping)
                else str(repository.get("full_name") or row["repo"])
            )
            archive = (
                f"verified archive {row['repo']} "
                f"{row['run_id']}:{row['attempt']}"
            ).encode()
            connection.execute(
                """
                UPDATE attempts SET
                  archive_source='fixture-verified-archive',
                  archive_sha256=?,archive_size=?
                WHERE repo=? AND run_id=? AND attempt=?
                """,
                (
                    hashlib.sha256(archive).hexdigest(),
                    len(archive),
                    row["repo"],
                    row["run_id"],
                    row["attempt"],
                ),
            )
            for endpoint, page_no, http_status in (
                (
                    f"/repos/{canonical_repo}/actions/runs/{row['run_id']}/"
                    f"attempts/{row['attempt']}/logs",
                    None,
                    302,
                ),
                (
                    f"/repos/{canonical_repo}/actions/runs/{row['run_id']}/"
                    f"attempts/{row['attempt']}/jobs",
                    1,
                    200,
                ),
            ):
                connection.execute(
                    """
                    INSERT INTO request_ledger(
                      requested_at,repo,run_id,attempt,endpoint,page_no,
                      request_attempt,http_status,outcome,latency_ms
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        "2026-07-26T10:00:00Z",
                        row["repo"],
                        row["run_id"],
                        row["attempt"],
                        endpoint,
                        page_no,
                        1,
                        http_status,
                        "success",
                        10,
                    ),
                )
        connection.commit()
    finally:
        connection.close()


def _append_request_and_binding(state_path: Path) -> None:
    connection = sqlite3.connect(state_path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        attempt = connection.execute(
            "SELECT repo,run_id,attempt FROM attempts ORDER BY repo,run_id,attempt LIMIT 1"
        ).fetchone()
        assert attempt is not None
        connection.execute(
            """
            INSERT INTO request_ledger(
              requested_at,repo,run_id,attempt,endpoint,page_no,
              request_attempt,http_status,outcome,latency_ms,
              error_class,error_message
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "2026-07-26T10:00:00Z",
                str(attempt[0]),
                int(attempt[1]),
                int(attempt[2]),
                "/actions/runs/logs",
                None,
                1,
                200,
                "success",
                10,
                None,
                None,
            ),
        )
        connection.execute(
            """
            INSERT INTO binding_upgrades(
              binding_key,from_sha256,to_sha256,reason,upgraded_at
            ) VALUES ('fetcher_script_sha256',?,?,?,?)
            """,
            (
                "6" * 64,
                "4" * 64,
                "fixture upgrade",
                "2026-07-26T09:00:00Z",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _replace_binding_history(
    state_path: Path,
    transitions: tuple[tuple[str, str], ...],
    *,
    binding_key: str = "fetcher_script_sha256",
    clear: bool = True,
    upgraded_at: str = "2026-07-26T08:00:00Z",
) -> None:
    connection = sqlite3.connect(state_path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        if clear:
            connection.execute("DELETE FROM binding_upgrades")
        for source, destination in transitions:
            connection.execute(
                """
                INSERT INTO binding_upgrades(
                  binding_key,from_sha256,to_sha256,reason,upgraded_at
                ) VALUES (?,?,?,?,?)
                """,
                (
                    binding_key,
                    source,
                    destination,
                    f"fixture {source[:8]} to {destination[:8]}",
                    upgraded_at,
                ),
            )
        connection.commit()
    finally:
        connection.close()


def _fetch_receipt(
    shard: BuiltShard,
    tokenizer: ExactTokenizer,
) -> dict[str, Any]:
    store_receipt = json.loads(shard.store_receipt.read_text(encoding="utf-8"))
    binding_store = SimpleNamespace(
        root=Path(shard.original_store).resolve(),
        receipt=store_receipt,
    )
    with FrozenStore(shard.store, shard.store_receipt) as store:
        with FrozenFetchState(
            shard.state,
            tokenizer=tokenizer,
            store=cast(Any, binding_store),
        ) as state:
            binding = state.receipt_binding()
            binding["artifact"]["path"] = shard.original_state
            receipt = {
                "schema": FETCH_RECEIPT_SCHEMA,
                "completed_at": "2026-07-26T11:00:00Z",
                "target_exact_unique_payload_tokens": 0,
                "fetch_state": state.summary,
                "frozen_fetch_state": binding,
                "content_store_receipt": store_receipt,
                "inventory_path": shard.original_inventory,
                "tokenizer_contract": tokenizer.contract,
                "tokenizer_fingerprint": tokenizer.fingerprint,
            }
            state.require_unchanged()
        store.require_unchanged()
    return receipt


def _build_shard(
    root: Path,
    tokenizer: ExactTokenizer,
    inventory_template: Path,
    inventory_template_receipt: dict[str, Any],
    records: list[tuple[str, dict[str, Any]]],
    *,
    relocated_prefix: str | None = None,
    add_ledgers: bool = False,
) -> BuiltShard:
    root.mkdir()
    inventory = root / "inventory.sqlite3"
    shutil.copyfile(inventory_template, inventory)
    store, store_receipt, state = _build_store(
        root,
        tokenizer,
        records,
        target_unique_tokens=0,
    )
    _materialize_genuine_done_evidence(state)
    if relocated_prefix is None:
        original_inventory = str(inventory)
        original_store = str(store.resolve())
        original_state = str(state)
    else:
        original_inventory = f"{relocated_prefix}/inventory.sqlite3"
        original_store = f"{relocated_prefix}/content-store"
        original_state = f"{relocated_prefix}/fetch-state.sqlite3"
    _set_original_bindings(
        state,
        inventory_path=original_inventory,
        store_path=original_store,
    )
    if add_ledgers:
        _append_request_and_binding(state)
    inventory_receipt = root / "inventory-receipt.json"
    inventory_value = dict(inventory_template_receipt)
    inventory_value["database"] = original_inventory
    inventory_value["database_artifact"] = dict(
        inventory_template_receipt["database_artifact"]
    )
    inventory_value["database_artifact"]["path"] = original_inventory
    _write_json(inventory_receipt, inventory_value)
    fetch_receipt = root / "fetch-receipt.json"
    built = BuiltShard(
        root=root,
        inventory=inventory,
        inventory_receipt=inventory_receipt,
        store=store,
        store_receipt=store_receipt,
        state=state,
        fetch_receipt=fetch_receipt,
        original_inventory=original_inventory,
        original_store=original_store,
        original_state=original_state,
    )
    _write_json(fetch_receipt, _fetch_receipt(built, tokenizer))
    return built


def _promote_to_exhaustive_receipt(
    shard: BuiltShard,
) -> None:
    assert shard.inventory_receipt is not None
    inventory_receipt = json.loads(
        shard.inventory_receipt.read_text(encoding="utf-8")
    )
    inventory_receipt["database"] = shard.original_inventory
    inventory_receipt["database_artifact"]["path"] = (
        shard.original_inventory
    )
    _write_json(shard.inventory_receipt, inventory_receipt)
    inventory_receipt_sha256 = _sha256(shard.inventory_receipt)
    with sqlite3.connect(shard.inventory) as connection:
        last_run = connection.execute(
            """
            SELECT created_at,repo_key,run_id,run_attempt
            FROM runs
            ORDER BY created_at,repo_key,run_id,run_attempt
            """
        ).fetchall()
    _write_json(
        exhaustive_discovery_sidecar_path(shard.state),
        {
            "schema": EXHAUSTIVE_DISCOVERY_SCHEMA,
            "completion_mode": COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
            "inventory_receipt_sha256": inventory_receipt_sha256,
            "inventory_database_sha256": (
                inventory_receipt["database_artifact"]["sha256"]
            ),
            "inventory_db_logical_sha256": inventory_receipt[
                "db_logical_sha256"
            ],
            "expected_run_count": inventory_receipt["run_count"],
            "expected_attempt_count": inventory_receipt[
                "expected_attempt_count"
            ],
            "expected_attempt_set_sha256": inventory_receipt[
                "expected_attempt_set_sha256"
            ],
            "cursor": (
                None
                if not last_run
                else [
                    str(last_run[-1][0]),
                    str(last_run[-1][1]),
                    int(last_run[-1][2]),
                    int(last_run[-1][3]),
                ]
            ),
            "discovery_eof": True,
            "batches": 1,
            "rows_seen": inventory_receipt["run_count"],
            "started_at": "2026-07-26T11:00:00Z",
            "updated_at": "2026-07-26T11:00:00Z",
        },
    )
    finalize_fetch_receipts(
        state_path=shard.state,
        content_store_path=shard.store,
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=1,
        fetch_receipt_path=shard.fetch_receipt,
        store_receipt_path=shard.store_receipt,
        original_state_path=shard.original_state,
        original_content_store_path=shard.original_store,
        original_inventory_path=shard.original_inventory,
        completion_mode=COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        inventory_receipt_path=shard.inventory_receipt,
    )


def _record(
    text: str,
    *,
    run_id: int,
    archive_member: str,
    ordinal: int = 0,
) -> tuple[str, dict[str, Any]]:
    provenance = _provenance(
        text,
        ordinal=ordinal,
        archive_member=archive_member,
    )
    provenance["run_id"] = run_id
    provenance["run_attempt"] = 1
    return text, provenance


def _record_for_repo(
    text: str,
    *,
    repo_key: str,
    run_id: int,
    archive_member: str,
) -> tuple[str, dict[str, Any]]:
    record = _record(
        text,
        run_id=run_id,
        archive_member=archive_member,
    )
    provenance = record[1]
    for key in (
        "repository",
        "repository_requested",
        "source_repository",
        "repository_scope_key",
    ):
        provenance[key] = repo_key
    return record


def _record_with_source_binding(
    *,
    run_id: int,
    archive_member: str,
    binding_mode: str,
) -> tuple[str, dict[str, Any]]:
    text = f"clang++ -c src/{binding_mode}.cpp"
    record = _record(
        text,
        run_id=run_id,
        archive_member=archive_member,
    )
    provenance = record[1]
    source_input = "src/main.cpp"
    cwd = "/home/runner/work/repo/repo/build"
    binding_provenance = {
        "repository": provenance["repository"],
        "source_repository": provenance["source_repository"],
        "run": provenance["workflow"],
    }
    current_binding = _repo_source_binding(
        source_input,
        binding_provenance,
        cwd=cwd,
    )
    assert current_binding is not None
    if binding_mode == "current":
        binding = current_binding
    elif binding_mode == "legacy":
        binding = {
            **current_binding,
            "source_path": source_input,
            "confidence": {
                "score": 0.95,
                "level": "high",
                "source": "relative_source_path_v1",
            },
        }
    elif binding_mode == "third":
        binding = {
            **current_binding,
            "repository": "third/semantics",
        }
    else:
        raise AssertionError(f"unsupported binding mode: {binding_mode}")
    provenance["chunk"]["training_sidecars"]["build_actions"] = [
        {
            "normalization_schema": "cppmega_ci_build_action_normalization_v1",
            "tool": "clang++",
            "kind": "compile",
            "cwd": cwd,
            "source_inputs": [source_input],
            "source_input_count": 1,
            "outputs": [],
            "output_count": 0,
            "flags": ["-c"],
            "repository_source_bindings": [binding],
            "repository_source_binding_count": 1,
            "command_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "action_shape_sha256": hashlib.sha256(
                f"shape:{text}".encode()
            ).hexdigest(),
            "start_char": 0,
            "end_char": len(text),
            "line_index": 0,
            "section_ordinal": 0,
            "step_ordinal": None,
            "confidence": {
                "score": 0.98,
                "level": "high",
                "source": "fixture",
            },
        }
    ]
    return record


def _set_parser_binding_history(
    shard: BuiltShard,
    tokenizer: ExactTokenizer,
    transitions: tuple[tuple[str, str], ...],
) -> None:
    _replace_binding_history(
        shard.state,
        transitions,
        binding_key="parser_script_sha256",
    )
    _write_json(shard.fetch_receipt, _fetch_receipt(shard, tokenizer))


def _make_time_subset(
    shard: BuiltShard,
    anchor: BuiltShard,
    *,
    created_at_gte: str = "2026-06-01T00:00:00Z",
    created_at_lt: str = "2026-08-01T00:00:00Z",
) -> BuiltShard:
    subset_path = shard.root / "time-shard-inventory.sqlite3"
    state = sqlite3.connect(shard.state)
    try:
        seed_keys = [
            (str(row[0]), int(row[1]), int(row[2]))
            for row in state.execute(
                """
                SELECT DISTINCT repo,run_id,inventory_seed_attempt
                FROM attempts ORDER BY repo,run_id,inventory_seed_attempt
                """
            )
        ]
    finally:
        state.close()
    anchor_connection = sqlite3.connect(anchor.inventory)
    anchor_connection.row_factory = sqlite3.Row
    subset = sqlite3.connect(subset_path)
    try:
        subset.execute("PRAGMA journal_mode=DELETE")
        subset.executescript(_TIME_SHARD_SQL)
        columns = ",".join(_INVENTORY_RUN_COLUMNS)
        for key in seed_keys:
            row = anchor_connection.execute(
                f"""
                SELECT {columns} FROM runs
                WHERE repo_key=? AND run_id=? AND run_attempt=?
                """,
                key,
            ).fetchone()
            assert row is not None
            subset.execute(
                f"""
                INSERT INTO runs({columns})
                VALUES ({",".join("?" for _ in _INVENTORY_RUN_COLUMNS)})
                """,
                tuple(row[column] for column in _INVENTORY_RUN_COLUMNS),
            )
        metadata = {
            "schema": TIME_SHARD_INVENTORY_SCHEMA,
            "source_inventory_path": anchor.original_inventory,
            "created_at": "2026-07-27T12:00:00Z",
            "created_at_gte": created_at_gte,
            "created_at_lt": created_at_lt,
            "run_count": str(len(seed_keys)),
        }
        subset.executemany(
            "INSERT INTO shard_meta(key,value) VALUES (?,?)",
            sorted(metadata.items()),
        )
        subset.commit()
    finally:
        subset.close()
        anchor_connection.close()
    return replace(
        shard,
        inventory=subset_path,
        inventory_receipt=None,
    )


def _inventory_manifest_descriptor(shard: BuiltShard) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": str(shard.inventory),
        "sha256": _sha256(shard.inventory),
    }
    if shard.inventory_receipt is not None:
        descriptor["receipt"] = {
            "path": str(shard.inventory_receipt),
            "sha256": _sha256(shard.inventory_receipt),
        }
    return descriptor


def _manifest(
    tmp_path: Path,
    tokenizer: ExactTokenizer,
    shards: list[BuiltShard],
    *,
    target: int = 0,
    occurrences_per_batch: int = 2,
) -> tuple[Path, Path]:
    del tokenizer
    destination = tmp_path / "union"
    value = {
        "schema": MANIFEST_SCHEMA,
        "destination": {
            "bundle_path": str(destination),
            "target_exact_unique_payload_tokens": target,
        },
        "tokenizer": {
            "path": str(TOKENIZER_JSON),
            "sha256": _sha256(TOKENIZER_JSON),
        },
        "limits": {
            "max_shards": 8,
            "occurrences_per_batch": occurrences_per_batch,
            "state_rows_per_batch": 2,
            "uncompressed_bytes_per_batch": 2 * 1024 * 1024,
            "max_content_bytes": 512 * 1024,
            "max_provenance_bytes": 1024 * 1024,
            "max_state_blob_bytes": 1024 * 1024,
        },
        "shards": [
            {
                "id": f"s{index:02d}",
                "original_paths": {
                    "inventory": shard.original_inventory,
                    "content_store": shard.original_store,
                    "fetch_state": shard.original_state,
                },
                "staged": {
                    "inventory": _inventory_manifest_descriptor(shard),
                    "content_store": {
                        "path": str(shard.store),
                        "artifact_set_sha256": frozen_store_artifact_set_sha256(
                            shard.store,
                            shard.store_receipt,
                        ),
                        "receipt": {
                            "path": str(shard.store_receipt),
                            "sha256": _sha256(shard.store_receipt),
                        },
                    },
                    "fetch_state": {
                        "path": str(shard.state),
                        "sha256": _sha256(shard.state),
                        "receipt": {
                            "path": str(shard.fetch_receipt),
                            "sha256": _sha256(shard.fetch_receipt),
                        },
                    },
                },
            }
            for index, shard in enumerate(shards)
        ],
    }
    path = tmp_path / "union-manifest.json"
    path.write_bytes(_canonical_json_bytes(value) + b"\n")
    return path, destination


def _downgrade_fetch_state_to_legacy_v3(
    state_path: Path,
    fetch_receipt_path: Path,
) -> None:
    legacy_schema = _STATE_SCHEMA.replace("    archive_zlib BLOB,\n", "")
    temporary = state_path.with_name(f".{state_path.name}.legacy-v3")
    source = sqlite3.connect(state_path)
    source.row_factory = sqlite3.Row
    destination = sqlite3.connect(temporary)
    destination.row_factory = sqlite3.Row
    try:
        destination.execute("PRAGMA journal_mode=DELETE")
        destination.executescript(legacy_schema)
        for table in (
            "settings",
            "attempts",
            "members",
            "request_ledger",
            "binding_upgrades",
        ):
            columns = tuple(
                str(row["name"])
                for row in destination.execute(
                    f"PRAGMA table_info({table})"
                )
            )
            rows = source.execute(
                f"SELECT {','.join(columns)} FROM {table}"
            ).fetchall()
            destination.executemany(
                f"""
                INSERT INTO {table}({",".join(columns)})
                VALUES ({",".join("?" for _column in columns)})
                """,
                (tuple(row[column] for column in columns) for row in rows),
            )
        destination.execute(
            "UPDATE settings SET value=? WHERE key='schema'",
            (LEGACY_FETCH_STATE_SCHEMA,),
        )
        destination.commit()
        assert (
            _sqlite_schema_sha256(destination)
            == LEGACY_V3_SQLITE_SCHEMA_SHA256
        )
    finally:
        destination.close()
        source.close()
    temporary.replace(state_path)

    receipt = json.loads(fetch_receipt_path.read_text(encoding="utf-8"))
    connection = sqlite3.connect(
        f"{state_path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        settings = {
            str(row["key"]): str(row["value"])
            for row in connection.execute(
                "SELECT key,value FROM settings ORDER BY key"
            )
        }
        logical_sha256 = _fetch_state_logical_digest(connection)
    finally:
        connection.close()
    stat_result = state_path.stat()
    binding = receipt["frozen_fetch_state"]
    receipt["fetch_state"].pop("binding_upgrades", None)
    binding["summary"].pop("binding_upgrades", None)
    binding["schema"] = LEGACY_FETCH_STATE_SCHEMA
    binding["artifact"].update(
        {
            "byte_size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "inode": stat_result.st_ino,
            "sha256": _sha256(state_path),
        }
    )
    binding["sqlite_schema_sha256"] = LEGACY_V3_SQLITE_SCHEMA_SHA256
    binding["sqlite_logical_sha256"] = logical_sha256
    binding["settings"] = settings
    _write_json(fetch_receipt_path, receipt)


def _downgrade_merge_journal_to_legacy_v2(journal_path: Path) -> None:
    legacy_sql = _JOURNAL_SQL.replace(
        """        'done_replaced_zero_evidence_pending',
        'lower_evidence_shadowed','higher_evidence_replaced',
        'exact_overlap_promoted_higher_precedence'
""",
        """        'done_replaced_zero_evidence_pending'
""",
    )
    winners_start = legacy_sql.index(
        "CREATE TABLE IF NOT EXISTS attempt_winners"
    )
    member_start = legacy_sql.index(
        "CREATE TABLE IF NOT EXISTS member_map"
    )
    legacy_sql = legacy_sql[:winners_start] + legacy_sql[member_start:]

    temporary = journal_path.with_name(f".{journal_path.name}.legacy-v2")
    source = sqlite3.connect(journal_path)
    source.row_factory = sqlite3.Row
    destination = sqlite3.connect(temporary)
    destination.row_factory = sqlite3.Row
    try:
        destination.execute("PRAGMA journal_mode=DELETE")
        destination.executescript(legacy_sql)
        for table in (
            "settings",
            "store_progress",
            "state_progress",
            "attempt_map",
            "member_map",
            "request_id_map",
            "binding_id_map",
        ):
            columns = tuple(
                str(row["name"])
                for row in destination.execute(
                    f"PRAGMA table_info({table})"
                )
            )
            rows = source.execute(
                f"SELECT {','.join(columns)} FROM {table}"
            ).fetchall()
            destination.executemany(
                f"""
                INSERT INTO {table}({",".join(columns)})
                VALUES ({",".join("?" for _column in columns)})
                """,
                (tuple(row[column] for column in columns) for row in rows),
            )
        destination.execute(
            "UPDATE settings SET value=? WHERE key='schema'",
            (LEGACY_JOURNAL_SCHEMA,),
        )
        destination.commit()
        assert (
            _sqlite_schema_sha256(destination)
            == LEGACY_JOURNAL_V2_SQLITE_SCHEMA_SHA256
        )
    finally:
        destination.close()
        source.close()
    temporary.replace(journal_path)


def _tree_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _remove_only_occurrence_witness(
    *,
    store_root: Path,
    store_receipt_path: Path,
    fetch_receipt_path: Path,
    content_sha256: str,
) -> None:
    database = store_root / "index.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        deleted = connection.execute(
            "DELETE FROM occurrences WHERE content_sha256=?",
            (content_sha256,),
        )
        assert deleted.rowcount == 1
        aggregate = connection.execute(
            """
            SELECT
              (SELECT COALESCE(SUM(contents.raw_size), 0)
               FROM occurrences
               JOIN contents
                 ON contents.sha256=occurrences.content_sha256)
                AS raw_occurrence_bytes,
              (SELECT COALESCE(SUM(raw_size), 0) FROM contents)
                AS unique_bytes,
              (SELECT COUNT(*) FROM contents) AS unique_content_count,
              (SELECT COUNT(*) FROM occurrences) AS occurrence_count,
              (SELECT COUNT(*) FROM contents
               WHERE token_sequence_sha256 IS NOT NULL)
                AS tokenized_unique_content_count,
              (SELECT COUNT(*) FROM token_sequences)
                AS unique_token_sequence_count,
              (SELECT COALESCE(SUM(token_count), 0)
               FROM token_sequences)
                AS exact_unique_payload_tokens
            """
        ).fetchone()
        assert aggregate is not None
        raw_occurrence_bytes = int(aggregate["raw_occurrence_bytes"])
        unique_bytes = int(aggregate["unique_bytes"])
        assert raw_occurrence_bytes >= unique_bytes
        connection.execute(
            """
            UPDATE stats
            SET raw_occurrence_bytes=?,
                unique_bytes=?,
                duplicate_bytes=?,
                unique_content_count=?,
                occurrence_count=?,
                tokenized_unique_content_count=?,
                unique_token_sequence_count=?,
                exact_unique_payload_tokens=?
            WHERE singleton=1
            """,
            (
                raw_occurrence_bytes,
                unique_bytes,
                raw_occurrence_bytes - unique_bytes,
                int(aggregate["unique_content_count"]),
                int(aggregate["occurrence_count"]),
                int(aggregate["tokenized_unique_content_count"]),
                int(aggregate["unique_token_sequence_count"]),
                int(aggregate["exact_unique_payload_tokens"]),
            ),
        )
        connection.commit()

    with CIContentStore(store_root) as store:
        receipt = store.write_completion_receipt(
            store_receipt_path,
            target_unique_tokens=0,
        )
    fetch_receipt = json.loads(
        fetch_receipt_path.read_text(encoding="utf-8")
    )
    fetch_receipt["content_store_receipt"] = receipt
    _write_json(fetch_receipt_path, fetch_receipt)


def _anchor_and_time_subset(
    tmp_path: Path,
    tokenizer: ExactTokenizer,
    *,
    relocated: bool = False,
) -> tuple[BuiltShard, BuiltShard]:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    anchor = _build_shard(
        tmp_path / "anchor-stage",
        tokenizer,
        inventory,
        inventory_receipt,
        [_record("anchor payload", run_id=100, archive_member="anchor.txt")],
        relocated_prefix=(
            "/Volumes/frozen/ci/full-anchor" if relocated else None
        ),
    )
    subset = _build_shard(
        tmp_path / "subset-stage",
        tokenizer,
        inventory,
        inventory_receipt,
        [_record("subset payload", run_id=200, archive_member="subset.txt")],
        relocated_prefix=(
            "/home/davidgor/frozen-ci/time-subset" if relocated else None
        ),
    )
    return anchor, _make_time_subset(subset, anchor)


def _production_shard_for_repo(
    tmp_path: Path,
    tokenizer: ExactTokenizer,
    *,
    label: str,
    repo_key: str,
    user_version: int = 0,
) -> BuiltShard:
    inventory_root = tmp_path / f"{label}-inventory"
    inventory_root.mkdir()
    inventory, inventory_receipt = _empty_inventory_template(
        inventory_root,
        repo_key=repo_key,
    )
    shard = _build_shard(
        tmp_path / f"{label}-shard",
        tokenizer,
        inventory,
        inventory_receipt,
        [
            _record_for_repo(
                f"run {run_id}",
                repo_key=repo_key,
                run_id=run_id,
                archive_member=f"{run_id}.txt",
            )
            for run_id in (100, 200, 300)
        ],
    )
    if user_version:
        with sqlite3.connect(shard.inventory) as connection:
            connection.execute(f"PRAGMA user_version={user_version}")
        assert shard.inventory_receipt is not None
        _write_json(
            shard.inventory_receipt,
            InventoryDB(
                shard.inventory,
                initialize_schema=False,
            ).completion_receipt(),
        )
    _promote_to_exhaustive_receipt(shard)
    return shard


def _update_time_subset_meta(
    subset: BuiltShard,
    key: str,
    value: str,
) -> None:
    connection = sqlite3.connect(subset.inventory)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            "UPDATE shard_meta SET value=? WHERE key=?",
            (value, key),
        )
        connection.commit()
    finally:
        connection.close()


def _insert_zero_evidence_pending(
    destination_state: Path,
    source_done_state: Path,
) -> None:
    source = sqlite3.connect(source_done_state)
    source.row_factory = sqlite3.Row
    try:
        row = dict(source.execute("SELECT * FROM attempts").fetchone())
    finally:
        source.close()
    row.update(
        {
            "status": "pending",
            "tries": 0,
            "archive_source": None,
            "archive_sha256": None,
            "archive_size": None,
            "jobs_sha256": None,
            "jobs_raw_size": None,
            "jobs_zlib": None,
            "member_count": 0,
            "chunk_count": 0,
            "occurrence_tokens": 0,
            "terminal_http_status": None,
            "terminal_body_sha256": None,
            "error_class": None,
            "error_message": None,
            "discovered_at": "2026-07-26T09:59:59Z",
            "updated_at": "2026-07-26T10:00:02Z",
        }
    )
    connection = sqlite3.connect(destination_state)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        columns = tuple(row)
        connection.execute(
            f"""
            INSERT INTO attempts({",".join(columns)})
            VALUES ({",".join("?" for _ in columns)})
            """,
            tuple(row[column] for column in columns),
        )
        connection.commit()
    finally:
        connection.close()


def _insert_attempt_variant(
    destination_state: Path,
    source_done_state: Path,
    *,
    status: str,
) -> None:
    source = sqlite3.connect(source_done_state)
    source.row_factory = sqlite3.Row
    try:
        row = dict(source.execute("SELECT * FROM attempts").fetchone())
    finally:
        source.close()
    row.update(
        {
            "status": status,
            "tries": 1,
            "member_count": 0,
            "chunk_count": 0,
            "occurrence_tokens": 0,
            "error_class": None,
            "error_message": None,
        }
    )
    if status == "empty":
        empty_zip = b"PK\x05\x06" + (b"\x00" * 18)
        row.update(
            {
                "archive_source": "fixture-verified-empty-zip",
                "archive_sha256": hashlib.sha256(empty_zip).hexdigest(),
                "archive_size": len(empty_zip),
                "archive_zlib": sqlite3.Binary(zlib.compress(empty_zip, 6)),
                "terminal_http_status": None,
                "terminal_body_sha256": None,
            }
        )
    elif status in {"terminal_404", "terminal_410"}:
        row.update(
            {
                "archive_source": None,
                "archive_sha256": None,
                "archive_size": None,
                "jobs_sha256": None,
                "jobs_raw_size": None,
                "jobs_zlib": None,
                "terminal_http_status": (
                    404 if status == "terminal_404" else 410
                ),
                "terminal_body_sha256": "f" * 64,
            }
        )
    else:
        raise AssertionError(status)
    connection = sqlite3.connect(destination_state)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        columns = tuple(row)
        connection.execute(
            f"""
            INSERT INTO attempts({",".join(columns)})
            VALUES ({",".join("?" for _ in columns)})
            """,
            tuple(row[column] for column in columns),
        )
        connection.commit()
    finally:
        connection.close()


def _prepare_ready_partial(manifest: Path, destination: Path) -> Path:
    with pytest.raises(MergePaused):
        merge_shards(manifest, max_batches=1)
    partial = destination.with_name(f".{destination.name}.partial")
    blocker = partial / "ready-phase-blocker"
    blocker.write_bytes(b"block publication after finalization")
    with pytest.raises(MergeError, match="unexpected=.*ready-phase-blocker"):
        merge_shards(manifest)
    blocker.unlink()
    return partial


def test_disjoint_union_preserves_request_multiplicity_and_final_state_binding(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard_a = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("alpha build output", run_id=100, archive_member="a.txt")],
        add_ledgers=True,
    )
    shard_b = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("beta test output", run_id=200, archive_member="b.txt")],
        add_ledgers=True,
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_a, shard_b],
    )
    source_request_ids: list[int] = []
    for shard in (shard_a, shard_b):
        with sqlite3.connect(shard.state) as connection:
            source_request_ids.extend(
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM request_ledger ORDER BY id"
                )
            )
    expected_request_count = len(source_request_ids)

    with pytest.raises(MergePaused):
        merge_shards(manifest, max_batches=4)
    assert not destination.exists()
    partial = destination.with_name(f".{destination.name}.partial")
    stale_temporary = partial / ".fetch_receipt.json.tmp-999999"
    stale_temporary.write_text("stale interrupted write", encoding="utf-8")
    receipt = merge_shards(manifest)

    assert receipt["status"] == "complete"
    assert not (destination / stale_temporary.name).exists()
    assert receipt["store_conservation"]["output_union"]["occurrence_count"] == 2
    assert receipt["fetch_state_conservation"]["request_multiplicity_preserved"]
    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM members").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM request_ledger"
            ).fetchone()[0]
            == expected_request_count
        )
        assert [
            row[0]
            for row in connection.execute("SELECT id FROM request_ledger ORDER BY id")
        ] == list(range(1, expected_request_count + 1))
        assert connection.execute("SELECT COUNT(*) FROM binding_upgrades").fetchone()[0] == 1
    request_map = [
        json.loads(line)
        for line in (destination / "ledgers/request_id_map.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["destination_id"] for record in request_map] == list(
        range(1, expected_request_count + 1)
    )
    assert [record["source_id"] for record in request_map] == source_request_ids

    fetch_receipt = json.loads(
        (destination / "fetch_receipt.json").read_text(encoding="utf-8")
    )
    store_receipt = json.loads(
        (destination / "store_receipt.json").read_text(encoding="utf-8")
    )
    with FrozenStore(
        destination / "content_store",
        destination / "store_receipt.json",
    ) as store:
        with FrozenFetchState(
            destination / "fetch_state.sqlite3",
            tokenizer=exact_tokenizer,
            store=store,
        ) as state:
            assert fetch_receipt["frozen_fetch_state"] == state.receipt_binding()
            assert receipt["frozen_fetch_state"] == state.receipt_binding()
            assert fetch_receipt["content_store_receipt"] == store_receipt


def test_official_continuation_clone_conserves_base_without_hardlinks(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "source-shard",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("base payload", run_id=100, archive_member="base.txt")],
    )
    manifest, base_union = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    merge_shards(manifest)
    source_state_hash = _sha256(base_union / "fetch_state.sqlite3")
    source_index_hash = _sha256(
        base_union / "content_store" / "index.sqlite3"
    )
    tokenizer_copy = tmp_path / "clone-tokenizer.json"
    shutil.copy2(TOKENIZER_JSON, tokenizer_copy)
    continuation = tmp_path / "continuation"

    receipt = clone_union_for_resume(
        base_union=base_union,
        destination=continuation,
        tokenizer_path=tokenizer_copy,
    )

    assert receipt["status"] == "complete"
    assert receipt["base_inclusion"]["no_mutable_hardlinks"] is True
    assert receipt["controls"]["self_contained"] is True
    assert not Path(receipt["base_union"]["path"]).is_absolute()
    assert not Path(receipt["tokenizer"]["path"]).is_absolute()
    assert not Path(
        receipt["continuation_inventory"]["source_path"]
    ).is_absolute()
    assert _sha256(base_union / "fetch_state.sqlite3") == source_state_hash
    assert (
        _sha256(base_union / "content_store" / "index.sqlite3")
        == source_index_hash
    )
    source_state = (base_union / "fetch_state.sqlite3").stat()
    copied_state = (continuation / "fetch_state.sqlite3").stat()
    assert (source_state.st_dev, source_state.st_ino) != (
        copied_state.st_dev,
        copied_state.st_ino,
    )
    with sqlite3.connect(
        continuation / "fetch_state.sqlite3"
    ) as connection:
        settings = dict(
            connection.execute("SELECT key,value FROM settings")
        )
        connection.execute(
            """
            UPDATE attempts SET status='retry'
            WHERE repo='owner/repo' AND run_id=100 AND attempt=1
            """
        )
    assert settings["inventory_path"] == str(
        continuation / "inventory.sqlite3"
    )
    assert settings["content_store_path"] == str(
        continuation / "content_store"
    )
    with sqlite3.connect(
        base_union / "fetch_state.sqlite3"
    ) as connection:
        assert connection.execute(
            """
            SELECT status FROM attempts
            WHERE repo='owner/repo' AND run_id=100 AND attempt=1
            """
        ).fetchone()[0] == "done"
    with sqlite3.connect(
        continuation / "fetch_state.sqlite3"
    ) as connection:
        connection.execute(
            """
            UPDATE attempts SET status='done'
            WHERE repo='owner/repo' AND run_id=100 AND attempt=1
            """
        )
    relocated_parent = tmp_path / "relocated"
    relocated_parent.mkdir()
    relocated = relocated_parent / "continuation"
    original_control_state = (
        continuation
        / "continuation_seed_controls"
        / "base_union"
        / "fetch_state.sqlite3"
    )
    shutil.copytree(continuation, relocated, copy_function=shutil.copy2)
    relocated_control_state = (
        relocated
        / "continuation_seed_controls"
        / "base_union"
        / "fetch_state.sqlite3"
    )
    assert (
        original_control_state.stat().st_dev,
        original_control_state.stat().st_ino,
    ) != (
        relocated_control_state.stat().st_dev,
        relocated_control_state.stat().st_ino,
    )
    shutil.rmtree(continuation)
    shutil.rmtree(base_union)
    shutil.rmtree(shard.root)
    inventory.unlink()
    tokenizer_copy.unlink()

    inclusion = verify_continuation_seed_inclusion(
        relocated / "continuation_seed_receipt.json",
        final_state_path=relocated / "fetch_state.sqlite3",
        final_store_root=relocated / "content_store",
    )
    assert inclusion["base_terminal_evidence_unchanged"] is True
    assert inclusion["base_cas_rows_unchanged"] is True
    assert Path(inclusion["base_union_path"]).is_relative_to(relocated)
    relocated_control_state.write_bytes(
        relocated_control_state.read_bytes() + b"tamper"
    )
    with pytest.raises(
        ReceiptFinalizationError,
        match=r"staged controls changed",
    ):
        verify_continuation_seed_inclusion(
            relocated / "continuation_seed_receipt.json",
            final_state_path=relocated / "fetch_state.sqlite3",
            final_store_root=relocated / "content_store",
        )


def test_continuation_tree_snapshot_is_bounded_aggregate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    (root / "packs").mkdir(parents=True)
    (root / "index.sqlite3").write_bytes(b"index")
    (root / "packs" / "000001.pack").write_bytes(b"payload")

    snapshot = _snapshot_tree(root)

    assert set(snapshot) == {
        "file_count",
        "byte_size",
        "artifact_set_sha256",
    }
    assert snapshot["file_count"] == 2
    assert snapshot["byte_size"] == 12
    assert len(str(snapshot["artifact_set_sha256"])) == 64


def _build_clone_preflight_base(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
    inventory: Path,
    inventory_receipt: dict[str, Any],
    *,
    record_attempt: int = 1,
) -> tuple[BuiltShard, Path]:
    text, provenance = _record(
        "clone preflight base",
        run_id=100,
        archive_member="base.txt",
    )
    if record_attempt != 1:
        provenance["run_attempt"] = record_attempt
        evidence = cast(
            dict[str, Any],
            provenance["run_metadata_evidence"],
        )
        evidence["source_attempt"] = record_attempt
        evidence["inventory_seed_attempt"] = record_attempt
    shard = _build_shard(
        tmp_path / "clone-base-shard",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [(text, provenance)],
    )
    manifest, base_union = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    merge_shards(manifest)
    return shard, base_union


def _refresh_clone_inventory_window_accounting(
    connection: sqlite3.Connection,
) -> None:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT repo_key,run_id,run_attempt,metadata_sha256
        FROM window_runs
        ORDER BY repo_key,run_id,run_attempt
        """
    ).fetchall()
    run_keys_sha256 = _hash_lines(
        f"{row['repo_key']}\t{row['run_id']}\t"
        f"{row['run_attempt']}\t{row['metadata_sha256']}"
        for row in rows
    )
    count = len(rows)
    connection.execute(
        """
        UPDATE search_windows
        SET expected_total=?,raw_items=?,distinct_items=?,
            duplicate_items=0,run_keys_sha256=?
        """,
        (count, count, count, run_keys_sha256),
    )
    connection.execute(
        """
        UPDATE window_pages
        SET total_count=?,item_count=?,distinct_item_count=?,
            duplicate_item_count=0,run_keys_sha256=?
        """,
        (count, count, count, run_keys_sha256),
    )


def _write_clone_inventory_receipt(
    inventory: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    receipt = InventoryDB(
        inventory,
        initialize_schema=False,
    ).completion_receipt()
    _write_json(receipt_path, receipt)
    return receipt


def test_continuation_clone_rejects_inventory_missing_a_base_run_before_partial(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    _shard, base_union = _build_clone_preflight_base(
        tmp_path,
        exact_tokenizer,
        inventory,
        inventory_receipt,
    )
    selected_inventory = tmp_path / "missing-run-inventory.sqlite3"
    shutil.copy2(inventory, selected_inventory)
    with sqlite3.connect(selected_inventory) as connection:
        connection.execute(
            "DELETE FROM window_runs WHERE run_id=300"
        )
        connection.execute("DELETE FROM runs WHERE run_id=300")
        _refresh_clone_inventory_window_accounting(connection)
        connection.commit()
    selected_receipt = tmp_path / "missing-run-receipt.json"
    _write_clone_inventory_receipt(
        selected_inventory,
        selected_receipt,
    )
    destination = tmp_path / "must-not-exist" / "continuation"

    with pytest.raises(
        CloneError,
        match=r"missing base run owner/repo#300",
    ):
        clone_union_for_resume(
            base_union=base_union,
            destination=destination,
            tokenizer_path=TOKENIZER_JSON,
            inventory_path=selected_inventory,
            inventory_receipt_path=selected_receipt,
        )

    assert not destination.parent.exists()


def test_continuation_clone_rejects_lower_run_attempt_ceiling_before_partial(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    (
        base_inventory,
        _base_inventory_receipt_path,
        base_inventory_receipt,
    ) = _inventory_with_new_attempt(
        tmp_path,
        inventory,
        run_id=100,
        run_attempt=2,
    )
    _shard, base_union = _build_clone_preflight_base(
        tmp_path,
        exact_tokenizer,
        base_inventory,
        base_inventory_receipt,
        record_attempt=2,
    )
    selected_receipt = tmp_path / "lower-ceiling-receipt.json"
    _write_json(selected_receipt, inventory_receipt)
    destination = tmp_path / "must-not-exist" / "continuation"

    with pytest.raises(
        CloneError,
        match=r"run_attempt ceiling shrank for owner/repo#100: 1 < 2",
    ):
        clone_union_for_resume(
            base_union=base_union,
            destination=destination,
            tokenizer_path=TOKENIZER_JSON,
            inventory_path=inventory,
            inventory_receipt_path=selected_receipt,
        )

    assert not destination.parent.exists()


def test_continuation_clone_rejects_unrelated_repository_scope_before_partial(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    _shard, base_union = _build_clone_preflight_base(
        tmp_path,
        exact_tokenizer,
        inventory,
        inventory_receipt,
    )
    selected_inventory = tmp_path / "unrelated-scope-inventory.sqlite3"
    shutil.copy2(inventory, selected_inventory)
    with sqlite3.connect(selected_inventory) as connection:
        connection.execute(
            """
            UPDATE repos
            SET repo_key='other/repo',owner='other',
                name='repo',canonical='other/repo'
            """
        )
        for table in ("search_windows", "runs", "window_runs"):
            connection.execute(
                f"UPDATE {table} SET repo_key='other/repo'"
            )
        connection.execute(
            """
            UPDATE request_ledger
            SET repo_key='other/repo',
                endpoint='/repos/other/repo/actions/runs'
            """
        )
        connection.execute(
            """
            UPDATE inventory_meta SET value=?
            WHERE key='repo_scope_sha256'
            """,
            (_hash_lines(("other/repo",)),),
        )
        connection.execute(
            """
            UPDATE inventory_meta SET value=?
            WHERE key='repo_list_sha256'
            """,
            ("9" * 64,),
        )
        _refresh_clone_inventory_window_accounting(connection)
        connection.commit()
    selected_receipt = tmp_path / "unrelated-scope-receipt.json"
    _write_clone_inventory_receipt(
        selected_inventory,
        selected_receipt,
    )
    destination = tmp_path / "must-not-exist" / "continuation"

    with pytest.raises(
        CloneError,
        match=r"repository scope or declared interval differs",
    ):
        clone_union_for_resume(
            base_union=base_union,
            destination=destination,
            tokenizer_path=TOKENIZER_JSON,
            inventory_path=selected_inventory,
            inventory_receipt_path=selected_receipt,
        )

    assert not destination.parent.exists()


def test_continuation_clone_rejects_legacy_creator_bound_store_before_partial(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    _shard, base_union = _build_clone_preflight_base(
        tmp_path,
        exact_tokenizer,
        inventory,
        inventory_receipt,
    )
    legacy_creator = "0" * 64
    with sqlite3.connect(
        base_union / "content_store" / "index.sqlite3"
    ) as connection:
        connection.execute(
            """
            UPDATE settings SET value=?
            WHERE key='creator_script_sha256'
            """,
            (legacy_creator,),
        )
        connection.commit()
    with CIContentStore(base_union / "content_store") as store:
        legacy_store_receipt = store.completion_receipt(
            target_unique_tokens=0
        )
    _write_json(
        base_union / "store_receipt.json",
        legacy_store_receipt,
    )
    with sqlite3.connect(
        base_union / "fetch_state.sqlite3"
    ) as connection:
        connection.execute(
            """
            UPDATE settings SET value=?
            WHERE key='content_store_script_sha256'
            """,
            (legacy_creator,),
        )
        connection.commit()

    fetch_receipt_path = base_union / "fetch_receipt.json"
    fetch_receipt = json.loads(
        fetch_receipt_path.read_text(encoding="utf-8")
    )
    with FrozenStore(
        base_union / "content_store",
        base_union / "store_receipt.json",
    ) as frozen_store:
        with FrozenFetchState(
            base_union / "fetch_state.sqlite3",
            tokenizer=exact_tokenizer,
            store=frozen_store,
        ) as frozen_state:
            frozen_binding = frozen_state.receipt_binding()
            frozen_binding["artifact"]["path"] = str(
                base_union / "fetch_state.sqlite3"
            )
            fetch_receipt["fetch_state"] = frozen_state.summary
            fetch_receipt["frozen_fetch_state"] = frozen_binding
            fetch_receipt["content_store_receipt"] = (
                legacy_store_receipt
            )
            frozen_state.require_unchanged()
        frozen_store.require_unchanged()
    _write_json(fetch_receipt_path, fetch_receipt)

    merge_receipt_path = base_union / "merge_receipt.json"
    merge_receipt = json.loads(
        merge_receipt_path.read_text(encoding="utf-8")
    )
    merge_receipt["frozen_fetch_state"] = frozen_binding
    for artifact in merge_receipt["artifacts"]:
        artifact_path = base_union / artifact["path"]
        artifact["byte_size"] = artifact_path.stat().st_size
        artifact["sha256"] = _sha256(artifact_path)
    _write_json(merge_receipt_path, merge_receipt)
    destination = tmp_path / "must-not-exist" / "continuation"

    with pytest.raises(
        CloneError,
        match=r"not continuation-writable.*explicit logical",
    ):
        clone_union_for_resume(
            base_union=base_union,
            destination=destination,
            tokenizer_path=TOKENIZER_JSON,
        )

    assert not destination.parent.exists()


def test_newer_rerun_continues_base_through_clone_merge_and_export(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    base_shard = _build_shard(
        tmp_path / "base-shard",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [
            _record("base run 100", run_id=100, archive_member="100.txt"),
            _record("base run 200", run_id=200, archive_member="200.txt"),
            _record("base run 300", run_id=300, archive_member="300.txt"),
        ],
    )
    _promote_to_exhaustive_receipt(base_shard)
    base_manifest = tmp_path / "base-production-manifest.json"
    base_union = tmp_path / "base-production-union"
    build_canonical_manifest(
        base_manifest,
        destination=base_union,
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=1,
        shards=[
            {
                "id": "base",
                "role": "coverage",
                "inventory": base_shard.inventory,
                "inventory_receipt": base_shard.inventory_receipt,
                "content_store": base_shard.store,
                "store_receipt": base_shard.store_receipt,
                "fetch_state": base_shard.state,
                "fetch_receipt": base_shard.fetch_receipt,
            }
        ],
    )
    merge_shards(base_manifest)

    (
        rerun_inventory,
        rerun_inventory_receipt,
        _rerun_receipt,
    ) = _inventory_with_new_attempt(
        tmp_path,
        inventory,
        run_id=100,
        run_attempt=2,
    )
    continuation = tmp_path / "rerun-continuation"
    clone_union_for_resume(
        base_union=base_union,
        destination=continuation,
        tokenizer_path=TOKENIZER_JSON,
        inventory_path=rerun_inventory,
        inventory_receipt_path=rerun_inventory_receipt,
    )

    continuation_inventory_receipt = json.loads(
        (continuation / "inventory_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    artifact = cast(
        Mapping[str, Any],
        continuation_inventory_receipt["database_artifact"],
    )
    binding = ExhaustiveInventoryBinding(
        receipt_path=continuation / "inventory_receipt.json",
        receipt_sha256=_sha256(
            continuation / "inventory_receipt.json"
        ),
        database_sha256=str(artifact["sha256"]),
        db_logical_sha256=str(
            continuation_inventory_receipt["db_logical_sha256"]
        ),
        expected_run_count=int(
            continuation_inventory_receipt["run_count"]
        ),
        expected_attempt_count=int(
            continuation_inventory_receipt["expected_attempt_count"]
        ),
        expected_attempt_set_sha256=str(
            continuation_inventory_receipt[
                "expected_attempt_set_sha256"
            ]
        ),
    )
    state = FetchState(
        continuation / "fetch_state.sqlite3",
        inventory_path=continuation / "inventory.sqlite3",
        content_store_path=continuation / "content_store",
        tokenizer=exact_tokenizer,
        resume=True,
        allow_fetcher_script_upgrade_from_sha256="4" * 64,
        fetcher_script_upgrade_reason=(
            "continue fixture union with the verified current fetcher"
        ),
    )
    try:
        assert state.discover(
            row_limit=100,
            exhaustive_inventory=binding,
        ) == 1
        attempt = state.next_attempt()
        assert attempt is not None
        assert (attempt.repo, attempt.run_id, attempt.attempt) == (
            "owner/repo",
            100,
            2,
        )
        assert attempt.run_metadata_exact is True
        empty_zip = b"PK\x05\x06" + (b"\x00" * 18)
        for endpoint, page_no in (
            (
                "/repos/owner/repo/actions/runs/100/attempts/2/logs",
                None,
            ),
            (
                "/repos/owner/repo/actions/runs/100/attempts/2/jobs",
                1,
            ),
        ):
            state.record_request(
                attempt,
                endpoint=endpoint,
                page_no=page_no,
                request_attempt=1,
                http_status=200,
                outcome="success",
                latency_ms=1,
            )
        state.finish_attempt(
            attempt,
            status="empty",
            archive_source="github-api-empty-zip",
            archive_sha256=hashlib.sha256(empty_zip).hexdigest(),
            archive_size=len(empty_zip),
            archive_bytes=empty_zip,
            jobs=[],
        )
        assert state.next_attempt() is None
    finally:
        state.close()

    continuation_fetch_receipt = continuation / "fetch_receipt.json"
    continued = finalize_fetch_receipts(
        state_path=continuation / "fetch_state.sqlite3",
        content_store_path=continuation / "content_store",
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=1,
        fetch_receipt_path=continuation_fetch_receipt,
        store_receipt_path=continuation / "store_receipt.json",
        original_state_path=continuation / "fetch_state.sqlite3",
        original_content_store_path=continuation / "content_store",
        original_inventory_path=continuation / "inventory.sqlite3",
        completion_mode=COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        inventory_receipt_path=continuation / "inventory_receipt.json",
        continuation_seed_receipt_path=(
            continuation / "continuation_seed_receipt.json"
        ),
    )
    assert continued["production_complete"] is True
    assert continued["continuation_seed"][
        "base_terminal_evidence_unchanged"
    ] is True
    assert continued["exhaustive_coverage"]["expected_attempt_count"] == 4

    final_manifest = tmp_path / "rerun-production-manifest.json"
    final_union = tmp_path / "rerun-production-union"
    build_canonical_manifest(
        final_manifest,
        destination=final_union,
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=1,
        shards=[
            {
                "id": "rerun-coverage",
                "role": "coverage",
                "inventory": continuation / "inventory.sqlite3",
                "inventory_receipt": (
                    continuation / "inventory_receipt.json"
                ),
                "content_store": continuation / "content_store",
                "store_receipt": continuation / "store_receipt.json",
                "fetch_state": continuation / "fetch_state.sqlite3",
                "fetch_receipt": continuation_fetch_receipt,
            }
        ],
    )
    merged = merge_shards(final_manifest)
    assert merged["production_complete"] is True
    with sqlite3.connect(
        final_union / "fetch_state.sqlite3"
    ) as connection:
        assert connection.execute(
            """
            SELECT status FROM attempts
            WHERE repo='owner/repo' AND run_id=100 AND attempt=1
            """
        ).fetchone()[0] == "done"
        assert connection.execute(
            """
            SELECT status FROM attempts
            WHERE repo='owner/repo' AND run_id=100 AND attempt=2
            """
        ).fetchone()[0] == "empty"

    exported = export_store(
        store_root=final_union / "content_store",
        store_receipt=final_union / "store_receipt.json",
        fetch_state=final_union / "fetch_state.sqlite3",
        tokenizer_json=TOKENIZER_JSON,
        output=tmp_path / "rerun-case5-production",
        completion_mode=COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        inventory=final_union / "inventory.sqlite3",
        inventory_receipt=final_union / "inventory_receipt.json",
        fetch_receipt=final_union / "fetch_receipt.json",
        merge_receipt=final_union / "merge_receipt.json",
    )
    assert exported["production_complete"] is True


def test_production_merge_composes_disjoint_inventory_scopes(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    shards = [
        _production_shard_for_repo(
            tmp_path,
            exact_tokenizer,
            label="base",
            repo_key="owner/base",
        ),
        _production_shard_for_repo(
            tmp_path,
            exact_tokenizer,
            label="supplemental",
            repo_key="owner/supplemental",
        ),
    ]
    manifest = tmp_path / "disjoint-production-manifest.json"
    destination = tmp_path / "disjoint-production-union"
    build_canonical_manifest(
        manifest,
        destination=destination,
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=1,
        shards=[
            {
                "id": label,
                "role": "coverage",
                "inventory": shard.inventory,
                "inventory_receipt": shard.inventory_receipt,
                "content_store": shard.store,
                "store_receipt": shard.store_receipt,
                "fetch_state": shard.state,
                "fetch_receipt": shard.fetch_receipt,
            }
            for label, shard in zip(
                ("base", "supplemental"),
                shards,
                strict=True,
            )
        ],
    )

    merged = merge_shards(manifest)

    assert merged["production_complete"] is True
    assert merged["inventory"]["policy"] == (
        "disjoint-completed-production-inventory-union-v1"
    )
    with sqlite3.connect(destination / "inventory.sqlite3") as inventory:
        assert inventory.execute("SELECT COUNT(*) FROM repos").fetchone()[0] == 2
        assert inventory.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 6
        assert inventory.execute(
            "SELECT id FROM request_ledger ORDER BY id"
        ).fetchall() == [(1,), (2,)]
        assert inventory.execute(
            """
            SELECT COUNT(*) FROM request_ledger request
            JOIN search_windows window ON window.id=request.window_id
            WHERE request.repo_key=window.repo_key
            """
        ).fetchone()[0] == 2
    with sqlite3.connect(destination / "fetch_state.sqlite3") as state:
        assert state.execute(
            "SELECT COUNT(DISTINCT repo) FROM attempts"
        ).fetchone()[0] == 2
        assert state.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 6
    fetch_receipt = json.loads(
        (destination / "fetch_receipt.json").read_text(encoding="utf-8")
    )
    assert fetch_receipt["inventory_binding"]["repo_count"] == 2
    assert fetch_receipt["exhaustive_coverage"]["missing_attempt_count"] == 0
    assert merged["store_conservation"]["equations"]["tokens"] is True
    exported = export_store(
        store_root=destination / "content_store",
        store_receipt=destination / "store_receipt.json",
        fetch_state=destination / "fetch_state.sqlite3",
        tokenizer_json=TOKENIZER_JSON,
        output=tmp_path / "disjoint-case5",
        completion_mode=COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        inventory=destination / "inventory.sqlite3",
        inventory_receipt=destination / "inventory_receipt.json",
        fetch_receipt=destination / "fetch_receipt.json",
        merge_receipt=destination / "merge_receipt.json",
    )
    assert exported["production_complete"] is True


def test_production_merge_rejects_overlapping_inventory_scopes(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    shards = [
        _production_shard_for_repo(
            tmp_path,
            exact_tokenizer,
            label="first",
            repo_key="owner/shared",
        ),
        _production_shard_for_repo(
            tmp_path,
            exact_tokenizer,
            label="second",
            repo_key="owner/shared",
            user_version=1,
        ),
    ]
    manifest = tmp_path / "overlap-production-manifest.json"
    destination = tmp_path / "overlap-production-union"
    build_canonical_manifest(
        manifest,
        destination=destination,
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=1,
        shards=[
            {
                "id": label,
                "role": "coverage",
                "inventory": shard.inventory,
                "inventory_receipt": shard.inventory_receipt,
                "content_store": shard.store,
                "store_receipt": shard.store_receipt,
                "fetch_state": shard.state,
                "fetch_receipt": shard.fetch_receipt,
            }
            for label, shard in zip(("first", "second"), shards, strict=True)
        ],
    )

    with pytest.raises(MergeError, match="overlapping repository scope"):
        merge_shards(manifest)

    assert not destination.exists()


def test_production_merge_and_export_require_and_preserve_exhaustive_v4(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "full",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [
            _record("run one", run_id=100, archive_member="100.txt"),
            _record("run two", run_id=200, archive_member="200.txt"),
            _record("run three", run_id=300, archive_member="300.txt"),
        ],
    )
    _promote_to_exhaustive_receipt(shard)
    manifest = tmp_path / "production-manifest.json"
    destination = tmp_path / "production-union"
    value = build_canonical_manifest(
        manifest,
        destination=destination,
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=1,
        shards=[
            {
                "id": "full",
                "role": "coverage",
                "inventory": shard.inventory,
                "inventory_receipt": shard.inventory_receipt,
                "content_store": shard.store,
                "store_receipt": shard.store_receipt,
                "fetch_state": shard.state,
                "fetch_receipt": shard.fetch_receipt,
            }
        ],
    )
    assert value["schema"] == PRODUCTION_MANIFEST_SCHEMA

    merge_receipt = merge_shards(manifest)

    assert merge_receipt["production_complete"] is True
    assert merge_receipt["verification"][
        "exact_production_inventory_attempt_equality"
    ] is True
    fetch_receipt = json.loads(
        (destination / "fetch_receipt.json").read_text(encoding="utf-8")
    )
    assert fetch_receipt["production_complete"] is True
    assert fetch_receipt["exhaustive_coverage"][
        "missing_attempt_count"
    ] == 0
    output = tmp_path / "case5-production"
    export_receipt = export_store(
        store_root=destination / "content_store",
        store_receipt=destination / "store_receipt.json",
        fetch_state=destination / "fetch_state.sqlite3",
        tokenizer_json=TOKENIZER_JSON,
        output=output,
        completion_mode=COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        inventory=destination / "inventory.sqlite3",
        inventory_receipt=destination / "inventory_receipt.json",
        fetch_receipt=destination / "fetch_receipt.json",
        merge_receipt=destination / "merge_receipt.json",
    )
    assert export_receipt["production_complete"] is True
    assert export_receipt["acquisition_provenance"][
        "production_complete"
    ] is True


def test_merge_to_export_executes_current_semantics_through_unknown_lineage(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "unknown-current",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [
            _record_with_source_binding(
                run_id=100,
                archive_member="current.txt",
                binding_mode="current",
            )
        ],
    )
    current = target_parser_script_sha256()
    _set_parser_binding_history(
        shard,
        exact_tokenizer,
        ((_UNKNOWN_PARSER_SHA256, current),),
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )

    merge_shards(manifest)

    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        assert connection.execute(
            """
            SELECT from_sha256,to_sha256
            FROM binding_upgrades
            WHERE binding_key='parser_script_sha256'
            """
        ).fetchall() == [(_UNKNOWN_PARSER_SHA256, current)]
    output = tmp_path / "unknown-current-export"
    receipt = export_store(
        store_root=destination / "content_store",
        store_receipt=destination / "store_receipt.json",
        fetch_state=destination / "fetch_state.sqlite3",
        tokenizer_json=TOKENIZER_JSON,
        output=output,
    )
    projection = receipt["source_binding_projection"]
    assert projection["mode"] == "mixed_lineage_projection"
    assert projection["parser_lineage"] == [
        _UNKNOWN_PARSER_SHA256,
        current,
    ]
    assert projection["selection_counts"] == {"current_audit": 1}
    records = [
        record
        for record, _encoded in iter_canonical_parquet_ledger(
            output / projection["ledger_artifact"],
            expected_domain=SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
            expected_record_schema=SOURCE_BINDING_PROJECTION_SCHEMA,
        )
    ]
    assert {record["input_parser_sha256"] for record in records} == {
        current
    }
    assert {record["change_kind"] for record in records} == {"unchanged"}


def test_merge_preserves_shortcut_dag_but_export_executes_only_known_semantics(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "legacy-unknown-current",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [
            _record_with_source_binding(
                run_id=100,
                archive_member="legacy.txt",
                binding_mode="legacy",
            ),
            _record_with_source_binding(
                run_id=200,
                archive_member="current.txt",
                binding_mode="current",
            ),
        ],
    )
    current = target_parser_script_sha256()
    transitions = (
        (LEGACY_PARSER_SHA256, _UNKNOWN_PARSER_SHA256),
        (_UNKNOWN_PARSER_SHA256, current),
        (LEGACY_PARSER_SHA256, current),
    )
    _set_parser_binding_history(shard, exact_tokenizer, transitions)
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )

    merge_shards(manifest)

    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        retained_edges = set(
            connection.execute(
                """
                SELECT from_sha256,to_sha256
                FROM binding_upgrades
                WHERE binding_key='parser_script_sha256'
                """
            ).fetchall()
        )
    assert retained_edges == set(transitions)
    output = tmp_path / "legacy-unknown-current-export"
    receipt = export_store(
        store_root=destination / "content_store",
        store_receipt=destination / "store_receipt.json",
        fetch_state=destination / "fetch_state.sqlite3",
        tokenizer_json=TOKENIZER_JSON,
        output=output,
        source_binding_projection_from_parser_sha256=LEGACY_PARSER_SHA256,
    )
    projection = receipt["source_binding_projection"]
    assert projection["mode"] == "mixed_lineage_projection"
    assert projection["parser_lineage"] == [
        LEGACY_PARSER_SHA256,
        _UNKNOWN_PARSER_SHA256,
        current,
    ]
    assert projection["selection_counts"] == {
        "current_audit": 1,
        "legacy_projection": 1,
    }
    records = [
        record
        for record, _encoded in iter_canonical_parquet_ledger(
            output / projection["ledger_artifact"],
            expected_domain=SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
            expected_record_schema=SOURCE_BINDING_PROJECTION_SCHEMA,
        )
    ]
    assert {record["input_parser_sha256"] for record in records} == {
        LEGACY_PARSER_SHA256,
        current,
    }
    assert all(
        record["input_parser_sha256"] != _UNKNOWN_PARSER_SHA256
        for record in records
    )


def test_merge_to_export_rejects_unimplemented_third_parser_semantics(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "third-semantics",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [
            _record_with_source_binding(
                run_id=100,
                archive_member="third.txt",
                binding_mode="third",
            )
        ],
    )
    current = target_parser_script_sha256()
    _set_parser_binding_history(
        shard,
        exact_tokenizer,
        ((_UNKNOWN_PARSER_SHA256, current),),
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )

    merge_shards(manifest)

    output = tmp_path / "third-semantics-export"
    with pytest.raises(
        ExportError,
        match="every executable supported parser semantics",
    ):
        export_store(
            store_root=destination / "content_store",
            store_receipt=destination / "store_receipt.json",
            fetch_state=destination / "fetch_state.sqlite3",
            tokenizer_json=TOKENIZER_JSON,
            output=output,
        )
    assert not output.exists()


def test_production_merge_and_export_reject_threshold_v3(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "legacy",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("legacy", run_id=100, archive_member="legacy.txt")],
    )
    manifest = tmp_path / "must-refuse-v3.json"
    build_canonical_manifest(
        manifest,
        destination=tmp_path / "must-not-publish",
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=0,
        shards=[
            {
                "id": "legacy",
                "role": "coverage",
                "inventory": shard.inventory,
                "inventory_receipt": shard.inventory_receipt,
                "content_store": shard.store,
                "store_receipt": shard.store_receipt,
                "fetch_state": shard.state,
                "fetch_receipt": shard.fetch_receipt,
            }
        ],
    )
    with pytest.raises(MergeError, match="requires a verified.*v4"):
        merge_shards(manifest)

    legacy_manifest, legacy_union = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    merge_shards(legacy_manifest)
    with pytest.raises(
        ExportError,
        match="inventory-exhaustive fetch receipt v4",
    ):
        export_store(
            store_root=legacy_union / "content_store",
            store_receipt=legacy_union / "store_receipt.json",
            fetch_state=legacy_union / "fetch_state.sqlite3",
            tokenizer_json=TOKENIZER_JSON,
            output=tmp_path / "must-not-export",
            completion_mode=COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
            inventory=legacy_union / "inventory.sqlite3",
            inventory_receipt=legacy_union / "inventory_receipt.json",
            fetch_receipt=legacy_union / "fetch_receipt.json",
            merge_receipt=legacy_union / "merge_receipt.json",
        )


def test_legacy_v2_partial_replays_fresh_into_v3_journal_and_v4_state(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "legacy-input",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("legacy replay", run_id=100, archive_member="legacy.txt")],
        add_ledgers=True,
    )
    empty_witness = _build_shard(
        tmp_path / "legacy-empty-witness",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("empty witness", run_id=200, archive_member="empty.txt")],
    )
    _insert_attempt_variant(
        shard.state,
        empty_witness.state,
        status="empty",
    )
    _write_json(shard.fetch_receipt, _fetch_receipt(shard, exact_tokenizer))
    source_manifest, source_destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    merge_shards(source_manifest)
    _downgrade_fetch_state_to_legacy_v3(
        source_destination / "fetch_state.sqlite3",
        source_destination / "fetch_receipt.json",
    )
    _downgrade_merge_journal_to_legacy_v2(
        source_destination / "merge_journal.sqlite3"
    )
    legacy_partial = source_destination.with_name(
        f".{source_destination.name}.partial"
    )
    source_destination.replace(legacy_partial)
    legacy_before = _tree_file_hashes(legacy_partial)

    destination = tmp_path / "fresh-union"
    manifest = tmp_path / "fresh-replay-manifest.json"
    built = build_canonical_manifest(
        manifest,
        destination=destination,
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=0,
        completion_mode=COMPLETION_MODE_THRESHOLD,
        migration={
            "legacy_partial": legacy_partial,
            "source_manifest": source_manifest,
            "source_journal": (
                legacy_partial / "merge_journal.sqlite3"
            ),
        },
        shards=[
            {
                "id": "legacy-union",
                "role": "legacy-coverage",
                "inventory": legacy_partial / "inventory.sqlite3",
                "inventory_receipt": (
                    legacy_partial / "inventory_receipt.json"
                ),
                "content_store": legacy_partial / "content_store",
                "store_receipt": legacy_partial / "store_receipt.json",
                "fetch_state": legacy_partial / "fetch_state.sqlite3",
                "fetch_receipt": legacy_partial / "fetch_receipt.json",
            }
        ],
    )
    assert built["migration"]["schema"] == MIGRATION_SCHEMA
    assert built["migration"]["mode"] == MIGRATION_MODE

    with pytest.raises(MergePaused):
        merge_shards(manifest, max_batches=1)
    assert _tree_file_hashes(legacy_partial) == legacy_before
    fresh_partial = destination.with_name(f".{destination.name}.partial")
    assert fresh_partial.is_dir()
    assert not list(
        fresh_partial.rglob("*-effective-fetch-state-v4.sqlite3")
    )

    projection_residue = (
        fresh_partial
        / ".source-verification"
        / (
            "..legacy-union-effective-fetch-state-v4.sqlite3"
            ".project-v4-crash.sqlite"
        )
    )
    projection_residue.write_bytes(b"simulated interrupted projection")
    with pytest.raises(MergePaused):
        merge_shards(manifest, max_batches=1)
    assert not projection_residue.exists()
    assert not list(
        fresh_partial.rglob("*-effective-fetch-state-v4.sqlite3")
    )
    assert _tree_file_hashes(legacy_partial) == legacy_before

    legacy_pack = next(
        (legacy_partial / "content_store").glob("pack-*.cicp")
    )
    fresh_pack = fresh_partial / "content_store" / legacy_pack.name
    fresh_pack_backup = tmp_path / "fresh-pack-backup.cicp"
    shutil.copy2(fresh_pack, fresh_pack_backup)
    fresh_pack.unlink()
    os.link(legacy_pack, fresh_pack)
    assert (
        legacy_pack.stat().st_dev,
        legacy_pack.stat().st_ino,
    ) == (
        fresh_pack.stat().st_dev,
        fresh_pack.stat().st_ino,
    )
    with pytest.raises(
        MergeError,
        match="hardlinks an immutable legacy input",
    ):
        merge_shards(manifest)
    assert _tree_file_hashes(legacy_partial) == legacy_before
    fresh_pack.unlink()
    shutil.copy2(fresh_pack_backup, fresh_pack)

    receipt = merge_shards(manifest)

    assert destination.is_dir()
    assert _tree_file_hashes(legacy_partial) == legacy_before
    migration = receipt["migration"]
    assert migration["legacy_source"]["journal"]["schema"] == (
        LEGACY_JOURNAL_SCHEMA
    )
    assert migration["legacy_source"]["journal"]["resumed_or_mutated"] is False
    journal_audit = migration["legacy_source"]["journal"]["audit"]
    assert journal_audit["phase"] == "ready"
    assert journal_audit["store_progress"]["all_done"] is True
    assert journal_audit["state_progress"]["all_done"] is True
    assert (
        journal_audit["resolution_maps"]["counts_equal_state_progress"]
        is True
    )
    assert journal_audit["semantic_role"] == "immutable-lineage-only"
    assert journal_audit["used_for_resume"] is False
    assert journal_audit["used_for_output_semantics"] is False
    assert migration["state_projection"]["source"][
        "sqlite_schema_sha256"
    ] == LEGACY_V3_SQLITE_SCHEMA_SHA256
    assert migration["state_projection"]["destination"][
        "sqlite_schema_sha256"
    ] == CURRENT_V4_SQLITE_SCHEMA_SHA256
    assert migration["state_projection"]["requeued_attempts"] == 1
    assert len(migration["state_projection"]["ledger"]) == 1
    assert migration["state_projection"]["ledger"][0]["action"] == "requeue"
    assert len(migration["state_projection"]["ledger_sha256"]) == 64
    assert migration["cas_semantic_oracle"]["equal"] is True
    assert (
        migration["cas_semantic_oracle"]["source"]
        == migration["cas_semantic_oracle"]["destination"]
    )
    assert migration["orphan_content_rows_rejected_preflight"] is True
    assert (
        migration["fresh_destination"]["legacy_artifacts_overwritten"]
        is False
    )
    assert migration["fresh_destination"]["journal_schema"] == JOURNAL_SCHEMA
    assert migration["fresh_destination"]["state_schema"] == (
        "cppmega_ci_stream_fetch_v4"
    )
    source_audit = receipt["sources"][0]["staged"]["fetch_state"]
    assert source_audit["original"]["settings_schema"] == (
        LEGACY_FETCH_STATE_SCHEMA
    )
    assert source_audit["effective"]["settings_schema"] == (
        "cppmega_ci_stream_fetch_v4"
    )
    assert source_audit["effective"]["ephemeral_projection"] is True
    assert source_audit["projection"] == migration["state_projection"]

    with sqlite3.connect(
        destination / "merge_journal.sqlite3"
    ) as journal:
        journal.row_factory = sqlite3.Row
        journal_settings = dict(
            journal.execute(
                "SELECT key,value FROM settings ORDER BY key"
            )
        )
        assert journal_settings["schema"] == JOURNAL_SCHEMA
        assert journal_settings["manifest_sha256"] == _sha256(manifest)
        assert journal_settings["destination"] == str(destination)
        assert (
            _sqlite_schema_sha256(journal)
            != LEGACY_JOURNAL_V2_SQLITE_SCHEMA_SHA256
        )
    with sqlite3.connect(destination / "fetch_state.sqlite3") as state:
        state.row_factory = sqlite3.Row
        settings = {
            str(row["key"]): str(row["value"])
            for row in state.execute(
                "SELECT key,value FROM settings"
            )
        }
        assert settings["schema"] == "cppmega_ci_stream_fetch_v4"
        assert settings["fetcher_script_sha256"] == (
            _current_fetcher_script_sha256()
        )
        assert _sqlite_schema_sha256(state) == (
            CURRENT_V4_SQLITE_SCHEMA_SHA256
        )
        assert state.execute(
            """
            SELECT COUNT(*) FROM binding_upgrades
            WHERE binding_key='fetcher_script_sha256'
              AND from_sha256=?
              AND to_sha256=?
            """,
            ("4" * 64, _current_fetcher_script_sha256()),
        ).fetchone()[0] == 1
        assert state.execute(
            "SELECT status FROM attempts WHERE run_id=200"
        ).fetchone()[0] == "retry"


def test_legacy_fresh_replay_rejects_orphan_cas_content_preflight(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    orphan_text = "orphan"
    shard = _build_shard(
        tmp_path / "legacy-orphan-input",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [
            _record("repeat", run_id=100, archive_member="repeat-a.txt"),
            _record("repeat", run_id=200, archive_member="repeat-b.txt"),
            _record(orphan_text, run_id=300, archive_member="orphan.txt"),
        ],
    )
    source_manifest, source_destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    merge_shards(source_manifest)
    _remove_only_occurrence_witness(
        store_root=source_destination / "content_store",
        store_receipt_path=source_destination / "store_receipt.json",
        fetch_receipt_path=source_destination / "fetch_receipt.json",
        content_sha256=hashlib.sha256(orphan_text.encode("utf-8")).hexdigest(),
    )
    _downgrade_fetch_state_to_legacy_v3(
        source_destination / "fetch_state.sqlite3",
        source_destination / "fetch_receipt.json",
    )
    _downgrade_merge_journal_to_legacy_v2(
        source_destination / "merge_journal.sqlite3"
    )
    legacy_partial = source_destination.with_name(
        f".{source_destination.name}.partial"
    )
    source_destination.replace(legacy_partial)
    legacy_before = _tree_file_hashes(legacy_partial)

    destination = tmp_path / "must-not-publish"
    manifest = tmp_path / "orphan-replay-manifest.json"
    build_canonical_manifest(
        manifest,
        destination=destination,
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=0,
        completion_mode=COMPLETION_MODE_THRESHOLD,
        migration={
            "legacy_partial": legacy_partial,
            "source_manifest": source_manifest,
            "source_journal": (
                legacy_partial / "merge_journal.sqlite3"
            ),
        },
        shards=[
            {
                "id": "legacy-orphan-union",
                "role": "legacy-coverage",
                "inventory": legacy_partial / "inventory.sqlite3",
                "inventory_receipt": (
                    legacy_partial / "inventory_receipt.json"
                ),
                "content_store": legacy_partial / "content_store",
                "store_receipt": legacy_partial / "store_receipt.json",
                "fetch_state": legacy_partial / "fetch_state.sqlite3",
                "fetch_receipt": legacy_partial / "fetch_receipt.json",
            }
        ],
    )

    with pytest.raises(
        MergeError,
        match="content has no occurrence witness",
    ):
        merge_shards(manifest)

    assert not destination.exists()
    assert _tree_file_hashes(legacy_partial) == legacy_before


def test_legacy_fresh_replay_rejects_incomplete_v2_journal_audit(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "legacy-journal-input",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("journal audit", run_id=100, archive_member="audit.txt")],
    )
    source_manifest, source_destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    merge_shards(source_manifest)
    _downgrade_fetch_state_to_legacy_v3(
        source_destination / "fetch_state.sqlite3",
        source_destination / "fetch_receipt.json",
    )
    journal_path = source_destination / "merge_journal.sqlite3"
    _downgrade_merge_journal_to_legacy_v2(journal_path)
    with sqlite3.connect(journal_path) as journal:
        journal.execute("DELETE FROM attempt_map")
        journal.commit()
    legacy_partial = source_destination.with_name(
        f".{source_destination.name}.partial"
    )
    source_destination.replace(legacy_partial)
    legacy_before = _tree_file_hashes(legacy_partial)

    destination = tmp_path / "must-not-publish-journal-gap"
    manifest = tmp_path / "journal-gap-replay-manifest.json"
    build_canonical_manifest(
        manifest,
        destination=destination,
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=0,
        completion_mode=COMPLETION_MODE_THRESHOLD,
        migration={
            "legacy_partial": legacy_partial,
            "source_manifest": source_manifest,
            "source_journal": (
                legacy_partial / "merge_journal.sqlite3"
            ),
        },
        shards=[
            {
                "id": "legacy-journal-gap",
                "role": "legacy-coverage",
                "inventory": legacy_partial / "inventory.sqlite3",
                "inventory_receipt": (
                    legacy_partial / "inventory_receipt.json"
                ),
                "content_store": legacy_partial / "content_store",
                "store_receipt": legacy_partial / "store_receipt.json",
                "fetch_state": legacy_partial / "fetch_state.sqlite3",
                "fetch_receipt": legacy_partial / "fetch_receipt.json",
            }
        ],
    )

    with pytest.raises(
        MergeError,
        match="attempt_map does not account for every processed attempts row",
    ):
        merge_shards(manifest)

    assert not destination.exists()
    assert _tree_file_hashes(legacy_partial) == legacy_before


def test_exact_overlap_is_globally_deduplicated(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    first = _record("same payload", run_id=100, archive_member="same.txt")
    second = _record("same payload", run_id=100, archive_member="same.txt")
    shard_a = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [first],
        add_ledgers=True,
    )
    shard_b = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [second],
        add_ledgers=True,
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_a, shard_b],
    )
    expected_request_count = 0
    for shard in (shard_a, shard_b):
        with sqlite3.connect(shard.state) as connection:
            expected_request_count += int(
                connection.execute(
                    "SELECT COUNT(*) FROM request_ledger"
                ).fetchone()[0]
            )

    receipt = merge_shards(manifest)

    output = receipt["store_conservation"]["output_union"]
    overlap = receipt["store_conservation"]["overlap"]
    assert output["unique_content_count"] == 1
    assert output["occurrence_count"] == 1
    assert overlap["unique_contents"] == 1
    assert overlap["occurrences"] == 1
    assert receipt["fetch_state_conservation"]["overlap"]["attempts"] == 1
    assert receipt["fetch_state_conservation"]["overlap"]["members"] == 1
    assert (
        receipt["fetch_state_conservation"]["input_multiplicity"]["requests"]
        == expected_request_count
    )
    assert (
        receipt["fetch_state_conservation"]["output_union"]["requests"]
        == expected_request_count
    )
    assert receipt["fetch_state_conservation"]["overlap"]["bindings"] == 1
    assert destination.is_dir()


def test_convergent_shortcut_and_chain_binding_histories_publish(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard_a = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("first branch", run_id=100, archive_member="a.txt")],
        add_ledgers=True,
    )
    shard_b = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("second branch", run_id=200, archive_member="b.txt")],
        add_ledgers=True,
    )
    _replace_binding_history(
        shard_b.state,
        (("6" * 64, "7" * 64), ("7" * 64, "4" * 64)),
    )
    _write_json(shard_b.fetch_receipt, _fetch_receipt(shard_b, exact_tokenizer))
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_a, shard_b],
    )

    receipt = merge_shards(manifest)

    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        rows = connection.execute(
            """
            SELECT from_sha256,to_sha256
            FROM binding_upgrades
            WHERE binding_key='fetcher_script_sha256'
            ORDER BY id
            """
        ).fetchall()
    assert set(rows) == {
        ("6" * 64, "4" * 64),
        ("6" * 64, "7" * 64),
        ("7" * 64, "4" * 64),
    }
    assert receipt["fetch_state_conservation"]["binding_map_count"] == 3


def test_combined_binding_history_rollback_cycle_publishes_losslessly(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard_a = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("first history", run_id=100, archive_member="a.txt")],
        add_ledgers=True,
    )
    shard_b = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("second history", run_id=200, archive_member="b.txt")],
        add_ledgers=True,
    )
    _replace_binding_history(
        shard_a.state,
        (("6" * 64, "7" * 64), ("7" * 64, "4" * 64)),
    )
    _replace_binding_history(
        shard_b.state,
        (("7" * 64, "6" * 64), ("6" * 64, "4" * 64)),
    )
    _write_json(shard_a.fetch_receipt, _fetch_receipt(shard_a, exact_tokenizer))
    _write_json(shard_b.fetch_receipt, _fetch_receipt(shard_b, exact_tokenizer))
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_a, shard_b],
    )

    receipt = merge_shards(manifest)

    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        rows = connection.execute(
            """
            SELECT from_sha256,to_sha256
            FROM binding_upgrades
            WHERE binding_key='fetcher_script_sha256'
            ORDER BY id
            """
        ).fetchall()
    assert set(rows) == {
        ("6" * 64, "7" * 64),
        ("7" * 64, "6" * 64),
        ("6" * 64, "4" * 64),
        ("7" * 64, "4" * 64),
    }
    assert receipt["fetch_state_conservation"]["binding_map_count"] == 4


def test_single_shard_binding_rollback_to_current_publishes_losslessly(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "rollback",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("rollback history", run_id=100, archive_member="a.txt")],
        add_ledgers=True,
    )
    _replace_binding_history(
        shard.state,
        (("4" * 64, "7" * 64), ("7" * 64, "4" * 64)),
    )
    _write_json(shard.fetch_receipt, _fetch_receipt(shard, exact_tokenizer))
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )

    receipt = merge_shards(manifest)

    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        rows = connection.execute(
            """
            SELECT from_sha256,to_sha256
            FROM binding_upgrades
            WHERE binding_key='fetcher_script_sha256'
            ORDER BY id
            """
        ).fetchall()
    assert rows == [
        ("4" * 64, "7" * 64),
        ("7" * 64, "4" * 64),
    ]
    assert receipt["fetch_state_conservation"]["binding_map_count"] == 2


def test_compatible_binding_prefix_is_canonically_ordered(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard_suffix = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("suffix", run_id=100, archive_member="a.txt")],
        add_ledgers=True,
    )
    shard_full = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("full history", run_id=200, archive_member="b.txt")],
        add_ledgers=True,
    )
    _replace_binding_history(
        shard_suffix.state,
        (("7" * 64, "4" * 64),),
    )
    _replace_binding_history(
        shard_full.state,
        (("6" * 64, "7" * 64), ("7" * 64, "4" * 64)),
    )
    _write_json(
        shard_suffix.fetch_receipt,
        _fetch_receipt(shard_suffix, exact_tokenizer),
    )
    _write_json(
        shard_full.fetch_receipt,
        _fetch_receipt(shard_full, exact_tokenizer),
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_suffix, shard_full],
    )

    receipt = merge_shards(manifest)

    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        rows = connection.execute(
            """
            SELECT id,from_sha256,to_sha256
            FROM binding_upgrades ORDER BY id
            """
        ).fetchall()
    assert rows == [
        (1, "6" * 64, "7" * 64),
        (2, "7" * 64, "4" * 64),
    ]
    assert receipt["fetch_state_conservation"]["overlap"]["bindings"] == 1


def test_fetcher_and_parser_histories_merge_as_independent_canonical_chains(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard_suffix = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("suffix", run_id=100, archive_member="a.txt")],
        add_ledgers=True,
    )
    shard_full = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("full history", run_id=200, archive_member="b.txt")],
        add_ledgers=True,
    )
    with sqlite3.connect(shard_suffix.state) as connection:
        parser_current = str(
            connection.execute(
                """
                SELECT value FROM settings
                WHERE key='parser_script_sha256'
                """
            ).fetchone()[0]
        )

    _replace_binding_history(
        shard_suffix.state,
        (("7" * 64, "4" * 64),),
        upgraded_at="2026-07-26T08:00:00Z",
    )
    _replace_binding_history(
        shard_suffix.state,
        (("b" * 64, parser_current),),
        binding_key="parser_script_sha256",
        clear=False,
        upgraded_at="2026-07-26T08:01:00Z",
    )
    _replace_binding_history(
        shard_full.state,
        (("6" * 64, "7" * 64), ("7" * 64, "4" * 64)),
        upgraded_at="2026-07-26T09:00:00Z",
    )
    _replace_binding_history(
        shard_full.state,
        (("a" * 64, "b" * 64), ("b" * 64, parser_current)),
        binding_key="parser_script_sha256",
        clear=False,
        upgraded_at="2026-07-26T09:01:00Z",
    )
    _write_json(
        shard_suffix.fetch_receipt,
        _fetch_receipt(shard_suffix, exact_tokenizer),
    )
    _write_json(
        shard_full.fetch_receipt,
        _fetch_receipt(shard_full, exact_tokenizer),
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_suffix, shard_full],
    )

    receipt = merge_shards(manifest)

    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        rows = connection.execute(
            """
            SELECT id,binding_key,from_sha256,to_sha256,upgraded_at
            FROM binding_upgrades ORDER BY id
            """
        ).fetchall()
    assert rows == [
        (
            1,
            "fetcher_script_sha256",
            "6" * 64,
            "7" * 64,
            "2026-07-26T09:00:00Z",
        ),
        (
            2,
            "fetcher_script_sha256",
            "7" * 64,
            "4" * 64,
            "2026-07-26T08:00:00Z",
        ),
        (
            3,
            "parser_script_sha256",
            "a" * 64,
            "b" * 64,
            "2026-07-26T09:01:00Z",
        ),
        (
            4,
            "parser_script_sha256",
            "b" * 64,
            parser_current,
            "2026-07-26T08:01:00Z",
        ),
    ]
    conservation = receipt["fetch_state_conservation"]
    assert conservation["overlap"]["bindings"] == 2
    assert conservation["binding_outcomes"]["canonical_overlap"] == 2


def test_conflicting_occurrence_fails_without_publication(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard_a = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("first", run_id=100, archive_member="same.txt")],
    )
    shard_b = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("conflict", run_id=100, archive_member="same.txt")],
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_a, shard_b],
    )

    with pytest.raises(MergeError, match="CAS conflict"):
        merge_shards(manifest)

    assert not destination.exists()


def test_done_attempt_shadows_only_zero_evidence_pending_copy(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard_done = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("done payload", run_id=100, archive_member="done.txt")],
    )
    shard_pending = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("other payload", run_id=200, archive_member="other.txt")],
    )
    _insert_zero_evidence_pending(shard_pending.state, shard_done.state)
    _write_json(
        shard_pending.fetch_receipt,
        _fetch_receipt(shard_pending, exact_tokenizer),
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_done, shard_pending],
    )

    receipt = merge_shards(manifest)

    assert (
        receipt["fetch_state_conservation"]["attempt_outcomes"][
            "pending_shadowed_by_done"
        ]
        == 1
    )
    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        assert connection.execute(
            "SELECT status FROM attempts WHERE run_id=100"
        ).fetchone()[0] == "done"
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2


def test_pending_duplicates_then_done_replay_with_exact_outcomes(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard_done = _build_shard(
        tmp_path / "done",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("done payload", run_id=100, archive_member="done.txt")],
    )
    shard_pending_a = _build_shard(
        tmp_path / "pending-a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("first witness", run_id=200, archive_member="a.txt")],
    )
    shard_pending_b = _build_shard(
        tmp_path / "pending-b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("second witness", run_id=300, archive_member="b.txt")],
    )
    for shard in (shard_pending_a, shard_pending_b):
        _insert_zero_evidence_pending(shard.state, shard_done.state)
        _write_json(shard.fetch_receipt, _fetch_receipt(shard, exact_tokenizer))
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_pending_a, shard_pending_b, shard_done],
    )

    receipt = merge_shards(manifest)

    outcomes = receipt["fetch_state_conservation"]["attempt_outcomes"]
    assert outcomes["exact_overlap"] == 1
    assert outcomes["done_replaced_zero_evidence_pending"] == 1
    assert receipt["fetch_state_conservation"]["overlap"]["attempts"] == 2
    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        assert connection.execute(
            "SELECT status FROM attempts WHERE run_id=100"
        ).fetchone()[0] == "done"


def test_attempt_evidence_precedence_is_pending_then_terminal_then_empty(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    witness = _build_shard(
        tmp_path / "witness",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("witness", run_id=100, archive_member="witness.txt")],
    )
    pending = _build_shard(
        tmp_path / "a-pending",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("shared host", run_id=200, archive_member="host.txt")],
    )
    terminal = _build_shard(
        tmp_path / "b-terminal",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("terminal host", run_id=300, archive_member="terminal.txt")],
    )
    empty = _build_shard(
        tmp_path / "c-empty",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("shared host", run_id=200, archive_member="host.txt")],
    )
    _insert_zero_evidence_pending(pending.state, witness.state)
    _insert_attempt_variant(
        terminal.state,
        witness.state,
        status="terminal_410",
    )
    _insert_attempt_variant(empty.state, witness.state, status="empty")
    for shard in (pending, terminal, empty):
        _write_json(shard.fetch_receipt, _fetch_receipt(shard, exact_tokenizer))
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [pending, terminal, empty],
    )

    receipt = merge_shards(manifest)

    outcomes = receipt["fetch_state_conservation"]["attempt_outcomes"]
    assert outcomes["higher_evidence_replaced"] == 2
    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        assert connection.execute(
            "SELECT status FROM attempts WHERE run_id=100"
        ).fetchone()[0] == "empty"
        mapped = connection.execute(
            "SELECT COUNT(*) FROM attempts"
        ).fetchone()[0]
    assert receipt["fetch_state_conservation"]["attempt_map_count"] == 6
    assert sum(outcomes.values()) == 6
    assert mapped == 3


def test_seed_terminal_attempt_cannot_outrank_verified_coverage() -> None:
    terminal = {"status": "terminal_404"}

    assert _attempt_evidence_rank(terminal, role="seed") == 0
    assert _attempt_evidence_rank(terminal, role="coverage") == 2
    assert _attempt_evidence_rank({"status": "empty"}, role="coverage") == 3
    assert _attempt_evidence_rank({"status": "done"}, role="coverage") == 4


@pytest.mark.parametrize(
    ("order", "expected_promotions"),
    (
        (
            ("seed-terminal", "coverage-terminal", "seed-pending"),
            1,
        ),
        (
            ("seed-terminal", "seed-pending", "coverage-terminal"),
            0,
        ),
        (
            ("coverage-terminal", "seed-terminal", "seed-pending"),
            0,
        ),
        (
            ("coverage-terminal", "seed-pending", "seed-terminal"),
            0,
        ),
        (
            ("seed-pending", "seed-terminal", "coverage-terminal"),
            0,
        ),
        (
            ("seed-pending", "coverage-terminal", "seed-terminal"),
            0,
        ),
    ),
)
def test_exact_terminal_overlap_promotes_coverage_winner_for_all_orders(
    tmp_path: Path,
    order: tuple[str, str, str],
    expected_promotions: int,
) -> None:
    destination = tmp_path / "destination-state.sqlite3"
    destination_connection = sqlite3.connect(destination)
    try:
        destination_connection.executescript(_STATE_SCHEMA)
    finally:
        destination_connection.close()

    immutable = {
        "repo": "owner/repo",
        "run_id": 100,
        "attempt": 1,
        "created_at": "2026-07-27T00:00:00Z",
        "run_metadata_sha256": "1" * 64,
        "run_metadata_raw_size": 2,
        "run_metadata_zlib": zlib.compress(b"{}"),
        "run_metadata_source": "inventory-run-list",
        "run_metadata_source_attempt": 1,
        "run_metadata_exact": 1,
        "inventory_seed_attempt": 1,
        "inventory_seed_metadata_sha256": "2" * 64,
        "discovered_at": "2026-07-27T00:00:00Z",
        "updated_at": "2026-07-27T00:00:01Z",
    }
    terminal = {
        **immutable,
        "status": "terminal_404",
        "tries": 1,
        "archive_source": None,
        "archive_sha256": None,
        "archive_size": None,
        "archive_zlib": None,
        "jobs_sha256": None,
        "jobs_raw_size": None,
        "jobs_zlib": None,
        "member_count": 0,
        "chunk_count": 0,
        "occurrence_tokens": 0,
        "terminal_http_status": 404,
        "terminal_body_sha256": "3" * 64,
        "error_class": None,
        "error_message": None,
    }
    pending = {
        **terminal,
        "status": "pending",
        "tries": 0,
        "terminal_http_status": None,
        "terminal_body_sha256": None,
    }
    sources = {
        "seed-terminal": ("seed", terminal),
        "coverage-terminal": ("coverage", terminal),
        "seed-pending": ("seed", pending),
    }
    journal = sqlite3.connect(":memory:")
    journal.row_factory = sqlite3.Row
    try:
        journal.executescript(_JOURNAL_SQL)
        journal.execute(
            "ATTACH DATABASE ? AS destination",
            (str(destination),),
        )
        for index, name in enumerate(order):
            role, row = sources[name]
            _merge_attempt_row(
                journal,
                f"s{index:02d}",
                role,
                row,  # type: ignore[arg-type]
            )
        winner = journal.execute(
            "SELECT role,evidence_rank FROM attempt_winners"
        ).fetchone()
        final = journal.execute(
            "SELECT status FROM destination.attempts"
        ).fetchone()
        outcomes = {
            str(row["outcome"]): int(row["n"])
            for row in journal.execute(
                """
                SELECT outcome,COUNT(*) AS n
                FROM attempt_map GROUP BY outcome
                """
            )
        }
    finally:
        journal.close()
    assert winner is not None
    assert (winner["role"], winner["evidence_rank"]) == ("coverage", 2)
    assert final is not None and final["status"] == "terminal_404"
    assert outcomes.get(
        "exact_overlap_promoted_higher_precedence",
        0,
    ) == expected_promotions
    assert sum(outcomes.values()) == 3


def test_global_dedup_recomputes_threshold_and_refuses_below_target(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    text = "one globally shared token sequence"
    shard_a = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record(text, run_id=100, archive_member="same.txt")],
    )
    shard_b = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record(text, run_id=100, archive_member="same.txt")],
    )
    exact_tokens = len(exact_tokenizer.encode_batch([text])[0])
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_a, shard_b],
        target=exact_tokens + 1,
    )

    with pytest.raises(MergeError, match="below the requested token target"):
        merge_shards(manifest)

    assert not destination.exists()


def test_store_commit_ahead_of_journal_replays_idempotently(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [
            _record("first batch", run_id=100, archive_member="a.txt"),
            _record("second batch", run_id=200, archive_member="b.txt"),
        ],
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
        occurrences_per_batch=1,
    )

    with pytest.raises(MergePaused):
        merge_shards(manifest, max_batches=1)

    partial = destination.with_name(f".{destination.name}.partial")
    journal_path = partial / "merge_journal.sqlite3"
    with sqlite3.connect(journal_path) as journal:
        journal.execute(
            """
            UPDATE store_progress
            SET cursor_json=NULL,processed_rows=0,batches=0,done=0
            """
        )
        journal.commit()

    unexpected = partial / "unexpected.bin"
    unexpected.write_bytes(b"must never ride into a published bundle")
    with pytest.raises(MergeError, match="unexpected=.*unexpected.bin"):
        merge_shards(manifest)
    assert not destination.exists()
    unexpected.unlink()

    receipt = merge_shards(manifest)

    assert receipt["store_conservation"]["output_union"]["occurrence_count"] == 2
    assert receipt["store_conservation"]["overlap"]["occurrences"] == 0
    assert destination.is_dir()


@pytest.mark.parametrize(
    "fault_point",
    (
        "journal-file-created",
        "journal-schema-created",
        "journal-before-settings-commit",
        "state-file-created",
        "state-schema-created",
        "state-before-settings-commit",
    ),
)
def test_sigkill_during_atomic_database_initialization_resumes(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
    fault_point: str,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("crash-safe payload", run_id=100, archive_member="a.txt")],
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    script = Path(__file__).parents[1] / "scripts/merge_ci_stream_shards.py"

    crashed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(manifest),
            "--fault-inject-sigkill-after",
            fault_point,
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert crashed.returncode == -signal.SIGKILL, crashed.stderr
    assert not destination.exists()
    receipt = merge_shards(manifest)
    assert receipt["status"] == "complete"
    assert destination.is_dir()
    assert not any(".tmp-" in path.name for path in destination.rglob("*"))


def test_concurrent_process_fails_before_touching_partial(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("exclusive payload", run_id=100, archive_member="a.txt")],
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    script = Path(__file__).parents[1] / "scripts/merge_ci_stream_shards.py"
    descriptor = _acquire_merge_lock(destination)
    try:
        loser = subprocess.run(
            [sys.executable, str(script), str(manifest)],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        assert loser.returncode == 1
        assert "another shard union owns the destination lock" in loser.stderr
        assert not destination.exists()
        assert not destination.with_name(f".{destination.name}.partial").exists()
    finally:
        _release_merge_lock(descriptor)

    receipt = merge_shards(manifest)
    assert receipt["status"] == "complete"
    assert destination.is_dir()


def test_source_drift_audit_adds_note_without_replacing_primary_error(
    tmp_path: Path,
) -> None:
    primary = MergeError("primary merge failure")
    missing = tmp_path / "changed.sqlite3"
    audit = SimpleNamespace(
        spec=SimpleNamespace(
            shard_id="changed-source",
            inventory=SimpleNamespace(path=missing),
            state=SimpleNamespace(path=missing),
        )
    )

    _append_source_drift_notes(primary, [cast(Any, audit)])

    assert str(primary) == "primary merge failure"
    assert primary.__notes__
    assert (
        "source drift audit failed for changed-source"
        in primary.__notes__[0]
    )


def test_nested_destination_is_rejected_before_source_store_mutation(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("frozen payload", run_id=100, archive_member="a.txt")],
    )
    manifest, _destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    nested_destination = shard.store / "nested" / "union"
    value["destination"]["bundle_path"] = str(nested_destination)
    manifest.write_bytes(_canonical_json_bytes(value) + b"\n")
    before = frozen_store_artifact_set_sha256(
        shard.store,
        shard.store_receipt,
    )

    with pytest.raises(
        MergeError,
        match="output bundle cannot be nested in a source store",
    ):
        merge_shards(manifest)

    assert not (shard.store / "nested").exists()
    assert (
        frozen_store_artifact_set_sha256(shard.store, shard.store_receipt)
        == before
    )


def test_ready_resume_revalidates_inventory_bytes(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("payload", run_id=100, archive_member="a.txt")],
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    partial = _prepare_ready_partial(manifest, destination)
    (partial / "inventory.sqlite3").write_bytes(b"corrupted inventory")

    with pytest.raises(MergeError, match="inventory differs from its frozen source"):
        merge_shards(manifest)

    assert not destination.exists()


def test_ready_resume_rederives_destination_settings(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("payload", run_id=100, archive_member="a.txt")],
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    partial = _prepare_ready_partial(manifest, destination)
    with sqlite3.connect(partial / "fetch_state.sqlite3") as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            """
            UPDATE settings SET value=?
            WHERE key='parser_script_sha256'
            """,
            ("9" * 64,),
        )
        connection.commit()

    with pytest.raises(
        MergeError,
        match="settings differ from frozen inputs",
    ):
        merge_shards(manifest)

    assert not destination.exists()


def test_ready_resume_replays_resolution_outcomes_exactly(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("payload", run_id=100, archive_member="a.txt")],
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    partial = _prepare_ready_partial(manifest, destination)
    with sqlite3.connect(partial / "merge_journal.sqlite3") as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            "UPDATE attempt_map SET outcome='exact_overlap'"
        )
        connection.commit()

    with pytest.raises(
        MergeError,
        match="resolution maps differ from deterministic replay",
    ):
        merge_shards(manifest)

    assert not destination.exists()


def test_ready_resume_rejects_swapped_identical_request_provenance(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    record = _record("same payload", run_id=100, archive_member="same.txt")
    shard_a = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [record],
        add_ledgers=True,
    )
    shard_b = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [record],
        add_ledgers=True,
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_a, shard_b],
    )
    partial = _prepare_ready_partial(manifest, destination)
    with sqlite3.connect(partial / "merge_journal.sqlite3") as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        swapped = connection.execute(
            """
            SELECT shard_id,source_id,destination_id
            FROM request_id_map
            WHERE source_id=(
              SELECT MIN(source_id) FROM request_id_map
            )
            ORDER BY shard_id
            """
        ).fetchall()
        assert len(swapped) == 2
        temporary_id = int(
            connection.execute(
                "SELECT MAX(destination_id)+1 FROM request_id_map"
            ).fetchone()[0]
        )
        first, second = swapped
        connection.execute(
            """
            UPDATE request_id_map SET destination_id=?
            WHERE shard_id=? AND source_id=?
            """,
            (temporary_id, first[0], first[1]),
        )
        connection.execute(
            """
            UPDATE request_id_map SET destination_id=?
            WHERE shard_id=? AND source_id=?
            """,
            (first[2], second[0], second[1]),
        )
        connection.execute(
            """
            UPDATE request_id_map SET destination_id=?
            WHERE shard_id=? AND source_id=?
            """,
            (second[2], first[0], first[1]),
        )
        connection.commit()

    with pytest.raises(
        MergeError,
        match="resolution maps differ from deterministic replay",
    ):
        merge_shards(manifest)

    assert not destination.exists()


def test_final_tree_rechecks_receipted_artifact_bytes(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("payload", run_id=100, archive_member="a.txt")],
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )
    partial = _prepare_ready_partial(manifest, destination)
    receipt = json.loads(
        (partial / "merge_receipt.json").read_text(encoding="utf-8")
    )
    with (partial / "fetch_receipt.json").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(MergeError, match="size changed"):
        _require_complete_bundle_tree(partial, receipt)

    assert not destination.exists()


def test_relocated_original_paths_are_manifest_bound_without_source_rewrite(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "legion-stage",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("legion payload", run_id=300, archive_member="legion.txt")],
        relocated_prefix="/home/davidgor/frozen-ci",
    )
    relocated_receipt = json.loads(
        shard.fetch_receipt.read_text(encoding="utf-8")
    )
    relocated_receipt["frozen_fetch_state"]["artifact"]["inode"] = 101
    relocated_receipt["frozen_fetch_state"]["artifact"]["mtime_ns"] = 202
    _write_json(shard.fetch_receipt, relocated_receipt)
    before = _sha256(shard.state)
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )

    merge_shards(manifest)

    assert _sha256(shard.state) == before
    with sqlite3.connect(shard.state) as connection:
        settings = dict(connection.execute("SELECT key,value FROM settings"))
    assert settings["inventory_path"] == "/home/davidgor/frozen-ci/inventory.sqlite3"
    assert settings["content_store_path"] == "/home/davidgor/frozen-ci/content-store"
    with sqlite3.connect(destination / "fetch_state.sqlite3") as connection:
        output_settings = dict(connection.execute("SELECT key,value FROM settings"))
    assert output_settings["content_store_path"] == str(
        destination / "content_store"
    )
    assert output_settings["inventory_path"] == str(
        destination / "inventory.sqlite3"
    )


def test_source_fetch_receipt_requires_frozen_state_binding(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("payload", run_id=100, archive_member="a.txt")],
    )
    fetch_receipt = json.loads(shard.fetch_receipt.read_text(encoding="utf-8"))
    del fetch_receipt["frozen_fetch_state"]
    _write_json(shard.fetch_receipt, fetch_receipt)
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )

    with pytest.raises(MergeError, match="frozen_fetch_state must be an object"):
        merge_shards(manifest)

    assert not destination.exists()


def test_cas_bearing_non_done_attempt_fails_closed(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("payload", run_id=100, archive_member="a.txt")],
    )
    connection = sqlite3.connect(shard.state)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("UPDATE attempts SET status='retry'")
        connection.commit()
    finally:
        connection.close()
    store_receipt = json.loads(shard.store_receipt.read_text(encoding="utf-8"))
    fetch_receipt = json.loads(shard.fetch_receipt.read_text(encoding="utf-8"))
    fetch_receipt["content_store_receipt"] = store_receipt
    # The manifest hash is refreshed, but the semantic invalidity must still
    # fail before any destination is published.
    fetch_receipt["fetch_state"]["attempt_statuses"] = {"retry": 1}
    _write_json(shard.fetch_receipt, fetch_receipt)
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard],
    )

    with pytest.raises(
        MergeError,
        match="CAS-bearing non-done|run metadata is not exact",
    ):
        merge_shards(manifest)

    assert not destination.exists()


def test_time_bounded_row_subset_publishes_full_anchor_with_receipted_proof(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    anchor, subset = _anchor_and_time_subset(
        tmp_path,
        exact_tokenizer,
        relocated=True,
    )
    subset_connection = sqlite3.connect(subset.inventory)
    try:
        subset_connection.execute("PRAGMA journal_mode=DELETE")
        subset_connection.execute(
            """
            UPDATE runs SET first_seen_at='2026-07-26T08:00:00Z'
            WHERE run_id=200
            """
        )
        subset_connection.commit()
    finally:
        subset_connection.close()
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [subset, anchor],
    )

    receipt = merge_shards(manifest)

    assert _sha256(destination / "inventory.sqlite3") == _sha256(
        anchor.inventory
    )
    binding = receipt["inventory"]
    assert binding["schema"] == "cppmega_ci_stream_union_inventory_binding_v2"
    assert binding["policy"] == (
        "completed-anchor-with-time-bounded-authoritative-row-subsets-v2"
    )
    assert binding["coverage_semantics"] == (
        "subset_only_no_range_completeness"
    )
    assert binding["time_subset_count"] == 1
    assert binding["time_subset_run_count"] == 1
    sources = {
        item["source_id"]: item for item in binding["sources"]
    }
    assert sources["s00"]["role"] == (
        "authoritative_row_subset_with_observation_audit"
    )
    assert sources["s01"]["role"] == "anchor"
    subset_proof = sources["s00"]["proof"]
    assert subset_proof["matched_run_count"] == 1
    assert subset_proof["anchor_match_semantics"] == (
        "authoritative_github_fields_exact_first_seen_at_audited"
    )
    assert subset_proof["first_seen_at_equal_count"] == 0
    assert subset_proof["first_seen_at_difference_count"] == 1
    assert subset_proof["first_seen_at_subset_earlier_count"] == 1
    assert subset_proof["first_seen_at_subset_later_count"] == 0
    assert binding["time_subset_first_seen_at_difference_count"] == 1
    assert subset_proof["sqlite_schema_sha256"] == (
        "91990153359d65201c18e181b636d4e379443c54f7cbb71b03a0682f652d8f14"
    )
    assert len(subset_proof["anchor_match_logical_sha256"]) == 64
    assert (
        len(
            subset_proof[
                "first_seen_at_observation_pairs_logical_sha256"
            ]
        )
        == 64
    )
    assert receipt["verification"]["inventory_joined_attempts"] == 2
    assert len(receipt["verification"]["inventory_join_sha256"]) == 64
    fetch_receipt = json.loads(
        (destination / "fetch_receipt.json").read_text(encoding="utf-8")
    )
    assert fetch_receipt["inventory_binding"] == binding


def test_time_subset_row_absent_from_anchor_fails_closed(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    anchor, subset = _anchor_and_time_subset(tmp_path, exact_tokenizer)
    connection = sqlite3.connect(subset.inventory)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        columns = ",".join(_INVENTORY_RUN_COLUMNS)
        row = connection.execute(
            f"SELECT {columns} FROM runs WHERE run_id=200"
        ).fetchone()
        assert row is not None
        values = {
            column: row[column] for column in _INVENTORY_RUN_COLUMNS
        }
        values["run_id"] = 999
        connection.execute(
            f"""
            INSERT INTO runs({columns})
            VALUES ({",".join("?" for _ in _INVENTORY_RUN_COLUMNS)})
            """,
            tuple(values[column] for column in _INVENTORY_RUN_COLUMNS),
        )
        connection.execute(
            "UPDATE shard_meta SET value='2' WHERE key='run_count'"
        )
        connection.commit()
    finally:
        connection.close()
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [anchor, subset],
    )

    with pytest.raises(MergeError, match="absent from the anchor"):
        merge_shards(manifest)

    assert not destination.exists()


def test_state_inventory_join_allows_api_exact_created_at_to_differ(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    _anchor, subset = _anchor_and_time_subset(tmp_path, exact_tokenizer)
    state = sqlite3.connect(subset.state)
    state.row_factory = sqlite3.Row
    inventory = sqlite3.connect(subset.inventory)
    inventory.row_factory = sqlite3.Row
    try:
        state.execute("PRAGMA journal_mode=DELETE")
        state.execute(
            """
            UPDATE attempts SET
              run_metadata_source='github-workflow-run-attempt-api',
              created_at='2026-07-26T10:00:01Z'
            """
        )
        state.commit()
        count, digest = _verify_state_inventory_join(
            inventory,
            state,
            label="api-exact-test",
            max_blob_bytes=1024 * 1024,
        )
        assert count == 1
        assert len(digest) == 64
        state.execute(
            """
            UPDATE attempts
            SET inventory_seed_metadata_sha256=?
            """,
            ("0" * 64,),
        )
        state.commit()
        with pytest.raises(MergeError, match="seed binding differs"):
            _verify_state_inventory_join(
                inventory,
                state,
                label="api-exact-test",
                max_blob_bytes=1024 * 1024,
            )
    finally:
        inventory.close()
        state.close()


def test_historical_inventory_seed_requires_verified_continuation_lineage(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    shard = _build_shard(
        tmp_path / "historical-source",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("historical attempt", run_id=100, archive_member="old.txt")],
    )
    rerun_inventory, _rerun_receipt_path, _rerun_receipt = (
        _inventory_with_new_attempt(
            tmp_path,
            inventory,
            run_id=100,
            run_attempt=2,
        )
    )
    state = sqlite3.connect(shard.state)
    state.row_factory = sqlite3.Row
    current_inventory = sqlite3.connect(rerun_inventory)
    current_inventory.row_factory = sqlite3.Row
    try:
        with pytest.raises(
            MergeError,
            match="historical inventory seed lacks verified continuation",
        ):
            _verify_state_inventory_join(
                current_inventory,
                state,
                label="unproven-rerun",
                max_blob_bytes=1024 * 1024,
            )
        count, digest = _verify_state_inventory_join(
            current_inventory,
            state,
            label="verified-rerun",
            max_blob_bytes=1024 * 1024,
            continuation_base_states=(shard.state,),
        )
        assert count == 1
        assert len(digest) == 64
    finally:
        current_inventory.close()
        state.close()


@pytest.mark.parametrize(
    "column",
    ["updated_at", "metadata_blob"],
)
def test_time_subset_authoritative_column_or_blob_mismatch_fails_closed(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
    column: str,
) -> None:
    anchor, subset = _anchor_and_time_subset(tmp_path, exact_tokenizer)
    connection = sqlite3.connect(subset.inventory)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        if column == "metadata_blob":
            blob = bytes(
                connection.execute(
                    "SELECT metadata_blob FROM runs WHERE run_id=200"
                ).fetchone()[0]
            )
            raw = zlib.decompress(blob)
            replacement: object = sqlite3.Binary(zlib.compress(raw, 0))
            assert bytes(replacement) != blob
        else:
            replacement = "2026-07-27T13:00:00Z"
        connection.execute(
            f"UPDATE runs SET {column}=? WHERE run_id=200",
            (replacement,),
        )
        connection.commit()
    finally:
        connection.close()
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [anchor, subset],
    )

    with pytest.raises(MergeError, match=rf"column {column} differs"):
        merge_shards(manifest)

    assert not destination.exists()


def test_time_subset_noncanonical_first_seen_at_fails_closed(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    anchor, subset = _anchor_and_time_subset(tmp_path, exact_tokenizer)
    connection = sqlite3.connect(subset.inventory)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            "UPDATE runs SET first_seen_at='not-a-time' WHERE run_id=200"
        )
        connection.commit()
    finally:
        connection.close()
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [anchor, subset],
    )

    with pytest.raises(
        MergeError,
        match=r"time-shard .* first_seen_at is not a valid UTC instant",
    ):
        merge_shards(manifest)

    assert not destination.exists()


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("bounds", "bounds are empty or reversed"),
        ("lower-escape", "bounds escape the anchor interval"),
        ("upper-escape", "bounds escape the anchor interval"),
        ("count", "run_count differs"),
        ("path", "no completed anchor binds"),
        ("meta-schema", "metadata schema is not v1"),
        ("sqlite-schema", "inventory schema is not exact v1"),
    ],
)
def test_time_subset_bad_bounds_count_path_or_schema_fails_closed(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
    mutation: str,
    error: str,
) -> None:
    anchor, subset = _anchor_and_time_subset(tmp_path, exact_tokenizer)
    if mutation == "bounds":
        _update_time_subset_meta(
            subset,
            "created_at_gte",
            "2026-08-01T00:00:00Z",
        )
    elif mutation == "lower-escape":
        _update_time_subset_meta(
            subset,
            "created_at_gte",
            "2026-05-31T23:59:59Z",
        )
    elif mutation == "upper-escape":
        _update_time_subset_meta(
            subset,
            "created_at_lt",
            "2026-08-01T00:00:01Z",
        )
    elif mutation == "count":
        _update_time_subset_meta(subset, "run_count", "2")
    elif mutation == "path":
        _update_time_subset_meta(
            subset,
            "source_inventory_path",
            "/frozen/wrong-anchor.sqlite3",
        )
    elif mutation == "meta-schema":
        _update_time_subset_meta(subset, "schema", "invented_schema_v9")
    else:
        connection = sqlite3.connect(subset.inventory)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("CREATE TABLE unexpected(value TEXT)")
            connection.commit()
        finally:
            connection.close()
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [anchor, subset],
    )

    with pytest.raises(MergeError, match=error):
        merge_shards(manifest)

    assert not destination.exists()


def test_time_subset_without_completed_anchor_is_role_ambiguous(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    _anchor, subset = _anchor_and_time_subset(tmp_path, exact_tokenizer)
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [subset],
    )

    with pytest.raises(MergeError, match="completed full inventory anchor"):
        merge_shards(manifest)

    assert not destination.exists()


def test_multiple_distinct_completed_anchors_fail_closed(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    second_inventory = tmp_path / "inventory-template-second.sqlite3"
    shutil.copyfile(inventory, second_inventory)
    connection = sqlite3.connect(second_inventory)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA user_version=1")
        connection.commit()
    finally:
        connection.close()
    second_inventory_receipt = InventoryDB(
        second_inventory
    ).completion_receipt()
    assert _sha256(second_inventory) != _sha256(inventory)
    anchor_a = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("a", run_id=100, archive_member="a.txt")],
    )
    anchor_b = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        second_inventory,
        second_inventory_receipt,
        [_record("b", run_id=200, archive_member="b.txt")],
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [anchor_a, anchor_b],
    )

    with pytest.raises(MergeError, match="multiple distinct completed"):
        merge_shards(manifest)

    assert not destination.exists()


def test_byte_identical_anchor_and_shared_receipt_preserve_identical_mode(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    inventory, inventory_receipt = _empty_inventory_template(tmp_path)
    anchor = _build_shard(
        tmp_path / "a",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("a", run_id=100, archive_member="a.txt")],
    )
    second = _build_shard(
        tmp_path / "b",
        exact_tokenizer,
        inventory,
        inventory_receipt,
        [_record("b", run_id=200, archive_member="b.txt")],
    )
    _set_original_bindings(
        second.state,
        inventory_path=anchor.original_inventory,
        store_path=second.original_store,
    )
    shared = replace(
        second,
        inventory=anchor.inventory,
        inventory_receipt=anchor.inventory_receipt,
        original_inventory=anchor.original_inventory,
    )
    _write_json(
        shared.fetch_receipt,
        _fetch_receipt(shared, exact_tokenizer),
    )
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [anchor, shared],
    )

    receipt = merge_shards(manifest)

    roles = [
        source["role"] for source in receipt["inventory"]["sources"]
    ]
    assert roles == ["anchor", "byte_identical_anchor_alias"]
    assert _sha256(destination / "inventory.sqlite3") == _sha256(
        anchor.inventory
    )


def test_ready_resume_rejects_time_subset_first_seen_at_tamper(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    anchor, subset = _anchor_and_time_subset(tmp_path, exact_tokenizer)
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [anchor, subset],
    )
    partial = _prepare_ready_partial(manifest, destination)
    connection = sqlite3.connect(subset.inventory)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            """
            UPDATE runs SET first_seen_at='2026-07-27T13:00:00Z'
            WHERE run_id=200
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(MergeError, match="inventory hash differs"):
        merge_shards(manifest)

    assert partial.exists()
    assert not destination.exists()


def test_ready_resume_rejects_destination_inventory_seed_tamper(
    tmp_path: Path,
    exact_tokenizer: ExactTokenizer,
) -> None:
    anchor, subset = _anchor_and_time_subset(tmp_path, exact_tokenizer)
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [subset, anchor],
    )
    partial = _prepare_ready_partial(manifest, destination)
    connection = sqlite3.connect(partial / "fetch_state.sqlite3")
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            """
            UPDATE attempts
            SET inventory_seed_metadata_sha256=?
            WHERE run_id=200
            """,
            ("0" * 64,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(MergeError):
        merge_shards(manifest)

    assert partial.exists()
    assert not destination.exists()


def test_genuine_inventory_fetch_merge_export_source_and_macro_e2e(
    tmp_path: Path,
) -> None:
    """Exercise the production path without hand-promoting synthetic rows."""

    git_root = tmp_path / "git"
    git_root.mkdir()
    mirror, head_sha, _base_sha = _git_fixture(git_root)
    expected_source = b"int main() { return 0; }\n"

    repo_list = tmp_path / "repositories.json"
    repo_list.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "repo_names": ["Owner/Repo"],
                "repos": [],
                "unresolved": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    run = _inventory_run(
        901,
        ci_inventory.parse_utc_instant(INVENTORY_START) + 1,
    )
    run["head_sha"] = head_sha
    run["head_commit"] = {
        "id": head_sha,
        "message": "compile source",
        "author": {"name": "CI Builder"},
        "committer": {"name": "CI Builder"},
    }
    run["path"] = ".github/workflows/ci.yml"
    run["display_title"] = "Compile source"
    run["repository"] = {"full_name": "Owner/Repo", "id": 1}
    run["head_repository"] = {"full_name": "Owner/Repo", "id": 1}
    inventory_api = DatasetAPI([run])
    inventory_path = tmp_path / "inventory.sqlite3"
    inventory_receipt_path = tmp_path / "inventory_receipt.json"
    inventory = ci_inventory.GitHubActionsInventory(
        db_path=inventory_path,
        scope=ci_inventory.load_repo_scope(repo_list),
        start=INVENTORY_START,
        end=INVENTORY_END,
        tokens=["inventory-token"],
        progress_path=tmp_path / "inventory.progress.json",
        requester=inventory_api,
        sleeper=lambda _seconds: None,
    )
    inventory.run()
    inventory_receipt = inventory.write_completion_receipt(
        inventory_receipt_path
    )
    assert inventory_receipt["production_complete"] is True
    assert inventory_receipt["expected_attempt_count"] == 1

    raw_log = (
        b"Working directory is "
        b"'/home/runner/work/workspace/checkout/build'\n"
        b"[command]clang++ -I../include -std=c++20 "
        b"-c ../src/nested/main.cpp -o main.o\n"
    )
    github = FakeGitHub(_zip_bytes({"0_99.txt": raw_log}))
    fetch_root = tmp_path / "fetch"
    fetch_root.mkdir()
    state_path = fetch_root / "fetch_state.sqlite3"
    store_root = fetch_root / "content_store"
    fetch_receipt_path = fetch_root / "fetch_receipt.json"
    store_receipt_path = fetch_root / "store_receipt.json"
    fetcher = CIStreamFetcher(
        inventory_path=inventory_path,
        inventory_receipt_path=inventory_receipt_path,
        state_path=state_path,
        content_store_path=store_root,
        tokenizer_path=TOKENIZER_JSON,
        tokens=["fetch-token"],
        progress_path=fetch_root / "progress.json",
        receipt_path=fetch_receipt_path,
        completion_mode=COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        target_unique_tokens=1,
        parser_workers=0,
        requester=github.request,
        archive_downloader=github.download,
        sleeper=lambda _seconds: None,
    )
    try:
        fetcher.run(continuous=False, workers=1)
        assert fetcher.exhaustive_completion_ready() is True
    finally:
        fetcher.close()
    fetched = finalize_fetch_receipts(
        state_path=state_path,
        content_store_path=store_root,
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=1,
        fetch_receipt_path=fetch_receipt_path,
        store_receipt_path=store_receipt_path,
        original_state_path=state_path,
        original_content_store_path=store_root,
        original_inventory_path=inventory_path,
        completion_mode=COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        inventory_receipt_path=inventory_receipt_path,
    )
    assert fetched["production_complete"] is True

    manifest_path = tmp_path / "production_manifest.json"
    union_root = tmp_path / "union"
    build_canonical_manifest(
        manifest_path,
        destination=union_root,
        tokenizer_path=TOKENIZER_JSON,
        target_unique_tokens=1,
        shards=[
            {
                "id": "genuine",
                "role": "coverage",
                "inventory": inventory_path,
                "inventory_receipt": inventory_receipt_path,
                "content_store": store_root,
                "store_receipt": store_receipt_path,
                "fetch_state": state_path,
                "fetch_receipt": fetch_receipt_path,
            }
        ],
    )
    merged = merge_shards(manifest_path)
    assert merged["production_complete"] is True

    case5_root = tmp_path / "case5"
    exported = export_store(
        store_root=union_root / "content_store",
        store_receipt=union_root / "store_receipt.json",
        fetch_state=union_root / "fetch_state.sqlite3",
        tokenizer_json=TOKENIZER_JSON,
        output=case5_root,
        completion_mode=COMPLETION_MODE_INVENTORY_EXHAUSTIVE,
        inventory=union_root / "inventory.sqlite3",
        inventory_receipt=union_root / "inventory_receipt.json",
        fetch_receipt=union_root / "fetch_receipt.json",
        merge_receipt=union_root / "merge_receipt.json",
    )
    assert exported["production_complete"] is True

    binding_inventory_path = tmp_path / "source_binding_inventory.jsonl"
    representative_ledger = (
        case5_root / str(exported["representatives"]["ledger_artifact"])
    )
    extracted = extract_binding_inventory(
        union_root / "content_store",
        union_root / "fetch_receipt.json",
        content_store_receipt_path=union_root / "store_receipt.json",
        case5_export_receipt_path=case5_root / "export_receipt.json",
        representative_ledger_path=representative_ledger,
        output_path=binding_inventory_path,
    )
    assert extracted["status"] == "complete"
    frozen_inventory = verify_binding_inventory(binding_inventory_path)
    binding_records = [
        json.loads(line)
        for line in binding_inventory_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    binding = next(
        record
        for record in binding_records
        if record.get("record_type") == "binding"
    )
    canonical_repository = "Owner/Repo"
    assert binding["repository"] == canonical_repository
    assert binding["head_sha"] == head_sha
    assert binding["source_path"] == "src/nested/main.cpp"

    source_store_root = tmp_path / "source_store"
    source_receipt_path = tmp_path / "source_store_receipt.json"
    source_ledger_path = tmp_path / "source_reference_ledger.jsonl"
    source_receipt = materialize_inventory(
        binding_inventory_path,
        {canonical_repository: mirror},
        source_store_root,
        receipt_path=source_receipt_path,
        ledger_path=source_ledger_path,
    )
    assert source_receipt["status"] == "complete"
    assert source_receipt["missing_binding_count"] == 0
    assert (
        source_receipt["input_inventory_artifact_sha256"]
        == frozen_inventory.artifact_sha256
    )
    with SourceSidecarStore(source_store_root) as source_store:
        content_row = source_store._connection.execute(
            """
            SELECT content_sha256 FROM bindings
            WHERE repository=? AND head_sha=? AND source_path=?
            """,
            (
                canonical_repository,
                head_sha,
                "src/nested/main.cpp",
            ),
        ).fetchone()
        assert content_row is not None
        assert source_store.read_blob(str(content_row["content_sha256"])) == (
            expected_source
        )

    buckets = tuple(
        sorted(
            {
                int(artifact["bucket"])
                for artifact in exported["artifacts"]
                if artifact["kind"] == "case5_parquet"
            }
        )
    )
    allowed, normalized = _load_ci_manifest_allowlist(
        case5_root / "export_receipt.json",
        case5_root,
        buckets,
        cppmega_mlx_commit="unused-for-content-store-export",
        cppmega_mlx_tree_sha256="unused-for-content-store-export",
    )
    assert normalized["schema"] == exported["schema"]
    assert sum(len(files) for files in allowed.values()) >= 1
