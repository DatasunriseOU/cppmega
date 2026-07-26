from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import signal
import shutil
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast
import zlib

import pytest

from scripts.ci_stream_fetch import (
    RECEIPT_SCHEMA as FETCH_RECEIPT_SCHEMA,
    ExactTokenizer,
)
from scripts.ci_stream_inventory import (
    METADATA_ENCODING,
    SCHEMA_VERSION as INVENTORY_SCHEMA,
    _SCHEMA_SQL as INVENTORY_SQL,
    _hash_lines,
    InventoryDB,
)
from scripts.export_ci_content_store_case5 import FrozenFetchState, FrozenStore
from scripts.merge_ci_stream_shards import (
    MANIFEST_SCHEMA,
    MergeError,
    MergePaused,
    TIME_SHARD_INVENTORY_SCHEMA,
    _INVENTORY_RUN_COLUMNS,
    _TIME_SHARD_SQL,
    _acquire_merge_lock,
    _canonical_json_bytes,
    _release_merge_lock,
    _require_complete_bundle_tree,
    _verify_state_inventory_join,
    frozen_store_artifact_set_sha256,
    merge_shards,
)
from tests.test_export_ci_content_store_case5 import (
    TOKENIZER_JSON,
    _build_store,
    _provenance,
    _run_metadata,
)


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


def _empty_inventory_template(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
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
            f"owner/repo\t{run_id}\t1\t{metadata_sha256}"
            for run_id, _raw, metadata_sha256, _metadata in run_rows
        )
        metadata = {
            "schema": INVENTORY_SCHEMA,
            "repo_list_path": "/frozen/repositories.json",
            "repo_list_sha256": "1" * 64,
            "repo_scope_sha256": _hash_lines(("owner/repo",)),
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
        }
        connection.executemany(
            "INSERT INTO inventory_meta(key,value) VALUES (?,?)",
            sorted(metadata.items()),
        )
        connection.execute(
            """
            INSERT INTO repos(repo_key,owner,name,canonical,ordinal)
            VALUES ('owner/repo','owner','repo','owner/repo',0)
            """
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
                  'owner/repo',1780272000,1785542400,NULL,0,'done',
                  3,1,1,3,3,0,?,
                  '2026-07-26T09:00:00Z','2026-07-26T09:00:00Z'
                )
                """,
                (run_keys_sha256,),
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
                    "owner/repo",
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
                (window_id, "owner/repo", run_id, 1, metadata_sha256),
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
        connection.commit()
    finally:
        connection.close()
    receipt = InventoryDB(path).completion_receipt()
    for suffix in ("-wal", "-shm", "-journal"):
        assert not Path(f"{path}{suffix}").exists()
    return path, receipt


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
) -> None:
    connection = sqlite3.connect(state_path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("DELETE FROM binding_upgrades")
        for source, destination in transitions:
            connection.execute(
                """
                INSERT INTO binding_upgrades(
                  binding_key,from_sha256,to_sha256,reason,upgraded_at
                ) VALUES ('fetcher_script_sha256',?,?,?,?)
                """,
                (
                    source,
                    destination,
                    f"fixture {source[:8]} to {destination[:8]}",
                    "2026-07-26T08:00:00Z",
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
        assert connection.execute("SELECT COUNT(*) FROM request_ledger").fetchone()[0] == 2
        assert [
            row[0]
            for row in connection.execute("SELECT id FROM request_ledger ORDER BY id")
        ] == [1, 2]
        assert connection.execute("SELECT COUNT(*) FROM binding_upgrades").fetchone()[0] == 1
    request_map = [
        json.loads(line)
        for line in (destination / "ledgers/request_id_map.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["destination_id"] for record in request_map] == [1, 2]
    assert [record["source_id"] for record in request_map] == [1, 1]

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

    receipt = merge_shards(manifest)

    output = receipt["store_conservation"]["output_union"]
    overlap = receipt["store_conservation"]["overlap"]
    assert output["unique_content_count"] == 1
    assert output["occurrence_count"] == 1
    assert overlap["unique_contents"] == 1
    assert overlap["occurrences"] == 1
    assert receipt["fetch_state_conservation"]["overlap"]["attempts"] == 1
    assert receipt["fetch_state_conservation"]["overlap"]["members"] == 1
    assert receipt["fetch_state_conservation"]["input_multiplicity"]["requests"] == 2
    assert receipt["fetch_state_conservation"]["output_union"]["requests"] == 2
    assert receipt["fetch_state_conservation"]["overlap"]["bindings"] == 1
    assert destination.is_dir()


def test_divergent_binding_upgrade_branch_fails_without_publication(
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

    with pytest.raises(MergeError, match="binding-upgrade branch"):
        merge_shards(manifest)

    assert not destination.exists()


def test_convergent_binding_histories_fail_without_publication(
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
        shard_b.state,
        (("8" * 64, "4" * 64),),
    )
    _write_json(shard_b.fetch_receipt, _fetch_receipt(shard_b, exact_tokenizer))
    manifest, destination = _manifest(
        tmp_path,
        exact_tokenizer,
        [shard_a, shard_b],
    )

    with pytest.raises(MergeError, match="not one linear chain"):
        merge_shards(manifest)

    assert not destination.exists()


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
        connection.execute(
            "UPDATE request_id_map SET destination_id=destination_id+10"
        )
        connection.execute(
            """
            UPDATE request_id_map
            SET destination_id=CASE shard_id
              WHEN 's00' THEN 2
              WHEN 's01' THEN 1
            END
            """
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
    assert binding["policy"] == (
        "completed-anchor-with-time-bounded-row-subsets-v1"
    )
    assert binding["coverage_semantics"] == (
        "subset_only_no_range_completeness"
    )
    assert binding["time_subset_count"] == 1
    assert binding["time_subset_run_count"] == 1
    sources = {
        item["source_id"]: item for item in binding["sources"]
    }
    assert sources["s00"]["role"] == "byte_identical_row_subset"
    assert sources["s01"]["role"] == "anchor"
    subset_proof = sources["s00"]["proof"]
    assert subset_proof["matched_run_count"] == 1
    assert subset_proof["sqlite_schema_sha256"] == (
        "91990153359d65201c18e181b636d4e379443c54f7cbb71b03a0682f652d8f14"
    )
    assert len(subset_proof["anchor_match_logical_sha256"]) == 64
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


@pytest.mark.parametrize(
    "column",
    ["updated_at", "first_seen_at", "metadata_blob"],
)
def test_time_subset_any_column_or_blob_mismatch_fails_closed(
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
