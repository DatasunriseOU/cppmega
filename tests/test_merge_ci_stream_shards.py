from __future__ import annotations

from dataclasses import dataclass
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
    _acquire_merge_lock,
    _canonical_json_bytes,
    _release_merge_lock,
    _require_complete_bundle_tree,
    frozen_store_artifact_set_sha256,
    merge_shards,
)
from tests.test_export_ci_content_store_case5 import (
    TOKENIZER_JSON,
    _build_store,
    _provenance,
)


@dataclass(frozen=True)
class BuiltShard:
    root: Path
    inventory: Path
    inventory_receipt: Path
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
        metadata = {
            "schema": INVENTORY_SCHEMA,
            "repo_list_path": "/frozen/repositories.json",
            "repo_list_sha256": "1" * 64,
            "repo_scope_sha256": _hash_lines(()),
            "repo_count": "0",
            "original_repo_count": "0",
            "unresolved_count": "0",
            "start_epoch": "1",
            "end_epoch": "2",
            "start_utc": "1970-01-01T00:00:01Z",
            "end_utc": "1970-01-01T00:00:02Z",
            "script_sha256": "2" * 64,
            "metadata_encoding": METADATA_ENCODING,
            "smoke": "0",
            "max_repos": "",
        }
        connection.executemany(
            "INSERT INTO inventory_meta(key,value) VALUES (?,?)",
            sorted(metadata.items()),
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
                    "inventory": {
                        "path": str(shard.inventory),
                        "sha256": _sha256(shard.inventory),
                        "receipt": {
                            "path": str(shard.inventory_receipt),
                            "sha256": _sha256(shard.inventory_receipt),
                        },
                    },
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
