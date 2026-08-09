from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tools.clang_indexer.dedup_store import DedupStore


def test_discard_local_stage_removes_malformed_stage_without_touching_global(
    tmp_path: Path,
) -> None:
    global_db = tmp_path / "global.sqlite"
    global_contents = b"global database must remain untouched"
    global_db.write_bytes(global_contents)
    stage_db = tmp_path / "failed-stage.sqlite"
    stage_db.write_bytes(b"not a sqlite database")
    for suffix in ("-journal", "-wal", "-shm"):
        Path(f"{stage_db}{suffix}").write_bytes(b"discardable sidecar")

    DedupStore.discard_stage(
        str(global_db),
        "code:timed-out-repository",
        stage_db_path=str(stage_db),
    )

    assert global_db.read_bytes() == global_contents
    for suffix in ("", "-journal", "-wal", "-shm"):
        assert not Path(f"{stage_db}{suffix}").exists()


def test_discard_current_local_stage_closes_connections_before_unlink(
    tmp_path: Path,
) -> None:
    global_db = tmp_path / "global.sqlite"
    DedupStore(str(global_db), near=False).close()
    stage_db = tmp_path / "active-stage.sqlite"
    store = DedupStore(
        str(global_db),
        near=False,
        stage_id="code:active-repository",
        stage_db_path=str(stage_db),
    )
    store.stage_conn.execute(
        "INSERT INTO exact_stage(stage_id, hash) VALUES (?, ?)",
        ("code:active-repository", b"claim"),
    )
    store.stage_conn.commit()

    store.discard_current_stage()

    with sqlite3.connect(global_db) as global_connection:
        assert global_connection.execute(
            "SELECT COUNT(*) FROM exact"
        ).fetchone() == (0,)
    with pytest.raises(sqlite3.ProgrammingError):
        store.stage_conn.execute("SELECT 1")
    store.close()
    assert not stage_db.exists()
