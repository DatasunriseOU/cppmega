from __future__ import annotations

from pathlib import Path

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
