from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from scripts import commit_source_conveyor_supervisor as commit_supervisor
from scripts import source_conveyor_supervisor as source_supervisor
from scripts import streaming_conveyor as conveyor
from scripts.pr_ingest import pr_store


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_list(path: Path, *, project: str = "owner/project") -> Path:
    bare_name = project.rsplit("/", 1)[-1]
    _write_json(
        path,
        {
            "schema_version": 2,
            "repos": [
                {
                    "bare_name": bare_name,
                    "project_identity": project,
                    "owner_repo": project,
                }
            ],
            "by_bare_name": {bare_name: project},
            "project_identities": [project],
            "repo_names": [project],
            "unresolved": [],
        },
    )
    return path


def _pr_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_list = _repo_list(tmp_path / "source_repos.json")
    pr_list = _repo_list(tmp_path / "pr_repos.json")
    store = tmp_path / "prs.sqlite"
    connection = pr_store.connect(str(store), create=True)
    try:
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    completion = tmp_path / "pr_completion.json"
    _write_json(
        completion,
        {
            "schema": "cppmega_pr_completion_v2",
            "status": "verified",
            "repo_list": {
                "path": str(pr_list),
                "sha256": _sha256(pr_list),
            },
            "pr_store": {
                "path": str(store),
                "sha256": _sha256(store),
                "size": store.stat().st_size,
            },
            "expected_repos_sha256": source_supervisor._canonical_sha256(
                ["owner/project"]
            ),
            "scan_id": "7" * 64,
            "expected_repo_count": 1,
            "stored_pr_count": 0,
            "declared_pr_count": 0,
            "unverified_store_pr_count": 0,
        },
    )
    return source_list, store, pr_list, completion


def test_commit_pr_inputs_reject_store_mutation_and_wal(tmp_path: Path) -> None:
    source_list, store, pr_list, completion = _pr_fixture(tmp_path)
    with store.open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(conveyor.PRCompletionBindingError, match="hash mismatch"):
        commit_supervisor.validate_pr_inputs(
            source_repo_list=source_list,
            pr_store=store,
            pr_repo_list=pr_list,
            completion_receipt=completion,
        )

    source_list, store, pr_list, completion = _pr_fixture(tmp_path / "wal")
    Path(f"{store}-wal").write_bytes(b"uncheckpointed")
    with pytest.raises(conveyor.PRCompletionBindingError, match="uncheckpointed WAL"):
        commit_supervisor.validate_pr_inputs(
            source_repo_list=source_list,
            pr_store=store,
            pr_repo_list=pr_list,
            completion_receipt=completion,
        )


def test_commit_pr_inputs_reject_scope_and_finish_drift(tmp_path: Path) -> None:
    source_list, store, pr_list, completion = _pr_fixture(tmp_path)
    _repo_list(source_list, project="owner/other")
    with pytest.raises(conveyor.RepoListBindingError, match="does not match"):
        commit_supervisor.validate_pr_inputs(
            source_repo_list=source_list,
            pr_store=store,
            pr_repo_list=pr_list,
            completion_receipt=completion,
        )

    source_list, store, pr_list, completion = _pr_fixture(tmp_path / "finish")
    pr_inputs, snapshot, paths = commit_supervisor.validate_pr_inputs(
        source_repo_list=source_list,
        pr_store=store,
        pr_repo_list=pr_list,
        completion_receipt=completion,
    )
    _repo_list(pr_list, project="owner/drifted")
    with pytest.raises(conveyor.PRCompletionBindingError, match="hash mismatch"):
        conveyor.revalidate_pr_completion_binding(
            pr_inputs["completion_binding"],
            paths[2],
            pr_store=paths[0],
            repo_list=paths[1],
            repo_list_snapshot=snapshot,
        )


def test_commit_run_root_collision_and_nonzero_child_fail_closed(
    tmp_path: Path,
) -> None:
    collision_root = tmp_path / "collision"
    collision_root.mkdir()
    (collision_root / "old-output").write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dedicated new run root"):
        source_supervisor._resume_attempt(
            collision_root / "launch_receipt.json",
            run_binding={
                "schema": source_supervisor.RUN_BINDING_SCHEMA,
                "streams": "commits",
            },
        )

    run_root = tmp_path / "nonzero"
    run_root.mkdir()
    launch = run_root / "launch_receipt.json"
    source_supervisor._atomic_json(
        launch,
        {
            "schema": source_supervisor.LAUNCH_SCHEMA,
            "status": "running",
        },
    )
    started: list[int] = []
    return_code = source_supervisor.run_supervised_child(
        [sys.executable, "-c", "raise SystemExit(7)"],
        cwd=tmp_path,
        environment=os.environ.copy(),
        log_path=run_root / "run.log",
        on_started=started.append,
    )
    receipt = source_supervisor.write_exit_receipt(
        run_root / "exit_receipt.json",
        launch_path=launch,
        code_revision="8" * 40,
        return_code=return_code,
        manifest_path=run_root / "conveyor" / "_done.json",
        completion_path=run_root / "conveyor" / "completion_receipt.json",
    )

    assert started and return_code == 7
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 7
    assert receipt["done_manifest"] is None
    assert receipt["completion_receipt"] is None


def test_shared_child_stops_process_when_started_receipt_fails(
    tmp_path: Path,
) -> None:
    started: list[int] = []

    def reject_started(pid: int) -> None:
        started.append(pid)
        raise RuntimeError("receipt write failed")

    with pytest.raises(RuntimeError, match="receipt write failed"):
        source_supervisor.run_supervised_child(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            log_path=tmp_path / "child.log",
            on_started=reject_started,
        )

    assert started
    with pytest.raises(ProcessLookupError):
        os.kill(started[0], 0)
