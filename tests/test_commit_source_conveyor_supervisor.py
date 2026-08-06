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
from tests.test_source_conveyor_supervisor import _input_fixture


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


def test_commit_accepts_failed_base_plus_repair_and_plans_all_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, argv = _input_fixture(tmp_path)
    base_args = source_supervisor.parse_args(argv)
    base_inputs = source_supervisor.validate_inputs(base_args, repo_root=repo)
    base_root = Path(base_args.run_root)
    (base_root / "reindexed").mkdir()
    (base_root / "reindexed-commits").mkdir()
    (base_root / "dedup.sqlite").write_bytes(b"dedup")
    base_command = source_supervisor.build_command(base_args, base_inputs)
    base_binding = source_supervisor.build_run_binding(base_args, base_inputs)
    base_launch = source_supervisor.build_launch_receipt(
        base_args,
        inputs=base_inputs,
        command=base_command,
        run_binding=base_binding,
        attempt=1,
    )
    base_launch["status"] = "running"
    base_launch["repository_root"] = str(repo)
    base_launch_path = base_root / "launch_receipt.json"
    source_supervisor._atomic_json(base_launch_path, base_launch)
    base_manifest = base_root / "conveyor" / "_done.json"
    source_supervisor._atomic_json(
        base_manifest,
        {
            "code_revision": base_inputs["code_revision"],
            "done": {},
            "failed": {"project::code": {"stage": "index_project"}},
        },
    )
    source_supervisor.write_exit_receipt(
        base_root / "exit_receipt.json",
        launch_path=base_launch_path,
        code_revision=base_args.expected_code_revision,
        return_code=1,
        manifest_path=base_manifest,
        completion_path=base_root / "conveyor" / "completion_receipt.json",
    )

    repair_root = tmp_path / "repair"
    repair_root.mkdir()
    repair_argv = list(argv)
    repair_argv[repair_argv.index("--run-root") + 1] = str(repair_root)
    repair_argv.extend(
        [
            "--repair-base-code-run-root",
            str(base_root),
            "--only-repo",
            "project",
            "--code-index-timeout-s",
            "86400",
            "--code-index-stall-timeout-s",
            "1800",
        ]
    )
    repair_args = source_supervisor.parse_args(repair_argv)
    repair_inputs = source_supervisor.validate_inputs(repair_args, repo_root=repo)
    repair_base = source_supervisor.load_repair_base_code_run(base_root)
    repair_command = source_supervisor.build_command(
        repair_args,
        repair_inputs,
        repair_base,
    )
    repair_binding = source_supervisor.build_run_binding(
        repair_args,
        repair_inputs,
        repair_base,
    )
    repair_launch = source_supervisor.build_launch_receipt(
        repair_args,
        inputs=repair_inputs,
        command=repair_command,
        run_binding=repair_binding,
        attempt=1,
        repair_base=repair_base,
    )
    repair_launch["status"] = "running"
    repair_launch["repository_root"] = str(repo)
    repair_launch_path = repair_root / "launch_receipt.json"
    source_supervisor._atomic_json(repair_launch_path, repair_launch)
    repair_manifest = repair_root / "conveyor" / "_done.json"
    source_supervisor._atomic_json(
        repair_manifest,
        {
            "code_revision": repair_inputs["code_revision"],
            "done": {
                "project::code": {
                    "artifact_filename": "project.parquet",
                    "lengths": {
                        str(length): {"rows": 1}
                        for length in repair_args.target_lengths
                    },
                }
            },
            "failed": {},
        },
    )
    completion_path = repair_root / "conveyor" / "completion_receipt.json"
    source_supervisor._atomic_json(
        completion_path,
        {
            "schema": source_supervisor.COMPLETION_SCHEMA,
            "status": "success",
            "streams": "code",
            "interrupted": False,
            "manifest": {
                "path": str(repair_manifest.resolve()),
                "size_bytes": repair_manifest.stat().st_size,
                "sha256": _sha256(repair_manifest),
            },
            "total_done_unit_count": 1,
            "failed_unit_count": 0,
            "non_code_done_unit_count": 0,
            "code_repositories": ["project"],
            "code_repository_names_sha256": source_supervisor._canonical_sha256(
                ["project"]
            ),
            "code_revision": repair_inputs["code_revision"],
            "source_repo_list": {
                "schema": "cppmega_source_repo_list_binding_v1",
                "sha256": repair_inputs["repo_list"]["sha256"],
                "canonical_mapping_sha256": repair_inputs["repo_list"][
                    "canonical_mapping_sha256"
                ],
                "mapping_count": repair_inputs["repo_list"]["mapping_entry_count"],
            },
            "source_repo_list_reverified_at_finish": True,
        },
    )
    source_supervisor.write_exit_receipt(
        repair_root / "exit_receipt.json",
        launch_path=repair_launch_path,
        code_revision=repair_args.expected_code_revision,
        return_code=0,
        manifest_path=repair_manifest,
        completion_path=completion_path,
        terminal_coverage={
            "status": "complete",
            "expected_repository_count": 1,
            "successful_repository_count": 1,
            "immutable_inputs_reverified_at_finish": True,
        },
        schema=source_supervisor.TARGETED_EXIT_SCHEMA,
        selected_repositories=["project"],
        repair_base_code_run=repair_base["identity"],
    )

    revalidated_run_roots: list[Path] = []
    original_revalidate = source_supervisor.revalidate_recorded_inputs

    def record_revalidation(
        launch: dict[str, object],
        *,
        run_root: Path,
        repo_root: Path,
    ) -> tuple[dict[str, object], object]:
        revalidated_run_roots.append(run_root)
        return original_revalidate(
            launch,
            run_root=run_root,
            repo_root=repo_root,
        )

    monkeypatch.setattr(
        source_supervisor,
        "revalidate_recorded_inputs",
        record_revalidation,
    )
    code_run = commit_supervisor.load_terminal_code_run(
        base_root,
        repair_run_roots=(repair_root,),
    )
    assert code_run["repositories"] == ("project",)
    assert len(code_run["identities"]) == 2
    assert revalidated_run_roots == [base_root.resolve(), repair_root.resolve()]

    _source_list, store, pr_list, pr_completion = _pr_fixture(tmp_path / "pr")
    pr_inputs, _snapshot, _paths = commit_supervisor.validate_pr_inputs(
        source_repo_list=Path(code_run["inputs"]["repo_list"]["path"]),
        pr_store=store,
        pr_repo_list=pr_list,
        completion_receipt=pr_completion,
    )
    commit_root = tmp_path / "commits"
    args = commit_supervisor.parse_args(
        [
            "--code-run-root",
            str(base_root),
            "--code-repair-run-root",
            str(repair_root),
            "--run-root",
            str(commit_root),
            "--pr-store",
            str(store),
            "--pr-repo-list",
            str(pr_list),
            "--pr-completion-receipt",
            str(pr_completion),
            "--minimum-free-bytes",
            "0",
        ]
    )
    command = commit_supervisor.build_command(args, code_run, pr_inputs)
    assert command[command.index("--code-output-root") + 1] == str(
        base_root / "reindexed"
    )
    assert command[command.index("--dedup-db") + 1] == str(
        base_root / "dedup.sqlite"
    )

    plan_path = tmp_path / "composition.json"
    commit_supervisor._write_composition_plan(
        plan_path,
        code_run=code_run,
        commit_run_root=commit_root,
        dedup_receipt=tmp_path / "dedup-receipt.json",
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [run["run_id"] for run in plan["runs"]] == [
        "full-code",
        "code-repair-1",
        "full-commits",
    ]

    source_supervisor._atomic_json(
        repair_manifest,
        {
            "code_revision": repair_inputs["code_revision"],
            "done": {},
            "failed": {"project::code": {"stage": "index_project"}},
        },
    )
    source_supervisor.write_exit_receipt(
        repair_root / "exit_receipt.json",
        launch_path=repair_launch_path,
        code_revision=repair_args.expected_code_revision,
        return_code=1,
        manifest_path=repair_manifest,
        completion_path=completion_path,
        schema=source_supervisor.TARGETED_EXIT_SCHEMA,
        selected_repositories=["project"],
        repair_base_code_run=repair_base["identity"],
    )
    with pytest.raises(RuntimeError, match="code repair 1 exit code is non-zero"):
        commit_supervisor.load_terminal_code_run(
            base_root,
            repair_run_roots=(repair_root,),
        )


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
