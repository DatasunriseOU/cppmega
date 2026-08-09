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
from tests.test_source_conveyor_supervisor import _git, _input_fixture


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


def test_commit_upgrade_flags_require_fixed_audit_timestamp(tmp_path: Path) -> None:
    common = [
        "--code-run-root",
        str(tmp_path / "code"),
        "--run-root",
        str(tmp_path / "commit"),
        "--pr-store",
        str(tmp_path / "prs.sqlite"),
        "--pr-repo-list",
        str(tmp_path / "pr-repos.json"),
        "--pr-completion-receipt",
        str(tmp_path / "completion.json"),
        "--expected-code-revision",
        "b" * 40,
        "--allow-code-revision-upgrade-from",
        "a" * 40,
        "--code-revision-upgrade-reason",
        "skip known gitlink failures",
    ]
    with pytest.raises(SystemExit, match="authorized-at"):
        commit_supervisor.parse_args(common)

    args = commit_supervisor.parse_args(
        [*common, "--code-revision-upgrade-authorized-at", "2026-08-09T10:00:00Z"]
    )
    assert args.expected_code_revision == "b" * 40
    assert args.allow_code_revision_upgrade_from == "a" * 40
    assert args.code_revision_upgrade_authorized_at == "2026-08-09T10:00:00Z"


def test_commit_build_command_reuses_state_roots_and_emits_upgrade_flags(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "old-state"
    args = commit_supervisor.parse_args(
        [
            "--code-run-root",
            str(tmp_path / "code"),
            "--run-root",
            str(tmp_path / "new-commit"),
            "--resume-from-run-root",
            str(state_root),
            "--pr-store",
            str(tmp_path / "prs.sqlite"),
            "--pr-repo-list",
            str(tmp_path / "pr-repos.json"),
            "--pr-completion-receipt",
            str(tmp_path / "completion.json"),
            "--expected-code-revision",
            "b" * 40,
            "--allow-code-revision-upgrade-from",
            "a" * 40,
            "--code-revision-upgrade-reason",
            "skip known gitlink failures",
            "--code-revision-upgrade-authorized-at",
            "2026-08-09T10:00:00Z",
        ]
    )
    code_run = {
        "launch": {"code_revision": "a" * 40},
        "inputs": {
            "archive": {"resolved_path": "/data/archive.tar.zst"},
            "repo_list": {"path": "/data/source-repos.json"},
            "source_quarantine_manifest": {"path": "/data/quarantine.json"},
            "python": {"path": "/venv/bin/python"},
        },
        "target_lengths": (1024, 2048),
        "code_output_root": "/data/code",
        "commit_output_root": "/data/commits",
        "dedup_db": "/data/dedup.sqlite",
    }
    pr_inputs = {
        "repo_list": {"path": "/data/pr-repos.json"},
        "store": {"path": "/data/prs.sqlite"},
        "completion": {"path": "/data/pr-completion.json"},
    }
    command = commit_supervisor.build_command(args, code_run, pr_inputs)
    assert command[command.index("--expected-code-revision") + 1] == "b" * 40
    assert command[command.index("--conveyor-root") + 1] == str(
        state_root / "conveyor"
    )
    assert command[command.index("--work-parent-dir") + 1] == str(
        state_root / "work-parent"
    )
    assert command[command.index("--allow-code-revision-upgrade-from") + 1] == "a" * 40


def test_commit_upgrade_uses_live_quarantine_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, argv = _input_fixture(tmp_path)
    quarantine = repo / "configs" / "source_quarantine_manifest.json"
    quarantine.parent.mkdir()
    _write_json(quarantine, {"schema": "fixture", "entries": ["old"]})
    _git(repo, "add", str(quarantine.relative_to(repo)))
    _git(repo, "commit", "-q", "-m", "old quarantine")
    historical_revision = _git(repo, "rev-parse", "HEAD")
    argv[argv.index("--source-quarantine-manifest") + 1] = str(quarantine)
    argv[argv.index("--expected-code-revision") + 1] = historical_revision

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
        code_revision=historical_revision,
        return_code=1,
        manifest_path=base_manifest,
        completion_path=base_root / "conveyor" / "completion_receipt.json",
    )

    _write_json(quarantine, {"schema": "fixture", "entries": ["new"]})
    _git(repo, "add", str(quarantine.relative_to(repo)))
    _git(repo, "commit", "-q", "-m", "new quarantine")
    execution_revision = _git(repo, "rev-parse", "HEAD")
    cache_root = tmp_path / "private-cache"
    monkeypatch.setattr(
        source_supervisor,
        "_historical_input_cache_root",
        lambda: cache_root,
    )

    code_run = commit_supervisor.load_commit_source_run(
        base_root,
        execution_code_revision=execution_revision,
        allowed_historical_code_revisions={historical_revision},
    )

    assert code_run["repair_required"] is True
    assert code_run["inputs"]["source_quarantine_manifest"] == {
        "path": str(quarantine.resolve()),
        "sha256": _sha256(quarantine),
    }
    historical_sha256 = base_inputs["source_quarantine_manifest"]["sha256"]
    assert _sha256(
        cache_root / f"source_quarantine_manifest.{historical_sha256}.json"
    ) == historical_sha256
    assert not (base_root / "frozen_inputs").exists()

    args = commit_supervisor.parse_args(
        [
            "--code-run-root",
            str(base_root),
            "--run-root",
            str(tmp_path / "commit-run"),
            "--pr-store",
            str(tmp_path / "prs.sqlite"),
            "--pr-repo-list",
            str(tmp_path / "pr-repos.json"),
            "--pr-completion-receipt",
            str(tmp_path / "pr-completion.json"),
            "--expected-code-revision",
            execution_revision,
            "--allow-code-revision-upgrade-from",
            historical_revision,
            "--code-revision-upgrade-reason",
            "use current quarantine binding after source revision upgrade",
            "--code-revision-upgrade-authorized-at",
            "2026-08-09T12:00:00Z",
        ]
    )
    command = commit_supervisor.build_command(
        args,
        code_run,
        {
            "repo_list": {"path": str(tmp_path / "pr-repos.json")},
            "store": {"path": str(tmp_path / "prs.sqlite")},
            "completion": {"path": str(tmp_path / "pr-completion.json")},
        },
    )
    assert command[command.index("--source-quarantine-manifest") + 1] == str(
        quarantine.resolve()
    )


def test_resume_state_root_binds_old_manifest_and_rejects_wrong_source(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old-commits"
    manifest_path = old_root / "conveyor" / "_done.json"
    completion_path = old_root / "conveyor" / "completion_receipt.json"
    (old_root / "conveyor" / "locks").mkdir(parents=True)
    supervisor_lock = old_root / "launch.lock"
    commit_lock = old_root / "conveyor" / "locks" / "commits.lock"
    supervisor_lock.touch()
    commit_lock.touch()
    manifest = {
        "code_revision": {"git_commit": "a" * 40},
        "done": {"project::r0": {"rows": 1}},
        "failed": {"project::commits": {"stage": "extract"}},
    }
    source_supervisor._atomic_json(manifest_path, manifest)
    launch_path = old_root / "launch_receipt.json"
    binding = {
        "schema": source_supervisor.RUN_BINDING_SCHEMA,
        "streams": "commits",
    }
    source_supervisor._atomic_json(
        launch_path,
        {
            "schema": source_supervisor.LAUNCH_SCHEMA,
            "status": "running",
            "code_revision": "a" * 40,
            "command": ["python", "scripts/streaming_conveyor.py", "--streams", "commits"],
            "run_binding": binding,
            "run_binding_sha256": source_supervisor._canonical_sha256(binding),
            "outputs": {
                "conveyor_manifest": str(manifest_path),
                "completion_receipt": str(completion_path),
            },
        },
    )
    source_supervisor.write_exit_receipt(
        old_root / "exit_receipt.json",
        launch_path=launch_path,
        code_revision="a" * 40,
        return_code=130,
        manifest_path=manifest_path,
        completion_path=completion_path,
    )

    state = commit_supervisor._validate_resume_state_root(
        old_root,
        expected_revision="b" * 40,
        allow_from="a" * 40,
    )
    assert state["manifest_path"] == manifest_path
    with pytest.raises(RuntimeError, match="authorized source"):
        commit_supervisor._validate_resume_state_root(
            old_root,
            expected_revision="b" * 40,
            allow_from="c" * 40,
        )
    commit_lock.unlink()
    with pytest.raises(RuntimeError, match="resume commit stream lock is missing"):
        commit_supervisor._validate_resume_state_root(
            old_root,
            expected_revision="b" * 40,
            allow_from="a" * 40,
        )
    assert not commit_lock.exists()


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

    commit_source = commit_supervisor.load_commit_source_run(base_root)
    assert commit_source["repair_required"] is True
    assert commit_source["repositories"] == ("project",)
    assert commit_source["identities"] == [commit_source["identity"]]
    with pytest.raises(RuntimeError, match="run root does not exist"):
        commit_supervisor.wait_for_terminal_code_run(
            base_root,
            (tmp_path / "missing-repair",),
            poll_seconds=0.01,
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
    with pytest.raises(ValueError, match="contributed no code success"):
        commit_supervisor.load_terminal_code_run(
            base_root,
            repair_run_roots=(repair_root,),
        )


def test_commit_waits_for_later_repair_after_partial_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial_root = tmp_path / "partial"
    final_root = tmp_path / "final"
    partial_root.mkdir()
    final_root.mkdir()
    _write_json(partial_root / "exit_receipt.json", {"exit_code": 130})
    expected = {"status": "complete"}

    def load_chain(*args: object, **kwargs: object) -> dict[str, str]:
        if not (final_root / "exit_receipt.json").exists():
            raise RuntimeError("later repair is still running")
        return expected

    def finish_later(_seconds: float) -> None:
        _write_json(final_root / "exit_receipt.json", {"exit_code": 0})

    monkeypatch.setattr(commit_supervisor, "load_terminal_code_run", load_chain)
    result = commit_supervisor.wait_for_terminal_code_run(
        tmp_path / "base",
        (partial_root, final_root),
        poll_seconds=0.01,
        sleeper=finish_later,
    )

    assert result is expected


def test_commit_chain_accepts_useful_partial_nonzero_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base_root = tmp_path / "base"
    partial_root = tmp_path / "partial"
    final_root = tmp_path / "final"
    for root in (base_root, partial_root, final_root):
        (root / "conveyor").mkdir(parents=True)
    _write_json(partial_root / "exit_receipt.json", {"exit_code": 130})
    partial_salvaged_exit = partial_root / source_supervisor.SALVAGED_EXIT_FILENAME
    _write_json(partial_salvaged_exit, {"schema": "salvaged fixture"})
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    code_root.mkdir()
    commit_root.mkdir()
    dedup_db = tmp_path / "dedup.sqlite"
    dedup_db.write_bytes(b"dedup")
    base_identity = {
        "launch_sha256": "1" * 64,
        "exit_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
    }
    stored_inputs = {
        "source_quarantine_manifest": {
            "path": "/historical/quarantine.json",
            "sha256": "8" * 64,
        }
    }
    live_inputs = {
        "source_quarantine_manifest": {
            "path": "/current/quarantine.json",
            "sha256": "9" * 64,
        }
    }
    base = {
        "root": base_root,
        "launch_path": base_root / "launch_receipt.json",
        "exit_path": base_root / "exit_receipt.json",
        "manifest_path": base_root / "conveyor" / "_done.json",
        "launch": {"repository_root": str(repo)},
        "inputs": stored_inputs,
        "producer_root": repo,
        "code_output_root": code_root,
        "commit_output_root": commit_root,
        "dedup_db": dedup_db,
        "target_lengths": (1024,),
        "repositories": ("alpha", "beta"),
        "failed_repositories": {"alpha", "beta"},
        "successful_repositories": set(),
        "code_artifacts": set(),
        "identity": base_identity,
        "archive_identity": "archive",
        "input_binding": "inputs",
    }
    monkeypatch.setattr(
        source_supervisor,
        "load_repair_base_code_run",
        lambda _root: base,
    )
    monkeypatch.setattr(
        commit_supervisor,
        "_revalidate_recorded_inputs",
        lambda *args, **kwargs: (live_inputs, object()),
    )

    verified_inputs: list[dict[str, object]] = []

    def verify_completion(*args: object, **kwargs: object) -> dict[str, int]:
        verified_inputs.append(kwargs["inputs"])
        return {"successful_repository_count": 1}

    monkeypatch.setattr(
        source_supervisor,
        "verify_completion_receipt",
        verify_completion,
    )

    launches: dict[Path, dict[str, object]] = {}
    for root in (partial_root, final_root):
        launch = {
            "repository_root": str(repo),
            "inputs": stored_inputs,
            "run_binding": {},
            "run_binding_sha256": source_supervisor._canonical_sha256({}),
        }
        source_supervisor._atomic_json(root / "launch_receipt.json", launch)
        launches[root] = launch
    completion_path = final_root / "conveyor" / "completion_receipt.json"
    source_supervisor._atomic_json(completion_path, {})
    source_supervisor._atomic_json(
        final_root / "exit_receipt.json",
        {
            "completion_receipt": {
                "path": str(completion_path),
                "sha256": _sha256(completion_path),
            },
            "terminal_coverage": {
                "status": "complete",
                "expected_repository_count": 1,
                "successful_repository_count": 1,
                "immutable_inputs_reverified_at_finish": True,
            },
        },
    )

    def loaded_run(raw: object, **_kwargs: object) -> tuple:
        run_id = raw["run_id"]
        is_partial = run_id == "code-repair-1"
        root = partial_root if is_partial else final_root
        expected_exit = (
            partial_salvaged_exit if is_partial else final_root / "exit_receipt.json"
        )
        assert Path(raw["exit_receipt"]) == expected_exit
        repository = "alpha" if is_partial else "beta"
        selected = ["alpha", "beta"] if is_partial else ["beta"]
        done = {
            f"{repository}::code": {
                "artifact_filename": f"{repository}.parquet",
                "lengths": {"1024": {"rows": 1}},
            }
        }
        portable = {
            "launch": {
                "schema": source_supervisor.TARGETED_LAUNCH_SCHEMA,
                "sha256": _sha256(root / "launch_receipt.json"),
            },
            "exit": {
                "exit_code": 130 if is_partial else 0,
                "sha256": ("4" if is_partial else "5") * 64,
            },
            "manifest": {"sha256": ("6" if is_partial else "7") * 64},
            "streams": "code",
            "selected_repositories": selected,
            "repair_base_code_run": base_identity,
        }
        allowlist = {
            ("code", 1024): {f"{repository}.parquet": 1},
            ("commits", 1024): {},
        }
        details = {
            "done": done,
            "dedup_path": str(dedup_db),
        }
        return (
            portable,
            allowlist,
            {},
            {repository},
            set(),
            set(),
            set(),
            "archive",
            "inputs",
            details,
        )

    monkeypatch.setattr(commit_supervisor, "_load_run", loaded_run)
    with pytest.raises(RuntimeError, match="final code repair exit code is non-zero"):
        commit_supervisor.load_terminal_code_run_chain(
            base_root,
            (partial_root,),
        )
    result = commit_supervisor.load_terminal_code_run_chain(
        base_root,
        (partial_root, final_root),
        execution_code_revision="b" * 40,
        allowed_historical_code_revisions={"a" * 40},
    )

    assert result["repositories"] == ("alpha", "beta")
    assert len(result["identities"]) == 3
    assert result["inputs"] == live_inputs
    assert verified_inputs == [stored_inputs]


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
