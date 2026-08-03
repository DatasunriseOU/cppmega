from __future__ import annotations

import hashlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import source_conveyor_supervisor as supervisor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_repo(root: Path) -> tuple[Path, str]:
    repo = root / "repo"
    indexer = repo / "tools" / "clang_indexer"
    tokenizer = repo / "cppmega" / "tokenizer"
    indexer.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    (indexer / "index_project.py").write_text(
        "from indexer_helper import VALUE\n",
        encoding="utf-8",
    )
    (indexer / "indexer_helper.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Supervisor Test")
    _git(repo, "config", "user.email", "supervisor@example.test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


def _input_fixture(tmp_path: Path) -> tuple[Path, list[str]]:
    repo, revision = _source_repo(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    run_root = tmp_path / "run"
    run_root.mkdir()
    archive = inputs / "source.tar.zst"
    archive.write_bytes(b"verified archive fixture")
    archive_stat = archive.stat()
    archive_receipt = inputs / "archive-sha.json"
    _write_json(
        archive_receipt,
        {
            "schema": supervisor.ARCHIVE_SHA_SCHEMA,
            "status": "verified",
            "exit_code": 0,
            "resolved_path": str(archive.resolve()),
            "size_bytes": archive_stat.st_size,
            "mtime_epoch": int(archive_stat.st_mtime),
            "inode": archive_stat.st_ino,
            "device": archive_stat.st_dev,
            "sha256": _sha256(archive),
        },
    )
    repo_list = inputs / "repo-list.json"
    _write_json(
        repo_list,
        {
            "schema_version": 2,
            "repos": [
                {
                    "bare_name": "project",
                    "project_identity": "owner/project",
                    "owner_repo": "owner/project",
                }
            ],
            "by_bare_name": {"project": "owner/project"},
            "project_identities": ["owner/project"],
            "repo_names": ["owner/project"],
            "unresolved": [],
        },
    )
    inventory = inputs / "archive-inventory.json"
    _write_json(
        inventory,
        {
            "schema": supervisor.ARCHIVE_INVENTORY_SCHEMA,
            "status": "verified",
            "archive_sha256_receipt": {
                "path": str(archive_receipt),
                "sha256": _sha256(archive_receipt),
            },
            "archive_unique_worktree_repo_count": 1,
            "archive_sorted_repo_names_json_sha256": (
                supervisor._canonical_sha256(["project"])
            ),
            "canonical_repo_list": {
                "path": str(repo_list),
                "sha256": _sha256(repo_list),
                "mapping_entry_count": 1,
                "archive_repos_without_mapping": 0,
            },
            "streaming_contract": {
                "expected_attempted_repo_count": 1,
                "persistent_source_cache": False,
                "one_repo_materialized_at_a_time": True,
            },
        },
    )
    quarantine = inputs / "quarantine.json"
    _write_json(quarantine, {"entries": []})
    argv = [
        "--run-root",
        str(run_root),
        "--archive",
        str(archive),
        "--archive-sha256-receipt",
        str(archive_receipt),
        "--archive-inventory-receipt",
        str(inventory),
        "--repo-list",
        str(repo_list),
        "--source-quarantine-manifest",
        str(quarantine),
        "--tokenizer",
        str(repo / "cppmega" / "tokenizer" / "tokenizer.json"),
        "--expected-code-revision",
        revision,
        "--python",
        sys.executable,
        "--libclang",
        sys.executable,
        "--minimum-free-bytes",
        "0",
    ]
    return repo, argv


def test_supervisor_validates_bound_inputs_and_builds_repo_native_command(
    tmp_path: Path,
) -> None:
    repo, argv = _input_fixture(tmp_path)
    args = supervisor.parse_args(argv)

    inputs = supervisor.validate_inputs(args, repo_root=repo)
    command = supervisor.build_command(args, inputs)

    assert inputs["archive_inventory_receipt"]["repository_count"] == 1
    assert inputs["repo_list"]["mapping_entry_count"] == 1
    assert inputs["code_revision"]["git_commit"] == args.expected_code_revision
    assert command[command.index("--source-archive") + 1] == str(
        Path(args.archive).resolve()
    )
    assert command[command.index("--repo-workers") + 1] == "1"
    assert command[command.index("--parse-workers") + 1] == "1"
    assert command[command.index("--completion-receipt") + 1].endswith(
        "/conveyor/completion_receipt.json"
    )
    parsed_child = supervisor.conveyor.parse_args(command[3:])
    assert parsed_child.streams == "code"
    assert parsed_child.completion_receipt.endswith("/conveyor/completion_receipt.json")
    assert "--no-resume" not in command
    assert "--max-repos" not in command
    assert "--only-repo" not in command


def test_targeted_repair_reuses_base_outputs_and_binds_receipts(
    tmp_path: Path,
) -> None:
    repo, argv = _input_fixture(tmp_path)
    base_args = supervisor.parse_args(argv)
    base_inputs = supervisor.validate_inputs(base_args, repo_root=repo)
    base_root = Path(base_args.run_root)
    (base_root / "reindexed").mkdir()
    (base_root / "reindexed-commits").mkdir()
    (base_root / "dedup.sqlite").write_bytes(b"dedup")
    base_command = supervisor.build_command(base_args, base_inputs)
    base_binding = supervisor.build_run_binding(base_args, base_inputs)
    base_launch = supervisor.build_launch_receipt(
        base_args,
        inputs=base_inputs,
        command=base_command,
        run_binding=base_binding,
        attempt=1,
    )
    base_launch["status"] = "running"
    base_launch_path = base_root / "launch_receipt.json"
    supervisor._atomic_json(base_launch_path, base_launch)
    base_manifest = base_root / "conveyor" / "_done.json"
    supervisor._atomic_json(
        base_manifest,
        {
            "code_revision": base_inputs["code_revision"],
            "done": {},
            "failed": {"project::code": {"stage": "index_project"}},
        },
    )
    supervisor.write_exit_receipt(
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
        ]
    )
    repair_args = supervisor.parse_args(repair_argv)
    repair_inputs = supervisor.validate_inputs(repair_args, repo_root=repo)
    repair_base = supervisor.load_repair_base_code_run(base_root)
    for invalid_lengths in ([0, 2048], [-1, 2048]):
        invalid_base_launch = dict(base_launch)
        invalid_base_launch["target_lengths"] = invalid_lengths
        supervisor._atomic_json(base_launch_path, invalid_base_launch)
        with pytest.raises(RuntimeError, match="target lengths are invalid"):
            supervisor.load_repair_base_code_run(base_root)
    supervisor._atomic_json(base_launch_path, base_launch)
    wrong_selection_argv = list(repair_argv)
    wrong_selection_argv[-1] = "owner/project"
    with pytest.raises(RuntimeError, match="not failed by its base run"):
        supervisor.validate_repair_request(
            supervisor.parse_args(wrong_selection_argv),
            repair_inputs,
            repair_base,
        )
    supervisor.validate_repair_request(repair_args, repair_inputs, repair_base)
    command = supervisor.build_command(repair_args, repair_inputs, repair_base)
    binding = supervisor.build_run_binding(repair_args, repair_inputs, repair_base)
    launch = supervisor.build_launch_receipt(
        repair_args,
        inputs=repair_inputs,
        command=command,
        run_binding=binding,
        attempt=1,
        repair_base=repair_base,
    )

    assert launch["schema"] == supervisor.TARGETED_LAUNCH_SCHEMA
    assert launch["selected_repositories"] == ["project"]
    assert launch["repair_base_code_run"] == repair_base["identity"]
    assert launch["outputs"]["code_output_root"] == str(
        base_root / "reindexed"
    )
    assert launch["outputs"]["commit_output_root"] == str(
        base_root / "reindexed-commits"
    )
    assert launch["outputs"]["dedup_db"] == str(base_root / "dedup.sqlite")
    assert command[command.index("--only-repo") + 1] == "project"
    assert command[command.index("--max-repos") + 1] == "1"
    assert command[command.index("--conveyor-root") + 1] == str(
        repair_root / "conveyor"
    )
    parsed_child = supervisor.conveyor.parse_args(command[3:])
    assert parsed_child.only_repo == ["project"]
    assert parsed_child.max_repos == 1
    assert parsed_child.code_output_root == str(base_root / "reindexed")

    launch["status"] = "running"
    repair_launch_path = repair_root / "launch_receipt.json"
    supervisor._atomic_json(repair_launch_path, launch)
    repair_manifest = repair_root / "conveyor" / "_done.json"
    supervisor._atomic_json(
        repair_manifest,
        {
            "code_revision": repair_inputs["code_revision"],
            "done": {"project::code": {}},
            "failed": {},
        },
    )
    exit_receipt = supervisor.write_exit_receipt(
        repair_root / "exit_receipt.json",
        launch_path=repair_launch_path,
        code_revision=repair_args.expected_code_revision,
        return_code=0,
        manifest_path=repair_manifest,
        completion_path=repair_root / "conveyor" / "completion_receipt.json",
        schema=supervisor.TARGETED_EXIT_SCHEMA,
        selected_repositories=["project"],
        repair_base_code_run=repair_base["identity"],
    )
    assert exit_receipt["schema"] == supervisor.TARGETED_EXIT_SCHEMA
    assert exit_receipt["selected_repositories"] == ["project"]
    assert exit_receipt["repair_base_code_run"] == repair_base["identity"]

    for name in ("reindexed", "reindexed-commits"):
        output_root = base_root / name
        output_root.rmdir()
        with pytest.raises(RuntimeError, match="cannot be resolved"):
            supervisor.load_repair_base_code_run(base_root)
        output_root.write_text("not a directory", encoding="utf-8")
        with pytest.raises(RuntimeError, match="is not a directory"):
            supervisor.load_repair_base_code_run(base_root)
        output_root.unlink()
        output_root.mkdir()


def test_targeted_repairs_exclusively_lock_shared_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, argv = _input_fixture(tmp_path)
    base_root = tmp_path / "base"
    base_root.mkdir()
    repair_argv = list(argv)
    repair_argv.extend(
        [
            "--repair-base-code-run-root",
            str(base_root),
            "--only-repo",
            "project",
        ]
    )
    args = supervisor.parse_args(repair_argv)
    inputs = supervisor.validate_inputs(args, repo_root=repo)
    monkeypatch.setattr(supervisor, "validate_inputs", lambda _args: inputs)
    loaded = False

    def reject_base_load(_root: Path) -> dict[str, object]:
        nonlocal loaded
        loaded = True
        raise AssertionError("base state was read before acquiring its write lock")

    monkeypatch.setattr(supervisor, "load_repair_base_code_run", reject_base_load)
    lock_path = base_root / "targeted-repair.lock"
    held = lock_path.open("a+b")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="another targeted repair"):
            supervisor._run(args)
    finally:
        held.close()
    assert loaded is False


def test_supervisor_rejects_inventory_that_does_not_bind_sha_receipt(
    tmp_path: Path,
) -> None:
    repo, argv = _input_fixture(tmp_path)
    args = supervisor.parse_args(argv)
    inventory = Path(args.archive_inventory_receipt)
    value = json.loads(inventory.read_text(encoding="utf-8"))
    value["archive_sha256_receipt"]["sha256"] = "0" * 64
    _write_json(inventory, value)

    with pytest.raises(
        RuntimeError,
        match="does not bind the live SHA receipt",
    ):
        supervisor.validate_inputs(args, repo_root=repo)


def test_supervisor_terminal_revalidation_can_ignore_launch_only_disk_gate(
    tmp_path: Path,
) -> None:
    repo, argv = _input_fixture(tmp_path)
    args = supervisor.parse_args(argv)
    args.minimum_free_bytes = 2**63

    with pytest.raises(RuntimeError, match="insufficient free disk"):
        supervisor.validate_inputs(args, repo_root=repo)

    inputs = supervisor.validate_inputs(
        args,
        repo_root=repo,
        enforce_minimum_free_disk=False,
    )
    assert inputs["free_disk_bytes"] < args.minimum_free_bytes


def test_supervisor_revalidates_recorded_inputs_through_shared_seam(
    tmp_path: Path,
) -> None:
    repo, argv = _input_fixture(tmp_path)
    args = supervisor.parse_args(argv)
    inputs = supervisor.validate_inputs(args, repo_root=repo)

    live, revalidation_args = supervisor.revalidate_recorded_inputs(
        {
            "code_revision": args.expected_code_revision,
            "inputs": inputs,
        },
        run_root=Path(args.run_root),
        repo_root=repo,
    )

    assert live["python"] == inputs["python"]
    assert live["libclang"] == inputs["libclang"]
    assert revalidation_args.expected_code_revision == args.expected_code_revision


def test_supervisor_rejects_incompatible_resume_binding(
    tmp_path: Path,
) -> None:
    repo, argv = _input_fixture(tmp_path)
    args = supervisor.parse_args(argv)
    inputs = supervisor.validate_inputs(args, repo_root=repo)
    binding = supervisor.build_run_binding(args, inputs)
    assert binding["execution_policy"] == {
        "repo_workers": 1,
        "max_active_repos": 1,
        "workers": 1,
        "parse_workers": 1,
        "memory_limit_gb": 16.0,
        "code_index_timeout_s": 0,
        "code_index_stall_timeout_s": 0,
        "minimum_free_bytes": 0,
        "resume": True,
        "persistent_source_cache": False,
    }
    launch_path = Path(args.run_root) / "launch_receipt.json"
    supervisor._atomic_json(
        launch_path,
        {
            "schema": supervisor.LAUNCH_SCHEMA,
            "attempt": 1,
            "run_binding": binding,
            "run_binding_sha256": supervisor._canonical_sha256(binding),
        },
    )

    changed_binding = dict(binding)
    changed_binding["tokenizer_sha256"] = "f" * 64
    with pytest.raises(
        RuntimeError,
        match="bound to different immutable inputs",
    ):
        supervisor._resume_attempt(
            launch_path,
            run_binding=changed_binding,
        )

    assert (
        supervisor._resume_attempt(
            launch_path,
            run_binding=binding,
        )
        == 2
    )


def test_supervisor_rejects_old_artifacts_without_resume_binding(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "dedup.sqlite").write_bytes(b"old run")

    with pytest.raises(
        RuntimeError,
        match="first launch requires a dedicated new run root",
    ):
        supervisor._resume_attempt(
            run_root / "launch_receipt.json",
            run_binding={"schema": supervisor.RUN_BINDING_SCHEMA},
        )


def test_supervisor_sanitizes_ambient_macro_overrides(
    tmp_path: Path,
) -> None:
    inputs = {
        "libclang": {"path": str(tmp_path / "libclang.dylib")},
        "tokenizer": {"path": str(tmp_path / "tokenizer.json")},
    }
    environment = supervisor.build_child_environment(
        inputs,
        ambient={
            "PATH": "/usr/bin",
            "PYTHONPATH": "/unbound/imports",
            "CPPMEGA_MAX_RETAINED_MACROS": "999999999",
            "CPPMEGA_MAX_MACRO_VISIBILITY_BYTES": "999999999999",
        },
    )

    assert environment["PATH"] == "/usr/bin"
    assert "PYTHONPATH" not in environment
    assert environment["CPPMEGA_MAX_RETAINED_MACROS"] == "250000"
    assert environment["CPPMEGA_MAX_MACRO_VISIBILITY_BYTES"] == "262000000"
    assert environment["NANOCHAT_TOKENIZER_PATH"] == str(tmp_path / "tokenizer.json")


def test_supervisor_preserves_symlinked_venv_python_launcher(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "venv"
    launcher = venv / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(sys.executable)
    config = venv / "pyvenv.cfg"
    config.write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    binding = supervisor._python_executable_binding(str(launcher))

    assert binding["path"] == str(launcher)
    assert binding["resolved_binary_path"] == str(Path(sys.executable).resolve())
    assert binding["venv_config"] == {
        "path": str(config),
        "sha256": _sha256(config),
    }


def test_supervisor_requires_exact_terminal_completion_coverage(
    tmp_path: Path,
) -> None:
    repo, argv = _input_fixture(tmp_path)
    args = supervisor.parse_args(argv)
    inputs = supervisor.validate_inputs(args, repo_root=repo)
    manifest = Path(args.run_root) / "conveyor" / "_done.json"
    manifest.parent.mkdir()
    _write_json(
        manifest,
        {
            "code_revision": inputs["code_revision"],
            "source_repo_list": {
                "schema": "cppmega_source_repo_list_binding_v1",
                "sha256": inputs["repo_list"]["sha256"],
                "canonical_mapping_sha256": inputs["repo_list"][
                    "canonical_mapping_sha256"
                ],
                "mapping_count": inputs["repo_list"]["mapping_entry_count"],
            },
            "done": {"project::code": {}},
            "failed": {},
        },
    )

    completion = manifest.parent / "completion_receipt.json"

    def write_completion(repositories: list[str]) -> None:
        _write_json(
            completion,
            {
                "schema": supervisor.COMPLETION_SCHEMA,
                "status": "success",
                "streams": "code",
                "interrupted": False,
                "manifest": {
                    "path": str(manifest.resolve()),
                    "size_bytes": manifest.stat().st_size,
                    "sha256": _sha256(manifest),
                },
                "total_done_unit_count": len(repositories),
                "failed_unit_count": 0,
                "non_code_done_unit_count": 0,
                "code_repositories": repositories,
                "code_repository_names_sha256": (
                    supervisor._canonical_sha256(repositories)
                ),
                "code_revision": inputs["code_revision"],
                "source_repo_list": {
                    "schema": "cppmega_source_repo_list_binding_v1",
                    "sha256": inputs["repo_list"]["sha256"],
                    "canonical_mapping_sha256": inputs["repo_list"][
                        "canonical_mapping_sha256"
                    ],
                    "mapping_count": inputs["repo_list"]["mapping_entry_count"],
                },
                "source_repo_list_reverified_at_finish": True,
            },
        )

    write_completion(["project"])
    coverage = supervisor.verify_completion_receipt(
        completion,
        manifest_path=manifest,
        args=args,
        inputs=inputs,
    )
    assert coverage["successful_repository_count"] == 1

    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["done"] = {}
    _write_json(manifest, value)
    write_completion([])
    with pytest.raises(
        RuntimeError,
        match="repository coverage is incomplete",
    ):
        supervisor.verify_completion_receipt(
            completion,
            manifest_path=manifest,
            args=args,
            inputs=inputs,
        )


def test_supervisor_first_signal_targets_driver_not_inflight_grandchild(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready"
    observed = tmp_path / "observed"
    grandchild_pid_path = tmp_path / "grandchild-pid"
    grandchild_observed = tmp_path / "grandchild-observed"
    grandchild_ready = tmp_path / "grandchild-ready"
    grandchild_program = (
        "import signal,sys,time\n"
        "from pathlib import Path\n"
        "def stop(*_):\n"
        " Path(sys.argv[1]).write_text('grandchild', encoding='utf-8')\n"
        " raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "Path(sys.argv[2]).write_text('ready', encoding='utf-8')\n"
        "while True: time.sleep(0.1)\n"
    )
    program = (
        "import signal,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "def stop(*_):\n"
        " Path(sys.argv[2]).write_text('one', encoding='utf-8')\n"
        " raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"grandchild = subprocess.Popen([sys.executable, '-c', {grandchild_program!r}, sys.argv[4], sys.argv[5]])\n"
        "Path(sys.argv[3]).write_text(str(grandchild.pid), encoding='utf-8')\n"
        "deadline = time.monotonic() + 5\n"
        "while not Path(sys.argv[5]).exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        "Path(sys.argv[1]).write_text('ready', encoding='utf-8')\n"
        "while True: time.sleep(0.1)\n"
    )
    with (tmp_path / "child.log").open("ab", buffering=0) as log:
        child = supervisor._spawn_child(
            [
                sys.executable,
                "-c",
                program,
                str(ready),
                str(observed),
                str(grandchild_pid_path),
                str(grandchild_observed),
                str(grandchild_ready),
            ],
            cwd=tmp_path,
            environment=os.environ.copy(),
            log=log,
        )
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists()
            assert grandchild_pid_path.exists()
            assert os.getpgid(child.pid) != os.getpgrp()
            supervisor._forward_child_signal(child, signal.SIGTERM)
            assert child.wait(timeout=10) == 0
            assert observed.read_text(encoding="utf-8") == "one"
            assert not grandchild_observed.exists()
            os.kill(int(grandchild_pid_path.read_text(encoding="utf-8")), 0)
        finally:
            supervisor._terminate_child_group(child, grace_seconds=0.1)
    deadline = time.monotonic() + 10
    while not grandchild_observed.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert grandchild_observed.read_text(encoding="utf-8") == "grandchild"


def test_supervisor_atomic_receipt_and_signal_exit_code(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    supervisor._atomic_json(receipt, {"status": "first"})
    supervisor._atomic_json(receipt, {"status": "second"})

    assert json.loads(receipt.read_text(encoding="utf-8")) == {"status": "second"}
    assert supervisor._portable_exit_code(-signal.SIGTERM) == (128 + signal.SIGTERM)
