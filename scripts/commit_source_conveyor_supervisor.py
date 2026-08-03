#!/usr/bin/env python3
"""Launch the commit conveyor from a terminal canonical code-run chain."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega.data.source_conveyor_composition import (  # noqa: E402
    SOURCE_COMPOSITION_PLAN_SCHEMA,
    _load_run,
    build_packed_source_inventory_receipt,
    load_source_composition,
)
from scripts import source_conveyor_supervisor as source_supervisor  # noqa: E402
from scripts import streaming_conveyor as conveyor  # noqa: E402
from scripts.data.verify_global_dedup_store import (  # noqa: E402
    verify_global_dedup_store,
)

DEFAULT_MINIMUM_FREE_BYTES = 50 * 1024**3


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return dict(value)


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def _plain_file(path: Path, *, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} cannot be resolved: {path}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not a regular file: {resolved}")
    return resolved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-run-root", required=True)
    parser.add_argument(
        "--code-repair-run-root",
        action="append",
        default=[],
        help="Targeted code-repair run root (repeatable, ordered).",
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pr-store", required=True)
    parser.add_argument("--pr-repo-list", required=True)
    parser.add_argument("--pr-completion-receipt", required=True)
    parser.add_argument("--repo-workers", type=int, default=1)
    parser.add_argument("--max-active-repos", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument(
        "--minimum-free-bytes",
        type=int,
        default=DEFAULT_MINIMUM_FREE_BYTES,
    )
    parser.add_argument("--dedup-busy-timeout-seconds", type=float, default=300.0)
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    for name in ("repo_workers", "max_active_repos", "workers"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be > 0")
    if args.max_active_repos < args.repo_workers:
        raise SystemExit("--max-active-repos must be >= --repo-workers")
    if args.memory_limit_gb <= 0:
        raise SystemExit("--memory-limit-gb must be > 0")
    if args.minimum_free_bytes < 0:
        raise SystemExit("--minimum-free-bytes must be >= 0")
    if args.dedup_busy_timeout_seconds <= 0:
        raise SystemExit("--dedup-busy-timeout-seconds must be > 0")
    if (
        Path(args.code_run_root).expanduser().resolve()
        == Path(args.run_root).expanduser().resolve()
    ):
        raise SystemExit("--run-root must be separate from --code-run-root")
    roots = [
        Path(value).expanduser().resolve() for value in args.code_repair_run_root
    ]
    if len(roots) != len(set(roots)):
        raise SystemExit("--code-repair-run-root values must be unique")
    reserved = {
        Path(args.code_run_root).expanduser().resolve(),
        Path(args.run_root).expanduser().resolve(),
    }
    if any(root in reserved for root in roots):
        raise SystemExit(
            "code repair run roots must be separate from base and commit run roots"
        )
    return args


def load_terminal_code_run(
    code_run_root: Path,
    *,
    repair_run_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Revalidate one full code run through the canonical production loaders."""

    if repair_run_roots:
        return load_terminal_code_run_chain(code_run_root, repair_run_roots)

    root = code_run_root.expanduser().resolve()
    launch_path = root / "launch_receipt.json"
    exit_path = root / "exit_receipt.json"
    manifest_path = root / "conveyor" / "_done.json"
    completion_path = root / "conveyor" / "completion_receipt.json"
    launch, launch_sha256 = source_supervisor._read_json_snapshot(
        launch_path,
        label="code launch receipt",
    )
    outputs = _object(launch.get("outputs"), label="code outputs")
    lengths = launch.get("target_lengths")
    if (
        not isinstance(lengths, list)
        or not lengths
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in lengths
        )
    ):
        raise RuntimeError("code target lengths are invalid")
    buckets = tuple(lengths)
    code_root = Path(
        _text(outputs.get("code_output_root"), label="code root")
    ).resolve()
    commit_root = Path(
        _text(outputs.get("commit_output_root"), label="commit root")
    ).resolve()
    (
        portable,
        _allowlist,
        _files,
        code_terminal,
        commit_terminal,
        failed_repositories,
        failed_units,
        _archive_identity,
        _input_binding,
        details,
    ) = _load_run(
        {
            "run_id": "full-code",
            "launch_receipt": str(launch_path),
            "exit_receipt": str(exit_path),
            "manifest": str(manifest_path),
        },
        buckets=buckets,
        code_root=code_root,
        commit_root=commit_root,
    )
    if portable["launch"]["sha256"] != launch_sha256:
        raise RuntimeError("code launch changed while it was being validated")
    repositories = tuple(sorted(code_terminal))
    inventory = _object(details["inventory"], label="archive inventory")
    if (
        portable["streams"] != "code"
        or portable["exit"]["exit_code"] != 0
        or commit_terminal
        or failed_repositories
        or failed_units
        or inventory.get("archive_unique_worktree_repo_count") != len(repositories)
        or inventory.get("archive_sorted_repo_names_json_sha256")
        != source_supervisor._canonical_sha256(list(repositories))
    ):
        raise RuntimeError("code run is not a clean full code success")

    producer_root = Path(
        _text(launch.get("repository_root"), label="code repository root")
    )
    if producer_root.is_symlink():
        raise RuntimeError(
            f"code repository root must not be a symlink: {producer_root}"
        )
    producer_root = producer_root.resolve(strict=True)
    stored_inputs = _object(launch.get("inputs"), label="code inputs")
    live_inputs, validation_args = source_supervisor.revalidate_recorded_inputs(
        launch,
        run_root=root,
        repo_root=producer_root,
    )
    coverage = source_supervisor.verify_completion_receipt(
        completion_path,
        manifest_path=manifest_path,
        args=validation_args,
        inputs=live_inputs,
    )
    if coverage["successful_repository_count"] != len(repositories):
        raise RuntimeError("code completion coverage drifted")
    run_binding = _object(launch.get("run_binding"), label="code run binding")
    run_binding_sha256 = source_supervisor._require_sha256(
        launch.get("run_binding_sha256"),
        label="code run binding sha256",
    )
    if source_supervisor._canonical_sha256(run_binding) != run_binding_sha256:
        raise RuntimeError("code run binding digest drifted")
    dedup_db = _plain_file(
        Path(_text(details["dedup_path"], label="global dedup database")),
        label="global dedup database",
    )
    return {
        "root": root,
        "launch_path": launch_path,
        "exit_path": exit_path,
        "manifest_path": manifest_path,
        "launch": launch,
        "inputs": stored_inputs,
        "producer_root": producer_root,
        "code_output_root": code_root,
        "commit_output_root": commit_root,
        "dedup_db": dedup_db,
        "target_lengths": buckets,
        "repositories": repositories,
        "identity": {
            "launch_sha256": launch_sha256,
            "exit_sha256": portable["exit"]["sha256"],
            "manifest_sha256": portable["manifest"]["sha256"],
            "completion_sha256": source_supervisor._sha256_file(completion_path),
            "run_binding_sha256": run_binding_sha256,
        },
    }


def load_terminal_code_run_chain(
    code_run_root: Path,
    repair_run_roots: Sequence[Path],
) -> dict[str, Any]:
    """Validate a failed full-code run plus targeted repairs as one final corpus."""

    base = source_supervisor.load_repair_base_code_run(code_run_root)
    expected_repositories = set(base["repositories"])
    base_failed = set(base["failed_repositories"])
    successful = set(base["successful_repositories"])
    code_artifacts = set(base["code_artifacts"])
    repair_runs: list[dict[str, Any]] = []
    identities = [base["identity"]]

    for index, raw_root in enumerate(repair_run_roots, start=1):
        root = raw_root.expanduser()
        if root.is_symlink():
            raise RuntimeError(f"code repair run root must not be a symlink: {root}")
        try:
            root = root.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"code repair run root cannot be resolved: {root}") from exc
        if not root.is_dir():
            raise RuntimeError(f"code repair run root is not a directory: {root}")
        launch_path = root / "launch_receipt.json"
        exit_path = root / "exit_receipt.json"
        manifest_path = root / "conveyor" / "_done.json"
        completion_path = root / "conveyor" / "completion_receipt.json"
        launch, launch_sha256 = source_supervisor._read_json_snapshot(
            launch_path,
            label=f"code repair {index} launch receipt",
        )
        (
            portable,
            allowlist,
            _files,
            code_terminal,
            commit_terminal,
            _failed_repositories,
            _failed_units,
            archive_identity,
            input_binding,
            details,
        ) = _load_run(
            {
                "run_id": f"code-repair-{index}",
                "launch_receipt": str(launch_path),
                "exit_receipt": str(exit_path),
                "manifest": str(manifest_path),
            },
            buckets=base["target_lengths"],
            code_root=base["code_output_root"],
            commit_root=base["commit_output_root"],
        )
        if portable["exit"]["exit_code"] != 0:
            raise RuntimeError(f"code repair {index} exit code is non-zero")
        selected = set(portable["selected_repositories"])
        if (
            portable["launch"]["sha256"] != launch_sha256
            or portable["launch"]["schema"]
            != source_supervisor.TARGETED_LAUNCH_SCHEMA
            or portable["streams"] != "code"
            or commit_terminal
            or code_terminal != selected
            or not selected <= base_failed
            or portable.get("repair_base_code_run") != base["identity"]
            or archive_identity != base["archive_identity"]
            or input_binding != base["input_binding"]
            or Path(str(details["dedup_path"])).resolve() != base["dedup_db"]
        ):
            raise RuntimeError(
                f"code repair {index} is not bound to its failed full-code base"
            )
        done = details["done"]
        if not isinstance(done, dict):
            raise TypeError(f"code repair {index} done manifest must be an object")
        repair_success = {
            unit[: -len("::code")]
            for unit in done
            if unit.endswith("::code")
        }
        duplicate_success = successful & repair_success
        if duplicate_success:
            raise RuntimeError(
                "code repair duplicates successful repositories: "
                + ", ".join(sorted(duplicate_success))
            )
        repair_artifacts = {
            (bucket, filename)
            for (kind, bucket), files in allowlist.items()
            if kind == "code"
            for filename in files
        }
        if code_artifacts & repair_artifacts:
            raise RuntimeError("code repair duplicates a published code shard")
        successful.update(repair_success)
        code_artifacts.update(repair_artifacts)

        stored_inputs = _object(launch.get("inputs"), label="code repair inputs")
        run_binding = _object(
            launch.get("run_binding"),
            label=f"code repair {index} run binding",
        )
        run_binding_sha256 = source_supervisor._require_sha256(
            launch.get("run_binding_sha256"),
            label=f"code repair {index} run binding sha256",
        )
        if source_supervisor._canonical_sha256(run_binding) != run_binding_sha256:
            raise RuntimeError(f"code repair {index} run binding digest drifted")
        if portable["exit"]["exit_code"] == 0:
            coverage = source_supervisor.verify_completion_receipt(
                completion_path,
                manifest_path=manifest_path,
                args=argparse.Namespace(only_repo=sorted(selected)),
                inputs=stored_inputs,
            )
            if coverage["successful_repository_count"] != len(selected):
                raise RuntimeError(f"code repair {index} completion coverage drifted")
            exit_receipt = source_supervisor._read_json(
                exit_path,
                label=f"code repair {index} exit receipt",
            )
            completion_binding = _object(
                exit_receipt.get("completion_receipt"),
                label=f"code repair {index} exit completion binding",
            )
            terminal_coverage = _object(
                exit_receipt.get("terminal_coverage"),
                label=f"code repair {index} terminal coverage",
            )
            if (
                Path(
                    _text(
                        completion_binding.get("path"),
                        label=f"code repair {index} completion path",
                    )
                ).resolve()
                != completion_path.resolve()
                or completion_binding.get("sha256")
                != source_supervisor._sha256_file(completion_path)
                or terminal_coverage.get("status") != "complete"
                or terminal_coverage.get("expected_repository_count")
                != len(selected)
                or terminal_coverage.get("successful_repository_count")
                != len(selected)
                or terminal_coverage.get("immutable_inputs_reverified_at_finish")
                is not True
            ):
                raise RuntimeError(
                    f"code repair {index} exit completion binding drifted"
                )
        identity = {
            "launch_sha256": launch_sha256,
            "exit_sha256": portable["exit"]["sha256"],
            "manifest_sha256": portable["manifest"]["sha256"],
        }
        repair_runs.append(
            {
                "run_id": f"code-repair-{index}",
                "root": root,
                "launch_path": launch_path,
                "exit_path": exit_path,
                "manifest_path": manifest_path,
                "launch": launch,
                "inputs": stored_inputs,
                "identity": identity,
            }
        )
        identities.append(identity)

    if successful != expected_repositories:
        raise RuntimeError(
            "final code-success coverage is incomplete: "
            f"expected={len(expected_repositories)} actual={len(successful)} "
            f"missing={sorted(expected_repositories - successful)[:20]}"
        )

    execution = repair_runs[-1]
    producer_root = Path(
        _text(execution["launch"].get("repository_root"), label="repair repository root")
    )
    if producer_root.is_symlink():
        raise RuntimeError(
            f"repair repository root must not be a symlink: {producer_root}"
        )
    producer_root = producer_root.resolve(strict=True)
    source_supervisor.revalidate_recorded_inputs(
        execution["launch"],
        run_root=execution["root"],
        repo_root=producer_root,
    )

    return {
        "root": base["root"],
        "launch_path": base["launch_path"],
        "exit_path": base["exit_path"],
        "manifest_path": base["manifest_path"],
        "launch": execution["launch"],
        "inputs": execution["inputs"],
        "producer_root": producer_root,
        "code_output_root": base["code_output_root"],
        "commit_output_root": base["commit_output_root"],
        "dedup_db": base["dedup_db"],
        "target_lengths": base["target_lengths"],
        "repositories": tuple(sorted(expected_repositories)),
        "identity": base["identity"],
        "identities": identities,
        "repair_runs": repair_runs,
    }


def validate_pr_inputs(
    *,
    source_repo_list: Path,
    pr_store: Path,
    pr_repo_list: Path,
    completion_receipt: Path,
) -> tuple[dict[str, Any], conveyor.RepoListSnapshot, tuple[Path, Path, Path]]:
    """Bind the verified PR corpus using the commit indexer's validators."""

    store = _plain_file(pr_store, label="PR store")
    repo_list = _plain_file(pr_repo_list, label="PR repo list")
    completion = _plain_file(completion_receipt, label="PR completion receipt")
    _source, pr_snapshot = conveyor.load_repo_list_contracts(
        source_repo_list,
        repo_list,
    )
    binding = conveyor.load_pr_completion_binding(
        completion,
        pr_store=store,
        repo_list=repo_list,
        repo_list_snapshot=pr_snapshot,
    )
    observation = conveyor.observe_immutable_pr_store(store)
    if observation["pr_rows"] != binding["stored_pr_count"]:
        raise RuntimeError("PR store row count differs from its completion receipt")
    return (
        {
            "repo_list": {
                "path": str(repo_list),
                "sha256": binding["repo_list_sha256"],
            },
            "completion": {
                "path": str(completion),
                "sha256": binding["receipt_sha256"],
            },
            "store": {
                "path": str(store),
                "sha256": binding["pr_store_sha256"],
                **observation,
            },
            "completion_binding": binding,
        },
        pr_snapshot,
        (store, repo_list, completion),
    )


def build_command(
    args: argparse.Namespace,
    code_run: Mapping[str, Any],
    pr_inputs: Mapping[str, object],
) -> list[str]:
    run_root = Path(args.run_root).expanduser().resolve()
    inputs = _object(code_run["inputs"], label="code inputs")
    options = (
        ("--streams", "commits"),
        (
            "--source-archive",
            _object(inputs["archive"], label="source archive")["resolved_path"],
        ),
        (
            "--expected-code-revision",
            _object(code_run["launch"], label="code launch")["code_revision"],
        ),
        (
            "--target-lengths-commits",
            ",".join(str(value) for value in code_run["target_lengths"]),
        ),
        ("--repo-list", _object(inputs["repo_list"], label="source repo list")["path"]),
        (
            "--pr-repo-list",
            _object(pr_inputs["repo_list"], label="PR repo list")["path"],
        ),
        ("--pr-store", _object(pr_inputs["store"], label="PR store")["path"]),
        (
            "--pr-completion-receipt",
            _object(pr_inputs["completion"], label="PR completion")["path"],
        ),
        (
            "--source-quarantine-manifest",
            _object(
                inputs["source_quarantine_manifest"],
                label="source quarantine manifest",
            )["path"],
        ),
        ("--work-parent-dir", run_root / "work-parent"),
        ("--code-output-root", code_run["code_output_root"]),
        ("--commit-output-root", code_run["commit_output_root"]),
        ("--conveyor-root", run_root / "conveyor"),
        ("--dedup-db", code_run["dedup_db"]),
        ("--progress-jsonl", run_root / "conveyor" / "progress.jsonl"),
        ("--completion-receipt", run_root / "conveyor" / "completion_receipt.json"),
        ("--run-lock-dir", run_root / "conveyor" / "locks"),
        ("--reservation-file", run_root / "conveyor" / "reservations.json"),
        ("--repo-workers", args.repo_workers),
        ("--max-active-repos", args.max_active_repos),
        ("--workers", args.workers),
        ("--memory-limit-gb", args.memory_limit_gb),
        ("--commit-memory-limit-gb", args.memory_limit_gb),
        ("--min-free-disk-gb", args.minimum_free_bytes / 1024**3),
    )
    return [
        str(_object(inputs["python"], label="Python")["path"]),
        "-u",
        "scripts/streaming_conveyor.py",
        *(item for option in options for item in (option[0], str(option[1]))),
    ]


def _write_composition_plan(
    path: Path,
    *,
    code_run: Mapping[str, Any],
    commit_run_root: Path,
    dedup_receipt: Path,
) -> None:
    code_runs = [
        {
            "run_id": "full-code",
            "launch_receipt": str(code_run["launch_path"]),
            "exit_receipt": str(code_run["exit_path"]),
            "manifest": str(code_run["manifest_path"]),
        },
        *(
            {
                "run_id": repair["run_id"],
                "launch_receipt": str(repair["launch_path"]),
                "exit_receipt": str(repair["exit_path"]),
                "manifest": str(repair["manifest_path"]),
            }
            for repair in code_run.get("repair_runs", ())
        ),
    ]
    source_supervisor._atomic_json(
        path,
        {
            "schema": SOURCE_COMPOSITION_PLAN_SCHEMA,
            "runs": [
                *code_runs,
                {
                    "run_id": "full-commits",
                    "launch_receipt": str(commit_run_root / "launch_receipt.json"),
                    "exit_receipt": str(commit_run_root / "exit_receipt.json"),
                    "manifest": str(commit_run_root / "conveyor" / "_done.json"),
                },
            ],
            "dedup_receipt": str(dedup_receipt),
        },
    )


def _run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    lock_stream = (run_root / "launch.lock").open("a+b")
    try:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("commit conveyor supervisor is already running") from exc

        repair_run_roots = tuple(Path(value) for value in args.code_repair_run_root)
        code_run = load_terminal_code_run(
            Path(args.code_run_root),
            repair_run_roots=repair_run_roots,
        )
        free_bytes = shutil.disk_usage(run_root).free
        if free_bytes < args.minimum_free_bytes:
            raise RuntimeError(
                f"insufficient free disk: {free_bytes} < {args.minimum_free_bytes}"
            )
        source_repo_list = Path(
            _object(code_run["inputs"], label="code inputs")["repo_list"]["path"]
        )
        pr_inputs, pr_snapshot, pr_paths = validate_pr_inputs(
            source_repo_list=source_repo_list,
            pr_store=Path(args.pr_store),
            pr_repo_list=Path(args.pr_repo_list),
            completion_receipt=Path(args.pr_completion_receipt),
        )
        command = build_command(args, code_run, pr_inputs)
        run_binding = {
            "schema": source_supervisor.RUN_BINDING_SCHEMA,
            "streams": "commits",
            "source_code_run": code_run["identity"],
            "source_code_runs": code_run.get("identities", [code_run["identity"]]),
            "pr_completion": pr_inputs["completion_binding"],
            "command": command,
        }
        launch_path = run_root / "launch_receipt.json"
        attempt = source_supervisor._resume_attempt(
            launch_path,
            run_binding=run_binding,
        )
        launch: dict[str, Any] = {
            "schema": source_supervisor.LAUNCH_SCHEMA,
            "status": "validated",
            "created_at": source_supervisor._utc_now(),
            "attempt": attempt,
            "repository_identity": "cppmega",
            "code_revision": code_run["launch"]["code_revision"],
            "repository_root": str(code_run["producer_root"]),
            "supervisor": {
                "path": str(Path(__file__).resolve()),
                "sha256": source_supervisor._sha256_file(Path(__file__).resolve()),
            },
            "source_code_run": code_run["identity"],
            "source_code_runs": code_run.get("identities", [code_run["identity"]]),
            "inputs": code_run["inputs"],
            "pr_inputs": pr_inputs,
            "outputs": {
                "run_root": str(run_root),
                "code_output_root": str(code_run["code_output_root"]),
                "commit_output_root": str(code_run["commit_output_root"]),
                "conveyor_manifest": str(run_root / "conveyor" / "_done.json"),
                "completion_receipt": str(
                    run_root / "conveyor" / "completion_receipt.json"
                ),
                "dedup_db": str(code_run["dedup_db"]),
            },
            "source_cache": {
                "enabled": False,
                "mode": "direct_verified_archive_stream",
            },
            "target_lengths": list(code_run["target_lengths"]),
            "expected_repository_count": len(code_run["repositories"]),
            "run_binding": run_binding,
            "run_binding_sha256": source_supervisor._canonical_sha256(run_binding),
            "command": command,
        }
        source_supervisor._atomic_json(launch_path, launch)

        def mark_started(child_pid: int) -> None:
            launch.update(
                status="running",
                started_at=source_supervisor._utc_now(),
                supervisor_pid=os.getpid(),
                child_pid=child_pid,
            )
            source_supervisor._atomic_json(launch_path, launch)

        manifest_path = run_root / "conveyor" / "_done.json"
        completion_path = run_root / "conveyor" / "completion_receipt.json"
        exit_path = run_root / "exit_receipt.json"
        return_code = source_supervisor.run_supervised_child(
            command,
            cwd=code_run["producer_root"],
            environment=source_supervisor.build_child_environment(code_run["inputs"]),
            log_path=run_root / "run.log",
            on_started=mark_started,
        )
        if return_code != 0:
            source_supervisor.write_exit_receipt(
                exit_path,
                launch_path=launch_path,
                code_revision=code_run["launch"]["code_revision"],
                return_code=return_code,
                manifest_path=manifest_path,
                completion_path=completion_path,
            )
            return return_code

        try:
            current_code_run = load_terminal_code_run(
                Path(args.code_run_root),
                repair_run_roots=repair_run_roots,
            )
            if current_code_run.get("identities", [current_code_run["identity"]]) != (
                code_run.get("identities", [code_run["identity"]])
            ):
                raise RuntimeError("terminal code run changed during commit extraction")
            pr_store, pr_repo_list, pr_completion = pr_paths
            conveyor.revalidate_pr_completion_binding(
                pr_inputs["completion_binding"],
                pr_completion,
                pr_store=pr_store,
                repo_list=pr_repo_list,
                repo_list_snapshot=pr_snapshot,
            )
            terminal_coverage = {
                "status": "complete",
                "expected_repository_count": len(code_run["repositories"]),
                "successful_repository_count": len(code_run["repositories"]),
                "immutable_inputs_reverified_at_finish": True,
            }
            source_supervisor.write_exit_receipt(
                exit_path,
                launch_path=launch_path,
                code_revision=code_run["launch"]["code_revision"],
                return_code=0,
                manifest_path=manifest_path,
                completion_path=completion_path,
                terminal_coverage=terminal_coverage,
            )
            dedup_receipt = run_root / "global_dedup_receipt.json"
            verify_global_dedup_store(
                code_run["dedup_db"],
                output_path=dedup_receipt,
                busy_timeout_seconds=args.dedup_busy_timeout_seconds,
            )
            plan_path = run_root / "source_composition_plan.json"
            _write_composition_plan(
                plan_path,
                code_run=current_code_run,
                commit_run_root=run_root,
                dedup_receipt=dedup_receipt,
            )
            composition = load_source_composition(
                plan_path,
                buckets=code_run["target_lengths"],
                code_root=code_run["code_output_root"],
                commit_root=code_run["commit_output_root"],
            )
            code_inventory_path = run_root / "code_packed_inventory.receipt.json"
            commit_inventory_path = (
                run_root / "commit_packed_inventory.receipt.json"
            )
            source_supervisor._atomic_json(
                code_inventory_path,
                build_packed_source_inventory_receipt(
                    composition,
                    kind="code",
                    input_root=code_run["code_output_root"],
                ),
            )
            source_supervisor._atomic_json(
                commit_inventory_path,
                build_packed_source_inventory_receipt(
                    composition,
                    kind="commits",
                    input_root=code_run["commit_output_root"],
                ),
            )
        except Exception:
            source_supervisor.write_exit_receipt(
                exit_path,
                launch_path=launch_path,
                code_revision=code_run["launch"]["code_revision"],
                return_code=2,
                manifest_path=manifest_path,
                completion_path=completion_path,
            )
            raise

        print(
            json.dumps(
                {
                    "status": "complete",
                    "run_root": str(run_root),
                    "composition_plan": str(plan_path),
                    "dedup_receipt": str(dedup_receipt),
                    "code_inventory_receipt": str(code_inventory_path),
                    "commit_inventory_receipt": str(commit_inventory_path),
                    "coverage": composition.receipt["coverage"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        lock_stream.close()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        return _run(args)
    except Exception as error:  # noqa: BLE001 - persist every supervisor failure
        run_root = Path(args.run_root).expanduser().resolve()
        try:
            source_supervisor._atomic_json(
                run_root / "supervisor_failure.json",
                {
                    "schema": source_supervisor.FAILURE_SCHEMA,
                    "failed_at": source_supervisor._utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        except OSError:
            pass
        print(f"commit conveyor supervisor failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
