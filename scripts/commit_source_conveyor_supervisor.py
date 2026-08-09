#!/usr/bin/env python3
"""Launch commits from a clean code run or a terminal failed code base."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import sys
import time
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
DEFAULT_REPAIR_POLL_SECONDS = 30.0


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
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} cannot be resolved: {path}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not a regular file: {resolved}")
    return resolved


def _recorded_code_revision(run_root: Path, *, label: str) -> str:
    launch, _digest = source_supervisor._read_json_snapshot(
        run_root.expanduser().resolve() / "launch_receipt.json",
        label=f"{label} launch receipt",
    )
    revision = launch.get("code_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError(f"{label} launch code revision is invalid")
    return revision


def _historical_code_revisions(
    code_run_root: Path,
    repair_run_roots: Sequence[Path],
) -> set[str]:
    roots = [code_run_root, *repair_run_roots]
    return {
        _recorded_code_revision(root, label=f"code run {index}")
        for index, root in enumerate(roots, start=1)
    }


def _assert_lock_available(path: Path, *, label: str) -> None:
    """Prove a prior supervisor/stream is no longer writing its state."""

    lock_path = _plain_file(path, label=label)
    try:
        stream = lock_path.open("rb")
    except OSError as exc:
        raise RuntimeError(f"cannot open {label}: {lock_path}") from exc
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"{label} is still held: {lock_path}") from exc
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _validate_resume_state_root(
    raw_root: Path,
    *,
    expected_revision: str,
    allow_from: str | None,
) -> dict[str, Any]:
    """Load and freeze the old commit state before a new supervisor starts."""

    root = raw_root.expanduser()
    if root.is_symlink():
        raise RuntimeError(f"resume state root must not be a symlink: {root}")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"resume state root is not a directory: {root}")
    launch_path = root / "launch_receipt.json"
    exit_path = root / "exit_receipt.json"
    manifest_path = root / "conveyor" / "_done.json"
    completion_path = root / "conveyor" / "completion_receipt.json"
    launch, launch_sha256 = source_supervisor._read_json_snapshot(
        launch_path,
        label="resume state launch receipt",
    )
    if launch.get("schema") != source_supervisor.LAUNCH_SCHEMA:
        raise RuntimeError("resume state is not a full commit supervisor run")
    command = launch.get("command")
    if not isinstance(command, list) or "commits" not in command:
        raise RuntimeError("resume state launch is not bound to the commit stream")
    outputs = _object(launch.get("outputs"), label="resume state outputs")
    if (
        Path(
            _text(
                outputs.get("conveyor_manifest"),
                label="resume state output manifest",
            )
        ).resolve()
        != manifest_path.resolve()
        or Path(
            _text(
                outputs.get("completion_receipt"),
                label="resume state output completion",
            )
        ).resolve()
        != completion_path.resolve()
    ):
        raise RuntimeError("resume state output paths drifted")
    binding = _object(launch.get("run_binding"), label="resume state run binding")
    binding_sha256 = source_supervisor._require_sha256(
        launch.get("run_binding_sha256"),
        label="resume state run binding sha256",
    )
    if source_supervisor._canonical_sha256(binding) != binding_sha256:
        raise RuntimeError("resume state run binding digest drifted")
    exit_receipt = source_supervisor._read_json(
        exit_path,
        label="resume state exit receipt",
    )
    if exit_receipt.get("launch_receipt_sha256") != launch_sha256:
        raise RuntimeError("resume state exit receipt does not bind its launch")
    exit_code = exit_receipt.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code == 0:
        raise RuntimeError("resume state must have a non-zero terminal exit receipt")
    done_binding = _object(
        exit_receipt.get("done_manifest"),
        label="resume state done manifest binding",
    )
    if (
        Path(_text(done_binding.get("path"), label="resume state manifest path")).resolve()
        != manifest_path.resolve()
        or done_binding.get("sha256") != source_supervisor._sha256_file(manifest_path)
    ):
        raise RuntimeError("resume state manifest binding drifted")
    _assert_lock_available(root / "launch.lock", label="resume supervisor lock")
    _assert_lock_available(
        root / "conveyor" / "locks" / "commits.lock",
        label="resume commit stream lock",
    )
    manifest = conveyor.ConcurrentManifest.load(manifest_path)
    if not manifest.done and not manifest.failed:
        raise RuntimeError("resume state has no checkpointed commit work")
    manifest_revision = _object(
        manifest.code_revision,
        label="resume state manifest code revision",
    )
    manifest_commit = _text(
        manifest_revision.get("git_commit"),
        label="resume state manifest Git commit",
    )
    if manifest_commit == expected_revision:
        if allow_from is not None:
            if manifest.code_revision_upgrade is None:
                raise RuntimeError(
                    "resume state revision matches the target but has no "
                    "persisted migration authorization"
                )
            if manifest.code_revision_upgrade["from"]["git_commit"] != allow_from:
                raise RuntimeError(
                    "resume state upgrade source does not match authorization"
                )
    elif allow_from is None or manifest_commit != allow_from:
        raise RuntimeError(
            "resume state manifest revision does not match the authorized source"
        )
    if completion_path.exists():
        completion = source_supervisor._read_json(
            completion_path,
            label="resume state completion receipt",
        )
        if completion.get("status") == "success" and not manifest.failed:
            raise RuntimeError("resume state is already complete")
    return {
        "root": root,
        "launch": launch,
        "launch_sha256": launch_sha256,
        "exit": exit_receipt,
        "exit_sha256": source_supervisor._sha256_file(exit_path),
        "manifest_path": manifest_path,
        "completion_path": completion_path,
        "manifest_sha256": source_supervisor._sha256_file(manifest_path),
    }


def _revalidate_recorded_inputs(
    launch: dict[str, Any],
    *,
    run_root: Path,
    repo_root: Path,
    execution_code_revision: str | None = None,
    allowed_historical_code_revisions: set[str] | None = None,
) -> tuple[dict[str, Any], argparse.Namespace]:
    """Call the shared validator without widening legacy monkeypatch contracts."""

    kwargs: dict[str, Any] = {
        "run_root": run_root,
        "repo_root": repo_root,
    }
    if execution_code_revision is not None:
        kwargs["execution_code_revision"] = execution_code_revision
        kwargs["allowed_historical_code_revisions"] = (
            allowed_historical_code_revisions or set()
        )
    return source_supervisor.revalidate_recorded_inputs(launch, **kwargs)


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
    parser.add_argument(
        "--resume-from-run-root",
        default=None,
        help="Prior commit supervisor run root whose conveyor state, extract "
             "cache, work-parent and dedup outputs are resumed in this new run.",
    )
    parser.add_argument(
        "--expected-code-revision",
        default=None,
        help="Exact clean checkout revision used by the resumed commit conveyor. "
             "Required for a controlled code-revision upgrade.",
    )
    parser.add_argument(
        "--allow-code-revision-upgrade-from",
        default=None,
        help="Exact prior conveyor revision authorized for one manifest migration.",
    )
    parser.add_argument(
        "--code-revision-upgrade-reason",
        default=None,
        help="Printable audit reason for the one-time revision migration.",
    )
    parser.add_argument(
        "--code-revision-upgrade-authorized-at",
        default=None,
        help="Canonical UTC timestamp for the revision migration authorization.",
    )
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
    parser.add_argument(
        "--repair-poll-seconds",
        type=float,
        default=DEFAULT_REPAIR_POLL_SECONDS,
        help="Seconds between terminal repair receipt checks after commit extraction.",
    )
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    for name in ("expected_code_revision", "allow_code_revision_upgrade_from"):
        value = getattr(args, name)
        if value is not None and re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise SystemExit(
                f"--{name.replace('_', '-')} must be an exact lowercase "
                "40-character Git commit"
            )
    upgrade_from = args.allow_code_revision_upgrade_from
    upgrade_reason = args.code_revision_upgrade_reason
    authorized_at = args.code_revision_upgrade_authorized_at
    if (upgrade_from is None) != (upgrade_reason is None):
        raise SystemExit(
            "--allow-code-revision-upgrade-from and "
            "--code-revision-upgrade-reason must be provided together"
        )
    if upgrade_from is not None:
        if args.expected_code_revision is None:
            raise SystemExit(
                "--expected-code-revision is required for a code revision upgrade"
            )
        if authorized_at is None or not conveyor._canonical_utc_timestamp(authorized_at):
            raise SystemExit(
                "--code-revision-upgrade-authorized-at must be canonical UTC "
                "YYYY-MM-DDTHH:MM:SSZ"
            )
        if (
            not isinstance(upgrade_reason, str)
            or not 1 <= len(upgrade_reason) <= 200
            or any(ord(char) < 32 or ord(char) == 127 for char in upgrade_reason)
        ):
            raise SystemExit(
                "--code-revision-upgrade-reason must be 1-200 printable "
                "characters"
            )
    elif authorized_at is not None:
        raise SystemExit(
            "--code-revision-upgrade-authorized-at requires "
            "--allow-code-revision-upgrade-from"
        )
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
    if args.repair_poll_seconds <= 0:
        raise SystemExit("--repair-poll-seconds must be > 0")
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
    if args.resume_from_run_root is not None:
        resume_root = Path(args.resume_from_run_root).expanduser().resolve()
        if resume_root in reserved or resume_root in set(roots):
            raise SystemExit(
                "--resume-from-run-root must be separate from code and repair roots"
            )
        if resume_root == Path(args.run_root).expanduser().resolve():
            raise SystemExit("--resume-from-run-root must be separate from --run-root")
    return args


def load_terminal_code_run(
    code_run_root: Path,
    *,
    repair_run_roots: Sequence[Path] = (),
    execution_code_revision: str | None = None,
    allowed_historical_code_revisions: set[str] | None = None,
) -> dict[str, Any]:
    """Revalidate one full code run through the canonical production loaders."""

    if repair_run_roots:
        return load_terminal_code_run_chain(
            code_run_root,
            repair_run_roots,
            execution_code_revision=execution_code_revision,
            allowed_historical_code_revisions=allowed_historical_code_revisions,
        )

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
    live_inputs, validation_args = _revalidate_recorded_inputs(
        launch,
        run_root=root,
        repo_root=producer_root,
        execution_code_revision=execution_code_revision,
        allowed_historical_code_revisions=allowed_historical_code_revisions,
    )
    coverage = source_supervisor.verify_completion_receipt(
        completion_path,
        manifest_path=manifest_path,
        args=validation_args,
        inputs=(
            stored_inputs
            if execution_code_revision is not None
            else live_inputs
        ),
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
        "inputs": (
            live_inputs
            if execution_code_revision is not None
            else stored_inputs
        ),
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


def _load_failed_base_commit_metadata(
    code_run_root: Path,
    *,
    execution_code_revision: str | None = None,
    allowed_historical_code_revisions: set[str] | None = None,
) -> dict[str, Any]:
    """Load the immutable source metadata needed before code repair is terminal."""

    base = source_supervisor.load_repair_base_code_run(code_run_root)
    producer_root = Path(
        _text(base["launch"].get("repository_root"), label="base repository root")
    )
    if producer_root.is_symlink():
        raise RuntimeError(f"base repository root must not be a symlink: {producer_root}")
    producer_root = producer_root.resolve(strict=True)
    live_inputs, _validation_args = _revalidate_recorded_inputs(
        base["launch"],
        run_root=base["root"],
        repo_root=producer_root,
        execution_code_revision=execution_code_revision,
        allowed_historical_code_revisions=allowed_historical_code_revisions,
    )
    identity = dict(base["identity"])
    return {
        "root": base["root"],
        "launch_path": base["launch_path"],
        "exit_path": base["exit_path"],
        "manifest_path": base["manifest_path"],
        "launch": base["launch"],
        "inputs": (
            live_inputs
            if execution_code_revision is not None
            else base["inputs"]
        ),
        "producer_root": producer_root,
        "code_output_root": base["code_output_root"],
        "commit_output_root": base["commit_output_root"],
        "dedup_db": base["dedup_db"],
        "target_lengths": base["target_lengths"],
        "repositories": base["repositories"],
        "identity": identity,
        "identities": [identity],
        "repair_runs": [],
        "repair_required": True,
    }


def load_commit_source_run(
    code_run_root: Path,
    *,
    execution_code_revision: str | None = None,
    allowed_historical_code_revisions: set[str] | None = None,
) -> dict[str, Any]:
    """Load commit inputs from a clean code run or its terminal failed base."""

    root = code_run_root.expanduser().resolve()
    _read_launch, _launch_sha256 = source_supervisor._read_json_snapshot(
        root / "launch_receipt.json",
        label="code launch receipt",
    )
    exit_receipt = source_supervisor._read_json(
        root / "exit_receipt.json",
        label="code exit receipt",
    )
    exit_code = exit_receipt.get("exit_code")
    if exit_code == 0:
        result = load_terminal_code_run(
            root,
            execution_code_revision=execution_code_revision,
            allowed_historical_code_revisions=allowed_historical_code_revisions,
        )
        result["repair_required"] = False
        return result
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise RuntimeError("code exit receipt has an invalid exit code")
    return _load_failed_base_commit_metadata(
        root,
        execution_code_revision=execution_code_revision,
        allowed_historical_code_revisions=allowed_historical_code_revisions,
    )


def _repair_exit_code(root: Path) -> int | None:
    """Return a repair exit code once its immutable exit receipt is present."""

    root = root.expanduser()
    if root.is_symlink():
        raise RuntimeError(f"code repair run root must not be a symlink: {root}")
    exit_path = root / "exit_receipt.json"
    if not exit_path.is_file() or exit_path.is_symlink():
        return None
    try:
        receipt = source_supervisor._read_json(
            exit_path,
            label="code repair exit receipt",
        )
    except (OSError, TypeError, ValueError):
        return None
    value = receipt.get("exit_code")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _repair_exit_receipt_path(root: Path) -> Path:
    """Select an attested salvage only for a noncanonical original receipt."""

    original = root / "exit_receipt.json"
    salvaged = root / source_supervisor.SALVAGED_EXIT_FILENAME
    if not salvaged.exists():
        return original
    if salvaged.is_symlink() or not salvaged.is_file():
        raise RuntimeError(f"code repair salvaged exit is not a regular file: {salvaged}")
    original_receipt = source_supervisor._read_json(
        original,
        label="code repair original exit receipt",
    )
    if original_receipt.get("schema") == source_supervisor.TARGETED_EXIT_SCHEMA:
        raise RuntimeError("code repair has both canonical and salvaged exit receipts")
    return salvaged


def _validate_repair_run_roots(
    repair_run_roots: Sequence[Path],
) -> tuple[Path, ...]:
    """Validate repair root configuration before any long-running child starts."""

    validated: list[Path] = []
    for index, raw_root in enumerate(repair_run_roots, start=1):
        root = raw_root.expanduser()
        if root.is_symlink():
            raise RuntimeError(
                f"code repair {index} run root must not be a symlink: {root}"
            )
        try:
            root = root.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                f"code repair {index} run root does not exist: {root}"
            ) from exc
        if not root.is_dir():
            raise RuntimeError(
                f"code repair {index} run root is not a directory: {root}"
            )
        validated.append(root)
    return tuple(validated)


def wait_for_terminal_code_run(
    code_run_root: Path,
    repair_run_roots: Sequence[Path],
    *,
    poll_seconds: float = DEFAULT_REPAIR_POLL_SECONDS,
    sleeper: Any = time.sleep,
    execution_code_revision: str | None = None,
    allowed_historical_code_revisions: set[str] | None = None,
) -> dict[str, Any]:
    """Wait until the failed base has a fully validated terminal repair chain."""

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if not repair_run_roots:
        raise RuntimeError("failed base code run has no repair run roots")
    repair_run_roots = _validate_repair_run_roots(repair_run_roots)
    while True:
        try:
            return load_terminal_code_run(
                code_run_root,
                repair_run_roots=repair_run_roots,
                execution_code_revision=execution_code_revision,
                allowed_historical_code_revisions=allowed_historical_code_revisions,
            )
        except Exception:  # noqa: BLE001 - receipts may be mid-write
            exit_codes = [
                _repair_exit_code(Path(root)) for root in repair_run_roots
            ]
            if all(exit_code is not None for exit_code in exit_codes):
                raise
            sleeper(poll_seconds)


def load_terminal_code_run_chain(
    code_run_root: Path,
    repair_run_roots: Sequence[Path],
    *,
    execution_code_revision: str | None = None,
    allowed_historical_code_revisions: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a failed full-code run plus targeted repairs as one final corpus."""

    base = source_supervisor.load_repair_base_code_run(code_run_root)
    base_producer_root = Path(
        _text(base["launch"].get("repository_root"), label="base repository root")
    )
    if base_producer_root.is_symlink():
        raise RuntimeError(
            f"base repository root must not be a symlink: {base_producer_root}"
        )
    base_producer_root = base_producer_root.resolve(strict=True)
    base_live_inputs, _base_validation_args = _revalidate_recorded_inputs(
        base["launch"],
        run_root=base["root"],
        repo_root=base_producer_root,
        execution_code_revision=execution_code_revision,
        allowed_historical_code_revisions=allowed_historical_code_revisions,
    )
    if execution_code_revision is not None:
        base["inputs"] = base_live_inputs
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
        exit_path = _repair_exit_receipt_path(root)
        manifest_path = root / "conveyor" / "_done.json"
        completion_path = root / "conveyor" / "completion_receipt.json"
        launch, launch_sha256 = source_supervisor._read_json_snapshot(
            launch_path,
            label=f"code repair {index} launch receipt",
        )
        producer_root = Path(
            _text(
                launch.get("repository_root"),
                label=f"code repair {index} repository root",
            )
        )
        if producer_root.is_symlink():
            raise RuntimeError(
                f"code repair {index} repository root must not be a symlink: "
                f"{producer_root}"
            )
        producer_root = producer_root.resolve(strict=True)
        live_inputs, validation_args = _revalidate_recorded_inputs(
            launch,
            run_root=root,
            repo_root=producer_root,
            execution_code_revision=execution_code_revision,
            allowed_historical_code_revisions=allowed_historical_code_revisions,
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
        exit_code = int(portable["exit"]["exit_code"])
        if exit_code != 0 and index == len(repair_run_roots):
            raise RuntimeError("final code repair exit code is non-zero")
        selected = set(portable["selected_repositories"])
        if (
            portable["launch"]["sha256"] != launch_sha256
            or portable["launch"]["schema"]
            != source_supervisor.TARGETED_LAUNCH_SCHEMA
            or portable["streams"] != "code"
            or commit_terminal
            or not code_terminal
            or not code_terminal <= selected
            or (exit_code == 0 and code_terminal != selected)
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
        if exit_code != 0 and not repair_success:
            raise RuntimeError(
                f"code repair {index} exit code is non-zero and contributed "
                "no successful repositories"
            )
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
                args=validation_args,
                inputs=(
                    stored_inputs
                    if execution_code_revision is not None
                    else live_inputs
                ),
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
                "inputs": (
                    live_inputs
                    if execution_code_revision is not None
                    else stored_inputs
                ),
                "producer_root": producer_root,
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
    producer_root = execution["producer_root"]

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
    state_root = (
        Path(args.resume_from_run_root).expanduser().resolve()
        if args.resume_from_run_root
        else run_root
    )
    execution_code_revision = (
        args.expected_code_revision
        or _object(code_run["launch"], label="code launch")["code_revision"]
    )
    inputs = _object(code_run["inputs"], label="code inputs")
    options = (
        ("--streams", "commits"),
        (
            "--source-archive",
            _object(inputs["archive"], label="source archive")["resolved_path"],
        ),
        (
            "--expected-code-revision",
            execution_code_revision,
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
        ("--work-parent-dir", state_root / "work-parent"),
        ("--code-output-root", code_run["code_output_root"]),
        ("--commit-output-root", code_run["commit_output_root"]),
        ("--conveyor-root", state_root / "conveyor"),
        ("--dedup-db", code_run["dedup_db"]),
        ("--progress-jsonl", state_root / "conveyor" / "progress.jsonl"),
        ("--completion-receipt", state_root / "conveyor" / "completion_receipt.json"),
        ("--run-lock-dir", state_root / "conveyor" / "locks"),
        ("--reservation-file", state_root / "conveyor" / "reservations.json"),
        ("--repo-workers", args.repo_workers),
        ("--max-active-repos", args.max_active_repos),
        ("--workers", args.workers),
        ("--memory-limit-gb", args.memory_limit_gb),
        ("--commit-memory-limit-gb", args.memory_limit_gb),
        ("--min-free-disk-gb", args.minimum_free_bytes / 1024**3),
    )
    command = [
        str(_object(inputs["python"], label="Python")["path"]),
        "-u",
        "scripts/streaming_conveyor.py",
        *(item for option in options for item in (option[0], str(option[1]))),
    ]
    if args.allow_code_revision_upgrade_from is not None:
        command.extend(
            (
                "--allow-code-revision-upgrade-from",
                args.allow_code_revision_upgrade_from,
                "--code-revision-upgrade-reason",
                args.code_revision_upgrade_reason,
                "--code-revision-upgrade-authorized-at",
                args.code_revision_upgrade_authorized_at,
            )
        )
    return command


def _write_composition_plan(
    path: Path,
    *,
    code_run: Mapping[str, Any],
    commit_run_root: Path,
    dedup_receipt: Path,
    commit_state_root: Path | None = None,
) -> None:
    state_root = commit_state_root or commit_run_root
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
                    "manifest": str(state_root / "conveyor" / "_done.json"),
                },
            ],
            "dedup_receipt": str(dedup_receipt),
        },
    )


def _bind_terminal_code_runs(
    launch: dict[str, Any],
    *,
    code_run: Mapping[str, Any],
) -> None:
    """Bind the commit receipt to every code generation used by composition."""

    identities = list(code_run.get("identities", [code_run["identity"]]))
    if not identities:
        raise RuntimeError("terminal code run has no identities")
    # Keep the launch run_binding immutable so a supervisor can resume after
    # the repair chain becomes terminal; the receipt fields below carry the
    # complete composed provenance.
    launch["source_code_run"] = identities[0]
    launch["source_code_runs"] = identities


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
        code_run_root = Path(args.code_run_root).expanduser().resolve()
        base_code_revision = _recorded_code_revision(
            code_run_root,
            label="base code run",
        )
        execution_code_revision = args.expected_code_revision or base_code_revision
        if args.allow_code_revision_upgrade_from is not None and (
            args.resume_from_run_root is None
        ):
            raise RuntimeError(
                "code revision upgrade requires --resume-from-run-root so the "
                "old conveyor state remains immutable and auditable"
            )
        resume_state = None
        state_root = run_root
        if args.resume_from_run_root is not None:
            state_root = Path(args.resume_from_run_root).expanduser().resolve()
            resume_state = _validate_resume_state_root(
                state_root,
                expected_revision=execution_code_revision,
                allow_from=args.allow_code_revision_upgrade_from,
            )
        historical_revisions = {base_code_revision}
        if repair_run_roots:
            repair_run_roots = _validate_repair_run_roots(repair_run_roots)
            historical_revisions.update(
                _historical_code_revisions(code_run_root, repair_run_roots)
            )
        code_run = load_commit_source_run(
            code_run_root,
            execution_code_revision=(
                execution_code_revision
                if args.expected_code_revision is not None
                else None
            ),
            allowed_historical_code_revisions=historical_revisions,
        )
        if not code_run.get("repair_required") and repair_run_roots:
            raise RuntimeError(
                "code repair run roots are only valid for a failed base code run"
            )
        if code_run.get("repair_required"):
            repair_run_roots = _validate_repair_run_roots(repair_run_roots)
            historical_revisions.update(
                _historical_code_revisions(code_run_root, repair_run_roots)
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
            "execution_code_revision": execution_code_revision,
            "resume_state": (
                {
                    "root": str(state_root),
                    "launch_sha256": resume_state["launch_sha256"],
                    "exit_sha256": resume_state["exit_sha256"],
                }
                if resume_state is not None
                else None
            ),
            "code_revision_upgrade": (
                {
                    "from": args.allow_code_revision_upgrade_from,
                    "reason": args.code_revision_upgrade_reason,
                    "authorized_at": args.code_revision_upgrade_authorized_at,
                }
                if args.allow_code_revision_upgrade_from is not None
                else None
            ),
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
            "code_revision": execution_code_revision,
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
                "state_root": str(state_root),
                "conveyor_manifest": str(state_root / "conveyor" / "_done.json"),
                "completion_receipt": str(state_root / "conveyor" / "completion_receipt.json"),
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

        manifest_path = state_root / "conveyor" / "_done.json"
        completion_path = state_root / "conveyor" / "completion_receipt.json"
        exit_path = run_root / "exit_receipt.json"
        return_code = source_supervisor.run_supervised_child(
            command,
            cwd=code_run["producer_root"],
            environment=source_supervisor.build_child_environment(code_run["inputs"]),
            log_path=run_root / "run.log",
            on_started=mark_started,
        )
        if return_code != 0:
            persisted_upgrade = (
                conveyor.ConcurrentManifest.load(manifest_path).code_revision_upgrade
                if manifest_path.is_file()
                else None
            )
            source_supervisor.write_exit_receipt(
                exit_path,
                launch_path=launch_path,
                code_revision=execution_code_revision,
                return_code=return_code,
                manifest_path=manifest_path,
                completion_path=completion_path,
                code_revision_upgrade=persisted_upgrade,
            )
            return return_code

        try:
            if code_run.get("repair_required"):
                current_code_run = wait_for_terminal_code_run(
                    code_run_root,
                    repair_run_roots,
                    poll_seconds=args.repair_poll_seconds,
                    execution_code_revision=execution_code_revision,
                    allowed_historical_code_revisions=historical_revisions,
                )
            else:
                current_code_run = load_terminal_code_run(
                    code_run_root,
                    execution_code_revision=execution_code_revision
                    if args.expected_code_revision is not None
                    else None,
                    allowed_historical_code_revisions=historical_revisions,
                )
            initial_identity = code_run["identity"]
            current_identities = current_code_run.get(
                "identities",
                [current_code_run["identity"]],
            )
            if not current_identities or current_identities[0] != initial_identity:
                raise RuntimeError("base code run changed during commit extraction")
            _bind_terminal_code_runs(launch, code_run=current_code_run)
            source_supervisor._atomic_json(launch_path, launch)
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
            persisted_upgrade = conveyor.ConcurrentManifest.load(
                manifest_path
            ).code_revision_upgrade
            terminal_coverage["code_revision_upgrade"] = persisted_upgrade
            source_supervisor.write_exit_receipt(
                exit_path,
                launch_path=launch_path,
                code_revision=execution_code_revision,
                return_code=0,
                manifest_path=manifest_path,
                completion_path=completion_path,
                terminal_coverage=terminal_coverage,
                code_revision_upgrade=persisted_upgrade,
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
                commit_state_root=state_root,
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
                code_revision=execution_code_revision,
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
