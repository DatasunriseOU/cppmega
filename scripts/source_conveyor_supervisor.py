#!/usr/bin/env python3
"""Launch a revision-bound full or targeted source-code conveyor."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega.data.source_conveyor_composition import _load_run  # noqa: E402
from scripts import streaming_conveyor as conveyor  # noqa: E402

LAUNCH_SCHEMA = "cppmega.canonical_source_launch_v1"
EXIT_SCHEMA = "cppmega.canonical_source_exit_v1"
TARGETED_LAUNCH_SCHEMA = "cppmega.canonical_source_targeted_retry_launch_v1"
TARGETED_EXIT_SCHEMA = "cppmega.canonical_source_targeted_retry_exit_v1"
FAILURE_SCHEMA = "cppmega.canonical_source_supervisor_failure_v1"
ARCHIVE_SHA_SCHEMA = "cppmega.source_archive_sha256_verification_v1"
ARCHIVE_INVENTORY_SCHEMA = "cppmega.source_archive_inventory_binding_v1"
MAX_METADATA_BYTES = 4 * 1024 * 1024
DEFAULT_TARGET_LENGTHS = (1024, 2048, 4096, 8192, 16384)
DEFAULT_MINIMUM_FREE_BYTES = 50 * 1024**3
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
RUN_BINDING_SCHEMA = "cppmega.canonical_source_run_binding_v1"
COMPLETION_SCHEMA = "cppmega.source_conveyor_completion_v1"
PINNED_SOURCE_ENVIRONMENT = {
    "CPPMEGA_MACRO_INCLUDE_DEPTH": "0",
    "CPPMEGA_MACRO_INCLUDE_FILES_PER_ROOT": "0",
    "CPPMEGA_MACRO_DIRECTIVE_CACHE_ENTRIES": "4096",
    "CPPMEGA_MACRO_RESOLVE_CACHE_ENTRIES": "65536",
    "CPPMEGA_MAX_MACRO_CANDIDATES_PER_ROOT": "200000",
    "CPPMEGA_MAX_RETAINED_MACROS": "250000",
    "CPPMEGA_MAX_MACRO_VISIBILITY_BYTES": "262000000",
    "CPPMEGA_EXTRACT_BAD_UNIT_POLICY": "fail",
    "CPPMEGA_EXTRACT_MAX_BAD_UNITS": "0",
    "CPPMEGA_DEDUP_PROMOTE_LOCK_TIMEOUT_SECONDS": "600",
    "CPPMEGA_DEDUP_MAX_PENDING_BEFORE_COMMIT": "128",
    "CPPMEGA_DEDUP_WAL_AUTOCHECKPOINT_PAGES": "10000",
    "CPPMEGA_DEDUP_JOURNAL_SIZE_LIMIT_BYTES": str(1024**3),
}
_RECORDED_INPUT_BINDINGS = (
    "archive",
    "archive_sha256_receipt",
    "archive_inventory_receipt",
    "repo_list",
    "source_quarantine_manifest",
    "tokenizer",
    "python",
    "libclang",
    "code_revision",
)
_REMOVED_CHILD_ENVIRONMENT = frozenset(
    {
        "PYTHONHOME",
        "PYTHONPATH",
        "LIBCLANG_PATH",
        "CLANG_LIBRARY_PATH",
        "NANOCHAT_BASE_DIR",
        "NANOCHAT_COMPILE_COMMANDS",
        "NANOCHAT_LIBCLANG_PATH",
        "NANOCHAT_TOKENIZER_PATH",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    value, _digest = _read_json_snapshot(path, label=label)
    return value


def _read_json_snapshot(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise RuntimeError(
            f"{label} exceeds the {MAX_METADATA_BYTES}-byte metadata bound"
        )
    try:
        payload = path.read_bytes()
        if len(payload) > MAX_METADATA_BYTES:
            raise RuntimeError(
                f"{label} exceeds the {MAX_METADATA_BYTES}-byte metadata bound"
            )
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}")
    return value


def _regular_file(value: str, *, label: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} cannot be resolved: {value}") from exc
    if not path.is_file():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    return path


def _python_executable_binding(value: str) -> dict[str, Any]:
    launcher = Path(os.path.abspath(Path(value).expanduser()))
    if not launcher.is_file():
        raise RuntimeError(f"Python executable is not a regular file: {launcher}")
    if not os.access(launcher, os.X_OK):
        raise RuntimeError(f"Python executable is not executable: {launcher}")
    try:
        resolved_binary = launcher.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Python executable cannot be resolved: {launcher}") from exc
    venv_config = launcher.parent.parent / "pyvenv.cfg"
    return {
        "path": str(launcher),
        "resolved_binary_path": str(resolved_binary),
        "sha256": _sha256_file(resolved_binary),
        "venv_config": (
            {
                "path": str(venv_config),
                "sha256": _sha256_file(venv_config),
            }
            if venv_config.is_file()
            else None
        ),
    }


def _parse_target_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "target lengths must be comma-separated integers"
        ) from exc
    if (
        not lengths
        or any(length <= 0 for length in lengths)
        or tuple(sorted(set(lengths))) != lengths
    ):
        raise argparse.ArgumentTypeError(
            "target lengths must be unique positive integers in ascending order"
        )
    return lengths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--archive-sha256-receipt", required=True)
    parser.add_argument("--archive-inventory-receipt", required=True)
    parser.add_argument("--repo-list", required=True)
    parser.add_argument("--source-quarantine-manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--libclang", required=True)
    parser.add_argument(
        "--minimum-free-bytes",
        type=int,
        default=DEFAULT_MINIMUM_FREE_BYTES,
    )
    parser.add_argument(
        "--target-lengths",
        type=_parse_target_lengths,
        default=DEFAULT_TARGET_LENGTHS,
    )
    parser.add_argument("--repo-workers", type=int, default=1)
    parser.add_argument("--max-active-repos", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parse-workers", type=int, default=1)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--code-index-timeout-s", type=int, default=0)
    parser.add_argument("--code-index-stall-timeout-s", type=int, default=0)
    parser.add_argument(
        "--repair-base-code-run-root",
        help="Terminal failed full-code run whose outputs and dedup DB are reused.",
    )
    parser.add_argument(
        "--only-repo",
        action="append",
        default=[],
        help="Failed bare repository name to repair (repeatable).",
    )
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    if re.fullmatch(r"[0-9a-f]{40}", args.expected_code_revision) is None:
        raise SystemExit(
            "--expected-code-revision must be an exact lowercase "
            "40-character Git commit"
        )
    if args.minimum_free_bytes < 0:
        raise SystemExit("--minimum-free-bytes must be >= 0")
    for name in ("repo_workers", "max_active_repos", "workers", "parse_workers"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be > 0")
    if args.max_active_repos < args.repo_workers:
        raise SystemExit("--max-active-repos must be >= --repo-workers")
    if args.memory_limit_gb <= 0:
        raise SystemExit("--memory-limit-gb must be > 0")
    for name in ("code_index_timeout_s", "code_index_stall_timeout_s"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")
    if bool(args.repair_base_code_run_root) != bool(args.only_repo):
        raise SystemExit(
            "--repair-base-code-run-root and at least one --only-repo "
            "must be provided together"
        )
    if len(args.only_repo) != len(set(args.only_repo)) or any(
        not repository for repository in args.only_repo
    ):
        raise SystemExit("--only-repo values must be unique non-empty names")
    args.only_repo = sorted(args.only_repo)
    if args.repair_base_code_run_root and (
        Path(args.repair_base_code_run_root).expanduser().resolve()
        == Path(args.run_root).expanduser().resolve()
    ):
        raise SystemExit("--run-root must be separate from the repair base run")
    return args


def validate_inputs(
    args: argparse.Namespace,
    *,
    repo_root: Path = _REPO_ROOT,
    enforce_minimum_free_disk: bool = True,
) -> dict[str, Any]:
    archive = _regular_file(args.archive, label="source archive")
    archive_sha_receipt_path = _regular_file(
        args.archive_sha256_receipt,
        label="archive SHA-256 receipt",
    )
    archive_inventory_path = _regular_file(
        args.archive_inventory_receipt,
        label="archive inventory receipt",
    )
    repo_list = _regular_file(args.repo_list, label="canonical repo list")
    quarantine = _regular_file(
        args.source_quarantine_manifest,
        label="source quarantine manifest",
    )
    tokenizer = _regular_file(args.tokenizer, label="tokenizer")
    python_binding = _python_executable_binding(args.python)
    libclang = _regular_file(args.libclang, label="libclang")

    expected_tokenizer = (
        repo_root / "cppmega" / "tokenizer" / "tokenizer.json"
    ).resolve(strict=True)
    if tokenizer != expected_tokenizer:
        raise RuntimeError(
            "tokenizer differs from the conveyor runtime tokenizer: "
            f"{tokenizer} != {expected_tokenizer}"
        )

    revision_guard = conveyor.CodeRevisionGuard.for_production(
        args.expected_code_revision,
        repo_root=repo_root,
    )
    archive_receipt, archive_receipt_digest = _read_json_snapshot(
        archive_sha_receipt_path,
        label="archive SHA-256 receipt",
    )
    inventory, inventory_digest = _read_json_snapshot(
        archive_inventory_path,
        label="archive inventory receipt",
    )
    if (
        archive_receipt.get("schema") != ARCHIVE_SHA_SCHEMA
        or archive_receipt.get("status") != "verified"
        or archive_receipt.get("exit_code") != 0
    ):
        raise RuntimeError("source archive SHA-256 receipt is not verified")
    if (
        inventory.get("schema") != ARCHIVE_INVENTORY_SCHEMA
        or inventory.get("status") != "verified"
    ):
        raise RuntimeError("source archive inventory receipt is not verified")

    receipt_archive = (
        Path(str(archive_receipt.get("resolved_path", "")))
        .expanduser()
        .resolve(strict=True)
    )
    if receipt_archive != archive:
        raise RuntimeError(
            f"source archive path differs from its verified receipt: "
            f"{archive} != {receipt_archive}"
        )
    archive_stat = archive.stat()
    expected_stat = {
        "size_bytes": _require_int(
            archive_receipt.get("size_bytes"),
            label="archive receipt size_bytes",
            minimum=1,
        ),
        "mtime_epoch": _require_int(
            archive_receipt.get("mtime_epoch"),
            label="archive receipt mtime_epoch",
        ),
        "inode": _require_int(
            archive_receipt.get("inode"),
            label="archive receipt inode",
            minimum=1,
        ),
        "device": _require_int(
            archive_receipt.get("device"),
            label="archive receipt device",
            minimum=1,
        ),
    }
    actual_stat = {
        "size_bytes": archive_stat.st_size,
        "mtime_epoch": int(archive_stat.st_mtime),
        "inode": archive_stat.st_ino,
        "device": archive_stat.st_dev,
    }
    if actual_stat != expected_stat:
        raise RuntimeError(
            f"source archive identity changed: expected={expected_stat} "
            f"actual={actual_stat}"
        )

    inventory_archive = inventory.get("archive_sha256_receipt")
    if (
        not isinstance(inventory_archive, dict)
        or not isinstance(inventory_archive.get("path"), str)
        or not inventory_archive["path"]
        or inventory_archive.get("sha256") != archive_receipt_digest
    ):
        raise RuntimeError("archive inventory does not bind the live SHA receipt")

    repo_snapshot = conveyor.load_repo_list_snapshot(repo_list, role="source")
    repo_list_digest = repo_snapshot.sha256
    canonical_repo_list = inventory.get("canonical_repo_list")
    if (
        not isinstance(canonical_repo_list, dict)
        or not isinstance(canonical_repo_list.get("path"), str)
        or not canonical_repo_list["path"]
        or canonical_repo_list.get("sha256") != repo_list_digest
    ):
        raise RuntimeError("canonical repo list differs from archive inventory")
    if (
        _require_int(
            canonical_repo_list.get("archive_repos_without_mapping"),
            label="archive_repos_without_mapping",
        )
        != 0
    ):
        raise RuntimeError("archive inventory contains repositories without identity")

    mapping_count = _require_int(
        canonical_repo_list.get("mapping_entry_count"),
        label="canonical repo-list mapping_entry_count",
        minimum=1,
    )
    if repo_snapshot.mapping_count != mapping_count:
        raise RuntimeError(
            "canonical repo-list mapping count differs from archive inventory"
        )
    repository_count = _require_int(
        inventory.get("archive_unique_worktree_repo_count"),
        label="archive_unique_worktree_repo_count",
        minimum=1,
    )
    repository_names_sha256 = _require_sha256(
        inventory.get("archive_sorted_repo_names_json_sha256"),
        label="archive_sorted_repo_names_json_sha256",
    )
    streaming_contract = inventory.get("streaming_contract")
    if not isinstance(streaming_contract, dict):
        raise TypeError("archive inventory has no streaming contract")
    if (
        streaming_contract.get("expected_attempted_repo_count") != repository_count
        or streaming_contract.get("persistent_source_cache") is not False
        or streaming_contract.get("one_repo_materialized_at_a_time") is not True
    ):
        raise RuntimeError("archive inventory streaming contract is inconsistent")

    run_root = Path(args.run_root).expanduser().resolve()
    free_bytes = shutil.disk_usage(run_root).free
    if enforce_minimum_free_disk and free_bytes < args.minimum_free_bytes:
        raise RuntimeError(
            f"insufficient free disk: {free_bytes} bytes "
            f"< {args.minimum_free_bytes} bytes"
        )

    return {
        "archive": {
            "requested_path": str(Path(args.archive).expanduser()),
            "resolved_path": str(archive),
            "sha256": _require_sha256(
                archive_receipt.get("sha256"),
                label="archive receipt sha256",
            ),
            **actual_stat,
            "verification_mode": ("prior_full_sha256_plus_current_stat_identity"),
        },
        "archive_sha256_receipt": {
            "path": str(archive_sha_receipt_path),
            "sha256": archive_receipt_digest,
        },
        "archive_inventory_receipt": {
            "path": str(archive_inventory_path),
            "sha256": inventory_digest,
            "repository_count": repository_count,
            "repository_names_sha256": repository_names_sha256,
            "unmapped_repository_count": 0,
        },
        "repo_list": {
            "path": str(repo_list),
            "sha256": repo_list_digest,
            "mapping_entry_count": repo_snapshot.mapping_count,
            "canonical_mapping_sha256": (repo_snapshot.canonical_mapping_sha256),
        },
        "source_quarantine_manifest": {
            "path": str(quarantine),
            "sha256": _sha256_file(quarantine),
        },
        "tokenizer": {
            "path": str(tokenizer),
            "sha256": _sha256_file(tokenizer),
        },
        "python": python_binding,
        "libclang": {
            "path": str(libclang),
            "sha256": _sha256_file(libclang),
        },
        "code_revision": revision_guard.receipt,
        "free_disk_bytes": free_bytes,
    }


def load_repair_base_code_run(code_run_root: Path) -> dict[str, Any]:
    """Validate one terminal failed full-code run used by targeted repairs."""

    root = code_run_root.expanduser()
    if root.is_symlink():
        raise RuntimeError(f"repair base run root must not be a symlink: {root}")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"repair base run root cannot be resolved: {root}") from exc
    if not root.is_dir():
        raise RuntimeError(f"repair base run root is not a directory: {root}")

    launch_path = root / "launch_receipt.json"
    exit_path = root / "exit_receipt.json"
    manifest_path = root / "conveyor" / "_done.json"
    launch, launch_sha256 = _read_json_snapshot(
        launch_path,
        label="repair base launch receipt",
    )
    outputs = launch.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError("repair base launch outputs must be an object")
    lengths = launch.get("target_lengths")
    if (
        not isinstance(lengths, list)
        or not lengths
        or any(isinstance(value, bool) or not isinstance(value, int) for value in lengths)
        or tuple(sorted(set(lengths))) != tuple(lengths)
    ):
        raise RuntimeError("repair base target lengths are invalid")

    def output_path(name: str) -> Path:
        value = outputs.get(name)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"repair base {name} is missing")
        path = Path(value).expanduser()
        if path.is_symlink():
            raise RuntimeError(f"repair base {name} must not be a symlink: {path}")
        return path.resolve()

    code_output_root = output_path("code_output_root")
    commit_output_root = output_path("commit_output_root")
    dedup_db = output_path("dedup_db")
    if not dedup_db.is_file():
        raise RuntimeError(f"repair base dedup database is not a regular file: {dedup_db}")
    (
        portable,
        allowlist,
        _files,
        code_terminal,
        commit_terminal,
        failed_repositories,
        failed_units,
        archive_identity,
        input_binding,
        details,
    ) = _load_run(
        {
            "run_id": "repair-base-code",
            "launch_receipt": str(launch_path),
            "exit_receipt": str(exit_path),
            "manifest": str(manifest_path),
        },
        buckets=tuple(lengths),
        code_root=code_output_root,
        commit_root=commit_output_root,
    )
    inventory = details["inventory"]
    if not isinstance(inventory, dict):
        raise TypeError("repair base archive inventory is invalid")
    expected_count = _require_int(
        inventory.get("archive_unique_worktree_repo_count"),
        label="repair base archive repository count",
        minimum=1,
    )
    expected_names_sha256 = _require_sha256(
        inventory.get("archive_sorted_repo_names_json_sha256"),
        label="repair base archive repository names sha256",
    )
    if (
        portable["launch"]["schema"] != LAUNCH_SCHEMA
        or portable["streams"] != "code"
        or portable["exit"]["exit_code"] == 0
        or commit_terminal
        or not failed_repositories
        or not failed_units
        or launch.get("expected_repository_count") != expected_count
        or len(code_terminal) != expected_count
        or _canonical_sha256(sorted(code_terminal)) != expected_names_sha256
    ):
        raise RuntimeError("repair base is not a terminal failed full code run")
    inputs = launch.get("inputs")
    if not isinstance(inputs, dict):
        raise TypeError("repair base launch inputs must be an object")
    run_binding = launch.get("run_binding")
    if not isinstance(run_binding, dict):
        raise TypeError("repair base run binding must be an object")
    run_binding_sha256 = _require_sha256(
        launch.get("run_binding_sha256"),
        label="repair base run binding sha256",
    )
    if _canonical_sha256(run_binding) != run_binding_sha256:
        raise RuntimeError("repair base run binding digest drifted")
    identity = {
        "launch_sha256": launch_sha256,
        "exit_sha256": portable["exit"]["sha256"],
        "manifest_sha256": portable["manifest"]["sha256"],
    }
    done = details["done"]
    if not isinstance(done, dict):
        raise TypeError("repair base done manifest must be an object")
    return {
        "root": root,
        "launch_path": launch_path,
        "exit_path": exit_path,
        "manifest_path": manifest_path,
        "launch": launch,
        "inputs": inputs,
        "identity": identity,
        "archive_identity": archive_identity,
        "input_binding": input_binding,
        "code_output_root": code_output_root,
        "commit_output_root": commit_output_root,
        "dedup_db": dedup_db,
        "target_lengths": tuple(lengths),
        "repositories": tuple(sorted(code_terminal)),
        "failed_repositories": tuple(sorted(failed_repositories)),
        "successful_repositories": tuple(
            sorted(
                unit[: -len("::code")]
                for unit in done
                if unit.endswith("::code")
            )
        ),
        "code_artifacts": tuple(
            sorted(
                (bucket, filename)
                for (kind, bucket), files in allowlist.items()
                if kind == "code"
                for filename in files
            )
        ),
    }


def validate_repair_request(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    repair_base: dict[str, Any],
) -> None:
    """Require a repair to preserve every immutable input except quarantine policy."""

    base_inputs = repair_base["inputs"]
    for name in (
        "archive_sha256_receipt",
        "archive_inventory_receipt",
        "repo_list",
        "tokenizer",
    ):
        if base_inputs[name]["sha256"] != inputs[name]["sha256"]:
            raise RuntimeError(f"repair {name} differs from its base code run")
    if base_inputs["archive"]["sha256"] != inputs["archive"]["sha256"]:
        raise RuntimeError("repair source archive differs from its base code run")
    if tuple(args.target_lengths) != repair_base["target_lengths"]:
        raise RuntimeError("repair target lengths differ from its base code run")
    selected = set(args.only_repo)
    failed = set(repair_base["failed_repositories"])
    if not selected <= failed:
        raise RuntimeError(
            "targeted repair includes repositories not failed by its base run: "
            + ", ".join(sorted(selected - failed))
        )


def _output_paths(
    args: argparse.Namespace,
    repair_base: dict[str, Any] | None,
) -> dict[str, Path]:
    run_root = Path(args.run_root).expanduser().resolve()
    return {
        "code_output_root": (
            repair_base["code_output_root"]
            if repair_base is not None
            else run_root / "reindexed"
        ),
        "commit_output_root": (
            repair_base["commit_output_root"]
            if repair_base is not None
            else run_root / "reindexed-commits"
        ),
        "dedup_db": (
            repair_base["dedup_db"]
            if repair_base is not None
            else run_root / "dedup.sqlite"
        ),
    }


def build_command(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    repair_base: dict[str, Any] | None = None,
) -> list[str]:
    run_root = Path(args.run_root).expanduser().resolve()
    outputs = _output_paths(args, repair_base)
    lengths = ",".join(str(length) for length in args.target_lengths)
    minimum_free_gib = args.minimum_free_bytes / 1024**3
    command = [
        str(inputs["python"]["path"]),
        "-u",
        "scripts/streaming_conveyor.py",
        "--streams",
        "code",
        "--source-archive",
        str(inputs["archive"]["resolved_path"]),
        "--expected-code-revision",
        args.expected_code_revision,
        "--target-lengths-code",
        lengths,
        "--repo-list",
        str(inputs["repo_list"]["path"]),
        "--source-quarantine-manifest",
        str(inputs["source_quarantine_manifest"]["path"]),
        "--work-parent-dir",
        str(run_root / "work-parent"),
        "--code-output-root",
        str(outputs["code_output_root"]),
        "--commit-output-root",
        str(outputs["commit_output_root"]),
        "--conveyor-root",
        str(run_root / "conveyor"),
        "--dedup-db",
        str(outputs["dedup_db"]),
        "--progress-jsonl",
        str(run_root / "conveyor" / "progress.jsonl"),
        "--completion-receipt",
        str(run_root / "conveyor" / "completion_receipt.json"),
        "--run-lock-dir",
        str(run_root / "conveyor" / "locks"),
        "--reservation-file",
        str(run_root / "conveyor" / "reservations.json"),
        "--repo-workers",
        str(args.repo_workers),
        "--max-active-repos",
        str(args.max_active_repos),
        "--workers",
        str(args.workers),
        "--parse-workers",
        str(args.parse_workers),
        "--memory-limit-gb",
        str(args.memory_limit_gb),
        "--code-memory-limit-gb",
        str(args.memory_limit_gb),
        "--code-index-timeout-s",
        str(args.code_index_timeout_s),
        "--code-index-stall-timeout-s",
        str(args.code_index_stall_timeout_s),
        "--min-free-disk-gb",
        str(minimum_free_gib),
    ]
    for repository in args.only_repo:
        command.extend(("--only-repo", repository))
    if args.only_repo:
        command.extend(("--max-repos", str(len(args.only_repo))))
    return command


def build_run_binding(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    repair_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    binding = {
        "schema": RUN_BINDING_SCHEMA,
        "streams": "code",
        "code_revision": args.expected_code_revision,
        "archive_sha256": inputs["archive"]["sha256"],
        "archive_sha256_receipt_sha256": (inputs["archive_sha256_receipt"]["sha256"]),
        "archive_inventory_receipt_sha256": (
            inputs["archive_inventory_receipt"]["sha256"]
        ),
        "repository_count": inputs["archive_inventory_receipt"]["repository_count"],
        "repository_names_sha256": inputs["archive_inventory_receipt"][
            "repository_names_sha256"
        ],
        "repo_list_sha256": inputs["repo_list"]["sha256"],
        "repo_list_canonical_mapping_sha256": inputs["repo_list"][
            "canonical_mapping_sha256"
        ],
        "source_quarantine_manifest_sha256": inputs["source_quarantine_manifest"][
            "sha256"
        ],
        "tokenizer_sha256": inputs["tokenizer"]["sha256"],
        "python_sha256": inputs["python"]["sha256"],
        "python_launcher_path": inputs["python"]["path"],
        "python_resolved_binary_path": inputs["python"]["resolved_binary_path"],
        "python_venv_config": inputs["python"]["venv_config"],
        "libclang_sha256": inputs["libclang"]["sha256"],
        "target_lengths": list(args.target_lengths),
        "dedup_policy": "exact_plus_near_default",
        "pinned_source_environment": dict(PINNED_SOURCE_ENVIRONMENT),
        "execution_policy": {
            "repo_workers": args.repo_workers,
            "max_active_repos": args.max_active_repos,
            "workers": args.workers,
            "parse_workers": args.parse_workers,
            "memory_limit_gb": args.memory_limit_gb,
            "code_index_timeout_s": args.code_index_timeout_s,
            "code_index_stall_timeout_s": args.code_index_stall_timeout_s,
            "minimum_free_bytes": args.minimum_free_bytes,
            "resume": True,
            "persistent_source_cache": False,
        },
    }
    if repair_base is not None:
        binding["repair_base_code_run"] = repair_base["identity"]
        binding["selected_repositories"] = list(args.only_repo)
    return binding


def build_launch_receipt(
    args: argparse.Namespace,
    *,
    inputs: dict[str, Any],
    command: list[str],
    run_binding: dict[str, Any],
    attempt: int,
    repair_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the persisted launch contract for a full or targeted code run."""

    run_root = Path(args.run_root).expanduser().resolve()
    outputs = _output_paths(args, repair_base)
    receipt: dict[str, Any] = {
        "schema": TARGETED_LAUNCH_SCHEMA if repair_base else LAUNCH_SCHEMA,
        "status": "validated",
        "created_at": _utc_now(),
        "attempt": attempt,
        "repository_identity": "cppmega",
        "code_revision": args.expected_code_revision,
        "repository_root": str(_REPO_ROOT),
        "supervisor": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "inputs": inputs,
        "outputs": {
            "run_root": str(run_root),
            "code_output_root": str(outputs["code_output_root"]),
            "commit_output_root": str(outputs["commit_output_root"]),
            "conveyor_manifest": str(run_root / "conveyor" / "_done.json"),
            "completion_receipt": str(
                run_root / "conveyor" / "completion_receipt.json"
            ),
            "dedup_db": str(outputs["dedup_db"]),
        },
        "source_cache": {
            "enabled": False,
            "mode": "direct_verified_archive_stream",
        },
        "target_lengths": list(args.target_lengths),
        "run_binding": run_binding,
        "run_binding_sha256": _canonical_sha256(run_binding),
        "command": command,
    }
    if repair_base is None:
        receipt["expected_repository_count"] = inputs[
            "archive_inventory_receipt"
        ]["repository_count"]
    else:
        receipt["repair_base_code_run"] = repair_base["identity"]
        receipt["selected_repositories"] = list(args.only_repo)
        receipt["expected_selected_repository_count"] = len(args.only_repo)
    return receipt


def _resume_attempt(
    launch_path: Path,
    *,
    run_binding: dict[str, Any],
) -> int:
    if not launch_path.exists():
        allowed_names = {"launch.lock", "supervisor_failure.json"}
        unexpected = sorted(
            path.name
            for path in launch_path.parent.iterdir()
            if path.name not in allowed_names
        )
        if unexpected:
            raise RuntimeError(
                "first launch requires a dedicated new run root; "
                f"found existing artifacts: {', '.join(unexpected)}"
            )
        return 1
    existing = _read_json(launch_path, label="existing launch receipt")
    if existing.get("schema") not in {LAUNCH_SCHEMA, TARGETED_LAUNCH_SCHEMA}:
        raise RuntimeError("existing run root has an unsupported launch receipt")
    existing_binding = existing.get("run_binding")
    if not isinstance(existing_binding, dict):
        raise TypeError("existing launch receipt has no immutable run binding")
    existing_digest = _require_sha256(
        existing.get("run_binding_sha256"),
        label="existing run binding sha256",
    )
    if _canonical_sha256(existing_binding) != existing_digest:
        raise RuntimeError("existing launch receipt run binding is corrupt")
    current_digest = _canonical_sha256(run_binding)
    if existing_digest != current_digest:
        raise RuntimeError(
            "existing run root is bound to different immutable inputs; "
            "use a new run root"
        )
    return (
        _require_int(
            existing.get("attempt"),
            label="existing launch attempt",
            minimum=1,
        )
        + 1
    )


def _portable_exit_code(return_code: int) -> int:
    return 128 + abs(return_code) if return_code < 0 else return_code


def build_child_environment(
    inputs: dict[str, Any],
    *,
    ambient: os._Environ[str] | dict[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if ambient is None else ambient
    environment = {
        key: value
        for key, value in source.items()
        if not key.startswith("CPPMEGA_") and key not in _REMOVED_CHILD_ENVIRONMENT
    }
    environment.update(PINNED_SOURCE_ENVIRONMENT)
    environment["NANOCHAT_LIBCLANG_PATH"] = str(inputs["libclang"]["path"])
    environment["NANOCHAT_TOKENIZER_PATH"] = str(inputs["tokenizer"]["path"])
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def revalidate_recorded_inputs(
    launch: dict[str, Any],
    *,
    run_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], argparse.Namespace]:
    """Re-run production input validation for an existing launch receipt."""

    stored = launch.get("inputs")
    if not isinstance(stored, dict):
        raise TypeError("source launch inputs must be an object")

    def field(name: str, key: str) -> object:
        binding = stored.get(name)
        if not isinstance(binding, dict) or key not in binding:
            raise RuntimeError(f"source launch {name}.{key} is missing")
        return binding[key]

    args = argparse.Namespace(
        archive=field("archive", "resolved_path"),
        archive_sha256_receipt=field("archive_sha256_receipt", "path"),
        archive_inventory_receipt=field("archive_inventory_receipt", "path"),
        repo_list=field("repo_list", "path"),
        source_quarantine_manifest=field("source_quarantine_manifest", "path"),
        tokenizer=field("tokenizer", "path"),
        python=field("python", "path"),
        libclang=field("libclang", "path"),
        expected_code_revision=launch.get("code_revision"),
        run_root=str(run_root),
        minimum_free_bytes=0,
        only_repo=list(launch.get("selected_repositories", [])),
    )
    live = validate_inputs(
        args,
        repo_root=repo_root,
        enforce_minimum_free_disk=False,
    )

    def identity(inputs: dict[str, Any]) -> dict[str, Any]:
        result = {name: dict(inputs[name]) for name in _RECORDED_INPUT_BINDINGS}
        result["archive"].pop("requested_path", None)
        return result

    if _canonical_sha256(identity(live)) != _canonical_sha256(identity(stored)):
        raise RuntimeError("source launch inputs drifted")
    return live, args


def _spawn_child(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log: Any,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _signal_child_group(
    child: subprocess.Popen[bytes],
    signum: int,
) -> None:
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def _forward_child_signal(
    child: subprocess.Popen[bytes],
    signum: int,
) -> None:
    if child.poll() is None:
        child.send_signal(signum)


def _terminate_child_group(
    child: subprocess.Popen[bytes],
    *,
    grace_seconds: float = 30.0,
) -> None:
    _signal_child_group(child, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except (PermissionError, ProcessLookupError):
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    else:
        _signal_child_group(child, signal.SIGKILL)
    if child.poll() is None:
        child.wait()


def run_supervised_child(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    on_started: Callable[[int], None],
) -> int:
    """Run one process group with the canonical two-stage signal policy."""

    child: subprocess.Popen[bytes] | None = None
    forwarded = 0

    def forward(signum: int, _frame: object) -> None:
        nonlocal forwarded
        if child is not None:
            forwarded += 1
            if forwarded == 1:
                _forward_child_signal(child, signum)
            else:
                _signal_child_group(child, signum)

    previous_sigint = signal.signal(signal.SIGINT, forward)
    previous_sigterm = signal.signal(signal.SIGTERM, forward)
    try:
        with log_path.open("ab", buffering=0) as log:
            child = _spawn_child(
                command,
                cwd=cwd,
                environment=environment,
                log=log,
            )
            try:
                on_started(child.pid)
                return _portable_exit_code(child.wait())
            except BaseException:
                _terminate_child_group(child)
                raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def write_exit_receipt(
    path: Path,
    *,
    launch_path: Path,
    code_revision: str,
    return_code: int,
    manifest_path: Path,
    completion_path: Path,
    terminal_coverage: dict[str, Any] | None = None,
    schema: str = EXIT_SCHEMA,
    selected_repositories: list[str] | None = None,
    repair_base_code_run: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write the canonical exit binding shared by source supervisors."""

    receipt: dict[str, Any] = {
        "schema": schema,
        "finished_at": _utc_now(),
        "exit_code": return_code,
        "status": "success" if return_code == 0 else "failed",
        "code_revision": code_revision,
        "launch_receipt_sha256": _sha256_file(launch_path),
        "done_manifest": None,
        "completion_receipt": None,
        "terminal_coverage": terminal_coverage,
    }
    if schema == TARGETED_EXIT_SCHEMA:
        if not selected_repositories or repair_base_code_run is None:
            raise RuntimeError("targeted exit requires selection and repair-base binding")
        receipt["selected_repositories"] = selected_repositories
        receipt["repair_base_code_run"] = repair_base_code_run
    elif selected_repositories is not None or repair_base_code_run is not None:
        raise RuntimeError("full exit must not contain targeted repair fields")
    for field, artifact in (
        ("done_manifest", manifest_path),
        ("completion_receipt", completion_path),
    ):
        if artifact.is_file() and not artifact.is_symlink():
            receipt[field] = {
                "path": str(artifact.resolve()),
                "sha256": _sha256_file(artifact),
            }
    _atomic_json(path, receipt)
    return receipt


def verify_completion_receipt(
    path: Path,
    *,
    manifest_path: Path,
    args: argparse.Namespace,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    receipt = _read_json(path, label="source completion receipt")
    if (
        receipt.get("schema") != COMPLETION_SCHEMA
        or receipt.get("status") != "success"
        or receipt.get("streams") != "code"
        or receipt.get("interrupted") is not False
        or receipt.get("source_repo_list_reverified_at_finish") is not True
    ):
        raise RuntimeError("source completion receipt is not a clean code success")

    manifest = receipt.get("manifest")
    if not isinstance(manifest, dict):
        raise TypeError("source completion receipt has no manifest binding")
    try:
        receipt_manifest_path = Path(str(manifest.get("path", ""))).resolve(strict=True)
        expected_manifest_path = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "source completion receipt manifest path is invalid"
        ) from exc
    if receipt_manifest_path != expected_manifest_path:
        raise RuntimeError("source completion receipt binds a different manifest")
    manifest_size = _require_int(
        manifest.get("size_bytes"),
        label="completion manifest size_bytes",
        minimum=1,
    )
    if expected_manifest_path.stat().st_size != manifest_size:
        raise RuntimeError("source completion manifest size drifted")
    manifest_sha256 = _require_sha256(
        manifest.get("sha256"),
        label="completion manifest sha256",
    )
    if _sha256_file(expected_manifest_path) != manifest_sha256:
        raise RuntimeError("source completion manifest digest drifted")

    repositories = receipt.get("code_repositories")
    if (
        not isinstance(repositories, list)
        or not all(isinstance(repository, str) for repository in repositories)
        or repositories != sorted(set(repositories))
    ):
        raise RuntimeError("source completion repository projection is invalid")
    failed_unit_count = _require_int(
        receipt.get("failed_unit_count"),
        label="completion failed_unit_count",
    )
    non_code_done_unit_count = _require_int(
        receipt.get("non_code_done_unit_count"),
        label="completion non_code_done_unit_count",
    )
    total_done_unit_count = _require_int(
        receipt.get("total_done_unit_count"),
        label="completion total_done_unit_count",
    )
    if failed_unit_count:
        raise RuntimeError(
            f"completion manifest retains {failed_unit_count} failed units"
        )
    if non_code_done_unit_count or total_done_unit_count != len(repositories):
        raise RuntimeError("completion manifest contains non-code done units")

    selected = sorted(getattr(args, "only_repo", []))
    expected_count = (
        len(selected)
        if selected
        else int(inputs["archive_inventory_receipt"]["repository_count"])
    )
    if len(repositories) != expected_count or len(set(repositories)) != expected_count:
        raise RuntimeError(
            "completion manifest repository coverage is incomplete: "
            f"expected={expected_count} actual={len(set(repositories))}"
        )
    repository_names_sha256 = _canonical_sha256(repositories)
    expected_names_sha256 = inputs["archive_inventory_receipt"][
        "repository_names_sha256"
    ]
    if (selected and repositories != selected) or (
        not selected and repository_names_sha256 != expected_names_sha256
    ):
        raise RuntimeError("completion manifest repository set differs from inventory")
    receipt_repository_names_sha256 = _require_sha256(
        receipt.get("code_repository_names_sha256"),
        label="completion code_repository_names_sha256",
    )
    if receipt_repository_names_sha256 != repository_names_sha256:
        raise RuntimeError("completion receipt repository digest is invalid")

    manifest_revision = receipt.get("code_revision")
    if not isinstance(manifest_revision, dict) or (
        conveyor._code_revision_identity(manifest_revision)
        != conveyor._code_revision_identity(inputs["code_revision"])
    ):
        raise RuntimeError("completion manifest code revision binding drifted")
    manifest_repo_list = receipt.get("source_repo_list")
    expected_repo_list = inputs["repo_list"]
    if (
        not isinstance(manifest_repo_list, dict)
        or manifest_repo_list.get("schema") != "cppmega_source_repo_list_binding_v1"
        or manifest_repo_list.get("sha256") != expected_repo_list["sha256"]
        or manifest_repo_list.get("canonical_mapping_sha256")
        != expected_repo_list["canonical_mapping_sha256"]
        or manifest_repo_list.get("mapping_count")
        != expected_repo_list["mapping_entry_count"]
    ):
        raise RuntimeError("completion manifest repo-list binding drifted")
    return {
        "status": "complete",
        "expected_repository_count": expected_count,
        "successful_repository_count": len(repositories),
        "failed_unit_count": 0,
        "repository_names_sha256": repository_names_sha256,
        "manifest_sha256": manifest_sha256,
    }


def _run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    lock_stream = (run_root / "launch.lock").open("a+b")
    try:
        try:
            fcntl.flock(
                lock_stream.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError("source conveyor supervisor is already running") from exc

        inputs = validate_inputs(args)
        repair_base = (
            load_repair_base_code_run(Path(args.repair_base_code_run_root))
            if args.repair_base_code_run_root
            else None
        )
        if repair_base is not None:
            validate_repair_request(args, inputs, repair_base)
        command = build_command(args, inputs, repair_base)
        run_binding = build_run_binding(args, inputs, repair_base)
        launch_path = run_root / "launch_receipt.json"
        attempt = _resume_attempt(
            launch_path,
            run_binding=run_binding,
        )
        launch_receipt = build_launch_receipt(
            args,
            inputs=inputs,
            command=command,
            run_binding=run_binding,
            attempt=attempt,
            repair_base=repair_base,
        )
        _atomic_json(launch_path, launch_receipt)

        def mark_started(child_pid: int) -> None:
            launch_receipt.update(
                status="running",
                started_at=_utc_now(),
                supervisor_pid=os.getpid(),
                child_pid=child_pid,
            )
            _atomic_json(launch_path, launch_receipt)

        return_code = run_supervised_child(
            command,
            cwd=_REPO_ROOT,
            environment=build_child_environment(inputs),
            log_path=run_root / "run.log",
            on_started=mark_started,
        )
        done_manifest = run_root / "conveyor" / "_done.json"
        completion_receipt = run_root / "conveyor" / "completion_receipt.json"
        terminal_coverage = None
        if return_code == 0:
            terminal_inputs = validate_inputs(
                args,
                enforce_minimum_free_disk=False,
            )
            terminal_repair_base = (
                load_repair_base_code_run(Path(args.repair_base_code_run_root))
                if args.repair_base_code_run_root
                else None
            )
            if terminal_repair_base is not None:
                validate_repair_request(args, terminal_inputs, terminal_repair_base)
            if _canonical_sha256(
                build_run_binding(args, terminal_inputs, terminal_repair_base)
            ) != (
                _canonical_sha256(run_binding)
            ):
                raise RuntimeError(
                    "immutable source inputs changed while the conveyor was running"
                )
            terminal_coverage = verify_completion_receipt(
                completion_receipt,
                manifest_path=done_manifest,
                args=args,
                inputs=terminal_inputs,
            )
            terminal_coverage["immutable_inputs_reverified_at_finish"] = True
        write_exit_receipt(
            run_root / "exit_receipt.json",
            launch_path=launch_path,
            code_revision=args.expected_code_revision,
            return_code=return_code,
            manifest_path=done_manifest,
            completion_path=completion_receipt,
            terminal_coverage=terminal_coverage,
            schema=TARGETED_EXIT_SCHEMA if repair_base else EXIT_SCHEMA,
            selected_repositories=(list(args.only_repo) if repair_base else None),
            repair_base_code_run=(repair_base["identity"] if repair_base else None),
        )
        return return_code
    finally:
        lock_stream.close()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        return _run(args)
    except Exception as error:  # noqa: BLE001 - persist every supervisor failure
        run_root = Path(args.run_root).expanduser().resolve()
        failure = {
            "schema": FAILURE_SCHEMA,
            "failed_at": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        try:
            _atomic_json(run_root / "supervisor_failure.json", failure)
        except OSError:
            pass
        print(f"source conveyor supervisor failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
