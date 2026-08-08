#!/usr/bin/env python3
"""Build a hash-pinned, upload-ready GCP source-map pilot payload."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    require_sha256,
    run_checked,
    sha256_file,
)
from scripts.distributed_data_prep.source_manifest import (
    build_source_manifest,
)
from scripts.distributed_data_prep.source_slot_scheduler import (
    logical_worker_count,
    validate_slot_resources,
)

RUNNER_PLACEHOLDERS = {
    "__CPPMEGA_BUNDLE_SHA256__": "bundle",
    "__CPPMEGA_OVERLAY_SHA256__": "overlay",
    "__CPPMEGA_MANIFEST_SHA256__": "manifest",
}
OPTIONAL_RUNNER_PLACEHOLDERS = {
    "__CPPMEGA_REPAIR_CONTRACT_SHA256__": "none",
}


def render_runner(
    template: str,
    hashes: dict[str, str],
    *,
    repair_contract_sha256: str | None = None,
) -> str:
    """Render every immutable runner input, including an optional repair gate."""

    rendered = template
    for placeholder, key in RUNNER_PLACEHOLDERS.items():
        if rendered.count(placeholder) != 1:
            raise ContractError(f"runner template placeholder drifted: {placeholder}")
        rendered = rendered.replace(
            placeholder,
            require_sha256(hashes.get(key), where=f"runner {key} SHA-256"),
        )
    for placeholder, absent_value in OPTIONAL_RUNNER_PLACEHOLDERS.items():
        if rendered.count(placeholder) != 1:
            raise ContractError(f"runner template placeholder drifted: {placeholder}")
        replacement = (
            require_sha256(
                repair_contract_sha256, where="runner repair contract SHA-256"
            )
            if repair_contract_sha256 is not None
            else absent_value
        )
        rendered = rendered.replace(placeholder, replacement)
    if "__CPPMEGA_" in rendered:
        raise ContractError("runner contains an unresolved placeholder")
    return rendered


def _regular_file(path: Path, *, where: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ContractError(f"{where} must be a regular file: {path}")
    return resolved


def _tracked_clean(repo_root: Path) -> str:
    revision = run_checked(["git", "-C", repo_root, "rev-parse", "HEAD"]).stdout.strip()
    status = run_checked(
        [
            "git",
            "-C",
            repo_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        ]
    ).stdout
    if status:
        raise ContractError("tracked repository files are dirty")
    return revision


def _build_overlay(repo_root: Path, destination: Path) -> None:
    members = [
        "scripts/distributed_data_prep/__init__.py",
        "scripts/distributed_data_prep/_common.py",
        "scripts/distributed_data_prep/source_manifest.py",
        "scripts/distributed_data_prep/source_quarantine_projection.py",
        "scripts/distributed_data_prep/source_slot_scheduler.py",
        "scripts/distributed_data_prep/source_work_queue.py",
        "scripts/distributed_data_prep/source_worker.py",
    ]
    for member in members:
        _regular_file(repo_root / member, where=f"overlay member {member}")
    uncompressed = destination.with_suffix("")
    with tarfile.open(uncompressed, "w", format=tarfile.USTAR_FORMAT) as archive:
        for member in members:
            source = repo_root / member
            info = archive.gettarinfo(str(source), arcname=member)
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    try:
        with destination.open("wb") as output:
            completed = subprocess.run(
                ["zstd", "-19", "-T1", "--no-progress", "-c", "--", uncompressed],
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"zstd failed: {completed.stderr[-4000:].decode(errors='replace')}"
            )
    finally:
        uncompressed.unlink(missing_ok=True)
    run_checked(["zstd", "-t", destination])


def prepare_pilot(
    *,
    repo_root: Path,
    repositories_path: Path,
    runner_template: Path,
    output_root: Path,
    gcs_output_prefix: str,
    worker_count: int,
    slots_per_worker: int = 1,
    parse_workers_per_slot: int = 8,
    memory_limit_gb_per_slot: float = 48.0,
    cpu_budget_vcpus: int = 16,
    memory_budget_gb: float = 56.0,
) -> dict[str, object]:
    """Create immutable bundle, overlay, source manifest, and rendered runner."""

    repo_root = repo_root.resolve()
    resources = validate_slot_resources(
        slots_per_worker=slots_per_worker,
        parse_workers_per_slot=parse_workers_per_slot,
        memory_limit_gb_per_slot=memory_limit_gb_per_slot,
        cpu_budget_vcpus=cpu_budget_vcpus,
        memory_budget_gb=memory_budget_gb,
    )
    logical_count = logical_worker_count(worker_count, slots_per_worker)
    revision = _tracked_clean(repo_root)
    if output_root.exists():
        raise ContractError(f"pilot output already exists: {output_root}")
    _raw = json.loads(
        _regular_file(repositories_path, where="repositories").read_text()
    )
    repositories = _raw.get("repositories")
    if not isinstance(repositories, list):
        raise ContractError("repositories document needs a repositories list")

    template = _regular_file(runner_template, where="runner template").read_text(
        encoding="utf-8"
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.", dir=output_root.parent
    ) as raw_stage:
        stage = Path(raw_stage)
        bootstrap = stage / "bootstrap"
        manifests = stage / "manifests"
        bootstrap.mkdir()
        manifests.mkdir()

        bundle = bootstrap / "cppmega.bundle"
        run_checked(["git", "-C", repo_root, "bundle", "create", bundle, "HEAD"])
        run_checked(["git", "-C", repo_root, "bundle", "verify", bundle])

        overlay = bootstrap / "distributed-data-prep.tar.zst"
        _build_overlay(repo_root, overlay)

        indexer = _regular_file(
            repo_root / "tools/clang_indexer/index_project.py", where="indexer"
        )
        tokenizer = _regular_file(
            repo_root / "cppmega/tokenizer/tokenizer.json", where="tokenizer"
        )
        quarantine = _regular_file(
            repo_root / "configs/source_quarantine_manifest.json",
            where="quarantine manifest",
        )
        source_manifest = build_source_manifest(
            repositories,
            worker_count=logical_count,
            gcs_output_prefix=gcs_output_prefix,
            code_revision=revision,
            indexer_sha256=sha256_file(indexer),
            tokenizer_sha256=sha256_file(tokenizer),
            quarantine_manifest_sha256=sha256_file(quarantine),
        )
        manifest_path = manifests / "source-manifest.json"
        atomic_write_json(manifest_path, source_manifest)

        hashes = {
            "bundle": sha256_file(bundle),
            "overlay": sha256_file(overlay),
            "manifest": sha256_file(manifest_path),
        }
        rendered = render_runner(template, hashes)
        runner = bootstrap / "source-worker-runner"
        runner.write_text(rendered, encoding="utf-8", newline="\n")
        runner.chmod(0o555)

        artifacts = []
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                artifacts.append(
                    {
                        "path": path.relative_to(stage).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        receipt = {
            "schema": "cppmega.gcp_source_pilot_payload_v1",
            "status": "ready",
            "code_revision": revision,
            "worker_count": worker_count,
            "physical_worker_count": worker_count,
            "slots_per_worker": slots_per_worker,
            "logical_worker_count": logical_count,
            "scheduler_mode": "immutable_assignment_work_stealing_v1",
            "resources": resources,
            "gcs_output_prefix": gcs_output_prefix,
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "artifacts": artifacts,
        }
        atomic_write_json(stage / "payload-receipt.json", receipt)
        os.replace(stage, output_root)
    return receipt


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--repositories", required=True, type=Path)
    parser.add_argument("--runner-template", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gcs-output-prefix", required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--slots-per-worker", type=int, default=1)
    parser.add_argument("--parse-workers-per-slot", type=int, default=8)
    parser.add_argument("--memory-limit-gb-per-slot", type=float, default=48.0)
    parser.add_argument("--cpu-budget-vcpus", type=int, default=16)
    parser.add_argument("--memory-budget-gb", type=float, default=56.0)
    args = parser.parse_args(argv)
    try:
        prepare_pilot(
            repo_root=args.repo_root,
            repositories_path=args.repositories,
            runner_template=args.runner_template,
            output_root=args.output_root,
            gcs_output_prefix=args.gcs_output_prefix,
            worker_count=args.worker_count,
            slots_per_worker=args.slots_per_worker,
            parse_workers_per_slot=args.parse_workers_per_slot,
            memory_limit_gb_per_slot=args.memory_limit_gb_per_slot,
            cpu_budget_vcpus=args.cpu_budget_vcpus,
            memory_budget_gb=args.memory_budget_gb,
        )
    except (
        ContractError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        parser.exit(2, f"GCP source pilot preparation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
