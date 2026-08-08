#!/usr/bin/env python3
"""Build a hash-pinned GCP payload for a physical cloud-lane worker pool."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    load_json_object,
    require_int,
    require_sha256,
    run_checked,
    sha256_file,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.cloud_lane import (  # noqa: E402
    load_cloud_lane_manifest,
)


PAYLOAD_SCHEMA = "cppmega.gcp_cloud_lane_payload_v1"
RUNNER_PLACEHOLDERS = {
    "__CPPMEGA_BUNDLE_SHA256__": "bundle_sha256",
    "__CPPMEGA_OVERLAY_SHA256__": "overlay_sha256",
    "__CPPMEGA_MANIFEST_SHA256__": "manifest_file_sha256",
    "__CPPMEGA_MANIFEST_LOGICAL_SHA256__": "manifest_sha256",
    "__CPPMEGA_CODE_REVISION__": "code_revision",
    "__CPPMEGA_ADAPTER_SHA256__": "adapter_sha256",
    "__CPPMEGA_REQUIREMENTS_SHA256__": "requirements_sha256",
    "__CPPMEGA_FETCH_RECEIPT_SHA256__": "fetch_receipt_sha256",
    "__CPPMEGA_TOKENIZERS_VERSION__": "tokenizers_version",
}
_RUN_ID_RE = re.compile(r"^[a-z0-9]([-a-z0-9]{0,26}[a-z0-9])?$")


def _regular_file(path: Path, *, where: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ContractError(f"{where} must be a regular file: {path}")
    return resolved


def _tracked_clean(repo_root: Path) -> str:
    revision = run_checked(["git", "-C", repo_root, "rev-parse", "HEAD"]).stdout.strip()
    status = run_checked(
        ["git", "-C", repo_root, "status", "--porcelain", "--untracked-files=no"]
    ).stdout
    if status:
        raise ContractError("tracked repository files are dirty")
    return revision


def _parse_run_root(run_root: str) -> tuple[str, str, str]:
    normalized = validate_gcs_uri(run_root.rstrip("/"), where="run_root")
    relative = normalized.removeprefix("gs://")
    bucket, separator, object_name = relative.partition("/")
    if not separator:
        raise ContractError("run_root must include a GCS object prefix")
    prefix, separator, run_id = object_name.rpartition("/")
    if not separator or not prefix or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ContractError("run_root must end in a canonical 1-28 character run_id")
    return bucket, prefix, run_id


def _load_fetch_contract(path: Path) -> tuple[str, str]:
    _raw, value = load_json_object(path, where="frozen fetch receipt")
    contract = value.get("tokenizer_contract")
    if not isinstance(contract, Mapping):
        raise ContractError("frozen fetch receipt has no tokenizer_contract")
    if contract.get("library") != "tokenizers":
        raise ContractError("frozen fetch receipt tokenizer library drifted")
    version = contract.get("library_version")
    artifact = contract.get("artifact_sha256")
    if not isinstance(version, str) or not version:
        raise ContractError("frozen fetch receipt tokenizer version is missing")
    return version, require_sha256(artifact, where="fetch tokenizer artifact SHA-256")


def _validate_requirements(path: Path, *, tokenizers_version: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    wanted = f"tokenizers=={tokenizers_version}"
    matching = [
        index for index, line in enumerate(lines) if line.split(" \\", 1)[0] == wanted
    ]
    if len(matching) != 1:
        raise ContractError(
            "requirements must hash-pin the frozen tokenizers version exactly once"
        )
    index = matching[0]
    requirement = lines[index]
    continuation = lines[index + 1].strip() if index + 1 < len(lines) else ""
    if "--hash=sha256:" not in requirement and not continuation.startswith(
        "--hash=sha256:"
    ):
        raise ContractError("tokenizers requirement is not hash-pinned")


def _write_overlay(
    destination: Path, *, requirements: Path, fetch_receipt: Path
) -> None:
    members = (
        (requirements, "ci-case5-linux-cp311-requirements.txt"),
        (fetch_receipt, "fetch-receipt.json"),
    )
    raw_tar = destination.with_suffix("")
    with tarfile.open(raw_tar, "w", format=tarfile.USTAR_FORMAT) as archive:
        for source, name in members:
            info = archive.gettarinfo(str(source), arcname=name)
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            info.mode = 0o444
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    try:
        with destination.open("wb") as output:
            completed = subprocess.run(
                ["zstd", "-19", "-T1", "--no-progress", "-c", "--", raw_tar],
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                "zstd failed: "
                + completed.stderr[-4000:].decode("utf-8", errors="replace")
            )
    finally:
        raw_tar.unlink(missing_ok=True)
    run_checked(["zstd", "-t", destination])


def _artifact(path: Path, *, root: Path, uri: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if uri is not None:
        value["uri"] = uri
    return value


def prepare_cloud_lane_payload(
    *,
    repo_root: Path,
    manifest_path: Path,
    runner_template: Path,
    requirements_path: Path,
    fetch_receipt_path: Path,
    output_root: Path,
    run_root: str,
    project_id: str,
    region: str,
    zone: str,
    physical_worker_count: int,
    slots_per_worker: int,
    machine_type: str,
    local_ssd_count: int,
) -> dict[str, object]:
    """Create immutable bootstrap objects, tfvars, and a payload receipt."""

    repo_root = repo_root.resolve()
    revision = _tracked_clean(repo_root)
    manifest_path = _regular_file(manifest_path, where="cloud lane manifest")
    runner_template = _regular_file(runner_template, where="runner template")
    requirements_path = _regular_file(requirements_path, where="requirements lock")
    fetch_receipt_path = _regular_file(
        fetch_receipt_path, where="frozen fetch receipt"
    )
    if output_root.exists() or output_root.is_symlink():
        raise ContractError(f"payload output already exists: {output_root}")
    bucket, gcs_prefix, run_id = _parse_run_root(run_root)
    manifest, manifest_file_sha256 = load_cloud_lane_manifest(manifest_path)
    if manifest["kind"] != "ci":
        raise ContractError("CASE5 payload requires a ci cloud lane manifest")
    pipeline = manifest["pipeline"]
    assert isinstance(pipeline, Mapping)
    if pipeline["code_revision"] != revision:
        raise ContractError("manifest code_revision differs from repository HEAD")
    adapter = _regular_file(
        repo_root / "scripts/distributed_data_prep/ci_case5_snapshot.py",
        where="CASE5 adapter",
    )
    adapter_sha256 = sha256_file(adapter)
    if pipeline["runner_sha256"] != adapter_sha256:
        raise ContractError("manifest runner_sha256 differs from CASE5 adapter bytes")
    tokenizer = _regular_file(
        repo_root / "cppmega/tokenizer/tokenizer.json", where="tokenizer"
    )
    if pipeline["tokenizer_sha256"] != sha256_file(tokenizer):
        raise ContractError("manifest tokenizer SHA-256 differs from repository bytes")
    tokenizers_version, tokenizer_artifact = _load_fetch_contract(fetch_receipt_path)
    if tokenizer_artifact != pipeline["tokenizer_sha256"]:
        raise ContractError("frozen fetch receipt tokenizer differs from manifest")
    _validate_requirements(requirements_path, tokenizers_version=tokenizers_version)

    physical_count = require_int(
        physical_worker_count, where="physical_worker_count", minimum=1
    )
    logical_count = len(manifest["workers"])
    if physical_count > logical_count:
        raise ContractError("physical worker count exceeds logical workers")
    slots = require_int(slots_per_worker, where="slots_per_worker", minimum=1)
    smallest_share = logical_count // physical_count
    if slots > min(16, smallest_share):
        raise ContractError("slots exceed the smallest physical worker share")
    if not isinstance(local_ssd_count, int) or local_ssd_count not in {2, 4, 8, 16, 24}:
        raise ContractError("local_ssd_count is unsupported by the worker module")
    if not project_id or not region or not zone.startswith(region + "-"):
        raise ContractError("project/region/zone binding is invalid")
    output_bucket = str(manifest["gcs_output_prefix"]).removeprefix("gs://").split("/", 1)[0]
    if output_bucket != bucket:
        raise ContractError("payload run root and manifest output must share a bucket")

    template = runner_template.read_text(encoding="utf-8")
    for placeholder in RUNNER_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ContractError(f"runner template placeholder drifted: {placeholder}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.", dir=output_root.parent
    ) as raw_stage:
        stage = Path(raw_stage)
        bootstrap = stage / "bootstrap"
        manifests = stage / "manifests"
        bootstrap.mkdir()
        manifests.mkdir()

        bundle_stage = bootstrap / "cppmega.bundle"
        run_checked(["git", "-C", repo_root, "bundle", "create", bundle_stage, "HEAD"])
        run_checked(["git", "bundle", "verify", bundle_stage])
        advertised = run_checked(["git", "bundle", "list-heads", bundle_stage]).stdout.splitlines()
        if len(advertised) != 1 or advertised[0].split()[0] != revision:
            raise ContractError("bundle does not advertise exactly the manifest revision")
        bundle_sha256 = sha256_file(bundle_stage)
        bundle = bootstrap / f"{bundle_sha256}.cppmega.bundle"
        os.replace(bundle_stage, bundle)

        overlay_stage = bootstrap / "cloud-lane-bootstrap.tar.zst"
        _write_overlay(
            overlay_stage,
            requirements=requirements_path,
            fetch_receipt=fetch_receipt_path,
        )
        overlay_sha256 = sha256_file(overlay_stage)
        overlay = bootstrap / f"{overlay_sha256}.cloud-lane-bootstrap.tar.zst"
        os.replace(overlay_stage, overlay)

        manifest_copy = manifests / f"{manifest_file_sha256}.cloud-lane-manifest.json"
        manifest_copy.write_bytes(manifest_path.read_bytes())
        os.chmod(manifest_copy, 0o444)

        values = {
            "bundle_sha256": bundle_sha256,
            "overlay_sha256": overlay_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
            "code_revision": revision,
            "adapter_sha256": adapter_sha256,
            "requirements_sha256": sha256_file(requirements_path),
            "fetch_receipt_sha256": sha256_file(fetch_receipt_path),
            "tokenizers_version": tokenizers_version,
        }
        rendered = template
        for placeholder, key in RUNNER_PLACEHOLDERS.items():
            rendered = rendered.replace(placeholder, str(values[key]))
        if "__CPPMEGA_" in rendered:
            raise ContractError("runner contains an unresolved placeholder")
        runner_stage = bootstrap / "cloud-lane-worker-runner"
        runner_stage.write_text(rendered, encoding="utf-8", newline="\n")
        runner_sha256 = sha256_file(runner_stage)
        runner = bootstrap / f"{runner_sha256}.cloud-lane-worker-runner"
        os.replace(runner_stage, runner)
        runner.chmod(0o555)

        artifact_uris = {
            bundle: f"{run_root.rstrip('/')}/bootstrap/{bundle.name}",
            overlay: f"{run_root.rstrip('/')}/bootstrap/{overlay.name}",
            runner: f"{run_root.rstrip('/')}/bootstrap/{runner.name}",
            manifest_copy: f"{run_root.rstrip('/')}/manifests/{manifest_copy.name}",
        }
        tfvars = {
            "project_id": project_id,
            "region": region,
            "zone": zone,
            "bucket_name": bucket,
            "gcs_prefix": gcs_prefix,
            "run_id": run_id,
            "worker_count": physical_count,
            "slots_per_worker": slots,
            "parse_workers_per_slot": 1,
            "memory_limit_gb_per_slot": 24,
            "cpu_budget_vcpus": 16,
            "memory_budget_gb": 56,
            "machine_type": machine_type,
            "local_ssd_count": local_ssd_count,
            "use_spot": False,
            "compact_placement": physical_count <= 22,
            "runner_role": "cloud-lane",
            "bootstrap_script_gcs_uri": artifact_uris[runner],
            "bootstrap_script_sha256": runner_sha256,
            "bootstrap_bundle_sha256": bundle_sha256,
            "bootstrap_overlay_sha256": overlay_sha256,
            "bootstrap_manifest_sha256": manifest_file_sha256,
            "labels": {"lane": "ci-case5", "manifest": str(manifest["manifest_sha256"])[:12]},
        }
        tfvars_path = stage / f"{run_id}.tfvars.json"
        atomic_write_json(tfvars_path, tfvars)

        artifacts = [
            _artifact(path, root=stage, uri=uri)
            for path, uri in sorted(artifact_uris.items(), key=lambda item: str(item[0]))
        ]
        artifacts.append(_artifact(tfvars_path, root=stage))
        receipt: dict[str, object] = {
            "schema": PAYLOAD_SCHEMA,
            "status": "ready_not_uploaded_not_applied",
            "run_root": run_root.rstrip("/"),
            "run_id": run_id,
            "code_revision": revision,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": manifest_file_sha256,
            "adapter_sha256": adapter_sha256,
            "logical_worker_count": logical_count,
            "physical_worker_count": physical_count,
            "slots_per_worker": slots,
            "training_ready": False,
            "artifacts": artifacts,
        }
        atomic_write_json(stage / "payload-receipt.json", receipt)
        os.replace(stage, output_root)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runner-template", required=True, type=Path)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--fetch-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--zone", default="us-central1-a")
    parser.add_argument("--physical-worker-count", type=int, default=4)
    parser.add_argument("--slots-per-worker", type=int, default=2)
    parser.add_argument("--machine-type", default="n2-standard-16")
    parser.add_argument("--local-ssd-count", type=int, default=2)
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        receipt = prepare_cloud_lane_payload(
            repo_root=args.repo_root,
            manifest_path=args.manifest,
            runner_template=args.runner_template,
            requirements_path=args.requirements,
            fetch_receipt_path=args.fetch_receipt,
            output_root=args.output_root,
            run_root=args.run_root,
            project_id=args.project_id,
            region=args.region,
            zone=args.zone,
            physical_worker_count=args.physical_worker_count,
            slots_per_worker=args.slots_per_worker,
            machine_type=args.machine_type,
            local_ssd_count=args.local_ssd_count,
        )
    except (ContractError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"GCP cloud lane payload preparation failed: {exc}\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["PAYLOAD_SCHEMA", "prepare_cloud_lane_payload"]
