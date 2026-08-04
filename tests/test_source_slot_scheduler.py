from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    gcs_join,
    sha256_file,
)
from scripts.distributed_data_prep.source_manifest import (
    LOSSLESS_INDEX_MAX_TOKENS,
    PRE_GLOBAL_SCHEMA,
    build_source_manifest,
)
from scripts.distributed_data_prep.source_slot_scheduler import (
    SLOT_COMPLETION_RECEIPT_SCHEMA,
    SlotSpec,
    load_resumable_slot_receipt,
    logical_worker_count,
    run_source_slot_scheduler,
    slot_completion_uri,
    slot_specs,
    validate_manifest_topology,
    validate_slot_completion_receipt,
    validate_slot_resources,
)
from scripts.distributed_data_prep.source_worker import (
    ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA,
    LocalObjectStore,
    _load_completed_assignment,
    assignment_completion_uri,
    validate_assignment_completion_receipt,
)


def _manifest(*, worker_count: int = 2) -> tuple[dict[str, object], str]:
    manifest = build_source_manifest(
        [
            {
                "repo": "project",
                "project_id": "owner/project",
                "source": {
                    "kind": "immutable_gcs_tar",
                    "uri": "gs://inputs/project.tar.zst",
                    "generation": "1",
                    "sha256": "1" * 64,
                    "archive_format": "tar.zst",
                    "strip_components": 1,
                },
            }
        ],
        worker_count=worker_count,
        gcs_output_prefix="gs://outputs/source-run",
        code_revision="2" * 40,
        indexer_sha256="3" * 64,
        tokenizer_sha256="4" * 64,
        quarantine_manifest_sha256="5" * 64,
    )
    return manifest, "6" * 64


def _worker_receipt(manifest: dict[str, object], manifest_file_sha: str) -> dict[str, object]:
    job = manifest["repositories"][0]
    assert isinstance(job, dict)
    compressed_sha = "7" * 64
    quarantine_sha = "8" * 64
    return {
        "schema": "cppmega.distributed_source_worker_receipt_v2",
        "status": "complete",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha,
        "assignment": {
            key: job[key]
            for key in ("ordinal", "repo", "project_id", "worker", "assignment_sha256")
        },
        "source_snapshot": {},
        "candidate": {
            "schema": PRE_GLOBAL_SCHEMA,
            "document_order": "canonical_enriched_json_v1",
            "documents": 1,
            "canonical_stream_sha256": "9" * 64,
            "dedup_applied": False,
        },
        "artifact": {
            "uri": gcs_join(
                str(manifest["gcs_output_prefix"]),
                "source-candidates",
                str(manifest["manifest_sha256"]),
                "00000-project",
                f"{compressed_sha}.jsonl.zst",
            ),
            "generation": "1",
            "size_bytes": 3,
            "crc32c": None,
            "md5_hash": None,
            "sha256": compressed_sha,
            "compression": {
                "compression": "zstd",
                "level": 19,
                "threads": 1,
                "zstd_version": "zstd v1",
                "size_bytes": 3,
                "sha256": compressed_sha,
            },
        },
        "quarantine_artifact": {
            "uri": gcs_join(
                str(manifest["gcs_output_prefix"]),
                "source-quarantine-receipts",
                str(manifest["manifest_sha256"]),
                "00000-project",
                f"{quarantine_sha}.quarantine.json",
            ),
            "generation": "1",
            "size_bytes": 3,
            "crc32c": None,
            "md5_hash": None,
            "sha256": quarantine_sha,
        },
        "indexer": {
            "mode": "single_project_pre_global_enriched_v1",
            "project_id": "owner/project",
            "enriched": True,
            "max_tokens": LOSSLESS_INDEX_MAX_TOKENS,
            "parse_workers": 6,
            "memory_limit_gb": 24.0,
            "excluded_directories": ["__pycache__", "node_modules", "build", ".git"],
            "dedup_applied": False,
            "tokenizer_passed_to_indexer": False,
            "raw_output_sha256": "a" * 64,
            "quarantine_receipt_sha256": quarantine_sha,
        },
        "training_ready": False,
    }


def test_slot_topology_is_contiguous_and_manifest_bound() -> None:
    assert logical_worker_count(4, 2) == 8
    assert [item.worker for item in slot_specs(
        physical_worker_index=2, physical_worker_count=4, slots_per_worker=2
    )] == ["worker-0004", "worker-0005"]
    manifest, _ = _manifest(worker_count=8)
    validated, specs = validate_manifest_topology(
        manifest,
        physical_worker_index=3,
        physical_worker_count=4,
        slots_per_worker=2,
    )
    assert validated == manifest
    assert [spec.worker for spec in specs] == ["worker-0006", "worker-0007"]

    with pytest.raises(ContractError, match="logical workers"):
        validate_manifest_topology(
            _manifest(worker_count=2)[0],
            physical_worker_index=0,
            physical_worker_count=4,
            slots_per_worker=2,
        )


def test_slot_resource_bounds_are_strict() -> None:
    resources = validate_slot_resources(
        slots_per_worker=2,
        parse_workers_per_slot=6,
        memory_limit_gb_per_slot=24,
        cpu_budget_vcpus=16,
        memory_budget_gb=56,
    )
    assert resources["parse_workers_per_slot"] == 6
    with pytest.raises(ContractError, match="CPU budget"):
        validate_slot_resources(
            slots_per_worker=2,
            parse_workers_per_slot=9,
            memory_limit_gb_per_slot=24,
            cpu_budget_vcpus=16,
            memory_budget_gb=56,
        )
    with pytest.raises(ContractError, match="memory budget"):
        validate_slot_resources(
            slots_per_worker=2,
            parse_workers_per_slot=6,
            memory_limit_gb_per_slot=29,
            cpu_budget_vcpus=16,
            memory_budget_gb=56,
        )


def test_slot_completion_receipt_is_exact_and_resumable(tmp_path: Path) -> None:
    manifest, manifest_file_sha = _manifest()
    spec = slot_specs(
        physical_worker_index=0, physical_worker_count=1, slots_per_worker=2
    )[0]
    # The one-repository manifest assigns this job to worker-0000.
    resources = validate_slot_resources(
        slots_per_worker=2,
        parse_workers_per_slot=6,
        memory_limit_gb_per_slot=24,
        cpu_budget_vcpus=16,
        memory_budget_gb=56,
    )
    worker_receipt = _worker_receipt(manifest, manifest_file_sha)
    source_path = tmp_path / "source-receipt.json"
    atomic_write_json(source_path, worker_receipt)
    store = LocalObjectStore(tmp_path / "objects")
    job = manifest["repositories"][0]
    assert isinstance(job, dict)
    source_uri = gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-receipts",
        str(manifest["manifest_sha256"]),
        "00000-project",
        "7" * 64 + ".receipt.json",
    )
    source_metadata = store.publish_if_absent(source_path, source_uri)
    source_entry = {
        "ordinal": job["ordinal"],
        "repo": job["repo"],
        "project_id": job["project_id"],
        "worker": job["worker"],
        "assignment_sha256": job["assignment_sha256"],
        "uri": source_uri,
        "generation": source_metadata["generation"],
        "size_bytes": source_metadata["size_bytes"],
        "sha256": sha256_file(source_path),
    }
    assignment_pointer = {
        "schema": ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA,
        "status": "complete",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha,
        "assignment": {
            key: job[key]
            for key in ("ordinal", "repo", "project_id", "worker", "assignment_sha256")
        },
        "source_receipt": {
            key: source_entry[key]
            for key in ("uri", "generation", "size_bytes", "sha256")
        },
        "training_ready": False,
    }
    validate_assignment_completion_receipt(
        assignment_pointer,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha,
        job=job,
    )
    pointer_path = tmp_path / "assignment-pointer.json"
    atomic_write_json(pointer_path, assignment_pointer)
    store.publish_if_absent(pointer_path, assignment_completion_uri(manifest, job))
    assert _load_completed_assignment(
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha,
        job=job,
        object_store=store,
        scratch_root=tmp_path / "assignment-resume",
    ) == worker_receipt
    slot_receipt = {
        "schema": SLOT_COMPLETION_RECEIPT_SCHEMA,
        "status": "complete",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha,
        "topology": {
            "physical_worker_index": spec.physical_worker_index,
            "physical_worker_count": spec.physical_worker_count,
            "slots_per_worker": spec.slots_per_worker,
            "slot_index": spec.slot_index,
            "worker": spec.worker,
        },
        "resources": resources,
        "source_receipts": [source_entry],
        "training_ready": False,
    }
    validate_slot_completion_receipt(
        slot_receipt,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha,
        spec=spec,
        resources=resources,
        object_store=store,
        verification_root=tmp_path / "verify",
    )
    completion_path = tmp_path / "completion.json"
    atomic_write_json(completion_path, slot_receipt)
    completion_uri = slot_completion_uri(manifest, spec.worker)
    store.publish_if_absent(completion_path, completion_uri)
    assert load_resumable_slot_receipt(
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha,
        spec=spec,
        resources=resources,
        object_store=store,
        verification_root=tmp_path / "resume",
    ) == slot_receipt

    tampered = copy.deepcopy(slot_receipt)
    tampered["resources"]["memory_limit_gb_per_slot"] = 25.0
    with pytest.raises(ContractError, match="resource binding"):
        validate_slot_completion_receipt(
            tampered,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha,
            spec=spec,
            resources=resources,
        )


def test_empty_logical_slot_is_valid() -> None:
    manifest, manifest_file_sha = _manifest(worker_count=2)
    spec = slot_specs(
        physical_worker_index=0, physical_worker_count=1, slots_per_worker=2
    )[1]
    resources = validate_slot_resources(
        slots_per_worker=2,
        parse_workers_per_slot=6,
        memory_limit_gb_per_slot=24,
        cpu_budget_vcpus=16,
        memory_budget_gb=56,
    )
    receipt = {
        "schema": SLOT_COMPLETION_RECEIPT_SCHEMA,
        "status": "complete",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha,
        "topology": {
            "physical_worker_index": 0,
            "physical_worker_count": 1,
            "slots_per_worker": 2,
            "slot_index": 1,
            "worker": "worker-0001",
        },
        "resources": resources,
        "source_receipts": [],
        "training_ready": False,
    }
    validate_slot_completion_receipt(
        receipt,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha,
        spec=spec,
        resources=resources,
    )


def _empty_slot_scheduler_fixture(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "bundle-source"
    repo.mkdir()
    for command in (
        ["git", "-C", str(repo), "init", "-q"],
        ["git", "-C", str(repo), "config", "user.name", "Scheduler Test"],
        ["git", "-C", str(repo), "config", "user.email", "test@example.test"],
    ):
        subprocess.run(command, check=True, capture_output=True)
    (repo / "README").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "fixture"],
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    bundle = tmp_path / "cppmega.bundle"
    subprocess.run(
        ["git", "-C", str(repo), "bundle", "create", str(bundle), "HEAD"],
        check=True,
        capture_output=True,
    )
    overlay_tar = tmp_path / "overlay.tar"
    with tarfile.open(overlay_tar, "w", format=tarfile.USTAR_FORMAT):
        pass
    overlay = tmp_path / "overlay.tar.zst"
    subprocess.run(
        ["zstd", "-q", "-f", str(overlay_tar), "-o", str(overlay)],
        check=True,
        capture_output=True,
    )
    manifest = build_source_manifest(
        [],
        worker_count=2,
        gcs_output_prefix="gs://outputs/empty-slots",
        code_revision=revision,
        indexer_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        quarantine_manifest_sha256="3" * 64,
    )
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_file_sha": sha256_file(manifest_path),
        "bundle": bundle,
        "overlay": overlay,
        "stage": tmp_path / "stage",
        "scheduler_receipt": tmp_path / "scheduler-receipt.json",
        "store": LocalObjectStore(tmp_path / "objects"),
    }


def _fake_source_worker(path: Path, body: str) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n" + textwrap.dedent(body), encoding="utf-8"
    )
    path.chmod(0o755)
    return path


def _run_empty_slots(fixture: dict[str, object], fake_worker: Path) -> dict[str, object]:
    manifest = fixture["manifest"]
    assert isinstance(manifest, dict)
    return run_source_slot_scheduler(
        manifest_path=fixture["manifest_path"],
        manifest_file_sha256=str(fixture["manifest_file_sha"]),
        run_root=str(manifest["gcs_output_prefix"]),
        bundle=fixture["bundle"],
        overlay=fixture["overlay"],
        stage_root=fixture["stage"],
        scheduler_receipt_path=fixture["scheduler_receipt"],
        physical_worker_index=0,
        physical_worker_count=1,
        slots_per_worker=2,
        parse_workers_per_slot=6,
        memory_limit_gb_per_slot=24,
        cpu_budget_vcpus=16,
        memory_budget_gb=56,
        python=Path(sys.executable),
        source_worker=fake_worker,
        object_store=fixture["store"],
        check_host=False,
        poll_interval=0.01,
    )


def test_scheduler_isolates_slots_and_resumes_exact_completions(tmp_path: Path) -> None:
    fixture = _empty_slot_scheduler_fixture(tmp_path)
    fake_worker = _fake_source_worker(
        tmp_path / "fake-worker.py",
        """
        import sys
        sys.exit(0)
        """,
    )
    first = _run_empty_slots(fixture, fake_worker)
    assert first["completed_slots"] == ["worker-0000", "worker-0001"]
    assert first["resumed_slots"] == []
    stage = fixture["stage"]
    assert isinstance(stage, Path)
    attempts = sorted(stage.glob("slots/worker-*/attempt-*"))
    assert len(attempts) == 2
    assert attempts[0].parent != attempts[1].parent

    second = _run_empty_slots(fixture, fake_worker)
    assert second["completed_slots"] == ["worker-0000", "worker-0001"]
    assert second["resumed_slots"] == ["worker-0000", "worker-0001"]
    assert sorted(stage.glob("slots/worker-*/attempt-*")) == attempts


def test_scheduler_terminates_siblings_after_slot_failure(tmp_path: Path) -> None:
    fixture = _empty_slot_scheduler_fixture(tmp_path)
    fake_worker = _fake_source_worker(
        tmp_path / "failing-worker.py",
        """
        import os
        import sys
        import time

        if os.environ["CPPMEGA_SOURCE_SLOT"] == "worker-0000":
            sys.exit(7)
        time.sleep(30)
        """,
    )
    with pytest.raises(RuntimeError, match="worker-0000 failed"):
        _run_empty_slots(fixture, fake_worker)
    store = fixture["store"]
    manifest = fixture["manifest"]
    assert isinstance(store, LocalObjectStore)
    assert isinstance(manifest, dict)
    assert store.describe_if_present(slot_completion_uri(manifest, "worker-0000")) is None
    assert store.describe_if_present(slot_completion_uri(manifest, "worker-0001")) is None
