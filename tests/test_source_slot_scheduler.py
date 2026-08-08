from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest

from scripts.distributed_data_prep import source_slot_scheduler as scheduler_module
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
    DYNAMIC_SCHEDULER_MODE,
    dynamic_incomplete_receipt_uri,
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
    source_receipt_uri,
)
from scripts.distributed_data_prep.source_worker import (
    ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA,
    LocalObjectStore,
    PINNED_TREE_PROJECTION_MODE,
    TransientTransportError,
    _load_completed_assignment,
    assignment_completion_uri,
    validate_assignment_completion_receipt,
    validate_worker_receipt,
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


def _worker_receipt(
    manifest: dict[str, object],
    manifest_file_sha: str,
    job: dict[str, object] | None = None,
) -> dict[str, object]:
    selected = job or manifest["repositories"][0]
    assert isinstance(selected, dict)
    job = selected
    stem = f"{int(job['ordinal']):05d}-{job['repo']}"
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
                stem,
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
                stem,
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
            "project_id": job["project_id"],
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


def _publish_worker_completion(
    *,
    manifest: dict[str, object],
    manifest_file_sha: str,
    job: dict[str, object],
    store: LocalObjectStore,
    root: Path,
) -> None:
    receipt = _worker_receipt(manifest, manifest_file_sha, job)
    receipt_path = root / f"{int(job['ordinal']):05d}-{job['repo']}.json"
    atomic_write_json(receipt_path, receipt)
    receipt_uri = source_receipt_uri(manifest, job, receipt)
    metadata = store.publish_if_absent(receipt_path, receipt_uri)
    pointer = {
        "schema": ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA,
        "status": "complete",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": manifest_file_sha,
        "assignment": {
            key: job[key]
            for key in (
                "ordinal",
                "repo",
                "project_id",
                "worker",
                "assignment_sha256",
            )
        },
        "source_receipt": {
            "uri": receipt_uri,
            "generation": metadata["generation"],
            "size_bytes": metadata["size_bytes"],
            "sha256": sha256_file(receipt_path),
        },
        "training_ready": False,
    }
    validate_assignment_completion_receipt(
        pointer,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha,
        job=job,
    )
    pointer_path = root / f"{job['assignment_sha256']}.complete.json"
    atomic_write_json(pointer_path, pointer)
    store.publish_if_absent(pointer_path, assignment_completion_uri(manifest, job))


def test_worker_projection_artifact_and_scheduler_command_are_namespace_bound(
    tmp_path: Path,
) -> None:
    manifest, manifest_file_sha = _manifest()
    job = manifest["repositories"][0]
    assert isinstance(job, dict)
    receipt = _worker_receipt(manifest, manifest_file_sha, job)
    projection_sha = "b" * 64
    stem = f"{int(job['ordinal']):05d}-{job['repo']}"
    receipt["quarantine_projection_artifact"] = {
        "uri": gcs_join(
            str(manifest["gcs_output_prefix"]),
            "source-quarantine-projections",
            str(manifest["manifest_sha256"]),
            stem,
            f"{projection_sha}.projection.json",
        ),
        "generation": "1",
        "size_bytes": 7,
        "crc32c": None,
        "md5_hash": None,
        "sha256": projection_sha,
    }
    assert validate_worker_receipt(receipt, manifest=manifest, job=job) == receipt

    escaped = copy.deepcopy(receipt)
    escaped["quarantine_projection_artifact"]["uri"] = (
        "gs://other-bucket/projection.json"
    )
    with pytest.raises(ContractError, match="projection artifact URI"):
        validate_worker_receipt(escaped, manifest=manifest, job=job)

    command = scheduler_module._build_worker_command(
        source_worker=tmp_path / "source-worker.py",
        manifest=tmp_path / "manifest.json",
        spec=SlotSpec(0, 1, 1, 0, "worker-0000"),
        repo_root=tmp_path / "repo",
        scratch_root=tmp_path / "scratch",
        receipt_root=tmp_path / "receipts",
        python=Path(sys.executable),
        resources={"parse_workers_per_slot": 1, "memory_limit_gb_per_slot": 1.0},
        quarantine_projection_mode=PINNED_TREE_PROJECTION_MODE,
    )
    mode_index = command.index("--quarantine-projection-mode")
    assert command[mode_index + 1] == PINNED_TREE_PROJECTION_MODE


def test_gcp_overlay_and_runner_ship_pinned_projection_support(
    tmp_path: Path,
) -> None:
    from scripts.prepare_gcp_source_pilot import _build_overlay

    repo_root = Path(__file__).parents[1]
    overlay = tmp_path / "distributed-data-prep.tar.zst"
    _build_overlay(repo_root, overlay)
    tar_bytes = subprocess.run(
        ["zstd", "-dc", "--", str(overlay)],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
        names = set(archive.getnames())
    assert (
        "scripts/distributed_data_prep/source_quarantine_projection.py" in names
    )

    runner = (
        repo_root / "infra/gcp_corpus_pool/pilot/source-worker-runner.sh.tmpl"
    ).read_text(encoding="utf-8")
    assert "--quarantine-projection-mode pinned_source_tree_v1" in runner


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


def _empty_slot_scheduler_fixture(
    tmp_path: Path,
    *,
    repositories: list[dict[str, object]] | None = None,
) -> dict[str, object]:
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
        repositories or [],
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


def _queue_repositories(count: int) -> list[dict[str, object]]:
    return [
        {
            "repo": f"project-{index}",
            "project_id": f"owner/project-{index}",
            "source": {
                "kind": "immutable_gcs_tar",
                "uri": f"gs://inputs/project-{index}.tar.zst",
                "generation": "1",
                "sha256": f"{index + 1:x}" * 64,
                "archive_format": "tar.zst",
                "strip_components": 1,
            },
        }
        for index in range(count)
    ]


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


def _install_fake_dynamic_launcher(
    *,
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, object],
    poll_delays: dict[int, int],
    exit_codes: dict[int, int],
    started: list[tuple[str, str, int]],
) -> None:
    manifest = fixture["manifest"]
    store = fixture["store"]
    stage = fixture["stage"]
    assert isinstance(manifest, dict)
    assert isinstance(store, LocalObjectStore)
    assert isinstance(stage, Path)
    publication_root = stage / "fake-worker-publications"

    class FakeProcess:
        def __init__(self, job: dict[str, object]) -> None:
            self.job = job
            self.remaining = poll_delays.get(int(job["ordinal"]), 0)
            self.returncode = exit_codes.get(int(job["ordinal"]), 0)
            self.finished = False

        def poll(self) -> int | None:
            if self.finished:
                return self.returncode
            if self.remaining > 0:
                self.remaining -= 1
                return None
            if self.returncode == 0:
                _publish_worker_completion(
                    manifest=manifest,
                    manifest_file_sha=str(fixture["manifest_file_sha"]),
                    job=self.job,
                    store=store,
                    root=publication_root,
                )
            self.finished = True
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 2
            self.finished = True

        def kill(self) -> None:
            self.terminate()

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.finished = True
            return self.returncode

    def fake_start(**kwargs: object) -> dict[str, object]:
        spec = kwargs["spec"]
        lease = kwargs["lease"]
        executor = kwargs["executor"]
        assert isinstance(spec, SlotSpec)
        assert isinstance(lease, scheduler_module.AssignmentLease)
        assert isinstance(executor, dict)
        job = lease.job
        started.append((spec.worker, str(job["worker"]), int(job["ordinal"])))
        attempt_root = executor["attempt_root"]
        assert isinstance(attempt_root, Path)
        log_path = attempt_root / f"fake-{int(job['ordinal']):05d}.log"
        log = log_path.open("ab")
        return {
            "spec": spec,
            "lease": lease,
            "process": FakeProcess(job),
            "log": log,
            "log_path": log_path,
            "last_heartbeat_index": 0,
        }

    monkeypatch.setattr(scheduler_module, "_start_dynamic_assignment", fake_start)


def _run_dynamic_slots(
    fixture: dict[str, object], fake_worker: Path, *, scheduler_instance: str
) -> dict[str, object]:
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
        poll_interval=0.001,
        dynamic_queue=True,
        scheduler_instance=scheduler_instance,
        queue_poll_interval=0.001,
        claim_lease_seconds=30,
        claim_heartbeat_seconds=5,
        max_claim_attempts=10,
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


def test_scheduler_preserves_and_resumes_sibling_after_slot_failure(tmp_path: Path) -> None:
    fixture = _empty_slot_scheduler_fixture(tmp_path)
    fake_worker = _fake_source_worker(
        tmp_path / "failing-worker.py",
        """
        import os
        import sys

        if os.environ["CPPMEGA_SOURCE_SLOT"] == "worker-0000":
            sys.exit(7)
        sys.exit(0)
        """,
    )
    with pytest.raises(RuntimeError, match=r"worker-0000 exit=7"):
        _run_empty_slots(fixture, fake_worker)
    store = fixture["store"]
    manifest = fixture["manifest"]
    scheduler_receipt = fixture["scheduler_receipt"]
    stage = fixture["stage"]
    assert isinstance(store, LocalObjectStore)
    assert isinstance(manifest, dict)
    assert isinstance(scheduler_receipt, Path)
    assert isinstance(stage, Path)
    assert store.describe_if_present(slot_completion_uri(manifest, "worker-0000")) is None
    assert store.describe_if_present(slot_completion_uri(manifest, "worker-0001")) is not None
    assert not scheduler_receipt.exists()
    first_attempts = sorted(stage.glob("slots/worker-*/attempt-*"))
    assert len(first_attempts) == 2

    repaired_worker = _fake_source_worker(
        tmp_path / "repaired-worker.py",
        """
        import sys
        sys.exit(0)
        """,
    )
    receipt = _run_empty_slots(fixture, repaired_worker)
    assert receipt["completed_slots"] == ["worker-0000", "worker-0001"]
    assert receipt["resumed_slots"] == ["worker-0001"]
    assert scheduler_receipt.is_file()
    assert len(list((stage / "slots" / "worker-0000").glob("attempt-*"))) == 2
    assert len(list((stage / "slots" / "worker-0001").glob("attempt-*"))) == 1


def test_scheduler_preserves_sibling_after_transient_slot_failure(tmp_path: Path) -> None:
    fixture = _empty_slot_scheduler_fixture(tmp_path)
    fake_worker = _fake_source_worker(
        tmp_path / "transient-worker.py",
        """
        import os
        import sys

        if os.environ["CPPMEGA_SOURCE_SLOT"] == "worker-0000":
            sys.exit(75)
        sys.exit(0)
        """,
    )
    with pytest.raises(TransientTransportError, match="worker-0000"):
        _run_empty_slots(fixture, fake_worker)
    store = fixture["store"]
    manifest = fixture["manifest"]
    assert isinstance(store, LocalObjectStore)
    assert isinstance(manifest, dict)
    assert store.describe_if_present(slot_completion_uri(manifest, "worker-0000")) is None
    assert store.describe_if_present(slot_completion_uri(manifest, "worker-0001")) is not None


def test_dynamic_scheduler_refills_fast_slot_and_steals_slow_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _empty_slot_scheduler_fixture(
        tmp_path, repositories=_queue_repositories(4)
    )
    fake_worker = _fake_source_worker(tmp_path / "unused-worker.py", "import sys\n")
    started: list[tuple[str, str, int]] = []
    _install_fake_dynamic_launcher(
        monkeypatch=monkeypatch,
        fixture=fixture,
        poll_delays={0: 40},
        exit_codes={},
        started=started,
    )

    receipt = _run_dynamic_slots(fixture, fake_worker, scheduler_instance="vm-0.boot-a")

    assert receipt["scheduler_mode"] == DYNAMIC_SCHEDULER_MODE
    assert receipt["executed_assignment_count"] == 4
    assert receipt["source_receipt_count"] == 4
    assert receipt["completed_slots"] == ["worker-0000", "worker-0001"]
    assert ("worker-0001", "worker-0000", 2) in started
    assert [ordinal for _executor, _owner, ordinal in started] == [0, 1, 3, 2]

    second = _run_dynamic_slots(fixture, fake_worker, scheduler_instance="vm-0.boot-a")
    assert second["executed_assignment_count"] == 0
    assert second["resumed_slots"] == ["worker-0000", "worker-0001"]
    assert len(started) == 4


def test_dynamic_scheduler_preserves_sibling_and_publishes_deterministic_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _empty_slot_scheduler_fixture(
        tmp_path, repositories=_queue_repositories(4)
    )
    fake_worker = _fake_source_worker(tmp_path / "unused-worker.py", "import sys\n")
    started: list[tuple[str, str, int]] = []
    _install_fake_dynamic_launcher(
        monkeypatch=monkeypatch,
        fixture=fixture,
        poll_delays={1: 8},
        exit_codes={0: 2},
        started=started,
    )

    with pytest.raises(RuntimeError, match="queue drained with 1 deterministic"):
        _run_dynamic_slots(fixture, fake_worker, scheduler_instance="vm-0.boot-a")

    manifest = fixture["manifest"]
    store = fixture["store"]
    assert isinstance(manifest, dict)
    assert isinstance(store, LocalObjectStore)
    failed_job = manifest["repositories"][0]
    assert isinstance(failed_job, dict)
    assert (
        _load_completed_assignment(
            manifest=manifest,
            manifest_file_sha256=str(fixture["manifest_file_sha"]),
            job=failed_job,
            object_store=store,
            scratch_root=tmp_path / "failed-check",
        )
        is None
    )
    for sibling_job in manifest["repositories"][1:]:
        assert isinstance(sibling_job, dict)
        assert (
            _load_completed_assignment(
                manifest=manifest,
                manifest_file_sha256=str(fixture["manifest_file_sha"]),
                job=sibling_job,
                object_store=store,
                scratch_root=tmp_path / f"sibling-{sibling_job['ordinal']}-check",
            )
            is not None
        )
    assert len(started) == 4
    assert any(ordinal == 2 for _executor, _owner, ordinal in started)
    scheduler_receipt = fixture["scheduler_receipt"]
    assert isinstance(scheduler_receipt, Path)
    incomplete = json.loads(scheduler_receipt.read_text(encoding="utf-8"))
    assert incomplete["status"] == "incomplete"
    assert incomplete["assignment_count"] == 4
    assert incomplete["completed_assignment_count"] == 3
    assert incomplete["terminal_assignment_count"] == 1
    assert incomplete["unresolved_assignment_count"] == 0
    assert incomplete["terminal_assignments"][0]["assignment"]["ordinal"] == 0
    assert (
        store.describe_if_present(dynamic_incomplete_receipt_uri(manifest, 0))
        is not None
    )

    with pytest.raises(RuntimeError, match="queue drained with 1 deterministic"):
        _run_dynamic_slots(fixture, fake_worker, scheduler_instance="vm-0.boot-b")
    assert len(started) == 4


def test_dynamic_incomplete_accounting_rejects_completion_terminal_overlap(
    tmp_path: Path,
) -> None:
    fixture = _empty_slot_scheduler_fixture(
        tmp_path, repositories=_queue_repositories(1)
    )
    manifest = fixture["manifest"]
    assert isinstance(manifest, dict)
    job = manifest["repositories"][0]
    assert isinstance(job, dict)
    assignment_sha256 = str(job["assignment_sha256"])
    specs = slot_specs(
        physical_worker_index=0,
        physical_worker_count=1,
        slots_per_worker=2,
    )
    resources = validate_slot_resources(
        slots_per_worker=2,
        parse_workers_per_slot=6,
        memory_limit_gb_per_slot=24,
        cpu_budget_vcpus=16,
        memory_budget_gb=56,
    )

    with pytest.raises(ContractError, match="accounting overlaps"):
        scheduler_module._dynamic_incomplete_receipt(
            manifest=manifest,
            manifest_file_sha256=str(fixture["manifest_file_sha"]),
            specs=specs,
            resources=resources,
            completed_assignments={assignment_sha256},
            terminal_assignments={assignment_sha256: {}},
        )
