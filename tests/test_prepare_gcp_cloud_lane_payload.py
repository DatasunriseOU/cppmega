from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from scripts.distributed_data_prep.cloud_lane import build_cloud_lane_manifest
from scripts.prepare_gcp_cloud_lane_payload import prepare_cloud_lane_payload


RUNNER_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "infra/gcp_corpus_pool/pilot/cloud-lane-worker-runner.sh.tmpl"
)


def _run(*args: object, cwd: Path) -> str:
    completed = subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "scripts/distributed_data_prep").mkdir(parents=True)
    (repo / "cppmega/tokenizer").mkdir(parents=True)
    (repo / "scripts/distributed_data_prep/ci_case5_snapshot.py").write_text(
        "print('adapter')\n", encoding="utf-8"
    )
    (repo / "cppmega/tokenizer/tokenizer.json").write_text(
        '{"version":"test"}\n', encoding="utf-8"
    )
    _run("git", "init", "-q", cwd=repo)
    _run("git", "config", "user.email", "test@example.invalid", cwd=repo)
    _run("git", "config", "user.name", "test", cwd=repo)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "fixture", cwd=repo)
    return repo, _run("git", "rev-parse", "HEAD", cwd=repo)


def _manifest(repo: Path, revision: str, destination: Path) -> None:
    adapter_sha = sha256_file(
        repo / "scripts/distributed_data_prep/ci_case5_snapshot.py"
    )
    tokenizer_sha = sha256_file(repo / "cppmega/tokenizer/tokenizer.json")
    snapshots = [
        {
            "name": "fetch-state",
            "role": "membership",
            "uri": "gs://fixture-bucket/inputs/fetch-state.sqlite3",
            "generation": "1",
            "size_bytes": 1,
            "sha256": "1" * 64,
            "content_set_sha256": "2" * 64,
            "schema_sha256": "3" * 64,
            "format": "sqlite3",
            "record_count": 1,
        },
        {
            "name": "occurrences",
            "role": "primary",
            "uri": "gs://fixture-bucket/inputs/occurrences.jsonl",
            "generation": "2",
            "size_bytes": 1,
            "sha256": "4" * 64,
            "content_set_sha256": "5" * 64,
            "schema_sha256": "6" * 64,
            "format": "jsonl",
            "record_count": 1,
        },
    ]
    manifest = build_cloud_lane_manifest(
        kind="ci",
        input_snapshots=snapshots,
        work_items=[
            {
                "item_id": "occurrences/000000000000-000000000001",
                "record_start": 0,
                "record_count": 1,
                "partition_sha256": "7" * 64,
            }
        ],
        worker_count=4,
        gcs_output_prefix="gs://fixture-bucket/runs/output/outputs",
        code_revision=revision,
        runner_sha256=adapter_sha,
        tokenizer_sha256=tokenizer_sha,
        dataset_schema_sha256=canonical_sha256({"dataset": "fixture"}),
        membership_policy_sha256=canonical_sha256({"policy": "fixture"}),
        candidate_schema_sha256=canonical_sha256({"candidate": "fixture"}),
    )
    atomic_write_json(destination, manifest)


def _bootstrap_inputs(tmp_path: Path, repo: Path) -> tuple[Path, Path]:
    tokenizer_sha = sha256_file(repo / "cppmega/tokenizer/tokenizer.json")
    fetch = tmp_path / "fetch-receipt.json"
    fetch.write_text(
        json.dumps(
            {
                "tokenizer_contract": {
                    "library": "tokenizers",
                    "library_version": "0.23.1",
                    "artifact_sha256": tokenizer_sha,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "tokenizers==0.23.1 \\\n"
        "    --hash=sha256:" + "8" * 64 + "\n",
        encoding="utf-8",
    )
    return fetch, requirements


def test_prepare_cloud_lane_payload_is_content_addressed_and_pool_bound(
    tmp_path: Path,
) -> None:
    repo, revision = _repo(tmp_path)
    manifest = tmp_path / "manifest.json"
    _manifest(repo, revision, manifest)
    fetch, requirements = _bootstrap_inputs(tmp_path, repo)
    output = tmp_path / "payload"

    receipt = prepare_cloud_lane_payload(
        repo_root=repo,
        manifest_path=manifest,
        runner_template=RUNNER_TEMPLATE,
        requirements_path=requirements,
        fetch_receipt_path=fetch,
        output_root=output,
        run_root="gs://fixture-bucket/runs/ci-case5-smoke-001",
        project_id="fixture-project-12345",
        region="us-central1",
        zone="us-central1-a",
        physical_worker_count=1,
        slots_per_worker=2,
        machine_type="n2-standard-16",
        local_ssd_count=2,
    )

    assert receipt["status"] == "ready_not_uploaded_not_applied"
    assert receipt["training_ready"] is False
    assert receipt["code_revision"] == revision
    artifacts = {
        item["path"]: item for item in receipt["artifacts"] if "uri" in item
    }
    assert len(artifacts) == 4
    for relative, descriptor in artifacts.items():
        artifact = output / relative
        assert sha256_file(artifact) == descriptor["sha256"]
        assert artifact.name.startswith(str(descriptor["sha256"]))

    runner = next((output / "bootstrap").glob("*.cloud-lane-worker-runner"))
    runner_text = runner.read_text(encoding="utf-8")
    assert "__CPPMEGA_" not in runner_text
    assert "cloud_lane_pool_worker.py" in runner_text
    assert "cloud_lane_worker.py" not in runner_text
    subprocess.run(["bash", "-n", str(runner)], check=True)

    overlay = next((output / "bootstrap").glob("*.cloud-lane-bootstrap.tar.zst"))
    raw_tar = tmp_path / "overlay.tar"
    with raw_tar.open("wb") as stream:
        subprocess.run(["zstd", "-dc", str(overlay)], stdout=stream, check=True)
    with tarfile.open(raw_tar) as archive:
        assert archive.getnames() == [
            "ci-case5-linux-cp311-requirements.txt",
            "fetch-receipt.json",
        ]

    tfvars = json.loads(
        (output / "ci-case5-smoke-001.tfvars.json").read_text(encoding="utf-8")
    )
    assert tfvars["runner_role"] == "cloud-lane"
    assert tfvars["worker_count"] == 1
    assert tfvars["slots_per_worker"] == 2
    assert tfvars["bootstrap_script_gcs_uri"].endswith(
        f"/{sha256_file(runner)}.cloud-lane-worker-runner"
    )


def test_prepare_cloud_lane_payload_rejects_manifest_revision_drift(
    tmp_path: Path,
) -> None:
    repo, _revision = _repo(tmp_path)
    manifest = tmp_path / "manifest.json"
    _manifest(repo, "a" * 40, manifest)
    fetch, requirements = _bootstrap_inputs(tmp_path, repo)

    with pytest.raises(ContractError, match="code_revision"):
        prepare_cloud_lane_payload(
            repo_root=repo,
            manifest_path=manifest,
            runner_template=RUNNER_TEMPLATE,
            requirements_path=requirements,
            fetch_receipt_path=fetch,
            output_root=tmp_path / "payload",
            run_root="gs://fixture-bucket/runs/ci-case5-smoke-001",
            project_id="fixture-project-12345",
            region="us-central1",
            zone="us-central1-a",
            physical_worker_count=1,
            slots_per_worker=1,
            machine_type="n2-standard-16",
            local_ssd_count=2,
        )
