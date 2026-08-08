from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)
from scripts.distributed_data_prep.cloud_lane import build_cloud_lane_manifest
from scripts.distributed_data_prep.cloud_lane_pool_worker import (
    ConfirmedHttp429,
    POOL_COMPLETION_SCHEMA,
    POOL_FAILURE_SCHEMA,
    pool_completion_sha256,
    run_cloud_lane_pool_worker,
)
from scripts.distributed_data_prep.source_worker import LocalObjectStore


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _adapter(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
request = json.load(open(a.request, encoding='utf-8'))
assignment = request['assignment']
with open(a.output, 'x', encoding='utf-8') as stream:
    for ordinal in range(assignment['record_start'], assignment['record_start'] + assignment['record_count']):
        value = {
            'schema': request['output_schema'],
            'source_record_ordinal': ordinal,
            'document_ordinal': 0,
            'valid_tokens': ordinal + 1,
            'payload': {'ordinal': ordinal},
        }
        stream.write(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\\n')
""",
        encoding="utf-8",
    )
    return path


class CountingStore:
    def __init__(self, delegate: LocalObjectStore) -> None:
        self.delegate = delegate
        self.input_downloads = 0

    def publish_if_absent(self, source: Path, uri: str):
        return self.delegate.publish_if_absent(source, uri)

    def download(self, uri: str, destination: Path, *, generation=None):
        if uri.startswith("gs://inputs/"):
            self.input_downloads += 1
        return self.delegate.download(uri, destination, generation=generation)


@pytest.fixture
def pool_fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    adapter = _adapter(repo / "adapter.py")
    _git(repo, "add", "adapter.py")
    _git(repo, "commit", "--quiet", "-m", "fixture adapter")
    revision = _git(repo, "rev-parse", "HEAD")

    delegate = LocalObjectStore(tmp_path / "objects")
    snapshots = []
    for name, role, records in (
        ("ancillary", "ancillary", 1),
        ("membership", "membership", 4),
        ("primary", "primary", 4),
    ):
        path = tmp_path / f"{name}.jsonl"
        path.write_bytes(
            b"".join(
                canonical_json_bytes({"ordinal": ordinal}) + b"\n"
                for ordinal in range(records)
            )
        )
        uri = f"gs://inputs/{name}.jsonl"
        publication = delegate.publish_if_absent(path, uri)
        snapshots.append(
            {
                "name": name,
                "role": role,
                "uri": uri,
                "generation": publication["generation"],
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "content_set_sha256": _sha(f"content:{name}"),
                "schema_sha256": _sha(f"schema:{name}"),
                "format": "canonical-jsonl-v1",
                "record_count": records,
            }
        )
    manifest = build_cloud_lane_manifest(
        kind="ci",
        input_snapshots=snapshots,
        work_items=[
            {
                "item_id": f"range/{ordinal:06d}-{ordinal + 1:06d}",
                "record_start": ordinal,
                "record_count": 1,
                "partition_sha256": _sha(f"partition:{ordinal}"),
            }
            for ordinal in range(4)
        ],
        worker_count=4,
        gcs_output_prefix="gs://outputs/lane-001",
        code_revision=revision,
        runner_sha256=sha256_file(adapter),
        tokenizer_sha256=_sha("tokenizer"),
        dataset_schema_sha256=_sha("dataset"),
        membership_policy_sha256=_sha("membership-policy"),
        candidate_schema_sha256=_sha("candidate-schema"),
    )
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "tmp_path": tmp_path,
        "repo": repo,
        "adapter": adapter,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "delegate": delegate,
    }


def _run(pool_fixture, *, store=None):
    return run_cloud_lane_pool_worker(
        manifest_path=pool_fixture["manifest_path"],
        adapter_path=pool_fixture["adapter"],
        repo_root=pool_fixture["repo"],
        physical_index=0,
        physical_count=2,
        slots=2,
        control_prefix="gs://control/lane-001",
        stage_root=pool_fixture["tmp_path"] / "stage",
        object_store=store or pool_fixture["delegate"],
    )


def test_pool_worker_maps_logical_workers_and_reuses_one_snapshot_cache(
    pool_fixture,
) -> None:
    store = CountingStore(pool_fixture["delegate"])
    result = _run(pool_fixture, store=store)
    receipt = result["receipt"]

    assert receipt["schema"] == POOL_COMPLETION_SCHEMA
    assert receipt["logical_workers"] == ["worker-0000", "worker-0002"]
    assert receipt["totals"] == {
        "source_record_count": 2,
        "candidate_document_count": 2,
        "valid_tokens": 4,
        "assignment_receipt_count": 2,
    }
    assert receipt["training_ready"] is False
    assert pool_completion_sha256(receipt) == receipt["receipt_sha256"]
    assert store.input_downloads == len(pool_fixture["manifest"]["input_snapshots"])
    assert result["publication"]["uri"].endswith("physical-0000.complete.json")

    repeated = _run(pool_fixture, store=store)
    assert repeated == result
    assert store.input_downloads == len(pool_fixture["manifest"]["input_snapshots"])


def test_pool_worker_marks_only_explicit_429_as_retryable(
    pool_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.distributed_data_prep.cloud_lane_pool_worker as module

    def fail_with_429(**_kwargs):
        raise RuntimeError("exact GCS request failed with HTTP 429")

    monkeypatch.setattr(module, "run_cloud_lane_worker", fail_with_429)
    with pytest.raises(ConfirmedHttp429, match="confirmed HTTP 429 diagnostics"):
        _run(pool_fixture)
    failure = json.loads(
        (pool_fixture["tmp_path"] / "stage/receipts/physical-0000.failed.json").read_text()
    )
    assert failure["schema"] == POOL_FAILURE_SCHEMA
    assert failure["retry_exit_code"] == 75
    assert all(item["confirmed_http_429"] for item in failure["diagnostics"])
    assert all("exact GCS request" not in json.dumps(item) for item in failure["diagnostics"])


def test_pool_worker_keeps_deterministic_failure_at_exit_two(
    pool_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.distributed_data_prep.cloud_lane_pool_worker as module

    def fail_deterministically(**_kwargs):
        raise RuntimeError("adapter contract drifted")

    monkeypatch.setattr(module, "run_cloud_lane_worker", fail_deterministically)
    with pytest.raises(ContractError, match="deterministic diagnostics"):
        _run(pool_fixture)
    failure = json.loads(
        (pool_fixture["tmp_path"] / "stage/receipts/physical-0000.failed.json").read_text()
    )
    assert failure["retry_exit_code"] == 2
    assert not any(item["confirmed_http_429"] for item in failure["diagnostics"])
