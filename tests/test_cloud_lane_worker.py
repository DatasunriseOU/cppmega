from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)
from scripts.distributed_data_prep.cloud_lane import build_cloud_lane_manifest
from scripts.distributed_data_prep.cloud_lane_worker import (
    ADAPTER_OUTPUT_SCHEMA,
    WORKER_COMPLETION_SCHEMA,
    _download_snapshots,
    run_cloud_lane_worker,
)
from scripts.distributed_data_prep.source_worker import LocalObjectStore


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshot(
    root: Path,
    store: LocalObjectStore,
    *,
    name: str,
    role: str,
    records: int,
) -> dict[str, object]:
    path = root / f"{name}.jsonl"
    path.write_bytes(
        b"".join(canonical_json_bytes({"ordinal": index}) + b"\n" for index in range(records))
    )
    uri = f"gs://inputs/{name}.jsonl"
    published = store.publish_if_absent(path, uri)
    return {
        "name": name,
        "role": role,
        "uri": uri,
        "generation": published["generation"],
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "content_set_sha256": _sha(f"content:{name}"),
        "schema_sha256": _sha(f"schema:{name}"),
        "format": "canonical-jsonl-v1",
        "record_count": records,
    }


def _adapter(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--request', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if os.environ.get('FAIL_IF_CALLED') == '1':
    raise SystemExit(19)
request = json.loads(a.request.read_text())
assignment = request['assignment']
escape = int(os.environ.get('ESCAPE_RANGE', '0'))
with a.output.open('w', encoding='utf-8') as stream:
    for ordinal in range(assignment['record_start'], assignment['record_start'] + assignment['record_count']):
        value = {
            'schema': request['output_schema'],
            'source_record_ordinal': ordinal + escape,
            'document_ordinal': 0,
            'valid_tokens': ordinal + 1,
            'payload': {'kind': request['kind'], 'ordinal': ordinal},
        }
        stream.write(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\\n')
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def worker_fixture(tmp_path: Path):
    store = LocalObjectStore(tmp_path / "objects")
    snapshots = [
        _snapshot(tmp_path, store, name="ancillary", role="ancillary", records=1),
        _snapshot(tmp_path, store, name="membership", role="membership", records=4),
        _snapshot(tmp_path, store, name="primary", role="primary", records=4),
    ]
    adapter = _adapter(tmp_path / "adapter.py")
    adapter_sha = sha256_file(adapter)
    manifest = build_cloud_lane_manifest(
        kind="github_pr",
        input_snapshots=snapshots,
        work_items=[
            {
                "item_id": "range/000000-000002",
                "record_start": 0,
                "record_count": 2,
                "partition_sha256": _sha("partition-0"),
            },
            {
                "item_id": "range/000002-000004",
                "record_start": 2,
                "record_count": 2,
                "partition_sha256": _sha("partition-1"),
            },
        ],
        worker_count=1,
        gcs_output_prefix="gs://outputs/run-001",
        code_revision="a" * 40,
        runner_sha256=adapter_sha,
        tokenizer_sha256=_sha("tokenizer"),
        dataset_schema_sha256=_sha("dataset"),
        membership_policy_sha256=_sha("membership-policy"),
        candidate_schema_sha256=_sha("candidate-schema"),
    )
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "tmp_path": tmp_path,
        "store": store,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "adapter": adapter,
        "adapter_sha": adapter_sha,
        "ledger": tmp_path / "ledger.json",
        "scratch": tmp_path / "scratch",
        "receipts": tmp_path / "receipts",
    }


def _run(
    fixture,
    *,
    env=None,
    adapter_session=None,
    verified_snapshots=None,
):
    return run_cloud_lane_worker(
        manifest_path=fixture["manifest_path"],
        worker="worker-0000",
        adapter_command=[sys.executable, str(fixture["adapter"])],
        adapter_sha256=fixture["adapter_sha"],
        scratch_root=fixture["scratch"],
        receipt_root=fixture["receipts"],
        ledger_path=fixture["ledger"],
        object_store=fixture["store"],
        adapter_env=env,
        adapter_session=adapter_session,
        verified_snapshots=verified_snapshots,
    )


def test_worker_publishes_and_exactly_resumes_all_assignments(worker_fixture) -> None:
    first = _run(worker_fixture)

    assert first["schema"] == WORKER_COMPLETION_SCHEMA
    assert first["totals"] == {
        "source_record_count": 4,
        "candidate_document_count": 4,
        "valid_tokens": 10,
        "assignment_receipt_count": 2,
    }
    assert first["training_ready"] is False
    assert (worker_fixture["receipts"] / "worker-0000.complete.json").is_file()
    assert len(first["assignment_receipts"]) == 2

    second = _run(worker_fixture, env={"FAIL_IF_CALLED": "1"})
    assert second == first


def test_snapshot_cache_reuses_verified_bytes_and_repairs_tampering(
    worker_fixture,
) -> None:
    class CountingStore:
        def __init__(self, delegate):
            self.delegate = delegate
            self.downloads = 0

        def download(self, uri, destination, *, generation=None):
            self.downloads += 1
            return self.delegate.download(uri, destination, generation=generation)

    store = CountingStore(worker_fixture["store"])
    cache = worker_fixture["tmp_path"] / "shared-snapshot-cache"
    first = _download_snapshots(
        worker_fixture["manifest"], object_store=store, input_root=cache
    )
    assert store.downloads == len(worker_fixture["manifest"]["input_snapshots"])

    second = _download_snapshots(
        worker_fixture["manifest"], object_store=store, input_root=cache
    )
    assert store.downloads == len(worker_fixture["manifest"]["input_snapshots"])
    assert second == first

    Path(str(first[0]["local_path"])).write_bytes(b"tampered\n")
    repaired = _download_snapshots(
        worker_fixture["manifest"], object_store=store, input_root=cache
    )
    assert store.downloads == len(worker_fixture["manifest"]["input_snapshots"]) + 1
    assert repaired == first


def test_adapter_session_guards_freshly_downloaded_snapshots(worker_fixture) -> None:
    class TamperingSession:
        def run(self, *, request_path: Path, output_path: Path) -> dict[str, object]:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            assignment = request["assignment"]
            assert isinstance(assignment, dict)
            with output_path.open("x", encoding="utf-8") as stream:
                for ordinal in range(
                    assignment["record_start"],
                    assignment["record_start"] + assignment["record_count"],
                ):
                    stream.write(
                        json.dumps(
                            {
                                "schema": request["output_schema"],
                                "source_record_ordinal": ordinal,
                                "document_ordinal": 0,
                                "valid_tokens": ordinal + 1,
                                "payload": {"ordinal": ordinal},
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            snapshots = request["snapshots"]
            assert isinstance(snapshots, list)
            Path(str(snapshots[0]["local_path"])).write_bytes(b"tampered\n")
            return {"status": "complete"}

        def close(self) -> None:
            pass

    with pytest.raises(ContractError, match="adapter modified an immutable input snapshot"):
        _run(worker_fixture, adapter_session=TamperingSession())


def test_worker_rejects_adapter_output_outside_assignment(worker_fixture) -> None:
    with pytest.raises(ContractError, match="escaped its assigned source range"):
        _run(worker_fixture, env={"ESCAPE_RANGE": "10"})


def test_worker_rejects_tampered_resume_ledger(worker_fixture) -> None:
    _run(worker_fixture)
    value = json.loads(worker_fixture["ledger"].read_text())
    value["assignments"][0]["receipt"]["totals"]["valid_tokens"] += 1
    atomic_write_json(worker_fixture["ledger"], value)

    with pytest.raises(ContractError, match="ledger digest drifted"):
        _run(worker_fixture)


def test_worker_rejects_adapter_not_bound_to_manifest(worker_fixture) -> None:
    with pytest.raises(ContractError, match="differs from manifest runner_sha256"):
        run_cloud_lane_worker(
            manifest_path=worker_fixture["manifest_path"],
            worker="worker-0000",
            adapter_command=[sys.executable, str(worker_fixture["adapter"])],
            adapter_sha256="f" * 64,
            scratch_root=worker_fixture["scratch"],
            receipt_root=worker_fixture["receipts"],
            ledger_path=worker_fixture["ledger"],
            object_store=worker_fixture["store"],
        )


@pytest.mark.parametrize("kind", ["github_pr", "gitlab_mr", "ci"])
def test_adapter_output_contract_is_shared_across_lane_kinds(
    worker_fixture, kind: str
) -> None:
    manifest = copy.deepcopy(worker_fixture["manifest"])
    # Rebuild through the public constructor because kind participates in every digest.
    rebuilt = build_cloud_lane_manifest(
        kind=kind,
        input_snapshots=manifest["input_snapshots"],
        work_items=[
            {
                key: assignment[key]
                for key in ("item_id", "record_start", "record_count", "partition_sha256")
            }
            for assignment in manifest["assignments"]
        ],
        worker_count=1,
        gcs_output_prefix="gs://outputs/run-001",
        code_revision="a" * 40,
        runner_sha256=worker_fixture["adapter_sha"],
        tokenizer_sha256=_sha("tokenizer"),
        dataset_schema_sha256=_sha("dataset"),
        membership_policy_sha256=_sha("membership-policy"),
        candidate_schema_sha256=_sha("candidate-schema"),
    )
    atomic_write_json(worker_fixture["manifest_path"], rebuilt)
    completion = _run(worker_fixture)
    assert completion["kind"] == kind
    assert completion["totals"]["candidate_document_count"] == 4
    assert ADAPTER_OUTPUT_SCHEMA.endswith("_v1")
