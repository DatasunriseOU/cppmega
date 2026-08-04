from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts.ci_stream_fetch import ExactTokenizer
from scripts.distributed_data_prep._common import ContractError, atomic_write_json, sha256_file
from scripts.distributed_data_prep.ci_case5_snapshot import (
    ADAPTER_OUTPUT_SCHEMA,
    CASE5_PAYLOAD_SCHEMA,
    FETCH_RECEIPT_NAME,
    MODE_PRODUCTION,
    MODE_THRESHOLD,
    build_ci_case5_manifest,
    partition_ci_occurrences,
    prepare_ci_case5_snapshot,
    run_ci_case5_adapter,
    validate_snapshot_set,
)
from scripts.distributed_data_prep.cloud_lane_worker import run_cloud_lane_worker
from scripts.distributed_data_prep.source_worker import LocalObjectStore
from scripts.export_ci_content_store_case5 import FrozenFetchState, FrozenStore
from tests.test_export_ci_content_store_case5 import (
    TOKENIZER_JSON,
    _build_store,
    _provenance,
)


def _fetch_receipt(
    *,
    store_root: Path,
    store_receipt_path: Path,
    fetch_state: Path,
    tokenizer: ExactTokenizer,
) -> dict[str, object]:
    store_receipt = json.loads(store_receipt_path.read_text(encoding="utf-8"))
    with FrozenStore(store_root, store_receipt_path) as store:
        with FrozenFetchState(
            fetch_state,
            tokenizer=tokenizer,
            store=store,
        ) as frozen:
            frozen_binding = frozen.receipt_binding()
    return {
        "schema": "cppmega_ci_stream_fetch_receipt_v3",
        "content_store_receipt": store_receipt,
        "frozen_fetch_state": frozen_binding,
        "fetch_state": frozen_binding["summary"],
        "tokenizer_contract": tokenizer.contract,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "target_exact_unique_payload_tokens": 0,
        "completed_at": "2026-08-04T00:00:00Z",
    }


@pytest.fixture
def case5_fixture(tmp_path: Path) -> dict[str, Any]:
    tokenizer = ExactTokenizer(TOKENIZER_JSON)
    records = [
        ("compile alpha.cpp", _provenance("compile alpha.cpp", ordinal=0)),
        ("test beta.cpp", _provenance("test beta.cpp", ordinal=0)),
    ]
    # Keep the records in distinct members so member-sidecar output is visibly
    # preserved for both records.
    records[1][1]["archive"]["member"] = "test.log"
    store_root, store_receipt, fetch_state = _build_store(tmp_path, tokenizer, records)
    fetch_receipt = tmp_path / "fetch-receipt.json"
    atomic_write_json(
        fetch_receipt,
        _fetch_receipt(
            store_root=store_root,
            store_receipt_path=store_receipt,
            fetch_state=fetch_state,
            tokenizer=tokenizer,
        ),
    )
    object_store = LocalObjectStore(tmp_path / "objects")
    prepared = prepare_ci_case5_snapshot(
        store_root=store_root,
        store_receipt=store_receipt,
        fetch_state=fetch_state,
        fetch_receipt=fetch_receipt,
        tokenizer=TOKENIZER_JSON,
        object_store=object_store,
        gcs_input_prefix="gs://case5-inputs/test-run",
        source_mode=MODE_THRESHOLD,
        scratch_root=tmp_path / "scratch",
        receipt_path=tmp_path / "snapshot-set.json",
    )
    adapter = (
        Path(__file__).resolve().parents[1]
        / "scripts/distributed_data_prep/ci_case5_snapshot.py"
    )
    manifest = build_ci_case5_manifest(
        prepared["receipt"],
        worker_count=1,
        records_per_item=1,
        gcs_output_prefix="gs://case5-output/test-run",
        code_revision="a" * 40,
        adapter_path=adapter,
    )
    with FrozenStore(store_root, store_receipt) as frozen_store:
        expected_by_ordinal = {
            ordinal: {
                "text": frozen_store.read_content(
                    frozen_store.get_content_record(occurrence.content_sha256)
                ).decode("utf-8"),
                "archive_member": occurrence.provenance["archive"]["member"],
            }
            for ordinal, occurrence in enumerate(frozen_store.iter_occurrences())
        }
    return {
        "tmp_path": tmp_path,
        "tokenizer": tokenizer,
        "store": object_store,
        "prepared": prepared,
        "manifest": manifest,
        "adapter": adapter,
        "expected_by_ordinal": expected_by_ordinal,
    }


def _request(fixture: dict[str, Any], *, assignment_index: int) -> dict[str, object]:
    manifest = fixture["manifest"]
    snapshots = []
    for snapshot in manifest["input_snapshots"]:
        local = fixture["store"]._path(snapshot["uri"])
        snapshots.append({**snapshot, "local_path": str(local)})
    return {
        "schema": "cppmega.distributed_cloud_lane_adapter_request_v1",
        "kind": "ci",
        "manifest_sha256": manifest["manifest_sha256"],
        "input_snapshot_set_sha256": manifest["input_snapshot_set_sha256"],
        "assignment": manifest["assignments"][assignment_index],
        "snapshots": snapshots,
        "output_schema": ADAPTER_OUTPUT_SCHEMA,
        "training_ready": False,
    }


def test_prepare_is_idempotent_and_builds_deterministic_manifest(case5_fixture) -> None:
    prepared = case5_fixture["prepared"]
    receipt = validate_snapshot_set(prepared["receipt"])

    assert receipt["source_mode"] == MODE_THRESHOLD
    assert receipt["production_complete"] is False
    assert receipt["training_ready"] is False
    assert receipt["primary_record_count"] == 2
    assert [snapshot["name"] for snapshot in receipt["input_snapshots"]] == sorted(
        snapshot["name"] for snapshot in receipt["input_snapshots"]
    )

    manifest = case5_fixture["manifest"]
    assert manifest["kind"] == "ci"
    assert [item["record_start"] for item in manifest["assignments"]] == [0, 1]
    assert [item["record_count"] for item in manifest["assignments"]] == [1, 1]

    partitions = partition_ci_occurrences(
        2,
        records_per_item=1,
        input_snapshot_set_sha256=receipt["input_snapshot_set_sha256"],
    )
    assert [item["partition_sha256"] for item in partitions] == [
        item["partition_sha256"] for item in manifest["assignments"]
    ]

    tmp_path = case5_fixture["tmp_path"]
    repeated = prepare_ci_case5_snapshot(
        store_root=tmp_path / "store",
        store_receipt=tmp_path / "store-receipt.json",
        fetch_state=tmp_path / "fetch-state.sqlite3",
        fetch_receipt=tmp_path / "fetch-receipt.json",
        tokenizer=TOKENIZER_JSON,
        object_store=case5_fixture["store"],
        gcs_input_prefix="gs://case5-inputs/test-run",
        source_mode=MODE_THRESHOLD,
        scratch_root=tmp_path / "scratch-repeat",
    )
    assert repeated["receipt"] == receipt


def test_adapter_emits_one_canonical_lossless_candidate_per_assigned_occurrence(
    case5_fixture,
) -> None:
    request_path = case5_fixture["tmp_path"] / "request.json"
    output_path = case5_fixture["tmp_path"] / "candidate.jsonl"
    atomic_write_json(request_path, _request(case5_fixture, assignment_index=1))

    result = run_ci_case5_adapter(request_path=request_path, output_path=output_path)

    assert result["record_start"] == 1
    assert result["record_count"] == 1
    lines = output_path.read_bytes().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["schema"] == ADAPTER_OUTPUT_SCHEMA
    assert row["source_record_ordinal"] == 1
    assert row["document_ordinal"] == 0
    assert row["valid_tokens"] > 0
    payload = row["payload"]
    assert payload["schema"] == CASE5_PAYLOAD_SCHEMA
    expected = case5_fixture["expected_by_ordinal"][1]
    assert payload["content"]["text"] == expected["text"]
    assert payload["token"]["count"] == row["valid_tokens"]
    assert payload["provenance"]["archive"]["member"] == expected["archive_member"]
    assert (
        payload["membership"]["sidecar"]["sidecar_sha256"]
        == payload["provenance"]["parser_sidecar_sha256"]
    )


@pytest.mark.parametrize("kind", ["fetch-receipt", "pack"])
def test_adapter_rejects_tampered_exact_snapshot(case5_fixture, kind: str) -> None:
    request = _request(case5_fixture, assignment_index=0)
    target = next(
        snapshot
        for snapshot in request["snapshots"]
        if (
            snapshot["name"] == FETCH_RECEIPT_NAME
            if kind == "fetch-receipt"
            else snapshot["name"].startswith("pack-")
        )
    )
    Path(target["local_path"]).write_bytes(b"tampered\n")
    request_path = case5_fixture["tmp_path"] / "tampered-request.json"
    atomic_write_json(request_path, request)

    with pytest.raises(ContractError, match="bytes differ from manifest"):
        run_ci_case5_adapter(
            request_path=request_path,
            output_path=case5_fixture["tmp_path"] / "tampered-output.jsonl",
        )


def test_production_mode_refuses_v3_threshold_receipt(case5_fixture) -> None:
    tmp_path = case5_fixture["tmp_path"]
    inventory = tmp_path / "inventory.sqlite3"
    inventory.write_bytes(b"not-an-inventory")
    inventory_receipt = tmp_path / "inventory-receipt.json"
    merge_receipt = tmp_path / "merge-receipt.json"
    atomic_write_json(inventory_receipt, {"schema": "fixture"})
    atomic_write_json(merge_receipt, {"schema": "fixture"})
    store_receipt = tmp_path / "store-receipt.json"
    fetch_state = tmp_path / "fetch-state.sqlite3"
    fetch_receipt = tmp_path / "fetch-receipt.json"
    store_root = tmp_path / "store"

    with pytest.raises(ContractError, match="fetch receipt cppmega_ci_stream_fetch_receipt_v4"):
        prepare_ci_case5_snapshot(
            store_root=store_root,
            store_receipt=store_receipt,
            fetch_state=fetch_state,
            fetch_receipt=fetch_receipt,
            tokenizer=TOKENIZER_JSON,
            object_store=case5_fixture["store"],
            gcs_input_prefix="gs://case5-inputs/production-refusal",
            source_mode=MODE_PRODUCTION,
            inventory=inventory,
            inventory_receipt=inventory_receipt,
            merge_receipt=merge_receipt,
            scratch_root=tmp_path / "scratch-production",
        )
    assert case5_fixture["prepared"]["receipt"]["training_ready"] is False


def test_worker_ledger_resume_does_not_invoke_completed_case5_adapter(
    case5_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = case5_fixture["tmp_path"]
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, case5_fixture["manifest"])
    adapter = case5_fixture["adapter"]
    adapter_sha = sha256_file(adapter)

    first = run_cloud_lane_worker(
        manifest_path=manifest_path,
        worker="worker-0000",
        adapter_command=[sys.executable, str(adapter)],
        adapter_sha256=adapter_sha,
        scratch_root=tmp_path / "worker-scratch",
        receipt_root=tmp_path / "worker-receipts",
        ledger_path=tmp_path / "worker-ledger.json",
        object_store=case5_fixture["store"],
    )
    assert first["totals"]["source_record_count"] == 2

    import scripts.distributed_data_prep.cloud_lane_worker as worker_module

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("completed CASE5 assignment executed again")

    monkeypatch.setattr(worker_module, "_run_adapter", fail_if_called)
    resumed = worker_module.run_cloud_lane_worker(
        manifest_path=manifest_path,
        worker="worker-0000",
        adapter_command=[sys.executable, str(adapter)],
        adapter_sha256=adapter_sha,
        scratch_root=tmp_path / "worker-scratch",
        receipt_root=tmp_path / "worker-receipts",
        ledger_path=tmp_path / "worker-ledger.json",
        object_store=case5_fixture["store"],
    )
    assert resumed == first
