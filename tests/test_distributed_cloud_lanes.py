from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
)
from scripts.distributed_data_prep.cloud_lane import (
    CANDIDATE_ENVELOPE_SCHEMA,
    LANE_KINDS,
    advance_checkpoint,
    assignments_for_worker,
    build_cloud_lane_manifest,
    build_completion_receipt,
    build_lane_completion_receipt,
    cloud_lane_manifest_sha256,
    initial_checkpoint,
    publish_checkpoint,
    publish_completion_receipt,
    publish_lane_completion_receipt,
    publish_manifest,
    publish_segment,
    resume_checkpoint,
    sealing_bindings_from_completion,
    validate_checkpoint,
    validate_cloud_lane_manifest,
    validate_completion_receipt,
    validate_lane_completion_receipt,
    verify_input_snapshots,
)
from scripts.distributed_data_prep.seal_outputs import (
    OUTPUT_MANIFEST_SCHEMA,
    TARGET_LENGTHS,
)
from scripts.distributed_data_prep.source_worker import (
    LocalObjectStore,
    compress_zstd,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _publish_snapshot(
    tmp_path: Path,
    store: LocalObjectStore,
    *,
    name: str,
    role: str,
    record_count: int,
) -> dict[str, object]:
    source = tmp_path / f"{name}.snapshot"
    source.write_bytes(f"{name}:{role}:{record_count}\n".encode("ascii"))
    uri = f"gs://lane-inputs/snapshots/{name}.snapshot"
    metadata = dict(store.publish_if_absent(source, uri))
    return {
        "name": name,
        "role": role,
        "uri": uri,
        "generation": metadata["generation"],
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "content_set_sha256": _sha(f"content:{name}"),
        "schema_sha256": _sha(f"schema:{name}"),
        "format": "snapshot-manifest-v1",
        "record_count": record_count,
    }


@pytest.fixture
def lane_fixture(tmp_path: Path):
    store = LocalObjectStore(tmp_path / "objects")
    snapshots = [
        _publish_snapshot(
            tmp_path, store, name="payload-store", role="primary", record_count=6
        ),
        _publish_snapshot(
            tmp_path,
            store,
            name="primary-membership",
            role="membership",
            record_count=6,
        ),
        _publish_snapshot(
            tmp_path, store, name="sidecars", role="ancillary", record_count=2
        ),
    ]
    work_items = [
        {
            "item_id": "range/000003-000006",
            "record_start": 3,
            "record_count": 3,
            "partition_sha256": _sha("partition-1"),
        },
        {
            "item_id": "range/000000-000003",
            "record_start": 0,
            "record_count": 3,
            "partition_sha256": _sha("partition-0"),
        },
    ]
    manifest = build_cloud_lane_manifest(
        kind="github_pr",
        input_snapshots=list(reversed(snapshots)),
        work_items=work_items,
        worker_count=2,
        gcs_output_prefix="gs://lane-output/run-001",
        code_revision="a" * 40,
        runner_sha256=_sha("runner"),
        tokenizer_sha256=_sha("tokenizer"),
        dataset_schema_sha256=_sha("dataset-schema"),
        membership_policy_sha256=_sha("membership-policy"),
        candidate_schema_sha256=_sha("candidate-schema"),
    )
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "tmp_path": tmp_path,
        "store": store,
        "snapshots": snapshots,
        "work_items": work_items,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_file_sha256": sha256_file(manifest_path),
    }


@pytest.mark.parametrize("kind", LANE_KINDS)
def test_manifest_is_deterministic_complete_and_sealing_compatible(
    lane_fixture, kind: str
) -> None:
    fixture = lane_fixture
    kwargs = {
        "kind": kind,
        "input_snapshots": fixture["snapshots"],
        "work_items": fixture["work_items"],
        "worker_count": 2,
        "gcs_output_prefix": "gs://lane-output/run-001",
        "code_revision": "a" * 40,
        "runner_sha256": _sha("runner"),
        "tokenizer_sha256": _sha("tokenizer"),
        "dataset_schema_sha256": _sha("dataset-schema"),
        "membership_policy_sha256": _sha("membership-policy"),
        "candidate_schema_sha256": _sha("candidate-schema"),
    }
    forward = build_cloud_lane_manifest(**kwargs)
    reverse = build_cloud_lane_manifest(
        **{
            **kwargs,
            "input_snapshots": list(reversed(fixture["snapshots"])),
            "work_items": list(reversed(fixture["work_items"])),
        }
    )

    assert forward == reverse
    assert forward["kind"] == kind
    assert forward["pipeline"]["target_lengths"] == list(TARGET_LENGTHS)
    assert (
        forward["pipeline"]["downstream_output_manifest_schema"]
        == OUTPUT_MANIFEST_SCHEMA
    )
    assert forward["pipeline"]["global_dedup_applied"] is False
    assert forward["pipeline"]["training_ready"] is False
    assert [assignment["record_start"] for assignment in forward["assignments"]] == [
        0,
        3,
    ]
    assert [assignment["worker"] for assignment in forward["assignments"]] == [
        "worker-0000",
        "worker-0001",
    ]
    assert len(assignments_for_worker(forward, "worker-0000")) == 1
    assert validate_cloud_lane_manifest(forward) == forward


def test_manifest_fails_closed_on_snapshot_assignment_and_ladder_drift(
    lane_fixture,
) -> None:
    manifest = lane_fixture["manifest"]

    missing_membership = [
        snapshot
        for snapshot in lane_fixture["snapshots"]
        if snapshot["role"] != "membership"
    ]
    with pytest.raises(ContractError, match="membership"):
        build_cloud_lane_manifest(
            kind="github_pr",
            input_snapshots=missing_membership,
            work_items=lane_fixture["work_items"],
            worker_count=2,
            gcs_output_prefix="gs://lane-output/run-001",
            code_revision="a" * 40,
            runner_sha256=_sha("runner"),
            tokenizer_sha256=_sha("tokenizer"),
            dataset_schema_sha256=_sha("dataset-schema"),
            membership_policy_sha256=_sha("membership-policy"),
            candidate_schema_sha256=_sha("candidate-schema"),
        )

    gap = copy.deepcopy(lane_fixture["work_items"])
    gap[1]["record_start"] = 1
    with pytest.raises(ContractError, match="contiguous"):
        build_cloud_lane_manifest(
            kind="github_pr",
            input_snapshots=lane_fixture["snapshots"],
            work_items=gap,
            worker_count=2,
            gcs_output_prefix="gs://lane-output/run-001",
            code_revision="a" * 40,
            runner_sha256=_sha("runner"),
            tokenizer_sha256=_sha("tokenizer"),
            dataset_schema_sha256=_sha("dataset-schema"),
            membership_policy_sha256=_sha("membership-policy"),
            candidate_schema_sha256=_sha("candidate-schema"),
        )

    assignment_drift = copy.deepcopy(manifest)
    assignment_drift["assignments"][0]["worker"] = "worker-0001"
    assignment_drift["manifest_sha256"] = cloud_lane_manifest_sha256(assignment_drift)
    with pytest.raises(ContractError, match="worker assignment"):
        validate_cloud_lane_manifest(assignment_drift)

    ladder_drift = copy.deepcopy(manifest)
    ladder_drift["pipeline"]["target_lengths"].pop()
    ladder_drift["manifest_sha256"] = cloud_lane_manifest_sha256(ladder_drift)
    with pytest.raises(ContractError, match="pre-sealing"):
        validate_cloud_lane_manifest(ladder_drift)


def test_exact_generation_snapshot_verification_and_manifest_publication(
    lane_fixture,
) -> None:
    manifest = lane_fixture["manifest"]
    store = lane_fixture["store"]
    scratch = lane_fixture["tmp_path"] / "scratch"

    verified = verify_input_snapshots(
        manifest, object_store=store, scratch_root=scratch
    )
    assert [snapshot["name"] for snapshot in verified] == [
        "payload-store",
        "primary-membership",
        "sidecars",
    ]
    publication = publish_manifest(
        lane_fixture["manifest_path"],
        object_store=store,
        scratch_root=scratch,
    )
    assert publication["generation"] == "1"
    assert publication["logical_sha256"] == manifest["manifest_sha256"]
    assert publication["uri"].endswith(
        f"/{lane_fixture['manifest_file_sha256']}.manifest.json"
    )

    primary = next(
        snapshot
        for snapshot in manifest["input_snapshots"]
        if snapshot["role"] == "primary"
    )
    store._path(str(primary["uri"])).write_bytes(b"tampered\n")
    with pytest.raises(ContractError, match="verification failed"):
        verify_input_snapshots(manifest, object_store=store, scratch_root=scratch)


def _compressed_segment(
    tmp_path: Path,
    name: str,
    rows: list[tuple[int, int, str, int]],
    *,
    kind: str = "github_pr",
) -> Path:
    raw = tmp_path / f"{name}.jsonl"
    raw.write_bytes(
        b"".join(
            canonical_json_bytes(
                {
                    "schema": CANDIDATE_ENVELOPE_SCHEMA,
                    "kind": kind,
                    "source_record_ordinal": record_ordinal,
                    "document_ordinal": document_ordinal,
                    "valid_tokens": valid_tokens,
                    "payload": {"text": text},
                    "payload_sha256": canonical_sha256({"text": text}),
                }
            )
            + b"\n"
            for record_ordinal, document_ordinal, text, valid_tokens in rows
        )
    )
    compressed = tmp_path / f"{name}.jsonl.zst"
    compress_zstd(raw, compressed)
    return compressed


def test_checkpoint_resume_completion_and_receipt_last_publication(
    lane_fixture,
) -> None:
    manifest = lane_fixture["manifest"]
    assignment = manifest["assignments"][0]
    store = lane_fixture["store"]
    tmp_path = lane_fixture["tmp_path"]
    scratch = tmp_path / "scratch"

    checkpoint = initial_checkpoint(
        manifest,
        assignment,
        manifest_file_sha256=lane_fixture["manifest_file_sha256"],
    )
    assert checkpoint["next_record_ordinal"] == 0

    first_file = _compressed_segment(tmp_path, "first", [(0, 0, "a", 7)])
    first = publish_segment(
        first_file,
        manifest=manifest,
        assignment=assignment,
        checkpoint=checkpoint,
        source_record_count=1,
        candidate_document_count=1,
        valid_tokens=7,
        object_store=store,
        scratch_root=scratch,
    )
    checkpoint = advance_checkpoint(
        checkpoint, first, manifest=manifest, assignment=assignment
    )
    assert checkpoint["status"] == "in_progress"
    checkpoint_path = tmp_path / "checkpoint.json"
    atomic_write_json(checkpoint_path, checkpoint)
    descriptor = publish_checkpoint(
        checkpoint_path,
        manifest=manifest,
        assignment=assignment,
        object_store=store,
        scratch_root=scratch,
    )
    resumed = resume_checkpoint(
        descriptor,
        manifest=manifest,
        assignment=assignment,
        object_store=store,
        scratch_root=scratch,
    )
    assert resumed == checkpoint

    second_file = _compressed_segment(
        tmp_path, "second", [(1, 0, "b", 7), (2, 0, "c", 8)]
    )
    second = publish_segment(
        second_file,
        manifest=manifest,
        assignment=assignment,
        checkpoint=resumed,
        source_record_count=2,
        candidate_document_count=2,
        valid_tokens=15,
        object_store=store,
        scratch_root=scratch,
    )
    complete_checkpoint = advance_checkpoint(
        resumed, second, manifest=manifest, assignment=assignment
    )
    assert complete_checkpoint["status"] == "complete"
    assert complete_checkpoint["next_record_ordinal"] == 3
    assert (
        validate_checkpoint(
            complete_checkpoint, manifest=manifest, assignment=assignment
        )
        == complete_checkpoint
    )
    complete_checkpoint_path = tmp_path / "complete-checkpoint.json"
    atomic_write_json(complete_checkpoint_path, complete_checkpoint)
    complete_checkpoint_publication = publish_checkpoint(
        complete_checkpoint_path,
        manifest=manifest,
        assignment=assignment,
        object_store=store,
        scratch_root=scratch,
    )

    receipt = build_completion_receipt(
        complete_checkpoint,
        manifest=manifest,
        assignment=assignment,
        checkpoint_publication=complete_checkpoint_publication,
    )
    assert receipt["totals"] == {
        "source_record_count": 3,
        "candidate_document_count": 3,
        "valid_tokens": 22,
        "segment_count": 2,
    }
    assert receipt["training_ready"] is False

    receipt_path = tmp_path / "receipt.json"
    atomic_write_json(receipt_path, receipt)
    receipt_publication = publish_completion_receipt(
        receipt_path,
        manifest=manifest,
        assignment=assignment,
        object_store=store,
        scratch_root=scratch,
    )
    assert receipt_publication["generation"] == "1"
    assert receipt_publication["logical_sha256"] == receipt["receipt_sha256"]
    assert store._path(str(receipt_publication["uri"])).is_file()

    with pytest.raises(ContractError, match="cover every assignment"):
        build_lane_completion_receipt(
            [(receipt, receipt_publication)],
            manifest=manifest,
            manifest_file_sha256=lane_fixture["manifest_file_sha256"],
        )

    second_assignment = manifest["assignments"][1]
    second_checkpoint = initial_checkpoint(
        manifest,
        second_assignment,
        manifest_file_sha256=lane_fixture["manifest_file_sha256"],
    )
    final_file = _compressed_segment(
        tmp_path,
        "final-assignment",
        [(3, 0, "d", 5), (4, 0, "e", 5), (5, 0, "f", 5)],
    )
    final_segment = publish_segment(
        final_file,
        manifest=manifest,
        assignment=second_assignment,
        checkpoint=second_checkpoint,
        source_record_count=3,
        candidate_document_count=3,
        valid_tokens=15,
        object_store=store,
        scratch_root=scratch,
    )
    final_checkpoint = advance_checkpoint(
        second_checkpoint,
        final_segment,
        manifest=manifest,
        assignment=second_assignment,
    )
    final_checkpoint_path = tmp_path / "final-checkpoint.json"
    atomic_write_json(final_checkpoint_path, final_checkpoint)
    final_checkpoint_publication = publish_checkpoint(
        final_checkpoint_path,
        manifest=manifest,
        assignment=second_assignment,
        object_store=store,
        scratch_root=scratch,
    )
    final_receipt = build_completion_receipt(
        final_checkpoint,
        manifest=manifest,
        assignment=second_assignment,
        checkpoint_publication=final_checkpoint_publication,
    )
    final_receipt_path = tmp_path / "final-receipt.json"
    atomic_write_json(final_receipt_path, final_receipt)
    final_receipt_publication = publish_completion_receipt(
        final_receipt_path,
        manifest=manifest,
        assignment=second_assignment,
        object_store=store,
        scratch_root=scratch,
    )

    lane_receipt = build_lane_completion_receipt(
        [
            (final_receipt, final_receipt_publication),
            (receipt, receipt_publication),
        ],
        manifest=manifest,
        manifest_file_sha256=lane_fixture["manifest_file_sha256"],
    )
    assert lane_receipt["totals"] == {
        "source_record_count": 6,
        "candidate_document_count": 6,
        "valid_tokens": 37,
        "segment_count": 3,
        "assignment_receipt_count": 2,
    }
    assert (
        validate_lane_completion_receipt(lane_receipt, manifest=manifest)
        == lane_receipt
    )
    bindings = sealing_bindings_from_completion(lane_receipt, manifest=manifest)
    assert set(bindings) == {
        "source_receipt_sha256",
        "producer_sha256",
        "tokenizer_sha256",
        "dataset_schema_sha256",
    }
    assert bindings["source_receipt_sha256"] == lane_receipt["receipt_sha256"]
    lane_receipt_path = tmp_path / "lane-receipt.json"
    atomic_write_json(lane_receipt_path, lane_receipt)
    lane_publication = publish_lane_completion_receipt(
        lane_receipt_path,
        manifest=manifest,
        object_store=store,
        scratch_root=scratch,
    )
    assert lane_publication["logical_sha256"] == lane_receipt["receipt_sha256"]


def test_segment_publication_rejects_unverified_candidate_bytes(lane_fixture) -> None:
    manifest = lane_fixture["manifest"]
    assignment = manifest["assignments"][0]
    checkpoint = initial_checkpoint(
        manifest,
        assignment,
        manifest_file_sha256=lane_fixture["manifest_file_sha256"],
    )
    bad_zstd = lane_fixture["tmp_path"] / "bad.jsonl.zst"
    bad_zstd.write_bytes(b"not-zstd")
    with pytest.raises(ContractError, match="valid ZSTD"):
        publish_segment(
            bad_zstd,
            manifest=manifest,
            assignment=assignment,
            checkpoint=checkpoint,
            source_record_count=1,
            candidate_document_count=0,
            valid_tokens=0,
            object_store=lane_fixture["store"],
            scratch_root=lane_fixture["tmp_path"] / "scratch",
        )

    valid = _compressed_segment(
        lane_fixture["tmp_path"], "count-drift", [(0, 0, "a", 4)]
    )
    with pytest.raises(ContractError, match="token count"):
        publish_segment(
            valid,
            manifest=manifest,
            assignment=assignment,
            checkpoint=checkpoint,
            source_record_count=1,
            candidate_document_count=1,
            valid_tokens=5,
            object_store=lane_fixture["store"],
            scratch_root=lane_fixture["tmp_path"] / "scratch",
        )


def test_resume_and_completion_fail_closed_on_tampering(lane_fixture) -> None:
    manifest = lane_fixture["manifest"]
    assignment = manifest["assignments"][0]
    store = lane_fixture["store"]
    tmp_path = lane_fixture["tmp_path"]
    scratch = tmp_path / "scratch"
    checkpoint = initial_checkpoint(
        manifest,
        assignment,
        manifest_file_sha256=lane_fixture["manifest_file_sha256"],
    )
    segment_file = _compressed_segment(
        tmp_path,
        "all",
        [(0, 0, "a", 4), (1, 0, "b", 4), (2, 0, "c", 4)],
    )
    segment = publish_segment(
        segment_file,
        manifest=manifest,
        assignment=assignment,
        checkpoint=checkpoint,
        source_record_count=3,
        candidate_document_count=3,
        valid_tokens=12,
        object_store=store,
        scratch_root=scratch,
    )
    complete = advance_checkpoint(
        checkpoint, segment, manifest=manifest, assignment=assignment
    )
    complete_path = tmp_path / "complete.json"
    atomic_write_json(complete_path, complete)
    complete_publication = publish_checkpoint(
        complete_path,
        manifest=manifest,
        assignment=assignment,
        object_store=store,
        scratch_root=scratch,
    )
    receipt = build_completion_receipt(
        complete,
        manifest=manifest,
        assignment=assignment,
        checkpoint_publication=complete_publication,
    )

    tampered_receipt = copy.deepcopy(receipt)
    tampered_receipt["training_ready"] = True
    with pytest.raises(ContractError, match="bindings"):
        validate_completion_receipt(
            tampered_receipt, manifest=manifest, assignment=assignment
        )

    tampered_checkpoint = copy.deepcopy(complete)
    tampered_checkpoint["segments"][0]["source_record_start"] = 1
    with pytest.raises(ContractError, match="contiguous"):
        validate_checkpoint(
            tampered_checkpoint, manifest=manifest, assignment=assignment
        )

    # A receipt is not published when a referenced generation no longer has the
    # exact bytes.  LocalObjectStore gives us a deterministic corruption probe.
    receipt_path = tmp_path / "tampered-object-receipt.json"
    atomic_write_json(receipt_path, receipt)
    store._path(str(segment["uri"])).write_bytes(b"corrupted")
    with pytest.raises(ContractError, match="readback"):
        publish_completion_receipt(
            receipt_path,
            manifest=manifest,
            assignment=assignment,
            object_store=store,
            scratch_root=scratch,
        )
    receipt_prefix = f"lane-receipts/{manifest['kind']}/{manifest['manifest_sha256']}/"
    bucket_root = lane_fixture["tmp_path"] / "objects" / "lane-output"
    assert not any(
        receipt_prefix in path.as_posix()
        for path in bucket_root.rglob("*.receipt.json")
    )
