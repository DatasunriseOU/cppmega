from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import textwrap

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
    advance_checkpoint,
    build_cloud_lane_manifest,
    build_completion_receipt,
    build_lane_completion_receipt,
    initial_checkpoint,
    publish_checkpoint,
    publish_completion_receipt,
    publish_segment,
)
from scripts.distributed_data_prep.lane_materializer import (
    make_adapter_spec,
    run_lane_materializer,
)
from scripts.distributed_data_prep.seal_outputs import TARGET_LENGTHS
from scripts.distributed_data_prep.source_worker import LocalObjectStore, compress_zstd


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _publish_snapshot(
    root: Path,
    store: LocalObjectStore,
    *,
    name: str,
    role: str,
    record_count: int,
) -> dict[str, object]:
    path = root / f"{name}.snapshot"
    path.write_bytes(f"{name}:{role}:{record_count}\n".encode("ascii"))
    uri = f"gs://lane-input/{name}.snapshot"
    metadata = store.publish_if_absent(path, uri)
    return {
        "name": name,
        "role": role,
        "uri": uri,
        "generation": metadata["generation"],
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "content_set_sha256": _sha(f"content:{name}"),
        "schema_sha256": _sha(f"schema:{name}"),
        "format": "canonical-store-snapshot-v1",
        "record_count": record_count,
    }


def _write_segment(root: Path, *, kind: str) -> Path:
    raw = root / "candidate.jsonl"
    document = {
        "schema": CANDIDATE_ENVELOPE_SCHEMA,
        "kind": kind,
        "source_record_ordinal": 0,
        "document_ordinal": 0,
        "valid_tokens": 1024,
        "payload": {"body": "immutable PR candidate"},
        "payload_sha256": canonical_sha256({"body": "immutable PR candidate"}),
    }
    raw.write_bytes(canonical_json_bytes(document) + b"\n")
    compressed = root / "candidate.jsonl.zst"
    compress_zstd(raw, compressed)
    return compressed


def _complete_lane(tmp_path: Path, *, kind: str = "github_pr") -> dict[str, object]:
    store = LocalObjectStore(tmp_path / "objects")
    snapshots = [
        _publish_snapshot(
            tmp_path, store, name="payload-store", role="primary", record_count=2
        ),
        _publish_snapshot(
            tmp_path,
            store,
            name="primary-membership",
            role="membership",
            record_count=2,
        ),
        _publish_snapshot(
            tmp_path, store, name="sidecars", role="ancillary", record_count=1
        ),
    ]
    manifest = build_cloud_lane_manifest(
        kind=kind,
        input_snapshots=snapshots,
        work_items=[
            {
                "item_id": "range/000000-000002",
                "record_start": 0,
                "record_count": 2,
                "partition_sha256": _sha("partition"),
            }
        ],
        worker_count=1,
        gcs_output_prefix="gs://lane-output/run-001",
        code_revision="a" * 40,
        runner_sha256=_sha("runner"),
        tokenizer_sha256=_sha("tokenizer"),
        dataset_schema_sha256=_sha("dataset"),
        membership_policy_sha256=_sha("membership"),
        candidate_schema_sha256=_sha("candidate"),
    )
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest_file_sha256 = sha256_file(manifest_path)
    assignment = manifest["assignments"][0]
    checkpoint = initial_checkpoint(
        manifest, assignment, manifest_file_sha256=manifest_file_sha256
    )
    segment = publish_segment(
        _write_segment(tmp_path, kind=kind),
        manifest=manifest,
        assignment=assignment,
        checkpoint=checkpoint,
        source_record_count=2,
        candidate_document_count=1,
        valid_tokens=1024,
        object_store=store,
        scratch_root=tmp_path / "scratch",
    )
    checkpoint = advance_checkpoint(
        checkpoint, segment, manifest=manifest, assignment=assignment
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    atomic_write_json(checkpoint_path, checkpoint)
    checkpoint_publication = publish_checkpoint(
        checkpoint_path,
        manifest=manifest,
        assignment=assignment,
        object_store=store,
        scratch_root=tmp_path / "scratch",
    )
    receipt = build_completion_receipt(
        checkpoint,
        manifest=manifest,
        assignment=assignment,
        checkpoint_publication=checkpoint_publication,
    )
    receipt_path = tmp_path / "assignment-receipt.json"
    atomic_write_json(receipt_path, receipt)
    receipt_publication = publish_completion_receipt(
        receipt_path,
        manifest=manifest,
        assignment=assignment,
        object_store=store,
        scratch_root=tmp_path / "scratch",
    )
    completion = build_lane_completion_receipt(
        [(receipt, receipt_publication)],
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
    )
    completion_path = tmp_path / "lane-completion.json"
    atomic_write_json(completion_path, completion)
    return {
        "store": store,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "completion": completion,
        "completion_path": completion_path,
    }


_ADAPTER = r'''
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def descriptor(root, relative):
    raw = (root / relative).read_bytes()
    return {"path": relative, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write_payload(root, relative, payload, role, format_name, compression):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        **descriptor(root, relative),
        "role": role,
        "format": format_name,
        "compression": compression,
        "contract_sha256": sha("contract:" + role),
    }


def write_json(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def artifact_set(records):
    payload = [
        {"path": str(record["path"]), "size": int(record["size"]), "sha256": str(record["sha256"])}
        for record in sorted(records, key=lambda record: str(record["path"]))
    ]
    return hashlib.sha256(canonical(payload)).hexdigest()


def contracts(records):
    payload = [
        {
            "path": str(record["path"]),
            "role": str(record["role"]),
            "format": str(record["format"]),
            "compression": str(record["compression"]),
            "contract_sha256": str(record["contract_sha256"]),
        }
        for record in sorted(records, key=lambda record: str(record["path"]))
    ]
    return hashlib.sha256(canonical(payload)).hexdigest()


def mmididx(token_count):
    return b"".join((
        b"MMIDIDX\x00\x00",
        struct.pack("<Q", 1),
        struct.pack("<B", 8),
        struct.pack("<Q", 1),
        struct.pack("<Q", 2),
        struct.pack("<i", token_count),
        struct.pack("<q", 0),
        struct.pack("<qq", 0, 1),
    ))


request = json.loads(Path(os.environ["CPPMEGA_LANE_REQUEST"]).read_text(encoding="utf-8"))
payloads = Path(os.environ["CPPMEGA_LANE_PAYLOADS"]).read_bytes().splitlines()
candidates = Path(os.environ["CPPMEGA_LANE_CANDIDATES"]).read_bytes().splitlines()
assert len(payloads) == len(candidates) == request["candidate_snapshot"]["candidate_document_count"]
counter = os.environ.get("TEST_LANE_ADAPTER_COUNT")
if counter:
    count_path = Path(counter)
    count_path.write_text(str(int(count_path.read_text() or "0") + 1), encoding="ascii")

root = Path(os.environ["CPPMEGA_LANE_ARTIFACT_ROOT"])
kind = request["kind"]
bindings = dict(request["bindings"])
if os.environ.get("TEST_LANE_ADAPTER_BAD_BINDING") == "1":
    bindings["producer_sha256"] = "0" * 64
buckets = []
for length in request["target_lengths"]:
    prefix = f"{kind}/{length}"
    if length != 1024:
        relative = f"{prefix}/verified-zero.json"
        zero = {
            "schema": "cppmega.distributed_verified_zero_v1",
            "status": "verified_zero",
            "kind": kind,
            "sequence_length": length,
            "reason": "route-by-fit had no eligible rows",
            **bindings,
            "eligibility_query_sha256": sha(f"zero:{length}"),
            "document_count": 0,
            "row_count": 0,
            "valid_tokens": 0,
            "trained_tokens": 0,
        }
        write_json(root, relative, zero)
        buckets.append(
            {
                "sequence_length": length,
                "status": "verified_zero",
                "zero_receipt": descriptor(root, relative),
            }
        )
        continue
    token_count = length - 1 if os.environ.get("TEST_LANE_ADAPTER_TOKEN_DRIFT") == "1" else length
    parquet = write_payload(
        root,
        f"{prefix}/data.parquet",
        b"PAR1fake-zstd-parquet\x00" + struct.pack("<I", 1) + b"PAR1",
        "parquet",
        "parquet",
        "zstd",
    )
    data = write_payload(
        root,
        f"{prefix}/shard.bin",
        b"\x00\x00" * token_count,
        "megatron_bin",
        "megatron_mmididx_data",
        "none",
    )
    index = write_payload(
        root,
        f"{prefix}/shard.idx",
        mmididx(token_count),
        "megatron_idx",
        "megatron_mmididx_index",
        "none",
    )
    names = (
        "shard_loss_mask.bin",
        "shard_graph_offsets.bin",
        "shard_graph_data.bin",
        "shard_sequence_doc_offsets.bin",
        "shard_doc_platform_offsets.bin",
        "shard_platform_ids.bin",
        "shard_source_identity_registry.sqlite",
        "shard_objective_ids.bin",
    )
    sidecars = [
        write_payload(
            root,
            f"{prefix}/{name}",
            ("sidecar:" + name).encode("ascii"),
            "megatron_sidecar",
            "megatron_sidecar",
            "none",
        )
        for name in names
    ]
    prefix_manifest = {
        "tokenizer_contract": "megacpp",
        "vocab_size": 65536,
        "dtype": "uint16",
        "token_count": token_count,
        "document_count": 1,
        "trained_token_count": token_count,
        "loss_mask_alignment": "source_token_predicts_next_v1",
        "graph_sidecar_schema": "cppmega_graph_routes_v2",
        "side_channel_paths": {"loss_mask": {"path": "shard_loss_mask.bin", "dtype": "uint8"}},
        "graph_sidecar_paths": {
            "token_chunks": {
                "offsets_path": "shard_graph_offsets.bin",
                "data_path": "shard_graph_data.bin",
            }
        },
        "source_platform_sidecar": {
            "schema": "cppmega_source_platform_v1",
            "sequence_doc_offsets_path": "shard_sequence_doc_offsets.bin",
            "doc_platform_offsets_path": "shard_doc_platform_offsets.bin",
            "platform_ids_path": "shard_platform_ids.bin",
        },
        "source_identity_registry": {"path": "shard_source_identity_registry.sqlite"},
        "objective_contract": {"objective_id_sidecar": {"path": "shard_objective_ids.bin"}},
    }
    manifest_relative = f"{prefix}/shard.json"
    write_json(root, manifest_relative, prefix_manifest)
    megatron_manifest = {
        **descriptor(root, manifest_relative),
        "role": "megatron_manifest",
        "format": "megatron_prefix_manifest",
        "compression": "none",
        "contract_sha256": sha("contract:megatron_manifest"),
    }
    artifacts = sorted(
        [parquet, data, index, megatron_manifest, *sidecars],
        key=lambda record: record["path"],
    )
    counts = {
        "document_count": 1,
        "row_count": 1,
        "valid_tokens": token_count,
        "trained_tokens": token_count,
        "payload_artifact_count": len(artifacts),
    }
    audit = {
        "schema": "cppmega.distributed_bucket_audit_v1",
        "status": "verified",
        "kind": kind,
        "sequence_length": length,
        **bindings,
        "payload_artifact_set_sha256": artifact_set(artifacts),
        "artifact_contracts_sha256": contracts(artifacts),
        "payload_artifact_count": len(artifacts),
        "counts": counts,
        "bad_files": 0,
        "bad_rows": 0,
        "hashes_verified": True,
        "schema_verified": True,
        "parquet_verified": True,
        "megatron_verified": True,
        "packing_verified": True,
        "token_conservation_verified": True,
    }
    audit_relative = f"{prefix}/audit.json"
    write_json(root, audit_relative, audit)
    buckets.append(
        {
            "sequence_length": length,
            "status": "materialized",
            "counts": counts,
            "artifacts": artifacts,
            "audit": descriptor(root, audit_relative),
        }
    )

output = {
    "schema": "cppmega.distributed_output_manifest_v1",
    "status": "complete",
    "kind": kind,
    "sequence_lengths": request["target_lengths"],
    "bindings": bindings,
    "buckets": buckets,
}
output["manifest_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
Path(os.environ["CPPMEGA_LANE_OUTPUT_MANIFEST"]).write_text(
    json.dumps(output, sort_keys=True) + "\n", encoding="utf-8"
)
'''


def _adapter(tmp_path: Path) -> tuple[Path, object]:
    path = tmp_path / "adapter.py"
    path.write_text(textwrap.dedent(_ADAPTER), encoding="utf-8")
    return path, make_adapter_spec(
        adapter_id="test-lane-adapter",
        argv=(sys.executable, str(path.resolve())),
        entrypoint=path,
    )


def _run(lane: dict[str, object], root: Path, adapter: object) -> dict[str, object]:
    return run_lane_materializer(
        manifest_path=lane["manifest_path"],  # type: ignore[arg-type]
        completion_path=lane["completion_path"],  # type: ignore[arg-type]
        output_root=root / "output",
        scratch_root=root / "scratch",
        object_store=lane["store"],  # type: ignore[arg-type]
        adapter_spec=adapter,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("kind", ("github_pr", "gitlab_mr", "ci"))
def test_lane_materializer_runs_pinned_adapter_and_reuses_verified_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    lane = _complete_lane(tmp_path, kind=kind)
    _path, adapter = _adapter(tmp_path)
    count = tmp_path / "adapter-count"
    count.write_text("0", encoding="ascii")
    monkeypatch.setenv("TEST_LANE_ADAPTER_COUNT", str(count))

    receipt = _run(lane, tmp_path, adapter)
    assert receipt["training_ready"] is False
    assert receipt["global_seal_required"] is True
    assert receipt["target_lengths"] == list(TARGET_LENGTHS)
    assert count.read_text(encoding="ascii") == "1"
    result = (
        tmp_path
        / "output"
        / "materialized"
        / kind
        / lane["completion"]["receipt_sha256"]  # type: ignore[index]
    )
    assert (result / "output-manifest.json").is_file()
    assert (result / "artifacts" / kind / "1024" / "shard.idx").is_file()

    resumed = _run(lane, tmp_path, adapter)
    assert resumed == receipt
    assert count.read_text(encoding="ascii") == "1"


def test_lane_materializer_resumes_checkpointed_adapter_output_without_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.distributed_data_prep.lane_materializer as materializer

    lane = _complete_lane(tmp_path)
    _path, adapter = _adapter(tmp_path)
    count = tmp_path / "adapter-count"
    count.write_text("0", encoding="ascii")
    monkeypatch.setenv("TEST_LANE_ADAPTER_COUNT", str(count))
    original = materializer._publish_completed_attempt
    raised = False

    def interrupted(**kwargs):
        nonlocal raised
        if not raised:
            raised = True
            raise RuntimeError("simulated host interruption after checkpoint")
        return original(**kwargs)

    monkeypatch.setattr(materializer, "_publish_completed_attempt", interrupted)
    with pytest.raises(RuntimeError, match="simulated host interruption"):
        _run(lane, tmp_path, adapter)
    assert count.read_text(encoding="ascii") == "1"
    monkeypatch.setattr(materializer, "_publish_completed_attempt", original)

    receipt = _run(lane, tmp_path, adapter)
    assert receipt["status"] == "verified"
    assert count.read_text(encoding="ascii") == "1"


def test_lane_materializer_recovers_valid_output_before_adapter_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.distributed_data_prep.lane_materializer as materializer

    lane = _complete_lane(tmp_path)
    _path, adapter = _adapter(tmp_path)
    count = tmp_path / "adapter-count"
    count.write_text("0", encoding="ascii")
    monkeypatch.setenv("TEST_LANE_ADAPTER_COUNT", str(count))
    original = materializer._validated_adapter_output
    raised = False

    def interrupted(*args, **kwargs):
        nonlocal raised
        if not raised:
            raised = True
            raise RuntimeError("simulated interruption before output checkpoint")
        return original(*args, **kwargs)

    monkeypatch.setattr(materializer, "_validated_adapter_output", interrupted)
    with pytest.raises(RuntimeError, match="before output checkpoint"):
        _run(lane, tmp_path, adapter)
    assert count.read_text(encoding="ascii") == "1"
    monkeypatch.setattr(materializer, "_validated_adapter_output", original)

    receipt = _run(lane, tmp_path, adapter)
    assert receipt["status"] == "verified"
    assert count.read_text(encoding="ascii") == "1"


def test_lane_materializer_rejects_bad_adapter_bindings_and_snapshot_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = _complete_lane(tmp_path)
    _path, adapter = _adapter(tmp_path)
    monkeypatch.setenv("TEST_LANE_ADAPTER_BAD_BINDING", "1")
    with pytest.raises(ContractError, match="bindings"):
        _run(lane, tmp_path / "bad-adapter-output", adapter)
    monkeypatch.delenv("TEST_LANE_ADAPTER_BAD_BINDING")

    good_root = tmp_path / "good-adapter-output"
    receipt = _run(lane, good_root, adapter)
    snapshot_root = (
        good_root
        / "output"
        / "candidate-snapshots"
        / f"github_pr-{lane['completion']['receipt_sha256']}"  # type: ignore[index]
    )
    (snapshot_root / "candidates.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(ContractError, match="descriptor differs"):
        _run(lane, good_root, adapter)
    assert receipt["training_ready"] is False


def test_lane_materializer_rejects_non_lossless_adapter_token_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = _complete_lane(tmp_path)
    _path, adapter = _adapter(tmp_path)
    monkeypatch.setenv("TEST_LANE_ADAPTER_TOKEN_DRIFT", "1")
    with pytest.raises(ContractError, match="does not losslessly close"):
        _run(lane, tmp_path, adapter)
