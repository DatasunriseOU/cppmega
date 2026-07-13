from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

import scripts.data.publish_megatron_bundle_to_nebius_s3 as publisher
from scripts.data.publish_megatron_bundle_to_nebius_s3 import (
    _head,
    _head_matches,
    _upload_file,
    _validate_archive,
    _validate_archive_member_names,
    _validate_bundle,
    main,
)


def _bundle(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    artifact = prefix.with_suffix(".bin")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return artifact, digest


def _archive(tmp_path, artifact):
    raw_archive = tmp_path / "bundle.tar"
    archive = tmp_path / "bundle.tar.zst"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    with tarfile.open(raw_archive, "w") as tar:
        for record in manifest["artifacts"]:
            tar.add(tmp_path / record["path"], arcname=record["path"])
        tar.add(tmp_path / "manifest.json", arcname="manifest.json")
    subprocess.run(
        ["zstd", "-q", "-1", str(raw_archive), "-o", str(archive)], check=True
    )
    return archive


def _write_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _prefix_bundle(tmp_path):
    prefix = tmp_path / "data" / "seq_1024" / "cppmega_train"
    token_count = 3
    document_count = 1
    _write_bytes(prefix.with_suffix(".bin"), b"\x01\x00\x02\x00\x03\x00")
    _write_bytes(
        prefix.with_suffix(".idx"),
        b"MMIDIDX\x00\x00"
        + struct.pack("<QBQQ", 1, 8, document_count, document_count + 1)
        + struct.pack("<i", token_count)
        + struct.pack("<q", 0)
        + struct.pack("<2q", 0, document_count),
    )

    side_channel_paths = {}
    for name in sorted(publisher.REQUIRED_TOKEN_SIDECARS):
        dtype = "uint8" if name in {
            "loss_mask",
            "token_confidence_ids",
            "token_structure_ids",
            "token_def_use",
            "token_change_mask_pre",
            "token_change_mask_post",
        } else "uint16"
        if name in {
            "token_entity_ids",
            "token_scope_ids",
            "token_symbol_ids",
            "token_call_targets",
            "token_type_refs",
        }:
            dtype = "uint32"
        rel = f"{prefix.name}_{name}.bin"
        payload = bytearray(token_count * publisher.DTYPE_SIZES[dtype])
        if name == "token_structure_ids":
            payload[0] = 1
        _write_bytes(prefix.parent / rel, payload)
        side_channel_paths[name] = {"path": rel, "dtype": dtype}

    graph_sidecar_paths = {}
    for name in sorted(publisher.REQUIRED_GRAPH_SIDECARS):
        if name in {"token_call_edges", "token_type_edges"}:
            kind = "edge_pairs"
            dtype = "int32"
            shape_tail = [2]
            item_count = 0
        elif name.endswith("_edges"):
            kind = "edge_triples"
            dtype = "int32"
            shape_tail = [3]
            item_count = 0
        else:
            kind = "ragged_1d"
            dtype = "uint32" if name in {"token_chunk_starts", "token_chunk_ends"} else "uint16"
            shape_tail = [1]
            item_count = 1
        offsets_rel = f"{prefix.name}_{name}_offsets.bin"
        data_rel = f"{prefix.name}_{name}_data.bin"
        _write_bytes(prefix.parent / offsets_rel, struct.pack("<2q", 0, item_count))
        _write_bytes(
            prefix.parent / data_rel,
            b"\x00" * (item_count * shape_tail[0] * publisher.DTYPE_SIZES[dtype]),
        )
        graph_sidecar_paths[name] = {
            "kind": kind,
            "offsets_path": offsets_rel,
            "data_path": data_rel,
            "offset_dtype": "int64",
            "dtype": dtype,
            "item_count": item_count,
            "shape_tail": shape_tail,
        }

    source_platform = {
        "schema": "cppmega_source_platform_v1",
        "sequence_doc_offsets_path": f"{prefix.name}_source_platform_sequence_doc_offsets.bin",
        "doc_platform_offsets_path": f"{prefix.name}_source_platform_doc_id_offsets.bin",
        "platform_ids_path": f"{prefix.name}_source_platform_ids.bin",
        "source_document_count": 1,
        "platform_id_count": 1,
    }
    _write_bytes(prefix.parent / source_platform["sequence_doc_offsets_path"], struct.pack("<2q", 0, 1))
    _write_bytes(prefix.parent / source_platform["doc_platform_offsets_path"], struct.pack("<2q", 0, 1))
    _write_bytes(prefix.parent / source_platform["platform_ids_path"], b"\x01\x00")
    prefix_manifest = {
        "vocab_size": 65536,
        "tokenizer_contract": "megacpp",
        "dtype": "uint16",
        "token_count": token_count,
        "document_count": document_count,
        "graph_sidecar_schema": "cppmega_graph_routes_v2",
        "side_channel_paths": side_channel_paths,
        "graph_sidecar_paths": graph_sidecar_paths,
        "source_platform_sidecar": source_platform,
    }
    prefix.with_suffix(".json").write_text(json.dumps(prefix_manifest), encoding="utf-8")

    tokenizer_dir = tmp_path / "tokenizer"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "data/tokenizer_v2",
        tokenizer_dir,
    )

    paths = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    records = [
        {
            "path": str(path.relative_to(tmp_path)),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    artifact_set_sha256 = hashlib.sha256(
        json.dumps(records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    tokenizer_records = [
        record for record in records if str(record["path"]).startswith("tokenizer/")
    ]
    tokenizer_set_sha256 = hashlib.sha256(
        json.dumps(tokenizer_records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema": "cppmega_megatron_bundle_v1",
        "bundle_id": f"test-bundle-{artifact_set_sha256[:16]}",
        "tokenizer_contract": "megacpp-vocab-65536",
        "vocab_size": 65536,
        "buckets": [1024],
        "tokenizer": {
            "path": "tokenizer",
            "contract": "megacpp-vocab-65536",
            "vocab_size": 65536,
            "files": tokenizer_records,
            "artifact_set_sha256": tokenizer_set_sha256,
        },
        "bucket_results": [{"bucket": 1024, "prefix": str(prefix.relative_to(tmp_path)), "manifest": prefix_manifest}],
        "artifact_count": len(records),
        "artifact_bytes": sum(int(record["size"]) for record in records),
        "artifact_set_sha256": artifact_set_sha256,
        "artifacts": records,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return prefix


def _rehash_bundle_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for record in manifest["artifacts"]:
        path = tmp_path / record["path"]
        records.append(
            {
                "path": record["path"],
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    records.sort(key=lambda record: record["path"])
    artifact_set_sha256 = hashlib.sha256(
        json.dumps(records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest["artifacts"] = records
    manifest["artifact_count"] = len(records)
    manifest["artifact_bytes"] = sum(record["size"] for record in records)
    manifest["artifact_set_sha256"] = artifact_set_sha256
    manifest["bundle_id"] = f"test-bundle-{artifact_set_sha256[:16]}"
    tokenizer_records = [
        record for record in records if record["path"].startswith("tokenizer/")
    ]
    manifest["tokenizer"]["files"] = tokenizer_records
    manifest["tokenizer"]["artifact_set_sha256"] = hashlib.sha256(
        json.dumps(tokenizer_records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_validate_bundle_rehashes_every_manifest_artifact(tmp_path):
    artifact, digest = _bundle(tmp_path)

    manifest, records = _validate_bundle(tmp_path, hash_jobs=2)

    assert manifest["artifact_bytes"] == sum(record["size"] for record in records)
    artifact_record = next(record for record in records if record["local_path"] == str(artifact))
    assert artifact_record["sha256"] == digest


def test_validate_bundle_rejects_manifest_path_escape(tmp_path):
    artifact, digest = _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "../sample.bin"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe artifact path"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_artifact_count_mismatch(tmp_path):
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_count"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_missing_per_file_sha256(tmp_path):
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0].pop("sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="valid sha256"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_wrong_tokenizer_contract(tmp_path):
    _bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer_contract"] = "wrong-tokenizer"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer_contract"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_requires_hashed_tokenizer_artifacts(tmp_path):
    _prefix_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("tokenizer")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer descriptor"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_tokenizer_vocab_drift_even_when_rehashed(tmp_path):
    _prefix_bundle(tmp_path)
    tokenizer_path = tmp_path / "tokenizer/tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model"]["vocab"].pop(next(iter(tokenizer["model"]["vocab"])))
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    _rehash_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="tokenizer vocab size"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_tokenizer_rejects_same_size_special_token_id_drift(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "data/tokenizer_v2",
        tokenizer_dir,
    )
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model"]["vocab"]["<BOS>"] = 3
    tokenizer["model"]["vocab"]["<EOS>"] = 2
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")

    with pytest.raises(ValueError, match="token.*<BOS>.*(disagrees|must remain)"):
        publisher._validate_tokenizer_directory(tokenizer_dir)


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (9, struct.pack("<Q", 2), "MMIDIDX version"),
        (17, struct.pack("<B", 4), "MMIDIDX dtype"),
        (38, struct.pack("<q", 2), "sequence pointers"),
        (46, struct.pack("<q", 1), "document indices"),
    ],
)
def test_validate_prefix_rejects_mmididx_contract_drift(
    tmp_path, offset, replacement, message
):
    prefix = _prefix_bundle(tmp_path)
    index_path = prefix.with_suffix(".idx")
    payload = bytearray(index_path.read_bytes())
    payload[offset : offset + len(replacement)] = replacement
    index_path.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_prefix_rejects_in_tree_sidecar_symlink(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    sidecar = prefix.parent / manifest["side_channel_paths"]["loss_mask"]["path"]
    sidecar.unlink()
    sidecar.symlink_to(prefix.with_suffix(".bin"))

    with pytest.raises(ValueError, match="regular file.*symlink"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_validate_bundle_rejects_embedded_prefix_manifest_drift(tmp_path):
    _prefix_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bucket_results"][0]["manifest"]["vocab_size"] = 42
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="embedded prefix manifest"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_wrong_prefix_graph_schema(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["graph_sidecar_schema"] = "flat_tokens_only"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="graph_sidecar_schema"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_bundle_rejects_bad_graph_csr_offsets(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    bad_offsets = prefix.parent / f"{prefix.name}_token_chunk_starts_offsets.bin"
    bad_offsets.write_bytes(struct.pack("<2q", 1, 1))

    with pytest.raises(ValueError, match="CSR offsets"):
        _validate_bundle(tmp_path, hash_jobs=1)


def test_validate_prefix_rejects_misaligned_graph_chunk_csr_counts(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    manifest_path = prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest["graph_sidecar_paths"]["token_chunk_ends"]
    spec["item_count"] = 2
    (prefix.parent / spec["offsets_path"]).write_bytes(struct.pack("<2q", 0, 2))
    (prefix.parent / spec["data_path"]).write_bytes(struct.pack("<2I", 1, 2))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="chunk CSR item counts"):
        publisher._validate_prefix_manifest_contract(prefix)


def test_head_contract_requires_size_and_sha_metadata():
    assert _head_matches(
        {"ContentLength": 8, "Metadata": {"sha256": "abc"}},
        size=8,
        sha256="abc",
    )
    assert not _head_matches(
        {"ContentLength": 7, "Metadata": {"sha256": "abc"}},
        size=8,
        sha256="abc",
    )
    assert not _head_matches(
        {"ContentLength": 8, "Metadata": {}}, size=8, sha256="abc"
    )


def test_head_contract_accepts_nebius_metadata_key_casing():
    assert _head_matches(
        {"ContentLength": 8, "Metadata": {"Sha256": "abc"}},
        size=8,
        sha256="abc",
    )


def test_head_distinguishes_missing_object_from_transport_failure(monkeypatch):
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=254, stdout="", stderr="An error occurred (404) when calling HeadObject"
        ),
    )
    assert _head(endpoint="https://s3.invalid", bucket="b", key="missing", env={}) is None

    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=255, stdout="", stderr="Could not connect to endpoint"
        ),
    )
    with pytest.raises(RuntimeError, match="remote HEAD failed"):
        _head(endpoint="https://s3.invalid", bucket="b", key="unknown", env={})


def test_dry_run_upload_never_calls_aws(tmp_path):
    artifact, digest = _bundle(tmp_path)

    receipt = _upload_file(
        local=artifact,
        endpoint="https://example.invalid",
        bucket="bucket",
        key="prefix/sample.bin",
        size=artifact.stat().st_size,
        sha256=digest,
        env={},
        dry_run=True,
    )

    assert receipt["status"] == "dry_run"
    assert receipt["sha256"] == digest


def test_immutable_bundle_object_rejects_existing_remote_mismatch(
    tmp_path, monkeypatch
):
    artifact, digest = _bundle(tmp_path)
    monkeypatch.setattr(
        publisher,
        "_head",
        lambda **_kwargs: {
            "ContentLength": artifact.stat().st_size,
            "Metadata": {"sha256": "different"},
        },
    )

    def forbidden_upload(*_args, **_kwargs):
        raise AssertionError("immutable mismatch must fail before aws s3 cp")

    monkeypatch.setattr(publisher.subprocess, "run", forbidden_upload)
    with pytest.raises(RuntimeError, match="immutable remote object mismatch"):
        _upload_file(
            local=artifact,
            endpoint="https://example.invalid",
            bucket="bucket",
            key="bundles/test-bundle/data/sample.bin",
            size=artifact.stat().st_size,
            sha256=digest,
            env={},
            dry_run=False,
        )


def test_archive_member_set_must_be_exact_and_unique():
    _validate_archive_member_names(
        ["data/sample.bin", "manifest.json"],
        {"data/sample.bin", "manifest.json"},
    )
    with pytest.raises(ValueError, match="duplicate"):
        _validate_archive_member_names(
            ["manifest.json", "manifest.json"], {"manifest.json"}
        )
    with pytest.raises(ValueError, match="member set mismatch"):
        _validate_archive_member_names(
            ["manifest.json", "unexpected.bin"],
            {"manifest.json", "data/sample.bin"},
        )


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_validate_archive_binds_exact_members_and_manifest(tmp_path):
    artifact, _digest = _bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    archive = _archive(tmp_path, artifact)

    size, digest = _validate_archive(
        bundle=tmp_path, archive=archive, manifest=manifest
    )

    assert size == archive.stat().st_size
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_validate_archive_rejects_payload_that_disagrees_with_manifest(tmp_path):
    artifact, _digest = _bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    artifact.write_bytes(b"\x09\x00\x02\x00\x03\x00")
    archive = _archive(tmp_path, artifact)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate_archive(bundle=tmp_path, archive=archive, manifest=manifest)


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_archive_publish_runs_full_prefix_contract_validation(tmp_path):
    prefix = _prefix_bundle(tmp_path)
    prefix_manifest_path = prefix.with_suffix(".json")
    prefix_manifest = json.loads(prefix_manifest_path.read_text(encoding="utf-8"))
    prefix_manifest["graph_sidecar_schema"] = "stale_graph_schema"
    prefix_manifest_path.write_text(json.dumps(prefix_manifest), encoding="utf-8")
    manifest = _rehash_bundle_manifest(tmp_path)
    manifest["bucket_results"][0]["manifest"] = prefix_manifest
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive = _archive(tmp_path, prefix.with_suffix(".bin"))

    with pytest.raises(ValueError, match="graph_sidecar_schema"):
        main(["--bundle", str(tmp_path), "--archive", str(archive), "--dry-run"])


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_archive_transport_dry_run_writes_commit_order_receipt(tmp_path):
    artifact, _digest = _bundle(tmp_path)
    archive = _archive(tmp_path, artifact)

    assert main(["--bundle", str(tmp_path), "--archive", str(archive), "--dry-run"]) == 0

    receipt = json.loads(
        (tmp_path / "archive_publish_receipt.json").read_text(encoding="utf-8")
    )
    bundle_id = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )["bundle_id"]
    assert receipt["archive"]["key"].endswith(f"/{bundle_id}/bundle.tar.zst")
    assert receipt["logical_manifest"]["key"].endswith(
        f"/{bundle_id}/logical_manifest.json"
    )
    assert receipt["transport"]["key"].endswith(f"/{bundle_id}/transport.json")
    assert receipt["latest_transport"]["key"].endswith("/latest_transport.json")
    assert receipt["archive"]["status"] == "dry_run"
    assert receipt["status"] == "complete"
    assert receipt["archive_validation"] == {
        "status": "verified",
        "member_count": len(
            json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[
                "artifacts"
            ]
        )
        + 1,
        "artifact_set_sha256": json.loads(
            (tmp_path / "manifest.json").read_text(encoding="utf-8")
        )["artifact_set_sha256"],
        "logical_manifest_sha256": hashlib.sha256(
            (tmp_path / "manifest.json").read_bytes()
        ).hexdigest(),
    }


def test_loose_publish_receipt_is_incremental_and_bundle_bound(tmp_path):
    _prefix_bundle(tmp_path)

    assert main(["--bundle", str(tmp_path), "--dry-run", "--jobs", "2"]) == 0

    receipt_path = tmp_path / "publish_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["bundle_id"] == manifest["bundle_id"]
    assert receipt["artifact_set_sha256"] == manifest["artifact_set_sha256"]
    assert len(receipt["artifacts"]) == manifest["artifact_count"]

    receipt["bundle_id"] = "different-bundle"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="publish receipt binding"):
        main(["--bundle", str(tmp_path), "--dry-run"])


def test_loose_publish_rejects_stale_artifact_receipt_entry(tmp_path):
    _prefix_bundle(tmp_path)
    assert main(["--bundle", str(tmp_path), "--dry-run"]) == 0
    receipt_path = tmp_path / "publish_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"][0]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="publish receipt artifact mismatch"):
        main(["--bundle", str(tmp_path), "--dry-run"])
