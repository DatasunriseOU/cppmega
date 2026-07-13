from __future__ import annotations

import hashlib
import json
import shutil
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
    artifact = tmp_path / "data" / "sample.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"cppmega")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    records = [
        {
            "path": "data/sample.bin",
            "size": artifact.stat().st_size,
            "sha256": digest,
        }
    ]
    artifact_set_sha256 = hashlib.sha256(
        json.dumps(records, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema": "cppmega_megatron_bundle_v1",
        "bundle_id": f"test-bundle-{artifact_set_sha256[:16]}",
        "artifact_count": 1,
        "artifact_bytes": artifact.stat().st_size,
        "artifact_set_sha256": artifact_set_sha256,
        "artifacts": records,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return artifact, digest


def _archive(tmp_path, artifact):
    raw_archive = tmp_path / "bundle.tar"
    archive = tmp_path / "bundle.tar.zst"
    with tarfile.open(raw_archive, "w") as tar:
        tar.add(artifact, arcname="data/sample.bin")
        tar.add(tmp_path / "manifest.json", arcname="manifest.json")
    subprocess.run(
        ["zstd", "-q", "-1", str(raw_archive), "-o", str(archive)], check=True
    )
    return archive


def test_validate_bundle_rehashes_every_manifest_artifact(tmp_path):
    artifact, digest = _bundle(tmp_path)

    manifest, records = _validate_bundle(tmp_path, hash_jobs=2)

    assert manifest["artifact_bytes"] == artifact.stat().st_size
    assert records == [
        {
            "path": "data/sample.bin",
            "size": artifact.stat().st_size,
            "sha256": digest,
            "local_path": str(artifact),
        }
    ]


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
    artifact.write_bytes(b"cppmegb")
    archive = _archive(tmp_path, artifact)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate_archive(bundle=tmp_path, archive=archive, manifest=manifest)


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
    assert receipt["archive_validation"] == {
        "status": "verified",
        "member_count": 2,
        "artifact_set_sha256": json.loads(
            (tmp_path / "manifest.json").read_text(encoding="utf-8")
        )["artifact_set_sha256"],
        "logical_manifest_sha256": hashlib.sha256(
            (tmp_path / "manifest.json").read_bytes()
        ).hexdigest(),
    }
