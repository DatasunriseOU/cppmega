from __future__ import annotations

import io
import json
import hashlib
from datetime import datetime, timezone
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from cppmega.receipt_binding import build_implementation_binding
import scripts.data.restore_megatron_bundle_from_nebius_s3 as restore
from scripts.data.restore_megatron_bundle_from_nebius_s3 import (
    _acquire_archive,
    _acquire_restore_lock,
    _extract_tar_zst,
    _prefix_manifest_sha256s,
    _require_free_space,
    _s3_request_target,
    _sigv4_headers,
    _validate_run_id,
    _validate_tar_zst_members,
    _validate_transport,
    build_arg_parser,
)


def _transport() -> dict:
    return {
        "schema": "cppmega_megatron_bundle_transport_v1",
        "bundle_id": "bundle-1",
        "logical_manifest_sha256": "d" * 64,
        "artifact_set_sha256": "b" * 64,
        "artifact_count": 1,
        "artifact_bytes": 8,
        "logical_manifest": {
            "uri": "s3://bucket/logical_manifest.json",
            "size": 100,
            "sha256": "d" * 64,
        },
        "archive": {
            "uri": "s3://bucket/bundle.tar.zst",
            "size": 10,
            "sha256": "c" * 64,
            "format": "tar.zst",
        },
    }


def test_validate_transport_accepts_bound_archive():
    _validate_transport(_transport(), expected_bundle_id="bundle-1")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong", "schema"),
        ("bundle_id", "bundle-2", "bundle mismatch"),
    ],
)
def test_validate_transport_rejects_wrong_contract(field, value, message):
    transport = _transport()
    transport[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_transport(transport, expected_bundle_id="bundle-1")


def test_validate_transport_rejects_bundle_path_traversal():
    transport = _transport()
    transport["bundle_id"] = "../bundle-1"
    with pytest.raises(ValueError, match="unsafe transport bundle ID"):
        _validate_transport(transport)


def test_validate_transport_rejects_unverified_archive_fields():
    transport = _transport()
    transport["archive"]["sha256"] = "short"
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_transport(transport, expected_bundle_id="bundle-1")

    transport = _transport()
    transport["archive"]["uri"] = "https://example.invalid/bundle.tar.zst"
    with pytest.raises(ValueError, match="s3://"):
        _validate_transport(transport, expected_bundle_id="bundle-1")

    transport = _transport()
    transport["logical_manifest"]["sha256"] = "short"
    with pytest.raises(ValueError, match="logical manifest SHA-256"):
        _validate_transport(transport, expected_bundle_id="bundle-1")

    transport = _transport()
    transport["logical_manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="hashes disagree"):
        _validate_transport(transport, expected_bundle_id="bundle-1")


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_extract_tar_zst_streams_into_destination(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("cppmega\n", encoding="utf-8")
    raw_tar = tmp_path / "bundle.tar"
    archive = tmp_path / "bundle.tar.zst"
    with tarfile.open(raw_tar, "w") as tar:
        tar.add(source, arcname="data/source.txt")
    subprocess.run(
        ["zstd", "-q", "-1", str(raw_tar), "-o", str(archive)], check=True
    )

    destination = tmp_path / "restored"
    _extract_tar_zst(archive, destination)

    assert (destination / "data/source.txt").read_text(encoding="utf-8") == "cppmega\n"


def _write_tar_zst(
    tmp_path: Path, members: list[tuple[tarfile.TarInfo, bytes]]
) -> Path:
    raw_tar = tmp_path / "bundle.tar"
    archive = tmp_path / "bundle.tar.zst"
    with tarfile.open(raw_tar, "w") as tar:
        for member, payload in members:
            if member.isfile():
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
            else:
                tar.addfile(member)
    subprocess.run(
        ["zstd", "-q", "-1", str(raw_tar), "-o", str(archive)], check=True
    )
    return archive


def _regular_member(name: str, payload: bytes = b"x") -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.type = tarfile.REGTYPE
    return member, payload


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "/abs.txt",
        "safe/../../escape.txt",
        r"safe\..\escape.txt",
        "safe//noncanonical.txt",
    ],
)
def test_extract_tar_zst_rejects_path_traversal_before_extracting(tmp_path, name):
    archive = _write_tar_zst(tmp_path, [_regular_member(name)])
    destination = tmp_path / "restored"

    with pytest.raises(ValueError, match="unsafe archive member path"):
        _extract_tar_zst(archive, destination)

    assert not destination.exists()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
@pytest.mark.parametrize(
    ("member_type", "message"),
    [
        (tarfile.SYMTYPE, "unsupported member type"),
        (tarfile.LNKTYPE, "unsupported member type"),
        (tarfile.CHRTYPE, "unsupported member type"),
        (tarfile.BLKTYPE, "unsupported member type"),
        (tarfile.FIFOTYPE, "unsupported member type"),
    ],
)
def test_extract_tar_zst_rejects_links_and_device_entries_before_extracting(
    tmp_path, member_type, message
):
    member = tarfile.TarInfo("unsafe")
    member.type = member_type
    member.linkname = "target"
    member.devmajor = 1
    member.devminor = 3
    archive = _write_tar_zst(tmp_path, [(member, b"")])
    destination = tmp_path / "restored"

    with pytest.raises(ValueError, match=message):
        _extract_tar_zst(archive, destination)

    assert not destination.exists()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_restore_archive_member_set_must_be_complete_before_extraction(tmp_path):
    archive = _write_tar_zst(tmp_path, [_regular_member("manifest.json", b"{}")])
    destination = tmp_path / "restored"

    with pytest.raises(ValueError, match="member set mismatch"):
        _extract_tar_zst(
            archive,
            destination,
            expected_member_names={"manifest.json", "data/sample.bin"},
        )

    assert not destination.exists()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_restore_archive_rejects_duplicate_members_before_extraction(tmp_path):
    archive = _write_tar_zst(
        tmp_path,
        [
            _regular_member("manifest.json", b"{}"),
            _regular_member("manifest.json", b"{}"),
        ],
    )
    destination = tmp_path / "restored"

    with pytest.raises(ValueError, match="duplicate member"):
        _extract_tar_zst(
            archive,
            destination,
            expected_member_names={"manifest.json"},
        )

    assert not destination.exists()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
def test_validate_tar_zst_members_accepts_only_expected_regular_files(tmp_path):
    archive = _write_tar_zst(
        tmp_path,
        [
            _regular_member("manifest.json", b"{}"),
            _regular_member("data/sample.bin", b"cppmega"),
        ],
    )

    seen = _validate_tar_zst_members(
        archive, {"manifest.json", "data/sample.bin"}
    )

    assert seen == ["manifest.json", "data/sample.bin"]


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required")
@pytest.mark.parametrize(
    ("expected_size", "expected_sha256", "message"),
    [
        (8, hashlib.sha256(b"cppmega").hexdigest(), "size mismatch"),
        (7, hashlib.sha256(b"changed").hexdigest(), "SHA-256 mismatch"),
    ],
)
def test_restore_validates_member_size_and_hash_before_extraction(
    tmp_path, expected_size, expected_sha256, message
):
    archive = _write_tar_zst(
        tmp_path, [_regular_member("data/sample.bin", b"cppmega")]
    )
    destination = tmp_path / "restored"

    with pytest.raises(ValueError, match=message):
        _extract_tar_zst(
            archive,
            destination,
            expected_member_names={"data/sample.bin"},
            expected_member_records={
                "data/sample.bin": (expected_size, expected_sha256)
            },
        )

    assert not destination.exists()


def test_restore_script_supports_direct_cli_execution():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/data/restore_megatron_bundle_from_nebius_s3.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Restore and verify" in result.stdout


def test_restore_cli_requires_explicit_run_identity() -> None:
    required = {
        action.dest
        for action in build_arg_parser()._actions
        if getattr(action, "required", False)
    }

    assert {"output_root", "run_id"} <= required


def test_restore_defaults_to_dependency_free_s3_client() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(["--output-root", "/data/restore", "--run-id", "run-1"])

    assert args.s3_client == "python"
    assert args.s3_region == "eu-north1"


def test_s3_request_target_uses_nebius_path_style_without_normalizing_key() -> None:
    url, canonical_uri = _s3_request_target(
        "s3://bucket-1/a//space here/%literal",
        endpoint="https://storage.eu-north1.nebius.cloud",
    )

    assert canonical_uri == "/bucket-1/a//space%20here/%25literal"
    assert url == (
        "https://storage.eu-north1.nebius.cloud"
        "/bucket-1/a//space%20here/%25literal"
    )


def test_sigv4_matches_official_aws_get_object_vector() -> None:
    headers = _sigv4_headers(
        "https://examplebucket.s3.amazonaws.com/test.txt",
        canonical_uri="/test.txt",
        region="us-east-1",
        env={
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": (
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
            ),
        },
        timestamp=datetime(2013, 5, 24, tzinfo=timezone.utc),
        extra_headers={"range": "bytes=0-9"},
    )

    assert headers["x-amz-content-sha256"] == (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )
    assert headers["Authorization"] == (
        "AWS4-HMAC-SHA256 "
        "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,"
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    )


@pytest.mark.parametrize(
    ("uri", "endpoint"),
    [
        ("https://bucket/key", "https://storage.example"),
        ("s3://bucket/key?query=1", "https://storage.example"),
        ("s3://bucket/key", "http://storage.example"),
        ("s3://bucket/key", "https://user:secret@storage.example"),
    ],
)
def test_s3_request_target_rejects_unsafe_transport(
    uri: str, endpoint: str
) -> None:
    with pytest.raises(ValueError):
        _s3_request_target(uri, endpoint=endpoint)


def test_restore_binding_derives_all_prefix_manifest_hashes() -> None:
    manifest = {
        "artifacts": [
            {"path": "data/seq_1024/train.json", "sha256": "a" * 64},
            {"path": "data/seq_2048/train.json", "sha256": "b" * 64},
        ],
        "bucket_results": [
            {"prefix": "data/seq_2048/train"},
            {"prefix": "data/seq_1024/train"},
        ],
    }

    assert _prefix_manifest_sha256s(manifest) == {
        "data/seq_1024/train.json": "a" * 64,
        "data/seq_2048/train.json": "b" * 64,
    }


def test_restore_lock_rejects_same_bundle_and_run_concurrency(tmp_path: Path) -> None:
    first = _acquire_restore_lock(tmp_path, bundle_id="bundle-1", run_id="run-1")
    try:
        with pytest.raises(RuntimeError, match="restore already active"):
            _acquire_restore_lock(tmp_path, bundle_id="bundle-1", run_id="run-1")
    finally:
        first.close()


def test_restore_lock_rejects_different_runs_for_same_bundle(tmp_path: Path) -> None:
    first = _acquire_restore_lock(tmp_path, bundle_id="bundle-1", run_id="run-1")
    try:
        with pytest.raises(RuntimeError, match="restore already active"):
            _acquire_restore_lock(tmp_path, bundle_id="bundle-1", run_id="run-2")
    finally:
        first.close()


@pytest.mark.parametrize(
    "run_id",
    ["x/../victim", "../victim", "/absolute", ".", "x" * 129, "space id"],
)
def test_restore_rejects_unsafe_run_id_before_filesystem_mutation(
    tmp_path: Path, run_id: str
) -> None:
    output_root = tmp_path / "restore-output"
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="restore run_id"):
        restore.main(
            [
                "--output-root",
                str(output_root),
                "--run-id",
                run_id,
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not output_root.exists()


def test_validate_restore_run_id_accepts_safe_identifier() -> None:
    assert _validate_run_id("cold-restore_20260714.01") == (
        "cold-restore_20260714.01"
    )


def test_fresh_restore_rejects_nonempty_output_root(tmp_path: Path) -> None:
    (tmp_path / "stale-marker").write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="fresh restore requires an empty output root"):
        restore._require_empty_output_root(tmp_path)


def test_restore_requires_space_for_archive_expansion(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.data.restore_megatron_bundle_from_nebius_s3.shutil.disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=100, used=90, free=10),
    )

    with pytest.raises(RuntimeError, match="insufficient free space"):
        _require_free_space(
            tmp_path, artifact_bytes=8, archive_bytes=3, headroom_gb=0
        )


def test_restore_rejects_legacy_manifest_before_archive_download(
    tmp_path, monkeypatch
):
    logical_manifest = {
        "schema": "cppmega_megatron_bundle_v3",
        "bundle_id": "bundle-1",
        "tokenizer_contract": "megacpp-vocab-65536",
        "vocab_size": 65536,
        "training_contract": "legacy_causal",
        "artifact_set_sha256": "b" * 64,
        "artifact_count": 1,
        "artifact_bytes": 8,
    }
    logical_bytes = json.dumps(logical_manifest).encode("utf-8")
    transport = _transport()
    transport["logical_manifest"]["size"] = len(logical_bytes)
    transport["logical_manifest"]["sha256"] = hashlib.sha256(logical_bytes).hexdigest()
    transport["logical_manifest_sha256"] = transport["logical_manifest"]["sha256"]
    transport_bytes = json.dumps(transport).encode("utf-8")

    def fake_read(uri, **_kwargs):
        if uri.endswith("transport.json"):
            return transport_bytes
        if uri.endswith("logical_manifest.json"):
            return logical_bytes
        raise AssertionError(f"unexpected remote read: {uri}")

    monkeypatch.setattr(restore, "_load_env_file", lambda _path: None)
    monkeypatch.setattr(restore, "_s3_env", lambda: {})
    monkeypatch.setattr(restore, "_aws_read", fake_read)
    monkeypatch.setattr(
        restore,
        "_acquire_archive",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("archive download must not start")
        ),
    )

    with pytest.raises(ValueError, match="training_contract"):
        restore.main(
            [
                "--output-root",
                str(tmp_path),
                "--bundle-id",
                "bundle-1",
                "--run-id",
                "preflight",
                "--megatron-commit",
                "1" * 40,
                "--s3-client",
                "aws",
            ]
        )


def test_archive_download_uses_atomic_staging_and_resumable_receipt(
    tmp_path, monkeypatch
):
    payload = b"verified archive"
    digest = hashlib.sha256(payload).hexdigest()
    archive = tmp_path / ".bundle.tar.zst"
    calls = []

    def fake_download(uri, destination, *, endpoint, env):
        calls.append((uri, destination, endpoint, env))
        assert destination.name.endswith(".download")
        destination.write_bytes(payload)

    monkeypatch.setattr(restore, "_aws_download", fake_download)
    first = _acquire_archive(
        uri="s3://bucket/bundle.tar.zst",
        archive=archive,
        endpoint="https://storage.example",
        env={"AWS_ACCESS_KEY_ID": "test"},
        expected_size=len(payload),
        expected_sha256=digest,
    )
    second = _acquire_archive(
        uri="s3://bucket/bundle.tar.zst",
        archive=archive,
        endpoint="https://storage.example",
        env={"AWS_ACCESS_KEY_ID": "test"},
        expected_size=len(payload),
        expected_sha256=digest,
    )

    assert len(calls) == 1
    assert first["status"] == "downloaded_verified"
    assert second["status"] == "reused_verified"
    assert archive.read_bytes() == payload
    assert not archive.with_name(archive.name + ".download").exists()
    receipt = json.loads(
        archive.with_name(archive.name + ".receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["sha256"] == digest
    assert receipt["uri"] == "s3://bucket/bundle.tar.zst"


def test_archive_download_never_promotes_partial_payload(tmp_path, monkeypatch):
    payload = b"complete"
    digest = hashlib.sha256(payload).hexdigest()
    archive = tmp_path / ".bundle.tar.zst"

    def fake_download(_uri, destination, *, endpoint, env):
        destination.write_bytes(b"partial")

    monkeypatch.setattr(restore, "_aws_download", fake_download)

    with pytest.raises(ValueError, match="downloaded archive size"):
        _acquire_archive(
            uri="s3://bucket/bundle.tar.zst",
            archive=archive,
            endpoint="https://storage.example",
            env={},
            expected_size=len(payload),
            expected_sha256=digest,
        )

    assert not archive.exists()
    assert not archive.with_name(archive.name + ".receipt.json").exists()


def test_archive_download_reacquires_matching_archive_with_failed_receipt(tmp_path):
    payload = b"reacquired archive"
    digest = hashlib.sha256(payload).hexdigest()
    archive = tmp_path / ".bundle.tar.zst"
    archive.write_bytes(payload)
    receipt_path = archive.with_name(archive.name + ".receipt.json")
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "cppmega_megatron_archive_download_receipt_v1",
                "uri": "s3://bucket/bundle.tar.zst",
                "size": len(payload),
                "sha256": digest,
                "status": "failed",
            }
        ),
        encoding="utf-8",
    )

    aws_bin = tmp_path / "bin"
    aws_bin.mkdir()
    marker = tmp_path / "reacquired.marker"
    aws = aws_bin / "aws"
    aws.write_text(
        "#!/bin/sh\n"
        f"printf reacquired > {marker}\n"
        "printf 'reacquired archive' > \"$4\"\n",
        encoding="utf-8",
    )
    aws.chmod(0o755)

    result = _acquire_archive(
        uri="s3://bucket/bundle.tar.zst",
        archive=archive,
        endpoint="https://storage.example",
        env={"PATH": str(aws_bin)},
        expected_size=len(payload),
        expected_sha256=digest,
    )

    assert marker.read_text(encoding="utf-8") == "reacquired"
    assert result["status"] in {
        "downloaded_verified",
        "recovered_download_verified",
    }
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] != "failed"


def test_archive_download_receipt_rejects_stale_case6_binding(tmp_path, monkeypatch):
    payload = b"verified archive"
    digest = hashlib.sha256(payload).hexdigest()
    archive = tmp_path / ".bundle.tar.zst"
    binding = {
        "schema": "cppmega_case6_receipt_binding_v2",
        "bundle_id": "bundle-1",
        "artifact_set_sha256": "a" * 64,
        "prefix_manifest_sha256s": {"data/train.json": "b" * 64},
        "checkpoint_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "command_sha256": "e" * 64,
        "run_id": "cold-restore-1",
        "implementation": build_implementation_binding(
            cppmega_commit="1" * 40,
            cppmega_tree_sha256="2" * 64,
            megatron_commit="3" * 40,
            cppmega_mlx_commit="4" * 40,
            cppmega_mlx_tree_sha256="5" * 64,
            clang_indexer_sha256="6" * 64,
            clang_indexer_dependency_closure_sha256="7" * 64,
        ),
    }

    def fake_download(_uri, destination, *, endpoint, env):
        destination.write_bytes(payload)

    monkeypatch.setattr(restore, "_aws_download", fake_download)
    _acquire_archive(
        uri="s3://bucket/bundle.tar.zst",
        archive=archive,
        endpoint="https://storage.example",
        env={},
        expected_size=len(payload),
        expected_sha256=digest,
        receipt_binding=binding,
    )
    stale = dict(binding)
    stale["run_id"] = "cold-restore-2"

    with pytest.raises(ValueError, match="receipt binding.*run_id"):
        _acquire_archive(
            uri="s3://bucket/bundle.tar.zst",
            archive=archive,
            endpoint="https://storage.example",
            env={},
            expected_size=len(payload),
            expected_sha256=digest,
            receipt_binding=stale,
        )
