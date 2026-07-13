from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts.data.restore_megatron_bundle_from_nebius_s3 import (
    _extract_tar_zst,
    _require_free_space,
    _validate_transport,
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


def test_restore_requires_space_for_archive_expansion(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.data.restore_megatron_bundle_from_nebius_s3.shutil.disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=100, used=90, free=10),
    )

    with pytest.raises(RuntimeError, match="insufficient free space"):
        _require_free_space(
            tmp_path, artifact_bytes=8, archive_bytes=3, headroom_gb=0
        )
