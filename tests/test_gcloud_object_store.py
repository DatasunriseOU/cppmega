from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.distributed_data_prep import source_worker
from scripts.distributed_data_prep._common import ContractError
from scripts.distributed_data_prep.source_worker import GcloudObjectStore


def _metadata(uri: str, size: int) -> dict[str, object]:
    return {
        "uri": uri,
        "generation": "42",
        "size_bytes": size,
        "crc32c": "crc",
        "md5_hash": None,
    }


def test_large_upload_uses_bounded_parallel_resumable_gcloud_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "large snapshot.sqlite3"
    source.write_bytes(b"immutable snapshot")
    uri = "gs://snapshot-bucket/run/input.sqlite3"
    observed: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("CLOUDSDK_STORAGE_MAX_RETRIES", "999")
    monkeypatch.setattr(source_worker.subprocess, "run", run)
    store = GcloudObjectStore(executable="test-gcloud")
    monkeypatch.setattr(
        store, "describe", lambda target: _metadata(target, source.stat().st_size)
    )

    assert store.publish_if_absent(source, uri) == _metadata(uri, source.stat().st_size)

    assert observed["command"] == [
        "test-gcloud",
        "storage",
        "cp",
        str(source),
        uri,
        "--if-generation-match=0",
        "--content-type=application/octet-stream",
        "--quiet",
    ]
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    upload_env = kwargs["env"]
    assert isinstance(upload_env, dict)
    assert upload_env["CLOUDSDK_STORAGE_MAX_RETRIES"] == "5"
    assert upload_env["CLOUDSDK_STORAGE_RESUMABLE_THRESHOLD"] == "8Mi"
    assert upload_env["CLOUDSDK_STORAGE_UPLOAD_CHUNK_SIZE"] == "64Mi"
    assert upload_env["CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED"] == "true"
    assert upload_env["CLOUDSDK_STORAGE_PROCESS_COUNT"] == "1"
    assert upload_env["CLOUDSDK_STORAGE_THREAD_COUNT"] == "8"
    assert upload_env["CLOUDSDK_STORAGE_CHECK_HASHES"] == "always"


def test_failed_upload_accepts_only_exact_generation_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "snapshot.sqlite3"
    source.write_bytes(b"same bytes")
    uri = "gs://snapshot-bucket/run/input.sqlite3"
    metadata = _metadata(uri, source.stat().st_size)
    readback: dict[str, object] = {}

    monkeypatch.setattr(
        source_worker.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="412 precondition failed"
        ),
    )
    store = GcloudObjectStore()
    monkeypatch.setattr(store, "describe", lambda target: metadata)

    def download(
        target: str, destination: Path, *, generation: str | None = None
    ) -> dict[str, object]:
        readback.update(uri=target, generation=generation)
        destination.write_bytes(source.read_bytes())
        return metadata

    monkeypatch.setattr(store, "download", download)

    assert store.publish_if_absent(source, uri) == metadata
    assert readback == {"uri": uri, "generation": "42"}


def test_failed_upload_rejects_immutable_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "snapshot.sqlite3"
    source.write_bytes(b"expected bytes")
    uri = "gs://snapshot-bucket/run/input.sqlite3"
    metadata = _metadata(uri, source.stat().st_size)

    monkeypatch.setattr(
        source_worker.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="412 precondition failed"
        ),
    )
    store = GcloudObjectStore()
    monkeypatch.setattr(store, "describe", lambda target: metadata)

    def download(
        target: str, destination: Path, *, generation: str | None = None
    ) -> dict[str, object]:
        destination.write_bytes(b"different bytes")
        return metadata

    monkeypatch.setattr(store, "download", download)

    with pytest.raises(ContractError, match="already exists with different bytes"):
        store.publish_if_absent(source, uri)


def test_failed_upload_without_verifiable_object_preserves_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "snapshot.sqlite3"
    source.write_bytes(b"snapshot")
    uri = "gs://snapshot-bucket/run/input.sqlite3"

    monkeypatch.setattr(
        source_worker.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="bounded upload failed"
        ),
    )
    store = GcloudObjectStore()

    def describe(target: str) -> dict[str, object]:
        raise RuntimeError("object is absent")

    monkeypatch.setattr(store, "describe", describe)

    with pytest.raises(RuntimeError, match="bounded upload failed"):
        store.publish_if_absent(source, uri)
