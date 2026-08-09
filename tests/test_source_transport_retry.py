from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.distributed_data_prep import source_worker
from scripts.distributed_data_prep.source_worker import (
    GcloudObjectStore,
    TransientTransportError,
    _is_transient_curl_failure,
    _is_transient_git_failure,
    _run_git_network_command,
)
from scripts.distributed_data_prep._common import ContractError


def _completed(command: list[str], returncode: int, *, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        command, returncode, stdout=stdout, stderr=stderr
    )


def test_git_transport_retries_429_and_503_then_succeeds() -> None:
    results = [
        _completed([], 1, stderr="RPC failed; HTTP 429"),
        _completed([], 1, stderr="remote returned error: 503"),
        _completed([], 0),
    ]
    calls: list[dict[str, object]] = []
    sleeps: list[float] = []

    def runner(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return results.pop(0)

    result = _run_git_network_command(
        ["git", "clone", "--mirror", "https://example.invalid/repo", "mirror"],
        operation="mirror clone",
        runner=runner,
        sleeper=sleeps.append,
        max_retries=3,
    )

    assert result.returncode == 0
    assert sleeps == [1.0, 2.0]
    assert len(calls) == 3
    assert calls[0]["env"]["GIT_TERMINAL_PROMPT"] == "0"


@pytest.mark.parametrize(
    "stderr",
    ["Authentication failed", "remote: repository not found", "HTTP 401"],
)
def test_git_auth_and_contract_failures_do_not_retry(stderr: str) -> None:
    calls: list[int] = []

    def runner(command, **kwargs):
        calls.append(1)
        return _completed(command, 1, stderr=stderr)

    with pytest.raises(RuntimeError):
        _run_git_network_command(
            ["git", "clone", "https://example.invalid/repo", "mirror"],
            operation="mirror clone",
            runner=runner,
            sleeper=lambda _delay: pytest.fail("terminal Git error was retried"),
        )
    assert len(calls) == 1


def test_transport_classifiers_reject_auth_and_accept_network_statuses() -> None:
    assert _is_transient_git_failure(
        returncode=1, stdout="", stderr="HTTP 429"
    )
    assert _is_transient_git_failure(
        returncode=1, stdout="", stderr="HTTP/2 503"
    )
    assert not _is_transient_git_failure(
        returncode=1, stdout="", stderr="HTTP 401 Unauthorized"
    )
    assert _is_transient_curl_failure(returncode=22, status=429)
    assert _is_transient_curl_failure(returncode=7, status=None)
    assert not _is_transient_curl_failure(returncode=22, status=403)


def test_gcs_metadata_retries_429_without_retrying_terminal_401(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []
    curl_attempts = 0

    def runner(command, **kwargs):
        nonlocal curl_attempts
        command = list(command)
        calls.append(command)
        if command[0] == "gcloud":
            return _completed(command, 0, stdout="token")
        curl_attempts += 1
        output = Path(command[command.index("--output") + 1])
        if curl_attempts == 1:
            output.write_text('{"message":"slow"}', encoding="utf-8")
            return _completed(command, 0, stdout="429")
        output.write_text(
            '{"name":"object","generation":"7","size":"3"}',
            encoding="utf-8",
        )
        return _completed(command, 0, stdout="200")

    store = GcloudObjectStore(
        runner=runner, sleeper=sleeps.append, max_retries=2
    )
    metadata = store.describe_if_present("gs://bucket/object")
    assert metadata is not None
    assert metadata["generation"] == "7"
    assert sleeps == [1.0]
    assert curl_attempts == 2
    metadata_command = next(command for command in calls if command[0] == "curl")
    assert "https://storage.googleapis.com/storage/v1/b/bucket/o/object" in metadata_command

    terminal_calls = 0

    def terminal_runner(command, **kwargs):
        nonlocal terminal_calls
        command = list(command)
        if command[0] == "gcloud":
            return _completed(command, 0, stdout="token")
        terminal_calls += 1
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"message":"no"}', encoding="utf-8")
        return _completed(command, 0, stdout="401")

    terminal_store = GcloudObjectStore(
        runner=terminal_runner,
        sleeper=lambda _delay: pytest.fail("HTTP 401 was retried"),
        max_retries=2,
    )
    with pytest.raises(RuntimeError):
        terminal_store.describe_if_present("gs://bucket/object")
    assert terminal_calls == 1


def test_gcs_download_streams_to_target_filesystem_without_second_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"large immutable snapshot"
    destination = tmp_path / "local-ssd" / "content-store-index.sqlite3"
    download_outputs: list[Path] = []

    def runner(command, **kwargs):
        command = list(command)
        if command[0] == "gcloud":
            return _completed(command, 0, stdout="token")
        output = Path(command[command.index("--output") + 1])
        if "/download/storage/" in command[-1]:
            download_outputs.append(output)
            output.write_bytes(payload)
        else:
            output.write_text(
                '{"name":"snapshot","generation":"7","size":"'
                + str(len(payload))
                + '"}',
                encoding="utf-8",
            )
        return _completed(command, 0, stdout="200")

    monkeypatch.setattr(
        source_worker.shutil,
        "copyfile",
        lambda *_args, **_kwargs: pytest.fail("download performed a second full copy"),
    )
    store = GcloudObjectStore(runner=runner)

    metadata = store.download(
        "gs://bucket/snapshot", destination, generation="7"
    )

    assert metadata["size_bytes"] == len(payload)
    assert destination.read_bytes() == payload
    assert len(download_outputs) == 1
    assert download_outputs[0].parent == destination.parent
    assert download_outputs[0].name.startswith(f".{destination.name}.")
    assert not download_outputs[0].exists()


def test_gcs_download_failure_preserves_destination_and_omits_binary_body(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "local-ssd" / "snapshot"
    destination.parent.mkdir()
    destination.write_bytes(b"existing snapshot")
    staged: list[Path] = []

    def runner(command, **kwargs):
        command = list(command)
        if command[0] == "gcloud":
            return _completed(command, 0, stdout="token")
        output = Path(command[command.index("--output") + 1])
        staged.append(output)
        output.write_bytes(b"binary-response-must-not-reach-the-diagnostic")
        return _completed(
            command,
            23,
            stdout="200",
            stderr="curl: (23) failure writing output",
        )

    store = GcloudObjectStore(runner=runner, max_retries=0)
    with pytest.raises(RuntimeError) as raised:
        store.download("gs://bucket/snapshot", destination, generation="7")

    assert "curl: (23) failure writing output" in str(raised.value)
    assert "binary-response-must-not-reach" not in str(raised.value)
    assert destination.read_bytes() == b"existing snapshot"
    assert len(staged) == 1
    assert not staged[0].exists()


def test_gcs_download_failure_preserves_small_json_error_body(tmp_path: Path) -> None:
    destination = tmp_path / "local-ssd" / "snapshot"

    def runner(command, **kwargs):
        command = list(command)
        if command[0] == "gcloud":
            return _completed(command, 0, stdout="token")
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"error":{"message":"object denied"}}', encoding="utf-8")
        return _completed(command, 0, stdout="403")

    store = GcloudObjectStore(runner=runner, max_retries=0)
    with pytest.raises(RuntimeError, match="object denied"):
        store.download("gs://bucket/snapshot", destination, generation="7")

    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


class _PublishFixtureStore(GcloudObjectStore):
    def __init__(self, source_bytes: bytes, statuses: list[int]):
        super().__init__(runner=lambda *_args, **_kwargs: _completed([], 0, stdout="token"))
        self.source_bytes = source_bytes
        self.statuses = list(statuses)
        self.existing = False
        self.existing_bytes = source_bytes
        self.post_calls = 0
        self.download_destinations: list[Path] = []

    def _access_token(self) -> str:
        return "token"

    def _curl_once(self, **kwargs):
        response = kwargs["response"]
        status = self.statuses.pop(0)
        self.post_calls += 1
        response.write_text('{"generation":"7"}', encoding="utf-8")
        return _completed([], 0, stdout=str(status)), status

    def describe_if_present(self, uri: str, *, generation: str | None = None):
        if not self.existing:
            return None
        return {"uri": uri, "generation": "7", "size_bytes": len(self.existing_bytes)}

    def describe(self, uri: str, *, generation: str | None = None):
        return {"uri": uri, "generation": "7", "size_bytes": len(self.source_bytes)}

    def download(self, uri: str, destination: Path, *, generation: str | None = None):
        self.download_destinations.append(destination)
        destination.write_bytes(self.existing_bytes)
        return {"uri": uri, "generation": "7", "size_bytes": len(self.existing_bytes)}


def test_immutable_upload_reconciles_lost_success_and_rejects_different_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "payload"
    source.write_bytes(b"same bytes")
    store = _PublishFixtureStore(source.read_bytes(), [503])
    store.existing = True
    metadata = store.publish_if_absent(source, "gs://bucket/object")
    assert metadata["generation"] == "7"
    assert store.post_calls == 1
    assert len(store.download_destinations) == 1
    assert store.download_destinations[0].parent.parent == source.parent

    different = _PublishFixtureStore(source.read_bytes(), [412])
    different.existing = True
    different.existing_bytes = b"different"
    with pytest.raises(ContractError, match="different bytes"):
        different.publish_if_absent(source, "gs://bucket/object")

    absent = _PublishFixtureStore(source.read_bytes(), [412])
    with pytest.raises(ContractError, match="existing object was not readable"):
        absent.publish_if_absent(source, "gs://bucket/object")


def test_exhausted_git_transport_is_resumable_exit_class() -> None:
    def runner(command, **kwargs):
        return _completed(command, 1, stderr="HTTP 503")

    with pytest.raises(TransientTransportError):
        _run_git_network_command(
            ["git", "fetch", "origin"],
            operation="mirror fetch",
            runner=runner,
            sleeper=lambda _delay: None,
            max_retries=1,
        )
