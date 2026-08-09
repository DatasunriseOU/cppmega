from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Mapping

import pytest

from scripts.distributed_data_prep._common import ContractError
from scripts.distributed_data_prep.cloud_lane_heartbeat import (
    WORKER_HEARTBEAT_PUBLICATION_SCHEMA,
    build_worker_heartbeat,
    publish_worker_heartbeat,
    run_worker_heartbeat_loop,
    validate_worker_heartbeat,
    validate_worker_heartbeat_publication,
    validate_worker_heartbeat_uri,
    verify_published_worker_heartbeat,
    worker_heartbeat_uri,
)
from scripts.distributed_data_prep.source_worker import LocalObjectStore


CONTROL_PREFIX = "gs://fixture-bucket/runs/ci-case5-smoke-004"
MANIFEST_SHA256 = "1" * 64
MANIFEST_FILE_SHA256 = "2" * 64
CODE_REVISION = "3" * 40
RUNNER_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "infra/gcp_corpus_pool/pilot/cloud-lane-worker-runner.sh.tmpl"
)


def _heartbeat(sequence: int = 0) -> dict[str, object]:
    return build_worker_heartbeat(
        sequence=sequence,
        manifest_sha256=MANIFEST_SHA256,
        manifest_file_sha256=MANIFEST_FILE_SHA256,
        code_revision=CODE_REVISION,
        physical_worker="physical-0000",
        physical_worker_count=1,
        emitted_at=f"2026-08-09T10:00:0{sequence}Z",
    )


def _object_files(root: Path) -> list[Path]:
    prefix = (
        root
        / "fixture-bucket/runs/ci-case5-smoke-004/control/cloud-lane-heartbeats"
        / MANIFEST_SHA256
        / "physical-0000"
    )
    return sorted(prefix.glob("*.heartbeat.json"))


def _runner_shell_function(name: str) -> str:
    lines = RUNNER_TEMPLATE.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{name}() {{")
    for end in range(start + 1, len(lines)):
        if lines[end] == "}":
            return "\n".join(lines[start : end + 1])
    raise AssertionError(f"unterminated runner function: {name}")


def test_heartbeat_schema_digest_and_uri_are_strictly_bound() -> None:
    heartbeat = _heartbeat()
    uri = worker_heartbeat_uri(CONTROL_PREFIX, heartbeat)

    assert uri.endswith(
        f"/physical-0000/000000-{heartbeat['receipt_sha256']}.heartbeat.json"
    )
    assert validate_worker_heartbeat_uri(
        uri, control_prefix=CONTROL_PREFIX, value=heartbeat
    ) == uri

    tampered = dict(heartbeat, training_ready=True)
    with pytest.raises(ContractError, match="training_ready=false"):
        validate_worker_heartbeat(tampered)
    negative_sequence = dict(heartbeat, sequence=-1)
    negative_sequence["receipt_sha256"] = heartbeat["receipt_sha256"]
    with pytest.raises(ContractError, match=r"integer >= 0"):
        validate_worker_heartbeat(negative_sequence)
    with pytest.raises(ContractError, match="URI binding drifted"):
        validate_worker_heartbeat_uri(
            uri.replace("000000-", "000001-"),
            control_prefix=CONTROL_PREFIX,
            value=heartbeat,
        )


def test_heartbeat_publication_is_create_only_and_exact_generation_verified(
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    heartbeat = _heartbeat()
    kwargs = {
        "control_prefix": CONTROL_PREFIX,
        "receipt_root": tmp_path / "receipts",
        "scratch_root": tmp_path / "scratch",
        "object_store": store,
    }

    first = publish_worker_heartbeat(heartbeat, **kwargs)
    second = publish_worker_heartbeat(heartbeat, **kwargs)
    assert first == second
    assert first["schema"] == WORKER_HEARTBEAT_PUBLICATION_SCHEMA
    assert first["generation"] == "1"
    assert first["training_ready"] is False
    assert validate_worker_heartbeat_publication(
        first, heartbeat=heartbeat, control_prefix=CONTROL_PREFIX
    ) == first
    assert len(_object_files(tmp_path / "objects")) == 1
    assert verify_published_worker_heartbeat(
        uri=str(first["uri"]),
        generation=str(first["generation"]),
        control_prefix=CONTROL_PREFIX,
        object_store=store,
        scratch_root=tmp_path / "consumer-readback",
    ) == heartbeat

    _object_files(tmp_path / "objects")[0].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="immutable object collision"):
        publish_worker_heartbeat(heartbeat, **kwargs)


def test_heartbeat_loop_publishes_immediately_periodically_and_stops_normally(
    tmp_path: Path,
) -> None:
    stop_event = threading.Event()
    publications: list[Mapping[str, object]] = []

    def published(value: Mapping[str, object]) -> None:
        publications.append(value)
        if len(publications) == 3:
            stop_event.set()

    count = run_worker_heartbeat_loop(
        control_prefix=CONTROL_PREFIX,
        receipt_root=tmp_path / "receipts",
        scratch_root=tmp_path / "scratch",
        manifest_sha256=MANIFEST_SHA256,
        manifest_file_sha256=MANIFEST_FILE_SHA256,
        code_revision=CODE_REVISION,
        physical_worker="physical-0000",
        physical_worker_count=1,
        interval_seconds=0.01,
        object_store=LocalObjectStore(tmp_path / "objects"),
        stop_event=stop_event,
        on_publication=published,
    )

    assert count == 3
    assert len(publications) == 3
    assert len(_object_files(tmp_path / "objects")) == 3
    assert [
        json.loads(path.read_text(encoding="utf-8"))["sequence"]
        for path in _object_files(tmp_path / "objects")
    ] == [0, 1, 2]


def test_heartbeat_loop_retries_same_immutable_sequence_after_publication_failure(
    tmp_path: Path,
) -> None:
    class FailFirstStore(LocalObjectStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.attempted_receipts: list[str] = []

        def publish_if_absent(
            self, source: Path, uri: str
        ) -> Mapping[str, object]:
            self.attempted_receipts.append(
                str(json.loads(source.read_text(encoding="utf-8"))["receipt_sha256"])
            )
            if len(self.attempted_receipts) == 1:
                raise RuntimeError("bounded fixture transport failure")
            return super().publish_if_absent(source, uri)

    store = FailFirstStore(tmp_path / "objects")
    stop_event = threading.Event()

    def published(_value: Mapping[str, object]) -> None:
        stop_event.set()

    count = run_worker_heartbeat_loop(
        control_prefix=CONTROL_PREFIX,
        receipt_root=tmp_path / "receipts",
        scratch_root=tmp_path / "scratch",
        manifest_sha256=MANIFEST_SHA256,
        manifest_file_sha256=MANIFEST_FILE_SHA256,
        code_revision=CODE_REVISION,
        physical_worker="physical-0000",
        physical_worker_count=1,
        interval_seconds=0.01,
        object_store=store,
        stop_event=stop_event,
        on_publication=published,
    )

    assert count == 1
    assert store.attempted_receipts[0] == store.attempted_receipts[1]
    objects = _object_files(tmp_path / "objects")
    assert len(objects) == 1
    assert json.loads(objects[0].read_text(encoding="utf-8"))["sequence"] == 0


def test_heartbeat_service_sigterm_stops_without_late_publication(
    tmp_path: Path,
) -> None:
    harness = """
from pathlib import Path
import sys
from scripts.distributed_data_prep.cloud_lane_heartbeat import run_worker_heartbeat_service
from scripts.distributed_data_prep.source_worker import LocalObjectStore

root = Path(sys.argv[1])
raise SystemExit(run_worker_heartbeat_service(
    control_prefix=sys.argv[2],
    receipt_root=root / 'receipts',
    scratch_root=root / 'scratch',
    manifest_sha256=sys.argv[3],
    manifest_file_sha256=sys.argv[4],
    code_revision=sys.argv[5],
    physical_worker='physical-0000',
    physical_worker_count=1,
    interval_seconds=0.05,
    object_store=LocalObjectStore(root / 'objects'),
))
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            harness,
            str(tmp_path),
            CONTROL_PREFIX,
            MANIFEST_SHA256,
            MANIFEST_FILE_SHA256,
            CODE_REVISION,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while len(_object_files(tmp_path / "objects")) < 2:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"heartbeat exited early: {stdout=} {stderr=}")
        if time.monotonic() >= deadline:
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(f"heartbeat did not start: {stdout=} {stderr=}")
        time.sleep(0.01)

    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, (stdout, stderr)
    terminal_count = len(_object_files(tmp_path / "objects"))
    time.sleep(0.12)
    assert len(_object_files(tmp_path / "objects")) == terminal_count


@pytest.mark.parametrize(
    ("mode", "signal_number", "expected_returncode"),
    (
        ("normal", None, 0),
        ("runner-sigint", signal.SIGINT, 130),
        ("runner-sigterm", signal.SIGTERM, 143),
    ),
)
def test_runner_heartbeat_cleanup_is_exercised_on_normal_and_signal_exit(
    tmp_path: Path,
    mode: str,
    signal_number: signal.Signals | None,
    expected_returncode: int,
) -> None:
    service = tmp_path / "heartbeat-service.py"
    service.write_text(
        """
from pathlib import Path
import sys
from scripts.distributed_data_prep.cloud_lane_heartbeat import run_worker_heartbeat_service
from scripts.distributed_data_prep.source_worker import LocalObjectStore

root = Path(sys.argv[1])
raise SystemExit(run_worker_heartbeat_service(
    control_prefix=sys.argv[2],
    receipt_root=root / 'receipts',
    scratch_root=root / 'scratch',
    manifest_sha256=sys.argv[3],
    manifest_file_sha256=sys.argv[4],
    code_revision=sys.argv[5],
    physical_worker='physical-0000',
    physical_worker_count=1,
    interval_seconds=0.05,
    object_store=LocalObjectStore(root / 'objects'),
))
""",
        encoding="utf-8",
    )
    harness = tmp_path / "runner-heartbeat-harness.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        + _runner_shell_function("stop_worker_heartbeat")
        + "\n"
        + """
heartbeat_pid=""
"$1" "$2" "$3" "$4" "$5" "$6" "$7" &
heartbeat_pid=$!
printf '%s\n' "$heartbeat_pid" >"$8"
trap stop_worker_heartbeat EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
while [[ ! -d "$9" ]] || (( $(find "$9" -name '*.heartbeat.json' | wc -l) < 2 )); do
  kill -0 "$heartbeat_pid"
  sleep 0.01
done
touch "${10}"
if [[ "${11}" == normal ]]; then
  stop_worker_heartbeat
  trap - EXIT INT TERM
  exit 0
fi
while true; do sleep 0.05; done
""",
        encoding="utf-8",
    )
    harness.chmod(0o755)
    pidfile = tmp_path / "heartbeat.pid"
    ready = tmp_path / "runner.ready"
    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repo_root), environment.get("PYTHONPATH", "")))
    )
    process = subprocess.Popen(
        [
            "bash",
            str(harness),
            sys.executable,
            str(service),
            str(tmp_path),
            CONTROL_PREFIX,
            MANIFEST_SHA256,
            MANIFEST_FILE_SHA256,
            CODE_REVISION,
            str(pidfile),
            str(tmp_path / "objects"),
            str(ready),
            mode,
        ],
        cwd=repo_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not ready.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"runner harness exited early: {stdout=} {stderr=}")
        if time.monotonic() >= deadline:
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(f"runner heartbeat did not start: {stdout=} {stderr=}")
        time.sleep(0.01)

    if signal_number is not None:
        process.send_signal(signal_number)
    stdout, stderr = process.communicate(timeout=7)
    assert process.returncode == expected_returncode, (stdout, stderr)
    heartbeat_pid = int(pidfile.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(heartbeat_pid, 0)
    terminal_count = len(_object_files(tmp_path / "objects"))
    time.sleep(0.12)
    assert len(_object_files(tmp_path / "objects")) == terminal_count
