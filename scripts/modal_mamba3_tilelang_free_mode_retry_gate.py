"""Read-only R2/R4 full parity gate for TileLang a760fe58 on old51 Mamba."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
from typing import Any

IS_SANDBOX_WORKER = os.environ.get("CPPMEGA_SANDBOX_WORKER") == "1"
if not IS_SANDBOX_WORKER:
    import modal

SCRIPT_PATH = pathlib.Path(__file__).resolve()
LOCAL_CPPMEGA_ROOT = pathlib.Path("/Volumes/external/sources/cppmega")
LOCAL_CANDIDATE_ROOT = pathlib.Path(
    "/Volumes/external/cppmega_data/tilelang_candidate_wheels/"
    "a760fe587995def0f3108ee204be453d87467c5d/linux-cuda13.2-cp313"
)
OLD_BASE_IMAGE_REF = (
    "ghcr.io/datasunriseou/cppmega@"
    "sha256:85a09018ab4689c09025c8eb2e732242c80392167e4d5d59418da49184af970a"
)
OLD51_D66_IMAGE_ID = "im-p7seeW5FdaipoteRTDe1Lo"
TILELANG_COMMIT = "a760fe587995def0f3108ee204be453d87467c5d"
TVM_COMMIT = "84af17279edb5edad29749bd6b0eea2ed9393105"
TVM_FFI_COMMIT = "e4353339293459e3e8a393afc1b6a6a869e75b13"
TILELANG_WHEEL_SHA256 = (
    "3bbda181f28820e8b9bc7c4a894238ae64a945d09a16de01c67576a867ad7fcf"
)
TVM_FFI_WHEEL_SHA256 = (
    "8233d526de8dd9a8c7cdd88e8e6085a04b577d24a7461791beca9451f3f912f3"
)
BUILD_MANIFEST_SHA256 = (
    "60a7666f93a390d803e0e99955430ada7f63a35d2ed46bf093e9a3286983c0eb"
)
MAMBA_SOURCE_SHA256 = "51dab809a47bd33a9b610725599dff956187edd70cae1518618f1ef31115d320"
MAMBA_BACKUP_SHA256 = "980dadcec29cdd318c51c1660697d54b5a7d3311d2b681b4a68b31e7d21e64b9"
MEGATRON_COMMIT = "ba7b5ebce12af60627a80985792a1449ce45f46c"
MAMBA_SOURCE = (
    "/usr/local/lib/python3.13/dist-packages/"
    "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"
)
MAMBA_BACKUP = MAMBA_SOURCE + ".cppmega_stage2_force_nontma.bak"
REMOTE_TILELANG_WHEEL = "/tmp/tilelang-0.1.9-cp38-abi3-linux_x86_64.whl"
REMOTE_TVM_FFI_WHEEL = "/tmp/apache_tvm_ffi-0.1.13.post5-cp313-cp313-linux_x86_64.whl"
REMOTE_RUNNER = "/opt/cppmega/scripts/modal_mamba3_tilelang_free_mode_retry_gate.py"
TEST_NODES = (
    "tests/test_cppmega_mamba3_tp_mixer.py::test_tp2_sp_on_parity_vs_tp1",
    (
        "tests/test_cppmega_mamba3_tp_mixer.py::"
        "test_tp2_sp_off_replicated_parameter_gradient_parity_vs_tp1"
    ),
    (
        "tests/test_cppmega_mamba3_tp_mixer.py::"
        "test_cp2_actual_mamba3_forward_backward_parity_vs_cp1"
    ),
)
GPU_SPEC = "H200:2"

NUMERICAL_PHASE = os.environ.get("CPPMEGA_NUMERICAL_PHASE")
try:
    MIMO_RANK, CHUNK_SIZE = {
        "r2": (2, 32),
        "r4": (4, 16),
    }[NUMERICAL_PHASE]
except KeyError:
    raise RuntimeError("CPPMEGA_NUMERICAL_PHASE must be exactly 'r2' or 'r4'") from None

RESULT_STEM = f"mamba3-a760-old51-{NUMERICAL_PHASE}-full-parity-a3"
APP_NAME = f"cppmega-a760-old51-{NUMERICAL_PHASE}-full-parity-a3"

SOURCE_BINDINGS = (
    (
        "tests/test_cppmega_mamba3_tp_mixer.py",
        "/opt/cppmega/tests/test_cppmega_mamba3_tp_mixer.py",
    ),
    (
        "cppmega/features/mamba3/__init__.py",
        "/opt/cppmega/cppmega/features/mamba3/__init__.py",
    ),
    (
        "cppmega/features/mamba3/config.py",
        "/opt/cppmega/cppmega/features/mamba3/config.py",
    ),
    (
        "cppmega/megatron/cppmega_mamba3_tp_mixer.py",
        "/opt/cppmega/cppmega/megatron/cppmega_mamba3_tp_mixer.py",
    ),
    (
        "cppmega/megatron/document_isolation.py",
        "/opt/cppmega/cppmega/megatron/document_isolation.py",
    ),
    (
        "cppmega/megatron/mamba_local_spec.py",
        "/opt/cppmega/cppmega/megatron/mamba_local_spec.py",
    ),
    (
        "cppmega/megatron/tilelang_mimo_autograd.py",
        "/opt/cppmega/cppmega/megatron/tilelang_mimo_autograd.py",
    ),
)
SOURCE_BINDING_SHA256 = {
    "/opt/cppmega/tests/test_cppmega_mamba3_tp_mixer.py": (
        "1469705db22bfd33060c0dd54c7f11360e3e9891ec6e9fc0ae65e038b3b38f42"
    ),
    "/opt/cppmega/cppmega/features/mamba3/__init__.py": (
        "5c07007032c0cbdf3ba8343a5160954640bd2c02230f98b21e5a92e67b36c530"
    ),
    "/opt/cppmega/cppmega/features/mamba3/config.py": (
        "bdc9226cd065f495435418d61b296fc53d883f3bb23e2407a6aeb68369ba6d3a"
    ),
    "/opt/cppmega/cppmega/megatron/cppmega_mamba3_tp_mixer.py": (
        "2e88244514e85615f63c5ed6996ee4731e6691d99ef7af13f1235ea5a98137dd"
    ),
    "/opt/cppmega/cppmega/megatron/document_isolation.py": (
        "d5bbf8c4c718c984281ad7dcb89e813c4285ca9bc0c10d180dadbd0227d24c02"
    ),
    "/opt/cppmega/cppmega/megatron/mamba_local_spec.py": (
        "e6339d4b0c127f2e7e4c3d8217054f1a1d07d7caad7c02076962290063501fad"
    ),
    "/opt/cppmega/cppmega/megatron/tilelang_mimo_autograd.py": (
        "d2e4589b5e6975ed87f4da2e600b1055bddff32359b9f5b88ed869c1e62caaaa"
    ),
}

if not IS_SANDBOX_WORKER and modal.is_local():
    manifest_path = LOCAL_CANDIDATE_ROOT / "BUILD_MANIFEST.json"
    observed_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if observed_manifest_sha256 != BUILD_MANIFEST_SHA256:
        raise RuntimeError("local candidate build manifest drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_source = {
        key: manifest.get(key)
        for key in (
            "status",
            "tilelang_commit",
            "tvm_commit",
            "tvm_ffi_commit",
        )
    }
    expected_source = {
        "status": "success",
        "tilelang_commit": TILELANG_COMMIT,
        "tvm_commit": TVM_COMMIT,
        "tvm_ffi_commit": TVM_FFI_COMMIT,
    }
    if observed_source != expected_source:
        raise RuntimeError(
            f"local candidate source provenance drifted: {observed_source!r}"
        )
    observed_wheels = {
        "tilelang": hashlib.sha256(
            (
                LOCAL_CANDIDATE_ROOT / pathlib.Path(REMOTE_TILELANG_WHEEL).name
            ).read_bytes()
        ).hexdigest(),
        "tvm_ffi": hashlib.sha256(
            (
                LOCAL_CANDIDATE_ROOT / pathlib.Path(REMOTE_TVM_FFI_WHEEL).name
            ).read_bytes()
        ).hexdigest(),
    }
    expected_wheels = {
        "tilelang": TILELANG_WHEEL_SHA256,
        "tvm_ffi": TVM_FFI_WHEEL_SHA256,
    }
    if observed_wheels != expected_wheels:
        raise RuntimeError(
            f"local candidate wheel payloads drifted: {observed_wheels!r}"
        )
    observed_bindings = {
        remote: hashlib.sha256((LOCAL_CPPMEGA_ROOT / relative).read_bytes()).hexdigest()
        for relative, remote in SOURCE_BINDINGS
    }
    if observed_bindings != SOURCE_BINDING_SHA256:
        raise RuntimeError(
            "current local source bindings drifted from the exact runner contract: "
            f"{observed_bindings!r}"
        )

SCRIPT_SHA256 = hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()


def _image() -> modal.Image:
    image: Any = modal.Image.from_id(OLD51_D66_IMAGE_ID).env(
        {
            "CPPMEGA_MAMBA3_TEST_CHUNK_SIZE": str(CHUNK_SIZE),
            "CPPMEGA_MAMBA3_TEST_MIMO_RANK": str(MIMO_RANK),
            "CPPMEGA_MEGATRON_COMMIT": MEGATRON_COMMIT,
            "CPPMEGA_NUMERICAL_PHASE": NUMERICAL_PHASE,
            "CUDA_LAUNCH_BLOCKING": "1",
            "MEGATRON_LM_REPO": "/opt/megatron-lm",
            "NCCL_DEBUG": "INFO",
            "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
            "TORCH_NCCL_DESYNC_DEBUG": "1",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_ENABLE_TIMING": "1",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
            "TORCH_SHOW_CPP_STACKTRACES": "1",
            "WANDB_MODE": "disabled",
        }
    )
    for relative, remote in SOURCE_BINDINGS:
        image = image.add_local_file(
            str(LOCAL_CPPMEGA_ROOT / relative),
            remote_path=remote,
            copy=True,
        )
    image = image.add_local_file(
        str(LOCAL_CANDIDATE_ROOT / pathlib.Path(REMOTE_TILELANG_WHEEL).name),
        remote_path=REMOTE_TILELANG_WHEEL,
        copy=True,
    ).add_local_file(
        str(LOCAL_CANDIDATE_ROOT / pathlib.Path(REMOTE_TVM_FFI_WHEEL).name),
        remote_path=REMOTE_TVM_FFI_WHEEL,
        copy=True,
    )
    image = image.add_local_file(
        str(SCRIPT_PATH),
        remote_path=REMOTE_RUNNER,
        copy=True,
    )
    source_checks = " ".join(
        f"echo '{digest}  {remote}' | sha256sum -c -;"
        for remote, digest in sorted(SOURCE_BINDING_SHA256.items())
    )
    return image.run_commands(
        "set -eux; "
        f"{source_checks} "
        f"echo '{MAMBA_SOURCE_SHA256}  {MAMBA_SOURCE}' | sha256sum -c -; "
        f"echo '{MAMBA_BACKUP_SHA256}  {MAMBA_BACKUP}' | sha256sum -c -; "
        f"echo '{TILELANG_WHEEL_SHA256}  {REMOTE_TILELANG_WHEEL}' "
        "| sha256sum -c -; "
        f"echo '{TVM_FFI_WHEEL_SHA256}  {REMOTE_TVM_FFI_WHEEL}' "
        "| sha256sum -c -; "
        f"echo '{SCRIPT_SHA256}  {REMOTE_RUNNER}' | sha256sum -c -; "
        "command -v timeout; "
        f"python -m pip install --force-reinstall --no-deps "
        f"'{REMOTE_TVM_FFI_WHEEL}' '{REMOTE_TILELANG_WHEEL}'; "
        f"test \"$(git -C /opt/megatron-lm rev-parse HEAD)\" = '{MEGATRON_COMMIT}'"
    )


if not IS_SANDBOX_WORKER:
    app = modal.App(APP_NAME)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_paths(root: pathlib.Path, run_id: str) -> dict[str, pathlib.Path]:
    prefix = root / f"{RESULT_STEM}-{run_id}"
    return {
        "receipt": prefix.with_suffix(".json"),
        "log": prefix.with_suffix(".log"),
        "junit": prefix.with_name(f"{prefix.name}-junit.xml"),
    }


def run_gate() -> dict[str, Any]:
    import importlib.metadata
    import subprocess
    import sys
    import time
    import traceback
    import uuid
    import xml.etree.ElementTree as ET

    sandbox_id = os.environ["CPPMEGA_MODAL_SANDBOX_ID"]
    run_id = os.environ["CPPMEGA_GATE_RUN_ID"]
    paths = _artifact_paths(pathlib.Path("/tmp"), run_id)
    receipt_path = paths["receipt"]
    log_path = paths["log"]
    junit_path = paths["junit"]
    temporary_junit = junit_path
    temporary_log = log_path

    def write_receipt(payload: dict[str, Any]) -> None:
        temporary = receipt_path.with_name(
            f".{receipt_path.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(receipt_path)

    def runtime_hashes() -> dict[str, str]:
        return {
            **{path: _sha256(pathlib.Path(path)) for path in SOURCE_BINDING_SHA256},
            MAMBA_SOURCE: _sha256(pathlib.Path(MAMBA_SOURCE)),
            MAMBA_BACKUP: _sha256(pathlib.Path(MAMBA_BACKUP)),
        }

    def junit_counts() -> dict[str, Any]:
        empty = {
            "present": False,
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        }
        if not temporary_junit.is_file():
            return empty
        try:
            root = ET.parse(temporary_junit).getroot()
        except (ET.ParseError, OSError) as exc:
            return {**empty, "present": True, "parse_error": repr(exc)}
        cases = root.findall(".//testcase")
        return {
            "present": True,
            "tests": len(cases),
            "failures": sum(
                any(child.tag.rsplit("}", 1)[-1] == "failure" for child in case)
                for case in cases
            ),
            "errors": sum(
                any(child.tag.rsplit("}", 1)[-1] == "error" for child in case)
                for case in cases
            ),
            "skipped": sum(
                any(child.tag.rsplit("}", 1)[-1] == "skipped" for child in case)
                for case in cases
            ),
        }

    started = time.time()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "gate": (f"H200_{NUMERICAL_PHASE.upper()}_A760_OLD51_FULL_NUMERICAL_PARITY"),
        "phase": NUMERICAL_PHASE,
        "status": "running",
        "started_unix": started,
        "runner_sha256": SCRIPT_SHA256,
        "selected_tests": list(TEST_NODES),
        "expected_test_count": len(TEST_NODES),
        "factorization": {
            "mimo_rank": MIMO_RANK,
            "chunk_size": CHUNK_SIZE,
        },
        "candidate": {
            "old_base_image_ref": OLD_BASE_IMAGE_REF,
            "old51_d66_image_id": OLD51_D66_IMAGE_ID,
            "tilelang_commit": TILELANG_COMMIT,
            "tvm_commit": TVM_COMMIT,
            "tvm_ffi_commit": TVM_FFI_COMMIT,
            "build_manifest_sha256": BUILD_MANIFEST_SHA256,
            "tilelang_wheel_sha256": TILELANG_WHEEL_SHA256,
            "tvm_ffi_wheel_sha256": TVM_FFI_WHEEL_SHA256,
            "mamba_source_sha256": MAMBA_SOURCE_SHA256,
            "mamba_backup_sha256": MAMBA_BACKUP_SHA256,
            "megatron_commit": MEGATRON_COMMIT,
            "bound_source_sha256": SOURCE_BINDING_SHA256,
        },
        "modal": {
            "sandbox_id": sandbox_id,
            "run_id": run_id,
            "gpu_spec": GPU_SPEC,
            "app_name": APP_NAME,
        },
        "artifacts": {
            "receipt": str(receipt_path),
            "log": str(log_path),
            "junit": str(junit_path),
        },
    }
    write_receipt(receipt)
    try:
        mutation_env = {
            name: os.environ.get(name)
            for name in (
                "CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA",
                "MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION",
                "CPPMEGA_MAMBA3_GQA_BWD",
                "MAMBA3_GQA_BWD_ALLOW_FILE_MUTATION",
            )
        }
        if any(mutation_env.values()):
            raise RuntimeError(
                f"runtime mutation variables must be unset: {mutation_env!r}"
            )
        receipt["runtime_mutation_environment"] = mutation_env

        before = runtime_hashes()
        expected_hashes = {
            **SOURCE_BINDING_SHA256,
            MAMBA_SOURCE: MAMBA_SOURCE_SHA256,
            MAMBA_BACKUP: MAMBA_BACKUP_SHA256,
        }
        if before != expected_hashes:
            raise RuntimeError(f"runtime source binding mismatch: observed={before!r}")
        receipt["source_sha256_before_test"] = before

        receipt["wheel_payload_sha256"] = {
            REMOTE_TILELANG_WHEEL: _sha256(pathlib.Path(REMOTE_TILELANG_WHEEL)),
            REMOTE_TVM_FFI_WHEEL: _sha256(pathlib.Path(REMOTE_TVM_FFI_WHEEL)),
        }
        if receipt["wheel_payload_sha256"] != {
            REMOTE_TILELANG_WHEEL: TILELANG_WHEEL_SHA256,
            REMOTE_TVM_FFI_WHEEL: TVM_FFI_WHEEL_SHA256,
        }:
            raise RuntimeError("candidate wheel payload hashes drifted")

        megatron_head = subprocess.run(
            ["git", "-C", "/opt/megatron-lm", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        if megatron_head != MEGATRON_COMMIT:
            raise RuntimeError(f"Megatron revision drifted: observed={megatron_head!r}")
        receipt["megatron_head"] = megatron_head
        receipt["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("tilelang", "apache-tvm-ffi", "mamba-ssm")
        }

        import torch

        devices = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if len(devices) != 2 or any("H200" not in name.upper() for name in devices):
            raise RuntimeError(f"expected exactly two H200s, observed={devices!r}")
        receipt["devices"] = devices
        receipt["capabilities"] = [
            list(torch.cuda.get_device_capability(index)) for index in range(2)
        ]

        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            "3000s",
            sys.executable,
            "-m",
            "pytest",
            *TEST_NODES,
            "-vv",
            "-s",
            "--tb=long",
            f"--junitxml={temporary_junit}",
        ]
        receipt["command"] = command
        print(
            "TILELANG_A760_OLD51_FULL_PARITY_START="
            + json.dumps(
                {
                    "phase": NUMERICAL_PHASE,
                    "image": OLD51_D66_IMAGE_ID,
                    "mamba_source_sha256": MAMBA_SOURCE_SHA256,
                    "selected_tests": list(TEST_NODES),
                    "factorization": receipt["factorization"],
                    "devices": devices,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        with temporary_log.open("w", encoding="utf-8") as output_stream:
            process = subprocess.Popen(
                command,
                cwd="/opt/cppmega",
                env=os.environ.copy(),
                stdout=output_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            heartbeat_bucket = -1
            while process.poll() is None:
                time.sleep(5)
                elapsed = time.time() - started
                current_bucket = int(elapsed // 30)
                if current_bucket != heartbeat_bucket:
                    heartbeat_bucket = current_bucket
                    print(
                        "TILELANG_A760_OLD51_FULL_PARITY_HEARTBEAT="
                        + json.dumps(
                            {
                                "phase": NUMERICAL_PHASE,
                                "elapsed_seconds": round(elapsed, 3),
                                "output_bytes": temporary_log.stat().st_size,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        returncode = process.returncode
        assert returncode is not None
        output = temporary_log.read_text(encoding="utf-8")
        for line in output.splitlines():
            if "CPPMEGA_PARITY" in line or "completes to compile kernel" in line:
                print(line, flush=True)
        counts = junit_counts()
        markers = {
            marker: output.count(marker)
            for marker in (
                "Data race detected",
                "SCALARIZE",
                "Layout infer conflict",
                "Layout may conflict with ReduceOp",
                "ReduceOp",
                "SIGSEGV",
                "Segmentation fault",
            )
        }
        after = runtime_hashes()
        receipt.update(
            {
                "returncode": returncode,
                "elapsed_seconds": round(time.time() - started, 3),
                "junit": counts,
                "marker_counts": markers,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "output_bytes": len(output.encode()),
                "output_tail": output[-16000:],
                "source_sha256_after_test": after,
            }
        )
        expected_junit = {
            "present": True,
            "tests": len(TEST_NODES),
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        }
        exact_pass = (
            returncode == 0
            and counts == expected_junit
            and after == before
            and markers["ReduceOp"] == 0
            and markers["SIGSEGV"] == 0
            and markers["Segmentation fault"] == 0
        )
        if not exact_pass:
            raise RuntimeError(
                "full parity contract failed: "
                f"returncode={returncode}, junit={counts}, markers={markers}, "
                f"source_unchanged={after == before}"
            )
        receipt["status"] = "green"
        receipt["verdict"] = {
            "numerical_parity_passed": True,
            "data_race_warnings_diagnostic_only": True,
            "source_identity_preserved": True,
            "merge_authorized": False,
        }
    except BaseException as exc:  # noqa: BLE001 - always seal a terminal receipt
        receipt.update(
            {
                "status": "red",
                "elapsed_seconds": round(time.time() - started, 3),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback_tail": traceback.format_exc()[-16000:],
                "verdict": {
                    "numerical_parity_passed": False,
                    "merge_authorized": False,
                },
            }
        )
        if "source_sha256_before_test" in receipt:
            try:
                receipt.setdefault("source_sha256_after_test", runtime_hashes())
            except OSError as post_exc:
                receipt["post_failure_source_hash_error"] = repr(post_exc)
    write_receipt(receipt)
    print(
        "TILELANG_A760_OLD51_FULL_PARITY_RESULT=" + json.dumps(receipt, sort_keys=True),
        flush=True,
    )
    return receipt


def main() -> None:
    import time
    import traceback
    import uuid

    run_id = uuid.uuid4().hex
    remote_paths = _artifact_paths(pathlib.Path("/tmp"), run_id)
    local_paths = _artifact_paths(LOCAL_CANDIDATE_ROOT, run_id)
    sandbox: modal.Sandbox | None = None
    sandbox_id: str | None = None
    process_returncode: int | None = None
    transport_error: dict[str, str] | None = None
    artifact_read_errors: dict[str, str] = {}
    started = time.time()

    def persist(name: str, payload: bytes) -> None:
        destination = local_paths[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{run_id}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)

    try:
        sandbox = modal.Sandbox.create(
            "sleep",
            "infinity",
            app=app,
            image=_image(),
            gpu=GPU_SPEC,
            cpu=8,
            memory=131_072,
            timeout=3600,
            workdir="/opt/cppmega",
        )
        sandbox_id = sandbox.object_id
        process = sandbox.exec(
            "python",
            REMOTE_RUNNER,
            "--sandbox-worker",
            env={
                "CPPMEGA_GATE_RUN_ID": run_id,
                "CPPMEGA_MODAL_SANDBOX_ID": sandbox_id,
                "CPPMEGA_SANDBOX_WORKER": "1",
            },
            timeout=3300,
            workdir="/opt/cppmega",
        )
        for line in process.stdout:
            print(line, end="", flush=True)
        stderr = process.stderr.read()
        if stderr:
            print(stderr, file=sys.stderr, end="", flush=True)
        process_returncode = process.wait()
    except BaseException as exc:  # noqa: BLE001 - preserve evidence before exit
        transport_error = {
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback_tail": traceback.format_exc()[-16000:],
        }
    finally:
        if sandbox is not None:
            for name, remote_path in remote_paths.items():
                try:
                    persist(name, sandbox.filesystem.read_bytes(str(remote_path)))
                except BaseException as exc:  # noqa: BLE001 - collect all artifacts
                    artifact_read_errors[name] = repr(exc)
            try:
                sandbox.terminate(wait=True)
            except BaseException as exc:  # noqa: BLE001 - retain prior evidence
                artifact_read_errors["sandbox_terminate"] = repr(exc)

    receipt_path = local_paths["receipt"]
    if not receipt_path.is_file():
        fallback = {
            "schema_version": 1,
            "gate": (
                f"H200_{NUMERICAL_PHASE.upper()}_A760_OLD51_FULL_NUMERICAL_PARITY"
            ),
            "phase": NUMERICAL_PHASE,
            "status": "red",
            "started_unix": started,
            "elapsed_seconds": round(time.time() - started, 3),
            "runner_sha256": SCRIPT_SHA256,
            "modal": {
                "sandbox_id": sandbox_id,
                "run_id": run_id,
                "gpu_spec": GPU_SPEC,
                "app_name": APP_NAME,
            },
            "transport": {
                "process_returncode": process_returncode,
                "error": transport_error,
                "artifact_read_errors": artifact_read_errors,
            },
            "artifacts": {name: str(path) for name, path in local_paths.items()},
            "verdict": {
                "numerical_parity_passed": False,
                "merge_authorized": False,
            },
        }
        persist("receipt", json.dumps(fallback, indent=2, sort_keys=True).encode())

    result = json.loads(receipt_path.read_text(encoding="utf-8"))
    print(
        "TILELANG_A760_OLD51_FULL_PARITY_SUMMARY="
        + json.dumps(
            {
                "phase": NUMERICAL_PHASE,
                "status": result.get("status"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "junit": result.get("junit"),
                "marker_counts": result.get("marker_counts"),
                "artifacts": {name: str(path) for name, path in local_paths.items()},
                "sandbox_id": sandbox_id,
                "run_id": run_id,
                "process_returncode": process_returncode,
                "artifact_read_errors": artifact_read_errors,
            },
            sort_keys=True,
        )
    )
    if (
        result.get("status") != "green"
        or process_returncode != 0
        or transport_error is not None
        or artifact_read_errors
    ):
        raise RuntimeError(
            "sandbox H200 gate did not produce a complete green durable receipt: "
            f"status={result.get('status')!r}, "
            f"process_returncode={process_returncode!r}, "
            f"transport_error={transport_error!r}, "
            f"artifact_read_errors={artifact_read_errors!r}"
        )


if not IS_SANDBOX_WORKER:
    main = app.local_entrypoint()(main)


if __name__ == "__main__":
    if not IS_SANDBOX_WORKER or sys.argv[1:] != ["--sandbox-worker"]:
        raise SystemExit("expected exactly --sandbox-worker")
    run_gate()
