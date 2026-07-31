"""Run packed-document SP/CP NCCL parity on two Modal H200 GPUs.

Usage:
    modal run scripts/modal_document_isolation_cp.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any

import modal

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GHCR_IMAGE_DIGEST = (
    "sha256:ef5398736aced3de8e5f5c544554dca1789e312abfffae06b4ab701297e60fc4"
)
_GHCR_REF = (
    f"{os.environ.get('GHCR_REPO', 'ghcr.io/datasunriseou/cppmega')}"
    f"@{_GHCR_IMAGE_DIGEST}"
)
_MEGATRON_COMMIT = "ba7b5ebce12af60627a80985792a1449ce45f46c"
_GPU_SPEC = "H200:2"
_RESULT_PATH = "/results/document-isolation-cp-h200-latest.json"
_JUNIT_PATH = "/tmp/document-isolation-cp-h200-junit.xml"
_EXPECTED_TEST_COUNT = 2
_SELECTED_TESTS = (
    "tests/test_document_isolation_cp.py::"
    "test_nccl_two_gpu_sp_cp_document_isolation_forward_backward_parity",
    "tests/test_document_isolation_cp.py::"
    "test_nccl_two_gpu_actual_m2rnn_sp_cp_document_isolation_parity",
)
_SOURCE_PATHS = (
    "cppmega/megatron/document_isolation.py",
    "cppmega/megatron/author_mamba3_spec.py",
    "cppmega/megatron/cppmega_mamba3_tp_mixer.py",
    "cppmega/megatron/m2rnn_spec.py",
    "tests/test_document_isolation_cp.py",
)
_SCRIPT_SHA256 = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()


def _image() -> modal.Image:
    image: Any = modal.Image.from_registry(
        _GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    ).env(
        {
            "CPPMEGA_MEGATRON_COMMIT": _MEGATRON_COMMIT,
            "MEGATRON_LM_REPO": "/opt/megatron-lm",
            "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
            "WANDB_MODE": "disabled",
        }
    )
    return (
        image.pip_install("pytest")
        .add_local_dir(
            str(_REPO_ROOT / "cppmega"),
            remote_path="/opt/cppmega/cppmega",
            copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
        .add_local_file(
            str(_REPO_ROOT / "tests" / "test_document_isolation_cp.py"),
            remote_path="/opt/cppmega/tests/test_document_isolation_cp.py",
            copy=True,
        )
        .add_local_file(
            str(_REPO_ROOT / "data" / "domain_schema_v1.json"),
            remote_path="/opt/cppmega/data/domain_schema_v1.json",
            copy=True,
        )
        .add_local_file(
            str(
                _REPO_ROOT
                / "data"
                / "tokenizer_v2"
                / "tokenizer_contract_v1.json"
            ),
            remote_path=(
                "/opt/cppmega/data/tokenizer_v2/tokenizer_contract_v1.json"
            ),
            copy=True,
        )
        .add_local_file(
            str(_REPO_ROOT / "pyproject.toml"),
            remote_path="/opt/cppmega/pyproject.toml",
            copy=True,
        )
    )


app = modal.App("cppmega-document-isolation-cp")
results = modal.Volume.from_name("cppmega-test-results", create_if_missing=True)


@app.function(
    image=_image(),
    gpu=_GPU_SPEC,
    timeout=900,
    volumes={"/results": results},
)
def run_nccl_parity() -> dict[str, Any]:
    import hashlib
    from importlib import metadata
    import subprocess
    import sys
    import time
    import traceback
    import uuid
    import xml.etree.ElementTree as ET

    import torch

    def write_receipt(receipt: dict[str, Any]) -> None:
        path = pathlib.Path(_RESULT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
        results.commit()

    def source_hashes() -> dict[str, str]:
        hashes = {"scripts/modal_document_isolation_cp.py": _SCRIPT_SHA256}
        for relative_path in _SOURCE_PATHS:
            path = pathlib.Path("/opt/cppmega") / relative_path
            hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes

    def stack_provenance() -> dict[str, Any]:
        revision = subprocess.run(
            ["git", "-C", "/opt/megatron-lm", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        versions = {}
        for distribution in (
            "mamba-ssm",
            "tilelang",
            "apache-tvm-ffi",
            "transformer-engine",
        ):
            try:
                versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                versions[distribution] = None
        return {
            "megatron_expected_commit": _MEGATRON_COMMIT,
            "megatron_observed_commit": revision.stdout.strip(),
            "megatron_rev_parse_returncode": revision.returncode,
            "megatron_rev_parse_stderr": revision.stderr,
            "versions": {
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                **versions,
            },
        }

    def junit_counts() -> dict[str, int | bool]:
        path = pathlib.Path(_JUNIT_PATH)
        if not path.is_file():
            return {
                "present": False,
                "tests": 0,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
            }
        root = ET.parse(path).getroot()
        if root.tag.rsplit("}", 1)[-1] == "testsuite":
            suites = [root]
        else:
            suites = [
                child
                for child in root
                if child.tag.rsplit("}", 1)[-1] == "testsuite"
            ]
        if not suites:
            raise RuntimeError(f"{_JUNIT_PATH} contains no testsuite elements")
        counts: dict[str, int | bool] = {
            name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
            for name in ("tests", "failures", "errors", "skipped")
        }
        counts["present"] = True
        return counts

    started = time.time()
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "run_id": str(uuid.uuid4()),
        "status": "running",
        "image": _GHCR_REF,
        "image_digest": _GHCR_IMAGE_DIGEST,
        "gpu_spec": _GPU_SPEC,
        "selected_tests": list(_SELECTED_TESTS),
        "expected_test_count": _EXPECTED_TEST_COUNT,
        "started_unix": started,
    }
    # Overwrite any older green receipt before setup/test work starts.
    write_receipt(receipt)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *_SELECTED_TESTS,
        "-vv",
        "-s",
        "--tb=long",
        f"--junitxml={_JUNIT_PATH}",
    ]
    try:
        receipt["source_sha256"] = source_hashes()
        receipt["stack_provenance"] = stack_provenance()
        observed_megatron = receipt["stack_provenance"][
            "megatron_observed_commit"
        ]
        if observed_megatron != _MEGATRON_COMMIT:
            raise RuntimeError(
                "Megatron source mismatch: "
                f"expected {_MEGATRON_COMMIT}, observed {observed_megatron!r}"
            )
        process = subprocess.run(
            command,
            cwd="/opt/cppmega",
            capture_output=True,
            text=True,
            check=False,
            timeout=780,
        )
        receipt.update(
            {
                "returncode": process.returncode,
                "command": command,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        )
        counts = junit_counts()
        receipt["junit"] = counts
        exact_pass = (
            process.returncode == 0
            and counts["present"] is True
            and counts["tests"] == _EXPECTED_TEST_COUNT
            and counts["failures"] == 0
            and counts["errors"] == 0
            and counts["skipped"] == 0
        )
        if not exact_pass:
            raise RuntimeError(
                "pytest did not produce the exact required result: "
                f"returncode={process.returncode}, junit={counts}, "
                f"expected_tests={_EXPECTED_TEST_COUNT}"
            )
    except Exception as exc:
        receipt.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        write_receipt(receipt)
        raise RuntimeError(json.dumps(receipt, indent=2, sort_keys=True)) from exc

    receipt.update(
        {
            "status": "passed",
            "elapsed_seconds": round(time.time() - started, 3),
        }
    )
    write_receipt(receipt)
    return receipt


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run_nccl_parity.remote(), indent=2))
