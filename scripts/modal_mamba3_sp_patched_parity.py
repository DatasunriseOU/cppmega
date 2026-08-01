"""Run Mamba3 TP+SP parity with cppmega's existing GQA backward overlay.

This is a diagnostic overlay, not proof that the unmodified GHCR image works:

    modal run scripts/modal_mamba3_sp_patched_parity.py
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
    "sha256:10dcebb221795e54f32954068b1c158b122d53bc170187b96489e554c4dbeacc"
)
_GHCR_REF = (
    f"{os.environ.get('GHCR_REPO', 'ghcr.io/datasunriseou/cppmega')}"
    f"@{_GHCR_IMAGE_DIGEST}"
)
_MEGATRON_COMMIT = "ba7b5ebce12af60627a80985792a1449ce45f46c"
_GPU_SPEC = "H200:2"
_RESULT_PATH = "/results/mamba3-sp-gqa-overlay-h200-latest.json"
_JUNIT_PATH = "/tmp/mamba3-sp-gqa-overlay-h200-junit.xml"
_SELECTED_TESTS = (
    "tests/test_cppmega_mamba3_tp_mixer.py::test_tp2_sp_on_parity_vs_tp1",
    "tests/test_cppmega_mamba3_tp_mixer.py::"
    "test_tp2_sp_off_replicated_parameter_gradient_parity_vs_tp1",
    "tests/test_cppmega_mamba3_tp_mixer.py::"
    "test_cp2_actual_mamba3_forward_backward_parity_vs_cp1",
)
_EXPECTED_TEST_COUNT = len(_SELECTED_TESTS)
_SOURCE_PATHS = (
    "cppmega/megatron/document_isolation.py",
    "cppmega/megatron/cppmega_mamba3_tp_mixer.py",
    "cppmega/megatron/upstream_patches/apply_mamba3_gqa_bwd_patches.py",
    "tests/test_cppmega_mamba3_tp_mixer.py",
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
            str(_REPO_ROOT / "tests" / "test_cppmega_mamba3_tp_mixer.py"),
            remote_path="/opt/cppmega/tests/test_cppmega_mamba3_tp_mixer.py",
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


app = modal.App("cppmega-mamba3-sp-gqa-overlay")
results = modal.Volume.from_name("cppmega-test-results", create_if_missing=True)


@app.function(
    image=_image(),
    gpu=_GPU_SPEC,
    timeout=900,
    volumes={"/results": results},
)
def run_patched_mamba3_sp_parity() -> dict[str, Any]:
    import contextlib
    import hashlib
    from importlib import metadata
    import io
    import subprocess
    import sys
    import time
    import traceback
    import uuid
    import xml.etree.ElementTree as ET

    import torch

    from cppmega.megatron.upstream_patches import (
        apply_mamba3_gqa_bwd_patches as overlay,
    )

    def write_receipt(receipt: dict[str, Any]) -> None:
        path = pathlib.Path(_RESULT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
        results.commit()

    def hash_paths(paths: dict[str, pathlib.Path]) -> dict[str, str]:
        return {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        }

    def source_hashes() -> dict[str, str]:
        hashes = hash_paths(
            {
                relative_path: pathlib.Path("/opt/cppmega") / relative_path
                for relative_path in _SOURCE_PATHS
            }
        )
        hashes["scripts/modal_mamba3_sp_patched_parity.py"] = _SCRIPT_SHA256
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

    def runtime_fingerprints() -> dict[str, Any]:
        def fingerprint_file(path: pathlib.Path) -> dict[str, Any]:
            try:
                resolved = path.resolve(strict=True)
                return {
                    "path": str(resolved),
                    "size_bytes": resolved.stat().st_size,
                    "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
                }
            except Exception as exc:
                return {
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }

        native_artifacts: dict[str, list[dict[str, Any]]] = {}
        for distribution_name in ("tilelang", "apache-tvm-ffi"):
            try:
                distribution = metadata.distribution(distribution_name)
                files = distribution.files or ()
                candidates = sorted(
                    (
                        file
                        for file in files
                        if pathlib.PurePosixPath(str(file)).suffix
                        in {".so", ".dylib", ".pyd"}
                    ),
                    key=str,
                )
                native_artifacts[distribution_name] = [
                    {
                        "distribution_path": str(file),
                        **fingerprint_file(
                            pathlib.Path(distribution.locate_file(file))
                        ),
                    }
                    for file in candidates
                ]
            except Exception as exc:
                native_artifacts[distribution_name] = [
                    {"error": f"{type(exc).__name__}: {exc}"}
                ]

        python_version = subprocess.run(
            [sys.executable, "-VV"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        relevant_markers = (
            "cuda",
            "cutlass",
            "flash-attn",
            "mamba",
            "nvidia",
            "tilelang",
            "torch",
            "transformer-engine",
            "triton",
            "tvm",
        )
        pip_freeze_subset = [
            line
            for line in pip_freeze.stdout.splitlines()
            if any(marker in line.lower() for marker in relevant_markers)
        ]
        return {
            "python": {
                "executable": sys.executable,
                "version": sys.version,
                "version_command": [sys.executable, "-VV"],
                "version_returncode": python_version.returncode,
                "version_stdout": python_version.stdout,
                "version_stderr": python_version.stderr,
            },
            "nvidia_smi": {
                "command": [
                    "nvidia-smi",
                    "--query-gpu=driver_version,name",
                    "--format=csv,noheader",
                ],
                "returncode": nvidia_smi.returncode,
                "stdout": nvidia_smi.stdout,
                "stderr": nvidia_smi.stderr,
            },
            "pip_freeze": {
                "command": [sys.executable, "-m", "pip", "freeze", "--all"],
                "returncode": pip_freeze.returncode,
                "subset": pip_freeze_subset,
                "stderr": pip_freeze.stderr,
            },
            "torch_cuda_init": fingerprint_file(
                pathlib.Path(torch.cuda.__file__)
            ),
            "native_artifacts": native_artifacts,
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
        "overlay": {
            "name": (
                "cppmega.megatron.upstream_patches."
                "apply_mamba3_gqa_bwd_patches"
            ),
            "status": "pending",
            "coverage": "GQA backward fixed-length and varlen files only",
            "not_covered": (
                "upstream_prs/05_mamba3_dt_fp32_gqa_bwd.patch also changes "
                "mamba_ssm/modules/mamba3.py DT precision; this ephemeral "
                "overlay does not apply that production-wheel change"
            ),
            "production_image_mutated": False,
            "scope": "ephemeral Modal container source files only",
        },
    }
    # Overwrite any older green receipt before overlay/setup/test work starts.
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
        installed_paths = overlay._find_mamba3_bwd_files()
        before_hashes = hash_paths(installed_paths)
        before_applied = overlay._is_gqa_bwd_patch_applied()
        before_absent = overlay._is_gqa_bwd_patch_absent()
        receipt["overlay"].update(
            {
                "before_applied": before_applied,
                "before_absent": before_absent,
                "installed_source_sha256_before": before_hashes,
            }
        )
        if not before_applied and not before_absent:
            receipt["overlay"]["status"] = "invalid_partial_or_unknown_before"
            raise RuntimeError(
                "Mamba3 GQA backward sources are partially patched or "
                "unrecognized; refusing to label them already applied"
            )

        os.environ["CPPMEGA_MAMBA3_GQA_BWD"] = "1"
        os.environ["MAMBA3_GQA_BWD_ALLOW_FILE_MUTATION"] = "1"
        overlay_log = io.StringIO()
        with contextlib.redirect_stdout(overlay_log):
            overlay.apply_all()
        after_applied = overlay._is_gqa_bwd_patch_applied()
        after_hashes = hash_paths(installed_paths)
        receipt["overlay"].update(
            {
                "after_applied": after_applied,
                "installed_source_sha256_after": after_hashes,
                "status": (
                    "gqa_backward_already_present"
                    if before_applied
                    else "gqa_backward_applied_during_run"
                ),
                "container_source_mutated": before_absent,
                "log": overlay_log.getvalue(),
            }
        )
        if not after_applied:
            raise RuntimeError("Mamba3 GQA backward overlay did not apply")

        receipt["runtime_fingerprints"] = runtime_fingerprints()
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
    print(json.dumps(run_patched_mamba3_sp_parity.remote(), indent=2))
