"""Run the real NAM56R no-conv document-isolation gate on Modal H200s."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any
import uuid

import modal


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_SOURCE_SHA = "0c2967ee214d7f86d1b2b6172f23f1a4012b0c69"
_IMAGE_SOURCE_SHA = "54f0c8c5ed6166f1bee02928404b2477dab6b69e"
_IMAGE_DIGEST = "sha256:85a09018ab4689c09025c8eb2e732242c80392167e4d5d59418da49184af970a"
_IMAGE_REF = f"ghcr.io/datasunriseou/cppmega@{_IMAGE_DIGEST}"
_KNOWN_CRYPTOGRAPHY_DEFECT_IMAGE_DIGEST = (
    "sha256:85a09018ab4689c09025c8eb2e732242c80392167e4d5d59418da49184af970a"
)
_MEGATRON_SHA = "ba7b5ebce12af60627a80985792a1449ce45f46c"
_REMOTE_ROOT = Path("/opt/cppmega")
_REMOTE_RESULTS_ROOT = Path("/results/noconv-document-isolation-h200")
_LOCAL_RESULTS_ROOT = Path(
    os.environ.get(
        "CPPMEGA_H200_RECEIPT_ROOT",
        "/Volumes/external/cppmega_data/h200_receipts/noconv_document_isolation",
    )
)
_TRACKED_SOURCE_PATHS = (
    "cppmega/megatron/document_isolation.py",
    "cppmega/megatron/nam56r_noconv_spec.py",
    "cppmega/megatron/noconv_mamba_mixer.py",
    "cppmega/megatron/structure_dataset_patch.py",
)
_TEST_PATH = "tests/test_noconv_document_isolation_h200.py"
_OVERLAY_PATHS = (
    *_TRACKED_SOURCE_PATHS,
    _TEST_PATH,
)
_PRODUCTION_SOURCE_SHA256 = {
    "cppmega/megatron/document_isolation.py": (
        "d5bbf8c4c718c984281ad7dcb89e813c4285ca9bc0c10d180dadbd0227d24c02"
    ),
    "cppmega/megatron/nam56r_noconv_spec.py": (
        "842a514c3f2ebb15b21a3a67e4407fb7912345ea44feb4392652a39cfab606bf"
    ),
    "cppmega/megatron/noconv_mamba_mixer.py": (
        "2fcbf6142e75b4732c7bea939cc60575f6c25797100acf6de2c6a9db2bc82b63"
    ),
    "cppmega/megatron/structure_dataset_patch.py": (
        "c7d6d96e8359c0096487f4b557c3c970aa99fce9fcdbab57a262c847e146d91c"
    ),
}
_TEST_SHA256 = "e6b410e19d1a0e7fa981badd6f104da58c9036b1733804be8e973e4c1c2e8541"
_PRODUCTION_MANIFEST_SHA256 = hashlib.sha256(
    json.dumps(
        _PRODUCTION_SOURCE_SHA256,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
if modal.is_local():
    _CHECKOUT_SHA = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
else:
    _CHECKOUT_SHA = os.environ["CPPMEGA_GATE_CHECKOUT_COMMIT"]
    _SCRIPT_SHA256 = os.environ["CPPMEGA_NOCONV_GATE_SCRIPT_SHA256"]
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _SCRIPT_SHA256:
        raise RuntimeError("Modal gate script hash mismatch")
_PHASES = {
    "tp1": (
        "tests/test_noconv_document_isolation_h200.py::"
        "test_real_noconv_tp1_document_isolation",
    ),
    "distributed2": (
        "tests/test_noconv_document_isolation_h200.py::"
        "test_real_noconv_tp2_sequence_parallel_document_isolation",
        "tests/test_noconv_document_isolation_h200.py::"
        "test_real_noconv_cp2_zigzag_document_isolation",
    ),
    "cartesian4": (
        "tests/test_noconv_document_isolation_h200.py::"
        "test_real_noconv_tp2_cp2_cartesian_document_isolation",
    ),
}
_EXPECTED_EVIDENCE = {
    "tp1": ("tp1",),
    "distributed2": ("tp2_sp", "cp2"),
    "cartesian4": ("tp2_cp2",),
}
_EXPECTED_VERSIONS = {
    "flash-attn": "2.8.3",
    "mamba-ssm": "2.3.1",
    "transformer-engine": "2.16.0+4220403e",
}


def _image() -> modal.Image:
    image: Any = modal.Image.from_registry(
        _IMAGE_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    ).env(
        {
            "CPPMEGA_GATE_CHECKOUT_COMMIT": _CHECKOUT_SHA,
            "CPPMEGA_MEGATRON_COMMIT": _MEGATRON_SHA,
            "CPPMEGA_NOCONV_GATE_SCRIPT_SHA256": _SCRIPT_SHA256,
            "CPPMEGA_PRODUCTION_SOURCE_COMMIT": _PRODUCTION_SOURCE_SHA,
            "MEGATRON_LM_REPO": "/opt/megatron-lm",
            "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
            "WANDB_MODE": "disabled",
        }
    )
    for relative_path in _OVERLAY_PATHS:
        image = image.add_local_file(
            str(_REPO_ROOT / relative_path),
            remote_path=str(_REMOTE_ROOT / relative_path),
            copy=True,
        )
    return image


app = modal.App("cppmega-noconv-document-isolation-h200")
results = modal.Volume.from_name("cppmega-test-results", create_if_missing=True)
runtime_image = _image() if modal.is_local() else None


def _remote_source_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    return {
        path: hashlib.sha256((_REMOTE_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _junit_counts(path: Path) -> dict[str, int | bool]:
    import xml.etree.ElementTree as ET

    if not path.is_file():
        return {
            "present": False,
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        }
    root = ET.parse(path).getroot()
    suites = [root] if root.tag.rsplit("}", 1)[-1] == "testsuite" else [
        child
        for child in root
        if child.tag.rsplit("}", 1)[-1] == "testsuite"
    ]
    if not suites:
        raise RuntimeError(f"{path} contains no testsuite elements")
    counts: dict[str, int | bool] = {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    counts["present"] = True
    return counts


def _runtime_provenance(expected_gpus: int) -> dict[str, Any]:
    from importlib import metadata
    import sys

    import flash_attn
    import mamba_ssm
    import torch
    import transformer_engine
    import transformer_engine.pytorch

    image_source = json.loads(
        Path("/opt/cppmega-image-source.json").read_text()
    )
    if image_source.get("cppmega_sha") != _IMAGE_SOURCE_SHA:
        raise RuntimeError(
            "production image source mismatch: "
            f"expected {_IMAGE_SOURCE_SHA}, observed {image_source!r}"
        )

    megatron = subprocess.run(
        ["git", "-C", "/opt/megatron-lm", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    observed_megatron = megatron.stdout.strip()
    if megatron.returncode != 0 or observed_megatron != _MEGATRON_SHA:
        raise RuntimeError(
            "Megatron source mismatch: "
            f"expected {_MEGATRON_SHA}, observed {observed_megatron!r}, "
            f"stderr={megatron.stderr!r}"
        )

    versions = {
        distribution: metadata.version(distribution)
        for distribution in _EXPECTED_VERSIONS
    }
    mismatches = {
        name: {"expected": expected, "observed": versions[name]}
        for name, expected in _EXPECTED_VERSIONS.items()
        if versions[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"production dependency mismatch: {mismatches}")
    if str(torch.__version__) != "2.13.0+cu132" or torch.version.cuda != "13.2":
        raise RuntimeError(
            "torch/CUDA mismatch: "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpus:
        raise RuntimeError(
            f"expected exactly {expected_gpus} CUDA devices, "
            f"observed {torch.cuda.device_count()}"
        )
    devices = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]
    if any("H200" not in device for device in devices):
        raise RuntimeError(f"expected only H200 devices, observed {devices}")

    try:
        cryptography = metadata.distribution("cryptography")
    except metadata.PackageNotFoundError:
        if _IMAGE_DIGEST == _KNOWN_CRYPTOGRAPHY_DEFECT_IMAGE_DIGEST:
            cryptography_provenance = {
                "distribution": "cryptography",
                "expectation": "known defect in the pinned old production image",
                "status": "known_defect_distribution_not_discoverable",
                "gate_impact": (
                    "excluded from this kernel-scoped gate; "
                    "no whole-image health claim"
                ),
            }
        else:
            cryptography_provenance = {
                "distribution": "cryptography",
                "expectation": "absent in the fixed image candidate",
                "status": "absent",
            }
    else:
        cryptography_wheel = cryptography.read_text("WHEEL") or ""
        cryptography_tags = [
            line.removeprefix("Tag: ").strip()
            for line in cryptography_wheel.splitlines()
            if line.startswith("Tag: ")
        ]
        if _IMAGE_DIGEST != _KNOWN_CRYPTOGRAPHY_DEFECT_IMAGE_DIGEST:
            raise RuntimeError(
                "fixed image candidate unexpectedly contains cryptography: "
                f"version={cryptography.version}, tags={cryptography_tags}"
            )
        if (
            cryptography.version != "41.0.7"
            or (
                cryptography_tags
                and cryptography_tags != ["cp312-cp312-linux_x86_64"]
            )
            or sys.implementation.cache_tag != "cpython-313"
        ):
            raise RuntimeError(
                "known cryptography defect changed unexpectedly: "
                f"version={cryptography.version}, tags={cryptography_tags}, "
                f"runtime_cache_tag={sys.implementation.cache_tag}"
            )
        cryptography_provenance = {
            "distribution": "cryptography",
            "expectation": "known defect in the pinned old production image",
            "status": "known_incompatible_wheel",
            "version": cryptography.version,
            "wheel_metadata_status": (
                "present" if cryptography_tags else "missing"
            ),
            "wheel_tags": cryptography_tags,
            "runtime_cache_tag": sys.implementation.cache_tag,
            "gate_impact": (
                "excluded from this kernel-scoped gate; no whole-image health claim"
            ),
        }
    nvidia_smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return {
        "base_image_ref": _IMAGE_REF,
        "base_image_digest": _IMAGE_DIGEST,
        "image_source": image_source,
        "megatron_expected_sha": _MEGATRON_SHA,
        "megatron_observed_sha": observed_megatron,
        "versions": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            **versions,
        },
        "scoped_imports": {
            "flash_attn": flash_attn.__name__,
            "mamba_ssm": mamba_ssm.__name__,
            "torch": torch.__name__,
            "transformer_engine": transformer_engine.__name__,
            "transformer_engine.pytorch": transformer_engine.pytorch.__name__,
        },
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "device_capabilities": [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ],
        "cryptography": cryptography_provenance,
        "nvidia_smi": {
            "returncode": nvidia_smi.returncode,
            "stdout": nvidia_smi.stdout,
            "stderr": nvidia_smi.stderr,
        },
    }


def _write_remote(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    results.commit()


def _run_phase(
    run_id: str,
    phase: str,
    *,
    expected_gpus: int,
) -> dict[str, Any]:
    import sys
    import time
    import traceback

    selected_tests = _PHASES[phase]
    phase_root = _REMOTE_RESULTS_ROOT / run_id / phase
    receipt_path = phase_root / "receipt.json"
    stdout_path = phase_root / "stdout.log"
    stderr_path = phase_root / "stderr.log"
    junit_path = phase_root / "junit.xml"
    evidence_dir = phase_root / "evidence"
    started = time.time()
    receipt: dict[str, Any] = {
        "schema": "cppmega.noconv_document_isolation_h200.receipt",
        "schema_version": 1,
        "run_id": run_id,
        "phase": phase,
        "status": "running",
        "function_call_id": modal.current_function_call_id(),
        "source": {
            "checkout_sha": _CHECKOUT_SHA,
            "required_production_ancestor": _PRODUCTION_SOURCE_SHA,
            "production_manifest_sha256": _PRODUCTION_MANIFEST_SHA256,
            "production_sha256_expected": _PRODUCTION_SOURCE_SHA256,
            "test_path": _TEST_PATH,
            "test_sha256_expected": _TEST_SHA256,
            "script_sha256": _SCRIPT_SHA256,
        },
        "image": {
            "ref": _IMAGE_REF,
            "digest": _IMAGE_DIGEST,
            "source_sha": _IMAGE_SOURCE_SHA,
        },
        "selected_tests": list(selected_tests),
        "expected_test_count": len(selected_tests),
        "expected_gpus": expected_gpus,
        "started_unix": started,
        "remote_artifacts": {
            "receipt": str(receipt_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "junit": str(junit_path),
            "evidence_dir": str(evidence_dir),
        },
    }
    artifacts = {"stdout": "", "stderr": "", "junit_xml": ""}
    _write_remote(receipt_path, receipt)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *selected_tests,
        "-vv",
        "-s",
        "--tb=long",
        f"--junitxml={junit_path}",
    ]
    try:
        observed_production_hashes = _remote_source_hashes(_TRACKED_SOURCE_PATHS)
        if observed_production_hashes != _PRODUCTION_SOURCE_SHA256:
            raise RuntimeError(
                "production source overlay mismatch: "
                f"expected {_PRODUCTION_SOURCE_SHA256}, "
                f"observed {observed_production_hashes}"
            )
        observed_test_hash = _remote_source_hashes((_TEST_PATH,))[_TEST_PATH]
        if observed_test_hash != _TEST_SHA256:
            raise RuntimeError(
                "test overlay mismatch: "
                f"expected {_TEST_SHA256}, observed {observed_test_hash}"
            )
        receipt["source"]["production_sha256_observed_before"] = (
            observed_production_hashes
        )
        receipt["source"]["test_sha256_observed_before"] = observed_test_hash
        receipt["runtime"] = _runtime_provenance(expected_gpus)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CPPMEGA_NOCONV_EVIDENCE_DIR"] = str(evidence_dir)
        environment["CPPMEGA_STRUCTURE_ENABLED"] = "1"
        process = subprocess.run(
            command,
            cwd=_REMOTE_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=1500,
        )
        stdout_path.write_text(process.stdout)
        stderr_path.write_text(process.stderr)
        artifacts["stdout"] = process.stdout
        artifacts["stderr"] = process.stderr
        receipt["pytest"] = {
            "command": command,
            "returncode": process.returncode,
            "junit": _junit_counts(junit_path),
        }
        observed_production_after = _remote_source_hashes(_TRACKED_SOURCE_PATHS)
        observed_test_after = _remote_source_hashes((_TEST_PATH,))[_TEST_PATH]
        receipt["source"]["production_sha256_observed_after"] = (
            observed_production_after
        )
        receipt["source"]["test_sha256_observed_after"] = observed_test_after
        if (
            observed_production_after != observed_production_hashes
            or observed_test_after != observed_test_hash
        ):
            raise RuntimeError("source overlay changed while the gate was running")

        evidence = {}
        for topology in _EXPECTED_EVIDENCE[phase]:
            evidence_path = evidence_dir / f"{topology}.json"
            if not evidence_path.is_file():
                raise RuntimeError(f"missing runtime evidence: {evidence_path}")
            reports = json.loads(evidence_path.read_text())
            expected_world = {
                "tp1": 1,
                "tp2_sp": 2,
                "cp2": 2,
                "tp2_cp2": 4,
            }[topology]
            if len(reports) != expected_world or any(
                report.get("status") != "passed" for report in reports
            ):
                raise RuntimeError(
                    f"invalid {topology} runtime evidence: {reports}"
                )
            for report in reports:
                if not report["mixer_class"].endswith(".NoConvMamba3BCMixer"):
                    raise RuntimeError(
                        f"{topology} selected the wrong mixer: {report}"
                    )
                if not report["kernel_autograd_nodes"]:
                    raise RuntimeError(
                        f"{topology} has no actual SSD autograd evidence: {report}"
                    )
            evidence[topology] = reports
        receipt["evidence"] = evidence

        counts = receipt["pytest"]["junit"]
        exact_pass = (
            process.returncode == 0
            and counts["present"] is True
            and counts["tests"] == len(selected_tests)
            and counts["failures"] == 0
            and counts["errors"] == 0
            and counts["skipped"] == 0
        )
        if not exact_pass:
            raise RuntimeError(
                "pytest did not produce the exact required result: "
                f"returncode={process.returncode}, junit={counts}"
            )
        receipt["status"] = "passed"
    except Exception as exc:
        failure_traceback = traceback.format_exc()
        receipt.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": failure_traceback,
            }
        )
        if not artifacts["stderr"]:
            artifacts["stderr"] = failure_traceback
    stdout_path.write_text(artifacts["stdout"])
    stderr_path.write_text(artifacts["stderr"])
    if junit_path.is_file():
        artifacts["junit_xml"] = junit_path.read_text()
    receipt["elapsed_seconds"] = round(time.time() - started, 3)
    _write_remote(receipt_path, receipt)
    return {"receipt": receipt, "artifacts": artifacts}


@app.function(
    image=runtime_image,
    gpu="H200:1",
    timeout=1800,
    volumes={"/results": results},
)
def run_tp1(run_id: str) -> dict[str, Any]:
    return _run_phase(run_id, "tp1", expected_gpus=1)


@app.function(
    image=runtime_image,
    gpu="H200:2",
    timeout=1800,
    volumes={"/results": results},
)
def run_distributed2(run_id: str) -> dict[str, Any]:
    return _run_phase(run_id, "distributed2", expected_gpus=2)


@app.function(
    image=runtime_image,
    gpu="H200:4",
    timeout=1800,
    volumes={"/results": results},
)
def run_cartesian4(run_id: str) -> dict[str, Any]:
    return _run_phase(run_id, "cartesian4", expected_gpus=4)


def _local_source_guard() -> None:
    observed_production_hashes = {
        path: hashlib.sha256((_REPO_ROOT / path).read_bytes()).hexdigest()
        for path in _TRACKED_SOURCE_PATHS
    }
    if observed_production_hashes != _PRODUCTION_SOURCE_SHA256:
        raise RuntimeError(
            "local production source overlay mismatch: "
            f"expected {_PRODUCTION_SOURCE_SHA256}, "
            f"observed {observed_production_hashes}"
        )
    observed_test_hash = hashlib.sha256(
        (_REPO_ROOT / _TEST_PATH).read_bytes()
    ).hexdigest()
    if observed_test_hash != _TEST_SHA256:
        raise RuntimeError(
            "local test hash mismatch: "
            f"expected {_TEST_SHA256}, observed {observed_test_hash}"
        )
    observed_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if observed_sha != _CHECKOUT_SHA:
        raise RuntimeError(
            f"checkout moved after runner import: expected {_CHECKOUT_SHA}, "
            f"observed {observed_sha}"
        )
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            _PRODUCTION_SOURCE_SHA,
            observed_sha,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"required production source {_PRODUCTION_SOURCE_SHA} is not an "
            f"ancestor of checkout {observed_sha}: {ancestor.stderr}"
        )
    for path in _TRACKED_SOURCE_PATHS:
        changed = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", path],
            cwd=_REPO_ROOT,
            check=False,
        )
        if changed.returncode != 0:
            raise RuntimeError(f"tracked source overlay is dirty: {path}")


def _persist_local(
    run_root: Path,
    phase: str,
    result: dict[str, Any],
) -> None:
    receipt = result["receipt"]
    artifacts = result["artifacts"]
    phase_root = run_root / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    (phase_root / "receipt.raw.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True)
    )
    (phase_root / "stdout.log").write_text(artifacts["stdout"])
    (phase_root / "stderr.log").write_text(artifacts["stderr"])
    (phase_root / "junit.xml").write_text(artifacts["junit_xml"])
    for topology, reports in receipt.get("evidence", {}).items():
        evidence_path = phase_root / "evidence" / f"{topology}.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(reports, indent=2, sort_keys=True))


@app.local_entrypoint()
def main() -> None:
    _local_source_guard()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}-{_CHECKOUT_SHA[:8]}"
    run_root = _LOCAL_RESULTS_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    phase_calls = [
        ("tp1", run_tp1),
        ("distributed2", run_distributed2),
        ("cartesian4", run_cartesian4),
    ]

    summaries = []
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    for phase, function in phase_calls:
        result = function.remote(run_id)
        _persist_local(run_root, phase, result)
        receipt = result["receipt"]
        junit = receipt.get("pytest", {}).get("junit", {})
        total_tests += int(junit.get("tests", 0))
        total_failures += int(junit.get("failures", 0))
        total_errors += int(junit.get("errors", 0))
        total_skipped += int(junit.get("skipped", 0))
        summaries.append(
            {
                "phase": phase,
                "status": receipt["status"],
                "function_call_id": receipt.get("function_call_id"),
                "elapsed_seconds": receipt.get("elapsed_seconds"),
            }
        )
        if receipt["status"] != "passed":
            manifest = {
                "schema": "cppmega.noconv_document_isolation_h200.run",
                "schema_version": 1,
                "run_id": run_id,
                "status": "failed",
                "checkout_sha": _CHECKOUT_SHA,
                "required_production_ancestor": _PRODUCTION_SOURCE_SHA,
                "production_manifest_sha256": _PRODUCTION_MANIFEST_SHA256,
                "test_sha256": _TEST_SHA256,
                "script_sha256": _SCRIPT_SHA256,
                "image_digest": _IMAGE_DIGEST,
                "phases": summaries,
            }
            (run_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
            raise RuntimeError(
                f"{phase} failed; durable receipt: "
                f"{run_root / phase / 'receipt.raw.json'}"
            )

    if (
        total_tests != 4
        or total_failures != 0
        or total_errors != 0
        or total_skipped != 0
    ):
        raise RuntimeError(
            "aggregate pytest result was not exactly 4 passed / 0 skipped: "
            f"tests={total_tests}, failures={total_failures}, "
            f"errors={total_errors}, skipped={total_skipped}"
        )
    manifest = {
        "schema": "cppmega.noconv_document_isolation_h200.run",
        "schema_version": 1,
        "run_id": run_id,
        "status": "passed",
        "checkout_sha": _CHECKOUT_SHA,
        "required_production_ancestor": _PRODUCTION_SOURCE_SHA,
        "production_manifest_sha256": _PRODUCTION_MANIFEST_SHA256,
        "test_sha256": _TEST_SHA256,
        "script_sha256": _SCRIPT_SHA256,
        "image_ref": _IMAGE_REF,
        "image_digest": _IMAGE_DIGEST,
        "image_source_sha": _IMAGE_SOURCE_SHA,
        "megatron_sha": _MEGATRON_SHA,
        "junit": {
            "tests": total_tests,
            "failures": total_failures,
            "errors": total_errors,
            "skipped": total_skipped,
        },
        "phases": summaries,
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"NOCONV_RECEIPT_DIR={run_root}")
