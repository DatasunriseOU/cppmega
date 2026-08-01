"""Exact release-image H200 gate for the native TileLang Bind-role fix.

The immutable release image is selected only by an externally supplied OCI
digest.  The ordered phases are:

``one`` (one R2/32 test) -> ``r2`` (three R2/32 tests) ->
``r4`` (three R4/16 tests).

Stage2 must already be applied by the immutable OCI build. Modal only adds the
gate runner and exact release-wheel receipts. Runtime validation and tests are
read-only: GQA is verified, source hashes are checked before and after pytest,
and no production or test source is replaced or patched.

Required local input:

    CPPMEGA_CANDIDATE_CPPMEGA_SHA=<40 lowercase hex> \
    CPPMEGA_CANDIDATE_IMAGE_DIGEST=sha256:<64 lowercase hex> \
    CPPMEGA_RELEASE_MANIFEST_SHA256=<64 lowercase hex> \
    CPPMEGA_CANDIDATE_WHEELS_JSON='{"tilelang-...whl":"...","apache_tvm_ffi-...whl":"..."}' \
    CPPMEGA_H200_GATE_PHASE=one|r2|r4 \
      modal run scripts/modal_mamba3_tilelang_role_release_gate.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
from typing import Any

import modal

_SCRIPT_PATH = pathlib.Path(__file__).resolve()
_LOCAL_ROOT = _SCRIPT_PATH.parents[1]


def _required_env(name: str, pattern: str) -> str:
    value = os.environ.get(name, "")
    if re.fullmatch(pattern, value) is None:
        raise RuntimeError(f"{name} does not match required pattern {pattern!r}")
    return value


_CANDIDATE_CPPMEGA_SHA = _required_env(
    "CPPMEGA_CANDIDATE_CPPMEGA_SHA",
    r"[0-9a-f]{40}",
)
_CANDIDATE_TILELANG_SHA = "629e3414b13274ddcfdff2082db86373b0c218ae"
_CANDIDATE_TVM_SHA = "e25ca6ae50beee0e907b1e5ed32949879caddde1"
_CANDIDATE_TVM_FFI_SHA = "521efeb30bfd9e4946b248b3d76e6391028233a3"
_BASE_SOURCE_SHA = "dbfe51e1b9173e8cc9550c6b269da2c8d20c7f39"
_MEGATRON_COMMIT = "ba7b5ebce12af60627a80985792a1449ce45f46c"
_RELEASE_TAG = f"wheels-{_CANDIDATE_CPPMEGA_SHA}"
_RELEASE_MANIFEST_SHA256 = _required_env(
    "CPPMEGA_RELEASE_MANIFEST_SHA256",
    r"[0-9a-f]{64}",
)
_RELEASE_MANIFEST_URL = (
    "https://github.com/DatasunriseOU/cppmega/releases/download/"
    f"{_RELEASE_TAG}/CANDIDATE_MANIFEST.json"
)
try:
    _CANDIDATE_WHEELS_VALUE = json.loads(
        os.environ.get("CPPMEGA_CANDIDATE_WHEELS_JSON", "")
    )
except json.JSONDecodeError as exc:
    raise RuntimeError(
        "CPPMEGA_CANDIDATE_WHEELS_JSON must be exact JSON"
    ) from exc
if not isinstance(_CANDIDATE_WHEELS_VALUE, dict):
    raise TypeError("CPPMEGA_CANDIDATE_WHEELS_JSON must be a JSON object")
_CANDIDATE_WHEELS = {
    str(filename): str(digest)
    for filename, digest in _CANDIDATE_WHEELS_VALUE.items()
}
if (
    len(_CANDIDATE_WHEELS) != 2
    or sum(name.startswith("tilelang-") for name in _CANDIDATE_WHEELS) != 1
    or sum(name.startswith("apache_tvm_ffi-") for name in _CANDIDATE_WHEELS) != 1
    or any(
        re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", name) is None
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for name, digest in _CANDIDATE_WHEELS.items()
    )
):
    raise RuntimeError(
        "CPPMEGA_CANDIDATE_WHEELS_JSON must contain exactly one pinned "
        "TileLang wheel and one pinned apache-tvm-ffi wheel"
    )
_IMAGE_DIGEST = _required_env(
    "CPPMEGA_CANDIDATE_IMAGE_DIGEST",
    r"sha256:[0-9a-f]{64}",
)
_IMAGE_REF = f"ghcr.io/datasunriseou/cppmega@{_IMAGE_DIGEST}"
_ATTEMPT = os.environ.get("CPPMEGA_H200_GATE_ATTEMPT", "a1")
if re.fullmatch(r"a[1-9][0-9]*", _ATTEMPT) is None:
    raise RuntimeError("CPPMEGA_H200_GATE_ATTEMPT must match a positive aN token")

_THREE_TESTS = (
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
_PHASE_CONFIG = {
    "one": {
        "selected_tests": _THREE_TESTS[:1],
        "mimo_rank": 2,
        "chunk_size": 32,
        "prerequisite_phase": None,
    },
    "r2": {
        "selected_tests": _THREE_TESTS,
        "mimo_rank": 2,
        "chunk_size": 32,
        "prerequisite_phase": "one",
    },
    "r4": {
        "selected_tests": _THREE_TESTS,
        "mimo_rank": 4,
        "chunk_size": 16,
        "prerequisite_phase": "r2",
    },
}
_PHASE = os.environ.get("CPPMEGA_H200_GATE_PHASE", "")
if _PHASE not in _PHASE_CONFIG:
    raise RuntimeError("CPPMEGA_H200_GATE_PHASE must be exactly one of: one, r2, r4")
_CONFIG = _PHASE_CONFIG[_PHASE]
_SELECTED_TESTS = tuple(_CONFIG["selected_tests"])
_MIMO_RANK = int(_CONFIG["mimo_rank"])
_CHUNK_SIZE = int(_CONFIG["chunk_size"])
_PREREQUISITE_PHASE = _CONFIG["prerequisite_phase"]


def _result_stem(phase: str) -> str:
    return (
        f"mamba3-tilelang-bind-release-"
        f"{_CANDIDATE_CPPMEGA_SHA[:8]}-{phase}-{_ATTEMPT}"
    )


_RESULT_STEM = _result_stem(_PHASE)
_RESULT_PATH = f"/results/{_RESULT_STEM}.json"
_JUNIT_PATH = f"/tmp/{_RESULT_STEM}.xml"
_DURABLE_JUNIT_PATH = f"/results/{_RESULT_STEM}-junit.xml"
_DURABLE_DEBUG_ARCHIVE_PATH = f"/results/{_RESULT_STEM}-tvm-debug.tar.gz"
_EXPECTED_TEST_COUNT = len(_SELECTED_TESTS)
_GPU_SPEC = "H200:2"
_STAGE2_PATCH_REL = (
    "upstream_prs/examples/13_tilelang_floormod_dbz/"
    "mamba3_bwd_stage2_force_nontma.patch"
)
_SOURCE_RELATIVE_PATHS = (
    "cppmega/megatron/document_isolation.py",
    "cppmega/megatron/cppmega_mamba3_tp_mixer.py",
    "cppmega/megatron/upstream_patches/apply_mamba3_gqa_bwd_patches.py",
    "cppmega/megatron/upstream_patches/apply_mamba3_stage2_force_nontma_patches.py",
    "tests/test_cppmega_mamba3_tp_mixer.py",
    _STAGE2_PATCH_REL,
)
_EXPECTED_MAMBA_INITIAL_SHA256 = {
    "mamba3_mimo_bwd.py": (
        "980dadcec29cdd318c51c1660697d54b5a7d3311d2b681b4a68b31e7d21e64b9"
    ),
    "mamba3_mimo_bwd_varlen.py": (
        "2229d2b7770ef7867ec61a6971efa7ec3e8e2fc2c47c73b42b9c3bf0fe5995a6"
    ),
}
_EXPECTED_MAMBA_AFTER_SHA256 = {
    "mamba3_mimo_bwd.py": (
        "9b1662c53c31c2387a88cd63abdbfd9da6efba8008812f84356d2419a285d423"
    ),
    "mamba3_mimo_bwd_varlen.py": _EXPECTED_MAMBA_INITIAL_SHA256[
        "mamba3_mimo_bwd_varlen.py"
    ],
}


def _sha256_path(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if modal.is_local():
    revision = subprocess.run(
        ["git", "-C", str(_LOCAL_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(_LOCAL_ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    tracked_script = subprocess.run(
        [
            "git",
            "-C",
            str(_LOCAL_ROOT),
            "ls-files",
            "--error-unmatch",
            str(_SCRIPT_PATH.relative_to(_LOCAL_ROOT)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if (
        revision.returncode != 0
        or revision.stdout.strip() != _CANDIDATE_CPPMEGA_SHA
        or status.returncode != 0
        or status.stdout
        or tracked_script.returncode != 0
    ):
        raise RuntimeError(
            "release gate requires the clean, tracked candidate checkout: "
            f"revision={revision.stdout.strip()!r}, "
            f"expected={_CANDIDATE_CPPMEGA_SHA!r}, "
            f"status={status.stdout!r}, tracked_script={tracked_script.returncode}"
        )
    _EXPECTED_SOURCE_SHA256 = {
        relative: _sha256_path(_LOCAL_ROOT / relative)
        for relative in _SOURCE_RELATIVE_PATHS
    }
    _SCRIPT_SHA256 = _sha256_path(_SCRIPT_PATH)
else:
    try:
        _EXPECTED_SOURCE_SHA256 = json.loads(
            os.environ["CPPMEGA_EXPECTED_SOURCE_SHA256_JSON"]
        )
    except (KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "remote gate lacks CPPMEGA_EXPECTED_SOURCE_SHA256_JSON"
        ) from exc
    _SCRIPT_SHA256 = _required_env(
        "CPPMEGA_RELEASE_GATE_SCRIPT_SHA256",
        r"[0-9a-f]{64}",
    )

_EXPECTED_STAGE2_PATCH_SHA256 = _EXPECTED_SOURCE_SHA256[_STAGE2_PATCH_REL]


def _image() -> modal.Image:
    wheel_downloads = " ".join(
        (
            "curl --fail --location --retry 3 --retry-delay 2 "
            "'https://github.com/DatasunriseOU/cppmega/releases/download/"
            f"{_RELEASE_TAG}/{filename}' "
            f"-o '/opt/cppmega-gate/release-wheels/{filename}'; "
            f"echo '{digest}  "
            f"/opt/cppmega-gate/release-wheels/{filename}' | sha256sum -c -;"
        )
        for filename, digest in sorted(_CANDIDATE_WHEELS.items())
    )
    image: Any = modal.Image.from_registry(
        _IMAGE_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    image = (
        image.env(
            {
                "CPPMEGA_CANDIDATE_CPPMEGA_SHA": _CANDIDATE_CPPMEGA_SHA,
                "CPPMEGA_CANDIDATE_IMAGE_DIGEST": _IMAGE_DIGEST,
                "CPPMEGA_CANDIDATE_TILELANG_SHA": _CANDIDATE_TILELANG_SHA,
                "CPPMEGA_CANDIDATE_WHEELS_JSON": json.dumps(
                    _CANDIDATE_WHEELS,
                    sort_keys=True,
                ),
                "CPPMEGA_EXPECTED_SOURCE_SHA256_JSON": json.dumps(
                    _EXPECTED_SOURCE_SHA256,
                    sort_keys=True,
                ),
                "CPPMEGA_H200_GATE_ATTEMPT": _ATTEMPT,
                "CPPMEGA_H200_GATE_PHASE": _PHASE,
                "CPPMEGA_MAMBA3_TEST_CHUNK_SIZE": str(_CHUNK_SIZE),
                "CPPMEGA_MAMBA3_TEST_MIMO_RANK": str(_MIMO_RANK),
                "CPPMEGA_MEGATRON_COMMIT": _MEGATRON_COMMIT,
                "CPPMEGA_RELEASE_GATE_SCRIPT_SHA256": _SCRIPT_SHA256,
                "CPPMEGA_RELEASE_MANIFEST_SHA256": _RELEASE_MANIFEST_SHA256,
                "CPPMEGA_RELEASE_TAG": _RELEASE_TAG,
                "CUDA_LAUNCH_BLOCKING": "1",
                "MEGATRON_LM_REPO": "/opt/megatron-lm",
                "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
                "TORCH_SHOW_CPP_STACKTRACES": "1",
                "WANDB_MODE": "disabled",
            }
        )
        .add_local_file(
            str(_SCRIPT_PATH),
            remote_path="/opt/cppmega-gate/release-gate.py",
            copy=True,
        )
        .run_commands(
            (
                "set -eux; "
                "mkdir -p /opt/cppmega-gate/release-wheels; "
                "curl --fail --location --retry 3 --retry-delay 2 "
                f"'{_RELEASE_MANIFEST_URL}' "
                "-o /opt/cppmega-gate/CANDIDATE_MANIFEST.json; "
                f"echo '{_RELEASE_MANIFEST_SHA256}  "
                "/opt/cppmega-gate/CANDIDATE_MANIFEST.json' | sha256sum -c -; "
                f"{wheel_downloads} "
                f"echo '{_EXPECTED_STAGE2_PATCH_SHA256}  "
                f"/opt/cppmega/{_STAGE2_PATCH_REL}' | sha256sum -c -"
            ),
            (
                "set -eux; "
                "cd /opt/cppmega; "
                "python -c '"
                "from cppmega.megatron.upstream_patches import "
                "apply_mamba3_gqa_bwd_patches as gqa, "
                "apply_mamba3_stage2_force_nontma_patches as stage2; "
                "assert stage2._is_stage2_patch_applied(); "
                "assert not stage2._is_stage2_patch_absent(); "
                "assert gqa._is_gqa_bwd_patch_applied(); "
                "assert not gqa._is_gqa_bwd_patch_absent()"
                "'"
            ),
        )
    )
    return image


app = modal.App(
    f"cppmega-bind-release-{_CANDIDATE_CPPMEGA_SHA[:8]}-{_PHASE}-{_ATTEMPT}"
)
results = modal.Volume.from_name("cppmega-test-results", create_if_missing=True)


@app.function(
    image=_image(),
    gpu=_GPU_SPEC,
    timeout=3600,
    volumes={"/results": results},
)
def run_release_gate() -> dict[str, Any]:
    import subprocess
    import sys
    import tarfile
    import time
    import traceback
    import uuid
    import xml.etree.ElementTree as ET
    from importlib import metadata

    import torch

    from cppmega.megatron.upstream_patches import (
        apply_mamba3_gqa_bwd_patches as gqa,
    )
    from cppmega.megatron.upstream_patches import (
        apply_mamba3_stage2_force_nontma_patches as stage2,
    )

    def sha256_path(path: pathlib.Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def canonical_sha256(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def write_receipt(receipt: dict[str, Any]) -> None:
        path = pathlib.Path(_RESULT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
        results.commit()

    def persist_junit() -> dict[str, Any]:
        source = pathlib.Path(_JUNIT_PATH)
        if not source.is_file():
            return {
                "present": False,
                "temporary_path": _JUNIT_PATH,
                "durable_path": _DURABLE_JUNIT_PATH,
            }
        destination = pathlib.Path(_DURABLE_JUNIT_PATH)
        destination.write_bytes(source.read_bytes())
        results.commit()
        return {
            "present": True,
            "temporary_path": _JUNIT_PATH,
            "durable_path": _DURABLE_JUNIT_PATH,
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_path(destination),
        }

    def persist_tvm_debug_artifacts() -> dict[str, Any]:
        root = pathlib.Path("/tmp/tvm-debug-mode-tempdirs")
        files = sorted(path for path in root.rglob("*") if path.is_file())
        inventory = [
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
            for path in files
        ]
        if not inventory:
            return {
                "present": False,
                "source_root": str(root),
                "durable_path": _DURABLE_DEBUG_ARCHIVE_PATH,
                "files": [],
            }
        destination = pathlib.Path(_DURABLE_DEBUG_ARCHIVE_PATH)
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(root, arcname=root.name, recursive=True)
        results.commit()
        return {
            "present": True,
            "source_root": str(root),
            "durable_path": _DURABLE_DEBUG_ARCHIVE_PATH,
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_path(destination),
            "files": inventory,
        }

    def source_hashes() -> dict[str, str]:
        hashes = {
            relative: sha256_path(pathlib.Path("/opt/cppmega") / relative)
            for relative in _EXPECTED_SOURCE_SHA256
        }
        if hashes != _EXPECTED_SOURCE_SHA256:
            raise RuntimeError(
                "release-image source hashes do not match candidate commit: "
                f"observed={hashes!r} expected={_EXPECTED_SOURCE_SHA256!r}"
            )
        copied_script_hash = sha256_path(
            pathlib.Path("/opt/cppmega-gate/release-gate.py")
        )
        image_script_hash = sha256_path(
            pathlib.Path("/opt/cppmega")
            / "scripts/modal_mamba3_tilelang_role_release_gate.py"
        )
        if (
            copied_script_hash != _SCRIPT_SHA256
            or image_script_hash != _SCRIPT_SHA256
        ):
            raise RuntimeError(
                "candidate-image/local gate script mismatch: "
                f"image={image_script_hash}, copied={copied_script_hash}, "
                f"expected={_SCRIPT_SHA256}"
            )
        return {
            **hashes,
            "scripts/modal_mamba3_tilelang_role_release_gate.py": image_script_hash,
            "scripts/release-gate.py": copied_script_hash,
        }

    def release_manifest() -> dict[str, Any]:
        path = pathlib.Path("/opt/cppmega-gate/CANDIDATE_MANIFEST.json")
        observed_hash = sha256_path(path)
        if observed_hash != _RELEASE_MANIFEST_SHA256:
            raise RuntimeError(
                "candidate release manifest hash mismatch: "
                f"observed={observed_hash} expected={_RELEASE_MANIFEST_SHA256}"
            )
        manifest = json.loads(path.read_text())
        expected_candidate = {
            "cppmega_ref": "candidate/tilelang-role-3dff66ef",
            "cppmega_sha": _CANDIDATE_CPPMEGA_SHA,
            "tilelang_sha": _CANDIDATE_TILELANG_SHA,
            "tvm_ffi_sha": _CANDIDATE_TVM_FFI_SHA,
            "tvm_sha": _CANDIDATE_TVM_SHA,
        }
        if manifest.get("candidate") != expected_candidate:
            raise RuntimeError(
                "candidate manifest revision mismatch: "
                f"observed={manifest.get('candidate')!r} "
                f"expected={expected_candidate!r}"
            )
        if manifest.get("candidate_wheels") != _CANDIDATE_WHEELS:
            raise RuntimeError(
                "candidate manifest wheel mismatch: "
                f"observed={manifest.get('candidate_wheels')!r} "
                f"expected={_CANDIDATE_WHEELS!r}"
            )
        expected_base = {
            "tag": f"wheels-{_BASE_SOURCE_SHA}",
            "source_sha": _BASE_SOURCE_SHA,
        }
        if manifest.get("base_release") != expected_base:
            raise RuntimeError(
                "candidate manifest base-release mismatch: "
                f"observed={manifest.get('base_release')!r} "
                f"expected={expected_base!r}"
            )
        complete_wheels = manifest.get("complete_wheel_set")
        if not isinstance(complete_wheels, dict) or any(
            complete_wheels.get(filename) != digest
            for filename, digest in _CANDIDATE_WHEELS.items()
        ):
            raise RuntimeError(
                "complete wheel set does not contain the pinned candidate pair: "
                f"{complete_wheels!r}"
            )
        github_run = manifest.get("github_run", {})
        if (
            not str(github_run.get("id", "")).isdigit()
            or not str(github_run.get("attempt", "")).isdigit()
            or int(github_run.get("attempt", 0)) < 1
            or not str(github_run.get("url", "")).endswith(
                f"/{github_run.get('id', '')}"
            )
        ):
            raise RuntimeError(f"candidate manifest workflow mismatch: {github_run!r}")
        return {
            "path": str(path),
            "url": _RELEASE_MANIFEST_URL,
            "sha256": observed_hash,
            "content": manifest,
        }

    def installed_wheel_identity() -> dict[str, Any]:
        import base64
        import csv
        import io
        import zipfile

        wheel_dir = pathlib.Path("/opt/cppmega-gate/release-wheels")
        distribution_names = {
            "tilelang": "tilelang",
            "apache_tvm_ffi": "apache-tvm-ffi",
        }
        identities: dict[str, Any] = {}
        for wheel_prefix, distribution_name in distribution_names.items():
            filenames = [
                filename
                for filename in _CANDIDATE_WHEELS
                if filename.startswith(f"{wheel_prefix}-")
            ]
            if len(filenames) != 1:
                raise RuntimeError(
                    f"expected one pinned {wheel_prefix} wheel: {filenames!r}"
                )
            filename = filenames[0]
            wheel_path = wheel_dir / filename
            wheel_sha256 = sha256_path(wheel_path)
            expected_wheel_sha256 = _CANDIDATE_WHEELS[filename]
            if wheel_sha256 != expected_wheel_sha256:
                raise RuntimeError(
                    f"downloaded release wheel mismatch for {filename}: "
                    f"observed={wheel_sha256}, expected={expected_wheel_sha256}"
                )

            distribution = metadata.distribution(distribution_name)
            verified_payload: dict[str, dict[str, Any]] = {}
            native_payload: dict[str, str] = {}
            with zipfile.ZipFile(wheel_path) as archive:
                record_names = [
                    name
                    for name in archive.namelist()
                    if name.endswith(".dist-info/RECORD")
                ]
                if len(record_names) != 1:
                    raise RuntimeError(
                        f"{filename} must contain exactly one RECORD: "
                        f"{record_names!r}"
                    )
                record_bytes = archive.read(record_names[0])
                rows = csv.reader(io.StringIO(record_bytes.decode()))
                for row in rows:
                    if len(row) < 2 or not row[1]:
                        continue
                    relative_path, encoded_hash = row[:2]
                    algorithm, separator, digest_text = encoded_hash.partition("=")
                    if separator != "=" or algorithm != "sha256" or not digest_text:
                        raise RuntimeError(
                            f"{filename} has unsupported RECORD hash: {row!r}"
                        )
                    expected_digest = base64.urlsafe_b64decode(
                        digest_text + "=" * (-len(digest_text) % 4)
                    ).hex()
                    archive_bytes = archive.read(relative_path)
                    archive_digest = hashlib.sha256(archive_bytes).hexdigest()
                    installed_path = pathlib.Path(
                        str(distribution.locate_file(relative_path))
                    ).resolve(strict=True)
                    installed_digest = sha256_path(installed_path)
                    if (
                        archive_digest != expected_digest
                        or installed_digest != expected_digest
                    ):
                        raise RuntimeError(
                            "installed payload differs from exact release wheel: "
                            f"wheel={filename}, path={relative_path}, "
                            f"record={expected_digest}, archive={archive_digest}, "
                            f"installed={installed_digest}"
                        )
                    verified_payload[relative_path] = {
                        "sha256": installed_digest,
                        "size_bytes": installed_path.stat().st_size,
                    }
                    if pathlib.PurePosixPath(relative_path).suffix in {
                        ".so",
                        ".dylib",
                        ".pyd",
                    }:
                        native_payload[relative_path] = installed_digest
            if not verified_payload or not native_payload:
                raise RuntimeError(
                    f"{filename} lacks verified payload/native identity: "
                    f"payloads={len(verified_payload)}, natives={native_payload!r}"
                )
            identities[distribution_name] = {
                "wheel_path": str(wheel_path),
                "wheel_sha256": wheel_sha256,
                "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
                "verified_payload_count": len(verified_payload),
                "verified_payload_identity_sha256": canonical_sha256(
                    verified_payload
                ),
                "native_payload": native_payload,
            }
        return identities

    def stack_provenance() -> dict[str, Any]:
        megatron_revision = subprocess.run(
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
        provenance = {
            "megatron": {
                "path": "/opt/megatron-lm",
                "expected": _MEGATRON_COMMIT,
                "commit": megatron_revision.stdout.strip(),
                "returncode": megatron_revision.returncode,
                "stderr": megatron_revision.stderr,
            },
            "versions": {
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                **versions,
            },
        }
        if (
            provenance["megatron"]["returncode"] != 0
            or provenance["megatron"]["commit"] != _MEGATRON_COMMIT
        ):
            raise RuntimeError(f"Megatron revision mismatch: {provenance!r}")
        expected_versions = {
            "tilelang": "0.1.9",
            "apache-tvm-ffi": "0.1.13.post5",
        }
        for name, expected in expected_versions.items():
            if versions[name] != expected:
                raise RuntimeError(
                    f"installed {name} version mismatch: "
                    f"observed={versions[name]!r} expected={expected!r}"
                )
        return provenance

    def mamba_overlay_state() -> dict[str, Any]:
        installed_paths = gqa._find_mamba3_bwd_files()
        installed_hashes = {
            name: sha256_path(path) for name, path in installed_paths.items()
        }
        stage2_path = stage2._find_mamba3_bwd_file()
        backup_path = stage2._backup_path(stage2_path)
        backup_hash = sha256_path(backup_path) if backup_path.is_file() else None
        stage2_applied = stage2._is_stage2_patch_applied()
        stage2_absent = stage2._is_stage2_patch_absent()
        gqa_applied = gqa._is_gqa_bwd_patch_applied()
        gqa_absent = gqa._is_gqa_bwd_patch_absent()
        if (
            installed_hashes != _EXPECTED_MAMBA_AFTER_SHA256
            or backup_hash != _EXPECTED_MAMBA_INITIAL_SHA256["mamba3_mimo_bwd.py"]
            or not stage2_applied
            or stage2_absent
            or not gqa_applied
            or gqa_absent
        ):
            raise RuntimeError(
                "image-built Mamba overlay mismatch: "
                f"installed={installed_hashes!r}, backup_hash={backup_hash!r}, "
                f"stage2_applied={stage2_applied}, "
                f"stage2_absent={stage2_absent}, gqa_applied={gqa_applied}, "
                f"gqa_absent={gqa_absent}"
            )
        return {
            "installed_paths": {
                name: str(path) for name, path in installed_paths.items()
            },
            "installed_sha256": installed_hashes,
            "stage2_backup_path": str(backup_path),
            "stage2_backup_sha256": backup_hash,
            "stage2_applied": stage2_applied,
            "stage2_absent": stage2_absent,
            "gqa_applied": gqa_applied,
            "gqa_absent": gqa_absent,
            "stage2_applied_in_image_build": True,
            "gqa_verified_only": True,
            "runtime_source_mutated": False,
        }

    def runtime_fingerprints(
        source_sha256: dict[str, str],
        mamba_state: dict[str, Any],
        installed_wheels: dict[str, Any],
    ) -> dict[str, Any]:
        def fingerprint_file(path: pathlib.Path) -> dict[str, Any]:
            try:
                resolved = path.resolve(strict=True)
                return {
                    "path": str(resolved),
                    "size_bytes": resolved.stat().st_size,
                    "sha256": sha256_path(resolved),
                }
            except OSError as exc:
                return {
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }

        native_artifacts: dict[str, list[dict[str, Any]]] = {}
        distribution_metadata: dict[str, dict[str, Any]] = {}
        for distribution_name in ("tilelang", "apache-tvm-ffi"):
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
                        pathlib.Path(str(distribution.locate_file(str(file))))
                    ),
                }
                for file in candidates
            ]
            if not native_artifacts[distribution_name] or any(
                "error" in artifact for artifact in native_artifacts[distribution_name]
            ):
                raise RuntimeError(
                    "installed distribution lacks complete native provenance: "
                    f"{distribution_name}={native_artifacts[distribution_name]!r}"
                )
            distribution_metadata[distribution_name] = {}
            for metadata_name in ("METADATA", "RECORD", "direct_url.json"):
                text = distribution.read_text(metadata_name)
                distribution_metadata[distribution_name][metadata_name] = (
                    None
                    if text is None
                    else {
                        "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "size_bytes": len(text.encode()),
                    }
                )

        module_provenance: dict[str, dict[str, Any]] = {}
        for module_name in ("tilelang", "tvm", "tvm_ffi"):
            module = __import__(module_name)
            module_path = getattr(module, "__file__", None)
            if not module_path:
                raise RuntimeError(
                    f"imported module has no file provenance: {module_name}"
                )
            module_provenance[module_name] = fingerprint_file(pathlib.Path(module_path))
            if "error" in module_provenance[module_name]:
                raise RuntimeError(
                    f"module provenance failed: "
                    f"{module_name}={module_provenance[module_name]!r}"
                )

        runtime_matches = [
            artifact
            for artifact in native_artifacts["tilelang"]
            if artifact["distribution_path"].endswith("tilelang/lib/libtvm_runtime.so")
        ]
        if len(runtime_matches) != 1:
            raise RuntimeError(
                "expected exactly one installed libtvm_runtime.so: "
                f"observed={runtime_matches!r}"
            )
        readelf = subprocess.run(
            ["readelf", "-d", runtime_matches[0]["path"]],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        requires_stub = "Shared library: [libcuda_stub.so]" in readelf.stdout
        requires_direct_cuda = re.search(
            r"Shared library: \[libcuda\.so(?:\.|\])", readelf.stdout
        )
        if readelf.returncode != 0 or not requires_stub or requires_direct_cuda:
            raise RuntimeError(
                "TileLang runtime CUDA linkage contract failed: "
                f"returncode={readelf.returncode}, requires_stub={requires_stub}, "
                f"requires_direct_cuda={bool(requires_direct_cuda)}, "
                f"stdout={readelf.stdout!r}, stderr={readelf.stderr!r}"
            )

        installed_native_identity = {
            distribution_name: {
                artifact["distribution_path"]: artifact["sha256"]
                for artifact in artifacts
            }
            for distribution_name, artifacts in native_artifacts.items()
        }
        artifact_identity = {
            "candidate_image_digest": _IMAGE_DIGEST,
            "release_tag": _RELEASE_TAG,
            "release_manifest_sha256": _RELEASE_MANIFEST_SHA256,
            "candidate_wheels": _CANDIDATE_WHEELS,
            "installed_release_wheels": installed_wheels,
            "source_sha256": source_sha256,
            "mamba_installed_sha256": mamba_state["installed_sha256"],
            "module_sha256": {
                name: fingerprint["sha256"]
                for name, fingerprint in module_provenance.items()
            },
            "native_artifacts": installed_native_identity,
            "distribution_metadata": distribution_metadata,
        }
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return {
            "candidate_wheels": _CANDIDATE_WHEELS,
            "installed_release_wheels": installed_wheels,
            "module_provenance": module_provenance,
            "native_artifacts": native_artifacts,
            "distribution_metadata": distribution_metadata,
            "artifact_identity": artifact_identity,
            "artifact_identity_sha256": canonical_sha256(artifact_identity),
            "tilelang_runtime_readelf": {
                "path": runtime_matches[0]["path"],
                "returncode": readelf.returncode,
                "stdout": readelf.stdout,
                "stderr": readelf.stderr,
                "requires_libcuda_stub": requires_stub,
                "requires_direct_libcuda": bool(requires_direct_cuda),
            },
            "nvidia_smi": {
                "returncode": nvidia_smi.returncode,
                "stdout": nvidia_smi.stdout,
                "stderr": nvidia_smi.stderr,
            },
            "python": {
                "executable": sys.executable,
                "version": sys.version,
            },
        }

    def verify_prerequisite() -> dict[str, Any] | None:
        if _PREREQUISITE_PHASE is None:
            return None

        def verify_phase_artifact(phase: str) -> dict[str, Any]:
            config = _PHASE_CONFIG[phase]
            expected_tests = list(config["selected_tests"])
            expected_test_count = len(expected_tests)
            stem = _result_stem(phase)
            path = pathlib.Path(f"/results/{stem}.json")
            junit_path = pathlib.Path(f"/results/{stem}-junit.xml")
            if not path.is_file() or not junit_path.is_file():
                raise RuntimeError(
                    "required prior phase receipt/JUnit is missing: "
                    f"phase={phase}, receipt={path}, junit={junit_path}"
                )
            prior = json.loads(path.read_text())
            expected = {
                "status": "passed",
                "gate_kind": "release-image",
                "phase": phase,
                "cppmega_sha": _CANDIDATE_CPPMEGA_SHA,
                "tilelang_sha": _CANDIDATE_TILELANG_SHA,
                "release_tag": _RELEASE_TAG,
                "manifest_sha256": _RELEASE_MANIFEST_SHA256,
                "candidate_wheels": _CANDIDATE_WHEELS,
                "image_digest": _IMAGE_DIGEST,
                "script_sha256": _SCRIPT_SHA256,
                "selected_tests": expected_tests,
                "expected_test_count": expected_test_count,
                "mimo_rank": int(config["mimo_rank"]),
                "chunk_size": int(config["chunk_size"]),
                "junit": {
                    "present": True,
                    "tests": expected_test_count,
                    "failures": 0,
                    "errors": 0,
                    "skipped": 0,
                },
            }
            observed = {
                "status": prior.get("status"),
                "gate_kind": prior.get("gate_kind"),
                "phase": prior.get("phase"),
                "cppmega_sha": prior.get("candidate", {}).get("cppmega_sha"),
                "tilelang_sha": prior.get("candidate", {}).get("tilelang_sha"),
                "release_tag": prior.get("release", {}).get("tag"),
                "manifest_sha256": prior.get("release", {}).get(
                    "manifest_sha256"
                ),
                "candidate_wheels": prior.get("release", {}).get(
                    "candidate_wheels"
                ),
                "image_digest": prior.get("image", {}).get("digest"),
                "script_sha256": prior.get("script_sha256"),
                "selected_tests": prior.get("selected_tests"),
                "expected_test_count": prior.get("expected_test_count"),
                "mimo_rank": prior.get("test_factorization", {}).get(
                    "R_mimo_rank"
                ),
                "chunk_size": prior.get("test_factorization", {}).get(
                    "chunk_size"
                ),
                "junit": prior.get("junit"),
            }
            if observed != expected:
                raise RuntimeError(
                    "prior phase receipt does not match this exact release gate: "
                    f"phase={phase}, observed={observed!r}, expected={expected!r}"
                )

            junit_artifact = prior.get("junit_artifact", {})
            actual_junit_sha = sha256_path(junit_path)
            if (
                junit_artifact.get("present") is not True
                or junit_artifact.get("durable_path") != str(junit_path)
                or junit_artifact.get("sha256") != actual_junit_sha
            ):
                raise RuntimeError(
                    "prior phase durable JUnit identity mismatch: "
                    f"phase={phase}, artifact={junit_artifact!r}, "
                    f"actual_sha256={actual_junit_sha}"
                )
            prior_identity = prior.get("runtime_fingerprints", {}).get(
                "artifact_identity"
            )
            prior_identity_sha = prior.get("runtime_fingerprints", {}).get(
                "artifact_identity_sha256"
            )
            if (
                not isinstance(prior_identity, dict)
                or canonical_sha256(prior_identity) != prior_identity_sha
            ):
                raise RuntimeError(
                    f"prior phase {phase} runtime artifact identity is corrupt"
                )
            if prior.get("source_sha256_before_test") != prior.get(
                "source_sha256_after_test"
            ) or prior.get("mamba_overlay_before_test") != prior.get(
                "mamba_overlay_after_test"
            ):
                raise RuntimeError(
                    f"prior phase {phase} did not prove read-only runtime identity"
                )

            prerequisite_phase = config["prerequisite_phase"]
            actual_prerequisite = (
                None
                if prerequisite_phase is None
                else verify_phase_artifact(str(prerequisite_phase))
            )
            if prior.get("prerequisite") != actual_prerequisite:
                raise RuntimeError(
                    "prior phase embedded prerequisite differs from durable "
                    f"artifact chain: phase={phase}, "
                    f"embedded={prior.get('prerequisite')!r}, "
                    f"actual={actual_prerequisite!r}"
                )
            return {
                "path": str(path),
                "sha256": sha256_path(path),
                "junit_path": str(junit_path),
                "junit_sha256": actual_junit_sha,
                "run_id": prior.get("run_id"),
                "phase": phase,
                "status": "passed",
                "cppmega_sha": _CANDIDATE_CPPMEGA_SHA,
                "tilelang_sha": _CANDIDATE_TILELANG_SHA,
                "release_tag": _RELEASE_TAG,
                "manifest_sha256": _RELEASE_MANIFEST_SHA256,
                "candidate_wheels": _CANDIDATE_WHEELS,
                "image_digest": _IMAGE_DIGEST,
                "script_sha256": _SCRIPT_SHA256,
                "selected_tests": expected_tests,
                "expected_test_count": expected_test_count,
                "mimo_rank": int(config["mimo_rank"]),
                "chunk_size": int(config["chunk_size"]),
                "junit": prior.get("junit"),
                "artifact_identity": prior_identity,
                "artifact_identity_sha256": prior_identity_sha,
                "prerequisite": actual_prerequisite,
            }

        return verify_phase_artifact(str(_PREREQUISITE_PHASE))

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
                child for child in root if child.tag.rsplit("}", 1)[-1] == "testsuite"
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
    existing_outputs = [
        path
        for path in (
            pathlib.Path(_RESULT_PATH),
            pathlib.Path(_DURABLE_JUNIT_PATH),
            pathlib.Path(_DURABLE_DEBUG_ARCHIVE_PATH),
        )
        if path.exists()
    ]
    if existing_outputs:
        raise RuntimeError(
            "refusing to overwrite an existing exact gate attempt; select a "
            f"new CPPMEGA_H200_GATE_ATTEMPT: {existing_outputs!r}"
        )
    receipt: dict[str, Any] = {
        "schema_version": 6,
        "run_id": str(uuid.uuid4()),
        "attempt": f"release-{_CANDIDATE_CPPMEGA_SHA[:8]}-{_PHASE}-{_ATTEMPT}",
        "gate_kind": "release-image",
        "phase": _PHASE,
        "modal": {
            "function_call_id": modal.current_function_call_id(),
            "input_id": modal.current_input_id(),
            "task_id": os.environ.get("MODAL_TASK_ID"),
        },
        "status": "running",
        "candidate": {
            "cppmega_sha": _CANDIDATE_CPPMEGA_SHA,
            "tilelang_sha": _CANDIDATE_TILELANG_SHA,
            "tvm_sha": _CANDIDATE_TVM_SHA,
            "tvm_ffi_sha": _CANDIDATE_TVM_FFI_SHA,
            "base_source_sha": _BASE_SOURCE_SHA,
        },
        "release": {
            "tag": _RELEASE_TAG,
            "manifest_url": _RELEASE_MANIFEST_URL,
            "manifest_sha256": _RELEASE_MANIFEST_SHA256,
            "candidate_wheels": _CANDIDATE_WHEELS,
        },
        "image": {
            "ref": _IMAGE_REF,
            "digest": _IMAGE_DIGEST,
        },
        "gpu_spec": _GPU_SPEC,
        "selected_tests": list(_SELECTED_TESTS),
        "expected_test_count": _EXPECTED_TEST_COUNT,
        "started_unix": started,
        "script_sha256": _SCRIPT_SHA256,
        "test_factorization": {
            "batch": 2,
            "seq_len": 128,
            "d_model": 256,
            "nheads": 8,
            "ngroups": 4,
            "R_mimo_rank": _MIMO_RANK,
            "chunk_size": _CHUNK_SIZE,
            "N_state_dim": 32,
            "P_head_dim": 32,
        },
        "gate_contract": {
            "ordered_phase": _PHASE,
            "prerequisite_phase": _PREREQUISITE_PHASE,
            "exact_two_h200": True,
            "exact_junit_no_skips": True,
            "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        },
        "mutation_contract": {
            "base_release_image_mutated": False,
            "immutable_image_stage2_applied": True,
            "modal_derived_image_stage2_mutated": False,
            "checked_in_source_mutated": False,
            "test_source_mutated": False,
            "runtime_source_mutated": False,
            "runtime_shim_added": False,
            "gqa_verified_only": True,
            "stage2_image_build_applier": (
                "cppmega.megatron.upstream_patches."
                "apply_mamba3_stage2_force_nontma_patches"
            ),
        },
    }
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
                f"runtime mutation gates must be unset: {mutation_env!r}"
            )
        receipt["runtime_mutation_environment"] = mutation_env
        receipt["prerequisite"] = verify_prerequisite()
        receipt["release_manifest"] = release_manifest()
        receipt["installed_release_wheels"] = installed_wheel_identity()
        receipt["stack_provenance"] = stack_provenance()
        receipt["source_sha256_before_test"] = source_hashes()
        receipt["mamba_overlay_before_test"] = mamba_overlay_state()

        import tilelang  # noqa: F401  # Registers the wheel's vendored TVM.
        import tvm
        from tvm.s_tir.analysis import is_pure_function

        attr_body = tvm.tirx.AttrStmt(
            None,
            "threadblock_swizzle_pattern",
            0,
            tvm.tirx.Evaluate(0),
        )
        attr_node_is_none = attr_body.node is None
        attr_primfunc_is_pure = is_pure_function(tvm.tirx.PrimFunc([], attr_body))
        if not attr_node_is_none or not attr_primfunc_is_pure:
            raise RuntimeError(
                "TileLang AttrStmt(None) purity smoke failed: "
                f"node_is_none={attr_node_is_none}, "
                f"primfunc_is_pure={attr_primfunc_is_pure}"
            )
        receipt["tilelang_import_smoke"] = {
            "attrstmt_node_is_none": attr_node_is_none,
            "attrstmt_primfunc_is_pure": attr_primfunc_is_pure,
        }
        receipt["runtime_fingerprints"] = runtime_fingerprints(
            receipt["source_sha256_before_test"],
            receipt["mamba_overlay_before_test"],
            receipt["installed_release_wheels"],
        )
        if receipt["prerequisite"] is not None:
            prior_identity = receipt["prerequisite"]["artifact_identity"]
            current_identity = receipt["runtime_fingerprints"]["artifact_identity"]
            if prior_identity != current_identity:
                raise RuntimeError(
                    "release runtime artifact identity changed between ordered "
                    f"phases: prior={prior_identity!r}, current={current_identity!r}"
                )

        device_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if len(device_names) != 2 or any(
            "H200" not in name.upper() for name in device_names
        ):
            raise RuntimeError(
                f"gate requires exactly two H200 GPUs: observed={device_names!r}"
            )
        gpu_health = []
        for index, device_name in enumerate(device_names):
            with torch.cuda.device(index):
                values = torch.arange(
                    256,
                    dtype=torch.float32,
                    device=torch.device("cuda", index),
                )
                observed_sum = float(values.sum().item())
                torch.cuda.synchronize(index)
            if observed_sum != 32640.0:
                raise RuntimeError(
                    f"H200 arithmetic preflight failed on device {index}: "
                    f"sum={observed_sum}"
                )
            gpu_health.append(
                {
                    "index": index,
                    "name": device_name,
                    "arange_sum": observed_sum,
                    "synchronized": True,
                }
            )
        receipt["gpu_health_before_test"] = gpu_health
        process = subprocess.run(
            command,
            cwd="/opt/cppmega",
            capture_output=True,
            text=True,
            check=False,
            timeout=3480,
        )
        receipt.update(
            {
                "returncode": process.returncode,
                "command": command,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "devices": device_names,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        )
        receipt["source_sha256_after_test"] = source_hashes()
        receipt["mamba_overlay_after_test"] = mamba_overlay_state()
        if (
            receipt["source_sha256_after_test"] != receipt["source_sha256_before_test"]
            or receipt["mamba_overlay_after_test"]
            != receipt["mamba_overlay_before_test"]
        ):
            raise RuntimeError("pytest changed release source or Mamba overlay state")

        receipt["tvm_debug_artifact"] = persist_tvm_debug_artifacts()
        receipt["junit_artifact"] = persist_junit()
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
        if isinstance(exc, subprocess.TimeoutExpired):
            receipt.update(
                {
                    "command": command,
                    "timeout_seconds": exc.timeout,
                    "stdout": (
                        exc.stdout.decode(errors="replace")
                        if isinstance(exc.stdout, bytes)
                        else exc.stdout
                    ),
                    "stderr": (
                        exc.stderr.decode(errors="replace")
                        if isinstance(exc.stderr, bytes)
                        else exc.stderr
                    ),
                }
            )
        if (
            "source_sha256_before_test" in receipt
            and "source_sha256_after_test" not in receipt
        ):
            try:
                receipt["source_sha256_after_test"] = source_hashes()
            except (OSError, RuntimeError) as post_exc:
                receipt["source_after_test_error"] = (
                    f"{type(post_exc).__name__}: {post_exc}"
                )
        if (
            "mamba_overlay_before_test" in receipt
            and "mamba_overlay_after_test" not in receipt
        ):
            try:
                receipt["mamba_overlay_after_test"] = mamba_overlay_state()
            except (OSError, RuntimeError) as post_exc:
                receipt["mamba_after_test_error"] = (
                    f"{type(post_exc).__name__}: {post_exc}"
                )
        if "tvm_debug_artifact" not in receipt:
            receipt["tvm_debug_artifact"] = persist_tvm_debug_artifacts()
        if "junit_artifact" not in receipt:
            receipt["junit_artifact"] = persist_junit()
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
    print(json.dumps(run_release_gate.remote(), indent=2))
