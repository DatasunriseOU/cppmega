import ast
import base64
import hashlib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from cppmega.megatron.release_gate_integrity import (
    is_runtime_source_path,
    require_module_payload_bindings,
    sha256_path,
    validate_complete_wheel_set,
    validate_exact_junit,
    validate_mamba_overlay_state,
    validate_source_manifest,
    verify_wheel_record_payloads,
)

_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = _ROOT / "scripts/modal_mamba3_tilelang_role_release_gate.py"
_CANDIDATE_TILELANG_SHA = "de8bb88cc382b0e78bc804244f79c4be8cc9e75f"


def test_candidate_tilelang_sha_is_bound_across_release_surfaces():
    workflow = (_ROOT / ".github/workflows/build-wheels.yml").read_text()
    stack = (_ROOT / "STACK.lock").read_text()
    harness = _HARNESS.read_text()
    changelog = (_ROOT / "docs/changelog.md").read_text()

    assert workflow.count(_CANDIDATE_TILELANG_SHA) == 1
    assert f"ref: {_CANDIDATE_TILELANG_SHA}" in stack
    assert f'_CANDIDATE_TILELANG_SHA = "{_CANDIDATE_TILELANG_SHA}"' in harness
    assert f"`{_CANDIDATE_TILELANG_SHA}`" in changelog


def _record_hash(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return f"sha256={digest.rstrip(b'=').decode()}"


def _synthetic_wheel(
    tmp_path: Path,
    *,
    blank_payload_hash: bool = False,
    wrong_payload_hash: bool = False,
) -> tuple[Path, Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    package_path = "demo/__init__.py"
    metadata_path = "demo-1.0.dist-info/METADATA"
    record_path = "demo-1.0.dist-info/RECORD"
    package_bytes = b"VALUE = 'release'\n"
    metadata_bytes = b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n"
    if blank_payload_hash:
        package_row = f"{package_path},,"
    else:
        package_hash = _record_hash(b"wrong" if wrong_payload_hash else package_bytes)
        package_row = f"{package_path},{package_hash},{len(package_bytes)}"
    record_bytes = (
        f"{package_row}\n"
        f"{metadata_path},{_record_hash(metadata_bytes)},"
        f"{len(metadata_bytes)}\n"
        f"{record_path},,\n"
    ).encode()
    wheel_path = tmp_path / "demo-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr(package_path, package_bytes)
        archive.writestr(metadata_path, metadata_bytes)
        archive.writestr(record_path, record_bytes)
    installed_root = tmp_path / "site-packages"
    installed_package = installed_root / package_path
    installed_package.parent.mkdir(parents=True)
    installed_package.write_bytes(package_bytes)
    installed_metadata = installed_root / metadata_path
    installed_metadata.parent.mkdir(parents=True)
    installed_metadata.write_bytes(metadata_bytes)
    return wheel_path, installed_root, package_path


def test_source_manifest_rejects_stale_tracked_bytes(tmp_path: Path):
    source = tmp_path / "cppmega" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    expected = {"cppmega/runtime.py": sha256_path(source)}

    assert validate_source_manifest(tmp_path, expected)["file_count"] == 1

    source.write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="mismatched=.*cppmega/runtime.py"):
        validate_source_manifest(tmp_path, expected)


def test_source_manifest_rejects_untracked_import_shadow(tmp_path: Path):
    tracked = tmp_path / "cppmega" / "runtime.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("VALUE = 1\n")
    expected = {"cppmega/runtime.py": sha256_path(tracked)}
    shadow = tmp_path / "tilelang.py"
    shadow.write_text("raise RuntimeError('shadowed')\n")

    with pytest.raises(RuntimeError, match="untracked_shadowable=.*tilelang.py"):
        validate_source_manifest(tmp_path, expected)


def test_generated_outputs_are_excluded_from_runtime_image_manifest():
    assert not is_runtime_source_path("outputs/evals/generated.py")
    assert is_runtime_source_path("cppmega/runtime.py")
    assert "outputs/" in (_ROOT / ".dockerignore").read_text().splitlines()


def test_module_binding_rejects_source_shadow_and_accepts_record_payload(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    source_module = source_root / "tilelang.py"
    source_module.parent.mkdir()
    source_module.write_text("VERSION = 'shadow'\n")
    source_provenance = {
        "tilelang": {
            "path": str(source_module),
            "sha256": sha256_path(source_module),
        }
    }

    with pytest.raises(RuntimeError, match="shadows release wheels"):
        require_module_payload_bindings(
            source_provenance,
            {},
            forbidden_root=source_root,
        )

    site_module = tmp_path / "site-packages" / "tilelang" / "__init__.py"
    site_module.parent.mkdir(parents=True)
    site_module.write_text("VERSION = 'release'\n")
    digest = sha256_path(site_module)
    provenance = {
        "tilelang": {
            "path": str(site_module),
            "sha256": digest,
        }
    }
    verified = {
        str(site_module.resolve()): {
            "wheel": "tilelang-0.1.9-cp38-abi3-linux_x86_64.whl",
            "relative_path": "tilelang/__init__.py",
            "sha256": digest,
        }
    }

    bound = require_module_payload_bindings(
        provenance,
        verified,
        forbidden_root=source_root,
    )

    assert bound["tilelang"]["release_wheel"].startswith("tilelang-0.1.9-")
    assert bound["tilelang"]["release_wheel_path"] == "tilelang/__init__.py"


def test_complete_wheel_set_rejects_mismatched_release_manifest():
    expected = {
        "tilelang-0.1.9-cp38-abi3-linux_x86_64.whl": "a" * 64,
        "apache_tvm_ffi-0.1.13.post5-cp313-cp313-linux_x86_64.whl": ("b" * 64),
    }
    observed = dict(expected)
    observed["tilelang-0.1.9-cp38-abi3-linux_x86_64.whl"] = "c" * 64

    with pytest.raises(RuntimeError, match="complete_wheel_set mismatch"):
        validate_complete_wheel_set(
            observed,
            expected,
            ("tilelang", "apache_tvm_ffi"),
        )


def test_release_gate_accepts_exact_new_mamba_overlay_and_rejects_old_hash():
    initial_bwd = "980dadcec29cdd318c51c1660697d54b5a7d3311d2b681b4a68b31e7d21e64b9"
    old_bwd = "9b1662c53c31c2387a88cd63abdbfd9da6efba8008812f84356d2419a285d423"
    new_bwd = "51dab809a47bd33a9b610725599dff956187edd70cae1518618f1ef31115d320"
    varlen = "2229d2b7770ef7867ec61a6971efa7ec3e8e2fc2c47c73b42b9c3bf0fe5995a6"
    expected = {
        "mamba3_mimo_bwd.py": new_bwd,
        "mamba3_mimo_bwd_varlen.py": varlen,
    }
    exact_state = {
        "backup_hash": initial_bwd,
        "expected_backup_hash": initial_bwd,
        "stage2_applied": True,
        "stage2_absent": False,
        "gqa_applied": True,
        "gqa_absent": False,
    }

    harness = _HARNESS.read_text()
    assert new_bwd in harness
    assert old_bwd not in harness
    validate_mamba_overlay_state(expected, expected, **exact_state)

    old_installed = dict(expected)
    old_installed["mamba3_mimo_bwd.py"] = old_bwd
    with pytest.raises(RuntimeError, match="image-built Mamba overlay mismatch"):
        validate_mamba_overlay_state(old_installed, expected, **exact_state)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backup_hash", "0" * 64),
        ("stage2_applied", False),
        ("stage2_absent", True),
        ("gqa_applied", False),
        ("gqa_absent", True),
    ],
)
def test_release_gate_rejects_non_exact_mamba_overlay_state(field, value):
    expected = {
        "mamba3_mimo_bwd.py": (
            "51dab809a47bd33a9b610725599dff956187edd70cae1518618f1ef31115d320"
        ),
        "mamba3_mimo_bwd_varlen.py": (
            "2229d2b7770ef7867ec61a6971efa7ec3e8e2fc2c47c73b42b9c3bf0fe5995a6"
        ),
    }
    initial_bwd = "980dadcec29cdd318c51c1660697d54b5a7d3311d2b681b4a68b31e7d21e64b9"
    state: dict[str, Any] = {
        "backup_hash": initial_bwd,
        "expected_backup_hash": initial_bwd,
        "stage2_applied": True,
        "stage2_absent": False,
        "gqa_applied": True,
        "gqa_absent": False,
    }
    state[field] = value

    with pytest.raises(RuntimeError, match="image-built Mamba overlay mismatch"):
        validate_mamba_overlay_state(expected, expected, **state)


def test_exact_prerequisite_junit_rejects_failure_and_digest_drift(
    tmp_path: Path,
):
    junit = tmp_path / "prior-junit.xml"
    junit.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>')
    exact_digest = sha256_path(junit)

    validated = validate_exact_junit(
        junit,
        expected_test_count=1,
        expected_sha256=exact_digest,
    )
    assert validated["counts"]["tests"] == 1

    junit.write_text('<testsuite tests="1" failures="1" errors="0" skipped="0"/>')
    with pytest.raises(RuntimeError, match="durable JUnit is not exact"):
        validate_exact_junit(
            junit,
            expected_test_count=1,
            expected_sha256=sha256_path(junit),
        )

    junit.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>')
    with pytest.raises(RuntimeError, match="durable JUnit digest mismatch"):
        validate_exact_junit(
            junit,
            expected_test_count=1,
            expected_sha256="0" * 64,
        )


def test_wheel_record_rejects_unhashed_and_mismatched_payloads(
    tmp_path: Path,
):
    unhashed_wheel, installed_root, _ = _synthetic_wheel(
        tmp_path / "unhashed",
        blank_payload_hash=True,
    )
    with pytest.raises(RuntimeError, match="unhashed payload RECORD row"):
        verify_wheel_record_payloads(
            unhashed_wheel,
            expected_wheel_sha256=sha256_path(unhashed_wheel),
            expected_distribution_name="demo",
            installed_root=installed_root,
            verified_absolute_paths={},
        )

    mismatched_wheel, mismatched_root, _ = _synthetic_wheel(
        tmp_path / "mismatched",
        wrong_payload_hash=True,
    )
    with pytest.raises(
        RuntimeError,
        match="wheel archive differs from its exact RECORD",
    ):
        verify_wheel_record_payloads(
            mismatched_wheel,
            expected_wheel_sha256=sha256_path(mismatched_wheel),
            expected_distribution_name="demo",
            installed_root=mismatched_root,
            verified_absolute_paths={},
        )


def test_wheel_record_allows_stage2_transform_and_rejects_duplicate_path(
    tmp_path: Path,
):
    wheel, installed_root, package_path = _synthetic_wheel(tmp_path / "transformed")
    installed_package = installed_root / package_path
    backup = installed_package.with_suffix(".py.bak")
    backup.write_bytes(installed_package.read_bytes())
    installed_package.write_text("VALUE = 'stage2-patched'\n")
    transformations = {
        package_path: {
            "kind": "cppmega-stage2-build-time-patch",
            "backup_path": str(backup),
            "installed_sha256": sha256_path(installed_package),
        }
    }
    verified: dict[str, dict[str, str]] = {}

    identity = verify_wheel_record_payloads(
        wheel,
        expected_wheel_sha256=sha256_path(wheel),
        expected_distribution_name="demo",
        installed_root=installed_root,
        verified_absolute_paths=verified,
        allowed_transformations=transformations,
    )

    assert identity["verified_payload_count"] == 2
    assert verified[str(installed_package.resolve())]["wheel"] == wheel.name
    with pytest.raises(RuntimeError, match="same installed payload"):
        verify_wheel_record_payloads(
            wheel,
            expected_wheel_sha256=sha256_path(wheel),
            expected_distribution_name="demo",
            installed_root=installed_root,
            verified_absolute_paths=verified,
            allowed_transformations=transformations,
        )


def test_image_build_binds_full_source_receipt_to_docker_args():
    workflow = (_ROOT / ".github/workflows/build-image.yml").read_text()
    dockerfile = (_ROOT / "docker/Dockerfile").read_text()

    for name in (
        "CPPMEGA_SOURCE_SHA",
        "CPPMEGA_SOURCE_TREE",
        "CPPMEGA_SOURCE_MANIFEST_SHA256",
        "CPPMEGA_SOURCE_MANIFEST_FILE_COUNT",
    ):
        assert f"{name}=${{{{ steps.source.outputs." in workflow
        assert f"ARG {name}" in dockerfile
    assert "> /opt/cppmega-image-source.json" in dockerfile
    assert "COPY . /opt/cppmega" in dockerfile
    assert "CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1" in dockerfile
    assert "MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1" in dockerfile
    assert "assert stage2._is_stage2_patch_applied()" in dockerfile
    assert "assert not stage2._is_stage2_patch_absent()" in dockerfile


def test_release_gate_keeps_ordered_read_only_h200_contract():
    harness = _HARNESS.read_text()

    for required_env in (
        "CPPMEGA_CANDIDATE_CPPMEGA_SHA",
        "CPPMEGA_CANDIDATE_IMAGE_DIGEST",
        "CPPMEGA_RELEASE_MANIFEST_SHA256",
        "CPPMEGA_COMPLETE_WHEELS_JSON",
        "CPPMEGA_H200_GATE_PHASE",
    ):
        assert required_env in harness
    assert '"prerequisite_phase": "one"' in harness
    assert '"prerequisite_phase": "r2"' in harness
    assert "verify_phase_artifact(str(prerequisite_phase))" in harness
    assert "validated_junit = validate_exact_junit(" in harness
    assert 'junit_artifact.get("durable_path")' in harness
    assert 'prior.get("prerequisite") != actual_prerequisite' in harness
    assert '"modal_derived_image_stage2_mutated": False' in harness
    assert "refusing to overwrite an existing exact gate attempt" in harness
    assert "write_receipt(receipt, progress_path)" in harness
    assert harness.count("write_receipt(receipt, progress_path, _RESULT_PATH)") == 2
    assert "_DURABLE_JUNIT_PATH" not in harness
    assert 'receipt["gpu_health_before_test"]' in harness
    assert "isinstance(exc, subprocess.TimeoutExpired)" in harness
    assert ".add_local_file(\n            str(_LOCAL_STAGE2_PATCH)" not in harness


def test_release_gate_remote_hydration_never_assembles_local_image():
    tree = ast.parse(_HARNESS.read_text())
    run_release_gate = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_release_gate"
    )
    function_decorator = next(
        decorator
        for decorator in run_release_gate.decorator_list
        if isinstance(decorator, ast.Call)
        and ast.unparse(decorator.func) == "app.function"
    )
    image_keyword = next(
        keyword for keyword in function_decorator.keywords if keyword.arg == "image"
    )
    memory_keyword = next(
        keyword for keyword in function_decorator.keywords if keyword.arg == "memory"
    )

    assert ast.unparse(image_keyword.value) == "_image() if modal.is_local() else None"
    assert ast.literal_eval(memory_keyword.value) == 131_072
