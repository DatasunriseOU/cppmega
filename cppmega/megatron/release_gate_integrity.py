"""Pure integrity checks shared by the immutable-image H200 release gate."""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any

_IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "dist",
    "venv",
}
_IGNORED_FILE_SUFFIXES = (".bak", ".log", ".pyc")
_IGNORED_TOP_LEVEL_NAMES = {"outputs", "wheels"}
_SHADOWABLE_FILE_SUFFIXES = (".dylib", ".py", ".pyd", ".so")
_RUNTIME_ONLY_EXCLUSIONS = {
    ".dockerignore",
    "docker/Dockerfile",
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_runtime_source_path(relative_path: str) -> bool:
    path = Path(relative_path)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and bool(path.parts)
        and not any(
            part in _IGNORED_DIRECTORY_NAMES or part.endswith(".egg-info")
            for part in path.parts[:-1]
        )
        and not path.name.endswith(_IGNORED_FILE_SUFFIXES)
        and relative_path not in _RUNTIME_ONLY_EXCLUSIONS
        and path.parts[0] not in _IGNORED_TOP_LEVEL_NAMES
    )


def untracked_shadowable_files(
    root: Path,
    tracked_paths: set[str],
) -> tuple[str, ...]:
    extras: list[str] = []
    for current_root, directory_names, filenames in os.walk(root):
        current = Path(current_root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in _IGNORED_DIRECTORY_NAMES
            and not name.endswith(".egg-info")
            and (current != root or name not in _IGNORED_TOP_LEVEL_NAMES)
        )
        for filename in sorted(filenames):
            path = current / filename
            relative = path.relative_to(root).as_posix()
            if (
                path.name.endswith(_SHADOWABLE_FILE_SUFFIXES)
                and relative not in tracked_paths
            ):
                extras.append(relative)
    return tuple(extras)


def validate_source_manifest(
    root: Path,
    expected: dict[str, str],
) -> dict[str, Any]:
    expected_paths = set(expected)
    observed: dict[str, str] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected_digest in expected.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        observed_digest = sha256_path(path)
        observed[relative] = observed_digest
        if observed_digest != expected_digest:
            mismatched.append(relative)
    shadowable = untracked_shadowable_files(root, expected_paths)
    if missing or mismatched or shadowable:
        raise RuntimeError(
            "runtime source differs from exact candidate tree: "
            f"missing={missing!r}, mismatched={mismatched!r}, "
            f"untracked_shadowable={shadowable!r}"
        )
    return {
        "file_count": len(observed),
        "manifest_sha256": canonical_sha256(observed),
    }


def validate_complete_wheel_set(
    observed: Any,
    expected: dict[str, str],
    required_prefixes: tuple[str, ...],
) -> dict[str, str]:
    if observed != expected:
        raise RuntimeError(
            "release complete_wheel_set mismatch: "
            f"observed={observed!r}, expected={expected!r}"
        )
    if len(expected) != len(required_prefixes):
        raise RuntimeError(
            "complete wheel inventory cardinality mismatch: "
            f"wheels={len(expected)}, required={len(required_prefixes)}"
        )
    for filename, digest in expected.items():
        if (
            re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename) is None
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise RuntimeError(f"invalid complete wheel identity: {filename}={digest}")
    for prefix in required_prefixes:
        matches = [
            filename for filename in expected if filename.startswith(f"{prefix}-")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one complete wheel for {prefix}: {matches!r}"
            )
    return expected


def validate_mamba_overlay_state(
    installed_hashes: dict[str, str],
    expected_installed_hashes: dict[str, str],
    *,
    backup_hash: str | None,
    expected_backup_hash: str,
    stage2_applied: bool,
    stage2_absent: bool,
    gqa_applied: bool,
    gqa_absent: bool,
) -> None:
    if (
        installed_hashes != expected_installed_hashes
        or backup_hash != expected_backup_hash
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


def junit_counts(path: Path) -> dict[str, int | bool]:
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
        raise RuntimeError(f"{path} contains no testsuite elements")
    counts: dict[str, int | bool] = {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    counts["present"] = True
    return counts


def validate_exact_junit(
    path: Path,
    *,
    expected_test_count: int,
    expected_sha256: str,
) -> dict[str, Any]:
    counts = junit_counts(path)
    expected_counts: dict[str, int | bool] = {
        "present": True,
        "tests": expected_test_count,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    if counts != expected_counts:
        raise RuntimeError(
            "durable JUnit is not exact: "
            f"path={path}, observed={counts!r}, expected={expected_counts!r}"
        )
    observed_sha256 = sha256_path(path)
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or observed_sha256 != expected_sha256
    ):
        raise RuntimeError(
            "durable JUnit digest mismatch: "
            f"path={path}, observed={observed_sha256}, "
            f"expected={expected_sha256}"
        )
    return {
        "path": str(path),
        "sha256": observed_sha256,
        "counts": counts,
    }


def wheel_distribution_name(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(
                f"{wheel_path.name} must contain one METADATA: {metadata_names!r}"
            )
        wheel_metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
    distribution_name = str(wheel_metadata["Name"] or "")
    if not distribution_name:
        raise RuntimeError(f"{wheel_path.name} METADATA does not define Name")
    return distribution_name


def _record_sha256(encoded_hash: str, *, wheel: str, path: str) -> str:
    algorithm, separator, digest_text = encoded_hash.partition("=")
    if separator != "=" or algorithm != "sha256" or not digest_text:
        raise RuntimeError(
            f"{wheel} has unsupported RECORD hash: path={path}, hash={encoded_hash!r}"
        )
    try:
        digest = base64.urlsafe_b64decode(digest_text + "=" * (-len(digest_text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(
            f"{wheel} has invalid RECORD digest: path={path}, hash={encoded_hash!r}"
        ) from exc
    if len(digest) != hashlib.sha256().digest_size:
        raise RuntimeError(
            f"{wheel} has non-SHA256 RECORD digest: path={path}, hash={encoded_hash!r}"
        )
    return digest.hex()


def verify_wheel_record_payloads(
    wheel_path: Path,
    *,
    expected_wheel_sha256: str,
    expected_distribution_name: str,
    installed_root: Path,
    verified_absolute_paths: dict[str, dict[str, str]],
    allowed_transformations: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    wheel_name = wheel_path.name
    wheel_sha256 = sha256_path(wheel_path)
    if wheel_sha256 != expected_wheel_sha256:
        raise RuntimeError(
            f"downloaded release wheel mismatch for {wheel_name}: "
            f"observed={wheel_sha256}, expected={expected_wheel_sha256}"
        )
    transformations = allowed_transformations or {}
    used_transformations: set[str] = set()
    verified_payload: dict[str, dict[str, Any]] = {}
    native_payload: dict[str, str] = {}
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        record_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
        ]
        if len(metadata_names) != 1 or len(record_names) != 1:
            raise RuntimeError(
                f"{wheel_name} must contain one METADATA and RECORD: "
                f"metadata={metadata_names!r}, records={record_names!r}"
            )
        wheel_metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        distribution_name = str(wheel_metadata["Name"] or "")
        if distribution_name != expected_distribution_name:
            raise RuntimeError(
                f"{wheel_name} distribution mismatch: "
                f"observed={distribution_name!r}, "
                f"expected={expected_distribution_name!r}"
            )
        record_name = record_names[0]
        record_bytes = archive.read(record_name)
        rows = csv.reader(io.StringIO(record_bytes.decode()))
        recorded_paths: set[str] = set()
        for row in rows:
            if len(row) != 3 or not row[0]:
                raise RuntimeError(f"{wheel_name} has malformed RECORD row: {row!r}")
            relative_path, encoded_hash, encoded_size = row
            path_parts = Path(relative_path).parts
            if (
                Path(relative_path).is_absolute()
                or ".." in path_parts
                or relative_path in recorded_paths
            ):
                raise RuntimeError(
                    f"{wheel_name} has invalid/duplicate RECORD path: {relative_path!r}"
                )
            recorded_paths.add(relative_path)
            if not encoded_hash:
                if relative_path != record_name or encoded_size:
                    raise RuntimeError(
                        f"{wheel_name} has unhashed payload RECORD row: {row!r}"
                    )
                continue
            if not encoded_size.isdigit():
                raise RuntimeError(f"{wheel_name} has invalid RECORD size: {row!r}")
            expected_digest = _record_sha256(
                encoded_hash,
                wheel=wheel_name,
                path=relative_path,
            )
            try:
                archive_bytes = archive.read(relative_path)
            except KeyError as exc:
                raise RuntimeError(
                    f"{wheel_name} RECORD names absent archive payload: {relative_path}"
                ) from exc
            archive_digest = hashlib.sha256(archive_bytes).hexdigest()
            if archive_digest != expected_digest or len(archive_bytes) != int(
                encoded_size
            ):
                raise RuntimeError(
                    "wheel archive differs from its exact RECORD: "
                    f"wheel={wheel_name}, path={relative_path}, "
                    f"record={expected_digest}, archive={archive_digest}, "
                    f"record_size={encoded_size}, "
                    f"archive_size={len(archive_bytes)}"
                )
            installed_path = (installed_root / relative_path).resolve(strict=True)
            installed_digest = sha256_path(installed_path)
            transformation: dict[str, Any] | None = None
            if installed_digest != expected_digest:
                allowed = transformations.get(relative_path)
                if allowed is None:
                    raise RuntimeError(
                        "installed payload differs from exact release wheel: "
                        f"wheel={wheel_name}, path={relative_path}, "
                        f"record={expected_digest}, "
                        f"installed={installed_digest}"
                    )
                expected_installed = allowed.get("installed_sha256", "")
                backup_path_value = allowed.get("backup_path", "")
                kind = allowed.get("kind", "")
                backup_path = Path(backup_path_value)
                backup_digest = (
                    sha256_path(backup_path) if backup_path.is_file() else None
                )
                if (
                    not kind
                    or re.fullmatch(r"[0-9a-f]{64}", expected_installed) is None
                    or installed_digest != expected_installed
                    or backup_digest != expected_digest
                ):
                    raise RuntimeError(
                        "installed payload transformation differs from exact "
                        f"contract: wheel={wheel_name}, path={relative_path}, "
                        f"record={expected_digest}, "
                        f"installed={installed_digest}, "
                        f"backup={backup_digest}, allowed={allowed!r}"
                    )
                transformation = {
                    "kind": kind,
                    "backup_path": str(backup_path),
                    "backup_sha256": backup_digest,
                    "installed_sha256": installed_digest,
                }
                used_transformations.add(relative_path)
            absolute_key = str(installed_path)
            if absolute_key in verified_absolute_paths:
                raise RuntimeError(
                    "release wheels claim the same installed payload: "
                    f"path={absolute_key}, "
                    f"first={verified_absolute_paths[absolute_key]!r}, "
                    f"second={wheel_name}"
                )
            verified_absolute_paths[absolute_key] = {
                "wheel": wheel_name,
                "relative_path": relative_path,
                "sha256": installed_digest,
            }
            verified_payload[relative_path] = {
                "installed_path": absolute_key,
                "record_sha256": expected_digest,
                "sha256": installed_digest,
                "size_bytes": installed_path.stat().st_size,
                "transformation": transformation,
            }
            if Path(relative_path).suffix in {".so", ".dylib", ".pyd"}:
                native_payload[relative_path] = installed_digest
        archive_paths = {
            info.filename for info in archive.infolist() if not info.is_dir()
        }
        if recorded_paths != archive_paths:
            raise RuntimeError(
                f"{wheel_name} RECORD/archive inventory mismatch: "
                f"unrecorded={sorted(archive_paths - recorded_paths)!r}, "
                f"absent={sorted(recorded_paths - archive_paths)!r}"
            )
    if used_transformations != set(transformations):
        raise RuntimeError(
            f"{wheel_name} expected payload transformations were not observed: "
            f"expected={sorted(transformations)!r}, "
            f"observed={sorted(used_transformations)!r}"
        )
    if not verified_payload:
        raise RuntimeError(f"{wheel_name} lacks verified payload identity")
    return {
        "wheel_path": str(wheel_path),
        "wheel_sha256": wheel_sha256,
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "verified_payload_count": len(verified_payload),
        "verified_payload_identity_sha256": canonical_sha256(verified_payload),
        "native_payload": native_payload,
    }


def require_module_payload_bindings(
    module_provenance: dict[str, dict[str, Any]],
    verified_payload_paths: dict[str, dict[str, str]],
    *,
    forbidden_root: Path = Path("/opt/cppmega"),
) -> dict[str, dict[str, Any]]:
    forbidden = forbidden_root.resolve()
    for module_name, provenance in module_provenance.items():
        resolved = Path(str(provenance.get("path", ""))).resolve()
        try:
            resolved.relative_to(forbidden)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                f"imported module shadows release wheels: {module_name}={resolved}"
            )
        verified = verified_payload_paths.get(str(resolved))
        if verified is None or provenance.get("sha256") != verified.get("sha256"):
            raise RuntimeError(
                "imported module is not bound to an exact verified wheel "
                f"RECORD entry: module={module_name}, provenance={provenance!r}, "
                f"verified={verified!r}"
            )
        provenance["release_wheel"] = verified["wheel"]
        provenance["release_wheel_path"] = verified["relative_path"]
    return module_provenance
