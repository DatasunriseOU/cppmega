"""Deterministic real-Megatron bootstrap for the cppmega pytest suite.

The local test environment uses a sibling Megatron-LM source checkout rather
than an installed ``megatron`` distribution.  Import it before test collection
so collection-time fallback stubs cannot claim the namespace first.
"""

from __future__ import annotations

import importlib
from importlib.machinery import ModuleSpec
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


_REPO_ROOT = Path(__file__).resolve().parent
_SOURCE_ROOT_ENV = ("MEGATRON_LM_REPO", "MEGATRON_ROOT")
_DEFAULT_SOURCE_ROOTS = (
    _REPO_ROOT.parent / "Megatron-LM",
    _REPO_ROOT.parent / "megatron-lm",
    Path("/opt/megatron-lm"),
)
_ENVIRONMENT_MANIFEST = "cppmega-environment.json"
_ENVIRONMENT_SCHEMA = 1
_PORTABLE_TEST_PROFILE = "portable-data"
_NATIVE_TOOLCHAIN_MARKER = "native_toolchain"
_EXPLICIT_SOURCE_COMMIT_ENV = "CPPMEGA_MEGATRON_COMMIT"
_PORTABLE_TEST_ALLOWLIST = frozenset(
    {
        "tests/test_ci_stream_inventory.py",
        "tests/test_ci_source_binding_projection.py",
        "tests/test_ci_source_sidecars.py",
        "tests/test_export_ci_content_store_case5.py",
        "tests/test_data_pipeline_contracts.py",
        "tests/test_data_prep_parquet_to_megatron.py",
        "tests/test_case5_ksh_domain_contract.py",
        "tests/test_domain_megatron_sidecars.py",
        "tests/test_eval_domain_routed_codegen.py",
        "tests/test_ksh_python_domain_parsers.py",
        "tests/test_nebius_h200_megatron_cpp_generation_eval.py",
        "tests/test_nebius_h200_megatron_cpp_world_sweep.py",
        "tests/test_subset_megatron_sidecar_prefix.py",
        "tests/test_build_macro_routes_megatron_bundle.py",
        "tests/test_publish_megatron_bundle_to_nebius_s3.py",
        "tests/test_graphql_pr_stream.py",
        "tests/test_pr_completion_receipt.py",
        "tests/test_pr_export_batches.py",
        "tests/test_restore_megatron_bundle_from_nebius_s3.py",
        "tests/test_streaming_overlong_lossless.py",
        "tests/test_workflow_runner_policy.py",
        "tests/ci/test_repository_ci_runner.py",
    }
)


def _is_megatron_source_root(path: Path) -> bool:
    return (path / "megatron" / "core" / "__init__.py").is_file()


def _same_location(left: Path, right: Path) -> bool:
    left = left.expanduser()
    right = right.expanduser()
    try:
        return os.path.samefile(left, right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def _environment_manifest() -> tuple[Path, dict[str, object]] | None:
    manifest_path = Path(sys.prefix) / _ENVIRONMENT_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid Megatron environment manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"invalid Megatron environment manifest: {manifest_path}")
    manifest_repo_value = manifest.get("repo_root")
    if not isinstance(manifest_repo_value, str) or not manifest_repo_value:
        raise RuntimeError(f"{manifest_path} does not define a valid repo_root")
    if manifest.get("schema") != _ENVIRONMENT_SCHEMA:
        raise RuntimeError(
            f"{manifest_path} has unsupported environment schema: "
            f"{manifest.get('schema')!r}"
        )
    if not isinstance(manifest.get("megatron_root"), str) or not manifest.get(
        "megatron_root"
    ):
        raise RuntimeError(f"{manifest_path} does not define a valid megatron_root")
    if not isinstance(manifest.get("megatron_commit"), str) or not manifest.get(
        "megatron_commit"
    ):
        raise RuntimeError(f"{manifest_path} does not define megatron_commit")
    if not isinstance(manifest.get("source_dirty"), bool):
        raise RuntimeError(f"{manifest_path} does not define boolean source_dirty")

    # ``repo_root`` records the checkout used to create the environment. A
    # repository runner may execute the same receipt from an extracted,
    # temporary checkout, so it must not be compared with this module's path.
    return manifest_path, manifest


def _reject_unowned_repo_venv(
    manifest: tuple[Path, dict[str, object]] | None,
) -> None:
    repo_venv = _REPO_ROOT / ".venv"
    if not (repo_venv.exists() or repo_venv.is_symlink()):
        return
    if not _same_location(repo_venv, Path(sys.prefix)):
        return
    if manifest is None:
        raise RuntimeError(
            "cppmega pytest refuses the repository .venv because it resolves to "
            f"a shared venv ({Path(sys.prefix).resolve(strict=False)}) without a "
            "matching cppmega environment manifest"
        )


def _validate_manifest_source(
    manifest_path: Path,
    manifest: dict[str, object],
    source_root: Path,
) -> None:
    """Validate the source receipt before allowing pytest to import it."""

    expected_commit = manifest.get("megatron_commit")
    if not isinstance(expected_commit, str) or not expected_commit:
        raise RuntimeError(f"{manifest_path} does not define megatron_commit")
    if manifest.get("source_dirty") is not False:
        raise RuntimeError(
            f"Megatron source receipt is not clean: {manifest_path}"
        )

    try:
        head_result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"cannot inspect Megatron source receipt root {source_root}"
        ) from exc
    if head_result.returncode != 0 or status_result.returncode != 0:
        detail = (head_result.stderr or status_result.stderr).strip()
        raise RuntimeError(
            f"cannot inspect Megatron source receipt root {source_root}: {detail}"
        )
    actual_commit = head_result.stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"Megatron source HEAD {actual_commit} differs from receipt "
            f"{expected_commit} ({manifest_path})"
        )
    if status_result.stdout.strip():
        raise RuntimeError(
            f"Megatron source changed after receipt creation: {source_root}"
        )


def _resolve_megatron_source_root() -> Path | None:
    manifest_entry = _environment_manifest()
    _reject_unowned_repo_venv(manifest_entry)

    explicit = [
        (name, Path(value).expanduser().resolve(strict=False))
        for name in _SOURCE_ROOT_ENV
        if (value := os.environ.get(name))
    ]
    if explicit:
        first_root = explicit[0][1]
        if any(not _same_location(first_root, path) for _name, path in explicit[1:]):
            configured = ", ".join(f"{name}={path}" for name, path in explicit)
            raise RuntimeError(
                "conflicting explicit Megatron-LM source roots: " + configured
            )
        name, root = explicit[0]
        if not _is_megatron_source_root(root):
            raise RuntimeError(
                f"{name}={root} does not point to a real Megatron-LM source root; "
                "expected megatron/core/__init__.py"
            )
        if manifest_entry is not None:
            manifest_path, manifest = manifest_entry
            manifest_root_value = manifest.get("megatron_root")
            manifest_root = (
                Path(manifest_root_value).expanduser().resolve(strict=False)
                if isinstance(manifest_root_value, str) and manifest_root_value
                else None
            )
            if manifest_root is not None and _same_location(root, manifest_root):
                # The explicit path points at the source recorded by the
                # environment receipt. Validate its pinned commit and clean
                # state, but keep the explicit path authoritative.
                _validate_manifest_source(manifest_path, manifest, root)
            else:
                # A caller may intentionally select a separately checked-out
                # source tree. The receipt for another root must not silently
                # bless it; require an exact commit contract for that root.
                expected_commit = os.environ.get(_EXPLICIT_SOURCE_COMMIT_ENV)
                if not expected_commit:
                    raise RuntimeError(
                        f"{name} points outside the environment receipt and requires "
                        f"{_EXPLICIT_SOURCE_COMMIT_ENV}"
                    )
                _validate_manifest_source(
                    Path(f"<explicit {name} source contract>"),
                    {
                        "megatron_commit": expected_commit,
                        "source_dirty": False,
                    },
                    root,
                )
        else:
            expected_commit = os.environ.get(_EXPLICIT_SOURCE_COMMIT_ENV)
            if not expected_commit:
                raise RuntimeError(
                    f"{name} requires {_EXPLICIT_SOURCE_COMMIT_ENV} or a dedicated "
                    "cppmega-environment.json receipt"
                )
            _validate_manifest_source(
                Path(f"<explicit {name} source contract>"),
                {
                    "megatron_commit": expected_commit,
                    "source_dirty": False,
                },
                root,
            )
        return root

    if manifest_entry is not None:
        manifest_path, manifest = manifest_entry
        manifest_root_value = manifest.get("megatron_root") if isinstance(manifest, dict) else None
        if not isinstance(manifest_root_value, str) or not manifest_root_value:
            raise RuntimeError(
                f"{manifest_path} does not define a valid megatron_root"
            )
        manifest_root = Path(manifest_root_value).expanduser().resolve()
        if not _is_megatron_source_root(manifest_root):
            raise RuntimeError(
                f"Megatron environment manifest points to an invalid source root: "
                f"{manifest_root}"
            )
        _validate_manifest_source(manifest_path, manifest, manifest_root)
        return manifest_root

    return None


def _prepend_source_root(source_root: Path) -> None:
    value = str(source_root)
    sys.path[:] = [entry for entry in sys.path if entry != value]
    sys.path.insert(0, value)


def _module_file(module: ModuleType) -> Path | None:
    value = getattr(module, "__file__", None)
    if not isinstance(value, (str, os.PathLike)):
        return None
    return Path(value).resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_real_megatron(
    megatron: ModuleType,
    core: ModuleType,
    *,
    source_root: Path | None,
) -> Path:
    for name, module in (("megatron", megatron), ("megatron.core", core)):
        if getattr(module, "__cppmega_stub__", False) is True:
            raise RuntimeError(
                f"{name} is a cppmega test stub, not a real Megatron-LM import"
            )
        if not isinstance(getattr(module, "__spec__", None), ModuleSpec):
            raise RuntimeError(
                f"{name} has no valid ModuleSpec; sys.modules is polluted"
            )

    core_file = _module_file(core)
    if core_file is None:
        raise RuntimeError(
            "megatron.core has no source file; sys.modules does not contain a real "
            "Megatron-LM import"
        )

    if source_root is None:
        return core_file.parents[2]

    package_root = (source_root / "megatron").resolve()
    namespace = getattr(megatron, "__path__", None)
    if namespace is None:
        raise RuntimeError(
            "megatron is not a package; sys.modules is polluted instead of "
            "containing the real Megatron-LM namespace"
        )
    namespace_paths = [Path(path).resolve() for path in namespace]
    if not namespace_paths or namespace_paths[0] != package_root:
        raise RuntimeError(
            "megatron namespace did not resolve from the selected real Megatron-LM "
            f"source root {source_root}; paths={namespace_paths}"
        )
    if not _is_within(core_file, package_root):
        raise RuntimeError(
            "megatron.core was already imported from the wrong source: "
            f"{core_file}; expected under {package_root}"
        )

    wrong_origins: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if name != "megatron" and not name.startswith("megatron."):
            continue
        if not isinstance(module, ModuleType):
            wrong_origins.append(f"{name}=<{type(module).__name__}>")
            continue
        module_file = _module_file(module)
        if module_file is not None and not _is_within(module_file, package_root):
            wrong_origins.append(f"{name}={module_file}")
    if wrong_origins:
        raise RuntimeError(
            "Megatron sys.modules pollution detected outside the selected source root: "
            + ", ".join(wrong_origins[:8])
        )
    return source_root


def _bootstrap_real_megatron() -> Path:
    source_root = _resolve_megatron_source_root()
    if source_root is None:
        raise RuntimeError(
            "cppmega tests require a dedicated cppmega-environment.json receipt "
            "or an explicit MEGATRON_LM_REPO/MEGATRON_ROOT plus "
            f"{_EXPLICIT_SOURCE_COMMIT_ENV}; refusing ambient Megatron imports"
        )
    _prepend_source_root(source_root)

    try:
        megatron = importlib.import_module("megatron")
        core = importlib.import_module("megatron.core")
    except Exception as exc:
        probed = ", ".join(str(path) for path in _DEFAULT_SOURCE_ROOTS)
        raise RuntimeError(
            "cppmega tests require a real Megatron-LM import. Set "
            "MEGATRON_LM_REPO or MEGATRON_ROOT to a source root containing "
            f"megatron/core/__init__.py. Probed: {probed}"
        ) from exc

    return _validate_real_megatron(megatron, core, source_root=source_root)


def _portable_profile_requested(
    _argv: tuple[str, ...] | None = None,
) -> bool:
    """Return whether the explicit Megatron-free CI profile is requested.

    Test selection is validated after pytest has parsed options and collected
    items. Parsing ``sys.argv`` here misclassifies node IDs and option values.
    The argument is retained for compatibility with environment-contract tests.
    """
    return os.environ.get("CPPMEGA_TEST_PROFILE") == _PORTABLE_TEST_PROFILE


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Enforce the portable allowlist and its explicit dependency boundary."""
    if not _portable_profile_requested():
        return
    selected: list[str] = []
    for item in items:
        path = Path(str(item.fspath)).resolve()
        try:
            relative = path.relative_to(_REPO_ROOT).as_posix()
        except ValueError as exc:
            raise pytest.UsageError(
                "portable-data pytest selection must stay inside cppmega: "
                f"{path}"
            ) from exc
        selected.append(relative)
    disallowed = sorted(set(selected) - _PORTABLE_TEST_ALLOWLIST)
    if not selected:
        raise pytest.UsageError(
            "portable-data profile requires explicit allowlisted test files"
        )
    if disallowed:
        raise pytest.UsageError(
            "portable-data profile refuses non-portable tests: "
            + ", ".join(disallowed)
        )
    native_toolchain_skip = pytest.mark.skip(
        reason=(
            "portable-data profile excludes tests that require libclang "
            "or a local C/C++ compiler"
        )
    )
    for item in items:
        if item.get_closest_marker(_NATIVE_TOOLCHAIN_MARKER) is not None:
            item.add_marker(native_toolchain_skip)


MEGATRON_SOURCE_ROOT = (
    None if _portable_profile_requested() else _bootstrap_real_megatron()
)
