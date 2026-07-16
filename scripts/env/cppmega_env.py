#!/usr/bin/env python3
"""Create and verify an isolated cppmega source environment.

The tool deliberately does not install packages. It creates a dedicated venv,
wires the cppmega and Megatron-LM source trees through a small ``.pth`` file,
records the exact Megatron commit, and then verifies the resulting interpreter.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "cppmega-environment.json"
SOURCE_PATHS_NAME = "00_cppmega_sources.pth"
MANIFEST_SCHEMA = 1
PROFILES = ("locked", "source")


class EnvError(RuntimeError):
    """Raised when a bootstrap or verification guard fails."""


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    title: str = "cppmega environment verification"

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(Check(name=name, ok=ok, detail=detail))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {"name": check.name, "ok": check.ok, "detail": check.detail}
                for check in self.checks
            ],
        }

    def render(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [f"{status} {self.title}"]
        for check in self.checks:
            marker = "PASS" if check.ok else "FAIL"
            lines.append(f"[{marker}] {check.name}: {check.detail}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SourceInfo:
    root: Path
    head: str
    expected_commit: str
    expected_ref: str
    dirty_entries: tuple[str, ...]
    version: str
    requires_python: str
    dependencies: tuple[str, ...]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _same_location(left: Path, right: Path) -> bool:
    left = left.expanduser()
    right = right.expanduser()
    try:
        return os.path.samefile(left, right)
    except OSError:
        return _resolved(left) == _resolved(right)


def _consistent_explicit_root(
    configured: Sequence[tuple[str, str]],
) -> Path | None:
    if not configured:
        return None
    resolved = [(name, _resolved(Path(value))) for name, value in configured]
    first = resolved[0][1]
    if any(not _same_location(first, path) for _name, path in resolved[1:]):
        detail = ", ".join(f"{name}={path}" for name, path in resolved)
        raise EnvError("conflicting explicit Megatron-LM source roots: " + detail)
    return first


def _read_pyvenv_cfg(env_dir: Path) -> dict[str, str]:
    path = env_dir / "pyvenv.cfg"
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw_line.partition("=")
        if separator:
            values[key.strip().lower()] = value.strip()
    return values


def ensure_target_isolation(repo_root: Path, env_dir: Path) -> Path:
    """Reject repository/shared/symlink environments before any mutation."""

    repo_root = _resolved(repo_root)
    requested = env_dir.expanduser()
    if requested.is_symlink():
        raise EnvError(
            f"environment path is a symlink and may target a shared venv: {requested}"
        )

    target = _resolved(requested)
    if target == repo_root or _is_within(target, repo_root):
        raise EnvError(f"environment must live outside the cppmega checkout: {target}")

    forbidden: list[tuple[str, Path]] = []
    repo_venv = repo_root / ".venv"
    if repo_venv.exists() or repo_venv.is_symlink():
        forbidden.append(("cppmega .venv", _resolved(repo_venv)))

    active = os.environ.get("VIRTUAL_ENV")
    if active:
        forbidden.append(("active shared VIRTUAL_ENV", _resolved(Path(active))))

    current_prefix = _resolved(Path(sys.prefix))
    if current_prefix != target and _is_within(target, current_prefix):
        forbidden.append(("another active interpreter venv", current_prefix))

    for label, root in forbidden:
        if target == root or _is_within(target, root):
            raise EnvError(f"environment resolves inside {label}: {root} (shared venv)")

    if target.exists():
        cfg = _read_pyvenv_cfg(target)
        inherited = cfg.get("include-system-site-packages")
        if inherited is None or inherited.lower() not in {"false", "0", "no"}:
            raise EnvError(
                "environment is not proven isolated; "
                f"include-system-site-packages={inherited!r} in {target / 'pyvenv.cfg'}"
            )
    return target


def _clean_environment(env_dir: Path | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_PROMPT",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
    ):
        environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if env_dir is not None:
        environment["VIRTUAL_ENV"] = str(env_dir)
        bin_dir = env_dir / ("Scripts" if os.name == "nt" else "bin")
        environment["PATH"] = os.pathsep.join(
            [str(bin_dir), environment.get("PATH", "")]
        )
    return environment


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env_dir: Path | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=_clean_environment(env_dir),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _run_bootstrap_venv(base_python: Path, env_dir: Path) -> None:
    command = [str(base_python), "-m", "venv", "--copies", str(env_dir)]
    result = subprocess.run(
        command,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown venv error"
        raise EnvError(f"venv bootstrap failed: {detail}")


def _target_python(env_dir: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return env_dir / relative


def _python_identity(python: Path) -> dict[str, Any]:
    code = (
        "import json, sys; "
        "print(json.dumps({'version': list(sys.version_info[:3]), "
        "'prefix': sys.prefix, 'base_prefix': sys.base_prefix}))"
    )
    result = _run([str(python), "-c", code])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise EnvError(f"cannot execute Python {python}: {detail}")
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise EnvError(
            f"Python identity probe returned invalid JSON: {result.stdout!r}"
        ) from exc


def _site_packages(env_dir: Path) -> Path:
    python = _target_python(env_dir)
    code = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    result = _run([str(python), "-c", code], env_dir=env_dir)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise EnvError(f"cannot resolve target site-packages: {detail}")
    return _resolved(Path(result.stdout.strip()))


def write_source_paths(
    site_packages: Path, repo_root: Path, megatron_root: Path
) -> Path:
    site_packages.mkdir(parents=True, exist_ok=True)
    path = site_packages / SOURCE_PATHS_NAME
    content = f"{_resolved(repo_root)}\n{_resolved(megatron_root)}\n"
    path.write_text(content, encoding="utf-8")
    return path


def _git(megatron_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", "-C", str(megatron_root), *args])


def _git_output(megatron_root: Path, *args: str) -> str:
    result = _git(megatron_root, *args)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise EnvError(f"git {' '.join(args)} failed for {megatron_root}: {detail}")
    return result.stdout.strip()


def _stack_value(repo_root: Path, key: str) -> str:
    stack_path = repo_root / "STACK.lock"
    if not stack_path.is_file():
        raise EnvError(f"missing stack contract: {stack_path}")
    pattern = re.compile(
        rf"^\s{{2}}{re.escape(key)}:\s*['\"]?([^'\"\s#]+)", re.MULTILINE
    )
    match = pattern.search(stack_path.read_text(encoding="utf-8"))
    if not match:
        raise EnvError(f"STACK.lock does not define base.{key}")
    return match.group(1)


def _stack_megatron_ref(repo_root: Path) -> str:
    stack_path = repo_root / "STACK.lock"
    if not stack_path.is_file():
        raise EnvError(f"missing stack contract: {stack_path}")
    in_megatron = False
    for line in stack_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("  megatron_lm:"):
            in_megatron = True
            continue
        if in_megatron and line.startswith("  ") and not line.startswith("    "):
            break
        if in_megatron:
            match = re.match(r"^\s{4}ref:\s*['\"]?([^'\"\s#]+)", line)
            if match:
                return match.group(1)
    raise EnvError("STACK.lock does not define sources.megatron_lm.ref")


def _literal_version(package_info: Path) -> str:
    tree = ast.parse(
        package_info.read_text(encoding="utf-8"), filename=str(package_info)
    )
    constants: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            constants[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    direct = constants.get("__version__")
    if isinstance(direct, str):
        return direct
    pieces = [constants.get(name) for name in ("MAJOR", "MINOR", "PATCH")]
    if all(isinstance(value, int) for value in pieces):
        suffix = constants.get("PRE_RELEASE", "")
        return ".".join(str(value) for value in pieces) + str(suffix)
    raise EnvError(f"cannot read Megatron version from {package_info}")


def _source_project(megatron_root: Path) -> tuple[str, tuple[str, ...]]:
    try:
        import tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - Python 3.10 only
        raise EnvError("cppmega environment bootstrap requires Python 3.11+") from exc
    data = tomllib.loads((megatron_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    requires_python = str(project.get("requires-python", ""))
    dependencies = tuple(str(item) for item in project.get("dependencies", ()))
    return requires_python, dependencies


def _project_data_dependencies(repo_root: Path) -> tuple[str, ...]:
    project_path = repo_root / "pyproject.toml"
    if not project_path.is_file():
        return ()
    try:
        import tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - Python 3.10 only
        raise EnvError("cppmega environment verification requires Python 3.11+") from exc
    data = tomllib.loads(project_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    optional = project.get("optional-dependencies", {})
    requirements = optional.get("data", ()) if isinstance(optional, dict) else ()
    if not isinstance(requirements, list) or not all(
        isinstance(item, str) for item in requirements
    ):
        raise EnvError(f"invalid project.optional-dependencies.data in {project_path}")
    return tuple(requirements)


def inspect_source(
    repo_root: Path,
    megatron_root: Path,
    expected_ref: str,
    *,
    allow_dirty: bool,
) -> SourceInfo:
    repo_root = _resolved(repo_root)
    megatron_root = _resolved(megatron_root)
    required = (
        megatron_root / "pyproject.toml",
        megatron_root / "megatron" / "core" / "__init__.py",
        megatron_root / "megatron" / "core" / "package_info.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise EnvError(
            "not a real Megatron-LM source root; missing: " + ", ".join(missing)
        )

    inside = _git_output(megatron_root, "rev-parse", "--is-inside-work-tree")
    if inside != "true":
        raise EnvError(f"Megatron source is not a git worktree: {megatron_root}")

    head = _git_output(megatron_root, "rev-parse", "HEAD")
    expected_commit = _git_output(
        megatron_root, "rev-parse", "--verify", f"{expected_ref}^{{commit}}"
    )
    status = _git_output(
        megatron_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    dirty_entries = tuple(line for line in status.splitlines() if line)

    problems: list[str] = []
    if head != expected_commit:
        problems.append(
            f"Megatron HEAD {head} does not match expected {expected_ref} ({expected_commit})"
        )
    if dirty_entries and not allow_dirty:
        problems.append("Megatron source is dirty: " + ", ".join(dirty_entries[:8]))
    if problems:
        raise EnvError("; ".join(problems))

    version = _literal_version(required[2])
    requires_python, dependencies = _source_project(megatron_root)
    return SourceInfo(
        root=megatron_root,
        head=head,
        expected_commit=expected_commit,
        expected_ref=expected_ref,
        dirty_entries=dirty_entries,
        version=version,
        requires_python=requires_python,
        dependencies=dependencies,
    )


def _default_megatron_root(repo_root: Path) -> Path:
    candidates = (
        repo_root.parent / "Megatron-LM",
        Path("/Volumes/external/sources/Megatron-LM"),
        repo_root.parent / "megatron-lm",
        Path("/opt/megatron-lm"),
    )
    for candidate in candidates:
        if (candidate / "megatron" / "core" / "__init__.py").is_file():
            return _resolved(candidate)
    return _resolved(candidates[0])


def _load_manifest(env_dir: Path) -> dict[str, Any] | None:
    path = env_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EnvError(f"invalid environment manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnvError(f"invalid environment manifest object: {path}")
    return value


def _write_manifest(env_dir: Path, value: dict[str, Any]) -> Path:
    path = env_dir / MANIFEST_NAME
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _manifest_mismatches(
    manifest: dict[str, Any], expected: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    mismatches: list[tuple[str, Any, Any]] = []
    for key, expected_value in expected.items():
        actual_value = manifest.get(key)
        if isinstance(expected_value, bool):
            matches = type(actual_value) is bool and actual_value is expected_value
        else:
            matches = actual_value == expected_value
        if not matches:
            mismatches.append((key, actual_value, expected_value))
    return mismatches


def _dependency_spec(raw: str) -> tuple[str, str]:
    requirement = raw.strip()
    match = re.match(
        r"^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^]]+\])?\s*(.*)$", requirement
    )
    if not match:
        raise EnvError(f"unsupported dependency requirement: {raw}")
    name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
    # Keep the complete PEP 508 requirement, including markers. The target
    # probe parses/evaluates it with ``packaging.Requirement``; this host-side
    # helper only needs the normalized project name for diagnostics/dedup.
    return name, requirement


def _dependency_contract(
    repo_root: Path, source: SourceInfo, profile: str
) -> list[dict[str, str]]:
    requirements = []
    for raw in (*source.dependencies, *_project_data_dependencies(repo_root)):
        name, requirement = _dependency_spec(raw)
        existing = next(
            (item for item in requirements if item["requirement"] == requirement),
            None,
        )
        if existing is None:
            requirements.append(
                {
                    "name": name,
                    "specifier": requirement.split(";", 1)[0]
                    .removeprefix(name)
                    .strip(),
                    "requirement": requirement,
                }
            )
    if profile == "locked":
        torch_pin = _stack_value(repo_root, "torch")
        pinned = f"torch=={torch_pin}"
        if not any(item["requirement"] == pinned for item in requirements):
            requirements.append(
                {"name": "torch", "specifier": f"=={torch_pin}", "requirement": pinned}
            )
    return requirements


_PROBE = r"""
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import sys

repo = Path(os.environ["CPPMEGA_PROBE_REPO"]).resolve()
source = Path(os.environ["CPPMEGA_PROBE_MEGATRON"]).resolve()
target = Path(os.environ["CPPMEGA_PROBE_ENV"]).resolve()
python_pin = os.environ["CPPMEGA_PROBE_PYTHON"]
requirements = json.loads(os.environ["CPPMEGA_PROBE_REQUIREMENTS"])
forbidden = [Path(value).resolve() for value in json.loads(os.environ["CPPMEGA_PROBE_FORBIDDEN"])]
allowed_roots = [
    repo,
    source,
    target,
    Path(sys.base_prefix).resolve(),
]
errors = []
result = {
    "python": ".".join(map(str, sys.version_info[:3])),
    "prefix": str(Path(sys.prefix).resolve()),
    "base_prefix": str(Path(sys.base_prefix).resolve()),
    "packages": {},
    "modules": {},
    "sys_path": list(sys.path),
}

def within(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

def registered_editable_placeholders():
    placeholders = set()
    for module in tuple(sys.modules.values()):
        try:
            placeholder = getattr(module, "PATH_PLACEHOLDER", None)
        except BaseException:
            continue
        if isinstance(placeholder, str) and placeholder.startswith("__editable__."):
            placeholders.add(placeholder)
    return placeholders

editable_placeholders = registered_editable_placeholders()

if Path(sys.prefix).resolve() != target:
    errors.append(f"sys.prefix={sys.prefix} expected {target}")
if ".".join(map(str, sys.version_info[:2])) != python_pin:
    errors.append(f"Python {sys.version_info.major}.{sys.version_info.minor} expected {python_pin}")
if os.environ.get("PYTHONPATH"):
    errors.append("PYTHONPATH leaked into target probe")

for entry in sys.path:
    if not entry:
        errors.append("empty sys.path entry")
        continue
    entry_path = Path(entry)
    # PEP 660 namespace finders register a non-filesystem placeholder in
    # sys.path. Accept only the exact placeholder exposed by a loaded finder;
    # an arbitrary external directory can otherwise evade the allowlist by
    # copying the ``__editable__`` basename.
    if entry_path.name.startswith("__editable__."):
        if entry not in editable_placeholders or entry_path.exists():
            errors.append(f"unapproved editable path: {entry}")
        continue
    path = entry_path.resolve()
    for root in forbidden:
        if path == root or within(path, root):
            errors.append(f"shared path leaked into sys.path: {path}")
    if not any(path == root or within(path, root) for root in allowed_roots):
        errors.append(f"unapproved sys.path entry: {path}")

try:
    from packaging.requirements import Requirement
except Exception as exc:
    Requirement = None
    errors.append(f"packaging import failed: {type(exc).__name__}: {exc}")

for requirement in requirements:
    raw_requirement = requirement.get("requirement") or requirement.get("name", "")
    if Requirement is None:
        name = requirement.get("name", "")
        if name:
            try:
                current = metadata.version(name)
            except metadata.PackageNotFoundError:
                errors.append(f"missing distribution: {raw_requirement}")
            else:
                result["packages"][name] = current
        continue
    try:
        parsed_requirement = Requirement(raw_requirement)
    except Exception as exc:
        errors.append(
            f"invalid dependency requirement {raw_requirement!r}: "
            f"{type(exc).__name__}: {exc}"
        )
        continue
    if parsed_requirement.marker is not None and not parsed_requirement.marker.evaluate():
        continue
    name = parsed_requirement.name
    specifier = str(parsed_requirement.specifier)
    try:
        current = metadata.version(name)
    except metadata.PackageNotFoundError:
        errors.append(f"missing distribution: {name}{specifier}")
        continue
    result["packages"][name] = current
    if not specifier:
        continue
    try:
        if current not in parsed_requirement.specifier:
            errors.append(f"{name}=={current} does not satisfy {specifier}")
    except Exception as exc:
        errors.append(
            f"could not evaluate {raw_requirement!r}: {type(exc).__name__}: {exc}"
        )

for module_name in (
    "cppmega",
    "megatron",
    "megatron.core",
    "megatron.core.package_info",
    "megatron.core.transformer.transformer_layer",
    "megatron.core.transformer.moe.moe_utils",
):
    try:
        module = importlib.import_module(module_name)
    except BaseException as exc:
        errors.append(f"{module_name} import failed: {type(exc).__name__}: {exc}")
        continue
    file_name = getattr(module, "__file__", None)
    paths = [str(Path(value).resolve()) for value in getattr(module, "__path__", ())]
    result["modules"][module_name] = {"file": file_name, "path": paths}
    if getattr(module, "__cppmega_stub__", False):
        errors.append(f"{module_name} is a cppmega test stub")
    if module_name == "cppmega":
        if file_name is None or not within(Path(file_name).resolve(), repo):
            errors.append(f"cppmega origin is outside {repo}: {file_name}")
    elif module_name == "megatron":
        package_root = source / "megatron"
        if not paths or any(not within(Path(value), package_root) for value in paths):
            errors.append(f"megatron namespace is outside {package_root}: {paths}")
    elif module_name.startswith("megatron."):
        package_root = source / "megatron"
        if file_name is None or not within(Path(file_name).resolve(), package_root):
            errors.append(f"{module_name} origin is outside {package_root}: {file_name}")

result["errors"] = errors
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if not errors else 1)
"""


def _probe_target(
    repo_root: Path,
    env_dir: Path,
    source: SourceInfo,
    profile: str,
) -> tuple[bool, str]:
    python = _target_python(env_dir)
    if not python.is_file():
        return False, f"missing target interpreter: {python}"

    forbidden: list[str] = []
    repo_venv = repo_root / ".venv"
    if repo_venv.exists() or repo_venv.is_symlink():
        forbidden.append(str(_resolved(repo_venv)))
    active = os.environ.get("VIRTUAL_ENV")
    if active and _resolved(Path(active)) != env_dir:
        forbidden.append(str(_resolved(Path(active))))

    environment = _clean_environment(env_dir)
    environment.update(
        {
            "CPPMEGA_PROBE_REPO": str(repo_root),
            "CPPMEGA_PROBE_MEGATRON": str(source.root),
            "CPPMEGA_PROBE_ENV": str(env_dir),
            "CPPMEGA_PROBE_PYTHON": _stack_value(repo_root, "python"),
            "CPPMEGA_PROBE_REQUIREMENTS": json.dumps(
                _dependency_contract(repo_root, source, profile)
            ),
            "CPPMEGA_PROBE_FORBIDDEN": json.dumps(sorted(set(forbidden))),
        }
    )
    timeout_seconds = 120
    try:
        result = subprocess.run(
            [str(python), "-c", _PROBE],
            cwd=Path(tempfile.gettempdir()),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"target environment probe timed out after {timeout_seconds}s"
    except OSError as exc:
        return False, f"target environment probe could not start: {exc}"
    payload: dict[str, Any] | None = None
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        detail = (
            result.stderr.strip() or result.stdout.strip() or "probe returned no JSON"
        )
        return False, detail
    errors = payload.get("errors", [])
    if result.returncode != 0 or errors:
        details = "; ".join(str(item) for item in errors)
        warnings = result.stderr.strip()
        if warnings:
            details = (
                f"{details}; stderr={warnings[-1200:]}" if details else warnings[-1200:]
            )
        return False, details or f"probe exited {result.returncode}"
    packages = ", ".join(
        f"{name}={version}"
        for name, version in sorted(payload.get("packages", {}).items())
    )
    return True, f"real cppmega/Megatron imports; {packages}"


def _pip_check(env_dir: Path) -> tuple[bool, str]:
    python = _target_python(env_dir)
    result = _run([str(python), "-m", "pip", "check"], env_dir=env_dir)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return False, output or f"pip check exited {result.returncode}"
    return True, output or "no broken requirements"


def _resolve_configuration(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, str, str, dict[str, Any] | None]:
    repo_root = _resolved(Path(args.repo_root))
    env_value = args.env_dir or os.environ.get("CPPMEGA_ENV_DIR")
    env_dir = (
        _resolved(Path(env_value)) if env_value else repo_root.parent / "cppmega-venv"
    )
    manifest = _load_manifest(env_dir) if env_dir.exists() else None

    if args.megatron_root:
        explicit_source = _resolved(Path(args.megatron_root))
    else:
        explicit_source = _consistent_explicit_root(
            tuple(
                (name, value)
                for name, value in (
                    ("MEGATRON_LM_REPO", os.environ.get("MEGATRON_LM_REPO")),
                    ("MEGATRON_ROOT", os.environ.get("MEGATRON_ROOT")),
                )
                if value
            )
        )
    source_value = explicit_source or (manifest or {}).get("megatron_root")
    megatron_root = (
        _resolved(Path(source_value))
        if source_value
        else _default_megatron_root(repo_root)
    )

    profile = args.profile or (manifest or {}).get("profile") or "locked"
    if profile not in PROFILES:
        raise EnvError(f"unsupported environment profile: {profile}")

    expected_ref = (
        args.megatron_ref
        or os.environ.get("CPPMEGA_MEGATRON_REF")
        or os.environ.get("CPPMEGA_MEGATRON_COMMIT")
        or (manifest or {}).get("megatron_commit")
        or _stack_megatron_ref(repo_root)
    )
    return repo_root, env_dir, megatron_root, str(expected_ref), profile, manifest


def verify(args: argparse.Namespace) -> Report:
    report = Report()
    try:
        repo_root, env_dir, megatron_root, expected_ref, profile, manifest = (
            _resolve_configuration(args)
        )
    except EnvError as exc:
        report.add("configuration", False, str(exc))
        return report

    python_path = os.environ.get("PYTHONPATH")
    report.add(
        "shell PYTHONPATH",
        not bool(python_path),
        "unset" if not python_path else f"must be unset, found {python_path}",
    )
    python_home = os.environ.get("PYTHONHOME")
    report.add(
        "shell PYTHONHOME",
        not bool(python_home),
        "unset" if not python_home else f"must be unset, found {python_home}",
    )
    active = os.environ.get("VIRTUAL_ENV")
    active_path = _resolved(Path(active)) if active else None
    active_ok = active_path is None or active_path == env_dir
    report.add(
        "shell VIRTUAL_ENV",
        active_ok,
        "unset or target environment"
        if active_ok
        else f"points to shared environment {active_path}, target is {env_dir}",
    )

    try:
        ensure_target_isolation(repo_root, env_dir)
    except EnvError as exc:
        report.add("target isolation", False, str(exc))
        return report
    report.add("target isolation", True, str(env_dir))

    cfg = _read_pyvenv_cfg(env_dir)
    if not cfg:
        report.add("target venv", False, f"missing {env_dir / 'pyvenv.cfg'}")
        return report
    inherited = cfg.get("include-system-site-packages")
    isolated = inherited is not None and inherited.lower() in {"false", "0", "no"}
    report.add(
        "target venv",
        isolated,
        "include-system-site-packages=false"
        if isolated
        else f"include-system-site-packages={inherited!r} is not an isolation proof",
    )
    if not isolated:
        return report

    try:
        source = inspect_source(
            repo_root,
            megatron_root,
            expected_ref,
            allow_dirty=args.allow_dirty_source,
        )
    except EnvError as exc:
        report.add("Megatron source", False, str(exc))
        return report
    dirty_note = "dirty override" if source.dirty_entries else "clean"
    report.add(
        "Megatron source",
        True,
        f"{source.root} at {source.head} ({dirty_note}, version {source.version})",
    )
    report.add(
        "source reproducibility",
        not source.dirty_entries,
        "source_dirty=false"
        if not source.dirty_entries
        else "source_dirty=true; a dirty receipt cannot be used by pytest",
    )

    if manifest is None:
        report.add("environment manifest", False, f"missing {env_dir / MANIFEST_NAME}")
    else:
        expected_manifest = {
            "schema": MANIFEST_SCHEMA,
            "repo_root": str(repo_root),
            "megatron_root": str(source.root),
            "megatron_commit": source.expected_commit,
            "source_dirty": bool(source.dirty_entries),
            "profile": profile,
        }
        mismatches = [
            f"{key}={actual!r} expected {expected!r}"
            for key, actual, expected in _manifest_mismatches(
                manifest, expected_manifest
            )
        ]
        report.add(
            "environment manifest",
            not mismatches,
            "matches requested source contract"
            if not mismatches
            else "; ".join(mismatches),
        )

    try:
        site_packages = _site_packages(env_dir)
        source_paths = site_packages / SOURCE_PATHS_NAME
        lines = (
            [
                line
                for line in source_paths.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if source_paths.is_file()
            else []
        )
        expected_lines = [str(repo_root), str(source.root)]
        report.add(
            "source path file",
            lines == expected_lines,
            str(source_paths)
            if lines == expected_lines
            else f"found {lines}, expected {expected_lines}",
        )
    except (EnvError, OSError) as exc:
        report.add("source path file", False, str(exc))

    probe_ok, probe_detail = _probe_target(repo_root, env_dir, source, profile)
    report.add("target import probe", probe_ok, probe_detail)
    pip_ok, pip_detail = _pip_check(env_dir)
    report.add("target pip check", pip_ok, pip_detail)
    return report


def bootstrap(args: argparse.Namespace) -> Report:
    repo_root, env_dir, megatron_root, expected_ref, profile, manifest = (
        _resolve_configuration(args)
    )
    target = ensure_target_isolation(repo_root, env_dir)
    source = inspect_source(
        repo_root,
        megatron_root,
        expected_ref,
        allow_dirty=args.allow_dirty_source,
    )
    python_pin = _stack_value(repo_root, "python")
    base_python = _resolved(Path(args.python or sys.executable))
    identity = _python_identity(base_python)
    major_minor = ".".join(str(value) for value in identity["version"][:2])
    if major_minor != python_pin:
        raise EnvError(
            f"bootstrap Python {major_minor} does not match STACK.lock {python_pin}"
        )

    if target.exists():
        if manifest is None:
            raise EnvError(
                f"refusing to mutate existing unowned environment without {MANIFEST_NAME}: {target}"
            )
        expected_existing = {
            "schema": MANIFEST_SCHEMA,
            "repo_root": str(repo_root),
            "megatron_root": str(source.root),
            "megatron_commit": source.expected_commit,
            "source_dirty": bool(source.dirty_entries),
            "profile": profile,
        }
        mismatches = [
            key
            for key, _actual, _expected in _manifest_mismatches(
                manifest, expected_existing
            )
        ]
        if mismatches:
            raise EnvError(
                "existing environment manifest belongs to a different contract: "
                + ", ".join(mismatches)
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        _run_bootstrap_venv(base_python, target)

    ensure_target_isolation(repo_root, target)
    target_identity = _python_identity(_target_python(target))
    if _resolved(Path(target_identity["prefix"])) != target:
        raise EnvError(
            f"target interpreter prefix {target_identity['prefix']} does not match {target}"
        )
    site_packages = _site_packages(target)
    write_source_paths(site_packages, repo_root, source.root)
    _write_manifest(
        target,
        {
            "schema": MANIFEST_SCHEMA,
            "repo_root": str(repo_root),
            "megatron_root": str(source.root),
            "megatron_ref": source.expected_ref,
            "megatron_commit": source.expected_commit,
            "megatron_version": source.version,
            "source_dirty": bool(source.dirty_entries),
            "profile": profile,
            "python": python_pin,
            "bootstrap_python": str(base_python),
            "source_paths": SOURCE_PATHS_NAME,
        },
    )

    if args.skip_verify:
        report = Report(title="cppmega environment bootstrap")
        report.add("bootstrap", True, f"created isolated environment {target}")
        report.add("package installation", True, "not attempted by design")
        return report
    return verify(args)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--env", dest="env_dir")
    parser.add_argument("--megatron-root")
    parser.add_argument(
        "--megatron-ref",
        help="Exact commit or git ref. Defaults to the manifest, then STACK.lock.",
    )
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap")
    _add_common_arguments(bootstrap_parser)
    bootstrap_parser.add_argument("--python", help="Base Python; must match STACK.lock")
    bootstrap_parser.add_argument("--skip-verify", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    _add_common_arguments(verify_parser)
    return parser


def _print_report(report: Report, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(report.render())


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = bootstrap(args) if args.command == "bootstrap" else verify(args)
    except EnvError as exc:
        report = Report(title=f"cppmega environment {args.command}")
        report.add(args.command, False, str(exc))
    _print_report(report, json_output=args.json)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
