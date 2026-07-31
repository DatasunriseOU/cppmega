"""Fail-closed direct CI orchestration for repository-owned machines."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import ipaddress
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "cppmega.repository-ci.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOSTS_CONFIG = REPO_ROOT / "configs" / "ci" / "hosts.json"
DEFAULT_LANES_CONFIG = REPO_ROOT / "configs" / "ci" / "lanes.json"
DEFAULT_RECEIPT_BASE = (
    REPO_ROOT / "outputs" / "ci_diagnostics" / "repository-ci"
)

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SAFE_USER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SAFE_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\Z")
_HEX_SHA = re.compile(r"[0-9a-f]{40,64}\Z")
_SECRET_FIELD_NAMES = {
    "authorization",
    "credential",
    "credentials",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
}
_SHELL_EXECUTABLES = {
    "bash",
    "cmd",
    "cmd.exe",
    "dash",
    "fish",
    "powershell",
    "pwsh",
    "sh",
    "zsh",
}
_PACKAGE_MANAGERS = {
    "apt",
    "apt-get",
    "brew",
    "conda",
    "dnf",
    "mamba",
    "npm",
    "pip",
    "pip3",
    "pnpm",
    "uv",
    "yum",
    "yarn",
}
_MUTATING_PACKAGE_ACTIONS = {
    "add",
    "create",
    "install",
    "sync",
    "update",
    "upgrade",
}
_PASSTHROUGH_ENV = {
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CPPMEGA_MEGATRON_COMMIT",
    "CPPMEGA_MLX_REFERENCE_COMMIT",
    "CPPMEGA_TEST_PROFILE",
    "CPPMEGA_MLX_REFERENCE_ROOT",
    "CPPMEGA_RECIPE_PARITY_PEER_ROOT",
    "CPPMEGA_RECIPE_PARITY_PEER_COMMIT",
    "CPPMEGA_RECIPE_PARITY_PYTHON",
    "DYLD_LIBRARY_PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LD_LIBRARY_PATH",
    "MEGATRON_LM_REPO",
    "MEGATRON_ROOT",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "SSH_AUTH_SOCK",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
}
_SECRET_ENV_NAME = re.compile(
    r"(?:AUTH|CREDENTIAL|KEY|PASSWORD|PASSWD|SECRET|TOKEN)", re.IGNORECASE
)
_LABELED_SECRET = re.compile(
    r"(?i)\b(password|passwd|token|secret|api[_-]?key|authorization|credential)"
    r"(\s*[=:]\s*)([^\s,;]+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_COMMON_SECRET = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{16,}|"
    r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{16,})\b"
)


class RepositoryCIError(RuntimeError):
    """Raised when a CI safety or execution contract fails closed."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _new_run_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"direct-{stamp}-{uuid.uuid4().hex[:8]}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_key_fingerprint(public_key: str) -> str:
    try:
        decoded = base64.b64decode(public_key, validate=True)
    except (ValueError, TypeError) as exc:
        raise RepositoryCIError("host public key is not valid base64") from exc
    digest = base64.b64encode(hashlib.sha256(decoded).digest()).decode("ascii")
    return f"SHA256:{digest.rstrip('=')}"


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepositoryCIError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RepositoryCIError(f"{label} must be a JSON object")
    return payload


def _reject_secret_fields(value: Any, *, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _SECRET_FIELD_NAMES:
                raise RepositoryCIError(
                    f"credential field {path}.{key} is forbidden in CI config"
                )
            _reject_secret_fields(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_fields(child, path=f"{path}[{index}]")


def _validate_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise RepositoryCIError(f"invalid {label}: {value!r}")
    return value


def _validate_immutable_command(command: dict[str, Any], *, lane_id: str) -> None:
    name = _validate_id(command.get("name"), label=f"step name in {lane_id}")
    timeout = command.get("timeout_seconds")
    if not isinstance(timeout, int) or timeout <= 0:
        raise RepositoryCIError(f"step {lane_id}/{name} needs a positive timeout")
    argv = command.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(
            isinstance(part, str) and part and "\x00" not in part for part in argv
        )
    ):
        raise RepositoryCIError(f"step {lane_id}/{name} has invalid argv")
    placeholders = {
        match.group(0) for part in argv for match in re.finditer(r"\{[^{}]+\}", part)
    }
    if placeholders - {"{python}"}:
        raise RepositoryCIError(
            f"step {lane_id}/{name} has unsupported placeholders: "
            f"{sorted(placeholders)}"
        )
    executable = Path(argv[0]).name.lower()
    if executable in _SHELL_EXECUTABLES:
        raise RepositoryCIError(
            f"step {lane_id}/{name} may not use a shell interpreter"
        )
    lowered = [part.lower() for part in argv]
    if executable in _PACKAGE_MANAGERS and any(
        action in lowered[1:] for action in _MUTATING_PACKAGE_ACTIONS
    ):
        raise RepositoryCIError(f"step {lane_id}/{name} may not mutate dependencies")
    if (
        len(lowered) >= 4
        and lowered[1:3] == ["-m", "pip"]
        and any(action in lowered[3:] for action in _MUTATING_PACKAGE_ACTIONS)
    ):
        raise RepositoryCIError(f"step {lane_id}/{name} may not mutate dependencies")


def load_lanes(path: Path = DEFAULT_LANES_CONFIG) -> dict[str, dict[str, Any]]:
    payload = _read_json(path, label="lane config")
    _reject_secret_fields(payload)
    if payload.get("schema_version") != 1:
        raise RepositoryCIError("lane config schema_version must be 1")
    rows = payload.get("lanes")
    if not isinstance(rows, list) or not rows:
        raise RepositoryCIError("lane config needs a non-empty lanes list")
    lanes: dict[str, dict[str, Any]] = {}
    for lane in rows:
        if not isinstance(lane, dict):
            raise RepositoryCIError("every lane must be an object")
        lane_id = _validate_id(lane.get("id"), label="lane id")
        if lane_id in lanes:
            raise RepositoryCIError(f"duplicate lane id: {lane_id}")
        system = lane.get("system")
        machines = lane.get("machines")
        if system not in {"darwin", "linux"}:
            raise RepositoryCIError(f"lane {lane_id} has unsupported system")
        if (
            not isinstance(machines, list)
            or not machines
            or not all(isinstance(machine, str) and machine for machine in machines)
        ):
            raise RepositoryCIError(f"lane {lane_id} has invalid machines")
        modules = lane.get("required_modules")
        if not isinstance(modules, list) or not all(
            isinstance(module, str) and _SAFE_MODULE.fullmatch(module)
            for module in modules
        ):
            raise RepositoryCIError(f"lane {lane_id} has invalid required_modules")
        timeout = lane.get("timeout_seconds")
        if not isinstance(timeout, int) or timeout <= 0:
            raise RepositoryCIError(f"lane {lane_id} needs a positive timeout")
        test_profile = lane.get("test_profile")
        if test_profile is not None and (
            not isinstance(test_profile, str)
            or not test_profile
            or _SAFE_ID.fullmatch(test_profile) is None
        ):
            raise RepositoryCIError(f"lane {lane_id} has invalid test_profile")
        commands = lane.get("commands")
        if not isinstance(commands, list) or not commands:
            raise RepositoryCIError(f"lane {lane_id} needs commands")
        for command in commands:
            if not isinstance(command, dict):
                raise RepositoryCIError(f"lane {lane_id} command must be an object")
            _validate_immutable_command(command, lane_id=lane_id)
            if int(command["timeout_seconds"]) > timeout:
                raise RepositoryCIError(
                    f"step {lane_id}/{command['name']} exceeds the lane timeout"
                )
        lanes[lane_id] = lane
    return lanes


def load_hosts(
    path: Path = DEFAULT_HOSTS_CONFIG,
    *,
    lanes: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lane_map = lanes if lanes is not None else load_lanes()
    payload = _read_json(path, label="host config")
    _reject_secret_fields(payload)
    if payload.get("schema_version") != 1:
        raise RepositoryCIError("host config schema_version must be 1")
    rows = payload.get("hosts")
    if not isinstance(rows, list) or not rows:
        raise RepositoryCIError("host config needs a non-empty hosts list")
    seen: set[str] = set()
    for host in rows:
        if not isinstance(host, dict):
            raise RepositoryCIError("every host must be an object")
        host_id = _validate_id(host.get("id"), label="host id")
        if host_id in seen:
            raise RepositoryCIError(f"duplicate host id: {host_id}")
        seen.add(host_id)
        transport = host.get("transport")
        if transport not in {"local", "ssh"}:
            raise RepositoryCIError(f"host {host_id} has unsupported transport")
        system = host.get("system")
        machines = host.get("machines")
        if not isinstance(system, str) or not system:
            raise RepositoryCIError(f"host {host_id} has no system")
        if (
            not isinstance(machines, list)
            or not machines
            or not all(isinstance(machine, str) and machine for machine in machines)
        ):
            raise RepositoryCIError(f"host {host_id} has invalid machines")
        python = host.get("python")
        if not isinstance(python, str) or not python or "\x00" in python:
            raise RepositoryCIError(f"host {host_id} has invalid python")
        host_lanes = host.get("lanes")
        if not isinstance(host_lanes, list) or not all(
            isinstance(lane_id, str) and lane_id in lane_map for lane_id in host_lanes
        ):
            raise RepositoryCIError(f"host {host_id} references an unknown lane")
        for lane_id in host_lanes:
            lane = lane_map[lane_id]
            if lane["system"] != str(system).lower() or not {
                str(value).lower() for value in machines
            }.intersection(str(value).lower() for value in lane["machines"]):
                raise RepositoryCIError(
                    f"host {host_id} platform cannot run lane {lane_id}"
                )
        dispatch_enabled = host.get("dispatch_enabled")
        required = host.get("required")
        if not isinstance(dispatch_enabled, bool) or not isinstance(required, bool):
            raise RepositoryCIError(
                f"host {host_id} needs boolean dispatch_enabled and required"
            )
        if dispatch_enabled and not host_lanes:
            raise RepositoryCIError(f"dispatch host {host_id} has no lanes")
        if required and not dispatch_enabled:
            raise RepositoryCIError(f"required host {host_id} has dispatch disabled")
        if transport == "local":
            if host.get("address") != "local":
                raise RepositoryCIError(f"local host {host_id} address must be local")
            continue
        try:
            ipaddress.ip_address(str(host.get("address")))
        except ValueError as exc:
            raise RepositoryCIError(
                f"host {host_id} needs a literal IP address"
            ) from exc
        user = host.get("user")
        port = host.get("port")
        if not isinstance(user, str) or _SAFE_USER.fullmatch(user) is None:
            raise RepositoryCIError(f"host {host_id} has invalid SSH user")
        if not isinstance(port, int) or not (1 <= port <= 65535):
            raise RepositoryCIError(f"host {host_id} has invalid SSH port")
        trust = host.get("trust")
        if trust not in {"trusted", "quarantined"}:
            raise RepositoryCIError(f"host {host_id} has invalid trust state")
        if dispatch_enabled and trust != "trusted":
            raise RepositoryCIError(
                f"dispatch host {host_id} does not have a trusted identity"
            )
        if dispatch_enabled and host.get("remote_shell") != "posix":
            raise RepositoryCIError(
                f"dispatch host {host_id} must provide a POSIX remote shell"
            )
        if trust == "trusted":
            host_key = host.get("host_key")
            if not isinstance(host_key, dict):
                raise RepositoryCIError(f"trusted host {host_id} has no host key")
            algorithm = host_key.get("algorithm")
            public_key = host_key.get("public_key")
            fingerprint = host_key.get("fingerprint_sha256")
            if algorithm != "ssh-ed25519" or not isinstance(public_key, str):
                raise RepositoryCIError(
                    f"trusted host {host_id} needs a pinned ED25519 key"
                )
            if fingerprint != _host_key_fingerprint(public_key):
                raise RepositoryCIError(
                    f"trusted host {host_id} fingerprint does not match its key"
                )
    return payload, rows


def _git_run(repo_root: Path, *args: str, timeout: int = 20) -> bytes:
    try:
        result = subprocess.run(
            ("git", "-C", str(repo_root), *args),
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RepositoryCIError(f"git command failed: {' '.join(args)}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RepositoryCIError(
            f"git command failed ({result.returncode}): {' '.join(args)}: {detail}"
        )
    return result.stdout


def _git_optional(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(repo_root), *args),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def capture_provenance(repo_root: Path) -> dict[str, Any]:
    """Capture commit, tree, index/worktree, and worktree-identity fingerprints."""

    root = Path(
        _git_run(repo_root, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    git_dir = Path(
        _git_run(repo_root, "rev-parse", "--absolute-git-dir").decode().strip()
    ).resolve()
    head = _git_run(repo_root, "rev-parse", "--verify", "HEAD").decode().strip()
    tree = _git_run(repo_root, "rev-parse", "HEAD^{tree}").decode().strip()
    status = _git_run(
        repo_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        timeout=60,
    )
    diff = _git_run(
        repo_root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
        timeout=60,
    )
    branch = _git_optional(repo_root, "symbolic-ref", "--short", "-q", "HEAD")
    worktree_material = f"{root}\0{git_dir}".encode("utf-8")
    return {
        "captured_at": _utc_now(),
        "head_commit": head,
        "head_tree": tree,
        "branch": branch or "detached",
        "dirty": bool(status),
        "change_count": len([entry for entry in status.split(b"\0") if entry]),
        "status_sha256": _sha256_bytes(status),
        "diff_sha256": _sha256_bytes(diff),
        "worktree_id_sha256": _sha256_bytes(worktree_material),
    }


def provenance_unchanged(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> bool:
    if before is None or after is None:
        return False
    fields = (
        "head_commit",
        "head_tree",
        "branch",
        "dirty",
        "change_count",
        "status_sha256",
        "diff_sha256",
        "worktree_id_sha256",
    )
    return all(before.get(field) == after.get(field) for field in fields)


class _Redactor:
    def __init__(self) -> None:
        self._known_values = sorted(
            {
                value
                for name, value in os.environ.items()
                if _SECRET_ENV_NAME.search(name) and len(value) >= 8
            },
            key=len,
            reverse=True,
        )
        self._inside_private_key = False

    def line(self, value: str) -> str:
        if "-----BEGIN " in value and "PRIVATE KEY-----" in value:
            self._inside_private_key = True
            return "[REDACTED PRIVATE KEY]\n"
        if self._inside_private_key:
            if "-----END " in value and "PRIVATE KEY-----" in value:
                self._inside_private_key = False
            return ""
        redacted = value
        for known in self._known_values:
            redacted = redacted.replace(known, "<redacted>")
        redacted = _LABELED_SECRET.sub(r"\1\2<redacted>", redacted)
        redacted = _BEARER_SECRET.sub("Bearer <redacted>", redacted)
        redacted = _COMMON_SECRET.sub("<redacted>", redacted)
        return redacted

    def text(self, value: str) -> str:
        return "".join(self.line(line) for line in value.splitlines(keepends=True))


def _sanitized_environment(
    repo_root: Path,
    *,
    environment_overrides: dict[str, str | None] | None = None,
) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if key in _PASSTHROUGH_ENV
    }
    environment.update(
        {
            "CI": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(repo_root),
            "PYTHONSAFEPATH": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    for key, value in (environment_overrides or {}).items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    return environment


def _lane_environment_overrides(
    lane: dict[str, Any],
) -> dict[str, str | None] | None:
    return {"CPPMEGA_TEST_PROFILE": lane.get("test_profile")}


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()
    process.wait(timeout=10)


def _tail(path: Path, line_count: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


def run_step(
    *,
    name: str,
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    timeout_seconds: float,
    display_command: Sequence[str] | None = None,
    environment_overrides: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Run one command with a hard timeout and a redacted combined log."""

    if timeout_seconds <= 0:
        raise RepositoryCIError(f"step {name!r} has no time remaining")
    if not command:
        raise RepositoryCIError(f"step {name!r} has no command")
    redactor = _Redactor()
    shown = tuple(display_command or command)
    rendered = _Redactor().text(shlex.join(str(part) for part in shown))
    print(f"[repository-ci] start {name}: {rendered}", flush=True)
    started_at = _utc_now()
    started = time.monotonic()
    timed_out = False
    reader_errors: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            tuple(str(part) for part in command),
            cwd=cwd,
            env=_sanitized_environment(
                cwd, environment_overrides=environment_overrides
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None

        def consume_output() -> None:
            try:
                for line in process.stdout:
                    log_handle.write(redactor.line(line))
                    log_handle.flush()
            except (OSError, ValueError) as exc:
                reader_errors.append(str(exc))

        reader = threading.Thread(target=consume_output, daemon=True)
        reader.start()
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            exit_code = 124
        reader.join(timeout=5)
        process.stdout.close()
    duration = round(time.monotonic() - started, 3)
    if reader_errors and exit_code == 0:
        exit_code = 1
    status = "timed_out" if timed_out else ("passed" if exit_code == 0 else "failed")
    print(
        f"[repository-ci] {status} {name} in {duration:.3f}s (log: {log_path})",
        flush=True,
    )
    if exit_code != 0:
        tail = _tail(log_path)
        if tail:
            print(f"[repository-ci] {name} log tail:\n{tail}", file=sys.stderr)
    return {
        "name": name,
        "command": [_Redactor().text(str(part)) for part in shown],
        "status": status,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": duration,
        "timeout_seconds": round(float(timeout_seconds), 3),
        "log": log_path.name,
        "output_reader_error": reader_errors[0] if reader_errors else None,
    }


def _resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or "/" in value:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RepositoryCIError(f"executable is unavailable: {candidate}")
        return str(candidate)
    resolved = shutil.which(value)
    if resolved is None:
        raise RepositoryCIError(f"executable is unavailable on PATH: {value}")
    return resolved


def _render_command(command: dict[str, Any], *, python: str) -> tuple[str, ...]:
    return tuple(str(part).replace("{python}", python) for part in command["argv"])


def _probe_code(
    modules: Iterable[str], *, check_cuda: bool, tools: Iterable[str]
) -> str:
    module_list = sorted(set(modules))
    tool_list = sorted(set(tools))
    return f"""
import importlib.util
import json
import platform
import shutil
import sys

modules = {module_list!r}
module_status = {{}}
for name in modules:
    try:
        module_status[name] = importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        module_status[name] = False

cuda = {{"checked": {check_cuda!r}, "available": None, "device_count": None}}
if cuda["checked"]:
    try:
        import torch
        cuda["available"] = bool(torch.cuda.is_available())
        cuda["device_count"] = int(torch.cuda.device_count())
    except Exception:
        cuda["available"] = False
        cuda["device_count"] = 0

print(json.dumps({{
    "hostname": platform.node(),
    "system": platform.system().lower(),
    "release": platform.release(),
    "machine": platform.machine().lower(),
    "python": platform.python_version(),
    "python_ok": sys.version_info >= (3, 10),
    "modules": module_status,
    "tools": {{name: shutil.which(name) is not None for name in {tool_list!r}}},
    "cuda": cuda,
}}, sort_keys=True))
""".strip()


def _parse_probe_payload(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RepositoryCIError("host probe returned no JSON object")


def _lane_probe_status(
    host: dict[str, Any],
    lane: dict[str, Any],
    observed: dict[str, Any] | None,
    *,
    transport_ready: bool,
) -> dict[str, Any]:
    if not transport_ready or observed is None:
        return {
            "available": False,
            "status": "transport_unavailable",
            "missing_modules": list(lane["required_modules"]),
            "missing_tools": [],
            "cuda_available": None,
        }
    systems_match = observed.get("system") == str(lane["system"]).lower()
    machines_match = str(observed.get("machine", "")).lower() in {
        str(machine).lower() for machine in lane["machines"]
    }
    modules = (
        observed.get("modules") if isinstance(observed.get("modules"), dict) else {}
    )
    tools = observed.get("tools") if isinstance(observed.get("tools"), dict) else {}
    missing_modules = [
        module for module in lane["required_modules"] if modules.get(module) is not True
    ]
    missing_tools = [name for name, present in tools.items() if present is not True]
    cuda_payload = (
        observed.get("cuda") if isinstance(observed.get("cuda"), dict) else {}
    )
    cuda_available = cuda_payload.get("available")
    cuda_ok = not lane.get("requires_cuda") or cuda_available is True
    python_ok = observed.get("python_ok") is True
    available = bool(
        systems_match
        and machines_match
        and python_ok
        and not missing_modules
        and not missing_tools
        and cuda_ok
    )
    if available:
        status = "available"
    elif not systems_match or not machines_match:
        status = "platform_mismatch"
    elif not python_ok:
        status = "python_version_unsupported"
    elif missing_modules:
        status = "missing_preprovisioned_modules"
    elif missing_tools:
        status = "missing_preprovisioned_tools"
    else:
        status = "cuda_unavailable"
    return {
        "available": available,
        "status": status,
        "missing_modules": missing_modules,
        "missing_tools": missing_tools,
        "cuda_available": cuda_available,
    }


def _scan_host_key(host: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
    algorithm = (
        host.get("host_key", {}).get("algorithm")
        if host.get("trust") == "trusted"
        else "ssh-ed25519"
    )
    command = (
        "ssh-keyscan",
        "-T",
        str(timeout_seconds),
        "-p",
        str(host["port"]),
        "-t",
        str(algorithm).removeprefix("ssh-"),
        str(host["address"]),
    )
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 3,
        )
    except subprocess.TimeoutExpired:
        return {"status": "keyscan_timeout", "fingerprint_sha256": None, "key": None}
    except OSError as exc:
        return {
            "status": "keyscan_error",
            "fingerprint_sha256": None,
            "key": None,
            "detail": str(exc),
        }
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[1] != algorithm:
            continue
        try:
            fingerprint = _host_key_fingerprint(parts[2])
        except RepositoryCIError:
            continue
        return {
            "status": "observed",
            "fingerprint_sha256": fingerprint,
            "key": parts[2],
        }
    return {
        "status": "keyscan_failed",
        "fingerprint_sha256": None,
        "key": None,
        "exit_code": result.returncode,
    }


def _known_hosts_line(host: dict[str, Any]) -> str:
    marker = (
        str(host["address"])
        if int(host["port"]) == 22
        else f"[{host['address']}]:{host['port']}"
    )
    host_key = host["host_key"]
    return f"{marker} {host_key['algorithm']} {host_key['public_key']}\n"


def _write_known_hosts(host: dict[str, Any], directory: Path) -> Path:
    path = directory / f"{host['id']}.known_hosts"
    path.write_text(_known_hosts_line(host), encoding="ascii")
    path.chmod(0o600)
    return path


def _identity_file_from_environment() -> Path | None:
    value = os.environ.get("CPPMEGA_CI_SSH_IDENTITY_FILE")
    if not value:
        return None
    identity = Path(value).expanduser()
    if not identity.is_file() or not os.access(identity, os.R_OK):
        raise RepositoryCIError(
            "CPPMEGA_CI_SSH_IDENTITY_FILE does not name a readable file"
        )
    return identity


def _ssh_base(
    host: dict[str, Any],
    *,
    known_hosts: Path,
    connect_timeout: int,
    identity_file: Path | None,
) -> tuple[str, ...]:
    command = [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        "VerifyHostKeyDNS=no",
        "-o",
        f"HostKeyAlgorithms={host['host_key']['algorithm']}",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "RequestTTY=no",
        "-o",
        "LogLevel=ERROR",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        "ConnectionAttempts=1",
        "-p",
        str(host["port"]),
    ]
    if identity_file:
        command.extend(("-o", "IdentitiesOnly=yes", "-i", str(identity_file)))
    command.append(f"{host['user']}@{host['address']}")
    return tuple(command)


def _host_identity_decision(
    host: dict[str, Any], scan: dict[str, Any]
) -> dict[str, Any]:
    if host.get("trust") != "trusted":
        return {
            "may_authenticate": False,
            "status": "quarantined",
            "detail": "untrusted_host_identity",
            "identity_verified": False,
        }
    if scan.get("status") != "observed":
        return {
            "may_authenticate": False,
            "status": "unavailable",
            "detail": str(scan.get("status")),
            "identity_verified": False,
        }
    if scan.get("key") != host["host_key"]["public_key"]:
        return {
            "may_authenticate": False,
            "status": "unavailable",
            "detail": "host_key_mismatch",
            "identity_verified": False,
        }
    return {
        "may_authenticate": True,
        "status": "unavailable",
        "detail": None,
        "identity_verified": True,
    }


def classify_ssh_failure(stderr: str) -> str:
    lowered = stderr.lower()
    if (
        "host key verification failed" in lowered
        or "remote host identification" in lowered
    ):
        return "host_key_verification_failed"
    if "permission denied" in lowered:
        return "ssh_authentication_failed"
    if "connection refused" in lowered:
        return "ssh_connection_refused"
    if "no route to host" in lowered or "network is unreachable" in lowered:
        return "network_unreachable"
    if "timed out" in lowered or "operation timed out" in lowered:
        return "ssh_timeout"
    return "ssh_probe_failed"


def _remote_command(host: dict[str, Any], argv: Sequence[str]) -> str:
    if host.get("remote_shell") == "windows":
        return subprocess.list2cmdline([str(part) for part in argv])
    return shlex.join(str(part) for part in argv)


def _windows_inventory_probe_argv() -> tuple[str, ...]:
    command = (
        "$payload = [ordered]@{"
        "hostname=$env:COMPUTERNAME;"
        "system='windows';"
        "release=[Environment]::OSVersion.VersionString;"
        "machine=$env:PROCESSOR_ARCHITECTURE.ToLowerInvariant();"
        "python='not-required';"
        "python_ok=$true;"
        "modules=@{};"
        "tools=@{};"
        "cuda=[ordered]@{checked=$false;available=$null;device_count=$null}"
        "}; $payload | ConvertTo-Json -Compress -Depth 4"
    )
    return (
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        command,
    )


def probe_host(
    host: dict[str, Any],
    *,
    lanes: dict[str, dict[str, Any]],
    selected_lanes: Sequence[str],
    known_hosts_dir: Path,
    connect_timeout: int,
) -> dict[str, Any]:
    started = time.monotonic()
    modules = {
        module
        for lane_id in selected_lanes
        for module in lanes[lane_id]["required_modules"]
    }
    check_cuda = any(lanes[lane_id].get("requires_cuda") for lane_id in selected_lanes)
    tools = {"git"}
    if host["transport"] == "ssh" and host.get("dispatch_enabled"):
        tools.update(("bash", "sha256sum", "tar"))
    probe: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "host-preflight",
        "host_id": host["id"],
        "address": host["address"],
        "transport": host["transport"],
        "trust": host.get("trust", "local"),
        "dispatch_enabled": bool(host["dispatch_enabled"]),
        "required": bool(host["required"]),
        "selected_lanes": list(selected_lanes),
        "started_at": _utc_now(),
        "status": "unavailable",
        "detail": None,
        "identity_verified": host["transport"] == "local",
        "observed_host_key_fingerprint": None,
        "observed": None,
        "lane_status": {},
    }
    observed: dict[str, Any] | None = None
    transport_ready = False
    try:
        code = _probe_code(modules, check_cuda=check_cuda, tools=tools)
        if host["transport"] == "local":
            python = _resolve_executable(str(host["python"]))
            result = subprocess.run(
                (python, "-c", code),
                cwd=REPO_ROOT,
                env=_sanitized_environment(REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
                timeout=connect_timeout + 10,
            )
        else:
            scan = _scan_host_key(host, timeout_seconds=connect_timeout)
            probe["observed_host_key_fingerprint"] = scan.get("fingerprint_sha256")
            identity = _host_identity_decision(host, scan)
            probe["status"] = identity["status"]
            probe["detail"] = identity["detail"]
            probe["identity_verified"] = identity["identity_verified"]
            if not identity["may_authenticate"]:
                return probe
            known_hosts = _write_known_hosts(host, known_hosts_dir)
            ssh = _ssh_base(
                host,
                known_hosts=known_hosts,
                connect_timeout=connect_timeout,
                identity_file=_identity_file_from_environment(),
            )
            probe_argv = (
                _windows_inventory_probe_argv()
                if host.get("remote_shell") == "windows" and not selected_lanes
                else (str(host["python"]), "-c", code)
            )
            remote = _remote_command(host, probe_argv)
            result = subprocess.run(
                (*ssh, remote),
                check=False,
                capture_output=True,
                text=True,
                timeout=connect_timeout + 15,
            )
        if result.returncode != 0:
            probe["detail"] = (
                classify_ssh_failure(result.stderr)
                if host["transport"] == "ssh"
                else "local_probe_failed"
            )
        else:
            observed = _parse_probe_payload(result.stdout)
            probe["observed"] = observed
            expected_system = str(host["system"]).lower()
            expected_machines = {str(value).lower() for value in host["machines"]}
            platform_ok = (
                observed.get("system") == expected_system
                and str(observed.get("machine", "")).lower() in expected_machines
                and observed.get("python_ok") is True
            )
            if platform_ok:
                transport_ready = True
                probe["status"] = "available"
                probe["detail"] = "preflight_completed"
            else:
                probe["detail"] = "platform_or_python_mismatch"
    except subprocess.TimeoutExpired:
        probe["detail"] = "probe_timeout"
    except (OSError, RepositoryCIError, ValueError) as exc:
        probe["detail"] = f"probe_error: {exc}"
    finally:
        for lane_id in selected_lanes:
            probe["lane_status"][lane_id] = _lane_probe_status(
                host,
                lanes[lane_id],
                observed,
                transport_ready=transport_ready,
            )
        probe["completed_at"] = _utc_now()
        probe["duration_seconds"] = round(time.monotonic() - started, 3)
    return probe


def run_lane(args: argparse.Namespace) -> int:
    lanes = load_lanes(Path(args.lanes_config).resolve())
    lane = lanes[args.lane]
    repo_root = Path(args.repo_root).resolve()
    receipt_dir = Path(args.receipt_dir).resolve()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or _new_run_id()
    started_at = _utc_now()
    started = time.monotonic()
    deadline = started + int(lane["timeout_seconds"])
    steps: list[dict[str, Any]] = []
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    status = "failed"
    error: str | None = None
    exit_code = 1
    python = args.python
    environment_overrides = _lane_environment_overrides(lane)

    def remaining(limit: int) -> float:
        return max(0.0, min(float(limit), deadline - time.monotonic()))

    try:
        if not repo_root.is_dir():
            raise RepositoryCIError(f"repository root is unavailable: {repo_root}")
        actual_system = platform.system().lower()
        actual_machine = platform.machine().lower()
        if actual_system != lane["system"] or actual_machine not in {
            str(value).lower() for value in lane["machines"]
        }:
            raise RepositoryCIError(
                f"lane {lane['id']} requires {lane['system']}/{lane['machines']}; "
                f"host is {actual_system}/{actual_machine}"
            )
        python = _resolve_executable(args.python)
        before = capture_provenance(repo_root)
        if (
            args.expected_source_commit
            and before["head_commit"] != args.expected_source_commit
        ):
            raise RepositoryCIError(
                "staged worktree commit does not match the requested source commit: "
                f"expected {args.expected_source_commit}, got {before['head_commit']}"
            )
        if (
            args.expected_source_tree
            and before["head_tree"] != args.expected_source_tree
        ):
            raise RepositoryCIError(
                "staged worktree tree does not match the requested source tree"
            )
        preflight = run_step(
            name="preprovisioned-environment",
            command=(
                python,
                "-c",
                _probe_code(
                    lane["required_modules"],
                    check_cuda=bool(lane.get("requires_cuda")),
                    tools=("git",),
                ),
            ),
            cwd=repo_root,
            log_path=receipt_dir / "preprovisioned-environment.log",
            timeout_seconds=remaining(60),
            environment_overrides=environment_overrides,
        )
        steps.append(preflight)
        if preflight["exit_code"] != 0:
            status = preflight["status"]
            exit_code = 124 if preflight["timed_out"] else 1
        else:
            payload = _parse_probe_payload(
                (receipt_dir / "preprovisioned-environment.log").read_text(
                    encoding="utf-8"
                )
            )
            capability = _lane_probe_status(
                {"system": lane["system"]},
                lane,
                payload,
                transport_ready=True,
            )
            if not capability["available"]:
                raise RepositoryCIError(
                    f"preprovisioned environment is insufficient: {capability['status']}"
                )
            for command in lane["commands"]:
                step = run_step(
                    name=str(command["name"]),
                    command=_render_command(command, python=python),
                    cwd=repo_root,
                    log_path=receipt_dir / f"{command['name']}.log",
                    timeout_seconds=remaining(int(command["timeout_seconds"])),
                    environment_overrides=environment_overrides,
                )
                steps.append(step)
                if step["exit_code"] != 0:
                    status = step["status"]
                    exit_code = 124 if step["timed_out"] else 1
                    break
            else:
                status = "passed"
                exit_code = 0
    except (OSError, RepositoryCIError, subprocess.SubprocessError) as exc:
        error = str(exc)
        status = "failed"
        exit_code = 1
        print(f"[repository-ci] lane failed closed: {error}", file=sys.stderr)
    finally:
        try:
            after = capture_provenance(repo_root)
        except RepositoryCIError as exc:
            error = error or f"post-test provenance failed: {exc}"
        unchanged = provenance_unchanged(before, after)
        if before is not None and not unchanged:
            status = "failed_provenance_changed"
            exit_code = 1
            error = error or "commit/worktree provenance changed during the lane"
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": "lane",
            "run_id": run_id,
            "lane": lane["id"],
            "status": status,
            "exit_code": exit_code,
            "error": error,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "timeout_seconds": lane["timeout_seconds"],
            "source": {
                "requested_commit": args.expected_source_commit,
                "requested_tree": args.expected_source_tree,
                "archive_sha256": args.archive_sha256,
            },
            "provenance": {
                "before_tests": before,
                "after_tests": after,
                "unchanged": unchanged,
            },
            "host": {
                "hostname": socket.gethostname(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "python_executable": python,
            "steps": steps,
        }
        _write_json(receipt_dir / "receipt.json", receipt)
        print(f"[repository-ci] receipt: {receipt_dir / 'receipt.json'}", flush=True)
    return exit_code


def _safe_extract(archive_path: Path, destination: Path, *, mode: str = "r:") -> None:
    with tarfile.open(archive_path, mode=mode) as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RepositoryCIError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                target = Path(member.linkname)
                if target.is_absolute() or ".." in target.parts:
                    raise RepositoryCIError(
                        f"unsafe archive link target: {member.linkname}"
                    )
            if not (
                member.isfile() or member.isdir() or member.issym() or member.islnk()
            ):
                raise RepositoryCIError(
                    f"archive contains a special file: {member.name}"
                )
        archive.extractall(destination, members=members)


def _run_checked(
    command: Sequence[str],
    *,
    timeout: int,
    cwd: Path | None = None,
    stdin: Any = None,
    stdout: Any = subprocess.PIPE,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    try:
        result = subprocess.run(
            tuple(str(part) for part in command),
            cwd=cwd,
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=text,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RepositoryCIError("bounded command failed to complete") from exc
    if result.returncode != 0:
        stderr = result.stderr
        if isinstance(stderr, bytes):
            detail = stderr.decode("utf-8", errors="replace")
        else:
            detail = str(stderr or "")
        detail = _Redactor().text(detail[-2000:])
        raise RepositoryCIError(
            f"bounded command failed with exit {result.returncode}: {detail}"
        )
    return result


def _initialize_staged_worktree(checkout: Path, *, expected_tree: str) -> str:
    _run_checked(("git", "init", "-q", str(checkout)), timeout=30)
    _run_checked(
        ("git", "-C", str(checkout), "config", "core.autocrlf", "false"),
        timeout=15,
    )
    _run_checked(
        ("git", "-C", str(checkout), "config", "core.filemode", "true"),
        timeout=15,
    )
    _run_checked(("git", "-C", str(checkout), "add", "-f", "-A"), timeout=120)
    tree = (
        _run_checked(("git", "-C", str(checkout), "write-tree"), timeout=30)
        .stdout.decode()
        .strip()
    )
    if tree != expected_tree:
        raise RepositoryCIError(
            f"staged source tree mismatch: expected {expected_tree}, got {tree}"
        )
    environment = _sanitized_environment(checkout)
    environment.update(
        {
            "GIT_AUTHOR_NAME": "cppmega-repository-ci",
            "GIT_AUTHOR_EMAIL": "runner@cppmega.invalid",
            "GIT_COMMITTER_NAME": "cppmega-repository-ci",
            "GIT_COMMITTER_EMAIL": "runner@cppmega.invalid",
        }
    )
    result = subprocess.run(
        ("git", "-C", str(checkout), "commit-tree", tree, "-m", "CI source snapshot"),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RepositoryCIError("cannot create staged source commit")
    commit = result.stdout.strip()
    _run_checked(
        (
            "git",
            "-C",
            str(checkout),
            "update-ref",
            "refs/heads/ci-snapshot",
            commit,
        ),
        timeout=15,
    )
    _run_checked(
        (
            "git",
            "-C",
            str(checkout),
            "symbolic-ref",
            "HEAD",
            "refs/heads/ci-snapshot",
        ),
        timeout=15,
    )
    return commit


def _create_source_archive(
    repo_root: Path,
    *,
    commit: str,
    destination: Path,
) -> str:
    with destination.open("wb") as handle:
        _run_checked(
            ("git", "-C", str(repo_root), "archive", "--format=tar", commit),
            timeout=180,
            stdout=handle,
        )
    return _sha256_file(destination)


def _lane_cli(
    host: dict[str, Any],
    lane_id: str,
    *,
    repo_root: str,
    receipt_dir: str,
    run_id: str,
    source_commit: str,
    source_tree: str,
    archive_sha256: str,
) -> tuple[str, ...]:
    return (
        str(host["python"]),
        "scripts/ci/run_repository_ci.py",
        "lane",
        "--lane",
        lane_id,
        "--repo-root",
        repo_root,
        "--receipt-dir",
        receipt_dir,
        "--run-id",
        run_id,
        "--python",
        str(host["python"]),
        "--expected-source-commit",
        source_commit,
        "--expected-source-tree",
        source_tree,
        "--archive-sha256",
        archive_sha256,
    )


def _fallback_lane_receipt(
    host: dict[str, Any],
    lane_id: str,
    *,
    run_id: str,
    status: str,
    error: str,
    source_commit: str,
    source_tree: str,
    archive_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "lane",
        "run_id": run_id,
        "lane": lane_id,
        "inventory_host_id": host["id"],
        "status": status,
        "exit_code": 124 if status == "timed_out" else 1,
        "error": _Redactor().text(error),
        "started_at": _utc_now(),
        "completed_at": _utc_now(),
        "source": {
            "requested_commit": source_commit,
            "requested_tree": source_tree,
            "archive_sha256": archive_sha256,
        },
        "provenance": {
            "before_tests": None,
            "after_tests": None,
            "unchanged": False,
        },
        "steps": [],
    }


def _run_local_lane(
    host: dict[str, Any],
    lane_id: str,
    *,
    archive_path: Path,
    archive_sha256: str,
    source_commit: str,
    source_tree: str,
    output_dir: Path,
    run_id: str,
    lane_timeout: int,
) -> dict[str, Any]:
    staging = Path(tempfile.mkdtemp(prefix="cppmega-ci-local-"))
    checkout = staging / "checkout"
    checkout.mkdir()
    try:
        _safe_extract(archive_path, checkout)
        _initialize_staged_worktree(checkout, expected_tree=source_tree)
        command = _lane_cli(
            host,
            lane_id,
            repo_root=str(checkout),
            receipt_dir=str(output_dir),
            run_id=run_id,
            source_commit=source_commit,
            source_tree=source_tree,
            archive_sha256=archive_sha256,
        )
        transport = run_step(
            name=f"dispatch-{host['id']}-{lane_id}",
            command=command,
            cwd=checkout,
            log_path=output_dir / "transport.log",
            timeout_seconds=lane_timeout + 30,
        )
        receipt_path = output_dir / "receipt.json"
        if not receipt_path.is_file():
            raise RepositoryCIError("local lane did not produce receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["inventory_host_id"] = host["id"]
        receipt["transport"] = transport
        _write_json(receipt_path, receipt)
        return receipt
    except (OSError, RepositoryCIError, json.JSONDecodeError) as exc:
        receipt = _fallback_lane_receipt(
            host,
            lane_id,
            run_id=run_id,
            status="failed",
            error=str(exc),
            source_commit=source_commit,
            source_tree=source_tree,
            archive_sha256=archive_sha256,
        )
        _write_json(output_dir / "receipt.json", receipt)
        return receipt
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _stage_remote_archive(
    host: dict[str, Any],
    *,
    ssh: Sequence[str],
    archive_path: Path,
    archive_sha256: str,
    source_tree: str,
) -> str:
    script = f"""
set -euo pipefail
umask 077
root="$(mktemp -d /tmp/cppmega-ci.XXXXXX)"
checkout="$root/checkout"
mkdir -p "$checkout"
cat > "$root/source.tar"
printf '%s  %s\n' {shlex.quote(archive_sha256)} "$root/source.tar" | sha256sum -c - >/dev/null
tar -xf "$root/source.tar" -C "$checkout"
git init -q "$checkout"
git -C "$checkout" config core.autocrlf false
git -C "$checkout" config core.filemode true
git -C "$checkout" add -f -A
tree="$(git -C "$checkout" write-tree)"
test "$tree" = {shlex.quote(source_tree)}
commit="$(printf '%s\n' 'CI source snapshot' | \
  GIT_AUTHOR_NAME=cppmega-repository-ci \
  GIT_AUTHOR_EMAIL=runner@cppmega.invalid \
  GIT_COMMITTER_NAME=cppmega-repository-ci \
  GIT_COMMITTER_EMAIL=runner@cppmega.invalid \
  git -C "$checkout" commit-tree "$tree")"
git -C "$checkout" update-ref refs/heads/ci-snapshot "$commit"
git -C "$checkout" symbolic-ref HEAD refs/heads/ci-snapshot
rm -f "$root/source.tar"
printf '%s\n' "$root"
""".strip()
    with archive_path.open("rb") as archive_handle:
        result = _run_checked(
            (*ssh, "bash -lc " + shlex.quote(script)),
            timeout=180,
            stdin=archive_handle,
        )
    remote_root = result.stdout.decode("utf-8", errors="replace").strip()
    if re.fullmatch(r"/tmp/cppmega-ci\.[A-Za-z0-9]+", remote_root) is None:
        raise RepositoryCIError("remote staging returned an unsafe path")
    return remote_root


def _extract_receipt_archive(archive_path: Path, output_dir: Path) -> None:
    _safe_extract(archive_path, output_dir, mode="r:gz")


def _run_remote_lane(
    host: dict[str, Any],
    lane_id: str,
    *,
    ssh: Sequence[str],
    archive_path: Path,
    archive_sha256: str,
    source_commit: str,
    source_tree: str,
    output_dir: Path,
    run_id: str,
    lane_timeout: int,
) -> dict[str, Any]:
    remote_root: str | None = None
    try:
        remote_root = _stage_remote_archive(
            host,
            ssh=ssh,
            archive_path=archive_path,
            archive_sha256=archive_sha256,
            source_tree=source_tree,
        )
        remote_receipts = f"{remote_root}/receipts/{lane_id}"
        command = _lane_cli(
            host,
            lane_id,
            repo_root=f"{remote_root}/checkout",
            receipt_dir=remote_receipts,
            run_id=run_id,
            source_commit=source_commit,
            source_tree=source_tree,
            archive_sha256=archive_sha256,
        )
        remote_script = (
            f"set -euo pipefail; cd {shlex.quote(remote_root + '/checkout')}; "
            f"exec {shlex.join(command)}"
        )
        transport = run_step(
            name=f"dispatch-{host['id']}-{lane_id}",
            command=(*ssh, "bash -lc " + shlex.quote(remote_script)),
            display_command=(
                "ssh",
                "<identity-and-host-key-pinned>",
                f"{host['user']}@{host['address']}",
                f"run {lane_id}",
            ),
            cwd=REPO_ROOT,
            log_path=output_dir / "transport.log",
            timeout_seconds=lane_timeout + 30,
        )
        archive_out = output_dir / "receipts.tar.gz"
        collect = (
            f"set -euo pipefail; test -f {shlex.quote(remote_receipts + '/receipt.json')}; "
            f"tar -C {shlex.quote(remote_receipts)} -czf - ."
        )
        with archive_out.open("wb") as handle:
            _run_checked(
                (*ssh, "bash -lc " + shlex.quote(collect)),
                timeout=60,
                stdout=handle,
            )
        _extract_receipt_archive(archive_out, output_dir)
        archive_out.unlink()
        receipt_path = output_dir / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["inventory_host_id"] = host["id"]
        receipt["transport"] = transport
        _write_json(receipt_path, receipt)
        return receipt
    except subprocess.TimeoutExpired as exc:
        receipt = _fallback_lane_receipt(
            host,
            lane_id,
            run_id=run_id,
            status="timed_out",
            error=str(exc),
            source_commit=source_commit,
            source_tree=source_tree,
            archive_sha256=archive_sha256,
        )
        _write_json(output_dir / "receipt.json", receipt)
        return receipt
    except (OSError, RepositoryCIError, json.JSONDecodeError, tarfile.TarError) as exc:
        receipt = _fallback_lane_receipt(
            host,
            lane_id,
            run_id=run_id,
            status="failed",
            error=str(exc),
            source_commit=source_commit,
            source_tree=source_tree,
            archive_sha256=archive_sha256,
        )
        _write_json(output_dir / "receipt.json", receipt)
        return receipt
    finally:
        if remote_root is not None:
            cleanup = f"rm -rf -- {shlex.quote(remote_root)}"
            try:
                subprocess.run(
                    (*ssh, cleanup),
                    check=False,
                    capture_output=True,
                    timeout=20,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass


def _select_hosts(
    hosts: list[dict[str, Any]],
    *,
    requested_hosts: set[str],
    requested_lanes: set[str],
) -> list[tuple[dict[str, Any], list[str]]]:
    known_hosts = {str(host["id"]) for host in hosts}
    unknown_hosts = requested_hosts - known_hosts
    if unknown_hosts:
        raise RepositoryCIError(f"unknown host ids: {sorted(unknown_hosts)}")
    selected: list[tuple[dict[str, Any], list[str]]] = []
    for host in hosts:
        if requested_hosts and host["id"] not in requested_hosts:
            continue
        mapped = [
            lane_id
            for lane_id in host["lanes"]
            if not requested_lanes or lane_id in requested_lanes
        ]
        if requested_lanes and not mapped:
            continue
        selected.append((host, mapped))
    if not selected:
        raise RepositoryCIError("host/lane selection is empty")
    return selected


def _public_plan(
    repository: str | None,
    selected: Sequence[tuple[dict[str, Any], list[str]]],
    lanes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "hosts": [
            {
                "id": host["id"],
                "address": host["address"],
                "transport": host["transport"],
                "trust": host.get("trust", "local"),
                "dispatch_enabled": host["dispatch_enabled"],
                "required": host["required"],
                "lanes": lane_ids,
                "capabilities": host.get("capabilities", []),
                "host_key_fingerprint_sha256": (
                    host.get("host_key", {}).get("fingerprint_sha256")
                    if host.get("trust") == "trusted"
                    else None
                ),
                "reason": host.get("reason"),
            }
            for host, lane_ids in selected
        ],
        "lanes": {
            lane_id: {
                "system": lane["system"],
                "machines": lane["machines"],
                "requires_cuda": lane["requires_cuda"],
                "required_modules": lane["required_modules"],
                "timeout_seconds": lane["timeout_seconds"],
                "commands": lane["commands"],
            }
            for lane_id, lane in lanes.items()
            if any(lane_id in lane_ids for _, lane_ids in selected)
        },
    }


def list_plan(args: argparse.Namespace) -> int:
    lanes = load_lanes(Path(args.lanes_config).resolve())
    inventory, hosts = load_hosts(Path(args.hosts_config).resolve(), lanes=lanes)
    selected = _select_hosts(
        hosts,
        requested_hosts=set(args.host or ()),
        requested_lanes=set(args.lane or ()),
    )
    plan = _public_plan(inventory.get("repository"), selected, lanes)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    for host in plan["hosts"]:
        print(
            f"{host['id']}: address={host['address']} trust={host['trust']} "
            f"dispatch={str(host['dispatch_enabled']).lower()} "
            f"lanes={','.join(host['lanes']) or '-'}"
        )
    return 0


def orchestrate(args: argparse.Namespace) -> int:
    lanes = load_lanes(Path(args.lanes_config).resolve())
    inventory, hosts = load_hosts(Path(args.hosts_config).resolve(), lanes=lanes)
    requested_lanes = set(args.lane or ())
    unknown_lanes = requested_lanes - set(lanes)
    if unknown_lanes:
        raise RepositoryCIError(f"unknown lane ids: {sorted(unknown_lanes)}")
    selected = _select_hosts(
        hosts,
        requested_hosts=set(args.host or ()),
        requested_lanes=requested_lanes,
    )
    repo_root = Path(args.repo_root).resolve()
    source_commit = (
        _git_run(repo_root, "rev-parse", "--verify", f"{args.ref}^{{commit}}")
        .decode()
        .strip()
    )
    source_tree = (
        _git_run(repo_root, "rev-parse", f"{source_commit}^{{tree}}").decode().strip()
    )
    run_id = args.run_id or _new_run_id()
    receipt_root = Path(args.receipt_dir).resolve() / run_id
    receipt_root.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    started = time.monotonic()
    source_before = capture_provenance(repo_root)
    source_after: dict[str, Any] | None = None
    probes: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    blockers: list[str] = []
    status = "failed"
    exit_code = 1
    archive_sha256: str | None = None

    with tempfile.TemporaryDirectory(prefix="cppmega-ci-identity-") as identity_tmp:
        identity_dir = Path(identity_tmp)
        for host, lane_ids in selected:
            probe = probe_host(
                host,
                lanes=lanes,
                selected_lanes=lane_ids,
                known_hosts_dir=identity_dir,
                connect_timeout=args.connect_timeout,
            )
            probes.append(probe)
            _write_json(receipt_root / host["id"] / "preflight.json", probe)
            for lane_id in lane_ids:
                if not host["dispatch_enabled"]:
                    continue
                lane_status = probe["lane_status"].get(lane_id, {})
                if host["required"] and lane_status.get("available") is not True:
                    blockers.append(
                        f"{host['id']}:{lane_id}:{lane_status.get('status')}"
                    )
            print(
                f"[repository-ci] host={host['id']} status={probe['status']} "
                f"detail={probe.get('detail')}",
                flush=True,
            )

        if args.dry_run:
            status = "blocked_preflight" if blockers else "dry_run_passed"
            exit_code = 2 if blockers else 0
        else:
            with tempfile.TemporaryDirectory(prefix="cppmega-ci-source-") as source_tmp:
                archive_path = Path(source_tmp) / "source.tar"
                archive_sha256 = _create_source_archive(
                    repo_root,
                    commit=source_commit,
                    destination=archive_path,
                )
                probe_by_host = {probe["host_id"]: probe for probe in probes}
                for host, lane_ids in selected:
                    if not host["dispatch_enabled"]:
                        continue
                    probe = probe_by_host[host["id"]]
                    ssh: tuple[str, ...] | None = None
                    if host["transport"] == "ssh":
                        known_hosts = _write_known_hosts(host, identity_dir)
                        ssh = _ssh_base(
                            host,
                            known_hosts=known_hosts,
                            connect_timeout=args.connect_timeout,
                            identity_file=_identity_file_from_environment(),
                        )
                    for lane_id in lane_ids:
                        if probe["lane_status"][lane_id]["available"] is not True:
                            continue
                        output_dir = receipt_root / host["id"] / lane_id
                        output_dir.mkdir(parents=True, exist_ok=True)
                        kwargs = {
                            "archive_path": archive_path,
                            "archive_sha256": archive_sha256,
                            "source_commit": source_commit,
                            "source_tree": source_tree,
                            "output_dir": output_dir,
                            "run_id": run_id,
                            "lane_timeout": int(lanes[lane_id]["timeout_seconds"]),
                        }
                        if host["transport"] == "local":
                            result = _run_local_lane(host, lane_id, **kwargs)
                        else:
                            assert ssh is not None
                            result = _run_remote_lane(host, lane_id, ssh=ssh, **kwargs)
                        results.append(result)
                failed = [
                    result for result in results if result.get("status") != "passed"
                ]
                if failed:
                    status = "failed"
                    exit_code = 1
                elif blockers:
                    status = "partial_blocked"
                    exit_code = 2
                elif not results:
                    status = "blocked_no_runnable_lanes"
                    exit_code = 2
                else:
                    status = "passed"
                    exit_code = 0

    source_after = capture_provenance(repo_root)
    source_stable = provenance_unchanged(source_before, source_after)
    if not source_stable:
        status = "failed_source_provenance_changed"
        exit_code = 1
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "orchestration",
        "run_id": run_id,
        "status": status,
        "exit_code": exit_code,
        "dry_run": bool(args.dry_run),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "repository": inventory.get("repository"),
        "source": {
            "requested_ref": args.ref,
            "commit": source_commit,
            "tree": source_tree,
            "archive_sha256": archive_sha256,
        },
        "source_provenance": {
            "before_preflight": source_before,
            "after_execution": source_after,
            "unchanged": source_stable,
        },
        "plan": _public_plan(inventory.get("repository"), selected, lanes),
        "preflights": probes,
        "blockers": blockers,
        "results": results,
    }
    receipt_path = receipt_root / "orchestration.json"
    _write_json(receipt_path, receipt)
    print(f"[repository-ci] orchestration receipt: {receipt_path}", flush=True)
    return exit_code


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hosts-config", default=str(DEFAULT_HOSTS_CONFIG))
    parser.add_argument("--lanes-config", default=str(DEFAULT_LANES_CONFIG))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list hosts, lanes, and commands")
    _add_config_arguments(list_parser)
    list_parser.add_argument("--host", action="append")
    list_parser.add_argument("--lane", action="append")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(handler=list_plan)

    run_parser = subparsers.add_parser(
        "run",
        help="preflight and dispatch CI independently of GitHub Actions",
    )
    _add_config_arguments(run_parser)
    run_parser.add_argument("--repo-root", default=str(REPO_ROOT))
    run_parser.add_argument("--receipt-dir", default=str(DEFAULT_RECEIPT_BASE))
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--ref", default="HEAD")
    run_parser.add_argument("--host", action="append")
    run_parser.add_argument("--lane", action="append")
    run_parser.add_argument("--connect-timeout", type=int, default=5)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(handler=orchestrate)

    lane_parser = subparsers.add_parser("lane", help="run one lane on this machine")
    lane_parser.add_argument("--lanes-config", default=str(DEFAULT_LANES_CONFIG))
    lane_parser.add_argument("--lane", required=True)
    lane_parser.add_argument("--repo-root", default=str(REPO_ROOT))
    lane_parser.add_argument("--receipt-dir", required=True)
    lane_parser.add_argument("--run-id")
    lane_parser.add_argument("--python", default=sys.executable)
    lane_parser.add_argument("--expected-source-commit")
    lane_parser.add_argument("--expected-source-tree")
    lane_parser.add_argument("--archive-sha256")
    lane_parser.set_defaults(handler=run_lane)
    return parser


def _write_early_failure_receipt(
    args: argparse.Namespace, exc: BaseException
) -> None:
    """Best-effort minimal receipt when the lane orchestrator dies early.

    The workflow uploads ``--receipt-dir`` with ``if-no-files-found: error``,
    so a crash before ``run_lane`` writes its own receipt (unknown lane id,
    unreadable lane config, unexpected exception) must still leave a
    ``receipt.json`` plus a traceback log behind to preserve the root cause.
    """

    if getattr(args, "command", None) != "lane":
        return
    receipt_dir_value = getattr(args, "receipt_dir", None)
    if not receipt_dir_value:
        return
    redactor = _Redactor()
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        receipt_dir = Path(str(receipt_dir_value)).resolve()
        receipt_dir.mkdir(parents=True, exist_ok=True)
        now = _utc_now()
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": "lane",
            "run_id": getattr(args, "run_id", None),
            "lane": getattr(args, "lane", None),
            "status": "failed",
            "exit_code": 2,
            "failure_stage": "orchestrator",
            "error": redactor.text(f"orchestrator failed before lane completion: {exc}"),
            "started_at": now,
            "completed_at": now,
            "host": {
                "hostname": socket.gethostname(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "steps": [],
        }
        receipt_path = receipt_dir / "receipt.json"
        if not receipt_path.is_file():
            _write_json(receipt_path, receipt)
        (receipt_dir / "orchestrator-failure.log").write_text(
            redactor.text(detail), encoding="utf-8"
        )
        print(f"[repository-ci] failure receipt: {receipt_path}", flush=True)
    except OSError:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "connect_timeout", 1) <= 0:
        parser.error("--connect-timeout must be positive")
    try:
        return int(args.handler(args))
    except (OSError, RepositoryCIError, subprocess.SubprocessError) as exc:
        print(f"[repository-ci] fatal: {_Redactor().text(str(exc))}", file=sys.stderr)
        _write_early_failure_receipt(args, exc)
        return 2
    except Exception as exc:
        print(
            f"[repository-ci] fatal: unexpected {type(exc).__name__}: "
            f"{_Redactor().text(str(exc))}",
            file=sys.stderr,
        )
        _write_early_failure_receipt(args, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
