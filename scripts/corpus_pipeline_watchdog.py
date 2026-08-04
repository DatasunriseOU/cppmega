#!/usr/bin/env python3
"""Bounded liveness and explicit-429 resume monitor for cppmega pipelines.

The monitor deliberately does not infer failure from a quiet log or a long
running parser.  A lane is resumed only when its exact process is gone, it has
not published a configured terminal receipt, and its latest durable request
ledger entry explicitly records a transient HTTP 429.  Resume commands are
argv lists, never shell snippets, and state makes every evidence marker
single-use.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    load_json_object,
)


WATCHDOG_SCHEMA = "cppmega.pipeline_watchdog_v1"
WATCHDOG_STATE_SCHEMA = "cppmega.pipeline_watchdog_state_v1"
WATCHDOG_REPORT_SCHEMA = "cppmega.pipeline_watchdog_report_v1"
_LANE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_MAX_HANDLED_EVIDENCE = 64


class WatchdogError(ContractError):
    """The monitor configuration or durable pipeline state is unsafe."""


Runner = Callable[[], list[str]]
Launcher = Callable[[Sequence[str], Path | None, Path | None], int]


def _nonempty_string(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise WatchdogError(f"{where} must be a non-empty string")
    return value


def _path(value: object, *, where: str) -> Path:
    return Path(_nonempty_string(value, where=where))


def _string_list(value: object, *, where: str, minimum: int = 0) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise WatchdogError(f"{where} must be a list with at least {minimum} items")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_nonempty_string(item, where=f"{where}[{index}]"))
    return result


def _mapping(value: object, *, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise WatchdogError(f"{where} must be an object")
    return dict(value)


def _safe_regular_file(path: Path, *, where: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise WatchdogError(f"{where} must be a regular file: {path}")
    return path


def _validate_terminal_receipt(value: object, *, where: str) -> dict[str, object]:
    receipt = _mapping(value, where=where)
    allowed = {"path", "status_values"}
    if set(receipt) != allowed:
        raise WatchdogError(f"{where} fields must be {sorted(allowed)}")
    _path(receipt["path"], where=f"{where}.path")
    statuses = _string_list(receipt["status_values"], where=f"{where}.status_values", minimum=1)
    if len(statuses) != len(set(statuses)):
        raise WatchdogError(f"{where}.status_values must be unique")
    return receipt


def _validate_429_evidence(value: object, *, where: str) -> dict[str, object]:
    evidence = _mapping(value, where=where)
    kind = _nonempty_string(evidence.get("kind"), where=f"{where}.kind")
    if kind != "sqlite_request_ledger":
        raise WatchdogError(f"{where}.kind is unsupported: {kind}")
    allowed = {"kind", "path"}
    if set(evidence) != allowed:
        raise WatchdogError(f"{where} fields must be {sorted(allowed)}")
    _path(evidence["path"], where=f"{where}.path")
    return evidence


def _validate_resume(value: object, *, where: str) -> dict[str, object]:
    resume = _mapping(value, where=where)
    allowed = {"argv", "cwd", "stdout"}
    if set(resume) - allowed:
        raise WatchdogError(f"{where} has unsupported fields")
    if "argv" not in resume:
        raise WatchdogError(f"{where}.argv is required")
    _string_list(resume["argv"], where=f"{where}.argv", minimum=1)
    if "cwd" in resume:
        _path(resume["cwd"], where=f"{where}.cwd")
    if "stdout" in resume:
        _path(resume["stdout"], where=f"{where}.stdout")
    return resume


def _validate_resume_guard(value: object, *, where: str) -> dict[str, object]:
    guard = _mapping(value, where=where)
    allowed = {"unlocked_files"}
    if set(guard) != allowed:
        raise WatchdogError(f"{where} fields must be {sorted(allowed)}")
    paths = guard.get("unlocked_files")
    if not isinstance(paths, list) or not paths:
        raise WatchdogError(f"{where}.unlocked_files must be a non-empty list")
    checked = [
        _path(item, where=f"{where}.unlocked_files[{index}]")
        for index, item in enumerate(paths)
    ]
    if len({str(item) for item in checked}) != len(checked):
        raise WatchdogError(f"{where}.unlocked_files must be unique")
    return guard


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate a small, intentionally restrictive monitor configuration."""

    value = dict(config)
    allowed = {"schema", "state_path", "report_path", "lanes"}
    if set(value) != allowed:
        raise WatchdogError(
            "watchdog config fields drifted: "
            f"missing={sorted(allowed - set(value))} extra={sorted(set(value) - allowed)}"
        )
    if value.get("schema") != WATCHDOG_SCHEMA:
        raise WatchdogError("watchdog config schema is unsupported")
    _path(value["state_path"], where="state_path")
    _path(value["report_path"], where="report_path")
    lanes = value.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        raise WatchdogError("lanes must be a non-empty list")
    names: set[str] = set()
    checked_lanes: list[dict[str, object]] = []
    for index, raw_lane in enumerate(lanes):
        lane = _mapping(raw_lane, where=f"lanes[{index}]")
        allowed_lane = {
            "name",
            "process_contains",
            "process_excludes",
            "terminal_receipt",
            "transient_429",
            "resume",
            "resume_guard",
        }
        if set(lane) - allowed_lane:
            raise WatchdogError(f"lanes[{index}] has unsupported fields")
        name = _nonempty_string(lane.get("name"), where=f"lanes[{index}].name")
        if _LANE_NAME_RE.fullmatch(name) is None or name in names:
            raise WatchdogError(f"lanes[{index}].name is invalid or duplicated")
        names.add(name)
        _string_list(lane.get("process_contains"), where=f"lanes[{index}].process_contains", minimum=1)
        if "process_excludes" in lane:
            _string_list(lane["process_excludes"], where=f"lanes[{index}].process_excludes", minimum=1)
        if "terminal_receipt" in lane:
            _validate_terminal_receipt(lane["terminal_receipt"], where=f"lanes[{index}].terminal_receipt")
        has_evidence = "transient_429" in lane
        has_resume = "resume" in lane
        has_guard = "resume_guard" in lane
        if not (has_evidence == has_resume == has_guard):
            raise WatchdogError(
                f"lanes[{index}] must configure transient_429, resume, and resume_guard together"
            )
        if has_evidence:
            _validate_429_evidence(lane["transient_429"], where=f"lanes[{index}].transient_429")
            _validate_resume(lane["resume"], where=f"lanes[{index}].resume")
            _validate_resume_guard(lane["resume_guard"], where=f"lanes[{index}].resume_guard")
        checked_lanes.append(lane)
    value["lanes"] = checked_lanes
    return value


def load_config(path: Path) -> dict[str, object]:
    _raw, config = load_json_object(path, where="pipeline watchdog config")
    return validate_config(config)


def _default_process_lines() -> list[str]:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,args="],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise WatchdogError(f"ps failed with exit {completed.returncode}")
    return [line for line in completed.stdout.splitlines() if line.strip()]


def matching_processes(
    process_lines: Sequence[str],
    contains: Sequence[str],
    excludes: Sequence[str] = (),
) -> list[str]:
    """Return process lines that contain every exact configured argv fragment."""

    return [
        line
        for line in process_lines
        if all(fragment in line for fragment in contains)
        and not any(fragment in line for fragment in excludes)
    ]


def _terminal_receipt_status(spec: Mapping[str, object]) -> str | None:
    path = _path(spec["path"], where="terminal_receipt.path")
    if not path.exists():
        return None
    _raw, receipt = load_json_object(path, where="pipeline terminal receipt")
    status = receipt.get("status")
    if not isinstance(status, str):
        raise WatchdogError(f"pipeline terminal receipt has no string status: {path}")
    values = set(_string_list(spec["status_values"], where="terminal_receipt.status_values", minimum=1))
    return status if status in values else None


def _latest_sqlite_429_evidence(spec: Mapping[str, object]) -> dict[str, object] | None:
    path = _safe_regular_file(
        _path(spec["path"], where="transient_429.path"), where="transient_429.path"
    )
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        try:
            row = connection.execute(
                """
                SELECT id, requested_at, http_status, COALESCE(error_class, ''),
                       COALESCE(error_message, '')
                FROM request_ledger
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise WatchdogError(f"cannot read transient_429 request ledger: {path}") from exc
    if row is None:
        return None
    ledger_id, requested_at, http_status, error_class, error_message = row
    if isinstance(ledger_id, bool) or not isinstance(ledger_id, int) or ledger_id < 1:
        raise WatchdogError(f"request ledger id is invalid: {path}")
    message = f"{error_class}\n{error_message}".lower()
    if http_status != 429 and "429" not in message and "too many requests" not in message:
        return None
    payload = {
        "kind": "sqlite_request_ledger",
        "path": str(path),
        "ledger_id": ledger_id,
        "requested_at": str(requested_at),
        "http_status": http_status,
        "error_class": str(error_class),
        "error_message": str(error_message),
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    return payload


def _lane_evidence(lane: Mapping[str, object]) -> dict[str, object] | None:
    raw = lane.get("transient_429")
    if raw is None:
        return None
    spec = _validate_429_evidence(raw, where="transient_429")
    return _latest_sqlite_429_evidence(spec)


def _empty_state() -> dict[str, object]:
    return {"schema": WATCHDOG_STATE_SCHEMA, "lanes": {}}


def load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return _empty_state()
    _raw, state = load_json_object(path, where="pipeline watchdog state")
    if state.get("schema") != WATCHDOG_STATE_SCHEMA:
        raise WatchdogError("pipeline watchdog state schema is unsupported")
    lanes = state.get("lanes")
    if not isinstance(lanes, Mapping):
        raise WatchdogError("pipeline watchdog state lanes is invalid")
    checked: dict[str, object] = {}
    for name, raw_lane in lanes.items():
        if not isinstance(name, str) or _LANE_NAME_RE.fullmatch(name) is None:
            raise WatchdogError("pipeline watchdog state lane name is invalid")
        lane = _mapping(raw_lane, where=f"pipeline watchdog state lane {name}")
        handled = lane.get("handled_evidence", [])
        if not isinstance(handled, list) or any(
            not isinstance(item, str) or len(item) != 64 for item in handled
        ):
            raise WatchdogError("pipeline watchdog state handled evidence is invalid")
        checked[name] = {"handled_evidence": list(handled)}
    return {"schema": WATCHDOG_STATE_SCHEMA, "lanes": checked}


def _lane_state(state: dict[str, object], name: str) -> dict[str, object]:
    lanes = state["lanes"]
    assert isinstance(lanes, dict)
    raw = lanes.setdefault(name, {"handled_evidence": []})
    assert isinstance(raw, dict)
    return raw


def _remember_evidence(lane_state: dict[str, object], fingerprint: str) -> None:
    handled = lane_state.setdefault("handled_evidence", [])
    assert isinstance(handled, list)
    if fingerprint in handled:
        return
    handled.append(fingerprint)
    del handled[:-_MAX_HANDLED_EVIDENCE]


def _default_launcher(argv: Sequence[str], cwd: Path | None, stdout: Path | None) -> int:
    stream = None
    try:
        if stdout is not None:
            stdout.parent.mkdir(parents=True, exist_ok=True)
            stream = stdout.open("ab", buffering=0)
        child = subprocess.Popen(
            list(argv),
            cwd=str(cwd) if cwd is not None else None,
            stdout=stream if stream is not None else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if stream is not None else subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return int(child.pid)
    finally:
        if stream is not None:
            stream.close()


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="ascii") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WatchdogError("another pipeline watchdog is already running") from exc
        yield


def _resume_spec(lane: Mapping[str, object]) -> tuple[list[str], Path | None, Path | None]:
    raw = lane.get("resume")
    if raw is None:
        raise WatchdogError("resume is unavailable")
    spec = _validate_resume(raw, where="resume")
    argv = _string_list(spec["argv"], where="resume.argv", minimum=1)
    cwd = _path(spec["cwd"], where="resume.cwd") if "cwd" in spec else None
    stdout = _path(spec["stdout"], where="resume.stdout") if "stdout" in spec else None
    if cwd is not None and (cwd.is_symlink() or not cwd.is_dir()):
        raise WatchdogError(f"resume.cwd must be a regular directory: {cwd}")
    return argv, cwd, stdout


def _resume_guard_is_clear(
    lane: Mapping[str, object],
) -> tuple[bool, list[str], list[str]]:
    raw = lane.get("resume_guard")
    if raw is None:
        raise WatchdogError("resume_guard is unavailable")
    guard = _validate_resume_guard(raw, where="resume_guard")
    paths = _string_list(
        guard["unlocked_files"], where="resume_guard.unlocked_files", minimum=1
    )
    held: list[str] = []
    invalid: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            invalid.append(raw_path)
            continue
        with path.open("r+b") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held.append(raw_path)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return not held and not invalid, held, invalid


def run_watchdog(
    config: Mapping[str, object],
    *,
    process_runner: Runner = _default_process_lines,
    launcher: Launcher = _default_launcher,
    now: Callable[[], float] = time.time,
    dry_run: bool = False,
) -> dict[str, object]:
    """Check lanes once and restart at most one new durable 429 event per lane."""

    checked = validate_config(config)
    state_path = _path(checked["state_path"], where="state_path")
    report_path = _path(checked["report_path"], where="report_path")
    lock_path = state_path.with_name(f".{state_path.name}.lock")
    checked_at = int(now())
    with _exclusive_lock(lock_path):
        state = load_state(state_path)
        lines = process_runner()
        reports: list[dict[str, object]] = []
        lanes = checked["lanes"]
        assert isinstance(lanes, list)
        for raw_lane in lanes:
            assert isinstance(raw_lane, Mapping)
            lane = dict(raw_lane)
            name = _nonempty_string(lane["name"], where="lane.name")
            contains = _string_list(lane["process_contains"], where="lane.process_contains", minimum=1)
            excludes = _string_list(
                lane.get("process_excludes", []),
                where="lane.process_excludes",
            )
            matches = matching_processes(lines, contains, excludes)
            report: dict[str, object] = {"name": name, "matched_processes": len(matches)}
            lane_state = _lane_state(state, name)
            terminal_spec = lane.get("terminal_receipt")
            if terminal_spec is not None:
                status = _terminal_receipt_status(_validate_terminal_receipt(terminal_spec, where="terminal_receipt"))
                if status is not None:
                    report.update({"state": "terminal", "terminal_status": status})
                    reports.append(report)
                    continue
            if matches:
                report["state"] = "running"
                reports.append(report)
                continue
            evidence = _lane_evidence(lane)
            if evidence is None:
                report["state"] = "stopped_without_confirmed_429"
                reports.append(report)
                continue
            fingerprint = str(evidence["fingerprint"])
            handled = lane_state.get("handled_evidence", [])
            assert isinstance(handled, list)
            if fingerprint in handled:
                report.update({"state": "stopped_429_already_handled", "evidence": evidence})
                reports.append(report)
                continue
            # Re-read process state immediately before spawning to avoid a
            # duplicate writer if a human or another supervisor recovered it.
            raced_matches = matching_processes(process_runner(), contains, excludes)
            if raced_matches:
                report.update({"state": "running_race", "matched_processes": len(raced_matches)})
                reports.append(report)
                continue
            if dry_run:
                report.update({"state": "would_resume_after_429", "evidence": evidence})
                reports.append(report)
                continue
            guard_clear, held_paths, invalid_paths = _resume_guard_is_clear(lane)
            if not guard_clear:
                report.update(
                    {
                        "state": (
                            "stopped_429_resume_guard_invalid"
                            if invalid_paths
                            else "stopped_429_resume_guard_held"
                        ),
                        "held_paths": held_paths,
                        "invalid_paths": invalid_paths,
                        "evidence": evidence,
                    }
                )
                reports.append(report)
                continue
            argv, cwd, stdout = _resume_spec(lane)
            try:
                pid = launcher(argv, cwd, stdout)
            except OSError as exc:
                report.update({"state": "resume_launch_failed", "error": str(exc), "evidence": evidence})
                reports.append(report)
                continue
            _remember_evidence(lane_state, fingerprint)
            report.update({"state": "resumed_after_429", "pid": pid, "evidence": evidence})
            reports.append(report)
        state["updated_at_unix"] = checked_at
        atomic_write_json(state_path, state)
        report_payload: dict[str, object] = {
            "schema": WATCHDOG_REPORT_SCHEMA,
            "checked_at_unix": checked_at,
            "lanes": reports,
        }
        atomic_write_json(report_path, report_payload)
        return report_payload


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_watchdog(load_config(args.config), dry_run=args.dry_run)
    except (WatchdogError, OSError, ValueError) as exc:
        parser.exit(2, f"cppmega pipeline watchdog failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "WATCHDOG_REPORT_SCHEMA",
    "WATCHDOG_SCHEMA",
    "WATCHDOG_STATE_SCHEMA",
    "WatchdogError",
    "load_config",
    "matching_processes",
    "run_watchdog",
    "validate_config",
]
