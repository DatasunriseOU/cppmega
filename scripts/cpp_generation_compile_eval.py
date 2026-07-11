#!/usr/bin/env python3
"""Compile-and-run gate for C/C++ docstring-to-code generations.

This harness intentionally evaluates functional correctness, not text overlap.
Each case provides a prompt plus source prefix/suffix. A model completion is
inserted between the prefix and suffix, compiled with a local C++ compiler, then
executed. The report is JSON so the same cases can be used by local smoke tests
and by remote H200 generation jobs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_COMPILE_ARGS = ("-std=c++20", "-O0")
DEFAULT_C_COMPILE_ARGS = ("-std=c17", "-O0")
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_CLANG_FORMAT = "clang-format"
COMPLETION_KEYS = ("completion", "generated_text", "text", "candidate")
FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+.#-]+)?\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class CppGenerationCase:
    task_id: str
    prompt: str
    source_prefix: str
    source_suffix: str
    language: str = "cpp"
    compile_args: tuple[str, ...] = DEFAULT_COMPILE_ARGS
    timeout_s: float = DEFAULT_TIMEOUT_S
    sidecar_contract: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "CppGenerationCase":
        for key in ("task_id", "prompt", "source_prefix", "source_suffix"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"case row needs non-empty string field {key!r}")
        # task_id is used as a filesystem path component (work_root / task_id), so it
        # must be a single safe name — reject separators / traversal / absolute paths.
        if Path(row["task_id"]).name != row["task_id"] or row["task_id"] in {".", ".."}:
            raise ValueError(f"task_id must be a plain filename (no path separators): {row['task_id']!r}")
        language = str(row.get("language", "cpp"))
        if language not in {"c", "cpp", "c++", "cc"}:
            raise ValueError(f"{row['task_id']}: unsupported language {language!r}")
        default_compile_args = (
            list(DEFAULT_C_COMPILE_ARGS) if language == "c" else list(DEFAULT_COMPILE_ARGS)
        )
        compile_args = row.get("compile_args", default_compile_args)
        if not isinstance(compile_args, list) or not all(
            isinstance(item, str) for item in compile_args
        ):
            raise ValueError(f"{row['task_id']}: compile_args must be list[str]")
        timeout_s = float(row.get("timeout_s", DEFAULT_TIMEOUT_S))
        if timeout_s <= 0:
            raise ValueError(f"{row['task_id']}: timeout_s must be positive")
        sidecar_contract = row.get("sidecar_contract")
        if sidecar_contract is not None and not isinstance(sidecar_contract, dict):
            raise ValueError(f"{row['task_id']}: sidecar_contract must be object")
        return cls(
            task_id=row["task_id"],
            prompt=row["prompt"],
            source_prefix=row["source_prefix"],
            source_suffix=row["source_suffix"],
            language=language,
            compile_args=tuple(compile_args),
            timeout_s=timeout_s,
            sidecar_contract=sidecar_contract,
        )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object")
            yield row


def load_cases(path: Path) -> dict[str, CppGenerationCase]:
    cases: dict[str, CppGenerationCase] = {}
    for row in iter_jsonl(path):
        case = CppGenerationCase.from_json(row)
        if case.task_id in cases:
            raise ValueError(f"duplicate task_id in cases: {case.task_id}")
        cases[case.task_id] = case
    if not cases:
        raise ValueError(f"no cases in {path}")
    return cases


def _completion_from_row(row: dict[str, Any]) -> str:
    for key in COMPLETION_KEYS:
        value = row.get(key)
        if isinstance(value, str):
            return value
    raise ValueError(
        "completion row needs one of "
        + ", ".join(repr(key) for key in COMPLETION_KEYS)
    )


def load_completions(path: Path) -> dict[str, str]:
    completions: dict[str, str] = {}
    for row in iter_jsonl(path):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("completion row needs non-empty string field 'task_id'")
        if task_id in completions:
            raise ValueError(f"duplicate completion for task_id: {task_id}")
        completions[task_id] = _completion_from_row(row)
    if not completions:
        raise ValueError(f"no completions in {path}")
    return completions


def extract_code(raw_completion: str) -> str:
    """Return code from a plain or fenced model completion."""
    match = FENCE_RE.search(raw_completion)
    text = match.group(1) if match else raw_completion
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    return text.rstrip() + "\n"


def compose_source(case: CppGenerationCase, completion: str) -> str:
    return case.source_prefix + extract_code(completion) + case.source_suffix


def _trim_output(text: str, limit: int = 8192) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 120] + "\n...<truncated>...\n" + text[-100:]


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_s: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed = time.perf_counter() - start
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "stdout": _trim_output(proc.stdout),
            "stderr": _trim_output(proc.stderr),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "returncode": None,
            "elapsed_s": elapsed,
            "stdout": _trim_output(exc.stdout or ""),
            "stderr": _trim_output(exc.stderr or ""),
            "timeout": True,
        }


def _run_clang_format(
    source: str,
    *,
    clang_format: str,
    language: str,
    timeout_s: float,
) -> tuple[str, dict[str, Any]]:
    suffix = ".c" if language == "c" else ".cpp"
    cmd = [clang_format, f"--assume-filename=candidate{suffix}"]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            input=source,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed = time.perf_counter() - start
        result = {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "stdout": _trim_output(proc.stdout),
            "stderr": _trim_output(proc.stderr),
            "timeout": False,
            "cmd": cmd,
        }
        return (proc.stdout if proc.returncode == 0 else source), result
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        result = {
            "ok": False,
            "returncode": None,
            "elapsed_s": elapsed,
            "stdout": _trim_output(exc.stdout or ""),
            "stderr": _trim_output(exc.stderr or ""),
            "timeout": True,
            "cmd": cmd,
        }
        return source, result


def evaluate_case(
    case: CppGenerationCase,
    completion: str | None,
    *,
    compiler: str,
    clang_format: str | None,
    work_root: Path,
) -> dict[str, Any]:
    case_dir = work_root / case.task_id
    case_dir.mkdir(parents=True, exist_ok=True)
    source_path = case_dir / ("candidate.c" if case.language == "c" else "candidate.cpp")
    exe_path = case_dir / "candidate"

    if completion is None:
        return {
            "task_id": case.task_id,
            "passed": False,
            "compile_ok": False,
            "run_ok": False,
            "missing_completion": True,
            "sidecar_contract": case.sidecar_contract or {},
        }

    source = compose_source(case, completion)
    format_result: dict[str, Any] | None = None
    if clang_format is not None:
        source, format_result = _run_clang_format(
            source,
            clang_format=clang_format,
            language=case.language,
            timeout_s=case.timeout_s,
        )
    source_path.write_text(source, encoding="utf-8")

    if format_result is not None and not format_result["ok"]:
        return {
            "task_id": case.task_id,
            "passed": False,
            "compile_ok": False,
            "run_ok": False,
            "missing_completion": False,
            "source_path": str(source_path),
            "clang_format": format_result,
            "compile_cmd": None,
            "compile": None,
            "run": None,
            "sidecar_contract": case.sidecar_contract or {},
        }

    compile_cmd = [compiler, *case.compile_args, str(source_path), "-o", str(exe_path)]
    compile_result = _run_cmd(compile_cmd, cwd=case_dir, timeout_s=case.timeout_s)
    run_result: dict[str, Any] | None = None
    if compile_result["ok"]:
        run_result = _run_cmd([str(exe_path)], cwd=case_dir, timeout_s=case.timeout_s)

    compile_ok = bool(compile_result["ok"])
    run_ok = bool(run_result and run_result["ok"])
    return {
        "task_id": case.task_id,
        "passed": compile_ok and run_ok,
        "compile_ok": compile_ok,
        "run_ok": run_ok,
        "missing_completion": False,
        "source_path": str(source_path),
        "clang_format": format_result,
        "compile_cmd": compile_cmd,
        "compile": compile_result,
        "run": run_result,
        "sidecar_contract": case.sidecar_contract or {},
    }


def _evaluate_case_worker(
    args: tuple[CppGenerationCase, str | None, str, str | None, Path],
) -> dict[str, Any]:
    """Top-level (pickle-safe) worker so cases can run under ProcessPoolExecutor."""
    case, completion, compiler, clang_format, work_root = args
    return evaluate_case(
        case,
        completion,
        compiler=compiler,
        clang_format=clang_format,
        work_root=work_root,
    )


def evaluate_suite(
    cases: dict[str, CppGenerationCase],
    completions: dict[str, str],
    *,
    cpp_compiler: str,
    c_compiler: str,
    clang_format: str | None,
    keep_workdir: bool,
    jobs: int | None = None,
) -> dict[str, Any]:
    for compiler in {cpp_compiler, c_compiler}:
        if shutil.which(compiler) is None:
            raise FileNotFoundError(f"compiler not found on PATH: {compiler}")
    if clang_format is not None and shutil.which(clang_format) is None:
        raise FileNotFoundError(f"clang-format not found on PATH: {clang_format}")
    extra = sorted(set(completions) - set(cases))
    if extra:
        raise ValueError(f"completion task_id not present in cases: {extra[:10]}")
    if jobs is None:
        jobs = min(8, os.cpu_count() or 1)
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")

    with tempfile.TemporaryDirectory(prefix="cppmega_cpp_eval_") as tmp:
        work_root = Path(tmp)
        if keep_workdir:
            persistent = Path("outputs/evals/cpp_generation_compile_work").resolve()
            if persistent.exists():
                shutil.rmtree(persistent)
            persistent.mkdir(parents=True)
            work_root = persistent

        # evaluate_case isolates each case under work_root/task_id, so cases run in
        # parallel safely. executor.map preserves the sorted(cases) input order; a
        # worker Python exception propagates here (a normal compile/run failure is a
        # returned result, not an exception).
        worker_args = [
            (
                case,
                completions.get(task_id),
                c_compiler if case.language == "c" else cpp_compiler,
                clang_format,
                work_root,
            )
            for task_id, case in sorted(cases.items())
        ]
        if jobs == 1:
            results = [_evaluate_case_worker(item) for item in worker_args]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                results = list(executor.map(_evaluate_case_worker, worker_args))
    passed = sum(1 for item in results if item["passed"])
    compiled = sum(1 for item in results if item["compile_ok"])
    ran = sum(1 for item in results if item["run_ok"])
    formatted = sum(
        1
        for item in results
        if item.get("clang_format") is None or item["clang_format"]["ok"]
    )
    return {
        "summary": {
            "total": len(results),
            "passed": passed,
            "compiled": compiled,
            "ran": ran,
            "clang_format_ok": formatted,
            "pass_rate": passed / len(results) if results else 0.0,
            "cpp_compiler": cpp_compiler,
            "c_compiler": c_compiler,
            "clang_format": clang_format,
            "jobs": jobs,
        },
        "results": results,
    }


def write_prompts(cases: dict[str, CppGenerationCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for task_id, case in sorted(cases.items()):
            fh.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "language": case.language,
                        "prompt": case.prompt,
                        "sidecar_contract": case.sidecar_contract or {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts-out", type=Path)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "clang++"))
    parser.add_argument("--c-compiler", default=os.environ.get("CC", "clang"))
    parser.add_argument(
        "--clang-format",
        default=os.environ.get("CLANG_FORMAT", DEFAULT_CLANG_FORMAT),
        help="clang-format binary used before compile; default: %(default)s",
    )
    parser.add_argument(
        "--no-clang-format",
        action="store_true",
        help="Explicit ablation: compile generated source without clang-format.",
    )
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel compile/run workers (default: %(default)s)",
    )
    parser.add_argument("--fail-on-fail", action="store_true")
    parser.add_argument("--json", action="store_true", help="also print report JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cases = load_cases(args.cases)
    completions = load_completions(args.completions)
    if args.prompts_out:
        write_prompts(cases, args.prompts_out)
    report = evaluate_suite(
        cases,
        completions,
        cpp_compiler=args.compiler,
        c_compiler=args.c_compiler,
        clang_format=None if args.no_clang_format else args.clang_format,
        keep_workdir=args.keep_workdir,
        jobs=args.jobs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_fail and report["summary"]["passed"] != report["summary"]["total"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
