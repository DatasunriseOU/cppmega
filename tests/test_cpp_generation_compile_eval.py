from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cpp_generation_compile_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cpp_generation_compile_eval", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_clang_format_prefers_path_and_returns_executable() -> None:
    module = _load_module()

    assert module.resolve_clang_format("/opt/llvm/bin/clang-format") == (
        "/opt/llvm/bin/clang-format"
    )


def test_resolve_clang_format_uses_xcrun_when_path_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    calls: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        return "/usr/bin/xcrun" if name == "xcrun" else None

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="/Applications/Xcode.app/clang-format\n",
            stderr="",
        )

    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.resolve_clang_format("clang-format") == (
        "/Applications/Xcode.app/clang-format"
    )
    assert calls == [["xcrun", "--find", "clang-format"]]


def test_resolve_clang_format_tolerates_broken_xcrun_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: "/usr/bin/xcrun" if name == "xcrun" else None,
    )
    monkeypatch.setattr(module.Path, "is_file", lambda _path: False)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["xcrun"], timeout=5)
        ),
    )

    assert module.resolve_clang_format("clang-format") == "clang-format"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.skipif(shutil.which("clang++") is None, reason="clang++ not installed")
def test_reference_docstring_suite_compiles_and_runs(tmp_path: Path) -> None:
    gold_path = ROOT / "evals" / "cpp_docstring_compile_reference.jsonl"
    gold_rows = [json.loads(line) for line in gold_path.read_text().splitlines()]
    assert all(row["completion_source"] == "gold_fixture" for row in gold_rows)

    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cases",
            str(ROOT / "evals" / "cpp_docstring_compile_cases.jsonl"),
            "--completions",
            str(gold_path),
            "--out",
            str(out),
            "--fail-on-fail",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text())
    assert report["summary"]["total"] == 5
    assert report["summary"]["passed"] == 5
    assert report["summary"]["repository_cases"] == 1
    assert all(item["compile_ok"] and item["run_ok"] for item in report["results"])
    repo_result = next(
        item
        for item in report["results"]
        if item["task_id"] == "case3_add_one_checked"
    )
    assert repo_result["compile_context"] == "repository"
    assert len(repo_result["linked_sources"]) == 3


@pytest.mark.skipif(shutil.which("clang++") is None, reason="clang++ not installed")
def test_bad_completion_fails_the_hard_gate(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    completions = tmp_path / "completions.jsonl"
    out = tmp_path / "report.json"
    _write_jsonl(
        cases,
        [
            {
                "task_id": "always_one",
                "prompt": "Return one.",
                "source_prefix": "#include <cassert>\nint f() {\n",
                "source_suffix": "}\nint main(){ assert(f() == 1); }\n",
            }
        ],
    )
    _write_jsonl(completions, [{"task_id": "always_one", "completion": "return 2;"}])

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cases",
            str(cases),
            "--completions",
            str(completions),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text())
    assert report["summary"]["passed"] == 0
    assert report["results"][0]["compile_ok"] is True
    assert report["results"][0]["run_ok"] is False


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not installed")
def test_c_language_case_uses_c_compiler(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    completions = tmp_path / "completions.jsonl"
    out = tmp_path / "report.json"
    _write_jsonl(
        cases,
        [
            {
                "task_id": "c_abs",
                "language": "c",
                "prompt": "Return the absolute value.",
                "source_prefix": "#include <assert.h>\nint c_abs(int x) {\n",
                "source_suffix": "}\nint main(void){ assert(c_abs(-3) == 3); }\n",
            }
        ],
    )
    _write_jsonl(completions, [{"task_id": "c_abs", "completion": "return x < 0 ? -x : x;"}])

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cases",
            str(cases),
            "--completions",
            str(completions),
            "--out",
            str(out),
            "--fail-on-fail",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text())
    assert report["summary"]["passed"] == 1
    assert report["summary"]["c_compiler"] == "clang"


def test_from_json_rejects_task_id_path_traversal() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="plain filename"):
        module.CppGenerationCase.from_json(
            {"task_id": "../../pwn", "prompt": "p", "source_prefix": "a", "source_suffix": "b"}
        )


def test_evaluate_suite_rejects_nonpositive_jobs() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="jobs must be"):
        module.evaluate_suite(
            {},
            {},
            cpp_compiler="true",
            c_compiler="true",
            clang_format=None,
            keep_workdir=False,
            jobs=0,
        )


@pytest.mark.skipif(shutil.which("clang++") is None, reason="clang++ not installed")
def test_parallel_jobs_preserve_order_and_pass(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cases",
            str(ROOT / "evals" / "cpp_docstring_compile_cases.jsonl"),
            "--completions",
            str(ROOT / "evals" / "cpp_docstring_compile_reference.jsonl"),
            "--out",
            str(out),
            "--jobs",
            "3",
            "--fail-on-fail",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text())
    assert report["summary"]["total"] == 5
    assert report["summary"]["passed"] == 5
    assert report["summary"]["jobs"] == 3
    ids = [item["task_id"] for item in report["results"]]
    assert ids == sorted(ids)  # deterministic sorted order under parallelism


def test_extract_code_accepts_fenced_cpp_completion() -> None:
    module = _load_module()
    raw = "Here is the body:\n```cpp\nreturn 1;\n```\n"
    assert module.extract_code(raw) == "return 1;\n"


@pytest.mark.skipif(shutil.which("clang-format") is None, reason="clang-format not installed")
@pytest.mark.skipif(shutil.which("clang++") is None, reason="clang++ not installed")
def test_compile_gate_formats_candidate_before_compile(tmp_path: Path) -> None:
    module = _load_module()
    case = module.CppGenerationCase(
        task_id="format_me",
        prompt="Return one.",
        source_prefix="#include <cassert>\nint f(){",
        source_suffix="}\nint main(){assert(f()==1);}\n",
    )

    result = module.evaluate_case(
        case,
        "if(true){return 1;}return 0;",
        compiler="clang++",
        clang_format="clang-format",
        work_root=tmp_path,
    )

    assert result["passed"] is True
    assert result["clang_format"]["ok"] is True
    source = Path(result["source_path"]).read_text(encoding="utf-8")
    assert "if (true)" in source
    assert "int f() {" in source


def test_prompts_jsonl_preserves_sidecar_contract(tmp_path: Path) -> None:
    module = _load_module()
    cases = module.load_cases(ROOT / "evals" / "cpp_docstring_compile_cases.jsonl")
    prompts = tmp_path / "prompts.jsonl"
    module.write_prompts(cases, prompts)

    rows = [
        json.loads(line)
        for line in prompts.read_text(encoding="utf-8").splitlines()
    ]
    clamp = next(row for row in rows if row["task_id"] == "clamp_int")
    assert clamp["language"] == "cpp"
    assert "prompt" in clamp
    assert clamp["sidecar_contract"]["prompt_sidecars_required"] == [
        "platform_ids",
        "token_structure_ids",
        "token_ast_depth",
        "token_ast_node_type",
    ]
