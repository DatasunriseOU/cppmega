from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from cppmega.data.domain_schema import DomainKind, delimiter_token_ids
from cppmega.data.tokenizer_contract import RESERVED_ROLE_TOKEN_IDS


def _test_cpp_compiler() -> str | None:
    apple_clang = Path("/usr/bin/clang++")
    return str(apple_clang) if apple_clang.exists() else shutil.which("clang++")


def _load_eval_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_domain_routed_codegen.py"
    )
    spec = importlib.util.spec_from_file_location(
        "eval_domain_routed_codegen", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_executable(path: Path, body: str) -> str:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return str(path)


def test_domain_eval_prompt_loader_requires_domains(tmp_path):
    mod = _load_eval_module()
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "p0",
                "task_type": "cpp_docstring_to_code",
                "prompt": "// add\n",
                "required_domains": ["CPP"],
                "oracle_kind": "cpp_compile_run",
                "expected_sidecars": ["token_domain_ids", "token_role_ids"],
                "compile_suffix": "\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
                "run_binary": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = mod.load_prompts(path)

    assert len(prompts) == 1
    assert prompts[0].required_domains == ("CPP",)
    assert prompts[0].expected_sidecars == ("token_domain_ids", "token_role_ids")


def _case5_ksh_eval_row() -> dict:
    for line in Path("evals/domain_routed_prompts.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        row = json.loads(line)
        if row["id"] == "ksh_build_sidecar_route":
            return row
    raise AssertionError("missing ksh_build_sidecar_route eval fixture")


def test_domain_eval_freezes_structured_ksh_prompt_sidecars(tmp_path: Path) -> None:
    mod = _load_eval_module()
    row = _case5_ksh_eval_row()
    path = tmp_path / "ksh-prompts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    prompt = mod.load_prompts(path)[0]

    assert prompt.prompt_token_ids == (2, 245, 501, 502, 246, 215, 601, 602, 216, 3)
    assert prompt.prompt_sidecars["token_domain_ids"] == (
        0,
        24,
        24,
        24,
        24,
        43,
        43,
        43,
        43,
        0,
    )
    assert prompt.prompt_sidecars["token_shell_edges"] == (
        {"from": 2, "to": 3, "kind": 40},
    )
    assert prompt.prompt_sidecars["token_diagnostic_edges"] == (
        {"from": 6, "to": 7, "kind": 64},
    )
    assert prompt.prompt_sidecars["token_cross_domain_edges"] == (
        {"from": 3, "to": 6, "kind": 100},
    )

    report = mod.evaluate([prompt], {}, compile=False)
    assert report["rows"][0]["prompt_sidecar_receipt"] == {
        "token_count": 10,
        "columns": sorted(prompt.prompt_sidecars),
        "edge_counts": {
            "token_domain_edges": 0,
            "token_build_edges": 0,
            "token_shell_edges": 1,
            "token_diagnostic_edges": 1,
            "token_cross_domain_edges": 1,
        },
    }


def test_domain_eval_rejects_misaligned_prompt_sidecar_edges(tmp_path: Path) -> None:
    mod = _load_eval_module()
    row = _case5_ksh_eval_row()
    row["prompt_sidecars"]["token_shell_edges"][0]["to"] = len(
        row["prompt_token_ids"]
    )
    path = tmp_path / "bad-ksh-prompts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside prompt tokens"):
        mod.load_prompts(path)


def test_domain_eval_loads_shipped_prompts_with_typed_oracles():
    mod = _load_eval_module()
    path = Path("evals/domain_routed_prompts.jsonl")

    prompts = mod.load_prompts(path)

    assert [prompt.oracle_kind for prompt in prompts] == [
        "cpp_compile_run",
        "cpp_compile_run",
        "build_structure",
        "shell_syntax",
        "sidecar_structure",
        "diagnostic_structure",
        "diagnostic_structure",
    ]


def test_domain_eval_preserves_frozen_ksh_python_reserved_contracts() -> None:
    assert int(DomainKind.KSH) == 24
    assert int(DomainKind.PYTHON) == 31
    assert delimiter_token_ids(DomainKind.KSH) == (245, 246)
    assert delimiter_token_ids(DomainKind.PYTHON) == (247, 248)
    assert {
        role: RESERVED_ROLE_TOKEN_IDS[role]
        for role in ("KSH_START", "KSH_END", "PYTHON_START", "PYTHON_END")
    } == {
        "KSH_START": 245,
        "KSH_END": 246,
        "PYTHON_START": 247,
        "PYTHON_END": 248,
    }


def test_domain_eval_prompt_loader_requires_typed_oracle_contract(tmp_path: Path):
    mod = _load_eval_module()
    row = {
        "id": "p0",
        "task_type": "shell",
        "prompt": "echo ok\n",
        "required_domains": ["SH"],
    }
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="oracle_kind"):
        mod.load_prompts(path)


def test_domain_eval_rejects_oracle_domain_mismatch(tmp_path: Path):
    mod = _load_eval_module()
    row = {
        "id": "p0",
        "task_type": "shell",
        "prompt": "echo ok\n",
        "required_domains": ["SH"],
        "oracle_kind": "build_structure",
        "expected_sidecars": [
            "token_domain_ids",
            "token_role_ids",
            "token_shell_edges",
        ],
    }
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="build_structure oracle requires"):
        mod.load_prompts(path)


def test_domain_eval_evaluate_without_completion_fails_closed():
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=("token_domain_ids",),
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )

    report = mod.evaluate([prompt], {}, compile=True)

    assert report["missing_completion"] == 1
    assert report["failed"] == 1
    assert report["passed"] is False
    assert report["rows"][0]["status"] == "missing_completion"


def test_domain_eval_compile_gate_accepts_simple_cpp_completion():
    mod = _load_eval_module()
    compiler = _test_cpp_compiler()
    assert compiler is not None
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=("token_domain_ids",),
        oracle_kind="cpp_compile_run",
        compile_prefix="",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )

    result = mod.compile_cpp_completion(
        prompt,
        "int add(int a, int b){ return a + b; }",
        compiler=compiler,
    )

    assert result["status"] == "compile_passed"


def test_domain_eval_runtime_oracle_rejects_wrong_compilable_completion():
    mod = _load_eval_module()
    compiler = _test_cpp_compiler()
    assert compiler is not None
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=("token_domain_ids",),
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )

    result = mod.compile_cpp_completion(
        prompt,
        "int add(int, int){ return 0; }",
        compiler=compiler,
    )

    assert result["status"] == "runtime_failed"


def test_domain_eval_reports_compile_timeout_instead_of_raising(tmp_path: Path):
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="timeout",
        task_type="cpp_docstring_to_code",
        prompt="// timeout\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return f();}\n",
        run_binary=True,
    )

    compiler = _write_executable(tmp_path / "slow-compiler", "sleep 1")

    result = mod.compile_cpp_completion(
        prompt,
        "int f() { return 0; }",
        compiler=compiler,
        compile_timeout_s=0.01,
    )

    assert result["status"] == "compile_timeout"
    assert result["timeout_s"] == 0.01


def test_domain_eval_marks_no_compile_as_failure():
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )

    report = mod.evaluate(
        [prompt], {"add": "int add(int a,int b){return a+b;}"}, compile=False
    )

    assert report["rows"][0]["status"] == "not_compiled"
    assert report["failed"] == 1
    assert report["passed"] is False


def test_domain_eval_marks_missing_compile_oracle_as_failure():
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="unchecked",
        task_type="cpp_docstring_to_code",
        prompt="// unchecked\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        oracle_kind="cpp_compile_run",
    )

    report = mod.evaluate([prompt], {"unchecked": "int f(){return 0;}"})

    assert report["rows"][0]["status"] == "missing_compile_oracle"
    assert report["failed"] == 1
    assert report["passed"] is False


def test_domain_eval_marks_compiler_unavailable_as_failure():
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )
    report = mod.evaluate(
        [prompt],
        {"add": "int add(int a,int b){return a+b;"},
        tool_overrides={"cpp": None},
    )

    assert report["rows"][0]["status"] == "compile_oracle_unavailable"
    assert report["failed"] == 1
    assert report["passed"] is False


def test_domain_eval_counts_compile_failure() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="add",
        task_type="cpp_docstring_to_code",
        prompt="// add\n",
        required_domains=("CPP",),
        expected_sidecars=(),
        oracle_kind="cpp_compile_run",
        compile_suffix="\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
        run_binary=True,
    )
    report = mod.evaluate(
        [prompt],
        {"add": "int add(int a,int b){return ;"},
    )

    assert report["status_counts"] == {"compile_failed": 1}
    assert report["failed"] == 1
    assert report["passed"] is False


def test_domain_eval_cli_exits_nonzero_for_not_compiled(
    tmp_path: Path,
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    completions = tmp_path / "completions.jsonl"
    report_path = tmp_path / "report.json"
    prompts.write_text(
        json.dumps(
            {
                "id": "add",
                "task_type": "cpp_docstring_to_code",
                    "prompt": "// add\n",
                    "required_domains": ["CPP"],
                    "oracle_kind": "cpp_compile_run",
                    "expected_sidecars": ["token_domain_ids", "token_role_ids"],
                    "compile_suffix": "\nint main(){return add(2,3)==5 ? 0 : 1;}\n",
                "run_binary": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    completions.write_text(
        json.dumps({"id": "add", "completion": "int add(int a,int b){return a+b;}"})
        + "\n",
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_domain_routed_codegen.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--prompts",
            str(prompts),
            "--completions",
            str(completions),
            "--out",
            str(report_path),
            "--no-compile",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert json.loads(report_path.read_text(encoding="utf-8"))["failed"] == 1


@pytest.mark.parametrize(
    ("domain", "shell_name", "completion"),
    [
        ("BASH", "bash", "if true; then echo ok; fi\n"),
        ("ZSH", "zsh", "if true; then print ok; fi\n"),
        ("KSH", "ksh", "if true; then print ok; fi\n"),
        ("SH", "sh", "if true; then echo ok; fi\n"),
    ],
)
def test_shell_oracle_checks_declared_dialect(
    domain: str,
    shell_name: str,
    completion: str,
) -> None:
    if shutil.which(shell_name) is None:
        pytest.skip(f"{shell_name} is unavailable")
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id=f"{shell_name}-syntax",
        task_type="shell_syntax",
        prompt=completion,
        required_domains=(domain,),
        expected_sidecars=("token_domain_ids", "token_role_ids", "token_shell_edges"),
        oracle_kind="shell_syntax",
    )

    result = mod.shell_syntax_oracle(prompt, completion)

    assert result["status"] == "shell_syntax_passed"
    assert result["shell_kind"] == shell_name


def test_shell_oracle_rejects_invalid_syntax() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="bad-sh",
        task_type="shell_syntax",
        prompt="if true; then echo missing-fi\n",
        required_domains=("SH",),
        expected_sidecars=("token_domain_ids", "token_role_ids", "token_shell_edges"),
        oracle_kind="shell_syntax",
    )

    result = mod.shell_syntax_oracle(prompt, "if true; then echo missing-fi\n")

    assert result["status"] in {"shell_parse_failed", "shell_syntax_failed"}


def test_shell_oracle_reports_explicit_unavailability() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="missing-ksh",
        task_type="shell_syntax",
        prompt="print ok\n",
        required_domains=("KSH",),
        expected_sidecars=("token_domain_ids", "token_role_ids", "token_shell_edges"),
        oracle_kind="shell_syntax",
    )

    result = mod.shell_syntax_oracle(
        prompt,
        "print ok\n",
        tool_overrides={"ksh": None},
    )

    assert result["status"] == "shell_oracle_unavailable"
    assert result["failed_closed"] is True


@pytest.mark.parametrize(
    ("domain", "completion"),
    [
        ("CMAKE", "add_executable(app main.cpp)\n"),
        ("MAKE", "app: main.o\n\t$(CXX) -o app main.o\n"),
        ("NINJA", "rule cc\n  command = c++ -c $in -o $out\nbuild app.o: cc app.cc\n"),
        ("BAZEL", 'cc_binary(name = "app", srcs = ["main.cc"])\n'),
        ("CONFIGURE", "#!/bin/sh\nCC=cc\nexport CC\n"),
    ],
)
def test_build_structure_oracle_checks_declared_build_domain(
    domain: str,
    completion: str,
) -> None:
    mod = _load_eval_module()
    sidecar = "token_shell_edges" if domain == "CONFIGURE" else "token_build_edges"
    prompt = mod.DomainEvalPrompt(
        id=f"{domain.lower()}-structure",
        task_type="build_structure",
        prompt=completion,
        required_domains=(domain,),
        expected_sidecars=("token_domain_ids", "token_role_ids", sidecar),
        oracle_kind="build_structure",
    )

    result = mod.build_structure_oracle(prompt, completion)

    assert result["status"] == "build_structure_passed"
    assert result["build_kind"] == domain.lower()


def test_build_structure_oracle_rejects_unstructured_text() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="bad-cmake",
        task_type="build_structure",
        prompt="not a cmake command\n",
        required_domains=("CMAKE",),
        expected_sidecars=("token_domain_ids", "token_role_ids", "token_build_edges"),
        oracle_kind="build_structure",
    )

    result = mod.build_structure_oracle(prompt, "not a cmake command\n")

    assert result["status"] == "build_structure_failed"


def test_build_structure_oracle_enforces_required_graph_edges() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="cmake-link-graph",
        task_type="build_structure",
        prompt="link the Threads target",
        required_domains=("CMAKE",),
        expected_sidecars=("token_domain_ids", "token_role_ids", "token_build_edges"),
        oracle_kind="build_structure",
        oracle_edge_kinds=(20, 26),
        oracle_min_edges=2,
    )

    result = mod.build_structure_oracle(
        prompt,
        "find_package(Threads REQUIRED)\n"
        "target_link_libraries(app PRIVATE Threads::Threads)\n",
    )

    assert result["status"] == "build_structure_passed"
    assert {20, 26} <= set(result["edge_kinds"])

    missing_dep = mod.build_structure_oracle(
        prompt,
        "add_executable(app main.cpp)\n",
    )
    assert missing_dep["status"] == "build_structure_failed"


@pytest.mark.parametrize(
    ("domain", "completion", "edge_kind", "diagnostic_kind"),
    [
        (
            "COMPILER_ERROR",
            "src/main.cpp:7:5: error: use of undeclared identifier 'value'\n",
            60,
            "compiler",
        ),
        (
            "LINKER_ERROR",
            "ld: error: undefined reference to `pthread_create'\n",
            70,
            "linker",
        ),
    ],
)
def test_diagnostic_structure_oracle_parses_compiler_and_linker_packets(
    domain: str,
    completion: str,
    edge_kind: int,
    diagnostic_kind: str,
) -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id=f"{diagnostic_kind}-diagnostic",
        task_type="diagnostic_structure",
        prompt="return one normalized diagnostic packet",
        required_domains=(domain,),
        expected_sidecars=(
            "token_domain_ids",
            "token_role_ids",
            "token_diagnostic_edges",
        ),
        oracle_kind="diagnostic_structure",
        oracle_edge_kinds=(edge_kind,),
        oracle_min_edges=1,
    )

    result = mod.diagnostic_structure_oracle(prompt, completion)

    assert result["status"] == "diagnostic_structure_passed"
    assert result["diagnostic_kind"] == diagnostic_kind
    assert edge_kind in result["edge_kinds"]


def test_diagnostic_structure_oracle_rejects_unparsed_text() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="bad-compiler-diagnostic",
        task_type="diagnostic_structure",
        prompt="return one compiler diagnostic",
        required_domains=("COMPILER_ERROR",),
        expected_sidecars=(
            "token_domain_ids",
            "token_role_ids",
            "token_diagnostic_edges",
        ),
        oracle_kind="diagnostic_structure",
        oracle_edge_kinds=(60,),
        oracle_min_edges=1,
    )

    result = mod.diagnostic_structure_oracle(prompt, "everything is fine\n")

    assert result["status"] == "diagnostic_structure_failed"


def test_sidecar_structure_oracle_uses_frozen_graph_contract() -> None:
    mod = _load_eval_module()
    row = _case5_ksh_eval_row()
    path = Path("evals/domain_routed_prompts.jsonl")
    prompt = next(
        item
        for item in mod.load_prompts(path)
        if item.id == "ksh_build_sidecar_route"
    )

    result = mod.sidecar_structure_oracle(prompt, "The route is valid.")

    assert result["status"] == "sidecar_structure_passed"
    assert result["required_domains"] == ["KSH", "BUILD_ERROR"]
    assert result["cross_domain_edges"] == 1
    assert row["id"] == prompt.id


def test_sidecar_structure_oracle_fails_without_frozen_sidecars() -> None:
    mod = _load_eval_module()
    prompt = mod.DomainEvalPrompt(
        id="missing-sidecars",
        task_type="diagnostic",
        prompt="error: bad\n",
        required_domains=("COMPILER_ERROR",),
        expected_sidecars=("token_domain_ids", "token_role_ids", "token_diagnostic_edges"),
        oracle_kind="sidecar_structure",
    )

    result = mod.sidecar_structure_oracle(prompt, "add a header")

    assert result["status"] == "missing_sidecar_oracle"
    assert result["failed_closed"] is True


def test_domain_eval_rejects_cross_domain_edge_with_same_domain_endpoints(
    tmp_path: Path,
) -> None:
    mod = _load_eval_module()
    row = _case5_ksh_eval_row()
    row["prompt_sidecars"]["token_cross_domain_edges"] = [
        {"from": 2, "to": 3, "kind": 100}
    ]
    path = tmp_path / "same-domain-edge.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="distinct domain ids"):
        mod.load_prompts(path)


def test_evaluate_dispatches_shell_build_sidecar_and_diagnostic_oracles() -> None:
    mod = _load_eval_module()
    prompts = mod.load_prompts(Path("evals/domain_routed_prompts.jsonl"))[2:]
    completions = mod.load_completions(
        Path("evals/domain_routed_gold_completions.jsonl")
    )

    report = mod.evaluate(prompts, completions)

    assert [row["status"] for row in report["rows"]] == [
        "build_structure_passed",
        "shell_syntax_passed",
        "sidecar_structure_passed",
        "diagnostic_structure_passed",
        "diagnostic_structure_passed",
    ]
    assert report["oracle_passed"] == 5
    assert report["compile_passed"] == 0


def test_domain_eval_full_shipped_jsonl_is_green_with_gold_completions() -> None:
    mod = _load_eval_module()
    prompts = mod.load_prompts(Path("evals/domain_routed_prompts.jsonl"))
    completions = mod.load_completions(
        Path("evals/domain_routed_gold_completions.jsonl")
    )

    report = mod.evaluate(prompts, completions)

    assert report["prompts"] == 7
    assert report["completion_rows"] == 7
    assert report["passed"] is True
    assert report["failed"] == 0
