from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZE_ENTRYPOINT = REPO_ROOT / "scripts" / "data" / "prepare_tokenize_megacpp.py"
DATA_ENTRYPOINT = REPO_ROOT / "scripts" / "data" / "prepare_data.sh"
SOURCE_CONVEYOR = REPO_ROOT / "scripts" / "data" / "source_conveyor.py"
MEGATRON_CONVERTER = REPO_ROOT / "scripts" / "data_prep_parquet_to_megatron.py"


def _load_source_conveyor_module():
    spec = importlib.util.spec_from_file_location(
        "cppmega_source_conveyor_test", SOURCE_CONVEYOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entrypoint_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MEGACPP_NANOCHAT_ROOT"] = str(tmp_path / "missing-nanochat")
    env["MEGACPP_DATA_ROOT"] = str(tmp_path / "unused-data-root")
    env.pop("PYTHONPATH", None)
    return env


def _run(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _dry_run_commands(output: str) -> list[list[str]]:
    return [
        shlex.split(line.removeprefix("DRY-RUN "))
        for line in output.splitlines()
        if line.startswith("DRY-RUN ")
    ]


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_source_conveyor_default_is_the_complete_five_bucket_ladder() -> None:
    conveyor = _load_source_conveyor_module()

    args = conveyor.build_arg_parser().parse_args(
        ["--source-root", "/sources", "--output-root", "/packed"]
    )

    assert args.target_lengths == (1024, 2048, 4096, 8192, 16384)


def test_public_data_entrypoints_default_to_the_complete_five_bucket_ladder(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    output_root = tmp_path / "packed"
    tokenizer_result = _run(
        [
            sys.executable,
            str(TOKENIZE_ENTRYPOINT),
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        env=_entrypoint_env(tmp_path),
    )
    assert tokenizer_result.returncode == 0, tokenizer_result.stderr
    tokenizer_command = _dry_run_commands(tokenizer_result.stdout)[0]
    assert _option_value(tokenizer_command, "--target-lengths") == (
        "1024,2048,4096,8192,16384"
    )

    shell_result = _run(
        [
            "bash",
            str(DATA_ENTRYPOINT),
            "tokenize",
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        env=_entrypoint_env(tmp_path),
    )
    assert shell_result.returncode == 0, shell_result.stderr
    shell_command = _dry_run_commands(shell_result.stdout)[0]
    assert _option_value(shell_command, "--target-lengths") == (
        "1024,2048,4096,8192,16384"
    )


def test_root_source_entrypoint_writes_schema_v2_manifest(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for relative in ("scripts", "cppmega", "tools/clang_indexer"):
        shutil.copytree(
            REPO_ROOT / relative,
            checkout / relative,
            ignore=ignored,
        )
    (checkout / ".gitignore").write_text(
        "__pycache__/\n*.py[cod]\n",
        encoding="utf-8",
    )
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "Root Entry Test"],
        ["git", "config", "user.email", "root-entry@example.test"],
        ["git", "add", "."],
        ["git", "commit", "-q", "-m", "fixture"],
    ):
        subprocess.run(command, cwd=checkout, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    source_root = tmp_path / "sources"
    source_root.mkdir()
    output_root = tmp_path / "packed"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(checkout / "scripts" / "data" / "source_conveyor.py"),
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--tokenizer",
            str(checkout / "cppmega" / "tokenizer" / "tokenizer.json"),
            "--max-repos",
            "0",
            "--min-free-disk-gb",
            "0",
            "--expected-code-revision",
            revision,
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(
        (output_root / ".conveyor" / "_done.json").read_text(encoding="utf-8")
    )
    receipt = manifest["code_revision"]
    assert receipt["schema_version"] == 2
    assert receipt["producer_role"] == "canonical_source_conveyor"
    assert receipt["repository_identity"] == "cppmega"
    assert receipt["git_commit"] == revision
    assert receipt["dirty"] is False
    assert len(receipt["source_tree_sha256"]) == 64
    assert receipt["indexer_dependency_closure_sha256"] == (
        receipt["indexer_provenance"]["dependency_closure_sha256"]
    )
    assert {
        int(path.name)
        for path in output_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    } == {1024, 2048, 4096, 8192, 16384}


def test_root_data_entrypoint_help_is_explicit_and_has_no_sibling_contract(
    tmp_path: Path,
) -> None:
    env = _entrypoint_env(tmp_path)

    python_help = _run([sys.executable, str(TOKENIZE_ENTRYPOINT), "--help"], env=env)
    conveyor_help = _run([sys.executable, str(SOURCE_CONVEYOR), "--help"], env=env)
    shell_help = _run(["bash", str(DATA_ENTRYPOINT), "--help"], env=env)

    assert python_help.returncode == 0, python_help.stderr
    assert conveyor_help.returncode == 0, conveyor_help.stderr
    assert shell_help.returncode == 0, shell_help.stderr
    combined_help = python_help.stdout + conveyor_help.stdout + shell_help.stdout
    for option in (
        "--source-root",
        "--output-root",
        "--tokenizer",
        "--target-lengths",
        "--dry-run",
    ):
        assert option in combined_help

    for path in (TOKENIZE_ENTRYPOINT, SOURCE_CONVEYOR, DATA_ENTRYPOINT):
        source = path.read_text(encoding="utf-8")
        assert "MEGACPP_NANOCHAT_ROOT" not in source
        assert "cppmega.mlx" not in source


def test_prepare_tokenize_dry_run_delegates_only_to_local_conveyor(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    output_root = tmp_path / "packed"
    tokenizer = REPO_ROOT / "data" / "tokenizer_v2" / "tokenizer.json"
    repo_list = tmp_path / "repo-list.json"
    expected_revision = "b" * 40

    result = _run(
        [
            sys.executable,
            str(TOKENIZE_ENTRYPOINT),
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--tokenizer",
            str(tokenizer),
            "--target-lengths",
            "4096, 1024,4096",
            "--repo-list",
            str(repo_list),
            "--max-repos",
            "0",
            "--min-free-disk-gb",
            "0",
            "--expected-code-revision",
            expected_revision,
            "--dry-run",
        ],
        env=_entrypoint_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    commands = _dry_run_commands(result.stdout)
    assert commands == [
        [
            sys.executable,
            str(SOURCE_CONVEYOR),
            "--source-root",
            str(source_root.resolve()),
            "--output-root",
            str(output_root.resolve()),
            "--tokenizer",
            str(tokenizer.resolve()),
            "--target-lengths",
            "1024,4096",
            "--repo-list",
            str(repo_list.resolve()),
            "--max-repos",
            "0",
            "--min-free-disk-gb",
            "0.0",
            "--expected-code-revision",
            expected_revision,
        ]
    ]
    assert "missing-nanochat" not in result.stdout + result.stderr


def test_source_conveyor_non_dry_run_configures_streaming_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    output_root = tmp_path / "packed"
    tokenizer = REPO_ROOT / "data" / "tokenizer_v2" / "tokenizer.json"
    conveyor = _load_source_conveyor_module()
    captured: dict[str, object] = {}
    backend = SimpleNamespace(
        sr=SimpleNamespace(TOKENIZER_PATH=Path("wrong-tokenizer.json")),
    )

    def fake_main(argv: list[str]) -> int:
        captured["argv"] = argv
        captured["tokenizer"] = backend.sr.TOKENIZER_PATH
        return 17

    backend.main = fake_main
    monkeypatch.setattr(conveyor, "_load_backend", lambda: backend)
    monkeypatch.setattr(conveyor, "_current_revision", lambda: "a" * 40)

    result = conveyor.main(
        [
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--tokenizer",
            str(tokenizer),
            "--target-lengths",
            "4096,1024",
        ]
    )

    assert result == 17
    assert captured["tokenizer"] == tokenizer.resolve()
    backend_argv = captured["argv"]
    assert isinstance(backend_argv, list)
    assert "--streams" in backend_argv
    assert _option_value(backend_argv, "--streams") == "code"
    assert _option_value(backend_argv, "--source-dir-root") == str(
        source_root.resolve()
    )
    assert _option_value(backend_argv, "--code-output-root") == str(
        output_root.resolve()
    )
    assert _option_value(backend_argv, "--target-lengths-code") == "1024,4096"
    assert _option_value(backend_argv, "--expected-code-revision") == "a" * 40


def test_prepare_data_dry_run_keeps_conveyor_converter_and_contract_paths_aligned(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    source_root = tmp_path / "sources"
    output_root = tmp_path / "packed"
    tokenizer = tmp_path / "tokenizer" / "tokenizer.json"
    tokenizer_contract = tokenizer.with_name("tokenizer_contract_v1.json")
    env = _entrypoint_env(tmp_path)
    common = [
        "--dry-run",
        "--data-root",
        str(data_root),
        "--source-root",
        str(source_root),
        "--output-root",
        str(output_root),
        "--tokenizer",
        str(tokenizer),
        "--target-lengths",
        "1024,4096",
        "--dataset-name",
        "fixture",
    ]

    tokenize = _run(["bash", str(DATA_ENTRYPOINT), "tokenize", *common], env=env)
    audit = _run(["bash", str(DATA_ENTRYPOINT), "audit", *common], env=env)
    format_result = _run(
        ["bash", str(DATA_ENTRYPOINT), "format", *common], env=env
    )

    assert tokenize.returncode == 0, tokenize.stderr
    assert audit.returncode == 0, audit.stderr
    assert format_result.returncode == 0, format_result.stderr

    tokenize_commands = _dry_run_commands(tokenize.stdout)
    assert len(tokenize_commands) == 1
    tokenize_command = tokenize_commands[0]
    assert Path(tokenize_command[1]) == TOKENIZE_ENTRYPOINT
    assert _option_value(tokenize_command, "--source-root") == str(source_root)
    assert _option_value(tokenize_command, "--output-root") == str(output_root)
    assert _option_value(tokenize_command, "--tokenizer") == str(tokenizer)
    assert _option_value(tokenize_command, "--target-lengths") == "1024,4096"

    audit_commands = _dry_run_commands(audit.stdout)
    contract_command = next(
        command
        for command in audit_commands
        if command[1].endswith("verify_tokenizer_contract.py")
    )
    assert _option_value(contract_command, "--contract") == str(tokenizer_contract)
    assert _option_value(contract_command, "--tokenizer") == str(tokenizer)
    assert _option_value(contract_command, "--domain-schema") == str(
        REPO_ROOT / "data" / "domain_schema_v1.json"
    )

    manifest_commands = [
        command
        for command in audit_commands
        if command[1].endswith("build_dataset_manifest.py")
    ]
    assert len(manifest_commands) == 2
    for length, command in zip((1024, 4096), manifest_commands, strict=True):
        assert _option_value(command, "--dataset-dir") == str(output_root / str(length))
        assert _option_value(command, "--seq-len") == str(length)
        assert _option_value(command, "--contract") == str(tokenizer_contract)
        assert _option_value(command, "--tokenizer") == str(tokenizer)

    converter_commands = _dry_run_commands(format_result.stdout)
    assert len(converter_commands) == 2
    for length, command in zip((1024, 4096), converter_commands, strict=True):
        assert Path(command[1]) == MEGATRON_CONVERTER
        assert _option_value(command, "--input-dir") == str(output_root / str(length))
        assert _option_value(command, "--output-prefix") == str(
            data_root / "megatron" / f"fixture_{length}_train"
        )
        assert _option_value(command, "--split") == "all"
