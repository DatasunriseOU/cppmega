from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOSTED_RUNNER = re.compile(
    r"^\s*runs-on:\s*.*(?:ubuntu|macos|windows)-latest\s*$",
    re.MULTILINE,
)
JOB_BLOCK = re.compile(
    r"(?ms)^  (?P<name>[A-Za-z0-9_-]+):\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)"
)
TRUSTED_PR_GUARDS = (
    "github.event.pull_request.head.repo.full_name == github.repository",
    "github.event.pull_request.head.repo.fork == false",
)
NON_PR_ONLY_GUARD = (
    "github.event_name == 'schedule' || "
    "github.event_name == 'workflow_dispatch'"
)
ACTION_USE = re.compile(
    r"uses:\s*['\"]?(?P<action>actions/[A-Za-z0-9_.-]+)"
    r"@(?P<ref>[^\s'\"#]+)"
)
DOMAIN_CONTRACT_TESTS = (
    "tests/test_case5_ksh_domain_contract.py",
    "tests/test_eval_domain_routed_codegen.py",
    "tests/test_ksh_python_domain_parsers.py",
)
CI_STREAMING_CONTRACT_TESTS = frozenset(
    {
        "tests/test_ci_stream_inventory.py",
        "tests/test_ci_content_store.py",
        "tests/test_ci_log_sidecars.py",
        "tests/test_ci_stream_fetch.py",
        "tests/test_ci_job_log_rescue.py",
        "tests/test_recover_ci_preserved_archives.py",
        "tests/test_ci_source_binding_projection.py",
        "tests/test_ci_source_sidecars.py",
        "tests/test_export_ci_content_store_case5.py",
        "tests/test_merge_ci_stream_shards.py",
    }
)


def test_workflows_do_not_use_github_hosted_runners() -> None:
    violations = []
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        if HOSTED_RUNNER.search(workflow.read_text(encoding="utf-8")):
            violations.append(workflow.relative_to(REPO_ROOT).as_posix())

    assert not violations, f"GitHub-hosted runners are forbidden: {violations}"


def test_pull_requests_cannot_execute_on_persistent_self_hosted_runners() -> None:
    violations = []
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        if not re.search(r"(?m)^  pull_request:\s*$", text):
            continue
        jobs = text.partition("\njobs:\n")[2]
        for match in JOB_BLOCK.finditer(jobs):
            body = match.group("body")
            if "runs-on: [self-hosted" not in body:
                continue
            guarded = any(marker in body for marker in TRUSTED_PR_GUARDS)
            guarded = guarded or NON_PR_ONLY_GUARD in body
            if not guarded:
                violations.append(
                    f"{workflow.relative_to(REPO_ROOT).as_posix()}:{match.group('name')}"
                )

    assert not violations, (
        "pull_request jobs may not execute untrusted code on persistent "
        f"self-hosted runners: {violations}"
    )


def test_persistent_pr_ci_actions_are_pinned_to_commits() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ci-self-hosted.yml"
    ).read_text(encoding="utf-8")
    violations = [
        f"{match.group('action')}@{match.group('ref')}"
        for match in ACTION_USE.finditer(workflow)
        if re.fullmatch(r"[0-9a-f]{40}", match.group("ref")) is None
    ]

    assert not violations, f"mutable action references are forbidden: {violations}"


def test_workflow_delegates_to_authoritative_lanes_with_source_binding() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-self-hosted.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count("scripts/ci/run_repository_ci.py lane") == 2
    assert workflow.count("--lanes-config") == 2
    assert "--lane macos-contracts" in workflow
    assert "--lane linux-contracts" in workflow
    assert workflow.count("--expected-source-commit \"${GITHUB_SHA}\"") == 2
    assert workflow.count("--expected-source-tree \"${expected_tree}\"") == 2
    assert workflow.count("test \"$(git rev-parse HEAD)\" = \"${GITHUB_SHA}\"") == 2
    assert " -m pytest " not in workflow

    payload = json.loads(
        (REPO_ROOT / "configs" / "ci" / "lanes.json").read_text(encoding="utf-8")
    )
    lanes = {lane["id"]: lane for lane in payload["lanes"]}
    assert lanes["linux-contracts"]["test_profile"] == "portable-data"


def test_macos_workflow_does_not_expand_an_empty_array_under_bash_3() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-self-hosted.yml").read_text(
        encoding="utf-8"
    )

    assert "mlx_contract_args=()" not in workflow
    assert '"${mlx_contract_args[@]}"' not in workflow
    assert workflow.count("scripts/data/verify_tokenizer_contract.py") == 3
    assert workflow.count('--mlx-root "${CPPMEGA_MLX_REFERENCE_ROOT}"') == 1


def test_frozen_domain_eval_is_wired_into_repository_owned_ci() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-self-hosted.yml").read_text(
        encoding="utf-8"
    )
    assert workflow.count("scripts/ci/run_repository_ci.py lane") == 2
    assert "tests/test_eval_domain_routed_codegen.py" not in workflow

    payload = json.loads(
        (REPO_ROOT / "configs" / "ci" / "lanes.json").read_text(encoding="utf-8")
    )
    lanes = {lane["id"]: lane for lane in payload["lanes"]}
    for lane_id in ("macos-contracts", "linux-contracts"):
        commands = {command["name"]: command for command in lanes[lane_id]["commands"]}
        pytest_argv = next(
            command["argv"]
            for command in commands.values()
            if command["argv"][:4] == ["{python}", "-m", "pytest", "-q"]
        )
        assert set(DOMAIN_CONTRACT_TESTS) <= set(pytest_argv)
        assert commands["frozen-domain-eval"]["argv"] == [
            "{python}",
            "scripts/eval_domain_routed_codegen.py",
            "--prompts",
            "evals/domain_routed_prompts.jsonl",
            "--completions",
            "evals/domain_routed_gold_completions.jsonl",
            "--out",
            "outputs/ci_diagnostics/domain-routed-codegen.json",
        ]


def test_ci_streaming_contracts_are_portable_and_run_on_both_platforms() -> None:
    import conftest

    payload = json.loads(
        (REPO_ROOT / "configs" / "ci" / "lanes.json").read_text(encoding="utf-8")
    )
    lanes = {lane["id"]: lane for lane in payload["lanes"]}

    assert CI_STREAMING_CONTRACT_TESTS <= conftest._PORTABLE_TEST_ALLOWLIST
    for lane_id in ("macos-contracts", "linux-contracts"):
        pytest_argv = next(
            command["argv"]
            for command in lanes[lane_id]["commands"]
            if command["argv"][:4] == ["{python}", "-m", "pytest", "-q"]
        )
        assert CI_STREAMING_CONTRACT_TESTS <= set(pytest_argv)
        assert "tests/test_build_macro_routes_megatron_bundle.py" in pytest_argv


def test_wheel_build_keeps_gcc_and_gxx_in_one_alternatives_group() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-wheels.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count("--slave /usr/bin/g++ g++ /usr/bin/g++-15") == 2
    assert workflow.count("--set gcc /usr/bin/gcc-15") == 2
    assert "update-alternatives --install /usr/bin/g++ g++" not in workflow
    assert workflow.count('test "$(readlink -f /usr/bin/g++)"') == 2


def test_wheel_build_uses_torch_matched_cuda_libraries_without_system_replacement() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-wheels.yml"
    ).read_text(encoding="utf-8")

    assert '"cuda-toolkit-13-2=${CUDA_TOOLKIT_DEB_VERSION}"' in workflow
    assert "cudnn9-cuda-13 libnccl-dev libnccl2" not in workflow
    assert '"nvidia-cudnn-cu13": metadata.version("nvidia-cudnn-cu13")' in workflow
    assert '"nvidia-nccl-cu13": metadata.version("nvidia-nccl-cu13")' in workflow
    assert 'if torch.version.cuda != "13.2":' in workflow
    assert 'stream.write(f"CUDNN_PATH={cudnn_path}\\n")' in workflow
    assert 'nccl_path = resolve_package_root("nvidia.nccl")' in workflow
    assert 'nccl_path / "include/nccl.h"' in workflow
    assert 'glob("libnccl.so.2*")' in workflow
    assert 'stream.write(f"NCCL_PATH={nccl_path}\\n")' in workflow
    assert 'f"CPATH={nccl_path / \'include\'}:"' in workflow
    assert 'f"{cudnn_path / \'include\'}:"' in workflow
    assert 'f"LIBRARY_PATH={nccl_path / \'lib\'}:"' in workflow
    assert 'f"{cudnn_path / \'lib\'}:"' in workflow
    assert "grep -q 'release 13\\.2'" in workflow


def test_fa2_wheel_build_uses_upstream_arch_knob_and_exact_kernel_trim() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-wheels.yml"
    ).read_text(encoding="utf-8")

    assert "FLASH_ATTN_CUDA_ARCHS=120" in workflow
    assert "flash_fwd_split(_align)?_hdim" in workflow
    assert '[[ "$after" -ne 8 || -n "$unexpected" ]]' in workflow


def test_tilelang_tvm_pin_and_wheel_name_are_consistent() -> None:
    tilelang_commit = "8fdff5aa10f4265a4cb0e114d4c62613f3982180"
    tvm_commit = "36105156803007279d6ff481621b083441503cde"
    wheel_name = "tilelang-0.1.9-cp38-abi3-linux_x86_64.whl"

    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-wheels.yml"
    ).read_text(encoding="utf-8")
    stack = (REPO_ROOT / "STACK.lock").read_text(encoding="utf-8")
    rebuild = (REPO_ROOT / "scripts" / "rebuild_tilelang_wheel.sh").read_text(
        encoding="utf-8"
    )
    install = (REPO_ROOT / "scripts" / "install_tilelang_wheel.sh").read_text(
        encoding="utf-8"
    )
    modal_build = (
        REPO_ROOT / "scripts" / "modal_build_tilelang_beta23.py"
    ).read_text(encoding="utf-8")
    modal_base = (REPO_ROOT / "scripts" / "modal_cppmega_base.py").read_text(
        encoding="utf-8"
    )

    assert f"ref: {tilelang_commit}" in workflow
    assert f"ref: {tilelang_commit}" in stack
    for text in (rebuild, install, modal_build):
        assert tilelang_commit in text
        assert tvm_commit in text
    for text in (install, modal_build, modal_base):
        assert wheel_name in text
    assert "version: 0.1.9" in stack
    assert "/tmp/cppmega_wheels" not in modal_base
    assert '_REPO_ROOT / "wheels"' in modal_base
