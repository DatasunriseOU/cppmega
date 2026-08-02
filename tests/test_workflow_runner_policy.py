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


def test_linux_workflow_timeout_covers_the_authoritative_lane() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-self-hosted.yml").read_text(
        encoding="utf-8"
    )
    jobs = {
        match.group("name"): match.group("body")
        for match in JOB_BLOCK.finditer(workflow.partition("\njobs:\n")[2])
    }
    assert "linux-portable" in jobs
    timeout_match = re.search(
        r"(?m)^\s+timeout-minutes:\s*(\d+)\s*$", jobs["linux-portable"]
    )
    assert timeout_match is not None
    workflow_minutes = int(timeout_match.group(1))
    payload = json.loads(
        (REPO_ROOT / "configs" / "ci" / "lanes.json").read_text(encoding="utf-8")
    )
    lane = next(item for item in payload["lanes"] if item["id"] == "linux-contracts")

    assert workflow_minutes * 60 >= lane["timeout_seconds"] + 300


def test_macos_workflow_does_not_expand_an_empty_array_under_bash_3() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-self-hosted.yml").read_text(
        encoding="utf-8"
    )

    assert "mlx_contract_args=()" not in workflow
    assert '"${mlx_contract_args[@]}"' not in workflow
    assert workflow.count("scripts/data/verify_tokenizer_contract.py") == 3
    assert workflow.count('--mlx-root "${CPPMEGA_MLX_REFERENCE_ROOT}"') == 1


def test_macos_lane_writes_a_pre_python_failure_receipt() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-self-hosted.yml").read_text(
        encoding="utf-8"
    )
    jobs = {
        match.group("name"): match.group("body")
        for match in JOB_BLOCK.finditer(workflow.partition("\njobs:\n")[2])
    }
    macos = jobs["mac-contracts"]

    assert "set -eEuo pipefail" in macos
    assert "trap 'on_pre_python_error' ERR" in macos
    assert '"schema_version": "cppmega.repository-ci.v1"' in macos
    assert '"failure_stage": "workflow-preamble"' in macos
    assert 'pre_python_step="python_bin is executable"' in macos
    assert 'pre_python_step="verify tokenizer contract' in macos
    # the trap is disarmed before the orchestrator writes its own receipts
    assert macos.rindex("trap - ERR") < macos.index(
        "scripts/ci/run_repository_ci.py lane"
    )

    summary_step = re.search(
        r"(?ms)^      - name: Surface macOS lane failure receipt in job summary\n"
        r"(?P<body>.*?)(?=^      - name:|\Z)",
        macos,
    )
    assert summary_step is not None
    summary_body = summary_step.group("body")
    assert "if: failure()" in summary_body
    assert "GITHUB_STEP_SUMMARY" in summary_body
    assert 'cat "${receipt}"' in summary_body


def test_linux_lane_writes_a_pre_python_failure_receipt() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-self-hosted.yml").read_text(
        encoding="utf-8"
    )
    jobs = {
        match.group("name"): match.group("body")
        for match in JOB_BLOCK.finditer(workflow.partition("\njobs:\n")[2])
    }
    linux = jobs["linux-portable"]

    assert "set -eEuo pipefail" in linux
    assert "trap 'on_pre_python_error' ERR" in linux
    assert '"schema_version": "cppmega.repository-ci.v1"' in linux
    assert '"failure_stage": "workflow-preamble"' in linux
    assert 'pre_python_step="checkout matches GITHUB_SHA"' in linux
    assert 'pre_python_step="verify tokenizer contract"' in linux
    # the trap is disarmed before the orchestrator writes its own receipts
    assert linux.rindex("trap - ERR") < linux.index(
        "scripts/ci/run_repository_ci.py lane"
    )

    summary_step = re.search(
        r"(?ms)^      - name: Surface Linux lane failure receipt in job summary\n"
        r"(?P<body>.*?)(?=^      - name:|\Z)",
        linux,
    )
    assert summary_step is not None
    summary_body = summary_step.group("body")
    assert "if: failure()" in summary_body
    assert "GITHUB_STEP_SUMMARY" in summary_body
    assert 'cat "${receipt}"' in summary_body


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


def test_mamba_wheel_build_applies_the_pinned_gqa_backward_patch() -> None:
    patch = "upstream_prs/05_mamba3_dt_fp32_gqa_bwd.patch"
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-wheels.yml"
    ).read_text(encoding="utf-8")
    stack = (REPO_ROOT / "STACK.lock").read_text(encoding="utf-8")

    stack_block = re.search(
        r"(?ms)^  mamba_ssm:\n(?P<body>.*?)(?=^  \S[^:]*:\n|\Z)",
        stack,
    )
    workflow_block = re.search(
        r"(?ms)^          - name: mamba_ssm\n"
        r"(?P<body>.*?)(?=^          - name:|\Z)",
        workflow,
    )

    assert stack_block is not None
    assert workflow_block is not None
    for key in ("repo", "ref", "patch"):
        stack_value = re.search(
            rf"(?m)^    {key}: (?P<value>\S+)$",
            stack_block.group("body"),
        )
        workflow_value = re.search(
            rf"(?m)^            {key}: (?P<value>\S+)$",
            workflow_block.group("body"),
        )
        assert stack_value is not None
        assert workflow_value is not None
        assert stack_value.group("value") == workflow_value.group("value")
    assert f"    patch: {patch}" in stack_block.group("body")
    assert f"            patch: {patch}" in workflow_block.group("body")
    assert "MAMBA_FORCE_BUILD=TRUE" in stack_block.group("body")
    assert "MAMBA_FORCE_BUILD=TRUE" in workflow_block.group("body")
    assert (REPO_ROOT / patch).is_file()
    assert 'if: matrix.patch' in workflow
    assert 'git apply --check "$PATCH"' in workflow
    assert "Verify Mamba wheel contains the pinned GQA backward patch" in workflow
    assert "wheel_bytes != source_bytes" in workflow
    assert "F.softplus((dd_dt + self.dt_bias).to(torch.float32))" in workflow
    verify_step = re.search(
        r"(?ms)^      - name: Verify Mamba wheel contains the pinned GQA "
        r"backward patch\n(?P<body>.*?)(?=^      - name:|\Z)",
        workflow,
    )
    assert verify_step is not None
    verify_body = verify_step.group("body")
    for module in ("mamba3_mimo_bwd.py", "mamba3_mimo_bwd_varlen.py"):
        marker = re.compile(
            r'"mamba_ssm/ops/tilelang/mamba3/'
            + re.escape(module)
            + r'":\s*\[\s*"elif H % G == 0:",?\s*\]'
        )
        assert marker.search(verify_body), (
            f"verify step must pin the GQA marker for {module}"
        )


def test_assembled_images_import_the_patched_mamba_runtime() -> None:
    for path in ("docker/Dockerfile", "docker/Dockerfile.beta23"):
        dockerfile = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert (
            "from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import "
            "mamba_mimo_bwd_combined"
        ) in dockerfile
        assert (
            "from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd_varlen import "
            "mamba_mimo_bwd_combined_varlen"
        ) in dockerfile
        assert "assert _mamba3_module.mamba3_mimo_combined is not None" in dockerfile


def test_transformer_engine_wheel_build_uses_upstream_arch_knob() -> None:
    transformer_engine_commit = "4220403e831d29e93868f7793693ea83f6b8b05b"
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-wheels.yml"
    ).read_text(encoding="utf-8")
    stack = (REPO_ROOT / "STACK.lock").read_text(encoding="utf-8")

    assert f"ref: {transformer_engine_commit}" in workflow
    assert f"ref: {transformer_engine_commit}" in stack
    assert "NVTE_CUDA_ARCHS='90;100;120'" in workflow
    assert " CUDAARCHS=" not in workflow
    assert (
        "python -m pip install pydantic importlib-metadata nvdlfw-inspect onnx "
        "onnxscript"
    ) in workflow
    assert (
        "python -m pip install --force-reinstall --no-deps "
        "wheels/transformer_engine-*.whl"
    ) in workflow


def test_wheel_build_checks_out_the_resolved_source_commit() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-wheels.yml"
    ).read_text(encoding="utf-8")

    assert 'git checkout --detach "${{ steps.src.outputs.sha }}"' in workflow
    assert 'test "$(git rev-parse HEAD)" = "${{ steps.src.outputs.sha }}"' in workflow
    assert "git checkout ${{ matrix.ref }}" not in workflow


def test_wheel_and_image_sources_are_content_addressed() -> None:
    stack = (REPO_ROOT / "STACK.lock").read_text(encoding="utf-8")
    wheel_workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-wheels.yml"
    ).read_text(encoding="utf-8")
    image_workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-image.yml"
    ).read_text(encoding="utf-8")
    dockerfiles = [
        (REPO_ROOT / name).read_text(encoding="utf-8")
        for name in ("docker/Dockerfile", "docker/Dockerfile.beta23")
    ]

    stack_refs = re.findall(r"^    ref: (\S+)$", stack, re.MULTILINE)
    wheel_refs = re.findall(
        r"^            ref: (\S+)$", wheel_workflow, re.MULTILINE
    )
    assert len(stack_refs) == 10
    assert len(wheel_refs) == 8
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in stack_refs)
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in wheel_refs)
    assert re.search(
        r"^  cuda_image: \S+@sha256:[0-9a-f]{64}$", stack, re.MULTILINE
    )
    assert "sha256sum *.whl > SHA256SUMS" in wheel_workflow
    assert "wheels/SHA256SUMS --clobber" in wheel_workflow
    assert 'echo "tag=${{ github.sha }}"' in wheel_workflow
    assert '--target "${{ github.sha }}"' in wheel_workflow
    assert '"$tag_sha" != "${{ github.sha }}"' in wheel_workflow
    assert "inputs.tag" not in wheel_workflow
    assert "--pattern SHA256SUMS" in image_workflow
    assert "sha256sum -c SHA256SUMS" in image_workflow
    assert "awk '{print $2}' SHA256SUMS" in image_workflow
    assert "<(printf '%s\\n' *.whl" in image_workflow
    for dockerfile in dockerfiles:
        assert (
            "nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04"
            "@sha256:6435dc5a825b0095648d87a3c91240fd7788a85fafaf215739544d389ab74366"
        ) in dockerfile
        assert (
            "ARG MEGATRON_REF=ba7b5ebce12af60627a80985792a1449ce45f46c"
            in dockerfile
        )
        assert 'git -C /opt/megatron-lm checkout --detach FETCH_HEAD' in dockerfile
        assert 'rev-parse HEAD)" = "${MEGATRON_REF}"' in dockerfile


def test_image_build_binds_triggering_source_and_wheel_release() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-image.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count("github.event.workflow_run.head_sha") == 3
    assert workflow.count("inputs.source_sha") == 3
    assert 'test "$(git rev-parse HEAD)" = "$SOURCE_SHA"' in workflow
    assert 'test -z "$(git status --porcelain=v1 --untracked-files=all)"' in workflow
    for build_arg in (
        "CPPMEGA_SOURCE_SHA=${{ steps.source.outputs.sha }}",
        "CPPMEGA_SOURCE_TREE=${{ steps.source.outputs.tree }}",
        (
            "CPPMEGA_SOURCE_MANIFEST_SHA256="
            "${{ steps.source.outputs.manifest_sha256 }}"
        ),
        (
            "CPPMEGA_SOURCE_MANIFEST_FILE_COUNT="
            "${{ steps.source.outputs.manifest_file_count }}"
        ),
    ):
        assert build_arg in workflow
    assert 'TAG="wheels-${SOURCE_SHA}"' in workflow
    assert "inputs.wheels_tag" not in workflow
    assert "git/ref/tags/${TAG}" in workflow
    assert '"$tag_sha" != "$SOURCE_SHA"' in workflow
    assert "type=raw,value=sha-${{ steps.rel.outputs.source_sha }}" in workflow
    assert "type=raw,value=${{ steps.rel.outputs.short_sha }}" in workflow
    assert (
        "org.opencontainers.image.revision="
        "${{ steps.rel.outputs.source_sha }}"
    ) in workflow
    assert "{{is_default_branch}}" not in workflow
    assert "github.event.workflow_run.head_branch" in workflow
    assert "gh release list" not in workflow
    assert "type=sha," not in workflow


def test_tilelang_tvm_pin_and_wheel_name_are_consistent() -> None:
    tilelang_commit = "de8bb88cc382b0e78bc804244f79c4be8cc9e75f"
    tvm_commit = "e25ca6ae50beee0e907b1e5ed32949879caddde1"
    tvm_ffi_commit = "521efeb30bfd9e4946b248b3d76e6391028233a3"
    wheel_name = "tilelang-0.1.9-cp38-abi3-linux_x86_64.whl"
    ffi_wheel_name = (
        "apache_tvm_ffi-0.1.13.post5-cp313-cp313-linux_x86_64.whl"
    )

    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-wheels.yml"
    ).read_text(encoding="utf-8")
    image_workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-image.yml"
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
    assert "threadblock_swizzle_pattern" in workflow
    assert "is_pure_function(tvm.tirx.PrimFunc([], body))" in workflow
    modal_runtime = [
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "scripts/modal_fa4_beta23_parity.py",
        )
    ]
    dockerfiles = [
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in ("docker/Dockerfile", "docker/Dockerfile.beta23")
    ]

    assert f"ref: {tilelang_commit}" in workflow
    assert f"ref: {tilelang_commit}" in stack
    for text in (rebuild, install, modal_build):
        assert tilelang_commit in text
        assert tvm_commit in text
        assert tvm_ffi_commit in text
    for text in (install, modal_build, modal_base):
        assert wheel_name in text
        assert ffi_wheel_name in text
    assert "--no-build-isolation" not in install
    assert "Smoke TileLang wheel linkage and import" in workflow
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_APACHE_TVM_FFI=0.1.13.post5" in workflow
    assert "wheels/apache_tvm_ffi-*.whl" in workflow
    assert "'apache_tvm_ffi-*.whl'" in workflow
    assert "'apache_tvm_ffi-*.whl'" in image_workflow
    assert rebuild.index("git submodule update --init --recursive 3rdparty/tvm") < (
        rebuild.index("3rdparty/tvm/3rdparty/tvm-ffi rev-parse HEAD")
    )
    assert "python - <<'PY'" in modal_build
    assert 'python -c "{verify_code.strip()}"' not in modal_build
    assert "pip wheel . --no-build-isolation --no-deps" in modal_build
    for text in (
        workflow,
        stack,
        rebuild,
        install,
        modal_build,
        modal_base,
        *modal_runtime,
        *dockerfiles,
    ):
        assert "z3-solver==4.15.4.0" in text
    assert 'f"STDERR tail:\\n{r.stderr[-24_000:]}"' in modal_build
    assert "Shared library: [libcuda_stub.so]" in workflow
    assert "version: 0.1.9" in stack
    assert "version: 0.1.13.post5" in stack
    assert "/tmp/cppmega_wheels" not in modal_base
    assert '_REPO_ROOT / "wheels"' in modal_base
