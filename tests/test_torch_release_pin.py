from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_cuda_stack_uses_stable_torch_213_release() -> None:
    stack = _read("STACK.lock")
    dockerfile = _read("docker/Dockerfile")
    workflow = _read(".github/workflows/build-wheels.yml")

    assert 'torch: "2.13.0+cu132"' in stack
    assert "https://download.pytorch.org/whl/cu132" in stack
    assert "nightly/cu132" not in stack

    assert "ARG TORCH_VERSION=2.13.0+cu132" in dockerfile
    assert "ARG TORCH_INDEX=https://download.pytorch.org/whl/cu132" in dockerfile
    assert "pip install --pre" not in dockerfile

    assert 'TORCH_VERSION: "2.13.0+cu132"' in workflow
    assert 'TORCH_INDEX: "https://download.pytorch.org/whl/cu132"' in workflow
    assert "pip install --pre" not in workflow


def test_fa4_beta23_and_tvm_ffi_runtime_pins_are_exact() -> None:
    stack = _read("STACK.lock")

    assert 'package: "apache-tvm-ffi==0.1.13.post5"' in stack
    assert 'package: "flash-attn-4[cu13]==4.0.0b23"' in stack
    assert 'package: "quack-kernels==0.5.3"' in stack
    for path, version_check in (
        (
            "scripts/modal_build_tilelang_beta23.py",
            'metadata.version("flash-attn-4")',
        ),
        (
            "scripts/modal_cppmega_base.py",
            'metadata.version(\\"flash-attn-4\\")',
        ),
    ):
        source = _read(path)
        assert version_check in source, path
        assert "flash_attn.__version__" not in source, path
        if path != "scripts/modal_build_tilelang_beta23.py":
            assert (
                "test -f /usr/local/lib/python3.13/site-packages/z3/lib/"
                "libz3.so.4.15"
            ) in source
            assert "ln -sf" not in source
    for path in ("docker/Dockerfile", "docker/Dockerfile.beta23"):
        dockerfile = _read(path)
        assert "apache-tvm-ffi==0.1.13" in dockerfile, path
        assert "metadata.version('apache-tvm-ffi') == '0.1.13.post5'" in dockerfile
        assert "metadata.version('flash-attn-4') == '4.0.0b23'" in dockerfile
        assert "metadata.version('nvidia-cutlass-dsl') == '4.6.0.dev0'" in dockerfile
        assert "metadata.version('quack-kernels') == '0.5.3'" in dockerfile
        assert '"flash-attn-4[cu13]==4.0.0b23"' in dockerfile, path
        assert '"quack-kernels==0.5.3"' in dockerfile, path
        assert (
            "not any(path.startswith('flash_attn/cute/') for path in fa2_files)"
            in dockerfile
        ), path
        assert "'flash_attn/cute/utils.py' in fa4_files" in dockerfile, path
        assert '"flash-attn-4[cu13]==4.0.0b19"' not in dockerfile, path


def test_te216_fa2_pin_stays_within_the_supported_range() -> None:
    commit = "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
    stack = _read("STACK.lock")
    workflow = _read(".github/workflows/build-wheels.yml")
    patch = _read("upstream_prs/flash_attn_setup_sm120f.patch")

    assert f"    ref: {commit}\n    patch: upstream_prs/flash_attn_setup_sm120f.patch" in stack
    assert "    version: 2.8.3" in stack
    assert f"            ref: {commit}" in workflow
    assert workflow.count(f"ref: {commit}") == 1
    assert '"flash_attn.cute",' in patch
    assert '"flash_attn.cute.*",' in patch
    assert "Verify FA2 wheel excludes FA4 namespace" in workflow
    assert 'name.startswith("flash_attn/cute/")' in workflow
    assert (
        'metadata.version(\\"flash-attn\\") == \\"2.8.3\\"'
        in _read("scripts/modal_cppmega_base.py")
    )
    for path in ("docker/Dockerfile", "docker/Dockerfile.beta23"):
        assert (
            "metadata.version('flash-attn') == '2.8.3'" in _read(path)
        ), path


def test_te216_and_mamba_runtime_metadata_are_complete() -> None:
    stack = _read("STACK.lock")
    workflow = _read(".github/workflows/build-wheels.yml")
    metadata_patch = _read("upstream_prs/mamba_setup_tilelang_019.patch")

    for path in ("docker/Dockerfile", "docker/Dockerfile.beta23"):
        dockerfile = _read(path)
        assert "importlib-metadata nvdlfw-inspect" in dockerfile, path
        assert "python -m pip check" in dockerfile, path
    assert "metadata_patch: upstream_prs/mamba_setup_tilelang_019.patch" in stack
    assert (
        "metadata_patch: upstream_prs/mamba_setup_tilelang_019.patch" in workflow
    )
    assert "diff --git a/pyproject.toml b/pyproject.toml" in metadata_patch
    assert "diff --git a/setup.py b/setup.py" in metadata_patch
    for indent in ("    ", "        "):
        assert f'-{indent}"tilelang==0.1.8",' in metadata_patch
        assert f'+{indent}"tilelang==0.1.9",' in metadata_patch
    assert 'required_tilelang = "Requires-Dist: tilelang==0.1.9"' in workflow


def test_fa4_h200_gate_exercises_visible_bias_and_is_not_mislabeled() -> None:
    source = _read("scripts/modal_fa4_beta23_parity.py")

    assert "call_edges[0, 0] = torch.tensor([1, 0])" in source
    assert "visible_bias_nonzero" in source
    assert "manual_bias_effect" in source
    assert "te_manual_max_diff" in source
    assert "fa4_manual_max_diff" in source
    assert "test_fa4_miniblock_training_step" in source
    assert "test_megatron_training_step" not in source


def test_fa4_h200_document_gate_uses_the_immutable_runtime_stack() -> None:
    source = _read("tests/test_fa4_h200_parity.py")

    assert "CPPMEGA_CANDIDATE_IMAGE_DIGEST" in source
    assert "CPPMEGA_CANDIDATE_CPPMEGA_SHA" in source
    assert "/opt/cppmega-image-source.json" in source
    assert 'metadata.distribution("flash-attn")' in source
    assert 'metadata.distribution("flash-attn-4")' in source
    assert "[sys.executable, \"-m\", \"pip\", \"check\"]" in source
    assert "fix_cutlass_namespace.py" not in source
    assert "setup_commands" not in source
    assert "_wheels_vol" not in source


def test_h200_images_include_dependency_free_bundle_restore_runtime() -> None:
    for path in ("docker/Dockerfile", "docker/Dockerfile.beta23"):
        dockerfile = _read(path)
        assert "        zstd \\" in dockerfile, path
        assert "awscli" not in dockerfile, path


def test_executable_gpu_envs_do_not_pin_torch_nightlies() -> None:
    paths = (
        "scripts/modal_cppmega_base.py",
        "scripts/modal_nan_sweep_h100.py",
        "scripts/modal_cutile_b200.py",
        "scripts/modal_cutile_b200_variant_sweep.py",
        "scripts/modal_cutile_mamba_mimo.py",
    )
    for path in paths:
        source = _read(path)
        assert "2.13.0+cu132" in source, path
        assert "https://download.pytorch.org/whl/cu132" in source, path
        assert "nightly/cu132" not in source, path
        assert '"torch==2.12.*"' not in source, path
        assert "pre=True" not in source, path


def test_current_docs_and_launchers_advertise_the_stable_release() -> None:
    readme = _read("README.md")
    assert "PyTorch 2.13.0 stable + cu132" in readme
    assert "PyTorch 2.12 nightly" not in readme

    for path in (
        "scripts/run_bench3_golden_fp8.sh",
        "scripts/run_europe_baseline_bf16.sh",
        "scripts/cppmega_fp8_shim.py",
    ):
        source = _read(path)
        assert "2.13.0+cu132" in source, path
        assert "2.12" not in source, path

    dsa_bench = _read("scripts/modal_dsa_indexer_bench.py")
    assert "TORCH_NIGHTLY_INDEX" not in dsa_bench
    assert "nightly/cu132" not in dsa_bench

    setup = _read("scripts/remote_setup_bench.sh")
    assert "EXPECTED_TORCH_VERSION" in setup
    assert "2.13.0+cu132" in setup


def test_gpu_reproducers_accept_torch_213_release() -> None:
    fp8 = _read("upstream_prs/examples/03_sparse_mla_fp8_dispatch/requirements.txt")
    flce = _read("upstream_prs/examples/10_megatron_flce_hopper/requirements.txt")

    assert "torch==2.13.0+cu132" in fp8
    assert "torch==2.13.0+cu132" in flce
