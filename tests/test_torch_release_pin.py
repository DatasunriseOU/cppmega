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

    assert 'package: "apache-tvm-ffi==0.1.13"' in stack
    assert 'package: "flash-attn-4[cu13]==4.0.0b23"' in stack
    for path in ("docker/Dockerfile", "docker/Dockerfile.beta23"):
        dockerfile = _read(path)
        assert "apache-tvm-ffi==0.1.13" in dockerfile, path
        assert '"flash-attn-4[cu13]==4.0.0b23"' in dockerfile, path
        assert '"flash-attn-4[cu13]==4.0.0b19"' not in dockerfile, path


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
