"""Modal H200/H100: run the full cppmega pytest suite on CUDA.

Usage:
    CPPMEGA_MODAL_GPU=H200:1 modal run scripts/modal_cppmega_run_tests.py

Overlays the local repo (including tests/) onto the GHCR runtime image and
runs pytest with the real Megatron source root at /opt/megatron-lm.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Any

import modal

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "latest")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:1")
MEGATRON_ROOT = "/opt/megatron-lm"
PYTEST_TARGET = os.environ.get("CPPMEGA_MODAL_PYTEST_TARGET", "tests/")
PYTEST_ARGS = os.environ.get(
    "CPPMEGA_MODAL_PYTEST_ARGS",
    "-q --tb=short -n 4 --ignore=tests/test_mamba3_wgmma_wave10_copy_integration.py "
    "--ignore=tests/test_mamba3_wgmma_wave4_schedule.py "
    "--ignore=tests/test_mamba3_wgmma_wave6_copy_path.py "
    "--ignore=tests/test_mamba3_wgmma_wave7_copy_evidence.py "
    "--ignore=tests/test_mamba3_wgmma_wave8_copy_evidence.py "
    "--ignore=tests/test_mamba3_wgmma_wave9_copy_probe.py",
)

_MLX_ROOT = _REPO_ROOT.parent / "cppmega.mlx"
_MLX_REMOTE_ROOT = "/opt/cppmega-mlx"
_PYTHON_CACHE_IGNORE = ("**/__pycache__/**", "**/*.pyc", "**/*.pyo")
_MLX_OVERLAY_IGNORE = (
    *_PYTHON_CACHE_IGNORE,
    "training/native_optim/_build/**",
    "training/native_optim/*.so",
    "training/native_optim/*.dylib",
    "training/native_optim/*.metallib",
)


def _local_mlx_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_MLX_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - the optional sibling checkout may be absent
        return ""


LOCAL_MLX_SHA = _local_mlx_sha()


def _local_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - importing this launcher must survive missing git
        return "unknown"


LOCAL_GIT_SHA = _local_git_sha()


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    ).env(
        {
            "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
            "WANDB_MODE": "disabled",
            "MEGATRON_LM_REPO": MEGATRON_ROOT,
            # Remote module re-import sees no local env; carry the locally
            # selected pytest args so the container runs the same selection.
            "CPPMEGA_MODAL_PYTEST_ARGS": PYTEST_ARGS,
            "CPPMEGA_MODAL_PYTEST_TARGET": PYTEST_TARGET,
            "CPPMEGA_GHCR_REF": GHCR_REF,
            "CPPMEGA_SOURCE_COMMIT": LOCAL_GIT_SHA,
            "CPPMEGA_MEGATRON_COMMIT": os.environ.get(
                "CPPMEGA_MEGATRON_COMMIT", "980211ae"
            ),
        }
    )
    if _MLX_ROOT.is_dir() and LOCAL_MLX_SHA:
        img = img.env(
            {
                "CPPMEGA_MLX_REFERENCE_ROOT": _MLX_REMOTE_ROOT,
                "CPPMEGA_MLX_REFERENCE_COMMIT": LOCAL_MLX_SHA,
            }
        ).add_local_dir(
            str(_MLX_ROOT / "cppmega_mlx"),
            remote_path=f"{_MLX_REMOTE_ROOT}/cppmega_mlx",
            copy=True,
            ignore=_MLX_OVERLAY_IGNORE,
        )
    img = img.pip_install("pytest", "pytest-xdist", "hypothesis", "pyarrow", "mlx[cuda]")
    img = (
        img.add_local_dir(
            str(_REPO_ROOT / "cppmega"),
            remote_path="/opt/cppmega/cppmega",
            copy=True,
            ignore=_PYTHON_CACHE_IGNORE,
        )
        .add_local_dir(
            str(_REPO_ROOT / "tests"),
            remote_path="/opt/cppmega/tests",
            copy=True,
            ignore=_PYTHON_CACHE_IGNORE,
        )
        .add_local_dir(
            str(_REPO_ROOT / "scripts"),
            remote_path="/opt/cppmega/scripts",
            copy=True,
            ignore=_PYTHON_CACHE_IGNORE,
        )
        .add_local_dir(
            str(_REPO_ROOT / "tools"),
            remote_path="/opt/cppmega/tools",
            copy=True,
            ignore=_PYTHON_CACHE_IGNORE,
        )
        .add_local_dir(
            str(_REPO_ROOT / "configs"),
            remote_path="/opt/cppmega/configs",
            copy=True,
            ignore=_PYTHON_CACHE_IGNORE,
        )
        .add_local_dir(
            str(_REPO_ROOT / "data"),
            remote_path="/opt/cppmega/data",
            copy=True,
            ignore=_PYTHON_CACHE_IGNORE,
        )
        .add_local_dir(
            str(_REPO_ROOT / "upstream_prs"),
            remote_path="/opt/cppmega/upstream_prs",
            copy=True,
            ignore=_PYTHON_CACHE_IGNORE,
        )
        .add_local_file(str(_REPO_ROOT / "conftest.py"), remote_path="/opt/cppmega/conftest.py")
        .add_local_file(str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml")
    )
    return img


app = modal.App("cppmega-run-tests")
results_vol = modal.Volume.from_name("cppmega-test-results", create_if_missing=True)
RESULTS_PATH = "/results/latest.json"


@app.function(image=_image(), gpu=GPU_SPEC, timeout=3600, volumes={"/results": results_vol})
def run_tests() -> dict[str, Any]:
    import json as _json
    import subprocess as sp
    import sys

    megatron_head = sp.run(
        ["git", "-C", MEGATRON_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()

    env = os.environ.copy()
    env["MEGATRON_LM_REPO"] = MEGATRON_ROOT
    env["CPPMEGA_MEGATRON_COMMIT"] = megatron_head or env.get("CPPMEGA_MEGATRON_COMMIT", "")
    env["CUDA_VISIBLE_DEVICES"] = "0"

    sp.run(
        [sys.executable, "-m", "pip", "install", "-e", "/opt/cppmega", "--no-deps", "-q"],
        env=env, capture_output=True, check=False,
    )

    cmd = [sys.executable, "-m", "pytest", PYTEST_TARGET] + PYTEST_ARGS.split()
    proc = sp.run(
        cmd,
        cwd="/opt/cppmega",
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=3300,
    )
    result = {
        "returncode": proc.returncode,
        "megatron_head": megatron_head,
        "gpu": GPU_SPEC,
        "pytest_args": PYTEST_ARGS,
        "pytest_target": PYTEST_TARGET,
        "ghcr_ref": os.environ.get("CPPMEGA_GHCR_REF", GHCR_REF),
        "source_commit": os.environ.get("CPPMEGA_SOURCE_COMMIT", LOCAL_GIT_SHA),
        "mlx_reference_commit": os.environ.get("CPPMEGA_MLX_REFERENCE_COMMIT", ""),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-60:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-30:]),
    }
    pathlib.Path("/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(RESULTS_PATH).write_text(_json.dumps(result, indent=2))
    pathlib.Path("/results/stdout_full.txt").write_text(proc.stdout)
    pathlib.Path("/results/stderr_full.txt").write_text(proc.stderr)
    results_vol.commit()
    return result


@app.function(image=_image(), timeout=60, volumes={"/results": results_vol})
def read_results() -> str:
    p = pathlib.Path(RESULTS_PATH)
    if not p.exists():
        return '{"error": "no results yet"}'
    return p.read_text()


@app.local_entrypoint()
def main() -> None:
    import json

    result = run_tests.remote()
    print(json.dumps(result, indent=2))
