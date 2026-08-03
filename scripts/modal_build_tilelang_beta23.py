"""Build pinned TileLang on Modal and verify the FA4 beta23 compatibility lane.

Builds ABI-matched tvm-ffi and TileLang wheels on a Modal H100, then installs
flash-attn-4==4.0.0b23 and verifies the complete stack imports cleanly.

The fork at a760fe58 retains the FA4 compatibility fixes, adds stable
producer/consumer layout inference and cross-rank JIT-cache serialization,
and vendors DatasunriseOU/tvm@84af1727 plus tvm-ffi@e4353339 so C++ exception
RTTI and Python thread state survive FFI exception boundaries.

The clone and build directory are disposable container scratch. The compressed
wheel is written to the durable cppmega-wheels Modal Volume.

Usage:
    modal run scripts/modal_build_tilelang_beta23.py
    modal run scripts/modal_build_tilelang_beta23.py \
      --tilelang-commit <full-sha> --tilelang-branch <branch> \
      --tvm-commit <full-sha> --tvm-ffi-commit <full-sha>

Output wheels:
  - tilelang-0.1.9-cp38-abi3-linux_x86_64.whl
  - apache_tvm_ffi-0.1.13.post5-cp313-cp313-linux_x86_64.whl
"""
from __future__ import annotations

import pathlib

import modal

TILELANG_REPO = "https://github.com/DatasunriseOU/tilelang.git"
TILELANG_BRANCH = "fix/reducer-consumer-layout"
TILELANG_COMMIT = "a760fe587995def0f3108ee204be453d87467c5d"
TILELANG_TVM_COMMIT = "84af17279edb5edad29749bd6b0eea2ed9393105"
TILELANG_TVM_FFI_COMMIT = "e4353339293459e3e8a393afc1b6a6a869e75b13"
EXPECTED_WHEEL = "tilelang-0.1.9-cp38-abi3-linux_x86_64.whl"
EXPECTED_TVM_FFI_WHEEL = (
    "apache_tvm_ffi-0.1.13.post5-cp313-cp313-linux_x86_64.whl"
)
VERIFY_CODE = """
from importlib import metadata
import tilelang
import flash_attn
from flash_attn.cute.interface import flash_attn_func as fa4_flash_attn_func
import tvm.ffi
from tvm.s_tir.analysis import is_pure_function
assert tilelang.__version__ == "0.1.9", tilelang.__version__
assert metadata.version("apache-tvm-ffi") == "0.1.13.post5"
assert metadata.version("flash-attn-4") == "4.0.0b23"
body = tvm.tirx.AttrStmt(
    None, "threadblock_swizzle_pattern", 0, tvm.tirx.Evaluate(0)
)
assert body.node is None
assert is_pure_function(tvm.tirx.PrimFunc([], body))
print(f"tilelang version: {tilelang.__version__}")
print(f"flash-attn-4 version: {metadata.version('flash-attn-4')}")
print(f"tvm.ffi version: {tvm.ffi.__version__}")
print("COMPAT OK: tilelang + flash-attn-4 beta23 + tvm-ffi all import cleanly")
"""

PYTHON_VERSION = "3.13"
CUDA_BASE = "nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04"

app = modal.App("cppmega-build-tilelang-beta23")
wheels_vol = modal.Volume.from_name("cppmega-wheels", create_if_missing=True)


def _build_image() -> modal.Image:
    """Image with build toolchain for compiling tilelang from source."""
    return (
        modal.Image.from_registry(CUDA_BASE, add_python=PYTHON_VERSION)
        .apt_install(
            "git", "build-essential", "gcc", "g++", "cmake", "ninja-build",
            "curl", "ca-certificates", "pkg-config", "libnuma-dev",
        )
        .pip_install(
            "pip>=24.0", "setuptools", "wheel",
            "scikit-build-core>=0.10", "setuptools-scm", "cython",
            "z3-solver==4.15.4.0",
            "numpy", "packaging", "pybind11",
        )
        .env({
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/sbin",
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "MAX_JOBS": "8",
        })
    )


@app.function(
    image=_build_image(),
    gpu="H100:1",
    timeout=3600,
    volumes={"/wheels": wheels_vol},
)
def build_tilelang_wheel(
    tilelang_commit: str = TILELANG_COMMIT,
    tilelang_branch: str = TILELANG_BRANCH,
    tvm_commit: str = TILELANG_TVM_COMMIT,
    tvm_ffi_commit: str = TILELANG_TVM_FFI_COMMIT,
):
    """Build the exact TileLang/TVM pins and verify FA4 beta23 imports."""
    import hashlib
    import json
    import re
    import shutil
    import subprocess
    import sys

    for name, commit in (
        ("tilelang_commit", tilelang_commit),
        ("tvm_commit", tvm_commit),
        ("tvm_ffi_commit", tvm_ffi_commit),
    ):
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"{name} must be a full lowercase git SHA")
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", tilelang_branch)
        or ".." in tilelang_branch
        or "//" in tilelang_branch
    ):
        raise ValueError("tilelang_branch is not a safe git branch name")
    volume_subdir = (
        ""
        if tilelang_commit == TILELANG_COMMIT
        else f"candidates/{tilelang_commit}/linux-cuda13.2-cp313"
    )
    volume_dir = pathlib.Path("/wheels") / volume_subdir
    if volume_subdir and volume_dir.exists():
        raise RuntimeError(
            f"refusing to overwrite existing candidate directory: {volume_dir}"
        )

    def run(cmd: str, **kw):
        print(f"+ {cmd}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
        if r.returncode != 0:
            if r.stdout:
                print(f"STDOUT tail:\n{r.stdout[-24_000:]}", file=sys.stderr)
            if r.stderr:
                print(f"STDERR tail:\n{r.stderr[-24_000:]}", file=sys.stderr)
            raise RuntimeError(f"Command failed ({r.returncode}): {cmd}")
        if r.stdout:
            print(r.stdout[-4000:])
        return r.stdout

    # --- 1. Clone tilelang at the pinned commit with submodules ---
    run(
        f"git clone --recurse-submodules --shallow-submodules "
        f"{TILELANG_REPO} /tmp/tilelang"
    )
    run(
        "cd /tmp/tilelang && "
        f"git fetch origin refs/heads/{tilelang_branch} && "
        f"git checkout --detach {tilelang_commit}"
    )
    run("cd /tmp/tilelang && git submodule update --init --recursive")
    observed_commit = run("git -C /tmp/tilelang rev-parse HEAD").strip()
    assert observed_commit == tilelang_commit, observed_commit

    # Verify both the immutable TVM gitlink and the header whose absence broke
    # the H200 wheel build.
    out = run("cd /tmp/tilelang && git submodule status 3rdparty/tvm")
    print(f"TVM submodule: {out.strip()}")
    assert tvm_commit in out, (
        f"Expected TVM submodule at {tvm_commit}, got: {out}"
    )
    out = run(
        "cd /tmp/tilelang && "
        "git -C 3rdparty/tvm/3rdparty/tvm-ffi rev-parse HEAD"
    )
    assert out.strip() == tvm_ffi_commit, out
    run("test -f /tmp/tilelang/3rdparty/tvm/3rdparty/nvbench/l2_cache_flush.h")

    # --- 2. Install torch (needed for build) ---
    run(
        "pip install --extra-index-url https://download.pytorch.org/whl/cu132 "
        "'torch==2.13.0+cu132' --quiet"
    )

    # --- 3. Build the ABI-matched wheels ---
    run("mkdir -p /tmp/tilelang-wheel-out")
    run(
        "cd /tmp/tilelang && "
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_APACHE_TVM_FFI=0.1.13.post5 "
        "pip wheel 3rdparty/tvm/3rdparty/tvm-ffi --no-build-isolation --no-deps "
        "-w /tmp/tilelang-wheel-out"
    )
    run(
        "cd /tmp/tilelang && pip wheel . --no-build-isolation --no-deps "
        "-w /tmp/tilelang-wheel-out"
    )

    # --- 4. Verify wheel was produced ---
    wheel_path = f"/tmp/tilelang-wheel-out/{EXPECTED_WHEEL}"
    assert pathlib.Path(wheel_path).is_file(), (
        f"Expected exact wheel was not produced: {wheel_path}"
    )
    wheel_name = pathlib.Path(wheel_path).name
    print(f"Wheel: {wheel_name}")
    ffi_wheel_path = f"/tmp/tilelang-wheel-out/{EXPECTED_TVM_FFI_WHEEL}"
    assert pathlib.Path(ffi_wheel_path).is_file(), ffi_wheel_path

    # --- 5. Install FA4 beta23, then the exact linked FFI + TileLang wheels ---
    run(
        "pip install --pre --quiet "
        "--extra-index-url https://pypi.nvidia.com "
        "'flash-attn-4[cu13]==4.0.0b23'"
    )
    run(f"pip install --force-reinstall --no-deps {ffi_wheel_path} {wheel_path}")
    run(f"pip install {wheel_path}")
    run("pip check")

    # --- 6. Verify imports ---
    run(f"python - <<'PY'\n{VERIFY_CODE.strip()}\nPY")

    # --- 7. Publish only the verified wheel pair ---
    volume_dir.mkdir(parents=True, exist_ok=True)
    wheel_target = volume_dir / EXPECTED_WHEEL
    ffi_target = volume_dir / EXPECTED_TVM_FFI_WHEEL
    shutil.copy2(wheel_path, wheel_target)
    shutil.copy2(ffi_wheel_path, ffi_target)
    if volume_subdir:
        artifacts = {}
        for path in (wheel_target, ffi_target):
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            artifacts[path.name] = {
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        (volume_dir / "BUILD_MANIFEST.json").write_text(
            json.dumps(
                {
                    "schema": "cppmega_tilelang_candidate_build_v1",
                    "status": "success",
                    "tilelang_branch": tilelang_branch,
                    "tilelang_commit": tilelang_commit,
                    "tvm_commit": tvm_commit,
                    "tvm_ffi_commit": tvm_ffi_commit,
                    "artifacts": artifacts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    wheels_vol.commit()
    volume_path = str(pathlib.PurePosixPath(volume_subdir) / wheel_name)
    print(f"\nSUCCESS: {volume_path} written to cppmega-wheels volume")
    return volume_path


@app.function(
    image=_build_image(),
    gpu="H100:1",
    timeout=1800,
    volumes={"/wheels": wheels_vol},
)
def verify_existing_wheels():
    """Verify the committed exact wheels without rebuilding native code."""
    import subprocess
    import sys

    wheel_path = f"/wheels/{EXPECTED_WHEEL}"
    ffi_wheel_path = f"/wheels/{EXPECTED_TVM_FFI_WHEEL}"
    commands = [
        [
            sys.executable, "-m", "pip", "install", "--quiet",
            "--extra-index-url", "https://download.pytorch.org/whl/cu132",
            "torch==2.13.0+cu132",
        ],
        [
            sys.executable, "-m", "pip", "install", "--pre", "--quiet",
            "--extra-index-url", "https://pypi.nvidia.com",
            "flash-attn-4[cu13]==4.0.0b23",
        ],
        [
            sys.executable, "-m", "pip", "install", "--force-reinstall",
            "--no-deps", ffi_wheel_path, wheel_path,
        ],
        [sys.executable, "-m", "pip", "install", wheel_path],
        [sys.executable, "-m", "pip", "check"],
        [sys.executable, "-c", VERIFY_CODE],
    ]
    for index, command in enumerate(commands, start=1):
        print(f"STAGE {index}/{len(commands)}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
    return EXPECTED_WHEEL


@app.local_entrypoint()
def main(
    tilelang_commit: str = TILELANG_COMMIT,
    tilelang_branch: str = TILELANG_BRANCH,
    tvm_commit: str = TILELANG_TVM_COMMIT,
    tvm_ffi_commit: str = TILELANG_TVM_FFI_COMMIT,
):
    wheel_path = build_tilelang_wheel.remote(
        tilelang_commit, tilelang_branch, tvm_commit, tvm_ffi_commit
    )
    print(f"\nDone. Wheel: {wheel_path}")
    print(
        "To download: modal volume get cppmega-wheels "
        f"/{wheel_path} --output wheels/{pathlib.Path(wheel_path).name}"
    )
