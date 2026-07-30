"""Build pinned TileLang on Modal and verify the FA4 beta23 compatibility lane.

Builds ABI-matched tvm-ffi and TileLang wheels on a Modal H100, then installs
flash-attn-4==4.0.0b23 and verifies the complete stack imports cleanly.

The fork at fff5cfcc carries apache/tvm#18938 (TVMDerivedObject.__slots__ fix),
restores TVM's required nvbench CUDA header via DatasunriseOU/tvm@ada9cffb,
removes the apache-tvm-ffi<0.1.10 cap, and completes the CUDA driver stub.

The clone and build directory are disposable container scratch. The compressed
wheel is written to the durable cppmega-wheels Modal Volume.

Usage:
    modal run scripts/modal_build_tilelang_beta23.py

Output wheel: tilelang-0.1.9-cp38-abi3-linux_x86_64.whl
"""
from __future__ import annotations

import pathlib

import modal

TILELANG_REPO = "https://github.com/DatasunriseOU/tilelang.git"
TILELANG_COMMIT = "fff5cfcc60fed16d163f13cca991256b6ebe1573"
TILELANG_TVM_COMMIT = "ada9cffbb381695651e265039f77c326c146d6b7"
TILELANG_TVM_FFI_COMMIT = "4e74cb45fbcf6117b69a9864bbe5548f1a7e17a2"
EXPECTED_WHEEL = "tilelang-0.1.9-cp38-abi3-linux_x86_64.whl"
EXPECTED_TVM_FFI_WHEEL = (
    "apache_tvm_ffi-0.1.13.post1-cp313-cp313-linux_x86_64.whl"
)

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
            "scikit-build-core>=0.10", "setuptools-scm", "cython", "z3-solver",
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
def build_tilelang_wheel():
    """Build the exact TileLang/TVM pins and verify FA4 beta23 imports."""
    import shutil
    import subprocess
    import sys

    def run(cmd: str, **kw):
        print(f"+ {cmd}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
        if r.stdout:
            print(r.stdout[-4000:])
        if r.returncode != 0:
            print(f"STDERR: {r.stderr[-4000:]}", file=sys.stderr)
            raise RuntimeError(f"Command failed ({r.returncode}): {cmd}")
        return r.stdout

    # --- 1. Clone tilelang at the pinned commit with submodules ---
    run(
        f"git clone --recurse-submodules --shallow-submodules "
        f"{TILELANG_REPO} /tmp/tilelang"
    )
    run(f"cd /tmp/tilelang && git checkout {TILELANG_COMMIT}")
    run("cd /tmp/tilelang && git submodule update --init --recursive")

    # Verify both the immutable TVM gitlink and the header whose absence broke
    # the H200 wheel build.
    out = run("cd /tmp/tilelang && git submodule status 3rdparty/tvm")
    print(f"TVM submodule: {out.strip()}")
    assert TILELANG_TVM_COMMIT in out, (
        f"Expected TVM submodule at {TILELANG_TVM_COMMIT}, got: {out}"
    )
    out = run(
        "cd /tmp/tilelang && "
        "git -C 3rdparty/tvm/3rdparty/tvm-ffi rev-parse HEAD"
    )
    assert out.strip() == TILELANG_TVM_FFI_COMMIT, out
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
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_APACHE_TVM_FFI=0.1.13.post1 "
        "pip wheel 3rdparty/tvm/3rdparty/tvm-ffi --no-build-isolation --no-deps "
        "-w /tmp/tilelang-wheel-out 2>&1 | tail -20"
    )
    run(
        "cd /tmp/tilelang && pip wheel . --no-build-isolation "
        "-w /tmp/tilelang-wheel-out 2>&1 | tail -20"
    )

    # --- 4. Verify wheel was produced ---
    wheel_path = f"/tmp/tilelang-wheel-out/{EXPECTED_WHEEL}"
    assert pathlib.Path(wheel_path).is_file(), (
        f"Expected exact wheel was not produced: {wheel_path}"
    )
    wheel_name = pathlib.Path(wheel_path).name
    print(f"Wheel: {wheel_name}")
    shutil.copy2(wheel_path, f"/wheels/{EXPECTED_WHEEL}")
    ffi_wheel_path = f"/tmp/tilelang-wheel-out/{EXPECTED_TVM_FFI_WHEEL}"
    assert pathlib.Path(ffi_wheel_path).is_file(), ffi_wheel_path
    shutil.copy2(ffi_wheel_path, f"/wheels/{EXPECTED_TVM_FFI_WHEEL}")

    # --- 5. Install FA4 beta23, then the exact linked FFI + TileLang wheels ---
    run(
        "pip install --pre --quiet "
        "--extra-index-url https://pypi.nvidia.com "
        "'flash-attn-4[cu13]==4.0.0b23'"
    )
    run(f"pip install --force-reinstall --no-deps {ffi_wheel_path} {wheel_path}")

    # --- 6. Verify imports ---
    verify_code = """
from importlib import metadata
import tilelang
import flash_attn
from flash_attn.cute.interface import flash_attn_func as fa4_flash_attn_func
import tvm.ffi
assert tilelang.__version__ == "0.1.9", tilelang.__version__
assert metadata.version("apache-tvm-ffi") == "0.1.13.post1"
print(f"tilelang version: {tilelang.__version__}")
print(f"flash_attn version: {flash_attn.__version__}")
print(f"tvm.ffi version: {tvm.ffi.__version__}")
print("COMPAT OK: tilelang + flash-attn-4 beta23 + tvm-ffi all import cleanly")
"""
    run(f"python - <<'PY'\n{verify_code.strip()}\nPY")

    # --- 7. Commit wheel to volume ---
    wheels_vol.commit()
    print(f"\nSUCCESS: {wheel_name} written to cppmega-wheels volume")
    print(f"Expected filename: {EXPECTED_WHEEL}")
    return wheel_name


@app.local_entrypoint()
def main():
    wheel_name = build_tilelang_wheel.remote()
    print(f"\nDone. Wheel: {wheel_name}")
    print(
        "To download: modal volume get cppmega-wheels "
        f"/{wheel_name} --output wheels/{wheel_name}"
    )
