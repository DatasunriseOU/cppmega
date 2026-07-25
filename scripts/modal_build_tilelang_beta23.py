"""Modal: build TileLang wheel from DatasunriseOU/tilelang@5952468a and verify FA4 beta23 compat.

Builds the tilelang wheel (cp38-abi3, CUDA) on a Modal H100, then installs
flash-attn-4==4.0.0b23 + apache-tvm-ffi>=0.1.12 and verifies both import cleanly.

The fork at 5952468a carries apache/tvm#18938 (TVMDerivedObject.__slots__ fix,
via vendored TVM DatasunriseOU/tvm@9b0a1667) and removes the
apache-tvm-ffi<0.1.10 cap, so it imports cleanly under tvm-ffi >=0.1.12.

The resulting wheel is written to the cppmega-wheels Modal Volume and also
printed as base64 for local extraction.

Usage:
    modal run scripts/modal_build_tilelang_beta23.py

Output wheel: tilelang-0.1.9+cuda.git5952468a-cp38-abi3-linux_x86_64.whl
"""
from __future__ import annotations

import pathlib

import modal

TILELANG_REPO = "https://github.com/DatasunriseOU/tilelang.git"
TILELANG_COMMIT = "5952468a"
EXPECTED_WHEEL = "tilelang-0.1.9+cuda.git5952468a-cp38-abi3-linux_x86_64.whl"

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
            "scikit-build-core>=0.10", "cython", "z3-solver",
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
    """Clone tilelang@5952468a with submodules, build wheel, verify FA4 beta23."""
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

    # Verify the TVM submodule points to DatasunriseOU/tvm@9b0a1667 (apache/tvm#18938 fix)
    out = run("cd /tmp/tilelang && git submodule status 3rdparty/tvm")
    print(f"TVM submodule: {out.strip()}")
    assert "9b0a1667" in out, (
        f"Expected TVM submodule at 9b0a1667 (apache/tvm#18938 fix), got: {out}"
    )

    # --- 2. Install torch (needed for build) ---
    run(
        "pip install --extra-index-url https://download.pytorch.org/whl/cu132 "
        "'torch==2.13.0+cu132' --quiet"
    )

    # --- 3. Build the wheel ---
    run(
        "cd /tmp/tilelang && pip wheel . --no-build-isolation -w /wheels 2>&1 | tail -20"
    )

    # --- 4. Verify wheel was produced ---
    import glob
    wheels = glob.glob("/wheels/tilelang*.whl")
    print(f"Built wheels: {wheels}")
    assert len(wheels) >= 1, "No tilelang wheel produced!"
    wheel_path = wheels[0]
    wheel_name = pathlib.Path(wheel_path).name
    print(f"Wheel: {wheel_name}")

    # --- 5. Install the built wheel + FA4 beta23 + tvm-ffi ---
    run(f"pip install --no-deps {wheel_path}")
    run(
        "pip install --pre --quiet "
        "--extra-index-url https://pypi.nvidia.com "
        "'flash-attn-4[cu13]==4.0.0b23' "
        "'apache-tvm-ffi>=0.1.12,<0.2'"
    )

    # --- 6. Verify imports ---
    verify_code = """
import tilelang
import flash_attn
from flash_attn.cute.interface import flash_attn_func as fa4_flash_attn_func
import tvm.ffi
print(f"tilelang version: {tilelang.__version__}")
print(f"flash_attn version: {flash_attn.__version__}")
print(f"tvm.ffi version: {tvm.ffi.__version__}")
print("COMPAT OK: tilelang + flash-attn-4 beta23 + tvm-ffi all import cleanly")
"""
    run(f'python -c "{verify_code.strip()}"')

    # --- 7. Commit wheel to volume ---
    wheels_vol.commit()
    print(f"\nSUCCESS: {wheel_name} written to cppmega-wheels volume")
    print(f"Expected filename: {EXPECTED_WHEEL}")
    return wheel_name


@app.local_entrypoint()
def main():
    wheel_name = build_tilelang_wheel.remote()
    print(f"\nDone. Wheel: {wheel_name}")
    print(f"To download: modal volume get cppmega-wheels /{wheel_name} --output wheels/{wheel_name}")
