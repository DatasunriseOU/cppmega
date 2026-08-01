"""Build pinned TileLang on Modal and verify the FA4 beta23 compatibility lane.

Builds ABI-matched tvm-ffi and TileLang wheels on a Modal H100, then installs
flash-attn-4==4.0.0b23 and verifies the complete stack imports cleanly.

The fork at 3a495bb5 carries apache/tvm#18938 (TVMDerivedObject.__slots__ fix),
restores TVM's required nvbench CUDA header via DatasunriseOU/tvm@e25ca6ae,
removes the apache-tvm-ffi<0.1.10 cap, completes the CUDA driver stub, and
restores special-scope CUDA declarations for TIRx AllocBuffer nodes.

The clone and build directory are disposable container scratch. The compressed
wheel is written to the durable cppmega-wheels Modal Volume.

Usage:
    modal run scripts/modal_build_tilelang_beta23.py

Output wheels:
  - tilelang-0.1.9-cp38-abi3-linux_x86_64.whl
  - apache_tvm_ffi-0.1.13.post5-cp313-cp313-linux_x86_64.whl
"""
from __future__ import annotations

import pathlib

import modal

TILELANG_REPO = "https://github.com/DatasunriseOU/tilelang.git"
TILELANG_COMMIT = "3a495bb573bbdf3a263d728a50ace59239bc5159"
TILELANG_TVM_COMMIT = "e25ca6ae50beee0e907b1e5ed32949879caddde1"
TILELANG_TVM_FFI_COMMIT = "521efeb30bfd9e4946b248b3d76e6391028233a3"
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
def build_tilelang_wheel():
    """Build the exact TileLang/TVM pins and verify FA4 beta23 imports."""
    import shutil
    import subprocess
    import sys

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
    run(f"pip install {wheel_path}")
    run("pip check")

    # --- 6. Verify imports ---
    run(f"python - <<'PY'\n{VERIFY_CODE.strip()}\nPY")

    # --- 7. Commit wheel to volume ---
    wheels_vol.commit()
    print(f"\nSUCCESS: {wheel_name} written to cppmega-wheels volume")
    print(f"Expected filename: {EXPECTED_WHEEL}")
    return wheel_name


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
def main():
    wheel_name = build_tilelang_wheel.remote()
    print(f"\nDone. Wheel: {wheel_name}")
    print(
        "To download: modal volume get cppmega-wheels "
        f"/{wheel_name} --output wheels/{wheel_name}"
    )
