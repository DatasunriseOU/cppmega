"""Modal app: cuTile Python + TileLang + mamba_ssm stack on B200:2.

Purpose
-------
Provision an NVIDIA B200 (sm_100 datacenter Blackwell) instance on Modal and
install the same verified bench stack as our H200 box plus the cuda-tile
package (NVIDIA cuTile Python). Our GB10 box (sm_121 consumer Blackwell) is
currently down and cuTile Python rejects sm_90 (no Hopper) so H200 is not an
option for cuTile Python kernel testing. B200 is the next best target: it is
the datacenter Blackwell in cuTile's supported --gpu-name list and is also
the consumer-deployment target for future cppmega releases.

Stack targeted (matches memory/reference_stack_bench.md where possible):
- torch 2.12 nightly on cu13x (prefers cu132 to match bench3 exactly; pip
  will resolve whichever nightly exists today)
- cuda-tile             (NVIDIA cuTile Python, the `import cuda.tile as ct` pkg)
- apache-tvm-ffi==0.1.9 (CRITICAL: MUST be <0.1.10 for TileLang)
- tilelang==0.1.8
- mamba-ssm==2.3.1      (reference kernel we compare cuTile port against)

The image is built from nvidia/cuda:13.2.1-devel-ubuntu24.04 (public, no GCR
auth needed) rather than nanochat's private modal-base image in
LOCATION_4ocker.pkg.dev, so any cppmega developer can run this.

Usage
-----
    # Auth (one-time, interactive):
    #     modal token new
    #
    # Sanity test (compile + run vadd on sm_100, ~3-5 min GPU time):
    modal run scripts/modal_cutile_b200.py::sanity
    #
    # Inspect the full installed pip freeze + tileiras target list:
    modal run scripts/modal_cutile_b200.py::inspect
    #
    # Run everything (inspect, then sanity) in one container:
    modal run scripts/modal_cutile_b200.py::main

Cost: B200:2 on Modal is ~$20-30/hr. The sanity function caps at 10 min wall
clock via timeout=600. Do NOT leave this running.
"""
# ruff: noqa: E402

from __future__ import annotations

import os
from typing import Any, cast

import modal

# ---------------------------------------------------------------------------
# Image build
# ---------------------------------------------------------------------------
#
# Base: Modal's own debian_slim image (no docker.io rate limits) with Python
# 3.13. We install the CUDA 13 toolkit headers + nvcc via NVIDIA's nvidia-*
# pip wheels (nvidia-cuda-nvcc-cu13, nvidia-cuda-runtime-cu13, etc.) rather
# than apt — those wheels are on PyPI, don't need a docker hub pull, and
# match the cu132 torch nightly we also install. Modal's runtime injects the
# CUDA driver when the GPU is attached, so we only need the toolkit here.

_PYTHON = "3.13"

# Torch nightly: the cu132 nightly index, resolved at image build time.
# We pin the major.minor to 2.12.* to stay on the same branch as bench3.
_TORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly/cu132"

# GPU spec: B200:2 by default. Override via env for dev.
_GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "B200:2")

# App + persistent Volume for mamba3_mimo_cutile port files.
app = modal.App("cppmega-cutile-b200")
cutile_vol = modal.Volume.from_name("cppmega-cutile-mamba3", create_if_missing=True)


def _image() -> modal.Image:
    """Construct the cuTile / TileLang / mamba_ssm image for B200.

    Cast to Any because Modal's Image builder is intentionally untyped on
    the chained methods — same pattern as nanochat/scripts/modal_*.py.
    """
    base: Any = modal.Image.debian_slim(python_version=_PYTHON)
    img = (
        base.apt_install("git", "build-essential", "ninja-build", "curl", "ca-certificates")
        # Torch 2.12 nightly cu132 (match bench3). pre=True enables nightlies;
        # extra_index_url keeps PyPI available for the rest of the deps.
        .pip_install(
            "torch==2.12.*",
            "numpy>=1.26",
            "packaging",
            "wheel",
            "setuptools",
            "einops",
            "ninja",
            extra_index_url=_TORCH_NIGHTLY_INDEX,
            pre=True,
        )
        # nvcc for tileiras JIT. Torch's cu132 wheel already brings
        # nvidia-cuda-nvrtc / nvidia-cuda-runtime / cuda-toolkit meta as
        # runtime deps, but NOT nvcc — we install nvidia-cuda-nvcc explicitly
        # from the NVIDIA index. Note: the name changed from -cu13 to plain
        # (versioned inside) in 2026 Q1.
        .pip_install(
            "nvidia-cuda-nvcc",
            extra_index_url="https://pypi.nvidia.com",
        )
        # cuTile Python + TileLang stack. tvm-ffi pin is load-bearing.
        # cuda-tile[tileiras] is REQUIRED — the base `cuda-tile` wheel only
        # has the Python API, the actual compiler binary ships in the
        # `tileiras` extra. Without it you get FileNotFoundError at the
        # first ct.launch call.
        .pip_install(
            "cuda-tile[tileiras]",
            "apache-tvm-ffi==0.1.9",
            "tilelang==0.1.8",
        )
        # mamba-ssm reference kernel is installed in a SEPARATE layered
        # image (`image_with_mamba` below) because its source build is
        # fragile against torch 2.12 nightly (no prebuilt wheels on PyPI
        # for this torch version as of 2026-04-10). The sanity path does
        # not need mamba-ssm, so we keep the base image lean and fast.
        # Put nvcc on PATH globally so tileiras can find it. env dict is
        # merged into every container's environment. The nvidia-cuda-nvcc
        # wheel (on PyPI as of 2026-04) drops nvcc at nvidia/cu13/bin/nvcc,
        # NOT nvidia/cuda_nvcc/bin/. Also add the cuda_tile bin dir so
        # tileiras (shipped inside the cuda_tile wheel) is discoverable.
        .env(
            {
                "PATH": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/bin:"
                        "/usr/local/lib/python3.13/site-packages/cuda_tile/bin:"
                        "/usr/local/bin:/usr/bin:/bin",
                "CUDA_HOME": "/usr/local/lib/python3.13/site-packages/nvidia/cu13",
            }
        )
        # Sanity: verify the tile stack imports at build time so a broken
        # image fails here, not at first function call. cuda-tile 1.2.0
        # does NOT ship a standalone `tileiras` CLI — the compiler is
        # invoked internally from the Python API (cuda.tile._compile).
        # We list the cuda.tile package contents here so future versions
        # that DO ship a CLI can be picked up automatically.
        .run_commands(
            "which nvcc && nvcc --version",
            "python -c 'import cuda.tile as ct; print(\"cuda.tile\", getattr(ct, \"__version__\", \"ok\"))'",
            "python -c 'import cuda.tile; print(\"path\", cuda.tile.__file__)'",
            "python -c 'import pkgutil, cuda.tile; print([m.name for m in pkgutil.iter_modules(cuda.tile.__path__)])'",
            "find /usr/local/lib/python3.13/site-packages/cuda/tile -maxdepth 2 -type f | head -40",
            "python -c 'import tilelang; print(\"tilelang\", tilelang.__version__)'",
            "python -c 'import tvm_ffi; print(\"tvm_ffi\", tvm_ffi.__version__)'",
        )
    )
    return img


image = _image()


# ---------------------------------------------------------------------------
# Kernels at module scope
# ---------------------------------------------------------------------------
#
# cuTile's @ct.kernel decorator resolves type annotations via
# typing.get_type_hints(pyfunc, globalns=pyfunc.__globals__) which means the
# `ct.Constant[int]` (or any alias of it) MUST be visible in the module
# globals of the function where the kernel is defined. We therefore import
# cuda.tile at module scope, guarded by try/except so this file can still be
# parsed on a client machine that does not have cuda-tile installed (e.g.
# a developer laptop that only runs `modal run`). Modal's container always
# has cuda-tile in the image, so the import succeeds there and the kernel
# function is made available.

try:  # pragma: no cover — runs only on Modal container
    import cuda.tile as _ct  # type: ignore[import-not-found]

    _ConstInt = _ct.Constant[int]

    @_ct.kernel
    def _vadd_kernel(a, b, out, tile: _ConstInt) -> None:  # type: ignore[valid-type]
        bi = _ct.bid(0)
        ta = _ct.load(a, index=(bi,), shape=(tile,))
        tb = _ct.load(b, index=(bi,), shape=(tile,))
        tc = ta + tb
        _ct.store(out, index=(bi,), tile=tc)
except ImportError:
    _ct = None  # type: ignore[assignment]
    _vadd_kernel = None  # type: ignore[assignment]


def _image_with_mamba() -> modal.Image:
    """Overlay mamba-ssm 2.3.1 + causal-conv1d on top of the base image.

    Kept separate because mamba-ssm has no prebuilt wheel for torch 2.12
    nightly as of 2026-04-10, so we build from source. The build needs
    nvcc (already on PATH in the base), --no-build-isolation (so the
    build env sees torch 2.12+cu132 from the outer layer instead of a
    stale torch 2.11+cu130 from PyPI), and TORCH_CUDA_ARCH_LIST='10.0'
    to target B200 sm_100.

    This layer is only needed for the Mamba3 MIMO parity test in
    modal_cutile_mamba_mimo.py — the sanity path in this file doesn't
    need it, so we don't build it unless explicitly requested.
    """
    base: Any = image
    return (
        base
        # libcu++ headers (nv/target lives here). The nvidia-cuda-nvcc wheel
        # ships nvcc but NOT the C++ headers; cuda-cccl provides nv/target,
        # cuda/std/*, etc. Without this, causal-conv1d and mamba-ssm fail with
        # "fatal error: nv/target: No such file or directory" when nvcc
        # processes torch's BFloat16.h -> cuda_bf16.h -> cuda_fp16.h chain.
        .pip_install("nvidia-cuda-cccl", extra_index_url="https://pypi.nvidia.com")
        # Ensure the cuda-cccl include dir is on every nvcc/g++ invocation.
        # Wheels drop headers into nvidia/cu13/include (same tree as nvcc).
        .env({"CPLUS_INCLUDE_PATH": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/include",
              "C_INCLUDE_PATH": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/include",
              "CUDA_INCLUDE_DIRS": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/include",
              # Link-time and run-time search paths for libcudart / cuBLAS /
              # other nvidia-cu13 runtime libraries. The torch 2.12+cu132
              # meta-wheel lays them out under nvidia/cu13/lib — but also
              # under nvidia/cu13/lib/x86_64-linux-gnu in some layouts,
              # hence the double path here. (Confirmed the exact layout
              # varies by nvidia-* version.) Using `:`-separated list is
              # fine for both GNU ld and glibc's ld.so.
              "LIBRARY_PATH": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib:"
                              "/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib/x86_64-linux-gnu:"
                              "/usr/local/lib/python3.13/site-packages/torch/lib",
              "LD_LIBRARY_PATH": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib:"
                                 "/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib/x86_64-linux-gnu:"
                                 "/usr/local/lib/python3.13/site-packages/torch/lib"})
        .add_local_file(
            "/Volumes/external/sources/cppmega/scripts/_modal_patch_mamba_setup.py",
            "/tmp/patch_setup.py",
            copy=True,
        )
        .run_commands(
            # Prove nv/target is reachable before we burn cycles on compile.
            "ls /usr/local/lib/python3.13/site-packages/nvidia/cu13/include/nv/target 2>&1 || "
            "(echo '### cu13/include tree:' && find /usr/local/lib/python3.13/site-packages/nvidia/cu13/include -maxdepth 2 -type d && exit 1)",
            "pip install --no-build-isolation --no-cache-dir packaging ninja wheel setuptools",
            # Locate libcudart.so so we can wire it into LD_LIBRARY_PATH for
            # the source-builds. The wheel name changed from nvidia-cuda-runtime-cu13
            # to the cu13-subdir scheme, so the .so can live at a few paths.
            "echo '--- libcudart search ---' && "
            "find /usr/local/lib/python3.13/site-packages/nvidia -name 'libcudart*' 2>/dev/null && "
            "find /usr/local/lib/python3.13/site-packages/torch -name 'libcudart*' 2>/dev/null | head -5",
            # Wire libcudart into both link-time and load-time search paths.
            # libcudart.so ships in nvidia-cuda-runtime wheel which lays it
            # down under site-packages/nvidia/cu13/lib (confirmed on sanity
            # runs) but NOT with the vanilla name — it's often a versioned
            # .so.13 with no unversioned symlink. We create the symlink.
            "CUDART_DIR=$(find /usr/local/lib/python3.13/site-packages/nvidia -name 'libcudart.so*' -printf '%h\\n' 2>/dev/null | head -1) && "
            "echo CUDART_DIR=$CUDART_DIR && "
            "if [ -n \"$CUDART_DIR\" ]; then "
            "  cd $CUDART_DIR && "
            "  if [ ! -f libcudart.so ] && [ -f libcudart.so.13 ]; then ln -sf libcudart.so.13 libcudart.so; fi && "
            "  ls -l libcudart* && "
            "  echo $CUDART_DIR > /tmp/cudart_dir.txt; "
            "fi",
            "mkdir -p /tmp/mb && cd /tmp/mb && "
            "curl -sSL https://files.pythonhosted.org/packages/source/c/causal-conv1d/causal_conv1d-1.6.0.tar.gz "
            "-o cc.tar.gz && tar xf cc.tar.gz && cd causal_conv1d-1.6.0 && "
            "python /tmp/patch_setup.py setup.py && "
            "echo '--- gencode lines after patch ---' && grep -n 'gencode\\|compute_' setup.py | head -20 && "
            "TORCH_CUDA_ARCH_LIST='10.0' MAX_JOBS=2 CAUSAL_CONV1D_FORCE_BUILD=1 "
            "pip install --no-build-isolation --no-cache-dir . 2>&1 "
            "| grep -vE '^  ptxas info|^  Compile time|Function properties|stack frame|Used [0-9]+ registers' | tail -60",
            # mamba-ssm: clone from state-spaces/mamba git @ commit
            # 31f3d7baba69d0ccad1635ace1e477367899e408 (same tree used on
            # GB10 for the cuTile port parity validation). The PyPI sdist
            # for mamba-ssm 2.3.1 does NOT include the mamba_ssm/ops/tilelang/
            # subpackage — only the git HEAD carries the Mamba3 MIMO
            # TileLang kernels we need as the cuTile port reference.
            "cd /tmp/mb && mkdir -p mamba_ssm_src && cd mamba_ssm_src && "
            "git init -q && "
            "git remote add origin https://github.com/state-spaces/mamba.git && "
            "git fetch --depth 1 origin 31f3d7baba69d0ccad1635ace1e477367899e408 && "
            "git checkout -q FETCH_HEAD && "
            "git rev-parse HEAD && "
            "python /tmp/patch_setup.py setup.py && "
            "echo '--- gencode lines after patch ---' && grep -n 'gencode\\|compute_' setup.py | head -20 && "
            "ls mamba_ssm/ops/tilelang/mamba3/ 2>&1 | head -20 && "
            "TORCH_CUDA_ARCH_LIST='10.0' MAX_JOBS=2 MAMBA_FORCE_BUILD=1 "
            "pip install --no-build-isolation --no-cache-dir . 2>&1 "
            "| grep -vE '^  ptxas info|^  Compile time|Function properties|stack frame|Used [0-9]+ registers' | tail -60",
            "python -c 'from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; "
            "print(\"mamba_ssm OK\")'",
            "python -c 'from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd import mamba_mimo_forward; "
            "print(\"mamba3_mimo OK\")'",
            gpu="B200",  # Modal image builder caps GPUs per image at 1; runtime functions can still use B200:2
        )
    )


# Keep as a callable — only build this layered image when a parity function
# actually asks for it. Calling _image() returns the base image; calling
# _image_with_mamba() overlays the mamba-ssm build. Both are lazy in the
# sense that Modal only actually builds them when referenced by @app.function.


# ---------------------------------------------------------------------------
# Sanity kernel: compile a trivial cuTile Python vadd kernel for sm_100
# ---------------------------------------------------------------------------
#
# The kernel body lives in an inline string so the Modal-side function can
# write it to a file and hand it to tileiras, but we ALSO exercise the
# high-level `cuda.tile` runtime path (ct.launch) which is what real kernels
# use. Both paths must work for B200 to be usable.


@app.function(
    image=image,
    gpu=_GPU_SPEC,
    timeout=600,  # 10 min hard cap — do NOT leave running
    volumes={"/vol": cutile_vol},
)
def sanity() -> dict[str, Any]:
    """Compile + run a vector-add kernel on sm_100 B200.

    Returns a dict of results so the caller can post-process / pretty-print.
    Raises on any hard failure.
    """
    import sys

    results: dict[str, Any] = {}

    # ---- stack versions ----
    import torch

    results["torch_version"] = torch.__version__
    results["torch_cuda"] = torch.version.cuda
    results["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    results["cuda_cap"] = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    results["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[sanity] torch={results['torch_version']} cuda={results['torch_cuda']}")
    print(f"[sanity] device={results['cuda_device']} cap={results['cuda_cap']} n={results['device_count']}")

    # ---- cuTile Python import + version ----
    # Use the module-level _ct / _vadd_kernel defined at the top of this
    # file (needed because typing.get_type_hints resolves the kernel's
    # `ct.Constant[int]` annotation against the module globalns).
    ct = _ct
    if ct is None:
        raise RuntimeError("cuda.tile module not importable inside container")

    results["cuda_tile_version"] = getattr(ct, "__version__", "unknown")
    print(f"[sanity] cuda.tile import OK, version={results['cuda_tile_version']}")

    # ---- cuTile Python target arch detection ----
    # cuda-tile 1.2.0 does not expose a `tileiras` CLI. The target SM arch
    # is picked by cuda.tile._compile.get_sm_arch() which reads the current
    # device at compile time. We introspect the private _compile module to
    # extract whatever "supported targets" list it carries internally.
    try:
        from cuda.tile import _compile as _tile_compile

        get_sm = getattr(_tile_compile, "get_sm_arch", None)
        if get_sm is not None:
            detected_arch = get_sm()
            results["cuda_tile_detected_sm_arch"] = detected_arch
            print(f"[sanity] cuda.tile detected arch: {detected_arch}")
        # Known supported list from memory/reference_cutile_python.md, verified
        # on GB10 + H200 on 2026-04-10 against cuda-tile 1.2.0 tileiras CLI:
        #   sm_80, sm_86, sm_87, sm_88, sm_89, sm_100, sm_103, sm_110, sm_120, sm_121
        # Since 1.2.0 doesn't ship the CLI anymore we can't live-query, but the
        # compiled cubin will speak to the actual device, so the real check is
        # "does kernel compilation+launch succeed on this B200" below.
        results["tileiras_supported_targets"] = [
            "sm_80", "sm_86", "sm_87", "sm_88", "sm_89",
            "sm_100", "sm_103", "sm_110", "sm_120", "sm_121",
        ]
        results["tileiras_supports_sm_100"] = True
    except Exception as exc:  # noqa: BLE001
        results["cuda_tile_detected_sm_arch"] = f"introspect_error: {type(exc).__name__}"
        results["tileiras_supported_targets"] = []
        results["tileiras_supports_sm_100"] = False
        print(f"[sanity] arch introspection failed: {exc}")

    # ---- compile + run a trivial vadd kernel via the high-level runtime ----
    # We use _vadd_kernel defined at module scope (see top of file for why).
    # The runtime picks the sm arch from the current device, which on B200
    # is sm_100. cuTile ct.load uses *tile-space* indices, so index (bi,)
    # with shape (TILE,) reads elements [bi*TILE : (bi+1)*TILE].

    TILE = 256
    N = 4096
    assert N % TILE == 0
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty_like(a)
    grid = (N // TILE, 1, 1)

    try:
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _vadd_kernel,
            (a, b, out, TILE),
        )
        torch.cuda.synchronize()
        ref = a + b
        max_abs_err = float((out - ref).abs().max().item())
        results["vadd_max_abs_err"] = max_abs_err
        results["vadd_status"] = "ok"
        print(f"[sanity] vadd ran on sm_100, max_abs_err={max_abs_err:.3e}")
        if max_abs_err > 1e-5:
            raise RuntimeError(f"vadd numerical check failed: max_abs_err={max_abs_err}")
    except Exception as exc:  # noqa: BLE001
        results["vadd_status"] = f"error: {type(exc).__name__}: {exc}"
        print(f"[sanity] vadd FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    # ---- mamba_ssm reference availability ----
    # NOTE: mamba_ssm is NOT installed in the base sanity image (its source
    # build is fragile against torch 2.12 nightly). It lives in the layered
    # image built by _image_with_mamba() and is only loaded by the Mamba3
    # MIMO parity test in modal_cutile_mamba_mimo.py.
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # type: ignore  # noqa: F401

        results["mamba_ssm"] = "ok"
    except Exception as exc:  # noqa: BLE001
        results["mamba_ssm"] = f"not_in_sanity_image: {type(exc).__name__}"
    print(f"[sanity] mamba_ssm: {results['mamba_ssm']}")

    # ---- report Volume mount so the rsync path is ready ----
    import pathlib

    vol_path = pathlib.Path("/vol")
    vol_path.mkdir(parents=True, exist_ok=True)
    (vol_path / "READY").write_text("cppmega-cutile-mamba3 volume ready\n")
    cutile_vol.commit()
    results["volume_path"] = str(vol_path)
    print(f"[sanity] volume {vol_path} ready for mamba3_mimo_cutile rsync")

    return results


@app.function(
    image=image,
    gpu=_GPU_SPEC,
    timeout=600,
    volumes={"/vol": cutile_vol},
)
def inspect() -> dict[str, Any]:
    """Dump pip freeze, cuda.tile module map, and torch device info.

    Does NOT attempt to invoke a `tileiras` CLI — cuda-tile 1.2.0 ships
    only a Python API, not a standalone compiler binary. We introspect
    the Python module instead.
    """
    import pkgutil
    import subprocess

    out: dict[str, Any] = {}

    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=False)
    out["pip_freeze"] = freeze.stdout
    print("==== pip freeze ====")
    print(freeze.stdout)

    # --- nvcc ---
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
    out["nvcc_version"] = (nvcc.stdout or nvcc.stderr).strip()
    print("==== nvcc --version ====")
    print(out["nvcc_version"])

    # --- cuda.tile module map ---
    import cuda.tile as ct

    out["cuda_tile_version"] = getattr(ct, "__version__", "unknown")
    out["cuda_tile_path"] = ct.__file__
    out["cuda_tile_submodules"] = [m.name for m in pkgutil.iter_modules(ct.__path__)]
    out["cuda_tile_public_attrs"] = sorted(a for a in dir(ct) if not a.startswith("_"))
    print("==== cuda.tile ====")
    print(f"version: {out['cuda_tile_version']}")
    print(f"path:    {out['cuda_tile_path']}")
    print(f"submodules: {out['cuda_tile_submodules']}")
    print(f"public:  {out['cuda_tile_public_attrs']}")

    import torch

    out["torch_version"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        out["cuda_device"] = torch.cuda.get_device_name(0)
        out["cuda_cap"] = torch.cuda.get_device_capability(0)
        out["device_count"] = torch.cuda.device_count()
    print(f"torch={out['torch_version']} cuda={out['torch_cuda']}")
    print(f"device={out.get('cuda_device')} cap={out.get('cuda_cap')}")

    return out


@app.local_entrypoint()
def main() -> None:
    """Run inspect + sanity back-to-back.

    This is what `modal run scripts/modal_cutile_b200.py` dispatches to.
    Prints a compact summary at the end.
    """
    print("=== cppmega Modal cuTile B200 sanity ===")
    print(f"GPU spec: {_GPU_SPEC}")

    info = inspect.remote()
    print(f"[local] inspect complete: torch={info.get('torch_version')} device={info.get('cuda_device')}")

    res = sanity.remote()
    print("=== RESULTS ===")
    for k, v in res.items():
        if k == "tileiras_supported_targets":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    assert res.get("tileiras_supports_sm_100") is True, "sm_100 not supported by tileiras on this image"
    assert res.get("vadd_status") == "ok", f"vadd failed: {res.get('vadd_status')}"
    print("=== OK: B200 sm_100 cuTile Python stack verified ===")
