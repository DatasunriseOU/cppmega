"""Modal app: run the full cuTile Python Mamba3 MIMO port on B200:2.

Parity + perf validation for the cuTile Python port of mamba_mimo_fwd /
bwd_fwd / bwd_bwd. Runs unchanged test scripts from the GB10 port
(synced locally via rsync) against the upstream mamba_ssm==2.3.1 TileLang
reference on datacenter Blackwell sm_100.

What this runs
--------------
1. test_fwd_correctness.py        — forward kernel parity
2. test_bwd_fwd_correctness.py    — bwd_fwd kernel parity
3. test_bwd_bwd_correctness.py    — bwd_bwd kernel parity (validates the
                                    DFACTOR + DQ fixes landed 2026-04-10)
4. test_e2e_optimization.py       — 50-step AdamW convergence vs TileLang
5. bench_bwd.py                   — TileLang vs cuTile wall-clock

The port source files live at /Volumes/external/sources/cppmega/.tmp/mamba3_mimo_cutile/
(rsynced from GB10). The tests hardcode sys.path.insert(0, "/home/dave/mamba3_mimo_cutile")
and importlib.spec_from_file_location("/tmp/mamba3_mimo_bwd_phase0.py"). We do NOT modify
the source files — instead we create symlinks in the container so those absolute paths
still resolve inside Modal. See _runner_entrypoint below.

Cost: image build ~10-15 min (mamba_ssm from source), suite run ~10-15 min.
Hard cap 30 min total via timeout=1800.
"""
# ruff: noqa: E402

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Image build — fully self-contained (does not import modal_cutile_b200.py)
# ---------------------------------------------------------------------------
#
# The layers match modal_cutile_b200.py::_image() + _image_with_mamba() but
# are defined inline here so this file is hermetic inside a Modal container
# (where /root/modal_cutile_mamba_mimo.py has no sibling `scripts/` package
# on sys.path). The image graph is memoised by Modal based on the layer
# spec hash, so re-running this file does NOT rebuild the base cuTile stack
# that modal_cutile_b200.py already built — as long as the layers are
# byte-identical. Keep the apt/pip/env layers below in lockstep with
# modal_cutile_b200.py::_image() so the cache hit survives.

_PYTHON = "3.13"
_TORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly/cu132"


def _base_cutile_image() -> modal.Image:
    """Build the base cuTile + TileLang + torch image (no mamba-ssm)."""
    base: Any = modal.Image.debian_slim(python_version=_PYTHON)
    img = (
        base.apt_install("git", "build-essential", "ninja-build", "curl", "ca-certificates")
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
        .pip_install(
            "nvidia-cuda-nvcc",
            extra_index_url="https://pypi.nvidia.com",
        )
        .pip_install(
            "cuda-tile[tileiras]",
            "apache-tvm-ffi==0.1.9",
            "tilelang==0.1.8",
        )
        .env(
            {
                "PATH": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/bin:"
                        "/usr/local/lib/python3.13/site-packages/cuda_tile/bin:"
                        "/usr/local/bin:/usr/bin:/bin",
                "CUDA_HOME": "/usr/local/lib/python3.13/site-packages/nvidia/cu13",
            }
        )
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


_PATCH_FILE = "/Volumes/external/sources/cppmega/scripts/_modal_patch_mamba_setup.py"


def _image_with_mamba_inline() -> modal.Image:
    """Overlay mamba-ssm + causal-conv1d source builds for sm_100 B200.

    Mirrors modal_cutile_b200.py::_image_with_mamba() one-to-one so Modal's
    layer cache can be reused. The `.add_local_file` for the patch script
    uses the repo-absolute path since this file is only ever launched from
    the cppmega root via `modal run scripts/modal_cutile_mamba_mimo.py`.
    """
    base: Any = _base_cutile_image()
    return (
        base
        .pip_install("nvidia-cuda-cccl", extra_index_url="https://pypi.nvidia.com")
        .env({"CPLUS_INCLUDE_PATH": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/include",
              "C_INCLUDE_PATH": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/include",
              "CUDA_INCLUDE_DIRS": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/include",
              "LIBRARY_PATH": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib:"
                              "/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib/x86_64-linux-gnu:"
                              "/usr/local/lib/python3.13/site-packages/torch/lib",
              "LD_LIBRARY_PATH": "/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib:"
                                 "/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib/x86_64-linux-gnu:"
                                 "/usr/local/lib/python3.13/site-packages/torch/lib"})
        .add_local_file(_PATCH_FILE, "/tmp/patch_setup.py", copy=True)
        .run_commands(
            "ls /usr/local/lib/python3.13/site-packages/nvidia/cu13/include/nv/target 2>&1 || "
            "(echo '### cu13/include tree:' && find /usr/local/lib/python3.13/site-packages/nvidia/cu13/include -maxdepth 2 -type d && exit 1)",
            "pip install --no-build-isolation --no-cache-dir packaging ninja wheel setuptools",
            "echo '--- libcudart search ---' && "
            "find /usr/local/lib/python3.13/site-packages/nvidia -name 'libcudart*' 2>/dev/null && "
            "find /usr/local/lib/python3.13/site-packages/torch -name 'libcudart*' 2>/dev/null | head -5",
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
            gpu="B200",
        )
    )


_image_with_mamba = _image_with_mamba_inline

_GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "B200:2")
_LOCAL_PORT_DIR = pathlib.Path(
    os.environ.get(
        "CPPMEGA_CUTILE_PORT_DIR",
        "/Volumes/external/sources/cppmega/.tmp/mamba3_mimo_cutile",
    )
)

app = modal.App("cppmega-cutile-mamba3-mimo")
cutile_vol = modal.Volume.from_name("cppmega-cutile-mamba3", create_if_missing=True)


def _image_with_port() -> modal.Image:
    """Overlay mamba_ssm build AND the local mamba3_mimo_cutile port files.

    Layer order:
      1. base cuTile + tilelang + torch image
      2. mamba_ssm source build (_image_with_mamba, needs GPU)
      3. add_local_dir of the rsynced cuTile port + patched phase0 reference

    The `add_local_dir` step is skipped unconditionally when Modal evaluates
    this file inside the container — the container already has everything
    baked into the image on the local side, and re-executing the step would
    fail because `_LOCAL_PORT_DIR` is a macOS-host absolute path.
    """
    img: Any = _image_with_mamba()
    # Only call add_local_dir when the local path exists. Inside the Modal
    # container this file is re-imported (to look up `run_parity_suite`) but
    # the local dir obviously doesn't exist there — in that case we just
    # return the base image, since the add_local_dir blob was already
    # attached on the original local invocation and the container picks it
    # up from Modal's blob store.
    if _LOCAL_PORT_DIR.exists() and any(_LOCAL_PORT_DIR.iterdir()):
        img = img.add_local_dir(
            str(_LOCAL_PORT_DIR),
            "/root/mamba3_mimo_cutile",
            copy=True,
        )
    return img


image = _image_with_port()


@app.function(
    image=image,
    gpu=_GPU_SPEC,
    timeout=1800,  # 30 min hard cap (mamba_ssm build already baked in)
    volumes={"/vol": cutile_vol},
)
def run_parity_suite() -> dict[str, Any]:
    """Run all four correctness tests + bench_bwd inside the B200:2 container.

    Returns a dict of per-test status/stdout so the caller can post-process.
    """
    import shutil

    results: dict[str, Any] = {}

    # ---- stack versions ----
    import torch

    results["torch_version"] = torch.__version__
    results["torch_cuda"] = torch.version.cuda
    results["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    results["cuda_cap"] = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    results["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[suite] torch={results['torch_version']} cuda={results['torch_cuda']}")
    print(f"[suite] device={results['cuda_device']} cap={results['cuda_cap']} n={results['device_count']}")

    # ---- mamba_ssm sanity ----
    try:
        import mamba_ssm  # type: ignore[import-not-found]

        results["mamba_ssm_version"] = mamba_ssm.__version__
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd import mamba_mimo_forward  # type: ignore[import-not-found]  # noqa: F401
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import mamba_mimo_bwd_fwd  # type: ignore[import-not-found]  # noqa: F401

        results["mamba_ssm_tilelang_import"] = "ok"
    except Exception as exc:  # noqa: BLE001
        results["mamba_ssm_version"] = "missing"
        results["mamba_ssm_tilelang_import"] = f"error: {type(exc).__name__}: {exc}"
        raise

    # ---- Recreate GB10's absolute paths so the unmodified test scripts work. ----
    # The scripts all do `sys.path.insert(0, "/home/dave/mamba3_mimo_cutile")`
    # and refer to "/tmp/mamba3_mimo_bwd_phase0.py". We honor that by symlinking
    # rather than editing the source files (port source files are frozen).
    home_dave_path = pathlib.Path("/home/dave")
    home_dave_path.mkdir(parents=True, exist_ok=True)
    gb10_port_path = home_dave_path / "mamba3_mimo_cutile"
    if gb10_port_path.exists() or gb10_port_path.is_symlink():
        if gb10_port_path.is_symlink():
            gb10_port_path.unlink()
        else:
            shutil.rmtree(gb10_port_path)
    gb10_port_path.symlink_to("/root/mamba3_mimo_cutile")

    phase0_src = pathlib.Path("/root/mamba3_mimo_cutile/mamba3_mimo_bwd_phase0.py")
    phase0_dst = pathlib.Path("/tmp/mamba3_mimo_bwd_phase0.py")
    if phase0_dst.exists() or phase0_dst.is_symlink():
        phase0_dst.unlink()
    if not phase0_src.exists():
        raise RuntimeError(
            f"Patched TileLang reference missing at {phase0_src}. "
            "The rsync from gb10:/tmp/mamba3_mimo_bwd_phase0.py must have been skipped."
        )
    phase0_dst.symlink_to(phase0_src)

    print(f"[suite] symlink: {gb10_port_path} -> /root/mamba3_mimo_cutile")
    print(f"[suite] symlink: {phase0_dst} -> {phase0_src}")
    print(f"[suite] port dir listing:")
    for p in sorted(gb10_port_path.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")

    # ---- Run each test script as a subprocess so stdout is captured whole. ----
    # Using subprocess isolates torch state and matches how the GB10 agent ran
    # them. We use /usr/local/bin/python (modal image ships python here).
    python_exe = sys.executable
    port_root = str(gb10_port_path)
    test_scripts = [
        ("test_fwd_correctness", "test_fwd_correctness.py"),
        ("test_bwd_fwd_correctness", "test_bwd_fwd_correctness.py"),
        ("test_bwd_bwd_correctness", "test_bwd_bwd_correctness.py"),
        ("test_e2e_optimization", "test_e2e_optimization.py"),
        ("bench_bwd", "bench_bwd.py"),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for key, script in test_scripts:
        print(f"\n======================================================")
        print(f"[suite] running {script}")
        print(f"======================================================")
        t0 = time.time()
        proc = subprocess.run(
            [python_exe, f"{port_root}/{script}"],
            capture_output=True,
            text=True,
            env=env,
            cwd=port_root,
            timeout=900,  # 15 min per test
            check=False,
        )
        dt = time.time() - t0
        tail_lines = 200
        out_tail = "\n".join(proc.stdout.splitlines()[-tail_lines:])
        err_tail = "\n".join(proc.stderr.splitlines()[-tail_lines:])
        # Echo to stdout so modal run logs have the full output too.
        print(proc.stdout)
        if proc.stderr:
            print("---- STDERR ----", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
        results[key] = {
            "rc": proc.returncode,
            "elapsed_sec": round(dt, 2),
            "stdout_tail": out_tail,
            "stderr_tail": err_tail,
        }
        print(f"[suite] {script} rc={proc.returncode} elapsed={dt:.2f}s")

    # ---- Also run a per-kernel bench for fwd / bwd_fwd / bwd_bwd individually.
    # bench_bwd.py only times the full chain (bwd_fwd + bwd_bwd). We want
    # per-kernel numbers too. Run bench_fwd.py if present, and run an inline
    # bench that times bwd_fwd and bwd_bwd in isolation.
    print(f"\n======================================================")
    print(f"[suite] running bench_fwd.py (standalone forward bench)")
    print(f"======================================================")
    fwd_bench_path = pathlib.Path(port_root) / "bench_fwd.py"
    if fwd_bench_path.exists():
        t0 = time.time()
        proc = subprocess.run(
            [python_exe, str(fwd_bench_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=port_root,
            timeout=900,
            check=False,
        )
        dt = time.time() - t0
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        results["bench_fwd"] = {
            "rc": proc.returncode,
            "elapsed_sec": round(dt, 2),
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-200:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-200:]),
        }
    else:
        results["bench_fwd"] = {"rc": -1, "elapsed_sec": 0, "stdout_tail": "bench_fwd.py not present", "stderr_tail": ""}

    # Inline split bench (bwd_fwd only / bwd_bwd only). Mirrors bench_bwd.py
    # but separates the two calls so we get per-kernel TileLang-vs-cuTile numbers.
    print(f"\n======================================================")
    print(f"[suite] running split bench (bwd_fwd only / bwd_bwd only)")
    print(f"======================================================")
    split_bench_py = r"""
import importlib.util, sys, time
sys.path.insert(0, "/home/dave/mamba3_mimo_cutile")
import torch
from mamba3_mimo_fwd_cutile import compute_gamma_and_trap_scale
from mamba3_mimo_bwd_fwd_cutile import mamba3_mimo_bwd_fwd_cutile
from mamba3_mimo_bwd_bwd_cutile import mamba3_mimo_bwd_bwd_cutile

spec = importlib.util.spec_from_file_location("mamba3_mimo_bwd_phase0", "/tmp/mamba3_mimo_bwd_phase0.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
tl_bwd_fwd_factory = mod.mamba_mimo_bwd_fwd
tl_bwd_bwd_factory = mod.mamba_mimo_bwd_bwd

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

torch.manual_seed(42)
device = torch.device("cuda")
dtype = torch.bfloat16
B, S, H, G, N, P, R = 2, 256, 8, 1, 64, 64, 4
chunk_size = 16
rotary_dim_divisor = 4
nchunks = S // chunk_size

Q = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.1
K = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.1
V = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.1
DOUT = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.1
Q_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.1
K_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.1
MIMO_V = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.1
MIMO_O = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.1
angles = torch.randn(B, S, H, N // rotary_dim_divisor, device=device, dtype=torch.float32)
dt = torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.01
trap = torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.5
trap_bf = trap.to(dtype)
adt = -torch.abs(torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.1)
DA_CS, DA_CS_REV, SEGSUM = compute_dacs_segsum_triton(adt, chunk_size)
gamma, trap_scale = compute_gamma_and_trap_scale(dt, trap_bf, chunk_size, dtype)

z_dummy = torch.zeros(B, S, H, P, dtype=dtype, device=device)
dz_dummy = torch.zeros(B, S, H, P, dtype=dtype, device=device)
dmimo_z_dummy = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
mimo_z_dummy = torch.zeros(H, R, P, dtype=torch.float32, device=device)
D_dummy = torch.zeros(H, dtype=torch.float32, device=device)

tl_bf_kernel = tl_bwd_fwd_factory(B, S, H, G, N, P, R, False, False, True, chunk_size, rotary_dim_divisor, "bfloat16")
tl_bb_kernel = tl_bwd_bwd_factory(B, S, H, G, N, P, R, False, False, True, chunk_size, rotary_dim_divisor, "bfloat16")

tl_dmimo_o = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
tl_states = torch.zeros(B, H, nchunks, N, P, dtype=dtype, device=device)
tl_qk_dot = torch.zeros(B, H, S, R, R, dtype=dtype, device=device)
tl_dk = torch.zeros(B, S * R, H, N, dtype=dtype, device=device)
tl_dv = torch.zeros(B, S, H, P, dtype=dtype, device=device)
tl_dmimo_v = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
tl_dq = torch.zeros(B, S * R, H, N, dtype=dtype, device=device)
tl_dfactor = torch.zeros(B, H, S, dtype=torch.float32, device=device)
tl_dgamma_diag = torch.zeros(B, H, S, dtype=torch.float32, device=device)
tl_dangles = torch.zeros(B, S, H, N // rotary_dim_divisor, dtype=angles.dtype, device=device)
tl_dD = torch.zeros(B, H, dtype=torch.float32, device=device)
tl_dda = torch.zeros(B, H, S, dtype=torch.float32, device=device)
tl_dssda = torch.zeros(B, H, nchunks, chunk_size, chunk_size, dtype=torch.float32, device=device)
tl_dda_cs_rev = torch.zeros(B, H, S, dtype=torch.float32, device=device)
tl_dda_cs = torch.zeros(B, H, S, dtype=torch.float32, device=device)

def tl_bf_fn():
    tl_bf_kernel(
        DOUT, Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_O,
        tl_dmimo_o, tl_states, z_dummy, mimo_z_dummy, dz_dummy, dmimo_z_dummy,
        angles, DA_CS, DA_CS_REV, dt, trap_bf, D_dummy, tl_qk_dot, SEGSUM)

def tl_bb_fn():
    tl_bb_kernel(
        DOUT, Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_O,
        tl_dk, tl_dv, tl_dmimo_v, tl_states, tl_dq,
        z_dummy, mimo_z_dummy, angles, DA_CS, DA_CS_REV, dt, trap_bf,
        tl_dfactor, tl_dgamma_diag, tl_dangles, D_dummy, tl_dD,
        tl_qk_dot, tl_dda, tl_dssda, tl_dda_cs_rev, tl_dda_cs, SEGSUM)

def ct_bf_fn():
    return mamba3_mimo_bwd_fwd_cutile(
        DOUT=DOUT, Q=Q, K=K, V=V, Q_bias=Q_bias, K_bias=K_bias,
        MIMO_V=MIMO_V, MIMO_O=MIMO_O,
        Z=None, D=None, MIMO_Z=None,
        angles=angles, DA_CS=DA_CS, DA_CS_REV=DA_CS_REV,
        DT_gamma=gamma, TRAP_scale=trap_scale, SEGSUM=SEGSUM,
        chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor)

# Prime a cuTile bwd_fwd output so ct_bb has STATES/QK_DOT
_ct_dmimo_o, _ct_states, _ct_qk_dot, _, _ = ct_bf_fn()

def ct_bb_fn():
    _ = mamba3_mimo_bwd_bwd_cutile(
        DOUT=DOUT, Q=Q, K=K, V=V, Q_bias=Q_bias, K_bias=K_bias,
        MIMO_V=MIMO_V, MIMO_O=MIMO_O,
        STATES=_ct_states, QK_DOT=_ct_qk_dot,
        Z=None, D=None, MIMO_Z=None,
        angles=angles, DA_CS=DA_CS, DA_CS_REV=DA_CS_REV,
        DT_gamma=gamma, TRAP_scale=trap_scale, SEGSUM=SEGSUM,
        chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor)

def time_fn(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

# Warm both paths first
for _ in range(3):
    tl_bf_fn(); tl_bb_fn(); ct_bf_fn(); ct_bb_fn()
torch.cuda.synchronize()

tl_bf_ms = time_fn(tl_bf_fn)
tl_bb_ms = time_fn(tl_bb_fn)
ct_bf_ms = time_fn(ct_bf_fn)
ct_bb_ms = time_fn(ct_bb_fn)

print(f"SPLIT_BENCH shape B={B} S={S} H={H} N={N} P={P} R={R}")
print(f"SPLIT_BENCH tl_bwd_fwd_ms  {tl_bf_ms:.4f}")
print(f"SPLIT_BENCH tl_bwd_bwd_ms  {tl_bb_ms:.4f}")
print(f"SPLIT_BENCH ct_bwd_fwd_ms  {ct_bf_ms:.4f}")
print(f"SPLIT_BENCH ct_bwd_bwd_ms  {ct_bb_ms:.4f}")
print(f"SPLIT_BENCH tl_chain_ms    {tl_bf_ms + tl_bb_ms:.4f}")
print(f"SPLIT_BENCH ct_chain_ms    {ct_bf_ms + ct_bb_ms:.4f}")
print(f"SPLIT_BENCH ratio_bf       {ct_bf_ms/tl_bf_ms:.3f}")
print(f"SPLIT_BENCH ratio_bb       {ct_bb_ms/tl_bb_ms:.3f}")
print(f"SPLIT_BENCH ratio_chain    {(ct_bf_ms+ct_bb_ms)/(tl_bf_ms+tl_bb_ms):.3f}")
"""
    t0 = time.time()
    proc = subprocess.run(
        [python_exe, "-c", split_bench_py],
        capture_output=True,
        text=True,
        env=env,
        cwd=port_root,
        timeout=900,
        check=False,
    )
    dt = time.time() - t0
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    results["split_bench"] = {
        "rc": proc.returncode,
        "elapsed_sec": round(dt, 2),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-200:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-200:]),
    }

    return results


@app.local_entrypoint()
def main() -> None:
    if not _LOCAL_PORT_DIR.exists():
        print(f"[error] Port dir not found at {_LOCAL_PORT_DIR}.")
        print("[error] Steps: (1) bring GB10 back, (2) rsync the port dir, (3) rerun.")
        sys.exit(1)

    print(f"=== cppmega cuTile Mamba3 MIMO parity + perf on {_GPU_SPEC} ===")
    print(f"[local] port dir: {_LOCAL_PORT_DIR}")
    port_files = sorted(_LOCAL_PORT_DIR.iterdir())
    print(f"[local] {len(port_files)} files will be injected into /root/mamba3_mimo_cutile")

    res = run_parity_suite.remote()

    print("\n=== RESULTS SUMMARY ===")
    keys = [
        "torch_version", "cuda_device", "cuda_cap", "mamba_ssm_version",
        "mamba_ssm_tilelang_import",
    ]
    for k in keys:
        print(f"  {k}: {res.get(k)}")

    test_keys = [
        "test_fwd_correctness", "test_bwd_fwd_correctness",
        "test_bwd_bwd_correctness", "test_e2e_optimization",
        "bench_bwd", "bench_fwd", "split_bench",
    ]
    for k in test_keys:
        entry = res.get(k, {})
        rc = entry.get("rc", "?")
        elapsed = entry.get("elapsed_sec", "?")
        print(f"\n  [{k}] rc={rc} elapsed={elapsed}s")
        tail = entry.get("stdout_tail", "")
        if tail:
            for line in tail.splitlines()[-30:]:
                print(f"    | {line}")

    import json
    out_path = pathlib.Path("/Volumes/external/sources/cppmega/.tmp/modal_b200_parity_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[local] full results written to {out_path}")
