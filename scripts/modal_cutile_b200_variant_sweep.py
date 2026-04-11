"""Modal app: sweep cuTile bwd_bwd variants on B200:2 (sm_100a).

Background
----------
GB10 (sm_121a consumer Blackwell) exhausted algorithmic rewrites of the
cuTile Python port of mamba3 MIMO bwd_bwd — all 5 variants regressed vs
the 2-kernel A/B split baseline (624 us). The likely root cause is
GB10's narrow smem budget (99 KB) + absent TMEM/tcgen05 + absent TMA
multicast: the compiler has no room to relax register pressure under
fusion or invariant hoisting.

This script retargets the SAME variants at B200:2 (sm_100a datacenter
Blackwell), which has 228 KB smem/SM, TMEM, tcgen05, and TMA multicast.
The hypothesis: variants that LOST on GB10 (V2 fused monolithic, V3
3-kernel split, V4 hoisted loop invariants) may WIN on B200 because the
compiler has ~2.3x more smem to spill cold tiles into without thrashing
the register allocator.

We do NOT modify the existing port files under
`.tmp/mamba3_mimo_cutile/` — we sandbox copies of the variant files
under `.tmp/modal_b200_cutile/` and swap them in as
`mamba3_mimo_bwd_bwd_cutile.py` inside the Modal container for each
variant of the sweep.

Ground-truth reference (measured 2026-04-10 on the same B200 image):
- TileLang bwd_bwd (on B200): 0.1790 ms = 179 us
- TileLang bwd_fwd (on B200): 0.0724 ms =  72 us
- cuTile baseline bwd_bwd    : 0.6815 ms = 681 us  (3.81x slower than TL)
- cuTile baseline bwd_fwd    : 0.1626 ms = 163 us  (2.25x slower than TL)

So: TL chain 253 us ; cuTile chain 844 us on B200 pre-sweep.

Usage
-----
    modal run scripts/modal_cutile_b200_variant_sweep.py::main

Cost: ~8 min image-cache hit + 4-5 min B200:2 compute per variant x 4
variants = < 25 min on B200:2 = ~$10-12 upper bound.
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
# Image build — MUST match modal_cutile_mamba_mimo.py byte-for-byte so we hit
# the existing Modal image cache (mamba_ssm source build is ~10 min otherwise).
# ---------------------------------------------------------------------------

_PYTHON = "3.13"
_TORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly/cu132"

_PATCH_FILE = "/Volumes/external/sources/cppmega/scripts/_modal_patch_mamba_setup.py"


def _base_cutile_image() -> modal.Image:
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


def _image_with_mamba_inline() -> modal.Image:
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
_LOCAL_VARIANTS_DIR = pathlib.Path(
    "/Volumes/external/sources/cppmega/.tmp/modal_b200_cutile"
)

app = modal.App("cppmega-cutile-b200-variant-sweep")
cutile_vol = modal.Volume.from_name("cppmega-cutile-mamba3", create_if_missing=True)


def _image_with_port_and_variants() -> modal.Image:
    """Base mamba_ssm image + port files + variant sandbox."""
    img: Any = _image_with_mamba()
    if _LOCAL_PORT_DIR.exists() and any(_LOCAL_PORT_DIR.iterdir()):
        img = img.add_local_dir(
            str(_LOCAL_PORT_DIR),
            "/root/mamba3_mimo_cutile",
            copy=True,
        )
    if _LOCAL_VARIANTS_DIR.exists() and any(_LOCAL_VARIANTS_DIR.iterdir()):
        img = img.add_local_dir(
            str(_LOCAL_VARIANTS_DIR),
            "/root/variants",
            copy=True,
        )
    return img


image = _image_with_port_and_variants()


# ---------------------------------------------------------------------------
# Variant sweep body
# ---------------------------------------------------------------------------

_VARIANTS_ALL = [
    # name, filename in /root/variants, notes
    ("baseline",     "variant_baseline.py",     "current 2-kernel A/B split (already in repo)"),
    ("v2_fused",     "variant_v2_fused_mono.py","single fused kernel, no A/B split"),
    ("v3_split3",    "variant_v3_split3.py",    "4-kernel split (A_dv + A_dk + B + carry)"),
    ("v4_hoisted",   "variant_v4_hoisted.py",   "baseline + hoisted loop invariants"),
    ("v7_occ1",      "variant_v7_occupancy1.py","baseline + @ct.kernel(occupancy=1)"),
    ("v8_hoist_occ1","variant_v8_hoisted_occ1.py","v4 hoisted + @ct.kernel(occupancy=1)"),
]

# Sweep set is controlled by env var so we can re-run with only the new
# variants after the first sweep finishes. CPPMEGA_SWEEP_VARIANTS=v7_occ1,v8_hoist_occ1
# restricts to those two; empty/unset runs everything.
_VARIANT_FILTER = os.environ.get("CPPMEGA_SWEEP_VARIANTS", "").strip()
if _VARIANT_FILTER:
    _selected = set(v.strip() for v in _VARIANT_FILTER.split(",") if v.strip())
    _VARIANTS = [v for v in _VARIANTS_ALL if v[0] in _selected]
else:
    _VARIANTS = _VARIANTS_ALL


_BENCH_ONLY_PY = r"""
import importlib, importlib.util, sys, time, json, os

sys.path.insert(0, "/home/dave/mamba3_mimo_cutile")
import torch

# Forward drops the mimo fwd helper we need for gamma/trap_scale
from mamba3_mimo_fwd_cutile import compute_gamma_and_trap_scale
from mamba3_mimo_bwd_fwd_cutile import mamba3_mimo_bwd_fwd_cutile

# Import the CURRENT mamba3_mimo_bwd_bwd_cutile.py in the port dir (whichever
# variant the outer runner just copied in).
from mamba3_mimo_bwd_bwd_cutile import mamba3_mimo_bwd_bwd_cutile

# TileLang reference factory (patched)
spec = importlib.util.spec_from_file_location(
    "mamba3_mimo_bwd_phase0", "/tmp/mamba3_mimo_bwd_phase0.py"
)
_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(_mod)
tl_bwd_fwd_factory = _mod.mamba_mimo_bwd_fwd
tl_bwd_bwd_factory = _mod.mamba_mimo_bwd_bwd

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

# --- TileLang persistent scratch buffers (same as bench_bwd.py) ---
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
        MIMO_V=MIMO_V, MIMO_O=MIMO_O, Z=None, D=None, MIMO_Z=None,
        angles=angles, DA_CS=DA_CS, DA_CS_REV=DA_CS_REV,
        DT_gamma=gamma, TRAP_scale=trap_scale, SEGSUM=SEGSUM,
        chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor)

_ct_dmimo_o, _ct_states, _ct_qk_dot, _, _ = ct_bf_fn()

def ct_bb_fn():
    return mamba3_mimo_bwd_bwd_cutile(
        DOUT=DOUT, Q=Q, K=K, V=V, Q_bias=Q_bias, K_bias=K_bias,
        MIMO_V=MIMO_V, MIMO_O=MIMO_O,
        STATES=_ct_states, QK_DOT=_ct_qk_dot,
        Z=None, D=None, MIMO_Z=None,
        angles=angles, DA_CS=DA_CS, DA_CS_REV=DA_CS_REV,
        DT_gamma=gamma, TRAP_scale=trap_scale, SEGSUM=SEGSUM,
        chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor)

# --- correctness check vs TileLang ---
tl_bf_fn()
tl_bb_fn()
torch.cuda.synchronize()

ct_out = ct_bb_fn()
torch.cuda.synchronize()

# ct_out order: (DK_OUT, DV_OUT, DMIMO_V_OUT, DQ_OUT, DANGLES_OUT, DFACTOR_OUT,
#                DGAMMA_DIAG_OUT, DDA_OUT, DSSDA_OUT, DDA_CS_REV_OUT, DDA_CS_OUT, DD_or_None)
ct_DK, ct_DV, ct_DMIMO_V, ct_DQ, ct_DANGLES, ct_DFACTOR, ct_DGAMMA_DIAG, ct_DDA, ct_DSSDA, ct_DDA_CS_REV, ct_DDA_CS, _ = ct_out

def _diff(a, b, name):
    if a.shape != b.shape:
        return (name, float("nan"), float("nan"), float("nan"), "shape_mismatch")
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)
    d = (a_f - b_f).abs()
    return (name, float(d.max().item()), float(d.mean().item()), float(a_f.abs().max().item()), "ok")

checks = [
    _diff(tl_dv, ct_DV, "DV"),
    _diff(tl_dmimo_v, ct_DMIMO_V, "DMIMO_V"),
    _diff(tl_dfactor, ct_DFACTOR, "DFACTOR"),
    _diff(tl_dgamma_diag, ct_DGAMMA_DIAG, "DGAMMA_DIAG"),
    _diff(tl_dda, ct_DDA, "DDA"),
    _diff(tl_dssda, ct_DSSDA, "DSSDA"),
    _diff(tl_dda_cs_rev, ct_DDA_CS_REV, "DDA_CS_REV"),
    _diff(tl_dda_cs, ct_DDA_CS, "DDA_CS"),
    _diff(tl_dk, ct_DK, "DK"),
    _diff(tl_dq, ct_DQ, "DQ"),
    _diff(tl_dangles, ct_DANGLES, "DANGLES"),
]

GATE_ATOL = 1e-2
correctness_ok = True
for name, mx, mn, tla, status in checks:
    print(f"CHECK {name:12s}  max_diff={mx:.6f}  mean={mn:.6f}  tl_max_abs={tla:.6f}  status={status}")
    if status != "ok" or mx > GATE_ATOL:
        correctness_ok = False

print(f"CORRECTNESS_OK={correctness_ok}")

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

print(f"BENCH tl_bwd_fwd_us={tl_bf_ms*1000:.2f}")
print(f"BENCH tl_bwd_bwd_us={tl_bb_ms*1000:.2f}")
print(f"BENCH ct_bwd_fwd_us={ct_bf_ms*1000:.2f}")
print(f"BENCH ct_bwd_bwd_us={ct_bb_ms*1000:.2f}")
print(f"BENCH ct_chain_us={(ct_bf_ms+ct_bb_ms)*1000:.2f}")
print(f"BENCH tl_chain_us={(tl_bf_ms+tl_bb_ms)*1000:.2f}")
print(f"BENCH ratio_bwd_bwd={ct_bb_ms/tl_bb_ms:.3f}")
print(f"BENCH ratio_chain={(ct_bf_ms+ct_bb_ms)/(tl_bf_ms+tl_bb_ms):.3f}")
"""


@app.function(
    image=image,
    gpu=_GPU_SPEC,
    timeout=3600,
    volumes={"/vol": cutile_vol},
)
def run_variant_sweep() -> dict[str, Any]:
    """Swap each variant into the port dir, run correctness + bench, collect."""
    import shutil

    results: dict[str, Any] = {}

    # ---- stack versions ----
    import torch

    results["torch_version"] = torch.__version__
    results["torch_cuda"] = torch.version.cuda
    results["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    results["cuda_cap"] = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    results["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[sweep] torch={results['torch_version']} cuda={results['torch_cuda']}")
    print(f"[sweep] device={results['cuda_device']} cap={results['cuda_cap']} n={results['device_count']}")

    # ---- sanity: cuda.tile import + sm detect ----
    import cuda.tile as ct  # type: ignore[import-not-found]
    try:
        from cuda.tile import _compile as _tile_compile  # type: ignore[import-not-found]
        get_sm = getattr(_tile_compile, "get_sm_arch", None)
        if get_sm is not None:
            results["cuda_tile_detected_sm_arch"] = get_sm()
            print(f"[sweep] cuda.tile detected arch: {results['cuda_tile_detected_sm_arch']}")
    except Exception as exc:  # noqa: BLE001
        results["cuda_tile_detected_sm_arch"] = f"err: {type(exc).__name__}: {exc}"

    # ---- shim absolute paths the port scripts expect ----
    home_dave_path = pathlib.Path("/home/dave")
    home_dave_path.mkdir(parents=True, exist_ok=True)
    gb10_port_path = home_dave_path / "mamba3_mimo_cutile"
    if gb10_port_path.exists() or gb10_port_path.is_symlink():
        if gb10_port_path.is_symlink():
            gb10_port_path.unlink()
        else:
            shutil.rmtree(gb10_port_path)
    # NOTE: we need the port dir to be a real, writable directory so we can
    # swap `mamba3_mimo_bwd_bwd_cutile.py`. Copy rather than symlink.
    shutil.copytree("/root/mamba3_mimo_cutile", gb10_port_path)

    phase0_src = pathlib.Path("/root/mamba3_mimo_cutile/mamba3_mimo_bwd_phase0.py")
    phase0_dst = pathlib.Path("/tmp/mamba3_mimo_bwd_phase0.py")
    if phase0_dst.exists() or phase0_dst.is_symlink():
        phase0_dst.unlink()
    phase0_dst.symlink_to(phase0_src)

    print(f"[sweep] port dir: {gb10_port_path}")
    print(f"[sweep] variants dir: /root/variants")
    print(f"[sweep] port contents:")
    for p in sorted(gb10_port_path.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")
    print(f"[sweep] variant files:")
    for p in sorted(pathlib.Path("/root/variants").iterdir()):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")

    python_exe = sys.executable
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Force fresh JIT cache per variant so we don't accidentally reuse an
    # old compiled cubin across variants.
    env["CUDA_CACHE_DISABLE"] = "1"

    variant_target = gb10_port_path / "mamba3_mimo_bwd_bwd_cutile.py"

    # Save the original baseline file so we can always restore it.
    baseline_backup = gb10_port_path / "mamba3_mimo_bwd_bwd_cutile.baseline.py"
    shutil.copy2(variant_target, baseline_backup)

    per_variant: list[dict[str, Any]] = []

    for name, filename, notes in _VARIANTS:
        print(f"\n{'=' * 72}")
        print(f"[sweep] VARIANT: {name}  ({filename})")
        print(f"[sweep] {notes}")
        print(f"{'=' * 72}")

        # Copy the variant file in as mamba3_mimo_bwd_bwd_cutile.py
        src = pathlib.Path("/root/variants") / filename
        if not src.exists():
            per_variant.append(
                {"variant": name, "status": "missing_file", "path": str(src)}
            )
            continue
        shutil.copy2(src, variant_target)
        assert variant_target.read_bytes() == src.read_bytes()
        print(f"[sweep] copied {src} -> {variant_target}")

        t0 = time.time()
        proc = subprocess.run(
            [python_exe, "-c", _BENCH_ONLY_PY],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(gb10_port_path),
            timeout=1200,
            check=False,
        )
        dt = time.time() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        # Extract the BENCH + CHECK lines explicitly so we can machine-read them.
        bench_lines = [l for l in stdout.splitlines() if l.startswith("BENCH ") or l.startswith("CHECK ") or l.startswith("CORRECTNESS_OK")]
        parsed: dict[str, Any] = {}
        correctness_ok = None
        for l in bench_lines:
            if l.startswith("BENCH "):
                key_val = l[len("BENCH "):]
                if "=" in key_val:
                    k, v = key_val.split("=", 1)
                    try:
                        parsed[k.strip()] = float(v.strip())
                    except ValueError:
                        parsed[k.strip()] = v.strip()
            elif l.startswith("CORRECTNESS_OK="):
                correctness_ok = l.split("=", 1)[1].strip() == "True"

        # Echo the tail so modal run logs are readable.
        print(stdout[-4000:])
        if stderr:
            print("---- STDERR ----", file=sys.stderr)
            print(stderr[-4000:], file=sys.stderr)

        entry = {
            "variant": name,
            "notes": notes,
            "rc": proc.returncode,
            "elapsed_sec": round(dt, 2),
            "correctness_ok": correctness_ok,
            "bench": parsed,
            "bench_lines": bench_lines,
            "stdout_tail": "\n".join(stdout.splitlines()[-120:]),
            "stderr_tail": "\n".join(stderr.splitlines()[-60:]),
        }
        per_variant.append(entry)
        print(f"[sweep] {name} rc={proc.returncode} elapsed={dt:.2f}s "
              f"correctness_ok={correctness_ok} "
              f"ct_bwd_bwd_us={parsed.get('ct_bwd_bwd_us')} "
              f"tl_bwd_bwd_us={parsed.get('tl_bwd_bwd_us')}")

    # Always restore baseline file before exit so the port dir ends in a
    # known state if anyone cats /root/mamba3_mimo_cutile/ from the volume.
    shutil.copy2(baseline_backup, variant_target)

    results["per_variant"] = per_variant
    return results


@app.local_entrypoint()
def main() -> None:
    if not _LOCAL_PORT_DIR.exists():
        print(f"[error] Port dir not found at {_LOCAL_PORT_DIR}.")
        sys.exit(1)
    if not _LOCAL_VARIANTS_DIR.exists():
        print(f"[error] Variants dir not found at {_LOCAL_VARIANTS_DIR}.")
        sys.exit(1)

    print(f"=== cppmega cuTile bwd_bwd variant sweep on {_GPU_SPEC} ===")
    print(f"[local] port dir:     {_LOCAL_PORT_DIR}")
    print(f"[local] variants dir: {_LOCAL_VARIANTS_DIR}")
    variant_files = sorted(_LOCAL_VARIANTS_DIR.iterdir())
    print(f"[local] {len(variant_files)} variant files staged:")
    for p in variant_files:
        print(f"  - {p.name}  ({p.stat().st_size} bytes)")

    res = run_variant_sweep.remote()

    print("\n=== SWEEP SUMMARY ===")
    print(f"torch={res.get('torch_version')} device={res.get('cuda_device')} cap={res.get('cuda_cap')}")

    header = f"{'variant':20s}  {'rc':>3}  {'correct':>8}  " \
             f"{'tl_bf_us':>10}  {'tl_bb_us':>10}  " \
             f"{'ct_bf_us':>10}  {'ct_bb_us':>10}  {'bb_ratio':>9}"
    print(header)
    print("-" * len(header))
    for entry in res.get("per_variant", []):
        b = entry.get("bench", {}) or {}
        print(f"{entry['variant']:20s}  "
              f"{entry.get('rc', '?'):>3}  "
              f"{str(entry.get('correctness_ok')):>8}  "
              f"{b.get('tl_bwd_fwd_us', float('nan')):10.2f}  "
              f"{b.get('tl_bwd_bwd_us', float('nan')):10.2f}  "
              f"{b.get('ct_bwd_fwd_us', float('nan')):10.2f}  "
              f"{b.get('ct_bwd_bwd_us', float('nan')):10.2f}  "
              f"{b.get('ratio_bwd_bwd', float('nan')):9.3f}")

    import json
    out_path = pathlib.Path(
        "/Volumes/external/sources/cppmega/.tmp/modal_b200_variant_sweep_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[local] full results written to {out_path}")
