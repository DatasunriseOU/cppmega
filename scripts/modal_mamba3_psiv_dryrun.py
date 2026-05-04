"""Modal GHCR dry-run for Mamba3 MIMO Hoist-PsiV.

This is intentionally non-mutating:
  * pulls the prebuilt cppmega image from GHCR,
  * overlays local cppmega scaffolding and a source checkout of state-spaces/mamba,
  * probes source-based ``mamba_ssm.ops.tilelang.mamba3`` patch sites,
  * measures PsiV materialization/write price on real GPUs,
  * optionally compiles the known FloorMod reproducer with a temp-only
    no-FloorMod source rewrite.

Run examples:

    CPPMEGA_MODAL_GPU=H200:2 modal run scripts/modal_mamba3_psiv_dryrun.py
    CPPMEGA_MODAL_GPU=B200+:2 modal run scripts/modal_mamba3_psiv_dryrun.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "latest")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")

APP_NAME = "cppmega-mamba3-psiv-dryrun"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    img = img.add_local_dir("cppmega", f"{CPPMEGA_ROOT}/cppmega", copy=True)
    img = img.add_local_dir(
        "upstream_prs/examples/13_tilelang_floormod_dbz",
        f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz",
        copy=True,
    )
    img = img.add_local_dir(
        "/home/dave/state-spaces-mamba/mamba_ssm",
        f"{SOURCE_ROOT}/mamba_ssm",
        copy=True,
    )
    return img


app = modal.App(APP_NAME)


def _install_source_paths() -> None:
    import sys

    for path in (CPPMEGA_ROOT, SOURCE_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)


def _reset_mamba_imports() -> None:
    import sys

    for name in list(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            del sys.modules[name]


def _device_report(requested_gpu: str) -> dict[str, Any]:
    import torch

    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "requested_gpu_spec": requested_gpu,
        "image_ref": GHCR_REF,
    }


def _probe_patch_sites() -> list[dict[str, Any]]:
    _install_source_paths()
    _reset_mamba_imports()
    os.environ["MAMBA_SSM_SOURCE_DIR"] = SOURCE_ROOT

    from dataclasses import asdict

    from cppmega.megatron.upstream_patches.apply_mamba3_p2_psiv_patches import (
        probe_all_candidate_roots,
    )

    return [asdict(result) for result in probe_all_candidate_roots()]


def _time_cuda(fn, *, warmup: int, iters: int) -> dict[str, float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return {
        "mean_ms": sum(timings) / len(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "iters": float(iters),
    }


def _benchmark_psiv_costs() -> list[dict[str, Any]]:
    import torch

    _install_source_paths()
    from cppmega.megatron.mamba3_psiv_cache import estimate_cache_bytes, precompute_psi_v

    shapes = [
        {
            "name": "design_b1_s8192_h16_r4_p64",
            "B": 1,
            "S": 8192,
            "H": 16,
            "R": 4,
            "P": 64,
            "warmup": 5,
            "iters": 30,
        },
        {
            "name": "prod_mbs4_s4096_h32_r4_p128",
            "B": 4,
            "S": 4096,
            "H": 32,
            "R": 4,
            "P": 128,
            "warmup": 4,
            "iters": 20,
        },
        {
            "name": "stress_mbs10_s4096_h32_r4_p128",
            "B": 10,
            "S": 4096,
            "H": 32,
            "R": 4,
            "P": 128,
            "warmup": 2,
            "iters": 10,
        },
    ]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    out: list[dict[str, Any]] = []

    for shape in shapes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(1234)
        B, S, H, R, P = shape["B"], shape["S"], shape["H"], shape["R"], shape["P"]
        V = torch.randn(B, S, H, P, device=device, dtype=dtype)
        mimo_v = torch.randn(H, R, P, device=device, dtype=torch.float32)
        psi_bf16 = mimo_v.to(dtype=dtype)
        cache = torch.empty((B, S, H, R, P), device=device, dtype=dtype)

        def alloc_cast():
            result = precompute_psi_v(V, mimo_v)
            return result

        def out_precast():
            torch.mul(V.unsqueeze(3), psi_bf16.unsqueeze(0).unsqueeze(0), out=cache)
            return cache

        alloc_stats = _time_cuda(alloc_cast, warmup=shape["warmup"], iters=shape["iters"])
        out_stats = _time_cuda(out_precast, warmup=shape["warmup"], iters=shape["iters"])
        torch.cuda.synchronize()

        cache_bytes = estimate_cache_bytes(B, S, H, R, P, dtype)
        result = {
            "name": shape["name"],
            "shape": {"B": B, "S": S, "H": H, "R": R, "P": P, "dtype": str(dtype)},
            "cache_bytes": cache_bytes,
            "cache_gib": cache_bytes / (1024**3),
            "alloc_cast": alloc_stats,
            "out_precast": out_stats,
            "out_precast_effective_write_gib_s": (cache_bytes / (1024**3)) / (out_stats["mean_ms"] / 1000.0),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        }
        out.append(result)
        del V, mimo_v, psi_bf16, cache
        torch.cuda.empty_cache()

    return out


def _apply_text_replacements(path: str, replacements: list[tuple[str, str]]) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    counts: dict[str, int] = {}
    for index, (old, new) in enumerate(replacements):
        count = text.count(old)
        counts[f"replacement_{index}"] = count
        if count:
            text = text.replace(old, new)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    counts["remaining_percent_R"] = text.count("% R")
    counts["remaining_floor_div_R"] = text.count("// R")
    return counts


def _prepare_floor_mod_module(*, no_floormod: bool) -> tuple[str, dict[str, Any]]:
    import shutil
    import subprocess
    import tempfile

    src = f"{SOURCE_ROOT}/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"
    patch_file = (
        f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz/"
        "mamba3_bwd_layout_fix.patch"
    )
    work = tempfile.mkdtemp(prefix="cppmega_floormod_")
    dst = os.path.join(work, "mamba3_mimo_bwd.py")
    shutil.copy(src, dst)

    with open(patch_file, "rb") as handle:
        patch_bytes = handle.read()
    proc = subprocess.run(
        ["patch", "-p4", dst],
        input=patch_bytes,
        capture_output=True,
        cwd=work,
        check=False,
    )
    meta: dict[str, Any] = {
        "work": work,
        "patch_rc": proc.returncode,
        "patch_stdout_tail": proc.stdout.decode(errors="replace")[-2000:],
        "patch_stderr_tail": proc.stderr.decode(errors="replace")[-2000:],
        "no_floormod": no_floormod,
    }
    if proc.returncode != 0:
        return dst, meta

    replacements = [
        (
            "tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,",
            "tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,",
        ),
        (
            "tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,",
            "tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,",
        ),
    ]
    if no_floormod:
        replacements.extend(
            [
                (
                    "for csr, n in T.Parallel(fused_chunk_size, N):\n"
                    "                    q_frag[csr, n] += q_bias_frag[csr % R, n]",
                    "for cs, r, n in T.Parallel(chunk_size, R, N):\n"
                    "                    q_frag[cs * R + r, n] += q_bias_frag[r, n]",
                ),
                (
                    "for csr, n in T.Parallel(fused_chunk_size, N):\n"
                    "                    k_frag[csr, n] += k_bias_frag[csr % R, n]",
                    "for cs, r, n in T.Parallel(chunk_size, R, N):\n"
                    "                    k_frag[cs * R + r, n] += k_bias_frag[r, n]",
                ),
                (
                    "for csr, p in T.Parallel(fused_chunk_size, P):\n"
                    "                    cs = csr // R\n"
                    "                    r_in = csr % R\n"
                    "                    for r_out in T.serial(R):\n"
                    "                        csr_out = cs * R + r_out\n"
                    "                        dPsiV_D_fused_frag[csr, p] += dPhiO_shared[csr_out, p] * qk_dot_frag[cs, r_out * R + r_in] * gamma_dPsiV_frag[cs]",
                    "for cs, r_in, p in T.Parallel(chunk_size, R, P):\n"
                    "                    csr = cs * R + r_in\n"
                    "                    for r_out in T.serial(R):\n"
                    "                        csr_out = cs * R + r_out\n"
                    "                        dPsiV_D_fused_frag[csr, p] += dPhiO_shared[csr_out, p] * qk_dot_frag[cs, r_out * R + r_in] * gamma_dPsiV_frag[cs]",
                ),
            ]
        )
    meta["replacement_counts"] = _apply_text_replacements(dst, replacements)
    return dst, meta


def _compile_floor_mod_variant(*, no_floormod: bool) -> dict[str, Any]:
    import importlib.util
    import sys
    import textwrap
    import time
    import traceback

    _install_source_paths()
    _reset_mamba_imports()
    import mamba_ssm.ops.tilelang.mamba3  # noqa: F401

    path, meta = _prepare_floor_mod_module(no_floormod=no_floormod)
    if meta.get("patch_rc") != 0:
        meta["status"] = "patch_failed"
        return meta

    name = f"cppmega_mamba3_bwd_floormod_{'no' if no_floormod else 'baseline'}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        meta["status"] = "import_spec_failed"
        return meta
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    t0 = time.time()
    try:
        spec.loader.exec_module(mod)
        kernel = mod.mamba_mimo_bwd_bwd(
            1,
            64,
            4,
            1,
            64,
            64,
            4,
            False,
            False,
            True,
            16,
            4,
            "bfloat16",
            256,
            0,
        )
        if hasattr(kernel, "get_kernel_source"):
            source = kernel.get_kernel_source()
            meta["kernel_source_chars"] = len(source)
        meta["status"] = "compiled"
    except Exception as exc:  # noqa: BLE001
        combined = (str(exc) + "\n" + traceback.format_exc()).lower()
        meta["status"] = "crashed"
        meta["exception_type"] = type(exc).__name__
        meta["exception_short"] = textwrap.shorten(str(exc), width=800)
        meta["is_floormod_dbz"] = (
            "divide by zero" in combined
            and ("floormod" in combined or "layoutinference" in combined or "tryconstfold" in combined)
        )
        meta["is_tma_inputdim"] = "inputdim() == 2" in combined or "cannot detect tma layout" in combined
    finally:
        meta["elapsed_sec"] = round(time.time() - t0, 3)
    return meta


def _benchmark_tilelang_bwd_split() -> dict[str, Any]:
    import math
    import textwrap
    import time
    import traceback

    import torch

    _install_source_paths()
    _reset_mamba_imports()
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import (
        mamba_mimo_bwd_bwd,
        mamba_mimo_bwd_fwd,
    )
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, S, H, G, N, P, R = 2, 1024, 16, 1, 64, 64, 4
    chunk_size = 16
    rotary_dim_divisor = 4
    nchunks = math.ceil(S / chunk_size)
    result: dict[str, Any] = {
        "shape": {"B": B, "S": S, "H": H, "G": G, "N": N, "P": P, "R": R, "chunk": chunk_size},
    }
    t0 = time.time()
    try:
        torch.manual_seed(42)
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
        trap = (torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.5).to(dtype)
        adt = -torch.abs(torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.1)
        DA_CS, DA_CS_REV, SEGSUM = compute_dacs_segsum_triton(adt, chunk_size)
        z_dummy = torch.zeros(B, S, H, P, dtype=dtype, device=device)
        dz_dummy = torch.zeros(B, S, H, P, dtype=dtype, device=device)
        dmimo_z_dummy = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
        mimo_z_dummy = torch.zeros(H, R, P, dtype=torch.float32, device=device)
        D_dummy = torch.zeros(H, dtype=torch.float32, device=device)

        bf_kernel = mamba_mimo_bwd_fwd(
            B, S, H, G, N, P, R, False, False, True, chunk_size, rotary_dim_divisor, "bfloat16"
        )
        bb_kernel = mamba_mimo_bwd_bwd(
            B, S, H, G, N, P, R, False, False, True, chunk_size, rotary_dim_divisor, "bfloat16"
        )

        dmimo_o = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
        states = torch.zeros(B, H, nchunks, N, P, dtype=dtype, device=device)
        qk_dot = torch.zeros(B, H, S, R, R, dtype=dtype, device=device)
        dk = torch.zeros(B, S * R, H, N, dtype=dtype, device=device)
        dv = torch.zeros(B, S, H, P, dtype=dtype, device=device)
        dmimo_v = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
        dq = torch.zeros(B, S * R, H, N, dtype=dtype, device=device)
        dfactor = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dgamma_diag = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dangles = torch.zeros(B, S, H, N // rotary_dim_divisor, dtype=angles.dtype, device=device)
        dD = torch.zeros(B, H, dtype=torch.float32, device=device)
        dda = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dssda = torch.zeros(B, H, nchunks, chunk_size, chunk_size, dtype=torch.float32, device=device)
        dda_cs_rev = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dda_cs = torch.zeros(B, H, S, dtype=torch.float32, device=device)

        def bwd_fwd_fn():
            bf_kernel(
                DOUT,
                Q,
                K,
                V,
                Q_bias,
                K_bias,
                MIMO_V,
                MIMO_O,
                dmimo_o,
                states,
                z_dummy,
                mimo_z_dummy,
                dz_dummy,
                dmimo_z_dummy,
                angles,
                DA_CS,
                DA_CS_REV,
                dt,
                trap,
                D_dummy,
                qk_dot,
                SEGSUM,
            )

        def bwd_bwd_fn():
            bb_kernel(
                DOUT,
                Q,
                K,
                V,
                Q_bias,
                K_bias,
                MIMO_V,
                MIMO_O,
                dk,
                dv,
                dmimo_v,
                states,
                dq,
                z_dummy,
                mimo_z_dummy,
                angles,
                DA_CS,
                DA_CS_REV,
                dt,
                trap,
                dfactor,
                dgamma_diag,
                dangles,
                D_dummy,
                dD,
                qk_dot,
                dda,
                dssda,
                dda_cs_rev,
                dda_cs,
                SEGSUM,
            )

        bwd_fwd_fn()
        bwd_bwd_fn()
        torch.cuda.synchronize()
        bf = _time_cuda(bwd_fwd_fn, warmup=3, iters=20)
        bb = _time_cuda(bwd_bwd_fn, warmup=3, iters=20)
        result.update(
            {
                "status": "ok",
                "bwd_fwd": bf,
                "bwd_bwd": bb,
                "chain_mean_ms": bf["mean_ms"] + bb["mean_ms"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "status": "crashed",
                "exception_type": type(exc).__name__,
                "exception_short": textwrap.shorten(str(exc), width=800),
                "traceback_tail": traceback.format_exc()[-2000:],
            }
        )
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 3)
    return result


@app.function(image=_image(), gpu=GPU_SPEC, timeout=2400)
def run_probe(requested_gpu: str) -> dict[str, Any]:
    _install_source_paths()
    report = {
        "device": _device_report(requested_gpu),
        "patch_sites": _probe_patch_sites(),
        "psiv_costs": _benchmark_psiv_costs(),
        "tilelang_bwd_split": _benchmark_tilelang_bwd_split(),
        "floormod_layout_patch": _compile_floor_mod_variant(no_floormod=False),
        "floormod_no_floormod_rewrite": _compile_floor_mod_variant(no_floormod=True),
    }
    return report


@app.local_entrypoint()
def main() -> None:
    result = run_probe.remote(GPU_SPEC)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
