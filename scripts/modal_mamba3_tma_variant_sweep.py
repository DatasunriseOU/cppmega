"""Modal Hopper sweep for Mamba3 MIMO backward TMA variants.

The probe is intentionally temp-only:
  * pulls the prebuilt cppmega image from GHCR,
  * overlays local cppmega plus a source checkout of state-spaces/mamba,
  * copies ``mamba3_mimo_bwd.py`` to a temporary directory in the container,
  * applies selected 3D->2D TMA and qk_dot source rewrites,
  * compiles and smoke/bench-runs bwd_fwd + bwd_bwd on real Hopper GPUs.

Run examples:

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100:2 modal run scripts/modal_mamba3_tma_variant_sweep.py
    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 modal run scripts/modal_mamba3_tma_variant_sweep.py

Optional:

    CPPMEGA_TMA_VARIANTS=qk_direct,qk_recompute
    CPPMEGA_TMA_BENCH_ITERS=20
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

APP_NAME = "cppmega-mamba3-tma-variant-sweep"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"

LAYOUT_PATCH = "mamba3_bwd_layout_fix.patch"
DIRECT_PATCH = "mamba3_bwd_hopper_tma_ws_fix.patch"

DEFAULT_VARIANTS = [
    "baseline_notma",
    "layout_patch_tma_ws",
    "qk_serial_p",
    "qk_direct",
    "qk_direct_smem_bias",
    "qk_recompute",
    "qk_dot_rs_layout",
]

DEFAULT_CONFIGS = [
    {"name": "bf128_bb256_s0", "bf_threads": 128, "bf_num_stages": 0, "bb_threads": 256, "bb_num_stages": 0},
    {"name": "bf128_bb128_s0", "bf_threads": 128, "bf_num_stages": 0, "bb_threads": 128, "bb_num_stages": 0},
    {"name": "bf128_bb512_s0", "bf_threads": 128, "bf_num_stages": 0, "bb_threads": 512, "bb_num_stages": 0},
    {"name": "bf128_bb256_s1", "bf_threads": 128, "bf_num_stages": 0, "bb_threads": 256, "bb_num_stages": 1},
    {"name": "bf128_bb256_s2", "bf_threads": 128, "bf_num_stages": 0, "bb_threads": 256, "bb_num_stages": 2},
]

VARIANT_CONFIGS = {
    "baseline_notma": ["bf128_bb256_s0"],
    "layout_patch_tma_ws": ["bf128_bb256_s0"],
    "qk_serial_p": ["bf128_bb256_s0"],
    "qk_direct": [config["name"] for config in DEFAULT_CONFIGS],
    "qk_direct_smem_bias": ["bf128_bb256_s0"],
    "qk_recompute": ["bf128_bb256_s0"],
    "qk_dot_rs_layout": ["bf128_bb256_s0"],
}


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


def _tilelang_report() -> dict[str, Any]:
    import importlib.metadata
    import os
    import subprocess

    import tilelang

    report: dict[str, Any] = {
        "module_file": getattr(tilelang, "__file__", None),
        "module_version": getattr(tilelang, "__version__", None),
    }
    try:
        report["package_version"] = importlib.metadata.version("tilelang")
    except importlib.metadata.PackageNotFoundError:
        report["package_version"] = None

    module_file = report["module_file"]
    if isinstance(module_file, str):
        probe_dir = os.path.dirname(os.path.abspath(module_file))
        proc = subprocess.run(
            ["git", "-C", probe_dir, "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        report["git_head"] = proc.stdout.strip() if proc.returncode == 0 else None
    return report


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
    counts["qk_serial_p"] = text.count("for p in T.serial(P):")
    return counts


def _variant_replacements(variant: str) -> list[tuple[str, str]]:
    if variant == "baseline_notma":
        return []

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
    if variant in {"no_floormod", "layout_patch_tma_ws", "qk_serial_p", "qk_shared_direct"}:
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
    if variant == "qk_serial_p":
        replacements.append(
            (
                "for cs, r_in, p in T.Parallel(chunk_size, R, P):\n"
                "                    csr = cs * R + r_in\n"
                "                    for r_out in T.serial(R):\n"
                "                        csr_out = cs * R + r_out\n"
                "                        dPsiV_D_fused_frag[csr, p] += dPhiO_shared[csr_out, p] * qk_dot_frag[cs, r_out * R + r_in] * gamma_dPsiV_frag[cs]",
                "for cs, r_in in T.Parallel(chunk_size, R):\n"
                "                    csr = cs * R + r_in\n"
                "                    for p in T.serial(P):\n"
                "                        for r_out in T.serial(R):\n"
                "                            csr_out = cs * R + r_out\n"
                "                            dPsiV_D_fused_frag[csr, p] += dPhiO_shared[csr_out, p] * qk_dot_frag[cs, r_out * R + r_in] * gamma_dPsiV_frag[cs]",
            )
        )
    if variant == "qk_shared_direct":
        replacements.extend(
            [
                (
                    "                qk_dot_frag = T.alloc_fragment([chunk_size, R * R], dtype)\n",
                    "",
                ),
                (
                    "                T.copy(qk_dot_shared, qk_dot_frag)\n",
                    "",
                ),
                (
                    "qk_dot_frag[cs, r_out * R + r_in]",
                    "qk_dot_shared[cs, r_out * R + r_in]",
                ),
            ]
        )
    if variant == "qk_direct_smem_bias":
        replacements.extend(
            [
                (
                    "                q_frag = T.alloc_fragment([fused_chunk_size, N], dtype)\n"
                    "                T.copy(q_shared, q_frag)\n"
                    "                for cs, r, n in T.Parallel(chunk_size, R, N):\n"
                    "                    q_frag[cs * R + r, n] += q_bias_frag[r, n]\n"
                    "                T.copy(q_frag, q_shared)\n",
                    "                for cs, r, n in T.Parallel(chunk_size, R, N):\n"
                    "                    q_shared[cs * R + r, n] += q_bias_frag[r, n]\n",
                ),
                (
                    "                k_frag = T.alloc_fragment([fused_chunk_size, N], dtype)\n"
                    "                T.copy(k_shared, k_frag)\n"
                    "                for cs, r, n in T.Parallel(chunk_size, R, N):\n"
                    "                    k_frag[cs * R + r, n] += k_bias_frag[r, n]\n"
                    "                T.copy(k_frag, k_shared)\n",
                    "                for cs, r, n in T.Parallel(chunk_size, R, N):\n"
                    "                    k_shared[cs * R + r, n] += k_bias_frag[r, n]\n",
                ),
                (
                    "                q_frag = T.alloc_fragment([fused_chunk_size, N], dtype)\n"
                    "                T.copy(q_shared, q_frag)\n"
                    "                for cs, r, n in T.Parallel(chunk_size, R, N):\n"
                    "                    q_frag[cs * R + r, n] += q_bias_frag[r, n]\n"
                    "                T.copy(q_frag, q_shared)\n"
                    "                T.copy(q_shared, q_pre_rot_shared) # Save pre-rotated q for later:\n",
                    "                for cs, r, n in T.Parallel(chunk_size, R, N):\n"
                    "                    q_shared[cs * R + r, n] += q_bias_frag[r, n]\n"
                    "                T.copy(q_shared, q_pre_rot_shared) # Save pre-rotated q for later:\n",
                ),
                (
                    "                k_frag = T.alloc_fragment([fused_chunk_size, N], dtype)\n"
                    "                T.copy(k_pre_trap_shared, k_frag)\n"
                    "                for cs, r, n in T.Parallel(chunk_size, R, N):\n"
                    "                    k_frag[cs * R + r, n] += k_bias_frag[r, n]\n"
                    "                T.copy(k_frag, k_pre_trap_shared)\n",
                    "                for cs, r, n in T.Parallel(chunk_size, R, N):\n"
                    "                    k_pre_trap_shared[cs * R + r, n] += k_bias_frag[r, n]\n",
                ),
            ]
        )
    if variant == "qk_recompute":
        replacements.extend(
            [
                (
                    "                # Output QK_DOT for the bwd_bwd kernel (per-time-step blocks only).\n"
                    "                # TMA-fix: QK_DOT is now [B, H, S, R*R]; pack (r_out, r_in) into the last dim.\n"
                    "                for cs, r_out, r_in in T.Parallel(chunk_size, R, R):\n"
                    "                    QK_DOT[i_b, i_h, chunk_start + cs, r_out * R + r_in] = \\\n"
                    "                        qk_dot_full_shared[cs * R + r_out, cs * R + r_in]\n",
                    "                # Variant: bwd_bwd recomputes diagonal qk_dot, so skip the global QK_DOT store.\n",
                ),
                (
                    "                # TMA-fix: qk_dot_frag flattened to [chunk_size, R*R]; load from\n"
                    "                # [B, H, S, R*R] gmem into [chunk_size, R*R] smem (rank-2 TMA legal).\n"
                    "                T.copy(QK_DOT[i_b, i_h, chunk_start:chunk_start+chunk_size, :], qk_dot_shared)\n",
                    "                # Variant: recompute the per-step R x R diagonal qk_dot blocks from\n"
                    "                # pre-rotary Q/K instead of loading the bwd_fwd cache from global memory.\n"
                    "                qk_dot_full_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], accum_dtype)\n"
                    "                T.gemm(q_pre_rot_shared, k_pre_rot_shared, qk_dot_full_frag, transpose_B=True, clear_accum=True)\n"
                    "                for cs, r_out, r_in in T.Parallel(chunk_size, R, R):\n"
                    "                    qk_dot_shared[cs, r_out * R + r_in] = qk_dot_full_frag[cs * R + r_out, cs * R + r_in]\n",
                ),
            ]
        )
    if variant == "qk_dot_rs_layout":
        replacements.extend(
            [
                (
                    "QK_DOT: T.Tensor([B, H, S, R * R], dtype)",
                    "QK_DOT: T.Tensor([B, H, R * R, S], dtype)",
                ),
                (
                    "QK_DOT is now [B, H, S, R*R]",
                    "QK_DOT is now [B, H, R*R, S]",
                ),
                (
                    "QK_DOT[i_b, i_h, chunk_start + cs, r_out * R + r_in]",
                    "QK_DOT[i_b, i_h, r_out * R + r_in, chunk_start + cs]",
                ),
                (
                    "qk_dot_shared = T.alloc_shared([chunk_size, R * R], dtype)",
                    "qk_dot_shared = T.alloc_shared([R * R, chunk_size], dtype)",
                ),
                (
                    "T.copy(QK_DOT[i_b, i_h, chunk_start:chunk_start+chunk_size, :], qk_dot_shared)",
                    "T.copy(QK_DOT[i_b, i_h, :, chunk_start:chunk_start+chunk_size], qk_dot_shared)",
                ),
                (
                    "qk_dot_shared[cs, r_out * R + r_in]",
                    "qk_dot_shared[r_out * R + r_in, cs]",
                ),
                (
                    "                T.copy(qk_dot_shared, dgamma_diag_prereduce_frag)\n"
                    "                T.copy(dqk_from_diag_frag, dqk_from_diag_shared)\n",
                    "                for cs, r_out, r_in in T.Parallel(chunk_size, R, R):\n"
                    "                    dgamma_diag_prereduce_frag[cs, r_out * R + r_in] = qk_dot_shared[r_out * R + r_in, cs]\n"
                    "                T.copy(dqk_from_diag_frag, dqk_from_diag_shared)\n",
                ),
                (
                    "qk_dot = torch.zeros([B, H, S, R * R], dtype=q.dtype, device=q.device)",
                    "qk_dot = torch.zeros([B, H, R * R, S], dtype=q.dtype, device=q.device)",
                ),
            ]
        )
    return replacements


def _prepare_variant(variant: str) -> tuple[str, dict[str, Any]]:
    import shutil
    import subprocess
    import tempfile

    src = f"{SOURCE_ROOT}/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"
    work = tempfile.mkdtemp(prefix=f"cppmega_mamba3_tma_{variant}_")
    dst = os.path.join(work, "mamba3_mimo_bwd.py")
    shutil.copy(src, dst)

    patch_name: str | None
    if variant == "baseline_notma":
        patch_name = None
    elif variant in {"qk_direct", "qk_direct_smem_bias", "qk_recompute", "qk_dot_rs_layout"}:
        patch_name = DIRECT_PATCH
    else:
        patch_name = LAYOUT_PATCH

    meta: dict[str, Any] = {
        "variant": variant,
        "work": work,
        "patch": patch_name,
    }

    if patch_name is None:
        meta.update({"patch_rc": 0, "patch_stdout_tail": "", "patch_stderr_tail": ""})
    else:
        patch_file = (
            f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz/"
            f"{patch_name}"
        )
        with open(patch_file, "rb") as handle:
            patch_bytes = handle.read()
        proc = subprocess.run(
            ["patch", "-p4", dst],
            input=patch_bytes,
            capture_output=True,
            cwd=work,
            check=False,
        )
        meta.update(
            {
                "patch_rc": proc.returncode,
                "patch_stdout_tail": proc.stdout.decode(errors="replace")[-2000:],
                "patch_stderr_tail": proc.stderr.decode(errors="replace")[-2000:],
            }
        )
        if proc.returncode != 0:
            return dst, meta

    meta["replacement_counts"] = _apply_text_replacements(dst, _variant_replacements(variant))
    return dst, meta


def _classify_exception(exc: BaseException, traceback_text: str) -> dict[str, Any]:
    import textwrap

    combined = (str(exc) + "\n" + traceback_text).lower()
    return {
        "exception_type": type(exc).__name__,
        "exception_short": textwrap.shorten(str(exc), width=1000),
        "traceback_tail": traceback_text[-5000:],
        "is_floormod_dbz": (
            "divide by zero" in combined
            and ("floormod" in combined or "layoutinference" in combined or "tryconstfold" in combined)
        ),
        "is_loop_layout_injective": "loop layout is not injective" in combined,
        "mentions_qk_dot": "qk_dot" in combined,
        "is_tma_inputdim": "inputdim() == 2" in combined or "cannot detect tma layout" in combined,
    }


def _compile_variant(variant: str, config: dict[str, int | str]) -> dict[str, Any]:
    import importlib.util
    import sys
    import time
    import traceback

    _install_source_paths()
    _reset_mamba_imports()
    import mamba_ssm.ops.tilelang.mamba3  # noqa: F401

    path, meta = _prepare_variant(variant)
    if meta.get("patch_rc") != 0:
        meta["status"] = "patch_failed"
        return meta

    name = f"cppmega_mamba3_bwd_tma_{variant}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        meta["status"] = "import_spec_failed"
        return meta

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    t0 = time.time()
    try:
        spec.loader.exec_module(mod)
        bf_kernel = mod.mamba_mimo_bwd_fwd(
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
            int(config["bf_threads"]),
            int(config["bf_num_stages"]),
        )
        bb_kernel = mod.mamba_mimo_bwd_bwd(
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
            int(config["bb_threads"]),
            int(config["bb_num_stages"]),
        )
        meta["bwd_fwd_source_chars"] = len(bf_kernel.get_kernel_source()) if hasattr(bf_kernel, "get_kernel_source") else None
        meta["bwd_bwd_source_chars"] = len(bb_kernel.get_kernel_source()) if hasattr(bb_kernel, "get_kernel_source") else None
        meta["status"] = "compiled"
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "crashed"
        meta.update(_classify_exception(exc, traceback.format_exc()))
    finally:
        meta["elapsed_sec"] = round(time.time() - t0, 3)
        meta["config"] = config
    return meta


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


def _smoke_bench_variant(variant: str, config: dict[str, int | str]) -> dict[str, Any]:
    import importlib.util
    import math
    import sys
    import time
    import traceback

    import torch

    _install_source_paths()
    _reset_mamba_imports()
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

    path, meta = _prepare_variant(variant)
    if meta.get("patch_rc") != 0:
        meta["status"] = "patch_failed"
        return meta

    name = f"cppmega_mamba3_bwd_tma_smoke_{variant}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        meta["status"] = "import_spec_failed"
        return meta

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    t0 = time.time()
    try:
        spec.loader.exec_module(mod)

        device = torch.device("cuda")
        dtype = torch.bfloat16
        B = int(os.environ.get("CPPMEGA_TMA_BENCH_B", "1"))
        S = int(os.environ.get("CPPMEGA_TMA_BENCH_S", "256"))
        H = int(os.environ.get("CPPMEGA_TMA_BENCH_H", "8"))
        G = int(os.environ.get("CPPMEGA_TMA_BENCH_G", "1"))
        N = int(os.environ.get("CPPMEGA_TMA_BENCH_N", "64"))
        P = int(os.environ.get("CPPMEGA_TMA_BENCH_P", "64"))
        R = int(os.environ.get("CPPMEGA_TMA_BENCH_R", "4"))
        chunk_size = 16
        rotary_dim_divisor = 4
        nchunks = math.ceil(S / chunk_size)

        torch.manual_seed(123)
        q = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.01
        k = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.01
        v = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.01
        dout = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.01
        q_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.01
        k_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.01
        mimo_v = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.01
        mimo_o = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.01
        angles = torch.randn(B, S, H, N // rotary_dim_divisor, device=device, dtype=torch.float32) * 0.01
        dt = torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.01
        trap = torch.randn(B, H, S, device=device, dtype=dtype) * 0.01
        adt = -torch.abs(torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.01)
        da_cs, da_cs_rev, segsum = compute_dacs_segsum_triton(adt, chunk_size)

        z = torch.zeros(B, S, H, P, device=device, dtype=dtype)
        dz = torch.zeros(B, S, H, P, device=device, dtype=dtype)
        mimo_z = torch.zeros(H, R, P, device=device, dtype=torch.float32)
        dmimo_z = torch.zeros(B, H, R, P, device=device, dtype=torch.float32)
        d = torch.zeros(H, device=device, dtype=torch.float32)

        q_arg = q if variant == "baseline_notma" else q.view(B, S * R, G, N)
        k_arg = k if variant == "baseline_notma" else k.view(B, S * R, G, N)
        qk_shape = (B, H, S, R, R) if variant == "baseline_notma" else (B, H, S, R * R)
        if variant == "qk_dot_rs_layout":
            qk_shape = (B, H, R * R, S)

        dmimo_o = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
        states = torch.zeros(B, H, nchunks, N, P, dtype=dtype, device=device)
        qk_dot = torch.zeros(qk_shape, dtype=dtype, device=device)
        dk = torch.zeros(B, S * R, H, N, dtype=dtype, device=device)
        dv = torch.zeros(B, S, H, P, dtype=dtype, device=device)
        dmimo_v = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
        dq = torch.zeros(B, S * R, H, N, dtype=dtype, device=device)
        dfactor = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dgamma_diag = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dangles = torch.zeros(B, S, H, N // rotary_dim_divisor, dtype=torch.float32, device=device)
        dd = torch.zeros(B, H, dtype=torch.float32, device=device)
        dda = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dssda = torch.zeros(B, H, nchunks, chunk_size, chunk_size, dtype=torch.float32, device=device)
        dda_cs_rev = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dda_cs = torch.zeros(B, H, S, dtype=torch.float32, device=device)

        bf_kernel = mod.mamba_mimo_bwd_fwd(
            B,
            S,
            H,
            G,
            N,
            P,
            R,
            False,
            False,
            True,
            chunk_size,
            rotary_dim_divisor,
            "bfloat16",
            int(config["bf_threads"]),
            int(config["bf_num_stages"]),
        )
        bb_kernel = mod.mamba_mimo_bwd_bwd(
            B,
            S,
            H,
            G,
            N,
            P,
            R,
            False,
            False,
            True,
            chunk_size,
            rotary_dim_divisor,
            "bfloat16",
            int(config["bb_threads"]),
            int(config["bb_num_stages"]),
        )

        def run_once():
            bf_kernel(
                dout, q_arg, k_arg, v, q_bias, k_bias, mimo_v, mimo_o, dmimo_o,
                states, z, mimo_z, dz, dmimo_z, angles, da_cs, da_cs_rev, dt, trap,
                d, qk_dot, segsum,
            )
            bb_kernel(
                dout, q_arg, k_arg, v, q_bias, k_bias, mimo_v, mimo_o, dk, dv,
                dmimo_v, states, dq, z, mimo_z, angles, da_cs, da_cs_rev, dt, trap,
                dfactor, dgamma_diag, dangles, d, dd, qk_dot, dda, dssda,
                dda_cs_rev, dda_cs, segsum,
            )
            return dq, dk, dv

        run_t0 = time.time()
        outputs = run_once()
        torch.cuda.synchronize()
        single_call_wall_sec = round(time.time() - run_t0, 3)
        bench_iters = int(os.environ.get("CPPMEGA_TMA_BENCH_ITERS", "0"))
        if bench_iters > 0:
            bench = _time_cuda(
                run_once,
                warmup=int(os.environ.get("CPPMEGA_TMA_BENCH_WARMUP", "0")),
                iters=bench_iters,
            )
            bench["single_call_wall_sec"] = single_call_wall_sec
        else:
            bench = {"iters": 0.0, "single_call_wall_sec": single_call_wall_sec}
        finite = all(torch.isfinite(t).all().item() for t in outputs if isinstance(t, torch.Tensor))
        meta.update(
            {
                "status": "smoke_ok",
                "shape": {"B": B, "S": S, "H": H, "G": G, "N": N, "P": P, "R": R},
                "bench": bench,
                "all_outputs_finite": bool(finite),
                "dq_absmax": float(outputs[0].abs().max().item()),
                "dk_absmax": float(outputs[1].abs().max().item()),
                "dv_absmax": float(outputs[2].abs().max().item()),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "crashed"
        meta.update(_classify_exception(exc, traceback.format_exc()))
    finally:
        meta["elapsed_sec"] = round(time.time() - t0, 3)
        meta["config"] = config
    return meta


@app.function(image=_image(), gpu=GPU_SPEC, timeout=2400)
def run_probe(requested_gpu: str) -> dict[str, Any]:
    requested_variants = os.environ.get("CPPMEGA_TMA_VARIANTS")
    if requested_variants:
        variants = [item.strip() for item in requested_variants.split(",") if item.strip()]
    else:
        variants = DEFAULT_VARIANTS

    config_by_name = {str(config["name"]): config for config in DEFAULT_CONFIGS}
    requested_configs = os.environ.get("CPPMEGA_TMA_CONFIGS")
    config_filter = {item.strip() for item in requested_configs.split(",") if item.strip()} if requested_configs else None
    compile_results = []
    smoke_results = []
    for variant in variants:
        config_names = VARIANT_CONFIGS.get(variant, ["bf128_bb256_s0"])
        if config_filter is not None:
            config_names = [name for name in config_names if name in config_filter]
        for config_name in config_names:
            config = config_by_name[config_name]
            result = _compile_variant(variant, config)
            compile_results.append(result)
            if result.get("status") == "compiled":
                smoke_results.append(_smoke_bench_variant(variant, config))
    return {
        "device": _device_report(requested_gpu),
        "tilelang": _tilelang_report(),
        "variants": variants,
        "compile": compile_results,
        "smoke": smoke_results,
    }


@app.local_entrypoint()
def main() -> None:
    result = run_probe.remote(GPU_SPEC)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
