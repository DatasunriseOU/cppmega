"""Modal Hopper probe for Mamba3 MIMO backward TMA/warp-specialized path.

The probe is intentionally temp-only:
  * pulls the prebuilt cppmega image from GHCR,
  * overlays local cppmega plus a source checkout of state-spaces/mamba,
  * copies ``mamba3_mimo_bwd.py`` to a temporary directory in the container,
  * applies the existing 3D->2D TMA layout patch and selected source rewrites,
  * compiles and, if possible, smoke-runs bwd_fwd + bwd_bwd on real Hopper GPUs.

Run examples:

    CPPMEGA_MODAL_GPU=H100:2 modal run scripts/modal_mamba3_hopper_tma_probe.py
    CPPMEGA_MODAL_GPU=H200:2 modal run scripts/modal_mamba3_hopper_tma_probe.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "latest")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")

APP_NAME = "cppmega-mamba3-hopper-tma-probe"
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
    if variant in {"no_floormod", "qk_serial_p", "qk_shared_direct"}:
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
    return replacements


def _prepare_variant(variant: str) -> tuple[str, dict[str, Any]]:
    import shutil
    import subprocess
    import tempfile

    src = f"{SOURCE_ROOT}/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"
    patch_file = (
        f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz/"
        "mamba3_bwd_layout_fix.patch"
    )
    work = tempfile.mkdtemp(prefix=f"cppmega_mamba3_tma_{variant}_")
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
        "variant": variant,
        "work": work,
        "patch_rc": proc.returncode,
        "patch_stdout_tail": proc.stdout.decode(errors="replace")[-2000:],
        "patch_stderr_tail": proc.stderr.decode(errors="replace")[-2000:],
    }
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


def _compile_variant(variant: str) -> dict[str, Any]:
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
            1, 64, 4, 1, 64, 64, 4, False, False, True, 16, 4, "bfloat16"
        )
        bb_kernel = mod.mamba_mimo_bwd_bwd(
            1, 64, 4, 1, 64, 64, 4, False, False, True, 16, 4, "bfloat16", 256, 0
        )
        meta["bwd_fwd_source_chars"] = len(bf_kernel.get_kernel_source()) if hasattr(bf_kernel, "get_kernel_source") else None
        meta["bwd_bwd_source_chars"] = len(bb_kernel.get_kernel_source()) if hasattr(bb_kernel, "get_kernel_source") else None
        meta["status"] = "compiled"
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "crashed"
        meta.update(_classify_exception(exc, traceback.format_exc()))
    finally:
        meta["elapsed_sec"] = round(time.time() - t0, 3)
    return meta


def _smoke_variant(variant: str) -> dict[str, Any]:
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
        B, S, H, G, N, P, R = 1, 64, 4, 1, 64, 64, 4
        chunk_size = 16
        rotary_dim_divisor = 4
        nchunks = math.ceil(S / chunk_size)

        torch.manual_seed(123)
        q = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.01
        k = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.01
        q_flat = q.view(B, S * R, G, N)
        k_flat = k.view(B, S * R, G, N)
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

        dmimo_o = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
        states = torch.zeros(B, H, nchunks, N, P, dtype=dtype, device=device)
        qk_dot = torch.zeros(B, H, S, R * R, dtype=dtype, device=device)
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
            B, S, H, G, N, P, R, False, False, True, chunk_size, rotary_dim_divisor, "bfloat16"
        )
        bb_kernel = mod.mamba_mimo_bwd_bwd(
            B, S, H, G, N, P, R, False, False, True, chunk_size, rotary_dim_divisor, "bfloat16", 256, 0
        )

        bf_kernel(
            dout, q_flat, k_flat, v, q_bias, k_bias, mimo_v, mimo_o, dmimo_o,
            states, z, mimo_z, dz, dmimo_z, angles, da_cs, da_cs_rev, dt, trap,
            d, qk_dot, segsum,
        )
        bb_kernel(
            dout, q_flat, k_flat, v, q_bias, k_bias, mimo_v, mimo_o, dk, dv,
            dmimo_v, states, dq, z, mimo_z, angles, da_cs, da_cs_rev, dt, trap,
            dfactor, dgamma_diag, dangles, d, dd, qk_dot, dda, dssda,
            dda_cs_rev, dda_cs, segsum,
        )
        torch.cuda.synchronize()
        meta.update(
            {
                "status": "smoke_ok",
                "qk_dot_absmax": float(qk_dot.abs().max().item()),
                "dq_absmax": float(dq.abs().max().item()),
                "dk_absmax": float(dk.abs().max().item()),
                "dv_absmax": float(dv.abs().max().item()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "crashed"
        meta.update(_classify_exception(exc, traceback.format_exc()))
    finally:
        meta["elapsed_sec"] = round(time.time() - t0, 3)
    return meta


@app.function(image=_image(), gpu=GPU_SPEC, timeout=2400)
def run_probe(requested_gpu: str) -> dict[str, Any]:
    variants = ["layout_patch", "no_floormod", "qk_serial_p", "qk_shared_direct"]
    compile_results = [_compile_variant(variant) for variant in variants]
    smoke_results = []
    for result in compile_results:
        if result.get("status") == "compiled":
            smoke_results.append(_smoke_variant(str(result["variant"])))
    return {
        "device": _device_report(requested_gpu),
        "tilelang": _tilelang_report(),
        "compile": compile_results,
        "smoke": smoke_results,
    }


@app.local_entrypoint()
def main() -> None:
    result = run_probe.remote(GPU_SPEC)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
