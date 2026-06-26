"""Modal Hopper perf/correctness check for Mamba3 MIMO bwd TMA workaround.

This harness is temp-only: it compares the installed/source non-TMA baseline
against a copied ``mamba3_mimo_bwd.py`` with the Hopper TMA layout patch plus
the ``qk_shared_direct`` workaround. It does not change production defaults.

Run examples:

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100:2 modal run scripts/modal_mamba3_hopper_tma_perf.py
    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 modal run scripts/modal_mamba3_hopper_tma_perf.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "latest")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")

APP_NAME = "cppmega-mamba3-hopper-tma-perf"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"


@dataclass(frozen=True)
class Shape:
    name: str
    B: int
    S: int
    H: int
    G: int
    N: int
    P: int
    R: int
    chunk_size: int = 16
    rotary_dim_divisor: int = 4
    enabled: bool = True


SHAPES = [
    Shape("smoke", B=1, S=64, H=4, G=1, N=64, P=64, R=4),
    Shape("representative", B=2, S=1024, H=16, G=1, N=64, P=64, R=4),
    Shape("productionish", B=4, S=4096, H=32, G=1, N=64, P=128, R=4),
]


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    img = img.env(
        {
            "GHCR_REPO": GHCR_REPO,
            "GHCR_TAG": GHCR_TAG,
            "CPPMEGA_IMAGE_REF": GHCR_REF,
        }
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

    image_ref = os.environ.get("CPPMEGA_IMAGE_REF", GHCR_REF)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "requested_gpu_spec": requested_gpu,
        "image_ref": image_ref,
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


def _qk_shared_direct_replacements() -> list[tuple[str, str]]:
    return [
        (
            "tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,",
            "tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,",
        ),
        (
            "tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,",
            "tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,",
        ),
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


def _prepare_module_source(variant: str) -> tuple[str, dict[str, Any]]:
    import shutil
    import subprocess
    import tempfile

    src = f"{SOURCE_ROOT}/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"
    work = tempfile.mkdtemp(prefix=f"cppmega_mamba3_tma_perf_{variant}_")
    dst = os.path.join(work, "mamba3_mimo_bwd.py")
    shutil.copy(src, dst)

    meta: dict[str, Any] = {"variant": variant, "work": work}
    if variant == "baseline_non_tma":
        return dst, meta

    patch_file = (
        f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz/"
        "mamba3_bwd_layout_fix.patch"
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
    if proc.returncode == 0:
        meta["replacement_counts"] = _apply_text_replacements(
            dst,
            _qk_shared_direct_replacements(),
        )
    return dst, meta


def _load_module(variant: str, shape_name: str) -> tuple[Any, dict[str, Any]]:
    import importlib.util
    import sys
    import time

    _install_source_paths()
    _reset_mamba_imports()
    import mamba_ssm.ops.tilelang.mamba3  # noqa: F401

    path, meta = _prepare_module_source(variant)
    if meta.get("patch_rc", 0) != 0:
        meta["status"] = "patch_failed"
        return None, meta

    name = f"cppmega_mamba3_tma_perf_{shape_name}_{variant}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        meta["status"] = "import_spec_failed"
        return None, meta

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    t0 = time.time()
    spec.loader.exec_module(module)
    meta["import_elapsed_sec"] = round(time.time() - t0, 3)
    meta["status"] = "imported"
    return module, meta


def _empty_outputs(shape: Shape, dtype: Any, device: Any, flat_qk_dot: bool) -> dict[str, Any]:
    import math
    import torch

    B, S, H, N, P, R = shape.B, shape.S, shape.H, shape.N, shape.P, shape.R
    nchunks = math.ceil(S / shape.chunk_size)
    qk_shape = (B, H, S, R * R) if flat_qk_dot else (B, H, S, R, R)
    return {
        "z": torch.zeros(B, S, H, P, device=device, dtype=dtype),
        "dz": torch.zeros(B, S, H, P, device=device, dtype=dtype),
        "mimo_z": torch.zeros(H, R, P, device=device, dtype=torch.float32),
        "dmimo_z": torch.zeros(B, H, R, P, device=device, dtype=torch.float32),
        "dmimo_o": torch.zeros(B, H, R, P, device=device, dtype=torch.float32),
        "states": torch.zeros(B, H, nchunks, N, P, device=device, dtype=dtype),
        "qk_dot": torch.zeros(*qk_shape, device=device, dtype=dtype),
        "dk": torch.zeros(B, S * R, H, N, device=device, dtype=dtype),
        "dv": torch.zeros(B, S, H, P, device=device, dtype=dtype),
        "dmimo_v": torch.zeros(B, H, R, P, device=device, dtype=torch.float32),
        "dq": torch.zeros(B, S * R, H, N, device=device, dtype=dtype),
        "dfactor": torch.zeros(B, H, S, device=device, dtype=torch.float32),
        "dgamma_diag": torch.zeros(B, H, S, device=device, dtype=torch.float32),
        "dangles": torch.zeros(B, S, H, N // shape.rotary_dim_divisor, device=device, dtype=torch.float32),
        "dd": torch.zeros(B, H, device=device, dtype=torch.float32),
        "dda": torch.zeros(B, H, S, device=device, dtype=torch.float32),
        "dssda": torch.zeros(B, H, nchunks, shape.chunk_size, shape.chunk_size, device=device, dtype=torch.float32),
        "dda_cs_rev": torch.zeros(B, H, S, device=device, dtype=torch.float32),
        "dda_cs": torch.zeros(B, H, S, device=device, dtype=torch.float32),
    }


def _make_inputs(shape: Shape) -> dict[str, Any]:
    import torch
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, S, H, G, N, P, R = shape.B, shape.S, shape.H, shape.G, shape.N, shape.P, shape.R

    torch.manual_seed(123)
    q = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.01
    k = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.01
    v = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.01
    dout = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.01
    q_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.01
    k_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.01
    mimo_v = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.01
    mimo_o = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.01
    angles = torch.randn(B, S, H, N // shape.rotary_dim_divisor, device=device, dtype=torch.float32) * 0.01
    dt = torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.01
    trap = torch.randn(B, H, S, device=device, dtype=dtype) * 0.01
    adt = -torch.abs(torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.01)
    da_cs, da_cs_rev, segsum = compute_dacs_segsum_triton(adt, shape.chunk_size)
    d = torch.zeros(H, device=device, dtype=torch.float32)

    return {
        "q": q,
        "k": k,
        "q_flat": q.view(B, S * R, G, N),
        "k_flat": k.view(B, S * R, G, N),
        "v": v,
        "dout": dout,
        "q_bias": q_bias,
        "k_bias": k_bias,
        "mimo_v": mimo_v,
        "mimo_o": mimo_o,
        "angles": angles,
        "da_cs": da_cs,
        "da_cs_rev": da_cs_rev,
        "dt": dt,
        "trap": trap,
        "d": d,
        "segsum": segsum,
    }


def _compile_kernels(module: Any, shape: Shape) -> tuple[Any, Any, dict[str, Any]]:
    import time

    B, S, H, G, N, P, R = shape.B, shape.S, shape.H, shape.G, shape.N, shape.P, shape.R
    t0 = time.time()
    bf_kernel = module.mamba_mimo_bwd_fwd(
        B, S, H, G, N, P, R, False, False, True, shape.chunk_size, shape.rotary_dim_divisor, "bfloat16"
    )
    bf_elapsed = time.time() - t0
    t1 = time.time()
    bb_kernel = module.mamba_mimo_bwd_bwd(
        B, S, H, G, N, P, R, False, False, True, shape.chunk_size, shape.rotary_dim_divisor, "bfloat16", 256, 0
    )
    bb_elapsed = time.time() - t1
    return bf_kernel, bb_kernel, {
        "bwd_fwd_compile_sec": round(bf_elapsed, 3),
        "bwd_bwd_compile_sec": round(bb_elapsed, 3),
        "bwd_fwd_source_chars": len(bf_kernel.get_kernel_source()) if hasattr(bf_kernel, "get_kernel_source") else None,
        "bwd_bwd_source_chars": len(bb_kernel.get_kernel_source()) if hasattr(bb_kernel, "get_kernel_source") else None,
    }


def _run_pair(
    bf_kernel: Any,
    bb_kernel: Any,
    shape: Shape,
    inputs: dict[str, Any],
    flat_inputs: bool,
    flat_qk_dot: bool,
) -> dict[str, Any]:
    import torch

    outputs = _empty_outputs(shape, torch.bfloat16, torch.device("cuda"), flat_qk_dot)
    q_arg = inputs["q_flat"] if flat_inputs else inputs["q"]
    k_arg = inputs["k_flat"] if flat_inputs else inputs["k"]
    bf_kernel(
        inputs["dout"], q_arg, k_arg, inputs["v"], inputs["q_bias"], inputs["k_bias"],
        inputs["mimo_v"], inputs["mimo_o"], outputs["dmimo_o"], outputs["states"],
        outputs["z"], outputs["mimo_z"], outputs["dz"], outputs["dmimo_z"], inputs["angles"],
        inputs["da_cs"], inputs["da_cs_rev"], inputs["dt"], inputs["trap"], inputs["d"],
        outputs["qk_dot"], inputs["segsum"],
    )
    bb_kernel(
        inputs["dout"], q_arg, k_arg, inputs["v"], inputs["q_bias"], inputs["k_bias"],
        inputs["mimo_v"], inputs["mimo_o"], outputs["dk"], outputs["dv"],
        outputs["dmimo_v"], outputs["states"], outputs["dq"], outputs["z"],
        outputs["mimo_z"], inputs["angles"], inputs["da_cs"], inputs["da_cs_rev"],
        inputs["dt"], inputs["trap"], outputs["dfactor"], outputs["dgamma_diag"],
        outputs["dangles"], inputs["d"], outputs["dd"], outputs["qk_dot"],
        outputs["dda"], outputs["dssda"], outputs["dda_cs_rev"], outputs["dda_cs"],
        inputs["segsum"],
    )
    torch.cuda.synchronize()
    return outputs


def _time_ms(callable_obj: Any, warmup: int = 5, repeat: int = 20) -> float:
    import torch

    for _ in range(warmup):
        callable_obj()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        callable_obj()
    end.record()
    torch.cuda.synchronize()
    return round(float(start.elapsed_time(end)) / repeat, 4)


def _time_pair(
    bf_kernel: Any,
    bb_kernel: Any,
    shape: Shape,
    inputs: dict[str, Any],
    flat_inputs: bool,
    flat_qk_dot: bool,
) -> dict[str, float]:
    import torch

    outputs = _empty_outputs(shape, torch.bfloat16, torch.device("cuda"), flat_qk_dot)
    q_arg = inputs["q_flat"] if flat_inputs else inputs["q"]
    k_arg = inputs["k_flat"] if flat_inputs else inputs["k"]
    bf_args = (
        inputs["dout"], q_arg, k_arg, inputs["v"], inputs["q_bias"], inputs["k_bias"],
        inputs["mimo_v"], inputs["mimo_o"], outputs["dmimo_o"], outputs["states"],
        outputs["z"], outputs["mimo_z"], outputs["dz"], outputs["dmimo_z"], inputs["angles"],
        inputs["da_cs"], inputs["da_cs_rev"], inputs["dt"], inputs["trap"], inputs["d"],
        outputs["qk_dot"], inputs["segsum"],
    )
    bb_args = (
        inputs["dout"], q_arg, k_arg, inputs["v"], inputs["q_bias"], inputs["k_bias"],
        inputs["mimo_v"], inputs["mimo_o"], outputs["dk"], outputs["dv"],
        outputs["dmimo_v"], outputs["states"], outputs["dq"], outputs["z"],
        outputs["mimo_z"], inputs["angles"], inputs["da_cs"], inputs["da_cs_rev"],
        inputs["dt"], inputs["trap"], outputs["dfactor"], outputs["dgamma_diag"],
        outputs["dangles"], inputs["d"], outputs["dd"], outputs["qk_dot"],
        outputs["dda"], outputs["dssda"], outputs["dda_cs_rev"], outputs["dda_cs"],
        inputs["segsum"],
    )

    def run_bf() -> None:
        bf_kernel(*bf_args)

    def run_bb() -> None:
        bb_kernel(*bb_args)

    def run_combined() -> None:
        bf_kernel(*bf_args)
        bb_kernel(*bb_args)

    # Ensure bwd_bwd has initialized states/qk_dot before timing it alone.
    bf_kernel(*bf_args)
    torch.cuda.synchronize()

    return {
        "bwd_fwd_ms": _time_ms(run_bf),
        "bwd_bwd_ms": _time_ms(run_bb),
        "combined_ms": _time_ms(run_combined),
    }


def _compare_outputs(shape: Shape, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    names = [
        "dmimo_o",
        "states",
        "qk_dot",
        "dk",
        "dv",
        "dmimo_v",
        "dq",
        "dfactor",
        "dgamma_diag",
        "dangles",
        "dd",
        "dda",
        "dssda",
        "dda_cs_rev",
        "dda_cs",
    ]
    diffs: dict[str, Any] = {}
    for name in names:
        lhs = baseline[name]
        rhs = candidate[name]
        if name == "qk_dot":
            lhs = lhs.reshape(shape.B, shape.H, shape.S, shape.R * shape.R)
            rhs = rhs.reshape(shape.B, shape.H, shape.S, shape.R * shape.R)
        diff = (lhs.float() - rhs.float()).abs()
        ref = lhs.float().abs()
        max_abs = float(diff.max().item())
        ref_max = float(ref.max().item())
        diffs[name] = {
            "max_abs": max_abs,
            "ref_absmax": ref_max,
            "rel_to_ref_absmax": max_abs / max(ref_max, 1.0e-12),
        }
    return diffs


def _shape_bytes_estimate(shape: Shape) -> int:
    import math

    B, S, H, N, P, R = shape.B, shape.S, shape.H, shape.N, shape.P, shape.R
    nchunks = math.ceil(S / shape.chunk_size)
    bf16 = 2
    fp32 = 4
    total = 0
    total += 2 * B * S * R * shape.G * N * bf16
    total += 2 * B * S * H * P * bf16
    total += B * H * nchunks * N * P * bf16
    total += B * H * S * R * R * bf16
    total += 2 * B * S * R * H * N * bf16
    total += B * S * H * P * bf16
    total += 6 * B * H * S * fp32
    total += B * H * nchunks * shape.chunk_size * shape.chunk_size * fp32
    return total


def _run_shape(shape: Shape) -> dict[str, Any]:
    import traceback
    import torch

    result: dict[str, Any] = {
        "shape": asdict(shape),
        "estimated_tensor_bytes": _shape_bytes_estimate(shape),
    }
    try:
        inputs = _make_inputs(shape)

        base_mod, base_meta = _load_module("baseline_non_tma", shape.name)
        cand_mod, cand_meta = _load_module("qk_shared_direct", shape.name)
        result["module_meta"] = {"baseline_non_tma": base_meta, "qk_shared_direct": cand_meta}
        if base_mod is None or cand_mod is None:
            result["status"] = "module_load_failed"
            return result

        base_bf, base_bb, base_compile = _compile_kernels(base_mod, shape)
        cand_bf, cand_bb, cand_compile = _compile_kernels(cand_mod, shape)
        result["compile"] = {
            "baseline_non_tma": base_compile,
            "qk_shared_direct": cand_compile,
        }

        baseline_outputs = _run_pair(base_bf, base_bb, shape, inputs, flat_inputs=False, flat_qk_dot=False)
        candidate_outputs = _run_pair(cand_bf, cand_bb, shape, inputs, flat_inputs=True, flat_qk_dot=True)
        result["diffs"] = _compare_outputs(shape, baseline_outputs, candidate_outputs)

        result["timings_ms"] = {
            "baseline_non_tma": _time_pair(base_bf, base_bb, shape, inputs, flat_inputs=False, flat_qk_dot=False),
            "qk_shared_direct": _time_pair(cand_bf, cand_bb, shape, inputs, flat_inputs=True, flat_qk_dot=True),
        }
        base_combined = result["timings_ms"]["baseline_non_tma"]["combined_ms"]
        cand_combined = result["timings_ms"]["qk_shared_direct"]["combined_ms"]
        result["speedup_vs_baseline"] = round(base_combined / cand_combined, 4) if cand_combined else None
        result["max_main_grad_abs_diff"] = max(
            result["diffs"][name]["max_abs"]
            for name in ("dq", "dk", "dv", "dmimo_v", "dmimo_o")
        )
        result["status"] = "ok"
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        result["status"] = "crashed"
        result["exception_type"] = type(exc).__name__
        result["exception_short"] = str(exc)[:1000]
        result["traceback_tail"] = traceback.format_exc()[-5000:]
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return result


@app.function(image=_image(), gpu=GPU_SPEC, timeout=7200)
def run_perf(requested_gpu: str) -> dict[str, Any]:
    _install_source_paths()
    return {
        "device": _device_report(requested_gpu),
        "tilelang": _tilelang_report(),
        "shapes": [_run_shape(shape) for shape in SHAPES if shape.enabled],
    }


@app.local_entrypoint()
def main() -> None:
    result = run_perf.remote(GPU_SPEC)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
