"""Bounded Modal benchmark for Mamba3 MIMO bwd_bwd streaming live-set rewrite.

Compares two non-production variants on Hopper GPUs:
  * baseline: upstream non-TMA/non-WS TileLang kernels
  * stage2_force_nontma: qk_shared_direct + bf_num_stages=1 / bb_num_stages=0,
    with only small float32 vector slice copies forced off TMA. This keeps the
    useful bwd_fwd WS/TMA path while leaving bwd_bwd on the faster non-WS path.
  * streaming_live_set_*: stage2 plus an incremental bwd_bwd live-set rewrite
    that updates dPsiV in-place, builds PsiV directly in shared memory, and
    streams DGAMMA_DIAG reduction without prereduce fragment storage.

The harness writes full JSON/CSV/source artifacts to a Modal Volume and prints a
compact summary for run logs. It does not change production defaults.

Run examples:

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 10m \
        modal run scripts/modal_mamba3_bwd_bwd_streaming_live_set_benchmark.py \
        --shape-csv representative --iters 8 --warmup 2

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 12m \
        modal run scripts/modal_mamba3_bwd_bwd_streaming_live_set_benchmark.py \
        --shape-csv productionish --iters 4 --warmup 1

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 15m \
        modal run scripts/modal_mamba3_bwd_bwd_streaming_live_set_benchmark.py \
        --shape-csv productionish \
        --variant-csv stage2_bf1_bb0,streaming_live_set_bf1_bb0,streaming_live_set_bf1_bb1 \
        --torch-profile \
        --iters 4 --warmup 1
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")
BENCH_VOLUME_NAME = os.environ.get("CPPMEGA_MODAL_BENCH_VOLUME", "cppmega-mamba3-benchmarks")

APP_NAME = "cppmega-mamba3-bwd-bwd-streaming-live-set-benchmark"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"
BENCH_ROOT = "/benchmarks"
BENCH_PREFIX = "mamba3_bwd_bwd_streaming_live_set_benchmark"

bench_volume = modal.Volume.from_name(BENCH_VOLUME_NAME, create_if_missing=True)


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
    chunk: int = 16
    rotary_dim_divisor: int = 4


SHAPES: dict[str, Shape] = {
    "smoke": Shape("smoke", B=1, S=256, H=4, G=1, N=64, P=64, R=4),
    "representative": Shape("representative", B=2, S=1024, H=16, G=1, N=64, P=64, R=4),
    "productionish": Shape("productionish", B=4, S=4096, H=32, G=1, N=64, P=128, R=4),
}


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "patch": None,
        "flattened_inputs": False,
        "flat_qk_dot": False,
        "bf_threads": 128,
        "bf_num_stages": 0,
        "bb_threads": 256,
        "bb_num_stages": 0,
    },
    "stage2_force_nontma": {
        "patch": "stage2_force_nontma",
        "patch_file": "mamba3_bwd_stage2_force_nontma.patch",
        "flattened_inputs": True,
        "flat_qk_dot": True,
        "bf_threads": 128,
        "bf_num_stages": 1,
        "bb_threads": 256,
        "bb_num_stages": 0,
    },
    "stage2_bf1_bb0": {
        "patch": "stage2_force_nontma",
        "patch_file": "mamba3_bwd_stage2_force_nontma.patch",
        "flattened_inputs": True,
        "flat_qk_dot": True,
        "bf_threads": 128,
        "bf_num_stages": 1,
        "bb_threads": 256,
        "bb_num_stages": 0,
    },
    "stage2_bf0_bb1": {
        "patch": "stage2_force_nontma",
        "patch_file": "mamba3_bwd_stage2_force_nontma.patch",
        "flattened_inputs": True,
        "flat_qk_dot": True,
        "bf_threads": 128,
        "bf_num_stages": 0,
        "bb_threads": 256,
        "bb_num_stages": 1,
    },
    "streaming_live_set_bf1_bb0": {
        "patch": "stage2_force_nontma",
        "patch_file": "mamba3_bwd_stage2_force_nontma.patch",
        "extra_patch_files": ["mamba3_bwd_bwd_streaming_live_set.patch"],
        "flattened_inputs": True,
        "flat_qk_dot": True,
        "bf_threads": 128,
        "bf_num_stages": 1,
        "bb_threads": 256,
        "bb_num_stages": 0,
    },
    "streaming_live_set_bf1_bb1": {
        "patch": "stage2_force_nontma",
        "patch_file": "mamba3_bwd_stage2_force_nontma.patch",
        "extra_patch_files": ["mamba3_bwd_bwd_streaming_live_set.patch"],
        "flattened_inputs": True,
        "flat_qk_dot": True,
        "bf_threads": 128,
        "bf_num_stages": 1,
        "bb_threads": 256,
        "bb_num_stages": 1,
    },
}


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

    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "requested_gpu_spec": requested_gpu,
        "image_ref": os.environ.get("CPPMEGA_IMAGE_REF", GHCR_REF),
    }


def _tilelang_report() -> dict[str, Any]:
    import importlib.metadata
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


def _selected_shapes(shape_csv: str) -> list[Shape]:
    shapes: list[Shape] = []
    for name in [part.strip() for part in shape_csv.split(",")]:
        if not name:
            continue
        if name not in SHAPES:
            raise ValueError(f"unknown shape {name!r}; choose one of {sorted(SHAPES)}")
        shapes.append(SHAPES[name])
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def _selected_variants(variant_csv: str) -> list[str]:
    variants: list[str] = []
    for name in [part.strip() for part in variant_csv.split(",")]:
        if not name:
            continue
        if name not in VARIANTS:
            raise ValueError(f"unknown variant {name!r}; choose one of {sorted(VARIANTS)}")
        variants.append(name)
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


def _apply_patch(dst: str, patch_name: str) -> dict[str, Any]:
    import subprocess

    patch_file = f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz/{patch_name}"
    with open(patch_file, "rb") as handle:
        patch_bytes = handle.read()
    proc = subprocess.run(
        ["patch", "-p4", dst],
        input=patch_bytes,
        capture_output=True,
        cwd=os.path.dirname(dst),
        check=False,
    )
    return {
        "patch_file": patch_file,
        "patch_rc": proc.returncode,
        "patch_stdout_tail": proc.stdout.decode(errors="replace")[-2000:],
        "patch_stderr_tail": proc.stderr.decode(errors="replace")[-2000:],
    }


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

    counts["disable_tma_false"] = text.count("TL_DISABLE_TMA_LOWER: False")
    counts["disable_ws_false"] = text.count("TL_DISABLE_WARP_SPECIALIZED: False")
    counts["qk_shared_direct_refs"] = text.count("qk_dot_shared[cs, r_out * R + r_in]")
    counts["qk_direct_refs"] = text.count("qk_dot_frag[cs, r_out * R + r_in]")
    counts["dpsiv_d_fused_frag_refs"] = text.count("dPsiV_D_fused_frag")
    counts["psiv_frag_refs"] = text.count("PsiV_frag")
    counts["dgamma_prereduce_refs"] = text.count("dgamma_diag_prereduce_frag")
    counts["streaming_live_set_refs"] = text.count("Streaming live-set rewrite")
    counts["per_copy_disable_tma"] = text.count("disable_tma=True")
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
            "                        dPsiV_D_fused_frag[csr, p] += dPhiO_shared[csr_out, p] * qk_dot_shared[cs, r_out * R + r_in] * gamma_dPsiV_frag[cs]",
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


def _prepare_variant(variant: str) -> tuple[str, dict[str, Any]]:
    import shutil
    import tempfile

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    src = f"{SOURCE_ROOT}/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"
    work = tempfile.mkdtemp(prefix=f"cppmega_mamba3_stage2_force_nontma_{variant}_")
    dst = os.path.join(work, "mamba3_mimo_bwd.py")
    shutil.copy(src, dst)

    meta: dict[str, Any] = {"variant": variant, "work": work, "source_path": dst, "patch": None}
    patch_kind = VARIANTS[variant]["patch"]
    if patch_kind is None:
        return dst, meta

    if patch_kind == "stage2_force_nontma":
        patch_file = str(VARIANTS[variant].get("patch_file", "mamba3_bwd_stage2_force_nontma.patch"))
        patch_meta = _apply_patch(dst, patch_file)
        meta.update({"patch": patch_file, **patch_meta})
        if patch_meta["patch_rc"] != 0:
            return dst, meta
        for extra_patch_file in VARIANTS[variant].get("extra_patch_files", []):
            extra_meta = _apply_patch(dst, str(extra_patch_file))
            meta.setdefault("extra_patches", []).append(extra_meta)
            if extra_meta["patch_rc"] != 0:
                meta["patch_rc"] = extra_meta["patch_rc"]
                return dst, meta
        meta["replacement_counts"] = _apply_text_replacements(dst, [])
    else:
        patch_meta = _apply_patch(dst, "mamba3_bwd_layout_fix.patch")
        meta.update({"patch": "mamba3_bwd_layout_fix.patch", **patch_meta})
        if patch_meta["patch_rc"] != 0:
            return dst, meta

        if patch_kind == "layout_fix_plus_qk_shared_direct":
            meta["replacement_counts"] = _apply_text_replacements(dst, _qk_shared_direct_replacements())
        elif patch_kind != "layout_fix":
            raise ValueError(f"unknown patch kind: {patch_kind}")
    return dst, meta


def _import_variant(path: str, variant: str, shape: Shape):
    import importlib.util
    import sys
    import time

    name = f"cppmega_mamba3_stage2_force_nontma_{shape.name}_{variant}_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import variant module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _source_markers(source: str, consumer_threads: int) -> dict[str, Any]:
    launch_bounds = sorted(set(re.findall(r"__launch_bounds__\((\d+),\s*(\d+)\)", source)))
    launch_bound_threads = {int(item[0]) for item in launch_bounds}
    producer_guard = f"if ({consumer_threads} <= ((int)threadIdx.x))"
    return {
        "source_chars": len(source),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "tma_load_count": source.count("tl::tma_load"),
        "tma_store_count": source.count("tl::tma_store"),
        "mbarrier_wait_count": source.count("mbarrier_wait"),
        "launch_bounds": launch_bounds,
        "producer_guard": producer_guard in source,
        "expected_ws_launch_bound": any(bound > consumer_threads for bound in launch_bound_threads),
        "contains_qk_shared_direct": "qk_dot_shared[cs, r_out * R + r_in]" in source,
    }


def _source_meta(kernel: Any, path: str, consumer_threads: int) -> dict[str, Any]:
    if not hasattr(kernel, "get_kernel_source"):
        return {"has_get_kernel_source": False}
    source = kernel.get_kernel_source()
    if not isinstance(source, str):
        source = str(source)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    meta = _source_markers(source, consumer_threads)
    meta.update({"has_get_kernel_source": True, "artifact": path, "basename": os.path.basename(path)})
    return meta


def _make_kernels(mod: Any, shape: Shape, variant: str) -> tuple[Any, Any, dict[str, Any]]:
    import time

    cfg = VARIANTS[variant]
    args = (
        shape.B,
        shape.S,
        shape.H,
        shape.G,
        shape.N,
        shape.P,
        shape.R,
        False,
        False,
        True,
        shape.chunk,
        shape.rotary_dim_divisor,
        "bfloat16",
    )

    t0 = time.time()
    bf_kernel = mod.mamba_mimo_bwd_fwd(*args, cfg["bf_threads"], cfg["bf_num_stages"])
    bf_sec = time.time() - t0
    t1 = time.time()
    bb_kernel = mod.mamba_mimo_bwd_bwd(*args, cfg["bb_threads"], cfg["bb_num_stages"])
    bb_sec = time.time() - t1
    return bf_kernel, bb_kernel, {
        "bf_threads": cfg["bf_threads"],
        "bf_num_stages": cfg["bf_num_stages"],
        "bb_threads": cfg["bb_threads"],
        "bb_num_stages": cfg["bb_num_stages"],
        "bwd_fwd_compile_sec": round(bf_sec, 3),
        "bwd_bwd_compile_sec": round(bb_sec, 3),
    }


def _make_inputs(shape: Shape) -> dict[str, Any]:
    import torch
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(20260429)

    q = torch.randn(shape.B, shape.S, shape.R, shape.G, shape.N, device=device, dtype=dtype) * 0.01
    k = torch.randn(shape.B, shape.S, shape.R, shape.G, shape.N, device=device, dtype=dtype) * 0.01
    v = torch.randn(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype) * 0.01
    dout = torch.randn(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype) * 0.01
    q_bias = torch.randn(shape.H, shape.R, shape.N, device=device, dtype=torch.float32) * 0.01
    k_bias = torch.randn(shape.H, shape.R, shape.N, device=device, dtype=torch.float32) * 0.01
    mimo_v = torch.randn(shape.H, shape.R, shape.P, device=device, dtype=torch.float32) * 0.01
    mimo_o = torch.randn(shape.H, shape.R, shape.P, device=device, dtype=torch.float32) * 0.01
    angles = (
        torch.randn(
            shape.B,
            shape.S,
            shape.H,
            shape.N // shape.rotary_dim_divisor,
            device=device,
            dtype=torch.float32,
        )
        * 0.01
    )
    dt = torch.randn(shape.B, shape.H, shape.S, device=device, dtype=torch.float32) * 0.01
    trap = torch.randn(shape.B, shape.H, shape.S, device=device, dtype=dtype) * 0.01
    adt = -torch.abs(torch.randn(shape.B, shape.H, shape.S, device=device, dtype=torch.float32) * 0.01)
    da_cs, da_cs_rev, segsum = compute_dacs_segsum_triton(adt, shape.chunk)

    return {
        "q": q,
        "k": k,
        "q_flat": q.view(shape.B, shape.S * shape.R, shape.G, shape.N),
        "k_flat": k.view(shape.B, shape.S * shape.R, shape.G, shape.N),
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
        "d": torch.zeros(shape.H, device=device, dtype=torch.float32),
        "segsum": segsum,
    }


def _empty_outputs(shape: Shape, flat_qk_dot: bool) -> dict[str, Any]:
    import math
    import torch

    device = torch.device("cuda")
    dtype = torch.bfloat16
    nchunks = math.ceil(shape.S / shape.chunk)
    qk_shape = (shape.B, shape.H, shape.S, shape.R * shape.R) if flat_qk_dot else (
        shape.B,
        shape.H,
        shape.S,
        shape.R,
        shape.R,
    )
    return {
        "z": torch.zeros(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype),
        "dz": torch.zeros(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype),
        "mimo_z": torch.zeros(shape.H, shape.R, shape.P, device=device, dtype=torch.float32),
        "dmimo_z": torch.zeros(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32),
        "dmimo_o": torch.zeros(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32),
        "states": torch.zeros(shape.B, shape.H, nchunks, shape.N, shape.P, device=device, dtype=dtype),
        "qk_dot": torch.zeros(*qk_shape, device=device, dtype=dtype),
        "dk": torch.zeros(shape.B, shape.S * shape.R, shape.H, shape.N, device=device, dtype=dtype),
        "dv": torch.zeros(shape.B, shape.S, shape.H, shape.P, device=device, dtype=dtype),
        "dmimo_v": torch.zeros(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32),
        "dq": torch.zeros(shape.B, shape.S * shape.R, shape.H, shape.N, device=device, dtype=dtype),
        "dfactor": torch.zeros(shape.B, shape.H, shape.S, device=device, dtype=torch.float32),
        "dgamma_diag": torch.zeros(shape.B, shape.H, shape.S, device=device, dtype=torch.float32),
        "dangles": torch.zeros(
            shape.B,
            shape.S,
            shape.H,
            shape.N // shape.rotary_dim_divisor,
            device=device,
            dtype=torch.float32,
        ),
        "dd": torch.zeros(shape.B, shape.H, device=device, dtype=torch.float32),
        "dda": torch.zeros(shape.B, shape.H, shape.S, device=device, dtype=torch.float32),
        "dssda": torch.zeros(
            shape.B,
            shape.H,
            nchunks,
            shape.chunk,
            shape.chunk,
            device=device,
            dtype=torch.float32,
        ),
        "dda_cs_rev": torch.zeros(shape.B, shape.H, shape.S, device=device, dtype=torch.float32),
        "dda_cs": torch.zeros(shape.B, shape.H, shape.S, device=device, dtype=torch.float32),
    }


def _kernel_args(
    shape: Shape,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    *,
    flattened_inputs: bool,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    q_arg = inputs["q_flat"] if flattened_inputs else inputs["q"]
    k_arg = inputs["k_flat"] if flattened_inputs else inputs["k"]
    bf_args = (
        inputs["dout"],
        q_arg,
        k_arg,
        inputs["v"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["mimo_v"],
        inputs["mimo_o"],
        outputs["dmimo_o"],
        outputs["states"],
        outputs["z"],
        outputs["mimo_z"],
        outputs["dz"],
        outputs["dmimo_z"],
        inputs["angles"],
        inputs["da_cs"],
        inputs["da_cs_rev"],
        inputs["dt"],
        inputs["trap"],
        inputs["d"],
        outputs["qk_dot"],
        inputs["segsum"],
    )
    bb_args = (
        inputs["dout"],
        q_arg,
        k_arg,
        inputs["v"],
        inputs["q_bias"],
        inputs["k_bias"],
        inputs["mimo_v"],
        inputs["mimo_o"],
        outputs["dk"],
        outputs["dv"],
        outputs["dmimo_v"],
        outputs["states"],
        outputs["dq"],
        outputs["z"],
        outputs["mimo_z"],
        inputs["angles"],
        inputs["da_cs"],
        inputs["da_cs_rev"],
        inputs["dt"],
        inputs["trap"],
        outputs["dfactor"],
        outputs["dgamma_diag"],
        outputs["dangles"],
        inputs["d"],
        outputs["dd"],
        outputs["qk_dot"],
        outputs["dda"],
        outputs["dssda"],
        outputs["dda_cs_rev"],
        outputs["dda_cs"],
        inputs["segsum"],
    )
    return bf_args, bb_args


def _run_pair(
    shape: Shape,
    bf_kernel: Any,
    bb_kernel: Any,
    inputs: dict[str, Any],
    *,
    flattened_inputs: bool,
    flat_qk_dot: bool,
) -> dict[str, Any]:
    import torch

    outputs = _empty_outputs(shape, flat_qk_dot)
    bf_args, bb_args = _kernel_args(shape, inputs, outputs, flattened_inputs=flattened_inputs)
    bf_kernel(*bf_args)
    bb_kernel(*bb_args)
    torch.cuda.synchronize()
    return outputs


def _stats(values: list[float]) -> dict[str, Any]:
    import math

    ordered = sorted(values)
    if not ordered:
        return {"count": 0}

    def pct(percent: float) -> float:
        idx = min(len(ordered) - 1, max(0, math.ceil((percent / 100.0) * len(ordered)) - 1))
        return ordered[idx]

    mean = sum(ordered) / len(ordered)
    var = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean_ms": mean,
        "min_ms": ordered[0],
        "p50_ms": pct(50),
        "p90_ms": pct(90),
        "p95_ms": pct(95),
        "max_ms": ordered[-1],
        "std_ms": math.sqrt(var),
        "samples_ms": values,
    }


def _time_cuda_events(fn: Any, *, warmup: int, iters: int) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _time_pair(
    shape: Shape,
    bf_kernel: Any,
    bb_kernel: Any,
    inputs: dict[str, Any],
    *,
    flattened_inputs: bool,
    flat_qk_dot: bool,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    import torch

    outputs = _empty_outputs(shape, flat_qk_dot)
    bf_args, bb_args = _kernel_args(shape, inputs, outputs, flattened_inputs=flattened_inputs)

    def run_bf() -> None:
        bf_kernel(*bf_args)

    def run_bb() -> None:
        bb_kernel(*bb_args)

    def run_chain() -> None:
        bf_kernel(*bf_args)
        bb_kernel(*bb_args)

    bf_kernel(*bf_args)
    torch.cuda.synchronize()
    return {
        "bwd_fwd": _stats(_time_cuda_events(run_bf, warmup=warmup, iters=iters)),
        "bwd_bwd": _stats(_time_cuda_events(run_bb, warmup=warmup, iters=iters)),
        "chain": _stats(_time_cuda_events(run_chain, warmup=warmup, iters=iters)),
    }


def _profile_with_torch_profiler(
    variant: str,
    shape: Shape,
    bf_kernel: Any,
    bb_kernel: Any,
    inputs: dict[str, Any],
    artifact_dir: str,
    *,
    flattened_inputs: bool,
    flat_qk_dot: bool,
) -> dict[str, Any]:
    import traceback

    import torch

    trace_path = os.path.join(artifact_dir, f"{shape.name}_{variant}_torch_trace.json")
    table_path = os.path.join(artifact_dir, f"{shape.name}_{variant}_torch_cuda_table.txt")
    try:
        outputs = _empty_outputs(shape, flat_qk_dot)
        bf_args, bb_args = _kernel_args(shape, inputs, outputs, flattened_inputs=flattened_inputs)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for _ in range(3):
                with torch.profiler.record_function(f"{variant}.bwd_fwd"):
                    torch.cuda.nvtx.range_push(f"{variant}.bwd_fwd")
                    bf_kernel(*bf_args)
                    torch.cuda.nvtx.range_pop()
                with torch.profiler.record_function(f"{variant}.bwd_bwd"):
                    torch.cuda.nvtx.range_push(f"{variant}.bwd_bwd")
                    bb_kernel(*bb_args)
                    torch.cuda.nvtx.range_pop()
                prof.step()
        torch.cuda.synchronize()
        prof.export_chrome_trace(trace_path)
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
        with open(table_path, "w", encoding="utf-8") as handle:
            handle.write(table)
        return {"status": "ok", "trace": trace_path, "table": table_path}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback_tail": traceback.format_exc()[-4000:],
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


def _max_main_grad_diff(diffs: dict[str, Any]) -> float:
    return max(diffs[name]["max_abs"] for name in ("dq", "dk", "dv", "dmimo_v", "dmimo_o"))


def _shape_bytes_estimate(shape: Shape) -> int:
    import math

    nchunks = math.ceil(shape.S / shape.chunk)
    bf16 = 2
    fp32 = 4
    total = 0
    total += 2 * shape.B * shape.S * shape.R * shape.G * shape.N * bf16
    total += 2 * shape.B * shape.S * shape.H * shape.P * bf16
    total += shape.B * shape.H * nchunks * shape.N * shape.P * bf16
    total += shape.B * shape.H * shape.S * shape.R * shape.R * bf16
    total += 2 * shape.B * shape.S * shape.R * shape.H * shape.N * bf16
    total += shape.B * shape.S * shape.H * shape.P * bf16
    total += 6 * shape.B * shape.H * shape.S * fp32
    total += shape.B * shape.H * nchunks * shape.chunk * shape.chunk * fp32
    return total


def _benchmark_variant(
    variant: str,
    shape: Shape,
    inputs: dict[str, Any],
    shape_dir: str,
    *,
    warmup: int,
    iters: int,
    torch_profile: bool,
) -> dict[str, Any]:
    import time
    import traceback

    import torch

    t0 = time.time()
    cfg = VARIANTS[variant]
    result: dict[str, Any] = {"variant": variant, "config": cfg}
    try:
        variant_dir = os.path.join(shape_dir, variant)
        os.makedirs(variant_dir, exist_ok=True)
        path, prep_meta = _prepare_variant(variant)
        result["prepare"] = prep_meta
        if prep_meta.get("patch_rc", 0) != 0:
            result["status"] = "patch_failed"
            return result

        mod = _import_variant(path, variant, shape)
        bf_kernel, bb_kernel, compile_meta = _make_kernels(mod, shape, variant)
        result["compile"] = compile_meta
        result["tilelang_source"] = {
            "bwd_fwd": _source_meta(
                bf_kernel,
                os.path.join(variant_dir, "bwd_fwd_kernel_source.cu"),
                int(cfg["bf_threads"]),
            ),
            "bwd_bwd": _source_meta(
                bb_kernel,
                os.path.join(variant_dir, "bwd_bwd_kernel_source.cu"),
                int(cfg["bb_threads"]),
            ),
        }
        result["correctness_outputs"] = _run_pair(
            shape,
            bf_kernel,
            bb_kernel,
            inputs,
            flattened_inputs=bool(cfg["flattened_inputs"]),
            flat_qk_dot=bool(cfg["flat_qk_dot"]),
        )
        result["elapsed"] = _time_pair(
            shape,
            bf_kernel,
            bb_kernel,
            inputs,
            flattened_inputs=bool(cfg["flattened_inputs"]),
            flat_qk_dot=bool(cfg["flat_qk_dot"]),
            warmup=warmup,
            iters=iters,
        )
        if torch_profile:
            result["torch_profiler"] = _profile_with_torch_profiler(
                variant,
                shape,
                bf_kernel,
                bb_kernel,
                inputs,
                variant_dir,
                flattened_inputs=bool(cfg["flattened_inputs"]),
                flat_qk_dot=bool(cfg["flat_qk_dot"]),
            )
        result["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        result["max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "status": "crashed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback_tail": traceback.format_exc()[-6000:],
            }
        )
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 3)
    return result


def _strip_tensors(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    out.pop("correctness_outputs", None)
    return out


def _compare_shape(shape_result: dict[str, Any]) -> dict[str, Any]:
    variants = {entry["variant"]: entry for entry in shape_result["variants"]}
    reference = next((entry for entry in shape_result["variants"] if entry.get("status") == "ok"), None)
    if not reference:
        return {"status": "missing_ok_baseline"}

    comparisons: dict[str, Any] = {
        "status": "ok",
        "baseline_variant": reference["variant"],
        "vs_baseline": {},
    }
    for variant_name, variant_result in variants.items():
        if variant_name == reference["variant"]:
            continue
        if variant_result.get("status") != "ok":
            comparisons["vs_baseline"][variant_name] = {"status": "missing_ok_variant"}
            continue
        diffs = _compare_outputs(
            Shape(**shape_result["shape"]),
            reference["correctness_outputs"],
            variant_result["correctness_outputs"],
        )
        speedups: dict[str, float | None] = {}
        for phase in ("bwd_fwd", "bwd_bwd", "chain"):
            base_mean = reference["elapsed"][phase]["mean_ms"]
            cand_mean = variant_result["elapsed"][phase]["mean_ms"]
            speedups[phase] = base_mean / cand_mean if cand_mean else None
        comparisons["vs_baseline"][variant_name] = {
            "status": "ok",
            "speedup": speedups,
            "diffs": diffs,
            "max_main_grad_abs_diff": _max_main_grad_diff(diffs),
        }
    return comparisons


def _summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    summary_shapes: list[dict[str, Any]] = []
    for shape_result in report["shapes"]:
        row: dict[str, Any] = {
            "shape": shape_result["shape"]["name"],
            "status": shape_result["status"],
            "variants": {},
            "comparisons": shape_result.get("comparison", {}).get("vs_baseline", {}),
        }
        for variant in shape_result["variants"]:
            if variant.get("status") != "ok":
                row["variants"][variant["variant"]] = {
                    "status": variant.get("status"),
                    "exception_type": variant.get("exception_type"),
                    "exception": variant.get("exception"),
                }
                continue
            row["variants"][variant["variant"]] = {
                "status": "ok",
                "bwd_fwd_mean_ms": variant["elapsed"]["bwd_fwd"]["mean_ms"],
                "bwd_bwd_mean_ms": variant["elapsed"]["bwd_bwd"]["mean_ms"],
                "chain_mean_ms": variant["elapsed"]["chain"]["mean_ms"],
                "bwd_fwd_ws": variant["tilelang_source"]["bwd_fwd"]["producer_guard"],
                "bwd_bwd_ws": variant["tilelang_source"]["bwd_bwd"]["producer_guard"],
                "bwd_fwd_tma_loads": variant["tilelang_source"]["bwd_fwd"]["tma_load_count"],
                "bwd_bwd_tma_loads": variant["tilelang_source"]["bwd_bwd"]["tma_load_count"],
                "torch_profiler": variant.get("torch_profiler"),
            }
        summary_shapes.append(row)
    return {
        "run_id": report["run_id"],
        "volume": report["volume"],
        "volume_relpath": report["volume_relpath"],
        "device": report["device"],
        "settings": report["settings"],
        "artifacts": report["artifacts"],
        "shapes": summary_shapes,
    }


def _write_summary_csv(summary: dict[str, Any], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "shape",
                "variant",
                "status",
                "bwd_fwd_mean_ms",
                "bwd_bwd_mean_ms",
                "chain_mean_ms",
                "speedup_bwd_fwd_vs_baseline",
                "speedup_bwd_bwd_vs_baseline",
                "speedup_chain_vs_baseline",
                "max_main_grad_abs_diff_vs_baseline",
                "bwd_fwd_ws",
                "bwd_bwd_ws",
                "bwd_fwd_tma_loads",
                "bwd_bwd_tma_loads",
            ],
        )
        writer.writeheader()
        for shape in summary["shapes"]:
            comparisons = shape.get("comparisons", {})
            for variant, data in shape["variants"].items():
                compare = comparisons.get(variant, {})
                speedup = compare.get("speedup", {})
                writer.writerow(
                    {
                        "shape": shape["shape"],
                        "variant": variant,
                        "status": data.get("status"),
                        "bwd_fwd_mean_ms": data.get("bwd_fwd_mean_ms"),
                        "bwd_bwd_mean_ms": data.get("bwd_bwd_mean_ms"),
                        "chain_mean_ms": data.get("chain_mean_ms"),
                        "speedup_bwd_fwd_vs_baseline": speedup.get("bwd_fwd"),
                        "speedup_bwd_bwd_vs_baseline": speedup.get("bwd_bwd"),
                        "speedup_chain_vs_baseline": speedup.get("chain"),
                        "max_main_grad_abs_diff_vs_baseline": compare.get("max_main_grad_abs_diff"),
                        "bwd_fwd_ws": data.get("bwd_fwd_ws"),
                        "bwd_bwd_ws": data.get("bwd_bwd_ws"),
                        "bwd_fwd_tma_loads": data.get("bwd_fwd_tma_loads"),
                        "bwd_bwd_tma_loads": data.get("bwd_bwd_tma_loads"),
                    }
                )


@app.function(image=_image(), gpu=GPU_SPEC, timeout=1200, volumes={BENCH_ROOT: bench_volume})
def run_benchmark(
    requested_gpu: str,
    run_id: str | None,
    shape_csv: str,
    variant_csv: str,
    warmup: int,
    iters: int,
    torch_profile: bool,
) -> dict[str, Any]:
    import time

    import torch

    _install_source_paths()
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_rel = f"{BENCH_PREFIX}/{run_id}"
    run_dir = os.path.join(BENCH_ROOT, run_rel)
    os.makedirs(run_dir, exist_ok=True)

    report: dict[str, Any] = {
        "run_id": run_id,
        "volume": BENCH_VOLUME_NAME,
        "volume_relpath": f"/{run_rel}",
        "artifact_dir": run_dir,
        "device": _device_report(requested_gpu),
        "tilelang": _tilelang_report(),
        "settings": {
            "shape_csv": shape_csv,
            "variant_csv": variant_csv,
            "warmup": warmup,
            "iters": iters,
            "torch_profile": torch_profile,
            "variants": _selected_variants(variant_csv),
        },
        "shapes": [],
    }

    selected_variants = _selected_variants(variant_csv)
    for shape in _selected_shapes(shape_csv):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        shape_dir = os.path.join(run_dir, shape.name)
        os.makedirs(shape_dir, exist_ok=True)
        inputs = _make_inputs(shape)
        shape_result: dict[str, Any] = {
            "shape": asdict(shape),
            "estimated_tensor_bytes": _shape_bytes_estimate(shape),
            "variants": [],
            "status": "ok",
        }
        for variant in selected_variants:
            _reset_mamba_imports()
            torch.cuda.empty_cache()
            variant_result = _benchmark_variant(
                variant,
                shape,
                inputs,
                shape_dir,
                warmup=warmup,
                iters=iters,
                torch_profile=torch_profile,
            )
            shape_result["variants"].append(variant_result)
            if variant_result.get("status") != "ok":
                shape_result["status"] = "variant_failed"
        shape_result["comparison"] = _compare_shape(shape_result)
        shape_result["variants"] = [_strip_tensors(item) for item in shape_result["variants"]]
        report["shapes"].append(shape_result)

    report["artifacts"] = {
        "report_json": os.path.join(run_dir, "report.json"),
        "summary_json": os.path.join(run_dir, "summary.json"),
        "summary_csv": os.path.join(run_dir, "summary.csv"),
    }
    summary = _summarize_report(report)
    with open(report["artifacts"]["report_json"], "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    with open(report["artifacts"]["summary_json"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    _write_summary_csv(summary, report["artifacts"]["summary_csv"])
    bench_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_id: str | None = None,
    shape_csv: str = "representative",
    variant_csv: str = "baseline,stage2_bf1_bb0,streaming_live_set_bf1_bb0,streaming_live_set_bf1_bb1",
    warmup: int = 2,
    iters: int = 8,
    torch_profile: bool = False,
) -> None:
    result = run_benchmark.remote(GPU_SPEC, run_id, shape_csv, variant_csv, warmup, iters, torch_profile)
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
