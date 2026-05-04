"""Modal benchmark for Mamba3 bwd_bwd on-chip PsiV reuse variants.

This runner reuses the stage2 force-nonTMA benchmark harness and adds two
incremental bwd_bwd-only patches. Each candidate applies:

1. mamba3_bwd_stage2_force_nontma.patch
2. one mamba3_bwd_bwd_onchip_*.patch

Run:

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 900s \
        modal run scripts/modal_mamba3_bwd_bwd_onchip_reuse.py \
        --shape-csv productionish --warmup 2 --iters 8
"""

from __future__ import annotations

import json
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import modal

_BASE_SCRIPT = Path(__file__).with_name("modal_mamba3_stage2_force_nontma_benchmark.py")
if not _BASE_SCRIPT.exists():
    _BASE_SCRIPT = Path("/opt/cppmega/scripts/modal_mamba3_stage2_force_nontma_benchmark.py")
_BASE_SPEC = importlib.util.spec_from_file_location("modal_mamba3_stage2_force_nontma_benchmark_base", _BASE_SCRIPT)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"could not import benchmark base from {_BASE_SCRIPT}")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = base
_BASE_SPEC.loader.exec_module(base)

APP_NAME = "cppmega-mamba3-bwd-bwd-onchip-reuse"
BENCH_PREFIX = "mamba3_bwd_bwd_onchip_reuse"
PATCH_DIR = f"{base.CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz"

VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": dict(base.VARIANTS["baseline"]),
    "stage2_force_nontma": dict(base.VARIANTS["stage2_force_nontma"]),
    "onchip_psiv_direct": {
        **base.VARIANTS["stage2_force_nontma"],
        "patch": "stage2_plus_onchip",
        "patch_chain": [
            "mamba3_bwd_stage2_force_nontma.patch",
            "mamba3_bwd_bwd_onchip_psiv_direct.patch",
        ],
    },
    "onchip_psiv_direct_dpsi_acc": {
        **base.VARIANTS["stage2_force_nontma"],
        "patch": "stage2_plus_onchip",
        "patch_chain": [
            "mamba3_bwd_stage2_force_nontma.patch",
            "mamba3_bwd_bwd_onchip_psiv_direct_dpsi_acc.patch",
        ],
    },
}


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        base.GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    img = img.env(
        {
            "GHCR_REPO": base.GHCR_REPO,
            "GHCR_TAG": base.GHCR_TAG,
            "CPPMEGA_IMAGE_REF": base.GHCR_REF,
        }
    )
    img = img.add_local_dir("cppmega", f"{base.CPPMEGA_ROOT}/cppmega", copy=True)
    img = img.add_local_dir("scripts", f"{base.CPPMEGA_ROOT}/scripts", copy=True)
    img = img.add_local_dir(
        "upstream_prs/examples/13_tilelang_floormod_dbz",
        PATCH_DIR,
        copy=True,
    )
    img = img.add_local_dir(
        "/home/dave/state-spaces-mamba/mamba_ssm",
        f"{base.SOURCE_ROOT}/mamba_ssm",
        copy=True,
    )
    return img


app = modal.App(APP_NAME)


def _apply_patch_file(dst: str, patch_name: str) -> dict[str, Any]:
    patch_file = f"{PATCH_DIR}/{patch_name}"
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


def _prepare_variant(variant: str) -> tuple[str, dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    src = f"{base.SOURCE_ROOT}/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"
    work = tempfile.mkdtemp(prefix=f"cppmega_mamba3_bwd_bwd_onchip_{variant}_")
    dst = os.path.join(work, "mamba3_mimo_bwd.py")
    shutil.copy(src, dst)

    cfg = VARIANTS[variant]
    meta: dict[str, Any] = {"variant": variant, "work": work, "source_path": dst, "patches": []}
    if cfg.get("patch") is None:
        return dst, meta

    if cfg.get("patch") == "stage2_plus_onchip":
        for patch_name in cfg["patch_chain"]:
            patch_meta = _apply_patch_file(dst, str(patch_name))
            meta["patches"].append(patch_meta)
            if patch_meta["patch_rc"] != 0:
                meta.update(patch_meta)
                return dst, meta
        meta["patch"] = "+".join(cfg["patch_chain"])
        meta["patch_rc"] = 0
        return dst, meta

    patch_file = str(cfg.get("patch_file", "mamba3_bwd_stage2_force_nontma.patch"))
    patch_meta = _apply_patch_file(dst, patch_file)
    meta.update({"patch": patch_file, **patch_meta})
    return dst, meta


base.VARIANTS = VARIANTS
base.BENCH_PREFIX = BENCH_PREFIX
base._prepare_variant = _prepare_variant


@app.function(image=_image(), gpu=base.GPU_SPEC, timeout=1200, volumes={base.BENCH_ROOT: base.bench_volume})
def run_benchmark(
    requested_gpu: str,
    run_id: str | None,
    shape_csv: str,
    variant_csv: str,
    warmup: int,
    iters: int,
    torch_profile: bool,
) -> dict[str, Any]:
    return base.run_benchmark.local(
        requested_gpu,
        run_id,
        shape_csv,
        variant_csv,
        warmup,
        iters,
        torch_profile,
    )


@app.local_entrypoint()
def main(
    run_id: str | None = None,
    shape_csv: str = "representative",
    variant_csv: str = "baseline,stage2_force_nontma,onchip_psiv_direct,onchip_psiv_direct_dpsi_acc",
    warmup: int = 2,
    iters: int = 8,
    torch_profile: bool = False,
) -> None:
    result = run_benchmark.remote(base.GPU_SPEC, run_id, shape_csv, variant_csv, warmup, iters, torch_profile)
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
