"""Modal H200 benchmark for the isolated Mamba3 bwd_bwd R x R diagonal path.

This does not patch production Mamba.  It benchmarks the companion
``rr_diag_specialization.py`` harness against the current full fused
``dPhiO @ PsiV.T`` diagonal extraction and records correctness for
``DGAMMA_DIAG`` plus the diagonal contributions to ``DK`` and ``DQ``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200")
APP_NAME = "cppmega-mamba3-rr-diag-benchmark"
CPPMEGA_ROOT = "/opt/cppmega"


def _image() -> modal.Image:
    image: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    image = image.env({"CPPMEGA_IMAGE_REF": GHCR_REF})
    image = image.add_local_dir(
        "upstream_prs/examples/13_tilelang_floormod_dbz",
        f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz",
        copy=True,
    )
    return image


app = modal.App(APP_NAME)


@app.function(image=_image(), gpu=GPU_SPEC, timeout=20 * 60)
def run_remote(shape: str, iters: int, warmup: int, triton: bool) -> dict[str, Any]:
    import importlib.util
    import pathlib
    import sys

    path = pathlib.Path(CPPMEGA_ROOT) / "upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_specialization.py"
    spec = importlib.util.spec_from_file_location("rr_diag_specialization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rr_diag_specialization"] = mod
    spec.loader.exec_module(mod)

    presets = {
        "smoke": {"B": 1, "S": 256, "H": 4, "N": 64, "P": 128},
        "representative": {"B": 2, "S": 1024, "H": 16, "N": 64, "P": 128},
        "productionish": {"B": 4, "S": 4096, "H": 32, "N": 64, "P": 128},
    }
    if shape not in presets:
        raise ValueError(f"unknown shape {shape!r}; choose one of {sorted(presets)}")

    args = mod.argparse.Namespace(
        **presets[shape],
        R=4,
        chunk=16,
        dtype="bf16",
        device="cuda",
        seed=20260429,
        warmup=warmup,
        iters=iters,
        triton=triton,
        num_warps=4,
    )
    return mod.run(args)


@app.local_entrypoint()
def main(
    shape_csv: str = "smoke,productionish",
    iters: int = 20,
    warmup: int = 5,
    triton: bool = True,
) -> None:
    print(f"image={GHCR_REF} gpu={GPU_SPEC} shapes={shape_csv}")
    results = []
    for shape in [part.strip() for part in shape_csv.split(",") if part.strip()]:
        result = run_remote.remote(shape, iters, warmup, triton)
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True))
    print("SUMMARY_JSON=" + json.dumps(results, sort_keys=True))
