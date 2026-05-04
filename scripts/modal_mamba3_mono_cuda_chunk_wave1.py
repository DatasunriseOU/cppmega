"""Modal runner for the Wave 1 monolithic CUDA chunk-owner prototype."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200")
APP_NAME = "cppmega-mamba3-mono-cuda-chunk-wave1-" + re.sub(r"[^0-9A-Za-z]+", "-", GPU_SPEC).lower()
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


@app.function(image=_image(), gpu=GPU_SPEC, timeout=30 * 60)
def run_remote(
    shape: str,
    iters: int,
    warmup: int,
    mono_p_tile: int,
) -> dict[str, Any]:
    import importlib.util
    import pathlib
    import sys
    import traceback

    os.environ["RR_DIAG_THREADS"] = "256"
    os.environ["RR_DIAG_DMIMO_P_TILE"] = "32"
    os.environ["RR_DIAG_DMIMO_UNROLL"] = "1"
    os.environ["RR_DIAG_DMIMO_BROADCAST_QK"] = "0"
    os.environ["RR_DIAG_MONO_P_TILE"] = str(mono_p_tile)
    os.environ["RR_DIAG_CUDA_EXT_SUFFIX"] = f"mono_wave1_m{mono_p_tile}_{shape}"
    os.environ["RR_DIAG_CUDA_VERBOSE_BUILD"] = "1"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

    sys.modules.pop("rr_diag_cuda_extension", None)
    sys.modules.pop("rr_mono_cuda_chunk_wave1", None)

    bench_dir = pathlib.Path(CPPMEGA_ROOT) / "upstream_prs/examples/13_tilelang_floormod_dbz"
    sys.path.insert(0, str(bench_dir))
    path = bench_dir / "rr_mono_cuda_chunk_wave1.py"
    module_name = "rr_mono_cuda_chunk_wave1_" + re.sub(r"[^0-9A-Za-z_]+", "_", f"{shape}_{mono_p_tile}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    args = mod.argparse.Namespace(
        shape=shape,
        B=1,
        S=256,
        H=4,
        G=1,
        N=64,
        P=128,
        R=4,
        chunk=16,
        dtype="bf16",
        device="cuda",
        seed=20260430,
        warmup=warmup,
        iters=iters,
    )
    try:
        result = mod.run(args)
    except BaseException:
        traceback.print_exc()
        raise
    result["mono_p_tile"] = mono_p_tile
    return result


@app.local_entrypoint()
def main(
    shape_csv: str = "smoke,productionish",
    iters: int = 10,
    warmup: int = 3,
    mono_p_tile: int = 32,
) -> None:
    print(f"image={GHCR_REF} gpu={GPU_SPEC} shapes={shape_csv} mono_p_tile={mono_p_tile}")
    results = []
    for shape in [part.strip() for part in shape_csv.split(",") if part.strip()]:
        result = run_remote.remote(shape, iters, warmup, mono_p_tile)
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True))
    print("SUMMARY_JSON=" + json.dumps(results, sort_keys=True))
