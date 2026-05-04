"""Modal H200 runner for the wave7 chunk-owner CUDA bwd_bwd prototype."""

from __future__ import annotations

import json
import os
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200")
APP_NAME = "cppmega-mamba3-rr-diag-wave7-chunk-owner-cuda"
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
def run_remote(
    shape: str,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    import importlib.util
    import pathlib
    import sys

    bench_dir = pathlib.Path(CPPMEGA_ROOT) / "upstream_prs/examples/13_tilelang_floormod_dbz"
    sys.path.insert(0, str(bench_dir))
    path = bench_dir / "rr_diag_wave7_chunk_owner_cuda.py"
    spec = importlib.util.spec_from_file_location("rr_diag_wave7_chunk_owner_cuda", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rr_diag_wave7_chunk_owner_cuda"] = mod
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
    return mod.run(args)


@app.local_entrypoint()
def main(
    shape_csv: str = "smoke,productionish",
    iters: int = 20,
    warmup: int = 5,
) -> None:
    print(f"image={GHCR_REF} gpu={GPU_SPEC} shapes={shape_csv}")
    results = []
    for shape in [part.strip() for part in shape_csv.split(",") if part.strip()]:
        result = run_remote.remote(shape, iters, warmup)
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True))
    print("SUMMARY_JSON=" + json.dumps(results, sort_keys=True))
