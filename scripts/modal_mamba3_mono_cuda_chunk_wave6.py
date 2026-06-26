"""Modal runner for the Wave 6 chunk-group owner CUDA prototype."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200")
APP_NAME = "cppmega-mamba3-mono-cuda-chunk-wave6-" + re.sub(r"[^0-9A-Za-z]+", "-", GPU_SPEC).lower()
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
    shape_csv: str,
    iters: int,
    warmup: int,
    chunk_group_size: int,
) -> list[dict[str, Any]]:
    import importlib.util
    import pathlib
    import sys
    import traceback

    os.environ["RR_DIAG_THREADS"] = "256"
    os.environ["RR_DIAG_CUDA_EXT_SUFFIX"] = "mono_wave6_chunk_group_owner"
    os.environ["RR_DIAG_CUDA_VERBOSE_BUILD"] = "0"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

    for module in (
        "rr_mono_cuda_chunk_wave1",
        "rr_mono_cuda_chunk_wave2",
        "rr_mono_cuda_chunk_wave6",
        "rr_mono_cuda_chunk_wave6_extension",
    ):
        sys.modules.pop(module, None)

    bench_dir = pathlib.Path(CPPMEGA_ROOT) / "upstream_prs/examples/13_tilelang_floormod_dbz"
    sys.path.insert(0, str(bench_dir))
    path = bench_dir / "rr_mono_cuda_chunk_wave6.py"
    module_name = "rr_mono_cuda_chunk_wave6_" + re.sub(r"[^0-9A-Za-z_]+", "_", shape_csv)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    results = []
    for shape in [part.strip() for part in shape_csv.split(",") if part.strip()]:
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
            chunk_group_size=chunk_group_size,
            dtype="bf16",
            device="cuda",
            seed=20260430,
            warmup=warmup,
            iters=iters,
        )
        try:
            results.append(mod.run(args))
        except BaseException:
            traceback.print_exc()
            raise
    for result in results:
        print(json.dumps(result, indent=2, sort_keys=True))
    print("SUMMARY_JSON=" + json.dumps(results, sort_keys=True))
    return results


@app.local_entrypoint()
def main(
    shape_csv: str = "smoke",
    iters: int = 3,
    warmup: int = 1,
    chunk_group_size: int = 8,
    background: bool = False,
) -> None:
    print(f"image={GHCR_REF} gpu={GPU_SPEC} shapes={shape_csv} chunk_group_size={chunk_group_size}")
    if background:
        call = run_remote.spawn(shape_csv, iters, warmup, chunk_group_size)
        print(f"FUNCTION_CALL_ID={call.object_id}")
        return
    results = run_remote.remote(shape_csv, iters, warmup, chunk_group_size)
    for result in results:
        print(json.dumps(result, indent=2, sort_keys=True))
    print("SUMMARY_JSON=" + json.dumps(results, sort_keys=True))
