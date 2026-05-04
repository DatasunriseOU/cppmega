"""Modal H200 runner for the Wave2 tile-pruned monolithic Triton model."""

from __future__ import annotations

import json
import os
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200")
APP_NAME = "cppmega-mamba3-mono-triton-pruned-wave2"
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
    block_p: int,
    num_warps: int,
    iters: int,
    warmup: int,
    check_torch: bool,
    bench_torch: bool,
    torch_iters: int,
    torch_warmup: int,
    torch_reference_batch: int,
) -> dict[str, Any]:
    import importlib.util
    import pathlib
    import sys
    import traceback

    try:
        bench_dir = pathlib.Path(CPPMEGA_ROOT) / "upstream_prs/examples/13_tilelang_floormod_dbz"
        sys.path.insert(0, str(bench_dir))
        path = bench_dir / "rr_diag_wave2_mono_triton_pruned_model.py"
        print(f"loading {path}", flush=True)
        spec = importlib.util.spec_from_file_location("rr_diag_wave2_mono_triton_pruned_model", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rr_diag_wave2_mono_triton_pruned_model"] = mod
        spec.loader.exec_module(mod)
        print(
            f"module loaded; starting shape={shape} block_p={block_p} num_warps={num_warps} "
            f"check_torch={check_torch} bench_torch={bench_torch}",
            flush=True,
        )
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
            handoff_dtype="bf16",
            device="cuda",
            seed=20260430,
            warmup=warmup,
            iters=iters,
            block_p=block_p,
            num_warps=num_warps,
            check_torch=check_torch,
            bench_torch=bench_torch,
            torch_warmup=torch_warmup,
            torch_iters=torch_iters,
            torch_reference_batch=torch_reference_batch,
        )
        return mod.run(args)
    except BaseException:
        traceback.print_exc()
        raise


@app.local_entrypoint()
def main(
    shape_csv: str = "smoke,productionish",
    block_p: int = 128,
    num_warps_csv: str = "4",
    iters: int = 10,
    warmup: int = 3,
    check_torch_shapes: str = "smoke",
    bench_torch_shapes: str = "",
    torch_iters: int = 3,
    torch_warmup: int = 1,
    torch_reference_batch: int = 512,
) -> None:
    print(f"image={GHCR_REF} gpu={GPU_SPEC} shapes={shape_csv} block_p={block_p} num_warps={num_warps_csv}")
    results = []
    shapes = [part.strip() for part in shape_csv.split(",") if part.strip()]
    num_warps_values = [int(part.strip()) for part in num_warps_csv.split(",") if part.strip()]
    check_shapes = {part.strip() for part in check_torch_shapes.split(",") if part.strip()}
    bench_shapes = {part.strip() for part in bench_torch_shapes.split(",") if part.strip()}
    for shape in shapes:
        for num_warps in num_warps_values:
            result = run_remote.remote(
                shape,
                block_p,
                num_warps,
                iters,
                warmup,
                shape in check_shapes,
                shape in bench_shapes,
                torch_iters,
                torch_warmup,
                torch_reference_batch,
            )
            results.append(result)
            print(json.dumps(result, indent=2, sort_keys=True))
    print("SUMMARY_JSON=" + json.dumps(results, sort_keys=True))
