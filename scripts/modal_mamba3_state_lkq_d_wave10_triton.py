"""Modal H200 runner for the wave10 Triton state/LKQ/D prototype."""

from __future__ import annotations

import json
import os
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200")
APP_NAME = "cppmega-mamba3-state-lkq-d-wave10-triton"
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
    block_p: int,
    lkq_apply_dtype: str,
    iters: int,
    warmup: int,
    skip_reference: bool,
) -> dict[str, Any]:
    import importlib.util
    import pathlib
    import sys
    import traceback

    try:
        bench_dir = pathlib.Path(CPPMEGA_ROOT) / "upstream_prs/examples/13_tilelang_floormod_dbz"
        sys.path.insert(0, str(bench_dir))
        path = bench_dir / "rr_diag_wave10_state_lkq_d_triton.py"
        print(f"loading {path}", flush=True)
        spec = importlib.util.spec_from_file_location("rr_diag_wave10_state_lkq_d_triton", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rr_diag_wave10_state_lkq_d_triton"] = mod
        spec.loader.exec_module(mod)
        print(
            f"module loaded; starting shape={shape} block_p={block_p} "
            f"lkq_apply_dtype={lkq_apply_dtype}",
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
            device="cuda",
            seed=20260430,
            warmup=warmup,
            iters=iters,
            block_p=block_p,
            num_warps=8,
            lkq_apply_dtype=lkq_apply_dtype,
            reduce_block_p=64,
            reduce_block_chunks=256,
            reduce_num_warps=8,
            skip_reference=skip_reference,
        )
        return mod.run(args)
    except BaseException:
        traceback.print_exc()
        raise


@app.local_entrypoint()
def main(
    shape_csv: str = "smoke,productionish",
    block_p_csv: str = "64,128",
    lkq_apply_dtype_csv: str = "fp32",
    iters: int = 10,
    warmup: int = 3,
    skip_reference: bool = False,
) -> None:
    print(
        f"image={GHCR_REF} gpu={GPU_SPEC} shapes={shape_csv} "
        f"block_p={block_p_csv} lkq_apply_dtype={lkq_apply_dtype_csv}"
    )
    results = []
    shapes = [part.strip() for part in shape_csv.split(",") if part.strip()]
    block_ps = [int(part.strip()) for part in block_p_csv.split(",") if part.strip()]
    lkq_apply_dtypes = [part.strip() for part in lkq_apply_dtype_csv.split(",") if part.strip()]
    for shape in shapes:
        for block_p in block_ps:
            for lkq_apply_dtype in lkq_apply_dtypes:
                result = run_remote.remote(shape, block_p, lkq_apply_dtype, iters, warmup, skip_reference)
                results.append(result)
                print(json.dumps(result, indent=2, sort_keys=True))
    print("SUMMARY_JSON=" + json.dumps(results, sort_keys=True))
