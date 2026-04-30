"""Modal runner for wave10 CUDA bwd_bwd DMIMO_V tuning."""

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
APP_NAME = "cppmega-mamba3-rr-diag-wave10-cuda-tuning-" + re.sub(r"[^0-9A-Za-z]+", "-", GPU_SPEC).lower()
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


def _parse_variant(variant: str) -> dict[str, int]:
    match = re.fullmatch(r"t(?P<threads>\d+)p(?P<p_tile>\d+)u(?P<unroll>\d+)(?:b(?P<broadcast_qk>[01]))?", variant)
    if not match:
        raise ValueError(f"variant must look like t128p32u1 or t128p32u1b1, got {variant!r}")
    parsed = {key: int(value) for key, value in match.groupdict(default="0").items()}
    return parsed


app = modal.App(APP_NAME)


@app.function(image=_image(), gpu=GPU_SPEC, timeout=20 * 60)
def run_remote(
    shape: str,
    variant: str,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    import importlib.util
    import pathlib
    import sys
    import traceback

    parsed = _parse_variant(variant)
    os.environ["RR_DIAG_THREADS"] = str(parsed["threads"])
    os.environ["RR_DIAG_DMIMO_P_TILE"] = str(parsed["p_tile"])
    os.environ["RR_DIAG_DMIMO_UNROLL"] = str(parsed["unroll"])
    os.environ["RR_DIAG_DMIMO_BROADCAST_QK"] = str(parsed["broadcast_qk"])
    os.environ["RR_DIAG_CUDA_EXT_SUFFIX"] = variant

    for name in [
        "rr_diag_cuda_extension",
        "rr_diag_wave6_inlaunch_cuda",
        "rr_diag_wave7_chunk_owner_cuda",
        "rr_diag_wave8_chunk_owner_cuda",
    ]:
        sys.modules.pop(name, None)

    bench_dir = pathlib.Path(CPPMEGA_ROOT) / "upstream_prs/examples/13_tilelang_floormod_dbz"
    sys.path.insert(0, str(bench_dir))
    path = bench_dir / "rr_diag_wave8_chunk_owner_cuda.py"
    module_name = "rr_diag_wave10_" + re.sub(r"[^0-9A-Za-z_]+", "_", f"{variant}_{shape}")
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
    result["variant"] = variant
    result["variant_config"] = parsed
    return result


@app.local_entrypoint()
def main(
    shape_csv: str = "productionish",
    variants_csv: str = "t256p32u1",
    iters: int = 20,
    warmup: int = 5,
) -> None:
    print(f"image={GHCR_REF} gpu={GPU_SPEC} shapes={shape_csv} variants={variants_csv}")
    results = []
    shapes = [part.strip() for part in shape_csv.split(",") if part.strip()]
    variants = [part.strip() for part in variants_csv.split(",") if part.strip()]
    for variant in variants:
        for shape in shapes:
            result = run_remote.remote(shape, variant, iters, warmup)
            results.append(result)
            print(json.dumps(result, indent=2, sort_keys=True))
    print("SUMMARY_JSON=" + json.dumps(results, sort_keys=True))
