"""Modal H200 probe for minimal TileLang DSTATES_PTILE scratch copy legality.

The probe isolates global scratch <-> shared/fragment copy forms from the full
Mamba3 bwd_bwd kernel.  It checks rank-2 flattened scratch descriptors with
dynamic p_start indexing and rank-5 DSTATES_PTILE-style descriptors, both with
and without per-copy ``disable_tma=True``.

Run:

    python -m py_compile scripts/modal_tilelang_dstates_ptile_copy_probe.py
    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:1 timeout 15m modal run \
        scripts/modal_tilelang_dstates_ptile_copy_probe.py
"""

import hashlib
import json
import os
import re
import traceback
from typing import Any

import modal
import tilelang
import tilelang.language as T

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:1")

APP_NAME = "cppmega-tilelang-dstates-ptile-copy-probe"


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    return img.env({"CPPMEGA_IMAGE_REF": GHCR_REF})


app = modal.App(APP_NAME)


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def rank2_dynamic_shared_tma_on(
    B=1,
    H=2,
    N=16,
    P=128,
    P_TILE=64,
    dtype="bfloat16",
    disable_copy_tma=False,
):
    n_p_tiles = P // P_TILE

    @T.prim_func
    def kernel(
        SRC: T.Tensor([B * H * N, P], dtype),  # type: ignore
        DST: T.Tensor([B * H * N, P], dtype),  # type: ignore
    ):
        with T.Kernel(n_p_tiles, B * H, threads=128) as (p_block, bh):
            tile_shared = T.alloc_shared([N, P_TILE], dtype)
            tile_frag = T.alloc_fragment([N, P_TILE], dtype)
            row_start = bh * N
            p_start = p_block * P_TILE
            if disable_copy_tma:
                T.copy(SRC[row_start : row_start + N, p_start : p_start + P_TILE], tile_shared, disable_tma=True)
                T.copy(tile_shared, tile_frag)
                T.copy(tile_frag, tile_shared)
                T.copy(tile_shared, DST[row_start : row_start + N, p_start : p_start + P_TILE], disable_tma=True)
            else:
                T.copy(SRC[row_start : row_start + N, p_start : p_start + P_TILE], tile_shared)
                T.copy(tile_shared, tile_frag)
                T.copy(tile_frag, tile_shared)
                T.copy(tile_shared, DST[row_start : row_start + N, p_start : p_start + P_TILE])

    return kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def rank2_dynamic_shared_tma_off(
    B=1,
    H=2,
    N=16,
    P=128,
    P_TILE=64,
    dtype="bfloat16",
    disable_copy_tma=False,
):
    n_p_tiles = P // P_TILE

    @T.prim_func
    def kernel(
        SRC: T.Tensor([B * H * N, P], dtype),  # type: ignore
        DST: T.Tensor([B * H * N, P], dtype),  # type: ignore
    ):
        with T.Kernel(n_p_tiles, B * H, threads=128) as (p_block, bh):
            tile_shared = T.alloc_shared([N, P_TILE], dtype)
            tile_frag = T.alloc_fragment([N, P_TILE], dtype)
            row_start = bh * N
            p_start = p_block * P_TILE
            if disable_copy_tma:
                T.copy(SRC[row_start : row_start + N, p_start : p_start + P_TILE], tile_shared, disable_tma=True)
                T.copy(tile_shared, tile_frag)
                T.copy(tile_frag, tile_shared)
                T.copy(tile_shared, DST[row_start : row_start + N, p_start : p_start + P_TILE], disable_tma=True)
            else:
                T.copy(SRC[row_start : row_start + N, p_start : p_start + P_TILE], tile_shared)
                T.copy(tile_shared, tile_frag)
                T.copy(tile_frag, tile_shared)
                T.copy(tile_shared, DST[row_start : row_start + N, p_start : p_start + P_TILE])

    return kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def rank2_dynamic_fragment_tma_on(
    B=1,
    H=2,
    N=16,
    P=128,
    P_TILE=64,
    dtype="bfloat16",
    disable_copy_tma=False,
):
    n_p_tiles = P // P_TILE

    @T.prim_func
    def kernel(
        SRC: T.Tensor([B * H * N, P], dtype),  # type: ignore
        DST: T.Tensor([B * H * N, P], dtype),  # type: ignore
    ):
        with T.Kernel(n_p_tiles, B * H, threads=128) as (p_block, bh):
            tile_frag = T.alloc_fragment([N, P_TILE], dtype)
            row_start = bh * N
            p_start = p_block * P_TILE
            if disable_copy_tma:
                T.copy(SRC[row_start : row_start + N, p_start : p_start + P_TILE], tile_frag, disable_tma=True)
                T.copy(tile_frag, DST[row_start : row_start + N, p_start : p_start + P_TILE], disable_tma=True)
            else:
                T.copy(SRC[row_start : row_start + N, p_start : p_start + P_TILE], tile_frag)
                T.copy(tile_frag, DST[row_start : row_start + N, p_start : p_start + P_TILE])

    return kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def rank2_dynamic_fragment_tma_off(
    B=1,
    H=2,
    N=16,
    P=128,
    P_TILE=64,
    dtype="bfloat16",
    disable_copy_tma=False,
):
    n_p_tiles = P // P_TILE

    @T.prim_func
    def kernel(
        SRC: T.Tensor([B * H * N, P], dtype),  # type: ignore
        DST: T.Tensor([B * H * N, P], dtype),  # type: ignore
    ):
        with T.Kernel(n_p_tiles, B * H, threads=128) as (p_block, bh):
            tile_frag = T.alloc_fragment([N, P_TILE], dtype)
            row_start = bh * N
            p_start = p_block * P_TILE
            if disable_copy_tma:
                T.copy(SRC[row_start : row_start + N, p_start : p_start + P_TILE], tile_frag, disable_tma=True)
                T.copy(tile_frag, DST[row_start : row_start + N, p_start : p_start + P_TILE], disable_tma=True)
            else:
                T.copy(SRC[row_start : row_start + N, p_start : p_start + P_TILE], tile_frag)
                T.copy(tile_frag, DST[row_start : row_start + N, p_start : p_start + P_TILE])

    return kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def rank5_static_shared_tma_on(
    B=1,
    H=2,
    N=16,
    P=128,
    P_TILE=64,
    dtype="bfloat16",
    disable_copy_tma=False,
):
    n_p_tiles = P // P_TILE

    @T.prim_func
    def kernel(
        SRC: T.Tensor([B, H, n_p_tiles, N, P_TILE], dtype),  # type: ignore
        DST: T.Tensor([B, H, n_p_tiles, N, P_TILE], dtype),  # type: ignore
    ):
        with T.Kernel(n_p_tiles, B * H, threads=128) as (p_block, bh):
            tile_shared = T.alloc_shared([N, P_TILE], dtype)
            tile_frag = T.alloc_fragment([N, P_TILE], dtype)
            i_b = bh // H
            i_h = bh - i_b * H
            if disable_copy_tma:
                T.copy(SRC[i_b, i_h, p_block, :, :], tile_shared, disable_tma=True)
                T.copy(tile_shared, tile_frag)
                T.copy(tile_frag, tile_shared)
                T.copy(tile_shared, DST[i_b, i_h, p_block, :, :], disable_tma=True)
            else:
                T.copy(SRC[i_b, i_h, p_block, :, :], tile_shared)
                T.copy(tile_shared, tile_frag)
                T.copy(tile_frag, tile_shared)
                T.copy(tile_shared, DST[i_b, i_h, p_block, :, :])

    return kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def rank5_static_shared_tma_off(
    B=1,
    H=2,
    N=16,
    P=128,
    P_TILE=64,
    dtype="bfloat16",
    disable_copy_tma=False,
):
    n_p_tiles = P // P_TILE

    @T.prim_func
    def kernel(
        SRC: T.Tensor([B, H, n_p_tiles, N, P_TILE], dtype),  # type: ignore
        DST: T.Tensor([B, H, n_p_tiles, N, P_TILE], dtype),  # type: ignore
    ):
        with T.Kernel(n_p_tiles, B * H, threads=128) as (p_block, bh):
            tile_shared = T.alloc_shared([N, P_TILE], dtype)
            tile_frag = T.alloc_fragment([N, P_TILE], dtype)
            i_b = bh // H
            i_h = bh - i_b * H
            if disable_copy_tma:
                T.copy(SRC[i_b, i_h, p_block, :, :], tile_shared, disable_tma=True)
                T.copy(tile_shared, tile_frag)
                T.copy(tile_frag, tile_shared)
                T.copy(tile_shared, DST[i_b, i_h, p_block, :, :], disable_tma=True)
            else:
                T.copy(SRC[i_b, i_h, p_block, :, :], tile_shared)
                T.copy(tile_shared, tile_frag)
                T.copy(tile_frag, tile_shared)
                T.copy(tile_shared, DST[i_b, i_h, p_block, :, :])

    return kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def rank5_static_fragment_tma_on(
    B=1,
    H=2,
    N=16,
    P=128,
    P_TILE=64,
    dtype="bfloat16",
    disable_copy_tma=False,
):
    n_p_tiles = P // P_TILE

    @T.prim_func
    def kernel(
        SRC: T.Tensor([B, H, n_p_tiles, N, P_TILE], dtype),  # type: ignore
        DST: T.Tensor([B, H, n_p_tiles, N, P_TILE], dtype),  # type: ignore
    ):
        with T.Kernel(n_p_tiles, B * H, threads=128) as (p_block, bh):
            tile_frag = T.alloc_fragment([N, P_TILE], dtype)
            i_b = bh // H
            i_h = bh - i_b * H
            if disable_copy_tma:
                T.copy(SRC[i_b, i_h, p_block, :, :], tile_frag, disable_tma=True)
                T.copy(tile_frag, DST[i_b, i_h, p_block, :, :], disable_tma=True)
            else:
                T.copy(SRC[i_b, i_h, p_block, :, :], tile_frag)
                T.copy(tile_frag, DST[i_b, i_h, p_block, :, :])

    return kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def rank5_static_fragment_tma_off(
    B=1,
    H=2,
    N=16,
    P=128,
    P_TILE=64,
    dtype="bfloat16",
    disable_copy_tma=False,
):
    n_p_tiles = P // P_TILE

    @T.prim_func
    def kernel(
        SRC: T.Tensor([B, H, n_p_tiles, N, P_TILE], dtype),  # type: ignore
        DST: T.Tensor([B, H, n_p_tiles, N, P_TILE], dtype),  # type: ignore
    ):
        with T.Kernel(n_p_tiles, B * H, threads=128) as (p_block, bh):
            tile_frag = T.alloc_fragment([N, P_TILE], dtype)
            i_b = bh // H
            i_h = bh - i_b * H
            if disable_copy_tma:
                T.copy(SRC[i_b, i_h, p_block, :, :], tile_frag, disable_tma=True)
                T.copy(tile_frag, DST[i_b, i_h, p_block, :, :], disable_tma=True)
            else:
                T.copy(SRC[i_b, i_h, p_block, :, :], tile_frag)
                T.copy(tile_frag, DST[i_b, i_h, p_block, :, :])

    return kernel


KERNELS = {
    ("rank2_dynamic_shared", True): rank2_dynamic_shared_tma_on,
    ("rank2_dynamic_shared", False): rank2_dynamic_shared_tma_off,
    ("rank2_dynamic_fragment", True): rank2_dynamic_fragment_tma_on,
    ("rank2_dynamic_fragment", False): rank2_dynamic_fragment_tma_off,
    ("rank5_static_shared", True): rank5_static_shared_tma_on,
    ("rank5_static_shared", False): rank5_static_shared_tma_off,
    ("rank5_static_fragment", True): rank5_static_fragment_tma_on,
    ("rank5_static_fragment", False): rank5_static_fragment_tma_off,
}

DEFAULT_FORMS = ",".join(sorted({name for name, _ in KERNELS}))


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

    return {
        "module_file": getattr(tilelang, "__file__", None),
        "module_version": getattr(tilelang, "__version__", None),
        "package_version": importlib.metadata.version("tilelang"),
    }


def _parse_bool_csv(raw: str) -> list[bool]:
    values: list[bool] = []
    for item in raw.split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value in {"1", "true", "yes", "on"}:
            values.append(True)
        elif value in {"0", "false", "no", "off"}:
            values.append(False)
        else:
            raise ValueError(f"invalid bool value {item!r}")
    if not values:
        raise ValueError("empty bool csv")
    return values


def _parse_forms(raw: str) -> list[str]:
    if raw.strip() == "all":
        raw = DEFAULT_FORMS
    forms = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(forms) - {name for name, _ in KERNELS})
    if unknown:
        raise ValueError(f"unknown forms {unknown}; choose from {DEFAULT_FORMS}")
    if not forms:
        raise ValueError("at least one form is required")
    return forms


def _source_markers(kernel: Any) -> dict[str, Any]:
    source = kernel.get_kernel_source()
    launch_bounds = sorted(set(re.findall(r"__launch_bounds__\((\d+),\s*(\d+)\)", source)))
    return {
        "source_chars": len(source),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "tma_load_count": source.count("tl::tma_load"),
        "tma_store_count": source.count("tl::tma_store"),
        "mbarrier_wait_count": source.count("mbarrier_wait"),
        "launch_bounds": launch_bounds,
        "has_dynamic_p_start": "blockIdx.x) * 64" in source or "p_block * P_TILE" in source,
    }


def _make_tensors(form: str, B: int, H: int, N: int, P: int, P_TILE: int, dtype: str) -> tuple[Any, Any]:
    import torch

    torch_dtype = getattr(torch, dtype)
    n_p_tiles = P // P_TILE
    if form.startswith("rank2"):
        shape = (B * H * N, P)
    else:
        shape = (B, H, n_p_tiles, N, P_TILE)
    src = torch.arange(0, math_prod(shape), device="cuda", dtype=torch.float32).reshape(shape).to(torch_dtype)
    dst = torch.zeros_like(src)
    return src, dst


def math_prod(values: tuple[int, ...]) -> int:
    product = 1
    for value in values:
        product *= value
    return product


def _run_one(
    form: str,
    tma_lower: bool,
    disable_copy_tma: bool,
    B: int,
    H: int,
    N: int,
    P: int,
    P_TILE: int,
    dtype: str,
) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "form": form,
        "tma_lower_enabled": tma_lower,
        "per_copy_disable_tma": disable_copy_tma,
        "shape": {"B": B, "H": H, "N": N, "P": P, "P_TILE": P_TILE},
        "dtype": dtype,
    }
    try:
        factory = KERNELS[(form, tma_lower)]
        kernel = factory(B, H, N, P, P_TILE, dtype, disable_copy_tma)
        result["compile_status"] = "ok"
        result["source"] = _source_markers(kernel)
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "compile_status": "failed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback_tail": traceback.format_exc()[-4000:],
            }
        )
        return result

    try:
        src, dst = _make_tensors(form, B, H, N, P, P_TILE, dtype)
        kernel(src, dst)
        torch.cuda.synchronize()
        diff = (src.float() - dst.float()).abs()
        result.update(
            {
                "run_status": "ok",
                "max_abs": float(diff.max().item()),
                "allclose_0": bool(torch.allclose(src.float(), dst.float(), rtol=0.0, atol=0.0)),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "run_status": "failed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback_tail": traceback.format_exc()[-4000:],
            }
        )
    return result


@app.function(image=_image(), gpu=GPU_SPEC, timeout=1200)
def run_probe(
    requested_gpu: str,
    forms_csv: str,
    tma_lower_csv: str,
    disable_copy_tma_csv: str,
    B: int,
    H: int,
    N: int,
    P: int,
    P_TILE: int,
    dtype: str,
) -> dict[str, Any]:
    forms = _parse_forms(forms_csv)
    tma_lowers = _parse_bool_csv(tma_lower_csv)
    disable_copy_tmas = _parse_bool_csv(disable_copy_tma_csv)
    return {
        "app_name": APP_NAME,
        "device": _device_report(requested_gpu),
        "tilelang": _tilelang_report(),
        "settings": {
            "forms": forms,
            "tma_lower_enabled": tma_lowers,
            "per_copy_disable_tma": disable_copy_tmas,
            "B": B,
            "H": H,
            "N": N,
            "P": P,
            "P_TILE": P_TILE,
            "dtype": dtype,
        },
        "results": [
            _run_one(form, tma_lower, disable_copy_tma, B, H, N, P, P_TILE, dtype)
            for form in forms
            for tma_lower in tma_lowers
            for disable_copy_tma in disable_copy_tmas
        ],
    }


@app.local_entrypoint()
def main(
    forms: str = "all",
    tma_lower: str = "true,false",
    disable_copy_tma: str = "false,true",
    b: int = 1,
    h: int = 2,
    n: int = 16,
    p: int = 128,
    p_tile: int = 64,
    dtype: str = "bfloat16",
) -> None:
    result = run_probe.remote(GPU_SPEC, forms, tma_lower, disable_copy_tma, b, h, n, p, p_tile, dtype)
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
