"""Modal B200/SM100 probe for Mamba3 MIMO kernel paths.

This harness is intentionally separate from the Hopper Hoist-PsiV dry-run:

* only requests B200-family Modal specs (`B200+:2`, `B200:2`, `B200:1`);
* does not patch `mamba_ssm` or cppmega production defaults;
* uses source-overlay Mamba3 TileLang to measure today's baseline on SM100;
* records B200-specific go/no-go signals for TileLang, cuTile/CuTe DSL,
  Hugging Face kernel replacement candidates, and Hoist-PsiV write cost.

Examples:

    modal run scripts/modal_mamba3_b200_paths.py
    CPPMEGA_MAMBA3_B200_SPECS=B200:1 modal run scripts/modal_mamba3_b200_paths.py
    CPPMEGA_MAMBA3_B200_SPECS=B200+:2,B200:2,B200:1 modal run scripts/modal_mamba3_b200_paths.py
    GHCR_TAG=785c3fd CPPMEGA_MAMBA3_B200_SPECS=B200:1 modal run scripts/modal_mamba3_b200_paths.py

If Modal accepts a spec but provisioning does not allocate a container, stop
the app and paste the app id/status into the companion status doc.

Image note: `GHCR_TAG=f6c15a2` is the stale cppmega baseline observed under
`latest`; prefer `GHCR_TAG=785c3fd` once the refresh build is present.
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Callable

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "latest")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"

APP_NAME = "cppmega-mamba3-b200-paths"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"

DEFAULT_SPECS = ("B200+:2", "B200:2", "B200:1")
SPEC_ENV = "CPPMEGA_MAMBA3_B200_SPECS"


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
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


def _time_cuda(fn, *, warmup: int, iters: int) -> dict[str, float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return {
        "mean_ms": sum(timings) / len(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "iters": float(iters),
    }


def _device_report(requested_gpu: str) -> dict[str, Any]:
    import torch

    actual = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": actual,
        "capability": capability,
        "is_sm100": tuple(capability or ()) == (10, 0),
        "is_b200_name": bool(actual and "B200" in actual.upper()),
        "requested_gpu_spec": requested_gpu,
        "image_ref": GHCR_REF,
    }


def _probe_patch_sites() -> list[dict[str, Any]]:
    _install_source_paths()
    _reset_mamba_imports()
    os.environ["MAMBA_SSM_SOURCE_DIR"] = SOURCE_ROOT

    from dataclasses import asdict

    from cppmega.megatron.upstream_patches.apply_mamba3_p2_psiv_patches import (
        probe_all_candidate_roots,
    )

    return [asdict(result) for result in probe_all_candidate_roots()]


def _benchmark_psiv_costs() -> list[dict[str, Any]]:
    import torch

    _install_source_paths()
    from cppmega.megatron.mamba3_psiv_cache import estimate_cache_bytes, precompute_psi_v

    shapes = [
        {
            "name": "design_b1_s8192_h16_r4_p64",
            "B": 1,
            "S": 8192,
            "H": 16,
            "R": 4,
            "P": 64,
            "warmup": 5,
            "iters": 30,
        },
        {
            "name": "prod_mbs4_s4096_h32_r4_p128",
            "B": 4,
            "S": 4096,
            "H": 32,
            "R": 4,
            "P": 128,
            "warmup": 4,
            "iters": 20,
        },
        {
            "name": "stress_mbs10_s4096_h32_r4_p128",
            "B": 10,
            "S": 4096,
            "H": 32,
            "R": 4,
            "P": 128,
            "warmup": 2,
            "iters": 10,
        },
    ]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    out: list[dict[str, Any]] = []

    for shape in shapes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(1234)
        B, S, H, R, P = shape["B"], shape["S"], shape["H"], shape["R"], shape["P"]
        V = torch.randn(B, S, H, P, device=device, dtype=dtype)
        mimo_v = torch.randn(H, R, P, device=device, dtype=torch.float32)
        psi_bf16 = mimo_v.to(dtype=dtype)
        cache = torch.empty((B, S, H, R, P), device=device, dtype=dtype)

        def alloc_cast():
            return precompute_psi_v(V, mimo_v)

        def out_precast():
            torch.mul(V.unsqueeze(3), psi_bf16.unsqueeze(0).unsqueeze(0), out=cache)
            return cache

        alloc_stats = _time_cuda(alloc_cast, warmup=shape["warmup"], iters=shape["iters"])
        out_stats = _time_cuda(out_precast, warmup=shape["warmup"], iters=shape["iters"])
        torch.cuda.synchronize()

        cache_bytes = estimate_cache_bytes(B, S, H, R, P, dtype)
        result = {
            "name": shape["name"],
            "shape": {"B": B, "S": S, "H": H, "R": R, "P": P, "dtype": str(dtype)},
            "cache_bytes": cache_bytes,
            "cache_gib": cache_bytes / (1024**3),
            "alloc_cast": alloc_stats,
            "out_precast": out_stats,
            "out_precast_effective_write_gib_s": (cache_bytes / (1024**3))
            / (out_stats["mean_ms"] / 1000.0),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        }
        out.append(result)
        del V, mimo_v, psi_bf16, cache
        torch.cuda.empty_cache()

    return out


def _benchmark_tilelang_bwd_split() -> dict[str, Any]:
    import math
    import time
    import traceback

    import torch

    _install_source_paths()
    _reset_mamba_imports()
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import (
        mamba_mimo_bwd_bwd,
        mamba_mimo_bwd_fwd,
    )
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, S, H, G, N, P, R = 2, 1024, 16, 1, 64, 64, 4
    chunk_size = 16
    rotary_dim_divisor = 4
    nchunks = math.ceil(S / chunk_size)
    result: dict[str, Any] = {
        "shape": {"B": B, "S": S, "H": H, "G": G, "N": N, "P": P, "R": R, "chunk": chunk_size},
    }
    t0 = time.time()
    try:
        torch.manual_seed(42)
        Q = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.1
        K = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.1
        V = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.1
        DOUT = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.1
        Q_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.1
        K_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.1
        MIMO_V = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.1
        MIMO_O = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.1
        angles = torch.randn(B, S, H, N // rotary_dim_divisor, device=device, dtype=torch.float32)
        dt = torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.01
        trap = (torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.5).to(dtype)
        adt = -torch.abs(torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.1)
        DA_CS, DA_CS_REV, SEGSUM = compute_dacs_segsum_triton(adt, chunk_size)
        z_dummy = torch.zeros(B, S, H, P, dtype=dtype, device=device)
        dz_dummy = torch.zeros(B, S, H, P, dtype=dtype, device=device)
        dmimo_z_dummy = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
        mimo_z_dummy = torch.zeros(H, R, P, dtype=torch.float32, device=device)
        D_dummy = torch.zeros(H, dtype=torch.float32, device=device)

        bf_kernel = mamba_mimo_bwd_fwd(
            B, S, H, G, N, P, R, False, False, True, chunk_size, rotary_dim_divisor, "bfloat16"
        )
        bb_kernel = mamba_mimo_bwd_bwd(
            B, S, H, G, N, P, R, False, False, True, chunk_size, rotary_dim_divisor, "bfloat16"
        )

        dmimo_o = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
        states = torch.zeros(B, H, nchunks, N, P, dtype=dtype, device=device)
        qk_dot = torch.zeros(B, H, S, R, R, dtype=dtype, device=device)
        dk = torch.zeros(B, S * R, H, N, dtype=dtype, device=device)
        dv = torch.zeros(B, S, H, P, dtype=dtype, device=device)
        dmimo_v = torch.zeros(B, H, R, P, dtype=torch.float32, device=device)
        dq = torch.zeros(B, S * R, H, N, dtype=dtype, device=device)
        dfactor = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dgamma_diag = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dangles = torch.zeros(B, S, H, N // rotary_dim_divisor, dtype=angles.dtype, device=device)
        dD = torch.zeros(B, H, dtype=torch.float32, device=device)
        dda = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dssda = torch.zeros(B, H, nchunks, chunk_size, chunk_size, dtype=torch.float32, device=device)
        dda_cs_rev = torch.zeros(B, H, S, dtype=torch.float32, device=device)
        dda_cs = torch.zeros(B, H, S, dtype=torch.float32, device=device)

        def bwd_fwd_fn():
            bf_kernel(
                DOUT,
                Q,
                K,
                V,
                Q_bias,
                K_bias,
                MIMO_V,
                MIMO_O,
                dmimo_o,
                states,
                z_dummy,
                mimo_z_dummy,
                dz_dummy,
                dmimo_z_dummy,
                angles,
                DA_CS,
                DA_CS_REV,
                dt,
                trap,
                D_dummy,
                qk_dot,
                SEGSUM,
            )

        def bwd_bwd_fn():
            bb_kernel(
                DOUT,
                Q,
                K,
                V,
                Q_bias,
                K_bias,
                MIMO_V,
                MIMO_O,
                dk,
                dv,
                dmimo_v,
                states,
                dq,
                z_dummy,
                mimo_z_dummy,
                angles,
                DA_CS,
                DA_CS_REV,
                dt,
                trap,
                dfactor,
                dgamma_diag,
                dangles,
                D_dummy,
                dD,
                qk_dot,
                dda,
                dssda,
                dda_cs_rev,
                dda_cs,
                SEGSUM,
            )

        bwd_fwd_fn()
        bwd_bwd_fn()
        torch.cuda.synchronize()
        bf = _time_cuda(bwd_fwd_fn, warmup=3, iters=20)
        bb = _time_cuda(bwd_bwd_fn, warmup=3, iters=20)
        result.update(
            {
                "status": "ok",
                "bwd_fwd": bf,
                "bwd_bwd": bb,
                "chain_mean_ms": bf["mean_ms"] + bb["mean_ms"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "status": "crashed",
                "exception_type": type(exc).__name__,
                "exception_short": textwrap.shorten(str(exc), width=800),
                "traceback_tail": traceback.format_exc()[-2000:],
            }
        )
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 3)
    return result


def _inspect_cutile_cute_possibility() -> dict[str, Any]:
    """Classify B200-specific cuTile/CuTe options without importing Hopper kernels."""
    import importlib.util
    import os

    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
    report: dict[str, Any] = {
        "requested_cute_arch": os.environ.get("CUTE_DSL_ARCH"),
        "existing_cppmega_cute_files_are_hopper_first": [
            "cppmega/megatron/cute_dsl_mimo/single_gemm_test.py",
            "cppmega/megatron/cute_dsl_mimo/fa4_bwd_adapter.py",
            "cppmega/megatron/cute_dsl_mimo/fused_bwd_bwd_sm90_p4.py",
        ],
        "together_mamba3_kernel_note": {
            "url": "https://www.together.ai/blog/mamba-3",
            "summary": "Mamba-3 code is described as a Triton + TileLang + CuTe DSL mix; "
            "TileLang is used for MIMO prefill, CuTe DSL for decode kernels on Hopper.",
        },
        "tilelang_blackwell_tcgen05_reference": {
            "local_path": "/home/dave/tilelang-build/testing/python/kernel/"
            "test_tilelang_kernel_bf16_gemm_tcgen5_ts.py",
            "assumptions": [
                "requires CUDA compute version 10",
                "uses T.alloc_tmem and T.tcgen05_gemm",
                "checks generated source for tcgen05mma_ts and tcgen05_st",
                "keeps TL_DISABLE_WARP_SPECIALIZED=True in PASS_CFG",
            ],
        },
        "do_not_reuse_hopper_path_as_b200": True,
    }
    for module_name in ("cuda.tile", "cutlass.cute", "tilelang"):
        try:
            spec = importlib.util.find_spec(module_name)
            report[module_name] = "available" if spec is not None else "missing"
        except Exception as exc:  # noqa: BLE001
            report[module_name] = f"error: {type(exc).__name__}: {exc}"
    return report


def _audit_hf_kernel_candidates() -> dict[str, Any]:
    import json as json_module
    import urllib.error
    import urllib.request

    candidates = [
        {
            "name": "kernels-community/mamba-ssm",
            "url": "https://huggingface.co/kernels-community/mamba-ssm",
            "kernel_url": "https://huggingface.co/kernels/kernels-community/mamba-ssm",
            "listed_hardware": ["B200", "H200", "H100"],
            "fit": "Partial replacement candidate for selective scan / Mamba and Mamba2 public APIs; "
            "not a drop-in Mamba3 MIMO TileLang replacement.",
            "expected_functions": [
                "selective_scan_fn",
                "mamba_inner_fn",
                "selective_state_update",
                "mamba_chunk_scan_combined",
                "mamba_split_conv1d_scan_combined",
                "Mamba",
                "Mamba2",
                "MambaLMHeadModel",
            ],
            "benchmark_status": "No benchmark listed on the Hugging Face kernel card.",
        },
        {
            "name": "kernels-community/flash-attn4",
            "url": "https://huggingface.co/kernels-community/flash-attn4",
            "fit": "Attention reference only; useful for Blackwell packaging/build patterns, not Mamba transitions.",
            "expected_functions": [],
        },
    ]
    out: dict[str, Any] = {"candidates": candidates}
    try:
        with urllib.request.urlopen("https://huggingface.co/api/models?other=kernels&search=mamba", timeout=20) as resp:
            out["hf_api_mamba_search"] = json_module.loads(resp.read().decode("utf-8"))[:5]
    except (urllib.error.URLError, TimeoutError, json_module.JSONDecodeError) as exc:
        out["hf_api_mamba_search_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import kernels  # type: ignore[import-not-found]

        out["kernels_package"] = getattr(kernels, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        out["kernels_package"] = f"not usable in image: {type(exc).__name__}: {exc}"
    return out


def _run_probe(requested_gpu: str) -> dict[str, Any]:
    _install_source_paths()
    return {
        "device": _device_report(requested_gpu),
        "patch_sites": _probe_patch_sites(),
        "psiv_costs": _benchmark_psiv_costs(),
        "tilelang_bwd_split": _benchmark_tilelang_bwd_split(),
        "cutile_cute_possibility": _inspect_cutile_cute_possibility(),
        "hf_kernel_candidates": _audit_hf_kernel_candidates(),
    }


@app.function(image=_image(), gpu="B200+:2", timeout=2400)
def run_b200_plus_2() -> dict[str, Any]:
    return _run_probe("B200+:2")


@app.function(image=_image(), gpu="B200:2", timeout=2400)
def run_b200_2() -> dict[str, Any]:
    return _run_probe("B200:2")


@app.function(image=_image(), gpu="B200:1", timeout=2400)
def run_b200_1() -> dict[str, Any]:
    return _run_probe("B200:1")


_RUNNERS: dict[str, Callable[[], Any]] = {
    "B200+:2": run_b200_plus_2.remote,
    "B200:2": run_b200_2.remote,
    "B200:1": run_b200_1.remote,
}


def _selected_specs() -> list[str]:
    raw = os.environ.get(SPEC_ENV)
    if not raw:
        return list(DEFAULT_SPECS)
    specs = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [spec for spec in specs if spec not in _RUNNERS]
    if unknown:
        raise ValueError(f"Unsupported {SPEC_ENV}: {unknown}; expected one of {sorted(_RUNNERS)}")
    return specs


@app.local_entrypoint()
def main() -> None:
    results: dict[str, Any] = {}
    for spec in _selected_specs():
        print(f"=== Modal Mamba3 B200 probe: {spec} ===", flush=True)
        try:
            results[spec] = _RUNNERS[spec]()
        except Exception as exc:  # noqa: BLE001
            results[spec] = {
                "status": "local_or_remote_exception",
                "exception_type": type(exc).__name__,
                "exception_short": textwrap.shorten(str(exc), width=1000),
            }
        print(json.dumps({spec: results[spec]}, indent=2, sort_keys=True, default=str), flush=True)
    print("=== combined ===")
    print(json.dumps(results, indent=2, sort_keys=True, default=str))
