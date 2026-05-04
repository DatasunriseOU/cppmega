"""Modal GHCR A/B for Hopper Mamba3 Hoist-PsiV, non-TMA path only.

This script patches a copied source tree inside the Modal container:
  * regular fwd writes `PsiV_out` while computing the existing PsiV fragment;
  * regular bwd_fwd/bwd_bwd consume `PsiV_in`;
  * production defaults and installed site-packages are not mutated.

Run:
    CPPMEGA_MODAL_GPU=H100:2 modal run scripts/modal_mamba3_psiv_hopper_ab.py
    CPPMEGA_MODAL_GPU=H200:2 modal run scripts/modal_mamba3_psiv_hopper_ab.py
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "latest")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")
SHAPE_NAME = os.environ.get("CPPMEGA_PSIV_AB_SHAPE", "prod_mbs4")
WARMUP = int(os.environ.get("CPPMEGA_PSIV_AB_WARMUP", "3"))
ITERS = int(os.environ.get("CPPMEGA_PSIV_AB_ITERS", "10"))

APP_NAME = "cppmega-mamba3-psiv-hopper-ab"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    img = img.add_local_dir("cppmega", f"{CPPMEGA_ROOT}/cppmega", copy=True)
    img = img.add_local_dir(
        "/home/dave/state-spaces-mamba/mamba_ssm",
        f"{SOURCE_ROOT}/mamba_ssm",
        copy=True,
    )
    return img


app = modal.App(APP_NAME)


def _reset_mamba_imports() -> None:
    import sys

    for name in list(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            del sys.modules[name]


def _use_source_root(source_root: str) -> None:
    import sys

    for path in (source_root, CPPMEGA_ROOT):
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, CPPMEGA_ROOT)
    sys.path.insert(0, source_root)
    os.environ["MAMBA_SSM_SOURCE_DIR"] = source_root
    _reset_mamba_imports()


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
        "image_ref": GHCR_REF,
    }


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


def _max_abs(name: str, a, b) -> dict[str, Any]:
    import torch

    diff = (a.float() - b.float()).abs()
    denom = torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-6)
    rel = diff / denom
    return {
        "name": name,
        "max_abs": float(diff.max().item()),
        "max_rel": float(rel.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _copy_and_patch_source() -> tuple[str, dict[str, Any]]:
    import shutil
    import tempfile
    from dataclasses import asdict

    _use_source_root(SOURCE_ROOT)
    tmp_root = tempfile.mkdtemp(prefix="cppmega_mamba3_psiv_ab_")
    shutil.copytree(f"{SOURCE_ROOT}/mamba_ssm", f"{tmp_root}/mamba_ssm")
    from cppmega.megatron.upstream_patches.apply_mamba3_p2_psiv_patches import (
        patch_source_tree_for_hopper_psiv_ab,
    )

    result = patch_source_tree_for_hopper_psiv_ab(f"{tmp_root}/mamba_ssm/ops/tilelang/mamba3")
    return tmp_root, asdict(result)


def _make_inputs(shape: dict[str, Any]) -> dict[str, Any]:
    import torch
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, S, H, G = shape["B"], shape["S"], shape["H"], shape["G"]
    N, P, R = shape["N"], shape["P"], shape["R"]
    chunk_size = shape["chunk"]
    rotary_dim_divisor = shape["rotary_dim_divisor"]

    torch.manual_seed(1234)
    q = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.1
    dout = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.1
    q_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.1
    k_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.1
    mimo_v = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.1
    mimo_o = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.1
    angles = torch.randn(B, S, H, N // rotary_dim_divisor, device=device, dtype=torch.float32)
    dt = torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.01
    trap = (torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.5).to(dtype)
    adt = -torch.abs(torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.1)
    da_cs, da_cs_rev, segsum = compute_dacs_segsum_triton(adt, chunk_size)

    psi_v = torch.empty((B, S, H, R, P), dtype=dtype, device=device)
    mimo_v_bf16 = mimo_v.to(dtype=dtype)

    def write_psiv():
        torch.mul(v.unsqueeze(3), mimo_v_bf16.unsqueeze(0).unsqueeze(0), out=psi_v)

    write_psiv()
    torch.cuda.synchronize()

    return {
        "shape": shape,
        "dtype": dtype,
        "q": q,
        "k": k,
        "v": v,
        "dout": dout,
        "q_bias": q_bias,
        "k_bias": k_bias,
        "mimo_v": mimo_v,
        "mimo_o": mimo_o,
        "angles": angles,
        "dt": dt,
        "trap": trap,
        "da_cs": da_cs,
        "da_cs_rev": da_cs_rev,
        "segsum": segsum,
        "psi_v": psi_v,
        "write_psiv": write_psiv,
        "z_dummy": torch.empty(B, S, H, P, dtype=dtype, device=device),
        "dz_dummy": torch.empty(B, S, H, P, dtype=dtype, device=device),
        "mimo_z_dummy": torch.empty(H, R, P, dtype=torch.float32, device=device),
        "dmimo_z_dummy": torch.empty(B, H, R, P, dtype=torch.float32, device=device),
        "d_dummy": torch.empty(H, dtype=torch.float32, device=device),
    }


def _alloc_bwd_outputs(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch

    shape = inputs["shape"]
    B, S, H, R = shape["B"], shape["S"], shape["H"], shape["R"]
    N, P, chunk = shape["N"], shape["P"], shape["chunk"]
    nchunks = math.ceil(S / chunk)
    dtype = inputs["dtype"]
    device = torch.device("cuda")
    return {
        "dmimo_o": torch.empty(B, H, R, P, dtype=torch.float32, device=device),
        "states": torch.empty(B, H, nchunks, N, P, dtype=dtype, device=device),
        "qk_dot": torch.empty(B, H, S, R, R, dtype=dtype, device=device),
        "dk": torch.empty(B, S * R, H, N, dtype=dtype, device=device),
        "dv": torch.empty(B, S, H, P, dtype=dtype, device=device),
        "dmimo_v": torch.empty(B, H, R, P, dtype=torch.float32, device=device),
        "dq": torch.empty(B, S * R, H, N, dtype=dtype, device=device),
        "dfactor": torch.zeros(B, H, S, dtype=torch.float32, device=device),
        "dgamma_diag": torch.zeros(B, H, S, dtype=torch.float32, device=device),
        "dangles": torch.zeros(B, S, H, N // shape["rotary_dim_divisor"], dtype=torch.float32, device=device),
        "dD": torch.empty(B, H, dtype=torch.float32, device=device),
        "dda": torch.zeros(B, H, S, dtype=torch.float32, device=device),
        "dssda": torch.zeros(B, H, nchunks, chunk, chunk, dtype=torch.float32, device=device),
        "dda_cs_rev": torch.zeros(B, H, S, dtype=torch.float32, device=device),
        "dda_cs": torch.zeros(B, H, S, dtype=torch.float32, device=device),
    }


def _load_kernels(source_root: str, *, patched: bool, shape: dict[str, Any]) -> dict[str, Any]:
    _use_source_root(source_root)
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import (
        mamba_mimo_bwd_bwd,
        mamba_mimo_bwd_fwd,
    )
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd import mamba_mimo_fwd

    B, S, H, G = shape["B"], shape["S"], shape["H"], shape["G"]
    N, P, R = shape["N"], shape["P"], shape["R"]
    chunk = shape["chunk"]
    div = shape["rotary_dim_divisor"]
    return {
        "patched": patched,
        "fwd": mamba_mimo_fwd(B, S, H, G, N, P, R, False, False, True, False, chunk, div, "bfloat16"),
        "bf": mamba_mimo_bwd_fwd(B, S, H, G, N, P, R, False, False, True, chunk, div, "bfloat16"),
        "bb": mamba_mimo_bwd_bwd(B, S, H, G, N, P, R, False, False, True, chunk, div, "bfloat16"),
    }


def _call_fwd(kernel, inputs: dict[str, Any], *, patched: bool):
    import torch

    shape = inputs["shape"]
    B, S, H, P = shape["B"], shape["S"], shape["H"], shape["P"]
    o = torch.empty(B, S, H, P, dtype=inputs["dtype"], device="cuda")
    psi_v_out = torch.empty_like(inputs["psi_v"])
    args = [
        inputs["q"], inputs["k"], inputs["v"], o,
        inputs["q_bias"], inputs["k_bias"], inputs["mimo_v"],
    ]
    if patched:
        args.append(psi_v_out)
    args.extend([
        inputs["mimo_o"], inputs["z_dummy"], inputs["d_dummy"], inputs["mimo_z_dummy"],
        inputs["angles"], inputs["da_cs"], inputs["da_cs_rev"], inputs["dt"],
        inputs["trap"], inputs["segsum"], None, None,
    ])
    kernel(*args)
    return {"o": o, "psi_v_out": psi_v_out}


def _call_bwd_fwd(kernel, inputs: dict[str, Any], outputs: dict[str, Any], *, patched: bool) -> None:
    args = [
        inputs["dout"], inputs["q"], inputs["k"], inputs["v"],
        inputs["q_bias"], inputs["k_bias"], inputs["mimo_v"],
    ]
    if patched:
        args.append(inputs["psi_v"])
    args.extend([
        inputs["mimo_o"], outputs["dmimo_o"], outputs["states"],
        inputs["z_dummy"], inputs["mimo_z_dummy"], inputs["dz_dummy"], inputs["dmimo_z_dummy"],
        inputs["angles"], inputs["da_cs"], inputs["da_cs_rev"], inputs["dt"],
        inputs["trap"], inputs["d_dummy"], outputs["qk_dot"], inputs["segsum"],
    ])
    kernel(*args)


def _call_bwd_bwd(kernel, inputs: dict[str, Any], outputs: dict[str, Any], *, patched: bool) -> None:
    args = [
        inputs["dout"], inputs["q"], inputs["k"], inputs["v"],
        inputs["q_bias"], inputs["k_bias"], inputs["mimo_v"],
    ]
    if patched:
        args.append(inputs["psi_v"])
    args.extend([
        inputs["mimo_o"], outputs["dk"], outputs["dv"], outputs["dmimo_v"],
        outputs["states"], outputs["dq"], inputs["z_dummy"], inputs["mimo_z_dummy"],
        inputs["angles"], inputs["da_cs"], inputs["da_cs_rev"], inputs["dt"],
        inputs["trap"], outputs["dfactor"], outputs["dgamma_diag"], outputs["dangles"],
        inputs["d_dummy"], outputs["dD"], outputs["qk_dot"], outputs["dda"],
        outputs["dssda"], outputs["dda_cs_rev"], outputs["dda_cs"], inputs["segsum"],
    ])
    kernel(*args)


def _shape_from_name(preset: str) -> dict[str, Any]:
    shapes = {
        "smoke": {"B": 2, "S": 1024, "H": 16, "G": 1, "N": 64, "P": 64, "R": 4, "chunk": 16, "rotary_dim_divisor": 4},
        "prod_mbs4": {"B": 4, "S": 4096, "H": 32, "G": 1, "N": 64, "P": 128, "R": 4, "chunk": 16, "rotary_dim_divisor": 4},
    }
    if preset not in shapes:
        raise ValueError(f"Unknown CPPMEGA_PSIV_AB_SHAPE={preset}; choices={sorted(shapes)}")
    return {"name": preset, **shapes[preset]}


def _benchmark_ab(*, shape_name: str, warmup: int, iters: int) -> dict[str, Any]:
    import torch

    shape = _shape_from_name(shape_name)
    patched_root, patch_result = _copy_and_patch_source()

    _use_source_root(SOURCE_ROOT)
    inputs = _make_inputs(shape)
    cache_bytes = shape["B"] * shape["S"] * shape["H"] * shape["R"] * shape["P"] * 2

    baseline = _load_kernels(SOURCE_ROOT, patched=False, shape=shape)
    patched = _load_kernels(patched_root, patched=True, shape=shape)

    fwd_base = _call_fwd(baseline["fwd"], inputs, patched=False)
    fwd_patch = _call_fwd(patched["fwd"], inputs, patched=True)
    torch.cuda.synchronize()

    bf_base_out = _alloc_bwd_outputs(inputs)
    bf_patch_out = _alloc_bwd_outputs(inputs)
    _call_bwd_fwd(baseline["bf"], inputs, bf_base_out, patched=False)
    _call_bwd_fwd(patched["bf"], inputs, bf_patch_out, patched=True)
    torch.cuda.synchronize()

    bb_base_out = _alloc_bwd_outputs(inputs)
    bb_patch_out = _alloc_bwd_outputs(inputs)
    bb_base_out["states"].copy_(bf_base_out["states"])
    bb_base_out["qk_dot"].copy_(bf_base_out["qk_dot"])
    bb_patch_out["states"].copy_(bf_base_out["states"])
    bb_patch_out["qk_dot"].copy_(bf_base_out["qk_dot"])
    _call_bwd_bwd(baseline["bb"], inputs, bb_base_out, patched=False)
    _call_bwd_bwd(patched["bb"], inputs, bb_patch_out, patched=True)
    torch.cuda.synchronize()

    def fwd_base_fn():
        _call_fwd(baseline["fwd"], inputs, patched=False)

    def fwd_patch_fn():
        _call_fwd(patched["fwd"], inputs, patched=True)

    def bf_base_fn():
        _call_bwd_fwd(baseline["bf"], inputs, bf_base_out, patched=False)

    def bf_patch_fn():
        _call_bwd_fwd(patched["bf"], inputs, bf_patch_out, patched=True)

    def bb_base_fn():
        _call_bwd_bwd(baseline["bb"], inputs, bb_base_out, patched=False)

    def bb_patch_fn():
        _call_bwd_bwd(patched["bb"], inputs, bb_patch_out, patched=True)

    psiv_write = _time_cuda(inputs["write_psiv"], warmup=warmup, iters=iters)
    fwd_base_t = _time_cuda(fwd_base_fn, warmup=warmup, iters=iters)
    fwd_patch_t = _time_cuda(fwd_patch_fn, warmup=warmup, iters=iters)
    bf_base_t = _time_cuda(bf_base_fn, warmup=warmup, iters=iters)
    bf_patch_t = _time_cuda(bf_patch_fn, warmup=warmup, iters=iters)
    bb_base_t = _time_cuda(bb_base_fn, warmup=warmup, iters=iters)
    bb_patch_t = _time_cuda(bb_patch_fn, warmup=warmup, iters=iters)

    bf_saved = bf_base_t["mean_ms"] - bf_patch_t["mean_ms"]
    bb_saved = bb_base_t["mean_ms"] - bb_patch_t["mean_ms"]
    fwd_write_delta = fwd_patch_t["mean_ms"] - fwd_base_t["mean_ms"]
    bwd_saved = bf_saved + bb_saved
    return {
        "shape": shape,
        "patch_result": patch_result,
        "cache_bytes": cache_bytes,
        "cache_gib": cache_bytes / (1024**3),
        "correctness": [
            _max_abs("fwd_o", fwd_base["o"], fwd_patch["o"]),
            _max_abs("fwd_psiv_out", inputs["psi_v"], fwd_patch["psi_v_out"]),
            _max_abs("bwd_fwd_dmimo_o", bf_base_out["dmimo_o"], bf_patch_out["dmimo_o"]),
            _max_abs("bwd_fwd_states", bf_base_out["states"], bf_patch_out["states"]),
            _max_abs("bwd_fwd_qk_dot", bf_base_out["qk_dot"], bf_patch_out["qk_dot"]),
            _max_abs("bwd_bwd_dq", bb_base_out["dq"], bb_patch_out["dq"]),
            _max_abs("bwd_bwd_dk", bb_base_out["dk"], bb_patch_out["dk"]),
            _max_abs("bwd_bwd_dv", bb_base_out["dv"], bb_patch_out["dv"]),
            _max_abs("bwd_bwd_dmimo_v", bb_base_out["dmimo_v"], bb_patch_out["dmimo_v"]),
        ],
        "timings": {
            "psiv_write_precast_out": psiv_write,
            "fwd_baseline": fwd_base_t,
            "fwd_write": fwd_patch_t,
            "bwd_fwd_baseline": bf_base_t,
            "bwd_fwd_psiv_in": bf_patch_t,
            "bwd_bwd_baseline": bb_base_t,
            "bwd_bwd_psiv_in": bb_patch_t,
        },
        "deltas_ms": {
            "fwd_write_delta": fwd_write_delta,
            "bwd_fwd_saved": bf_saved,
            "bwd_bwd_saved": bb_saved,
            "bwd_saved_total": bwd_saved,
            "net_vs_precast_write": bwd_saved - psiv_write["mean_ms"],
            "net_vs_fwd_write_delta": bwd_saved - fwd_write_delta,
        },
        "go": bwd_saved > psiv_write["mean_ms"] and bwd_saved > fwd_write_delta,
    }


@app.function(image=_image(), gpu=GPU_SPEC, timeout=3600)
def run_probe(requested_gpu: str, shape_name: str, warmup: int, iters: int) -> dict[str, Any]:
    return {
        "device": _device_report(requested_gpu),
        "ab": _benchmark_ab(shape_name=shape_name, warmup=warmup, iters=iters),
    }


@app.local_entrypoint()
def main() -> None:
    result = run_probe.remote(GPU_SPEC, SHAPE_NAME, WARMUP, ITERS)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
