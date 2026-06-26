"""Modal H100 harness for Mamba3 stage2 TileLang bwd tuning.

Runs only H100 component shapes. The script compares:

* installed baseline mamba_ssm TileLang source
* current guarded stage2 force-nonTMA patch, bf=1,bb=0
* Lane C direct-fragment candidate on top of that patch

The candidate removes two bwd_bwd shared-memory vector temporaries by copying
the non-TMA vector loads directly into fragments.
"""
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any

import modal

_REPO_ROOT = pathlib.Path(__file__).parent.parent

APP_NAME = "cppmega-wave28-lane-c-h100"
RESULTS_VOL = "cppmega-mamba3-benchmarks"
BENCH_DIR = "/benchmarks/mamba3_wave28_lane_c_h100"
GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"


def _image() -> modal.Image:
    return (
        modal.Image.from_registry(
            GHCR_REF,
            secret=modal.Secret.from_name("ghcr-pull"),
            add_python=None,
        )
        .env({"PYTHONPATH": "/opt/cppmega:/opt/megatron-lm"})
        .add_local_dir(str(_REPO_ROOT / "cppmega"), remote_path="/opt/cppmega/cppmega")
        .add_local_dir(
            str(_REPO_ROOT / "upstream_prs"),
            remote_path="/opt/cppmega/upstream_prs",
        )
        .add_local_file(str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml")
    )


app = modal.App(APP_NAME)
results_vol = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)
image = _image()


_BENCH_CODE = r"""
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import time

import torch

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton


PATCH_PATH = pathlib.Path("/opt/cppmega/upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_stage2_force_nontma.patch")

DIRECT_FRAGMENT_REPLACEMENTS = [
    (
        "                dA_cs_rev_shared = T.alloc_shared([chunk_size], T.float32)\n"
        "                T.copy(DA_CS_REV[i_b, i_h, chunk_start:chunk_start+chunk_size], dA_cs_rev_shared, disable_tma=True)\n"
        "                T.copy(dA_cs_rev_shared, dA_cs_rev_frag)\n",
        "                T.copy(DA_CS_REV[i_b, i_h, chunk_start:chunk_start+chunk_size], dA_cs_rev_frag, disable_tma=True)\n",
    ),
    (
        "                dA_cs_shared = T.alloc_shared([chunk_size], T.float32)\n\n"
        "                T.copy(DA_CS[i_b, i_h, chunk_start:chunk_start+chunk_size], dA_cs_shared, disable_tma=True)\n"
        "                T.copy(dA_cs_shared, dA_cs_dq_frag)\n",
        "                T.copy(DA_CS[i_b, i_h, chunk_start:chunk_start+chunk_size], dA_cs_dq_frag, disable_tma=True)\n",
    ),
]


def _mamba_bwd_path():
    spec = importlib.util.find_spec("mamba_ssm.ops.tilelang.mamba3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("mamba_ssm.ops.tilelang.mamba3 not importable")
    path = pathlib.Path(next(iter(spec.submodule_search_locations))) / "mamba3_mimo_bwd.py"
    if not path.exists():
        raise RuntimeError(f"missing mamba3_mimo_bwd.py at {path}")
    return path


def _apply_stage2_patch(src, dst):
    shutil.copy2(src, dst)
    proc = subprocess.run(
        ["patch", "--ignore-whitespace", "-p4", str(dst)],
        input=PATCH_PATH.read_bytes(),
        cwd=dst.parent,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "stage2 patch failed\n"
            + proc.stdout.decode(errors="replace")[-4000:]
            + proc.stderr.decode(errors="replace")[-4000:]
        )


def _make_variant(src, name, workdir):
    dst = workdir / f"mamba3_mimo_bwd_{name}.py"
    if name == "baseline":
        shutil.copy2(src, dst)
    else:
        _apply_stage2_patch(src, dst)
        if name == "lane_c_direct_frag":
            text = dst.read_text()
            for old, new in DIRECT_FRAGMENT_REPLACEMENTS:
                if old not in text:
                    raise RuntimeError(f"replacement marker missing for {name}")
                text = text.replace(old, new)
            dst.write_text(text)
    return dst


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(f"wave28_lane_c_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _inputs(shape):
    torch.manual_seed(20260430 + shape["seed"])
    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, S, H, G, N, P, R = [shape[k] for k in ("B", "S", "H", "G", "N", "P", "R")]
    chunk_size = shape["chunk_size"]
    rotary_dim_divisor = 4
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
    dA_cs, dA_cs_rev, segsum = compute_dacs_segsum_triton(adt, chunk_size)
    D = torch.randn(H, device=device, dtype=torch.float32) * 0.1
    return {
        "dout": dout, "q": q, "k": k, "v": v, "q_bias": q_bias, "k_bias": k_bias,
        "mimo_v": mimo_v, "mimo_o": mimo_o, "z": None, "mimo_z": None,
        "angles": angles, "dA_cs": dA_cs, "dA_cs_rev": dA_cs_rev, "dt": dt,
        "trap": trap, "D": D, "segsum": segsum, "chunk_size": chunk_size,
        "rotary_dim_divisor": rotary_dim_divisor, "dtype": dtype,
    }


def _run_combined(mod, args):
    return mod.mamba_mimo_bwd_combined(
        args["dout"], args["q"], args["k"], args["v"],
        args["q_bias"], args["k_bias"], args["mimo_v"], args["mimo_o"],
        args["z"], args["mimo_z"], args["angles"], args["dA_cs"], args["dA_cs_rev"],
        args["dt"], args["trap"], args["D"], args["segsum"],
        args["chunk_size"], args["rotary_dim_divisor"], args["dtype"],
    )


def _bench_combined(mod, args, warmup, iters):
    for _ in range(warmup):
        _run_combined(mod, args)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = None
    for _ in range(iters):
        out = _run_combined(mod, args)
    end.record()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    return {
        "chain_ms": start.elapsed_time(end) / iters,
        "peak_mem_mib_delta": (peak - before) / (1024 * 1024),
        "end_mem_mib_delta": (after - before) / (1024 * 1024),
        "output_count": len(out) if out is not None else 0,
    }


def _split_once(mod, args):
    B, S, R, G, N = args["q"].shape
    H, P = args["v"].shape[-2], args["v"].shape[-1]
    chunk_size = args["chunk_size"]
    nchunks = (S + chunk_size - 1) // chunk_size
    dtype_str = str(args["dtype"]).replace("torch.", "")
    q_flat = args["q"].view(B, S * R, G, N)
    k_flat = args["k"].view(B, S * R, G, N)
    qk_cols = R * R if "R * R" in pathlib.Path(mod.__file__).read_text() else R
    qk_shape = (B, H, S, qk_cols) if qk_cols == R * R else (B, H, S, R, R)
    states = torch.empty(B, H, nchunks, N, P, dtype=args["v"].dtype, device=args["v"].device)
    qk_dot = torch.zeros(qk_shape, dtype=args["q"].dtype, device=args["q"].device)
    dmimo_o = torch.empty(B, H, R, P, dtype=args["mimo_v"].dtype, device=args["v"].device)
    bf = mod.mamba_mimo_bwd_fwd(B, S, H, G, N, P, R, False, True, True, chunk_size, args["rotary_dim_divisor"], dtype_str, 128, 1 if "bf_num_stages=1" in pathlib.Path(mod.__file__).read_text() else 0)
    bf(
        args["dout"], q_flat if qk_cols == R * R else args["q"], k_flat if qk_cols == R * R else args["k"],
        args["v"], args["q_bias"], args["k_bias"], args["mimo_v"], args["mimo_o"],
        dmimo_o, states, None, None, None, None, args["angles"], args["dA_cs"],
        args["dA_cs_rev"], args["dt"], args["trap"], args["D"], qk_dot, args["segsum"],
    )
    dk = torch.empty(B, S * R, H, N, dtype=args["k"].dtype, device=args["v"].device)
    dv = torch.empty_like(args["v"])
    dmimo_v = torch.empty(B, H, R, P, dtype=args["mimo_v"].dtype, device=args["v"].device)
    dq = torch.empty(B, S * R, H, N, dtype=args["q"].dtype, device=args["v"].device)
    dfactor = torch.zeros(B, H, S, dtype=torch.float32, device=args["v"].device)
    dgamma_diag = torch.zeros(B, H, S, dtype=torch.float32, device=args["v"].device)
    dangles = torch.zeros(B, S, H, N // args["rotary_dim_divisor"], dtype=args["angles"].dtype, device=args["v"].device)
    dD = torch.empty(B, H, dtype=torch.float32, device=args["v"].device)
    ddA = torch.zeros(B, H, S, dtype=torch.float32, device=args["v"].device)
    dSSdA = torch.zeros(B, H, nchunks, chunk_size, chunk_size, dtype=torch.float32, device=args["v"].device)
    ddA_cs_rev = torch.zeros(B, H, S, dtype=torch.float32, device=args["v"].device)
    ddA_cs = torch.zeros(B, H, S, dtype=torch.float32, device=args["v"].device)
    bb = mod.mamba_mimo_bwd_bwd(B, S, H, G, N, P, R, False, True, True, chunk_size, args["rotary_dim_divisor"], dtype_str, 256, 0)
    return bf, bb, (states, qk_dot, dk, dv, dmimo_v, dq, dfactor, dgamma_diag, dangles, dD, ddA, dSSdA, ddA_cs_rev, ddA_cs, q_flat, k_flat)


def _bench_split(mod, args, warmup, iters):
    bf, bb, bufs = _split_once(mod, args)
    states, qk_dot, dk, dv, dmimo_v, dq, dfactor, dgamma_diag, dangles, dD, ddA, dSSdA, ddA_cs_rev, ddA_cs, q_flat, k_flat = bufs
    B, S, R, G, N = args["q"].shape
    H, P = args["v"].shape[-2], args["v"].shape[-1]
    q_arg = q_flat if len(qk_dot.shape) == 4 else args["q"]
    k_arg = k_flat if len(qk_dot.shape) == 4 else args["k"]
    def run_bf():
        bf(args["dout"], q_arg, k_arg, args["v"], args["q_bias"], args["k_bias"],
           args["mimo_v"], args["mimo_o"], torch.empty(B, H, R, P, dtype=args["mimo_v"].dtype, device=args["v"].device),
           states, None, None, None, None, args["angles"], args["dA_cs"], args["dA_cs_rev"],
           args["dt"], args["trap"], args["D"], qk_dot, args["segsum"])
    def run_bb():
        bb(args["dout"], q_arg, k_arg, args["v"], args["q_bias"], args["k_bias"], args["mimo_v"],
           args["mimo_o"], dk, dv, dmimo_v, states, dq, None, None, args["angles"],
           args["dA_cs"], args["dA_cs_rev"], args["dt"], args["trap"], dfactor,
           dgamma_diag, dangles, args["D"], dD, qk_dot, ddA, dSSdA, ddA_cs_rev,
           ddA_cs, args["segsum"])
    def time_fn(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    return {"bwd_fwd_ms": time_fn(run_bf), "bwd_bwd_ms": time_fn(run_bb)}


def _diff(a, b):
    diffs = []
    for idx, (x, y) in enumerate(zip(a, b)):
        if x is None or y is None:
            diffs.append({"idx": idx, "skipped": True})
            continue
        d = (x.float() - y.float()).abs()
        diffs.append({"idx": idx, "max_abs": float(d.max().item()), "mean_abs": float(d.mean().item())})
    return diffs


def main():
    installed = _mamba_bwd_path()
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="wave28_lane_c_h100_"))
    variant_paths = {
        name: _make_variant(installed, name, workdir)
        for name in ("baseline", "stage2_current", "lane_c_direct_frag")
    }
    modules = {name: _load_module(path, name) for name, path in variant_paths.items()}
    shapes = [
        {"name": "smoke", "B": 1, "S": 128, "H": 4, "G": 1, "N": 64, "P": 64, "R": 4, "chunk_size": 16, "seed": 1},
        {"name": "representative", "B": 2, "S": 512, "H": 8, "G": 1, "N": 64, "P": 64, "R": 4, "chunk_size": 16, "seed": 2},
    ]
    results = {
        "gpu_name": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "installed_mamba3_bwd": str(installed),
        "patch_path": str(PATCH_PATH),
        "variants": {},
        "shapes": [],
    }
    for name, path in variant_paths.items():
        text = path.read_text()
        results["variants"][name] = {
            "path": str(path),
            "bf_num_stages_1": "bf_num_stages=1" in text,
            "bb_num_stages_0": "bb_num_stages=0" in text,
            "direct_fragment_candidate": name == "lane_c_direct_frag",
            "disable_tma_count": text.count("disable_tma=True"),
            "shared_vector_staging_count": text.count("dA_cs_rev_shared") + text.count("dA_cs_shared"),
        }
    for shape in shapes:
        shape_result = {"shape": shape, "bench": {}, "diffs": {}}
        args = _inputs(shape)
        outputs = {}
        for name in ("baseline", "stage2_current", "lane_c_direct_frag"):
            outputs[name] = _run_combined(modules[name], args)
            torch.cuda.synchronize()
        shape_result["diffs"]["stage2_current_vs_baseline"] = _diff(outputs["baseline"], outputs["stage2_current"])
        shape_result["diffs"]["lane_c_vs_stage2_current"] = _diff(outputs["stage2_current"], outputs["lane_c_direct_frag"])
        del outputs
        torch.cuda.empty_cache()
        for name in ("baseline", "stage2_current", "lane_c_direct_frag"):
            args = _inputs(shape)
            chain = _bench_combined(modules[name], args, warmup=2, iters=8)
            split = _bench_split(modules[name], args, warmup=2, iters=8)
            shape_result["bench"][name] = {**chain, **split}
            del args
            torch.cuda.empty_cache()
        results["shapes"].append(shape_result)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
"""


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/vol": results_vol},
)
def run_bench(run_id: str) -> dict[str, Any]:
    out_dir = pathlib.Path("/vol") / BENCH_DIR.lstrip("/") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = "/opt/cppmega:/opt/megatron-lm"

    # Verify the production applier gates against the installed source path.
    no_op = subprocess.run(
        ["python", "-m", "cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches"],
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    gated_env = child_env.copy()
    gated_env["CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA"] = "1"
    gated = subprocess.run(
        ["python", "-m", "cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches"],
        env=gated_env,
        capture_output=True,
        text=True,
        check=False,
    )

    bench_file = out_dir / "bench_driver.py"
    bench_file.write_text(_BENCH_CODE)
    proc = subprocess.run(
        ["python", str(bench_file)],
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=3300,
    )

    (out_dir / "applier_noop.txt").write_text(no_op.stdout + no_op.stderr)
    (out_dir / "applier_gated.txt").write_text(gated.stdout + gated.stderr)
    (out_dir / "stdout.txt").write_text(proc.stdout)
    (out_dir / "stderr.txt").write_text(proc.stderr)

    if proc.returncode != 0:
        result = {
            "run_id": run_id,
            "returncode": proc.returncode,
            "applier_noop": no_op.stdout + no_op.stderr,
            "applier_gated": gated.stdout + gated.stderr,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    else:
        parsed = json.loads(proc.stdout[proc.stdout.find("{"):])
        result = {
            "run_id": run_id,
            "returncode": proc.returncode,
            "applier_noop": no_op.stdout + no_op.stderr,
            "applier_gated": gated.stdout + gated.stderr,
            "report": parsed,
        }
    (out_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    results_vol.commit()
    return result


@app.function(image=image, gpu="H100", timeout=600)
def verify_applier_mutation() -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/cppmega:/opt/megatron-lm"
    env["CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA"] = "1"
    env["MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION"] = "1"

    find_code = (
        "import importlib.util, pathlib;"
        "spec=importlib.util.find_spec('mamba_ssm.ops.tilelang.mamba3');"
        "p=pathlib.Path(next(iter(spec.submodule_search_locations))) / 'mamba3_mimo_bwd.py';"
        "print(p)"
    )
    path_proc = subprocess.run(
        ["python", "-c", find_code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    kernel_path = pathlib.Path(path_proc.stdout.strip())
    original = kernel_path.read_bytes()

    apply_proc = subprocess.run(
        ["python", "-m", "cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    text = kernel_path.read_text()
    patched = {
        "flat_q": "Q: T.Tensor([B, S * R, G, N], dtype)" in text,
        "flat_qk": "QK_DOT: T.Tensor([B, H, S, R * R], dtype)" in text,
        "bf_num_stages_1": "bf_num_stages=1" in text,
        "bb_num_stages_0": "bb_num_stages=0" in text,
        "disable_tma_count": text.count("disable_tma=True"),
    }

    rollback_env = env.copy()
    rollback_env["CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK"] = "1"
    rollback_proc = subprocess.run(
        ["python", "-m", "cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches"],
        env=rollback_env,
        capture_output=True,
        text=True,
        check=False,
    )
    restored = kernel_path.read_bytes() == original
    return {
        "kernel_path": str(kernel_path),
        "apply_returncode": apply_proc.returncode,
        "apply_output": apply_proc.stdout + apply_proc.stderr,
        "patched": patched,
        "rollback_returncode": rollback_proc.returncode,
        "rollback_output": rollback_proc.stdout + rollback_proc.stderr,
        "restored_original_bytes": restored,
    }


@app.local_entrypoint()
def main(run_id: str = "wave28_lane_c_h100", verify_applier: bool = False):
    result = verify_applier_mutation.remote() if verify_applier else run_bench.remote(run_id)
    print(json.dumps(result, indent=2, sort_keys=True))
