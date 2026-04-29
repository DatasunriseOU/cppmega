"""Nsight Compute profile for Mamba3 MIMO bwd_fwd baseline vs stage2.

This is intentionally separate from the timing harness. It runs a bounded Modal
job on H200:2, verifies ncu availability, and profiles only the bwd_fwd launch
inside a CUDA profiler range.

Example:

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1800s \
        modal run scripts/modal_mamba3_bwd_fwd_ncu_profile.py \
        --run-id mamba3_bwd_fwd_ncu_h200_20260429_1
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")
BENCH_VOLUME_NAME = os.environ.get("CPPMEGA_MODAL_BENCH_VOLUME", "cppmega-mamba3-benchmarks")
NCU_DEB_URL = os.environ.get("CPPMEGA_NCU_DEB_URL", "")

APP_NAME = "cppmega-mamba3-bwd-fwd-ncu-profile"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"
BENCH_ROOT = "/benchmarks"
BENCH_PREFIX = "mamba3_bwd_fwd_ncu_profile"
NCU_BIN = os.environ.get("CPPMEGA_NCU_BIN", "/usr/local/cuda/bin/ncu")

bench_volume = modal.Volume.from_name(BENCH_VOLUME_NAME, create_if_missing=True)


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    img = img.env(
        {
            "GHCR_REPO": GHCR_REPO,
            "GHCR_TAG": GHCR_TAG,
            "CPPMEGA_IMAGE_REF": GHCR_REF,
            "CPPMEGA_NCU_BIN": NCU_BIN,
        }
    )
    if NCU_DEB_URL:
        img = img.run_commands(
            "apt-get update",
            "apt-get install -y ca-certificates wget",
            f"wget -q {NCU_DEB_URL} -O /tmp/nsight-compute.deb",
            "apt-get install -y /tmp/nsight-compute.deb",
            "rm -f /tmp/nsight-compute.deb",
        )
    img = img.add_local_dir("cppmega", f"{CPPMEGA_ROOT}/cppmega", copy=True)
    img = img.add_local_dir("scripts", f"{CPPMEGA_ROOT}/scripts", copy=True)
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


def _ncu_version() -> dict[str, Any]:
    import shutil
    import subprocess

    proc = subprocess.run(
        [NCU_BIN, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "path": NCU_BIN,
        "which": shutil.which("ncu"),
        "exists": os.path.exists(NCU_BIN),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _device_report(requested_gpu: str) -> dict[str, Any]:
    import subprocess

    import torch

    nvidia_smi = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "requested_gpu_spec": requested_gpu,
        "image_ref": os.environ.get("CPPMEGA_IMAGE_REF", GHCR_REF),
        "nvidia_smi_returncode": nvidia_smi.returncode,
        "nvidia_smi_stdout_tail": nvidia_smi.stdout[-4000:],
        "nvidia_smi_stderr_tail": nvidia_smi.stderr[-2000:],
    }


def _write_workload_script(path: str) -> None:
    workload = r'''
from __future__ import annotations

import argparse
import json
import os
import sys

CPPMEGA_ROOT = "/opt/cppmega"
SOURCE_ROOT = "/opt/state-spaces-mamba"
for item in (CPPMEGA_ROOT, SOURCE_ROOT):
    if item not in sys.path:
        sys.path.insert(0, item)

from scripts.modal_mamba3_stage2_force_nontma_benchmark import (  # noqa: E402
    SHAPES,
    VARIANTS,
    _empty_outputs,
    _import_variant,
    _kernel_args,
    _make_inputs,
    _make_kernels,
    _prepare_variant,
    _source_markers,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--shape", default="productionish", choices=sorted(SHAPES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--launches", type=int, default=1)
    parser.add_argument("--meta-out", required=True)
    args = parser.parse_args()

    import torch

    shape = SHAPES[args.shape]
    cfg = VARIANTS[args.variant]
    path, prep_meta = _prepare_variant(args.variant)
    if prep_meta.get("patch_rc", 0) != 0:
        raise RuntimeError(f"patch failed: {prep_meta}")

    mod = _import_variant(path, args.variant, shape)
    bf_kernel, _bb_kernel, compile_meta = _make_kernels(mod, shape, args.variant)
    inputs = _make_inputs(shape)
    outputs = _empty_outputs(shape, bool(cfg["flat_qk_dot"]))
    bf_args, _bb_args = _kernel_args(
        shape,
        inputs,
        outputs,
        flattened_inputs=bool(cfg["flattened_inputs"]),
    )

    for _ in range(args.warmup):
        bf_kernel(*bf_args)
    torch.cuda.synchronize()

    source = bf_kernel.get_kernel_source()
    meta = {
        "variant": args.variant,
        "shape": shape.__dict__,
        "config": cfg,
        "prepare": prep_meta,
        "compile": compile_meta,
        "source": _source_markers(source, int(cfg["bf_threads"])),
    }
    with open(args.meta_out, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True, default=str)

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.launches):
        bf_kernel(*bf_args)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
'''
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(workload).lstrip())


def _run_ncu(
    *,
    variant: str,
    run_dir: str,
    shape: str,
    launches: int,
    ncu_timeout_sec: int,
) -> dict[str, Any]:
    import glob
    import shutil
    import subprocess

    variant_dir = os.path.join(run_dir, shape, variant)
    os.makedirs(variant_dir, exist_ok=True)
    workload_path = os.path.join(variant_dir, "ncu_bwd_fwd_workload.py")
    meta_path = os.path.join(variant_dir, "workload_meta.json")
    report_base = os.path.join(variant_dir, "bwd_fwd_ncu")
    stdout_path = os.path.join(variant_dir, "ncu_stdout.txt")
    stderr_path = os.path.join(variant_dir, "ncu_stderr.txt")
    cmd_path = os.path.join(variant_dir, "ncu_command.txt")
    _write_workload_script(workload_path)

    cmd = [
        NCU_BIN,
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--launch-count",
        str(launches),
        "--section",
        "SpeedOfLight",
        "--section",
        "Occupancy",
        "--section",
        "MemoryWorkloadAnalysis",
        "--force-overwrite",
        "--export",
        report_base,
        "--log-file",
        stdout_path,
        "python",
        workload_path,
        "--variant",
        variant,
        "--shape",
        shape,
        "--warmup",
        "1",
        "--launches",
        str(launches),
        "--meta-out",
        meta_path,
    ]
    with open(cmd_path, "w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n")

    env = dict(os.environ)
    driver_lib_paths = [
        "/usr/local/nvidia/lib64",
        "/usr/local/nvidia/lib",
        "/usr/local/cuda/compat",
        "/usr/lib/x86_64-linux-gnu",
    ]
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        [path for path in driver_lib_paths if os.path.exists(path)]
        + ([existing_ld] if existing_ld else [])
    )

    proc = subprocess.run(
        cmd,
        cwd=CPPMEGA_ROOT,
        capture_output=True,
        text=True,
        timeout=ncu_timeout_sec,
        env=env,
        check=False,
    )
    with open(stderr_path, "w", encoding="utf-8") as handle:
        handle.write(proc.stderr)
    nsight_logs: list[str] = []
    for log_path in glob.glob("/tmp/nsight-compute*.log"):
        dst = os.path.join(variant_dir, os.path.basename(log_path))
        shutil.copyfile(log_path, dst)
        nsight_logs.append(dst)
    return {
        "variant": variant,
        "returncode": proc.returncode,
        "command": cmd,
        "ld_library_path": env["LD_LIBRARY_PATH"],
        "artifacts": {
            "workload": workload_path,
            "workload_meta": meta_path,
            "ncu_command": cmd_path,
            "ncu_stdout": stdout_path,
            "ncu_stderr": stderr_path,
            "ncu_report": f"{report_base}.ncu-rep",
            "nsight_compute_logs": nsight_logs,
        },
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


@app.function(image=_image(), gpu=GPU_SPEC, timeout=1800, volumes={BENCH_ROOT: bench_volume})
def profile_bwd_fwd(
    requested_gpu: str,
    run_id: str | None,
    shape: str,
    variant_csv: str,
    launches: int,
    ncu_timeout_sec: int,
) -> dict[str, Any]:
    import time

    _install_source_paths()
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    variants = [item.strip() for item in variant_csv.split(",") if item.strip()]
    run_rel = f"{BENCH_PREFIX}/{run_id}"
    run_dir = os.path.join(BENCH_ROOT, run_rel)
    os.makedirs(run_dir, exist_ok=True)

    report: dict[str, Any] = {
        "run_id": run_id,
        "volume": BENCH_VOLUME_NAME,
        "volume_relpath": f"/{run_rel}",
        "artifact_dir": run_dir,
        "device": _device_report(requested_gpu),
        "ncu": _ncu_version(),
        "settings": {
            "shape": shape,
            "variant_csv": variant_csv,
            "variants": variants,
            "launches": launches,
            "sections": ["SpeedOfLight", "Occupancy", "MemoryWorkloadAnalysis"],
            "ncu_timeout_sec": ncu_timeout_sec,
        },
        "variants": [],
    }
    for variant in variants:
        report["variants"].append(
            _run_ncu(
                variant=variant,
                run_dir=run_dir,
                shape=shape,
                launches=launches,
                ncu_timeout_sec=ncu_timeout_sec,
            )
        )

    report["artifacts"] = {
        "report_json": os.path.join(run_dir, "report.json"),
    }
    with open(report["artifacts"]["report_json"], "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    bench_volume.commit()
    return report


@app.local_entrypoint()
def main(
    run_id: str | None = None,
    shape: str = "productionish",
    variant_csv: str = "baseline,stage2_force_nontma",
    launches: int = 1,
    ncu_timeout_sec: int = 720,
) -> None:
    result = profile_bwd_fwd.remote(
        GPU_SPEC,
        run_id,
        shape,
        variant_csv,
        launches,
        ncu_timeout_sec,
    )
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
