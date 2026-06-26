"""Wave7 Modal profiling harness for Mamba3 stage2 bwd_bwd and related kernels.

This is intentionally separate from the wave6 profiler harness. It installs an
Nsight Compute 2025.3.1 package into the Modal image and prefers that binary over
the CUDA 13.2 bundled Nsight Compute 2026.1 binary, avoiding the known R580/R595
tool-driver mismatch seen in wave6.

Example:

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 3600s \
      modal run scripts/modal_mamba3_stage2_profile_wave7.py \
        --run-id mamba3_stage2_profile_wave7_h200_prod_20260430_1 \
        --shape-csv productionish \
        --variant-csv stage2_force_nontma \
        --phase-csv bwd_bwd \
        --warmup 2 \
        --iters 8
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")
BENCH_VOLUME_NAME = os.environ.get("CPPMEGA_MODAL_BENCH_VOLUME", "cppmega-mamba3-benchmarks")

APP_NAME = "cppmega-mamba3-stage2-profile-wave7"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"
BENCH_ROOT = "/benchmarks"
BENCH_PREFIX = "mamba3_stage2_profile_wave7"
NCU_2025_PACKAGE_URL = os.environ.get(
    "CPPMEGA_NCU_2025_PACKAGE_URL",
    "https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/"
    "nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb",
)
NCU_2025_EXTRACT_ROOT = "/opt/nvidia/nsight-compute-2025.3.1-deb"
NCU_2025_VERSION_PREFIX = os.environ.get("CPPMEGA_NCU_2025_VERSION_PREFIX", "2025.3")
DEFAULT_NCU_BIN = os.environ.get(
    "CPPMEGA_NCU_BIN",
    f"{NCU_2025_EXTRACT_ROOT}/opt/nvidia/nsight-compute/2025.3.1/ncu",
)

bench_volume = modal.Volume.from_name(BENCH_VOLUME_NAME, create_if_missing=True)


def _install_ncu_2025_command() -> str:
    return (
        "bash -lc '"
        "set -eux; "
        "apt-get update; "
        "apt-get install -y --no-install-recommends ca-certificates wget; "
        f"rm -rf {NCU_2025_EXTRACT_ROOT}; "
        f"mkdir -p {NCU_2025_EXTRACT_ROOT}; "
        "tmp_dir=$(mktemp -d); "
        f"wget -nv -O \"$tmp_dir/nsight-compute-2025.3.1.deb\" \"{NCU_2025_PACKAGE_URL}\"; "
        f"dpkg-deb -x \"$tmp_dir/nsight-compute-2025.3.1.deb\" {NCU_2025_EXTRACT_ROOT}; "
        f"ncu_path=$(find {NCU_2025_EXTRACT_ROOT} -name ncu -type f | sort | head -n 1); "
        "test -n \"$ncu_path\"; "
        "ln -sf \"$ncu_path\" /usr/local/bin/ncu-2025.3; "
        "\"$ncu_path\" --version; "
        "rm -rf \"$tmp_dir\" /var/lib/apt/lists/*"
        "'"
    )


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    img = img.run_commands(_install_ncu_2025_command())
    img = img.env(
        {
            "GHCR_REPO": GHCR_REPO,
            "GHCR_TAG": GHCR_TAG,
            "CPPMEGA_IMAGE_REF": GHCR_REF,
            "CPPMEGA_NCU_BIN": DEFAULT_NCU_BIN,
            "CPPMEGA_NCU_2025_PACKAGE_URL": NCU_2025_PACKAGE_URL,
            "CPPMEGA_NCU_2025_VERSION_PREFIX": NCU_2025_VERSION_PREFIX,
        }
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


def _safe_tail(text: str | bytes | None, limit: int = 4000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return text[-limit:]


def _run_command(
    command: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_sec: int | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
) -> dict[str, Any]:
    import subprocess

    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        status = "ok" if proc.returncode == 0 else "failed"
        returncode: int | None = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _safe_tail(exc.stdout, 20000)
        stderr = _safe_tail(exc.stderr, 20000)
        status = "timeout"
        returncode = None

    if stdout_path is not None:
        Path(stdout_path).write_text(stdout, encoding="utf-8")
    if stderr_path is not None:
        Path(stderr_path).write_text(stderr, encoding="utf-8")
    return {
        "status": status,
        "returncode": returncode,
        "command": command,
        "stdout_artifact": stdout_path,
        "stderr_artifact": stderr_path,
        "stdout_tail": _safe_tail(stdout),
        "stderr_tail": _safe_tail(stderr),
    }


def _run_probe_command(name: str, command: list[str], artifact_dir: str) -> dict[str, Any]:
    return _run_command(
        command,
        timeout_sec=60,
        stdout_path=os.path.join(artifact_dir, f"{name}_stdout.txt"),
        stderr_path=os.path.join(artifact_dir, f"{name}_stderr.txt"),
    )


def _glob_existing(patterns: list[str]) -> list[str]:
    import glob

    paths: list[str] = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.exists(path):
                paths.append(path)
    return sorted(set(paths))


def _ncu_ld_paths() -> list[str]:
    candidates = [
        "/usr/local/nvidia/lib64",
        "/usr/local/nvidia/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/compat",
        "/usr/local/cuda/extras/CUPTI/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    candidates.extend(
        _glob_existing(
            [
                "/usr/local/cuda/nsight-compute*/host/linux-desktop-glibc_2_11_3-x64",
                "/usr/local/cuda/nsight-compute*/target/linux-desktop-glibc_2_11_3-x64",
                "/opt/nvidia/nsight-compute*/host/linux-desktop-glibc_2_11_3-x64",
                "/opt/nvidia/nsight-compute*/target/linux-desktop-glibc_2_11_3-x64",
                "/opt/nvidia/nsight-compute/*/host/linux-desktop-glibc_2_11_3-x64",
                "/opt/nvidia/nsight-compute/*/target/linux-desktop-glibc_2_11_3-x64",
                f"{NCU_2025_EXTRACT_ROOT}/opt/nvidia/nsight-compute/*/host/linux-desktop-glibc_2_11_3-x64",
                f"{NCU_2025_EXTRACT_ROOT}/opt/nvidia/nsight-compute/*/target/linux-desktop-glibc_2_11_3-x64",
            ]
        )
    )
    return [path for path in candidates if os.path.exists(path)]


def _with_ncu_ld_fix(base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    existing = env.get("LD_LIBRARY_PATH", "")
    paths = _ncu_ld_paths()
    if existing:
        paths.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(path for path in paths if path))
    return env


def _with_ncu_host_driver_ld(base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    existing = [
        path
        for path in env.get("LD_LIBRARY_PATH", "").split(":")
        if path and path != "/usr/local/cuda/compat"
    ]
    preferred = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/nvidia/lib64",
        "/usr/local/nvidia/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
    ]
    preferred.extend(
        _glob_existing(
            [
                f"{NCU_2025_EXTRACT_ROOT}/opt/nvidia/nsight-compute/*/host/linux-desktop-glibc_2_11_3-x64",
                f"{NCU_2025_EXTRACT_ROOT}/opt/nvidia/nsight-compute/*/target/linux-desktop-glibc_2_11_3-x64",
            ]
        )
    )
    paths = [*preferred, *existing]
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(path for path in paths if os.path.exists(path)))
    return env


def _version_tuple(text: str) -> tuple[int, ...]:
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text)
    if not match:
        return ()
    return tuple(int(part) for part in match.groups(default="0"))


def _ncu_candidate_paths() -> list[str]:
    candidates = [
        os.environ.get("CPPMEGA_NCU_BIN", ""),
        DEFAULT_NCU_BIN,
        "/usr/local/bin/ncu-2025.3",
        f"{NCU_2025_EXTRACT_ROOT}/opt/nvidia/nsight-compute/2025.3.1/ncu",
        "/opt/nvidia/nsight-compute/2025.3.1/ncu",
        "/usr/local/cuda-13.0/bin/ncu",
        "/usr/local/cuda/bin/ncu",
        shutil.which("ncu") or "",
    ]
    candidates.extend(
        _glob_existing(
            [
                f"{NCU_2025_EXTRACT_ROOT}/opt/nvidia/nsight-compute/*/ncu",
                f"{NCU_2025_EXTRACT_ROOT}/**/ncu",
                "/opt/nvidia/nsight-compute/*/ncu",
                "/opt/nvidia/nsight-compute*/ncu",
                "/opt/nvidia/nsight-compute/*/target/linux-*/ncu",
                "/opt/nvidia/nsight-compute*/target/linux-*/ncu",
                "/usr/local/cuda/nsight-compute*/ncu",
                "/usr/local/cuda/nsight-compute*/target/linux-*/ncu",
            ]
        )
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _select_ncu_bin() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for path in _ncu_candidate_paths():
        exists = os.path.exists(path)
        executable = os.access(path, os.X_OK) if exists else False
        report: dict[str, Any] = {
            "path": path,
            "exists": exists,
            "executable": executable,
            "realpath": os.path.realpath(path) if exists else path,
        }
        if exists and executable:
            version = _run_command([path, "--version"], timeout_sec=60)
            version_text = f"{version.get('stdout_tail', '')}\n{version.get('stderr_tail', '')}".strip()
            report.update(
                {
                    "version_status": version.get("status"),
                    "version_returncode": version.get("returncode"),
                    "version_text": version_text,
                    "version_tuple": _version_tuple(version_text),
                }
            )
            if version.get("status") == "ok" and fallback is None:
                fallback = report
            if version.get("status") == "ok" and NCU_2025_VERSION_PREFIX in version_text:
                selected = report
        candidates.append(report)
        if selected is not None:
            break

    chosen = selected or fallback
    return {
        "status": "ok" if chosen else "unavailable",
        "ncu_bin": chosen.get("path") if chosen else None,
        "selected_realpath": chosen.get("realpath") if chosen else None,
        "selected_version_text": chosen.get("version_text") if chosen else "",
        "selection_reason": (
            f"preferred Nsight Compute {NCU_2025_VERSION_PREFIX}"
            if selected
            else ("fallback first runnable ncu" if fallback else "no runnable ncu found")
        ),
        "required_version_prefix": NCU_2025_VERSION_PREFIX,
        "candidates": candidates,
    }


def _environment_report(run_dir: str) -> dict[str, Any]:
    import subprocess

    env_dir = os.path.join(run_dir, "environment")
    os.makedirs(env_dir, exist_ok=True)
    ncu_selection = _select_ncu_bin()
    ncu_bin = ncu_selection.get("ncu_bin") or os.environ.get("CPPMEGA_NCU_BIN", DEFAULT_NCU_BIN)
    ncu_resolved = os.path.realpath(ncu_bin) if os.path.exists(ncu_bin) else ncu_bin
    nsys_bin = shutil.which("nsys") or "/usr/local/cuda/bin/nsys"

    report: dict[str, Any] = {
        "env": {
            "PATH": os.environ.get("PATH", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
            "NVIDIA_DRIVER_CAPABILITIES": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", ""),
        },
        "ncu_bin": ncu_bin,
        "ncu_resolved": ncu_resolved,
        "ncu_selection": ncu_selection,
        "ncu_2025_package_url": NCU_2025_PACKAGE_URL,
        "ncu_2025_extract_root": NCU_2025_EXTRACT_ROOT,
        "ncu_ld_fix_paths": _ncu_ld_paths(),
        "ncu_host_driver_ld_library_path": _with_ncu_host_driver_ld(os.environ).get("LD_LIBRARY_PATH", ""),
        "nsys_bin": nsys_bin,
        "which": {
            "ncu": shutil.which("ncu"),
            "ncu-2025.3": shutil.which("ncu-2025.3"),
            "nsys": shutil.which("nsys"),
            "strace": shutil.which("strace"),
            "sqlite3": shutil.which("sqlite3"),
        },
        "library_inventory": {
            "libcuda": _glob_existing(
                [
                    "/usr/local/nvidia/lib*/libcuda.so*",
                    "/usr/local/cuda/compat/libcuda.so*",
                    "/usr/lib/x86_64-linux-gnu/libcuda.so*",
                ]
            ),
            "libcupti": _glob_existing(
                [
                    "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so*",
                    "/usr/local/cuda/lib64/libcupti.so*",
                    "/usr/lib/x86_64-linux-gnu/libcupti.so*",
                ]
            ),
            "libnvidia_ml": _glob_existing(
                [
                    "/usr/local/nvidia/lib*/libnvidia-ml.so*",
                    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so*",
                ]
            ),
        },
        "commands": {},
    }

    probes = {
        "nvidia_smi": ["nvidia-smi"],
        "nvidia_smi_query": [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pci.bus_id",
            "--format=csv,noheader",
        ],
        "ncu_version": [ncu_bin, "--version"],
        "nsys_version": [nsys_bin, "--version"],
        "dpkg_nsight_compute": [
            "dpkg-query",
            "-W",
            "nsight-compute-2025.3.1",
            "nsight-compute-2026.1.1",
            "cuda-nsight-compute-13-2",
        ],
    }
    if os.path.exists(ncu_resolved):
        probes["ncu_file"] = ["file", ncu_resolved]
        probes["ncu_ldd"] = ["ldd", ncu_resolved]
    for name, command in probes.items():
        if shutil.which(command[0]) or os.path.exists(command[0]):
            report["commands"][name] = _run_probe_command(name, command, env_dir)
        else:
            report["commands"][name] = {"status": "unavailable", "command": command}

    params_path = Path("/proc/driver/nvidia/params")
    version_path = Path("/proc/driver/nvidia/version")
    for path in (params_path, version_path):
        try:
            report[path.as_posix()] = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            report[path.as_posix()] = f"unavailable: {exc}"

    try:
        report["ldconfig_cuda"] = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        ).stdout
    except Exception as exc:  # noqa: BLE001
        report["ldconfig_cuda"] = f"unavailable: {type(exc).__name__}: {exc}"
    return report


def _extract_ncu_highlights(text: str) -> list[str]:
    needles = (
        "librarynotloaded",
        "compatible driver",
        "launch",
        "register",
        "shared",
        "occupancy",
        "duration",
        "sm__",
        "smsp__",
        "dram__",
        "throughput",
        "roofline",
        "fatal",
        "timeconversion",
        "internalerrorexception",
        "error",
        "warning",
    )
    lines: list[str] = []
    for line in text.splitlines():
        lower = line.lower()
        if any(needle in lower for needle in needles):
            lines.append(line[:1000])
        if len(lines) >= 120:
            break
    return lines


def _parse_ncu_csv_metrics(stdout_path: str | None) -> dict[str, Any]:
    if not stdout_path or not os.path.exists(stdout_path):
        return {"status": "missing", "stdout_path": stdout_path}

    interesting_terms = (
        "launch__",
        "occupancy",
        "gpu__time_duration",
        "sm__throughput",
        "sm__warps_active",
        "smsp__warps_active",
        "dram__",
        "lts__",
        "l1tex__",
        "memory",
        "throughput",
        "bytes",
    )
    try:
        rows: list[dict[str, str]] = []
        header: list[str] | None = None
        with open(stdout_path, newline="", encoding="utf-8", errors="replace") as handle:
            for raw in csv.reader(handle):
                if not raw:
                    continue
                if "Metric Name" in raw and "Metric Value" in raw:
                    header = raw
                    continue
                if header is None or len(raw) < len(header):
                    continue
                record = dict(zip(header, raw))
                metric = record.get("Metric Name", "")
                section = record.get("Section Name", "")
                haystack = f"{section} {metric}".lower()
                if any(term in haystack for term in interesting_terms):
                    rows.append(
                        {
                            "kernel": record.get("Kernel Name", ""),
                            "section": section,
                            "metric": metric,
                            "unit": record.get("Metric Unit", ""),
                            "value": record.get("Metric Value", ""),
                        }
                    )
                if len(rows) >= 160:
                    break
        return {
            "status": "ok",
            "stdout_path": stdout_path,
            "interesting_metrics": rows,
            "metric_count": len(rows),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "stdout_path": stdout_path,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }


def _write_workload_script(path: str) -> None:
    workload = r'''
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def _kernel_resources(kernel):
    data = {
        "type": f"{type(kernel).__module__}.{type(kernel).__name__}",
        "repr": repr(kernel),
    }
    for attr in ("dynamic_smem_bytes", "shared_memory_size", "smem_bytes"):
        value = getattr(kernel, attr, None)
        if isinstance(value, (int, float, str, bool)) or value is None:
            data[attr] = value
    for meth in ("get_dynamic_smem_bytes", "get_shared_memory_size"):
        fn = getattr(kernel, meth, None)
        if callable(fn):
            try:
                data[meth] = fn()
            except Exception as exc:  # noqa: BLE001
                data[meth] = f"{type(exc).__name__}: {exc}"
    interesting = []
    for name in dir(kernel):
        lower = name.lower()
        if any(part in lower for part in ("smem", "shared", "reg", "occup", "func", "mod", "kernel")):
            interesting.append(name)
    data["interesting_attrs"] = sorted(interesting)[:100]
    return data


def _profiler_start():
    import torch

    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception:
        torch.cuda.profiler.start()


def _profiler_stop():
    import torch

    try:
        torch.cuda.cudart().cudaProfilerStop()
    except Exception:
        torch.cuda.profiler.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--shape", default="productionish", choices=sorted(SHAPES))
    parser.add_argument("--phase", required=True, choices=("bwd_fwd", "bwd_bwd", "chain"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--launches", type=int, default=1)
    parser.add_argument("--meta-out", required=True)
    parser.add_argument("--source-dir", required=True)
    args = parser.parse_args()

    import torch

    shape = SHAPES[args.shape]
    cfg = VARIANTS[args.variant]
    source_dir = Path(args.source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    path, prep_meta = _prepare_variant(args.variant)
    if prep_meta.get("patch_rc", 0) != 0:
        raise RuntimeError(f"patch failed: {prep_meta}")

    mod = _import_variant(path, args.variant, shape)
    bf_kernel, bb_kernel, compile_meta = _make_kernels(mod, shape, args.variant)
    inputs = _make_inputs(shape)
    outputs = _empty_outputs(shape, bool(cfg["flat_qk_dot"]))
    bf_args, bb_args = _kernel_args(
        shape,
        inputs,
        outputs,
        flattened_inputs=bool(cfg["flattened_inputs"]),
    )

    bf_source = bf_kernel.get_kernel_source()
    bb_source = bb_kernel.get_kernel_source()
    (source_dir / "bwd_fwd_kernel_source.cu").write_text(bf_source, encoding="utf-8")
    (source_dir / "bwd_bwd_kernel_source.cu").write_text(bb_source, encoding="utf-8")
    meta = {
        "variant": args.variant,
        "phase": args.phase,
        "shape": shape.__dict__,
        "config": cfg,
        "prepare": prep_meta,
        "compile": compile_meta,
        "source": {
            "bwd_fwd": _source_markers(bf_source, int(cfg["bf_threads"])),
            "bwd_bwd": _source_markers(bb_source, int(cfg["bb_threads"])),
        },
        "resources": {
            "bwd_fwd": _kernel_resources(bf_kernel),
            "bwd_bwd": _kernel_resources(bb_kernel),
        },
    }
    Path(args.meta_out).write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def run_bwd_fwd():
        torch.cuda.nvtx.range_push(f"{args.variant}.bwd_fwd")
        bf_kernel(*bf_args)
        torch.cuda.nvtx.range_pop()

    def run_bwd_bwd():
        torch.cuda.nvtx.range_push(f"{args.variant}.bwd_bwd")
        bb_kernel(*bb_args)
        torch.cuda.nvtx.range_pop()

    def run_chain():
        torch.cuda.nvtx.range_push(f"{args.variant}.chain")
        bf_kernel(*bf_args)
        bb_kernel(*bb_args)
        torch.cuda.nvtx.range_pop()

    # bwd_bwd depends on bwd_fwd-produced state/qk_dot. Keep that setup outside
    # the profiler range so NCU/NSYS isolate the requested phase.
    if args.phase == "bwd_bwd":
        bf_kernel(*bf_args)
        active = run_bwd_bwd
    elif args.phase == "bwd_fwd":
        active = run_bwd_fwd
    else:
        active = run_chain

    for _ in range(args.warmup):
        active()
    torch.cuda.synchronize()

    _profiler_start()
    for _ in range(args.launches):
        active()
    torch.cuda.synchronize()
    _profiler_stop()


if __name__ == "__main__":
    main()
'''
    Path(path).write_text(textwrap.dedent(workload).lstrip(), encoding="utf-8")


def _workload_args(
    workload_path: str,
    *,
    variant: str,
    shape: str,
    phase: str,
    warmup: int,
    launches: int,
    meta_path: str,
    source_dir: str,
) -> list[str]:
    import sys

    return [
        sys.executable,
        workload_path,
        "--variant",
        variant,
        "--shape",
        shape,
        "--phase",
        phase,
        "--warmup",
        str(warmup),
        "--launches",
        str(launches),
        "--meta-out",
        meta_path,
        "--source-dir",
        source_dir,
    ]


def _ncu_attempts(ncu_bin: str, report_base: str, launches: int) -> list[dict[str, Any]]:
    common = [
        ncu_bin,
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--launch-count",
        str(launches),
        "--clock-control",
        "none",
        "--force-overwrite",
        "--csv",
        "--page",
        "raw",
    ]
    return [
        {
            "name": "launch_occupancy_memory_host_driver_ld",
            "ld_fix": True,
            "ld_mode": "host_driver",
            "args": common
            + [
                "--section",
                "LaunchStats",
                "--section",
                "Occupancy",
                "--section",
                "MemoryWorkloadAnalysis",
                "--export",
                f"{report_base}_launch_occupancy_memory_host_driver_ld",
            ],
        },
        {
            "name": "launch_occupancy_memory_ldfix",
            "ld_fix": True,
            "ld_mode": "compat_ldfix",
            "args": common
            + [
                "--section",
                "LaunchStats",
                "--section",
                "Occupancy",
                "--section",
                "MemoryWorkloadAnalysis",
                "--export",
                f"{report_base}_launch_occupancy_memory",
            ],
        },
        {
            "name": "launchstats_ldfix",
            "ld_fix": True,
            "ld_mode": "compat_ldfix",
            "args": common + ["--section", "LaunchStats", "--export", f"{report_base}_launchstats"],
        },
        {
            "name": "launchstats_default_env",
            "ld_fix": False,
            "ld_mode": "default",
            "args": common + ["--section", "LaunchStats", "--export", f"{report_base}_default_env"],
        },
        {
            "name": "basic_ldfix",
            "ld_fix": True,
            "ld_mode": "compat_ldfix",
            "args": common + ["--set", "basic", "--export", f"{report_base}_basic"],
        },
        {
            "name": "speed_of_light_ldfix",
            "ld_fix": True,
            "ld_mode": "compat_ldfix",
            "args": common
            + [
                "--section",
                "SpeedOfLight",
                "--export",
                f"{report_base}_speed_of_light",
            ],
        },
        {
            "name": "launch_metrics_ldfix",
            "ld_fix": True,
            "ld_mode": "compat_ldfix",
            "args": common
            + [
                "--metrics",
                (
                    "launch__registers_per_thread,"
                    "launch__shared_mem_per_block_static,"
                    "launch__shared_mem_per_block_dynamic"
                ),
                "--export",
                f"{report_base}_launch_metrics",
            ],
        },
    ]


def _run_strace_ncu(
    command: list[str],
    *,
    artifact_dir: str,
    env: dict[str, str],
    timeout_sec: int,
) -> dict[str, Any]:
    strace = shutil.which("strace")
    if not strace:
        return {"status": "unavailable", "reason": "strace not found"}
    strace_path = os.path.join(artifact_dir, "ncu_strace_openat.log")
    stdout_path = os.path.join(artifact_dir, "ncu_strace_stdout.txt")
    stderr_path = os.path.join(artifact_dir, "ncu_strace_stderr.txt")
    strace_command = [
        strace,
        "-f",
        "-e",
        "trace=openat,access",
        "-o",
        strace_path,
        *command,
    ]
    result = _run_command(
        strace_command,
        cwd=CPPMEGA_ROOT,
        env=env,
        timeout_sec=timeout_sec,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    result["strace_artifact"] = strace_path
    return result


def _run_ncu_profile(
    *,
    run_dir: str,
    variant: str,
    shape: str,
    phase: str,
    warmup: int,
    launches: int,
    timeout_sec: int,
    strace_on_failure: bool,
) -> dict[str, Any]:
    ncu_selection = _select_ncu_bin()
    ncu_bin = ncu_selection.get("ncu_bin")
    if not ncu_bin or not os.path.exists(ncu_bin):
        return {
            "status": "unavailable",
            "reason": f"compatible ncu not found: {ncu_bin}",
            "selection": ncu_selection,
        }

    artifact_dir = os.path.join(run_dir, shape, variant, phase, "ncu")
    os.makedirs(artifact_dir, exist_ok=True)
    workload_path = os.path.join(artifact_dir, "profile_workload.py")
    meta_path = os.path.join(artifact_dir, "workload_meta.json")
    source_dir = os.path.join(artifact_dir, "sources")
    report_base = os.path.join(artifact_dir, f"{variant}_{phase}")
    _write_workload_script(workload_path)
    workload_args = _workload_args(
        workload_path,
        variant=variant,
        shape=shape,
        phase=phase,
        warmup=warmup,
        launches=launches,
        meta_path=meta_path,
        source_dir=source_dir,
    )

    version = _run_command(
        [ncu_bin, "--version"],
        timeout_sec=60,
        stdout_path=os.path.join(artifact_dir, "ncu_version_stdout.txt"),
        stderr_path=os.path.join(artifact_dir, "ncu_version_stderr.txt"),
    )
    base_env = dict(os.environ)
    ld_env = _with_ncu_ld_fix(base_env)
    host_driver_env = _with_ncu_host_driver_ld(base_env)
    attempts: list[dict[str, Any]] = []
    first_failed_ld_command: list[str] | None = None
    for attempt in _ncu_attempts(ncu_bin, report_base, launches):
        command = [*attempt["args"], *workload_args]
        command_path = os.path.join(artifact_dir, f"{attempt['name']}_command.json")
        Path(command_path).write_text(json.dumps(command, indent=2), encoding="utf-8")
        ld_mode = attempt.get("ld_mode", "compat_ldfix" if attempt["ld_fix"] else "default")
        if ld_mode == "host_driver":
            env = host_driver_env
        elif attempt["ld_fix"]:
            env = ld_env
        else:
            env = base_env
        stdout_path = os.path.join(artifact_dir, f"{attempt['name']}_stdout.csv")
        stderr_path = os.path.join(artifact_dir, f"{attempt['name']}_stderr.txt")
        result = _run_command(
            command,
            cwd=CPPMEGA_ROOT,
            env=env,
            timeout_sec=timeout_sec,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        report = {
            "name": attempt["name"],
            "ld_fix": attempt["ld_fix"],
            "ld_mode": ld_mode,
            "ld_library_path": env.get("LD_LIBRARY_PATH", ""),
            "command_artifact": command_path,
            **result,
        }
        combined = (
            Path(stdout_path).read_text(encoding="utf-8", errors="replace")
            + "\n"
            + Path(stderr_path).read_text(encoding="utf-8", errors="replace")
        )
        report["highlights"] = _extract_ncu_highlights(combined)
        attempts.append(report)
        if report["status"] == "ok":
            return {
                "status": "ok",
                "ncu_bin": ncu_bin,
                "selection": ncu_selection,
                "version": version,
                "successful_attempt": attempt["name"],
                "workload": workload_path,
                "workload_meta": meta_path,
                "source_dir": source_dir,
                "attempts": attempts,
            }
        if attempt["ld_fix"] and first_failed_ld_command is None:
            first_failed_ld_command = command

    strace_result = None
    if strace_on_failure and first_failed_ld_command is not None:
        strace_result = _run_strace_ncu(
            first_failed_ld_command,
            artifact_dir=artifact_dir,
            env=ld_env,
            timeout_sec=min(timeout_sec, 180),
        )

    return {
        "status": "failed",
        "ncu_bin": ncu_bin,
        "selection": ncu_selection,
        "version": version,
        "workload": workload_path,
        "workload_meta": meta_path,
        "source_dir": source_dir,
        "attempts": attempts,
        "strace": strace_result,
    }


def _run_nsys_stats(nsys_bin: str, rep_path: str, artifact_dir: str) -> dict[str, Any]:
    reports = ("cuda_gpu_kern_sum", "cuda_gpu_trace", "nvtx_sum")
    results: dict[str, Any] = {}
    for report_name in reports:
        output_base = os.path.join(artifact_dir, f"nsys_{report_name}")
        stdout_path = os.path.join(artifact_dir, f"nsys_stats_{report_name}_stdout.txt")
        stderr_path = os.path.join(artifact_dir, f"nsys_stats_{report_name}_stderr.txt")
        result = _run_command(
            [
                nsys_bin,
                "stats",
                "--report",
                report_name,
                "--format",
                "csv",
                "--output",
                output_base,
                rep_path,
            ],
            timeout_sec=300,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        result["csv_artifacts"] = _glob_existing([f"{output_base}*.csv"])
        results[report_name] = result
    return results


def _query_nsys_sqlite(sqlite_path: str) -> dict[str, Any]:
    import sqlite3

    if not os.path.exists(sqlite_path):
        return {"status": "missing", "sqlite": sqlite_path}
    try:
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        tables = [
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        kernel_table = next((name for name in tables if name == "CUPTI_ACTIVITY_KIND_KERNEL"), None)
        if kernel_table is None:
            kernel_table = next((name for name in tables if "KERNEL" in name.upper()), None)
        if kernel_table is None:
            return {"status": "no_kernel_table", "tables": tables[:100]}

        cols = [row["name"] for row in conn.execute(f"PRAGMA table_info({kernel_table})")]
        name_expr = "NULL"
        join = ""
        if "demangledName" in cols:
            name_expr = "demangledName"
        elif "name" in cols:
            name_expr = "name"
        elif "shortName" in cols and "StringIds" in tables:
            name_expr = "StringIds.value"
            join = f" LEFT JOIN StringIds ON {kernel_table}.shortName = StringIds.id"
        elif "shortName" in cols:
            name_expr = "shortName"

        def col(name: str) -> str:
            return name if name in cols else "NULL"

        has_duration = "start" in cols and "end" in cols
        duration_expr = "(end - start) / 1000000.0" if has_duration else "NULL"
        order_expr = "(end - start) DESC" if has_duration else "rowid DESC"
        query = f"""
            SELECT
              {name_expr} AS name,
              {duration_expr} AS duration_ms,
              {col("gridX")} AS grid_x,
              {col("gridY")} AS grid_y,
              {col("gridZ")} AS grid_z,
              {col("blockX")} AS block_x,
              {col("blockY")} AS block_y,
              {col("blockZ")} AS block_z,
              {col("registersPerThread")} AS registers_per_thread,
              {col("staticSharedMemory")} AS static_smem_bytes,
              {col("dynamicSharedMemory")} AS dynamic_smem_bytes
            FROM {kernel_table}
            {join}
            ORDER BY {order_expr}
            LIMIT 30
        """
        rows = [dict(row) for row in conn.execute(query)]
        return {
            "status": "ok",
            "sqlite": sqlite_path,
            "kernel_table": kernel_table,
            "columns": cols,
            "top_kernels": rows,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "sqlite": sqlite_path,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
    finally:
        try:
            conn.close()  # type: ignore[possibly-undefined]
        except Exception:
            pass


def _run_nsys_profile(
    *,
    run_dir: str,
    variant: str,
    shape: str,
    phase: str,
    warmup: int,
    launches: int,
    timeout_sec: int,
) -> dict[str, Any]:
    nsys_bin = shutil.which("nsys") or "/usr/local/cuda/bin/nsys"
    if not os.path.exists(nsys_bin):
        return {"status": "unavailable", "reason": f"nsys not found: {nsys_bin}"}

    artifact_dir = os.path.join(run_dir, shape, variant, phase, "nsys")
    os.makedirs(artifact_dir, exist_ok=True)
    workload_path = os.path.join(artifact_dir, "profile_workload.py")
    meta_path = os.path.join(artifact_dir, "workload_meta.json")
    source_dir = os.path.join(artifact_dir, "sources")
    _write_workload_script(workload_path)
    workload_args = _workload_args(
        workload_path,
        variant=variant,
        shape=shape,
        phase=phase,
        warmup=warmup,
        launches=launches,
        meta_path=meta_path,
        source_dir=source_dir,
    )
    attempts = [
        {
            "name": "cudaProfilerApi",
            "args": [
                nsys_bin,
                "profile",
                "--trace=cuda,nvtx",
                "--sample=none",
                "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--force-overwrite=true",
                "--output",
                os.path.join(artifact_dir, "cuda_profiler_api"),
            ],
        },
        {
            "name": "full",
            "args": [
                nsys_bin,
                "profile",
                "--trace=cuda,nvtx",
                "--sample=none",
                "--cpuctxsw=none",
                "--force-overwrite=true",
                "--output",
                os.path.join(artifact_dir, "full"),
            ],
        },
    ]

    attempt_reports: list[dict[str, Any]] = []
    for attempt in attempts:
        output_prefix = attempt["args"][-1]
        stdout_path = os.path.join(artifact_dir, f"{attempt['name']}_stdout.txt")
        stderr_path = os.path.join(artifact_dir, f"{attempt['name']}_stderr.txt")
        command = [*attempt["args"], *workload_args]
        result = _run_command(
            command,
            cwd=CPPMEGA_ROOT,
            timeout_sec=timeout_sec,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        rep_path = f"{output_prefix}.nsys-rep"
        report = {
            "name": attempt["name"],
            "report": rep_path,
            **result,
        }
        attempt_reports.append(report)
        if result["status"] != "ok" or not os.path.exists(rep_path):
            continue

        stats = _run_nsys_stats(nsys_bin, rep_path, artifact_dir)
        sqlite_path = os.path.join(artifact_dir, f"{attempt['name']}.sqlite")
        export = _run_command(
            [
                nsys_bin,
                "export",
                "--type",
                "sqlite",
                "--force-overwrite=true",
                "--output",
                sqlite_path,
                rep_path,
            ],
            timeout_sec=300,
            stdout_path=os.path.join(artifact_dir, f"{attempt['name']}_export_stdout.txt"),
            stderr_path=os.path.join(artifact_dir, f"{attempt['name']}_export_stderr.txt"),
        )
        sqlite_summary = _query_nsys_sqlite(sqlite_path)
        report.update({"stats": stats, "export": export, "sqlite_summary": sqlite_summary})
        return {
            "status": "ok",
            "nsys_bin": nsys_bin,
            "successful_attempt": attempt["name"],
            "workload": workload_path,
            "workload_meta": meta_path,
            "source_dir": source_dir,
            "attempts": attempt_reports,
        }

    return {
        "status": "failed",
        "nsys_bin": nsys_bin,
        "workload": workload_path,
        "workload_meta": meta_path,
        "source_dir": source_dir,
        "attempts": attempt_reports,
    }


def _load_json_if_exists(path: str | None) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed_to_load", "exception_type": type(exc).__name__, "exception": str(exc)}


def _summarize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    ncu = profile.get("ncu", {})
    nsys = profile.get("nsys", {})
    meta = _load_json_if_exists(ncu.get("workload_meta"))
    if meta is None:
        meta = _load_json_if_exists(nsys.get("workload_meta"))
    ncu_metrics = None
    if ncu.get("status") == "ok":
        successful_attempt = ncu.get("successful_attempt")
        attempt = next(
            (
                item
                for item in ncu.get("attempts", [])
                if item.get("name") == successful_attempt or item.get("status") == "ok"
            ),
            None,
        )
        if attempt is not None:
            ncu_metrics = _parse_ncu_csv_metrics(attempt.get("stdout_artifact"))
    nsys_top = None
    nsys_highlights: list[str] = []
    if nsys.get("status") == "ok":
        for attempt in nsys.get("attempts", []):
            sqlite_summary = attempt.get("sqlite_summary", {})
            if sqlite_summary.get("status") == "ok":
                nsys_top = sqlite_summary.get("top_kernels", [])[:8]
                break
    else:
        for attempt in nsys.get("attempts", []):
            combined = f"{attempt.get('stdout_tail', '')}\n{attempt.get('stderr_tail', '')}"
            nsys_highlights.extend(_extract_ncu_highlights(combined))
    return {
        "shape": profile["shape"],
        "variant": profile["variant"],
        "phase": profile["phase"],
        "ncu_status": ncu.get("status"),
        "ncu_bin": ncu.get("ncu_bin"),
        "ncu_selected_version": ncu.get("selection", {}).get("selected_version_text"),
        "ncu_successful_attempt": ncu.get("successful_attempt"),
        "ncu_metrics": ncu_metrics,
        "ncu_highlights": [
            line
            for attempt in ncu.get("attempts", [])
            for line in attempt.get("highlights", [])
        ][:20],
        "nsys_status": nsys.get("status"),
        "nsys_successful_attempt": nsys.get("successful_attempt"),
        "nsys_highlights": nsys_highlights[:20],
        "nsys_top_kernels": nsys_top,
        "workload_resource_meta": {
            "resources": meta.get("resources") if isinstance(meta, dict) else None,
            "source": meta.get("source") if isinstance(meta, dict) else None,
        },
    }


def _write_profile_summary_csv(summary: dict[str, Any], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "shape",
                "variant",
                "phase",
                "ncu_status",
                "ncu_successful_attempt",
                "nsys_status",
                "nsys_successful_attempt",
                "top_kernel",
                "top_kernel_duration_ms",
                "top_kernel_registers_per_thread",
                "top_kernel_static_smem_bytes",
                "top_kernel_dynamic_smem_bytes",
            ],
        )
        writer.writeheader()
        for profile in summary["profiles"]:
            top = (profile.get("nsys_top_kernels") or [{}])[0]
            writer.writerow(
                {
                    "shape": profile["shape"],
                    "variant": profile["variant"],
                    "phase": profile["phase"],
                    "ncu_status": profile.get("ncu_status"),
                    "ncu_successful_attempt": profile.get("ncu_successful_attempt"),
                    "nsys_status": profile.get("nsys_status"),
                    "nsys_successful_attempt": profile.get("nsys_successful_attempt"),
                    "top_kernel": top.get("name"),
                    "top_kernel_duration_ms": top.get("duration_ms"),
                    "top_kernel_registers_per_thread": top.get("registers_per_thread"),
                    "top_kernel_static_smem_bytes": top.get("static_smem_bytes"),
                    "top_kernel_dynamic_smem_bytes": top.get("dynamic_smem_bytes"),
                }
            )


def _selected_variants_no_auto_baseline(stage2: Any, variant_csv: str) -> list[str]:
    variants: list[str] = []
    for name in [part.strip() for part in variant_csv.split(",")]:
        if not name:
            continue
        if name not in stage2.VARIANTS:
            raise ValueError(f"unknown variant {name!r}; choose one of {sorted(stage2.VARIANTS)}")
        variants.append(name)
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


def _infer_ncu_root_cause(environment: dict[str, Any], profiles: list[dict[str, Any]]) -> dict[str, Any]:
    all_failed = profiles and all(profile.get("ncu", {}).get("status") != "ok" for profile in profiles)
    failure_text = "\n".join(
        line
        for profile in profiles
        for attempt in profile.get("ncu", {}).get("attempts", [])
        for line in attempt.get("highlights", [])
    )
    ncu_version_text = (
        environment.get("commands", {})
        .get("ncu_version", {})
        .get("stdout_tail", "")
        + "\n"
        + environment.get("commands", {})
        .get("ncu_version", {})
        .get("stderr_tail", "")
    )
    smi_query = environment.get("commands", {}).get("nvidia_smi_query", {}).get("stdout_tail", "")
    driver_match = re.search(r",\s*([0-9]+(?:\.[0-9]+)+)\s*,", smi_query)
    driver_version = driver_match.group(1) if driver_match else None
    selected_version_text = environment.get("ncu_selection", {}).get("selected_version_text", "")
    ncu_version = _version_tuple(ncu_version_text)
    driver_tuple = _version_tuple(driver_version or "")

    likely: list[str] = []
    if "LibraryNotLoaded" in failure_text or "compatible driver" in failure_text:
        likely.append("ncu reports LibraryNotLoaded / compatible-driver initialization failure")
    if NCU_2025_VERSION_PREFIX in selected_version_text and driver_tuple >= (580, 82, 7):
        likely.append(
            "Nsight Compute 2025.3 was selected and the visible driver satisfies the documented R580 floor"
        )
    if ncu_version >= (2026, 1) and driver_tuple and driver_tuple < (595, 58, 3):
        likely.append(
            "Nsight Compute 2026.1 is present while the visible NVIDIA driver is older than 595.58.03"
        )
    if all_failed and not likely:
        likely.append("all NCU attempts failed; inspect per-attempt stderr and strace artifacts")
    return {
        "status": "suspected" if likely else "unknown",
        "all_ncu_attempts_failed": all_failed,
        "ncu_version_text": ncu_version_text.strip(),
        "selected_ncu_version_text": selected_version_text,
        "driver_version": driver_version,
        "signals": likely,
    }


@app.function(image=_image(), gpu=GPU_SPEC, timeout=4200, volumes={BENCH_ROOT: bench_volume})
def run_profile(
    requested_gpu: str,
    run_id: str | None,
    shape_csv: str,
    variant_csv: str,
    phase_csv: str,
    warmup: int,
    iters: int,
    profiler_warmup: int,
    profiler_launches: int,
    ncu_timeout_sec: int,
    nsys_timeout_sec: int,
    nsys_fallback: bool,
    strace_on_ncu_failure: bool,
) -> dict[str, Any]:
    import time

    _install_source_paths()
    import torch
    from scripts import modal_mamba3_stage2_force_nontma_benchmark as stage2

    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_rel = f"{BENCH_PREFIX}/{run_id}"
    run_dir = os.path.join(BENCH_ROOT, run_rel)
    os.makedirs(run_dir, exist_ok=True)

    selected_shapes = stage2._selected_shapes(shape_csv)
    selected_variants = _selected_variants_no_auto_baseline(stage2, variant_csv)
    phases = [part.strip() for part in phase_csv.split(",") if part.strip()]
    bad_phases = [phase for phase in phases if phase not in ("bwd_fwd", "bwd_bwd", "chain")]
    if bad_phases:
        raise ValueError(f"unknown phases: {bad_phases}")

    report: dict[str, Any] = {
        "run_id": run_id,
        "volume": BENCH_VOLUME_NAME,
        "volume_relpath": f"/{run_rel}",
        "artifact_dir": run_dir,
        "device": stage2._device_report(requested_gpu),
        "tilelang": stage2._tilelang_report(),
        "environment": _environment_report(run_dir),
        "settings": {
            "shape_csv": shape_csv,
            "variant_csv": variant_csv,
            "phase_csv": phase_csv,
            "variants": selected_variants,
            "warmup": warmup,
            "iters": iters,
            "profiler_warmup": profiler_warmup,
            "profiler_launches": profiler_launches,
            "ncu_timeout_sec": ncu_timeout_sec,
            "nsys_timeout_sec": nsys_timeout_sec,
            "nsys_fallback": nsys_fallback,
            "strace_on_ncu_failure": strace_on_ncu_failure,
        },
        "timing_shapes": [],
        "profiles": [],
    }
    report["artifacts"] = {
        "report_json": os.path.join(run_dir, "report.json"),
        "summary_json": os.path.join(run_dir, "summary.json"),
        "summary_csv": os.path.join(run_dir, "summary.csv"),
        "partial_report_json": os.path.join(run_dir, "partial_report.json"),
    }

    timing_dir = os.path.join(run_dir, "timing")
    os.makedirs(timing_dir, exist_ok=True)
    for shape in selected_shapes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        shape_dir = os.path.join(timing_dir, shape.name)
        os.makedirs(shape_dir, exist_ok=True)
        inputs = stage2._make_inputs(shape)
        shape_result: dict[str, Any] = {
            "shape": asdict(shape),
            "estimated_tensor_bytes": stage2._shape_bytes_estimate(shape),
            "variants": [],
            "status": "ok",
        }
        for variant in selected_variants:
            stage2._reset_mamba_imports()
            torch.cuda.empty_cache()
            variant_result = stage2._benchmark_variant(
                variant,
                shape,
                inputs,
                shape_dir,
                warmup=warmup,
                iters=iters,
                torch_profile=True,
            )
            shape_result["variants"].append(variant_result)
            if variant_result.get("status") != "ok":
                shape_result["status"] = "variant_failed"
        shape_result["comparison"] = stage2._compare_shape(shape_result)
        shape_result["variants"] = [stage2._strip_tensors(item) for item in shape_result["variants"]]
        report["timing_shapes"].append(shape_result)

    with open(report["artifacts"]["partial_report_json"], "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    bench_volume.commit()

    for shape in selected_shapes:
        for variant in selected_variants:
            for phase in phases:
                profile: dict[str, Any] = {
                    "shape": shape.name,
                    "variant": variant,
                    "phase": phase,
                }
                ncu_profile = _run_ncu_profile(
                    run_dir=run_dir,
                    variant=variant,
                    shape=shape.name,
                    phase=phase,
                    warmup=profiler_warmup,
                    launches=profiler_launches,
                    timeout_sec=ncu_timeout_sec,
                    strace_on_failure=strace_on_ncu_failure,
                )
                profile["ncu"] = ncu_profile
                if ncu_profile.get("status") == "ok":
                    profile["nsys"] = {"status": "skipped", "reason": "NCU succeeded"}
                elif not nsys_fallback:
                    profile["nsys"] = {
                        "status": "skipped",
                        "reason": "nsys fallback disabled for this run",
                    }
                else:
                    profile["nsys"] = _run_nsys_profile(
                        run_dir=run_dir,
                        variant=variant,
                        shape=shape.name,
                        phase=phase,
                        warmup=profiler_warmup,
                        launches=profiler_launches,
                        timeout_sec=nsys_timeout_sec,
                    )
                report["profiles"].append(profile)
                with open(report["artifacts"]["partial_report_json"], "w", encoding="utf-8") as handle:
                    json.dump(report, handle, indent=2, sort_keys=True, default=str)
                bench_volume.commit()

    summary = {
        "run_id": run_id,
        "volume": BENCH_VOLUME_NAME,
        "volume_relpath": f"/{run_rel}",
        "device": report["device"],
        "tilelang": report["tilelang"],
        "settings": report["settings"],
        "timing_summary": stage2._summarize_report(
            {
                "run_id": run_id,
                "volume": BENCH_VOLUME_NAME,
                "volume_relpath": f"/{run_rel}",
                "device": report["device"],
                "settings": report["settings"],
                "artifacts": report["artifacts"],
                "shapes": report["timing_shapes"],
            }
        ),
        "profiles": [_summarize_profile(profile) for profile in report["profiles"]],
        "ncu_root_cause": _infer_ncu_root_cause(report["environment"], report["profiles"]),
        "artifacts": report["artifacts"],
    }
    with open(report["artifacts"]["report_json"], "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    with open(report["artifacts"]["summary_json"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    _write_profile_summary_csv(summary, report["artifacts"]["summary_csv"])
    bench_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_id: str | None = None,
    shape_csv: str = "productionish",
    variant_csv: str = "baseline,stage2_force_nontma",
    phase_csv: str = "bwd_bwd,bwd_fwd",
    warmup: int = 2,
    iters: int = 8,
    profiler_warmup: int = 1,
    profiler_launches: int = 1,
    ncu_timeout_sec: int = 480,
    nsys_timeout_sec: int = 600,
    nsys_fallback: bool = True,
    strace_on_ncu_failure: bool = True,
) -> None:
    result = run_profile.remote(
        GPU_SPEC,
        run_id,
        shape_csv,
        variant_csv,
        phase_csv,
        warmup,
        iters,
        profiler_warmup,
        profiler_launches,
        ncu_timeout_sec,
        nsys_timeout_sec,
        nsys_fallback,
        strace_on_ncu_failure,
    )
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
