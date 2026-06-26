"""Modal A/B harness for guarded Mamba3 stage2 plus optional CUDA diag.

This harness keeps the mergeable stage2 candidate separate from the experimental
R x R CUDA diagonal work:

* baseline: upstream TileLang bwd_fwd + bwd_bwd;
* stage2_bf1_bb0: guarded force-nonTMA stage2 patch with bf=1, bb=0;
* stage2_bf1_bb0_plus_wave4_cuda_diag_host_split: optional read-only wave4
  CUDA diag call timed after the stage2 chain. This is a pessimistic host-call
  envelope, not a production integration.

The diag source defaults to Lane A artifacts if they have been copied into this
worktree, otherwise it can read the wave4 microkernel worktree via
CPPMEGA_MAMBA3_RR_DIAG_SOURCE_DIR. No production defaults are changed.

Run example:

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1800s \
      modal run scripts/modal_mamba3_stage2_cuda_ab_benchmark.py \
        --run-id stage2_cuda_ab_h200_20260429_1 \
        --shape-csv productionish \
        --warmup 2 \
        --iters 8 \
        --diag-mode wave4-readonly \
        --ncu
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")
BENCH_VOLUME_NAME = os.environ.get("CPPMEGA_MODAL_BENCH_VOLUME", "cppmega-mamba3-benchmarks")

APP_NAME = "cppmega-mamba3-stage2-cuda-ab-benchmark"
SOURCE_ROOT = "/opt/state-spaces-mamba"
CPPMEGA_ROOT = "/opt/cppmega"
DIAG_ROOT = "/opt/cppmega_rr_diag"
BENCH_ROOT = "/benchmarks"
BENCH_PREFIX = "mamba3_stage2_cuda_ab_benchmark"

DIAG_REQUIRED = (
    "rr_diag_wave4_cuda_microbench.py",
    "rr_diag_cuda_extension.py",
    "rr_diag_cuda_kernel.cu",
    "rr_diag_specialization.py",
    "rr_diag_wave3_microbench.py",
)

bench_volume = modal.Volume.from_name(BENCH_VOLUME_NAME, create_if_missing=True)


def _diag_source_candidates() -> list[Path]:
    env = os.environ.get("CPPMEGA_MAMBA3_RR_DIAG_SOURCE_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path("upstream_prs/examples/13_tilelang_floormod_dbz"))
    candidates.append(
        Path(
            "/home/dave/source/cppmega/.claude/worktrees/"
            "mamba3-rr-diag-microkernel/upstream_prs/examples/13_tilelang_floormod_dbz"
        )
    )
    return candidates


def _is_diag_source(path: Path) -> bool:
    return all((path / name).exists() for name in DIAG_REQUIRED)


def _resolve_diag_source_dir() -> str | None:
    for candidate in _diag_source_candidates():
        if _is_diag_source(candidate):
            return str(candidate)
    return None


DIAG_SOURCE_DIR = _resolve_diag_source_dir()


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
            "CPPMEGA_STAGE2_AB_DIAG_SOURCE_DIR": DIAG_SOURCE_DIR or "",
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
    if DIAG_SOURCE_DIR:
        img = img.add_local_dir(DIAG_SOURCE_DIR, DIAG_ROOT, copy=True)
    return img


app = modal.App(APP_NAME)


def _load_stage2_helper() -> Any:
    """Import the existing stage2 Modal script without running Modal decorators."""
    import importlib.util
    import sys
    import types

    class _NoopModalObject:
        @classmethod
        def from_name(cls, *args: Any, **kwargs: Any) -> "_NoopModalObject":
            return cls()

        @classmethod
        def from_registry(cls, *args: Any, **kwargs: Any) -> "_NoopModalObject":
            return cls()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> "_NoopModalObject":
            return self

        def __getattr__(self, name: str) -> Any:
            if name in ("function", "local_entrypoint"):
                return self._decorator_factory
            return self

        def _decorator_factory(self, *args: Any, **kwargs: Any) -> Any:
            def decorate(fn: Any) -> Any:
                return fn

            return decorate

        def env(self, *args: Any, **kwargs: Any) -> "_NoopModalObject":
            return self

        def add_local_dir(self, *args: Any, **kwargs: Any) -> "_NoopModalObject":
            return self

        def commit(self) -> None:
            return None

    fake_modal = types.ModuleType("modal")
    fake_modal.Volume = _NoopModalObject
    fake_modal.Image = _NoopModalObject
    fake_modal.Secret = _NoopModalObject
    fake_modal.App = _NoopModalObject

    old_modal = sys.modules.get("modal")
    sys.modules["modal"] = fake_modal
    try:
        path = Path(CPPMEGA_ROOT) / "scripts/modal_mamba3_stage2_force_nontma_benchmark.py"
        spec = importlib.util.spec_from_file_location("stage2_force_nontma_helper", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load stage2 helper from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["stage2_force_nontma_helper"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        if old_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = old_modal


def _load_diag_wave4() -> Any:
    import importlib.util
    import sys

    bench_dir = Path(DIAG_ROOT)
    path = bench_dir / "rr_diag_wave4_cuda_microbench.py"
    if not _is_diag_source(bench_dir):
        raise RuntimeError(
            "wave4 diag source is not mounted; set CPPMEGA_MAMBA3_RR_DIAG_SOURCE_DIR "
            "or copy Lane A artifacts into upstream_prs/examples/13_tilelang_floormod_dbz"
        )
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    spec = importlib.util.spec_from_file_location("rr_diag_wave4_cuda_microbench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load wave4 diag harness from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rr_diag_wave4_cuda_microbench"] = mod
    spec.loader.exec_module(mod)
    return mod


def _time_cuda_events(fn: Any, *, warmup: int, iters: int) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _stats(values: list[float]) -> dict[str, Any]:
    import math

    ordered = sorted(values)
    if not ordered:
        return {"count": 0}

    mean = sum(ordered) / len(ordered)
    var = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean_ms": mean,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "std_ms": math.sqrt(var),
        "samples_ms": values,
    }


def _diag_args(mod: Any, shape_name: str, warmup: int, iters: int, num_warps: int) -> Any:
    return mod.argparse.Namespace(
        shape=shape_name,
        B=1,
        S=256,
        H=4,
        N=64,
        P=128,
        R=4,
        chunk=16,
        dtype="bf16",
        device="cuda",
        seed=20260429,
        warmup=warmup,
        iters=iters,
        num_warps=num_warps,
    )


def _prepare_diag_context(
    shape_name: str,
    *,
    warmup: int,
    iters: int,
    num_warps: int,
) -> dict[str, Any]:
    import torch

    mod = _load_diag_wave4()
    args = _diag_args(mod, shape_name, warmup, iters, num_warps)
    shape = mod._shape_from_args(args)
    device = torch.device("cuda")
    dtype = mod._dtype(args.dtype)
    inputs = mod.make_inputs(shape, dtype=dtype, device=device, seed=args.seed)

    ref = mod.full_fused_reference(inputs, shape)
    cuda = mod.rr_specialized_cuda(inputs)
    torch.cuda.synchronize()
    correctness = {"cuda_rr_vs_full": mod.max_diffs(ref, cuda)}
    metadata = mod.rr_cuda_kernel_metadata(inputs)

    def run_cuda_diag() -> None:
        mod.rr_specialized_cuda(inputs)

    timing = _stats(_time_cuda_events(run_cuda_diag, warmup=warmup, iters=iters))
    return {
        "status": "ok",
        "module": mod,
        "inputs": inputs,
        "shape": asdict(shape),
        "timing": timing,
        "correctness": correctness,
        "metadata": metadata,
        "source_dir": os.environ.get("CPPMEGA_STAGE2_AB_DIAG_SOURCE_DIR", ""),
        "mounted_dir": DIAG_ROOT,
    }


def _time_stage2_chain_plus_diag(
    stage2: Any,
    shape: Any,
    inputs: dict[str, Any],
    diag_context: dict[str, Any],
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    import torch

    variant = "stage2_force_nontma"
    cfg = stage2.VARIANTS[variant]
    path, prep_meta = stage2._prepare_variant(variant)
    if prep_meta.get("patch_rc", 0) != 0:
        return {"status": "patch_failed", "prepare": prep_meta}

    mod = stage2._import_variant(path, f"{variant}_plus_diag", shape)
    bf_kernel, bb_kernel, compile_meta = stage2._make_kernels(mod, shape, variant)
    outputs = stage2._empty_outputs(shape, bool(cfg["flat_qk_dot"]))
    bf_args, bb_args = stage2._kernel_args(
        shape,
        inputs,
        outputs,
        flattened_inputs=bool(cfg["flattened_inputs"]),
    )
    diag_mod = diag_context["module"]
    diag_inputs = diag_context["inputs"]

    def run_chain() -> None:
        bf_kernel(*bf_args)
        bb_kernel(*bb_args)

    def run_chain_plus_diag() -> None:
        bf_kernel(*bf_args)
        bb_kernel(*bb_args)
        diag_mod.rr_specialized_cuda(diag_inputs)

    run_chain_plus_diag()
    torch.cuda.synchronize()
    return {
        "status": "ok",
        "prepare": prep_meta,
        "compile": compile_meta,
        "chain_only_retimed": _stats(_time_cuda_events(run_chain, warmup=warmup, iters=iters)),
        "chain_plus_cuda_diag": _stats(
            _time_cuda_events(run_chain_plus_diag, warmup=warmup, iters=iters)
        ),
        "read": (
            "Pessimistic host-split envelope: the standalone CUDA diag is added after "
            "the current stage2 chain and does not replace fused bwd_bwd work."
        ),
    }


def _strip_stage2_results(shape_result: dict[str, Any], stage2: Any) -> dict[str, Any]:
    shape_result["comparison"] = stage2._compare_shape(shape_result)
    shape_result["variants"] = [stage2._strip_tensors(item) for item in shape_result["variants"]]
    return shape_result


def _summarize_ab(report: dict[str, Any]) -> dict[str, Any]:
    shapes: list[dict[str, Any]] = []
    for shape_result in report["shapes"]:
        variants = {item["variant"]: item for item in shape_result["variants"]}
        baseline = variants.get("baseline", {})
        stage2_variant = variants.get("stage2_force_nontma", {})
        baseline_chain = (
            baseline.get("elapsed", {}).get("chain", {}).get("mean_ms")
            if baseline.get("status") == "ok"
            else None
        )
        stage2_chain = (
            stage2_variant.get("elapsed", {}).get("chain", {}).get("mean_ms")
            if stage2_variant.get("status") == "ok"
            else None
        )
        diag = shape_result.get("cuda_diag_readonly", {})
        combined = diag.get("stage2_chain_plus_diag", {})
        combined_ms = combined.get("chain_plus_cuda_diag", {}).get("mean_ms")
        diag_ms = diag.get("wave4_cuda_diag", {}).get("timing", {}).get("mean_ms")
        shapes.append(
            {
                "shape": shape_result["shape"]["name"],
                "status": shape_result["status"],
                "baseline_chain_mean_ms": baseline_chain,
                "stage2_bf1_bb0_chain_mean_ms": stage2_chain,
                "stage2_speedup_vs_baseline": (
                    baseline_chain / stage2_chain if baseline_chain and stage2_chain else None
                ),
                "diag_mode": diag.get("mode", "none"),
                "wave4_cuda_diag_mean_ms": diag_ms,
                "stage2_plus_diag_chain_mean_ms": combined_ms,
                "stage2_plus_diag_speedup_vs_baseline": (
                    baseline_chain / combined_ms if baseline_chain and combined_ms else None
                ),
                "comparison": shape_result.get("comparison", {}).get("vs_baseline", {}),
                "cuda_diag_readonly": {
                    key: value
                    for key, value in diag.items()
                    if key not in ("wave4_cuda_diag", "stage2_chain_plus_diag")
                },
            }
        )
    return {
        "run_id": report["run_id"],
        "volume": report["volume"],
        "volume_relpath": report["volume_relpath"],
        "device": report["device"],
        "tilelang": report["tilelang"],
        "settings": report["settings"],
        "artifacts": report["artifacts"],
        "ncu": report.get("ncu"),
        "shapes": shapes,
    }


def _write_summary_csv(summary: dict[str, Any], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "shape",
                "status",
                "baseline_chain_mean_ms",
                "stage2_bf1_bb0_chain_mean_ms",
                "stage2_speedup_vs_baseline",
                "diag_mode",
                "wave4_cuda_diag_mean_ms",
                "stage2_plus_diag_chain_mean_ms",
                "stage2_plus_diag_speedup_vs_baseline",
            ],
        )
        writer.writeheader()
        for row in summary["shapes"]:
            writer.writerow({field: row.get(field) for field in writer.fieldnames})


def _extract_ncu_highlights(text: str) -> list[str]:
    needles = (
        "occupancy",
        "register",
        "shared",
        "memory",
        "dram",
        "throughput",
        "launch",
        "block",
        "grid",
        "duration",
        "sm ",
        "sm__",
        "smsp__",
    )
    lines: list[str] = []
    for line in text.splitlines():
        lower = line.lower()
        if any(needle in lower for needle in needles):
            lines.append(line[:800])
        if len(lines) >= 80:
            break
    return lines


def _ncu_stage2_script(shape_kwargs: dict[str, Any]) -> str:
    return f"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

CPPMEGA_ROOT = {CPPMEGA_ROOT!r}
SOURCE_ROOT = {SOURCE_ROOT!r}
SHAPE_KWARGS = {shape_kwargs!r}


class _NoopModalObject:
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()

    @classmethod
    def from_registry(cls, *args, **kwargs):
        return cls()

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        if name in ("function", "local_entrypoint"):
            return self._decorator_factory
        return self

    def _decorator_factory(self, *args, **kwargs):
        def decorate(fn):
            return fn
        return decorate

    def env(self, *args, **kwargs):
        return self

    def add_local_dir(self, *args, **kwargs):
        return self

    def commit(self):
        return None


def load_stage2():
    fake_modal = types.ModuleType("modal")
    fake_modal.Volume = _NoopModalObject
    fake_modal.Image = _NoopModalObject
    fake_modal.Secret = _NoopModalObject
    fake_modal.App = _NoopModalObject
    old_modal = sys.modules.get("modal")
    sys.modules["modal"] = fake_modal
    try:
        path = Path(CPPMEGA_ROOT) / "scripts/modal_mamba3_stage2_force_nontma_benchmark.py"
        spec = importlib.util.spec_from_file_location("stage2_force_nontma_helper", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {{path}}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["stage2_force_nontma_helper"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        if old_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = old_modal


stage2 = load_stage2()
stage2._install_source_paths()
stage2._reset_mamba_imports()
shape = stage2.Shape(**SHAPE_KWARGS)
inputs = stage2._make_inputs(shape)
path, prep = stage2._prepare_variant("stage2_force_nontma")
if prep.get("patch_rc", 0) != 0:
    raise RuntimeError(prep)
mod = stage2._import_variant(path, "stage2_force_nontma_ncu", shape)
bf_kernel, bb_kernel, _ = stage2._make_kernels(mod, shape, "stage2_force_nontma")
cfg = stage2.VARIANTS["stage2_force_nontma"]
outputs = stage2._empty_outputs(shape, bool(cfg["flat_qk_dot"]))
bf_args, bb_args = stage2._kernel_args(
    shape,
    inputs,
    outputs,
    flattened_inputs=bool(cfg["flattened_inputs"]),
)
bf_kernel(*bf_args)
torch.cuda.synchronize()
try:
    torch.cuda.cudart().cudaProfilerStart()
except Exception:
    torch.cuda.profiler.start()
bb_kernel(*bb_args)
torch.cuda.synchronize()
try:
    torch.cuda.cudart().cudaProfilerStop()
except Exception:
    torch.cuda.profiler.stop()
"""


def _run_ncu_stage2_bwd_bwd(
    shape: Any,
    artifact_dir: str,
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    import subprocess
    import sys
    import tempfile

    ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    if not ncu or not os.path.exists(ncu):
        return {
            "status": "unavailable",
            "reason": "ncu was not found in PATH or /usr/local/cuda/bin/ncu",
        }

    version_proc = subprocess.run([ncu, "--version"], capture_output=True, text=True, check=False)
    script_path = Path(tempfile.mkdtemp(prefix="cppmega_stage2_ncu_")) / "profile_stage2_bwd_bwd.py"
    script_path.write_text(_ncu_stage2_script(asdict(shape)), encoding="utf-8")

    attempts = [
        [
            ncu,
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--launch-count",
            "1",
            "--section",
            "LaunchStats",
            "--section",
            "Occupancy",
            "--section",
            "MemoryWorkloadAnalysis",
            "--csv",
            "--page",
            "raw",
            sys.executable,
            str(script_path),
        ],
        [
            ncu,
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--launch-count",
            "1",
            "--set",
            "basic",
            "--csv",
            "--page",
            "raw",
            sys.executable,
            str(script_path),
        ],
    ]

    attempt_reports: list[dict[str, Any]] = []
    for index, command in enumerate(attempts, start=1):
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            attempt_reports.append(
                {
                    "attempt": index,
                    "status": "timeout",
                    "command": command,
                    "timeout_sec": timeout_sec,
                    "stdout_tail": (exc.stdout or "")[-4000:],
                    "stderr_tail": (exc.stderr or "")[-4000:],
                }
            )
            continue

        stdout_path = os.path.join(artifact_dir, f"ncu_attempt_{index}_stdout.csv")
        stderr_path = os.path.join(artifact_dir, f"ncu_attempt_{index}_stderr.txt")
        with open(stdout_path, "w", encoding="utf-8") as handle:
            handle.write(proc.stdout)
        with open(stderr_path, "w", encoding="utf-8") as handle:
            handle.write(proc.stderr)
        report = {
            "attempt": index,
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "command": command,
            "stdout_artifact": stdout_path,
            "stderr_artifact": stderr_path,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "highlights": _extract_ncu_highlights(proc.stdout + "\n" + proc.stderr),
        }
        attempt_reports.append(report)
        if proc.returncode == 0:
            return {
                "status": "ok",
                "ncu_path": ncu,
                "version": version_proc.stdout.strip() or version_proc.stderr.strip(),
                "shape": shape.name,
                "successful_attempt": index,
                "attempts": attempt_reports,
            }

    return {
        "status": "failed",
        "ncu_path": ncu,
        "version": version_proc.stdout.strip() or version_proc.stderr.strip(),
        "shape": shape.name,
        "attempts": attempt_reports,
    }


@app.function(image=_image(), gpu=GPU_SPEC, timeout=2400, volumes={BENCH_ROOT: bench_volume})
def run_benchmark(
    requested_gpu: str,
    run_id: str | None,
    shape_csv: str,
    warmup: int,
    iters: int,
    diag_mode: str,
    diag_num_warps: int,
    ncu: bool,
    ncu_shape: str,
    ncu_timeout_sec: int,
) -> dict[str, Any]:
    import time

    import torch

    if diag_mode not in ("none", "wave4-readonly"):
        raise ValueError("diag_mode must be one of: none, wave4-readonly")

    stage2 = _load_stage2_helper()
    stage2._install_source_paths()

    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_rel = f"{BENCH_PREFIX}/{run_id}"
    run_dir = os.path.join(BENCH_ROOT, run_rel)
    os.makedirs(run_dir, exist_ok=True)

    report: dict[str, Any] = {
        "run_id": run_id,
        "volume": BENCH_VOLUME_NAME,
        "volume_relpath": f"/{run_rel}",
        "artifact_dir": run_dir,
        "device": stage2._device_report(requested_gpu),
        "tilelang": stage2._tilelang_report(),
        "settings": {
            "shape_csv": shape_csv,
            "variants": "baseline,stage2_force_nontma",
            "stage2_candidate": "stage2_bf1_bb0",
            "warmup": warmup,
            "iters": iters,
            "diag_mode": diag_mode,
            "diag_num_warps": diag_num_warps,
            "diag_source_dir": os.environ.get("CPPMEGA_STAGE2_AB_DIAG_SOURCE_DIR", ""),
            "ncu": ncu,
            "ncu_shape": ncu_shape,
        },
        "shapes": [],
    }

    selected_shapes = stage2._selected_shapes(shape_csv)
    for shape in selected_shapes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        shape_dir = os.path.join(run_dir, shape.name)
        os.makedirs(shape_dir, exist_ok=True)
        inputs = stage2._make_inputs(shape)
        shape_result: dict[str, Any] = {
            "shape": asdict(shape),
            "estimated_tensor_bytes": stage2._shape_bytes_estimate(shape),
            "variants": [],
            "status": "ok",
        }
        for variant in ("baseline", "stage2_force_nontma"):
            stage2._reset_mamba_imports()
            torch.cuda.empty_cache()
            variant_result = stage2._benchmark_variant(
                variant,
                shape,
                inputs,
                shape_dir,
                warmup=warmup,
                iters=iters,
                torch_profile=False,
            )
            shape_result["variants"].append(variant_result)
            if variant_result.get("status") != "ok":
                shape_result["status"] = "variant_failed"

        if diag_mode == "wave4-readonly" and shape_result["status"] == "ok":
            diag_result: dict[str, Any] = {"mode": diag_mode}
            try:
                diag_context = _prepare_diag_context(
                    shape.name,
                    warmup=warmup,
                    iters=iters,
                    num_warps=diag_num_warps,
                )
                diag_public = {key: value for key, value in diag_context.items() if key not in ("module", "inputs")}
                diag_result["wave4_cuda_diag"] = diag_public
                diag_result["stage2_chain_plus_diag"] = _time_stage2_chain_plus_diag(
                    stage2,
                    shape,
                    inputs,
                    diag_context,
                    warmup=warmup,
                    iters=iters,
                )
                diag_result["status"] = diag_result["stage2_chain_plus_diag"].get("status", "ok")
            except Exception as exc:  # noqa: BLE001
                import traceback

                diag_result.update(
                    {
                        "status": "failed",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "traceback_tail": traceback.format_exc()[-4000:],
                    }
                )
                shape_result["status"] = "diag_failed"
            shape_result["cuda_diag_readonly"] = diag_result
        else:
            shape_result["cuda_diag_readonly"] = {"mode": diag_mode, "status": "skipped"}

        report["shapes"].append(_strip_stage2_results(shape_result, stage2))

    if ncu:
        ncu_matches = [shape for shape in selected_shapes if shape.name == ncu_shape]
        if ncu_matches:
            ncu_dir = os.path.join(run_dir, "ncu")
            os.makedirs(ncu_dir, exist_ok=True)
            report["ncu"] = _run_ncu_stage2_bwd_bwd(
                ncu_matches[0],
                ncu_dir,
                timeout_sec=ncu_timeout_sec,
            )
        else:
            report["ncu"] = {
                "status": "skipped",
                "reason": f"ncu_shape={ncu_shape!r} was not included in shape_csv={shape_csv!r}",
            }

    report["artifacts"] = {
        "report_json": os.path.join(run_dir, "report.json"),
        "summary_json": os.path.join(run_dir, "summary.json"),
        "summary_csv": os.path.join(run_dir, "summary.csv"),
    }
    summary = _summarize_ab(report)
    with open(report["artifacts"]["report_json"], "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    with open(report["artifacts"]["summary_json"], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    _write_summary_csv(summary, report["artifacts"]["summary_csv"])
    bench_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_id: str | None = None,
    shape_csv: str = "productionish",
    warmup: int = 2,
    iters: int = 8,
    diag_mode: str = "wave4-readonly",
    diag_num_warps: int = 4,
    ncu: bool = False,
    ncu_shape: str = "productionish",
    ncu_timeout_sec: int = 600,
) -> None:
    result = run_benchmark.remote(
        GPU_SPEC,
        run_id,
        shape_csv,
        warmup,
        iters,
        diag_mode,
        diag_num_warps,
        ncu,
        ncu_shape,
        ncu_timeout_sec,
    )
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
