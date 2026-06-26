"""Modal/local probe for the Mamba3 bwd_bwd PsiV-hoist patch.

The patch is intentionally non-production: it materializes PsiV in bwd_fwd into
a temporary row-flattened cache and has bwd_bwd consume it instead of rebuilding
PsiV from V and MIMO_V.

Examples:

    python scripts/modal_mamba3_psiv_hoist_probe.py --local-dry-run
    CPPMEGA_MODAL_GPU=H200:1 timeout 10m modal run scripts/modal_mamba3_psiv_hoist_probe.py
    CPPMEGA_MODAL_GPU=H200:1 CPPMEGA_PSIV_SHAPE=productionish timeout 15m modal run scripts/modal_mamba3_psiv_hoist_probe.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import modal


GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "latest")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:1")

APP_NAME = "cppmega-mamba3-psiv-hoist-probe"
CPPMEGA_ROOT = "/opt/cppmega"
SOURCE_ROOT = "/opt/state-spaces-mamba"
PATCH_BASENAME = "mamba3_bwd_psiv_hoist_probe.patch"
AFTER_STAGE2_PATCH_BASENAME = "mamba3_bwd_psiv_hoist_after_stage2_probe.patch"
STAGE2_PATCH_SOURCE = (
    "/home/dave/source/cppmega/.claude/worktrees/mamba3-stage2-force-nontma/"
    "upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_stage2_force_nontma.patch"
)
STAGE2_PATCH_BASENAME = "mamba3_bwd_stage2_force_nontma.patch"
PATCH_MODE = os.environ.get("CPPMEGA_PSIV_PATCH_MODE", "after_stage2")
SHAPE_NAME = os.environ.get("CPPMEGA_PSIV_SHAPE", "smoke")
WARMUP = int(os.environ.get("CPPMEGA_PSIV_WARMUP", "2"))
ITERS = int(os.environ.get("CPPMEGA_PSIV_ITERS", "6"))


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
        }
    )
    img = img.add_local_dir(
        "upstream_prs/examples/13_tilelang_floormod_dbz",
        f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz",
        copy=True,
    )
    if Path(STAGE2_PATCH_SOURCE).exists():
        img = img.add_local_file(
            STAGE2_PATCH_SOURCE,
            f"{CPPMEGA_ROOT}/upstream_prs/examples/13_tilelang_floormod_dbz/{STAGE2_PATCH_BASENAME}",
            copy=True,
        )
    img = img.add_local_dir(
        "/home/dave/state-spaces-mamba/mamba_ssm",
        f"{SOURCE_ROOT}/mamba_ssm",
        copy=True,
    )
    return img


app = modal.App(APP_NAME)


def _source_file(source_root: str) -> Path:
    return Path(source_root) / "mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py"


def _patch_file(cppmega_root: str, basename: str = PATCH_BASENAME) -> Path:
    path = (
        Path(cppmega_root)
        / "upstream_prs/examples/13_tilelang_floormod_dbz"
        / basename
    )
    if basename == STAGE2_PATCH_BASENAME and not path.exists():
        return Path(STAGE2_PATCH_SOURCE)
    return path


def _run_patch(dst: Path, patch_path: Path, *, dry_run: bool) -> dict[str, Any]:
    cmd = ["patch"]
    if dry_run:
        cmd.append("--dry-run")
    cmd += ["-p4", str(dst)]

    proc = subprocess.run(
        cmd,
        input=patch_path.read_bytes(),
        capture_output=True,
        cwd=dst.parent,
        check=False,
    )
    return {
        "patch": str(patch_path),
        "dry_run": dry_run,
        "patch_rc": proc.returncode,
        "patch_stdout_tail": proc.stdout.decode(errors="replace")[-4000:],
        "patch_stderr_tail": proc.stderr.decode(errors="replace")[-4000:],
    }


def _apply_patch_to_temp(
    source_root: str,
    cppmega_root: str,
    *,
    dry_run: bool,
    patch_mode: str = PATCH_MODE,
) -> dict[str, Any]:
    work = Path(tempfile.mkdtemp(prefix="cppmega_mamba3_psiv_hoist_"))
    dst = work / "mamba3_mimo_bwd.py"
    shutil.copy(_source_file(source_root), dst)

    patches: list[dict[str, Any]] = []
    if patch_mode == "after_stage2":
        # Even for local validation, apply the stage2 base to the temp file so
        # the PsiV incremental patch is dry-run against the intended source.
        patches.append(_run_patch(dst, _patch_file(cppmega_root, STAGE2_PATCH_BASENAME), dry_run=False))
        patches[-1]["base_apply_for_dry_run"] = dry_run
        if patches[-1]["patch_rc"] != 0:
            return {"work": str(work), "source": str(dst), "patch_mode": patch_mode, "patches": patches}
        patches.append(_run_patch(dst, _patch_file(cppmega_root, AFTER_STAGE2_PATCH_BASENAME), dry_run=dry_run))
    elif patch_mode == "standalone":
        patches.append(_run_patch(dst, _patch_file(cppmega_root, PATCH_BASENAME), dry_run=dry_run))
    else:
        raise ValueError(f"unknown patch_mode {patch_mode!r}")

    return {
        "work": str(work),
        "source": str(dst),
        "patch_mode": patch_mode,
        "patches": patches,
        "patch_rc": patches[-1]["patch_rc"] if patches else None,
    }


def _apply_stage2_to_temp(source_root: str, cppmega_root: str, *, dry_run: bool) -> dict[str, Any]:
    work = Path(tempfile.mkdtemp(prefix="cppmega_mamba3_stage2_only_"))
    dst = work / "mamba3_mimo_bwd.py"
    shutil.copy(_source_file(source_root), dst)
    patch = _run_patch(dst, _patch_file(cppmega_root, STAGE2_PATCH_BASENAME), dry_run=dry_run)
    return {
        "work": str(work),
        "source": str(dst),
        "patch_mode": "stage2_only",
        "patches": [patch],
        "patch_rc": patch["patch_rc"],
    }


def _load_module(path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _device_report() -> dict[str, Any]:
    import torch

    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "image_ref": os.environ.get("CPPMEGA_IMAGE_REF", GHCR_REF),
    }


def _make_inputs(shape: dict[str, int]) -> dict[str, Any]:
    import torch

    torch.manual_seed(1234)
    B, S, H, G, N, P, R, chunk = (shape[k] for k in ("B", "S", "H", "G", "N", "P", "R", "chunk"))
    nchunks = (S + chunk - 1) // chunk
    dtype = torch.bfloat16
    dev = "cuda"
    return {
        "dout": torch.randn(B, S, H, P, device=dev, dtype=dtype) * 0.01,
        "q": torch.randn(B, S, R, G, N, device=dev, dtype=dtype) * 0.01,
        "k": torch.randn(B, S, R, G, N, device=dev, dtype=dtype) * 0.01,
        "v": torch.randn(B, S, H, P, device=dev, dtype=dtype) * 0.01,
        "q_bias": torch.randn(H, R, N, device=dev, dtype=torch.float32) * 0.01,
        "k_bias": torch.randn(H, R, N, device=dev, dtype=torch.float32) * 0.01,
        "mimo_v": torch.randn(H, R, P, device=dev, dtype=torch.float32) * 0.01,
        "mimo_o": torch.randn(H, R, P, device=dev, dtype=torch.float32) * 0.01,
        "z": None,
        "mimo_z": None,
        "angles": torch.randn(B, S, H, N // 4, device=dev, dtype=torch.float32) * 0.01,
        "dA_cs": torch.randn(B, H, S, device=dev, dtype=torch.float32) * 0.01,
        "dA_cs_rev": torch.randn(B, H, S, device=dev, dtype=torch.float32) * 0.01,
        "dt": torch.randn(B, H, S, device=dev, dtype=torch.float32) * 0.01,
        "trap": torch.randn(B, H, S, device=dev, dtype=dtype) * 0.01,
        "D": None,
        "segsum": torch.randn(B, H, nchunks, chunk, chunk, device=dev, dtype=torch.float32) * 0.01,
        "chunk_size": chunk,
        "rotary_dim_divisor": 4,
        "dtype": dtype,
        "bf_threads": 128,
        "bf_num_stages": 0,
        "bb_threads": 256,
        "bb_num_stages": 0,
    }


def _shape_catalog() -> dict[str, dict[str, int]]:
    return {
        "smoke": {"B": 1, "S": 64, "H": 4, "G": 1, "N": 64, "P": 64, "R": 4, "chunk": 16},
        "productionish": {"B": 4, "S": 4096, "H": 32, "G": 1, "N": 64, "P": 128, "R": 4, "chunk": 16},
    }


def _selected_shape(name: str) -> dict[str, int]:
    catalog = _shape_catalog()
    if name not in catalog:
        raise ValueError(f"unknown CPPMEGA_PSIV_SHAPE={name!r}; choose one of {sorted(catalog)}")
    return catalog[name]


def _run_combined(mod: Any, inputs: dict[str, Any]) -> tuple[Any, ...]:
    return mod.mamba_mimo_bwd_combined(**inputs)


def _compare_outputs(lhs: tuple[Any, ...], rhs: tuple[Any, ...]) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    names = [
        "dQ", "dK", "dV", "dADT", "dDT", "dTrap", "dQ_bias", "dK_bias",
        "dMIMO_V", "dMIMO_Z", "dMIMO_Out", "dAngles", "dD", "dZ",
    ]
    for name, a, b in zip(names, lhs, rhs, strict=True):
        if a is None or b is None:
            rows.append({"name": name, "status": "none" if a is b else "mismatch_none"})
            continue
        diff = (a.float() - b.float()).abs()
        denom = torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-6)
        rows.append(
            {
                "name": name,
                "shape": list(a.shape),
                "absmax": float(diff.max().item()),
                "relmax": float((diff / denom).max().item()),
                "bit_exact": bool(torch.equal(a, b)),
            }
        )
    return rows


def _time_combined(mod: Any, inputs: dict[str, Any], *, warmup: int = 2, iters: int = 6) -> dict[str, Any]:
    import time
    import torch

    for _ in range(warmup):
        _run_combined(mod, inputs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _run_combined(mod, inputs)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "warmup": warmup,
        "iters": iters,
        "total_ms": elapsed_ms,
        "mean_ms": elapsed_ms / iters,
    }


@app.function(image=_image(), gpu=GPU_SPEC, timeout=900)
def remote_smoke(patch_mode: str, shape_name: str, warmup: int, iters: int) -> dict[str, Any]:
    sys.path.insert(0, SOURCE_ROOT)
    report: dict[str, Any] = {
        "device": _device_report(),
        "settings": {
            "patch_mode": patch_mode,
            "shape_name": shape_name,
            "warmup": warmup,
            "iters": iters,
        },
    }
    prep = _apply_patch_to_temp(SOURCE_ROOT, CPPMEGA_ROOT, dry_run=False, patch_mode=patch_mode)
    report["prepare"] = prep
    if prep["patch_rc"] != 0:
        report["status"] = "patch_failed"
        return report
    if patch_mode == "after_stage2":
        base_prep = _apply_stage2_to_temp(SOURCE_ROOT, CPPMEGA_ROOT, dry_run=False)
        report["baseline_prepare"] = base_prep
        if base_prep["patch_rc"] != 0:
            report["status"] = "baseline_patch_failed"
            return report
        baseline_source = base_prep["source"]
        baseline_label = "stage2"
    else:
        baseline_source = str(_source_file(SOURCE_ROOT))
        baseline_label = "upstream"

    shape = _selected_shape(shape_name)
    report["shape"] = shape
    report["baseline_label"] = baseline_label
    try:
        import torch

        base = _load_module(baseline_source, "mamba3_bwd_base_psiv_probe")
        patched = _load_module(prep["source"], "mamba3_bwd_patched_psiv_probe")
        inputs = _make_inputs(shape)
        torch.cuda.synchronize()
        base_out = _run_combined(base, inputs)
        torch.cuda.synchronize()
        patched_out = _run_combined(patched, inputs)
        torch.cuda.synchronize()
        report["comparison"] = _compare_outputs(base_out, patched_out)
        report["timing"] = {
            baseline_label: _time_combined(base, inputs, warmup=warmup, iters=iters),
            "psiv_hoist": _time_combined(patched, inputs, warmup=warmup, iters=iters),
        }
        report["status"] = "smoke_ok"
    except BaseException as exc:
        report["status"] = "smoke_failed"
        report["exception_type"] = type(exc).__name__
        report["exception"] = str(exc)
        report["traceback_tail"] = traceback.format_exc()[-6000:]
    return report


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(remote_smoke.remote(PATCH_MODE, SHAPE_NAME, WARMUP, ITERS), indent=2, sort_keys=True))


def _local_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dry-run", action="store_true")
    parser.add_argument("--patch-mode", choices=["after_stage2", "standalone"], default=PATCH_MODE)
    args = parser.parse_args()
    if args.local_dry_run:
        root = os.environ.get("MAMBA_SOURCE_ROOT", "/home/dave/state-spaces-mamba")
        report = _apply_patch_to_temp(root, os.getcwd(), dry_run=True, patch_mode=args.patch_mode)
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        raise SystemExit("Use `modal run ...` for H200 smoke or --local-dry-run for local patch validation.")


if __name__ == "__main__":
    _local_main()
