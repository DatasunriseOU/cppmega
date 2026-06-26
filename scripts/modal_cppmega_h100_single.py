"""Modal H100:1 probe for the cppmega runtime image plus current repo overlay.

This is the smallest useful Modal execution surface for cppmega itself:

    CPPMEGA_MODAL_GPU=H100:1 modal run scripts/modal_cppmega_h100_single.py

It does not use nanochat code, GCP, or external training data.  The probe pulls
the cppmega GHCR runtime image, overlays the current local repo code into
``/opt/cppmega``, checks the patched Megatron/cppmega import surface, and
validates cppmega's real Megatron sidecar data path.  By default this requires
an already prepared Megatron ``.bin/.idx/.json`` prefix with the full sidecar
contract. Formatting from legacy parquet is opt-in via
``CPPMEGA_MODAL_FORMAT_SIDECAR=1`` and an explicit
``CPPMEGA_MODAL_PARQUET_DATASET``.
"""

from __future__ import annotations

import importlib.util
import importlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import traceback
from typing import Any

import modal

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "latest")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H100:1")
OVERLAY_REPO = os.environ.get("CPPMEGA_MODAL_OVERLAY_REPO", "1") != "0"
INSTALL_CUTLASS = os.environ.get("CPPMEGA_MODAL_INSTALL_CUTLASS", "0") == "1"
INSTALL_FLASH_ATTN_4 = os.environ.get("CPPMEGA_MODAL_INSTALL_FLASH_ATTN_4", "0") == "1"
RUN_TINY_CUDA_SMOKE = os.environ.get("CPPMEGA_MODAL_TINY_CUDA_SMOKE", "0") == "1"
RUN_SIDECAR_SMOKE = os.environ.get("CPPMEGA_MODAL_SIDECAR_SMOKE", "1") != "0"
FORMAT_SIDECAR_IF_MISSING = os.environ.get("CPPMEGA_MODAL_FORMAT_SIDECAR", "0") == "1"
DATA_VOLUME_NAME = os.environ.get("CPPMEGA_MODAL_DATA_VOLUME", "nanochat-training-data")
DATA_MOUNT = os.environ.get("CPPMEGA_MODAL_DATA_MOUNT", "/data_vol")
PARQUET_ROOT = os.environ.get("CPPMEGA_MODAL_PARQUET_ROOT", f"{DATA_MOUNT}/parquet")
PARQUET_DATASET = os.environ.get("CPPMEGA_MODAL_PARQUET_DATASET", "")
FORMAT_MAX_SHARDS = int(os.environ.get("CPPMEGA_MODAL_FORMAT_MAX_SHARDS", "1"))
SIDECAR_OUTPUT_ROOT = os.environ.get(
    "CPPMEGA_MODAL_SIDECAR_OUTPUT_ROOT",
    "/tmp/cppmega_sidecar_smoke",
)


def _local_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


LOCAL_GIT_SHA = _local_git_sha()


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    ).env(
        {
            "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
            "WANDB_MODE": "disabled",
            "CPPMEGA_IMAGE_REF": GHCR_REF,
            "CPPMEGA_GHCR_REPO": GHCR_REPO,
            "CPPMEGA_GHCR_TAG": GHCR_TAG,
            "CPPMEGA_LOCAL_GIT_SHA": LOCAL_GIT_SHA,
            "CPPMEGA_OVERLAY_REPO": "1" if OVERLAY_REPO else "0",
            "CPPMEGA_INSTALL_CUTLASS_OVERLAY": "1" if INSTALL_CUTLASS else "0",
            "CPPMEGA_INSTALL_FLASH_ATTN_4_OVERLAY": "1" if INSTALL_FLASH_ATTN_4 else "0",
            "CPPMEGA_MODAL_TINY_CUDA_SMOKE": "1" if RUN_TINY_CUDA_SMOKE else "0",
            "CPPMEGA_MODAL_SIDECAR_SMOKE": "1" if RUN_SIDECAR_SMOKE else "0",
            "CPPMEGA_MODAL_FORMAT_SIDECAR": "1" if FORMAT_SIDECAR_IF_MISSING else "0",
            "CPPMEGA_MODAL_DATA_VOLUME": DATA_VOLUME_NAME,
            "CPPMEGA_MODAL_DATA_MOUNT": DATA_MOUNT,
            "CPPMEGA_MODAL_PARQUET_ROOT": PARQUET_ROOT,
            "CPPMEGA_MODAL_PARQUET_DATASET": PARQUET_DATASET,
            "CPPMEGA_MODAL_FORMAT_MAX_SHARDS": str(FORMAT_MAX_SHARDS),
            "CPPMEGA_MODAL_SIDECAR_OUTPUT_ROOT": SIDECAR_OUTPUT_ROOT,
            "CPPMEGA_MODAL_DATA_PREFIX": os.environ.get("CPPMEGA_MODAL_DATA_PREFIX", ""),
            "CPPMEGA_MODAL_DATA_PREFIXES": os.environ.get("CPPMEGA_MODAL_DATA_PREFIXES", ""),
        }
    )
    if INSTALL_CUTLASS:
        img = img.pip_install(
            "nvidia-cutlass-dsl==4.4.2",
            "quack-kernels==0.3.10",
            extra_index_url="https://pypi.nvidia.com",
        )
    if INSTALL_FLASH_ATTN_4:
        img = img.pip_install(
            "flash-attn-4[cu13]==4.0.0b19",
            extra_index_url="https://pypi.nvidia.com",
        )
    if OVERLAY_REPO:
        img = (
            img.add_local_dir(str(_REPO_ROOT / "cppmega"), remote_path="/opt/cppmega/cppmega", copy=True)
            .add_local_dir(str(_REPO_ROOT / "scripts"), remote_path="/opt/cppmega/scripts", copy=True)
            .add_local_file(str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml")
        )
    return img


app = modal.App("cppmega-h100-single")
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)


def _import_report(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "ok": True,
            "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }


def _tiny_torch_train() -> dict[str, Any]:
    import torch

    torch.manual_seed(17)
    device = torch.device("cuda")
    model = torch.nn.Sequential(
        torch.nn.Embedding(1024, 128),
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(16 * 128, 1024),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses: list[float] = []
    for step in range(3):
        x = torch.randint(0, 1024, (8, 16), device=device)
        y = torch.randint(0, 1024, (8,), device=device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach().cpu()))
    return {
        "ok": True,
        "losses": losses,
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
    }


def _dtype_from_sidecar(dtype: str) -> Any:
    import numpy as np

    if dtype == "uint32":
        # Converter accepts uint32 for old CLI compatibility but writes int32
        # because Megatron MMIDIDX has no uint32 dtype code.
        return np.int32
    return np.dtype(dtype)


def _load_default_side_channels() -> list[tuple[str, str]]:
    spec_path = pathlib.Path("/opt/cppmega/scripts/data/prepare_format_megacpp.py")
    module_spec = importlib.util.spec_from_file_location(
        "prepare_format_megacpp",
        spec_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load side-channel contract from {spec_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return list(module.DEFAULT_SIDE_CHANNELS)


def _validate_sidecar_prefix(prefix: str) -> dict[str, Any]:
    prefix_path = pathlib.Path(prefix)
    required = {
        "bin": prefix_path.with_suffix(".bin"),
        "idx": prefix_path.with_suffix(".idx"),
        "json": prefix_path.with_suffix(".json"),
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"sidecar dataset prefix {prefix!r} missing {missing}; "
            f"checked {[str(p) for p in required.values()]}"
        )

    meta = json.loads(required["json"].read_text())
    side_paths = meta.get("side_channel_paths")
    if not side_paths or not isinstance(side_paths, dict):
        raise ValueError(f"{required['json']} has no side_channel_paths")

    base_dir = required["json"].parent
    missing_sidecars: list[str] = []
    side_sizes: dict[str, int] = {}
    for name, entry in side_paths.items():
        rel_path = entry.get("path")
        if not rel_path:
            missing_sidecars.append(f"{name}:<missing path>")
            continue
        side_path = base_dir / rel_path
        if not side_path.exists():
            missing_sidecars.append(f"{name}:{side_path}")
        else:
            side_sizes[name] = side_path.stat().st_size
    if missing_sidecars:
        raise FileNotFoundError(
            "sidecar JSON declares missing side-channel files: "
            + ", ".join(missing_sidecars)
        )

    expected_names = {name for name, _dtype in _load_default_side_channels()}
    present_names = set(side_paths)
    missing_expected = sorted(expected_names - present_names)
    if missing_expected:
        raise KeyError(
            "sidecar dataset does not carry the full default side-channel set; "
            f"missing {missing_expected}; present {sorted(present_names)}"
        )

    return {
        "ok": True,
        "prefix": prefix,
        "token_count": int(meta.get("token_count", 0)),
        "vocab_size": int(meta.get("vocab_size", 0)),
        "dtype": meta.get("dtype"),
        "side_channel_count": len(side_paths),
        "side_channels": sorted(side_paths),
        "bin_bytes": required["bin"].stat().st_size,
        "idx_bytes": required["idx"].stat().st_size,
        "side_channel_bytes": side_sizes,
    }


def _candidate_sidecar_prefixes() -> list[str]:
    env_prefixes = [
        item.strip()
        for item in os.environ.get("CPPMEGA_MODAL_DATA_PREFIXES", "").split(",")
        if item.strip()
    ]
    explicit = os.environ.get("CPPMEGA_MODAL_DATA_PREFIX")
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(env_prefixes)
    candidates.extend(
        [
            f"{DATA_MOUNT}/megatron/reindexed_4k_train",
            f"{DATA_MOUNT}/data/megatron/reindexed_4k_train",
        ]
    )
    if PARQUET_DATASET:
        candidates.extend(
            [
                f"{DATA_MOUNT}/megatron/{PARQUET_DATASET}_train",
                f"{DATA_MOUNT}/data/megatron/{PARQUET_DATASET}_train",
            ]
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _find_existing_sidecar_prefix() -> dict[str, Any] | None:
    errors: dict[str, str] = {}
    for prefix in _candidate_sidecar_prefixes():
        try:
            report = _validate_sidecar_prefix(prefix)
            report["source"] = "existing_sidecar_prefix"
            report["checked_candidates"] = _candidate_sidecar_prefixes()
            return report
        except Exception as exc:
            errors[prefix] = f"{type(exc).__name__}: {exc}"
    return {
        "ok": False,
        "source": "existing_sidecar_prefix",
        "checked_candidates": _candidate_sidecar_prefixes(),
        "errors": errors,
    }


def _format_sidecar_smoke_dataset() -> dict[str, Any]:
    if not PARQUET_DATASET:
        raise ValueError(
            "CPPMEGA_MODAL_PARQUET_DATASET is required when "
            "CPPMEGA_MODAL_FORMAT_SIDECAR=1; legacy parquet formatting is opt-in"
        )
    source_dir = pathlib.Path(PARQUET_ROOT) / PARQUET_DATASET
    if not source_dir.is_dir():
        raise FileNotFoundError(f"parquet dataset dir not found: {source_dir}")

    shards = sorted(source_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {source_dir}")
    selected = shards[: max(1, FORMAT_MAX_SHARDS)]

    work_root = pathlib.Path(SIDECAR_OUTPUT_ROOT)
    if work_root.exists():
        shutil.rmtree(work_root)
    input_dir = work_root / "parquet" / PARQUET_DATASET
    input_dir.mkdir(parents=True, exist_ok=True)
    for idx, shard in enumerate(selected):
        (input_dir / f"shard_{idx:05d}.parquet").symlink_to(shard)

    cmd = [
        sys.executable,
        "/opt/cppmega/scripts/data/prepare_format_megacpp.py",
        "--data-root",
        str(work_root),
        "--dataset-name",
        PARQUET_DATASET,
        "--splits",
        "train",
        "--dtype",
        "int32",
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "prepare_format_megacpp.py failed with exit "
            f"{proc.returncode}\n" + "\n".join(proc.stdout.splitlines()[-80:])
        )
    prefix = str(work_root / "megatron" / f"{PARQUET_DATASET}_train")
    report = _validate_sidecar_prefix(prefix)
    report.update(
        {
            "source": "formatted_from_real_parquet",
            "parquet_dataset": str(source_dir),
            "selected_shards": [str(p) for p in selected],
            "formatter_stdout_tail": proc.stdout.splitlines()[-40:],
        }
    )
    return report


def _real_sidecar_cuda_train(prefix: str) -> dict[str, Any]:
    import numpy as np
    import torch

    report = _validate_sidecar_prefix(prefix)
    meta = json.loads(pathlib.Path(prefix + ".json").read_text())
    side_paths = meta["side_channel_paths"]
    token_dtype = _dtype_from_sidecar(str(meta.get("dtype", "int32")))
    token_count = int(meta.get("token_count", 0))
    vocab_size = int(meta.get("vocab_size", 65536)) or 65536

    batch = 4
    seq = 64
    needed = batch * seq + 1
    if token_count < needed:
        raise ValueError(f"sidecar dataset too small for smoke: {token_count} < {needed}")

    tokens = np.memmap(prefix + ".bin", mode="r", dtype=token_dtype, shape=(token_count,))
    token_np = np.asarray(tokens[:needed], dtype=np.int64)

    structure_entry = side_paths["token_structure_ids"]
    structure_path = pathlib.Path(prefix + ".json").parent / structure_entry["path"]
    structure = np.memmap(
        structure_path,
        mode="r",
        dtype=np.dtype(structure_entry.get("dtype", "uint8")),
        shape=(token_count,),
    )
    structure_np = np.asarray(structure[: batch * seq], dtype=np.int64)

    device = torch.device("cuda")
    x = torch.as_tensor(token_np[:-1].reshape(batch, seq) % vocab_size, device=device)
    y = torch.as_tensor(token_np[1:].reshape(batch, seq) % 2048, device=device)
    s = torch.as_tensor(structure_np.reshape(batch, seq).clip(0, 255), device=device)

    model = torch.nn.ModuleDict(
        {
            "tok": torch.nn.Embedding(vocab_size, 128),
            "structure": torch.nn.Embedding(256, 128),
            "out": torch.nn.Linear(128, 2048),
        }
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses: list[float] = []
    for _step in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model["tok"](x) + model["structure"](s)
            logits = model["out"](h)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                y.reshape(-1),
            )
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach().cpu()))

    return {
        "ok": True,
        "prefix": prefix,
        "token_count": report["token_count"],
        "side_channel_count": report["side_channel_count"],
        "losses": losses,
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
    }


def _sidecar_data_smoke(*, run_cuda_train: bool) -> dict[str, Any]:
    if not RUN_SIDECAR_SMOKE:
        return {"ok": False, "skipped": True, "reason": "CPPMEGA_MODAL_SIDECAR_SMOKE=0"}

    existing = _find_existing_sidecar_prefix()
    if existing and existing.get("ok"):
        report = existing
    elif FORMAT_SIDECAR_IF_MISSING:
        report = _format_sidecar_smoke_dataset()
    else:
        return {
            "ok": False,
            "error": "no existing sidecar prefix and CPPMEGA_MODAL_FORMAT_SIDECAR=0",
            "existing_probe": existing,
        }

    train = (
        _real_sidecar_cuda_train(report["prefix"])
        if run_cuda_train
        else {"ok": False, "skipped": True, "reason": "CUDA train disabled"}
    )
    return {
        "ok": bool(report.get("ok")) and (bool(train.get("ok")) if run_cuda_train else True),
        "dataset": report,
        "real_sidecar_cuda_train": train,
    }


def _cppmega_contract_probe() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from cppmega.recipes.nam56r_launch import build_nam56r_lite_main_pattern
        from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan
        from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args

        pattern = build_nam56r_lite_main_pattern(
            pattern="AF",
            depth=2,
            mtp_depths=0,
        )
        plan = build_nam56r_feature_plan(pattern="AF", depth=2, mtp_depths=0)
        args = build_nam56r_megatron_native_args(
            plan=plan,
            enable_mla=False,
            enable_mtp=False,
            enable_moe=False,
        )
        out["ok"] = True
        out["hybrid_layer_pattern"] = pattern
        out["native_args_fragment"] = args.to_shell_fragment()
    except Exception as exc:
        out["ok"] = False
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)
        out["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
    return out


def _runtime_report(*, run_tiny_train: bool) -> dict[str, Any]:
    import torch

    imports = {
        name: _import_report(name)
        for name in (
            "cppmega",
            "cppmega.megatron.nam56r_lite_spec",
            "cppmega.megatron.custom_mamba_model",
            "cppmega.megatron.structure_dataset_patch",
            "megatron",
            "megatron.core",
            "transformer_engine",
            "transformer_engine.pytorch",
            "flash_attn",
            "flash_attn.cute",
            "flash_attn.cute.interface",
            "flash_attn.cute.block_sparsity",
            "flash_attn_3",
            "mamba_ssm",
            "tilelang",
            "cutlass",
            "cutlass.cute",
            "quack",
            "qoptim_cuda",
        )
    }
    return {
        "image_ref": os.environ.get("CPPMEGA_IMAGE_REF", GHCR_REF),
        "ghcr_repo": os.environ.get("CPPMEGA_GHCR_REPO", GHCR_REPO),
        "ghcr_tag": os.environ.get("CPPMEGA_GHCR_TAG", GHCR_TAG),
        "gpu_spec": GPU_SPEC,
        "overlay_repo": os.environ.get("CPPMEGA_OVERLAY_REPO", "1" if OVERLAY_REPO else "0") == "1",
        "install_cutlass_overlay": os.environ.get(
            "CPPMEGA_INSTALL_CUTLASS_OVERLAY",
            "1" if INSTALL_CUTLASS else "0",
        ) == "1",
        "install_flash_attn_4_overlay": os.environ.get(
            "CPPMEGA_INSTALL_FLASH_ATTN_4_OVERLAY",
            "1" if INSTALL_FLASH_ATTN_4 else "0",
        ) == "1",
        "data_volume": DATA_VOLUME_NAME,
        "parquet_dataset": PARQUET_DATASET,
        "sidecar_smoke_enabled": RUN_SIDECAR_SMOKE,
        "local_git_sha": os.environ.get("CPPMEGA_LOCAL_GIT_SHA", LOCAL_GIT_SHA),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        },
        "imports": imports,
        "cppmega_contract": _cppmega_contract_probe(),
        "tiny_torch_cuda_smoke": (
            _tiny_torch_train()
            if RUN_TINY_CUDA_SMOKE and run_tiny_train and torch.cuda.is_available()
            else {"ok": False, "skipped": True, "reason": "CPPMEGA_MODAL_TINY_CUDA_SMOKE is not enabled"}
        ),
        "sidecar_data_smoke": _sidecar_data_smoke(
            run_cuda_train=run_tiny_train and torch.cuda.is_available()
        ),
    }


@app.function(image=_image(), timeout=1800, volumes={DATA_MOUNT: data_vol})
def inspect_cpu() -> dict[str, Any]:
    return _runtime_report(run_tiny_train=False)


@app.function(image=_image(), gpu=GPU_SPEC, timeout=1800, volumes={DATA_MOUNT: data_vol})
def inspect_h100() -> dict[str, Any]:
    return _runtime_report(run_tiny_train=True)


@app.local_entrypoint()
def main() -> None:
    if os.environ.get("CPPMEGA_MODAL_CPU_ONLY") == "1":
        result = inspect_cpu.remote()
    else:
        result = inspect_h100.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
