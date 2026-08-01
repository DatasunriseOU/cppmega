"""Env-gated applier for the Mamba3 stage2 force-nonTMA bwd patch.

This is a production-control wrapper around the benchmarked patch:

    upstream_prs/examples/13_tilelang_floormod_dbz/
        mamba3_bwd_stage2_force_nontma.patch

Default behavior is a no-op. To mutate the installed ``mamba_ssm`` source,
both gates must be set:

    CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1
    MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1

Rollback guard:

    CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK=1
    python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches

The patch keeps both backward kernels fail-closed with TMA lowering and warp
specialization disabled. Exact H200 runtime validation showed that the
flattened bwd_fwd TMA path can compile but raises
``CUDA_ERROR_ILLEGAL_INSTRUCTION``. The structural flattening and targeted
non-TMA copies remain useful, while ``bf_num_stages=1`` and
``bb_num_stages=0`` stay as the existing call defaults.
"""

from __future__ import annotations

import ast
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_ENV_FLAG = "CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA"
_ALLOW_MUTATION_FLAG = "MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION"
_ROLLBACK_FLAG = "CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK"
_LOCK_NAME = "cppmega_mamba3_stage2_force_nontma.lock"
_BACKUP_SUFFIX = ".cppmega_stage2_force_nontma.bak"

_PATCH_REL = Path(
    "upstream_prs/examples/13_tilelang_floormod_dbz/"
    "mamba3_bwd_stage2_force_nontma.patch"
)

_PATCHED_MARKERS = {
    "flat_q": "Q: T.Tensor([B, S * R, G, N], dtype)",
    "flat_qk": "QK_DOT: T.Tensor([B, H, S, R * R], dtype)",
    "bf_default": "bf_num_stages=1",
    "bb_default": "bb_num_stages=0",
    "direct_qk": "qk_dot_shared[cs, r_out * R + r_in]",
    "bf_q_direct_fragment": (
        "T.copy(Q[i_b, fused_chunk_start:fused_chunk_start+fused_chunk_size, "
        "i_h_qk, :], q_frag, disable_tma=True)"
    ),
    "bf_k_direct_fragment": (
        "T.copy(K[i_b, fused_chunk_start:fused_chunk_start+fused_chunk_size, "
        "i_h_qk, :], k_frag, disable_tma=True)"
    ),
    "bf_q_biased_shared": "T.copy(q_frag, q_shared)",
    "bf_k_biased_shared": "T.copy(k_frag, k_shared)",
    "bwd_tma_disabled": "tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True",
    "bwd_ws_disabled": "tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True",
}

_STRUCTURAL_PATCHED_MARKERS = {
    name: marker
    for name, marker in _PATCHED_MARKERS.items()
    if name in ("flat_q", "flat_qk", "direct_qk")
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _patch_path() -> Path:
    path = _repo_root() / _PATCH_REL
    if not path.exists():
        raise RuntimeError(f"stage2 force-nonTMA patch file missing: {path}")
    return path


def _find_mamba3_bwd_file() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("mamba_ssm.ops.tilelang.mamba3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "mamba_ssm.ops.tilelang.mamba3 not importable - is mamba_ssm installed?"
        )
    path = Path(next(iter(spec.submodule_search_locations))) / "mamba3_mimo_bwd.py"
    if not path.exists():
        raise RuntimeError(f"Mamba3 bwd kernel file missing: {path}")
    return path


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + _BACKUP_SUFFIX)


def _is_patched(text: str) -> bool:
    return all(marker in text for marker in _PATCHED_MARKERS.values())


def _has_partial_stage2_markers(text: str) -> bool:
    structural_count = sum(
        marker in text for marker in _STRUCTURAL_PATCHED_MARKERS.values()
    )
    if structural_count == 0:
        return False
    full_count = sum(marker in text for marker in _PATCHED_MARKERS.values())
    return full_count < len(_PATCHED_MARKERS)


def _validate_patched(path: Path) -> None:
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise RuntimeError(f"{path}: patched source is not valid Python") from exc

    def exact_function(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{path}: expected exactly one {name} definition")
        return matches[0]

    bwd_fwd = exact_function("mamba_mimo_bwd_fwd")
    bwd_bwd = exact_function("mamba_mimo_bwd_bwd")
    combined = exact_function("mamba_mimo_bwd_combined")
    positional_args = [*combined.args.posonlyargs, *combined.args.args]
    positional_defaults = [
        *([None] * (len(positional_args) - len(combined.args.defaults))),
        *combined.args.defaults,
    ]
    defaults_by_name = {
        arg.arg: default
        for arg, default in zip(positional_args, positional_defaults, strict=True)
    }
    defaults_by_name.update(
        {
            arg.arg: default
            for arg, default in zip(
                combined.args.kwonlyargs,
                combined.args.kw_defaults,
                strict=True,
            )
        }
    )
    expected_stage_defaults = {"bf_num_stages": 1, "bb_num_stages": 0}
    observed_stage_defaults = {
        name: (
            default.value
            if isinstance(default, ast.Constant) and type(default.value) is int
            else None
        )
        for name, expected in expected_stage_defaults.items()
        if (default := defaults_by_name.get(name)) is not None
    }
    if observed_stage_defaults != expected_stage_defaults:
        raise RuntimeError(
            f"{path}: backward stage defaults must stay exact; "
            f"observed={observed_stage_defaults}, "
            f"expected={expected_stage_defaults}"
        )

    def copy_signature(
        node: ast.AST,
    ) -> tuple[str | None, str | None, bool] | None:
        if (
            not isinstance(node, ast.Call)
            or ast.unparse(node.func) != "T.copy"
            or len(node.args) < 2
        ):
            return None
        source = node.args[0]
        while isinstance(source, (ast.Attribute, ast.Subscript)):
            source = source.value
        destination = node.args[1]
        while isinstance(destination, (ast.Attribute, ast.Subscript)):
            destination = destination.value
        disable_tma = any(
            keyword.arg == "disable_tma"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
        return (
            source.id if isinstance(source, ast.Name) else None,
            destination.id if isinstance(destination, ast.Name) else None,
            disable_tma,
        )

    bwd_fwd_copies = [
        signature
        for node in ast.walk(bwd_fwd)
        if (signature := copy_signature(node)) is not None
    ]
    shared_write_counts = {
        f"{name}_shared": sum(
            source == f"{name}_frag" and destination == f"{name}_shared"
            for source, destination, _ in bwd_fwd_copies
        )
        for name in ("q", "k")
    }
    invalid_shared_write_counts = {
        buffer_name: count
        for buffer_name, count in shared_write_counts.items()
        if count != 1
    }
    if invalid_shared_write_counts:
        raise RuntimeError(
            f"{path}: bwd_fwd must have exactly one biased fragment-to-shared "
            f"write for each Q/K buffer; counts={invalid_shared_write_counts}"
        )
    direct_fragment_loads = {
        f"{name}_frag": [
            disable_tma
            for source, destination, disable_tma in bwd_fwd_copies
            if source == name.upper() and destination == f"{name}_frag"
        ]
        for name in ("q", "k")
    }
    invalid_direct_fragment_loads = {
        fragment_name: values
        for fragment_name, values in direct_fragment_loads.items()
        if values != [True]
    }
    if invalid_direct_fragment_loads:
        raise RuntimeError(
            f"{path}: bwd_fwd must have exactly one direct global-to-fragment "
            f"load with disable_tma=True for each Q/K buffer; "
            f"observed={invalid_direct_fragment_loads}"
        )
    overlapping = [
        f"{name}_shared"
        for name in ("q", "k")
        if any(
            source == name.upper() and destination == f"{name}_shared"
            for source, destination, _ in bwd_fwd_copies
        )
    ]
    if overlapping:
        raise RuntimeError(
            f"{path}: bwd_fwd has overlapping raw and biased shared writes for "
            f"{overlapping}; raw Q/K must load directly into fragments"
        )
    missing = [name for name, marker in _PATCHED_MARKERS.items() if marker not in text]
    if missing:
        raise RuntimeError(
            f"{path}: patched validation failed, missing markers {missing}"
        )

    required_pass_configs = (
        "tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER",
        "tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED",
    )
    invalid_pass_configs: dict[str, dict[str, list[object]]] = {}
    for function in (bwd_fwd, bwd_bwd):
        jit_decorators = [
            decorator
            for decorator in function.decorator_list
            if isinstance(decorator, ast.Call)
            and ast.unparse(decorator.func) == "tilelang.jit"
        ]
        pass_config_nodes = [
            keyword.value
            for decorator in jit_decorators
            for keyword in decorator.keywords
            if keyword.arg == "pass_configs"
        ]
        observed: dict[str, list[object]] = {}
        if len(pass_config_nodes) == 1 and isinstance(pass_config_nodes[0], ast.Dict):
            for key, value in zip(
                pass_config_nodes[0].keys,
                pass_config_nodes[0].values,
                strict=True,
            ):
                if key is None:
                    continue
                key_name = ast.unparse(key)
                if key_name in required_pass_configs:
                    observed.setdefault(key_name, []).append(
                        (
                            value.value
                            if isinstance(value, ast.Constant)
                            and type(value.value) is bool
                            else None
                        )
                    )
        invalid = {
            key: observed.get(key, [])
            for key in required_pass_configs
            if observed.get(key) != [True]
        }
        if len(jit_decorators) != 1 or invalid:
            invalid_pass_configs[function.name] = invalid
    if invalid_pass_configs:
        raise RuntimeError(
            f"{path}: both backward kernels must keep exact TMA lowering and "
            f"warp specialization disable pass configs; "
            f"observed={invalid_pass_configs}"
        )

    disable_tma_count = sum(
        bool(signature[2])
        for node in ast.walk(tree)
        if (signature := copy_signature(node)) is not None
    )
    if disable_tma_count < 10:
        raise RuntimeError(
            f"{path}: expected targeted per-copy disable_tma guards, "
            f"found only {disable_tma_count}"
        )


def _atomic_replace_from(src: Path, dst: Path) -> None:
    import py_compile

    py_compile.compile(str(src), doraise=True)
    tmp = dst.with_name(f"{dst.name}.cppmega_stage2.tmp.{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _apply_patch_to_temp(src: Path, patch_file: Path) -> Path:
    work = Path(tempfile.mkdtemp(prefix="cppmega_mamba3_stage2_force_nontma_"))
    dst = work / src.name
    shutil.copy2(src, dst)
    patch_bytes = patch_file.read_bytes()
    proc = subprocess.run(
        ["patch", "--ignore-whitespace", "-p4", str(dst)],
        input=patch_bytes,
        capture_output=True,
        cwd=work,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "stage2 force-nonTMA patch failed\n"
            f"stdout:\n{proc.stdout.decode(errors='replace')[-4000:]}\n"
            f"stderr:\n{proc.stderr.decode(errors='replace')[-4000:]}"
        )
    _validate_patched(dst)
    return dst


def _do_patch() -> None:
    path = _find_mamba3_bwd_file()
    text = path.read_text()
    print(f"Mamba3 bwd kernel path: {path}")

    if _is_patched(text):
        _validate_patched(path)
        print("  OK   stage2 force-nonTMA patch already applied")
        return
    if _has_partial_stage2_markers(text):
        raise RuntimeError(
            f"{path}: partial stage2 force-nonTMA markers detected. "
            f"Set {_ROLLBACK_FLAG}=1 to rollback from backup/reverse patch, "
            "or reinstall mamba_ssm before retrying."
        )

    backup = _backup_path(path)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  DONE backup written: {backup}")
    else:
        print(f"  OK   backup already exists: {backup}")

    patched = _apply_patch_to_temp(path, _patch_path())
    _atomic_replace_from(patched, path)
    _validate_patched(path)
    print("  DONE stage2 force-nonTMA patch applied")
    print(
        "  Active default: bf_num_stages=1, bb_num_stages=0, "
        "bwd TMA/warp specialization disabled"
    )


def _reverse_patch(path: Path, patch_file: Path) -> bool:
    work = Path(tempfile.mkdtemp(prefix="cppmega_mamba3_stage2_force_nontma_rollback_"))
    dst = work / path.name
    shutil.copy2(path, dst)
    proc = subprocess.run(
        ["patch", "--ignore-whitespace", "-R", "-p4", str(dst)],
        input=patch_file.read_bytes(),
        capture_output=True,
        cwd=work,
        check=False,
    )
    if proc.returncode != 0:
        return False
    _atomic_replace_from(dst, path)
    return True


def rollback() -> None:
    """Restore the pre-patch file from backup, falling back to reverse patch."""
    path = _find_mamba3_bwd_file()
    backup = _backup_path(path)
    print(f"Mamba3 bwd kernel path: {path}")

    if backup.exists():
        _atomic_replace_from(backup, path)
        print(f"  DONE restored backup: {backup}")
        return

    if _is_patched(path.read_text()) and _reverse_patch(path, _patch_path()):
        print("  DONE reverted stage2 force-nonTMA patch with patch -R")
        return

    raise RuntimeError(
        f"No backup found at {backup} and reverse patch failed. "
        "Reinstall mamba_ssm to restore the upstream kernel file."
    )


def _is_stage2_patch_applied() -> bool:
    try:
        path = _find_mamba3_bwd_file()
        if not _is_patched(path.read_text()):
            return False
        _validate_patched(path)
        return True
    except Exception:
        log.debug("stage2 force-nonTMA patch detection failed", exc_info=True)
        return False


def _is_stage2_patch_absent() -> bool:
    try:
        text = _find_mamba3_bwd_file().read_text()
        return not _is_patched(text) and not _has_partial_stage2_markers(text)
    except Exception:
        log.debug("stage2 force-nonTMA patch-absence detection failed", exc_info=True)
        return False


def _run_once_with_local_rank_guard(fn, is_done=None) -> None:
    try:
        import torch.distributed as dist
    except Exception:
        log.debug(
            "torch.distributed unavailable; falling back to file-lock guard",
            exc_info=True,
        )
        dist = None  # type: ignore[assignment]

    if dist is not None and dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            fn()
        else:
            print(f"[mamba3_stage2_force_nontma] rank={rank} waiting on rank-0")
        dist.barrier()
        return

    import fcntl
    import time

    lock_path = Path(tempfile.gettempdir()) / _LOCK_NAME
    local_rank = int(os.environ.get("LOCAL_RANK") or "0")
    rank_env = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "?"
    sentinel = "DONE\n"

    if local_rank == 0:
        lock_path.unlink(missing_ok=True)
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                print(
                    "[mamba3_stage2_force_nontma] "
                    f"local_rank=0 rank={rank_env} mutating file"
                )
                fn()
                lock_fh.write(sentinel)
                lock_fh.flush()
                os.fsync(lock_fh.fileno())
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return

    print(
        "[mamba3_stage2_force_nontma] "
        f"local_rank={local_rank} rank={rank_env} waiting for local_rank=0"
    )
    deadline = time.time() + 120.0
    while time.time() < deadline:
        if lock_path.exists():
            with open(lock_path) as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_SH)
                try:
                    if sentinel in lock_fh.read() and (is_done is None or is_done()):
                        return
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        time.sleep(0.1)
    raise RuntimeError(
        f"Timed out waiting for local_rank=0 stage2 patch. lockfile={lock_path}"
    )


def apply_all() -> None:
    """Apply the stage2 patch if explicit mutation gates are set."""
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        _run_once_with_local_rank_guard(rollback, _is_stage2_patch_absent)
        return
    if os.environ.get(_ENV_FLAG, "0") not in ("1", "true", "True"):
        print(f"  SKIP {_ENV_FLAG} is not set")
        return
    if os.environ.get(_ALLOW_MUTATION_FLAG, "0") not in ("1", "true", "True"):
        raise RuntimeError(
            f"Refusing to mutate installed mamba_ssm without {_ALLOW_MUTATION_FLAG}=1"
        )
    _run_once_with_local_rank_guard(_do_patch, _is_stage2_patch_applied)


def apply_if_requested() -> bool:
    """Env-gated entry point for a future shim."""
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        apply_all()
        return True
    if os.environ.get(_ENV_FLAG, "0") not in ("1", "true", "True"):
        log.debug("stage2 force-nonTMA patch not requested: %s is not set", _ENV_FLAG)
        return False
    apply_all()
    return True


if __name__ == "__main__":
    try:
        apply_all()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
