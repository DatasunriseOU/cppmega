"""Env-gated Mamba3 bwd_bwd vectorized diagonal-consumer experiment.

Default behavior is a no-op. The candidate first applies the guarded stage2
force-nonTMA source layout patch, then changes the bwd_bwd qk-dot diagonal
consumer path:

* compute same-step ``R x R`` blocks with a per-step vectorized
  ``R*R x P`` product plus ``T.reduce_sum`` microkernel;
* stage those diagonal blocks in ``[chunk, R * R]`` shared;
* feed ``DGAMMA_DIAG``, ``DK`` and ``DQ`` from that compact shared tile.

This is deliberately different from the Wave31 scalar shared-diag candidate:
it keeps the ``P`` dimension vectorized for the diagonal path instead of
replacing the product with scalar loops over ``P``. Direct tiny ``T.gemm``
forms were rejected by TileLang MMA layout inference on H100.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from cppmega.megatron.upstream_patches import (
    apply_mamba3_stage2_force_nontma_patches as stage2,
)

_ENV_FLAG = "CPPMEGA_MAMBA3_BWD_BWD_VECTORIZED_DIAG"
_ALLOW_MUTATION_FLAG = "MAMBA3_BWD_BWD_VECTORIZED_DIAG_ALLOW_FILE_MUTATION"
_ROLLBACK_FLAG = "CPPMEGA_MAMBA3_BWD_BWD_VECTORIZED_DIAG_ROLLBACK"
_BACKUP_SUFFIX = ".cppmega_bwd_bwd_vectorized_diag.bak"

_ALLOC_ORIGINAL = (
    "            dqk_from_diag_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], accum_dtype)\n"
)
_ALLOC_PATCHED = (
    "            # Wave32: keep full vectorized GEMM, but only stage same-step\n"
    "            # R x R diagonal blocks consumed by DGAMMA/DK/DQ downstream.\n"
    "            dqk_diag_shared = T.alloc_shared([chunk_size, R * R], accum_dtype)\n"
)

_LAYOUT_ORIGINAL = (
    "                    dqk_from_diag_shared: tilelang.layout.make_swizzled_layout(dqk_from_diag_shared),\n"
)
_LAYOUT_PATCHED = (
    "                    dqk_diag_shared: tilelang.layout.make_swizzled_layout(dqk_diag_shared),\n"
)

_GEMM_ORIGINAL = (
    "                # Compute dqk_from_diag, which is the contribution to dQ/dK from qk_dot:\n"
    "                dqk_from_diag_frag = T.alloc_fragment([fused_chunk_size, fused_chunk_size], accum_dtype)\n"
    "                T.gemm(dPhiO_shared, PsiV_shared, dqk_from_diag_frag, transpose_B=True, clear_accum=True) # (cs*r_out, cs*r_in)\n"
    "                # Compute dgamma_diag.\n"
    "                # TMA-fix: dgamma_diag_prereduce_frag flattened to [chunk_size, R*R] to\n"
    "                # match the 2D qk_dot_shared. The reduce_sum below no longer needs a view.\n"
    "                dgamma_diag_prereduce_frag = T.alloc_fragment([chunk_size, R * R], accum_dtype)\n"
    "                T.copy(qk_dot_shared, dgamma_diag_prereduce_frag)\n"
    "                T.copy(dqk_from_diag_frag, dqk_from_diag_shared)\n"
    "                for cs, r_out, r_in in T.Parallel(chunk_size, R, R):\n"
    "                    dgamma_diag_prereduce_frag[cs, r_out * R + r_in] *= dqk_from_diag_shared[cs*R + r_out, cs*R + r_in]\n"
)
_GEMM_PATCHED = (
    "                # Compute dqk_from_diag, which is the contribution to dQ/dK from qk_dot:\n"
    "                # Wave32: vectorized per-step R*R x P reduction microkernel. This avoids\n"
    "                # the full fused_chunk_size x fused_chunk_size shared tile without the\n"
    "                # Wave31 scalar loop over P for every (cs, r_out, r_in).\n"
    "                dqk_step_prereduce_frag = T.alloc_fragment([R * R, P], accum_dtype)\n"
    "                dqk_step_frag = T.alloc_fragment([R * R], accum_dtype)\n"
    "                for cs in T.serial(chunk_size):\n"
    "                    for rr, p in T.Parallel(R * R, P):\n"
    "                        r_out = rr // R\n"
    "                        r_in = rr % R\n"
    "                        dqk_step_prereduce_frag[rr, p] = dPhiO_shared[cs * R + r_out, p] * PsiV_shared[cs * R + r_in, p]\n"
    "                    T.reduce_sum(dqk_step_prereduce_frag, dqk_step_frag, dim=-1, clear=True)\n"
    "                    for rr in T.Parallel(R * R):\n"
    "                        dqk_diag_shared[cs, rr] = dqk_step_frag[rr]\n"
    "                # Compute dgamma_diag.\n"
    "                # TMA-fix: dgamma_diag_prereduce_frag flattened to [chunk_size, R*R] to\n"
    "                # match the 2D qk_dot_shared. The reduce_sum below no longer needs a view.\n"
    "                dgamma_diag_prereduce_frag = T.alloc_fragment([chunk_size, R * R], accum_dtype)\n"
    "                T.copy(qk_dot_shared, dgamma_diag_prereduce_frag)\n"
    "                for cs, r_out, r_in in T.Parallel(chunk_size, R, R):\n"
    "                    dgamma_diag_prereduce_frag[cs, r_out * R + r_in] *= dqk_diag_shared[cs, r_out * R + r_in]\n"
)

_GAMMA_SCALE_ORIGINAL = (
    "                for csr_i, csr_j in T.Parallel(fused_chunk_size, fused_chunk_size):\n"
    "                    dqk_from_diag_frag[csr_i, csr_j] *= gamma_qk_frag[csr_i//R]\n"
    "                T.copy(dqk_from_diag_frag, dqk_from_diag_shared)\n"
)
_GAMMA_SCALE_PATCHED = (
    "                for cs, r_out, r_in in T.Parallel(chunk_size, R, R):\n"
    "                    dqk_diag_shared[cs, r_out * R + r_in] *= gamma_qk_frag[cs]\n"
)

_DK_ORIGINAL = (
    "                        dk_combined_frag[csr_in, n] += dqk_from_diag_shared[csr_out, csr_in] * q_dk_frag_reshaped[cs, r_out, n]  \n"
)
_DK_PATCHED = (
    "                        dk_combined_frag[csr_in, n] += dqk_diag_shared[cs, r_out * R + (csr_in % R)] * q_dk_frag_reshaped[cs, r_out, n]  \n"
)

_DQ_ORIGINAL = (
    "                        dq_frag[csr_out, n] += dqk_from_diag_shared[csr_out, csr_in] * k_pre_rot_shared[csr_in, n]\n"
)
_DQ_PATCHED = (
    "                        dq_frag[csr_out, n] += dqk_diag_shared[cs, (csr_out % R) * R + r_in] * k_pre_rot_shared[csr_in, n]\n"
)

_PATCHED_MARKERS = {
    "diag_shared": "dqk_diag_shared = T.alloc_shared([chunk_size, R * R], accum_dtype)",
    "vectorized_gemm_marker": "Wave32: vectorized per-step R*R x P reduction microkernel",
    "dk_consumer": "dqk_diag_shared[cs, r_out * R + (csr_in % R)]",
    "dq_consumer": "dqk_diag_shared[cs, (csr_out % R) * R + r_in]",
}

_REMOVED_MARKERS = (
    "dqk_from_diag_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], accum_dtype)",
    "dqk_from_diag_shared: tilelang.layout.make_swizzled_layout(dqk_from_diag_shared)",
    "T.copy(dqk_from_diag_frag, dqk_from_diag_shared)",
)


def _find_mamba3_bwd_file() -> Path:
    return stage2._find_mamba3_bwd_file()


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + _BACKUP_SUFFIX)


def _flag_enabled() -> bool:
    value = os.environ.get(_ENV_FLAG, "0")
    return value in ("1", "true", "True", "yes", "on", "vectorized_diag")


def _is_patched_text(text: str) -> bool:
    return all(marker in text for marker in _PATCHED_MARKERS.values()) and not any(
        marker in text for marker in _REMOVED_MARKERS
    )


def _has_partial_markers(text: str) -> bool:
    present = sum(marker in text for marker in _PATCHED_MARKERS.values())
    if 0 < present < len(_PATCHED_MARKERS):
        return True
    removed_present = sum(marker in text for marker in _REMOVED_MARKERS)
    return present > 0 and removed_present > 0


def _replace_once(text: str, original: str, patched: str, label: str) -> str:
    count = text.count(original)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one {label} source block in stage2 Mamba3 bwd, found {count}"
        )
    return text.replace(original, patched, 1)


def _patch_stage2_text(text: str) -> tuple[str, bool]:
    if _is_patched_text(text):
        return text, False
    if _has_partial_markers(text):
        raise RuntimeError("partial Wave32 vectorized diag markers detected")
    if not stage2._is_patched(text):
        raise RuntimeError(
            "Wave32 vectorized diag patch expects the stage2 force-nonTMA layout patch first"
        )

    new_text = text
    replacements = (
        (_ALLOC_ORIGINAL, _ALLOC_PATCHED, "dqk shared allocation"),
        (_LAYOUT_ORIGINAL, _LAYOUT_PATCHED, "dqk shared layout"),
        (_GEMM_ORIGINAL, _GEMM_PATCHED, "dqk GEMM diagonal extraction"),
        (_GAMMA_SCALE_ORIGINAL, _GAMMA_SCALE_PATCHED, "dqk gamma scaling"),
        (_DK_ORIGINAL, _DK_PATCHED, "DK diagonal consumer"),
        (_DQ_ORIGINAL, _DQ_PATCHED, "DQ diagonal consumer"),
    )
    for original, patched, label in replacements:
        new_text = _replace_once(new_text, original, patched, label)
    return new_text, True


def _validate_patched_text(text: str) -> None:
    if not stage2._is_patched(text):
        raise RuntimeError("stage2 validation failed after Wave32 vectorized diag patch")
    missing = [name for name, marker in _PATCHED_MARKERS.items() if marker not in text]
    if missing:
        raise RuntimeError(f"Wave32 vectorized diag validation failed, missing {missing}")
    removed = [marker for marker in _REMOVED_MARKERS if marker in text]
    if removed:
        raise RuntimeError(f"Wave32 vectorized diag validation failed, stale markers {removed}")


def _stage2_base_for(src: Path) -> Path:
    text = src.read_text()
    if stage2._is_patched(text):
        stage2._validate_patched(src)
        work = Path(tempfile.mkdtemp(prefix="cppmega_mamba3_bwd_bwd_vectorized_stage2_"))
        dst = work / src.name
        shutil.copy2(src, dst)
        return dst
    if stage2._has_partial_stage2_markers(text):
        raise RuntimeError(
            f"{src}: partial stage2 force-nonTMA markers detected; rollback or "
            "reinstall mamba_ssm before applying Wave32 vectorized diag patch"
        )
    return stage2._apply_patch_to_temp(src, stage2._patch_path())


def _atomic_write_text(path: Path, text: str) -> None:
    import py_compile

    tmp = path.with_name(f"{path.name}.cppmega_bwd_bwd_vectorized_diag.tmp.{os.getpid()}")
    tmp.write_text(text)
    try:
        py_compile.compile(str(tmp), doraise=True)
    except py_compile.PyCompileError:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _do_patch() -> None:
    path = _find_mamba3_bwd_file()
    text = path.read_text()
    print(f"Mamba3 bwd kernel path: {path}")

    if _is_patched_text(text):
        _validate_patched_text(text)
        print("  OK   bwd_bwd vectorized diag patch already applied")
        return
    if _has_partial_markers(text):
        raise RuntimeError(
            f"{path}: partial bwd_bwd vectorized diag markers detected. Set "
            f"{_ROLLBACK_FLAG}=1 to rollback from backup, or reinstall mamba_ssm before retrying."
        )

    backup = _backup_path(path)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  DONE backup written: {backup}")
    else:
        print(f"  OK   backup already exists: {backup}")

    stage2_base = _stage2_base_for(path)
    patched_text, changed = _patch_stage2_text(stage2_base.read_text())
    if not changed:
        _validate_patched_text(patched_text)
        print("  OK   bwd_bwd vectorized diag patch already applied in stage2 base")
        return
    _validate_patched_text(patched_text)
    _atomic_write_text(path, patched_text)
    _validate_patched_text(path.read_text())
    print("  DONE bwd_bwd vectorized diag patch applied")


def rollback() -> None:
    path = _find_mamba3_bwd_file()
    backup = _backup_path(path)
    print(f"Mamba3 bwd kernel path: {path}")
    if backup.exists():
        stage2._atomic_replace_from(backup, path)
        print(f"  DONE restored backup: {backup}")
        return
    raise RuntimeError(
        f"No backup found at {backup}. Reinstall mamba_ssm to restore the kernel file."
    )


def _is_vectorized_diag_patch_applied() -> bool:
    try:
        text = _find_mamba3_bwd_file().read_text()
        if not _is_patched_text(text):
            return False
        _validate_patched_text(text)
        return True
    except Exception:
        return False


def _is_vectorized_diag_patch_absent() -> bool:
    try:
        text = _find_mamba3_bwd_file().read_text()
        return not _is_patched_text(text) and not _has_partial_markers(text)
    except Exception:
        return False


def apply_all() -> None:
    """Apply the Wave32 vectorized diag patch if explicit mutation gates are set."""
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        stage2._run_once_with_local_rank_guard(rollback, _is_vectorized_diag_patch_absent)
        return
    if not _flag_enabled():
        print(f"  SKIP {_ENV_FLAG} is not set to vectorized_diag")
        return
    if os.environ.get(_ALLOW_MUTATION_FLAG, "0") not in ("1", "true", "True"):
        raise RuntimeError(
            f"Refusing to mutate installed mamba_ssm without {_ALLOW_MUTATION_FLAG}=1"
        )
    stage2._run_once_with_local_rank_guard(_do_patch, _is_vectorized_diag_patch_applied)


def apply_if_requested() -> bool:
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        apply_all()
        return True
    if not _flag_enabled():
        return False
    apply_all()
    return True


if __name__ == "__main__":
    try:
        apply_all()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
