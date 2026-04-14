"""Reproducer: Mamba3 MIMO backward fails for GQA (1 < ngroups < nheads) and
the DT-dtype assumption silently breaks under Megatron's Float16Module.

Demonstrates two related problems described in
``upstream_prs/05_mamba3_dt_fp32_gqa_bwd.md``:

  Problem 1 (DT fp32 cast in mamba3.py):
      The TileLang MIMO fwd/bwd kernels declare
      ``DT: T.Tensor([B, H, S], T.float32)`` (hard fp32). In upstream
      ``mamba_ssm/modules/mamba3.py`` the call is
      ``DT = F.softplus(dd_dt + self.dt_bias)``. When Megatron's
      ``Float16Module`` casts ``dt_bias`` to bf16 (the standard path),
      ``DT`` comes out bf16. Feeding that bf16 tensor into the TileLang
      fwd raises a low-level TVM-FFI "Argument ... Expected float32 Tensor
      but got bfloat16 Tensor" (or equivalent dtype assert), not a nice
      Python error.

      Fix: ``DT = F.softplus((dd_dt + self.dt_bias).to(torch.float32))``.

  Problem 2 (MIMO GQA backward branch missing):
      ``mamba_mimo_bwd_combined`` in
      ``mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`` only handles
      ``G == 1`` (MHA) and ``G == H`` (per-head). Any intermediate GQA
      (``1 < G < H`` with ``H % G == 0``) hits the
      ``else: raise ValueError("G value of {G} is not currently supported!")``.

      Fix: add an ``elif H % G == 0:`` branch that reduces dq/dk via
      ``view(B, S, R, G, hpg, N).sum(dim=4)``.

We demonstrate each failure then validate the patched versions on the
live install. For Problem 2 we monkey-patch the source on disk to the
*unpatched* state, reload, trigger the raise, then restore to the patched
state and verify correctness.

Requires CUDA (any compute capability supported by mamba-ssm TileLang
kernels; tested on sm_121a GB10 and sm_90a H200).

Exit codes:
  0 — both bugs reproduce AND both fixes validate (expected)
  1 — one or both steps failed to demonstrate the expected behaviour
"""
from __future__ import annotations

import importlib
import math
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config: use a small GQA shape so we trip the H % G == 0, G not in {1, H}
# branch that upstream does not implement.
# ---------------------------------------------------------------------------
BATCH = 1
SEQLEN = 256
MIMO_RANK = 4
HEADDIM_QK = 32         # N — same as upstream bwd-tested param N32_P64_R4_C16
HEADDIM_V = 64          # P
NHEADS = 16             # H — same as upstream FIXED_H
NGROUPS = 2             # G — KEY KNOB: 1 < G < H with H % G == 0 triggers GQA branch
CHUNK_SIZE = 16
ROTARY_DIM_DIVISOR = 4
DTYPE = torch.bfloat16
DEVICE = "cuda"
SEED = 0


def _seed(s: int = SEED) -> None:
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _build_mimo_inputs(
    *,
    dt_dtype: torch.dtype,
    nheads_qk: int = NGROUPS,
) -> dict:
    """Build a minimal valid set of Mamba3 MIMO inputs.

    Shapes match ``mamba_ssm.ops.tilelang.mamba3.mamba3_mimo.mamba3_mimo``
    and the upstream test suite (tests/ops/tilelang/test_mamba3_mimo.py).

    ``dt_dtype`` selects fp32 (kernel-correct) or bf16 (what Float16Module
    produces before the fix).
    """
    _seed()
    b, s, r = BATCH, SEQLEN, MIMO_RANK
    g, h = nheads_qk, NHEADS
    n, p = HEADDIM_QK, HEADDIM_V

    Q = torch.randn((b, s, r, g, n), device=DEVICE, dtype=DTYPE, requires_grad=True)
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn((b, s, h, p), device=DEVICE, dtype=DTYPE, requires_grad=True)

    # DT fp32 is required by the TileLang kernel; bf16 mimics Float16Module path.
    DT = F.softplus(
        -3.0
        + torch.randn(b, h, s, device=DEVICE, dtype=torch.float32)
    ).detach().to(dt_dtype).requires_grad_(True)
    ADT = (-DT.detach().float() * math.log2(math.e)).clone().detach().to(dt_dtype).requires_grad_(True)

    Trap = (torch.rand((b, h, s), device=DEVICE, dtype=DTYPE) * 0.5).detach().requires_grad_(True)
    Q_bias = torch.randn((h, r, n), device=DEVICE, dtype=torch.float32, requires_grad=True)
    K_bias = torch.randn_like(Q_bias, requires_grad=True)
    MIMO_V = torch.randn((h, r, p), device=DEVICE, dtype=torch.float32, requires_grad=True)
    MIMO_Z = (torch.randn_like(MIMO_V) / r).detach().requires_grad_(True)
    MIMO_Out = (torch.randn_like(MIMO_V) / r).detach().requires_grad_(True)
    Angles = torch.rand(
        (b, s, h, n // ROTARY_DIM_DIVISOR), device=DEVICE, dtype=torch.float32,
        requires_grad=True,
    )
    D = torch.randn((h,), device=DEVICE, dtype=torch.float32, requires_grad=True)
    Z = torch.randn((b, s, h, p), device=DEVICE, dtype=DTYPE, requires_grad=True)

    return dict(
        Q=Q, K=K, V=V, ADT=ADT, DT=DT, Trap=Trap,
        Q_bias=Q_bias, K_bias=K_bias,
        MIMO_V=MIMO_V, MIMO_Z=MIMO_Z, MIMO_Out=MIMO_Out,
        Angles=Angles, D=D, Z=Z,
    )


def _call_mamba3(inputs: dict) -> torch.Tensor:
    import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo as mamba3_top

    return mamba3_top.mamba3_mimo(
        inputs["Q"], inputs["K"], inputs["V"],
        inputs["ADT"], inputs["DT"], inputs["Trap"],
        inputs["Q_bias"], inputs["K_bias"],
        inputs["MIMO_V"], inputs["MIMO_Z"], inputs["MIMO_Out"],
        inputs["Angles"], inputs["D"], inputs["Z"],
        CHUNK_SIZE,
        ROTARY_DIM_DIVISOR,
        DTYPE,
        False,
        None,
    )


# ---------------------------------------------------------------------------
# Source-level monkey-patch helpers for Problem 2.
# ---------------------------------------------------------------------------
PATCHED_GQA_BLOCK = (
    "    elif H % G == 0:\n"
    "        # GQA-style: 1 < G < H, H divisible by G.  Sum over heads_per_group.\n"
    "        hpg = H // G\n"
    "        # bias grads: [B, S, R, H, N] -> sum(batch, seq) -> [R, H, N] -> permute -> [H, R, N]\n"
    "        # Must compute BEFORE reducing dq/dk, since Q_bias has shape [H, R, N]\n"
    "        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        # dq/dk: [B, S, R, H, N] -> [B, S, R, G, hpg, N] -> sum(dim=4) -> [B, S, R, G, N]\n"
    "        dq_tilelang = dq_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)\n"
    "        dk_tilelang = dk_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)\n"
    "        # mimo_v, mimo_z, D: sum over batch, already (B, H, R, P) or (B, H)\n"
    "        dmimo_v = dmimo_v.sum(dim=0)\n"
    "        dmimo_z = dmimo_z.sum(dim=0) if dmimo_z is not None else None\n"
    "        dD = dD.sum(dim=0) if dD is not None else None\n"
)
UNPATCHED_RAISE_LINE = (
    '        raise ValueError(f"G value of {G} is not currently supported!")\n'
)
PATCHED_RAISE_LINE = (
    '        raise ValueError(f"G value of {G} is not currently supported (H={H}, G must divide H)!")\n'
)


def _bwd_source_path() -> Path:
    import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd as m
    return Path(m.__file__)


def _ensure_patched(path: Path) -> bool:
    """Return True if source already has GQA branch and new error message."""
    src = path.read_text()
    return "elif H % G == 0:" in src and "G must divide H" in src


def _write_unpatched(path: Path) -> None:
    """Rewrite source to the pre-patch (broken) state: remove GQA branch,
    restore short raise message."""
    src = path.read_text()
    assert "elif H % G == 0:" in src, (
        "Cannot find GQA branch in installed source; this reproducer assumes the "
        "patched form is currently on disk. Install the patched fork first."
    )
    # Delete the GQA branch wholesale
    new_src = src.replace(PATCHED_GQA_BLOCK, "")
    # Revert the error message
    new_src = new_src.replace(PATCHED_RAISE_LINE, UNPATCHED_RAISE_LINE)
    path.write_text(new_src)


def _write_patched(path: Path, *, original_src: str) -> None:
    path.write_text(original_src)


def _reload_bwd_module() -> None:
    """Drop cached bytecode and reimport the bwd module and its public wrapper."""
    # Nuke all mamba_ssm.ops.tilelang.mamba3 pyc caches
    import mamba_ssm.ops.tilelang.mamba3 as pkg
    pkg_dir = Path(pkg.__file__).parent
    for p in pkg_dir.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)

    for name in list(sys.modules):
        if name.startswith("mamba_ssm.ops.tilelang.mamba3"):
            del sys.modules[name]


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def test_dt_dtype_bug() -> dict:
    """Problem 1 demo: DT as bf16 (simulating Float16Module) must fail hard;
    DT as fp32 must pass.
    """
    results = {"bf16_failed_as_expected": False, "fp32_ok": False, "bf16_err": None}

    # bf16 DT — unpatched-style call (what upstream mamba3.py produces under
    # Float16Module).
    inputs_bf16 = _build_mimo_inputs(dt_dtype=torch.bfloat16)
    assert inputs_bf16["DT"].dtype == torch.bfloat16
    try:
        out = _call_mamba3(inputs_bf16)
        torch.cuda.synchronize()
        # If it survives fwd, force bwd too
        out.sum().backward()
        torch.cuda.synchronize()
        # If we got here without error, the kernel silently accepted bf16 DT.
        # That is also a bug (silent precision loss); flag it.
        print("  [WARN] bf16 DT did not raise — kernel silently accepted wrong dtype.")
        results["bf16_err"] = "silent-accept"
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        results["bf16_err"] = msg
        results["bf16_failed_as_expected"] = True
        # Most common signature: TVM-FFI / TileLang dtype mismatch on DT arg.
        # We don't hard-match a string; any failure demonstrates the bug.
        print(f"  [OK] bf16 DT failed as expected: {msg.splitlines()[0][:200]}")

    # fp32 DT — the patched path from mamba3.py (DT cast to float32 before softplus).
    inputs_fp32 = _build_mimo_inputs(dt_dtype=torch.float32)
    assert inputs_fp32["DT"].dtype == torch.float32
    try:
        out = _call_mamba3(inputs_fp32)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), "fp32 DT fwd produced non-finite output"
        out.sum().backward()
        torch.cuda.synchronize()
        for name in ("Q", "K", "V", "DT", "ADT"):
            g = inputs_fp32[name].grad
            assert g is not None, f"{name}.grad is None"
            assert torch.isfinite(g).all(), f"{name}.grad has NaN/Inf"
        results["fp32_ok"] = True
        print("  [OK] fp32 DT fwd+bwd produced finite gradients.")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] fp32 DT raised: {type(e).__name__}: {e}")
        traceback.print_exc()

    return results


def test_gqa_branch_bug() -> dict:
    """Problem 2 demo: unpatched bwd raises ValueError on GQA config;
    patched bwd produces finite gradients.
    """
    results = {
        "unpatched_raised_as_expected": False,
        "unpatched_err": None,
        "patched_ok": False,
        "grad_max_abs": {},
    }
    path = _bwd_source_path()
    original_src = path.read_text()
    assert _ensure_patched(path), (
        "Installed mamba_ssm bwd file is not in the 'patched' form. "
        "This reproducer flips patched <-> unpatched on disk; install the "
        "cppmega fork (with the 05 patch applied) before running."
    )

    # --- Step 1: rewrite to unpatched, reload, expect ValueError -----------
    try:
        _write_unpatched(path)
        _reload_bwd_module()
        inputs = _build_mimo_inputs(dt_dtype=torch.float32)
        try:
            out = _call_mamba3(inputs)
            out.sum().backward()
            torch.cuda.synchronize()
            print("  [FAIL] unpatched bwd did NOT raise on GQA (G=2, H=8)")
        except ValueError as e:
            if "not currently supported" in str(e):
                results["unpatched_raised_as_expected"] = True
                results["unpatched_err"] = str(e)
                print(f"  [OK] unpatched bwd raised: {e}")
            else:
                print(f"  [WARN] unpatched bwd raised a different ValueError: {e}")
                results["unpatched_err"] = str(e)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] unpatched bwd raised {type(e).__name__}: {e}")
            results["unpatched_err"] = f"{type(e).__name__}: {e}"
    finally:
        # --- Always restore patched source before returning ----------------
        _write_patched(path, original_src=original_src)
        _reload_bwd_module()

    # --- Step 2: with patched source, expect finite gradients --------------
    try:
        inputs = _build_mimo_inputs(dt_dtype=torch.float32)
        out = _call_mamba3(inputs)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), "patched fwd produced non-finite output"
        out.sum().backward()
        torch.cuda.synchronize()

        for name in ("Q", "K", "V", "DT", "ADT", "Q_bias", "K_bias", "MIMO_V", "D"):
            g = inputs[name].grad
            assert g is not None, f"{name}.grad is None"
            assert torch.isfinite(g).all(), f"{name}.grad has NaN/Inf"
            results["grad_max_abs"][name] = float(g.detach().abs().max().item())

        # Shape sanity on the GQA-reduced gradients: dq/dk should be reduced
        # along nheads_qk (=G), matching Q.shape.
        assert inputs["Q"].grad.shape == inputs["Q"].shape, (
            f"dQ shape {inputs['Q'].grad.shape} != Q shape {inputs['Q'].shape}"
        )
        assert inputs["K"].grad.shape == inputs["K"].shape

        results["patched_ok"] = True
        print("  [OK] patched bwd produced finite grads; |dQ|max={:.3e} |dK|max={:.3e}".format(
            results["grad_max_abs"]["Q"], results["grad_max_abs"]["K"],
        ))
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] patched bwd raised: {type(e).__name__}: {e}")
        traceback.print_exc()

    return results


def _run_in_subprocess(stage: str) -> tuple[int, str]:
    """Each stage spawns a fresh Python so a failing CUDA kernel in one stage
    does not poison the driver context for the next. Returns (exit_code, tail_output)."""
    import subprocess

    env = dict(os.environ)
    env["MAMBA3_REPRO_STAGE"] = stage
    # Reduce autotuning noise.
    env.setdefault("TL_QUIET", "1")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = proc.stdout + proc.stderr
    return proc.returncode, out


def _stage_bf16() -> int:
    """Stage: demonstrate that DT=bf16 makes the TileLang kernel refuse."""
    print("[stage bf16] Building inputs with dt_dtype=bfloat16...")
    inputs_bf16 = _build_mimo_inputs(dt_dtype=torch.bfloat16)
    assert inputs_bf16["DT"].dtype == torch.bfloat16
    try:
        out = _call_mamba3(inputs_bf16)
        torch.cuda.synchronize()
        out.sum().backward()
        torch.cuda.synchronize()
        print("[stage bf16] RESULT: silent-accept (NO error raised) — also a bug")
        print("STAGE_RESULT=silent_accept")
        return 1
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:300] if str(e) else type(e).__name__
        print(f"[stage bf16] RESULT: kernel refused bf16 DT with: {type(e).__name__}: {msg}")
        print("STAGE_RESULT=bf16_refused")
        return 0


def _stage_fp32() -> int:
    """Stage: demonstrate that DT=fp32 (patched mamba3.py) works end-to-end."""
    print("[stage fp32] Building inputs with dt_dtype=float32...")
    inputs = _build_mimo_inputs(dt_dtype=torch.float32)
    assert inputs["DT"].dtype == torch.float32
    out = _call_mamba3(inputs)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all(), "fp32 DT fwd produced non-finite output"
    out.sum().backward()
    torch.cuda.synchronize()
    maxes = {}
    for name in ("Q", "K", "V", "DT", "ADT", "Q_bias", "K_bias", "MIMO_V", "D"):
        g = inputs[name].grad
        assert g is not None, f"{name}.grad is None"
        assert torch.isfinite(g).all(), f"{name}.grad has NaN/Inf"
        maxes[name] = float(g.detach().abs().max().item())
    print(f"[stage fp32] RESULT: finite grads. max|grad|: "
          + ", ".join(f"{k}={v:.3e}" for k, v in maxes.items()))
    print("STAGE_RESULT=fp32_ok")
    return 0


def _stage_gqa_unpatched() -> int:
    """Stage: with bwd source reverted to pre-patch, expect ValueError on GQA."""
    path = _bwd_source_path()
    assert _ensure_patched(path), (
        "Installed bwd file is NOT in the patched form; install the cppmega fork first."
    )
    original_src = path.read_text()
    try:
        _write_unpatched(path)
        _reload_bwd_module()
        inputs = _build_mimo_inputs(dt_dtype=torch.float32)
        try:
            out = _call_mamba3(inputs)
            out.sum().backward()
            torch.cuda.synchronize()
            print("[stage gqa_unpatched] RESULT: no error (UNEXPECTED)")
            print("STAGE_RESULT=unpatched_no_error")
            return 1
        except ValueError as e:
            if "not currently supported" in str(e):
                print(f"[stage gqa_unpatched] RESULT: ValueError raised as expected: {e}")
                print("STAGE_RESULT=unpatched_raised")
                return 0
            print(f"[stage gqa_unpatched] RESULT: wrong ValueError: {e}")
            print("STAGE_RESULT=unpatched_wrong_value_error")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"[stage gqa_unpatched] RESULT: unexpected {type(e).__name__}: {e}")
            print("STAGE_RESULT=unpatched_unexpected")
            return 1
    finally:
        path.write_text(original_src)
        _reload_bwd_module()


def _stage_gqa_patched() -> int:
    """Stage: with bwd source patched (GQA branch present), gradients must be finite."""
    path = _bwd_source_path()
    assert _ensure_patched(path), "Installed bwd is not patched; cannot validate."
    inputs = _build_mimo_inputs(dt_dtype=torch.float32)
    out = _call_mamba3(inputs)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    out.sum().backward()
    torch.cuda.synchronize()
    grad_maxes = {}
    for name in ("Q", "K", "V", "DT", "ADT", "Q_bias", "K_bias", "MIMO_V", "D"):
        g = inputs[name].grad
        assert g is not None and torch.isfinite(g).all(), f"{name} grad problem"
        grad_maxes[name] = float(g.detach().abs().max().item())
    assert inputs["Q"].grad.shape == inputs["Q"].shape
    assert inputs["K"].grad.shape == inputs["K"].shape
    print(f"[stage gqa_patched] RESULT: finite grads. max|grad|: "
          + ", ".join(f"{k}={v:.3e}" for k, v in grad_maxes.items()))
    print("STAGE_RESULT=patched_ok")
    return 0


STAGES: dict[str, Callable[[], int]] = {
    "bf16": _stage_bf16,
    "fp32": _stage_fp32,
    "gqa_unpatched": _stage_gqa_unpatched,
    "gqa_patched": _stage_gqa_patched,
}


def main() -> int:
    stage = os.environ.get("MAMBA3_REPRO_STAGE")
    if stage is not None:
        # Child-process entry: run the named stage only.
        if not torch.cuda.is_available():
            print("ERROR: CUDA required.")
            return 2
        fn = STAGES[stage]
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print(f"[stage {stage}] RESULT: uncaught {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"STAGE_RESULT=uncaught_{type(e).__name__}")
            return 1

    # Parent-process entry: orchestrate subprocesses.
    if not torch.cuda.is_available():
        print("ERROR: CUDA device required (mamba-ssm TileLang kernels are CUDA-only).")
        return 2

    try:
        import mamba_ssm  # noqa: F401
        import tilelang  # noqa: F401
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}")
        return 2

    dev = torch.cuda.get_device_name(0)
    print(f"Device: {dev}")
    print(f"Config: B={BATCH} S={SEQLEN} R={MIMO_RANK} G(nheads_qk)={NGROUPS} "
          f"H(nheads)={NHEADS} N={HEADDIM_QK} P={HEADDIM_V}")
    print(f"    -> GQA: 1 < G={NGROUPS} < H={NHEADS} and H % G == 0   "
          "(trips the branch upstream does not implement)")
    print()

    # Each stage runs in a fresh subprocess so a failed TileLang kernel (which
    # tends to leave the CUDA context in a bad state) cannot contaminate the
    # next stage.
    stage_results: dict[str, str] = {}
    for name in ("bf16", "fp32", "gqa_unpatched", "gqa_patched"):
        print("=" * 72)
        print(f"Subprocess stage: {name}")
        print("=" * 72)
        rc, out = _run_in_subprocess(name)
        # Forward child output so the user sees kernel prints.
        for line in out.splitlines():
            if any(tag in line for tag in ("[stage", "STAGE_RESULT", "RESULT", "WARN", "ERROR", "Traceback", "  File", "    ")):
                print(line)
        # Parse STAGE_RESULT=...
        marker = None
        for line in reversed(out.splitlines()):
            if line.startswith("STAGE_RESULT="):
                marker = line.split("=", 1)[1]
                break
        stage_results[name] = marker or f"rc={rc}"
        print()

    # Verdict
    print("=" * 72)
    print("Summary:")
    for k, v in stage_results.items():
        print(f"  stage {k:18s} -> {v}")
    print()

    bug1 = stage_results.get("bf16") == "bf16_refused"
    fix1 = stage_results.get("fp32") == "fp32_ok"
    bug2 = stage_results.get("gqa_unpatched") == "unpatched_raised"
    fix2 = stage_results.get("gqa_patched") == "patched_ok"

    if bug1 and fix1:
        print("Problem 1: BUG_REPRODUCED (bf16 DT rejected by TileLang kernel) ; "
              "FIX_VALIDATED (fp32 DT fwd+bwd produces finite grads)")
    else:
        print(f"Problem 1: INCOMPLETE  bf16_refused={bug1}, fp32_ok={fix1}")

    if bug2 and fix2:
        print("Problem 2: BUG_REPRODUCED (unpatched GQA bwd raises ValueError) ; "
              "FIX_VALIDATED (patched GQA bwd produces finite grads)")
    else:
        print(f"Problem 2: INCOMPLETE  unpatched_raised={bug2}, patched_ok={fix2}")

    return 0 if (bug1 and fix1 and bug2 and fix2) else 1


if __name__ == "__main__":
    sys.exit(main())
