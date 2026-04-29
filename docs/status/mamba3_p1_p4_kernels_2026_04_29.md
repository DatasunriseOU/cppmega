# Mamba3 MIMO P1/P4 Kernel Status - 2026-04-29

Scope: P1 smem/register pressure in TileLang MIMO kernels, plus H200-specific
TMA/warp-specialization only where the kernel layout can actually support it.
Apple Newton and ParaRNN were not evaluated.

## Four waves

| Wave | Checked/profiled | Hypothesis | Patch/bench/no-go | Result |
| --- | --- | --- | --- | --- |
| 1. Baseline source/layout | Resolved source with `PYTHONPATH=/home/dave/state-spaces-mamba`; `py_compile` on the four MIMO TileLang files; static scan of jit flags and `T.alloc_shared` sites. | The local installed wheel is not a valid runtime target, but the source checkout is sufficient for P1/P4 static analysis. | No patch. Read-only source analysis. | Installed `mamba_ssm` lacks `mamba_ssm.ops.tilelang`. Source import resolves to `/home/dave/state-spaces-mamba/mamba_ssm/...`. All 6 MIMO jit sites have `TL_DISABLE_TMA_LOWER=True`, `TL_DISABLE_WARP_SPECIALIZED=True`, and no aggressive smem merge. Backward keeps a rank-3 `qk_dot_shared = T.alloc_shared([chunk_size, R, R], dtype)`. |
| 2. Selective P1 patch probe | Ran `apply_mamba3_mimo_p1_patches` internals against a `/tmp` copy of the four source files, then `py_compile` on the patched copy. | Existing selective-fwd patch is mechanically valid and should not flip backward TMA. | Bench/probe only; no source or site-packages mutation. Hardened the local appplier gate because live mutation is still unsafe. | Probe changed exactly 2 fwd sites and inserted 6 aggressive-merge entries. `mamba3_mimo_bwd*.py` stayed at `TL_DISABLE_TMA_LOWER=True` with zero P1 markers. Patched temp copy compiled. |
| 3. P4/H200 TMA layout viability | Ran existing TileLang reproducers with source `PYTHONPATH`: `08_tilelang_tma_bulk_copy_3d_smem` and `13_tilelang_floormod_dbz`. | Full backward TMA/warpspec may be blocked by current TileLang layout inference even after 3D->2D smem flatten. | No-go for runtime bwd TMA/warpspec. | `08` passed on local GB10/TileLang `0.1.8+cuda.gitf309d814`: rank-3 smem no longer hard-asserts. `13` reproduced `LayoutInference FloorMod divide-by-zero` in `mamba_mimo_bwd_bwd` after flatten + TMA/WS ON. This is the current hard blocker for full P1/P4. |
| 4. Runtime/profile decision | Checked GPU and Modal availability; local device is GB10 sm_121, Modal CLI exists. Reviewed prior H200 selective-fwd measurement. | H100 Modal smoke would not answer H200-specific bwd TMA viability; local GB10 cannot measure H200 occupancy/register wins. | No Modal run. Status doc + safety patch only. | Runtime H200/B200 profile is not measurable here without a target H200/B200 source build. Prior bench3 selective-fwd result remains the best H200 data: 183.016 vs 183.005 TFLOP/s, -0.006%, peak reserved +0.76 GiB. |

## Commands and outputs

Key commands run from this worktree:

```bash
git status --short --branch
python - <<'PY'  # import discovery for installed mamba_ssm/tilelang/torch
PY
find /home/dave -path '*/mamba_ssm/ops/tilelang/mamba3/*.py' -print 2>/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/dave/state-spaces-mamba:$PYTHONPATH \
  python -m py_compile \
  /home/dave/state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py \
  /home/dave/state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py \
  /home/dave/state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd_varlen.py \
  /home/dave/state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd_varlen.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/dave/state-spaces-mamba:$PYTHONPATH \
  python upstream_prs/examples/08_tilelang_tma_bulk_copy_3d_smem/reproducer.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/dave/state-spaces-mamba:$PYTHONPATH \
  python upstream_prs/examples/13_tilelang_floormod_dbz/reproducer.py
```

Measured/probed facts:

- Local GPU: `NVIDIA GB10`, compute capability `(12, 1)`, driver `595.58.03`.
- Torch: `2.13.0.dev20260417+cu132`; CUDA available with 1 device.
- Modal CLI: `1.4.2`.
- Source path: `/home/dave/state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3/`.
- Build copy also exists at `/home/dave/state-spaces-mamba/build/lib.linux-aarch64-cpython-313/mamba_ssm/ops/tilelang/mamba3/`.
- Installed `/home/dave/cppmega-venv/.../mamba_ssm` does not expose `mamba_ssm.ops.tilelang`.
- P1 temp-copy probe: `flips 2 aggr 6`; fwd files have 1 P1 marker each; bwd files have 0 P1 markers and 2 remaining `TL_DISABLE_TMA_LOWER: True` sites each.
- `08_tilelang_tma_bulk_copy_3d_smem`: exit 0, rank-3 and rank-2 cases compile; verdict OK for PR #746 warn/fallback behavior.
- `13_tilelang_floormod_dbz`: exit 1 by design, reproduced `Check failed: pb->value != 0 (0 vs. 0) : Divide by zero` through `tilelang.transform.LayoutInference`.

## Changed files

- `cppmega/megatron/upstream_patches/apply_mamba3_mimo_p1_patches.py`
  - Enforces `MAMBA3_P1_ALLOW_FILE_MUTATION=1` in addition to `CPPMEGA_MAMBA3_P1=1` before mutating live `mamba_ssm` files.
  - Updates usage text so script mode is no longer an accidental mutation path.
- `tests/test_mamba3_mimo_p1_patches.py`
  - Covers no-op primary gate, refusal without mutation acknowledgement, and successful call-through when both gates are set.
- `docs/status/mamba3_p1_p4_kernels_2026_04_29.md`
  - This four-wave status note.

## Recommendation

Do not ship a runtime P1/P4 patch today. Keep all backward MIMO kernels with
TMA lower and warp specialization disabled. Selective forward P1 is mechanically
safe but already measured as a wash on H200, so it should remain default-off.

Next H200/B200 experiment:

1. Build or select a source-based `mamba_ssm` runtime where
   `mamba_ssm.ops.tilelang.mamba3` imports from the intended checkout, not the
   installed wheel.
2. Patch TileLang/TVM for the `FloorMod` LayoutInference crash, or use a
   TileLang build where `upstream_prs/examples/13_tilelang_floormod_dbz` exits 0.
3. Only then apply the 3D->2D bwd flatten plus TMA/warpspec on `bwd_fwd` and
   `bwd_bwd`, and run a bounded H200/B200 profile comparing:
   `mamba_mimo_fwd`, `mamba_mimo_bwd_fwd`, `mamba_mimo_bwd_bwd`, requested smem,
   registers/thread, achieved occupancy, and NAM56R iter TFLOP/s.
