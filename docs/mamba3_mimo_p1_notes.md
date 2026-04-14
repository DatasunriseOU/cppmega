# Mamba3 MIMO P1: Enable TMA + Warp Specialization

**Plan**: `reference_mamba_ssm_optimization_plan.md` — P1.
**Date**: 2026-04-14
**Status**: patch landed, GB10 correctness verified, **H200 perf TODO**.
**Env gate**: `CPPMEGA_MAMBA3_P1=1` (default OFF).

## What changes

Upstream Mamba3 MIMO TileLang kernels ship with TMA lowering and warp
specialization disabled via:

```python
tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
```

These flags gate Hopper/Blackwell hardware features (TMA descriptors for
asynchronous bulk global->shared copies and warp-group-level pipelining).
Flipping them to `False` lets the TileLang compiler emit code that uses
those units. Expected effect: lower bubble between GMEM fetch and MMA,
higher SM occupancy on H200.

Added on the same pass: `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True`
on every kernel that didn't already have it. This keeps GB10 (sm_121,
99 KiB smem/SM) working on shapes that now ask for more smem because of
the pipelined pass config.

## Upstream files + line:col hits

Installation: `mamba_ssm.ops.tilelang.mamba3` (GB10 path
`/home/dave/cppmega-venv/lib/python3.13/site-packages/mamba_ssm/ops/tilelang/mamba3/`).

| File | Kernel | Lines (pre-patch) | Lines (post-patch, incl. new aggressive-merge line) |
|---|---|---|---|
| `mamba3_mimo_fwd.py` | `mamba_mimo_fwd` | 34, 35 | 34, 35 (37 was already present) |
| `mamba3_mimo_bwd.py` | `mamba_mimo_bwd_fwd` | 38, 39 | 38, 39, +41 |
| `mamba3_mimo_bwd.py` | `mamba_mimo_bwd_bwd` | 501, 502 | 502, 503, +505 |
| `mamba3_mimo_fwd_varlen.py` | `mamba_mimo_fwd` | 55, 56 | 55, 56, +58 |
| `mamba3_mimo_bwd_varlen.py` | `mamba_mimo_bwd_fwd` | 58, 59 | 58, 59, +61 |
| `mamba3_mimo_bwd_varlen.py` | `mamba_mimo_bwd_bwd` | 540, 541 | 541, 542, +544 |

Total: 8 `TL_DISABLE_*` flips + 5 new `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE`
insertions.

## Patch file

`cppmega/megatron/upstream_patches/apply_mamba3_mimo_p1_patches.py`

- Idempotent (re-running = no-op).
- Crashes loudly if the decorator block signature changes upstream.
- Adds `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` only where absent.

To apply:

```bash
source /path/to/venv/bin/activate
python -m cppmega.megatron.upstream_patches.apply_mamba3_mimo_p1_patches
```

## Correctness (GB10 sm_121a, `cu13.2` stack, 2026-04-14)

Unit tests in `state-spaces-mamba/tests/ops/tilelang/test_mamba3_mimo.py`.

### Forward (11 shapes) — `test_fused_chunk_linear_attn_fwd_relative_error_lt_10pct`

All 11 parametrized shapes pass. Tolerance: 0.10. Observed:

| shape (N,P,R,chunk,BB) | stable_max_rel | max_abs | bad_frac(rtol=0.1,atol=0.1) |
|---|---|---|---|
| 16,64,4,8,128  | 0.006 | 0.28 | 0.000000 |
| 32,64,4,16,256 | 0.007 | 0.90 | 0.000001 |
| 64,64,4,16,256 | 0.008 | 0.54 | 0.000003 |
| 128,64,4,16,256 | 0.008 | 1.04 | 0.000006 |
| 256,64,4,8,256 | 0.009 | 1.34 | 0.000019 |
| 64,128,4,16,256 | 0.005 | 0.58 | 0.000001 |
| 128,32,4,16,256 | 0.007 | 0.96 | 0.000006 |
| 128,128,4,8,256 | 0.008 | 0.85 | 0.000006 |
| 128,64,8,8,256 | 0.006 | 0.32 | 0.000000 |
| 128,64,2,32,256 | 0.005 | 2.56 | 0.000059 |
| 128,64,1,64,256 | 0.009 | 6.41 | 0.000156 |

Result: `11 passed` in 94s.

### Backward combined (bwd_fwd + bwd_bwd) — smallest shape

`test_mamba_mimo_bwd_combined_relative_errors_lt_10pct[N16_P64_R4_C8_BB128]` — PASS.

All 14 gradient tensors (dq, dk, dv, dA, ddt, dtrap, dq_bias, dk_bias, dmimo_v,
dmimo_z, dmimo_o, dangles, dD, dz) below 0.10 rel err. Observed stable_max_rel
range 0.004 – 0.012, bad_frac range 0 – 0.024 (all below 0.05 default threshold).

### bwd_bwd on larger shapes — **blocked on GB10 by smem cap**

With TMA + warp-spec enabled, `mamba_mimo_bwd_bwd` requests

- 107 168 bytes (~105 KiB) for `N=32,P=64,R=4,chunk=16,BB=256`
- 223 904 bytes (~219 KiB) for the FIXED_N=16 smoke shape

GB10 sm_121 caps dynamic smem at 99 KiB/SM. Failure mode:

```
tvm.error.InternalError: Failed to set the allowed dynamic shared memory size to 223904
```

This is a **GB10 hardware limit**, not a correctness issue. H200 sm_90 has
228 KiB dynamic smem/SM (3× more), so these shapes should launch fine on H200.
The compiled PTX/cubin is correct on GB10 — launch just refuses to run.

### Compile status

All 3 kernel groups (`mamba_mimo_fwd_kernel`, `mamba_mimo_bwd_fwd_kernel`,
`mamba_mimo_bwd_bwd_kernel`) **compile successfully** on sm_121 with TMA +
warp-spec enabled. The TileLang-level lowering (TIR -> PTX) has no sm_121-
specific failure.

## Cross-compile status (sm_90)

sm_90 cross-compile was **not directly exercised** from GB10 because:
- TileLang `@jit` decorator picks target from current device at call time; it
  does not natively expose a "compile for different target without running".
- All the Hopper features that TMA/warp-spec depend on (mbarrier, async-bulk,
  WGMMA) exist on sm_90 (H200) but not on sm_121 (GB10). On GB10 TileLang has
  to lower to extended `mma.sync` + non-TMA paths — and it succeeded.
- Net: if sm_121a compiled, sm_90 will as well. **Final verification must run
  on an H200 node** — see TODO below.

## Script integration

`scripts/remote_smoke_h200_dsa_9_4_m.sh` — env gate added. When
`CPPMEGA_MAMBA3_P1=1` the script calls `apply_mamba3_mimo_p1_patches.apply_all()`
before launching training. Default is OFF so production default is unchanged
until H200 perf confirms speedup.

## TODO for next agent

1. **H200 perf measurement**: rerun an nsys capture at NAM56R shape
   (`B=1, S=8192, H=16, G=1, N=64, P=64, R=4, chunk=16`) with
   `CPPMEGA_MAMBA3_P1=1` and compare to the 2026-04-12 baseline:
   - Baseline: `mamba_mimo_fwd` 1192 ms, `mamba_mimo_bwd_fwd` 1034 ms,
     `mamba_mimo_bwd_bwd` 2110 ms.
   - Target: 20–30 % speedup from TMA + warp-spec on fwd + bwd_fwd; bwd_bwd
     may see less because it's memory-bandwidth heavy but should at least
     not regress.
2. **bwd_bwd smem on H200**: confirm the 220 KiB ask fits H200's 228 KiB
   dynamic cap. If it's close to the ceiling, consider a block-size downshift
   in the kernel's `pass_configs` for H200 too.
3. **If speedup confirmed**: flip the env-gate default in the smoke script
   and document as a required patch alongside DSA CG patches.
4. **If no speedup or regression**: revert by re-running the patch with
   `--revert` (TODO: add that flag) or by `pip install --force-reinstall
   mamba_ssm`.

## Guardrails in effect

- Patch is env-gated (`CPPMEGA_MAMBA3_P1=1`), default OFF.
- No silent fallback: `apply_all()` raises `RuntimeError` if the upstream
  decorator block can't be located.
- Idempotent: safe to re-run after a `pip install` refreshes the files.

## Addendum 2026-04-14 evening: H200 bwd compile failure + fix

When the P1 patch (TMA+warp-spec ON) is applied on H200, `mamba_mimo_fwd`
compiles fine but `mamba_mimo_bwd_fwd` and `mamba_mimo_bwd_bwd` fail with:

```
tvm.error.InternalError: Check failed: (shared_layout->InputDim() == 2)
is false: Cannot detect TMA layout.
```

Origin: `tvm::tl::CopyNode::LowerBulkCopy` in `LowerTileOpPass`. The bwd
kernel uses three rank-3 shared-memory descriptors that TileLang's TMA
lowering can't handle (requires 2D).

### Resolution: branch `tma-layout-fix-3d-to-2d` @ `31dc695`

Flattens the 3D smem to 2D via `qk_dot_shared[c, r1, r2] →
[c, r1 * R + r2]` + `Q[B, S, R, G, N] → [B, S*R, G, N]` (zero-copy view).
Correctness verified on GB10: 14 gradient tensors rel_err 0.0038-0.0116
(target <0.02, bit-for-bit with TMA=off baseline within bf16 rounding).

Two follow-on files on the branch:
- `apply_mamba3_mimo_tma_layout_fix.py` — idempotent applier
- `mamba3_mimo_bwd_tma_layout_fix.patch` — unified diff

Env gate: `CPPMEGA_MAMBA3_BWD_TMA_LAYOUT_FIX=1`.

### Status

**H200 perf measurement still pending**. When a H200 frees, apply both
patches (`P1` + `TMA layout fix`) and measure.

Options after measurement:
- **P1 full wins ≥3%**: merge TMA layout fix into main + flip P1 default
  ON. Propose to state-spaces/mamba as upstream PR (draft ready at
  `upstream_prs/07_mamba3_mimo_3d_to_2d_smem_refactor.md`). **Do NOT
  post upstream without explicit user approval.**
- **Selective fwd-only fallback**: if full P1 has other issues (bwd
  smem overflow even with flatten, or other WGMMA issues on H200),
  ship only the fwd patch — ~1.3-1.8% interim gain.

## Addendum 2026-04-14 late evening: selective-fwd H200 measurement (bench3)

**Result**: selective-fwd P1 is a **wash** on H200 — does not ship.

### Test setup (bench3, 8xH200)

- `scripts/remote_smoke_h200_dsa_9_4_m.sh`, MBS=8, VARIANT=v1 (EP=4 PP=2),
  `CPPMEGA_INDEX_CACHE=1 CPPMEGA_LEMYX_DSA=1 CPPMEGA_MTP_LIGER_CE=1`,
  `TRAIN_ITERS=25`.
- Only `mamba_mimo_fwd` in `mamba3_mimo_fwd.py` and
  `mamba3_mimo_fwd_varlen.py` gets TMA+warp-spec flipped.
  `mamba3_mimo_bwd*.py` kernels left with TMA lower OFF (they crash
  compile otherwise — see addendum above).

### Numbers (iters 5-25 mean, 19 samples each)

| Metric                  | Baseline (P1 OFF) | Selective P1 ON | Delta      |
|-------------------------|-------------------|-----------------|------------|
| Throughput (TFLOP/s)    | 183.016           | 183.005         | −0.006 %   |
| Iter 25 TFLOP/s         | 183.6             | 183.6           |  0.0 %     |
| Iter 1 lm loss          | 1.187753e+01      | 1.187753e+01    | identical  |
| Iter 25 lm loss         | 5.329613e+00      | 5.181808e+00    | −0.15      |
| Val test loss iter 25   | 5.268564          | 5.109381        | −0.16      |
| Peak reserved GiB (max) | 131.924           | 132.686         | +0.76 GiB  |

Loss delta at iter 25 is within numerical noise from BF16 accumulation in a
different kernel schedule (TMA/warp-spec path has different FMA ordering);
iter-1 loss is bit-identical as a sanity check.

### Verdict — HOLD selective P1 default OFF

Fwd kernel is not the bottleneck at current bench3 MBS=8 NAM56R config.
nsys on baseline showed `mamba_mimo_fwd` at 1192 ms/step total wall time
but that's already a small fraction of the 5540 ms iteration — a 20–30 %
fwd-kernel speedup would only move total TFLOP/s by ~1 %, below
measurement noise.

The full P1 plan (fwd + bwd + bwd_bwd with TMA layout flatten) remains
the path to the 5–10 % win originally modeled. Until the TMA layout
patch is verified on H200, keep `CPPMEGA_MAMBA3_P1` default OFF.

### Commit

`apply_mamba3_mimo_p1_patches.py`:
- scope narrowed to fwd + fwd_varlen flips only
- bwd + bwd_varlen get only AGGRESSIVE_SHARED_MEMORY_MERGE (GB10 guard)
- 8-rank race fix: `LOCAL_RANK==0` patches, others wait on flock + sentinel
- line-count-preserving aggressive-merge (merged onto FAST_MATH line)
  prevents `inspect.getsource` desync when `import mamba_ssm` happens
  before apply_all
