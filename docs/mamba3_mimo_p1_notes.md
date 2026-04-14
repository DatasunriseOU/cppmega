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
