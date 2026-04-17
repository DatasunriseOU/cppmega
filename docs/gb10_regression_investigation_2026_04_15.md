# GB10 Regression Investigation — 2026-04-15

## Executive summary

GB10 NAM56R-half single-GPU training fails in forward with
`cudaErrorMisalignedAddress` at `mamba_mimo_fwd_kernel`. Root cause:
**commit `4f115ea` (2026-04-14 03:11 +0200) "P1 — enable TMA + warp
specialization in Mamba3 MIMO kernels"** was applied to the installed
mamba_ssm site-packages kernels and the modification persists on disk.
The env gate `CPPMEGA_MAMBA3_P1` is a no-op for preventing regression
once the files have been mutated.

## Current broken state (2026-04-15)

- `torch 2.12.0.dev20260407+cu132`
- `transformer_engine 2.13.0` (installed 2026-04-10 14:36)
- `tilelang 0.1.8+cuda.gitf309d814` (built 2026-04-14 10:07)
- `mamba_ssm 2.3.1` in `/home/dave/cppmega-venv/lib/python3.13/site-packages/mamba_ssm/`
  (editable-linked fork at `/home/dave/state-spaces-mamba`, HEAD `31f3d7b`)
- `triton 3.7.0`
- cuDNN 9.20.0.48 (system apt `libcudnn9-cuda-13`) plus bundled
  `nvidia/cudnn/lib/libcudnn*.so.9` in venv
- Current cppmega on GB10: `/home/dave/cppmega-nan-test` HEAD `4b979e7`
  (branch does NOT contain commit `4f115ea`)

Failing log: `/home/dave/logs/gb10_nan_test4_20260415_000925.log`
Failure site: `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py:462` —
`mamba_mimo_fwd_kernel_kernel: cudaErrorMisalignedAddress`.

## Root cause — the installed kernels are still carrying P1 patches

`stat` on `/home/dave/cppmega-venv/lib/python3.13/site-packages/mamba_ssm/ops/tilelang/mamba3/`:

| file | mtime | cppmega P1 markers |
|------|-------|--------------------|
| `mamba3_mimo_fwd.py` | 2026-04-14 01:02:59 UTC | 2 |
| `mamba3_mimo_bwd.py` | 2026-04-14 10:52:46 UTC | 6 |
| `mamba3_mimo_fwd_varlen.py` | 2026-04-14 01:02:59 UTC | 3 |
| `mamba3_mimo_bwd_varlen.py` | 2026-04-14 01:02:59 UTC | 6 |

`diff` vs upstream fork (`/home/dave/state-spaces-mamba`):

```
<         tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,  # cppmega P1
<         tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,  # cppmega P1
<         tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
---
>         tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
>         tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
```

Plus `@autotune(...)` enabled. The TMA-lower path on TileLang
`f309d814` produces bulk-copy descriptors that assume aligned
multi-byte boundaries; the combination of these flags against tile
shapes in `mamba_mimo_fwd_kernel` produces unaligned addresses on
sm_121a, exactly the CUTLASS sm_120/sm_121 alignment-bug class noted
in the task.

## Why CPPMEGA_MAMBA3_P1 did not protect us

`scripts/remote_smoke_h200_dsa_9_4_m.sh:412` reads the env var at
launch. If unset (default 0), the script skips `_apply_mamba3_p1()`.
However the patch is **non-reversible on disk** — running the script
once with `CPPMEGA_MAMBA3_P1=1` (which happened on 2026-04-14 between
01:02 and 10:52 UTC; earlier logs have been rotated so we can't point
at the exact run) rewrote the site-packages `.py` files. Every
subsequent run — even with `CPPMEGA_MAMBA3_P1=0` — picks up the
mutated source directly via Python import.

The current `apply_mamba3_mimo_p1_patches.py` has no restore path.

Also note: current nan-test branch `4b979e7` predates `4f115ea` in
main's history, so the script calling `_apply_mamba3_p1` is NOT in the
code path today — the disk state alone is driving the failure.

## Was GB10 ever fully working end-to-end on this box?

No evidence on disk. `/home/dave/logs/` has only the 4 `gb10_nan_test*`
runs from 2026-04-15 00:03–00:10, all of which crash before iter 0
completes. The only other NAM56R-shaped logs are on the bench3 nodes
(`/home/dave/cppmega/.tmp/modal_b200_*` are B200/Modal sweeps, not
training). The memory note `reference_gb10_working_stack_2026_04.md`
describes what libs work; it does not claim a live NAM56R-half training
run on GB10.

## cuDNN fused-attn sublib failure (second error)

Not reproduced in log4 — the TileLang misalign fires first, killing
forward before the 4 MLA full-attention layers execute. Likely
explanation when re-enabled: TE 2.13 requires cuDNN `>=9.20`; bundled
venv `nvidia/cudnn/lib` and system apt are both 9.20.0.48, which
should satisfy — but `LD_LIBRARY_PATH` order in the wrapper puts
venv cudnn *after* `nccl/lib`, and the sublib loader may be finding a
mismatched `libcudnn_engines_*` combination. Investigate after the
TileLang fix so you can actually reach the MLA layer.

## Recommended fix path

1. **Restore the installed mamba_ssm kernels to upstream**:
   ```
   cd /home/dave/state-spaces-mamba
   MAMBA_FORCE_BUILD=TRUE pip install --no-deps --force-reinstall .
   ```
   (the `reference_mamba_ssm_reinstall.md` caveat — never use
   `pip --force-reinstall mamba_ssm`; reinstall from fork dir only).

2. **Harden the patch** — `apply_mamba3_mimo_p1_patches.py` must write
   to a copy (e.g. `mamba_ssm_p1/…`) monkey-patched at import time, NOT
   to the installed files. Same principle as the linear-CE patch. File
   an issue and block future runs of the in-place script.

3. **Only then** retest the cuDNN fused-attn error. Likely fix is
   explicit `LD_PRELOAD=/home/dave/cppmega-venv/lib/python3.13/site-packages/nvidia/cudnn/lib/libcudnn.so.9`
   or reorder `LD_LIBRARY_PATH` so venv cudnn wins over apt.

## Files inspected

- `/home/dave/cppmega-venv/lib/python3.13/site-packages/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py`
- `/home/dave/state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py`
- `/home/dave/cppmega/cppmega/megatron/upstream_patches/apply_mamba3_mimo_p1_patches.py`
- `/home/dave/cppmega-nan-test/scripts/remote_smoke_h200_dsa_9_4_m.sh`
- `/home/dave/logs/gb10_nan_test4_20260415_000925.log`
