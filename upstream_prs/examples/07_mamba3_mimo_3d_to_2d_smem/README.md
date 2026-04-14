# 07 — Mamba3 MIMO bwd: 3D → 2D smem refactor (reproducer)

Validates the source-side kernel refactor proposed in
`upstream_prs/07_mamba3_mimo_3d_to_2d_smem_refactor.md`, targeting
`state-spaces/mamba` `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`.

**NOT the same as reproducer 08.** 08 exercises TileLang's
`LowerBulkCopy` assertion (compiler-side). 07 exercises the kernel
author's workaround: flatten the three rank-3 smem descriptors
(`qk_dot_shared [C, R, R]`, Q/K views `[C, R, N]`) and the 5D gmem
signature `Q[B, S, R, G, N]` → `Q[B, S*R, G, N]` so the copies become
rank-2 and TMA+warp-spec light up.

## What it checks

Two tiny kernels, same numerical recipe, different smem layout:

| Variant | Q gmem            | Q smem            | qk_dot smem        |
|---------|-------------------|-------------------|--------------------|
| 3D (pre)  | `[B, S, R, G, N]` | `[C, R, N]`       | `[C, R, R]`        |
| 2D (post) | `[B, S*R, G, N]`  | `[C*R, N]`        | `[C, R*R]`         |

It then verifies:

1. **ASSERTION_HIT_AT_3D** or **COMPILE_FALLBACK_AT_3D** when evidence is
   actually captured — the 3D path cannot reach TMA. On TileLang pre-#746 it
   hard-asserts; post-#746 it falls back to `cp.async` and emits warning-like
   evidence. If no assertion/fallback signal is captured, the reproducer just
   reports `STATUS: OK` for the 3D branch and does not synthesize a fallback
   tag.
2. **CLEAN_COMPILE_AT_2D** — the refactor compiles cleanly with TMA
   lowering enabled.
3. **CORRECTNESS_PASS** — both variants produce identical `qk_dot`
   output for the same Q/K input (same fp32 accumulate, same bf16
   inputs; indexing is algebraically equivalent).

This reproducer is meant as a legality/correctness proof for the source
rewrite. It shows that flattening the rank-3 descriptors to rank-2 is a
clean compile-and-correctness refactor. It is not an H200 performance
receipt.

## Run

CUDA hardware required. The reproducer is self-contained — it embeds the
indexing pattern directly and only depends on `torch` + `tilelang` (pins
listed in `requirements.txt`). No `mamba_ssm` checkout is needed.

```bash
pip install -r requirements.txt
python reproducer.py
```

Validated environments / scope (2026-04-14 session):

- GB10 (sm_121a): retained legality/correctness validation with
  `TILELANG_VERSION=0.1.8+cuda` via Modal image.
- H200 (sm_90): compatible target environment for the same compile path,
  but retained H200 performance validation is still pending.

### Expected output

```
TILELANG_VERSION : 0.1.8+cuda.gitXXXXXXXX
DEVICE           : NVIDIA GB10 (sm_121)   # or H200 (sm_90)
SHAPES           : B=1 S=8 R=4 G=1 N=16 chunk=4
PR 07 target     : state-spaces/mamba — mamba3_mimo_bwd.py

[A] 3D smem variant ...
     STATUS: ASSERTION_HIT_AT_3D       (pre-PR-746 TileLang)
      or     COMPILE_FALLBACK_AT_3D     (post-PR-746 TileLang, warning captured)
      or     OK                         (compiled, no fallback warning captured)

[B] 2D smem variant ...
    STATUS: CLEAN_COMPILE_AT_2D

[C] Correctness ...
    STATUS: CORRECTNESS_PASS

TAGS: [optional ASSERTION_HIT_AT_3D or COMPILE_FALLBACK_AT_3D] CLEAN_COMPILE_AT_2D CORRECTNESS_PASS
VERDICT: OK
```

Exit codes: `0` on success (2D compiles + correctness), `1` if the 2D
refactor itself breaks or 3D/2D disagree numerically.

## Validation status

- Retained evidence in this pack: GB10 legality/correctness validation.
- Not yet retained in this pack: H200 performance measurement.
- Read this example as "rewrite validated for legality/correctness; H200
  perf still pending," not as a broad H200 performance proof.

## Files

- `reproducer.py` — self-contained kernel pair.
- `requirements.txt` — pin notes.
- (No external deps on `mamba_ssm` — the reproducer embeds the
  offending indexing pattern directly, so it runs in any environment
  with TileLang + torch + CUDA.)
