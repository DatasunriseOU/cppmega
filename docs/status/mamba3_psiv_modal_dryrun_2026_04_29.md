# Mamba3 Hoist-PsiV Modal Dry-Run - 2026-04-29

Branch: `worker/mamba3-psiv-modal-dryrun`

Goal: source-based GHCR dry-run for Hoist-PsiV on
`mamba_ssm.ops.tilelang.mamba3`, with real GPU measurement before any
production move.

## Modal Runs

| Requested | App | Actual device | Status |
| --- | --- | --- | --- |
| `H100:2` | `ap-lxELUnil4kpWenGBWZy6B1` | `NVIDIA H100 80GB HBM3` | completed |
| `H200:2` | `ap-1hJRxUmDtR6ZAmnlsbBYCJ` | `NVIDIA H200` | completed |
| `B200+:2` | `ap-Pm9kV6s8cd3P6EzqCT1ZVR` | not allocated | stopped after provisioning wait |
| `B200:2` | `ap-SRa7KnD4mgxALM7UHX00EC` | not allocated | stopped after provisioning wait |
| `B200:1` | `ap-n3nUopYua5RlIfk8ucnnEk` | not allocated | stopped after provisioning wait |

The `H100:2` run was executed before the report-label fix, so its JSON says
`gpu_spec=H200:2`; the actual CUDA device was H100 and the Modal request was
`H100:2`.

## Source-Based Probe

The GHCR image overlays local cppmega plus source checkout:

- `/opt/cppmega/cppmega`
- `/opt/state-spaces-mamba/mamba_ssm`
- `/opt/cppmega/upstream_prs/examples/13_tilelang_floormod_dbz`

Patch-site probe on `/opt/state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3`:

| Probe | Count |
| --- | ---: |
| required files present | `5/5` |
| fwd PsiV materialization sites | `2` |
| bwd PsiV materialization sites | `8` |
| reverse-scan `dstates` updates | `2` |
| autograd wrapper present | yes |
| ready for patch skeleton | yes |

## PsiV Materialization Cost

Times are CUDA-event means. `alloc_cast` is the current Python helper
(`V.unsqueeze(3) * MIMO_V.to(dtype)` + contiguous allocation). `out_precast`
uses a preallocated output and precast `MIMO_V`, which is the closest proxy for
kernel-side write price.

| GPU | Shape | Cache | `alloc_cast` | `out_precast` |
| --- | --- | ---: | ---: | ---: |
| H100 | `B=1 S=8192 H=16 R=4 P=64` | 64 MiB | 0.122 ms | 0.112 ms |
| H100 | `B=4 S=4096 H=32 R=4 P=128` | 512 MiB | 0.799 ms | 0.785 ms |
| H100 | `B=10 S=4096 H=32 R=4 P=128` | 1.25 GiB | 1.952 ms | 1.939 ms |
| H200 | `B=1 S=8192 H=16 R=4 P=64` | 64 MiB | 0.118 ms | 0.109 ms |
| H200 | `B=4 S=4096 H=32 R=4 P=128` | 512 MiB | 0.792 ms | 0.783 ms |
| H200 | `B=10 S=4096 H=32 R=4 P=128` | 1.25 GiB | 1.950 ms | 1.937 ms |

For 9 Mamba3 layers/rank, raw one-time PsiV write price is roughly:

- MBS=4 production shape: `9 * 0.783 ms ~= 7.0 ms/step` plus `4.5 GiB`
  activation memory.
- MBS=10 stress shape: `9 * 1.937 ms ~= 17.4 ms/step` plus `11.25 GiB`
  activation memory.

That is not a blocker by itself, but it is also not free. The production gate
should remain: Hoist-PsiV only ships if patched `bwd_fwd`/`bwd_bwd` drops by
more than this write/memory pressure.

## Source TileLang Baseline

Source-overlay TileLang split bench at `B=2 S=1024 H=16 G=1 N=P=64 R=4 chunk=16`:

| GPU | `bwd_fwd` | `bwd_bwd` | chain |
| --- | ---: | ---: | ---: |
| H100 | 0.316 ms | 0.659 ms | 0.975 ms |
| H200 | 0.266 ms | 0.664 ms | 0.930 ms |

This verifies the source overlay path compiles and runs the current baseline.
It is not an A/B Hoist-PsiV speedup measurement yet.

## FloorMod / TMA-WS Probe

The existing layout-fix + TMA/warpspec variant still reproduces the TileLang
FloorMod divide-by-zero:

- `InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero`
- `is_floormod_dbz=true`

The temp-only no-FloorMod rewrite removed all `% R` occurrences introduced by
the patch (`remaining_percent_R=0`), so the original DBZ is bypassed. Compile
then fails on a different TileLang layout-inference issue:

- `Loop layout is not injective`
- buffer: `qk_dot_frag` / `qk_dot_shared`
- `is_floormod_dbz=false`

Verdict: no, backward TMA/warpspec is not fixed. We fixed the immediate
FloorMod symptom in a temp rewrite, but the branch remains blocked by a second
LayoutInference failure.

## Decision

Do not move Hoist-PsiV into production yet.

What is validated:

- source-based GHCR Modal harness works for H100/H200;
- patch sites are present and ready for a non-mutating patch skeleton;
- PsiV memory/write price is measured and small enough to justify a real A/B
  kernel prototype;
- current TMA/WS layout-fix branch remains non-production.

Next useful implementation step:

1. Implement Hoist-PsiV against the non-TMA production TileLang path first.
2. Benchmark patched vs baseline `bwd_fwd` and `bwd_bwd` on H200 at production
   shape.
3. Only if kernel time drops by more than the 7-18 ms/rank write-price band,
   wire it behind an env gate for larger NAM56R runs.
