# Mamba3 MIMO P3: Register-Pressure Split Design

**Plan**: `reference_mamba_ssm_optimization_plan.md` — P3 (hard, ~500-800 ms saving).
**Status**: design only (test-loop hour 6-8 prep, 2026-04-14). Implementation deferred
until P1 full ships.
**Target**: cut `mamba_mimo_bwd_bwd` register count from **255 → ~130** so occupancy
climbs from 12.5% (SASS-confirmed) to ~25%. Expected kernel speedup on H200 ≈
30-50% of the 2110 ms bwd_bwd time → **~1% total TFLOP/s**.

## Current kernel profile (2026-04-12 nsys, europe H200)

| Kernel | Time | Regs | Smem | Occupancy |
|---|---|---|---|---|
| mamba_mimo_fwd     | 1192 ms | 239 | 196 KiB | 6.2% |
| mamba_mimo_bwd_fwd | 1034 ms | 255 | 196 KiB | 6.2% |
| **mamba_mimo_bwd_bwd** | **2110 ms** | **255** | **228 KiB** | **12.5%** |

Register cap on H200 per SM = **65 536 x 32b**. Per-thread limit that compiler enforces
to achieve 100% occupancy at `threads=128` is `65536 / (2 × 128) = 256`. We sit right
at 255 = compiler has already clamped to the ceiling and is spilling on top. A 100-reg
cut doubles active warps per SM.

## Live-set analysis of `bwd_bwd`

`mamba3_mimo_bwd.py::mamba_mimo_bwd_bwd_kernel` computes, per chunk, the second-order
gradients for the MIMO scan:

1. **Load Q, K, V** (post-rotary) → ~5 frag tiles live
2. **Recompute `PsiV = V * Psi`** (Psi from bias/A/dt recompute) → 3 fragments
3. **Recompute `qk_dot`** (Q @ K^T per R×R block) → 1 fragment
4. **Read saved STATES tensor** (from bwd_fwd) + apply reverse scan → **10+ state frags**
   live concurrently (this is the biggest live-set)
5. **Compute d(dstates)** from dP, dP̂ → 3 frags
6. **dq, dk accumulators** → 2-4 frags
7. **dtrap, dq_bias, dk_bias, dA, ddt accumulators** → 5-7 small frags
8. **Write dq/dk/dv/dtrap/d*** to gmem

Dominant live-set = **step 4** (state-reversal accumulators) + **step 2** (PsiV
recompute). Both exist simultaneously for the inner-most loop body. Together they
need ~140-150 reg-tile units.

## Split proposal

Partition `bwd_bwd` into **two kernels** connected by gmem tensors:

### Pass 1: State-reverse — computes `dstates` ring buffer + `dq/dk` from state path

Inputs: `STATES` (from bwd_fwd), `dO`, `Q_rope`, `K_rope` (read-only), topk indices.
Outputs (to gmem): `dstates_per_chunk [B, H, nchunks, N, P]`, `dq_from_state`,
`dk_from_state` accumulators. Size ≈ 2 GiB extra for NAM56R MBS=8.

- Loads only Q/K/dO/STATES.
- Does the state-reverse inner loop end-to-end, accumulating dstates + partial dq/dk.
- Drops PsiV / qk_dot recompute for this pass — those belong to pass 2.
- Live set after split: ~10 state accumulators (unchanged) but NO PsiV/qk_dot frags.
  Expected reg count: **~140-150** (compiler can then unroll less aggressively too).

### Pass 2: Chunk-local — consumes `dstates_per_chunk`, computes remaining gradients

Inputs: `V`, `Q_rope`, `K_rope`, `Psi`, `dO`, `dstates_per_chunk` (from Pass 1),
topk indices.
Outputs: `dv`, `dtrap`, `dq_bias`, `dk_bias`, `dA`, `ddt`, `dD`, `dz` + adds to
`dq`, `dk`.

- Recomputes `PsiV = V * Psi` once per chunk.
- Recomputes `qk_dot = Q @ K^T`.
- Does the chunk-local backprop for the bias / D / z / A / dt terms.
- Live set: just PsiV/qk_dot + small accumulators. **~100-110 regs**.

### Memory cost of split

Extra gmem for `dstates_per_chunk`: shape `[B, H, nchunks, N, P]` with H=16, nchunks=seq/chunk=256, N=64, P=64 → 16 · 256 · 64 · 64 · 4 bytes (fp32 accum) = **67 MiB** per sample.

At MBS=8 → **~540 MiB** extra per rank. Production MBS=8 currently at ~118 GiB peak, so
+0.5 GiB is negligible.

### Performance analysis

Naive: Pass 1 + Pass 2 serially = 2 × kernel launch overhead (launch = ~10 µs). At
~400 launches per iter → +4 ms per iter (~0.05% total). Within noise.

Real gain: both passes have **lower reg count** → higher occupancy → **higher ILP**.
Typical 12.5% → 25% occupancy jump gives 1.3-1.8× throughput on compute-bound kernels.
Our bwd_bwd is compute-bound (AI=479 >> ridge 206 on H200). Expected 2110 ms →
1200-1600 ms = **500-900 ms saving**, matching P3 plan estimate.

Potential downside: `dstates_per_chunk` gmem round-trip adds memory bandwidth pressure.
H200 HBM3 at 4.8 TB/s, 67 MiB read+write per sample = ~28 µs per sample ≈ 0.2 ms at
MBS=8. Small vs the 500+ ms kernel saving.

## Integration plan (when we decide to implement)

1. Write Pass 1 as a new TileLang kernel `mamba_mimo_bwd_bwd_state_pass`.
2. Write Pass 2 as `mamba_mimo_bwd_bwd_chunk_pass`.
3. Python wrapper `mamba_mimo_bwd_bwd` launches both, allocates `dstates_per_chunk`
   gmem buffer, stitches gradients.
4. Correctness: diff vs single-kernel baseline, rel_err < 0.02 on all 14 gradients
   (same criterion as TMA layout fix).
5. Perf: nsys before/after, expect combined time 1200-1600 ms vs 2110 ms.

## Dependencies

- **Needs TMA layout fix landed first** (otherwise both pass kernels would also hit
  the 3D smem `InputDim != 2` bug since both handle the same `qk_dot_shared` tile).
- Either needs to reuse `apply_mamba3_mimo_tma_layout_fix.py` infrastructure or be
  written from scratch already 2D.

## Why not now

P3 is **~1 week of work** (new kernel + correctness + perf validation), estimated
final gain ~1% total TFLOP/s. P1 full (~5% potential) is cheaper and is still
in-flight. Ship P1 first, then re-evaluate P3 ROI at that point.

Better than P3 for total throughput (deferred further):
- P1 full (in flight, ≈5% if wins)
- `--fp8-param-gather` + custom-module fp8_model_init porting (likely +1-2% if MBS
  headroom opens at 10+)
- Algorithmic: reduce 255 regs via hoisting common subexpressions back to bwd_fwd
  (free if we cache post-rotary Q/K + PsiV per Task B of GB10 P2 investigation).

## Status

- Design doc: this file.
- Implementation: not started.
- Branch: would be `p3-bwd-bwd-register-split` once we have P1 full result to decide
  whether to pursue.

## Implementation blockers (2026-04-14, Phase-3 agent audit)

An attempt to start P3 uncovered three stop-the-work issues. The original design is
still **on paper only**; do not follow it as-is.

### Blocker 1 — split point is not clean (design assumption false)

The original design proposed splitting into Pass-1 (state-reverse) + Pass-2 (chunk-local).
Reading `mamba3_mimo_bwd.py::mamba_mimo_bwd_bwd_kernel` (lines 543-1170) line-by-line:

- Loop-carried state `dstates_frag` is **updated at the END of each reverse chunk** via
  `T.gemm(q_shared, dPhiO_scaled_frag, dstates_frag, clear_accum=False)` (line ~1151),
  then copied to `dstates_shared` for the **next** reverse iteration.
- This means Pass-1 (state-reverse path) must still hold `q_shared`, `dPhiO_shared`,
  and `dstates_frag` live — the exact fragments the design claimed could be dropped.
- The 10-fragment live-set attributed to "state-reverse" is actually shared between
  passes. Separating them would require (a) saving `dPhiO_shared` per-chunk to gmem
  (an extra `[B, H, nchunks, chunk_size·R, P]` buffer = 3×bigger than dstates buffer,
  ~200 MiB/sample), (b) re-deriving rotated-and-trap-scaled Q/K inside Pass-1 from
  scratch.

**Consequence**: the forecast "Pass-1 live-set ~140-150 regs" is too optimistic.
Realistic estimate after split is **~200-220 regs** — still over the 256 ceiling that
causes spilling, just slightly better. Occupancy bump is marginal.

This matches the prior research agent (Agent 6) verdict: **PARTIALLY VALID**,
**realistic gain 1.3-2.1% total TFLOP/s**, not 30-50% per-kernel as doc claims.

### Blocker 2 — GB10 is not a viable correctness platform for this kernel

Empirical test 2026-04-14: `mamba3_mimo` forward kernel fails to compile on GB10
(sm_121a, 99 KiB smem) at **every tested shape**, including tiny shapes:

```
B=1, S=128, H=4, G=1, N=32, P=32, R=2  →  TMA desc init error 716 (InternalError)
B=1, S=64,  H=16, G=1, N=64, P=64, R=4 →  autotune RuntimeError (0 configs compiled)
```

So even the single-kernel baseline does not run on GB10 at any shape we can reach.
Correctness validation for a P3 split — which would need **baseline vs split** output
comparison — is **not possible on GB10** without first fixing the baseline kernel's
GB10 compile path (separate unrelated work).

### Blocker 3 — bench3 SSH key is broken (as of 2026-04-14)

`ssh h200_1` returns `Permission denied (publickey)` for the
`google_compute_engine` identity. Until that is restored we have no H200 to
measure P3 perf on. Europe H200 is also gated (separate lane).

### Recommended path forward (deprecates the split-kernel plan)

1. **Skip the full split** — ROI is 1-2% at ≥1 week of kernel work plus a 540 MiB
   gmem buffer (`dstates_per_chunk`), not the 30-50% the doc sold.

2. **Try the Hoist-PsiV alternative (line 119 of this doc) first**: cache post-rotary
   Q/K + PsiV from `bwd_fwd` into gmem, consume them in `bwd_bwd`. This removes ~3 frag
   tiles from bwd_bwd's inner live-set without touching the reverse-scan. Cost is
   adding two gmem outputs to `bwd_fwd`; bwd_fwd already has 196 KiB smem + 255 regs so
   this also risks pushing it over, needs an actual test.
   - 2-3 days implementation vs 8-12 days for the full split.
   - Same expected ~1-2% total TFLOP/s gain, same risk profile.

3. **Preferred actual next step**: the cheaper unrealized wins on the plan list beat
   P3 comfortably:
   - MBS=12 with reduction-mean bugfix + Liger main-head (blocked on backward-NaN per
     `reference_main_head_liger_ce_gap.md`)
   - `--fp8-param-gather` for custom modules (+1-2% if MBS headroom opens, per
     `reference_fp8_param_gather_net_neutral.md`)
   - P1 full once TileLang ≥ **TMA lower with InputDim>2** supports bwd kernels

4. **If P3 is pursued anyway** (by a future agent with restored bench3 access):
   start with Pass-2 only (consume a **separately-produced** `dstates_per_chunk` from a
   modified `bwd_fwd`, not a new Pass-1 kernel). This avoids the loop-carried-state
   duplication issue in Blocker 1 entirely. Pass-2 alone at NAM56R shape also stays
   inside 228 KiB smem. The perf win will be smaller (~0.5-1%) but the code change is
   tractable.

### Files inspected

- `/home/dave/cppmega-venv/lib/python*/site-packages/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`
  (1365 lines; bwd_bwd_kernel lines 543-1170)
- `/Volumes/external/sources/cppmega/cppmega/megatron/upstream_patches/apply_mamba3_mimo_p1_patches.py`
  (patch infra referenced for future env-gated P3 patch)

### Empirical GB10 baseline compile failure

```
File ".../mamba3_mimo_fwd.py", line 462, in mamba_mimo_forward
  kernel( q, k, v, ...)
File ".../tilelang/jit/adapter/tvm_ffi.py", line 244, in func
  executable(*tensor_list)
tvm.error.InternalError: Failed to initialize the TMA descriptor 716
  TMA Desc Addr:   0xffffe26eb138
  format         7
  dim            3
  gmem_address   0x32ee1ea00
  globalDim      [128, 4, 1]
  globalStrides  [4, 512, 2048]
  boxDim         [16, 1, 1]
```

Same kernel at NAM56R small (B=1, S=64, H=16, N=64, P=64, R=4):
`RuntimeError: Auto-tuning failed: No configuration successfully compiled`.

### Decision

**Do not ship P3.** Pursue Hoist-PsiV + cheaper known wins. Close this design doc
as "superseded, not implemented". Keep the document as a historical record of why
the naive split was rejected.
