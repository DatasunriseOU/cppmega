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
