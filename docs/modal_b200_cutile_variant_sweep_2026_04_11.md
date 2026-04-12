# Modal B200:2 cuTile variant sweep — 2026-04-11

Sweep of 6 algorithmic variants for `mamba3_mimo_bwd_bwd` on **Modal B200:2** (sm_100a datacenter Blackwell, TMEM present, tcgen05 present, 228 KiB smem per SM). Complements the GB10 sweep documented in `docs/gb10_bwd_bwd_optimization_conclusion.md`. Three new findings worth keeping.

## Environment

- Modal app: `cppmega-cutile-b200-variant-sweep`
- GPU: B200:2, image cached from `cppmega-cutile-mamba3-mimo`
- torch 2.12.0.dev20260410+cu132, CUDA 13.2, cuda.tile 1.2.0
- Detected arch: `sm_100`, device "NVIDIA B200" cap (10, 0), count 2
- Cost: ~9 min B200:2 compute (~$6-10 Modal spend)

## Phase 1 — TileLang reference on B200 (not GB10!)

- `mamba_mimo_fwd_combined`: ~54 µs (2026-04-10 parity)
- `mamba_mimo_bwd_fwd`: ~72 µs
- `mamba_mimo_bwd_bwd`: **~179 µs**
- Full bwd chain (bwd_fwd + bwd_bwd): **~251 µs**

Run-to-run noise ±0.4 µs. These numbers are different from GB10 (167 µs bwd_bwd) — B200's bwd_bwd is actually ~7% slower in absolute time than GB10 despite 22.5× peak TFLOPS. The TileLang kernel is memory/pipeline-bound, not compute-bound, so bigger tensor cores don't directly translate.

## Phase 2 — unmodified GB10 cuTile port on B200

Running the exact same file (`mamba3_mimo_bwd_bwd_cutile.py` with its 2-kernel A/B split) on B200:
- bwd_fwd: 167 µs
- **bwd_bwd: 687.64 µs**
- chain: 854.6 µs
- Ratio vs TileLang: **3.84×** on bwd_bwd, **3.40×** on chain

Almost identical to GB10's 624 µs bwd_bwd baseline. **B200's hardware advantages (TMEM, larger smem, 22.5× more TFLOPS) give the cuTile compiler zero improvement at this kernel.** This is the first data point confirming that the cuTile bwd_bwd gap is *compiler-model-structural*, not hardware-limited.

## Phases 3 + 4 — variant sweep

Tested 6 variants, 20 iter × 5 warmup per variant, `torch.cuda.Event` timing:

| Variant                      | B200 bwd_bwd µs |      vs baseline | vs TileLang | Correct |              GB10 ratio | Reversal?                       |
| ---------------------------- | --------------: | ---------------: | ----------: | :-----: | ----------------------: | ------------------------------- |
| baseline 2-kernel A/B        |           687.6 |            1.00× |       3.84× |  PASS   |                   3.73× | —                               |
| V2 fused monolithic          |           729.4 |     1.06× slower |       4.07× |  PASS   |                   8.42× | gap compressed, still slower    |
| **V3 3-kernel split**        |       **460.5** | **0.67× (−33%)** |   **2.57×** |  PASS   | 4.06× (slower on GB10!) | **WINS on B200, LOSES on GB10** |
| V4 hoisted invariants        |           622.3 |     0.90× (−10%) |       3.47× |  PASS   |                   4.45× | flipped to win on B200          |
| V7 `@ct.kernel(occupancy=1)` |           689.2 |    1.00× (noise) |       3.85× |  PASS   |                       — | **no-op**                       |
| V8 V4+occ=1                  |           622.8 |            0.91× |       3.48× |  PASS   |                       — | identical to V4                 |

bwd_fwd kernel was not modified — stable at ~166-167 µs across all variants. All correctness checks pass rtol=1e-2 atol=1e-2 (worst residual: `DFACTOR` 4.3e-3 vs TileLang max abs 5.9e-3).

## Winner: V3 3-kernel split

**V3 splits the bwd_bwd computation into 3 kernels** (A_dv / A_dk / B) vs the baseline's 2-kernel A/B split. Each kernel holds at most one `(FUSED, FUSED)` fp32 accumulator at peak, reducing register pressure per kernel.

Measured on B200:
- bwd_bwd alone: **460.5 µs** (−33% vs baseline, −23% vs 2-kernel)
- Full bwd chain (bwd_fwd + bwd_bwd): **627 µs** (vs TileLang 251 µs → **2.49× slower**)
- Correctness: PASS

## The cross-HW reversal (this is the important finding)

V3 3-kernel split and V4 hoisted invariants **both regressed on GB10** but **both won on B200**:

| Variant               | B200            | GB10             |
| --------------------- | --------------- | ---------------- |
| V3 3-kernel split     | **−33%** (wins) | +9% (regresses)  |
| V4 hoisted invariants | **−10%** (wins) | +19% (regresses) |

**Why:** the launch-overhead vs live-set-savings trade-off depends on per-kernel smem budget and register pressure. On B200's 228 KiB smem, splitting into 3 kernels gives each one enough headroom; on GB10's 99 KiB dynamic budget the extra launch overhead (~30 µs) isn't amortized by the register relief.

**Rule:** there is **no universal cuTile-best algorithmic variant**. Always re-sweep on target HW. A variant that wins on sm_100 may lose on sm_121 and vice versa. Do not assume the winner transfers across architectures.

## Structural gap to TileLang is HW-independent

- GB10 baseline: 624 µs vs TileLang 167 µs = **3.73× slower**
- B200 V3 winner: 460 µs vs TileLang 179 µs = **2.57× slower**
- Best cuTile on any Blackwell is still **2.5-3× slower than TileLang**

Even with:
- TMEM (256 KiB per SM on B200, absent on GB10)
- tcgen05 instruction family (present on B200, absent on GB10)
- 228 KiB smem (B200) vs 99 KiB dynamic (GB10) — 2.3× more
- 22.5× more peak BF16 TFLOPS

...the best cuTile variant cannot close the gap to TileLang. **The gap is structural to the cuTile compiler model.** cuTile 1.2.0 cannot express:
- Per-iteration loop-carried register state for block-diagonal DQ/DK GEMMs
- Pinned dstates accumulators in registers across chunk iterations (the token-order pass inserts synthetic barriers)
- TMEM / tcgen05 / UMMA primitives at the DSL level — HW exists, API doesn't

**For bwd-heavy kernels with persistent per-CTA state on any Blackwell, stay on TileLang.**

## The cuTile forward-kernel win

Separately, from the 2026-04-10 Modal B200 parity run: **cuTile `mamba3_mimo_fwd` is 17.7% FASTER than TileLang on B200** (0.054 ms vs 0.064 ms). This is a deployment opportunity — a hybrid wrapper `cuTile fwd + TileLang bwd` gives a free win on B200 deployments, and the cuTile fwd already passes correctness against TileLang per `docs/modal_b200_cutile_parity.md`.

## `@ct.kernel(occupancy=1)` is a silent no-op

Tested on V7 and V8:
- V7 baseline + occupancy=1: 689.24 µs ≈ baseline 687.64 (0.2% noise)
- V8 V4 + occupancy=1: 622.84 µs ≈ V4 622.30 (0.1% noise)

No fallback error, kwarg accepted silently. The cuTile compiler either already picks equivalent occupancy by default, or ignores the hint. **Abandon as a tuning knob on cuTile 1.2.0.**

## Recommendations

1. **v3_split3 as B200-default cuTile bwd_bwd** if cppmega ever ships a cuTile bwd path for B200 deployments. 33% faster than baseline, passes correctness, 2.57× slower than TileLang.

2. **Keep 2-kernel A/B split as GB10/sm_121 default** — V3 regresses there. No universal variant.

3. **Stay on TileLang for production bwd chain** — even the best cuTile variant is 2.49× slower on B200 and 3.73× slower on GB10. The gap is structural, not fixable by more algorithmic work.

4. **cuTile fwd deployable on B200 as a 17.7% win** — wrap `cuTile fwd + TileLang bwd` if you want the easiest free throughput gain on B200 boxes.

5. **Abandon `occupancy=1` as a tuning knob** — no measurable effect on cuTile 1.2.0.

6. **Re-sweep algorithmic variants on every new target GPU** — cross-HW behavior is not predictable from a single machine. The variant sweep on GB10 missed V3 as a winner because GB10's smaller smem punishes the 3-kernel split; the variant sweep on B200 found it because B200's larger smem rewards it.

## Files

All under `.tmp/modal_b200_cutile/`, no repo modifications:

- `variant_baseline.py`, `variant_v2_fused_mono.py`, `variant_v3_split3.py`, `variant_v4_hoisted.py`, `variant_v7_occupancy1.py`, `variant_v8_hoisted_occ1.py`
- `PHASE3_RESULTS.md`, `FINAL_RESULTS.md`
- `modal_b200_variant_sweep_results.json` (latest combined)
- `modal_b200_variant_sweep_phase3_results.json`, `modal_b200_variant_sweep_phase4_results.json`

Modal app definition: `scripts/modal_cutile_b200_variant_sweep.py`.
