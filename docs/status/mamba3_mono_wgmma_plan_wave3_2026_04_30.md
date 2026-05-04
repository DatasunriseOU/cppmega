# Mamba3 Mono WGMMA Plan Wave3 - 2026-04-30

Status: design receipt for Lane D
Canonical: no
Branch: `worker/mamba3-mono-triton-model`

## Scope

Wave2 answered the Triton performance question: the tile-pruned Triton owner
measured `8.79331 ms` on H200 and is not a viable runtime path. Its value is
the ledger: reuse, causal pruning, output writes, and tile validity.

This document converts that ledger into a concrete CuTe/Hopper WGMMA design
receipt for Lane D. No long Modal run was needed in this wave.

Receipt JSON:

- `docs/status/mamba3_mono_wgmma_plan_wave3_receipt_2026_04_30.json`
- generated/checkable by `tools/probes/mamba3_wgmma_wave3_receipt.py`

## Production Shape

Target shape inherited from Wave1/2:

| field | value |
| --- | ---: |
| `B` | `4` |
| `S` | `4096` |
| `H` | `32` |
| `G` | `1` |
| `N` | `64` |
| `P` | `128` |
| `R` | `4` |
| `chunk` | `16` |
| `nchunks` | `256` |
| fused rows `chunk*R` | `64` |
| chunk bodies `B*H*nchunks` | `32768` |

Reference timings:

| path | H200 mean ms | read |
| --- | ---: | --- |
| TileLang `stage2_bf1_bb0` full `bwd_bwd` | `3.70674` | A/B baseline |
| Wave1 full-mask Triton checksum | `4.53881` | compute lower bound, no full stores |
| Wave2 tile-pruned Triton checksum | `8.79331` | negative runtime result |
| Wave8 CUDA before state/LKQ/D | `2.45093` | useful prior component cost only |

## FMA Receipt

Wave2 model counters for the production shape:

| model | FMA |
| --- | ---: |
| separate recompute | `125.37B` |
| monolithic full-mask reuse | `114.63B` |
| Wave2 4-step tile-pruned | `101.75B` |
| ideal triangular applies | `96.38B` |
| optional scan-owner `dstates += Q.T @ dPhiO` update | `17.18B` |
| scan-owner ideal plus `dstates` update | `113.56B` |

Ideal triangular component ledger:

| component | FMA |
| --- | ---: |
| `state_dpsi = K @ dstates` | `17.18B` |
| `LKQ = K @ Q.T` once | `8.59B` |
| causal `LKQ @ dPhiO` | `8.05B` |
| `qk_dot -> dPsiV` once | `1.07B` |
| `DV` and `DMIMO_V` R reductions | `0.54B` |
| `dk_state = PsiV @ dstates.T` | `17.18B` |
| `dk_intra = PsiV @ dPhiO.T` once | `17.18B` |
| causal `dk_intra @ Q` | `4.03B` |
| `dq_state = dPhiO @ states.T` | `17.18B` |
| causal `dk_intra.T @ K` | `4.03B` |
| scalar/reduction elementwise work | `1.35B` |

The A-gate rule for Lane D: if `LKQ`, `dk_intra`, or `dk_intra.T` is
recomputed by another GMMA instead of being reused, the candidate fails the
receipt even if an early timing sample looks good.

## Output Bytes

| output group | bytes | MiB |
| --- | ---: | ---: |
| `DV[B,S,H,P]` bf16 | `134217728` | `128.00` |
| `DK[B,S*R,H,N]` bf16 | `268435456` | `256.00` |
| `DQ[B,S*R,H,N]` bf16 | `268435456` | `256.00` |
| `DMIMO_V[B,H,R,P]` fp32 | `262144` | `0.25` |
| scalar outputs including `DSSDA` and `DANGLES` | `77594624` | `74.00` |
| scan-owner required writes | `748945408` | `714.25` |
| chunk-owner required writes including `DMIMO_V` partials | `816054272` | `778.25` |
| chunk-owner final `DMIMO_V` reducer extra R/W | `134479872` | `128.25` |

Avoided global temporaries remain the same Wave1 memory win: `dPsiV`, `LKQ`,
`dk_intra`, state `dPsiV`, `DK`, and `DQ` intermediates account for about
`3.5 GiB` if materialized.

## CTA Ownership

Preferred Lane D owner: one CTA owns one `(B,H)` stream and iterates the `256`
chunks in reverse order. This matches the actual `bwd_bwd` state recurrence,
keeps `dstates` local to the CTA, and lets `DMIMO_V[B,H,R,P]` accumulate
locally and write once. For the production shape that is `128` CTAs, each with
`256` chunk iterations. This is close to H200 SM count and avoids the 64 MiB
`DMIMO_V` partial tensor.

Fallback owner: one CTA owns one `(B,H,chunk)` body, only if Lane D receives a
precomputed per-chunk `dstates` handoff from an upstream component. This maps
most directly to the Wave1/2 cost model and creates `32768` CTAs, but it must
write `DMIMO_V[B,H,nchunks,R,P]` partials plus a final reducer.

Do not mix the owners silently. The receipt must say whether `dstates` is an
input or loop-carried local state; the FMA and byte budgets differ by
`17.18B` FMA and `64.0 MiB` of partial output.

## GMMA Shapes And K/P Tiling

Use the existing CuTe DSL P4 direction as the implementation base:

- SM90 BF16 GMMA atom: `m64n64k16 -> fp32`.
- `P=128` is two resident `n64` panels, not one long-lived `n128`
  accumulator. The two-panel design keeps `dPsiV`, `DV`, `DMIMO_V`, and
  `dstates` lifetimes separable.
- `N=64`, `fcs=64`, and `P_panel=64` make every dense product a `64x64`
  output tile with four or eight `k16` GMMA groups.
- Pre-transposed global inputs are required for legal K-major SMEM operands:
  `Q_T[N,fcs]`, `K_T[N,fcs]`, and `dPhiO_T[P,fcs]`.
- `LKQ` and `dk_intra` are computed as fp32 GMMA accumulators, used for
  scalar products while fp32, then downcast to bf16 SMEM only for apply GMMA.

Tensor-core products per chunk body:

| product | shape | K groups | notes |
| --- | --- | ---: | --- |
| `K @ dstates_panel` | `64x64` | `4` | run once per P panel |
| `K @ Q.T` | `64x64` | `4` | build `LKQ` once |
| `LKQ @ dPhiO_panel` | `64x64` | pruned | causal apply, see schedule |
| `PsiV @ dstates.T` | `64x64` | `8` | `P=128` streamed as two panels |
| `PsiV @ dPhiO.T` | `64x64` | `8` | build `dk_intra` once |
| `dk_intra @ Q` | `64x64` | pruned | causal apply |
| `dPhiO @ states.T` | `64x64` | `8` | `P=128` streamed as two panels |
| `dk_intra.T @ K` | `64x64` | pruned | uses transpose SMEM view/copy, no second GMMA build |
| optional `Q.T @ dPhiO_panel` | `64x64` | `4` | only for scan owner, updates loop-carried `dstates` |

## Shared Memory Plan

Resident SMEM target: keep the hot window near `112-128 KiB` and below the
H200 opt-in limit. Do not keep full `P=128` copies of every operand live.

| buffer | shape | bytes | lifetime |
| --- | --- | ---: | --- |
| `sK`, `sQ`, `sQ_T`, `sK_T` | each `64x64` bf16 | `8192` each | chunk-wide |
| `sDStatePanel`, `sStatePanel` | each `64x64` bf16 | `8192` each | P panel |
| `sDPhPanel`, `sPsiPanel` | each `64x64` bf16 | `8192` each | P panel stream |
| `sLKQ` | `64x64` bf16 | `8192` | after `LKQ` build through `dPsiV` and `DSSDA` |
| `sDKI` | `64x64` bf16 | `8192` | after `dk_intra` build through `DK/DQ` |
| `sDKI_T` | `64x64` bf16 view/copy | `8192` | transpose feed for `DQ`, no duplicate GMMA |
| `sOut` | `64x64` bf16 | `8192` | reused for C-store staging |
| qk/scalar scratch | small fp32/bf16 | `<4096` | vector slices, no TMA |
| scan-owner `DMIMO` accumulator | `R x P` fp32 | `2048` | CTA-wide, write once |

The P4 prototype aliases `sOut` over dead `sK`. Keep that pattern. The extra
Lane D pressure comes from holding both `sLKQ` and `sDKI` while streaming
P-panels; avoid allocating duplicate full-output staging buffers.

## Triangular Tile Schedule

Full `LKQ` and full `dk_intra` are still computed because `DSSDA` needs the
unmasked products. Causal consumers must not consume below-frontier tiles.

Let each timestep have four fused lanes. The ideal valid apply set is:

```text
for row_time in 0..15:
  for col_time in row_time+1..15:
    apply the 4x4 lane block
```

That is `1920` fused entries per chunk. The Wave2 4-step schedule groups four
timesteps into one `16x16` FCS tile:

```text
for row_block in 0..3:
  for col_block in row_block..3:
    if col_block > row_block:
      apply the full 16x16 tile
    else:
      split the diagonal block into 4x4 timestep subtiles
```

Lane D should implement the diagonal split above if it claims the `96.38B`
ideal-triangular receipt. If the diagonal block is instead internally masked
as a dense `16x16` tile, the valid receipt is Wave2's `101.75B` tile-pruned
budget, not the ideal budget.

Important lowering constraint: dense Hopper WGMMA is rectangular. It cannot
magically skip arbitrary 4x4 lower-triangle entries inside one `m64n64k16`
operation. The honest implementation choices are:

1. Use dense WGMMA for the full producers, then use a CuTe sub-MMA or
   scalar/warp helper for the pruned off-diagonal and diagonal apply tiles.
2. Accept the 4-step tile-pruned FMA ledger and report `101.75B`.
3. Fall back to full-mask applies and report `114.63B`.

Any timing report must identify which of those three schedules produced it.

## Output Staging

For the preferred scan owner:

1. Keep `DMIMO_V[R,P]` as a CTA-local fp32 accumulator and write
   `DMIMO_V[B,H,R,P]` once after the reverse chunk loop.
2. Store `DV` once per chunk/P-panel after bf16 handoff from `dPsiV_D`.
3. Store `DK` and `DQ` after trap scaling, inverse rotary, and qk-diagonal
   additions; output dtype is bf16.
4. Store scalar outputs (`DDA_CS`, `DDA_CS_REV`, `DFACTOR`, `DGAMMA_DIAG`,
   `DSSDA`, `DDA`, `DANGLES`) with normal coalesced copies. Keep the prior
   force-nonTMA lesson: do not use TMA for tiny vector slices.

For the fallback chunk owner:

1. Write `DMIMO_V` partials `[B,H,nchunks,R,P]` fp32.
2. Run the Wave8-style all-R output-owner reducer. Budget is `<=0.05 ms`;
   anything materially above that means the partial layout is wrong.

## A/B Gate Budget

| gate | target |
| --- | ---: |
| green full-kernel target | `<=3.35 ms` |
| yellow full-kernel target | `>3.35 ms` and `<=3.70674 ms` |
| red | `>3.70674 ms` |
| chunk-owner main body | `<=3.20 ms` |
| chunk-owner `DMIMO_V` reducer | `<=0.05 ms` |
| scan-owner main body | `<=3.30 ms` |

Green is about 10% faster than TileLang and is the only budget worth treating
as a performance path. Yellow is a useful implementation milestone but not
enough margin to ship.

A-gate requirements before timing:

- output-wise correctness for all listed outputs on smoke;
- no global `dPsiV`, `LKQ`, `dk_intra`, `DK`, or `DQ` temps;
- no duplicate `LKQ`, `dk_intra`, or `dk_intra.T` GMMA builds;
- receipt reports one of full-mask, 4-step tile-pruned, or ideal triangular
  apply schedules;
- resource metadata includes registers/thread, dynamic SMEM, static SMEM, and
  active CTAs/SM.

## Lane D Component Inputs

The JSON receipt contains the machine-readable inputs Lane D should consume:

- modeled FMA totals and component FMA ledger;
- required output bytes for scan-owner and chunk-owner variants;
- required output names and dtypes;
- CTA owner, GMMA atom, P/K tiling, and triangular schedule string;
- A/B timing budgets.

Local check:

```text
python tools/probes/mamba3_wgmma_wave3_receipt.py \
  --check docs/status/mamba3_mono_wgmma_plan_wave3_receipt_2026_04_30.json
```

## Read

The viable Lane D path is not "optimize Triton." It is a CuTe/Hopper owner
that keeps the Wave1 reuse ledger, reports exactly how much triangular work it
actually prunes, and avoids materializing the `3.5 GiB` intermediate set.

Preferred implementation order:

1. Build the scan-owner CuTe skeleton around the existing P4 layout pattern and
   verify all output stores.
2. Land full-mask WGMMA correctness first, but mark it as `114.63B` and do not
   call it a performance path.
3. Add 4-step tile pruning and measure whether CuTe avoids Wave2's scheduler
   collapse.
4. Split diagonal 4x4 subtiles only if the 4-step path is close enough to the
   green budget.
