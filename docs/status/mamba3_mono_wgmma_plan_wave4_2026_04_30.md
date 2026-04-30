# Mamba3 Mono WGMMA Plan Wave4 - 2026-04-30

Status: executable schedule skeleton / receipt validator for Lane A/B
Canonical: no
Branch: `worker/mamba3-mono-triton-model`

## Scope

Wave3 picked the viable owner: one CTA per `(B,H)` stream, reverse over 256
chunks, keep `dstates` and `DMIMO_V[R,P]` local, use SM90 BF16
`m64n64k16 -> fp32` GMMA, split `P=128` as two `n64` panels, build `LKQ`
once, build `dk_intra` once, and feed `dk_intra.T` from an SMEM view/copy.

Wave4 makes that plan executable for Lane A/B with a CPU-only generator and
validator:

- `tools/probes/mamba3_wgmma_wave4_schedule.py`
- `docs/status/mamba3_mono_wgmma_plan_wave4_receipt_2026_04_30.json`
- local check:

```text
python tools/probes/mamba3_wgmma_wave4_schedule.py \
  --check docs/status/mamba3_mono_wgmma_plan_wave4_receipt_2026_04_30.json
```

No long Modal run was needed for this wave. This is a static schedule and
resource receipt, not a measured kernel.

## Production Shape

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
| P panels | `2 x 64` |

Preferred owner:

| item | value |
| --- | ---: |
| CTA grid | `B*H = 128` |
| chunk iterations per CTA | `256` |
| chunk order | reverse, `255 downto 0` |
| local state | `dstates[N,P]`, loop-carried |
| local reduction | `DMIMO_V[R,P]`, final write only |

Fallback owner remains `32768` CTAs, one per `(B,H,chunk)`, only if an
upstream handoff provides correct per-chunk `dstates`. It adds a 64.0 MiB
`DMIMO_V` partial tensor plus 128.25 MiB reducer R/W and is not the preferred
Wave4 path.

## Executable Skeleton

The generator can emit the implementation skeleton:

```text
python tools/probes/mamba3_wgmma_wave4_schedule.py --format skeleton
```

The skeleton order is:

1. Map `blockIdx.x` to one `(b,h)` stream.
2. Initialize CTA-local `dstates[N,P]` and `DMIMO_V[R,P]`.
3. Loop chunks in reverse.
4. Load/copy `K`, `Q`, transpose views, and P-panel operands into SMEM.
5. Build `LKQ = K @ Q.T` once.
6. For each P panel, compute state `dPsiV`, causal `LKQ @ dPhiO`, same-time
   qk contribution, `DV`, and local `DMIMO_V`.
7. Build `dk_state`, build `dk_intra = PsiV @ dPhiO.T` once, emit scalar
   products, then write `DK` and `DQ`.
8. Feed `DQ` from a transpose SMEM view/copy of `dk_intra`; do not rebuild it.
9. Store scalar outputs and update CTA-local `dstates += Q.T @ dPhiO`.
10. After the reverse loop, store final `DMIMO_V[B,H,R,P]`.

## GMMA Counts

Counts are per chunk body unless noted. Dense full-mask values are actual
`m64n64k16` instruction counts. Causal-pruned values are
`m64n64k16`-equivalent useful work because the ideal diagonal split cannot be
claimed by issuing one dense rectangular GMMA over masked lower-triangle data.

| component | dense full-mask GMMA | Wave2 4-step equiv | ideal triangular equiv |
| --- | ---: | ---: | ---: |
| `state_dpsi = K @ dstates_panel`, two P panels | `8` | `8.0` | `8.0` |
| `LKQ = K @ Q.T` once | `4` | `4.0` | `4.0` |
| causal `LKQ @ dPhiO`, two P panels | `8` | `5.0` | `3.75` |
| `dk_state = PsiV @ dstates.T` | `8` | `8.0` | `8.0` |
| `dk_intra = PsiV @ dPhiO.T` once | `8` | `8.0` | `8.0` |
| causal `dk_intra @ Q` | `4` | `2.5` | `1.875` |
| `dq_state = dPhiO @ states.T` | `8` | `8.0` | `8.0` |
| causal `dk_intra.T @ K` | `4` | `2.5` | `1.875` |
| scan-owner `dstates += Q.T @ dPhiO` | `8` | `8.0` | `8.0` |

Totals:

| schedule | per chunk | per CTA stream | grid total |
| --- | ---: | ---: | ---: |
| full-mask dense, excluding scan update | `52` | `13312` | `1703936` |
| full-mask dense, with scan update | `60` | `15360` | `1966080` |
| Wave2 4-step equiv, excluding scan update | `46.0` | `11776.0` | `1507328.0` |
| Wave2 4-step equiv, with scan update | `54.0` | `13824.0` | `1769472.0` |
| ideal triangular equiv, excluding scan update | `43.5` | `11136.0` | `1425408.0` |
| ideal triangular equiv, with scan update | `51.5` | `13184.0` | `1687552.0` |

FMA totals from the receipt:

| schedule | FMA |
| --- | ---: |
| separate recompute | `125.37B` |
| monolithic full-mask | `114.63B` |
| Wave2 4-step tile-pruned | `101.75B` |
| ideal triangular apply | `96.38B` |
| scan-owner `dstates` update | `17.18B` |
| scan-owner ideal plus `dstates` update | `113.56B` |

## SMEM And Registers

The planned resident SMEM is below the Wave3 `112-128 KiB` target:

| item | bytes | KiB |
| --- | ---: | ---: |
| logical unique buffers | `102400` | `100.0` |
| aliased peak (`sOut`/`sK`, `sDKI_T` view or alias) | `86016` | `84.0` |
| peak with 16 KiB alignment guard | `118784` | `116.0` |
| pass budget | `131072` | `128.0` |
| kill budget | `163840` | `160.0` |

Key buffers are `sK`, `sQ`, `sQ_T`, `sK_T`, state/dstate panels,
`sDPhPanel`, `sPsiPanel`, `sLKQ`, `sDKI`, `sDKI_T`, `sOut`, and a small
qk/scalar scratch region. `sOut` must alias dead operand storage where legal;
`sDKI_T` should be a view or an aliased copy, not another independent long
lifetime 64x64 tile.

Register estimate:

| item | regs/thread |
| --- | ---: |
| accumulator tile `64x64 fp32` per warpgroup | `32` |
| panelized `dstates` | `32` |
| full local `dstates[N,P]` | `64` |
| distributed `DMIMO_V[R,P]` | `4` |
| estimated panelized path | `148` |
| estimated full-local-dstates path | `180` |
| danger path with third live accumulator | `212` |
| pass budget | `192` |
| kill budget | `224` |

Lane A/B must report actual ptxas registers/thread, static SMEM, dynamic SMEM,
and spills. Any local-memory spill in the hot CTA is a kill even if the static
estimate is under budget.

## Output And Bytes/FMA

Preferred scan-owner output writes:

| item | value |
| --- | ---: |
| per chunk excluding final `DMIMO_V` | `22848 B` |
| per CTA stream including final `DMIMO_V` | `5851136 B` |
| full scan-owner output writes | `748945408 B` / `714.25 MiB` |
| fallback chunk-owner output writes | `816054272 B` / `778.25 MiB` |
| fallback reducer extra R/W | `134479872 B` / `128.25 MiB` |

Bytes/FMA from required output writes:

| schedule | bytes/FMA |
| --- | ---: |
| scan-owner ideal plus `dstates` update | `0.006595349782` |
| scan-owner Wave2 4-step plus `dstates` update | `0.006297612330` |
| scan-owner full-mask plus `dstates` update | `0.005681998982` |
| chunk-owner ideal, no scan update | `0.008467338324` |

The lower full-mask bytes/FMA is not a win; it is more compute for the same
required writes. The performance receipt is still the ideal triangular path
only if the implementation proves the diagonal 4x4 causal split or an
equivalent no-work-lower-triangle mechanism.

## Pass/Kill Criteria

Pass:

- all output slots compare on smoke: `DV`, `DK`, `DQ`, `DMIMO_V`,
  `DDA_CS`, `DDA_CS_REV`, `DFACTOR`, `DGAMMA_DIAG`, `DSSDA`, `DDA`,
  and `DANGLES`;
- preferred owner is exactly one CTA per `(B,H)` stream with reverse chunks;
- `LKQ` and `dk_intra` are each built once per chunk;
- `dk_intra.T` uses SMEM view/copy, not another GMMA build;
- no global `dPsiV`, `LKQ`, `dk_intra`, `DK`, `DQ`, or `DMIMO_V` partial
  temp is materialized;
- dynamic SMEM is `<=128 KiB` including guard, registers/thread `<=192`, and
  no local-memory spills;
- productionish H200 timing is green at `<=3.35 ms`, or at least yellow at
  `<=3.70674 ms`.

Kill:

- any duplicate `LKQ`, `dk_intra`, or `dk_intra.T` GMMA build;
- any `[B,H,nchunks,R,P]` `DMIMO_V` partial tensor in the preferred scan-owner
  path;
- any silent owner mix where `dstates` is both loop-carried local state and a
  precomputed chunk input;
- dynamic SMEM `>160 KiB`, registers/thread `>224`, or local-memory spills;
- productionish H200 full-kernel timing `>3.70674 ms` after correctness;
- claiming the `96.38B` / `113.56B` ideal FMA receipt without diagonal 4x4
  causal split or equivalent proof.

## Lane A/B Next

Lane A should implement the CuTe/CUDA skeleton in this order:

1. Land full-mask correctness with the preferred scan owner and all output
   slots wired, but report it as the `60` GMMA/chunk scan-owner full-mask
   skeleton, not as a performance path.
2. Add the 4-step causal apply schedule and prove `LKQ`/`dk_intra` are still
   single-build.
3. Add the 4x4 diagonal split only after the 4-step path is close enough to
   justify the extra scheduler complexity.

Lane B should wire the receipt validator into the gate:

1. Run the JSON `--check` command before accepting schedule changes.
2. Add compiled resource metadata: ptxas registers/thread, static SMEM,
   dynamic SMEM, active CTAs/SM, and spill count.
3. Reject any implementation whose component receipt does not match the
   single-build `LKQ`, single-build `dk_intra`, SMEM-transpose `dk_intra.T`,
   local `dstates`, and final-only `DMIMO_V` contract.
