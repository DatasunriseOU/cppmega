# Production Status (2026-04-14)

Single source of truth for current production configs on 8xH200. All other
docs (`README.md`, `plan.md`, `docs/findings_2026_04_14_session.md`) reference
this file rather than restating throughput numbers.

Last validated: 2026-04-14 (session 3 corrections post Liger grad-corruption
audit). Liger workaround = `reduction="mean"` broadcast (sidesteps Liger #968
silent gradient corruption).

---

## Bench3 (LOCATION_1) — FP8 tensorwise

- **Config**: PP=1 TP=1 EP=8 DP=1, MBS=10 GBS=80, seq_len=4096, MTP_DEPTHS=2
- **Precision**: FP8 tensorwise (global `--fp8-format hybrid`)
- **Variant**: `VARIANT=v3` in `scripts/remote_smoke_h200_dsa_9_4_m.sh`
- **Throughput**: **268 TFLOP/s ± 0.5** (27.1% MFU), steady-state iters 3-15
- **Liger main-head path**: `reduction="mean"` broadcast (correct gradients,
  Liger #968 workaround via `cppmega/megatron/apply_linear_ce_patch.py`)
- **MTP**: Liger chunked CE (`CPPMEGA_LIGER_CE=1`, known correct)
- **Index cache**: `CPPMEGA_INDEX_CACHE=1` (67% DSA indexer savings)
- **lemyx fused DSA**: `CPPMEGA_LEMYX_DSA=1`
- **DSA indexer fused**: `CPPMEGA_DSA_INDEXER_FUSED=1` (per-head bmm)
- **CUDA graphs**: OFF (`CG_FLAGS=NONE`; required at PP=1)
- **Peak memory**: ~115 GiB / 141 GiB per rank (was measured at session 3
  golden snapshot; subsequent re-verify on 2026-04-14 evening showed
  130.5 GiB without `CG_FLAGS=NONE` — see warning below)
- **WARNING — `CG_FLAGS=NONE` is mandatory for MBS=10**: the script default
  is `--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba
  moe_router moe_preprocess`, which holds a 63.5 GiB CUDA Graph private pool.
  At MBS=10 this pushes peak past 140 GiB and OOMs at iter 1. Always pass
  `CG_FLAGS=NONE` explicitly in env. Verified on bench3 2026-04-14.
- **WARNING — `CPPMEGA_DSA_INDEXER_FUSED` default flipped to OFF**: the
  per-head streamed indexer creates a 640 MiB fp32 buffer per DSA layer and
  9 DSA layers × 640 MiB = ~5.7 GiB resident across forward (autograd holds
  for backward). Adds memory pressure on top of the CG private pool. To
  enable explicitly set `CPPMEGA_DSA_INDEXER_FUSED=1` (production override).
- **Do NOT enable**: `CPPMEGA_MTP_NATIVE_HOPPER_CE=1` — produces
  `grad_norm=NaN`, Suspects #1+#2 both empirically refuted on bench3
  (2026-04-14). Suspects #3-5 (shared-weight dual-bwd, mask handling,
  dtype) under investigation

## Europe (LOCATION_2) — BF16

- **Config**: PP=1 TP=1 EP=4 DP=2, MBS=8 GBS=64, seq_len=4096, MTP_DEPTHS=2
- **Precision**: BF16 (FP8 tensorwise regresses -34% on this fabric)
- **Variant**: `VARIANT=v1` in `scripts/remote_smoke_h200_dsa_9_4_m.sh`
- **Throughput**: **289 TFLOP/s** (29.2% MFU), gold record
- **Liger main-head path**: same as bench3 — `reduction="mean"` broadcast
- **MTP**: Liger chunked CE (same as bench3)
- **Index cache / lemyx / fused indexer**: same env gates as bench3
- **CUDA graphs**: OFF (`CG_FLAGS=NONE`; required at PP=1)
- **Peak memory**: ~127 GiB / 141 GiB per rank (MBS=10 OOMs)
- **FP8 on europe**: every FP8 variant regresses on this fabric — keep BF16

---

## Launching production

```bash
# europe (289 TFLOP/s, BF16, MBS=8 EP=4)
PP_SIZE=1 VPP_SIZE=1 MBS=8 EP_SIZE_OVERRIDE=4 \
CG_FLAGS=NONE \
CPPMEGA_INDEX_CACHE=1 \
CPPMEGA_LEMYX_DSA=1 \
CPPMEGA_LIGER_CE=1 \
CPPMEGA_MAIN_HEAD_LINEAR_CE=1 \
CPPMEGA_LINEAR_CE_KERNEL=liger \
CPPMEGA_DSA_INDEXER_FUSED=1 \
bash scripts/remote_smoke_h200_dsa_9_4_m.sh

# bench3 (268 TFLOP/s, FP8 tensorwise, MBS=10 EP=8)
VARIANT=v3 PP_SIZE=1 VPP_SIZE=1 MBS=10 EP_SIZE_OVERRIDE=8 \
CG_FLAGS=NONE \
CPPMEGA_FP8=1 \
CPPMEGA_INDEX_CACHE=1 \
CPPMEGA_LEMYX_DSA=1 \
CPPMEGA_LIGER_CE=1 \
CPPMEGA_MAIN_HEAD_LINEAR_CE=1 \
CPPMEGA_LINEAR_CE_KERNEL=liger \
CPPMEGA_DSA_INDEXER_FUSED=1 \
bash scripts/remote_smoke_h200_dsa_9_4_m.sh
```

---

## Deprecated measurements (do not cite)

| Measurement | Status | Reason |
|---|---|---|
| bench3 269.4 TFLOP/s (Liger reduction="none") | **SUPERSEDED** | Silent gradient corruption via Liger #968; 268 with `reduction="mean"` broadcast is the new canonical |
| bench3 253 TFLOP/s (PP=1 EP=4 MBS=8 FP8) | superseded | Topology replaced by EP=8 MBS=10 (v3) |
| europe 193 TFLOP/s (PP=2 VPP=2 MBS=4) | superseded | Stream L topology, kept as default in launcher only for backward compat |
| "205 TFLOP/s DualPipeV baseline" | never real | See `project_dualpipev_unwired.md` |
| bench3 "100 TFLOP/s gap vs europe" | refuted | Was PP=2 bubble artifact; correct gap is 13% HW variance |

---

## Gap to stretch goal

- europe 289 TFLOP/s = 29.2% MFU vs 50% target → **1.7x gap remaining**
- Empirically confirmed: no config/topology tuning closes this gap on
  this architecture + H200 hardware (15+ empirical tests, 2026-04-14)
- Near-term +5-10% path: Mamba3 P2 PsiV cache, P3 register split (weeks)
- 50% MFU requires CUTLASS WGMMA rewrites (months) or B200+ hardware

---

## References

- `reference_golden_config_2026_04_13.md` — superseded by this file (see
  "europe regression" + "bench3 FP8 is memory win" entries for context)
- `reference_fp8_mbs10_bench3_wins.md` — bench3 FP8 empirical data
- `reference_fp8_mbs10_europe_regression.md` — europe FP8 regression data
- `reference_main_head_liger_ce_gap.md` — `apply_linear_ce_patch.py` details
- `reference_ep8_v3_machine_specific.md` — per-machine EP sizing rationale
