# Mamba3 Mono AB Prod Wave 8 - 2026-04-30

Status: complete
Canonical production decision: `docs/status/mamba3_cuda_bwd_bwd_10wave_summary_2026_04_30.md`
Scope: Lane D production gate and receipt update for the Wave7 CUDA row-stream
and Wave7 CuTe fused-consumer receipts.

## Gate Change

No production algorithm changed in this wave.

The gate behavior is unchanged and remains fail-closed. A low-live-set receipt
does not earn production credit if its integrated or component timing misses
the H200 TileLang stage2 `bwd_bwd` budget. A local one-chunk CuTe timing does
not earn production credit without full-boundary integrated timing, even when
the local chain is faster and numerically correct.

## Wave7 Receipts

The cumulative receipt file
`docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json`
now ingests:

| receipt id | source | timing | read |
| --- | --- | ---: | --- |
| `wave7_cuda_row_stream_low_live_set` | `worker/mamba3-mono-cuda-chunk` commit `a73439f` | `179.76535034179688 ms` H200 productionish | reject as scalar recompute dead end |
| `wave7_cute_fused_state_apply_consumers_one_chunk` | `worker/mamba3-mono-cute-chunk` commit `2a1467a` | `63.419 us/chain` H200 one-chunk | promising research, zero production credit |

CUDA row-stream useful signal:

- `2` active H200 blocks/SM, `125` regs/thread, `42244 B` dynamic smem.
- `32768` owner CTAs, so grid fill is no longer the blocker.
- Subset correctness passes for `DV`, final `DMIMO_V`, and `DSSDA`.

CUDA row-stream production blocker:

- Runtime regressed to `179.76535034179688 ms`, `48.5x` slower than the
  `3.70674 ms` TileLang stage2 `bwd_bwd` budget.
- It still covers only `dv`, `dmimo_v`, and `dssda`, not the full
  `mamba_mimo_bwd_bwd` boundary.
- The schedule proves low live-set is possible, but scalar row recompute is
  not a production direction.

CuTe fused-consumer useful signal:

- The tested one-chunk path removes global `LKQ`, `state`, `apply`, and
  `dpsi` materialization.
- H200 local correctness passes within `1e-5`.
- Local chain improves to `63.419 us/chain`.

CuTe fused-consumer production blocker:

- It is one fixed `64x64x64` chunk, not the full boundary.
- It still uses scalar BF16 copy/consumer mechanics.
- There is no integrated full-slot timing, resource, CTA, memory, or training
  A/B receipt.

## Current Branch Ranking

Ranking is by production readiness, not by isolated local speed.

| rank | branch/path | current stance |
| ---: | --- | --- |
| 1 | guarded TileLang stage2 `(bf=1, bb=0)` | only production movement; exact and repeatedly faster on H200 chain by about `1.6-1.9%` |
| 2 | CUDA covered subset / warp-owner path | best R&D economics, `2.48042 ms` covered subset versus `3.70674 ms` TileLang, but missing full boundary, memory, and training A/B |
| 3 | CuTe fused one-chunk consumers | promising materialization direction; zero production credit until lifted to full-boundary integrated timing |
| 4 | CUDA row-stream low-live-set | reject as scalar recompute dead end despite `2` blocks/SM |
| 5 | older CuTe micro-GEMM / WMMA / scan-owner receipts | retained as evidence only; no production credit |

Production stance remains unchanged: guarded TileLang stage2 `(bf=1, bb=0)` is
the only shippable movement. CuTe and CUDA replacement work stays research
until a candidate covers all 12 real `mamba_mimo_bwd_bwd` output slots, reports
integrated resource/CTA/memory evidence, has clean Modal hygiene, and beats the
H200 `3.70674 ms` TileLang budget.

## Validation

Local checks for this gate update:

```text
python -m json.tool docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json
PYTHONPATH=. pytest -q tests/test_mamba3_mono_ab_schema.py tests/test_mamba3_mono_ab_modal_hygiene.py
PYTHONPATH=. python -m py_compile cppmega/megatron/mamba3_mono_ab_schema.py scripts/modal_mamba3_cuda_full_bwd_ab.py
git diff --check
```
