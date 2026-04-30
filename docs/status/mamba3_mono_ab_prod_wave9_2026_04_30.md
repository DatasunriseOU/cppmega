# Mamba3 Mono AB Prod Wave 9 - 2026-04-30

Status: complete
Canonical production decision: `docs/status/mamba3_cuda_bwd_bwd_10wave_summary_2026_04_30.md`
Scope: Lane D production gate and receipt update for Wave8 CUDA, CuTe, and
WGMMA copy-lane evidence.

## Gate Change

The production receipt gate now requires explicit full-boundary integrated
timing. A receipt with only component or local timing earns zero production
credit even if it reports every output slot and full-boundary correctness.

Fail-closed behavior is now:

- no full-boundary integrated timing marker: `non_integrated_timing_receipt`
- missing any of the 12 `mamba_mimo_bwd_bwd` output slots:
  `missing_required_output_slots`
- no full-boundary correctness report:
  `full_boundary_correctness_not_reported`
- missing resource/CTA/H200/Modal/budget evidence: zero production credit

## Wave8 Receipts

The cumulative receipt file
`docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json`
now ingests:

| receipt id | source | timing/evidence | read |
| --- | --- | ---: | --- |
| `wave8_cuda_tile_stream_wmma_subset` | `worker/mamba3-mono-cuda-chunk` commit `e8a67b5` | `11.180607795715332 ms` H200 productionish | correct subset, rejected |
| `wave8_cute_multichunk_state_apply_consumers` | `worker/mamba3-mono-cute-chunk` commit `130b935` | `76-78 us/scan` for 2/4/8 chunks | promising research only |
| `wave8_wgmma_copy_path_12tile_uint4` | `worker/mamba3-mono-triton-model` commit `b0aff52` | 12-tile `uint4`, ptxas `40` regs, `0` spills, `98304 B` dynamic smem | copy evidence only |

CUDA tile-stream useful signal:

- `DV`, final `DMIMO_V`, and `DSSDA` subset correctness passes.
- H200 ptxas/resource shape is reasonable: `72` regs/thread, `50692 B`
  dynamic smem, `3` active blocks/SM.
- It fixes the Wave7 scalar cliff, but the subset is still `3.016x` slower
  than the full TileLang stage2 `bwd_bwd` reference.

CuTe multi-chunk useful signal:

- 2/4/8 chunk fused scan passes local `1e-5` tolerance.
- `LKQ`, state, apply, and `dpsi` stay off global memory in the tested path.
- Still bounded/local: no full boundary, no dA/diag/qk, no scalar output set,
  scalar copies remain.

Copy-lane useful signal:

- all 12 logical tiles have retained alignment/layout evidence;
- `K_T` and `Q_T` are admitted only as GMMA physical-layout operands;
- the compile probe retains ptxas metadata for the narrow `uint4` path.

## Current Branch Ranking

Ranking is by production readiness, not isolated component speed.

| rank | branch/path | current stance |
| ---: | --- | --- |
| 1 | guarded TileLang stage2 `(bf=1, bb=0)` | only production movement; exact and repeatedly faster on H200 chain by about `1.6-1.9%` |
| 2 | CUDA covered subset / warp-owner path | best R&D economics, `2.48042 ms` covered subset versus `3.70674 ms` TileLang, but missing full boundary, memory, and training A/B |
| 3 | CuTe multi-chunk fused state/apply consumers | best materialization direction; zero production credit until full-boundary integrated timing with all slots |
| 4 | WGMMA copy path `uint4` 12-tile evidence | implementation enabler only; no output slots or timing credit |
| 5 | CUDA tile-stream WMMA subset | correct subset and good resource shape, but `11.18 ms` is slower than TileLang full `bwd_bwd` |
| 6 | CUDA row-stream low-live-set | rejected scalar recompute dead end despite `2` blocks/SM |
| 7 | older CuTe micro-GEMM / WMMA / scan-owner receipts | retained as research evidence only |

## Production Stance

Unchanged. Guarded TileLang stage2 `(bf=1, bb=0)` remains the only shippable
movement. CUDA/CuTe replacement work stays research until a candidate covers
all 12 real `mamba_mimo_bwd_bwd` output slots, reports full-boundary integrated
timing/resource/CTA/memory evidence, has clean Modal hygiene, and beats the
H200 `3.70674 ms` TileLang budget.

## Validation

Local checks for this gate update:

```text
python -m json.tool docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json
PYTHONPATH=. pytest -q tests/test_mamba3_mono_ab_schema.py tests/test_mamba3_mono_ab_modal_hygiene.py
PYTHONPATH=. python -m py_compile cppmega/megatron/mamba3_mono_ab_schema.py scripts/modal_mamba3_cuda_full_bwd_ab.py
git diff --check
```
