# Mamba3 Mono AB Prod Wave 5 - 2026-04-30

Status: complete
Canonical production decision: `docs/status/mamba3_cuda_bwd_bwd_10wave_summary_2026_04_30.md`
Scope: Lane D production A/B gate update for Wave4 CuTe correctness receipts.

## Gate Change

Component receipts now carry a computed `production_gate` verdict. The verdict
does not trust receipt status text for production credit. A receipt earns
production credit only when all of these are true:

1. It is not a micro-GEMM-only receipt.
2. It covers every real `mamba_mimo_bwd_bwd` output slot:
   `dk`, `dv`, `dmimo_v`, `dq`, `dfactor`, `dgamma_diag`, `dangles`,
   `dd`, `dda`, `dssda`, `dda_cs_rev`, and `dda_cs`.
3. It reports full-boundary correctness, not component correctness.
4. It reports resource metadata: registers/thread, shared-memory bytes,
   active blocks/SM, and theoretical occupancy.
5. It carries an H200 hardware tag before consuming the H200 production budget.
   H100 and B200 tags remain optional portability evidence tags.
6. Its Modal hygiene receipt is clean.
7. Its integrated `bwd_bwd` timing meets the `3.70674 ms` TileLang stage2
   productionish budget.

Failing any item sets `production_credit=false`, `production_credit_ms=0.0`,
and `credited_output_slots=[]`.

## Wave4 CuTe Receipt

The cumulative receipt file
`docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json`
now ingests:

| receipt id | evidence | read |
| --- | --- | --- |
| `wave4_cute_handwritten_wgmma_micro_gemm_correct` | H200 CuTe DSL `64x64x64` BF16 WGMMA, exact deterministic/random cases, `28.254240 us/iter` | candidate receipt only; zero production credit |

This supersedes the Wave3 wrong-numerics CuTe micro-GEMM receipt as a
correctness signal, but it still does not cover any real Mamba3 output slot.
It also lacks integrated Mamba3 boundary resource metadata and performance, so
it must not enter production A/B as a replacement candidate.

## Production Stance

Unchanged: guarded TileLang stage2 `(bf=1, bb=0)` remains the only reasonable
production movement. CuTe/CUDA monolithic work remains research until it covers
all output slots at the real `mamba_mimo_bwd_bwd` boundary, reports integrated
resources/memory, and beats the H200 TileLang budget.

## Validation

Local checks for this gate update:

```text
python -m json.tool docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json
PYTHONPATH=. pytest -q tests/test_mamba3_mono_ab_schema.py tests/test_mamba3_mono_ab_modal_hygiene.py
PYTHONPATH=. python -m py_compile cppmega/megatron/mamba3_mono_ab_schema.py scripts/modal_mamba3_cuda_full_bwd_ab.py
```
