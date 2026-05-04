# Mamba3 Mono AB Prod Wave 6 - 2026-04-30

Status: complete
Canonical production decision: `docs/status/mamba3_cuda_bwd_bwd_10wave_summary_2026_04_30.md`
Scope: Lane D production A/B gate update for the Wave5 scan-owner CUDA receipt.

## Gate Change

The component receipt gate now reports CTA fill explicitly under
`production_gate.cta_count_occupancy`.

The gate requires production candidates to report total CTA count and to launch
at least one CTA per H200 SM. For H200 productionish receipts the default SM
count is `132`, unless the receipt reports a different `h200_sm_count`.
Theoretical occupancy remains a required resource metadata dimension.

Fail-closed behavior is unchanged: any missing output slots, subset-only
correctness, missing/incomplete resource metadata, underfilled CTA count, dirty
Modal hygiene, missing H200 tag, or missed timing budget keeps
`production_credit=false`, `production_credit_ms=0.0`, and
`credited_output_slots=[]`.

## Wave5 Scan-Owner Receipt

The cumulative receipt file
`docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json`
now ingests:

| receipt id | source | shape | measured ms | covered slots | CTA read |
| --- | --- | --- | ---: | --- | --- |
| `wave5_cuda_scan_owner_dv_dmimov_dssda` | `worker/mamba3-mono-cuda-chunk` commit `a3bb497` | productionish | `14.08131217956543` | `dv`, `dmimo_v`, `dssda` | `128` total CTAs for `132` H200 SMs, underfilled |

Subset correctness is useful research progress:

- vs bf16 staged torch reference: `DV=4.76837158203125e-07`,
  `DMIMO_V=2.561137080192566e-09`, `DSSDA=2.6645352591003757e-15`
- resources: `190` regs/thread, `68612 B` dynamic smem,
  `1` active block/SM, `12.5%` theoretical occupancy

The production gate rejects it because it is subset-only, slower than the full
TileLang stage2 `bwd_bwd` budget, underfills the H200 grid, and does not cover
`dk`, `dq`, or the scalar output set.

## Production Stance

Unchanged. No production algorithm changed in this wave. Guarded TileLang
stage2 `(bf=1, bb=0)` remains the production movement; CUDA scan-owner work
stays research until it covers the full `mamba_mimo_bwd_bwd` boundary with
enough CTA parallelism, integrated memory proof, and H200 timing below the
TileLang budget.

## Validation

Local checks for this gate update:

```text
python -m json.tool docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json
PYTHONPATH=. pytest -q tests/test_mamba3_mono_ab_schema.py tests/test_mamba3_mono_ab_modal_hygiene.py
PYTHONPATH=. python -m py_compile cppmega/megatron/mamba3_mono_ab_schema.py scripts/modal_mamba3_cuda_full_bwd_ab.py
git diff --check
```
