# Mamba3 Mono AB Prod Wave 7 - 2026-04-30

Status: complete
Canonical production decision: `docs/status/mamba3_cuda_bwd_bwd_10wave_summary_2026_04_30.md`
Scope: Lane D production A/B gate and receipt update for the Wave6 CuTe fused
masked-LKQ apply receipt.

## Gate Change

The component receipt gate now rejects local component speedups explicitly with
`non_integrated_timing_receipt`. A local tile/component timing, even if it is
faster than a scalar local path, earns no production credit unless it is a
full-slot integrated `mamba_mimo_bwd_bwd` timing receipt.

The Modal hygiene helper also reports `reused_description_warnings` when Modal
app descriptions are reused within the configured Mamba3 campaign prefix. Reused
descriptions are a warning rather than a production-credit gate, but they must
be fixed in future runs because they make cleanup and receipt attribution
ambiguous.

Fail-closed behavior is unchanged: any local-only timing, missing output slots,
subset-only correctness, missing/incomplete resource metadata, missing CTA fill,
dirty Modal hygiene, missing H200 tag, or missed timing budget keeps
`production_credit=false`, `production_credit_ms=0.0`, and
`credited_output_slots=[]`.

## Wave6 CuTe Receipt

The cumulative receipt file
`docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json`
now ingests:

| receipt id | source | local timing | read |
| --- | --- | ---: | --- |
| `wave6_cute_fused_masked_lkq_apply_tile_chain` | `worker/mamba3-mono-cute-chunk` commit `f163df8` | scalar `104.710 us/chain` to fused `63.550 us/chain` | research progress only; zero production credit |

Useful progress:

- The tested fused path removes the global LKQ output and consumes masked LKQ
  from swizzled shared memory inside the CuTe tile chain.
- H200 correctness stayed within the Wave6 local tolerance (`1e-5`).
- The local chain speed improved by about `1.65x`.

Production blockers:

- It writes only the BF16 `apply` tile, not the full `bwd_bwd` output slots.
- `state` and `apply` remain global tiles, while `dpsi`, `DV`, and `DMIMO_V`
  are still torch-side/scalar consumers.
- Copies remain scalar BF16 universal G2S/S2G.
- There is no integrated full-boundary resource, CTA, memory, or performance
  receipt.

## Lane A Context

Wave6 CUDA chunk-group ownership remains rejected. It recovered CTA count but
timed around `14.5 ms` on H200 productionish, added `DMIMO_V` scratch plus a
reduction launch, and still lacked DK/DQ plus the scalar output set. It remains
research context only.

## Production Stance

Unchanged. Guarded TileLang stage2 `(bf=1, bb=0)` remains the production
movement. CuTe/CUDA monolithic work remains research until a candidate covers
all 12 real `mamba_mimo_bwd_bwd` output slots at the integrated boundary,
reports resource/CTA/memory evidence, has clean Modal hygiene with unique app
descriptions, and beats the H200 `3.70674 ms` TileLang budget.

## Validation

Local checks for this gate update:

```text
python -m json.tool docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json
PYTHONPATH=. pytest -q tests/test_mamba3_mono_ab_schema.py tests/test_mamba3_mono_ab_modal_hygiene.py
PYTHONPATH=. python -m py_compile cppmega/megatron/mamba3_mono_ab_schema.py scripts/modal_mamba3_cuda_full_bwd_ab.py
git diff --check
```
