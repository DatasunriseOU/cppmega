# Mamba3 Mono AB Prod Wave 3 - 2026-04-30

Worktree: `/home/dave/source/cppmega/.claude/worktrees/mamba3-mono-ab-prod`

Branch: `worker/mamba3-mono-ab-prod`

Scope: Lane D receipt ingestion and production gate report for the monolithic
`mamba_mimo_bwd_bwd` campaign. This wave did not implement a kernel. It used
the Wave2 component receipt gate to ingest component timings and produce the
numbers future agents must hit.

## Added

- `docs/status/mamba3_mono_ab_component_receipts_wave2_wave3_2026_04_30.json`
- `docs/status/mamba3_mono_ab_prod_wave3_2026_04_30.md`
- `tests/test_mamba3_mono_ab_modal_hygiene.py`
- receipt coverage assertions in `tests/test_mamba3_mono_ab_schema.py`
- dry-run example in `scripts/modal_mamba3_cuda_full_bwd_ab.py` now uses
  `--modal-hygiene-enforcement fail`

## Ingested Receipts

Receipt file:
`docs/status/mamba3_mono_ab_component_receipts_wave2_wave3_2026_04_30.json`

| receipt id | source | shape | measured ms | covered slots | status |
| --- | --- | --- | ---: | --- | --- |
| `wave2_cuda_wmma_state_lkq_d` | `worker/mamba3-mono-cuda-chunk`, Wave2 | productionish | `8.919168` | `dv`, `dmimo_v`, `dssda` | negative |
| `wave2_triton_pruned_lower_bound` | `worker/mamba3-mono-triton-model`, Wave2 | productionish | `8.79331` | none, checksum only | negative lower bound |
| `wave2_wmma_fallback_smoke` | `worker/mamba3-mono-cute-chunk`, Wave2 | custom `B=1,S=64,H=4,P=64` smoke | `0.155613` | smoke-only partial `dk/dq/dv/dmimo_v` | smoke only |
| `wave3_rr_diag_timestep_cta` | existing `mamba3_rr_diag_microkernel_wave3` doc | productionish | `2.6777` | `dk`, `dq`, `dgamma_diag` same-time slice | partial microbench |

I did not find newer same-campaign `mamba3_mono_*wave3*` status docs in the
mono sibling worktrees at scan time. The only Wave3 component doc already
present in this tree was the R x R diagonal microbench, so it is included as a
partial-slot receipt.

## Canonical Gate

H200 productionish gate values:

| item | ms | source |
| --- | ---: | --- |
| TileLang guarded stage2 full `bwd_bwd` reference | `3.70674` | `docs/status/mamba3_cuda_bwd_bwd_10wave_summary_2026_04_30.md` |
| current best covered CUDA subset | `2.48042` | same summary |
| remaining budget for all missing work | `1.22632` | `3.70674 - 2.48042` |

This means:

- A full integrated replacement must be `<= 3.70674 ms` and pass all 12 output
  slots at the real `mamba_mimo_bwd_bwd` boundary.
- Any component that is added on top of the current covered CUDA subset must
  leave the total at `<= 3.70674 ms`, so the remaining missing-work bundle has
  only `1.22632 ms`.
- A component that replaces an existing covered-slice incumbent must beat the
  incumbent slice, not merely fit under the full `bwd_bwd` number.

## Component Targets

| component lane | current receipt | must hit | gap |
| --- | ---: | ---: | ---: |
| WMMA state/LKQ/D chunk-owner bundle | `8.919168 ms` | `<= 1.22632 ms` if it fills the missing-work bundle | `7.692848 ms` too slow, needs `7.27x` speedup |
| Triton tile-pruned owner model | `8.79331 ms` | `< 3.70674 ms` before output stores just to stay plausible; `<= 1.22632 ms` if used as missing-work bundle | `5.08657 ms` over full `bwd_bwd`; `7.56699 ms` over remaining budget |
| WMMA fallback smoke | `0.155613 ms` on custom small smoke | no production credit yet; next receipt must use standard smoke and productionish | not comparable |
| R x R diagonal timestep CTA | `2.6777 ms` | `<= 1.57673 ms` if kept as a separate diag slice beside qk/dV `0.36735 ms` and qk-`DMIMO_V` `0.53634 ms` | `1.10097 ms` too slow for same-slot replacement |
| full monolithic candidate | not run | `<= 3.70674 ms`, all 12 slots, memory `<=` guarded stage2, no Modal leak | pending |

The gate for the next agents is therefore concrete:

1. If you own the missing state/off-time/scalar bundle, your total contribution
   must be `<= 1.22632 ms` on H200 productionish, including any extra launch
   cost and output writes.
2. If you own a same-slot replacement, state the incumbent you are replacing:
   `wave7 diag+qk/dV` is `1.92990 ms`, qk/dV alone is `0.36735 ms`, and
   qk-`DMIMO_V` sidecar is `0.53634 ms`. Beat that number and preserve the
   full-slot contract.
3. If you claim full replacement, do not use component credit: run the A/B
   harness at the real call boundary and beat `3.70674 ms` while passing every
   slot: `dk`, `dv`, `dmimo_v`, `dq`, `dfactor`, `dgamma_diag`, `dangles`,
   `dd`, `dda`, `dssda`, `dda_cs_rev`, `dda_cs`.
4. Smoke-only receipts do not consume productionish budget. They only prove the
   path is worth a standard-shape receipt.

## Validation

Local tests:

```text
PYTHONPATH=. pytest -q \
  tests/test_mamba3_mono_ab_schema.py \
  tests/test_mamba3_mono_ab_modal_hygiene.py
# 11 passed
```

Syntax and fixture checks:

```text
python -m json.tool \
  docs/status/mamba3_mono_ab_component_receipts_wave2_wave3_2026_04_30.json >/dev/null

PYTHONPATH=. python -m py_compile \
  cppmega/megatron/mamba3_mono_ab_schema.py \
  scripts/modal_mamba3_cuda_full_bwd_ab.py

git diff --check
```

All passed.

Modal schema dry-run command used for ingestion:

```text
timeout 300s modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
  --dry-run-schema \
  --run-id mono_ab_wave3_gate_dry_run_20260430_r3 \
  --shape-csv productionish \
  --monolithic-candidate-csv mono_wave3_target \
  --candidate-record-path-csv docs/status/mamba3_mono_ab_component_receipts_wave2_wave3_2026_04_30.json \
  --modal-hygiene-enforcement fail
```

The dry-run ingested the receipt file and produced the expected component
projections. It exited non-zero because fail-mode hygiene found active
same-campaign Modal apps from sibling lanes. That is the intended gate behavior.

Observed active apps stopped from this lane:

- `ap-NL8CdazzMOazWpuy8wlVOh` -
  `cppmega-mamba3-mono-cuda-chunk-wave3-h200`
- `ap-IRqCp9ih4lwne1QjLwb2UG` -
  `cppmega-mamba3-mono-chunk-wave2-h200`
- `ap-MhSJWMZqBAbSt3J4jmT1JH` -
  `cppmega-mamba3-mono-chunk-wave2-h200`
- `ap-KZ6798Yf7aAebYXzEi5gO6` -
  `cppmega-mamba3-mono-chunk-wave2-h200`
- `ap-qVjr0Tx7zzWIG4ePKptu6K` -
  `cppmega-mamba3-mono-cuda-chunk-wave3-h200`
- `ap-LdsDSizc1grfxzLUUc4UcN` -
  `cppmega-mamba3-mono-cuda-chunk-wave3-h200`

Final delayed `modal app list --json` check showed active count `0`.

## Readout

Wave2 component receipts are all outside the production gate:

- CUDA WMMA moved the dead scalar path in the right direction, but
  `8.919168 ms` is already `2.406x` slower than the entire TileLang `bwd_bwd`.
- Triton tile pruning is also negative: `8.79331 ms` before output stores or
  full slot coverage.
- WMMA fallback smoke is useful only as a correctness/dataflow receipt.
- The R x R Wave3 microkernel remains useful evidence for local math, but it
  does not beat the current covered-subset allocation as a separate component.

Remaining budget for all missing production work is `1.22632 ms`.
