# Mamba3 Mono AB Prod Wave 4 - 2026-04-30

Status: active
Canonical: `docs/status/mamba3_cuda_bwd_bwd_10wave_summary_2026_04_30.md` for the production decision
Scope: Lane D production gate update and guarded stage2 training A/B prep.

Worktree: `/home/dave/source/cppmega/.claude/worktrees/mamba3-mono-ab-prod`

Branch: `worker/mamba3-mono-ab-prod`

## Added

- `docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json`
- `docs/status/mamba3_mono_ab_prod_wave4_2026_04_30.md`
- `guarded_stage2_training_ab_stub()` in `cppmega/megatron/mamba3_mono_ab_schema.py`
- `--print-training-ab-stub` in `scripts/modal_mamba3_cuda_full_bwd_ab.py`
- receipt-gate and Modal hygiene fail-mode tests

No production defaults changed. The guarded stage2 path remains default-off and
requires the explicit mutation gates from main commit `bc8c3f9`.

## Ingested Receipts

Cumulative Wave3/4 receipt file:
`docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json`

| receipt id | source | shape | measured ms | gate read |
| --- | --- | --- | ---: | --- |
| `wave3_cuda_wmma_triangular_chunk_owner` | `worker/mamba3-mono-cuda-chunk` Wave3 | productionish | `8.467136` | negative; current WMMA is `2.284x` the full TileLang `bwd_bwd` |
| `wave3_cute_handwritten_wgmma_wrong_numerics` | `worker/mamba3-mono-cute-chunk` Wave3 | `64x64x64` GEMM | n/a | failed correctness, `max_abs=17.318359`; no production credit |
| `wave3_wgmma_plan_budget` | `worker/mamba3-mono-triton-model` Wave3 | productionish | n/a | design budget only |
| `wave4_rr_diag_cuda_timestep_cta` | existing R x R Wave4 doc | productionish | `2.0560` | useful partial same-time slice, not a boundary replacement |

WGMMA plan budgets now preserved in the normalized component record:

| budget | ms |
| --- | ---: |
| green full-kernel target | `<= 3.35` |
| yellow full-kernel target | `<= 3.70674` |
| chunk-owner main body | `<= 3.20` |
| chunk-owner `DMIMO_V` reducer | `<= 0.05` |
| scan-owner main body | `<= 3.30` |

## Gate Update

H200 productionish gate stays unchanged:

| item | ms | source |
| --- | ---: | --- |
| TileLang guarded stage2 full `bwd_bwd` reference | `3.70674` | `mamba3_cuda_bwd_bwd_10wave_summary_2026_04_30.md` |
| current best covered CUDA subset | `2.48042` | same summary |
| remaining budget for all missing work | `1.22632` | `3.70674 - 2.48042` |

The current WMMA receipt is still outside the gate:

- `8.467136 - 3.70674 = 4.760396 ms` slower than the full TileLang reference.
- `8.467136 - 1.22632 = 7.240816 ms` over the remaining missing-work budget.
- It covers only `dv`, `dmimo_v`, and `dssda`, so even a faster timing would
  still need full boundary integration.

CuTe/Hopper remains a research lane, not a production path. The package stack
and quack-kernels path are viable, but the hand-written CuTe WGMMA kernel still
has wrong numerics, so it cannot enter production A/B.

Wave4 R x R CUDA improves the standalone diagonal microbench to `2.0560 ms`,
but it is still a partial same-time slice for `dk`, `dq`, and `dgamma_diag`.
It does not change the production decision.

## Training A/B Prep

The A/B harness can now print a guarded stage2 training launcher receipt:

```text
timeout 300s modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
  --print-training-ab-stub \
  --run-id guarded_stage2_train_ab_wave4_20260430 \
  --no-modal-hygiene
```

The emitted stub is intentionally operationally conservative:

1. Roll back stage2 before the baseline leg.
2. Run the baseline training command.
3. Apply stage2 only with `CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1` and
   `MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1`.
4. Run the stage2 `(bf=1,bb=0)` candidate leg.
5. Roll back stage2 again.

The stub records the fields the training A/B must compare: loss curve/final
loss, tokens/sec, max allocated/reserved memory, and failures/restarts. It is a
launcher/checklist only; it does not enable stage2 by default.

## Validation

Local checks:

```text
PYTHONPATH=. pytest -q \
  tests/test_mamba3_mono_ab_schema.py \
  tests/test_mamba3_mono_ab_modal_hygiene.py
# 14 passed

python -m json.tool \
  docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json

PYTHONPATH=. python -m py_compile \
  cppmega/megatron/mamba3_mono_ab_schema.py \
  scripts/modal_mamba3_cuda_full_bwd_ab.py
```

Modal hygiene:

```text
modal app list --json
modal app stop --yes ap-5Fwtxk17W3MosOpy66Ah9B
```

The first list showed all previously listed `cppmega-mamba3-*` and
`cppmega-mamba3-cuda-full-bwd-ab` apps stopped. A final post-commit check found
one zero-task same-campaign ephemeral app:

- `ap-5Fwtxk17W3MosOpy66Ah9B` -
  `cppmega-mamba3-mono-chunk-wave2-h200 ephemeral`

It was stopped from this lane. The final check reported:

```text
active_total=0 active_cppmega_mamba3=0
```

## Readiness

Production/readiness state after Wave4:

- Ready for production movement: guarded TileLang stage2 `(bf=1,bb=0)` behind
  explicit gates from `bc8c3f9`.
- Not ready: CUDA monolithic replacement, because it still lacks full boundary
  parity, integrated memory proof, state/off-time/scalar outputs, and training
  A/B.
- Not ready: hand-written CuTe WGMMA, because current numerics are wrong.
- Useful research direction: WGMMA full-kernel plan only if it meets the
  preserved green/yellow budgets and all full-boundary gates.
