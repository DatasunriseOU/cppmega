# Mamba3 Mono AB Prod Wave 1 - 2026-04-30

Worktree: `/home/dave/source/cppmega/.claude/worktrees/mamba3-mono-ab-prod`

Branch: `worker/mamba3-mono-ab-prod`

Scope: Lane D production A/B and integration harness for a future monolithic
`mamba_mimo_bwd_bwd` replacement. This wave did not implement a monolithic
kernel. It made the A/B harness able to receive one.

## Implemented

- Brought main's guarded stage2 production-control commit into this worktree:
  `bc8c3f9` / `cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches`.
- Added `cppmega/megatron/mamba3_mono_ab_schema.py`.
- Extended `scripts/modal_mamba3_cuda_full_bwd_ab.py` with:
  - `schema_version = mamba3-mono-ab/v1`;
  - common shape/config/candidate report schema;
  - `main_guarded_stage2`, `cuda_covered_subset_wave9`, and future
    monolithic candidate rows;
  - full `mamba_mimo_bwd_bwd` output-slot schema;
  - per-shape memory accounting;
  - `--monolithic-candidate-csv`;
  - `--dry-run-schema`;
  - Modal app list checks and exact-name, zero-task auto-stop after runs.
- Added `tests/test_mamba3_mono_ab_schema.py`.

## Candidate Rows

| candidate | role | status in wave1 |
| --- | --- | --- |
| `main_guarded_stage2` | production reference | maps to main `bc8c3f9`, TileLang `stage2_bf1_bb0`, `bf_num_stages=1`, `bb_num_stages=0` |
| `cuda_covered_subset_wave9` | prior component timing floor | partial only; covers same-time slices for `DK/DQ/DV/DMIMO_V/DGAMMA_DIAG` |
| `monolithic_chunk_candidate` or names from `--monolithic-candidate-csv` | future full-boundary candidate | schema slot only until a real implementation is wired into the call boundary |

## Boundary Output Slots

The monolithic candidate report reserves correctness slots for every
`mamba_mimo_bwd_bwd` output:

| slot | dtype | canonical shape |
| --- | --- | --- |
| `dk` | bf16 | `[B, S*R, H, N]` |
| `dv` | bf16 | `[B, S, H, P]` |
| `dmimo_v` | fp32 | `[B, H, R, P]` |
| `dq` | bf16 | `[B, S*R, H, N]` |
| `dfactor` | fp32 | `[B, H, S]` |
| `dgamma_diag` | fp32 | `[B, H, S]` |
| `dangles` | fp32 | `[B, S, H, N/rotary_dim_divisor]` |
| `dd` | fp32 | `[B, H]` |
| `dda` | fp32 | `[B, H, S]` |
| `dssda` | fp32 | `[B, H, nchunks, chunk, chunk]` |
| `dda_cs_rev` | fp32 | `[B, H, S]` |
| `dda_cs` | fp32 | `[B, H, S]` |

The report also carries the `bwd_fwd` handoff slots `dmimo_o`, `states`, and
`qk_dot` so full-chain correctness remains inspectable.

## Memory Accounting

Analytical schema numbers:

| shape | handoff cache MiB | bwd_bwd output MiB | estimated live floor MiB |
| --- | ---: | ---: | ---: |
| `smoke` | 0.535156 | 1.273453 | 2.400421 |
| `productionish` | 528.250000 | 714.250488 | 1553.688110 |

The live floor is only shape accounting. Production A/B must use measured
`torch.cuda.max_memory_allocated/reserved` in the real autograd lifetime.

## Validation

Local checks passed:

```text
python -m py_compile \
  cppmega/megatron/mamba3_mono_ab_schema.py \
  scripts/modal_mamba3_cuda_full_bwd_ab.py \
  cppmega/megatron/upstream_patches/apply_mamba3_stage2_force_nontma_patches.py

pytest -q tests/test_mamba3_mono_ab_schema.py
# 5 passed

PYTHONPATH=. python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches
# SKIP CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA is not set
```

Modal dry-run:

```text
modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
  --dry-run-schema \
  --shape-csv smoke \
  --monolithic-candidate-csv mono_chunk_wave1 \
  --no-modal-auto-stop
```

Receipt:

- app: `ap-zgpK9as87enh3a8Tn3HmkY`
- result: completed and stopped
- remote GPU work: none; this was schema dry-run only

At the hygiene check, the harness's own app was stopped. A separate
same-campaign H200 app, `cppmega-mamba3-mono-cuda-chunk-wave1-h200`, had
`Tasks=1`, so it was left untouched as not safe to stop from this lane.

H200 smoke:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 timeout 900s \
  modal run scripts/modal_mamba3_cuda_full_bwd_ab.py \
  --run-id mono_ab_wave1_h200_smoke_20260430_1 \
  --shape-csv smoke \
  --monolithic-candidate-csv mono_chunk_wave1 \
  --iters 1 \
  --warmup 0 \
  --cuda-iters 1 \
  --cuda-warmup 0
```

Receipt:

- app: `ap-rVqWD3QRe4dNC0UCCfYddz`
- result: completed and stopped
- device: `NVIDIA H200`, torch `2.13.0.dev20260426+cu132`, CUDA `13.2`
- artifacts:
  - `/benchmarks/mamba3_cuda_full_bwd_ab/mono_ab_wave1_h200_smoke_20260430_1/report.json`
  - `/benchmarks/mamba3_cuda_full_bwd_ab/mono_ab_wave1_h200_smoke_20260430_1/summary.json`
  - `/benchmarks/mamba3_cuda_full_bwd_ab/mono_ab_wave1_h200_smoke_20260430_1/summary.csv`

Single-iteration smoke timings:

| path | bwd_fwd ms | bwd_bwd ms | chain ms | max allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.100544 | 0.211200 | 0.242528 | 0.004676 |
| main guarded stage2 | 0.110272 | 0.271136 | 0.247264 | 0.006695 |

Correctness:

- guarded stage2 vs baseline: all tracked diffs `0.0`;
- `max_main_grad_abs_diff=0.0`;
- CUDA covered subset component max abs: `2.274e-13`;
- CUDA covered subset combined same-time slice: `0.035296 ms`;
- CUDA component peak allocated: `0.011453 GiB`.

This is a smoke receipt only. The one-sample smoke timing should not supersede
the prior productionish H200 evidence for guarded stage2.

At the final Modal check after the smoke, this harness app was stopped with
`Tasks=0`. A separate same-campaign H200 app still had `Tasks=1`, so it was
left running.

## Production A/B Entry Gates

A monolithic `mamba_mimo_bwd_bwd` kernel may enter production A/B only after:

1. Every output slot above matches `main_guarded_stage2` within the declared
   tolerance on the real call boundary.
2. Off-time/state work and full `DK/DQ/DV/DMIMO_V` accumulation are in-kernel,
   not delegated to component sidecars or Python epilogues.
3. Scalar/state outputs `dfactor`, `dangles`, `dd`, `dda`, `dssda`,
   `dda_cs_rev`, and `dda_cs` are implemented and checked.
4. Integrated peak allocated/reserved memory is at or below guarded stage2.
5. Launch count is one `bwd_bwd` replacement launch, or extra launches are
   justified by measured chain speedup.
6. H200 smoke plus productionish A/B pass against guarded stage2.
7. H100 or agreed portability smoke passes.
8. Modal hygiene shows no left-running wave-owned apps after artifact capture.
9. A guarded training A/B confirms the microbench result survives workload
   variance.
