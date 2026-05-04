# Mamba3 Stage2 CUDA A/B Wave5 - 2026-04-29

Branch: `worker/mamba3-stage2-cuda-ab`

Scope: Lane C benchmarking/profiling/integration harness for the guarded
stage2 production candidate, plus a read-only CUDA diag A/B envelope. This lane
did not implement or integrate the CUDA diag kernel into production.

## Harness

Added:

- `scripts/modal_mamba3_stage2_cuda_ab_benchmark.py`

The harness compares:

- `baseline`: upstream TileLang MIMO `bwd_fwd` + `bwd_bwd`;
- `stage2_bf1_bb0`: stage2 force-nonTMA patch, flattened Q/K and QK_DOT,
  `bf_num_stages=1`, `bb_num_stages=0`;
- optional `stage2_bf1_bb0_plus_wave4_cuda_diag_host_split`: read-only wave4
  CUDA diag call timed after the current stage2 chain.

The CUDA diag mode is intentionally labeled as a host-split envelope. It adds
the standalone diag call after the current chain and therefore measures call
overhead plus duplicate work. It is not a production integration and should not
be read as the expected performance of a device-side fused implementation.

Diag source resolution:

1. `CPPMEGA_MAMBA3_RR_DIAG_SOURCE_DIR`, if set;
2. copied Lane A files in `upstream_prs/examples/13_tilelang_floormod_dbz`;
3. read-only wave4 reference worktree:
   `/home/dave/source/cppmega/.claude/worktrees/mamba3-rr-diag-microkernel/upstream_prs/examples/13_tilelang_floormod_dbz`.

The production-control path remains default-off and unchanged.

## Commands

Local compile:

```text
python -m py_compile scripts/modal_mamba3_stage2_cuda_ab_benchmark.py
```

H200 smoke:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 900s \
modal run scripts/modal_mamba3_stage2_cuda_ab_benchmark.py \
  --run-id stage2_cuda_ab_h200_smoke_20260429_1 \
  --shape-csv smoke \
  --warmup 0 \
  --iters 1 \
  --diag-mode none
```

H200 productionish A/B plus read-only CUDA diag envelope and NCU attempt:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 2400s \
modal run scripts/modal_mamba3_stage2_cuda_ab_benchmark.py \
  --run-id stage2_cuda_ab_h200_prod_diag_ncu_20260429_1 \
  --shape-csv productionish \
  --warmup 2 \
  --iters 8 \
  --diag-mode wave4-readonly \
  --diag-num-warps 4 \
  --ncu \
  --ncu-shape productionish \
  --ncu-timeout-sec 600
```

Artifacts in Modal Volume `cppmega-mamba3-benchmarks`:

- `/benchmarks/mamba3_stage2_cuda_ab_benchmark/stage2_cuda_ab_h200_prod_diag_ncu_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_cuda_ab_benchmark/stage2_cuda_ab_h200_prod_diag_ncu_20260429_1/summary.json`
- `/benchmarks/mamba3_stage2_cuda_ab_benchmark/stage2_cuda_ab_h200_prod_diag_ncu_20260429_1/summary.csv`
- NCU attempt stdout/stderr files under the same run's `ncu/` directory.

## Device

- GPU: `NVIDIA H200`
- device count: `2`
- capability: `(9, 0)`
- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`
- TileLang: `0.1.8+cu132.gitf309d814`

## Productionish Stage2 Result

Shape: `productionish` (`B=4, S=4096, H=32, G=1, N=64, P=128, R=4, bf16`)

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | bwd_fwd WS/TMA | bwd_bwd WS/TMA | chain speedup |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| baseline | 1.8677 | 3.6943 | 5.5316 | no / 0 | no / 0 | 1.0000x |
| stage2 `(bf=1,bb=0)` | 1.7845 | 3.6919 | 5.4394 | yes / 5 | no / 0 | 1.0169x |

Correctness versus baseline:

- `max_main_grad_abs_diff=0.0`
- tracked `qk_dot`, `states`, and all listed gradient/output diffs had
  `max_abs=0.0`

Read: the guarded stage2 path survives this cycle. The measured win is modest
but consistent with prior H200 productionish runs: `bwd_fwd` improves, `bwd_bwd`
stays essentially flat, and correctness is exact in the harness.

## CUDA Diag Envelope

Read-only source:

```text
/home/dave/source/cppmega/.claude/worktrees/mamba3-rr-diag-microkernel/upstream_prs/examples/13_tilelang_floormod_dbz
```

Standalone wave4 CUDA diag timing on the productionish shape:

| path | mean ms | min ms | max ms |
| --- | ---: | ---: | ---: |
| wave4 CUDA R x R diag | 2.0502 | 2.0473 | 2.0563 |

CUDA diag correctness versus the standalone full reference:

- `dgamma_max_abs=7.15e-7`
- `dk_delta_max_abs=4.77e-7`
- `dq_delta_max_abs=4.77e-7`

CUDA diag metadata:

| field | value |
| --- | ---: |
| threads per block | 128 |
| registers per thread | 40 |
| dynamic smem bytes | 8256 |
| active blocks per SM | 12 |
| active threads per SM | 1536 |
| theoretical occupancy | 75.0% |

Host-split envelope:

| path | mean ms | min ms | max ms | speedup vs baseline |
| --- | ---: | ---: | ---: | ---: |
| stage2 chain retimed | 5.4395 | 5.4329 | 5.4521 | 1.0169x |
| stage2 chain + standalone CUDA diag | 7.4869 | 7.4831 | 7.4885 | 0.7388x |

Read: the CUDA diag microkernel remains promising as a standalone subproblem,
but a host-side post-kernel call is not viable. Any CUDA diag candidate must
replace work inside the `bwd_bwd` launch boundary or be called as a device-side
helper; adding the standalone call after stage2 loses badly.

## NCU

NCU was present:

```text
/usr/local/cuda/bin/ncu
NVIDIA Nsight Compute 2026.1.1.0
```

Two attempts were made for productionish `stage2_force_nontma` `bwd_bwd`:

1. `--section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis`
2. fallback `--set basic`

Both failed with return code `9`:

```text
Failed to initialize the profiler: LibraryNotLoaded. Check that a compatible driver library is loaded.
```

Result: no NCU occupancy/register/memory counters were collected for stage2
`bwd_bwd` in Modal. The CUDA diag standalone resource numbers above come from
the wave4 extension's `cudaFuncGetAttributes` / occupancy API, not NCU.

## App Hygiene

Apps started by this lane:

- `ap-qBDSac57NzIZ8tCRgokUCS`
  (`cppmega-mamba3-stage2-cuda-ab-benchmark` smoke): stopped, `Tasks=0`
- `ap-osqdAOlwWvl1hiSCtDhHTU`
  (`cppmega-mamba3-stage2-cuda-ab-benchmark` productionish): stopped,
  `Tasks=0`

`modal app list --json` also showed an active
`cppmega-mamba3-stage2-force-nontma-benchmark` app with `Tasks=2` from parallel
work. It was not started by this lane and was left untouched.

## Merge Readiness

Safe to take now:

- guarded production-control stage2 path, default-off;
- production candidate constrained to `bf_num_stages=1`, `bb_num_stages=0`;
- this A/B harness and status docs.

Still experimental:

- CUDA diag standalone kernel and all host-split envelope measurements;
- any variant that enables `bb_num_stages > 0`;
- NCU counter-level attribution in Modal, until the profiler library issue is
  fixed or profiling is moved to an environment with compatible driver support.

Next wave recommendation:

1. Merge guarded stage2 `(bf=1,bb=0)` as the only current mergeable candidate.
2. Run a longer end-to-end training A/B before any default-on exposure.
3. When Lane A produces copied integration artifacts, rerun this harness with
   `CPPMEGA_MAMBA3_RR_DIAG_SOURCE_DIR` pointed at those artifacts.
4. Integrate CUDA diag only inside the `bwd_bwd` launch boundary, replacing the
   existing diagonal subpath rather than adding a host-side call.
