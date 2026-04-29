# Mamba3 bwd_bwd Streaming Live-Set Probe - 2026-04-29

Branch: `worker/mamba3-bwd-bwd-streaming-live-set`

Base: `worker/mamba3-stage2-force-nontma` at `972608d`.

Goal: test a deeper TileLang `bwd_bwd` rewrite that reduces live fragment
pressure without changing output math, compared against the stage2 baseline
`bf_num_stages=1, bb_num_stages=0`.

## Patch

Patch: `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_bwd_streaming_live_set.patch`

This is an incremental patch applied after
`mamba3_bwd_stage2_force_nontma.patch`.

Changes:

- update `dPsiV_frag` in-place for D and `qk_dot` diagonal contributions,
  removing `dPsiV_D_fused_frag [chunk*R, P]`;
- build `PsiV_shared` directly from `v_frag * Psi_frag`, removing the staging
  `PsiV_frag [chunk, R, P]` in `bwd_bwd`;
- stream `DGAMMA_DIAG` over each step's `R x R` block without
  `dgamma_diag_prereduce_frag [chunk, R*R]`.

The first attempt used `T.Parallel(chunk_size, R, R)` for the streaming
`DGAMMA_DIAG` accumulation. TileLang correctly warned about a data race and the
run produced bad `DGAMMA_DIAG`. The committed patch uses `cs` parallel with
serial `R x R` accumulation.

## Harness

Script: `scripts/modal_mamba3_bwd_bwd_streaming_live_set_benchmark.py`

Default comparison uses the first selected variant as the reference, so the
streaming probe can compare directly against `stage2_bf1_bb0` without compiling
upstream baseline.

Variant set:

- `stage2_bf1_bb0`: stage2 reference, `bf=1`, `bb=0`;
- `streaming_live_set_bf1_bb0`: stage2 plus live-set patch, `bf=1`, `bb=0`;
- `streaming_live_set_bf1_bb1`: same patch with `bwd_bwd` WS/TMA enabled.

## Runs

Device:

- GPU: `NVIDIA H200`
- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`

Known launched apps from successful runs:

- smoke: `ap-2ueutqTBc8lZrRkklIpENA`, stopped
- productionish bb0: `ap-nERihrIGGH9alGO8fKlZfX`, stopped
- productionish bb1: `ap-eLMPF2W587PJ7rHeVosCyG`, stopped

### Smoke

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s \
modal run scripts/modal_mamba3_bwd_bwd_streaming_live_set_benchmark.py \
  --run-id streaming_live_set_h200_smoke_20260429_6 \
  --shape-csv smoke \
  --variant-csv stage2_bf1_bb0,streaming_live_set_bf1_bb0,streaming_live_set_bf1_bb1 \
  --warmup 1 \
  --iters 3
```

Artifacts:

- `/benchmarks/mamba3_bwd_bwd_streaming_live_set_benchmark/streaming_live_set_h200_smoke_20260429_6/report.json`
- `/benchmarks/mamba3_bwd_bwd_streaming_live_set_benchmark/streaming_live_set_h200_smoke_20260429_6/summary.csv`

Shape: `B=1, S=256, H=4, G=1, N=64, P=64, R=4`.

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | bwd_bwd WS/TMA | speedup bwd_bwd vs stage2 | speedup chain vs stage2 | correctness |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `stage2_bf1_bb0` | 0.0814 | 0.1612 | 0.2255 | no / 0 | 1.0000x | 1.0000x | reference |
| `streaming_live_set_bf1_bb0` | 0.0804 | 0.1649 | 0.2320 | no / 0 | 0.9774x | 0.9722x | main grads `0.0`; `DGAMMA_DIAG` max diff `2.78e-16` |
| `streaming_live_set_bf1_bb1` | 0.0810 | 0.1739 | 0.2355 | yes / 6 | 0.9272x | 0.9579x | main grads `0.0`; `DGAMMA_DIAG` max diff `2.78e-16` |

### Productionish bb0

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1500s \
modal run scripts/modal_mamba3_bwd_bwd_streaming_live_set_benchmark.py \
  --run-id streaming_live_set_h200_productionish_20260429_1 \
  --shape-csv productionish \
  --variant-csv stage2_bf1_bb0,streaming_live_set_bf1_bb0 \
  --warmup 1 \
  --iters 4
```

Artifacts:

- `/benchmarks/mamba3_bwd_bwd_streaming_live_set_benchmark/streaming_live_set_h200_productionish_20260429_1/report.json`
- `/benchmarks/mamba3_bwd_bwd_streaming_live_set_benchmark/streaming_live_set_h200_productionish_20260429_1/summary.csv`

Shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4`.

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | speedup bwd_bwd vs stage2 | speedup chain vs stage2 | correctness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `stage2_bf1_bb0` | 1.7906 | 3.7013 | 5.4639 | 1.0000x | 1.0000x | reference |
| `streaming_live_set_bf1_bb0` | 1.7930 | 3.7029 | 5.4641 | 0.9996x | 1.0000x | main grads `0.0`; `DGAMMA_DIAG` max diff `8.88e-16` |

### Productionish bb1

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1500s \
modal run scripts/modal_mamba3_bwd_bwd_streaming_live_set_benchmark.py \
  --run-id streaming_live_set_h200_productionish_bb1_20260429_1 \
  --shape-csv productionish \
  --variant-csv stage2_bf1_bb0,streaming_live_set_bf1_bb1 \
  --warmup 1 \
  --iters 3
```

Artifacts:

- `/benchmarks/mamba3_bwd_bwd_streaming_live_set_benchmark/streaming_live_set_h200_productionish_bb1_20260429_1/report.json`
- `/benchmarks/mamba3_bwd_bwd_streaming_live_set_benchmark/streaming_live_set_h200_productionish_bb1_20260429_1/summary.csv`

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | bwd_bwd WS/TMA | speedup bwd_bwd vs stage2 | speedup chain vs stage2 | correctness |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `stage2_bf1_bb0` | 1.8086 | 3.7191 | 5.4982 | no / 0 | 1.0000x | 1.0000x | reference |
| `streaming_live_set_bf1_bb1` | 1.8075 | 4.0661 | 5.8483 | yes / 7 | 0.9147x | 0.9401x | max main grad diff `1.80e-9` |

## Verdict

Discard for performance.

The rewrite is correctness-valid for the non-WS productionish path and it does
reduce the intended live fragments, but the schedule does not beat stage2 on
H200. The `bb=0` productionish result is effectively neutral/slightly slower
(`3.7029 ms` vs `3.7013 ms`), and `bb=1` remains a clear regression despite the
smaller live-set (`4.0661 ms` vs `3.7191 ms`).

The useful information is negative: removing these three live fragments is not
enough to make bwd_bwd WS/TMA viable, and serializing `DGAMMA_DIAG` eliminates
the race but likely offsets the register-pressure reduction.
