# Mamba3 bwd_bwd State/Chunk Split Probe - 2026-04-29

Status: evidence
Canonical: none
Date: 2026-04-29
Scope: H200 viability probe for a correctness-valid Mamba3 TileLang bwd_bwd state/chunk split.

Branch: `worker/mamba3-bwd-bwd-state-chunk-split`

Base: `worker/mamba3-stage2-force-nontma` at `972608d`.

Goal: test a larger `bwd_bwd` redesign, not the rejected narrow DGAMMA split and
not the old naive state/chunk split. The viability bar is a net productionish H200
saving greater than 0.40 ms versus stage2 `(bf_num_stages=1, bb_num_stages=0)`.

## Redesign Tested

The clean split point is `dstates_before_chunk`, not "state path vs chunk path".

- Pass1 computes fp32 `dstates_before_chunks [B, H, nchunks, N, P]`.
- Pass1 recurrence:
  `dstates = dstates * exp(dA_cs_sum) + Q_c^T @ dPhiO_scaled`.
- Pass2 loads `dstates_before_chunks[:, :, chunk]` and computes the normal stitched
  outputs (`dq`, `dk`, `dv`, `DDA*`, `DSSDA`, `DANGLES`, `DMIMO_V`) without carrying
  `dstates` across reverse chunks.

This avoids the false assumption called out in
`docs/mamba3_mimo_p3_register_split_design.md`: the pass that advances dstates
must still see Q and dPhiO. The split is correctness-valid, but it duplicates
Q/dPhiO preparation and adds a large fp32 handoff tensor.

## Correctness

Local CPU smoke:

```text
pytest -q tests/test_mamba3_bwd_bwd_state_chunk_split.py
2 passed
```

Modal H200 GPU smoke:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 900s \
modal run scripts/modal_mamba3_bwd_bwd_state_chunk_split.py \
  --run-id state_chunk_split_h200_20260429_1 \
  --shape-csv smoke,productionish \
  --warmup 5 \
  --iters 20
```

Artifact:

- `/benchmarks/mamba3_bwd_bwd_state_chunk_split/state_chunk_split_h200_20260429_1/summary.json`

Device:

- GPU: `NVIDIA H200`
- capability: `(9, 0)`
- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`

Correctness result:

| output | max abs diff vs monolithic PyTorch reference |
| --- | ---: |
| `DQ`, `DK`, `DV`, `DMIMO_V` | 0.0 |
| `DDA`, `DDA_CS`, `DDA_CS_REV`, `DSSDA` | 0.0 |
| `DFACTOR`, `DGAMMA_DIAG`, `DANGLES` | 0.0 |
| `DSTATES_BEFORE_CHUNKS` pass1 vs monolithic capture | 0.0 |

## Stage2 Baseline Comparison

Fresh H200 run from this worktree:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s \
modal run scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id state_chunk_split_stage2_h200_20260429_1 \
  --shape-csv smoke,productionish \
  --variant-csv stage2_force_nontma \
  --warmup 1 \
  --iters 4
```

Artifacts:

- `/benchmarks/mamba3_stage2_force_nontma_benchmark/state_chunk_split_stage2_h200_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/state_chunk_split_stage2_h200_20260429_1/summary.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/state_chunk_split_stage2_h200_20260429_1/summary.csv`

CUDA-event timing:

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | correctness vs unpatched baseline |
| --- | --- | ---: | ---: | ---: | --- |
| smoke | baseline | 0.0804 | 0.1624 | 0.2245 | reference |
| smoke | stage2 `(1,0)` | 0.0807 | 0.1611 | 0.2269 | `max_main_grad_abs_diff=0.0` |
| productionish | baseline | 1.8995 | 3.7311 | 5.6023 | reference |
| productionish | stage2 `(1,0)` | 1.8078 | 3.7156 | 5.4938 | `max_main_grad_abs_diff=0.0` |

The comparison target for this task is therefore:

- productionish stage2 bwd_bwd: `3.7156 ms`
- productionish stage2 chain: `5.4938 ms`

## Handoff Cost Lower Bound

The split handoff tensor for productionish
`B=4,S=4096,H=32,G=1,N=64,P=128,R=4,chunk=16` is:

```text
B * H * nchunks * N * P * sizeof(fp32)
= 4 * 32 * 256 * 64 * 128 * 4
= 1,073,741,824 bytes = 1.0 GiB
```

A single device copy of that tensor represents the minimum pass1-store plus
pass2-load traffic: one read + one write = 2.0 GiB traffic. It does not include
the pass1 recurrence work or duplicated Q/dPhiO preparation.

Measured on H200:

| shape | tensor size | traffic | mean ms | min ms | effective GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 1 MiB | 2 MiB | 0.0244 | 0.0194 | 80 |
| productionish | 1 GiB | 2 GiB | 0.5152 | 0.5119 | 3882 |

## Viability Read

To clear the requested ROI, the split would need:

```text
net saving > 0.40 ms
pass2 speedup > 0.40 ms + 0.515 ms handoff + pass1 recurrence cost
pass2 speedup > 0.915 ms + pass1 recurrence cost
```

Against a `3.7156 ms` stage2 bwd_bwd, that means pass2 alone would need to save
at least 24.6% before accounting for the extra pass1 compute. The pass1 compute is
not small: it still rotates Q, builds dPhiO, applies dA scaling, and runs the
`Q^T @ dPhiO_scaled` recurrence for every chunk. Pass2 still has to run almost all
remaining original work to produce `dq/dk/dv/DDA*` correctly.

The design is correctness-valid, but the measured handoff floor already exceeds
the entire 0.40 ms target before adding pass1 arithmetic. It is therefore not a
viable production optimization on H200 for this shape.

## Decision

Discard for production.

Keep the PyTorch prototype as a regression/spec reference for any future attempt
that can avoid the 1 GiB fp32 handoff, for example by fusing a smaller compressed
state handoff or by moving the split boundary into an existing cached forward pass.
Do not spend the next kernel iteration on a full TileLang pass1/pass2
implementation unless the handoff tensor is eliminated or made much smaller.

## Modal Apps

- `ap-nNWxqEuDI22BnmmegS9wbD`: state/chunk split probe, completed and stopped.
- `ap-kGTBpVqJDZDsj0P3LC6Aya`: stage2 comparison run, completed and stopped.
- Extra ephemeral apps observed after the runs: `ap-UPzob9FpbWfUZ3hvqQ8kLA`,
  `ap-FAQWmMrM0IB2KmO2IBFkGf`, `ap-eFmxe2O29KvxbAPieif0oa`,
  `ap-QT131DYZSYi437WItu5xbO`, `ap-W5sOZ7YhjoSn9TkBq2ngob`,
  `ap-lPnxaz2CM4NZnwDieOMgQQ`, `ap-jziAXT0DsMN0rZvtsMs8Jp`.
  `modal app stop --yes` was issued for each.
