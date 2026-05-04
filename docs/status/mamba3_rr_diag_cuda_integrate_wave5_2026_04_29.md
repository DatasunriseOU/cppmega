# Mamba3 R x R Diagonal CUDA Integration Wave 5 - 2026-04-29

Branch: `worker/mamba3-rr-diag-cuda-integrate`

## Goal

Turn the positive wave4 standalone CUDA signal into a realistic `bwd_bwd`
integration experiment.

Wave4 reference point on H200 productionish:

| path | mean ms |
| --- | ---: |
| full torch diagonal reference | 6.8310 |
| wave3 Triton diagonal microbench | 2.6853 |
| wave4 CUDA diagonal microbench | 2.0560 |

## Harness Inspection

Wave4 standalone harness:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave4_cuda_microbench.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_kernel.cu`
- uses prepacked tensors shaped like `[tiles, chunk, R, P]` and
  `[tiles, chunk, R, N]`;
- launches one CUDA block per `(tile, timestep)` and returns standalone
  `DGAMMA_DIAG`, DK diagonal delta, and DQ diagonal delta;
- does not pay production `bwd_fwd`, `bwd_bwd`, output ownership, or launch
  boundary costs.

Stage2 benchmark harness:

- `scripts/modal_mamba3_stage2_force_nontma_benchmark.py`
- copies upstream `mamba3_mimo_bwd.py` into a tempdir, applies local patch
  stacks, imports the patched module, and times `bwd_fwd`, `bwd_bwd`, and the
  chain with CUDA events;
- `stage2_bf1_bb0` remains the production baseline: flattened Q/K and
  WS/TMA for `bwd_fwd`, non-WS/non-TMA for `bwd_bwd`;
- older `stage2_rr_diag_triton` variants skip the TileLang diagonal work and
  rebuild it with a post-kernel. I fixed the mode plumbing so `"chunk"` remains
  distinct from boolean `True`.

## Implemented Experiment

Added a production-layout CUDA post-kernel:

- `stage2_rr_diag_post_kernel` in
  `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_kernel.cu`;
- wrapper functions in
  `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py`;
- benchmark variant `stage2_rr_diag_cuda` in
  `scripts/modal_mamba3_stage2_force_nontma_benchmark.py`.

Mechanics:

1. Apply `mamba3_bwd_stage2_force_nontma.patch`.
2. Apply `mamba3_bwd_stage2_rr_diag_skip.patch`, which removes the full
   `dqk_from_diag` TileLang GEMM and its `DGAMMA_DIAG` / DK / DQ consumers.
3. Run the normal stage2 `bwd_fwd`.
4. Run the skipped `bwd_bwd`.
5. Launch the CUDA post-kernel against the real stage2 tensors:
   `DOUT`, flattened `Q/K`, `V`, `Q/K_BIAS`, `MIMO_V/O`, flat `QK_DOT`,
   `DT`, `TRAP`, `DK`, `DQ`, and `DGAMMA_DIAG`.

This is still a split post-kernel, but unlike wave4 it measures the real
stage2 launch boundary and production tensor layout.

## Local Checks

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py \
  scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  scripts/modal_mamba3_rr_diag_wave4_cuda_microbench.py

git diff --check
```

Both passed.

## H200 Runs

Smoke compile/correctness:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s \
modal run --timestamps scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_rr_diag_cuda_smoke_20260429_1 \
  --shape-csv smoke \
  --variant-csv baseline,stage2_bf1_bb0,stage2_rr_diag_cuda \
  --warmup 1 \
  --iters 3
```

Representative and productionish:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1500s \
modal run --timestamps scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_rr_diag_cuda_h200_rep_prod_20260429_1 \
  --shape-csv representative,productionish \
  --variant-csv baseline,stage2_bf1_bb0,stage2_rr_diag_cuda \
  --warmup 1 \
  --iters 4
```

Artifacts in Modal Volume `cppmega-mamba3-benchmarks`:

- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_cuda_smoke_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_cuda_smoke_20260429_1/summary.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_cuda_h200_rep_prod_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_cuda_h200_rep_prod_20260429_1/summary.json`

Device:

- GPU: `NVIDIA H200`
- device count: `2`
- capability: `(9, 0)`
- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`

## Correctness

All `stage2_bf1_bb0` diffs versus baseline were exactly zero.

For `stage2_rr_diag_cuda`:

| shape | max main grad abs diff | dgamma max abs diff | qk_dot diff | states diff |
| --- | ---: | ---: | ---: | ---: |
| smoke | 7.276e-12 | 6.509e-12 | 0.0 | 0.0 |
| representative | 7.276e-12 | 2.559e-11 | 0.0 | 0.0 |
| productionish | 1.455e-11 | 2.232e-11 | 0.0 | 0.0 |

Correctness is fine. The small nonzero diffs are from the split CUDA reduction
order and bf16 output update boundary.

## Performance

Smoke, `B=1,S=256,H=4,G=1,N=64,P=64,R=4`:

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | post diag |
| --- | ---: | ---: | ---: | --- |
| baseline | 0.08096 | 0.16337 | 0.22954 | none |
| stage2_bf1_bb0 | 0.08071 | 0.16196 | 0.22821 | none |
| stage2_rr_diag_cuda | 0.08606 | 0.15666 | 0.22082 | CUDA |

Smoke is positive, but it is not representative.

Representative, `B=2,S=1024,H=16,G=1,N=64,P=64,R=4`:

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | post diag |
| --- | ---: | ---: | ---: | --- |
| baseline | 0.27622 | 0.66137 | 0.93657 | none |
| stage2_bf1_bb0 | 0.28038 | 0.65680 | 0.92459 | none |
| stage2_rr_diag_cuda | 0.27434 | 0.78660 | 1.07370 | CUDA |

Against `stage2_bf1_bb0`, the CUDA split is:

- `0.835x` on `bwd_bwd`;
- `0.861x` on chain.

Productionish, `B=4,S=4096,H=32,G=1,N=64,P=128,R=4`:

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | post diag |
| --- | ---: | ---: | ---: | --- |
| baseline | 1.86639 | 3.71197 | 5.55595 | none |
| stage2_bf1_bb0 | 1.79373 | 3.69708 | 5.45278 | none |
| stage2_rr_diag_cuda | 1.78870 | 6.53346 | 8.29052 | CUDA |

Against `stage2_bf1_bb0`, the CUDA split is:

- `0.566x` on `bwd_bwd`;
- `0.658x` on chain.

## Read

The standalone CUDA microkernel signal does not survive the realistic split
integration. The productionish path regresses `bwd_bwd` by `+2.8364 ms` and
chain by `+2.8377 ms` versus `stage2_bf1_bb0`.

Primary blockers:

- the CUDA post-kernel rereads production-layout globals and recomputes
  `DOUT/V * MIMO_O/V`, instead of consuming resident `dPhiO` / `PsiV` inside
  the TileLang `bwd_bwd` body;
- the split path stores non-diagonal DK/DQ to bf16, reloads them, adds the
  diagonal delta, and stores again;
- the extra host launch is paid on the critical `bwd_bwd` boundary;
- current TileLang in-body alternatives are still unattractive: the existing
  patch serializes over `P`, while padded per-timestep GEMMs recreate the full
  tensor-core work or require a large `[chunk, R, R, P]` accumulator footprint.

## In-Launch Sketch

The only remaining viable version would be a real in-launch custom CUDA/CuTe
`bwd_bwd` port:

1. Keep stage2 `bwd_fwd` as-is.
2. In `bwd_bwd`, after the CTA has `dPhiO`, `PsiV`, pre-rotated Q, pre-rotated
   K, `QK_DOT`, and gamma-equivalent values live, call a device helper that
   computes the same `R=4` diagonal block and immediately accumulates into
   `DGAMMA_DIAG`, DK, and DQ.
3. Keep the reverse-causal off-time intrachunk path unchanged.
4. Avoid the bf16 store/reload boundary and avoid a separate kernel launch.

I did not attempt this full kernel port in this wave. TileLang does not offer a
clean way to call this CUDA helper from the generated kernel at the required
point, so doing it properly means owning more of `bwd_bwd` in CUDA/CuTe.

## Modal Cleanup

Runs started in this wave:

- `ap-Ml0QbF58Bux9ZWrsak2YNj`: stopped, tasks=0.
- `ap-k9iMnPsa2Gwc1kjw2VsTsQ`: stopped, tasks=0.

`modal app list` showed no running tasks for these apps. The pre-existing
deployed `cppmega-prebuilt` app had tasks=0 and was left untouched.

## Recommendation

Do not continue with split/post-kernel diagonal variants for this path.

This path survives only as a larger custom in-launch `bwd_bwd` rewrite. For the
near-term production branch, keep `stage2_bf1_bb0` as the baseline and spend the
next wave elsewhere unless there is appetite to port enough of `bwd_bwd` to
CUDA/CuTe to eliminate the launch and store/reload costs.
