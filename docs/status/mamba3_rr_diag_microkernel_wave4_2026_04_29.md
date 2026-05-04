# Mamba3 R x R Diagonal Microkernel Wave 4 - 2026-04-29

Branch: `worker/mamba3-rr-diag-microkernel`

## Goal

Implement and benchmark a custom CUDA kernel for the same standalone
`R x R` same-time diagonal subproblem used in wave3:

- compute `dqk = dPhiO @ PsiV.T` for each `(tile, timestep)` 4x4 diagonal
  block;
- consume that block for `DGAMMA_DIAG`, DK diagonal delta, and DQ diagonal
  delta;
- compare against the full diagonal reference and the wave3 Triton
  timestep-CTA microbench.

No host-side full-chain split or production `bwd_bwd` integration was attempted
in this wave.

## Implemented Output

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_kernel.cu`
  - PyTorch C++/CUDA extension source;
  - one CUDA block per `(tile, timestep)`;
  - specializes the current target `R=4`;
  - parallel-reduces `P` into a 4x4 `dqk` block in shared memory, then applies
    `DGAMMA_DIAG`, DK, and DQ consumers before the block exits;
  - exposes `cudaFuncGetAttributes` and occupancy metadata.
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py`
  - JIT-build/load wrapper using `torch.utils.cpp_extension.load`;
  - builds for `sm_90` by default with `-O3`, `--use_fast_math`, `-lineinfo`,
    and `--ptxas-options=-v`.
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave4_cuda_microbench.py`
  - standalone benchmark comparing full reference, torch oracle, wave3 Triton,
    and wave4 CUDA.
- `scripts/modal_mamba3_rr_diag_wave4_cuda_microbench.py`
  - Modal H200 runner that mounts only the isolated
    `13_tilelang_floormod_dbz` directory.

## Local Checks

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave4_cuda_microbench.py \
  scripts/modal_mamba3_rr_diag_wave4_cuda_microbench.py

python upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave4_cuda_microbench.py \
  --shape smoke --device cpu --iters 1 --warmup 0
```

Both passed. CPU smoke had exact equality for the torch R x R oracle versus the
full reference.

## H200 Run

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 timeout 1200s \
modal run --timestamps scripts/modal_mamba3_rr_diag_wave4_cuda_microbench.py \
  --shape-csv representative,productionish \
  --iters 10 \
  --warmup 3
```

Modal app:

- `ap-fYxBr6WyAzJOsCht0Y3sJd`, stopped, tasks=0.

Image:

- `ghcr.io/jewelmusicee/cppmega:785c3fd`

GPU:

- `NVIDIA H200`

Torch:

- `2.13.0.dev20260426+cu132`

## Correctness

Representative `B=2,S=1024,H=16,N=64,P=64,R=4,chunk=16`:

| path | dgamma max abs | dk delta max abs | dq delta max abs |
| --- | ---: | ---: | ---: |
| torch R x R vs full | 3.58e-7 | 1.79e-7 | 1.49e-7 |
| Triton R x R vs full | 3.58e-7 | 2.38e-7 | 1.79e-7 |
| CUDA R x R vs full | 3.58e-7 | 1.49e-7 | 1.49e-7 |
| CUDA R x R vs Triton R x R | 4.77e-7 | 2.38e-7 | 2.38e-7 |

Productionish `B=4,S=4096,H=32,N=64,P=128,R=4,chunk=16`:

| path | dgamma max abs | dk delta max abs | dq delta max abs |
| --- | ---: | ---: | ---: |
| torch R x R vs full | 7.15e-7 | 5.96e-7 | 4.77e-7 |
| Triton R x R vs full | 1.19e-6 | 7.15e-7 | 7.15e-7 |
| CUDA R x R vs full | 7.15e-7 | 4.77e-7 | 4.77e-7 |
| CUDA R x R vs Triton R x R | 9.54e-7 | 5.96e-7 | 7.15e-7 |

## Performance

Representative:

| path | mean ms | min ms | speedup vs full | speedup vs wave3 Triton |
| --- | ---: | ---: | ---: | ---: |
| full fused torch reference | 0.5712 | 0.5599 | 1.00x | 0.36x |
| torch R x R oracle | 0.5411 | 0.5298 | 1.06x | 0.38x |
| wave3 Triton timestep CTA | 0.2030 | 0.2001 | 2.81x | 1.00x |
| wave4 CUDA timestep CTA | 0.1437 | 0.1423 | 3.97x | 1.41x |

Productionish:

| path | mean ms | min ms | speedup vs full | speedup vs wave3 Triton |
| --- | ---: | ---: | ---: | ---: |
| full fused torch reference | 6.8310 | 6.8274 | 1.00x | 0.39x |
| torch R x R oracle | 7.2146 | 7.2074 | 0.95x | 0.37x |
| wave3 Triton timestep CTA | 2.6853 | 2.6799 | 2.54x | 1.00x |
| wave4 CUDA timestep CTA | 2.0560 | 2.0524 | 3.32x | 1.31x |

Read: the custom CUDA kernel beats the wave3 Triton microbench on both shapes,
including `1.31x` on productionish H200.

## CUDA Metadata

From `cudaFuncGetAttributes` / occupancy API for the committed 128-thread
kernel:

| field | value |
| --- | ---: |
| threads per block | 128 |
| registers per thread | 40 |
| dynamic smem bytes | 8256 |
| static smem bytes | 0 |
| local bytes | 0 |
| active blocks per SM | 12 |
| active threads per SM | 1536 |
| max threads per SM | 2048 |
| theoretical occupancy | 75% |
| PTX / binary version | 90 / 90 |

I briefly tried a 64-thread variant. It improved the representative shape, but
the productionish Modal run was stopped twice before producing a result, so it
was not committed.

## Modal Cleanup

Wave4 apps after the runs:

- `ap-fYxBr6WyAzJOsCht0Y3sJd`: stopped, tasks=0;
- `ap-EHs3NnaK6LAw7NwF5CzoHq`: stopped, tasks=0;
- `ap-f92L3fIHZrAZYfmGTy5mL6`: stopped, tasks=0.

Pre-existing deployed app `cppmega-prebuilt-smoke` had tasks=0 and was left
untouched.

## Wave5 Recommendation

Continue Lane A with CUDA/CuTe-style integration. The standalone CUDA kernel is
faster than wave3 Triton and has acceptable resource usage for H200.

Wave5 should not revisit host-side post-kernel full-chain splits. Move the
timestep-owned `R=4` block logic into the `bwd_bwd` launch boundary, either by
porting the surrounding kernel to CUDA/CuTe or by adding a device-side helper
inside the fused kernel. Keep the existing off-time reverse-causal
`dk_intrachunk` / `dq_intrachunk` path unchanged until a separate triangular
algorithm is designed.
