# Mamba3 R x R Diagonal Microkernel Wave 3 - 2026-04-29

Branch: `worker/mamba3-rr-diag-microkernel`

Base: wave2 commit `0424c43ef10acc0569c5df0c5568e3fd5bd51c4c`.

## Goal

Stop testing host-side stage2 post-kernel split variants and answer the narrower
Lane A question: can the R x R same-time diagonal subgraph itself be fast when
it has enough CTA parallelism?

I did not rerun `stage2_rr_diag_triton` or `stage2_rr_diag_triton_chunk`.

## Implemented Output

Accepted output type: standalone Triton microbench plus integration plan.

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave3_microbench.py`
  - standalone benchmark wrapper around the existing diagonal oracle and Triton
    kernel;
  - adds stage2-matching presets:
    - `representative`: `B=2,S=1024,H=16,N=64,P=64,R=4,chunk=16`;
    - `productionish`: `B=4,S=4096,H=32,N=64,P=128,R=4,chunk=16`;
  - reports CTA count and an integration plan.
- `scripts/modal_mamba3_rr_diag_wave3_microbench.py`
  - Modal H200 runner that mounts only the isolated 13_tilelang_floormod_dbz
    directory.

This is not production integration. It uses the existing timestep-owned Triton
microkernel as the custom-kernel proof point: one CTA per `(B,H,timestep)`,
computing the 4x4 `dPhiO @ PsiV.T` block and applying the `DGAMMA_DIAG`, `DK`,
and `DQ` diagonal consumers.

## Local Checks

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave3_microbench.py \
  scripts/modal_mamba3_rr_diag_wave3_microbench.py

python upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave3_microbench.py \
  --shape smoke --device cpu --iters 1 --warmup 0
```

Both passed. CPU smoke had exact equality vs full reference for `DGAMMA_DIAG`,
`DK` diagonal delta, and `DQ` diagonal delta.

## H200 Run

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 timeout 1200s \
modal run --timestamps scripts/modal_mamba3_rr_diag_wave3_microbench.py \
  --shape-csv representative,productionish \
  --iters 10 \
  --warmup 3
```

Modal app:

- `ap-p0owcn1eN75AjPij5hRhD3`, stopped, tasks=0.

Image:

- `ghcr.io/jewelmusicee/cppmega:785c3fd`

GPU:

- `NVIDIA H200`

Torch:

- `2.13.0.dev20260426+cu132`

## Correctness

Representative:

| path | dgamma max abs | dk delta max abs | dq delta max abs |
| --- | ---: | ---: | ---: |
| torch R x R vs full | 3.58e-7 | 1.79e-7 | 1.49e-7 |
| Triton R x R vs full | 3.58e-7 | 2.38e-7 | 1.79e-7 |

Productionish:

| path | dgamma max abs | dk delta max abs | dq delta max abs |
| --- | ---: | ---: | ---: |
| torch R x R vs full | 7.15e-7 | 5.96e-7 | 4.77e-7 |
| Triton R x R vs full | 1.19e-6 | 7.15e-7 | 7.15e-7 |

## Performance

Reference is the current full `[chunk * R, chunk * R]` diagonal extraction path
implemented by `torch.bmm` plus the same three diagonal consumers.

Representative `B=2,S=1024,H=16,N=64,P=64,R=4`:

| path | mean ms | min ms | speedup vs full reference |
| --- | ---: | ---: | ---: |
| full fused torch reference | 0.4275 | 0.4255 | 1.00x |
| torch R x R oracle | 0.4620 | 0.4597 | 0.93x |
| Triton R x R timestep CTA | 0.1767 | 0.1741 | 2.42x |

Productionish `B=4,S=4096,H=32,N=64,P=128,R=4`:

| path | mean ms | min ms | speedup vs full reference |
| --- | ---: | ---: | ---: |
| full fused torch reference | 6.8181 | 6.8152 | 1.00x |
| torch R x R oracle | 7.2079 | 7.2040 | 0.95x |
| Triton R x R timestep CTA | 2.6777 | 2.6720 | 2.55x |

CTA model:

| shape | one CTA/chunk | one CTA/timestep | CTAs per 132-SM H200 |
| --- | ---: | ---: | ---: |
| representative | 2,048 | 32,768 | 248.2 |
| productionish | 32,768 | 524,288 | 3,971.9 |

Read: the isolated R x R diagonal subgraph is not the blocker. With timestep
CTA ownership it beats the full reference on both required shapes.

## Integration Plan

Do not use a host-side post-kernel; wave1 and wave2 already showed the extra
launch loses in the full chain.

Next viable integration is a custom bwd_bwd launch boundary:

1. Keep stage2 `bwd_fwd` WS/TMA unchanged.
2. Replace the same-time `dqk_from_diag` users inside `bwd_bwd` with the
   timestep-owned R x R microkernel logic:
   - one CTA per `(B,H,timestep)`;
   - compute the 4x4 `dPhiO @ PsiV.T` block;
   - immediately apply `DGAMMA_DIAG`, DK diagonal delta, and DQ diagonal delta.
3. Keep the existing full reverse-causal off-time
   `dk_intrachunk` / `dq_intrachunk` path unchanged. That path is local within a
   chunk but not same-time diagonal.
4. Because current TileLang cannot express the useful form without either
   `T.serial(P)` or reintroducing full padded GEMM work, the next implementation
   should be a custom CUDA/CuTe-style `bwd_bwd` kernel or a TileLang extern
   call around such a kernel.

## Modal Cleanup

After the run, full-width `modal app list` showed:

- wave3 app `ap-p0owcn1eN75AjPij5hRhD3`: stopped, tasks=0;
- pre-existing deployed `cppmega-prebuilt`: deployed, tasks=0, left untouched.
- a non-wave detached app `cppmega-mamba3-bwd-bwd-crossp-accum`
  (`ap-1qIdttCKabrRdLADafQd5y`): running, tasks=1, left untouched because it
  does not belong to this worktree/task.

## Conclusion

Lane A should continue, but not through TileLang serial-P or host-side split
post-kernels.

The microkernel evidence is positive: R x R diagonal can beat the full diagonal
reference by `2.42x` representative and `2.55x` productionish. The remaining
risk is integration: the work has to move into the `bwd_bwd` launch boundary to
avoid the launch overhead that killed wave1/wave2.
