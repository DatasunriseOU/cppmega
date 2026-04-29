# Mamba3 bwd_bwd R x R Diagonal Specialization - 2026-04-29

Branch: `worker/mamba3-bwd-bwd-diag-rr-specialize`

Base: `worker/mamba3-stage2-force-nontma` commit `972608d`

## Goal

Replace the expensive full `(chunk_size * R, chunk_size * R)` diagonal
computation in `mamba_mimo_bwd_bwd` with per-time `(R, R)` work where the full
matrix is not semantically consumed.

The valid replacement scope is the `dqk_from_diag` subgraph:

- `DGAMMA_DIAG`
- diagonal contribution into `DK`
- diagonal contribution into `DQ`

This is not a narrow DGAMMA-only split. The same specialized `dqk` block feeds
all three downstream users.

The off-time reverse-causal `dk_intrachunk` / `dq_intrachunk` path remains full
local-masked work. It is local within the chunk, but not per-time diagonal, so a
plain `(R, R)` replacement would drop valid off-time terms.

## Files

- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_bwd_rr_diag_tilelang.patch`
  - TileLang patch against `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`.
  - Replaces full `dqk_from_diag` GEMM with `[chunk_size, R, R]` accumulation.
  - Changes the shared handoff from `[fused_chunk_size, fused_chunk_size]` to
    `[chunk_size, R, R]`.
  - Rewrites DK/DQ diagonal consumers to index by explicit `cs, r` loops.
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_specialization.py`
  - Isolated oracle + Triton companion kernel.
  - Compares full fused diagonal extraction vs R x R specialization.
- `scripts/modal_mamba3_rr_diag_benchmark.py`
  - Modal H200 wrapper for smoke and productionish isolated benchmarks.

## Patch Applicability

Dry-run against local canonical state-spaces/mamba source:

```text
patch --dry-run -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_bwd_rr_diag_tilelang.patch
checking file /tmp/.../mamba3_mimo_bwd.py
Hunk #2 succeeded at 901 (offset 1 line).
Hunk #3 succeeded at 927 (offset 1 line).
Hunk #4 succeeded at 1088 with fuzz 2 (offset -1 lines).
Hunk #5 succeeded at 1128 with fuzz 2 (offset 1 line).
```

Read: the patch applies to the current local source, but the last two hunks
need light context cleanup before upstreaming because they apply with fuzz.

## Correctness

CPU oracle:

```text
python upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_specialization.py \
  --device cpu --B 1 --S 64 --H 2 --N 16 --P 32 --iters 1 --warmup 0
```

Result: exact equality for `DGAMMA_DIAG`, `DK` diagonal delta, and `DQ`
diagonal delta.

GB10 smoke, `B=1,S=256,H=4,N=64,P=128,R=4`:

| implementation | dgamma max abs | dk delta max abs | dq delta max abs |
| --- | ---: | ---: | ---: |
| torch R x R vs full | 5.96e-7 | 2.38e-7 | 1.79e-7 |
| Triton R x R vs full | 4.77e-7 | 2.98e-7 | 2.38e-7 |

GB10 productionish, `B=4,S=4096,H=32,N=64,P=128,R=4`:

| implementation | dgamma max abs | dk delta max abs | dq delta max abs |
| --- | ---: | ---: | ---: |
| torch R x R vs full | 7.15e-7 | 4.77e-7 | 6.56e-7 |
| Triton R x R vs full | 9.54e-7 | 5.96e-7 | 7.75e-7 |

H200 productionish, `B=4,S=4096,H=32,N=64,P=128,R=4`:

| implementation | dgamma max abs | dk delta max abs | dq delta max abs |
| --- | ---: | ---: | ---: |
| torch R x R vs full | 7.15e-7 | 5.96e-7 | 4.77e-7 |
| Triton R x R vs full | 1.19e-6 | 7.15e-7 | 7.15e-7 |

## Performance

All timings are isolated subgraph timings, not full `mamba_mimo_bwd_bwd`
kernel timings.

### GB10 local

Smoke `B=1,S=256,H=4,N=64,P=128,R=4`, 5 iters:

| path | mean ms | speedup vs full fused torch |
| --- | ---: | ---: |
| full fused torch diagonal baseline | 0.0890 | 1.00x |
| torch R x R | 0.0764 | 1.16x |
| Triton R x R companion | 0.0244 | 3.64x |

Productionish `B=4,S=4096,H=32,N=64,P=128,R=4`, 3 iters:

| path | mean ms | speedup vs full fused torch |
| --- | ---: | ---: |
| full fused torch diagonal baseline | 54.4863 | 1.00x |
| torch R x R | 41.8785 | 1.30x |
| Triton R x R companion | 11.6179 | 4.69x |

### H200 Modal

Run:

```text
CPPMEGA_MODAL_GPU=H200 timeout 900s \
modal run scripts/modal_mamba3_rr_diag_benchmark.py \
  --shape-csv smoke,productionish --iters 3 --warmup 1
```

Modal app launched by this run:

- `ap-4IrCe1ZKUEtlDNR7TRILpv`
- state after run: `stopped`, `Tasks=0`

Smoke `B=1,S=256,H=4,N=64,P=128,R=4`:

| path | mean ms | speedup vs full fused torch |
| --- | ---: | ---: |
| full fused torch diagonal baseline | 0.2098 | 1.00x |
| torch R x R | 0.1541 | 1.36x |
| Triton R x R companion | 0.0509 | 4.12x |

Productionish `B=4,S=4096,H=32,N=64,P=128,R=4`:

| path | mean ms | speedup vs full fused torch |
| --- | ---: | ---: |
| full fused torch diagonal baseline | 6.8915 | 1.00x |
| torch R x R | 7.2804 | 0.95x |
| Triton R x R companion | 2.6902 | 2.56x |

## FLOP Model

For productionish `B=4,S=4096,H=32,N=64,P=128,R=4,chunk=16`:

| component | FLOPs |
| --- | ---: |
| current full `dqk_from_diag` product | 34.36e9 |
| R x R `dqk_from_diag` product | 2.15e9 |
| reduction | 16.0x |
| R x R plus DK/DQ diagonal consumers | 4.31e9 |

The isolated companion kernel is only 2.56x faster than the torch full baseline
on H200 because it is a standalone Triton microkernel with many small programs
and no fusion with the surrounding TileLang state. The algorithmic reduction is
still the expected 16x for the `dqk_from_diag` product itself.

## Full-kernel Win Estimate

Prior H200 stage2 baseline for productionish:

- `bwd_bwd`: 3.7216 ms
- chain: 5.5642 ms

If the diagonal subgraph accounts for 10-20% of `bwd_bwd`, replacing it with a
2.5-4.7x faster specialized path yields roughly:

- `bwd_bwd`: 6-16% faster
- chain: 4-11% faster

This is a scoped estimate; a true full-kernel result requires integrating the
TileLang patch and compiling/timing the production kernel. The patch reduces
shared memory by replacing one `[64,64]` fp32 shared tile with `[16,4,4]`, but
the serial-P accumulation may need a better TileLang reduction form to avoid
trading FLOPs for under-parallelized scalar loops.

## Next Transfer Step

Port `mamba3_bwd_bwd_rr_diag_tilelang.patch` into the stage2 full-kernel harness
as a new variant, then compare against `baseline` and `stage2_force_nontma`:

- first on smoke/representative for correctness vs baseline outputs,
- then on productionish with `bb_num_stages=0`,
- keep `dk_intrachunk` / `dq_intrachunk` full local-masked until a separate
  triangular local specialization exists.
