# Mamba3 State/LKQ/D Tensor Wave10 - 2026-04-30

Status: evidence / blocker
Canonical: no
Branch: `worker/mamba3-cuda-dmimo-reduce`

## Scope

Wave10 Lane B tried one serious tensorized prototype for the remaining
state/LKQ/D `dPsiV` producer in Mamba3 MIMO `bwd_bwd`, without adding another
scalar CUDA loop kernel.

Baseline context:

- Wave8 combined CUDA target before this lane: `2.45093 ms`.
- TileLang `stage2_bf1_bb0` full `bwd_bwd`: `3.70674 ms`.
- Remaining budget after wave8 if full CUDA is to beat TileLang:
  `3.70674 - 2.45093 = 1.25581 ms`.
- Wave9 scalar state/LKQ/D ownership skeleton: `27.05544 ms`.

## Inspection

I used:

- `cppmega/megatron/cute_dsl_mimo/full_bwd_bwd_epilogue.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_stage2_force_nontma.patch`
- `docs/status/mamba3_cuda_remaining_state_wave9_2026_04_30.md`

The largest tensor-core-shaped remaining producer is the non-qk `dPsiV` path
inside the chunk body:

1. state: `K[64,64] @ dstates[64,P]`
2. LKQ construction: `K[64,64] @ Q[64,64].T`
3. LKQ apply: `masked(LKQ)[64,64] @ dPhiO[64,P]`
4. direct D: `D[h] * dPhiO`

For productionish `B=4, S=4096, H=32, N=64, P=128, R=4, chunk=16`:

| item | value |
| --- | ---: |
| chunk programs `(B*H*nchunks)` | `32,768` |
| fused chunk rows `chunk*R` | `64` |
| causal LKQ entries per chunk | `1,920` |
| useful state FMA | `17.18B` |
| useful causal LKQ FMA | `4.03B` |
| useful causal LKQ apply FMA | `8.05B` |
| useful state/LKQ total | `29.26B FMA` |

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave10_state_lkq_d_triton.py`
- `scripts/modal_mamba3_state_lkq_d_wave10_triton.py`

The Triton prototype:

- uses `tl.dot` for all three matrix products;
- owns one `(B, H, chunk, P tile)` program;
- writes wave9-equivalent `DV`, `DD`, and `[B,H,nchunks,R,P]` `DMIMO_V`
  partials;
- uses a second Triton reducer for `DMIMO_V`;
- has two LKQ-apply modes:
  - `fp32`: reference-style `LKQ(fp32) @ dPhiO(fp32)` via TF32;
  - `bf16`: casts masked LKQ to bf16 before the apply dot, modeling a dtype
    shared-memory handoff and giving the fastest measured path.

No global `dPsiV` temp is written. The partial tensor is still `64.0 MiB`, with
`128.25 MiB` extra read/write traffic for partial reduction.

## H200 Runs

Smoke + productionish, fp32 and block sweep:

```text
timeout 1200s modal run --timestamps \
  scripts/modal_mamba3_state_lkq_d_wave10_triton.py \
  --shape-csv smoke,productionish \
  --block-p-csv 64,128 \
  --warmup 3 --iters 10
```

Best bf16 LKQ-apply path:

```text
timeout 1200s modal run --timestamps \
  scripts/modal_mamba3_state_lkq_d_wave10_triton.py \
  --shape-csv smoke,productionish \
  --block-p-csv 128 \
  --lkq-apply-dtype-csv bf16 \
  --warmup 3 --iters 10
```

Runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

## Correctness

All checks compare against the torch reference for the isolated state/LKQ/D
contribution.

| shape | variant | `DV` max diff | `DD` max diff | `DMIMO_V` max diff |
| --- | --- | ---: | ---: | ---: |
| smoke | `BLOCK_P=64`, fp32 apply | `4.768e-07` | `2.384e-07` | `2.768e-07` |
| smoke | `BLOCK_P=128`, bf16 apply | `1.192e-07` | `4.172e-07` | `4.610e-08` |
| productionish | `BLOCK_P=64`, fp32 apply | `4.768e-07` | `2.289e-05` | `8.405e-07` |
| productionish | `BLOCK_P=128`, fp32 apply | `4.768e-07` | `1.526e-05` | `8.405e-07` |
| productionish | `BLOCK_P=128`, bf16 apply | `4.768e-07` | `1.335e-05` | `3.986e-07` |

## Timings

Productionish:

| variant | producer `DV+DD+partials` | partial reduce | two-pass total | projected with wave8 | ratio vs TileLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BLOCK_P=64`, fp32 apply | `4.88213 ms` | `0.04562 ms` | `4.89726 ms` | `7.34819 ms` | `1.982x` |
| `BLOCK_P=128`, fp32 apply | `3.48196 ms` | `0.04406 ms` | `3.50443 ms` | `5.95536 ms` | `1.607x` |
| `BLOCK_P=128`, bf16 apply | `2.84092 ms` | `0.03635 ms` | `2.86062 ms` | `5.31155 ms` | `1.433x` |

`BLOCK_P=128`, bf16 apply also measured `2.33980 ms` for `DV+DD` without
`DMIMO_V` partial writes. Even that is `1.86x` over the entire `1.25581 ms`
remaining budget before any DK/DQ or scalar state paths are included.

The best tensorized state/LKQ/D path is `27.05544 / 2.86062 = 9.46x` faster
than the wave9 scalar ownership skeleton, but it is not viable for the full
budget.

## Quantitative Blocker

The best measured Triton path (`BLOCK_P=128`, bf16 apply) executes:

| dot work | FMA |
| --- | ---: |
| state `K @ dstates` | `17.18B` |
| full LKQ `K @ Q.T` | `8.59B` |
| masked LKQ apply | `17.18B` |
| total executed dot work | `42.95B` |

The useful causal work is only `29.26B FMA`; the prototype pays full matrix
products for masked causal regions. With `BLOCK_P=64`, LKQ is recomputed for
two P tiles and executed dot work rises to `51.54B FMA`.

Measured throughput for the best producer is about:

```text
42.95B FMA / 2.84092 ms = 15.1 TFMA/s
```

To fit just this state/LKQ/D slice into the full remaining budget would require:

```text
42.95B FMA / 1.25581 ms = 34.2 TFMA/s
```

That still leaves no time for remaining `DK/DQ` state+intra paths,
`DDA_CS`, `DDA_CS_REV`, `DFACTOR`, `DSSDA`, `DDA`, `DANGLES`, and rotary/trap
plumbing. Wave9 estimated those paths add roughly another `80-90B`
FMA-equivalent. For the whole remaining body to fit in `1.25581 ms`, the
combined remaining work would need roughly `87-95 TFMA/s`, while this actual
tiny-matmul Triton shape reaches only `15 TFMA/s`.

Memory is not the primary blocker here:

- no global `dPsiV` temp is written;
- `DMIMO_V` partials are `64.0 MiB`;
- partial reduction traffic is `128.25 MiB`;
- the reducer itself is only `0.036 ms` on H200.

The blocker is arithmetic shape and scheduling: many `64x64x128` or
`64x64x64` products, full causal-mask work, and insufficient reuse across
state, LKQ, DV, DMIMO, and DK/DQ consumers.

## Required Design

An incremental Triton side kernel is not enough. To have a plausible chance,
the remaining work needs a monolithic CUDA/CuTe or TileLang-level rewrite that:

1. owns one full `(B,H,chunk)` body, not one P tile;
2. computes LKQ once per chunk and reuses it across all P tiles and the DK/DQ
   intra paths;
3. keeps `K`, `Q`, `dPhiO`, `PsiV`, state, and masked LKQ live long enough to
   feed `DV`, `DMIMO_V` partials, `DSSDA`, `DK`, and `DQ`;
4. avoids or reduces the full lower-triangle causal zero work;
5. collapses wave7/wave8/wave10 launch boundaries so the qk, state, and
   intra-chunk consumers share data instead of rereading/recomputing.

## Conclusion

This wave proves tensorization helps the isolated state/LKQ/D producer
substantially, but the measured Triton design does not fit the budget.

Full CUDA `bwd_bwd` is no longer likely to beat TileLang through incremental
slice replacement. It remains possible only with a full fused CuTe/CUDA chunk
kernel that reuses LKQ and the surrounding state/intra intermediates across all
remaining consumers. Without that larger rewrite, the projected full CUDA path
is `~5.31 ms` for wave8 plus this best state/LKQ/D slice, already slower than
TileLang `3.70674 ms` before the rest of `bwd_bwd` is implemented.
