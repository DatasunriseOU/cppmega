# Mamba3 CUDA Full bwd_bwd AB Harness Wave 9 - 2026-04-30

Branch: `worker/mamba3-cuda-full-bwd-ab`

Lane C scope: update the full-bwd AB/readiness harness for the qk-`DMIMO_V`
output-owner all-R sidecar result, replacement budget, memory accounting, and
readiness verdict. Writes were kept in this worktree.

## Implemented

Updated:

- `scripts/modal_mamba3_cuda_full_bwd_ab.py`

Added copied read-only sidecar files from `worker/mamba3-cuda-dmimo-reduce`
commit `9308289`:

- `scripts/modal_mamba3_dmimo_reduce_wave8_cuda.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/dmimo_reduce_cuda_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/dmimo_reduce_cuda_kernel.cu`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_dmimo_reduce_cuda.py`

The AB harness now reports:

- sidecar qk-`DMIMO_V` output-owner all-R receipts;
- projected wave7 + qk-`DMIMO_V` replacement totals;
- remaining budget versus TileLang stage2 `bwd_bwd`;
- launch-count accounting for the sidecar split;
- analytical temp/partial memory for atomic, two-pass, and output-owner paths.

No production `mamba_mimo_bwd_bwd` integration was attempted in this lane.

## Validation

Local:

```text
python -m py_compile \
  scripts/modal_mamba3_cuda_full_bwd_ab.py \
  scripts/modal_mamba3_dmimo_reduce_wave8_cuda.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/dmimo_reduce_cuda_extension.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_dmimo_reduce_cuda.py

python upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_dmimo_reduce_cuda.py \
  --shape smoke --device cpu --iters 1 --warmup 0
```

H100 smoke:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H100:2 timeout 900s \
  modal run scripts/modal_mamba3_dmimo_reduce_wave8_cuda.py \
  --shape-csv smoke --iters 5 --warmup 1
```

Modal app completed:

```text
https://modal.com/apps/jewelmusic/main/ap-66obVmma7eJqXwccTAnWPf
```

H200 productionish uses the already measured Wave 8 AB and Wave 8 sidecar
receipts. This lane did not rerun the full H200 TileLang+CUDA AB path because
the task allowed sidecar numbers and the relevant productionish numbers were
already available from the same image/runtime.

## H200 Productionish Inputs

Productionish shape:
`B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

Source receipts:

| source | component | mean ms | note |
| --- | --- | ---: | --- |
| Wave 8 AB | TileLang stage2 `bwd_fwd` | 1.81156 | H200, `stage2_bf1_bb0` |
| Wave 8 AB | TileLang stage2 `bwd_bwd` | 3.72232 | replacement reference |
| Wave 8 AB | TileLang stage2 chain | 5.50098 | `bwd_fwd + bwd_bwd` measured chain |
| Wave 8 AB | wave7 diag + qk/dV | 1.92990 | one CUDA launch |
| Wave 8 sidecar | qk-`DMIMO_V` output-owner all-R | 0.53634 | one CUDA launch, no temp |

The sidecar's own H200 productionish canonical projection was
`1.91459 + 0.53634 = 2.45093 ms`, or `0.661x` of its TileLang stage2
`bwd_bwd` reference (`3.70674 ms`). Normalized to the Wave 8 AB H200 receipt,
the combined replacement math is:

| path | bwd_bwd-equivalent ms | ratio vs stage2 `bwd_bwd` | speedup floor | remaining budget |
| --- | ---: | ---: | ---: | ---: |
| wave7 diag + qk/dV only | 1.92990 | 0.51845 | 1.929x | 1.79242 ms |
| wave7 + qk-`DMIMO_V` all-R sidecar | 2.46624 | 0.66256 | 1.509x | 1.25608 ms |

End-to-end floor if this incomplete candidate replaced `bwd_bwd` while keeping
stage2 `bwd_fwd` unchanged:

| path | chain ms | speedup vs stage2 chain |
| --- | ---: | ---: |
| stage2 TileLang chain | 5.50098 | 1.000x |
| wave7-only floor | 3.74146 | 1.470x |
| wave7 + qk-`DMIMO_V` floor | 4.27780 | 1.286x |

Read: the sidecar consumes `0.53634 ms` of the previous `1.79242 ms` budget and
still leaves `1.25608 ms` for missing full-kernel work before merely matching
TileLang stage2 on H200 productionish.

## Memory Accounting

Productionish qk-`DMIMO_V` analytical memory:

| path | temp allocation | extra global R/W | other cost |
| --- | ---: | ---: | --- |
| atomic chunk | 0 MiB | final output only | 16.78M global atomics |
| two-pass partials | 64.00 MiB partial tensor | 128.25 MiB | partial writer dominates latency |
| output-owner all-R | 0 MiB | final output only | loops over `S`, writes unique output |

The final `DMIMO_V` output itself is only `0.25 MiB` at productionish shape.
The two-pass final reducer is cheap (`0.03641 ms` in the H200 sidecar), but the
partial writer costs `1.97627 ms` and materializes the 64 MiB partial tensor.
That route should not be accepted as production unless a future full chunk
kernel piggybacks partial accumulation while `dPsiV` is already live.

Standalone harness peak memory is not a production memory claim. In Wave 8 AB,
the H200 CUDA component harness peaked at `6.92774 GiB` allocated versus
TileLang stage2 at `4.73024 GiB`, a `+2.19750 GiB` standalone delta. That comes
from independent inputs, outputs, torch references, and sidecar/test temporaries
being alive together. A production integrated kernel must avoid:

1. materializing two-pass `DMIMO_V` partials;
2. retaining torch reference tensors;
3. allocating duplicate full output tensors for comparison;
4. keeping standalone component inputs alive outside the real autograd lifetime.

The acceptable memory direction is the output-owner all-R path: no partial
tensor, no atomics, and unique writes into the final `DMIMO_V` output.

## H100 Smoke

Device: `NVIDIA H100 80GB HBM3`, torch `2.13.0.dev20260426+cu132`.

Smoke shape:
`B=1, S=256, H=4, G=1, N=64, P=64, R=4, chunk=16`.

| component | mean ms | correctness max abs |
| --- | ---: | ---: |
| wave7 diag + qk/dV | 0.031936 | - |
| qk-`DMIMO_V` atomic chunk | 0.020525 | 1.388e-15 |
| qk-`DMIMO_V` two-pass total | 0.016627 | 1.360e-15 |
| qk-`DMIMO_V` output-owner `(B,H,R,Ptile)` | 0.023174 | 2.665e-15 |
| qk-`DMIMO_V` output-owner all-R `(B,H,Ptile)` | 0.026003 | 2.665e-15 |

Smoke memory model:

| item | value |
| --- | ---: |
| final `DMIMO_V` output | 0.003906 MiB |
| two-pass partial tensor | 0.062500 MiB |
| two-pass extra global R/W | 0.128906 MiB |
| atomic adds | 16,384 |

Smoke is correctness/portability only. The tiny shape is underfilled; the
two-pass route winning there does not override the H200 productionish result,
where output-owner all-R is the only measured path that preserves the speed
budget and avoids partial memory.

## Launch Accounting

| path | bwd_bwd launch count | chain launch count with stage2 `bwd_fwd` |
| --- | ---: | ---: |
| TileLang stage2 | 1 | 2 |
| wave7 current candidate | 1 | 2 |
| wave7 + qk-`DMIMO_V` sidecar | 2 | 3 |
| desired production integrated replacement | 1 if ownership can be reconciled | 2 |

The output-owner all-R sidecar is the right memory/perf direction for
`DMIMO_V`, but it introduces an ownership split versus the wave7 `(B,H,chunk)`
kernel. A production design can keep the split only if measured end-to-end
speedup survives launch overhead and memory lifetime effects; otherwise the
full custom kernel needs a single-call-boundary design that handles both
ownership styles without allocating global partials.

## Readiness Verdict

Not ready for production replacement.

The qk-`DMIMO_V` ownership problem has a viable direction: output-owner all-R
adds `0.53634 ms` on H200 productionish and keeps the wave7+`DMIMO_V` floor at
about `2.46 ms`, still faster than TileLang stage2 `bwd_bwd`. The remaining
`1.25608 ms` budget is meaningful, but the candidate still lacks full
off-time/state work, full `DK/DQ/DV/DMIMO_V` accumulation, scalar outputs, and
real `mamba_mimo_bwd_bwd` integration.

Ready criteria before replacement:

1. Full output parity against TileLang for every `bwd_bwd` output tensor.
2. No global partial tensor for `DMIMO_V` in the production path.
3. Integrated memory peak at or below the TileLang stage2 path, not the
   standalone harness peak.
4. Launch count justified by measured chain speedup, or reduced to one
   `bwd_bwd` replacement launch.
5. H200 productionish rerun from the integrated path plus H100 smoke.
