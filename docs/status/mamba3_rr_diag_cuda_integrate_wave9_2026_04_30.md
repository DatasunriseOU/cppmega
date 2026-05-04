# Mamba3 R x R Diagonal CUDA Integration Wave 9 - 2026-04-30

Branch: `worker/mamba3-rr-diag-cuda-integrate`

## Goal

Port the faster sidecar `DMIMO_V` ownership from
`worker/mamba3-cuda-dmimo-reduce` into the main wave8 combined CUDA path.

Wave8 used sequence-owner `DMIMO_V` CTAs and measured `3.00059 ms`
productionish combined.  The sidecar proved the better ownership is
output-owner all-R: one CTA owns `(B, H, Ptile)` and computes all four `R`
rows for that tile.  Sidecar productionish `DMIMO_V` slice: `0.53634 ms`,
projected wave7+`DMIMO_V`: `2.45093 ms`.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_kernel.cu`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_chunk_owner_cuda.py`
- `scripts/modal_mamba3_rr_diag_wave8_chunk_owner_cuda.py`
- `docs/status/mamba3_rr_diag_cuda_integrate_wave9_2026_04_30.md`

CUDA changes:

- Added `stage2_qk_dmimo_v_output_owner_rvec_kernel`, adapted from the sidecar
  all-R output-owner implementation.
- Preserved the old combined sequence-owner path as
  `stage2_rr_diag_qk_dv_dmimo_v_sequence_owner*` for direct comparison.
- Replaced the main `stage2_rr_diag_qk_dv_dmimo_v_owner*` combined path with
  all-R output-owner `DMIMO_V` CTAs.
- Renamed the JIT extension to `rr_diag_cuda_ext_wave9` to avoid stale cached
  wave4/wave8 shared objects missing the new symbols.

The combined launch is still one CUDA kernel: first `B*H*nchunks` CTAs run the
wave7 chunk-warp diagonal plus qk/dV body, then `B*H*ceil(P/32)` CTAs run the
all-R output-owner `DMIMO_V` branch.

## Local Checks

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_chunk_owner_cuda.py \
  scripts/modal_mamba3_rr_diag_wave8_chunk_owner_cuda.py

python upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_chunk_owner_cuda.py \
  --shape smoke --device cpu --iters 1 --warmup 0

git diff --check
```

All passed.

## H200 Run

```text
timeout 1200s modal run --timestamps \
  scripts/modal_mamba3_rr_diag_wave8_chunk_owner_cuda.py \
  --shape-csv smoke,productionish \
  --warmup 3 \
  --iters 10
```

Device/runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- Modal app completed and stopped normally.

## Correctness

Productionish max absolute diffs:

| output | wave9 check | max abs diff |
| --- | --- | ---: |
| `DGAMMA_DIAG` | combined vs wave5 CUDA post reference | `7.105e-15` |
| `DK` | combined vs wave5 CUDA post reference | `9.095e-13` |
| `DQ` | combined vs wave5 CUDA post reference | `4.547e-13` |
| `DV` | combined vs torch qk/dV reference | `1.455e-11` |
| `DMIMO_V` | combined vs torch qk/DMIMO_V reference | `1.066e-13` |
| `DMIMO_V` | old sequence combined vs new all-R combined | `1.315e-13` |

Smoke passed the same checks:

- `DGAMMA_DIAG`: `1.776e-15`
- `DK`: `2.842e-14`
- `DQ`: `5.684e-14`
- `DV`: `2.274e-13`
- `DMIMO_V`: `2.665e-15`

## Performance

Productionish shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

| path | mean ms | notes |
| --- | ---: | --- |
| wave7 diag + qk/dV combined | `1.92187` | refreshed in wave9 harness |
| old wave8 sequence qk/`DMIMO_V` only | `2.52359` | retained comparison hook |
| new wave9 output-owner all-R qk/`DMIMO_V` only | `0.53122` | sidecar ownership, integrated extension |
| old wave8 sequence combined | `2.99021` | was `3.00059 ms` in wave8 doc |
| new wave9 output-owner all-R combined | `2.48042` | main combined path |
| TileLang `stage2_bf1_bb0` `bwd_bwd` | `3.70674` | comparison baseline |

Productionish read:

- all-R `DMIMO_V` slice is `4.75x` faster than the sequence-owner slice;
- new combined path is `0.50979 ms` faster than old wave8 sequence combined;
- new combined path is `0.55855 ms` over refreshed wave7 combined;
- new combined path is `66.9%` of TileLang `bwd_bwd`, leaving `1.22632 ms`
  of margin;
- actual combined `2.48042 ms` is close to the sidecar projection
  `2.45093 ms` and recovers the margin that wave8 lost.

Smoke timing is underfilled/noisy, but still directionally matched:

| path | mean ms |
| --- | ---: |
| wave7 diag + qk/dV combined | `0.02426` |
| old wave8 sequence combined | `0.04355` |
| new wave9 output-owner all-R combined | `0.02454` |

## Resource Metadata

H200 productionish metadata:

| kernel | regs/thread | static smem | active blocks/SM | occupancy |
| --- | ---: | ---: | ---: | ---: |
| wave8 qk/`DMIMO_V` sequence owner | 48 | 0 B | 10 | 62.5% |
| wave9 qk/`DMIMO_V` all-R output owner | 40 | 2048 B | 12 | 75.0% |
| old wave8 sequence combined | 80 | 0 B | 6 | 37.5% |
| new wave9 all-R combined | 80 | 2048 B | 6 | 37.5% |

## Read

The main CUDA path regained enough margin for this slice.  The all-R
output-owner branch brings combined productionish time from about `3.0 ms` back
to `2.48 ms`, below both the old wave5 diagonal-only envelope and the current
TileLang `bwd_bwd` baseline.

The remaining caveat is scope: this still covers the same-time qk-dot
`DMIMO_V` contribution.  State/LKQ/D and non-diagonal `DK/DQ` remain outside
this prototype and need their own ownership plan.
