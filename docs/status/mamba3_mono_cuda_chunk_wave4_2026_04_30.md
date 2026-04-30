# Mamba3 Monolithic CUDA Chunk Wave 4 - 2026-04-30

Branch: `worker/mamba3-mono-cuda-chunk`

## Goal

Make one last non-incremental WMMA schedule break for the CUDA fallback path.

Baseline context:

- Wave 1 scalar monolithic state/LKQ/D slice: `89.02105560302735 ms` on H200
  productionish.
- Wave 2 WMMA chunk-owner slice: `8.919168281555176 ms`.
- Wave 3 WMMA + shared-memory reuse + triangular tile-k pruning:
  `8.467136001586914 ms`.
- TileLang `stage2_bf1_bb0` full `bwd_bwd`: `3.70674 ms`.
- Fast covered subset projection gate: this slice would need to land near
  `1.2 ms`, not near `8 ms`, to combine into a competitive path.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave4.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave4_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave4_kernel.cu`
- `scripts/modal_mamba3_mono_cuda_chunk_wave4.py`
- `docs/status/mamba3_mono_cuda_chunk_wave4_2026_04_30.md`

Schedule change:

- Split the P dimension into fixed P64 panels.
- Productionish `P=128` now launches two CTAs per logical `(B,H,chunk)`, each
  owning `(B,H,chunk,P64-panel)`.
- Each panel CTA writes its own `DV` and per-chunk `DMIMO_V` P slice.
- `DSSDA` is accumulated across panels in-kernel.  P64 stores directly; P128
  zeros `DSSDA` before launch and uses `atomicAdd` for panel partials.
- Kept the Wave 3 triangular-pruned `masked(LKQ) @ dPhi` WMMA schedule and
  dead `Q^T` shared-memory reuse.

## Resource Delta

Productionish shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

| metric | Wave 3 | Wave 4 P64 panels |
| --- | ---: | ---: |
| owner | `(B,H,chunk)` | `(B,H,chunk,P64-panel)` |
| CTAs | `32768` | `65536` |
| registers/thread | `72` | `40` |
| dynamic smem | `98308 B` | `65540 B` |
| active blocks/SM | `2` | `3` |
| theoretical occupancy | `25.0%` | `37.5%` |
| LKQ WMMA ops/logical chunk | `64` | `128` |
| triangular LKQ apply ops/logical chunk | `80` | `80` |
| state WMMA ops/logical chunk | `128` | `128` |
| dki WMMA ops/logical chunk | `128` | `128` |

The panel split delivered the intended lower-smem/higher-occupancy shape, but
it duplicates Q/K staging and `LKQ = K @ Q.T` for every P panel and adds a
`DSSDA` zero+atomic envelope for P128.

## Local Checks

Commands:

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave4.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave4_extension.py \
  scripts/modal_mamba3_mono_cuda_chunk_wave4.py

git diff --check
```

GB10 compile/correctness checks:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave4_p64 \
  RR_DIAG_CUDA_VERBOSE_BUILD=1 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave4.py \
  --shape smoke --device cuda --iters 1 --warmup 0

env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave4_p64 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave4.py \
  --B 1 --S 256 --H 4 --G 1 --N 64 --P 128 --R 4 --chunk 16 \
  --device cuda --iters 2 --warmup 1
```

GB10 P64 smoke correctness vs bf16-staged torch reference:

| output | max diff |
| --- | ---: |
| `DV` | `2.9802322387695312e-08` |
| per-chunk `DMIMO_V` | `7.275957614183426e-11` |
| `DSSDA` | `6.661338147750939e-16` |

GB10 P128 two-panel correctness vs bf16-staged torch reference:

| output | max diff |
| --- | ---: |
| `DV` | `2.384185791015625e-07` |
| per-chunk `DMIMO_V` | `8.731149137020111e-11` |
| `DSSDA` | `1.3322676295501878e-15` |

## H200 Smoke And Productionish

Command:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave4.py::run_remote \
  --shape-csv smoke,productionish \
  --warmup 2 \
  --iters 5
```

Runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- Modal app: `ap-fzA45qZ1f93CwPnCKV7Lm0`
- Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

Smoke correctness:

| comparison | `DV` | per-chunk `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `1.862645149230957e-09` | `7.275957614183426e-11` | `6.661338147750939e-16` |
| vs Wave 1 fp32 ref | `4.76837158203125e-07` | `3.893546818289906e-07` | `9.892642260922457e-12` |

Smoke timing:

| path | mean ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| Wave 4 P64-panel WMMA chunk-owner slice | `0.037100800126791` | `0.03532800078392029` | `0.03683200106024742` | `0.03932800143957138` |

Productionish correctness:

| comparison | `DV` | per-chunk `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `4.76837158203125e-07` | `3.069544618483633e-10` | `2.6645352591003757e-15` |
| vs Wave 1 fp32 ref | `9.5367431640625e-07` | `6.133341230452061e-07` | `3.33120198092729e-11` |

Productionish timing:

| path | mean ms | std ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wave 4 P64-panel WMMA chunk-owner slice | `8.784351921081543` | `0.005556777687018592` | `8.7774076461792` | `8.782303810119629` | `8.793855667114258` |

Samples:
`[8.793855667114258, 8.7774076461792, 8.78649616241455, 8.782303810119629, 8.781696319580078]`

Comparison:

| comparison | value |
| --- | ---: |
| speedup vs Wave 1 scalar monolithic slice | `10.134049318924252x` |
| speedup vs Wave 2 WMMA slice | `1.0153473314462833x` |
| delta vs Wave 2 WMMA slice | `-0.13481636047363388 ms` |
| speedup vs Wave 3 WMMA triangular slice | `0.9638885233259675x` |
| delta vs Wave 3 WMMA triangular slice | `+0.3172159194946289 ms` |
| Wave 4 slice / TileLang full `bwd_bwd` | `2.369832230229674x` |
| projected Wave10 two-launch + Wave 4 slice | `10.881081921081542 ms` |
| projected combined / TileLang full `bwd_bwd` | `2.9354856075909135x` |
| projected margin vs TileLang | `-7.174341921081542 ms` |

## Read

Wave 4 was the intended major break for the WMMA fallback: it materially changed
CTA ownership and resource shape rather than tuning scalar loops.  It improved
the resource envelope (`98 KB -> 65 KB` smem, `2 -> 3` active blocks/SM,
`72 -> 40` registers/thread on H200), but not the runtime.

The reason is structural.  Splitting P removes per-CTA smem pressure, but this
kernel is not smem-bound enough for that to pay for duplicated chunk-local work.
Each P64 panel redoes Q/K staging and `LKQ = K @ Q.T`; production P128 doubles
that LKQ work from `64` to `128` WMMA ops per logical chunk.  The `DSSDA`
partial accumulation also adds a memset plus atomics.  The higher occupancy does
not compensate, so Wave 4 regresses by `0.317 ms` vs Wave 3.

Recommendation: kill the CUDA WMMA fallback as a performance path for this
stage.  It is useful negative evidence and a correctness microbench, but after
WMMA tensorization, shared-memory reuse, triangular pruning, and P-panel
ownership, the slice still runs `8.78 ms` while the full TileLang `bwd_bwd` is
`3.71 ms`.  Another warp-level WMMA schedule tweak is not plausibly going to
recover the missing multi-millisecond budget; a real replacement would need a
different implementation class such as a CuTe/CUTLASS WGMMA mainloop/epilogue
or direct TileLang work, not this fallback.
